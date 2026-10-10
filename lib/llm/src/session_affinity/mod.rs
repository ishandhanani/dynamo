// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The frontend's share of session affinity. The shared table and lifecycle
//! live in `dynamo_kv_router::services::selection::affinity`; this module
//! holds the frontend's session-key sources, liveness source, lease owner
//! (the response stream), event-plane replication, and error mapping.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use dynamo_kv_router::protocols::WorkerConfigLike;
use dynamo_kv_router::services::selection::affinity::{AffinityError, AffinityLease};
use dynamo_runtime::{
    component::Client,
    error::{DynamoError, ErrorType},
    pipeline::{Error, ManyOut, ResponseStream},
};
use futures::Stream;

use crate::{
    discovery::RuntimeConfigWatch,
    preprocessor::PreprocessedRequest,
    protocols::common::{
        extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
        timing::RequestPhase,
    },
};

/// The pipeline's routing target; the table's `AffinityTarget` has the same
/// shape and the two convert at this boundary.
pub type AffinityTarget = dynamo_runtime::pipeline::RouteTarget;

type TableTarget = dynamo_kv_router::services::selection::affinity::AffinityTarget;

pub(crate) fn to_table(target: AffinityTarget) -> TableTarget {
    TableTarget::new(target.worker_id, target.dp_rank)
}

pub(crate) fn from_table(target: TableTarget) -> AffinityTarget {
    AffinityTarget::new(target.worker_id, target.dp_rank)
}

mod replica_sync;

pub(crate) use dynamo_kv_router::services::selection::affinity::Hold;
pub use dynamo_kv_router::services::selection::affinity::{
    MAX_SESSION_AFFINITY_TTL_SECS, MIN_SESSION_AFFINITY_TTL_SECS, SessionAffinity,
    SessionAffinityConfig, SessionAffinityMode,
};
pub(crate) use replica_sync::enable_replica_sync;

/// Builtin routing modes own a table without a selection partition.
pub(crate) async fn standalone_with_replica_sync(
    ttl: Option<Duration>,
    mode: SessionAffinityMode,
    client: Client,
) -> Result<Option<SessionAffinity>, Error> {
    let Some(ttl) = ttl else {
        return Ok(None);
    };
    let table = SessionAffinity::with_config(SessionAffinityConfig::new(ttl).with_mode(mode))
        .map_err(affinity_error)?;
    enable_replica_sync(
        &table,
        crate::kv_router::embedded::embedded_partition_key(None),
        client,
        None,
    )
    .await?;
    Ok(Some(table))
}

/// Membership and rank validity, independent of transient health or overload.
pub(crate) fn target_is_live(
    client: &Client,
    workers: Option<&RuntimeConfigWatch>,
    target: TableTarget,
) -> bool {
    if !client.is_instance_discovered(target.worker_id) {
        return false;
    }
    let Some(workers) = workers else {
        return true;
    };
    let workers = workers.borrow();
    let Some(config) = workers.get(&target.worker_id) else {
        return true;
    };
    let Some(dp_rank) = target.dp_rank else {
        return true;
    };
    let start = config.data_parallel_start_rank();
    let end = start.saturating_add(config.data_parallel_size());
    (start..end).contains(&dp_rank)
}

/// Keep `lease` for as long as `stream` runs: the binding is in use until the
/// response ends, however it ends.
pub(crate) fn tracked_stream(
    lease: AffinityLease,
    stream: ManyOut<LlmResponse>,
) -> ManyOut<LlmResponse> {
    let context = stream.context();
    ResponseStream::new(
        Box::pin(AffinityTrackedStream {
            stream,
            lease: Some(lease),
        }),
        context,
    )
}

struct AffinityTrackedStream {
    stream: ManyOut<LlmResponse>,
    lease: Option<AffinityLease>,
}

impl Stream for AffinityTrackedStream {
    type Item = LlmResponse;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.stream).poll_next(cx) {
            Poll::Ready(None) => {
                drop(self.lease.take());
                Poll::Ready(None)
            }
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            poll => poll,
        }
    }
}

pub fn affinity_id(
    request: &dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
) -> Result<Option<Arc<SessionAffinityId>>, Error> {
    request
        .get_optional::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
        .map_err(|message| invalid_argument(format!("invalid session affinity context: {message}")))
}

pub fn explicit_target(
    request: &PreprocessedRequest,
    phase: RequestPhase,
) -> Result<Option<AffinityTarget>, Error> {
    let Some(routing) = request.routing.as_ref() else {
        return Ok(None);
    };
    let (worker_id, dp_rank) = match phase {
        RequestPhase::Prefill => (
            routing.prefill_worker_id.or(routing.backend_instance_id),
            routing.prefill_dp_rank.or(routing.dp_rank),
        ),
        RequestPhase::Decode | RequestPhase::Aggregated => (
            routing.decode_worker_id.or(routing.backend_instance_id),
            routing.dp_rank,
        ),
    };
    if worker_id.is_none() && dp_rank.is_some() {
        return Err(invalid_argument(
            "DP rank requires an explicit worker for session affinity",
        ));
    }
    Ok(worker_id.map(|worker_id| AffinityTarget { worker_id, dp_rank }))
}

pub(crate) fn affinity_error(error: AffinityError) -> Error {
    match error {
        AffinityError::InvalidArgument(message) => invalid_argument(message),
        AffinityError::ResourceExhausted(message) => DynamoError::builder()
            .error_type(ErrorType::ResourceExhausted)
            .message(message)
            .build()
            .into(),
        cancelled @ AffinityError::Cancelled => DynamoError::builder()
            .error_type(ErrorType::Cancelled)
            .message(cancelled.to_string())
            .build()
            .into(),
        AffinityError::Dropped => anyhow::anyhow!("session affinity table dropped"),
    }
}

/// Session and worker identifiers are private diagnostics and must not be copied into client responses.
pub(crate) fn invalid_argument(message: impl Into<String>) -> Error {
    DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(message.into())
        .build()
        .into()
}

pub type LlmResponse =
    crate::types::Annotated<crate::protocols::common::llm_backend::LLMEngineOutput>;

/// The binding key for the subagents of one parent session.
///
/// The `\u{1}` prefix cannot appear in an HTTP header value, so no client can claim this key as
/// its own session id; the constant length keeps a long parent id from overflowing the limit.
pub(crate) fn subagent_group_affinity_id(parent_session_id: &str) -> String {
    let digest = blake3::hash(parent_session_id.as_bytes());
    format!("\u{1}sg:{}", digest.to_hex())
}

#[cfg(test)]
mod tests;
