// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What only the frontend knows about session affinity: where the session key
//! comes from (request context and routing hints), the pipeline response
//! stream that owns a lease, the runtime event plane that replicates bindings
//! between frontends, and how affinity failures map to pipeline errors. The
//! session layer itself is the shared [`AffinityResolver`].

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use dynamo_kv_router::services::selection::affinity::{
    AffinityError, AffinityLease, AffinityResolver, SessionAffinity, SessionAffinityConfig,
    TargetLiveness,
};
use dynamo_runtime::{
    component::Client,
    error::{DynamoError, ErrorType},
    pipeline::{Error, ManyOut, ResponseStream},
};
use futures::Stream;

use super::{LlmResponse, SessionAffinityMode, replica_sync::ReplicaSyncRuntime};
use crate::{
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

struct Inner {
    resolver: Arc<AffinityResolver>,
    replica: tokio::sync::OnceCell<ReplicaSyncRuntime>,
}

/// Session affinity as a frontend routing host holds it: the resolver (the
/// embedded partition's for KV routing, the host's own table for the builtin
/// modes) and the event-plane replication attached to its table. Cheap to
/// clone; the replication runtime lives as long as the last clone.
#[derive(Clone)]
pub struct HostAffinity {
    inner: Arc<Inner>,
}

impl HostAffinity {
    /// Wrap a resolver another component owns (an embedded selection partition).
    pub(crate) fn from_resolver(resolver: Arc<AffinityResolver>) -> Self {
        Self {
            inner: Arc::new(Inner {
                resolver,
                replica: tokio::sync::OnceCell::new(),
            }),
        }
    }

    /// A resolver over this host's own table, for hosts without a selection
    /// partition (builtin routing modes). `liveness` is the host's view of
    /// which bound workers can still take requests.
    pub fn standalone(
        ttl: Duration,
        mode: SessionAffinityMode,
        liveness: Arc<dyn TargetLiveness>,
    ) -> Result<Self, Error> {
        let table = SessionAffinity::with_config(SessionAffinityConfig::new(ttl).with_mode(mode))
            .map_err(affinity_error)?;
        Ok(Self::from_resolver(Arc::new(AffinityResolver::new(
            table, liveness,
        ))))
    }

    /// [`Self::standalone`] with runtime replication over `client`'s event
    /// plane; `None` when session affinity is disabled.
    pub(crate) async fn standalone_with_replica_sync(
        ttl: Option<Duration>,
        mode: SessionAffinityMode,
        liveness: Arc<dyn TargetLiveness>,
        client: Client,
    ) -> Result<Option<Self>, Error> {
        let Some(ttl) = ttl else {
            return Ok(None);
        };
        let affinity = Self::standalone(ttl, mode, liveness)?;
        affinity.enable_replica_sync(client).await?;
        Ok(Some(affinity))
    }

    pub(crate) async fn enable_replica_sync(&self, client: Client) -> Result<(), Error> {
        self.inner
            .replica
            .get_or_try_init(|| async {
                let table = self.inner.resolver.table();
                let (replica, router_id) =
                    ReplicaSyncRuntime::start(client, table.downgrade()).await?;
                if !table.enable_replication(router_id, replica.sink()) {
                    return Err(anyhow::anyhow!(
                        "session affinity table already has a replica sink installed"
                    ));
                }
                Ok(replica)
            })
            .await?;
        Ok(())
    }

    pub(crate) fn resolver(&self) -> &AffinityResolver {
        &self.inner.resolver
    }

    pub(crate) fn mode(&self) -> SessionAffinityMode {
        self.inner.resolver.table().mode()
    }
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
