// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend stream lifetime, error conversion, and session-key contracts.
//! Shared table mechanics are tested in `dynamo_kv_router`.

use std::{sync::Arc, time::Duration};

use dynamo_runtime::{
    engine::AsyncEngineContext,
    error::ErrorType,
    pipeline::{Context, ResponseStream, context::Controller},
    protocols::maybe_error::MaybeError,
};
use futures::{StreamExt, stream};

use super::SessionAffinityMode::Hard;
use super::{
    AffinityTarget, Hold, LlmResponse, affinity_error, affinity_id, explicit_target,
    replica_sync::ReplicaSyncRuntime, subagent_group_affinity_id, to_table, tracked_stream,
};
use crate::{
    preprocessor::PreprocessedRequest,
    protocols::common::{
        extensions::SESSION_AFFINITY_CONTEXT_KEY, llm_backend::LLMEngineOutput,
        preprocessor::RoutingHints, timing::RequestPhase,
    },
    types::Annotated,
};
use dynamo_kv_router::services::selection::affinity::{
    AffinityError, MAX_SESSION_AFFINITY_ID_BYTES, SessionAffinity, SessionAffinityConfig,
};

fn affinity() -> SessionAffinity {
    SessionAffinity::with_config(SessionAffinityConfig::new(Duration::from_secs(10))).unwrap()
}

fn target(worker_id: u64, dp_rank: Option<u32>) -> AffinityTarget {
    AffinityTarget { worker_id, dp_rank }
}

fn table_target(
    worker_id: u64,
    dp_rank: Option<u32>,
) -> dynamo_kv_router::services::selection::affinity::AffinityTarget {
    to_table(target(worker_id, dp_rank))
}

fn response_stream(items: usize) -> dynamo_runtime::pipeline::ManyOut<LlmResponse> {
    let items = (0..items).map(|_| Annotated::from_data(LLMEngineOutput::default()));
    ResponseStream::new(
        Box::pin(stream::iter(items)),
        Arc::new(Controller::default()),
    )
}

fn error_response_stream() -> dynamo_runtime::pipeline::ManyOut<LlmResponse> {
    ResponseStream::new(
        Box::pin(stream::iter([Annotated::from_error("backend failed")])),
        Arc::new(Controller::default()),
    )
}

fn cancelled_response_stream() -> dynamo_runtime::pipeline::ManyOut<LlmResponse> {
    let controller = Controller::new("cancelled-stream".to_string());
    controller.stop();
    ResponseStream::new(Box::pin(stream::empty()), Arc::new(controller))
}

async fn assert_binding_expires_after_refreshed_ttl(affinity: &SessionAffinity) {
    tokio::time::advance(Duration::from_secs(9)).await;
    assert_eq!(
        affinity.query_target("s", None).unwrap(),
        Some(table_target(7, Some(0)))
    );
    tokio::time::advance(Duration::from_secs(2)).await;
    assert_eq!(affinity.query_target("s", None).unwrap(), None);
}

fn request_with_routing(routing: RoutingHints) -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("test".to_string())
        .token_ids(vec![1])
        .stop_conditions(Default::default())
        .sampling_options(Default::default())
        .output_options(Default::default())
        .routing(Some(routing))
        .build()
        .unwrap()
}

#[test]
fn session_affinity_explicit_targets_are_phase_local_and_preserve_rank_zero() {
    let request = request_with_routing(RoutingHints {
        backend_instance_id: Some(1),
        prefill_worker_id: Some(2),
        decode_worker_id: Some(3),
        dp_rank: Some(0),
        prefill_dp_rank: Some(4),
        ..Default::default()
    });

    assert_eq!(
        explicit_target(&request, RequestPhase::Prefill).unwrap(),
        Some(target(2, Some(4)))
    );
    assert_eq!(
        explicit_target(&request, RequestPhase::Decode).unwrap(),
        Some(target(3, Some(0)))
    );
    assert_eq!(
        explicit_target(&request, RequestPhase::Aggregated).unwrap(),
        Some(target(3, Some(0)))
    );

    let decode_only = request_with_routing(RoutingHints {
        decode_worker_id: Some(3),
        ..Default::default()
    });
    assert_eq!(
        explicit_target(&decode_only, RequestPhase::Aggregated).unwrap(),
        Some(target(3, None))
    );

    let rank_without_worker = request_with_routing(RoutingHints {
        dp_rank: Some(0),
        ..Default::default()
    });
    assert!(explicit_target(&rank_without_worker, RequestPhase::Decode).is_err());
}

#[test]
fn session_affinity_context_type_errors_are_preserved() {
    let mut request = Context::new(request_with_routing(RoutingHints::default()));
    request.insert(SESSION_AFFINITY_CONTEXT_KEY, "wrong type".to_string());

    let error = affinity_id(&request).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("invalid session affinity context")
    );
}

#[tokio::test(start_paused = true)]
async fn session_affinity_stream_drop_refreshes_idle_ttl() {
    let affinity = affinity();
    let Hold::Initialize(initializer) = affinity.acquire("s", None).await.unwrap() else {
        panic!("first request must initialize");
    };
    let lease = initializer.commit(table_target(7, Some(0))).unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    let mut stream = tracked_stream(lease, response_stream(1));
    assert!(stream.next().await.is_some());
    drop(stream);

    assert_binding_expires_after_refreshed_ttl(&affinity).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_empty_stream_refreshes_idle_ttl() {
    let affinity = affinity();
    let Hold::Initialize(initializer) = affinity.acquire("s", None).await.unwrap() else {
        panic!("first request must initialize");
    };
    let lease = initializer.commit(table_target(7, Some(0))).unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    let mut stream = tracked_stream(lease, response_stream(0));
    assert!(stream.next().await.is_none());

    assert_binding_expires_after_refreshed_ttl(&affinity).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_cancelled_stream_refreshes_idle_ttl() {
    let affinity = affinity();
    let Hold::Initialize(initializer) = affinity.acquire("s", None).await.unwrap() else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(table_target(7, Some(0))).unwrap());

    tokio::time::advance(Duration::from_secs(9)).await;
    let Hold::Bound {
        target: bound_target,
        lease,
    } = affinity.acquire("s", None).await.unwrap()
    else {
        panic!("continuation must acquire the existing binding");
    };
    assert_eq!(bound_target, table_target(7, Some(0)));
    let mut stream = tracked_stream(lease, cancelled_response_stream());
    assert!(stream.next().await.is_none());

    assert_binding_expires_after_refreshed_ttl(&affinity).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_committed_binding_survives_cancelled_stream_until_ttl() {
    let affinity = affinity();
    let operation = affinity.acquire("s", None).await.unwrap();
    let mut stream = tracked_stream(
        affinity
            .commit(operation, to_table(target(7, Some(0))))
            .unwrap(),
        cancelled_response_stream(),
    );
    tokio::time::advance(Duration::from_secs(9)).await;
    assert!(stream.next().await.is_none());

    assert_binding_expires_after_refreshed_ttl(&affinity).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_error_stream_refreshes_idle_ttl() {
    let affinity = affinity();
    let Hold::Initialize(initializer) = affinity.acquire("s", None).await.unwrap() else {
        panic!("first request must initialize");
    };
    let lease = initializer.commit(table_target(7, Some(0))).unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    let mut stream = tracked_stream(lease, error_response_stream());
    assert!(stream.next().await.unwrap().is_err());
    assert!(stream.next().await.is_none());

    assert_binding_expires_after_refreshed_ttl(&affinity).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_stream_eof_refreshes_idle_ttl() {
    let affinity = affinity();
    let Hold::Initialize(initializer) = affinity.acquire("s", None).await.unwrap() else {
        panic!("first request must initialize");
    };
    let lease = initializer.commit(table_target(7, Some(0))).unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    let mut stream = tracked_stream(lease, response_stream(1));
    while stream.next().await.is_some() {}

    assert_binding_expires_after_refreshed_ttl(&affinity).await;
}

#[test]
fn session_affinity_rejects_invalid_ttl_before_starting_reaper() {
    for ttl in [
        Duration::ZERO,
        Duration::from_secs(super::MAX_SESSION_AFFINITY_TTL_SECS + 1),
    ] {
        let Err(error) =
            SessionAffinity::with_config(SessionAffinityConfig::new(ttl).with_mode(Hard))
                .map_err(affinity_error)
        else {
            panic!("invalid TTL must fail host construction");
        };
        assert!(dynamo_runtime::error::match_error_chain(
            error.as_ref(),
            &[dynamo_runtime::error::ErrorType::InvalidArgument],
            &[]
        ));
        assert!(error.to_string().contains("session affinity TTL"));
    }
}

#[tokio::test(start_paused = true)]
async fn session_affinity_publishes_after_dispatch_and_lease_completion() {
    let affinity = affinity();
    let (replica, mut updates) = ReplicaSyncRuntime::for_test(4);
    assert!(affinity.enable_replication(99, replica.clone()));
    let selected_target = target(7, Some(0));
    let operation = affinity.acquire("s", None).await.unwrap();
    let stream = tracked_stream(
        affinity
            .commit(operation, to_table(selected_target))
            .unwrap(),
        response_stream(1),
    );

    let after_dispatch = updates.recv().await.unwrap();
    assert_eq!(after_dispatch.session_id, "s");
    assert_eq!(after_dispatch.worker_id, selected_target.worker_id);
    assert_eq!(after_dispatch.dp_rank, selected_target.dp_rank);
    assert_eq!(after_dispatch.writer_id, 99);

    drop(stream);
    let after_completion = updates.recv().await.unwrap();
    assert_eq!(after_completion, after_dispatch);
    assert!(updates.try_recv().is_err());
}

#[test]
fn a_subagent_group_id_is_namespaced_and_fixed_size() {
    let a = subagent_group_affinity_id("parent-1");
    let b = subagent_group_affinity_id("parent-1");
    let c = subagent_group_affinity_id("parent-2");
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert!(
        a.starts_with("\u{1}sg:"),
        "the key must be un-claimable via a header value"
    );
    let long = subagent_group_affinity_id(&"p".repeat(10_000));
    assert_eq!(long.len(), a.len());
    assert!(long.len() <= MAX_SESSION_AFFINITY_ID_BYTES);
}

#[test]
fn affinity_errors_keep_frontend_error_types() {
    for (error, expected) in [
        (
            AffinityError::InvalidArgument("bad session".into()),
            ErrorType::InvalidArgument,
        ),
        (AffinityError::Cancelled, ErrorType::Cancelled),
    ] {
        let error = affinity_error(error);
        assert!(dynamo_runtime::error::match_error_chain(
            error.as_ref(),
            &[expected],
            &[]
        ));
    }
}
