// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Affinity lifecycle contract tests exercise the shared table and frontend stream adapter.

use std::{sync::Arc, time::Duration};

use dynamo_runtime::{
    engine::AsyncEngineContext,
    error::ErrorType,
    pipeline::{Context, ResponseStream, context::Controller},
    protocols::maybe_error::MaybeError,
};
use futures::{StreamExt, stream};

use super::SessionAffinityMode::{self, Hard};
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
    AffinityBindingEvent, AffinityVersion, MAX_SESSION_AFFINITY_ID_BYTES, ReplicaApplyOutcome,
    SessionAffinity, SessionAffinityConfig,
};

const TTL: Duration = Duration::from_secs(10);

fn session_id() -> String {
    "session-1".into()
}

fn coordinator() -> SessionAffinity {
    SessionAffinity::with_config(SessionAffinityConfig::new(Duration::from_secs(10))).unwrap()
}

async fn bind(table: &SessionAffinity, target: dynamo_kv_router::protocols::WorkerAffinityTarget) {
    let hold = table.acquire(&session_id(), None).await.unwrap();
    drop(table.commit(hold, target).unwrap());
}

fn enable_replica(
    table: &SessionAffinity,
    writer_id: u64,
    capacity: usize,
) -> tokio::sync::mpsc::Receiver<AffinityBindingEvent> {
    let (sink, rx) = ReplicaSyncRuntime::for_test(capacity);
    assert!(table.enable_replication(writer_id, sink));
    rx
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

async fn assert_binding_expires_after_refreshed_ttl(coordinator: &SessionAffinity) {
    tokio::time::advance(Duration::from_secs(9)).await;
    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(table_target(7, Some(0)))
    );
    tokio::time::advance(Duration::from_secs(2)).await;
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
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
async fn session_affinity_initialization_is_atomic() {
    let coordinator = coordinator();
    let first = coordinator.acquire(&session_id(), None).await.unwrap();
    let Hold::Initialize(first) = first else {
        panic!("first request must initialize");
    };

    let waiter_coordinator = coordinator.clone();
    let waiter = tokio::spawn(async move { waiter_coordinator.acquire(&session_id(), None).await });
    coordinator.wait_for_initializing_waiter().await;
    assert!(!waiter.is_finished());

    let first_lease = first.commit(table_target(7, Some(0))).unwrap();
    let second = waiter.await.unwrap().unwrap();
    let Hold::Bound {
        target: second_target,
        lease: second_lease,
    } = second
    else {
        panic!("waiter must acquire the committed binding");
    };
    assert_eq!(second_target, table_target(7, Some(0)));
    drop(first_lease);
    drop(second_lease);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_initializer_cancellation_wakes_waiter() {
    let coordinator = coordinator();
    let first = coordinator.acquire(&session_id(), None).await.unwrap();
    let Hold::Initialize(first) = first else {
        panic!("first request must initialize");
    };

    let waiter_coordinator = coordinator.clone();
    let waiter = tokio::spawn(async move { waiter_coordinator.acquire(&session_id(), None).await });
    coordinator.wait_for_initializing_waiter().await;
    drop(first);

    let next = waiter.await.unwrap().unwrap();
    assert!(matches!(&next, Hold::Initialize(_)));
    drop(next);
    assert_eq!(coordinator.entry_count(), 0);
    assert!(matches!(
        coordinator.acquire(&session_id(), None).await.unwrap(),
        Hold::Initialize(_)
    ));
}

#[tokio::test(start_paused = true)]
async fn session_affinity_wait_stops_when_request_is_cancelled() {
    let coordinator = coordinator();
    let first = coordinator.acquire(&session_id(), None).await.unwrap();
    let Hold::Initialize(first) = first else {
        panic!("first request must initialize");
    };

    let context = Arc::new(Controller::default());
    let waiter_context = context.clone();
    let waiter_coordinator = coordinator.clone();
    let waiter = tokio::spawn(async move {
        waiter_coordinator
            .resolve(&session_id(), None, |_| true, waiter_context.stopped())
            .await
            .map_err(affinity_error)
    });
    coordinator.wait_for_initializing_waiter().await;
    context.stop();

    let Err(error) = waiter.await.unwrap() else {
        panic!("cancelled waiter must return an error");
    };
    assert!(dynamo_runtime::error::match_error_chain(
        error.as_ref(),
        &[ErrorType::Cancelled],
        &[]
    ));
    drop(first);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_validates_worker_and_rank_contract() {
    let coordinator = coordinator();
    let Hold::Initialize(initializer) = coordinator
        .acquire(&session_id(), Some(table_target(7, None)))
        .await
        .unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(table_target(7, None)).unwrap());

    assert!(
        coordinator
            .acquire(&session_id(), Some(table_target(8, None)))
            .await
            .is_err()
    );
    assert!(
        coordinator
            .acquire(&session_id(), Some(table_target(7, Some(0))))
            .await
            .is_err()
    );
    assert!(
        coordinator
            .acquire(&session_id(), Some(table_target(7, None)))
            .await
            .is_ok()
    );
}

#[tokio::test(start_paused = true)]
async fn session_affinity_failed_bound_operation_invalidates_binding() {
    let coordinator = coordinator();
    let Hold::Initialize(initializer) = coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(table_target(7, Some(0))).unwrap());

    let operation = coordinator.acquire(&session_id(), None).await.unwrap();
    assert_eq!(operation.target(), Some(table_target(7, Some(0))));
    operation.invalidate();

    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
    assert_eq!(coordinator.entry_count(), 0);
    assert!(matches!(
        coordinator.acquire(&session_id(), None).await.unwrap(),
        Hold::Initialize(_)
    ));
}

#[tokio::test(start_paused = true)]
async fn session_affinity_stream_drop_refreshes_idle_ttl() {
    let coordinator = coordinator();
    let Hold::Initialize(initializer) = coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    let lease = initializer.commit(table_target(7, Some(0))).unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    let mut stream = tracked_stream(lease, response_stream(1));
    assert!(stream.next().await.is_some());
    drop(stream);

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_empty_stream_refreshes_idle_ttl() {
    let coordinator = coordinator();
    let Hold::Initialize(initializer) = coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    let lease = initializer.commit(table_target(7, Some(0))).unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    let mut stream = tracked_stream(lease, response_stream(0));
    assert!(stream.next().await.is_none());

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_cancelled_stream_refreshes_idle_ttl() {
    let coordinator = coordinator();
    let Hold::Initialize(initializer) = coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(table_target(7, Some(0))).unwrap());

    tokio::time::advance(Duration::from_secs(9)).await;
    let Hold::Bound {
        target: bound_target,
        lease,
    } = coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("continuation must acquire the existing binding");
    };
    assert_eq!(bound_target, table_target(7, Some(0)));
    let mut stream = tracked_stream(lease, cancelled_response_stream());
    assert!(stream.next().await.is_none());

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_committed_binding_survives_cancelled_stream_until_ttl() {
    let coordinator = coordinator();
    let operation = coordinator.acquire(&session_id(), None).await.unwrap();
    let mut stream = tracked_stream(
        coordinator
            .commit(operation, to_table(target(7, Some(0))))
            .unwrap(),
        cancelled_response_stream(),
    );
    tokio::time::advance(Duration::from_secs(9)).await;
    assert!(stream.next().await.is_none());

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_error_stream_refreshes_idle_ttl() {
    let coordinator = coordinator();
    let Hold::Initialize(initializer) = coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    let lease = initializer.commit(table_target(7, Some(0))).unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    let mut stream = tracked_stream(lease, error_response_stream());
    assert!(stream.next().await.unwrap().is_err());
    assert!(stream.next().await.is_none());

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_stream_eof_refreshes_idle_ttl() {
    let coordinator = coordinator();
    let Hold::Initialize(initializer) = coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    let lease = initializer.commit(table_target(7, Some(0))).unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    let mut stream = tracked_stream(lease, response_stream(1));
    while stream.next().await.is_some() {}

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_bound_lease_drop_refreshes_idle_ttl() {
    let coordinator = coordinator();
    let Hold::Initialize(initializer) = coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(table_target(7, Some(0))).unwrap());

    tokio::time::advance(Duration::from_secs(9)).await;
    let Hold::Bound { lease, .. } = coordinator.acquire(&session_id(), None).await.unwrap() else {
        panic!("continuation must acquire the binding");
    };
    tokio::time::advance(Duration::from_secs(2)).await;
    tokio::task::yield_now().await;
    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(table_target(7, Some(0)))
    );
    drop(lease);

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_query_is_read_only() {
    let coordinator = coordinator();
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
    assert_eq!(coordinator.entry_count(), 0);

    let initializing = coordinator.acquire(&session_id(), None).await.unwrap();
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
    assert_eq!(coordinator.entry_count(), 1);
    drop(initializing);
    assert_eq!(coordinator.entry_count(), 0);

    let Hold::Initialize(initializer) = coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(table_target(7, Some(0))).unwrap());
    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(table_target(7, Some(0)))
    );
    coordinator.expire_for_test(&session_id());
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
    assert_eq!(coordinator.entry_count(), 1);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_reaper_removes_idle_entries_and_stops_on_drop() {
    let coordinator = coordinator();
    let cancellation = coordinator.cancellation_token();
    let Hold::Initialize(initializer) = coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(table_target(7, Some(0))).unwrap());

    coordinator.wait_for_reaper().await;
    tokio::time::advance(Duration::from_secs(10)).await;
    tokio::task::yield_now().await;
    assert_eq!(coordinator.entry_count(), 0);

    drop(coordinator);
    cancellation.cancelled().await;
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
async fn session_affinity_replica_ignores_initializing_sessions() {
    let coordinator = coordinator();
    let initializing = coordinator.acquire(&session_id(), None).await.unwrap();

    assert_eq!(
        coordinator.apply_replica_update(
            session_id().into(),
            table_target(7, Some(0)),
            AffinityVersion {
                sequence: 0,
                writer_id: 0
            }
        ),
        ReplicaApplyOutcome::IgnoredInitializing
    );
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
    drop(initializing);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_publishes_after_dispatch_and_lease_completion() {
    let coordinator = coordinator();
    let (replica, mut updates) = ReplicaSyncRuntime::for_test(4);
    assert!(coordinator.enable_replication(99, replica.clone()));
    let selected_target = target(7, Some(0));
    let operation = coordinator.acquire(&session_id(), None).await.unwrap();
    let stream = tracked_stream(
        coordinator
            .commit(operation, to_table(selected_target))
            .unwrap(),
        response_stream(1),
    );

    let after_dispatch = updates.recv().await.unwrap();
    assert_eq!(after_dispatch.session_id, session_id());
    assert_eq!(after_dispatch.worker_id, selected_target.worker_id);
    assert_eq!(after_dispatch.dp_rank, selected_target.dp_rank);
    assert_eq!(after_dispatch.writer_id, 99);

    drop(stream);
    let after_completion = updates.recv().await.unwrap();
    assert_eq!(after_completion, after_dispatch);
    assert!(updates.try_recv().is_err());
}

#[tokio::test(start_paused = true)]
async fn session_affinity_worker_only_binding_allows_ranked_dispatch_without_narrowing() {
    let coordinator = coordinator();
    let worker_binding = table_target(7, None);

    let initialization = coordinator.acquire(&session_id(), None).await.unwrap();
    drop(coordinator.commit(initialization, worker_binding).unwrap());

    let continuation = coordinator.acquire(&session_id(), None).await.unwrap();
    let lease = coordinator
        .commit(continuation, table_target(7, Some(3)))
        .expect("worker-only coordinator must allow the scheduler to select a DP rank");

    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(worker_binding)
    );
    drop(lease);
}

#[tokio::test(start_paused = true)]
async fn soft_worker_only_binding_stays_worker_scoped_after_ranked_dispatch() {
    let coordinator = SessionAffinity::with_config(
        SessionAffinityConfig::new(TTL).with_mode(SessionAffinityMode::Soft),
    )
    .unwrap();
    let worker_binding = table_target(7, None);
    bind(&coordinator, worker_binding).await;

    let continuation = coordinator.acquire(&session_id(), None).await.unwrap();
    let lease = coordinator
        .commit(continuation, table_target(7, Some(3)))
        .unwrap();
    drop(lease);

    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(worker_binding)
    );
}

#[tokio::test(start_paused = true)]
async fn soft_ranked_binding_is_not_widened_by_worker_only_dispatch() {
    let coordinator = SessionAffinity::with_config(
        SessionAffinityConfig::new(TTL).with_mode(SessionAffinityMode::Soft),
    )
    .unwrap();
    let ranked_binding = table_target(7, Some(2));
    bind(&coordinator, ranked_binding).await;

    let continuation = coordinator.acquire(&session_id(), None).await.unwrap();
    let lease = coordinator
        .commit(continuation, table_target(8, None))
        .unwrap();
    drop(lease);

    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(ranked_binding)
    );
}

#[tokio::test(start_paused = true)]
async fn session_affinity_ranked_binding_rejects_mismatched_rank_dispatch() {
    let coordinator = coordinator();
    let binding = table_target(7, Some(2));

    let initialization = coordinator.acquire(&session_id(), None).await.unwrap();
    drop(coordinator.commit(initialization, binding).unwrap());

    let continuation = coordinator.acquire(&session_id(), None).await.unwrap();
    assert!(
        coordinator
            .commit(continuation, table_target(7, Some(3)))
            .is_err()
    );
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
}

#[tokio::test(start_paused = true)]
async fn soft_affinity_rebinds_after_dispatch() {
    let coordinator = SessionAffinity::with_config(
        SessionAffinityConfig::new(TTL).with_mode(SessionAffinityMode::Soft),
    )
    .unwrap();
    let original = table_target(7, Some(0));
    let replacement = table_target(8, Some(1));
    bind(&coordinator, original).await;

    let failed_attempt = coordinator.acquire(&session_id(), None).await.unwrap();
    drop(failed_attempt);
    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(original)
    );

    let successful_attempt = coordinator.acquire(&session_id(), None).await.unwrap();
    let lease = coordinator.commit(successful_attempt, replacement).unwrap();
    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(replacement)
    );
    drop(lease);
}

#[tokio::test(start_paused = true)]
async fn concurrent_soft_rebind_uses_observed_version_cas() {
    let coordinator = SessionAffinity::with_config(
        SessionAffinityConfig::new(TTL).with_mode(SessionAffinityMode::Soft),
    )
    .unwrap();
    let original = table_target(7, Some(0));
    let winner = table_target(8, Some(0));
    let stale = table_target(9, Some(0));
    bind(&coordinator, original).await;
    let first = coordinator.acquire(&session_id(), None).await.unwrap();
    let second = coordinator.acquire(&session_id(), None).await.unwrap();

    let first_lease = coordinator.commit(first, winner).unwrap();
    let second_lease = coordinator.commit(second, stale).unwrap();
    drop(first_lease);
    drop(second_lease);
    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(winner)
    );
}

#[tokio::test(start_paused = true)]
async fn replica_applies_only_newer_affinity_versions() {
    let coordinator = coordinator();
    let first = table_target(7, Some(0));
    let newer = table_target(8, Some(0));
    let stale = table_target(9, Some(0));

    assert_eq!(
        coordinator.apply_replica_update(
            "ordered".into(),
            first,
            AffinityVersion {
                sequence: 3,
                writer_id: 10
            }
        ),
        ReplicaApplyOutcome::Inserted
    );
    assert_eq!(
        coordinator.apply_replica_update(
            "ordered".into(),
            newer,
            AffinityVersion {
                sequence: 4,
                writer_id: 10
            }
        ),
        ReplicaApplyOutcome::ReplacedNewer
    );
    assert_eq!(
        coordinator.apply_replica_update(
            "ordered".into(),
            stale,
            AffinityVersion {
                sequence: 3,
                writer_id: 11
            }
        ),
        ReplicaApplyOutcome::IgnoredConflict
    );
    assert_eq!(
        coordinator.query_target("ordered", None).unwrap(),
        Some(newer)
    );
}

#[tokio::test(start_paused = true)]
async fn stale_lease_cannot_invalidate_or_refresh_newer_replica_binding() {
    let coordinator = coordinator();
    let original = table_target(7, Some(0));
    let replacement = table_target(8, Some(0));
    bind(&coordinator, original).await;
    let stale = coordinator.acquire(&session_id(), None).await.unwrap();
    let replacement_sequence = u64::MAX / 2;
    coordinator.apply_replica_update(
        session_id().into(),
        replacement,
        AffinityVersion {
            sequence: replacement_sequence,
            writer_id: 1,
        },
    );
    stale.invalidate();
    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(replacement)
    );

    let stale = coordinator.acquire(&session_id(), None).await.unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    coordinator.apply_replica_update(
        session_id().into(),
        replacement,
        AffinityVersion {
            sequence: replacement_sequence.saturating_add(1),
            writer_id: 1,
        },
    );
    tokio::time::advance(Duration::from_secs(9)).await;
    drop(stale);
    tokio::time::advance(Duration::from_secs(2)).await;
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_completion_restores_expired_remote_binding() {
    let origin = coordinator();
    let mut updates = enable_replica(&origin, 99, 4);
    let replica = coordinator();
    let replicated_target = table_target(7, Some(0));
    let operation = origin.acquire(&session_id(), None).await.unwrap();
    let lease = origin.commit(operation, replicated_target).unwrap();

    let after_dispatch = updates.recv().await.unwrap();
    let version = after_dispatch.version();
    assert_eq!(
        replica.apply_replica_update(
            after_dispatch.session_id,
            table_target(after_dispatch.worker_id, after_dispatch.dp_rank),
            version
        ),
        ReplicaApplyOutcome::Inserted
    );
    tokio::time::advance(Duration::from_secs(11)).await;
    assert_eq!(replica.query_target(&session_id(), None).unwrap(), None);

    drop(lease);
    let after_completion = updates.recv().await.unwrap();
    let version = after_completion.version();
    assert_eq!(
        replica.apply_replica_update(
            after_completion.session_id,
            table_target(after_completion.worker_id, after_completion.dp_rank),
            version
        ),
        ReplicaApplyOutcome::ReplacedExpired
    );
    assert_eq!(
        replica.query_target(&session_id(), None).unwrap(),
        Some(replicated_target)
    );
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
