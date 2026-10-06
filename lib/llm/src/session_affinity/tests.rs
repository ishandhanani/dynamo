// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The frontend's session-affinity behavior over the shared resolver: the
//! lease-owning response stream, the frontend error mapping, replication
//! through the frontend sink, and the session-key sources. The table and
//! resolver rules themselves are pinned in `dynamo_kv_router`; these tests
//! drive them the way the routing hosts do.

use std::{future::Future, sync::Arc, time::Duration};

use dynamo_runtime::{
    engine::AsyncEngineContext,
    error::ErrorType,
    pipeline::{Context, Error, ManyOut, ResponseStream, context::Controller},
    protocols::maybe_error::MaybeError,
};
use futures::{StreamExt, stream};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::SessionAffinityMode::{Hard, Soft};
use super::{
    AffinityTarget, AlwaysLive, Hold, HostAffinity, LlmResponse, affinity_error, affinity_id,
    explicit_target, from_table, replica_sync::ReplicaSyncRuntime, subagent_group_affinity_id,
    to_table, tracked_stream,
};
use crate::{
    preprocessor::PreprocessedRequest,
    protocols::common::{
        extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
        llm_backend::LLMEngineOutput,
        preprocessor::RoutingHints,
        timing::RequestPhase,
    },
    types::Annotated,
};
use dynamo_kv_router::services::selection::affinity::{
    AffinityBindingEvent, AffinityResolver, AffinityVersion, MAX_SESSION_AFFINITY_ID_BYTES,
    ReplicaApplyOutcome, Resolution, SessionAffinity, SessionAffinityConfig,
};

fn session_id() -> SessionAffinityId {
    SessionAffinityId::new("session-1")
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

/// A frontend host's session layer as the routing hosts drive it: the shared
/// resolver over a table whose bound workers are all live, with the frontend's
/// stream-owned leases and error mapping, plus a test replica sink.
#[derive(Clone)]
struct TestAffinity {
    resolver: Arc<AffinityResolver>,
    replica: Arc<std::sync::Mutex<Option<ReplicaSyncRuntime>>>,
}

impl TestAffinity {
    fn with_config(config: SessionAffinityConfig) -> Self {
        let table = SessionAffinity::with_config(config).expect("affinity table");
        Self {
            resolver: Arc::new(AffinityResolver::new(table, Arc::new(AlwaysLive))),
            replica: Arc::default(),
        }
    }

    fn with_test_limits(max_entries: usize, max_session_id_bytes: usize) -> Self {
        Self::with_config(SessionAffinityConfig {
            max_entries,
            max_session_id_bytes,
            ..SessionAffinityConfig::new(Duration::from_secs(10))
        })
    }

    fn table(&self) -> &SessionAffinity {
        self.resolver.table()
    }

    async fn resolve(
        &self,
        session_id: &SessionAffinityId,
        requested: Option<AffinityTarget>,
        cancel: impl Future<Output = ()>,
    ) -> Result<Resolution, Error> {
        self.resolver
            .resolve(session_id.as_str(), requested.map(to_table), cancel)
            .await
            .map_err(affinity_error)
    }

    /// Hold the session, as an admitted request does; a full table is a
    /// resolver outcome, not an error, so it panics here.
    async fn acquire(
        &self,
        session_id: &SessionAffinityId,
        requested: Option<AffinityTarget>,
    ) -> Result<Hold, Error> {
        let resolution = self
            .resolve(session_id, requested, std::future::pending())
            .await?;
        Ok(resolution.hold.expect("session affinity table has room"))
    }

    /// Hold the session, abandoning the wait when the request context stops.
    async fn acquire_with_context(
        &self,
        session_id: &SessionAffinityId,
        requested: Option<AffinityTarget>,
        context: &dyn AsyncEngineContext,
    ) -> Result<Hold, Error> {
        let resolution = self
            .resolve(session_id, requested, async {
                tokio::select! {
                    _ = context.stopped() => {}
                    _ = context.killed() => {}
                }
            })
            .await?;
        Ok(resolution.hold.expect("session affinity table has room"))
    }

    fn query_target(
        &self,
        session_id: &SessionAffinityId,
        requested: Option<AffinityTarget>,
    ) -> Result<Option<AffinityTarget>, Error> {
        self.resolver
            .query(session_id.as_str(), requested.map(to_table))
            .map(|requirement| requirement.map(|requirement| from_table(requirement.target)))
            .map_err(affinity_error)
    }

    /// Commit after dispatch and hold the lease until `stream` ends.
    fn commit_to_stream(
        &self,
        hold: Hold,
        dispatched_target: AffinityTarget,
        stream: ManyOut<LlmResponse>,
    ) -> Result<ManyOut<LlmResponse>, Error> {
        match self
            .resolver
            .commit(hold, to_table(dispatched_target))
            .map_err(affinity_error)?
        {
            Some(lease) => Ok(tracked_stream(lease, stream)),
            None => Ok(stream),
        }
    }

    fn entry_count(&self) -> usize {
        self.table().entry_count()
    }

    fn cancellation_token(&self) -> CancellationToken {
        self.table().cancellation_token()
    }

    async fn wait_for_reaper(&self) {
        self.table().wait_for_reaper().await;
    }

    async fn wait_for_initializing_waiter(&self) {
        self.table().wait_for_initializing_waiter().await;
    }

    fn expire_for_test(&self, session_id: &SessionAffinityId) {
        self.table().expire_for_test(session_id.as_str());
    }

    fn enable_test_replica(
        &self,
        router_id: u64,
        capacity: usize,
    ) -> mpsc::Receiver<AffinityBindingEvent> {
        let (replica, rx) = ReplicaSyncRuntime::for_test(capacity);
        assert!(
            self.table().enable_replication(router_id, replica.sink()),
            "session affinity test replica already enabled"
        );
        *self.replica.lock().unwrap() = Some(replica);
        rx
    }

    fn apply_replica_update_for_test(
        &self,
        session_id: impl Into<String>,
        target: AffinityTarget,
    ) -> ReplicaApplyOutcome {
        self.apply_versioned_replica_update_for_test(session_id, target, 0, 0)
    }

    fn apply_versioned_replica_update_for_test(
        &self,
        session_id: impl Into<String>,
        target: AffinityTarget,
        sequence: u64,
        writer_id: u64,
    ) -> ReplicaApplyOutcome {
        self.table().apply_replica_update(
            session_id.into(),
            to_table(target),
            AffinityVersion {
                sequence,
                writer_id,
            },
        )
    }
}

fn affinity() -> TestAffinity {
    TestAffinity::with_config(SessionAffinityConfig::new(Duration::from_secs(10)))
}

fn soft_affinity() -> TestAffinity {
    TestAffinity::with_config(SessionAffinityConfig::new(Duration::from_secs(10)).with_mode(Soft))
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

async fn bind(affinity: &TestAffinity, target: AffinityTarget) {
    let operation = affinity.acquire(&session_id(), None).await.unwrap();
    let mut stream = affinity
        .commit_to_stream(operation, target, response_stream(1))
        .unwrap();
    while stream.next().await.is_some() {}
}

async fn assert_binding_expires_after_refreshed_ttl(affinity: &TestAffinity) {
    tokio::time::advance(Duration::from_secs(9)).await;
    assert_eq!(
        affinity.query_target(&session_id(), None).unwrap(),
        Some(target(7, Some(0)))
    );
    tokio::time::advance(Duration::from_secs(2)).await;
    assert_eq!(affinity.query_target(&session_id(), None).unwrap(), None);
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
    let affinity = affinity();
    let first = affinity.acquire(&session_id(), None).await.unwrap();
    let Hold::Initialize(first) = first else {
        panic!("first request must initialize");
    };

    let waiter_affinity = affinity.clone();
    let waiter = tokio::spawn(async move { waiter_affinity.acquire(&session_id(), None).await });
    affinity.wait_for_initializing_waiter().await;
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
    let affinity = affinity();
    let first = affinity.acquire(&session_id(), None).await.unwrap();
    let Hold::Initialize(first) = first else {
        panic!("first request must initialize");
    };

    let waiter_affinity = affinity.clone();
    let waiter = tokio::spawn(async move { waiter_affinity.acquire(&session_id(), None).await });
    affinity.wait_for_initializing_waiter().await;
    drop(first);

    let next = waiter.await.unwrap().unwrap();
    assert!(matches!(&next, Hold::Initialize(_)));
    drop(next);
    assert_eq!(affinity.entry_count(), 0);
    assert!(matches!(
        affinity.acquire(&session_id(), None).await.unwrap(),
        Hold::Initialize(_)
    ));
}

#[tokio::test(start_paused = true)]
async fn session_affinity_wait_stops_when_request_is_cancelled() {
    let affinity = affinity();
    let first = affinity.acquire(&session_id(), None).await.unwrap();
    let Hold::Initialize(first) = first else {
        panic!("first request must initialize");
    };

    let context = Arc::new(Controller::default());
    let waiter_context = context.clone();
    let waiter_affinity = affinity.clone();
    let waiter = tokio::spawn(async move {
        waiter_affinity
            .acquire_with_context(&session_id(), None, waiter_context.as_ref())
            .await
    });
    affinity.wait_for_initializing_waiter().await;
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
    let affinity = affinity();
    let Hold::Initialize(initializer) = affinity
        .acquire(&session_id(), Some(target(7, None)))
        .await
        .unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(table_target(7, None)).unwrap());

    assert!(
        affinity
            .acquire(&session_id(), Some(target(8, None)))
            .await
            .is_err()
    );
    assert!(
        affinity
            .acquire(&session_id(), Some(target(7, Some(0))))
            .await
            .is_err()
    );
    assert!(
        affinity
            .acquire(&session_id(), Some(target(7, None)))
            .await
            .is_ok()
    );
}

#[tokio::test(start_paused = true)]
async fn session_affinity_failed_bound_operation_invalidates_binding() {
    let affinity = affinity();
    let Hold::Initialize(initializer) = affinity.acquire(&session_id(), None).await.unwrap() else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(table_target(7, Some(0))).unwrap());

    let operation = affinity.acquire(&session_id(), None).await.unwrap();
    assert_eq!(operation.target(), Some(table_target(7, Some(0))));
    operation.invalidate();

    assert_eq!(affinity.query_target(&session_id(), None).unwrap(), None);
    assert_eq!(affinity.entry_count(), 0);
    assert!(matches!(
        affinity.acquire(&session_id(), None).await.unwrap(),
        Hold::Initialize(_)
    ));
}

#[tokio::test(start_paused = true)]
async fn session_affinity_stream_drop_refreshes_idle_ttl() {
    let affinity = affinity();
    let Hold::Initialize(initializer) = affinity.acquire(&session_id(), None).await.unwrap() else {
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
    let Hold::Initialize(initializer) = affinity.acquire(&session_id(), None).await.unwrap() else {
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
    let Hold::Initialize(initializer) = affinity.acquire(&session_id(), None).await.unwrap() else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(table_target(7, Some(0))).unwrap());

    tokio::time::advance(Duration::from_secs(9)).await;
    let Hold::Bound {
        target: bound_target,
        lease,
    } = affinity.acquire(&session_id(), None).await.unwrap()
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
    let operation = affinity.acquire(&session_id(), None).await.unwrap();
    let mut stream = affinity
        .commit_to_stream(operation, target(7, Some(0)), cancelled_response_stream())
        .unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    assert!(stream.next().await.is_none());

    assert_binding_expires_after_refreshed_ttl(&affinity).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_error_stream_refreshes_idle_ttl() {
    let affinity = affinity();
    let Hold::Initialize(initializer) = affinity.acquire(&session_id(), None).await.unwrap() else {
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
    let Hold::Initialize(initializer) = affinity.acquire(&session_id(), None).await.unwrap() else {
        panic!("first request must initialize");
    };
    let lease = initializer.commit(table_target(7, Some(0))).unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    let mut stream = tracked_stream(lease, response_stream(1));
    while stream.next().await.is_some() {}

    assert_binding_expires_after_refreshed_ttl(&affinity).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_bound_lease_drop_refreshes_idle_ttl() {
    let affinity = affinity();
    let Hold::Initialize(initializer) = affinity.acquire(&session_id(), None).await.unwrap() else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(table_target(7, Some(0))).unwrap());

    tokio::time::advance(Duration::from_secs(9)).await;
    let Hold::Bound { lease, .. } = affinity.acquire(&session_id(), None).await.unwrap() else {
        panic!("continuation must acquire the binding");
    };
    tokio::time::advance(Duration::from_secs(2)).await;
    tokio::task::yield_now().await;
    assert_eq!(
        affinity.query_target(&session_id(), None).unwrap(),
        Some(target(7, Some(0)))
    );
    drop(lease);

    assert_binding_expires_after_refreshed_ttl(&affinity).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_query_is_read_only() {
    let affinity = affinity();
    assert_eq!(affinity.query_target(&session_id(), None).unwrap(), None);
    assert_eq!(affinity.entry_count(), 0);

    let initializing = affinity.acquire(&session_id(), None).await.unwrap();
    assert_eq!(affinity.query_target(&session_id(), None).unwrap(), None);
    assert_eq!(affinity.entry_count(), 1);
    drop(initializing);
    assert_eq!(affinity.entry_count(), 0);

    let Hold::Initialize(initializer) = affinity.acquire(&session_id(), None).await.unwrap() else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(table_target(7, Some(0))).unwrap());
    assert_eq!(
        affinity.query_target(&session_id(), None).unwrap(),
        Some(target(7, Some(0)))
    );
    affinity.expire_for_test(&session_id());
    assert_eq!(affinity.query_target(&session_id(), None).unwrap(), None);
    assert_eq!(affinity.entry_count(), 1);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_reaper_removes_idle_entries_and_stops_on_drop() {
    let affinity = affinity();
    let cancellation = affinity.cancellation_token();
    let Hold::Initialize(initializer) = affinity.acquire(&session_id(), None).await.unwrap() else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(table_target(7, Some(0))).unwrap());

    affinity.wait_for_reaper().await;
    tokio::time::advance(Duration::from_secs(10)).await;
    tokio::task::yield_now().await;
    assert_eq!(affinity.entry_count(), 0);

    drop(affinity);
    cancellation.cancelled().await;
}

#[test]
fn session_affinity_rejects_invalid_ttl_before_starting_reaper() {
    for ttl in [
        Duration::ZERO,
        Duration::from_secs(super::MAX_SESSION_AFFINITY_TTL_SECS + 1),
    ] {
        let Err(error) = HostAffinity::standalone(ttl, Hard, Arc::new(AlwaysLive)) else {
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
async fn session_affinity_enforces_id_and_entry_limits() {
    let affinity = TestAffinity::with_test_limits(1, 8);
    let oversized = SessionAffinityId::new("123456789");
    let Err(error) = affinity.acquire(&oversized, None).await else {
        panic!("oversized session ID must fail");
    };
    assert!(dynamo_runtime::error::match_error_chain(
        error.as_ref(),
        &[ErrorType::InvalidArgument],
        &[]
    ));
    assert_eq!(affinity.entry_count(), 0);

    let first_id = SessionAffinityId::new("first");
    let first = affinity.acquire(&first_id, None).await.unwrap();
    let second_id = SessionAffinityId::new("second");
    let second = affinity
        .resolve(&second_id, None, std::future::pending())
        .await
        .expect("a full table is not a request failure");
    assert!(
        second.hold.is_none() && second.affinity.is_none(),
        "the second session routes without affinity"
    );
    assert_eq!(affinity.resolver.full_table_fallbacks(), 1);

    drop(first);
    assert_eq!(affinity.entry_count(), 0);
    assert!(matches!(
        affinity.acquire(&second_id, None).await.unwrap(),
        Hold::Initialize(_)
    ));
}

#[tokio::test(start_paused = true)]
async fn session_affinity_replica_applies_first_live_binding_wins() {
    let affinity = affinity();
    let local_target = target(7, Some(0));
    let conflicting_target = target(8, Some(1));

    assert_eq!(
        affinity.apply_replica_update_for_test("replicated", local_target),
        ReplicaApplyOutcome::Inserted
    );
    assert_eq!(
        affinity
            .query_target(&SessionAffinityId::new("replicated"), None)
            .unwrap(),
        Some(local_target)
    );
    assert_eq!(
        affinity.apply_replica_update_for_test("replicated", conflicting_target),
        ReplicaApplyOutcome::IgnoredConflict
    );

    affinity.expire_for_test(&SessionAffinityId::new("replicated"));
    assert_eq!(
        affinity.apply_replica_update_for_test("replicated", conflicting_target),
        ReplicaApplyOutcome::ReplacedExpired
    );
    assert_eq!(
        affinity
            .query_target(&SessionAffinityId::new("replicated"), None)
            .unwrap(),
        Some(conflicting_target)
    );
}

#[tokio::test(start_paused = true)]
async fn session_affinity_replica_duplicate_refreshes_local_ttl() {
    let affinity = affinity();
    let replicated_id = SessionAffinityId::new("replicated");
    let replicated_target = target(7, Some(0));

    assert_eq!(
        affinity.apply_replica_update_for_test("replicated", replicated_target),
        ReplicaApplyOutcome::Inserted
    );
    tokio::time::advance(Duration::from_secs(9)).await;
    assert_eq!(
        affinity.apply_replica_update_for_test("replicated", replicated_target),
        ReplicaApplyOutcome::Refreshed
    );
    tokio::time::advance(Duration::from_secs(2)).await;
    assert_eq!(
        affinity.query_target(&replicated_id, None).unwrap(),
        Some(replicated_target)
    );
    tokio::time::advance(Duration::from_secs(9)).await;
    assert_eq!(affinity.query_target(&replicated_id, None).unwrap(), None);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_replica_ignores_initializing_sessions() {
    let affinity = affinity();
    let initializing = affinity.acquire(&session_id(), None).await.unwrap();

    assert_eq!(
        affinity.apply_replica_update_for_test(session_id().as_str(), target(7, Some(0))),
        ReplicaApplyOutcome::IgnoredInitializing
    );
    assert_eq!(affinity.query_target(&session_id(), None).unwrap(), None);
    drop(initializing);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_replica_enforces_id_and_entry_limits() {
    let affinity = TestAffinity::with_test_limits(1, 8);

    assert_eq!(
        affinity.apply_replica_update_for_test("123456789", target(7, Some(0))),
        ReplicaApplyOutcome::RejectedSessionId
    );
    assert_eq!(
        affinity.apply_replica_update_for_test("first", target(7, Some(0))),
        ReplicaApplyOutcome::Inserted
    );
    assert_eq!(
        affinity.apply_replica_update_for_test("second", target(8, Some(0))),
        ReplicaApplyOutcome::RejectedCapacity
    );
    assert_eq!(affinity.entry_count(), 1);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_publishes_after_dispatch_and_lease_completion() {
    let affinity = affinity();
    let mut updates = affinity.enable_test_replica(99, 4);
    let selected_target = target(7, Some(0));
    let operation = affinity.acquire(&session_id(), None).await.unwrap();
    let stream = affinity
        .commit_to_stream(operation, selected_target, response_stream(1))
        .unwrap();

    let after_dispatch = updates.recv().await.unwrap();
    assert_eq!(after_dispatch.session_id, session_id().as_str());
    assert_eq!(after_dispatch.worker_id, selected_target.worker_id);
    assert_eq!(after_dispatch.dp_rank, selected_target.dp_rank);
    assert_eq!(after_dispatch.writer_id, 99);

    drop(stream);
    let after_completion = updates.recv().await.unwrap();
    assert_eq!(after_completion, after_dispatch);
    assert!(updates.try_recv().is_err());
}

#[tokio::test(start_paused = true)]
async fn session_affinity_republishes_the_stored_replica_version() {
    let affinity = affinity();
    let mut updates = affinity.enable_test_replica(99, 1);
    let replicated_target = target(7, Some(0));
    assert_eq!(
        affinity.apply_versioned_replica_update_for_test(
            session_id().as_str(),
            replicated_target,
            123,
            7,
        ),
        ReplicaApplyOutcome::Inserted
    );

    let Hold::Bound { lease, .. } = affinity.acquire(&session_id(), None).await.unwrap() else {
        panic!("replicated binding must be acquired");
    };
    drop(lease);

    let update = updates.recv().await.unwrap();
    assert_eq!(update.sequence, 123);
    assert_eq!(update.writer_id, 7);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_worker_only_binding_allows_ranked_dispatch_without_narrowing() {
    let affinity = affinity();
    let worker_binding = target(7, None);

    let initialization = affinity.acquire(&session_id(), None).await.unwrap();
    drop(
        affinity
            .commit_to_stream(initialization, worker_binding, response_stream(1))
            .unwrap(),
    );

    let continuation = affinity.acquire(&session_id(), None).await.unwrap();
    let stream = affinity
        .commit_to_stream(continuation, target(7, Some(3)), response_stream(1))
        .expect("worker-only affinity must allow the scheduler to select a DP rank");

    assert_eq!(
        affinity.query_target(&session_id(), None).unwrap(),
        Some(worker_binding)
    );
    drop(stream);
}

#[tokio::test(start_paused = true)]
async fn soft_worker_only_binding_stays_worker_scoped_after_ranked_dispatch() {
    let affinity = soft_affinity();
    let worker_binding = target(7, None);
    bind(&affinity, worker_binding).await;

    let continuation = affinity.acquire(&session_id(), None).await.unwrap();
    let mut stream = affinity
        .commit_to_stream(continuation, target(7, Some(3)), response_stream(1))
        .unwrap();
    while stream.next().await.is_some() {}

    assert_eq!(
        affinity.query_target(&session_id(), None).unwrap(),
        Some(worker_binding)
    );
}

#[tokio::test(start_paused = true)]
async fn soft_ranked_binding_is_not_widened_by_worker_only_dispatch() {
    let affinity = soft_affinity();
    let ranked_binding = target(7, Some(2));
    bind(&affinity, ranked_binding).await;

    let continuation = affinity.acquire(&session_id(), None).await.unwrap();
    let mut stream = affinity
        .commit_to_stream(continuation, target(8, None), response_stream(1))
        .unwrap();
    while stream.next().await.is_some() {}

    assert_eq!(
        affinity.query_target(&session_id(), None).unwrap(),
        Some(ranked_binding)
    );
}

#[tokio::test(start_paused = true)]
async fn session_affinity_ranked_binding_rejects_mismatched_rank_dispatch() {
    let affinity = affinity();
    let binding = target(7, Some(2));

    let initialization = affinity.acquire(&session_id(), None).await.unwrap();
    drop(
        affinity
            .commit_to_stream(initialization, binding, response_stream(1))
            .unwrap(),
    );

    let continuation = affinity.acquire(&session_id(), None).await.unwrap();
    assert!(
        affinity
            .commit_to_stream(continuation, target(7, Some(3)), response_stream(1))
            .is_err()
    );
    assert_eq!(affinity.query_target(&session_id(), None).unwrap(), None);
}

#[tokio::test(start_paused = true)]
async fn soft_affinity_rebinds_after_dispatch() {
    let affinity = soft_affinity();
    let original = target(7, Some(0));
    let replacement = target(8, Some(1));
    bind(&affinity, original).await;

    let failed_attempt = affinity.acquire(&session_id(), None).await.unwrap();
    drop(failed_attempt);
    assert_eq!(
        affinity.query_target(&session_id(), None).unwrap(),
        Some(original)
    );

    let successful_attempt = affinity.acquire(&session_id(), None).await.unwrap();
    let stream = affinity
        .commit_to_stream(successful_attempt, replacement, response_stream(1))
        .unwrap();
    assert_eq!(
        affinity.query_target(&session_id(), None).unwrap(),
        Some(replacement)
    );
    drop(stream);
}

#[tokio::test(start_paused = true)]
async fn concurrent_soft_rebind_uses_observed_version_cas() {
    let affinity = soft_affinity();
    let original = target(7, Some(0));
    let winner = target(8, Some(0));
    let stale = target(9, Some(0));
    bind(&affinity, original).await;
    let first = affinity.acquire(&session_id(), None).await.unwrap();
    let second = affinity.acquire(&session_id(), None).await.unwrap();

    let mut first_stream = affinity
        .commit_to_stream(first, winner, response_stream(1))
        .unwrap();
    let mut second_stream = affinity
        .commit_to_stream(second, stale, response_stream(1))
        .unwrap();
    while first_stream.next().await.is_some() {}
    while second_stream.next().await.is_some() {}
    assert_eq!(
        affinity.query_target(&session_id(), None).unwrap(),
        Some(winner)
    );
}

#[tokio::test(start_paused = true)]
async fn replica_applies_only_newer_affinity_versions() {
    let affinity = affinity();
    let first = target(7, Some(0));
    let newer = target(8, Some(0));
    let stale = target(9, Some(0));

    assert_eq!(
        affinity.apply_versioned_replica_update_for_test("ordered", first, 3, 10),
        ReplicaApplyOutcome::Inserted
    );
    assert_eq!(
        affinity.apply_versioned_replica_update_for_test("ordered", newer, 4, 10),
        ReplicaApplyOutcome::ReplacedNewer
    );
    assert_eq!(
        affinity.apply_versioned_replica_update_for_test("ordered", stale, 3, 11),
        ReplicaApplyOutcome::IgnoredConflict
    );
    assert_eq!(
        affinity
            .query_target(&SessionAffinityId::new("ordered"), None)
            .unwrap(),
        Some(newer)
    );
}

#[tokio::test(start_paused = true)]
async fn stale_lease_cannot_invalidate_or_refresh_newer_replica_binding() {
    let affinity = affinity();
    let original = target(7, Some(0));
    let replacement = target(8, Some(0));
    bind(&affinity, original).await;
    let stale = affinity.acquire(&session_id(), None).await.unwrap();
    let replacement_sequence = u64::MAX / 2;
    affinity.apply_versioned_replica_update_for_test(
        session_id().as_str(),
        replacement,
        replacement_sequence,
        1,
    );
    stale.invalidate();
    assert_eq!(
        affinity.query_target(&session_id(), None).unwrap(),
        Some(replacement)
    );

    let stale = affinity.acquire(&session_id(), None).await.unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    affinity.apply_versioned_replica_update_for_test(
        session_id().as_str(),
        replacement,
        replacement_sequence.saturating_add(1),
        1,
    );
    tokio::time::advance(Duration::from_secs(9)).await;
    drop(stale);
    tokio::time::advance(Duration::from_secs(2)).await;
    assert_eq!(affinity.query_target(&session_id(), None).unwrap(), None);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_completion_restores_expired_remote_binding() {
    let origin = affinity();
    let mut updates = origin.enable_test_replica(99, 4);
    let replica = affinity();
    let replicated_target = target(7, Some(0));
    let operation = origin.acquire(&session_id(), None).await.unwrap();
    let stream = origin
        .commit_to_stream(operation, replicated_target, response_stream(1))
        .unwrap();

    let after_dispatch = updates.recv().await.unwrap();
    assert_eq!(
        replica.apply_replica_update_for_test(
            after_dispatch.session_id,
            target(after_dispatch.worker_id, after_dispatch.dp_rank),
        ),
        ReplicaApplyOutcome::Inserted
    );
    tokio::time::advance(Duration::from_secs(11)).await;
    assert_eq!(replica.query_target(&session_id(), None).unwrap(), None);

    drop(stream);
    let after_completion = updates.recv().await.unwrap();
    assert_eq!(
        replica.apply_replica_update_for_test(
            after_completion.session_id,
            target(after_completion.worker_id, after_completion.dp_rank),
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
