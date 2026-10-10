// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How a routing host drives a [`SessionAffinity`] table for one request.
//!
//! `resolve` runs before worker selection and yields the [`Hold`] to commit
//! plus the [`AffinityRequirement`] the scheduler enforces. `commit` runs
//! once the host knows where the request went: after booking for a host that
//! owns the booking, after dispatch for a host that owns the response stream.
//! `query` serves selections that must not bind anything.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::table::{
    AcquireStep, AffinityError, AffinityLease, AffinityTarget, Hold, SessionAffinity,
    SessionAffinityMode, validate_dispatch_target,
};
use crate::scheduling::AffinityRequirement;

/// `try_acquire` then `join_initializing` attempts before a commit that finds
/// the session initializing routes unpinned. The join misses only if that
/// initialization resolved between the two calls; one retry covers that, and
/// a booking never waits on a request queued behind it.
const JOIN_ATTEMPTS: usize = 2;
/// Replica updates can keep re-binding a session to a departed worker fast
/// enough that `resolve` never leaves its invalidate loop; yield periodically.
const INVALIDATIONS_BEFORE_YIELD: usize = 32;

/// What the host knows about worker liveness that the table does not: whether
/// a bound target can receive this partition's requests right now.
pub trait TargetLiveness: Send + Sync {
    fn is_schedulable(&self, target: AffinityTarget) -> bool;
}

/// What [`AffinityResolver::resolve`] found: the hold to commit and the
/// requirement the scheduler enforces.
#[must_use]
pub struct Resolution {
    /// The session, held until the host commits it. `None` when the table is
    /// full: the request routes without affinity.
    pub hold: Option<Hold>,
    /// The bound target with the table's mode as its strength; `None` for a
    /// new session or a full table.
    pub affinity: Option<AffinityRequirement>,
}

/// Session resolution for one partition: the table plus the host's liveness
/// source.
pub struct AffinityResolver {
    table: SessionAffinity,
    liveness: Arc<dyn TargetLiveness>,
    full_table_fallbacks: AtomicU64,
    #[cfg(any(test, feature = "testing"))]
    after_invalidation: std::sync::OnceLock<Box<dyn Fn() + Send + Sync>>,
}

impl AffinityResolver {
    pub fn new(table: SessionAffinity, liveness: Arc<dyn TargetLiveness>) -> Self {
        Self {
            table,
            liveness,
            full_table_fallbacks: AtomicU64::new(0),
            #[cfg(any(test, feature = "testing"))]
            after_invalidation: std::sync::OnceLock::new(),
        }
    }

    pub fn table(&self) -> &SessionAffinity {
        &self.table
    }

    /// Requests that ran without a session binding because the table was
    /// full: at `resolve`, or at a failover `commit` that found no room to
    /// re-bind.
    pub fn full_table_fallbacks(&self) -> u64 {
        self.full_table_fallbacks.load(Ordering::Relaxed)
    }

    fn requirement(&self, target: AffinityTarget) -> AffinityRequirement {
        AffinityRequirement {
            target,
            strength: self.table.mode().into(),
        }
    }

    /// Hold `session_id` for a request about to select a worker.
    ///
    /// A binding whose target the host can no longer schedule is dropped and
    /// the session re-initialized, so the request selects normally and
    /// `commit` binds the worker it lands on. `requested` is an explicit
    /// target the request carries; a bound session must agree with it. A
    /// full table is a router-side limit, not a client fault: the request
    /// routes unpinned and the fallback is counted. `cancel` abandons a wait
    /// on another request's initialization, or the yield after a burst of
    /// invalidations; an acquisition that completes at once is not checked.
    pub async fn resolve(
        &self,
        session_id: &str,
        requested: Option<AffinityTarget>,
        cancel: impl Future<Output = ()>,
    ) -> Result<Resolution, AffinityError> {
        tokio::pin!(cancel);
        let mut invalidations = 0;
        loop {
            let hold = match self.table.try_acquire(session_id, requested) {
                Ok(AcquireStep::Held(hold)) => hold,
                Ok(AcquireStep::Wait(notified)) => {
                    tokio::select! {
                        biased;
                        _ = &mut cancel => return Err(AffinityError::Cancelled),
                        _ = notified => continue,
                    }
                }
                Err(AffinityError::ResourceExhausted(_)) => {
                    self.full_table_fallbacks.fetch_add(1, Ordering::Relaxed);
                    tracing::debug!(
                        session_id,
                        "Affinity table full; routing without session affinity"
                    );
                    return Ok(Resolution {
                        hold: None,
                        affinity: None,
                    });
                }
                Err(error) => return Err(error),
            };
            match hold {
                Hold::Bound { target, mut lease } if !self.liveness.is_schedulable(target) => {
                    tracing::debug!(
                        session_id,
                        worker_id = target.worker_id,
                        dp_rank = ?target.dp_rank,
                        "Session affinity target is not schedulable; re-initializing"
                    );
                    lease.invalidate();
                    drop(lease);
                    #[cfg(any(test, feature = "testing"))]
                    if let Some(hook) = self.after_invalidation.get() {
                        hook();
                    }
                    invalidations += 1;
                    if invalidations == INVALIDATIONS_BEFORE_YIELD {
                        tokio::select! {
                            biased;
                            _ = &mut cancel => return Err(AffinityError::Cancelled),
                            _ = tokio::task::yield_now() => {}
                        }
                        invalidations = 0;
                    }
                }
                hold => {
                    let affinity = hold.target().map(|target| self.requirement(target));
                    return Ok(Resolution {
                        hold: Some(hold),
                        affinity,
                    });
                }
            }
        }
    }

    /// The session's current target as a requirement, without a hold.
    pub fn query(
        &self,
        session_id: &str,
        requested: Option<AffinityTarget>,
    ) -> Result<Option<AffinityRequirement>, AffinityError> {
        Ok(self
            .table
            .query_target(session_id, requested)?
            .map(|target| self.requirement(target)))
    }

    /// Bind the held session to `dispatched`. In `Hard` mode a dispatch away
    /// from a live binding is rejected and the binding dropped. A rejection
    /// whose bound worker departed after `resolve` checked it is not a client
    /// fault: the session is re-bound to `dispatched`, joining another
    /// request's initialization when one is in flight. `None` means the
    /// request runs without a lease (the table filled up during failover).
    pub fn commit(
        &self,
        hold: Hold,
        dispatched: AffinityTarget,
    ) -> Result<Option<AffinityLease>, AffinityError> {
        let bound = hold.target();
        let session_id = hold.shared_session_id();
        let error = match self.table.commit(hold, dispatched) {
            Ok(lease) => return Ok(Some(lease)),
            Err(error) => error,
        };
        let departed = bound.is_some_and(|target| !self.liveness.is_schedulable(target));
        if !departed {
            return Err(error);
        }
        // `commit` invalidated the stale binding. Another request may already
        // be initializing the replacement: join it rather than wait, binding
        // it to this successful dispatch's target.
        for _ in 0..JOIN_ATTEMPTS {
            match self.table.try_acquire(&session_id, None) {
                Ok(AcquireStep::Held(hold)) => {
                    if self.table.mode() == SessionAffinityMode::Hard
                        && let Some(target) = hold.target()
                    {
                        // A competing failover already bound a valid target;
                        // reject a mismatch without erasing it.
                        validate_dispatch_target(&session_id, target, dispatched)?;
                    }
                    return self.table.commit(hold, dispatched).map(Some);
                }
                Ok(AcquireStep::Wait(_)) => {
                    if let Some(lease) = self.table.join_initializing(&session_id, dispatched) {
                        return Ok(Some(lease));
                    }
                }
                Err(AffinityError::ResourceExhausted(_)) => {
                    self.full_table_fallbacks.fetch_add(1, Ordering::Relaxed);
                    return Ok(None);
                }
                Err(error) => return Err(error),
            }
        }
        Ok(None)
    }

    /// Whether `commit` would reject `dispatched` for a live `Hard` binding.
    ///
    /// For a host that commits after dispatch, this is the check to run
    /// before dispatching: a `Hard` binding whose worker is live but was not
    /// selected (the request's own constraints excluded it) would be rejected
    /// at commit, after the worker has already started the request. The hold
    /// is left as it is; the host decides whether to release or invalidate it.
    pub fn check_dispatch(
        &self,
        hold: &Hold,
        dispatched: AffinityTarget,
    ) -> Result<(), AffinityError> {
        if self.table.mode() != SessionAffinityMode::Hard {
            return Ok(());
        }
        let Some(target) = hold.target() else {
            return Ok(());
        };
        if !self.liveness.is_schedulable(target) {
            // `commit` fails this binding over to `dispatched` instead.
            return Ok(());
        }
        validate_dispatch_target(hold.session_id(), target, dispatched)
    }

    /// Give up a hold because the selection policy filtered out every
    /// candidate. In `Hard` mode that includes the bound worker, which the
    /// scheduler had limited selection to: drop the binding so the session
    /// re-binds instead of failing on every retry. A `Soft` binding is kept.
    pub fn release_filtered(&self, hold: Hold) {
        if self.table.mode() == SessionAffinityMode::Hard {
            hold.invalidate();
        }
    }

    /// Run `hook` after each invalidation of an unschedulable binding.
    #[cfg(any(test, feature = "testing"))]
    pub fn set_after_invalidation(&self, hook: Box<dyn Fn() + Send + Sync>) {
        assert!(
            self.after_invalidation.set(hook).is_ok(),
            "after-invalidation hook already set"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    use parking_lot::Mutex;
    use tokio_util::sync::CancellationToken;

    use super::super::table::{AffinityVersion, SessionAffinityConfig};
    use super::*;
    use crate::protocols::WorkerId;
    use crate::scheduling::AffinityStrength;

    const TTL: Duration = Duration::from_secs(10);

    struct Liveness(Mutex<HashSet<WorkerId>>);

    impl Liveness {
        fn set(&self, live: &[WorkerId]) {
            *self.0.lock() = live.iter().copied().collect();
        }
    }

    impl TargetLiveness for Liveness {
        fn is_schedulable(&self, target: AffinityTarget) -> bool {
            self.0.lock().contains(&target.worker_id)
        }
    }

    fn make_resolver_with(
        config: SessionAffinityConfig,
        live: &[WorkerId],
    ) -> (AffinityResolver, Arc<Liveness>) {
        let liveness = Arc::new(Liveness(Mutex::new(live.iter().copied().collect())));
        let table = SessionAffinity::with_config(config).expect("affinity table");
        (AffinityResolver::new(table, liveness.clone()), liveness)
    }

    fn make_resolver(
        mode: SessionAffinityMode,
        live: &[WorkerId],
    ) -> (AffinityResolver, Arc<Liveness>) {
        make_resolver_with(SessionAffinityConfig::new(TTL).with_mode(mode), live)
    }

    fn target(worker_id: WorkerId, dp_rank: Option<u32>) -> AffinityTarget {
        AffinityTarget::new(worker_id, dp_rank)
    }

    fn never() -> std::future::Pending<()> {
        std::future::pending()
    }

    async fn hold(resolver: &AffinityResolver, session_id: &str) -> Hold {
        resolver
            .resolve(session_id, None, never())
            .await
            .expect("resolve")
            .hold
            .expect("table has room")
    }

    /// Bind `session_id` to `target` and release the lease.
    async fn bind(resolver: &AffinityResolver, session_id: &str, target: AffinityTarget) {
        let hold = hold(resolver, session_id).await;
        assert!(matches!(hold, Hold::Initialize(_)), "session must be new");
        drop(
            resolver
                .commit(hold, target)
                .expect("commit")
                .expect("lease"),
        );
    }

    fn bound(resolver: &AffinityResolver, session_id: &str) -> Option<AffinityTarget> {
        resolver
            .table()
            .query_target(session_id, None)
            .expect("query")
    }

    #[tokio::test]
    async fn new_session_resolves_to_an_initializing_hold() {
        let (resolver, _) = make_resolver(SessionAffinityMode::Hard, &[1]);
        let resolution = resolver.resolve("s", None, never()).await.expect("resolve");
        assert!(matches!(resolution.hold, Some(Hold::Initialize(_))));
        assert!(resolution.affinity.is_none());
        assert_eq!(
            resolution
                .hold
                .as_ref()
                .map(Hold::shared_session_id)
                .as_deref(),
            Some("s")
        );
    }

    #[tokio::test]
    async fn bound_live_session_resolves_to_its_target_with_the_table_mode() {
        for (mode, strength) in [
            (SessionAffinityMode::Hard, AffinityStrength::Hard),
            (SessionAffinityMode::Soft, AffinityStrength::Soft),
        ] {
            let (resolver, _) = make_resolver(mode, &[1, 2]);
            bind(&resolver, "s", target(1, Some(0))).await;
            let resolution = resolver.resolve("s", None, never()).await.expect("resolve");
            assert!(matches!(resolution.hold, Some(Hold::Bound { .. })));
            assert_eq!(
                resolution.affinity,
                Some(AffinityRequirement {
                    target: target(1, Some(0)),
                    strength
                })
            );
            assert_eq!(resolver.table().lease_count("s"), Some(1));
        }
    }

    #[tokio::test]
    async fn bound_unschedulable_target_is_invalidated_and_reinitialized() {
        let (resolver, liveness) = make_resolver(SessionAffinityMode::Hard, &[1, 2]);
        bind(&resolver, "s", target(1, Some(0))).await;
        liveness.set(&[2]);
        let resolution = resolver.resolve("s", None, never()).await.expect("resolve");
        assert!(matches!(resolution.hold, Some(Hold::Initialize(_))));
        assert!(resolution.affinity.is_none());
        assert_eq!(bound(&resolver, "s"), None, "the stale binding is dropped");
        let lease = resolver
            .commit(resolution.hold.unwrap(), target(2, Some(0)))
            .expect("commit")
            .expect("lease");
        assert_eq!(bound(&resolver, "s"), Some(target(2, Some(0))));
        drop(lease);
    }

    #[tokio::test]
    async fn repeated_invalidation_yields_and_preserves_new_bindings() {
        for cancel in [false, true] {
            let (resolver, _) = make_resolver(SessionAffinityMode::Hard, &[1]);
            let table = resolver.table().clone();
            let unavailable = target(2, Some(0));
            table.apply_replica_update(
                "s".into(),
                unavailable,
                AffinityVersion {
                    sequence: 1,
                    writer_id: 99,
                },
            );
            let invalidations = Arc::new(AtomicUsize::new(0));
            resolver.set_after_invalidation(Box::new({
                let table = table.clone();
                let invalidations = invalidations.clone();
                move || {
                    let count = invalidations.fetch_add(1, Ordering::SeqCst) + 1;
                    // A finite burst also makes a missing yield fail instead of hanging the test.
                    if count <= 64 {
                        table.apply_replica_update(
                            "s".into(),
                            unavailable,
                            AffinityVersion {
                                sequence: count as u64 + 1,
                                writer_id: 99,
                            },
                        );
                    }
                }
            }));
            let token = CancellationToken::new();
            let mut pending = Box::pin(resolver.resolve("s", None, token.cancelled()));
            let mut context = std::task::Context::from_waker(std::task::Waker::noop());
            assert!(pending.as_mut().poll(&mut context).is_pending());
            assert_eq!(invalidations.load(Ordering::SeqCst), 32);
            if cancel {
                token.cancel();
                assert!(matches!(pending.await, Err(AffinityError::Cancelled)));
            } else {
                let healthy = target(1, Some(0));
                table.apply_replica_update(
                    "s".into(),
                    healthy,
                    AffinityVersion {
                        sequence: 100,
                        writer_id: 99,
                    },
                );
                let resolution = pending.await.unwrap();
                assert!(
                    matches!(resolution.hold, Some(Hold::Bound { target, .. }) if target == healthy)
                );
                assert_eq!(
                    resolution.affinity,
                    Some(AffinityRequirement::hard(healthy))
                );
                assert_eq!(invalidations.load(Ordering::SeqCst), 32);
            }
        }
    }

    #[tokio::test]
    async fn cancellation_stops_waiting_on_an_initializing_entry() {
        let (resolver, _) = make_resolver(SessionAffinityMode::Hard, &[1]);
        let _initializing = hold(&resolver, "s").await;
        let token = CancellationToken::new();
        let mut waiting = Box::pin(resolver.resolve("s", None, token.cancelled()));
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(waiting.as_mut().poll(&mut context).is_pending());
        token.cancel();
        assert!(matches!(waiting.await, Err(AffinityError::Cancelled)));
        assert_eq!(
            resolver.table().entry_count(),
            1,
            "the initializer keeps its slot"
        );
    }

    #[tokio::test]
    async fn explicit_target_must_agree_with_a_live_binding() {
        let (resolver, _) = make_resolver(SessionAffinityMode::Hard, &[1, 2]);
        bind(&resolver, "s", target(1, Some(0))).await;

        let Err(error) = resolver.resolve("s", Some(target(2, None)), never()).await else {
            panic!("a disagreeing target is rejected");
        };
        assert!(matches!(error, AffinityError::InvalidArgument(_)));
        assert_eq!(
            bound(&resolver, "s"),
            Some(target(1, Some(0))),
            "rejection keeps the binding"
        );

        let resolution = resolver
            .resolve("s", Some(target(1, None)), never())
            .await
            .expect("an agreeing target binds");
        assert!(matches!(resolution.hold, Some(Hold::Bound { .. })));

        // A new session with an explicit target binds only to that target.
        let pinned = resolver
            .resolve("n", Some(target(2, Some(0))), never())
            .await
            .expect("resolve")
            .hold
            .expect("hold");
        assert!(matches!(
            resolver.commit(pinned, target(1, Some(0))),
            Err(AffinityError::InvalidArgument(_))
        ));
        assert_eq!(bound(&resolver, "n"), None);
        let hold = hold(&resolver, "n").await;
        drop(
            resolver
                .commit(hold, target(2, Some(0)))
                .expect("commit")
                .expect("lease"),
        );
        assert_eq!(bound(&resolver, "n"), Some(target(2, Some(0))));
    }

    #[tokio::test]
    async fn full_table_routes_unpinned_and_counts_the_fallback() {
        let (resolver, _) = make_resolver_with(
            SessionAffinityConfig {
                max_entries: 1,
                ..SessionAffinityConfig::new(TTL)
            },
            &[1],
        );
        let _first = hold(&resolver, "a").await;
        let resolution = resolver.resolve("b", None, never()).await.expect("resolve");
        assert!(resolution.hold.is_none());
        assert!(resolution.affinity.is_none());
        assert_eq!(resolver.full_table_fallbacks(), 1);
    }

    #[tokio::test]
    async fn hard_commit_away_from_a_live_binding_rejects_and_invalidates() {
        let (resolver, _) = make_resolver(SessionAffinityMode::Hard, &[1, 2]);
        bind(&resolver, "s", target(1, Some(0))).await;
        let hold = hold(&resolver, "s").await;
        let Err(error) = resolver.commit(hold, target(2, Some(0))) else {
            panic!("hard affinity rejects a dispatch away from a live binding");
        };
        assert!(matches!(error, AffinityError::InvalidArgument(_)));
        assert_eq!(bound(&resolver, "s"), None, "the binding is dropped");
    }

    #[tokio::test]
    async fn departed_binding_fails_over_to_the_dispatched_worker() {
        let (resolver, liveness) = make_resolver(SessionAffinityMode::Hard, &[1, 2]);
        bind(&resolver, "s", target(1, Some(0))).await;
        let hold = hold(&resolver, "s").await;
        liveness.set(&[2]);
        let lease = resolver
            .commit(hold, target(2, Some(0)))
            .expect("a departure after the hold is not a client fault")
            .expect("lease");
        assert_eq!(bound(&resolver, "s"), Some(target(2, Some(0))));
        assert_eq!(resolver.table().lease_count("s"), Some(1));
        drop(lease);
        assert_eq!(resolver.table().lease_count("s"), Some(0));
    }

    #[tokio::test]
    async fn departed_binding_joins_a_pending_initialization() {
        let (resolver, liveness) = make_resolver(SessionAffinityMode::Hard, &[1, 2]);
        bind(&resolver, "s", target(1, Some(0))).await;
        let stale = hold(&resolver, "s").await;
        liveness.set(&[2]);
        // A later request sees the departure and re-initializes the session.
        let initializing = hold(&resolver, "s").await;
        assert!(matches!(initializing, Hold::Initialize(_)));

        let joined = resolver
            .commit(stale, target(2, Some(0)))
            .expect("failover joins the initialization")
            .expect("lease");
        assert_eq!(bound(&resolver, "s"), Some(target(2, Some(0))));
        assert_eq!(resolver.table().lease_count("s"), Some(2));

        // The initializer commits to the same target and keeps its own lease.
        let lease = resolver
            .commit(initializing, target(2, Some(0)))
            .expect("commit")
            .expect("lease");
        assert_eq!(resolver.table().lease_count("s"), Some(2));
        drop(joined);
        drop(lease);
        assert_eq!(resolver.table().lease_count("s"), Some(0));
    }

    #[tokio::test]
    async fn late_failover_mismatch_preserves_the_committed_replacement() {
        let (resolver, liveness) = make_resolver(SessionAffinityMode::Hard, &[1, 2, 3]);
        bind(&resolver, "s", target(1, Some(0))).await;
        let old_hold = hold(&resolver, "s").await;
        liveness.set(&[2, 3]);
        let new_hold = hold(&resolver, "s").await;
        let lease = resolver
            .commit(new_hold, target(2, Some(0)))
            .expect("bind replacement")
            .expect("lease");
        let result = resolver.commit(old_hold, target(3, Some(0)));
        assert!(matches!(result, Err(AffinityError::InvalidArgument(_))));
        assert_eq!(bound(&resolver, "s"), Some(target(2, Some(0))));
        assert_eq!(resolver.table().lease_count("s"), Some(1));
        drop(lease);
        assert_eq!(resolver.table().lease_count("s"), Some(0));
    }

    #[tokio::test]
    async fn soft_commit_follows_the_dispatch() {
        let (resolver, _) = make_resolver(SessionAffinityMode::Soft, &[1, 2]);
        bind(&resolver, "s", target(1, Some(0))).await;
        let hold = hold(&resolver, "s").await;
        let lease = resolver
            .commit(hold, target(2, Some(1)))
            .expect("soft affinity follows the dispatch")
            .expect("lease");
        assert_eq!(bound(&resolver, "s"), Some(target(2, Some(1))));
        drop(lease);
    }

    #[tokio::test]
    async fn query_returns_the_target_without_a_lease() {
        let (resolver, _) = make_resolver(SessionAffinityMode::Soft, &[1]);
        assert_eq!(resolver.query("s", None).expect("query"), None);
        bind(&resolver, "s", target(1, None)).await;
        assert_eq!(
            resolver.query("s", None).expect("query"),
            Some(AffinityRequirement::soft(target(1, None)))
        );
        assert_eq!(resolver.table().lease_count("s"), Some(0));
        assert!(matches!(
            resolver.query("s", Some(target(2, None))),
            Err(AffinityError::InvalidArgument(_))
        ));
    }

    #[tokio::test]
    async fn check_dispatch_rejects_only_a_live_hard_mismatch() {
        let (hard, liveness) = make_resolver(SessionAffinityMode::Hard, &[1, 2]);
        bind(&hard, "s", target(1, Some(0))).await;
        let held = hold(&hard, "s").await;
        assert!(hard.check_dispatch(&held, target(1, Some(0))).is_ok());
        assert!(matches!(
            hard.check_dispatch(&held, target(2, Some(0))),
            Err(AffinityError::InvalidArgument(_))
        ));
        assert_eq!(
            bound(&hard, "s"),
            Some(target(1, Some(0))),
            "the check changes nothing"
        );
        liveness.set(&[2]);
        assert!(
            hard.check_dispatch(&held, target(2, Some(0))).is_ok(),
            "a departed binding is left for commit to fail over"
        );
        drop(held);

        let (soft, _) = make_resolver(SessionAffinityMode::Soft, &[1, 2]);
        bind(&soft, "s", target(1, Some(0))).await;
        let held = hold(&soft, "s").await;
        assert!(soft.check_dispatch(&held, target(2, Some(0))).is_ok());
        let fresh = hold(&hard, "fresh").await;
        assert!(hard.check_dispatch(&fresh, target(2, Some(0))).is_ok());
    }

    #[tokio::test]
    async fn filtered_hard_binding_is_released_and_soft_binding_kept() {
        let (hard, _) = make_resolver(SessionAffinityMode::Hard, &[1]);
        bind(&hard, "s", target(1, None)).await;
        hard.release_filtered(hold(&hard, "s").await);
        assert_eq!(bound(&hard, "s"), None);

        let (soft, _) = make_resolver(SessionAffinityMode::Soft, &[1]);
        bind(&soft, "s", target(1, None)).await;
        soft.release_filtered(hold(&soft, "s").await);
        assert_eq!(bound(&soft, "s"), Some(target(1, None)));
    }
}
