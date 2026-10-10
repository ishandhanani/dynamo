// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How a routing host drives a [`SessionAffinity`] table for one request.
//!
//! `resolve` runs before worker selection and yields the [`Hold`] to commit.
//! Its bound target supplies the [`AffinityRequirement`] the scheduler enforces. `commit_or_failover` runs
//! after booking for a host that owns the booking. A frontend serializes failover
//! before dispatch and uses `commit` after obtaining the response stream.
//! `query_target` serves selections that must not bind anything.

use std::future::Future;

use super::{
    AcquireStep, AffinityError, AffinityLease, AffinityTarget, Hold, SessionAffinity,
    SessionAffinityMode, validate_bound_target, validate_dispatch_target,
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

impl SessionAffinity {
    pub fn requirement(&self, target: AffinityTarget) -> AffinityRequirement {
        AffinityRequirement {
            target,
            mode: self.mode(),
        }
    }

    /// Hold `session_id` for a request about to select a worker.
    ///
    /// A binding whose target the host can no longer schedule is dropped and
    /// the session re-initialized, so the request selects normally and
    /// `commit` binds the worker it lands on. `requested` is an explicit
    /// target the request carries; a bound session must agree with it. A
    /// full table is a router-side limit, not a client fault: the request
    /// routes unpinned. `cancel` abandons a wait
    /// on another request's initialization, or the yield after a burst of
    /// invalidations; an acquisition that completes at once is not checked.
    pub async fn resolve(
        &self,
        session_id: &str,
        requested: Option<AffinityTarget>,
        is_live: impl Fn(AffinityTarget) -> bool,
        cancel: impl Future<Output = ()>,
    ) -> Result<Option<Hold>, AffinityError> {
        tokio::pin!(cancel);
        let mut invalidations = 0;
        loop {
            let hold = match self.try_acquire(session_id, None) {
                Ok(AcquireStep::Held(hold)) => hold,
                Ok(AcquireStep::Wait(notified)) => {
                    tokio::select! {
                        biased;
                        _ = &mut cancel => return Err(AffinityError::Cancelled),
                        _ = notified => continue,
                    }
                }
                Err(AffinityError::ResourceExhausted(_)) => {
                    tracing::debug!(
                        session_id,
                        "Affinity table full; routing without session affinity"
                    );
                    return Ok(None);
                }
                Err(error) => return Err(error),
            };
            match hold {
                Hold::Bound { target, mut lease } if !is_live(target) => {
                    tracing::debug!(
                        session_id,
                        worker_id = target.worker_id,
                        dp_rank = ?target.dp_rank,
                        "Session affinity target is not schedulable; re-initializing"
                    );
                    lease.invalidate();
                    drop(lease);
                    #[cfg(any(test, feature = "testing"))]
                    if let Some(hook) = self.inner.after_invalidation.get() {
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
                Hold::Initialize(mut initialization) => {
                    initialization.requested_target = requested;
                    return Ok(Some(Hold::Initialize(initialization)));
                }
                hold @ Hold::Bound { target, .. } => {
                    validate_bound_target(session_id, target, requested)?;
                    return Ok(Some(hold));
                }
            }
        }
    }

    /// Bind the held session to `dispatched`. In `Hard` mode a dispatch away
    /// from a live binding is rejected and the binding dropped. A rejection
    /// whose bound worker departed after `resolve` checked it is not a client
    /// fault: the session is re-bound to `dispatched`, joining another
    /// request's initialization when one is in flight. `None` means the
    /// request runs without a lease (the table filled up during failover).
    pub fn commit_or_failover(
        &self,
        hold: Hold,
        dispatched: AffinityTarget,
        is_live: impl Fn(AffinityTarget) -> bool,
    ) -> Result<Option<AffinityLease>, AffinityError> {
        let bound = hold.target();
        let session_id = hold.shared_session_id();
        let error = match self.commit(hold, dispatched) {
            Ok(lease) => return Ok(Some(lease)),
            Err(error) => error,
        };
        let departed = bound.is_some_and(|target| !is_live(target));
        if !departed {
            return Err(error);
        }
        // `commit` invalidated the stale binding. Another request may already
        // be initializing the replacement: join it rather than wait, binding
        // it to this successful dispatch's target.
        for _ in 0..JOIN_ATTEMPTS {
            match self.try_acquire(&session_id, None) {
                Ok(AcquireStep::Held(hold)) => {
                    if self.mode() == SessionAffinityMode::Hard
                        && let Some(target) = hold.target()
                    {
                        // A competing failover already bound a valid target;
                        // reject a mismatch without erasing it.
                        validate_dispatch_target(&session_id, target, dispatched)?;
                    }
                    return self.commit(hold, dispatched).map(Some);
                }
                Ok(AcquireStep::Wait(_)) => {
                    if let Some(lease) = self.join_initializing(&session_id, dispatched) {
                        return Ok(Some(lease));
                    }
                }
                Err(AffinityError::ResourceExhausted(_)) => {
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
        is_live: impl Fn(AffinityTarget) -> bool,
    ) -> Result<(), AffinityError> {
        if self.mode() != SessionAffinityMode::Hard {
            return Ok(());
        }
        let Some(target) = hold.target() else {
            return Ok(());
        };
        if !is_live(target) {
            // The frontend must release its booking and reacquire this session
            // before dispatching elsewhere; service bookings use `commit_or_failover`.
            return Ok(());
        }
        validate_dispatch_target(hold.session_id(), target, dispatched)
    }

    /// Give up a hold after the scheduler reports `HardAffinityTargetFiltered`:
    /// selection was limited to the bound target and the policy rejected it.
    /// Drop the binding so the session
    /// re-binds instead of failing on every retry. A `Soft` binding is kept.
    pub fn release_filtered(&self, hold: Hold) {
        if self.mode() == SessionAffinityMode::Hard {
            hold.invalidate();
        }
    }

    /// Run `hook` after each invalidation of an unschedulable binding.
    #[cfg(any(test, feature = "testing"))]
    pub fn set_after_invalidation(&self, hook: Box<dyn Fn() + Send + Sync>) {
        assert!(
            self.inner.after_invalidation.set(hook).is_ok(),
            "after-invalidation hook already set"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use parking_lot::Mutex;
    use tokio_util::sync::CancellationToken;

    use super::super::{AffinityVersion, SessionAffinityConfig};
    use super::*;
    use crate::protocols::WorkerId;
    use std::sync::Arc;

    const TTL: Duration = Duration::from_secs(10);

    struct Liveness(Mutex<HashSet<WorkerId>>);

    impl Liveness {
        fn set(&self, live: &[WorkerId]) {
            *self.0.lock() = live.iter().copied().collect();
        }
    }

    impl Liveness {
        fn is_schedulable(&self, target: AffinityTarget) -> bool {
            self.0.lock().contains(&target.worker_id)
        }
    }

    fn make_table_with(
        config: SessionAffinityConfig,
        live: &[WorkerId],
    ) -> (SessionAffinity, Arc<Liveness>) {
        let liveness = Arc::new(Liveness(Mutex::new(live.iter().copied().collect())));
        let table = SessionAffinity::with_config(config).expect("affinity table");
        (table, liveness)
    }

    fn make_table(
        mode: SessionAffinityMode,
        live: &[WorkerId],
    ) -> (SessionAffinity, Arc<Liveness>) {
        make_table_with(SessionAffinityConfig::new(TTL).with_mode(mode), live)
    }

    fn target(worker_id: WorkerId, dp_rank: Option<u32>) -> AffinityTarget {
        AffinityTarget::new(worker_id, dp_rank)
    }

    fn never() -> std::future::Pending<()> {
        std::future::pending()
    }

    async fn hold(table: &SessionAffinity, liveness: &Liveness, session_id: &str) -> Hold {
        table
            .resolve(
                session_id,
                None,
                |target| liveness.is_schedulable(target),
                never(),
            )
            .await
            .expect("resolve")
            .expect("table has room")
    }

    /// Bind `session_id` to `target` and release the lease.
    async fn bind(table: &SessionAffinity, session_id: &str, target: AffinityTarget) {
        let hold = table
            .resolve(session_id, None, |_| true, never())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(hold, Hold::Initialize(_)), "session must be new");
        drop(
            table
                .commit_or_failover(hold, target, |_| true)
                .expect("commit")
                .expect("lease"),
        );
    }

    fn bound(table: &SessionAffinity, session_id: &str) -> Option<AffinityTarget> {
        table.query_target(session_id, None).expect("query")
    }

    #[tokio::test]
    async fn new_session_resolves_to_an_initializing_hold() {
        let (table, liveness) = make_table(SessionAffinityMode::Hard, &[1]);
        let resolution = table
            .resolve("s", None, |target| liveness.is_schedulable(target), never())
            .await
            .expect("resolve");
        assert!(matches!(resolution, Some(Hold::Initialize(_))));
        assert!(resolution.as_ref().and_then(Hold::target).is_none());
        assert_eq!(resolution.as_ref().map(Hold::session_id), Some("s"));
    }

    #[tokio::test]
    async fn bound_live_session_resolves_to_its_target_with_the_table_mode() {
        for mode in [SessionAffinityMode::Hard, SessionAffinityMode::Soft] {
            let (table, liveness) = make_table(mode, &[1, 2]);
            bind(&table, "s", target(1, Some(0))).await;
            let resolution = table
                .resolve("s", None, |target| liveness.is_schedulable(target), never())
                .await
                .expect("resolve");
            assert!(matches!(resolution, Some(Hold::Bound { .. })));
            assert_eq!(
                resolution
                    .as_ref()
                    .and_then(Hold::target)
                    .map(|target| table.requirement(target)),
                Some(AffinityRequirement {
                    target: target(1, Some(0)),
                    mode
                })
            );
            assert_eq!(table.lease_count("s"), Some(1));
        }
    }

    #[tokio::test]
    async fn bound_unschedulable_target_is_invalidated_and_reinitialized() {
        let (table, liveness) = make_table(SessionAffinityMode::Hard, &[1, 2]);
        bind(&table, "s", target(1, Some(0))).await;
        liveness.set(&[2]);
        let resolution = table
            .resolve("s", None, |target| liveness.is_schedulable(target), never())
            .await
            .expect("resolve");
        assert!(matches!(resolution, Some(Hold::Initialize(_))));
        assert!(resolution.as_ref().and_then(Hold::target).is_none());
        assert_eq!(bound(&table, "s"), None, "the stale binding is dropped");
        let lease = table
            .commit_or_failover(resolution.unwrap(), target(2, Some(0)), |target| {
                liveness.is_schedulable(target)
            })
            .expect("commit")
            .expect("lease");
        assert_eq!(bound(&table, "s"), Some(target(2, Some(0))));
        drop(lease);
    }

    #[tokio::test]
    async fn repeated_invalidation_yields_and_preserves_new_bindings() {
        for cancel in [false, true] {
            let (table, liveness) = make_table(SessionAffinityMode::Hard, &[1]);
            let table = table.clone();
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
            table.set_after_invalidation(Box::new({
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
            let mut pending = Box::pin(table.resolve(
                "s",
                None,
                |target| liveness.is_schedulable(target),
                token.cancelled(),
            ));
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
                    matches!(resolution, Some(Hold::Bound { target, .. }) if target == healthy)
                );
                assert_eq!(
                    resolution
                        .as_ref()
                        .and_then(Hold::target)
                        .map(|target| table.requirement(target)),
                    Some(AffinityRequirement::hard(healthy))
                );
                assert_eq!(invalidations.load(Ordering::SeqCst), 32);
            }
        }
    }

    #[tokio::test]
    async fn cancellation_stops_waiting_on_an_initializing_entry() {
        let (table, liveness) = make_table(SessionAffinityMode::Hard, &[1]);
        let _initializing = hold(&table, &liveness, "s").await;
        let token = CancellationToken::new();
        let mut waiting = Box::pin(table.resolve(
            "s",
            None,
            |target| liveness.is_schedulable(target),
            token.cancelled(),
        ));
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(waiting.as_mut().poll(&mut context).is_pending());
        token.cancel();
        assert!(matches!(waiting.await, Err(AffinityError::Cancelled)));
        assert_eq!(table.entry_count(), 1, "the initializer keeps its slot");
    }

    #[tokio::test]
    async fn explicit_target_must_agree_with_a_live_binding() {
        let (table, liveness) = make_table(SessionAffinityMode::Hard, &[1, 2]);
        bind(&table, "s", target(1, Some(0))).await;

        let Err(error) = table
            .resolve(
                "s",
                Some(target(2, None)),
                |target| liveness.is_schedulable(target),
                never(),
            )
            .await
        else {
            panic!("a disagreeing target is rejected");
        };
        assert!(matches!(error, AffinityError::InvalidArgument(_)));
        assert_eq!(
            bound(&table, "s"),
            Some(target(1, Some(0))),
            "rejection keeps the binding"
        );

        let resolution = table
            .resolve(
                "s",
                Some(target(1, None)),
                |target| liveness.is_schedulable(target),
                never(),
            )
            .await
            .expect("an agreeing target binds");
        assert!(matches!(resolution, Some(Hold::Bound { .. })));

        // A new session with an explicit target binds only to that target.
        let pinned = table
            .resolve(
                "n",
                Some(target(2, Some(0))),
                |target| liveness.is_schedulable(target),
                never(),
            )
            .await
            .expect("resolve")
            .expect("hold");
        assert!(matches!(
            table.commit_or_failover(pinned, target(1, Some(0)), |target| liveness
                .is_schedulable(target)),
            Err(AffinityError::InvalidArgument(_))
        ));
        assert_eq!(bound(&table, "n"), None);
        let hold = hold(&table, &liveness, "n").await;
        drop(
            table
                .commit_or_failover(hold, target(2, Some(0)), |target| {
                    liveness.is_schedulable(target)
                })
                .expect("commit")
                .expect("lease"),
        );
        assert_eq!(bound(&table, "n"), Some(target(2, Some(0))));
    }

    #[tokio::test]
    async fn explicit_target_can_replace_a_departed_binding() {
        let (table, liveness) = make_table(SessionAffinityMode::Hard, &[1, 2]);
        bind(&table, "s", target(1, Some(0))).await;
        liveness.set(&[2]);
        let held = table
            .resolve(
                "s",
                Some(target(2, Some(0))),
                |target| liveness.is_schedulable(target),
                never(),
            )
            .await
            .expect("a departed binding must not reject the new target")
            .expect("hold");
        assert!(matches!(held, Hold::Initialize(_)));
        let lease = table
            .commit_or_failover(held, target(2, Some(0)), |target| {
                liveness.is_schedulable(target)
            })
            .expect("commit");
        assert_eq!(bound(&table, "s"), Some(target(2, Some(0))));
        drop(lease);
    }

    #[tokio::test]
    async fn full_table_routes_unpinned() {
        let (table, liveness) = make_table_with(
            SessionAffinityConfig {
                max_entries: 1,
                ..SessionAffinityConfig::new(TTL)
            },
            &[1],
        );
        let _first = hold(&table, &liveness, "a").await;
        let resolution = table
            .resolve("b", None, |target| liveness.is_schedulable(target), never())
            .await
            .expect("resolve");
        assert!(resolution.is_none());
        assert!(resolution.as_ref().and_then(Hold::target).is_none());
        assert_eq!(table.entry_count(), 1);
    }

    #[tokio::test]
    async fn hard_commit_away_from_a_live_binding_rejects_and_invalidates() {
        let (table, liveness) = make_table(SessionAffinityMode::Hard, &[1, 2]);
        bind(&table, "s", target(1, Some(0))).await;
        let hold = hold(&table, &liveness, "s").await;
        let Err(error) = table.commit_or_failover(hold, target(2, Some(0)), |target| {
            liveness.is_schedulable(target)
        }) else {
            panic!("hard affinity rejects a dispatch away from a live binding");
        };
        assert!(matches!(error, AffinityError::InvalidArgument(_)));
        assert_eq!(bound(&table, "s"), None, "the binding is dropped");
    }

    #[tokio::test]
    async fn departed_binding_fails_over_to_the_dispatched_worker() {
        let (table, liveness) = make_table(SessionAffinityMode::Hard, &[1, 2]);
        bind(&table, "s", target(1, Some(0))).await;
        let hold = hold(&table, &liveness, "s").await;
        liveness.set(&[2]);
        let lease = table
            .commit_or_failover(hold, target(2, Some(0)), |target| {
                liveness.is_schedulable(target)
            })
            .expect("a departure after the hold is not a client fault")
            .expect("lease");
        assert_eq!(bound(&table, "s"), Some(target(2, Some(0))));
        assert_eq!(table.lease_count("s"), Some(1));
        drop(lease);
        assert_eq!(table.lease_count("s"), Some(0));
    }

    #[tokio::test]
    async fn departed_binding_joins_a_pending_initialization() {
        let (table, liveness) = make_table(SessionAffinityMode::Hard, &[1, 2]);
        bind(&table, "s", target(1, Some(0))).await;
        let stale = hold(&table, &liveness, "s").await;
        liveness.set(&[2]);
        // A later request sees the departure and re-initializes the session.
        let initializing = hold(&table, &liveness, "s").await;
        assert!(matches!(initializing, Hold::Initialize(_)));

        let joined = table
            .commit_or_failover(stale, target(2, Some(0)), |target| {
                liveness.is_schedulable(target)
            })
            .expect("failover joins the initialization")
            .expect("lease");
        assert_eq!(bound(&table, "s"), Some(target(2, Some(0))));
        assert_eq!(table.lease_count("s"), Some(2));

        // The initializer commits to the same target and keeps its own lease.
        let lease = table
            .commit_or_failover(initializing, target(2, Some(0)), |target| {
                liveness.is_schedulable(target)
            })
            .expect("commit")
            .expect("lease");
        assert_eq!(table.lease_count("s"), Some(2));
        drop(joined);
        drop(lease);
        assert_eq!(table.lease_count("s"), Some(0));
    }

    #[tokio::test]
    async fn late_failover_mismatch_preserves_the_committed_replacement() {
        let (table, liveness) = make_table(SessionAffinityMode::Hard, &[1, 2, 3]);
        bind(&table, "s", target(1, Some(0))).await;
        let old_hold = hold(&table, &liveness, "s").await;
        liveness.set(&[2, 3]);
        let new_hold = hold(&table, &liveness, "s").await;
        let lease = table
            .commit_or_failover(new_hold, target(2, Some(0)), |target| {
                liveness.is_schedulable(target)
            })
            .expect("bind replacement")
            .expect("lease");
        let result = table.commit_or_failover(old_hold, target(3, Some(0)), |target| {
            liveness.is_schedulable(target)
        });
        assert!(matches!(result, Err(AffinityError::InvalidArgument(_))));
        assert_eq!(bound(&table, "s"), Some(target(2, Some(0))));
        assert_eq!(table.lease_count("s"), Some(1));
        drop(lease);
        assert_eq!(table.lease_count("s"), Some(0));
    }

    #[tokio::test]
    async fn soft_commit_follows_the_dispatch() {
        let (table, liveness) = make_table(SessionAffinityMode::Soft, &[1, 2]);
        bind(&table, "s", target(1, Some(0))).await;
        let hold = hold(&table, &liveness, "s").await;
        let lease = table
            .commit_or_failover(hold, target(2, Some(1)), |target| {
                liveness.is_schedulable(target)
            })
            .expect("soft affinity follows the dispatch")
            .expect("lease");
        assert_eq!(bound(&table, "s"), Some(target(2, Some(1))));
        drop(lease);
    }

    #[tokio::test]
    async fn query_returns_the_target_without_a_lease() {
        let (table, _) = make_table(SessionAffinityMode::Soft, &[1]);
        assert_eq!(table.query_target("s", None).expect("query"), None);
        bind(&table, "s", target(1, None)).await;
        assert_eq!(
            table.query_target("s", None).expect("query"),
            Some(target(1, None))
        );
        assert_eq!(table.lease_count("s"), Some(0));
        assert!(matches!(
            table.query_target("s", Some(target(2, None))),
            Err(AffinityError::InvalidArgument(_))
        ));
    }

    #[tokio::test]
    async fn check_dispatch_rejects_only_a_live_hard_mismatch() {
        let (hard, liveness) = make_table(SessionAffinityMode::Hard, &[1, 2]);
        bind(&hard, "s", target(1, Some(0))).await;
        let held = hold(&hard, &liveness, "s").await;
        assert!(
            hard.check_dispatch(&held, target(1, Some(0)), |target| liveness
                .is_schedulable(target))
                .is_ok()
        );
        assert!(matches!(
            hard.check_dispatch(&held, target(2, Some(0)), |target| liveness
                .is_schedulable(target)),
            Err(AffinityError::InvalidArgument(_))
        ));
        assert_eq!(
            bound(&hard, "s"),
            Some(target(1, Some(0))),
            "the check changes nothing"
        );
        liveness.set(&[2]);
        assert!(
            hard.check_dispatch(&held, target(2, Some(0)), |target| liveness
                .is_schedulable(target))
                .is_ok(),
            "a departed binding is left for commit to fail over"
        );
        drop(held);

        let (soft, liveness) = make_table(SessionAffinityMode::Soft, &[1, 2]);
        bind(&soft, "s", target(1, Some(0))).await;
        let held = hold(&soft, &liveness, "s").await;
        assert!(
            soft.check_dispatch(&held, target(2, Some(0)), |target| liveness
                .is_schedulable(target))
                .is_ok()
        );
        let fresh = hold(&hard, &liveness, "fresh").await;
        assert!(
            hard.check_dispatch(&fresh, target(2, Some(0)), |target| liveness
                .is_schedulable(target))
                .is_ok()
        );
    }

    #[tokio::test]
    async fn filtered_hard_binding_is_released_and_soft_binding_kept() {
        let (hard, liveness) = make_table(SessionAffinityMode::Hard, &[1]);
        bind(&hard, "s", target(1, None)).await;
        hard.release_filtered(hold(&hard, &liveness, "s").await);
        assert_eq!(bound(&hard, "s"), None);

        let (soft, liveness) = make_table(SessionAffinityMode::Soft, &[1]);
        bind(&soft, "s", target(1, None)).await;
        soft.release_filtered(hold(&soft, &liveness, "s").await);
        assert_eq!(bound(&soft, "s"), Some(target(1, None)));
    }
}
