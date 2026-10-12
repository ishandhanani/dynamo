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
