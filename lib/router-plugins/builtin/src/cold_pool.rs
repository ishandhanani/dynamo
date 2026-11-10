// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cold Pool (#12803) on the router's request classifier and dispatch gate.
//!
//! Very long cache-miss prefills (Cold) are contained on a small, stable worker set chosen by
//! rendezvous hashing. While Cold work is pending, Warm requests avoid that set; while Cold work
//! is only running, Warm requests avoid just the workers running it. A Cold request first waits
//! for a member with no active Warm work, and after `soft_drain_timeout_ms` may use any member.
//! The classifier narrows each request's allowed workers when it is classified; the gate only
//! decides when a Cold head may go, and onto which clean member.
//! Cold requests without a session wait in their own `cold_pool` lane, so a held Cold head never
//! blocks the class's shared lane, where sessionless Warm requests wait.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

use async_trait::async_trait;
use dynamo_kv_router::plugins::RouterPluginRegistry;
use dynamo_kv_router::plugins::request_classifier::{
    ClassifierError, ClassifyEvent, ClassifyRequest, DispatchDecision, DispatchGate, DispatchHead,
    DispatchRecheck, RequestClassifier, RequestClassifierContext, RequestClassifierFactory,
    RequestClassifierParameters, RequestClassifierProviderError, RequestClassifierRegistryError,
};
use dynamo_kv_router::plugins::worker_selection::{WorkerCandidates, WorkerInputs};
use dynamo_kv_router::protocols::{WorkerId, WorkerWithDpRank};
use parking_lot::Mutex;
use serde::Deserialize;
use tokio::time::Instant;

pub(crate) const COLD_POOL_CLASSIFIER_TYPE: &str = "cold_pool";
/// Lane for sessionless Cold requests.
const COLD_LANE: &str = "cold_pool";

/// Longest the gate waits between rechecks while Cold work is pending, matching #12806.
const RECHECK_INTERVAL: Duration = Duration::from_millis(100);
/// How often the pool logs its counters.
const STATS_INTERVAL: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColdPoolConfig {
    /// Effective-prefill token threshold: a request at or above it, after its best cached
    /// prefix, is Cold.
    pub request_threshold: usize,
    /// Requested pool size, capped at one third of the registered workers. `0` keeps
    /// classification and the Cold lane and disables containment, for a same-queue baseline.
    #[serde(default = "default_workers")]
    pub workers: usize,
    /// Longest a Cold request waits, from queue admission, for a member with no active Warm work.
    pub soft_drain_timeout_ms: u64,
}

fn default_workers() -> usize {
    1
}

impl ColdPoolConfig {
    fn validate(&self) -> Result<(), String> {
        if self.request_threshold == 0 {
            return Err("cold_pool.request_threshold must be positive".into());
        }
        Ok(())
    }

    fn soft_drain_timeout(&self) -> Duration {
        Duration::from_millis(self.soft_drain_timeout_ms)
    }

    /// Effective pool size for `worker_count` registered workers: none below two workers,
    /// else `workers` clamped to `[1, max(1, worker_count / 3)]`.
    fn effective_worker_count(&self, worker_count: usize) -> usize {
        if worker_count < 2 || self.workers == 0 {
            return 0;
        }
        let topology_cap = (worker_count / 3).max(1);
        self.workers.clamp(1, topology_cap)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lane {
    Warm,
    Cold,
    /// Long but cache-affine, or Cold with no eligible member: routed without the pool.
    Unmanaged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Queued,
    Dispatched,
    Active(WorkerWithDpRank),
}

#[derive(Debug)]
struct Tracked {
    lane: Lane,
    phase: Phase,
}

#[derive(Debug, Default, Clone, Copy)]
struct RankActivity {
    total: usize,
    cold: usize,
}

#[derive(Debug, Default)]
struct Stats {
    warm: u64,
    cold: u64,
    unmanaged: u64,
    cold_to_clean: u64,
    cold_after_drain: u64,
    cold_held: u64,
    warm_narrowed: u64,
}

struct State {
    config: ColdPoolConfig,
    /// Every registered worker, with the first rank seen for it.
    workers: HashMap<WorkerId, WorkerWithDpRank>,
    members: HashSet<WorkerId>,
    requests: HashMap<String, Tracked>,
    pending_cold: usize,
    active: HashMap<WorkerWithDpRank, RankActivity>,
    stats: Stats,
}

impl State {
    fn new(config: ColdPoolConfig) -> Self {
        Self {
            config,
            workers: HashMap::new(),
            members: HashSet::new(),
            requests: HashMap::new(),
            pending_cold: 0,
            active: HashMap::new(),
            stats: Stats::default(),
        }
    }

    fn reconcile_membership(&mut self, ranks: impl IntoIterator<Item = WorkerWithDpRank>) {
        self.workers.clear();
        for rank in ranks {
            self.workers
                .entry(rank.worker_id)
                .and_modify(|first| *first = (*first).min(rank))
                .or_insert(rank);
        }
        let mut worker_ids: Vec<_> = self.workers.keys().copied().collect();
        let effective = self.config.effective_worker_count(worker_ids.len());
        worker_ids.sort_unstable_by(|left, right| {
            rendezvous_score(*right)
                .cmp(&rendezvous_score(*left))
                .then_with(|| left.cmp(right))
        });
        let next: HashSet<_> = worker_ids.into_iter().take(effective).collect();
        if next != self.members {
            tracing::info!(target: "cold_pool", members = ?next, "Cold Pool membership changed");
            self.members = next;
        }
    }

    /// #12806 classification: short is Warm; long is Cold when its best effective prefill
    /// still reaches the threshold, else unmanaged.
    fn lane(&self, isl_tokens: usize, cached_tokens: usize) -> Lane {
        if isl_tokens < self.config.request_threshold {
            return Lane::Warm;
        }
        if isl_tokens.saturating_sub(cached_tokens) >= self.config.request_threshold {
            return Lane::Cold;
        }
        Lane::Unmanaged
    }

    fn active_warm(&self, worker: &WorkerWithDpRank) -> usize {
        self.active
            .get(worker)
            .map_or(0, |activity| activity.total.saturating_sub(activity.cold))
    }

    /// All members while Cold work is pending; otherwise the workers running Cold work.
    fn warm_avoid(&self) -> HashSet<WorkerId> {
        if self.pending_cold > 0 {
            return self.members.clone();
        }
        self.active
            .iter()
            .filter_map(|(worker, activity)| (activity.cold > 0).then_some(worker.worker_id))
            .collect()
    }

    /// Members with no rank running Warm work.
    fn clean_members(&self) -> HashSet<WorkerId> {
        let mut clean = self.members.clone();
        for (worker, _) in self.active.iter() {
            if self.active_warm(worker) > 0 {
                clean.remove(&worker.worker_id);
            }
        }
        clean
    }

    /// Warm's allowed workers among `eligible`: everything but the workers to avoid, or `None`
    /// when there is nothing to avoid or nothing would be left.
    fn warm_allowed(&self, eligible: &HashSet<WorkerId>) -> Option<HashSet<WorkerId>> {
        if self.members.is_empty() {
            return None;
        }
        let avoid = self.warm_avoid();
        if avoid.is_empty() {
            return None;
        }
        let preferred: HashSet<_> = eligible.difference(&avoid).copied().collect();
        (!preferred.is_empty()).then_some(preferred)
    }

    /// The clean member to send a Cold head to: the one running the fewest requests.
    fn clean_target(&self) -> Option<WorkerWithDpRank> {
        self.clean_members()
            .iter()
            .filter_map(|worker_id| self.workers.get(worker_id))
            .min_by_key(|rank| {
                let total = self
                    .active
                    .iter()
                    .filter(|(worker, _)| worker.worker_id == rank.worker_id)
                    .map(|(_, activity)| activity.total)
                    .sum::<usize>();
                (total, **rank)
            })
            .copied()
    }

    /// Only Cold heads wait: for a clean member, else until the soft-drain deadline, after which
    /// the selector picks among the members its allowed workers already name.
    fn check(&mut self, request_id: &str, queued_at: Instant, now: Instant) -> DispatchDecision {
        let go = DispatchDecision::Go { target: None };
        let Some(lane) = self.requests.get(request_id).map(|tracked| tracked.lane) else {
            return go;
        };
        if lane != Lane::Cold || self.members.is_empty() {
            return go;
        }
        if now.saturating_duration_since(queued_at) >= self.config.soft_drain_timeout() {
            return go;
        }
        match self.clean_target() {
            Some(target) => DispatchDecision::Go {
                target: Some(target),
            },
            None => DispatchDecision::Hold {
                until: Some(queued_at + self.config.soft_drain_timeout()),
            },
        }
    }

    fn dispatched(&mut self, request_id: &str, queued_at: Instant, now: Instant) {
        let pool_on = !self.members.is_empty();
        let expired = now.saturating_duration_since(queued_at) >= self.config.soft_drain_timeout();
        let clean_available = !self.clean_members().is_empty();
        let Some(tracked) = self.requests.get_mut(request_id) else {
            return;
        };
        if tracked.phase != Phase::Queued {
            return;
        }
        tracked.phase = Phase::Dispatched;
        if tracked.lane != Lane::Cold {
            return;
        }
        self.pending_cold = self.pending_cold.saturating_sub(1);
        if pool_on {
            if expired && !clean_available {
                self.stats.cold_after_drain += 1;
            } else {
                self.stats.cold_to_clean += 1;
            }
            if now.saturating_duration_since(queued_at) >= RECHECK_INTERVAL {
                self.stats.cold_held += 1;
            }
        }
    }

    /// Returns whether a worker's activity changed, so held heads should be asked again.
    fn on_event(&mut self, event: &ClassifyEvent) -> bool {
        match event {
            ClassifyEvent::Sent { request_id, worker } => {
                let Some(tracked) = self.requests.get_mut(request_id) else {
                    return false;
                };
                if tracked.phase == Phase::Queued && tracked.lane == Lane::Cold {
                    // Admitted without asking the gate (no gate installed yet, or a class
                    // that bypasses it).
                    self.pending_cold = self.pending_cold.saturating_sub(1);
                }
                if let Phase::Active(_) = tracked.phase {
                    return false;
                }
                tracked.phase = Phase::Active(*worker);
                let activity = self.active.entry(*worker).or_default();
                activity.total += 1;
                if tracked.lane == Lane::Cold {
                    activity.cold += 1;
                }
                false
            }
            ClassifyEvent::Completed { request_id, .. }
            | ClassifyEvent::Aborted { request_id, .. } => {
                let Some(tracked) = self.requests.remove(request_id) else {
                    return false;
                };
                match tracked.phase {
                    Phase::Queued if tracked.lane == Lane::Cold => {
                        self.pending_cold = self.pending_cold.saturating_sub(1);
                        true
                    }
                    Phase::Active(worker) => {
                        if let Some(activity) = self.active.get_mut(&worker) {
                            activity.total = activity.total.saturating_sub(1);
                            if tracked.lane == Lane::Cold {
                                activity.cold = activity.cold.saturating_sub(1);
                            }
                            if activity.total == 0 {
                                self.active.remove(&worker);
                            }
                        }
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }
}

/// SplitMix64 gives every worker ID a stable, well-distributed HRW score (same as #12806).
fn rendezvous_score(worker_id: WorkerId) -> u64 {
    let mut value = worker_id.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

struct Inner {
    state: Mutex<State>,
    context: Option<RequestClassifierContext>,
    recheck: OnceLock<DispatchRecheck>,
}

impl Inner {
    fn notify(&self) {
        if let Some(recheck) = self.recheck.get() {
            recheck.notify();
        }
    }

    fn reconcile(&self, state: &mut State) {
        if let Some(context) = &self.context {
            state.reconcile_membership(context.workers().iter().map(|info| info.worker()));
        }
    }
}

/// Wakes held heads every [`RECHECK_INTERVAL`] while Cold work is pending, so a soft-drain
/// expiry is seen even when no capacity changes; logs counters every [`STATS_INTERVAL`].
async fn tick(inner: Weak<Inner>) {
    let mut interval = tokio::time::interval(RECHECK_INTERVAL);
    let mut last_stats = Instant::now();
    loop {
        interval.tick().await;
        let Some(inner) = inner.upgrade() else {
            return;
        };
        let pending = {
            let mut state = inner.state.lock();
            inner.reconcile(&mut state);
            if last_stats.elapsed() >= STATS_INTERVAL {
                last_stats = Instant::now();
                let stats = &state.stats;
                tracing::info!(
                    target: "cold_pool",
                    members = ?state.members,
                    pending_cold = state.pending_cold,
                    tracked = state.requests.len(),
                    warm = stats.warm,
                    cold = stats.cold,
                    unmanaged = stats.unmanaged,
                    cold_to_clean = stats.cold_to_clean,
                    cold_after_drain = stats.cold_after_drain,
                    cold_held = stats.cold_held,
                    warm_narrowed = stats.warm_narrowed,
                    "Cold Pool stats"
                );
            }
            state.pending_cold
        };
        if pending > 0 {
            inner.notify();
        }
    }
}

pub struct ColdPoolClassifier {
    inner: Arc<Inner>,
}

impl ColdPoolClassifier {
    pub fn new(config: ColdPoolConfig, context: Option<RequestClassifierContext>) -> Self {
        tracing::info!(
            target: "cold_pool",
            request_threshold = config.request_threshold,
            workers = config.workers,
            soft_drain_timeout_ms = config.soft_drain_timeout_ms,
            "Cold Pool enabled"
        );
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(State::new(config)),
                context,
                recheck: OnceLock::new(),
            }),
        }
    }
}

#[async_trait]
impl RequestClassifier for ColdPoolClassifier {
    /// Rows list the request's eligible workers, so a Cold request with no eligible member
    /// bypasses the pool as in #12806.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::ACTIVE_REQUESTS
    }

    fn classify(
        &mut self,
        mut request: ClassifyRequest,
        workers: WorkerCandidates<'_>,
    ) -> Result<ClassifyRequest, Box<ClassifierError>> {
        let mut state = self.inner.state.lock();
        self.inner.reconcile(&mut state);
        let eligible: HashSet<WorkerId> = workers
            .iter()
            .map(|candidate| candidate.worker().worker_id)
            .collect();
        let mut lane = state.lane(request.isl_tokens(), request.cached_tokens());
        if lane == Lane::Cold && !state.members.is_empty() && eligible.is_disjoint(&state.members) {
            lane = Lane::Unmanaged;
        }
        match lane {
            Lane::Cold => {
                request.set_lane(COLD_LANE);
                if !state.members.is_empty() {
                    request.set_allowed_workers(state.members.clone());
                }
            }
            Lane::Warm => {
                if let Some(allowed) = state.warm_allowed(&eligible) {
                    request.set_allowed_workers(allowed);
                    state.stats.warm_narrowed += 1;
                }
            }
            Lane::Unmanaged => {}
        }
        match lane {
            Lane::Warm => state.stats.warm += 1,
            Lane::Cold => state.stats.cold += 1,
            Lane::Unmanaged => state.stats.unmanaged += 1,
        }
        let Some(request_id) = request.request_id().map(str::to_owned) else {
            return Ok(request);
        };
        if lane == Lane::Cold {
            state.pending_cold += 1;
        }
        if let Some(previous) = state.requests.insert(
            request_id,
            Tracked {
                lane,
                phase: Phase::Queued,
            },
        ) && previous.lane == Lane::Cold
            && previous.phase == Phase::Queued
        {
            state.pending_cold = state.pending_cold.saturating_sub(1);
        }
        Ok(request)
    }

    async fn on_event(&mut self, event: ClassifyEvent) {
        let changed = self.inner.state.lock().on_event(&event);
        if changed {
            self.inner.notify();
        }
    }

    fn dispatch_gate(&self) -> Option<Arc<dyn DispatchGate>> {
        Some(Arc::new(ColdPoolGate {
            inner: Arc::clone(&self.inner),
        }))
    }
}

struct ColdPoolGate {
    inner: Arc<Inner>,
}

impl DispatchGate for ColdPoolGate {
    fn check(&self, head: &DispatchHead<'_>) -> DispatchDecision {
        let Some(request_id) = head.request_id() else {
            return DispatchDecision::Go { target: None };
        };
        self.inner
            .state
            .lock()
            .check(request_id, head.queued_at(), Instant::now())
    }

    fn dispatched(&self, head: &DispatchHead<'_>, _target: Option<WorkerWithDpRank>) {
        let Some(request_id) = head.request_id() else {
            return;
        };
        self.inner
            .state
            .lock()
            .dispatched(request_id, head.queued_at(), Instant::now());
    }

    fn attach(&self, recheck: DispatchRecheck) {
        if self.inner.recheck.set(recheck).is_err() {
            return;
        }
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(tick(Arc::downgrade(&self.inner)));
        }
    }
}

fn classifier_provider(
    parameters: &RequestClassifierParameters,
) -> Result<RequestClassifierFactory, RequestClassifierProviderError> {
    let config: ColdPoolConfig = parameters.deserialize()?;
    config
        .validate()
        .map_err(RequestClassifierProviderError::new)?;
    Ok(Arc::new(move |context| {
        Box::new(ColdPoolClassifier::new(config.clone(), Some(context)))
    }))
}

pub(crate) fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), RequestClassifierRegistryError> {
    registry.register_request_classifier(COLD_POOL_CLASSIFIER_TYPE, Arc::new(classifier_provider))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(workers: usize) -> ColdPoolConfig {
        ColdPoolConfig {
            request_threshold: 128_000,
            workers,
            soft_drain_timeout_ms: 1_000,
        }
    }

    fn rank(worker_id: WorkerId) -> WorkerWithDpRank {
        WorkerWithDpRank::new(worker_id, 0)
    }

    fn state_with(workers: usize, ids: &[WorkerId]) -> State {
        let mut state = State::new(config(workers));
        state.reconcile_membership(ids.iter().copied().map(rank));
        state
    }

    fn track(state: &mut State, id: &str, lane: Lane) {
        if lane == Lane::Cold {
            state.pending_cold += 1;
        }
        state.requests.insert(
            id.into(),
            Tracked {
                lane,
                phase: Phase::Queued,
            },
        );
    }

    fn sent(state: &mut State, id: &str, worker: WorkerId) {
        state.on_event(&ClassifyEvent::Sent {
            request_id: id.into(),
            worker: rank(worker),
        });
    }

    fn completed(state: &mut State, id: &str, worker: WorkerId) {
        state.on_event(&ClassifyEvent::Completed {
            request_id: id.into(),
            worker: rank(worker),
            context_tokens: None,
        });
    }

    fn all(state: &State) -> HashSet<WorkerId> {
        state.workers.keys().copied().collect()
    }

    #[test]
    fn membership_is_stable_and_bounded_by_topology() {
        let mut state = state_with(10, &[1, 2, 3, 4]);
        assert_eq!(state.members.len(), 1);
        let first = state.members.clone();
        state.reconcile_membership([4, 3, 2, 1].map(rank));
        assert_eq!(state.members, first);
        state.reconcile_membership([rank(1)]);
        assert!(state.members.is_empty());

        let expected = [
            (0, 0),
            (1, 0),
            (2, 1),
            (3, 1),
            (5, 1),
            (6, 2),
            (8, 2),
            (9, 3),
        ];
        for (worker_count, cold_count) in expected {
            assert_eq!(config(10).effective_worker_count(worker_count), cold_count);
        }
        assert_eq!(config(0).effective_worker_count(9), 0);
    }

    #[test]
    fn classification_uses_effective_prefill() {
        let state = state_with(1, &[1, 2]);
        assert_eq!(state.lane(127_999, 0), Lane::Warm);
        assert_eq!(state.lane(128_000, 0), Lane::Cold);
        assert_eq!(state.lane(400_000, 272_000), Lane::Cold);
        assert_eq!(state.lane(400_000, 272_001), Lane::Unmanaged);
    }

    #[test]
    fn warm_avoids_all_members_while_cold_is_pending_then_only_busy_ones() {
        let mut state = state_with(2, &[1, 2, 3, 4, 5, 6]);
        let members = state.members.clone();
        let now = Instant::now();
        assert_eq!(state.warm_allowed(&all(&state)), None);

        track(&mut state, "c", Lane::Cold);
        let allowed = state.warm_allowed(&all(&state)).unwrap();
        assert!(allowed.is_disjoint(&members));
        assert_eq!(allowed.len(), 4);
        // Nothing left once the avoid set covers every eligible worker: no narrowing.
        assert_eq!(state.warm_allowed(&members), None);

        // Cold runs on one member: Warm avoids only that one.
        let cold_worker = *members.iter().next().unwrap();
        state.dispatched("c", now, now);
        sent(&mut state, "c", cold_worker);
        let allowed = state.warm_allowed(&all(&state)).unwrap();
        assert_eq!(allowed.len(), 5);
        assert!(!allowed.contains(&cold_worker));

        completed(&mut state, "c", cold_worker);
        assert_eq!(state.warm_allowed(&all(&state)), None);
        // Warm heads are never held.
        track(&mut state, "w", Lane::Warm);
        assert_eq!(
            state.check("w", now, now),
            DispatchDecision::Go { target: None }
        );
    }

    #[test]
    fn cold_targets_a_clean_member_then_holds_then_drains() {
        let mut state = state_with(2, &[1, 2, 3, 4, 5, 6]);
        let mut members: Vec<_> = state.members.iter().copied().collect();
        members.sort();
        let queued_at = Instant::now();
        let target = |state: &mut State, now| match state.check("c", queued_at, now) {
            DispatchDecision::Go { target } => target.map(|rank| rank.worker_id),
            hold => panic!("expected Go, got {hold:?}"),
        };
        track(&mut state, "c", Lane::Cold);
        assert!(members.contains(&target(&mut state, queued_at).unwrap()));

        // Warm work on one member leaves the other clean.
        track(&mut state, "w1", Lane::Warm);
        sent(&mut state, "w1", members[0]);
        assert_eq!(target(&mut state, queued_at), Some(members[1]));

        // Warm on both members: hold until the drain deadline.
        track(&mut state, "w2", Lane::Warm);
        sent(&mut state, "w2", members[1]);
        assert_eq!(
            state.check("c", queued_at, queued_at),
            DispatchDecision::Hold {
                until: Some(queued_at + Duration::from_secs(1))
            }
        );

        // After the timeout, no target: the selector picks among the allowed members.
        let later = queued_at + Duration::from_secs(1);
        assert_eq!(target(&mut state, later), None);

        // A Warm completion cleans its member again.
        completed(&mut state, "w1", members[0]);
        assert_eq!(target(&mut state, queued_at), Some(members[0]));
    }

    #[test]
    fn pending_cold_settles_on_dispatch_and_on_abort_in_queue() {
        let mut state = state_with(1, &[1, 2, 3]);
        let now = Instant::now();
        track(&mut state, "a", Lane::Cold);
        track(&mut state, "b", Lane::Cold);
        assert_eq!(state.pending_cold, 2);
        state.dispatched("a", now, now);
        assert_eq!(state.pending_cold, 1);
        state.on_event(&ClassifyEvent::Aborted {
            request_id: "b".into(),
            worker: None,
            error: None,
        });
        assert_eq!(state.pending_cold, 0);
        assert!(state.warm_avoid().is_empty());
    }

    #[test]
    fn disabled_pool_never_narrows() {
        let mut state = state_with(0, &[1, 2, 3, 4, 5, 6]);
        let now = Instant::now();
        track(&mut state, "c", Lane::Cold);
        track(&mut state, "w", Lane::Warm);
        assert_eq!(
            state.check("c", now, now),
            DispatchDecision::Go { target: None }
        );
        assert_eq!(
            state.check("w", now, now),
            DispatchDecision::Go { target: None }
        );
    }

    #[test]
    fn classify_bypasses_the_pool_without_an_eligible_member() {
        use dynamo_kv_router::plugins::worker_selection::{WorkerCapacity, WorkerInfo};
        let context = RequestClassifierContext::new(16, || {
            (1..=3)
                .map(|id| WorkerInfo::new(rank(id), WorkerCapacity::default()))
                .collect()
        });
        let mut classifier = ColdPoolClassifier::new(config(1), Some(context));
        let request = ClassifyRequest::for_test(200_000, Some("c"), None);
        // No rows: no eligible member, so the long cache miss bypasses the pool.
        classifier
            .classify(request, WorkerCandidates::empty())
            .unwrap();
        let state = classifier.inner.state.lock();
        assert_eq!(state.pending_cold, 0);
        assert_eq!(state.requests["c"].lane, Lane::Unmanaged);
    }
}
