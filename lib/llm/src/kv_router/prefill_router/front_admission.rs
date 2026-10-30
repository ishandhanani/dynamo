// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Proof of concept: one admission queue in front of both disaggregated pools.
//!
//! A request first prepares both legs on its own task: the prefill and decode selections
//! run their index lookups, classification, candidate rows and planning setup, and each
//! stops at its queue-actor step, which it hands to this stage through a
//! [`ScheduleRelay`]. Once both legs have handed over, the request joins one policy-class
//! queue (DRR across classes). As on main, where a request enters a pool's queue only
//! after its selection inputs are prepared, queue order is the order in which requests
//! became ready, and no request waits on another's preparation. A request whose leg fails
//! first never joins. The serializer pops a request when a prefill worker has room under
//! its class prefill limit and a decode worker has room under its class decode limit. It
//! then runs the prefill booking and the decode booking, in that order, before it pops
//! the next request, so every admission sees every earlier booking in both pools.
//! Each booking runs on the serializer's task under its pool queue actor's lock
//! (`DYN_ROUTER_FRONT_ADMISSION_DIRECT_BOOK=0` sends it to the actor instead). The two booked [`RoutePlan`]s go back to the caller, which attaches them
//! to the request; the prefill hop and the decode host send them without selecting or
//! queueing again.
//!
//! Enabled by `DYN_ROUTER_FRONT_ADMISSION=1` (KV routing on both hops only).
//! Not covered: deadlines, session-affinity-aware room checks, per-request decode
//! eligibility in the room check, and prefill pool succession beyond rebuilding the
//! stage when the prefill binding changes. A decode leg whose prefill worker imposes
//! KV-transfer constraints is re-planned after admission, outside the serialized step.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use dynamo_kv_router::{
    protocols::RoutingConstraints,
    scheduling::{
        BusyThresholds, KvSchedulerError, PolicyClassConfig, PolicyProfile, PolicyQueue,
        QueueSnapshot, WorkerPlacement,
    },
    services::selection::{ScheduleRelay, ScheduleSubmit, ScheduledBooking},
};
use dynamo_runtime::pipeline::{Context, SingleIn};
use parking_lot::Mutex;
use tokio::sync::{Notify, mpsc, oneshot, watch};

use crate::{
    kv_router::{KvRouter, RoutingHost, routing_host::RoutePlan},
    protocols::common::{
        extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
        llm_backend::PreprocessedRequest,
        timing::RequestPhase,
    },
};

/// Environment switch for the proof of concept.
pub(crate) const FRONT_ADMISSION_ENV: &str = "DYN_ROUTER_FRONT_ADMISSION";
const COMMAND_CAPACITY: usize = 65_536;
const STATS_INTERVAL: Duration = Duration::from_secs(5);
/// Safety net while requests wait on room: recheck even if no room signal arrives.
const BLOCKED_RECHECK: Duration = Duration::from_millis(20);
const DECODE_NEEDS_CONSTRAINTS: &str =
    "front admission: decode leg needs the prefill worker's KV-transfer constraints";

pub(crate) fn front_admission_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| dynamo_runtime::config::env_is_truthy(FRONT_ADMISSION_ENV))
}

/// Derives the decode leg's routing constraints from the prefill worker the
/// serializer booked (its KV-transfer domain).
pub(crate) type DecodeConstraintsFn =
    Arc<dyn Fn(u64) -> Result<Option<RoutingConstraints>> + Send + Sync + 'static>;

/// What one admission needs: the two planning requests, built by the caller.
pub(crate) struct FrontAdmissionRequest {
    pub(crate) request_id: String,
    pub(crate) metadata: BTreeMap<String, String>,
    pub(crate) session_affinity: Option<SessionAffinityId>,
    pub(crate) policy_class: Option<String>,
    /// The prefill leg as the prefill host will see it (`max_tokens = 1`).
    pub(crate) prefill: PreprocessedRequest,
    /// The decode leg's routing view: the decode router override applied, no
    /// handoff fields yet. Only used to select and book the decode worker.
    pub(crate) decode: PreprocessedRequest,
    pub(crate) decode_constraints: DecodeConstraintsFn,
}

#[derive(Clone, Copy)]
enum Leg {
    Prefill,
    Decode,
}

struct LegStep {
    submit: ScheduleSubmit,
    reply: oneshot::Sender<ScheduledBooking>,
}

#[derive(Default)]
struct SlotState {
    prefill: Option<LegStep>,
    decode: Option<LegStep>,
    /// A leg's selection ended without handing over its actor step.
    leg_failed: bool,
    /// The caller stopped waiting.
    abandoned: bool,
    /// No booking will run for this admission; a leg handed over now fails at once.
    closed: bool,
}

/// One admission's meeting point between the request task and the serializer.
#[derive(Default)]
struct AdmissionSlot {
    state: Mutex<SlotState>,
    /// Wakes the request task when a leg hands over its step or fails.
    changed: Notify,
}

enum SlotReadiness {
    Waiting,
    Ready,
    Discard,
}

impl AdmissionSlot {
    fn readiness(&self) -> SlotReadiness {
        let state = self.state.lock();
        if state.abandoned || state.leg_failed {
            SlotReadiness::Discard
        } else if state.prefill.is_some() && state.decode.is_some() {
            SlotReadiness::Ready
        } else {
            SlotReadiness::Waiting
        }
    }

    /// Wait until both legs have handed over their steps (`true`) or a leg failed
    /// (`false`).
    async fn wait_ready(&self) -> bool {
        loop {
            match self.readiness() {
                SlotReadiness::Ready => return true,
                SlotReadiness::Discard => return false,
                SlotReadiness::Waiting => self.changed.notified().await,
            }
        }
    }

    fn fail_leg(&self) {
        self.state.lock().leg_failed = true;
        self.changed.notify_one();
    }

    /// Take both steps and close the slot.
    fn take(&self) -> (Option<LegStep>, Option<LegStep>) {
        let mut state = self.state.lock();
        state.closed = true;
        (state.prefill.take(), state.decode.take())
    }
}

/// Hands one leg's queue-actor step to the serializer and waits for its result.
struct LegRelay {
    slot: Arc<AdmissionSlot>,
    leg: Leg,
}

impl ScheduleRelay for LegRelay {
    fn relay(&self, submit: ScheduleSubmit) -> ScheduleSubmit {
        let (reply, rx) = oneshot::channel();
        {
            let mut state = self.slot.state.lock();
            if state.closed {
                return Box::pin(async {
                    Err(KvSchedulerError::BookingFailed(
                        "front admission: request no longer admissible".to_string(),
                    ))
                });
            }
            let step = Some(LegStep { submit, reply });
            match self.leg {
                Leg::Prefill => state.prefill = step,
                Leg::Decode => state.decode = step,
            }
        }
        self.slot.changed.notify_one();
        Box::pin(async move {
            rx.await
                .unwrap_or(Err(KvSchedulerError::SubscriberShutdown))
        })
    }
}

/// Runs the actor step immediately (used to re-plan a constrained decode leg).
struct DirectRelay;

impl ScheduleRelay for DirectRelay {
    fn relay(&self, submit: ScheduleSubmit) -> ScheduleSubmit {
        submit
    }
}

/// Marks the slot abandoned if the caller stops waiting before admission completes.
struct AbandonGuard(Arc<AdmissionSlot>, Arc<Notify>);

impl Drop for AbandonGuard {
    fn drop(&mut self) {
        self.0.state.lock().abandoned = true;
        self.1.notify_one();
    }
}

struct Pending {
    slot: Arc<AdmissionSlot>,
    class_index: usize,
    snapshot: QueueSnapshot,
    decode_constraints: DecodeConstraintsFn,
    enqueued_at: Instant,
}

#[derive(Default)]
struct FrontStats {
    admitted: AtomicU64,
    failed: AtomicU64,
    discarded: AtomicU64,
    rejected: AtomicU64,
    /// Serializer time spent in actor steps (prefill + decode), in ns.
    busy_ns: AtomicU64,
    prefill_ns: AtomicU64,
    decode_ns: AtomicU64,
    max_prefill_ns: AtomicU64,
    max_decode_ns: AtomicU64,
    wait_ns: AtomicU64,
    max_wait_ns: AtomicU64,
    /// Drain passes that found the head blocked on room.
    blocked_passes: AtomicU64,
    depth: AtomicUsize,
    max_depth: AtomicUsize,
}

impl FrontStats {
    fn take(counter: &AtomicU64) -> u64 {
        counter.swap(0, Ordering::Relaxed)
    }
}

/// Handle to the serializer task. Dropping it stops the task once queued work drains.
pub(crate) struct FrontAdmission {
    tx: mpsc::Sender<Pending>,
    wake: Arc<Notify>,
    profile: Arc<PolicyProfile>,
    prefill_host: Arc<RoutingHost>,
    decode_host: Arc<RoutingHost>,
}

impl FrontAdmission {
    pub(crate) fn spawn(
        prefill_host: Arc<RoutingHost>,
        decode_host: Arc<RoutingHost>,
        label: String,
    ) -> Result<Arc<Self>> {
        let prefill_router = prefill_host
            .kv_router_if_enabled()
            .cloned()
            .ok_or_else(|| anyhow!("front admission needs a KV-routed prefill host"))?;
        let decode_router = decode_host
            .kv_router_if_enabled()
            .cloned()
            .ok_or_else(|| anyhow!("front admission needs a KV-routed decode host"))?;
        let profile = Arc::new(
            decode_router
                .kv_router_config()
                .configured_policy_profile()
                .map_err(|error| anyhow!("front admission policy profile: {error}"))?,
        );
        let (tx, rx) = mpsc::channel(COMMAND_CAPACITY);
        let wake = Arc::new(Notify::new());
        let stats = Arc::new(FrontStats::default());
        let serializer = Serializer {
            queue: PolicyQueue::new(&profile),
            profile: Arc::clone(&profile),
            prefill_room: prefill_router.subscribe_room_changed(),
            decode_room: decode_router.subscribe_room_changed(),
            prefill_router,
            decode_router,
            wake: Arc::clone(&wake),
            stats: Arc::clone(&stats),
            start: Instant::now(),
        };
        tokio::spawn(serializer.run(rx));
        tokio::spawn(report_stats(Arc::downgrade(&stats), label));
        Ok(Arc::new(Self {
            tx,
            wake,
            profile,
            prefill_host,
            decode_host,
        }))
    }

    /// Wait for admission, then return the booked prefill and decode routes.
    pub(crate) async fn admit(
        &self,
        request: FrontAdmissionRequest,
    ) -> Result<(RoutePlan, RoutePlan)> {
        let FrontAdmissionRequest {
            request_id,
            metadata,
            session_affinity,
            policy_class,
            prefill,
            decode,
            decode_constraints,
        } = request;
        let isl = prefill.token_ids.len();
        let class_index = self
            .profile
            .resolve_class_index(policy_class.as_deref(), isl);
        let slot = Arc::new(AdmissionSlot::default());
        let _abandon = AbandonGuard(Arc::clone(&slot), Arc::clone(&self.wake));

        let prefill_context =
            planning_context(prefill, &request_id, &metadata, session_affinity.as_ref());
        let decode_context = planning_context(
            decode.clone(),
            &request_id,
            &metadata,
            session_affinity.as_ref(),
        );
        let leg = |leg: Leg| -> Arc<dyn ScheduleRelay> {
            Arc::new(LegRelay {
                slot: Arc::clone(&slot),
                leg,
            })
        };
        // Prepare both legs, then join the queue once both have handed over their
        // steps, so the request waits only behind requests that were ready first.
        let (prefill_plan, decode_plan, ()) = tokio::join!(
            async {
                let plan = self
                    .prefill_host
                    .plan_admitted_upstream(
                        &prefill_context,
                        RequestPhase::Prefill,
                        leg(Leg::Prefill),
                    )
                    .await;
                if plan.is_err() {
                    slot.fail_leg();
                }
                plan
            },
            async {
                let plan = self
                    .decode_host
                    .plan_admitted_upstream(&decode_context, RequestPhase::Decode, leg(Leg::Decode))
                    .await;
                if plan.is_err() {
                    slot.fail_leg();
                }
                plan
            },
            join_when_ready(
                &self.tx,
                Arc::clone(&slot),
                class_index,
                // POC: the legs' overlap is not read back, so the DRR cost is the raw ISL.
                QueueSnapshot::new(isl, 0),
                Arc::clone(&decode_constraints),
            ),
        );
        match (prefill_plan, decode_plan) {
            (Ok(prefill_plan), Ok(decode_plan)) => Ok((prefill_plan, decode_plan)),
            (Ok(prefill_plan), Err(error))
                if format!("{error:#}").contains(DECODE_NEEDS_CONSTRAINTS) =>
            {
                // The prefill worker restricts where decode may run: plan decode again
                // under its constraints, outside the serialized step.
                let result = async {
                    let mut decode = decode;
                    let worker_id = prefill_plan.worker().worker_id;
                    if let Some(constraints) = decode_constraints(worker_id)? {
                        super::merge_decode_topology_constraints(&mut decode, constraints);
                    }
                    let context =
                        planning_context(decode, &request_id, &metadata, session_affinity.as_ref());
                    self.decode_host
                        .plan_admitted_upstream(
                            &context,
                            RequestPhase::Decode,
                            Arc::new(DirectRelay),
                        )
                        .await
                }
                .await;
                match result {
                    Ok(decode_plan) => Ok((prefill_plan, decode_plan)),
                    Err(error) => {
                        prefill_plan.abort(Some(&error)).await;
                        Err(error)
                    }
                }
            }
            (Ok(prefill_plan), Err(error)) => {
                prefill_plan.abort(Some(&error)).await;
                Err(error)
            }
            (Err(error), Ok(decode_plan)) => {
                decode_plan.abort(Some(&error)).await;
                Err(error)
            }
            (Err(error), Err(_)) => Err(error),
        }
    }
}

struct Serializer {
    queue: PolicyQueue<Pending>,
    profile: Arc<PolicyProfile>,
    prefill_router: Arc<KvRouter>,
    decode_router: Arc<KvRouter>,
    prefill_room: watch::Receiver<()>,
    decode_room: watch::Receiver<()>,
    wake: Arc<Notify>,
    stats: Arc<FrontStats>,
    start: Instant,
}

/// Room for one class: a prefill rank within the class prefill limits and a decode
/// rank within the class decode limit, each read from its own pool's load book.
fn class_has_room(
    prefill_router: &KvRouter,
    decode_router: &KvRouter,
    class: &PolicyClassConfig,
) -> bool {
    if !class.queueing_enabled() {
        return true;
    }
    let limits = class.busy_thresholds;
    let prefill_limits = BusyThresholds {
        prefill_busy_threshold: limits.prefill_busy_threshold,
        prefill_busy_threshold_frac: limits.prefill_busy_threshold_frac,
        decode_busy_threshold_frac: None,
    };
    let decode_limits = BusyThresholds {
        prefill_busy_threshold: None,
        prefill_busy_threshold_frac: None,
        decode_busy_threshold_frac: limits.decode_busy_threshold_frac,
    };
    (!prefill_limits.is_configured() || prefill_router.any_rank_has_room(&prefill_limits))
        && (!decode_limits.is_configured() || decode_router.any_rank_has_room(&decode_limits))
}

impl Serializer {
    async fn run(mut self, mut rx: mpsc::Receiver<Pending>) {
        let mut prefill_room_open = true;
        let mut decode_room_open = true;
        let wake = Arc::clone(&self.wake);
        loop {
            self.drain().await;
            let blocked = self.queue.pending_count() > 0;
            tokio::select! {
                command = rx.recv() => match command {
                    Some(pending) => self.enqueue(pending),
                    None => break,
                },
                _ = wake.notified() => {}
                changed = self.prefill_room.changed(), if blocked && prefill_room_open => {
                    prefill_room_open = changed.is_ok();
                }
                changed = self.decode_room.changed(), if blocked && decode_room_open => {
                    decode_room_open = changed.is_ok();
                }
                _ = tokio::time::sleep(BLOCKED_RECHECK), if blocked => {}
            }
            // Take everything already queued before the next drain pass.
            while let Ok(pending) = rx.try_recv() {
                self.enqueue(pending);
            }
        }
        // Senders are gone: fail whatever is still waiting.
        while let Some(entry) = self.queue.pop_next(|_, _, _| true) {
            fail_legs(
                &entry.into_payload().slot,
                "front admission serializer stopped",
            );
        }
    }

    fn enqueue(&mut self, pending: Pending) {
        let class_index = pending.class_index;
        let snapshot = pending.snapshot;
        let arrival = pending
            .enqueued_at
            .saturating_duration_since(self.start)
            .as_secs_f64();
        let workers = self.prefill_router.reachable_worker_count().max(1);
        match self.queue.enqueue(
            class_index,
            workers,
            snapshot,
            arrival,
            0.0,
            0,
            WorkerPlacement::Any,
            pending,
        ) {
            Ok(()) => {
                let depth = self.queue.pending_count();
                self.stats.depth.store(depth, Ordering::Relaxed);
                self.stats.max_depth.fetch_max(depth, Ordering::Relaxed);
            }
            Err((rejection, pending)) => {
                self.stats.rejected.fetch_add(1, Ordering::Relaxed);
                fail_legs(&pending.slot, &rejection.to_string());
            }
        }
    }

    async fn drain(&mut self) {
        loop {
            // Mark both room signals seen before reading room, so a release after
            // this point wakes the next wait.
            self.prefill_room.borrow_and_update();
            self.decode_room.borrow_and_update();
            let mut class_room: Vec<Option<bool>> = vec![None; self.profile.classes().len()];
            let popped = {
                let prefill_router = &self.prefill_router;
                let decode_router = &self.decode_router;
                self.queue
                    .pop_next(|class_index, class, pending: &Pending| {
                        // A request joins only when both legs are ready, so a head
                        // waits on room alone.
                        matches!(pending.slot.readiness(), SlotReadiness::Discard)
                            || *class_room[class_index].get_or_insert_with(|| {
                                class_has_room(prefill_router, decode_router, class)
                            })
                    })
            };
            self.stats
                .depth
                .store(self.queue.pending_count(), Ordering::Relaxed);
            let Some(entry) = popped else {
                if self.queue.pending_count() > 0 {
                    self.stats.blocked_passes.fetch_add(1, Ordering::Relaxed);
                }
                return;
            };
            let pending = entry.into_payload();
            match pending.slot.readiness() {
                SlotReadiness::Ready => self.admit_one(pending).await,
                _ => {
                    self.stats.discarded.fetch_add(1, Ordering::Relaxed);
                    fail_legs(
                        &pending.slot,
                        "front admission: request no longer admissible",
                    );
                }
            }
        }
    }

    async fn admit_one(&mut self, pending: Pending) {
        let Pending {
            slot,
            decode_constraints,
            enqueued_at,
            ..
        } = pending;
        let wait = enqueued_at.elapsed().as_nanos() as u64;
        self.stats.wait_ns.fetch_add(wait, Ordering::Relaxed);
        self.stats.max_wait_ns.fetch_max(wait, Ordering::Relaxed);

        let (Some(prefill), Some(decode)) = slot.take() else {
            fail_legs(&slot, "front admission: legs missing");
            return;
        };
        let started = Instant::now();
        let prefill_result = (prefill.submit).await;
        let prefill_ns = started.elapsed().as_nanos() as u64;
        let decode_started = Instant::now();
        let prefill_worker = prefill_result
            .as_ref()
            .ok()
            .map(|(admitted, _)| admitted.response.best_worker.worker_id);
        let constraints_started = Instant::now();
        let constraints = prefill_worker.map(|worker| decode_constraints(worker));
        dynamo_kv_router::scheduling::poc_timing::record_constraints(
            constraints_started.elapsed().as_nanos() as u64,
        );
        let decode_result = match constraints {
            // The prefill booking failed: do not book decode.
            None => Err(KvSchedulerError::BookingFailed(
                "front admission: prefill leg was not booked".to_string(),
            )),
            Some(Ok(None)) => (decode.submit).await,
            // Decode must stay in the prefill worker's KV-transfer domain, which its
            // prepared selection did not know: the caller re-plans it.
            Some(Ok(Some(_))) | Some(Err(_)) => Err(KvSchedulerError::BookingFailed(
                DECODE_NEEDS_CONSTRAINTS.to_string(),
            )),
        };
        let finished = Instant::now();
        let decode_ns = (finished - decode_started).as_nanos() as u64;
        self.stats
            .prefill_ns
            .fetch_add(prefill_ns, Ordering::Relaxed);
        self.stats
            .max_prefill_ns
            .fetch_max(prefill_ns, Ordering::Relaxed);
        self.stats.decode_ns.fetch_add(decode_ns, Ordering::Relaxed);
        self.stats
            .max_decode_ns
            .fetch_max(decode_ns, Ordering::Relaxed);
        self.stats
            .busy_ns
            .fetch_add((finished - started).as_nanos() as u64, Ordering::Relaxed);
        if prefill_result.is_ok() && decode_result.is_ok() {
            self.stats.admitted.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.failed.fetch_add(1, Ordering::Relaxed);
        }
        // A caller that went away drops the results, and with them the bookings.
        let _ = prefill.reply.send(prefill_result);
        let _ = decode.reply.send(decode_result);
    }
}

/// Join the front queue once both legs have handed over their steps, stamping the
/// arrival then. A request whose leg failed first never joins: the other leg's step, if
/// handed over, is answered with an error, and a later handover fails at once.
async fn join_when_ready(
    tx: &mpsc::Sender<Pending>,
    slot: Arc<AdmissionSlot>,
    class_index: usize,
    snapshot: QueueSnapshot,
    decode_constraints: DecodeConstraintsFn,
) {
    if !slot.wait_ready().await {
        fail_legs(
            &slot,
            "front admission: a leg failed before the request joined the queue",
        );
        return;
    }
    let pending = Pending {
        slot,
        class_index,
        snapshot,
        decode_constraints,
        enqueued_at: Instant::now(),
    };
    if let Err(mpsc::error::SendError(pending)) = tx.send(pending).await {
        fail_legs(&pending.slot, "front admission serializer stopped");
    }
}

/// Answer every leg already handed over with `reason`.
fn fail_legs(slot: &AdmissionSlot, reason: &str) {
    let (prefill, decode) = slot.take();
    for step in [prefill, decode].into_iter().flatten() {
        let _ = step
            .reply
            .send(Err(KvSchedulerError::BookingFailed(reason.to_string())));
    }
}

fn planning_context(
    request: PreprocessedRequest,
    request_id: &str,
    metadata: &BTreeMap<String, String>,
    session_affinity: Option<&SessionAffinityId>,
) -> SingleIn<PreprocessedRequest> {
    let mut context =
        Context::with_id_and_metadata(request, request_id.to_string(), metadata.clone());
    if let Some(session_affinity) = session_affinity {
        context.insert(SESSION_AFFINITY_CONTEXT_KEY, session_affinity.clone());
    }
    context
}

/// Without the front stage, still report per-selection timing for comparison.
pub(crate) fn ensure_selection_timing_reporter() {
    static STARTED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    if front_admission_enabled() {
        return;
    }
    STARTED.get_or_init(|| {
        tokio::spawn(async {
            let mut interval = tokio::time::interval(STATS_INTERVAL);
            interval.tick().await;
            loop {
                interval.tick().await;
                let timing = dynamo_kv_router::scheduling::poc_timing::take();
                tracing::info!(
                    target: "front_admission",
                    selections = timing.selections,
                    mean_select_inputs_us =
                        timing.inputs_ns as f64 / timing.selections.max(1) as f64 / 1e3,
                    mean_select_schedule_us =
                        timing.schedule_ns as f64 / timing.selections.max(1) as f64 / 1e3,
                    "selection timing"
                );
            }
        });
    });
}

async fn report_stats(stats: std::sync::Weak<FrontStats>, label: String) {
    let mut interval = tokio::time::interval(STATS_INTERVAL);
    interval.tick().await;
    loop {
        interval.tick().await;
        let Some(stats) = stats.upgrade() else {
            return;
        };
        let admitted = FrontStats::take(&stats.admitted);
        let failed = FrontStats::take(&stats.failed);
        let discarded = FrontStats::take(&stats.discarded);
        let rejected = FrontStats::take(&stats.rejected);
        let busy_ns = FrontStats::take(&stats.busy_ns);
        let prefill_ns = FrontStats::take(&stats.prefill_ns);
        let decode_ns = FrontStats::take(&stats.decode_ns);
        let max_prefill_ns = FrontStats::take(&stats.max_prefill_ns);
        let max_decode_ns = FrontStats::take(&stats.max_decode_ns);
        let wait_ns = FrontStats::take(&stats.wait_ns);
        let max_wait_ns = FrontStats::take(&stats.max_wait_ns);
        let blocked_passes = FrontStats::take(&stats.blocked_passes);
        let max_depth = stats.max_depth.swap(0, Ordering::Relaxed);
        let depth = stats.depth.load(Ordering::Relaxed);
        let planned = (admitted + failed).max(1);
        let interval_ns = STATS_INTERVAL.as_nanos() as f64;
        let timing = dynamo_kv_router::scheduling::poc_timing::take();
        let [p, d] = timing.legs;
        tracing::info!(
            target: "front_admission",
            stage = %label,
            admitted,
            failed,
            discarded,
            rejected,
            admitted_per_s = admitted as f64 / STATS_INTERVAL.as_secs_f64(),
            serializer_busy_frac = busy_ns as f64 / interval_ns,
            mean_prefill_plan_us = prefill_ns as f64 / planned as f64 / 1e3,
            mean_decode_plan_us = decode_ns as f64 / planned as f64 / 1e3,
            max_prefill_plan_us = max_prefill_ns as f64 / 1e3,
            max_decode_plan_us = max_decode_ns as f64 / 1e3,
            mean_wait_us = wait_ns as f64 / planned as f64 / 1e3,
            max_wait_us = max_wait_ns as f64 / 1e3,
            blocked_passes,
            depth,
            max_depth,
            selections = timing.selections,
            mean_select_inputs_us = timing.inputs_ns as f64 / timing.selections.max(1) as f64 / 1e3,
            booking_prepares = timing.booking_prepares,
            mean_booking_prepare_us =
                timing.booking_prepare_ns as f64 / timing.booking_prepares.max(1) as f64 / 1e3,
            mean_actor_admit_us = (p.admit_us + d.admit_us) / 2.0,
            mean_actor_after_us = (p.after_us + d.after_us) / 2.0,
            p_pickup_us = p.pickup_us,
            p_admit_us = p.admit_us,
            p_after_us = p.after_us,
            p_submit_us = p.submit_us,
            d_pickup_us = d.pickup_us,
            d_admit_us = d.admit_us,
            d_after_us = d.after_us,
            d_submit_us = d.submit_us,
            constraints_us = timing.constraints_us,
            p_cap_wakes_per_s = p.capacity_wakes as f64 / STATS_INTERVAL.as_secs_f64(),
            p_cap_busy_frac = p.capacity_total_us / (interval_ns / 1e3),
            p_other_per_s = p.other_commands as f64 / STATS_INTERVAL.as_secs_f64(),
            p_other_busy_frac = p.other_total_us / (interval_ns / 1e3),
            d_cap_wakes_per_s = d.capacity_wakes as f64 / STATS_INTERVAL.as_secs_f64(),
            d_cap_busy_frac = d.capacity_total_us / (interval_ns / 1e3),
            d_other_per_s = d.other_commands as f64 / STATS_INTERVAL.as_secs_f64(),
            d_other_busy_frac = d.other_total_us / (interval_ns / 1e3),
            "front admission stats"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_HANG: Duration = Duration::from_secs(5);

    fn hand_over(slot: &Arc<AdmissionSlot>, leg: Leg) -> ScheduleSubmit {
        let relay = LegRelay {
            slot: Arc::clone(slot),
            leg,
        };
        relay.relay(Box::pin(async {
            Err(KvSchedulerError::BookingFailed(
                "not run in this test".to_string(),
            ))
        }))
    }

    fn spawn_join(
        tx: &mpsc::Sender<Pending>,
        slot: &Arc<AdmissionSlot>,
    ) -> tokio::task::JoinHandle<()> {
        let tx = tx.clone();
        let slot = Arc::clone(slot);
        let decode_constraints: DecodeConstraintsFn = Arc::new(|_| Ok(None));
        tokio::spawn(async move {
            join_when_ready(&tx, slot, 0, QueueSnapshot::new(1, 0), decode_constraints).await;
        })
    }

    #[tokio::test]
    async fn requests_join_in_readiness_order() {
        let (tx, mut rx) = mpsc::channel(4);
        let first = Arc::new(AdmissionSlot::default());
        let second = Arc::new(AdmissionSlot::default());
        let first_join = spawn_join(&tx, &first);
        let second_join = spawn_join(&tx, &second);
        let _first_prefill = hand_over(&first, Leg::Prefill);
        let _second_prefill = hand_over(&second, Leg::Prefill);
        let _second_decode = hand_over(&second, Leg::Decode);
        tokio::time::timeout(NO_HANG, second_join)
            .await
            .unwrap()
            .unwrap();

        // The request that arrived first is still preparing its decode leg.
        let joined = rx.try_recv().expect("the ready request joined");
        assert!(Arc::ptr_eq(&joined.slot, &second));
        tokio::task::yield_now().await;
        assert!(!first_join.is_finished());
        assert!(rx.try_recv().is_err());

        let _first_decode = hand_over(&first, Leg::Decode);
        tokio::time::timeout(NO_HANG, first_join)
            .await
            .unwrap()
            .unwrap();
        let next = rx.try_recv().expect("the request joined once ready");
        assert!(Arc::ptr_eq(&next.slot, &first));
        assert!(joined.enqueued_at <= next.enqueued_at);
    }

    #[tokio::test]
    async fn leg_failure_before_ready_never_joins() {
        let (tx, mut rx) = mpsc::channel(4);
        let slot = Arc::new(AdmissionSlot::default());
        let join = spawn_join(&tx, &slot);
        let prefill = hand_over(&slot, Leg::Prefill);
        slot.fail_leg();
        tokio::time::timeout(NO_HANG, join).await.unwrap().unwrap();
        assert!(rx.try_recv().is_err());

        // The leg already handed over is answered, not left waiting.
        let answer = tokio::time::timeout(NO_HANG, prefill).await.unwrap();
        let Err(KvSchedulerError::BookingFailed(reason)) = answer else {
            panic!("the handed-over leg was not failed");
        };
        assert!(reason.contains("before the request joined"));
        // A leg handed over afterwards fails at once.
        let late = tokio::time::timeout(NO_HANG, hand_over(&slot, Leg::Decode))
            .await
            .unwrap();
        assert!(matches!(late, Err(KvSchedulerError::BookingFailed(_))));
    }
}
