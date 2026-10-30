// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Proof of concept: one admission queue in front of both disaggregated pools.
//!
//! A request enters one policy-class queue (DRR across classes) at once. Meanwhile, on
//! its own task, it prepares both legs: the prefill and decode selections run their
//! index lookups, classification, candidate rows and planning setup, and each stops at
//! its queue-actor step, which it hands to this stage through a [`ScheduleRelay`]. The
//! serializer pops a request when both legs are ready and a prefill worker has room
//! under its class prefill limit and a decode worker has room under its class decode
//! limit. It then runs the prefill actor step and the decode actor step, in that order,
//! before it pops the next request, so every admission sees every earlier booking in
//! both pools. The two booked [`RoutePlan`]s go back to the caller, which attaches them
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

/// Run the decode booking alongside the prefill booking instead of after it.
const PARALLEL_SUBMIT_ENV: &str = "DYN_ROUTER_FRONT_ADMISSION_PARALLEL_SUBMIT";

fn parallel_submit_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| dynamo_runtime::config::env_is_truthy(PARALLEL_SUBMIT_ENV))
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
    /// The serializer is done with this admission; a leg handed over now fails at once.
    closed: bool,
}

/// One admission's meeting point between the request task and the serializer.
#[derive(Default)]
struct AdmissionSlot {
    state: Mutex<SlotState>,
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
    wake: Arc<Notify>,
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
        self.wake.notify_one();
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
    /// Drain passes that found the head blocked on room or on unprepared legs.
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

        // Join the queue first so waiting overlaps preparation.
        self.tx
            .send(Pending {
                slot: Arc::clone(&slot),
                class_index,
                // POC: no overlap lookup before admission, so the DRR cost is the raw ISL.
                snapshot: QueueSnapshot::new(isl, 0),
                decode_constraints: Arc::clone(&decode_constraints),
                enqueued_at: Instant::now(),
            })
            .await
            .map_err(|_| anyhow!("front admission serializer stopped"))?;

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
                wake: Arc::clone(&self.wake),
            })
        };
        let mark_failed = |slot: &AdmissionSlot| {
            slot.state.lock().leg_failed = true;
            self.wake.notify_one();
        };
        let (prefill_plan, decode_plan) = tokio::join!(
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
                    mark_failed(&slot);
                }
                plan
            },
            async {
                let plan = self
                    .decode_host
                    .plan_admitted_upstream(&decode_context, RequestPhase::Decode, leg(Leg::Decode))
                    .await;
                if plan.is_err() {
                    mark_failed(&slot);
                }
                plan
            },
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
                        match pending.slot.readiness() {
                            SlotReadiness::Discard => true,
                            SlotReadiness::Waiting => false,
                            SlotReadiness::Ready => {
                                *class_room[class_index].get_or_insert_with(|| {
                                    class_has_room(prefill_router, decode_router, class)
                                })
                            }
                        }
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
        let LegStep {
            submit: decode_submit,
            reply: decode_reply,
        } = decode;
        let mut decode_submit = Some(decode_submit);
        let started = Instant::now();
        // In parallel, decode is booked before the prefill worker, and with it the decode
        // leg's KV-transfer constraints, is known. A decode booking that turns out to be
        // constrained is dropped below, which releases it, and re-planned by the caller.
        let (prefill_result, mut early_decode) = match decode_submit.take() {
            Some(decode_submit) if parallel_submit_enabled() => {
                let (prefill_result, decode_result) = tokio::join!(prefill.submit, decode_submit);
                (prefill_result, Some(decode_result))
            }
            submit => {
                decode_submit = submit;
                ((prefill.submit).await, None)
            }
        };
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
            Some(Ok(None)) => match (early_decode.take(), decode_submit.take()) {
                (Some(result), _) => result,
                (None, Some(submit)) => submit.await,
                (None, None) => Err(KvSchedulerError::BookingFailed(
                    "front admission: decode leg missing".to_string(),
                )),
            },
            // Decode must stay in the prefill worker's KV-transfer domain, which its
            // prepared selection did not know: the caller re-plans it.
            Some(Ok(Some(_))) | Some(Err(_)) => Err(KvSchedulerError::BookingFailed(
                DECODE_NEEDS_CONSTRAINTS.to_string(),
            )),
        };
        // A decode booking made in parallel but not used is released here.
        drop(early_decode);
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
        let _ = decode_reply.send(decode_result);
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
            "front admission stats"
        );
    }
}
