// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Proof of concept: one admission queue in front of both disaggregated pools.
//!
//! A request waits in one policy-class queue (DRR across classes) until a prefill
//! worker has room under its class prefill limit and a decode worker has room under
//! its class decode limit. One serializer then books the prefill worker and the
//! decode worker, in that order, before it pops the next request, so every admission
//! sees every earlier booking in both pools. The two booked [`RoutePlan`]s go back to
//! the caller, which attaches them to the request; the prefill hop and the decode
//! host send them without selecting or queueing again.
//!
//! Enabled by `DYN_ROUTER_FRONT_ADMISSION=1` (KV routing on both hops only).
//! Not covered: deadlines, session-affinity-aware room checks, per-request decode
//! eligibility in the room check, and prefill pool succession beyond rebuilding the
//! stage when the prefill binding changes.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use dynamo_kv_router::{
    protocols::RoutingConstraints,
    scheduling::{
        BusyThresholds, PolicyClassConfig, PolicyProfile, PolicyQueue, QueueSnapshot,
        WorkerPlacement,
    },
};
use dynamo_runtime::pipeline::{Context, SingleIn};
use tokio::sync::{mpsc, oneshot, watch};

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

/// Diagnostic switch: plan the prefill and decode legs concurrently within one
/// admission step instead of one after the other.
const FRONT_ADMISSION_PARALLEL_ENV: &str = "DYN_ROUTER_FRONT_ADMISSION_PARALLEL";

pub(crate) fn front_admission_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| dynamo_runtime::config::env_is_truthy(FRONT_ADMISSION_ENV))
}

fn parallel_planning_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| dynamo_runtime::config::env_is_truthy(FRONT_ADMISSION_PARALLEL_ENV))
}

/// Derives the decode leg's routing constraints from the prefill worker the
/// serializer booked (its KV-transfer domain).
pub(crate) type DecodeConstraintsFn =
    Box<dyn FnOnce(u64) -> Result<Option<RoutingConstraints>> + Send + 'static>;

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

struct Pending {
    request: FrontAdmissionRequest,
    enqueued_at: Instant,
    response: oneshot::Sender<Result<(RoutePlan, RoutePlan)>>,
}

#[derive(Default)]
struct FrontStats {
    admitted: AtomicU64,
    failed: AtomicU64,
    cancelled: AtomicU64,
    rejected: AtomicU64,
    /// Serializer time spent planning (prefill plan + decode plan), in ns.
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
    profile: Arc<PolicyProfile>,
    stats: Arc<FrontStats>,
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
        let stats = Arc::new(FrontStats::default());
        let serializer = Serializer {
            queue: PolicyQueue::new(&profile),
            profile: Arc::clone(&profile),
            prefill_room: prefill_router.subscribe_room_changed(),
            decode_room: decode_router.subscribe_room_changed(),
            prefill_router,
            decode_router,
            prefill_host,
            decode_host,
            stats: Arc::clone(&stats),
            start: Instant::now(),
        };
        tokio::spawn(serializer.run(rx));
        tokio::spawn(report_stats(Arc::downgrade(&stats), label));
        Ok(Arc::new(Self { tx, profile, stats }))
    }

    /// Wait for admission, then return the booked prefill and decode routes.
    pub(crate) async fn admit(
        &self,
        request: FrontAdmissionRequest,
    ) -> Result<(RoutePlan, RoutePlan)> {
        let (response, rx) = oneshot::channel();
        self.tx
            .send(Pending {
                request,
                enqueued_at: Instant::now(),
                response,
            })
            .await
            .map_err(|_| anyhow!("front admission serializer stopped"))?;
        rx.await
            .map_err(|_| anyhow!("front admission serializer dropped the request"))?
    }

    #[allow(dead_code)]
    pub(crate) fn pending(&self) -> usize {
        self.stats.depth.load(Ordering::Relaxed)
    }

    #[allow(dead_code)]
    pub(crate) fn profile(&self) -> &PolicyProfile {
        &self.profile
    }
}

struct Serializer {
    queue: PolicyQueue<Pending>,
    profile: Arc<PolicyProfile>,
    prefill_router: Arc<KvRouter>,
    decode_router: Arc<KvRouter>,
    prefill_host: Arc<RoutingHost>,
    decode_host: Arc<RoutingHost>,
    prefill_room: watch::Receiver<()>,
    decode_room: watch::Receiver<()>,
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
        loop {
            self.drain().await;
            let blocked = self.queue.pending_count() > 0;
            tokio::select! {
                command = rx.recv() => match command {
                    Some(pending) => self.enqueue(pending),
                    None => break,
                },
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
            let pending = entry.into_payload();
            let _ = pending
                .response
                .send(Err(anyhow!("front admission serializer stopped")));
        }
    }

    fn enqueue(&mut self, pending: Pending) {
        let isl = pending.request.prefill.token_ids.len();
        let class_index = self
            .profile
            .resolve_class_index(pending.request.policy_class.as_deref(), isl);
        // POC: no overlap lookup before admission, so the DRR cost is the raw ISL.
        let snapshot = QueueSnapshot::new(isl, 0);
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
                let _ = pending.response.send(Err(anyhow!(rejection)));
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
                        if pending.response.is_closed() {
                            return true;
                        }
                        *class_room[class_index].get_or_insert_with(|| {
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
            if pending.response.is_closed() {
                self.stats.cancelled.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            self.admit_one(pending).await;
        }
    }

    async fn admit_one(&mut self, pending: Pending) {
        let Pending {
            request,
            enqueued_at,
            response,
        } = pending;
        let FrontAdmissionRequest {
            request_id,
            metadata,
            session_affinity,
            policy_class: _,
            prefill,
            mut decode,
            decode_constraints,
        } = request;

        let wait = enqueued_at.elapsed().as_nanos() as u64;
        self.stats.wait_ns.fetch_add(wait, Ordering::Relaxed);
        self.stats.max_wait_ns.fetch_max(wait, Ordering::Relaxed);

        let started = Instant::now();
        let (result, prefill_ns, decode_ns) = if parallel_planning_enabled() {
            self.plan_parallel(
                prefill,
                decode,
                decode_constraints,
                &request_id,
                &metadata,
                session_affinity.as_ref(),
            )
            .await
        } else {
            let prefill_context =
                planning_context(prefill, &request_id, &metadata, session_affinity.as_ref());
            let prefill_plan = self
                .prefill_host
                .plan_admitted_upstream(&prefill_context, RequestPhase::Prefill)
                .await;
            let prefill_ns = started.elapsed().as_nanos() as u64;
            let decode_started = Instant::now();
            let result = match prefill_plan {
                Err(error) => Err(error),
                Ok(prefill_plan) => match decode_constraints(prefill_plan.worker().worker_id) {
                    Err(error) => {
                        prefill_plan.abort(Some(&error)).await;
                        Err(error)
                    }
                    Ok(constraints) => {
                        if let Some(constraints) = constraints {
                            super::merge_decode_topology_constraints(&mut decode, constraints);
                        }
                        let decode_context = planning_context(
                            decode,
                            &request_id,
                            &metadata,
                            session_affinity.as_ref(),
                        );
                        match self
                            .decode_host
                            .plan_admitted_upstream(&decode_context, RequestPhase::Decode)
                            .await
                        {
                            Ok(decode_plan) => Ok((prefill_plan, decode_plan)),
                            Err(error) => {
                                prefill_plan.abort(Some(&error)).await;
                                Err(error)
                            }
                        }
                    }
                },
            };
            (
                result,
                prefill_ns,
                decode_started.elapsed().as_nanos() as u64,
            )
        };
        let finished = Instant::now();
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

        match result {
            Ok(plans) => {
                self.stats.admitted.fetch_add(1, Ordering::Relaxed);
                if let Err(Ok((prefill_plan, decode_plan))) = response.send(Ok(plans)) {
                    // The caller went away after admission: give both bookings back.
                    self.stats.cancelled.fetch_add(1, Ordering::Relaxed);
                    prefill_plan.abort(None).await;
                    decode_plan.abort(None).await;
                }
            }
            Err(error) => {
                self.stats.failed.fetch_add(1, Ordering::Relaxed);
                let _ = response.send(Err(error));
            }
        }
    }

    /// Diagnostic variant: plan both legs at once inside the same serialized step.
    /// Still books both before the next pop. The decode leg is planned before the
    /// prefill worker is known, so if that worker imposes KV-transfer constraints the
    /// decode plan is given back and planned again under them.
    async fn plan_parallel(
        &self,
        prefill: PreprocessedRequest,
        decode: PreprocessedRequest,
        decode_constraints: DecodeConstraintsFn,
        request_id: &str,
        metadata: &BTreeMap<String, String>,
        session_affinity: Option<&SessionAffinityId>,
    ) -> (Result<(RoutePlan, RoutePlan)>, u64, u64) {
        let prefill_context = planning_context(prefill, request_id, metadata, session_affinity);
        let decode_context =
            planning_context(decode.clone(), request_id, metadata, session_affinity);
        let timed = |phase: RequestPhase,
                     host: Arc<RoutingHost>,
                     context: SingleIn<PreprocessedRequest>| async move {
            let started = Instant::now();
            let plan = host.plan_admitted_upstream(&context, phase).await;
            (plan, started.elapsed().as_nanos() as u64)
        };
        let ((prefill_plan, prefill_ns), (decode_plan, decode_ns)) = tokio::join!(
            timed(
                RequestPhase::Prefill,
                Arc::clone(&self.prefill_host),
                prefill_context
            ),
            timed(
                RequestPhase::Decode,
                Arc::clone(&self.decode_host),
                decode_context
            ),
        );
        let result = match (prefill_plan, decode_plan) {
            (Ok(prefill_plan), Ok(decode_plan)) => {
                match decode_constraints(prefill_plan.worker().worker_id) {
                    Ok(None) => Ok((prefill_plan, decode_plan)),
                    Ok(Some(constraints)) => {
                        decode_plan.abort(None).await;
                        let mut decode = decode;
                        super::merge_decode_topology_constraints(&mut decode, constraints);
                        let decode_context =
                            planning_context(decode, request_id, metadata, session_affinity);
                        match self
                            .decode_host
                            .plan_admitted_upstream(&decode_context, RequestPhase::Decode)
                            .await
                        {
                            Ok(decode_plan) => Ok((prefill_plan, decode_plan)),
                            Err(error) => {
                                prefill_plan.abort(Some(&error)).await;
                                Err(error)
                            }
                        }
                    }
                    Err(error) => {
                        prefill_plan.abort(Some(&error)).await;
                        decode_plan.abort(Some(&error)).await;
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
        };
        (result, prefill_ns, decode_ns)
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
        let cancelled = FrontStats::take(&stats.cancelled);
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
        tracing::info!(
            target: "front_admission",
            stage = %label,
            admitted,
            failed,
            cancelled,
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
            "front admission stats"
        );
    }
}
