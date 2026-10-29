// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use arc_swap::ArcSwapOption;
use crossbeam_queue::SegQueue;
use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHashSet};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::Instant;

use self::capacity::CapacityUpdates;
use crate::overlap::SelectedWorkerTierSnapshot;
use crate::overlap_refresh::{
    NoopOverlapScoresRefresh, OverlapScoresRefresh, read_overlap_refresh_after, refresh_overlap,
};
use crate::scheduling_compat::{ClassificationOverrides, ClassifyRequest};
use dynamo_admission::policy_queue::{PolicyQueue, QueueMetadata, QueueSnapshot};
#[cfg(test)]
use dynamo_router_types::config::RouterQueuePolicy;
use dynamo_router_types::config::policy::{
    BusyThresholds, FullRankCounts, PolicyClassConfig, PolicyProfile, RejectThresholdsProvider,
};
use dynamo_router_types::eligibility::{RoutingEligibility, worker_config_for_rank};
use dynamo_router_types::prefill_load::{PrefillLoadEstimator, effective_prefill_tokens};
use dynamo_router_types::protocols::WorkerPlacement;
use dynamo_router_types::protocols::{
    LocalBlockHash, PrefillLoadHint, SelectionTrace, SelectionTraceSpec, WorkerConfigLike,
    WorkerId, WorkerSelectionResult, WorkerWithDpRank,
};
use dynamo_router_types::sched::{
    AdvisorySchedulingResponse, AdvisoryWorkerLoad, AttemptId, KvSchedulerError,
    NonMaxOverlapSelection, NonMaxOverlapSelectionObserver, OverloadedWorkerProvider,
    SchedulingContext, SchedulingRequest, SchedulingResponse, WorkerAvailabilityProvider,
};
use dynamo_slot_tracker::{
    ActiveSequencesMultiWorker, SequenceError, SequencePublisher, SequenceRequest,
    WorkerLoadProjection,
};
use dynamo_worker_selection::plugin::{
    RuntimePolicyConfig, RuntimePolicyError, RuntimePolicyUpdate,
};
use dynamo_worker_selection::plugin::{WorkerCapacity, WorkerInputRows, WorkerLoadInput};
use dynamo_worker_selection::selector::{
    CandidatePreparer, PreparedCandidates, WorkerInputs, WorkerSelectionInput, WorkerSelector,
    worker_input_rows,
};

// The fallback lives beside `WorkerCapacity`; this path stays for existing importers.
pub use dynamo_router_types::load::DEFAULT_MAX_BATCHED_TOKENS;

mod capacity;

const ADMISSION_CHANNEL_CAPACITY: usize = 65_536;

struct ClassQueueCounters {
    pending_count: AtomicUsize,
    pending_isl_tokens: AtomicUsize,
    pending_cached_tokens: AtomicUsize,
    received_total: AtomicU64,
    rejected_due_time_passed_total: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassQueueStats {
    pub pending_count: usize,
    pub pending_isl_tokens: usize,
    pub pending_cached_tokens: usize,
    /// Attempts received by the admission actor; includes retries, excludes advisory probes.
    pub received_total: u64,
    /// Attempts rejected because their due time passed, counted once at rejection.
    pub rejected_due_time_passed_total: u64,
}

struct QueuedRequest {
    request: SchedulingRequest,
    attempt_tx: Option<oneshot::Sender<AttemptId>>,
    lifecycle_transfer: Option<Arc<AdmissionLifecycleTransfer>>,
    enqueue_at: Instant,
    due_at: Option<Instant>,
    block_hashes: Option<Vec<LocalBlockHash>>,
    /// Actor turn that received the request. Its prepared rows last only that turn.
    enqueue_turn: u64,
}

/// Candidate rows built on the caller task, before the request reaches the actor.
///
/// The actor selects from them only when it would build the same rows: the worker
/// configuration, availability, and overload it reads at selection match the ones
/// the rows were built from, and its load projection found no rank full. Load inputs
/// depend on the actor's earlier bookings, so it fills them at selection.
struct PreparedAdmission {
    candidates: PreparedCandidates,
    /// Configuration generation, loaded before the configuration was read.
    workers_generation: u64,
    available_worker_ids: Option<Arc<HashSet<WorkerId>>>,
    overloaded_worker_ids: Option<HashSet<WorkerId>>,
}

impl PreparedAdmission {
    /// The rows, when built from the inputs the actor selects with.
    fn into_current(
        self: Box<Self>,
        workers_generation: u64,
        available_worker_ids: Option<&Arc<HashSet<WorkerId>>>,
        overloaded_worker_ids: Option<&HashSet<WorkerId>>,
    ) -> Option<PreparedCandidates> {
        // Availability is one shared snapshot until it changes, so this usually
        // compares pointers.
        (self.workers_generation == workers_generation
            && self.available_worker_ids.as_ref() == available_worker_ids
            && self.overloaded_worker_ids.as_ref() == overloaded_worker_ids)
            .then_some(self.candidates)
    }
}

/// Eligibility for one selection: caller constraints, hard availability,
/// transient overload, the ranks the load projection found full, and an
/// eligible exclusive affinity target.
fn selection_eligibility<'a, C: WorkerConfigLike>(
    request: &'a SchedulingRequest,
    workers: &HashMap<WorkerId, C>,
    overloaded_worker_ids: Option<&'a HashSet<WorkerId>>,
    available_worker_ids: Option<&'a HashSet<WorkerId>>,
    full_ranks: Option<&'a FullRankCounts>,
    exclusive_affinity: bool,
) -> RoutingEligibility<'a> {
    let mut eligibility = request
        .eligibility_with_overloaded(overloaded_worker_ids)
        .with_available_workers(available_worker_ids);
    if let Some(full_ranks) = full_ranks {
        // Workers whose every rank the projection pass found past the host's reject
        // thresholds are skipped like overloaded ones; nothing is scanned here.
        eligibility = eligibility.with_full_worker_ranks(full_ranks);
    }
    match request.affinity_target {
        Some(target)
            if exclusive_affinity && eligibility.affinity_target_is_eligible(workers, target) =>
        {
            eligibility.with_affinity_target(target)
        }
        _ => eligibility,
    }
}

struct SelectedWorkerForRequest {
    selection: WorkerSelectionResult,
    selection_trace: Option<SelectionTrace>,
    selected_worker_tiers: SelectedWorkerTierSnapshot,
    selected_worker_load: AdvisoryWorkerLoad,
    non_max_overlap_selection: Option<NonMaxOverlapSelection>,
}

fn non_max_overlap_selection<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    eligibility: RoutingEligibility<'_>,
    selected_worker: WorkerWithDpRank,
    selected_overlap_blocks: f64,
) -> Option<NonMaxOverlapSelection> {
    if eligibility.pinned_worker().is_some() {
        return None;
    }

    let mut highest_overlap = None;
    for (&worker, &overlap_blocks) in &request.overlap.effective_overlap_blocks {
        if overlap_blocks <= selected_overlap_blocks
            || eligibility.validate_worker_rank(workers, worker).is_err()
        {
            continue;
        }
        let is_better = highest_overlap.is_none_or(
            |(current_worker, current_overlap): (WorkerWithDpRank, f64)| {
                overlap_blocks > current_overlap
                    || (overlap_blocks == current_overlap && worker < current_worker)
            },
        );
        if is_better {
            highest_overlap = Some((worker, overlap_blocks));
        }
    }

    let (highest_overlap_worker, highest_overlap_blocks) = highest_overlap?;
    (highest_overlap_blocks > selected_overlap_blocks).then_some(NonMaxOverlapSelection {
        selected_worker,
        highest_overlap_worker,
        highest_overlap_blocks,
        selected_overlap_blocks,
    })
}

#[allow(clippy::large_enum_variant)]
enum AdmissionCommand {
    PolicyConfig {
        update: Option<RuntimePolicyUpdate>,
        response: oneshot::Sender<Result<RuntimePolicyConfig, RuntimePolicyError>>,
    },
    Enqueue {
        request: SchedulingRequest,
        attempt_tx: Option<oneshot::Sender<AttemptId>>,
        block_hashes: Option<Vec<LocalBlockHash>>,
        prepared: Option<Box<PreparedAdmission>>,
        queue_metadata: QueueMetadata,
        lease: Option<Box<RequestLifecycleLease>>,
        ack_tx: oneshot::Sender<Option<Box<RequestLifecycleLease>>>,
    },
    SelectWithoutAdmission {
        request: SchedulingRequest,
        resp_tx: oneshot::Sender<Result<AdvisorySchedulingResponse, KvSchedulerError>>,
    },
    Update {
        ack_tx: oneshot::Sender<()>,
    },
    Cleanup,
}

/// Control handle for the policy owned by one scheduler actor. Updates do not replace the
/// scheduler, its queued requests, or active bookings. A stopped actor rejects further operations.
pub struct RuntimePolicyControl {
    tx: mpsc::WeakSender<AdmissionCommand>,
}

impl RuntimePolicyControl {
    pub async fn get(&self) -> Result<RuntimePolicyConfig, RuntimePolicyError> {
        self.request(None).await
    }

    pub async fn update(
        &self,
        update: RuntimePolicyUpdate,
    ) -> Result<RuntimePolicyConfig, RuntimePolicyError> {
        self.request(Some(update)).await
    }

    async fn request(
        &self,
        update: Option<RuntimePolicyUpdate>,
    ) -> Result<RuntimePolicyConfig, RuntimePolicyError> {
        let (response, receiver) = oneshot::channel();
        let sender = self.tx.upgrade().ok_or(RuntimePolicyError::Unavailable)?;
        sender
            .send(AdmissionCommand::PolicyConfig { update, response })
            .await
            .map_err(|_| RuntimePolicyError::Unavailable)?;
        drop(sender);
        receiver
            .await
            .map_err(|_| RuntimePolicyError::Unavailable)?
    }
}

pub use dynamo_router_types::protocols::SchedulerBookingDescriptor;

#[derive(Debug)]
enum SchedulerCleanupTarget {
    AdmissionLifecycle(Arc<AdmissionLifecycleTransfer>),
    Booking(SchedulerBookingDescriptor),
    ExpiredBooking(SchedulerBookingDescriptor),
}

#[derive(Debug)]
enum AdmissionLifecycleState {
    Unarmed { request_id: String },
    Pending { request_id: String },
    Booking(SchedulerBookingDescriptor),
    Disarmed,
}

#[derive(Debug)]
struct AdmissionLifecycleTransfer {
    state: Mutex<AdmissionLifecycleState>,
}

enum AdmissionLifecycleCleanup {
    Pending { request_id: String },
    Booking(SchedulerBookingDescriptor),
}

impl AdmissionLifecycleTransfer {
    fn new(request_id: String) -> Self {
        Self {
            state: Mutex::new(AdmissionLifecycleState::Unarmed { request_id }),
        }
    }

    fn arm_pending(&self) {
        let mut state = self.state.lock();
        let AdmissionLifecycleState::Unarmed { request_id } = &*state else {
            return;
        };
        *state = AdmissionLifecycleState::Pending {
            request_id: request_id.clone(),
        };
    }

    fn arm_booking(&self, booking: SchedulerBookingDescriptor) {
        let mut state = self.state.lock();
        if matches!(
            *state,
            AdmissionLifecycleState::Unarmed { .. } | AdmissionLifecycleState::Pending { .. }
        ) {
            *state = AdmissionLifecycleState::Booking(booking);
        }
    }

    fn disarm(&self) {
        *self.state.lock() = AdmissionLifecycleState::Disarmed;
    }

    fn cleanup_target(&self) -> Option<AdmissionLifecycleCleanup> {
        let mut state = self.state.lock();
        let cleanup = match &*state {
            AdmissionLifecycleState::Unarmed { .. } | AdmissionLifecycleState::Disarmed => None,
            AdmissionLifecycleState::Pending { request_id } => {
                Some(AdmissionLifecycleCleanup::Pending {
                    request_id: request_id.clone(),
                })
            }
            AdmissionLifecycleState::Booking(booking) => {
                Some(AdmissionLifecycleCleanup::Booking(booking.clone()))
            }
        };
        *state = AdmissionLifecycleState::Disarmed;
        cleanup
    }

    fn is_armed(&self) -> bool {
        matches!(
            *self.state.lock(),
            AdmissionLifecycleState::Pending { .. } | AdmissionLifecycleState::Booking(_)
        )
    }
}

struct AdmissionCleanupEntry {
    target: SchedulerCleanupTarget,
}

#[derive(Default)]
struct AdmissionCleanup {
    dirty: SegQueue<AdmissionCleanupEntry>,
    pending: AtomicBool,
}

impl AdmissionCleanup {
    fn enqueue(&self, cleanup: AdmissionCleanupEntry) -> bool {
        self.dirty.push(cleanup);
        !self.pending.swap(true, AtomicOrdering::AcqRel)
    }

    fn drain(&self) -> Vec<AdmissionCleanupEntry> {
        if !self.pending.load(AtomicOrdering::Acquire) {
            return Vec::new();
        }

        // Drain to a quiescent point to preserve the coalesced-wake handoff. A large burst of
        // active lease drops can delay actor commands; any bounded or interleaved drain
        // must preserve wake correctness and be benchmarked.
        let mut dirty = Vec::new();
        loop {
            while let Some(cleanup) = self.dirty.pop() {
                dirty.push(cleanup);
            }
            self.pending.store(false, AtomicOrdering::Release);
            if self.dirty.is_empty() {
                return dirty;
            }
            self.pending.store(true, AtomicOrdering::Release);
        }
    }
}

fn enqueue_cleanup_entry(
    cleanup: &AdmissionCleanup,
    actor_tx: &mpsc::Sender<AdmissionCommand>,
    entry: AdmissionCleanupEntry,
) {
    if !cleanup.enqueue(entry) {
        return;
    }
    match actor_tx.try_send(AdmissionCommand::Cleanup) {
        Ok(()) | Err(mpsc::error::TrySendError::Full(_)) => {}
        Err(mpsc::error::TrySendError::Closed(_)) => {
            // No actor remains to consume destructor-only cleanup.
            cleanup.drain();
        }
    }
}

/// Cloneable non-blocking ingress for attempt-scoped scheduler cleanup.
#[doc(hidden)]
#[derive(Clone)]
pub struct SchedulerBookingCleanup {
    cleanup: Arc<AdmissionCleanup>,
    actor_tx: mpsc::Sender<AdmissionCommand>,
    direct_release: DirectBookingRelease,
}

type DirectBookingRelease =
    Arc<dyn Fn(&SchedulerBookingDescriptor) -> Result<(), SequenceError> + Send + Sync>;

#[doc(hidden)]
pub struct SchedulerCleanupAck {
    response: Result<(), SequenceError>,
}

impl SchedulerCleanupAck {
    pub async fn wait(self) -> Result<(), SequenceError> {
        self.response
    }
}

impl SchedulerBookingCleanup {
    fn enqueue_entry(&self, entry: AdmissionCleanupEntry) {
        enqueue_cleanup_entry(&self.cleanup, &self.actor_tx, entry);
    }

    pub fn enqueue(&self, booking: SchedulerBookingDescriptor) {
        self.enqueue_entry(AdmissionCleanupEntry {
            target: SchedulerCleanupTarget::Booking(booking),
        });
    }

    pub fn enqueue_expired(&self, booking: SchedulerBookingDescriptor) {
        self.enqueue_entry(AdmissionCleanupEntry {
            target: SchedulerCleanupTarget::ExpiredBooking(booking),
        });
    }

    /// Apply explicit release now. The acknowledgement covers the mutation and
    /// capacity notification; pending admission runs asynchronously.
    pub fn enqueue_acknowledged(&self, booking: SchedulerBookingDescriptor) -> SchedulerCleanupAck {
        SchedulerCleanupAck {
            response: (self.direct_release)(&booking),
        }
    }

    pub async fn enqueue_and_wait(
        &self,
        booking: SchedulerBookingDescriptor,
    ) -> Result<(), SequenceError> {
        self.enqueue_acknowledged(booking).wait().await
    }
}

/// A booking whose release this handle owns until `commit` hands it over.
/// Dropping an armed handle frees the booking through the scheduler's cleanup queue.
#[doc(hidden)]
#[must_use]
pub struct BookingHandle {
    booking: SchedulerBookingDescriptor,
    cleanup: SchedulerBookingCleanup,
    armed: bool,
}

impl std::fmt::Debug for BookingHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BookingHandle")
            .field("booking", &self.booking)
            .field("armed", &self.armed)
            .finish()
    }
}

impl BookingHandle {
    /// Hand the booking to a longer-lived owner; the handle stops guarding it.
    #[must_use]
    pub fn commit(mut self) -> SchedulerBookingDescriptor {
        self.armed = false;
        SchedulerBookingDescriptor {
            request_id: std::mem::take(&mut self.booking.request_id),
            worker: self.booking.worker,
            attempt_id: self.booking.attempt_id,
        }
    }

    /// Free the booking directly and record a capacity notification.
    /// This does not wait for the actor to recheck pending admissions.
    pub async fn release(mut self) -> Result<(), SequenceError> {
        self.armed = false;
        self.cleanup
            .enqueue_acknowledged(self.booking.clone())
            .wait()
            .await
    }
}

impl Drop for BookingHandle {
    fn drop(&mut self) {
        if self.armed {
            self.cleanup.enqueue(self.booking.clone());
        }
    }
}

/// Single-owner cleanup lease for one scheduler-tracked request.
#[doc(hidden)]
pub struct RequestLifecycleLease {
    cleanup: Arc<AdmissionCleanup>,
    actor_tx: mpsc::Sender<AdmissionCommand>,
    transfer: Option<Arc<AdmissionLifecycleTransfer>>,
}

impl std::fmt::Debug for RequestLifecycleLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RequestLifecycleLease")
            .field("transfer", &self.transfer)
            .finish_non_exhaustive()
    }
}

impl RequestLifecycleLease {
    #[doc(hidden)]
    pub fn disarm(&mut self) {
        if let Some(transfer) = self.transfer.take() {
            transfer.disarm();
        }
    }

    /// Hand the booking to its long-term owner: the lease stops guarding it.
    #[must_use]
    pub fn commit(mut self) -> Option<SchedulerBookingDescriptor> {
        let booking = match self.transfer.as_ref().map(|transfer| transfer.state.lock()) {
            Some(state) => match &*state {
                AdmissionLifecycleState::Booking(booking) => Some(booking.clone()),
                _ => None,
            },
            None => None,
        };
        self.disarm();
        booking
    }
}

impl Drop for RequestLifecycleLease {
    fn drop(&mut self) {
        let Some(transfer) = self.transfer.take() else {
            return;
        };
        if !transfer.is_armed() {
            return;
        }
        enqueue_cleanup_entry(
            &self.cleanup,
            &self.actor_tx,
            AdmissionCleanupEntry {
                target: SchedulerCleanupTarget::AdmissionLifecycle(transfer),
            },
        );
    }
}

struct SchedulerQueueActor<
    P: SequencePublisher,
    C: WorkerConfigLike,
    Sel: WorkerSelector<C>,
    RF: OverlapScoresRefresh,
> {
    pending: PolicyQueue<QueuedRequest>,
    cleanup: Arc<AdmissionCleanup>,
    capacity_updates: Option<Arc<CapacityUpdates>>,
    capacity_workers: FxHashSet<WorkerWithDpRank>,
    profile: Arc<PolicyProfile>,
    pending_count: Arc<AtomicUsize>,
    pending_isl_tokens: Arc<AtomicUsize>,
    class_counters: Arc<Vec<ClassQueueCounters>>,
    slots: Arc<ActiveSequencesMultiWorker<P>>,
    workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
    /// Advanced when the actor observes a configuration change; see `PreparedAdmission`.
    workers_generation: Arc<AtomicU64>,
    /// Current preparer, republished after a runtime policy update.
    candidate_preparer: Arc<ArcSwapOption<CandidatePreparer>>,
    /// One per command or wakeup. A request enqueued in an earlier turn rebuilds its rows.
    turn: u64,
    /// Rows prepared for the request enqueued this turn, dropped when the turn ends.
    turn_prepared: Option<Box<PreparedAdmission>>,
    projected_loads: FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
    /// Request-agnostic load snapshot the class-queue gate judges `has_room` on. Refilled in
    /// place once per gate pass; the actor never allocates it per admission.
    gate_loads: FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
    /// Ranks the current request's load projection found past the host's reject thresholds,
    /// counted per worker. Refilled by `with_projected_loads`; empty while nothing is full.
    full_ranks: FullRankCounts,
    block_size: u32,
    selector: Sel,
    prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    overlap_scores_refresh: Option<Arc<RF>>,
    overlap_refresh_after: Option<Duration>,
    overloaded_worker_provider: Option<OverloadedWorkerProvider>,
    available_worker_provider: Option<WorkerAvailabilityProvider>,
    non_max_overlap_selection_observer: Arc<OnceLock<NonMaxOverlapSelectionObserver>>,
    /// Per-candidate cost capture for admitted selections, installed once by the host.
    selection_trace: Arc<OnceLock<SelectionTraceSpec>>,
    reject_thresholds: Option<RejectThresholdsProvider>,
}

/// Queue that gates scheduling requests behind a capacity check.
/// When all workers exceed `threshold_frac` utilisation the request is parked in `pending`.
/// When capacity frees up (`update()`), pending requests are scheduled in priority order.
/// If queueing is disabled (threshold_frac is None), requests are scheduled immediately.
pub struct SchedulerQueue<
    P: SequencePublisher,
    C: WorkerConfigLike,
    Sel: WorkerSelector<C> = dynamo_worker_selection::selector::WorkerSelectionPolicy,
    RF: OverlapScoresRefresh = NoopOverlapScoresRefresh,
> {
    admission_tx: mpsc::Sender<AdmissionCommand>,
    runtime_policy: Arc<RuntimePolicyControl>,
    cleanup: Arc<AdmissionCleanup>,
    capacity_updates: Option<Arc<CapacityUpdates>>,
    direct_release: DirectBookingRelease,
    /// Number of requests currently parked in the pending queue.
    /// Incremented after push, decremented after pop. Lock-free reads via `Relaxed` load.
    pending_count: Arc<AtomicUsize>,
    /// Sum of `isl_tokens` for requests currently parked in the pending queue.
    /// Incremented after push, decremented after pop. Lock-free reads via `Relaxed` load.
    pending_isl_tokens: Arc<AtomicUsize>,
    class_counters: Arc<Vec<ClassQueueCounters>>,
    workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
    workers_generation: Arc<AtomicU64>,
    candidate_preparer: Arc<ArcSwapOption<CandidatePreparer>>,
    exclusive_affinity: bool,
    block_size: u32,
    profile: Arc<PolicyProfile>,
    start_time: Instant,
    available_worker_provider: Option<WorkerAvailabilityProvider>,
    overloaded_worker_provider: Option<OverloadedWorkerProvider>,
    supports_overlap_refresh: bool,
    non_max_overlap_selection_observer: Arc<OnceLock<NonMaxOverlapSelectionObserver>>,
    /// Per-candidate cost capture for admitted selections, installed once by the host.
    selection_trace: Arc<OnceLock<SelectionTraceSpec>>,
    reject_thresholds: Option<RejectThresholdsProvider>,
    #[allow(clippy::type_complexity)]
    // Covariant type markers, without ownership or auto-trait bounds.
    _marker: PhantomData<fn() -> (P, C, Sel, RF)>,
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueue<P, C, Sel, RF>
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        slots: Arc<ActiveSequencesMultiWorker<P>>,
        workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
        profile: PolicyProfile,
        block_size: u32,
        selector: Sel,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        overlap_scores_refresh: Option<Arc<RF>>,
        overloaded_worker_provider: Option<OverloadedWorkerProvider>,
        available_worker_provider: Option<WorkerAvailabilityProvider>,
        reject_thresholds: Option<RejectThresholdsProvider>,
    ) -> Self {
        Self::new_with_capacity(
            slots,
            workers_with_configs,
            profile,
            block_size,
            selector,
            prefill_load_estimator,
            overlap_scores_refresh,
            overloaded_worker_provider,
            available_worker_provider,
            reject_thresholds,
            ADMISSION_CHANNEL_CAPACITY,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_capacity(
        slots: Arc<ActiveSequencesMultiWorker<P>>,
        workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
        profile: PolicyProfile,
        block_size: u32,
        selector: Sel,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        overlap_scores_refresh: Option<Arc<RF>>,
        overloaded_worker_provider: Option<OverloadedWorkerProvider>,
        available_worker_provider: Option<WorkerAvailabilityProvider>,
        reject_thresholds: Option<RejectThresholdsProvider>,
        admission_channel_capacity: usize,
    ) -> Self {
        let profile = Arc::new(profile);
        let pending = PolicyQueue::new(&profile);
        let queueing_enabled = profile
            .classes()
            .iter()
            .any(PolicyClassConfig::queueing_enabled);
        for class in profile.classes() {
            tracing::info!(
                policy_class = class.name,
                queue_policy = %class.queue_policy,
                quantum = class.quantum,
                prefill_busy_threshold = ?class.busy_thresholds.prefill_busy_threshold,
                prefill_busy_threshold_frac = ?class.busy_thresholds.prefill_busy_threshold_frac,
                "Router policy class configured"
            );
        }
        let overlap_refresh_after = if overlap_scores_refresh.is_some() {
            let configured = read_overlap_refresh_after();
            match configured {
                Some(d) => tracing::info!(
                    "Router queue overlap-score refresh enabled after {:.1}s wait",
                    d.as_secs_f64()
                ),
                None => tracing::info!(
                    "Router queue overlap-score refresh disabled via DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS"
                ),
            }
            configured
        } else {
            None
        };
        let pending_count = Arc::new(AtomicUsize::new(0));
        let pending_isl_tokens = Arc::new(AtomicUsize::new(0));
        let class_counters = Arc::new(
            profile
                .classes()
                .iter()
                .map(|_| ClassQueueCounters {
                    pending_count: AtomicUsize::new(0),
                    pending_isl_tokens: AtomicUsize::new(0),
                    pending_cached_tokens: AtomicUsize::new(0),
                    received_total: AtomicU64::new(0),
                    rejected_due_time_passed_total: AtomicU64::new(0),
                })
                .collect(),
        );
        let (admission_tx, admission_rx) = mpsc::channel(admission_channel_capacity);
        let runtime_policy = Arc::new(RuntimePolicyControl {
            tx: admission_tx.downgrade(),
        });
        let cleanup = Arc::new(AdmissionCleanup::default());
        let non_max_overlap_selection_observer = Arc::new(OnceLock::new());
        let selection_trace = Arc::new(OnceLock::new());
        let start_time = Instant::now();
        let capacity_updates = queueing_enabled.then(|| Arc::new(CapacityUpdates::new()));
        let direct_release: DirectBookingRelease = {
            let slots = Arc::clone(&slots);
            let capacity_updates = capacity_updates.clone();
            Arc::new(move |booking| {
                let outcome = slots.free_if_booking(
                    &booking.request_id,
                    booking.worker,
                    booking.attempt_id,
                    Instant::now(),
                )?;
                if outcome.is_applied()
                    && let Some(updates) = &capacity_updates
                {
                    updates.record(Some(booking.worker));
                }
                Ok(())
            })
        };
        let workers_generation = Arc::new(AtomicU64::new(0));
        let candidate_preparer = Arc::new(ArcSwapOption::from(
            selector.candidate_preparer().map(Arc::new),
        ));
        let exclusive_affinity = selector.uses_exclusive_affinity_target();
        let actor = SchedulerQueueActor {
            pending,
            cleanup: Arc::clone(&cleanup),
            capacity_updates: capacity_updates.clone(),
            capacity_workers: FxHashSet::default(),
            profile: Arc::clone(&profile),
            pending_count: Arc::clone(&pending_count),
            pending_isl_tokens: Arc::clone(&pending_isl_tokens),
            class_counters: Arc::clone(&class_counters),
            slots,
            workers_with_configs: workers_with_configs.clone(),
            workers_generation: Arc::clone(&workers_generation),
            candidate_preparer: Arc::clone(&candidate_preparer),
            turn: 0,
            turn_prepared: None,
            projected_loads: FxHashMap::default(),
            gate_loads: FxHashMap::default(),
            full_ranks: FullRankCounts::default(),
            block_size,
            selector,
            prefill_load_estimator,
            overlap_scores_refresh,
            overlap_refresh_after,
            overloaded_worker_provider: overloaded_worker_provider.clone(),
            available_worker_provider: available_worker_provider.clone(),
            non_max_overlap_selection_observer: Arc::clone(&non_max_overlap_selection_observer),
            selection_trace: Arc::clone(&selection_trace),
            reject_thresholds: reject_thresholds.clone(),
        };
        tokio::spawn(actor.run(admission_rx));
        Self {
            admission_tx,
            runtime_policy,
            cleanup,
            capacity_updates,
            direct_release,
            pending_count,
            pending_isl_tokens,
            class_counters,
            workers_with_configs,
            workers_generation,
            candidate_preparer,
            exclusive_affinity,
            block_size,
            profile,
            start_time,
            available_worker_provider,
            overloaded_worker_provider,
            supports_overlap_refresh: overlap_refresh_after.is_some(),
            non_max_overlap_selection_observer,
            selection_trace,
            reject_thresholds,
            _marker: PhantomData,
        }
    }
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueue<P, C, Sel, RF>
{
    /// Install the observer for admitted selections that sacrifice KV overlap.
    ///
    /// Returns `false` when an observer is already installed.
    pub fn set_non_max_overlap_selection_observer(
        &self,
        observer: NonMaxOverlapSelectionObserver,
    ) -> bool {
        self.non_max_overlap_selection_observer
            .set(observer)
            .is_ok()
    }

    /// Install per-candidate cost capture for admitted selections.
    ///
    /// Returns `false` when a spec is already installed.
    pub fn set_selection_trace(&self, spec: SelectionTraceSpec) -> bool {
        self.selection_trace.set(spec).is_ok()
    }

    /// Enqueue a new request through the router-owned policy queue.
    ///
    /// Every request enters the policy queue and the actor drains the ready
    /// backlog in the same turn, so an uncontended request is admitted before
    /// the enqueue acknowledgement returns. Two direct-admission carve-outs
    /// bypass the queue: classes with any zero per-worker limit (which can
    /// never hold a queued entry) and windows with zero discovered workers
    /// (so selection reports `NoEndpoints` instead of a scaled-to-zero
    /// queue-limit rejection).
    pub async fn enqueue(&self, request: SchedulingRequest) {
        self.enqueue_with_block_hashes(request, None).await;
    }

    pub async fn enqueue_with_block_hashes(
        &self,
        request: SchedulingRequest,
        block_hashes: Option<Vec<LocalBlockHash>>,
    ) {
        let _ = self
            .enqueue_with_block_hashes_and_lease(request, block_hashes, None)
            .await;
    }

    #[doc(hidden)]
    pub async fn enqueue_with_block_hashes_and_lease(
        &self,
        request: SchedulingRequest,
        block_hashes: Option<Vec<LocalBlockHash>>,
        lease: Option<Box<RequestLifecycleLease>>,
    ) -> Option<Box<RequestLifecycleLease>> {
        self.enqueue_admitted_with_block_hashes_and_lease(
            request,
            block_hashes,
            lease,
            None,
            None,
            Instant::now(),
        )
        .await
    }

    #[doc(hidden)]
    pub async fn enqueue_admitted_with_block_hashes_and_lease(
        &self,
        mut request: SchedulingRequest,
        block_hashes: Option<Vec<LocalBlockHash>>,
        lease: Option<Box<RequestLifecycleLease>>,
        attempt_tx: Option<oneshot::Sender<AttemptId>>,
        classified_request: Option<ClassifyRequest>,
        ingress_at: Instant,
    ) -> Option<Box<RequestLifecycleLease>> {
        if self.capacity_updates.is_some()
            && lease.is_none()
            && request.mode.lifecycle_request_id().is_some()
        {
            request.respond(Err(KvSchedulerError::BookingFailed(
                "admission-managed requests must be scheduled through LocalScheduler".to_string(),
            )));
            return None;
        }

        let queue_metadata = match classified_request {
            Some(classified) => self.validate_classification(&mut request, classified, ingress_at),
            None => Ok(self.default_queue_metadata(&request, ingress_at)),
        };
        let queue_metadata = match queue_metadata {
            Ok(metadata) => metadata,
            Err(error) => {
                request.respond(Err(error));
                return None;
            }
        };

        let eligibility = request.eligibility();

        if let Err(error) = eligibility.validate_pinned_worker_allowed() {
            request.respond(Err(error));
            return None;
        }

        let prepared = self.prepare_candidates(&request);
        let (ack_tx, ack_rx) = oneshot::channel();
        let command = AdmissionCommand::Enqueue {
            request,
            attempt_tx,
            block_hashes: self.prepare_block_hashes_for_refresh(block_hashes),
            prepared,
            queue_metadata,
            lease,
            ack_tx,
        };

        if let Err(error) = self.admission_tx.send(command).await {
            let AdmissionCommand::Enqueue { mut request, .. } = error.0 else {
                return None;
            };
            request.respond(Err(KvSchedulerError::SubscriberShutdown));
            return None;
        }

        match ack_rx.await {
            Ok(lease) => lease,
            Err(_) => {
                tracing::warn!("scheduler queue actor dropped enqueue acknowledgement");
                None
            }
        }
    }

    /// Build the classifier's input and the rows for its declared worker inputs. Uses
    /// `request.isl_tokens` — the routing-token basis the queue's bucketing, limits, and DRR
    /// cost already use — so a pass-through classifier is behavior-identical to no
    /// classifier.
    #[doc(hidden)]
    pub fn build_classify_request(
        &self,
        request: &SchedulingRequest,
        ingress_at: Instant,
        worker_inputs: WorkerInputs,
    ) -> (ClassifyRequest, WorkerInputRows) {
        let available = self
            .available_worker_provider
            .as_ref()
            .and_then(|provider| provider(request));
        let workers = self.workers_with_configs.borrow();
        let cached_tokens = SchedulingContext::new(request, &workers)
            .with_available_workers(available.as_deref())
            .best_cached_tokens();
        let reject_thresholds = self
            .reject_thresholds
            .as_ref()
            .map(|provider| provider())
            .unwrap_or_default();
        let mut classification =
            ClassifyRequest::with_timing(request.isl_tokens, cached_tokens, ingress_at)
                .with_pinned_worker(request.pinned_worker)
                .with_sequence_hashes(request.token_seq.as_deref())
                .with_busy_thresholds(Arc::clone(&self.profile), reject_thresholds);
        if let Some(request_id) = request.mode.request_id() {
            classification = classification.with_request_id(request_id);
        }
        if let Some(policy_class) = request.policy_class.as_deref() {
            classification = classification.with_initial_policy_class(policy_class);
        }
        if let Some(session_context) = request.session_context.clone() {
            classification = classification.with_session_context(session_context);
        }
        let rows = if worker_inputs == WorkerInputs::NONE {
            WorkerInputRows::empty()
        } else {
            let eligibility = request
                .eligibility()
                .with_available_workers(available.as_deref());
            worker_input_rows(&workers, request, eligibility, worker_inputs)
        };
        (classification, rows)
    }

    /// Enqueue a request with `due_at` for actor-expiry tests.
    #[cfg(test)]
    #[doc(hidden)]
    pub async fn enqueue_with_due_at_for_test(&self, request: SchedulingRequest, due_at: Instant) {
        let mut queue_metadata = self.default_queue_metadata(&request, Instant::now());
        queue_metadata.due_at = Some(due_at);
        let (ack_tx, ack_rx) = oneshot::channel();
        let command = AdmissionCommand::Enqueue {
            request,
            attempt_tx: None,
            block_hashes: None,
            prepared: None,
            queue_metadata,
            lease: None,
            ack_tx,
        };
        self.admission_tx
            .send(command)
            .await
            .expect("scheduler queue actor gone");
        let _ = ack_rx.await;
    }

    fn default_queue_metadata(
        &self,
        request: &SchedulingRequest,
        ingress_at: Instant,
    ) -> QueueMetadata {
        let available = self
            .available_worker_provider
            .as_ref()
            .and_then(|provider| provider(request));
        let workers = self.workers_with_configs.borrow();
        let snapshot = QueueSnapshot::new(
            request.isl_tokens,
            SchedulingContext::new(request, &workers)
                .with_available_workers(available.as_deref())
                .best_cached_tokens(),
        );
        let class_index = self
            .profile
            .resolve_class_index(request.policy_class.as_deref(), snapshot.uncached_tokens);
        QueueMetadata {
            class_index,
            snapshot,
            due_at: None,
            // "Arrival" is the router ingress time, stamped before the
            // admission channel — one basis for classified and unclassified
            // requests alike, so time spent in a classifier does not reorder
            // FCFS/LCFS. Two racing callers can still enqueue in the opposite
            // order of their offsets; `enqueue_seq` stays the authoritative
            // FCFS tiebreak for same-score entries.
            arrival_offset_secs: ingress_at
                .saturating_duration_since(self.start_time)
                .as_secs_f64(),
        }
    }

    fn validate_classification(
        &self,
        request: &mut SchedulingRequest,
        classified_request: ClassifyRequest,
        ingress_at: Instant,
    ) -> Result<QueueMetadata, KvSchedulerError> {
        let ClassificationOverrides {
            policy_class,
            due_at,
            scheduling_cost_tokens,
            worker_selection_target,
        } = classified_request.into_queue_inputs();

        if scheduling_cost_tokens == Some(0) {
            return Err(KvSchedulerError::InvalidClassificationMetadata(
                "scheduling cost must be greater than zero".to_string(),
            ));
        }
        // An already-expired `due_at` is not validated here: the actor is the
        // single deadline authority and rejects it at enqueue on its own clock.

        // Hard pins must not acquire a conflicting soft target. Apply a permitted
        // override before recomputing cache eligibility for the resulting target.
        if request.pinned_worker.is_none()
            && let Some(target) = worker_selection_target
        {
            request.affinity_target = target;
        }

        // Queue inputs are recomputed from the current workers, exactly as the
        // default path computes them: worker state may have changed while the
        // classification was pending, and only explicit overrides survive.
        let mut metadata = self.default_queue_metadata(request, ingress_at);
        if let Some(policy_class) = policy_class {
            metadata.class_index = self
                .profile
                .resolve_class_index_strict(&policy_class, metadata.snapshot.uncached_tokens)
                .ok_or_else(|| {
                    KvSchedulerError::InvalidClassificationMetadata(format!(
                        "unknown policy class {policy_class:?}"
                    ))
                })?;
        }
        if let Some(cost) = scheduling_cost_tokens {
            metadata.snapshot.scheduling_cost_tokens = cost;
        }
        metadata.due_at = due_at;
        Ok(metadata)
    }

    #[doc(hidden)]
    pub fn new_request_lifecycle_lease(
        &self,
        request_id: Option<&str>,
    ) -> Option<Box<RequestLifecycleLease>> {
        let request_id = request_id?;
        Some(Box::new(RequestLifecycleLease {
            cleanup: Arc::clone(&self.cleanup),
            actor_tx: self.admission_tx.clone(),
            transfer: Some(Arc::new(AdmissionLifecycleTransfer::new(
                request_id.to_string(),
            ))),
        }))
    }

    pub fn booking_cleanup(&self) -> SchedulerBookingCleanup {
        SchedulerBookingCleanup {
            cleanup: Arc::clone(&self.cleanup),
            actor_tx: self.admission_tx.clone(),
            direct_release: Arc::clone(&self.direct_release),
        }
    }

    /// An armed handle for an existing booking.
    #[doc(hidden)]
    pub fn booking_handle(&self, booking: SchedulerBookingDescriptor) -> BookingHandle {
        BookingHandle {
            booking,
            cleanup: self.booking_cleanup(),
            armed: true,
        }
    }

    /// Select a worker from current scheduler state without entering admission.
    ///
    /// This is for advisory policy probes that must not wait in the router
    /// queue and must not book active scheduler state.
    pub async fn select_without_admission(
        &self,
        request: SchedulingRequest,
    ) -> Result<AdvisorySchedulingResponse, KvSchedulerError> {
        request.eligibility().validate_pinned_worker_allowed()?;

        let (resp_tx, resp_rx) = oneshot::channel();
        let command = AdmissionCommand::SelectWithoutAdmission { request, resp_tx };
        if self.admission_tx.send(command).await.is_err() {
            return Err(KvSchedulerError::SubscriberShutdown);
        }

        resp_rx
            .await
            .map_err(|_| KvSchedulerError::SubscriberShutdown)?
    }

    /// Wait for a full admission recheck when queueing is enabled and the actor is running.
    /// With queueing disabled, return immediately without an admission barrier.
    pub async fn update(&self) {
        if self.capacity_updates.is_none() {
            return;
        }

        let _ = self.admission_barrier().await;
    }

    /// Process previously submitted admission commands and recheck ready work.
    ///
    /// Unlike [`Self::update`], this waits for the actor even when queueing is
    /// disabled. It does not wait for capacity-blocked requests to be admitted,
    /// or submit scheduling futures that the caller has not polled yet.
    /// Returns an error if the actor stops before acknowledging the barrier.
    pub async fn admission_barrier(&self) -> Result<(), KvSchedulerError> {
        let (ack_tx, ack_rx) = oneshot::channel();
        self.admission_tx
            .send(AdmissionCommand::Update { ack_tx })
            .await
            .map_err(|_| KvSchedulerError::SubscriberShutdown)?;
        ack_rx
            .await
            .map_err(|_| KvSchedulerError::SubscriberShutdown)
    }

    pub fn policy_control(&self) -> &Arc<RuntimePolicyControl> {
        &self.runtime_policy
    }

    /// Notify pending admission after the authoritative mutation is complete.
    #[doc(hidden)]
    pub fn capacity_changed(&self, worker: Option<WorkerWithDpRank>) {
        if let Some(updates) = &self.capacity_updates {
            updates.record(worker);
        }
    }

    #[doc(hidden)]
    pub fn ensure_running(&self) -> Result<(), KvSchedulerError> {
        if self.admission_tx.is_closed() {
            Err(KvSchedulerError::SubscriberShutdown)
        } else {
            Ok(())
        }
    }

    /// Number of requests currently parked in the pending queue (lock-free).
    pub fn pending_count(&self) -> usize {
        self.pending_count.load(AtomicOrdering::Relaxed)
    }

    /// Sum of `isl_tokens` for requests currently parked in the pending queue (lock-free).
    pub fn pending_isl_tokens(&self) -> usize {
        self.pending_isl_tokens.load(AtomicOrdering::Relaxed)
    }

    pub fn class_queue_stats(&self, class_index: usize) -> Option<ClassQueueStats> {
        let counters = self.class_counters.get(class_index)?;
        Some(ClassQueueStats {
            pending_count: counters.pending_count.load(AtomicOrdering::Relaxed),
            pending_isl_tokens: counters.pending_isl_tokens.load(AtomicOrdering::Relaxed),
            pending_cached_tokens: counters.pending_cached_tokens.load(AtomicOrdering::Relaxed),
            received_total: counters.received_total.load(AtomicOrdering::Relaxed),
            rejected_due_time_passed_total: counters
                .rejected_due_time_passed_total
                .load(AtomicOrdering::Relaxed),
        })
    }

    /// Build candidate rows on the caller task, leaving the actor only the
    /// load inputs that depend on its bookings.
    fn prepare_candidates(&self, request: &SchedulingRequest) -> Option<Box<PreparedAdmission>> {
        if request.pinned_worker.is_some() {
            return None;
        }
        let preparer = self.candidate_preparer.load().as_deref().copied()?;
        // Load before reading the configuration: a change after this load
        // advances the generation before the actor selects with these rows.
        let workers_generation = self.workers_generation.load(AtomicOrdering::Acquire);
        let available_worker_ids = self
            .available_worker_provider
            .as_ref()
            .and_then(|provider| provider(request));
        let overloaded_worker_ids = self
            .overloaded_worker_provider
            .as_ref()
            .and_then(|provider| provider());
        let workers = self.workers_with_configs.borrow();
        // The actor's load projection decides which ranks are full; rows are
        // built without that cut and used only when it found none.
        let eligibility = selection_eligibility(
            request,
            &workers,
            overloaded_worker_ids.as_ref(),
            available_worker_ids.as_deref(),
            None,
            self.exclusive_affinity,
        );
        let candidates = preparer.prepare(WorkerSelectionInput::configured(
            &workers,
            request,
            eligibility,
            self.block_size,
        ))?;
        Some(Box::new(PreparedAdmission {
            candidates,
            workers_generation,
            available_worker_ids,
            overloaded_worker_ids,
        }))
    }

    fn prepare_block_hashes_for_refresh(
        &self,
        block_hashes: Option<Vec<LocalBlockHash>>,
    ) -> Option<Vec<LocalBlockHash>> {
        if !self.supports_overlap_refresh {
            return None;
        }
        block_hashes.filter(|hashes| !hashes.is_empty())
    }
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueueActor<P, C, Sel, RF>
{
    async fn run(mut self, mut rx: mpsc::Receiver<AdmissionCommand>) {
        let mut commands_since_cleanup = 0usize;
        let capacity_updates = self.capacity_updates.clone();
        loop {
            self.turn = self.turn.wrapping_add(1);
            let due_at = self.pending.next_due_at();
            let command = if capacity_updates.is_none() && due_at.is_none() {
                rx.recv().await
            } else {
                tokio::select! {
                    command = rx.recv() => command,
                    _ = async {
                        if let Some(updates) = &capacity_updates {
                            updates.notified().await;
                        }
                    }, if capacity_updates.is_some() => {
                        let drain_cleanup = rx.is_empty() || commands_since_cleanup == 256;
                        if drain_cleanup {
                            commands_since_cleanup = 0;
                        }
                        self.handle_capacity_updates(drain_cleanup).await;
                        continue;
                    }
                    _ = async {
                        if let Some(due_at) = due_at {
                            tokio::time::sleep_until(due_at).await;
                        }
                    }, if due_at.is_some() => {
                        self.handle_update().await;
                        continue;
                    }
                }
            };
            let Some(command) = command else { break };
            commands_since_cleanup += 1;
            let drain_cleanup = rx.is_empty() || commands_since_cleanup == 256;
            if drain_cleanup {
                commands_since_cleanup = 0;
            }
            match command {
                AdmissionCommand::Enqueue {
                    request,
                    attempt_tx,
                    block_hashes,
                    prepared,
                    queue_metadata,
                    lease,
                    ack_tx,
                } => {
                    let lifecycle_transfer = lease
                        .as_ref()
                        .and_then(|lease| lease.transfer.as_ref().map(Arc::clone));
                    let enqueue_ready = self.handle_enqueue(
                        request,
                        attempt_tx,
                        lifecycle_transfer,
                        block_hashes,
                        prepared,
                        queue_metadata,
                    );
                    let cleanup_ready = drain_cleanup && self.drain_cleanup();
                    if cleanup_ready {
                        self.handle_update().await;
                    } else if enqueue_ready {
                        self.handle_enqueued().await;
                    }
                    // A request still queued rebuilds its rows when admitted.
                    self.turn_prepared = None;
                    let _ = ack_tx.send(lease);
                }
                AdmissionCommand::SelectWithoutAdmission { request, resp_tx } => {
                    let result = self.select_without_admission_inner(request, Instant::now());
                    if drain_cleanup && self.drain_cleanup() {
                        self.handle_update().await;
                    }
                    let _ = resp_tx.send(result);
                }
                AdmissionCommand::PolicyConfig { update, response } => {
                    // A cancelled or timed-out caller must not leave a queued mutation behind.
                    if !response.is_closed() {
                        let result = match update {
                            Some(update) => {
                                let result = self.selector.update_runtime_config(update);
                                if result.is_ok() {
                                    // A replacement may need fewer inputs; prepare what it reads.
                                    self.candidate_preparer
                                        .store(self.selector.candidate_preparer().map(Arc::new));
                                }
                                result
                            }
                            None => self.selector.runtime_config(),
                        };
                        let _ = response.send(result);
                    }
                    if drain_cleanup && self.drain_cleanup() {
                        self.handle_update().await;
                    }
                }
                AdmissionCommand::Update { ack_tx } => {
                    self.handle_update().await;
                    if drain_cleanup && self.drain_cleanup() {
                        self.handle_update().await;
                    }
                    let _ = ack_tx.send(());
                }
                AdmissionCommand::Cleanup => {
                    if self.drain_cleanup() {
                        self.handle_update().await;
                    }
                }
            }
        }
        self.drain_cleanup();

        let class_counters = Arc::clone(&self.class_counters);
        for entry in self.pending.drain() {
            let class_index = entry.class_index();
            let snapshot = entry.snapshot();
            self.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
            self.pending_isl_tokens
                .fetch_sub(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
            let counters = &class_counters[class_index];
            counters.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
            counters
                .pending_isl_tokens
                .fetch_sub(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
            counters
                .pending_cached_tokens
                .fetch_sub(snapshot.cached_tokens, AtomicOrdering::Relaxed);

            let mut request = entry.into_payload().request;
            request.respond(Err(KvSchedulerError::SubscriberShutdown));
        }
    }

    fn handle_enqueue(
        &mut self,
        mut request: SchedulingRequest,
        attempt_tx: Option<oneshot::Sender<AttemptId>>,
        lifecycle_transfer: Option<Arc<AdmissionLifecycleTransfer>>,
        block_hashes: Option<Vec<LocalBlockHash>>,
        prepared: Option<Box<PreparedAdmission>>,
        queue_metadata: QueueMetadata,
    ) -> bool {
        let class_index = queue_metadata.class_index;
        self.class_counters[class_index]
            .received_total
            .fetch_add(1, AtomicOrdering::Relaxed);
        let decay_now = Instant::now();
        if queue_metadata
            .due_at
            .is_some_and(|due_at| due_at <= decay_now)
        {
            self.reject_due_time_passed(class_index, &mut request);
            return false;
        }
        let snapshot = queue_metadata.snapshot;
        let class = self.profile.class(class_index);
        // With zero discovered workers every per-worker limit scales to zero and
        // `queue_rejection` would reject with `limit: 0`. Admit directly instead
        // so selection reports the real condition, `NoEndpoints`.
        if self.workers_with_configs.borrow().is_empty() {
            return self.admit_one(request, attempt_tx, lifecycle_transfer, decay_now, prepared);
        }
        // A class with any zero per-worker limit never queues: keep the direct
        // admission semantics by admitting while a worker has prefill capacity,
        // and otherwise fall through so `queue_rejection` produces the same
        // limit-zero rejection the queued path always has.
        if class.never_queues() {
            let available = self
                .available_worker_provider
                .as_ref()
                .and_then(|provider| provider(&request));
            let should_queue = class.queueing_enabled()
                && (self.pending.has_backlog(class_index) || {
                    self.slots.project_worker_loads_into(
                        None,
                        decay_now,
                        &mut self.gate_loads,
                        None,
                    );
                    let configs = self.workers_with_configs.borrow();
                    Self::all_eligible_workers_full(
                        &self.gate_loads,
                        &configs,
                        class.busy_thresholds,
                        request
                            .eligibility()
                            .with_available_workers(available.as_deref()),
                    )
                });
            if !should_queue {
                return self.admit_one(
                    request,
                    attempt_tx,
                    lifecycle_transfer,
                    decay_now,
                    prepared,
                );
            }
        }
        tracing::trace!(policy_class = class.name, "ordering request");
        let priority_jump = request.priority_jump;
        let strict_priority = request.strict_priority;
        let placement = request
            .pinned_worker
            .map_or(WorkerPlacement::Any, WorkerPlacement::Exact);
        let queued = QueuedRequest {
            request,
            attempt_tx,
            lifecycle_transfer: lifecycle_transfer.clone(),
            enqueue_at: decay_now,
            due_at: queue_metadata.due_at,
            block_hashes,
            enqueue_turn: self.turn,
        };
        self.turn_prepared = prepared;
        let worker_count = self.workers_with_configs.borrow().len();
        if let Err((rejection, queued)) = self.pending.enqueue_with_due_at(
            queue_metadata,
            worker_count,
            priority_jump,
            strict_priority,
            placement,
            queued,
        ) {
            let mut request = queued.request;
            request.respond(Err(KvSchedulerError::QueueRejected(rejection)));
            return false;
        }
        if let Some(transfer) = lifecycle_transfer {
            transfer.arm_pending();
        }
        self.pending_count.fetch_add(1, AtomicOrdering::Relaxed);
        self.pending_isl_tokens
            .fetch_add(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
        self.add_class_counters(class_index, snapshot);
        true
    }

    fn reject_due_time_passed(&self, class_index: usize, request: &mut SchedulingRequest) {
        self.class_counters[class_index]
            .rejected_due_time_passed_total
            .fetch_add(1, AtomicOrdering::Relaxed);
        request.respond(Err(KvSchedulerError::DeadlineExceeded));
    }

    fn reject_expired(&mut self, now: Instant) {
        for entry in self.pending.take_expired(now) {
            let class_index = entry.class_index();
            self.subtract_pending_counters(class_index, entry.snapshot());
            let mut request = entry.into_payload().request;
            self.reject_due_time_passed(class_index, &mut request);
        }
    }

    fn drain_cleanup(&mut self) -> bool {
        let dirty = self.cleanup.drain();
        if dirty.is_empty() {
            return false;
        }

        let mut made_ready = false;
        let mut removed_ready_head = false;
        let mut unmanaged_request_ids = HashSet::new();
        for cleanup in dirty {
            match cleanup.target {
                SchedulerCleanupTarget::AdmissionLifecycle(transfer) => {
                    let result = match transfer.cleanup_target() {
                        Some(AdmissionLifecycleCleanup::Pending { request_id }) => {
                            unmanaged_request_ids.insert(request_id);
                            Ok(())
                        }
                        Some(AdmissionLifecycleCleanup::Booking(booking)) => self
                            .slots
                            .free_if_booking(
                                &booking.request_id,
                                booking.worker,
                                booking.attempt_id,
                                Instant::now(),
                            )
                            .map(|outcome| {
                                made_ready |= outcome.is_applied();
                            }),
                        None => Ok(()),
                    };
                    if let Err(error) = result {
                        tracing::error!(%error, "Failed to release scheduler admission lifecycle");
                    }
                }
                SchedulerCleanupTarget::Booking(booking) => {
                    let result = self
                        .slots
                        .free_if_booking(
                            &booking.request_id,
                            booking.worker,
                            booking.attempt_id,
                            Instant::now(),
                        )
                        .map(|outcome| {
                            made_ready |= outcome.is_applied();
                        });
                    if let Err(error) = result {
                        tracing::error!(
                            request_id = %booking.request_id,
                            worker = ?booking.worker,
                            attempt_id = %booking.attempt_id,
                            %error,
                            "Failed to release request-attempt scheduler booking"
                        );
                    }
                }
                SchedulerCleanupTarget::ExpiredBooking(booking) => {
                    let result = self
                        .slots
                        .expire_if_booking(
                            &booking.request_id,
                            booking.worker,
                            booking.attempt_id,
                            Instant::now(),
                        )
                        .map(|outcome| {
                            made_ready |= outcome.is_applied();
                        });
                    if let Err(error) = result {
                        tracing::error!(
                            request_id = %booking.request_id,
                            worker = ?booking.worker,
                            attempt_id = %booking.attempt_id,
                            %error,
                            "Failed to expire request-attempt scheduler booking"
                        );
                    }
                }
            }
        }
        if !unmanaged_request_ids.is_empty() {
            for class_index in 0..self.profile.classes().len() {
                let (removed, class_head_removed) =
                    self.pending.take_if_in_class(class_index, |queued| {
                        queued
                            .request
                            .mode
                            .tracked_request_id()
                            .is_some_and(|request_id| unmanaged_request_ids.contains(request_id))
                    });
                removed_ready_head |= class_head_removed;
                for entry in removed {
                    self.subtract_pending_counters(class_index, entry.snapshot());
                }
            }
        }
        made_ready || (removed_ready_head && self.has_dispatchable_ready_head())
    }

    fn has_dispatchable_ready_head(&mut self) -> bool {
        let decay_now = Instant::now();
        let slots = &self.slots;
        let provider = self.available_worker_provider.as_ref();
        let workers = &self.workers_with_configs;
        let gate_loads = &mut self.gate_loads;
        let mut gate_fresh = false;
        self.pending.any_ready_head(|_, class, queued| {
            if !class.queueing_enabled() {
                return true;
            }
            if !gate_fresh {
                slots.project_worker_loads_into(None, decay_now, gate_loads, None);
                gate_fresh = true;
            }
            let available = provider.and_then(|provider| provider(&queued.request));
            let configs = workers.borrow();
            !Self::all_eligible_workers_full(
                gate_loads,
                &configs,
                class.busy_thresholds,
                queued
                    .request
                    .eligibility()
                    .with_available_workers(available.as_deref()),
            )
        })
    }

    fn subtract_pending_counters(&self, class_index: usize, snapshot: QueueSnapshot) {
        self.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
        self.pending_isl_tokens
            .fetch_sub(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
        self.subtract_class_counters(class_index, snapshot);
    }

    /// Consume recorded capacity changes before choosing among shared and pinned lanes.
    fn recheck_capacity_updates(&mut self, mut recheck_all: bool) {
        if let Some(updates) = &self.capacity_updates {
            recheck_all |= updates.drain_into(&mut self.capacity_workers);
        }
        // Consume hints even when there is no pending work. Blocked pinned lanes
        // count as ready here and still need their capacity rechecked.
        if !self.pending.has_ready() {
            return;
        }
        if !recheck_all {
            let configs = self.workers_with_configs.borrow();
            recheck_all = self
                .capacity_workers
                .iter()
                .any(|&worker| worker_config_for_rank(&*configs, worker).is_err());
        }
        if recheck_all {
            self.pending.recheck_all_workers();
        } else {
            for &worker in &self.capacity_workers {
                self.pending.recheck_worker(worker);
            }
        }
    }

    async fn handle_capacity_updates(&mut self, drain_cleanup: bool) {
        self.reject_expired(Instant::now());
        let cleanup_ready = drain_cleanup && self.drain_cleanup();
        self.recheck_capacity_updates(cleanup_ready);
        if !self.pending.has_ready() {
            return;
        }
        self.drain_ready().await;
    }

    async fn handle_update(&mut self) {
        self.reject_expired(Instant::now());
        self.recheck_capacity_updates(true);
        if !self.pending.has_ready() {
            return;
        }
        self.drain_ready().await;
    }

    /// Recheck recorded capacity changes before an enqueue drains the queue, so
    /// an older blocked pinned lane competes with later shared arrivals.
    async fn handle_enqueued(&mut self) {
        self.reject_expired(Instant::now());
        self.recheck_capacity_updates(false);
        if !self.pending.has_ready() {
            return;
        }
        self.drain_ready().await;
    }

    async fn drain_ready(&mut self) {
        // Continuation draining stays actor-local; never self-send through the
        // bounded command channel while processing an update.
        loop {
            if !self.pending.has_ready() {
                break;
            }
            let decay_now = Instant::now();
            // Share a capacity snapshot within this pass only. The next admission
            // must observe the capacity reserved by the previous one.
            let popped = {
                let slots = &self.slots;
                let provider = self.available_worker_provider.as_ref();
                let workers = &self.workers_with_configs;
                let gate_loads = &mut self.gate_loads;
                let mut gate_fresh = false;
                self.pending.pop_next(|_, class, queued| {
                    if !class.queueing_enabled() {
                        return true;
                    }
                    if !gate_fresh {
                        slots.project_worker_loads_into(None, decay_now, gate_loads, None);
                        gate_fresh = true;
                    }
                    let available = provider.and_then(|provider| provider(&queued.request));
                    let configs = workers.borrow();
                    // TODO: This preserves head-of-line blocking within each policy
                    // class. A blocked constrained head can stall later entries in
                    // that class until a bounded non-HOL policy is introduced.
                    !Self::all_eligible_workers_full(
                        gate_loads,
                        &configs,
                        class.busy_thresholds,
                        queued
                            .request
                            .eligibility()
                            .with_available_workers(available.as_deref()),
                    )
                })
            };
            let Some(mut popped) = popped else {
                break;
            };
            let snapshot = popped.snapshot();
            let current_pending_count = self.pending_count.load(AtomicOrdering::Relaxed);
            debug_assert!(
                current_pending_count > 0,
                "pending_count underflow on queue drain"
            );
            self.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
            let current_pending_isl_tokens = self.pending_isl_tokens.load(AtomicOrdering::Relaxed);
            debug_assert!(
                current_pending_isl_tokens >= snapshot.raw_isl_tokens,
                "pending_isl_tokens underflow: pending={} request_isl_tokens={}",
                current_pending_isl_tokens,
                snapshot.raw_isl_tokens
            );
            self.pending_isl_tokens
                .fetch_sub(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
            self.subtract_class_counters(popped.class_index(), snapshot);
            let class_index = popped.class_index();
            let queued = popped.payload_mut();
            // NOTE: Overlap refresh is expected to be very short. We intentionally
            // accept load crossing the class threshold during this await: busy
            // thresholds guide admission, not reservation. This differs from main
            // to avoid reversing counters, heap state, and charged DRR credit.
            let refreshed = refresh_overlap(
                self.overlap_scores_refresh.as_deref(),
                self.overlap_refresh_after,
                queued.block_hashes.as_deref(),
                queued.request.retain_kv_transfer_chain,
                queued.enqueue_at,
                decay_now,
            )
            .await;
            // Deadlines can pass while the refresh is awaited and the actor
            // timer cannot fire mid-command. Sweep the still-queued entries so
            // a stalled refresh delays their rejection by at most one refresh,
            // and an expired entry is never popped into a refresh of its own.
            self.reject_expired(Instant::now());
            if queued.due_at.is_some_and(|due_at| due_at <= Instant::now()) {
                // The pop already charged this entry's scheduling cost against
                // the class deficit; the credit is intentionally not refunded,
                // matching every other post-pop terminal outcome.
                let mut request = popped.into_payload().request;
                self.reject_due_time_passed(class_index, &mut request);
                continue;
            }
            let wait_ms = queued.enqueue_at.elapsed().as_millis() as u64;
            // Only this turn's request has rows, and a refresh replaces the overlap
            // they were built from.
            let prepared = if queued.enqueue_turn == self.turn && refreshed.is_none() {
                self.turn_prepared.take()
            } else {
                None
            };
            if let Some(snapshot) = refreshed {
                tracing::info!(
                    request_id = queued.request.mode.request_id().unwrap_or("unknown"),
                    wait_ms,
                    "refreshed overlap scores after long queue wait"
                );
                queued.request.overlap = snapshot.overlap;
                queued.request.kv_transfer_candidates = if queued.request.retain_kv_transfer_chain {
                    snapshot.kv_transfer_candidates
                } else {
                    None
                };
            }
            let admit_now = Instant::now();
            let class_index = popped.class_index();
            let class = self.profile.class(class_index);
            let queued = popped.into_payload();
            tracing::trace!(
                policy_class = class.name,
                "scheduling request from pending queue"
            );
            self.admit_one(
                queued.request,
                queued.attempt_tx,
                queued.lifecycle_transfer,
                admit_now,
                prepared,
            );
        }
    }

    /// Refresh derived load for every request, retaining the allocation through
    /// selection, booking, and response-field capture, including error paths.
    fn with_projected_loads<R>(
        &mut self,
        mut request: SchedulingRequest,
        decay_now: Instant,
        handle: impl FnOnce(&Self, &mut SchedulingRequest) -> R,
    ) -> R {
        request.worker_loads = std::mem::take(&mut self.projected_loads);
        // The host's reject thresholds are judged in the same pass that projects each rank's
        // load, so selection only consults the ranks found full: no second read per rank.
        let full_ranks = match self
            .reject_thresholds
            .as_ref()
            .map(|provider| provider())
            .filter(BusyThresholds::is_configured)
        {
            Some(limits) => {
                self.full_ranks.reset(limits);
                Some(&mut self.full_ranks)
            }
            None => {
                self.full_ranks.clear();
                None
            }
        };
        self.slots.project_worker_loads_into(
            request.token_seq.as_deref(),
            decay_now,
            &mut request.worker_loads,
            full_ranks,
        );
        let result = handle(self, &mut request);
        self.projected_loads = std::mem::take(&mut request.worker_loads);
        result
    }

    /// `admitted` selections may carry the installed selection trace; advisory
    /// probes never do, so they do not pay for capture.
    fn select_worker_for_request(
        &self,
        request: &SchedulingRequest,
        admitted: bool,
        prepared: Option<Box<PreparedAdmission>>,
    ) -> Result<SelectedWorkerForRequest, KvSchedulerError> {
        let available_worker_ids = self
            .available_worker_provider
            .as_ref()
            .and_then(|provider| provider(request));

        {
            let workers = self.workers_with_configs.borrow();
            let overloaded_worker_ids = self
                .overloaded_worker_provider
                .as_ref()
                .and_then(|provider| provider());
            let eligibility = selection_eligibility(
                request,
                &workers,
                overloaded_worker_ids.as_ref(),
                available_worker_ids.as_deref(),
                Some(&self.full_ranks),
                self.selector.uses_exclusive_affinity_target(),
            );
            // Rows were built without the full-rank cut, so they stand only while no
            // rank is full. The configuration is checked under the borrow selection
            // uses: a change since the actor last advanced the generation is not
            // counted in it yet.
            let prepared = prepared
                .filter(|_| self.full_ranks.is_empty() && !workers.has_changed())
                .and_then(|prepared| {
                    prepared.into_current(
                        self.workers_generation.load(AtomicOrdering::Relaxed),
                        available_worker_ids.as_ref(),
                        overloaded_worker_ids.as_ref(),
                    )
                });
            let trace = if admitted {
                self.selection_trace.get().copied()
            } else {
                None
            };
            self.selector
                .select_worker_prepared(
                    WorkerSelectionInput::configured(
                        &workers,
                        request,
                        eligibility,
                        self.block_size,
                    ),
                    prepared,
                    trace,
                )
                .map(|(selection, selection_trace)| {
                    let non_max_overlap_selection = if request.mode.is_tracked()
                        && self.non_max_overlap_selection_observer.get().is_some()
                    {
                        non_max_overlap_selection(
                            &workers,
                            request,
                            eligibility,
                            selection.worker,
                            selection.effective_overlap_blocks,
                        )
                    } else {
                        None
                    };
                    let config = workers
                        .get(&selection.worker.worker_id)
                        .expect("selected worker config must exist");
                    let selected_worker_tiers = request
                        .overlap
                        .selected_worker_tiers(selection.worker, config);
                    let worker_load = request.worker_load_for(selection.worker);
                    let capacity = WorkerCapacity::of(config);
                    let selected_worker_load = AdvisoryWorkerLoad {
                        active_prefill_tokens: worker_load.active_prefill_tokens,
                        prefill_token_capacity: capacity.prefill_budget() as usize,
                        // TODO(rank-aware-kv-capacity): resolve the selected DP rank and preserve
                        // capacity quality. Estimated fallbacks must not authorize load-based
                        // admission/bypass decisions that require an exact or conservative bound.
                        total_kv_blocks: capacity.total_kv_blocks().map(|blocks| blocks as usize),
                    };
                    SelectedWorkerForRequest {
                        selection,
                        selection_trace,
                        selected_worker_tiers,
                        selected_worker_load,
                        non_max_overlap_selection,
                    }
                })
        }
    }

    fn select_without_admission_inner(
        &mut self,
        request: SchedulingRequest,
        decay_now: Instant,
    ) -> Result<AdvisorySchedulingResponse, KvSchedulerError> {
        self.with_projected_loads(request, decay_now, |actor, request| {
            let selected = actor.select_worker_for_request(request, false, None)?;

            Ok(AdvisorySchedulingResponse {
                selected_worker_load: selected.selected_worker_load,
                response: SchedulingResponse {
                    best_worker: selected.selection.worker,
                    effective_overlap_blocks: selected.selection.effective_overlap_blocks,
                    cached_tokens: selected.selection.cached_tokens,
                    max_raw_cached_tokens: selected.selection.max_raw_cached_tokens,
                    selected_raw_cached_tokens: selected.selection.selected_raw_cached_tokens,
                    selected_worker_tiers: selected.selected_worker_tiers,
                    kv_transfer_source: selected.selection.kv_transfer_source,
                    kv_transfer_candidates: request.kv_transfer_candidates.take(),
                    potential_decode_blocks: selected.selection.potential_decode_blocks,
                    selection_trace: selected.selection_trace,
                },
            })
        })
    }

    /// Run the full scheduling pipeline for a single request:
    /// compute projected load -> select worker -> book tracked state -> respond.
    fn admit_one(
        &mut self,
        request: SchedulingRequest,
        attempt_tx: Option<oneshot::Sender<AttemptId>>,
        lifecycle_transfer: Option<Arc<AdmissionLifecycleTransfer>>,
        decay_now: Instant,
        prepared: Option<Box<PreparedAdmission>>,
    ) -> bool {
        if prepared.is_some() {
            self.refresh_workers_generation();
        }
        self.with_projected_loads(request, decay_now, |actor, request| {
            actor.admit_one_projected(request, attempt_tx, lifecycle_transfer, prepared)
        })
    }

    /// Advance the generation when the configuration changed since the actor
    /// last looked, so rows prepared against an earlier one are rebuilt.
    fn refresh_workers_generation(&mut self) {
        if self.workers_with_configs.borrow_and_update().has_changed() {
            self.workers_generation
                .fetch_add(1, AtomicOrdering::Release);
        }
    }

    fn admit_one_projected(
        &self,
        request: &mut SchedulingRequest,
        attempt_tx: Option<oneshot::Sender<AttemptId>>,
        lifecycle_transfer: Option<Arc<AdmissionLifecycleTransfer>>,
        prepared: Option<Box<PreparedAdmission>>,
    ) -> bool {
        let selected = match self.select_worker_for_request(request, true, prepared) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("scheduling failed: {e}");
                request.respond(Err(e));
                return false;
            }
        };

        let response = SchedulingResponse {
            best_worker: selected.selection.worker,
            effective_overlap_blocks: selected.selection.effective_overlap_blocks,
            cached_tokens: selected.selection.cached_tokens,
            max_raw_cached_tokens: selected.selection.max_raw_cached_tokens,
            selected_raw_cached_tokens: selected.selection.selected_raw_cached_tokens,
            selected_worker_tiers: selected.selected_worker_tiers,
            kv_transfer_source: selected.selection.kv_transfer_source,
            kv_transfer_candidates: request.kv_transfer_candidates.take(),
            potential_decode_blocks: selected.selection.potential_decode_blocks,
            selection_trace: selected.selection_trace,
        };
        let non_max_overlap_selection = selected.non_max_overlap_selection;

        if !request.mode.is_tracked() {
            request.respond(Ok(response));
            return false;
        }

        let request_id = request
            .mode
            .tracked_request_id()
            .expect("tracked mode always has a request ID")
            .to_string();

        let prefill_load_hint = self.prefill_load_hint_for(
            request.isl_tokens,
            selected.selection.cached_tokens,
            request.track_prefill_tokens,
        );

        let sequence_request = SequenceRequest {
            request_id,
            token_sequence: request.token_seq.take(),
            track_prefill_tokens: request.track_prefill_tokens,
            expected_output_tokens: request.expected_output_tokens,
            prefill_load_hint,
            worker: selected.selection.worker,
            lora_name: request.lora_name.take(),
        };
        self.book_and_respond(
            request,
            attempt_tx,
            lifecycle_transfer,
            sequence_request,
            response,
            non_max_overlap_selection,
        )
    }

    /// A closed receiver means the actor-owned request was abandoned before
    /// booking, so there is nothing to install. Otherwise booking precedes the
    /// response: once delivery succeeds, the response channel no longer tracks
    /// request lifetime and the caller must install its RAII cleanup owner. If
    /// delivery loses that race, roll back the booking here.
    fn book_and_respond(
        &self,
        request: &mut SchedulingRequest,
        attempt_tx: Option<oneshot::Sender<AttemptId>>,
        lifecycle_transfer: Option<Arc<AdmissionLifecycleTransfer>>,
        sequence_request: SequenceRequest,
        response: SchedulingResponse,
        non_max_overlap_selection: Option<NonMaxOverlapSelection>,
    ) -> bool {
        if request.response_is_closed() {
            tracing::debug!(
                request_id = %sequence_request.request_id,
                "Skipping scheduler booking for cancelled request"
            );
            return false;
        }

        let request_id = sequence_request.request_id.clone();
        let worker = sequence_request.worker;
        let attempt_id = match self
            .slots
            .add_request_admitted(sequence_request, Instant::now())
        {
            Ok(attempt_id) => attempt_id,
            Err(error) => {
                tracing::warn!(%request_id, %error, "Failed to book scheduler state");
                request.respond(Err(KvSchedulerError::BookingFailed(error.to_string())));
                return false;
            }
        };
        if let Some(transfer) = lifecycle_transfer {
            transfer.arm_booking(SchedulerBookingDescriptor {
                request_id: request_id.clone(),
                worker,
                attempt_id,
            });
        }
        if let Some(attempt_tx) = attempt_tx {
            let _ = attempt_tx.send(attempt_id);
        }
        if request.respond(Ok(response)) {
            if let Some(selection) = non_max_overlap_selection {
                self.dispatch_non_max_overlap_selection(request_id, selection);
            }
            return true;
        }

        tracing::debug!(%request_id, "Rolling back undelivered scheduler booking");
        if let Err(error) =
            self.slots
                .free_if_booking(&request_id, worker, attempt_id, Instant::now())
        {
            tracing::error!(%request_id, %error, "Failed to roll back scheduler booking");
        }
        false
    }

    fn dispatch_non_max_overlap_selection(
        &self,
        request_id: String,
        selection: NonMaxOverlapSelection,
    ) {
        let Some(observer) = self.non_max_overlap_selection_observer.get() else {
            return;
        };
        let observer = Arc::clone(observer);
        let _observer_task = tokio::task::spawn_blocking(move || {
            observer(&request_id, selection);
        });
    }

    fn prefill_load_hint_for(
        &self,
        isl_tokens: usize,
        cached_tokens: usize,
        track_prefill_tokens: bool,
    ) -> Option<PrefillLoadHint> {
        if !track_prefill_tokens {
            return None;
        }

        let effective_isl = effective_prefill_tokens(isl_tokens, cached_tokens);
        if effective_isl == 0 {
            return None;
        }
        let prefix = isl_tokens - effective_isl;

        let expected_prefill_duration = match &self.prefill_load_estimator {
            Some(estimator) => match estimator.predict_prefill_duration(1, effective_isl, prefix) {
                Ok(expected_prefill_duration) => Some(expected_prefill_duration),
                Err(error) => {
                    tracing::warn!(
                        effective_isl,
                        prefix,
                        "failed to predict prefill duration for active load tracking: {error}"
                    );
                    None
                }
            },
            None => None,
        };

        Some(PrefillLoadHint {
            initial_effective_prefill_tokens: effective_isl,
            expected_prefill_duration,
        })
    }

    /// Whether every eligible rank is past `limits`, judged on the same `LOAD` row worker
    /// selection reads.
    fn all_eligible_workers_full(
        worker_loads: &FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
        configs: &HashMap<WorkerId, C>,
        limits: BusyThresholds,
        eligibility: RoutingEligibility<'_>,
    ) -> bool {
        let has_room = |worker: WorkerWithDpRank, config: &C| {
            let load =
                WorkerLoadInput::for_worker(worker_loads, worker, WorkerLoadProjection::default);
            limits.has_room(&load, WorkerCapacity::of(config))
        };

        if let Some(worker) = eligibility.pinned_worker() {
            let Ok(config) = eligibility.validate_worker_rank(configs, worker) else {
                return false;
            };
            return !has_room(worker, config);
        }

        let mut checked_any = false;
        let has_available = eligibility.any_eligible_worker_rank(configs, |worker, config| {
            checked_any = true;
            has_room(worker, config)
        });

        checked_any && !has_available
    }

    fn add_class_counters(&self, class_index: usize, snapshot: QueueSnapshot) {
        let counters = &self.class_counters[class_index];
        counters.pending_count.fetch_add(1, AtomicOrdering::Relaxed);
        counters
            .pending_isl_tokens
            .fetch_add(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
        counters
            .pending_cached_tokens
            .fetch_add(snapshot.cached_tokens, AtomicOrdering::Relaxed);
    }

    fn subtract_class_counters(&self, class_index: usize, snapshot: QueueSnapshot) {
        let counters = &self.class_counters[class_index];
        counters.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
        counters
            .pending_isl_tokens
            .fetch_sub(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
        counters
            .pending_cached_tokens
            .fetch_sub(snapshot.cached_tokens, AtomicOrdering::Relaxed);
    }
}

#[cfg(test)]
mod tests;
