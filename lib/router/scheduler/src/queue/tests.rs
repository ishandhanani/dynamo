// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the scheduler queue actor and its booking lifecycle.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex as StdMutex};
use std::time::Duration;

use async_trait::async_trait;
use rustc_hash::FxHashMap;
use tokio::sync::{Barrier, watch};

use super::*;
use crate::scheduling_compat::OverlapSignals;
use crate::scheduling_compat::RefreshedOverlap;
use crate::test_utils::{NoopSequencePublisher, SimpleWorkerConfig};
use dynamo_router_protocol::kv_hints::KvTransferCandidates;
use dynamo_router_protocol::protocols::{
    ActiveSequenceEvent, ExternalSequenceBlockHash, WorkerSelectionResult, WorkerWithDpRank,
};
use dynamo_router_protocol::sched::{KvSchedulerError, ScheduleMode, WorkerSelectionPolicyError};
use dynamo_slot_tracker::topology::WorkerDpRange;
use dynamo_slot_tracker::{ActiveSequencesMultiWorker, SequencePublisher};
use dynamo_worker_selection::plugin::{
    WorkerCandidates, WorkerInputView, WorkerPicker, WorkerSelectionContext,
};
use dynamo_worker_selection::selector::{
    CandidatePreparer, DefaultWorkerSelector, PreparedCandidates, WorkerInputs,
    WorkerSelectionPolicy, WorkerSelector,
};

fn decay_now() -> Instant {
    Instant::now()
}

struct FixedPrefillLoadEstimator {
    duration: Duration,
}

impl PrefillLoadEstimator for FixedPrefillLoadEstimator {
    fn predict_prefill_duration(
        &self,
        _batch_size: usize,
        _effective_isl: usize,
        _prefix: usize,
    ) -> anyhow::Result<Duration> {
        Ok(self.duration)
    }
}

type SchedulingResponseReceiver =
    tokio::sync::oneshot::Receiver<Result<SchedulingResponse, KvSchedulerError>>;

#[test]
fn admission_lifecycle_cleanup_upgrades_to_the_exact_booking() {
    let transfer = AdmissionLifecycleTransfer::new("retryable-request".to_string());
    transfer.arm_pending();
    let booking = SchedulerBookingDescriptor {
        request_id: "retryable-request".to_string(),
        worker: WorkerWithDpRank::new(7, 3),
        attempt_id: AttemptId::new(41),
    };
    transfer.arm_booking(booking.clone());

    let Some(AdmissionLifecycleCleanup::Booking(cleanup)) = transfer.cleanup_target() else {
        panic!("admitted cleanup must carry the exact booking");
    };
    assert_eq!(cleanup, booking);
}

fn book_directly(
    slots: &ActiveSequencesMultiWorker<NoopSequencePublisher>,
    request_id: &str,
) -> SchedulerBookingDescriptor {
    let worker = WorkerWithDpRank::new(0, 0);
    let attempt_id = slots
        .add_request_admitted(
            dynamo_slot_tracker::SequenceRequest {
                request_id: request_id.to_string(),
                token_sequence: None,
                track_prefill_tokens: true,
                expected_output_tokens: None,
                prefill_load_hint: None,
                worker,
                lora_name: None,
            },
            Instant::now(),
        )
        .expect("worker 0 is registered");
    SchedulerBookingDescriptor {
        request_id: request_id.to_string(),
        worker,
        attempt_id,
    }
}

async fn wait_freed(
    slots: &ActiveSequencesMultiWorker<NoopSequencePublisher>,
    booking: &SchedulerBookingDescriptor,
) {
    tokio::time::timeout(Duration::from_secs(2), async {
        while slots.has_booking(booking) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("booking was freed");
}

/// The handle frees its booking exactly when it is dropped armed; `commit`
/// hands the booking over untouched and `release` frees it with an ack.
#[tokio::test(flavor = "multi_thread")]
async fn booking_handle_frees_only_while_armed() {
    let (queue, slots) = make_queue(1, 16, 512, Some(0.0));

    let armed = book_directly(&slots, "armed");
    drop(queue.booking_handle(armed.clone()));
    wait_freed(&slots, &armed).await;

    let committed = book_directly(&slots, "committed");
    let handle = queue.booking_handle(committed.clone());
    let request_id_ptr = handle.booking.request_id.as_ptr();
    let descriptor = handle.commit();
    assert_eq!(descriptor, committed);
    assert_eq!(descriptor.request_id.as_ptr(), request_id_ptr);

    // Explicit release is immediate. The following actor barrier also
    // proves commit did not enqueue deferred cleanup for its booking.
    let released = book_directly(&slots, "released");
    queue
        .booking_handle(released.clone())
        .release()
        .await
        .expect("acknowledged release");
    queue.update().await;
    assert!(
        !slots.has_booking(&released),
        "release frees before it returns"
    );
    assert!(
        slots.has_booking(&committed),
        "a committed handle leaves the booking to its new owner"
    );
    slots.free(&committed.request_id, decay_now()).unwrap();
    slots.assert_completely_drained(decay_now());
}

struct DropResponseOnLoadPublisher {
    response_rx: Arc<StdMutex<Option<SchedulingResponseReceiver>>>,
}

impl SequencePublisher for DropResponseOnLoadPublisher {
    fn enqueue_event(&self, _event: ActiveSequenceEvent) -> anyhow::Result<()> {
        Ok(())
    }

    fn publish_scheduler_load(&self, _load: dynamo_slot_tracker::SchedulerLoadSnapshot) {
        self.response_rx.lock().unwrap().take();
    }

    fn observe_load(&self, _: &WorkerWithDpRank, _: &str, _: dynamo_slot_tracker::LocalWorkerLoad) {
    }
}

#[derive(Default)]
struct SelectorRendezvous {
    arrivals: StdMutex<usize>,
    cv: Condvar,
}

impl SelectorRendezvous {
    fn wait_for_peer(&self) {
        let mut arrivals = self.arrivals.lock().unwrap();
        *arrivals += 1;

        if *arrivals == 1 {
            let _ = self
                .cv
                .wait_timeout(arrivals, Duration::from_millis(100))
                .unwrap();
            return;
        }

        self.cv.notify_all();
    }
}

#[derive(Clone)]
struct MinDecodeSelector {
    rendezvous: Option<Arc<SelectorRendezvous>>,
}

impl WorkerSelector<SimpleWorkerConfig> for MinDecodeSelector {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    fn select_worker(
        &self,
        input: WorkerSelectionInput<'_, SimpleWorkerConfig>,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        let (workers, request, eligibility, block_size) = input.into_configured();
        if let Some(rendezvous) = &self.rendezvous {
            rendezvous.wait_for_peer();
        }

        let mut best_worker = None;
        eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
            let load = request.worker_load_for(worker);
            let potential_prefill_tokens = if request.track_prefill_tokens {
                load.active_prefill_tokens
                    .saturating_add(effective_prefill_tokens(
                        request.isl_tokens,
                        request.effective_cached_tokens_for(worker),
                    ))
            } else {
                0
            };
            let potential_decode_blocks = load.potential_decode_blocks();
            let key = (
                potential_prefill_tokens,
                potential_decode_blocks,
                worker.worker_id,
                worker.dp_rank,
            );
            if best_worker.is_none_or(|(_, best_key)| key < best_key) {
                best_worker = Some((worker, key));
            }
        });

        let Some((worker, _)) = best_worker else {
            return Err(KvSchedulerError::NoEndpoints);
        };

        Ok(WorkerSelectionResult {
            worker,
            required_blocks: request.request_blocks(block_size),
            effective_overlap_blocks: request.effective_overlap_blocks_for(worker),
            cached_tokens: request.effective_cached_tokens_for(worker),
            max_raw_cached_tokens: None,
            selected_raw_cached_tokens: None,
            potential_decode_blocks: request
                .potential_decode_blocks_after_admission(worker, block_size),
            kv_transfer_source: None,
        })
    }
}

#[allow(clippy::type_complexity)]
fn make_queue(
    num_workers: usize,
    block_size: u32,
    isl: usize,
    threshold_frac: Option<f64>,
) -> (
    Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>>,
    Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
) {
    let (queue, slots, _tx) =
        make_queue_with_sender(num_workers, block_size, isl, threshold_frac, None);
    (queue, slots)
}

#[allow(clippy::type_complexity)]
fn make_queue_with_custom_selector<Sel: WorkerSelector<SimpleWorkerConfig> + Send + 'static>(
    num_workers: usize,
    block_size: u32,
    isl: usize,
    threshold_frac: Option<f64>,
    selector: Sel,
) -> (
    Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, Sel>>,
    Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
) {
    let dp_range: HashMap<u64, (u32, u32)> =
        (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
    let slots = Arc::new(ActiveSequencesMultiWorker::new(
        NoopSequencePublisher,
        block_size as usize,
        dp_range,
        false,
        0,
        "test",
    ));

    let mut configs: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
    for id in 0..num_workers as u64 {
        configs.insert(
            id,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(isl as u64),
                ..Default::default()
            },
        );
    }
    let (_cfg_tx, cfg_rx) = watch::channel(configs);

    let queue = Arc::new(SchedulerQueue::new(
        Arc::clone(&slots),
        cfg_rx,
        PolicyProfile::synthetic(threshold_frac, RouterQueuePolicy::Fcfs),
        block_size,
        selector,
        None,
        None,
        None,
        None,
        None,
    ));

    (queue, slots)
}

#[allow(clippy::type_complexity)]
fn make_queue_with_sender(
    num_workers: usize,
    block_size: u32,
    isl: usize,
    threshold_frac: Option<f64>,
    prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
) -> (
    Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>>,
    Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    watch::Sender<HashMap<u64, SimpleWorkerConfig>>,
) {
    let dp_range: HashMap<u64, (u32, u32)> =
        (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
    let slots = Arc::new(ActiveSequencesMultiWorker::new(
        NoopSequencePublisher,
        block_size as usize,
        dp_range,
        false,
        0,
        "test",
    ));

    let mut configs: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
    for id in 0..num_workers as u64 {
        configs.insert(
            id,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(isl as u64),
                ..Default::default()
            },
        );
    }
    let (cfg_tx, cfg_rx) = watch::channel(configs);

    let selector = DefaultWorkerSelector::new(None, "test");
    let queue = Arc::new(SchedulerQueue::new(
        Arc::clone(&slots),
        cfg_rx,
        PolicyProfile::synthetic(threshold_frac, RouterQueuePolicy::Fcfs),
        block_size,
        selector,
        prefill_load_estimator,
        None,
        None,
        None,
        None,
    ));

    (queue, slots, cfg_tx)
}

fn policy_profile(yaml: &str) -> PolicyProfile {
    PolicyProfile::from_yaml(yaml).unwrap()
}

#[allow(clippy::type_complexity)]
fn make_queue_with_profile(
    num_workers: usize,
    block_size: u32,
    max_num_batched_tokens: usize,
    profile: PolicyProfile,
) -> (
    Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>>,
    Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
) {
    let (queue, slots, _cfg_tx) = make_queue_with_profile_and_sender(
        num_workers,
        block_size,
        max_num_batched_tokens,
        profile,
    );
    (queue, slots)
}

#[allow(clippy::type_complexity)]
fn make_queue_with_profile_and_sender(
    num_workers: usize,
    block_size: u32,
    max_num_batched_tokens: usize,
    profile: PolicyProfile,
) -> (
    Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>>,
    Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    watch::Sender<HashMap<u64, SimpleWorkerConfig>>,
) {
    make_queue_with_profile_sender_and_reject(
        num_workers,
        block_size,
        max_num_batched_tokens,
        profile,
        None,
    )
}

#[allow(clippy::type_complexity)]
fn make_queue_with_profile_sender_and_reject(
    num_workers: usize,
    block_size: u32,
    max_num_batched_tokens: usize,
    profile: PolicyProfile,
    reject_thresholds: Option<RejectThresholdsProvider>,
) -> (
    Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>>,
    Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    watch::Sender<HashMap<u64, SimpleWorkerConfig>>,
) {
    let dp_range: HashMap<u64, (u32, u32)> =
        (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
    let slots = Arc::new(ActiveSequencesMultiWorker::new(
        NoopSequencePublisher,
        block_size as usize,
        dp_range,
        false,
        0,
        "test",
    ));
    let configs = (0..num_workers as u64)
        .map(|id| {
            (
                id,
                SimpleWorkerConfig {
                    max_num_batched_tokens: Some(max_num_batched_tokens as u64),
                    ..Default::default()
                },
            )
        })
        .collect();
    let (cfg_tx, cfg_rx) = watch::channel(configs);
    set_worker_capacities_from_configs(&slots, &cfg_rx.borrow());
    let queue = Arc::new(SchedulerQueue::new(
        Arc::clone(&slots),
        cfg_rx,
        profile,
        block_size,
        DefaultWorkerSelector::new(None, "test"),
        None,
        None,
        None,
        None,
        reject_thresholds,
    ));
    (queue, slots, cfg_tx)
}

/// What `LocalScheduler::reconcile_worker_configs` does after a config publish: the
/// registry learns each rank's reported capacity so the projection pass can judge it.
fn set_worker_capacities_from_configs(
    slots: &ActiveSequencesMultiWorker<NoopSequencePublisher>,
    configs: &HashMap<u64, SimpleWorkerConfig>,
) {
    slots.set_worker_capacities(configs.iter().map(|(&id, cfg)| {
        (
            dynamo_slot_tracker::topology::WorkerDpRange::new(
                id,
                cfg.data_parallel_start_rank(),
                cfg.data_parallel_size(),
            ),
            WorkerCapacity::of(cfg),
        )
    }));
}

#[allow(clippy::type_complexity)]
fn make_queue_with_providers(
    num_workers: usize,
    block_size: u32,
    isl: usize,
    overloaded_worker_provider: Option<OverloadedWorkerProvider>,
    available_worker_provider: Option<WorkerAvailabilityProvider>,
) -> (
    Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>>,
    Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
) {
    let dp_range: HashMap<u64, (u32, u32)> =
        (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
    let slots = Arc::new(ActiveSequencesMultiWorker::new(
        NoopSequencePublisher,
        block_size as usize,
        dp_range,
        false,
        0,
        "test",
    ));

    let mut configs: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
    for id in 0..num_workers as u64 {
        configs.insert(
            id,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(isl as u64),
                ..Default::default()
            },
        );
    }
    let (_cfg_tx, cfg_rx) = watch::channel(configs);

    let selector = DefaultWorkerSelector::new(None, "test");
    let queue = Arc::new(SchedulerQueue::new(
        Arc::clone(&slots),
        cfg_rx,
        PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs),
        block_size,
        selector,
        None,
        None,
        overloaded_worker_provider,
        available_worker_provider,
        None,
    ));

    (queue, slots)
}

struct CountingRefresher {
    calls: AtomicUsize,
    last_retain_kv_transfer_chain: AtomicBool,
    response: RefreshedOverlap,
}

#[async_trait]
impl OverlapScoresRefresh for CountingRefresher {
    async fn refresh(
        &self,
        _block_hashes: &[LocalBlockHash],
        retain_kv_transfer_chain: bool,
    ) -> Option<RefreshedOverlap> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.last_retain_kv_transfer_chain
            .store(retain_kv_transfer_chain, Ordering::Relaxed);
        Some(self.response.clone())
    }
}

struct BlockingRefresher {
    calls: AtomicUsize,
    started: tokio::sync::Notify,
    release: tokio::sync::Notify,
    response: RefreshedOverlap,
}

impl BlockingRefresher {
    fn new(response: RefreshedOverlap) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            started: tokio::sync::Notify::new(),
            release: tokio::sync::Notify::new(),
            response,
        }
    }

    async fn wait_for_calls(&self, target: usize) {
        while self.calls.load(Ordering::Relaxed) < target {
            self.started.notified().await;
        }
    }

    fn release_one(&self) {
        self.release.notify_one();
    }
}

#[async_trait]
impl OverlapScoresRefresh for BlockingRefresher {
    async fn refresh(
        &self,
        _block_hashes: &[LocalBlockHash],
        _retain_kv_transfer_chain: bool,
    ) -> Option<RefreshedOverlap> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.started.notify_one();
        self.release.notified().await;
        Some(self.response.clone())
    }
}

#[allow(clippy::type_complexity)]
fn make_queue_with_refresher(
    num_workers: usize,
    block_size: u32,
    isl: usize,
    threshold_frac: Option<f64>,
    refresher: Arc<CountingRefresher>,
) -> (
    Arc<
        SchedulerQueue<
            NoopSequencePublisher,
            SimpleWorkerConfig,
            DefaultWorkerSelector,
            CountingRefresher,
        >,
    >,
    Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
) {
    let dp_range: HashMap<u64, (u32, u32)> =
        (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
    let slots = Arc::new(ActiveSequencesMultiWorker::new(
        NoopSequencePublisher,
        block_size as usize,
        dp_range,
        false,
        0,
        "test",
    ));

    let mut configs: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
    for id in 0..num_workers as u64 {
        configs.insert(
            id,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(isl as u64),
                ..Default::default()
            },
        );
    }
    let (_cfg_tx, cfg_rx) = watch::channel(configs);

    let queue = Arc::new(SchedulerQueue::new(
        Arc::clone(&slots),
        cfg_rx,
        PolicyProfile::synthetic(threshold_frac, RouterQueuePolicy::Fcfs),
        block_size,
        DefaultWorkerSelector::new(None, "test"),
        None,
        Some(refresher),
        None,
        None,
        None,
    ));

    (queue, slots)
}

#[allow(clippy::type_complexity)]
fn make_queue_with_blocking_refresher(
    num_workers: usize,
    block_size: u32,
    isl: usize,
    threshold_frac: Option<f64>,
    refresher: Arc<BlockingRefresher>,
    admission_channel_capacity: usize,
) -> (
    Arc<
        SchedulerQueue<
            NoopSequencePublisher,
            SimpleWorkerConfig,
            DefaultWorkerSelector,
            BlockingRefresher,
        >,
    >,
    Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
) {
    let dp_range: HashMap<u64, (u32, u32)> =
        (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
    let slots = Arc::new(ActiveSequencesMultiWorker::new(
        NoopSequencePublisher,
        block_size as usize,
        dp_range,
        false,
        0,
        "test",
    ));

    let mut configs: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
    for id in 0..num_workers as u64 {
        configs.insert(
            id,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(isl as u64),
                ..Default::default()
            },
        );
    }
    let (_cfg_tx, cfg_rx) = watch::channel(configs);

    let queue = Arc::new(SchedulerQueue::new_with_capacity(
        Arc::clone(&slots),
        cfg_rx,
        PolicyProfile::synthetic(threshold_frac, RouterQueuePolicy::Fcfs),
        block_size,
        DefaultWorkerSelector::new(None, "test"),
        None,
        Some(refresher),
        None,
        None,
        None,
        admission_channel_capacity,
    ));

    (queue, slots)
}

fn make_request(
    request_id: &str,
    isl_tokens: usize,
) -> (
    SchedulingRequest,
    tokio::sync::oneshot::Receiver<
        Result<SchedulingResponse, dynamo_router_protocol::sched::KvSchedulerError>,
    >,
) {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let req = SchedulingRequest {
        mode: ScheduleMode::Tracked {
            request_id: request_id.to_string(),
        },
        token_seq: None,
        isl_tokens,
        overlap: OverlapSignals::default(),
        kv_transfer_candidates: None,
        retain_kv_transfer_chain: false,
        worker_loads: FxHashMap::default(),
        track_prefill_tokens: true,
        router_config_override: None,
        lora_name: None,
        priority_jump: 0.0,
        strict_priority: 0,
        policy_class: None,
        session_context: None,
        expected_output_tokens: None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: dynamo_router_protocol::protocols::RoutingConstraints::default(),
        shared_cache_hits: None,
        resp_tx: Some(Box::new(
            dynamo_worker_selection::selector::OneshotResponder(tx),
        )),
    };
    (req, rx)
}

#[tokio::test]
async fn classifier_sequence_hashes_preserve_absence_empty_and_order() {
    use dynamo_admission::classifier::{ClassifyFuture, RequestClassifier};

    struct InspectHashes(Option<Vec<u64>>, WorkerWithDpRank);
    impl RequestClassifier for InspectHashes {
        fn classify(
            &mut self,
            request: ClassifyRequest,
            _workers: WorkerCandidates<'_>,
        ) -> ClassifyFuture {
            let expected = self.0.clone();
            let pin = self.1;
            Box::pin(async move {
                tokio::task::yield_now().await;
                assert_eq!(request.sequence_hashes(), expected.as_deref());
                assert_eq!(request.pinned_worker(), Some(pin));
                Ok(request)
            })
        }
    }

    let profile = PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs);
    let (queue, _slots) = make_queue_with_profile(1, 64, 256, profile);
    let pin = WorkerWithDpRank::new(0, 0);
    for hashes in [None, Some(vec![]), Some(vec![11, 7, 11])] {
        let (mut request, _rx) = make_request("prefix-input", 193);
        request.token_seq = hashes.clone();
        request.pinned_worker = Some(pin);
        let (classified, workers) =
            queue.build_classify_request(&request, Instant::now(), WorkerInputs::NONE);
        assert_eq!(request.token_seq, hashes);
        let mut classifier = InspectHashes(hashes, pin);
        let pending = workers.read(|workers| classifier.classify(classified, workers));
        // The independently pollable future must not borrow actor-owned data.
        request.token_seq = Some(vec![99]);
        request.pinned_worker = None;
        drop(request);
        pending.await.unwrap();
    }
}

/// Measures real queue projection with snapshots held in pending classifier futures.
/// Run in release mode with CLASSIFIER_BENCH_HASHES ("none" or a count) and
/// CLASSIFIER_BENCH_PENDING set; source setup and cancellation are not timed.
#[tokio::test]
#[ignore = "CPU microbenchmark; run explicitly with --release --ignored --nocapture"]
async fn classifier_input_projection_benchmark() {
    use dynamo_admission::classifier::{ClassifyFuture, RequestClassifier};
    use std::hint::black_box;
    use std::task::Context;
    use std::time::Instant as WallClock;

    struct Park;
    impl RequestClassifier for Park {
        fn classify(
            &mut self,
            request: ClassifyRequest,
            _workers: WorkerCandidates<'_>,
        ) -> ClassifyFuture {
            Box::pin(async move {
                std::future::pending::<()>().await;
                Ok(black_box(request))
            })
        }
    }

    let hash_count = std::env::var("CLASSIFIER_BENCH_HASHES").unwrap();
    let hashes = if hash_count == "none" {
        None
    } else {
        Some(hash_count.parse::<usize>().unwrap())
    };
    let retained: usize = std::env::var("CLASSIFIER_BENCH_PENDING")
        .unwrap()
        .parse()
        .unwrap();
    assert!(retained > 0 && retained <= 1024);
    assert!(hashes.unwrap_or(0) <= 62_500);
    let profile = PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs);
    let (queue, _slots) = make_queue_with_profile(1, 16, 1_000_000, profile);
    let sources: Vec<_> = (0..retained)
        .map(|i| {
            let (mut request, rx) = make_request(
                &format!("copy-bench-{i}"),
                hashes.unwrap_or(0).saturating_mul(16).max(1),
            );
            request.token_seq = hashes.map(|n| (0..n as u64).collect());
            (request, rx)
        })
        .collect();
    let mut classifier = Park;
    let mut futures = Vec::with_capacity(retained);
    let mut cx = Context::from_waker(futures_util::task::noop_waker_ref());
    let ingress_at = Instant::now();
    let repeats = (1024 / retained).max(1);
    let mut samples = Vec::with_capacity(40);
    for sample in 0..45 {
        let mut elapsed = std::time::Duration::ZERO;
        for _ in 0..repeats {
            let start = WallClock::now();
            for (request, _) in &sources {
                let (input, workers) = queue.build_classify_request(
                    black_box(request),
                    ingress_at,
                    WorkerInputs::NONE,
                );
                let mut future = workers.read(|workers| classifier.classify(input, workers));
                assert!(future.as_mut().poll(&mut cx).is_pending());
                futures.push(future);
            }
            elapsed += start.elapsed();
            // Keep every snapshot live until the batch ends; cancellation is excluded.
            black_box(&futures);
            futures.clear();
        }
        if sample >= 5 {
            samples.push(elapsed.as_secs_f64() * 1e9 / (retained * repeats) as f64);
        }
    }
    println!(
        "CLASSIFIER_BENCH {}",
        serde_json::json!({
            "hashes": hashes,
            "pending": retained,
            "repeats_per_sample": repeats,
            "classify_request_bytes": std::mem::size_of::<ClassifyRequest>(),
            "ns_per_request_samples": samples,
        })
    );
}

#[tokio::test]
async fn classifiers_read_class_and_reject_thresholds() {
    let profile = policy_profile(
        r#"
default_policy_family: latency
uncached_isl_buckets:
  - min_tokens: 0
    bucket: short
  - min_tokens: 32
    bucket: long
policy_classes:
  - name: latency_short
    policy_family: latency
    cache_bucket: short
    quantum: 1
    decode_busy_threshold_frac: 0.5
  - name: latency_long
    policy_family: latency
    cache_bucket: long
    quantum: 1
  - name: bulk_short
    policy_family: bulk
    cache_bucket: short
    quantum: 1
    prefill_busy_threshold_frac: 0.25
  - name: bulk_long
    policy_family: bulk
    cache_bucket: long
    quantum: 1
"#,
    );
    let reject = crate::scheduling_compat::BusyThresholds {
        prefill_busy_threshold_frac: Some(1.0),
        ..Default::default()
    };
    let (queue, _slots, _cfg_tx) = make_queue_with_profile_sender_and_reject(
        1,
        16,
        64,
        profile,
        Some(Arc::new(move || reject)),
    );
    let (mut request, _rx) = make_request("classified", 16);
    request.policy_class = Some("latency".to_string());
    let (mut classified, _workers) =
        queue.build_classify_request(&request, Instant::now(), WorkerInputs::NONE);

    let latency = crate::scheduling_compat::BusyThresholds {
        decode_busy_threshold_frac: Some(0.5),
        ..Default::default()
    };
    let bulk = crate::scheduling_compat::BusyThresholds {
        prefill_busy_threshold_frac: Some(0.25),
        ..Default::default()
    };
    // 16 uncached tokens fall in the `short` bucket.
    assert_eq!(classified.queue_thresholds(), latency);
    assert_eq!(classified.queue_thresholds_for("bulk"), Some(bulk));
    assert_eq!(classified.queue_thresholds_for("missing"), None);
    assert_eq!(classified.reject_thresholds(), reject);

    classified.set_policy_class("bulk");
    assert_eq!(classified.queue_thresholds(), bulk);
}

#[tokio::test]
async fn classifier_overrides_are_validated_before_enqueue() {
    let profile = policy_profile(
        r#"
default_policy_family: latency
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
  - min_tokens: 32
    bucket: uncached
policy_classes:
  - name: latency_cached
    policy_family: latency
    cache_bucket: cached
    quantum: 1
  - name: latency_uncached
    policy_family: latency
    cache_bucket: uncached
    quantum: 1
  - name: bulk_cached
    policy_family: bulk
    cache_bucket: cached
    quantum: 1
  - name: bulk_uncached
    policy_family: bulk
    cache_bucket: uncached
    quantum: 1
  - name: custom_priority
    quantum: 1
"#,
    );
    let (queue, _slots) = make_queue_with_profile(1, 16, 64, profile);
    let (mut request, _rx) = make_request("classified", 16);
    request.policy_class = Some("latency".to_string());
    let ingress_at = Instant::now();
    let due_at = ingress_at + Duration::from_secs(20);
    let mut classified = queue
        .build_classify_request(&request, ingress_at, WorkerInputs::NONE)
        .0;
    assert_eq!(classified.ingress_at(), ingress_at);
    classified.set_policy_class("bulk");
    classified.set_due_at(due_at);
    classified.set_scheduling_cost_tokens(7);

    let metadata = queue
        .validate_classification(&mut request, classified, ingress_at)
        .unwrap();
    assert_eq!(
        queue.profile.class(metadata.class_index).name,
        "bulk_cached"
    );
    assert_eq!(metadata.snapshot.scheduling_cost_tokens, 7);
    assert_eq!(metadata.due_at, Some(due_at));

    let mut invalid = queue
        .build_classify_request(&request, ingress_at, WorkerInputs::NONE)
        .0;
    invalid.set_policy_class("missing");
    assert!(matches!(
        queue.validate_classification(&mut request, invalid, ingress_at),
        Err(KvSchedulerError::InvalidClassificationMetadata(_))
    ));

    let mut physical = queue
        .build_classify_request(&request, ingress_at, WorkerInputs::NONE)
        .0;
    physical.set_policy_class("latency_cached");
    assert!(matches!(
        queue.validate_classification(&mut request, physical, ingress_at),
        Err(KvSchedulerError::InvalidClassificationMetadata(_))
    ));

    let mut zero_cost = queue
        .build_classify_request(&request, ingress_at, WorkerInputs::NONE)
        .0;
    zero_cost.set_scheduling_cost_tokens(0);
    assert!(matches!(
        queue.validate_classification(&mut request, zero_cost, ingress_at),
        Err(KvSchedulerError::InvalidClassificationMetadata(_))
    ));
}

#[tokio::test]
async fn classifier_targets_preserve_pins_and_caller_eligibility() {
    let (queue, _slots) = make_queue(2, 16, 64, None);
    let (mut request, _rx) = make_request("targeted", 48);
    let worker = WorkerWithDpRank::new(1, 0);
    let pin = WorkerWithDpRank::new(0, 0);
    let now = Instant::now();

    let mut classified = queue
        .build_classify_request(&request, now, WorkerInputs::NONE)
        .0;
    classified.set_worker_selection_target(worker);
    queue
        .validate_classification(&mut request, classified, now)
        .unwrap();
    assert_eq!(request.affinity_target, Some(worker.into()));

    let mut classified = queue
        .build_classify_request(&request, now, WorkerInputs::NONE)
        .0;
    classified.clear_worker_selection_target();
    queue
        .validate_classification(&mut request, classified, now)
        .unwrap();
    assert!(request.affinity_target.is_none());

    request.pinned_worker = Some(pin);
    let mut classified = queue
        .build_classify_request(&request, now, WorkerInputs::NONE)
        .0;
    assert_eq!(classified.pinned_worker(), Some(pin));
    classified.set_worker_selection_target(worker);
    queue
        .validate_classification(&mut request, classified, now)
        .unwrap();
    assert_eq!(request.pinned_worker, Some(pin));
    assert!(request.affinity_target.is_none());
    request.pinned_worker = None;
    request.allowed_worker_ids = Some(HashSet::from([0]));
    let mut classified = queue
        .build_classify_request(&request, now, WorkerInputs::NONE)
        .0;
    classified.set_worker_selection_target(worker);
    queue
        .validate_classification(&mut request, classified, now)
        .unwrap();
    assert_eq!(request.allowed_worker_ids, Some(HashSet::from([0])));
    // The selector still applies the allowlist after a classifier supplies a target.
    let result = queue.select_without_admission(request).await.unwrap();
    assert_eq!(result.response.best_worker, pin);
}

#[tokio::test]
async fn validation_recomputes_cache_eligibility_at_enqueue() {
    let profile = policy_profile(
        r#"
default_policy_family: latency
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
  - min_tokens: 32
    bucket: uncached
policy_classes:
  - name: latency_cached
    policy_family: latency
    cache_bucket: cached
    quantum: 1
  - name: latency_uncached
    policy_family: latency
    cache_bucket: uncached
    quantum: 1
"#,
    );
    let (queue, _slots, cfg_tx) = make_queue_with_profile_and_sender(1, 16, 64, profile);
    let (mut request, _rx) = make_request("classified", 48);
    request
        .overlap
        .effective_cached_tokens
        .insert(WorkerWithDpRank::new(0, 0), 32);

    let ingress_at = Instant::now();
    let classified = queue
        .build_classify_request(&request, ingress_at, WorkerInputs::NONE)
        .0;

    // The only cache-bearing worker disappears while the classification is
    // pending; enqueue-time state governs, so the request buckets as
    // uncached — exactly as an unclassified request admitted now would.
    cfg_tx.send(HashMap::new()).unwrap();
    let metadata = queue
        .validate_classification(&mut request, classified, ingress_at)
        .unwrap();
    assert_eq!(
        queue.profile.class(metadata.class_index).name,
        "latency_uncached"
    );
}

#[tokio::test]
async fn expired_deadline_is_rejected_at_enqueue() {
    let (queue, _slots) = make_queue(1, 16, 64, Some(0.0));
    let (probe, _probe_rx) = make_request("advisory", 64);
    queue.select_without_admission(probe).await.unwrap();
    let (request, response_rx) = make_request("expired-on-arrival", 64);
    queue
        .enqueue_with_due_at_for_test(request, Instant::now())
        .await;
    assert!(matches!(
        response_rx.await.unwrap(),
        Err(KvSchedulerError::DeadlineExceeded)
    ));
    assert_eq!(queue.pending_count(), 0);
    queue.update().await;
    let stats = queue.class_queue_stats(0).unwrap();
    assert_eq!(stats.received_total, 1);
    assert_eq!(stats.rejected_due_time_passed_total, 1);
}

#[tokio::test(start_paused = true)]
async fn parked_request_expires_via_actor_timer() {
    let (queue, _slots) = make_queue(1, 16, 64, Some(0.0));
    let (active, active_rx) = make_request("active", 64);
    queue.enqueue(active).await;
    active_rx.await.unwrap().unwrap();

    // The worker is busy, so the deadline-bearing request parks; the actor
    // timer must fire at due_at and reject it without any further command.
    let (parked, parked_rx) = make_request("parked-deadline", 64);
    queue
        .enqueue_with_due_at_for_test(parked, Instant::now() + Duration::from_secs(5))
        .await;
    assert_eq!(queue.pending_count(), 1);

    assert!(matches!(
        parked_rx.await.unwrap(),
        Err(KvSchedulerError::DeadlineExceeded)
    ));
    assert_eq!(queue.pending_count(), 0);
    queue.update().await;
    let stats = queue.class_queue_stats(0).unwrap();
    assert_eq!(stats.received_total, 2);
    assert_eq!(stats.rejected_due_time_passed_total, 1);
}

#[tokio::test(start_paused = true)]
async fn deadline_expiry_during_refresh_is_counted_once() {
    let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::default()));
    let (queue, slots) = make_queue_with_blocking_refresher(
        1,
        16,
        64,
        Some(0.0),
        Arc::clone(&refresher),
        ADMISSION_CHANNEL_CAPACITY,
    );
    let (active, active_rx) = make_request("active", 64);
    queue.enqueue(active).await;
    active_rx.await.unwrap().unwrap();

    let (request, response_rx) = make_request("expires-during-refresh", 64);
    let mut queue_metadata = queue.default_queue_metadata(&request, Instant::now());
    queue_metadata.due_at = Some(Instant::now() + Duration::from_secs(30));
    let (ack_tx, ack_rx) = oneshot::channel();
    queue
        .admission_tx
        .send(AdmissionCommand::Enqueue {
            request,
            attempt_tx: None,
            block_hashes: Some(vec![LocalBlockHash(42)]),
            prepared: None,
            queue_metadata,
            lease: None,
            ack_tx,
        })
        .await
        .unwrap();
    ack_rx.await.unwrap();
    slots
        .mark_prefill_completed(&"active".to_string(), decay_now())
        .unwrap();
    slots.free(&"active".to_string(), decay_now()).unwrap();
    tokio::time::advance(Duration::from_secs(11)).await;
    let updating_queue = Arc::clone(&queue);
    let update = tokio::spawn(async move { updating_queue.update().await });
    refresher.wait_for_calls(1).await;
    tokio::time::advance(Duration::from_secs(20)).await;
    refresher.release_one();
    update.await.unwrap();
    assert!(matches!(
        response_rx.await.unwrap(),
        Err(KvSchedulerError::DeadlineExceeded)
    ));
    queue.update().await;
    assert_eq!(queue.pending_count(), 0);
    let stats = queue.class_queue_stats(0).unwrap();
    assert_eq!(stats.received_total, 2);
    assert_eq!(stats.rejected_due_time_passed_total, 1);
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn stalled_refresh_sweeps_expired_entries_without_refreshing_them() {
    let block_size = 16;
    let isl = 64;
    let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::default()));
    let (queue, slots) = make_queue_with_blocking_refresher(
        1,
        block_size,
        isl,
        Some(0.0),
        Arc::clone(&refresher),
        ADMISSION_CHANNEL_CAPACITY,
    );

    let (active, active_rx) = make_request("active", isl);
    queue.enqueue(active).await;
    active_rx.await.unwrap().unwrap();

    // Two deadline-bearing entries with the same due time: FCFS pops the
    // first into a refresh that stalls past both deadlines.
    let ingress_at = Instant::now();
    let due_at = ingress_at + Duration::from_secs(12);
    let mut receivers = Vec::new();
    for (name, hash) in [("stalled", 41), ("swept", 42)] {
        let (queued, queued_rx) = make_request(name, isl);
        let mut classified = queue
            .build_classify_request(&queued, ingress_at, WorkerInputs::NONE)
            .0;
        classified.set_due_at(due_at);
        queue
            .enqueue_admitted_with_block_hashes_and_lease(
                queued,
                Some(vec![LocalBlockHash(hash)]),
                None,
                None,
                Some(classified),
                ingress_at,
            )
            .await;
        receivers.push(queued_rx);
    }
    assert_eq!(queue.pending_count(), 2);

    slots
        .mark_prefill_completed(&"active".to_string(), decay_now())
        .unwrap();
    slots.free(&"active".to_string(), decay_now()).unwrap();
    tokio::time::advance(Duration::from_secs(11)).await;

    let update = {
        let queue = Arc::clone(&queue);
        tokio::spawn(async move { queue.update().await })
    };
    refresher.wait_for_calls(1).await;
    tokio::time::advance(Duration::from_secs(2)).await;
    refresher.release_one();
    // Pre-arm a second release so a regression (popping the expired entry
    // into its own refresh) fails the call-count assertion below instead
    // of hanging the test.
    refresher.release_one();
    update.await.unwrap();

    for queued_rx in receivers {
        assert!(matches!(
            queued_rx.await.unwrap(),
            Err(KvSchedulerError::DeadlineExceeded)
        ));
    }
    assert_eq!(queue.pending_count(), 0);
    assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
    slots.assert_completely_drained(decay_now());
}

#[tokio::test]
async fn mixed_classes_recheck_capacity_after_each_admission() {
    let profile = policy_profile(
        r#"
default_policy_family: capped
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: uncapped
    policy_family: uncapped
    cache_bucket: all
    quantum: 1
  - name: capped
    policy_family: capped
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 0
"#,
    );
    let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);
    let worker = WorkerWithDpRank::new(0, 0);
    let (mut active, active_rx) = make_request("active", 64);
    active.policy_class = Some("uncapped".into());
    queue.enqueue(active).await;
    active_rx.await.unwrap().unwrap();

    let (mut first, mut first_rx) = make_request("first", 64);
    first.pinned_worker = Some(worker);
    queue.enqueue(first).await;
    let (second, mut second_rx) = make_request("second", 64);
    queue.enqueue(second).await;
    assert_eq!(queue.pending_count(), 2);

    // The uncapped class can still run while the capped class waits.
    let (mut uncapped, uncapped_rx) = make_request("uncapped", 64);
    uncapped.policy_class = Some("uncapped".into());
    queue.enqueue(uncapped).await;
    assert_eq!(uncapped_rx.await.unwrap().unwrap().best_worker, worker);
    assert_eq!(queue.pending_count(), 2);
    assert!(matches!(
        first_rx.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));
    assert!(matches!(
        second_rx.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));

    for id in ["active", "uncapped"] {
        slots
            .mark_prefill_completed(&id.to_string(), decay_now())
            .unwrap();
        slots.free(&id.to_string(), decay_now()).unwrap();
    }
    queue.update().await;
    assert_eq!(first_rx.try_recv().unwrap().unwrap().best_worker, worker);
    // Reusing the pre-admission snapshot would also admit the second request.
    assert!(matches!(
        second_rx.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));
    assert_eq!(queue.pending_count(), 1);

    slots
        .mark_prefill_completed(&"first".to_string(), decay_now())
        .unwrap();
    slots.free(&"first".to_string(), decay_now()).unwrap();
    queue.update().await;
    assert_eq!(second_rx.try_recv().unwrap().unwrap().best_worker, worker);
    assert_eq!(queue.pending_count(), 0);
    slots
        .mark_prefill_completed(&"second".to_string(), decay_now())
        .unwrap();
    slots.free(&"second".to_string(), decay_now()).unwrap();
    slots.assert_completely_drained(decay_now());
}

#[tokio::test]
async fn zero_class_queue_limit_never_queues() {
    let profile = policy_profile(
        r#"
default_policy_family: capped
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: capped
    policy_family: capped
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 0
    request_queue_limit_per_worker: 0
"#,
    );
    let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);

    // An idle worker admits a zero-limit request directly, bypassing the
    // policy queue.
    let (admitted, admitted_rx) = make_request("admitted-idle", 64);
    queue.enqueue(admitted).await;
    admitted_rx.await.unwrap().unwrap();
    assert_eq!(queue.pending_count(), 0);

    // With every worker busy the request would have to queue, so the zero
    // limit rejects it instead of parking it.
    let (rejected, rejected_rx) = make_request("rejected-busy", 64);
    queue.enqueue(rejected).await;
    assert!(matches!(
        rejected_rx.await.unwrap(),
        Err(KvSchedulerError::QueueRejected(_))
    ));
    assert_eq!(queue.pending_count(), 0);

    slots
        .mark_prefill_completed(&"admitted-idle".to_string(), decay_now())
        .unwrap();
    slots
        .free(&"admitted-idle".to_string(), decay_now())
        .unwrap();
    slots.assert_completely_drained(decay_now());
}

#[tokio::test]
async fn zero_token_queue_limit_admits_idle_workers() {
    let profile = policy_profile(
        r#"
default_policy_family: capped
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: capped
    policy_family: capped
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 0
    raw_isl_token_queue_limit_per_worker: 0
"#,
    );
    let (queue, _slots) = make_queue_with_profile(1, 16, 64, profile);

    // A zero token limit means the class can never hold a queued entry, so
    // an idle worker must still admit directly instead of tripping the
    // limit-zero rejection.
    let (admitted, admitted_rx) = make_request("admitted-idle-token-limit", 64);
    queue.enqueue(admitted).await;
    admitted_rx.await.unwrap().unwrap();
    assert_eq!(queue.pending_count(), 0);

    let (rejected, rejected_rx) = make_request("rejected-busy-token-limit", 64);
    queue.enqueue(rejected).await;
    assert!(matches!(
        rejected_rx.await.unwrap(),
        Err(KvSchedulerError::QueueRejected(_))
    ));
    assert_eq!(queue.pending_count(), 0);
}

#[tokio::test]
async fn zero_workers_report_no_endpoints_not_queue_rejection() {
    let profile = policy_profile(
        r#"
default_policy_family: capped
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: capped
    policy_family: capped
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 0
    request_queue_limit_per_worker: 4
"#,
    );
    let (queue, _slots) = make_queue_with_profile(0, 16, 64, profile);

    // With zero discovered workers every per-worker limit scales to zero;
    // the request must surface the real condition instead of a
    // `QueueRejected { limit: 0 }`.
    let (request, response_rx) = make_request("no-workers", 64);
    queue.enqueue(request).await;
    assert!(matches!(
        response_rx.await.unwrap(),
        Err(KvSchedulerError::NoEndpoints)
    ));
    assert_eq!(queue.pending_count(), 0);
}

#[tokio::test]
async fn direct_release_wakes_queued_and_mixed_classes_with_full_admission_channel() {
    let mixed = policy_profile(
        r#"
default_policy_family: queued
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: direct
    policy_family: direct
    cache_bucket: all
    quantum: 1
  - name: queued
    policy_family: queued
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 32
"#,
    );
    for profile in [
        PolicyProfile::synthetic(Some(0.5), RouterQueuePolicy::Fcfs),
        mixed,
    ] {
        for overflow in [false, true] {
            let (queue, slots) = make_queue_with_profile(1, 16, 64, profile.clone());
            let (mut active, active_rx) = make_request("active", 64);
            // In the mixed profile, capacity released by a direct class
            // must wake requests waiting in the queued class.
            active.policy_class = Some("direct".into());
            let (attempt_tx, attempt_rx) = oneshot::channel();
            queue
                .enqueue_admitted_with_block_hashes_and_lease(
                    active,
                    None,
                    None,
                    Some(attempt_tx),
                    None,
                    Instant::now(),
                )
                .await;
            let worker = active_rx.await.unwrap().unwrap().best_worker;
            let booking = SchedulerBookingDescriptor {
                request_id: "active".into(),
                worker,
                attempt_id: attempt_rx.await.unwrap(),
            };
            let (mut queued, queued_rx) = make_request("queued", 64);
            queued.pinned_worker = Some(worker);
            queue.enqueue(queued).await;
            assert_eq!(queue.pending_count(), 1);

            // Hold every admission permit. Capacity progress must not need
            // space in the command channel, or a periodic update task.
            let permits = queue
                .admission_tx
                .reserve_many(queue.admission_tx.max_capacity())
                .await
                .unwrap();
            let released = queue.booking_cleanup().enqueue_acknowledged(booking);
            assert!(released.response.is_ok());
            assert_eq!(slots.active_tokens(Instant::now())[&worker], 0);
            if overflow {
                for _ in 0..300 {
                    queue.capacity_changed(Some(worker));
                }
            }
            let response = tokio::time::timeout(Duration::from_secs(1), queued_rx)
                .await
                .expect("capacity wake must drain without command permits")
                .unwrap()
                .unwrap();
            assert_eq!(response.best_worker, worker);
            assert_eq!(queue.pending_count(), 0);
            drop(permits);
            slots.free(&"queued".to_owned(), Instant::now()).unwrap();
            queue.update().await;
            slots.assert_completely_drained(Instant::now());
        }
    }
}

// The caller drives this actor directly. Its hints are separate from the
// idle actor spawned by make_queue, so tests control notification ordering.
fn actor_for_test<Sel: WorkerSelector<SimpleWorkerConfig>>(
    queue: &SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, Sel>,
    slots: &Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    selector: Sel,
) -> SchedulerQueueActor<NoopSequencePublisher, SimpleWorkerConfig, Sel, NoopOverlapScoresRefresh> {
    SchedulerQueueActor {
        pending: PolicyQueue::new(&queue.profile),
        cleanup: Arc::clone(&queue.cleanup),
        capacity_updates: None,
        capacity_workers: FxHashSet::default(),
        profile: Arc::clone(&queue.profile),
        pending_count: Arc::clone(&queue.pending_count),
        pending_isl_tokens: Arc::clone(&queue.pending_isl_tokens),
        class_counters: Arc::clone(&queue.class_counters),
        slots: Arc::clone(slots),
        workers_with_configs: queue.workers_with_configs.clone(),
        workers_generation: Arc::clone(&queue.workers_generation),
        candidate_preparer: Arc::clone(&queue.candidate_preparer),
        turn: 0,
        turn_prepared: None,
        projected_loads: FxHashMap::with_capacity_and_hasher(8, Default::default()),
        gate_loads: FxHashMap::default(),
        full_ranks: FullRankCounts::default(),
        block_size: 16,
        selector,
        prefill_load_estimator: None,
        overlap_scores_refresh: None::<Arc<NoopOverlapScoresRefresh>>,
        overlap_refresh_after: None,
        overloaded_worker_provider: None,
        available_worker_provider: None,
        non_max_overlap_selection_observer: Arc::clone(&queue.non_max_overlap_selection_observer),
        selection_trace: Arc::clone(&queue.selection_trace),
        reject_thresholds: queue.reject_thresholds.clone(),
    }
}

/// Delegates to a composed policy and counts selections from prepared rows.
struct CountingPreparedPolicy {
    inner: WorkerSelectionPolicy,
    prepared_selections: Arc<AtomicUsize>,
}

impl WorkerSelector<SimpleWorkerConfig> for CountingPreparedPolicy {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerSelector::<SimpleWorkerConfig>::required_worker_inputs(&self.inner)
    }

    fn select_worker(
        &self,
        input: WorkerSelectionInput<'_, SimpleWorkerConfig>,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        self.inner.select_worker(input)
    }

    fn candidate_preparer(&self) -> Option<CandidatePreparer> {
        WorkerSelector::<SimpleWorkerConfig>::candidate_preparer(&self.inner)
    }

    fn select_worker_prepared(
        &self,
        input: WorkerSelectionInput<'_, SimpleWorkerConfig>,
        prepared: Option<PreparedCandidates>,
        trace: Option<SelectionTraceSpec>,
    ) -> Result<(WorkerSelectionResult, Option<SelectionTrace>), KvSchedulerError> {
        if prepared.is_some() {
            self.prepared_selections.fetch_add(1, Ordering::Relaxed);
        }
        self.inner.select_worker_prepared(input, prepared, trace)
    }
}

/// Picks the least-loaded worker; ties go to the lowest worker.
struct LeastLoaded;

impl WorkerPicker for LeastLoaded {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let (candidates, loads) = (input.candidates(), input.load().unwrap());
        Ok((0..candidates.len())
            .min_by_key(|&i| {
                let load = loads[i].active_requests() + loads[i].active_prefill_tokens();
                (load, candidates[i].worker())
            })
            .unwrap())
    }
}

fn counting_policy(prepared_selections: &Arc<AtomicUsize>) -> CountingPreparedPolicy {
    CountingPreparedPolicy {
        inner: WorkerSelectionPolicy::new("test", vec![], Box::new(LeastLoaded)),
        prepared_selections: Arc::clone(prepared_selections),
    }
}

/// Two workers behind a counting policy, with availability and overload the
/// test can change.
struct PreparedQueue {
    queue: Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, CountingPreparedPolicy>>,
    slots: Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    cfg_tx: watch::Sender<HashMap<u64, SimpleWorkerConfig>>,
    available: Arc<StdMutex<Option<Arc<HashSet<WorkerId>>>>>,
    overloaded: Arc<StdMutex<Option<HashSet<WorkerId>>>>,
    prepared_selections: Arc<AtomicUsize>,
}

impl PreparedQueue {
    fn prepared_selections(&self) -> usize {
        self.prepared_selections.load(Ordering::Relaxed)
    }
}

fn make_prepared_queue() -> PreparedQueue {
    let slots = Arc::new(ActiveSequencesMultiWorker::new(
        NoopSequencePublisher,
        16,
        (0..2).map(|id| (id, (0, 1))).collect(),
        false,
        0,
        "test",
    ));
    let configs = (0..2)
        .map(|id| {
            let config = SimpleWorkerConfig {
                max_num_batched_tokens: Some(64),
                ..Default::default()
            };
            (id, config)
        })
        .collect();
    let (cfg_tx, cfg_rx) = watch::channel(configs);
    let available = Arc::new(StdMutex::new(None));
    let available_provider: WorkerAvailabilityProvider = {
        let available = Arc::clone(&available);
        Arc::new(move |_| available.lock().unwrap().clone())
    };
    let overloaded = Arc::new(StdMutex::new(None));
    let overloaded_provider: OverloadedWorkerProvider = {
        let overloaded = Arc::clone(&overloaded);
        Arc::new(move || overloaded.lock().unwrap().clone())
    };
    let prepared_selections = Arc::new(AtomicUsize::new(0));
    let queue = Arc::new(SchedulerQueue::new(
        Arc::clone(&slots),
        cfg_rx,
        PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs),
        16,
        counting_policy(&prepared_selections),
        None,
        None,
        Some(overloaded_provider),
        Some(available_provider),
        None,
    ));
    PreparedQueue {
        queue,
        slots,
        cfg_tx,
        available,
        overloaded,
        prepared_selections,
    }
}

/// Enqueue with rows prepared earlier, as if the caller were slow to send.
async fn enqueue_prepared(
    queue: &SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, CountingPreparedPolicy>,
    request: SchedulingRequest,
    prepared: Option<Box<PreparedAdmission>>,
) {
    let queue_metadata = queue.default_queue_metadata(&request, Instant::now());
    let (ack_tx, ack_rx) = oneshot::channel();
    queue
        .admission_tx
        .send(AdmissionCommand::Enqueue {
            request,
            attempt_tx: None,
            block_hashes: None,
            prepared,
            queue_metadata,
            lease: None,
            ack_tx,
        })
        .await
        .unwrap();
    ack_rx.await.unwrap();
}

#[tokio::test]
async fn prepared_rows_see_bookings_made_after_preparation() {
    let prepared_queue = make_prepared_queue();
    let queue = &prepared_queue.queue;
    let (later, later_rx) = make_request("later", 64);
    let prepared = queue.prepare_candidates(&later);
    assert!(prepared.is_some());

    let (earlier, earlier_rx) = make_request("earlier", 64);
    queue.enqueue(earlier).await;
    let earlier = earlier_rx.await.unwrap().unwrap().best_worker;
    assert_eq!(earlier, WorkerWithDpRank::new(0, 0));

    // Load inputs are read at selection, so the earlier booking moves this request.
    enqueue_prepared(queue, later, prepared).await;
    let later = later_rx.await.unwrap().unwrap().best_worker;
    assert_eq!(later, WorkerWithDpRank::new(1, 0));
    assert_eq!(prepared_queue.prepared_selections(), 2);
}

#[tokio::test]
async fn prepared_rows_are_rebuilt_when_selection_inputs_change() {
    for change in ["configuration", "availability", "overload"] {
        let prepared_queue = make_prepared_queue();
        let (request, rx) = make_request("request", 64);
        let prepared = prepared_queue.queue.prepare_candidates(&request);
        // Worker 0 wins on the prepared rows; each change rules it out afterwards.
        match change {
            "configuration" => prepared_queue.cfg_tx.send_modify(|configs| {
                configs.remove(&0);
            }),
            "availability" => {
                *prepared_queue.available.lock().unwrap() = Some(Arc::new(HashSet::from([1])))
            }
            _ => *prepared_queue.overloaded.lock().unwrap() = Some(HashSet::from([0])),
        }

        enqueue_prepared(&prepared_queue.queue, request, prepared).await;
        let selected = rx.await.unwrap().unwrap().best_worker;
        assert_eq!(selected, WorkerWithDpRank::new(1, 0), "{change}");
        assert_eq!(prepared_queue.prepared_selections(), 0, "{change}");
    }
}

#[tokio::test]
async fn prepared_rows_recheck_config_under_selection_borrow() {
    let prepared_queue = make_prepared_queue();
    let mut actor = actor_for_test(
        &prepared_queue.queue,
        &prepared_queue.slots,
        counting_policy(&prepared_queue.prepared_selections),
    );
    let (request, mut rx) = make_request("request", 64);
    let prepared = prepared_queue.queue.prepare_candidates(&request);
    assert!(prepared.is_some());

    // Worker 0 leaves after the actor last advanced the generation.
    actor.refresh_workers_generation();
    prepared_queue.cfg_tx.send_modify(|configs| {
        configs.remove(&0);
    });
    assert!(
        actor.with_projected_loads(request, Instant::now(), |actor, request| {
            actor.admit_one_projected(request, None, None, prepared)
        })
    );
    let selected = rx.try_recv().unwrap().unwrap().best_worker;
    assert_eq!(selected, WorkerWithDpRank::new(1, 0));
    assert_eq!(prepared_queue.prepared_selections(), 0);
}

#[tokio::test]
async fn gate_snapshot_reuses_the_actor_scratch_across_admissions() {
    // Queueing enabled, nothing ever full: every enqueue runs the gate and admits.
    let (queue, slots) = make_queue(1, 16, 64, Some(1.0));
    let mut actor = actor_for_test(&queue, &slots, DefaultWorkerSelector::new(None, "test"));
    assert_eq!(actor.gate_loads.capacity(), 0);

    let (first, mut first_rx) = make_request("first", 64);
    let metadata = queue.default_queue_metadata(&first, Instant::now());
    assert!(actor.handle_enqueue(first, None, None, None, None, metadata));
    actor.handle_enqueued().await;
    assert!(first_rx.try_recv().unwrap().is_ok());
    let capacity = actor.gate_loads.capacity();
    assert!(
        capacity > 0,
        "the gate projected load into the actor's scratch"
    );

    let (second, _second_rx) = make_request("second", 64);
    let metadata = queue.default_queue_metadata(&second, Instant::now());
    assert!(actor.handle_enqueue(second, None, None, None, None, metadata));
    actor.handle_enqueued().await;
    assert_eq!(
        actor.gate_loads.capacity(),
        capacity,
        "a later gate pass refills the same allocation"
    );
}

#[tokio::test]
async fn enqueue_rechecks_capacity_before_later_shared_admission() {
    for overflow in [false, true] {
        let (queue, slots) = make_queue(1, 16, 64, Some(0.0));
        let mut actor = actor_for_test(&queue, &slots, DefaultWorkerSelector::new(None, "test"));
        let updates = Arc::new(CapacityUpdates::new());
        actor.capacity_updates = Some(Arc::clone(&updates));
        let worker = WorkerWithDpRank::new(0, 0);

        let (active, mut active_rx) = make_request("active", 64);
        let metadata = queue.default_queue_metadata(&active, Instant::now());
        assert!(actor.handle_enqueue(active, None, None, None, None, metadata));
        actor.handle_enqueued().await;
        assert_eq!(active_rx.try_recv().unwrap().unwrap().best_worker, worker);

        let (mut older, mut older_rx) = make_request("older-pinned", 64);
        older.pinned_worker = Some(worker);
        let metadata = queue.default_queue_metadata(&older, Instant::now());
        assert!(actor.handle_enqueue(older, None, None, None, None, metadata));
        actor.handle_enqueued().await;
        assert!(older_rx.try_recv().is_err());

        slots.free(&"active".to_owned(), Instant::now()).unwrap();
        for _ in 0..if overflow { 257 } else { 1 } {
            updates.record(Some(worker));
        }
        let (later, mut later_rx) = make_request("later-shared", 64);
        let metadata = queue.default_queue_metadata(&later, Instant::now());
        assert!(actor.handle_enqueue(later, None, None, None, None, metadata));
        // Force enqueue processing before the capacity select branch.
        actor.handle_enqueued().await;
        assert_eq!(older_rx.try_recv().unwrap().unwrap().best_worker, worker);
        assert!(later_rx.try_recv().is_err());
        assert_eq!(queue.pending_count(), 1);

        slots
            .free(&"older-pinned".to_owned(), Instant::now())
            .unwrap();
        updates.record(Some(worker));
        actor.handle_capacity_updates(false).await;
        assert_eq!(later_rx.try_recv().unwrap().unwrap().best_worker, worker);
        slots
            .free(&"later-shared".to_owned(), Instant::now())
            .unwrap();
        slots.assert_completely_drained(Instant::now());
    }
}

#[tokio::test]
async fn cleanup_progresses_with_admission_commands_and_capacity_wakes() {
    let (queue, slots) = make_queue(1, 16, 64, Some(0.0));
    let booking = book_directly(&slots, "abandoned");
    let mut permits = queue
        .admission_tx
        .reserve_many(queue.admission_tx.max_capacity())
        .await
        .unwrap();
    // No permit is available for the destructor's explicit Cleanup command.
    drop(queue.booking_handle(booking.clone()));
    assert!(slots.has_booking(&booking));
    let mut responses = Vec::new();
    for _ in 0..512 {
        let (request, _rx) = make_request("probe", 64);
        let (resp_tx, resp_rx) = oneshot::channel();
        permits
            .next()
            .unwrap()
            .send(AdmissionCommand::SelectWithoutAdmission { request, resp_tx });
        responses.push(resp_rx);
        queue.capacity_changed(Some(booking.worker));
    }
    tokio::time::timeout(Duration::from_secs(2), async {
        for response in responses {
            response.await.unwrap().unwrap();
        }
    })
    .await
    .expect("admission commands must progress with capacity wakes");
    assert!(!slots.has_booking(&booking));
    drop(permits);
    slots.assert_completely_drained(Instant::now());
}

#[tokio::test]
async fn runtime_control_drains_cleanup_when_the_command_channel_was_full() {
    let (queue, slots) = make_queue(1, 16, 64, Some(0.0));
    let booking = book_directly(&slots, "abandoned");
    let mut permits = queue
        .admission_tx
        .reserve_many(queue.admission_tx.max_capacity())
        .await
        .unwrap();
    drop(queue.booking_handle(booking.clone()));
    assert!(slots.has_booking(&booking));
    let (response, receiver) = oneshot::channel();
    permits
        .next()
        .unwrap()
        .send(AdmissionCommand::PolicyConfig {
            update: None,
            response,
        });
    let result = tokio::time::timeout(Duration::from_secs(2), receiver)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(result, Err(RuntimePolicyError::Unsupported)));
    assert!(!slots.has_booking(&booking));
    drop(permits);
    slots.assert_completely_drained(Instant::now());
}

#[tokio::test]
async fn projection_storage_survives_admission_and_errors_without_stale_entries() {
    let (queue, slots) = make_queue(1, 16, 64, None);
    let mut actor = actor_for_test(&queue, &slots, DefaultWorkerSelector::new(None, "test"));
    let worker = WorkerWithDpRank::new(0, 0);
    let stale = WorkerWithDpRank::new(999, 0);
    let capacity = actor.projected_loads.capacity();
    // The second admission refreshes load after the first booking. The
    // duplicate then exercises booking failure with the same allocation.
    for (id, succeeds) in [("first", true), ("second", true), ("first", false)] {
        actor
            .projected_loads
            .insert(stale, WorkerLoadProjection::default());
        let now = Instant::now();
        let expected = slots.project_worker_loads(None, now);
        let (request, mut rx) = make_request(id, 64);
        let storage = actor.with_projected_loads(request, now, |actor, request| {
            assert_eq!(request.worker_loads, expected);
            assert!(!request.worker_loads.contains_key(&stale));
            let storage = request.worker_loads.get(&worker).unwrap() as *const _;
            assert_eq!(
                actor.admit_one_projected(request, None, None, None),
                succeeds
            );
            assert_eq!(rx.try_recv().unwrap().is_ok(), succeeds);
            assert_eq!(request.worker_loads, expected);
            storage
        });
        assert_eq!(actor.projected_loads.capacity(), capacity);
        assert_eq!(
            actor.projected_loads.get(&worker).unwrap() as *const _,
            storage
        );
    }
    let (mut request, _rx) = make_request("missing-worker", 64);
    request.pinned_worker = Some(stale);
    assert!(
        actor
            .select_without_admission_inner(request, Instant::now())
            .is_err()
    );
    assert_eq!(actor.projected_loads.capacity(), capacity);
    for id in ["first", "second"] {
        slots.free(&id.to_owned(), Instant::now()).unwrap();
    }
    slots.assert_completely_drained(Instant::now());
}

#[test]
fn non_max_overlap_selection_ignores_ties_pins_and_ineligible_workers() {
    let workers = HashMap::from([
        (0, SimpleWorkerConfig::default()),
        (1, SimpleWorkerConfig::default()),
    ]);
    let worker0 = WorkerWithDpRank::new(0, 0);
    let worker1 = WorkerWithDpRank::new(1, 0);
    let (mut request, _rx) = make_request("locality-exclusions", 64);
    request
        .overlap
        .effective_overlap_blocks
        .extend([(worker0, 8.0), (worker1, 8.0)]);
    assert!(
        non_max_overlap_selection(&workers, &request, request.eligibility(), worker1, 8.0)
            .is_none()
    );

    request
        .overlap
        .effective_overlap_blocks
        .insert(worker1, 2.0);
    request.pinned_worker = Some(worker1);
    assert!(
        non_max_overlap_selection(&workers, &request, request.eligibility(), worker1, 2.0)
            .is_none()
    );

    request.pinned_worker = None;
    request.allowed_worker_ids = Some(HashSet::from([worker1.worker_id]));
    assert!(
        non_max_overlap_selection(&workers, &request, request.eligibility(), worker1, 2.0)
            .is_none()
    );
}

#[tokio::test]
async fn scheduler_observer_receives_non_max_overlap_selection() {
    let (queue, _slots) =
        make_queue_with_custom_selector(2, 16, 64, None, MinDecodeSelector { rendezvous: None });
    let worker0 = WorkerWithDpRank::new(0, 0);
    let worker1 = WorkerWithDpRank::new(1, 0);
    let (observer_tx, mut observer_rx) = tokio::sync::mpsc::unbounded_channel();
    assert!(
        queue.set_non_max_overlap_selection_observer(Arc::new(move |request_id, event| {
            observer_tx
                .send((request_id.to_string(), event))
                .expect("observer receiver should remain open");
        }))
    );
    let (mut request, response_rx) = make_request("locality-response", 64);
    request
        .overlap
        .effective_overlap_blocks
        .extend([(worker0, 1.0), (worker1, 4.0)]);

    queue.enqueue(request).await;
    let response = response_rx.await.unwrap().unwrap();

    assert_eq!(response.best_worker, worker0);
    let event = tokio::time::timeout(Duration::from_secs(1), observer_rx.recv())
        .await
        .expect("observer did not run")
        .expect("observer channel closed");
    assert_eq!(event.1.overlap_blocks_lost(), 3.0);
    assert_eq!(
        event,
        (
            "locality-response".to_string(),
            NonMaxOverlapSelection {
                selected_worker: worker0,
                highest_overlap_worker: worker1,
                highest_overlap_blocks: 4.0,
                selected_overlap_blocks: 1.0,
            },
        )
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scheduler_observer_does_not_block_actor() {
    let (queue, _slots) =
        make_queue_with_custom_selector(2, 16, 64, None, MinDecodeSelector { rendezvous: None });
    let observer_gate = Arc::new((StdMutex::new(false), Condvar::new()));
    let callback_gate = Arc::clone(&observer_gate);
    let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
    let (finished_tx, mut finished_rx) = tokio::sync::mpsc::unbounded_channel();
    assert!(
        queue.set_non_max_overlap_selection_observer(Arc::new(move |_request_id, _event| {
            started_tx.send(()).unwrap();
            let (released, wake) = &*callback_gate;
            let mut released = released.lock().unwrap();
            while !*released {
                released = wake.wait(released).unwrap();
            }
            finished_tx.send(()).unwrap();
        }))
    );

    let worker0 = WorkerWithDpRank::new(0, 0);
    let worker1 = WorkerWithDpRank::new(1, 0);
    let (mut first, first_rx) = make_request("blocking-observer", 64);
    first
        .overlap
        .effective_overlap_blocks
        .extend([(worker0, 1.0), (worker1, 4.0)]);
    let first_queue = Arc::clone(&queue);
    let first_enqueue = tokio::spawn(async move {
        first_queue.enqueue(first).await;
    });

    tokio::time::timeout(Duration::from_secs(1), started_rx.recv())
        .await
        .expect("observer did not start")
        .expect("observer start channel closed");

    let (second, second_rx) = make_request("actor-remains-responsive", 64);
    tokio::time::timeout(Duration::from_secs(1), queue.enqueue(second))
        .await
        .expect("observer blocked the scheduler actor");

    let (released, wake) = &*observer_gate;
    *released.lock().unwrap() = true;
    wake.notify_one();

    first_enqueue.await.unwrap();
    first_rx.await.unwrap().unwrap();
    second_rx.await.unwrap().unwrap();
    tokio::time::timeout(Duration::from_secs(1), finished_rx.recv())
        .await
        .expect("observer did not finish")
        .expect("observer finish channel closed");
}

#[tokio::test]
async fn scheduler_observer_ignores_advisory_and_abandoned_requests() {
    let (queue, _slots) =
        make_queue_with_custom_selector(2, 16, 64, None, MinDecodeSelector { rendezvous: None });
    let worker0 = WorkerWithDpRank::new(0, 0);
    let worker1 = WorkerWithDpRank::new(1, 0);
    let observed = Arc::new(StdMutex::new(Vec::new()));
    let observer_events = Arc::clone(&observed);
    assert!(
        queue.set_non_max_overlap_selection_observer(Arc::new(move |request_id, event| {
            observer_events
                .lock()
                .unwrap()
                .push((request_id.to_string(), event));
        }))
    );

    let (mut advisory, _advisory_rx) = make_request("locality-advisory", 64);
    advisory
        .overlap
        .effective_overlap_blocks
        .extend([(worker0, 1.0), (worker1, 4.0)]);
    let response = queue.select_without_admission(advisory).await.unwrap();
    assert_eq!(response.response.best_worker, worker0);

    let (mut abandoned, abandoned_rx) = make_request("locality-abandoned", 64);
    abandoned
        .overlap
        .effective_overlap_blocks
        .extend([(worker0, 1.0), (worker1, 4.0)]);
    drop(abandoned_rx);
    queue.enqueue(abandoned).await;

    assert!(observed.lock().unwrap().is_empty());
}

#[tokio::test]
async fn disabled_queueing_lifecycle_lease_releases_an_admitted_booking_on_cancellation() {
    let isl = 512;
    let (queue, slots) = make_queue(1, 16, isl, None);
    let request_id = "default-path-cancelled";
    let (mut request, response_rx) = make_request(request_id, isl);
    request.mode = ScheduleMode::TrackedWithLifecycle {
        request_id: request_id.to_string(),
    };

    let lease = queue
        .new_request_lifecycle_lease(Some(request_id))
        .expect("lifecycle-tracked request must receive a cancellation lease");
    let lease = queue
        .enqueue_with_block_hashes_and_lease(request, None, Some(lease))
        .await
        .expect("scheduler must return the admitted request lease");
    response_rx
        .await
        .expect("scheduler response sender dropped")
        .expect("request should be admitted before cancellation");

    drop(lease);
    tokio::time::timeout(Duration::from_secs(1), async {
        while slots.any_worker_matches_active_tokens(Instant::now(), |_, tokens| tokens != 0) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("cancellation lease cleanup did not release the admitted booking");
    slots.assert_completely_drained(decay_now());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_cancelled_pending_request_is_not_booked() {
    let isl = 512;
    let (queue, slots) = make_queue(1, 16, isl, Some(0.0));

    let (first, first_rx) = make_request("first", isl);
    queue.enqueue(first).await;
    first_rx
        .await
        .expect("first response sender dropped")
        .expect("first request should be scheduled");

    let (cancelled, cancelled_rx) = make_request("cancelled", isl);
    queue.enqueue(cancelled).await;
    assert_eq!(queue.pending_count(), 1);
    drop(cancelled_rx);

    slots.free(&"first".to_string(), decay_now()).unwrap();
    queue.update().await;

    assert_eq!(queue.pending_count(), 0);
    slots.assert_completely_drained(decay_now());
}

#[tokio::test]
async fn dropped_legacy_lease_retracts_pending_request_immediately() {
    let isl = 512;
    let (queue, slots) = make_queue(1, 16, isl, Some(0.0));

    let (first, first_rx) = make_request("legacy-first", isl);
    queue.enqueue(first).await;
    first_rx.await.unwrap().unwrap();

    let (cancelled, cancelled_rx) = make_request("legacy-cancelled", isl);
    let lease = queue
        .new_request_lifecycle_lease(Some("legacy-cancelled"))
        .unwrap();
    let lease = queue
        .enqueue_with_block_hashes_and_lease(cancelled, None, Some(lease))
        .await
        .unwrap();
    assert_eq!(queue.pending_count(), 1);

    drop(cancelled_rx);
    drop(lease);
    tokio::time::timeout(Duration::from_secs(1), async {
        while queue.pending_count() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("legacy lease did not retract pending request");

    slots.free(&"legacy-first".to_owned(), decay_now()).unwrap();
    slots.assert_completely_drained(decay_now());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_strict_priority_drains_before_policy_score() {
    let isl = 512;
    let (queue, slots) = make_queue(1, 16, isl, Some(0.0));

    let (first, first_rx) = make_request("first", isl);
    queue.enqueue(first).await;
    first_rx.await.unwrap().unwrap();

    let (mut low, mut low_rx) = make_request("low", isl);
    low.priority_jump = 10_000.0;
    queue.enqueue(low).await;

    let (mut high, high_rx) = make_request("high", isl);
    high.strict_priority = 1;
    queue.enqueue(high).await;
    assert_eq!(queue.pending_count(), 2);

    slots.free(&"first".to_string(), decay_now()).unwrap();
    queue.update().await;

    let high_response = high_rx.await.unwrap().unwrap();
    assert_eq!(high_response.best_worker, WorkerWithDpRank::new(0, 0));
    assert!(
        low_rx.try_recv().is_err(),
        "lower strict priority should remain queued"
    );

    slots.free(&"high".to_string(), decay_now()).unwrap();
    queue.update().await;
    low_rx.await.unwrap().unwrap();
    assert_eq!(queue.pending_count(), 0);

    slots.free(&"low".to_string(), decay_now()).unwrap();
    slots.assert_completely_drained(decay_now());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_failed_response_delivery_rolls_back_booking() {
    let isl = 512;
    let response_rx = Arc::new(StdMutex::new(None));
    let publisher = DropResponseOnLoadPublisher {
        response_rx: Arc::clone(&response_rx),
    };
    let slots = Arc::new(ActiveSequencesMultiWorker::new(
        publisher,
        16,
        HashMap::from([(0, (0, 1))]),
        false,
        0,
        "test",
    ));
    let (_cfg_tx, cfg_rx) = watch::channel(HashMap::from([(
        0,
        SimpleWorkerConfig {
            max_num_batched_tokens: Some(isl as u64),
            ..Default::default()
        },
    )]));
    let queue = SchedulerQueue::new(
        Arc::clone(&slots),
        cfg_rx,
        PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs),
        16,
        DefaultWorkerSelector::new(None, "test"),
        None,
        None::<Arc<NoopOverlapScoresRefresh>>,
        None,
        None,
        None,
    );

    let (request, receiver) = make_request("delivery-race", isl);
    *response_rx.lock().unwrap() = Some(receiver);
    queue.enqueue(request).await;

    assert!(response_rx.lock().unwrap().is_none());
    slots.assert_completely_drained(decay_now());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_flood() {
    let block_size = 16;
    let isl = 512;
    let num_workers = 4;
    let num_tasks = 25;

    let (queue, slots) = make_queue(num_workers, block_size, isl, None);

    let mut handles = Vec::new();
    for i in 0..num_tasks {
        let queue = Arc::clone(&queue);
        let slots = Arc::clone(&slots);
        handles.push(tokio::spawn(async move {
            let req_id = format!("req-{i}");
            let (req, rx) = make_request(&req_id, isl);
            queue.enqueue(req).await;
            let resp = rx.await.expect("oneshot dropped");
            let resp = resp.expect("scheduling failed");
            assert!(resp.best_worker.worker_id < num_workers as u64);

            slots.mark_prefill_completed(&req_id, decay_now()).unwrap();
            slots.free(&req_id, decay_now()).unwrap();
            queue.update().await;
        }));
    }

    for h in handles {
        h.await.expect("task panicked");
    }

    let active = slots.active_tokens(decay_now());
    for (worker, tokens) in &active {
        assert_eq!(
            *tokens, 0,
            "worker {worker:?} still has {tokens} active tokens"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_immediate_admissions_see_prior_booking() {
    let selector = MinDecodeSelector {
        rendezvous: Some(Arc::new(SelectorRendezvous::default())),
    };
    let (queue, slots) = make_queue_with_custom_selector(2, 16, 512, None, selector);
    let barrier = Arc::new(Barrier::new(3));

    let (req1, rx1) = make_request("req-1", 512);
    let queue1 = Arc::clone(&queue);
    let barrier1 = Arc::clone(&barrier);
    let handle1 = tokio::spawn(async move {
        barrier1.wait().await;
        queue1.enqueue(req1).await;
    });

    let (req2, rx2) = make_request("req-2", 512);
    let queue2 = Arc::clone(&queue);
    let barrier2 = Arc::clone(&barrier);
    let handle2 = tokio::spawn(async move {
        barrier2.wait().await;
        queue2.enqueue(req2).await;
    });

    barrier.wait().await;
    handle1.await.unwrap();
    handle2.await.unwrap();

    let resp1 = rx1.await.unwrap().unwrap();
    let resp2 = rx2.await.unwrap().unwrap();
    assert_ne!(
        resp1.best_worker, resp2.best_worker,
        "second admission should see the first booking and choose the other idle worker"
    );

    for request_id in ["req-1", "req-2"] {
        slots
            .mark_prefill_completed(&request_id.to_string(), decay_now())
            .unwrap();
        slots.free(&request_id.to_string(), decay_now()).unwrap();
    }
}

#[tokio::test]
async fn reject_thresholds_shed_full_workers_and_reject_when_none_remain() {
    let profile = PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs);
    let (queue, slots, cfg_tx) = make_queue_with_profile_sender_and_reject(
        2,
        16,
        1_000_000,
        profile,
        Some(Arc::new(|| crate::scheduling_compat::BusyThresholds {
            decode_busy_threshold_frac: Some(0.5),
            ..Default::default()
        })),
    );
    let worker_config = SimpleWorkerConfig {
        max_num_batched_tokens: Some(1_000_000),
        total_kv_blocks: Some(8),
        ..Default::default()
    };
    cfg_tx
        .send(HashMap::from([
            (0, worker_config.clone()),
            (1, worker_config),
        ]))
        .unwrap();
    set_worker_capacities_from_configs(&slots, &cfg_tx.borrow());
    // Hold 6 of 8 KV blocks on `worker` with a decoding request.
    let hold = |request_id: &'static str, worker: WorkerWithDpRank| {
        let (mut held, held_rx) = make_request(request_id, 96);
        held.token_seq = Some((0..6).collect());
        held.pinned_worker = Some(worker);
        (held, held_rx)
    };
    let first = WorkerWithDpRank::new(0, 0);
    let second = WorkerWithDpRank::new(1, 0);
    let (held, held_rx) = hold("held-0", first);
    queue.enqueue(held).await;
    held_rx.await.unwrap().unwrap();
    slots
        .mark_prefill_completed(&"held-0".to_string(), decay_now())
        .unwrap();

    let (next, next_rx) = make_request("next", 16);
    queue.enqueue(next).await;
    assert_eq!(
        next_rx.await.unwrap().unwrap().best_worker,
        second,
        "a worker past the reject thresholds must be skipped"
    );
    slots.free(&"next".to_string(), decay_now()).unwrap();

    // The scan, not the picker's preference, keeps `first` out: pinning to it fails.
    let (mut pinned, pinned_rx) = make_request("pinned-full", 16);
    pinned.pinned_worker = Some(first);
    queue.enqueue(pinned).await;
    assert!(matches!(
        pinned_rx.await.unwrap(),
        Err(KvSchedulerError::PinnedWorkerOverloaded { worker_id: 0 })
    ));

    let (held, held_rx) = hold("held-1", second);
    queue.enqueue(held).await;
    held_rx.await.unwrap().unwrap();
    slots
        .mark_prefill_completed(&"held-1".to_string(), decay_now())
        .unwrap();

    let (rejected, rejected_rx) = make_request("rejected", 16);
    queue.enqueue(rejected).await;
    assert!(matches!(
        rejected_rx.await.unwrap(),
        Err(KvSchedulerError::AllEligibleWorkersOverloaded)
    ));
}

#[tokio::test]
async fn reject_thresholds_skip_a_worker_only_when_every_rank_is_full() {
    let profile = PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs);
    let (queue, slots, cfg_tx) = make_queue_with_profile_sender_and_reject(
        1,
        16,
        100,
        profile,
        Some(Arc::new(|| crate::scheduling_compat::BusyThresholds {
            prefill_busy_threshold_frac: Some(0.5),
            ..Default::default()
        })),
    );
    // Worker 0 grows to two ranks with a 100-token prefill budget each.
    let worker_config = SimpleWorkerConfig {
        max_num_batched_tokens: Some(100),
        data_parallel_size: 2,
        ..Default::default()
    };
    cfg_tx.send(HashMap::from([(0, worker_config)])).unwrap();
    slots
        .reconcile_workers([dynamo_slot_tracker::topology::WorkerDpRange::new(0, 0, 2)])
        .unwrap();
    set_worker_capacities_from_configs(&slots, &cfg_tx.borrow());
    let rank0 = WorkerWithDpRank::new(0, 0);
    let rank1 = WorkerWithDpRank::new(0, 1);

    let (mut held, held_rx) = make_request("held-rank0", 96);
    held.pinned_worker = Some(rank0);
    queue.enqueue(held).await;
    held_rx.await.unwrap().unwrap();

    // One full rank: the worker stays eligible and the request lands on the other rank.
    let (next, next_rx) = make_request("next", 16);
    queue.enqueue(next).await;
    assert_eq!(next_rx.await.unwrap().unwrap().best_worker, rank1);
    slots.free(&"next".to_string(), decay_now()).unwrap();

    let (mut held, held_rx) = make_request("held-rank1", 96);
    held.pinned_worker = Some(rank1);
    queue.enqueue(held).await;
    held_rx.await.unwrap().unwrap();

    // Both ranks full: the worker is skipped and nothing remains.
    let (rejected, rejected_rx) = make_request("rejected", 16);
    queue.enqueue(rejected).await;
    assert!(matches!(
        rejected_rx.await.unwrap(),
        Err(KvSchedulerError::AllEligibleWorkersOverloaded)
    ));
}

#[tokio::test]
async fn reject_thresholds_follow_the_live_provider_on_prefill_tokens() {
    let live = Arc::new(Mutex::new(crate::scheduling_compat::BusyThresholds {
        prefill_busy_threshold_frac: Some(0.5),
        ..Default::default()
    }));
    let provider = Arc::clone(&live);
    let profile = PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs);
    let (queue, slots, _cfg_tx) = make_queue_with_profile_sender_and_reject(
        2,
        16,
        100,
        profile,
        Some(Arc::new(move || *provider.lock())),
    );
    let first = WorkerWithDpRank::new(0, 0);
    let second = WorkerWithDpRank::new(1, 0);

    // 96 booked prefill tokens on `first` is past half of its 100-token budget.
    let (mut held, held_rx) = make_request("held", 96);
    held.pinned_worker = Some(first);
    queue.enqueue(held).await;
    held_rx.await.unwrap().unwrap();

    let (mut next, next_rx) = make_request("next", 16);
    next.pinned_worker = Some(first);
    queue.enqueue(next).await;
    assert!(matches!(
        next_rx.await.unwrap(),
        Err(KvSchedulerError::PinnedWorkerOverloaded { worker_id: 0 })
    ));
    let (free, free_rx) = make_request("free", 16);
    queue.enqueue(free).await;
    assert_eq!(
        free_rx.await.unwrap().unwrap().best_worker,
        second,
        "an unpinned request goes to the worker with room"
    );
    slots.free(&"free".to_string(), decay_now()).unwrap();

    // Lifting the thresholds through the live provider takes effect at the next selection.
    *live.lock() = crate::scheduling_compat::BusyThresholds::default();
    let (mut again, again_rx) = make_request("again", 16);
    again.pinned_worker = Some(first);
    queue.enqueue(again).await;
    assert_eq!(again_rx.await.unwrap().unwrap().best_worker, first);
}

#[tokio::test]
async fn decode_limit_holds_new_work_until_held_kv_frees() {
    let profile = policy_profile(
        r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: decode-bound
    policy_family: standard
    cache_bucket: all
    quantum: 1
    decode_busy_threshold_frac: 0.5
"#,
    );
    let (queue, slots, cfg_tx) = make_queue_with_profile_and_sender(1, 16, 1_000_000, profile);
    cfg_tx
        .send(HashMap::from([(
            0,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(1_000_000),
                total_kv_blocks: Some(8),
                ..Default::default()
            },
        )]))
        .unwrap();

    // A decoding request holds 6 of the worker's 8 KV blocks; its prefill load is gone.
    let (mut held, held_rx) = make_request("held", 96);
    held.token_seq = Some((0..6).collect());
    queue.enqueue(held).await;
    held_rx.await.unwrap().unwrap();
    slots
        .mark_prefill_completed(&"held".to_string(), decay_now())
        .unwrap();

    let (next, mut next_rx) = make_request("next", 16);
    queue.enqueue(next).await;
    queue.update().await;
    assert!(
        next_rx.try_recv().is_err(),
        "held KV above the decode limit must keep new work queued"
    );

    slots.free(&"held".to_string(), decay_now()).unwrap();
    queue.update().await;
    next_rx
        .await
        .unwrap()
        .expect("freed KV must release the queued request");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_queueing_under_pressure() {
    let block_size = 16;
    let isl = 512;
    let num_workers = 2;
    let num_requests = 10;

    let (queue, slots) = make_queue(num_workers, block_size, isl, Some(0.0));

    let mut receivers = Vec::new();
    let mut req_ids = Vec::new();

    for i in 0..num_requests {
        let req_id = format!("pressure-{i}");
        let (req, rx) = make_request(&req_id, isl);
        queue.enqueue(req).await;
        receivers.push(rx);
        req_ids.push(req_id);
    }

    // Drain pending by cycling mark_prefill_completed + free + update
    // on already-scheduled requests until all receivers have a response.
    for _ in 0..num_requests {
        queue.update().await;
        for rid in &req_ids {
            let _ = slots.mark_prefill_completed(rid, decay_now());
            let _ = slots.free(rid, decay_now());
        }
    }
    queue.update().await;

    let mut ok_count = 0;
    for mut rx in receivers {
        if let Ok(result) = rx.try_recv() {
            result.expect("scheduling returned error");
            ok_count += 1;
        }
    }
    assert_eq!(ok_count, num_requests, "not all requests were scheduled");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pending_requests_receive_shutdown_on_queue_drop() {
    let block_size = 16;
    let isl = 512;
    let (queue, _slots) = make_queue(1, block_size, isl, Some(0.0));

    let (req1, rx1) = make_request("req-1", isl);
    queue.enqueue(req1).await;
    rx1.await
        .expect("first response sender dropped")
        .expect("first request should be scheduled");

    let (req2, rx2) = make_request("req-2", isl);
    queue.enqueue(req2).await;
    assert_eq!(queue.pending_count(), 1);

    drop(queue);

    let response = tokio::time::timeout(Duration::from_secs(1), rx2)
        .await
        .expect("shutdown response timed out")
        .expect("pending response sender dropped");
    assert!(matches!(
        response,
        Err(KvSchedulerError::SubscriberShutdown)
    ));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pending_count() {
    let block_size = 16;
    let isl = 512;
    let num_workers = 1;

    // threshold_frac=0.0 means any active tokens trigger queueing
    let (queue, slots) = make_queue(num_workers, block_size, isl, Some(0.0));
    assert_eq!(queue.pending_count(), 0);

    // First request goes through (worker is idle)
    let (req1, rx1) = make_request("req-1", isl);
    queue.enqueue(req1).await;
    let _resp1 = rx1.await.unwrap().unwrap();
    assert_eq!(queue.pending_count(), 0); // scheduled immediately

    // Second and third requests should be queued (worker is now prefill-busy)
    let (req2, _rx2) = make_request("req-2", isl);
    queue.enqueue(req2).await;
    assert_eq!(queue.pending_count(), 1);

    let (req3, _rx3) = make_request("req-3", isl);
    queue.enqueue(req3).await;
    assert_eq!(queue.pending_count(), 2);

    // Free the first request and update — should drain one from pending
    slots
        .mark_prefill_completed(&"req-1".to_string(), decay_now())
        .unwrap();
    slots.free(&"req-1".to_string(), decay_now()).unwrap();
    queue.update().await;

    // After update, one pending request should have been scheduled
    assert!(
        queue.pending_count() < 2,
        "pending_count should decrease after free+update, got {}",
        queue.pending_count()
    );

    // Free req-2 and update to drain remaining
    let _ = slots.mark_prefill_completed(&"req-2".to_string(), decay_now());
    let _ = slots.free(&"req-2".to_string(), decay_now());
    queue.update().await;
    let _ = slots.mark_prefill_completed(&"req-3".to_string(), decay_now());
    let _ = slots.free(&"req-3".to_string(), decay_now());
    queue.update().await;

    assert_eq!(queue.pending_count(), 0, "all requests should be drained");
}

#[tokio::test(flavor = "multi_thread")]
async fn policy_classes_apply_independent_thresholds_and_preserve_backlog_order() {
    let profile = policy_profile(
        r#"
default_policy_family: latency
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: latency
    policy_family: latency
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 0
  - name: bulk
    policy_family: bulk
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 1024
"#,
    );
    let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);

    let (mut active, active_rx) = make_request("active", 64);
    active.policy_class = Some("latency".to_string());
    queue.enqueue(active).await;
    active_rx.await.unwrap().unwrap();

    let (mut bulk, bulk_rx) = make_request("bulk", 64);
    bulk.policy_class = Some("bulk".to_string());
    queue.enqueue(bulk).await;
    bulk_rx.await.unwrap().unwrap();

    let (mut queued_first, mut queued_first_rx) = make_request("queued-first", 64);
    queued_first.policy_class = Some("latency".to_string());
    queue.enqueue(queued_first).await;
    assert_eq!(queue.pending_count(), 1);

    for request_id in ["active", "bulk"] {
        slots
            .mark_prefill_completed(&request_id.to_string(), decay_now())
            .unwrap();
        slots.free(&request_id.to_string(), decay_now()).unwrap();
    }

    let (mut queued_second, mut queued_second_rx) = make_request("queued-second", 64);
    queued_second.policy_class = Some("latency".to_string());
    queue.enqueue(queued_second).await;
    assert_eq!(
        queue.pending_count(),
        1,
        "enqueue should admit the existing head before the new arrival"
    );
    queued_first_rx
        .try_recv()
        .expect("first queued request should be admitted")
        .expect("first queued request failed");
    assert!(
        queued_second_rx.try_recv().is_err(),
        "second request should remain behind the admitted head"
    );

    slots
        .mark_prefill_completed(&"queued-first".to_string(), decay_now())
        .unwrap();
    slots
        .free(&"queued-first".to_string(), decay_now())
        .unwrap();
    queue.update().await;
    queued_second_rx.await.unwrap().unwrap();
}

// No-bypass ordering per DEP #13891: freed capacity goes to the queued
// head, never to the arriving request.
#[tokio::test]
async fn arrival_drains_other_class_backlog_instead_of_bypassing() {
    let profile = policy_profile(
        r#"
default_policy_family: latency
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: latency
    policy_family: latency
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 1024
  - name: bulk
    policy_family: bulk
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 0
"#,
    );
    let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);

    let (mut active, active_rx) = make_request("active", 64);
    active.policy_class = Some("latency".to_string());
    queue.enqueue(active).await;
    active_rx.await.unwrap().unwrap();

    // The bulk class sees the worker as busy, so its request parks.
    let (mut bulk_head, mut bulk_head_rx) = make_request("bulk-head", 64);
    bulk_head.policy_class = Some("bulk".to_string());
    queue.enqueue(bulk_head).await;
    assert_eq!(queue.pending_count(), 1);

    // Free the worker without emitting a capacity update; the parked head
    // stays queued until something drains.
    slots
        .mark_prefill_completed(&"active".to_string(), decay_now())
        .unwrap();
    slots.free(&"active".to_string(), decay_now()).unwrap();
    assert_eq!(queue.pending_count(), 1);

    // A latency arrival must not bypass the bulk backlog: its enqueue-turn
    // drain admits the parked bulk head alongside the arrival itself.
    let (mut arrival, arrival_rx) = make_request("latency-arrival", 64);
    arrival.policy_class = Some("latency".to_string());
    queue.enqueue(arrival).await;
    assert_eq!(queue.pending_count(), 0);
    bulk_head_rx
        .try_recv()
        .expect("parked bulk head should drain on the arrival's turn")
        .expect("bulk head failed");
    arrival_rx.await.unwrap().unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn policy_families_and_cache_buckets_select_physical_queues() {
    let profile = policy_profile(
        r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
  - min_tokens: 32
    bucket: uncached
policy_classes:
  - name: cached
    policy_family: standard
    cache_bucket: cached
    quantum: 1
    prefill_busy_threshold: 0
  - name: uncached
    policy_family: standard
    cache_bucket: uncached
    quantum: 1
    prefill_busy_threshold: 0
  - name: latency_cached
    policy_family: latency
    cache_bucket: cached
    quantum: 1
    prefill_busy_threshold: 0
  - name: latency_uncached
    policy_family: latency
    cache_bucket: uncached
    quantum: 1
    prefill_busy_threshold: 0
  - name: custom_priority
    quantum: 1
    prefill_busy_threshold: 0
"#,
    );
    let (queue, _slots) = make_queue_with_profile(1, 16, 64, profile);
    let worker = WorkerWithDpRank::new(0, 0);

    let (active, active_rx) = make_request("active", 64);
    queue.enqueue(active).await;
    active_rx.await.unwrap().unwrap();

    let (mut latency_cached, _latency_cached_rx) = make_request("latency-cached", 64);
    latency_cached.policy_class = Some("latency".to_string());
    latency_cached
        .overlap
        .effective_cached_tokens
        .insert(worker, 64);
    queue.enqueue(latency_cached).await;

    let (mut latency_uncached, _latency_uncached_rx) = make_request("latency-uncached", 64);
    latency_uncached.policy_class = Some("latency".to_string());
    queue.enqueue(latency_uncached).await;

    let (mut unknown_cached, _unknown_cached_rx) = make_request("unknown-cached", 64);
    unknown_cached.policy_class = Some("unknown".to_string());
    unknown_cached
        .overlap
        .effective_cached_tokens
        .insert(worker, 64);
    queue.enqueue(unknown_cached).await;

    let (mut ordinary_class_name, _ordinary_class_name_rx) =
        make_request("ordinary-class-name", 64);
    ordinary_class_name.policy_class = Some("latency_cached".to_string());
    queue.enqueue(ordinary_class_name).await;

    let (mut custom, _custom_rx) = make_request("custom", 64);
    custom.policy_class = Some("custom_priority".to_string());
    queue.enqueue(custom).await;

    assert_eq!(
        queue.class_queue_stats(0),
        Some(ClassQueueStats {
            received_total: 1,
            rejected_due_time_passed_total: 0,
            pending_count: 1,
            pending_isl_tokens: 64,
            pending_cached_tokens: 64,
        })
    );
    assert_eq!(
        queue.class_queue_stats(1),
        Some(ClassQueueStats {
            received_total: 2,
            rejected_due_time_passed_total: 0,
            pending_count: 1,
            pending_isl_tokens: 64,
            pending_cached_tokens: 0,
        })
    );
    assert_eq!(
        queue.class_queue_stats(2),
        Some(ClassQueueStats {
            received_total: 1,
            rejected_due_time_passed_total: 0,
            pending_count: 1,
            pending_isl_tokens: 64,
            pending_cached_tokens: 64,
        })
    );
    assert_eq!(
        queue.class_queue_stats(3),
        Some(ClassQueueStats {
            received_total: 1,
            rejected_due_time_passed_total: 0,
            pending_count: 1,
            pending_isl_tokens: 64,
            pending_cached_tokens: 0,
        })
    );
    assert_eq!(
        queue.class_queue_stats(4),
        Some(ClassQueueStats {
            received_total: 1,
            rejected_due_time_passed_total: 0,
            pending_count: 1,
            pending_isl_tokens: 64,
            pending_cached_tokens: 0,
        })
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn class_local_limit_rejection_is_typed_and_not_overload() {
    let profile = policy_profile(
        r#"
default_policy_family: capped
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: capped
    policy_family: capped
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 0
    request_queue_limit_per_worker: 1
"#,
    );
    let (queue, _slots) = make_queue_with_profile(1, 16, 64, profile);

    let (active, active_rx) = make_request("active", 64);
    queue.enqueue(active).await;
    active_rx.await.unwrap().unwrap();

    let (queued, _queued_rx) = make_request("queued", 64);
    queue.enqueue(queued).await;

    let (rejected, rejected_rx) = make_request("rejected", 64);
    queue.enqueue(rejected).await;
    let error = rejected_rx.await.unwrap().unwrap_err();
    let KvSchedulerError::QueueRejected(rejection) = &error else {
        panic!("expected queue rejection, got {error:?}");
    };
    assert_eq!(rejection.policy_class, "capped");
    assert_eq!(
        rejection.limit_kind,
        dynamo_router_protocol::protocols::QueueLimitKind::Requests
    );
    assert_eq!(rejection.current, 1);
    assert_eq!(rejection.limit, 1);
    assert!(!error.is_overload());

    assert_eq!(
        queue.class_queue_stats(0),
        Some(ClassQueueStats {
            received_total: 3,
            rejected_due_time_passed_total: 0,
            pending_count: 1,
            pending_isl_tokens: 64,
            pending_cached_tokens: 0,
        })
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn per_worker_limit_tracks_discovered_worker_count_without_evicting() {
    let profile = policy_profile(
        r#"
default_policy_family: capped
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: capped
    policy_family: capped
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 0
    request_queue_limit_per_worker: 1
"#,
    );
    let (queue, _slots, cfg_tx) = make_queue_with_profile_and_sender(1, 16, 64, profile);

    let (active, active_rx) = make_request("active", 64);
    queue.enqueue(active).await;
    active_rx.await.unwrap().unwrap();

    let (first, mut first_rx) = make_request("first", 64);
    queue.enqueue(first).await;

    cfg_tx.send_modify(|configs| {
        configs.insert(
            1,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(64),
                ..Default::default()
            },
        );
    });
    let (second, _second_rx) = make_request("second", 64);
    queue.enqueue(second).await;
    first_rx
        .try_recv()
        .expect("new worker capacity should admit the existing head")
        .expect("first queued request failed");
    assert_eq!(queue.pending_count(), 1);

    cfg_tx.send_modify(|configs| {
        configs.remove(&1);
    });
    let (rejected, rejected_rx) = make_request("rejected", 64);
    queue.enqueue(rejected).await;
    let error = rejected_rx.await.unwrap().unwrap_err();
    let KvSchedulerError::QueueRejected(rejection) = error else {
        panic!("expected queue rejection, got {error:?}");
    };
    assert_eq!(rejection.current, 1);
    assert_eq!(rejection.limit, 1);
    assert_eq!(queue.pending_count(), 1);
}

#[tokio::test(start_paused = true)]
async fn test_queue_update_uses_decayed_oldest_prefill_load() {
    let estimator: Arc<dyn PrefillLoadEstimator> = Arc::new(FixedPrefillLoadEstimator {
        duration: Duration::from_secs(10),
    });
    let (queue, _slots, _cfg_tx) = make_queue_with_sender(1, 16, 100, Some(0.5), Some(estimator));

    let (req1, rx1) = make_request("req-1", 100);
    queue.enqueue(req1).await;
    let _ = rx1.await.unwrap().unwrap();

    let (req2, mut rx2) = make_request("req-2", 100);
    queue.enqueue(req2).await;
    assert_eq!(queue.pending_count(), 1);

    tokio::time::advance(Duration::from_secs(6)).await;
    queue.update().await;

    let scheduled = rx2
        .try_recv()
        .expect("queued request should have been scheduled");
    let response = scheduled.expect("scheduling returned error");
    assert_eq!(response.best_worker.worker_id, 0);
    assert_eq!(queue.pending_count(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_overloaded_provider_filters_at_admission() {
    let overloaded_worker_provider: OverloadedWorkerProvider =
        Arc::new(|| Some(HashSet::from([0])));
    let (queue, _slots) =
        make_queue_with_providers(1, 16, 256, Some(overloaded_worker_provider), None);

    let (req, rx) = make_request("overloaded", 256);
    queue.enqueue(req).await;

    let resp = rx.await.expect("oneshot dropped");
    assert!(matches!(
        resp,
        Err(KvSchedulerError::AllEligibleWorkersOverloaded)
    ));
}

#[tokio::test(flavor = "multi_thread")]
async fn hard_availability_provider_filters_unpinned_and_pinned_selection() {
    let available_worker_provider: WorkerAvailabilityProvider =
        Arc::new(|_| Some(Arc::new(HashSet::from([1]))));
    let (queue, slots) =
        make_queue_with_providers(2, 16, 256, None, Some(available_worker_provider));

    let request_id = "available".to_string();
    let (request, response_rx) = make_request(&request_id, 256);
    queue.enqueue(request).await;
    let response = response_rx.await.unwrap().unwrap();
    assert_eq!(response.best_worker.worker_id, 1);
    slots
        .mark_prefill_completed(&request_id, decay_now())
        .unwrap();
    slots.free(&request_id, decay_now()).unwrap();

    let (mut pinned, pinned_rx) = make_request("unavailable-pin", 256);
    pinned.pinned_worker = Some(WorkerWithDpRank::from_worker_id(0));
    queue.enqueue(pinned).await;
    assert!(matches!(
        pinned_rx.await.unwrap(),
        Err(KvSchedulerError::NoEndpoints)
    ));
}

#[tokio::test]
async fn queue_snapshot_excludes_workers_unavailable_to_the_request() {
    let provider: WorkerAvailabilityProvider = Arc::new(|request| {
        assert_eq!(request.lora_name.as_deref(), Some("adapter"));
        Some(Arc::new(HashSet::from([1])))
    });
    let (queue, _slots) = make_queue_with_providers(2, 16, 256, None, Some(provider));
    let (mut request, _response) = make_request("adapter-request", 256);
    request.lora_name = Some("adapter".to_string());
    request.overlap.effective_cached_tokens = FxHashMap::from_iter([
        (WorkerWithDpRank::from_worker_id(0), 224),
        (WorkerWithDpRank::from_worker_id(1), 32),
    ]);
    assert_eq!(
        queue
            .default_queue_metadata(&request, Instant::now())
            .snapshot
            .cached_tokens,
        32
    );
    assert_eq!(
        queue
            .build_classify_request(&request, Instant::now(), WorkerInputs::NONE)
            .0
            .scheduling_cost_tokens(),
        224
    );

    request.pinned_worker = Some(WorkerWithDpRank::from_worker_id(0));
    assert_eq!(
        queue
            .default_queue_metadata(&request, Instant::now())
            .snapshot
            .cached_tokens,
        0
    );
    assert_eq!(
        queue
            .build_classify_request(&request, Instant::now(), WorkerInputs::NONE)
            .0
            .scheduling_cost_tokens(),
        256
    );
}

/// A queue starting with zero workers can route after slots and configs arrive.
#[tokio::test(flavor = "multi_thread")]
async fn test_worker_updates_after_empty_start() {
    let block_size = 16;
    let isl = 512;

    // Start with zero workers (mimics skip_initial_worker_wait=true)
    let (queue, slots, cfg_tx) = make_queue_with_sender(0, block_size, isl, None, None);

    // Routing with no workers must fail
    let (req_fail, rx_fail) = make_request("before-register", isl);
    queue.enqueue(req_fail).await;
    let resp = rx_fail.await.expect("oneshot dropped");
    assert!(
        matches!(
            resp,
            Err(dynamo_router_protocol::sched::KvSchedulerError::NoEndpoints)
        ),
        "expected NoEndpoints before worker updates, got {resp:?}"
    );

    // Add two workers to the slot tracker, then publish their configs.
    slots.upsert_worker(WorkerDpRange::new(100, 0, 1)).unwrap();
    slots.upsert_worker(WorkerDpRange::new(200, 0, 1)).unwrap();

    // Also update the config watch so the selector can see these workers
    let mut configs = HashMap::new();
    for &id in &[100_u64, 200_u64] {
        configs.insert(
            id,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(isl as u64),
                ..Default::default()
            },
        );
    }
    cfg_tx.send(configs).unwrap();

    // Routing after registration must succeed and pick one of the registered workers
    let (req_ok, rx_ok) = make_request("after-register", isl);
    queue.enqueue(req_ok).await;
    let resp = rx_ok
        .await
        .expect("oneshot dropped")
        .expect("scheduling failed");
    assert!(
        resp.best_worker.worker_id == 100 || resp.best_worker.worker_id == 200,
        "expected worker 100 or 200, got {}",
        resp.best_worker.worker_id
    );

    // Clean up
    slots
        .mark_prefill_completed(&"after-register".to_string(), decay_now())
        .unwrap();
    slots
        .free(&"after-register".to_string(), decay_now())
        .unwrap();
}

/// Upserting a new worker preserves existing workers.
#[tokio::test(flavor = "multi_thread")]
async fn test_worker_updates_are_additive() {
    let block_size = 16;
    let isl = 256;

    let (queue, slots, cfg_tx) = make_queue_with_sender(0, block_size, isl, None, None);

    // Register worker 10 in slots and config
    slots.upsert_worker(WorkerDpRange::new(10, 0, 1)).unwrap();

    let mut configs = HashMap::new();
    configs.insert(
        10_u64,
        SimpleWorkerConfig {
            max_num_batched_tokens: Some(isl as u64),
            ..Default::default()
        },
    );
    cfg_tx.send(configs.clone()).unwrap();

    // Register worker 20 (worker 10 must NOT be evicted)
    slots.upsert_worker(WorkerDpRange::new(20, 0, 1)).unwrap();

    configs.insert(
        20_u64,
        SimpleWorkerConfig {
            max_num_batched_tokens: Some(isl as u64),
            ..Default::default()
        },
    );
    cfg_tx.send(configs).unwrap();

    // Send enough requests to statistically prove both workers are available
    let mut seen = std::collections::HashSet::new();
    for i in 0..20 {
        let req_id = format!("add-{i}");
        let (req, rx) = make_request(&req_id, isl);
        queue.enqueue(req).await;
        let resp = rx
            .await
            .expect("oneshot dropped")
            .expect("scheduling failed");
        seen.insert(resp.best_worker.worker_id);
        slots.mark_prefill_completed(&req_id, decay_now()).unwrap();
        slots.free(&req_id, decay_now()).unwrap();
    }

    assert!(
        seen.contains(&10) && seen.contains(&20),
        "both workers should be reachable after additive registration, saw: {seen:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn allowed_worker_request_joins_backlog_and_dispatches_within_allow_list() {
    let block_size = 16;
    let isl = 256;
    let (queue, slots) = make_queue(2, block_size, isl, Some(0.0));

    let (active_a, active_a_rx) = make_request("active-a", isl);
    queue.enqueue(active_a).await;
    let active_a_worker = active_a_rx.await.unwrap().unwrap().best_worker.worker_id;

    let (active_b, active_b_rx) = make_request("active-b", isl);
    queue.enqueue(active_b).await;
    active_b_rx.await.unwrap().unwrap();

    let (backlog_head, mut backlog_head_rx) = make_request("backlog-head", isl);
    queue.enqueue(backlog_head).await;
    assert_eq!(queue.pending_count(), 1);

    slots
        .mark_prefill_completed(&"active-a".to_string(), decay_now())
        .unwrap();
    slots.free(&"active-a".to_string(), decay_now()).unwrap();

    let (mut allowed, mut allowed_rx) = make_request("allowed", isl);
    allowed.allowed_worker_ids = Some(HashSet::from([active_a_worker]));
    queue.enqueue(allowed).await;
    assert_eq!(
        queue.pending_count(),
        1,
        "enqueue should dispatch the existing head before the allow-list request"
    );
    assert!(allowed_rx.try_recv().is_err());

    let backlog_head_worker = backlog_head_rx
        .try_recv()
        .expect("existing backlog head should be admitted")
        .expect("backlog head failed")
        .best_worker
        .worker_id;
    assert!(allowed_rx.try_recv().is_err());

    slots
        .mark_prefill_completed(&"backlog-head".to_string(), decay_now())
        .unwrap();
    slots
        .free(&"backlog-head".to_string(), decay_now())
        .unwrap();
    queue.update().await;

    let allowed_worker = allowed_rx.await.unwrap().unwrap().best_worker.worker_id;
    assert_eq!(allowed_worker, active_a_worker);

    for request_id in ["active-b", "allowed"] {
        slots
            .mark_prefill_completed(&request_id.to_string(), decay_now())
            .unwrap();
        slots.free(&request_id.to_string(), decay_now()).unwrap();
    }
    assert_eq!(backlog_head_worker, active_a_worker);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pinned_worker_conflict_with_allowed_ids_fails_early() {
    let (queue, _slots) = make_queue(1, 16, 256, Some(0.0));
    let (mut req, rx) = make_request("conflict", 256);
    req.pinned_worker = Some(WorkerWithDpRank::new(0, 0));
    req.allowed_worker_ids = Some(HashSet::from([1]));

    queue.enqueue(req).await;

    let resp = rx.await.expect("oneshot dropped");
    assert!(matches!(
        resp,
        Err(KvSchedulerError::PinnedWorkerNotAllowed { worker_id: 0 })
    ));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_disallowed_worker_ids_fail_without_queueing() {
    let (queue, _slots) = make_queue(1, 16, 256, Some(0.0));
    let (mut req, rx) = make_request("disallowed", 256);
    req.allowed_worker_ids = Some(HashSet::from([999]));

    queue.enqueue(req).await;

    let resp = rx.await.expect("oneshot dropped");
    assert!(matches!(resp, Err(KvSchedulerError::NoEndpoints)));
    assert_eq!(queue.pending_count(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_incompatible_required_taints_fail_without_queueing() {
    let (queue, _slots, cfg_tx) = make_queue_with_sender(1, 16, 256, Some(0.0), None);
    let mut configs = HashMap::new();
    configs.insert(
        0_u64,
        SimpleWorkerConfig {
            max_num_batched_tokens: Some(256),
            taints: HashSet::from(["mdc-a".to_string()]),
            ..Default::default()
        },
    );
    cfg_tx.send(configs).unwrap();

    let (mut req, rx) = make_request("tainted", 256);
    req.routing_constraints = dynamo_router_protocol::protocols::RoutingConstraints {
        required_taints: HashSet::from(["mdc-b".to_string()]),
        preferred_taints: HashMap::new(),
    };

    queue.enqueue(req).await;

    let resp = rx.await.expect("oneshot dropped");
    assert!(matches!(resp, Err(KvSchedulerError::NoEndpoints)));
    assert_eq!(queue.pending_count(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_blocked_pinned_lane_does_not_block_other_worker() {
    let (queue, slots) = make_queue(2, 16, 256, Some(0.0));

    let (mut first, first_rx) = make_request("pinned-1", 256);
    first.pinned_worker = Some(WorkerWithDpRank::new(1, 0));
    queue.enqueue(first).await;
    let first_resp = first_rx.await.unwrap().unwrap();
    assert_eq!(first_resp.best_worker, WorkerWithDpRank::new(1, 0));

    let (mut second, mut second_rx) = make_request("pinned-2", 256);
    second.pinned_worker = Some(WorkerWithDpRank::new(1, 0));
    queue.enqueue(second).await;
    assert_eq!(queue.pending_count(), 1);
    assert!(
        second_rx.try_recv().is_err(),
        "request should remain queued"
    );

    let (mut other_worker, mut other_worker_rx) = make_request("pinned-0", 256);
    other_worker.pinned_worker = Some(WorkerWithDpRank::new(0, 0));
    queue.enqueue(other_worker).await;
    assert_eq!(queue.pending_count(), 1);
    let other_worker_resp = other_worker_rx
        .try_recv()
        .expect("other worker request should have been scheduled")
        .expect("scheduling returned error");
    assert_eq!(other_worker_resp.best_worker, WorkerWithDpRank::new(0, 0));
    assert!(
        second_rx.try_recv().is_err(),
        "pinned request should still be queued"
    );

    slots
        .mark_prefill_completed(&"pinned-1".to_string(), decay_now())
        .unwrap();
    slots.free(&"pinned-1".to_string(), decay_now()).unwrap();
    queue.capacity_changed(Some(WorkerWithDpRank::new(1, 0)));

    let second_resp = tokio::time::timeout(Duration::from_secs(1), second_rx)
        .await
        .expect("capacity notification should schedule the pinned request")
        .expect("response channel should remain open");
    let second_resp = second_resp.expect("scheduling returned error");
    assert_eq!(second_resp.best_worker, WorkerWithDpRank::new(1, 0));
    assert_eq!(queue.pending_count(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_queue_prefill_busy_check_ignores_untracked_prefill_tokens() {
    let (queue, slots) = make_queue(1, 16, 256, Some(0.0));

    let (mut req1, rx1) = make_request("req-1", 256);
    req1.track_prefill_tokens = false;
    queue.enqueue(req1).await;
    let _resp1 = rx1.await.unwrap().unwrap();
    assert_eq!(
        slots
            .active_tokens(decay_now())
            .get(&WorkerWithDpRank::new(0, 0))
            .copied(),
        Some(0)
    );

    let (req2, rx2) = make_request("req-2", 256);
    queue.enqueue(req2).await;
    let _resp2 = rx2.await.unwrap().unwrap();
    assert_eq!(queue.pending_count(), 0);

    let _ = slots.mark_prefill_completed(&"req-1".to_string(), decay_now());
    let _ = slots.free(&"req-1".to_string(), decay_now());
    let _ = slots.mark_prefill_completed(&"req-2".to_string(), decay_now());
    let _ = slots.free(&"req-2".to_string(), decay_now());
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn update_refresh_can_change_selected_worker_after_queue_wait() {
    let block_size = 16u32;
    let isl = 64usize;
    let refresher = Arc::new(CountingRefresher {
        calls: AtomicUsize::new(0),
        last_retain_kv_transfer_chain: AtomicBool::new(false),
        response: RefreshedOverlap {
            kv_transfer_candidates: Some(KvTransferCandidates {
                block_hashes: vec![
                    ExternalSequenceBlockHash(101),
                    ExternalSequenceBlockHash(102),
                ],
                owner_prefix_blocks: vec![(WorkerWithDpRank::new(1, 0).into(), 2)],
                routing_snapshot: None,
            }),
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: FxHashMap::from_iter([
                    (WorkerWithDpRank::new(0, 0), 1.0),
                    (WorkerWithDpRank::new(1, 0), 9.0),
                ]),
                effective_cached_tokens: FxHashMap::from_iter([
                    (WorkerWithDpRank::new(0, 0), 16),
                    (WorkerWithDpRank::new(1, 0), 144),
                ]),
            },
        },
    });
    let (queue, slots) =
        make_queue_with_refresher(2, block_size, isl, Some(0.0), refresher.clone());

    let (mut req1, rx1) = make_request("req-1", isl);
    req1.overlap
        .effective_overlap_blocks
        .insert(WorkerWithDpRank::new(0, 0), 3.0);
    req1.overlap
        .effective_cached_tokens
        .insert(WorkerWithDpRank::new(0, 0), 48);
    queue.enqueue(req1).await;
    let resp1 = rx1.await.expect("rx1 dropped").expect("req-1 failed");
    assert_eq!(resp1.best_worker, WorkerWithDpRank::new(0, 0));

    let (mut req2, rx2) = make_request("req-2", isl);
    req2.overlap
        .effective_overlap_blocks
        .insert(WorkerWithDpRank::new(1, 0), 3.0);
    req2.overlap
        .effective_cached_tokens
        .insert(WorkerWithDpRank::new(1, 0), 48);
    queue.enqueue(req2).await;
    let resp2 = rx2.await.expect("rx2 dropped").expect("req-2 failed");
    assert_eq!(resp2.best_worker, WorkerWithDpRank::new(1, 0));

    let (mut req3, rx3) = make_request("req-3", isl);
    req3.retain_kv_transfer_chain = true;
    req3.overlap
        .effective_overlap_blocks
        .insert(WorkerWithDpRank::new(0, 0), 8.0);
    req3.overlap
        .effective_overlap_blocks
        .insert(WorkerWithDpRank::new(1, 0), 2.0);
    req3.overlap
        .effective_cached_tokens
        .insert(WorkerWithDpRank::new(0, 0), 128);
    req3.overlap
        .effective_cached_tokens
        .insert(WorkerWithDpRank::new(1, 0), 32);
    queue
        .enqueue_with_block_hashes(req3, Some(vec![LocalBlockHash(42)]))
        .await;
    assert_eq!(queue.pending_count(), 1);
    assert_eq!(refresher.calls.load(Ordering::Relaxed), 0);

    tokio::time::advance(Duration::from_secs(11)).await;

    slots.free(&"req-1".to_string(), decay_now()).unwrap();
    slots.free(&"req-2".to_string(), decay_now()).unwrap();
    queue.update().await;

    let resp3 = rx3.await.expect("rx3 dropped").expect("req-3 failed");
    assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
    assert!(
        refresher
            .last_retain_kv_transfer_chain
            .load(Ordering::Relaxed)
    );
    assert_eq!(resp3.best_worker, WorkerWithDpRank::new(1, 0));
    assert_eq!(resp3.effective_overlap_blocks, 9.0);
    assert_eq!(resp3.cached_tokens, 144);
    assert_eq!(
        resp3
            .kv_transfer_candidates
            .as_ref()
            .map(|candidates| candidates.owner_prefix_blocks.as_slice()),
        Some(&[(WorkerWithDpRank::new(1, 0).into(), 2)][..])
    );
    assert_eq!(queue.pending_count(), 0);
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn update_refresh_drops_kv_transfer_candidates_when_retention_disabled() {
    let block_size = 16u32;
    let isl = 64usize;
    let worker = WorkerWithDpRank::new(0, 0);
    let refresher = Arc::new(CountingRefresher {
        calls: AtomicUsize::new(0),
        last_retain_kv_transfer_chain: AtomicBool::new(true),
        response: RefreshedOverlap {
            kv_transfer_candidates: Some(KvTransferCandidates {
                block_hashes: vec![ExternalSequenceBlockHash(101)],
                owner_prefix_blocks: vec![(worker.into(), 1)],
                routing_snapshot: None,
            }),
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: FxHashMap::from_iter([(worker, 5.0)]),
                effective_cached_tokens: FxHashMap::from_iter([(worker, 80)]),
            },
        },
    });
    let (queue, slots) =
        make_queue_with_refresher(1, block_size, isl, Some(0.0), refresher.clone());

    let (req1, rx1) = make_request("req-1", isl);
    queue.enqueue(req1).await;
    let _ = rx1.await.expect("rx1 dropped").expect("req-1 failed");

    let (req2, rx2) = make_request("req-2", isl);
    queue
        .enqueue_with_block_hashes(req2, Some(vec![LocalBlockHash(42)]))
        .await;
    assert_eq!(queue.pending_count(), 1);

    tokio::time::advance(Duration::from_secs(11)).await;
    slots.free(&"req-1".to_string(), decay_now()).unwrap();
    queue.update().await;

    let resp2 = rx2.await.expect("rx2 dropped").expect("req-2 failed");
    assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
    assert!(
        !refresher
            .last_retain_kv_transfer_chain
            .load(Ordering::Relaxed)
    );
    assert_eq!(resp2.best_worker, worker);
    assert_eq!(resp2.effective_overlap_blocks, 5.0);
    assert_eq!(resp2.cached_tokens, 80);
    assert!(resp2.kv_transfer_candidates.is_none());
    assert_eq!(queue.pending_count(), 0);
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn selected_request_dispatches_after_refresh_if_worker_becomes_busy() {
    let block_size = 16u32;
    let isl = 64usize;
    let worker = WorkerWithDpRank::new(0, 0);
    let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::from_overlap(
        OverlapSignals {
            tier_overlap_blocks: Default::default(),
            effective_overlap_blocks: FxHashMap::from_iter([(worker, 7.0)]),
            effective_cached_tokens: FxHashMap::from_iter([(worker, 56)]),
        },
    )));
    let (queue, slots) = make_queue_with_blocking_refresher(
        1,
        block_size,
        isl,
        Some(0.0),
        refresher.clone(),
        ADMISSION_CHANNEL_CAPACITY,
    );

    let (req1, rx1) = make_request("req-1", isl);
    queue.enqueue(req1).await;
    let _ = rx1.await.expect("rx1 dropped").expect("req-1 failed");

    let (mut req2, rx2) = make_request("req-2", isl);
    req2.overlap
        .effective_overlap_blocks
        .insert(WorkerWithDpRank::new(0, 0), 4.0);
    req2.overlap
        .effective_cached_tokens
        .insert(WorkerWithDpRank::new(0, 0), 64);
    queue
        .enqueue_with_block_hashes(req2, Some(vec![LocalBlockHash(42)]))
        .await;
    assert_eq!(queue.pending_count(), 1);
    assert_eq!(
        queue.class_queue_stats(0).unwrap().pending_cached_tokens,
        64
    );

    slots
        .mark_prefill_completed(&"req-1".to_string(), decay_now())
        .unwrap();
    slots.free(&"req-1".to_string(), decay_now()).unwrap();

    tokio::time::advance(Duration::from_secs(11)).await;

    let update = {
        let queue = Arc::clone(&queue);
        tokio::spawn(async move {
            queue.update().await;
        })
    };
    refresher.wait_for_calls(1).await;
    assert_eq!(
        queue.pending_count(),
        0,
        "DRR-selected request must be removed before refresh"
    );
    assert_eq!(
        queue.class_queue_stats(0).unwrap().pending_cached_tokens,
        0,
        "queue counters must reflect the irrevocable dequeue"
    );

    slots
        .add_request(
            SequenceRequest {
                request_id: "occupy-during-refresh".to_string(),
                token_sequence: None,
                track_prefill_tokens: true,
                expected_output_tokens: None,
                prefill_load_hint: Some(PrefillLoadHint {
                    initial_effective_prefill_tokens: isl,
                    expected_prefill_duration: None,
                }),
                worker,
                lora_name: None,
            },
            decay_now(),
        )
        .unwrap();

    refresher.release_one();
    update.await.unwrap();

    let resp2 = rx2.await.expect("rx2 dropped").expect("req-2 failed");
    assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
    assert_eq!(resp2.best_worker, worker);
    assert_eq!(resp2.effective_overlap_blocks, 7.0);
    assert_eq!(resp2.cached_tokens, 56);
    assert_eq!(queue.pending_count(), 0);

    for request_id in ["occupy-during-refresh", "req-2"] {
        slots
            .mark_prefill_completed(&request_id.to_string(), decay_now())
            .unwrap();
        slots.free(&request_id.to_string(), decay_now()).unwrap();
    }
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn cancelled_enqueue_wait_keeps_cleanup_behind_command() {
    let block_size = 16u32;
    let isl = 64usize;
    let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::default()));
    let (queue, slots) =
        make_queue_with_blocking_refresher(1, block_size, isl, Some(0.0), refresher.clone(), 1);

    let (active, active_rx) = make_request("active", isl);
    queue.enqueue(active).await;
    active_rx.await.unwrap().unwrap();

    let (queued, queued_rx) = make_request("queued", isl);
    queue
        .enqueue_with_block_hashes(queued, Some(vec![LocalBlockHash(42)]))
        .await;
    slots.free(&"active".to_owned(), decay_now()).unwrap();
    tokio::time::advance(Duration::from_secs(11)).await;

    let update = {
        let queue = Arc::clone(&queue);
        tokio::spawn(async move { queue.update().await })
    };
    refresher.wait_for_calls(1).await;

    let (cancelled, cancelled_rx) = make_request("cancelled", isl);
    let lease = queue
        .new_request_lifecycle_lease(Some("cancelled"))
        .unwrap();
    let enqueue = {
        let queue = Arc::clone(&queue);
        tokio::spawn(async move {
            queue
                .enqueue_with_block_hashes_and_lease(cancelled, None, Some(lease))
                .await
        })
    };
    tokio::task::yield_now().await;
    assert_eq!(queue.admission_tx.capacity(), 0);
    drop(cancelled_rx);
    enqueue.abort();
    assert!(enqueue.await.unwrap_err().is_cancelled());

    // Cancellation drops the acknowledgement receiver, but the lease remains
    // inside the accepted command until the actor establishes request ownership.
    refresher.release_one();
    update.await.unwrap();
    queue.update().await;
    assert_eq!(queue.pending_count(), 0);

    queued_rx.await.unwrap().unwrap();
    slots.free(&"queued".to_owned(), decay_now()).unwrap();
    slots.assert_completely_drained(decay_now());
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn continuation_drain_does_not_self_send_into_saturated_actor_channel() {
    let block_size = 16u32;
    let isl = 64usize;
    let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::default()));
    let (queue, slots) =
        make_queue_with_blocking_refresher(1, block_size, isl, Some(0.0), refresher.clone(), 1);

    let (active, active_rx) = make_request("active", isl);
    queue.enqueue(active).await;
    active_rx.await.unwrap().unwrap();

    let (queued, queued_rx) = make_request("queued", isl);
    queue
        .enqueue_with_block_hashes(queued, Some(vec![LocalBlockHash(42)]))
        .await;
    slots
        .mark_prefill_completed(&"active".to_string(), decay_now())
        .unwrap();
    slots.free(&"active".to_string(), decay_now()).unwrap();
    tokio::time::advance(Duration::from_secs(11)).await;

    let update = {
        let queue = Arc::clone(&queue);
        tokio::spawn(async move { queue.update().await })
    };
    refresher.wait_for_calls(1).await;

    let (following, following_rx) = make_request("following", isl);
    let enqueue = {
        let queue = Arc::clone(&queue);
        tokio::spawn(async move { queue.enqueue(following).await })
    };
    tokio::task::yield_now().await;
    assert_eq!(
        queue.admission_tx.capacity(),
        0,
        "test must saturate the actor command channel"
    );

    refresher.release_one();
    tokio::time::timeout(Duration::from_secs(1), update)
        .await
        .expect("update deadlocked with a full actor command channel")
        .unwrap();
    queued_rx.await.unwrap().unwrap();

    slots
        .mark_prefill_completed(&"queued".to_string(), decay_now())
        .unwrap();
    slots.free(&"queued".to_string(), decay_now()).unwrap();
    queue.update().await;
    following_rx.await.unwrap().unwrap();
    enqueue.await.unwrap();
}
