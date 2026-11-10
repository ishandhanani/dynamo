// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cold Pool through the router's queue, with Warm and Cold in one policy class: the classifier
//! puts Cold in its own lane, the dispatch gate narrows each head's workers with `GoAmong`, and the
//! host's capacity check and selection honor the narrowed set.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use dynamo_kv_router::plugins::request_classifier::RequestClassifierContext;
use dynamo_kv_router::plugins::worker_selection::{WorkerCapacity, WorkerInfo};
use dynamo_kv_router::protocols::{RoutingConstraints, WorkerId, WorkerWithDpRank};
use dynamo_kv_router::scheduling::config::RouterQueuePolicy;
use dynamo_kv_router::scheduling::overlap_refresh::NoopOverlapScoresRefresh;
use dynamo_kv_router::scheduling::policy_config::RouterPolicyConfig;
use dynamo_kv_router::scheduling::selector::DefaultWorkerSelector;
use dynamo_kv_router::scheduling::{
    AdmittedSchedulingResponse, KvSchedulerError, LocalScheduler, OverlapSignals, RequestLifecycle,
    ScheduleMode, ScheduleRequest,
};
use dynamo_kv_router::test_utils::SimpleWorkerConfig;
use dynamo_kv_router::{ActiveSequencesMultiWorker, KvRouterConfig, NoopSequencePublisher};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

type Scheduler = LocalScheduler<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>;
type Admission =
    Pin<Box<dyn Future<Output = Result<AdmittedSchedulingResponse, KvSchedulerError>>>>;

const BLOCK_SIZE: u32 = 64;
const WORKERS: WorkerId = 6;
const COLD_TOKENS: usize = 2_000;
const WARM_TOKENS: usize = 100;

fn rank(worker_id: WorkerId) -> WorkerWithDpRank {
    WorkerWithDpRank::new(worker_id, 0)
}

/// The plugin's membership rule (#12806): the top `count` workers by SplitMix64 HRW score.
fn expected_members(count: usize) -> HashSet<WorkerId> {
    fn score(worker_id: WorkerId) -> u64 {
        let mut value = worker_id.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
    let mut ids: Vec<_> = (0..WORKERS).collect();
    ids.sort_by(|left, right| score(*right).cmp(&score(*left)).then(left.cmp(right)));
    ids.into_iter().take(count).collect()
}

/// One policy class: Warm and Cold share it, and Cold waits in its own lane.
const POLICY: &str = r#"
default_policy_class: default
policy_classes:
  - name: default
    quantum: 4096
    queue_policy: fcfs
    prefill_busy_threshold_frac: 1.0
request_classifier:
  type: cold_pool
  parameters:
    request_threshold: 1000
    workers: 2
    soft_drain_timeout_ms: 300
"#;

fn scheduler() -> (Arc<Scheduler>, CancellationToken) {
    let config = SimpleWorkerConfig {
        max_num_batched_tokens: Some(10_000),
        total_kv_blocks: Some(10_000),
        ..Default::default()
    };
    let slots = Arc::new(ActiveSequencesMultiWorker::new(
        NoopSequencePublisher,
        BLOCK_SIZE as usize,
        (0..WORKERS).map(|id| (id, (0, 1))).collect(),
        false,
        0,
        "test",
    ));
    let (configs, config_rx) = watch::channel(
        (0..WORKERS)
            .map(|id| (id, config.clone()))
            .collect::<HashMap<_, _>>(),
    );
    std::mem::forget(configs);
    let profile = RouterPolicyConfig::from_yaml(POLICY)
        .unwrap()
        .resolve_profile(None, None, RouterQueuePolicy::Fcfs);
    let cancel = CancellationToken::new();
    let scheduler = Arc::new(LocalScheduler::new(
        slots,
        config_rx,
        profile,
        BLOCK_SIZE,
        DefaultWorkerSelector::new(None, "test"),
        None,
        None::<Arc<NoopOverlapScoresRefresh>>,
        None,
        None,
        None,
        Duration::from_secs(60),
        true,
        cancel.clone(),
        "test",
        false,
    ));

    let policy = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(policy.path(), POLICY).unwrap();
    let mut registry = dynamo_router_policies::default_registry();
    dynamo_router_policies::register(&mut registry).unwrap();
    let plugins = registry
        .resolve_plugins(&KvRouterConfig {
            router_policy_config: Some(policy.path().display().to_string()),
            ..Default::default()
        })
        .unwrap();
    let context = RequestClassifierContext::new(BLOCK_SIZE, move || {
        (0..WORKERS)
            .map(|id| WorkerInfo::new(rank(id), WorkerCapacity::new(Some(10_000), Some(10_000))))
            .collect()
    });
    let classifier = plugins.request_classifier().unwrap()(context);
    assert!(scheduler.install_request_classifier(classifier, cancel.clone()));
    (scheduler, cancel)
}

fn request(request_id: &str, isl_tokens: usize, pinned: Option<WorkerId>) -> ScheduleRequest {
    ScheduleRequest {
        mode: ScheduleMode::TrackedWithLifecycle {
            request_id: request_id.into(),
        },
        token_seq: None,
        block_hashes: None,
        isl_tokens,
        lora_name: None,
        expected_output_tokens: None,
        affinity_target: None,
        pinned_worker: pinned.map(rank),
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
        router_config_override: None,
        priority_jump: 0.0,
        strict_priority: 0,
        policy_class: None,
        session_context: None,
        overlap: OverlapSignals::default(),
        kv_transfer_candidates: None,
        retain_kv_transfer_chain: false,
        shared_cache_hits: None,
    }
}

fn poll_once(admission: &mut Admission) -> Poll<WorkerWithDpRank> {
    match admission
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        Poll::Ready(Ok(admitted)) => Poll::Ready(admitted.response.best_worker),
        Poll::Ready(Err(error)) => panic!("admission failed: {error}"),
        Poll::Pending => Poll::Pending,
    }
}

struct Harness {
    scheduler: Arc<Scheduler>,
    lifecycles: HashMap<String, RequestLifecycle>,
}

impl Harness {
    async fn submit(&mut self, request: ScheduleRequest) -> Admission {
        let request_id = request.mode.request_id().unwrap().to_owned();
        let lifecycle = self
            .scheduler
            .begin_request_lifecycle(&request_id)
            .unwrap()
            .unwrap();
        self.lifecycles.insert(request_id, lifecycle);
        let scheduler = Arc::clone(&self.scheduler);
        let admission: Admission =
            Box::pin(async move { scheduler.schedule_request_admitted(request).await });
        admission
    }

    /// Admits, then reports `Sent` so the plugin sees the request on its worker.
    async fn admitted(&mut self, request_id: &str, admission: &mut Admission) -> WorkerWithDpRank {
        for _ in 0..200 {
            if let Poll::Ready(worker) = poll_once(admission) {
                self.lifecycles.get_mut(request_id).unwrap().sent(worker);
                // Let the event task deliver `Sent` before the next decision.
                for _ in 0..8 {
                    tokio::task::yield_now().await;
                }
                self.scheduler.admission_barrier().await.unwrap();
                return worker;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("{request_id} was not admitted");
    }

    async fn still_held(&self, admission: &mut Admission) -> bool {
        self.scheduler.admission_barrier().await.unwrap();
        poll_once(admission).is_pending()
    }

    async fn finish(&mut self, request_id: &str) {
        self.lifecycles.get_mut(request_id).unwrap().complete();
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_waits_for_a_clean_member_while_warm_avoids_the_pool() {
    let members = expected_members(2);
    let mut member_ids: Vec<_> = members.iter().copied().collect();
    member_ids.sort();
    let (scheduler, cancel) = scheduler();
    let mut harness = Harness {
        scheduler,
        lifecycles: HashMap::new(),
    };

    // An idle pool: Cold goes straight to a member.
    let mut c0 = harness.submit(request("c0", COLD_TOKENS, None)).await;
    let worker = harness.admitted("c0", &mut c0).await;
    assert!(members.contains(&worker.worker_id), "c0 went to {worker:?}");
    harness.finish("c0").await;

    // Warm work pinned onto both members (a pin outranks the gate's narrowing).
    let mut busy = Vec::new();
    for (index, member) in member_ids.iter().enumerate() {
        let id = format!("pin{index}");
        let mut admission = harness
            .submit(request(&id, WARM_TOKENS, Some(*member)))
            .await;
        assert_eq!(harness.admitted(&id, &mut admission).await, rank(*member));
        busy.push(id);
    }

    // Cold now has no clean member and is held; Warm arriving meanwhile avoids the pool.
    let mut c1 = harness.submit(request("c1", COLD_TOKENS, None)).await;
    assert!(harness.still_held(&mut c1).await);
    for index in 0..8 {
        let id = format!("w{index}");
        let mut admission = harness.submit(request(&id, WARM_TOKENS, None)).await;
        let worker = harness.admitted(&id, &mut admission).await;
        assert!(
            !members.contains(&worker.worker_id),
            "{id} went to {worker:?}"
        );
    }
    assert!(harness.still_held(&mut c1).await);

    // The first member drains: Cold goes there, before the soft-drain timeout.
    harness.finish(&busy[0]).await;
    let worker = harness.admitted("c1", &mut c1).await;
    assert_eq!(worker, rank(member_ids[0]));

    // Cold is running on member 0 only: Warm avoids it but may use member 1 again.
    let mut seen = HashSet::new();
    for index in 0..24 {
        let id = format!("x{index}");
        let mut admission = harness.submit(request(&id, WARM_TOKENS, None)).await;
        let worker = harness.admitted(&id, &mut admission).await;
        assert_ne!(
            worker.worker_id, member_ids[0],
            "{id} landed on the running Cold worker"
        );
        seen.insert(worker.worker_id);
    }
    assert!(seen.len() > 1);
    cancel.cancel();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_takes_a_busy_member_after_the_soft_drain_timeout() {
    let members = expected_members(2);
    let (scheduler, cancel) = scheduler();
    let mut harness = Harness {
        scheduler,
        lifecycles: HashMap::new(),
    };
    for (index, member) in members.iter().enumerate() {
        let id = format!("pin{index}");
        let mut admission = harness
            .submit(request(&id, WARM_TOKENS, Some(*member)))
            .await;
        harness.admitted(&id, &mut admission).await;
    }
    let mut cold = harness.submit(request("c", COLD_TOKENS, None)).await;
    assert!(harness.still_held(&mut cold).await);
    let started = tokio::time::Instant::now();
    let worker = harness.admitted("c", &mut cold).await;
    assert!(members.contains(&worker.worker_id), "c went to {worker:?}");
    assert!(started.elapsed() >= Duration::from_millis(200));
    cancel.cancel();
}
