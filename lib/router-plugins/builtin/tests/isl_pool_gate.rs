// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ISL pool through the router's queue: two tainted pools in one policy class, the classifier
//! narrows each request to the pool its input length selects, and the host's capacity check and
//! selection honor the narrowed set. A pool taint no worker carries leaves routing unrestricted.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use dynamo_kv_router::plugins::request_classifier::RequestClassifierContext;
use dynamo_kv_router::plugins::worker_selection::{WorkerCapacity, WorkerInfo};
use dynamo_kv_router::protocols::{RoutingConstraints, WorkerId, WorkerWithDpRank};
use dynamo_kv_router::scheduling::config::RouterQueuePolicy;
use dynamo_kv_router::scheduling::overlap_refresh::NoopOverlapScoresRefresh;
use dynamo_kv_router::scheduling::policy_config::RouterPolicyConfig;
use dynamo_kv_router::scheduling::selector::DefaultWorkerSelector;
use dynamo_kv_router::scheduling::{LocalScheduler, OverlapSignals, ScheduleMode, ScheduleRequest};
use dynamo_kv_router::test_utils::SimpleWorkerConfig;
use dynamo_kv_router::{ActiveSequencesMultiWorker, KvRouterConfig, NoopSequencePublisher};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

type Scheduler = LocalScheduler<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>;

const BLOCK_SIZE: u32 = 64;
const WORKERS: WorkerId = 6;
const SMALL_TOKENS: usize = 100;
const LARGE_TOKENS: usize = 5_000;

fn rank(worker_id: WorkerId) -> WorkerWithDpRank {
    WorkerWithDpRank::new(worker_id, 0)
}

fn small_pool() -> HashSet<WorkerId> {
    (0..3).collect()
}

fn large_pool() -> HashSet<WorkerId> {
    (3..WORKERS).collect()
}

fn policy(large_taint: &str, small_taint: &str) -> String {
    format!(
        r#"
default_policy_class: default
policy_classes:
  - name: default
    quantum: 4096
    queue_policy: fcfs
    prefill_busy_threshold_frac: 1.0
request_classifier:
  type: isl_pool
  parameters:
    threshold_tokens: 1000
    large_taint: {large_taint}
    small_taint: {small_taint}
"#
    )
}

/// Six workers: 0-2 carry `pool/small`, 3-5 carry `pool/large`.
fn scheduler(policy: &str) -> (Arc<Scheduler>, CancellationToken) {
    let configs: HashMap<WorkerId, SimpleWorkerConfig> = (0..WORKERS)
        .map(|id| {
            let taint = if small_pool().contains(&id) {
                "pool/small"
            } else {
                "pool/large"
            };
            (
                id,
                SimpleWorkerConfig {
                    max_num_batched_tokens: Some(100_000),
                    total_kv_blocks: Some(100_000),
                    taints: HashSet::from([taint.to_string()]),
                    ..Default::default()
                },
            )
        })
        .collect();
    let slots = Arc::new(ActiveSequencesMultiWorker::new(
        NoopSequencePublisher,
        BLOCK_SIZE as usize,
        (0..WORKERS).map(|id| (id, (0, 1))).collect(),
        false,
        0,
        "test",
    ));
    let (configs_tx, config_rx) = watch::channel(configs);
    std::mem::forget(configs_tx);
    let profile = RouterPolicyConfig::from_yaml(policy)
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

    let policy_file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(policy_file.path(), policy).unwrap();
    let mut registry = dynamo_router_policies::default_registry();
    dynamo_router_policies::register(&mut registry).unwrap();
    let plugins = registry
        .resolve_plugins(&KvRouterConfig {
            router_policy_config: Some(policy_file.path().display().to_string()),
            ..Default::default()
        })
        .unwrap();
    let context = RequestClassifierContext::new(BLOCK_SIZE, move || {
        (0..WORKERS)
            .map(|id| WorkerInfo::new(rank(id), WorkerCapacity::new(Some(100_000), Some(100_000))))
            .collect()
    });
    let classifier = plugins.request_classifier().unwrap()(context);
    assert!(scheduler.install_request_classifier(classifier, cancel.clone()));
    (scheduler, cancel)
}

fn request(request_id: &str, isl_tokens: usize) -> ScheduleRequest {
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
        pinned_worker: None,
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

/// Admit `count` requests of `isl_tokens` and return the worker IDs they landed on, keeping each
/// admitted request active so selection must spread within the allowed pool.
async fn admit_all(
    scheduler: &Arc<Scheduler>,
    prefix: &str,
    isl_tokens: usize,
    count: usize,
) -> HashSet<WorkerId> {
    let mut landed = HashSet::new();
    let mut lifecycles = Vec::new();
    for index in 0..count {
        let request_id = format!("{prefix}{index}");
        let lifecycle = scheduler
            .begin_request_lifecycle(&request_id)
            .unwrap()
            .unwrap();
        let admitted = tokio::time::timeout(
            Duration::from_secs(5),
            scheduler.schedule_request_admitted(request(&request_id, isl_tokens)),
        )
        .await
        .unwrap_or_else(|_| panic!("{request_id} was not admitted"))
        .unwrap();
        let worker = admitted.response.best_worker;
        landed.insert(worker.worker_id);
        lifecycles.push((lifecycle, worker));
    }
    for (mut lifecycle, worker) in lifecycles {
        lifecycle.sent(worker);
        lifecycle.complete();
    }
    landed
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn input_length_selects_the_pool() {
    let (scheduler, cancel) = scheduler(&policy("pool/large", "pool/small"));

    let small = admit_all(&scheduler, "s", SMALL_TOKENS, 12).await;
    assert!(
        small.is_subset(&small_pool()),
        "small requests landed on {small:?}"
    );
    assert!(small.len() > 1, "small requests did not spread: {small:?}");

    let large = admit_all(&scheduler, "l", LARGE_TOKENS, 12).await;
    assert!(
        large.is_subset(&large_pool()),
        "large requests landed on {large:?}"
    );
    assert!(large.len() > 1, "large requests did not spread: {large:?}");
    cancel.cancel();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_taint_no_worker_carries_leaves_routing_unrestricted() {
    // The large pool is mistyped: large requests still dispatch, to any worker.
    let (scheduler, cancel) = scheduler(&policy("pool/larg", "pool/small"));

    let large = admit_all(&scheduler, "l", LARGE_TOKENS, 12).await;
    assert!(
        !large.is_subset(&large_pool()),
        "an unmatched pool must not narrow routing: {large:?}"
    );
    let small = admit_all(&scheduler, "s", SMALL_TOKENS, 12).await;
    assert!(
        small.is_subset(&small_pool()),
        "the matched pool still narrows: {small:?}"
    );
    cancel.cancel();
}
