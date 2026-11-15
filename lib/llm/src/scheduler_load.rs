// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router scheduler load snapshots published on the event plane.
//!
//! The frontend-side counterpart is [`crate::frontend_load`]; the Pool Relay aggregates both.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use dynamo_kv_router::services::selection::SelectionService;
use dynamo_router_protocol::worker::WorkerWithDpRank;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventPublisher;
use dynamo_scheduler::host::ModelLoadResponse;
use dynamo_slot_tracker::host::{SchedulerGroup, SchedulerLoad};
use dynamo_sync::load::{
    SCHEDULER_LOAD_PUBLISH_INTERVAL, SCHEDULER_LOAD_SUBJECT, serving_load_publishing_enabled,
};
use tokio_util::sync::CancellationToken;

use dynamo_runtime::utils::retry::{Backoff, FailureStreak};

const SCHEDULER_LOAD_RETRY_INITIAL: Duration = Duration::from_millis(100);
const SCHEDULER_LOAD_RETRY_MAX: Duration = Duration::from_secs(5);

/// Publishes scheduler snapshots on the secondary runtime when serving-load reporting is enabled.
pub(crate) fn start_scheduler_load_publisher(
    endpoint: &Endpoint,
    service: Arc<SelectionService>,
    replica_sync: bool,
    cancellation_token: CancellationToken,
) {
    if !serving_load_publishing_enabled() {
        return;
    }
    let group = if replica_sync {
        SchedulerGroup::ReplicaGroup {
            group_id: endpoint.id().to_string(),
        }
    } else {
        // One process may host independent routers for the same endpoint; DRT identity
        // would merge their disjoint request sets.
        SchedulerGroup::Standalone {
            scheduler_id: rand::random(),
        }
    };
    endpoint
        .drt()
        .runtime()
        .secondary()
        .spawn(run_scheduler_load_publisher(
            endpoint.clone(),
            service,
            group,
            cancellation_token,
        ));
}

fn scheduler_load_batch(
    models: Vec<ModelLoadResponse>,
    group: &SchedulerGroup,
) -> Vec<SchedulerLoad> {
    let mut ranks = HashMap::<WorkerWithDpRank, SchedulerLoad>::new();
    for load in models.into_iter().flat_map(|model| model.loads) {
        let rank = WorkerWithDpRank::new(load.worker_id, load.dp_rank);
        ranks
            .entry(rank)
            .and_modify(|current| {
                // Each model/routing-group partition tracks disjoint requests.
                current.active_decode_blocks = current
                    .active_decode_blocks
                    .saturating_add(load.potential_decode_blocks as u64);
                current.active_prefill_tokens = current
                    .active_prefill_tokens
                    .saturating_add(load.potential_prefill_tokens as u64);
            })
            .or_insert_with(|| SchedulerLoad {
                worker_id: load.worker_id,
                dp_rank: load.dp_rank,
                // loads() supplies no hypothetical request, so these are active values.
                active_decode_blocks: load.potential_decode_blocks as u64,
                active_prefill_tokens: load.potential_prefill_tokens as u64,
                group: group.clone(),
            });
    }
    ranks.into_values().collect()
}

async fn run_scheduler_load_publisher(
    endpoint: Endpoint,
    service: Arc<SelectionService>,
    group: SchedulerGroup,
    cancellation_token: CancellationToken,
) {
    // Publish on the first tick; retry pacing matches the frontend load publisher.
    let mut publish_tick = tokio::time::interval(SCHEDULER_LOAD_PUBLISH_INTERVAL);
    publish_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut retry = Backoff::new(SCHEDULER_LOAD_RETRY_INITIAL, SCHEDULER_LOAD_RETRY_MAX);
    let mut failures = FailureStreak::default();

    loop {
        let connected = tokio::select! {
            _ = cancellation_token.cancelled() => return,
            connected = EventPublisher::for_endpoint(&endpoint, SCHEDULER_LOAD_SUBJECT) => connected,
        };
        let error = match connected {
            Ok(publisher) => loop {
                tokio::select! {
                    _ = cancellation_token.cancelled() => return,
                    _ = publish_tick.tick() => {}
                }
                let loads = scheduler_load_batch(service.loads(None, None), &group);
                let published = tokio::select! {
                    _ = cancellation_token.cancelled() => return,
                    published = publisher.publish(&loads) => published,
                };
                match published {
                    Ok(()) => {
                        retry.reset();
                        if let Some(failures) = failures.recover() {
                            tracing::info!(failures, "Scheduler load publishing recovered");
                        }
                    }
                    Err(error) => break error,
                }
            },
            Err(error) => error,
        };
        let delay = retry.next_delay();
        if failures.fail() {
            tracing::warn!(%error, retry_ms = delay.as_millis(), "Scheduler load publishing failed; reconnecting");
        } else {
            tracing::debug!(%error, failures = failures.failures(), retry_ms = delay.as_millis(), "Scheduler load publishing still failing");
        }
        tokio::select! {
            _ = cancellation_token.cancelled() => return,
            _ = tokio::time::sleep(delay) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use dynamo_runtime::transports::event_plane::EventSubscriber;

    #[test]
    fn scheduler_batch_sums_disjoint_partition_loads_per_rank() {
        let model = |name: &str, worker_id, blocks, tokens| ModelLoadResponse {
            model_name: name.into(),
            routing_group: "default".into(),
            loads: vec![dynamo_scheduler::host::PotentialLoad {
                worker_id,
                dp_rank: 0,
                potential_decode_blocks: blocks,
                potential_prefill_tokens: tokens,
                active_requests: 0,
            }],
            pending_count: 0,
            pending_isl_tokens: 0,
        };
        let group = SchedulerGroup::Standalone { scheduler_id: 9 };
        let mut batch = scheduler_load_batch(
            vec![
                model("a", 1, 3, 7),
                model("b", 1, 5, 4),
                model("b", 2, 2, 1),
            ],
            &group,
        );
        batch.sort_by_key(|load| load.worker_id);
        assert_eq!(batch.len(), 2);
        assert_eq!(
            (
                batch[0].active_decode_blocks,
                batch[0].active_prefill_tokens
            ),
            (8, 11)
        );
        assert_eq!((batch[1].worker_id, batch[1].active_decode_blocks), (2, 2));
        assert!(batch.iter().all(|load| load.group == group));
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn scheduler_publisher_batches_live_loads_and_removes_departed_ranks() -> Result<()> {
        use dynamo_kv_router::services::selection::SelectionServiceBuilder;
        use dynamo_scheduler::host::WorkerRequest;

        let distributed = dynamo_runtime::DistributedRuntime::new(
            dynamo_runtime::Runtime::from_current()?,
            dynamo_runtime::distributed::DistributedConfig::process_local(),
        )
        .await?;
        let endpoint = distributed
            .namespace(format!("scheduler-load-{}", uuid::Uuid::new_v4()))?
            .component("workers")?
            .endpoint("generate");
        let service = Arc::new(
            SelectionServiceBuilder::new(
                dynamo_kv_router::config::KvRouterConfig {
                    use_kv_events: false,
                    router_queue_threshold: None,
                    ..Default::default()
                },
                dynamo_router_protocol::worker_type::WorkerType::Aggregated,
                dynamo_router_policies::default_registry(),
            )
            .indexer_threads(1)
            .build()
            .await?,
        );
        for worker_id in [42, 43] {
            service
                .upsert_worker(WorkerRequest {
                    worker_id,
                    endpoint: Some("http://127.0.0.1:8000".into()),
                    block_size: Some(4),
                    data_parallel_start_rank: Some(0),
                    data_parallel_size: Some(2),
                    max_num_batched_tokens: Some(1024),
                    total_kv_blocks: Some(100),
                    is_eagle: Some(false),
                    ..Default::default()
                })
                .await?;
        }
        // Explicit reservations need a hint to account for prefill tokens.
        service
            .create_reservation(serde_json::from_value(serde_json::json!({
                "selection_id": "busy", "worker_id": 42, "dp_rank": 0,
                "token_ids": [1, 2, 3, 4], "track_prefill_tokens": true,
                "effective_prefill_tokens": 4
            }))?)
            .await?;
        let mut subscriber = EventSubscriber::for_endpoint(&endpoint, SCHEDULER_LOAD_SUBJECT)
            .await?
            .typed::<Vec<SchedulerLoad>>();
        let cancel = CancellationToken::new();
        let task = tokio::spawn(run_scheduler_load_publisher(
            endpoint,
            Arc::clone(&service),
            SchedulerGroup::Standalone { scheduler_id: 9 },
            cancel.clone(),
        ));
        let (_, first) = tokio::time::timeout(Duration::from_secs(5), subscriber.next())
            .await?
            .expect("scheduler publisher stopped")?;
        assert_eq!(first.len(), 4);
        let busy = first
            .iter()
            .find(|load| load.worker_id == 42 && load.dp_rank == 0)
            .unwrap();
        assert_eq!(busy.active_decode_blocks, 1);
        assert_eq!(busy.active_prefill_tokens, 4);
        service.free_reservation("busy").await?;
        service.delete_worker(42).await?;
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let (_, loads) = subscriber
                    .next()
                    .await
                    .expect("scheduler publisher stopped")?;
                if loads.iter().all(|load| load.worker_id == 43) {
                    assert_eq!(loads.len(), 2);
                    assert!(
                        loads.iter().all(|load| load.active_decode_blocks == 0
                            && load.active_prefill_tokens == 0)
                    );
                    return Ok::<_, anyhow::Error>(());
                }
            }
        })
        .await??;
        cancel.cancel();
        task.await?;
        distributed.shutdown();
        Ok(())
    }
}
