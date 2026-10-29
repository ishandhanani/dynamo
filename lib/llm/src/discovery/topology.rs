// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The worker topology one endpoint sees: the join of instance availability
//! and the base model cards, published as one snapshot every router-side
//! reader shares.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use dynamo_router_types::protocols::WorkerId;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::discovery::DiscoveryQuery;
use dynamo_runtime::prelude::DistributedRuntimeProvider;
use dynamo_scheduler::topology::{Topology, WorkerRow};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use super::runtime_configs::{RuntimeConfigWatch, base_runtime_config_watch};
use crate::local_model::runtime_config::ModelRuntimeConfig;

/// Every live worker of an endpoint with the runtime configuration it advertised.
pub type EndpointTopology = Topology<Arc<ModelRuntimeConfig>>;

/// The latest [`EndpointTopology`]; readers wake on every membership change.
pub type TopologyWatch = watch::Receiver<Arc<EndpointTopology>>;

/// Join instance availability and config discovery into one topology watch.
///
/// A worker is a row only while it has BOTH an instance registration AND a
/// base runtime config. The join task recomputes on either source and
/// publishes only when the rows changed, so readers never wake on a no-op.
///
/// `lifecycle` bounds the base-card discovery task and the join task. The
/// `Client` behind the availability source scopes its own task to its last
/// receiver, so dropping this watch stops it. A caller scoped to something
/// narrower than the process must pass that scope's token, or the discovery
/// task outlives every dropped receiver and leaks until shutdown.
pub async fn topology_watch(
    endpoint: &Endpoint,
    lifecycle: CancellationToken,
) -> anyhow::Result<TopologyWatch> {
    let component = endpoint.component();
    let cancel_token = component.drt().primary_token();

    // Source 1: instance availability (watches DiscoveryQuery::Endpoint)
    let client = endpoint.client().await?;
    let mut instance_ids_rx = client.instance_avail_watcher();

    // Source 2: runtime configs from discovery (watches DiscoveryQuery::EndpointModels)
    let discovery = component.drt().discovery();
    let eid = endpoint.id();
    let stream = discovery
        .list_and_watch(
            DiscoveryQuery::EndpointModels {
                namespace: eid.namespace.clone(),
                component: eid.component.clone(),
                endpoint: eid.name.clone(),
            },
            Some(cancel_token.clone()),
        )
        .await?;
    let mut configs_rx = base_runtime_config_watch(stream, lifecycle.clone());

    let (tx, rx) = watch::channel(Arc::new(EndpointTopology::default()));

    tokio::spawn(async move {
        let mut generation = 0u64;
        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => break,
                _ = lifecycle.cancelled() => break,
                _ = tx.closed() => break,
                result = instance_ids_rx.changed() => { if result.is_err() { break; } }
                result = configs_rx.changed() => { if result.is_err() { break; } }
            }

            let instances: HashSet<WorkerId> = instance_ids_rx
                .borrow_and_update()
                .iter()
                .copied()
                .collect();
            let rows = {
                let configs = configs_rx.borrow_and_update();
                instances
                    .iter()
                    .filter_map(|worker_id| {
                        configs.get(worker_id).map(|config| WorkerRow {
                            worker_id: *worker_id,
                            config: Arc::new(config.clone()),
                        })
                    })
                    .collect()
            };
            let next = EndpointTopology::new(generation + 1, rows);

            // Publish only a real change, so downstream `changed()` waits stay quiet
            // on no-op recomputations.
            if tx.borrow().same_rows(&next) {
                continue;
            }
            generation += 1;

            // Stop when every receiver is gone (e.g. a TOCTOU loser in the model manager).
            if tx.send(Arc::new(next)).is_err() {
                break;
            }
        }
    });

    Ok(rx)
}

/// A snapshot built from a per-worker configuration map, for callers (and
/// tests) that hold membership as a map.
pub fn topology_from_configs(
    generation: u64,
    configs: &HashMap<WorkerId, ModelRuntimeConfig>,
) -> EndpointTopology {
    EndpointTopology::new(
        generation,
        configs
            .iter()
            .map(|(worker_id, config)| WorkerRow {
                worker_id: *worker_id,
                config: Arc::new(config.clone()),
            })
            .collect(),
    )
}

/// Test-side publisher of topology snapshots from configuration maps.
#[cfg(test)]
pub(crate) struct TopologySender {
    sender: watch::Sender<Arc<EndpointTopology>>,
    generation: std::sync::atomic::AtomicU64,
}

#[cfg(test)]
impl TopologySender {
    /// Publish `configs` as the next snapshot.
    pub(crate) fn send(
        &self,
        configs: HashMap<WorkerId, ModelRuntimeConfig>,
    ) -> Result<(), watch::error::SendError<Arc<EndpointTopology>>> {
        let generation = self
            .generation
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        self.sender
            .send(Arc::new(topology_from_configs(generation, &configs)))
    }
}

/// A topology watch over an initial configuration map, for tests.
#[cfg(test)]
pub(crate) fn topology_channel(
    configs: HashMap<WorkerId, ModelRuntimeConfig>,
) -> (TopologySender, TopologyWatch) {
    let (sender, receiver) = watch::channel(Arc::new(topology_from_configs(0, &configs)));
    (
        TopologySender {
            sender,
            generation: std::sync::atomic::AtomicU64::new(0),
        },
        receiver,
    )
}

/// The per-worker configuration map, derived from a topology for the readers
/// that still consume it as a map. One projection task per call; the model
/// manager caches one per endpoint.
pub fn runtime_config_watch_from(
    mut topology: TopologyWatch,
    lifecycle: CancellationToken,
) -> RuntimeConfigWatch {
    fn project(topology: &EndpointTopology) -> HashMap<WorkerId, ModelRuntimeConfig> {
        topology
            .iter()
            .map(|(worker_id, config)| (worker_id, ModelRuntimeConfig::clone(config)))
            .collect()
    }
    let (tx, rx) = watch::channel(project(&topology.borrow()));
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = lifecycle.cancelled() => break,
                _ = tx.closed() => break,
                result = topology.changed() => { if result.is_err() { break; } }
            }
            let next = project(&topology.borrow_and_update());
            if tx.send(next).is_err() {
                break;
            }
        }
    });
    rx
}
