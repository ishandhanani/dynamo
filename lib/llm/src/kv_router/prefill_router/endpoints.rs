// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the prefill hop needs from the owner of the prefill endpoint's shared state.

use std::sync::Arc;

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_model_card::local_model::runtime_config::DisaggregatedEndpoint;
use dynamo_router_protocol::{worker::WorkerId, worker_type::WorkerType};
use dynamo_runtime::{
    component::{Client, Endpoint},
    pipeline::async_trait,
    protocols::EndpointId,
};
use dynamo_scheduler::host::{PrefillLoadEstimator, RoutingConstraints};
use dynamo_sync::{load::SchedulerLoadSender, runtime_configs::RuntimeConfigWatch};
use tokio_util::sync::CancellationToken;

use crate::kv_router::{KvRouter, SelectionPolicySource};

/// The per-endpoint state a prefill worker set shares with every other router of its endpoint,
/// as far as [`PrefillRouter`](super::PrefillRouter) needs it: the runtime configs its workers
/// publish, and the `KvRouter` built on them.
///
/// The model manager owns that state (the runtime-config watch, the KV event-source membership,
/// the shared cache client, the LoRA domain) and implements this trait. The prefill hop is below
/// the model manager and reaches it only through here.
#[async_trait]
pub trait PrefillEndpointProvider: Send + Sync {
    /// Start, or reuse, the watch of the endpoint's per-worker runtime configs. The per-worker
    /// lookups below answer only for endpoints that have one.
    async fn get_or_create_runtime_config_watcher(
        &self,
        endpoint: &Endpoint,
    ) -> anyhow::Result<RuntimeConfigWatch>;

    /// Whether this worker advertises the runtime capability. False for an unknown worker.
    fn worker_supports_runtime_capability(
        &self,
        endpoint_id: &EndpointId,
        worker_id: WorkerId,
        capability: &str,
    ) -> bool;

    /// The bootstrap endpoint this worker publishes for the prefill-to-decode handoff.
    fn get_disaggregated_endpoint(
        &self,
        endpoint_id: &EndpointId,
        worker_id: WorkerId,
    ) -> Option<DisaggregatedEndpoint>;

    /// The worker's `data_parallel_size`, which encodes the prefill DP rank into the bootstrap
    /// room.
    fn get_data_parallel_size(&self, endpoint_id: &EndpointId, worker_id: WorkerId) -> Option<u32>;

    /// The topology constraints decode must satisfy to receive this prefill worker's KV.
    fn get_kv_transfer_routing_constraints(
        &self,
        endpoint_id: &EndpointId,
        worker_id: WorkerId,
    ) -> anyhow::Result<Option<RoutingConstraints>>;

    /// Build the `KvRouter` of the worker set behind `client`.
    #[allow(clippy::too_many_arguments)]
    async fn kv_chooser_for_with_policy_and_client(
        &self,
        client: Client,
        kv_cache_block_size: u32,
        policy: SelectionPolicySource,
        kv_router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        worker_role: Option<WorkerType>,
        metric_worker_type: &'static str,
        model_name: Option<String>,
        is_eagle: bool,
        scheduler_load: SchedulerLoadSender,
        cancellation_token: CancellationToken,
    ) -> anyhow::Result<Arc<KvRouter>>;
}

/// A provider for tests that knows no worker: every lookup answers as for an endpoint without
/// runtime configs, and the `KvRouter` has no shared cache, no LoRA policy and no scheduler-load
/// publisher.
#[cfg(test)]
pub(crate) struct TestEndpointProvider;

#[cfg(test)]
#[async_trait]
impl PrefillEndpointProvider for TestEndpointProvider {
    async fn get_or_create_runtime_config_watcher(
        &self,
        endpoint: &Endpoint,
    ) -> anyhow::Result<RuntimeConfigWatch> {
        use dynamo_runtime::traits::DistributedRuntimeProvider;
        dynamo_sync::runtime_configs::runtime_config_watch(endpoint, endpoint.drt().primary_token())
            .await
    }

    fn worker_supports_runtime_capability(&self, _: &EndpointId, _: WorkerId, _: &str) -> bool {
        false
    }

    fn get_disaggregated_endpoint(
        &self,
        _: &EndpointId,
        _: WorkerId,
    ) -> Option<DisaggregatedEndpoint> {
        None
    }

    fn get_data_parallel_size(&self, _: &EndpointId, _: WorkerId) -> Option<u32> {
        None
    }

    fn get_kv_transfer_routing_constraints(
        &self,
        _: &EndpointId,
        _: WorkerId,
    ) -> anyhow::Result<Option<RoutingConstraints>> {
        Ok(None)
    }

    async fn kv_chooser_for_with_policy_and_client(
        &self,
        client: Client,
        kv_cache_block_size: u32,
        policy: SelectionPolicySource,
        kv_router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        worker_role: Option<WorkerType>,
        metric_worker_type: &'static str,
        model_name: Option<String>,
        is_eagle: bool,
        scheduler_load: SchedulerLoadSender,
        cancellation_token: CancellationToken,
    ) -> anyhow::Result<Arc<KvRouter>> {
        use dynamo_runtime::traits::DistributedRuntimeProvider;
        let endpoint = client.endpoint.clone();
        let runtime_configs = self.get_or_create_runtime_config_watcher(&endpoint).await?;
        let kv_source_membership = dynamo_sync::membership::KvSourceMembershipCoordinator::start(
            endpoint.id(),
            runtime_configs.clone(),
            endpoint.drt().discovery(),
        )
        .subscribe();
        let router = KvRouter::new_with_worker_role_and_scheduler_load(
            endpoint,
            client,
            runtime_configs,
            Some(kv_source_membership),
            kv_cache_block_size,
            policy,
            kv_router_config,
            prefill_load_estimator,
            worker_role,
            metric_worker_type,
            model_name,
            is_eagle,
            None,
            None,
            scheduler_load,
            None,
            cancellation_token,
        )
        .await?;
        Ok(Arc::new(router))
    }
}
