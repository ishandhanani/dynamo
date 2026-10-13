// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The frontend's liveness source for session bindings: discovery knows the
//! worker and, when its runtime config is known, the bound rank is within the
//! worker's data-parallel range. Health is deliberately not consulted: an
//! unavailable worker is still known, and transient overload must not drop a
//! binding.

use dynamo_kv_router::protocols::WorkerConfigLike;
use dynamo_kv_router::services::selection::affinity::{
    AffinityTarget as TableTarget, TargetLiveness,
};
use dynamo_runtime::component::Client;

use crate::discovery::RuntimeConfigWatch;

pub(crate) struct DiscoveryLiveness {
    client: Client,
    /// Runtime configs for the worker set, when the host tracks them (KV
    /// routing). Builtin hosts know only discovery.
    workers: Option<RuntimeConfigWatch>,
}

impl DiscoveryLiveness {
    pub(crate) fn new(client: Client, workers: Option<RuntimeConfigWatch>) -> Self {
        Self { client, workers }
    }
}

impl TargetLiveness for DiscoveryLiveness {
    fn is_schedulable(&self, target: TableTarget) -> bool {
        if !self.client.is_instance_discovered(target.worker_id) {
            return false;
        }
        let Some(workers) = &self.workers else {
            return true;
        };
        let workers = workers.borrow();
        let Some(config) = workers.get(&target.worker_id) else {
            return true;
        };
        let Some(dp_rank) = target.dp_rank else {
            return true;
        };
        let start = config.data_parallel_start_rank();
        let end = start.saturating_add(config.data_parallel_size());
        (start..end).contains(&dp_rank)
    }
}
