// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use dynamo_runtime::protocols::EndpointId;

use dynamo_frontend_protocols::common::timing::WORKER_TYPE_DECODE;
use dynamo_model_card::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_model_card::model_type::ModelType;
use dynamo_router_protocol::worker_type::WorkerType;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkerGroupState {
    Ready,
    Pending,
    MaterializationFailed,
    CommitBlocked,
}

#[derive(Clone, Debug)]
pub struct WorkerGroupObservation {
    pub model: String,
    pub endpoint: EndpointId,
    pub worker_type: &'static str,
    pub model_type: ModelType,
    pub workers: HashMap<u64, ModelRuntimeConfig>,
    pub committed: HashSet<u64>,
    pub checksum_mismatches: HashSet<u64>,
    pub state: WorkerGroupState,
}

impl WorkerGroupObservation {
    pub fn metric_worker_type(&self) -> &'static str {
        // Encode workers can serve the public token-generation path, whose load and timing
        // attribution is decode even though its topology role remains encode.
        if self.worker_type == WorkerType::Encode.as_str()
            && self
                .model_type
                .intersects(ModelType::Chat | ModelType::Completions)
        {
            WORKER_TYPE_DECODE
        } else {
            self.worker_type
        }
    }
}

/// Tracks discovered groups independently of successful catalog admission.
#[derive(Default)]
pub struct WorkerInventory {
    groups: parking_lot::Mutex<HashMap<String, WorkerGroupObservation>>,
}

impl WorkerInventory {
    pub fn publish(&self, group_id: String, observation: Option<WorkerGroupObservation>) {
        let mut groups = self.groups.lock();
        if let Some(observation) = observation {
            groups.insert(group_id, observation);
        } else {
            groups.remove(&group_id);
        }
    }

    pub fn snapshot(&self) -> Vec<(String, WorkerGroupObservation)> {
        self.groups
            .lock()
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }
}
