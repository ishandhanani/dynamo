// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The router's stage configuration, assembled by the host: one slice per stage, each owned by
//! its crate. The kernel reads the slices and derives the few values that span stages.

use std::time::Duration;

use dynamo_admission::config::{AdmissionConfig, ClassifierConfig};
use dynamo_kv_indexer::config::IndexerConfig;
use dynamo_router_protocol::config::{RouterConfigOverride, TrackingHashConfig};
use dynamo_slot_tracker::config::SlotsConfig;
use dynamo_worker_selection::config::SelectionConfig;
use serde::Serialize;

/// The stage slices the kernel runs with.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct RouterConfig {
    pub hashing: TrackingHashConfig,
    pub indexer: IndexerConfig,
    pub slots: SlotsConfig,
    pub admission: AdmissionConfig,
    pub classifier: ClassifierConfig,
    pub selection: SelectionConfig,
}

impl RouterConfig {
    /// Whether this request assumes KV reuse: its override, else the selection slice.
    pub fn assume_kv_reuse(&self, config_override: Option<&RouterConfigOverride>) -> bool {
        config_override
            .and_then(|cfg| cfg.assume_kv_reuse)
            .unwrap_or(self.selection.assume_kv_reuse)
    }

    /// How often a queued request rechecks worker capacity: fast when prefill load is
    /// modeled and anything queues, otherwise the conservative default.
    pub fn queue_recheck_interval(&self) -> Duration {
        const DEFAULT_RECHECK_INTERVAL: Duration = Duration::from_secs(60);
        const PREFILL_LOAD_RECHECK_INTERVAL: Duration = Duration::from_millis(100);
        if self.slots.prefill_load_model.is_enabled()
            && (self.admission.has_routing_profiles() || self.admission.queue_threshold.is_some())
        {
            return PREFILL_LOAD_RECHECK_INTERVAL;
        }
        DEFAULT_RECHECK_INTERVAL
    }

    /// The effective settings as one JSON document, for operators.
    pub fn effective(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("router config serializes")
    }
}
