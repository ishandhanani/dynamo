// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The indexer slice of the router configuration.

use dynamo_router_types::config::{ApproximateCachePolicyKind, KvRouterConfig, TrackingHashConfig};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IndexerConfig {
    pub event_threads: u32,
    pub approximate_cache_policy: ApproximateCachePolicyKind,
    pub enable_session_prefix_index: bool,
    /// Whether the index is fed by worker KV events at all.
    pub use_kv_events: bool,
    pub hashing: TrackingHashConfig,
}

impl From<&KvRouterConfig> for IndexerConfig {
    fn from(config: &KvRouterConfig) -> Self {
        Self {
            event_threads: config.router_event_threads,
            approximate_cache_policy: config.router_approximate_cache_policy,
            enable_session_prefix_index: config.enable_session_prefix_index,
            use_kv_events: config.use_kv_events,
            hashing: TrackingHashConfig::from(config),
        }
    }
}
