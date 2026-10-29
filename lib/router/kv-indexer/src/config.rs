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
    /// Approximate (TTL) retention for the local side indexer, in seconds.
    pub ttl_secs: f64,
    /// Predicted-TTL side retention beside event-driven indexing, when set.
    pub predicted_ttl_secs: Option<f64>,
    pub hashing: TrackingHashConfig,
}

impl Default for IndexerConfig {
    /// The router's defaults; the aggregate's `Default` is the single source until it leaves
    /// the shared crate.
    fn default() -> Self {
        Self::from(&KvRouterConfig::default())
    }
}

impl From<&KvRouterConfig> for IndexerConfig {
    fn from(config: &KvRouterConfig) -> Self {
        Self {
            event_threads: config.router_event_threads,
            approximate_cache_policy: config.router_approximate_cache_policy,
            enable_session_prefix_index: config.enable_session_prefix_index,
            use_kv_events: config.use_kv_events,
            ttl_secs: config.router_ttl_secs,
            predicted_ttl_secs: config.router_predicted_ttl_secs,
            hashing: TrackingHashConfig::from(config),
        }
    }
}
