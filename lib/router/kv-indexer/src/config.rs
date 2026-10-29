// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The indexer slice of the router configuration.

use std::env::{self, VarError};
use std::fmt;
use std::str::FromStr;

use dynamo_router_protocol::config::TrackingHashConfig;
use serde::{Deserialize, Serialize};

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
    /// The router's defaults; `dynamo-kv-router` asserts the flat defaults map onto these.
    fn default() -> Self {
        Self {
            event_threads: 4,
            approximate_cache_policy: ApproximateCachePolicyKind::default(),
            enable_session_prefix_index: false,
            use_kv_events: true,
            ttl_secs: 120.0,
            predicted_ttl_secs: None,
            hashing: TrackingHashConfig::default(),
        }
    }
}

/// Retention policy for a router-local primary approximate indexer.
///
/// This selector is intentionally process-local. Workers do not advertise it in
/// model cards because request lifetime and release ownership live in the router.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApproximateCachePolicyKind {
    /// Expire predicted entries after `router_ttl_secs`.
    #[default]
    Ttl,
    /// Retain predicted entries until per-rank KV capacity requires LRU eviction.
    Lru,
}

impl fmt::Display for ApproximateCachePolicyKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ttl => f.write_str("ttl"),
            Self::Lru => f.write_str("lru"),
        }
    }
}

impl FromStr for ApproximateCachePolicyKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "ttl" => Ok(Self::Ttl),
            "lru" => Ok(Self::Lru),
            _ => Err(format!(
                "unknown approximate cache policy {value:?}, expected 'ttl' or 'lru'"
            )),
        }
    }
}

pub const DYN_ROUTER_MIN_INITIAL_WORKERS: &str = "DYN_ROUTER_MIN_INITIAL_WORKERS";

pub fn min_initial_workers_from_env() -> anyhow::Result<usize> {
    match env::var(DYN_ROUTER_MIN_INITIAL_WORKERS) {
        Ok(value) => value.parse::<usize>().map_err(|error| {
            anyhow::anyhow!(
                "{DYN_ROUTER_MIN_INITIAL_WORKERS} must be a non-negative integer, got {value:?}: {error}"
            )
        }),
        Err(VarError::NotPresent) => Ok(0),
        Err(VarError::NotUnicode(_)) => {
            anyhow::bail!("{DYN_ROUTER_MIN_INITIAL_WORKERS} must be valid unicode")
        }
    }
}
