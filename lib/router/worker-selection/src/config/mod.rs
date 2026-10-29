// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The worker-selection slice of the router configuration.

pub use dynamo_router_types::config::worker_selection::*;
use dynamo_router_types::config::{ConditionalDisaggPolicyKind, KvRouterConfig, SharedCacheType};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelectionConfig {
    pub overlap_score_credit: f64,
    pub overlap_score_credit_decay: f64,
    pub prefill_load_scale: f64,
    pub decode_active_request_weight: f64,
    pub host_cache_hit_weight: f64,
    pub disk_cache_hit_weight: f64,
    pub router_temperature: f64,
    pub shared_cache_multiplier: Option<f64>,
    pub shared_cache_type: SharedCacheType,
    pub assume_kv_reuse: bool,
    pub conditional_disagg: ConditionalDisaggConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConditionalDisaggConfig {
    pub enabled: bool,
    pub policy: ConditionalDisaggPolicyKind,
    pub eff_isl_threshold: usize,
    pub eff_isl_ratio_threshold: f64,
    pub prefill_busy_threshold: Option<f64>,
    pub decode_busy_threshold: Option<f64>,
}

impl Default for SelectionConfig {
    /// The selection defaults are the router's defaults; the aggregate's `Default` stays the
    /// single source until it leaves the shared crate.
    fn default() -> Self {
        Self::from(&KvRouterConfig::default())
    }
}

impl From<&KvRouterConfig> for SelectionConfig {
    fn from(config: &KvRouterConfig) -> Self {
        Self {
            overlap_score_credit: config.overlap_score_credit,
            overlap_score_credit_decay: config.overlap_score_credit_decay,
            prefill_load_scale: config.prefill_load_scale,
            decode_active_request_weight: config.decode_active_request_weight,
            host_cache_hit_weight: config.host_cache_hit_weight,
            disk_cache_hit_weight: config.disk_cache_hit_weight,
            router_temperature: config.router_temperature,
            shared_cache_multiplier: config.shared_cache_multiplier,
            shared_cache_type: config.shared_cache_type,
            assume_kv_reuse: config.router_assume_kv_reuse,
            conditional_disagg: ConditionalDisaggConfig {
                enabled: config.conditional_disagg_enabled,
                policy: config.conditional_disagg_policy,
                eff_isl_threshold: config.conditional_disagg_eff_isl_threshold,
                eff_isl_ratio_threshold: config.conditional_disagg_eff_isl_ratio_threshold,
                prefill_busy_threshold: config.conditional_disagg_prefill_busy_threshold,
                decode_busy_threshold: config.conditional_disagg_decode_busy_threshold,
            },
        }
    }
}
