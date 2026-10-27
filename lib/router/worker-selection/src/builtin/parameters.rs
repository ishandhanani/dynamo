// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup parameters and provider registration for the default policy.

use dynamo_router_types::config::KvRouterConfig;
use dynamo_router_types::config::SharedCacheType;

/// YAML parameters of the `dynamo-default-cost-fn` provider; every field falls back to the router config.
#[derive(Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DefaultCostFnParameters {
    overlap_score_credit: Option<f64>,
    overlap_score_credit_decay: Option<f64>,
    prefill_load_scale: Option<f64>,
    decode_active_request_weight: Option<f64>,
    host_cache_hit_weight: Option<f64>,
    disk_cache_hit_weight: Option<f64>,
    shared_cache_multiplier: Option<f64>,
    router_temperature: Option<f64>,
}

/// Only the startup values consumed by the default scorer and picker.
#[derive(Clone, Copy, serde::Serialize)]
pub struct PolicyParameters {
    pub overlap_score_credit: f64,
    pub overlap_score_credit_decay: f64,
    pub prefill_load_scale: f64,
    pub decode_active_request_weight: f64,
    pub host_cache_hit_weight: f64,
    pub disk_cache_hit_weight: f64,
    pub shared_cache_multiplier: f64,
    pub router_temperature: f64,
}

impl From<&KvRouterConfig> for PolicyParameters {
    fn from(config: &KvRouterConfig) -> Self {
        DefaultCostFnParameters::default().resolve(config)
    }
}

impl DefaultCostFnParameters {
    /// Reject non-finite or out-of-range values; the message names the offending field.
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("overlap_score_credit", self.overlap_score_credit),
            (
                "overlap_score_credit_decay",
                self.overlap_score_credit_decay,
            ),
            ("prefill_load_scale", self.prefill_load_scale),
            (
                "decode_active_request_weight",
                self.decode_active_request_weight,
            ),
            ("host_cache_hit_weight", self.host_cache_hit_weight),
            ("disk_cache_hit_weight", self.disk_cache_hit_weight),
            ("shared_cache_multiplier", self.shared_cache_multiplier),
            ("router_temperature", self.router_temperature),
        ] {
            if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
                return Err(format!("{name} must be finite and non-negative"));
            }
        }
        Ok(())
    }

    pub fn resolve(&self, config: &KvRouterConfig) -> PolicyParameters {
        let overlap_score_credit = self
            .overlap_score_credit
            .unwrap_or(config.overlap_score_credit);
        // Legacy load-only configs retain tier weights for host accounting. Only explicit
        // policy parameters enable tier scoring independently of device credit.
        let inherited_cache_weight = |weight| {
            if overlap_score_credit == 0.0 {
                0.0
            } else {
                weight
            }
        };
        PolicyParameters {
            overlap_score_credit,
            overlap_score_credit_decay: self
                .overlap_score_credit_decay
                .unwrap_or(config.overlap_score_credit_decay),
            prefill_load_scale: self.prefill_load_scale.unwrap_or(config.prefill_load_scale),
            decode_active_request_weight: self
                .decode_active_request_weight
                .unwrap_or(config.decode_active_request_weight),
            host_cache_hit_weight: self
                .host_cache_hit_weight
                .unwrap_or(inherited_cache_weight(config.host_cache_hit_weight)),
            disk_cache_hit_weight: self
                .disk_cache_hit_weight
                .unwrap_or(inherited_cache_weight(config.disk_cache_hit_weight)),
            shared_cache_multiplier: self.shared_cache_multiplier.unwrap_or_else(|| {
                inherited_cache_weight(config.shared_cache_multiplier.unwrap_or(
                    match config.shared_cache_type {
                        SharedCacheType::None => 0.0,
                        SharedCacheType::Hicache => 0.5,
                    },
                ))
            }),
            router_temperature: self.router_temperature.unwrap_or(config.router_temperature),
        }
    }
}

impl PolicyParameters {
    pub fn snapshot(&self) -> serde_json::Map<String, serde_json::Value> {
        let serde_json::Value::Object(parameters) = serde_json::json!(self) else {
            unreachable!("PolicyParameters is a struct that serializes to a JSON object")
        };
        parameters
    }

    pub fn from_runtime(
        values: serde_json::Map<String, serde_json::Value>,
        config: &KvRouterConfig,
    ) -> Result<Self, crate::plugin::RuntimePolicyError> {
        use crate::plugin::RuntimePolicyError;
        // Runtime maps are resolved; startup parameters may be absent or null.
        if values.values().any(serde_json::Value::is_null) {
            return Err(RuntimePolicyError::InvalidParameters(
                "resolved runtime parameters must be numbers".into(),
            ));
        }
        let parameters: DefaultCostFnParameters =
            serde_json::from_value(serde_json::Value::Object(values))
                .map_err(|error| RuntimePolicyError::InvalidParameters(error.to_string()))?;
        parameters
            .validate()
            .map_err(|error| RuntimePolicyError::InvalidParameters(error.to_string()))?;
        Ok(parameters.resolve(config))
    }
}
