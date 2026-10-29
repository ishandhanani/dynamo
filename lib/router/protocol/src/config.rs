// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router configuration that belongs to the shared vocabulary: the per-request override a
//! caller sends with a request, and the tracking-hash trio that hashing, the indexer, and the
//! workers share. Everything else about the router's configuration lives with the stage it
//! configures; `dynamo-kv-router` owns the flat `KvRouterConfig` and assembles the slices.

use std::path::PathBuf;
use std::time::Duration;

use derive_builder::Builder;
use serde::{Deserialize, Serialize};

use crate::tracking_hash::TrackingHashAlgorithm;

pub const OVERLAP_SCORE_CREDIT_RANGE_ERROR: &str =
    "overlap_score_credit must be a finite, non-negative number";

pub fn overlap_score_credit_error_message(value: f64) -> Option<&'static str> {
    if value.is_finite() && value >= 0.0 {
        None
    } else {
        Some(OVERLAP_SCORE_CREDIT_RANGE_ERROR)
    }
}

#[doc(hidden)]
pub fn validate_overlap_score_credit(value: f64) -> Result<(), String> {
    let Some(message) = overlap_score_credit_error_message(value) else {
        return Ok(());
    };
    Err(message.to_string())
}

#[doc(hidden)]
pub fn validate_min(field: &str, value: f64, min: f64) -> Result<(), String> {
    if value >= min {
        return Ok(());
    }
    Err(format!("{field} must be greater than or equal to {min}"))
}

#[doc(hidden)]
pub fn validate_duration_secs(field: &str, value: f64) -> Result<(), String> {
    Duration::try_from_secs_f64(value)
        .map(|_| ())
        .map_err(|error| format!("{field} is not a valid duration: {error}"))
}

#[doc(hidden)]
pub fn validate_finite_non_negative(field: &str, value: f64) -> Result<(), String> {
    if value.is_finite() && value >= 0.0 {
        return Ok(());
    }
    Err(format!("{field} must be finite and non-negative"))
}

#[doc(hidden)]
pub fn validate_range(field: &str, value: f64, min: f64, max: f64) -> Result<(), String> {
    if value >= min && value <= max {
        return Ok(());
    }
    Err(format!("{field} must be between {min} and {max}"))
}

pub fn apply_deprecated_overlap_score_weight_override(
    value: f64,
    overlap_score_credit: &mut f64,
    prefill_load_scale: &mut f64,
) {
    *prefill_load_scale = value;
    if value == 0.0 {
        *overlap_score_credit = 0.0;
    }
}

fn apply_deprecated_overlap_score_weight_override_option(
    value: f64,
    overlap_score_credit: &mut Option<f64>,
    prefill_load_scale: &mut Option<f64>,
) {
    *prefill_load_scale = Some(value);
    if value == 0.0 {
        *overlap_score_credit = Some(0.0);
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RouterConfigOverrideSerde {
    overlap_score_credit: Option<f64>,
    prefill_load_scale: Option<f64>,
    overlap_score_weight: Option<f64>,
    router_temperature: Option<f64>,
    assume_kv_reuse: Option<bool>,
    track_prefill_tokens: Option<bool>,
    shared_cache_multiplier: Option<f64>,
}

/// Override configuration for router settings that can be specified per-request
#[derive(Debug, Clone, Default, Builder, Serialize, Deserialize)]
#[serde(try_from = "RouterConfigOverrideSerde")]
pub struct RouterConfigOverride {
    /// Device-local prefix-overlap credit multiplier applied to the prefill
    /// load before sampling. Values must be finite and non-negative. Values above
    /// 1.0 give device overlap extra credit. Set to 0.0 to ignore prefix matching.
    #[builder(default)]
    pub overlap_score_credit: Option<f64>,

    /// Scale applied to the adjusted prefill load after device/lower-tier
    /// cache-hit credits have been subtracted.
    #[builder(default)]
    pub prefill_load_scale: Option<f64>,

    #[builder(default)]
    pub router_temperature: Option<f64>,

    #[builder(default)]
    pub assume_kv_reuse: Option<bool>,

    #[builder(default)]
    pub track_prefill_tokens: Option<bool>,

    /// Per-request override of `shared_cache_multiplier`.
    #[builder(default)]
    pub shared_cache_multiplier: Option<f64>,
}

impl RouterConfigOverride {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(value) = self.overlap_score_credit {
            validate_overlap_score_credit(value)?;
        }
        if let Some(value) = self.prefill_load_scale {
            validate_finite_non_negative("prefill_load_scale", value)?;
        }
        if let Some(value) = self.router_temperature {
            validate_finite_non_negative("router_temperature", value)?;
        }
        if let Some(value) = self.shared_cache_multiplier {
            validate_range("shared_cache_multiplier", value, 0.0, 1.0)?;
        }
        Ok(())
    }
}

impl TryFrom<RouterConfigOverrideSerde> for RouterConfigOverride {
    type Error = String;

    fn try_from(compat: RouterConfigOverrideSerde) -> Result<Self, Self::Error> {
        let mut overlap_score_credit = compat.overlap_score_credit;
        let mut prefill_load_scale = compat.prefill_load_scale;

        if let Some(overlap_score_weight) = compat.overlap_score_weight {
            apply_deprecated_overlap_score_weight_override_option(
                overlap_score_weight,
                &mut overlap_score_credit,
                &mut prefill_load_scale,
            );
        }

        let config = Self {
            overlap_score_credit,
            prefill_load_scale,
            router_temperature: compat.router_temperature,
            assume_kv_reuse: compat.assume_kv_reuse,
            track_prefill_tokens: compat.track_prefill_tokens,
            shared_cache_multiplier: compat.shared_cache_multiplier,
        };
        config.validate()?;
        Ok(config)
    }
}

/// The tracking-hash trio: part of the wire contract shared by hashing, the indexer, and workers.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct TrackingHashConfig {
    pub algorithm: TrackingHashAlgorithm,
    pub key_file: Option<PathBuf>,
    pub key_id: Option<String>,
}
