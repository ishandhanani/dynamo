// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The worker-selection slice of the router configuration.

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

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
    /// The router's defaults; `dynamo-kv-router` asserts the flat defaults map onto these.
    fn default() -> Self {
        Self {
            overlap_score_credit: 1.0,
            overlap_score_credit_decay: 0.0,
            prefill_load_scale: 1.0,
            decode_active_request_weight: 0.0,
            host_cache_hit_weight: 0.75,
            disk_cache_hit_weight: 0.25,
            router_temperature: 0.0,
            shared_cache_multiplier: None,
            shared_cache_type: SharedCacheType::default(),
            assume_kv_reuse: true,
            conditional_disagg: ConditionalDisaggConfig {
                enabled: false,
                policy: ConditionalDisaggPolicyKind::default(),
                eff_isl_threshold: DEFAULT_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD,
                eff_isl_ratio_threshold: DEFAULT_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD,
                prefill_busy_threshold: None,
                decode_busy_threshold: None,
            },
        }
    }
}

/// Default effective-ISL absolute threshold (tokens). A request bypasses to
/// AGG only if its net-new prefill stays under this cap.
pub const DEFAULT_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD: usize = 2048;

/// Default effective-ISL ratio threshold. A request bypasses to AGG only if
/// `eff_isl / prompt_tokens` stays under this fraction.
pub const DEFAULT_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD: f64 = 0.7;

/// Type of external shared KV cache to query during routing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SharedCacheType {
    /// No shared cache (default).
    #[default]
    None,
    /// HiCache L3 shared cache — queries sglang workers via the request plane.
    Hicache,
}

impl fmt::Display for SharedCacheType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::Hicache => f.write_str("hicache"),
        }
    }
}

impl FromStr for SharedCacheType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(Self::None),
            "hicache" => Ok(Self::Hicache),
            _ => Err(format!(
                "unknown shared_cache_type: {s:?}, expected 'none' or 'hicache'"
            )),
        }
    }
}

/// Which conditional-disagg bypass policy to run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionalDisaggPolicyKind {
    /// Bypass when effective ISL is below both the absolute and ratio thresholds.
    #[default]
    IslBounding,
    /// Bypass when the chosen prefill worker is over the prefill-busy line.
    PrefillLoad,
    /// Bypass when either `isl_bounding` or `prefill_load` would bypass.
    IslOrLoad,
}

impl fmt::Display for ConditionalDisaggPolicyKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IslBounding => f.write_str("isl_bounding"),
            Self::PrefillLoad => f.write_str("prefill_load"),
            Self::IslOrLoad => f.write_str("isl_or_load"),
        }
    }
}

impl FromStr for ConditionalDisaggPolicyKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "isl_bounding" => Ok(Self::IslBounding),
            "prefill_load" => Ok(Self::PrefillLoad),
            "isl_or_load" => Ok(Self::IslOrLoad),
            _ => Err(format!(
                "unknown conditional_disagg_policy: {s:?}, expected 'isl_bounding', 'prefill_load', or 'isl_or_load'"
            )),
        }
    }
}

/// Process-wide configuration for worker selection in a custom Dynamo image.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerSelectionConfig {
    aggregated: Option<String>,
    prefill: Option<String>,
    decode: Option<String>,
    encode: Option<String>,
    instances: HashMap<String, WorkerSelectionInstance>,
}

impl WorkerSelectionConfig {
    /// The resolved selections; `dynamo-kv-router` builds this from the policy file.
    #[doc(hidden)]
    pub fn new(
        aggregated: Option<String>,
        prefill: Option<String>,
        decode: Option<String>,
        encode: Option<String>,
        instances: HashMap<String, WorkerSelectionInstance>,
    ) -> Self {
        Self {
            aggregated,
            prefill,
            decode,
            encode,
            instances,
        }
    }

    /// The selected instance for a full-request worker pool.
    pub fn aggregated_instance(&self) -> Option<&str> {
        self.aggregated.as_deref()
    }

    #[doc(hidden)]
    pub fn prefill_instance(&self) -> Option<&str> {
        self.prefill.as_deref()
    }

    #[doc(hidden)]
    pub fn decode_instance(&self) -> Option<&str> {
        self.decode.as_deref()
    }

    #[doc(hidden)]
    pub fn encode_instance(&self) -> Option<&str> {
        self.encode.as_deref()
    }

    /// Look up one named instance.
    pub fn instance(&self, name: &str) -> Option<&WorkerSelectionInstance> {
        self.instances.get(name)
    }

    /// Return configured instance names in stable order for diagnostics.
    pub fn instance_names(&self) -> Vec<String> {
        let mut names = self.instances.keys().cloned().collect::<Vec<_>>();
        names.sort_unstable();
        names
    }
}

/// One named, parameterized worker-selection policy instance.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerSelectionInstance {
    policy_type: String,
    parameters: serde_yaml::Value,
}

impl WorkerSelectionInstance {
    /// One configured policy instance; `dynamo-kv-router` builds it from the policy file.
    #[doc(hidden)]
    pub fn new(policy_type: String, parameters: serde_yaml::Value) -> Self {
        Self {
            policy_type,
            parameters,
        }
    }

    /// The policy type registered by a linked policy crate.
    pub fn policy_type(&self) -> &str {
        &self.policy_type
    }

    /// YAML parameters owned and validated by the linked policy crate.
    pub fn parameters(&self) -> &serde_yaml::Value {
        &self.parameters
    }
}
