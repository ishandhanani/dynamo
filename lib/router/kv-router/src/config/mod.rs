// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The router configuration, assembled: the per-stage slices owned by their crates, the hashing
//! trio from the types crate, and the composition switches the builder reads once. The flat
//! [`KvRouterConfig`] stays the wire shape embedded in model deployment cards; this is the one
//! place it is converted.

use dynamo_admission::config::{AdmissionConfig, ClassifierConfig};
use dynamo_admission::policy_config::RouterPolicyConfigError;
use dynamo_kv_indexer::config::IndexerConfig;
pub use dynamo_router_types::config::{RouterConfigOverride, TrackingHashConfig};
// The flat configuration's field types, so a host that handles only the flat shape reaches
// them here instead of depending on every stage crate.
pub use dynamo_admission::config::RouterQueuePolicy;
pub use dynamo_kv_indexer::config::ApproximateCachePolicyKind;
pub use dynamo_slot_tracker::config::RouterPrefillLoadModel;
pub use dynamo_worker_selection::config::{ConditionalDisaggPolicyKind, SharedCacheType};

mod flat;
mod policy_file;

use dynamo_scheduler::config::RouterConfig;
use dynamo_slot_tracker::config::SlotsConfig;
use dynamo_worker_selection::config::{ConditionalDisaggConfig, SelectionConfig};
pub use flat::*;
pub use policy_file::{RouterPolicyConfig, RouterSettings};
use serde::Serialize;

/// Build a [`KvRouterConfig`] from defaults and standard Dynamo environment variables.
///
/// # Panics
///
/// Panics when `DYN_ROUTER_TRACKING_HASH` is not a supported algorithm. Startup
/// paths should use [`try_kv_router_config_from_dynamo_env`] to report the error.
pub fn kv_router_config_from_dynamo_env() -> KvRouterConfig {
    try_kv_router_config_from_dynamo_env()
        .unwrap_or_else(|error| panic!("invalid Dynamo router environment configuration: {error}"))
}

/// Build a [`KvRouterConfig`] from standard Dynamo environment variables.
pub fn try_kv_router_config_from_dynamo_env() -> Result<KvRouterConfig, String> {
    let config = kv_router_config_from_lookup(|key| std::env::var(key).ok())?;
    log_env_config(&config);
    Ok(config)
}

fn log_env_config(config: &KvRouterConfig) {
    tracing::info!(
        overlap_score_credit = config.overlap_score_credit,
        overlap_score_credit_decay = config.overlap_score_credit_decay,
        prefill_load_scale = config.prefill_load_scale,
        decode_active_request_weight = config.decode_active_request_weight,
        router_temperature = config.router_temperature,
        use_kv_events = config.use_kv_events,
        router_replica_sync = config.router_replica_sync,
        router_track_active_blocks = config.router_track_active_blocks,
        router_track_output_blocks = config.router_track_output_blocks,
        router_assume_kv_reuse = config.router_assume_kv_reuse,
        router_track_prefill_tokens = config.router_track_prefill_tokens,
        router_tracking_hash = %config.router_tracking_hash,
        router_tracking_key_id = ?config.router_tracking_key_id,
        router_queue_threshold = ?config.router_queue_threshold,
        router_policy_config = ?config.router_policy_config,
        router_prefill_policy = ?config.router_prefill_policy,
        router_decode_policy = ?config.router_decode_policy,
        conditional_disagg_enabled = config.conditional_disagg_enabled,
        conditional_disagg_policy = ?config.conditional_disagg_policy,
        conditional_disagg_eff_isl_threshold = config.conditional_disagg_eff_isl_threshold,
        conditional_disagg_eff_isl_ratio_threshold = config.conditional_disagg_eff_isl_ratio_threshold,
        conditional_disagg_prefill_busy_threshold = ?config.conditional_disagg_prefill_busy_threshold,
        conditional_disagg_decode_busy_threshold = ?config.conditional_disagg_decode_busy_threshold,
        router_predicted_ttl_secs = ?config.router_predicted_ttl_secs,
        router_ttl_secs = config.router_ttl_secs,
        router_event_threads = config.router_event_threads,
        router_queue_policy = %config.router_queue_policy,
        use_remote_indexer = config.use_remote_indexer,
        shared_cache_multiplier = ?config.shared_cache_multiplier,
        shared_cache_type = %config.shared_cache_type,
        host_cache_hit_weight = config.host_cache_hit_weight,
        disk_cache_hit_weight = config.disk_cache_hit_weight,
        router_prefill_load_model = %config.router_prefill_load_model,
        router_approximate_cache_policy = %config.router_approximate_cache_policy,
        "KvRouterConfig initialized (DYN_* env overrides applied)"
    );
}

/// Composition switches: read once when the service is wired, never by a stage.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WiringConfig {
    pub use_kv_events: bool,
    pub replica_sync: bool,
    pub use_remote_indexer: bool,
    pub serve_indexer: bool,
    pub skip_initial_worker_wait: bool,
    pub shared_cache_type: SharedCacheType,
    pub policy_model_name: Option<String>,
    pub router_policy_config: Option<String>,
    pub prefill_policy: Option<String>,
    pub decode_policy: Option<String>,
}

impl From<&KvRouterConfig> for WiringConfig {
    fn from(config: &KvRouterConfig) -> Self {
        Self {
            use_kv_events: config.use_kv_events,
            replica_sync: config.router_replica_sync,
            use_remote_indexer: config.use_remote_indexer,
            serve_indexer: config.serve_indexer,
            skip_initial_worker_wait: config.skip_initial_worker_wait,
            shared_cache_type: config.shared_cache_type,
            policy_model_name: config.policy_model_name.clone(),
            router_policy_config: config.router_policy_config.clone(),
            prefill_policy: config.router_prefill_policy.clone(),
            decode_policy: config.router_decode_policy.clone(),
        }
    }
}

/// Everything the composition reads from one flat configuration: the kernel's stage slices
/// and the wiring switches the builder consumes.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResolvedConfig {
    pub router: RouterConfig,
    pub wiring: WiringConfig,
}

impl ResolvedConfig {
    /// The single flat-to-nested conversion: applies the policy file's router settings,
    /// validates the flat config, then builds every slice from values that passed the checks.
    pub fn from_kv_router_config(mut config: KvRouterConfig) -> Result<Self, String> {
        config.apply_policy_config()?;
        config.validate_config()?;
        Ok(Self {
            router: router_config(&config)?,
            wiring: WiringConfig::from(&config),
        })
    }

    /// Read the `DYN_ROUTER_*` knobs from the process environment.
    pub fn from_env() -> Result<Self, String> {
        Self::from_kv_router_config(try_kv_router_config_from_dynamo_env()?)
    }

    /// The effective settings as one JSON document, for operators.
    pub fn effective(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("router config serializes")
    }
}

/// The kernel's stage slices from a flat configuration whose policy file is already applied.
pub fn router_config(config: &KvRouterConfig) -> Result<RouterConfig, String> {
    Ok(RouterConfig {
        hashing: TrackingHashConfig::from(config),
        indexer: IndexerConfig::from(config),
        slots: SlotsConfig::from(config),
        admission: admission_config(config).map_err(|e| e.to_string())?,
        classifier: classifier_config(config).map_err(|e| e.to_string())?,
        selection: SelectionConfig::from(config),
    })
}

/// The kernel's stage slices from a flat configuration, applying and validating it first.
pub fn resolve_router_config(config: KvRouterConfig) -> Result<RouterConfig, String> {
    ResolvedConfig::from_kv_router_config(config).map(|resolved| resolved.router)
}

/// The admission slice: queue limits and the loaded policy file's profiles.
pub fn admission_config(
    config: &KvRouterConfig,
) -> Result<AdmissionConfig, RouterPolicyConfigError> {
    Ok(AdmissionConfig {
        queue_threshold: config.router_queue_threshold,
        queue_policy: config.router_queue_policy,
        profiles: config
            .router_policy_config
            .as_deref()
            .map(RouterPolicyConfig::from_path)
            .transpose()?
            .map(|policy| policy.profiles()),
    })
}

/// The classifier slice: the request classifier the policy file selects, if any.
pub fn classifier_config(
    config: &KvRouterConfig,
) -> Result<ClassifierConfig, RouterPolicyConfigError> {
    Ok(ClassifierConfig {
        request_classifier: config.request_classifier_config()?.cloned(),
    })
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

impl From<&KvRouterConfig> for SlotsConfig {
    fn from(config: &KvRouterConfig) -> Self {
        Self {
            track_active_blocks: config.router_track_active_blocks,
            track_output_blocks: config.router_track_output_blocks,
            track_prefill_tokens: config.router_track_prefill_tokens,
            prefill_load_model: config.router_prefill_load_model,
            ttl_secs: config.router_ttl_secs,
            predicted_ttl_secs: config.router_predicted_ttl_secs,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_config_round_trips_the_defaults() {
        let flat = KvRouterConfig::default();
        let resolved = ResolvedConfig::from_kv_router_config(flat.clone()).unwrap();
        let nested = &resolved.router;
        assert_eq!(nested.indexer.event_threads, flat.router_event_threads);
        assert_eq!(nested.slots.ttl_secs, flat.router_ttl_secs);
        assert_eq!(nested.admission.queue_policy, flat.router_queue_policy);
        assert_eq!(nested.selection.router_temperature, flat.router_temperature);
        assert_eq!(resolved.wiring.use_kv_events, flat.use_kv_events);
        assert_eq!(nested.hashing.algorithm, flat.router_tracking_hash);
        assert!(resolved.effective().get("router").is_some());
        assert_eq!(
            *nested,
            RouterConfig::default(),
            "stage defaults mirror the flat defaults"
        );
    }

    #[test]
    fn env_lookup_with_no_overrides_matches_the_defaults() {
        let config = kv_router_config_from_lookup(|_| None).unwrap();
        assert_eq!(
            serde_json::to_value(&config).unwrap(),
            serde_json::to_value(KvRouterConfig::default()).unwrap()
        );
    }
}
