// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The router policy file: the `router:` settings applied onto the flat configuration, the
//! queue profiles admission owns, and the plugin selections the registries resolve. The raw
//! YAML shapes live here; the resolved types live with the stage that reads them.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use dynamo_admission::classifier::RequestClassifierConfig;
use dynamo_admission::config::RouterQueuePolicy;
use dynamo_admission::policy_config::{
    PolicyProfile, PolicyProfiles, RawPolicyClassConfig, RawPolicyProfile, RawUncachedIslBucket,
    RouterPolicyConfigError, resolve_profile, validate_identifier,
};
use dynamo_worker_selection::config::{WorkerSelectionConfig, WorkerSelectionInstance};

use super::KvRouterConfig;

/// Process-wide cache and tracking settings, separate from policy parameters.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(transparent)]
pub struct RouterSettings(HashMap<String, serde_json::Value>);

impl RouterSettings {
    pub fn apply(&self, config: &mut KvRouterConfig) -> Result<(), String> {
        // Deserialize into each field's existing type, including nullable fields.
        // Do not round-trip KvRouterConfig: it has process-local, non-wire fields.
        macro_rules! apply_fields {
            ($($field:ident),* $(,)?) => {
                for (name, value) in &self.0 {
                    match name.as_str() {
                        $(stringify!($field) => {
                            config.$field = serde_json::from_value(value.clone())
                                .map_err(|error| format!("router.{name}: {error}"))?;
                        })*
                        _ => return Err(format!("unknown router setting: {name}")),
                    }
                }
            };
        }
        apply_fields!(
            host_cache_hit_weight,
            disk_cache_hit_weight,
            use_kv_events,
            router_replica_sync,
            router_track_active_blocks,
            router_track_output_blocks,
            router_assume_kv_reuse,
            router_track_prefill_tokens,
            router_tracking_hash,
            router_tracking_key_file,
            router_tracking_key_id,
            router_prefill_load_model,
            router_ttl_secs,
            router_approximate_cache_policy,
            router_event_threads,
            use_remote_indexer,
            serve_indexer,
            enable_session_prefix_index,
            shared_cache_type,
            router_predicted_ttl_secs,
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RouterPolicyConfig {
    router: Option<RouterSettings>,
    root: Option<PolicyProfile>,
    models: HashMap<String, PolicyProfile>,
    worker_selection: Option<WorkerSelectionConfig>,
    request_classifier: Option<RequestClassifierConfig>,
}

impl RouterPolicyConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, RouterPolicyConfigError> {
        let path = path.as_ref();
        let contents =
            fs::read_to_string(path).map_err(|source| RouterPolicyConfigError::Read {
                path: path.display().to_string(),
                source,
            })?;
        Self::from_yaml(&contents).map_err(|error| match error {
            RouterPolicyConfigError::Parse { source, .. } => RouterPolicyConfigError::Parse {
                path: path.display().to_string(),
                source,
            },
            other => other,
        })
    }

    pub fn from_yaml(contents: &str) -> Result<Self, RouterPolicyConfigError> {
        let raw: RawRouterPolicyConfig =
            serde_yaml::from_str(contents).map_err(|source| RouterPolicyConfigError::Parse {
                path: "<inline>".to_string(),
                source,
            })?;
        raw.resolve()
    }

    pub fn resolve_profile(
        &self,
        model_name: Option<&str>,
        fallback_threshold: Option<f64>,
        fallback_policy: RouterQueuePolicy,
    ) -> PolicyProfile {
        // Model profiles replace the root wholesale; the synthetic profile is
        // constructed only when neither configured profile applies.
        model_name
            .and_then(|name| self.models.get(name))
            .or(self.root.as_ref())
            .cloned()
            .unwrap_or_else(|| PolicyProfile::synthetic(fallback_threshold, fallback_policy))
    }

    pub(crate) fn router(&self) -> Option<&RouterSettings> {
        self.router.as_ref()
    }

    /// Returns the process-wide worker-selection policy configuration, if present.
    pub fn worker_selection(&self) -> Option<&WorkerSelectionConfig> {
        self.worker_selection.as_ref()
    }

    /// Returns the process-wide request-classifier plugin configuration, if present.
    pub fn request_classifier(&self) -> Option<&RequestClassifierConfig> {
        self.request_classifier.as_ref()
    }

    /// Whether this document configures queue policy profiles.
    pub fn has_routing_profiles(&self) -> bool {
        self.root.is_some() || !self.models.is_empty()
    }

    /// The queue profiles for the admission slice: the root and per-model profiles.
    pub fn profiles(&self) -> PolicyProfiles {
        PolicyProfiles::new(self.root.clone(), self.models.clone())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRouterPolicyConfig {
    router: Option<RouterSettings>,
    default_policy_family: Option<String>,
    policy_classes: Option<Vec<RawPolicyClassConfig>>,
    uncached_isl_buckets: Option<Vec<RawUncachedIslBucket>>,
    #[serde(default)]
    models: HashMap<String, RawPolicyProfile>,
    worker_selection: Option<RawWorkerSelectionConfig>,
    request_classifier: Option<RawRequestClassifierConfig>,
}

impl RawRouterPolicyConfig {
    fn resolve(self) -> Result<RouterPolicyConfig, RouterPolicyConfigError> {
        if let Some(router) = &self.router {
            router
                .apply(&mut KvRouterConfig::default())
                .map_err(RouterPolicyConfigError::Validation)?;
        }
        let root = match (
            self.default_policy_family,
            self.policy_classes,
            self.uncached_isl_buckets,
        ) {
            (None, None, None) => None,
            (Some(default_policy_family), Some(policy_classes), Some(uncached_isl_buckets)) => {
                Some(resolve_profile(
                    RawPolicyProfile {
                        default_policy_family,
                        policy_classes,
                        uncached_isl_buckets,
                    },
                    "root",
                )?)
            }
            _ => {
                return Err(RouterPolicyConfigError::Validation(
                    "root profile must specify default_policy_family, uncached_isl_buckets, and policy_classes when any root profile field is present".to_string(),
                ));
            }
        };

        let mut models = HashMap::with_capacity(self.models.len());
        for (model_name, profile) in self.models {
            if model_name.is_empty() {
                return Err(RouterPolicyConfigError::Validation(
                    "model profile name must not be empty".to_string(),
                ));
            }
            let resolved = resolve_profile(profile, &format!("model {model_name:?}"))?;
            models.insert(model_name, resolved);
        }

        let worker_selection = match self.worker_selection {
            Some(config) => Some(config.resolve()?),
            None => None,
        };

        let request_classifier = self
            .request_classifier
            .map(|config| config.resolve())
            .transpose()?;
        if self.router.is_none()
            && root.is_none()
            && models.is_empty()
            && worker_selection.is_none()
            && request_classifier.is_none()
        {
            return Err(RouterPolicyConfigError::Validation(
                "router policy config must define router settings, a root profile, at least one model profile, worker_selection, or request_classifier".to_string(),
            ));
        }

        Ok(RouterPolicyConfig {
            router: self.router,
            root,
            models,
            worker_selection,
            request_classifier,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[doc(hidden)]
pub struct RawWorkerSelectionConfig {
    aggregated: Option<String>,
    prefill: Option<String>,
    decode: Option<String>,
    encode: Option<String>,
    #[serde(default)]
    instances: Vec<RawWorkerSelectionInstance>,
}

impl RawWorkerSelectionConfig {
    #[doc(hidden)]
    pub fn resolve(self) -> Result<WorkerSelectionConfig, RouterPolicyConfigError> {
        if self.instances.is_empty()
            && self.aggregated.is_none()
            && self.prefill.is_none()
            && self.decode.is_none()
            && self.encode.is_none()
        {
            return Err(RouterPolicyConfigError::Validation(
                "worker_selection must define an instance or an aggregated, prefill, decode, or encode selection"
                    .to_string(),
            ));
        }

        let mut instances = HashMap::with_capacity(self.instances.len());
        for raw in self.instances {
            validate_identifier(&raw.name, "instance", "worker_selection")?;
            if raw.name == "default" {
                return Err(RouterPolicyConfigError::Validation(
                    "worker_selection instance name 'default' is reserved for Dynamo's built-in selector".to_string(),
                ));
            }
            validate_identifier(&raw.policy_type, "policy type", "worker_selection")?;
            if !matches!(raw.parameters, serde_yaml::Value::Mapping(_)) {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "worker_selection instance {:?} parameters must be a YAML mapping",
                    raw.name
                )));
            }
            let instance = WorkerSelectionInstance::new(raw.policy_type, raw.parameters);
            if instances.insert(raw.name.clone(), instance).is_some() {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "worker_selection contains duplicate instance {:?}",
                    raw.name
                )));
            }
        }

        for (stage, selected) in [
            ("aggregated", self.aggregated.as_deref()),
            ("prefill", self.prefill.as_deref()),
            ("decode", self.decode.as_deref()),
            ("encode", self.encode.as_deref()),
        ] {
            if let Some(selected) = selected
                && selected != "default"
                && !instances.contains_key(selected)
            {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "worker_selection {stage} {selected:?} does not name a configured instance"
                )));
            }
        }

        Ok(WorkerSelectionConfig::new(
            self.aggregated,
            self.prefill,
            self.decode,
            self.encode,
            instances,
        ))
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawWorkerSelectionInstance {
    name: String,
    #[serde(rename = "type")]
    policy_type: String,
    #[serde(default = "empty_parameters")]
    parameters: serde_yaml::Value,
}

fn empty_parameters() -> serde_yaml::Value {
    serde_yaml::Value::Mapping(Default::default())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[doc(hidden)]
pub struct RawRequestClassifierConfig {
    #[serde(rename = "type")]
    classifier_type: String,
    #[serde(default = "empty_parameters")]
    parameters: serde_yaml::Value,
}

impl RawRequestClassifierConfig {
    #[doc(hidden)]
    pub fn resolve(self) -> Result<RequestClassifierConfig, RouterPolicyConfigError> {
        validate_identifier(
            &self.classifier_type,
            "classifier type",
            "request_classifier",
        )?;
        if self.classifier_type == "default" {
            return Err(RouterPolicyConfigError::Validation(
                "request_classifier type 'default' is reserved; omit request_classifier to use Dynamo's pass-through classifier"
                    .to_string(),
            ));
        }
        if !matches!(self.parameters, serde_yaml::Value::Mapping(_)) {
            return Err(RouterPolicyConfigError::Validation(
                "request_classifier parameters must be a YAML mapping".to_string(),
            ));
        }
        Ok(RequestClassifierConfig::new(
            self.classifier_type,
            self.parameters,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_admission::policy_config::*;
    use dynamo_router_protocol::load::{WorkerCapacity, WorkerLoadInput};
    use dynamo_router_protocol::protocols::{WorkerLoadProjection, WorkerWithDpRank};
    use rustc_hash::FxHashMap;

    fn prefill_load(active_prefill_tokens: usize) -> WorkerLoadInput {
        WorkerLoadInput::from_projection(WorkerLoadProjection {
            active_prefill_tokens,
            ..Default::default()
        })
    }

    #[test]
    fn worker_selection_only_config_preserves_parameter_mapping() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
worker_selection:
  aggregated: example
  instances:
    - name: example
      type: example-policy
      parameters:
        score_weight: 1.0
"#,
        )
        .unwrap();

        let selection = config.worker_selection().unwrap();
        assert_eq!(selection.aggregated_instance(), Some("example"));
        let instance = selection.instance("example").unwrap();
        assert_eq!(instance.policy_type(), "example-policy");
        assert!(matches!(
            instance.parameters(),
            serde_yaml::Value::Mapping(_)
        ));
        assert_eq!(
            config
                .resolve_profile(None, Some(2.0), RouterQueuePolicy::Wspt)
                .default_class()
                .queue_policy,
            RouterQueuePolicy::Wspt
        );
    }

    #[test]
    fn request_classifier_only_config_preserves_parameter_mapping() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
request_classifier:
  type: thunderagent
  parameters:
    pause_threshold: 0.9
"#,
        )
        .unwrap();

        let classifier = config.request_classifier().unwrap();
        assert_eq!(classifier.classifier_type(), "thunderagent");
        assert!(matches!(
            classifier.parameters(),
            serde_yaml::Value::Mapping(_)
        ));
    }

    #[test]
    fn rejects_invalid_request_classifier_config() {
        for yaml in [
            r#"
request_classifier:
  type: default
"#,
            r#"
request_classifier:
  type: thunderagent
  parameters: 1
"#,
            r#"
request_classifier:
  type: ""
"#,
        ] {
            assert!(
                RouterPolicyConfig::from_yaml(yaml).is_err(),
                "unexpectedly accepted {yaml}"
            );
        }
    }

    #[test]
    fn worker_selection_accepts_every_worker_type() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
worker_selection:
  prefill: prefill-policy
  decode: default
  encode: encode-policy
  instances:
    - name: prefill-policy
      type: cache-aware
      parameters: {}
    - name: encode-policy
      type: media-aware
      parameters: {}
"#,
        )
        .unwrap();

        let selection = config.worker_selection().unwrap();
        assert_eq!(selection.aggregated_instance(), None);
        assert_eq!(selection.prefill_instance(), Some("prefill-policy"));
        assert_eq!(selection.decode_instance(), Some("default"));
        assert_eq!(selection.encode_instance(), Some("encode-policy"));
    }

    #[test]
    fn rejects_invalid_worker_selection_config() {
        for yaml in [
            r#"
worker_selection: {}
"#,
            r#"
worker_selection:
  aggregated: missing
  instances:
    - name: present
      type: alpha
"#,
            r#"
worker_selection:
  prefill: missing
  instances:
    - name: present
      type: alpha
"#,
            r#"
worker_selection:
  decode: missing
  instances:
    - name: present
      type: alpha
"#,
            r#"
worker_selection:
  encode: missing
  instances:
    - name: present
      type: alpha
"#,
            r#"
worker_selection:
  default: present
  instances:
    - name: present
      type: alpha
"#,
            r#"
worker_selection:
  instances:
    - name: default
      type: alpha
"#,
            r#"
worker_selection:
  instances:
    - name: alpha
      type: alpha
      parameters: 1
"#,
        ] {
            assert!(
                RouterPolicyConfig::from_yaml(yaml).is_err(),
                "unexpectedly accepted {yaml}"
            );
        }
    }

    #[test]
    fn model_profile_replaces_root_and_unmatched_model_uses_root() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: root-default
    policy_family: standard
    cache_bucket: all
    queue_policy: wspt
    quantum: 8
    prefill_busy_threshold: 100
models:
  exact-model:
    default_policy_family: latency
    uncached_isl_buckets:
      - min_tokens: 0
        bucket: cached
      - min_tokens: 32
        bucket: uncached
    policy_classes:
      - name: model-cached
        policy_family: latency
        cache_bucket: cached
        quantum: 2
        request_queue_limit_per_worker: 0
      - name: model-uncached
        policy_family: latency
        cache_bucket: uncached
        quantum: 4
        prefill_busy_threshold_frac: 0.0
"#,
        )
        .unwrap();

        let exact = config.resolve_profile(Some("exact-model"), Some(3.0), RouterQueuePolicy::Wspt);
        assert_eq!(exact.classes().len(), 2);
        assert_eq!(exact.default_class().name, "model-cached");
        assert_eq!(
            exact
                .default_class()
                .busy_thresholds
                .prefill_busy_threshold_frac,
            None
        );
        assert!(!exact.default_class().queueing_enabled());
        assert!(
            exact
                .class(exact.resolve_class_index(None, usize::MAX))
                .queueing_enabled()
        );
        assert_eq!(exact.default_class().queue_policy, RouterQueuePolicy::Fcfs);
        assert_eq!(
            exact.default_class().request_queue_limit_per_worker,
            Some(0)
        );
        assert_eq!(
            exact
                .class(exact.resolve_class_index(Some("unknown"), usize::MAX))
                .name,
            "model-uncached",
            "unknown policies must use the model's default family and observed bucket"
        );

        let unmatched = config.resolve_profile(Some("other"), Some(3.0), RouterQueuePolicy::Fcfs);
        assert_eq!(unmatched.default_class().name, "root-default");
        assert_eq!(
            unmatched
                .default_class()
                .busy_thresholds
                .prefill_busy_threshold,
            Some(100)
        );
        assert_eq!(
            unmatched
                .default_class()
                .busy_thresholds
                .prefill_busy_threshold_frac,
            None
        );
    }

    #[test]
    fn has_room_without_limits_or_reported_budget() {
        let mut limits = BusyThresholds::default();
        assert!(!limits.is_configured());
        assert!(limits.has_room(&prefill_load(usize::MAX), WorkerCapacity::default()));

        // An unreported budget is taken as DEFAULT_MAX_BATCHED_TOKENS.
        limits.prefill_busy_threshold_frac = Some(0.5);
        let half_default = (dynamo_router_protocol::load::DEFAULT_MAX_BATCHED_TOKENS / 2) as usize;
        assert!(limits.has_room(&prefill_load(half_default), WorkerCapacity::default()));
        assert!(!limits.has_room(&prefill_load(half_default + 1), WorkerCapacity::default()));
    }

    #[test]
    fn decode_limit_bounds_held_kv_blocks() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: decode-bound
    policy_family: standard
    cache_bucket: all
    quantum: 4
    decode_busy_threshold_frac: 0.5
"#,
        )
        .unwrap();
        let profile = config.resolve_profile(None, None, RouterQueuePolicy::Fcfs);
        let class = profile.default_class();
        assert!(class.queueing_enabled());

        let limits = class.busy_thresholds;
        let held = |active_decode_blocks| {
            WorkerLoadInput::from_projection(WorkerLoadProjection {
                active_decode_blocks,
                // The incoming request's blocks do not count toward the limit.
                additional_active_blocks: 1_000,
                ..Default::default()
            })
        };
        let kv = |total_kv_blocks| WorkerCapacity::new(None, total_kv_blocks);
        assert!(limits.has_room(&held(50), kv(Some(100))));
        assert!(!limits.has_room(&held(51), kv(Some(100))));
        // An unreported KV capacity skips the decode limit.
        assert!(limits.has_room(&held(1_000_000), kv(None)));

        let invalid = RouterPolicyConfig::from_yaml(
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: decode-bound
    policy_family: standard
    cache_bucket: all
    quantum: 4
    decode_busy_threshold_frac: -0.1
"#,
        )
        .unwrap_err();
        assert!(
            invalid
                .to_string()
                .contains("decode_busy_threshold_frac must be finite and non-negative")
        );
    }

    #[test]
    fn rootless_model_config_falls_back_for_unmatched_model() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
models:
  exact-model:
    default_policy_family: standard
    uncached_isl_buckets:
      - min_tokens: 0
        bucket: all
    policy_classes:
      - name: absolute
        policy_family: standard
        cache_bucket: all
        quantum: 4
        prefill_busy_threshold: 10
        prefill_busy_threshold_frac: 0.5
"#,
        )
        .unwrap();

        let exact = config.resolve_profile(Some("exact-model"), Some(7.0), RouterQueuePolicy::Wspt);
        let limits = exact.default_class().busy_thresholds;
        let budget = |tokens| WorkerCapacity::new(Some(tokens), None);
        assert!(!limits.has_room(&prefill_load(11), budget(10_000_000)));
        assert!(!limits.has_room(&prefill_load(6), budget(10)));
        assert!(limits.has_room(&prefill_load(5), budget(10)));

        let fallback = config.resolve_profile(Some("other"), Some(7.0), RouterQueuePolicy::Wspt);
        assert_eq!(fallback.default_class().name, SYNTHETIC_POLICY_CLASS);
        assert_eq!(
            fallback
                .default_class()
                .busy_thresholds
                .prefill_busy_threshold_frac,
            Some(7.0)
        );
        assert_eq!(
            fallback.default_class().queue_policy,
            RouterQueuePolicy::Wspt
        );
    }

    #[test]
    fn rejects_interacting_profile_errors() {
        for yaml in [
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
  - min_tokens: 32
    bucket: uncached
policy_classes:
  - name: cached
    policy_family: standard
    cache_bucket: cached
    quantum: 1
"#,
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
policy_classes:
  - name: first
    policy_family: standard
    cache_bucket: cached
    quantum: 1
  - name: second
    policy_family: standard
    cache_bucket: cached
    quantum: 2
"#,
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
policy_classes:
  - name: invalid-family
    policy_family: invalid/family
    cache_bucket: cached
    quantum: 1
"#,
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
policy_classes:
  - name: missing-bucket
    policy_family: standard
    cache_bucket: absent
    quantum: 1
"#,
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
policy_classes:
  - name: partial
    policy_family: standard
    quantum: 1
"#,
            r#"
default_policy_family: priority
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
policy_classes:
  - name: priority
    quantum: 1
  - name: paired
    policy_family: priority
    cache_bucket: cached
    quantum: 1
"#,
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 1
    bucket: cached
policy_classes:
  - name: cached
    policy_family: standard
    cache_bucket: cached
    quantum: 1
"#,
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
  - min_tokens: 32
    bucket: cached
policy_classes:
  - name: cached
    policy_family: standard
    cache_bucket: cached
    quantum: 1
"#,
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
  - min_tokens: 64
    bucket: uncached
  - min_tokens: 32
    bucket: large
policy_classes:
  - name: cached
    policy_family: standard
    cache_bucket: cached
    quantum: 1
"#,
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
policy_classes:
  - name: zero
    policy_family: standard
    cache_bucket: cached
    quantum: 0
"#,
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
policy_classes:
  - name: lcfs
    policy_family: standard
    cache_bucket: cached
    queue_policy: lcfs
    quantum: 1
"#,
        ] {
            assert!(
                RouterPolicyConfig::from_yaml(yaml).is_err(),
                "unexpectedly accepted {yaml}"
            );
        }
    }

    #[test]
    fn documented_sample_exercises_root_model_and_unknown_class_semantics() {
        let config = RouterPolicyConfig::from_yaml(include_str!(
            "../../../../../examples/router/policy-class-queues.yaml"
        ))
        .unwrap();

        let root = config.resolve_profile(None, None, RouterQueuePolicy::Fcfs);
        assert_eq!(root.classes().len(), 5);
        assert_eq!(root.default_class().name, "cached");
        assert_eq!(
            root.class(root.resolve_class_index(Some("latency"), 0))
                .name,
            "latency_cached"
        );
        assert_eq!(
            root.class(root.resolve_class_index(Some("latency"), usize::MAX))
                .name,
            "latency_uncached"
        );
        assert_eq!(
            root.class(root.resolve_class_index(Some("unknown"), 0))
                .name,
            "cached"
        );
        assert_eq!(
            root.class(root.resolve_class_index(None, 3071)).name,
            "cached"
        );
        assert_eq!(
            root.class(root.resolve_class_index(None, 3072)).name,
            "uncached"
        );
        assert_eq!(
            root.class(root.resolve_class_index(None, usize::MAX)).name,
            "uncached"
        );
        assert_eq!(
            root.class(root.resolve_class_index(Some("cached"), usize::MAX))
                .name,
            "uncached",
            "ordinary physical class names must not bypass family and bucket classification"
        );
        assert_eq!(
            root.class(root.resolve_class_index(Some("custom_priority"), usize::MAX))
                .name,
            "custom_priority",
            "explicit classes intentionally bypass cache classification"
        );
        assert_eq!(
            root.default_class()
                .busy_thresholds
                .prefill_busy_threshold_frac,
            Some(16.0)
        );

        let model = config.resolve_profile(
            Some("example/large-model"),
            Some(3.0),
            RouterQueuePolicy::Fcfs,
        );
        assert_eq!(model.classes().len(), 4);
        assert_eq!(model.default_class().name, "latency_cached");
        assert_eq!(
            model
                .class(model.resolve_class_index(Some("unknown"), usize::MAX))
                .name,
            "latency_uncached",
            "unknown policies must use the model's default family and bucket mapping"
        );
        assert_eq!(
            model
                .class(model.resolve_class_index(Some("batch"), 0))
                .name,
            "batch_cached"
        );
        assert!(
            model
                .classes()
                .iter()
                .all(|class| class.name != "custom_priority"),
            "model profiles must completely replace root classes"
        );
    }

    #[test]
    fn prefill_budget_defaults_when_unreported() {
        use dynamo_router_protocol::load::DEFAULT_MAX_BATCHED_TOKENS;
        assert_eq!(
            WorkerCapacity::default().prefill_budget(),
            DEFAULT_MAX_BATCHED_TOKENS
        );
        assert_eq!(WorkerCapacity::new(Some(64), None).prefill_budget(), 64);
        assert_eq!(WorkerCapacity::default().max_num_batched_tokens(), None);
        // Every reported value round-trips; there is no sentinel.
        let max = WorkerCapacity::new(Some(u64::MAX), Some(u64::MAX));
        assert_eq!(max.max_num_batched_tokens(), Some(u64::MAX));
        assert_eq!(max.total_kv_blocks(), Some(u64::MAX));
    }

    #[test]
    fn unobserved_rank_row_is_unavailable_and_idle() {
        let seen = WorkerWithDpRank::new(1, 0);
        let unseen = WorkerWithDpRank::new(1, 1);
        let mut loads = FxHashMap::default();
        loads.insert(
            seen,
            WorkerLoadProjection {
                active_prefill_tokens: 7,
                ..Default::default()
            },
        );
        let unobserved = || WorkerLoadProjection {
            additional_active_blocks: 3,
            ..Default::default()
        };

        let row = WorkerLoadInput::for_worker(&loads, seen, unobserved);
        assert!(row.is_available());
        assert_eq!(row.active_prefill_tokens(), 7);

        let row = WorkerLoadInput::for_worker(&loads, unseen, unobserved);
        assert!(!row.is_available());
        assert_eq!(row.active_prefill_tokens(), 0);
        assert_eq!(row.decode_cost_blocks(), 3.0);
    }

    #[test]
    fn worker_is_full_only_when_every_rank_is_full() {
        let limits = BusyThresholds {
            prefill_busy_threshold: Some(10),
            ..Default::default()
        };
        let config = dynamo_router_protocol::test_utils::SimpleWorkerConfig {
            data_parallel_start_rank: 2,
            data_parallel_size: 2,
            ..Default::default()
        };
        let busy = WorkerLoadProjection {
            active_prefill_tokens: 11,
            ..Default::default()
        };
        let mut loads = FxHashMap::default();

        loads.insert(WorkerWithDpRank::new(5, 2), busy);
        // Rank 3 has no projection and counts as idle.
        assert!(!limits.worker_is_full(5, &config, &loads));

        loads.insert(WorkerWithDpRank::new(5, 3), busy);
        assert!(limits.worker_is_full(5, &config, &loads));

        // A rank outside the worker's range does not stand in for one inside it.
        loads.remove(&WorkerWithDpRank::new(5, 3));
        loads.insert(WorkerWithDpRank::new(5, 0), busy);
        assert!(!limits.worker_is_full(5, &config, &loads));
    }

    #[test]
    fn zero_capacity_is_unreported() {
        let zero = WorkerCapacity::new(Some(0), Some(0));
        assert_eq!(zero.max_num_batched_tokens(), None);
        assert_eq!(zero.total_kv_blocks(), None);
        assert_eq!(zero, WorkerCapacity::default());

        // vLLM under the Ray data-parallel backend registers `total_kv_blocks = 0` as a
        // placeholder; the decode limit skips such a rank instead of calling it full.
        let limits = BusyThresholds {
            decode_busy_threshold_frac: Some(0.5),
            ..Default::default()
        };
        let held = WorkerLoadInput::from_projection(WorkerLoadProjection {
            active_decode_blocks: 1_000,
            ..Default::default()
        });
        assert!(limits.has_room(&held, WorkerCapacity::new(None, Some(0))));
        assert!(!limits.has_room(&held, WorkerCapacity::new(None, Some(1_000))));

        // A zero prefill budget falls back to the default budget instead of being full at once.
        let limits = BusyThresholds {
            prefill_busy_threshold_frac: Some(0.5),
            ..Default::default()
        };
        assert!(limits.has_room(&prefill_load(1), WorkerCapacity::new(Some(0), None)));
    }
}
