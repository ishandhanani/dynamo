// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Queue policy profiles: the classes a request can be admitted into, how a request is
//! classified into them, and the raw YAML shapes `dynamo-kv-router` reads them from.

use std::collections::{HashMap, HashSet};

use serde::Deserialize;
use thiserror::Error;

use crate::config::RouterQueuePolicy;

#[doc(hidden)]
pub const SYNTHETIC_POLICY_CLASS: &str = "default";

#[derive(Debug, Error)]
pub enum RouterPolicyConfigError {
    #[error("failed to read router policy config {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse router policy config {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_yaml::Error,
    },
    #[error("invalid router policy config: {0}")]
    Validation(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyClassConfig {
    pub name: String,
    pub queue_policy: RouterQueuePolicy,
    pub quantum: usize,
    /// The limits this class waits on.
    pub busy_thresholds: BusyThresholds,
    pub request_queue_limit_per_worker: Option<usize>,
    pub raw_isl_token_queue_limit_per_worker: Option<usize>,
    pub cached_token_queue_limit_per_worker: Option<usize>,
}

impl PolicyClassConfig {
    pub fn queueing_enabled(&self) -> bool {
        self.busy_thresholds.is_configured()
    }

    /// Any zero per-worker limit means `queue_rejection` rejects every queued
    /// entry (`current >= limit` holds at zero usage), so the class can never
    /// hold a queued request and must keep the direct admission path.
    pub fn never_queues(&self) -> bool {
        self.request_queue_limit_per_worker == Some(0)
            || self.raw_isl_token_queue_limit_per_worker == Some(0)
            || self.cached_token_queue_limit_per_worker == Some(0)
    }
}

pub use dynamo_router_types::busy::{BusyThresholds, FullRankCounts, RejectThresholdsProvider};

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyProfile {
    classes: Vec<PolicyClassConfig>,
    classifier: PolicyClassifier,
}

#[derive(Debug, Clone, PartialEq)]
enum PolicyClassifier {
    SyntheticSingle { class_index: usize },
    FamilyBucket(FamilyBucketClassifier),
}

#[derive(Debug, Clone, PartialEq)]
struct FamilyBucketClassifier {
    default_family_index: usize,
    family_indices: HashMap<String, usize>,
    explicit_class_indices: HashMap<String, usize>,
    buckets: Vec<UncachedIslBucket>,
    class_by_family_bucket: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UncachedIslBucket {
    min_tokens: usize,
}

impl FamilyBucketClassifier {
    /// Returns only selections that do not require a cache snapshot.
    fn direct_class_index(&self, requested: Option<&str>) -> Option<usize> {
        requested.and_then(|name| self.explicit_class_indices.get(name).copied())
    }

    /// Combines a recognized family (or the default) with the observed bucket.
    fn class_index(&self, requested: Option<&str>, uncached_tokens: usize) -> usize {
        if let Some(class_index) = self.direct_class_index(requested) {
            return class_index;
        }

        let family_index = requested
            .and_then(|name| self.family_indices.get(name).copied())
            .unwrap_or(self.default_family_index);
        self.class_index_for_family(family_index, uncached_tokens)
    }

    fn strict_class_index(&self, requested: &str, uncached_tokens: usize) -> Option<usize> {
        self.explicit_class_indices
            .get(requested)
            .copied()
            .or_else(|| {
                self.family_indices
                    .get(requested)
                    .map(|family_index| self.class_index_for_family(*family_index, uncached_tokens))
            })
    }

    fn class_index_for_family(&self, family_index: usize, uncached_tokens: usize) -> usize {
        let bucket_index = self
            .buckets
            .partition_point(|bucket| bucket.min_tokens <= uncached_tokens)
            .saturating_sub(1);
        self.class_by_family_bucket[family_index * self.buckets.len() + bucket_index]
    }
}

impl PolicyProfile {
    pub fn synthetic(
        router_queue_threshold: Option<f64>,
        router_queue_policy: RouterQueuePolicy,
    ) -> Self {
        let class = PolicyClassConfig {
            name: SYNTHETIC_POLICY_CLASS.to_string(),
            queue_policy: router_queue_policy,
            quantum: 1,
            busy_thresholds: BusyThresholds {
                prefill_busy_threshold: None,
                prefill_busy_threshold_frac: router_queue_threshold,
                decode_busy_threshold_frac: None,
            },
            request_queue_limit_per_worker: None,
            raw_isl_token_queue_limit_per_worker: None,
            cached_token_queue_limit_per_worker: None,
        };
        Self {
            classes: vec![class],
            classifier: PolicyClassifier::SyntheticSingle { class_index: 0 },
        }
    }

    pub fn classes(&self) -> &[PolicyClassConfig] {
        &self.classes
    }

    pub fn default_class(&self) -> &PolicyClassConfig {
        &self.classes[self.resolve_class_index(None, 0)]
    }

    /// Resolves synthetic and explicit requests without observing cache state.
    pub fn direct_class_index(&self, requested: Option<&str>) -> Option<usize> {
        match &self.classifier {
            PolicyClassifier::SyntheticSingle { class_index } => Some(*class_index),
            PolicyClassifier::FamilyBucket(classifier) => classifier.direct_class_index(requested),
        }
    }

    /// Resolves a requested family and exact uncached ISL to a physical queue.
    pub fn resolve_class_index(&self, requested: Option<&str>, uncached_tokens: usize) -> usize {
        match &self.classifier {
            PolicyClassifier::SyntheticSingle { class_index } => *class_index,
            PolicyClassifier::FamilyBucket(classifier) => {
                // TODO: Add bounded observability for unknown requested policy values.
                classifier.class_index(requested, uncached_tokens)
            }
        }
    }

    pub fn class(&self, index: usize) -> &PolicyClassConfig {
        &self.classes[index]
    }

    #[doc(hidden)]
    pub fn resolve_class_index_strict(
        &self,
        requested: &str,
        uncached_tokens: usize,
    ) -> Option<usize> {
        match &self.classifier {
            PolicyClassifier::SyntheticSingle { class_index } => {
                (self.classes[*class_index].name == requested).then_some(*class_index)
            }
            PolicyClassifier::FamilyBucket(classifier) => {
                classifier.strict_class_index(requested, uncached_tokens)
            }
        }
    }
}

/// The YAML shape of one queue profile; `dynamo-kv-router` reads the policy file into it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[doc(hidden)]
pub struct RawPolicyProfile {
    pub default_policy_family: String,
    pub policy_classes: Vec<RawPolicyClassConfig>,
    pub uncached_isl_buckets: Vec<RawUncachedIslBucket>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[doc(hidden)]
pub struct RawUncachedIslBucket {
    min_tokens: usize,
    bucket: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[doc(hidden)]
pub struct RawPolicyClassConfig {
    name: String,
    policy_family: Option<String>,
    cache_bucket: Option<String>,
    #[serde(default)]
    queue_policy: RouterQueuePolicy,
    quantum: usize,
    prefill_busy_threshold: Option<usize>,
    prefill_busy_threshold_frac: Option<f64>,
    decode_busy_threshold_frac: Option<f64>,
    request_queue_limit_per_worker: Option<usize>,
    raw_isl_token_queue_limit_per_worker: Option<usize>,
    cached_token_queue_limit_per_worker: Option<usize>,
}

#[doc(hidden)]
pub fn resolve_profile(
    profile: RawPolicyProfile,
    location: &str,
) -> Result<PolicyProfile, RouterPolicyConfigError> {
    validate_identifier(&profile.default_policy_family, "policy family", location)?;
    if profile.policy_classes.is_empty() {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy_classes must not be empty"
        )));
    }

    let resolved_buckets = resolve_uncached_isl_buckets(profile.uncached_isl_buckets, location)?;
    let mut names = HashSet::with_capacity(profile.policy_classes.len());
    let mut classes = Vec::with_capacity(profile.policy_classes.len());
    let mut bindings = Vec::with_capacity(profile.policy_classes.len());
    for raw in profile.policy_classes {
        let resolved = resolve_policy_class(raw, &resolved_buckets.indices, location)?;
        if !names.insert(resolved.config.name.clone()) {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} contains duplicate policy class {:?}",
                resolved.config.name
            )));
        }
        classes.push(resolved.config);
        bindings.push(resolved.binding);
    }

    let mut family_names = Vec::new();
    let mut family_indices = HashMap::new();
    for binding in &bindings {
        let ClassBinding::FamilyBucket { policy_family, .. } = binding else {
            continue;
        };
        if !family_indices.contains_key(policy_family) {
            let family_index = family_names.len();
            family_names.push(policy_family.clone());
            family_indices.insert(policy_family.clone(), family_index);
        }
    }

    let Some(default_family_index) = family_indices.get(&profile.default_policy_family).copied()
    else {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} default_policy_family {:?} does not name a configured family",
            profile.default_policy_family
        )));
    };

    let mut explicit_class_indices = HashMap::new();
    let mut class_by_family_bucket = vec![
        None;
        family_names
            .len()
            .saturating_mul(resolved_buckets.buckets.len())
    ];
    for (class_index, binding) in bindings.into_iter().enumerate() {
        match binding {
            ClassBinding::Explicit => {
                let class_name = &classes[class_index].name;
                if family_indices.contains_key(class_name) {
                    return Err(RouterPolicyConfigError::Validation(format!(
                        "{location} explicit policy class {class_name:?} collides with a policy family"
                    )));
                }
                explicit_class_indices.insert(class_name.clone(), class_index);
            }
            ClassBinding::FamilyBucket {
                policy_family,
                bucket_index,
            } => {
                let family_index = family_indices[&policy_family];
                let table_index = family_index * resolved_buckets.buckets.len() + bucket_index;
                if class_by_family_bucket[table_index]
                    .replace(class_index)
                    .is_some()
                {
                    return Err(RouterPolicyConfigError::Validation(format!(
                        "{location} contains duplicate policy classes for family {policy_family:?} and bucket {:?}",
                        resolved_buckets.names[bucket_index]
                    )));
                }
            }
        }
    }

    for (family_index, family_name) in family_names.iter().enumerate() {
        for (bucket_index, bucket_name) in resolved_buckets.names.iter().enumerate() {
            if class_by_family_bucket[family_index * resolved_buckets.buckets.len() + bucket_index]
                .is_none()
            {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "{location} is missing a policy class for family {family_name:?} and bucket {bucket_name:?}"
                )));
            }
        }
    }

    Ok(PolicyProfile {
        classes,
        classifier: PolicyClassifier::FamilyBucket(FamilyBucketClassifier {
            default_family_index,
            family_indices,
            explicit_class_indices,
            buckets: resolved_buckets.buckets,
            class_by_family_bucket: class_by_family_bucket
                .into_iter()
                .map(|class_index| class_index.expect("validated complete policy matrix"))
                .collect(),
        }),
    })
}

struct ResolvedPolicyClass {
    config: PolicyClassConfig,
    binding: ClassBinding,
}

enum ClassBinding {
    Explicit,
    FamilyBucket {
        policy_family: String,
        bucket_index: usize,
    },
}

fn resolve_policy_class(
    raw: RawPolicyClassConfig,
    bucket_indices: &HashMap<String, usize>,
    location: &str,
) -> Result<ResolvedPolicyClass, RouterPolicyConfigError> {
    validate_identifier(&raw.name, "policy class", location)?;
    if raw.quantum == 0 {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy class {:?} quantum must be greater than zero",
            raw.name
        )));
    }
    if raw.queue_policy == RouterQueuePolicy::Lcfs {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy class {:?} queue_policy must be fcfs or wspt",
            raw.name
        )));
    }
    for (field, value) in [
        (
            "prefill_busy_threshold_frac",
            raw.prefill_busy_threshold_frac,
        ),
        ("decode_busy_threshold_frac", raw.decode_busy_threshold_frac),
    ] {
        if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} policy class {:?} {field} must be finite and non-negative",
                raw.name
            )));
        }
    }

    let binding = match (raw.policy_family.as_deref(), raw.cache_bucket.as_deref()) {
        (None, None) => ClassBinding::Explicit,
        (Some(policy_family), Some(cache_bucket)) => {
            validate_identifier(policy_family, "policy family", location)?;
            validate_identifier(cache_bucket, "cache bucket", location)?;
            let Some(bucket_index) = bucket_indices.get(cache_bucket).copied() else {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "{location} policy class {:?} references unknown cache bucket {:?}",
                    raw.name, cache_bucket
                )));
            };
            ClassBinding::FamilyBucket {
                policy_family: policy_family.to_string(),
                bucket_index,
            }
        }
        _ => {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} policy class {:?} must specify both policy_family and cache_bucket or neither for an explicit class",
                raw.name
            )));
        }
    };
    if raw.prefill_busy_threshold.is_some() {
        tracing::warn!(
            policy_class = %raw.name,
            "prefill_busy_threshold is deprecated and will be removed in v1.7; use prefill_busy_threshold_frac"
        );
    }
    Ok(ResolvedPolicyClass {
        config: PolicyClassConfig {
            name: raw.name,
            queue_policy: raw.queue_policy,
            quantum: raw.quantum,
            busy_thresholds: BusyThresholds {
                prefill_busy_threshold: raw.prefill_busy_threshold,
                prefill_busy_threshold_frac: raw.prefill_busy_threshold_frac,
                decode_busy_threshold_frac: raw.decode_busy_threshold_frac,
            },
            request_queue_limit_per_worker: raw.request_queue_limit_per_worker,
            raw_isl_token_queue_limit_per_worker: raw.raw_isl_token_queue_limit_per_worker,
            cached_token_queue_limit_per_worker: raw.cached_token_queue_limit_per_worker,
        },
        binding,
    })
}

struct ResolvedBuckets {
    buckets: Vec<UncachedIslBucket>,
    names: Vec<String>,
    indices: HashMap<String, usize>,
}

fn resolve_uncached_isl_buckets(
    raw_buckets: Vec<RawUncachedIslBucket>,
    location: &str,
) -> Result<ResolvedBuckets, RouterPolicyConfigError> {
    if raw_buckets.is_empty() {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} uncached_isl_buckets must not be empty"
        )));
    }
    if raw_buckets[0].min_tokens != 0 {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} uncached_isl_buckets must start at min_tokens 0"
        )));
    }
    for window in raw_buckets.windows(2) {
        if window[1].min_tokens <= window[0].min_tokens {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} uncached_isl_buckets min_tokens must be strictly increasing"
            )));
        }
    }

    let mut bucket_names = Vec::with_capacity(raw_buckets.len());
    let mut bucket_indices = HashMap::with_capacity(raw_buckets.len());
    let mut buckets = Vec::with_capacity(raw_buckets.len());
    for raw in raw_buckets {
        validate_identifier(&raw.bucket, "cache bucket", location)?;
        let bucket_index = bucket_names.len();
        if bucket_indices
            .insert(raw.bucket.clone(), bucket_index)
            .is_some()
        {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} contains duplicate cache bucket {:?}",
                raw.bucket
            )));
        }
        bucket_names.push(raw.bucket);
        buckets.push(UncachedIslBucket {
            min_tokens: raw.min_tokens,
        });
    }

    Ok(ResolvedBuckets {
        buckets,
        names: bucket_names,
        indices: bucket_indices,
    })
}

#[doc(hidden)]
pub fn validate_identifier(
    name: &str,
    kind: &str,
    location: &str,
) -> Result<(), RouterPolicyConfigError> {
    if !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
    {
        return Ok(());
    }

    Err(RouterPolicyConfigError::Validation(format!(
        "{location} {kind} name {name:?} must match [A-Za-z0-9_.-]+"
    )))
}

/// The queue profiles the policy file defines: a root profile and per-model overrides. The
/// admission slice carries them so a model's profile is one lookup, not a file read.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PolicyProfiles {
    root: Option<PolicyProfile>,
    models: HashMap<String, PolicyProfile>,
}

impl PolicyProfiles {
    pub fn new(root: Option<PolicyProfile>, models: HashMap<String, PolicyProfile>) -> Self {
        Self { root, models }
    }

    /// The profile for `model_name`: its own, else the root, else one synthesized from the
    /// threshold and queue policy. Model profiles replace the root wholesale.
    pub fn resolve_profile(
        &self,
        model_name: Option<&str>,
        fallback_threshold: Option<f64>,
        fallback_policy: RouterQueuePolicy,
    ) -> PolicyProfile {
        model_name
            .and_then(|name| self.models.get(name))
            .or(self.root.as_ref())
            .cloned()
            .unwrap_or_else(|| PolicyProfile::synthetic(fallback_threshold, fallback_policy))
    }

    /// Whether any profile is defined.
    pub fn has_routing_profiles(&self) -> bool {
        self.root.is_some() || !self.models.is_empty()
    }
}

impl PolicyProfile {
    /// Parse one profile from YAML (`default_policy_family`, `policy_classes`,
    /// `uncached_isl_buckets`), for tests and hosts that bypass the policy file.
    #[doc(hidden)]
    pub fn from_yaml(contents: &str) -> Result<Self, RouterPolicyConfigError> {
        let raw: RawPolicyProfile =
            serde_yaml::from_str(contents).map_err(|source| RouterPolicyConfigError::Parse {
                path: "<inline>".to_string(),
                source,
            })?;
        resolve_profile(raw, "root")
    }
}
