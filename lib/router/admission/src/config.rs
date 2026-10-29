// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The admission slice of the router configuration: queue policy and limits, plus the policy
//! classes and classifier settings (re-exported from the types crate).

use dynamo_router_types::config::{KvRouterConfig, RouterQueuePolicy};
pub use dynamo_router_types::config::{policy, request_classifier};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AdmissionConfig {
    pub queue_threshold: Option<f64>,
    pub queue_policy: RouterQueuePolicy,
    /// The loaded policy file, when the router was given one.
    #[serde(skip)]
    pub policy_config: Option<policy::RouterPolicyConfig>,
}

impl AdmissionConfig {
    /// Read the admission slice, loading the policy file the aggregate names.
    pub fn try_from_kv_router_config(
        config: &KvRouterConfig,
    ) -> Result<Self, policy::RouterPolicyConfigError> {
        Ok(Self {
            queue_threshold: config.router_queue_threshold,
            queue_policy: config.router_queue_policy,
            policy_config: config
                .router_policy_config
                .as_deref()
                .map(policy::RouterPolicyConfig::from_path)
                .transpose()?,
        })
    }

    /// The queue profile for `model_name`: the policy file's, or one synthesized from the
    /// threshold and queue policy.
    pub fn policy_profile(&self, model_name: Option<&str>) -> policy::PolicyProfile {
        match &self.policy_config {
            Some(policy_config) => {
                policy_config.resolve_profile(model_name, self.queue_threshold, self.queue_policy)
            }
            None => policy::PolicyProfile::synthetic(self.queue_threshold, self.queue_policy),
        }
    }

    /// Whether any class of `model_name`'s profile queues.
    pub fn queueing_enabled(&self, model_name: Option<&str>) -> bool {
        self.policy_profile(model_name)
            .classes()
            .iter()
            .any(policy::PolicyClassConfig::queueing_enabled)
    }

    /// Whether the policy file defines queue profiles (not only plugin selections).
    pub fn has_routing_profiles(&self) -> bool {
        self.policy_config
            .as_ref()
            .is_some_and(policy::RouterPolicyConfig::has_routing_profiles)
    }
}

impl Default for AdmissionConfig {
    /// The router's defaults; the aggregate's `Default` is the single source until it leaves
    /// the shared crate.
    fn default() -> Self {
        Self::try_from_kv_router_config(&KvRouterConfig::default())
            .expect("the default router configuration names no policy file")
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ClassifierConfig {
    /// The classifier instance the policy document selects, if any.
    pub request_classifier: Option<request_classifier::RequestClassifierConfig>,
}

impl ClassifierConfig {
    /// Resolve the classifier from the flat config; reading the policy document can fail.
    pub fn try_from_kv_router_config(config: &KvRouterConfig) -> Result<Self, String> {
        Ok(Self {
            request_classifier: config
                .request_classifier_config()
                .map_err(|e| e.to_string())?
                .cloned(),
        })
    }
}
