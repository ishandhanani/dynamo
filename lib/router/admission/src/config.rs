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
    /// The router-policy document path, when one is configured; the classes are resolved from it.
    pub policy_config: Option<String>,
}

impl From<&KvRouterConfig> for AdmissionConfig {
    fn from(config: &KvRouterConfig) -> Self {
        Self {
            queue_threshold: config.router_queue_threshold,
            queue_policy: config.router_queue_policy,
            policy_config: config.router_policy_config.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
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
