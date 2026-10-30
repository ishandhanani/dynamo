// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The admission slice of the router configuration: queue policy and limits, plus the policy
//! classes and classifier settings (the profiles in `policy_config`).

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

pub use crate::policy_config as policy;
use crate::policy_config::PolicyProfiles;

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct AdmissionConfig {
    pub queue_threshold: Option<f64>,
    pub queue_policy: RouterQueuePolicy,
    /// The policy file's queue profiles, when the router was given one.
    #[serde(skip)]
    pub profiles: Option<PolicyProfiles>,
}

impl AdmissionConfig {
    /// The queue profile for `model_name`: the policy file's, or one synthesized from the
    /// threshold and queue policy.
    pub fn policy_profile(&self, model_name: Option<&str>) -> policy::PolicyProfile {
        match &self.profiles {
            Some(profiles) => {
                profiles.resolve_profile(model_name, self.queue_threshold, self.queue_policy)
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
        self.profiles
            .as_ref()
            .is_some_and(PolicyProfiles::has_routing_profiles)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ClassifierConfig {
    /// The classifier instance the policy document selects, if any.
    pub request_classifier: Option<crate::classifier::RequestClassifierConfig>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RouterQueuePolicy {
    #[default]
    Fcfs,
    Lcfs,
    Wspt,
}

impl fmt::Display for RouterQueuePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fcfs => f.write_str("fcfs"),
            Self::Lcfs => f.write_str("lcfs"),
            Self::Wspt => f.write_str("wspt"),
        }
    }
}

impl FromStr for RouterQueuePolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "fcfs" => Ok(Self::Fcfs),
            "lcfs" => Ok(Self::Lcfs),
            "wspt" => Ok(Self::Wspt),
            _ => Err(format!(
                "unknown queue policy: {s:?}, expected 'fcfs', 'lcfs', or 'wspt'"
            )),
        }
    }
}
