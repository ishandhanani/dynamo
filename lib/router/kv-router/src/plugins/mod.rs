// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registration and resolution of statically linked router plugins.
//!
//! Plugin authors implement the traits in `dynamo_worker_selection::plugin` or
//! `dynamo_admission::classifier` and register their providers with [`RouterPluginRegistry`]. Signal accessors and callback signatures are unchanged.
//! Scheduling owns execution, eligibility, lifecycle synchronization, and capacity accounting.
//!
//! Legacy imports remain compatibility re-exports until v1.7; new plugins should use this module.

mod registry;

/// The worker-selection plugin API at its published path.
pub mod worker_selection {
    pub use dynamo_worker_selection::plugin::*;
}

/// The request-classifier plugin API at its published path.
pub mod request_classifier {
    pub use dynamo_admission::classifier::*;
}

pub use registry::{
    DYN_ROUTER_DECODE_POLICY, DYN_ROUTER_PREFILL_POLICY, DYN_ROUTER_WORKER_SELECTION_POLICY,
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProvider,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};

use dynamo_admission::classifier::{RequestClassifierFactory, RequestClassifierRegistryError};
use dynamo_worker_selection::plugin::WorkerSelectionPolicyFactory;

/// Configured factories shared across router construction, with fresh instances per router.
#[derive(Clone, Default)]
pub struct RouterPlugins {
    worker_selection: Option<WorkerSelectionPolicyFactory>,
    custom_worker_selection: bool,
    request_classifier: Option<RequestClassifierFactory>,
}

impl RouterPlugins {
    pub fn with_worker_selection(mut self, factory: WorkerSelectionPolicyFactory) -> Self {
        self.worker_selection = Some(factory);
        self.custom_worker_selection = true;
        self
    }

    pub fn with_request_classifier(mut self, factory: RequestClassifierFactory) -> Self {
        self.request_classifier = Some(factory);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.worker_selection.is_none() && self.request_classifier.is_none()
    }

    /// Whether worker selection was explicitly supplied, rather than filled by the host default.
    /// Hosts use this to require typed worker discovery without delaying stock router startup.
    pub fn has_custom_worker_selection(&self) -> bool {
        self.custom_worker_selection
    }

    /// Whether explicit policy or classifier choices require custom-plugin frontend support.
    pub fn has_custom_plugins(&self) -> bool {
        self.custom_worker_selection || self.request_classifier.is_some()
    }

    pub fn worker_selection(&self) -> Option<&WorkerSelectionPolicyFactory> {
        self.worker_selection.as_ref()
    }

    pub fn request_classifier(&self) -> Option<&RequestClassifierFactory> {
        self.request_classifier.as_ref()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RouterPluginRegistryError {
    #[error(transparent)]
    WorkerSelection(#[from] WorkerSelectionPolicyRegistryError),
    #[error(transparent)]
    RequestClassifier(#[from] RequestClassifierRegistryError),
}
