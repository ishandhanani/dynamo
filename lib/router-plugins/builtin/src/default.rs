// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The default cost-function policy lives in `dynamo-worker-selection`; this crate registers its
//! named provider so router-policy YAML can select `dynamo-default-cost-fn`.

use std::sync::Arc;

use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistryError,
};
use dynamo_router_types::config::KvRouterConfig;
use dynamo_worker_selection::builtin::{DefaultCostFnParameters, policy_for_role};
pub use dynamo_worker_selection::builtin::{
    DefaultWorkerSelector, DefaultWorkerSelectorControl, default_factory, default_policy,
    startup_policy_parameters,
};

pub(super) fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(
        "dynamo-default-cost-fn",
        Arc::new(|parameters| {
            let parameters: DefaultCostFnParameters = parameters.deserialize()?;
            parameters
                .validate()
                .map_err(WorkerSelectionPolicyProviderError::new)?;
            Ok(Arc::new(
                move |config: &KvRouterConfig, role, _partition| {
                    policy_for_role(config.clone(), role, parameters.resolve(config))
                },
            ))
        }),
    )
}
