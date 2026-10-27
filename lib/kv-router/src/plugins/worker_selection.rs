// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-selection plugin API, owned by `dynamo-worker-selection`; this module keeps the old paths.

pub use dynamo_router_types::selection_inputs as inputs;
pub use dynamo_worker_selection::plugin::*;

pub use super::registry::{
    WorkerSelectionPolicyParameters, WorkerSelectionPolicyProvider,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistryError,
};
