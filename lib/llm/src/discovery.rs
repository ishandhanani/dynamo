// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod image_edits;
mod model;
pub(crate) mod readiness;
pub use model::Model;
pub(crate) use model::{
    GenerateEngineSelection, SystemOneExecutionSelection, SystemOneSelectionError,
};

mod model_manager;
pub(crate) mod router_policy;
pub use model_manager::{ModelManager, ModelManagerError, UNKNOWN_METRIC_MODEL};

mod controller;
pub(crate) mod worker_inventory;

mod allocator;

mod worker_set;
pub use worker_set::WorkerSet;
pub(crate) use worker_set::{CommittedWorkerSetTarget, WorkerSetTarget, WorkerSetTargetId};

mod endpoint_card;
pub use endpoint_card::wait_for_endpoint_model_card;

mod watcher;
pub use watcher::{ModelUpdate, ModelWatcher};
