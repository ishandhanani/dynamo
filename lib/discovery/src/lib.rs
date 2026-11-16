// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo Discovery
//!
//! The frontend's model catalog: [`ModelWatcher`] admits workers into WorkerSets and builds
//! each one's routed pipeline, [`ModelManager`] holds the resulting models and their engines.
//! It builds the routers of `dynamo-router-host` and never depends on `dynamo-llm`.

pub mod image_edits;
pub mod lora;
mod model;
pub mod readiness;
pub use model::Model;
pub use model::{GenerateEngineSelection, SystemOneExecutionSelection, SystemOneSelectionError};

mod model_manager;
pub mod router_policy;
pub use model_manager::{ModelManager, ModelManagerError, UNKNOWN_METRIC_MODEL};

mod controller;
pub mod worker_inventory;

mod allocator;

mod worker_set;
pub use worker_set::WorkerSet;

mod endpoint_card;
pub use endpoint_card::wait_for_endpoint_model_card;

mod pipeline;
pub use pipeline::{
    ChatEngineFactoryCallback, PrefillRoutedEngine, PreprocessedRouting, build_preprocessed_routing,
};

mod watcher;
pub use watcher::{ModelUpdate, ModelWatcher};

pub mod engines;
mod scheduler_load;
pub mod shared_cache;

#[cfg(test)]
#[path = "../../runtime/src/test_utils.rs"]
mod test_utils;
