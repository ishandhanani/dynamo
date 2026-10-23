// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA routing and allocation for the frontend, plus re-exports of the LoRA
//! downloading and caching infrastructure that lives in `dynamo-local-model`.

pub mod config;
pub mod controller;
pub mod filter;
pub mod load_estimator;
pub mod predictor;
pub mod routing;
pub mod state_tracker;

pub use config::{LoraAllocationConfig, McfConfig};
pub use controller::LoraController;
#[cfg(feature = "lora-s3")]
pub use dynamo_local_model::lora::S3LoRASource;
pub use dynamo_local_model::lora::{
    HuggingFaceLoRASource, LORA_REQUIRES_REGISTRATION, LoRASource, LocalLoRASource,
    lora_serving_enabled,
};
pub use dynamo_local_model::lora::{LoRACache, LoRADownloader};
pub use filter::LoraFilter;
pub use load_estimator::{LoadEstimator, LoadEstimatorConfig};
pub use routing::{
    AllocationAlgorithmType, LoraAllocator, LoraReplicaConfig, LoraRoutingTable,
    McfPlacementResult, McfPlacementSolver, McfSolveParams, RendezvousHasher,
    create_lora_allocator,
};
pub use state_tracker::LoraStateTracker;
