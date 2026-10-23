// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Model Deployment Card
//!
//! The card itself (the wire type) lives in `dynamo_llm_protocol::model_card` and the
//! loading side (discovering a checkout on disk, fetching and staging metadata files,
//! and building tokenizers) in `dynamo_local_model::model_card`; both are re-exported
//! here.

pub use dynamo_llm_protocol::model_card::*;

pub use dynamo_local_model::model_card::loader;
pub use loader::{
    GenerationConfigLoader, ModelDeploymentCardLoader, ModelInfo, ModelInfoTypeLoader,
    PromptFormatterArtifactLoader, TokenizerKindLoader,
};
