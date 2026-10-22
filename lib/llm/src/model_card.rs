// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Model Deployment Card
//!
//! The card itself (the wire type) lives in `dynamo_llm_protocol::model_card` and is
//! re-exported here. This module adds the loading side: discovering a checkout on
//! disk, fetching and staging metadata files, and building tokenizers.

pub use dynamo_llm_protocol::model_card::*;

pub mod loader;

pub(crate) use loader::is_weight_file;
pub use loader::{
    GenerationConfigLoader, ModelDeploymentCardLoader, ModelInfo, ModelInfoTypeLoader,
    PromptFormatterArtifactLoader, TokenizerKindLoader,
};
