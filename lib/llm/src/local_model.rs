// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The local model loading stack lives in `dynamo-local-model`; every item is
//! re-exported here at its historical path.

pub use dynamo_local_model::{
    DEFAULT_HTTP_PORT, ENV_SELF_HOST_METADATA, LocalModel, LocalModelBuilder, derive_lora_suffix,
    register_model_card, runtime_config, supported_tool_call_parsers, update_model_taints,
    validate_runtime_config,
};
