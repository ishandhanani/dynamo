// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Accepted names for the `tool_call_parser` runtime-config field beyond the v1
//! registry in `dynamo_parsers`: the `dynamo-parsers-v2` unified families and the
//! Dynamo-specific aliases that pair into them. `dynamo-llm` routes requests with
//! these names; `dynamo-local-model` validates them at model registration.

use std::sync::LazyLock;

/// Map Dynamo tool-parser names onto the corresponding v2 registry family.
pub const V2_FAMILIES: &[&str] = &["qwen3_coder", "deepseek_v4", "deepseek-v4", "deepseekv4"];

/// The `dynamo-parsers-v2` unified family that serves Qwen3.
///
/// `REGISTERED_UNIFIED_FAMILIES` accepts both `qwen3` and the `qwen3_coder` alias for
/// the same XML grammar; `qwen3` is the canonical registry name and the one the
/// conformance corpus uses, so it is what the frontend passes and logs.
pub const QWEN3_UNIFIED_FAMILY: &str = "qwen3";
pub const KIMI_K2_UNIFIED_FAMILY: &str = "kimi_k2";
pub const KIMI_K3_UNIFIED_FAMILY: &str = "kimi_k3";
pub const DEEPSEEK_V41_UNIFIED_FAMILY: &str = "deepseek_v41";
pub const DEEPSEEK_V4_UNIFIED_FAMILY: &str = "deepseek_v4";
pub const GEMMA4_UNIFIED_FAMILY: &str = "gemma4";
pub const GLM47_UNIFIED_FAMILY: &str = "glm47";
pub const MUSE_UNIFIED_FAMILY: &str = "muse_glimmer";

/// Dynamo parser names that resolve to a unified family under a different name.
pub const DYNAMO_FAMILY_ALIASES: &[(&str, &str)] = &[
    ("deepseek-v4", DEEPSEEK_V4_UNIFIED_FAMILY),
    ("deepseekv4", DEEPSEEK_V4_UNIFIED_FAMILY),
    ("muse", MUSE_UNIFIED_FAMILY),
    ("gemma-4", GEMMA4_UNIFIED_FAMILY),
    ("glm45", GLM47_UNIFIED_FAMILY),
    ("kimi_k25", KIMI_K2_UNIFIED_FAMILY),
];

/// Every unified family name plus every Dynamo alias, sorted and deduplicated.
pub static FAMILY_NAMES: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    let mut names = dynamo_parsers_v2::REGISTERED_UNIFIED_FAMILIES.to_vec();
    names.extend(DYNAMO_FAMILY_ALIASES.iter().map(|(alias, _)| *alias));
    names.sort_unstable();
    names.dedup();
    names
});

/// The unified (v2) families and their Dynamo aliases this build accepts.
pub fn unified_family_names() -> &'static [&'static str] {
    &FAMILY_NAMES
}
