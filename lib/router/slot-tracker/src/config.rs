// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The slot-tracker slice of the router configuration.

use dynamo_router_types::config::{KvRouterConfig, RouterPrefillLoadModel};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SlotsConfig {
    pub track_active_blocks: bool,
    pub track_output_blocks: bool,
    pub track_prefill_tokens: bool,
    pub prefill_load_model: RouterPrefillLoadModel,
    pub ttl_secs: f64,
    pub predicted_ttl_secs: Option<f64>,
}

impl From<&KvRouterConfig> for SlotsConfig {
    fn from(config: &KvRouterConfig) -> Self {
        Self {
            track_active_blocks: config.router_track_active_blocks,
            track_output_blocks: config.router_track_output_blocks,
            track_prefill_tokens: config.router_track_prefill_tokens,
            prefill_load_model: config.router_prefill_load_model,
            ttl_secs: config.router_ttl_secs,
            predicted_ttl_secs: config.router_predicted_ttl_secs,
        }
    }
}
