// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The slot-tracker slice of the router configuration.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SlotsConfig {
    pub track_active_blocks: bool,
    pub track_output_blocks: bool,
    pub track_prefill_tokens: bool,
    pub prefill_load_model: RouterPrefillLoadModel,
    pub ttl_secs: f64,
    pub predicted_ttl_secs: Option<f64>,
}

impl Default for SlotsConfig {
    /// The router's defaults; `dynamo-kv-router` asserts the flat defaults map onto these.
    fn default() -> Self {
        Self {
            track_active_blocks: true,
            track_output_blocks: false,
            track_prefill_tokens: true,
            prefill_load_model: RouterPrefillLoadModel::default(),
            ttl_secs: 120.0,
            predicted_ttl_secs: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RouterPrefillLoadModel {
    #[default]
    None,
    // v1.5 frontends only recognize "aic" in worker cards.
    // TODO(v1.8): write "ais" with alias="aic" when v1.5 leaves N-2.
    // TODO(v1.10): remove the wire "aic" alias when v1.7 writers leave N-2,
    // provided v1.8 and later consistently emit "ais".
    #[serde(rename = "aic", alias = "ais")]
    Ais,
}

impl fmt::Display for RouterPrefillLoadModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::Ais => f.write_str("ais"),
        }
    }
}

impl FromStr for RouterPrefillLoadModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(Self::None),
            "ais" | "aic" => Ok(Self::Ais),
            _ => Err(format!(
                "unknown prefill load model: {s:?}, expected 'none' or 'ais'"
            )),
        }
    }
}

impl RouterPrefillLoadModel {
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::None)
    }
}
