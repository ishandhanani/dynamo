// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request routing modes. The mode is configuration shared by frontends and
//! workers (a worker may publish one on its model deployment card), so the enum
//! lives here; the routers that implement each mode live in `dynamo-runtime`.

use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouterMode {
    #[default]
    RoundRobin,
    Random,
    PowerOfTwoChoices,
    KV,
    Direct,
    LeastLoaded,
    /// Device-aware weighted routing for heterogeneous workers.
    DeviceAwareWeighted,
}

impl RouterMode {
    pub const fn telemetry_label(self) -> &'static str {
        match self {
            Self::RoundRobin => "round-robin",
            Self::Random => "random",
            Self::PowerOfTwoChoices => "power-of-two-choices",
            Self::KV => "kv",
            Self::Direct => "direct",
            Self::LeastLoaded => "least-loaded",
            Self::DeviceAwareWeighted => "device-aware-weighted",
        }
    }

    pub fn is_kv_routing(&self) -> bool {
        *self == RouterMode::KV
    }

    pub fn is_direct_routing(&self) -> bool {
        *self == RouterMode::Direct
    }

    /// Whether this mode admits requests against host-owned occupancy counters.
    pub const fn requires_occupancy(self) -> bool {
        matches!(
            self,
            Self::PowerOfTwoChoices | Self::LeastLoaded | Self::DeviceAwareWeighted
        )
    }
}
