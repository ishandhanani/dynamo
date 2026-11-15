// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Whether a router needs worker KV event sources, derived from its role, its configuration
//! and the worker inputs its selection policy reads.

use std::fmt;

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::WorkerInputs;
use dynamo_router_protocol::worker_type::WorkerType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvEventSourceRequirement {
    NotRequired,
    CacheAwareRouting,
    ConditionalDisaggDecodeCache,
    Unknown,
}

impl KvEventSourceRequirement {
    pub(crate) fn derive(
        worker_role: Option<WorkerType>,
        config: &KvRouterConfig,
        inputs: WorkerInputs,
    ) -> Self {
        if !inputs.reads_kv_index() || config.use_remote_indexer || !config.use_kv_events {
            return Self::NotRequired;
        }

        match worker_role {
            Some(WorkerType::Decode) if config.conditional_disagg_enabled => {
                Self::ConditionalDisaggDecodeCache
            }
            Some(_) => Self::CacheAwareRouting,
            None => Self::Unknown,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::NotRequired => "not_required",
            Self::CacheAwareRouting => "cache_aware_routing",
            Self::ConditionalDisaggDecodeCache => "conditional_disagg_decode_cache",
            Self::Unknown => "unknown",
        }
    }

    pub(crate) fn requires_source(self) -> bool {
        matches!(
            self,
            Self::CacheAwareRouting | Self::ConditionalDisaggDecodeCache
        )
    }

    pub(crate) fn should_subscribe(self) -> bool {
        self != Self::NotRequired
    }
}

impl fmt::Display for KvEventSourceRequirement {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}
