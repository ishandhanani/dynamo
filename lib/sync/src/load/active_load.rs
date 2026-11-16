// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The active-load message workers publish on the event plane. Its publisher
//! (`kv_publisher::worker_metrics`) is in `dynamo-backend-common` and the Pool Relay
//! (`pool_relay`) in `dynamo-llm`; the other readers (`load::metrics_subscriber`,
//! `load::worker_monitor`) are in this crate. No router crate names it.

use dynamo_router_protocol::worker::{DpRank, WorkerId};
use serde::{Deserialize, Serialize};

/// Active load metrics for a worker, used for overload detection.
///
/// Published by workers (with `kv_used_blocks`) and by the scheduler (with
/// `active_decode_blocks` and `active_prefill_tokens`).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ActiveLoad {
    pub worker_id: WorkerId,
    #[serde(default)]
    pub dp_rank: DpRank,
    /// Scheduler-reported decode block load.
    pub active_decode_blocks: Option<u64>,
    /// Number of active prefill tokens (from scheduler's view).
    pub active_prefill_tokens: Option<u64>,
    /// Total KV blocks currently in use on the worker.
    ///
    /// This is published by workers only and is the authoritative signal for
    /// backend KV occupancy used by overload detection.
    pub kv_used_blocks: Option<u64>,
}
