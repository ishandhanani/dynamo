// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-rank capacity and active-load vocabulary: the budget a rank advertises in its runtime
//! config and the load row the slot tracker projects for one selection. The busy checks, the
//! admission classifier, and worker selection all read the same row for the same rank.

use crate::protocols::{WorkerConfigLike, WorkerLoadProjection, WorkerWithDpRank};
use rustc_hash::FxHashMap;

/// Active-load values for one worker.
#[derive(Clone, Copy, Default)]
pub struct WorkerLoadInput {
    #[doc(hidden)]
    pub available: bool,
    #[doc(hidden)]
    pub active_prefill_tokens: usize,
    #[doc(hidden)]
    pub active_decode_blocks: usize,
    #[doc(hidden)]
    pub decode_cost_blocks: f64,
    #[doc(hidden)]
    pub active_requests: usize,
}

// Every policy copies these rows per candidate per selection. `active_decode_blocks` took the
// row from 32 to 40 bytes; the selection benches put that within noise at 10k workers, while
// narrowing the counters to `u32` cost 2-5% in per-row saturating conversions. Keep the sizes
// from drifting further without a measurement.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<WorkerLoadInput>() == 40);

/// Prefill budget assumed for a rank that does not report `max_num_batched_tokens`. Large
/// enough that a fractional prefill threshold never fires on an unreported budget.
pub const DEFAULT_MAX_BATCHED_TOKENS: u64 = 10_000_000;

/// Advertised capacity of one worker rank, from its runtime config. It changes only when
/// the worker registers again, so hosts serve it from cached discovery state, not per request.
/// Components read it through worker selection's `WorkerInfo::capacity`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorkerCapacity {
    max_num_batched_tokens: Option<u64>,
    total_kv_blocks: Option<u64>,
}

impl WorkerCapacity {
    /// A reported zero means unreported. vLLM registers `total_kv_blocks = 0` as a placeholder
    /// under the Ray data-parallel backend, and a zero budget would mark every rank full.
    pub fn new(max_num_batched_tokens: Option<u64>, total_kv_blocks: Option<u64>) -> Self {
        Self {
            max_num_batched_tokens: max_num_batched_tokens.filter(|&value| value > 0),
            total_kv_blocks: total_kv_blocks.filter(|&value| value > 0),
        }
    }

    /// Read the capacity a worker advertises in its runtime config.
    pub fn of<C: WorkerConfigLike>(config: &C) -> Self {
        Self::new(config.max_num_batched_tokens(), config.total_kv_blocks())
    }

    /// Prefill tokens the rank batches per engine step, when reported.
    pub fn max_num_batched_tokens(self) -> Option<u64> {
        self.max_num_batched_tokens
    }

    /// Total KV blocks the worker advertises for this rank, when reported: capacity, not free
    /// blocks. Some adapters derive it from aggregate data (vLLM data parallelism publishes
    /// the process total divided by the rank count), so treat it as approximate.
    pub fn total_kv_blocks(self) -> Option<u64> {
        self.total_kv_blocks
    }

    /// The budget fractional prefill thresholds are taken of: `max_num_batched_tokens`, or
    /// [`DEFAULT_MAX_BATCHED_TOKENS`] when the rank does not report it.
    #[doc(hidden)]
    pub fn prefill_budget(self) -> u64 {
        self.max_num_batched_tokens
            .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS)
    }
}

impl WorkerLoadInput {
    /// Convert an observed load projection into a `LOAD` row.
    #[doc(hidden)]
    pub fn from_projection(load: WorkerLoadProjection) -> Self {
        Self {
            available: true,
            active_prefill_tokens: load.active_prefill_tokens,
            active_decode_blocks: load.active_decode_blocks,
            decode_cost_blocks: load.potential_decode_blocks() as f64,
            active_requests: load.active_requests,
        }
    }

    /// The `LOAD` row for `worker`: its projection in `loads` when the host observed one,
    /// otherwise `unobserved()` flagged unavailable. Worker selection, the policy-class
    /// queue, and offline replay all build rows here, so they read the same row for the
    /// same worker by construction.
    #[doc(hidden)]
    pub fn for_worker(
        loads: &FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
        worker: WorkerWithDpRank,
        unobserved: impl FnOnce() -> WorkerLoadProjection,
    ) -> Self {
        match loads.get(&worker) {
            Some(&load) => Self::from_projection(load),
            None => Self {
                available: false,
                ..Self::from_projection(unobserved())
            },
        }
    }
}

impl WorkerLoadInput {
    /// Whether the host supplied a load projection for this worker in this selection.
    /// False distinguishes a missing observation from an observed idle worker.
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Return the tokens active in this worker's prefill stage.
    pub fn active_prefill_tokens(&self) -> usize {
        self.active_prefill_tokens
    }

    /// Return the KV blocks this worker's active requests hold. Unlike
    /// [`decode_cost_blocks`](Self::decode_cost_blocks), this excludes the incoming request.
    /// Counts each request's prompt blocks, plus the blocks generation adds only when the
    /// router runs with `--router-track-output-blocks` (off by default). Zero when the router
    /// runs with `--no-router-track-active-blocks`, as the prefill KV router always does.
    pub fn active_decode_blocks(&self) -> usize {
        self.active_decode_blocks
    }

    /// Return the projected active decode footprint in KV blocks, including the incoming
    /// request's blocks not already active on this worker.
    pub fn decode_cost_blocks(&self) -> f64 {
        self.decode_cost_blocks
    }

    /// Return this worker's active request count.
    pub fn active_requests(&self) -> usize {
        self.active_requests
    }
}
