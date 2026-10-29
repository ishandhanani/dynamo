// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Busy-threshold vocabulary shared by the slot tracker (room accounting) and the admission queue.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::load::{WorkerCapacity, WorkerLoadInput};
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerLoadProjection, WorkerWithDpRank};

/// Supplies the host's live reject thresholds. Selection skips a worker whose every rank is
/// past them, and rejects the request when no eligible worker remains.
pub type RejectThresholdsProvider = Arc<dyn Fn() -> BusyThresholds + Send + Sync + 'static>;

/// Per-rank load limits. A rank is full once its booked load passes any configured limit.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BusyThresholds {
    /// Booked prefill tokens above which a rank is full.
    // TODO(v1.7): Remove the absolute threshold; use `prefill_busy_threshold_frac`.
    pub prefill_busy_threshold: Option<usize>,
    /// Fraction of the rank's `max_num_batched_tokens` above which it is full. When the
    /// rank does not report it, the fraction is taken of
    /// [`DEFAULT_MAX_BATCHED_TOKENS`](crate::scheduling::queue::DEFAULT_MAX_BATCHED_TOKENS).
    pub prefill_busy_threshold_frac: Option<f64>,
    /// Fraction of the rank's `total_kv_blocks` above which the KV blocks its active
    /// requests hold make it full. A rank that does not report `total_kv_blocks` skips it.
    pub decode_busy_threshold_frac: Option<f64>,
}

impl BusyThresholds {
    /// Whether any limit is set.
    pub fn is_configured(&self) -> bool {
        self.prefill_busy_threshold.is_some()
            || self.prefill_busy_threshold_frac.is_some()
            || self.decode_busy_threshold_frac.is_some()
    }

    /// Whether a rank with this `LOAD` row and capacity is within every configured limit.
    pub fn has_room(&self, load: &WorkerLoadInput, capacity: WorkerCapacity) -> bool {
        let tokens = load.active_prefill_tokens();
        let prefill_budget = capacity.prefill_budget();
        let absolute_full = self
            .prefill_busy_threshold
            .is_some_and(|threshold| tokens > threshold);
        let fractional_full = self
            .prefill_busy_threshold_frac
            .is_some_and(|threshold| (tokens as f64) > threshold * (prefill_budget as f64));
        let decode_full = self
            .decode_busy_threshold_frac
            .zip(capacity.total_kv_blocks())
            .is_some_and(|(threshold, total_kv_blocks)| {
                (load.active_decode_blocks() as f64) > threshold * (total_kv_blocks as f64)
            });
        !(absolute_full || fractional_full || decode_full)
    }

    /// Whether no rank of `worker_id` has room, judged on the `LOAD` rows in `loads`. A rank
    /// without a projection counts as idle. The policy-class queue and offline replay share
    /// this loop so they enumerate ranks and read rows the same way.
    pub fn worker_is_full<C: WorkerConfigLike>(
        &self,
        worker_id: WorkerId,
        config: &C,
        loads: &FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
    ) -> bool {
        let capacity = WorkerCapacity::of(config);
        let start = config.data_parallel_start_rank();
        let end = start.saturating_add(config.data_parallel_size());
        (start..end).all(|dp_rank| {
            let worker = WorkerWithDpRank::new(worker_id, dp_rank);
            let load = WorkerLoadInput::for_worker(loads, worker, WorkerLoadProjection::default);
            !self.has_room(&load, capacity)
        })
    }
}

/// Ranks past a set of busy thresholds, counted per worker while a request's load is
/// projected. Only a worker with at least one full rank has an entry, so the map stays empty
/// while no worker is near its limits and selection pays nothing to consult it.
#[derive(Debug, Default)]
#[doc(hidden)]
pub struct FullRankCounts {
    limits: BusyThresholds,
    full_ranks: FxHashMap<WorkerId, u32>,
}

impl FullRankCounts {
    /// Start a pass judged against `limits`, forgetting the previous pass.
    #[doc(hidden)]
    pub fn reset(&mut self, limits: BusyThresholds) {
        self.limits = limits;
        self.full_ranks.clear();
    }

    /// Forget what the current pass recorded; it starts over from the first rank.
    #[doc(hidden)]
    pub fn clear(&mut self) {
        self.full_ranks.clear();
    }

    /// Judge one rank's projected load against the limits and count it when it has no room.
    #[inline]
    #[doc(hidden)]
    pub fn record(
        &mut self,
        worker: WorkerWithDpRank,
        capacity: WorkerCapacity,
        projection: &WorkerLoadProjection,
    ) {
        if !self
            .limits
            .has_room(&WorkerLoadInput::from_projection(*projection), capacity)
        {
            *self.full_ranks.entry(worker.worker_id).or_insert(0) += 1;
        }
    }

    #[doc(hidden)]
    pub fn is_empty(&self) -> bool {
        self.full_ranks.is_empty()
    }

    /// Whether every one of the worker's `dp_size` ranks was counted full.
    #[inline]
    #[doc(hidden)]
    pub fn worker_is_full(&self, worker_id: WorkerId, dp_size: u32) -> bool {
        self.full_ranks
            .get(&worker_id)
            .is_some_and(|&full| full >= dp_size)
    }
}
