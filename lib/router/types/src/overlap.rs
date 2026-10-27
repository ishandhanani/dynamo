// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Overlap signals: the per-rank cache-hit vocabulary the kernel derives from indexer results.

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::protocols::{DpRank, WorkerConfigLike, WorkerId, WorkerWithDpRank};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TierOverlapBlocks {
    #[serde(default)]
    pub device: FxHashMap<WorkerWithDpRank, usize>,
    #[serde(default)]
    pub host_pinned: FxHashMap<WorkerWithDpRank, usize>,
    #[serde(default)]
    pub disk: FxHashMap<WorkerWithDpRank, usize>,
}

#[derive(Debug, Clone, Default)]
pub struct CacheHitEstimates {
    pub effective_overlap_blocks: FxHashMap<WorkerWithDpRank, f64>,
    pub cached_tokens: FxHashMap<WorkerWithDpRank, usize>,
}

/// Compact overlap state retained while a request waits for scheduling.
#[derive(Debug, Clone, Default)]
pub struct OverlapSignals {
    pub tier_overlap_blocks: TierOverlapBlocks,
    pub effective_overlap_blocks: FxHashMap<WorkerWithDpRank, f64>,
    pub effective_cached_tokens: FxHashMap<WorkerWithDpRank, usize>,
}

impl OverlapSignals {
    pub fn selected_worker_tiers<C: WorkerConfigLike>(
        &self,
        worker: WorkerWithDpRank,
        config: &C,
    ) -> SelectedWorkerTierSnapshot {
        let start = config.data_parallel_start_rank();
        let end = start.saturating_add(config.data_parallel_size());
        let mut dp_device_blocks = Vec::new();
        let mut gpu_blocks = 0;
        let mut host_pinned_blocks = 0;
        let mut disk_blocks = 0;

        for dp_rank in start..end {
            let rank = WorkerWithDpRank::new(worker.worker_id, dp_rank);
            let device = saturating_u32(
                self.tier_overlap_blocks
                    .device
                    .get(&rank)
                    .copied()
                    .unwrap_or(0),
            );
            let host = device.saturating_add(saturating_u32(
                self.tier_overlap_blocks
                    .host_pinned
                    .get(&rank)
                    .copied()
                    .unwrap_or(0),
            ));
            let disk = host.saturating_add(saturating_u32(
                self.tier_overlap_blocks
                    .disk
                    .get(&rank)
                    .copied()
                    .unwrap_or(0),
            ));

            dp_device_blocks.push((dp_rank, device));
            gpu_blocks = gpu_blocks.max(device);
            host_pinned_blocks = host_pinned_blocks.max(host);
            disk_blocks = disk_blocks.max(disk);
        }

        if dp_device_blocks.is_empty() {
            let device = saturating_u32(
                self.tier_overlap_blocks
                    .device
                    .get(&worker)
                    .copied()
                    .unwrap_or(0),
            );
            let host = device.saturating_add(saturating_u32(
                self.tier_overlap_blocks
                    .host_pinned
                    .get(&worker)
                    .copied()
                    .unwrap_or(0),
            ));
            let disk = host.saturating_add(saturating_u32(
                self.tier_overlap_blocks
                    .disk
                    .get(&worker)
                    .copied()
                    .unwrap_or(0),
            ));
            dp_device_blocks.push((worker.dp_rank, device));
            gpu_blocks = device;
            host_pinned_blocks = host;
            disk_blocks = disk;
        }

        SelectedWorkerTierSnapshot {
            dp_device_blocks,
            gpu_blocks,
            host_pinned_blocks,
            disk_blocks,
        }
    }
}

/// Raw selected-worker overlap from the exact inputs used for final scheduling.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SelectedWorkerTierSnapshot {
    pub dp_device_blocks: Vec<(DpRank, u32)>,
    pub gpu_blocks: u32,
    pub host_pinned_blocks: u32,
    pub disk_blocks: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkerOverlapScore {
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
    pub device_blocks: usize,
    pub host_pinned_blocks: usize,
    pub disk_blocks: usize,
    pub host_pinned_extension_blocks: usize,
    pub disk_extension_blocks: usize,
    pub shared_beyond_device_blocks: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SharedCacheOverlapScore {
    pub enabled: bool,
    pub total_hit_blocks: u32,
    pub ranges: Vec<(u32, u32)>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OverlapScoresResponse {
    pub block_size: u32,
    pub num_blocks: usize,
    pub workers: Vec<WorkerOverlapScore>,
    pub shared_cache: SharedCacheOverlapScore,
}

#[doc(hidden)]
pub fn saturating_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}
