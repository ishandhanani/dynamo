// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::plugins::worker_selection::WorkerLoadInput;
use crate::protocols::WorkerWithDpRank;

/// One registered worker rank and its advertised capacity, which may be absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestClassifierWorker {
    worker: WorkerWithDpRank,
    total_kv_blocks: Option<u64>,
}

impl RequestClassifierWorker {
    pub fn new(worker: WorkerWithDpRank, total_kv_blocks: Option<u64>) -> Self {
        Self {
            worker,
            total_kv_blocks,
        }
    }

    pub fn worker(&self) -> WorkerWithDpRank {
        self.worker
    }

    /// Total capacity from the worker's runtime config, not currently free blocks.
    /// Each block contains [`RequestClassifierContext::block_size`] tokens.
    pub fn total_kv_blocks(&self) -> Option<u64> {
        self.total_kv_blocks
    }
}

/// Cached discovery inputs for one router. Workers appear and disappear as the
/// host processes registration updates; this does not probe worker health.
#[derive(Clone)]
pub struct RequestClassifierContext {
    block_size: u32,
    workers: Arc<dyn Fn() -> Vec<RequestClassifierWorker> + Send + Sync>,
    worker_loads: Option<Arc<WorkerLoadsFn>>,
}

type WorkerLoadsFn = dyn Fn() -> Vec<(WorkerWithDpRank, WorkerLoadInput)> + Send + Sync;

impl RequestClassifierContext {
    /// The callback must return cached state without blocking I/O.
    pub fn new(
        block_size: u32,
        workers: impl Fn() -> Vec<RequestClassifierWorker> + Send + Sync + 'static,
    ) -> Self {
        Self {
            block_size,
            workers: Arc::new(workers),
            worker_loads: None,
        }
    }

    /// Attach the host's live load reader. The callback must read in-memory state
    /// without blocking I/O.
    pub fn with_worker_loads(
        mut self,
        worker_loads: impl Fn() -> Vec<(WorkerWithDpRank, WorkerLoadInput)> + Send + Sync + 'static,
    ) -> Self {
        self.worker_loads = Some(Arc::new(worker_loads));
        self
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn workers(&self) -> Vec<RequestClassifierWorker> {
        (self.workers)()
    }

    /// Read the router's current load on every registered worker rank, in unspecified
    /// order. Returns `None` when the host does not track load.
    ///
    /// Each call reads fresh state, so a classifier that holds requests can use it when
    /// it decides to release one. The values come from the router's own bookings, not
    /// from engine KV occupancy. No request is attached, so `decode_cost_blocks` is the
    /// current decode footprint; in [`ClassifyRequest::read_worker_inputs`] it also
    /// includes that request's uncached prompt blocks.
    ///
    /// [`ClassifyRequest::read_worker_inputs`]: super::ClassifyRequest::read_worker_inputs
    pub fn worker_loads(&self) -> Option<Vec<(WorkerWithDpRank, WorkerLoadInput)>> {
        self.worker_loads
            .as_ref()
            .map(|worker_loads| worker_loads())
    }
}

impl std::fmt::Debug for RequestClassifierContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RequestClassifierContext")
            .field("block_size", &self.block_size)
            .finish_non_exhaustive()
    }
}
