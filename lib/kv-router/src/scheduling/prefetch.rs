// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deadline-led prefetch hints from the admission queue.
//!
//! A queued request can have its KV pages loaded onto a worker before it is
//! admitted: `lead` before its `due_at` (or at enqueue, for a request flagged
//! `prefetch_on_arrival` or when the trigger fires `on_queue`), while the entry is still queued, the actor selects a
//! worker without booking and hands the request's shared-store-resident prefix
//! to the sink. When a hint was sent, that worker becomes the entry's
//! `affinity_target`, with that field's usual semantics: the default selector
//! treats it as exclusive while the worker stays eligible, like session
//! affinity, and falls back to a free selection otherwise; custom policies see
//! it as advisory. No hint, no affinity.

use std::sync::Arc;
use std::time::Duration;

use crate::protocols::WorkerWithDpRank;

/// Receives prefetch hints chosen by the admission queue. Implementations must
/// return immediately: the call runs on the scheduler actor.
pub trait PrefetchHintSink: Send + Sync {
    fn prefetch(&self, worker: WorkerWithDpRank, tokens: &[u32]);

    /// Expected milliseconds for `worker` to load a `tokens`-long hint from the
    /// shared store, when the sink can tell. `None` means unknown.
    fn load_estimate_ms(&self, _worker: WorkerWithDpRank, _tokens: usize) -> Option<f64> {
        None
    }
}

/// Installed once per scheduler; see the module docs.
#[derive(Clone)]
pub struct PrefetchTrigger {
    /// How long before an entry's `due_at` the hint fires.
    pub lead: Duration,
    /// Fire at enqueue for every parked entry, deadline or not: a request that
    /// is waiting anyway can have its prefix loaded while it waits.
    pub on_queue: bool,
    pub sink: Arc<dyn PrefetchHintSink>,
}
