// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request load slots: the write path for a worker's in-flight load.
//!
//! A [`LoadTracker`] hands out one [`SlotId`] per reservation and takes
//! [`SlotEvent`]s against it, so every writer (a request guard, a tracked
//! response stream, a transport fallback) speaks the same two verbs and the
//! slot is the one fence between them.

use crate::protocols::WorkerId;

/// One reserved unit of load on a worker, issued by a [`LoadTracker`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SlotId(u64);

impl SlotId {
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// What can happen to a reserved slot after it was issued.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotEvent {
    /// Transport resolution sent the request to another worker; the slot's
    /// load moves with it.
    Retarget { slot: SlotId, to: WorkerId },
    /// The request is done with its worker; the slot's load is released.
    Release { slot: SlotId },
}

/// Host-side in-flight load, one counter per worker, written through slots.
pub trait LoadTracker: Send + Sync {
    /// Charge one unit of load to `worker` and return the slot that holds it.
    fn reserve(&self, worker: WorkerId) -> SlotId;

    /// Apply `event` to its slot. `false` when the slot is unknown, which
    /// means it was already released.
    fn apply(&self, event: SlotEvent) -> bool;

    /// The load currently charged to `worker`.
    fn load(&self, worker: WorkerId) -> u64;
}
