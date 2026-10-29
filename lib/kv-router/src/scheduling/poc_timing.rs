// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! POC diagnostics: process-wide time spent in the two awaits of every admitted
//! selection (input preparation with the index lookup, and the scheduler actor
//! round trip). Read and reset by the front admission stats reporter.

use std::sync::atomic::{AtomicU64, Ordering};

pub static SELECTIONS: AtomicU64 = AtomicU64::new(0);
pub static PREPARE_NS: AtomicU64 = AtomicU64::new(0);
pub static SCHEDULE_NS: AtomicU64 = AtomicU64::new(0);

pub fn record(prepare_ns: u64, schedule_ns: u64) {
    SELECTIONS.fetch_add(1, Ordering::Relaxed);
    PREPARE_NS.fetch_add(prepare_ns, Ordering::Relaxed);
    SCHEDULE_NS.fetch_add(schedule_ns, Ordering::Relaxed);
}

/// `(selections, prepare_ns, schedule_ns)` since the last call.
pub fn take() -> (u64, u64, u64) {
    (
        SELECTIONS.swap(0, Ordering::Relaxed),
        PREPARE_NS.swap(0, Ordering::Relaxed),
        SCHEDULE_NS.swap(0, Ordering::Relaxed),
    )
}
