// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! POC diagnostics, process-wide: time every booked selection spends preparing its inputs
//! (hashing and index lookup) and waiting for its queue-actor step, plus the time spent
//! preparing a relayed booking on the caller's task, and, for bookings admitted upstream,
//! the queue actor's own time: the booking itself, then the work it does before
//! acknowledging. Read and reset by the front admission stats reporter.

use std::sync::atomic::{AtomicU64, Ordering};

static SELECTIONS: AtomicU64 = AtomicU64::new(0);
static INPUTS_NS: AtomicU64 = AtomicU64::new(0);
static SCHEDULE_NS: AtomicU64 = AtomicU64::new(0);
static BOOKING_PREPARES: AtomicU64 = AtomicU64::new(0);
static BOOKING_PREPARE_NS: AtomicU64 = AtomicU64::new(0);
static ACTOR_STEPS: AtomicU64 = AtomicU64::new(0);
static ACTOR_ADMIT_NS: AtomicU64 = AtomicU64::new(0);
static ACTOR_AFTER_NS: AtomicU64 = AtomicU64::new(0);

pub fn record(inputs_ns: u64, schedule_ns: u64) {
    SELECTIONS.fetch_add(1, Ordering::Relaxed);
    INPUTS_NS.fetch_add(inputs_ns, Ordering::Relaxed);
    SCHEDULE_NS.fetch_add(schedule_ns, Ordering::Relaxed);
}

pub fn record_prepare(prepare_ns: u64) {
    BOOKING_PREPARES.fetch_add(1, Ordering::Relaxed);
    BOOKING_PREPARE_NS.fetch_add(prepare_ns, Ordering::Relaxed);
}

pub fn record_actor(admit_ns: u64, after_ns: u64) {
    ACTOR_STEPS.fetch_add(1, Ordering::Relaxed);
    ACTOR_ADMIT_NS.fetch_add(admit_ns, Ordering::Relaxed);
    ACTOR_AFTER_NS.fetch_add(after_ns, Ordering::Relaxed);
}

/// Totals since the last call.
#[derive(Debug, Default, Clone, Copy)]
pub struct Snapshot {
    pub selections: u64,
    pub inputs_ns: u64,
    pub schedule_ns: u64,
    pub booking_prepares: u64,
    pub booking_prepare_ns: u64,
    pub actor_steps: u64,
    pub actor_admit_ns: u64,
    pub actor_after_ns: u64,
}

pub fn take() -> Snapshot {
    Snapshot {
        selections: SELECTIONS.swap(0, Ordering::Relaxed),
        inputs_ns: INPUTS_NS.swap(0, Ordering::Relaxed),
        schedule_ns: SCHEDULE_NS.swap(0, Ordering::Relaxed),
        booking_prepares: BOOKING_PREPARES.swap(0, Ordering::Relaxed),
        booking_prepare_ns: BOOKING_PREPARE_NS.swap(0, Ordering::Relaxed),
        actor_steps: ACTOR_STEPS.swap(0, Ordering::Relaxed),
        actor_admit_ns: ACTOR_ADMIT_NS.swap(0, Ordering::Relaxed),
        actor_after_ns: ACTOR_AFTER_NS.swap(0, Ordering::Relaxed),
    }
}
