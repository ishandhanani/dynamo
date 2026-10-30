// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! POC diagnostics, process-wide: time every booked selection spends preparing its inputs
//! (hashing and index lookup) and waiting for its queue-actor step, plus the time spent
//! preparing a relayed booking on the caller's task. For bookings admitted upstream, per
//! leg (prefill or decode): the wait from send to actor pickup, the booking itself, the
//! actor's work before acknowledging, and the caller's whole submit. Read and reset by
//! the front admission stats reporter.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static SELECTIONS: AtomicU64 = AtomicU64::new(0);
static INPUTS_NS: AtomicU64 = AtomicU64::new(0);
static SCHEDULE_NS: AtomicU64 = AtomicU64::new(0);
static BOOKING_PREPARES: AtomicU64 = AtomicU64::new(0);
static BOOKING_PREPARE_NS: AtomicU64 = AtomicU64::new(0);
static CONSTRAINTS: AtomicU64 = AtomicU64::new(0);
static CONSTRAINTS_NS: AtomicU64 = AtomicU64::new(0);

#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);
static SENT_AT_NS: [AtomicU64; 2] = [ZERO; 2];
static ACTOR_STEPS: [AtomicU64; 2] = [ZERO; 2];
static ACTOR_PICKUP_NS: [AtomicU64; 2] = [ZERO; 2];
static ACTOR_ADMIT_NS: [AtomicU64; 2] = [ZERO; 2];
static ACTOR_AFTER_NS: [AtomicU64; 2] = [ZERO; 2];
static SUBMITS: [AtomicU64; 2] = [ZERO; 2];
static SUBMIT_NS: [AtomicU64; 2] = [ZERO; 2];
/// Queue-actor work other than upstream bookings, per leg: capacity-update wakeups and
/// all other commands.
static CAPACITY_WAKES: [AtomicU64; 2] = [ZERO; 2];
static CAPACITY_NS: [AtomicU64; 2] = [ZERO; 2];
static OTHER_COMMANDS: [AtomicU64; 2] = [ZERO; 2];
static OTHER_COMMAND_NS: [AtomicU64; 2] = [ZERO; 2];

/// Queue actors hold their "Selected worker" logs until after they reply.
pub fn defer_selection_logs() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("DYN_ROUTER_POC_DEFER_SELECTION_LOG").is_ok_and(|value| value == "1")
    })
}

/// Requests admitted upstream are booked on the caller's task under the queue actor's
/// lock, not through its command channel. On unless
/// `DYN_ROUTER_FRONT_ADMISSION_DIRECT_BOOK=0`.
pub fn direct_booking() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("DYN_ROUTER_FRONT_ADMISSION_DIRECT_BOOK").map_or(true, |value| value != "0")
    })
}

/// Leg of an upstream-admitted booking: 0 for prefill, 1 for decode (a decode leg does
/// not track prefill tokens).
pub fn leg_of(track_prefill_tokens: bool) -> usize {
    usize::from(!track_prefill_tokens)
}

/// Nanoseconds since the first call in this process.
pub fn now_ns() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

pub fn record(inputs_ns: u64, schedule_ns: u64) {
    SELECTIONS.fetch_add(1, Ordering::Relaxed);
    INPUTS_NS.fetch_add(inputs_ns, Ordering::Relaxed);
    SCHEDULE_NS.fetch_add(schedule_ns, Ordering::Relaxed);
}

pub fn record_prepare(prepare_ns: u64) {
    BOOKING_PREPARES.fetch_add(1, Ordering::Relaxed);
    BOOKING_PREPARE_NS.fetch_add(prepare_ns, Ordering::Relaxed);
}

pub fn record_constraints(ns: u64) {
    CONSTRAINTS.fetch_add(1, Ordering::Relaxed);
    CONSTRAINTS_NS.fetch_add(ns, Ordering::Relaxed);
}

/// The serializer runs one booking per leg at a time, so one slot per leg is exact.
pub fn mark_sent(leg: usize) {
    SENT_AT_NS[leg].store(now_ns(), Ordering::Relaxed);
}

pub fn record_actor(leg: usize, started_ns: u64, admit_ns: u64, after_ns: u64) {
    let sent = SENT_AT_NS[leg].load(Ordering::Relaxed);
    ACTOR_STEPS[leg].fetch_add(1, Ordering::Relaxed);
    ACTOR_PICKUP_NS[leg].fetch_add(started_ns.saturating_sub(sent), Ordering::Relaxed);
    ACTOR_ADMIT_NS[leg].fetch_add(admit_ns, Ordering::Relaxed);
    ACTOR_AFTER_NS[leg].fetch_add(after_ns, Ordering::Relaxed);
}

pub fn record_submit(leg: usize, ns: u64) {
    SUBMITS[leg].fetch_add(1, Ordering::Relaxed);
    SUBMIT_NS[leg].fetch_add(ns, Ordering::Relaxed);
}

pub fn record_capacity_wake(leg: usize, ns: u64) {
    CAPACITY_WAKES[leg].fetch_add(1, Ordering::Relaxed);
    CAPACITY_NS[leg].fetch_add(ns, Ordering::Relaxed);
}

pub fn record_other_command(leg: usize, ns: u64) {
    OTHER_COMMANDS[leg].fetch_add(1, Ordering::Relaxed);
    OTHER_COMMAND_NS[leg].fetch_add(ns, Ordering::Relaxed);
}

/// Means for one leg since the last call, in microseconds.
#[derive(Debug, Default, Clone, Copy)]
pub struct LegSnapshot {
    pub steps: u64,
    pub pickup_us: f64,
    pub admit_us: f64,
    pub after_us: f64,
    pub submit_us: f64,
    pub capacity_wakes: u64,
    /// Total time in capacity-update wakeups, in microseconds.
    pub capacity_total_us: f64,
    pub other_commands: u64,
    /// Total time in other commands, in microseconds.
    pub other_total_us: f64,
}

/// Totals since the last call.
#[derive(Debug, Default, Clone, Copy)]
pub struct Snapshot {
    pub selections: u64,
    pub inputs_ns: u64,
    pub schedule_ns: u64,
    pub booking_prepares: u64,
    pub booking_prepare_ns: u64,
    pub constraints_us: f64,
    pub legs: [LegSnapshot; 2],
}

fn mean_us(total: &AtomicU64, count: u64) -> f64 {
    total.swap(0, Ordering::Relaxed) as f64 / count.max(1) as f64 / 1e3
}

pub fn take() -> Snapshot {
    let constraints = CONSTRAINTS.swap(0, Ordering::Relaxed);
    let leg = |leg: usize| {
        let steps = ACTOR_STEPS[leg].swap(0, Ordering::Relaxed);
        let submits = SUBMITS[leg].swap(0, Ordering::Relaxed);
        LegSnapshot {
            steps,
            pickup_us: mean_us(&ACTOR_PICKUP_NS[leg], steps),
            admit_us: mean_us(&ACTOR_ADMIT_NS[leg], steps),
            after_us: mean_us(&ACTOR_AFTER_NS[leg], steps),
            submit_us: mean_us(&SUBMIT_NS[leg], submits),
            capacity_wakes: CAPACITY_WAKES[leg].swap(0, Ordering::Relaxed),
            capacity_total_us: CAPACITY_NS[leg].swap(0, Ordering::Relaxed) as f64 / 1e3,
            other_commands: OTHER_COMMANDS[leg].swap(0, Ordering::Relaxed),
            other_total_us: OTHER_COMMAND_NS[leg].swap(0, Ordering::Relaxed) as f64 / 1e3,
        }
    };
    Snapshot {
        selections: SELECTIONS.swap(0, Ordering::Relaxed),
        inputs_ns: INPUTS_NS.swap(0, Ordering::Relaxed),
        schedule_ns: SCHEDULE_NS.swap(0, Ordering::Relaxed),
        booking_prepares: BOOKING_PREPARES.swap(0, Ordering::Relaxed),
        booking_prepare_ns: BOOKING_PREPARE_NS.swap(0, Ordering::Relaxed),
        constraints_us: mean_us(&CONSTRAINTS_NS, constraints),
        legs: [leg(0), leg(1)],
    }
}
