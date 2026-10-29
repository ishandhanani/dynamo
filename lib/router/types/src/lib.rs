// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared vocabulary of the Dynamo router.
//!
//! Every type that crosses a stage boundary lives here: wire protocols and KV events
//! (`protocols`), identities, router configuration, the per-rank load row, overlap signals,
//! busy thresholds, and the indexer's query and recovery DTOs. Stage crates depend on this
//! crate and never on each other for a type. No tokio, no transport, no runtime.

pub mod affinity;
pub mod busy;
pub mod cache;
pub mod config;
pub mod eligibility;
pub mod identity;
pub mod kv_hints;
pub mod load;
pub mod overlap;
pub mod prefill_load;
pub mod protocols;
pub mod sched;
pub mod slots;
pub mod tracking_hash;
pub mod worker_type;

#[cfg(any(test, feature = "testing", feature = "bench"))]
pub mod test_utils;

pub use worker_type::WorkerType;

pub const CLEANUP_INTERVAL_MS: u64 = 5 * 60 * 1000;
