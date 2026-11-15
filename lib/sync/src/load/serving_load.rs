// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The serving-load streams a Pool Relay aggregates: their event-plane subjects, their
//! cadence, and the flag that turns publishing on. The publishers live with the code that
//! owns what they publish; the Relay subscribes.

use std::time::Duration;

/// Router scheduler views of per-worker load, consumed by the Pool Relay.
///
/// Kept separate from [`KV_METRICS_SUBJECT`](crate::load::KV_METRICS_SUBJECT) so one router's view never
/// overwrites another router's worker-overload state.
pub const SCHEDULER_LOAD_SUBJECT: &str = "scheduler_load";

/// One complete rank batch per interval; the relay derives freshness from this cadence.
pub const SCHEDULER_LOAD_PUBLISH_INTERVAL: Duration = Duration::from_secs(1);

/// Whether this process publishes the serving-load streams a Relay
/// aggregates. Off unless `DYN_SERVING_LOAD_PUBLISH` is truthy.
pub fn serving_load_publishing_enabled() -> bool {
    dynamo_runtime::config::env_is_truthy(
        dynamo_runtime::config::environment_names::llm::DYN_SERVING_LOAD_PUBLISH,
    )
}
