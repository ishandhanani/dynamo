// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo Sync
//!
//! Keeps a router's view of its workers up to date.
//!
//! The request path (`dynamo_router_host`) and the model catalog (`dynamo_llm::discovery`)
//! read what this crate maintains. It depends on neither: both are above it.
//!
//! - [`indexer`]: which KV-cache blocks each worker holds. Ingest of worker KV events, recovery,
//!   and the remote indexer.
//! - [`sequence`]: active-sequence (slot-tracker) replica sync between routers.
//! - [`load`]: the load workers publish and its subscription, the worker monitor with its busy
//!   thresholds, and the routing load context.
//! - [`membership`]: which worker publishes KV events for which rank, and the vocabulary the
//!   KV state agents share with the index.
//! - [`runtime_configs`]: the watch of per-worker runtime configs.
//! - [`metrics`]: the Prometheus metrics of the above.
//! - `direct_zmq_fan_in`, `direct_zmq_sub_pool`: the direct-ZMQ subscription the sync channels
//!   share.

pub mod direct_zmq_fan_in;
pub mod direct_zmq_sub_pool;
pub mod indexer;
pub mod load;
pub mod membership;
pub mod metrics;
pub mod runtime_configs;
pub mod sequence;

// Only the tracing guard of the shared test helpers is used here.
#[cfg(test)]
#[allow(dead_code)]
#[path = "../../runtime/src/test_utils.rs"]
mod test_utils;
