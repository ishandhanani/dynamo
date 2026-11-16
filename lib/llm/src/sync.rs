// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Keeps the router's view of workers up to date.
//!
//! The request path ([`crate::kv_router`]) and the model catalog ([`crate::discovery`]) read
//! what this module maintains. It imports neither of them, nor any other module of this crate.
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

pub(crate) mod direct_zmq_fan_in;
pub(crate) mod direct_zmq_sub_pool;
pub mod indexer;
pub mod load;
pub mod membership;
pub mod metrics;
pub mod runtime_configs;
pub mod sequence;
