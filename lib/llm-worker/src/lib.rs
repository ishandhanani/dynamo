// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo LLM Worker
//!
//! Worker-side runtime glue shared by the Rust backends and sidecars: the KV
//! event publisher and state agent, the worker admission gate, the first-token
//! notifier, active-sequence publication, KV source membership, and the
//! direct-ZMQ subscription pool and fan-in. It depends on `dynamo-runtime`,
//! `dynamo-kv-router`, and `dynamo-llm-protocol` only; `dynamo-llm` re-exports
//! every module here at its historical path, so existing
//! `dynamo_llm::{first_token, admission_gate, kv_router::{publisher, sequence},
//! discovery::kv_source_membership, discovery::kv_state_agent}` imports keep
//! working.

pub mod admission_gate;
pub mod direct_zmq_fan_in;
pub mod direct_zmq_sub_pool;
pub mod first_token;
pub mod kv_source_membership;
pub mod kv_state;
pub mod publisher;
pub mod sequence;
pub mod worker_query_endpoint;

// for metric publishing (push-based)
pub const KV_METRICS_SUBJECT: &str = "kv_metrics";
pub const MULTIMODAL_EMBEDDING_CACHE_SUBJECT: &str = "multimodal_embedding_cache";

// for inter-router comms
pub const ACTIVE_SEQUENCES_SUBJECT: &str = "active_sequences_events";

// for worker-local kvindexer query
pub const WORKER_KV_INDEXER_BUFFER_SIZE: usize = 1024; // store 1024 most recent events in worker buffer
