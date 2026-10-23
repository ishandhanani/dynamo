// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared runtime glue for Rust LLM backends: the single worker library.
//!
//! Two-type abstraction: [`LLMEngine`] (the engine trait an author implements)
//! and [`Worker`] (the runtime lifecycle owner), plus a [`run()`] helper called
//! from each backend's `main.rs`.
//!
//! The worker-side building blocks the bootstrap wires together are public
//! modules of this crate as well: the KV event publisher and state agent
//! ([`kv_publisher`]), the KV state wire contracts ([`kv_state`]) and source
//! membership ([`kv_source_membership`]), the worker admission gate
//! ([`admission_gate`]), the first-token notifier ([`first_token`]),
//! active-sequence publication ([`sequence`]), the per-worker runtime-config
//! watch ([`runtime_configs`]), and the worker-local KV indexer query endpoint
//! ([`worker_query_endpoint`]). `dynamo-llm` re-exports each of them at its
//! historical path.
//!
//! Engines work directly with [`PreprocessedRequest`] and [`LLMEngineOutput`]
//! — the same types the rest of the Rust pipeline uses.
//!
//! See `CLAUDE.md` in this crate for the design contract.

mod adapter;
pub mod admission_gate;
pub mod args;
pub mod disagg;
pub mod engine;
pub mod error;
pub mod first_token;
pub mod kv_publisher;
pub mod kv_source_membership;
pub mod kv_state;
pub mod metrics;
mod publisher;
mod rl;
pub mod run;
pub mod runtime_configs;
pub mod sequence;
pub mod snapshot_publisher;
pub mod telemetry;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
mod topology;
#[cfg(debug_assertions)]
mod validate;
pub mod worker;
pub mod worker_query_endpoint;

// Event-plane subjects for the worker-side publishers (push-based metrics,
// multimodal embedding cache, inter-router active-sequence events).
pub const KV_METRICS_SUBJECT: &str = "kv_metrics";
pub const MULTIMODAL_EMBEDDING_CACHE_SUBJECT: &str = "multimodal_embedding_cache";
pub const ACTIVE_SEQUENCES_SUBJECT: &str = "active_sequences_events";

// Worker-local KV indexer query buffer: the 1024 most recent events.
pub const WORKER_KV_INDEXER_BUFFER_SIZE: usize = 1024;

pub use args::CommonArgs;
pub use disagg::DisaggregationMode;
pub use dynamo_llm_protocol::model_type::ModelInput;
pub use engine::{
    AsyncEngineContext, BootstrapInfo, CompletionUsage, ComponentSnapshot, EngineConfig,
    FinishReason, FirstTokenNotifier, GenerateContext, GuidedDecodingOptions, HEALTH_CHECK_KEY,
    KV_HINT_TRANSFER_CAPABILITY_KEY, KvEventPublisher, KvEventSource, KvHint, KvHintAction,
    KvSourceLocationsPayload, LLMEngine, LLMEngineOutput, LLMEngineOutputExt, LlmRegistration,
    LogProbs, Metrics, MetricsBindings, MetricsCtx, MultimodalData, OnPublisherReady,
    OnSnapshotPublisherReady, OutputOptions, PrefillResult, PreprocessedRequest,
    PromptTokensDetails, RawEngine, SamplingOptions, StopConditions, StopReason, TopLogprob,
    TopLogprobs, chunk, usage,
};
pub use error::{BackendError, DynamoError, ErrorType};
pub use metrics::{ComponentGauges, EngineMetrics, LifecycleGauges};
pub use rl::{RlAdminBaseUrl, RlWorkerMetadata};
pub use run::{run, run_raw};
pub use snapshot_publisher::SnapshotPublisher;
pub use worker::{RuntimeConfig, Worker, WorkerConfig};
