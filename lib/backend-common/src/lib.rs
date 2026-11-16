// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared runtime glue for Rust LLM backends.
//!
//! Two-type abstraction: [`LLMEngine`] (the engine trait an author implements)
//! and [`Worker`] (the runtime lifecycle owner), plus a [`run()`] helper called
//! from each backend's `main.rs`.
//!
//! Engines work directly with [`PreprocessedRequest`] and [`LLMEngineOutput`]
//! — the same types the rest of the Rust pipeline uses.
//!
//! The crate also owns the services that run in a worker next to its engine:
//! [`admission_gate`] (one admission gate per backend worker, in front of the engine),
//! [`first_token`] (the first-token notification), [`kv_publisher`] (the KV event and
//! worker metrics publishers, the ZMQ listener and its recovery, the KV state agent) and
//! [`fpm_publisher`] (the forward-pass-metrics relay). It never depends on `dynamo-llm`;
//! `dynamo-llm` depends on this crate.
//!
//! See `CLAUDE.md` in this crate for the design contract.

mod adapter;
pub mod admission_gate;
pub mod args;
pub mod disagg;
pub mod engine;
pub mod error;
pub mod first_token;
pub mod fpm_publisher;
pub mod kv_publisher;
pub mod metrics;
mod publisher;
mod rl;
pub mod run;
pub mod snapshot_publisher;
pub mod telemetry;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
mod topology;
#[cfg(debug_assertions)]
mod validate;
pub mod worker;

// Only the tracing guard of the shared test helpers is used here.
#[cfg(test)]
#[allow(dead_code)]
#[path = "../../runtime/src/test_utils.rs"]
mod test_utils;

pub use args::CommonArgs;
pub use disagg::DisaggregationMode;
pub use dynamo_model_card::model_type::ModelInput;
pub use engine::{
    AsyncEngineContext, BootstrapInfo, CompletionUsage, ComponentSnapshot, EngineConfig,
    FinishReason, FirstTokenNotifier, GenerateContext, GuidedDecodingOptions, HEALTH_CHECK_KEY,
    KV_HINT_TRANSFER_CAPABILITY_KEY, KvEventPublisher, KvEventSource, KvHint, KvHintAction,
    KvSourceLocationsPayload, LLMEngine, LLMEngineOutput, LLMEngineOutputExt, LlmRegistration,
    LogProbs, Metrics, MetricsBindings, MetricsCtx, MultimodalData, MultimodalItemRef,
    OnPublisherReady, OnSnapshotPublisherReady, OutputOptions, PrefillResult, PreprocessedRequest,
    PromptTokensDetails, RawEngine, SamplingOptions, StopConditions, StopReason, TopLogprob,
    TopLogprobs, chunk, usage,
};
pub use error::{BackendError, DynamoError, ErrorType};
pub use metrics::{ComponentGauges, EngineMetrics, LifecycleGauges};
pub use rl::{RlAdminBaseUrl, RlWorkerMetadata};
pub use run::{run, run_raw};
pub use snapshot_publisher::SnapshotPublisher;
pub use worker::{RuntimeConfig, Worker, WorkerConfig};
