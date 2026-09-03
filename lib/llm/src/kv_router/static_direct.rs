// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-free frontend assembly.
//!
//! [`StaticDirectEngine`] serves one model from a static worker catalog with
//! no `DistributedRuntime`: membership comes from a workers file (the same
//! JSON list the standalone selection service loads with `--workers-file`),
//! KV-aware selection runs on an embedded [`SelectionService`] that subscribes
//! to the workers' ZMQ KV events itself, and admitted requests go to the
//! workers over the direct engine transport (`DirectEngineFactory`) rather than
//! the request plane. Discovery, etcd, NATS, and the runtime request plane are
//! not involved.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, anyhow};
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::WorkerId;
use dynamo_kv_router::services::selection::{
    PromptRequest, SelectAndReserveRequest, SelectionService, SelectionServiceBuilder,
    WorkerRequest, WorkerSelectionPolicyRegistry,
};
use dynamo_kv_router::{DEFAULT_ROUTING_GROUP, WorkerType};
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, ResponseStream, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use super::direct_dispatch::{
    DirectDispatchRegistry, DirectEngineFactory, NATIVE_GRPC_ENDPOINT_RUNTIME_KEY,
};
use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::preprocessor::PreprocessedRequest;
use crate::protocols::common::llm_backend::LLMEngineOutput;
use crate::protocols::common::preprocessor::RoutingHints;

/// What the assembly needs; nothing here references the runtime.
pub struct StaticDirectArgs {
    pub kv_router_config: KvRouterConfig,
    /// Model name the workers are registered under (`WorkerRequest.model_name`).
    pub model_name: String,
    /// KV block size for the selection partition.
    pub block_size: u32,
    /// JSON list of `WorkerRequest`; `endpoint` is the worker's direct engine
    /// endpoint (for vLLM sidecars, the `--advertise-grpc-endpoint` value).
    pub workers_file: PathBuf,
    pub factory: Arc<dyn DirectEngineFactory>,
    pub cancel: CancellationToken,
}

/// Selection plus dispatch for a static worker set. Implements the same engine
/// contract as a request-plane client, so the standard tokenizing pipeline
/// (`build_pipeline`) sits directly on top of it.
pub struct StaticDirectEngine {
    service: Arc<SelectionService>,
    registry: Arc<DirectDispatchRegistry>,
    model_name: String,
    worker_ids: Vec<WorkerId>,
}

impl StaticDirectEngine {
    pub async fn start(args: StaticDirectArgs) -> Result<Arc<Self>> {
        let contents = std::fs::read(&args.workers_file)
            .with_context(|| format!("read workers file {}", args.workers_file.display()))?;
        let mut workers: Vec<WorkerRequest> = serde_json::from_slice(&contents)
            .with_context(|| format!("parse workers file {}", args.workers_file.display()))?;
        if workers.is_empty() {
            anyhow::bail!(
                "workers file {} lists no workers",
                args.workers_file.display()
            );
        }
        for worker in &mut workers {
            worker.model_name = args.model_name.clone();
            if worker.block_size.is_none() {
                worker.block_size = Some(args.block_size);
            }
        }

        // With KV event endpoints the service subscribes to the workers' ZMQ
        // publishers itself. Without any, the indexer stays empty and selection
        // is load-based; the service must then not require the endpoints.
        let has_kv_events = workers.iter().any(|worker| {
            worker.kv_events_endpoint.is_some() || !worker.kv_events_endpoints.is_empty()
        });
        let mut builder = SelectionServiceBuilder::new(
            args.kv_router_config,
            WorkerType::Aggregated,
            WorkerSelectionPolicyRegistry::default(),
        )
        .indexer_threads(1)
        .initial_workers(workers.clone());
        if !has_kv_events {
            tracing::warn!(
                "no worker in the workers file publishes KV events; routing is load-based only"
            );
            builder = builder.external_kv_events();
        }
        let service = builder
            .build()
            .await
            .context("start embedded selection service")?;
        let service = Arc::new(service);

        let registry = Arc::new(DirectDispatchRegistry::new());
        let mut worker_ids = Vec::with_capacity(workers.len());
        for worker in &workers {
            let Some(endpoint) = worker.endpoint.as_deref().filter(|e| !e.is_empty()) else {
                tracing::warn!(
                    worker_id = worker.worker_id,
                    "worker has no direct endpoint; it is registered for selection but unreachable"
                );
                continue;
            };
            let config = runtime_config_from_worker(worker, endpoint);
            match args
                .factory
                .connect(worker.worker_id, endpoint, &config)
                .await
            {
                Ok(engine) => {
                    registry.insert(worker.worker_id, endpoint, engine);
                    worker_ids.push(worker.worker_id);
                }
                Err(error) => {
                    tracing::warn!(
                        worker_id = worker.worker_id,
                        endpoint,
                        %error,
                        "failed to connect direct engine; worker stays unreachable"
                    );
                }
            }
        }
        if registry.is_empty() {
            anyhow::bail!("no worker in {} is reachable", args.workers_file.display());
        }

        let shutdown = Arc::clone(&service);
        let cancel = args.cancel.clone();
        tokio::spawn(async move {
            cancel.cancelled().await;
            shutdown.shutdown().await;
        });

        tracing::info!(
            model = %args.model_name,
            workers = workers.len(),
            reachable = registry.len(),
            "static direct frontend assembled without a distributed runtime"
        );
        Ok(Arc::new(Self {
            service,
            registry,
            model_name: args.model_name,
            worker_ids,
        }))
    }

    /// Workers with a connected direct engine.
    pub fn reachable_worker_ids(&self) -> &[WorkerId] {
        &self.worker_ids
    }

    pub fn selection_service(&self) -> &Arc<SelectionService> {
        &self.service
    }
}

fn runtime_config_from_worker(worker: &WorkerRequest, endpoint: &str) -> ModelRuntimeConfig {
    let mut config = ModelRuntimeConfig {
        total_kv_blocks: worker.total_kv_blocks,
        max_num_batched_tokens: worker.max_num_batched_tokens,
        ..Default::default()
    };
    config.runtime_data.insert(
        NATIVE_GRPC_ENDPOINT_RUNTIME_KEY.to_string(),
        serde_json::Value::String(endpoint.to_string()),
    );
    config
}

/// Frees the selection booking when the response stream is dropped or ends.
struct BookingGuard {
    service: Arc<SelectionService>,
    selection_id: String,
    prefill_done: AtomicBool,
}

impl BookingGuard {
    fn mark_prefill_complete(&self) {
        if self.prefill_done.swap(true, Ordering::AcqRel) {
            return;
        }
        let service = Arc::clone(&self.service);
        let selection_id = self.selection_id.clone();
        tokio::spawn(async move {
            if let Err(error) = service.prefill_complete(&selection_id).await {
                tracing::debug!(selection_id, %error, "prefill_complete after first token");
            }
        });
    }
}

impl Drop for BookingGuard {
    fn drop(&mut self) {
        let service = Arc::clone(&self.service);
        let selection_id = std::mem::take(&mut self.selection_id);
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                if let Err(error) = service.free_reservation(&selection_id).await {
                    tracing::debug!(selection_id, %error, "free_reservation on stream end");
                }
            });
        }
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for StaticDirectEngine
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let selection_id = request.context().id().to_string();
        let expected_output_tokens = request.stop_conditions.max_tokens;
        let response = self
            .service
            .select_and_reserve(SelectAndReserveRequest {
                model_name: self.model_name.clone(),
                routing_group: DEFAULT_ROUTING_GROUP.to_string(),
                selection_id: Some(selection_id.clone()),
                prompt: PromptRequest {
                    token_ids: Some(request.token_ids.clone()),
                    ..Default::default()
                },
                router_config_override: request.router_config_override.clone(),
                expected_output_tokens,
                priority_jump: None,
                strict_priority: None,
                session_id: None,
                session_context: None,
                affinity_target: None,
                pinned_worker: None,
                allowed_worker_ids: None,
                routing_constraints: Default::default(),
            })
            .await
            .map_err(|error| anyhow!("selection failed: {error}"))?;
        let guard = Arc::new(BookingGuard {
            service: Arc::clone(&self.service),
            selection_id,
            prefill_done: AtomicBool::new(false),
        });

        let Some(engine) = self.registry.get(response.worker_id) else {
            // Guard drop frees the booking.
            return Err(anyhow!(
                "selected worker {} has no direct engine",
                response.worker_id
            ));
        };

        let (mut req, ctx) = request.into_parts();
        let hints = req.routing.get_or_insert_with(RoutingHints::default);
        hints.backend_instance_id = Some(response.worker_id);
        hints.dp_rank = Some(response.dp_rank);
        let ctx_arc = ctx.context();
        let stream = engine.generate(ctx.map(|_| req)).await?;
        let stream = stream.map(move |item| {
            if item.data.is_some() {
                guard.mark_prefill_complete();
            }
            item
        });
        Ok(ResponseStream::new(Box::pin(stream), ctx_arc))
    }
}

/// Parse `WorkerRequest`s the way [`StaticDirectEngine::start`] does, for callers
/// that want to validate a workers file up front.
pub fn parse_workers_file(path: &std::path::Path) -> Result<Vec<WorkerRequest>> {
    let contents =
        std::fs::read(path).with_context(|| format!("read workers file {}", path.display()))?;
    serde_json::from_slice(&contents)
        .with_context(|| format!("parse workers file {}", path.display()))
}
