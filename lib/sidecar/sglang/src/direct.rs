// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct mode: the frontend dials SGLang's native gRPC server itself.
//!
//! No Dynamo process runs beside the engine. The frontend probes each engine
//! URL once ([`probe`]) to learn what the sidecar would have registered, then
//! routes to it through [`SglangDirectDispatch`], the final hop beneath the
//! router's `PushRouter`. KV events are read from the engine's own ZMQ
//! publisher; the probe records those endpoints in the runtime config. The
//! engine-agnostic fleet and fault mapping live in
//! [`dynamo_sidecar_common::direct`].

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::{DynamoError, LLMEngineOutput};
use dynamo_llm::local_model::runtime_config::{DisaggregatedEndpoint, ModelRuntimeConfig};
use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;
use dynamo_runtime::discovery::EndpointInstanceId;
use dynamo_runtime::engine::ResponseStream;
use dynamo_runtime::pipeline::{
    AsyncEngineContextProvider, Error, ManyIn, ManyOut, SingleIn, StreamingDispatch,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::protocols::maybe_error::MaybeError;
use dynamo_sidecar_common::direct::{
    DirectBackend, DirectEngineFacts, RoleHint, record_kv_event_sources, remap_transport_error,
};
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig, startup_deadline};
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::client::{self, Pool};
use crate::engine::{
    GrpcGenerateTarget, await_engine_ready, build_engine_config, combined_token_limit,
    discover_kv_event_sources, discovery_bootstrap_port, discovery_string, grpc_generate_stream,
    resolve_bootstrap_host,
};

/// Everything the frontend needs to route to one SGLang engine.
#[derive(Clone, Debug)]
pub struct DirectEngine {
    pub facts: DirectEngineFacts,
    pub tokenizer_path: String,
    pub reasoning_parser: Option<String>,
    pub tool_call_parser: Option<String>,
    bootstrap_host: Option<String>,
    bootstrap_port: Option<u16>,
    combined_token_limit: Option<u32>,
}

/// Connect to `url`, wait for readiness, and read the engine's discovery RPCs.
///
/// SGLang reports its own disaggregation role; `role` must agree with it.
pub async fn probe(
    url: &str,
    transport: GrpcTransportConfig,
    role: RoleHint,
) -> Result<DirectEngine, DynamoError> {
    let endpoint = GrpcEndpoint::parse(url, "--engine-url")?;
    let deadline = startup_deadline(transport.startup_deadline)?;
    let pool = Pool::connect(&endpoint, &transport, deadline).await?;
    let mut control = pool.control_client();
    await_engine_ready(
        &mut control,
        deadline,
        transport.startup_deadline,
        transport.retry_interval,
    )
    .await?;
    let discovery = client::discover(&mut control, deadline).await?;
    let disaggregation_mode = client::discovery_mode(&discovery.server_info)?;
    match role {
        RoleHint::Prefill if !disaggregation_mode.is_prefill() => {
            return Err(client::invalid_arg(format!(
                "{endpoint} was listed as a prefill engine but reports {disaggregation_mode:?}"
            )));
        }
        RoleHint::Serving if disaggregation_mode.is_prefill() => {
            return Err(client::invalid_arg(format!(
                "{endpoint} reports a prefill role; list it with --prefill-engine-url"
            )));
        }
        _ => {}
    }
    let (bootstrap_host, bootstrap_port) = if disaggregation_mode.is_prefill() {
        (
            resolve_bootstrap_host(None, endpoint.as_str(), &discovery)?,
            discovery_bootstrap_port(&discovery)?,
        )
    } else {
        (None, None)
    };
    let mut config = build_engine_config(
        &discovery,
        disaggregation_mode,
        bootstrap_host.clone(),
        bootstrap_port,
    )?;
    let kv_sources =
        discover_kv_event_sources(&discovery, &mut config, &endpoint, deadline).await?;
    let llm = config
        .llm
        .as_ref()
        .ok_or_else(|| client::invalid_arg("SGLang engine did not register an LLM"))?;

    let mut runtime_data = config.runtime_data.clone();
    // SGLang's node metadata does not surface a replay socket; live events only.
    record_kv_event_sources(
        &mut runtime_data,
        kv_sources
            .iter()
            .map(|source| (source.dp_rank, source.endpoint.clone())),
        None,
    );
    let disaggregated_endpoint = match (&bootstrap_host, bootstrap_port) {
        (Some(host), Some(port)) => Some(DisaggregatedEndpoint {
            bootstrap_host: Some(host.clone()),
            bootstrap_port: Some(port),
        }),
        _ => None,
    };
    let reasoning_parser = discovery_string(&discovery.server_info, "reasoning_parser");
    let tool_call_parser = discovery_string(&discovery.server_info, "tool_call_parser");
    let runtime_config = ModelRuntimeConfig {
        context_length: llm.context_length,
        total_kv_blocks: llm.total_kv_blocks,
        max_num_seqs: llm.max_num_seqs,
        max_num_batched_tokens: llm.max_num_batched_tokens,
        max_gpu_lora_count: llm.max_gpu_lora_count,
        data_parallel_size: llm.data_parallel_size.unwrap_or(1),
        data_parallel_start_rank: llm.data_parallel_start_rank.unwrap_or(0),
        enable_eagle: llm.enable_eagle,
        tool_call_parser: tool_call_parser.clone(),
        reasoning_parser: reasoning_parser.clone(),
        disaggregated_endpoint,
        runtime_data,
        ..ModelRuntimeConfig::default()
    };

    tracing::info!(
        %endpoint,
        mode = ?disaggregation_mode,
        model = %discovery.model_path,
        kv_event_sources = kv_sources.len(),
        "probed SGLang engine for direct routing"
    );
    Ok(DirectEngine {
        facts: DirectEngineFacts {
            endpoint,
            disaggregation_mode,
            model_path: discovery.model_path.clone(),
            served_model_name: discovery.served_model_name.clone(),
            kv_cache_block_size: llm.kv_cache_block_size,
            runtime_config,
        },
        tokenizer_path: discovery.tokenizer_path.clone(),
        reasoning_parser,
        tool_call_parser,
        bootstrap_host,
        bootstrap_port,
        combined_token_limit: combined_token_limit(&discovery),
    })
}

struct DispatchTarget {
    engine: DirectEngine,
    pool: tokio::sync::OnceCell<Arc<Pool>>,
}

/// Final hop for direct mode: one gRPC channel pool per registered engine.
///
/// Instances are registered by id before the router can select them; the
/// pool connects lazily on first use and is dropped when discovery removes
/// the instance.
pub struct SglangDirectDispatch {
    transport: GrpcTransportConfig,
    targets: dashmap::DashMap<u64, Arc<DispatchTarget>>,
    cancel: CancellationToken,
}

impl SglangDirectDispatch {
    pub fn new(transport: GrpcTransportConfig) -> Arc<Self> {
        Arc::new(Self {
            transport,
            targets: dashmap::DashMap::new(),
            cancel: CancellationToken::new(),
        })
    }

    async fn pool(&self, target: &DispatchTarget) -> Result<Arc<Pool>, DynamoError> {
        target
            .pool
            .get_or_try_init(|| async {
                let deadline = startup_deadline(self.transport.connect_attempt_timeout)?;
                Pool::connect(&target.engine.facts.endpoint, &self.transport, deadline)
                    .await
                    .map(Arc::new)
            })
            .await
            .cloned()
    }
}

#[async_trait]
impl DirectBackend for SglangDirectDispatch {
    type Engine = DirectEngine;
    const NAME: &'static str = "SGLang";
    const REPORTS_ROLE: bool = true;

    async fn probe(&self, url: &str, role: RoleHint) -> Result<DirectEngine, DynamoError> {
        probe(url, self.transport, role).await
    }

    fn facts(engine: &DirectEngine) -> &DirectEngineFacts {
        &engine.facts
    }

    fn facts_mut(engine: &mut DirectEngine) -> &mut DirectEngineFacts {
        &mut engine.facts
    }

    fn register(&self, instance_id: u64, engine: DirectEngine) {
        self.targets.insert(
            instance_id,
            Arc::new(DispatchTarget {
                engine,
                pool: tokio::sync::OnceCell::new(),
            }),
        );
    }
}

#[async_trait]
impl StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>> for SglangDirectDispatch {
    async fn generate(
        &self,
        request: SingleIn<dynamo_runtime::pipeline::AddressedRequest<PreprocessedRequest>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (addressed, context) = request.transfer(());
        let (request, address, instance) = addressed.into_parts();
        let instance_id = instance
            .map(|instance| instance.id())
            .ok_or_else(|| anyhow::anyhow!("direct dispatch requires the selected instance"))?;
        let target = self
            .targets
            .get(&instance_id)
            .map(|entry| entry.value().clone())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no SGLang engine registered for instance {instance_id} ({address})"
                )
            })?;
        let pool = self
            .pool(&target)
            .await
            .map_err(|error| anyhow::Error::from(remap_transport_error(error)))?;
        let ctx = context.context();
        let stream = grpc_generate_stream(
            GrpcGenerateTarget {
                pool: &pool,
                mode: target.engine.facts.disaggregation_mode,
                bootstrap_host: target.engine.bootstrap_host.as_deref(),
                bootstrap_port: target.engine.bootstrap_port,
                combined_token_limit: target.engine.combined_token_limit,
                abort_on_cancel: Some(self.transport.connect_attempt_timeout),
            },
            request,
            ctx.clone(),
            self.cancel.child_token(),
        )
        .map_err(|error| anyhow::Error::from(remap_transport_error(error)))?;
        let mapped = stream.map(|item| match item {
            Ok(output) => Annotated::from_data(output),
            Err(error) => Annotated::from_err(remap_transport_error(error)),
        });
        Ok(ResponseStream::new(Box::pin(mapped), ctx))
    }

    async fn generate_bidirectional(
        &self,
        _instance: dynamo_runtime::component::Instance,
        _address: String,
        _input: ManyIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        Err(anyhow::anyhow!(
            "bidirectional streaming is not supported by the SGLang direct dispatcher"
        ))
    }

    async fn on_instance_removed(&self, id: &EndpointInstanceId) {
        self.targets.remove(&id.instance_id);
    }
}
