// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct mode: the frontend dials vLLM's native gRPC server (`vllm-rs`) itself.
//!
//! No Dynamo process runs beside the engine. The frontend probes each engine
//! URL once ([`probe`]) to learn what the sidecar would have registered, then
//! routes to it through [`VllmDirectDispatch`], the final hop beneath the
//! router's `PushRouter`. KV events are read from the engine's own ZMQ
//! publisher. vLLM does not report its disaggregation role, so the operator's
//! prefill list decides it. The engine-agnostic fleet and fault mapping live in
//! [`dynamo_sidecar_common::direct`].

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::{DisaggregationMode, DynamoError, LLMEngineOutput};
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;
use dynamo_runtime::discovery::EndpointInstanceId;
use dynamo_runtime::engine::ResponseStream;
use dynamo_runtime::pipeline::{
    AddressedRequest, AsyncEngineContextProvider, Error, ManyIn, ManyOut, SingleIn,
    StreamingDispatch,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::protocols::maybe_error::MaybeError;
use dynamo_sidecar_common::direct::{
    DirectBackend, DirectEngineFacts, RoleHint, record_kv_event_sources, remap_transport_error,
};
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig};
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::client::{self, CONTROL_SERVICE, INFERENCE_SERVICE, VllmClient};
use crate::engine::{PreparedGenerate, open_and_consume, prepare_generate, zmq_connect_endpoint};
use crate::model::DiscoveredModel;

/// Everything the frontend needs to route to one vLLM engine.
#[derive(Clone, Debug)]
pub struct VllmDirectEngine {
    pub facts: DirectEngineFacts,
    model: DiscoveredModel,
}

/// Connect to `url`, wait for its Control and Inference services, and read
/// the engine's discovery RPCs. `role` decides the disaggregation role.
pub async fn probe(
    url: &str,
    transport: GrpcTransportConfig,
    role: RoleHint,
) -> Result<VllmDirectEngine, DynamoError> {
    let endpoint = GrpcEndpoint::parse(url, "--engine-url")?;
    let deadline = client::startup_deadline(transport.startup_deadline)?;
    let probe_transport = GrpcTransportConfig {
        connections: std::num::NonZeroUsize::MIN,
        ..transport
    };
    let client = VllmClient::connect(&endpoint, probe_transport, deadline, false).await?;
    client
        .wait_for_services(
            &[CONTROL_SERVICE, INFERENCE_SERVICE],
            deadline,
            transport.retry_interval,
        )
        .await?;
    let (model, server) = client.discover(deadline).await?;
    let model = DiscoveredModel::from_proto(model, server)?;
    let disaggregation_mode = match role {
        RoleHint::Prefill => DisaggregationMode::Prefill,
        RoleHint::Serving => DisaggregationMode::Aggregated,
    };
    let config = model.engine_config(true)?;
    let llm = config
        .llm
        .as_ref()
        .ok_or_else(|| client::invalid_argument("vLLM engine did not register an LLM"))?;

    let mut runtime_data = config.runtime_data.clone();
    let sources = client.kv_event_sources().await?;
    let mut replay = None;
    let endpoints: Vec<(u32, String)> = sources
        .into_iter()
        .filter(|source| source.transport == "zmq")
        .filter_map(|source| {
            let rank = source.data_parallel_rank?;
            if !source.replay_endpoint.trim().is_empty() {
                replay.get_or_insert_with(|| {
                    zmq_connect_endpoint(&source.replay_endpoint, &endpoint)
                });
            }
            Some((rank, zmq_connect_endpoint(&source.endpoint, &endpoint)))
        })
        .collect();
    let kv_event_sources = endpoints.len();
    record_kv_event_sources(&mut runtime_data, endpoints, replay);

    let runtime_config = direct_runtime_config(&config, runtime_data)?;

    tracing::info!(
        %endpoint,
        mode = ?disaggregation_mode,
        model = %model.source,
        served_model_name = %model.served_name,
        kv_event_sources,
        vllm_tool_call_parser = model.tool_call_parser().unwrap_or("none"),
        vllm_reasoning_parser = model.reasoning_parser().unwrap_or("none"),
        "probed vLLM engine for direct routing"
    );
    Ok(VllmDirectEngine {
        facts: DirectEngineFacts {
            endpoint,
            disaggregation_mode,
            model_path: model.source.clone(),
            served_model_name: Some(model.served_name.clone()),
            kv_cache_block_size: llm.kv_cache_block_size,
            runtime_config,
        },
        model,
    })
}

/// The card's runtime config for a probed engine.
///
/// The parser fields stay empty. vLLM names its parsers differently from
/// Dynamo (`qwen3_xml` where Dynamo has `qwen3`), so the engine's names would
/// fail the card's parser check; the sidecar likewise takes Dynamo's parsers
/// from the operator, never from the engine. The frontend fills them from
/// `--dyn-tool-call-parser` and `--dyn-reasoning-parser`.
fn direct_runtime_config(
    config: &dynamo_backend_common::EngineConfig,
    runtime_data: std::collections::HashMap<String, serde_json::Value>,
) -> Result<ModelRuntimeConfig, DynamoError> {
    let llm = config
        .llm
        .as_ref()
        .ok_or_else(|| client::invalid_argument("vLLM engine did not register an LLM"))?;
    Ok(ModelRuntimeConfig {
        context_length: llm.context_length,
        total_kv_blocks: llm.total_kv_blocks,
        max_num_seqs: llm.max_num_seqs,
        max_num_batched_tokens: llm.max_num_batched_tokens,
        max_gpu_lora_count: llm.max_gpu_lora_count,
        data_parallel_size: llm.data_parallel_size.unwrap_or(1),
        data_parallel_start_rank: llm.data_parallel_start_rank.unwrap_or(0),
        enable_eagle: llm.enable_eagle,
        runtime_data,
        ..ModelRuntimeConfig::default()
    })
}

struct DispatchTarget {
    engine: VllmDirectEngine,
    client: tokio::sync::OnceCell<Arc<VllmClient>>,
}

/// Final hop for direct mode: one gRPC channel pool per registered engine.
pub struct VllmDirectDispatch {
    transport: GrpcTransportConfig,
    targets: dashmap::DashMap<u64, Arc<DispatchTarget>>,
    cancel: CancellationToken,
}

impl VllmDirectDispatch {
    pub fn new(transport: GrpcTransportConfig) -> Arc<Self> {
        Arc::new(Self {
            transport,
            targets: dashmap::DashMap::new(),
            cancel: CancellationToken::new(),
        })
    }

    async fn client(&self, target: &DispatchTarget) -> Result<Arc<VllmClient>, DynamoError> {
        target
            .client
            .get_or_try_init(|| async {
                let deadline = client::startup_deadline(self.transport.connect_attempt_timeout)?;
                VllmClient::connect(
                    &target.engine.facts.endpoint,
                    self.transport,
                    deadline,
                    false,
                )
                .await
                .map(Arc::new)
            })
            .await
            .cloned()
    }
}

#[async_trait]
impl DirectBackend for VllmDirectDispatch {
    type Engine = VllmDirectEngine;
    const NAME: &'static str = "vLLM";
    const REPORTS_ROLE: bool = false;

    async fn probe(&self, url: &str, role: RoleHint) -> Result<VllmDirectEngine, DynamoError> {
        probe(url, self.transport, role).await
    }

    fn facts(engine: &VllmDirectEngine) -> &DirectEngineFacts {
        &engine.facts
    }

    fn facts_mut(engine: &mut VllmDirectEngine) -> &mut DirectEngineFacts {
        &mut engine.facts
    }

    fn register(&self, instance_id: u64, engine: VllmDirectEngine) {
        self.targets.insert(
            instance_id,
            Arc::new(DispatchTarget {
                engine,
                client: tokio::sync::OnceCell::new(),
            }),
        );
    }
}

#[async_trait]
impl StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>> for VllmDirectDispatch {
    async fn generate(
        &self,
        request: SingleIn<AddressedRequest<PreprocessedRequest>>,
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
                anyhow::anyhow!("no vLLM engine registered for instance {instance_id} ({address})")
            })?;
        let client = self
            .client(&target)
            .await
            .map_err(|error| anyhow::Error::from(remap_transport_error(error)))?;
        let ctx = context.context();
        let request_id = ctx.id().to_string();
        let mode = target.engine.facts.disaggregation_mode;
        let PreparedGenerate {
            request: proto_request,
            data_parallel_rank,
            state,
        } = prepare_generate(&target.engine.model, mode, request, &request_id)
            .map_err(|error| anyhow::Error::from(remap_transport_error(error)))?;
        let submit_client = client.clone();
        let submit = async move {
            submit_client
                .generate_stream(proto_request, data_parallel_rank)
                .await
        };
        let stream = open_and_consume(
            submit,
            state,
            ctx.clone(),
            self.cancel.child_token(),
            mode.is_decode(),
            || {},
            Some((client, request_id)),
        )
        .await
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
            "bidirectional streaming is not supported by the vLLM direct dispatcher"
        ))
    }

    async fn on_instance_removed(&self, id: &EndpointInstanceId) {
        self.targets.remove(&id.instance_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{model_info, server_info};

    #[test]
    fn card_leaves_parsers_to_the_operator() {
        let model = DiscoveredModel::from_proto(model_info(), server_info()).unwrap();
        assert!(model.tool_call_parser().is_some());
        assert!(model.reasoning_parser().is_some());
        let config = model.engine_config(true).unwrap();
        let runtime_config = direct_runtime_config(&config, config.runtime_data.clone()).unwrap();
        assert_eq!(runtime_config.tool_call_parser, None);
        assert_eq!(runtime_config.reasoning_parser, None);
        assert_eq!(
            runtime_config.context_length,
            config.llm.as_ref().unwrap().context_length
        );
    }
}
