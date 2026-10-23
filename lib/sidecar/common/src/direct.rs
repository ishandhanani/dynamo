// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-agnostic half of the frontend's direct mode.
//!
//! A [`DirectBackend`] probes one engine URL and dispatches requests to the
//! engines it registered; [`DirectFleet`] probes a URL list through it and
//! turns the result into the frontend's [`EngineConfig::Direct`]. Each engine
//! crate supplies the backend; roles, instance identity and transport-fault
//! mapping live here once.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::{BackendError, DisaggregationMode, DynamoError, LLMEngineOutput};
use dynamo_llm::entrypoint::input::direct::PREFILL_COMPONENT;
use dynamo_llm::entrypoint::{DirectWorker, EngineConfig};
use dynamo_llm::local_model::LocalModel;
use dynamo_llm::local_model::runtime_config::{
    DIRECT_KV_EVENTS_ENDPOINTS_RUNTIME_KEY, DIRECT_KV_REPLAY_ENDPOINT_RUNTIME_KEY,
    ModelRuntimeConfig,
};
use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;
use dynamo_llm::worker_type::WorkerType;
use dynamo_runtime::component::{Instance, TransportType};
use dynamo_runtime::error::ErrorType;
use dynamo_runtime::pipeline::StreamingDispatch;
use dynamo_runtime::protocols::annotated::Annotated;

use crate::GrpcEndpoint;
use crate::error::invalid_argument;

/// What the operator said about an engine URL before it was probed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoleHint {
    /// Listed under the prefill URLs.
    Prefill,
    /// Listed under the plain engine URLs: aggregated, or decode when prefill
    /// engines exist.
    Serving,
}

/// What the frontend needs to know about one engine, whatever its API.
#[derive(Clone, Debug)]
pub struct DirectEngineFacts {
    pub endpoint: GrpcEndpoint,
    pub disaggregation_mode: DisaggregationMode,
    /// Model path or Hugging Face id the engine loaded.
    pub model_path: String,
    pub served_model_name: Option<String>,
    pub kv_cache_block_size: Option<u32>,
    /// What a sidecar would have advertised in the model card, including the
    /// engine's KV-event sockets (see [`record_kv_event_sources`]).
    pub runtime_config: ModelRuntimeConfig,
}

/// One engine family's probe and final hop.
#[async_trait]
pub trait DirectBackend:
    StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>> + Send + Sync + 'static
{
    /// Probe result; holds the facts plus whatever dispatch needs later.
    type Engine: Send + Sync + 'static;

    const NAME: &'static str;
    /// Whether the engine reports its own prefill/decode role. When it does
    /// not, the operator's [`RoleHint`] decides.
    const REPORTS_ROLE: bool;

    async fn probe(&self, url: &str, role: RoleHint) -> Result<Self::Engine, DynamoError>;
    fn facts(engine: &Self::Engine) -> &DirectEngineFacts;
    fn facts_mut(engine: &mut Self::Engine) -> &mut DirectEngineFacts;
    /// Make `engine` routable as `instance_id`.
    fn register(&self, instance_id: u64, engine: Self::Engine);
}

/// Every engine behind one direct frontend, probed and registered with the
/// backend that will route to them.
pub struct DirectFleet<B: DirectBackend> {
    backend: Arc<B>,
    engines: Vec<(DirectEngineFacts, u64)>,
}

impl<B: DirectBackend> DirectFleet<B> {
    /// Probe every URL and register it with `backend`.
    ///
    /// All engines must serve one model. A URL listed twice is rejected because
    /// it would collide on the derived instance id. For a backend that cannot
    /// report roles, plain URLs become decode engines as soon as any prefill
    /// URL is present.
    pub async fn probe(
        backend: Arc<B>,
        urls: &[String],
        prefill_urls: &[String],
    ) -> Result<Self, DynamoError> {
        if urls.is_empty() {
            return Err(invalid_argument(format!(
                "direct mode needs at least one {} engine URL",
                B::NAME
            )));
        }
        let mut probed: Vec<(B::Engine, u64)> = Vec::with_capacity(urls.len() + prefill_urls.len());
        let listed = urls
            .iter()
            .map(|url| (url, RoleHint::Serving))
            .chain(prefill_urls.iter().map(|url| (url, RoleHint::Prefill)));
        for (url, role) in listed {
            let instance_id = instance_id_for_url(url);
            if probed.iter().any(|(_, existing)| *existing == instance_id) {
                return Err(invalid_argument(format!(
                    "engine URL {url} is listed twice"
                )));
            }
            let engine = backend.probe(url, role).await?;
            probed.push((engine, instance_id));
        }
        if !B::REPORTS_ROLE
            && probed
                .iter()
                .any(|(engine, _)| B::facts(engine).disaggregation_mode.is_prefill())
        {
            for (engine, _) in probed.iter_mut() {
                let facts = B::facts_mut(engine);
                if facts.disaggregation_mode == DisaggregationMode::Aggregated {
                    facts.disaggregation_mode = DisaggregationMode::Decode;
                }
            }
        }
        let mut engines = Vec::with_capacity(probed.len());
        for (engine, instance_id) in probed {
            engines.push((B::facts(&engine).clone(), instance_id));
            backend.register(instance_id, engine);
        }
        let fleet = Self { backend, engines };
        let primary = fleet.primary();
        for (facts, _) in &fleet.engines {
            if facts.model_path != primary.model_path {
                return Err(invalid_argument(format!(
                    "direct engines must serve one model: {} serves {}, {} serves {}",
                    primary.endpoint, primary.model_path, facts.endpoint, facts.model_path
                )));
            }
        }
        Ok(fleet)
    }

    /// The engine whose advertisement shapes the model card: the first one
    /// that is not a prefill worker, else the first engine.
    pub fn primary(&self) -> &DirectEngineFacts {
        self.engines
            .iter()
            .find(|(facts, _)| !facts.disaggregation_mode.is_prefill())
            .or_else(|| self.engines.first())
            .map(|(facts, _)| facts)
            .expect("a fleet has at least one engine")
    }

    /// The frontend's engine configuration for `model` over this fleet.
    ///
    /// Aggregated and decode engines share the model's endpoint; prefill
    /// engines live on the prefill component so the prefill hop has its own
    /// endpoint to route over.
    pub fn into_engine_config(self, model: LocalModel) -> EngineConfig {
        let endpoint_id = model.endpoint_id().clone();
        let workers = self
            .engines
            .into_iter()
            .map(|(facts, instance_id)| {
                let worker_type = match facts.disaggregation_mode {
                    DisaggregationMode::Prefill => WorkerType::Prefill,
                    DisaggregationMode::Decode => WorkerType::Decode,
                    _ => WorkerType::Aggregated,
                };
                let component = if worker_type == WorkerType::Prefill {
                    PREFILL_COMPONENT.to_string()
                } else {
                    endpoint_id.component.clone()
                };
                DirectWorker {
                    instance: Instance {
                        namespace: endpoint_id.namespace.clone(),
                        component,
                        endpoint: endpoint_id.name.clone(),
                        instance_id,
                        transport: TransportType::Grpc(facts.endpoint.as_str().to_string()),
                        device_type: None,
                        request_plane_codec: None,
                    },
                    runtime_config: facts.runtime_config,
                    worker_type,
                }
            })
            .collect();
        EngineConfig::Direct {
            model: Box::new(model),
            workers,
            dispatch: self.backend,
        }
    }
}

/// Record the engine's ZMQ KV-event publishers in the card's runtime data so
/// the frontend's direct ingress subscribes to them per rank. `replay` is kept
/// only for single-rank engines: the replay protocol carries no rank.
pub fn record_kv_event_sources(
    runtime_data: &mut HashMap<String, serde_json::Value>,
    endpoints: impl IntoIterator<Item = (u32, String)>,
    replay: Option<String>,
) {
    let endpoints: HashMap<String, String> = endpoints
        .into_iter()
        .map(|(rank, endpoint)| (rank.to_string(), endpoint))
        .collect();
    if endpoints.is_empty() {
        return;
    }
    let single_rank = endpoints.len() == 1;
    runtime_data.insert(
        DIRECT_KV_EVENTS_ENDPOINTS_RUNTIME_KEY.to_string(),
        serde_json::to_value(endpoints).expect("string map serializes"),
    );
    if let Some(replay) = replay.filter(|replay| single_rank && !replay.trim().is_empty()) {
        runtime_data.insert(
            DIRECT_KV_REPLAY_ENDPOINT_RUNTIME_KEY.to_string(),
            serde_json::Value::String(replay),
        );
    }
}

/// Lift gRPC transport failures to the top-level `ErrorType` variants the
/// router's fault detection and migration key on.
///
/// A sidecar reports the same failures as `Backend(..)` because there the
/// engine is "the backend"; on the direct path the frontend owns the
/// connection, so a lost engine is a transport fault.
pub fn remap_transport_error(error: DynamoError) -> DynamoError {
    let top_level = match error.error_type() {
        ErrorType::Backend(BackendError::CannotConnect) => ErrorType::CannotConnect,
        ErrorType::Backend(BackendError::ConnectionTimeout) => ErrorType::ConnectionTimeout,
        ErrorType::Backend(BackendError::ResponseTimeout) => ErrorType::ResponseTimeout,
        ErrorType::Backend(BackendError::Cancelled) => ErrorType::Cancelled,
        ErrorType::Backend(BackendError::EngineShutdown)
        | ErrorType::Backend(BackendError::Disconnected)
        | ErrorType::Backend(BackendError::StreamIncomplete) => ErrorType::Disconnected,
        _ => return error,
    };
    DynamoError::builder()
        .error_type(top_level)
        .message(error.message())
        .cause(error)
        .build()
}

/// Stable instance id for an engine URL, shared by every frontend replica.
pub fn instance_id_for_url(url: &str) -> u64 {
    dynamo_runtime::discovery::hash_pod_name(url.trim().trim_end_matches('/'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend(kind: BackendError) -> DynamoError {
        DynamoError::builder()
            .error_type(ErrorType::Backend(kind))
            .message("Generate: boom")
            .build()
    }

    #[test]
    fn transport_failures_become_top_level_variants() {
        for (kind, expected) in [
            (BackendError::CannotConnect, ErrorType::CannotConnect),
            (
                BackendError::ConnectionTimeout,
                ErrorType::ConnectionTimeout,
            ),
            (BackendError::EngineShutdown, ErrorType::Disconnected),
            (BackendError::StreamIncomplete, ErrorType::Disconnected),
            (BackendError::Cancelled, ErrorType::Cancelled),
        ] {
            let remapped = remap_transport_error(backend(kind));
            assert_eq!(remapped.error_type(), expected);
            assert_eq!(remapped.message(), "Generate: boom");
        }
    }

    #[test]
    fn request_failures_keep_their_backend_shape() {
        let error = remap_transport_error(backend(BackendError::InvalidArgument));
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
    }

    #[test]
    fn instance_ids_are_stable_and_normalized() {
        assert_eq!(
            instance_id_for_url("http://h:30001"),
            instance_id_for_url("http://h:30001/")
        );
        assert_ne!(
            instance_id_for_url("http://h:30001"),
            instance_id_for_url("http://h:30002")
        );
    }

    #[test]
    fn replay_is_recorded_for_single_rank_engines_only() {
        let mut one = HashMap::new();
        record_kv_event_sources(
            &mut one,
            [(0, "tcp://h:5557".to_string())],
            Some("tcp://h:5558".to_string()),
        );
        assert!(one.contains_key(DIRECT_KV_REPLAY_ENDPOINT_RUNTIME_KEY));
        let mut two = HashMap::new();
        record_kv_event_sources(
            &mut two,
            [
                (0, "tcp://h:5557".to_string()),
                (1, "tcp://h:5559".to_string()),
            ],
            Some("tcp://h:5558".to_string()),
        );
        assert!(!two.contains_key(DIRECT_KV_REPLAY_ENDPOINT_RUNTIME_KEY));
        assert_eq!(
            two[DIRECT_KV_EVENTS_ENDPOINTS_RUNTIME_KEY]["1"],
            "tcp://h:5559"
        );
    }
}
