// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-agnostic half of the frontend's direct mode.
//!
//! A [`DirectBackend`] probes one engine URL and dispatches requests to the
//! engines it registered; [`DirectFleet`] probes a URL list through it and
//! turns the result into the frontend's [`EngineConfig::Direct`]. A
//! [`DirectMembership`] adds and removes engines after startup, and
//! [`pods`] drives one from Kubernetes pods. Alternatively, an
//! [`EngineRegistrar`](discovery::EngineRegistrar) registers each engine in
//! the frontend's discovery as a sidecar would. Each engine crate supplies the
//! backend; roles, instance identity and transport-fault mapping live here
//! once.

pub mod discovery;
pub mod pods;

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
use dynamo_runtime::protocols::EndpointId;
use dynamo_runtime::protocols::annotated::Annotated;
use tokio::sync::watch;

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
    /// endpoint to route over. The fleet is fixed; see
    /// [`into_membership`](Self::into_membership) to change it later.
    pub fn into_engine_config(self, model: LocalModel) -> EngineConfig {
        self.into_membership(model).0
    }

    /// The frontend's engine configuration for `model` over this fleet, plus
    /// the handle that adds and removes engines while it serves.
    ///
    /// Whether the deployment is disaggregated is fixed here: by the presence
    /// of a prefill engine in the fleet.
    pub fn into_membership(self, model: LocalModel) -> (EngineConfig, DirectMembership<B>) {
        let primary = self.primary();
        let model_path = primary.model_path.clone();
        let disaggregated = self
            .engines
            .iter()
            .any(|(facts, _)| facts.disaggregation_mode.is_prefill());
        let endpoint_id = model.endpoint_id().clone();
        let workers = self
            .engines
            .into_iter()
            .map(|(facts, instance_id)| direct_worker(&endpoint_id, facts, instance_id))
            .collect();
        let (sender, workers) = watch::channel(workers);
        let membership = DirectMembership {
            backend: self.backend.clone(),
            endpoint_id,
            model_path,
            disaggregated,
            workers: sender,
        };
        let config = EngineConfig::Direct {
            model: Box::new(model),
            workers,
            dispatch: self.backend,
        };
        (config, membership)
    }
}

/// Adds and removes engines behind a serving direct frontend.
///
/// An added engine is probed, checked against the fleet's model and role, and
/// registered with the backend before routing can select it. A removed engine
/// leaves routing and the selection catalog; the backend drops its dispatch
/// state when the router reports the instance removed. Dropping the
/// membership freezes the engine list.
pub struct DirectMembership<B: DirectBackend> {
    backend: Arc<B>,
    endpoint_id: EndpointId,
    model_path: String,
    disaggregated: bool,
    workers: watch::Sender<Vec<DirectWorker>>,
}

impl<B: DirectBackend> DirectMembership<B> {
    /// Probe `url` and make it routable. Adding an engine that is already
    /// routable is a no-op.
    pub async fn add(&self, url: &str, role: RoleHint) -> Result<u64, DynamoError> {
        let instance_id = instance_id_for_url(url);
        if self.contains(instance_id) {
            return Ok(instance_id);
        }
        let mut engine = self.backend.probe(url, role).await?;
        let facts = B::facts_mut(&mut engine);
        if facts.model_path != self.model_path {
            return Err(invalid_argument(format!(
                "direct engines must serve one model: {} serves {}, the fleet serves {}",
                facts.endpoint, facts.model_path, self.model_path
            )));
        }
        if !B::REPORTS_ROLE
            && self.disaggregated
            && facts.disaggregation_mode == DisaggregationMode::Aggregated
        {
            facts.disaggregation_mode = DisaggregationMode::Decode;
        }
        if !self.disaggregated && facts.disaggregation_mode.is_prefill() {
            return Err(invalid_argument(format!(
                "{} is a prefill engine, but this direct frontend started without one",
                facts.endpoint
            )));
        }
        let worker = direct_worker(&self.endpoint_id, facts.clone(), instance_id);
        self.backend.register(instance_id, engine);
        self.workers.send_modify(|workers| {
            workers.retain(|existing| existing.instance.id() != instance_id);
            workers.push(worker);
        });
        Ok(instance_id)
    }

    /// Withdraw `url` from routing. Returns whether it was routable.
    pub fn remove(&self, url: &str) -> bool {
        let instance_id = instance_id_for_url(url);
        self.workers.send_if_modified(|workers| {
            let before = workers.len();
            workers.retain(|worker| worker.instance.id() != instance_id);
            workers.len() != before
        })
    }

    /// Whether `instance_id` is routable.
    pub fn contains(&self, instance_id: u64) -> bool {
        self.workers
            .borrow()
            .iter()
            .any(|worker| worker.instance.id() == instance_id)
    }

    /// Instance ids of every routable engine.
    pub fn instance_ids(&self) -> Vec<u64> {
        self.workers
            .borrow()
            .iter()
            .map(|worker| worker.instance.id())
            .collect()
    }
}

/// The routable identity of an engine probed for the model at `endpoint_id`.
fn direct_worker(
    endpoint_id: &EndpointId,
    facts: DirectEngineFacts,
    instance_id: u64,
) -> DirectWorker {
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
pub(crate) mod tests {
    use super::*;
    use std::sync::Mutex;

    use dynamo_llm::local_model::LocalModelBuilder;
    use dynamo_runtime::pipeline::{AddressedRequest, Error, ManyIn, ManyOut, SingleIn};

    /// Engine family whose probe answers from memory. Engines cannot report
    /// roles, so the operator's hint decides.
    pub(crate) struct FakeBackend {
        state: Mutex<FakeState>,
    }

    #[derive(Default)]
    struct FakeState {
        probes: HashMap<String, usize>,
        failures: HashMap<String, usize>,
        models: HashMap<String, String>,
        registered: Vec<u64>,
    }

    impl FakeBackend {
        pub(crate) fn new() -> Arc<Self> {
            Arc::new(Self {
                state: Mutex::new(FakeState::default()),
            })
        }

        /// Fail the next `n` probes of `url`.
        pub(crate) fn fail_next(&self, url: &str, n: usize) {
            self.state
                .lock()
                .unwrap()
                .failures
                .insert(url.to_string(), n);
        }

        /// Make `url` serve `model` instead of the fleet's.
        pub(crate) fn serve(&self, url: &str, model: &str) {
            let mut state = self.state.lock().unwrap();
            state.models.insert(url.to_string(), model.to_string());
        }

        pub(crate) fn probes(&self, url: &str) -> usize {
            self.state
                .lock()
                .unwrap()
                .probes
                .get(url)
                .copied()
                .unwrap_or(0)
        }

        pub(crate) fn registered(&self) -> Vec<u64> {
            self.state.lock().unwrap().registered.clone()
        }
    }

    #[async_trait]
    impl DirectBackend for FakeBackend {
        type Engine = DirectEngineFacts;
        const NAME: &'static str = "fake";
        const REPORTS_ROLE: bool = false;

        async fn probe(&self, url: &str, role: RoleHint) -> Result<DirectEngineFacts, DynamoError> {
            let mut state = self.state.lock().unwrap();
            *state.probes.entry(url.to_string()).or_default() += 1;
            if let Some(failures) = state.failures.get_mut(url)
                && *failures > 0
            {
                *failures -= 1;
                return Err(DynamoError::builder()
                    .error_type(ErrorType::Backend(BackendError::CannotConnect))
                    .message(format!("{url} is not answering"))
                    .build());
            }
            Ok(DirectEngineFacts {
                endpoint: GrpcEndpoint::parse(url, "--engine-url")?,
                disaggregation_mode: match role {
                    RoleHint::Prefill => DisaggregationMode::Prefill,
                    RoleHint::Serving => DisaggregationMode::Aggregated,
                },
                model_path: state
                    .models
                    .get(url)
                    .cloned()
                    .unwrap_or_else(|| "org/model".to_string()),
                served_model_name: None,
                kv_cache_block_size: Some(16),
                runtime_config: ModelRuntimeConfig::default(),
            })
        }

        fn facts(engine: &DirectEngineFacts) -> &DirectEngineFacts {
            engine
        }

        fn facts_mut(engine: &mut DirectEngineFacts) -> &mut DirectEngineFacts {
            engine
        }

        fn register(&self, instance_id: u64, _engine: DirectEngineFacts) {
            self.state.lock().unwrap().registered.push(instance_id);
        }
    }

    #[async_trait]
    impl StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>> for FakeBackend {
        async fn generate(
            &self,
            _request: SingleIn<AddressedRequest<PreprocessedRequest>>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            Err(anyhow::anyhow!("the fake backend serves no requests"))
        }

        async fn generate_bidirectional(
            &self,
            _instance: Instance,
            _address: String,
            _input: ManyIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            Err(anyhow::anyhow!("the fake backend serves no requests"))
        }
    }

    /// Probe `urls` and `prefill_urls` into a frontend config and membership.
    pub(crate) async fn fleet(
        backend: &Arc<FakeBackend>,
        urls: &[&str],
        prefill_urls: &[&str],
    ) -> (EngineConfig, DirectMembership<FakeBackend>) {
        let urls: Vec<String> = urls.iter().map(|url| url.to_string()).collect();
        let prefill_urls: Vec<String> = prefill_urls.iter().map(|url| url.to_string()).collect();
        let fleet = DirectFleet::probe(backend.clone(), &urls, &prefill_urls)
            .await
            .expect("probe fake engines");
        let mut builder = LocalModelBuilder::default();
        builder.endpoint_id(Some(EndpointId {
            namespace: "ns".to_string(),
            component: "backend".to_string(),
            name: "generate".to_string(),
        }));
        let model = builder.build().await.expect("local model");
        fleet.into_membership(model)
    }

    fn workers(config: &EngineConfig) -> watch::Receiver<Vec<DirectWorker>> {
        match config {
            EngineConfig::Direct { workers, .. } => workers.clone(),
            _ => unreachable!("fleets build direct configs"),
        }
    }

    fn roles(workers: &watch::Receiver<Vec<DirectWorker>>) -> Vec<(String, WorkerType)> {
        workers
            .borrow()
            .iter()
            .map(|worker| {
                (
                    worker.instance.transport.address().to_string(),
                    worker.worker_type,
                )
            })
            .collect()
    }

    #[tokio::test]
    async fn added_engines_are_registered_before_they_become_routable() {
        let backend = FakeBackend::new();
        let (config, membership) = fleet(&backend, &["http://h:1"], &[]).await;
        let mut workers = workers(&config);
        workers.borrow_and_update();

        let id = membership
            .add("http://h:2", RoleHint::Serving)
            .await
            .unwrap();
        assert_eq!(id, instance_id_for_url("http://h:2"));
        assert!(workers.has_changed().unwrap());
        assert!(backend.registered().contains(&id));
        assert_eq!(
            roles(&workers),
            vec![
                ("http://h:1".to_string(), WorkerType::Aggregated),
                ("http://h:2".to_string(), WorkerType::Aggregated),
            ]
        );
        assert_eq!(membership.instance_ids().len(), 2);
    }

    #[tokio::test]
    async fn adding_a_routable_engine_does_not_probe_it_again() {
        let backend = FakeBackend::new();
        let (_config, membership) = fleet(&backend, &["http://h:1"], &[]).await;
        membership
            .add("http://h:1/", RoleHint::Serving)
            .await
            .unwrap();
        assert_eq!(backend.probes("http://h:1"), 1);
        assert_eq!(backend.probes("http://h:1/"), 0);
    }

    #[tokio::test]
    async fn removed_engines_leave_the_list() {
        let backend = FakeBackend::new();
        let (config, membership) = fleet(&backend, &["http://h:1", "http://h:2"], &[]).await;
        let workers = workers(&config);
        assert!(membership.remove("http://h:2"));
        assert!(!membership.remove("http://h:2"));
        assert_eq!(
            roles(&workers),
            vec![("http://h:1".to_string(), WorkerType::Aggregated)]
        );
    }

    #[tokio::test]
    async fn an_engine_serving_another_model_is_rejected() {
        let backend = FakeBackend::new();
        let (_config, membership) = fleet(&backend, &["http://h:1"], &[]).await;
        backend.serve("http://h:2", "org/other");
        let error = membership
            .add("http://h:2", RoleHint::Serving)
            .await
            .unwrap_err();
        assert!(error.message().contains("must serve one model"), "{error}");
        assert_eq!(membership.instance_ids().len(), 1);
    }

    #[tokio::test]
    async fn roles_follow_the_fleet_shape_fixed_at_startup() {
        let backend = FakeBackend::new();
        let (config, membership) = fleet(&backend, &["http://d:1"], &["http://p:1"]).await;
        let workers = workers(&config);
        membership
            .add("http://d:2", RoleHint::Serving)
            .await
            .unwrap();
        membership
            .add("http://p:2", RoleHint::Prefill)
            .await
            .unwrap();
        let mut roles = roles(&workers);
        roles.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            roles,
            vec![
                ("http://d:1".to_string(), WorkerType::Decode),
                ("http://d:2".to_string(), WorkerType::Decode),
                ("http://p:1".to_string(), WorkerType::Prefill),
                ("http://p:2".to_string(), WorkerType::Prefill),
            ]
        );

        let (_config, aggregated) = fleet(&backend, &["http://a:1"], &[]).await;
        let error = aggregated
            .add("http://p:3", RoleHint::Prefill)
            .await
            .unwrap_err();
        assert!(error.message().contains("started without one"), "{error}");
    }

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
