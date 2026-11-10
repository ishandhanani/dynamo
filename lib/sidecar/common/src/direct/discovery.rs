// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct engines published to the frontend's own discovery.
//!
//! An [`EngineRegistrar`] does in the frontend what a sidecar does next to its
//! engine: it probes the engine, builds the model card from what the engine
//! reports, and registers the engine's endpoint instance and card. The
//! frontend's ordinary model watcher then serves the model, with the final
//! hop dialing the engine through the [`DirectBackend`]. Prefill engines may
//! come and go like any other, because the watcher, not a fixed list, decides
//! the topology.
//!
//! Each engine registers under the id the Kubernetes discovery backend gives
//! its pod, [`hash_pod_name`]; the URL is only where it listens.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::{BackendError, DisaggregationMode, DynamoError};
use dynamo_llm::entrypoint::input::direct::PREFILL_COMPONENT;
use dynamo_llm::local_model::{LocalModel, LocalModelBuilder, register_model_card_as};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::model_type::{ModelInput, ModelType};
use dynamo_llm::worker_type::WorkerType;
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::component::{Endpoint, TransportType};
use dynamo_runtime::discovery::{DiscoveryInstance, DiscoverySpec, hash_pod_name};
use dynamo_runtime::error::ErrorType;
use dynamo_runtime::protocols::EndpointId;
use dynamo_runtime::traits::DistributedRuntimeProvider;

use super::pods::EngineSet;
use super::{DirectBackend, DirectEngineFacts, RoleHint};

/// What the frontend's own flags say about every engine's model card. The
/// operator's names win over anything an engine reports, as the sidecar's
/// `--dyn-*` flags do.
#[derive(Clone, Debug, Default)]
pub struct CardOverrides {
    pub model_name: Option<String>,
    pub tool_call_parser: Option<String>,
    pub reasoning_parser: Option<String>,
}

/// Registers direct engines in the frontend's discovery, one endpoint instance
/// and model card per engine.
pub struct EngineRegistrar<B: DirectBackend> {
    backend: Arc<B>,
    serving: Endpoint,
    prefill: Endpoint,
    /// Whether a prefill pool exists, which makes engines that cannot report
    /// their role decode engines rather than aggregated ones.
    disaggregated: bool,
    overrides: CardOverrides,
    /// Local copy of each model's metadata files, by the path engines report.
    fetched: tokio::sync::Mutex<HashMap<String, PathBuf>>,
    registered: tokio::sync::Mutex<HashMap<String, Vec<DiscoveryInstance>>>,
}

impl<B: DirectBackend> EngineRegistrar<B> {
    /// Serving engines register on `serving`; prefill engines on the
    /// [`PREFILL_COMPONENT`] of its namespace.
    pub fn new(
        drt: &DistributedRuntime,
        backend: Arc<B>,
        serving: &EndpointId,
        disaggregated: bool,
        overrides: CardOverrides,
    ) -> anyhow::Result<Self> {
        let namespace = drt.namespace(&serving.namespace)?;
        Ok(Self {
            backend,
            serving: namespace
                .component(&serving.component)?
                .endpoint(&serving.name),
            prefill: namespace
                .component(PREFILL_COMPONENT)?
                .endpoint(&serving.name),
            disaggregated,
            overrides,
            fetched: Default::default(),
            registered: Default::default(),
        })
    }

    /// The card a sidecar next to this engine would register.
    async fn card(&self, facts: &DirectEngineFacts) -> anyhow::Result<ModelDeploymentCard> {
        let model_path = self.fetch(&facts.model_path).await?;
        let mut runtime_config = facts.runtime_config.clone();
        if let Some(parser) = &self.overrides.tool_call_parser {
            runtime_config.tool_call_parser = Some(parser.clone());
        }
        if let Some(parser) = &self.overrides.reasoning_parser {
            runtime_config.reasoning_parser = Some(parser.clone());
        }
        let name = self
            .overrides
            .model_name
            .clone()
            .or_else(|| facts.served_model_name.clone())
            .unwrap_or_else(|| facts.model_path.clone());
        let mut builder = LocalModelBuilder::default();
        builder
            .model_path(model_path.clone())
            .model_name(Some(name))
            .kv_cache_block_size(facts.kv_cache_block_size)
            .runtime_config(runtime_config);
        if model_path.as_os_str() != facts.model_path.as_str() {
            builder.source_path(PathBuf::from(&facts.model_path));
        }
        let mut card = builder.build().await?.into_card();

        // As the sidecar registers each role: prefill engines expose no
        // OpenAI surface and carry the legacy prefill marker.
        let (worker_type, needs) = match facts.disaggregation_mode {
            DisaggregationMode::Prefill => (WorkerType::Prefill, vec![vec![WorkerType::Decode]]),
            DisaggregationMode::Decode => (WorkerType::Decode, vec![vec![WorkerType::Prefill]]),
            _ => (WorkerType::Aggregated, Vec::new()),
        };
        card.model_type = if worker_type == WorkerType::Prefill {
            ModelType::Prefill
        } else {
            ModelType::Chat | ModelType::Completions
        };
        card.model_input = ModelInput::Tokens;
        card.worker_type = Some(worker_type);
        card.needs = needs;
        Ok(card)
    }

    /// The model's metadata files on this host: the path itself when it
    /// exists here, else a download without weights.
    async fn fetch(&self, model_path: &str) -> anyhow::Result<PathBuf> {
        let mut fetched = self.fetched.lock().await;
        if let Some(path) = fetched.get(model_path) {
            return Ok(path.clone());
        }
        let path = if std::path::Path::new(model_path).exists() {
            PathBuf::from(model_path)
        } else {
            LocalModel::fetch(model_path, true).await?
        };
        fetched.insert(model_path.to_string(), path.clone());
        Ok(path)
    }

    /// Register `facts`' endpoint instance, then its card, under `instance_id`.
    async fn publish(
        &self,
        facts: &DirectEngineFacts,
        card: &ModelDeploymentCard,
        instance_id: u64,
    ) -> anyhow::Result<Vec<DiscoveryInstance>> {
        let endpoint = if facts.disaggregation_mode.is_prefill() {
            &self.prefill
        } else {
            &self.serving
        };
        let discovery = endpoint.drt().discovery();
        let spec = DiscoverySpec::Endpoint {
            namespace: endpoint.component().namespace().name().to_string(),
            component: endpoint.component().name().to_string(),
            endpoint: endpoint.name().to_string(),
            transport: TransportType::Grpc(facts.endpoint.as_str().to_string()),
            device_type: None,
            request_plane_codec: None,
        };
        let instance = discovery.register_as(spec, instance_id).await?;
        match register_model_card_as(endpoint, card, instance_id).await {
            Ok(model) => Ok(vec![instance, model]),
            Err(error) => {
                if let Err(rollback) = discovery.unregister(instance).await {
                    tracing::warn!(%rollback, "direct engine endpoint left registered");
                }
                Err(error)
            }
        }
    }
}

#[async_trait]
impl<B: DirectBackend> EngineSet for EngineRegistrar<B> {
    async fn add(&self, pod: &str, url: &str, role: RoleHint) -> Result<u64, DynamoError> {
        let instance_id = hash_pod_name(pod);
        if self.registered.lock().await.contains_key(url) {
            return Ok(instance_id);
        }
        let mut engine = self.backend.probe(url, role).await?;
        let facts = B::facts_mut(&mut engine);
        if !B::REPORTS_ROLE
            && self.disaggregated
            && facts.disaggregation_mode == DisaggregationMode::Aggregated
        {
            facts.disaggregation_mode = DisaggregationMode::Decode;
        }
        let facts = facts.clone();
        let card = self
            .card(&facts)
            .await
            .map_err(|error| unknown(format!("model card for {}: {error:#}", facts.endpoint)))?;
        self.backend.register(instance_id, engine);
        let instances = self
            .publish(&facts, &card, instance_id)
            .await
            .map_err(|error| unknown(format!("register {}: {error:#}", facts.endpoint)))?;
        self.registered
            .lock()
            .await
            .insert(url.to_string(), instances);
        Ok(instance_id)
    }

    async fn remove(&self, url: &str) -> bool {
        let Some(instances) = self.registered.lock().await.remove(url) else {
            return false;
        };
        let discovery = self.serving.drt().discovery();
        // The card first, so the watcher never serves a model with no endpoint.
        for instance in instances.into_iter().rev() {
            if let Err(error) = discovery.unregister(instance).await {
                tracing::warn!(%url, %error, "direct engine left registered");
            }
        }
        true
    }
}

fn unknown(message: String) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::Unknown))
        .message(message)
        .build()
}
