// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct dispatch: a second request transport beside the Dynamo request plane.
//!
//! A worker that serves a native engine RPC (for example vLLM's gRPC service)
//! advertises that endpoint in its runtime config under
//! [`NATIVE_GRPC_ENDPOINT_RUNTIME_KEY`]. When a [`DirectEngineFactory`] is
//! installed, the [`DirectDispatchRegistry`] connects one [`DirectEngine`] per
//! such worker and `RoutingHost` sends admitted requests to it instead of the
//! request plane. Selection, booking, cancellation, and the error taxonomy the
//! migration layer keys on are unchanged: the direct engine returns the same
//! `Annotated<LLMEngineOutput>` stream and `DynamoError` chain the request
//! plane does, so a dispatch failure releases the booking and reaches the
//! retry manager exactly as before.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use dynamo_kv_router::protocols::WorkerId;
use dynamo_runtime::pipeline::{AsyncEngine, Error, ManyOut, SingleIn, async_trait};
use dynamo_runtime::protocols::annotated::Annotated;
use parking_lot::RwLock;
use tokio_util::sync::CancellationToken;

use crate::discovery::RuntimeConfigWatch;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::preprocessor::PreprocessedRequest;
use crate::protocols::common::llm_backend::LLMEngineOutput;

/// `ModelRuntimeConfig.runtime_data` key under which a worker advertises the
/// engine endpoint a frontend may dispatch to directly.
pub const NATIVE_GRPC_ENDPOINT_RUNTIME_KEY: &str = "native_grpc_endpoint";

/// `ModelRuntimeConfig.runtime_data` key naming the disaggregation role the
/// directly reachable engine serves (`agg`, `prefill`, `decode`).
pub const NATIVE_GRPC_MODE_RUNTIME_KEY: &str = "native_grpc_mode";

/// A worker reachable without the request plane. Same request and response
/// shape as the request-plane client, so `RoutingHost` treats both alike.
pub type DirectEngine =
    Arc<dyn AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>>;

/// Builds a [`DirectEngine`] for a worker from its advertised endpoint.
///
/// Implementations live with the engine protocol they speak (the vLLM sidecar
/// crate implements it over vLLM's gRPC service) and are installed by the
/// process that assembles the frontend.
#[async_trait]
pub trait DirectEngineFactory: Send + Sync {
    async fn connect(
        &self,
        worker_id: WorkerId,
        endpoint: &str,
        config: &ModelRuntimeConfig,
    ) -> anyhow::Result<DirectEngine>;
}

/// The advertised direct endpoint of a worker, if any.
pub fn native_grpc_endpoint(config: &ModelRuntimeConfig) -> Option<&str> {
    config
        .runtime_data
        .get(NATIVE_GRPC_ENDPOINT_RUNTIME_KEY)?
        .as_str()
        .filter(|endpoint| !endpoint.is_empty())
}

static INSTALLED_FACTORY: OnceLock<Arc<dyn DirectEngineFactory>> = OnceLock::new();

/// Install the process-wide factory the frontend assembly uses for every
/// routing host it builds. Returns `false` if one is already installed.
pub fn install_direct_engine_factory(factory: Arc<dyn DirectEngineFactory>) -> bool {
    INSTALLED_FACTORY.set(factory).is_ok()
}

/// The installed factory, if any.
pub fn installed_direct_engine_factory() -> Option<Arc<dyn DirectEngineFactory>> {
    INSTALLED_FACTORY.get().cloned()
}

/// Worker-keyed set of directly reachable engines.
#[derive(Default)]
pub struct DirectDispatchRegistry {
    engines: RwLock<HashMap<WorkerId, (String, DirectEngine)>>,
}

impl DirectDispatchRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, worker_id: WorkerId, endpoint: impl Into<String>, engine: DirectEngine) {
        self.engines
            .write()
            .insert(worker_id, (endpoint.into(), engine));
    }

    pub fn remove(&self, worker_id: WorkerId) -> Option<DirectEngine> {
        self.engines.write().remove(&worker_id).map(|(_, e)| e)
    }

    /// The engine for `worker_id`, if it is directly reachable.
    pub fn get(&self, worker_id: WorkerId) -> Option<DirectEngine> {
        self.engines
            .read()
            .get(&worker_id)
            .map(|(_, engine)| Arc::clone(engine))
    }

    pub fn endpoint(&self, worker_id: WorkerId) -> Option<String> {
        self.engines
            .read()
            .get(&worker_id)
            .map(|(endpoint, _)| endpoint.clone())
    }

    pub fn len(&self) -> usize {
        self.engines.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.engines.read().is_empty()
    }

    /// Keep the registry in step with discovery: connect every worker that
    /// advertises a direct endpoint (retrying failed connections on the next
    /// membership change), and drop workers that leave or stop advertising.
    pub fn spawn_feeder(
        self: &Arc<Self>,
        factory: Arc<dyn DirectEngineFactory>,
        mut watch: RuntimeConfigWatch,
        cancel: CancellationToken,
    ) {
        let registry = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                let snapshot = watch.borrow_and_update().clone();
                registry.reconcile(&factory, &snapshot).await;
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    changed = watch.changed() => {
                        if changed.is_err() {
                            break;
                        }
                    }
                }
            }
        });
    }

    async fn reconcile(
        &self,
        factory: &Arc<dyn DirectEngineFactory>,
        snapshot: &HashMap<WorkerId, ModelRuntimeConfig>,
    ) {
        let desired: HashMap<WorkerId, (&str, &ModelRuntimeConfig)> = snapshot
            .iter()
            .filter_map(|(worker_id, config)| {
                native_grpc_endpoint(config).map(|endpoint| (*worker_id, (endpoint, config)))
            })
            .collect();

        let stale: Vec<WorkerId> = self
            .engines
            .read()
            .iter()
            .filter(|(worker_id, (endpoint, _))| {
                desired
                    .get(worker_id)
                    .is_none_or(|(desired_endpoint, _)| desired_endpoint != endpoint)
            })
            .map(|(worker_id, _)| *worker_id)
            .collect();
        for worker_id in stale {
            self.remove(worker_id);
            tracing::info!(worker_id, "direct dispatch: dropped engine");
        }

        for (worker_id, (endpoint, config)) in desired {
            if self.engines.read().contains_key(&worker_id) {
                continue;
            }
            match factory.connect(worker_id, endpoint, config).await {
                Ok(engine) => {
                    tracing::info!(worker_id, endpoint, "direct dispatch: connected engine");
                    self.insert(worker_id, endpoint, engine);
                }
                Err(error) => {
                    tracing::warn!(
                        worker_id,
                        endpoint,
                        %error,
                        "direct dispatch: connect failed; requests use the request plane until the next membership change"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advertised_endpoint_requires_non_empty_string() {
        let mut config = ModelRuntimeConfig::default();
        assert_eq!(native_grpc_endpoint(&config), None);
        config.runtime_data.insert(
            NATIVE_GRPC_ENDPOINT_RUNTIME_KEY.to_string(),
            serde_json::Value::String(String::new()),
        );
        assert_eq!(native_grpc_endpoint(&config), None);
        config.runtime_data.insert(
            NATIVE_GRPC_ENDPOINT_RUNTIME_KEY.to_string(),
            serde_json::Value::String("http://worker:50051".to_string()),
        );
        assert_eq!(native_grpc_endpoint(&config), Some("http://worker:50051"));
    }
}
