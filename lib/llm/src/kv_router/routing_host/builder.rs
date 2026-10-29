// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Construction of a [`RoutingHost`].
//!
//! One builder serves every routing mode and every host. The frontend, the
//! prefill binding, the Python bindings and tests all describe what they have
//! (a discovery client, a routing mode, the load context to start or one they
//! hold, the [`KvRouter`] to construct or one they hold, affinity, LoRA, a
//! transport override) and the builder derives the host's policy, its
//! transport and its capability checks from that in one place.

use std::{sync::Arc, time::Duration};

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_router_types::prefill_load::PrefillLoadEstimator;
use dynamo_runtime::{
    component::Client,
    engine::EngineContextGuard,
    pipeline::{
        MultimodalCacheIndex, MultimodalCacheKeyExtractor, PushRouter, PushRouterOptions,
        RouterMode, StreamingDispatch, WorkerLoadMonitor,
    },
    protocols::annotated::Annotated,
};
use dynamo_worker_selection::selector::WorkerInputs;
use tokio_util::sync::CancellationToken;

use super::{
    HostedOccupancy, LoraRouting, RoutingHost, RoutingPolicy,
    builtin::{builtin_inputs, builtin_picker},
};
use crate::{
    discovery::{LoadThresholdHandle, ModelManager},
    kv_router::{
        KvRouter, RouterLoadSource, RoutingLoadContext, SelectionPolicySource,
        metrics::RouterRequestMetrics, plugins::RouterPluginBuilder,
    },
    lora::{LoadEstimator, LoraFilter},
    preprocessor::PreprocessedRequest,
    protocols::common::llm_backend::LLMEngineOutput,
    session_affinity::{AffinityCoordinator, SessionAffinityMode, create_affinity_coordinator},
    worker_type::WorkerType,
};

/// The transport a [`RoutingHost`] sends through.
pub type LlmPushRouter = PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>;

/// A caller-supplied final hop in place of the request plane.
pub type LlmDispatch = Arc<dyn StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>>>;

enum Affinity {
    None,
    /// A coordinator the caller already built; used as is.
    Coordinator(AffinityCoordinator),
    /// Built here: from the [`KvRouter`] under KV routing, standalone otherwise,
    /// with replica sync enabled on the host's client.
    Session {
        ttl: Duration,
        mode: SessionAffinityMode,
    },
}

/// What [`RoutingHostBuilder::load`] starts: the host's load context, which
/// owns the shared client, overload monitoring for `source`, and the
/// cancellation subtree every router task of the host lives under.
pub struct LoadContextSpec {
    pub source: RouterLoadSource,
    pub thresholds: LoadThresholdHandle,
    pub parent_token: CancellationToken,
    pub task_guard: Option<EngineContextGuard>,
}

/// How the KV router the builder constructs chooses its selection policy.
pub enum KvPolicy {
    /// The plugin bundle; its policy is validated against the config and the
    /// bundle is installed on the router after construction.
    Plugins(RouterPluginBuilder),
    /// A policy source the caller already resolved.
    Source(SelectionPolicySource),
}

/// What [`RoutingHostBuilder::kv`] constructs: a [`KvRouter`] through the
/// model manager, on the host's load context.
pub struct KvRouterSpec {
    pub model_manager: Arc<ModelManager>,
    pub block_size: u32,
    pub policy: KvPolicy,
    pub kv_router_config: Option<KvRouterConfig>,
    pub prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    pub worker_role: Option<WorkerType>,
    pub metric_worker_type: &'static str,
    pub model_name: Option<String>,
    pub is_eagle: bool,
    /// Keeps the router's teardown tasks alive as long as the owning worker set.
    pub teardown_task_guard: Option<EngineContextGuard>,
}

/// Builds a [`RoutingHost`]. Obtain one from [`RoutingHost::builder`].
pub struct RoutingHostBuilder {
    client: Client,
    router_mode: RouterMode,
    kv_router: Option<Arc<KvRouter>>,
    load_context: Option<Arc<RoutingLoadContext>>,
    load: Option<LoadContextSpec>,
    kv: Option<KvRouterSpec>,
    affinity: Affinity,
    /// Whether a session affinity built here syncs its bindings across replicas.
    session_replication: bool,
    lora: Option<(Arc<LoraFilter>, Arc<LoadEstimator>)>,
    dispatch: Option<LlmDispatch>,
    worker_monitor: Option<Arc<dyn WorkerLoadMonitor>>,
    multimodal_cache: Option<(
        Arc<dyn MultimodalCacheIndex>,
        MultimodalCacheKeyExtractor<PreprocessedRequest>,
    )>,
    /// A transport built by the caller. When set, `build` wires the host over
    /// it instead of constructing a [`PushRouter`] from `client`.
    transport: Option<LlmPushRouter>,
}

impl RoutingHost {
    /// Start building a host that discovers workers through `client` and selects
    /// them with `router_mode`. KV routing also needs [`RoutingHostBuilder::kv_router`].
    pub fn builder(client: Client, router_mode: RouterMode) -> RoutingHostBuilder {
        RoutingHostBuilder {
            client,
            router_mode,
            kv_router: None,
            load_context: None,
            load: None,
            kv: None,
            affinity: Affinity::None,
            session_replication: true,
            lora: None,
            dispatch: None,
            worker_monitor: None,
            multimodal_cache: None,
            transport: None,
        }
    }
}

impl RoutingHostBuilder {
    /// A builder over a transport the caller already constructed. The client and
    /// routing mode come from the transport.
    pub(crate) fn from_transport(transport: LlmPushRouter) -> Self {
        let mut builder = RoutingHost::builder(transport.client.clone(), transport.router_mode());
        builder.transport = Some(transport);
        builder
    }

    /// The KV-aware selection plane. Required for [`RouterMode::KV`], rejected otherwise.
    pub fn kv_router(mut self, kv_router: impl Into<Option<Arc<KvRouter>>>) -> Self {
        self.kv_router = kv_router.into();
        self
    }

    /// The shared client, overload state and cancellation subtree this host
    /// belongs to, when the caller already started one. Required for every
    /// mode except KV unless [`Self::load`] starts it.
    pub fn load_context(
        mut self,
        load_context: impl Into<Option<Arc<RoutingLoadContext>>>,
    ) -> Self {
        self.load_context = load_context.into();
        self
    }

    /// Start the host's load context from `spec` during [`Self::build`].
    pub fn load(mut self, spec: LoadContextSpec) -> Self {
        self.load = Some(spec);
        self
    }

    /// Construct the host's [`KvRouter`] from `spec` during [`Self::build`], on
    /// the load context the builder starts or was given.
    pub fn kv(mut self, spec: KvRouterSpec) -> Self {
        self.kv = Some(spec);
        self
    }

    /// Whether a session affinity built by [`Self::session_affinity`] syncs its
    /// bindings across router replicas over the event plane. On by default; a
    /// standalone router with no peers turns it off.
    pub fn session_affinity_replication(mut self, enabled: bool) -> Self {
        self.session_replication = enabled;
        self
    }

    /// Session affinity with a coordinator the caller built. `None` disables it.
    pub fn affinity_coordinator(
        mut self,
        affinity: impl Into<Option<AffinityCoordinator>>,
    ) -> Self {
        self.affinity = match affinity.into() {
            Some(coordinator) => Affinity::Coordinator(coordinator),
            None => Affinity::None,
        };
        self
    }

    /// Session affinity built by the host: a KV host derives the coordinator from
    /// its [`KvRouter`], any other host builds a standalone one. Replica sync is
    /// enabled on the host's client either way. `None` disables it.
    pub fn session_affinity(mut self, ttl: Option<Duration>, mode: SessionAffinityMode) -> Self {
        self.affinity = match ttl {
            Some(ttl) => Affinity::Session { ttl, mode },
            None => Affinity::None,
        };
        self
    }

    /// LoRA-aware selection for the round-robin and random modes.
    pub fn lora(mut self, lora: impl Into<Option<(Arc<LoraFilter>, Arc<LoadEstimator>)>>) -> Self {
        self.lora = lora.into();
        self
    }

    /// Send through `dispatch` instead of the request plane. A host that does not
    /// send requests itself, such as a gateway endpoint picker, supplies the hop
    /// that records the selection and relays observed progress.
    pub fn dispatch(mut self, dispatch: impl Into<Option<LlmDispatch>>) -> Self {
        self.dispatch = dispatch.into();
        self
    }

    /// Durable overload detection for the transport.
    pub fn worker_monitor(
        mut self,
        monitor: impl Into<Option<Arc<dyn WorkerLoadMonitor>>>,
    ) -> Self {
        self.worker_monitor = monitor.into();
        self
    }

    /// Multimodal embedding-cache state for [`RouterMode::DeviceAwareWeighted`].
    pub fn multimodal_cache(
        mut self,
        indexer: Arc<dyn MultimodalCacheIndex>,
        key_extractor: MultimodalCacheKeyExtractor<PreprocessedRequest>,
    ) -> Self {
        self.multimodal_cache = Some((indexer, key_extractor));
        self
    }

    /// Build the host: start the load context and construct the KV router when
    /// asked to, then the transport unless one was supplied.
    pub async fn build(self) -> anyhow::Result<RoutingHost> {
        let Self {
            client,
            router_mode,
            kv_router,
            load_context,
            load,
            kv,
            affinity,
            session_replication,
            lora,
            dispatch,
            worker_monitor,
            multimodal_cache,
            transport,
        } = self;
        let load_context = match (load_context, load) {
            (Some(context), None) => Some(context),
            (None, Some(spec)) => Some(
                RoutingLoadContext::start(
                    client.clone(),
                    spec.source,
                    spec.thresholds,
                    &spec.parent_token,
                    spec.task_guard,
                )
                .await?,
            ),
            (None, None) => None,
            (Some(_), Some(_)) => {
                anyhow::bail!("pass either a load context or the parameters to start one")
            }
        };
        let kv_router = match (kv_router, kv) {
            (Some(router), None) => Some(router),
            (None, Some(spec)) => Some(construct_kv_router(spec, load_context.as_ref()).await?),
            (None, None) => None,
            (Some(_), Some(_)) => {
                anyhow::bail!("pass either a KvRouter or the parameters to construct one")
            }
        };
        validate_policy_inputs(router_mode, kv_router.as_ref(), load_context.as_ref())?;

        let affinity = match affinity {
            Affinity::None => None,
            Affinity::Coordinator(coordinator) => Some(coordinator),
            Affinity::Session { ttl, mode } => match kv_router.as_ref() {
                Some(kv_router) => {
                    let coordinator = kv_router.affinity_coordinator(ttl, mode)?;
                    if session_replication {
                        coordinator.enable_replica_sync(client.clone()).await?;
                    }
                    Some(coordinator)
                }
                None => create_affinity_coordinator(Some(ttl), mode, client.clone()).await?,
            },
        };

        let transport = match transport {
            Some(transport) => {
                if transport.router_mode() != router_mode {
                    anyhow::bail!(
                        "transport routes with {:?} but the host was asked for {router_mode:?}",
                        transport.router_mode()
                    );
                }
                transport
            }
            None => {
                let (multimodal_cache_indexer, multimodal_cache_key_extractor) =
                    match multimodal_cache {
                        Some((indexer, extractor)) => (Some(indexer), Some(extractor)),
                        None => (None, None),
                    };
                LlmPushRouter::from_client_with_options(
                    client,
                    router_mode,
                    PushRouterOptions {
                        worker_monitor,
                        multimodal_cache_indexer,
                        multimodal_cache_key_extractor,
                        dispatch,
                        ..PushRouterOptions::default()
                    },
                )
                .await?
            }
        };

        assemble(transport, kv_router, load_context, affinity, lora)
    }

    /// [`Self::build`] for a builder that already holds its transport, so no await
    /// is needed. Fails if no transport was supplied or if the affinity still has
    /// to be built.
    pub(crate) fn build_sync(self) -> anyhow::Result<RoutingHost> {
        let Self {
            router_mode,
            kv_router,
            load_context,
            affinity,
            lora,
            transport,
            ..
        } = self;
        let Some(transport) = transport else {
            anyhow::bail!("build_sync requires a transport; use build() to construct one");
        };
        if transport.router_mode() != router_mode {
            anyhow::bail!(
                "transport routes with {:?} but the host was asked for {router_mode:?}",
                transport.router_mode()
            );
        }
        validate_policy_inputs(router_mode, kv_router.as_ref(), load_context.as_ref())?;
        let affinity = match affinity {
            Affinity::None => None,
            Affinity::Coordinator(coordinator) => Some(coordinator),
            Affinity::Session { .. } => {
                anyhow::bail!("build_sync cannot build session affinity; pass a coordinator")
            }
        };
        assemble(transport, kv_router, load_context, affinity, lora)
    }
}

/// Construct the KV router a [`KvRouterSpec`] describes on the host's load
/// context: the model manager owns the per-endpoint topology, shared caches,
/// and KV-source membership the router joins.
async fn construct_kv_router(
    spec: KvRouterSpec,
    load_context: Option<&Arc<RoutingLoadContext>>,
) -> anyhow::Result<Arc<KvRouter>> {
    let load_context = load_context
        .ok_or_else(|| anyhow::anyhow!("constructing a KvRouter requires a load context"))?;
    let KvRouterSpec {
        model_manager,
        block_size,
        policy,
        kv_router_config,
        prefill_load_estimator,
        worker_role,
        metric_worker_type,
        model_name,
        is_eagle,
        teardown_task_guard,
    } = spec;
    let client = load_context.client().clone();
    let scheduler_load = load_context.scheduler_load_sender();
    let cancellation_token = load_context.cancellation_token();
    let mut router = match policy {
        KvPolicy::Plugins(plugins) => {
            model_manager
                .kv_chooser_for_with_plugins_and_client(
                    client,
                    block_size,
                    &plugins,
                    kv_router_config,
                    prefill_load_estimator,
                    worker_role,
                    metric_worker_type,
                    model_name,
                    is_eagle,
                    scheduler_load,
                    cancellation_token,
                )
                .await?
        }
        KvPolicy::Source(policy) => {
            model_manager
                .kv_chooser_for_with_policy_and_client(
                    client,
                    block_size,
                    policy,
                    kv_router_config,
                    prefill_load_estimator,
                    worker_role,
                    metric_worker_type,
                    model_name,
                    is_eagle,
                    scheduler_load,
                    cancellation_token,
                )
                .await?
        }
    };
    if let Some(guard) = teardown_task_guard {
        Arc::get_mut(&mut router)
            .expect("new KV chooser must have one owner")
            .set_teardown_task_guard(guard);
    }
    Ok(router)
}

fn validate_policy_inputs(
    router_mode: RouterMode,
    kv_router: Option<&Arc<KvRouter>>,
    load_context: Option<&Arc<RoutingLoadContext>>,
) -> anyhow::Result<()> {
    match (router_mode, kv_router) {
        (RouterMode::KV, None) => anyhow::bail!("RouterMode::KV requires a KvRouter"),
        (RouterMode::KV, Some(_)) => {}
        (mode, Some(_)) => {
            anyhow::bail!("a KvRouter is only used with RouterMode::KV, not {mode:?}")
        }
        (mode, None) if load_context.is_none() => {
            anyhow::bail!("{mode:?} routing requires a RoutingLoadContext")
        }
        (_, None) => {}
    }
    Ok(())
}

/// Wire a host over an existing transport. This is the one place that turns
/// construction inputs into a [`RoutingPolicy`] and its capabilities.
fn assemble(
    inner: LlmPushRouter,
    kv_router: Option<Arc<KvRouter>>,
    load_context: Option<Arc<RoutingLoadContext>>,
    affinity: Option<AffinityCoordinator>,
    lora: Option<(Arc<LoraFilter>, Arc<LoadEstimator>)>,
) -> anyhow::Result<RoutingHost> {
    let session_affinity_mode = affinity
        .as_ref()
        .map(AffinityCoordinator::mode)
        .unwrap_or_default();

    if let Some(kv_router) = kv_router {
        if lora.is_some() {
            anyhow::bail!("LoRA filtering is unsupported with KV routing");
        }
        // Eagerly register router request metrics (as zeros) so they are
        // scrapeable before any requests arrive. Both the frontend pipeline
        // and the standalone router create RoutingHost, so this covers both.
        let request_metrics =
            RouterRequestMetrics::from_component(kv_router.client().endpoint.component());
        return Ok(RoutingHost {
            inner,
            policy: RoutingPolicy::Kv(kv_router),
            request_metrics,
            session_affinity_mode,
            affinity,
            hosted_occupancy: None,
            lora: None,
            routing_context: load_context,
        });
    }

    let Some(load_context) = load_context else {
        anyhow::bail!(
            "{:?} routing requires a RoutingLoadContext",
            inner.router_mode()
        );
    };
    if affinity.is_some() && lora.is_some() {
        anyhow::bail!("session affinity and LoRA filtering cannot both be enabled");
    }
    let mode = inner.router_mode();
    let policy = match mode {
        RouterMode::Direct => RoutingPolicy::Direct,
        RouterMode::DeviceAwareWeighted => RoutingPolicy::DeviceAwareWeighted,
        mode => RoutingPolicy::Builtin(
            builtin_picker(mode)
                .ok_or_else(|| anyhow::anyhow!("{mode:?} routing is not a first-party policy"))?,
        ),
    };
    let required_worker_inputs = match &policy {
        RoutingPolicy::Builtin(_) => builtin_inputs(mode),
        RoutingPolicy::DeviceAwareWeighted => WorkerInputs::OCCUPANCY,
        RoutingPolicy::Direct => WorkerInputs::NONE,
        RoutingPolicy::Kv(_) => unreachable!("KV hosts are assembled above"),
    };
    let hosted_occupancy = (matches!(&policy, RoutingPolicy::Builtin(_))
        && required_worker_inputs.contains(WorkerInputs::OCCUPANCY))
    .then(|| HostedOccupancy::new(&inner))
    .transpose()?;
    if lora.is_some() && !matches!(mode, RouterMode::RoundRobin | RouterMode::Random) {
        anyhow::bail!("LoRA filtering is unsupported with {mode:?} routing");
    }
    let lora = lora.map(|(filter, load_estimator)| LoraRouting {
        filter,
        load_estimator,
        selector: builtin_picker(mode).expect("LoRA routing mode was validated above"),
    });
    let request_metrics = RouterRequestMetrics::from_component(inner.client.endpoint.component());
    Ok(RoutingHost {
        inner,
        policy,
        request_metrics,
        session_affinity_mode,
        affinity,
        hosted_occupancy,
        lora,
        routing_context: Some(load_context),
    })
}
