// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serve a model from engines the frontend probed itself (`--direct`).
//!
//! The same routed pipeline the discovery-driven frontend builds per WorkerSet
//! (preprocessing, migration, prefill hop, KV routing) is assembled once here
//! from a static worker list. Membership comes from
//! [`DistributedRuntime::register_static_endpoint_instances`], the selection
//! catalog from a seeded runtime-config watch, and the final hop from the
//! caller's [`LlmDispatch`].

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Context as _;
use dynamo_kv_router::protocols::WorkerId;
use dynamo_renderer::PromptFormatter;
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::component::Instance;
use dynamo_runtime::pipeline::RouterMode;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::discovery::{
    CommittedWorkerSetTarget, LoadThresholdHandle, ModelManager, WorkerSetTarget,
};
use crate::entrypoint::DirectWorker;
use crate::entrypoint::input::common::build_preprocessed_routing_with_session_affinity_mode;
use crate::http::service::metrics::Metrics;
use crate::kv_router::{
    LlmDispatch, PrefillRouter, RouterLoadSource, RoutingLoadContext, SelectionPolicySource,
};
use crate::local_model::LocalModel;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::preprocessor::{OpenAIPreprocessor, prompt::prompt_formatter_from_mdc};
use crate::protocols::common::timing::WORKER_TYPE_DECODE;
use crate::types::openai::{
    chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
    completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
};
use crate::worker_type::WorkerType;

/// Component that holds the prefill engines of a direct deployment.
pub const PREFILL_COMPONENT: &str = "prefill";

/// Build and publish the chat and completions pipelines for `model` over `workers`.
pub async fn register_direct_model(
    drt: &DistributedRuntime,
    manager: Arc<ModelManager>,
    metrics: Arc<Metrics>,
    model: &LocalModel,
    workers: Vec<DirectWorker>,
    dispatch: LlmDispatch,
) -> anyhow::Result<()> {
    anyhow::ensure!(!workers.is_empty(), "direct mode needs at least one engine");
    manager.set_direct_dispatch(dispatch)?;

    let (prefill, decode): (Vec<_>, Vec<_>) = workers
        .into_iter()
        .partition(|worker| worker.worker_type == WorkerType::Prefill);
    anyhow::ensure!(
        !decode.is_empty(),
        "direct mode needs at least one aggregated or decode engine"
    );
    let decode_role = if prefill.is_empty() {
        WorkerType::Aggregated
    } else {
        WorkerType::Decode
    };

    let router_config = model.card().router_config.clone().unwrap_or_default();
    let router_mode = router_config.router_mode;
    let mut card = model.card().clone();
    card.worker_type = Some(decode_role);
    let model_name = card.display_name.clone();
    let endpoint_id = model.endpoint_id().clone();
    // Lifetime of everything built here: the frontend process.
    let cancellation = drt.primary_token().child_token();

    let decode_endpoint = drt
        .namespace(&endpoint_id.namespace)?
        .component(&endpoint_id.component)?
        .endpoint(&endpoint_id.name);
    install_static_workers(drt, &manager, &decode_endpoint, &decode, &cancellation);
    let client = decode_endpoint.client().await?;

    let load_thresholds = LoadThresholdHandle::new(router_config.load_threshold_config.clone());
    let load_context = RoutingLoadContext::start(
        client.clone(),
        RouterLoadSource::from_worker_type(decode_role),
        load_thresholds.clone(),
        &cancellation,
        None,
    )
    .await
    .context("RoutingLoadContext::start")?;

    let kv_chooser = if router_mode == RouterMode::KV {
        Some(
            manager
                .kv_chooser_for_with_policy_and_client(
                    client.clone(),
                    card.kv_cache_block_size,
                    SelectionPolicySource::Registry,
                    Some(router_config.kv_router_config.clone()),
                    None,
                    Some(decode_role),
                    WORKER_TYPE_DECODE,
                    Some(model_name.clone()),
                    card.runtime_config.enable_eagle,
                    load_context.scheduler_load_sender(),
                    load_context.cancellation_token(),
                )
                .await
                .context("kv_chooser_for_with_policy_and_client")?,
        )
    } else {
        None
    };

    let prefill_chooser = if prefill.is_empty() {
        None
    } else {
        let prefill_endpoint = drt
            .namespace(&endpoint_id.namespace)?
            .component(PREFILL_COMPONENT)?
            .endpoint(&endpoint_id.name);
        install_static_workers(drt, &manager, &prefill_endpoint, &prefill, &cancellation);

        let mut prefill_config = router_config.kv_router_config.clone();
        prefill_config.router_track_active_blocks = false;
        let router = PrefillRouter::new_with_selection_policy(
            None,
            manager.clone(),
            router_mode,
            card.kv_cache_block_size,
            Some(prefill_config),
            SelectionPolicySource::Registry,
            None,
            router_config.session_affinity_ttl_secs,
            router_config.session_affinity_mode,
            model_name.clone(),
            endpoint_id.namespace.clone(),
            load_thresholds.clone(),
            cancellation.child_token(),
            None,
        );
        // The prefill hop reads its role and advertisement from the committed
        // card, so hand it one shaped like a prefill worker's registration.
        let mut prefill_card = card.clone();
        prefill_card.worker_type = Some(WorkerType::Prefill);
        prefill_card.runtime_config = prefill[0].runtime_config.clone();
        let admitted = keep_alive(
            &cancellation,
            prefill.iter().map(|worker| worker.instance.id()).collect(),
        );
        router.set_target(Some(WorkerSetTarget::Committed(CommittedWorkerSetTarget {
            endpoint: prefill_endpoint,
            group: "direct-prefill".to_string(),
            generation: 0,
            card: Arc::new(prefill_card),
            admitted_ids: admitted,
        })));
        Some(router)
    };

    let routing = build_preprocessed_routing_with_session_affinity_mode(
        &client,
        manager.clone(),
        router_mode,
        load_context,
        kv_chooser,
        prefill_chooser,
        None,
        false,
        router_config.session_affinity_ttl_secs,
        router_config.session_affinity_mode,
    )
    .await
    .context("build_preprocessed_routing")?;

    let tokenizer = card.tokenizer().context("tokenizer")?;
    let checksum = card.mdcsum().to_string();
    let migration_limit = model.migration_limit();
    let migration_max_seq_len = model.migration_max_seq_len();

    let PromptFormatter::OAI(formatter) =
        prompt_formatter_from_mdc(&card).context("prompt_formatter_from_mdc")?;
    let chat_preprocessor =
        OpenAIPreprocessor::new_with_parts(card.clone(), formatter, tokenizer.clone())?;
    let chat = routing
        .build_pipeline::<NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse>(
            &card,
            chat_preprocessor,
            tokenizer.clone(),
            migration_limit,
            migration_max_seq_len,
            metrics.clone(),
        )?;
    manager.add_chat_completions_model(model.display_name(), &checksum, chat)?;

    let PromptFormatter::OAI(no_op) = PromptFormatter::no_op();
    let cmpl_preprocessor =
        OpenAIPreprocessor::new_with_parts(card.clone(), no_op, tokenizer.clone())?;
    let cmpl = routing.build_pipeline::<NvCreateCompletionRequest, NvCreateCompletionResponse>(
        &card,
        cmpl_preprocessor,
        tokenizer,
        migration_limit,
        migration_max_seq_len,
        metrics,
    )?;
    manager.add_completions_model(model.display_name(), &checksum, cmpl)?;

    tracing::info!(
        model = %model_name,
        decode_engines = decode.len(),
        prefill_engines = prefill.len(),
        router_mode = ?router_mode,
        "serving directly dialed engines"
    );
    Ok(())
}

/// Make `workers` the complete membership of `endpoint` for routing and for
/// the selection catalog.
fn install_static_workers(
    drt: &DistributedRuntime,
    manager: &ModelManager,
    endpoint: &dynamo_runtime::component::Endpoint,
    workers: &[DirectWorker],
    cancellation: &CancellationToken,
) {
    let instances: Vec<Instance> = workers.iter().map(|w| w.instance.clone()).collect();
    drt.register_static_endpoint_instances(endpoint, keep_alive(cancellation, instances));
    let configs: HashMap<WorkerId, ModelRuntimeConfig> = workers
        .iter()
        .map(|w| (w.instance.id(), w.runtime_config.clone()))
        .collect();
    manager.insert_runtime_config_watcher(endpoint, keep_alive(cancellation, configs));
}

/// A watch whose sender lives until `cancellation` fires, so receivers see a
/// stable value rather than a closed channel.
fn keep_alive<T: Send + Sync + 'static>(
    cancellation: &CancellationToken,
    value: T,
) -> watch::Receiver<T> {
    let (tx, rx) = watch::channel(value);
    let cancellation = cancellation.clone();
    tokio::spawn(async move {
        cancellation.cancelled().await;
        drop(tx);
    });
    rx
}
