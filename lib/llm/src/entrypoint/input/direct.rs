// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serve a model from engines the frontend probed itself (`--direct`).
//!
//! The same routed pipeline the discovery-driven frontend builds per WorkerSet
//! (preprocessing, migration, prefill hop, KV routing) is assembled once here
//! from a caller-owned worker list. Membership comes from
//! [`DistributedRuntime::register_static_endpoint_instances`], the selection
//! catalog from a runtime-config watch, and the final hop from the caller's
//! [`LlmDispatch`]. All three follow the list as engines come and go; whether
//! the deployment is disaggregated is fixed by the list at startup.

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
    workers: watch::Receiver<Vec<DirectWorker>>,
    dispatch: LlmDispatch,
) -> anyhow::Result<()> {
    let initial = workers.borrow().clone();
    anyhow::ensure!(!initial.is_empty(), "direct mode needs at least one engine");
    manager.set_direct_dispatch(dispatch)?;

    let (prefill, decode): (Vec<_>, Vec<_>) = initial
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
    install_workers(
        drt,
        &manager,
        &decode_endpoint,
        &workers,
        Role::Serving,
        &cancellation,
    );
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
        install_workers(
            drt,
            &manager,
            &prefill_endpoint,
            &workers,
            Role::Prefill,
            &cancellation,
        );

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
        let admitted = follow(&cancellation, workers.clone(), |workers| {
            Role::Prefill
                .select(workers)
                .map(|worker| worker.instance.id())
                .collect::<Vec<_>>()
        });
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

/// Which engines of the list an endpoint serves.
#[derive(Clone, Copy, Debug)]
enum Role {
    /// Prefill engines, on [`PREFILL_COMPONENT`].
    Prefill,
    /// Aggregated or decode engines, on the model's own component.
    Serving,
}

impl Role {
    fn select(self, workers: &[DirectWorker]) -> impl Iterator<Item = &DirectWorker> {
        workers.iter().filter(move |worker| {
            (worker.worker_type == WorkerType::Prefill) == matches!(self, Role::Prefill)
        })
    }
}

/// Make the `role` engines of `workers` the membership of `endpoint` for
/// routing and for the selection catalog, following every change to the list.
fn install_workers(
    drt: &DistributedRuntime,
    manager: &ModelManager,
    endpoint: &dynamo_runtime::component::Endpoint,
    workers: &watch::Receiver<Vec<DirectWorker>>,
    role: Role,
    cancellation: &CancellationToken,
) {
    let instances = follow(cancellation, workers.clone(), move |workers| {
        role.select(workers)
            .map(|worker| worker.instance.clone())
            .collect::<Vec<Instance>>()
    });
    drt.register_static_endpoint_instances(endpoint, instances);
    let configs = follow(cancellation, workers.clone(), move |workers| {
        role.select(workers)
            .map(|worker| (worker.instance.id(), worker.runtime_config.clone()))
            .collect::<HashMap<WorkerId, ModelRuntimeConfig>>()
    });
    manager.insert_runtime_config_watcher(endpoint, configs);
}

/// A watch of `project(workers)` that follows `workers` until `cancellation`
/// fires, publishing only when the projection changes. Receivers see a stable
/// value rather than a closed channel: when the list's owner drops its sender,
/// the last projection stays until `cancellation`.
fn follow<T: PartialEq + Send + Sync + 'static>(
    cancellation: &CancellationToken,
    mut workers: watch::Receiver<Vec<DirectWorker>>,
    project: impl Fn(&[DirectWorker]) -> T + Send + 'static,
) -> watch::Receiver<T> {
    let (tx, rx) = watch::channel(project(&workers.borrow_and_update()));
    let cancellation = cancellation.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = cancellation.cancelled() => break,
                changed = workers.changed() => {
                    if changed.is_err() {
                        cancellation.cancelled().await;
                        break;
                    }
                    let next = project(&workers.borrow_and_update());
                    tx.send_if_modified(|current| {
                        let modified = *current != next;
                        if modified {
                            *current = next;
                        }
                        modified
                    });
                }
            }
        }
        drop(tx);
    });
    rx
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::component::TransportType;

    fn worker(instance_id: u64, worker_type: WorkerType) -> DirectWorker {
        DirectWorker {
            instance: Instance {
                namespace: "ns".to_string(),
                component: "backend".to_string(),
                endpoint: "generate".to_string(),
                instance_id,
                transport: TransportType::Grpc(format!("http://h:{instance_id}")),
                device_type: None,
                request_plane_codec: None,
            },
            runtime_config: ModelRuntimeConfig::default(),
            worker_type,
        }
    }

    fn ids(workers: &[DirectWorker], role: Role) -> Vec<u64> {
        role.select(workers).map(|w| w.instance.id()).collect()
    }

    #[tokio::test]
    async fn follow_tracks_additions_and_removals_per_role() {
        let cancellation = CancellationToken::new();
        let (tx, rx) = watch::channel(vec![
            worker(1, WorkerType::Decode),
            worker(2, WorkerType::Prefill),
        ]);
        let mut serving = follow(&cancellation, rx.clone(), |w| ids(w, Role::Serving));
        let mut prefill = follow(&cancellation, rx, |w| ids(w, Role::Prefill));
        assert_eq!(*serving.borrow_and_update(), vec![1]);
        assert_eq!(*prefill.borrow_and_update(), vec![2]);

        tx.send_modify(|w| w.push(worker(3, WorkerType::Decode)));
        serving.changed().await.unwrap();
        assert_eq!(*serving.borrow_and_update(), vec![1, 3]);

        tx.send_modify(|w| w.retain(|w| w.instance.id() != 2));
        prefill.changed().await.unwrap();
        assert!(prefill.borrow_and_update().is_empty());
        // The prefill-only change left the serving projection untouched.
        assert!(!serving.has_changed().unwrap());
        cancellation.cancel();
    }

    #[tokio::test]
    async fn follow_keeps_the_last_list_when_its_owner_goes_away() {
        let cancellation = CancellationToken::new();
        let (tx, rx) = watch::channel(vec![worker(1, WorkerType::Aggregated)]);
        let mut serving = follow(&cancellation, rx, |w| ids(w, Role::Serving));
        drop(tx);
        tokio::task::yield_now().await;
        assert_eq!(*serving.borrow_and_update(), vec![1]);
        assert!(!serving.has_changed().unwrap());

        cancellation.cancel();
        assert!(serving.changed().await.is_err());
    }
}
