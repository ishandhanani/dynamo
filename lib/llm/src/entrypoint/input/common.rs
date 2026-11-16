// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::time::Duration;

use dynamo_renderer::PromptFormatter;

use crate::{entrypoint::EngineConfig, namespace::NamespaceFilter};
use dynamo_discovery::{ModelManager, ModelWatcher, engines::StreamingEngineAdapter};
use dynamo_frontend_protocols::common::llm_backend::{BackendOutput, PreprocessedRequest};
use dynamo_frontend_protocols::types::{
    Annotated,
    openai::{
        chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
            OpenAIChatCompletionsStreamingEngine,
        },
        completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    },
};
use dynamo_http_server::metrics::Metrics;
use dynamo_model_card::{model_card::ModelDeploymentCard, request_template::RequestTemplate};
use dynamo_preprocessor::backend::{Backend, ExecutionContext};
use dynamo_preprocessor::{OpenAIPreprocessor, prompt::prompt_formatter_from_mdc};

use dynamo_runtime::{
    DistributedRuntime,
    engine::{AsyncEngineStream, Data},
    pipeline::{Context, ManyOut, Operator, ServiceBackend, ServiceFrontend, SingleIn, Source},
};
use std::sync::Arc;

pub struct PreparedEngine {
    pub service_name: String,
    pub engine: OpenAIChatCompletionsStreamingEngine,
    pub inspect_template: bool,
    pub request_template: Option<RequestTemplate>,
}

/// Turns an EngineConfig into an OpenAI chat-completions and completions supported StreamingEngine.
pub async fn prepare_engine(
    distributed_runtime: DistributedRuntime,
    engine_config: EngineConfig,
) -> anyhow::Result<PreparedEngine> {
    match engine_config {
        EngineConfig::Dynamic {
            model: local_model,
            prefill_load_estimator,
            ..
        } => {
            let model_manager = Arc::new(ModelManager::new());
            // Create metrics for migration tracking (not exposed via /metrics in Dynamic engine mode)
            let metrics = Arc::new(Metrics::new());
            let mut watcher = ModelWatcher::new(
                distributed_runtime.clone(),
                model_manager.clone(),
                local_model.router_config().clone(),
                local_model.migration_limit(),
                local_model.migration_max_seq_len(),
                None,
                prefill_load_estimator,
                metrics,
            );
            if !local_model.path().as_os_str().is_empty() {
                watcher.set_local_model_path(Some(local_model.path().to_path_buf()));
            }
            watcher.set_tokenizer_backend(local_model.runtime_config().tokenizer_backend);
            watcher.set_tokenizer_fallback_enabled(
                local_model.runtime_config().tokenizer_fallback_enabled,
            );
            let watch_obj = Arc::new(watcher);
            let namespace_filter = NamespaceFilter::from_namespace_and_prefix(
                local_model.namespace(),
                local_model.namespace_prefix(),
            );
            let discovery = distributed_runtime.discovery();
            let discovery_stream = discovery
                .list_and_watch(
                    namespace_filter.model_query(),
                    Some(distributed_runtime.primary_token().clone()),
                )
                .await?;
            let inner_watch_obj = watch_obj.clone();
            let _watcher_task = tokio::spawn(async move {
                inner_watch_obj
                    .watch(discovery_stream, namespace_filter)
                    .await;
            });
            tracing::info!("Waiting for remote model..");

            // TODO: We use the first model to appear, usually we have only one
            // We should add slash commands to text input `/model <name>` to choose,
            // '/models` to list, and notifications when models are added / removed.

            let model_service_name = watch_obj.wait_for_chat_model().await;
            tracing::info!("Connected to {model_service_name}");
            // In disaggregated deployments the model may be listed before the prefill
            // router is fully activated, causing a transient ModelUnavailable. Retry
            // with a timeout so the startup path doesn't fail during this cold-start
            // window, but also doesn't hang indefinitely on misconfiguration.
            let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
            let engine = loop {
                match model_manager.get_chat_completions_engine(&model_service_name) {
                    Ok(engine) => break engine,
                    Err(dynamo_discovery::ModelManagerError::ModelUnavailable(_))
                        if tokio::time::Instant::now() < deadline =>
                    {
                        tracing::debug!(
                            model = %model_service_name,
                            "Model listed but not yet servable, waiting for prefill activation"
                        );
                        tokio::time::sleep(Duration::from_millis(500)).await;
                    }
                    Err(e) => return Err(e.into()),
                }
            };
            Ok(PreparedEngine {
                service_name: model_service_name,
                engine,
                inspect_template: false,
                request_template: local_model.request_template(),
            })
        }
        EngineConfig::InProcessText { engine, model, .. } => {
            let service_name = model.service_name().to_string();
            tracing::debug!("Model: {service_name} with engine pre-processing");
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            Ok(PreparedEngine {
                service_name,
                engine,
                inspect_template: false,
                request_template: model.request_template(),
            })
        }
        EngineConfig::InProcessTokens {
            engine: inner_engine,
            model,
            ..
        } => {
            let pipeline = build_pipeline::<
                NvCreateChatCompletionRequest,
                NvCreateChatCompletionStreamResponse,
            >(model.card(), inner_engine, model.card().tokenizer()?)
            .await?;

            let service_name = model.service_name().to_string();
            tracing::debug!("Model: {service_name} with Dynamo pre-processing");
            Ok(PreparedEngine {
                service_name,
                engine: pipeline,
                inspect_template: true,
                request_template: model.request_template(),
            })
        }
    }
}

pub async fn build_pipeline<Req, Resp>(
    card: &ModelDeploymentCard,
    engine: ExecutionContext,
    tokenizer: crate::tokenizers::Tokenizer,
) -> anyhow::Result<Arc<ServiceFrontend<SingleIn<Req>, ManyOut<Annotated<Resp>>>>>
where
    Req: Data,
    Resp: Data,
    OpenAIPreprocessor: Operator<
            Context<Req>,
            Pin<Box<dyn AsyncEngineStream<Annotated<Resp>>>>,
            Context<PreprocessedRequest>,
            Pin<Box<dyn AsyncEngineStream<Annotated<BackendOutput>>>>,
        >,
{
    let PromptFormatter::OAI(formatter) = prompt_formatter_from_mdc(card)?;
    let preprocessor =
        OpenAIPreprocessor::new_with_parts(card.clone(), formatter, tokenizer.clone())?;
    link_pipeline(engine, tokenizer, preprocessor)
}

type InputFrontend<Req, Resp> = Arc<ServiceFrontend<SingleIn<Req>, ManyOut<Annotated<Resp>>>>;

pub(crate) fn build_completions_pipeline(
    card: &ModelDeploymentCard,
    engine: ExecutionContext,
    tokenizer: crate::tokenizers::Tokenizer,
) -> anyhow::Result<InputFrontend<NvCreateCompletionRequest, NvCreateCompletionResponse>> {
    let preprocessor = OpenAIPreprocessor::new_for_completions(card.clone(), tokenizer.clone())?;
    link_pipeline(engine, tokenizer, preprocessor)
}

fn link_pipeline<Req, Resp>(
    engine: ExecutionContext,
    tokenizer: crate::tokenizers::Tokenizer,
    preprocessor: Arc<OpenAIPreprocessor>,
) -> anyhow::Result<InputFrontend<Req, Resp>>
where
    Req: Data,
    Resp: Data,
    OpenAIPreprocessor: Operator<
            Context<Req>,
            Pin<Box<dyn AsyncEngineStream<Annotated<Resp>>>>,
            Context<PreprocessedRequest>,
            Pin<Box<dyn AsyncEngineStream<Annotated<BackendOutput>>>>,
        >,
{
    let frontend = ServiceFrontend::<SingleIn<Req>, ManyOut<Annotated<Resp>>>::new();
    let preprocessor = preprocessor.into_operator();
    let backend = Backend::from_tokenizer(tokenizer).into_operator();
    let engine = ServiceBackend::from_engine(engine);

    Ok(frontend
        .link(preprocessor.forward_edge())?
        .link(backend.forward_edge())?
        .link(engine)?
        .link(backend.backward_edge())?
        .link(preprocessor.backward_edge())?
        .link_terminal(frontend)?)
}
