// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::{ModelManager, ModelWatcher};
use crate::engines::StreamingEngineAdapter;
use crate::entrypoint::EngineConfig;
use crate::entrypoint::input::common;
use crate::http::service::metrics::Metrics;
use crate::namespace::NamespaceFilter;
use crate::request_template::RequestTemplate;
use crate::types::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
    OpenAIChatCompletionsStreamingEngine,
};
use dynamo_protocols::types::ChatCompletionMessageContent;
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::pipeline::Context;
use futures::StreamExt;
use std::io::{ErrorKind, Write};
use std::sync::Arc;
use std::time::Duration;

/// Max response tokens for each single query. Must be less than model context size.
/// TODO: Cmd line flag to overwrite this
const MAX_TOKENS: u32 = 8192;

pub async fn run(
    distributed_runtime: DistributedRuntime,
    single_prompt: Option<String>,
    engine_config: EngineConfig,
) -> anyhow::Result<()> {
    let prepared_engine = prepare_engine(distributed_runtime.clone(), engine_config).await?;
    // TODO: Pass prepared_engine directly
    main_loop(
        distributed_runtime,
        &prepared_engine.service_name,
        prepared_engine.engine,
        single_prompt,
        prepared_engine.inspect_template,
        prepared_engine.request_template,
    )
    .await
}

struct PreparedEngine {
    service_name: String,
    engine: OpenAIChatCompletionsStreamingEngine,
    inspect_template: bool,
    request_template: Option<RequestTemplate>,
}

/// Turns an EngineConfig into an OpenAI chat-completions and completions supported StreamingEngine.
async fn prepare_engine(
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
            let discovery = distributed_runtime.discovery();
            let discovery_stream = discovery
                .list_and_watch(
                    dynamo_runtime::discovery::DiscoveryQuery::AllModels,
                    Some(distributed_runtime.primary_token().clone()),
                )
                .await?;
            let inner_watch_obj = watch_obj.clone();
            let namespace_filter = NamespaceFilter::from_namespace_and_prefix(
                local_model.namespace(),
                local_model.namespace_prefix(),
            );
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
                    Err(crate::discovery::ModelManagerError::ModelUnavailable(_))
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
            let pipeline = common::build_pipeline::<
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

async fn main_loop(
    distributed_runtime: DistributedRuntime,
    service_name: &str,
    engine: OpenAIChatCompletionsStreamingEngine,
    mut initial_prompt: Option<String>,
    _inspect_template: bool,
    template: Option<RequestTemplate>,
) -> anyhow::Result<()> {
    let cancel_token = distributed_runtime.primary_token();
    if initial_prompt.is_none() {
        tracing::info!("Ctrl-c to exit");
    }
    let theme = dialoguer::theme::ColorfulTheme::default();

    // Initial prompt is from piped stdin.
    // We run that single prompt and exit
    let single = initial_prompt.is_some();
    let mut history = dialoguer::BasicHistory::default();
    let mut messages = vec![];
    while !cancel_token.is_cancelled() {
        // User input
        let prompt = match initial_prompt.take() {
            Some(p) => p,
            None => {
                let input_ui = dialoguer::Input::<String>::with_theme(&theme)
                    .history_with(&mut history)
                    .with_prompt("User");
                match input_ui.interact_text() {
                    Ok(prompt) => prompt,
                    Err(dialoguer::Error::IO(err)) => {
                        match err.kind() {
                            ErrorKind::Interrupted => {
                                // Ctrl-C
                                // Unfortunately I could not make dialoguer handle Ctrl-d
                            }
                            k => {
                                tracing::info!("IO error: {k}");
                            }
                        }
                        break;
                    }
                }
            }
        };

        // Construct messages
        let user_message = dynamo_protocols::types::ChatCompletionRequestMessage::User(
            dynamo_protocols::types::ChatCompletionRequestUserMessage {
                content: dynamo_protocols::types::ChatCompletionRequestUserMessageContent::Text(
                    prompt,
                ),
                name: None,
            },
        );
        messages.push(user_message);
        // Request
        let inner = dynamo_protocols::types::CreateChatCompletionRequestArgs::default()
            .messages(messages.clone())
            .model(
                template
                    .as_ref()
                    .map_or_else(|| service_name.to_string(), |t| t.model.clone()),
            )
            .stream(true)
            .max_completion_tokens(
                template
                    .as_ref()
                    .map_or(MAX_TOKENS, |t| t.max_completion_tokens),
            )
            .temperature(template.as_ref().map_or(0.7, |t| t.temperature))
            .n(1) // only generate one response
            .build()?;

        let req = NvCreateChatCompletionRequest {
            inner,
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            thinking: None,
            media_io_kwargs: None,
            return_tokens_as_token_ids: None,
            thinking_token_budget: None,
            unsupported_fields: Default::default(),
        };

        // Call the model
        let mut stream = match engine.generate(Context::new(req)).await {
            Ok(stream) => stream,
            Err(err) => {
                tracing::error!(%err, "Request failed.");
                continue;
            }
        };

        // Stream the output to stdout
        let mut stdout = std::io::stdout();
        let mut assistant_message = String::new();
        let mut assistant_reasoning = String::new();
        while let Some(item) = stream.next().await {
            if cancel_token.is_cancelled() {
                break;
            }
            match (item.data.as_ref(), item.event.as_deref()) {
                (Some(data), _) => {
                    // Normal case
                    let Some(chat_comp) = data.inner.choices.first() else {
                        continue;
                    };
                    if let Some(c) = &chat_comp.delta.content {
                        match c {
                            ChatCompletionMessageContent::Text(text) => {
                                let _ = stdout.write(text.as_bytes());
                                let _ = stdout.flush();
                                assistant_message += text;
                            }
                            ChatCompletionMessageContent::Parts(_) => {
                                // (ayushag) TODO: Handle multimodal content for multiturn conversations
                                // Multimodal content - for now just print a placeholder
                                let _ = stdout.write(b"[multimodal content]");
                                let _ = stdout.flush();
                            }
                        }
                    }
                    if let Some(reasoning) = &chat_comp.delta.reasoning_content {
                        assistant_reasoning += reasoning;
                    }
                    if let Some(reason) = chat_comp.finish_reason {
                        tracing::trace!("finish reason: {reason:?}");
                        break;
                    }
                }
                (None, Some("error")) => {
                    // There's only one error but we loop in case that changes
                    for err in item.comment.unwrap_or_default() {
                        tracing::error!("Engine error: {err}");
                    }
                }
                (None, Some(annotation)) => {
                    tracing::debug!("Annotation. {annotation}: {:?}", item.comment);
                }
                _ => {
                    unreachable!("Event from engine with no data, no error, no annotation.");
                }
            }
        }
        println!();

        let assistant_content =
            dynamo_protocols::types::ChatCompletionRequestAssistantMessageContent::Text(
                assistant_message,
            );

        let assistant_message = dynamo_protocols::types::ChatCompletionRequestMessage::Assistant(
            dynamo_protocols::types::ChatCompletionRequestAssistantMessage {
                content: Some(assistant_content),
                reasoning_content: (!assistant_reasoning.is_empty()).then_some(
                    dynamo_protocols::types::ReasoningContent::Text(assistant_reasoning),
                ),
                ..Default::default()
            },
        );
        messages.push(assistant_message);

        if single {
            break;
        }
    }
    println!();

    // Stop the runtime and wait for it to stop
    distributed_runtime.shutdown();
    cancel_token.cancelled().await;

    Ok(())
}
