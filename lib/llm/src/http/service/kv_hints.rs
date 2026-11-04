// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `POST /v1/hints/prefetch`: a client-driven KV cache prefetch.
//!
//! The body is the `/v1/chat/completions` request body the client is about to send
//! (same type, so tools, templates, and response formats render the same prompt);
//! generation parameters are ignored. `nvext.token_data`, when present, is used as
//! the prompt verbatim; otherwise `messages` are rendered and tokenized with the
//! model's own preprocessor off the async runtime. The router picks a worker without
//! booking, its hint controller sends the tokens to that worker's scheduler hint
//! socket, and the `202` reply names the worker. An `x-dyn-wakeup` header records
//! the session tag so the real request's own wake-up prefetch is skipped inside the
//! dedup window. Served on the main listener only when `DYN_ENABLE_KVM_HINTS_API` is
//! truthy (off by default).

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
};
use dashmap::DashMap;
use serde_json::json;

use super::{RouteDoc, service_v2};
use crate::kv_router::FindBestMatchOutcome;
use crate::kv_router::hints::{wakeup_tag_from_headers, worker_target};
use crate::model_card::ModelDeploymentCard;
use crate::preprocessor::OpenAIPreprocessor;
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;

pub const PREFETCH_PATH: &str = "/v1/hints/prefetch";

struct HintsState {
    state: Arc<service_v2::State>,
    /// One preprocessor (tokenizer and chat template) per model card checksum, built
    /// on first use off the async runtime; a redeployed model with a new card gets a
    /// fresh one.
    preprocessors: DashMap<String, Arc<OpenAIPreprocessor>>,
}

fn error(status: StatusCode, message: impl ToString) -> Response {
    (status, Json(json!({"error": message.to_string()}))).into_response()
}

/// Boxed so `Result<_, Response>` stays small (clippy `result_large_err`).
type Rejected = Box<Response>;

fn reject(status: StatusCode, message: impl ToString) -> Rejected {
    Box::new(error(status, message))
}

impl HintsState {
    fn card(&self, model: &str) -> Result<ModelDeploymentCard, Rejected> {
        let manager = self.state.manager();
        let canonical = manager.resolve_canonical_name(model);
        manager
            .get_model_cards()
            .into_iter()
            .find(|card| card.display_name == model || card.display_name == canonical)
            .ok_or_else(|| reject(StatusCode::NOT_FOUND, format!("unknown model {model:?}")))
    }

    /// Render and tokenize `request` with the card's preprocessor on a blocking
    /// thread; loading the tokenizer and templating a long prompt both take time.
    async fn tokenize(
        self: &Arc<Self>,
        card: ModelDeploymentCard,
        request: NvCreateChatCompletionRequest,
    ) -> Result<Vec<u32>, Rejected> {
        let hints = Arc::clone(self);
        tokio::task::spawn_blocking(move || -> Result<Vec<u32>, Rejected> {
            let preprocessor = match hints.preprocessors.get(card.mdcsum()) {
                Some(preprocessor) => Arc::clone(&preprocessor),
                None => {
                    let mdcsum = card.mdcsum().to_string();
                    let preprocessor = OpenAIPreprocessor::new(card).map_err(|err| {
                        reject(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("failed to load tokenizer: {err}"),
                        )
                    })?;
                    hints
                        .preprocessors
                        .insert(mdcsum, Arc::clone(&preprocessor));
                    preprocessor
                }
            };
            let prompt = preprocessor
                .apply_template(&request)
                .map_err(|err| reject(StatusCode::BAD_REQUEST, err))?
                .ok_or_else(|| reject(StatusCode::BAD_REQUEST, "messages rendered no prompt"))?;
            let encoding = preprocessor
                .tokenize_rendered_prompt(&prompt)
                .map_err(|err| reject(StatusCode::INTERNAL_SERVER_ERROR, err))?;
            Ok(encoding.token_ids().to_vec())
        })
        .await
        .map_err(|err| reject(StatusCode::INTERNAL_SERVER_ERROR, err))?
    }
}

async fn prefetch_hint(
    State(hints): State<Arc<HintsState>>,
    headers: HeaderMap,
    Json(request): Json<NvCreateChatCompletionRequest>,
) -> Response {
    let model = request.inner.model.clone();
    let card = match hints.card(&model) {
        Ok(card) => card,
        Err(response) => return *response,
    };
    let context_length = card.runtime_config.context_length;
    let tokens = match request
        .nvext
        .as_ref()
        .and_then(|nvext| nvext.token_data.clone())
    {
        Some(tokens) => tokens,
        None => match hints.tokenize(card, request).await {
            Ok(tokens) => tokens,
            Err(response) => return *response,
        },
    };
    if tokens.is_empty() {
        return error(StatusCode::BAD_REQUEST, "no tokens to prefetch");
    }
    if let Some(limit) = context_length
        && tokens.len() > limit as usize
    {
        return error(
            StatusCode::BAD_REQUEST,
            format!(
                "prompt of {} tokens exceeds the model context length {limit}",
                tokens.len()
            ),
        );
    }
    let Some(router) = hints.state.manager().kv_router_for_model(&model) else {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("model {model:?} has no KV router"),
        );
    };
    let Some(controller) = router.hint_controller() else {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("model {model:?} router has no hint controller"),
        );
    };
    let worker = match router.find_best_match_without_admission(&tokens).await {
        Ok(FindBestMatchOutcome::Routed { worker, .. }) => worker,
        Ok(FindBestMatchOutcome::QueueRejected { rejection }) => {
            return error(StatusCode::SERVICE_UNAVAILABLE, rejection);
        }
        Err(err) => return error(StatusCode::SERVICE_UNAVAILABLE, err),
    };
    if let (Some(dedup), Some(tag)) = (router.wakeup_dedup(), wakeup_tag_from_headers(&headers)) {
        // Mark the session so the real request's header-triggered prefetch is skipped.
        dedup.mark(&tag.0);
    }
    let token_count = tokens.len();
    controller.prefetch(worker_target(worker), Arc::from(tokens));
    (
        StatusCode::ACCEPTED,
        Json(json!({
            "worker": worker.worker_id,
            "dp_rank": worker.dp_rank,
            "tokens": token_count,
        })),
    )
        .into_response()
}

pub(crate) fn router(state: Arc<service_v2::State>) -> (Vec<RouteDoc>, Router) {
    let docs = vec![RouteDoc::new(Method::POST, PREFETCH_PATH)];
    let router = Router::new()
        .route(PREFETCH_PATH, post(prefetch_hint))
        .with_state(Arc::new(HintsState {
            state,
            preprocessors: DashMap::new(),
        }));
    (docs, router)
}
