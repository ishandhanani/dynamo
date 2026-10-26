// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-EPP routing through the frontend's own routing pipeline.
//!
//! With `DYN_EPP_ROUTING_HOST=true` the runtime EPP builds the same
//! [`PreprocessedRequest`] the frontend would and hands it to the frontend's
//! `RoutingHost`, whose final hop is a [`GatewayDispatch`]: instead of sending
//! the request, it gives the selected worker back to the EPP, and the EPP feeds
//! the response progress Envoy relays into the host's response stream. Session
//! affinity, request classifiers, load booking, output tracking and rejection
//! statuses therefore come from the frontend's code rather than an EPP copy.
//!
//! Aggregated routing only: while prefill workers are registered the EPP keeps
//! its existing disaggregated path.

use std::collections::HashSet;
use std::sync::Arc;

use dashmap::DashMap;
use dynamo_llm::http::service::{
    axum::http::{HeaderMap, HeaderName, HeaderValue},
    client_error_response, preprocessed_request_context,
};
use dynamo_llm::kv_router::gateway_dispatch::{GatewayDispatch, GatewayProgress};
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::common::FinishReason;
use dynamo_llm::protocols::common::extensions::{
    HEADER_DP_RANK, HEADER_WORKER_INSTANCE_ID, apply_frontend_nvext_policy,
};
use dynamo_llm::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use dynamo_llm::types::Annotated;
use dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest;
use dynamo_llm::types::openai::completions::NvCreateCompletionRequest;
use dynamo_protocols::types::{CompletionUsage, Prompt};
use dynamo_runtime::pipeline::{Context, ManyOut, ServiceEngine, SingleIn};
use futures::StreamExt;

use crate::picker::{PickError, PickResult};

/// Enables this path for the runtime EPP.
pub const ROUTING_HOST_ENV: &str = "DYN_EPP_ROUTING_HOST";
/// Session-affinity TTL for this path; the same variable the standalone EPP reads.
pub const SESSION_AFFINITY_TTL_ENV: &str = "DYN_EPP_SESSION_AFFINITY_TTL_SECS";

/// An SSE line longer than this is not a token delta; drop it rather than
/// buffer it (the delta it might have carried counts as one token).
const MAX_PENDING_SSE_BYTES: usize = 1024 * 1024;

pub(crate) type RoutingEngine =
    ServiceEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>>;

pub(crate) struct GatewayRouting {
    engine: RoutingEngine,
    dispatch: Arc<GatewayDispatch>,
    responses: DashMap<String, ResponseProgress>,
}

/// A request ready for the routing host.
pub(crate) struct RoutableRequest {
    context: Context<PreprocessedRequest>,
}

impl GatewayRouting {
    pub(crate) fn new(engine: RoutingEngine, dispatch: Arc<GatewayDispatch>) -> Self {
        Self {
            engine,
            dispatch,
            responses: DashMap::new(),
        }
    }

    /// Whether `booking_id` was routed through this path.
    pub(crate) fn owns(&self, booking_id: &str) -> bool {
        self.responses.contains_key(booking_id)
    }

    /// Build the request the frontend's routing host would receive: the
    /// frontend's header policy (priority, pins, cache salt), its preprocessor,
    /// and its request context (session affinity, agent context, metadata).
    ///
    /// `Ok(None)` when the body is not a chat or completions request this path
    /// handles; the caller falls back to the existing path.
    pub(crate) async fn prepare(
        preprocessor: &OpenAIPreprocessor,
        body: &str,
        headers: &[(String, String)],
        request_id: &str,
        allowed_worker_ids: Option<HashSet<u64>>,
    ) -> Result<Option<RoutableRequest>, PickError> {
        let headers = header_map(headers);
        let value: serde_json::Value = serde_json::from_str(body)
            .map_err(|e| PickError::InvalidRequest(format!("invalid JSON body: {e}")))?;
        let is_chat = value
            .get("messages")
            .and_then(|messages| messages.as_array())
            .is_some_and(|messages| !messages.is_empty());

        let mut preprocessed = if is_chat {
            let mut request: NvCreateChatCompletionRequest = serde_json::from_value(value)
                .map_err(|e| PickError::InvalidRequest(e.to_string()))?;
            request.nvext = apply_frontend_nvext_policy(request.nvext.take(), &headers, true);
            let (preprocessed, _, _) = preprocessor
                .preprocess_request(&request, None)
                .await
                .map_err(|e| PickError::InvalidRequest(format!("{e:#}")))?;
            preprocessed
        } else if value.get("prompt").is_some() {
            let mut request: NvCreateCompletionRequest = serde_json::from_value(value)
                .map_err(|e| PickError::InvalidRequest(e.to_string()))?;
            // A multi-prompt batch is split into one engine request per prompt
            // downstream; it keeps the existing path, which routes on the first.
            if is_multi_prompt_batch(&request.inner.prompt) {
                return Ok(None);
            }
            request.nvext = apply_frontend_nvext_policy(request.nvext.take(), &headers, true);
            let (preprocessed, _, _) = preprocessor
                .preprocess_request(&request, None)
                .await
                .map_err(|e| PickError::InvalidRequest(format!("{e:#}")))?;
            preprocessed
        } else {
            return Ok(None);
        };

        if allowed_worker_ids.is_some() {
            preprocessed.routing_mut().allowed_worker_ids = allowed_worker_ids;
        }
        let context = preprocessed_request_context(preprocessed, request_id.to_string(), &headers)
            .map_err(|e| PickError::InvalidRequest(e.to_string()))?;
        Ok(Some(RoutableRequest { context }))
    }

    /// Route through the host and return where Envoy should send the request.
    /// Pre-dispatch failures carry the status the frontend would return.
    pub(crate) async fn pick(
        &self,
        request: RoutableRequest,
        resolve_endpoint: impl Fn(u64) -> Option<String>,
    ) -> Result<PickResult, PickError> {
        let request_id = request.context.id().to_string();
        // If this future is dropped (the ext_proc stream closed mid-pick) after
        // the host dispatched, release the dispatch entry it left behind.
        let mut pending = AbortOnDrop {
            dispatch: &self.dispatch,
            request_id: &request_id,
            armed: true,
        };

        let stream = self
            .engine
            .generate(request.context)
            .await
            .map_err(|error| {
                let (status, message) = client_error_response(error);
                PickError::Rejected { status, message }
            })?;
        let route = self.dispatch.take_route(&request_id).ok_or_else(|| {
            PickError::RoutingFailed("routing host returned without dispatching".to_string())
        })?;
        // The host's request lifecycle runs as its response stream is polled:
        // keep polling until the EPP completes or aborts the stream.
        tokio::spawn(stream.for_each(|_| async {}));

        let Some(endpoint) = resolve_endpoint(route.worker_id) else {
            tracing::warn!(
                worker_id = route.worker_id,
                "Selected worker has no resolved endpoint"
            );
            return Err(PickError::NoEndpoints);
        };
        self.responses
            .insert(request_id.clone(), ResponseProgress::default());
        pending.armed = false;

        let dp_rank = route.dp_rank.unwrap_or(0);
        tracing::info!(
            worker_id = route.worker_id,
            dp_rank,
            endpoint = %endpoint,
            token_count = route.token_ids.len(),
            "Picked endpoint through the routing host"
        );
        Ok(PickResult {
            endpoint,
            headers: vec![
                (
                    HEADER_WORKER_INSTANCE_ID.to_string(),
                    route.worker_id.to_string(),
                ),
                (HEADER_DP_RANK.to_string(), dp_rank.to_string()),
                (
                    "x-dynamo-routing-mode".to_string(),
                    "aggregated".to_string(),
                ),
            ],
            token_ids: Some(route.token_ids),
            reservation_id: Some(request_id.clone()),
            ..Default::default()
        })
    }

    /// Relay response progress Envoy delivered for `booking_id`.
    pub(crate) fn on_response_body(
        &self,
        booking_id: &str,
        chunk: &[u8],
        streaming: bool,
        end_of_stream: bool,
    ) {
        let Some(mut response) = self.responses.get_mut(booking_id) else {
            return;
        };
        let progress = if streaming {
            response.ingest_sse(chunk, end_of_stream)
        } else if end_of_stream {
            response.ingest_unary(chunk)
        } else {
            Vec::new()
        };
        drop(response);
        for progress in progress {
            self.dispatch.progress(booking_id, progress);
        }
    }

    /// The ext_proc stream for `booking_id` ended: complete the host's stream
    /// when the response finished, and abort it otherwise.
    pub(crate) fn on_request_complete(&self, booking_id: &str) {
        let Some((_, response)) = self.responses.remove(booking_id) else {
            return;
        };
        if response.finished {
            self.dispatch.complete(booking_id);
        } else {
            self.dispatch.abort(
                booking_id,
                "gateway stream ended before the response finished",
            );
        }
    }
}

struct AbortOnDrop<'a> {
    dispatch: &'a GatewayDispatch,
    request_id: &'a str,
    armed: bool,
}

impl Drop for AbortOnDrop<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.dispatch
                .abort(self.request_id, "pick did not complete");
        }
    }
}

fn is_multi_prompt_batch(prompt: &Prompt) -> bool {
    match prompt {
        Prompt::StringArray(prompts) => prompts.len() > 1,
        Prompt::ArrayOfIntegerArray(prompts) => prompts.len() > 1,
        Prompt::String(_) | Prompt::IntegerArray(_) => false,
    }
}

fn header_map(headers: &[(String, String)]) -> HeaderMap {
    let mut map = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(name.as_bytes()),
            HeaderValue::from_str(value),
        ) {
            map.append(name, value);
        }
    }
    map
}

/// Response progress read off the gateway's copy of an OpenAI response.
///
/// Every SSE event is a token delta until proven otherwise, so deltas are
/// counted with a byte scan; JSON is parsed only for the one or two events
/// per response that carry a `finish_reason` or `usage`.
#[derive(Default)]
struct ResponseProgress {
    /// Trailing partial SSE line from the previous chunk.
    pending: Vec<u8>,
    observed: Observed,
    finished: bool,
}

/// What the terminal events of a response say about it.
#[derive(Default)]
struct Observed {
    reason: Option<FinishReason>,
    usage: Option<CompletionUsage>,
}

impl ResponseProgress {
    /// Count one token per SSE event that carries output. Envoy may split
    /// events across chunks, so an incomplete trailing line is held; whole
    /// chunks are scanned in place without copying.
    fn ingest_sse(&mut self, chunk: &[u8], end_of_stream: bool) -> Vec<GatewayProgress> {
        let tokens = if self.pending.is_empty() {
            let complete = if end_of_stream {
                chunk.len()
            } else {
                line_boundary(chunk)
            };
            let tokens = self.observed.scan_lines(&chunk[..complete]);
            tokens + self.hold(&chunk[complete..])
        } else {
            self.pending.extend_from_slice(chunk);
            let complete = if end_of_stream {
                self.pending.len()
            } else {
                line_boundary(&self.pending)
            };
            let tokens = self.observed.scan_lines(&self.pending[..complete]);
            self.pending.drain(..complete);
            tokens + self.cap_pending()
        };
        let mut progress = Vec::with_capacity(2);
        if tokens > 0 {
            progress.push(GatewayProgress::Tokens(tokens));
        }
        if end_of_stream {
            progress.push(self.finish());
        }
        progress
    }

    /// Keep a partial trailing line for the next chunk.
    fn hold(&mut self, tail: &[u8]) -> usize {
        if tail.is_empty() {
            return 0;
        }
        self.pending.extend_from_slice(tail);
        self.cap_pending()
    }

    /// An SSE line longer than [`MAX_PENDING_SSE_BYTES`] is not a token delta;
    /// drop it rather than buffer it (the delta it might have carried counts
    /// as one token).
    fn cap_pending(&mut self) -> usize {
        if self.pending.len() > MAX_PENDING_SSE_BYTES {
            self.pending.clear();
            1
        } else {
            0
        }
    }

    /// A buffered (non-streaming) response reports its output only through
    /// `usage.completion_tokens`.
    fn ingest_unary(&mut self, body: &[u8]) -> Vec<GatewayProgress> {
        let mut progress = Vec::with_capacity(2);
        if let Ok(response) = serde_json::from_slice::<UnaryResponse>(body) {
            self.observed.reason = response
                .choices
                .iter()
                .find_map(|choice| choice.finish_reason.as_deref())
                .map(finish_reason);
            self.observed.usage = response.usage;
            let tokens = self
                .observed
                .usage
                .as_ref()
                .map_or(0, |usage| usage.completion_tokens as usize);
            if tokens > 0 {
                progress.push(GatewayProgress::Tokens(tokens));
            }
        }
        progress.push(self.finish());
        progress
    }

    fn finish(&mut self) -> GatewayProgress {
        self.finished = true;
        GatewayProgress::Finish {
            reason: self.observed.reason.take(),
            usage: self.observed.usage.take(),
        }
    }
}

impl Observed {
    /// Scan complete SSE lines: count the events that carry output and record
    /// a finish reason or usage when an event names one.
    fn scan_lines(&mut self, lines: &[u8]) -> usize {
        let mut tokens = 0;
        for line in lines.split(|&b| b == b'\n') {
            let Some(data) = line.trim_ascii().strip_prefix(b"data:") else {
                continue;
            };
            let data = data.trim_ascii();
            if data.is_empty() || data == b"[DONE]" {
                continue;
            }
            let Ok(event) = std::str::from_utf8(data) else {
                continue;
            };
            let scan = scan_event(event);
            if scan.carries_output {
                tokens += 1;
            }
            if let Some(reason) = scan.finish_reason {
                self.reason = Some(finish_reason(reason));
            }
            if scan.has_usage
                && let Ok(terminal) = serde_json::from_str::<UsageEvent>(event)
                && let Some(usage) = terminal.usage
            {
                self.usage = Some(usage);
            }
        }
        tokens
    }
}

/// What one pass over a serialized event found.
#[derive(Default)]
struct EventScan<'a> {
    /// A non-empty `content`, `reasoning_content` or `tool_calls` delta, or a
    /// non-empty completions `text`.
    carries_output: bool,
    /// `finish_reason` when it is a string (`null` is not a finish).
    finish_reason: Option<&'a str>,
    /// A non-null `usage` object is present.
    has_usage: bool,
}

/// One pass over the event: at every `":` key separator, look at the key
/// before it and the value after it. JSON strings escape their quotes, so a
/// key name inside generated text cannot match.
fn scan_event(event: &str) -> EventScan<'_> {
    let mut scan = EventScan::default();
    for (at, _) in event.match_indices("\":") {
        let key = &event[..at];
        let value = event[at + 2..].trim_start();
        if key.ends_with("\"content")
            || key.ends_with("\"reasoning_content")
            || key.ends_with("\"tool_calls")
            || key.ends_with("\"text")
        {
            if !value_is_empty(value) {
                scan.carries_output = true;
            }
        } else if key.ends_with("\"finish_reason") {
            if let Some(string) = value.strip_prefix('"')
                && let Some(end) = string.find('"')
            {
                scan.finish_reason = Some(&string[..end]);
            }
        } else if key.ends_with("\"usage") && value.starts_with('{') {
            scan.has_usage = true;
        }
    }
    scan
}

/// An empty string, array or object, or `null`, carries nothing.
fn value_is_empty(value: &str) -> bool {
    value.starts_with("\"\"")
        || value.starts_with("[]")
        || value.starts_with("{}")
        || value.starts_with("null")
}

/// The fields of a terminal event this path needs; everything else is skipped.
#[derive(serde::Deserialize)]
struct UsageEvent {
    usage: Option<CompletionUsage>,
}

#[derive(serde::Deserialize)]
struct UnaryResponse {
    #[serde(default)]
    choices: Vec<UnaryChoice>,
    usage: Option<CompletionUsage>,
}

#[derive(serde::Deserialize)]
struct UnaryChoice {
    finish_reason: Option<String>,
}

/// Index just past the last newline in `bytes`, or 0 when it has none.
fn line_boundary(bytes: &[u8]) -> usize {
    bytes
        .iter()
        .rposition(|&b| b == b'\n')
        .map_or(0, |index| index + 1)
}

fn finish_reason(reason: &str) -> FinishReason {
    match reason {
        "length" => FinishReason::Length,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

#[cfg(test)]
mod tests;
