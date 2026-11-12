// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV cache hints: a wake-up call (`x-dynamo-experimental-wakeup`) prefetches its prompt to the
//! worker the router would pick, and a tool call pins a session's pages in the Mooncake master
//! (`DYN_KVM_PIN_ENDPOINT`).

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::http::HeaderMap;
use dynamo_kv_hints::pin::{ToolCallPinSlot, parse_sim_tool_call, pin_ttl_and_priority};
use dynamo_kv_hints::sender::HintSender;
use dynamo_kv_hints::{KvHint, KvHintAction};
use dynamo_router_protocol::worker::WorkerWithDpRank;
use dynamo_runtime::config::environment_names::llm as env_llm;
use dynamo_runtime::engine::AsyncEngineContextProvider;
use dynamo_runtime::pipeline::SingleIn;
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};

use crate::discovery::RuntimeConfigWatch;
use crate::protocols::common::llm_backend::{BackendOutput, PreprocessedRequest};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;

/// Runtime-config key of a worker's scheduler hint socket, published by the SGLang sidecar.
pub const HINT_ZMQ_ENDPOINT_KEY: &str = "hint_zmq_endpoint";
pub const WAKEUP_CONTEXT_KEY: &str = "kv_hint_wakeup";
pub const PIN_SLOT_CONTEXT_KEY: &str = "kv_hint_pin_slot";
pub const SIM_TOOL_CALL_CONTEXT_KEY: &str = "kv_hint_sim_tool_call";

/// Request-context marker: a wake-up call, which prefetches its prompt instead of running.
pub struct Wakeup;
/// Request-context value: a tool call follows the response, expected to take this long.
pub struct SimToolCall(pub Option<Duration>);

pub fn pin_endpoint() -> Option<&'static str> {
    static ENDPOINT: OnceLock<Option<String>> = OnceLock::new();
    ENDPOINT
        .get_or_init(|| std::env::var(env_llm::DYN_KVM_PIN_ENDPOINT).ok())
        .as_deref()
        .filter(|endpoint| !endpoint.is_empty())
}

pub fn wakeup_from_headers(headers: &HeaderMap) -> Option<Wakeup> {
    headers
        .contains_key("x-dynamo-experimental-wakeup")
        .then_some(Wakeup)
}

pub fn sim_tool_call_from_headers(headers: &HeaderMap) -> Option<SimToolCall> {
    let value = headers
        .get("x-dynamo-experimental-sim-tool-call")?
        .to_str()
        .ok()?;
    pin_endpoint().map(|_| SimToolCall(parse_sim_tool_call(value)))
}

/// One router's hints to its workers and their store.
pub struct KvHints {
    sender: HintSender,
    configs: RuntimeConfigWatch,
}

impl KvHints {
    pub fn new(configs: RuntimeConfigWatch) -> Arc<Self> {
        Arc::new(Self {
            sender: HintSender::new(pin_endpoint()),
            configs,
        })
    }

    /// Ask `worker`'s scheduler to load `request`'s prompt from the shared store.
    pub(crate) fn prefetch(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        worker: WorkerWithDpRank,
    ) {
        let endpoint = (self.configs.borrow().get(&worker.worker_id))
            .map(|config| config.get_engine_specific::<String>(HINT_ZMQ_ENDPOINT_KEY));
        if let Some(Ok(Some(endpoint))) = endpoint {
            let action = KvHintAction::prefetch("0", &request.token_ids, worker.dp_rank);
            let hint = KvHint::new(request.context().id(), vec![action]);
            self.sender.send(&endpoint, &hint);
        }
    }

    /// `request` was routed to `worker`: arm its pin.
    pub(crate) fn on_route(
        self: &Arc<Self>,
        request: &SingleIn<PreprocessedRequest>,
        worker: WorkerWithDpRank,
    ) {
        if let Ok(slot) = request.get::<ToolCallPinSlot>(PIN_SLOT_CONTEXT_KEY) {
            let hints = Arc::clone(self);
            let prompt = request.token_ids.clone();
            slot.arm(move |output, expected| {
                let tokens: Vec<u32> = prompt.iter().chain(output).copied().collect();
                let configs = hints.configs.borrow();
                let Some((tenant, keys)) = configs.get(&worker.worker_id).and_then(|config| {
                    super::shared_cache::mooncake_pin_keys(worker.worker_id, config, &tokens)
                }) else {
                    return;
                };
                let (ttl, priority) = pin_ttl_and_priority(expected, tokens.len());
                hints.sender.pin(&tenant, &keys, ttl, priority);
            });
        }
    }
}

type Boxed<T> = std::pin::Pin<Box<dyn Stream<Item = Annotated<T>> + Send>>;

/// Record the engine's output tokens in `slot`.
pub fn record_output<S>(stream: S, slot: Option<Arc<ToolCallPinSlot>>) -> Boxed<BackendOutput>
where
    S: Stream<Item = Annotated<BackendOutput>> + Send + 'static,
{
    let Some(slot) = slot else {
        return Box::pin(stream);
    };
    Box::pin(stream.inspect(move |response| {
        if let Some(output) = &response.data {
            slot.push_output(&output.token_ids);
        }
    }))
}

/// Record tool calls in `slot` and finish it when the response ends.
pub fn pin_on_tool_call<S>(
    stream: S,
    slot: Option<Arc<ToolCallPinSlot>>,
    announced: Option<Option<Duration>>,
) -> Boxed<NvCreateChatCompletionStreamResponse>
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    let Some(slot) = slot else {
        return Box::pin(stream);
    };
    let seen = Arc::clone(&slot);
    let chunks = stream.inspect(move |response| {
        let choices = response.data.iter().flat_map(|chunk| &chunk.inner.choices);
        if choices
            .into_iter()
            .any(|c| matches!(&c.delta.tool_calls, Some(t) if !t.is_empty()))
        {
            seen.tool_called();
        }
    });
    let end = futures::stream::once(async move { slot.finish(announced) });
    Box::pin(chunks.chain(end.filter_map(|()| async { None })))
}
