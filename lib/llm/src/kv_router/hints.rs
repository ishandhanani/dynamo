// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-router KV cache hint controller and its triggers.
//!
//! A router builds one [`HintController`] over its workers when a hint consumer is
//! enabled: `DYN_KVM_PREFETCH_LEAD_MS` (deadline-led prefetch from the admission
//! queue), `DYN_ENABLE_KVM_HINTS_API` (the `POST /v1/hints/prefetch` route),
//! `DYN_KVM_WAKEUP_HEADER` (prefetch at enqueue for requests carrying `x-dyn-wakeup`),
//! or `DYN_KVM_PIN_ENDPOINT` (pin a request's pages in the shared store when its
//! response calls a tool; see [`ToolCallPinSlot`]).
//! Hints go straight to each worker's scheduler over the ZMQ endpoint the worker
//! publishes in its runtime config as [`HINT_ZMQ_ENDPOINT_KEY`]. Every environment
//! knob is read here; `dynamo-kvm` itself reads none.

use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use axum::http::HeaderMap;
use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_kv_router::scheduling::{PrefetchHintSink, PrefetchTrigger};
use dynamo_kvm::{
    HintConfig, HintController, HintEndpointResolver, HintEndpoints, PinHint, WorkerTarget,
    ZmqHintTransport, ZmqStorePinSink,
};
use dynamo_runtime::config::env_is_truthy;
use dynamo_runtime::config::environment_names::llm as env_llm;

use crate::discovery::RuntimeConfigWatch;

/// Runtime-config key under which the SGLang sidecar (`lib/sidecar/sglang`) publishes
/// a worker's scheduler inbound hint socket as a connectable `tcp://host:port`.
pub const HINT_ZMQ_ENDPOINT_KEY: &str = "hint_zmq_endpoint";

/// Request header whose presence asks for a prefetch at enqueue; the value is an
/// opaque session tag used for deduplication and logging.
pub const WAKEUP_HEADER: &str = "x-dyn-wakeup";
/// Request-context key carrying the `x-dyn-wakeup` value to the routing host.
pub const WAKEUP_CONTEXT_KEY: &str = "kv_hint_wakeup";
/// Longest `x-dyn-wakeup` value honored; longer tags are ignored.
pub const WAKEUP_TAG_MAX_BYTES: usize = 128;
/// Most wake-up tags remembered at once; beyond this, new tags are not deduplicated.
const WAKEUP_MAX_TRACKED_TAGS: usize = 4096;

/// The `x-dyn-wakeup` header value.
#[derive(Debug, Clone)]
pub struct WakeupTag(pub String);

const DEFAULT_WAKEUP_DEDUP_MS: u64 = 2_000;

/// Whether `x-dyn-wakeup` is honored (`DYN_KVM_WAKEUP_HEADER`), read once per process.
pub fn wakeup_header_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_is_truthy(env_llm::DYN_KVM_WAKEUP_HEADER))
}

/// The wake-up tag on a request, when present, printable, and within
/// [`WAKEUP_TAG_MAX_BYTES`].
pub fn wakeup_tag_from_headers(headers: &HeaderMap) -> Option<WakeupTag> {
    let value = headers.get(WAKEUP_HEADER)?;
    let Ok(tag) = value.to_str() else {
        tracing::debug!("x-dyn-wakeup ignored: value is not visible ASCII");
        return None;
    };
    if tag.is_empty() || tag.len() > WAKEUP_TAG_MAX_BYTES {
        tracing::debug!(
            bytes = tag.len(),
            max = WAKEUP_TAG_MAX_BYTES,
            "x-dyn-wakeup ignored: empty or too long"
        );
        return None;
    }
    Some(WakeupTag(tag.to_string()))
}

/// Prompt-side inputs for a hint on one selection.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrefetchHintInput<'a> {
    /// The prompt tokens as the handle the caller already owns, so the scheduler
    /// clones a handle rather than the tokens when a trigger is installed.
    pub shared_tokens: Option<&'a Arc<Vec<u32>>>,
    /// Ask for the hint as soon as the request is queued (a wake-up).
    pub on_arrival: bool,
}

struct RecentTags {
    tags: HashMap<String, Instant>,
    last_prune: Instant,
}

/// Recent wake-up prefetches by session tag, so a client's explicit
/// `/v1/hints/prefetch` and its real request's `x-dyn-wakeup` header within the
/// dedup window produce one hint, not two. Expired tags are pruned once per window
/// (or when the map is full), not on every call; the map holds at most
/// [`WAKEUP_MAX_TRACKED_TAGS`] tags.
pub struct WakeupDedup {
    recent: parking_lot::Mutex<RecentTags>,
    window: Duration,
}

impl WakeupDedup {
    pub fn new(window: Duration) -> Self {
        Self {
            recent: parking_lot::Mutex::new(RecentTags {
                tags: HashMap::new(),
                last_prune: Instant::now(),
            }),
            window,
        }
    }

    /// Record a prefetch for `tag` now; `false` when one fired within the window.
    pub fn mark(&self, tag: &str) -> bool {
        let now = Instant::now();
        let mut recent = self.recent.lock();
        if now.duration_since(recent.last_prune) >= self.window
            || recent.tags.len() >= WAKEUP_MAX_TRACKED_TAGS
        {
            let window = self.window;
            recent.tags.retain(|_, at| now.duration_since(*at) < window);
            recent.last_prune = now;
        }
        if recent
            .tags
            .get(tag)
            .is_some_and(|at| now.duration_since(*at) < self.window)
        {
            return false;
        }
        if recent.tags.len() >= WAKEUP_MAX_TRACKED_TAGS {
            tracing::debug!(
                max = WAKEUP_MAX_TRACKED_TAGS,
                "wake-up dedup table full; hint sent without recording the tag"
            );
            return true;
        }
        recent.tags.insert(tag.to_string(), now);
        true
    }
}

/// Read a `DYN_KVM_*_MS` duration; `0`, unset, or invalid values mean disabled.
fn env_duration_ms(name: &str) -> Option<Duration> {
    let value = std::env::var(name).ok()?;
    match value.trim().parse::<u64>() {
        Ok(0) => None,
        Ok(ms) => Some(Duration::from_millis(ms)),
        Err(error) => {
            tracing::warn!(env = name, %value, %error, "ignoring invalid duration");
            None
        }
    }
}

/// Read a numeric knob; unset, empty, or unparsable values fall back to `default`.
fn env_parsed<T: FromStr>(name: &str, default: T) -> T {
    let Some(value) = std::env::var(name).ok() else {
        return default;
    };
    let value = value.trim();
    if value.is_empty() {
        return default;
    }
    value.parse::<T>().unwrap_or_else(|_| {
        tracing::warn!(env = name, value, "invalid value; using default");
        default
    })
}

fn hint_config_from_env() -> HintConfig {
    let defaults = HintConfig::default();
    HintConfig {
        queue_depth: env_parsed(env_llm::DYN_KVM_QUEUE_DEPTH, defaults.queue_depth).max(1),
        concurrency: env_parsed(env_llm::DYN_KVM_CONCURRENCY, defaults.concurrency).max(1),
        timeout: env_duration_ms(env_llm::DYN_KVM_TIMEOUT_MS).unwrap_or(defaults.timeout),
    }
}

fn published_endpoint(
    runtime: &crate::local_model::runtime_config::ModelRuntimeConfig,
) -> Option<String> {
    runtime
        .get_engine_specific::<String>(HINT_ZMQ_ENDPOINT_KEY)
        .ok()
        .flatten()
        .filter(|endpoint| !endpoint.is_empty())
}

fn endpoints_by_worker(configs: &RuntimeConfigWatch) -> HashMap<u64, String> {
    configs
        .borrow()
        .iter()
        .filter_map(|(worker_id, runtime)| Some((*worker_id, published_endpoint(runtime)?)))
        .collect()
}

/// Close the hint socket of every worker that leaves the runtime-config watch (or
/// moves to another endpoint) while no remaining worker uses that endpoint. Ends
/// when the watch does.
async fn forget_removed_workers(mut configs: RuntimeConfigWatch, transport: Arc<ZmqHintTransport>) {
    let mut known = endpoints_by_worker(&configs);
    while configs.changed().await.is_ok() {
        let current = endpoints_by_worker(&configs);
        let in_use: HashSet<&String> = current.values().collect();
        for (worker_id, endpoint) in &known {
            if current.get(worker_id) != Some(endpoint) && !in_use.contains(endpoint) {
                transport.forget(endpoint).await;
            }
        }
        known = current;
    }
}

/// Where each worker's hints go: the endpoint it publishes, unless
/// `DYN_KVM_HINT_ENDPOINT` pins one endpoint for all of them.
fn hint_endpoints(configs: &RuntimeConfigWatch) -> HintEndpoints {
    if let Some(endpoint) = std::env::var(env_llm::DYN_KVM_HINT_ENDPOINT)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        tracing::warn!(
            endpoint,
            "{} is set: every worker's prefetch hints go to this one scheduler socket \
             instead of the hint_zmq_endpoint each worker publishes",
            env_llm::DYN_KVM_HINT_ENDPOINT
        );
        return HintEndpoints::Fixed(endpoint);
    }
    let configs = configs.clone();
    let resolver: HintEndpointResolver = Arc::new(move |worker_id| {
        configs
            .borrow()
            .get(&worker_id)
            .and_then(published_endpoint)
    });
    HintEndpoints::PerWorker(resolver)
}

pub(crate) fn worker_target(worker: WorkerWithDpRank) -> WorkerTarget {
    WorkerTarget::new(worker.worker_id, Some(worker.dp_rank))
}

/// Build the controller for the workers in `configs` when a hint consumer is enabled.
pub fn hint_controller_for(configs: RuntimeConfigWatch) -> Option<HintController> {
    let enabled = env_duration_ms(env_llm::DYN_KVM_PREFETCH_LEAD_MS).is_some()
        || env_is_truthy(env_llm::DYN_ENABLE_KVM_HINTS_API)
        || wakeup_header_enabled()
        || pin_endpoint().is_some();
    if !enabled {
        return None;
    }
    let config = hint_config_from_env();
    let endpoints = hint_endpoints(&configs);
    let per_worker = matches!(endpoints, HintEndpoints::PerWorker(_));
    let transport = Arc::new(ZmqHintTransport::new(
        endpoints,
        config.timeout,
        config.concurrency,
    ));
    if per_worker {
        tokio::spawn(forget_removed_workers(configs, Arc::clone(&transport)));
    }
    tracing::info!(
        queue_depth = config.queue_depth,
        concurrency = config.concurrency,
        timeout_ms = config.timeout.as_millis() as u64,
        "KV hint controller enabled (ZMQ to scheduler)"
    );
    let controller = HintController::new(config, transport);
    Some(match pin_endpoint() {
        Some(endpoint) => {
            tracing::info!(%endpoint, "KV pin hints enabled (ZMQ to the shared store)");
            controller.with_pin_sink(Arc::new(ZmqStorePinSink::new(endpoint, PIN_SEND_HWM)))
        }
        None => controller,
    })
}

/// Frames a slow or absent store may hold before pins fail; pins are advisory.
const PIN_SEND_HWM: i32 = 1_000;
const DEFAULT_PIN_TTL: Duration = Duration::from_secs(120);

/// The shared store's hint socket (`DYN_KVM_PIN_ENDPOINT`), read once per process.
fn pin_endpoint() -> Option<&'static str> {
    static ENDPOINT: OnceLock<Option<String>> = OnceLock::new();
    ENDPOINT
        .get_or_init(|| {
            std::env::var(env_llm::DYN_KVM_PIN_ENDPOINT)
                .ok()
                .filter(|endpoint| !endpoint.trim().is_empty())
        })
        .as_deref()
}

/// Whether the frontend pins a request's pages when its response calls a tool.
pub fn tool_call_pins_enabled() -> bool {
    pin_endpoint().is_some()
}

/// Request-context key of a request's [`ToolCallPinSlot`].
pub const TOOL_CALL_PIN_CONTEXT_KEY: &str = "kv_hint_tool_call_pin";

/// Request header saying the response is followed by a tool call the client will
/// run, for clients whose responses carry no parsable tool call (a load generator
/// replaying token ids): `x-dynamo-sim-tool-call: <tool>[; expected_ms=<ms>]`. The
/// response's end then fires the same pin a parsed tool call would.
pub const SIM_TOOL_CALL_HEADER: &str = "x-dynamo-sim-tool-call";
/// Request-context key carrying the parsed [`SIM_TOOL_CALL_HEADER`].
pub const SIM_TOOL_CALL_CONTEXT_KEY: &str = "kv_hint_sim_tool_call";
/// Longest tool name honored in [`SIM_TOOL_CALL_HEADER`].
const SIM_TOOL_NAME_MAX_BYTES: usize = 64;

/// The tool call a client announced with [`SIM_TOOL_CALL_HEADER`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimToolCall {
    pub tool: String,
    /// How long the client expects the tool to run, when it says.
    pub expected: Option<Duration>,
}

impl SimToolCall {
    /// `<tool>[; expected_ms=<ms>]`; unknown parameters are ignored.
    pub fn parse(value: &str) -> Option<Self> {
        let mut parts = value.split(';');
        let tool = parts.next()?.trim();
        if tool.is_empty()
            || tool.len() > SIM_TOOL_NAME_MAX_BYTES
            || !tool.bytes().all(|b| b.is_ascii_graphic())
        {
            return None;
        }
        let expected = parts.find_map(|part| {
            let (name, value) = part.split_once('=')?;
            (name.trim() == "expected_ms")
                .then(|| value.trim().parse::<u64>().ok())
                .flatten()
                .map(Duration::from_millis)
        });
        Some(Self {
            tool: tool.to_string(),
            expected,
        })
    }
}

/// The announced tool call on a request, when tool-call pins are enabled.
pub fn sim_tool_call_from_headers(headers: &HeaderMap) -> Option<SimToolCall> {
    if !tool_call_pins_enabled() {
        return None;
    }
    SimToolCall::parse(headers.get(SIM_TOOL_CALL_HEADER)?.to_str().ok()?)
}

/// What fired a request's pin: the tool's name and the client's own estimate.
#[derive(Debug, Clone, Default)]
pub struct FiredToolCall {
    pub tool: Option<String>,
    pub expected: Option<Duration>,
}

type ArmedPin = Box<dyn Fn(&FiredToolCall) + Send + Sync>;

/// One request's tool-call pin. The preprocessor puts an empty slot in the request
/// context, the router arms it with the worker it routed to, and the preprocessor
/// fires it when the response first carries a tool call (or, for a request carrying
/// [`SIM_TOOL_CALL_HEADER`], when the response ends): the session is about to wait
/// on a tool, so its pages should outlive the wait. Keys are computed on fire, off
/// the routing path.
#[derive(Default)]
pub struct ToolCallPinSlot {
    pin: OnceLock<ArmedPin>,
}

impl ToolCallPinSlot {
    pub(crate) fn arm(&self, pin: impl Fn(&FiredToolCall) + Send + Sync + 'static) {
        let _ = self.pin.set(Box::new(pin));
    }

    /// Send the armed pin, if the router armed one.
    pub fn fire(&self, call: &FiredToolCall) {
        match self.pin.get() {
            Some(pin) => pin(call),
            None => tracing::debug!("tool call seen; no pin armed for this request"),
        }
    }
}

type ChatChunk = dynamo_runtime::protocols::annotated::Annotated<
    crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse,
>;

/// Fire `slot` once: on the first chunk whose delta carries a tool call, or, when
/// the client announced one with [`SIM_TOOL_CALL_HEADER`], at the response's end.
pub fn fire_on_tool_call<S>(
    stream: S,
    slot: Option<Arc<ToolCallPinSlot>>,
    announced: Option<Arc<SimToolCall>>,
) -> std::pin::Pin<Box<dyn futures::Stream<Item = ChatChunk> + Send>>
where
    S: futures::Stream<Item = ChatChunk> + Send + 'static,
{
    use futures::StreamExt;
    use std::sync::atomic::{AtomicBool, Ordering};
    let Some(slot) = slot else {
        return Box::pin(stream);
    };
    let fired = Arc::new(AtomicBool::new(false));
    let on_chunk = Arc::clone(&fired);
    let on_chunk_slot = Arc::clone(&slot);
    let chunks = stream.inspect(move |response| {
        if on_chunk.load(Ordering::Relaxed) {
            return;
        }
        let tool = response.data.as_ref().and_then(|chunk| {
            chunk.inner.choices.iter().find_map(|choice| {
                let call = choice.delta.tool_calls.as_ref()?.first()?;
                Some(call.function.as_ref().and_then(|f| f.name.clone()))
            })
        });
        if let Some(tool) = tool
            && !on_chunk.swap(true, Ordering::Relaxed)
        {
            on_chunk_slot.fire(&FiredToolCall {
                tool,
                expected: None,
            });
        }
    });
    let Some(announced) = announced else {
        return Box::pin(chunks);
    };
    let end = futures::stream::once(async move {
        if !fired.swap(true, Ordering::Relaxed) {
            slot.fire(&FiredToolCall {
                tool: Some(announced.tool.clone()),
                expected: announced.expected,
            });
        }
    })
    .filter_map(|()| async { None });
    Box::pin(chunks.chain(end))
}

/// Builds tool-call pins for one router's workers.
pub struct ToolCallPinner {
    controller: HintController,
    configs: RuntimeConfigWatch,
    policy: dynamo_kvm::policy::DefaultPinPolicy,
    /// `DYN_KVM_PIN_PRIORITY`, when set, instead of the policy's score.
    fixed_priority: Option<u32>,
}

impl ToolCallPinner {
    /// Arm `slot` to pin the full pages of `tokens` on `worker_id`.
    pub(crate) fn arm(
        self: &Arc<Self>,
        slot: &ToolCallPinSlot,
        worker_id: u64,
        tokens: Arc<[u32]>,
    ) {
        let pinner = Arc::clone(self);
        slot.arm(move |call| pinner.pin(worker_id, &tokens, call));
    }

    fn pin(&self, worker_id: u64, tokens: &[u32], call: &FiredToolCall) {
        use dynamo_kvm::policy::{PinPolicy, SessionSignals, ToolCallInfo};
        let keys = self.configs.borrow().get(&worker_id).and_then(|runtime| {
            crate::kv_router::shared_cache::mooncake_pin_keys(worker_id, runtime, tokens)
        });
        let Some((tenant, keys)) = keys else {
            tracing::debug!(
                worker_id,
                "tool-call pin skipped: worker publishes no Mooncake object layout"
            );
            return;
        };
        let decision = self.policy.decide(
            &ToolCallInfo {
                name: call.tool.as_deref(),
                arguments: None,
                expected_duration: call.expected,
            },
            &SessionSignals {
                recompute_tokens: Some(tokens.len() as u64),
                ..SessionSignals::default()
            },
        );
        let priority = self.fixed_priority.unwrap_or(decision.priority);
        tracing::debug!(
            worker_id,
            keys = keys.len(),
            tool = ?call.tool,
            ttl_ms = decision.ttl.as_millis() as u64,
            priority,
            "tool-call pin"
        );
        self.controller.pin(PinHint {
            tenant,
            keys: keys.into(),
            ttl: decision.ttl,
            priority,
        });
    }
}

/// The tool-call pinner for `controller`, when `DYN_KVM_PIN_ENDPOINT` is set.
pub fn tool_call_pinner(
    controller: Option<&HintController>,
    configs: RuntimeConfigWatch,
) -> Option<Arc<ToolCallPinner>> {
    use dynamo_kvm::policy::{
        DefaultPinPolicy, QuantileConfig, QuantilePredictor, RecomputeTokensScorer, ToolNameKeyer,
        TtlBounds,
    };
    pin_endpoint()?;
    let bounds = TtlBounds {
        default: env_duration_ms(env_llm::DYN_KVM_PIN_TTL_MS).unwrap_or(DEFAULT_PIN_TTL),
        ..TtlBounds::default()
    };
    Some(Arc::new(ToolCallPinner {
        controller: controller?.clone(),
        configs,
        policy: DefaultPinPolicy::new(
            Box::new(ToolNameKeyer),
            Box::new(QuantilePredictor::new(QuantileConfig::default())),
            Box::new(RecomputeTokensScorer::default()),
            bounds,
        ),
        fixed_priority: std::env::var(env_llm::DYN_KVM_PIN_PRIORITY)
            .ok()
            .and_then(|value| value.trim().parse().ok()),
    }))
}

/// The wake-up header's dedup state, when `DYN_KVM_WAKEUP_HEADER` is on.
pub fn wakeup_dedup_from_env() -> Option<WakeupDedup> {
    wakeup_header_enabled().then(|| {
        WakeupDedup::new(
            env_duration_ms(env_llm::DYN_KVM_WAKEUP_DEDUP_MS)
                .unwrap_or(Duration::from_millis(DEFAULT_WAKEUP_DEDUP_MS)),
        )
    })
}

struct ControllerSink(HintController);

impl PrefetchHintSink for ControllerSink {
    fn prefetch(&self, worker: WorkerWithDpRank, tokens: &[u32]) {
        self.0.prefetch(worker_target(worker), Arc::from(tokens));
    }
}

/// The deadline-led prefetch trigger for `controller`, when `DYN_KVM_PREFETCH_LEAD_MS` is set.
pub fn prefetch_trigger(controller: Option<&HintController>) -> Option<PrefetchTrigger> {
    let lead = env_duration_ms(env_llm::DYN_KVM_PREFETCH_LEAD_MS)?;
    Some(PrefetchTrigger {
        lead,
        sink: Arc::new(ControllerSink(controller?.clone())),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sim_tool_call_header_parses_tool_and_estimate() {
        assert_eq!(
            SimToolCall::parse("shell; expected_ms=20000"),
            Some(SimToolCall {
                tool: "shell".into(),
                expected: Some(Duration::from_millis(20_000)),
            })
        );
        assert_eq!(
            SimToolCall::parse(" read ;foo=1"),
            Some(SimToolCall {
                tool: "read".into(),
                expected: None,
            })
        );
        assert_eq!(SimToolCall::parse(""), None);
        assert_eq!(SimToolCall::parse("has space"), None);
        assert_eq!(SimToolCall::parse(&"x".repeat(65)), None);
        assert_eq!(
            SimToolCall::parse("shell; expected_ms=soon")
                .unwrap()
                .expected,
            None
        );
    }

    fn text_chunk() -> ChatChunk {
        serde_json::from_value(serde_json::json!({
            "data": {
                "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}}]
            }
        }))
        .unwrap()
    }

    fn recording_slot() -> (
        Arc<ToolCallPinSlot>,
        Arc<std::sync::Mutex<Vec<FiredToolCall>>>,
    ) {
        let slot = Arc::new(ToolCallPinSlot::default());
        let fired = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = Arc::clone(&fired);
        slot.arm(move |call| seen.lock().unwrap().push(call.clone()));
        (slot, fired)
    }

    #[tokio::test]
    async fn an_announced_tool_call_fires_once_at_the_end() {
        use futures::StreamExt;
        let (slot, fired) = recording_slot();
        let announced = Arc::new(SimToolCall {
            tool: "shell".into(),
            expected: Some(Duration::from_secs(20)),
        });
        let out: Vec<_> = fire_on_tool_call(
            futures::stream::iter(vec![text_chunk(), text_chunk()]),
            Some(slot),
            Some(announced),
        )
        .collect()
        .await;
        assert_eq!(out.len(), 2, "the stream passes through unchanged");
        let fired = fired.lock().unwrap();
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].tool.as_deref(), Some("shell"));
        assert_eq!(fired[0].expected, Some(Duration::from_secs(20)));
    }

    #[tokio::test]
    async fn plain_text_without_an_announcement_does_not_fire() {
        use futures::StreamExt;
        let (slot, fired) = recording_slot();
        let _: Vec<_> =
            fire_on_tool_call(futures::stream::iter(vec![text_chunk()]), Some(slot), None)
                .collect()
                .await;
        assert!(fired.lock().unwrap().is_empty());
    }
    use crate::local_model::runtime_config::ModelRuntimeConfig;
    use axum::http::HeaderValue;

    fn runtime_config(endpoint: Option<&str>) -> ModelRuntimeConfig {
        let mut config = ModelRuntimeConfig::default();
        if let Some(endpoint) = endpoint {
            config
                .set_engine_specific(HINT_ZMQ_ENDPOINT_KEY, endpoint)
                .unwrap();
        }
        config
    }

    /// Endpoint resolution reads the runtime-config watch and nothing else, so a
    /// worker whose sidecar publishes `hint_zmq_endpoint` is reachable the moment
    /// discovery delivers its config and unreachable once it leaves.
    #[test]
    fn endpoints_follow_the_runtime_config_watch() {
        let (sender, watch) = tokio::sync::watch::channel(HashMap::from([
            (1u64, runtime_config(Some("tcp://10.0.0.7:5558"))),
            (2u64, runtime_config(None)),
        ]));
        let HintEndpoints::PerWorker(resolver) = hint_endpoints(&watch) else {
            panic!("per-worker resolution when DYN_KVM_HINT_ENDPOINT is unset");
        };
        assert_eq!(resolver(1).as_deref(), Some("tcp://10.0.0.7:5558"));
        assert_eq!(resolver(2), None);
        assert_eq!(resolver(3), None);

        sender.send_modify(|configs| {
            configs.remove(&1);
            configs.insert(3, runtime_config(Some("tcp://10.0.0.8:5558")));
        });
        assert_eq!(resolver(1), None);
        assert_eq!(resolver(3).as_deref(), Some("tcp://10.0.0.8:5558"));
        assert_eq!(
            endpoints_by_worker(&watch),
            HashMap::from([(3u64, "tcp://10.0.0.8:5558".to_string())])
        );
    }

    #[test]
    fn wakeup_dedup_skips_a_tag_inside_the_window_only() {
        let dedup = WakeupDedup::new(Duration::from_millis(50));
        assert!(dedup.mark("session-a"));
        assert!(!dedup.mark("session-a"));
        assert!(dedup.mark("session-b"));
        std::thread::sleep(Duration::from_millis(60));
        assert!(dedup.mark("session-a"));
    }

    #[test]
    fn wakeup_dedup_stops_recording_when_full() {
        let dedup = WakeupDedup::new(Duration::from_secs(60));
        for i in 0..WAKEUP_MAX_TRACKED_TAGS {
            assert!(dedup.mark(&format!("tag-{i}")));
        }
        // Full: a new tag is still sent, but not remembered.
        assert!(dedup.mark("overflow"));
        assert!(dedup.mark("overflow"));
        assert_eq!(dedup.recent.lock().tags.len(), WAKEUP_MAX_TRACKED_TAGS);
    }

    #[test]
    fn wakeup_tag_is_bounded() {
        let mut headers = HeaderMap::new();
        headers.insert(WAKEUP_HEADER, HeaderValue::from_static("session-1"));
        assert_eq!(wakeup_tag_from_headers(&headers).unwrap().0, "session-1");
        headers.insert(
            WAKEUP_HEADER,
            HeaderValue::from_str(&"x".repeat(WAKEUP_TAG_MAX_BYTES + 1)).unwrap(),
        );
        assert!(wakeup_tag_from_headers(&headers).is_none());
        headers.insert(WAKEUP_HEADER, HeaderValue::from_static(""));
        assert!(wakeup_tag_from_headers(&headers).is_none());
    }
}
