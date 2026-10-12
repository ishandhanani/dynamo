// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone (selector) endpoint picker.
//!
//! This is the runtime-free counterpart to [`crate::epp::Router`]. It runs with
//! no Dynamo `DistributedRuntime`, no etcd/NATS, and no embedded KV router.
//! Instead it composes:
//!
//! - a [`RenderClient`] that tokenizes prompts via a render sidecar,
//! - a [`PodDiscovery`] that discovers Ready worker pods from Kubernetes,
//! - a [`TopologyAdapter`] that registers those pods into the selector, and
//! - a [`Selector`] (in-process, runtime-free selection service) that picks a
//!   worker.
//!
//! On each request it tokenizes the prompt, asks the selection service for a
//! worker constrained to the currently-Ready pods, and tells Envoy where to send
//! the request via routing headers.

use std::collections::HashSet;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use tokio::sync::Semaphore;

use dynamo_kv_router::DEFAULT_ROUTING_GROUP;
use dynamo_kv_router::services::selection::{PromptRequest, SelectAndReserveRequest};
use dynamo_kv_router::services::selection::{SelectionError, WorkerSelectionPolicyRegistry};
use dynamo_llm::http::service::metadata::extract_metadata_from_header_pairs;
use dynamo_llm::protocols::common::extensions::{
    agent_context_from_headers, apply_frontend_nvext_policy, request_routing_hints,
    resolve_cache_namespace, session_affinity_from_headers, to_selection_session_context,
};
use dynamo_llm::protocols::common::timing::RequestPhase;

use crate::epp_standalone_config::{EppStandaloneConfig, RendererProtocol};
use crate::picker::{
    CacheSaltForwarding, Endpoint, EndpointPicker, PickError, PickResult, RequestInfo,
    request_header_map,
};
use crate::pod_discovery::PodDiscovery;
use crate::render_http::RenderError;
use crate::request::{RemoteRequest, renderer_request_body};
use crate::selector::Selector;
use crate::sglang_renderer_client::SglangRendererClient;
use crate::topology_adapter::{RegistrationDefaults, TopologyAdapter};
use crate::vllm_render_client::VllmRenderClient;

/// Resolve the request's scheduling policy class from the Dynamo metadata
/// headers. Goes through the frontend's metadata extractor (rather than a
/// hardcoded header name) so custom `DYN_METADATA_HEADER` prefixes, trimming,
/// and duplicate handling stay aligned with the integrated router.
pub(crate) fn requested_policy_class(
    headers: &[(String, String)],
) -> Result<Option<String>, PickError> {
    let metadata =
        extract_metadata_from_header_pairs(headers.iter().map(|(k, v)| (k.as_str(), v.as_str())))
            .map_err(PickError::MetadataHeadersTooLarge)?;
    Ok(metadata.get("policy-class").cloned())
}

/// Protocol-dispatched render client for the standalone EPP.
enum RenderClient {
    Vllm(VllmRenderClient),
    Sglang(SglangRendererClient),
}

impl RenderClient {
    async fn render(&self, body: bytes::Bytes, completion: bool) -> Result<Vec<u32>, RenderError> {
        match (self, completion) {
            (Self::Vllm(c), false) => c.render_chat(body).await,
            (Self::Sglang(c), false) => c.render_chat(body).await,
            (Self::Vllm(c), true) => c.render_completion(body).await,
            (Self::Sglang(c), true) => c.render_completion(body).await,
        }
    }
}

/// Standalone endpoint picker backed by the standalone selection service.
pub struct EppRouter {
    renderer: RenderClient,
    reflector: Arc<PodDiscovery>,
    selector: Arc<Selector>,
    // Kept alive for the lifetime of the router; the reconcile loop runs on it.
    _adapter: TopologyAdapter,
    reflector_ready: Arc<AtomicBool>,
    model_name: String,
    nvext_enabled: bool,
    /// Bounds total concurrent in-flight `pick()`s. HTTP/2 stream multiplexing
    /// means the TCP-connection cap (`MAX_CONCURRENT_CONNECTIONS`) does NOT bound
    /// requests, so without this a burst could fan out unbounded tokenizer/render
    /// calls and buffer unbounded request bodies. A permit is taken per `pick()`
    /// and released (RAII) when it returns or is dropped/cancelled; when none are
    /// available the request is shed with `PickError::Overloaded` (not queued).
    inflight: Arc<Semaphore>,
}

impl EppRouter {
    /// Assemble the standalone runtime from the validated selector config.
    pub async fn from_selector(
        cfg: EppStandaloneConfig,
        policy_registry: WorkerSelectionPolicyRegistry,
    ) -> Result<Self> {
        let selector = Arc::new(Selector::new(&cfg, policy_registry).await?);
        let timeout = Duration::from_millis(cfg.tokenization_timeout_ms);
        let max_response_bytes = cfg.tokenizer_max_response_bytes;
        let renderer = match cfg.renderer_protocol {
            RendererProtocol::VllmRender => RenderClient::Vllm(VllmRenderClient::new(
                &cfg.tokenizer_service_url,
                timeout,
                max_response_bytes,
            )?),
            RendererProtocol::SglangRenderer => RenderClient::Sglang(SglangRendererClient::new(
                &cfg.tokenizer_service_url,
                timeout,
                max_response_bytes,
            )?),
        };
        let (reflector, reflector_ready) = PodDiscovery::spawn(&cfg).await?;
        let reflector = Arc::new(reflector);
        let defaults = RegistrationDefaults::from_config(&cfg);
        let adapter =
            TopologyAdapter::spawn(reflector.as_ref().clone(), selector.clone(), defaults);

        // Readiness is driven solely by the live pod+pool signal (see `is_ready`);
        // we do not block startup on a schedulable worker. A valid, empty pool is
        // ready immediately and returns 503 per-request until capacity appears.
        Ok(Self {
            renderer,
            reflector,
            selector,
            _adapter: adapter,
            reflector_ready,
            model_name: cfg.model_name,
            nvext_enabled: !dynamo_runtime::config::env_is_truthy("DYN_DISABLE_FRONTEND_NVEXT"),
            inflight: Arc::new(Semaphore::new(cfg.max_inflight_requests)),
        })
    }

    /// Overall EPP readiness for the gRPC health signal: the pod reflector has
    /// synced workers and resolved its InferencePool. Polled by the health mirror in `main`.
    pub fn is_ready(&self) -> bool {
        self.reflector_ready.load(Ordering::Acquire)
    }

    /// Prepare canonical selection inputs using the same NvExt and header policy as
    /// the frontend. The remote renderer retains ownership of prompt semantics.
    async fn prepare_request(
        &self,
        req: &RequestInfo,
        reservation_id: String,
        allowed_worker_ids: Option<HashSet<u64>>,
    ) -> Result<SelectAndReserveRequest, PickError> {
        prepare_remote_request(
            &self.renderer,
            &self.selector,
            &self.model_name,
            self.nvext_enabled,
            req,
            reservation_id,
            allowed_worker_ids,
        )
        .await
    }

    /// Ready workers inside an Envoy `candidate_subset`, resolved in a single index
    /// pass (no full-ready set materialized). The reflector's endpoints are
    /// scheme-less `ip:port`, so a worker matches the subset's full `ip:port` or
    /// bare `ip`; empty means nothing matched.
    fn subset_worker_ids(&self, candidate_subset: &[String]) -> HashSet<u64> {
        let candidates: HashSet<&str> = candidate_subset.iter().map(String::as_str).collect();
        let candidate_ips: HashSet<IpAddr> = candidate_subset
            .iter()
            .filter_map(|candidate| candidate.parse().ok())
            .collect();
        // Single index pass; the predicate borrows each endpoint (no clone).
        self.reflector.ready_worker_ids_matching(|endpoint| {
            endpoint_in_subset(endpoint, &candidates, &candidate_ips)
        })
    }
}

/// True if a scheme-less `ip:port` endpoint is covered by an Envoy subset,
/// matching either the full `ip:port` or the bare `ip`.
///
/// Matches the bare-IP case via `IpAddr`, never `endpoint.split(':')`: a
/// bracketed IPv6 endpoint (`[fd00::2]:8000`) splits into garbage on `:`,
/// silently never matching a bare `fd00::2` candidate. Shared with
/// [`crate::epp::Router::subset_to_worker_ids`], the other Envoy
/// candidate_subset matcher in this crate.
pub(crate) fn endpoint_in_subset(
    endpoint: &str,
    candidates: &HashSet<&str>,
    candidate_ips: &HashSet<IpAddr>,
) -> bool {
    candidates.contains(endpoint)
        || endpoint
            .parse::<SocketAddr>()
            .is_ok_and(|address| candidate_ips.contains(&address.ip()))
}

/// Kept separate from Kubernetes discovery so the actual rendering-to-selection
/// adapter can be exercised against a local renderer and selection service.
#[allow(clippy::too_many_arguments)]
async fn prepare_remote_request(
    renderer: &RenderClient,
    selector: &Selector,
    model_name: &str,
    nvext_enabled: bool,
    req: &RequestInfo,
    reservation_id: String,
    allowed_worker_ids: Option<HashSet<u64>>,
) -> Result<SelectAndReserveRequest, PickError> {
    let mut envelope: RemoteRequest =
        serde_json::from_slice(&req.body).map_err(PickError::invalid_request_body)?;
    let completion = envelope.is_completion()?;
    if envelope.model != model_name {
        return Err(PickError::InvalidRequest("request model must match the configured standalone model; LoRA aliases are unsupported".into()));
    }
    let headers = request_header_map(&req.headers)?;
    let mut session = agent_context_from_headers(&headers);
    if let Some(session) = &mut session {
        session.input_trigger = Some(envelope.input_trigger());
    }
    let nvext = apply_frontend_nvext_policy(envelope.nvext.take(), &headers, nvext_enabled);
    if nvext
        .as_ref()
        .is_some_and(|ext| ext.token_data.is_some() || ext.use_raw_prompt == Some(true))
    {
        return Err(PickError::InvalidRequest(
            "remote renderers do not implement nvext.token_data or nvext.use_raw_prompt".into(),
        ));
    }
    // The frontend policy above has already applied the tenant header to NvExt.
    let cache_namespace = resolve_cache_namespace(
        None,
        nvext.as_ref().and_then(|ext| ext.cache_salt.as_deref()),
        envelope.cache_namespace.as_deref(),
    )
    .map(str::to_owned);
    let routing = request_routing_hints(nvext.as_ref(), None, cache_namespace).unwrap_or_default();
    if routing.prefill_worker_id.is_some() || routing.prefill_dp_rank.is_some() {
        return Err(PickError::InvalidRequest(
            "standalone EPP does not support prefill worker pins".into(),
        ));
    }
    let pinned_worker = routing
        .worker_pin(RequestPhase::Aggregated)
        .map(|(worker_id, rank)| selector.resolve_pinned_worker(worker_id, rank))
        .transpose()
        .map_err(|error| PickError::InvalidRequest(error.to_string()))?;
    let token_ids = renderer
        .render(renderer_request_body(&req.body)?, completion)
        .await
        .map_err(|error| render_error_to_pick_error(error, &req.request_id))?;
    Ok(SelectAndReserveRequest {
        model_name: model_name.to_owned(),
        routing_group: DEFAULT_ROUTING_GROUP.to_string(),
        selection_id: Some(reservation_id),
        prompt: PromptRequest {
            token_ids: Some(token_ids),
            cache_namespace: routing.cache_namespace,
            lora_name: routing.lora_name,
            ..Default::default()
        },
        router_config_override: None,
        expected_output_tokens: routing.expected_output_tokens,
        priority_jump: routing.priority_jump,
        strict_priority: routing.strict_priority,
        session_id: session_affinity_from_headers(&headers).map(|id| id.as_str().to_owned()),
        session_context: session.as_ref().map(to_selection_session_context),
        affinity_target: None,
        pinned_worker,
        allowed_worker_ids,
        routing_constraints: routing.routing_constraints.unwrap_or_default(),
    })
}

#[tonic::async_trait]
impl EndpointPicker for EppRouter {
    async fn pick(
        &self,
        req: &RequestInfo,
        _endpoints: &[Endpoint],
    ) -> Result<PickResult, PickError> {
        if !self.reflector_ready.load(Ordering::Acquire) {
            return Err(PickError::RoutingFailed(
                "pod reflector cache not ready".to_string(),
            ));
        }

        if !self.reflector.has_ready_workers() {
            return Err(PickError::NoEndpoints);
        }

        // Bound total in-flight picks. This caps the tokenizer/render fan-out,
        // `select_and_reserve`, and the buffered request bodies held for the
        // duration of the pick — the connection cap does NOT, because HTTP/2 stream
        // multiplexing lets one connection carry unbounded concurrent requests.
        // `try_acquire_owned` sheds (never blocks/awaits) so we don't grow an
        // unbounded wait queue; the permit is held until `pick()` returns or the
        // future is dropped/cancelled, releasing it (RAII).
        let _inflight_permit = self
            .inflight
            .clone()
            .try_acquire_owned()
            .map_err(|_| PickError::Overloaded)?;

        // Ordinary path: pass `None` so the SelectionService schedules over its
        // own catalog ("selector owns eligibility") — no O(worker-count) id set is
        // built per request. We accept that the catalog lags the reflector by ~ms
        // after a pod event: the system already tolerates far larger staleness
        // (pod readiness), and the post-select `resolve_endpoint` guard still
        // refuses to route to a worker the reflector can no longer resolve. The
        // freshness-preserving alternative (re-assert the ready set every request)
        // would need an `Arc`-shared set threaded through the core to stay O(1) —
        // not worth the complexity. Only a subset hint (info the selector lacks)
        // needs an explicit id set, built lazily below.
        let allowed: Option<HashSet<u64>> = if req.candidate_subset.is_empty() {
            None
        } else {
            // Honor Envoy's subset hint (`x-gateway-destination-endpoint-subset`):
            // constrain to Ready workers in the subset, refusing (not falling back
            // to the full set) when nothing matches.
            let filtered = self.subset_worker_ids(&req.candidate_subset);
            if filtered.is_empty() {
                tracing::warn!(
                    subset = ?req.candidate_subset,
                    "No Ready pod matches the subset hint; refusing to route outside the subset"
                );
                return Err(PickError::NoEndpoints);
            }
            Some(filtered)
        };

        // Body-less requests (no prompt to tokenize) route to any Ready worker,
        // staying inside the subset when one was given.
        if req.body.is_empty() {
            let endpoint = match &allowed {
                Some(ids) => {
                    let worker_id = *ids.iter().next().ok_or(PickError::NoEndpoints)?;
                    self.reflector
                        .resolve_endpoint(worker_id)
                        .ok_or(PickError::NoEndpoints)?
                }
                None => self
                    .reflector
                    .resolve_any_endpoint()
                    .ok_or(PickError::NoEndpoints)?,
            };
            return Ok(PickResult {
                endpoint,
                ..Default::default()
            });
        }

        let policy_class = requested_policy_class(&req.headers)?;
        let reservation_id = uuid::Uuid::new_v4().to_string();
        let select_req = self
            .prepare_request(req, reservation_id.clone(), allowed)
            .await?;
        let cache_namespace = select_req.prompt.cache_namespace.clone();

        // Free the booking if this pick is dropped before it is adopted — the
        // ext-proc stream can close after the scheduler booked but before the
        // server stores `booking_id`, and a booked (past-queue) reservation is not
        // reclaimed by the queue's drop-retraction. Disarmed on the handled paths
        // below; until then, dropping this future frees the reservation.
        let mut reservation_guard =
            ReservationGuard::new(self.selector.clone(), reservation_id.clone());

        // On either error return below the guard (still armed) frees the booking.

        let resp = match self
            .selector
            .select_and_reserve(select_req, policy_class)
            .await
        {
            Ok(resp) => resp,
            Err(SelectionError::BadRequest(message)) => {
                return Err(PickError::InvalidRequest(message));
            }
            Err(e) => return Err(PickError::RoutingFailed(e.to_string())),
        };

        // The reflector owns the address + readiness. If it can no longer resolve
        // the selected worker, the pod left Ready in the race, so the selection is
        // stale: refuse rather than route to a stale address.
        let Some(endpoint) = self.reflector.resolve_endpoint(resp.worker_id) else {
            tracing::warn!(
                worker_id = resp.worker_id,
                "Selected worker no longer resolvable in reflector; treating selection as stale"
            );
            return Err(PickError::NoEndpoints);
        };

        // Success: the caller adopts `reservation_id` synchronously (there is no
        // await between this return and the server storing `booking_id`), so the
        // lifecycle callbacks now own the free — disarm the guard.
        reservation_guard.disarm();

        // Routing comes from the destination mutation; aggregated raw workers
        // read no `x-dynamo-*` headers. (Disaggregated will add its own contract.)
        Ok(PickResult {
            endpoint,
            // Worker re-tokenizes the forwarded request (llm-d parity); no inject.
            token_ids: None,
            cache_namespace,
            cache_salt_forwarding: match &self.renderer {
                RenderClient::Vllm(_) => CacheSaltForwarding::NativeVllm,
                RenderClient::Sglang(_) => CacheSaltForwarding::NativeSglang,
            },
            // Booking id for the server's lifecycle callbacks (no shared map).
            reservation_id: Some(reservation_id),
            ..Default::default()
        })
    }

    /// Response complete: release the booking from `pick`. `booking_id` is that
    /// reservation id; `free_reservation` is idempotent (body-less pick → no-op).
    async fn on_request_complete(&self, booking_id: &str) {
        if let Err(e) = self.selector.free_reservation(booking_id).await {
            tracing::warn!(reservation_id = booking_id, error = %e, "Failed to free reservation");
        }
    }

    /// First token: release prefill load, keep decode booked until completion.
    /// `booking_id` is `pick`'s reservation id; `prefill_complete` is idempotent.
    async fn on_prefill_complete(&self, booking_id: &str) {
        if let Err(e) = self.selector.prefill_complete(booking_id).await {
            tracing::warn!(reservation_id = booking_id, error = %e, "Failed to mark prefill complete");
        }
    }
}

/// Releases a minted reservation when its [`ReservationGuard`] fires. The
/// production impl (`Arc<Selector>`) spawns the idempotent `free_reservation`;
/// tests use a lightweight stub. Kept a monomorphized trait so the guard is a
/// plain struct — no per-request `Box<dyn FnOnce>` allocation on the hot path.
trait ReservationReleaser: Send + 'static {
    fn release(&self, reservation_id: String);
}

impl ReservationReleaser for Arc<Selector> {
    fn release(&self, reservation_id: String) {
        let selector = self.clone();
        tokio::spawn(async move {
            if let Err(e) = selector.free_reservation(&reservation_id).await {
                tracing::debug!(%reservation_id, error = %e, "reservation cleanup on dropped pick");
            }
        });
    }
}

/// RAII cleanup for a minted reservation. Armed when `reservation_id` is minted;
/// if the pick future is dropped before the result is adopted (ext-proc stream
/// closed after a booking), `Drop` releases it (an idempotent `free_reservation`).
/// Disarmed once the pick is handled, so a successful, adopted pick or an error
/// return does not double-free. Holds the releaser + id by value (no boxing).
struct ReservationGuard<R: ReservationReleaser> {
    releaser: R,
    reservation_id: String,
    armed: bool,
}

impl<R: ReservationReleaser> ReservationGuard<R> {
    fn new(releaser: R, reservation_id: String) -> Self {
        Self {
            releaser,
            reservation_id,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl<R: ReservationReleaser> Drop for ReservationGuard<R> {
    fn drop(&mut self) {
        if self.armed {
            self.releaser
                .release(std::mem::take(&mut self.reservation_id));
        }
    }
}

/// Map to a client-safe [`PickError`], logging the detailed cause (which may
/// include upstream URLs/bodies) server-side rather than returning it.
fn render_error_to_pick_error(error: RenderError, request_id: &str) -> PickError {
    tracing::warn!(request_id, error = %error, "Tokenization render failed");
    match error {
        RenderError::Unavailable { .. } => PickError::TokenizerUnavailable,
        RenderError::Timeout { .. } => PickError::TokenizerTimeout,
        RenderError::InvalidResponse { .. } | RenderError::ResponseTooLarge { .. } => {
            PickError::TokenizerUpstreamError
        }
        RenderError::UpstreamStatus { status, .. } => {
            match status.as_u16() {
                // Only payload-validation statuses (400/422) mean the
                // client's request was bad → surface as a client 400.
                // Auth/misconfig (401/403/404), overload (429/503), any
                // other 4xx, and 5xx are the renderer's or our own fault.
                400 | 422 => PickError::InvalidRequest(
                    "request rejected by tokenization service".to_string(),
                ),
                // Renderer overloaded / temporarily unavailable → retryable.
                429 | 503 => PickError::TokenizerUnavailable,
                _ => PickError::TokenizerUpstreamError,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requested_policy_class_uses_frontend_metadata_extraction() {
        // The class rides a Dynamo metadata header; the extractor strips the
        // prefix, trims, and honors the first of repeated headers.
        let headers: Vec<(String, String)> = vec![
            (
                "x-dynamo-meta-policy-class".to_string(),
                " latency ".to_string(),
            ),
            (
                "x-dynamo-meta-policy-class".to_string(),
                "throughput".to_string(),
            ),
            ("x-request-id".to_string(), "irrelevant".to_string()),
        ];
        assert_eq!(
            requested_policy_class(&headers).unwrap().as_deref(),
            Some("latency")
        );

        // Mixed-case header names match as well.
        let headers: Vec<(String, String)> = vec![(
            "X-Dynamo-Meta-Policy-Class".to_string(),
            "express".to_string(),
        )];
        assert_eq!(
            requested_policy_class(&headers).unwrap().as_deref(),
            Some("express")
        );

        // No metadata header → no policy class.
        let headers: Vec<(String, String)> = vec![("x-request-id".to_string(), "r1".to_string())];
        assert_eq!(requested_policy_class(&headers).unwrap(), None);
    }

    #[test]
    fn requested_policy_class_preserves_typed_limit_error() {
        use dynamo_llm::http::service::metadata::MetadataHeaderError;

        let headers: Vec<(String, String)> = (0..65)
            .map(|i| (format!("x-dynamo-meta-key-{i:02}"), "v".to_string()))
            .collect();
        let err = requested_policy_class(&headers).expect_err("65 metadata entries must fail");
        assert!(
            matches!(
                err,
                PickError::MetadataHeadersTooLarge(MetadataHeaderError::TooManyEntries { .. })
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn render_upstream_status_maps_to_correct_pick_error() {
        use reqwest::StatusCode;

        let map = |status: StatusCode| {
            render_error_to_pick_error(
                RenderError::UpstreamStatus {
                    status,
                    body: String::new(),
                },
                "req-1",
            )
        };

        // Renderer validated the client's payload and rejected it → client 400.
        assert!(matches!(
            map(StatusCode::BAD_REQUEST),
            PickError::InvalidRequest(_)
        ));
        assert!(matches!(
            map(StatusCode::UNPROCESSABLE_ENTITY),
            PickError::InvalidRequest(_)
        ));

        // Auth / misconfiguration is NOT an invalid client payload → upstream 502,
        // not a misleading 400.
        for status in [
            StatusCode::UNAUTHORIZED,
            StatusCode::FORBIDDEN,
            StatusCode::NOT_FOUND,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_GATEWAY,
        ] {
            assert!(
                matches!(map(status), PickError::TokenizerUpstreamError),
                "{status} should map to an upstream error, not a client 400"
            );
        }

        // Overloaded / temporarily unavailable → retryable 503.
        assert!(matches!(
            map(StatusCode::TOO_MANY_REQUESTS),
            PickError::TokenizerUnavailable
        ));
        assert!(matches!(
            map(StatusCode::SERVICE_UNAVAILABLE),
            PickError::TokenizerUnavailable
        ));
    }

    #[test]
    fn endpoint_in_subset_matches_ip_port_or_bare_ip() {
        fn matches(endpoint: &str, values: &[&str]) -> bool {
            let candidates: HashSet<&str> = values.iter().copied().collect();
            let candidate_ips: HashSet<IpAddr> = values
                .iter()
                .filter_map(|candidate| candidate.parse().ok())
                .collect();
            endpoint_in_subset(endpoint, &candidates, &candidate_ips)
        }

        // Full ip:port match.
        assert!(matches("10.0.0.1:8000", &["10.0.0.1:8000"]));
        // Bare-ip match (subset lists just the IP).
        assert!(matches("10.0.0.2:8000", &["10.0.0.2"]));
        // Subset pinned a full ip:port, so a different port on that IP does NOT match.
        assert!(!matches("10.0.0.1:9999", &["10.0.0.1:8000"]));
        // Unrelated endpoint does not match.
        assert!(!matches("10.0.0.3:8000", &["10.0.0.2"]));

        // Full bracketed IPv6 endpoint match.
        assert!(matches("[fd00::1]:8000", &["[fd00::1]:8000"]));
        // Bare IPv6 match uses the normalized address, without brackets.
        assert!(matches("[fd00::2]:8000", &["fd00::2"]));
        // A different port does not match a full-endpoint-only candidate.
        assert!(!matches("[fd00::1]:9999", &["[fd00::1]:8000"]));
    }

    #[test]
    fn reservation_guard_frees_on_drop_unless_disarmed() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        // Small test seam: a releaser that records whether it fired.
        struct StubReleaser(Arc<AtomicBool>);
        impl ReservationReleaser for StubReleaser {
            fn release(&self, _reservation_id: String) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        // Dropped while armed — the pick future cancelled after the scheduler
        // booked but before the server adopts the result: cleanup runs.
        let fired = Arc::new(AtomicBool::new(false));
        {
            let _guard = ReservationGuard::new(StubReleaser(fired.clone()), "r1".to_string());
        }
        assert!(fired.load(Ordering::SeqCst));

        // Disarmed (successful, adopted pick): cleanup does not run.
        let fired = Arc::new(AtomicBool::new(false));
        {
            let mut guard = ReservationGuard::new(StubReleaser(fired.clone()), "r1".to_string());
            guard.disarm();
        }
        assert!(!fired.load(Ordering::SeqCst));
    }
    #[tokio::test]
    async fn local_and_remote_metadata_match() {
        use axum::{Json, Router, routing::post};
        use dynamo_kv_router::services::selection::CatalogReconciler;
        use dynamo_llm::preprocessor::OpenAIPreprocessor;
        use serde_json::{Value, json};

        let preprocessor =
            OpenAIPreprocessor::new(crate::epp::tests::model_card("TinyLlama_v1.1")).unwrap();
        let render = |uri: axum::http::Uri, Json(body): Json<Value>| async move {
            assert!(body.get("nvext").is_none());
            let completion = body.get("prompt").is_some();
            assert_eq!(uri.path().ends_with("/v1/completions/render"), completion);
            let tokens = json!({"token_ids":[1,42,99],"input_ids":[1,42,99]});
            Json(if body["n"] == 2 {
                json!([tokens, tokens])
            } else if completion {
                json!([tokens])
            } else {
                tokens
            })
        };
        let prefix = "/prefix/chat/completions/render";
        let app = Router::new()
            .route(
                &format!("{prefix}/v1/chat/completions/render"),
                post(render),
            )
            .route(&format!("{prefix}/v1/completions/render"), post(render));
        let (url, server) = crate::vllm_render_client::tests::spawn_server(app).await;
        let url = format!("{url}{prefix}");
        let selector = crate::selector::tests::selector_with_schedulable_worker().await;
        let mut worker = crate::selector::tests::schedulable_registration(1);
        worker.taints.insert("gpu=test".into());
        CatalogReconciler::new(selector.service.core().clone())
            .apply(&[worker])
            .await
            .unwrap();

        for sglang in [false, true] {
            let renderer = if sglang {
                RenderClient::Sglang(
                    SglangRendererClient::new(&url, Duration::from_secs(5), 1024 * 1024).unwrap(),
                )
            } else {
                RenderClient::Vllm(
                    VllmRenderClient::new(&url, Duration::from_secs(5), 1024 * 1024).unwrap(),
                )
            };
            for input in [
                json!({"messages":[{"role":"user","content":"hello"},{"role":"tool","tool_call_id":"call-1","content":"done"}]}),
                json!({"prompt":"hello world"}),
                json!({"prompt":[1,42,99]}),
            ] {
                for (enabled, pinned) in [(false, false), (true, false), (true, true)] {
                    let mut body = input.clone();
                    body["model"] = "test-model".into();
                    body["cache_salt"] = "legacy-salt".into();
                    body["nvext"] = json!({"cache_salt":"body-salt","agent_hints":{"priority":-3,"strict_priority":9,"osl":128},
                        "routing_constraints":{"required_taints":["gpu=test"]}});
                    let mut headers = vec![
                        (":path", "/gateway/v1/chat/completions"),
                        ("x-dynamo-dp-rank", "0"),
                        ("x-dynamo-request-priority", "7"),
                        ("x-dynamo-request-strict-priority", "malformed"),
                        ("x-tenant-id", "old"),
                        ("X-Tenant-ID", " tenant-header "),
                        ("x-tenant-id", " "),
                        ("x-claude-code-session-id", "root"),
                        ("x-claude-code-agent-id", "child"),
                        ("x-dynamo-session-final", "true"),
                        ("x-codex-future", "one"),
                        ("x-codex-future", "two"),
                    ];
                    if pinned {
                        body["nvext"]["backend_instance_id"] = 2.into();
                        headers.push(("x-dynamo-worker-instance-id", "1"));
                    }
                    let req = RequestInfo {
                        request_id: "parity".into(),
                        model: "test-model".into(),
                        headers: headers
                            .into_iter()
                            .map(|(k, v)| (k.into(), v.into()))
                            .collect(),
                        body: bytes::Bytes::from(body.to_string()),
                        candidate_subset: Vec::new(),
                    };
                    let local = crate::epp::prepare_local_request(
                        &preprocessor,
                        std::str::from_utf8(&req.body).unwrap(),
                        &req.headers,
                        enabled,
                    )
                    .await
                    .unwrap()
                    .request;
                    let remote = prepare_remote_request(
                        &renderer,
                        &selector,
                        "test-model",
                        enabled,
                        &req,
                        req.request_id.clone(),
                        None,
                    )
                    .await
                    .unwrap();
                    let routing = local.routing.unwrap();
                    assert_eq!(
                        remote.prompt.token_ids.as_deref(),
                        Some([1, 42, 99].as_slice())
                    );
                    assert_eq!(remote.prompt.cache_namespace, routing.cache_namespace);
                    assert_eq!(remote.priority_jump, routing.priority_jump);
                    assert_eq!(remote.strict_priority, routing.strict_priority);
                    assert_eq!(
                        remote.expected_output_tokens,
                        routing.expected_output_tokens
                    );
                    assert_eq!(
                        remote.session_context,
                        local
                            .agent_context
                            .as_ref()
                            .map(to_selection_session_context)
                    );
                    assert_eq!(
                        remote.routing_constraints,
                        routing.routing_constraints.unwrap_or_default()
                    );
                    assert_eq!(
                        remote.pinned_worker.map(|w| w.worker_id),
                        pinned.then_some(1)
                    );
                    let selected = selector.select_and_reserve(remote, None).await.unwrap();
                    assert_eq!(selected.worker_id, 1);
                    selector
                        .free_reservation(&selected.reservation_id)
                        .await
                        .unwrap();
                }
            }
            assert!(matches!(
                renderer
                    .render(bytes::Bytes::from_static(br#"{"prompt":[1],"n":2}"#), true)
                    .await,
                Err(RenderError::InvalidResponse { .. })
            ));
        }
        server.abort();
    }

    #[tokio::test]
    async fn unsupported_remote_inputs_fail_before_render() {
        use serde_json::json;
        let selector = crate::selector::tests::selector_with_schedulable_worker().await;
        let renderer = RenderClient::Vllm(
            VllmRenderClient::new("http://renderer.invalid", Duration::from_millis(10), 1024)
                .unwrap(),
        );
        for body in [
            json!({"model":"adapter","prompt":"hi"}),
            json!({"model":"test-model","prompt":["a","b"]}),
            json!({"model":"test-model","prompt":[[1],[2]]}),
            json!({"model":"test-model","prompt":"hi","n":2}),
            json!({"model":"test-model","prompt":"hi","prompt_embeds":"AAAA"}),
            json!({"model":"test-model","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAA"}}]}]}),
            json!({"model":"test-model","prompt":"hi","nvext":{"agent_context":{"session_id":"forged"}}}),
            json!({"model":"test-model","prompt":"hi","nvext":{"token_data":[1]}}),
            json!({"model":"test-model","prompt":"hi","nvext":{"backend_instance_id":1,"dp_rank":7}}),
            json!({"model":"test-model","prompt":"hi","nvext":{"prefill_worker_id":1}}),
        ] {
            let req = RequestInfo {
                request_id: "invalid".into(),
                model: "test-model".into(),
                body: bytes::Bytes::from(body.to_string()),
                headers: Vec::new(),
                candidate_subset: Vec::new(),
            };
            let error = prepare_remote_request(
                &renderer,
                &selector,
                "test-model",
                true,
                &req,
                "invalid".into(),
                None,
            )
            .await
            .unwrap_err();
            assert!(
                matches!(error, PickError::InvalidRequest(_)),
                "{body}: {error}"
            );
        }
    }
}
