// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed interception for stateful protocol runtimes hosted by Dynamo.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::Json;
use axum::body::to_bytes;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::response::Response as HttpResponse;
use axum::response::sse::{KeepAlive, Sse};
use dynamo_agent_protocol_host::{
    AnthropicInference, AnthropicProtocolOutput, ProtocolExtensionError,
    ProtocolExtensionErrorKind, ProtocolFuture, ProtocolOutput, ProtocolRequestContext,
    RequestCancellation, ResponsesInference, ResponsesProtocolOutput,
};
use dynamo_protocols::types::anthropic::{
    AnthropicCreateMessageRequest, AnthropicErrorResponse, AnthropicStreamEvent,
};
use dynamo_protocols::types::responses::{CreateResponse, ResponseStreamEvent};
use dynamo_runtime::engine::AsyncEngineContext;
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
use futures::StreamExt;
use tracing::Instrument;

use super::anthropic::anthropic_messages_native;
use super::disconnect::{TypedStreamTerminal, create_connection_monitor, monitor_typed_stream};
use super::metrics::{CancellationLabels, Endpoint, ErrorType, InflightGuard};
use super::openai::{ErrorResponse, responses_native};
use super::{anthropic, openai, service_v2};
use crate::protocols::anthropic::stream_converter::serialize_anthropic_frame;
use crate::protocols::common::extensions::{
    AGENT_CONTEXT_CONTEXT_KEY, AgentContext, HEADER_TENANT_ID, ROUTING_HEADER_NAMES,
    SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId,
};
use crate::protocols::common::input_trigger::{
    classify_anthropic_request, classify_response_request,
};
use crate::protocols::openai::responses::stream_converter::ResponseEventSerializer;
use crate::protocols::openai::responses::{NvCreateResponse, NvResponse, ResponseParams};
use crate::request_template::{RequestTemplate, resolve_request_model};

struct PipelineCancellation(Arc<dyn AsyncEngineContext>);

impl RequestCancellation for PipelineCancellation {
    fn is_cancelled(&self) -> bool {
        self.0.is_stopped()
    }

    fn cancelled(&self) -> ProtocolFuture<'static, ()> {
        let cancellation = self.0.clone();
        Box::pin(async move { cancellation.stopped().await })
    }
}

fn protocol_request_context<T: Send + Sync + 'static>(
    request: &Context<T>,
    headers: &HeaderMap,
) -> ProtocolRequestContext {
    let idempotency_key = ["idempotency-key", "x-idempotency-key"]
        .into_iter()
        .find_map(|name| headers.get(name))
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(Arc::from);
    ProtocolRequestContext::new(
        Arc::<str>::from(request.id()),
        idempotency_key,
        Arc::new(PipelineCancellation(request.context())),
    )
}

fn extension_status(kind: ProtocolExtensionErrorKind) -> StatusCode {
    match kind {
        ProtocolExtensionErrorKind::BadRequest => StatusCode::BAD_REQUEST,
        ProtocolExtensionErrorKind::Unauthorized => StatusCode::UNAUTHORIZED,
        ProtocolExtensionErrorKind::Forbidden => StatusCode::FORBIDDEN,
        ProtocolExtensionErrorKind::NotFound => StatusCode::NOT_FOUND,
        ProtocolExtensionErrorKind::Conflict => StatusCode::CONFLICT,
        ProtocolExtensionErrorKind::RateLimited => StatusCode::TOO_MANY_REQUESTS,
        ProtocolExtensionErrorKind::NotImplemented => StatusCode::NOT_IMPLEMENTED,
        ProtocolExtensionErrorKind::Overloaded => super::error::overload_status_code(),
        ProtocolExtensionErrorKind::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        ProtocolExtensionErrorKind::Cancelled => {
            StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST)
        }
        ProtocolExtensionErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn extension_error_from_http(
    status: StatusCode,
    message: impl Into<String>,
) -> ProtocolExtensionError {
    let kind = if status == super::error::overload_status_code() {
        ProtocolExtensionErrorKind::Overloaded
    } else {
        match status {
            StatusCode::BAD_REQUEST => ProtocolExtensionErrorKind::BadRequest,
            StatusCode::UNAUTHORIZED => ProtocolExtensionErrorKind::Unauthorized,
            StatusCode::FORBIDDEN => ProtocolExtensionErrorKind::Forbidden,
            StatusCode::NOT_FOUND => ProtocolExtensionErrorKind::NotFound,
            StatusCode::CONFLICT => ProtocolExtensionErrorKind::Conflict,
            StatusCode::UNPROCESSABLE_ENTITY => ProtocolExtensionErrorKind::BadRequest,
            StatusCode::TOO_MANY_REQUESTS => ProtocolExtensionErrorKind::RateLimited,
            StatusCode::NOT_IMPLEMENTED => ProtocolExtensionErrorKind::NotImplemented,
            StatusCode::SERVICE_UNAVAILABLE => ProtocolExtensionErrorKind::Unavailable,
            status if status.as_u16() == 499 => ProtocolExtensionErrorKind::Cancelled,
            _ => ProtocolExtensionErrorKind::Internal,
        }
    };
    ProtocolExtensionError::new(kind, message)
}

#[derive(Clone)]
struct InvocationCarrier {
    metadata: BTreeMap<String, String>,
    agent_context: Option<AgentContext>,
    session_affinity: Option<SessionAffinityId>,
    trace_request_id: Option<String>,
    parent_context: Arc<dyn AsyncEngineContext>,
}

impl InvocationCarrier {
    fn from_pipeline<T: Send + Sync + 'static>(request: &Context<T>) -> Self {
        Self {
            metadata: request.metadata().clone(),
            agent_context: request
                .get::<AgentContext>(AGENT_CONTEXT_CONTEXT_KEY)
                .ok()
                .map(|context| context.as_ref().clone()),
            session_affinity: request
                .get::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
                .ok()
                .map(|affinity| affinity.as_ref().clone()),
            trace_request_id: request
                .get::<String>(crate::request_trace::X_REQUEST_ID_CONTEXT_KEY)
                .ok()
                .map(|request_id| request_id.as_ref().clone()),
            parent_context: request.context(),
        }
    }

    fn context<T: Send + Sync + 'static>(
        &self,
        request: T,
        input_trigger: crate::protocols::common::extensions::InputTrigger,
    ) -> Context<T> {
        let mut request = Context::with_id_and_metadata(
            request,
            uuid::Uuid::new_v4().to_string(),
            self.metadata.clone(),
        );
        if let Some(trace_request_id) = &self.trace_request_id {
            request.insert(
                crate::request_trace::X_REQUEST_ID_CONTEXT_KEY,
                trace_request_id.clone(),
            );
        }
        if let Some(agent_context) = &self.agent_context {
            let mut agent_context = agent_context.clone();
            agent_context.session_final = None;
            agent_context.kv_hints = None;
            agent_context.input_trigger = Some(input_trigger);
            request.insert(AGENT_CONTEXT_CONTEXT_KEY, agent_context);
        }
        if let Some(session_affinity) = &self.session_affinity {
            request.insert(SESSION_AFFINITY_CONTEXT_KEY, session_affinity.clone());
        }
        let child_context = request.context();
        self.parent_context.link_child(child_context.clone());
        if self.parent_context.is_killed() {
            child_context.kill();
        } else if self.parent_context.is_stopped() {
            child_context.stop();
        }
        request
    }
}

struct ResponsesInferenceInner {
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    carrier: InvocationCarrier,
    nvext: Option<crate::protocols::common::extensions::NvExt>,
    chat_template_args: Option<std::collections::HashMap<String, serde_json::Value>>,
}

fn responses_inference(
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    request: &Context<NvCreateResponse>,
) -> ResponsesInference {
    let inner = Arc::new(ResponsesInferenceInner {
        state,
        template,
        carrier: InvocationCarrier::from_pipeline(request),
        nvext: request.nvext.clone(),
        chat_template_args: request.chat_template_args.clone(),
    });
    ResponsesInference::new(move |request| {
        let inner = inner.clone();
        Box::pin(async move { inner.infer(request).await })
    })
}

impl ResponsesInferenceInner {
    async fn infer(
        &self,
        request: CreateResponse,
    ) -> Result<ResponsesProtocolOutput, ProtocolExtensionError> {
        let streaming = request.stream.unwrap_or(false);
        let model = resolve_request_model(
            request.model.as_deref().unwrap_or(""),
            self.template.as_ref(),
        )
        .to_owned();
        let wrapper = NvCreateResponse {
            inner: request,
            nvext: self.nvext.clone(),
            chat_template_args: self.chat_template_args.clone(),
        };
        let trigger = classify_response_request(&wrapper);
        let pipeline_request = self.carrier.context(wrapper, trigger);
        let labels = CancellationLabels {
            model: self.state.manager().metric_model_for(&model).to_owned(),
            endpoint: Endpoint::AgentInference.to_string(),
            request_type: if streaming {
                "host_stream"
            } else {
                "host_unary"
            }
            .to_owned(),
        };
        let (mut connection_monitor, stream_handle) = create_connection_monitor(
            pipeline_request.context(),
            Some(self.state.metrics_clone()),
            labels,
        )
        .await;
        let output = responses_native(
            self.state.clone(),
            self.template.clone(),
            pipeline_request,
            stream_handle,
        )
        .await;
        connection_monitor.disarm();
        output.map_err(|(status, error)| extension_error_from_http(status, error.message()))
    }
}

struct AnthropicInferenceInner {
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    carrier: InvocationCarrier,
    routing_headers: HeaderMap,
}

fn anthropic_inference(
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    headers: &HeaderMap,
    request: &Context<AnthropicCreateMessageRequest>,
) -> AnthropicInference {
    let inner = Arc::new(AnthropicInferenceInner {
        state,
        template,
        carrier: InvocationCarrier::from_pipeline(request),
        routing_headers: filtered_routing_headers(headers),
    });
    AnthropicInference::new(move |request| {
        let inner = inner.clone();
        Box::pin(async move { inner.infer(request).await })
    })
}

impl AnthropicInferenceInner {
    async fn infer(
        &self,
        request: AnthropicCreateMessageRequest,
    ) -> Result<AnthropicProtocolOutput, ProtocolExtensionError> {
        let streaming = request.stream;
        let model = resolve_request_model(&request.model, self.template.as_ref()).to_owned();
        let trigger = classify_anthropic_request(&request);
        let pipeline_request = self.carrier.context(request, trigger);
        let request_id = pipeline_request.id().to_owned();
        let metric_model = self.state.manager().metric_model_for(&model).to_owned();
        let inflight_guard = self.state.metrics_clone().create_inflight_guard(
            &metric_model,
            Endpoint::AgentInference,
            streaming,
            &request_id,
        );
        let labels = CancellationLabels {
            model: metric_model,
            endpoint: Endpoint::AgentInference.to_string(),
            request_type: if streaming {
                "host_stream"
            } else {
                "host_unary"
            }
            .to_owned(),
        };
        let (mut connection_monitor, stream_handle) = create_connection_monitor(
            pipeline_request.context(),
            Some(self.state.metrics_clone()),
            labels,
        )
        .await;
        let output = anthropic_messages_native(
            self.state.clone(),
            self.template.clone(),
            pipeline_request,
            self.routing_headers.clone(),
            stream_handle,
            inflight_guard,
        )
        .await;
        connection_monitor.disarm();
        match output {
            Ok(output) => Ok(output),
            Err(response) => Err(anthropic_inference_error(response).await),
        }
    }
}

async fn anthropic_inference_error(response: HttpResponse) -> ProtocolExtensionError {
    let status = response.status();
    let message = match to_bytes(response.into_body(), openai::get_body_limit()).await {
        Ok(body) => serde_json::from_slice::<AnthropicErrorResponse>(&body)
            .map(|error| error.error.message)
            .unwrap_or_else(|_| "Dynamo Anthropic inference failed".to_owned()),
        Err(_) => "Dynamo Anthropic inference failed".to_owned(),
    };
    extension_error_from_http(status, message)
}

fn filtered_routing_headers(headers: &HeaderMap) -> HeaderMap {
    let mut filtered = HeaderMap::new();
    for name in ROUTING_HEADER_NAMES {
        if let Some(value) = headers.get(name) {
            filtered.insert(name, value.clone());
        }
    }
    if let Some(value) = headers.get(HEADER_TENANT_ID) {
        filtered.insert(HEADER_TENANT_ID, value.clone());
    }
    filtered
}

pub(super) async fn try_responses(
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    headers: &HeaderMap,
    request: &Context<NvCreateResponse>,
) -> Result<Option<HttpResponse>, ErrorResponse> {
    let Some(interceptor) = state.protocol_interceptor() else {
        return Ok(None);
    };
    let context = protocol_request_context(request, headers);
    if !interceptor.intercept_responses(&request.inner, &context) {
        return Ok(None);
    }
    let interceptor = interceptor.clone();

    let streaming = request.inner.stream.unwrap_or(false);
    let params = ResponseParams::from_create_response(&request.inner);
    let model = resolve_request_model(
        request.inner.model.as_deref().unwrap_or(""),
        template.as_ref(),
    );
    let canonical_model = state.manager().resolve_canonical_name(model);
    let metric_model = state
        .manager()
        .metric_model_for(&canonical_model)
        .to_owned();
    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        &metric_model,
        Endpoint::Responses,
        streaming,
        request.id(),
    );
    let inference = responses_inference(state.clone(), template, request);
    let labels = CancellationLabels {
        model: metric_model,
        endpoint: Endpoint::Responses.to_string(),
        request_type: if streaming { "stream" } else { "unary" }.to_owned(),
    };
    let (mut connection_handle, stream_handle) =
        create_connection_monitor(request.context(), Some(state.metrics_clone()), labels).await;
    let runtime_request = request.inner.clone();
    let task = tokio::spawn(
        async move {
            interceptor
                .responses(runtime_request, context, inference)
                .await
        }
        .in_current_span(),
    );
    let output = match task.await {
        Ok(output) => output,
        Err(error) => {
            connection_handle.disarm();
            inflight_guard.mark_error(ErrorType::Internal);
            tracing::error!(%error, "Responses protocol extension task failed");
            return Err(openai::ErrorMessage::internal_server_error(
                "Protocol extension task failed",
            ));
        }
    };
    connection_handle.disarm();
    let output = match output {
        Ok(output) => output,
        Err(error) => {
            inflight_guard.mark_error(extension_metric_error(error.kind()));
            return Err(openai_extension_error(error));
        }
    };
    let response = match (streaming, output) {
        (false, ProtocolOutput::Unary(response)) => {
            inflight_guard.mark_ok();
            Json(NvResponse {
                inner: response,
                nvext: None,
                presence_penalty: params.presence_penalty.unwrap_or(0.0),
                frequency_penalty: params.frequency_penalty.unwrap_or(0.0),
                store: params.store.unwrap_or(false),
            })
            .into_response()
        }
        (true, ProtocolOutput::Streaming(stream)) => {
            let serializer = ResponseEventSerializer::new(&params);
            let stream = monitor_typed_stream(
                stream,
                request.context(),
                inflight_guard,
                stream_handle,
                None,
                None,
                |event| match event {
                    ResponseStreamEvent::ResponseCompleted(_)
                    | ResponseStreamEvent::ResponseIncomplete(_) => {
                        Some(TypedStreamTerminal::Success)
                    }
                    ResponseStreamEvent::ResponseFailed(_)
                    | ResponseStreamEvent::ResponseError(_) => {
                        Some(TypedStreamTerminal::Error(ErrorType::Internal))
                    }
                    _ => None,
                },
            )
            .map(move |event| serializer.serialize(&event).map_err(axum::Error::new));
            let mut response = Sse::new(stream);
            if let Some(keep_alive) = state.sse_keep_alive_for_response(true) {
                response = response.keep_alive(KeepAlive::default().interval(keep_alive));
            }
            response.into_response()
        }
        _ => {
            inflight_guard.mark_error(ErrorType::Internal);
            return Err(openai::ErrorMessage::internal_server_error(
                "Protocol extension returned the wrong Responses output mode",
            ));
        }
    };
    Ok(Some(response))
}

pub(super) async fn try_anthropic(
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    headers: &HeaderMap,
    request: &Context<AnthropicCreateMessageRequest>,
    inflight_guard: &mut Option<InflightGuard>,
) -> Result<Option<HttpResponse>, HttpResponse> {
    let Some(interceptor) = state.protocol_interceptor() else {
        return Ok(None);
    };
    let context = protocol_request_context(request, headers);
    if !interceptor.intercept_anthropic(request, &context) {
        return Ok(None);
    }
    let interceptor = interceptor.clone();

    let Some(mut inflight_guard) = inflight_guard.take() else {
        return Err(anthropic::anthropic_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            "Anthropic interceptor lost request accounting state",
        ));
    };
    let streaming = request.stream;
    let inference = anthropic_inference(state.clone(), template, headers, request);
    let labels = CancellationLabels {
        model: state.manager().metric_model_for(&request.model).to_owned(),
        endpoint: Endpoint::AnthropicMessages.to_string(),
        request_type: if streaming { "stream" } else { "unary" }.to_owned(),
    };
    let (mut connection_handle, stream_handle) =
        create_connection_monitor(request.context(), Some(state.metrics_clone()), labels).await;
    let runtime_request = request.content().clone();
    let task = tokio::spawn(
        async move {
            interceptor
                .anthropic(runtime_request, context, inference)
                .await
        }
        .in_current_span(),
    );
    let output = match task.await {
        Ok(output) => output,
        Err(error) => {
            connection_handle.disarm();
            inflight_guard.mark_error(ErrorType::Internal);
            tracing::error!(%error, "Anthropic protocol extension task failed");
            return Err(anthropic::anthropic_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "Protocol extension task failed",
            ));
        }
    };
    connection_handle.disarm();
    let output = match output {
        Ok(output) => output,
        Err(error) => {
            inflight_guard.mark_error(extension_metric_error(error.kind()));
            return Err(anthropic_extension_error(error));
        }
    };
    let response = match (streaming, output) {
        (false, ProtocolOutput::Unary(response)) => {
            inflight_guard.mark_ok();
            Json(response).into_response()
        }
        (true, ProtocolOutput::Streaming(stream)) => {
            let stream = monitor_typed_stream(
                stream,
                request.context(),
                inflight_guard,
                stream_handle,
                None,
                None,
                |frame| match frame.event() {
                    AnthropicStreamEvent::MessageStop {} => Some(TypedStreamTerminal::Success),
                    AnthropicStreamEvent::Error { .. } => {
                        Some(TypedStreamTerminal::Error(ErrorType::Internal))
                    }
                    _ => None,
                },
            )
            .map(|frame| serialize_anthropic_frame(&frame).map_err(axum::Error::new));
            let mut response = Sse::new(stream);
            if let Some(keep_alive) = state.sse_keep_alive_for_response(true) {
                response = response.keep_alive(KeepAlive::default().interval(keep_alive));
            }
            response.into_response()
        }
        _ => {
            inflight_guard.mark_error(ErrorType::Internal);
            return Err(anthropic::anthropic_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "Protocol extension returned the wrong Anthropic output mode",
            ));
        }
    };
    Ok(Some(response))
}

fn extension_metric_error(kind: ProtocolExtensionErrorKind) -> ErrorType {
    match kind {
        ProtocolExtensionErrorKind::BadRequest
        | ProtocolExtensionErrorKind::Unauthorized
        | ProtocolExtensionErrorKind::Forbidden
        | ProtocolExtensionErrorKind::Conflict
        | ProtocolExtensionErrorKind::NotImplemented => ErrorType::Validation,
        ProtocolExtensionErrorKind::NotFound => ErrorType::NotFound,
        ProtocolExtensionErrorKind::RateLimited | ProtocolExtensionErrorKind::Overloaded => {
            ErrorType::Overload
        }
        ProtocolExtensionErrorKind::Unavailable => ErrorType::Unavailable,
        ProtocolExtensionErrorKind::Cancelled => ErrorType::Cancelled,
        ProtocolExtensionErrorKind::Internal => ErrorType::Internal,
    }
}

fn openai_extension_error(error: ProtocolExtensionError) -> ErrorResponse {
    openai::ErrorMessage::from_http_error(super::error::HttpError {
        code: extension_status(error.kind()).as_u16(),
        message: error.message().to_owned(),
    })
}

fn anthropic_extension_error(error: ProtocolExtensionError) -> HttpResponse {
    anthropic::anthropic_error(extension_status(error.kind()), "api_error", error.message())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::extensions::{InputTrigger, KvHints};

    #[test]
    fn routing_filter_drops_credentials_and_identity_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(
            crate::protocols::common::extensions::HEADER_REQUEST_PRIORITY,
            "7".parse().unwrap(),
        );
        headers.insert(HEADER_TENANT_ID, "cache-salt".parse().unwrap());
        headers.insert("authorization", "Bearer secret".parse().unwrap());
        headers.insert("x-dynamo-principal-id", "caller".parse().unwrap());

        let filtered = filtered_routing_headers(&headers);

        assert_eq!(
            filtered[crate::protocols::common::extensions::HEADER_REQUEST_PRIORITY],
            "7"
        );
        assert_eq!(filtered[HEADER_TENANT_ID], "cache-salt");
        assert!(!filtered.contains_key("authorization"));
        assert!(!filtered.contains_key("x-dynamo-principal-id"));
    }

    #[test]
    fn invocation_carrier_preserves_routing_context_but_clears_terminal_hints() {
        let mut metadata = BTreeMap::new();
        metadata.insert("x-routing-key".to_owned(), "worker-a".to_owned());
        let mut outer = Context::with_id_and_metadata((), "request-7".to_owned(), metadata.clone());
        outer.insert(
            AGENT_CONTEXT_CONTEXT_KEY,
            AgentContext {
                session_id: "session-1".to_owned(),
                parent_session_id: Some("parent-1".to_owned()),
                session_final: Some(true),
                compaction: None,
                kv_hints: Some(KvHints {
                    evict_session: true,
                }),
                input_trigger: Some(InputTrigger::UserMessage),
            },
        );
        outer.insert(
            SESSION_AFFINITY_CONTEXT_KEY,
            SessionAffinityId::new("session-1"),
        );
        outer.insert(
            crate::request_trace::X_REQUEST_ID_CONTEXT_KEY,
            "trace-9".to_owned(),
        );

        let carrier = InvocationCarrier::from_pipeline(&outer);
        let inner = carrier.context((), InputTrigger::ToolResult);
        let second_inner = carrier.context((), InputTrigger::ToolResult);

        assert_ne!(inner.id(), "request-7");
        assert_ne!(inner.id(), second_inner.id());
        assert_eq!(inner.metadata(), &metadata);
        let agent = inner
            .get::<AgentContext>(AGENT_CONTEXT_CONTEXT_KEY)
            .unwrap();
        assert_eq!(agent.session_id, "session-1");
        assert_eq!(agent.parent_session_id.as_deref(), Some("parent-1"));
        assert_eq!(agent.session_final, None);
        assert_eq!(agent.kv_hints, None);
        assert_eq!(agent.input_trigger, Some(InputTrigger::ToolResult));
        assert_eq!(
            inner
                .get::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
                .unwrap()
                .as_str(),
            "session-1"
        );
        assert_eq!(
            inner
                .get::<String>(crate::request_trace::X_REQUEST_ID_CONTEXT_KEY)
                .unwrap()
                .as_str(),
            "trace-9"
        );

        outer.context().stop();
        assert!(inner.context().is_stopped());
        let late_child = InvocationCarrier::from_pipeline(&outer).context((), InputTrigger::Other);
        assert!(late_child.context().is_stopped());
    }

    #[test]
    fn request_context_extracts_only_nonempty_idempotency_keys() {
        let request = Context::with_id_and_metadata((), "request-8".to_owned(), BTreeMap::new());
        let mut headers = HeaderMap::new();
        headers.insert("idempotency-key", "  turn-3  ".parse().unwrap());

        let context = protocol_request_context(&request, &headers);
        assert_eq!(context.request_id(), "request-8");
        assert_eq!(context.idempotency_key(), Some("turn-3"));

        headers.insert("idempotency-key", "   ".parse().unwrap());
        let context = protocol_request_context(&request, &headers);
        assert_eq!(context.idempotency_key(), None);
    }

    #[test]
    fn extension_errors_preserve_retry_and_capability_statuses() {
        assert_eq!(
            extension_status(ProtocolExtensionErrorKind::Overloaded),
            super::super::error::overload_status_code()
        );
        assert_eq!(
            extension_status(ProtocolExtensionErrorKind::RateLimited),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            extension_error_from_http(StatusCode::NOT_IMPLEMENTED, "unsupported").kind(),
            ProtocolExtensionErrorKind::NotImplemented
        );
    }
}
