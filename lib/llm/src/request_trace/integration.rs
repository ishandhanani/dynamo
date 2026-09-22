// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::Arc;

use dynamo_runtime::pipeline::Context;
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};

use crate::protocols::common::preprocessor::PreprocessedRequest;
use crate::protocols::common::timing::{
    RequestPhase, RequestTracker, WORKER_TYPE_DECODE, WORKER_TYPE_PREFILL,
};
use crate::protocols::openai::{
    chat_completions::NvCreateChatCompletionStreamResponse, completions::NvCreateCompletionResponse,
};
use crate::request_trace::{
    AgentContextTraceState, RequestReplayMetrics, SharedFinishReasonMetadata,
};

struct RequestTraceRequestEndState {
    request_tracker: Arc<RequestTracker>,
    replay_metrics: Arc<RequestReplayMetrics>,
}

pub(crate) struct RequestEndTraceState {
    agent: Option<AgentContextTraceState>,
    request: RequestTraceRequestEndState,
}

/// Holds a native HTTP trace through response completion or cancellation.
pub(crate) struct HttpRequestTrace {
    state: Option<RequestEndTraceState>,
    request_id: String,
    model: String,
    x_request_id: Option<String>,
}

impl HttpRequestTrace {
    pub(crate) fn new(
        request: &PreprocessedRequest,
        tracker: Arc<RequestTracker>,
        context: &Context<()>,
        block_size: usize,
    ) -> Option<Self> {
        let state = build_request_end_trace_state(request, &Some(tracker), context, block_size)?;
        Some(Self {
            state: Some(state),
            request_id: context.id().to_owned(),
            model: request.model.clone(),
            x_request_id: context
                .get::<String>(super::X_REQUEST_ID_CONTEXT_KEY)
                .ok()
                .map(|id| id.as_ref().clone()),
        })
    }

    pub(crate) async fn record_worker(&self, worker: u64, rank: Option<u32>, phase: RequestPhase) {
        if let Some(state) = &self.state {
            let tracker = &state.request.request_tracker;
            let _phase = tracker.set_phase(phase).await;
            let worker_type = match phase {
                RequestPhase::Prefill => WORKER_TYPE_PREFILL,
                _ => WORKER_TYPE_DECODE,
            };
            tracker.record_worker(worker, rank, worker_type);
        }
    }

    pub(crate) fn wrap_response(
        self,
        response: axum::response::Response,
    ) -> axum::response::Response {
        let (parts, body) = response.into_parts();
        let (body, done) = crate::telemetry::stream::notify_on_completion(body.into_data_stream());
        tokio::spawn(async move {
            done.await;
            drop(self);
        });
        axum::response::Response::from_parts(parts, axum::body::Body::from_stream(body))
    }
}

impl Drop for HttpRequestTrace {
    fn drop(&mut self) {
        let Some(state) = self.state.take() else {
            return;
        };
        state.request.request_tracker.record_finish();
        let (agent, mut metrics) = match state.agent {
            Some(agent) => {
                let (agent, metrics) = super::request_metrics_from_agent_state(
                    agent,
                    std::mem::take(&mut self.request_id),
                );
                (Some(agent), metrics)
            }
            None => (
                None,
                super::request_metrics(
                    std::mem::take(&mut self.request_id),
                    self.x_request_id.take(),
                    std::mem::take(&mut self.model),
                    Some(&state.request.request_tracker),
                ),
            ),
        };
        // HTTP chunks are not tokens. No engine output or timing is inferred from them.
        metrics.output_tokens = None;
        metrics.replay = Some(super::into_owned_replay_metrics(
            state.request.replay_metrics,
        ));
        super::record::emit_request_end_metrics(agent, metrics);
    }
}

fn request_trace_rejection(common_request: &PreprocessedRequest) -> Option<&'static str> {
    if common_request.prompt_embeds.is_some() {
        return Some("prompt embeddings are not supported");
    }
    if common_request.multi_modal_data.is_some() {
        return Some("multimodal inputs are not supported");
    }
    if common_request.sampling_options.n.unwrap_or(1) > 1 {
        return Some("multiple output choices are not supported");
    }
    if common_request.sampling_options.best_of.unwrap_or(1) > 1 {
        return Some("best_of greater than one is not supported");
    }
    None
}

fn shared_replay_metrics(
    token_ids: &[crate::protocols::TokenIdType],
    trace_block_size: usize,
) -> Option<Arc<RequestReplayMetrics>> {
    if trace_block_size == 0 {
        return None;
    }
    super::replay_metrics(token_ids, trace_block_size).map(Arc::new)
}

pub(crate) fn build_request_end_trace_state(
    common_request: &PreprocessedRequest,
    tracker: &Option<Arc<RequestTracker>>,
    context: &Context<()>,
    trace_block_size: usize,
) -> Option<RequestEndTraceState> {
    build_request_end_trace_state_for_policy(
        common_request,
        tracker,
        context,
        trace_block_size,
        super::policy().emit_request_end_records(),
    )
}

fn build_request_end_trace_state_for_policy(
    common_request: &PreprocessedRequest,
    tracker: &Option<Arc<RequestTracker>>,
    context: &Context<()>,
    trace_block_size: usize,
    request_trace_enabled: bool,
) -> Option<RequestEndTraceState> {
    let has_agent_context = common_request.agent_context.is_some();

    if !request_trace_enabled {
        return None;
    }

    let request_id = context.id();
    if let Some(reason) = request_trace_rejection(common_request) {
        tracing::warn!(
            %request_id,
            reason,
            "request trace skipped because the request cannot be represented as one replay request"
        );
        return None;
    }

    let request_tracker = match tracker {
        Some(tracker) => tracker.clone(),
        None => {
            tracing::warn!(
                %request_id,
                "request trace skipped because the request tracker is unavailable"
            );
            return None;
        }
    };

    let replay_metrics = match shared_replay_metrics(&common_request.token_ids, trace_block_size) {
        Some(metrics) => metrics,
        None => {
            tracing::warn!(
                %request_id,
                "request trace skipped because the KV cache block size is unavailable"
            );
            return None;
        }
    };

    let agent = has_agent_context
        .then(|| super::build_agent_context_trace_state(common_request, tracker, context))
        .flatten();

    let request = RequestTraceRequestEndState {
        request_tracker,
        replay_metrics,
    };

    Some(RequestEndTraceState { agent, request })
}

pub(crate) fn finish_reason_metadata_handle(
    trace_state: &Option<RequestEndTraceState>,
) -> Option<SharedFinishReasonMetadata> {
    trace_state
        .as_ref()
        .and_then(|state| state.agent.as_ref())
        .map(|state| state.finish_reason_metadata.clone())
}

fn wrap_request_end_stream<Resp>(
    stream: Pin<Box<dyn Stream<Item = Annotated<Resp>> + Send>>,
    trace_state: Option<RequestEndTraceState>,
    request_id: String,
) -> Pin<Box<dyn Stream<Item = Annotated<Resp>> + Send>>
where
    Resp: Send + 'static,
{
    let Some(trace_state) = trace_state else {
        return stream;
    };

    let (stream, done) = crate::telemetry::stream::notify_on_completion(stream);
    tokio::spawn(async move {
        done.await;
        let request_state = trace_state.request;
        if let Some(agent_state) = trace_state.agent {
            let (agent_context, mut metrics) =
                super::request_metrics_from_agent_state(agent_state, request_id.clone());
            metrics.replay = Some(super::into_owned_replay_metrics(
                request_state.replay_metrics,
            ));
            super::record::emit_request_end_metrics(Some(agent_context), metrics);
        } else {
            super::record::emit_request_end(
                request_id.clone(),
                &request_state.request_tracker,
                super::into_owned_replay_metrics(request_state.replay_metrics),
            );
        }
    });
    stream
}

pub(crate) fn wrap_chat_request_end_stream(
    stream: Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>>,
    trace_state: Option<RequestEndTraceState>,
    request_id: String,
) -> Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>> {
    let Some(finish_reason_metadata) = finish_reason_metadata_handle(&trace_state) else {
        return wrap_request_end_stream(stream, trace_state, request_id);
    };

    let stream = stream.map(move |response| {
        super::record_chat_finish_reason_metadata(&finish_reason_metadata, &response);
        response
    });
    wrap_request_end_stream(Box::pin(stream), trace_state, request_id)
}

pub(crate) fn wrap_completion_request_end_stream(
    stream: Pin<Box<dyn Stream<Item = Annotated<NvCreateCompletionResponse>> + Send>>,
    trace_state: Option<RequestEndTraceState>,
    request_id: String,
) -> Pin<Box<dyn Stream<Item = Annotated<NvCreateCompletionResponse>> + Send>> {
    let Some(finish_reason_metadata) = finish_reason_metadata_handle(&trace_state) else {
        return wrap_request_end_stream(stream, trace_state, request_id);
    };

    let stream = stream.map(move |response| {
        super::record_completion_finish_reason_metadata(&finish_reason_metadata, &response);
        response
    });
    wrap_request_end_stream(Box::pin(stream), trace_state, request_id)
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    use std::task::{Context as TaskContext, Poll};
    use std::time::{Duration, Instant};

    use super::*;
    use crate::protocols::common::extensions::{AgentCompaction, AgentContext, InputTrigger};
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use crate::request_trace::BUS;
    use crate::request_trace::RequestTraceEventSource;

    struct TrackerDropStream {
        tracker: Arc<RequestTracker>,
        dropped: Arc<AtomicBool>,
    }

    impl Stream for TrackerDropStream {
        type Item = Annotated<NvCreateCompletionResponse>;

        fn poll_next(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
            Poll::Pending
        }
    }

    impl Drop for TrackerDropStream {
        fn drop(&mut self) {
            self.tracker.record_osl(9);
            self.tracker.record_finish();
            self.dropped.store(true, Ordering::Release);
        }
    }

    fn preprocessed_request(sampling_options: SamplingOptions) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test-model".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions::default())
            .sampling_options(sampling_options)
            .output_options(OutputOptions::default())
            .eos_token_ids(vec![])
            .annotations(vec![])
            .build()
            .unwrap()
    }

    #[test]
    fn rejects_unsupported_request_shapes() {
        let mut multi_choice = preprocessed_request(SamplingOptions {
            n: Some(2),
            ..Default::default()
        });
        assert_eq!(
            request_trace_rejection(&multi_choice),
            Some("multiple output choices are not supported")
        );

        multi_choice.sampling_options.n = Some(1);
        multi_choice.sampling_options.best_of = Some(2);
        assert_eq!(
            request_trace_rejection(&multi_choice),
            Some("best_of greater than one is not supported")
        );

        multi_choice.sampling_options.best_of = Some(1);
        multi_choice.prompt_embeds = Some("embedding".to_string());
        assert_eq!(
            request_trace_rejection(&multi_choice),
            Some("prompt embeddings are not supported")
        );

        multi_choice.prompt_embeds = None;
        multi_choice.multi_modal_data = Some(Default::default());
        assert_eq!(
            request_trace_rejection(&multi_choice),
            Some("multimodal inputs are not supported")
        );
    }

    #[test]
    fn replay_hashing_requires_block_size() {
        assert!(shared_replay_metrics(&[1, 2, 3], 0).is_none());

        let replay = shared_replay_metrics(&[1, 2, 3], 2).unwrap();
        assert_eq!(replay.input_sequence_hashes.len(), 2);
    }

    #[test]
    fn long_isl_hashing_reports_mode_costs_without_threshold() {
        let token_ids = (0..131_072_u32).collect::<Vec<_>>();

        let started = Instant::now();
        let request_only = shared_replay_metrics(&token_ids, 64).unwrap();
        let request_elapsed = started.elapsed();

        let started = Instant::now();
        let repeated = shared_replay_metrics(&token_ids, 64).unwrap();
        let repeated_elapsed = started.elapsed();

        eprintln!(
            "long-ISL replay hashing: request_only={request_elapsed:?}, repeated={repeated_elapsed:?}"
        );
        assert_eq!(request_only.input_sequence_hashes.len(), 2_048);
        assert_eq!(
            request_only.input_sequence_hashes,
            repeated.input_sequence_hashes
        );
    }

    #[tokio::test]
    async fn cancellation_reads_tracker_after_inner_stream_drop() {
        BUS.init(16);
        let mut receiver = BUS.subscribe();
        let tracker = Arc::new(RequestTracker::new());
        let dropped = Arc::new(AtomicBool::new(false));
        let state = RequestEndTraceState {
            agent: None,
            request: RequestTraceRequestEndState {
                request_tracker: tracker.clone(),
                replay_metrics: Arc::new(RequestReplayMetrics {
                    trace_block_size: 2,
                    input_length: 2,
                    input_sequence_hashes: vec![11],
                }),
            },
        };
        let stream = TrackerDropStream {
            tracker,
            dropped: dropped.clone(),
        };

        let wrapped =
            wrap_request_end_stream(Box::pin(stream), Some(state), "req-drop".to_string());
        drop(wrapped);

        let record = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let record = receiver.recv().await.unwrap();
                if record
                    .request
                    .as_ref()
                    .is_some_and(|request| request.request_id == "req-drop")
                {
                    break record;
                }
            }
        })
        .await
        .unwrap();
        let request = record.request.as_ref().expect("request payload");
        assert!(dropped.load(Ordering::Acquire));
        assert_eq!(request.output_tokens, Some(9));
    }

    #[tokio::test]
    async fn agent_context_emits_enriched_request_trace_row() {
        BUS.init(16);
        let mut receiver = BUS.subscribe();
        let tracker = Arc::new(RequestTracker::new());
        let dropped = Arc::new(AtomicBool::new(false));
        let mut request = preprocessed_request(SamplingOptions::default());
        request.agent_context = Some(AgentContext {
            session_id: "root".to_string(),
            parent_session_id: None,
            session_final: None,
            compaction: Some(AgentCompaction {
                trigger: Some("manual".to_string()),
                reason: Some("user_requested".to_string()),
                implementation: Some("responses_compact".to_string()),
                phase: Some("standalone_turn".to_string()),
                strategy: Some("memento".to_string()),
            }),
            input_trigger: Some(InputTrigger::ToolResult),
        });
        let mut context = Context::new(());
        context.insert(
            crate::request_trace::X_REQUEST_ID_CONTEXT_KEY,
            "llm-call-1".to_string(),
        );
        let state = build_request_end_trace_state_for_policy(
            &request,
            &Some(tracker.clone()),
            &context,
            2,
            true,
        )
        .unwrap();
        let stream = TrackerDropStream {
            tracker,
            dropped: dropped.clone(),
        };

        let wrapped =
            wrap_request_end_stream(Box::pin(stream), Some(state), "req-agent".to_string());
        drop(wrapped);

        let record = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let record = receiver.recv().await.unwrap();
                if record
                    .request
                    .as_ref()
                    .is_some_and(|request| request.request_id == "req-agent")
                {
                    break record;
                }
            }
        })
        .await
        .unwrap();
        assert!(dropped.load(Ordering::Acquire));
        assert_eq!(record.event_source, Some(RequestTraceEventSource::Dynamo));
        let agent_context = record.agent_context.as_ref().expect("agent context");
        assert_eq!(agent_context.session_id, "root");
        assert_eq!(agent_context.input_trigger, Some(InputTrigger::ToolResult));
        assert_eq!(
            agent_context
                .compaction
                .as_ref()
                .and_then(|compaction| compaction.strategy.as_deref()),
            Some("memento")
        );
        let request = record.request.as_ref().expect("request payload");
        assert_eq!(request.model.as_deref(), Some("test-model"));
        assert_eq!(request.x_request_id.as_deref(), Some("llm-call-1"));
        assert_eq!(request.output_tokens, Some(9));
        assert_eq!(
            request
                .replay
                .as_ref()
                .expect("replay metrics")
                .input_length,
            3
        );
    }

    #[test]
    fn agent_context_does_not_bypass_request_trace_eligibility() {
        let mut request = preprocessed_request(SamplingOptions {
            best_of: Some(2),
            ..Default::default()
        });
        request.agent_context = Some(AgentContext {
            session_id: "root".to_string(),
            parent_session_id: None,
            session_final: None,
            compaction: None,
            input_trigger: None,
        });
        let tracker = Some(Arc::new(RequestTracker::new()));
        let context = Context::new(());

        let state = build_request_end_trace_state_for_policy(&request, &tracker, &context, 2, true);

        assert!(state.is_none());
    }

    #[tokio::test]
    async fn native_http_trace_preserves_response_and_reports_only_observed_fields() {
        use crate::protocols::common::{
            extensions::agent_context_from_headers, timing::RequestPhase,
        };
        use axum::{
            body::{Body, to_bytes},
            http::{HeaderMap, StatusCode},
            response::Response,
        };

        BUS.init(128);
        let mut receiver = BUS.subscribe();
        for (name, status, body) in [
            ("unary", StatusCode::OK, "{\"future_meta\":1e400}"),
            (
                "sse",
                StatusCode::OK,
                "data: {\"custom\":true}\n\ndata: [DONE]\n\n",
            ),
            (
                "rejected",
                StatusCode::TOO_MANY_REQUESTS,
                "engine rejection",
            ),
        ] {
            let request_id = format!("native-http-{name}");
            let tracker = Arc::new(RequestTracker::new());
            tracker.record_isl(3, None);
            let mut request = preprocessed_request(SamplingOptions::default());
            let mut headers = HeaderMap::new();
            headers.insert(
                "x-dynamo-session-id",
                "native-agent-session".parse().unwrap(),
            );
            request.agent_context = agent_context_from_headers(&headers);
            assert!(request.agent_context.is_some());
            let context = Context::with_id_and_metadata((), request_id.clone(), Default::default());
            let trace = HttpRequestTrace {
                state: build_request_end_trace_state_for_policy(
                    &request,
                    &Some(tracker),
                    &context,
                    2,
                    true,
                ),
                request_id: request_id.clone(),
                model: request.model,
                x_request_id: None,
            };
            trace
                .record_worker(11, Some(1), RequestPhase::Prefill)
                .await;
            trace.record_worker(22, Some(2), RequestPhase::Decode).await;
            let response = Response::builder()
                .status(status)
                .header("x-engine", "one")
                .header("x-engine", "two")
                .body(Body::from(body))
                .unwrap();
            let response = trace.wrap_response(response);
            assert_eq!(response.status(), status);
            assert_eq!(response.headers().get_all("x-engine").iter().count(), 2);
            assert_eq!(to_bytes(response.into_body(), 4096).await.unwrap(), body);
            let record = next_native_record(&mut receiver, &request_id).await;
            assert_eq!(
                record.agent_context.unwrap().session_id,
                "native-agent-session"
            );
            let metrics = record.request.unwrap();
            assert_eq!(metrics.model.as_deref(), Some("test-model"));
            assert_eq!(metrics.input_tokens, Some(3));
            assert_eq!(
                metrics.replay.unwrap().input_sequence_hashes,
                super::super::replay::input_sequence_hashes(&[1, 2, 3], 2)
            );
            let worker = metrics.worker.unwrap();
            assert_eq!(
                (worker.prefill_worker_id, worker.prefill_dp_rank),
                (Some(11), Some(1))
            );
            assert_eq!(
                (worker.decode_worker_id, worker.decode_dp_rank),
                (Some(22), Some(2))
            );
            assert!(metrics.total_time_ms.is_some());
            assert!(metrics.request_received_ms.is_some());
            assert!(metrics.output_tokens.is_none());
            assert!(metrics.ttft_ms.is_none());
            assert!(metrics.avg_itl_ms.is_none());
            assert!(metrics.prefill_time_ms.is_none());
            assert!(metrics.finish_reason_metadata.is_none());
        }
    }

    #[tokio::test]
    async fn native_http_trace_finishes_on_cancellation_and_dispatch_failure() {
        BUS.init(128);
        let mut receiver = BUS.subscribe();
        for failure in ["dispatch", "cancelled", "body"] {
            let request_id = format!("native-http-failure-{failure}");
            let request = preprocessed_request(SamplingOptions::default());
            let context = Context::with_id_and_metadata((), request_id.clone(), Default::default());
            let trace = HttpRequestTrace {
                state: build_request_end_trace_state_for_policy(
                    &request,
                    &Some(Arc::new(RequestTracker::new())),
                    &context,
                    2,
                    true,
                ),
                request_id: request_id.clone(),
                model: request.model,
                x_request_id: Some("client-call".into()),
            };
            match failure {
                "dispatch" => drop(trace),
                "cancelled" => {
                    let body = axum::body::Body::from_stream(futures::stream::pending::<
                        Result<bytes::Bytes, std::io::Error>,
                    >());
                    drop(trace.wrap_response(axum::response::Response::new(body)));
                }
                _ => {
                    let body = axum::body::Body::from_stream(futures::stream::iter([Err::<
                        bytes::Bytes,
                        _,
                    >(
                        std::io::Error::other("truncated response"),
                    )]));
                    let response = trace.wrap_response(axum::response::Response::new(body));
                    assert!(
                        axum::body::to_bytes(response.into_body(), 4096)
                            .await
                            .is_err()
                    );
                }
            }
            let record = next_native_record(&mut receiver, &request_id).await;
            assert!(record.agent_context.is_none());
            let metrics = record.request.unwrap();
            assert!(metrics.total_time_ms.is_some());
            assert_eq!(metrics.x_request_id.as_deref(), Some("client-call"));
            let json = serde_json::to_value(metrics).unwrap();
            assert!(json.get("output_tokens").is_none());
            assert!(json.get("finish_reason_metadata").is_none());
        }
    }

    async fn next_native_record(
        receiver: &mut tokio::sync::broadcast::Receiver<crate::request_trace::RequestTraceRecord>,
        request_id: &str,
    ) -> crate::request_trace::RequestTraceRecord {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let record = receiver.recv().await.unwrap();
                if record
                    .request
                    .as_ref()
                    .is_some_and(|request| request.request_id == request_id)
                {
                    return record;
                }
            }
        })
        .await
        .unwrap()
    }
}
