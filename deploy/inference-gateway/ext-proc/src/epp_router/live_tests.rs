// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in smoke test against an independently launched renderer and GPU backend.
//! Set EPP_LIVE_RENDER_URL, EPP_LIVE_BACKEND_URL, EPP_LIVE_MODEL_PATH,
//! EPP_LIVE_MODEL_NAME, EPP_LIVE_PROTOCOL (vllm-render or sglang-renderer), and
//! EPP_LIVE_REPORT (an output JSON path outside the checkout). Run with
//! `cargo test -p dynamo-ext-proc live_renderer_gateway_roundtrip -- --ignored --nocapture`.
//! A gRPC client supplies Envoy's messages and applies the returned mutations.
//! Worker discovery is a fixed fixture; rendering, selection, gateway handling,
//! inference and reservation cleanup use their production implementations.

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::Context;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::{BlockHashOptions, compute_block_hash_for_seq};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::common::extensions::to_worker_selection_session_context;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};

use super::*;
use crate::ExtProcServer;
use crate::pod_discovery::RawWorker;
use crate::proto::envoy::config::core::v3::{HeaderMap, HeaderValue};
use crate::proto::envoy::service::ext_proc::v3::{
    self as wire, external_processor_client::ExternalProcessorClient, processing_request::Request,
    processing_response::Response,
};

struct ObservedEpp {
    inner: EppRouter,
    bookings: Mutex<Vec<String>>,
}

#[tonic::async_trait]
impl EndpointPicker for ObservedEpp {
    async fn pick(
        &self,
        req: &RequestInfo,
        endpoints: &[Endpoint],
    ) -> Result<PickResult, PickError> {
        let result = self.inner.pick(req, endpoints).await?;
        self.bookings.lock().unwrap().push(
            result
                .reservation_id
                .clone()
                .expect("request must be booked"),
        );
        Ok(result)
    }

    async fn on_prefill_complete(&self, booking: &str) {
        self.inner.on_prefill_complete(booking).await;
    }

    async fn on_request_complete(&self, booking: &str) {
        self.inner.on_request_complete(booking).await;
    }
}

fn headers(values: &[(String, String)]) -> wire::HttpHeaders {
    wire::HttpHeaders {
        headers: Some(HeaderMap {
            headers: values
                .iter()
                .map(|(key, value)| HeaderValue {
                    key: key.clone(),
                    raw_value: value.as_bytes().to_vec(),
                    ..Default::default()
                })
                .collect(),
        }),
        end_of_stream: false,
    }
}

#[tokio::test]
#[ignore = "requires a real renderer, GPU backend and matching local model files; see module docs"]
async fn live_renderer_gateway_roundtrip() -> anyhow::Result<()> {
    let required = |key| std::env::var(key).with_context(|| format!("set {key}"));
    let render_url = required("EPP_LIVE_RENDER_URL")?;
    let backend_url = required("EPP_LIVE_BACKEND_URL")?;
    let model_path = required("EPP_LIVE_MODEL_PATH")?;
    let model_name = required("EPP_LIVE_MODEL_NAME")?;
    let protocol: RendererProtocol = required("EPP_LIVE_PROTOCOL")?.parse()?;
    let report_path = required("EPP_LIVE_REPORT")?;
    let backend = reqwest::Url::parse(&backend_url)?;
    let endpoint = format!(
        "{}:{}",
        backend.host_str().unwrap(),
        backend.port_or_known_default().unwrap()
    );
    let mut card = ModelDeploymentCard::load_from_disk(&model_path, None)?;
    card.set_name(&model_name);
    let preprocessor = OpenAIPreprocessor::new(card)?;
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;
    let mut results = Vec::new();

    for enabled in [true, false] {
        let mut cfg = crate::selector::tests::test_config();
        cfg.model_name = model_name.clone();
        cfg.tokenizer_service_url = render_url.clone();
        cfg.renderer_protocol = protocol;
        let selector = Arc::new(
            Selector::new_with_kv_router_config(
                &cfg,
                KvRouterConfig::default(),
                WorkerSelectionPolicyRegistry::default(),
            )
            .await?,
        );
        // Reserve an unused local port for the catalog's event address. KV event
        // ingestion/recovery is outside this request-preparation smoke test.
        let event_port = TcpListener::bind("127.0.0.1:0").await?;
        let (reflector, _changes) = PodDiscovery::for_test(vec![RawWorker {
            worker_id: 1,
            pod_name: "live-backend".into(),
            pod_ip: backend.host_str().unwrap().into(),
            http_endpoint: backend_url.clone(),
            kv_events_endpoints: HashMap::from([(
                0,
                format!("tcp://{}", event_port.local_addr()?),
            )]),
            replay_endpoint: None,
        }]);
        let adapter = TopologyAdapter::spawn(
            reflector.clone(),
            selector.clone(),
            RegistrationDefaults::from_config(&cfg),
        );
        tokio::time::timeout(Duration::from_secs(10), async {
            while !selector.any_ready().await {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await?;
        let renderer =
            match protocol {
                RendererProtocol::VllmRender => RenderClient::Vllm(VllmRenderClient::new(
                    &render_url,
                    Duration::from_secs(30),
                    1024 * 1024,
                )?),
                RendererProtocol::SglangRenderer => RenderClient::Sglang(
                    SglangRendererClient::new(&render_url, Duration::from_secs(30), 1024 * 1024)?,
                ),
            };
        let picker = Arc::new(ObservedEpp {
            inner: EppRouter {
                renderer,
                reflector: Arc::new(reflector),
                selector: selector.clone(),
                _adapter: adapter,
                reflector_ready: Arc::new(AtomicBool::new(true)),
                model_name: model_name.clone(),
                nvext_enabled: enabled,
                inflight: Arc::new(Semaphore::new(4)),
            },
            bookings: Mutex::new(Vec::new()),
        });
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let grpc_url = format!("http://{}", listener.local_addr()?);
        let stop = tokio_util::sync::CancellationToken::new();
        let server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(ExtProcServer::new(picker.clone()).into_service())
                .serve_with_incoming_shutdown(
                    TcpListenerStream::new(listener),
                    stop.clone().cancelled_owned(),
                ),
        );
        let mut client = ExternalProcessorClient::connect(grpc_url).await?;

        for (kind, input) in [
            (
                "chat",
                json!({"messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"Count from one to five. Café."}]}),
            ),
            ("text", json!({"prompt":"The capital of France is"})),
            ("tokens", json!({"prompt":[1,42,99]})),
        ] {
            for streaming in [false, true] {
                let path = if kind == "chat" {
                    "/v1/chat/completions"
                } else {
                    "/v1/completions"
                };
                let id = format!("live-{kind}-{enabled}-{streaming}");
                let mut body = input.clone();
                body["model"] = model_name.clone().into();
                body["max_tokens"] = 8.into();
                body["temperature"] = 0.into();
                body["stream"] = streaming.into();
                if streaming {
                    body["stream_options"] = json!({"include_usage":true});
                }
                body["cache_salt"] = "legacy-salt".into();
                body["nvext"] = json!({"cache_salt":"body-salt","agent_hints":{"priority":-3,"strict_priority":2,"osl":8}});
                let req = RequestInfo {
                    request_id: id.clone(),
                    headers: vec![
                        (":path".into(), path.into()),
                        ("content-type".into(), "application/json".into()),
                        ("x-request-id".into(), id.clone()),
                        ("x-tenant-id".into(), "old".into()),
                        ("x-tenant-id".into(), " live-tenant ".into()),
                        ("x-dynamo-worker-instance-id".into(), "1".into()),
                        ("x-dynamo-request-priority".into(), "7".into()),
                        ("x-claude-code-session-id".into(), id.clone()),
                        ("x-dynamo-session-final".into(), "true".into()),
                    ],
                    body: bytes::Bytes::from(serde_json::to_vec(&body)?),
                    model: model_name.clone(),
                    candidate_subset: Vec::new(),
                };
                let local = crate::epp::prepare_local_request(
                    &preprocessor,
                    std::str::from_utf8(&req.body)?,
                    &req.headers,
                    enabled,
                )
                .await?
                .request;
                let remote = picker.inner.prepare_request(&req, id.clone(), None).await?;
                assert_eq!(
                    remote.prompt.token_ids.as_deref(),
                    Some(local.token_ids.as_slice()),
                    "{id}: renderer tokens must match frontend"
                );
                let routing = local.routing.as_ref().unwrap();
                assert_eq!(remote.prompt.cache_namespace, routing.cache_namespace);
                assert_eq!(
                    remote.prompt.cache_namespace.as_deref(),
                    Some("live-tenant")
                );
                assert_eq!(remote.priority_jump, routing.priority_jump);
                assert_eq!(remote.strict_priority, routing.strict_priority);
                assert_eq!(
                    remote.expected_output_tokens,
                    routing.expected_output_tokens
                );
                assert_eq!(
                    remote.session_context.map(Into::into),
                    local
                        .agent_context
                        .as_ref()
                        .map(to_worker_selection_session_context)
                );
                assert_eq!(
                    remote.pinned_worker.map(|worker| worker.worker_id),
                    enabled.then_some(1)
                );
                let hashes = compute_block_hash_for_seq(
                    &local.token_ids,
                    cfg.block_size,
                    BlockHashOptions {
                        cache_namespace: routing.cache_namespace.as_deref(),
                        ..Default::default()
                    },
                );

                let (tx, rx) = tokio::sync::mpsc::channel(8);
                let mut responses = client.process(ReceiverStream::new(rx)).await?.into_inner();
                tx.send(wire::ProcessingRequest {
                    request: Some(Request::RequestHeaders(headers(&req.headers))),
                    ..Default::default()
                })
                .await?;
                tx.send(wire::ProcessingRequest {
                    request: Some(Request::RequestBody(wire::HttpBody {
                        body: req.body.to_vec(),
                        end_of_stream: true,
                    })),
                    ..Default::default()
                })
                .await?;
                let mut destination = None;
                let mut forwarded = None;
                for _ in 0..2 {
                    let message =
                        tokio::time::timeout(Duration::from_secs(40), responses.message())
                            .await??
                            .context("missing EPP response")?;
                    let common = match message.response {
                        Some(Response::RequestHeaders(value)) => value.response,
                        Some(Response::RequestBody(value)) => value.response,
                        other => anyhow::bail!("{id}: unexpected EPP response: {other:?}"),
                    }
                    .context("missing common response")?;
                    if let Some(mutation) = common.header_mutation {
                        for entry in mutation.set_headers {
                            if let Some(header) = entry.header
                                && header.key == "x-gateway-destination-endpoint"
                            {
                                destination = Some(String::from_utf8(header.raw_value)?);
                            }
                        }
                    }
                    if let Some(mutation) = common.body_mutation {
                        forwarded = Some(match mutation.mutation {
                            Some(wire::body_mutation::Mutation::StreamedResponse(value)) => {
                                value.body
                            }
                            Some(wire::body_mutation::Mutation::Body(value)) => value,
                            other => anyhow::bail!("unexpected body mutation: {other:?}"),
                        });
                    }
                }
                assert_eq!(destination.as_deref(), Some(endpoint.as_str()));
                let forwarded = forwarded.context("missing forwarded request")?;
                let forwarded_json: Value = serde_json::from_slice(&forwarded)?;
                let salt = match protocol {
                    RendererProtocol::VllmRender => "dynamo-cache-salt:live-tenant",
                    RendererProtocol::SglangRenderer => "live-tenant",
                };
                let mut expected_body = body;
                expected_body["cache_salt"] = salt.into();
                assert_eq!(
                    forwarded_json, expected_body,
                    "only the cache salt should be rewritten"
                );
                let booking = picker.bookings.lock().unwrap().last().unwrap().clone();
                let response = http
                    .post(format!("http://{}{path}", destination.unwrap()))
                    .header("content-type", "application/json")
                    .body(forwarded)
                    .send()
                    .await?;
                let status = response.status();
                let response_body = response.bytes().await?;
                assert!(
                    status.is_success(),
                    "{id}: backend {status}: {}",
                    String::from_utf8_lossy(&response_body)
                );
                let usage = if streaming {
                    let text = std::str::from_utf8(&response_body)?;
                    assert!(
                        text.contains("data: [DONE]"),
                        "{id}: stream did not complete"
                    );
                    text.lines()
                        .filter_map(|line| line.strip_prefix("data: "))
                        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
                        .filter_map(|value| {
                            value
                                .get("usage")
                                .filter(|usage| usage.is_object())
                                .cloned()
                        })
                        .next_back()
                        .context("stream has no usage")?
                } else {
                    serde_json::from_slice::<Value>(&response_body)?["usage"].clone()
                };
                assert_eq!(
                    usage["prompt_tokens"].as_u64(),
                    Some(local.token_ids.len() as u64),
                    "{id}: backend re-tokenized a different prompt"
                );
                assert!(usage["completion_tokens"].as_u64().unwrap_or_default() > 0);
                let response_headers = vec![
                    (":status".into(), status.as_u16().to_string()),
                    (
                        "content-type".into(),
                        if streaming {
                            "text/event-stream"
                        } else {
                            "application/json"
                        }
                        .into(),
                    ),
                ];
                tx.send(wire::ProcessingRequest {
                    request: Some(Request::ResponseHeaders(headers(&response_headers))),
                    ..Default::default()
                })
                .await?;
                tx.send(wire::ProcessingRequest {
                    request: Some(Request::ResponseBody(wire::HttpBody {
                        body: response_body.to_vec(),
                        end_of_stream: true,
                    })),
                    ..Default::default()
                })
                .await?;
                drop(tx);
                tokio::time::timeout(Duration::from_secs(10), async {
                    while responses.message().await?.is_some() {}
                    Ok::<_, tonic::Status>(())
                })
                .await??;
                assert!(
                    matches!(
                        selector.service.free_reservation(&booking).await,
                        Err(SelectionError::NotFound(_))
                    ),
                    "{id}: gateway must release the booking"
                );
                results.push(json!({"case":id,"usage":usage,"routing_tokens":local.token_ids,"block_hashes":hashes,"forwarded_cache_salt":salt,"destination":endpoint,"reservation_released":true}));
                println!("PASS {id}: usage={usage}, reservation released");
            }
        }
        stop.cancel();
        server.await??;
    }
    std::fs::write(
        report_path,
        serde_json::to_vec_pretty(
            &json!({"model":model_name,"model_path":model_path,"renderer":render_url,"backend":backend_url,"cases":results}),
        )?,
    )?;
    Ok(())
}
