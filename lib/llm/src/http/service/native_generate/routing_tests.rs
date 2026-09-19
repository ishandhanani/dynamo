// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashMap,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use async_trait::async_trait;
use axum::http::StatusCode;
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    component::Instance,
    distributed::DistributedConfig,
    pipeline::{
        AddressedRequest, AsyncEngineContextProvider, ManyIn, ManyOut, ResponseStream, SingleIn,
        StreamingDispatch,
    },
};

use super::*;
use crate::{
    discovery::WorkerSet,
    http::service::service_v2::HttpService,
    kv_router::{KvRouter, SelectionPolicySource, scheduling::config::KvRouterConfig},
    local_model::runtime_config::ModelRuntimeConfig,
    protocols::{
        Annotated,
        sglang::http::{ResponseFrame, lifecycle as control},
    },
};

const EPOCH: &str = "11111111111111111111111111111111";
const BODY: &[u8] = br#"{"input_ids": [[1,2], [3]], "stream":false, "routed_dp_rank":1, "sampling_params":{"n":2,"future_option":1e400}, "future_field":{"number":1e400,"float":1.234567890123456789} }"#;

const BEAM_BODY: &[u8] =
    br#"{"input_ids":[[1,2],[3]],"routed_dp_rank":1,"sampling_params":{"beam_width":4,"n":2}}"#;

#[derive(Default)]
struct Engine {
    worker_id: u64,
    received: Mutex<Option<http::Request>>,
    cleanup: AtomicBool,
    acknowledged: AtomicBool,
}

#[async_trait]
impl StreamingDispatch<http::Request, Annotated<ResponseFrame>> for Engine {
    async fn generate(
        &self,
        request: SingleIn<AddressedRequest<http::Request>>,
    ) -> anyhow::Result<ManyOut<Annotated<ResponseFrame>>> {
        let (addressed, context) = request.transfer(());
        let (request, _, instance) = addressed.into_parts();
        assert_eq!(instance.unwrap().id(), self.worker_id);
        assert_eq!(request.method, "PUT");
        assert!([BODY, BEAM_BODY].contains(&request.body.as_ref()));
        assert!(
            request
                .headers
                .iter()
                .any(|(name, value)| name == "x-override-routed-dp-rank" && value.as_ref() == b"1")
        );
        *self.received.lock().unwrap() = Some(request);
        let frames = [
            ResponseFrame::Head {
                status: 201,
                headers: vec![
                    ("x-engine-extension".into(), Bytes::from_static(b"native")),
                    (
                        "content-length".into(),
                        b"opaque, not JSON or SSE\r\n".len().to_string().into(),
                    ),
                ],
            },
            ResponseFrame::Body(Bytes::from_static(b"opaque, not JSON or SSE\r\n")),
            ResponseFrame::End,
        ];
        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter(
                frames.into_iter().map(Annotated::from_data),
            )),
            context.context(),
        ))
    }

    async fn generate_bidirectional(
        &self,
        _: Instance,
        _: String,
        _: ManyIn<http::Request>,
    ) -> anyhow::Result<ManyOut<Annotated<ResponseFrame>>> {
        unreachable!()
    }
}

#[async_trait]
impl StreamingDispatch<control::Request, Annotated<control::Response>> for Engine {
    async fn generate(
        &self,
        request: SingleIn<AddressedRequest<control::Request>>,
    ) -> anyhow::Result<ManyOut<Annotated<control::Response>>> {
        let (addressed, context) = request.transfer(());
        let (request, _, instance) = addressed.into_parts();
        assert_eq!(instance.unwrap().id(), self.worker_id);
        let response = match request {
            control::Request::Describe => control::Response::Descriptor(control::Descriptor {
                version: 1,
                incarnation: EPOCH.into(),
                header_overrides: true,
                native_disaggregation_version: 0,
                session_routing_version: 0,
                session_fencing_version: 0,
                session_open_version: 0,
            }),
            control::Request::Attempt {
                attempt_id,
                incarnation,
                operation,
            } => {
                assert_eq!(incarnation, EPOCH);
                if matches!(operation, control::Operation::Acknowledge) {
                    assert!(self.cleanup.load(Ordering::SeqCst));
                    self.acknowledged.store(true, Ordering::SeqCst);
                    control::Response::Acknowledged
                } else {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                    let received = self.received.lock().unwrap();
                    if let Some(received) = received.as_ref() {
                        assert!(
                            received
                                .headers
                                .iter()
                                .any(|(name, value)| name == control::ATTEMPT_HEADER
                                    && value.as_ref() == attempt_id.as_bytes())
                        );
                        let terminal = self.cleanup.load(Ordering::SeqCst);
                        let beam = received.body.as_ref() == BEAM_BODY;
                        control::Response::Snapshot(control::Snapshot {
                            incarnation,
                            attempt_id,
                            stage: "null".into(),
                            version: if terminal { 2 } else { 1 },
                            sealed: true,
                            cancel_requested: false,
                            terminal,
                            children: (0..if beam { 2 } else { 6 })
                                .map(|i| control::Child {
                                    child_id: format!("{i:032x}"),
                                    rid: format!("rid-{i}"),
                                    kind: if !beam && i < 2 {
                                        ChildKind::Warmup
                                    } else {
                                        ChildKind::Sample
                                    },
                                    dp_rank: Some(1),
                                    dispatched: true,
                                    prefill_complete: true,
                                    terminal,
                                })
                                .collect(),
                        })
                    } else {
                        control::Response::Rejected { status: 404 }
                    }
                }
            }
        };
        Ok(ResponseStream::new(
            Box::pin(futures::stream::once(async move {
                Annotated::from_data(response)
            })),
            context.context(),
        ))
    }

    async fn generate_bidirectional(
        &self,
        _: Instance,
        _: String,
        _: ManyIn<control::Request>,
    ) -> anyhow::Result<ManyOut<Annotated<control::Response>>> {
        unreachable!()
    }
}

#[tokio::test]
async fn public_native_generate_keeps_bytes_books_fanout_and_fences_retired_workers() {
    tokio::time::timeout(Duration::from_secs(15), async {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let endpoint = drt
            .namespace("native_frontend")
            .unwrap()
            .component("workers")
            .unwrap()
            .endpoint("generate");
        let discovery_client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let worker_id = discovery_client.wait_for_instances().await.unwrap()[0].id();
        let (admissions, admitted) = watch::channel(vec![worker_id]);
        let client = discovery_client
            .with_admitted_instances_and_cancellation(admitted.clone(), CancellationToken::new());
        client.wait_for_instances().await.unwrap();
        let config = ModelRuntimeConfig {
            data_parallel_size: 2,
            ..Default::default()
        };
        let (_configs, configs) = watch::channel(HashMap::from([(worker_id, config.clone())]));
        let kv = Arc::new(
            KvRouter::new(
                endpoint,
                client.clone(),
                configs,
                None,
                2,
                SelectionPolicySource::Registry,
                Some(KvRouterConfig {
                    skip_initial_worker_wait: true,
                    use_kv_events: false,
                    ..Default::default()
                }),
                None,
                "decode",
                None,
                false,
                None,
                None,
            )
            .await
            .unwrap(),
        );
        let host = Arc::new(
            RoutingHost::new(
                dynamo_runtime::pipeline::PushRouter::from_client(client.clone(), RouterMode::KV)
                    .await
                    .unwrap(),
                kv.clone(),
                None,
            )
            .unwrap(),
        );
        let engine = Arc::new(Engine {
            worker_id,
            ..Default::default()
        });
        let mut card = ModelDeploymentCard::with_name_only("native-model");
        card.runtime_config = config;
        card.runtime_config
            .runtime_data
            .insert(http::CAPABILITY.into(), true.into());
        card.runtime_config
            .runtime_data
            .insert(control::CAPABILITY.into(), true.into());
        let binding = Arc::new(NativeGenerateBinding {
            prefill: None,
            admitted_ids: admitted.clone(),
            cancellation: CancellationToken::new(),
            client: NativeGenerateClient::from_client_with_dispatch(
                client.clone(),
                RouterMode::Direct,
                engine.clone(),
            )
            .await
            .unwrap(),
            control: Arc::new(
                LifecycleClient::from_client_with_dispatch(
                    client,
                    RouterMode::Direct,
                    engine.clone(),
                )
                .await
                .unwrap(),
            ),
            host,
            card: card.clone(),
            tokenizer: None,
        });
        let mut workers = WorkerSet::new("native_frontend".into(), card.mdcsum().to_string(), card);
        workers.set_instance_watcher(admitted);
        workers.native_generate = Some(binding.clone());
        let service = HttpService::builder()
            .enable_engine_apis(true)
            .build()
            .unwrap();
        assert!(
            service
                .model_manager()
                .add_worker_set("native-model", "native_frontend", workers)
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let metrics = service.state().metrics_clone();
        let url = format!("http://{}/generate", listener.local_addr().unwrap());
        let stop = CancellationToken::new();
        let server = service.spawn_with_listener(stop.clone(), listener).await;
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let request = |body: &'static [u8]| {
            client
                .put(&url)
                .header("content-type", "application/json")
                .body(body)
        };
        let load = || async {
            kv.get_potential_loads(&[], None, None, None, None)
                .await
                .unwrap()
                .iter()
                .map(|load| load.active_requests)
                .sum::<usize>()
        };
        for body in [br#"{"input_ids":[]}"#.as_slice(), br#"{"input_ids":[[]]}"#] {
            let response = request(body).send().await.unwrap();
            assert_eq!(
                response.status(),
                StatusCode::BAD_REQUEST,
                "{}",
                response.text().await.unwrap()
            );
        }
        for (index, (body, rows)) in [(BODY, 6), (BEAM_BODY, 8)].into_iter().enumerate() {
            *engine.received.lock().unwrap() = None;
            engine.cleanup.store(false, Ordering::SeqCst);
            engine.acknowledged.store(false, Ordering::SeqCst);
            let response = request(body).send().await.unwrap();
            assert_eq!(
                response.status(),
                StatusCode::CREATED,
                "{}",
                response.text().await.unwrap()
            );
            assert_eq!(response.headers()["x-engine-extension"], "native");
            assert_eq!(
                response.bytes().await.unwrap().as_ref(),
                b"opaque, not JSON or SSE\r\n"
            );
            use crate::http::service::metrics::{
                Endpoint as MetricEndpoint, ErrorType, RequestType, Status,
            };
            assert_eq!(
                metrics.get_request_counter(
                    "native-model",
                    &MetricEndpoint::Generate,
                    &RequestType::Unary,
                    &Status::Success,
                    &ErrorType::None
                ),
                (index + 1) as u64
            );
            assert_eq!(
                load().await,
                rows,
                "HTTP completion cannot release native children"
            );
            engine.cleanup.store(true, Ordering::SeqCst);
            while !engine.acknowledged.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            assert_eq!(load().await, 0);
        }
        admissions.send_replace(vec![]);
        assert_eq!(
            request(BODY).send().await.unwrap().status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert!(
            binding
                .forward(
                    Method::PUT,
                    HeaderMap::new(),
                    Bytes::from_static(BODY),
                    Arc::new(crate::http::service::metrics::Metrics::new()),
                    "test-model"
                )
                .await
                .is_err(),
            "retained clients must not route after withdrawal"
        );
        assert_eq!(load().await, 0);
        stop.cancel();
        server.await.unwrap().unwrap();
        runtime.shutdown();
    })
    .await
    .unwrap();
}

#[test]
fn native_projection_counts_embeddings_and_rejects_unsafe_sampling() {
    let salted = br#"{"input_ids":[[1],[2]],"sampling_params":{"n":2},"cache_salt":["private",""],"lora_path":["adapter","other"]}"#;
    let children = Projection::read(salted)
        .unwrap()
        .children(None, true, false)
        .unwrap();
    assert_eq!(children.len(), 6);
    assert_eq!(children[2].cache_namespace.as_deref(), Some("private"));
    assert_eq!(children[2].lora.as_deref(), Some("adapter"));
    assert_eq!(children[5].cache_namespace, None);
    assert_eq!(children[5].lora.as_deref(), Some("other"));
    let body = br#"{"input_embeds":[[[0.1,0.2]],[[0.3,0.4]]],"sampling_params":{"n":2}}"#;
    let children = Projection::read(body)
        .unwrap()
        .children(None, false, false)
        .unwrap();
    assert_eq!(children.len(), 6);
    assert_eq!(
        children
            .iter()
            .filter(|c| c.kind == ChildKind::Warmup)
            .count(),
        2
    );
    for body in [
        br#"{"input_ids":[1],"sampling_params":{"n":18446744073709551615}}"#.as_slice(),
        br#"{"input_ids":[[1],[2]],"sampling_params":[{"n":2},{"n":3}]}"#,
    ] {
        assert!(
            Projection::read(body)
                .unwrap()
                .children(None, true, false)
                .is_err()
        );
    }
}

#[test]
fn native_projection_leaves_unrelated_numbers_opaque() {
    let body = br#"{"input_ids":[[1],[2]],"sampling_params":[{"n":2,"future_option":1e400},{"n":2,"future_option":-1e400}],"future_field":1e400}"#;
    let children = Projection::read(body)
        .unwrap()
        .children(None, true, false)
        .unwrap();
    assert_eq!(children.len(), 6);
    // Duplicate routing fields use the last value, including escaped keys.
    let body = br#"{"input_ids":[1],"routed_dp_rank":0,"routed_dp_ran\u006b":1}"#;
    assert_eq!(Projection::read(body).unwrap().dp_rank, Some(1));
    assert!(Projection::read(br#"{"input_ids":[1],"routed_dp_rank":1e400}"#).is_err());
    assert!(Projection::read(br#"{"input_ids":[1],"future_field":1e}"#).is_err());
}

#[test]
fn native_beam_projection_books_rows_under_each_leader() {
    let children = Projection::read(BEAM_BODY)
        .unwrap()
        .children(None, true, false)
        .unwrap();
    assert_eq!(children.len(), 2);
    assert!(
        children
            .iter()
            .all(|child| child.kind == ChildKind::Sample && child.decode_width == 4)
    );
    assert_eq!(children[0].tokens.as_slice(), &[1, 2]);
    let too_many = br#"{"input_ids":[1],"sampling_params":{"beam_width":4097}}"#;
    assert!(
        Projection::read(too_many)
            .unwrap()
            .children(None, true, false)
            .is_err()
    );
}
