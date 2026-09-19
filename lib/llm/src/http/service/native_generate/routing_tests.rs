// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashMap,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
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
    protocols::{Annotated, sglang::http::ResponseFrame},
};

const BODY: &[u8] = br#"{"input_ids": [[1,2], [3]], "stream":false, "routed_dp_rank":1, "sampling_params":{"n":2,"future_option":1e400}, "future_field":{"number":1e400,"float":1.234567890123456789} }"#;

const BEAM_BODY: &[u8] =
    br#"{"input_ids":[[1,2],[3]],"routed_dp_rank":1,"sampling_params":{"beam_width":4,"n":2}}"#;

struct Engine {
    worker_id: u64,
    kv: Arc<KvRouter>,
    expected_rows: AtomicUsize,
    received: Mutex<Option<http::Request>>,
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
        let rows: usize = self
            .kv
            .get_potential_loads(&[], None, None, None, None)
            .await
            .unwrap()
            .iter()
            .map(|load| load.active_requests)
            .sum();
        assert_eq!(rows, self.expected_rows.load(Ordering::SeqCst));
        let projection = Projection::read(&request.body).unwrap();
        assert_eq!(projection.dp_rank, Some(1));
        assert!(
            std::str::from_utf8(&request.body)
                .unwrap()
                .contains("1e400")
                || request
                    .body
                    .as_ref()
                    .windows(10)
                    .any(|s| s == b"beam_width")
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
            kv: kv.clone(),
            expected_rows: AtomicUsize::new(0),
            received: Mutex::new(None),
        });
        let mut card = ModelDeploymentCard::with_name_only("native-model");
        card.runtime_config = config;
        card.runtime_config
            .runtime_data
            .insert(http::CAPABILITY.into(), true.into());
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
        for (index, (body, rows)) in [(BODY, 4), (BEAM_BODY, 8)].into_iter().enumerate() {
            *engine.received.lock().unwrap() = None;
            engine.expected_rows.store(rows, Ordering::SeqCst);
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
            while load().await != 0 {
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
fn native_session_projection_requires_a_session_affinity_route() {
    let body = br#"{"input_ids":[1],"session_params":{"id":"s","rid":"turn","future":1e400}}"#;
    assert!(
        Projection::read(body)
            .unwrap()
            .children(None, true)
            .is_err()
    );
}

#[test]
fn native_projection_counts_embeddings_and_rejects_unsafe_sampling() {
    let salted = br#"{"input_ids":[[1],[2]],"sampling_params":{"n":2},"cache_salt":["private",""],"lora_path":["adapter","other"]}"#;
    let children = Projection::read(salted)
        .unwrap()
        .children(None, true)
        .unwrap();
    assert_eq!(children.len(), 4);
    assert_eq!(children[0].cache_namespace.as_deref(), Some("private"));
    assert_eq!(children[0].lora.as_deref(), Some("adapter"));
    assert_eq!(children[3].cache_namespace, None);
    assert_eq!(children[3].lora.as_deref(), Some("other"));
    let body = br#"{"input_embeds":[[[0.1,0.2]],[[0.3,0.4]]],"sampling_params":{"n":2}}"#;
    let children = Projection::read(body)
        .unwrap()
        .children(None, false)
        .unwrap();
    assert_eq!(children.len(), 4);
    for body in [
        br#"{"input_ids":[1],"sampling_params":{"n":18446744073709551615}}"#.as_slice(),
        br#"{"input_ids":[[1],[2]],"sampling_params":[{"n":2},{"n":3}]}"#,
    ] {
        assert!(
            Projection::read(body)
                .unwrap()
                .children(None, true)
                .is_err()
        );
    }
}

#[test]
fn native_projection_leaves_unrelated_numbers_opaque() {
    let body = br#"{"input_ids":[[1],[2]],"sampling_params":[{"n":2,"future_option":1e400},{"n":2,"future_option":-1e400}],"future_field":1e400}"#;
    let children = Projection::read(body)
        .unwrap()
        .children(None, true)
        .unwrap();
    assert_eq!(children.len(), 4);
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
        .children(None, true)
        .unwrap();
    assert_eq!(children.len(), 2);
    assert!(children.iter().all(|child| child.decode_width == 4));
    assert_eq!(children[0].tokens.as_slice(), &[1, 2]);
    let too_many = br#"{"input_ids":[1],"sampling_params":{"beam_width":4097}}"#;
    assert!(
        Projection::read(too_many)
            .unwrap()
            .children(None, true)
            .is_err()
    );
}
