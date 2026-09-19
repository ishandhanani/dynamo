// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc, time::Duration};

use axum::{
    Json, Router,
    body::Body,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use bytes::Bytes;
use dynamo_backend_common::sglang_http;
use dynamo_llm::{
    http::service::native_generate::{
        NativeGenerateClient, forward_accounted,
        lifecycle::{LifecycleClient as ControlClient, NativeAttempt, ReservedChild},
    },
    kv_router::{
        KvRouter, SelectionPolicySource, native::NativeReservation, protocols::RoutingConstraints,
        scheduling::config::KvRouterConfig,
    },
    local_model::runtime_config::ModelRuntimeConfig,
};
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    distributed::DistributedConfig,
    pipeline::{Context, RouterMode},
};
use dynamo_sidecar_common::HttpEndpoint;
use futures::StreamExt;
use tokio::{net::TcpListener, sync::watch};
use tokio_util::sync::CancellationToken;

use super::{LifecycleClient, NativeHttp, NativeHttpEndpoint, transport::HttpTransport};

const EPOCH: &str = "11111111111111111111111111111111";

fn snapshot(id: String, phase: u64) -> Response {
    if phase == 0 {
        return StatusCode::NOT_FOUND.into_response();
    }
    Json(serde_json::json!({
        "incarnation": EPOCH, "attempt_id": id, "stage": "null", "version": phase,
        "sealed": true, "cancel_requested": phase >= 2, "terminal": phase >= 3,
        "children": [{"child_id": "33333333333333333333333333333333", "rid": "client-rid",
            "kind": "sample", "dp_rank": 0, "dispatched": true, "prefill_complete": true,
            "terminal": phase >= 3}]
    }))
    .into_response()
}

async fn read_snapshot(
    State(phase): State<watch::Sender<u64>>,
    Path(id): Path<String>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    let after: i64 = query["after"].parse().unwrap();
    let mut rx = phase.subscribe();
    let current = *rx.borrow_and_update();
    if current != 0 && current < 3 && current as i64 <= after {
        rx.changed().await.unwrap();
    }
    let current = *rx.borrow();
    snapshot(id, current)
}

async fn control(
    State(phase): State<watch::Sender<u64>>,
    Path(id): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    if body["action"] == "acknowledge" {
        assert_eq!(*phase.borrow(), 3);
        phase.send_replace(4);
        return StatusCode::NO_CONTENT.into_response();
    }
    if body["action"] == "cancel" {
        phase.send_if_modified(|phase| {
            if *phase == 1 {
                *phase = 2;
                true
            } else {
                false
            }
        });
    }
    snapshot(id, *phase.borrow())
}

#[test]
fn disconnect_keeps_real_booking_until_engine_cleanup_acknowledgement() {
    super::test_runtime().block_on(async {
        tokio::time::timeout(Duration::from_secs(20), async {
            let (phase, mut observed) = watch::channel(0_u64);
            let app = Router::new()
                .route(
                    "/server_info",
                    get(|| async {
                        Json(serde_json::json!({"request_lifecycle": {
                            "version": 1, "incarnation": EPOCH, "header_overrides": true
                        }}))
                    }),
                )
                .route("/request_lifecycle/{id}", get(read_snapshot).post(control))
                .route(
                    "/generate",
                    post(
                        |State(phase): State<watch::Sender<u64>>,
                         headers: HeaderMap,
                         body: Bytes| async move {
                            assert_eq!(headers[sglang_http::lifecycle::INCARNATION_HEADER], EPOCH);
                            assert_eq!(headers["x-override-routed-dp-rank"], "0");
                            assert_eq!(
                                body.as_ref(),
                                b"{ \"input_ids\": [1, 2], \"stream\": true }"
                            );
                            phase.send_replace(1);
                            Body::from_stream(
                                futures::stream::once(async {
                                    Ok::<_, std::io::Error>(Bytes::from_static(
                                        b"opaque engine extension\r\n",
                                    ))
                                })
                                .chain(futures::stream::pending()),
                            )
                        },
                    ),
                )
                .with_state(phase.clone());
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let http = NativeHttp {
                transport: HttpTransport::new(
                    HttpEndpoint::parse(
                        &format!("http://{}", listener.local_addr().unwrap()),
                        "fixture",
                    )
                    .unwrap(),
                    Duration::from_secs(5),
                )
                .unwrap(),
            };
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let runtime = Runtime::from_current().unwrap();
            let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
            let primary = drt
                .namespace("native_accounting")
                .unwrap()
                .component("sidecar")
                .unwrap()
                .endpoint("generate");
            let endpoint = NativeHttpEndpoint::start(&primary, &http, CancellationToken::new())
                .await
                .unwrap();
            let worker_id = endpoint.instance().id();
            let lifecycle = LifecycleClient::discover(&http).await.unwrap().unwrap();
            let control_endpoint = lifecycle
                .start(&primary, CancellationToken::new())
                .await
                .unwrap();
            assert_eq!(control_endpoint.instance().id(), worker_id);
            let wire = primary
                .component()
                .endpoint(sglang_http::endpoint_name("generate"))
                .client()
                .await
                .unwrap();
            wire.wait_for_instances().await.unwrap();
            let wire = NativeGenerateClient::from_client(wire, RouterMode::Direct)
                .await
                .unwrap();
            let control = primary
                .component()
                .endpoint(sglang_http::lifecycle::endpoint_name("generate"))
                .client()
                .await
                .unwrap();
            control.wait_for_instances().await.unwrap();
            let control = Arc::new(
                ControlClient::from_client(control, RouterMode::Direct)
                    .await
                    .unwrap(),
            );
            let (_configs, rx) =
                watch::channel(HashMap::from([(worker_id, ModelRuntimeConfig::default())]));
            let router = Arc::new(
                KvRouter::new(
                    primary.clone(),
                    primary.client().await.unwrap(),
                    rx,
                    None,
                    2,
                    SelectionPolicySource::Registry,
                    Some(KvRouterConfig {
                        use_kv_events: false,
                        skip_initial_worker_wait: true,
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
            let booking = router
                .find_best_match_details_with_policy_class_admitted(
                    Some("native-child"),
                    &[1, 2],
                    None,
                    None,
                    true,
                    false,
                    None,
                    None,
                    0.0,
                    0,
                    None,
                    None,
                    Some(32),
                    None,
                    None,
                    RoutingConstraints::default(),
                )
                .await
                .unwrap()
                .into_parts()
                .1
                .unwrap();
            let descriptor = NativeAttempt::describe(&control, worker_id).await.unwrap();
            let attempt = NativeAttempt::new(
                control,
                descriptor,
                "null".into(),
                vec![ReservedChild {
                    kind: sglang_http::lifecycle::ChildKind::Sample,
                    reservation: NativeReservation::new(router.clone(), booking),
                }],
            )
            .unwrap();
            let response = forward_accounted(
                &wire,
                Context::new(sglang_http::Request {
                    method: "POST".into(),
                    headers: Vec::new(),
                    body: Bytes::from_static(b"{ \"input_ids\": [1, 2], \"stream\": true }"),
                }),
                attempt,
            )
            .await
            .unwrap();
            let mut body = response.into_body().into_data_stream();
            assert_eq!(
                body.next().await.unwrap().unwrap().as_ref(),
                b"opaque engine extension\r\n"
            );
            drop(body);
            observed.wait_for(|phase| *phase == 2).await.unwrap();
            let loads = router
                .get_potential_loads(&[], None, None, None, None)
                .await
                .unwrap();
            assert_eq!(
                loads[0].active_requests, 1,
                "HTTP drop is not an engine cleanup acknowledgement"
            );
            phase.send_replace(3);
            observed.wait_for(|phase| *phase == 4).await.unwrap();
            let loads = router
                .get_potential_loads(&[], None, None, None, None)
                .await
                .unwrap();
            assert_eq!(loads[0].active_requests, 0);
            control_endpoint.shutdown().await.unwrap();
            endpoint.shutdown().await.unwrap();
            runtime.shutdown();
            server.abort();
        })
        .await
        .unwrap();
    });
}
