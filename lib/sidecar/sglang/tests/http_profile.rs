// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use bytes::Bytes;
use dynamo_llm::{
    http::service::http_proxy::{HttpClient, forward},
    protocols::http::{Request, endpoint_name},
};
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    pipeline::{Context, RouterMode},
};

// Run against one task-owned SGLang sidecar with --enable-native-http. Use the
// normal ETCD/NATS settings and DYNAMO_PROFILE_NAMESPACE. Launch the engine under
// nsys --capture-range=cudaProfilerApi to also verify the resulting GPU trace.
#[tokio::test]
#[ignore = "requires a running SGLang engine and sidecar in DYNAMO_PROFILE_NAMESPACE"]
async fn profile_controls_use_shared_http_without_generate_adapter() {
    let runtime = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::from_settings(runtime.clone())
        .await
        .unwrap();
    let client = drt
        .namespace(std::env::var("DYNAMO_PROFILE_NAMESPACE").unwrap())
        .unwrap()
        .component("backend")
        .unwrap()
        .endpoint(endpoint_name("generate"))
        .client()
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(30), client.wait_for_instances())
        .await
        .unwrap()
        .unwrap();
    let workers = client.instance_ids();
    assert_eq!(workers.len(), 1, "use an isolated single-worker namespace");
    let worker = workers[0];
    let client = HttpClient::from_client(client, RouterMode::Direct)
        .await
        .unwrap();
    let send = |method: &str, path: &str, body: &'static [u8]| {
        let request = Context::new(Request {
            method: method.into(),
            path: path.into(),
            headers: vec![(
                "content-type".into(),
                Bytes::from_static(b"application/json"),
            )],
            body: Bytes::from_static(body),
        });
        let client = &client;
        async move {
            let response =
                tokio::time::timeout(Duration::from_secs(30), forward(client, request, worker))
                    .await
                    .unwrap()
                    .unwrap();
            assert!(response.status().is_success(), "{response:?}");
            tokio::time::timeout(
                Duration::from_secs(30),
                axum::body::to_bytes(response.into_body(), 1024 * 1024),
            )
            .await
            .unwrap()
            .unwrap()
        }
    };
    assert_eq!(
        send(
            "POST",
            "/start_profile",
            br#"{"activities":["CUDA_PROFILER"]}"#
        )
        .await,
        "Start profiling.\n"
    );
    let generation = send("POST", "/generate", br#"{"input_ids":[1,2,3],"sampling_params":{"max_new_tokens":16,"temperature":0,"ignore_eos":true}}"#).await;
    assert_eq!(
        send("GET", "/stop_profile", b"").await,
        "Stop profiling. This will take some time.\n"
    );
    let generation: serde_json::Value = serde_json::from_slice(&generation).unwrap();
    assert_eq!(generation["meta_info"]["completion_tokens"], 16);
    runtime.shutdown();
}
