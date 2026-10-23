// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct mode: the frontend dials mocker gRPC servers itself, with no
//! sidecar process and no discovery registration, for SGLang and vLLM.

use std::path::PathBuf;
use std::time::Duration;

use dynamo_backend_common::DisaggregationMode;
use dynamo_llm::endpoint_type::EndpointType;
use dynamo_llm::entrypoint::input::direct::register_direct_model;
use dynamo_llm::entrypoint::{EngineConfig, RouterConfig};
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_llm::local_model::LocalModelBuilder;
use dynamo_runtime::pipeline::RouterMode;
use dynamo_runtime::protocols::EndpointId;
use dynamo_sidecar_common::GrpcTransportConfig;
use dynamo_sidecar_common::direct::{DirectBackend, DirectFleet};
use dynamo_sidecar_testkit::{
    bounded,
    control::{Controller, Event, OpenAction, RequestPlan},
};
use serde_json::Value;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

#[path = "support/process.rs"]
#[allow(dead_code)]
mod process;
#[allow(dead_code)]
mod support;

use process::Environment;
use support::{FixtureConfig, HandoffFixture, ProcessFixture, sglang, vllm};

/// A mocker fixture plus the direct backend that dials it.
trait DirectFixture: ProcessFixture + HandoffFixture {
    type Backend: DirectBackend;
    fn backend() -> std::sync::Arc<Self::Backend>;
}

impl DirectFixture for sglang::Fixture {
    type Backend = dynamo_sglang_sidecar::direct::SglangDirectDispatch;
    fn backend() -> std::sync::Arc<Self::Backend> {
        dynamo_sglang_sidecar::direct::SglangDirectDispatch::new(GrpcTransportConfig::default())
    }
}

impl DirectFixture for vllm::Fixture {
    type Backend = dynamo_vllm_sidecar::direct::VllmDirectDispatch;
    fn backend() -> std::sync::Arc<Self::Backend> {
        dynamo_vllm_sidecar::direct::VllmDirectDispatch::new(GrpcTransportConfig::default())
    }
}

struct DirectFrontend {
    url: String,
    client: reqwest::Client,
    cancel: CancellationToken,
    service: JoinHandle<anyhow::Result<()>>,
}

impl DirectFrontend {
    async fn start<F: DirectFixture>(
        env: &Environment,
        urls: &[String],
        prefill_urls: &[String],
        router_mode: RouterMode,
    ) -> Self {
        let fleet = DirectFleet::probe(F::backend(), urls, prefill_urls)
            .await
            .expect("probe mocker engines");
        let primary = fleet.primary();
        assert_eq!(primary.model_path, env.model);
        let mut builder = LocalModelBuilder::default();
        builder
            .model_path(PathBuf::from(&env.model))
            .model_name(Some(env.model.clone()))
            .endpoint_id(Some(EndpointId {
                namespace: env.namespace.clone(),
                component: "backend".to_string(),
                name: "generate".to_string(),
            }))
            .kv_cache_block_size(primary.kv_cache_block_size)
            .runtime_config(primary.runtime_config.clone())
            .router_config(Some(RouterConfig {
                router_mode,
                ..Default::default()
            }));
        let local_model = builder.build().await.expect("local model");
        let EngineConfig::Direct {
            model,
            workers,
            dispatch,
        } = fleet.into_engine_config(local_model)
        else {
            unreachable!("fleet builds a direct engine config");
        };

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let service = HttpService::builder()
            .host("127.0.0.1")
            .port(address.port())
            .enable_cmpl_endpoints(true)
            .build()
            .unwrap();
        register_direct_model(
            &env.runtime,
            service.state().manager_clone(),
            service.state().metrics_clone(),
            &model,
            workers,
            dispatch,
        )
        .await
        .expect("register direct model");
        for endpoint_type in EndpointType::all() {
            if endpoint_type != EndpointType::Batch {
                service.enable_model_endpoint(endpoint_type, true).unwrap();
            }
        }
        let cancel = CancellationToken::new();
        let service = service.spawn_with_listener(cancel.clone(), listener).await;
        Self {
            url: format!("http://{address}/v1/completions"),
            client: reqwest::Client::builder().no_proxy().build().unwrap(),
            cancel,
            service,
        }
    }

    async fn complete(&self, model: &str, id: &str) -> Value {
        let response = bounded(
            "direct HTTP completion",
            self.client
                .post(&self.url)
                .header("x-dynamo-request-id", id)
                .json(&serde_json::json!({
                    "model": model, "prompt": "hello", "max_tokens": 3,
                    "stream": false, "temperature": 0, "ignore_eos": true,
                    "nvext": {"extra_fields": ["completion_token_ids"]},
                }))
                .send(),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        response.json().await.unwrap()
    }

    async fn shutdown(mut self) {
        self.cancel.cancel();
        bounded("direct HTTP frontend shutdown", &mut self.service)
            .await
            .unwrap()
            .unwrap();
    }
}

fn completion_token_ids(body: &Value) -> Vec<u32> {
    body["nvext"]["completion_token_ids"]
        .as_array()
        .unwrap_or_else(|| panic!("completion_token_ids missing: {body}"))
        .iter()
        .map(|token| token.as_u64().unwrap() as u32)
        .collect()
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

async fn aggregated<F: DirectFixture>(router_mode: RouterMode, kv_events: bool) {
    let env = Environment::new().await;
    let control = Controller::default();
    let mut peers = Vec::new();
    for _ in 0..2 {
        peers.push(
            F::start(
                control.clone(),
                FixtureConfig {
                    model: env.model.clone(),
                    kv_events_port: kv_events.then(free_port),
                    ..Default::default()
                },
            )
            .await,
        );
    }
    let urls: Vec<String> = peers.iter().map(ProcessFixture::endpoint).collect();
    let frontend = DirectFrontend::start::<F>(&env, &urls, &[], router_mode).await;

    for index in 0..4 {
        // The HTTP frontend only honors UUID-shaped request ids.
        let id = format!("00000000-0000-4000-8000-{:012}", index + 1);
        let handle = control.request(&id, RequestPlan::default());
        let body = frontend.complete(&env.model, &id).await;
        assert_eq!(body["choices"][0]["finish_reason"], "length", "{body}");
        assert_eq!(handle.tokens().len(), 3, "{body}");
        assert_eq!(completion_token_ids(&body), handle.tokens(), "{body}");
        bounded("native request release", handle.wait(Event::Dropped)).await;
    }

    frontend.shutdown().await;
    for mut peer in peers {
        peer.shutdown().await;
    }
}

async fn prefill_decode<F: DirectFixture>() {
    let env = Environment::new().await;
    let prefill_control = Controller::default();
    let decode_control = Controller::default();
    let mut prefill_peer = F::start(
        prefill_control.clone(),
        FixtureConfig {
            model: env.model.clone(),
            disaggregation_mode: DisaggregationMode::Prefill,
            ..Default::default()
        },
    )
    .await;
    let mut decode_peer = F::start(
        decode_control.clone(),
        FixtureConfig {
            model: env.model.clone(),
            disaggregation_mode: DisaggregationMode::Decode,
            ..Default::default()
        },
    )
    .await;
    let frontend = DirectFrontend::start::<F>(
        &env,
        &[decode_peer.endpoint()],
        &[prefill_peer.endpoint()],
        RouterMode::RoundRobin,
    )
    .await;

    for id in [
        "00000000-0000-4000-8000-000000000101",
        "00000000-0000-4000-8000-000000000102",
    ] {
        // With a bootstrap rendezvous (SGLang) decode is dispatched as soon
        // as prefill hands off its tuple, while prefill is still held open;
        // otherwise (vLLM) prefill completes first and decode follows.
        let observed_prefill = prefill_control.request(
            id,
            RequestPlan {
                open: if F::HAS_BOOTSTRAP {
                    OpenAction::Hold
                } else {
                    OpenAction::Continue
                },
                stream: None,
            },
        );
        let observed_decode = decode_control.request(id, RequestPlan::default());
        let orchestration = async {
            bounded(
                "prefill native acceptance",
                observed_prefill.wait(Event::Received),
            )
            .await;
            if F::HAS_BOOTSTRAP {
                bounded(
                    "decode native acceptance",
                    observed_decode.wait(Event::Received),
                )
                .await;
                assert!(observed_prefill.tokens().is_empty());
                observed_prefill.release();
            }
        };
        let (body, ()) = tokio::join!(frontend.complete(&env.model, id), orchestration);
        assert_eq!(body["choices"][0]["finish_reason"], "length", "{body}");
        assert_eq!(observed_decode.tokens().len(), 3, "{body}");
        assert_eq!(
            completion_token_ids(&body),
            observed_decode.tokens(),
            "{body}"
        );
        F::assert_handoff(&observed_prefill, &observed_decode, id);
    }

    frontend.shutdown().await;
    prefill_peer.shutdown().await;
    decode_peer.shutdown().await;
}

macro_rules! direct_tests {
    ($prefix:ident, $fixture:ty) => {
        mod $prefix {
            use super::*;

            #[tokio::test]
            async fn round_robin_completions_over_two_engines() {
                tokio::time::timeout(
                    Duration::from_secs(60),
                    aggregated::<$fixture>(RouterMode::RoundRobin, false),
                )
                .await
                .expect("test completes");
            }

            #[tokio::test]
            async fn kv_routing_reads_engine_kv_events() {
                tokio::time::timeout(
                    Duration::from_secs(60),
                    aggregated::<$fixture>(RouterMode::KV, true),
                )
                .await
                .expect("test completes");
            }

            #[tokio::test]
            async fn prefill_decode_handoff() {
                tokio::time::timeout(Duration::from_secs(60), prefill_decode::<$fixture>())
                    .await
                    .expect("test completes");
            }
        }
    };
}

direct_tests!(sglang_direct, sglang::Fixture);
direct_tests!(vllm_direct, vllm::Fixture);
