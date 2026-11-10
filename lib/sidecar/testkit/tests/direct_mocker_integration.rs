// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct mode: the frontend dials mocker gRPC servers itself, with no
//! sidecar process and no discovery registration.

use std::path::PathBuf;
use std::time::Duration;

use dynamo_backend_common::DisaggregationMode;
use dynamo_llm::endpoint_type::EndpointType;
use dynamo_llm::entrypoint::input::direct::register_direct_model;
use dynamo_llm::entrypoint::{EngineConfig, RouterConfig};
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_llm::local_model::{LocalModel, LocalModelBuilder};
use dynamo_runtime::pipeline::RouterMode;
use dynamo_runtime::protocols::EndpointId;
use dynamo_sidecar_common::GrpcTransportConfig;
use dynamo_sidecar_common::direct::{DirectBackend, DirectEngineFacts, DirectFleet, RoleHint};
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
use support::{FixtureConfig, HandoffFixture, ProcessFixture, sglang};

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
        let local_model = Self::local_model(env, fleet.primary(), router_mode).await;
        Self::serve(env, fleet.into_engine_config(local_model)).await
    }

    async fn local_model(
        env: &Environment,
        primary: &DirectEngineFacts,
        router_mode: RouterMode,
    ) -> LocalModel {
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
        builder.build().await.expect("local model")
    }

    async fn serve(env: &Environment, config: EngineConfig) -> Self {
        let EngineConfig::Direct {
            model,
            workers,
            dispatch,
        } = config
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
        // otherwise prefill completes first and decode follows.
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

/// Engine pods in a throwaway namespace of the cluster `KUBECONFIG` names.
///
/// The pods name no scheduler that exists, so no kubelet runs them; the test
/// writes their status itself, pointing each at a local mocker.
mod engine_pods {
    use k8s_openapi::api::core::v1::{Namespace, Pod};
    use kube::api::{Api, DeleteParams, Patch, PatchParams, PostParams};
    use serde_json::json;

    pub struct Cluster {
        pub client: kube::Client,
        pub namespace: String,
        pods: Api<Pod>,
    }

    impl Cluster {
        pub async fn new() -> Self {
            let client = kube::Client::try_default()
                .await
                .expect("KUBECONFIG cluster");
            let namespace = format!("dyn-direct-pods-{:08x}", rand::random::<u32>());
            let namespaces: Api<Namespace> = Api::all(client.clone());
            namespaces
                .create(
                    &PostParams::default(),
                    &serde_json::from_value(json!({"metadata": {"name": namespace}})).unwrap(),
                )
                .await
                .expect("create test namespace");
            let pods = Api::namespaced(client.clone(), &namespace);
            Self {
                client,
                namespace,
                pods,
            }
        }

        /// An engine pod of `app`, serving the mocker at `endpoint` once ready.
        pub async fn engine(&self, name: &str, app: &str, endpoint: &str) {
            let port = endpoint.rsplit(':').next().unwrap();
            let pod = json!({
                "metadata": {
                    "name": name,
                    "labels": {"app": app},
                    "annotations": {"nvidia.com/dynamo-engine-grpc-port": port},
                },
                "spec": {
                    "schedulerName": "dynamo-test-unscheduled",
                    "containers": [{"name": "engine", "image": "none"}],
                },
            });
            self.pods
                .create(
                    &PostParams::default(),
                    &serde_json::from_value(pod).unwrap(),
                )
                .await
                .expect("create engine pod");
        }

        pub async fn ready(&self, name: &str, ready: bool) {
            let status = json!({"status": {
                "phase": "Running",
                "podIP": "127.0.0.1",
                "podIPs": [{"ip": "127.0.0.1"}],
                "conditions": [{"type": "Ready", "status": if ready { "True" } else { "False" }}],
            }});
            self.pods
                .patch_status(name, &PatchParams::default(), &Patch::Merge(status))
                .await
                .expect("patch engine pod status");
        }

        pub async fn delete(self) {
            let namespaces: Api<Namespace> = Api::all(self.client);
            let _ = namespaces
                .delete(&self.namespace, &DeleteParams::default())
                .await;
        }
    }
}

/// Send completions until `want` engines' pattern holds `times` in a row.
async fn until_served_by<F: DirectFixture>(
    frontend: &DirectFrontend,
    model: &str,
    controls: &[&Controller<F::Protocol>],
    index: &mut usize,
    want: &[bool],
    times: usize,
) {
    let mut streak = 0;
    for _ in 0..40 {
        *index += 1;
        if served_by::<F>(frontend, model, controls, *index).await == want {
            streak += 1;
            if streak == times {
                return;
            }
        } else {
            streak = 0;
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }
    panic!("engines never served as {want:?}");
}

/// Send one completion; return which engines received it, by controller.
async fn served_by<F: DirectFixture>(
    frontend: &DirectFrontend,
    model: &str,
    controls: &[&Controller<F::Protocol>],
    index: usize,
) -> Vec<bool> {
    let id = format!("00000000-0000-4000-8000-{:012}", 500 + index);
    let handles: Vec<_> = controls
        .iter()
        .map(|control| control.request(&id, RequestPlan::default()))
        .collect();
    let body = frontend.complete(model, &id).await;
    assert_eq!(body["choices"][0]["finish_reason"], "length", "{body}");
    let served: Vec<bool> = handles
        .iter()
        .map(|handle| handle.reached(Event::Received))
        .collect();
    assert_eq!(served.iter().filter(|s| **s).count(), 1, "{served:?}");
    served
}

/// A frontend started from `--discovery-backend kubernetes-pods` follows its
/// engine pods: the first Ready pod seeds it, a second joins when Ready, and
/// the first leaves when it stops being Ready.
async fn engines_follow_pods<F: DirectFixture>() {
    use dynamo_sidecar_common::direct::pods::{PodSelector, probe_first_pods};

    let env = Environment::new().await;
    let cluster = engine_pods::Cluster::new().await;
    let (control_a, control_b) = (Controller::default(), Controller::default());
    let config = || FixtureConfig {
        model: env.model.clone(),
        ..Default::default()
    };
    let mut peer_a = F::start(control_a.clone(), config()).await;
    let mut peer_b = F::start(control_b.clone(), config()).await;

    cluster
        .engine("engine-a", "engine", &peer_a.endpoint())
        .await;
    cluster.ready("engine-a", true).await;
    let cancel = CancellationToken::new();
    let selector = PodSelector {
        namespace: cluster.namespace.clone(),
        labels: "app=engine".to_string(),
        port: None,
        role: RoleHint::Serving,
    };
    let first = bounded(
        "first engine pod",
        probe_first_pods(
            F::backend(),
            cluster.client.clone(),
            selector,
            None,
            &cancel,
        ),
    )
    .await
    .expect("probe first engine pod")
    .expect("not cancelled");
    let local_model =
        DirectFrontend::local_model(&env, first.fleet.primary(), RouterMode::RoundRobin).await;
    let frontend = DirectFrontend::serve(&env, first.serve(local_model, cancel.clone())).await;

    let controls = [&control_a, &control_b];
    let mut index = 0;
    until_served_by::<F>(
        &frontend,
        &env.model,
        &controls,
        &mut index,
        &[true, false],
        2,
    )
    .await;

    cluster
        .engine("engine-b", "engine", &peer_b.endpoint())
        .await;
    cluster.ready("engine-b", true).await;
    until_served_by::<F>(
        &frontend,
        &env.model,
        &controls,
        &mut index,
        &[false, true],
        1,
    )
    .await;

    cluster.ready("engine-a", false).await;
    until_served_by::<F>(
        &frontend,
        &env.model,
        &controls,
        &mut index,
        &[false, true],
        4,
    )
    .await;

    cancel.cancel();
    frontend.shutdown().await;
    cluster.delete().await;
    peer_a.shutdown().await;
    peer_b.shutdown().await;
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
            #[ignore = "needs a Kubernetes cluster from KUBECONFIG"]
            async fn engines_follow_kubernetes_pods() {
                tokio::time::timeout(Duration::from_secs(90), engines_follow_pods::<$fixture>())
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
