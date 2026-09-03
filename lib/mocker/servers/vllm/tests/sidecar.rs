// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_backend_common::{
    DisaggregationMode, FinishReason, GenerateContext, LLMEngine, OutputOptions, PrefillResult,
    PreprocessedRequest, SamplingOptions, StopConditions,
};
use dynamo_mocker::common::protocols::MockEngineArgs;
use dynamo_vllm_mocker::{MockerServerConfig, ServerMode, VllmMockerService};
use dynamo_vllm_sidecar::VllmSidecarEngine;
use dynamo_vllm_sidecar::proto::control_server::ControlServer;
use dynamo_vllm_sidecar::proto::inference_server::InferenceServer;
use futures::StreamExt;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_stream::wrappers::TcpListenerStream;

struct RunningServer {
    endpoint: String,
    service: VllmMockerService,
    shutdown: Option<oneshot::Sender<()>>,
}

impl RunningServer {
    async fn start(mode: ServerMode, engine_args: MockEngineArgs) -> Self {
        let service = VllmMockerService::new(
            MockerServerConfig {
                mode,
                ..Default::default()
            },
            engine_args,
        )
        .unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (shutdown, shutdown_rx) = oneshot::channel();
        let inference_service = service.clone();
        let control_service = service.clone();
        let (health, health_service) = tonic_health::server::health_reporter();
        health
            .set_serving::<ControlServer<VllmMockerService>>()
            .await;
        health
            .set_serving::<InferenceServer<VllmMockerService>>()
            .await;
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(InferenceServer::new(inference_service))
                .add_service(ControlServer::new(control_service))
                .add_service(health_service)
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                    let _ = shutdown_rx.await;
                })
                .await
                .unwrap();
        });
        Self {
            endpoint: format!("http://{address}"),
            service,
            shutdown: Some(shutdown),
        }
    }
}

impl Drop for RunningServer {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
    }
}

fn fast_engine_args() -> MockEngineArgs {
    MockEngineArgs::builder()
        .block_size(4)
        .num_gpu_blocks(4096)
        .max_num_seqs(Some(64))
        .max_num_batched_tokens(Some(1024))
        .speedup_ratio(0.0)
        .dp_size(1)
        .build()
        .unwrap()
}

async fn sidecar(endpoint: &str, mode: DisaggregationMode) -> VllmSidecarEngine {
    let mut argv = vec![
        "dynamo-vllm-sidecar".to_string(),
        "--grpc-endpoint".to_string(),
        endpoint.to_string(),
        "--grpc-connections".to_string(),
        "1".to_string(),
        "--grpc-startup-deadline-secs".to_string(),
        "5".to_string(),
        "--grpc-connect-attempt-timeout-secs".to_string(),
        "1".to_string(),
    ];
    if mode != DisaggregationMode::Aggregated {
        argv.extend(["--disaggregation-mode".to_string(), mode.to_string()]);
    }
    tokio::task::spawn_blocking(move || VllmSidecarEngine::from_args(Some(argv)))
        .await
        .unwrap()
        .unwrap()
        .0
}

fn request(max_tokens: u32) -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("mocker-model".to_string())
        .token_ids(vec![11, 22, 33, 44])
        .stop_conditions(StopConditions {
            max_tokens: Some(max_tokens),
            ignore_eos: Some(true),
            ..Default::default()
        })
        .sampling_options(SamplingOptions {
            temperature: Some(0.0),
            ..Default::default()
        })
        .output_options(OutputOptions {
            logprobs: Some(2),
            prompt_logprobs: Some(1),
            ..Default::default()
        })
        .build()
        .unwrap()
}

async fn collect(
    engine: &VllmSidecarEngine,
    request: PreprocessedRequest,
) -> Vec<dynamo_backend_common::LLMEngineOutput> {
    let context = dynamo_backend_common::testing::mock_context();
    engine
        .generate(request, GenerateContext::new(context, None))
        .await
        .unwrap()
        .map(|item| item.unwrap())
        .collect()
        .await
}

/// With `--selection-catalog-url`, the sidecar registers the worker with a
/// selection service over HTTP (dispatch endpoint = advertised gRPC address,
/// KV-event ZMQ endpoints rehosted onto it, leased), keeps the lease alive,
/// and deregisters on cleanup.
#[tokio::test]
async fn sidecar_registers_with_a_selection_catalog_and_deregisters_on_cleanup() {
    use dynamo_kv_router::WorkerType;
    use dynamo_kv_router::config::KvRouterConfig;
    use dynamo_kv_router::services::selection::{
        AppState, SelectionServiceBuilder, WorkerLifecycle, WorkerSelectionPolicyRegistry,
        create_router,
    };

    let selection = SelectionServiceBuilder::new(
        KvRouterConfig {
            use_kv_events: false,
            router_queue_threshold: None,
            ..Default::default()
        },
        WorkerType::Aggregated,
        WorkerSelectionPolicyRegistry::default(),
    )
    .indexer_threads(1)
    .build()
    .await
    .expect("selection service");
    let selection = Arc::new(selection);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let catalog_url = format!("http://{}", listener.local_addr().unwrap());
    let app = create_router(Arc::new(AppState {
        service: Arc::clone(&selection),
    }));
    let http = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let server = RunningServer::start(ServerMode::Aggregated, fast_engine_args()).await;
    let engine_argv = vec![
        "dynamo-vllm-sidecar".to_string(),
        "--grpc-endpoint".to_string(),
        server.endpoint.clone(),
        "--grpc-connections".to_string(),
        "1".to_string(),
        "--grpc-startup-deadline-secs".to_string(),
        "5".to_string(),
        "--grpc-connect-attempt-timeout-secs".to_string(),
        "1".to_string(),
        "--advertise-grpc-endpoint".to_string(),
        "http://worker-7.example:50051".to_string(),
        "--selection-catalog-url".to_string(),
        catalog_url.clone(),
        "--selection-catalog-ttl-secs".to_string(),
        "0.4".to_string(),
    ];
    let engine =
        tokio::task::spawn_blocking(move || VllmSidecarEngine::from_args(Some(engine_argv)))
            .await
            .unwrap()
            .unwrap()
            .0;
    engine
        .start(7)
        .await
        .expect("start with catalog registration");

    let records = selection.list_workers(None, None);
    assert_eq!(records.len(), 1, "{records:?}");
    let record = &records[0];
    assert_eq!(record.worker_id, 7);
    assert_eq!(record.routing_group, "default");
    assert_eq!(
        record.endpoint.as_deref(),
        Some("http://worker-7.example:50051")
    );
    assert_eq!(record.block_size, Some(4));
    assert_eq!(record.ttl_secs, Some(0.4));
    assert_eq!(
        record.lifecycle,
        WorkerLifecycle::Schedulable,
        "{:?}",
        record.not_schedulable_reasons
    );
    for endpoint in record.kv_events_endpoints.values() {
        assert!(
            endpoint.starts_with("tcp://worker-7.example:"),
            "KV event endpoint must be rehosted onto the advertised host: {endpoint}"
        );
    }

    // Heartbeats keep the lease alive well past the TTL.
    tokio::time::sleep(std::time::Duration::from_millis(900)).await;
    assert!(selection.expire_leases().await.is_empty());
    assert_eq!(
        selection.list_workers(None, None)[0].lifecycle,
        WorkerLifecycle::Schedulable
    );

    // Cleanup deregisters promptly instead of waiting for lease expiry.
    engine.cleanup().await.expect("cleanup");
    let record = &selection.list_workers(None, None)[0];
    assert_eq!(record.lifecycle, WorkerLifecycle::Unschedulable);

    // A registration that cannot reach a frontend is rejected up front.
    let bad_argv = vec![
        "dynamo-vllm-sidecar".to_string(),
        "--grpc-endpoint".to_string(),
        server.endpoint.clone(),
        "--selection-catalog-url".to_string(),
        catalog_url,
    ];
    let error = tokio::task::spawn_blocking(move || VllmSidecarEngine::from_args(Some(bad_argv)))
        .await
        .unwrap()
        .err()
        .expect("catalog registration requires an advertised endpoint");
    assert!(
        error.to_string().contains("advertise-grpc-endpoint"),
        "{error}"
    );

    http.abort();
    selection.shutdown().await;
}

/// A frontend with direct dispatch reaches the same vLLM gRPC service the
/// sidecar does, through the same request/response conversion and error
/// mapping, and an unreachable endpoint surfaces as a connect failure.
#[tokio::test]
async fn direct_engine_factory_streams_from_the_mock_server() {
    use dynamo_backend_common::{
        DirectEngineFactory, ModelRuntimeConfig, NATIVE_GRPC_MODE_RUNTIME_KEY,
    };
    use dynamo_runtime::pipeline::Context;
    use dynamo_sidecar_common::GrpcTransportConfig;
    use dynamo_vllm_sidecar::VllmDirectEngineFactory;
    use std::num::NonZeroUsize;
    use std::time::Duration;

    let server = RunningServer::start(ServerMode::Aggregated, fast_engine_args()).await;
    let transport = GrpcTransportConfig {
        connections: NonZeroUsize::MIN,
        startup_deadline: Duration::from_secs(5),
        connect_attempt_timeout: Duration::from_secs(1),
        ..Default::default()
    };
    let factory = VllmDirectEngineFactory::new(transport);
    let mut config = ModelRuntimeConfig::default();
    config.runtime_data.insert(
        NATIVE_GRPC_MODE_RUNTIME_KEY.to_string(),
        serde_json::Value::String("agg".to_string()),
    );

    let engine = factory
        .connect(7, &server.endpoint, &config)
        .await
        .expect("direct engine connects to the mock server");
    let outputs: Vec<_> = engine
        .generate(Context::new(request(3)))
        .await
        .expect("direct generate")
        .collect()
        .await;
    let tokens: usize = outputs
        .iter()
        .filter_map(|item| item.data.as_ref())
        .map(|output| output.token_ids.len())
        .sum();
    assert_eq!(tokens, 3, "{outputs:?}");
    assert!(outputs.iter().all(|item| item.error.is_none()));
    let terminal = outputs
        .iter()
        .rev()
        .find_map(|item| item.data.as_ref())
        .expect("terminal chunk");
    assert_eq!(terminal.finish_reason, Some(FinishReason::Length));
    assert_eq!(server.service.active_request_count(), 0);

    let unreachable = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let unreachable_endpoint = format!("http://{}", unreachable.local_addr().unwrap());
    drop(unreachable);
    let error = match factory.connect(8, &unreachable_endpoint, &config).await {
        Ok(_) => panic!("unreachable endpoint must fail to connect"),
        Err(error) => error,
    };
    let text = format!("{error:#}").to_ascii_lowercase();
    assert!(
        text.contains("connect") || text.contains("deadline") || text.contains("unavailable"),
        "{text}"
    );
}

#[tokio::test]
async fn sidecar_streams_mocker_tokens_logprobs_and_usage() {
    let server = RunningServer::start(ServerMode::Aggregated, fast_engine_args()).await;
    let engine = sidecar(&server.endpoint, DisaggregationMode::Aggregated).await;
    engine.start(0).await.unwrap();

    let outputs = collect(&engine, request(3)).await;
    assert_eq!(outputs.len(), 3);
    assert!(outputs.iter().all(|output| output.token_ids.len() == 1));
    assert!(
        outputs
            .iter()
            .all(|output| output.log_probs.as_ref().unwrap().len() == 1)
    );
    assert!(
        outputs
            .iter()
            .all(|output| output.top_logprobs.as_ref().unwrap()[0].len() == 3)
    );
    let terminal = outputs.last().unwrap();
    assert_eq!(terminal.finish_reason, Some(FinishReason::Length));
    let usage = terminal.completion_usage.as_ref().unwrap();
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (4, 3));
    assert!(terminal.engine_data.as_ref().unwrap()["prompt_logprobs"].is_array());
    assert_eq!(server.service.active_request_count(), 0);
}

#[tokio::test]
async fn prefill_handoff_round_trips_through_a_decode_server() {
    let prefill_server = RunningServer::start(ServerMode::Prefill, fast_engine_args()).await;
    let decode_server = RunningServer::start(ServerMode::Decode, fast_engine_args()).await;
    let prefill = sidecar(&prefill_server.endpoint, DisaggregationMode::Prefill).await;
    let decode = sidecar(&decode_server.endpoint, DisaggregationMode::Decode).await;
    prefill.start(0).await.unwrap();
    decode.start(1).await.unwrap();

    let prefill_outputs = collect(&prefill, request(3)).await;
    assert_eq!(prefill_outputs.len(), 1);
    assert!(prefill_outputs[0].token_ids.is_empty());
    let handoff = prefill_outputs[0]
        .disaggregated_params
        .clone()
        .expect("prefill response should carry an opaque KV handoff");
    assert_eq!(handoff["do_remote_prefill"], true);
    assert!(handoff["remote_engine_id"].is_string());
    // The non-rendezvous sentinel proves the sidecar preserved opaque handoff
    // fields rather than reconstructing only the keys it recognizes.
    assert!(
        handoff["mocker_request_id"].is_string(),
        "sidecar must forward opaque KV-transfer fields verbatim"
    );

    let mut decode_request = request(3);
    decode_request.prefill_result = Some(PrefillResult {
        disaggregated_params: handoff,
        prompt_tokens_details: None,
    });
    let decode_outputs = collect(&decode, decode_request).await;
    assert_eq!(decode_outputs.len(), 3);
    assert_eq!(
        decode_outputs.last().unwrap().finish_reason,
        Some(FinishReason::Length)
    );
}

#[tokio::test]
async fn dropping_sidecar_stream_cancels_mocker_work() {
    let mut args = fast_engine_args();
    args.speedup_ratio = 0.1;
    let server = RunningServer::start(ServerMode::Aggregated, args).await;
    let engine = sidecar(&server.endpoint, DisaggregationMode::Aggregated).await;
    engine.start(0).await.unwrap();

    let context = dynamo_backend_common::testing::mock_context();
    let mut stream = engine
        .generate(
            request(10_000),
            GenerateContext::new(Arc::clone(&context), None),
        )
        .await
        .unwrap();
    let first = stream.next().await.unwrap().unwrap();
    assert!(first.finish_reason.is_none());
    context.stop_generating();
    let terminal = stream.next().await.unwrap().unwrap();
    assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
    drop(stream);

    let mut metrics = server.service.metrics_receiver();
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            let snapshot = metrics.borrow_and_update().clone();
            if server.service.active_request_count() == 0
                && snapshot.running_requests == 0
                && snapshot.waiting_requests == 0
            {
                break;
            }
            metrics.changed().await.unwrap();
        }
    })
    .await
    .expect("dropping the gRPC stream should cancel scheduler work promptly");
}
