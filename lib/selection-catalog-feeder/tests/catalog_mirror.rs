// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The mirror against a real in-process selection service: Ready pods become
//! schedulable workers, absent pods are deregistered, heartbeats keep leases.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use dynamo_kv_router::WorkerType;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::services::selection::{
    AppState, SelectionServiceBuilder, WorkerLifecycle, WorkerSelectionPolicyRegistry,
    create_router,
};
use dynamo_selection_catalog_feeder::{
    CatalogMirror, FeederConfig, desired_registrations, parse_selector,
};
use dynamo_sidecar_common::{CatalogClient, HttpEndpoint};
use k8s_openapi::api::core::v1::{Pod, PodCondition, PodStatus};
use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
use tokio::net::TcpListener;

fn pod(name: &str, ip: &str, ready: bool) -> Pod {
    Pod {
        metadata: ObjectMeta {
            name: Some(name.into()),
            labels: Some(BTreeMap::from([("app".to_string(), "vllm".to_string())])),
            ..Default::default()
        },
        status: Some(PodStatus {
            pod_ip: Some(ip.into()),
            conditions: Some(vec![PodCondition {
                type_: "Ready".into(),
                status: if ready { "True" } else { "False" }.into(),
                ..Default::default()
            }]),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[tokio::test]
async fn mirror_registers_ready_pods_and_deregisters_absent_ones() {
    let selection = Arc::new(
        SelectionServiceBuilder::new(
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
        .expect("selection service"),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let catalog_url = format!("http://{}", listener.local_addr().unwrap());
    let app = create_router(Arc::new(AppState {
        service: Arc::clone(&selection),
    }));
    let http = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let config = FeederConfig {
        model_name: "m".into(),
        routing_group: "default".into(),
        endpoint_scheme: "http".into(),
        port: 50051,
        block_size: 16,
        data_parallel_size: 1,
        kv_event_port: None,
        kv_event_port_stride: 1,
        replay_port: None,
        total_kv_blocks: Some(4096),
        max_num_batched_tokens: Some(8192),
        ttl: Duration::from_secs(30),
    };
    let selector = parse_selector("app=vllm").unwrap();
    let client = CatalogClient::new(HttpEndpoint::parse(&catalog_url, "catalog").unwrap()).unwrap();
    let mut mirror = CatalogMirror::new(client);

    // Two Ready pods, one not Ready.
    let pods = [
        pod("w-a", "10.0.0.1", true),
        pod("w-b", "10.0.0.2", true),
        pod("w-c", "10.0.0.3", false),
    ];
    let desired = desired_registrations(pods.iter(), &selector, &config);
    assert_eq!(desired.len(), 2);
    mirror.reconcile(&desired).await;
    let records = selection.list_workers(None, None);
    assert_eq!(records.len(), 2, "{records:?}");
    assert!(
        records
            .iter()
            .all(|record| record.lifecycle == WorkerLifecycle::Schedulable),
        "{records:?}"
    );
    assert_eq!(mirror.registered_worker_ids().len(), 2);

    // Leases renew.
    mirror.heartbeat_all().await;
    assert_eq!(mirror.registered_worker_ids().len(), 2);

    // Pod w-b leaves; an unchanged w-a is not re-registered.
    let pods = [pod("w-a", "10.0.0.1", true), pod("w-c", "10.0.0.3", false)];
    let desired = desired_registrations(pods.iter(), &selector, &config);
    mirror.reconcile(&desired).await;
    // Deregistration drains: the record stays but is no longer schedulable.
    let schedulable: Vec<u64> = selection
        .list_workers(None, None)
        .into_iter()
        .filter(|record| record.lifecycle == WorkerLifecycle::Schedulable)
        .map(|record| record.worker_id)
        .collect();
    assert_eq!(
        schedulable,
        vec![dynamo_runtime::discovery::hash_pod_name("w-a")]
    );
    assert_eq!(mirror.registered_worker_ids().len(), 1);

    // Shutdown drains everything.
    mirror.clear().await;
    assert!(
        selection
            .list_workers(None, None)
            .iter()
            .all(|record| record.lifecycle != WorkerLifecycle::Schedulable)
    );
    assert!(mirror.registered_worker_ids().is_empty());

    http.abort();
    selection.shutdown().await;
}
