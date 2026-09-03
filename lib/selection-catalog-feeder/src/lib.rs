// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kubernetes pods -> selection-service catalog.
//!
//! The feeder watches pods matching a label selector and mirrors the `Ready`
//! ones into a standalone selection service (or a runtime-free frontend's
//! embedded service) through the catalog HTTP API: `POST /workers` when a pod
//! becomes Ready, `POST /workers/{id}/heartbeat` on a lease interval, and
//! `DELETE /workers/{id}` when it leaves. It is the general form of the EPP's
//! `InferencePool` discovery: no gateway objects, one label selector, any port
//! layout, and any catalog consumer. Workers stay registered only while the
//! feeder heartbeats them, so a feeder crash expires its registrations.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Duration;

use anyhow::{Context, Result};
use dynamo_runtime::discovery::hash_pod_name;
use dynamo_sidecar_common::{CatalogClient, CatalogRegistration};
use k8s_openapi::api::core::v1::Pod;
use tokio_util::sync::CancellationToken;

/// How pods map to catalog registrations.
#[derive(Debug, Clone)]
pub struct FeederConfig {
    pub model_name: String,
    pub routing_group: String,
    /// Scheme for the worker endpoint (`http` for vLLM gRPC sidecars, `grpc` is
    /// also accepted by the direct engine transport).
    pub endpoint_scheme: String,
    /// Port the frontend dispatches to on each pod.
    pub port: u16,
    pub block_size: u32,
    pub data_parallel_size: u32,
    /// KV-event ZMQ PUB port of DP rank 0; rank `r` publishes on
    /// `kv_event_port + r * kv_event_port_stride`. `None` registers no KV
    /// event endpoints (load-based routing).
    pub kv_event_port: Option<u16>,
    pub kv_event_port_stride: u16,
    pub replay_port: Option<u16>,
    pub total_kv_blocks: Option<u64>,
    pub max_num_batched_tokens: Option<u64>,
    pub ttl: Duration,
}

/// `True` Ready condition and not terminating.
pub fn pod_is_ready(pod: &Pod) -> bool {
    if pod.metadata.deletion_timestamp.is_some() {
        return false;
    }
    pod.status
        .as_ref()
        .and_then(|status| status.conditions.as_ref())
        .is_some_and(|conditions| {
            conditions
                .iter()
                .any(|condition| condition.type_ == "Ready" && condition.status == "True")
        })
}

/// Equality-based label match (`k=v,k2=v2`).
pub fn pod_matches(pod: &Pod, selector: &BTreeMap<String, String>) -> bool {
    let Some(labels) = pod.metadata.labels.as_ref() else {
        return selector.is_empty();
    };
    selector
        .iter()
        .all(|(key, value)| labels.get(key) == Some(value))
}

/// Parse `k=v,k2=v2` into an equality selector.
pub fn parse_selector(selector: &str) -> Result<BTreeMap<String, String>> {
    let mut labels = BTreeMap::new();
    for pair in selector.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (key, value) = pair
            .split_once('=')
            .with_context(|| format!("selector term `{pair}` is not key=value"))?;
        labels.insert(key.trim().to_string(), value.trim().to_string());
    }
    if labels.is_empty() {
        anyhow::bail!("selector must name at least one label");
    }
    Ok(labels)
}

/// The registration a Ready pod produces, or `None` when it has no name or IP.
pub fn registration_for_pod(pod: &Pod, config: &FeederConfig) -> Option<CatalogRegistration> {
    let name = pod.metadata.name.as_deref()?;
    let ip = pod.status.as_ref()?.pod_ip.as_deref()?;
    if ip.is_empty() {
        return None;
    }
    let mut kv_events_endpoints = HashMap::new();
    if let Some(base) = config.kv_event_port {
        for rank in 0..config.data_parallel_size {
            let port = base as u32 + rank * config.kv_event_port_stride as u32;
            kv_events_endpoints.insert(rank, format!("tcp://{ip}:{port}"));
        }
    }
    Some(CatalogRegistration {
        worker_id: hash_pod_name(name),
        model_name: config.model_name.clone(),
        routing_group: config.routing_group.clone(),
        endpoint: format!("{}://{ip}:{}", config.endpoint_scheme, config.port),
        block_size: config.block_size,
        data_parallel_start_rank: 0,
        data_parallel_size: config.data_parallel_size.max(1),
        max_num_batched_tokens: config.max_num_batched_tokens,
        total_kv_blocks: config.total_kv_blocks,
        kv_events_endpoints,
        replay_endpoint: config.replay_port.map(|port| format!("tcp://{ip}:{port}")),
        router_hint_worker_type: None,
        router_hint_source_control_endpoints: HashMap::new(),
        ttl_secs: config.ttl.as_secs_f64(),
    })
}

/// Desired catalog state derived from a pod snapshot: one registration per
/// Ready, selected pod.
pub fn desired_registrations<'a>(
    pods: impl IntoIterator<Item = &'a Pod>,
    selector: &BTreeMap<String, String>,
    config: &FeederConfig,
) -> HashMap<u64, CatalogRegistration> {
    pods.into_iter()
        .filter(|pod| pod_matches(pod, selector) && pod_is_ready(pod))
        .filter_map(|pod| registration_for_pod(pod, config))
        .map(|registration| (registration.worker_id, registration))
        .collect()
}

/// Mirrors a desired set into the catalog and keeps leases alive.
pub struct CatalogMirror {
    client: CatalogClient,
    /// Registrations the catalog currently holds, by worker id, with the
    /// endpoint they were registered under (re-register on change).
    registered: HashMap<u64, String>,
}

impl CatalogMirror {
    pub fn new(client: CatalogClient) -> Self {
        Self {
            client,
            registered: HashMap::new(),
        }
    }

    pub fn registered_worker_ids(&self) -> HashSet<u64> {
        self.registered.keys().copied().collect()
    }

    /// Register new or changed workers and deregister absent ones. Failures
    /// are logged and retried on the next reconcile; the lease expiry bounds
    /// the damage of a missed deregistration.
    pub async fn reconcile(&mut self, desired: &HashMap<u64, CatalogRegistration>) {
        for (worker_id, registration) in desired {
            if self.registered.get(worker_id) == Some(&registration.endpoint) {
                continue;
            }
            match self.client.register(registration).await {
                Ok(()) => {
                    tracing::info!(
                        worker_id,
                        endpoint = %registration.endpoint,
                        "registered worker in catalog"
                    );
                    self.registered
                        .insert(*worker_id, registration.endpoint.clone());
                }
                Err(error) => {
                    tracing::warn!(worker_id, %error, "catalog registration failed; retrying on next change");
                }
            }
        }
        let stale: Vec<u64> = self
            .registered
            .keys()
            .copied()
            .filter(|worker_id| !desired.contains_key(worker_id))
            .collect();
        for worker_id in stale {
            match self.client.deregister(worker_id).await {
                Ok(()) => {
                    tracing::info!(worker_id, "deregistered worker from catalog");
                    self.registered.remove(&worker_id);
                }
                Err(error) => {
                    tracing::warn!(worker_id, %error, "catalog deregistration failed; lease will expire");
                    self.registered.remove(&worker_id);
                }
            }
        }
    }

    /// Renew every held lease.
    pub async fn heartbeat_all(&mut self) {
        let ids: Vec<u64> = self.registered.keys().copied().collect();
        for worker_id in ids {
            if let Err(error) = self.client.heartbeat(worker_id).await {
                tracing::warn!(worker_id, %error, "catalog heartbeat failed; re-registering on next reconcile");
                self.registered.remove(&worker_id);
            }
        }
    }

    /// Deregister everything (shutdown).
    pub async fn clear(&mut self) {
        let empty = HashMap::new();
        self.reconcile(&empty).await;
    }
}

/// Watch pods in `namespace` matching `selector` and mirror them into the
/// catalog until `cancel` fires. Uses a reflector store so every reconcile
/// works from a complete snapshot.
pub async fn run(
    client: kube::Client,
    namespace: &str,
    selector: BTreeMap<String, String>,
    config: FeederConfig,
    mut mirror: CatalogMirror,
    cancel: CancellationToken,
) -> Result<()> {
    use futures::StreamExt;
    use kube::runtime::{WatchStreamExt, reflector, watcher};

    let label_selector = selector
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(",");
    let pods: kube::Api<Pod> = kube::Api::namespaced(client, namespace);
    let writer = reflector::store::Writer::default();
    let store = writer.as_reader();
    let stream = reflector::reflector(
        writer,
        watcher(pods, watcher::Config::default().labels(&label_selector)).default_backoff(),
    );
    tokio::pin!(stream);

    let heartbeat_every = config.ttl / 3;
    let mut heartbeat = tokio::time::interval(heartbeat_every.max(Duration::from_secs(1)));
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut synced = false;
    tracing::info!(
        namespace,
        selector = %label_selector,
        catalog = %mirror.client.base(),
        ttl_secs = config.ttl.as_secs_f64(),
        "feeding Ready pods into the selection catalog"
    );

    loop {
        tokio::select! {
            _ = cancel.cancelled() => break,
            _ = heartbeat.tick() => mirror.heartbeat_all().await,
            event = stream.next() => {
                let Some(event) = event else {
                    anyhow::bail!("pod watch stream ended");
                };
                match event {
                    Ok(watcher::Event::InitDone) => synced = true,
                    Ok(watcher::Event::Init | watcher::Event::InitApply(_)) => continue,
                    Ok(watcher::Event::Apply(_) | watcher::Event::Delete(_)) => {}
                    Err(error) => {
                        tracing::warn!(%error, "pod watch error; retrying");
                        continue;
                    }
                }
                if !synced {
                    continue;
                }
                let snapshot = store.state();
                let desired = desired_registrations(
                    snapshot.iter().map(|pod| pod.as_ref()),
                    &selector,
                    &config,
                );
                mirror.reconcile(&desired).await;
            }
        }
    }
    mirror.clear().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use k8s_openapi::api::core::v1::{PodCondition, PodStatus};
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;

    fn config() -> FeederConfig {
        FeederConfig {
            model_name: "m".into(),
            routing_group: "default".into(),
            endpoint_scheme: "http".into(),
            port: 50051,
            block_size: 16,
            data_parallel_size: 2,
            kv_event_port: Some(5557),
            kv_event_port_stride: 10,
            replay_port: Some(5600),
            total_kv_blocks: Some(1000),
            max_num_batched_tokens: Some(8192),
            ttl: Duration::from_secs(30),
        }
    }

    fn pod(name: &str, ip: Option<&str>, ready: bool, labels: &[(&str, &str)]) -> Pod {
        Pod {
            metadata: ObjectMeta {
                name: Some(name.into()),
                labels: Some(
                    labels
                        .iter()
                        .map(|(k, v)| (k.to_string(), v.to_string()))
                        .collect(),
                ),
                ..Default::default()
            },
            status: Some(PodStatus {
                pod_ip: ip.map(str::to_string),
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

    #[test]
    fn registration_mirrors_pod_layout() {
        let registration =
            registration_for_pod(&pod("w-0", Some("10.0.0.5"), true, &[]), &config()).unwrap();
        assert_eq!(registration.worker_id, hash_pod_name("w-0"));
        assert_eq!(registration.endpoint, "http://10.0.0.5:50051");
        assert_eq!(registration.data_parallel_size, 2);
        assert_eq!(registration.kv_events_endpoints[&0], "tcp://10.0.0.5:5557");
        assert_eq!(registration.kv_events_endpoints[&1], "tcp://10.0.0.5:5567");
        assert_eq!(
            registration.replay_endpoint.as_deref(),
            Some("tcp://10.0.0.5:5600")
        );
        assert_eq!(registration.ttl_secs, 30.0);
        assert!(registration_for_pod(&pod("w-1", None, true, &[]), &config()).is_none());
    }

    #[test]
    fn desired_set_keeps_only_ready_selected_pods_with_ips() {
        let selector = parse_selector("app=vllm, role=decode").unwrap();
        let pods = [
            pod(
                "a",
                Some("10.0.0.1"),
                true,
                &[("app", "vllm"), ("role", "decode")],
            ),
            pod(
                "b",
                Some("10.0.0.2"),
                false,
                &[("app", "vllm"), ("role", "decode")],
            ),
            pod(
                "c",
                Some("10.0.0.3"),
                true,
                &[("app", "vllm"), ("role", "prefill")],
            ),
            pod("d", None, true, &[("app", "vllm"), ("role", "decode")]),
        ];
        let desired = desired_registrations(pods.iter(), &selector, &config());
        assert_eq!(desired.len(), 1);
        assert!(desired.contains_key(&hash_pod_name("a")));
        assert!(parse_selector("").is_err());
        assert!(parse_selector("novalue").is_err());
    }
}
