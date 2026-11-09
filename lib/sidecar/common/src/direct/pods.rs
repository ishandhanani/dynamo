// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct engines from Kubernetes pods selected by label.
//!
//! A stock engine pod runs no Dynamo code and registers nothing, so the
//! frontend watches the pods matching a label selector and keeps one engine
//! per serving pod in a [`DirectMembership`]:
//!
//! - a pod serves once it is `Ready` and has no deletion timestamp;
//! - its engine URL is `http://<pod IP>:<port>`, the port from the pod's
//!   [`PORT_ANNOTATION`] or else the selector's pool-wide port;
//! - its role comes from the selector, so prefill and serving pools are two
//!   selectors over one membership.
//!
//! A pod that stops serving, is deleted, or is missing from a relist is
//! withdrawn. Each serving pod is probed by its own task, retrying with
//! backoff, so an engine that is not answering yet never holds back the
//! others.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use k8s_openapi::api::core::v1::Pod;
use kube::runtime::{WatchStreamExt, watcher};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::{DirectBackend, DirectFleet, DirectMembership, RoleHint};

/// Per-pod override of the selector's pool-wide gRPC port.
pub const PORT_ANNOTATION: &str = "nvidia.com/dynamo-engine-grpc-port";

/// Longest wait between attempts to reach a pod's engine.
const MAX_RETRY: Duration = Duration::from_secs(30);

/// Which pods are engines of one role, and where their gRPC servers listen.
#[derive(Clone, Debug)]
pub struct PodSelector {
    /// Namespace to watch.
    pub namespace: String,
    /// Kubernetes label selector, for example `app=sglang,tier=decode`.
    pub labels: String,
    /// Pool-wide gRPC port; a pod's [`PORT_ANNOTATION`] overrides it.
    pub port: Option<u16>,
    /// Role of every selected engine.
    pub role: RoleHint,
}

/// The engine URL `pod` serves right now, or `None` if it serves none.
pub fn pod_url(pod: &Pod, default_port: Option<u16>) -> Option<String> {
    if pod.metadata.deletion_timestamp.is_some() {
        return None;
    }
    let status = pod.status.as_ref()?;
    let ready = status
        .conditions
        .as_deref()?
        .iter()
        .any(|condition| condition.type_ == "Ready" && condition.status == "True");
    if !ready {
        return None;
    }
    let ip = status.pod_ip.as_deref().filter(|ip| !ip.is_empty())?;
    let port: u16 = match pod
        .metadata
        .annotations
        .as_ref()
        .and_then(|annotations| annotations.get(PORT_ANNOTATION))
    {
        Some(port) => port.parse().ok()?,
        None => default_port?,
    };
    let host = if ip.contains(':') {
        format!("[{ip}]")
    } else {
        ip.to_string()
    };
    Some(format!("http://{host}:{port}"))
}

/// The fleet a pod-driven frontend starts from, and the pods it came from.
pub struct FirstPods<B: DirectBackend> {
    pub fleet: DirectFleet<B>,
    pub serving_url: String,
    pub prefill_url: Option<String>,
}

/// Probe the first serving pod of each selector into a fleet, waiting until
/// one answers. The fleet seeds the model card and fixes whether the
/// deployment is disaggregated; [`PodEngines`] adds the remaining pods and
/// should [adopt](PodEngines::adopting) these.
///
/// Returns `None` if `cancel` fires first.
pub async fn probe_first_pods<B: DirectBackend>(
    backend: Arc<B>,
    client: &kube::Client,
    serving: &PodSelector,
    prefill: Option<&PodSelector>,
    cancel: &CancellationToken,
) -> anyhow::Result<Option<FirstPods<B>>> {
    let mut delay = Duration::from_secs(1);
    loop {
        let Some(serving_url) = first_serving_url(client, serving, cancel).await? else {
            return Ok(None);
        };
        let prefill_url = match prefill {
            Some(selector) => match first_serving_url(client, selector, cancel).await? {
                Some(url) => Some(url),
                None => return Ok(None),
            },
            None => None,
        };
        let prefill_urls: Vec<String> = prefill_url.iter().cloned().collect();
        match DirectFleet::probe(
            backend.clone(),
            std::slice::from_ref(&serving_url),
            &prefill_urls,
        )
        .await
        {
            Ok(fleet) => {
                return Ok(Some(FirstPods {
                    fleet,
                    serving_url,
                    prefill_url,
                }));
            }
            Err(error) => {
                tracing::warn!(%error, retry_in = ?delay, "first engine pods not answering");
                tokio::select! {
                    _ = cancel.cancelled() => return Ok(None),
                    _ = tokio::time::sleep(delay) => {}
                }
                delay = (delay * 2).min(MAX_RETRY);
            }
        }
    }
}

/// Watch `selector` until a pod serves; returns its URL, or `None` if
/// `cancel` fires first.
async fn first_serving_url(
    client: &kube::Client,
    selector: &PodSelector,
    cancel: &CancellationToken,
) -> anyhow::Result<Option<String>> {
    let api: kube::Api<Pod> = kube::Api::namespaced(client.clone(), &selector.namespace);
    let stream = watcher(api, watcher::Config::default().labels(&selector.labels))
        .default_backoff()
        .applied_objects();
    tokio::pin!(stream);
    loop {
        let pod = tokio::select! {
            _ = cancel.cancelled() => return Ok(None),
            pod = stream.next() => pod,
        };
        match pod {
            None => anyhow::bail!("engine pod watch for `{}` ended", selector.labels),
            Some(Err(error)) => tracing::warn!(%error, "engine pod watch"),
            Some(Ok(pod)) => {
                if let Some(url) = pod_url(&pod, selector.port) {
                    return Ok(Some(url));
                }
            }
        }
    }
}

/// Keeps a [`DirectMembership`] in step with the serving pods of one
/// selector.
pub struct PodEngines<B: DirectBackend> {
    membership: Arc<DirectMembership<B>>,
    selector: PodSelector,
    retry: Duration,
    pods: tokio::sync::Mutex<HashMap<String, Tracked>>,
    /// Engines added before the watch started, reconciled at the first relist.
    adopted: tokio::sync::Mutex<Vec<String>>,
}

struct Tracked {
    url: String,
    task: JoinHandle<()>,
}

impl<B: DirectBackend> PodEngines<B> {
    pub fn new(membership: Arc<DirectMembership<B>>, selector: PodSelector) -> Self {
        Self {
            membership,
            selector,
            retry: Duration::from_secs(1),
            pods: Default::default(),
            adopted: Default::default(),
        }
    }

    /// Engines of this selector already in the membership, such as the
    /// [`FirstPods`]. The first relist withdraws any that no pod serves.
    pub fn adopting(self, urls: impl IntoIterator<Item = String>) -> Self {
        Self {
            adopted: tokio::sync::Mutex::new(urls.into_iter().collect()),
            ..self
        }
    }

    /// First delay between attempts to reach a pod's engine; doubles to 30 s.
    pub fn with_retry(mut self, retry: Duration) -> Self {
        self.retry = retry;
        self
    }

    /// Watch the selected pods until `cancel` fires, then withdraw every
    /// engine this selector added.
    pub async fn watch(
        &self,
        client: kube::Client,
        cancel: CancellationToken,
    ) -> anyhow::Result<()> {
        let api: kube::Api<Pod> = kube::Api::namespaced(client, &self.selector.namespace);
        let config = watcher::Config::default().labels(&self.selector.labels);
        let stream = watcher(api, config).default_backoff();
        tokio::pin!(stream);
        tracing::info!(
            namespace = %self.selector.namespace,
            labels = %self.selector.labels,
            role = ?self.selector.role,
            "watching engine pods"
        );

        // A relist arrives as Init, InitApply per pod, InitDone.
        let mut relist: Option<Vec<Pod>> = None;
        loop {
            let event = tokio::select! {
                _ = cancel.cancelled() => break,
                event = stream.next() => event,
            };
            match event {
                None => break,
                Some(Err(error)) => tracing::warn!(%error, "engine pod watch"),
                Some(Ok(watcher::Event::Init)) => relist = Some(Vec::new()),
                Some(Ok(watcher::Event::InitApply(pod))) => {
                    relist.get_or_insert_with(Vec::new).push(pod)
                }
                Some(Ok(watcher::Event::InitDone)) => {
                    let pods = relist.take().unwrap_or_default();
                    tracing::info!(pods = pods.len(), "engine pod relist");
                    self.resync(&pods).await;
                }
                Some(Ok(watcher::Event::Apply(pod))) => self.apply(&pod).await,
                Some(Ok(watcher::Event::Delete(pod))) => self.delete(&pod).await,
            }
        }
        self.resync(&[]).await;
        Ok(())
    }

    /// A pod was added or changed.
    pub async fn apply(&self, pod: &Pod) {
        let Some(name) = pod.metadata.name.clone() else {
            return;
        };
        let url = pod_url(pod, self.selector.port);
        let mut pods = self.pods.lock().await;
        if pods.get(&name).map(|tracked| &tracked.url) == url.as_ref() {
            return;
        }
        if let Some(tracked) = pods.remove(&name) {
            self.retire(&name, tracked).await;
        }
        if let Some(url) = url {
            let task = self.host(name.clone(), url.clone());
            pods.insert(name, Tracked { url, task });
        }
    }

    /// A pod was deleted.
    pub async fn delete(&self, pod: &Pod) {
        let Some(name) = pod.metadata.name.as_deref() else {
            return;
        };
        let tracked = self.pods.lock().await.remove(name);
        if let Some(tracked) = tracked {
            self.retire(name, tracked).await;
        }
    }

    /// A full relist: these are all the selected pods.
    pub async fn resync(&self, pods: &[Pod]) {
        let listed: HashSet<&str> = pods
            .iter()
            .filter_map(|pod| pod.metadata.name.as_deref())
            .collect();
        let gone: Vec<(String, Tracked)> = {
            let mut tracked = self.pods.lock().await;
            let names: Vec<String> = tracked
                .keys()
                .filter(|name| !listed.contains(name.as_str()))
                .cloned()
                .collect();
            names
                .into_iter()
                .filter_map(|name| tracked.remove(&name).map(|t| (name, t)))
                .collect()
        };
        for (name, tracked) in gone {
            self.retire(&name, tracked).await;
        }
        let adopted = std::mem::take(&mut *self.adopted.lock().await);
        if !adopted.is_empty() {
            let served: HashSet<String> = pods
                .iter()
                .filter_map(|pod| pod_url(pod, self.selector.port))
                .collect();
            for url in adopted.iter().filter(|url| !served.contains(*url)) {
                if self.membership.remove(url) {
                    tracing::info!(%url, "engine pod withdrawn");
                }
            }
        }
        for pod in pods {
            self.apply(pod).await;
        }
    }

    /// Probe and add `url`, retrying until it serves.
    fn host(&self, pod: String, url: String) -> JoinHandle<()> {
        let membership = self.membership.clone();
        let role = self.selector.role;
        let mut delay = self.retry;
        tokio::spawn(async move {
            loop {
                match membership.add(&url, role).await {
                    Ok(instance_id) => {
                        tracing::info!(
                            %pod,
                            %url,
                            instance_id = format_args!("{instance_id:x}"),
                            "engine pod serving"
                        );
                        return;
                    }
                    Err(error) => {
                        tracing::warn!(%pod, %url, %error, retry_in = ?delay, "engine pod not reachable");
                        tokio::time::sleep(delay).await;
                        delay = (delay * 2).min(MAX_RETRY);
                    }
                }
            }
        })
    }

    /// Stop trying to add a pod's engine and withdraw it if added.
    async fn retire(&self, pod: &str, tracked: Tracked) {
        tracked.task.abort();
        let _ = tracked.task.await;
        if self.membership.remove(&tracked.url) {
            tracing::info!(%pod, url = %tracked.url, "engine pod withdrawn");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct::tests::{FakeBackend, fleet};
    use serde_json::json;

    fn pod(value: serde_json::Value) -> Pod {
        serde_json::from_value(value).unwrap()
    }

    fn serving_pod(name: &str, ip: &str, annotations: serde_json::Value) -> Pod {
        pod(json!({
            "metadata": { "name": name, "annotations": annotations },
            "status": {
                "podIP": ip,
                "conditions": [{ "type": "Ready", "status": "True" }],
            },
        }))
    }

    fn not_ready_pod(name: &str, ip: &str) -> Pod {
        pod(json!({
            "metadata": { "name": name },
            "status": {
                "podIP": ip,
                "conditions": [{ "type": "Ready", "status": "False" }],
            },
        }))
    }

    #[test]
    fn ready_pod_serves_at_its_ip_and_pool_port() {
        let pod = serving_pod("engine-0", "10.1.2.3", json!({}));
        assert_eq!(
            pod_url(&pod, Some(30001)).as_deref(),
            Some("http://10.1.2.3:30001")
        );
    }

    #[test]
    fn port_annotation_overrides_the_pool_port() {
        let pod = serving_pod("engine-0", "10.1.2.3", json!({ PORT_ANNOTATION: "30011" }));
        assert_eq!(
            pod_url(&pod, Some(30001)).as_deref(),
            Some("http://10.1.2.3:30011")
        );
        assert_eq!(
            pod_url(&pod, None).as_deref(),
            Some("http://10.1.2.3:30011")
        );
    }

    #[test]
    fn ipv6_pod_address_is_bracketed() {
        let pod = serving_pod("engine-0", "fd00::7", json!({}));
        assert_eq!(
            pod_url(&pod, Some(30001)).as_deref(),
            Some("http://[fd00::7]:30001")
        );
    }

    #[test]
    fn pod_without_a_port_serves_nothing() {
        assert_eq!(
            pod_url(&serving_pod("engine-0", "10.1.2.3", json!({})), None),
            None
        );
        let bad = serving_pod("engine-0", "10.1.2.3", json!({ PORT_ANNOTATION: "grpc" }));
        assert_eq!(pod_url(&bad, None), None);
    }

    #[test]
    fn pod_that_is_not_ready_or_has_no_ip_serves_nothing() {
        assert_eq!(
            pod_url(&not_ready_pod("engine-0", "10.1.2.3"), Some(30001)),
            None
        );
        let no_conditions = pod(json!({
            "metadata": { "name": "engine-0" },
            "status": { "podIP": "10.1.2.3" },
        }));
        assert_eq!(pod_url(&no_conditions, Some(30001)), None);
        let no_ip = pod(json!({
            "metadata": { "name": "engine-0" },
            "status": { "conditions": [{ "type": "Ready", "status": "True" }] },
        }));
        assert_eq!(pod_url(&no_ip, Some(30001)), None);
    }

    #[test]
    fn terminating_pod_serves_nothing() {
        let terminating = pod(json!({
            "metadata": { "name": "engine-0", "deletionTimestamp": "2026-11-09T00:00:00Z" },
            "status": {
                "podIP": "10.1.2.3",
                "conditions": [{ "type": "Ready", "status": "True" }],
            },
        }));
        assert_eq!(pod_url(&terminating, Some(30001)), None);
    }

    fn selector(role: RoleHint) -> PodSelector {
        PodSelector {
            namespace: "default".to_string(),
            labels: "app=engine".to_string(),
            port: Some(30001),
            role,
        }
    }

    async fn settle<B: DirectBackend>(membership: &DirectMembership<B>, expected: usize) {
        for _ in 0..200 {
            if membership.instance_ids().len() == expected {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!(
            "membership has {} engines, expected {expected}",
            membership.instance_ids().len()
        );
    }

    #[tokio::test]
    async fn pods_join_when_ready_and_leave_when_not() {
        let backend = FakeBackend::new();
        let (_config, membership) = fleet(&backend, &["http://10.0.0.1:30001"], &[]).await;
        let membership = Arc::new(membership);
        let pods = PodEngines::new(membership.clone(), selector(RoleHint::Serving))
            .with_retry(Duration::from_millis(5));

        pods.apply(&serving_pod("engine-1", "10.0.0.2", json!({})))
            .await;
        settle(&membership, 2).await;

        pods.apply(&not_ready_pod("engine-1", "10.0.0.2")).await;
        settle(&membership, 1).await;

        pods.apply(&serving_pod("engine-1", "10.0.0.2", json!({})))
            .await;
        settle(&membership, 2).await;
        pods.delete(&serving_pod("engine-1", "10.0.0.2", json!({})))
            .await;
        settle(&membership, 1).await;
    }

    #[tokio::test]
    async fn an_engine_that_is_not_answering_yet_is_retried() {
        let backend = FakeBackend::new();
        let (_config, membership) = fleet(&backend, &["http://10.0.0.1:30001"], &[]).await;
        let membership = Arc::new(membership);
        backend.fail_next("http://10.0.0.2:30001", 3);
        let pods = PodEngines::new(membership.clone(), selector(RoleHint::Serving))
            .with_retry(Duration::from_millis(5));

        pods.apply(&serving_pod("engine-1", "10.0.0.2", json!({})))
            .await;
        pods.apply(&serving_pod("engine-2", "10.0.0.3", json!({})))
            .await;
        settle(&membership, 3).await;
        assert_eq!(backend.probes("http://10.0.0.2:30001"), 4);
    }

    #[tokio::test]
    async fn first_relist_withdraws_an_adopted_engine_no_pod_serves() {
        let backend = FakeBackend::new();
        let first = "http://10.0.0.1:30001";
        let (_config, membership) = fleet(&backend, &[first], &[]).await;
        let membership = Arc::new(membership);
        let pods = PodEngines::new(membership.clone(), selector(RoleHint::Serving))
            .with_retry(Duration::from_millis(5))
            .adopting([first.to_string()]);

        pods.resync(&[serving_pod("engine-1", "10.0.0.2", json!({}))])
            .await;
        settle(&membership, 1).await;
        assert!(!membership.contains(crate::direct::instance_id_for_url(first)));
    }

    #[tokio::test]
    async fn relist_withdraws_pods_it_no_longer_lists() {
        let backend = FakeBackend::new();
        let (_config, membership) = fleet(&backend, &["http://10.0.0.1:30001"], &[]).await;
        let membership = Arc::new(membership);
        let pods = PodEngines::new(membership.clone(), selector(RoleHint::Serving))
            .with_retry(Duration::from_millis(5));

        pods.resync(&[
            serving_pod("engine-1", "10.0.0.2", json!({})),
            serving_pod("engine-2", "10.0.0.3", json!({})),
        ])
        .await;
        settle(&membership, 3).await;

        pods.resync(&[serving_pod("engine-2", "10.0.0.3", json!({}))])
            .await;
        settle(&membership, 2).await;
        assert!(!membership.contains(crate::direct::instance_id_for_url("http://10.0.0.2:30001")));
    }
}
