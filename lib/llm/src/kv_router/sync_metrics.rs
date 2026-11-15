// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics of the code that keeps the router's view of workers up to date.
//!
//! - [`WorkerLoadMetrics`]: Per-worker active decode blocks, prefill tokens, and
//!   active requests by request phase.
//!   Registered on the frontend's own `prometheus::Registry` (default port 8000).
//!   Populated by `KvWorkerMonitor` in the frontend when receiving ActiveLoad events.
//!   - Frontend (aggregated and disaggregated): available on default port 8000
//!   - Standalone router (`python -m dynamo.router`): not created (frontend-only)
//!
//! - `RouterWorkerStatusMetrics`: which workers the router has registered and how many lack a
//!   usable KV event source. Registered on the DRT `MetricsRegistry` hierarchy.
//!
//! - `KvZmqIngressMetrics`, `ActiveSequenceZmqIngressMetrics`: direct-ZMQ ingress of KV events
//!   and of active-sequence replica events.
//!
//! - [`RemoteIndexerMetrics`]: failed queries and writes against a remote indexer.
//!
//! The request path's metrics are in [`super::metrics`].

use std::sync::{Arc, LazyLock, OnceLock};

use dynamo_runtime::component::Component;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::metrics::prometheus_names::{frontend_service, labels, name_prefix, router};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_slot_tracker::host::LocalWorkerLoad;
use prometheus::{IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Opts};

mod worker_registered;
use worker_registered::RouterWorkerRegistered;
pub(crate) use worker_registered::{RegistrationAvailabilityProvider, RouterWorkerRegistration};

pub(crate) const ROUTER_WORKER_ID_LABEL: &str = "router_worker_id";
const TARGET_NAMESPACE_LABEL: &str = "target_namespace";
const TARGET_COMPONENT_LABEL: &str = "target_component";
const TARGET_ENDPOINT_LABEL: &str = "target_endpoint";

// ---------------------------------------------------------------------------
// ZMQ KV ingress metrics
// ---------------------------------------------------------------------------

pub(crate) struct KvZmqIngressMetrics {
    sources: IntGaugeVec,
    batches_total: IntCounter,
    lifecycle_total: IntCounterVec,
}

static KV_ZMQ_INGRESS_METRICS: OnceLock<Arc<KvZmqIngressMetrics>> = OnceLock::new();

impl KvZmqIngressMetrics {
    pub(crate) fn from_component(component: &Component) -> Arc<Self> {
        KV_ZMQ_INGRESS_METRICS
            .get_or_init(|| {
                let metrics = component.metrics();
                let sources = metrics
                    .create_intgaugevec(
                        "router_kv_zmq_ingress_sources",
                        "Number of ZMQ KV ingress sources by lifecycle state",
                        &["state"],
                        &[],
                    )
                    .expect("failed to create router_kv_zmq_ingress_sources gauge");
                let batches_total = metrics
                    .create_intcounter(
                        "router_kv_zmq_ingress_batches_total",
                        "Total ZMQ KV ingress batches handed to WorkerQueryClient",
                        &[],
                    )
                    .expect("failed to create router_kv_zmq_ingress_batches_total counter");
                let lifecycle_total = metrics
                    .create_intcountervec(
                        "router_kv_zmq_ingress_lifecycle_total",
                        "Total ZMQ KV ingress lifecycle transitions and errors",
                        &["action"],
                        &[],
                    )
                    .expect("failed to create router_kv_zmq_ingress_lifecycle_total counter");
                Arc::new(Self {
                    sources,
                    batches_total,
                    lifecycle_total,
                })
            })
            .clone()
    }

    pub(crate) fn increment_sources(&self, state: &'static str) {
        self.sources.with_label_values(&[state]).inc();
    }

    pub(crate) fn decrement_sources(&self, state: &'static str) {
        self.sources.with_label_values(&[state]).dec();
    }

    pub(crate) fn increment_batch(&self) {
        self.batches_total.inc();
    }

    pub(crate) fn increment_lifecycle(&self, action: &'static str) {
        self.lifecycle_total.with_label_values(&[action]).inc();
    }

    pub(crate) fn increment_lifecycle_by(&self, action: &'static str, value: u64) {
        self.lifecycle_total
            .with_label_values(&[action])
            .inc_by(value);
    }

    #[cfg(test)]
    pub(crate) fn lifecycle_count(&self, action: &'static str) -> u64 {
        self.lifecycle_total.with_label_values(&[action]).get()
    }
}

// ---------------------------------------------------------------------------
// Direct-ZMQ active-sequence ingress metrics
// ---------------------------------------------------------------------------

pub(crate) struct ActiveSequenceZmqIngressMetrics {
    tracked_sources: IntGauge,
    sequence_gaps_total: IntCounter,
    out_of_order_total: IntCounter,
    reconnects_total: IntCounter,
    replacements_total: IntCounter,
    envelope_decode_errors_total: IntCounter,
    payload_decode_errors_total: IntCounter,
    identity_errors_total: IntCounter,
    forced_aborts_total: IntCounter,
    dropped_events_total: IntCounter,
}

static ACTIVE_SEQUENCE_ZMQ_INGRESS_METRICS: OnceLock<Arc<ActiveSequenceZmqIngressMetrics>> =
    OnceLock::new();

impl ActiveSequenceZmqIngressMetrics {
    pub(crate) fn from_component(component: &Component) -> Arc<Self> {
        ACTIVE_SEQUENCE_ZMQ_INGRESS_METRICS
            .get_or_init(|| {
                let metrics = component.metrics();
                let counter = |name, help| {
                    metrics
                        .create_intcounter(name, help, &[])
                        .unwrap_or_else(|error| panic!("failed to create {name}: {error}"))
                };
                Arc::new(Self {
                    tracked_sources: metrics
                        .create_intgauge(
                            "router_active_sequence_zmq_ingress_sources",
                            "Number of tracked direct-ZMQ active-sequence sources",
                            &[],
                        )
                        .expect("failed to create router_active_sequence_zmq_ingress_sources"),
                    sequence_gaps_total: counter(
                        "router_active_sequence_zmq_ingress_sequence_gaps_total",
                        "Missing direct-ZMQ active-sequence envelopes inferred from sequence gaps",
                    ),
                    out_of_order_total: counter(
                        "router_active_sequence_zmq_ingress_out_of_order_total",
                        "Non-increasing direct-ZMQ active-sequence envelope sequences",
                    ),
                    reconnects_total: counter(
                        "router_active_sequence_zmq_ingress_reconnects_total",
                        "Direct-ZMQ active-sequence source reconnect attempts",
                    ),
                    replacements_total: counter(
                        "router_active_sequence_zmq_ingress_replacements_total",
                        "Direct-ZMQ active-sequence source task replacements",
                    ),
                    envelope_decode_errors_total: counter(
                        "router_active_sequence_zmq_ingress_envelope_decode_errors_total",
                        "Direct-ZMQ active-sequence envelope decode errors",
                    ),
                    payload_decode_errors_total: counter(
                        "router_active_sequence_zmq_ingress_payload_decode_errors_total",
                        "Direct-ZMQ active-sequence payload decode errors",
                    ),
                    identity_errors_total: counter(
                        "router_active_sequence_zmq_ingress_identity_errors_total",
                        "Direct-ZMQ active-sequence frame and envelope identity mismatches",
                    ),
                    forced_aborts_total: counter(
                        "router_active_sequence_zmq_ingress_forced_aborts_total",
                        "Direct-ZMQ active-sequence source tasks aborted after join timeout",
                    ),
                    dropped_events_total: counter(
                        "router_active_sequence_zmq_ingress_dropped_events_total",
                        "Direct-ZMQ active-sequence events dropped because the partition's inbound channel was full or closed",
                    ),
                })
            })
            .clone()
    }

    pub(crate) fn source_started(&self) {
        self.tracked_sources.inc();
    }

    pub(crate) fn source_stopped(&self) {
        self.tracked_sources.dec();
    }

    pub(crate) fn record_gap(&self, missing: u64) {
        self.sequence_gaps_total.inc_by(missing);
    }

    pub(crate) fn record_out_of_order(&self) {
        self.out_of_order_total.inc();
    }

    pub(crate) fn record_reconnect(&self) {
        self.reconnects_total.inc();
    }

    pub(crate) fn record_replacement(&self) {
        self.replacements_total.inc();
    }

    pub(crate) fn record_envelope_decode_error(&self) {
        self.envelope_decode_errors_total.inc();
    }

    pub(crate) fn record_payload_decode_error(&self) {
        self.payload_decode_errors_total.inc();
    }

    pub(crate) fn record_identity_error(&self) {
        self.identity_errors_total.inc();
    }

    pub(crate) fn record_forced_abort(&self) {
        self.forced_aborts_total.inc();
    }

    pub(crate) fn record_dropped_event(&self) {
        self.dropped_events_total.inc();
    }
}

// ---------------------------------------------------------------------------
// Router worker status metrics (component-scoped gauges)
// ---------------------------------------------------------------------------

/// Component-scoped router gauges for worker discovery.
pub(crate) struct RouterWorkerStatusMetrics {
    registered: RouterWorkerRegistered,
    pub kv_event_source_mismatch_workers: IntGaugeVec,
}

static ROUTER_WORKER_STATUS_METRICS: OnceLock<Arc<RouterWorkerStatusMetrics>> = OnceLock::new();

impl RouterWorkerStatusMetrics {
    /// Create component-scoped gauges for standalone router observability.
    ///
    /// The `MetricsHierarchy` injects labels such as `dynamo_namespace` and
    /// `dynamo_component`. It reserves `worker_id` for the metric producer, so
    /// the backend worker ID discovered by the router uses `router_worker_id`.
    pub fn from_component(component: &Component) -> Arc<Self> {
        ROUTER_WORKER_STATUS_METRICS
            .get_or_init(|| {
                let metrics = component.metrics();
                let registered = dynamo_runtime::metrics::create_metric(
                        component,
                        router::WORKER_REGISTERED,
                        "Whether the router currently has this worker/dp_rank registered (1 = registered)",
                        &[],
                        None,
                        None,
                    )
                    .expect("failed to create router_worker_registered gauge");
                let kv_event_source_mismatch_workers = metrics
                    .create_intgaugevec(
                        router::KV_EVENT_SOURCE_MISMATCH_WORKERS,
                        "Number of serving workers with missing or ambiguous KV sources, explicitly disabled KV event publishing, or without an expected RecoveryTarget",
                        &[
                            labels::MODEL,
                            labels::WORKER_TYPE,
                            TARGET_NAMESPACE_LABEL,
                            TARGET_COMPONENT_LABEL,
                            TARGET_ENDPOINT_LABEL,
                        ],
                        &[],
                    )
                    .expect("failed to create router_kv_event_source_mismatch_workers gauge");

                Arc::new(Self {
                    registered,
                    kv_event_source_mismatch_workers,
                })
            })
            .clone()
    }

    pub fn watch_workers(
        &self,
        workers: crate::discovery::RuntimeConfigWatch,
        worker_type: &'static str,
        cancellation: tokio_util::sync::CancellationToken,
        available_workers: Option<RegistrationAvailabilityProvider>,
    ) -> Arc<RouterWorkerRegistration> {
        self.registered
            .watch(workers, worker_type, cancellation, available_workers)
    }

    pub fn set_kv_event_source_mismatch_workers(
        &self,
        model: &str,
        worker_type: &str,
        target_namespace: &str,
        target_component: &str,
        target_endpoint: &str,
        count: usize,
    ) {
        self.kv_event_source_mismatch_workers
            .with_label_values(&[
                model,
                worker_type,
                target_namespace,
                target_component,
                target_endpoint,
            ])
            .set(count as i64);
    }
}

// ---------------------------------------------------------------------------
// Worker load metrics (gauges)
// ---------------------------------------------------------------------------

/// Per-worker active load gauges, published by `ActiveSequencesMultiWorker`
/// and cleaned up by `KvWorkerMonitor` when workers disappear.
pub struct WorkerLoadMetrics {
    pub active_decode_blocks: IntGaugeVec,
    pub active_prefill_tokens: IntGaugeVec,
    /// Booked requests split by `request_phase`: `prefill` until the request is
    /// marked prefill-complete, `decode` after.
    pub active_requests: IntGaugeVec,
}

const REQUEST_PHASE_PREFILL: &str = "prefill";
const REQUEST_PHASE_DECODE: &str = "decode";

impl WorkerLoadMetrics {
    fn new() -> Self {
        let worker_labels = &[labels::WORKER_ID, labels::DP_RANK, labels::WORKER_TYPE];
        let gauge = |suffix: &str, help: &str, label_names: &[&str]| {
            IntGaugeVec::new(
                Opts::new(format!("{}_{}", name_prefix::FRONTEND, suffix), help),
                label_names,
            )
            .unwrap_or_else(|error| panic!("Failed to create {suffix} gauge: {error}"))
        };
        Self {
            active_decode_blocks: gauge(
                frontend_service::WORKER_ACTIVE_DECODE_BLOCKS,
                "Active KV cache decode blocks per worker",
                worker_labels,
            ),
            active_prefill_tokens: gauge(
                frontend_service::WORKER_ACTIVE_PREFILL_TOKENS,
                "Active prefill tokens queued per worker",
                worker_labels,
            ),
            active_requests: gauge(
                frontend_service::WORKER_ACTIVE_REQUESTS,
                "Active requests booked per worker, by request phase",
                &[
                    labels::WORKER_ID,
                    labels::DP_RANK,
                    labels::WORKER_TYPE,
                    labels::REQUEST_PHASE,
                ],
            ),
        }
    }

    #[cfg(test)]
    fn register(&self, registry: &prometheus::Registry) -> Result<(), prometheus::Error> {
        registry.register(Box::new(self.active_decode_blocks.clone()))?;
        registry.register(Box::new(self.active_prefill_tokens.clone()))?;
        registry.register(Box::new(self.active_requests.clone()))?;
        Ok(())
    }

    pub fn observe(&self, worker_id: u64, dp_rank: u32, worker_type: &str, load: LocalWorkerLoad) {
        let worker_id = worker_id.to_string();
        let dp_rank = dp_rank.to_string();
        let labels = [worker_id.as_str(), dp_rank.as_str(), worker_type];
        self.active_decode_blocks
            .with_label_values(&labels)
            .set(load.active_blocks as i64);
        self.active_prefill_tokens
            .with_label_values(&labels)
            .set(load.active_tokens as i64);
        let decode_requests = load.active_requests.saturating_sub(load.prefill_requests);
        for (phase, count) in [
            (REQUEST_PHASE_PREFILL, load.prefill_requests),
            (REQUEST_PHASE_DECODE, decode_requests),
        ] {
            self.active_requests
                .with_label_values(&[worker_id.as_str(), dp_rank.as_str(), worker_type, phase])
                .set(count as i64);
        }
    }

    /// Remove every load series for one worker/dp_rank.
    pub fn remove(&self, worker_id: u64, dp_rank: u32, worker_type: &str) {
        let worker_id = worker_id.to_string();
        let dp_rank = dp_rank.to_string();
        let labels = [worker_id.as_str(), dp_rank.as_str(), worker_type];
        let _ = self.active_decode_blocks.remove_label_values(&labels);
        let _ = self.active_prefill_tokens.remove_label_values(&labels);
        for phase in [REQUEST_PHASE_PREFILL, REQUEST_PHASE_DECODE] {
            let _ = self.active_requests.remove_label_values(&[
                worker_id.as_str(),
                dp_rank.as_str(),
                worker_type,
                phase,
            ]);
        }
    }
}

pub static WORKER_LOAD_METRICS: LazyLock<WorkerLoadMetrics> = LazyLock::new(WorkerLoadMetrics::new);

// ---------------------------------------------------------------------------
// Remote indexer metrics
// ---------------------------------------------------------------------------

pub struct RemoteIndexerMetrics {
    pub query_failures_total: prometheus::IntCounter,
    pub write_failures_total: prometheus::IntCounter,
}

static REMOTE_INDEXER_METRICS: OnceLock<Arc<RemoteIndexerMetrics>> = OnceLock::new();

impl RemoteIndexerMetrics {
    pub fn from_component(component: &Component) -> Arc<Self> {
        REMOTE_INDEXER_METRICS
            .get_or_init(|| {
                let instance_id = component.drt().discovery().instance_id();
                let router_id = instance_id.to_string();
                let extra_labels: &[(&str, &str)] = &[(labels::ROUTER_ID, &router_id)];

                let metrics = component.metrics();
                let query_failures_total = metrics
                    .create_intcounter(
                        router::REMOTE_INDEXER_QUERY_FAILURES_TOTAL,
                        "Total number of remote indexer overlap queries that failed",
                        extra_labels,
                    )
                    .expect("failed to create router_remote_indexer_query_failures_total");
                let write_failures_total = metrics
                    .create_intcounter(
                        router::REMOTE_INDEXER_WRITE_FAILURES_TOTAL,
                        "Total number of remote indexer routing-decision writes that failed",
                        extra_labels,
                    )
                    .expect("failed to create router_remote_indexer_write_failures_total");

                Arc::new(Self {
                    query_failures_total,
                    write_failures_total,
                })
            })
            .clone()
    }

    pub fn increment_query_failures(&self) {
        self.query_failures_total.inc();
    }

    pub fn increment_write_failures(&self) {
        self.write_failures_total.inc();
    }
}

#[cfg(test)]
pub(crate) mod test_fixtures {
    use prometheus::{Encoder, TextEncoder};

    pub(crate) fn gather_pef(registry: &prometheus::Registry) -> String {
        let encoder = TextEncoder::new();
        let mut buffer = Vec::new();
        encoder.encode(&registry.gather(), &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::test_fixtures::gather_pef;
    use super::*;
    use dynamo_runtime::metrics::PrometheusMetric;

    #[test]
    fn test_worker_load_metrics_pef() {
        let registry = prometheus::Registry::new();
        let metrics = WorkerLoadMetrics::new();
        metrics.register(&registry).unwrap();

        metrics.observe(
            123,
            0,
            "decode",
            LocalWorkerLoad {
                active_blocks: 42,
                active_tokens: 100,
                active_requests: 5,
                prefill_requests: 2,
            },
        );

        let output = gather_pef(&registry);
        let expected = "\
# HELP dynamo_frontend_worker_active_decode_blocks Active KV cache decode blocks per worker
# TYPE dynamo_frontend_worker_active_decode_blocks gauge
dynamo_frontend_worker_active_decode_blocks{dp_rank=\"0\",worker_id=\"123\",worker_type=\"decode\"} 42
# HELP dynamo_frontend_worker_active_prefill_tokens Active prefill tokens queued per worker
# TYPE dynamo_frontend_worker_active_prefill_tokens gauge
dynamo_frontend_worker_active_prefill_tokens{dp_rank=\"0\",worker_id=\"123\",worker_type=\"decode\"} 100
# HELP dynamo_frontend_worker_active_requests Active requests booked per worker, by request phase
# TYPE dynamo_frontend_worker_active_requests gauge
dynamo_frontend_worker_active_requests{dp_rank=\"0\",request_phase=\"decode\",worker_id=\"123\",worker_type=\"decode\"} 3
dynamo_frontend_worker_active_requests{dp_rank=\"0\",request_phase=\"prefill\",worker_id=\"123\",worker_type=\"decode\"} 2
";
        assert_eq!(
            output, expected,
            "\nActual PEF:\n{output}\nExpected PEF:\n{expected}"
        );

        metrics.remove(123, 0, "decode");
        assert_eq!(gather_pef(&registry), "");
    }

    #[test]
    fn test_router_worker_status_metrics_pef() {
        let registry = prometheus::Registry::new();
        let metrics = RouterWorkerStatusMetrics {
            registered: RouterWorkerRegistered::with_opts(
                Opts::new(
                    format!(
                        "{}_{}",
                        name_prefix::COMPONENT,
                        router::WORKER_REGISTERED
                    ),
                    "Whether the router currently has this worker/dp_rank registered (1 = registered)",
                ),
            )
            .unwrap(),
            kv_event_source_mismatch_workers: IntGaugeVec::new(
                Opts::new(
                    format!(
                        "{}_{}",
                        name_prefix::COMPONENT,
                        router::KV_EVENT_SOURCE_MISMATCH_WORKERS
                    ),
                    "Number of workers expected to publish KV events but missing worker-local KV indexer query endpoints",
                ),
                &[
                    labels::MODEL,
                    labels::WORKER_TYPE,
                    TARGET_NAMESPACE_LABEL,
                    TARGET_COMPONENT_LABEL,
                    TARGET_ENDPOINT_LABEL,
                ],
            )
            .unwrap(),
        };
        registry
            .register(Box::new(metrics.registered.clone()))
            .unwrap();
        registry
            .register(Box::new(metrics.kv_event_source_mismatch_workers.clone()))
            .unwrap();

        let (workers_tx, workers_rx) =
            tokio::sync::watch::channel(std::collections::HashMap::from([(
                123,
                dynamo_model_card::local_model::runtime_config::ModelRuntimeConfig::default(),
            )]));
        let _registration = metrics.watch_workers(
            workers_rx,
            "decode",
            tokio_util::sync::CancellationToken::new(),
            None,
        );
        metrics.set_kv_event_source_mismatch_workers(
            "model-a", "decode", "ns-a", "decode", "generate", 2,
        );
        metrics.set_kv_event_source_mismatch_workers(
            "model-a", "prefill", "ns-a", "prefill", "generate", 0,
        );

        let output = gather_pef(&registry);
        assert!(
            output.contains(
                "dynamo_component_router_worker_registered{dp_rank=\"0\",router_worker_id=\"123\",worker_type=\"decode\"} 1"
            ),
            "\nActual PEF:\n{output}"
        );
        assert!(
            output.contains(
                "dynamo_component_router_kv_event_source_mismatch_workers{model=\"model-a\",target_component=\"decode\",target_endpoint=\"generate\",target_namespace=\"ns-a\",worker_type=\"decode\"} 2"
            ),
            "\nActual PEF:\n{output}"
        );
        assert!(
            output.contains(
                "dynamo_component_router_kv_event_source_mismatch_workers{model=\"model-a\",target_component=\"prefill\",target_endpoint=\"generate\",target_namespace=\"ns-a\",worker_type=\"prefill\"} 0"
            ),
            "\nActual PEF:\n{output}"
        );

        workers_tx.send(std::collections::HashMap::new()).unwrap();
        let output = gather_pef(&registry);
        assert!(
            !output.contains("router_worker_id=\"123\""),
            "\nActual PEF after remove:\n{output}"
        );
    }
}
