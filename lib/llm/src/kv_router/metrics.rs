// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics for the KV router.
//!
//! This module centralizes the router-side Prometheus metric definitions of the request path:
//!
//! - [`RoutingOverheadMetrics`]: Per-request routing phase latency histograms.
//!   Registered on the frontend's own `prometheus::Registry` (default port 8000).
//!   Populated by `RoutingHost` in the frontend during routing decisions.
//!   - Frontend (aggregated and disaggregated): available on default port 8000
//!   - Standalone router: not created (frontend-only)
//!
//! - [`RouterRequestMetrics`]: Per-request aggregate histograms and counters (TTFT, ITL,
//!   tokens, KV hit rate, and non-max-overlap routing decisions).
//!   Registered on the DRT `MetricsRegistry` hierarchy via `Component::metrics()`.
//!   Eagerly created so they appear as zeros before any requests arrive.
//!   Populated by `RoutingHost::generate()` and its `RequestGuard` as it observes
//!   the streaming response (TTFT on first token, ITL per output block,
//!   ISL/OSL/kv_hit_rate at routing and completion).
//!   - Frontend, builtin modes (with or without session affinity): request-level metrics
//!     populated by `RoutingHost`; KV metrics stay at zero (no KV index)
//!   - Frontend, KV mode (aggregated and disaggregated): available on default port
//!     8000 via the `drt_metrics` bridge, populated per-request
//!   - Standalone router (`python -m dynamo.router`): available on `DYN_SYSTEM_PORT`
//!     when set (default is `-1`, disabled), populated per-request
//!
//! The standalone router does not create `WorkerLoadMetrics` or
//! `RoutingOverheadMetrics` (those are frontend-only). It only exposes
//! `RouterRequestMetrics` and standard DRT transport metrics
//! (`dynamo_component_inflight_requests`, `dynamo_component_requests_total`, etc.)
//! via the system status server when `DYN_SYSTEM_PORT` is explicitly set.
//!
//! The metrics of the code that keeps the router's view of workers up to date (worker load,
//! worker status, KV and active-sequence ingress, the remote indexer) are in
//! [`dynamo_sync::metrics`]. The worker-side KV event publisher keeps its own metrics next to
//! it (`publisher/metrics.rs`).
//!
//! See also: `docs/fern/pages/reference/observability/metrics-catalog.mdx` (Router metrics section).

use std::sync::{Arc, LazyLock, OnceLock};
use std::time::Duration;

use dynamo_runtime::metrics::prometheus_names::{
    frontend_service, labels, name_prefix, router_request, routing_overhead,
};
use dynamo_runtime::{component::Component, metrics::MetricsHierarchy};

/// Build a router metric name: `"router_" + frontend_service_suffix`.
fn router_metric(suffix: &str) -> String {
    format!("{}{}", router_request::METRIC_PREFIX, suffix)
}
use dynamo_runtime::traits::DistributedRuntimeProvider;
use prometheus::{
    HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Opts,
};

use dynamo_frontend_protocols::common::timing::RequestPhase;
use dynamo_http_server::metrics::generate_log_buckets;

use dynamo_frontend_protocols::common::timing::WORKER_TYPE_PREFILL;
use dynamo_kv_indexer::host::ApproximateLruStats;

/// Buckets for CPU-bound compute phases (block hashing, sequence hashing).
fn compute_overhead_buckets() -> Vec<f64> {
    prometheus::exponential_buckets(0.001, 2.0, 15).unwrap()
}

/// Buckets for async phases (indexer find_matches, scheduling, total).
fn async_overhead_buckets() -> Vec<f64> {
    prometheus::exponential_buckets(0.01, 3.0, 17).unwrap()
}

// ---------------------------------------------------------------------------
// Active-sequence inbound funnel metrics (per partition)
// ---------------------------------------------------------------------------

const ROUTING_GROUP_LABEL: &str = "routing_group";

/// Instrumentation for the per-partition inbound replica funnel: one channel
/// and one apply task per partition, fed by every publisher's ingress source.
/// Updated once per drain batch from the apply task.
pub(crate) struct ActiveSequenceIngressMetrics {
    queue_depth: IntGaugeVec,
    events_applied_total: IntCounterVec,
    drain_batches_total: IntCounterVec,
}

/// Per-partition handles implementing the core's drain observer.
pub(crate) struct ActiveSequenceIngressMetricHandles {
    queue_depth: IntGauge,
    events_applied_total: IntCounter,
    drain_batches_total: IntCounter,
}

static ACTIVE_SEQUENCE_INGRESS_METRICS: OnceLock<Arc<ActiveSequenceIngressMetrics>> =
    OnceLock::new();

impl ActiveSequenceIngressMetrics {
    pub(crate) fn from_component(component: &Component) -> Arc<Self> {
        ACTIVE_SEQUENCE_INGRESS_METRICS
            .get_or_init(|| {
                let metrics = component.metrics();
                let labels = [labels::MODEL, ROUTING_GROUP_LABEL, labels::WORKER_TYPE];
                Arc::new(Self {
                    queue_depth: metrics
                        .create_intgaugevec(
                            "router_active_sequence_ingress_queue_depth",
                            "Active-sequence events waiting in the partition's inbound channel, sampled after each drain batch",
                            &labels,
                            &[],
                        )
                        .expect("failed to create router_active_sequence_ingress_queue_depth"),
                    events_applied_total: metrics
                        .create_intcountervec(
                            "router_active_sequence_ingress_events_applied_total",
                            "Active-sequence events applied from the partition's inbound channel",
                            &labels,
                            &[],
                        )
                        .expect(
                            "failed to create router_active_sequence_ingress_events_applied_total",
                        ),
                    drain_batches_total: metrics
                        .create_intcountervec(
                            "router_active_sequence_ingress_drain_batches_total",
                            "Drain batches applied from the partition's inbound channel",
                            &labels,
                            &[],
                        )
                        .expect(
                            "failed to create router_active_sequence_ingress_drain_batches_total",
                        ),
                })
            })
            .clone()
    }

    /// One handle set per router partition. `worker_type` separates the
    /// prefill and decode routers of the same model, which run separate
    /// ingress channels.
    pub(crate) fn handles(
        &self,
        model: &str,
        routing_group: &str,
        worker_type: &str,
    ) -> ActiveSequenceIngressMetricHandles {
        let labels = [model, routing_group, worker_type];
        ActiveSequenceIngressMetricHandles {
            queue_depth: self.queue_depth.with_label_values(&labels),
            events_applied_total: self.events_applied_total.with_label_values(&labels),
            drain_batches_total: self.drain_batches_total.with_label_values(&labels),
        }
    }
}

impl dynamo_scheduler::host::ReplicaIngressObserver for ActiveSequenceIngressMetricHandles {
    fn observe_drain(&self, queue_depth: usize, applied: usize) {
        self.queue_depth.set(queue_depth as i64);
        self.events_applied_total.inc_by(applied as u64);
        self.drain_batches_total.inc();
    }
}

// ---------------------------------------------------------------------------
// Router queue and admission metrics
// ---------------------------------------------------------------------------

/// Queue gauges and cumulative admission counters by model, worker type, and policy class.
pub struct RouterQueueMetrics {
    pub pending_requests: IntGaugeVec,
    pub pending_isl_tokens: IntGaugeVec,
    pub pending_cached_tokens: IntGaugeVec,
    pub backpressure_total: IntCounterVec,
    pub received_total: IntCounterVec,
    pub rejected_total: IntCounterVec,
}

#[derive(Clone)]
pub struct RouterQueueMetricHandles {
    pub pending_requests: IntGauge,
    pub pending_isl_tokens: IntGauge,
    pub pending_cached_tokens: IntGauge,
    pub request_limit_rejections: IntCounter,
    pub raw_isl_limit_rejections: IntCounter,
    pub cached_token_limit_rejections: IntCounter,
    pub received_total: IntCounter,
    pub rejected_due_time_passed_total: IntCounter,
    pub rejected_queue_size_exceeded_total: IntCounter,
    // Shared by clones for one scheduler; separate schedulers may share Prometheus labels.
    reported_received: Arc<std::sync::atomic::AtomicU64>,
    reported_due_time_passed: Arc<std::sync::atomic::AtomicU64>,
    reported_queue_size_exceeded: Arc<std::sync::atomic::AtomicU64>,
}

pub static ROUTER_QUEUE_METRICS: LazyLock<RouterQueueMetrics> = LazyLock::new(|| {
    RouterQueueMetrics {
        pending_requests: IntGaugeVec::new(
            Opts::new(
                format!(
                    "{}_{}",
                    name_prefix::FRONTEND,
                    frontend_service::ROUTER_QUEUE_PENDING_REQUESTS
                ),
                "Number of requests pending in the router scheduler queue",
            ),
            &[labels::MODEL, labels::WORKER_TYPE, "policy_class"],
        )
        .expect("Failed to create router_queue_pending_requests gauge"),
        pending_isl_tokens: IntGaugeVec::new(
            Opts::new(
                format!("{}_router_queue_pending_isl_tokens", name_prefix::FRONTEND),
                "Sum of isl_tokens for requests pending in the router scheduler queue",
            ),
            &[labels::MODEL, labels::WORKER_TYPE, "policy_class"],
        )
        .expect("Failed to create router_queue_pending_isl_tokens gauge"),
        pending_cached_tokens: IntGaugeVec::new(
            Opts::new(
                format!(
                    "{}_router_queue_pending_cached_tokens",
                    name_prefix::FRONTEND
                ),
                "Estimated cached tokens for requests pending in the router scheduler queue",
            ),
            &[labels::MODEL, labels::WORKER_TYPE, "policy_class"],
        )
        .expect("Failed to create router_queue_pending_cached_tokens gauge"),
        received_total: IntCounterVec::new(
            Opts::new(
                format!("{}_router_admission_received_total", name_prefix::FRONTEND),
                "Total attempts received by router admission; includes retries and excludes advisory probes",
            ),
            &[labels::MODEL, labels::WORKER_TYPE, "policy_class"],
        ).expect("valid router admission received metric"),
        rejected_total: IntCounterVec::new(
            Opts::new(
                format!("{}_router_admission_rejected_total", name_prefix::FRONTEND),
                "Total router admission rejections by reason",
            ),
            &[labels::MODEL, labels::WORKER_TYPE, "policy_class", "reason"],
        ).expect("valid router admission rejection metric"),
        backpressure_total: IntCounterVec::new(
            Opts::new(
                format!("{}_router_queue_backpressure_total", name_prefix::FRONTEND),
                "Total number of router scheduler queue backpressure rejections",
            ),
            &[labels::MODEL, labels::WORKER_TYPE, "policy_class", "reason"],
        )
        .expect("Failed to create router_queue_backpressure_total counter"),
    }
});

impl RouterQueueMetrics {
    pub fn handles(
        &self,
        model: &str,
        worker_type: &str,
        policy_class: &str,
    ) -> RouterQueueMetricHandles {
        let queue_labels = [model, worker_type, policy_class];
        let rejection = |reason| {
            self.backpressure_total
                .with_label_values(&[model, worker_type, policy_class, reason])
        };
        let admission_rejection = |reason| {
            self.rejected_total
                .with_label_values(&[model, worker_type, policy_class, reason])
        };
        // Reserved by DEP #13891; no predicted-miss admission policy exists yet.
        admission_rejection("predicted_miss");
        RouterQueueMetricHandles {
            pending_requests: self.pending_requests.with_label_values(&queue_labels),
            pending_isl_tokens: self.pending_isl_tokens.with_label_values(&queue_labels),
            pending_cached_tokens: self.pending_cached_tokens.with_label_values(&queue_labels),
            request_limit_rejections: rejection("request_limit"),
            raw_isl_limit_rejections: rejection("raw_isl_token_limit"),
            cached_token_limit_rejections: rejection("cached_token_limit"),
            received_total: self.received_total.with_label_values(&queue_labels),
            rejected_due_time_passed_total: admission_rejection("due_time_passed"),
            rejected_queue_size_exceeded_total: admission_rejection("queue_size_exceeded"),
            reported_received: Arc::default(),
            reported_due_time_passed: Arc::default(),
            reported_queue_size_exceeded: Arc::default(),
        }
    }
}

impl RouterQueueMetricHandles {
    /// Export only unseen increments of the scheduler's cumulative totals, including when
    /// concurrent updates carry older snapshots.
    pub(super) fn update_admission(
        &self,
        received: u64,
        due_time_passed: u64,
        queue_size_exceeded: u64,
    ) {
        for (counter, reported, total) in [
            (&self.received_total, &self.reported_received, received),
            (
                &self.rejected_due_time_passed_total,
                &self.reported_due_time_passed,
                due_time_passed,
            ),
            (
                &self.rejected_queue_size_exceeded_total,
                &self.reported_queue_size_exceeded,
                queue_size_exceeded,
            ),
        ] {
            let previous = reported.fetch_max(total, std::sync::atomic::Ordering::Relaxed);
            counter.inc_by(total.saturating_sub(previous));
        }
    }
}

/// Register router queue gauges and admission counters with the Prometheus registry.
/// Called during frontend HTTP service setup (`service_v2.rs`), served on port 8000.
pub fn register_router_queue_metrics(
    registry: &prometheus::Registry,
) -> Result<(), prometheus::Error> {
    let m = &*ROUTER_QUEUE_METRICS;
    registry.register(Box::new(m.pending_requests.clone()))?;
    registry.register(Box::new(m.pending_isl_tokens.clone()))?;
    registry.register(Box::new(m.pending_cached_tokens.clone()))?;
    registry.register(Box::new(m.backpressure_total.clone()))?;
    registry.register(Box::new(m.received_total.clone()))?;
    registry.register(Box::new(m.rejected_total.clone()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Routing overhead metrics (histograms)
// ---------------------------------------------------------------------------

/// Per-request routing phase latency histograms (milliseconds).
pub struct RoutingOverheadMetrics {
    pub block_hashing: prometheus::Histogram,
    pub indexer_find_matches: prometheus::Histogram,
    pub seq_hashing: prometheus::Histogram,
    pub scheduling: prometheus::Histogram,
    pub total: prometheus::Histogram,
    pub shared_cache_query: prometheus::Histogram,
    pub shared_cache_errors_total: prometheus::IntCounter,
}

static ROUTING_OVERHEAD_METRICS: OnceLock<Arc<RoutingOverheadMetrics>> = OnceLock::new();

impl RoutingOverheadMetrics {
    /// Register routing overhead histograms with the given registry and store for later use.
    /// Metric names: `dynamo_router_overhead_*` with const label `router_id=instance_id`.
    /// Called during frontend HTTP service setup (`service_v2.rs`), so these metrics
    /// are served on the frontend's own port (default 8000). Not available in the
    /// standalone router, which has no frontend HTTP server.
    pub fn register(
        registry: &prometheus::Registry,
        instance_id: u64,
    ) -> Result<(), prometheus::Error> {
        let m = ROUTING_OVERHEAD_METRICS.get_or_init(|| {
            let compute_buckets = compute_overhead_buckets();
            let async_buckets = async_overhead_buckets();
            let router_id = instance_id.to_string();
            let make = |suffix: &str, help: &str, buckets: Vec<f64>| {
                let name = format!("{}_{}", name_prefix::ROUTER, suffix);
                prometheus::Histogram::with_opts(
                    HistogramOpts::new(name, help)
                        .const_label(labels::ROUTER_ID, &router_id)
                        .buckets(buckets),
                )
            };
            let block_hashing = make(
                routing_overhead::BLOCK_HASHING_MS,
                "Time spent computing block hashes in milliseconds",
                compute_buckets.clone(),
            )
            .expect("overhead_block_hashing_ms");
            let indexer_find_matches = make(
                routing_overhead::INDEXER_FIND_MATCHES_MS,
                "Time spent in indexer find_matches in milliseconds",
                async_buckets.clone(),
            )
            .expect("overhead_indexer_find_matches_ms");
            let seq_hashing = make(
                routing_overhead::SEQ_HASHING_MS,
                "Time spent computing sequence hashes in milliseconds",
                compute_buckets,
            )
            .expect("overhead_seq_hashing_ms");
            let scheduling = make(
                routing_overhead::SCHEDULING_MS,
                "Time spent in scheduler worker selection in milliseconds",
                async_buckets.clone(),
            )
            .expect("overhead_scheduling_ms");
            let total = make(
                routing_overhead::TOTAL_MS,
                "Total routing overhead per request in milliseconds",
                async_buckets.clone(),
            )
            .expect("overhead_total_ms");
            let shared_cache_query = make(
                routing_overhead::SHARED_CACHE_QUERY_MS,
                "Time spent querying the shared KV cache in milliseconds",
                async_buckets,
            )
            .expect("overhead_shared_cache_query_ms");
            let shared_cache_errors_total = {
                let name = format!(
                    "{}_{}",
                    name_prefix::ROUTER,
                    routing_overhead::SHARED_CACHE_ERRORS_TOTAL
                );
                prometheus::IntCounter::with_opts(
                    Opts::new(name, "Total shared cache failures")
                        .const_label(labels::ROUTER_ID, &router_id),
                )
                .expect("shared_cache_errors_total")
            };
            Arc::new(Self {
                block_hashing,
                indexer_find_matches,
                seq_hashing,
                scheduling,
                total,
                shared_cache_query,
                shared_cache_errors_total,
            })
        });
        registry.register(Box::new(m.block_hashing.clone()))?;
        registry.register(Box::new(m.indexer_find_matches.clone()))?;
        registry.register(Box::new(m.seq_hashing.clone()))?;
        registry.register(Box::new(m.scheduling.clone()))?;
        registry.register(Box::new(m.total.clone()))?;
        registry.register(Box::new(m.shared_cache_query.clone()))?;
        registry.register(Box::new(m.shared_cache_errors_total.clone()))?;
        Ok(())
    }

    /// Returns the registered metrics if `register()` was called earlier.
    pub fn get() -> Option<Arc<Self>> {
        ROUTING_OVERHEAD_METRICS.get().cloned()
    }

    /// Observe routing overhead timings in milliseconds.
    ///
    /// `indexer_duration` and `shared_cache_duration` are independent wall-clock times
    /// measured inside the `tokio::join!` block. They run in parallel, so
    /// `find_matches_elapsed >= max(indexer_duration, shared_cache_duration)`.
    pub fn observe(
        &self,
        hash_elapsed: Duration,
        seq_hash_elapsed: Duration,
        indexer_duration: Duration,
        shared_cache_duration: Option<Duration>,
        find_matches_elapsed: Duration,
        total_elapsed: Duration,
    ) {
        self.block_hashing
            .observe(hash_elapsed.as_secs_f64() * 1000.0);
        self.seq_hashing
            .observe(seq_hash_elapsed.saturating_sub(hash_elapsed).as_secs_f64() * 1000.0);
        self.indexer_find_matches
            .observe(indexer_duration.as_secs_f64() * 1000.0);
        if let Some(sc_duration) = shared_cache_duration {
            self.shared_cache_query
                .observe(sc_duration.as_secs_f64() * 1000.0);
        }
        self.scheduling.observe(
            total_elapsed
                .saturating_sub(find_matches_elapsed)
                .as_secs_f64()
                * 1000.0,
        );
        self.total.observe(total_elapsed.as_secs_f64() * 1000.0);
    }

    /// Increment the shared cache error counter.
    pub fn inc_shared_cache_errors(&self) {
        self.shared_cache_errors_total.inc();
    }
}

// ---------------------------------------------------------------------------
// Router request metrics (dynamo_component_router_* via MetricsHierarchy)
// ---------------------------------------------------------------------------

/// Aggregate per-request metrics observed at the router level.
///
/// Component-scoped via `from_component()` to get automatic `dynamo_component_` prefix,
/// `dynamo_namespace`/`dynamo_component`/`dynamo_endpoint` labels, and registration
/// with the DRT `MetricsRegistry` hierarchy.
///
/// # Scrapeability
///
/// - **Frontend, non-KV modes**: Always zero (registered but never populated).
/// - **Frontend, KV mode (aggregated and disaggregated)**: Available on the
///   frontend's `/metrics` endpoint (default port 8000) via the `drt_metrics`
///   bridge, populated per-request.
/// - **Standalone router** (`python -m dynamo.router`): Available on the system
///   status server when `DYN_SYSTEM_PORT` is set, populated per-request.
///
/// # When these metrics are created
///
/// Eagerly in `RoutingHost::new()`, so they appear as zeros before any requests.
/// Both the frontend pipeline and the standalone router (via Python bindings)
/// create a `RoutingHost`, so both get these metrics registered automatically.
///
/// # Why component-scoped
///
/// These metrics MUST be registered through the Component hierarchy (not a standalone
/// registry). In global planner deployments, the frontend's router is the global
/// entry point, but each worker pool has its own local router (e.g. prefill pool,
/// decode pool). Component-scoped metrics let each local router emit metrics with
/// distinct `dynamo_component` labels, so pools can be monitored and scaled
/// independently.
#[cfg_attr(test, derive(Clone))]
pub struct RouterRequestMetrics {
    /// Total requests admitted by the router scheduler.
    pub requests_started_total: prometheus::IntCounter,
    pub requests_total: prometheus::IntCounter,
    pub time_to_first_token_seconds: prometheus::Histogram,
    pub inter_token_latency_seconds: prometheus::Histogram,
    pub input_sequence_tokens: HistogramVec,
    pub output_sequence_tokens: prometheus::Histogram,
    pub kv_hit_rate: prometheus::Histogram,
    pub kv_transfer_estimated_latency_seconds: prometheus::Histogram,
    pub shared_cache_hit_rate: prometheus::Histogram,
    pub shared_cache_beyond_blocks: prometheus::Histogram,
    pub non_max_overlap_selections_total: IntCounterVec,
    pub overlap_blocks_lost: HistogramVec,
    /// Raw cached prefix tokens on the best eligible worker at selection, one observation
    /// per tracked attempt; labels `phase` (aggregated | prefill | decode) and `model`.
    pub kv_best_eligible_cached_prefix_tokens: IntCounterVec,
    /// Raw cached prefix tokens on the selected worker and DP rank at selection; same labels.
    pub kv_selected_cached_prefix_tokens: IntCounterVec,
    /// Backend-reported reused tokens, counted once per attempt; same labels.
    pub kv_worker_reused_tokens: IntCounterVec,
}

const KV_PHASE_LABEL: &str = "phase";

static ROUTER_REQUEST_METRICS: OnceLock<Arc<RouterRequestMetrics>> = OnceLock::new();

impl RouterRequestMetrics {
    /// Returns the registered metrics if `from_component()` was called earlier.
    pub fn get() -> Option<Arc<Self>> {
        ROUTER_REQUEST_METRICS.get().cloned()
    }

    /// Create from a Component, memoized in a static OnceLock.
    /// Uses the MetricsHierarchy API which auto-prepends `dynamo_component_`,
    /// injects hierarchy labels, and registers with the DRT `MetricsRegistry`.
    /// Also adds `router_id` (discovery instance_id) to distinguish router instances.
    ///
    /// Called eagerly by `RoutingHost::new()` so metrics appear as zeros at startup.
    pub fn from_component(component: &Component) -> Arc<Self> {
        ROUTER_REQUEST_METRICS
            .get_or_init(|| {
                let instance_id = component.drt().discovery().instance_id();
                let router_id = instance_id.to_string();
                let extra_labels: &[(&str, &str)] = &[(labels::ROUTER_ID, &router_id)];

                Arc::new(Self::build(component, extra_labels))
            })
            .clone()
    }

    fn build<H: MetricsHierarchy>(hierarchy: &H, extra_labels: &[(&str, &str)]) -> Self {
        let metrics = hierarchy.metrics();
        let requests_started_total = metrics
            .create_intcounter(
                &router_metric(frontend_service::REQUESTS_STARTED_TOTAL),
                "Total number of requests admitted by the router scheduler",
                extra_labels,
            )
            .expect("failed to create router_requests_started_total");
        let requests_total = metrics
            .create_intcounter(
                &router_metric(frontend_service::REQUESTS_TOTAL),
                "Total number of requests processed by the router",
                extra_labels,
            )
            .expect("failed to create router_requests_total");
        let time_to_first_token_seconds = metrics
            .create_histogram(
                &router_metric(frontend_service::TIME_TO_FIRST_TOKEN_SECONDS),
                "Time to first token observed at the router",
                extra_labels,
                Some(generate_log_buckets(0.001, 480.0, 18)),
            )
            .expect("failed to create router_time_to_first_token_seconds");
        let inter_token_latency_seconds = metrics
            .create_histogram(
                &router_metric(frontend_service::INTER_TOKEN_LATENCY_SECONDS),
                "Average inter-token latency observed at the router",
                extra_labels,
                Some(generate_log_buckets(0.001, 2.0, 13)),
            )
            .expect("failed to create router_inter_token_latency_seconds");
        let input_sequence_tokens = metrics
            .create_histogramvec(
                &router_metric(frontend_service::INPUT_SEQUENCE_TOKENS),
                "Input sequence length in tokens observed at the router",
                &[KV_PHASE_LABEL, labels::MODEL],
                extra_labels,
                Some(generate_log_buckets(50.0, 128000.0, 12)),
            )
            .expect("failed to create router_input_sequence_tokens");
        let output_sequence_tokens = metrics
            .create_histogram(
                &router_metric(frontend_service::OUTPUT_SEQUENCE_TOKENS),
                "Output sequence length in tokens observed at the router",
                extra_labels,
                Some(generate_log_buckets(50.0, 32000.0, 10)),
            )
            .expect("failed to create router_output_sequence_tokens");
        let kv_hit_rate = metrics
            .create_histogram(
                &router_metric(frontend_service::KV_HIT_RATE),
                "Predicted KV cache hit rate at routing time (0.0-1.0)",
                extra_labels,
                Some(prometheus::linear_buckets(0.0, 0.05, 21).unwrap()),
            )
            .expect("failed to create router_kv_hit_rate");
        let kv_transfer_estimated_latency_seconds = metrics
            .create_histogram(
                &router_metric(frontend_service::KV_TRANSFER_ESTIMATED_LATENCY_SECONDS),
                "Upper-bound estimation of KV cache transfer latency in disaggregated serving (prefill_complete to first_token)",
                extra_labels,
                Some(generate_log_buckets(0.001, 10.0, 15)),
            )
            .expect("failed to create router_kv_transfer_estimated_latency_seconds");
        let shared_cache_hit_rate = metrics
            .create_histogram(
                &router_metric(frontend_service::SHARED_CACHE_HIT_RATE),
                "Fraction of request blocks found in the shared KV cache (0.0-1.0)",
                extra_labels,
                Some(prometheus::linear_buckets(0.0, 0.05, 21).unwrap()),
            )
            .expect("failed to create router_shared_cache_hit_rate");
        let shared_cache_beyond_blocks = metrics
            .create_histogram(
                &router_metric(frontend_service::SHARED_CACHE_BEYOND_BLOCKS),
                "Shared cache blocks beyond device overlap for the selected worker",
                extra_labels,
                Some(prometheus::exponential_buckets(1.0, 2.0, 12).unwrap()),
            )
            .expect("failed to create router_shared_cache_beyond_blocks");
        let non_max_overlap_selections_total = metrics
            .create_intcountervec(
                &router_metric(frontend_service::NON_MAX_OVERLAP_SELECTIONS_TOTAL),
                "Total admitted prefill scheduler selections with less KV cache overlap than another eligible worker",
                &[labels::WORKER_TYPE],
                extra_labels,
            )
            .expect("failed to create router_non_max_overlap_selections_total");
        let overlap_blocks_lost = metrics
            .create_histogramvec(
                &router_metric(frontend_service::OVERLAP_BLOCKS_LOST),
                "Difference in effective KV cache overlap between the highest-overlap eligible prefill worker and selected worker",
                &[labels::WORKER_TYPE],
                extra_labels,
                Some(prometheus::exponential_buckets(0.25, 2.0, 16).unwrap()),
            )
            .expect("failed to create router_overlap_blocks_lost");
        non_max_overlap_selections_total.with_label_values(&[WORKER_TYPE_PREFILL]);
        overlap_blocks_lost.with_label_values(&[WORKER_TYPE_PREFILL]);
        let kv_tokens_counter = |name: &str, help: &str| {
            metrics
                .create_intcountervec(
                    &router_metric(name),
                    help,
                    &[KV_PHASE_LABEL, labels::MODEL],
                    extra_labels,
                )
                .expect("failed to create router KV token counter")
        };
        let kv_best_eligible_cached_prefix_tokens = kv_tokens_counter(
            frontend_service::KV_BEST_ELIGIBLE_CACHED_PREFIX_TOKENS_TOTAL,
            "Raw cached prefix tokens on the best eligible worker at selection time",
        );
        let kv_selected_cached_prefix_tokens = kv_tokens_counter(
            frontend_service::KV_SELECTED_CACHED_PREFIX_TOKENS_TOTAL,
            "Raw cached prefix tokens on the selected worker and DP rank",
        );
        let kv_worker_reused_tokens = kv_tokens_counter(
            frontend_service::KV_WORKER_REUSED_TOKENS_TOTAL,
            "Worker-reported reused tokens per routing attempt",
        );
        Self {
            requests_started_total,
            requests_total,
            time_to_first_token_seconds,
            inter_token_latency_seconds,
            input_sequence_tokens,
            output_sequence_tokens,
            kv_hit_rate,
            kv_transfer_estimated_latency_seconds,
            shared_cache_hit_rate,
            shared_cache_beyond_blocks,
            non_max_overlap_selections_total,
            overlap_blocks_lost,
            kv_best_eligible_cached_prefix_tokens,
            kv_selected_cached_prefix_tokens,
            kv_worker_reused_tokens,
        }
    }

    /// Use fresh, unregistered lifecycle counters and retain all other metric handles.
    #[cfg(test)]
    pub(crate) fn with_isolated_counters_for_test(&self) -> Arc<Self> {
        Arc::new(Self {
            requests_started_total: prometheus::IntCounter::new("requests_started_total", "test")
                .unwrap(),
            requests_total: prometheus::IntCounter::new("requests_total", "test").unwrap(),
            ..self.clone()
        })
    }

    #[cfg(test)]
    pub(crate) fn for_test(registry: &dynamo_runtime::MetricsRegistry) -> Arc<Self> {
        let mut hierarchy = test_fixtures::FakeHierarchy::component("", "", 0);
        hierarchy.registry = registry.clone();
        hierarchy.connection_id = None;
        Arc::new(Self::build(&hierarchy, &[]))
    }

    /// Record a selection that sacrificed KV cache overlap.
    pub fn observe_non_max_overlap_selection(&self, worker_type: &str, overlap_blocks_lost: f64) {
        debug_assert!(overlap_blocks_lost > 0.0);
        self.non_max_overlap_selections_total
            .with_label_values(&[worker_type])
            .inc();
        self.overlap_blocks_lost
            .with_label_values(&[worker_type])
            .observe(overlap_blocks_lost);
    }

    /// Record the router's estimates for one tracked attempt at selection time and
    /// return its worker-reuse counter, already exported at zero for backends that
    /// never report.
    pub fn observe_kv_route_estimate(
        &self,
        phase: RequestPhase,
        model: &str,
        best_tokens: u64,
        selected_tokens: u64,
    ) -> IntCounter {
        let labels = &[phase.as_str(), model];
        self.kv_best_eligible_cached_prefix_tokens
            .with_label_values(labels)
            .inc_by(best_tokens);
        self.kv_selected_cached_prefix_tokens
            .with_label_values(labels)
            .inc_by(selected_tokens);
        self.kv_worker_reused_tokens.with_label_values(labels)
    }
}

/// Process-local observability for the experimental approximate LRU.
///
/// NOTE: Gauges are last-router-wins. Approximate primaries are expected to be
/// singleton/local for now; a distributed deployment first needs explicit cache
/// synchronization, at which point metric aggregation should be designed with it.
pub(crate) struct ApproximateLruMetrics {
    configured_policy: IntGaugeVec,
    effective_policy: IntGaugeVec,
    ranks: IntGaugeVec,
    blocks: IntGaugeVec,
    leases: IntGauge,
    mutation_queue_depth: IntGauge,
    fallback_activations_total: IntCounter,
    eviction_batches_total: IntCounter,
    evicted_blocks_total: IntCounter,
    output_batches_total: IntCounter,
    messages_per_request: prometheus::Histogram,
    mutation_wait_seconds: prometheus::Histogram,
}

static APPROXIMATE_LRU_METRICS: OnceLock<Arc<ApproximateLruMetrics>> = OnceLock::new();

impl ApproximateLruMetrics {
    pub(crate) fn from_component(component: &Component) -> Arc<Self> {
        APPROXIMATE_LRU_METRICS
            .get_or_init(|| {
                let metrics = component.metrics();
                let gauge_vec = |name, help, label| {
                    metrics
                        .create_intgaugevec(name, help, &[label], &[])
                        .unwrap_or_else(|error| panic!("failed to create {name}: {error}"))
                };
                let counter = |name, help| {
                    metrics
                        .create_intcounter(name, help, &[])
                        .unwrap_or_else(|error| panic!("failed to create {name}: {error}"))
                };
                Arc::new(Self {
                    configured_policy: gauge_vec(
                        "router_approximate_cache_configured_policy",
                        "Configured process-local approximate cache policy",
                        "policy",
                    ),
                    effective_policy: gauge_vec(
                        "router_approximate_cache_effective_policy",
                        "Effective process-local approximate cache policy",
                        "policy",
                    ),
                    ranks: gauge_vec(
                        "router_approximate_lru_ranks",
                        "Approximate index ranks by effective retention policy",
                        "policy",
                    ),
                    blocks: gauge_vec(
                        "router_approximate_lru_blocks",
                        "Approximate LRU physical blocks by state",
                        "state",
                    ),
                    leases: metrics
                        .create_intgauge(
                            "router_approximate_lru_leases",
                            "Live approximate LRU request leases",
                            &[],
                        )
                        .expect("failed to create router_approximate_lru_leases"),
                    mutation_queue_depth: metrics
                        .create_intgauge(
                            "router_approximate_lru_mutation_queue_depth",
                            "Aggregate approximate LRU FIFO depth at command enqueue",
                            &[],
                        )
                        .expect("failed to create approximate LRU queue-depth gauge"),
                    fallback_activations_total: counter(
                        "router_approximate_lru_fallback_activations_total",
                        "Total rank incarnations pinned to TTL because capacity was unavailable",
                    ),
                    eviction_batches_total: counter(
                        "router_approximate_lru_eviction_batches_total",
                        "Total approximate LRU mutation batches that evicted physical copies",
                    ),
                    evicted_blocks_total: counter(
                        "router_approximate_lru_evicted_blocks_total",
                        "Total physical block copies evicted by approximate LRU",
                    ),
                    output_batches_total: counter(
                        "router_approximate_lru_output_batches_total",
                        "Total completed output-block batches sent to approximate LRU",
                    ),
                    messages_per_request: metrics
                        .create_histogram(
                            "router_approximate_lru_messages_per_request",
                            "Approximate LRU FIFO messages per request over each observation interval",
                            &[],
                            Some(prometheus::linear_buckets(1.0, 1.0, 16).unwrap()),
                        )
                        .expect("failed to create approximate LRU messages/request histogram"),
                    mutation_wait_seconds: metrics
                        .create_histogram(
                            "router_approximate_lru_mutation_wait_seconds",
                            "Mean approximate LRU FIFO wait per command over each observation interval",
                            &[],
                            Some(generate_log_buckets(0.000_001, 1.0, 16)),
                        )
                        .expect("failed to create approximate LRU mutation wait histogram"),
                })
            })
            .clone()
    }

    pub(crate) fn set_policies(&self, configured: &str, effective: &str) {
        for policy in ["ttl", "lru", "disabled"] {
            self.configured_policy
                .with_label_values(&[policy])
                .set(i64::from(policy == configured));
            self.effective_policy
                .with_label_values(&[policy])
                .set(i64::from(policy == effective));
        }
    }

    pub(crate) fn observe(&self, current: ApproximateLruStats, previous: &mut ApproximateLruStats) {
        let gauge = |value: usize| i64::try_from(value).unwrap_or(i64::MAX);
        self.ranks
            .with_label_values(&["lru"])
            .set(gauge(current.ranks));
        self.ranks
            .with_label_values(&["ttl_fallback"])
            .set(gauge(current.fallback_ranks));
        for (state, value) in [
            ("resident", current.resident_blocks),
            ("active", current.active_blocks),
            ("inactive", current.inactive_blocks),
            ("private", current.private_blocks),
            ("overcapacity", current.overcapacity_blocks),
        ] {
            self.blocks.with_label_values(&[state]).set(gauge(value));
        }
        self.leases.set(gauge(current.leases));
        self.mutation_queue_depth
            .set(gauge(current.mutation_queue_depth));

        self.fallback_activations_total.inc_by(
            current
                .fallback_activations
                .saturating_sub(previous.fallback_activations),
        );
        self.eviction_batches_total.inc_by(
            current
                .eviction_batches
                .saturating_sub(previous.eviction_batches),
        );
        self.evicted_blocks_total.inc_by(
            current
                .evicted_blocks
                .saturating_sub(previous.evicted_blocks),
        );
        self.output_batches_total.inc_by(
            current
                .output_batches
                .saturating_sub(previous.output_batches),
        );
        let requests = current.requests.saturating_sub(previous.requests);
        if requests > 0 {
            let messages = current
                .request_messages
                .saturating_sub(previous.request_messages);
            self.messages_per_request
                .observe(messages as f64 / requests as f64);
        }
        let wait_samples = current
            .mutation_wait_samples
            .saturating_sub(previous.mutation_wait_samples);
        if wait_samples > 0 {
            let wait_ns = current
                .mutation_wait_ns
                .saturating_sub(previous.mutation_wait_ns);
            self.mutation_wait_seconds
                .observe(wait_ns as f64 / wait_samples as f64 / 1_000_000_000.0);
        }
        *previous = current;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_sync::metrics::test_fixtures::gather_pef;

    #[test]
    fn router_request_metrics_register_with_component_labels() {
        let hierarchy = test_fixtures::FakeHierarchy::component("dynamo", "frontend", 0x123);
        let metrics = RouterRequestMetrics::build(&hierarchy, &[(labels::ROUTER_ID, "291")]);
        metrics
            .observe_kv_route_estimate(RequestPhase::Prefill, "m", 96, 64)
            .inc_by(72);
        metrics
            .input_sequence_tokens
            .with_label_values(&["prefill", "m"])
            .observe(100.0);
        let families = hierarchy.registry.get_prometheus_registry().gather();
        for name in [
            "input_sequence_tokens",
            "kv_best_eligible_cached_prefix_tokens_total",
            "kv_selected_cached_prefix_tokens_total",
            "kv_worker_reused_tokens_total",
        ] {
            let family = families
                .iter()
                .find(|family| family.name() == format!("dynamo_component_router_{name}"))
                .unwrap();
            let sample = &family.get_metric()[0];
            for (name, value) in [
                (labels::NAMESPACE, "dynamo"),
                (labels::COMPONENT, "frontend"),
                (labels::WORKER_ID, "123"),
                (labels::ROUTER_ID, "291"),
                (KV_PHASE_LABEL, "prefill"),
                (labels::MODEL, "m"),
            ] {
                assert!(
                    sample
                        .get_label()
                        .iter()
                        .any(|label| { label.name() == name && label.value() == value }),
                    "missing {name}={value} on {}",
                    family.name()
                );
            }
        }
    }

    #[test]
    fn kv_estimates_and_input_tokens_have_matching_labels() {
        let registry = dynamo_runtime::MetricsRegistry::new();
        let metrics = RouterRequestMetrics::for_test(&registry);
        for (phase, model, input, best) in [
            (RequestPhase::Prefill, "m", 100, 96),
            (RequestPhase::Decode, "m", 200, 150),
            (RequestPhase::Prefill, "other", 300, 250),
        ] {
            metrics
                .input_sequence_tokens
                .with_label_values(&[phase.as_str(), model])
                .observe(input as f64);
            metrics.observe_kv_route_estimate(phase, model, best, best);
            let output = registry.prometheus_expfmt_combined().unwrap();
            let phase = phase.as_str();
            assert!(output.contains(&format!("router_input_sequence_tokens_sum{{model=\"{model}\",phase=\"{phase}\"}} {input}\n")), "{output}");
            assert!(output.contains(&format!("router_kv_best_eligible_cached_prefix_tokens_total{{model=\"{model}\",phase=\"{phase}\"}} {best}\n")), "{output}");
        }
    }

    #[test]
    fn missing_worker_reports_export_zero_reused_tokens() {
        let registry = dynamo_runtime::MetricsRegistry::new();
        let metrics = RouterRequestMetrics::for_test(&registry);
        metrics.observe_kv_route_estimate(RequestPhase::Aggregated, "m", 96, 64);
        let output = registry.prometheus_expfmt_combined().unwrap();
        assert!(
            output.contains("kv_worker_reused_tokens_total{model=\"m\",phase=\"aggregated\"} 0"),
            "{output}"
        );
    }

    #[test]
    fn test_router_queue_metrics_pef() {
        let registry = prometheus::Registry::new();
        let metrics = RouterQueueMetrics {
            pending_requests: IntGaugeVec::new(
                Opts::new(
                    format!(
                        "{}_{}",
                        name_prefix::FRONTEND,
                        frontend_service::ROUTER_QUEUE_PENDING_REQUESTS
                    ),
                    "Number of requests pending in the router scheduler queue",
                ),
                &[labels::MODEL, labels::WORKER_TYPE, "policy_class"],
            )
            .unwrap(),
            pending_isl_tokens: IntGaugeVec::new(
                Opts::new(
                    format!("{}_router_queue_pending_isl_tokens", name_prefix::FRONTEND),
                    "Sum of isl_tokens for requests pending in the router scheduler queue",
                ),
                &[labels::MODEL, labels::WORKER_TYPE, "policy_class"],
            )
            .unwrap(),
            pending_cached_tokens: IntGaugeVec::new(
                Opts::new(
                    format!(
                        "{}_router_queue_pending_cached_tokens",
                        name_prefix::FRONTEND
                    ),
                    "Estimated cached tokens for requests pending in the router scheduler queue",
                ),
                &[labels::MODEL, labels::WORKER_TYPE, "policy_class"],
            )
            .unwrap(),
            received_total: IntCounterVec::new(
            Opts::new(
                format!("{}_router_admission_received_total", name_prefix::FRONTEND),
                "Total attempts received by router admission; includes retries and excludes advisory probes",
            ),
            &[labels::MODEL, labels::WORKER_TYPE, "policy_class"],
        ).expect("valid router admission received metric"),
        rejected_total: IntCounterVec::new(
            Opts::new(
                format!("{}_router_admission_rejected_total", name_prefix::FRONTEND),
                "Total router admission rejections by reason",
            ),
            &[labels::MODEL, labels::WORKER_TYPE, "policy_class", "reason"],
        ).expect("valid router admission rejection metric"),
        backpressure_total: IntCounterVec::new(
                Opts::new(
                    format!("{}_router_queue_backpressure_total", name_prefix::FRONTEND),
                    "Total number of router scheduler queue backpressure rejections",
                ),
                &[labels::MODEL, labels::WORKER_TYPE, "policy_class", "reason"],
            )
            .unwrap(),
        };
        registry
            .register(Box::new(metrics.pending_requests.clone()))
            .unwrap();
        registry
            .register(Box::new(metrics.pending_isl_tokens.clone()))
            .unwrap();
        registry
            .register(Box::new(metrics.pending_cached_tokens.clone()))
            .unwrap();
        registry
            .register(Box::new(metrics.backpressure_total.clone()))
            .unwrap();

        registry
            .register(Box::new(metrics.received_total.clone()))
            .unwrap();
        registry
            .register(Box::new(metrics.rejected_total.clone()))
            .unwrap();
        let handles = metrics.handles("model", "decode", "default");
        handles.update_admission(10, 1, 3);
        handles.clone().update_admission(10, 1, 3);
        handles.update_admission(9, 0, 2);
        // A replacement scheduler with the same labels adds its own counts.
        metrics
            .handles("model", "decode", "default")
            .update_admission(2, 1, 1);
        handles.pending_requests.set(5);
        handles.pending_isl_tokens.set(1024);
        handles.pending_cached_tokens.set(512);

        let output = gather_pef(&registry);
        let expected = "\
# HELP dynamo_frontend_router_admission_received_total Total attempts received by router admission; includes retries and excludes advisory probes
# TYPE dynamo_frontend_router_admission_received_total counter
dynamo_frontend_router_admission_received_total{model=\"model\",policy_class=\"default\",worker_type=\"decode\"} 12
# HELP dynamo_frontend_router_admission_rejected_total Total router admission rejections by reason
# TYPE dynamo_frontend_router_admission_rejected_total counter
dynamo_frontend_router_admission_rejected_total{model=\"model\",policy_class=\"default\",reason=\"due_time_passed\",worker_type=\"decode\"} 2
dynamo_frontend_router_admission_rejected_total{model=\"model\",policy_class=\"default\",reason=\"predicted_miss\",worker_type=\"decode\"} 0
dynamo_frontend_router_admission_rejected_total{model=\"model\",policy_class=\"default\",reason=\"queue_size_exceeded\",worker_type=\"decode\"} 4
# HELP dynamo_frontend_router_queue_backpressure_total Total number of router scheduler queue backpressure rejections
# TYPE dynamo_frontend_router_queue_backpressure_total counter
dynamo_frontend_router_queue_backpressure_total{model=\"model\",policy_class=\"default\",reason=\"cached_token_limit\",worker_type=\"decode\"} 0
dynamo_frontend_router_queue_backpressure_total{model=\"model\",policy_class=\"default\",reason=\"raw_isl_token_limit\",worker_type=\"decode\"} 0
dynamo_frontend_router_queue_backpressure_total{model=\"model\",policy_class=\"default\",reason=\"request_limit\",worker_type=\"decode\"} 0
# HELP dynamo_frontend_router_queue_pending_cached_tokens Estimated cached tokens for requests pending in the router scheduler queue
# TYPE dynamo_frontend_router_queue_pending_cached_tokens gauge
dynamo_frontend_router_queue_pending_cached_tokens{model=\"model\",policy_class=\"default\",worker_type=\"decode\"} 512
# HELP dynamo_frontend_router_queue_pending_isl_tokens Sum of isl_tokens for requests pending in the router scheduler queue
# TYPE dynamo_frontend_router_queue_pending_isl_tokens gauge
dynamo_frontend_router_queue_pending_isl_tokens{model=\"model\",policy_class=\"default\",worker_type=\"decode\"} 1024
# HELP dynamo_frontend_router_queue_pending_requests Number of requests pending in the router scheduler queue
# TYPE dynamo_frontend_router_queue_pending_requests gauge
dynamo_frontend_router_queue_pending_requests{model=\"model\",policy_class=\"default\",worker_type=\"decode\"} 5
";
        assert_eq!(
            output, expected,
            "\nActual PEF:\n{output}\nExpected PEF:\n{expected}"
        );
    }

    #[test]
    fn test_routing_overhead_saturating_sub() {
        let buckets = prometheus::exponential_buckets(0.0001, 2.0, 18).unwrap();
        let make = |name: &str| {
            prometheus::Histogram::with_opts(
                prometheus::HistogramOpts::new(name, "test").buckets(buckets.clone()),
            )
            .unwrap()
        };
        let metrics = RoutingOverheadMetrics {
            block_hashing: make("test_block_hashing_ms"),
            indexer_find_matches: make("test_find_matches_ms"),
            seq_hashing: make("test_seq_hashing_ms"),
            scheduling: make("test_scheduling_ms"),
            total: make("test_total_ms"),
            shared_cache_query: make("test_shared_cache_query_ms"),
            shared_cache_errors_total: prometheus::IntCounter::new(
                "test_shared_cache_errors_total",
                "test",
            )
            .unwrap(),
        };

        // Out-of-order cumulative durations: each phase < previous (would panic without saturating_sub)
        metrics.observe(
            Duration::from_millis(10),
            Duration::from_millis(5),
            Duration::from_millis(4),
            None,
            Duration::from_millis(3),
            Duration::from_millis(1),
        );
        // Reaching here without panic confirms saturating_sub works
    }
}

#[cfg(test)]
pub(crate) mod test_fixtures {
    //! A metrics hierarchy for tests that register component metrics without a runtime.

    use dynamo_runtime::MetricsRegistry;
    use dynamo_runtime::metrics::MetricsHierarchy;

    /// Stand-in for the DRT → Namespace → Component chain, without a live runtime.
    /// `connection_id` is what drives the auto-injected `worker_id` const label.
    pub(crate) struct FakeHierarchy {
        basename: String,
        parents: Vec<FakeHierarchy>,
        pub(crate) registry: MetricsRegistry,
        pub(crate) connection_id: Option<u64>,
    }

    impl FakeHierarchy {
        /// Mirror a component-level hierarchy: `["" (drt), namespace, component]`.
        pub(crate) fn component(namespace: &str, component: &str, connection_id: u64) -> Self {
            Self {
                basename: component.to_string(),
                parents: vec![Self::bare(""), Self::bare(namespace)],
                registry: MetricsRegistry::new(),
                connection_id: Some(connection_id),
            }
        }

        fn bare(name: &str) -> Self {
            Self {
                basename: name.to_string(),
                parents: Vec::new(),
                registry: MetricsRegistry::new(),
                connection_id: None,
            }
        }

        /// Render what a `/metrics` scrape of this hierarchy would return.
        pub(crate) fn expfmt(&self) -> String {
            self.registry.prometheus_expfmt_combined().unwrap()
        }
    }

    impl MetricsHierarchy for FakeHierarchy {
        fn basename(&self) -> String {
            self.basename.clone()
        }

        fn parent_hierarchies(&self) -> Vec<&dyn MetricsHierarchy> {
            self.parents
                .iter()
                .map(|p| p as &dyn MetricsHierarchy)
                .collect()
        }

        fn get_metrics_registry(&self) -> &MetricsRegistry {
            &self.registry
        }

        fn connection_id(&self) -> Option<u64> {
            self.connection_id
        }
    }
}
