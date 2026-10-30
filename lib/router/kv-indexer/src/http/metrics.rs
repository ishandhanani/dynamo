// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The standalone indexer service's own Prometheus surface: HTTP request
//! metrics and registry gauges (feature `metrics`). The indexer's counters are
//! exported by [`crate::KvIndexerCounterCollector`], which the server
//! registers next to these.

use std::sync::LazyLock;
use std::time::Instant;

use axum::{extract::MatchedPath, http::Request, middleware::Next, response::Response};
use prometheus::{
    HistogramVec, IntCounterVec, IntGauge, IntGaugeVec, Opts, exponential_buckets, histogram_opts,
};

use crate::ingest::registry::{ListenerStatus, WorkerRegistryStats};

const METRICS_PREFIX: &str = "dynamo_kvindexer";
const REQUEST_DURATION_SECONDS: &str = "request_duration_seconds";
const REQUESTS_TOTAL: &str = "requests_total";
const ERRORS_TOTAL: &str = "errors_total";
const MODELS: &str = "models";
const WORKERS: &str = "workers";
const LISTENERS: &str = "listeners";

pub struct StandaloneIndexerMetrics {
    pub request_duration: HistogramVec,
    pub requests_total: IntCounterVec,
    pub errors_total: IntCounterVec,
    pub models: IntGauge,
    pub workers: IntGauge,
    pub listeners: IntGaugeVec,
}

static METRICS: LazyLock<StandaloneIndexerMetrics> = LazyLock::new(|| StandaloneIndexerMetrics {
    request_duration: HistogramVec::new(
        histogram_opts!(
            format!("{METRICS_PREFIX}_{REQUEST_DURATION_SECONDS}"),
            "HTTP request latency",
            exponential_buckets(0.0001, 2.0, 20).expect("valid bucket params")
        ),
        &["endpoint"],
    )
    .expect("valid histogram"),
    requests_total: IntCounterVec::new(
        Opts::new(
            format!("{METRICS_PREFIX}_{REQUESTS_TOTAL}"),
            "Total HTTP requests",
        ),
        &["endpoint", "method"],
    )
    .expect("valid counter"),
    errors_total: IntCounterVec::new(
        Opts::new(
            format!("{METRICS_PREFIX}_{ERRORS_TOTAL}"),
            "HTTP error responses (4xx/5xx)",
        ),
        &["endpoint", "status_class"],
    )
    .expect("valid counter"),
    models: IntGauge::new(
        format!("{METRICS_PREFIX}_{MODELS}"),
        "Number of active model+routing-group indexers",
    )
    .expect("valid gauge"),
    workers: IntGauge::new(
        format!("{METRICS_PREFIX}_{WORKERS}"),
        "Number of registered worker instances",
    )
    .expect("valid gauge"),
    listeners: IntGaugeVec::new(
        Opts::new(
            format!("{METRICS_PREFIX}_{LISTENERS}"),
            "Number of ZMQ listeners by status",
        ),
        &["status"],
    )
    .expect("valid gauge"),
});

pub fn register(registry: &prometheus::Registry) -> Result<(), prometheus::Error> {
    let m = &*METRICS;
    registry.register(Box::new(m.request_duration.clone()))?;
    registry.register(Box::new(m.requests_total.clone()))?;
    registry.register(Box::new(m.errors_total.clone()))?;
    registry.register(Box::new(m.models.clone()))?;
    registry.register(Box::new(m.workers.clone()))?;
    registry.register(Box::new(m.listeners.clone()))?;
    Ok(())
}

pub async fn metrics_middleware(req: Request<axum::body::Body>, next: Next) -> Response {
    let path = req
        .extensions()
        .get::<MatchedPath>()
        .map(|m| m.as_str().to_owned())
        .unwrap_or_else(|| "unknown".to_owned());
    let method = req.method().as_str().to_owned();
    let start = Instant::now();
    let response = next.run(req).await;
    let elapsed = start.elapsed().as_secs_f64();
    let m = &*METRICS;
    m.requests_total
        .with_label_values(&[path.as_str(), method.as_str()])
        .inc();
    m.request_duration
        .with_label_values(&[path.as_str()])
        .observe(elapsed);
    let status = response.status().as_u16();
    if status >= 400 {
        let class = if status < 500 { "4xx" } else { "5xx" };
        m.errors_total
            .with_label_values(&[path.as_str(), class])
            .inc();
    }
    response
}

/// Show the registry's current state on the gauges; the server calls this at
/// scrape time.
pub fn set_worker_state(stats: &WorkerRegistryStats) {
    METRICS.models.set(stats.indexers as i64);
    METRICS.workers.set(stats.workers as i64);

    for status in ListenerStatus::ALL {
        METRICS
            .listeners
            .with_label_values(&[status.as_str()])
            .set(stats.listeners[status.metric_index()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus::Encoder;

    #[test]
    fn register_and_encode() {
        let registry = prometheus::Registry::new();
        register(&registry).expect("registration should succeed");

        set_worker_state(&WorkerRegistryStats {
            indexers: 1,
            workers: 2,
            listeners: [1, 1, 0, 0],
        });
        METRICS
            .request_duration
            .with_label_values(&["/health"])
            .observe(0.001);
        METRICS
            .requests_total
            .with_label_values(&["/health", "GET"])
            .inc();
        METRICS
            .errors_total
            .with_label_values(&["/query", "4xx"])
            .inc();

        let encoder = prometheus::TextEncoder::new();
        let mut buf = Vec::new();
        encoder.encode(&registry.gather(), &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        assert!(output.contains("dynamo_kvindexer_request_duration_seconds"));
        assert!(output.contains("dynamo_kvindexer_requests_total"));
        assert!(output.contains("dynamo_kvindexer_errors_total"));
        assert!(output.contains("dynamo_kvindexer_models 1"));
        assert!(output.contains("dynamo_kvindexer_workers 2"));
        assert!(output.contains("dynamo_kvindexer_listeners{status=\"pending\"} 1"));
        assert!(output.contains("dynamo_kvindexer_listeners{status=\"active\"} 1"));
    }
}
