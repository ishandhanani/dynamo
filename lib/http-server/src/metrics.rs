// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-level accounting of an HTTP frontend: how many requests started and finished
//! with which outcome, how many are inflight or still queued, how long they took, and how
//! many clients went away.
//!
//! This is the protocol-agnostic half of a frontend's metrics. Every label is a string the
//! composition root chooses (`dynamo-llm` passes its endpoint, request-type, status, and
//! error-type enums as their label values), and the metric names are the
//! [`frontend_service`] names under a caller-provided prefix. Protocol-specific families,
//! such as token counts or time to first token, stay with the protocol that defines them.

use dynamo_metrics::prometheus_names::frontend_service;
use prometheus::{
    HistogramOpts, HistogramVec, IntCounterVec, IntGauge, IntGaugeVec, Opts, Registry,
};

/// The request-level collectors of one HTTP frontend.
///
/// Inflight covers a request from handler entry until its response is complete; queued is
/// the leading part of that, until the first response work begins, so queued time is a
/// subset of inflight time.
pub struct RequestMetrics {
    request_started_counter: IntCounterVec,
    request_counter: IntCounterVec,
    /// Deprecated: use `active_requests_gauge`. Kept for backwards compatibility.
    inflight_gauge: IntGaugeVec,
    active_requests_gauge: IntGaugeVec,
    client_disconnect_gauge: IntGauge,
    http_queue_gauge: IntGaugeVec,
    request_duration: HistogramVec,
}

impl RequestMetrics {
    /// Builds the collectors as `{prefix}_<name>`, with `request_duration_buckets` as the
    /// upper bounds of the duration histogram. `prefix` must already be a valid metric
    /// name prefix.
    pub fn new(prefix: &str, request_duration_buckets: Vec<f64>) -> Self {
        let name = |suffix: &str| format!("{prefix}_{suffix}");

        let request_counter = IntCounterVec::new(
            Opts::new(
                name(frontend_service::REQUESTS_TOTAL),
                "Total number of LLM requests processed",
            ),
            &["model", "endpoint", "request_type", "status", "error_type"],
        )
        .expect("valid request counter definition");

        let request_started_counter = IntCounterVec::new(
            Opts::new(
                name(frontend_service::REQUESTS_STARTED_TOTAL),
                "Total number of LLM requests accepted by the frontend handler",
            ),
            &["model", "endpoint", "request_type"],
        )
        .expect("valid started-request counter definition");

        let inflight_gauge = IntGaugeVec::new(
            Opts::new(
                name(frontend_service::INFLIGHT_REQUESTS),
                "Number of inflight requests",
            ),
            &["model"],
        )
        .expect("valid inflight gauge definition");

        let active_requests_gauge = IntGaugeVec::new(
            Opts::new(
                name(frontend_service::ACTIVE_REQUESTS),
                "Number of requests currently being handled by the frontend, from HTTP handler entry to response completion",
            ),
            &["model"],
        )
        .expect("valid active-request gauge definition");

        let client_disconnect_gauge = IntGauge::new(
            name(frontend_service::DISCONNECTED_CLIENTS),
            "Number of disconnected clients",
        )
        .expect("valid disconnect gauge definition");

        let http_queue_gauge = IntGaugeVec::new(
            Opts::new(
                name(frontend_service::QUEUED_REQUESTS),
                "Number of requests in HTTP processing queue",
            ),
            &["model"],
        )
        .expect("valid queue gauge definition");

        let request_duration = HistogramVec::new(
            HistogramOpts::new(
                name(frontend_service::REQUEST_DURATION_SECONDS),
                "Duration of LLM requests",
            )
            .buckets(request_duration_buckets),
            &["model"],
        )
        .expect("valid request duration histogram definition");

        Self {
            request_started_counter,
            request_counter,
            inflight_gauge,
            active_requests_gauge,
            client_disconnect_gauge,
            http_queue_gauge,
            request_duration,
        }
    }

    /// Registers every collector with `registry`.
    pub fn register(&self, registry: &Registry) -> Result<(), prometheus::Error> {
        registry.register(Box::new(self.request_started_counter.clone()))?;
        registry.register(Box::new(self.request_counter.clone()))?;
        registry.register(Box::new(self.inflight_gauge.clone()))?;
        registry.register(Box::new(self.active_requests_gauge.clone()))?;
        registry.register(Box::new(self.client_disconnect_gauge.clone()))?;
        registry.register(Box::new(self.http_queue_gauge.clone()))?;
        registry.register(Box::new(self.request_duration.clone()))?;
        Ok(())
    }

    /// A request was accepted by a handler.
    pub fn inc_request_started(&self, model: &str, endpoint: &str, request_type: &str) {
        self.request_started_counter
            .with_label_values(&[model, endpoint, request_type])
            .inc()
    }

    /// A request finished with `status` and `error_type`.
    pub fn inc_request_completed(
        &self,
        model: &str,
        endpoint: &str,
        request_type: &str,
        status: &str,
        error_type: &str,
    ) {
        self.request_counter
            .with_label_values(&[model, endpoint, request_type, status, error_type])
            .inc()
    }

    /// How many requests finished with these labels.
    pub fn request_count(
        &self,
        model: &str,
        endpoint: &str,
        request_type: &str,
        status: &str,
        error_type: &str,
    ) -> u64 {
        self.request_counter
            .with_label_values(&[model, endpoint, request_type, status, error_type])
            .get()
    }

    /// A request entered the handler; both the inflight and active gauges track it.
    pub fn inc_inflight(&self, model: &str) {
        self.inflight_gauge.with_label_values(&[model]).inc();
        self.active_requests_gauge.with_label_values(&[model]).inc();
    }

    /// A request's response completed.
    pub fn dec_inflight(&self, model: &str) {
        self.inflight_gauge.with_label_values(&[model]).dec();
        self.active_requests_gauge.with_label_values(&[model]).dec();
    }

    /// Requests of `model` currently inflight.
    pub fn inflight_count(&self, model: &str) -> i64 {
        self.inflight_gauge.with_label_values(&[model]).get()
    }

    /// A client went away before its response completed.
    pub fn inc_client_disconnect(&self) {
        self.client_disconnect_gauge.inc();
    }

    /// Clients that went away before their response completed.
    pub fn client_disconnect_count(&self) -> i64 {
        self.client_disconnect_gauge.get()
    }

    /// A request entered the HTTP processing queue.
    pub fn inc_queued(&self, model: &str) {
        self.http_queue_gauge.with_label_values(&[model]).inc()
    }

    /// A request left the HTTP processing queue.
    pub fn dec_queued(&self, model: &str) {
        self.http_queue_gauge.with_label_values(&[model]).dec()
    }

    /// Records the end-to-end duration of a finished request.
    pub fn observe_request_duration(&self, model: &str, seconds: f64) {
        self.request_duration
            .with_label_values(&[model])
            .observe(seconds)
    }

    pub fn request_started_counter(&self) -> &IntCounterVec {
        &self.request_started_counter
    }

    pub fn request_counter(&self) -> &IntCounterVec {
        &self.request_counter
    }

    pub fn inflight_gauge(&self) -> &IntGaugeVec {
        &self.inflight_gauge
    }

    pub fn active_requests_gauge(&self) -> &IntGaugeVec {
        &self.active_requests_gauge
    }

    pub fn client_disconnect_gauge(&self) -> &IntGauge {
        &self.client_disconnect_gauge
    }

    pub fn http_queue_gauge(&self) -> &IntGaugeVec {
        &self.http_queue_gauge
    }

    pub fn request_duration(&self) -> &HistogramVec {
        &self.request_duration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collectors_take_the_prefix_and_register_once() {
        let metrics = RequestMetrics::new("test_frontend", vec![0.0, 1.0, 10.0]);
        let registry = Registry::new();
        metrics.register(&registry).unwrap();
        assert!(
            metrics.register(&registry).is_err(),
            "registering the same collectors twice must fail"
        );

        metrics.inc_request_started("m", "chat_completions", "unary");
        metrics.inc_inflight("m");
        metrics.inc_queued("m");
        assert_eq!(metrics.inflight_count("m"), 1);
        metrics.dec_queued("m");
        metrics.dec_inflight("m");
        metrics.observe_request_duration("m", 0.5);
        metrics.inc_request_completed("m", "chat_completions", "unary", "success", "");
        metrics.inc_client_disconnect();

        assert_eq!(metrics.inflight_count("m"), 0);
        assert_eq!(
            metrics.request_count("m", "chat_completions", "unary", "success", ""),
            1
        );
        assert_eq!(metrics.client_disconnect_count(), 1);

        let names: Vec<String> = registry
            .gather()
            .into_iter()
            .map(|family| family.name().to_string())
            .collect();
        for suffix in [
            frontend_service::REQUESTS_TOTAL,
            frontend_service::REQUESTS_STARTED_TOTAL,
            frontend_service::INFLIGHT_REQUESTS,
            frontend_service::ACTIVE_REQUESTS,
            frontend_service::DISCONNECTED_CLIENTS,
            frontend_service::QUEUED_REQUESTS,
            frontend_service::REQUEST_DURATION_SECONDS,
        ] {
            let name = format!("test_frontend_{suffix}");
            assert!(names.contains(&name), "{name} missing from {names:?}");
        }
    }
}
