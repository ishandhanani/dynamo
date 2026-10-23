// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics for the worker-local KV event publisher and ZMQ relay.
//!
//! Registered on the DRT `MetricsRegistry` hierarchy via `Component::metrics()`
//! and populated by `KvEventPublisher` and the ZMQ listener when engines publish
//! KV events.

use std::sync::{Arc, OnceLock};

use dynamo_runtime::component::Component;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::metrics::prometheus_names::kv_publisher;
use prometheus::{IntCounter, IntCounterVec};

/// Metrics for the KV publisher, created via the MetricsHierarchy API.
/// This provides automatic `dynamo_namespace`, `dynamo_component`, and other
/// hierarchy labels for free.
pub struct KvPublisherMetrics {
    /// Total number of raw events dropped by engines before reaching publisher.
    pub engines_dropped_events_total: IntCounter,
    /// Total number of decoded ZMQ KV events by relay stage and event type.
    pub zmq_events_total: IntCounterVec,
    /// Total number of ZMQ KV events filtered before conversion.
    pub zmq_filtered_events_total: IntCounterVec,
    /// Total number of ZMQ KV events dropped due to conversion issues.
    pub zmq_conversion_issues_total: IntCounterVec,
    /// Total number of suspicious-but-forwarded ZMQ KV events.
    pub zmq_suspicious_events_total: IntCounterVec,
}

static KV_PUBLISHER_METRICS: OnceLock<Arc<KvPublisherMetrics>> = OnceLock::new();

impl KvPublisherMetrics {
    /// Create from a Component, memoized in a static OnceLock.
    /// Uses the MetricsHierarchy API which auto-prepends `dynamo_component_`,
    /// injects hierarchy labels (including `worker_id`), and registers with the
    /// DRT `MetricsRegistry`.
    pub fn from_component(component: &Component) -> Arc<Self> {
        KV_PUBLISHER_METRICS
            .get_or_init(|| Arc::new(Self::build(component)))
            .clone()
    }

    /// Register the publisher's metrics against any metrics hierarchy.
    ///
    /// Split out of [`Self::from_component`] so registration is reachable from
    /// tests: `from_component` needs a real `Component` (and therefore a live
    /// `DistributedRuntime`) and memoizes into a process-global `OnceLock`, so
    /// only the first call in a process runs this code at all.
    pub fn build<H: MetricsHierarchy>(hierarchy: &H) -> Self {
        let metrics = hierarchy.metrics();
        let engines_dropped_events_total = metrics
            .create_intcounter(
                kv_publisher::ENGINES_DROPPED_EVENTS_TOTAL,
                "Total number of raw events dropped by engines before reaching publisher (detected via event_id gaps)",
                &[],
            )
            .expect("failed to create kv_publisher_engines_dropped_events_total");
        let zmq_events_total = metrics
            .create_intcountervec(
                kv_publisher::ZMQ_EVENTS_TOTAL,
                "Total number of ZMQ KV events seen by the relay",
                &["stage", "event_type"],
                &[],
            )
            .expect("failed to create kv_publisher_zmq_events_total");
        let zmq_filtered_events_total = metrics
            .create_intcountervec(
                kv_publisher::ZMQ_FILTERED_EVENTS_TOTAL,
                "Total number of ZMQ KV events filtered before conversion",
                &["event_type", "reason"],
                &[],
            )
            .expect("failed to create kv_publisher_zmq_filtered_events_total");
        let zmq_conversion_issues_total = metrics
            .create_intcountervec(
                kv_publisher::ZMQ_CONVERSION_ISSUES_TOTAL,
                "Total number of ZMQ KV events dropped due to conversion issues",
                &["event_type", "reason"],
                &[],
            )
            .expect("failed to create kv_publisher_zmq_conversion_issues_total");
        let zmq_suspicious_events_total = metrics
            .create_intcountervec(
                kv_publisher::ZMQ_SUSPICIOUS_EVENTS_TOTAL,
                "Total number of suspicious-but-forwarded ZMQ KV events",
                &["event_type", "reason"],
                &[],
            )
            .expect("failed to create kv_publisher_zmq_suspicious_events_total");

        Self {
            engines_dropped_events_total,
            zmq_events_total,
            zmq_filtered_events_total,
            zmq_conversion_issues_total,
            zmq_suspicious_events_total,
        }
    }

    /// Increment the engines dropped events counter by the given amount.
    pub fn increment_engines_dropped_events(&self, count: u64) {
        self.engines_dropped_events_total.inc_by(count);
    }

    pub fn increment_zmq_event(&self, stage: &'static str, event_type: &'static str) {
        self.zmq_events_total
            .with_label_values(&[stage, event_type])
            .inc();
    }

    pub fn increment_zmq_filtered_event(&self, event_type: &'static str, reason: &'static str) {
        self.zmq_filtered_events_total
            .with_label_values(&[event_type, reason])
            .inc();
    }

    pub fn increment_zmq_conversion_issue(&self, event_type: &'static str, reason: &'static str) {
        self.zmq_conversion_issues_total
            .with_label_values(&[event_type, reason])
            .inc();
    }

    pub fn increment_zmq_suspicious_event(&self, event_type: &'static str, reason: &'static str) {
        self.zmq_suspicious_events_total
            .with_label_values(&[event_type, reason])
            .inc();
    }
}

pub fn kv_publisher_metrics() -> Option<Arc<KvPublisherMetrics>> {
    KV_PUBLISHER_METRICS.get().cloned()
}
