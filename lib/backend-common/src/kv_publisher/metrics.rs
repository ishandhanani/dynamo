// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics of the worker-side KV event publisher.
//!
//! `KvPublisherMetrics`: Worker-local KV event publisher and ZMQ relay counters.
//! Registered on the DRT `MetricsRegistry` hierarchy via `Component::metrics()`.
//! Populated by `KvEventPublisher` and the ZMQ listener when engines publish KV
//! events.

use std::sync::{Arc, OnceLock};

use dynamo_runtime::metrics::prometheus_names::kv_publisher;
use dynamo_runtime::{component::Component, metrics::MetricsHierarchy};
use prometheus::{IntCounter, IntCounterVec};

/// Metrics for the KV publisher, created via the MetricsHierarchy API.
/// This provides automatic `dynamo_namespace`, `dynamo_component`, and other
/// hierarchy labels for free.
pub(crate) struct KvPublisherMetrics {
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
    fn build<H: MetricsHierarchy>(hierarchy: &H) -> Self {
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

pub(crate) fn kv_publisher_metrics() -> Option<Arc<KvPublisherMetrics>> {
    KV_PUBLISHER_METRICS.get().cloned()
}

#[cfg(test)]
mod tests {
    //! Regression coverage for silently-unregistered KV publisher metrics.
    //!
    //! `engines_dropped_events_total` was once declared with `worker_id` as a
    //! *variable* label. The runtime auto-injects `worker_id` as a *const* label on
    //! every component metric, so the validator rejected the request, the error was
    //! swallowed into an unregistered fallback counter, and the metric never
    //! appeared on `/metrics`. Nothing else observably changed: the fallback
    //! counter still incremented, just nowhere anyone could scrape it.
    //!
    //! These tests build the metric set against a fake hierarchy that injects the
    //! same auto-labels a real `Component` does, then assert each counter reaches
    //! the exposition output. No `DistributedRuntime` and no NATS required.

    use super::*;
    use dynamo_router_host::metrics::test_fixtures::FakeHierarchy;
    use dynamo_runtime::metrics::prometheus_names::{labels, name_prefix};

    fn exported_name(suffix: &str) -> String {
        format!("{}_{}", name_prefix::COMPONENT, suffix)
    }

    /// First non-comment sample line for `name`, if the metric was exported at all.
    /// A `# HELP`/`# TYPE` pair without a sample line does not count as exported.
    fn sample_line<'a>(expfmt: &'a str, name: &str) -> Option<&'a str> {
        expfmt
            .lines()
            .filter(|line| !line.starts_with('#'))
            .find(|line| {
                line.strip_prefix(name)
                    .is_some_and(|rest| rest.starts_with('{') || rest.starts_with(' '))
            })
    }

    #[test]
    fn every_kv_publisher_metric_reaches_the_registry() {
        let hierarchy = FakeHierarchy::component("dynamo", "backend", 0x7f3a1c);
        let metrics = KvPublisherMetrics::build(&hierarchy);

        // A CounterVec with no observed label values exports no sample lines, so
        // touch each vector once. `engines_dropped_events_total` is deliberately
        // left at zero: a worker that never sees an event_id gap is exactly the
        // state the original bug hid in.
        metrics.increment_zmq_event("received", "stored");
        metrics.increment_zmq_filtered_event("stored", "no_blocks");
        metrics.increment_zmq_conversion_issue("stored", "bad_hash");
        metrics.increment_zmq_suspicious_event("stored", "large_batch");

        let expfmt = hierarchy.expfmt();
        for suffix in [
            kv_publisher::ENGINES_DROPPED_EVENTS_TOTAL,
            kv_publisher::ZMQ_EVENTS_TOTAL,
            kv_publisher::ZMQ_FILTERED_EVENTS_TOTAL,
            kv_publisher::ZMQ_CONVERSION_ISSUES_TOTAL,
            kv_publisher::ZMQ_SUSPICIOUS_EVENTS_TOTAL,
        ] {
            let name = exported_name(suffix);
            assert!(
                sample_line(&expfmt, &name).is_some(),
                "{name} never reached the metrics registry. Exposition was:\n{expfmt}"
            );
        }
    }

    #[test]
    fn engines_dropped_counter_is_exported_at_zero_with_auto_labels() {
        let hierarchy = FakeHierarchy::component("dynamo", "backend", 0x7f3a1c);
        let _metrics = KvPublisherMetrics::build(&hierarchy);

        let expfmt = hierarchy.expfmt();
        let name = exported_name(kv_publisher::ENGINES_DROPPED_EVENTS_TOTAL);
        let line = sample_line(&expfmt, &name)
            .unwrap_or_else(|| panic!("{name} absent from exposition. Exposition was:\n{expfmt}"));

        // The hierarchy labels a real Component would inject. `worker_id` is the
        // one that collided; it must arrive as a const label, rendered from
        // connection_id as lowercase hex.
        for expected in [
            r#"dynamo_namespace="dynamo""#,
            r#"dynamo_component="backend""#,
            r#"worker_id="7f3a1c""#,
        ] {
            assert!(
                line.contains(expected),
                "expected label {expected} on sample line: {line}"
            );
        }
        assert!(
            line.ends_with(" 0"),
            "counter should be exported at zero before any gap is detected: {line}"
        );
    }

    #[test]
    fn worker_id_is_rejected_as_a_variable_label() {
        // Pins the mechanism the bug tripped over. If this ever starts returning
        // Ok, the collision is no longer caught at creation time and the guard in
        // `build` above is the only thing standing between a bad label and a
        // silently missing metric.
        let hierarchy = FakeHierarchy::component("dynamo", "backend", 0x7f3a1c);
        let err = hierarchy
            .metrics()
            .create_intcountervec(
                kv_publisher::ENGINES_DROPPED_EVENTS_TOTAL,
                "Total number of raw events dropped by engines",
                &[labels::WORKER_ID],
                &[],
            )
            .expect_err("worker_id must not be usable as a variable label");
        assert!(
            err.to_string()
                .contains("conflicts with auto-injected const label"),
            "unexpected error: {err}"
        );
    }
}
