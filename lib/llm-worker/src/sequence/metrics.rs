// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics for the direct-ZMQ active-sequence ingress.

use std::sync::{Arc, OnceLock};

use dynamo_runtime::component::Component;
use dynamo_runtime::metrics::MetricsHierarchy;
use prometheus::{IntCounter, IntGauge};

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
