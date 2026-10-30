// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router queue gauges and admission totals, read from the scheduler's
//! per-class counters at scrape time.
//!
//! The scheduler keeps one set of atomics per policy class
//! (`ClassQueueStats`); nothing on the request path copies them into
//! Prometheus objects. Each embedded router registers its partition as a
//! [`RouterQueueSource`] and the collector reads every live source when the
//! registry is gathered. Partitions that share labels add up.

use std::sync::Arc;

use dynamo_metrics::collect::{SourceSet, descs_of};
use dynamo_scheduler::queue::ClassQueueStats;
use prometheus::{
    IntCounterVec, IntGaugeVec, Opts,
    core::{Collector, Desc},
    proto::MetricFamily,
};
use tokio_util::sync::CancellationToken;

/// Reads one partition's counters for a policy class index.
pub(crate) type QueueStatsReader =
    Arc<dyn Fn(usize) -> Option<ClassQueueStats> + Send + Sync + 'static>;

/// One router partition's queue as the collector sees it: the labels of its
/// policy classes, index-aligned with the scheduler's class indices, and how
/// to read their counters. Dropped by its owner or cancelled with its
/// lifecycle, the partition leaves the scrape.
pub(crate) struct RouterQueueSource {
    labels: Vec<[String; 3]>,
    reader: QueueStatsReader,
    cancellation: CancellationToken,
}

const QUEUE_LABELS: &[&str] = &["model", "worker_type", "policy_class"];
const REJECTION_LABELS: &[&str] = &["model", "worker_type", "policy_class", "reason"];

/// Collects the queue gauges and admission totals from every live source.
#[derive(Clone)]
pub(super) struct RouterQueueCollector {
    pending_requests: Opts,
    pending_isl_tokens: Opts,
    pending_cached_tokens: Opts,
    received_total: Opts,
    deadline_rejections_total: Opts,
    descs: Vec<Desc>,
    sources: SourceSet<RouterQueueSource>,
}

impl RouterQueueCollector {
    pub(super) fn new(
        pending_requests: Opts,
        pending_isl_tokens: Opts,
        pending_cached_tokens: Opts,
        received_total: Opts,
        deadline_rejections_total: Opts,
    ) -> Result<Self, prometheus::Error> {
        let mut descs = Vec::new();
        for opts in [
            &pending_requests,
            &pending_isl_tokens,
            &pending_cached_tokens,
        ] {
            descs.extend(descs_of(&IntGaugeVec::new(opts.clone(), QUEUE_LABELS)?));
        }
        descs.extend(descs_of(&IntCounterVec::new(
            received_total.clone(),
            QUEUE_LABELS,
        )?));
        descs.extend(descs_of(&IntCounterVec::new(
            deadline_rejections_total.clone(),
            REJECTION_LABELS,
        )?));
        Ok(Self {
            pending_requests,
            pending_isl_tokens,
            pending_cached_tokens,
            received_total,
            deadline_rejections_total,
            descs,
            sources: SourceSet::new(),
        })
    }

    /// Register a partition's queue. The returned handle keeps it in the scrape.
    pub(super) fn watch(
        &self,
        labels: Vec<[String; 3]>,
        reader: QueueStatsReader,
        cancellation: CancellationToken,
    ) -> Arc<RouterQueueSource> {
        let source = Arc::new(RouterQueueSource {
            labels,
            reader,
            cancellation,
        });
        self.sources.attach(&source);
        source
    }
}

impl Collector for RouterQueueCollector {
    fn desc(&self) -> Vec<&Desc> {
        self.descs.iter().collect()
    }

    fn collect(&self) -> Vec<MetricFamily> {
        // Fresh vectors per scrape: concurrent scrapes stay independent and a
        // partition that left takes its series with it.
        let pending_requests = IntGaugeVec::new(self.pending_requests.clone(), QUEUE_LABELS)
            .expect("validated router queue metric options");
        let pending_isl_tokens = IntGaugeVec::new(self.pending_isl_tokens.clone(), QUEUE_LABELS)
            .expect("validated router queue metric options");
        let pending_cached_tokens =
            IntGaugeVec::new(self.pending_cached_tokens.clone(), QUEUE_LABELS)
                .expect("validated router queue metric options");
        let received_total = IntCounterVec::new(self.received_total.clone(), QUEUE_LABELS)
            .expect("validated router queue metric options");
        let deadline_rejections_total =
            IntCounterVec::new(self.deadline_rejections_total.clone(), REJECTION_LABELS)
                .expect("validated router queue metric options");

        for source in self
            .sources
            .live(|source| !source.cancellation.is_cancelled())
        {
            for (class_index, [model, worker_type, policy_class]) in
                source.labels.iter().enumerate()
            {
                let Some(stats) = (source.reader)(class_index) else {
                    continue;
                };
                let labels = [model.as_str(), worker_type.as_str(), policy_class.as_str()];
                pending_requests
                    .with_label_values(&labels)
                    .add(stats.pending_count as i64);
                pending_isl_tokens
                    .with_label_values(&labels)
                    .add(stats.pending_isl_tokens as i64);
                pending_cached_tokens
                    .with_label_values(&labels)
                    .add(stats.pending_cached_tokens as i64);
                received_total
                    .with_label_values(&labels)
                    .inc_by(stats.received_total);
                deadline_rejections_total
                    .with_label_values(&[model, worker_type, policy_class, "due_time_passed"])
                    .inc_by(stats.rejected_due_time_passed_total);
                // Reserved by DEP #13891; no predicted-miss admission policy exists yet.
                deadline_rejections_total.with_label_values(&[
                    model,
                    worker_type,
                    policy_class,
                    "predicted_miss",
                ]);
            }
        }

        let mut families = pending_requests.collect();
        families.extend(pending_isl_tokens.collect());
        families.extend(pending_cached_tokens.collect());
        families.extend(received_total.collect());
        families.extend(deadline_rejections_total.collect());
        families
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus::{Registry, TextEncoder};

    fn collector() -> RouterQueueCollector {
        RouterQueueCollector::new(
            Opts::new("test_router_queue_pending_requests", "pending"),
            Opts::new("test_router_queue_pending_isl_tokens", "isl"),
            Opts::new("test_router_queue_pending_cached_tokens", "cached"),
            Opts::new("test_router_admission_received_total", "received"),
            Opts::new("test_router_admission_rejected_total", "rejected"),
        )
        .unwrap()
    }

    fn output(registry: &Registry) -> String {
        TextEncoder::new()
            .encode_to_string(&registry.gather())
            .unwrap()
    }

    fn stats(received: u64, rejected: u64, pending: usize) -> ClassQueueStats {
        ClassQueueStats {
            received_total: received,
            rejected_due_time_passed_total: rejected,
            pending_count: pending,
            pending_isl_tokens: pending * 64,
            pending_cached_tokens: pending * 32,
        }
    }

    #[test]
    fn scrape_reads_each_class_by_index_and_forgets_departed_partitions() {
        let collector = collector();
        let registry = Registry::new();
        registry.register(Box::new(collector.clone())).unwrap();
        let per_class = [stats(10, 1, 2), stats(20, 3, 3)];
        let labels = ["latency", "bulk"]
            .map(|class| ["m".to_string(), "decode".to_string(), class.to_string()])
            .to_vec();
        let source = collector.watch(
            labels,
            Arc::new(move |index| per_class.get(index).copied()),
            CancellationToken::new(),
        );

        let text = output(&registry);
        for line in [
            "test_router_queue_pending_requests{model=\"m\",policy_class=\"latency\",worker_type=\"decode\"} 2",
            "test_router_queue_pending_requests{model=\"m\",policy_class=\"bulk\",worker_type=\"decode\"} 3",
            "test_router_queue_pending_isl_tokens{model=\"m\",policy_class=\"bulk\",worker_type=\"decode\"} 192",
            "test_router_queue_pending_cached_tokens{model=\"m\",policy_class=\"latency\",worker_type=\"decode\"} 64",
            "test_router_admission_received_total{model=\"m\",policy_class=\"bulk\",worker_type=\"decode\"} 20",
            "test_router_admission_rejected_total{model=\"m\",policy_class=\"latency\",reason=\"due_time_passed\",worker_type=\"decode\"} 1",
            "test_router_admission_rejected_total{model=\"m\",policy_class=\"latency\",reason=\"predicted_miss\",worker_type=\"decode\"} 0",
        ] {
            assert!(text.contains(line), "missing {line}\n{text}");
        }

        drop(source);
        assert!(
            output(&registry).is_empty(),
            "a dropped partition must leave the scrape"
        );
    }

    #[test]
    fn partitions_sharing_labels_add_up_until_cancelled() {
        let collector = collector();
        let registry = Registry::new();
        registry.register(Box::new(collector.clone())).unwrap();
        let labels = vec![["m".to_string(), "decode".to_string(), "latency".to_string()]];
        let cancel = CancellationToken::new();
        let _first = collector.watch(
            labels.clone(),
            Arc::new(|_| Some(stats(5, 0, 1))),
            cancel.clone(),
        );
        let _second = collector.watch(
            labels,
            Arc::new(|_| Some(stats(7, 2, 4))),
            CancellationToken::new(),
        );
        let text = output(&registry);
        assert!(text.contains("test_router_queue_pending_requests{model=\"m\",policy_class=\"latency\",worker_type=\"decode\"} 5"), "{text}");
        assert!(text.contains("test_router_admission_received_total{model=\"m\",policy_class=\"latency\",worker_type=\"decode\"} 12"), "{text}");

        cancel.cancel();
        let text = output(&registry);
        assert!(text.contains("test_router_queue_pending_requests{model=\"m\",policy_class=\"latency\",worker_type=\"decode\"} 4"), "{text}");
        assert!(text.contains("test_router_admission_received_total{model=\"m\",policy_class=\"latency\",worker_type=\"decode\"} 7"), "{text}");
    }
}
