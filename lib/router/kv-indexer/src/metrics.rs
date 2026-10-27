// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Indexer counters.
//!
//! The indexer threads increment a fixed set of atomics, one per label
//! combination; nothing on the event path touches a Prometheus object. A
//! host that exposes them registers a [`KvIndexerCounterCollector`] (feature
//! `metrics`) and attaches the handles it wants in the scrape. The label
//! space is finite and known at compile time, so every series is always
//! exported, starting at zero.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dynamo_router_types::protocols::{KvCacheEventData, KvCacheEventError, RouterEvent};

static IGNORED_RESIDENCY_EVENT_COUNT: AtomicU64 = AtomicU64::new(0);

/// Lightweight, `Copy` discriminant for [`KvCacheEventData`].
///
/// Extracted before the event is moved into `apply_event()`, then passed to
/// [`PreBoundEventCounters::inc`] so the compiler enforces exhaustiveness
/// without requiring a clone of the full event payload.
///
/// `Display` produces the Prometheus label value (`"stored"`, `"removed"`,
/// `"cleared"`), so this enum is also the single source of truth for the
/// `event_type` label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Stored,
    Removed,
    Cleared,
}

impl EventKind {
    pub const ALL: [Self; 3] = [Self::Stored, Self::Removed, Self::Cleared];

    pub fn of(data: &KvCacheEventData) -> Self {
        match data {
            KvCacheEventData::Stored(_) => Self::Stored,
            KvCacheEventData::Removed(_) => Self::Removed,
            KvCacheEventData::Cleared => Self::Cleared,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Stored => METRIC_EVENT_STORED,
            Self::Removed => METRIC_EVENT_REMOVED,
            Self::Cleared => METRIC_EVENT_CLEARED,
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

impl std::fmt::Display for EventKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// The `status` label of an applied event: `ok` or the error it failed with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventStatus {
    Ok,
    ParentNotFound,
    BlockNotFound,
    InvalidBlock,
    CapacityExhausted,
    AllocationFailed,
    OwnershipDegreeOverflow,
    IndexerInvariantViolation,
    UnsupportedResidencyDomain,
}

impl EventStatus {
    pub const ALL: [Self; 9] = [
        Self::Ok,
        Self::ParentNotFound,
        Self::BlockNotFound,
        Self::InvalidBlock,
        Self::CapacityExhausted,
        Self::AllocationFailed,
        Self::OwnershipDegreeOverflow,
        Self::IndexerInvariantViolation,
        Self::UnsupportedResidencyDomain,
    ];

    pub fn of(result: Result<(), KvCacheEventError>) -> Self {
        match result {
            Ok(()) => Self::Ok,
            Err(KvCacheEventError::ParentBlockNotFound) => Self::ParentNotFound,
            Err(KvCacheEventError::BlockNotFound) => Self::BlockNotFound,
            Err(KvCacheEventError::InvalidBlockSequence) => Self::InvalidBlock,
            Err(KvCacheEventError::CapacityExhausted) => Self::CapacityExhausted,
            Err(KvCacheEventError::AllocationFailed) => Self::AllocationFailed,
            Err(KvCacheEventError::OwnershipDegreeOverflow) => Self::OwnershipDegreeOverflow,
            Err(KvCacheEventError::IndexerInvariantViolation) => Self::IndexerInvariantViolation,
            Err(KvCacheEventError::UnsupportedResidencyDomain) => Self::UnsupportedResidencyDomain,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Ok => METRIC_STATUS_OK,
            Self::ParentNotFound => METRIC_STATUS_PARENT_NOT_FOUND,
            Self::BlockNotFound => METRIC_STATUS_BLOCK_NOT_FOUND,
            Self::InvalidBlock => METRIC_STATUS_INVALID_BLOCK,
            Self::CapacityExhausted => METRIC_STATUS_CAPACITY_EXHAUSTED,
            Self::AllocationFailed => METRIC_STATUS_ALLOCATION_FAILED,
            Self::OwnershipDegreeOverflow => METRIC_STATUS_OWNERSHIP_DEGREE_OVERFLOW,
            Self::IndexerInvariantViolation => METRIC_STATUS_INDEXER_INVARIANT_VIOLATION,
            Self::UnsupportedResidencyDomain => METRIC_STATUS_UNSUPPORTED_RESIDENCY_DOMAIN,
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

impl std::fmt::Display for EventStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Record and rate-limit diagnostics for a wire event that cannot enter the
/// canonical residency model. The metric label has fixed cardinality.
pub fn record_unsupported_residency_event(metrics: Option<&KvIndexerMetrics>, event: &RouterEvent) {
    if let Some(metrics) = metrics {
        metrics.increment_event_applied(
            EventKind::of(&event.event.data),
            Err(KvCacheEventError::UnsupportedResidencyDomain),
        );
    }

    let ignored_count = IGNORED_RESIDENCY_EVENT_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if ignored_count.is_power_of_two() {
        tracing::warn!(
            ignored_count,
            worker_id = event.worker_id,
            dp_rank = event.event.dp_rank,
            storage_tier = ?event.storage_tier,
            "Ignoring KV event with unsupported residency domain"
        );
    }
}

/// Lightweight, `Copy` discriminant for KV event warnings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventWarningKind {
    DuplicateStore,
}

impl EventWarningKind {
    pub const ALL: [Self; 1] = [Self::DuplicateStore];

    pub fn label(self) -> &'static str {
        match self {
            Self::DuplicateStore => METRIC_WARNING_DUPLICATE_STORE,
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CkfMutationKind {
    UnknownRemove,
    CapacityExhausted,
}

impl CkfMutationKind {
    pub const ALL: [Self; 2] = [Self::UnknownRemove, Self::CapacityExhausted];

    pub fn label(self) -> &'static str {
        match self {
            Self::UnknownRemove => METRIC_CKF_MUTATION_UNKNOWN_REMOVE,
            Self::CapacityExhausted => METRIC_CKF_MUTATION_CAPACITY_EXHAUSTED,
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

impl std::fmt::Display for CkfMutationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl std::fmt::Display for EventWarningKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(feature = "bench")]
pub(crate) struct ShardedIndexerCounters {
    pub(crate) find_match_dispatches: AtomicU64,
    pub(crate) find_match_early_returns: AtomicU64,
    pub(crate) anchor_installs: AtomicU64,
    pub(crate) anchor_reuses: AtomicU64,
    pub(crate) remove_broadcasts: AtomicU64,
}

#[cfg(feature = "bench")]
impl ShardedIndexerCounters {
    fn new() -> Self {
        Self {
            find_match_dispatches: AtomicU64::new(0),
            find_match_early_returns: AtomicU64::new(0),
            anchor_installs: AtomicU64::new(0),
            anchor_reuses: AtomicU64::new(0),
            remove_broadcasts: AtomicU64::new(0),
        }
    }
}

#[cfg(feature = "bench")]
pub(crate) struct ShardedIndexerTiming {
    pub(crate) calls: AtomicU64,
    pub(crate) routing_ns: AtomicU64,
    pub(crate) shard_ns: AtomicU64,
}

#[cfg(feature = "bench")]
impl ShardedIndexerTiming {
    fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
            routing_ns: AtomicU64::new(0),
            shard_ns: AtomicU64::new(0),
        }
    }
}

#[cfg(feature = "bench")]
pub(crate) struct ShardedIndexerMetrics {
    pub(crate) counters: ShardedIndexerCounters,
    pub(crate) timing: ShardedIndexerTiming,
}

#[cfg(feature = "bench")]
impl ShardedIndexerMetrics {
    pub(crate) fn new() -> Self {
        Self {
            counters: ShardedIndexerCounters::new(),
            timing: ShardedIndexerTiming::new(),
        }
    }
}

/// Metric status labels.
pub const METRIC_STATUS_OK: &str = "ok";
pub const METRIC_STATUS_PARENT_NOT_FOUND: &str = "parent_block_not_found";
pub const METRIC_STATUS_BLOCK_NOT_FOUND: &str = "block_not_found";
pub const METRIC_STATUS_INVALID_BLOCK: &str = "invalid_block";
pub const METRIC_STATUS_CAPACITY_EXHAUSTED: &str = "capacity_exhausted";
pub const METRIC_STATUS_ALLOCATION_FAILED: &str = "allocation_failed";
pub const METRIC_STATUS_OWNERSHIP_DEGREE_OVERFLOW: &str = "ownership_degree_overflow";
pub const METRIC_STATUS_INDEXER_INVARIANT_VIOLATION: &str = "indexer_invariant_violation";
pub const METRIC_STATUS_UNSUPPORTED_RESIDENCY_DOMAIN: &str = "unsupported_residency_domain";

/// Metric event labels.
pub const METRIC_EVENT_STORED: &str = "stored";
pub const METRIC_EVENT_REMOVED: &str = "removed";
pub const METRIC_EVENT_CLEARED: &str = "cleared";

/// Metric warning labels.
pub const METRIC_WARNING_DUPLICATE_STORE: &str = "duplicate_store";

/// CKF mutation metric labels.
pub const METRIC_CKF_MUTATION_UNKNOWN_REMOVE: &str = "unknown_remove";
pub const METRIC_CKF_MUTATION_CAPACITY_EXHAUSTED: &str = "capacity_exhausted";

/// The counters behind one [`KvIndexerMetrics`] handle, shared by its clones.
struct IndexerCounters {
    events_applied: [[AtomicU64; EventStatus::ALL.len()]; EventKind::ALL.len()],
    event_warnings: [AtomicU64; EventWarningKind::ALL.len()],
    ckf_mutations: [AtomicU64; CkfMutationKind::ALL.len()],
}

impl IndexerCounters {
    fn new() -> Self {
        Self {
            events_applied: [const { [const { AtomicU64::new(0) }; EventStatus::ALL.len()] };
                EventKind::ALL.len()],
            event_warnings: [const { AtomicU64::new(0) }; EventWarningKind::ALL.len()],
            ckf_mutations: [const { AtomicU64::new(0) }; CkfMutationKind::ALL.len()],
        }
    }
}

/// Counters for the KV indexer: events applied by kind and status, event
/// warnings, and CKF mutation outcomes.
///
/// Cheap to clone; clones share one counter set. The indexer never exposes
/// them itself: a host registers a [`KvIndexerCounterCollector`] and attaches
/// the handle.
#[derive(Clone)]
pub struct KvIndexerMetrics {
    counters: Arc<IndexerCounters>,
}

impl Default for KvIndexerMetrics {
    fn default() -> Self {
        Self::new_unregistered()
    }
}

impl KvIndexerMetrics {
    /// A fresh counter set that no scrape reads until a host attaches it.
    pub fn new_unregistered() -> Self {
        Self {
            counters: Arc::new(IndexerCounters::new()),
        }
    }

    pub fn increment_event_applied(&self, kind: EventKind, result: Result<(), KvCacheEventError>) {
        self.counters.events_applied[kind.index()][EventStatus::of(result).index()]
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn increment_event_warning(&self, kind: EventWarningKind) {
        self.counters.event_warnings[kind.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// Events applied with this kind and status.
    pub fn events_applied(&self, kind: EventKind, status: EventStatus) -> u64 {
        self.counters.events_applied[kind.index()][status.index()].load(Ordering::Relaxed)
    }

    /// Warnings of this kind.
    pub fn event_warnings(&self, kind: EventWarningKind) -> u64 {
        self.counters.event_warnings[kind.index()].load(Ordering::Relaxed)
    }

    /// CKF mutation outcomes of this kind.
    pub fn ckf_mutations(&self, kind: CkfMutationKind) -> u64 {
        self.counters.ckf_mutations[kind.index()].load(Ordering::Relaxed)
    }

    /// A handle for one indexer thread's event loop.
    pub fn prebind(&self) -> PreBoundEventCounters {
        PreBoundEventCounters {
            counters: Arc::clone(&self.counters),
        }
    }
}

/// One indexer thread's handle on the counters, taken once at thread start.
///
/// Takes the `Copy` discriminants instead of label strings so the compiler
/// enforces exhaustiveness: a new [`EventKind`] or [`KvCacheEventError`]
/// variant is a compile error here.
pub struct PreBoundEventCounters {
    counters: Arc<IndexerCounters>,
}

impl PreBoundEventCounters {
    /// Count one applied event.
    #[inline]
    pub fn inc(&self, kind: EventKind, result: Result<(), KvCacheEventError>) {
        self.counters.events_applied[kind.index()][EventStatus::of(result).index()]
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_warning(&self, kind: EventWarningKind) {
        self.counters.event_warnings[kind.index()].fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_ckf_mutation(&self, kind: CkfMutationKind, count: u64) {
        if count == 0 {
            return;
        }
        self.counters.ckf_mutations[kind.index()].fetch_add(count, Ordering::Relaxed);
    }
}

#[cfg(feature = "metrics")]
pub use self::export::{IndexerCounterFamily, KvIndexerCounterCollector};

#[cfg(feature = "metrics")]
mod export {
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Weak};

    use parking_lot::Mutex;
    use prometheus::{
        IntCounterVec, Opts,
        core::{Collector, Desc},
        proto::MetricFamily,
    };

    use super::{CkfMutationKind, EventKind, EventStatus, EventWarningKind, IndexerCounters};

    /// One exported counter family of [`super::KvIndexerMetrics`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum IndexerCounterFamily {
        /// `event_type`, `status`: events applied to the index.
        EventsApplied,
        /// `warning_kind`: suspicious-but-valid events.
        EventWarnings,
        /// `outcome`: CKF block-level mutation outcomes.
        CkfMutations,
    }

    impl IndexerCounterFamily {
        pub const ALL: [Self; 3] = [Self::EventsApplied, Self::EventWarnings, Self::CkfMutations];

        /// The family's name when a host exports it without a hierarchy
        /// prefix, as the standalone indexer does.
        pub const fn name(self) -> &'static str {
            match self {
                Self::EventsApplied => "dynamo_kvrouter_kv_cache_events_applied",
                Self::EventWarnings => "dynamo_kvrouter_kv_cache_event_warnings",
                Self::CkfMutations => "dynamo_kvrouter_ckf_mutation_total",
            }
        }

        pub const fn help(self) -> &'static str {
            match self {
                Self::EventsApplied => "Total number of KV cache events applied to index",
                Self::EventWarnings => {
                    "Total number of suspicious KV cache events seen by the router indexer"
                }
                Self::CkfMutations => "Total number of CKF block-level mutation outcomes",
            }
        }

        pub const fn labels(self) -> &'static [&'static str] {
            match self {
                Self::EventsApplied => &["event_type", "status"],
                Self::EventWarnings => &["warning_kind"],
                Self::CkfMutations => &["outcome"],
            }
        }
    }

    /// Exports one counter family from every attached [`super::KvIndexerMetrics`]
    /// at scrape time. Handles that share a collector add up; a handle whose
    /// last clone dropped leaves the scrape.
    #[derive(Clone)]
    pub struct KvIndexerCounterCollector {
        family: IndexerCounterFamily,
        opts: Opts,
        descs: Vec<Desc>,
        sources: Arc<Mutex<Vec<Weak<IndexerCounters>>>>,
    }

    impl KvIndexerCounterCollector {
        /// A collector for `family` under `opts`; the host supplies the name
        /// and any const labels.
        pub fn new(family: IndexerCounterFamily, opts: Opts) -> Result<Self, prometheus::Error> {
            let descs = IntCounterVec::new(opts.clone(), family.labels())?
                .desc()
                .into_iter()
                .cloned()
                .collect();
            Ok(Self {
                family,
                opts,
                descs,
                sources: Arc::new(Mutex::new(Vec::new())),
            })
        }

        /// A collector for `family` under its default name.
        pub fn with_default_name(family: IndexerCounterFamily) -> Result<Self, prometheus::Error> {
            Self::new(family, Opts::new(family.name(), family.help()))
        }

        pub fn family(&self) -> IndexerCounterFamily {
            self.family
        }

        /// Put `metrics` in the scrape until its last clone drops.
        pub fn attach(&self, metrics: &super::KvIndexerMetrics) {
            let mut sources = self.sources.lock();
            sources.retain(|source| source.strong_count() > 0);
            sources.push(Arc::downgrade(&metrics.counters));
        }

        /// Register every family under its default name and attach `metrics`.
        pub fn register_all(
            registry: &prometheus::Registry,
            metrics: &super::KvIndexerMetrics,
        ) -> Result<(), prometheus::Error> {
            for family in IndexerCounterFamily::ALL {
                let collector = Self::with_default_name(family)?;
                collector.attach(metrics);
                registry.register(Box::new(collector))?;
            }
            Ok(())
        }
    }

    impl Collector for KvIndexerCounterCollector {
        fn desc(&self) -> Vec<&Desc> {
            self.descs.iter().collect()
        }

        fn collect(&self) -> Vec<MetricFamily> {
            // A fresh vector per scrape: concurrent scrapes stay independent.
            let vec = IntCounterVec::new(self.opts.clone(), self.family.labels())
                .expect("validated indexer counter options");
            let sources: Vec<Arc<IndexerCounters>> = {
                let mut sources = self.sources.lock();
                sources.retain(|source| source.strong_count() > 0);
                sources.iter().filter_map(Weak::upgrade).collect()
            };
            match self.family {
                IndexerCounterFamily::EventsApplied => {
                    for kind in EventKind::ALL {
                        for status in EventStatus::ALL {
                            let counter = vec.with_label_values(&[kind.label(), status.label()]);
                            for source in &sources {
                                counter.inc_by(
                                    source.events_applied[kind.index()][status.index()]
                                        .load(Ordering::Relaxed),
                                );
                            }
                        }
                    }
                }
                IndexerCounterFamily::EventWarnings => {
                    for kind in EventWarningKind::ALL {
                        let counter = vec.with_label_values(&[kind.label()]);
                        for source in &sources {
                            counter.inc_by(
                                source.event_warnings[kind.index()].load(Ordering::Relaxed),
                            );
                        }
                    }
                }
                IndexerCounterFamily::CkfMutations => {
                    for kind in CkfMutationKind::ALL {
                        let counter = vec.with_label_values(&[kind.label()]);
                        for source in &sources {
                            counter
                                .inc_by(source.ckf_mutations[kind.index()].load(Ordering::Relaxed));
                        }
                    }
                }
            }
            vec.collect()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::KvIndexerMetrics;
        use dynamo_router_types::protocols::KvCacheEventError;
        use prometheus::{Registry, TextEncoder};

        fn output(registry: &Registry) -> String {
            TextEncoder::new()
                .encode_to_string(&registry.gather())
                .unwrap()
        }

        #[test]
        fn register_all_exports_every_series_under_the_default_names() {
            let registry = Registry::new();
            let a = KvIndexerMetrics::new_unregistered();
            KvIndexerCounterCollector::register_all(&registry, &a).unwrap();

            a.increment_event_applied(EventKind::Stored, Ok(()));
            a.increment_event_applied(EventKind::Stored, Ok(()));
            a.increment_event_applied(EventKind::Removed, Err(KvCacheEventError::BlockNotFound));
            a.prebind().inc_warning(EventWarningKind::DuplicateStore);
            a.prebind()
                .inc_ckf_mutation(CkfMutationKind::CapacityExhausted, 3);

            let text = output(&registry);
            for line in [
                "dynamo_kvrouter_kv_cache_events_applied{event_type=\"stored\",status=\"ok\"} 2",
                "dynamo_kvrouter_kv_cache_events_applied{event_type=\"removed\",status=\"block_not_found\"} 1",
                "dynamo_kvrouter_kv_cache_events_applied{event_type=\"cleared\",status=\"ok\"} 0",
                "dynamo_kvrouter_kv_cache_event_warnings{warning_kind=\"duplicate_store\"} 1",
                "dynamo_kvrouter_ckf_mutation_total{outcome=\"capacity_exhausted\"} 3",
                "dynamo_kvrouter_ckf_mutation_total{outcome=\"unknown_remove\"} 0",
            ] {
                assert!(text.contains(line), "missing {line}\n{text}");
            }
        }

        #[test]
        fn handles_sharing_a_collector_add_up_until_dropped() {
            let registry = Registry::new();
            let collector =
                KvIndexerCounterCollector::with_default_name(IndexerCounterFamily::EventWarnings)
                    .unwrap();
            registry.register(Box::new(collector.clone())).unwrap();
            let a = KvIndexerMetrics::new_unregistered();
            let b = KvIndexerMetrics::new_unregistered();
            collector.attach(&a);
            collector.attach(&b);
            a.increment_event_warning(EventWarningKind::DuplicateStore);
            b.increment_event_warning(EventWarningKind::DuplicateStore);
            b.increment_event_warning(EventWarningKind::DuplicateStore);
            let line =
                "dynamo_kvrouter_kv_cache_event_warnings{warning_kind=\"duplicate_store\"} 3";
            assert!(output(&registry).contains(line));

            let b_clone = b.clone();
            drop(b);
            assert!(
                output(&registry).contains(line),
                "a live clone keeps b in the scrape"
            );
            drop(b_clone);
            let line =
                "dynamo_kvrouter_kv_cache_event_warnings{warning_kind=\"duplicate_store\"} 1";
            assert!(
                output(&registry).contains(line),
                "the last clone takes b out"
            );
        }

        #[test]
        fn const_labels_from_the_host_are_kept() {
            let registry = Registry::new();
            let opts = Opts::new("dynamo_component_kv_cache_event_warnings", "help")
                .const_label("dynamo_namespace", "ns");
            let collector =
                KvIndexerCounterCollector::new(IndexerCounterFamily::EventWarnings, opts).unwrap();
            registry.register(Box::new(collector.clone())).unwrap();
            let metrics = KvIndexerMetrics::new_unregistered();
            collector.attach(&metrics);
            assert!(output(&registry).contains(
                "dynamo_component_kv_cache_event_warnings{dynamo_namespace=\"ns\",warning_kind=\"duplicate_store\"} 0"
            ));
        }
    }
}
