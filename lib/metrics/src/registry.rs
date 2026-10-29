// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The hierarchy and its registries: `MetricsHierarchy` names a level (runtime, namespace,
//! component, endpoint) and owns a `MetricsRegistry`; `Metrics<H>` is the per-level handle
//! that creates metrics through [`crate::create_metric`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::Context as _;
use prometheus::Encoder;

use crate::metric::create_metric;

/// Wrapper struct that provides access to metrics functionality
/// This struct is accessed via the `.metrics()` method on DistributedRuntime, Namespace, Component, and Endpoint
pub struct Metrics<H: MetricsHierarchy> {
    hierarchy: H,
}

impl<H: MetricsHierarchy> Metrics<H> {
    pub fn new(hierarchy: H) -> Self {
        Self { hierarchy }
    }

    // TODO: Add support for additional Prometheus metric types:
    // - Counter: ✅ IMPLEMENTED - create_counter()
    // - CounterVec: ✅ IMPLEMENTED - create_countervec()
    // - Gauge: ✅ IMPLEMENTED - create_gauge()
    // - GaugeVec: ✅ IMPLEMENTED - create_gaugevec()
    // - GaugeHistogram: create_gauge_histogram() - for gauge histograms
    // - Histogram: ✅ IMPLEMENTED - create_histogram()
    // - HistogramVec: ✅ IMPLEMENTED - create_histogramvec()
    // - Info: create_info() - for info metrics with labels
    // - IntCounter: ✅ IMPLEMENTED - create_intcounter()
    // - IntCounterVec: ✅ IMPLEMENTED - create_intcountervec()
    // - IntGauge: ✅ IMPLEMENTED - create_intgauge()
    // - IntGaugeVec: ✅ IMPLEMENTED - create_intgaugevec()
    // - Stateset: create_stateset() - for state-based metrics
    // - Summary: create_summary() - for quantiles and sum/count metrics
    // - SummaryVec: create_summary_vec() - for labeled summaries
    // - Untyped: create_untyped() - for untyped metrics
    //
    // NOTE: The order of create_* methods below is mirrored in lib/bindings/python/rust/lib.rs::Metrics
    // Keep them synchronized when adding new metric types

    /// Create a Counter metric
    pub fn create_counter(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::Counter> {
        create_metric(&self.hierarchy, name, description, labels, None, None)
    }

    /// Create a CounterVec metric with label names (for dynamic labels)
    pub fn create_countervec(
        &self,
        name: &str,
        description: &str,
        const_labels: &[&str],
        const_label_values: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::CounterVec> {
        create_metric(
            &self.hierarchy,
            name,
            description,
            const_label_values,
            None,
            Some(const_labels),
        )
    }

    /// Create a Gauge metric
    pub fn create_gauge(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::Gauge> {
        create_metric(&self.hierarchy, name, description, labels, None, None)
    }

    /// Create a GaugeVec metric with label names (for dynamic labels)
    pub fn create_gaugevec(
        &self,
        name: &str,
        description: &str,
        const_labels: &[&str],
        const_label_values: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::GaugeVec> {
        create_metric(
            &self.hierarchy,
            name,
            description,
            const_label_values,
            None,
            Some(const_labels),
        )
    }

    /// Create a Histogram metric with custom buckets
    pub fn create_histogram(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
        buckets: Option<Vec<f64>>,
    ) -> anyhow::Result<prometheus::Histogram> {
        create_metric(&self.hierarchy, name, description, labels, buckets, None)
    }

    /// Create a HistogramVec metric with custom buckets and label names.
    pub fn create_histogramvec(
        &self,
        name: &str,
        description: &str,
        label_names: &[&str],
        labels: &[(&str, &str)],
        buckets: Option<Vec<f64>>,
    ) -> anyhow::Result<prometheus::HistogramVec> {
        create_metric(
            &self.hierarchy,
            name,
            description,
            labels,
            buckets,
            Some(label_names),
        )
    }

    /// Create an IntCounter metric
    pub fn create_intcounter(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::IntCounter> {
        create_metric(&self.hierarchy, name, description, labels, None, None)
    }

    /// Create an IntCounterVec metric with label names (for dynamic labels)
    pub fn create_intcountervec(
        &self,
        name: &str,
        description: &str,
        const_labels: &[&str],
        const_label_values: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::IntCounterVec> {
        create_metric(
            &self.hierarchy,
            name,
            description,
            const_label_values,
            None,
            Some(const_labels),
        )
    }

    /// Create an IntGauge metric
    pub fn create_intgauge(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::IntGauge> {
        create_metric(&self.hierarchy, name, description, labels, None, None)
    }

    /// Create an IntGaugeVec metric with label names (for dynamic labels)
    pub fn create_intgaugevec(
        &self,
        name: &str,
        description: &str,
        const_labels: &[&str],
        const_label_values: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::IntGaugeVec> {
        create_metric(
            &self.hierarchy,
            name,
            description,
            const_label_values,
            None,
            Some(const_labels),
        )
    }

    /// Get metrics in Prometheus text format
    pub fn prometheus_expfmt(&self) -> anyhow::Result<String> {
        self.hierarchy
            .get_metrics_registry()
            .prometheus_expfmt_combined()
    }
}

/// This trait should be implemented by all metric registries, including Prometheus, Envy, OpenTelemetry, and others.
/// It offers a unified interface for creating and managing metrics, organizing sub-registries, and
/// generating output in Prometheus text format.
pub trait MetricsHierarchy: Send + Sync {
    // ========================================================================
    // Required methods - must be implemented by all types
    // ========================================================================

    /// Get the name of this hierarchy (without any hierarchy prefix)
    fn basename(&self) -> String;

    /// Get the parent hierarchies as actual objects (not strings)
    /// Returns a vector of hierarchy references, ordered from root to immediate parent.
    /// For example, an Endpoint would return [DRT, Namespace, Component].
    fn parent_hierarchies(&self) -> Vec<&dyn MetricsHierarchy>;

    /// Get a reference to this hierarchy's metrics registry
    fn get_metrics_registry(&self) -> &MetricsRegistry;

    // ========================================================================
    // Provided methods - have default implementations
    // ========================================================================

    /// Get the connection ID (discovery instance ID) for this hierarchy level.
    ///
    /// Returns `Some(id)` when the hierarchy has access to the DistributedRuntime
    /// (e.g. Namespace, Component, Endpoint). Used by `create_metric()` to auto-inject
    /// the `worker_id` label. Returns `None` by default.
    fn connection_id(&self) -> Option<u64> {
        None
    }

    /// Access the metrics interface for this hierarchy
    /// This is a provided method that works for any type implementing MetricsHierarchy
    fn metrics(&self) -> Metrics<&Self>
    where
        Self: Sized,
    {
        Metrics::new(self)
    }
}

// Blanket implementation for references to types that implement MetricsHierarchy
impl<T: MetricsHierarchy + ?Sized> MetricsHierarchy for &T {
    fn basename(&self) -> String {
        (**self).basename()
    }

    fn parent_hierarchies(&self) -> Vec<&dyn MetricsHierarchy> {
        (**self).parent_hierarchies()
    }

    fn get_metrics_registry(&self) -> &MetricsRegistry {
        (**self).get_metrics_registry()
    }

    fn connection_id(&self) -> Option<u64> {
        (**self).connection_id()
    }
}

/// Type alias for runtime callback functions to reduce complexity
///
/// This type represents an Arc-wrapped callback function that can be:
/// - Shared efficiently across multiple threads and contexts
/// - Cloned without duplicating the underlying closure
/// - Used in generic contexts requiring 'static lifetime
///
/// The Arc wrapper is included in the type to make sharing explicit.
pub type PrometheusUpdateCallback = Arc<dyn Fn() -> anyhow::Result<()> + Send + Sync + 'static>;

/// One collection: the merged families, plus the UNIT metadata that came with
/// the typed ones.
///
/// Kept beside the families because `prometheus::proto::MetricFamily` has no
/// unit field, and the compatibility spec requires the unit reach OTLP when the
/// source declares one.
pub struct CollectedFamilies {
    pub families: Vec<prometheus::proto::MetricFamily>,
    /// Family name -> Prometheus unit word, for families that declared one.
    pub units: HashMap<String, String>,
    /// Family name -> Prometheus type word. Carried because the proto model has
    /// no Info variant: an Info family arrives here as a gauge, and the spec
    /// converts it to a non-monotonic Sum rather than a Gauge.
    pub prom_types: HashMap<String, String>,
}

/// Type alias for exposition text callback functions that return Prometheus text
pub type PrometheusExpositionFormatCallback =
    Arc<dyn Fn() -> anyhow::Result<String> + Send + Sync + 'static>;

/// Returns already-built families, so engine metrics reach the structured
/// path without being rendered to text and parsed back.
pub type PrometheusTypedCallback =
    Arc<dyn Fn() -> anyhow::Result<Vec<crate::prom_typed::BuiltFamily>> + Send + Sync + 'static>;

/// Structure to hold Prometheus registries and associated callbacks for a given hierarchy.
///
/// All fields are Arc-wrapped, so cloning shares state. This ensures metrics registered
/// on cloned instances (e.g., cloned Client/Endpoint) are visible to the original.
#[derive(Clone)]
pub struct MetricsRegistry {
    /// The Prometheus registry for this hierarchy.
    /// Arc-wrapped so clones share the same registry (metrics registered on clones are visible everywhere).
    pub prometheus_registry: Arc<std::sync::RwLock<prometheus::Registry>>,

    /// Child registries included when emitting combined `/metrics` output.
    ///
    /// Why this exists:
    /// - Previously, `create_metric()` registered every collector into *all* parent registries
    ///   (Endpoint → Component → Namespace → DRT) so scraping the root registry included everything.
    /// - That fan-out caused Prometheus collisions when different endpoints tried to register the
    ///   same metric name with different const-labels (descriptor mismatch).
    ///
    /// We now register metrics only into the local hierarchy registry to avoid collisions.
    /// `child_registries` rebuilds “what to scrape” as a tree of registries so `/metrics` can:
    /// - traverse registries recursively,
    /// - merge metric families into one exposition payload,
    /// - warn/drop exact duplicate series, while allowing same metric name with different labels.
    child_registries: Arc<std::sync::RwLock<Vec<MetricsRegistry>>>,

    /// Update callbacks invoked before metrics are scraped.
    /// Wrapped in Arc to preserve callbacks across clones (prevents callback loss when MetricsRegistry is cloned).
    pub prometheus_update_callbacks: Arc<std::sync::RwLock<Vec<PrometheusUpdateCallback>>>,

    /// Callbacks returning engine exposition text; these feed `/metrics` only.
    #[allow(clippy::type_complexity)]
    pub prometheus_expfmt_callbacks:
        Arc<std::sync::RwLock<Vec<PrometheusExpositionFormatCallback>>>,

    /// Callbacks returning engine metric families; these feed OTLP only.
    /// Wrapped in Arc to preserve callbacks across clones (e.g. vLLM callbacks
    /// registered at Endpoint remain reachable at DRT).
    pub prometheus_typed_callbacks: Arc<std::sync::RwLock<Vec<PrometheusTypedCallback>>>,
}

impl std::fmt::Debug for MetricsRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetricsRegistry")
            .field("prometheus_registry", &"<RwLock<Registry>>")
            .field(
                "prometheus_update_callbacks",
                &format!(
                    "<RwLock<Vec<Callback>>> with {} callbacks",
                    self.prometheus_update_callbacks.read().unwrap().len()
                ),
            )
            .field(
                "prometheus_expfmt_callbacks",
                &format!(
                    "<RwLock<Vec<Callback>>> with {} callbacks",
                    self.prometheus_expfmt_callbacks.read().unwrap().len()
                ),
            )
            .field(
                "prometheus_typed_callbacks",
                &format!(
                    "<RwLock<Vec<Callback>>> with {} callbacks",
                    self.prometheus_typed_callbacks.read().unwrap().len()
                ),
            )
            .finish()
    }
}

impl MetricsRegistry {
    /// Create a new metrics registry with an empty Prometheus registry and callback lists
    pub fn new() -> Self {
        Self {
            prometheus_registry: Arc::new(std::sync::RwLock::new(prometheus::Registry::new())),
            child_registries: Arc::new(std::sync::RwLock::new(Vec::new())),
            prometheus_update_callbacks: Arc::new(std::sync::RwLock::new(Vec::new())),
            prometheus_expfmt_callbacks: Arc::new(std::sync::RwLock::new(Vec::new())),
            prometheus_typed_callbacks: Arc::new(std::sync::RwLock::new(Vec::new())),
        }
    }

    /// Add a child registry to be included in combined /metrics output.
    ///
    /// Dedup is by underlying Prometheus registry pointer, so repeated registration via clones is safe.
    pub fn add_child_registry(&self, child: &MetricsRegistry) {
        let child_ptr = Arc::as_ptr(&child.prometheus_registry);
        let mut guard = self.child_registries.write().unwrap();
        if guard
            .iter()
            .any(|r| Arc::as_ptr(&r.prometheus_registry) == child_ptr)
        {
            return;
        }
        guard.push(child.clone());
    }

    fn registries_for_combined_scrape(&self) -> Vec<MetricsRegistry> {
        // Traverse child registries recursively so `prometheus_expfmt()` on any hierarchy
        // (DRT/namespace/component/endpoint) includes metrics from its descendants.
        //
        // Dedup by underlying Prometheus registry pointer so multiple paths (e.g. also registering
        // directly on the root) won't duplicate output.
        fn visit(
            registry: &MetricsRegistry,
            out: &mut Vec<MetricsRegistry>,
            seen: &mut HashSet<*const std::sync::RwLock<prometheus::Registry>>,
        ) {
            let ptr = Arc::as_ptr(&registry.prometheus_registry);
            if !seen.insert(ptr) {
                return;
            }

            out.push(registry.clone());

            let children: Vec<MetricsRegistry> = registry
                .child_registries
                .read()
                .unwrap()
                .iter()
                .cloned()
                .collect();
            for child in children {
                visit(&child, out, seen);
            }
        }

        let mut out = Vec::new();
        let mut seen: HashSet<*const std::sync::RwLock<prometheus::Registry>> = HashSet::new();
        visit(self, &mut out, &mut seen);
        out
    }

    /// Combine metrics across this registry and all registered children into one Prometheus exposition output.
    ///
    /// Engine metrics are appended as the text the engine produced. This is the
    /// scrape path and is deliberately independent of the typed path that feeds
    /// OTLP: `/metrics` output stays byte-for-byte what the engine emitted.
    ///
    /// - Families are merged by name; HELP and TYPE must match.
    /// - Multiple series for the same name are allowed if labels differ.
    /// - Exact duplicate series (same name + identical label pairs) are warned and dropped.
    pub fn prometheus_expfmt_combined(&self) -> anyhow::Result<String> {
        let registries = self.registries_for_combined_scrape();
        run_update_callbacks(&registries);

        let merged = FamilyMerger::from_gathered(&registries)?.into_sorted();

        let encoder = prometheus::TextEncoder::new();
        let mut buffer = Vec::new();
        encoder.encode(&merged, &mut buffer)?;
        let mut result = String::from_utf8(buffer)?;

        // Append expfmt callbacks deterministically in registry order.
        let mut expfmt = String::new();
        for registry in &registries {
            let text = registry.execute_expfmt_callbacks();
            if !text.is_empty() {
                if !expfmt.is_empty() && !expfmt.ends_with('\n') {
                    expfmt.push('\n');
                }
                expfmt.push_str(&text);
            }
        }

        if !expfmt.is_empty() {
            if !result.ends_with('\n') {
                result.push('\n');
            }
            result.push_str(&expfmt);
        }

        Ok(result)
    }

    /// Every metric family: Dynamo's own from `gather()`, plus engine families
    /// delivered typed. This is the OTLP export path.
    ///
    /// Engine metrics cross the binding as a structure rather than as
    /// exposition text, so their name, type and help arrive authoritative
    /// instead of being re-derived by a parser.
    ///
    /// Each call re-executes the typed callbacks, which cross into Python and
    /// take the GIL, so callers should collect on a deliberate cadence.
    pub fn metric_families_combined(&self) -> anyhow::Result<CollectedFamilies> {
        let registries = self.registries_for_combined_scrape();
        run_update_callbacks(&registries);

        let mut merger = FamilyMerger::from_gathered(&registries)?;

        // The unit travels beside the family, keyed by name: the proto model
        // has no unit field, so it would otherwise be dropped here.
        let mut units: HashMap<String, String> = HashMap::new();
        let mut prom_types: HashMap<String, String> = HashMap::new();
        for registry in &registries {
            for built in registry.execute_typed_callbacks() {
                let name = built.family.name().to_string();
                // Publish the side metadata only once the family is accepted.
                // The maps are keyed by name, so metadata from a family the
                // merger rejects would otherwise be applied to the family that
                // did survive under that name -- a rejected `info` could make a
                // surviving gauge encode as a non-monotonic Sum.
                if let Err(error) = merger.add_family(built.family) {
                    tracing::warn!(metric_name = %name, %error, "skipping typed metric family");
                    continue;
                }
                // First writer wins, matching how the merger itself resolves a
                // conflict: the first family fixes help and type, and later
                // ones must agree. A later family can be accepted and still
                // have every sample dropped as a duplicate series, so letting
                // it overwrite the unit would label the surviving data with a
                // unit from a source that contributed none of it.
                if !built.unit.is_empty() {
                    units.entry(name.clone()).or_insert(built.unit);
                }
                prom_types.entry(name).or_insert(built.prom_type);
            }
        }

        Ok(CollectedFamilies {
            families: merger.into_sorted(),
            units,
            prom_types,
        })
    }

    /// Add a callback function that receives a reference to any MetricsHierarchy
    pub fn add_update_callback(&self, callback: PrometheusUpdateCallback) {
        self.prometheus_update_callbacks
            .write()
            .unwrap()
            .push(callback);
    }

    /// Add an exposition text callback that returns Prometheus text.
    ///
    /// This feeds `/metrics` only. Engines that also want to be exported over
    /// OTLP must register a typed callback via [`Self::add_typed_callback`].
    pub fn add_expfmt_callback(&self, callback: PrometheusExpositionFormatCallback) {
        self.prometheus_expfmt_callbacks
            .write()
            .unwrap()
            .push(callback);
    }

    /// Execute all exposition text callbacks and return their concatenated text
    pub fn execute_expfmt_callbacks(&self) -> String {
        // Snapshot under the lock, then invoke without it held; see
        // `execute_typed_callbacks` for why holding it across the calls
        // deadlocks against registration from Python.
        let callbacks: Vec<PrometheusExpositionFormatCallback> = self
            .prometheus_expfmt_callbacks
            .read()
            .unwrap()
            .iter()
            .cloned()
            .collect();
        let mut result = String::new();
        for callback in callbacks.iter() {
            match callback() {
                Ok(text) => {
                    if !text.is_empty() {
                        if !result.is_empty() && !result.ends_with('\n') {
                            result.push('\n');
                        }
                        result.push_str(&text);
                    }
                }
                Err(e) => {
                    tracing::error!("Error executing exposition text callback: {e}");
                }
            }
        }
        result
    }

    /// Register a callback that returns typed families directly.
    ///
    /// This is how engine metrics reach OTLP.
    pub fn add_typed_callback(&self, callback: PrometheusTypedCallback) {
        self.prometheus_typed_callbacks
            .write()
            .unwrap()
            .push(callback);
    }

    /// Collect from every typed callback, logging and skipping failures so one
    /// broken engine cannot empty the whole collection.
    pub fn execute_typed_callbacks(&self) -> Vec<crate::prom_typed::BuiltFamily> {
        // Snapshot under the lock, then invoke without it held. The callbacks
        // take the Python GIL, and registration arrives from Python holding the
        // GIL and blocking on `write()` -- holding the read lock across the
        // calls would deadlock those two against each other.
        let callbacks: Vec<PrometheusTypedCallback> = self
            .prometheus_typed_callbacks
            .read()
            .unwrap()
            .iter()
            .cloned()
            .collect();

        callbacks
            .iter()
            .filter_map(|callback| match callback() {
                Ok(families) => Some(families),
                Err(error) => {
                    tracing::warn!(%error, "typed metrics callback failed");
                    None
                }
            })
            .flatten()
            .collect()
    }

    /// Execute all update callbacks and return their results
    pub fn execute_update_callbacks(&self) -> Vec<anyhow::Result<()>> {
        // Snapshot before invoking, for the same reason as the typed and
        // exposition callbacks: a callback may take the Python GIL or register
        // another callback, and holding the read lock across the call deadlocks
        // against a registration that holds the GIL and wants the write lock.
        let callbacks: Vec<PrometheusUpdateCallback> = self
            .prometheus_update_callbacks
            .read()
            .unwrap()
            .iter()
            .cloned()
            .collect();

        callbacks.iter().map(|callback| callback()).collect()
    }

    /// Add a Prometheus metric collector to this registry
    pub fn add_metric(
        &self,
        collector: Box<dyn prometheus::core::Collector>,
    ) -> anyhow::Result<()> {
        self.prometheus_registry
            .write()
            .unwrap()
            .register(collector)
            .map_err(|e| anyhow::anyhow!("Failed to register metric: {}", e))
    }

    /// Add a Prometheus metric collector, logging a warning on failure instead of returning an error.
    pub fn add_metric_or_warn(&self, collector: Box<dyn prometheus::core::Collector>, name: &str) {
        if let Err(e) = self.add_metric(collector) {
            tracing::warn!(error = %e, metric = name, "Failed to register metric");
        }
    }

    /// Get a read guard to the Prometheus registry for scraping
    pub fn get_prometheus_registry(&self) -> std::sync::RwLockReadGuard<'_, prometheus::Registry> {
        self.prometheus_registry.read().unwrap()
    }

    /// Returns true if a metric with the given name already exists in the Prometheus registry
    pub fn has_metric_named(&self, metric_name: &str) -> bool {
        self.prometheus_registry
            .read()
            .unwrap()
            .gather()
            .iter()
            .any(|mf| mf.name() == metric_name)
    }
}

/// Run every registry's update callbacks, logging rather than propagating
/// failures so one broken callback cannot poison the whole collection.
fn run_update_callbacks(registries: &[MetricsRegistry]) {
    for registry in registries {
        for result in registry.execute_update_callbacks() {
            if let Err(error) = result {
                tracing::error!(%error, "metrics update callback failed");
            }
        }
    }
}

/// Merges metric families from several sources under one set of rules.
///
/// Extracted so the text path (`prometheus_expfmt_combined`) and the structured
/// path (`metric_families_combined`) cannot drift: a series accepted by one must
/// be accepted by the other, or `/metrics` and OTLP would disagree.
struct FamilyMerger {
    by_name: HashMap<String, prometheus::proto::MetricFamily>,
    seen_series: HashSet<String>,
}

impl FamilyMerger {
    fn from_gathered(registries: &[MetricsRegistry]) -> anyhow::Result<Self> {
        let mut merger = Self {
            by_name: HashMap::new(),
            seen_series: HashSet::new(),
        };
        for (idx, registry) in registries.iter().enumerate() {
            for family in registry.get_prometheus_registry().gather() {
                merger
                    .add_family(family)
                    .with_context(|| format!("gathering from registry {idx}"))?;
            }
        }
        Ok(merger)
    }

    fn add_family(&mut self, mut family: prometheus::proto::MetricFamily) -> anyhow::Result<()> {
        let name = family.name().to_string();

        let entry = self.by_name.entry(name.clone()).or_insert_with(|| {
            let mut out = prometheus::proto::MetricFamily::new();
            out.set_name(name.clone());
            out.set_help(family.help().to_string());
            out.set_field_type(family.get_field_type());
            out
        });

        if entry.help() != family.help() || entry.get_field_type() != family.get_field_type() {
            return Err(anyhow::anyhow!(
                "Metric family '{}' has inconsistent help/type across sources",
                name
            ));
        }

        for metric in family.take_metric() {
            let mut labels: Vec<(String, String)> = metric
                .get_label()
                .iter()
                .map(|lp| (lp.name().to_string(), lp.value().to_string()))
                .collect();
            labels.sort_by(|(ka, va), (kb, vb)| (ka, va).cmp(&(kb, vb)));

            let key = format!(
                "{}|{}",
                name,
                labels
                    .iter()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect::<Vec<_>>()
                    .join(",")
            );

            if !self.seen_series.insert(key) {
                tracing::warn!(
                    metric_name = %name,
                    labels = ?labels,
                    "Duplicate Prometheus series while merging; dropping later sample"
                );
                continue;
            }

            entry.mut_metric().push(metric);
        }

        Ok(())
    }

    fn into_sorted(self) -> Vec<prometheus::proto::MetricFamily> {
        let mut merged: Vec<prometheus::proto::MetricFamily> = self.by_name.into_values().collect();
        merged.sort_by(|a, b| a.name().cmp(b.name()));
        merged
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test_metricsregistry_units {
    use super::*;
    use crate::names::build_component_metric_name;

    #[test]
    fn test_build_component_metric_name_with_prefix() {
        // Test that build_component_metric_name correctly prepends the dynamo_component prefix
        let result = build_component_metric_name("requests");
        assert_eq!(result, "dynamo_component_requests");

        let result = build_component_metric_name("counter");
        assert_eq!(result, "dynamo_component_counter");
    }

    #[test]
    fn test_parse_prometheus_metric() {
        use crate::test_helpers::parse_prometheus_metric;
        use std::collections::HashMap;

        // Test parsing a metric with labels
        let line = "http_requests_total{method=\"GET\",status=\"200\"} 1234";
        let parsed = parse_prometheus_metric(line);
        assert!(parsed.is_some());

        let (name, labels, value) = parsed.unwrap();
        assert_eq!(name, "http_requests_total");

        let mut expected_labels = HashMap::new();
        expected_labels.insert("method".to_string(), "GET".to_string());
        expected_labels.insert("status".to_string(), "200".to_string());
        assert_eq!(labels, expected_labels);

        assert_eq!(value, 1234.0);

        // Test parsing a metric without labels
        let line = "cpu_usage 98.5";
        let parsed = parse_prometheus_metric(line);
        assert!(parsed.is_some());

        let (name, labels, value) = parsed.unwrap();
        assert_eq!(name, "cpu_usage");
        assert!(labels.is_empty());
        assert_eq!(value, 98.5);

        // Test parsing a metric with float value
        let line = "response_time{service=\"api\"} 0.123";
        let parsed = parse_prometheus_metric(line);
        assert!(parsed.is_some());

        let (name, labels, value) = parsed.unwrap();
        assert_eq!(name, "response_time");

        let mut expected_labels = HashMap::new();
        expected_labels.insert("service".to_string(), "api".to_string());
        assert_eq!(labels, expected_labels);

        assert_eq!(value, 0.123);

        // Test parsing invalid lines
        assert!(parse_prometheus_metric("").is_none()); // Empty line
        assert!(parse_prometheus_metric("# HELP metric description").is_none()); // Help text
        assert!(parse_prometheus_metric("# TYPE metric counter").is_none()); // Type definition
        assert!(parse_prometheus_metric("metric_name").is_none()); // No value

        println!("✓ Prometheus metric parsing works correctly!");
    }

    #[test]
    fn test_metrics_registry_entry_callbacks() {
        use crate::MetricsRegistry;
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Test 1: Basic callback execution with counter increments
        {
            let registry = MetricsRegistry::new();
            let counter = Arc::new(AtomicUsize::new(0));

            // Add callbacks with different increment values
            for increment in [1, 10, 100] {
                let counter_clone = counter.clone();
                registry.add_update_callback(Arc::new(move || {
                    counter_clone.fetch_add(increment, Ordering::SeqCst);
                    Ok(())
                }));
            }

            // Verify counter starts at 0
            assert_eq!(counter.load(Ordering::SeqCst), 0);

            // First execution
            let results = registry.execute_update_callbacks();
            assert_eq!(results.len(), 3);
            assert!(results.iter().all(|r| r.is_ok()));
            assert_eq!(counter.load(Ordering::SeqCst), 111); // 1 + 10 + 100

            // Second execution - callbacks should be reusable
            let results = registry.execute_update_callbacks();
            assert_eq!(results.len(), 3);
            assert_eq!(counter.load(Ordering::SeqCst), 222); // 111 + 111

            // Test cloning - cloned entry shares callbacks (callbacks are Arc-wrapped)
            let cloned = registry.clone();
            assert_eq!(cloned.execute_update_callbacks().len(), 3);
            assert_eq!(counter.load(Ordering::SeqCst), 333); // 222 + 111

            // Original still has callbacks and shares the same Arc
            registry.execute_update_callbacks();
            assert_eq!(counter.load(Ordering::SeqCst), 444); // 333 + 111
        }

        // Test 2: Mixed success and error callbacks
        {
            let registry = MetricsRegistry::new();
            let counter = Arc::new(AtomicUsize::new(0));

            // Successful callback
            let counter_clone = counter.clone();
            registry.add_update_callback(Arc::new(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }));

            // Error callback
            registry.add_update_callback(Arc::new(|| Err(anyhow::anyhow!("Simulated error"))));

            // Another successful callback
            let counter_clone = counter.clone();
            registry.add_update_callback(Arc::new(move || {
                counter_clone.fetch_add(10, Ordering::SeqCst);
                Ok(())
            }));

            // Execute and verify mixed results
            let results = registry.execute_update_callbacks();
            assert_eq!(results.len(), 3);
            assert!(results[0].is_ok());
            assert!(results[1].is_err());
            assert!(results[2].is_ok());

            // Verify error message
            assert_eq!(
                results[1].as_ref().unwrap_err().to_string(),
                "Simulated error"
            );

            // Verify successful callbacks still executed
            assert_eq!(counter.load(Ordering::SeqCst), 11); // 1 + 10

            // Execute again - errors should be consistent
            let results = registry.execute_update_callbacks();
            assert!(results[1].is_err());
            assert_eq!(counter.load(Ordering::SeqCst), 22); // 11 + 11
        }

        // Test 3: Empty registry
        {
            let registry = MetricsRegistry::new();
            let results = registry.execute_update_callbacks();
            assert_eq!(results.len(), 0);
        }
    }
}

#[cfg(test)]
mod test_metric_families_combined {
    use super::*;
    use std::sync::Arc as StdArc;

    fn gauge_family(name: &str, help: &str, value: f64) -> prometheus::proto::MetricFamily {
        let mut family = prometheus::proto::MetricFamily::new();
        family.set_name(name.to_string());
        family.set_help(help.to_string());
        family.set_field_type(prometheus::proto::MetricType::GAUGE);
        let mut metric = prometheus::proto::Metric::new();
        let mut gauge = prometheus::proto::Gauge::new();
        gauge.set_value(value);
        metric.set_gauge(gauge);
        family.mut_metric().push(metric);
        family
    }

    fn registry() -> MetricsRegistry {
        let registry = MetricsRegistry::new();
        let counter = prometheus::IntCounter::new("dynamo_native_total", "Native counter").unwrap();
        counter.inc_by(7);
        registry
            .get_prometheus_registry()
            .register(Box::new(counter))
            .unwrap();
        registry.add_typed_callback(StdArc::new(|| {
            Ok(vec![crate::prom_typed::BuiltFamily {
                unit: String::new(),
                prom_type: "gauge".to_string(),
                family: gauge_family("vllm:num_requests_running", "Running requests", 4.0),
            }])
        }));
        registry
    }

    fn names(fs: &[prometheus::proto::MetricFamily]) -> Vec<&str> {
        fs.iter().map(|f| f.name()).collect()
    }

    #[test]
    fn typed_and_native_families_share_one_collection() {
        let collected = registry().metric_families_combined().expect("combined");
        assert!(names(&collected.families).contains(&"vllm:num_requests_running"));
        assert!(names(&collected.families).contains(&"dynamo_native_total"));
    }

    /// `units` and `prom_types` are keyed by family name, so a family the
    /// merger rejects must not leave its metadata behind: it would then be
    /// applied to whichever family did survive under that name. Here the
    /// rejected entry claims `info`, which would flip the surviving gauge to a
    /// non-monotonic Sum on the OTLP side.
    /// The merger resolves a conflict first-writer-wins: the first family
    /// fixes help and type and later ones must agree. The metadata maps must
    /// resolve the same way. A later family can pass that check and still have
    /// every sample dropped as a duplicate series, so a last-writer-wins unit
    /// would label the surviving data with a unit from a source that
    /// contributed none of it.
    #[test]
    fn metadata_from_a_source_that_contributed_no_samples_does_not_win() {
        let registry = MetricsRegistry::new();
        registry.add_typed_callback(StdArc::new(|| {
            Ok(vec![
                crate::prom_typed::BuiltFamily {
                    unit: "bytes".to_string(),
                    prom_type: "gauge".to_string(),
                    family: gauge_family("vllm:probe", "Same help", 1.0),
                },
                // Same name, help and type, so the merger accepts it -- but the
                // labels match too, so every sample is dropped as a duplicate.
                crate::prom_typed::BuiltFamily {
                    unit: "seconds".to_string(),
                    prom_type: "gauge".to_string(),
                    family: gauge_family("vllm:probe", "Same help", 2.0),
                },
            ])
        }));

        let collected = registry.metric_families_combined().expect("combined");

        let family = collected
            .families
            .iter()
            .find(|f| f.name() == "vllm:probe")
            .expect("family");
        assert_eq!(
            family.get_metric()[0].get_gauge().value(),
            1.0,
            "the surviving sample should be the first source's"
        );
        assert_eq!(
            collected.units.get("vllm:probe").map(String::as_str),
            Some("bytes"),
            "unit came from a source whose samples were all dropped"
        );
    }

    #[test]
    fn rejected_family_leaves_no_metadata_behind() {
        let registry = MetricsRegistry::new();
        registry.add_typed_callback(StdArc::new(|| {
            Ok(vec![
                crate::prom_typed::BuiltFamily {
                    unit: String::new(),
                    prom_type: "gauge".to_string(),
                    family: gauge_family("vllm:cache_usage", "Cache usage", 0.5),
                },
                // Same name, different help -> rejected by the merger.
                crate::prom_typed::BuiltFamily {
                    unit: "seconds".to_string(),
                    prom_type: "info".to_string(),
                    family: gauge_family("vllm:cache_usage", "A different help", 1.0),
                },
            ])
        }));

        let collected = registry.metric_families_combined().expect("combined");

        assert_eq!(
            collected
                .prom_types
                .get("vllm:cache_usage")
                .map(String::as_str),
            Some("gauge"),
            "the rejected family's type overwrote the accepted family's"
        );
        assert!(
            !collected.units.contains_key("vllm:cache_usage"),
            "the rejected family's unit was published for a family that has none"
        );
    }

    /// The two surfaces are fed independently: typed callbacks reach OTLP,
    /// exposition-text callbacks reach `/metrics`. A registry that registers
    /// only one appears on only that surface -- which is why the Python
    /// helpers always register both.
    #[test]
    fn typed_callbacks_feed_otlp_and_not_the_scrape() {
        let registry = registry();

        let text = registry.prometheus_expfmt_combined().expect("text");
        assert!(
            text.contains("dynamo_native_total"),
            "native metrics must still render on /metrics"
        );
        assert!(
            !text.contains("vllm:num_requests_running"),
            "typed callbacks feed OTLP only; /metrics is fed by expfmt callbacks"
        );

        registry.add_expfmt_callback(StdArc::new(|| {
            Ok("# HELP vllm:num_requests_running Running requests\n\
                # TYPE vllm:num_requests_running gauge\n\
                vllm:num_requests_running 4\n"
                .to_string())
        }));
        let text = registry.prometheus_expfmt_combined().expect("text");
        assert!(
            text.contains("vllm:num_requests_running"),
            "engine exposition text must be appended verbatim"
        );
    }

    /// Collection must not hold the callback lock while invoking callbacks.
    ///
    /// Registration arrives from Python holding the GIL and blocks on
    /// `write()`; a collection that held the read lock while calling into
    /// Python would deadlock the two against each other. Registering from
    /// inside a callback reproduces that ordering deterministically, without
    /// depending on a thread interleaving.
    ///
    /// Run on a worker thread with a deadline so a regression fails the test
    /// rather than hanging the suite forever.
    #[test]
    fn collection_does_not_hold_the_callback_lock() {
        for (name, collect) in [
            (
                "typed",
                Box::new(|r: &MetricsRegistry| {
                    r.execute_typed_callbacks();
                }) as Box<dyn Fn(&MetricsRegistry) + Send>,
            ),
            (
                "expfmt",
                Box::new(|r: &MetricsRegistry| {
                    r.execute_expfmt_callbacks();
                }),
            ),
        ] {
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let registry = MetricsRegistry::new();
                let reentrant = registry.clone();
                registry.add_typed_callback(StdArc::new(move || {
                    reentrant.add_typed_callback(StdArc::new(|| Ok(Vec::new())));
                    Ok(Vec::new())
                }));
                let reentrant = registry.clone();
                registry.add_expfmt_callback(StdArc::new(move || {
                    reentrant.add_expfmt_callback(StdArc::new(|| Ok(String::new())));
                    Ok(String::new())
                }));
                collect(&registry);
                let _ = tx.send(());
            });

            assert!(
                rx.recv_timeout(std::time::Duration::from_secs(10)).is_ok(),
                "{name} collection deadlocked: the callback lock was held while a \
                 callback registered another"
            );
        }
    }

    /// A failing engine callback is skipped, never fatal: one broken engine
    /// must not empty the collection for every other.
    #[test]
    fn failing_typed_callback_is_skipped() {
        let registry = MetricsRegistry::new();
        registry.add_typed_callback(StdArc::new(|| anyhow::bail!("engine exploded")));
        assert!(
            registry
                .metric_families_combined()
                .expect("collection survives")
                .families
                .is_empty()
        );
    }
}
