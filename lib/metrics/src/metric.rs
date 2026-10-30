// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The metric factory: `PrometheusMetric` over the Prometheus types, and `create_metric`,
//! which names a metric under its hierarchy, injects the hierarchy's labels, builds it, and
//! registers it in the hierarchy's registry.

use crate::names::{build_metric_name, labels, name_prefix, sanitize_prometheus_label};
use crate::registry::MetricsHierarchy;

fn validate_no_duplicate_label_keys(labels: &[(&str, &str)]) -> anyhow::Result<()> {
    let mut seen_keys = std::collections::HashSet::new();
    for (key, _) in labels {
        if !seen_keys.insert(*key) {
            return Err(anyhow::anyhow!(
                "Duplicate label key '{}' found in labels",
                key
            ));
        }
    }
    Ok(())
}

/// ==============================
/// Prometheus section
/// ==============================
/// Trait that defines common behavior for Prometheus metric types
pub trait PrometheusMetric: prometheus::core::Collector + Clone + Send + Sync + 'static {
    /// Create a new metric with the given options
    fn with_opts(opts: prometheus::Opts) -> Result<Self, prometheus::Error>
    where
        Self: Sized;

    /// Create a new metric with histogram options and custom buckets
    /// This is a default implementation that will panic for non-histogram metrics
    fn with_histogram_opts_and_buckets(
        _opts: prometheus::HistogramOpts,
        _buckets: Option<Vec<f64>>,
    ) -> Result<Self, prometheus::Error>
    where
        Self: Sized,
    {
        panic!("with_histogram_opts_and_buckets is not implemented for this metric type");
    }

    /// Create a histogram vector with custom buckets and label names.
    fn with_histogram_opts_buckets_and_label_names(
        _opts: prometheus::HistogramOpts,
        _buckets: Option<Vec<f64>>,
        _label_names: &[&str],
    ) -> Result<Self, prometheus::Error>
    where
        Self: Sized,
    {
        panic!(
            "with_histogram_opts_buckets_and_label_names is not implemented for this metric type"
        );
    }

    /// Create a new metric with counter options and label names (for CounterVec)
    /// This is a default implementation that will panic for non-countervec metrics
    fn with_opts_and_label_names(
        _opts: prometheus::Opts,
        _label_names: &[&str],
    ) -> Result<Self, prometheus::Error>
    where
        Self: Sized,
    {
        panic!("with_opts_and_label_names is not implemented for this metric type");
    }
}

// Implement the trait for Counter, IntCounter, and Gauge
impl PrometheusMetric for prometheus::Counter {
    fn with_opts(opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
        prometheus::Counter::with_opts(opts)
    }
}

impl PrometheusMetric for prometheus::IntCounter {
    fn with_opts(opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
        prometheus::IntCounter::with_opts(opts)
    }
}

impl PrometheusMetric for prometheus::Gauge {
    fn with_opts(opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
        prometheus::Gauge::with_opts(opts)
    }
}

impl PrometheusMetric for prometheus::IntGauge {
    fn with_opts(opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
        prometheus::IntGauge::with_opts(opts)
    }
}

impl PrometheusMetric for prometheus::GaugeVec {
    fn with_opts(_opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
        Err(prometheus::Error::Msg(
            "GaugeVec requires label names, use with_opts_and_label_names instead".to_string(),
        ))
    }

    fn with_opts_and_label_names(
        opts: prometheus::Opts,
        label_names: &[&str],
    ) -> Result<Self, prometheus::Error> {
        prometheus::GaugeVec::new(opts, label_names)
    }
}

impl PrometheusMetric for prometheus::IntGaugeVec {
    fn with_opts(_opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
        Err(prometheus::Error::Msg(
            "IntGaugeVec requires label names, use with_opts_and_label_names instead".to_string(),
        ))
    }

    fn with_opts_and_label_names(
        opts: prometheus::Opts,
        label_names: &[&str],
    ) -> Result<Self, prometheus::Error> {
        prometheus::IntGaugeVec::new(opts, label_names)
    }
}

impl PrometheusMetric for prometheus::IntCounterVec {
    fn with_opts(_opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
        Err(prometheus::Error::Msg(
            "IntCounterVec requires label names, use with_opts_and_label_names instead".to_string(),
        ))
    }

    fn with_opts_and_label_names(
        opts: prometheus::Opts,
        label_names: &[&str],
    ) -> Result<Self, prometheus::Error> {
        prometheus::IntCounterVec::new(opts, label_names)
    }
}

// Implement the trait for Histogram
impl PrometheusMetric for prometheus::Histogram {
    fn with_opts(opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
        // Convert Opts to HistogramOpts
        let histogram_opts = prometheus::HistogramOpts::new(opts.name, opts.help);
        prometheus::Histogram::with_opts(histogram_opts)
    }

    fn with_histogram_opts_and_buckets(
        mut opts: prometheus::HistogramOpts,
        buckets: Option<Vec<f64>>,
    ) -> Result<Self, prometheus::Error> {
        if let Some(custom_buckets) = buckets {
            opts = opts.buckets(custom_buckets);
        }
        prometheus::Histogram::with_opts(opts)
    }
}

impl PrometheusMetric for prometheus::HistogramVec {
    fn with_opts(_opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
        Err(prometheus::Error::Msg(
            "HistogramVec requires histogram options and label names".to_string(),
        ))
    }

    fn with_histogram_opts_buckets_and_label_names(
        mut opts: prometheus::HistogramOpts,
        buckets: Option<Vec<f64>>,
        label_names: &[&str],
    ) -> Result<Self, prometheus::Error> {
        if let Some(custom_buckets) = buckets {
            opts = opts.buckets(custom_buckets);
        }
        prometheus::HistogramVec::new(opts, label_names)
    }
}

// Implement the trait for CounterVec
impl PrometheusMetric for prometheus::CounterVec {
    fn with_opts(_opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
        // This will panic - CounterVec needs label names
        panic!("CounterVec requires label names, use with_opts_and_label_names instead");
    }

    fn with_opts_and_label_names(
        opts: prometheus::Opts,
        label_names: &[&str],
    ) -> Result<Self, prometheus::Error> {
        prometheus::CounterVec::new(opts, label_names)
    }
}

/// ==============================
/// Metrics section
/// ==============================
/// Public helper function to create metrics - accessible for Python bindings
pub fn create_metric<T: PrometheusMetric, H: MetricsHierarchy + ?Sized>(
    hierarchy: &H,
    metric_name: &str,
    metric_desc: &str,
    labels: &[(&str, &str)],
    buckets: Option<Vec<f64>>,
    const_labels: Option<&[&str]>,
) -> anyhow::Result<T> {
    create_metric_with_options(
        hierarchy,
        MetricOptions::default(),
        metric_name,
        metric_desc,
        labels,
        buckets,
        const_labels,
    )
}

/// How [`create_metric_with_options`] names and registers a metric. The
/// default is what [`create_metric`] does.
#[derive(Clone, Copy, Debug)]
pub struct MetricOptions {
    /// The metric is named `{prefix}_{metric_name}`. Only this prefix is ever
    /// used: none is inferred from the metric name.
    pub prefix: &'static str,
    /// When the hierarchy's registry refuses the metric, log a warning and
    /// return it unregistered, still usable but not exposed, instead of
    /// failing.
    pub warn_on_registration_failure: bool,
}

impl Default for MetricOptions {
    fn default() -> Self {
        Self {
            prefix: name_prefix::COMPONENT,
            warn_on_registration_failure: false,
        }
    }
}

/// [`create_metric`] under explicit [`MetricOptions`]: the same validation,
/// auto-labels from the hierarchy, construction and registration.
pub fn create_metric_with_options<T: PrometheusMetric, H: MetricsHierarchy + ?Sized>(
    hierarchy: &H,
    options: MetricOptions,
    metric_name: &str,
    metric_desc: &str,
    labels: &[(&str, &str)],
    buckets: Option<Vec<f64>>,
    const_labels: Option<&[&str]>,
) -> anyhow::Result<T> {
    // Validate that user-provided labels don't have duplicate keys
    validate_no_duplicate_label_keys(labels)?;
    // Note: stored labels functionality has been removed

    let basename = hierarchy.basename();
    let parent_hierarchies = hierarchy.parent_hierarchies();

    // Build hierarchy path as vector of strings: parent names + [basename]
    let mut hierarchy_names: Vec<String> =
        parent_hierarchies.iter().map(|p| p.basename()).collect();
    hierarchy_names.push(basename.clone());

    let metric_name = build_metric_name(options.prefix, metric_name);

    // Build updated_labels: auto-labels first, then `labels` + stored labels
    let mut updated_labels: Vec<(String, String)> = Vec::new();

    // Auto-label injection: Always add dynamo_namespace, dynamo_component, dynamo_endpoint labels
    // based on the hierarchy. Label constants defined in prometheus_names.rs labels module.
    //
    // Python counterpart: components/src/dynamo/common/utils/prometheus.py register_engine_metrics_callback()

    // Validate that user-provided labels don't conflict with auto-generated labels
    for (key, _) in labels {
        if *key == labels::NAMESPACE
            || *key == labels::COMPONENT
            || *key == labels::ENDPOINT
            || *key == labels::WORKER_ID
        {
            return Err(anyhow::anyhow!(
                "Label '{}' is automatically added by auto-label injection and cannot be manually set",
                key
            ));
        }
    }

    // Also validate that vector label names (const_labels) don't collide with auto-injected
    // const labels. A variable label named "worker_id" would conflict with the auto-injected
    // worker_id const label, causing a prometheus registration error or ambiguous output.
    if let Some(label_names) = const_labels {
        for name in label_names.iter() {
            if *name == labels::NAMESPACE
                || *name == labels::COMPONENT
                || *name == labels::ENDPOINT
                || *name == labels::WORKER_ID
            {
                return Err(anyhow::anyhow!(
                    "Variable label name '{}' conflicts with auto-injected const label and cannot be used",
                    name
                ));
            }
        }
    }

    // Add auto-generated labels with sanitized values
    // Hierarchy: [drt, namespace, component, endpoint]
    if hierarchy_names.len() > 1 {
        let namespace = &hierarchy_names[1];
        if !namespace.is_empty() {
            let valid_namespace = sanitize_prometheus_label(namespace)?;
            if !valid_namespace.is_empty() {
                updated_labels.push((labels::NAMESPACE.to_string(), valid_namespace));
            }
        }
    }
    if hierarchy_names.len() > 2 {
        let component = &hierarchy_names[2];
        if !component.is_empty() {
            let valid_component = sanitize_prometheus_label(component)?;
            if !valid_component.is_empty() {
                updated_labels.push((labels::COMPONENT.to_string(), valid_component));
            }
        }
    }
    if hierarchy_names.len() > 3 {
        let endpoint = &hierarchy_names[3];
        if !endpoint.is_empty() {
            let valid_endpoint = sanitize_prometheus_label(endpoint)?;
            if !valid_endpoint.is_empty() {
                updated_labels.push((labels::ENDPOINT.to_string(), valid_endpoint));
            }
        }
    }

    // Auto-inject worker_id label from the hierarchy's connection_id (discovery instance ID).
    // This provides a stable per-worker identity label so metrics from different workers
    // serving the same endpoint can be distinguished without relying on Kubernetes labels.
    if let Some(conn_id) = hierarchy.connection_id() {
        updated_labels.push((labels::WORKER_ID.to_string(), format!("{:x}", conn_id)));
    }

    // Add user labels
    updated_labels.extend(
        labels
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string())),
    );
    // Note: stored labels functionality has been removed

    // Handle different metric types
    let prometheus_metric = if std::any::TypeId::of::<T>()
        == std::any::TypeId::of::<prometheus::CounterVec>()
    {
        // Special handling for CounterVec with label names
        // const_labels parameter is required for CounterVec
        if buckets.is_some() {
            return Err(anyhow::anyhow!(
                "buckets parameter is not valid for CounterVec"
            ));
        }
        let mut opts = prometheus::Opts::new(&metric_name, metric_desc);
        for (key, value) in &updated_labels {
            opts = opts.const_label(key.clone(), value.clone());
        }
        let label_names = const_labels
            .ok_or_else(|| anyhow::anyhow!("CounterVec requires const_labels parameter"))?;
        T::with_opts_and_label_names(opts, label_names)?
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<prometheus::GaugeVec>() {
        // Special handling for GaugeVec with label names
        // const_labels parameter is required for GaugeVec
        if buckets.is_some() {
            return Err(anyhow::anyhow!(
                "buckets parameter is not valid for GaugeVec"
            ));
        }
        let mut opts = prometheus::Opts::new(&metric_name, metric_desc);
        for (key, value) in &updated_labels {
            opts = opts.const_label(key.clone(), value.clone());
        }
        let label_names = const_labels
            .ok_or_else(|| anyhow::anyhow!("GaugeVec requires const_labels parameter"))?;
        T::with_opts_and_label_names(opts, label_names)?
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<prometheus::HistogramVec>() {
        let mut opts = prometheus::HistogramOpts::new(&metric_name, metric_desc);
        for (key, value) in &updated_labels {
            opts = opts.const_label(key.clone(), value.clone());
        }
        let label_names = const_labels
            .ok_or_else(|| anyhow::anyhow!("HistogramVec requires const_labels parameter"))?;
        T::with_histogram_opts_buckets_and_label_names(opts, buckets, label_names)?
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<prometheus::Histogram>() {
        // Special handling for Histogram with custom buckets
        // buckets parameter is valid for Histogram, const_labels is not used
        if const_labels.is_some() {
            return Err(anyhow::anyhow!(
                "const_labels parameter is not valid for Histogram"
            ));
        }
        let mut opts = prometheus::HistogramOpts::new(&metric_name, metric_desc);
        for (key, value) in &updated_labels {
            opts = opts.const_label(key.clone(), value.clone());
        }
        T::with_histogram_opts_and_buckets(opts, buckets)?
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<prometheus::IntCounterVec>() {
        // Special handling for IntCounterVec with label names
        // const_labels parameter is required for IntCounterVec
        if buckets.is_some() {
            return Err(anyhow::anyhow!(
                "buckets parameter is not valid for IntCounterVec"
            ));
        }
        let mut opts = prometheus::Opts::new(&metric_name, metric_desc);
        for (key, value) in &updated_labels {
            opts = opts.const_label(key.clone(), value.clone());
        }
        let label_names = const_labels
            .ok_or_else(|| anyhow::anyhow!("IntCounterVec requires const_labels parameter"))?;
        T::with_opts_and_label_names(opts, label_names)?
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<prometheus::IntGaugeVec>() {
        // Special handling for IntGaugeVec with label names
        // const_labels parameter is required for IntGaugeVec
        if buckets.is_some() {
            return Err(anyhow::anyhow!(
                "buckets parameter is not valid for IntGaugeVec"
            ));
        }
        let mut opts = prometheus::Opts::new(&metric_name, metric_desc);
        for (key, value) in &updated_labels {
            opts = opts.const_label(key.clone(), value.clone());
        }
        let label_names = const_labels
            .ok_or_else(|| anyhow::anyhow!("IntGaugeVec requires const_labels parameter"))?;
        T::with_opts_and_label_names(opts, label_names)?
    } else {
        // Standard handling for Counter, IntCounter, Gauge, IntGauge
        // buckets and const_labels parameters are not valid for these types
        if buckets.is_some() {
            return Err(anyhow::anyhow!(
                "buckets parameter is not valid for Counter, IntCounter, Gauge, or IntGauge"
            ));
        }
        if const_labels.is_some() {
            return Err(anyhow::anyhow!(
                "const_labels parameter is not valid for Counter, IntCounter, Gauge, or IntGauge"
            ));
        }
        let mut opts = prometheus::Opts::new(&metric_name, metric_desc);
        for (key, value) in &updated_labels {
            opts = opts.const_label(key.clone(), value.clone());
        }
        T::with_opts(opts)?
    };

    let collector: Box<dyn prometheus::core::Collector> = Box::new(prometheus_metric.clone());
    let registry = hierarchy.get_metrics_registry();
    if options.warn_on_registration_failure {
        registry.add_metric_or_warn(collector, &metric_name);
    } else {
        registry.add_metric(collector)?;
    }

    Ok(prometheus_metric)
}

#[cfg(test)]
mod test_create_metric_with_options {
    use super::*;
    use crate::registry::{MetricsHierarchy, MetricsRegistry};
    use prometheus::core::Collector;

    /// A hierarchy level without a runtime, with a registry of its own.
    struct Level {
        name: &'static str,
        parents: Vec<Level>,
        registry: MetricsRegistry,
        connection_id: Option<u64>,
    }

    impl Level {
        fn new(name: &'static str, parents: Vec<Level>, connection_id: Option<u64>) -> Self {
            Self {
                name,
                parents,
                registry: MetricsRegistry::new(),
                connection_id,
            }
        }

        /// Component `comp` of namespace `ns-1`, on connection `0x1a`.
        fn component() -> Self {
            let root = Self::new("", Vec::new(), None);
            let namespace = Self::new("ns-1", Vec::new(), None);
            Self::new("comp", vec![root, namespace], Some(0x1a))
        }
    }

    impl MetricsHierarchy for Level {
        fn basename(&self) -> String {
            self.name.to_string()
        }

        fn parent_hierarchies(&self) -> Vec<&dyn MetricsHierarchy> {
            self.parents
                .iter()
                .map(|parent| parent as &dyn MetricsHierarchy)
                .collect()
        }

        fn get_metrics_registry(&self) -> &MetricsRegistry {
            &self.registry
        }

        fn connection_id(&self) -> Option<u64> {
            self.connection_id
        }
    }

    const CUSTOM: MetricOptions = MetricOptions {
        prefix: "dynamo_custom",
        warn_on_registration_failure: false,
    };

    /// A metric's name and its constant labels, as sorted `name=value` pairs.
    fn described(metric: &impl Collector) -> (String, Vec<String>) {
        let desc = metric.desc()[0];
        let labels = desc
            .const_label_pairs
            .iter()
            .map(|pair| format!("{}={}", pair.name(), pair.value()))
            .collect();
        (desc.fq_name.clone(), labels)
    }

    /// A prefix applies only when passed explicitly, and a metric named under
    /// one carries the same hierarchy labels as one named by default.
    #[test]
    fn an_explicit_prefix_names_the_metric_and_keeps_the_hierarchy_labels() {
        let component = Level::component();
        let custom: prometheus::IntCounterVec = create_metric_with_options(
            &component,
            CUSTOM,
            "requests_total",
            "help",
            &[],
            None,
            Some(&["dp_rank"]),
        )
        .unwrap();
        let default = component
            .metrics()
            .create_intcountervec("requests_total", "help", &["dp_rank"], &[])
            .unwrap();
        // A name that merely looks prefixed is still named by default.
        let lookalike: prometheus::IntCounter =
            create_metric(&component, "dynamo_custom_total", "help", &[], None, None).unwrap();

        let (name, labels) = described(&custom);
        assert_eq!(name, "dynamo_custom_requests_total");
        assert_eq!(
            labels,
            [
                "dynamo_component=comp",
                "dynamo_namespace=ns_1",
                "worker_id=1a"
            ]
        );
        assert_eq!(
            described(&default),
            ("dynamo_component_requests_total".to_string(), labels)
        );
        assert_eq!(
            described(&lookalike).0,
            "dynamo_component_dynamo_custom_total"
        );
    }

    /// Explicit options keep every label check, and choose what a registry's
    /// refusal does: fail, or keep the metric unregistered with a warning.
    #[test]
    fn explicit_options_keep_validation_and_choose_how_a_refusal_ends() {
        let component = Level::component();
        let gauge = |options, labels: &[(&str, &str)], label_names: &[&str]| {
            create_metric_with_options::<prometheus::IntGaugeVec, _>(
                &component,
                options,
                "depth",
                "help",
                labels,
                None,
                Some(label_names),
            )
        };
        assert!(gauge(CUSTOM, &[("worker_id", "7")], &["dp_rank"]).is_err());
        assert!(gauge(CUSTOM, &[], &["dynamo_namespace"]).is_err());
        assert!(gauge(CUSTOM, &[("a", "1"), ("a", "2")], &["dp_rank"]).is_err());

        let first = gauge(CUSTOM, &[], &["dp_rank"]).unwrap();
        assert!(gauge(CUSTOM, &[], &["dp_rank"]).is_err());
        let warn = MetricOptions {
            warn_on_registration_failure: true,
            ..CUSTOM
        };
        let refused = gauge(warn, &[], &["dp_rank"]).unwrap();
        first.with_label_values(&["0"]).set(3);
        refused.with_label_values(&["0"]).set(5);

        // The registry exposes only the first; the refused one still counts.
        let exposed = component.registry.prometheus_expfmt_combined().unwrap();
        let samples: Vec<&str> = exposed
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect();
        assert_eq!(
            samples,
            [
                r#"dynamo_custom_depth{dp_rank="0",dynamo_component="comp",dynamo_namespace="ns_1",worker_id="1a"} 3"#
            ]
        );
        assert_eq!(refused.with_label_values(&["0"]).get(), 5);
    }
}
