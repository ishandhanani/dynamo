// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metrics the Dynamo way: one source of names and label rules, the hierarchy that names a
//! metric and injects its labels, the registry tree a scrape walks, and scrape-time
//! collection from stage counters.
//!
//! Stage crates never depend on this crate; they expose counters as atomics. Hosts and
//! composition crates depend on it, create metrics through the hierarchy, and collect stage
//! counters with [`collect::SourceSet`].

pub mod collect;
mod metric;
pub mod names;
pub mod prom_typed;
mod registry;
#[cfg(any(test, feature = "testing"))]
pub mod test_helpers;

pub use metric::{MetricOptions, PrometheusMetric, create_metric, create_metric_with_options};
pub use registry::{
    CollectedFamilies, Metrics, MetricsHierarchy, MetricsRegistry,
    PrometheusExpositionFormatCallback, PrometheusTypedCallback, PrometheusUpdateCallback,
};
