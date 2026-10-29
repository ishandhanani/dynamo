// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Helpers for tests that read Prometheus exposition text.

use crate::names::name_prefix;

/// Base function to filter Prometheus output lines based on a predicate.
/// Returns lines that match the predicate, converted to String.
fn filter_prometheus_lines<F>(input: &str, mut predicate: F) -> Vec<String>
where
    F: FnMut(&str) -> bool,
{
    input
        .lines()
        .filter(|line| predicate(line))
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
}

/// Extracts all component metrics (excluding help text and type definitions).
/// Returns only the actual metric lines with values.
pub fn extract_metrics(input: &str) -> Vec<String> {
    filter_prometheus_lines(input, |line| {
        line.starts_with(&format!("{}_", name_prefix::COMPONENT))
            && !line.starts_with("#")
            && !line.trim().is_empty()
    })
}

/// Parses a Prometheus metric line and extracts the name, labels, and value.
/// Used instead of fetching metrics directly to test end-to-end results, not intermediate state.
///
/// # Example
/// ```
/// # use dynamo_metrics::test_helpers::parse_prometheus_metric;
/// let line = "http_requests_total{method=\"GET\"} 1234";
/// let (name, labels, value) = parse_prometheus_metric(line).unwrap();
/// assert_eq!(name, "http_requests_total");
/// assert_eq!(labels.get("method"), Some(&"GET".to_string()));
/// assert_eq!(value, 1234.0);
/// ```
pub fn parse_prometheus_metric(
    line: &str,
) -> Option<(String, std::collections::HashMap<String, String>, f64)> {
    if line.trim().is_empty() || line.starts_with('#') {
        return None;
    }

    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }

    let metric_part = parts[0];
    let value: f64 = parts[1].parse().ok()?;

    let (name, labels) = if metric_part.contains('{') {
        let brace_start = metric_part.find('{').unwrap();
        let brace_end = metric_part.rfind('}').unwrap_or(metric_part.len());
        let name = &metric_part[..brace_start];
        let labels_str = &metric_part[brace_start + 1..brace_end];

        let mut labels = std::collections::HashMap::new();
        for pair in labels_str.split(',') {
            if let Some((k, v)) = pair.split_once('=') {
                let v = v.trim_matches('"');
                labels.insert(k.trim().to_string(), v.to_string());
            }
        }
        (name.to_string(), labels)
    } else {
        (metric_part.to_string(), std::collections::HashMap::new())
    };

    Some((name, labels, value))
}

/// Injects a `worker_id` label into Prometheus metric data lines.
/// Prometheus places const labels (like worker_id) before special labels
/// (like histogram `le`), so for histogram bucket lines we insert before
/// `,le=`. For all other metric lines, we insert before the closing `}`.
/// Comment lines and lines without labels are left unchanged.
pub fn inject_worker_id(expected: &str, wid: &str) -> String {
    let wid_label = format!(",worker_id=\"{}\"", wid);
    expected
        .lines()
        .map(|line| {
            if line.starts_with('#') || line.trim().is_empty() || !line.contains('{') {
                line.to_string()
            } else if let Some(le_pos) = line.find(",le=") {
                // Histogram bucket lines: worker_id is a const label, `le` is special,
                // so worker_id sorts before `le` in Prometheus output.
                let mut s = line.to_string();
                s.insert_str(le_pos, &wid_label);
                s
            } else {
                line.replacen("}", &format!("{}}}", wid_label), 1)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}
