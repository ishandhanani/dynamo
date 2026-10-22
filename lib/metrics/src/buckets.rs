// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Log-spaced histogram bucket generation shared by Dynamo's metric families.

/// Generate log-spaced histogram buckets with values rounded to 2 significant figures.
///
/// # Arguments
/// * `min` - Minimum value for the buckets (must be > 0 for log spacing)
/// * `max` - Maximum value for the buckets (must be > min)
/// * `count` - Number of buckets to generate
///
/// # Returns
/// A vector of log-spaced values, always starting with 0.0 and ending with the rounded max value.
/// Duplicates created by rounding are removed, so the final count may be less than requested.
///
/// # Note
/// With 2 significant figures, there are roughly 90 unique values per order of magnitude.
/// Requesting more buckets than can be uniquely represented will result in deduplication.
pub fn generate_log_buckets(min: f64, max: f64, count: usize) -> Vec<f64> {
    if count == 0 {
        return vec![];
    }
    if count == 1 {
        return vec![0.0];
    }

    let requested_count = count;
    let mut buckets = Vec::with_capacity(count);
    buckets.push(0.0);

    // Generate log-spaced values from min to max
    for i in 1..count {
        let log_min = min.ln();
        let log_max = max.ln();
        let log_value = log_min + (log_max - log_min) * (i as f64) / ((count - 1) as f64);
        let value = log_value.exp();
        buckets.push(round_to_sig_figs(value, 2));
    }

    // Remove consecutive duplicates (buckets are already sorted)
    let original_len = buckets.len();
    buckets.dedup();

    // Warn if significant deduplication occurred
    if buckets.len() < original_len && (original_len - buckets.len()) > original_len / 10 {
        tracing::warn!(
            requested = requested_count,
            unique = buckets.len(),
            duplicates = original_len - buckets.len(),
            min = min,
            max = max,
            "Histogram bucket generation: Significant duplicate values after rounding to 2 sig figs. \
             Consider reducing bucket count or increasing range."
        );
    }

    buckets
}

/// Round a number to a specified number of significant figures
pub fn round_to_sig_figs(value: f64, sig_figs: u32) -> f64 {
    if value == 0.0 {
        return 0.0;
    }

    let magnitude = value.abs().log10().floor();
    let scale = 10_f64.powf(sig_figs as f64 - 1.0 - magnitude);
    (value * scale).round() / scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_to_sig_figs() {
        // Test rounding to 2 significant figures
        assert_eq!(round_to_sig_figs(0.0026, 2), 0.0026);
        assert_eq!(round_to_sig_figs(0.26, 2), 0.26);
        assert_eq!(round_to_sig_figs(0.2356, 2), 0.24);
        assert_eq!(round_to_sig_figs(1.234, 2), 1.2);
        assert_eq!(round_to_sig_figs(12.34, 2), 12.0);
        assert_eq!(round_to_sig_figs(123.4, 2), 120.0);
        assert_eq!(round_to_sig_figs(1234.0, 2), 1200.0);
        assert_eq!(round_to_sig_figs(0.0, 2), 0.0);

        // Test edge cases
        assert_eq!(round_to_sig_figs(0.999, 2), 1.0);
        assert_eq!(round_to_sig_figs(9.99, 2), 10.0);
        assert_eq!(round_to_sig_figs(99.9, 2), 100.0);
    }

    #[test]
    fn test_generate_log_buckets_basic() {
        // Test basic properties
        let buckets = generate_log_buckets(1.0, 100.0, 5);

        // Check length
        assert_eq!(buckets.len(), 5);

        // Check first value is 0
        assert_eq!(buckets[0], 0.0);

        // Check last value is approximately max (rounded to 2 sig figs)
        assert_eq!(buckets[buckets.len() - 1], 100.0);

        // Check values are increasing
        for i in 1..buckets.len() {
            assert!(
                buckets[i] > buckets[i - 1],
                "Bucket values should be increasing: {} <= {}",
                buckets[i - 1],
                buckets[i]
            );
        }
    }

    #[test]
    fn test_generate_log_buckets_edge_cases() {
        // Test empty buckets
        let buckets = generate_log_buckets(1.0, 100.0, 0);
        assert_eq!(buckets.len(), 0);

        // Test single bucket
        let buckets = generate_log_buckets(1.0, 100.0, 1);
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0], 0.0);

        // Test two buckets
        let buckets = generate_log_buckets(1.0, 100.0, 2);
        assert_eq!(buckets.len(), 2);
        assert_eq!(buckets[0], 0.0);
        assert_eq!(buckets[1], 100.0);
    }

    #[test]
    fn test_generate_log_buckets_always_includes_zero() {
        // Test various configurations
        for count in 1..=20 {
            let buckets = generate_log_buckets(0.1, 1000.0, count);
            assert_eq!(
                buckets[0], 0.0,
                "First bucket should always be 0.0 for count={}",
                count
            );
        }
    }

    #[test]
    fn test_all_buckets_are_two_sig_figs() {
        let test_cases = vec![
            (1.0, 512.0, 10),
            (50.0, 128000.0, 12),
            (50.0, 32000.0, 10),
            (0.001, 480.0, 18),
            (0.001, 80.0, 20),
            (0.001, 10.0, 14),
        ];

        for (min, max, count) in test_cases {
            let buckets = generate_log_buckets(min, max, count);
            for &value in buckets.iter().skip(1) {
                let rounded = round_to_sig_figs(value, 2);
                assert_eq!(
                    value, rounded,
                    "Value {} should be rounded to 2 sig figs (min={}, max={}, count={})",
                    value, min, max, count
                );
            }
        }
    }

    #[test]
    fn test_sig_fig_limitation_with_many_buckets() {
        // This test demonstrates that 2 sig figs limits the number of unique bucket values
        // With 1000 requested buckets but only 2 sig figs, we'll get automatic deduplication
        let buckets = generate_log_buckets(0.0001, 1.0, 1000);

        println!(
            "Requested 1000 buckets, got {} total values (including 0.0)",
            buckets.len()
        );

        // With 2 sig figs across 4 orders of magnitude (0.0001 to 1.0),
        // we can have roughly 90 unique values per order of magnitude
        // So we expect around 360 unique values maximum
        assert!(
            buckets.len() < 500,
            "Expected fewer than 500 unique buckets due to 2 sig fig limitation, got {}",
            buckets.len()
        );

        // Verify all values are unique (no duplicates remain after deduplication)
        let mut sorted_buckets = buckets.clone();
        sorted_buckets.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted_buckets.dedup();
        assert_eq!(
            buckets.len(),
            sorted_buckets.len(),
            "All buckets should be unique after deduplication"
        );

        // Verify first is still 0.0
        assert_eq!(buckets[0], 0.0);

        // Verify values are still in increasing order
        for i in 1..buckets.len() {
            assert!(
                buckets[i] > buckets[i - 1],
                "Buckets should be in increasing order"
            );
        }
    }

    #[test]
    fn test_deduplication_preserves_order() {
        // Test that deduplication maintains increasing order
        let buckets = generate_log_buckets(0.01, 1.0, 50);

        // Verify all values are unique
        let mut unique_check = std::collections::HashSet::new();
        for &bucket in &buckets {
            assert!(
                unique_check.insert(bucket.to_bits()),
                "Duplicate value {} found after deduplication",
                bucket
            );
        }

        // Verify order is maintained
        for i in 1..buckets.len() {
            assert!(
                buckets[i] > buckets[i - 1],
                "Bucket values should be in increasing order after deduplication"
            );
        }
    }
}
