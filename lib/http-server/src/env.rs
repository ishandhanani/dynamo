// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Environment variable names read by the HTTP server core and its metrics.
//!
//! `dynamo-runtime` re-exports these under `dynamo_runtime::config::environment_names::llm`
//! so the runtime's environment-variable registry stays complete; the definitions live here
//! because the HTTP server must build without the runtime.

/// HTTP body size limit in MB
pub const DYN_HTTP_BODY_LIMIT_MB: &str = "DYN_HTTP_BODY_LIMIT_MB";

/// Listen backlog of the frontend HTTP/HTTPS socket (default 4096).
pub const DYN_HTTP_LISTEN_BACKLOG: &str = "DYN_HTTP_LISTEN_BACKLOG";

pub const DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS: &str = "DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS";

/// Maximum stalled downstream write time per connection (default 120 seconds).
pub const DYN_HTTP_WRITE_IDLE_TIMEOUT_SECS: &str = "DYN_HTTP_WRITE_IDLE_TIMEOUT_SECS";

/// Environment variables that tune the frontend HTTP metrics.
pub mod metrics {
    /// Custom metrics prefix (overrides default "dynamo_frontend")
    pub const DYN_METRICS_PREFIX: &str = "DYN_METRICS_PREFIX";

    /// Histogram bucket configuration prefixes. Each is suffixed with `_MIN`,
    /// `_MAX`, or `_COUNT` to form the variable that tunes one frontend
    /// histogram's log-spaced buckets, for example `DYN_METRICS_ITL_MAX`.
    /// Values are read once, when the frontend builds its metrics.
    pub const DYN_METRICS_REQUEST_DURATION: &str = "DYN_METRICS_REQUEST_DURATION";
    /// See [`DYN_METRICS_REQUEST_DURATION`].
    pub const DYN_METRICS_INPUT_SEQUENCE: &str = "DYN_METRICS_INPUT_SEQUENCE";
    /// See [`DYN_METRICS_REQUEST_DURATION`].
    pub const DYN_METRICS_OUTPUT_SEQUENCE: &str = "DYN_METRICS_OUTPUT_SEQUENCE";
    /// See [`DYN_METRICS_REQUEST_DURATION`].
    pub const DYN_METRICS_TTFT: &str = "DYN_METRICS_TTFT";
    /// See [`DYN_METRICS_REQUEST_DURATION`].
    pub const DYN_METRICS_ITL: &str = "DYN_METRICS_ITL";
    /// Comma-separated inter-token latency histogram bucket boundaries.
    pub const DYN_METRICS_ITL_BUCKETS: &str = "DYN_METRICS_ITL_BUCKETS";
    /// See [`DYN_METRICS_REQUEST_DURATION`].
    pub const DYN_METRICS_EMBEDDING_LATENCY: &str = "DYN_METRICS_EMBEDDING_LATENCY";

    /// Deprecated prefix for the histogram bucket variables above.
    ///
    /// This was once prepended to prefixes that already started with
    /// `DYN_METRICS_`, so the variables were read under doubled names such as
    /// `DYN_HISTOGRAM_DYN_METRICS_ITL_MAX`. The doubled form is still accepted
    /// as a fallback, with a warning, and will be removed in a future release.
    pub const DEPRECATED_HISTOGRAM_PREFIX: &str = "DYN_HISTOGRAM_";

    /// Former name of [`DEPRECATED_HISTOGRAM_PREFIX`], kept so that code outside
    /// this workspace importing it keeps compiling. Remove together with the
    /// doubled-name fallback.
    #[deprecated(note = "use DEPRECATED_HISTOGRAM_PREFIX")]
    pub const HISTOGRAM_PREFIX: &str = DEPRECATED_HISTOGRAM_PREFIX;
}
