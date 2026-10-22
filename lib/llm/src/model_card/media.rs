// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Media decoding limits and URL fetch policy as carried on the model deployment card.
//!
//! These types are pure configuration and part of the worker-to-frontend wire contract.
//! The decoders and the HTTP fetcher that act on them live in `crate::preprocessor::media`
//! and convert from these types; they may carry process-local state that never reaches
//! the card.

use std::collections::HashSet;
use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Default cap on a single decoded image, in bytes.
pub const DEFAULT_IMAGE_MAX_ALLOC: u64 = 128 * 1024 * 1024; // 128 MB
/// Default cap on the decoded frames of a single video, in bytes.
pub const DEFAULT_VIDEO_MAX_ALLOC: u64 = 512 * 1024 * 1024; // 512 MB
/// User agent for frontend media fetches.
pub const DEFAULT_HTTP_USER_AGENT: &str = "dynamo-ai/dynamo";
/// Timeout for frontend media fetches.
pub const DEFAULT_HTTP_TIMEOUT: Duration = Duration::from_secs(30);

/// Image decoder limits - can only be set via server config, not runtime kwargs.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageDecoderLimits {
    #[serde(default)]
    pub max_image_width: Option<u32>,
    #[serde(default)]
    pub max_image_height: Option<u32>,
    /// Maximum allowed total allocation of the decoder in bytes
    #[serde(default)]
    pub max_alloc: Option<u64>,
}

impl Default for ImageDecoderLimits {
    fn default() -> Self {
        Self {
            max_image_width: None,
            max_image_height: None,
            max_alloc: Some(DEFAULT_IMAGE_MAX_ALLOC),
        }
    }
}

impl ImageDecoderLimits {
    /// Check a decoded image against the configured limits, returning its byte size.
    pub fn validate_output(&self, width: u32, height: u32, channels: usize) -> Result<usize> {
        if self.max_image_width.is_some_and(|limit| width > limit)
            || self.max_image_height.is_some_and(|limit| height > limit)
        {
            anyhow::bail!("Image dimensions exceed configured limits: {width}x{height}");
        }

        let nbytes = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|pixels| pixels.checked_mul(channels as u64))
            .ok_or_else(|| {
                anyhow::anyhow!("Image allocation size overflow for dimensions: {width}x{height}")
            })?;
        if let Some(limit) = self.max_alloc
            && nbytes > limit
        {
            anyhow::bail!("Image allocation {nbytes} bytes exceeds configured limit {limit} bytes");
        }
        usize::try_from(nbytes)
            .map_err(|_| anyhow::anyhow!("Image allocation does not fit in usize: {nbytes} bytes"))
    }
}

/// Video decoder limits - can only be set via server config, not runtime kwargs.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct VideoDecoderLimits {
    /// Maximum allowed total allocation of decoded frames in bytes
    #[serde(default)]
    pub max_alloc: Option<u64>,
}

impl Default for VideoDecoderLimits {
    fn default() -> Self {
        Self {
            max_alloc: Some(DEFAULT_VIDEO_MAX_ALLOC),
        }
    }
}

/// Image decoding settings advertised on the model card.
///
/// Wire-compatible with the frontend's runtime `ImageDecoder`, which additionally
/// carries the process-local JPEG backend choice.
#[derive(Clone, Debug, Default, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageDecoderConfig {
    #[serde(default)]
    pub limits: ImageDecoderLimits,
}

/// Video decoding settings advertised on the model card.
#[derive(Clone, Debug, Default, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct VideoDecoderConfig {
    #[serde(default)]
    pub limits: VideoDecoderLimits,
    /// sample N frames per second
    #[serde(default)]
    pub fps: Option<f64>,
    /// sample at most N frames
    #[serde(default)]
    pub max_frames: Option<u64>,
    /// sample N frames in total (linspace)
    #[serde(default)]
    pub num_frames: Option<u64>,
    /// fail if some frames fail to decode
    #[serde(default)]
    pub strict: bool,
}

/// Media decoder configuration carried on the model card.
///
/// `video` is always part of the wire shape; a frontend built without the
/// `media-ffmpeg` feature ignores it and leaves videos to the worker.
#[derive(Clone, Debug, Default, Serialize, Deserialize, ToSchema)]
pub struct MediaDecoderConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<ImageDecoderConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video: Option<VideoDecoderConfig>,
    // TODO: audio decoder
}

/// URL fetch policy for frontend media loading, as advertised on the model card.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MediaFetcherConfig {
    pub user_agent: String,
    pub allow_direct_ip: bool,
    pub allow_direct_port: bool,
    /// When `false` (default), reject URLs that target blocked locations:
    /// an IP literal in the RFC-range blocklist (`BLOCKED_IP_NETWORKS`),
    /// a hostname in the literal blocklist (`BLOCKED_HOSTS`, e.g.
    /// `localhost` / `metadata.google.internal`), or — in
    /// `check_if_url_allowed_with_dns` — a hostname that DNS-resolves
    /// to a blocked IP. The name reads "IP" but semantically this is a
    /// single "allow internal / on-prem targets" switch that covers
    /// both IP and hostname blocklists together: real on-prem
    /// deployments need both at once (private CIDRs *and* internal
    /// service names), and splitting them would give no useful config
    /// while doubling the footgun surface. **Never** set on anything
    /// public-facing.
    pub allow_private_ips: bool,
    pub allowed_media_domains: Option<HashSet<String>>,
    pub timeout: Option<Duration>,
}

impl Default for MediaFetcherConfig {
    fn default() -> Self {
        Self {
            user_agent: DEFAULT_HTTP_USER_AGENT.to_string(),
            allow_direct_ip: false,
            allow_direct_port: false,
            allow_private_ips: false,
            allowed_media_domains: None,
            timeout: Some(DEFAULT_HTTP_TIMEOUT),
        }
    }
}

impl MediaFetcherConfig {
    fn with_internal_access(allow_internal: bool) -> Self {
        Self {
            allow_direct_ip: allow_internal,
            allow_direct_port: allow_internal,
            allow_private_ips: allow_internal,
            ..Self::default()
        }
    }

    /// Defaults that respect the shared `DYN_MM_ALLOW_INTERNAL` environment
    /// variable. Mirrors the Python `UrlValidationPolicy.from_env()` behavior so
    /// both fetch paths (frontend decode in Rust, backend decode in Python) honor
    /// the same on-prem opt-in flag.
    ///
    /// `DYN_MM_ALLOW_INTERNAL=1` flips `allow_direct_ip`, `allow_direct_port`, and
    /// `allow_private_ips` all to `true` at once.
    pub fn from_env() -> Self {
        let allow_internal = std::env::var("DYN_MM_ALLOW_INTERNAL").ok().as_deref() == Some("1");
        Self::with_internal_access(allow_internal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_internal_access_defaults() {
        let f = MediaFetcherConfig::with_internal_access(false);
        assert!(!f.allow_private_ips);
        assert!(!f.allow_direct_ip);
        assert!(!f.allow_direct_port);

        let f = MediaFetcherConfig::with_internal_access(true);
        assert!(f.allow_private_ips);
        assert!(f.allow_direct_ip);
        assert!(f.allow_direct_port);
    }
}
