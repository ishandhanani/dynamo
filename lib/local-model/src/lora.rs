// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA downloading and caching infrastructure.
//!
//! A minimal, extensible interface for downloading LoRA adapters from various sources
//! (local filesystem, Hugging Face, S3 behind the `lora-s3` feature) with automatic
//! caching, plus the `DYN_LORA_ENABLED` switch. LoRA routing and allocation stay with the
//! frontend in `dynamo-llm`.

pub mod cache;
pub mod downloader;
pub mod source;

pub use cache::LoRACache;
pub use downloader::LoRADownloader;
#[cfg(feature = "lora-s3")]
pub use source::S3LoRASource;
pub use source::{HuggingFaceLoRASource, LoRASource, LocalLoRASource};

pub use dynamo_llm_protocol::local_model::runtime_config::LORA_REQUIRES_REGISTRATION;

/// Returns true when LoRA serving is enabled via a truthy `DYN_LORA_ENABLED` env var
/// (`1`/`true`/`on`/`yes`, case-insensitive). This gates the request-time LoRA filter on
/// both the KV and non-KV routing paths, so non-LoRA deployments keep the unmodified
/// routing path with zero added overhead.
pub fn lora_serving_enabled() -> bool {
    dynamo_runtime::config::env_is_truthy(
        dynamo_runtime::config::environment_names::llm::DYN_LORA_ENABLED,
    )
}
