// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-decoded media as carried in `PreprocessedRequest::multi_modal_data`.
//!
//! When the frontend decodes an image or video itself, the worker receives a
//! descriptor of the decoded tensor in NIXL-registered memory instead of the
//! original URL. These types are the wire form of that descriptor. The frontend's
//! `preprocessor::media` module converts the NIXL registration into them and keeps
//! the registered memory alive through [`MediaStorageHandle`], which never reaches
//! the wire.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Element type of a decoded media tensor.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum DataType {
    UINT8,
}

/// Pixel layout of a decoded image tensor.
#[allow(clippy::upper_case_acronyms)]
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum ImageLayout {
    HWC,
}

/// Decoded image metadata the worker needs to interpret the tensor.
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct ImageMetadata {
    pub format: Option<image::ImageFormat>,
    pub color_type: image::ColorType,
    pub layout: ImageLayout,
}

/// Decoded video metadata; sampled timestamps shape the model-visible token sequence.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VideoMetadata {
    pub source_fps: f64,
    pub source_duration: f64,
    pub sampled_timestamps: Vec<f64>,
}

/// Modality-specific metadata of a decoded tensor.
///
/// Both variants are always part of the wire shape; which ones a frontend can
/// produce depends on its decoder features.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum DecodedMediaMetadata {
    Image(ImageMetadata),
    Video(VideoMetadata),
}

/// Shape, element type, and metadata shared by decoded data and its RDMA descriptor.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MediaTensorInfo {
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub metadata: Option<DecodedMediaMetadata>,
}

/// Memory type of a NIXL-registered region. Mirrors `nixl::MemType` variant for
/// variant so the serialized form is identical; the frontend converts at the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NixlMemoryType {
    Dram,
    Vram,
    Block,
    Object,
    File,
    Unknown,
}

/// A NIXL memory region the worker can read the decoded tensor from.
///
/// Mirrors `dynamo_memory::nixl::NixlDescriptor` field for field so the serialized
/// form is identical; the frontend converts at the boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NixlMemoryDescriptor {
    /// Base address of the memory region.
    pub addr: u64,
    /// Size of the memory region in bytes.
    pub size: usize,
    /// Type of memory (host, device, etc.).
    pub mem_type: NixlMemoryType,
    /// Device identifier (GPU index for device memory, 0 for host memory).
    pub device_id: u64,
}

/// Process-local handle that keeps the decoded tensor's registered memory alive for
/// as long as the descriptor is referenced. Never serialized.
#[derive(Clone)]
pub struct MediaStorageHandle(Arc<dyn Any + Send + Sync>);

impl MediaStorageHandle {
    pub fn new<T: Any + Send + Sync>(storage: Arc<T>) -> Self {
        Self(storage)
    }

    /// Borrow the registration as its concrete type, if it is one.
    pub fn downcast_ref<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.0.downcast_ref::<T>()
    }

    /// Whether both handles keep the same registration alive.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl fmt::Debug for MediaStorageHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("MediaStorageHandle")
    }
}

/// Decoded media published to the worker as a NIXL descriptor.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RdmaMediaDataDescriptor {
    /// Base64 NIXL agent metadata of the publishing frontend.
    pub nixl_metadata: String,
    /// The registered region holding the tensor.
    pub nixl_descriptor: NixlMemoryDescriptor,

    #[serde(flatten)]
    pub tensor_info: MediaTensorInfo,

    /// Canonical xxh3-64 key for decoded media. Image identity covers shape,
    /// dtype, and RGB bytes; video identity additionally covers decoded
    /// metadata that affects the model-visible token sequence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,

    /// Keeps the registered memory alive on the publishing side; absent after deserialization.
    #[serde(skip, default)]
    pub source_storage: Option<MediaStorageHandle>,
}

impl RdmaMediaDataDescriptor {
    /// Canonical cache/routing key serialized on the descriptor.
    pub fn content_hash_key(&self) -> Option<&str> {
        self.content_hash.as_deref()
    }

    /// Numeric form used by MM-aware KV routing.
    pub fn content_hash(&self) -> Option<u64> {
        self.content_hash_key()
            .and_then(|key| u64::from_str_radix(key, 16).ok())
    }
}
