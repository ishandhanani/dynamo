// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use anyhow::Result;
use base64::{Engine as _, engine::general_purpose};
use dynamo_memory::SystemStorage;
use dynamo_memory::nixl::{self, AgentConfig, NixlAgent, NixlDescriptor, RegisteredView};
use dynamo_runtime::config::{
    env_config::parse_or_default, environment_names::llm::DYN_MM_NIXL_PROGRESS_DELAY_US,
};
use flate2::{Compression, write::ZlibEncoder};
use ndarray::{ArrayBase, Dimension, OwnedRepr};
use std::io::Write;
use std::sync::{Arc, LazyLock};

pub use crate::protocols::common::media::{
    DataType, DecodedMediaMetadata, MediaStorageHandle, MediaTensorInfo, NixlMemoryDescriptor,
    NixlMemoryType, RdmaMediaDataDescriptor,
};

// Decoded media data (image RGB, video frames pixels, ...)
#[derive(Debug)]
pub struct DecodedMediaData {
    pub(crate) data: SystemStorage,
    pub(crate) tensor_info: MediaTensorInfo,
    pub(crate) content_hash: Option<u64>,
}

/// Convert the frontend's NIXL registration into its wire form.
///
/// `NixlMemoryType` and `NixlMemoryDescriptor` are wire types owned by
/// `dynamo-llm-protocol` and `nixl::MemType` / `NixlDescriptor` belong to
/// `dynamo-memory`, so these conversions are free functions rather than `From` impls.
pub fn nixl_mem_type_to_wire(mem_type: nixl::MemType) -> NixlMemoryType {
    match mem_type {
        nixl::MemType::Dram => NixlMemoryType::Dram,
        nixl::MemType::Vram => NixlMemoryType::Vram,
        nixl::MemType::Block => NixlMemoryType::Block,
        nixl::MemType::Object => NixlMemoryType::Object,
        nixl::MemType::File => NixlMemoryType::File,
        nixl::MemType::Unknown => NixlMemoryType::Unknown,
    }
}

pub fn nixl_mem_type_from_wire(mem_type: NixlMemoryType) -> nixl::MemType {
    match mem_type {
        NixlMemoryType::Dram => nixl::MemType::Dram,
        NixlMemoryType::Vram => nixl::MemType::Vram,
        NixlMemoryType::Block => nixl::MemType::Block,
        NixlMemoryType::Object => nixl::MemType::Object,
        NixlMemoryType::File => nixl::MemType::File,
        NixlMemoryType::Unknown => nixl::MemType::Unknown,
    }
}

pub fn nixl_descriptor_to_wire(descriptor: NixlDescriptor) -> NixlMemoryDescriptor {
    NixlMemoryDescriptor {
        addr: descriptor.addr,
        size: descriptor.size,
        mem_type: nixl_mem_type_to_wire(descriptor.mem_type),
        device_id: descriptor.device_id,
    }
}

pub fn nixl_descriptor_from_wire(descriptor: NixlMemoryDescriptor) -> NixlDescriptor {
    NixlDescriptor {
        addr: descriptor.addr,
        size: descriptor.size,
        mem_type: nixl_mem_type_from_wire(descriptor.mem_type),
        device_id: descriptor.device_id,
    }
}

/// Process-local accessors over the decoded video payload a descriptor still
/// owns. `RdmaMediaDataDescriptor` is a wire type owned by `dynamo-llm-protocol`,
/// so these frontend-only helpers live behind an extension trait.
#[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
pub(crate) trait RdmaMediaDataDescriptorExt {
    /// Canonical identity for one frontend-decoded video:
    /// `(modality, shape, dtype, decoded metadata, sampled RGB bytes)`.
    /// Metadata is serialized from the typed object sent to the worker, so new
    /// model-visible fields automatically participate in the identity.
    fn video_content_hash(&self) -> Result<u64>;

    fn video_metadata(&self) -> Result<&super::decoders::VideoMetadata>;

    /// Validate the contiguous `[T, H, W, 3]` payload and return its video
    /// dimensions without copying or preprocessing pixels.
    fn video_dimensions(&self) -> Result<(usize, u32, u32)>;
}

#[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
fn local_payload(descriptor: &RdmaMediaDataDescriptor) -> Option<&[u8]> {
    use dynamo_memory::actions::Slice;
    let registered = descriptor
        .source_storage
        .as_ref()?
        .downcast_ref::<nixl::NixlRegistered<SystemStorage>>()?;
    let storage = registered.storage();
    // SAFETY: the descriptor keeps the registered SystemStorage alive and
    // request construction does not mutate it while this borrow exists.
    unsafe { storage.as_slice().ok() }
}

#[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
impl RdmaMediaDataDescriptorExt for RdmaMediaDataDescriptor {
    fn video_content_hash(&self) -> Result<u64> {
        self.video_metadata()?;
        self.content_hash()
            .context("decoded video content hash is missing")
    }

    fn video_metadata(&self) -> Result<&super::decoders::VideoMetadata> {
        match self.tensor_info.metadata.as_ref() {
            Some(DecodedMediaMetadata::Video(metadata)) => Ok(metadata),
            Some(_) => anyhow::bail!("decoded media metadata is not video metadata"),
            None => anyhow::bail!("decoded video metadata is missing"),
        }
    }

    fn video_dimensions(&self) -> Result<(usize, u32, u32)> {
        let bytes = local_payload(self)
            .ok_or_else(|| anyhow::anyhow!("decoded video has no local payload"))?;
        video_dimensions_from_parts(&self.tensor_info, bytes)
    }
}

#[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
fn hash_video_content(tensor_info: &MediaTensorInfo, bytes: &[u8]) -> Result<u64> {
    use xxhash_rust::xxh3::Xxh3;

    anyhow::ensure!(!bytes.is_empty(), "decoded video payload is empty");
    let metadata = tensor_info
        .metadata
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("decoded video metadata is missing"))?;
    let DecodedMediaMetadata::Video(metadata) = metadata else {
        anyhow::bail!("decoded media metadata is not video metadata");
    };
    let metadata_bytes = serde_json::to_vec(metadata)
        .map_err(|error| anyhow::anyhow!("failed to serialize video metadata: {error}"))?;

    let mut hasher = Xxh3::new();
    update_len_prefixed(&mut hasher, b"video");
    hasher.update(&(tensor_info.shape.len() as u64).to_le_bytes());
    for &dim in &tensor_info.shape {
        hasher.update(&(dim as u64).to_le_bytes());
    }
    let dtype_byte = match tensor_info.dtype {
        DataType::UINT8 => 0,
    };
    hasher.update(&[dtype_byte]);
    update_len_prefixed(&mut hasher, &metadata_bytes);
    update_len_prefixed(&mut hasher, bytes);
    Ok(hasher.digest())
}

#[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
fn video_dimensions_from_parts(
    tensor_info: &MediaTensorInfo,
    bytes: &[u8],
) -> Result<(usize, u32, u32)> {
    let [frames, height, width, channels] = tensor_info.shape.as_slice() else {
        anyhow::bail!(
            "decoded video shape must be [T, H, W, C], got {:?}",
            tensor_info.shape
        );
    };
    anyhow::ensure!(*frames > 0, "decoded video has no frames");
    anyhow::ensure!(
        *height > 0 && *width > 0,
        "decoded video dimensions are zero"
    );
    anyhow::ensure!(*channels == 3, "decoded video must contain RGB frames");
    anyhow::ensure!(
        tensor_info.dtype == DataType::UINT8,
        "decoded video dtype must be uint8"
    );

    let frame_len = height
        .checked_mul(*width)
        .and_then(|value| value.checked_mul(*channels))
        .ok_or_else(|| anyhow::anyhow!("decoded video frame size overflow"))?;
    let expected_len = frames
        .checked_mul(frame_len)
        .ok_or_else(|| anyhow::anyhow!("decoded video payload size overflow"))?;
    anyhow::ensure!(
        bytes.len() == expected_len,
        "decoded video payload has {} bytes, expected {}",
        bytes.len(),
        expected_len
    );
    let width = u32::try_from(*width).context("decoded video width exceeds u32")?;
    let height = u32::try_from(*height).context("decoded video height exceeds u32")?;

    Ok((*frames, width, height))
}

#[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
fn update_len_prefixed(hasher: &mut xxhash_rust::xxh3::Xxh3, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn canonical_content_hash(shape: &[usize], dtype: DataType, bytes: &[u8]) -> u64 {
    use xxhash_rust::xxh3::Xxh3;

    let mut hasher = Xxh3::new();
    // Rank and dimensions are fixed-width so distinct shapes cannot alias
    // after concatenation.
    hasher.update(&(shape.len() as u64).to_le_bytes());
    for &dim in shape {
        hasher.update(&(dim as u64).to_le_bytes());
    }
    // Widen this discriminant if DataType gains another variant.
    let dtype_byte: u8 = match dtype {
        DataType::UINT8 => 0,
    };
    hasher.update(&[dtype_byte]);
    hasher.update(bytes);
    hasher.digest()
}

fn content_hash_for_storage(
    tensor_info: &MediaTensorInfo,
    storage: &SystemStorage,
    _hash_video: bool,
) -> Option<u64> {
    use dynamo_memory::{MemoryDescriptor, actions::Slice};

    if storage.size() == 0 {
        return None;
    }

    // SAFETY: storage owns this stable buffer and is only read while the
    // descriptor is being built, before NIXL can access it concurrently.
    let bytes = unsafe { storage.as_slice().ok()? };
    match tensor_info.metadata.as_ref() {
        Some(DecodedMediaMetadata::Image(_)) => Some(canonical_content_hash(
            &tensor_info.shape,
            tensor_info.dtype,
            bytes,
        )),
        Some(DecodedMediaMetadata::Video(_)) => {
            #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
            if _hash_video {
                return match hash_video_content(tensor_info, bytes) {
                    Ok(hash) => Some(hash),
                    Err(error) => {
                        tracing::debug!(%error, "Skipping exact routing hash for decoded video");
                        None
                    }
                };
            }
            None
        }
        None => None,
    }
}

impl DecodedMediaData {
    /// Precompute the canonical media hash while still running on the decode
    /// thread. Images are always hashed; videos are hashed only when the
    /// request is eligible for exact MM routing.
    pub(crate) fn compute_content_hash(&mut self, hash_video: bool) {
        self.content_hash = content_hash_for_storage(&self.tensor_info, &self.data, hash_video);
    }

    pub fn into_rdma_descriptor(self, nixl_agent: &NixlAgent) -> Result<RdmaMediaDataDescriptor> {
        let source_storage = self.data;
        let content_hash = self.content_hash.map(|hash| format!("{hash:016x}"));
        let registered = nixl::register_with_nixl(source_storage, nixl_agent, None)
            .map_err(|_| anyhow::anyhow!("Failed to register storage with NIXL"))?;

        let nixl_descriptor = registered.descriptor();
        let nixl_metadata = get_nixl_metadata(nixl_agent, registered.storage())?;

        Ok(RdmaMediaDataDescriptor {
            nixl_metadata,
            nixl_descriptor: nixl_descriptor_to_wire(nixl_descriptor),
            tensor_info: self.tensor_info,
            content_hash,
            // Keep registered storage alive
            source_storage: Some(MediaStorageHandle::new(Arc::new(registered))),
        })
    }
}

// convert Array{N}<u8> to DecodedMediaData
// TODO: Array1<f32> for audio

impl<D: Dimension> TryFrom<ArrayBase<OwnedRepr<u8>, D>> for DecodedMediaData {
    type Error = anyhow::Error;

    fn try_from(array: ArrayBase<OwnedRepr<u8>, D>) -> Result<Self, Self::Error> {
        let shape = array.shape().to_vec();

        let (data_vec, _) = array.into_raw_vec_and_offset();
        let mut storage = SystemStorage::new(data_vec.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(data_vec.as_ptr(), storage.as_mut_ptr(), data_vec.len());
        }

        Ok(Self {
            data: storage,
            tensor_info: MediaTensorInfo {
                shape,
                dtype: DataType::UINT8,
                metadata: None,
            },
            content_hash: None,
        })
    }
}

// Get NIXL metadata for a descriptor
// Returns zlib-compressed, base64-encoded metadata in format: "b64:<compressed_base64>"
// This format matches what Python nixl_connect expects for RdmaMetadata.nixl_metadata
// TODO: pre-allocate a fixed NIXL-registered RAM pool so metadata can be cached on the target?
pub fn get_nixl_metadata(agent: &NixlAgent, _storage: &SystemStorage) -> Result<String> {
    // WAR: Until https://github.com/ai-dynamo/nixl/pull/970 is merged, can't use get_local_partial_md
    let nixl_md = agent.raw_agent().get_local_md()?;
    // let mut reg_desc_list = RegDescList::new(MemType::Dram)?;
    // reg_desc_list.add_storage_desc(storage)?;
    // let nixl_partial_md = agent.raw_agent().get_local_partial_md(&reg_desc_list, None)?;

    // Compress with zlib (level 6, matching Python's default)
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(6));
    encoder.write_all(&nixl_md)?;
    let compressed = encoder.finish()?;

    let b64_encoded = general_purpose::STANDARD.encode(&compressed);
    Ok(format!("b64:{}", b64_encoded))
}

// Readers send a notification with each read, and the agent keeps every notification until
// someone takes it. The frontend frees buffers by reference count and never reads them, so
// without this drain each read leaks about 100 bytes.
pub fn drain_nixl_notifications(agent: &NixlAgent) {
    let result = nixl::NotificationMap::new()
        .and_then(|mut notifs| agent.get_notifications(&mut notifs, None));
    if let Err(error) = result {
        tracing::warn!(%error, "failed to drain media-loader NIXL notifications");
    }
}

// NIXL's default progress-thread delay of 0 is the `poll()` timeout of the UCX progress
// thread, so each idle agent spins a core. With a delay, the thread sleeps in `poll()` for
// at most this long and wakes early on UCX events. Keep it short: with UCX over TCP, some
// events do not wake `poll()` and wait for the timeout.
const DEFAULT_PROGRESS_THREAD_DELAY_US: u64 = 1_000;
// NIXL 1.4.1 and earlier cast the delay to `int` before they clamp it. The cast wraps
// modulo 2^32, so values of 2^31 or more give a negative, zero, or unrelated positive
// `poll()` timeout. NIXL main clamps first. 1 s is far below that limit.
const MAX_PROGRESS_THREAD_DELAY_US: u64 = 1_000_000;

static PROGRESS_THREAD_DELAY_US: LazyLock<u64> = LazyLock::new(|| {
    progress_thread_delay_or_default(parse_or_default(
        DYN_MM_NIXL_PROGRESS_DELAY_US,
        DEFAULT_PROGRESS_THREAD_DELAY_US,
    ))
});

fn progress_thread_delay_or_default(delay_us: u64) -> u64 {
    if delay_us <= MAX_PROGRESS_THREAD_DELAY_US {
        return delay_us;
    }
    tracing::warn!(
        env_var = DYN_MM_NIXL_PROGRESS_DELAY_US,
        value = delay_us,
        max = MAX_PROGRESS_THREAD_DELAY_US,
        "invalid environment variable, using default"
    );
    DEFAULT_PROGRESS_THREAD_DELAY_US
}

pub fn get_nixl_agent() -> Result<NixlAgent> {
    let name = format!("media-loader-{}", uuid::Uuid::new_v4());
    let config = AgentConfig {
        pthr_delay_us: *PROGRESS_THREAD_DELAY_US,
        ..AgentConfig::default()
    };
    let mut nixl_agent = NixlAgent::new_with_config(&name, &config)?;
    nixl_agent
        .add_backend("UCX")
        .context("add UCX backend to media-loader NIXL agent")?;
    Ok(nixl_agent)
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_PROGRESS_THREAD_DELAY_US, DataType, MAX_PROGRESS_THREAD_DELAY_US,
        canonical_content_hash, progress_thread_delay_or_default,
    };

    #[test]
    fn progress_thread_delay_out_of_range_uses_default() {
        assert_eq!(progress_thread_delay_or_default(0), 0);
        assert_eq!(
            progress_thread_delay_or_default(MAX_PROGRESS_THREAD_DELAY_US),
            MAX_PROGRESS_THREAD_DELAY_US
        );
        assert_eq!(
            progress_thread_delay_or_default(MAX_PROGRESS_THREAD_DELAY_US + 1),
            DEFAULT_PROGRESS_THREAD_DELAY_US
        );
    }

    #[test]
    fn nixl_descriptor_wire_form_matches_nixl_types() {
        use super::{
            NixlDescriptor, nixl_descriptor_from_wire, nixl_descriptor_to_wire,
            nixl_mem_type_to_wire,
        };
        use dynamo_memory::nixl;
        for mem_type in [
            nixl::MemType::Dram,
            nixl::MemType::Vram,
            nixl::MemType::Block,
            nixl::MemType::Object,
            nixl::MemType::File,
            nixl::MemType::Unknown,
        ] {
            let native = NixlDescriptor {
                addr: 0x1000,
                size: 4096,
                mem_type,
                device_id: 3,
            };
            let wire = nixl_descriptor_to_wire(native.clone());
            assert_eq!(
                serde_json::to_value(&native).unwrap(),
                serde_json::to_value(&wire).unwrap(),
                "wire form must serialize identically for {mem_type:?}"
            );
            let back: NixlDescriptor = nixl_descriptor_from_wire(wire);
            assert_eq!(back.mem_type, mem_type);
            assert_eq!(
                nixl_mem_type_to_wire(mem_type),
                nixl_mem_type_to_wire(back.mem_type)
            );
        }
    }

    #[test]
    fn canonical_content_hash_payload_is_stable() {
        let bytes = [0_u8, 1, 2, 3, 4, 5];
        let hash = canonical_content_hash(&[1, 2, 3], DataType::UINT8, &bytes);

        assert_eq!(hash, 0x7a9b_bcb1_1a89_8630);
        assert_eq!(format!("{hash:016x}"), "7a9bbcb11a898630");
    }
}
#[cfg(all(test, feature = "mm-routing", feature = "media-ffmpeg"))]
mod video_tests {
    use super::*;
    use crate::preprocessor::media::decoders::VideoMetadata;

    fn video_info(sampled_timestamps: Vec<f64>) -> MediaTensorInfo {
        MediaTensorInfo {
            shape: vec![2, 1, 2, 3],
            dtype: DataType::UINT8,
            metadata: Some(DecodedMediaMetadata::Video(VideoMetadata {
                source_fps: 24.0,
                source_duration: 10.0,
                sampled_timestamps,
            })),
        }
    }

    #[test]
    fn video_hash_covers_metadata_shape_and_rgb_bytes() {
        let bytes = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let info = video_info(vec![0.0, 5.0]);
        let expected = hash_video_content(&info, &bytes).unwrap();

        assert_eq!(hash_video_content(&info, &bytes).unwrap(), expected);

        let changed_metadata = video_info(vec![0.0, 5.1]);
        assert_ne!(
            hash_video_content(&changed_metadata, &bytes).unwrap(),
            expected
        );

        let mut changed_shape = info.clone();
        changed_shape.shape = vec![1, 2, 2, 3];
        assert_ne!(
            hash_video_content(&changed_shape, &bytes).unwrap(),
            expected
        );

        let mut changed_bytes = bytes;
        changed_bytes[0] = 42;
        assert_ne!(hash_video_content(&info, &changed_bytes).unwrap(), expected);
    }

    #[test]
    fn video_hash_is_precomputed_from_decoded_storage() {
        let bytes = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let info = video_info(vec![0.0, 5.0]);
        let mut storage = SystemStorage::new(bytes.len()).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), storage.as_mut_ptr(), bytes.len());
        }

        assert_eq!(
            content_hash_for_storage(&info, &storage, true),
            Some(hash_video_content(&info, &bytes).unwrap())
        );
    }

    #[test]
    fn video_hash_is_skipped_when_exact_routing_is_ineligible() {
        let bytes = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let info = video_info(vec![0.0, 5.0]);
        let mut storage = SystemStorage::new(bytes.len()).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), storage.as_mut_ptr(), bytes.len());
        }

        assert_eq!(content_hash_for_storage(&info, &storage, false), None);
    }

    #[test]
    fn video_dimensions_validate_rgb_payload_layout() {
        let bytes = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let info = video_info(vec![0.0, 5.0]);
        let dimensions = video_dimensions_from_parts(&info, &bytes).unwrap();

        assert_eq!(dimensions, (2, 2, 1));

        let mut rgba = info.clone();
        rgba.shape[3] = 4;
        assert!(video_dimensions_from_parts(&rgba, &bytes).is_err());
        assert!(video_dimensions_from_parts(&info, &bytes[..11]).is_err());
    }
}
