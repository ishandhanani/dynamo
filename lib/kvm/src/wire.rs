// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The hint wire: SGLang's `PrefetchReqInput` / `PrefetchReqOutput` as msgpack.
//!
//! SGLang's `io_struct` types are msgspec structs declared `tag=True, array_like=True`,
//! so each message is one msgpack array whose first element is the class name and
//! whose remaining elements are the fields in declaration order, base class first.
//! Every field is written, including `None` ones. The structs here mirror that layout
//! field for field; `rmp_serde::to_vec` writes a struct as an array, which is the
//! same encoding, and the tests pin the exact bytes msgspec produces.
//!
//! Request (5 elements): `["PrefetchReqInput", rid, http_worker_ipc, token_ids, dp_rank]`.
//! Reply (7 elements): `["PrefetchReqOutput", rid, http_worker_ipc, matched_tokens,
//! loaded_tokens, elapsed_ms, error]`.

use serde::{Deserialize, Serialize};

/// msgspec tag of the request struct.
pub const PREFETCH_REQUEST_TAG: &str = "PrefetchReqInput";
/// msgspec tag of the reply struct.
pub const PREFETCH_REPLY_TAG: &str = "PrefetchReqOutput";

/// `class PrefetchReqInput(BaseReq)`: ask the scheduler to load the shared-store (L3)
/// pages for `token_ids` into device cache without a forward pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefetchReqInput {
    /// Always [`PREFETCH_REQUEST_TAG`].
    pub tag: String,
    /// `BaseReq.rid`: echoed in the reply, so a sender can match replies to requests.
    pub rid: String,
    /// `BaseReq.http_worker_ipc`: in-process routing only; always `None` on this wire.
    pub http_worker_ipc: Option<String>,
    pub token_ids: Vec<u32>,
    /// Data-parallel attention group that should hold the pages; `None` means group 0.
    pub dp_rank: Option<u32>,
}

impl PrefetchReqInput {
    pub fn new(rid: impl Into<String>, token_ids: Vec<u32>, dp_rank: Option<u32>) -> Self {
        Self {
            tag: PREFETCH_REQUEST_TAG.to_string(),
            rid: rid.into(),
            http_worker_ipc: None,
            token_ids,
            dp_rank,
        }
    }
}

/// `class PrefetchReqOutput(BaseReq)`: the scheduler's answer to one request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrefetchReqOutput {
    /// Always [`PREFETCH_REPLY_TAG`].
    pub tag: String,
    pub rid: String,
    pub http_worker_ipc: Option<String>,
    /// Page-aligned prefix resident on device after the prefetch.
    pub matched_tokens: u64,
    /// Tokens the scheduler loaded from the shared store into device cache.
    pub loaded_tokens: u64,
    /// Scheduler wall-clock from dispatch to the load landing.
    pub elapsed_ms: f64,
    /// `None` on success; otherwise why nothing (more) was loaded.
    pub error: Option<String>,
}

/// Encode one request as the single msgpack frame the scheduler's hint socket reads.
pub fn encode_prefetch_request(request: &PrefetchReqInput) -> anyhow::Result<Vec<u8>> {
    Ok(rmp_serde::to_vec(request)?)
}

/// Decode one reply frame. Fails on a malformed frame, a field-count mismatch, or a
/// tag other than [`PREFETCH_REPLY_TAG`]; a scheduler-side `error` is returned in the
/// struct, not as `Err`, so the caller can still route it by `rid`.
pub fn decode_prefetch_reply(frame: &[u8]) -> anyhow::Result<PrefetchReqOutput> {
    let reply: PrefetchReqOutput = rmp_serde::from_slice(frame)?;
    if reply.tag != PREFETCH_REPLY_TAG {
        anyhow::bail!("unexpected hint reply tag {:?}", reply.tag);
    }
    Ok(reply)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Produced by `msgspec.msgpack.Encoder` on the SGLang structs
    // (`BaseReq(msgspec.Struct, tag=True, kw_only=True, array_like=True)` with
    // `rid: Optional[str] = None`, `http_worker_ipc: Optional[str] = None`;
    // `PrefetchReqInput` adds `rid: str`, `token_ids: List[int]`,
    // `dp_rank: Optional[int] = None`; `PrefetchReqOutput` adds `rid: str`,
    // `matched_tokens: int = 0`, `loaded_tokens: int = 0`, `elapsed_ms: float = 0.0`,
    // `error: Optional[str] = None`). Inputs: `rid="kvm-1-2"`,
    // `token_ids=[1, 2, 300, 70000]`, replies `(5312, 5312, 112.5)` and
    // `error="no external linker"`.
    const REQUEST_BYTES: &str =
        "95b05072656665746368526571496e707574a76b766d2d312d32c0940102cd012cce00011170c0";
    const REQUEST_DP_RANK_1_BYTES: &str =
        "95b05072656665746368526571496e707574a76b766d2d312d32c0940102cd012cce0001117001";
    const REPLY_OK_BYTES: &str =
        "97b150726566657463685265714f7574707574a76b766d2d312d32c0cd14c0cd14c0cb405c200000000000c0";
    const REPLY_ERROR_BYTES: &str = "97b150726566657463685265714f7574707574a76b766d2d312d32c00000cb0000000000000000b26e6f2065787465726e616c206c696e6b6572";

    fn unhex(hex: &str) -> Vec<u8> {
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn request_encodes_byte_for_byte_like_msgspec() {
        let request = PrefetchReqInput::new("kvm-1-2", vec![1, 2, 300, 70_000], None);
        assert_eq!(
            encode_prefetch_request(&request).unwrap(),
            unhex(REQUEST_BYTES)
        );
        let request = PrefetchReqInput::new("kvm-1-2", vec![1, 2, 300, 70_000], Some(1));
        assert_eq!(
            encode_prefetch_request(&request).unwrap(),
            unhex(REQUEST_DP_RANK_1_BYTES)
        );
    }

    #[test]
    fn request_round_trips_with_every_field_present() {
        let request = PrefetchReqInput::new("r1", vec![5, 6], Some(3));
        let frame = encode_prefetch_request(&request).unwrap();
        // 5 elements: tag, rid, http_worker_ipc, token_ids, dp_rank.
        assert_eq!(frame[0], 0x95);
        let decoded: PrefetchReqInput = rmp_serde::from_slice(&frame).unwrap();
        assert_eq!(decoded, request);
        assert_eq!(decoded.http_worker_ipc, None);
        assert_eq!(decoded.dp_rank, Some(3));
    }

    #[test]
    fn reply_decodes_msgspec_success_bytes() {
        let reply = decode_prefetch_reply(&unhex(REPLY_OK_BYTES)).unwrap();
        assert_eq!(
            reply,
            PrefetchReqOutput {
                tag: PREFETCH_REPLY_TAG.to_string(),
                rid: "kvm-1-2".to_string(),
                http_worker_ipc: None,
                matched_tokens: 5312,
                loaded_tokens: 5312,
                elapsed_ms: 112.5,
                error: None,
            }
        );
    }

    #[test]
    fn reply_decodes_msgspec_error_bytes_with_its_rid() {
        let reply = decode_prefetch_reply(&unhex(REPLY_ERROR_BYTES)).unwrap();
        assert_eq!(reply.rid, "kvm-1-2");
        assert_eq!(reply.matched_tokens, 0);
        assert_eq!(reply.loaded_tokens, 0);
        assert_eq!(reply.elapsed_ms, 0.0);
        assert_eq!(reply.error.as_deref(), Some("no external linker"));
    }

    #[test]
    fn reply_rejects_wrong_tag_and_wrong_field_count() {
        let wrong_tag = rmp_serde::to_vec(&(
            PREFETCH_REQUEST_TAG,
            "r1",
            Option::<String>::None,
            0u64,
            0u64,
            0.0f64,
            Option::<String>::None,
        ))
        .unwrap();
        let error = decode_prefetch_reply(&wrong_tag).unwrap_err();
        assert!(
            error.to_string().contains("unexpected hint reply tag"),
            "{error}"
        );

        // Six elements: a reply written before `error` existed must not decode as one.
        let six_fields = rmp_serde::to_vec(&(
            PREFETCH_REPLY_TAG,
            "r1",
            Option::<String>::None,
            0u64,
            0u64,
            0.0f64,
        ))
        .unwrap();
        assert!(decode_prefetch_reply(&six_fields).is_err());
        assert!(decode_prefetch_reply(b"not msgpack").is_err());
    }
}
