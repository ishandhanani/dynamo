// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Control messages for native HTTP accounting. Never derived from response bytes.

use serde::{Deserialize, Serialize};

pub const CAPABILITY: &str = "sglang_generate_lifecycle_v1";
pub const ATTEMPT_HEADER: &str = "x-sglang-attempt-id";
pub const INCARNATION_HEADER: &str = "x-sglang-worker-incarnation";

pub fn endpoint_name(primary: &str) -> String {
    format!("{primary}_native_lifecycle_v1")
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Descriptor {
    pub version: u32,
    pub incarnation: String,
    pub header_overrides: bool,
    /// Same-body P/D, including parallel sampling and engine logprob handoff.
    /// Older engines omit this and remain eligible for aggregated serving.
    #[serde(default)]
    pub native_disaggregation_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Request {
    Describe,
    Attempt {
        incarnation: String,
        attempt_id: String,
        operation: Operation,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operation {
    Snapshot { after: i64 },
    Renew { lease_seconds: u32 },
    Cancel,
    Acknowledge,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Response {
    Descriptor(Descriptor),
    Snapshot(Snapshot),
    Acknowledged,
    // Unknown/expired state and incarnation conflicts are not terminal events.
    Rejected { status: u16 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub incarnation: String,
    pub attempt_id: String,
    pub stage: String,
    pub version: u64,
    pub sealed: bool,
    pub cancel_requested: bool,
    pub terminal: bool,
    pub children: Vec<Child>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Child {
    pub child_id: String,
    pub rid: String,
    pub kind: ChildKind,
    pub dp_rank: Option<u32>,
    pub dispatched: bool,
    pub prefill_complete: bool,
    pub terminal: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChildKind {
    Sample,
    Warmup,
}

pub fn is_valid_id(id: &str) -> bool {
    id.len() == 32
        && id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
