// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The frontend's share of session affinity. The session layer (table and
//! resolver) is `dynamo_kv_router::services::selection::affinity`; this module
//! holds the frontend's session-key sources, liveness source, lease owner
//! (the response stream), event-plane replication, and error mapping.

mod host;
mod liveness;
mod replica_sync;

pub(crate) use dynamo_kv_router::services::selection::affinity::Hold;
pub use dynamo_kv_router::services::selection::affinity::{
    MAX_SESSION_AFFINITY_TTL_SECS, SessionAffinityMode,
};
pub use host::{AffinityTarget, HostAffinity, explicit_target};
pub(crate) use host::{
    affinity_error, affinity_id, from_table, invalid_argument, to_table, tracked_stream,
};
pub(crate) use liveness::DiscoveryLiveness;
#[cfg(test)]
pub(crate) use liveness::{AlwaysLive, LiveWorkers};

pub type LlmResponse =
    crate::types::Annotated<crate::protocols::common::llm_backend::LLMEngineOutput>;

/// The binding key for the subagents of one parent session.
///
/// The `\u{1}` prefix cannot appear in an HTTP header value, so no client can claim this key as
/// its own session id; the constant length keeps a long parent id from overflowing the limit.
pub(crate) fn subagent_group_affinity_id(parent_session_id: &str) -> String {
    let digest = blake3::hash(parent_session_id.as_bytes());
    format!("\u{1}sg:{}", digest.to_hex())
}

#[cfg(test)]
mod tests;
