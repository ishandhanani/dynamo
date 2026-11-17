// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo KV State Agent
//!
//! The KV state agent turns an engine's raw KV events into router events outside the engine
//! process: the per-slot agent (`state_agent`), the standalone host that multiplexes agents
//! (`state_agent_host`), and the worker-side owner of the attachment intents
//! (`attachment_owner`).
//!
//! The agent shares the stages of the KV event publisher in `dynamo-backend-common`
//! (`kv_publisher`). This crate sits above `dynamo-backend-common` and `dynamo-sync` and never
//! depends on `dynamo-llm`.

mod attachment_owner;
mod state_agent;
mod state_agent_host;

pub use attachment_owner::{KvStateAttachmentDescriptor, KvStateAttachmentOwner};
pub use state_agent::{
    KvStateAgent, KvStateAgentAttachmentConfig, KvStateAgentConfig, KvStateAgentSlotConfig,
    KvStateAgentSockets, KvStateAgentVllmSource, resolve_stable_dp_slot_id,
};
pub use state_agent_host::{
    DEFAULT_KV_STATE_AGENT_MAX_SLOTS, DEFAULT_KV_STATE_AGENT_SLOTS_PER_SOCKET, KvStateAgentHost,
    KvStateAgentHostConfig,
};
