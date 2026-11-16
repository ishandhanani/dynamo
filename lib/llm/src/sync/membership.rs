// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV event-source membership: which publisher feeds the index for which worker and rank, the
//! watch that follows it through discovery, and the vocabulary the KV state agents advertise.

pub mod kv_source_membership;
pub use kv_source_membership::{
    KvEventSource, KvSourceAdvertisement, KvSourceAmbiguity, KvSourceId, KvSourceKey,
    KvSourceMembership, KvSourceMembershipError, KvSourceMembershipView, KvSourceStatus,
    KvSourceTransition, KvStateEndpointResolution, PublisherId, resolve_kv_state_endpoint,
};

mod kv_source_watch;
pub(crate) use kv_source_watch::KvSourceMembershipCoordinator;
pub use kv_source_watch::KvSourceMembershipWatch;

pub mod kv_state_agent;
