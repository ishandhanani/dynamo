// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session-affinity replication seam: the version order and the sink a replica mesh implements so
//! an affinity table can publish bindings without knowing the transport.

pub use crate::protocols::WorkerAffinityTarget as AffinityTarget;

/// Ordering for replicated bindings: higher wins, ties broken by writer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AffinityVersion {
    pub sequence: u64,
    pub writer_id: u64,
}

/// Receives every binding this table publishes for its replicas.
pub trait AffinityReplicaSink: Send + Sync {
    fn publish(&self, session_id: &str, target: AffinityTarget, version: AffinityVersion);
}
