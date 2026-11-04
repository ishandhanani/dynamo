// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::hash::{Hash, Hasher};

use crate::protocols::WorkerWithDpRank;

/// Which lane of its policy class a queued request waits in.
///
/// Lanes share one order: the best head across every lane of a class dispatches first. A lane
/// whose head cannot dispatch holds only that lane; the shared lane and the other lanes keep
/// draining.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WorkerPlacement {
    /// Preserve the request's existing routing constraints and wait in the class's shared lane.
    Any,
    /// Add an exact-worker constraint and wait in that worker's lane. The router validates it
    /// against the request's existing constraints before dispatch.
    Exact(WorkerWithDpRank),
    /// No worker constraint; wait in a per-session lane so a head the dispatch gate holds, or
    /// one whose eligible workers are full, blocks only its own session.
    Session(SessionLane),
}

/// Key of one per-session queue lane, derived from the session identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionLane(u64);

impl SessionLane {
    /// Derive the lane key for a session identifier. Two identifiers may share a lane; that
    /// couples their order within the class and nothing else.
    pub fn new(session_id: &str) -> Self {
        let mut hasher = rustc_hash::FxHasher::default();
        session_id.hash(&mut hasher);
        Self(hasher.finish())
    }

    pub(crate) fn key(self) -> u64 {
        self.0
    }
}
