// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::hash::{Hash, Hasher};

/// Key of one per-session queue lane, derived from the session identifier.
///
/// A class queue has one shared lane plus one lane per session that has queued requests. Lanes
/// share one order: the best head across every lane of a class dispatches first. A lane whose
/// head cannot dispatch, because the dispatch gate holds it or its eligible workers are full,
/// holds only that lane; the shared lane and the other sessions keep draining. The actor
/// assigns session lanes only while a dispatch gate is installed.
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
}
