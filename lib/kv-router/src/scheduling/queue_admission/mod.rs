// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::hash::{BuildHasher, RandomState};
use std::sync::OnceLock;

fn keys() -> &'static RandomState {
    static KEYS: OnceLock<RandomState> = OnceLock::new();
    KEYS.get_or_init(RandomState::new)
}

/// Key of one queue lane, derived from a session identifier or a classifier-named lane.
///
/// A class queue has one shared lane plus one lane per session, or per classifier-named lane,
/// that has queued requests. Lanes share one order: the best head across every lane of a class dispatches first. A lane whose
/// head cannot dispatch, because the dispatch gate holds it or its eligible workers are full,
/// holds only that lane; the shared lane and the other sessions keep draining. The actor
/// assigns a session lane to every request that carries a session context; a request without
/// one waits in the lane its classifier named, if any, else in the shared lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionLane(u64);

impl SessionLane {
    /// Derive the lane key for a session identifier, with a hash keyed once per process:
    /// session identifiers come from clients, so one client must not be able to pick an
    /// identifier that lands in another session's lane. Two identifiers that still collide
    /// share a lane, its order and its holds.
    pub fn new(session_id: &str) -> Self {
        Self(keys().hash_one(session_id))
    }

    /// Derive the key of a lane a request classifier names. Classifier lanes hash in their own
    /// domain, so a lane name never shares a key with a session identifier on purpose.
    pub fn named(lane: &str) -> Self {
        Self(keys().hash_one(("dynamo.classifier.lane", lane)))
    }
}
