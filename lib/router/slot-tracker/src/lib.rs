// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Slot tracker: the router's active-sequence write model, the derived prompt-registry read
//! model, the replica mesh that keeps peers' load views in sync (`mesh`, behind
//! `replica-sync-zmq`), and (behind `http`) the standalone slot-tracker service.

mod block_tracker;
mod compressed_path_arena;
pub mod multi_worker;
mod prefill_tracker;
mod prompt_membership_trie;
mod prompt_registry;
mod replica_sync;
mod request_maps;
mod sharded_lock;
pub mod single;
pub mod topology;

pub(crate) mod cleanup;

#[cfg(feature = "replica-sync-zmq")]
pub mod mesh;
#[cfg(feature = "replica-sync-zmq")]
pub mod registry;

#[cfg(feature = "http")]
pub mod http;

pub use multi_worker::*;
pub use prefill_tracker::PrefillTokenDeltas;
pub use prompt_registry::{PotentialLoadMaps, WorkerLoadProjection};
pub use single::*;
