// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Helpers shared by the standalone services, re-exported from the crates that own them.

#[cfg(feature = "standalone-indexer")]
pub(crate) use dynamo_kv_indexer::ingest::zmq;

#[cfg(any(feature = "standalone-slot-tracker", feature = "standalone-selection"))]
pub(crate) use dynamo_slot_tracker::http::envelope as http;
#[cfg(any(feature = "standalone-slot-tracker", feature = "standalone-selection"))]
pub(crate) use dynamo_slot_tracker::http::replica_sync as replica_sync_http;
#[cfg(any(feature = "standalone-slot-tracker", feature = "standalone-selection"))]
pub(crate) use dynamo_slot_tracker::mesh as replica_sync;
