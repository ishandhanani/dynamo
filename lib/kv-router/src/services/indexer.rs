// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone KV-cache indexer, owned by `dynamo-kv-indexer`; this module keeps the old paths.

pub use dynamo_kv_indexer::http::{
    IndexerConfig, logging, parse_workers, recovery, run_server, server,
};
pub use dynamo_kv_indexer::ingest::{
    backend, listener, lookup, metrics, recording, registry, session_updates,
};
