// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Discovery contracts for persistent KV state sources and engine attachments.
//!
//! Advertisements are immutable hints, not liveness. Reconciliation activates
//! a projection only after the advertised recovery/control endpoint returns an
//! exact matching status and recovery reaches the attachment cursor barrier.
//!
//! The wire contracts (topics, advertisements, intents, host status, the
//! ingress protocol, and the projection resolver) live in
//! `dynamo_llm_worker::kv_state` and are re-exported here at their historical
//! paths. This module keeps only the frontend-side projection controller,
//! which holds the shared router [`Indexer`].

use dynamo_kv_router::protocols::ResidencyProjectionError;

pub use dynamo_llm_worker::kv_state::*;

use crate::kv_router::indexer::Indexer;

/// Model-level owner of the one immutable projection snapshot consumed by lookups.
pub struct KvStateProjectionController {
    indexer: Indexer,
}

impl KvStateProjectionController {
    pub fn new(indexer: Indexer) -> Self {
        Self { indexer }
    }

    pub fn publish<'a>(
        &self,
        resolutions: impl IntoIterator<Item = &'a KvStateProjectionResolution>,
    ) -> Result<(), ResidencyProjectionError> {
        let projection = aggregate_residency_projection(resolutions)?;
        self.indexer.set_residency_projection(projection);
        Ok(())
    }
}
