// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "bench")]
use std::time::Instant;

use tokio::sync::oneshot;

pub use dynamo_router_types::cache::{
    IndexerQueryRequest, IndexerQueryResponse, IndexerRecordRoutingDecisionRequest,
    IndexerRecordRoutingDecisionResponse, KvRouterError, KvStateAgentIdentity, KvStateAgentStatus,
    KvStateAttachmentStatus, KvStateProtocolVersion, KvStateRecoveryReceipt, MatchDetails,
    MaybeError, RoutingDecisionHashes, WireLowerTierMatchDetails, WireOverlapScores,
    WireTieredMatchDetails, WorkerKvQueryKind, WorkerKvQueryRequest, WorkerKvQueryResponse,
};
use dynamo_router_types::protocols::*;
use dynamo_tokens::SequenceHash;

#[cfg(feature = "bench")]
use super::{EventCompletionBuffer, EventCompletionWriter, ObservationSeal};

/// Shared structural anchor used by branch-sharded routing when a routed
/// subtree starts on a different shard from its parent prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct AnchorRef {
    pub anchor_id: ExternalSequenceBlockHash,
    pub anchor_local_hash: LocalBlockHash,
    pub anchor_depth: usize,
}

/// Worker task payload that installs an [`AnchorRef`] into a shard-local
/// backend before dependent suffix events are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct AnchorTask {
    pub anchor_id: ExternalSequenceBlockHash,
    pub anchor_local_hash: LocalBlockHash,
    pub anchor_depth: usize,
}

// -------
// Distributed router - Worker KV Query types
// -------

// -------
// Standalone indexer query types (request plane)
// -------

/// Endpoint name for the standalone KV indexer query service.
pub const KV_INDEXER_QUERY_ENDPOINT: &str = "kv_indexer_query";
/// Endpoint name for recording approximate-mode routing decisions on a remote indexer.
pub const KV_INDEXER_RECORD_ROUTING_DECISION_ENDPOINT: &str = "kv_indexer_record_routing_decision";

impl From<&super::lower_tier::LowerTierMatchDetails> for WireLowerTierMatchDetails {
    fn from(d: &super::lower_tier::LowerTierMatchDetails) -> Self {
        Self {
            hits: d.hits.iter().map(|(w, h)| (*w, *h)).collect(),
        }
    }
}

impl From<WireLowerTierMatchDetails> for super::lower_tier::LowerTierMatchDetails {
    fn from(w: WireLowerTierMatchDetails) -> Self {
        // `next_continuations` is server-side intermediate state; consumers of
        // the tiered result never read it, so we reconstruct an empty map on
        // the wire-inbound path.
        Self {
            hits: w.hits.into_iter().collect(),
            next_continuations: Default::default(),
            kv_transfer_candidates: None,
            kv_transfer_extensions: None,
        }
    }
}

/// A request to find matches in the Radix Tree.
pub struct MatchRequest {
    /// A vector of `LocalBlockHash` representing the sequence to match.
    pub sequence: Vec<LocalBlockHash>,
    /// A boolean indicating whether to exit early if a single match is found.
    pub early_exit: bool,
    /// A channel sender to send the `OverlapScores` response.
    pub resp: oneshot::Sender<OverlapScores>,
    /// Timestamp when the request was created (for queue wait time measurement)
    #[cfg(feature = "bench")]
    pub created_at: Instant,
}

impl MatchRequest {
    pub(super) fn new(
        sequence: Vec<LocalBlockHash>,
        early_exit: bool,
        resp: oneshot::Sender<OverlapScores>,
    ) -> Self {
        Self {
            sequence,
            early_exit,
            resp,
            #[cfg(feature = "bench")]
            created_at: Instant::now(),
        }
    }
}

/// A request to find matches while also returning continuation metadata.
pub struct MatchDetailsRequest {
    /// A vector of `LocalBlockHash` representing the sequence to match.
    pub sequence: Vec<LocalBlockHash>,
    /// A boolean indicating whether to exit early if a single match is found.
    pub early_exit: bool,
    /// When true, retain the matched root-aligned external hash chain for KV transfer hints.
    pub retain_kv_transfer_chain: bool,
    /// A channel sender to send the `MatchDetails` response.
    pub resp: oneshot::Sender<MatchDetails>,
}

/// A request to test one worker's ownership of an external block hash.
pub struct ContainsWorkerBlockRequest {
    pub worker: WorkerWithDpRank,
    pub block_hash: ExternalSequenceBlockHash,
    pub resp: oneshot::Sender<bool>,
}

impl MatchDetailsRequest {
    pub(super) fn new(
        sequence: Vec<LocalBlockHash>,
        early_exit: bool,
        retain_kv_transfer_chain: bool,
        resp: oneshot::Sender<MatchDetails>,
    ) -> Self {
        Self {
            sequence,
            early_exit,
            retain_kv_transfer_chain,
            resp,
        }
    }
}

/// A request to dump the tree as events
pub struct DumpRequest {
    /// Channel to send the dumped events
    pub resp: oneshot::Sender<Vec<RouterEvent>>,
}

/// A request to wait until all previously submitted work is applied.
pub struct FlushRequest {
    /// Channel to acknowledge completion.
    pub resp: oneshot::Sender<()>,
}

/// A request to get all workers currently tracked
pub struct GetWorkersRequest {
    /// Channel to send the worker IDs
    pub resp: oneshot::Sender<Vec<WorkerId>>,
}

#[derive(Debug, Default)]
pub struct WorkerLookupStats {
    pub worker_blocks: Vec<(WorkerWithDpRank, usize)>,
}

impl WorkerLookupStats {
    pub fn from_worker_block_counts(
        counts: impl IntoIterator<Item = (WorkerWithDpRank, usize)>,
    ) -> Self {
        Self {
            worker_blocks: counts
                .into_iter()
                .filter(|(_, block_count)| *block_count > 0)
                .collect(),
        }
    }

    pub fn worker_count(&self) -> usize {
        self.worker_blocks.len()
    }

    pub fn block_count(&self) -> usize {
        self.worker_blocks
            .iter()
            .map(|(_, block_count)| *block_count)
            .sum()
    }

    pub fn block_count_for_worker(&self, worker: WorkerWithDpRank) -> Option<usize> {
        self.worker_blocks
            .iter()
            .find_map(|(candidate, block_count)| (*candidate == worker).then_some(*block_count))
    }
}

pub enum WorkerTask {
    Event(RouterEvent),
    EventWithAck {
        event: RouterEvent,
        resp: oneshot::Sender<bool>,
    },
    ApproximateLru(super::ApproximateLruTask),
    #[cfg(feature = "bench")]
    InstallObservation {
        writer: EventCompletionWriter,
        resp: oneshot::Sender<bool>,
    },
    #[cfg(feature = "bench")]
    ObservedEvent {
        event: RouterEvent,
        correlation_id: u32,
    },
    #[cfg(feature = "bench")]
    SealObservation(oneshot::Sender<Option<ObservationSeal>>),
    #[cfg(feature = "bench")]
    HarvestObservation(oneshot::Sender<EventCompletionBuffer>),
    Anchor {
        worker: WorkerWithDpRank,
        anchor: AnchorTask,
    },
    /// Permanently remove a worker from tracking.
    RemoveWorker {
        worker_id: WorkerId,
        /// True for the one shared-state backend task that owns structural cleanup.
        sweep_tree: bool,
        /// Acknowledges completion of this lane's cold-path removal phase.
        resp: oneshot::Sender<()>,
    },
    /// Remove a single dp_rank for a worker.
    RemoveWorkerDpRank {
        worker_id: WorkerId,
        dp_rank: DpRank,
        /// True for the one shared-state backend task that owns structural cleanup.
        sweep_tree: bool,
    },
    DumpEvents(oneshot::Sender<anyhow::Result<Vec<RouterEvent>>>),
    Stats(oneshot::Sender<WorkerLookupStats>),
    ContainsWorkerBlock {
        worker: WorkerWithDpRank,
        block_hash: ExternalSequenceBlockHash,
        resp: oneshot::Sender<bool>,
    },
    Flush(oneshot::Sender<()>),
    Terminate,
}

/// A request to process a routing decision.
pub(super) struct RoutingDecisionRequest {
    pub(super) worker: WorkerWithDpRank,
    pub(super) local_hashes: Vec<LocalBlockHash>,
    pub(super) sequence_hashes: Vec<SequenceHash>,
}
