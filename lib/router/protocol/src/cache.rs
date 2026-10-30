// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cache seam: the indexer's wire DTOs, query and recovery protocol types, and the shared-cache
//! trait. Index data structures and channel-bearing request types stay in the indexer crate.

use async_trait::async_trait;
use dynamo_tokens::SequenceHash;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::identity::CacheOwnerId;
use crate::kv_hints::KvTransferCandidates;
use crate::protocols::*;

/// Trait for types that may represent an error response.
/// Used for RPC-style responses that can indicate success or failure.
pub trait MaybeError {
    /// Construct an instance from an error.
    fn from_err(err: impl std::error::Error + 'static) -> Self;
    /// Convert to an error instance if this represents an error.
    fn err(&self) -> Option<Box<dyn std::error::Error + Send + Sync>>;
}

/// Errors that can occur in the KV Router.
#[derive(Debug, thiserror::Error)]
pub enum KvRouterError {
    #[error("Block not found")]
    BlockNotFound,

    #[error("Indexer is offline")]
    IndexerOffline,

    #[error("Indexer dropped the request")]
    IndexerDroppedRequest,

    #[error("Prune operation failed: {0}")]
    PruneFailed(String),

    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

/// Immutable protocol selected for one state-agent lifecycle.
///
/// A live worker/handler never switches this value. Upgrading the protocol
/// requires a fresh publisher incarnation and discovery advertisement.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum KvStateProtocolVersion {
    V2,
}

/// Exact identity returned by the state agent's callable status endpoint.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct KvStateAgentIdentity {
    pub cache_owner_id: CacheOwnerId,
    pub publisher_id: u64,
    pub protocol_version: KvStateProtocolVersion,
}

/// Current engine attachment observed by a state agent.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct KvStateAttachmentStatus {
    pub generation: u64,
    pub worker: WorkerWithDpRank,
    pub ready: bool,
    pub ready_at_outbound_cursor: u64,
}

/// Lightweight liveness response served without entering the dump queue.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct KvStateAgentStatus {
    pub identity: KvStateAgentIdentity,
    pub attachment: Option<KvStateAttachmentStatus>,
    /// False after a CacheOwner local/recovery transaction becomes uncertain.
    pub cache_owner_ready: bool,
    pub outbound_cursor: u64,
}

/// Proof that recovery completed against one exact state-agent incarnation.
///
/// The attachment generation is present when the response contains ephemeral
/// Worker ownership. CacheOwner-only recovery is fenced by the source identity.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct KvStateRecoveryReceipt {
    pub identity: KvStateAgentIdentity,
    pub attachment_generation: Option<u64>,
    pub recovered_through_cursor: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "snake_case")]
pub enum WorkerKvQueryKind {
    #[default]
    Recovery,
    StateAgentRecovery {
        expected: KvStateAgentIdentity,
        expected_attachment_generation: Option<u64>,
    },
    Status {
        expected: KvStateAgentIdentity,
        expected_attachment_generation: Option<u64>,
    },
}

/// Request to query a worker's local KV indexer.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WorkerKvQueryRequest {
    /// The worker ID of the worker to query.
    pub worker_id: WorkerId,
    /// Data-parallel rank owned by this worker query endpoint.
    pub dp_rank: DpRank,

    /// Start event ID (inclusive). If `None`, dumps entire tree.
    pub start_event_id: Option<u64>,
    /// End event ID (inclusive). Used for validation and `TooNew` responses.
    /// Successful buffer-backed recovery may still return through the current
    /// newest buffered event.
    pub end_event_id: Option<u64>,

    /// Opt in to an explicit [`WorkerKvQueryResponse::TreeDumpFailed`] result.
    /// Named MessagePack clients that predate this field deserialize it as false.
    #[serde(default)]
    pub supports_tree_dump_failed: bool,

    /// Recovery remains the default for old clients. Status is an explicit v2
    /// control-path request and never queues behind a full tree dump.
    #[serde(default)]
    pub kind: WorkerKvQueryKind,
}

/// Response from a worker's local KV indexer.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[non_exhaustive]
pub enum WorkerKvQueryResponse {
    /// Callable liveness and identity fence for a v2 state agent.
    Status(KvStateAgentStatus),
    /// Recovery payload and proof returned only for an explicit state-agent
    /// recovery request after identity/attachment revalidation.
    StateAgentRecovery {
        response: Box<WorkerKvQueryResponse>,
        receipt: KvStateRecoveryReceipt,
    },
    /// Events served from the circular buffer with original event IDs. The batch
    /// is recovery-equivalent to replaying the requested `start_event_id` through
    /// the current buffered tail. If the rank stream contains one or more `Cleared`
    /// all-domain `Cleared` events, the source may omit events before the latest such
    /// clear while preserving that clear event and all following events. Domain-scoped
    /// clears remain ordinary ordered events. `last_event_id` is taken from the same
    /// buffer snapshot and should be used as the recovery watermark after applying the
    /// batch.
    Events {
        events: Vec<RouterEvent>,
        last_event_id: u64,
    },
    /// Full replay-ordered tree dump (with synthetic 0-indexed event IDs).
    ///
    /// Parent-addressed stored events appear after the event that introduces their parent.
    /// Consumers may therefore rebuild exact source state in one pass. Indexers that describe
    /// state positionally may use `start_position`; consumers that require parent-addressed
    /// replay must reject unsupported non-zero orphan positions rather than treating them as
    /// independent roots.
    /// Includes `last_event_id`: the newest real event ID in the worker's buffer
    /// at the time of the dump, so the caller can set its tracking cursor correctly.
    TreeDump {
        events: Vec<RouterEvent>,
        last_event_id: u64,
        /// Scope authoritatively represented by this snapshot. Legacy responses
        /// omitted this field and always represented the whole rank.
        #[serde(default)]
        reset_scope: ResetScope,
    },
    /// The exact tree dump could not be produced. This is distinct from an
    /// authoritative empty tree; recovery must leave indexed state and its
    /// admission cursor unchanged.
    TreeDumpFailed { last_event_id: u64, message: String },
    /// Requested range is newer than available data
    TooNew {
        requested_start: Option<u64>,
        requested_end: Option<u64>,
        newest_available: u64,
    },
    /// Invalid range: end_id < start_id
    InvalidRange { start_id: u64, end_id: u64 },
    /// Query failed on worker (serialized error)
    Error(String),
}

impl MaybeError for WorkerKvQueryResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        WorkerKvQueryResponse::Error(err.to_string())
    }

    fn err(&self) -> Option<Box<dyn std::error::Error + Send + Sync>> {
        match self {
            WorkerKvQueryResponse::Error(msg) => Some(Box::new(std::io::Error::other(msg.clone()))),
            _ => None,
        }
    }
}

/// Request to query a served KV indexer for overlap scores.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IndexerQueryRequest {
    /// Model name to query the indexer for.
    pub model_name: String,
    /// Block hashes to find matches for in the radix tree.
    pub block_hashes: Vec<LocalBlockHash>,
    /// When true, the server skips the lower-tier walk and returns only the
    /// device-tier overlap. Older clients that omit this field default to
    /// `false`, preserving the full tiered response.
    #[serde(default)]
    pub device_only: bool,
}

/// Wire-friendly overlap scores for JSON serialization.
/// `OverlapScores` uses `FxHashMap<WorkerWithDpRank, _>` which can't be
/// serialized as JSON (struct keys aren't valid JSON map keys), so we flatten
/// to vecs of tuples for the wire protocol.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct WireOverlapScores {
    pub scores: Vec<(WorkerWithDpRank, u32)>,
    pub frequencies: Vec<usize>,
}

impl From<OverlapScores> for WireOverlapScores {
    fn from(s: OverlapScores) -> Self {
        Self {
            scores: s.scores.into_iter().collect(),
            frequencies: s.frequencies,
        }
    }
}

impl From<WireOverlapScores> for OverlapScores {
    fn from(w: WireOverlapScores) -> Self {
        Self {
            scores: w.scores.into_iter().collect(),
            frequencies: w.frequencies,
        }
    }
}

/// Wire-friendly lower-tier match payload for JSON serialization.
///
/// Mirrors `LowerTierMatchDetails.hits`. `next_continuations` and KV transfer
/// candidates are server-side intermediate state and are not carried over the wire.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct WireLowerTierMatchDetails {
    pub hits: Vec<(WorkerWithDpRank, usize)>,
}

/// Wire-friendly tiered match payload: device overlap plus per-tier hits.
///
/// Lower tiers are a `Vec<(StorageTier, _)>` rather than a map so we never
/// depend on `StorageTier` being a JSON-legal map key. Each `StorageTier` is
/// expected to appear at most once; the inbound conversion warns and keeps the
/// last entry if duplicates are observed.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct WireTieredMatchDetails {
    pub device: WireOverlapScores,
    pub lower_tier: Vec<(StorageTier, WireLowerTierMatchDetails)>,
}

/// Response from a served KV indexer query.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum IndexerQueryResponse {
    /// Tiered match details: device overlap plus per-tier hits.
    TieredScores(WireTieredMatchDetails),
    /// An error occurred processing the query.
    Error(String),
}

impl MaybeError for IndexerQueryResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        IndexerQueryResponse::Error(err.to_string())
    }

    fn err(&self) -> Option<Box<dyn std::error::Error + Send + Sync>> {
        match self {
            IndexerQueryResponse::Error(msg) => Some(Box::new(std::io::Error::other(msg.clone()))),
            _ => None,
        }
    }
}

/// Request to record a routing decision on a served approximate-mode indexer.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IndexerRecordRoutingDecisionRequest {
    /// Model name to update.
    pub model_name: String,
    /// Selected worker for this routing decision.
    pub worker: WorkerWithDpRank,
    /// Locally-computed block hashes for the routed request.
    pub local_hashes: Vec<LocalBlockHash>,
    /// Locally-computed rolling sequence hashes for the routed request.
    pub sequence_hashes: Vec<SequenceHash>,
}

/// Precomputed hashes for recording a route-time indexer update.
#[derive(Debug, Clone)]
pub struct RoutingDecisionHashes {
    pub local_hashes: Vec<LocalBlockHash>,
    pub sequence_hashes: Vec<SequenceHash>,
}

impl RoutingDecisionHashes {
    pub fn from_local_hashes(local_hashes: Vec<LocalBlockHash>) -> Self {
        let sequence_hashes = compute_seq_hash_for_block(&local_hashes);
        Self {
            local_hashes,
            sequence_hashes,
        }
    }
}

/// Response from a served approximate-mode routing-decision endpoint.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum IndexerRecordRoutingDecisionResponse {
    Recorded,
    Error(String),
}

impl MaybeError for IndexerRecordRoutingDecisionResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        IndexerRecordRoutingDecisionResponse::Error(err.to_string())
    }

    fn err(&self) -> Option<Box<dyn std::error::Error + Send + Sync>> {
        match self {
            IndexerRecordRoutingDecisionResponse::Error(msg) => {
                Some(Box::new(std::io::Error::other(msg.clone())))
            }
            _ => None,
        }
    }
}

/// Rich non-wire query result for router-local device tier lookups.
#[derive(Debug, Clone, Default)]
pub struct MatchDetails {
    /// Existing overlap scores used by scheduling.
    pub overlap_scores: OverlapScores,
    /// Last matched device sequence hash per worker, used to seed lower-tier queries.
    pub last_matched_hashes: FxHashMap<WorkerWithDpRank, ExternalSequenceBlockHash>,
    /// Optional root-aligned device candidates used to build compact KV transfer hints.
    pub kv_transfer_candidates: Option<KvTransferCandidates>,
}

impl MatchDetails {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn retain_kv_transfer_candidates(
        &mut self,
        mut block_hashes: Vec<ExternalSequenceBlockHash>,
    ) {
        let mut owner_prefix_blocks: Vec<_> = self
            .overlap_scores
            .scores
            .iter()
            .filter_map(|(worker, blocks)| {
                let blocks = usize::try_from(*blocks).ok()?;
                (blocks > 0 && blocks <= block_hashes.len()).then_some(((*worker).into(), blocks))
            })
            .collect();
        if block_hashes.is_empty() || owner_prefix_blocks.is_empty() {
            return;
        }
        let max_owner_prefix_blocks = owner_prefix_blocks
            .iter()
            .map(|(_, blocks)| *blocks)
            .max()
            .unwrap_or(0);
        block_hashes.truncate(max_owner_prefix_blocks);
        owner_prefix_blocks.sort_unstable_by_key(|(worker, _)| *worker);
        self.kv_transfer_candidates = Some(KvTransferCandidates {
            block_hashes,
            owner_prefix_blocks,
            routing_snapshot: None,
        });
    }
}

/// Trait for querying an external shared KV cache pool.
///
/// Implementations check which blocks/pages from a request's token sequence
/// exist in the shared cache. The returned `SharedCacheHits` describes which
/// block positions are available externally (and thus cheaper to prefill).
#[async_trait]
pub trait SharedKvCache: Send + Sync {
    /// Query which blocks exist in the shared cache for the given token sequence.
    ///
    /// `cache_namespace` must be honored by implementations to avoid scoring
    /// shared-cache hits across isolated cache namespaces.
    async fn check_blocks(
        &self,
        tokens: &[u32],
        block_size: u32,
        cache_namespace: Option<&str>,
    ) -> Result<SharedCacheHits, KvRouterError>;
}

/// Per-shard size snapshot returned by the indexer `shard_sizes` query.
///
/// `worker_count` and `block_count` are always populated. `node_count` is
/// reserved for backends that can expose a structural node count.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ShardSizeSnapshot {
    /// Zero-based shard index.
    pub shard_idx: usize,
    /// Distinct `(worker_id, dp_rank)` pairs stored in this shard.
    pub worker_count: usize,
    /// Total cached blocks across all workers in this shard.
    pub block_count: usize,
    /// Radix-tree node count, or zero when the backend does not expose one.
    pub node_count: usize,
}
