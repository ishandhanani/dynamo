// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::ConcurrentRadixTreeCompressed;
use crate::ThreadPoolIndexer;
use crate::approx::PruneConfig;
use crate::config::{ApproximateCachePolicyKind, KvRouterConfig};
use crate::indexer::{
    ApproximateRetentionConfig, KvIndexer, KvIndexerInterface, KvIndexerMetrics, KvRouterError,
    LowerTierIndexers, MatchDetails, RoutingDecisionHashes, SyncIndexer, TieredMatchDetails,
    TieredMatchProvider, query_lower_tiers,
};
use crate::protocols::{
    KvCacheEventData, LocalBlockHash, OverlapScores, RouterEvent, WorkerId, WorkerWithDpRank,
};

/// How the device-tier primary is populated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimaryRetention {
    /// Engine KV events are the only writer.
    EventDriven,
    /// No engine events: routing decisions are written into the primary and
    /// expire after the TTL. This is the frontend's `use_kv_events=false`
    /// approximate mode.
    ApproximateTtl(Duration),
}

/// Indexer shape derived from router configuration, mirroring the frontend
/// `KvRouter` indexer resolution so a selection service built from the same
/// `KvRouterConfig` indexes the same way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexerPolicy {
    pub primary: PrimaryRetention,
    /// TTL of the predict-on-route side indexer layered over an event-driven
    /// primary (`router_predicted_ttl_secs`). `None` disables it.
    pub side_ttl: Option<Duration>,
}

impl Default for IndexerPolicy {
    fn default() -> Self {
        Self::event_driven()
    }
}

impl IndexerPolicy {
    pub fn event_driven() -> Self {
        Self {
            primary: PrimaryRetention::EventDriven,
            side_ttl: None,
        }
    }

    /// Resolve the indexer shape for `config`.
    ///
    /// Rejects the same combinations the frontend rejects. Capacity-bounded LRU
    /// retention needs per-request acquire/release fencing the service does not
    /// yet have, so `router_approximate_cache_policy=lru` falls back to TTL
    /// with a warning instead of failing.
    pub fn from_router_config(config: &KvRouterConfig) -> Result<Self> {
        if config.use_kv_events
            && config.router_approximate_cache_policy == ApproximateCachePolicyKind::Lru
        {
            anyhow::bail!(
                "router_approximate_cache_policy=lru requires use_kv_events=false; the local side indexer is TTL-only"
            );
        }
        if config.router_predicted_ttl_secs.is_some() && !config.use_kv_events {
            anyhow::bail!(
                "router_predicted_ttl_secs requires use_kv_events=true; \
                 do not combine a primary approximate indexer with a side approximate indexer"
            );
        }
        if config.use_kv_events {
            return Ok(Self {
                primary: PrimaryRetention::EventDriven,
                side_ttl: config
                    .router_predicted_ttl_secs
                    .map(Duration::from_secs_f64),
            });
        }
        if config.router_approximate_cache_policy == ApproximateCachePolicyKind::Lru {
            tracing::warn!(
                "router_approximate_cache_policy=lru is not supported by the selection service yet; falling back to TTL retention"
            );
        }
        Ok(Self {
            primary: PrimaryRetention::ApproximateTtl(Duration::from_secs_f64(
                config.router_ttl_secs,
            )),
            side_ttl: None,
        })
    }
}

/// Predict-on-route side indexer: a short-TTL approximate index that routing
/// decisions populate so a worker gets cache credit for a prompt before the
/// engine's first KV event for it arrives. Always local, never dumped or
/// replayed, and never used to seed lower-tier lookups.
#[derive(Clone)]
pub enum SideIndexer {
    Single(KvIndexer),
    Concurrent(Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>),
}

impl SideIndexer {
    pub fn new(
        ttl: Duration,
        block_size: u32,
        num_threads: usize,
        metrics: Arc<KvIndexerMetrics>,
    ) -> Self {
        let prune_config = Some(PruneConfig { ttl });
        if num_threads > 1 {
            return Self::Concurrent(Arc::new(ThreadPoolIndexer::new_with_metrics_and_pruning(
                ConcurrentRadixTreeCompressed::new(),
                num_threads,
                block_size,
                Some(metrics),
                prune_config,
            )));
        }
        Self::Single(KvIndexer::new_with_pruning(
            CancellationToken::new(),
            block_size,
            metrics,
            prune_config,
        ))
    }

    async fn find_matches(
        &self,
        sequence: &[LocalBlockHash],
    ) -> std::result::Result<OverlapScores, KvRouterError> {
        match self {
            Self::Single(indexer) => indexer.find_matches(sequence.to_vec()).await,
            Self::Concurrent(indexer) => Ok(indexer.backend().find_matches(sequence, false)),
        }
    }

    async fn record_routing_decision(
        &self,
        worker: WorkerWithDpRank,
        hashes: RoutingDecisionHashes,
    ) -> std::result::Result<(), KvRouterError> {
        match self {
            Self::Single(indexer) => {
                indexer
                    .process_routing_decision_with_hashes(
                        worker,
                        hashes.local_hashes,
                        hashes.sequence_hashes,
                    )
                    .await
            }
            Self::Concurrent(indexer) => {
                indexer
                    .process_routing_decision_hash_slices(
                        worker,
                        &hashes.local_hashes,
                        &hashes.sequence_hashes,
                    )
                    .await
            }
        }
    }

    async fn remove_worker(&self, worker_id: WorkerId) {
        match self {
            Self::Single(indexer) => indexer.remove_worker(worker_id).await,
            Self::Concurrent(indexer) => indexer.remove_worker(worker_id).await,
        }
    }

    async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: u32) {
        match self {
            Self::Single(indexer) => indexer.remove_worker_dp_rank(worker_id, dp_rank).await,
            Self::Concurrent(indexer) => indexer.remove_worker_dp_rank(worker_id, dp_rank).await,
        }
    }
}

/// Merge side-indexer scores into the primary's device match details by
/// per-worker max. Side-only workers gain a score with no paired
/// `last_matched_hashes` entry, so the result must not seed lower-tier
/// continuations; callers run `query_lower_tiers` on the primary-only details
/// first.
fn merge_side_scores(mut primary: MatchDetails, side: OverlapScores) -> MatchDetails {
    for (worker, side_score) in side.scores {
        primary
            .overlap_scores
            .scores
            .entry(worker)
            .and_modify(|score| {
                if side_score > *score {
                    *score = side_score;
                }
            })
            .or_insert(side_score);
    }
    primary
}

async fn merge_side_or_warn(
    side: Option<&SideIndexer>,
    primary: MatchDetails,
    sequence: &[LocalBlockHash],
) -> MatchDetails {
    let Some(side) = side else {
        return primary;
    };
    match side.find_matches(sequence).await {
        Ok(side_scores) => merge_side_scores(primary, side_scores),
        Err(error) => {
            tracing::warn!(
                %error,
                "predict-on-route side indexer query failed; using primary only"
            );
            primary
        }
    }
}

/// Block-content indexer wrapping a primary device-tier backend plus a
/// per-tier registry of lower-tier indexers (host-pinned, disk, …).
///
/// The service owns a `lower_tier: LowerTierIndexers` so tier-tagged events
/// get routed alongside device events and tier-aware queries can return
/// per-tier hit counts.
///
/// `approx` is the optional predict-on-route side indexer, queried alongside
/// the primary and merged by per-worker max. `primary_records_routing_decisions`
/// marks an approximate primary (no engine events) that routing decisions
/// populate directly. At most one of the two is active.
#[derive(Clone)]
pub enum Indexer {
    Single {
        primary: KvIndexer,
        lower_tier: LowerTierIndexers,
        approx: Option<SideIndexer>,
        primary_records_routing_decisions: bool,
    },
    Concurrent {
        primary: Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>,
        lower_tier: LowerTierIndexers,
        approx: Option<SideIndexer>,
        primary_records_routing_decisions: bool,
    },
}

impl Indexer {
    fn approx(&self) -> Option<&SideIndexer> {
        match self {
            Self::Single { approx, .. } | Self::Concurrent { approx, .. } => approx.as_ref(),
        }
    }

    /// Whether booked routing decisions should be written into this indexer.
    pub fn records_routing_decisions(&self) -> bool {
        match self {
            Self::Single {
                approx,
                primary_records_routing_decisions,
                ..
            }
            | Self::Concurrent {
                approx,
                primary_records_routing_decisions,
                ..
            } => approx.is_some() || *primary_records_routing_decisions,
        }
    }

    /// Router-hint chain retention needs a primary whose hashes come only from
    /// engine events.
    pub fn supports_router_hint_chain_retention(&self) -> bool {
        !self.records_routing_decisions()
    }

    /// Record a booked routing decision. Writes to the side indexer when one
    /// is attached, else to an approximate primary, else is a no-op.
    pub async fn record_routing_decision(
        &self,
        worker: WorkerWithDpRank,
        hashes: RoutingDecisionHashes,
    ) -> std::result::Result<(), KvRouterError> {
        if let Some(side) = self.approx() {
            return side.record_routing_decision(worker, hashes).await;
        }
        match self {
            Self::Single {
                primary,
                primary_records_routing_decisions: true,
                ..
            } => {
                primary
                    .process_routing_decision_with_hashes(
                        worker,
                        hashes.local_hashes,
                        hashes.sequence_hashes,
                    )
                    .await
            }
            Self::Concurrent {
                primary,
                primary_records_routing_decisions: true,
                ..
            } => {
                primary
                    .process_routing_decision_hash_slices(
                        worker,
                        &hashes.local_hashes,
                        &hashes.sequence_hashes,
                    )
                    .await
            }
            Self::Single { .. } | Self::Concurrent { .. } => Ok(()),
        }
    }

    /// Apply an event without tier dispatch — kept for callers that have
    /// already determined this is a device-tier event. Most callers should
    /// use [`Self::apply_event_routed`].
    pub async fn apply_event(&self, event: RouterEvent) {
        match self {
            Indexer::Single { primary, .. } => primary.apply_event(event).await,
            Indexer::Concurrent { primary, .. } => primary.apply_event(event).await,
        }
    }

    /// Apply an event, routing to the device-tier primary when
    /// `event.storage_tier.is_gpu()` and to the appropriate lower-tier
    /// indexer otherwise. `Cleared` events fan out to their applicable physical
    /// indexes according to the event's reset scope.
    pub async fn apply_event_routed(&self, event: RouterEvent) -> Result<(), KvRouterError> {
        let targets_primary = match event.targets_primary() {
            Ok(targets_primary) => targets_primary,
            Err(_) => {
                match self {
                    Self::Single { lower_tier, .. } | Self::Concurrent { lower_tier, .. } => {
                        lower_tier.record_unsupported_residency_event(&event);
                    }
                }
                return Ok(());
            }
        };
        let is_clear = matches!(&event.event.data, KvCacheEventData::Cleared);
        match self {
            Indexer::Single {
                primary,
                lower_tier,
                ..
            } => {
                if is_clear {
                    let mut reset_error = None;
                    if targets_primary
                        && let Err(error) = primary.apply_event_and_wait(event.clone()).await
                    {
                        tracing::warn!(%error, "Failed to reset primary residency");
                        reset_error = Some(error);
                    }
                    for indexer in lower_tier.all() {
                        if let Err(error) = indexer.apply_event_and_wait(event.clone()).await {
                            tracing::warn!(%error, "Failed to reset lower-tier residency");
                            reset_error.get_or_insert(error);
                        }
                    }
                    if let Some(error) = reset_error {
                        return Err(error);
                    }
                } else if targets_primary {
                    primary.apply_event(event).await;
                } else {
                    lower_tier
                        .get_or_create(event.storage_tier)
                        .apply_event(event)
                        .await;
                }
            }
            Indexer::Concurrent {
                primary,
                lower_tier,
                ..
            } => {
                if is_clear {
                    let mut reset_error = None;
                    if targets_primary
                        && let Err(error) = primary.apply_event_and_wait(event.clone()).await
                    {
                        tracing::warn!(%error, "Failed to reset primary residency");
                        reset_error = Some(error);
                    }
                    for indexer in lower_tier.all() {
                        if let Err(error) = indexer.apply_event_and_wait(event.clone()).await {
                            tracing::warn!(%error, "Failed to reset lower-tier residency");
                            reset_error.get_or_insert(error);
                        }
                    }
                    if let Some(error) = reset_error {
                        return Err(error);
                    }
                } else if targets_primary {
                    primary.apply_event(event).await;
                } else {
                    lower_tier
                        .get_or_create(event.storage_tier)
                        .apply_event(event)
                        .await;
                }
            }
        }
        Ok(())
    }

    pub async fn remove_worker(&self, worker_id: WorkerId) {
        if let Some(side) = self.approx() {
            side.remove_worker(worker_id).await;
        }
        match self {
            Indexer::Single {
                primary,
                lower_tier,
                ..
            } => {
                for indexer in lower_tier.all() {
                    indexer.remove_worker(worker_id).await;
                }
                primary.remove_worker(worker_id).await;
            }
            Indexer::Concurrent {
                primary,
                lower_tier,
                ..
            } => {
                for indexer in lower_tier.all() {
                    indexer.remove_worker(worker_id).await;
                }
                primary.remove_worker(worker_id).await;
            }
        }
    }

    pub async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: u32) {
        if let Some(side) = self.approx() {
            side.remove_worker_dp_rank(worker_id, dp_rank).await;
        }
        match self {
            Indexer::Single {
                primary,
                lower_tier,
                ..
            } => {
                for indexer in lower_tier.all() {
                    indexer.remove_worker_dp_rank(worker_id, dp_rank).await;
                }
                primary.remove_worker_dp_rank(worker_id, dp_rank).await;
            }
            Indexer::Concurrent {
                primary,
                lower_tier,
                ..
            } => {
                for indexer in lower_tier.all() {
                    indexer.remove_worker_dp_rank(worker_id, dp_rank).await;
                }
                primary.remove_worker_dp_rank(worker_id, dp_rank).await;
            }
        }
    }

    /// Device-tier overlap scores, including side-indexer credit. Existing
    /// flat-shape callers continue to use this; tier-aware callers should use
    /// [`Self::find_tiered_matches`].
    pub async fn find_matches(&self, hashes: Vec<LocalBlockHash>) -> Result<OverlapScores> {
        let primary = match self {
            Indexer::Single { primary, .. } => primary.find_matches(hashes.clone()).await?,
            Indexer::Concurrent { primary, .. } => primary.find_matches(hashes.clone()).await?,
        };
        let Some(side) = self.approx() else {
            return Ok(primary);
        };
        let mut merged = MatchDetails::new();
        merged.overlap_scores = primary;
        Ok(merge_side_or_warn(Some(side), merged, &hashes)
            .await
            .overlap_scores)
    }

    /// Device match details + per-tier hits, suitable for building the
    /// Mooncake-RFC-shape per-instance breakdown.
    ///
    /// Lower-tier continuations are seeded from confirmed primary matches only;
    /// side-indexer scores are merged into the device tier afterwards.
    pub async fn find_tiered_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> std::result::Result<TieredMatchDetails, KvRouterError> {
        let (device, lower_tier) = match self {
            Indexer::Single {
                primary,
                lower_tier,
                ..
            } => {
                let device = primary.find_match_details(sequence.clone()).await?;
                let lt = query_lower_tiers(lower_tier, &sequence, &device);
                (device, lt)
            }
            Indexer::Concurrent {
                primary,
                lower_tier,
                ..
            } => {
                let device: MatchDetails =
                    primary.backend().find_match_details_impl(&sequence, false);
                let lt = query_lower_tiers(lower_tier, &sequence, &device);
                (device, lt)
            }
        };
        let device = merge_side_or_warn(self.approx(), device, &sequence).await;
        Ok(TieredMatchDetails { device, lower_tier })
    }

    /// Dump every event tracked by this indexer in a form replayable by a
    /// peer's [`Self::apply_event_routed`].
    ///
    /// Returns the primary device-tier dump first, followed by every allocated
    /// lower-tier indexer's dump with each event's `storage_tier` retagged to
    /// the tier it lives in. The `LowerTierIndexer::dump_events` synthetic
    /// events default `storage_tier` to `Device`, so without retagging a peer
    /// would replay them into the wrong slot.
    pub async fn dump_events(&self) -> Result<Vec<RouterEvent>> {
        let (primary_events, lower_tier_entries) = match self {
            Indexer::Single {
                primary,
                lower_tier,
                ..
            } => (
                primary.dump_events().await.map_err(anyhow::Error::from)?,
                lower_tier.entries(),
            ),
            Indexer::Concurrent {
                primary,
                lower_tier,
                ..
            } => (
                primary.dump_events().await.map_err(anyhow::Error::from)?,
                lower_tier.entries(),
            ),
        };

        let mut out = primary_events;
        for (tier, indexer) in lower_tier_entries {
            let events = indexer.dump_events().await.map_err(anyhow::Error::from)?;
            for mut event in events {
                event.storage_tier = tier;
                out.push(event);
            }
        }
        Ok(out)
    }
}

#[async_trait]
impl TieredMatchProvider for Indexer {
    async fn find_tiered_matches(
        &self,
        sequence: &[LocalBlockHash],
    ) -> std::result::Result<TieredMatchDetails, KvRouterError> {
        self.find_tiered_matches(sequence.to_vec()).await
    }
}

#[cfg(test)]
pub(crate) mod test_util {
    use crate::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, RouterEvent, StorageTier,
        compute_seq_hash_for_block,
    };

    /// Construct a STORE [`RouterEvent`] for `local_hashes` that chain off
    /// `prefix_hashes` (the parent path). The event is tagged `storage_tier`.
    /// Mirrors the helper used in the request-plane tests so behavior across
    /// the two indexer surfaces stays comparable.
    pub fn store_event(
        worker_id: u64,
        dp_rank: u32,
        event_id: u64,
        prefix_hashes: &[u64],
        local_hashes: &[u64],
        storage_tier: StorageTier,
    ) -> RouterEvent {
        let prefix_block_hashes: Vec<LocalBlockHash> =
            prefix_hashes.iter().copied().map(LocalBlockHash).collect();
        let parent_hash = compute_seq_hash_for_block(&prefix_block_hashes)
            .last()
            .copied()
            .map(ExternalSequenceBlockHash);

        let full_hashes: Vec<LocalBlockHash> = prefix_hashes
            .iter()
            .chain(local_hashes.iter())
            .copied()
            .map(LocalBlockHash)
            .collect();
        let full_sequence_hashes = compute_seq_hash_for_block(&full_hashes);
        let new_sequence_hashes = &full_sequence_hashes[prefix_hashes.len()..];
        let blocks = local_hashes
            .iter()
            .zip(new_sequence_hashes.iter())
            .map(|(&local_hash, &sequence_hash)| KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(sequence_hash),
                tokens_hash: LocalBlockHash(local_hash),
                mm_extra_info: None,
            })
            .collect();

        RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash,
                    start_position: None,
                    blocks,
                }),
                dp_rank,
            },
            storage_tier,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::test_util::store_event;
    use super::*;
    use crate::indexer::KvIndexerInterface;
    #[cfg(feature = "metrics")]
    use crate::indexer::{METRIC_EVENT_CLEARED, METRIC_STATUS_OK};
    use crate::protocols::{
        KvCacheEvent, LocalBlockHash, ResidencyDomain, StorageTier, WorkerWithDpRank,
    };

    /// Apply a Device store and a HostPinned store anchored on it. The tiered
    /// query must surface both tier hits, and the device-tier `find_matches`
    /// must still see the device store (i.e. dispatch routed it to the
    /// primary, not to the lower-tier slot).
    #[tokio::test]
    async fn apply_event_routed_dispatches_by_tier() {
        let indexer = create_indexer(4, 1);
        let worker = WorkerWithDpRank::new(7, 0);

        indexer
            .apply_event_routed(store_event(
                worker.worker_id,
                worker.dp_rank,
                1,
                &[],
                &[11, 12],
                StorageTier::Device,
            ))
            .await
            .unwrap();

        indexer
            .apply_event_routed(store_event(
                worker.worker_id,
                worker.dp_rank,
                2,
                &[11, 12],
                &[13],
                StorageTier::HostPinned,
            ))
            .await
            .unwrap();

        // Flush primary so the in-flight events are observable by the query.
        if let Indexer::Single { primary, .. } = &indexer {
            let _ = primary.flush().await;
        }
        if let Indexer::Single { lower_tier, .. } = &indexer {
            for inner in lower_tier.all() {
                let _ = inner.dump_events().await.unwrap();
            }
        }

        let sequence = vec![LocalBlockHash(11), LocalBlockHash(12), LocalBlockHash(13)];
        let tiered = indexer.find_tiered_matches(sequence).await.unwrap();

        assert_eq!(
            tiered.device.overlap_scores.scores.get(&worker).copied(),
            Some(2),
            "device should match 2 blocks for the worker"
        );
        let host_hits = tiered
            .lower_tier
            .get(&StorageTier::HostPinned)
            .expect("host-pinned tier should have been allocated");
        assert_eq!(
            host_hits.hits.get(&worker).copied(),
            Some(1),
            "host-pinned should report 1 additional matched block beyond device"
        );
    }

    /// Dump every tier's events from a populated indexer, replay through a
    /// fresh indexer with `apply_event_routed`, and assert the tiered query
    /// result is identical. This is the peer-recovery round trip: it would
    /// fail if `dump_events` skipped lower-tier events or omitted their tier
    /// tags (peer would replay HostPinned events into the device primary).
    #[tokio::test]
    async fn dump_events_round_trips_through_apply_event_routed() {
        let block_size = 4;
        let worker = WorkerWithDpRank::new(7, 0);
        let source = create_indexer(block_size, 1);

        source
            .apply_event_routed(store_event(
                worker.worker_id,
                worker.dp_rank,
                1,
                &[],
                &[11, 12],
                StorageTier::Device,
            ))
            .await
            .unwrap();
        source
            .apply_event_routed(store_event(
                worker.worker_id,
                worker.dp_rank,
                2,
                &[11, 12],
                &[13],
                StorageTier::HostPinned,
            ))
            .await
            .unwrap();
        source
            .apply_event_routed(store_event(
                worker.worker_id,
                worker.dp_rank,
                3,
                &[11, 12, 13],
                &[14],
                StorageTier::Disk,
            ))
            .await
            .unwrap();

        if let Indexer::Single {
            primary,
            lower_tier,
            ..
        } = &source
        {
            let _ = primary.flush().await;
            for inner in lower_tier.all() {
                let _ = inner.dump_events().await.unwrap();
            }
        }

        let dump = source.dump_events().await.unwrap();

        // Sanity: dump must surface events from every tier we fed in.
        assert!(
            dump.iter()
                .any(|e| matches!(e.storage_tier, StorageTier::Device)),
            "dump must contain Device events"
        );
        assert!(
            dump.iter()
                .any(|e| matches!(e.storage_tier, StorageTier::HostPinned)),
            "dump must contain HostPinned events with tier retagged"
        );
        assert!(
            dump.iter()
                .any(|e| matches!(e.storage_tier, StorageTier::Disk)),
            "dump must contain Disk events with tier retagged"
        );

        let replayed = create_indexer(block_size, 1);
        for event in dump {
            replayed.apply_event_routed(event).await.unwrap();
        }
        if let Indexer::Single {
            primary,
            lower_tier,
            ..
        } = &replayed
        {
            let _ = primary.flush().await;
            for inner in lower_tier.all() {
                let _ = inner.dump_events().await.unwrap();
            }
        }

        let sequence = vec![
            LocalBlockHash(11),
            LocalBlockHash(12),
            LocalBlockHash(13),
            LocalBlockHash(14),
        ];
        let tiered = replayed.find_tiered_matches(sequence).await.unwrap();

        assert_eq!(
            tiered.device.overlap_scores.scores.get(&worker).copied(),
            Some(2),
            "device should match 2 blocks after replay"
        );
        assert_eq!(
            tiered
                .lower_tier
                .get(&StorageTier::HostPinned)
                .and_then(|d| d.hits.get(&worker).copied()),
            Some(1),
            "host-pinned should report 1 additional block after replay"
        );
        assert_eq!(
            tiered
                .lower_tier
                .get(&StorageTier::Disk)
                .and_then(|d| d.hits.get(&worker).copied()),
            Some(1),
            "disk should report 1 additional block after replay"
        );
    }

    #[tokio::test]
    async fn clear_reports_partial_failure_after_attempting_every_tier() {
        let indexer = create_indexer(4, 1);
        let worker = WorkerWithDpRank::new(7, 0);
        let (failed_tier, healthy_tier) = match &indexer {
            Indexer::Single { lower_tier, .. } => (
                lower_tier.get_or_create(StorageTier::HostPinned),
                lower_tier.get_or_create(StorageTier::Disk),
            ),
            Indexer::Concurrent { .. } => unreachable!("test creates the single indexer"),
        };

        indexer
            .apply_event_routed(store_event(
                worker.worker_id,
                worker.dp_rank,
                1,
                &[],
                &[11],
                StorageTier::Disk,
            ))
            .await
            .unwrap();
        let _ = healthy_tier.dump_events().await.unwrap();
        failed_tier.shutdown();

        let error = indexer
            .apply_event_routed(RouterEvent::with_residency_domain(
                worker.worker_id,
                KvCacheEvent {
                    event_id: 2,
                    data: KvCacheEventData::Cleared,
                    dp_rank: worker.dp_rank,
                },
                StorageTier::Device,
                ResidencyDomain::Worker,
            ))
            .await
            .unwrap_err();

        assert!(matches!(error, KvRouterError::IndexerOffline));
        assert!(healthy_tier.dump_events().await.unwrap().is_empty());
    }

    fn policy_indexer(num_threads: usize, policy: IndexerPolicy) -> Indexer {
        create_indexer_with_policy(
            4,
            num_threads,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            &policy,
        )
    }

    /// Wait until every locally enqueued event and routing decision is applied.
    async fn flush(indexer: &Indexer) {
        match indexer {
            Indexer::Single {
                primary, approx, ..
            } => {
                let _ = primary.flush().await;
                match approx {
                    Some(SideIndexer::Single(side)) => {
                        let _ = side.flush().await;
                    }
                    Some(SideIndexer::Concurrent(side)) => {
                        let _ = side.flush().await;
                    }
                    None => {}
                }
            }
            Indexer::Concurrent {
                primary, approx, ..
            } => {
                let _ = primary.flush().await;
                match approx {
                    Some(SideIndexer::Single(side)) => {
                        let _ = side.flush().await;
                    }
                    Some(SideIndexer::Concurrent(side)) => {
                        let _ = side.flush().await;
                    }
                    None => {}
                }
            }
        }
    }

    #[tokio::test]
    async fn event_driven_policy_ignores_routing_decisions() {
        let indexer = policy_indexer(1, IndexerPolicy::event_driven());
        assert!(!indexer.records_routing_decisions());
        assert!(indexer.supports_router_hint_chain_retention());
        let worker = WorkerWithDpRank::new(7, 0);
        indexer
            .record_routing_decision(
                worker,
                RoutingDecisionHashes::from_local_hashes(vec![LocalBlockHash(11)]),
            )
            .await
            .unwrap();
        flush(&indexer).await;
        let tiered = indexer
            .find_tiered_matches(vec![LocalBlockHash(11)])
            .await
            .unwrap();
        assert!(tiered.device.overlap_scores.scores.is_empty());
    }

    #[tokio::test]
    async fn approximate_primary_records_routing_decisions() {
        for num_threads in [1, 2] {
            let indexer = policy_indexer(
                num_threads,
                IndexerPolicy {
                    primary: PrimaryRetention::ApproximateTtl(Duration::from_secs(60)),
                    side_ttl: None,
                },
            );
            assert!(indexer.records_routing_decisions());
            assert!(!indexer.supports_router_hint_chain_retention());
            let worker = WorkerWithDpRank::new(7, 0);
            indexer
                .record_routing_decision(
                    worker,
                    RoutingDecisionHashes::from_local_hashes(vec![
                        LocalBlockHash(11),
                        LocalBlockHash(12),
                    ]),
                )
                .await
                .unwrap();
            flush(&indexer).await;

            let sequence = vec![LocalBlockHash(11), LocalBlockHash(12), LocalBlockHash(13)];
            let tiered = indexer.find_tiered_matches(sequence.clone()).await.unwrap();
            assert_eq!(
                tiered.device.overlap_scores.scores.get(&worker).copied(),
                Some(2),
                "{num_threads} thread(s): approximate primary should credit the routed prefix"
            );
            let flat = indexer.find_matches(sequence).await.unwrap();
            assert_eq!(flat.scores.get(&worker).copied(), Some(2));

            // The recorded prefix is dumped like any other primary state, so a
            // recovering peer inherits it.
            assert!(!indexer.dump_events().await.unwrap().is_empty());
        }
    }

    #[tokio::test]
    async fn side_indexer_merges_predicted_scores_by_worker_max() {
        for num_threads in [1, 2] {
            let indexer = policy_indexer(
                num_threads,
                IndexerPolicy {
                    primary: PrimaryRetention::EventDriven,
                    side_ttl: Some(Duration::from_secs(60)),
                },
            );
            assert!(indexer.records_routing_decisions());
            let confirmed = WorkerWithDpRank::new(7, 0);
            let predicted = WorkerWithDpRank::new(8, 0);

            // Engine event: worker 7 confirmed holds [11, 12].
            indexer
                .apply_event_routed(store_event(
                    confirmed.worker_id,
                    confirmed.dp_rank,
                    1,
                    &[],
                    &[11, 12],
                    StorageTier::Device,
                ))
                .await
                .unwrap();
            // Routing decisions: worker 8 was just routed [11, 12, 13]; worker 7
            // was routed [11] (shorter than what the engine confirmed).
            indexer
                .record_routing_decision(
                    predicted,
                    RoutingDecisionHashes::from_local_hashes(vec![
                        LocalBlockHash(11),
                        LocalBlockHash(12),
                        LocalBlockHash(13),
                    ]),
                )
                .await
                .unwrap();
            indexer
                .record_routing_decision(
                    confirmed,
                    RoutingDecisionHashes::from_local_hashes(vec![LocalBlockHash(11)]),
                )
                .await
                .unwrap();
            flush(&indexer).await;

            let sequence = vec![LocalBlockHash(11), LocalBlockHash(12), LocalBlockHash(13)];
            let tiered = indexer.find_tiered_matches(sequence.clone()).await.unwrap();
            let scores = &tiered.device.overlap_scores.scores;
            assert_eq!(
                scores.get(&predicted).copied(),
                Some(3),
                "{num_threads} thread(s)"
            );
            assert_eq!(
                scores.get(&confirmed).copied(),
                Some(2),
                "{num_threads} thread(s): primary wins when it saw the longer prefix"
            );
            // Side scores never seed lower tiers and are never dumped.
            assert!(tiered.lower_tier.values().all(|tier| tier.hits.is_empty()));
            let dumped = indexer.dump_events().await.unwrap();
            assert!(
                dumped
                    .iter()
                    .all(|event| event.worker_id == confirmed.worker_id)
            );

            indexer.remove_worker(predicted.worker_id).await;
            flush(&indexer).await;
            let tiered = indexer.find_tiered_matches(sequence).await.unwrap();
            assert!(!tiered.device.overlap_scores.scores.contains_key(&predicted));
        }
    }

    #[test]
    fn indexer_policy_mirrors_frontend_resolution() {
        let event_driven = KvRouterConfig::default();
        assert_eq!(
            IndexerPolicy::from_router_config(&event_driven).unwrap(),
            IndexerPolicy::event_driven()
        );

        let side = KvRouterConfig {
            router_predicted_ttl_secs: Some(2.5),
            ..Default::default()
        };
        assert_eq!(
            IndexerPolicy::from_router_config(&side).unwrap().side_ttl,
            Some(Duration::from_secs_f64(2.5))
        );

        let approximate = KvRouterConfig {
            use_kv_events: false,
            router_ttl_secs: 30.0,
            ..Default::default()
        };
        assert_eq!(
            IndexerPolicy::from_router_config(&approximate).unwrap(),
            IndexerPolicy {
                primary: PrimaryRetention::ApproximateTtl(Duration::from_secs(30)),
                side_ttl: None,
            }
        );

        let lru_without_events = KvRouterConfig {
            use_kv_events: false,
            router_approximate_cache_policy: ApproximateCachePolicyKind::Lru,
            ..Default::default()
        };
        assert!(matches!(
            IndexerPolicy::from_router_config(&lru_without_events)
                .unwrap()
                .primary,
            PrimaryRetention::ApproximateTtl(_)
        ));

        let lru_with_events = KvRouterConfig {
            router_approximate_cache_policy: ApproximateCachePolicyKind::Lru,
            ..Default::default()
        };
        assert!(IndexerPolicy::from_router_config(&lru_with_events).is_err());

        let side_without_events = KvRouterConfig {
            use_kv_events: false,
            router_predicted_ttl_secs: Some(1.0),
            ..Default::default()
        };
        assert!(IndexerPolicy::from_router_config(&side_without_events).is_err());
    }

    #[cfg(feature = "metrics")]
    #[tokio::test]
    async fn acknowledged_clear_records_primary_event_for_both_indexers() {
        for num_threads in [1, 2] {
            let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
            let indexer = create_indexer_with_metrics(4, num_threads, metrics.clone());
            indexer
                .apply_event_routed(RouterEvent::with_residency_domain(
                    7,
                    KvCacheEvent {
                        event_id: 1,
                        data: KvCacheEventData::Cleared,
                        dp_rank: 0,
                    },
                    StorageTier::Device,
                    ResidencyDomain::Worker,
                ))
                .await
                .unwrap();

            assert_eq!(
                metrics
                    .kv_cache_events_applied
                    .get_metric_with_label_values(&[METRIC_EVENT_CLEARED, METRIC_STATUS_OK])
                    .unwrap()
                    .get(),
                1,
                "clear metric mismatch for {num_threads} indexer thread(s)"
            );
        }
    }
}

pub fn create_indexer(block_size: u32, num_threads: usize) -> Indexer {
    create_indexer_with_metrics(
        block_size,
        num_threads,
        Arc::new(KvIndexerMetrics::new_unregistered()),
    )
}

pub fn create_indexer_with_metrics(
    block_size: u32,
    num_threads: usize,
    metrics: Arc<KvIndexerMetrics>,
) -> Indexer {
    create_indexer_with_policy(
        block_size,
        num_threads,
        metrics,
        &IndexerPolicy::event_driven(),
    )
}

pub fn create_indexer_with_policy(
    block_size: u32,
    num_threads: usize,
    metrics: Arc<KvIndexerMetrics>,
    policy: &IndexerPolicy,
) -> Indexer {
    let retention = match policy.primary {
        PrimaryRetention::EventDriven => None,
        PrimaryRetention::ApproximateTtl(ttl) => {
            Some(ApproximateRetentionConfig::Ttl(PruneConfig { ttl }))
        }
    };
    let primary_records_routing_decisions = retention.is_some();
    let approx = policy
        .side_ttl
        .map(|ttl| SideIndexer::new(ttl, block_size, num_threads, Arc::clone(&metrics)));
    if num_threads > 1 {
        Indexer::Concurrent {
            primary: Arc::new(
                ThreadPoolIndexer::new_with_metrics_and_approximate_retention(
                    ConcurrentRadixTreeCompressed::new(),
                    num_threads,
                    block_size,
                    Some(metrics),
                    retention,
                ),
            ),
            lower_tier: LowerTierIndexers::new(num_threads, block_size),
            approx,
            primary_records_routing_decisions,
        }
    } else {
        Indexer::Single {
            primary: KvIndexer::new_with_approximate_retention(
                CancellationToken::new(),
                block_size,
                metrics,
                retention,
            ),
            lower_tier: LowerTierIndexers::new(1, block_size),
            approx,
            primary_records_routing_decisions,
        }
    }
}
