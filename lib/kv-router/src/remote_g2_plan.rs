// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::indexer::TieredMatchDetails;
use crate::protocols::{
    DpRank, LocalBlockHash, StorageTier, WorkerConfigLike, WorkerId, WorkerWithDpRank,
};

pub const REMOTE_KV_REUSE_PLAN_EXTRA_ARGS_KEY: &str = "remote_kv_reuse_plan";
pub const REMOTE_KV_REUSE_NO_PLAN_REASON_EXTRA_ARGS_KEY: &str = "remote_kv_reuse_no_plan_reason";
pub const REMOTE_KV_REUSE_PLAN_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteKvReusePlan {
    pub plan_id: String,
    pub request_id: String,
    pub target_worker_id: WorkerId,
    pub target_dp_rank: DpRank,
    pub source_worker_id: WorkerId,
    pub source_dp_rank: DpRank,
    /// Source route copied from the selected source worker's registered runtime
    /// metadata after target selection.
    pub source_host: String,
    /// Bootstrap port paired with `source_host` by the source worker's runtime
    /// metadata. The candidate query never receives arbitrary endpoints.
    pub source_bootstrap_port: u16,
    pub source_tier: StorageTier,
    /// Router-computed token block hashes for the contiguous planned request
    /// interval. These identify the route and request alignment, and they may
    /// differ in value from `engine_block_hashes` because the engine can use a
    /// different sequence hash.
    pub router_block_hashes: Vec<LocalBlockHash>,
    /// Position in the request's prefix where `router_block_hashes[0]` lives.
    /// Equals the source worker's device-tier match count at plan time.
    /// The target's connector uses this to verify alignment with its own
    /// `num_computed_tokens` before attaching descriptors.
    pub start_block_index: u32,
    pub planned_prefix_blocks: u32,
    pub block_size_tokens: u32,
    pub created_at_ms: u64,
    pub expires_at_ms: u64,
    pub plan_version: u32,
    /// Parallel to `router_block_hashes` once a plan is attached to the request,
    /// carrying each block's framework KV-event hash. The source framework uses
    /// these values to look up actual HostPinned blocks; `router_block_hashes`
    /// remains the router-visible plan identity.
    pub engine_block_hashes: Vec<u64>,
}

// Compatibility identity is intentionally deferred in v1; source resolve remains authoritative.

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RemoteKvReuseNoPlanReason {
    Disabled,
    NoRemoteG2Candidate,
    NoContiguousPrefix,
    BelowMinPlannedBlocks,
    BelowScoreTax,
    SourceIsTarget,
    IncompatibleBlockSize,
    PlanExpired,
    NoSourceBootstrapEndpoint,
    SerializationFailed,
}

impl RemoteKvReuseNoPlanReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::NoRemoteG2Candidate => "no_remote_g2_candidate",
            Self::NoContiguousPrefix => "no_contiguous_prefix",
            Self::BelowMinPlannedBlocks => "below_min_planned_blocks",
            Self::BelowScoreTax => "below_score_tax",
            Self::SourceIsTarget => "source_is_target",
            Self::IncompatibleBlockSize => "incompatible_block_size",
            Self::PlanExpired => "plan_expired",
            Self::NoSourceBootstrapEndpoint => "no_source_bootstrap_endpoint",
            Self::SerializationFailed => "serialization_failed",
        }
    }
}

pub struct RemoteKvReuseSelectionInput<'a> {
    pub request_id: &'a str,
    pub target: WorkerWithDpRank,
    pub block_hashes: &'a [LocalBlockHash],
    pub block_size_tokens: u32,
    pub tiered_matches: &'a TieredMatchDetails,
    pub created_at_ms: u64,
    pub expires_at_ms: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RemoteKvReuseSelectionStats {
    pub rejected_g1_candidates: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteKvReuseDecision {
    Plan {
        plan: RemoteKvReusePlan,
        stats: RemoteKvReuseSelectionStats,
    },
    NoPlan {
        reason: RemoteKvReuseNoPlanReason,
        stats: RemoteKvReuseSelectionStats,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteKvReuseCandidate {
    pub source: WorkerWithDpRank,
    pub start_block_index: usize,
    pub planned_prefix_blocks: usize,
    pub incremental_blocks: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteG2ScoredCandidate {
    pub candidate: RemoteKvReuseCandidate,
    /// Transfer benefit before fixed/per-block cost. This is capped by
    /// `score_cap_blocks` when configured.
    pub benefit_blocks: u32,
    /// Estimated transfer cost charged after the scheduler weights
    /// `benefit_blocks`. The query boundary keeps the cost attached to the same
    /// candidate that may later become the plan.
    pub cost_blocks: u32,
    /// Candidate benefit after cost/cap before applying the shared-cache
    /// multiplier. This remains useful for metrics and zero-score accounting.
    pub score_blocks: u32,
}

/// Direct G2 analogue of a shared-cache lookup.
///
/// The router first queries device/lower-tier availability, then this query
/// derives per-target Direct G2 candidates plus an estimated transfer cost. The
/// scheduler consumes benefit and cost maps; after target selection the router
/// reuses the selected candidate to attach the actual transfer plan.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RemoteG2CandidateQueryResult {
    pub scored_candidates: HashMap<WorkerWithDpRank, RemoteG2ScoredCandidate>,
    pub benefit_blocks_by_target: HashMap<WorkerWithDpRank, u32>,
    pub cost_blocks_by_target: HashMap<WorkerWithDpRank, u32>,
    pub score_blocks_by_target: HashMap<WorkerWithDpRank, u32>,
}

impl RemoteG2CandidateQueryResult {
    pub fn from_scored_candidates(
        scored_candidates: HashMap<WorkerWithDpRank, RemoteG2ScoredCandidate>,
    ) -> Self {
        let benefit_blocks_by_target = scored_candidates
            .iter()
            .filter_map(|(target, candidate)| {
                (candidate.benefit_blocks > 0).then_some((*target, candidate.benefit_blocks))
            })
            .collect();
        let cost_blocks_by_target = scored_candidates
            .iter()
            .filter_map(|(target, candidate)| {
                (candidate.benefit_blocks > 0).then_some((*target, candidate.cost_blocks))
            })
            .collect();
        let score_blocks_by_target = scored_candidates
            .iter()
            .filter_map(|(target, candidate)| {
                (candidate.score_blocks > 0).then_some((*target, candidate.score_blocks))
            })
            .collect();
        Self {
            scored_candidates,
            benefit_blocks_by_target,
            cost_blocks_by_target,
            score_blocks_by_target,
        }
    }

    pub fn selected_candidate(&self, target: WorkerWithDpRank) -> Option<RemoteG2ScoredCandidate> {
        self.scored_candidates.get(&target).copied()
    }

    pub fn scored_candidate_count(&self) -> usize {
        self.scored_candidates.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RemoteG2CostModel {
    FixedPlusPerBlock { fixed_blocks: u32, per_block: f64 },
}

impl RemoteG2CostModel {
    pub const fn fixed_blocks(blocks: u32) -> Self {
        Self::FixedPlusPerBlock {
            fixed_blocks: blocks,
            per_block: 0.0,
        }
    }

    pub const fn fixed_plus_per_block(fixed_blocks: u32, per_block: f64) -> Self {
        Self::FixedPlusPerBlock {
            fixed_blocks,
            per_block,
        }
    }

    fn estimate_blocks_for_candidate(
        &self,
        _target: WorkerWithDpRank,
        _request_blocks: usize,
        _tiered_matches: &TieredMatchDetails,
        candidate: &RemoteKvReuseCandidate,
    ) -> u32 {
        match self {
            Self::FixedPlusPerBlock {
                fixed_blocks,
                per_block,
            } => {
                let variable_cost = if per_block.is_finite() && *per_block > 0.0 {
                    ((*per_block * candidate.incremental_blocks as f64).ceil() as u64)
                        .min(u32::MAX as u64) as u32
                } else {
                    0
                };
                fixed_blocks.saturating_add(variable_cost)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RemoteG2CandidateQueryPolicy {
    pub cost_model: RemoteG2CostModel,
    pub score_cap_blocks: Option<u32>,
}

impl RemoteG2CandidateQueryPolicy {
    pub const fn fixed_cost(fixed_cost_blocks: u32, score_cap_blocks: Option<u32>) -> Self {
        Self {
            cost_model: RemoteG2CostModel::fixed_blocks(fixed_cost_blocks),
            score_cap_blocks,
        }
    }

    pub const fn fixed_plus_per_block_cost(
        fixed_cost_blocks: u32,
        per_block_cost: f64,
        score_cap_blocks: Option<u32>,
    ) -> Self {
        Self {
            cost_model: RemoteG2CostModel::fixed_plus_per_block(fixed_cost_blocks, per_block_cost),
            score_cap_blocks,
        }
    }

    pub const fn fixed(fixed_cost_blocks: u32, score_cap_blocks: Option<u32>) -> Self {
        Self::fixed_cost(fixed_cost_blocks, score_cap_blocks)
    }

    fn score_blocks_for_candidate(
        &self,
        target: WorkerWithDpRank,
        request_blocks: usize,
        tiered_matches: &TieredMatchDetails,
        candidate: &RemoteKvReuseCandidate,
    ) -> (u32, u32, u32) {
        let cost_blocks = self.cost_model.estimate_blocks_for_candidate(
            target,
            request_blocks,
            tiered_matches,
            candidate,
        );
        let benefit_blocks: u32 = candidate.incremental_blocks.try_into().unwrap_or(u32::MAX);
        let benefit_blocks = match self.score_cap_blocks.filter(|cap| *cap > 0) {
            Some(cap) => benefit_blocks.min(cap),
            None => benefit_blocks,
        };
        let score_blocks = benefit_blocks.saturating_sub(cost_blocks);
        (cost_blocks, benefit_blocks, score_blocks)
    }

    pub fn scored_candidate_for_target(
        &self,
        target: WorkerWithDpRank,
        request_blocks: usize,
        tiered_matches: &TieredMatchDetails,
    ) -> Option<RemoteG2ScoredCandidate> {
        let candidate =
            best_remote_g2_candidate_for_target(target, request_blocks, tiered_matches)?;
        let (cost_blocks, benefit_blocks, score_blocks) =
            self.score_blocks_for_candidate(target, request_blocks, tiered_matches, &candidate);
        Some(RemoteG2ScoredCandidate {
            candidate,
            benefit_blocks,
            cost_blocks,
            score_blocks,
        })
    }

    pub fn scored_candidates_by_target<C: WorkerConfigLike>(
        &self,
        workers: &HashMap<WorkerId, C>,
        request_blocks: usize,
        tiered_matches: &TieredMatchDetails,
    ) -> HashMap<WorkerWithDpRank, RemoteG2ScoredCandidate> {
        let mut candidates = HashMap::new();

        for (&worker_id, config) in workers {
            let dp_start = config.data_parallel_start_rank();
            let dp_end = dp_start.saturating_add(config.data_parallel_size());
            for dp_rank in dp_start..dp_end {
                let target = WorkerWithDpRank::new(worker_id, dp_rank);
                if let Some(candidate) =
                    self.scored_candidate_for_target(target, request_blocks, tiered_matches)
                {
                    candidates.insert(target, candidate);
                }
            }
        }

        candidates
    }

    pub fn query_candidates_by_target<C: WorkerConfigLike>(
        &self,
        workers: &HashMap<WorkerId, C>,
        request_blocks: usize,
        tiered_matches: &TieredMatchDetails,
    ) -> RemoteG2CandidateQueryResult {
        RemoteG2CandidateQueryResult::from_scored_candidates(self.scored_candidates_by_target(
            workers,
            request_blocks,
            tiered_matches,
        ))
    }
}

pub type RemoteG2ScoringPolicy = RemoteG2CandidateQueryPolicy;

fn device_match_blocks(tiered_matches: &TieredMatchDetails, worker: WorkerWithDpRank) -> usize {
    tiered_matches
        .device
        .overlap_scores
        .scores
        .get(&worker)
        .copied()
        .unwrap_or(0) as usize
}

fn host_pinned_hits(tiered_matches: &TieredMatchDetails, worker: WorkerWithDpRank) -> usize {
    tiered_matches
        .lower_tier
        .get(&StorageTier::HostPinned)
        .and_then(|matches| matches.hits.get(&worker).copied())
        .unwrap_or(0)
}

fn local_prefix_blocks(
    tiered_matches: &TieredMatchDetails,
    worker: WorkerWithDpRank,
    request_blocks: usize,
) -> usize {
    device_match_blocks(tiered_matches, worker)
        .saturating_add(host_pinned_hits(tiered_matches, worker))
        .min(request_blocks)
}

fn source_interval(
    tiered_matches: &TieredMatchDetails,
    worker: WorkerWithDpRank,
    request_blocks: usize,
) -> Option<(usize, usize)> {
    let host_continuation_hits = host_pinned_hits(tiered_matches, worker);
    let device_match = device_match_blocks(tiered_matches, worker);

    // Normal lower-tier semantics report HostPinned hits as a continuation
    // after the source worker's Device match. With write-through HiCache, the
    // same blocks are present in both GPU and CPU, so that continuation can be
    // zero even though a valid CPU-pinned chain exists from root.
    let (start, hits) = if host_continuation_hits > 0 {
        (device_match.min(request_blocks), host_continuation_hits)
    } else if device_match > 0 {
        (0, device_match.min(request_blocks))
    } else {
        return None;
    };

    let hits = hits.min(request_blocks.saturating_sub(start));
    (hits > 0).then_some((start, hits))
}

fn incremental_blocks_for_target(target_local_prefix: usize, start: usize, hits: usize) -> usize {
    let end = start.saturating_add(hits);
    if target_local_prefix < start || target_local_prefix >= end {
        0
    } else {
        end - target_local_prefix
    }
}

fn choose_better_candidate(
    best: &mut Option<RemoteKvReuseCandidate>,
    candidate: RemoteKvReuseCandidate,
) {
    match best {
        None => *best = Some(candidate),
        Some(best_candidate)
            if candidate.incremental_blocks > best_candidate.incremental_blocks
                || (candidate.incremental_blocks == best_candidate.incremental_blocks
                    && candidate.planned_prefix_blocks > best_candidate.planned_prefix_blocks)
                || (candidate.incremental_blocks == best_candidate.incremental_blocks
                    && candidate.planned_prefix_blocks == best_candidate.planned_prefix_blocks
                    && candidate.source < best_candidate.source) =>
        {
            *best = Some(candidate);
        }
        Some(_) => {}
    }
}

fn has_remote_g2_candidate(
    target: WorkerWithDpRank,
    _request_blocks: usize,
    tiered_matches: &TieredMatchDetails,
) -> bool {
    tiered_matches
        .lower_tier
        .get(&StorageTier::HostPinned)
        .is_some_and(|matches| matches.hits.keys().any(|&worker| worker != target))
}

pub fn best_remote_g2_candidate_for_target(
    target: WorkerWithDpRank,
    request_blocks: usize,
    tiered_matches: &TieredMatchDetails,
) -> Option<RemoteKvReuseCandidate> {
    let host_pinned_matches = tiered_matches.lower_tier.get(&StorageTier::HostPinned)?;
    let target_local_prefix = local_prefix_blocks(tiered_matches, target, request_blocks);
    let mut best = None;

    for &worker in host_pinned_matches.hits.keys() {
        if worker == target {
            continue;
        }
        let Some((start, hits)) = source_interval(tiered_matches, worker, request_blocks) else {
            continue;
        };
        let incremental_blocks = incremental_blocks_for_target(target_local_prefix, start, hits);
        if incremental_blocks == 0 {
            continue;
        }

        choose_better_candidate(
            &mut best,
            RemoteKvReuseCandidate {
                source: worker,
                start_block_index: start,
                planned_prefix_blocks: hits,
                incremental_blocks,
            },
        );
    }

    best
}

pub fn remote_g2_score_blocks_for_target(
    target: WorkerWithDpRank,
    request_blocks: usize,
    tiered_matches: &TieredMatchDetails,
    tax_blocks: u32,
    cap_blocks: Option<u32>,
) -> u32 {
    RemoteG2CandidateQueryPolicy::fixed_cost(tax_blocks, cap_blocks)
        .scored_candidate_for_target(target, request_blocks, tiered_matches)
        .map(|candidate| candidate.score_blocks)
        .unwrap_or(0)
}

pub fn remote_g2_scored_candidates_by_target<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request_blocks: usize,
    tiered_matches: &TieredMatchDetails,
    tax_blocks: u32,
    cap_blocks: Option<u32>,
) -> HashMap<WorkerWithDpRank, RemoteG2ScoredCandidate> {
    RemoteG2CandidateQueryPolicy::fixed_cost(tax_blocks, cap_blocks).scored_candidates_by_target(
        workers,
        request_blocks,
        tiered_matches,
    )
}

pub fn remote_g2_scored_candidate_for_target(
    target: WorkerWithDpRank,
    request_blocks: usize,
    tiered_matches: &TieredMatchDetails,
    tax_blocks: u32,
    cap_blocks: Option<u32>,
) -> Option<RemoteG2ScoredCandidate> {
    RemoteG2CandidateQueryPolicy::fixed_cost(tax_blocks, cap_blocks).scored_candidate_for_target(
        target,
        request_blocks,
        tiered_matches,
    )
}

pub fn remote_g2_candidate_query_by_target<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request_blocks: usize,
    tiered_matches: &TieredMatchDetails,
    tax_blocks: u32,
    cap_blocks: Option<u32>,
) -> RemoteG2CandidateQueryResult {
    remote_g2_candidate_query_by_target_with_cost(
        workers,
        request_blocks,
        tiered_matches,
        tax_blocks,
        cap_blocks,
        0.0,
    )
}

pub fn remote_g2_candidate_query_by_target_with_cost<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request_blocks: usize,
    tiered_matches: &TieredMatchDetails,
    tax_blocks: u32,
    cap_blocks: Option<u32>,
    cost_per_block: f64,
) -> RemoteG2CandidateQueryResult {
    RemoteG2CandidateQueryPolicy::fixed_plus_per_block_cost(tax_blocks, cost_per_block, cap_blocks)
        .query_candidates_by_target(workers, request_blocks, tiered_matches)
}

fn remote_g2_selection_stats(tiered_matches: &TieredMatchDetails) -> RemoteKvReuseSelectionStats {
    RemoteKvReuseSelectionStats {
        rejected_g1_candidates: tiered_matches
            .device
            .overlap_scores
            .scores
            .values()
            .filter(|&&overlap| overlap > 0)
            .count() as u32,
    }
}

fn remote_g2_plan_from_candidate(
    input: RemoteKvReuseSelectionInput<'_>,
    best: RemoteKvReuseCandidate,
    stats: RemoteKvReuseSelectionStats,
) -> RemoteKvReuseDecision {
    // Continuation candidates start where the source worker's Device chain ended.
    // Write-through candidates start at root because their HostPinned copy mirrors
    // blocks that are also still present in the source worker's Device tier.
    let source = best.source;
    let start = best.start_block_index;
    let planned_prefix_blocks = best.planned_prefix_blocks as u32;
    if planned_prefix_blocks == 0 {
        return RemoteKvReuseDecision::NoPlan {
            reason: RemoteKvReuseNoPlanReason::NoContiguousPrefix,
            stats,
        };
    }
    let end = start + planned_prefix_blocks as usize;

    RemoteKvReuseDecision::Plan {
        plan: RemoteKvReusePlan {
            plan_id: format!(
                "remote-g2:{}:{}:{}:{}",
                input.request_id, source.worker_id, source.dp_rank, input.created_at_ms
            ),
            request_id: input.request_id.to_string(),
            target_worker_id: input.target.worker_id,
            target_dp_rank: input.target.dp_rank,
            source_worker_id: source.worker_id,
            source_dp_rank: source.dp_rank,
            source_host: String::new(),
            source_bootstrap_port: 0,
            source_tier: StorageTier::HostPinned,
            router_block_hashes: input.block_hashes[start..end].to_vec(),
            start_block_index: start as u32,
            planned_prefix_blocks,
            block_size_tokens: input.block_size_tokens,
            created_at_ms: input.created_at_ms,
            expires_at_ms: input.expires_at_ms,
            plan_version: REMOTE_KV_REUSE_PLAN_VERSION,
            // Caller fills this in post-selection by walking the indexer for
            // the chosen source. Left empty here so the planner stays a pure
            // function of `tiered_matches` and does not depend on the indexer.
            engine_block_hashes: Vec::new(),
        },
        stats,
    }
}

pub fn select_remote_g2_reuse_plan_from_candidate(
    input: RemoteKvReuseSelectionInput<'_>,
    candidate: RemoteKvReuseCandidate,
) -> RemoteKvReuseDecision {
    let stats = remote_g2_selection_stats(input.tiered_matches);
    remote_g2_plan_from_candidate(input, candidate, stats)
}

pub fn select_remote_g2_reuse_plan(
    input: RemoteKvReuseSelectionInput<'_>,
) -> RemoteKvReuseDecision {
    let stats = remote_g2_selection_stats(input.tiered_matches);

    let request_blocks = input.block_hashes.len();
    let saw_remote_candidate =
        has_remote_g2_candidate(input.target, request_blocks, input.tiered_matches);
    let Some(best) =
        best_remote_g2_candidate_for_target(input.target, request_blocks, input.tiered_matches)
    else {
        return RemoteKvReuseDecision::NoPlan {
            reason: if saw_remote_candidate {
                RemoteKvReuseNoPlanReason::NoContiguousPrefix
            } else {
                RemoteKvReuseNoPlanReason::NoRemoteG2Candidate
            },
            stats,
        };
    };

    remote_g2_plan_from_candidate(input, best, stats)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::indexer::{LowerTierMatchDetails, MatchDetails, TieredMatchDetails};
    use crate::protocols::{LocalBlockHash, OverlapScores, StorageTier, WorkerWithDpRank};
    use crate::remote_g2_plan::{
        REMOTE_KV_REUSE_PLAN_VERSION, RemoteG2CandidateQueryPolicy, RemoteKvReuseDecision,
        RemoteKvReuseNoPlanReason, RemoteKvReusePlan, RemoteKvReuseSelectionInput,
        remote_g2_candidate_query_by_target, remote_g2_score_blocks_for_target,
        remote_g2_scored_candidates_by_target, select_remote_g2_reuse_plan,
        select_remote_g2_reuse_plan_from_candidate,
    };
    use crate::test_utils::SimpleWorkerConfig;

    fn test_plan() -> RemoteKvReusePlan {
        RemoteKvReusePlan {
            plan_id: "plan-1".to_string(),
            request_id: "request-1".to_string(),
            target_worker_id: 9,
            target_dp_rank: 0,
            source_worker_id: 7,
            source_dp_rank: 1,
            source_host: "10.0.0.7".to_string(),
            source_bootstrap_port: 41000,
            source_tier: StorageTier::HostPinned,
            router_block_hashes: vec![LocalBlockHash(11), LocalBlockHash(22)],
            start_block_index: 0,
            planned_prefix_blocks: 2,
            block_size_tokens: 16,
            created_at_ms: 1000,
            expires_at_ms: 2000,
            plan_version: REMOTE_KV_REUSE_PLAN_VERSION,
            engine_block_hashes: vec![],
        }
    }

    fn block_hashes(count: u64) -> Vec<LocalBlockHash> {
        (0..count).map(LocalBlockHash).collect()
    }

    fn tiered_matches(
        device_hits: &[(WorkerWithDpRank, u32)],
        host_pinned_hits: &[(WorkerWithDpRank, usize)],
    ) -> TieredMatchDetails {
        let mut device = MatchDetails {
            overlap_scores: OverlapScores::new(),
            ..Default::default()
        };
        device
            .overlap_scores
            .scores
            .extend(device_hits.iter().copied());

        let mut lower_tier = std::collections::HashMap::new();
        let mut host_pinned = LowerTierMatchDetails::default();
        host_pinned.hits.extend(host_pinned_hits.iter().copied());
        lower_tier.insert(StorageTier::HostPinned, host_pinned);

        TieredMatchDetails { device, lower_tier }
    }

    fn selection_input<'a>(
        target: WorkerWithDpRank,
        block_hashes: &'a [LocalBlockHash],
        tiered_matches: &'a TieredMatchDetails,
    ) -> RemoteKvReuseSelectionInput<'a> {
        RemoteKvReuseSelectionInput {
            request_id: "request-1",
            target,
            block_hashes,
            block_size_tokens: 16,
            tiered_matches,
            created_at_ms: 1000,
            expires_at_ms: 2000,
        }
    }

    #[test]
    fn remote_kv_reuse_plan_round_trips_json() {
        let plan = test_plan();
        let json = serde_json::to_string(&plan).unwrap();
        let decoded: RemoteKvReusePlan = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, plan);
    }

    #[test]
    fn remote_kv_reuse_plan_round_trips_engine_block_hashes() {
        // Populated engine_block_hashes must appear in the JSON and survive
        // a serialize → deserialize round trip with the exact same values.
        let mut plan = test_plan();
        plan.engine_block_hashes = vec![
            0xAAAA_AAAA_AAAA_AAAA,
            0xBBBB_BBBB_BBBB_BBBB,
            0xCCCC_CCCC_CCCC_CCCC,
        ];
        let json = serde_json::to_string(&plan).unwrap();
        assert!(
            json.contains("\"engine_block_hashes\""),
            "serialized plan missing engine_block_hashes field: {json}"
        );
        // Big values must serialize as integers, not stringified
        assert!(json.contains("12297829382473034410"));
        let decoded: RemoteKvReusePlan = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.engine_block_hashes, plan.engine_block_hashes);
    }

    #[test]
    fn remote_kv_reuse_plan_serialization_has_no_forbidden_router_truth() {
        let json = serde_json::to_string(&test_plan()).unwrap();
        for forbidden in [
            "virtual_address",
            "physical_address",
            "nixl_descriptor",
            "descriptor",
            "target_g1_block_id",
            "source_block_id",
            "block_ptr",
            "handle",
        ] {
            assert!(
                !json.contains(forbidden),
                "serialized plan contains forbidden router truth: {forbidden}"
            );
        }
    }

    #[test]
    fn no_plan_reason_is_low_cardinality_snake_case() {
        let json = serde_json::to_string(&RemoteKvReuseNoPlanReason::NoRemoteG2Candidate).unwrap();
        assert_eq!(json, "\"no_remote_g2_candidate\"");
        let json = serde_json::to_string(&RemoteKvReuseNoPlanReason::BelowScoreTax).unwrap();
        assert_eq!(json, "\"below_score_tax\"");
    }

    #[test]
    fn selects_longest_remote_g2_prefix() {
        let hashes = block_hashes(5);
        let target = WorkerWithDpRank::new(9, 0);
        let matches = tiered_matches(
            &[],
            &[
                (WorkerWithDpRank::new(7, 0), 2),
                (WorkerWithDpRank::new(8, 0), 4),
            ],
        );

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.source_worker_id, 8);
                assert_eq!(plan.source_dp_rank, 0);
                assert_eq!(plan.planned_prefix_blocks, 4);
                assert_eq!(plan.router_block_hashes, hashes[..4].to_vec());
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    #[test]
    fn remote_g2_tie_break_is_stable_by_worker_then_rank() {
        let hashes = block_hashes(4);
        let target = WorkerWithDpRank::new(9, 0);
        let matches = tiered_matches(
            &[],
            &[
                (WorkerWithDpRank::new(8, 1), 3),
                (WorkerWithDpRank::new(7, 3), 3),
                (WorkerWithDpRank::new(7, 1), 3),
            ],
        );

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.source_worker_id, 7);
                assert_eq!(plan.source_dp_rank, 1);
                assert_eq!(plan.planned_prefix_blocks, 3);
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    #[test]
    fn remote_g1_device_hits_are_rejected_not_selected() {
        let hashes = block_hashes(2);
        let target = WorkerWithDpRank::new(9, 0);
        let matches = TieredMatchDetails {
            device: {
                let mut device = MatchDetails {
                    overlap_scores: OverlapScores::new(),
                    ..Default::default()
                };
                device
                    .overlap_scores
                    .scores
                    .insert(WorkerWithDpRank::new(7, 0), 2);
                device
            },
            lower_tier: Default::default(),
        };

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::NoPlan { reason, stats } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::NoRemoteG2Candidate);
                assert!(stats.rejected_g1_candidates > 0);
            }
            other => panic!("expected no plan, got {other:?}"),
        }
    }

    #[test]
    fn source_selection_does_not_change_target() {
        let hashes = block_hashes(3);
        let target = WorkerWithDpRank::new(42, 2);
        let matches = tiered_matches(&[], &[(WorkerWithDpRank::new(7, 0), 3)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.target_worker_id, 42);
                assert_eq!(plan.target_dp_rank, 2);
                assert_eq!(plan.source_worker_id, 7);
                assert_eq!(plan.source_dp_rank, 0);
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    #[test]
    fn zero_host_pinned_hits_return_no_contiguous_prefix() {
        let hashes = block_hashes(2);
        let target = WorkerWithDpRank::new(9, 0);
        let matches = tiered_matches(&[], &[(WorkerWithDpRank::new(7, 0), 0)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::NoPlan { reason, .. } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::NoContiguousPrefix);
            }
            other => panic!("expected no plan, got {other:?}"),
        }
    }

    #[test]
    fn plan_start_block_index_is_zero_when_source_has_no_device_match() {
        // Source A has 0 device-tier matches and 3 HostPinned hits → plan
        // covers request positions [0, 3) and start_block_index == 0.
        let hashes = block_hashes(5);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[], &[(source, 3)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.start_block_index, 0);
                assert_eq!(plan.planned_prefix_blocks, 3);
                assert_eq!(plan.router_block_hashes, hashes[..3].to_vec());
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    #[test]
    fn plan_start_block_index_equals_source_device_match() {
        // Source A has 2 device-tier matches and 2 HostPinned hits chained
        // past them → plan covers request positions [2, 4) and
        // start_block_index == 2 (skip past A's device chain). Target already
        // has the first 2 blocks locally, so it can attach this continuation.
        let hashes = block_hashes(6);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[(target, 2), (source, 2)], &[(source, 2)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.start_block_index, 2);
                assert_eq!(plan.planned_prefix_blocks, 2);
                assert_eq!(plan.router_block_hashes, hashes[2..4].to_vec());
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    #[test]
    fn continuation_before_target_prefix_returns_no_plan() {
        let hashes = block_hashes(6);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[(source, 2)], &[(source, 2)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::NoPlan { reason, .. } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::NoContiguousPrefix);
            }
            other => panic!("expected no plan, got {other:?}"),
        }
    }

    #[test]
    fn remote_g2_score_counts_only_incremental_blocks_after_tax() {
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[(target, 2), (source, 2)], &[(source, 5)]);

        assert_eq!(
            remote_g2_score_blocks_for_target(target, 8, &matches, 0, None),
            5
        );
        assert_eq!(
            remote_g2_score_blocks_for_target(target, 8, &matches, 2, None),
            3
        );
        assert_eq!(
            remote_g2_score_blocks_for_target(target, 8, &matches, 8, None),
            0
        );
    }

    #[test]
    fn remote_g2_scored_candidates_are_returned_by_target() {
        let target0 = WorkerWithDpRank::new(9, 0);
        let target1 = WorkerWithDpRank::new(9, 1);
        let source = WorkerWithDpRank::new(7, 0);
        let workers = HashMap::from([(
            9,
            SimpleWorkerConfig {
                data_parallel_start_rank: 0,
                data_parallel_size: 2,
                ..Default::default()
            },
        )]);
        let matches = tiered_matches(&[(target0, 2), (target1, 4), (source, 2)], &[(source, 8)]);

        let candidates = remote_g2_scored_candidates_by_target(&workers, 10, &matches, 2, None);

        assert_eq!(candidates.len(), 2);
        let candidate0 = candidates.get(&target0).unwrap();
        assert_eq!(candidate0.candidate.source, source);
        assert_eq!(candidate0.candidate.start_block_index, 2);
        assert_eq!(candidate0.candidate.incremental_blocks, 8);
        assert_eq!(candidate0.benefit_blocks, 8);
        assert_eq!(candidate0.cost_blocks, 2);
        assert_eq!(candidate0.score_blocks, 6);

        let candidate1 = candidates.get(&target1).unwrap();
        assert_eq!(candidate1.candidate.source, source);
        assert_eq!(candidate1.candidate.incremental_blocks, 6);
        assert_eq!(candidate1.benefit_blocks, 6);
        assert_eq!(candidate1.score_blocks, 4);
    }

    #[test]
    fn remote_g2_candidate_query_keeps_zero_score_candidate_out_of_score_map() {
        let target0 = WorkerWithDpRank::new(9, 0);
        let target1 = WorkerWithDpRank::new(9, 1);
        let source = WorkerWithDpRank::new(7, 0);
        let workers = HashMap::from([(
            9,
            SimpleWorkerConfig {
                data_parallel_start_rank: 0,
                data_parallel_size: 2,
                ..Default::default()
            },
        )]);
        let matches = tiered_matches(&[(target0, 2), (target1, 8), (source, 2)], &[(source, 8)]);

        let result = remote_g2_candidate_query_by_target(&workers, 10, &matches, 2, None);

        assert_eq!(result.scored_candidates.len(), 2);
        assert_eq!(
            result.scored_candidates.get(&target0).unwrap().score_blocks,
            6
        );
        assert_eq!(
            result.scored_candidates.get(&target1).unwrap().score_blocks,
            0
        );
        assert_eq!(result.benefit_blocks_by_target.get(&target0), Some(&8));
        assert_eq!(result.cost_blocks_by_target.get(&target0), Some(&2));
        assert_eq!(result.benefit_blocks_by_target.get(&target1), Some(&2));
        assert_eq!(result.cost_blocks_by_target.get(&target1), Some(&2));
        assert_eq!(result.score_blocks_by_target.get(&target0), Some(&6));
        assert!(!result.score_blocks_by_target.contains_key(&target1));
    }

    #[test]
    fn remote_g2_query_candidate_materializes_only_for_selected_target() {
        let target0 = WorkerWithDpRank::new(9, 0);
        let target1 = WorkerWithDpRank::new(9, 1);
        let source = WorkerWithDpRank::new(7, 0);
        let workers = HashMap::from([(
            9,
            SimpleWorkerConfig {
                data_parallel_start_rank: 0,
                data_parallel_size: 2,
                ..Default::default()
            },
        )]);
        let hashes = block_hashes(10);
        let matches = tiered_matches(&[(target0, 2), (target1, 4), (source, 2)], &[(source, 8)]);

        let result = remote_g2_candidate_query_by_target(&workers, hashes.len(), &matches, 2, None);
        let selected = result
            .selected_candidate(target1)
            .expect("expected selected target candidate");

        let decision = select_remote_g2_reuse_plan_from_candidate(
            selection_input(target1, &hashes, &matches),
            selected.candidate,
        );

        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.target_worker_id, target1.worker_id);
                assert_eq!(plan.target_dp_rank, target1.dp_rank);
                assert_eq!(plan.source_worker_id, source.worker_id);
                assert_eq!(plan.source_dp_rank, source.dp_rank);
                assert_eq!(plan.start_block_index, 2);
                assert_eq!(plan.planned_prefix_blocks, 8);
                assert_eq!(plan.router_block_hashes, hashes[2..10].to_vec());
            }
            other => panic!("expected selected-target plan, got {other:?}"),
        }
    }

    #[test]
    fn remote_g2_score_caps_incremental_blocks_after_tax() {
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[(target, 2), (source, 2)], &[(source, 8)]);

        assert_eq!(
            remote_g2_score_blocks_for_target(target, 10, &matches, 1, Some(3)),
            2
        );
        assert_eq!(
            remote_g2_score_blocks_for_target(target, 10, &matches, 1, Some(0)),
            7
        );
    }

    #[test]
    fn remote_g2_candidate_query_policy_caps_benefit_before_cost() {
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[(target, 2), (source, 2)], &[(source, 8)]);
        let policy = RemoteG2CandidateQueryPolicy::fixed_cost(2, Some(3));

        let candidate = policy
            .scored_candidate_for_target(target, 10, &matches)
            .expect("expected scored candidate");

        assert_eq!(candidate.candidate.source, source);
        assert_eq!(candidate.candidate.incremental_blocks, 8);
        assert_eq!(candidate.benefit_blocks, 3);
        assert_eq!(candidate.cost_blocks, 2);
        assert_eq!(candidate.score_blocks, 1);
    }

    #[test]
    fn remote_g2_candidate_query_policy_charges_per_incremental_block_cost() {
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[(target, 4), (source, 2)], &[(source, 8)]);
        let policy = RemoteG2CandidateQueryPolicy::fixed_plus_per_block_cost(2, 0.5, None);

        let candidate = policy
            .scored_candidate_for_target(target, 10, &matches)
            .expect("expected scored candidate");

        assert_eq!(candidate.candidate.source, source);
        assert_eq!(candidate.candidate.planned_prefix_blocks, 8);
        assert_eq!(candidate.candidate.incremental_blocks, 6);
        assert_eq!(candidate.benefit_blocks, 6);
        assert_eq!(candidate.cost_blocks, 5);
        assert_eq!(candidate.score_blocks, 1);
    }

    #[test]
    fn remote_g2_score_credits_cold_target_root_transfer() {
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[(source, 5)], &[(source, 0)]);

        let decision =
            select_remote_g2_reuse_plan(selection_input(target, &block_hashes(8), &matches));
        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.start_block_index, 0);
                assert_eq!(plan.planned_prefix_blocks, 5);
            }
            other => panic!("expected post-hoc root plan, got {other:?}"),
        }
        assert_eq!(
            remote_g2_score_blocks_for_target(target, 8, &matches, 0, None),
            5
        );
    }
}
