// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DualPath decode load balancer.
//!
//! Ports the decode-engine selection from DualPath (arXiv 2602.21548, section 6.1, "DE
//! Scheduling"). Decode load has two dimensions: the request count on an engine sets its step
//! batch size, and its total KV-cache length sets the attention cost of each step. The policy
//! balances both.
//!
//! For one request `r` with post-assignment length `len_r` and candidate engines `E` holding
//! `tok_e` active KV tokens:
//!
//! 1. **Across groups.** When workers expose several data-parallel ranks, the ranks of one worker
//!    form a group. The group with the fewest total active tokens is chosen first.
//! 2. **Within the group.** Engines whose remaining KV capacity cannot hold the request are
//!    skipped when a capacity is configured. The high-token threshold is
//!    `Z = high_token_factor * (len_r + sum(tok_e)) / |E|`. Engines with `tok_e + len_r <= Z`
//!    are preferred; among them the one with the fewest active requests wins. If every engine
//!    would exceed `Z`, the one with the fewest tokens wins to limit HBM pressure.
//!
//! Net effect: a short request lands on the engine with the fewest concurrent requests even when
//! that engine holds a relatively large KV footprint, while a long request is pushed to the
//! engine with the smallest footprint.
//!
//! Dynamo's load projection already includes this request's unshared KV blocks in
//! `decode_cost_blocks`, so `tok_e` is recovered as the projection minus the request's blocks and
//! `tok_e + len_r` is the projection plus the expected output. The paper computes `Z` over the
//! whole batch of pending requests; the router schedules one request at a time, so the batch is
//! that request.
//!
//! Ties rotate by least recent assignment rather than by worker id, so an idle pool spreads work
//! instead of stacking it on the lowest-numbered rank.
//!
//! Session-affinity targets are advisory and ignored; explicit request pins stay with the host.

use std::sync::Arc;

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    ScoredWorkerCandidate, WorkerInputView, WorkerInputs, WorkerLoadInput, WorkerPicker,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
    WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "dynamo-dualpath-decode";

/// Tunables for [`POLICY_TYPE`]. Every field is optional; unknown keys are rejected at startup.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// Multiplier on the mean post-assignment token count that defines a high-token engine.
    high_token_factor: f64,
    /// Per-rank KV capacity in tokens. Engines that cannot hold the request are skipped while
    /// another engine can. Unset disables the capacity check.
    kv_capacity_tokens: Option<u64>,
    /// Output tokens assumed for a request that carries no expected output length.
    default_output_tokens: u32,
    /// Treat the data-parallel ranks of one worker as a group and pick the lightest group first.
    group_by_worker: bool,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            high_token_factor: 1.05,
            kv_capacity_tokens: None,
            default_output_tokens: 0,
            group_by_worker: true,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if !self.high_token_factor.is_finite() || self.high_token_factor < 1.0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "high_token_factor must be a finite number greater than or equal to 1.0",
            ));
        }
        if self.kv_capacity_tokens == Some(0) {
            return Err(WorkerSelectionPolicyProviderError::new(
                "kv_capacity_tokens must be greater than zero when set",
            ));
        }
        Ok(())
    }
}

struct DualPathPicker {
    parameters: Parameters,
    /// Scratch rows reused across selections.
    rows: Vec<usize>,
    /// Selection counter at each rank's last assignment; ties go to the least recently assigned.
    last_assigned: std::collections::HashMap<dynamo_kv_router::protocols::WorkerWithDpRank, u64>,
    selections: u64,
}

impl DualPathPicker {
    fn new(parameters: Parameters) -> Self {
        Self {
            parameters,
            rows: Vec::new(),
            last_assigned: std::collections::HashMap::new(),
            selections: 0,
        }
    }

    fn remember(&mut self, candidates: &[ScoredWorkerCandidate], row: usize) -> usize {
        self.selections += 1;
        self.last_assigned
            .insert(candidates[row].worker(), self.selections);
        row
    }

    fn select_row(
        &mut self,
        candidates: &[ScoredWorkerCandidate],
        load: &[WorkerLoadInput],
        block_size: f64,
        request_blocks: f64,
        output_tokens: f64,
    ) -> Option<usize> {
        if candidates.is_empty() || candidates.len() != load.len() {
            return None;
        }
        let request_tokens = request_blocks * block_size + output_tokens;
        // Tokens on the engine before this request, and after it including the expected output.
        let current =
            |row: usize| (load[row].decode_cost_blocks() - request_blocks).max(0.0) * block_size;
        let post = |row: usize| load[row].decode_cost_blocks() * block_size + output_tokens;

        self.rows.clear();
        self.rows.extend(0..candidates.len());

        if self.parameters.group_by_worker {
            let first = candidates[0].worker().worker_id;
            let several_workers = candidates
                .iter()
                .any(|candidate| candidate.worker().worker_id != first);
            let several_ranks = candidates.iter().enumerate().any(|(row, candidate)| {
                candidates[..row]
                    .iter()
                    .any(|earlier| earlier.worker().worker_id == candidate.worker().worker_id)
            });
            if several_workers && several_ranks {
                // (worker id, total current tokens, most recent assignment in the group)
                let mut lightest: Option<(u64, f64, u64)> = None;
                for &row in &self.rows {
                    let worker_id = candidates[row].worker().worker_id;
                    let (total, latest) = self
                        .rows
                        .iter()
                        .filter(|&&other| candidates[other].worker().worker_id == worker_id)
                        .fold((0.0, 0), |(total, latest), &other| {
                            let seen = self
                                .last_assigned
                                .get(&candidates[other].worker())
                                .copied()
                                .unwrap_or(0);
                            (total + current(other), latest.max(seen))
                        });
                    let better = match lightest {
                        None => true,
                        Some((best_id, best_total, best_latest)) => total
                            .total_cmp(&best_total)
                            .then_with(|| latest.cmp(&best_latest))
                            .then_with(|| worker_id.cmp(&best_id))
                            .is_lt(),
                    };
                    if better {
                        lightest = Some((worker_id, total, latest));
                    }
                }
                let (worker_id, _, _) = lightest?;
                self.rows
                    .retain(|&row| candidates[row].worker().worker_id == worker_id);
            }
        }

        if let Some(capacity) = self.parameters.kv_capacity_tokens {
            let capacity = capacity as f64;
            if self.rows.iter().any(|&row| post(row) <= capacity) {
                self.rows.retain(|&row| post(row) <= capacity);
            }
        }

        let total_current: f64 = self.rows.iter().map(|&row| current(row)).sum();
        let threshold = self.parameters.high_token_factor * (request_tokens + total_current)
            / self.rows.len() as f64;

        let last = |row: usize| {
            self.last_assigned
                .get(&candidates[row].worker())
                .copied()
                .unwrap_or(0)
        };
        let under_threshold = self
            .rows
            .iter()
            .copied()
            .filter(|&row| post(row) <= threshold)
            .min_by(|&left, &right| {
                load[left]
                    .active_requests()
                    .cmp(&load[right].active_requests())
                    .then_with(|| current(left).total_cmp(&current(right)))
                    .then_with(|| last(left).cmp(&last(right)))
                    .then_with(|| candidates[left].worker().cmp(&candidates[right].worker()))
            });
        if let Some(row) = under_threshold {
            return Some(self.remember(candidates, row));
        }
        let row = self.rows.iter().copied().min_by(|&left, &right| {
            post(left)
                .total_cmp(&post(right))
                .then_with(|| {
                    load[left]
                        .active_requests()
                        .cmp(&load[right].active_requests())
                })
                .then_with(|| last(left).cmp(&last(right)))
                .then_with(|| candidates[left].worker().cmp(&candidates[right].worker()))
        })?;
        Some(self.remember(candidates, row))
    }
}

impl WorkerPicker for DualPathPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let output_tokens = context
            .expected_output_tokens()
            .unwrap_or(self.parameters.default_output_tokens);
        self.select_row(
            input.candidates(),
            load,
            f64::from(context.block_size()),
            context.request_blocks() as f64,
            f64::from(output_tokens),
        )
        .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    parameters.validate()?;

    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                Vec::new(),
                Box::new(DualPathPicker::new(parameters)),
            )
        },
    ))
}

pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(POLICY_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
    use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
    use dynamo_kv_router::{
        SchedulingRequest, WorkerLoadProjection, WorkerSelectionInput, WorkerSelector,
    };

    use super::*;

    const BLOCK_SIZE: u32 = 64;

    /// A worker exposing `dp` data-parallel ranks.
    struct TestWorker {
        dp: u32,
    }

    impl WorkerConfigLike for TestWorker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }
        fn data_parallel_size(&self) -> u32 {
            self.dp
        }
        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }
        fn total_kv_blocks(&self) -> Option<u64> {
            Some(1 << 20)
        }
    }

    /// One candidate rank: `(worker_id, dp_rank, active_tokens, active_requests)`.
    type Rank = (u64, u32, usize, usize);

    fn select_with(
        parameters: Parameters,
        prompt_tokens: usize,
        output_tokens: Option<u32>,
        ranks: &[Rank],
    ) -> WorkerWithDpRank {
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: prompt_tokens,
            lora_name: None,
            expected_output_tokens: output_tokens,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            track_prefill_tokens: true,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            overlap: OverlapSignals::default(),
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            shared_cache_hits: None,
            worker_loads: Default::default(),
            resp_tx: None,
        };
        let request_blocks = prompt_tokens.div_ceil(BLOCK_SIZE as usize);
        let mut workers: HashMap<u64, TestWorker> = HashMap::new();
        for &(worker_id, dp_rank, active_tokens, active_requests) in ranks {
            let worker = workers.entry(worker_id).or_insert(TestWorker { dp: 0 });
            worker.dp = worker.dp.max(dp_rank + 1);
            // No prefix sharing: the whole request adds to the engine's footprint.
            request.worker_loads.insert(
                WorkerWithDpRank::new(worker_id, dp_rank),
                WorkerLoadProjection {
                    active_decode_blocks: active_tokens / BLOCK_SIZE as usize,
                    additional_active_blocks: request_blocks,
                    active_requests,
                    ..Default::default()
                },
            );
        }
        WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "test",
            Vec::new(),
            Box::new(DualPathPicker::new(parameters)),
        )
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            BLOCK_SIZE,
        ))
        .unwrap()
        .worker
    }

    fn select(prompt_tokens: usize, ranks: &[Rank]) -> WorkerWithDpRank {
        select_with(Parameters::default(), prompt_tokens, None, ranks)
    }

    #[test]
    fn short_requests_follow_request_count_when_tokens_are_balanced() {
        // Both engines stay under Z = 1.05 * (1K + 118K) / 2 = 62.5K, so the fewer-request
        // engine wins even though it holds slightly more KV.
        let ranks = [(1, 0, 60_000, 2), (2, 0, 58_000, 6)];
        assert_eq!(select(1_024, &ranks), WorkerWithDpRank::new(1, 0));
    }

    #[test]
    fn long_requests_go_to_the_smaller_footprint() {
        // Z = 1.05 * (20K + 118K) / 2 = 72.5K: both engines would exceed it, so tokens decide.
        let ranks = [(1, 0, 60_000, 2), (2, 0, 58_000, 6)];
        assert_eq!(select(20_480, &ranks), WorkerWithDpRank::new(2, 0));
    }

    #[test]
    fn a_heavy_engine_is_excluded_while_a_light_one_fits() {
        // Engine 1 holds 100K; Z = 1.05 * (1K + 120K) / 2 = 63.5K puts it in the high-token
        // category, so the busier but lighter engine 2 takes the short request.
        let ranks = [(1, 0, 100_000, 1), (2, 0, 20_000, 8)];
        assert_eq!(select(1_024, &ranks), WorkerWithDpRank::new(2, 0));
    }

    #[test]
    fn the_lightest_worker_group_is_chosen_first() {
        // Worker 1 totals 90K over two ranks; worker 2 totals 70K, so worker 2's ranks are the
        // group. Both stay under Z = 1.05 * (2K + 70K) / 2 = 37.8K, so request count decides.
        let ranks = [
            (1, 0, 10_000, 1),
            (1, 1, 80_000, 1),
            (2, 0, 35_000, 3),
            (2, 1, 35_000, 4),
        ];
        assert_eq!(select(2_048, &ranks), WorkerWithDpRank::new(2, 0));
        let flat = Parameters {
            group_by_worker: false,
            ..Parameters::default()
        };
        // Without grouping the global rule applies: all four fit under Z, fewest requests wins.
        assert_eq!(
            select_with(flat, 2_048, None, &ranks),
            WorkerWithDpRank::new(1, 0)
        );
    }

    #[test]
    fn capacity_filter_skips_engines_that_cannot_hold_the_request() {
        let capped = Parameters {
            kv_capacity_tokens: Some(64_000),
            ..Parameters::default()
        };
        // Engine 1 would reach 64,512 tokens, over capacity; engine 2 fits despite more requests.
        let ranks = [(1, 0, 60_000, 1), (2, 0, 30_000, 9)];
        assert_eq!(
            select_with(capped, 4_096, Some(416), &ranks),
            WorkerWithDpRank::new(2, 0)
        );
        // When nothing fits, the filter relaxes instead of failing.
        let tight = Parameters {
            kv_capacity_tokens: Some(1_000),
            ..Parameters::default()
        };
        assert_eq!(
            select_with(tight, 4_096, None, &ranks),
            WorkerWithDpRank::new(2, 0)
        );
    }

    #[test]
    fn expected_output_counts_toward_the_request_length() {
        // With no output estimate engine 1 (fewer requests) stays under Z and wins; a large
        // expected output pushes it over Z, so the request moves to the smaller footprint.
        let ranks = [(1, 0, 60_000, 2), (2, 0, 58_000, 6)];
        assert_eq!(
            select_with(Parameters::default(), 1_024, None, &ranks),
            WorkerWithDpRank::new(1, 0)
        );
        assert_eq!(
            select_with(Parameters::default(), 1_024, Some(20_000), &ranks),
            WorkerWithDpRank::new(2, 0)
        );
    }

    #[test]
    fn idle_ranks_rotate_instead_of_stacking_on_the_lowest_id() {
        let ranks = [(1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0)];
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: 1_024,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            track_prefill_tokens: true,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            overlap: OverlapSignals::default(),
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            shared_cache_hits: None,
            worker_loads: Default::default(),
            resp_tx: None,
        };
        let mut workers = HashMap::new();
        for &(worker_id, dp_rank, _, _) in &ranks {
            workers.insert(worker_id, TestWorker { dp: 2 });
            request.worker_loads.insert(
                WorkerWithDpRank::new(worker_id, dp_rank),
                WorkerLoadProjection {
                    additional_active_blocks: 16,
                    ..Default::default()
                },
            );
        }
        let policy = WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "test",
            Vec::new(),
            Box::new(DualPathPicker::new(Parameters::default())),
        );
        let mut picked = Vec::new();
        for _ in 0..4 {
            picked.push(
                policy
                    .select_worker(WorkerSelectionInput::configured(
                        &workers,
                        &request,
                        request.eligibility(),
                        BLOCK_SIZE,
                    ))
                    .unwrap()
                    .worker,
            );
        }
        picked.sort();
        // Both groups tie on tokens, so group selection alternates on id and each rank is used once.
        assert_eq!(
            picked,
            vec![
                WorkerWithDpRank::new(1, 0),
                WorkerWithDpRank::new(1, 1),
                WorkerWithDpRank::new(2, 0),
                WorkerWithDpRank::new(2, 1),
            ]
        );
    }

    #[test]
    fn rejects_invalid_parameters() {
        assert!(Parameters::default().validate().is_ok());
        assert!(
            Parameters {
                high_token_factor: 0.9,
                ..Parameters::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            Parameters {
                high_token_factor: f64::NAN,
                ..Parameters::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            Parameters {
                kv_capacity_tokens: Some(0),
                ..Parameters::default()
            }
            .validate()
            .is_err()
        );
    }
}
