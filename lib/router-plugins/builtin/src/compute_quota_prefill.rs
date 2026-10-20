// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compute-quota prefill load balancer.
//!
//! Ports the prefill-side routing policy described with DualPath (arXiv 2602.21548, section 6).
//! Routing is organized around an estimate of attention-sublayer work, the *compute quota*, from
//! a quadratic cost model fitted in advance:
//!
//! ```text
//! cost(q, c) = cost_quadratic * q^2 + cost_cross * q * c + cost_linear * q
//! ```
//!
//! `q` is the number of prompt tokens not covered by a cache hit and `c` is the length of the
//! prefix that is already computed. The defaults (`0.5`, `1.0`, `0.0`) count attention
//! query-key pairs, so one quota unit is one token pair.
//!
//! Three of the paper's four levels run here:
//!
//! 1. **Bucketing.** Requests fall into buckets by total prompt length (defaults `<16K`,
//!    `16K-64K`, `64K-256K`, `>=256K`). Once per round, engines (worker ids) are reassigned to
//!    buckets in proportion to each bucket's share of compute-quota demand in the previous round.
//!    A bucket with demand gets at least one engine when there are enough engines.
//! 2. **Across engines.** Within the request's bucket, the engine whose rank has the least
//!    remaining compute quota pulls the request. There is no prefix-affinity term; cache state
//!    only enters through `q`.
//! 3. **Within an engine.** The request goes to the data-parallel rank with the lowest remaining
//!    compute-quota load. Because Dynamo exposes `(worker, dp_rank)` candidates, levels 2 and 3
//!    collapse into one argmin over `remaining + cost(q, c)`.
//!
//! The fourth level, deriving each step's chunk size from a time budget, is engine scheduling
//! and is out of scope for a worker-selection policy.
//!
//! Remaining quota per rank is a leaky-bucket ledger owned by the policy: each assignment adds
//! `cost(q, c)` and the ledger drains at `drain_rate_per_sec`. When the host reports a rank with
//! no active prefill tokens and no active requests, that rank's ledger resets so a stale estimate
//! cannot starve an idle rank.
//!
//! Session-affinity targets are advisory and ignored here, matching the paper's choice not to
//! route by prefix affinity. Explicit request pins stay with the host.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerInputs, WorkerLoadInput, WorkerPicker, WorkerSelectionContext,
    WorkerSelectionPolicy, WorkerSelectionPolicyError, WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::protocols::WorkerWithDpRank;

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "dynamo-compute-quota-prefill";

/// Tunables for [`POLICY_TYPE`].
///
/// Every field is optional. Unknown keys are rejected at startup.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// Exclusive upper bounds of every bucket but the last, ascending, in prompt tokens.
    bucket_bounds: Vec<usize>,
    /// Length of one engine-allocation round in milliseconds.
    round_ms: u64,
    /// Coefficient of `q^2` in the cost model.
    cost_quadratic: f64,
    /// Coefficient of `q * c` in the cost model.
    cost_cross: f64,
    /// Coefficient of `q` in the cost model.
    cost_linear: f64,
    /// Compute quota one rank retires per second; converts the ledger into remaining work.
    drain_rate_per_sec: f64,
    /// Reset a rank's ledger when the host reports no active prefill tokens and no active requests.
    reconcile_with_host_load: bool,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            bucket_bounds: vec![16 * 1024, 64 * 1024, 256 * 1024],
            round_ms: 1000,
            cost_quadratic: 0.5,
            cost_cross: 1.0,
            cost_linear: 0.0,
            // One H100-class rank retiring roughly 32K uncached tokens (0.5 * 32768^2 pairs)
            // every two seconds. Calibrate from the deployment's measured prefill throughput.
            drain_rate_per_sec: 2.5e8,
            reconcile_with_host_load: true,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if self.bucket_bounds.windows(2).any(|pair| pair[0] >= pair[1])
            || self.bucket_bounds.first().is_some_and(|&bound| bound == 0)
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "bucket_bounds must be strictly ascending and greater than zero",
            ));
        }
        if self.round_ms == 0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "round_ms must be greater than zero",
            ));
        }
        let coefficients = [self.cost_quadratic, self.cost_cross, self.cost_linear];
        if coefficients
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "cost_quadratic, cost_cross, and cost_linear must be finite and non-negative",
            ));
        }
        if coefficients.iter().all(|value| *value == 0.0) {
            return Err(WorkerSelectionPolicyProviderError::new(
                "at least one cost coefficient must be greater than zero",
            ));
        }
        if !self.drain_rate_per_sec.is_finite() || self.drain_rate_per_sec <= 0.0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "drain_rate_per_sec must be finite and greater than zero",
            ));
        }
        Ok(())
    }

    fn cost(&self, uncached_tokens: f64, cached_tokens: f64) -> f64 {
        self.cost_quadratic * uncached_tokens * uncached_tokens
            + self.cost_cross * uncached_tokens * cached_tokens
            + self.cost_linear * uncached_tokens
    }

    fn bucket_count(&self) -> usize {
        self.bucket_bounds.len() + 1
    }

    fn bucket_index(&self, prompt_tokens: usize) -> usize {
        self.bucket_bounds
            .iter()
            .position(|&bound| prompt_tokens < bound)
            .unwrap_or(self.bucket_bounds.len())
    }
}

/// Split `engines` across buckets in proportion to `demand`, by largest remainder.
///
/// Every bucket with demand receives at least one engine when there are enough engines for all
/// of them; otherwise the smallest-demand buckets are left without engines. Zero total demand
/// returns zero everywhere, which callers treat as "unpartitioned".
fn allocate_engines(demand: &[f64], engines: usize) -> Vec<usize> {
    let total: f64 = demand.iter().sum();
    let mut counts = vec![0usize; demand.len()];
    if engines == 0 || total.is_nan() || total <= 0.0 {
        return counts;
    }
    let mut remainders: Vec<(f64, usize)> = Vec::with_capacity(demand.len());
    let mut assigned = 0;
    for (bucket, share) in demand.iter().enumerate() {
        let exact = share / total * engines as f64;
        let floor = exact.floor() as usize;
        counts[bucket] = floor;
        assigned += floor;
        remainders.push((exact - floor as f64, bucket));
    }
    remainders.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    for (_, bucket) in remainders.iter().take(engines.saturating_sub(assigned)) {
        counts[*bucket] += 1;
    }

    let active = demand.iter().filter(|share| **share > 0.0).count();
    if engines >= active {
        while let Some(starved) = counts
            .iter()
            .enumerate()
            .position(|(bucket, &count)| count == 0 && demand[bucket] > 0.0)
        {
            let Some(donor) = counts
                .iter()
                .enumerate()
                .filter(|&(_, &count)| count > 1)
                .max_by(|left, right| left.1.cmp(right.1))
                .map(|(bucket, _)| bucket)
            else {
                break;
            };
            counts[donor] -= 1;
            counts[starved] += 1;
        }
    }
    counts
}

struct RankLedger {
    remaining: f64,
    updated_at: Instant,
}

impl RankLedger {
    /// Drain the ledger to `now` and apply the host's idle signal.
    fn remaining_at(
        &mut self,
        now: Instant,
        drain_rate_per_sec: f64,
        load: &WorkerLoadInput,
        reconcile: bool,
    ) -> f64 {
        if now > self.updated_at {
            let drained = now.duration_since(self.updated_at).as_secs_f64() * drain_rate_per_sec;
            self.remaining = (self.remaining - drained).max(0.0);
            self.updated_at = now;
        }
        if reconcile
            && load.is_available()
            && load.active_prefill_tokens() == 0
            && load.active_requests() == 0
        {
            self.remaining = 0.0;
        }
        self.remaining
    }
}

struct ComputeQuotaPicker {
    parameters: Parameters,
    round: Duration,
    ledgers: HashMap<WorkerWithDpRank, RankLedger>,
    round_started_at: Instant,
    /// Compute-quota demand per bucket accumulated in the current round.
    round_demand: Vec<f64>,
    /// Engine-to-bucket assignment from the last rebalance; empty means unpartitioned.
    engine_bucket: HashMap<u64, usize>,
    /// Sorted distinct engines seen at the last rebalance; reused to avoid reallocating.
    engines: Vec<u64>,
    clock: Clock,
}

/// Monotonic time source, overridable in tests.
enum Clock {
    System,
    #[cfg(test)]
    Manual(Instant, Arc<std::sync::atomic::AtomicU64>),
}

impl Clock {
    fn now(&self) -> Instant {
        match self {
            Clock::System => Instant::now(),
            #[cfg(test)]
            Clock::Manual(base, nanos) => {
                *base + Duration::from_nanos(nanos.load(std::sync::atomic::Ordering::Relaxed))
            }
        }
    }
}

impl ComputeQuotaPicker {
    fn new(parameters: Parameters, clock: Clock) -> Self {
        let now = clock.now();
        Self {
            round: Duration::from_millis(parameters.round_ms),
            round_demand: vec![0.0; parameters.bucket_count()],
            parameters,
            ledgers: HashMap::new(),
            round_started_at: now,
            engine_bucket: HashMap::new(),
            engines: Vec::new(),
            clock,
        }
    }

    /// Reassign engines to buckets from the demand of the round that just ended.
    fn rebalance(&mut self, now: Instant, candidates: impl Iterator<Item = WorkerWithDpRank>) {
        self.engines.clear();
        self.engines
            .extend(candidates.map(|worker| worker.worker_id));
        self.engines.sort_unstable();
        self.engines.dedup();

        let counts = allocate_engines(&self.round_demand, self.engines.len());
        self.engine_bucket.clear();
        let mut next = 0;
        for (bucket, count) in counts.into_iter().enumerate() {
            for &engine in &self.engines[next..next + count] {
                self.engine_bucket.insert(engine, bucket);
            }
            next += count;
        }
        self.round_demand.fill(0.0);
        self.round_started_at = now;
        tracing::debug!(
            engines = self.engines.len(),
            assignment = ?self.engine_bucket,
            "compute-quota prefill policy rebalanced engine buckets"
        );
    }

    fn select(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let cache = input
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        if candidates.is_empty()
            || load.len() != candidates.len()
            || cache.len() != candidates.len()
        {
            return Err(WorkerSelectionPolicyError::failed("no eligible worker"));
        }

        let now = self.clock.now();
        if now.duration_since(self.round_started_at) >= self.round {
            self.rebalance(now, candidates.iter().map(|candidate| candidate.worker()));
        }

        let prompt_tokens = context.prompt_tokens();
        let bucket = self.parameters.bucket_index(prompt_tokens);
        let Self {
            parameters,
            ledgers,
            engine_bucket,
            round_demand,
            ..
        } = self;
        // Engines absent from the assignment (joined after the last rebalance) serve every bucket.
        let in_bucket = |worker: WorkerWithDpRank| {
            engine_bucket
                .get(&worker.worker_id)
                .is_none_or(|&assigned| assigned == bucket)
        };
        let restrict = candidates
            .iter()
            .any(|candidate| in_bucket(candidate.worker()));

        let mut best: Option<(usize, f64, f64)> = None;
        for (row, candidate) in candidates.iter().enumerate() {
            let worker = candidate.worker();
            if restrict && !in_bucket(worker) {
                continue;
            }
            let remaining = ledgers
                .entry(worker)
                .or_insert(RankLedger {
                    remaining: 0.0,
                    updated_at: now,
                })
                .remaining_at(
                    now,
                    parameters.drain_rate_per_sec,
                    &load[row],
                    parameters.reconcile_with_host_load,
                );
            let cached_tokens = cache
                .get(row)
                .map(|cache| cache.accounting_cache_estimate().1)
                .unwrap_or(0)
                .min(prompt_tokens);
            let cost =
                parameters.cost((prompt_tokens - cached_tokens) as f64, cached_tokens as f64);
            let total = remaining + cost;
            let better = match best {
                None => true,
                Some((best_row, best_total, best_remaining)) => total
                    .total_cmp(&best_total)
                    .then_with(|| remaining.total_cmp(&best_remaining))
                    .then_with(|| worker.cmp(&candidates[best_row].worker()))
                    .is_lt(),
            };
            if better {
                best = Some((row, total, remaining));
            }
        }

        let (row, total, remaining) =
            best.ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))?;
        if let Some(ledger) = ledgers.get_mut(&candidates[row].worker()) {
            ledger.remaining = total;
        }
        round_demand[bucket] += total - remaining;
        Ok(row)
    }
}

impl WorkerPicker for ComputeQuotaPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        self.select(context, input)
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
                Box::new(ComputeQuotaPicker::new(parameters.clone(), Clock::System)),
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
    use std::sync::atomic::{AtomicU64, Ordering};

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

    /// One candidate rank: `(worker_id, dp_rank, cached_tokens, active_prefill_tokens, active_requests)`.
    type Rank = (u64, u32, usize, usize, usize);

    fn request(prompt_tokens: usize, ranks: &[Rank]) -> SchedulingRequest {
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: prompt_tokens,
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
        for &(worker_id, dp_rank, cached, prefill_tokens, active_requests) in ranks {
            let worker = WorkerWithDpRank::new(worker_id, dp_rank);
            let blocks = cached / BLOCK_SIZE as usize;
            request
                .overlap
                .tier_overlap_blocks
                .device
                .insert(worker, blocks);
            request
                .overlap
                .effective_overlap_blocks
                .insert(worker, blocks as f64);
            request
                .overlap
                .effective_cached_tokens
                .insert(worker, cached);
            request.worker_loads.insert(
                worker,
                WorkerLoadProjection {
                    active_prefill_tokens: prefill_tokens,
                    active_requests,
                    ..Default::default()
                },
            );
        }
        request
    }

    struct Harness {
        policy: WorkerSelectionPolicy,
        nanos: Arc<AtomicU64>,
    }

    impl Harness {
        fn new(parameters: Parameters) -> Self {
            let nanos = Arc::new(AtomicU64::new(0));
            let picker =
                ComputeQuotaPicker::new(parameters, Clock::Manual(Instant::now(), nanos.clone()));
            Self {
                policy: WorkerSelectionPolicy::new(
                    KvRouterConfig::default(),
                    "test",
                    Vec::new(),
                    Box::new(picker),
                ),
                nanos,
            }
        }

        fn advance(&self, duration: Duration) {
            self.nanos
                .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        }

        fn select(&self, prompt_tokens: usize, ranks: &[Rank]) -> WorkerWithDpRank {
            let request = request(prompt_tokens, ranks);
            let mut workers: HashMap<u64, TestWorker> = HashMap::new();
            for &(worker_id, dp_rank, ..) in ranks {
                let worker = workers.entry(worker_id).or_insert(TestWorker { dp: 0 });
                worker.dp = worker.dp.max(dp_rank + 1);
            }
            self.policy
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    &request,
                    request.eligibility(),
                    BLOCK_SIZE,
                ))
                .unwrap()
                .worker
        }
    }

    fn busy(worker_id: u64, dp_rank: u32) -> Rank {
        (worker_id, dp_rank, 0, 1, 1)
    }

    #[test]
    fn cost_model_counts_attention_pairs_by_default() {
        let parameters = Parameters::default();
        assert_eq!(parameters.cost(0.0, 1000.0), 0.0);
        assert_eq!(parameters.cost(100.0, 0.0), 5_000.0);
        assert_eq!(parameters.cost(100.0, 1000.0), 105_000.0);
        assert_eq!(parameters.bucket_index(0), 0);
        assert_eq!(parameters.bucket_index(16 * 1024 - 1), 0);
        assert_eq!(parameters.bucket_index(16 * 1024), 1);
        assert_eq!(parameters.bucket_index(64 * 1024), 2);
        assert_eq!(parameters.bucket_index(256 * 1024), 3);
        assert_eq!(parameters.bucket_index(1 << 20), 3);
    }

    #[test]
    fn rejects_invalid_parameters() {
        let valid = Parameters::default();
        assert!(valid.validate().is_ok());
        let unsorted = Parameters {
            bucket_bounds: vec![65536, 16384],
            ..Parameters::default()
        };
        assert!(unsorted.validate().is_err());
        let no_cost = Parameters {
            cost_quadratic: 0.0,
            cost_cross: 0.0,
            cost_linear: 0.0,
            ..Parameters::default()
        };
        assert!(no_cost.validate().is_err());
        let no_drain = Parameters {
            drain_rate_per_sec: 0.0,
            ..Parameters::default()
        };
        assert!(no_drain.validate().is_err());
        let no_round = Parameters {
            round_ms: 0,
            ..Parameters::default()
        };
        assert!(no_round.validate().is_err());
        let no_buckets = Parameters {
            bucket_bounds: Vec::new(),
            ..Parameters::default()
        };
        assert!(no_buckets.validate().is_ok());
        assert_eq!(no_buckets.bucket_count(), 1);
    }

    #[test]
    fn allocation_follows_demand_share_with_a_floor_for_active_buckets() {
        assert_eq!(allocate_engines(&[0.0, 0.0, 0.0, 0.0], 8), vec![0, 0, 0, 0]);
        assert_eq!(allocate_engines(&[1.0, 1.0, 1.0, 1.0], 8), vec![2, 2, 2, 2]);
        assert_eq!(allocate_engines(&[7.0, 1.0, 0.0, 0.0], 8), vec![7, 1, 0, 0]);
        // 1% of demand still earns one engine when there are enough engines to go around; the
        // largest bucket donates it (5.2, 2.0, 0.72, 0.08 -> 5, 2, 1, 0 -> 4, 2, 1, 1).
        assert_eq!(
            allocate_engines(&[65.0, 25.0, 9.0, 1.0], 8),
            vec![4, 2, 1, 1]
        );
        // Fewer engines than active buckets: the smallest shares go without.
        assert_eq!(
            allocate_engines(&[65.0, 25.0, 9.0, 1.0], 2),
            vec![1, 1, 0, 0]
        );
        assert_eq!(allocate_engines(&[1.0, 1.0], 0), vec![0, 0]);
    }

    #[test]
    fn picks_the_rank_with_the_least_quota_after_assignment() {
        let harness = Harness::new(Parameters::default());
        let ranks = [busy(1, 0), busy(1, 1), busy(2, 0), busy(2, 1)];
        // Four equal-cost requests in one round fill all four ranks once each, lowest id first.
        let mut picked: Vec<_> = (0..4).map(|_| harness.select(4096, &ranks)).collect();
        picked.sort();
        assert_eq!(
            picked,
            vec![
                WorkerWithDpRank::new(1, 0),
                WorkerWithDpRank::new(1, 1),
                WorkerWithDpRank::new(2, 0),
                WorkerWithDpRank::new(2, 1),
            ]
        );
        // A fifth request lands on the rank whose ledger drained first: all equal, so lowest id.
        assert_eq!(harness.select(4096, &ranks), WorkerWithDpRank::new(1, 0));
    }

    #[test]
    fn cache_hits_lower_the_cost_through_uncached_tokens_only() {
        let harness = Harness::new(Parameters::default());
        // Worker 2 has the whole prompt cached, so q = 0 and its cost is zero.
        let ranks = [(1, 0, 0, 1, 1), (2, 0, 8192, 1, 1)];
        assert_eq!(harness.select(8192, &ranks), WorkerWithDpRank::new(2, 0));
        // Now worker 2 carries that ledger; a cold request still prefers the empty ledger.
        assert_eq!(harness.select(8192, &ranks), WorkerWithDpRank::new(2, 0));
        let cold = [(1, 0, 0, 1, 1), (2, 0, 0, 1, 1)];
        assert_eq!(harness.select(8192, &cold), WorkerWithDpRank::new(1, 0));
    }

    #[test]
    fn ledgers_drain_over_time_and_reset_on_an_idle_host_signal() {
        let parameters = Parameters {
            drain_rate_per_sec: 1.0e6,
            round_ms: 60_000,
            ..Parameters::default()
        };
        let harness = Harness::new(parameters);
        let ranks = [busy(1, 0), busy(2, 0)];
        // 4096 uncached tokens cost 0.5 * 4096^2 = 8.4M quota units on worker 1.
        assert_eq!(harness.select(4096, &ranks), WorkerWithDpRank::new(1, 0));
        // 1024 tokens cost 0.5M on worker 2, which then still has the lower ledger.
        assert_eq!(harness.select(1024, &ranks), WorkerWithDpRank::new(2, 0));
        assert_eq!(harness.select(1024, &ranks), WorkerWithDpRank::new(2, 0));
        // After nine seconds, worker 1 has drained 9M of its 8.4M; worker 2 drained to zero too.
        harness.advance(Duration::from_secs(9));
        assert_eq!(harness.select(1024, &ranks), WorkerWithDpRank::new(1, 0));
        // The host reporting worker 2 idle clears its ledger even without elapsed time.
        assert_eq!(harness.select(4096, &ranks), WorkerWithDpRank::new(2, 0));
        let idle_two = [busy(1, 0), (2, 0, 0, 0, 0)];
        assert_eq!(harness.select(1024, &idle_two), WorkerWithDpRank::new(2, 0));
    }

    #[test]
    fn buckets_partition_engines_by_previous_round_demand() {
        let parameters = Parameters {
            round_ms: 1000,
            drain_rate_per_sec: 1.0,
            ..Parameters::default()
        };
        let harness = Harness::new(parameters);
        let ranks: Vec<Rank> = (1..=4).map(|id| busy(id, 0)).collect();
        // Round one: short requests dominate demand (16 x 8K cost 0.5 * 8192^2 each = 537M),
        // one 64K request costs 2.1G, so demand is 20% bucket 0 and 80% bucket 2.
        for _ in 0..16 {
            harness.select(8192, &ranks);
        }
        harness.select(65536, &ranks);
        harness.advance(Duration::from_secs(1));
        // Round two rebalances on arrival: bucket 0 keeps one engine (the floor), bucket 2 gets
        // three. Round-one ledgers still hold (engine 1 carries the 64K request), so the three
        // long requests spread over the three long-bucket engines in ledger order.
        let short = harness.select(8192, &ranks);
        let long: Vec<_> = (0..3).map(|_| harness.select(65536, &ranks)).collect();
        assert_eq!(short, WorkerWithDpRank::new(1, 0));
        assert!(
            long.iter().all(|worker| worker.worker_id >= 2),
            "long requests must stay off the short bucket's engine: {long:?}"
        );
        assert_eq!(
            long.iter()
                .map(|worker| worker.worker_id)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            3,
            "three idle long-bucket engines take one request each: {long:?}"
        );
        // A bucket without engines falls back to the whole pool rather than failing.
        assert!(harness.select(300_000, &ranks).worker_id >= 1);
    }

    #[test]
    fn engines_that_join_after_a_rebalance_serve_every_bucket() {
        let parameters = Parameters {
            round_ms: 1000,
            drain_rate_per_sec: 1.0,
            ..Parameters::default()
        };
        let harness = Harness::new(parameters);
        let two = [busy(1, 0), busy(2, 0)];
        harness.select(8192, &two);
        harness.select(65536, &two);
        harness.advance(Duration::from_secs(1));
        // The next request rebalances over engines 1 and 2 only: engine 1 takes the short
        // bucket, engine 2 the long one.
        assert_eq!(harness.select(8192, &two), WorkerWithDpRank::new(1, 0));
        // Engine 3 appears afterwards with no assignment, so it serves every bucket; with an
        // empty ledger it beats engine 1, and the long-bucket engine 2 stays excluded.
        let three = [busy(1, 0), (2, 0, 0, 0, 0), (3, 0, 0, 0, 0)];
        assert_eq!(harness.select(8192, &three), WorkerWithDpRank::new(3, 0));
    }
}
