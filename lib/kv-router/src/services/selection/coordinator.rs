// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Disaggregated coordination over two selection pools.
//!
//! The coordinator books a request on a decode pool and, unless the
//! conditional-disaggregation policy bypasses remote prefill, on a prefill pool
//! constrained to the decode worker's KV-transfer domain. The two pools keep
//! independent ledgers; the coordinator never holds a cross-pool lock and
//! instead compensates every partial state explicitly (see
//! [`LinkedBookingState`]). Decode-first greedy pairing is by design: joint
//! prefill/decode optimization is out of scope.
//!
//! The coordinator is transport-agnostic. It only needs the selection API of
//! each pool ([`SelectionPool`]), which the standalone service, an embedded
//! service, and the EPP can all provide, so one implementation serves every
//! host.

use std::sync::Arc;

use async_trait::async_trait;

use crate::conditional_disagg::{ConditionalDisaggDecisionInput, ConditionalDisaggPolicy};
use crate::config::{KvRouterConfig, RouterConfigOverride};
use crate::protocols::{KvTransferEnforcement, RoutingConstraints, WorkerId, WorkerWithDpRank};

use super::error::SelectionError;
use super::input::PromptRequest;
use super::service::SelectionService;
use super::types::{
    SelectAndReserveRequest, SelectRequest, SelectResponse, SelectionSessionContext,
    WorkerCatalogRecord,
};

/// Canonical worker-taint form for a topology domain/value pair, matching the
/// frontend's `dynamo.topology/<domain>=<value>` convention.
pub fn topology_taint(domain: &str, value: &str) -> String {
    format!("dynamo.topology/{domain}={value}")
}

/// One selection pool as the coordinator sees it.
#[async_trait]
pub trait SelectionPool: Send + Sync {
    async fn select(&self, req: SelectRequest) -> Result<SelectResponse, SelectionError>;
    async fn select_and_reserve(
        &self,
        req: SelectAndReserveRequest,
    ) -> Result<SelectResponse, SelectionError>;
    async fn free_reservation(&self, selection_id: &str) -> Result<(), SelectionError>;
    async fn prefill_complete(&self, selection_id: &str) -> Result<(), SelectionError>;
    /// Catalog record for `worker_id`, used to derive KV-transfer constraints.
    fn worker_record(&self, worker_id: WorkerId) -> Option<WorkerCatalogRecord>;
}

#[async_trait]
impl SelectionPool for SelectionService {
    async fn select(&self, req: SelectRequest) -> Result<SelectResponse, SelectionError> {
        SelectionService::select(self, req).await
    }

    async fn select_and_reserve(
        &self,
        req: SelectAndReserveRequest,
    ) -> Result<SelectResponse, SelectionError> {
        SelectionService::select_and_reserve(self, req).await
    }

    async fn free_reservation(&self, selection_id: &str) -> Result<(), SelectionError> {
        SelectionService::free_reservation(self, selection_id).await
    }

    async fn prefill_complete(&self, selection_id: &str) -> Result<(), SelectionError> {
        SelectionService::prefill_complete(self, selection_id).await
    }

    fn worker_record(&self, worker_id: WorkerId) -> Option<WorkerCatalogRecord> {
        self.list_workers(None, None)
            .into_iter()
            .find(|record| record.worker_id == worker_id)
    }
}

/// Which pool a booking lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pool {
    Prefill,
    Decode,
}

/// One live booking in one pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolBooking {
    pub pool: Pool,
    pub selection_id: String,
    pub worker: WorkerWithDpRank,
    pub endpoint: String,
}

/// The linked-booking state machine. Every transition the coordinator makes is
/// recorded so a host can audit compensation, and tests can assert the exact
/// path a request took.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkedBookingState {
    /// Nothing booked. Terminal when the decode commit failed.
    Idle,
    /// Decode previewed (query-only); nothing booked.
    DecodePreviewed,
    /// Decode booked; prefill not yet attempted.
    DecodeCommitted,
    /// Decode booked and the policy chose to run prefill on it.
    Bypass,
    /// Decode and prefill both booked.
    Linked,
    /// Prefill freed after it completed; decode still booked.
    PrefillReleased,
    /// Prefill booking failed after decode was booked; decode kept and the
    /// request runs prefill on it (compensation by bypass).
    BypassAfterPrefillFailure,
    /// Prefill booking failed after decode was booked; decode freed.
    Compensated,
    /// Every booking released.
    Released,
}

/// What the coordinator decided and holds for one request.
#[derive(Debug)]
pub struct DisaggPlan {
    pub decode: PoolBooking,
    /// `None` when prefill runs on the decode worker.
    pub prefill: Option<PoolBooking>,
    pub decision: BypassDecision,
    pub decode_signals: DecodeSignals,
    state: LinkedBookingState,
    transitions: Vec<LinkedBookingState>,
}

impl DisaggPlan {
    pub fn state(&self) -> LinkedBookingState {
        self.state
    }

    /// Every state this plan passed through, in order.
    pub fn transitions(&self) -> &[LinkedBookingState] {
        &self.transitions
    }

    pub fn is_bypass(&self) -> bool {
        self.prefill.is_none()
    }

    fn transition(&mut self, state: LinkedBookingState) {
        self.state = state;
        self.transitions.push(state);
    }
}

/// Signals read from the decode preview that feed the bypass decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeSignals {
    pub cached_tokens: usize,
    pub potential_decode_blocks: u64,
    pub decode_busy: Option<bool>,
}

/// The conditional-disaggregation decision and its inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BypassDecision {
    pub policy_says_bypass: bool,
    pub prefill_busy: Option<bool>,
    pub decode_busy: Option<bool>,
    pub bypass: bool,
}

/// Inputs for one coordinated request.
#[derive(Debug, Clone)]
pub struct DisaggRequest {
    pub model_name: String,
    pub prefill_routing_group: String,
    pub decode_routing_group: String,
    pub selection_id: String,
    pub prompt: PromptRequest,
    pub expected_output_tokens: Option<u32>,
    pub session_context: Option<SelectionSessionContext>,
    pub routing_constraints: RoutingConstraints,
    /// Override applied to the decode booking. The coordinator forces
    /// `track_prefill_tokens=false` and `assume_kv_reuse=false`; normal
    /// disaggregation also forces zero overlap credit, bypass keeps the base.
    pub decode_router_config_override: Option<RouterConfigOverride>,
    pub prefill_router_config_override: Option<RouterConfigOverride>,
}

#[derive(Debug, thiserror::Error)]
pub enum CoordinatorError {
    #[error("decode preview failed: {0}")]
    DecodePreview(#[source] SelectionError),
    #[error("decode booking failed: {0}")]
    DecodeCommit(#[source] SelectionError),
    /// Prefill booking failed; the decode booking was released.
    #[error("prefill booking failed after decode was released: {0}")]
    PrefillCommit(#[source] SelectionError),
    #[error(
        "worker {worker_id} in the decode pool has kv_transfer_domain {domain:?} but no matching topology domain"
    )]
    MissingTopologyDomain { worker_id: WorkerId, domain: String },
    #[error(
        "worker {worker_id} in the decode pool has kv_transfer_domain {domain:?} but no kv_transfer_enforcement"
    )]
    MissingEnforcement { worker_id: WorkerId, domain: String },
    #[error(
        "worker {worker_id} in the decode pool has preferred KV transfer enforcement but no kv_transfer_preferred_weight"
    )]
    MissingPreferredWeight { worker_id: WorkerId },
}

/// How the coordinator compensates a prefill booking failure once decode is
/// already booked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrefillFailurePolicy {
    /// Free the decode booking and fail the request.
    #[default]
    FreeDecode,
    /// Keep the decode booking and run prefill on the decode worker.
    BypassOnDecode,
}

pub struct DisaggCoordinator<P: SelectionPool = SelectionService> {
    prefill: Arc<P>,
    decode: Arc<P>,
    policy: Arc<dyn ConditionalDisaggPolicy>,
    prefill_busy_threshold: Option<f64>,
    decode_busy_threshold: Option<f64>,
    prefill_failure: PrefillFailurePolicy,
}

impl<P: SelectionPool> DisaggCoordinator<P> {
    pub fn new(
        prefill: Arc<P>,
        decode: Arc<P>,
        policy: Arc<dyn ConditionalDisaggPolicy>,
        config: &KvRouterConfig,
    ) -> Self {
        Self {
            prefill,
            decode,
            policy,
            prefill_busy_threshold: config.conditional_disagg_prefill_busy_threshold,
            decode_busy_threshold: config.conditional_disagg_decode_busy_threshold,
            prefill_failure: PrefillFailurePolicy::default(),
        }
    }

    pub fn with_prefill_failure_policy(mut self, policy: PrefillFailurePolicy) -> Self {
        self.prefill_failure = policy;
        self
    }

    /// Preview, decide, and commit bookings for one request.
    ///
    /// Order: decode preview (query-only, no booking) -> optional prefill busy
    /// read (advisory, no booking) -> policy -> decode commit pinned to the
    /// previewed worker -> prefill commit constrained to the decode worker's
    /// KV-transfer domain. Every failure after the decode commit is compensated
    /// per [`PrefillFailurePolicy`] before the error is returned.
    pub async fn plan(&self, request: DisaggRequest) -> Result<DisaggPlan, CoordinatorError> {
        let mut transitions = vec![LinkedBookingState::Idle];

        // 1. Decode-anchored preview. Query-only: the scheduler evaluates the
        //    candidate against current load without booking, and reports the
        //    decode busy line when a threshold is configured.
        let preview = self
            .decode
            .select(SelectRequest {
                model_name: request.model_name.clone(),
                routing_group: request.decode_routing_group.clone(),
                selection_id: None,
                prompt: request.prompt.clone(),
                router_config_override: Some(decode_override(
                    request.decode_router_config_override.clone(),
                    true,
                )),
                expected_output_tokens: request.expected_output_tokens,
                priority_jump: None,
                strict_priority: None,
                session_id: None,
                session_context: request.session_context.clone(),
                affinity_target: None,
                pinned_worker: None,
                allowed_worker_ids: None,
                routing_constraints: request.routing_constraints.clone(),
                advisory: true,
            })
            .await
            .map_err(CoordinatorError::DecodePreview)?;
        transitions.push(LinkedBookingState::DecodePreviewed);
        let prompt_tokens = request
            .prompt
            .token_ids
            .as_ref()
            .map(Vec::len)
            .or(request.prompt.isl_tokens)
            .unwrap_or(0);
        let decode_signals = DecodeSignals {
            cached_tokens: prompt_tokens.saturating_sub(preview.effective_prefill_tokens),
            potential_decode_blocks: preview.potential_decode_blocks,
            decode_busy: preview.decode_busy,
        };
        let previewed_worker = WorkerWithDpRank::new(preview.worker_id, preview.dp_rank);

        // 2. Prefill busy read, only when the policy consumes it.
        let prefill_busy = if self.policy.is_enabled() && self.policy.needs_prefill_worker_busy() {
            match self.prefill_busy(&request).await {
                Ok(busy) => busy,
                Err(error) => {
                    tracing::debug!(%error, "prefill busy probe failed; treating load as unavailable");
                    None
                }
            }
        } else {
            None
        };

        // 3. Decision.
        let input =
            ConditionalDisaggDecisionInput::new(prompt_tokens, decode_signals.cached_tokens)
                .with_prefill_chosen_worker_busy(prefill_busy);
        let policy_says_bypass =
            self.policy.is_enabled() && self.policy.should_bypass_remote_prefill(input).await;
        // The decode busy gate is evaluated after the policy, on the previewed
        // decode worker: a bypass only stands when that worker is not busy (or
        // no gate is configured).
        let decode_gate_configured = self.decode_busy_threshold.is_some();
        let decode_busy = policy_says_bypass
            .then_some(decode_signals.decode_busy)
            .flatten();
        let bypass = policy_says_bypass && (!decode_gate_configured || decode_busy == Some(false));
        let decision = BypassDecision {
            policy_says_bypass,
            prefill_busy,
            decode_busy,
            bypass,
        };

        // 4. Pinned decode commit.
        let decode = match self
            .decode
            .select_and_reserve(SelectAndReserveRequest {
                model_name: request.model_name.clone(),
                routing_group: request.decode_routing_group.clone(),
                selection_id: Some(decode_selection_id(&request.selection_id)),
                prompt: request.prompt.clone(),
                router_config_override: Some(decode_override(
                    request.decode_router_config_override.clone(),
                    bypass,
                )),
                expected_output_tokens: request.expected_output_tokens,
                priority_jump: None,
                strict_priority: None,
                session_id: None,
                session_context: request.session_context.clone(),
                affinity_target: None,
                pinned_worker: Some(previewed_worker),
                allowed_worker_ids: None,
                routing_constraints: request.routing_constraints.clone(),
            })
            .await
        {
            Ok(response) => response,
            Err(error) => {
                transitions.push(LinkedBookingState::Idle);
                return Err(CoordinatorError::DecodeCommit(error));
            }
        };
        let decode = PoolBooking {
            pool: Pool::Decode,
            selection_id: decode
                .selection_id
                .clone()
                .expect("booked selection has an id"),
            worker: WorkerWithDpRank::new(decode.worker_id, decode.dp_rank),
            endpoint: decode.endpoint,
        };
        transitions.push(LinkedBookingState::DecodeCommitted);

        let mut plan = DisaggPlan {
            decode,
            prefill: None,
            decision,
            decode_signals,
            state: LinkedBookingState::DecodeCommitted,
            transitions,
        };
        if bypass {
            plan.transition(LinkedBookingState::Bypass);
            return Ok(plan);
        }

        // 5. Prefill commit constrained to the decode worker's KV-transfer domain.
        let prefill_request = match self.prefill_request(&request, &plan.decode) {
            Ok(prefill_request) => prefill_request,
            Err(error) => {
                self.compensate_prefill_failure(&mut plan).await;
                return match plan.state {
                    LinkedBookingState::BypassAfterPrefillFailure => Ok(plan),
                    _ => Err(error),
                };
            }
        };
        match self.prefill.select_and_reserve(prefill_request).await {
            Ok(response) => {
                plan.prefill = Some(PoolBooking {
                    pool: Pool::Prefill,
                    selection_id: response
                        .selection_id
                        .clone()
                        .expect("booked selection has an id"),
                    worker: WorkerWithDpRank::new(response.worker_id, response.dp_rank),
                    endpoint: response.endpoint,
                });
                plan.transition(LinkedBookingState::Linked);
                Ok(plan)
            }
            Err(error) => {
                self.compensate_prefill_failure(&mut plan).await;
                match plan.state {
                    LinkedBookingState::BypassAfterPrefillFailure => Ok(plan),
                    _ => Err(CoordinatorError::PrefillCommit(error)),
                }
            }
        }
    }

    /// Prefill finished (first decode token or handoff observed): release the
    /// prefill booking early so the prefill pool sees its capacity back.
    pub async fn prefill_complete(&self, plan: &mut DisaggPlan) -> Result<(), SelectionError> {
        let Some(prefill) = plan.prefill.take() else {
            return Ok(());
        };
        let result = self.prefill.free_reservation(&prefill.selection_id).await;
        plan.transition(LinkedBookingState::PrefillReleased);
        result
    }

    /// Release every booking the plan still holds (request finished or failed).
    pub async fn release(&self, plan: &mut DisaggPlan) {
        if let Some(prefill) = plan.prefill.take()
            && let Err(error) = self.prefill.free_reservation(&prefill.selection_id).await
        {
            tracing::debug!(%error, selection_id = %prefill.selection_id, "prefill booking already released");
        }
        if let Err(error) = self
            .decode
            .free_reservation(&plan.decode.selection_id)
            .await
        {
            tracing::debug!(%error, selection_id = %plan.decode.selection_id, "decode booking already released");
        }
        plan.transition(LinkedBookingState::Released);
    }

    async fn compensate_prefill_failure(&self, plan: &mut DisaggPlan) {
        match self.prefill_failure {
            PrefillFailurePolicy::BypassOnDecode => {
                plan.transition(LinkedBookingState::BypassAfterPrefillFailure);
            }
            PrefillFailurePolicy::FreeDecode => {
                if let Err(error) = self
                    .decode
                    .free_reservation(&plan.decode.selection_id)
                    .await
                {
                    tracing::warn!(
                        %error,
                        selection_id = %plan.decode.selection_id,
                        "failed to free decode booking while compensating a prefill failure"
                    );
                }
                plan.transition(LinkedBookingState::Compensated);
            }
        }
    }

    async fn prefill_busy(&self, request: &DisaggRequest) -> Result<Option<bool>, SelectionError> {
        let Some(threshold) = self.prefill_busy_threshold else {
            return Ok(None);
        };
        let response = self
            .prefill
            .select(SelectRequest {
                model_name: request.model_name.clone(),
                routing_group: request.prefill_routing_group.clone(),
                selection_id: None,
                prompt: request.prompt.clone(),
                router_config_override: request.prefill_router_config_override.clone(),
                expected_output_tokens: request.expected_output_tokens,
                priority_jump: None,
                strict_priority: None,
                session_id: None,
                session_context: request.session_context.clone(),
                affinity_target: None,
                pinned_worker: None,
                allowed_worker_ids: None,
                routing_constraints: request.routing_constraints.clone(),
                advisory: true,
            })
            .await?;
        // The service evaluates the busy line with its own threshold; when the
        // coordinator carries one and the service did not, derive it here.
        Ok(response.worker_load.map(|load| {
            load.prefill_busy.unwrap_or(
                load.active_prefill_tokens as f64 > threshold * load.prefill_token_capacity as f64,
            )
        }))
    }

    fn prefill_request(
        &self,
        request: &DisaggRequest,
        decode: &PoolBooking,
    ) -> Result<SelectAndReserveRequest, CoordinatorError> {
        let mut routing_constraints = request.routing_constraints.clone();
        if let Some(record) = self.decode.worker_record(decode.worker.worker_id) {
            merge_kv_transfer_constraints(&mut routing_constraints, &record)?;
        }
        Ok(SelectAndReserveRequest {
            model_name: request.model_name.clone(),
            routing_group: request.prefill_routing_group.clone(),
            selection_id: Some(prefill_selection_id(&request.selection_id)),
            prompt: request.prompt.clone(),
            router_config_override: request.prefill_router_config_override.clone(),
            expected_output_tokens: Some(1),
            priority_jump: None,
            strict_priority: None,
            session_id: None,
            session_context: request.session_context.clone(),
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints,
        })
    }
}

pub fn decode_selection_id(selection_id: &str) -> String {
    format!("{selection_id}/decode")
}

pub fn prefill_selection_id(selection_id: &str) -> String {
    format!("{selection_id}/prefill")
}

/// Decode never accounts prompt-side load. Normal disaggregation also forces
/// zero overlap credit so decode routing stays load-only; bypass keeps the base
/// credit because prefill runs on the chosen decode worker.
fn decode_override(
    existing: Option<RouterConfigOverride>,
    allow_decode_overlap_affinity: bool,
) -> RouterConfigOverride {
    let mut override_config = existing.unwrap_or_default();
    if !allow_decode_overlap_affinity {
        override_config.overlap_score_credit = Some(0.0);
    }
    override_config.assume_kv_reuse = Some(false);
    override_config.track_prefill_tokens = Some(false);
    override_config
}

/// Constrain prefill selection to workers in the decode worker's KV-transfer
/// domain, with the enforcement the decode worker advertises.
fn merge_kv_transfer_constraints(
    constraints: &mut RoutingConstraints,
    decode_record: &WorkerCatalogRecord,
) -> Result<(), CoordinatorError> {
    let Some(domain) = decode_record.kv_transfer_domain.as_deref() else {
        return Ok(());
    };
    let worker_id = decode_record.worker_id;
    let Some(value) = decode_record.topology_domains.get(domain) else {
        return Err(CoordinatorError::MissingTopologyDomain {
            worker_id,
            domain: domain.to_string(),
        });
    };
    let taint = topology_taint(domain, value);
    match decode_record.kv_transfer_enforcement {
        Some(KvTransferEnforcement::Required) => {
            constraints.required_taints.insert(taint);
        }
        Some(KvTransferEnforcement::Preferred) => {
            let Some(weight) = decode_record.kv_transfer_preferred_weight else {
                return Err(CoordinatorError::MissingPreferredWeight { worker_id });
            };
            constraints.preferred_taints.insert(taint, weight);
        }
        None => {
            return Err(CoordinatorError::MissingEnforcement {
                worker_id,
                domain: domain.to_string(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::*;
    use crate::conditional_disagg::{IslBoundingPolicy, RandomBypassConditionalDisaggPolicy};
    use crate::services::selection::types::WorkerRequest;

    const MODEL: &str = "model";
    const PREFILL: &str = "prefill";
    const DECODE: &str = "decode";

    fn config() -> KvRouterConfig {
        KvRouterConfig {
            use_kv_events: false,
            router_queue_threshold: None,
            ..Default::default()
        }
    }

    fn pool() -> Arc<SelectionService> {
        Arc::new(SelectionService::new_local_for_test(config(), 1))
    }

    fn worker(worker_id: WorkerId, routing_group: &str) -> WorkerRequest {
        WorkerRequest {
            worker_id,
            model_name: MODEL.to_string(),
            routing_group: routing_group.to_string(),
            endpoint: Some(format!("http://worker-{worker_id}:8000")),
            block_size: Some(4),
            max_num_batched_tokens: Some(1024),
            total_kv_blocks: Some(1000),
            ..WorkerRequest::default()
        }
    }

    async fn pools_with(
        prefill_workers: Vec<WorkerRequest>,
        decode_workers: Vec<WorkerRequest>,
    ) -> (Arc<SelectionService>, Arc<SelectionService>) {
        let prefill = pool();
        let decode = pool();
        for request in prefill_workers {
            prefill
                .upsert_worker(request)
                .await
                .expect("prefill worker");
        }
        for request in decode_workers {
            decode.upsert_worker(request).await.expect("decode worker");
        }
        (prefill, decode)
    }

    fn request(selection_id: &str) -> DisaggRequest {
        DisaggRequest {
            model_name: MODEL.to_string(),
            prefill_routing_group: PREFILL.to_string(),
            decode_routing_group: DECODE.to_string(),
            selection_id: selection_id.to_string(),
            prompt: PromptRequest {
                token_ids: Some((1..=8).collect()),
                ..PromptRequest::default()
            },
            expected_output_tokens: Some(16),
            session_context: None,
            routing_constraints: RoutingConstraints::default(),
            decode_router_config_override: None,
            prefill_router_config_override: None,
        }
    }

    fn active_requests(service: &SelectionService, routing_group: &str) -> usize {
        service
            .loads(Some(MODEL), Some(routing_group))
            .into_iter()
            .flat_map(|response| response.loads)
            .map(|load| load.active_requests)
            .sum()
    }

    fn disabled_policy() -> Arc<dyn ConditionalDisaggPolicy> {
        Arc::new(IslBoundingPolicy::disabled())
    }

    fn always_bypass() -> Arc<dyn ConditionalDisaggPolicy> {
        Arc::new(RandomBypassConditionalDisaggPolicy::new(true, 1.0))
    }

    #[tokio::test]
    async fn linked_plan_books_both_pools_and_releases_in_order() {
        let (prefill, decode) = pools_with(vec![worker(1, PREFILL)], vec![worker(2, DECODE)]).await;
        let coordinator = DisaggCoordinator::new(
            Arc::clone(&prefill),
            Arc::clone(&decode),
            disabled_policy(),
            &config(),
        );

        let mut plan = coordinator.plan(request("req")).await.expect("linked plan");
        assert_eq!(plan.state(), LinkedBookingState::Linked);
        assert!(!plan.is_bypass());
        assert_eq!(plan.decode.worker.worker_id, 2);
        assert_eq!(plan.decode.selection_id, "req/decode");
        let prefill_booking = plan.prefill.as_ref().expect("prefill booking");
        assert_eq!(prefill_booking.worker.worker_id, 1);
        assert_eq!(prefill_booking.selection_id, "req/prefill");
        assert_eq!(active_requests(&prefill, PREFILL), 1);
        assert_eq!(active_requests(&decode, DECODE), 1);
        assert!(!plan.decision.bypass);

        coordinator
            .prefill_complete(&mut plan)
            .await
            .expect("prefill freed");
        assert_eq!(plan.state(), LinkedBookingState::PrefillReleased);
        assert_eq!(active_requests(&prefill, PREFILL), 0);
        assert_eq!(active_requests(&decode, DECODE), 1);
        // Idempotent: a second completion is a no-op.
        coordinator
            .prefill_complete(&mut plan)
            .await
            .expect("no-op");

        coordinator.release(&mut plan).await;
        assert_eq!(plan.state(), LinkedBookingState::Released);
        assert_eq!(active_requests(&decode, DECODE), 0);
        assert_eq!(
            plan.transitions(),
            &[
                LinkedBookingState::Idle,
                LinkedBookingState::DecodePreviewed,
                LinkedBookingState::DecodeCommitted,
                LinkedBookingState::Linked,
                LinkedBookingState::PrefillReleased,
                LinkedBookingState::Released,
            ]
        );
    }

    #[tokio::test]
    async fn policy_bypass_books_decode_only() {
        let (prefill, decode) = pools_with(vec![worker(1, PREFILL)], vec![worker(2, DECODE)]).await;
        let coordinator = DisaggCoordinator::new(
            Arc::clone(&prefill),
            Arc::clone(&decode),
            always_bypass(),
            &config(),
        );
        let mut plan = coordinator.plan(request("req")).await.expect("bypass plan");
        assert_eq!(plan.state(), LinkedBookingState::Bypass);
        assert!(plan.is_bypass());
        assert!(plan.decision.policy_says_bypass && plan.decision.bypass);
        assert_eq!(plan.decision.decode_busy, None, "no decode gate configured");
        assert_eq!(active_requests(&prefill, PREFILL), 0);
        assert_eq!(active_requests(&decode, DECODE), 1);
        coordinator.release(&mut plan).await;
        assert_eq!(active_requests(&decode, DECODE), 0);
    }

    #[tokio::test]
    async fn decode_busy_gate_vetoes_a_policy_bypass() {
        let (prefill, _) = pools_with(vec![worker(1, PREFILL)], Vec::new()).await;
        let mut gated = config();
        gated.conditional_disagg_decode_busy_threshold = Some(0.0);
        let decode = Arc::new(SelectionService::new_local_for_test(gated.clone(), 1));
        decode
            .upsert_worker(worker(2, DECODE))
            .await
            .expect("decode worker");
        let coordinator = DisaggCoordinator::new(
            Arc::clone(&prefill),
            Arc::clone(&decode),
            always_bypass(),
            &gated,
        );
        let mut plan = coordinator.plan(request("req")).await.expect("linked plan");
        assert!(plan.decision.policy_says_bypass);
        assert_eq!(plan.decision.decode_busy, Some(true));
        assert!(!plan.decision.bypass);
        assert_eq!(plan.state(), LinkedBookingState::Linked);
        assert_eq!(active_requests(&prefill, PREFILL), 1);
        coordinator.release(&mut plan).await;
        assert_eq!(active_requests(&prefill, PREFILL), 0);
        assert_eq!(active_requests(&decode, DECODE), 0);
    }

    #[tokio::test]
    async fn prefill_failure_frees_decode_by_default() {
        // Prefill pool has no schedulable worker: booking fails after decode
        // was committed. Default compensation frees decode and fails the plan.
        let (prefill, decode) = pools_with(Vec::new(), vec![worker(2, DECODE)]).await;
        let coordinator = DisaggCoordinator::new(
            Arc::clone(&prefill),
            Arc::clone(&decode),
            disabled_policy(),
            &config(),
        );
        let error = coordinator
            .plan(request("req"))
            .await
            .expect_err("prefill fails");
        assert!(
            matches!(error, CoordinatorError::PrefillCommit(_)),
            "{error}"
        );
        assert_eq!(
            active_requests(&decode, DECODE),
            0,
            "decode must be compensated"
        );
    }

    #[tokio::test]
    async fn prefill_failure_can_bypass_on_the_decode_worker() {
        let (prefill, decode) = pools_with(Vec::new(), vec![worker(2, DECODE)]).await;
        let coordinator = DisaggCoordinator::new(
            Arc::clone(&prefill),
            Arc::clone(&decode),
            disabled_policy(),
            &config(),
        )
        .with_prefill_failure_policy(PrefillFailurePolicy::BypassOnDecode);
        let mut plan = coordinator
            .plan(request("req"))
            .await
            .expect("bypass on decode");
        assert_eq!(plan.state(), LinkedBookingState::BypassAfterPrefillFailure);
        assert!(plan.is_bypass());
        assert_eq!(active_requests(&decode, DECODE), 1);
        coordinator.release(&mut plan).await;
        assert_eq!(active_requests(&decode, DECODE), 0);
    }

    #[tokio::test]
    async fn decode_failure_books_nothing() {
        let (prefill, decode) = pools_with(vec![worker(1, PREFILL)], Vec::new()).await;
        let coordinator = DisaggCoordinator::new(
            Arc::clone(&prefill),
            Arc::clone(&decode),
            disabled_policy(),
            &config(),
        );
        let error = coordinator
            .plan(request("req"))
            .await
            .expect_err("decode fails");
        assert!(
            matches!(error, CoordinatorError::DecodePreview(_)),
            "{error}"
        );
        assert_eq!(active_requests(&prefill, PREFILL), 0);
    }

    fn zoned(mut request: WorkerRequest, zone: &str) -> WorkerRequest {
        request.topology_domains = HashMap::from([("zone".to_string(), zone.to_string())]);
        request.taints = HashSet::from([topology_taint("zone", zone)]);
        request
    }

    #[tokio::test]
    async fn prefill_is_constrained_to_the_decode_kv_transfer_domain() {
        let mut decode_worker = zoned(worker(2, DECODE), "a");
        decode_worker.kv_transfer_domain = Some("zone".to_string());
        decode_worker.kv_transfer_enforcement = Some(KvTransferEnforcement::Required);
        let (prefill, decode) = pools_with(
            vec![
                zoned(worker(10, PREFILL), "b"),
                zoned(worker(11, PREFILL), "a"),
            ],
            vec![decode_worker],
        )
        .await;
        let coordinator = DisaggCoordinator::new(
            Arc::clone(&prefill),
            Arc::clone(&decode),
            disabled_policy(),
            &config(),
        );
        for index in 0..6 {
            let mut plan = coordinator
                .plan(request(&format!("req-{index}")))
                .await
                .expect("linked plan");
            assert_eq!(
                plan.prefill
                    .as_ref()
                    .map(|booking| booking.worker.worker_id),
                Some(11),
                "prefill must land in the decode worker's zone"
            );
            coordinator.release(&mut plan).await;
        }
    }

    #[tokio::test]
    async fn incomplete_kv_transfer_metadata_is_an_error_and_frees_decode() {
        let mut decode_worker = zoned(worker(2, DECODE), "a");
        decode_worker.kv_transfer_domain = Some("zone".to_string());
        // No enforcement: the record cannot be turned into a constraint.
        let (prefill, decode) =
            pools_with(vec![zoned(worker(11, PREFILL), "a")], vec![decode_worker]).await;
        let coordinator = DisaggCoordinator::new(
            Arc::clone(&prefill),
            Arc::clone(&decode),
            disabled_policy(),
            &config(),
        );
        let error = coordinator
            .plan(request("req"))
            .await
            .expect_err("metadata error");
        assert!(
            matches!(
                error,
                CoordinatorError::MissingEnforcement { worker_id: 2, .. }
            ),
            "{error}"
        );
        assert_eq!(active_requests(&decode, DECODE), 0);
        assert_eq!(active_requests(&prefill, PREFILL), 0);
    }
}
