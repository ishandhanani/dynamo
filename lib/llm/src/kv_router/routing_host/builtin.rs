// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The first-party routing modes hosted directly by [`RoutingHost`]: the
//! runtime's stateless route pickers over the host's own worker set and
//! occupancy, plus the Direct and device-aware paths.

use dynamo_runtime::pipeline::{OverloadCheck, RouterMode, TransportFallback};

use super::*;

/// The picker for a first-party mode; `None` for the modes that select
/// elsewhere (KV, Direct, DeviceAwareWeighted).
pub(super) fn builtin_picker(mode: RouterMode) -> Option<BuiltinRoutePicker> {
    Some(match mode {
        RouterMode::RoundRobin => BuiltinRoutePicker::round_robin(),
        RouterMode::Random => BuiltinRoutePicker::random(),
        RouterMode::PowerOfTwoChoices => BuiltinRoutePicker::power_of_two_choices(),
        RouterMode::LeastLoaded => BuiltinRoutePicker::least_loaded(),
        RouterMode::KV | RouterMode::Direct | RouterMode::DeviceAwareWeighted => return None,
    })
}

/// What a first-party mode reads per worker: occupancy for the load-aware
/// pickers, nothing for the stateless ones.
pub(super) fn builtin_inputs(mode: RouterMode) -> WorkerInputs {
    if mode.requires_occupancy() {
        WorkerInputs::OCCUPANCY
    } else {
        WorkerInputs::NONE
    }
}

/// A first-party selection awaiting dispatch.
pub(super) struct HostedPlan {
    selection: HostedSelection,
    /// The LoRA replica set a transport fallback may move within.
    lora_fallback: Option<HashSet<u64>>,
    lora_load: Option<LoraLoadGuard>,
    /// Direct routing for a request without a session: the explicit target is
    /// sent with Direct's standalone transport fallback.
    direct_unbound: bool,
}

/// How a hosted send treats a selected worker that has left discovery.
#[derive(Clone, Copy)]
enum Departed {
    /// Direct and the stateless modes: an error only when no permitted
    /// fallback is free.
    OrFallback,
    /// The occupancy modes: the preselected worker must still be discovered;
    /// fallback covers only a departure after that check.
    Reject,
}

impl RoutingHost {
    /// Resolve the worker the request really goes to, let `prepare` rewrite
    /// the request for it, then send. A failed `prepare` is recorded as the
    /// route's outcome.
    async fn resolve_prepare_send<M>(
        &self,
        mut request: SingleIn<PreprocessedRequest>,
        worker_id: u64,
        fallback: TransportFallback<'_>,
        departed: Departed,
        prepare: impl FnOnce(&mut PreprocessedRequest, u64) -> Result<M, Error>,
    ) -> Result<(M, ManyOut<Annotated<LLMEngineOutput>>), Error> {
        match departed {
            Departed::OrFallback => self.inner.ensure_live_or_fallback(worker_id, fallback)?,
            Departed::Reject => self.inner.ensure_live_for_dispatch(worker_id)?,
        }
        let resolved =
            self.inner
                .resolve(&request, worker_id, fallback, OverloadCheck::Required)?;
        let worker_id = resolved.worker_id();
        let metadata = match prepare(&mut request, worker_id) {
            Ok(metadata) => metadata,
            Err(error) => {
                resolved.fail(error.as_ref());
                return Err(error);
            }
        };
        let stream = self.inner.send(request, resolved).await?;
        if matches!(departed, Departed::OrFallback) {
            tracing::info!(
                router_mode = self.inner.router_mode().telemetry_label(),
                worker_id,
                "Selected worker"
            );
        }
        Ok((metadata, stream))
    }
}

impl RoutingHost {
    fn select_lora_target(
        &self,
        request: &PreprocessedRequest,
    ) -> Result<Option<LoraSelection>, Error> {
        let Some(lora) = self.lora.as_ref() else {
            return Ok(None);
        };
        let Some(lora_name) = request
            .routing
            .as_ref()
            .and_then(|routing| routing.lora_name.clone())
        else {
            return Ok(None);
        };
        let load_guard = LoraLoadGuard::new(Arc::clone(&lora.load_estimator), lora_name.clone());
        let routable = self.inner.client.instance_ids_avail();
        let candidates = lora
            .filter
            .filter_worker_ids_for_lora(Some(&lora_name), &routable);
        if candidates.is_empty() {
            anyhow::bail!("No workers available after LoRA filtering (lora={lora_name})");
        }

        let free = self
            .inner
            .client
            .instance_ids_free()
            .into_iter()
            .collect::<HashSet<_>>();
        let candidates = candidates
            .into_iter()
            .filter(|worker_id| free.contains(worker_id))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Err(anyhow::anyhow!(
                DynamoError::builder()
                    .error_type(ErrorType::ResourceExhausted)
                    .message(format!(
                        "All eligible LoRA workers are overloaded (lora={lora_name})"
                    ))
                    .build()
            ));
        }
        let target = lora
            .selector
            .select_worker(&candidates, |_| 0)
            .ok_or(KvSchedulerError::NoEndpoints)?;
        tracing::debug!(
            lora = %lora_name,
            worker_id = target,
            candidates = candidates.len(),
            routable = routable.len(),
            free = free.len(),
            "LoRA-filtered router selected worker"
        );
        Ok(Some(LoraSelection {
            target,
            allowed_fallback: candidates.into_iter().collect(),
            load_guard,
        }))
    }

    pub(super) fn select_hosted_worker(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        target_constraint: Option<AffinityTarget>,
        affinity_target: Option<AffinityTarget>,
    ) -> Result<HostedSelection, Error> {
        let preferred_worker = || match affinity_target {
            Some(target) => self.inner.with_selectable_worker_ids(|ids| {
                ids.binary_search(&target.worker_id)
                    .is_ok()
                    .then_some(target.worker_id)
            }),
            None => Ok(None),
        };
        match &self.policy {
            RoutingPolicy::Kv(_) => unreachable!("hosted selection called for KV routing"),
            RoutingPolicy::Direct => {
                let target = target_constraint.ok_or_else(|| {
                    invalid_argument("Direct routing requires an exact affinity or request target")
                })?;
                Ok(HostedSelection {
                    initial_worker: target.worker_id,
                    target_constraint: Some(target),
                    occupancy_reservation: None,
                    candidate_count: 1,
                    selected_occupancy: None,
                    device_aware_telemetry: None,
                })
            }
            RoutingPolicy::Builtin(picker) => {
                let preferred_worker = preferred_worker()?;
                if self.inner.router_mode().requires_occupancy() {
                    let occupancy = self
                        .hosted_occupancy
                        .as_ref()
                        .expect("OCCUPANCY policy must have hosted occupancy state");
                    let selection = occupancy.select_and_reserve(
                        &self.inner,
                        picker,
                        target_constraint
                            .map(|target| target.worker_id)
                            .or(preferred_worker),
                    )?;
                    Ok(HostedSelection {
                        initial_worker: selection.worker_id,
                        target_constraint,
                        occupancy_reservation: Some(selection.reservation),
                        candidate_count: selection.candidate_count,
                        selected_occupancy: Some(selection.occupancy),
                        device_aware_telemetry: None,
                    })
                } else {
                    let worker_id = match target_constraint {
                        Some(target) => {
                            self.inner.ensure_routable(target.worker_id)?;
                            target.worker_id
                        }
                        None => self.inner.with_selectable_worker_ids(|ids| {
                            if let Some(worker_id) = preferred_worker {
                                Ok(worker_id)
                            } else {
                                picker
                                    .select_worker(ids, |_| 0)
                                    .ok_or(KvSchedulerError::NoEndpoints)
                            }
                        })??,
                    };
                    Ok(HostedSelection {
                        initial_worker: worker_id,
                        target_constraint,
                        occupancy_reservation: None,
                        candidate_count: 0,
                        selected_occupancy: None,
                        device_aware_telemetry: None,
                    })
                }
            }
            RoutingPolicy::DeviceAwareWeighted => {
                let preferred_worker = preferred_worker()?;
                let selection = self.inner.select_device_aware_and_reserve(
                    request.content(),
                    target_constraint
                        .map(|target| target.worker_id)
                        .or(preferred_worker),
                )?;
                let initial_worker = selection.worker_id();
                let candidate_count = selection.candidate_count();
                let selected_occupancy = Some(selection.load());
                let device_aware_telemetry = Some(DeviceAwareTelemetry {
                    is_cpu: selection.is_cpu(),
                    embedding_cache_hit: selection.embedding_cache_hit(),
                    request_cache_keys: selection.request_cache_keys(),
                });
                Ok(HostedSelection {
                    initial_worker,
                    target_constraint,
                    occupancy_reservation: selection.into_reservation(),
                    candidate_count,
                    selected_occupancy,
                    device_aware_telemetry,
                })
            }
        }
    }

    /// Select a first-party worker for `request` in `phase` and hold what the
    /// send must honor: the exact target, the occupancy reservation, the LoRA
    /// fallback set, and the session hold.
    pub(super) async fn plan_hosted_route(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        budget: CleanupBudget,
    ) -> Result<RoutePlan, Error> {
        let phase_label = phase.to_string();
        let _route_guard = StageGuard::new(STAGE_ROUTE, &phase_label);
        let explicit = explicit_target(request.content(), phase)?;
        let has_affinity_session = self.affinity.is_some() && affinity_id(request)?.is_some();
        let is_direct = matches!(&self.policy, RoutingPolicy::Direct);
        if is_direct && explicit.is_none() && !has_affinity_session {
            return Err(invalid_argument(format!(
                "worker ID required for {phase} request in Direct routing mode"
            )));
        }
        let is_query_only = request.get_annotation_value("query_instance_id").is_some();
        let (lora_target, lora_fallback, lora_load) =
            match self.select_lora_target(request.content())? {
                Some(selection) => (
                    Some(selection.target),
                    Some(selection.allowed_fallback),
                    Some(selection.load_guard),
                ),
                None => (None, None, None),
            };
        let (selection, affinity) = if let Some(target) = lora_target {
            (
                HostedSelection {
                    initial_worker: target,
                    target_constraint: None,
                    occupancy_reservation: None,
                    candidate_count: 0,
                    selected_occupancy: None,
                    device_aware_telemetry: None,
                },
                None,
            )
        } else {
            self.validate_explicit_worker(request.content(), phase)?;
            self.select_with_session_affinity(request, phase, is_query_only, &budget, |target| {
                let pinned_target = explicit.or(match self.session_affinity_mode {
                    SessionAffinityMode::Hard => target,
                    SessionAffinityMode::Soft if is_direct => target,
                    SessionAffinityMode::Soft => None,
                });
                let affinity_target = match (explicit, self.session_affinity_mode) {
                    (None, SessionAffinityMode::Soft) => target,
                    _ => None,
                };
                ready(self.select_hosted_worker(request, pinned_target, affinity_target))
            })
            .await?
        };
        Ok(RoutePlan {
            phase,
            selection: PlanSelection::Hosted(HostedPlan {
                selection,
                lora_fallback,
                lora_load,
                direct_unbound: is_direct && !has_affinity_session,
            }),
            affinity,
            budget,
        })
    }

    /// Send a first-party plan. An exact target is prepared for and sent
    /// without fallback; every other selection resolves its transport first,
    /// moves the occupancy slot if that changed the worker, and prepares for
    /// the worker the request really goes to.
    pub(super) async fn dispatch_hosted_plan<M, F>(
        &self,
        mut request: SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        plan: HostedPlan,
        affinity: &mut Option<Hold>,
        budget: &CleanupBudget,
        prepare: F,
    ) -> Result<(M, ManyOut<Annotated<LLMEngineOutput>>), Error>
    where
        F: FnOnce(&mut PreprocessedRequest, AffinityTarget) -> Result<M, Error>,
    {
        let phase_label = phase.to_string();
        let staged_kv = StagedKv::for_request(request.content());
        let HostedPlan {
            selection:
                HostedSelection {
                    initial_worker,
                    target_constraint,
                    occupancy_reservation,
                    candidate_count,
                    selected_occupancy,
                    device_aware_telemetry,
                },
            lora_fallback,
            lora_load,
            direct_unbound,
        } = plan;
        let soft_affinity_target = if self.session_affinity_mode == SessionAffinityMode::Soft {
            affinity.as_ref().and_then(Hold::target).map(from_table)
        } else {
            None
        };
        let target_for_worker = |worker_id| {
            AffinityTarget::new(
                worker_id,
                soft_affinity_target.and_then(|target| target.dp_rank),
            )
        };
        let uses_occupancy = self
            .required_worker_inputs()
            .contains(WorkerInputs::OCCUPANCY);
        let mut guard: RequestGuard = RequestGuard::new_builtin(
            self.request_metrics.clone(),
            initial_worker,
            occupancy_reservation,
            lora_load,
            &request,
        );
        let tracker = request.tracker.clone();
        let request_context = request.context().clone();
        self.request_metrics
            .input_sequence_tokens
            .with_label_values(&[request.phase().as_str(), &request.model])
            .observe(request.token_ids.len() as f64);

        guard.begin_dispatch(&phase_label, request.content());
        let dispatch_result = if let Some(target) = target_constraint.filter(|_| !direct_unbound) {
            // An exact target: prepare for it, then send without fallback.
            request.routing_mut().dp_rank = target.dp_rank;
            let metadata = match prepare(&mut request, target) {
                Ok(metadata) => metadata,
                Err(error) => {
                    guard.abort().await;
                    return Err(error);
                }
            };
            await_with_cleanup_policy(
                request_context.as_ref(),
                phase,
                staged_kv,
                "builtin.dispatch_exact",
                budget,
                self.inner.dispatch_exact(request, target.worker_id),
            )
            .await
            .and_then(|result| result)
            .map(|stream| (metadata, target, selected_occupancy, stream))
        } else {
            let (worker_id, fallback, departed, stage) = if direct_unbound {
                let target = target_constraint.expect("Direct routing requires an explicit target");
                (
                    target.worker_id,
                    TransportFallback::Allow,
                    Departed::OrFallback,
                    "builtin.dispatch_direct",
                )
            } else if uses_occupancy {
                (
                    initial_worker,
                    TransportFallback::Allow,
                    Departed::Reject,
                    "builtin.dispatch_occupancy",
                )
            } else {
                (
                    initial_worker,
                    lora_fallback
                        .as_ref()
                        .map_or(TransportFallback::Allow, TransportFallback::Within),
                    Departed::OrFallback,
                    "builtin.dispatch",
                )
            };
            let explicit = target_constraint.filter(|_| direct_unbound);
            await_with_cleanup_policy(
                request_context.as_ref(),
                phase,
                staged_kv,
                stage,
                budget,
                self.resolve_prepare_send(
                    request,
                    worker_id,
                    fallback,
                    departed,
                    |request, worker_id| {
                        let occupancy = guard.retarget_worker(worker_id);
                        let target = match explicit {
                            // Direct keeps the explicit rank only for the worker it named.
                            Some(explicit) => AffinityTarget::new(
                                worker_id,
                                explicit.dp_rank.filter(|_| worker_id == explicit.worker_id),
                            ),
                            None => target_for_worker(worker_id),
                        };
                        request.routing_mut().dp_rank = target.dp_rank;
                        prepare(request, target).map(|metadata| (metadata, target, occupancy))
                    },
                ),
            )
            .await
            .and_then(|result| result)
            .map(|((metadata, target, occupancy), stream)| (metadata, target, occupancy, stream))
        };

        let (metadata, target, final_occupancy, response_stream) = match dispatch_result {
            Ok(result) => result,
            Err(error) => {
                let expected_target =
                    target_constraint.unwrap_or_else(|| target_for_worker(initial_worker));
                self.fail_dispatch(&mut guard, affinity, expected_target, &error)
                    .await;
                return Err(error);
            }
        };
        guard.retarget_worker(target.worker_id);
        if let Some(telemetry) = device_aware_telemetry {
            let selection_survived_transport = target.worker_id == initial_worker;
            tracing::info!(
                router_mode = "device-aware-weighted",
                worker_id = target.worker_id,
                candidate_count,
                occupancy = ?final_occupancy,
                endpoint = %self.inner.client.endpoint.id(),
                is_cpu = ?selection_survived_transport.then_some(telemetry.is_cpu),
                embedding_cache_hit = ?selection_survived_transport
                    .then_some(telemetry.embedding_cache_hit),
                request_cache_keys = telemetry.request_cache_keys,
                transport_fallback = !selection_survived_transport,
                "Selected worker"
            );
        } else if uses_occupancy {
            tracing::info!(
                router_mode = self.inner.router_mode().telemetry_label(),
                worker_id = target.worker_id,
                candidate_count,
                occupancy = ?final_occupancy,
                transport_fallback = target.worker_id != initial_worker,
                "Selected worker"
            );
        }
        if let Some(tracker) = tracker {
            let worker_type = if tracker.phase() == RequestPhase::Prefill {
                WORKER_TYPE_PREFILL
            } else {
                WORKER_TYPE_DECODE
            };
            tracker.record_worker(target.worker_id, target.dp_rank, worker_type);
        }
        let stream = self
            .complete_dispatch(guard, affinity, target, response_stream)
            .await?;
        Ok((metadata, stream))
    }
}
