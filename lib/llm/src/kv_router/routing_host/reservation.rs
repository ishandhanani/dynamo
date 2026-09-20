// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::pipeline::{AsyncEngineContextProvider, SingleIn};
use dynamo_runtime::pipeline::{OccupancyReservation, RouteTarget};
use tokio_util::sync::CancellationToken;

use super::{
    RoutingHost,
    kv_selection::{RoutingRequestParts, SelectionOptions},
};
use crate::{
    kv_router::{FindBestMatchAdmission, request_lease::RequestAttemptLease},
    protocols::common::{preprocessor::PreprocessedRequest, timing::RequestPhase},
};

impl RoutingHost {
    /// Select using the same policy and state as the token pipeline. The caller
    /// owns response transport and releases this reservation when its request ends.
    pub(crate) async fn reserve_route(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
    ) -> anyhow::Result<RouteReservation> {
        if let Some(router) = self.kv_router_if_enabled() {
            let admitted: std::collections::HashSet<_> =
                self.inner.selectable_worker_ids()?.into_iter().collect();
            anyhow::ensure!(
                !admitted.is_empty(),
                "routed WorkerSet has no available workers"
            );
            let mut routing_request = request.content().clone();
            let routing = routing_request.routing.get_or_insert_default();
            routing.allowed_worker_ids = Some(match routing.allowed_worker_ids.take() {
                Some(allowed) => allowed.intersection(&admitted).copied().collect(),
                None => admitted,
            });
            let selection = self
                .select_worker_outcome(
                    request.context().id(),
                    &routing_request,
                    RoutingRequestParts::new(&routing_request),
                    phase,
                    false,
                    SelectionOptions {
                        pinned_target: None,
                        affinity_target: None,
                        planned_worker: None,
                        policy_class: request.metadata().get("policy-class").cloned(),
                        session_context: None,
                        admission: FindBestMatchAdmission::WithAdmission,
                    },
                )
                .await?
                .into_result()?;
            self.inner.ensure_routable(selection.worker.worker_id)?;
            let booking = selection
                .booking
                .ok_or_else(|| anyhow::anyhow!("routing did not reserve work"))?;
            let lease = router
                .request_lease_manager()
                .register_local(booking.commit(), None);
            return Ok(RouteReservation::new(
                RouteTarget::new(selection.worker.worker_id, Some(selection.worker.dp_rank)),
                Some(lease),
                None,
            ));
        }
        anyhow::ensure!(
            self.lora.is_none(),
            "external dispatch requires explicit LoRA worker routing"
        );
        let selection = self.select_hosted_worker(request, None, None)?;
        let requested_rank = request
            .content()
            .routing
            .as_ref()
            .and_then(|hints| hints.routing_constraints.as_ref())
            .and_then(|constraints| constraints.required_dp_rank);
        Ok(RouteReservation::new(
            RouteTarget::new(selection.initial_worker, requested_rank),
            None,
            selection.occupancy_reservation,
        ))
    }
}

/// Owns an admitted route until the caller finishes or drops its transport.
pub(crate) struct RouteReservation {
    pub(crate) target: RouteTarget,
    pub(crate) cancel: CancellationToken,
    _done: tokio_util::sync::DropGuard,
    _occupancy: Option<OccupancyReservation>,
}

impl RouteReservation {
    fn new(
        target: RouteTarget,
        lease: Option<RequestAttemptLease>,
        occupancy: Option<OccupancyReservation>,
    ) -> Self {
        let cancel = CancellationToken::new();
        let done = cancel.clone().drop_guard();
        if let Some(lease) = lease {
            let cancelled = cancel.clone();
            tokio::spawn(async move {
                let expiry =
                    dynamo_kv_router::multi_worker_sequence::active_request_expiry_duration();
                let mut heartbeat =
                    tokio::time::interval((expiry / 3).min(std::time::Duration::from_secs(5)));
                heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    tokio::select! {
                        biased;
                        _ = cancelled.cancelled() => break,
                        _ = heartbeat.tick() => {
                            if !lease.is_active() { cancelled.cancel(); break; }
                            lease.touch();
                        }
                    }
                }
                lease.finish().await;
            });
        }
        Self {
            target,
            cancel,
            _done: done,
            _occupancy: occupancy,
        }
    }
}
