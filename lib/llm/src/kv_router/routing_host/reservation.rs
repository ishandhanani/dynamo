// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_kv_router::scheduling::queue::BookingHandle;
use dynamo_runtime::pipeline::RouteTarget;
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, SingleIn};
use tokio_util::sync::CancellationToken;

use super::{
    RoutingHost,
    cleanup::RequestCleanup,
    kv_selection::{RoutingRequestParts, SelectionOptions},
};
use crate::{
    kv_router::{FindBestMatchAdmission, KvRouter},
    protocols::common::{preprocessor::PreprocessedRequest, timing::RequestPhase},
};

impl RoutingHost {
    /// Select using the same policy and state as the token pipeline. The caller
    /// owns response transport and releases this reservation when its request ends.
    pub(crate) async fn reserve_route(
        self: &Arc<Self>,
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
            return Ok(RouteReservation::new(router.clone(), booking));
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
        let worker =
            dynamo_runtime::pipeline::RouteTarget::new(selection.initial_worker, requested_rank);
        Ok(RouteReservation::hosted(
            worker,
            selection.occupancy_reservation,
            self.clone(),
        ))
    }
}

/// Host-owned admission for a caller that supplies its own response transport.
/// Uses the same policy cleanup as the canonical token path.
pub(crate) struct RouteReservation {
    target: RouteTarget,
    cleanup: RequestCleanup,
    _host: Option<Arc<RoutingHost>>,
}

pub(crate) enum RouteGuard {
    Held { _reservation: RouteReservation },
    Renewed { _done: tokio_util::sync::DropGuard },
}

impl RouteReservation {
    pub(crate) fn new(router: Arc<KvRouter>, booking: BookingHandle) -> Self {
        let (cleanup, worker) = RequestCleanup::unobserved_kv(router, booking);
        Self {
            target: RouteTarget::new(worker.worker_id, Some(worker.dp_rank)),
            cleanup,
            _host: None,
        }
    }

    fn hosted(
        target: RouteTarget,
        occupancy: Option<dynamo_runtime::pipeline::OccupancyReservation>,
        host: Arc<RoutingHost>,
    ) -> Self {
        Self {
            target,
            cleanup: RequestCleanup::Occupancy {
                worker_id: target.worker_id,
                reservation: occupancy,
            },
            _host: Some(host),
        }
    }

    pub(crate) fn target(&self) -> RouteTarget {
        self.target
    }

    pub(crate) fn touch(&self) -> anyhow::Result<()> {
        if let Some(lease) = self.cleanup.lifecycle() {
            anyhow::ensure!(
                lease.is_active(),
                "routing reservation expired while request was active"
            );
            lease.touch();
        }
        Ok(())
    }

    /// Keep this stage admitted while another stage waits for admission.
    pub(crate) async fn while_live<T>(
        &self,
        operation: impl std::future::Future<Output = anyhow::Result<T>>,
    ) -> anyhow::Result<T> {
        if self.cleanup.lifecycle().is_none() {
            return operation.await;
        }
        let mut heartbeat = renewal_interval();
        tokio::pin!(operation);
        loop {
            tokio::select! {
                result = &mut operation => return result,
                _ = heartbeat.tick() => self.touch()?,
            }
        }
    }

    /// Renew until the returned guard drops. Expired admission cancels dispatch.
    pub(crate) fn start(mut self, cancellation: CancellationToken) -> RouteGuard {
        if self.cleanup.lifecycle().is_none() {
            return RouteGuard::Held { _reservation: self };
        }
        let done = CancellationToken::new();
        let completed = done.clone();
        tokio::spawn(async move {
            let mut heartbeat = renewal_interval();
            loop {
                tokio::select! {
                    biased;
                    _ = completed.cancelled() => break,
                    _ = cancellation.cancelled() => break,
                    _ = heartbeat.tick() => {
                        if self.touch().is_err() { cancellation.cancel(); break; }
                    }
                }
            }
            self.cleanup.finish().await;
        });
        RouteGuard::Renewed {
            _done: done.drop_guard(),
        }
    }
}

fn renewal_interval() -> tokio::time::Interval {
    let expiry = dynamo_kv_router::multi_worker_sequence::active_request_expiry_duration();
    let mut interval = tokio::time::interval((expiry / 3).min(std::time::Duration::from_secs(5)));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    interval
}
