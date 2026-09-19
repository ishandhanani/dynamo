// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, SingleIn};

use super::{
    RoutingHost,
    kv_selection::{RoutingRequestParts, SelectionOptions},
};
use crate::{
    kv_router::{FindBestMatchAdmission, native::NativeReservation},
    protocols::common::{preprocessor::PreprocessedRequest, timing::RequestPhase},
    session_affinity::AffinityTarget,
};

impl RoutingHost {
    /// Select using the same policy and state as the token pipeline. The caller
    /// owns native response transport and releases this booking via engine control.
    pub(crate) async fn reserve_native(
        self: &Arc<Self>,
        request: &SingleIn<PreprocessedRequest>,
        pinned: Option<AffinityTarget>,
        requested_rank: Option<u32>,
        phase: RequestPhase,
    ) -> anyhow::Result<NativeReservation> {
        if let Some(router) = self.kv_router_if_enabled() {
            let planned_worker = pinned
                .map(|target| {
                    Ok::<_, anyhow::Error>(WorkerWithDpRank::new(
                        target.worker_id,
                        target
                            .dp_rank
                            .ok_or_else(|| anyhow::anyhow!("KV admission requires a DP rank"))?,
                    ))
                })
                .transpose()?;
            let admitted: std::collections::HashSet<_> =
                self.inner.selectable_worker_ids()?.into_iter().collect();
            anyhow::ensure!(
                !admitted.is_empty(),
                "native WorkerSet has no available workers"
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
                        planned_worker,
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
                .ok_or_else(|| anyhow::anyhow!("native routing did not reserve work"))?;
            return Ok(NativeReservation::new(router.clone(), booking));
        }
        anyhow::ensure!(
            self.lora.is_none(),
            "native HTTP requires explicit LoRA worker routing"
        );
        let selection = self.select_hosted_worker(request, pinned, None)?;
        let worker = AffinityTarget::new(
            selection.initial_worker,
            pinned.and_then(|target| target.dp_rank).or(requested_rank),
        );
        Ok(NativeReservation::hosted(
            worker,
            selection.occupancy_reservation,
            self.clone(),
        ))
    }
}
