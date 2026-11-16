// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The worker set an encode or prefill hop targets, and the id the hop keys its binding on.
//! Discovery chooses the target; the hops build their routing for it.

use std::sync::Arc;

use dynamo_model_card::model_card::ModelDeploymentCard;
use dynamo_runtime::{
    component::{Client, Endpoint},
    protocols::EndpointId,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

/// A topology hop must retain the provider's admission and configuration, even
/// when other cards share its endpoint or the endpoint is reused by a successor.
#[derive(Clone)]
pub(crate) struct CommittedWorkerSetTarget {
    pub(crate) endpoint: Endpoint,
    pub(crate) group: String,
    pub(crate) generation: u64,
    pub(crate) card: Arc<ModelDeploymentCard>,
    pub(crate) admitted_ids: watch::Receiver<Vec<u64>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum WorkerSetTargetId {
    Committed { group: String, generation: u64 },
    Legacy(EndpointId),
}

impl WorkerSetTargetId {
    pub(crate) fn matches_group(&self, group_id: &str, endpoint: &EndpointId) -> bool {
        match self {
            Self::Committed { group, .. } => group == group_id,
            Self::Legacy(legacy) => legacy == endpoint,
        }
    }
}

#[derive(Clone)]
pub(crate) enum WorkerSetTarget {
    Committed(CommittedWorkerSetTarget),
    /// Explicit endpoint constructors predate controller-owned admission.
    Legacy(Endpoint),
}

impl WorkerSetTarget {
    pub(crate) fn id(&self) -> WorkerSetTargetId {
        match self {
            Self::Committed(target) => WorkerSetTargetId::Committed {
                group: target.group.clone(),
                generation: target.generation,
            },
            Self::Legacy(endpoint) => WorkerSetTargetId::Legacy(endpoint.id()),
        }
    }

    pub(crate) fn endpoint(&self) -> &Endpoint {
        match self {
            Self::Committed(target) => &target.endpoint,
            Self::Legacy(endpoint) => endpoint,
        }
    }

    pub(crate) async fn client(&self, cancellation: CancellationToken) -> anyhow::Result<Client> {
        let client = self.endpoint().client().await?;
        Ok(match self {
            Self::Committed(target) => client.with_admitted_instances_and_cancellation(
                target.admitted_ids.clone(),
                cancellation,
            ),
            Self::Legacy(_) => client,
        })
    }
}
