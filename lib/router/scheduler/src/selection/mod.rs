// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-free worker selection service.
//!
//! The service owns worker selection and reservation state, but never forwards
//! model requests and never owns model responses.

pub use dynamo_worker_selection::affinity;
mod catalog;
mod core;
mod error;
mod ingress;
mod input;
mod membership;
mod pending;
#[cfg(feature = "http")]
mod server;
mod service;
mod types;

#[cfg(all(test, feature = "http"))]
mod tests;

// TODO(v1.7): Remove these compatibility re-exports; use crate::plugins instead.
pub use crate::plugins::{
    DYN_ROUTER_DECODE_POLICY, DYN_ROUTER_PREFILL_POLICY, DYN_ROUTER_WORKER_SELECTION_POLICY,
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProvider,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
pub use core::{
    HostCache, HostEligibility, HostLoad, HostReplication, HostTelemetry, KvIndexSource, Selected,
    SelectionAdmission, SelectionCore, SelectionHost, SelectionOperation, SelectionOutcome,
    SelectionPartition, SelectionRun, SelectionScheduler, SelectionServiceConfig, SessionBinding,
};
pub use dynamo_slot_tracker::mesh::{
    HostReplicaChannels, HostReplicaSyncFactory, ReplicaIngressObserver, ReplicaPeerError,
    SchedulerLoadSink,
};
pub use dynamo_worker_selection::plugin::WorkerSelectionPolicyFactory;
pub use error::SelectionError;
pub use ingress::KvEventIngress;
pub use input::{PromptRequest, PromptView};
pub use membership::{CatalogObserver, CatalogReconciler, WorkerCatalogSource};
pub use pending::SelectionCacheConfig;
#[cfg(feature = "http")]
pub use server::{AppState, run_server};
pub use service::{
    SelectionService, SelectionServiceBuilder, warn_for_unserved_worker_selection_policies,
};
pub use types::{
    DEFAULT_MODEL_NAME, ModelLoadResponse, OutputBlockRequest, OverlapScoresRequest,
    OverlapScoresResponse, PotentialLoadsRequest, ReadyResponse, ReservationRequest,
    ReservationResponse, SelectAndReserveRequest, SelectRequest, SelectResponse,
    SelectionInputTrigger, SelectionSessionContext, SelectionWorkerConfig, SelectionWorkerLoad,
    SharedCacheOverlapScore, WorkerCatalogRecord, WorkerLifecycle, WorkerOverlapScore,
    WorkerPatchRequest, WorkerRequest,
};
