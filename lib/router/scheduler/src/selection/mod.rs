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
mod types;

pub use core::{
    HostCache, HostEligibility, HostLoad, HostReplication, HostTelemetry, KvIndexSource, Selected,
    SelectionAdmission, SelectionCore, SelectionHost, SelectionOperation, SelectionOutcome,
    SelectionPartition, SelectionRun, SelectionScheduler, SessionBinding,
};
pub use dynamo_slot_tracker::mesh::{
    HostReplicaChannels, HostReplicaSyncFactory, ReplicaIngressObserver, ReplicaPeerError,
    SchedulerLoadSink,
};
pub use dynamo_worker_selection::plugin::WorkerSelectionPolicyFactory;
pub use error::SelectionError;
pub use ingress::KvEventIngress;
#[doc(hidden)]
pub use input::{MmRoutingInfoRequest, NormalizedPrompt, NormalizedReservation, TrackingHashInput};
pub use input::{PromptRequest, PromptView};
pub use membership::{CatalogObserver, CatalogReconciler, WorkerCatalogSource};
pub use pending::SelectionCacheConfig;
#[doc(hidden)]
pub use types::REQUEST_BODY_LIMIT_BYTES;
pub use types::{
    DEFAULT_MODEL_NAME, ModelLoadResponse, OutputBlockRequest, OverlapScoresRequest,
    OverlapScoresResponse, PotentialLoadsRequest, ReadyResponse, ReservationRequest,
    ReservationResponse, SelectAndReserveRequest, SelectRequest, SelectResponse,
    SelectionInputTrigger, SelectionSessionContext, SelectionWorkerConfig, SelectionWorkerLoad,
    SharedCacheOverlapScore, WorkerCatalogRecord, WorkerLifecycle, WorkerOverlapScore,
    WorkerPatchRequest, WorkerRequest,
};
