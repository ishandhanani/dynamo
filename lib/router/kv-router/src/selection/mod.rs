// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The selection service: the kernel's selection API composed into a service with
//! membership, ingress, replica sync, peer recovery, and the HTTP routes.

pub use dynamo_scheduler::selection::*;

#[cfg(feature = "http")]
mod http_error;
#[cfg(feature = "http")]
mod server;
mod service;
#[cfg(all(test, feature = "http"))]
mod tests;

#[cfg(feature = "http")]
pub use http_error::response as error_response;
#[cfg(feature = "http")]
pub use server::{AppState, run_server};
pub use service::{
    SelectionService, SelectionServiceBuilder, warn_for_unserved_worker_selection_policies,
};

// TODO(v1.7): Remove these compatibility re-exports; use crate::plugins instead.
pub use crate::plugins::{
    DYN_ROUTER_DECODE_POLICY, DYN_ROUTER_PREFILL_POLICY, DYN_ROUTER_WORKER_SELECTION_POLICY,
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProvider,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
