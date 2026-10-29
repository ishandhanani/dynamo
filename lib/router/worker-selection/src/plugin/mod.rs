// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public filter, scorer, and picker contracts and their input signals.

mod context;
mod runtime;

pub mod inputs;

#[doc(hidden)]
pub use runtime::RuntimePolicyState;
pub use runtime::{
    RuntimePolicyBuilder, RuntimePolicyConfig, RuntimePolicyError, RuntimePolicyUpdate,
};

pub use context::WorkerSelectionContext;
#[doc(hidden)]
pub use inputs::{CacheSnapshot, CandidateData, WorkerCacheData, WorkerInputRows};
pub use inputs::{
    ScoredWorkerCandidate, WorkerCacheInput, WorkerCacheInputs, WorkerCandidate, WorkerCandidates,
    WorkerCapacity, WorkerInfo, WorkerInputView, WorkerInputs, WorkerLoadInput,
    WorkerTransferInput,
};

pub use crate::config::SelectionConfig;
pub use crate::selector::WorkerSelectionPolicy;
pub use dynamo_router_types::config::worker_selection::{
    WorkerSelectionConfig, WorkerSelectionInstance,
};
pub use dynamo_router_types::sched::{
    SessionContext, WorkerSelectionInputTrigger, WorkerSelectionPolicyError,
};

use dynamo_router_types::WorkerType;
use dynamo_router_types::identity::RoutingPartitionRef;
use std::sync::Arc;

/// Factory that creates one worker-selection policy per routing partition.
pub type WorkerSelectionPolicyFactory = Arc<
    dyn for<'a> Fn(&SelectionConfig, WorkerType, RoutingPartitionRef<'a>) -> WorkerSelectionPolicy
        + Send
        + Sync,
>;

/// Adds one finite cost contribution to each eligible worker.
pub trait WorkerScorer: Send {
    /// Declare the worker-signal groups needed by this scorer.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::NONE
    }

    /// Score every candidate that survived host eligibility and policy filters in one call.
    /// The borrowed snapshot and its row order remain fixed for this selection. Each scorer
    /// can access only its own declared optional inputs, even when other components request more. Empty candidate
    /// sets skip scoring. Scorers run in declaration order; an error stops before picking.
    ///
    /// `costs` has the same length as `candidates` and is initialized to NaN before each call.
    /// Write one finite, lower-is-better contribution to every slot. The host validates and adds
    /// each contribution to that row's total. The candidate view and cost slice may not be retained after this call.
    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError>;
}

/// Filters run in declaration order for each candidate. Callback order across different
/// candidates and scorers is unspecified; implementations must not depend on filters and scorers
/// being interleaved.
pub trait WorkerFilter: Send {
    /// Declare the worker-signal groups needed by this filter.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::NONE
    }

    /// Return `true` to keep an eligible worker in the policy candidate set.
    fn keep(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidate: WorkerCandidate<'_>,
    ) -> Result<bool, WorkerSelectionPolicyError>;
}

/// Selects one row after all filters and scorers run.
pub trait WorkerPicker: Send {
    /// Declare the optional worker-signal columns needed by this picker.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::NONE
    }

    /// Return one row index from the host-owned eligible candidate table. Row order is
    /// unspecified; inspect candidate data instead of relying on a stable position.
    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError>;
}
