// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Dynamo router scheduler: the kernel (`local`, `queue`, `overlap`, `overlap_refresh`) that
//! runs gate, eligibility, candidates, policy, and booking under one serializer; the plugin
//! registry; `SelectionCore`, `SelectionService`, and `SelectionServiceBuilder` that compose the
//! index, the slot tracker, admission, and worker selection into one service (`selection`); the
//! selection HTTP routes behind `http`; and the nested router configuration (`config`).

pub mod config;
pub mod constraints;
pub mod local;
pub mod overlap;
pub mod overlap_refresh;
pub mod queue;
pub mod selection;
pub mod topology;

#[cfg(any(test, feature = "testing", feature = "bench"))]
pub mod test_utils;

#[doc(hidden)]
pub mod scheduling_compat {
    //! The old `scheduling::` names the moved code still reaches by bare path.
    pub use crate::local::LocalScheduler;
    pub use crate::overlap::*;
    pub use crate::overlap_refresh::*;
    pub use dynamo_admission::classifier::*;
    pub use dynamo_admission::policy_config::*;
    pub use dynamo_admission::policy_queue::*;
    pub use dynamo_admission::runtime::{RequestClassifierRuntime, RequestLifecycle};
    pub use dynamo_router_protocol::config;
    pub use dynamo_router_protocol::eligibility::*;
    pub use dynamo_router_protocol::prefill_load::*;
    pub use dynamo_router_protocol::protocols::{PotentialLoad, WorkerPlacement};
    pub use dynamo_router_protocol::sched::*;
    pub use dynamo_worker_selection::lora::*;
    pub use dynamo_worker_selection::plugin::{WorkerSelectionConfig, WorkerSelectionInstance};
    pub use dynamo_worker_selection::selector;
}

pub use local::LocalScheduler;
pub use queue::SchedulerQueue;
pub use selection::SelectionCore;
