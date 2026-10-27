// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Admission for the Dynamo router: the per-class policy queue (DRR over policy classes with
//! busy gating), the request-classifier plugin API, and the classifier runtime that drives a
//! plugin's lifecycle events.

pub mod classifier;
pub mod policy_queue;
pub mod runtime;

pub mod config {
    //! The admission slice of the router configuration: policy classes and classifier settings.
    pub use dynamo_router_types::config::{policy, request_classifier};
}
