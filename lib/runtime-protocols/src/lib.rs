// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dependency-free vocabulary shared by Dynamo crates.
//!
//! This leaf crate holds the types every other Dynamo crate agrees on without
//! pulling in the full runtime: the [`error`] taxonomy, [`protocols`] for
//! annotated streams and endpoint identity, the [`engine`] traits, and the
//! canonical [`environment_names`]. `dynamo-runtime` re-exports each module at
//! its historical path, so existing `dynamo_runtime::{engine, error, protocols}`
//! and `dynamo_runtime::config::environment_names` imports keep working.

pub mod engine;
pub mod environment_names;
pub mod error;
pub mod protocols;
