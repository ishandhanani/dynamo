// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Composes a Dynamo router from the stage crates.
//!
//! The kernel ([`dynamo_scheduler`]) runs the stages atomically for one request. This crate
//! is what a host builds and holds: [`selection::SelectionService`] and its builder, the
//! plugin API and registry ([`plugins`]), the nested router configuration ([`config`]), and
//! the standalone selection service behind the `http` feature.

pub mod config;
pub mod plugins;
pub mod selection;

pub use selection::{SelectionService, SelectionServiceBuilder};
