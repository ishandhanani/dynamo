// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Production HTTP server mechanics shared by Dynamo's frontends and services.
//!
//! This crate owns the transport: binding a listener, rebinding one whose socket died, the
//! accept loop with per-connection write-stall and transfer deadlines, TLS termination, graceful shutdown bounded by a
//! grace period, the readiness/draining lifecycle with inflight accounting, the
//! request-level metrics every frontend reports, and the client-disconnect monitor that
//! cancels engine work for a vanished client. It knows
//! nothing about routes, protocols, discovery, or the distributed runtime; the composition
//! root (such as `dynamo-llm`'s `HttpService`) builds the axum router and hands it here.

pub mod disconnect;
pub mod lifecycle;
pub mod listener;
pub mod metrics;
pub mod server;
pub mod write_timeout;
