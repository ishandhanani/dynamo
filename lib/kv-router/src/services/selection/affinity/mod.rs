// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session affinity: bind a session to the worker that served it so its
//! prefix cache is reused. [`SessionAffinity`] is the binding table and
//! [`AffinityResolver`] drives it for one request; `README.md` in this
//! directory maps the routing hosts onto both.

mod replication;
mod resolver;
mod table;

pub use replication::{AffinityBindingEvent, AffinityEventSink, ReplicaEventDisposition};
pub use resolver::{AffinityResolver, Resolution, TargetLiveness};
pub use table::*;
