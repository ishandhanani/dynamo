// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod occupancy;

pub(crate) use dynamo_worker_selection::builtin::route::{
    AdmissionKind, CandidateView, RouteCandidate, RouteContext, RouteDecision, RouteDevice,
    RoutePicker, RoutePolicy,
};
pub use dynamo_worker_selection::builtin::route::{BuiltinRoutePicker, RouteTarget};
pub(crate) use occupancy::get_or_create_routing_occupancy_state;
pub use occupancy::{OccupancyReservation, OccupancySelection, RoutingOccupancyState};
