// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The runtime's stateless route pickers: round-robin, random, power-of-two-choices, least-loaded,
//! and device-aware weighted. `dynamo-runtime` re-exports them for its push router.

mod picker;

pub use picker::{BuiltinRoutePicker, RoutePicker};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RouteTarget {
    pub worker_id: u64,
    pub dp_rank: Option<u32>,
}

impl RouteTarget {
    pub const fn worker(worker_id: u64) -> Self {
        Self {
            worker_id,
            dp_rank: None,
        }
    }

    pub const fn new(worker_id: u64, dp_rank: Option<u32>) -> Self {
        Self { worker_id, dp_rank }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutePolicy {
    RoundRobin,
    Random,
    PowerOfTwoChoices,
    LeastLoaded,
    DeviceAwareWeighted,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RouteDevice {
    Cpu,
    #[default]
    Accelerator,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteCandidate {
    pub target: RouteTarget,
    pub device: RouteDevice,
    pub cache_hits: usize,
}

impl RouteCandidate {
    pub const fn worker(worker_id: u64) -> Self {
        Self {
            target: RouteTarget::worker(worker_id),
            device: RouteDevice::Accelerator,
            cache_hits: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum CandidateView<'a> {
    Workers(&'a [u64]),
    DeviceAware(&'a [RouteCandidate]),
}

impl CandidateView<'_> {
    #[inline(always)]
    pub fn len(&self) -> usize {
        match self {
            Self::Workers(workers) => workers.len(),
            Self::DeviceAware(candidates) => candidates.len(),
        }
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline(always)]
    pub fn target(&self, index: usize) -> RouteTarget {
        match self {
            Self::Workers(workers) => RouteTarget::worker(workers[index]),
            Self::DeviceAware(candidates) => candidates[index].target,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RouteContext {
    pub required_cache_hits: usize,
    pub non_cpu_to_cpu_ratio: usize,
}

impl Default for RouteContext {
    fn default() -> Self {
        Self {
            required_cache_hits: 0,
            non_cpu_to_cpu_ratio: 8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionKind {
    None,
    Occupancy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteDecision {
    pub target: RouteTarget,
    pub admission: AdmissionKind,
}
