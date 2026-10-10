// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request metadata available to worker-selection components.

use super::SessionContext;
use crate::protocols::{WorkerAffinityTarget, WorkerWithDpRank};
use crate::scheduling::SchedulingRequest;

/// Request-level values available to custom filters, scorers, and pickers.
pub struct WorkerSelectionContext<'a> {
    pub(crate) request: &'a SchedulingRequest,
    pub(crate) request_blocks: u64,
    pub(crate) block_size: u32,
    pub(crate) track_prefill_tokens: bool,
    pub(crate) pinned_worker: Option<WorkerWithDpRank>,
    pub(crate) router_temperature_override: Option<f64>,
}

impl WorkerSelectionContext<'_> {
    /// The exact worker/rank the request pinned explicitly, if any. This is
    /// read-only routing metadata, not permission to change eligibility. A
    /// `Hard` session binding is not reported here; the host limits the
    /// candidate set to it instead.
    pub fn pinned_worker(&self) -> Option<WorkerWithDpRank> {
        self.pinned_worker
    }

    /// Exact incoming prompt length in tokens. Borrowed from this request; no rounding,
    /// cache weighting, or additional storage is involved. Zero when the request carries
    /// no prompt token IDs, such as embeddings-only input.
    pub fn prompt_tokens(&self) -> usize {
        self.request.isl_tokens
    }

    /// Return the incoming prompt size in KV blocks. Zero when
    /// [`prompt_tokens`](Self::prompt_tokens) is zero.
    pub fn request_blocks(&self) -> u64 {
        self.request_blocks
    }

    /// Return the number of tokens in one KV block.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Return whether this request contributes to prefill-load tracking.
    pub fn tracks_prefill_tokens(&self) -> bool {
        self.track_prefill_tokens
    }

    /// Return the session metadata available to worker selection.
    pub fn session_context(&self) -> Option<&SessionContext> {
        self.request.session_context.as_ref()
    }

    /// Return the session-affinity target resolved by the request host.
    ///
    /// Under `Hard` session affinity the host limits every policy's candidates to this target
    /// while it is eligible, so the candidate set already reflects it. Under `Soft` affinity it is
    /// a preference: the policy receives the full eligible candidate set and may select another
    /// worker. The target may be absent from the candidate set when it is unavailable or filtered.
    pub fn affinity_target(&self) -> Option<WorkerAffinityTarget> {
        self.request.affinity.map(|affinity| affinity.target)
    }

    /// Return the expected output length, if the request supplies one.
    pub fn expected_output_tokens(&self) -> Option<u32> {
        self.request.expected_output_tokens
    }

    /// Return the request's scheduler priority boost.
    pub fn priority_jump(&self) -> f64 {
        self.request.priority_jump
    }

    /// Return the request's strict integer priority.
    pub fn strict_priority(&self) -> u32 {
        self.request.strict_priority
    }

    /// Return the request-level router temperature override, if present.
    pub fn router_temperature_override(&self) -> Option<f64> {
        self.router_temperature_override
    }
}
