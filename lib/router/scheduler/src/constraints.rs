// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The worker set one request may be placed on.
//!
//! A host derives one [`Constraints`] per request from the request's routing hints, its
//! migration history, and its session binding, then hands it to selection. The pure
//! narrowing rules live here; anything that needs the worker table (taint compatibility,
//! rank resolution, eligibility) stays with the host or the eligibility stage.

use std::collections::HashSet;

use dynamo_router_types::config::RouterConfigOverride;
use dynamo_router_types::protocols::{
    DpRank, RoutingConstraints, WorkerAffinityTarget, WorkerId, WorkerWithDpRank,
};

/// What narrows the candidate set for one request, in one place.
#[derive(Clone, Debug, Default)]
pub struct Constraints {
    /// The exact worker the request must run on, if any. Its rank may still need
    /// resolving against the worker's data-parallel range.
    pub pin: Option<WorkerAffinityTarget>,
    /// A worker the policy should prefer without being bound to it.
    pub prefer: Option<WorkerAffinityTarget>,
    /// Workers the caller allows; `None` leaves every worker eligible.
    pub allow: Option<HashSet<WorkerId>>,
    /// Workers a retry must avoid because an earlier attempt failed on them.
    pub exclude: Vec<WorkerId>,
    /// Taint tolerations the worker must satisfy.
    pub taints: RoutingConstraints,
    /// The LoRA adapter the worker must serve.
    pub lora: Option<String>,
    /// Per-request policy parameter override.
    pub policy_override: Option<RouterConfigOverride>,
}

/// A constraint the request cannot satisfy as stated.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConstraintError {
    #[error(
        "Pinned worker {worker_id} requires an explicit dp_rank because it has multiple or unknown DP ranks"
    )]
    UnresolvedRank { worker_id: WorkerId },
    #[error(
        "Previewed worker {} dp_rank {} conflicts with requested worker {} dp_rank {:?}",
        planned.worker_id,
        planned.dp_rank,
        requested.worker_id,
        requested.dp_rank
    )]
    PlannedWorkerConflict {
        planned: WorkerWithDpRank,
        requested: WorkerAffinityTarget,
    },
}

impl Constraints {
    /// Drop the excluded workers, and every worker `compatible` rejects, from the allowed
    /// set, materializing it from `all` when the caller allowed everyone. Returns whether a
    /// worker is left.
    pub fn exclude_from(
        &mut self,
        all: impl IntoIterator<Item = WorkerId>,
        mut compatible: impl FnMut(WorkerId, &RoutingConstraints) -> bool,
    ) -> bool {
        let Self {
            allow,
            exclude,
            taints,
            ..
        } = self;
        let allow = allow.get_or_insert_with(|| all.into_iter().collect());
        allow.retain(|worker_id| compatible(*worker_id, taints) && !exclude.contains(worker_id));
        !allow.is_empty()
    }

    /// Bind a hard session-affinity target.
    ///
    /// A target with a rank becomes the pin; an explicit request pin for the same worker
    /// without a rank takes the target's rank, and one with its own rank keeps it. A
    /// worker-only target narrows the allowed set to that worker and leaves the rank to
    /// the scheduler, so a matching worker-only request pin is dropped rather than turned
    /// into an exact-rank pin.
    pub fn bind_hard(&mut self, target: WorkerAffinityTarget) {
        let Some(dp_rank) = target.dp_rank else {
            match &mut self.allow {
                Some(allowed) => allowed.retain(|worker_id| *worker_id == target.worker_id),
                None => self.allow = Some(HashSet::from([target.worker_id])),
            }
            if self
                .pin
                .is_some_and(|pin| pin.dp_rank.is_none() && pin.worker_id == target.worker_id)
            {
                self.pin = None;
            }
            return;
        };
        self.pin = Some(match self.pin {
            Some(pin) if pin.dp_rank.is_none() && pin.worker_id == target.worker_id => {
                WorkerAffinityTarget::new(pin.worker_id, Some(dp_rank))
            }
            Some(pin) => pin,
            None => target,
        });
    }

    /// Bind a soft session-affinity target: advisory to the policy, never binding.
    pub fn bind_soft(&mut self, target: WorkerAffinityTarget) {
        self.prefer = Some(target);
    }

    /// Check that the worker a route preview chose agrees with the pin the request asks
    /// for, so the plan admits the worker the preview reported.
    pub fn check_planned(&self, planned: WorkerWithDpRank) -> Result<(), ConstraintError> {
        match self.pin {
            Some(requested)
                if requested.worker_id != planned.worker_id
                    || requested
                        .dp_rank
                        .is_some_and(|dp_rank| dp_rank != planned.dp_rank) =>
            {
                Err(ConstraintError::PlannedWorkerConflict { planned, requested })
            }
            _ => Ok(()),
        }
    }
}

/// Resolve a pin's data-parallel rank: the requested rank, else the worker's only rank.
pub fn resolve_pinned_worker_rank(
    worker_id: WorkerId,
    requested_dp_rank: Option<DpRank>,
    unique_dp_rank: Option<DpRank>,
) -> Result<WorkerWithDpRank, ConstraintError> {
    let Some(dp_rank) = requested_dp_rank.or(unique_dp_rank) else {
        return Err(ConstraintError::UnresolvedRank { worker_id });
    };
    Ok(WorkerWithDpRank::new(worker_id, dp_rank))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pinned(worker_id: WorkerId, dp_rank: Option<DpRank>) -> Constraints {
        Constraints {
            pin: Some(WorkerAffinityTarget::new(worker_id, dp_rank)),
            ..Default::default()
        }
    }

    #[test]
    fn resolve_pinned_worker_rank_uses_explicit_rank_including_zero() {
        let worker = resolve_pinned_worker_rank(7, Some(0), Some(3)).unwrap();
        assert_eq!(worker, WorkerWithDpRank::new(7, 0));
    }

    #[test]
    fn resolve_pinned_worker_rank_uses_unique_rank_when_unset() {
        let worker = resolve_pinned_worker_rank(7, None, Some(3)).unwrap();
        assert_eq!(worker, WorkerWithDpRank::new(7, 3));
    }

    #[test]
    fn resolve_pinned_worker_rank_rejects_unresolved_rank() {
        let error = resolve_pinned_worker_rank(7, None, None).unwrap_err();
        assert_eq!(error, ConstraintError::UnresolvedRank { worker_id: 7 });
        assert!(error.to_string().contains("requires an explicit dp_rank"));
    }

    #[test]
    fn hard_target_supplies_rank_for_matching_explicit_worker() {
        let mut constraints = pinned(7, None);
        constraints.bind_hard(WorkerAffinityTarget::new(7, Some(0)));
        assert_eq!(constraints.pin, Some(WorkerAffinityTarget::new(7, Some(0))));

        let mut constraints = pinned(7, Some(2));
        constraints.bind_hard(WorkerAffinityTarget::new(7, Some(3)));
        assert_eq!(constraints.pin, Some(WorkerAffinityTarget::new(7, Some(2))));

        let mut constraints = Constraints::default();
        constraints.bind_hard(WorkerAffinityTarget::new(7, Some(3)));
        assert_eq!(constraints.pin, Some(WorkerAffinityTarget::new(7, Some(3))));
        assert!(constraints.allow.is_none());
    }

    #[test]
    fn worker_only_hard_target_narrows_allowed_workers_and_leaves_rank_open() {
        let mut constraints = pinned(7, None);
        constraints.allow = Some(HashSet::from([3, 7, 9]));
        constraints.bind_hard(WorkerAffinityTarget::new(7, None));
        assert_eq!(constraints.pin, None);
        assert_eq!(constraints.allow, Some(HashSet::from([7])));

        let mut constraints = pinned(8, None);
        constraints.bind_hard(WorkerAffinityTarget::new(7, None));
        assert_eq!(constraints.pin, Some(WorkerAffinityTarget::new(8, None)));
        assert_eq!(constraints.allow, Some(HashSet::from([7])));
    }

    #[test]
    fn planned_worker_must_agree_with_the_pin() {
        assert!(
            Constraints::default()
                .check_planned(WorkerWithDpRank::new(7, 1))
                .is_ok()
        );
        assert!(
            pinned(7, None)
                .check_planned(WorkerWithDpRank::new(7, 1))
                .is_ok()
        );
        assert!(
            pinned(7, Some(1))
                .check_planned(WorkerWithDpRank::new(7, 1))
                .is_ok()
        );
        let error = pinned(7, Some(2))
            .check_planned(WorkerWithDpRank::new(7, 1))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("conflicts with requested worker")
        );
        assert!(
            pinned(8, None)
                .check_planned(WorkerWithDpRank::new(7, 1))
                .is_err()
        );
    }

    #[test]
    fn excluding_workers_materializes_the_allowed_set_once() {
        let mut constraints = Constraints {
            exclude: vec![2],
            ..Default::default()
        };
        assert!(constraints.exclude_from([1, 2, 3], |worker_id, _| worker_id != 3));
        assert_eq!(constraints.allow, Some(HashSet::from([1])));

        let mut constraints = Constraints {
            allow: Some(HashSet::from([2])),
            exclude: vec![2],
            ..Default::default()
        };
        assert!(!constraints.exclude_from([1, 2, 3], |_, _| true));
        assert_eq!(constraints.allow, Some(HashSet::new()));
    }
}
