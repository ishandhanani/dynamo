// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The worker membership one router endpoint sees: one row per serving worker,
//! published as whole snapshots and diffed into membership events.
//!
//! A host builds a [`Topology`] from its discovery sources and hands the same
//! snapshot to every reader (selection, load probes, metrics, shared caches),
//! so there is one view of which workers exist and what they advertise. The
//! row's config type `C` is the host's: the kernel carries the shape and the
//! diff, not the engine configuration. Stage crates read a snapshot through
//! [`WorkerLookup`] and never name `Topology`.

use std::sync::Arc;

use dynamo_router_types::eligibility::WorkerLookup;
use dynamo_router_types::protocols::WorkerId;

/// One serving worker and the configuration it advertised.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkerRow<C> {
    pub worker_id: WorkerId,
    pub config: C,
}

/// The workers an endpoint can route to right now, sorted by worker id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Topology<C> {
    /// Increments on every published change, so a reader can tell two
    /// snapshots apart without comparing rows.
    generation: u64,
    rows: Vec<WorkerRow<C>>,
}

impl<C> Default for Topology<C> {
    fn default() -> Self {
        Self {
            generation: 0,
            rows: Vec::new(),
        }
    }
}

impl<C> Topology<C> {
    /// A snapshot from unsorted rows; duplicate worker ids keep the last row.
    pub fn new(generation: u64, mut rows: Vec<WorkerRow<C>>) -> Self {
        rows.sort_by_key(|row| row.worker_id);
        rows.dedup_by(|later, earlier| {
            if later.worker_id == earlier.worker_id {
                std::mem::swap(later, earlier);
                true
            } else {
                false
            }
        });
        Self { generation, rows }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn rows(&self) -> &[WorkerRow<C>] {
        &self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn contains(&self, worker_id: WorkerId) -> bool {
        self.get(worker_id).is_some()
    }

    pub fn get(&self, worker_id: WorkerId) -> Option<&C> {
        self.rows
            .binary_search_by_key(&worker_id, |row| row.worker_id)
            .ok()
            .map(|index| &self.rows[index].config)
    }

    pub fn worker_ids(&self) -> impl Iterator<Item = WorkerId> + '_ {
        self.rows.iter().map(|row| row.worker_id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (WorkerId, &C)> + '_ {
        self.rows.iter().map(|row| (row.worker_id, &row.config))
    }

    /// The same rows under the next generation.
    pub fn with_generation(mut self, generation: u64) -> Self {
        self.generation = generation;
        self
    }
}

impl<C: PartialEq> Topology<C> {
    /// Whether `rows` is the membership this snapshot already describes.
    pub fn same_rows(&self, other: &Self) -> bool {
        self.rows == other.rows
    }

    /// The events that take a reader from `self` to `next`. Both are sorted,
    /// so this is one merge pass.
    pub fn diff<'a>(&'a self, next: &'a Self) -> Vec<MembershipEvent<&'a C>> {
        let mut events = Vec::new();
        let (mut left, mut right) = (self.rows.iter().peekable(), next.rows.iter().peekable());
        loop {
            match (left.peek(), right.peek()) {
                (Some(old), Some(new)) if old.worker_id == new.worker_id => {
                    if old.config != new.config {
                        events.push(MembershipEvent::Updated(WorkerRow {
                            worker_id: new.worker_id,
                            config: &new.config,
                        }));
                    }
                    left.next();
                    right.next();
                }
                (Some(old), Some(new)) if old.worker_id < new.worker_id => {
                    events.push(MembershipEvent::Removed(old.worker_id));
                    left.next();
                }
                (Some(_), Some(new)) => {
                    events.push(MembershipEvent::Added(WorkerRow {
                        worker_id: new.worker_id,
                        config: &new.config,
                    }));
                    right.next();
                }
                (Some(old), None) => {
                    events.push(MembershipEvent::Removed(old.worker_id));
                    left.next();
                }
                (None, Some(new)) => {
                    events.push(MembershipEvent::Added(WorkerRow {
                        worker_id: new.worker_id,
                        config: &new.config,
                    }));
                    right.next();
                }
                (None, None) => break,
            }
        }
        events
    }
}

/// A shared, immutable snapshot: what a watch channel carries.
pub type SharedTopology<C> = Arc<Topology<C>>;

impl<C> WorkerLookup<C> for Topology<C> {
    fn config(&self, worker_id: WorkerId) -> Option<&C> {
        Topology::get(self, worker_id)
    }

    fn workers<'a>(&'a self) -> impl Iterator<Item = (WorkerId, &'a C)>
    where
        C: 'a,
    {
        self.iter()
    }

    fn capacity_hint(&self) -> usize {
        self.len()
    }
}

/// One change between two topology snapshots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MembershipEvent<C> {
    Added(WorkerRow<C>),
    Updated(WorkerRow<C>),
    Removed(WorkerId),
    /// A reader that missed events reloads from this snapshot.
    Resync(SharedTopology<C>),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topology(generation: u64, rows: &[(WorkerId, &str)]) -> Topology<String> {
        Topology::new(
            generation,
            rows.iter()
                .map(|(worker_id, config)| WorkerRow {
                    worker_id: *worker_id,
                    config: config.to_string(),
                })
                .collect(),
        )
    }

    #[test]
    fn rows_are_sorted_and_deduplicated_keeping_the_last() {
        let topology = topology(1, &[(7, "a"), (3, "b"), (7, "c")]);
        assert_eq!(topology.worker_ids().collect::<Vec<_>>(), [3, 7]);
        assert_eq!(topology.get(7).map(String::as_str), Some("c"));
        assert_eq!(topology.get(3).map(String::as_str), Some("b"));
        assert!(!topology.contains(4));
        assert_eq!(topology.len(), 2);
    }

    #[test]
    fn diff_reports_added_updated_and_removed_in_worker_order() {
        let before = topology(1, &[(1, "a"), (2, "b"), (4, "d")]);
        let after = topology(2, &[(2, "b2"), (3, "c"), (4, "d")]);
        let events = before.diff(&after);
        assert_eq!(
            events,
            vec![
                MembershipEvent::Removed(1),
                MembershipEvent::Updated(WorkerRow {
                    worker_id: 2,
                    config: &"b2".to_string(),
                }),
                MembershipEvent::Added(WorkerRow {
                    worker_id: 3,
                    config: &"c".to_string(),
                }),
            ]
        );
        assert!(after.diff(&after).is_empty());
        assert!(before.same_rows(&before.clone().with_generation(9)));
    }
}
