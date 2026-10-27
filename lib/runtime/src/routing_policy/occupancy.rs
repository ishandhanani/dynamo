// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashSet,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use dashmap::{DashMap, mapref::entry::Entry};
use dynamo_router_types::slots::{LoadTracker, SlotEvent, SlotId};

use super::{CandidateView, RouteContext, RouteDecision, RoutePicker, RoutePolicy};
use crate::{component::Endpoint, traits::DistributedRuntimeProvider};

/// The result of an atomic load-aware selection and reservation.
pub struct OccupancySelection {
    worker_id: u64,
    candidate_count: usize,
    load: u64,
    reservation: OccupancyReservation,
}

impl OccupancySelection {
    pub fn worker_id(&self) -> u64 {
        self.worker_id
    }

    pub fn candidate_count(&self) -> usize {
        self.candidate_count
    }

    /// Selected worker load after this request was admitted.
    pub fn load(&self) -> u64 {
        self.load
    }

    pub fn into_reservation(self) -> OccupancyReservation {
        self.reservation
    }
}

/// One reserved unit of load: the worker it is charged to and the counter it
/// incremented, kept so a retarget or release can balance exactly that counter
/// even after the worker left discovery and a same-id re-add made a new one.
struct Slot {
    worker_id: u64,
    counter: Arc<AtomicU64>,
}

/// Shared O(1) per-worker request occupancy, written through [`LoadTracker`]
/// slots.
///
/// Discovery controls eligibility separately from accounting. Removing a worker
/// marks it absent immediately, but a counter with live reservations remains until
/// its final reservation releases. Re-adding the same worker ID therefore sees the
/// retained load instead of starting from zero.
#[derive(Debug, Default)]
pub struct RoutingOccupancyState {
    counts: DashMap<u64, Arc<AtomicU64>>,
    slots: DashMap<u64, Slot>,
    next_slot: AtomicU64,
    discovered: parking_lot::RwLock<HashSet<u64>>,
    admission_lock: parking_lot::Mutex<()>,
}

impl std::fmt::Debug for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Slot")
            .field("worker_id", &self.worker_id)
            .field("load", &self.counter.load(Ordering::Relaxed))
            .finish()
    }
}

impl RoutingOccupancyState {
    fn increment_locked(&self, worker_id: u64) -> Arc<AtomicU64> {
        let count = self
            .counts
            .entry(worker_id)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        count.fetch_add(1, Ordering::Relaxed);
        count
    }

    /// Charge `worker_id` and record the slot; the admission lock must be held.
    fn reserve_locked(&self, worker_id: u64) -> SlotId {
        let counter = self.increment_locked(worker_id);
        let slot = SlotId::new(self.next_slot.fetch_add(1, Ordering::Relaxed));
        self.slots.insert(slot.raw(), Slot { worker_id, counter });
        slot
    }

    pub(crate) async fn select_exact_min_and_increment(&self, worker_ids: &[u64]) -> Option<u64> {
        let picker = RoutePicker::new(RoutePolicy::LeastLoaded);
        self.select_and_admit(
            &picker,
            CandidateView::Workers(worker_ids),
            RouteContext::default(),
        )
        .map(|(decision, _)| decision.target.worker_id)
    }

    pub(crate) fn peek(
        &self,
        picker: &RoutePicker,
        candidates: CandidateView<'_>,
        context: RouteContext,
    ) -> Option<RouteDecision> {
        picker.peek(candidates, context, |id| self.load(id))
    }

    /// Run `picker` against the live load view and, when the decision asks for
    /// occupancy admission, reserve its worker in the same critical section.
    pub(crate) fn select_and_admit(
        &self,
        picker: &RoutePicker,
        candidates: CandidateView<'_>,
        context: RouteContext,
    ) -> Option<(RouteDecision, Option<SlotId>)> {
        let _admission = self.admission_lock.lock();
        let decision = picker.select(candidates, context, |id| self.load(id))?;
        let slot = match decision.admission {
            super::AdmissionKind::None => None,
            super::AdmissionKind::Occupancy => Some(self.reserve_locked(decision.target.worker_id)),
        };
        Some((decision, slot))
    }

    /// Atomically run host-owned selection against the live load view and reserve its result.
    pub fn select_and_reserve_with<E>(
        self: &Arc<Self>,
        candidates: &[u64],
        select: impl FnOnce(&dyn Fn(u64) -> u64) -> Result<u64, E>,
    ) -> Result<OccupancySelection, E> {
        let _admission = self.admission_lock.lock();
        let worker_id = select(&|worker_id| self.load(worker_id))?;
        debug_assert!(candidates.contains(&worker_id));
        let slot = self.reserve_locked(worker_id);
        let load = self.load(worker_id);
        Ok(OccupancySelection {
            worker_id,
            candidate_count: candidates.len(),
            load,
            reservation: OccupancyReservation::from_slot(Arc::clone(self), worker_id, slot),
        })
    }

    /// Reserve an explicitly selected worker.
    pub fn reserve(self: &Arc<Self>, worker_id: u64) -> OccupancyReservation {
        let slot = LoadTracker::reserve(&**self, worker_id);
        OccupancyReservation::from_slot(Arc::clone(self), worker_id, slot)
    }

    /// Wrap a slot this tracker issued in a guard that releases it on drop.
    pub fn reservation(self: &Arc<Self>, worker_id: u64, slot: SlotId) -> OccupancyReservation {
        OccupancyReservation::from_slot(Arc::clone(self), worker_id, slot)
    }

    fn decrement_locked(&self, worker_id: u64, counter: &Arc<AtomicU64>) {
        let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            Some(current.saturating_sub(1))
        });

        if counter.load(Ordering::Relaxed) != 0 || self.discovered.read().contains(&worker_id) {
            return;
        }

        if let Entry::Occupied(entry) = self.counts.entry(worker_id)
            && Arc::ptr_eq(entry.get(), counter)
            && entry.get().load(Ordering::Relaxed) == 0
        {
            entry.remove();
        }
    }

    /// The load currently charged to `worker_id`.
    pub fn load(&self, worker_id: u64) -> u64 {
        self.counts
            .get(&worker_id)
            .map(|count| count.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    #[cfg(test)]
    pub(crate) fn increment(&self, worker_id: u64) -> SlotId {
        LoadTracker::reserve(self, worker_id)
    }

    #[cfg(test)]
    pub(crate) fn decrement(&self, worker_id: u64) {
        let _admission = self.admission_lock.lock();
        let counter = self.counts.get(&worker_id).map(|entry| entry.clone());
        if let Some(counter) = counter {
            self.decrement_locked(worker_id, &counter);
        }
    }

    /// Reconcile discovery eligibility without discarding guard-owned accounting.
    pub(crate) fn retain(&self, worker_ids: &[u64]) {
        let _admission = self.admission_lock.lock();
        let mut live = self.discovered.write();
        live.clear();
        live.extend(worker_ids.iter().copied());
        self.counts.retain(|worker_id, count| {
            live.contains(worker_id) || count.load(Ordering::Relaxed) != 0
        });
    }

    #[cfg(test)]
    pub(crate) fn contains_slot(&self, worker_id: u64) -> bool {
        self.counts.contains_key(&worker_id)
    }
}

impl LoadTracker for RoutingOccupancyState {
    fn reserve(&self, worker: u64) -> SlotId {
        let _admission = self.admission_lock.lock();
        self.reserve_locked(worker)
    }

    fn apply(&self, event: SlotEvent) -> bool {
        let _admission = self.admission_lock.lock();
        match event {
            SlotEvent::Retarget { slot, to } => {
                let Some(mut entry) = self.slots.get_mut(&slot.raw()) else {
                    return false;
                };
                if entry.worker_id == to {
                    return true;
                }
                let next = self.increment_locked(to);
                self.decrement_locked(entry.worker_id, &entry.counter);
                entry.worker_id = to;
                entry.counter = next;
                true
            }
            SlotEvent::Release { slot } => {
                let Some((_, released)) = self.slots.remove(&slot.raw()) else {
                    return false;
                };
                self.decrement_locked(released.worker_id, &released.counter);
                true
            }
        }
    }

    fn load(&self, worker: u64) -> u64 {
        RoutingOccupancyState::load(self, worker)
    }
}

/// One guard-owned occupancy booking: a [`SlotId`] released when the guard drops.
pub struct OccupancyReservation {
    state: Arc<RoutingOccupancyState>,
    worker_id: u64,
    slot: SlotId,
}

impl OccupancyReservation {
    fn from_slot(state: Arc<RoutingOccupancyState>, worker_id: u64, slot: SlotId) -> Self {
        Self {
            state,
            worker_id,
            slot,
        }
    }

    pub fn worker_id(&self) -> u64 {
        self.worker_id
    }

    pub fn slot(&self) -> SlotId {
        self.slot
    }

    /// Load on this booking's worker, including the booking itself.
    pub fn load(&self) -> u64 {
        self.state.load(self.worker_id)
    }

    /// Move this booking to the worker selected by transport fallback.
    pub fn retarget(&mut self, worker_id: u64) -> u64 {
        if self.worker_id != worker_id {
            self.state.apply(SlotEvent::Retarget {
                slot: self.slot,
                to: worker_id,
            });
            self.worker_id = worker_id;
        }
        self.load()
    }
}

impl Drop for OccupancyReservation {
    fn drop(&mut self) {
        self.state.apply(SlotEvent::Release { slot: self.slot });
    }
}

/// Get or create the shared routing occupancy state for an endpoint.
pub(crate) async fn get_or_create_routing_occupancy_state(
    endpoint: &Endpoint,
) -> Arc<RoutingOccupancyState> {
    let drt = endpoint.drt();
    let registry = drt.routing_occupancy_states();
    let mut registry = registry.lock().await;

    if let Some(weak) = registry.get(endpoint) {
        if let Some(state) = weak.upgrade() {
            return state;
        }
        registry.remove(endpoint);
    }

    let state = Arc::new(RoutingOccupancyState::default());
    registry.insert(endpoint.clone(), Arc::downgrade(&state));
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_worker_keeps_live_reservation_across_same_id_readd() {
        let state = Arc::new(RoutingOccupancyState::default());
        state.retain(&[7]);
        let old = state.reserve(7);

        state.retain(&[]);
        assert_eq!(state.load(7), 1);
        assert!(state.contains_slot(7));

        state.retain(&[7]);
        let new = state.reserve(7);
        assert_eq!(state.load(7), 2);

        drop(old);
        assert_eq!(state.load(7), 1);
        drop(new);
        assert_eq!(state.load(7), 0);
    }

    #[test]
    fn absent_worker_slot_is_removed_after_final_release() {
        let state = Arc::new(RoutingOccupancyState::default());
        state.retain(&[7]);
        let reservation = state.reserve(7);

        state.retain(&[]);
        assert!(state.contains_slot(7));
        drop(reservation);

        assert_eq!(state.load(7), 0);
        assert!(!state.contains_slot(7));
    }

    #[test]
    fn retarget_moves_exactly_one_booking() {
        let state = Arc::new(RoutingOccupancyState::default());
        state.retain(&[1, 2]);
        let mut reservation = state.reserve(1);

        assert_eq!(reservation.retarget(2), 1);
        assert_eq!(state.load(1), 0);
        assert_eq!(state.load(2), 1);

        drop(reservation);
        assert_eq!(state.load(2), 0);
    }

    #[test]
    fn concurrent_selection_and_reservation_stays_balanced() {
        let state = Arc::new(RoutingOccupancyState::default());
        state.retain(&[1, 2, 3]);
        let threads = (0..90)
            .map(|_| {
                let state = Arc::clone(&state);
                std::thread::spawn(move || {
                    state
                        .select_and_reserve_with(&[1, 2, 3], |load| {
                            Ok::<_, std::convert::Infallible>(
                                [1, 2, 3]
                                    .into_iter()
                                    .min_by_key(|worker_id| load(*worker_id))
                                    .unwrap(),
                            )
                        })
                        .unwrap()
                        .into_reservation()
                })
            })
            .collect::<Vec<_>>();
        let reservations = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();

        assert_eq!([state.load(1), state.load(2), state.load(3)], [30, 30, 30]);
        drop(reservations);
        assert_eq!([state.load(1), state.load(2), state.load(3)], [0, 0, 0]);
    }

    #[test]
    fn tracker_events_balance_the_slot_they_name() {
        let state = Arc::new(RoutingOccupancyState::default());
        state.retain(&[1, 2]);
        let slot = LoadTracker::reserve(&*state, 1);
        assert_eq!(state.load(1), 1);

        assert!(state.apply(SlotEvent::Retarget { slot, to: 2 }));
        assert_eq!([state.load(1), state.load(2)], [0, 1]);

        assert!(state.apply(SlotEvent::Release { slot }));
        assert_eq!([state.load(1), state.load(2)], [0, 0]);
        assert!(!state.apply(SlotEvent::Release { slot }));
        assert!(!state.apply(SlotEvent::Retarget { slot, to: 1 }));
        assert_eq!([state.load(1), state.load(2)], [0, 0]);
    }
}
