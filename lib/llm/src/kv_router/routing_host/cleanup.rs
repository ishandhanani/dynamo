// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Policy-specific reservation ownership shared by all RoutingHost dispatch paths.

use std::sync::Arc;

use dynamo_kv_router::{protocols::WorkerWithDpRank, scheduling::queue::BookingHandle};
use dynamo_runtime::pipeline::OccupancyReservation;

use crate::kv_router::{
    KvRouter, indexer::ApproximateRequestLease, request_lease::RequestAttemptLease,
};

/// Owns the shared attempt-scoped scheduler and approximate-LRU lifecycle after
/// a KV worker is selected.
pub(super) struct KvRequestCleanup {
    pub(super) chooser: Arc<KvRouter>,
    pub(super) context_id: String,
    pub(super) worker: WorkerWithDpRank,
    pub(super) approximate_lru: Option<ApproximateRequestLease>,
    pub(super) lifecycle: Option<RequestAttemptLease>,
}

impl KvRequestCleanup {
    pub(super) fn new(
        chooser: Arc<KvRouter>,
        context_id: String,
        worker: WorkerWithDpRank,
        booking: Option<BookingHandle>,
    ) -> Self {
        let booking = booking.map(BookingHandle::commit);
        let approximate_lru = booking.as_ref().and_then(|booking| {
            let registration = chooser.approximate_lru_rank_registration(worker)?;
            chooser.indexer().begin_approximate_lru_request(
                worker,
                registration.incarnation,
                booking.attempt_id,
            )
        });
        let lifecycle = booking.map(|booking| {
            chooser
                .request_lease_manager()
                .register_local(booking, approximate_lru.clone())
        });
        Self {
            chooser,
            context_id,
            worker,
            approximate_lru,
            lifecycle,
        }
    }

    pub(super) fn lifecycle(&self) -> Option<&RequestAttemptLease> {
        self.lifecycle.as_ref()
    }

    pub(super) async fn finish(&self) {
        if let Some(lifecycle) = &self.lifecycle {
            lifecycle.finish().await;
        }
    }
}

/// Policy-specific state released by the host's common request lifecycle.
pub(super) enum RequestCleanup {
    Kv(KvRequestCleanup),
    Stateless {
        worker_id: u64,
    },
    Occupancy {
        worker_id: u64,
        reservation: Option<OccupancyReservation>,
    },
}

impl RequestCleanup {
    pub(super) fn unobserved_kv(
        chooser: Arc<KvRouter>,
        booking: BookingHandle,
    ) -> (Self, WorkerWithDpRank) {
        let booking = booking.commit();
        let context_id = booking.request_id.clone();
        let worker = booking.worker;
        let lifecycle = chooser
            .request_lease_manager()
            .register_local(booking, None);
        (
            Self::Kv(KvRequestCleanup {
                chooser,
                context_id,
                worker,
                approximate_lru: None,
                lifecycle: Some(lifecycle),
            }),
            worker,
        )
    }

    pub(super) fn worker_id(&self) -> u64 {
        match self {
            Self::Kv(cleanup) => cleanup.worker.worker_id,
            Self::Stateless { worker_id } => *worker_id,
            Self::Occupancy { worker_id, .. } => *worker_id,
        }
    }

    pub(super) fn retarget_worker(&mut self, worker_id: u64) -> Option<u64> {
        match self {
            Self::Kv(_) => {
                debug_assert!(false, "KV cleanup target cannot be retargeted");
                None
            }
            Self::Stateless { worker_id: current } => {
                *current = worker_id;
                None
            }
            Self::Occupancy {
                worker_id: current,
                reservation,
            } => {
                let occupancy = reservation
                    .as_mut()
                    .map(|reservation| reservation.retarget(worker_id));
                *current = worker_id;
                occupancy
            }
        }
    }

    pub(super) fn context_id(&self) -> Option<&str> {
        match self {
            Self::Kv(cleanup) => Some(&cleanup.context_id),
            Self::Stateless { .. } | Self::Occupancy { .. } => None,
        }
    }

    pub(super) fn lifecycle(&self) -> Option<&RequestAttemptLease> {
        match self {
            Self::Kv(cleanup) => cleanup.lifecycle(),
            Self::Stateless { .. } | Self::Occupancy { .. } => None,
        }
    }

    pub(super) async fn finish(&mut self) {
        match self {
            Self::Kv(cleanup) => cleanup.finish().await,
            Self::Occupancy { reservation, .. } => drop(reservation.take()),
            Self::Stateless { .. } => {}
        }
    }
}
