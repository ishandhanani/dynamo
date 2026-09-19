// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A native HTTP child owns the same scheduler booking as a token request.

use std::sync::{Arc, Mutex};

use dynamo_kv_router::scheduling::queue::BookingHandle;
use dynamo_runtime::pipeline::RouteTarget;

use super::{KvRouter, request_lease::RequestAttemptLease};

/// Holds a booking until an independent engine lifecycle confirms cleanup.
/// Before dispatch, dropping this value rolls back admission. After dispatch,
/// its owner must survive the HTTP connection and reconcile the engine attempt.
pub struct NativeReservation {
    inner: Reservation,
}

enum Reservation {
    Kv {
        router: Arc<KvRouter>,
        lease: RequestAttemptLease,
    },
    Hosted {
        worker: RouteTarget,
        occupancy: Mutex<Option<dynamo_runtime::pipeline::OccupancyReservation>>,
        // Keep the selector's load context alive through engine cleanup.
        _host: Arc<super::RoutingHost>,
    },
}

impl NativeReservation {
    pub fn new(router: Arc<KvRouter>, booking: BookingHandle) -> Self {
        let lease = router
            .request_lease_manager()
            .register_local(booking.commit(), None);
        Self {
            inner: Reservation::Kv { router, lease },
        }
    }

    pub(crate) fn hosted(
        worker: RouteTarget,
        occupancy: Option<dynamo_runtime::pipeline::OccupancyReservation>,
        host: Arc<super::RoutingHost>,
    ) -> Self {
        Self {
            inner: Reservation::Hosted {
                worker,
                occupancy: Mutex::new(occupancy),
                _host: host,
            },
        }
    }

    /// KV admission chooses a rank. Worker-only policies may delegate it to the engine.
    pub fn target(&self) -> RouteTarget {
        match &self.inner {
            Reservation::Kv { lease, .. } => {
                let worker = lease.booking().worker;
                RouteTarget::new(worker.worker_id, Some(worker.dp_rank))
            }
            Reservation::Hosted { worker, .. } => *worker,
        }
    }

    pub(crate) fn touch(&self) -> anyhow::Result<()> {
        if let Reservation::Kv { lease, .. } = &self.inner {
            anyhow::ensure!(
                lease.is_active(),
                "native reservation expired before engine acknowledgement"
            );
            lease.touch();
        }
        Ok(())
    }

    pub(crate) async fn prefill_complete(&self) -> anyhow::Result<()> {
        if let Reservation::Kv { router, lease } = &self.inner {
            router
                .mark_prefill_completed_if_booking(lease.booking())
                .await?;
        }
        Ok(())
    }

    pub(crate) async fn finish(&self) {
        match &self.inner {
            Reservation::Kv { lease, .. } => lease.finish().await,
            Reservation::Hosted { occupancy, .. } => {
                occupancy.lock().unwrap().take();
            }
        }
    }
}
