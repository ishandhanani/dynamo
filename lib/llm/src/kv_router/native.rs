// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An HTTP request owns one routing booking until completion or disconnect.

use std::sync::Arc;

use dynamo_kv_router::scheduling::queue::BookingHandle;
use dynamo_runtime::pipeline::RouteTarget;
use tokio_util::sync::CancellationToken;

use super::{KvRouter, request_lease::RequestAttemptLease};

/// Holds a routing booking for the lifetime of the native HTTP request.
pub struct NativeReservation {
    inner: Reservation,
}

enum Reservation {
    Kv {
        lease: RequestAttemptLease,
    },
    Hosted {
        worker: RouteTarget,
        occupancy: Option<dynamo_runtime::pipeline::OccupancyReservation>,
        // Keep the selector's load context alive through HTTP completion.
        _host: Arc<super::RoutingHost>,
    },
}

impl NativeReservation {
    pub fn new(router: Arc<KvRouter>, booking: BookingHandle) -> Self {
        let lease = router
            .request_lease_manager()
            .register_local(booking.commit(), None);
        Self {
            inner: Reservation::Kv { lease },
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
                occupancy,
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
                "native reservation expired while HTTP request was active"
            );
            lease.touch();
        }
        Ok(())
    }

    async fn finish(self) {
        match self.inner {
            Reservation::Kv { lease, .. } => lease.finish().await,
            Reservation::Hosted { occupancy, .. } => {
                drop(occupancy);
            }
        }
    }
    // Renew router leases while HTTP is in flight; release on HTTP completion,
    // rejection or disconnect, without an engine cleanup handshake.
    pub(crate) fn start(self, cancellation: CancellationToken) -> tokio_util::sync::DropGuard {
        let done = CancellationToken::new();
        let completed = done.clone();
        tokio::spawn(async move {
            let expiry = dynamo_kv_router::multi_worker_sequence::active_request_expiry_duration();
            let mut heartbeat =
                tokio::time::interval((expiry / 3).min(std::time::Duration::from_secs(5)));
            heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    biased;
                    _ = completed.cancelled() => break,
                    _ = cancellation.cancelled() => break,
                    _ = heartbeat.tick() => {
                        if self.touch().is_err() {
                            cancellation.cancel();
                            break;
                        }
                    }
                }
            }
            self.finish().await;
        });
        done.drop_guard()
    }
}
