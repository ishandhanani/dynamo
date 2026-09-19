// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A native HTTP child owns the same scheduler booking as a token request.

use std::sync::Arc;

use dynamo_kv_router::{protocols::WorkerWithDpRank, scheduling::queue::BookingHandle};

use super::{KvRouter, request_lease::RequestAttemptLease};

/// Holds a booking until an independent engine lifecycle confirms cleanup.
/// Before dispatch, dropping this value rolls back admission. After dispatch,
/// its owner must survive the HTTP connection and reconcile the engine attempt.
pub struct NativeReservation {
    router: Arc<KvRouter>,
    lease: RequestAttemptLease,
}

impl NativeReservation {
    pub fn new(router: Arc<KvRouter>, booking: BookingHandle) -> Self {
        let lease = router
            .request_lease_manager()
            .register_local(booking.commit(), None);
        Self { router, lease }
    }

    pub fn worker(&self) -> WorkerWithDpRank {
        self.lease.booking().worker
    }

    pub(crate) fn touch(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.lease.is_active(),
            "native reservation expired before engine acknowledgement"
        );
        self.lease.touch();
        Ok(())
    }

    pub(crate) async fn prefill_complete(&self) -> anyhow::Result<()> {
        self.router
            .mark_prefill_completed_if_booking(self.lease.booking())
            .await?;
        Ok(())
    }

    pub(crate) async fn finish(&self) {
        self.lease.finish().await;
    }
}
