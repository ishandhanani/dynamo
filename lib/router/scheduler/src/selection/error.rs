// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::scheduling_compat::KvSchedulerError;
use dynamo_kv_indexer::KvRouterError;
use dynamo_slot_tracker::SequenceError;

#[derive(Debug, thiserror::Error)]
pub enum SelectionError {
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    NotReady(String),
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    Conflict(String),
    #[error("{0}")]
    Internal(String),
    #[error(transparent)]
    Scheduler(#[from] KvSchedulerError),
    #[error(transparent)]
    Sequence(#[from] SequenceError),
    /// The KV index lookup failed; an offline remote index is not ready,
    /// anything else is internal.
    #[error(transparent)]
    Indexer(#[from] KvRouterError),
}

impl SelectionError {
    /// HTTP-style status code for this error, for callers that consume the
    /// service in-process without an HTTP layer.
    pub fn status_code(&self) -> u16 {
        match self {
            Self::BadRequest(_) => 400,
            Self::NotReady(_) => 503,
            Self::NotFound(_) => 404,
            Self::Conflict(_) => 409,
            Self::Internal(_) => 500,
            Self::Scheduler(error) => scheduler_error_status(error),
            Self::Sequence(error) => sequence_error_status(error),
            Self::Indexer(KvRouterError::IndexerOffline) => 503,
            Self::Indexer(_) => 500,
        }
    }

    /// Stable, machine-readable category for this error.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::BadRequest(_) => "bad_request",
            Self::NotReady(_) => "not_ready",
            Self::NotFound(_) => "not_found",
            Self::Conflict(_) => "conflict",
            Self::Internal(_) => "internal",
            Self::Scheduler(_) => "scheduler",
            Self::Sequence(_) => "sequence",
            Self::Indexer(KvRouterError::IndexerOffline) => "not_ready",
            Self::Indexer(_) => "internal",
        }
    }
}

fn scheduler_error_status(error: &KvSchedulerError) -> u16 {
    match error {
        KvSchedulerError::NoEndpoints
        | KvSchedulerError::AllEligibleWorkersFiltered
        | KvSchedulerError::SubscriberShutdown
        | KvSchedulerError::InitFailed(_) => 503,
        KvSchedulerError::WorkerSelectionPolicy(_) => 500,
        // Deadline expiry is deliberately 429, not 504: the deadline elapsed
        // while waiting for capacity, so it is backpressure the client should
        // respond to like the overloaded family, not a gateway timeout.
        KvSchedulerError::AllEligibleWorkersOverloaded
        | KvSchedulerError::PinnedWorkerOverloaded { .. }
        | KvSchedulerError::QueueRejected(_)
        | KvSchedulerError::DeadlineExceeded => 429,
        KvSchedulerError::PinnedWorkerNotAllowed { .. } => 400,
        // A duplicate live request id, or a lifecycle the caller ended (or
        // re-registered) mid-classification, is caller-induced, like
        // `BookingFailed` and `SequenceError::DuplicateRequest`.
        KvSchedulerError::BookingFailed(_)
        | KvSchedulerError::DuplicateClassificationRequestId(_)
        | KvSchedulerError::ClassificationLifecycleEnded(_) => 409,
        KvSchedulerError::RequestClassifierPanicked(_)
        | KvSchedulerError::RequestClassifierFailed(_)
        | KvSchedulerError::InvalidClassificationMetadata(_) => 500,
    }
}

fn sequence_error_status(error: &SequenceError) -> u16 {
    match error {
        SequenceError::WorkerNotFound { .. } | SequenceError::RequestNotFound { .. } => 404,
        SequenceError::DuplicateRequest { .. } => 409,
        SequenceError::ReplicaSyncPublishFailed(_) => 500,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filtered_workers_are_unavailable_not_overloaded() {
        assert_eq!(
            SelectionError::Scheduler(KvSchedulerError::AllEligibleWorkersFiltered).status_code(),
            503
        );
        assert_eq!(
            SelectionError::Scheduler(KvSchedulerError::AllEligibleWorkersOverloaded).status_code(),
            429
        );
        assert_eq!(
            SelectionError::Scheduler(KvSchedulerError::DeadlineExceeded).status_code(),
            429
        );
    }
}
