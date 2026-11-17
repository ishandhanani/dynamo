// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod broker_zmq;
mod direct_zmq;
mod recovery_lane;
mod source_health;
mod state_agent;
mod subscriber;
mod target;
mod worker_query;
mod worker_query_state;
mod worker_query_transport;

pub(crate) use state_agent::start_state_agent_router;
pub(crate) use subscriber::{KvEventSubscriptionHandle, start_subscriber};
pub use subscriber::{RecoverySupervisor, start_target_subscriber};
pub(crate) use target::IndexerRecoveryTarget;
pub use target::{RecoveryResetReason, RecoveryTarget};
pub use worker_query::DEFAULT_RECOVERY_ATTEMPT_TIMEOUT;
pub use worker_query::TargetFaultDisposition;
#[cfg(any(test, feature = "testing"))]
pub use worker_query::WorkerQueryClient;
#[cfg(feature = "ckf-diagnostics")]
pub use worker_query::WorkerQueryHealthSnapshot;
pub(crate) use worker_query_transport::RuntimeWorkerQueryTransport;
#[cfg(any(test, feature = "testing"))]
pub use worker_query_transport::WorkerQueryTransport;
