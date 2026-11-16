// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker load: the load message workers publish, the subscription to it, the worker monitor
//! that turns it into busy and overloaded workers, and the load context of one routing endpoint.

pub mod active_load;
pub mod metrics_subscriber;
pub mod routing_load;
mod serving_load;
mod worker_monitor;

pub use active_load::ActiveLoad;
pub use metrics_subscriber::KV_METRICS_SUBJECT;
pub use routing_load::{RouterLoadSource, RoutingLoadContext, SchedulerLoadSender};
pub use serving_load::{
    SCHEDULER_LOAD_PUBLISH_INTERVAL, SCHEDULER_LOAD_SUBJECT, serving_load_publishing_enabled,
};
pub(crate) use worker_monitor::RoutableThresholds;
pub use worker_monitor::{KvWorkerMonitor, LoadThresholdHandle, WorkerLoadState};
