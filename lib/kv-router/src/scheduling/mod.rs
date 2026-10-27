// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub use dynamo_router_types::config;
pub(crate) mod filter {
    pub use dynamo_router_types::eligibility::*;
}
mod local;
pub use dynamo_worker_selection::lora as lora_filter;
pub mod overlap;
pub mod overlap_refresh;
pub mod policy_config;
pub use dynamo_admission::policy_queue;
pub use dynamo_router_types::prefill_load;
pub mod queue;
mod queue_admission;
pub mod request_classifier;
pub use dynamo_worker_selection::selector;

pub(crate) mod types {
    pub use dynamo_router_types::sched::*;
}
pub use filter::*;
pub use local::LocalScheduler;
pub use lora_filter::LoraWorkerFilter;
pub use lora_filter::narrow_allowed_worker_ids_by_lora;
pub use overlap::{
    CacheHitEstimates, OverlapAnalysis, OverlapScoresResponse, OverlapSignals,
    SelectedWorkerTierSnapshot, SharedCacheOverlapScore, WorkerOverlapScore,
};
pub use overlap_refresh::{
    NoopOverlapScoresRefresh, OverlapScoresRefresh, RefreshedOverlap, TieredOverlapRefresher,
};
pub use policy_config::{
    BusyThresholds, PolicyClassConfig, PolicyProfile, RejectThresholdsProvider, RouterPolicyConfig,
    RouterPolicyConfigError,
};
pub use policy_queue::{
    PolicyQueue, PolicyQueueEntry, QueueLimitKind, QueueRejection, QueueSnapshot,
};
pub use prefill_load::{
    InvalidEffectivePrefillTokens, PrefillLoadEstimator, effective_prefill_tokens,
    prefill_load_hint_from_effective_tokens,
};
pub use queue_admission::WorkerPlacement;
pub use request_classifier::RequestLifecycle;
// TODO(v1.7): Remove these compatibility re-exports; use crate::plugins instead.
pub use crate::plugins::request_classifier::{
    AbortCause, ClassifierError, ClassifyEvent, ClassifyFuture, ClassifyRequest, RequestClassifier,
    RequestClassifierConfig, RequestClassifierContext, RequestClassifierFactory,
    RequestClassifierParameters, RequestClassifierProvider, RequestClassifierProviderError,
    RequestClassifierRegistryError, RequestProgress, RequestProgressUpdater,
};
// TODO(v1.7): Remove these compatibility re-exports; use crate::plugins instead.
pub use crate::plugins::worker_selection::{WorkerSelectionConfig, WorkerSelectionInstance};
pub use types::*;
