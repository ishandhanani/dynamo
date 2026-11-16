// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA integration with frontend discovery, metrics, and event transport.

use std::sync::LazyLock;

use dynamo_runtime::metrics::prometheus_names::frontend_service;
use prometheus::{IntGaugeVec, Opts, Registry};

mod subscription;
pub use subscription::LoadEstimatorSubscription;

#[cfg(test)]
mod metrics_tests;

// ---------------------------------------------------------------------------
// LoRA allocation metrics (updated by LoraController each tick)
// ---------------------------------------------------------------------------

pub static LORA_REPLICA_FACTOR_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!("dynamo_frontend_{}", frontend_service::LORA_REPLICA_FACTOR),
            "Number of replicas allocated for a LoRA adapter",
        ),
        &["endpoint", "lora"],
    )
    .expect("Failed to create lora_replica_factor gauge")
});

pub static LORA_IS_ACTIVE_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!("dynamo_frontend_{}", frontend_service::LORA_IS_ACTIVE),
            "Whether a LoRA adapter is active (1) or inactive (0)",
        ),
        &["endpoint", "lora"],
    )
    .expect("Failed to create lora_is_active gauge")
});

pub static LORA_RAW_ARRIVAL_COUNT_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!(
                "dynamo_frontend_{}",
                frontend_service::LORA_RAW_ARRIVAL_COUNT
            ),
            "Raw arrival count (windowed rate counter) for a LoRA adapter",
        ),
        &["endpoint", "lora"],
    )
    .expect("Failed to create lora_raw_arrival_count gauge")
});

pub static LORA_ESTIMATED_LOAD_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!("dynamo_frontend_{}", frontend_service::LORA_ESTIMATED_LOAD),
            "Estimated load (windowed request count) for a LoRA adapter",
        ),
        &["endpoint", "lora"],
    )
    .expect("Failed to create lora_estimated_load gauge")
});

pub static LORA_ACTIVE_REQUESTS_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!("dynamo_frontend_{}", frontend_service::LORA_ACTIVE_REQUESTS),
            "Number of in-flight requests for a LoRA adapter",
        ),
        &["endpoint", "lora"],
    )
    .expect("Failed to create lora_active_requests gauge")
});

pub static LORA_CHURN_LOADS_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!(
                "dynamo_frontend_{}",
                frontend_service::LORA_CHURN_LOADS_TOTAL
            ),
            "Total LoRA loads (new placements) this tick",
        ),
        &["endpoint"],
    )
    .expect("Failed to create lora_churn_loads gauge")
});

pub static LORA_CHURN_UNLOADS_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!(
                "dynamo_frontend_{}",
                frontend_service::LORA_CHURN_UNLOADS_TOTAL
            ),
            "Total LoRA unloads (removed placements) this tick",
        ),
        &["endpoint"],
    )
    .expect("Failed to create lora_churn_unloads gauge")
});

pub static LORA_OVERFLOW_COUNT_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!("dynamo_frontend_{}", frontend_service::LORA_OVERFLOW_COUNT),
            "MCF solver overflow count (unplaceable replicas)",
        ),
        &["endpoint"],
    )
    .expect("Failed to create lora_overflow_count gauge")
});

pub fn register_lora_allocation_metrics(registry: &Registry) -> Result<(), prometheus::Error> {
    registry.register(Box::new(LORA_REPLICA_FACTOR_GAUGE.clone()))?;
    registry.register(Box::new(LORA_IS_ACTIVE_GAUGE.clone()))?;
    registry.register(Box::new(LORA_RAW_ARRIVAL_COUNT_GAUGE.clone()))?;
    registry.register(Box::new(LORA_ESTIMATED_LOAD_GAUGE.clone()))?;
    registry.register(Box::new(LORA_ACTIVE_REQUESTS_GAUGE.clone()))?;
    registry.register(Box::new(LORA_CHURN_LOADS_GAUGE.clone()))?;
    registry.register(Box::new(LORA_CHURN_UNLOADS_GAUGE.clone()))?;
    registry.register(Box::new(LORA_OVERFLOW_COUNT_GAUGE.clone()))?;
    Ok(())
}

/// Inject the frontend registry's existing series into the allocation controller.
pub fn controller_metrics() -> dynamo_lora::metrics::ControllerMetrics {
    dynamo_lora::metrics::ControllerMetrics {
        replica_factor: LORA_REPLICA_FACTOR_GAUGE.clone(),
        is_active: LORA_IS_ACTIVE_GAUGE.clone(),
        raw_arrival_count: LORA_RAW_ARRIVAL_COUNT_GAUGE.clone(),
        estimated_load: LORA_ESTIMATED_LOAD_GAUGE.clone(),
        active_requests: LORA_ACTIVE_REQUESTS_GAUGE.clone(),
        churn_loads: LORA_CHURN_LOADS_GAUGE.clone(),
        churn_unloads: LORA_CHURN_UNLOADS_GAUGE.clone(),
        overflow_count: LORA_OVERFLOW_COUNT_GAUGE.clone(),
    }
}
