// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct frontend dispatch to vLLM's gRPC service.

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::{
    DirectEngine, DirectEngineFactory, DisaggregationMode, ModelRuntimeConfig,
    NATIVE_GRPC_MODE_RUNTIME_KEY, direct_engine,
};
use dynamo_sidecar_common::GrpcTransportConfig;

use crate::engine::VllmSidecarEngine;

/// Builds directly dispatchable vLLM engines for workers that advertise a
/// `native_grpc_endpoint`. Install it in the frontend process with
/// `dynamo_llm::kv_router::install_direct_engine_factory`.
#[derive(Clone, Debug)]
pub struct VllmDirectEngineFactory {
    transport: GrpcTransportConfig,
}

impl Default for VllmDirectEngineFactory {
    fn default() -> Self {
        Self::new(GrpcTransportConfig::default())
    }
}

impl VllmDirectEngineFactory {
    pub fn new(transport: GrpcTransportConfig) -> Self {
        Self { transport }
    }
}

#[async_trait]
impl DirectEngineFactory for VllmDirectEngineFactory {
    async fn connect(
        &self,
        worker_id: u64,
        endpoint: &str,
        config: &ModelRuntimeConfig,
    ) -> anyhow::Result<DirectEngine> {
        // The worker names the role its engine serves; default to aggregated
        // when it does not, matching the sidecar's own default.
        let mode = config
            .runtime_data
            .get(NATIVE_GRPC_MODE_RUNTIME_KEY)
            .and_then(|value| value.as_str())
            .map(str::parse::<DisaggregationMode>)
            .transpose()
            .map_err(anyhow::Error::from)?
            .unwrap_or_default();
        let engine = VllmSidecarEngine::connect_direct(endpoint, mode, self.transport)
            .await
            .map_err(anyhow::Error::from)?;
        tracing::info!(worker_id, endpoint, %mode, "connected direct vLLM engine");
        Ok(direct_engine(Arc::new(engine), mode))
    }
}
