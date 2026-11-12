// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ISL pool: a minimal classifier that splits one model's workers into two pools by taint and
//! sends each request to one of them by input length.
//!
//! Workers carry taints in their model card (`--dyn-taints pool/large` on a Rust backend, or
//! `POST /engine/update/model_taints` on a running worker). Taints do not change the card
//! checksum, so workers of one model with different parallelism share a WorkerSet and one
//! router. The classifier reads each eligible worker's taints and narrows the request's allowed
//! workers to the pool its input length selects, in that pool's own lane so a backed-up pool
//! never blocks the other. Worker selection still picks within the pool.
//!
//! A pool that no eligible worker carries leaves the request unrestricted in the class's
//! shared lane, logged once until the pool appears, so a mistyped taint degrades to routing
//! without pools rather than to a dead lane.

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_kv_router::plugins::RouterPluginRegistry;
use dynamo_kv_router::plugins::request_classifier::{
    ClassifierError, ClassifyRequest, RequestClassifier, RequestClassifierFactory,
    RequestClassifierParameters, RequestClassifierProviderError, RequestClassifierRegistryError,
};
use dynamo_kv_router::plugins::worker_selection::{WorkerCandidates, WorkerInputs};
use dynamo_kv_router::protocols::WorkerId;
use serde::Deserialize;

pub const ISL_POOL_CLASSIFIER_TYPE: &str = "isl_pool";

/// Lane names, one per pool, so each pool's backlog holds only its own heads.
const LARGE_LANE: &str = "isl_pool.large";
const SMALL_LANE: &str = "isl_pool.small";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IslPoolConfig {
    /// Requests with more input tokens than this go to the large pool; the rest to the small
    /// pool.
    pub threshold_tokens: usize,
    /// Taint carried by the large pool's workers.
    pub large_taint: String,
    /// Taint carried by the small pool's workers.
    pub small_taint: String,
}

impl IslPoolConfig {
    fn validate(&self) -> Result<(), String> {
        for (name, taint) in [
            ("large_taint", &self.large_taint),
            ("small_taint", &self.small_taint),
        ] {
            if taint.trim().is_empty() {
                return Err(format!("{name} must be non-empty"));
            }
        }
        if self.large_taint == self.small_taint {
            return Err("large_taint and small_taint must differ".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pool {
    Large,
    Small,
}

impl Pool {
    fn lane(self) -> &'static str {
        match self {
            Pool::Large => LARGE_LANE,
            Pool::Small => SMALL_LANE,
        }
    }
}

pub struct IslPoolClassifier {
    config: IslPoolConfig,
    /// Whether the last classification for each pool found no member, so the warning is
    /// logged once per outage rather than per request.
    missing: [bool; 2],
}

impl IslPoolClassifier {
    pub fn new(config: IslPoolConfig) -> Self {
        tracing::info!(
            target: "isl_pool",
            threshold_tokens = config.threshold_tokens,
            large_taint = %config.large_taint,
            small_taint = %config.small_taint,
            "ISL pool enabled"
        );
        Self {
            config,
            missing: [false; 2],
        }
    }

    fn pool_for(&self, isl_tokens: usize) -> Pool {
        if isl_tokens > self.config.threshold_tokens {
            Pool::Large
        } else {
            Pool::Small
        }
    }
}

#[async_trait]
impl RequestClassifier for IslPoolClassifier {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::TAINTS
    }

    fn classify(
        &mut self,
        mut request: ClassifyRequest,
        workers: WorkerCandidates<'_>,
    ) -> Result<ClassifyRequest, Box<ClassifierError>> {
        let pool = self.pool_for(request.isl_tokens());
        let Self { config, missing } = self;
        let taint: &str = match pool {
            Pool::Large => &config.large_taint,
            Pool::Small => &config.small_taint,
        };
        let members: HashSet<WorkerId> = workers
            .iter()
            .filter(|candidate| {
                candidate
                    .taints()
                    .is_some_and(|taints| taints.contains(taint))
            })
            .map(|candidate| candidate.worker().worker_id)
            .collect();
        let missing = &mut missing[pool as usize];
        if members.is_empty() {
            if !*missing {
                *missing = true;
                let observed: BTreeSet<&str> = workers
                    .iter()
                    .flat_map(|candidate| candidate.taints())
                    .flat_map(|taints| taints.iter().map(String::as_str))
                    .collect();
                tracing::warn!(
                    target: "isl_pool",
                    pool = ?pool,
                    taint,
                    ?observed,
                    "no eligible worker carries the pool taint; routing without pools"
                );
            }
            return Ok(request);
        }
        if *missing {
            *missing = false;
            tracing::info!(target: "isl_pool", pool = ?pool, taint, members = members.len(), "pool has members again");
        }
        tracing::debug!(
            target: "isl_pool",
            request_id = request.request_id(),
            isl_tokens = request.isl_tokens(),
            pool = ?pool,
            members = members.len(),
            "classified"
        );
        request.set_lane(pool.lane());
        request.set_allowed_workers(members);
        Ok(request)
    }
}

fn classifier_provider(
    parameters: &RequestClassifierParameters,
) -> Result<RequestClassifierFactory, RequestClassifierProviderError> {
    let config: IslPoolConfig = parameters.deserialize()?;
    config
        .validate()
        .map_err(RequestClassifierProviderError::new)?;
    Ok(Arc::new(move |_context| {
        Box::new(IslPoolClassifier::new(config.clone()))
    }))
}

pub(crate) fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), RequestClassifierRegistryError> {
    registry.register_request_classifier(ISL_POOL_CLASSIFIER_TYPE, Arc::new(classifier_provider))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> IslPoolConfig {
        IslPoolConfig {
            threshold_tokens: 1_000,
            large_taint: "pool/large".to_string(),
            small_taint: "pool/small".to_string(),
        }
    }

    #[test]
    fn threshold_is_exclusive() {
        let classifier = IslPoolClassifier::new(config());
        assert_eq!(classifier.pool_for(999), Pool::Small);
        assert_eq!(classifier.pool_for(1_000), Pool::Small);
        assert_eq!(classifier.pool_for(1_001), Pool::Large);
    }

    #[test]
    fn config_rejects_empty_or_equal_taints() {
        let mut same = config();
        same.small_taint = same.large_taint.clone();
        assert!(same.validate().is_err());
        let mut empty = config();
        empty.small_taint = " ".to_string();
        assert!(empty.validate().is_err());
        assert!(config().validate().is_ok());
    }

    #[test]
    fn without_eligible_workers_the_request_stays_unrestricted() {
        let mut classifier = IslPoolClassifier::new(config());
        for isl_tokens in [10, 5_000] {
            let request = ClassifyRequest::for_test(isl_tokens, Some("r"), None);
            classifier
                .classify(request, WorkerCandidates::empty())
                .unwrap();
        }
        assert_eq!(classifier.missing, [true, true]);
    }
}
