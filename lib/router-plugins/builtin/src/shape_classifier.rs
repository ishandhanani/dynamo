// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `shape`: a request classifier that selects the policy class from the request's shape.
//!
//! A request whose `max_tokens` is at or above `decode_tokens_threshold` joins `decode_class`;
//! every other request joins `default_class` when one is configured. With `decode_cost_weight`
//! above zero the scheduling cost becomes `uncached_isl + weight * max_tokens`, so DRR charges
//! expected decode work, which the default cost ignores. `compaction_class` routes requests the
//! harness marks as context compaction (`x-claude-code-request-class: compaction` or a Codex
//! `request_kind` of `compaction`) to their own class ahead of the shape rule.

use std::sync::Arc;

use dynamo_kv_router::plugins::RouterPluginRegistry;
use dynamo_kv_router::plugins::request_classifier::{
    ClassifyFuture, ClassifyRequest, RequestClassifier, RequestClassifierFactory,
    RequestClassifierParameters, RequestClassifierProviderError, RequestClassifierRegistryError,
};
use dynamo_kv_router::plugins::worker_selection::WorkerCandidates;

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    decode_class: String,
    #[serde(default)]
    default_class: Option<String>,
    #[serde(default = "default_decode_tokens_threshold")]
    decode_tokens_threshold: u32,
    #[serde(default)]
    decode_cost_weight: f64,
    #[serde(default)]
    compaction_class: Option<String>,
}

fn default_decode_tokens_threshold() -> u32 {
    1024
}

struct ShapeClassifier {
    parameters: Parameters,
}

fn is_compaction(request: &ClassifyRequest) -> bool {
    let Some(session) = request.session_context() else {
        return false;
    };
    session.agent_headers().iter().any(|(name, values)| {
        let name = name.to_ascii_lowercase();
        if name == "x-claude-code-request-class" {
            values.iter().any(|v| v.eq_ignore_ascii_case("compaction"))
        } else if name == "x-codex-turn-metadata" {
            values
                .iter()
                .any(|v| v.contains("\"request_kind\":\"compaction\""))
        } else {
            false
        }
    })
}

impl Parameters {
    fn class_for(&self, expected_output_tokens: Option<u32>, compaction: bool) -> Option<&str> {
        match (&self.compaction_class, expected_output_tokens) {
            (Some(class), _) if compaction => Some(class.as_str()),
            (_, Some(tokens)) if tokens >= self.decode_tokens_threshold => {
                Some(self.decode_class.as_str())
            }
            _ => self.default_class.as_deref(),
        }
    }

    /// `None` leaves the default cost (uncached input tokens) in place.
    fn cost_for(&self, uncached_cost: usize, expected_output_tokens: Option<u32>) -> Option<usize> {
        if self.decode_cost_weight <= 0.0 {
            return None;
        }
        let tokens = expected_output_tokens?;
        let cost = uncached_cost as f64 + self.decode_cost_weight * f64::from(tokens);
        Some(cost.round().max(1.0) as usize)
    }
}

impl RequestClassifier for ShapeClassifier {
    fn classify(
        &mut self,
        mut request: ClassifyRequest,
        _workers: WorkerCandidates<'_>,
    ) -> ClassifyFuture {
        let expected = request.expected_output_tokens();
        if let Some(class) = self.parameters.class_for(expected, is_compaction(&request)) {
            request.set_policy_class(class);
        }
        if let Some(cost) = self
            .parameters
            .cost_for(request.scheduling_cost_tokens(), expected)
        {
            request.set_scheduling_cost_tokens(cost);
        }
        Box::pin(async move { Ok(request) })
    }
}

fn validate(parameters: &Parameters) -> Result<(), RequestClassifierProviderError> {
    if parameters.decode_class.is_empty() {
        return Err(RequestClassifierProviderError::new(
            "decode_class must name a configured policy class",
        ));
    }
    if parameters.decode_tokens_threshold == 0 {
        return Err(RequestClassifierProviderError::new(
            "decode_tokens_threshold must be greater than zero",
        ));
    }
    if !(0.0..=64.0).contains(&parameters.decode_cost_weight) {
        return Err(RequestClassifierProviderError::new(
            "decode_cost_weight must be between 0 and 64",
        ));
    }
    Ok(())
}

fn provider(
    parameters: &RequestClassifierParameters,
) -> Result<RequestClassifierFactory, RequestClassifierProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    validate(&parameters)?;
    Ok(Arc::new(move |_context| {
        Box::new(ShapeClassifier {
            parameters: parameters.clone(),
        })
    }))
}

pub(crate) fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), RequestClassifierRegistryError> {
    registry.register_request_classifier("shape", Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters(yaml: &str) -> Parameters {
        serde_yaml::from_str(yaml).unwrap()
    }

    #[test]
    fn long_outputs_join_the_decode_class_and_pay_for_decode() {
        let p = parameters(
            "decode_class: decode_heavy\ndefault_class: agentic\ndecode_tokens_threshold: 1024\ndecode_cost_weight: 1.0\n",
        );
        assert_eq!(p.class_for(Some(4096), false), Some("decode_heavy"));
        assert_eq!(p.cost_for(300, Some(4096)), Some(4396));
        assert_eq!(p.class_for(Some(64), false), Some("agentic"));
        assert_eq!(p.cost_for(6000, Some(64)), Some(6064));
        assert_eq!(p.class_for(None, false), Some("agentic"));
        assert_eq!(p.cost_for(6000, None), None);

        let p = parameters("decode_class: d\ncompaction_class: batch\n");
        assert_eq!(p.class_for(Some(4096), true), Some("batch"));
        assert_eq!(p.class_for(Some(16), false), None);
        assert_eq!(p.cost_for(10, Some(4096)), None);
    }

    #[test]
    fn parameters_are_validated() {
        for bad in [
            "decode_class: ''\n",
            "decode_class: d\ndecode_tokens_threshold: 0\n",
            "decode_class: d\ndecode_cost_weight: 100\n",
        ] {
            assert!(validate(&parameters(bad)).is_err(), "{bad}");
        }
        assert!(serde_yaml::from_str::<Parameters>("decode_class: d\nunknown: 1\n").is_err());
        assert!(validate(&parameters("decode_class: d\n")).is_ok());
    }
}
