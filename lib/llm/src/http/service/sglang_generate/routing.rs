// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native routing with HTTP-lifetime load accounting.

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use axum::{
    http::{HeaderMap, Method},
    response::Response,
};
use bytes::Bytes;
use dynamo_runtime::pipeline::{Context, RouterMode};
use serde_json::{Value, value::RawValue};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::http::service::{
    http_proxy::{HttpClient, forward_reserved, metrics::observe_response},
    metrics::{Endpoint as MetricEndpoint, ErrorType, Metrics, request_was_rejected},
    openai::get_or_create_request_id,
};
use crate::{
    discovery::CommittedWorkerSetTarget,
    kv_router::{RouteReservation, RoutingHost, prefill_router::PrefillRouter},
    model_card::ModelDeploymentCard,
    protocols::{
        common::{
            preprocessor::{PreprocessedRequest, RoutingHints},
            timing::RequestPhase,
        },
        http,
    },
    tokenizers::Tokenizer,
};

pub(crate) struct NativeGenerateBinding {
    target: CommittedWorkerSetTarget,
    cancellation: CancellationToken,
    client: HttpClient,
    host: Arc<RoutingHost>,
    tokenizer: Option<Tokenizer>,
    prefill: Option<Arc<PrefillRouter>>,
}

#[path = "disaggregation.rs"]
mod disaggregation;
use dynamo_kv_router::protocols::RoutingConstraints;

pub(crate) fn supports_native(card: &ModelDeploymentCard) -> bool {
    card.runtime_config
        .runtime_data
        .get(crate::protocols::sglang::HTTP_CAPABILITY)
        == Some(&Value::Bool(true))
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct NativeRequestError(anyhow::Error);

impl NativeGenerateBinding {
    pub(crate) async fn new(
        target: CommittedWorkerSetTarget,
        cancellation: CancellationToken,
        host: Arc<RoutingHost>,
        prefill: Option<Arc<PrefillRouter>>,
    ) -> anyhow::Result<Self> {
        let endpoint = &target.endpoint;
        let wire = endpoint
            .component()
            .endpoint(http::endpoint_name(&endpoint.id().name))
            .client()
            .await?
            .with_admitted_instances_and_cancellation(
                target.admitted_ids.clone(),
                cancellation.clone(),
            );
        Ok(Self {
            cancellation,
            client: HttpClient::from_client(wire, RouterMode::Direct).await?,
            host,
            tokenizer: target
                .card
                .has_tokenizer()
                .then(|| target.card.native_generate_tokenizer())
                .transpose()?,
            target,
            prefill,
        })
    }

    pub(crate) async fn forward(
        &self,
        method: Method,
        headers: HeaderMap,
        body: Bytes,
        metrics: Arc<Metrics>,
        metric_model: &str,
    ) -> anyhow::Result<Response> {
        let projection = Projection::read(&body, &headers).map_err(NativeRequestError)?;
        let streaming = projection
            .field("stream")
            .map_err(NativeRequestError)?
            .unwrap_or(false);
        let metadata = crate::http::service::metadata::extract_metadata_from_http(&headers)?;
        let mut guard = metrics.create_inflight_guard(
            metric_model,
            MetricEndpoint::Generate,
            streaming,
            &get_or_create_request_id(&headers),
        );
        guard.mark_error(ErrorType::Cancelled);
        let request = NativeRequest {
            input: projection,
            metadata,
            deadline: Instant::now() + Duration::from_secs(30),
            wire: http::Request {
                path: "/generate".into(),
                method: method.to_string(),
                headers: http::encode_headers(headers),
                body,
            },
        };
        let response = async {
            if let Some(prefill) = &self.prefill {
                return self.forward_disaggregated(prefill, request).await;
            }
            let admission = self
                .reserve(&request, RequestPhase::Aggregated, Default::default())
                .await?;
            self.send(&request, admission, Vec::new()).await
        }
        .await
        .inspect_err(|error| {
            guard.mark_error(if request_was_rejected(error.as_ref()) {
                ErrorType::Overload
            } else {
                ErrorType::Unavailable
            });
        })?;
        Ok(observe_response(response, guard))
    }
    async fn reserve(
        &self,
        request: &NativeRequest,
        phase: RequestPhase,
        mut constraints: RoutingConstraints,
    ) -> anyhow::Result<RouteReservation> {
        let target = &self.target;
        let config = &target.card.runtime_config;
        let uses_kv = self.host.kv_router_if_enabled().is_some();
        let requested_rank = if phase == RequestPhase::Prefill {
            request
                .input
                .field::<u32>("disagg_prefill_dp_rank")
                .map_err(NativeRequestError)?
                .or_else(|| {
                    (!uses_kv).then(|| {
                        config.data_parallel_start_rank
                            + rand::random::<u32>() % config.data_parallel_size.max(1)
                    })
                })
        } else {
            request.input.dp_rank
        };
        let ranks = config.data_parallel_start_rank
            ..config
                .data_parallel_start_rank
                .saturating_add(config.data_parallel_size);
        anyhow::ensure!(
            requested_rank.is_none_or(|rank| ranks.contains(&rank)),
            NativeRequestError(anyhow::anyhow!("DP rank is outside the admitted WorkerSet"))
        );
        let tokenizer = self.tokenizer.clone();
        let projection = request.input.clone();
        let input = tokio::task::spawn_blocking(move || {
            projection.routing_input(tokenizer.as_ref(), uses_kv)
        })
        .await?
        .map_err(NativeRequestError)?;
        constraints.required_dp_rank = requested_rank;
        let routing = PreprocessedRequest::builder()
            .model(target.card.name().to_string())
            .token_ids(input.0)
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .routing(Some(RoutingHints {
                routing_constraints: Some(constraints),
                ..input.1
            }))
            .build()?;
        let routing = Context::with_id_and_metadata(
            routing,
            uuid::Uuid::new_v4().to_string(),
            request.metadata.clone(),
        );
        let reservation =
            tokio::time::timeout_at(request.deadline, self.host.reserve_route(&routing, phase))
                .await??;
        anyhow::ensure!(
            !self.cancellation.is_cancelled()
                && target
                    .admitted_ids
                    .borrow()
                    .contains(&reservation.target.worker_id),
            "selected native worker is no longer admitted"
        );
        Ok(reservation)
    }
    async fn send(
        &self,
        request: &NativeRequest,
        admission: RouteReservation,
        mut controls: Vec<(&'static str, Value)>,
    ) -> anyhow::Result<Response> {
        if let Some(rank) = admission.target.dp_rank {
            controls.push(("routed_dp_rank", Value::from(rank)));
            controls.push(("data_parallel_rank", Value::from(rank)));
        }
        let mut wire = request.wire.clone();
        if !controls.is_empty() {
            wire.body = request.input.with_controls(controls)?;
        }
        forward_reserved(
            &self.client,
            Context::with_id_and_metadata(
                wire,
                uuid::Uuid::new_v4().to_string(),
                request.metadata.clone(),
            ),
            admission,
        )
        .await
    }
}

#[derive(Clone)]
struct Projection {
    fields: Arc<BTreeMap<String, Box<RawValue>>>,
    dp_rank: Option<u32>,
}

struct NativeRequest {
    input: Projection,
    wire: http::Request,
    metadata: BTreeMap<String, String>,
    deadline: Instant,
}

impl Projection {
    // Retain engine-owned values verbatim; only routing controls are replaced.
    fn with_controls(
        &self,
        controls: impl IntoIterator<Item = (&'static str, Value)>,
    ) -> anyhow::Result<Bytes> {
        let controls = controls
            .into_iter()
            .map(|(key, value)| serde_json::value::to_raw_value(&value).map(|value| (key, value)))
            .collect::<Result<Vec<_>, _>>()?;
        let mut fields: BTreeMap<&str, &RawValue> = self
            .fields
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_ref()))
            .collect();
        fields.extend(controls.iter().map(|(key, value)| (*key, value.as_ref())));
        Ok(serde_json::to_vec(&fields)?.into())
    }

    fn read(body: &[u8], headers: &HeaderMap) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !headers
                .keys()
                .any(|n| n.as_str().starts_with("x-override-")),
            "routing override headers are owned by Dynamo"
        );
        let mut input = Self {
            fields: Arc::new(serde_json::from_slice(body)?),
            dp_rank: None,
        };
        let header_rank = headers
            .get("x-data-parallel-rank")
            .map(|v| -> anyhow::Result<u32> { Ok(v.to_str()?.parse()?) })
            .transpose()?;
        let ranks = [
            input.field("routed_dp_rank")?,
            input.field("data_parallel_rank")?,
            header_rank,
        ];
        let mut ranks = ranks.into_iter().flatten();
        input.dp_rank = ranks.next();
        anyhow::ensure!(
            ranks.all(|r| Some(r) == input.dp_rank),
            "conflicting native DP ranks"
        );
        Ok(input)
    }

    fn raw(&self, key: &str) -> Option<&RawValue> {
        self.fields
            .get(key)
            .filter(|v| v.get() != "null")
            .map(AsRef::as_ref)
    }

    fn field<T: serde::de::DeserializeOwned>(&self, key: &str) -> anyhow::Result<Option<T>> {
        self.raw(key)
            .map(|v| serde_json::from_str(v.get()))
            .transpose()
            .map_err(Into::into)
    }

    fn first<T: serde::de::DeserializeOwned>(&self, key: &str) -> anyhow::Result<Option<T>> {
        let Some(raw) = self.raw(key) else {
            return Ok(None);
        };
        let raw = if raw.get().starts_with('[') {
            serde_json::from_str::<Vec<&RawValue>>(raw.get())?
                .first()
                .copied()
        } else {
            Some(raw)
        };
        raw.map(|v| serde_json::from_str(v.get()))
            .transpose()
            .map_err(Into::into)
    }

    // Route a batch as one HTTP request using its first prompt. SGLang owns
    // sample/beam expansion; accounting follows HTTP rather than child requests.
    fn routing_input(
        self,
        tokenizer: Option<&Tokenizer>,
        uses_kv: bool,
    ) -> anyhow::Result<(Arc<Vec<u32>>, RoutingHints)> {
        anyhow::ensure!(
            self.raw("session_params").is_none(),
            "native sessions are outside this endpoint's scope"
        );
        anyhow::ensure!(
            !uses_kv
                || [
                    "input_embeds",
                    "image_data",
                    "video_data",
                    "audio_data",
                    "extra_key"
                ]
                .iter()
                .all(|key| self.raw(key).is_none()),
            "these inputs require a load-based routing policy"
        );
        let tokens = if !uses_kv {
            Vec::new()
        } else if let Some(ids) = self.field::<Value>("input_ids")? {
            let ids = match ids {
                Value::Array(mut ids) if ids.first().is_some_and(Value::is_array) => {
                    ids.swap_remove(0)
                }
                ids => ids,
            };
            serde_json::from_value(ids)?
        } else {
            let text = self
                .first::<String>("text")?
                .ok_or_else(|| anyhow::anyhow!("KV routing requires text or input_ids"))?;
            tokenizer
                .ok_or_else(|| anyhow::anyhow!("KV routing requires the engine tokenizer"))?
                .encode(&text)?
                .token_ids()
                .to_vec()
        };
        anyhow::ensure!(
            !uses_kv || !tokens.is_empty(),
            "KV routing requires nonempty prompt tokens"
        );
        #[derive(serde::Deserialize, Default)]
        struct SamplingHint {
            max_new_tokens: Option<u32>,
        }
        let priority: i32 = self.field("priority")?.unwrap_or_default();
        Ok((
            Arc::new(tokens),
            RoutingHints {
                expected_output_tokens: self
                    .first::<SamplingHint>("sampling_params")?
                    .unwrap_or_default()
                    .max_new_tokens,
                priority: Some(priority),
                priority_jump: Some(priority.max(0) as f64),
                lora_name: self.first("lora_path")?,
                cache_namespace: self
                    .first::<String>("cache_salt")?
                    .filter(|s| !s.is_empty()),
                ..Default::default()
            },
        ))
    }
}

#[cfg(test)]
#[path = "routing_tests.rs"]
mod tests;
