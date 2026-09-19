// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native routing with HTTP-lifetime load accounting.

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use axum::{
    http::{HeaderMap, Method},
    response::Response,
};
use bytes::Bytes;
use dynamo_runtime::{
    component::Endpoint,
    pipeline::{Context, RouterMode},
};
use serde_json::{Value, value::RawValue};
use tokio::{sync::watch, time::Instant};
use tokio_util::sync::CancellationToken;

use crate::http::service::http_proxy::{HttpClient, forward_with_cancellation};
use crate::{
    kv_router::{RoutingHost, native::NativeReservation, prefill_router::PrefillRouter},
    model_card::ModelDeploymentCard,
    protocols::{
        common::{
            OutputOptions, SamplingOptions, StopConditions,
            preprocessor::{PreprocessedRequest, RoutingHints},
            timing::RequestPhase,
        },
        http,
    },
    tokenizers::Tokenizer,
};

pub(crate) struct NativeGenerateBinding {
    admitted_ids: watch::Receiver<Vec<u64>>,
    cancellation: CancellationToken,
    client: HttpClient,
    host: Arc<RoutingHost>,
    tokenizer: Option<Tokenizer>,
    card: ModelDeploymentCard,
    prefill: Option<Arc<PrefillRouter>>,
}

#[path = "disaggregation.rs"]
mod disaggregation;
#[derive(Default)]
struct NativeConstraints {
    routing: dynamo_kv_router::protocols::RoutingConstraints,
}

fn check_owned_headers(headers: &HeaderMap) -> anyhow::Result<()> {
    anyhow::ensure!(
        !headers
            .keys()
            .any(|name| name.as_str().starts_with("x-override-")),
        "native routing override headers are owned by Dynamo"
    );
    Ok(())
}

pub(crate) fn supports_native(card: &ModelDeploymentCard) -> bool {
    card.runtime_config
        .runtime_data
        .get(crate::protocols::sglang::HTTP_CAPABILITY)
        .and_then(Value::as_bool)
        == Some(true)
}

async fn forward_reserved(
    client: &HttpClient,
    request: dynamo_runtime::pipeline::SingleIn<http::Request>,
    admission: NativeReservation,
    cancellation: CancellationToken,
) -> anyhow::Result<Response> {
    let worker_id = admission.target().worker_id;
    let load = admission.start(cancellation.clone());
    let response =
        forward_with_cancellation(client, request, worker_id, cancellation.clone()).await?;
    let (parts, body) = response.into_parts();
    let stream = async_stream::try_stream! {
        let _load = load;
        let mut stream = body.into_data_stream();
        use futures::StreamExt;
        loop {
            let chunk = tokio::select! {
                biased;
                _ = cancellation.cancelled() => Err(anyhow::anyhow!("native HTTP request cancelled")),
                chunk = stream.next() => Ok(chunk),
            }?;
            let Some(chunk) = chunk else { break; };
            yield chunk?;
        }
    };
    Ok(Response::from_parts(
        parts,
        axum::body::Body::from_stream(
            Box::pin(stream) as futures::stream::BoxStream<'static, anyhow::Result<Bytes>>
        ),
    ))
}

// Rewrite routing controls without parsing engine-owned numeric extensions.
fn with_controls(
    body: &Bytes,
    controls: impl IntoIterator<Item = (&'static str, Value)>,
) -> anyhow::Result<Bytes> {
    let mut fields: BTreeMap<String, Box<RawValue>> = serde_json::from_slice(body)?;
    for (key, value) in controls {
        fields.insert(key.into(), serde_json::value::to_raw_value(&value)?);
    }
    Ok(serde_json::to_vec(&fields)?.into())
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct NativeRequestError(anyhow::Error);

// Keep the prefill booking alive while decode selection is queued.
async fn admission_wait<T>(
    reservation: &NativeReservation,
    deadline: Instant,
    operation: impl std::future::Future<Output = anyhow::Result<T>>,
) -> anyhow::Result<T> {
    let expiry = dynamo_kv_router::multi_worker_sequence::active_request_expiry_duration();
    let mut heartbeat = tokio::time::interval((expiry / 3).min(Duration::from_secs(5)));
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    tokio::pin!(operation);
    loop {
        tokio::select! {
            result = &mut operation => return result,
            _ = tokio::time::sleep_until(deadline) => anyhow::bail!("native batch admission timed out"),
            _ = heartbeat.tick() => {
                reservation.touch()?;
            }
        }
    }
}

impl NativeGenerateBinding {
    pub(crate) async fn new(
        endpoint: &Endpoint,
        admitted_ids: watch::Receiver<Vec<u64>>,
        cancellation: CancellationToken,
        host: Arc<RoutingHost>,
        card: &ModelDeploymentCard,
        prefill: Option<Arc<PrefillRouter>>,
    ) -> anyhow::Result<Self> {
        let wire = endpoint
            .component()
            .endpoint(http::endpoint_name(&endpoint.id().name))
            .client()
            .await?
            .with_admitted_instances_and_cancellation(admitted_ids.clone(), cancellation.clone());
        Ok(Self {
            admitted_ids,
            cancellation,
            client: HttpClient::from_client(wire, RouterMode::Direct).await?,
            host,
            tokenizer: if card.has_tokenizer() {
                Some(card.native_generate_tokenizer()?)
            } else {
                None
            },
            card: card.clone(),
            prefill,
        })
    }

    pub(crate) async fn forward(
        &self,
        method: Method,
        headers: HeaderMap,
        body: Bytes,
        metrics: Arc<crate::http::service::metrics::Metrics>,
        metric_model: &str,
    ) -> anyhow::Result<Response> {
        anyhow::ensure!(
            !self.cancellation.is_cancelled() && !self.admitted_ids.borrow().is_empty(),
            "native WorkerSet is retired or unavailable"
        );
        let prepared = async {
            check_owned_headers(&headers)?;
            let projection = Projection::read(&body)?;
            let streaming = projection.field("stream")?.unwrap_or(false);
            let uses_kv = self.host.kv_router_if_enabled().is_some();
            anyhow::ensure!(
                !uses_kv || !projection.engine_processed_input,
                "KV routing requires engine-compatible token metadata for multimodal or embedding inputs; select a load-based policy explicitly"
            );
            let header_rank = headers
                .get("x-data-parallel-rank")
                .map(|value| -> anyhow::Result<u32> { Ok(value.to_str()?.parse()?) })
                .transpose()?;
            anyhow::ensure!(
                header_rank
                    .zip(projection.dp_rank)
                    .is_none_or(|(a, b)| a == b),
                "conflicting body and header DP ranks"
            );
            let requested_rank = projection.dp_rank.or(header_rank);
            let config = &self.card.runtime_config;
            let default_rank = requested_rank.unwrap_or(config.data_parallel_start_rank);
            anyhow::ensure!(
                (config.data_parallel_start_rank
                    ..config
                        .data_parallel_start_rank
                        .saturating_add(config.data_parallel_size))
                    .contains(&default_rank),
                "requested DP rank is outside the admitted WorkerSet"
            );
            let tokenizer = self.tokenizer.clone();
            let projection_for_routing = projection.clone();
            let input = tokio::task::spawn_blocking(move || {
                projection_for_routing.routing_input(tokenizer.as_ref(), uses_kv)
            })
            .await??;
            let metadata = crate::http::service::metadata::extract_metadata_from_http(&headers)?;
            anyhow::Ok((input, requested_rank, metadata, streaming, projection))
        };
        let (input, requested_rank, metadata, streaming, projection) =
            prepared.await.map_err(NativeRequestError)?;
        let mut guard = metrics.create_inflight_guard(
            metric_model,
            crate::http::service::metrics::Endpoint::Generate,
            streaming,
            &crate::http::service::openai::get_or_create_request_id(&headers),
        );
        // Dropping the HTTP future or response records cancellation. The
        // HTTP response body owns its routing reservations.
        guard.mark_error(crate::http::service::metrics::ErrorType::Cancelled);
        let result = async {
            let deadline = Instant::now() + Duration::from_secs(30);
            if let Some(prefill) = &self.prefill {
                return self
                    .forward_disaggregated(
                        prefill,
                        method,
                        headers,
                        body,
                        metadata,
                        projection,
                        input,
                        requested_rank,
                        deadline,
                    )
                    .await;
            }
            let admission = self
                .reserve(
                    input,
                    requested_rank,
                    &metadata,
                    RequestPhase::Aggregated,
                    deadline,
                    NativeConstraints::default(),
                )
                .await?;
            let body = if let Some(rank) = admission.target().dp_rank {
                with_controls(
                    &body,
                    [
                        ("routed_dp_rank", Value::from(rank)),
                        ("data_parallel_rank", Value::from(rank)),
                    ],
                )?
            } else {
                body
            };
            let request = http::Request {
                path: "/generate".into(),
                method: method.to_string(),
                headers: http::encode_headers(headers),
                body,
            };
            forward_reserved(
                &self.client,
                Context::with_id_and_metadata(request, uuid::Uuid::new_v4().to_string(), metadata),
                admission,
                CancellationToken::new(),
            )
            .await
        }
        .await;
        match result {
            Ok(response) => Ok(crate::http::service::http_proxy::metrics::observe_response(
                response, guard,
            )),
            Err(error) => {
                use crate::http::service::metrics::{ErrorType, request_was_rejected};
                guard.mark_error(if request_was_rejected(error.as_ref()) {
                    ErrorType::Overload
                } else {
                    ErrorType::Unavailable
                });
                Err(error)
            }
        }
    }
    async fn reserve(
        &self,
        input: RoutingInput,
        requested_rank: Option<u32>,
        metadata: &BTreeMap<String, String>,
        phase: RequestPhase,
        deadline: Instant,
        mut constraints: NativeConstraints,
    ) -> anyhow::Result<NativeReservation> {
        constraints.routing.required_dp_rank = requested_rank;
        let request = PreprocessedRequest::builder()
            .model(self.card.name().to_string())
            .token_ids(input.tokens)
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .routing(Some(RoutingHints {
                expected_output_tokens: input.max_tokens,
                routing_constraints: Some(constraints.routing),
                priority_jump: Some(input.priority.max(0) as f64),
                priority: Some(input.priority),
                lora_name: input.lora,
                cache_namespace: input.cache_namespace,
                ..Default::default()
            }))
            .build()?;
        let request = Context::with_id_and_metadata(
            request,
            uuid::Uuid::new_v4().to_string(),
            metadata.clone(),
        );
        let reservation = tokio::time::timeout_at(
            deadline,
            self.host
                .reserve_native(&request, None, requested_rank, phase),
        )
        .await??;
        anyhow::ensure!(
            !self.cancellation.is_cancelled()
                && self
                    .admitted_ids
                    .borrow()
                    .contains(&reservation.target().worker_id),
            "selected native worker is no longer admitted"
        );
        Ok(reservation)
    }
}

#[derive(Clone)]
struct Projection {
    fields: BTreeMap<String, Box<RawValue>>,
    dp_rank: Option<u32>,
    engine_processed_input: bool,
}

struct RoutingInput {
    tokens: Arc<Vec<u32>>,
    max_tokens: Option<u32>,
    priority: i32,
    lora: Option<String>,
    cache_namespace: Option<String>,
}

impl Projection {
    fn read(body: &[u8]) -> anyhow::Result<Self> {
        let mut input = Self {
            fields: serde_json::from_slice(body)?,
            dp_rank: None,
            engine_processed_input: false,
        };
        let rank = input.field::<u32>("routed_dp_rank")?;
        let alias = input.field::<u32>("data_parallel_rank")?;
        anyhow::ensure!(
            rank.zip(alias).is_none_or(|(a, b)| a == b),
            "conflicting native DP ranks"
        );
        input.dp_rank = rank.or(alias);
        input.engine_processed_input = ["input_embeds", "image_data", "video_data", "audio_data"]
            .iter()
            .any(|key| input.raw(key).is_some());
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
    ) -> anyhow::Result<RoutingInput> {
        anyhow::ensure!(
            self.raw("session_params").is_none(),
            "native sessions are outside this endpoint's scope"
        );
        anyhow::ensure!(
            !uses_kv || (!self.engine_processed_input && self.raw("extra_key").is_none()),
            "these inputs require a load-based routing policy"
        );
        let tokens = if !uses_kv {
            Vec::new()
        } else if let Some(ids) = self.field::<Value>("input_ids")? {
            let ids = if ids
                .as_array()
                .and_then(|ids| ids.first())
                .is_some_and(Value::is_array)
            {
                ids[0].clone()
            } else {
                ids
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
        Ok(RoutingInput {
            tokens: Arc::new(tokens),
            max_tokens: self
                .first::<SamplingHint>("sampling_params")?
                .unwrap_or_default()
                .max_new_tokens,
            priority: self.field("priority")?.unwrap_or_default(),
            lora: self.first("lora_path")?,
            cache_namespace: self
                .first::<String>("cache_salt")?
                .filter(|s| !s.is_empty()),
        })
    }
}

#[cfg(test)]
#[path = "routing_tests.rs"]
mod tests;
