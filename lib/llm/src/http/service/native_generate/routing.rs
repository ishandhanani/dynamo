// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native request routing projection. This never reconstructs a generation body.

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

use super::{
    NativeGenerateClient, forward_accounted,
    lifecycle::{LifecycleClient, NativeAttempt, ReservedChild},
};
use crate::{
    kv_router::{RoutingHost, prefill_router::PrefillRouter},
    model_card::ModelDeploymentCard,
    protocols::{
        common::{
            OutputOptions, SamplingOptions, StopConditions,
            preprocessor::{PreprocessedRequest, RoutingHints},
            timing::RequestPhase,
        },
        sglang::http::{
            self,
            lifecycle::{ChildKind, Descriptor},
        },
    },
    tokenizers::Tokenizer,
};

pub(crate) struct NativeGenerateBinding {
    admitted_ids: watch::Receiver<Vec<u64>>,
    cancellation: CancellationToken,
    client: NativeGenerateClient,
    control: Arc<LifecycleClient>,
    host: Arc<RoutingHost>,
    tokenizer: Option<Tokenizer>,
    card: ModelDeploymentCard,
    prefill: Option<Arc<PrefillRouter>>,
}

#[path = "disaggregation.rs"]
mod disaggregation;

pub(crate) fn supports_native(card: &ModelDeploymentCard) -> bool {
    [http::CAPABILITY, http::lifecycle::CAPABILITY]
        .iter()
        .all(|key| {
            card.runtime_config
                .runtime_data
                .get(*key)
                .and_then(Value::as_bool)
                == Some(true)
        })
}

struct Admission {
    reservations: Vec<ReservedChild>,
    descriptor: Descriptor,
}

impl Admission {
    fn target(&self) -> crate::session_affinity::AffinityTarget {
        self.reservations[0].reservation.target()
    }

    fn into_attempt(
        self,
        binding: &NativeGenerateBinding,
        stage: &str,
    ) -> anyhow::Result<NativeAttempt> {
        NativeAttempt::new(
            binding.control.clone(),
            self.descriptor,
            stage.to_string(),
            self.reservations,
        )
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct NativeRequestError(anyhow::Error);

// Renew already-admitted children while another child waits in the router queue
// or descriptor discovery waits for a worker. No request has been dispatched yet.
async fn admission_wait<T>(
    reservations: &[ReservedChild],
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
                for child in reservations { child.touch()?; }
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
        let control = endpoint
            .component()
            .endpoint(http::lifecycle::endpoint_name(&endpoint.id().name))
            .client()
            .await?
            .with_admitted_instances_and_cancellation(admitted_ids.clone(), cancellation.clone());
        Ok(Self {
            admitted_ids,
            cancellation,
            client: NativeGenerateClient::from_client(wire, RouterMode::Direct).await?,
            // Recovery must keep querying after transient transport failures;
            // the retained admission receiver still fences retired generations.
            control: Arc::new(
                LifecycleClient::from_client_no_fault_detection(control, RouterMode::Direct)
                    .await?,
            ),
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
            anyhow::ensure!(
                !headers
                    .keys()
                    .any(|name| name.as_str().starts_with("x-override-")
                        || name == http::lifecycle::ATTEMPT_HEADER
                        || name == http::lifecycle::INCARNATION_HEADER),
                "native routing and lifecycle override headers are owned by Dynamo"
            );
            let projection = Projection::read(&body)?;
            if self.prefill.is_none() {
                projection.require_aggregated()?;
            }
            let streaming = projection
                .value
                .get("stream")
                .and_then(Value::as_bool)
                .unwrap_or(false);
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
            let child_projection = projection.clone();
            let disaggregated = self.prefill.is_some();
            let children = tokio::task::spawn_blocking(move || {
                child_projection.children(tokenizer.as_ref(), uses_kv, disaggregated)
            })
            .await??;
            let metadata = crate::http::service::metadata::extract_metadata_from_http(&headers)?;
            anyhow::Ok((children, requested_rank, metadata, streaming, projection))
        };
        let (children, requested_rank, metadata, streaming, projection) =
            prepared.await.map_err(NativeRequestError)?;
        let mut guard = metrics.create_inflight_guard(
            metric_model,
            crate::http::service::metrics::Endpoint::Generate,
            streaming,
            &crate::http::service::openai::get_or_create_request_id(&headers),
        );
        // Dropping the HTTP future or response records cancellation. The
        // independent lifecycle owner still retains its engine reservations.
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
                        children,
                        requested_rank,
                        deadline,
                    )
                    .await;
            }
            let admission = self
                .reserve(
                    children,
                    requested_rank,
                    &metadata,
                    RequestPhase::Aggregated,
                    deadline,
                    Default::default(),
                )
                .await?;
            let attempt = admission.into_attempt(self, "null")?;
            let request = http::Request {
                method: method.to_string(),
                headers: http::encode_headers(headers),
                body,
                operation: Default::default(),
            };
            forward_accounted(
                &self.client,
                Context::with_id_and_metadata(request, uuid::Uuid::new_v4().to_string(), metadata),
                attempt,
            )
            .await
        }
        .await;
        match result {
            Ok(response) => Ok(super::metrics::observe_response(response, guard)),
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
        children: Vec<Child>,
        requested_rank: Option<u32>,
        metadata: &BTreeMap<String, String>,
        phase: RequestPhase,
        deadline: Instant,
        mut constraints: dynamo_kv_router::protocols::RoutingConstraints,
    ) -> anyhow::Result<Admission> {
        constraints.required_dp_rank = requested_rank;
        let mut reservations: Vec<ReservedChild> = Vec::with_capacity(children.len());
        let mut worker = None;
        for child in children {
            let request = PreprocessedRequest::builder()
                .model(self.card.name().to_string())
                .token_ids(child.tokens)
                .stop_conditions(StopConditions::default())
                .sampling_options(SamplingOptions::default())
                .output_options(OutputOptions::default())
                .routing(Some(RoutingHints {
                    expected_output_tokens: child.max_tokens,
                    routing_constraints: Some(constraints.clone()),
                    priority_jump: Some(child.priority.max(0) as f64),
                    priority: Some(child.priority),
                    lora_name: child.lora,
                    cache_namespace: child.cache_namespace,
                    ..Default::default()
                }))
                .build()?;
            let width = if phase == RequestPhase::Prefill {
                1
            } else {
                child.decode_width
            };
            for row in 0..width {
                let mut row_request = request.clone();
                if row > 0 || phase == RequestPhase::Decode {
                    // Beam rows share the leader's prompt, but each owns
                    // decode capacity. Do not book prefill tokens twice.
                    row_request
                        .router_config_override
                        .get_or_insert_default()
                        .track_prefill_tokens = Some(false);
                }
                let request = Context::with_id_and_metadata(
                    row_request,
                    uuid::Uuid::new_v4().to_string(),
                    metadata.clone(),
                );
                let reservation = admission_wait(
                    &reservations,
                    deadline,
                    self.host
                        .reserve_native(&request, worker, requested_rank, phase),
                )
                .await?;
                anyhow::ensure!(
                    requested_rank.is_none_or(|rank| Some(rank) == reservation.target().dp_rank),
                    "configured KV policy selected a different DP rank than the native request constraint"
                );
                worker = Some(reservation.target());
                if row == 0 {
                    reservations.push(ReservedChild {
                        kind: child.kind,
                        reservation,
                        additional_reservations: Vec::with_capacity(child.decode_width - 1),
                    });
                } else {
                    reservations
                        .last_mut()
                        .unwrap()
                        .additional_reservations
                        .push(reservation);
                }
            }
        }
        let worker =
            worker.ok_or_else(|| anyhow::anyhow!("native request has no routable children"))?;
        let descriptor = admission_wait(
            &reservations,
            deadline,
            NativeAttempt::describe(&self.control, worker.worker_id),
        )
        .await?;
        anyhow::ensure!(
            phase == RequestPhase::Aggregated || descriptor.native_disaggregation_version == 1,
            "selected engine does not support native P/D v1"
        );
        anyhow::ensure!(
            !self.cancellation.is_cancelled()
                && self.admitted_ids.borrow().contains(&worker.worker_id),
            "selected native worker is no longer admitted"
        );
        Ok(Admission {
            reservations,
            descriptor,
        })
    }
}

#[derive(Clone)]
struct Projection {
    value: Value,
    dp_rank: Option<u32>,
    engine_processed_input: bool,
}

struct Child {
    kind: ChildKind,
    decode_width: usize,
    tokens: Arc<Vec<u32>>,
    max_tokens: Option<u32>,
    priority: i32,
    lora: Option<String>,
    cache_namespace: Option<String>,
}

impl Projection {
    fn read(body: &[u8]) -> anyhow::Result<Self> {
        // Borrow raw fields before inspecting routing inputs. Parsing the entire
        // request as Value rejects engine extensions such as an unused 1e400.
        // Keep duplicate-key behavior (last value wins) aligned with SGLang.
        let fields: BTreeMap<String, &RawValue> = serde_json::from_slice(body)?;
        let mut value = serde_json::Map::new();
        for key in [
            "text",
            "input_ids",
            "input_embeds",
            "image_data",
            "video_data",
            "audio_data",
            "stream",
            "routed_dp_rank",
            "data_parallel_rank",
            "priority",
            "lora_path",
            "cache_salt",
            "extra_key",
            "session_params",
            "bootstrap_host",
            "bootstrap_port",
            "bootstrap_room",
            "disagg_prefill_dp_rank",
        ] {
            if let Some(raw) = fields.get(key) {
                value.insert(key.into(), serde_json::from_str(raw.get())?);
            }
        }
        if let Some(raw) = fields.get("sampling_params") {
            let project = |raw: &RawValue| -> anyhow::Result<Value> {
                if raw.get() == "null" {
                    return Ok(Value::Null);
                }
                let fields: BTreeMap<String, &RawValue> = serde_json::from_str(raw.get())?;
                let mut value = serde_json::Map::new();
                for key in ["n", "max_new_tokens", "beam_width"] {
                    if let Some(raw) = fields.get(key) {
                        value.insert(key.into(), serde_json::from_str(raw.get())?);
                    }
                }
                Ok(Value::Object(value))
            };
            let params = if raw.get().starts_with('[') {
                let items: Vec<&RawValue> = serde_json::from_str(raw.get())?;
                Value::Array(
                    items
                        .into_iter()
                        .map(project)
                        .collect::<anyhow::Result<_>>()?,
                )
            } else {
                project(raw)?
            };
            value.insert("sampling_params".into(), params);
        }
        let value = Value::Object(value);
        anyhow::ensure!(
            value.get("session_params").is_none_or(Value::is_null),
            "native sessions require an owner binding"
        );
        let rank = |field: &str| {
            value
                .get(field)
                .filter(|v| !v.is_null())
                .map(|v| serde_json::from_value::<u32>(v.clone()))
                .transpose()
        };
        let dp_rank = rank("routed_dp_rank")?;
        let alias = rank("data_parallel_rank")?;
        anyhow::ensure!(
            dp_rank.zip(alias).is_none_or(|(a, b)| a == b),
            "conflicting native DP ranks"
        );
        let engine_processed_input = ["input_embeds", "image_data", "video_data", "audio_data"]
            .iter()
            .any(|key| value.get(key).is_some_and(|v| !v.is_null()));
        Ok(Self {
            value,
            dp_rank: dp_rank.or(alias),
            engine_processed_input,
        })
    }

    fn require_aggregated(&self) -> anyhow::Result<()> {
        let value = &self.value;
        for field in [
            "bootstrap_host",
            "bootstrap_port",
            "bootstrap_room",
            "disagg_prefill_dp_rank",
        ] {
            anyhow::ensure!(
                value.get(field).is_none_or(Value::is_null),
                "{field} requires a disaggregated native route"
            );
        }
        Ok(())
    }

    fn children(
        self,
        tokenizer: Option<&Tokenizer>,
        uses_kv: bool,
        disaggregated: bool,
    ) -> anyhow::Result<Vec<Child>> {
        anyhow::ensure!(
            !uses_kv || !self.engine_processed_input,
            "KV routing requires engine-compatible token metadata for multimodal or embedding inputs"
        );
        let v = &self.value;
        anyhow::ensure!(
            !uses_kv || v.get("extra_key").is_none_or(Value::is_null),
            "KV routing requires compatible metadata for extra_key; select a load-based policy explicitly"
        );
        let prompts: Vec<Vec<u32>> = if self.engine_processed_input && !uses_kv {
            let count = v
                .get("text")
                .and_then(Value::as_array)
                .map(Vec::len)
                .or_else(|| {
                    v.get("input_ids")
                        .and_then(Value::as_array)
                        .filter(|ids| ids.first().is_some_and(Value::is_array))
                        .map(Vec::len)
                })
                .or_else(|| {
                    v.get("input_embeds")
                        .and_then(Value::as_array)
                        .filter(|items| {
                            items
                                .first()
                                .and_then(Value::as_array)
                                .and_then(|item| item.first())
                                .is_some_and(Value::is_array)
                        })
                        .map(Vec::len)
                })
                .unwrap_or(1);
            vec![Vec::new(); count]
        } else if let Some(ids) = v.get("input_ids").filter(|v| !v.is_null()) {
            if ids
                .as_array()
                .and_then(|ids| ids.first())
                .is_some_and(Value::is_array)
            {
                serde_json::from_value(ids.clone())?
            } else {
                vec![serde_json::from_value(ids.clone())?]
            }
        } else {
            let text = v
                .get("text")
                .ok_or_else(|| anyhow::anyhow!("native routing requires input_ids or text"))?;
            let texts: Vec<String> = if text.is_array() {
                serde_json::from_value(text.clone())?
            } else {
                vec![serde_json::from_value(text.clone())?]
            };
            if uses_kv {
                let tokenizer = tokenizer.ok_or_else(|| {
                    anyhow::anyhow!("native text KV routing requires the engine tokenizer")
                })?;
                texts
                    .into_iter()
                    .map(|text| Ok(tokenizer.encode(&text)?.token_ids().to_vec()))
                    .collect::<anyhow::Result<_>>()?
            } else {
                vec![Vec::new(); texts.len()]
            }
        };
        anyhow::ensure!(
            !uses_kv || prompts.iter().all(|tokens| !tokens.is_empty()),
            "native KV routing requires nonempty effective prompt tokens"
        );
        let params = v.get("sampling_params");
        let parameters: Vec<&Value> = match params {
            Some(Value::Array(items)) => {
                anyhow::ensure!(
                    items.len() == prompts.len(),
                    "sampling_params must match the native batch"
                );
                items.iter().collect()
            }
            value => vec![value.unwrap_or(&Value::Null); prompts.len()],
        };
        let count = |p: &Value| -> anyhow::Result<usize> {
            Ok(p.get("n")
                .map(|v| serde_json::from_value(v.clone()))
                .transpose()?
                .unwrap_or(1))
        };
        let n = parameters
            .first()
            .map(|p| count(p))
            .transpose()?
            .unwrap_or(1);
        anyhow::ensure!(
            n > 0
                && parameters
                    .iter()
                    .all(|p| count(p).is_ok_and(|other| other == n)),
            "native batch sampling requires one positive n"
        );
        let width = |p: &Value| -> anyhow::Result<usize> {
            Ok(p.get("beam_width")
                .filter(|v| !v.is_null())
                .map(|v| serde_json::from_value::<usize>(v.clone()))
                .transpose()?
                .unwrap_or(1)
                .max(1))
        };
        // Native beam normalization uses the first prompt's beam setting.
        // `n` is the number of returned beams, not parallel sampling fan-out.
        let n = if parameters
            .first()
            .map(|p| width(p))
            .transpose()?
            .unwrap_or(1)
            > 1
        {
            1
        } else {
            n
        };
        let warmups = n > 1 && !disaggregated;
        let slots = prompts
            .len()
            .checked_mul(
                n.checked_add(usize::from(warmups))
                    .ok_or_else(|| anyhow::anyhow!("native child count overflow"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("native child count overflow"))?;
        anyhow::ensure!(
            slots > 0 && slots <= 4096,
            "native request exceeds lifecycle child capacity"
        );
        let mut rows = if warmups { prompts.len() } else { 0 };
        for params in &parameters {
            rows = width(params)?
                .checked_mul(n)
                .and_then(|count| rows.checked_add(count))
                .ok_or_else(|| anyhow::anyhow!("native decode row count overflow"))?;
        }
        anyhow::ensure!(rows <= 4096, "native request exceeds decode row capacity");
        let priority = v
            .get("priority")
            .filter(|v| !v.is_null())
            .map(|p| serde_json::from_value::<i32>(p.clone()))
            .transpose()?
            .unwrap_or_default();
        let per_prompt = |field: &str| -> anyhow::Result<Vec<Option<String>>> {
            match v.get(field) {
                Some(Value::Array(values)) => {
                    anyhow::ensure!(
                        values.len() == prompts.len(),
                        "{field} must match the native batch"
                    );
                    Ok(serde_json::from_value(Value::Array(values.clone()))?)
                }
                value => Ok(vec![
                    value
                        .filter(|v| !v.is_null())
                        .map(|v| serde_json::from_value::<String>(v.clone()))
                        .transpose()?;
                    prompts.len()
                ]),
            }
        };
        let loras = per_prompt("lora_path")?;
        let salts: Vec<_> = per_prompt("cache_salt")?
            .into_iter()
            .map(|salt| salt.filter(|s| !s.is_empty()))
            .collect();
        let prompts: Vec<_> = prompts.into_iter().map(Arc::new).collect();
        let mut children = Vec::with_capacity(slots);
        if warmups {
            for ((tokens, lora), salt) in prompts.iter().zip(&loras).zip(&salts) {
                children.push(Child {
                    kind: ChildKind::Warmup,
                    decode_width: 1,
                    tokens: tokens.clone(),
                    max_tokens: Some(0),
                    priority,
                    lora: lora.clone(),
                    cache_namespace: salt.clone(),
                });
            }
        }
        for (((tokens, params), lora), salt) in
            prompts.into_iter().zip(parameters).zip(loras).zip(salts)
        {
            let max_tokens = params
                .get("max_new_tokens")
                .filter(|v| !v.is_null())
                .map(|v| serde_json::from_value::<u32>(v.clone()))
                .transpose()?;
            for _ in 0..n {
                children.push(Child {
                    kind: ChildKind::Sample,
                    decode_width: width(params)?,
                    tokens: tokens.clone(),
                    max_tokens,
                    priority,
                    lora: lora.clone(),
                    cache_namespace: salt.clone(),
                });
            }
        }
        Ok(children)
    }
}

#[cfg(test)]
#[path = "routing_tests.rs"]
mod tests;
