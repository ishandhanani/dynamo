// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
use axum::body::Body;
use futures::StreamExt;

use super::*;

#[path = "logprobs.rs"]
mod logprobs;

impl NativeGenerateBinding {
    #[expect(clippy::too_many_arguments)]
    pub(super) async fn forward_disaggregated(
        &self,
        router: &PrefillRouter,
        method: Method,
        mut headers: HeaderMap,
        body: Bytes,
        metadata: BTreeMap<String, String>,
        projection: Projection,
        input: RoutingInput,
        requested_rank: Option<u32>,
        deadline: Instant,
    ) -> anyhow::Result<Response> {
        let (prefill, endpoint) = router.native_binding()?;
        let uses_kv = prefill.host.kv_router_if_enabled().is_some();
        let config = &prefill.card.runtime_config;
        let explicit_rank = projection
            .field::<u32>("disagg_prefill_dp_rank")
            .map_err(NativeRequestError)?;
        // A load-based worker policy delegates ranks in aggregated mode. P/D
        // needs a known prefill rank before either stage starts its rendezvous.
        let prefill_rank = explicit_rank.or_else(|| {
            (!uses_kv).then(|| {
                config.data_parallel_start_rank
                    + (uuid::Uuid::new_v4().as_u128()
                        % u128::from(config.data_parallel_size.max(1))) as u32
            })
        });
        anyhow::ensure!(
            prefill_rank.is_none_or(|rank| rank >= config.data_parallel_start_rank
                && rank - config.data_parallel_start_rank < config.data_parallel_size),
            NativeRequestError(anyhow::anyhow!(
                "native prefill DP rank is outside the admitted WorkerSet"
            ))
        );
        let tokenizer = prefill.tokenizer.clone();
        let child_projection = projection.clone();
        let prefill_input = tokio::task::spawn_blocking(move || {
            child_projection.routing_input(tokenizer.as_ref(), uses_kv)
        })
        .await?
        .map_err(NativeRequestError)?;
        let prefill_admission = prefill
            .reserve(
                prefill_input,
                prefill_rank,
                &metadata,
                RequestPhase::Prefill,
                deadline,
                Default::default(),
            )
            .await?;
        let target = prefill_admission.target();
        let rank = target
            .dp_rank
            .ok_or_else(|| anyhow::anyhow!("native prefill selection did not choose a DP rank"))?;
        let (bootstrap, constraints) = router.native_bootstrap(&endpoint, target.worker_id)?;
        let host = bootstrap
            .bootstrap_host
            .ok_or_else(|| anyhow::anyhow!("native prefill bootstrap host is missing"))?;
        let port = bootstrap
            .bootstrap_port
            .ok_or_else(|| anyhow::anyhow!("native prefill bootstrap port is missing"))?;
        if constraints.is_some() && self.host.kv_router_if_enabled().is_none() {
            anyhow::bail!("native topology-constrained decode requires KV routing");
        }
        let decode_admission = admission_wait(
            &prefill_admission,
            deadline,
            self.reserve(
                input,
                requested_rank,
                &metadata,
                RequestPhase::Decode,
                deadline,
                NativeConstraints {
                    routing: constraints.unwrap_or_default(),
                    ..Default::default()
                },
            ),
        )
        .await?;
        let room = projection
            .field::<Value>("bootstrap_room")?
            .unwrap_or_else(|| Value::from(bootstrap_room()));
        let body = with_controls(
            &body,
            [
                ("bootstrap_host", Value::from(host)),
                ("bootstrap_port", Value::from(port)),
                ("bootstrap_room", room),
                ("disagg_prefill_dp_rank", Value::from(rank)),
            ],
        )?;
        let logprobs = projection
            .field::<Value>("return_logprob")?
            .unwrap_or(Value::Null);
        let return_logprob = logprobs.as_bool() == Some(true)
            || logprobs
                .as_array()
                .is_some_and(|v| v.iter().any(|v| v.as_bool() == Some(true)));
        if return_logprob {
            // The P/D logprob adapter needs JSON/SSE, not a compressed body.
            headers.insert(axum::http::header::ACCEPT_ENCODING, "identity".parse()?);
        }
        // SGLang gives routed_dp_rank precedence over its deprecated alias.
        // Set both: the client's decode rank may differ from the prefill rank.
        let prefill_request = http::Request {
            path: "/generate".into(),
            method: method.to_string(),
            headers: http::encode_headers(headers.clone()),
            body: with_controls(
                &body,
                [
                    ("routed_dp_rank", Value::from(rank)),
                    ("data_parallel_rank", Value::from(rank)),
                ],
            )?,
        };
        let decode_body = match decode_admission.target().dp_rank {
            Some(rank) => with_controls(
                &body,
                [
                    ("routed_dp_rank", Value::from(rank)),
                    ("data_parallel_rank", Value::from(rank)),
                ],
            )?,
            None => body,
        };
        let decode_request = http::Request {
            path: "/generate".into(),
            method: method.to_string(),
            headers: http::encode_headers(headers),
            body: decode_body,
        };
        let prefill = forward_reserved(
            &prefill.client,
            Context::with_id_and_metadata(
                prefill_request,
                uuid::Uuid::new_v4().to_string(),
                metadata.clone(),
            ),
            prefill_admission,
            CancellationToken::new(),
        );
        let decode = forward_reserved(
            &self.client,
            Context::with_id_and_metadata(
                decode_request,
                uuid::Uuid::new_v4().to_string(),
                metadata,
            ),
            decode_admission,
            CancellationToken::new(),
        );
        let (decode, prompt) = dispatch(prefill, decode, return_logprob).await?;
        match prompt {
            Some(prompt) => logprobs::merge_response(decode, prompt).await,
            None => Ok(decode),
        }
    }
}

fn bootstrap_room() -> u64 {
    // Stock SGLang's torch scalar setter requires an int64-compatible value.
    (uuid::Uuid::new_v4().as_u128() as u64) & (i64::MAX as u64) & !4095
}

// Poll both stages concurrently: prefill may need the decode receiver before
// completing. Each future/response owns its HTTP connection and routing guard;
// an error drops its peer without cancelling the returned rejection body.
async fn dispatch(
    prefill: impl std::future::Future<Output = anyhow::Result<Response>>,
    decode: impl std::future::Future<Output = anyhow::Result<Response>>,
    collect: bool,
) -> anyhow::Result<(Response, Option<Bytes>)> {
    let prefill = async {
        let response = prefill.await?;
        if !response.status().is_success() {
            return Ok::<_, anyhow::Error>(Err(response));
        }
        let mut body = response.into_body().into_data_stream();
        let mut prompt = bytes::BytesMut::new();
        while let Some(chunk) = body.next().await {
            let chunk = chunk?;
            if collect {
                anyhow::ensure!(
                    prompt.len().saturating_add(chunk.len()) <= logprobs::MAX_BODY,
                    "prefill logprobs exceed body limit"
                );
                prompt.extend_from_slice(&chunk);
            }
        }
        Ok(Ok(collect.then(|| prompt.freeze())))
    };
    tokio::pin!(prefill, decode);
    let (response, prompt) = tokio::select! {
        result = &mut prefill => match result? {
            Err(response) => return Ok((response, None)),
            Ok(prompt) => (decode.await?, prompt),
        },
        response = &mut decode => {
            let response = response?;
            if !response.status().is_success() { return Ok((response, None)); }
            match prefill.await? {
                Err(rejection) => return Ok((rejection, None)),
                Ok(prompt) => (response, prompt),
            }
        }
    };
    let prompt = response.status().is_success().then_some(prompt).flatten();
    Ok((response, prompt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    #[tokio::test]
    async fn native_pd_dispatch_runs_concurrently_and_preserves_rejections() {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let prefill = async {
            rx.await.unwrap();
            Ok(Response::new(Body::from("prompt")))
        };
        let decode = async {
            tx.send(()).unwrap();
            Ok(Response::new(Body::from("decode")))
        };
        let (response, prompt) = dispatch(prefill, decode, true).await.unwrap();
        assert_eq!(prompt.unwrap(), "prompt");
        assert_eq!(
            axum::body::to_bytes(response.into_body(), 1024)
                .await
                .unwrap(),
            "decode"
        );
        for reject_prefill in [true, false] {
            let reject = async {
                Ok(Response::builder()
                    .status(400)
                    .body(Body::from("engine rejection"))
                    .unwrap())
            };
            let pending = std::future::pending();
            let result = if reject_prefill {
                dispatch(reject, pending, true).await
            } else {
                dispatch(pending, reject, true).await
            };
            let (response, prompt) = result.unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert!(prompt.is_none());
            assert_eq!(
                axum::body::to_bytes(response.into_body(), 1024)
                    .await
                    .unwrap(),
                "engine rejection"
            );
        }
    }

    #[tokio::test]
    async fn native_pd_prefill_body_failure_drops_waiting_decode() {
        let cancelled = CancellationToken::new();
        let guard = cancelled.clone().drop_guard();
        let prefill = async {
            Ok(Response::new(Body::from_stream(futures::stream::iter([
                Err::<Bytes, _>(std::io::Error::other("prefill disconnected")),
            ]))))
        };
        let decode = async move {
            let _guard = guard;
            std::future::pending().await
        };
        assert!(dispatch(prefill, decode, false).await.is_err());
        assert!(cancelled.is_cancelled());
    }
}
