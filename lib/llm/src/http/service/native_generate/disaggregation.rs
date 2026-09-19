// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::body::Body;
use futures::StreamExt;

use super::*;
use crate::http::service::native_generate::{BodyProgress, forward_accounted_with_cancellation};

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
        children: Vec<Child>,
        requested_rank: Option<u32>,
        deadline: Instant,
    ) -> anyhow::Result<Response> {
        let (prefill, endpoint) = router.native_binding()?;
        let uses_kv = prefill.host.kv_router_if_enabled().is_some();
        let config = &prefill.card.runtime_config;
        let explicit_rank = projection
            .value
            .get("disagg_prefill_dp_rank")
            .filter(|value| !value.is_null())
            .map(|value| serde_json::from_value::<u32>(value.clone()))
            .transpose()
            .map_err(|error| NativeRequestError(error.into()))?;
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
        let prefill_children = tokio::task::spawn_blocking(move || {
            child_projection.children(tokenizer.as_ref(), uses_kv, true)
        })
        .await?
        .map_err(NativeRequestError)?;
        let prefill_admission = prefill
            .reserve(
                prefill_children,
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
        for (key, expected) in [
            ("bootstrap_host", Value::from(host.clone())),
            ("bootstrap_port", Value::from(port)),
            ("disagg_prefill_dp_rank", Value::from(rank)),
        ] {
            projection
                .require_control(key, &expected)
                .map_err(NativeRequestError)?;
        }
        if constraints.is_some() && self.host.kv_router_if_enabled().is_none() {
            anyhow::bail!("native topology-constrained decode requires KV routing");
        }
        let decode_admission = admission_wait(
            &prefill_admission.reservations,
            deadline,
            self.reserve(
                children,
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
        headers.insert("x-override-bootstrap-host", host.parse()?);
        headers.insert("x-override-bootstrap-port", port.to_string().parse()?);
        headers.insert(
            "x-override-disagg-prefill-dp-rank",
            rank.to_string().parse()?,
        );
        // Preserve explicit scalar/list rooms. Otherwise reserve enough room
        // IDs for the maximum admitted fan-out without unsigned overflow.
        if projection
            .value
            .get("bootstrap_room")
            .is_none_or(Value::is_null)
        {
            let room = bootstrap_room();
            headers.insert("x-override-bootstrap-room", room.to_string().parse()?);
        }
        let request = http::Request {
            method: method.to_string(),
            headers: http::encode_headers(headers),
            body,
            operation: Default::default(),
        };
        let prefill_attempt = prefill_admission.into_attempt(&prefill, "prefill")?;
        let decode_attempt = decode_admission.into_attempt(self, "decode")?;
        let cancellation = CancellationToken::new();
        let guard = cancellation.clone().drop_guard();
        // Poll both dispatches together: a unary prefill response may wait for
        // the decode receiver, so awaiting prefill first would deadlock.
        let heads = join_heads(
            forward_accounted_with_cancellation(
                &prefill.client,
                Context::with_id_and_metadata(
                    request.clone(),
                    uuid::Uuid::new_v4().to_string(),
                    metadata.clone(),
                ),
                prefill_attempt,
                cancellation.clone(),
            ),
            forward_accounted_with_cancellation(
                &self.client,
                Context::with_id_and_metadata(request, uuid::Uuid::new_v4().to_string(), metadata),
                decode_attempt,
                cancellation.clone(),
            ),
            &cancellation,
        )
        .await?;
        let decode = match heads {
            Heads::Ready(decode) => decode,
            Heads::Rejected(response) => return Ok(response),
        };
        let response = decode_response(decode, cancellation);
        guard.disarm();
        Ok(response)
    }
}

fn bootstrap_room() -> u64 {
    // SGLang assigns the Python integer into a torch.uint64 tensor, whose
    // scalar setter still rejects values above i64::MAX. Reserve a whole
    // aligned fan-out range within the supported scalar assignment range.
    (uuid::Uuid::new_v4().as_u128() as u64) & (i64::MAX as u64) & !4095
}

enum Heads {
    Ready(Response),
    Rejected(Response),
}

async fn join_heads(
    prefill: impl std::future::Future<Output = anyhow::Result<Response>>,
    decode: impl std::future::Future<Output = anyhow::Result<Response>>,
    cancellation: &CancellationToken,
) -> anyhow::Result<Heads> {
    let join = async {
        tokio::pin!(prefill, decode);
        let decode = tokio::select! {
            prefill = &mut prefill => {
                let prefill = prefill?;
                if !prefill.status().is_success() { return Ok(Heads::Rejected(prefill)); }
                drain_prefill(prefill, cancellation.clone());
                decode.await?
            },
            decode = &mut decode => {
                let decode = decode?;
                if !decode.status().is_success() { return Ok(Heads::Rejected(decode)); }
                // Observe decode transport failure while prefill headers are
                // pending. Read-ahead stays bounded until the client can read.
                let decode = buffer_decode(decode, cancellation.clone());
                let prefill = prefill.await?;
                if !prefill.status().is_success() { return Ok(Heads::Rejected(prefill)); }
                drain_prefill(prefill, cancellation.clone());
                decode
            },
        };
        if !decode.status().is_success() {
            return Ok(Heads::Rejected(decode));
        }
        Ok(Heads::Ready(decode))
    };
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => anyhow::bail!("native P/D transport cancelled before response headers"),
        result = join => result,
    }
}

fn buffer_decode(response: Response, cancellation: CancellationToken) -> Response {
    let (parts, body) = response.into_parts();
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    tokio::spawn(async move {
        let mut body = body.into_data_stream();
        loop {
            let chunk = tokio::select! {
                biased;
                _ = cancellation.cancelled() => return,
                _ = tx.closed() => { cancellation.cancel(); return; },
                chunk = body.next() => chunk,
            };
            let chunk = match chunk {
                Some(Ok(chunk)) => chunk,
                Some(Err(error)) => {
                    tracing::warn!(%error, "native decode response transport failed");
                    cancellation.cancel();
                    return;
                }
                None => return,
            };
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => return,
                result = tx.send(chunk) => if result.is_err() {
                    cancellation.cancel();
                    return;
                },
            }
        }
    });
    Response::from_parts(
        parts,
        Body::from_stream(
            tokio_stream::wrappers::ReceiverStream::new(rx).map(Ok::<_, axum::Error>),
        ),
    )
}

impl Projection {
    fn require_control(&self, key: &str, expected: &Value) -> anyhow::Result<()> {
        let Some(value) = self.value.get(key).filter(|value| !value.is_null()) else {
            return Ok(());
        };
        anyhow::ensure!(
            match value {
                Value::Array(values) =>
                    !values.is_empty() && values.iter().all(|value| value == expected),
                value => value == expected,
            },
            "{key} conflicts with the selected native prefill route"
        );
        Ok(())
    }
}

/// Drain prefill independently with bounded transport backpressure. Decode is
/// authoritative; a failed prefill transport cuts its body without parsing it.
fn drain_prefill(prefill: Response, drain_cancel: CancellationToken) {
    tokio::spawn(async move {
        let mut body = prefill.into_body().into_data_stream();
        loop {
            tokio::select! {
                _ = drain_cancel.cancelled() => return,
                chunk = body.next() => match chunk {
                    None => return,
                    Some(Ok(_)) => {},
                    Some(Err(error)) => {
                        tracing::warn!(%error, "native prefill response transport failed");
                        drain_cancel.cancel();
                        return;
                    },
                }
            }
        }
    });
}

fn decode_response(decode: Response, cancellation: CancellationToken) -> Response {
    let mut progress = BodyProgress::new(decode.status(), decode.headers());
    let (parts, body) = decode.into_parts();
    let guard = cancellation.clone().drop_guard();
    // This outer guard cancels unfinished prefill even after decode completes.
    // Keep it outside the generator until first poll so an unpolled drop works.
    let stream: futures::stream::BoxStream<'static, anyhow::Result<Bytes>> = Box::pin(
        async_stream::try_stream! {
            let _guard = guard;
            let mut body = body.into_data_stream();
            loop {
                let chunk = tokio::select! {
                    biased;
                    _ = cancellation.cancelled() => Err(anyhow::anyhow!("native P/D transport cancelled")),
                    chunk = body.next() => Ok(chunk),
                }?;
                let Some(chunk) = chunk else { return; };
                let chunk = chunk?;
                progress.advance(chunk.len())?;
                if progress.complete() { cancellation.cancel(); }
                yield chunk;
                if progress.complete() { return; }
            }
        },
    );
    Response::from_parts(parts, Body::from_stream(stream))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    #[tokio::test]
    async fn native_pd_heads_run_concurrently_and_rejections_do_not_wait_for_peer() {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let prefill = async {
            rx.await?;
            anyhow::Ok(Response::new(Body::empty()))
        };
        let decode = async {
            tx.send(()).unwrap();
            anyhow::Ok(Response::new(Body::empty()))
        };
        assert!(matches!(
            tokio::time::timeout(
                Duration::from_secs(1),
                join_heads(prefill, decode, &CancellationToken::new())
            )
            .await
            .unwrap()
            .unwrap(),
            Heads::Ready(..)
        ));
        for reject_prefill in [true, false] {
            let reject = async {
                anyhow::Ok(
                    Response::builder()
                        .status(StatusCode::TOO_MANY_REQUESTS)
                        .body(Body::from("engine rejection"))
                        .unwrap(),
                )
            };
            let pending = std::future::pending::<anyhow::Result<Response>>();
            let result = tokio::time::timeout(Duration::from_secs(1), async {
                if reject_prefill {
                    join_heads(reject, pending, &CancellationToken::new()).await
                } else {
                    join_heads(pending, reject, &CancellationToken::new()).await
                }
            })
            .await
            .unwrap()
            .unwrap();
            let Heads::Rejected(response) = result else {
                panic!("expected engine rejection");
            };
            assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
            assert_eq!(
                axum::body::to_bytes(response.into_body(), 100)
                    .await
                    .unwrap(),
                "engine rejection"
            );
        }
    }

    #[tokio::test]
    async fn native_pd_body_failure_does_not_wait_for_peer_headers() {
        for fail_prefill in [true, false] {
            let cancel = CancellationToken::new();
            let failed = async {
                anyhow::Ok(Response::new(Body::from_stream(futures::stream::once(
                    async { Err::<Bytes, _>(std::io::Error::other("stage disconnected")) },
                ))))
            };
            let pending = std::future::pending::<anyhow::Result<Response>>();
            let result = tokio::time::timeout(Duration::from_secs(1), async {
                if fail_prefill {
                    join_heads(failed, pending, &cancel).await
                } else {
                    join_heads(pending, failed, &cancel).await
                }
            })
            .await
            .expect("body failure must cancel the pending peer header wait");
            assert!(result.is_err());
            assert!(cancel.is_cancelled());
        }
    }

    #[tokio::test]
    async fn native_pd_decode_read_ahead_preserves_bytes_and_unpolled_drop_cancels() {
        let cancel = CancellationToken::new();
        let response = buffer_decode(
            Response::builder()
                .header("x-engine", "preserved")
                .body(Body::from_stream(futures::stream::iter([
                    Ok::<_, std::io::Error>(Bytes::from_static(b"first")),
                    Ok(Bytes::from_static(b"second")),
                    Ok(Bytes::from_static(b"last")),
                ])))
                .unwrap(),
            cancel.clone(),
        );
        assert_eq!(response.headers()["x-engine"], "preserved");
        assert_eq!(
            axum::body::to_bytes(response.into_body(), 100)
                .await
                .unwrap(),
            "firstsecondlast"
        );
        assert!(!cancel.is_cancelled());
        let response = buffer_decode(
            Response::new(Body::from_stream(futures::stream::pending::<
                Result<Bytes, std::io::Error>,
            >())),
            cancel.clone(),
        );
        drop(response);
        tokio::time::timeout(Duration::from_secs(1), cancel.cancelled())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn native_pd_drain_failure_cancels_decode_and_unpolled_drop_cancels_prefill() {
        let cancel = CancellationToken::new();
        let prefill = Response::new(Body::from_stream(futures::stream::once(async {
            Err::<Bytes, _>(std::io::Error::other("truncated"))
        })));
        let decode = Response::new(Body::from_stream(futures::stream::pending::<
            Result<Bytes, std::io::Error>,
        >()));
        drain_prefill(prefill, cancel.clone());
        let response = decode_response(decode, cancel.clone());
        assert!(
            tokio::time::timeout(
                Duration::from_secs(1),
                axum::body::to_bytes(response.into_body(), 100)
            )
            .await
            .unwrap()
            .is_err()
        );
        assert!(cancel.is_cancelled());
        let cancel = CancellationToken::new();
        let response = decode_response(Response::new(Body::from("opaque")), cancel.clone());
        drop(response);
        assert!(cancel.is_cancelled());
    }

    #[tokio::test]
    async fn native_pd_decode_length_completion_cancels_unfinished_prefill() {
        let cancel = CancellationToken::new();
        let prefill = Response::new(Body::from_stream(futures::stream::pending::<
            Result<Bytes, std::io::Error>,
        >()));
        let decode = Response::builder()
            .header("content-length", "6")
            .header("x-engine", "preserved")
            .body(Body::from("opaque"))
            .unwrap();
        drain_prefill(prefill, cancel.clone());
        let response = decode_response(decode, cancel.clone());
        assert_eq!(response.headers()["x-engine"], "preserved");
        let mut body = response.into_body().into_data_stream();
        assert_eq!(body.next().await.unwrap().unwrap(), "opaque");
        assert!(
            cancel.is_cancelled(),
            "HTTP framing can finish without another body poll"
        );
        assert!(body.next().await.is_none());
    }

    #[test]
    fn native_pd_projection_omits_warmups_and_rejects_conflicting_controls() {
        for _ in 0..128 {
            assert!(bootstrap_room() + 4095 <= i64::MAX as u64);
        }
        let projection = Projection::read(br#"{"input_ids":[[1],[2]],"sampling_params":{"n":3},"bootstrap_host":["engine","engine"],"bootstrap_port":1234}"#).unwrap();
        projection
            .require_control("bootstrap_host", &Value::from("engine"))
            .unwrap();
        assert!(
            projection
                .require_control("bootstrap_port", &Value::from(4321))
                .is_err()
        );
        assert!(projection.require_aggregated().is_err());
        let children = projection.children(None, true, true).unwrap();
        assert_eq!(children.len(), 6);
        assert!(children.iter().all(|child| child.kind == ChildKind::Sample));
        assert_eq!(
            children
                .iter()
                .map(|child| child.tokens[0])
                .collect::<Vec<_>>(),
            [1, 1, 1, 2, 2, 2]
        );
    }
}
