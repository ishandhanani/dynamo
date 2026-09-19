// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP reconstruction for an already-selected HTTP worker.
//! Selection, capability admission and lifecycle accounting belong to the caller.

use axum::{body::Body, response::Response};
use dynamo_runtime::pipeline::{ManyOut, SingleIn, network::egress::push_router::PushRouter};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;

use crate::protocols::http::{Request, ResponseFrame, decode_headers};

pub(crate) mod metrics;

pub type HttpClient = PushRouter<Request, Annotated<ResponseFrame>>;

/// Hyper may stop polling immediately after Content-Length bytes, without
/// polling our final End frame. Completion must also follow HTTP framing.
pub(crate) struct BodyProgress {
    remaining: Option<u64>,
}

impl BodyProgress {
    pub(crate) fn new(status: axum::http::StatusCode, headers: &axum::http::HeaderMap) -> Self {
        let remaining = if status.is_informational() || matches!(status.as_u16(), 204 | 304) {
            Some(0)
        } else {
            headers
                .get(axum::http::header::CONTENT_LENGTH)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse().ok())
        };
        Self { remaining }
    }

    pub(crate) fn advance(&mut self, bytes: usize) -> anyhow::Result<()> {
        if let Some(remaining) = &mut self.remaining {
            *remaining = remaining
                .checked_sub(bytes as u64)
                .ok_or_else(|| anyhow::anyhow!("native HTTP body exceeded Content-Length"))?;
        }
        Ok(())
    }

    pub(crate) fn complete(&self) -> bool {
        self.remaining == Some(0)
    }

    pub(crate) fn end(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.remaining.is_none_or(|remaining| remaining == 0),
            "native HTTP body was truncated"
        );
        Ok(())
    }
}

/// Dispatch once to the selected worker. No fallback or generation replay.
/// The caller must supply a client restricted to the admitted native WorkerSet.
pub async fn forward(
    client: &HttpClient,
    request: SingleIn<Request>,
    worker_id: u64,
) -> anyhow::Result<Response> {
    let head_only = request.content().method == "HEAD";
    response(client.direct(request, worker_id).await?, None, head_only).await
}

/// Dispatch once; HTTP cancellation drops the runtime stream and upstream request.
pub(crate) async fn forward_with_cancellation(
    client: &HttpClient,
    request: SingleIn<Request>,
    worker_id: u64,
    cancellation: tokio_util::sync::CancellationToken,
) -> anyhow::Result<Response> {
    let head_only = request.content().method == "HEAD";
    let guard = cancellation.clone().drop_guard();
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => anyhow::bail!("native HTTP request cancelled"),
        result = async { response(client.direct(request, worker_id).await?, Some(guard), head_only).await } => result,
    }
}

pub(crate) async fn forward_reserved(
    client: &HttpClient,
    request: dynamo_runtime::pipeline::SingleIn<Request>,
    admission: crate::kv_router::RouteReservation,
    cancellation: tokio_util::sync::CancellationToken,
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
            Box::pin(stream) as futures::stream::BoxStream<'static, anyhow::Result<bytes::Bytes>>
        ),
    ))
}

async fn response(
    mut stream: ManyOut<Annotated<ResponseFrame>>,
    mut cancellation: Option<tokio_util::sync::DropGuard>,
    head_only: bool,
) -> anyhow::Result<Response> {
    let first = stream
        .next()
        .await
        .ok_or_else(|| anyhow::anyhow!("native HTTP stream ended before headers"))?
        .into_data()?;
    let Some(ResponseFrame::Head { status, headers }) = first else {
        anyhow::bail!("native HTTP stream must begin with headers");
    };
    let headers = decode_headers(headers)?;
    let status = axum::http::StatusCode::from_u16(status)?;
    let mut progress = BodyProgress::new(status, &headers);
    if head_only {
        progress.remaining = Some(0);
    }
    if progress.complete()
        && let Some(guard) = cancellation.take()
    {
        guard.disarm();
    }
    let body: futures::stream::BoxStream<'static, anyhow::Result<bytes::Bytes>> =
        Box::pin(async_stream::try_stream! {
            let mut cancellation = cancellation;
            while let Some(frame) = stream.next().await {
                match frame.into_data()? {
                    Some(ResponseFrame::Body(bytes)) => {
                        progress.advance(bytes.len())?;
                        if progress.complete() && let Some(guard) = cancellation.take() {
                            guard.disarm();
                        }
                        yield bytes;
                    },
                    Some(ResponseFrame::End) => {
                        progress.end()?;
                        if let Some(guard) = cancellation { guard.disarm(); }
                        return;
                    },
                    _ => Err(anyhow::anyhow!("invalid native HTTP response frame"))?,
                }
            }
            Err(anyhow::anyhow!("native HTTP response was truncated"))?;
        });
    let mut response = Response::new(Body::from_stream(body));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context, ResponseStream};
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn native_content_length_completes_without_end_poll_but_rejects_mismatch() {
        for (length, payload, drain, cancelled, fails) in [
            (2, b"ok".as_slice(), false, false, false),
            (0, b"".as_slice(), false, false, false),
            (3, b"ok".as_slice(), true, true, true),
            (1, b"ok".as_slice(), true, true, true),
        ] {
            let cancel = CancellationToken::new();
            let context = Context::new(());
            let frames = [
                ResponseFrame::Head {
                    status: 200,
                    headers: vec![("content-length".into(), length.to_string().into())],
                },
                ResponseFrame::Body(bytes::Bytes::from_static(payload)),
                ResponseFrame::End,
            ];
            let stream = ResponseStream::new(
                Box::pin(futures::stream::iter(
                    frames.into_iter().map(Annotated::from_data),
                )),
                context.context(),
            );
            let response = response(stream, Some(cancel.clone().drop_guard()), false)
                .await
                .unwrap();
            let mut body = response.into_body().into_data_stream();
            let mut failed = false;
            if drain {
                while let Some(chunk) = body.next().await {
                    if chunk.is_err() {
                        failed = true;
                        break;
                    }
                }
            } else if length > 0 {
                assert_eq!(body.next().await.unwrap().unwrap().as_ref(), payload);
            }
            drop(body);
            assert_eq!(failed, fails);
            assert_eq!(cancel.is_cancelled(), cancelled);
        }
    }

    #[tokio::test]
    async fn response_drop_cancels_even_when_body_is_never_polled_but_end_does_not() {
        for consume in [false, true] {
            let cancel = CancellationToken::new();
            let frames = futures::stream::iter([
                Annotated::from_data(ResponseFrame::Head {
                    status: 200,
                    headers: Vec::new(),
                }),
                Annotated::from_data(ResponseFrame::Body(bytes::Bytes::from_static(
                    b"not json or SSE",
                ))),
                Annotated::from_data(ResponseFrame::End),
            ]);
            let stream = ResponseStream::new(Box::pin(frames), Context::new(()).context());
            let response = response(stream, Some(cancel.clone().drop_guard()), false)
                .await
                .unwrap();
            if consume {
                let body = axum::body::to_bytes(response.into_body(), 1024)
                    .await
                    .unwrap();
                assert_eq!(body.as_ref(), b"not json or SSE");
                assert!(!cancel.is_cancelled());
            } else {
                drop(response);
                assert!(cancel.is_cancelled());
            }
        }
    }
}
