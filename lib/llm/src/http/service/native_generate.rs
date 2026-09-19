// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP reconstruction for an already-selected native SGLang worker.
//! Selection, capability admission and lifecycle accounting belong to the caller.

use axum::{body::Body, response::Response};
use dynamo_runtime::pipeline::{ManyOut, SingleIn, network::egress::push_router::PushRouter};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;

use crate::protocols::sglang::http::{Request, ResponseFrame, decode_headers};

pub mod lifecycle;
mod metrics;
pub(crate) mod routing;

pub type NativeGenerateClient = PushRouter<Request, Annotated<ResponseFrame>>;

/// Dispatch once to the selected worker. No fallback or generation replay.
/// The caller must supply a client restricted to the admitted native WorkerSet.
pub async fn forward(
    client: &NativeGenerateClient,
    request: SingleIn<Request>,
    worker_id: u64,
) -> anyhow::Result<Response> {
    response(client.direct(request, worker_id).await?, None).await
}

/// Forward an admitted attempt while its detached accounting owner watches the
/// engine. The request and response bodies remain opaque, including unary JSON.
pub async fn forward_accounted(
    client: &NativeGenerateClient,
    mut request: SingleIn<Request>,
    attempt: lifecycle::NativeAttempt,
) -> anyhow::Result<Response> {
    use crate::protocols::sglang::http::lifecycle::{ATTEMPT_HEADER, INCARNATION_HEADER};

    anyhow::ensure!(
        !request
            .headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case(ATTEMPT_HEADER)
                || name.eq_ignore_ascii_case(INCARNATION_HEADER)
                || name.eq_ignore_ascii_case("x-override-routed-dp-rank")),
        "native lifecycle headers are owned by Dynamo"
    );
    request.headers.push((
        ATTEMPT_HEADER.to_string(),
        attempt.attempt_id.clone().into(),
    ));
    request.headers.push((
        INCARNATION_HEADER.to_string(),
        attempt.incarnation.clone().into(),
    ));
    request.headers.push((
        "x-override-routed-dp-rank".to_string(),
        attempt.dp_rank.to_string().into(),
    ));
    let worker_id = attempt.worker_id();
    let cancellation = tokio_util::sync::CancellationToken::new();
    let guard = cancellation.clone().drop_guard();
    attempt.start(cancellation);
    response(client.direct(request, worker_id).await?, Some(guard)).await
}

async fn response(
    mut stream: ManyOut<Annotated<ResponseFrame>>,
    cancellation: Option<tokio_util::sync::DropGuard>,
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
    let body: futures::stream::BoxStream<'static, anyhow::Result<bytes::Bytes>> =
        Box::pin(async_stream::try_stream! {
            let cancellation = cancellation;
            while let Some(frame) = stream.next().await {
                match frame.into_data()? {
                    Some(ResponseFrame::Body(bytes)) => yield bytes,
                    Some(ResponseFrame::End) => {
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
            let response = response(stream, Some(cancel.clone().drop_guard()))
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
