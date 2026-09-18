// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP reconstruction for an already-selected native SGLang worker.
//! Selection, capability admission and lifecycle accounting belong to the caller.

use axum::{body::Body, response::Response};
use dynamo_runtime::pipeline::{ManyOut, SingleIn, network::egress::push_router::PushRouter};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;

use crate::protocols::sglang::http::{Request, ResponseFrame, decode_headers};

pub type NativeGenerateClient = PushRouter<Request, Annotated<ResponseFrame>>;

/// Dispatch once to the selected worker. No fallback or generation replay.
/// The caller must supply a client restricted to the admitted native WorkerSet.
pub async fn forward(
    client: &NativeGenerateClient,
    request: SingleIn<Request>,
    worker_id: u64,
) -> anyhow::Result<Response> {
    response(client.direct(request, worker_id).await?).await
}

async fn response(mut stream: ManyOut<Annotated<ResponseFrame>>) -> anyhow::Result<Response> {
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
            while let Some(frame) = stream.next().await {
                match frame.into_data()? {
                    Some(ResponseFrame::Body(bytes)) => yield bytes,
                    Some(ResponseFrame::End) => return,
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
