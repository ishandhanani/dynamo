// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    kv_router::RouteReservation,
    protocols::http::{Request, ResponseFrame, decode_headers},
};
use axum::{body::Body, response::Response};
use dynamo_runtime::pipeline::{ManyOut, SingleIn, network::egress::push_router::PushRouter};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;

pub(crate) mod metrics;
pub type HttpClient = PushRouter<Request, Annotated<ResponseFrame>>;

/// Dispatch to an already-selected, admitted worker without replay.
pub async fn forward(
    client: &HttpClient,
    request: SingleIn<Request>,
    worker: u64,
) -> anyhow::Result<Response> {
    response(client.direct(request, worker).await?, None).await
}

pub(crate) async fn forward_reserved(
    client: &HttpClient,
    request: SingleIn<Request>,
    reservation: RouteReservation,
) -> anyhow::Result<Response> {
    let cancelled = reservation.cancel.clone();
    tokio::select! {
        biased;
        _ = cancelled.cancelled() => anyhow::bail!("HTTP routing reservation expired"),
        result = async {
            response(client.direct(request, reservation.target.worker_id).await?, Some(reservation)).await
        } => result,
    }
}

async fn response(
    mut stream: ManyOut<Annotated<ResponseFrame>>,
    reservation: Option<RouteReservation>,
) -> anyhow::Result<Response> {
    let head = stream
        .next()
        .await
        .ok_or_else(|| anyhow::anyhow!("HTTP stream ended before headers"))?
        .into_data()?;
    let Some(ResponseFrame::Head { status, headers }) = head else {
        anyhow::bail!("HTTP stream must begin with headers");
    };
    let cancelled = reservation
        .as_ref()
        .map(|r| r.cancel.clone())
        .unwrap_or_default();
    let body: futures::stream::BoxStream<'static, anyhow::Result<bytes::Bytes>> = Box::pin(
        async_stream::try_stream! {
            let _reservation = reservation;
            loop {
                let frame = tokio::select! {
                    biased;
                    _ = cancelled.cancelled() => Err(anyhow::anyhow!("HTTP routing reservation expired")),
                    frame = stream.next() => frame.ok_or_else(|| anyhow::anyhow!("HTTP response was truncated")),
                }?;
                match frame.into_data()? {
                    Some(ResponseFrame::Body(bytes)) => yield bytes,
                    Some(ResponseFrame::End) => return,
                    _ => Err(anyhow::anyhow!("invalid HTTP response frame"))?,
                }
            }
        },
    );
    let mut response = Response::new(Body::from_stream(body));
    *response.status_mut() = axum::http::StatusCode::from_u16(status)?;
    *response.headers_mut() = decode_headers(headers)?;
    Ok(response)
}
