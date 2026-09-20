// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP completion metrics only. Token counts and engine finish reasons stay opaque.

use axum::{body::Body, http::StatusCode, response::Response};
use futures::StreamExt;

use crate::http::service::metrics::{ErrorType, InflightGuard};

fn mark_completion(guard: &mut InflightGuard, status: StatusCode) {
    if status.is_client_error() || status.is_server_error() {
        use crate::http::service::openai::{backend_http_error_class, metric_error_type_for_class};
        guard.mark_error(metric_error_type_for_class(backend_http_error_class(
            status,
        )));
    } else {
        guard.mark_ok();
    }
}

pub(crate) fn observe_response(response: Response, mut guard: InflightGuard) -> Response {
    let (parts, body) = response.into_parts();
    let status = parts.status;
    // Hyper can stop polling after Content-Length bytes, before our End frame.
    let mut remaining = parts
        .headers
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    if status.is_informational() || matches!(status.as_u16(), 204 | 304) {
        remaining = Some(0);
    }
    if remaining == Some(0) {
        mark_completion(&mut guard, status);
    }
    let mut stream = body.into_data_stream();
    let body = async_stream::stream! {
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    remaining = remaining.map(|n| n.saturating_sub(bytes.len() as u64));
                    if remaining == Some(0) { mark_completion(&mut guard, status); }
                    yield Ok(bytes);
                },
                Err(error) => {
                    guard.mark_error(ErrorType::Internal);
                    yield Err(error);
                    return;
                }
            }
        }
        mark_completion(&mut guard, status);
    };
    Response::from_parts(parts, Body::from_stream(body))
}
