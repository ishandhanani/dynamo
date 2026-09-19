// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP completion metrics only. Token counts and engine finish reasons stay opaque.

use axum::{body::Body, http::StatusCode, response::Response};
use futures::StreamExt;

use super::BodyProgress;
use crate::http::service::metrics::{ErrorType, InflightGuard};

fn mark_completion(guard: &mut InflightGuard, status: StatusCode) {
    if status.is_client_error() || status.is_server_error() {
        guard.mark_error(match status {
            StatusCode::TOO_MANY_REQUESTS => ErrorType::Overload,
            StatusCode::SERVICE_UNAVAILABLE => ErrorType::Unavailable,
            StatusCode::NOT_FOUND => ErrorType::NotFound,
            status if status.is_client_error() => ErrorType::Validation,
            _ => ErrorType::Internal,
        });
    } else {
        guard.mark_ok();
    }
}

pub(crate) fn observe_response(response: Response, mut guard: InflightGuard) -> Response {
    let (parts, body) = response.into_parts();
    let status = parts.status;
    let mut progress = BodyProgress::new(status, &parts.headers);
    if progress.complete() {
        mark_completion(&mut guard, status);
    }
    let mut stream = body.into_data_stream();
    let body = async_stream::stream! {
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    if let Err(error) = progress.advance(bytes.len()) {
                        guard.mark_error(ErrorType::Internal);
                        yield Err(axum::Error::new(error));
                        return;
                    }
                    if progress.complete() { mark_completion(&mut guard, status); }
                    yield Ok(bytes);
                },
                Err(error) => {
                    guard.mark_error(ErrorType::Internal);
                    yield Err(error);
                    return;
                }
            }
        }
        if let Err(error) = progress.end() {
            guard.mark_error(ErrorType::Internal);
            yield Err(axum::Error::new(error));
            return;
        }
        mark_completion(&mut guard, status);
    };
    Response::from_parts(parts, Body::from_stream(body))
}
