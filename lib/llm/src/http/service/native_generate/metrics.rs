// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP completion metrics only. Token counts and engine finish reasons stay opaque.

use axum::{body::Body, http::StatusCode, response::Response};
use futures::StreamExt;

use crate::http::service::metrics::{ErrorType, InflightGuard};

pub(super) fn observe_response(response: Response, mut guard: InflightGuard) -> Response {
    let (parts, body) = response.into_parts();
    let status = parts.status;
    let mut stream = body.into_data_stream();
    let body = async_stream::stream! {
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => yield Ok(bytes),
                Err(error) => {
                    guard.mark_error(ErrorType::Internal);
                    yield Err(error);
                    return;
                }
            }
        }
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
    };
    Response::from_parts(parts, Body::from_stream(body))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::http::service::metrics::{Endpoint, Metrics, RequestType, Status};

    #[tokio::test]
    async fn native_http_metrics_follow_completion_failure_and_unpolled_drop() {
        for (status, consume, broken, error) in [
            (StatusCode::OK, true, false, ErrorType::None),
            (StatusCode::BAD_REQUEST, true, false, ErrorType::Validation),
            (StatusCode::OK, false, false, ErrorType::Cancelled),
            (StatusCode::OK, true, true, ErrorType::Internal),
        ] {
            for streaming in [false, true] {
                let metrics = Arc::new(Metrics::new());
                let mut guard = metrics.clone().create_inflight_guard(
                    "native",
                    Endpoint::Generate,
                    streaming,
                    "request",
                );
                guard.mark_error(ErrorType::Cancelled);
                let body = if broken {
                    Body::from_stream(futures::stream::iter([Err::<bytes::Bytes, _>(
                        std::io::Error::other("truncated"),
                    )]))
                } else {
                    Body::from("opaque\r\n")
                };
                let response = Response::builder()
                    .status(status)
                    .header("x-engine", "unchanged")
                    .body(body)
                    .unwrap();
                let response = observe_response(response, guard);
                assert_eq!(metrics.get_inflight_count("native"), 1);
                assert_eq!(response.status(), status);
                assert_eq!(response.headers()["x-engine"], "unchanged");
                if consume {
                    let data = axum::body::to_bytes(response.into_body(), usize::MAX).await;
                    if broken {
                        assert!(data.is_err());
                    } else {
                        assert_eq!(data.unwrap(), "opaque\r\n");
                    }
                } else {
                    drop(response);
                }
                assert_eq!(metrics.get_inflight_count("native"), 0);
                let outcome = if matches!(error, ErrorType::None) {
                    Status::Success
                } else {
                    Status::Error
                };
                assert_eq!(
                    metrics.get_request_counter(
                        "native",
                        &Endpoint::Generate,
                        &if streaming {
                            RequestType::Stream
                        } else {
                            RequestType::Unary
                        },
                        &outcome,
                        &error
                    ),
                    1
                );
            }
        }
    }
}
