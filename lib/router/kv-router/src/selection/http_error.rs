// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP responses for [`SelectionError`]; the kernel's error carries the status code as a
//! number and knows nothing about axum.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use dynamo_router_protocol::sched::KvSchedulerError;

use super::SelectionError;

fn status(error: &SelectionError) -> StatusCode {
    StatusCode::from_u16(error.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
}

/// The response body and status for `error`.
pub fn response(error: SelectionError) -> Response {
    if matches!(
        &error,
        SelectionError::Scheduler(
            KvSchedulerError::RequestClassifierPanicked(_)
                | KvSchedulerError::RequestClassifierFailed(_)
                | KvSchedulerError::InvalidClassificationMetadata(_)
        )
    ) {
        // Plugin-produced detail (its error text, or the metadata it
        // returned) stays server-side: log it and return a fixed body.
        tracing::warn!(error = %error, "request classifier failure sanitized from response");
        return (
            status(&error),
            Json(serde_json::json!({"error": "request classifier failed"})),
        )
            .into_response();
    }
    if let SelectionError::Scheduler(KvSchedulerError::QueueRejected(rejection)) = &error {
        return (
            status(&error),
            Json(serde_json::json!({
                "error": error.to_string(),
                "details": rejection,
            })),
        )
            .into_response();
    }
    (
        status(&error),
        Json(serde_json::json!({"error": error.to_string()})),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    #[error("private plugin detail")]
    struct PrivateClassifierError;

    #[tokio::test]
    async fn classifier_error_response_is_sanitized() {
        let response = response(SelectionError::Scheduler(
            KvSchedulerError::RequestClassifierFailed(std::sync::Arc::new(PrivateClassifierError)),
        ));
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body.as_ref(), br#"{"error":"request classifier failed"}"#);
    }

    #[tokio::test]
    async fn invalid_classification_metadata_response_is_sanitized() {
        let response = response(SelectionError::Scheduler(
            KvSchedulerError::InvalidClassificationMetadata(
                "unknown policy class \"plugin-private-class\"".to_string(),
            ),
        ));
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body.as_ref(), br#"{"error":"request classifier failed"}"#);
    }
}
