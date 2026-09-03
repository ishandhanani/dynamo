// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

use dynamo_backend_common::{BackendError, DynamoError, ErrorType};

#[derive(Debug)]
pub enum SidecarStartupError {
    Cli(clap::Error),
    Dynamo(DynamoError),
}

impl SidecarStartupError {
    pub fn into_dynamo(self) -> DynamoError {
        match self {
            Self::Cli(error) => invalid_argument(error.to_string()),
            Self::Dynamo(error) => error,
        }
    }
}

impl fmt::Display for SidecarStartupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cli(error) => error.fmt(formatter),
            Self::Dynamo(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SidecarStartupError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cli(error) => Some(error),
            Self::Dynamo(error) => Some(error),
        }
    }
}

impl From<clap::Error> for SidecarStartupError {
    fn from(error: clap::Error) -> Self {
        Self::Cli(error)
    }
}

impl From<DynamoError> for SidecarStartupError {
    fn from(error: DynamoError) -> Self {
        Self::Dynamo(error)
    }
}

fn backend(kind: BackendError, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(kind))
        .message(message)
        .build()
}

pub fn invalid_argument(message: impl Into<String>) -> DynamoError {
    backend(BackendError::InvalidArgument, message)
}

pub fn protocol_error(peer: &str, message: impl Into<String>) -> DynamoError {
    backend(
        BackendError::Unknown,
        format!("invalid {peer} gRPC response: {}", message.into()),
    )
}

pub fn engine_shutdown(message: impl Into<String>) -> DynamoError {
    backend(BackendError::EngineShutdown, message)
}

pub fn cannot_connect(message: impl Into<String>) -> DynamoError {
    backend(BackendError::CannotConnect, message)
}

pub fn connection_timeout(message: impl Into<String>) -> DynamoError {
    backend(BackendError::ConnectionTimeout, message)
}

/// The engine stream ended without a terminal response (engine dropped
/// mid-stream). Migratable: another worker can replay the request.
pub fn stream_incomplete(message: impl Into<String>) -> DynamoError {
    backend(BackendError::StreamIncomplete, message)
}

/// Whether a gRPC status text describes the transport failing underneath an
/// established stream (peer died, connection reset) rather than an
/// application error the engine returned.
fn is_transport_failure(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    [
        "h2 protocol error",
        "transport error",
        "error reading a body",
        "connection reset",
        "connection refused",
        "broken pipe",
        "connection closed",
        "stream closed",
        "goaway",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

pub fn status_to_dynamo(rpc: &str, status: tonic::Status) -> DynamoError {
    let kind = match status.code() {
        tonic::Code::InvalidArgument
        | tonic::Code::NotFound
        | tonic::Code::OutOfRange
        | tonic::Code::FailedPrecondition
        | tonic::Code::AlreadyExists => BackendError::InvalidArgument,
        tonic::Code::Unavailable => BackendError::CannotConnect,
        tonic::Code::Cancelled => BackendError::Cancelled,
        tonic::Code::DeadlineExceeded => BackendError::ConnectionTimeout,
        // tonic reports a peer that vanished mid-stream as Unknown/Internal
        // with the h2 transport text; that is a lost connection, not an
        // engine-side failure, and the request can move to another worker.
        tonic::Code::Unknown | tonic::Code::Internal | tonic::Code::Aborted
            if is_transport_failure(status.message()) =>
        {
            BackendError::Disconnected
        }
        _ => BackendError::Unknown,
    };
    backend(
        kind,
        format!("{rpc}: {} ({:?})", status.message(), status.code()),
    )
}

#[cfg(test)]
mod tests {
    use dynamo_backend_common::{BackendError, ErrorType};

    use super::status_to_dynamo;

    #[test]
    fn mid_stream_transport_failures_are_disconnects() {
        let status =
            tonic::Status::unknown("h2 protocol error: error reading a body from connection");
        assert_eq!(
            status_to_dynamo("GenerateStream", status).error_type(),
            ErrorType::Backend(BackendError::Disconnected)
        );
        let status = tonic::Status::unknown("model rejected the prompt");
        assert_eq!(
            status_to_dynamo("GenerateStream", status).error_type(),
            ErrorType::Backend(BackendError::Unknown)
        );
    }

    #[test]
    fn maps_transport_statuses_to_backend_errors() {
        for (code, expected) in [
            (tonic::Code::InvalidArgument, BackendError::InvalidArgument),
            (tonic::Code::Unavailable, BackendError::CannotConnect),
            (tonic::Code::Cancelled, BackendError::Cancelled),
            (
                tonic::Code::DeadlineExceeded,
                BackendError::ConnectionTimeout,
            ),
            (tonic::Code::Internal, BackendError::Unknown),
        ] {
            let error = status_to_dynamo("Test", tonic::Status::new(code, "failure"));
            assert_eq!(error.error_type(), ErrorType::Backend(expected));
        }
    }
}
