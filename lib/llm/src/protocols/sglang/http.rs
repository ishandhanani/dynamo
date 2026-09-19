// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Versioned native HTTP request plane, separate from the token pipeline.
//! Bodies use binary MessagePack fields; JSON/SSE semantics belong to SGLang.

use axum::http::{HeaderMap, HeaderName, HeaderValue};
use bytes::Bytes;
use serde::{Deserialize, Serialize};

pub mod lifecycle;

pub const CAPABILITY: &str = "sglang_generate_http_v1";
pub const MAX_BODY_CHUNK: usize = 64 * 1024;

pub fn endpoint_name(primary: &str) -> String {
    format!("{primary}_native_http_v1")
}

/// A list preserves repeated headers and non-UTF-8 header values.
pub type Headers = Vec<(String, Bytes)>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub method: String,
    pub headers: Headers,
    pub body: Bytes,
    /// Session operations require all three session descriptor capabilities.
    /// Omit the default to preserve the original generate wire representation.
    #[serde(default, skip_serializing_if = "Operation::is_generate")]
    pub operation: Operation,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operation {
    #[default]
    Generate,
    OpenSession,
    CloseSession,
    SessionRouting,
}

impl Operation {
    fn is_generate(&self) -> bool {
        *self == Self::Generate
    }

    pub fn path(self) -> &'static str {
        match self {
            Self::Generate => "/generate",
            Self::OpenSession => "/open_session",
            Self::CloseSession => "/close_session",
            Self::SessionRouting => "/session_routing",
        }
    }

    pub fn supports_method(self, method: &str) -> bool {
        match self {
            Self::Generate => matches!(method, "POST" | "PUT"),
            Self::OpenSession | Self::CloseSession => matches!(method, "GET" | "POST"),
            Self::SessionRouting => method == "POST",
        }
    }
}

/// Exactly one head, zero or more body chunks, then End. Transport failures
/// use the runtime error envelope; they are never inserted into the HTTP body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseFrame {
    Head { status: u16, headers: Headers },
    Body(Bytes),
    End,
}

pub fn encode_headers(mut headers: HeaderMap) -> Headers {
    strip_hop_by_hop(&mut headers);
    headers
        .iter()
        .map(|(name, value)| {
            (
                name.as_str().to_owned(),
                Bytes::copy_from_slice(value.as_bytes()),
            )
        })
        .collect()
}

pub fn decode_headers(headers: Headers) -> anyhow::Result<HeaderMap> {
    let mut result = HeaderMap::new();
    for (name, value) in headers {
        result.append(
            HeaderName::from_bytes(name.as_bytes())?,
            HeaderValue::from_bytes(&value)?,
        );
    }
    strip_hop_by_hop(&mut result);
    Ok(result)
}

fn strip_hop_by_hop(headers: &mut HeaderMap) {
    let nominated: Vec<_> = headers
        .get_all("connection")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|name| HeaderName::from_bytes(name.trim().as_bytes()).ok())
        .collect();
    for name in nominated {
        headers.remove(name);
    }
    for name in [
        "connection",
        "keep-alive",
        "proxy-connection",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
    ] {
        headers.remove(name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize, Deserialize)]
    struct LegacyRequest {
        method: String,
        headers: Headers,
        body: Bytes,
    }

    #[test]
    fn native_generate_wire_is_unchanged_for_legacy_peers() {
        let legacy = LegacyRequest {
            method: "POST".into(),
            headers: Vec::new(),
            body: Bytes::from_static(b"{ \"stream\":false }"),
        };
        let native = Request {
            method: legacy.method.clone(),
            headers: Vec::new(),
            body: legacy.body.clone(),
            operation: Operation::Generate,
        };
        for (old, current) in [
            (
                rmp_serde::to_vec(&legacy).unwrap(),
                rmp_serde::to_vec(&native).unwrap(),
            ),
            (
                rmp_serde::to_vec_named(&legacy).unwrap(),
                rmp_serde::to_vec_named(&native).unwrap(),
            ),
        ] {
            assert_eq!(old, current);
            assert_eq!(
                rmp_serde::from_slice::<Request>(&old).unwrap().operation,
                Operation::Generate
            );
            assert_eq!(
                rmp_serde::from_slice::<LegacyRequest>(&current)
                    .unwrap()
                    .body,
                legacy.body
            );
        }
        let descriptor: lifecycle::Descriptor = serde_json::from_value(serde_json::json!({
            "version": 1, "incarnation": "engine", "header_overrides": true
        }))
        .unwrap();
        assert!(!descriptor.supports_sessions());
    }
}
