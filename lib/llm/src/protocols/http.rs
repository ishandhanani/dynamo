// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Versioned native HTTP request plane, separate from the token pipeline.
//! Bodies use binary MessagePack fields; endpoint semantics belong to the caller.

use axum::http::{HeaderMap, HeaderName, HeaderValue};
use bytes::Bytes;
use serde::{Deserialize, Serialize};

pub const MAX_BODY_CHUNK: usize = 64 * 1024;

pub fn endpoint_name(primary: &str) -> String {
    format!("{primary}_native_http_v1")
}

/// A list preserves repeated headers and non-UTF-8 header values.
pub type Headers = Vec<(String, Bytes)>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Headers,
    pub body: Bytes,
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
