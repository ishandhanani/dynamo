// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::{
    body::{Body, to_bytes},
    response::Response,
};
use bytes::Bytes;
use serde_json::Value;

pub(super) const MAX_BODY: usize = 64 * 1024 * 1024;

// Match SMG's scalar JSON merge. Stock streaming prefill returns SSE, which
// cannot supply this merge; batch responses also pass through unchanged.
pub(super) async fn merge_response(decode: Response, prefill: Bytes) -> anyhow::Result<Response> {
    let prompt = serde_json::from_slice::<Value>(&prefill)
        .ok()
        .and_then(|mut value| {
            value
                .pointer_mut("/meta_info/input_token_logprobs")
                .map(Value::take)
        });
    let Some(Value::Array(mut prompt)) = prompt else {
        return Ok(decode);
    };
    let (mut parts, body) = decode.into_parts();
    let original = to_bytes(body, MAX_BODY).await?;
    let mut value: Value = serde_json::from_slice(&original).unwrap_or(Value::Null);
    let body = match value
        .pointer_mut("/meta_info/input_token_logprobs")
        .and_then(Value::as_array_mut)
    {
        Some(tail) => {
            prompt.append(tail);
            *tail = prompt;
            for header in ["content-length", "etag", "content-md5"] {
                parts.headers.remove(header);
            }
            Bytes::from(serde_json::to_vec(&value)?)
        }
        None => original,
    };
    Ok(Response::from_parts(parts, Body::from(body)))
}
