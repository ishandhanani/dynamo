// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang's HTTP P/D router concatenates prefill and decode prompt logprobs.
//! This adapter is used only for disaggregated requests with return_logprob.

use axum::{body::Body, http::header, response::Response};
use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use serde_json::Value;

pub(super) const MAX_BODY: usize = 64 * 1024 * 1024;

pub(super) async fn merge_response(decode: Response, prefill: Bytes) -> anyhow::Result<Response> {
    let Ok(prefill) = serde_json::from_slice::<Value>(&prefill) else {
        return Ok(decode);
    };
    let streaming = decode
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/event-stream"));
    let (mut parts, body) = decode.into_parts();
    // The merged body has a different length and representation.
    for name in ["content-length", "etag", "content-md5"] {
        parts.headers.remove(name);
    }
    let body = if streaming {
        let stream: futures::stream::BoxStream<'static, anyhow::Result<Bytes>> = Box::pin(
            async_stream::try_stream! {
                let mut upstream = body.into_data_stream();
                let mut pending = BytesMut::new();
                let mut scan = 0;
                while let Some(chunk) = upstream.next().await {
                    pending.extend_from_slice(&chunk?);
                    while let Some(end) = event_end(&pending, &mut scan) {
                        if end > MAX_BODY { Err(anyhow::anyhow!("native logprob SSE event exceeds body limit"))?; }
                        let event = pending.split_to(end).freeze();
                        scan = 0;
                        yield merge_event(event, &prefill)?;
                    }
                    if pending.len() > MAX_BODY { Err(anyhow::anyhow!("native logprob SSE event exceeds body limit"))?; }
                }
                if !pending.is_empty() { Err(anyhow::anyhow!("native logprob SSE response ended mid-event"))?; }
            },
        );
        Body::from_stream(stream)
    } else {
        let original = axum::body::to_bytes(body, MAX_BODY).await?;
        let mut decode = serde_json::from_slice::<Value>(&original).ok();
        if decode
            .as_mut()
            .is_some_and(|decode| merge_json(&prefill, decode))
        {
            Body::from(serde_json::to_vec(&decode)?)
        } else {
            Body::from(original)
        }
    };
    Ok(Response::from_parts(parts, body))
}

// Scan each byte once even if an event arrives one byte at a time. Keep the
// possible delimiter suffix across chunks; HTTP chunks are not SSE events.
fn event_end(bytes: &[u8], scan: &mut usize) -> Option<usize> {
    for i in *scan..bytes.len() {
        if bytes[i] == b'\n'
            && (i > 0 && bytes[i - 1] == b'\n' || i > 2 && &bytes[i - 3..i] == b"\r\n\r")
        {
            return Some(i + 1);
        }
    }
    *scan = bytes.len();
    None
}

fn merge_event(event: Bytes, prefill: &Value) -> anyhow::Result<Bytes> {
    let Ok(text) = std::str::from_utf8(&event) else {
        return Ok(event);
    };
    let data = text
        .lines()
        .filter_map(|line| {
            line.strip_prefix("data:")
                .map(|s| s.strip_prefix(' ').unwrap_or(s))
        })
        .collect::<Vec<_>>()
        .join("\n");
    if data.is_empty() || data.trim() == "[DONE]" {
        return Ok(event);
    }
    let Ok(mut decode) = serde_json::from_str::<Value>(&data) else {
        return Ok(event);
    };
    if !merge_json(prefill, &mut decode) {
        return Ok(event);
    }
    let mut result = String::new();
    // Preserve event/id/retry/comment fields while replacing only data lines.
    for line in text
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with("data:"))
    {
        result.push_str(line);
        result.push('\n');
    }
    result.push_str("data: ");
    result.push_str(&serde_json::to_string(&decode)?);
    result.push_str("\n\n");
    Ok(result.into())
}

fn merge_json(prefill: &Value, decode: &mut Value) -> bool {
    if let Some(outputs) = decode.as_array_mut() {
        return outputs
            .iter_mut()
            .enumerate()
            .fold(false, |changed, (i, output)| {
                merge_json(
                    prefill.as_array().and_then(|v| v.get(i)).unwrap_or(prefill),
                    output,
                ) || changed
            });
    }
    let prefill = match prefill.as_array() {
        Some(outputs) => match decode
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|i| outputs.get(i as usize))
        {
            Some(output) => output,
            None => return false,
        },
        None => prefill,
    };
    let Some(prompt) = prefill
        .pointer("/meta_info/input_token_logprobs")
        .and_then(Value::as_array)
    else {
        return false;
    };
    let Some(tail) = decode
        .pointer_mut("/meta_info/input_token_logprobs")
        .and_then(Value::as_array_mut)
    else {
        return false;
    };
    let mut merged = Vec::with_capacity(prompt.len() + tail.len());
    merged.extend(prompt.iter().cloned());
    merged.append(tail);
    *tail = merged;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn native_pd_logprobs_merge_unary_and_split_sse() {
        let prefill = json!({"meta_info":{"input_token_logprobs":[[null,1],[-0.1,2]]}});
        let decode = json!({"text":"out","meta_info":{"input_token_logprobs":[[-0.2,3]],"output_token_logprobs":[[-0.3,4]],"future":7}});
        for streaming in [false, true] {
            let prompt = Bytes::from(serde_json::to_vec(&prefill).unwrap());
            let wire = if streaming {
                format!("event: token\r\ndata: {decode}\r\n\r\ndata: [DONE]\n\n")
            } else {
                decode.to_string()
            };
            let chunks: Vec<_> = wire
                .bytes()
                .map(|b| Ok::<_, std::io::Error>(Bytes::from(vec![b])))
                .collect();
            let response = Response::builder()
                .header(
                    "content-type",
                    if streaming {
                        "text/event-stream"
                    } else {
                        "application/json"
                    },
                )
                .header("content-length", wire.len())
                .header("x-engine", "kept")
                .body(Body::from_stream(futures::stream::iter(chunks)))
                .unwrap();
            let merged = merge_response(response, prompt).await.unwrap();
            assert!(!merged.headers().contains_key("content-length"));
            assert_eq!(merged.headers()["x-engine"], "kept");
            let body = axum::body::to_bytes(merged.into_body(), 4096)
                .await
                .unwrap();
            let text = std::str::from_utf8(&body).unwrap();
            let output: Value = serde_json::from_str(if streaming {
                assert!(text.starts_with("event: token\n"));
                assert!(text.ends_with("data: [DONE]\n\n"));
                text.lines()
                    .find_map(|line| line.strip_prefix("data: "))
                    .unwrap()
            } else {
                text
            })
            .unwrap();
            assert_eq!(
                output["meta_info"]["input_token_logprobs"]
                    .as_array()
                    .unwrap()
                    .len(),
                3
            );
            assert_eq!(
                output["meta_info"]["output_token_logprobs"],
                decode["meta_info"]["output_token_logprobs"]
            );
            assert_eq!(output["meta_info"]["future"], 7);
        }
    }

    #[test]
    fn native_pd_logprobs_match_batched_indices_and_preserve_unrelated_events() {
        let prefill = json!([{"meta_info":{"input_token_logprobs":[1]}},{"meta_info":{"input_token_logprobs":[2]}}]);
        let mut decode = json!({"index":1,"meta_info":{"input_token_logprobs":[3]}});
        assert!(merge_json(&prefill, &mut decode));
        assert_eq!(decode["meta_info"]["input_token_logprobs"], json!([2, 3]));
        for event in [
            b": keepalive\n\n".as_slice(),
            b"data: not-json\n\n",
            b"data: [DONE]\r\n\r\n",
            b"data: {\"error\":\"engine\"}\n\n",
        ] {
            let event = Bytes::copy_from_slice(event);
            assert_eq!(merge_event(event.clone(), &prefill).unwrap(), event);
        }
    }
}
