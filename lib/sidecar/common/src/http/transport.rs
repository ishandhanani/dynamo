// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte-preserving upstream HTTP transport, independent of generation semantics.
//!
//! HTTP status, end-to-end headers and body bytes remain engine-owned. Callers
//! decide whether to forward a response or adapt it for the legacy token pipeline.
//! This is the engine hop, not the frontend/sidecar wire protocol: a forwarding
//! handler must still remove hop-by-hop headers at its inbound/outbound boundaries.

use std::time::Duration;

use crate::HttpEndpoint;
use bytes::Bytes;
use reqwest::{Client, Method, Response, header::HeaderMap};

#[derive(Clone)]
pub struct HttpTransport {
    client: Client,
    pub endpoint: HttpEndpoint,
}

impl HttpTransport {
    pub fn new(endpoint: HttpEndpoint, connect_timeout: Duration) -> Result<Self, reqwest::Error> {
        let client = Client::builder()
            .connect_timeout(connect_timeout)
            // A redirect can replay generation on a different worker. Return it
            // to the caller, just like every other upstream status.
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            // Explicitly disable these even when Cargo feature unification
            // enables them: decoding changes both body bytes and headers.
            .no_gzip()
            .no_brotli()
            .no_deflate()
            .no_zstd()
            .build()?;
        Ok(Self { client, endpoint })
    }

    /// Return as soon as upstream headers arrive, without consuming the body.
    ///
    /// There is no response timeout, status translation, JSON/SSE parsing, or
    /// application retry. The caller owns cancellation and polls `bytes_stream()`
    /// to consume the body with backpressure. Dropping it ends transport interest;
    /// it does not acknowledge that the engine has stopped executing.
    pub async fn send(
        &self,
        method: Method,
        path: &str,
        headers: HeaderMap,
        body: Bytes,
    ) -> Result<Response, reqwest::Error> {
        let (path, query) = path
            .split_once('?')
            .map_or((path, None), |(p, q)| (p, Some(q)));
        let mut url = self.endpoint.with_path(path);
        url.set_query(query);
        self.client
            .request(method, url)
            .headers(headers)
            .body(body)
            .send()
            .await
    }
}
