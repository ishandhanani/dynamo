// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte-preserving upstream HTTP transport, independent of generation semantics.
//!
//! HTTP status, end-to-end headers and body bytes remain engine-owned. Callers
//! decide whether to forward a response or adapt it for the legacy token pipeline.
//! This is the engine hop, not the frontend/sidecar wire protocol: a forwarding
//! handler must still remove hop-by-hop headers at its inbound/outbound boundaries.

use std::time::Duration;

use bytes::Bytes;
use dynamo_sidecar_common::HttpEndpoint;
use reqwest::{Client, Method, Response, header::HeaderMap};

#[derive(Clone)]
pub(super) struct HttpTransport {
    client: Client,
    pub(super) endpoint: HttpEndpoint,
}

impl HttpTransport {
    pub(super) fn new(
        endpoint: HttpEndpoint,
        connect_timeout: Duration,
    ) -> Result<Self, reqwest::Error> {
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
    pub(super) async fn send(
        &self,
        method: Method,
        path: &str,
        headers: HeaderMap,
        body: Bytes,
    ) -> Result<Response, reqwest::Error> {
        self.client
            .request(method, self.endpoint.with_path(path))
            .headers(headers)
            .body(body)
            .send()
            .await
    }

    pub(super) async fn get_with_query(
        &self,
        path: &str,
        query: &[(&str, String)],
        headers: HeaderMap,
    ) -> Result<Response, reqwest::Error> {
        self.client
            .get(self.endpoint.with_path(path))
            .query(query)
            .headers(headers)
            .send()
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, body::Body, http::StatusCode, response::Response, routing::any};
    use futures::StreamExt;
    use tokio::{net::TcpListener, sync::oneshot, task::JoinHandle, time::timeout};

    const DEADLINE: Duration = Duration::from_secs(5);

    struct Server(JoinHandle<()>);

    impl Drop for Server {
        fn drop(&mut self) {
            self.0.abort();
        }
    }

    async fn serve(app: Router) -> (HttpTransport, Server) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = HttpEndpoint::parse(
            &format!("http://{}", listener.local_addr().unwrap()),
            "test",
        )
        .unwrap();
        let server = Server(tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        }));
        (HttpTransport::new(endpoint, DEADLINE).unwrap(), server)
    }

    #[tokio::test]
    async fn preserves_request_and_upstream_responses() {
        // gzip -n of {"ok":true}; gzip is enabled in dev-dependencies.
        let gzip: &[u8] = &[
            31, 139, 8, 0, 0, 0, 0, 0, 0, 3, 171, 86, 202, 207, 86, 178, 42, 41, 42, 77, 173, 5, 0,
            144, 95, 212, 167, 11, 0, 0, 0,
        ];
        let request = Bytes::from_static(b"{ \"stream\":false, \"future\":1.00 }\n");
        for (status, body, encoding) in [
            (
                StatusCode::CREATED,
                b"{ \"z\":1e-05, \"a\": 2 }\n".as_slice(),
                "identity",
            ),
            (
                StatusCode::TOO_MANY_REQUESTS,
                b"\xffengine overload\x00",
                "identity",
            ),
            (StatusCode::TEMPORARY_REDIRECT, b"moved", "identity"),
            (StatusCode::OK, gzip, "gzip"),
        ] {
            let expected = request.clone();
            let app = Router::new().route(
                "/generate",
                any(
                    move |method: Method, headers: HeaderMap, received: Bytes| async move {
                        assert_eq!(method, Method::PUT);
                        assert_eq!(headers["x-override-routed-dp-rank"], "3");
                        assert_eq!(received, expected);
                        Response::builder()
                            .status(status)
                            .header("content-encoding", encoding)
                            .header("location", "/must-not-follow")
                            .header("retry-after", "7")
                            .header("set-cookie", "a=1")
                            .header("set-cookie", "b=2")
                            .body(Body::from(body))
                            .unwrap()
                    },
                ),
            );
            let (transport, _server) = serve(app).await;
            timeout(DEADLINE, async {
                let headers = HeaderMap::from_iter([(
                    "x-override-routed-dp-rank".parse().unwrap(),
                    "3".parse().unwrap(),
                )]);
                let response = transport
                    .send(Method::PUT, "/generate", headers, request.clone())
                    .await
                    .unwrap();
                assert_eq!(response.status(), status);
                assert_eq!(response.headers()["content-encoding"], encoding);
                assert_eq!(response.headers()["retry-after"], "7");
                assert_eq!(response.headers().get_all("set-cookie").iter().count(), 2);
                assert_eq!(response.bytes().await.unwrap().as_ref(), body);
            })
            .await
            .unwrap();
        }
    }

    #[tokio::test]
    async fn streams_bytes_before_completion() {
        let (continue_tx, continue_rx) = oneshot::channel();
        let first = b": heartbeat\r\nevent: extension\r\ndata: not-json\r\n\r\n";
        let last = b"data: [DONE]\n\n";
        let body = Body::from_stream(async_stream::stream! {
            yield Ok::<_, std::io::Error>(Bytes::from_static(first));
            continue_rx.await.unwrap();
            yield Ok(Bytes::from_static(last));
        });
        let body = std::sync::Arc::new(std::sync::Mutex::new(Some(body)));
        let app = Router::new().route(
            "/generate",
            any(move || async move {
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(body.lock().unwrap().take().unwrap())
                    .unwrap()
            }),
        );
        let (transport, _server) = serve(app).await;
        timeout(DEADLINE, async {
            let response = transport
                .send(Method::POST, "/generate", HeaderMap::new(), Bytes::new())
                .await
                .unwrap();
            let mut stream = response.bytes_stream();
            let mut received = Vec::new();
            while received.len() < first.len() {
                received.extend_from_slice(&stream.next().await.unwrap().unwrap());
            }
            assert_eq!(received, first);
            continue_tx.send(()).unwrap();
            while let Some(chunk) = stream.next().await {
                received.extend_from_slice(&chunk.unwrap());
            }
            assert_eq!(received, [first.as_slice(), last.as_slice()].concat());
        })
        .await
        .unwrap();
    }
}
