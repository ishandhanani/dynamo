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
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use reqwest::{StatusCode, header};
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::{TcpListener, TcpStream},
        sync::oneshot,
        task::JoinHandle,
        time::timeout,
    };

    const DEADLINE: Duration = Duration::from_secs(5);

    struct RecordedRequest {
        head: String,
        body: Vec<u8>,
    }

    async fn read_request(socket: &mut TcpStream) -> RecordedRequest {
        let mut request = Vec::new();
        let mut buffer = [0; 4096];
        let header_end = loop {
            let count = socket.read(&mut buffer).await.unwrap();
            assert_ne!(count, 0, "request ended before its headers");
            request.extend_from_slice(&buffer[..count]);
            assert!(request.len() < 65536, "test request exceeds fixture limit");
            if let Some(end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
                break end + 4;
            }
        };
        let head = String::from_utf8(request[..header_end].to_vec()).unwrap();
        let content_length = head
            .lines()
            .filter_map(|line| line.split_once(':'))
            .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
            .map(|(_, value)| value.trim().parse::<usize>().unwrap())
            .unwrap_or_default();
        assert!(content_length < 65536);
        while request.len() < header_end + content_length {
            let count = socket.read(&mut buffer).await.unwrap();
            assert_ne!(count, 0, "request body was truncated");
            request.extend_from_slice(&buffer[..count]);
        }
        RecordedRequest {
            head,
            body: request[header_end..header_end + content_length].to_vec(),
        }
    }

    fn transport(listener: &TcpListener) -> HttpTransport {
        let endpoint = HttpEndpoint::parse(
            &format!("http://{}", listener.local_addr().unwrap()),
            "test",
        )
        .unwrap();
        HttpTransport::new(endpoint, DEADLINE).unwrap()
    }

    async fn serve_once(response: Vec<u8>) -> (HttpTransport, JoinHandle<RecordedRequest>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let transport = transport(&listener);
        let server = tokio::spawn(async move {
            timeout(DEADLINE, async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let request = read_request(&mut socket).await;
                socket.write_all(&response).await.unwrap();
                request
            })
            .await
            .expect("fixture timed out")
        });
        (transport, server)
    }

    fn response(status: &str, extra_headers: &str, body: &[u8]) -> Vec<u8> {
        let mut response = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n{extra_headers}\r\n",
            body.len(),
        )
        .into_bytes();
        response.extend_from_slice(body);
        response
    }

    #[tokio::test]
    async fn preserves_unary_request_response_status_and_headers() {
        let body = b"{ \"z\":1e-05, \"a\": 2, \"future\": [true] }\n";
        let (transport, server) = serve_once(response(
            "201 Created",
            "Content-Type: application/json\r\nX-Engine-Extension: native\r\nSet-Cookie: a=1\r\nSet-Cookie: b=2\r\n",
            body,
        ))
        .await;
        let request =
            Bytes::from_static(b"{ \"stream\":false, \"text\":\"hello\", \"future\":1.00 }\n");
        let headers = HeaderMap::from_iter([
            (
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("application/json"),
            ),
            (
                header::HeaderName::from_static("x-override-routed-dp-rank"),
                header::HeaderValue::from_static("3"),
            ),
        ]);
        let response = timeout(
            DEADLINE,
            transport.send(Method::PUT, "/generate", headers, request.clone()),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.status(), StatusCode::CREATED);
        assert_eq!(response.headers()["x-engine-extension"], "native");
        assert_eq!(
            response
                .headers()
                .get_all(header::SET_COOKIE)
                .iter()
                .count(),
            2
        );
        assert_eq!(response.bytes().await.unwrap().as_ref(), body);
        let received = server.await.unwrap();
        assert!(received.head.starts_with("PUT /generate HTTP/1.1\r\n"));
        assert!(received.head.contains("x-override-routed-dp-rank: 3\r\n"));
        assert_eq!(received.body, request);
    }

    #[tokio::test]
    async fn preserves_error_status_and_non_json_body() {
        let body = b"\xffengine-specific overload\x00";
        let (transport, server) = serve_once(response(
            "429 Too Many Requests",
            "Retry-After: 7\r\nContent-Type: application/octet-stream\r\n",
            body,
        ))
        .await;
        let response = timeout(
            DEADLINE,
            transport.send(Method::POST, "/generate", HeaderMap::new(), Bytes::new()),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()[header::RETRY_AFTER], "7");
        assert_eq!(response.bytes().await.unwrap().as_ref(), body);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn does_not_decompress_encoded_body() {
        // gzip -n of {"ok":true}; gzip is deliberately enabled in dev-dependencies.
        let body: &[u8] = &[
            31, 139, 8, 0, 0, 0, 0, 0, 0, 3, 171, 86, 202, 207, 86, 178, 42, 41, 42, 77, 173, 5, 0,
            144, 95, 212, 167, 11, 0, 0, 0,
        ];
        let (transport, server) = serve_once(response(
            "200 OK",
            "Content-Encoding: gzip\r\nContent-Type: application/json\r\n",
            body,
        ))
        .await;
        let response = timeout(
            DEADLINE,
            transport.send(Method::POST, "/generate", HeaderMap::new(), Bytes::new()),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.headers()[header::CONTENT_ENCODING], "gzip");
        assert_eq!(response.content_length(), Some(body.len() as u64));
        assert_eq!(response.bytes().await.unwrap().as_ref(), body);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn does_not_follow_generation_redirect() {
        let destination = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let headers = format!(
            "Location: http://{}/generate\r\n",
            destination.local_addr().unwrap()
        );
        let (transport, server) =
            serve_once(response("307 Temporary Redirect", &headers, b"moved")).await;
        let response = timeout(
            DEADLINE,
            transport.send(
                Method::POST,
                "/generate",
                HeaderMap::new(),
                Bytes::from_static(b"{}"),
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
        assert_eq!(response.bytes().await.unwrap().as_ref(), b"moved");
        assert!(
            timeout(Duration::from_millis(50), destination.accept())
                .await
                .is_err()
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn streams_before_completion_and_preserves_sse_bytes() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let transport = transport(&listener);
        let (continue_tx, continue_rx) = oneshot::channel();
        let first = b": engine heartbeat\r\nevent: extension\r\ndata: not-json\r\n\r\n";
        let last = b"data: [DONE]\n\n";
        let server = tokio::spawn(async move {
            timeout(DEADLINE, async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                read_request(&mut socket).await;
                socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n").await.unwrap();
                socket.write_all(format!("{:x}\r\n", first.len()).as_bytes()).await.unwrap();
                socket.write_all(first).await.unwrap();
                socket.write_all(b"\r\n").await.unwrap();
                continue_rx.await.unwrap();
                socket.write_all(format!("{:x}\r\n", last.len()).as_bytes()).await.unwrap();
                socket.write_all(last).await.unwrap();
                socket.write_all(b"\r\n0\r\n\r\n").await.unwrap();
            }).await.expect("stream fixture timed out");
        });
        let response = timeout(
            DEADLINE,
            transport.send(
                Method::POST,
                "/generate",
                HeaderMap::new(),
                Bytes::from_static(b"{\"stream\":true}"),
            ),
        )
        .await
        .unwrap()
        .unwrap();
        let mut stream = response.bytes_stream();
        let mut received = Vec::new();
        while received.len() < first.len() {
            received.extend_from_slice(
                &timeout(DEADLINE, stream.next())
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap(),
            );
        }
        assert_eq!(received, first);
        continue_tx.send(()).unwrap();
        while let Some(chunk) = timeout(DEADLINE, stream.next()).await.unwrap() {
            received.extend_from_slice(&chunk.unwrap());
        }
        assert_eq!(received, [first.as_slice(), last.as_slice()].concat());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn reports_truncated_body_as_failure() {
        let (transport, server) = serve_once(
            b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\npartial".to_vec(),
        )
        .await;
        let response = timeout(
            DEADLINE,
            transport.send(Method::POST, "/generate", HeaderMap::new(), Bytes::new()),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(timeout(DEADLINE, response.bytes()).await.unwrap().is_err());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn dropping_unfinished_response_closes_upstream_connection() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let transport = transport(&listener);
        let server = tokio::spawn(async move {
            timeout(DEADLINE, async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                read_request(&mut socket).await;
                socket
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n")
                    .await
                    .unwrap();
                let mut buffer = [0; 1];
                match socket.read(&mut buffer).await {
                    Ok(0) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::ConnectionReset => {}
                    result => panic!("expected transport disconnect: {result:?}"),
                }
            })
            .await
            .expect("dropping the response left the connection open");
        });
        let response = timeout(
            DEADLINE,
            transport.send(Method::POST, "/generate", HeaderMap::new(), Bytes::new()),
        )
        .await
        .unwrap()
        .unwrap();
        drop(response);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn does_not_replay_when_response_is_lost_after_request_delivery() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let transport = transport(&listener);
        let (stop_tx, stop_rx) = oneshot::channel();
        let server = tokio::spawn(async move {
            timeout(DEADLINE, async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let request = read_request(&mut socket).await;
                assert_eq!(request.body, b"{\"session_params\":{\"id\":\"session\"}}");
                // Engine accepted the full request, but its response is lost.
                drop(socket);
                tokio::select! {
                    biased;
                    connection = listener.accept() => panic!("generation was replayed: {connection:?}"),
                    _ = stop_rx => {},
                }
            }).await.expect("request was retried or hung");
        });
        let result = timeout(
            DEADLINE,
            transport.send(
                Method::POST,
                "/generate",
                HeaderMap::new(),
                Bytes::from_static(b"{\"session_params\":{\"id\":\"session\"}}"),
            ),
        )
        .await
        .unwrap();
        assert!(result.is_err());
        stop_tx.send(()).unwrap();
        server.await.unwrap();
    }
}
