// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::http::{self, MAX_BODY_CHUNK, Request, ResponseFrame};
use dynamo_runtime::{
    component::{Endpoint, StartedEndpoint},
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn,
        network::Ingress,
    },
    protocols::annotated::Annotated,
};
use futures::StreamExt;
use reqwest::Method;
use tokio_util::sync::CancellationToken;

mod transport;
pub use transport::HttpTransport;

pub struct HttpProxy {
    transport: HttpTransport,
    cancel: CancellationToken,
    paths: &'static [&'static str],
}

impl HttpProxy {
    pub async fn start(
        primary: &Endpoint,
        transport: HttpTransport,
        paths: &'static [&'static str],
        cancel: CancellationToken,
    ) -> anyhow::Result<StartedEndpoint> {
        let handler = Arc::new(Self {
            transport,
            cancel,
            paths,
        });
        primary
            .component()
            .endpoint(http::endpoint_name(&primary.id().name))
            .endpoint_builder()
            .handler(Ingress::for_engine(handler)?)
            .start_with_registration()
            .await
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<Request>, ManyOut<Annotated<ResponseFrame>>, anyhow::Error>
    for HttpProxy
{
    async fn generate(
        &self,
        input: SingleIn<Request>,
    ) -> anyhow::Result<ManyOut<Annotated<ResponseFrame>>> {
        let (request, context) = input.into_parts();
        let context = context.context();
        let method = Method::from_bytes(request.method.as_bytes())?;
        anyhow::ensure!(
            self.paths
                .contains(&request.path.split('?').next().unwrap_or_default()),
            "HTTP path is not enabled by this sidecar"
        );
        let mut headers = http::decode_headers(request.headers)?;
        // The HTTP client computes framing from the retained request bytes.
        headers.remove("content-length");
        let transport = self.transport.clone();
        let cancel = self.cancel.clone();
        let stream_context = context.clone();
        // Establish the runtime response stream before waiting for HTTP headers.
        // Its transport carries cancellation before the upstream server replies.
        let stream = async_stream::stream! {
            let response = tokio::select! {
                biased;
                _ = cancel.cancelled() => None,
                _ = stream_context.stopped() => None,
                _ = stream_context.killed() => None,
                response = transport.send(method, &request.path, headers, request.body) => Some(response),
            };
            let response = match response {
                Some(Ok(response)) => response,
                Some(Err(error)) => {
                    yield Annotated::from_error(error.to_string());
                    return;
                }
                None => {
                    yield Annotated::from_error("native HTTP request cancelled");
                    return;
                }
            };
            let head = ResponseFrame::Head {
                status: response.status().as_u16(),
                headers: http::encode_headers(response.headers().clone()),
            };
            let mut upstream = response.bytes_stream();
            yield Annotated::from_data(head);
            loop {
                let next = tokio::select! {
                    biased;
                    _ = cancel.cancelled() => break,
                    _ = stream_context.stopped() => break,
                    _ = stream_context.killed() => break,
                    next = upstream.next() => next,
                };
                match next {
                    Some(Ok(mut bytes)) => {
                        while !bytes.is_empty() {
                            let chunk = bytes.split_to(bytes.len().min(MAX_BODY_CHUNK));
                            yield Annotated::from_data(ResponseFrame::Body(chunk));
                        }
                    }
                    Some(Err(_)) => {
                        yield Annotated::from_error("native HTTP upstream body failed");
                        return;
                    }
                    None => {
                        yield Annotated::from_data(ResponseFrame::End);
                        return;
                    }
                }
            }
            // Closing transport interest is not an engine-stop acknowledgement.
            yield Annotated::from_error("native HTTP request cancelled");
        };
        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::HttpEndpoint;
    use axum::{Router, body::Body, http::HeaderMap, response::Response, routing::any};
    use bytes::Bytes;
    use dynamo_llm::http::service::http_proxy::{HttpClient, forward};
    use dynamo_runtime::pipeline::{Context, network::egress::push_router::RouterMode};
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
    use tokio::{
        net::TcpListener,
        sync::Notify,
        time::{Duration, timeout},
    };

    struct OnDrop(Option<Arc<Notify>>);

    impl Drop for OnDrop {
        fn drop(&mut self) {
            if let Some(notify) = &self.0 {
                notify.notify_one();
            }
        }
    }

    #[test]
    fn runtime_round_trip_preserves_unary_and_live_stream() {
        test_runtime().block_on(check_runtime_round_trip());
    }

    async fn check_runtime_round_trip() {
        const GZIP: &[u8] = &[
            31, 139, 8, 0, 0, 0, 0, 0, 0, 3, 171, 86, 202, 207, 86, 178, 42, 41, 42, 77, 173, 5, 0,
            144, 95, 212, 167, 11, 0, 0, 0,
        ];
        timeout(Duration::from_secs(20), async {
            const UNARY_REQUEST: &[u8] =
                b"{ \"stream\":false, \"input_ids\": [1,2], \"future\":1.00 }\n";
            const STREAM_REQUEST: &[u8] =
                b"{ \"stream\":true, \"input_ids\": [1,2], \"future\":1.00 }\n";
            let release = Arc::new(Notify::new());
            let closed = Arc::new(Notify::new());
            let waiting_for_headers = Arc::new(Notify::new());
            let engine_closed = closed.clone();
            let engine_release = release.clone();
            let engine_waiting = waiting_for_headers.clone();
            let first = Bytes::from_static(b": heartbeat\r\ndata: not-json\r\n\r\n");
            let last = Bytes::from_static(b"data: [DONE]\n\n");
            let unary = Bytes::from_static(b"{ \"future\":1e-05, \"engine\":\"overloaded\" }\n");
            let app = Router::new().route(
                "/fixture",
                any(move |method: Method, headers: HeaderMap, body: Bytes| {
                    let release = engine_release.clone();
                    let closed = engine_closed.clone();
                    let waiting_for_headers = engine_waiting.clone();
                    let (first, last, unary) = (first.clone(), last.clone(), unary.clone());
                    async move {
                        assert_eq!(method, Method::PUT);
                        assert!(!headers.contains_key("x-client-hop"));
                        assert_eq!(headers["x-override-routed-dp-rank"], "3");
                        let streaming = headers["x-fixture-mode"] != "unary";
                        assert_eq!(
                            body.as_ref(),
                            if streaming {
                                STREAM_REQUEST
                            } else {
                                UNARY_REQUEST
                            }
                        );
                        if headers["x-fixture-mode"] == "cancel_before_headers" {
                            let _dropped = OnDrop(Some(closed));
                            waiting_for_headers.notify_one();
                            std::future::pending::<()>().await;
                            unreachable!();
                        }
                        let truncated = headers["x-fixture-mode"] == "truncated";
                        let dropped =
                            OnDrop((headers["x-fixture-mode"] == "cancel").then_some(closed));
                        let body = Body::from_stream(async_stream::stream! {
                            let _dropped = dropped;
                            if streaming {
                                yield Ok::<_, std::io::Error>(first);
                            release.notified().await;
                            if truncated {
                                yield Err(std::io::Error::other("upstream disconnected"));
                            } else {
                                yield Ok(last);
                            }
                            } else {
                                yield Ok(unary);
                            }
                        });
                        Response::builder()
                            .status(if streaming { 200 } else { 429 })
                            .header(
                                "content-type",
                                if streaming {
                                    "text/event-stream"
                                } else {
                                    "application/json"
                                },
                            )
                            .header("set-cookie", "a=1")
                            .header("set-cookie", "b=2")
                            .header("connection", "x-engine-hop")
                            .header("x-engine-hop", "private")
                            .body(body)
                            .unwrap()
                    }
                }),
            );
            let app = app.route(
                "/bytes",
                any(|method: Method, uri: axum::http::Uri| async move {
                    assert_eq!(method, Method::GET);
                    assert_eq!(uri.query(), Some("format=raw"));
                    Response::builder()
                        .status(307)
                        .header("location", "/must-not-follow")
                        .header("content-encoding", "gzip")
                        .body(Body::from(GZIP))
                        .unwrap()
                }),
            );
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let transport = HttpTransport::new(
                HttpEndpoint::parse(
                    &format!("http://{}", listener.local_addr().unwrap()),
                    "test",
                )
                .unwrap(),
                Duration::from_secs(5),
            )
            .unwrap();
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let runtime = Runtime::from_current().unwrap();
            let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
            let primary = drt
                .namespace("native_http_test")
                .unwrap()
                .component("sidecar")
                .unwrap()
                .endpoint("generate");
            let started = HttpProxy::start(
                &primary,
                transport,
                &["/fixture", "/bytes"],
                CancellationToken::new(),
            )
            .await
            .unwrap();
            let worker = started.instance().id();
            let endpoint = primary
                .component()
                .endpoint(http::endpoint_name("generate"));
            let client = endpoint.client().await.unwrap();
            client.wait_for_instances().await.unwrap();
            let client = HttpClient::from_client(client, RouterMode::Direct)
                .await
                .unwrap();
            let response = forward(
                &client,
                Context::new(Request {
                    method: "GET".into(),
                    path: "/bytes?format=raw".into(),
                    headers: vec![],
                    body: Bytes::new(),
                }),
                worker,
            )
            .await
            .unwrap();
            assert_eq!(response.status(), 307);
            assert_eq!(response.headers()["content-encoding"], "gzip");
            assert_eq!(
                axum::body::to_bytes(response.into_body(), 1024)
                    .await
                    .unwrap(),
                GZIP
            );
            assert!(
                forward(
                    &client,
                    Context::new(Request {
                        method: "GET".into(),
                        path: "/not-registered".into(),
                        headers: vec![],
                        body: Bytes::new(),
                    }),
                    worker
                )
                .await
                .is_err()
            );
            for mode in [
                "unary",
                "stream",
                "truncated",
                "cancel",
                "cancel_before_headers",
            ] {
                let headers = HeaderMap::from_iter([
                    ("x-fixture-mode".parse().unwrap(), mode.parse().unwrap()),
                    (
                        "x-override-routed-dp-rank".parse().unwrap(),
                        "3".parse().unwrap(),
                    ),
                    (
                        "connection".parse().unwrap(),
                        "x-client-hop".parse().unwrap(),
                    ),
                    ("x-client-hop".parse().unwrap(), "private".parse().unwrap()),
                ]);
                let request = Request {
                    method: "PUT".into(),
                    path: "/fixture".into(),
                    headers: http::encode_headers(headers),
                    body: Bytes::from_static(if mode == "unary" {
                        UNARY_REQUEST
                    } else {
                        STREAM_REQUEST
                    }),
                };
                let mut pending = Box::pin(forward(&client, Context::new(request), worker));
                if mode == "cancel_before_headers" {
                    tokio::select! {
                        _ = waiting_for_headers.notified() => {},
                        _ = &mut pending => panic!("upstream must withhold response headers"),
                    }
                    drop(pending);
                    timeout(Duration::from_secs(2), closed.notified())
                        .await
                        .expect("cancellation must close upstream before response headers");
                    continue;
                }
                let response = pending.await.unwrap();
                assert_eq!(
                    response.status().as_u16(),
                    if mode != "unary" { 200 } else { 429 }
                );
                assert_eq!(response.headers().get_all("set-cookie").iter().count(), 2);
                assert!(!response.headers().contains_key("x-engine-hop"));
                let mut body = response.into_body().into_data_stream();
                let mut received = Vec::new();
                if mode != "unary" {
                    while received.len() < b": heartbeat\r\ndata: not-json\r\n\r\n".len() {
                        received.extend_from_slice(&body.next().await.unwrap().unwrap());
                    }
                    assert_eq!(received, b": heartbeat\r\ndata: not-json\r\n\r\n");
                    if mode == "cancel" {
                        drop(body);
                        closed.notified().await;
                        continue;
                    }
                    release.notify_one();
                }
                if mode == "truncated" {
                    assert!(body.next().await.unwrap().is_err());
                    continue;
                }
                while let Some(bytes) = body.next().await {
                    received.extend_from_slice(&bytes.unwrap());
                }
                let expected: &[u8] = if mode == "stream" {
                    b": heartbeat\r\ndata: not-json\r\n\r\ndata: [DONE]\n\n"
                } else {
                    b"{ \"future\":1e-05, \"engine\":\"overloaded\" }\n"
                };
                assert_eq!(received, expected);
            }

            started.shutdown().await.unwrap();
            runtime.shutdown();
            server.abort();
        })
        .await
        .unwrap();
    }

    // The request-plane TCP listener is process-wide. Its Tokio executor must
    // outlive every fixture that registers an endpoint on that listener.
    fn test_runtime() -> &'static tokio::runtime::Runtime {
        static RUNTIME: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
        RUNTIME.get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap()
        })
    }
}
