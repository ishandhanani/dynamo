// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::sglang_http::{self, MAX_BODY_CHUNK, Request, ResponseFrame};
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

use super::{NativeHttp, transport::HttpTransport};

pub(crate) struct NativeHttpEndpoint {
    transport: HttpTransport,
    cancel: CancellationToken,
}

impl NativeHttpEndpoint {
    pub(crate) async fn start(
        primary: &Endpoint,
        http: &NativeHttp,
        cancel: CancellationToken,
    ) -> anyhow::Result<StartedEndpoint> {
        let handler = Arc::new(Self {
            transport: http.transport.clone(),
            cancel,
        });
        primary
            .component()
            .endpoint(sglang_http::endpoint_name(&primary.id().name))
            .endpoint_builder()
            .handler(Ingress::for_engine(handler)?)
            .start_with_registration()
            .await
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<Request>, ManyOut<Annotated<ResponseFrame>>, anyhow::Error>
    for NativeHttpEndpoint
{
    async fn generate(
        &self,
        input: SingleIn<Request>,
    ) -> anyhow::Result<ManyOut<Annotated<ResponseFrame>>> {
        let (request, context) = input.into_parts();
        let context = context.context();
        let method = Method::from_bytes(request.method.as_bytes())?;
        anyhow::ensure!(
            method == Method::POST || method == Method::PUT,
            "native generate requires POST or PUT"
        );
        let mut headers = sglang_http::decode_headers(request.headers)?;
        // The HTTP client computes framing from the retained request bytes.
        headers.remove("content-length");
        let response = tokio::select! {
            biased;
            _ = self.cancel.cancelled() => anyhow::bail!("native HTTP endpoint shutting down"),
            _ = context.stopped() => anyhow::bail!("native HTTP request cancelled"),
            _ = context.killed() => anyhow::bail!("native HTTP request killed"),
            response = self.transport.send(method, "/generate", headers, request.body) => response?,
        };
        let head = ResponseFrame::Head {
            status: response.status().as_u16(),
            headers: sglang_http::encode_headers(response.headers().clone()),
        };
        let mut upstream = response.bytes_stream();
        let cancel = self.cancel.clone();
        let stream_context = context.clone();
        let stream = async_stream::stream! {
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
    use axum::{Router, body::Body, http::HeaderMap, response::Response, routing::any};
    use bytes::Bytes;
    use dynamo_llm::http::service::native_generate::{NativeGenerateClient, forward};
    use dynamo_runtime::pipeline::{Context, network::egress::push_router::RouterMode};
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
    use dynamo_sidecar_common::HttpEndpoint;
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn runtime_round_trip_preserves_unary_and_live_stream() {
        timeout(Duration::from_secs(20), async {
            const UNARY_REQUEST: &[u8] =
                b"{ \"stream\":false, \"input_ids\": [1,2], \"future\":1.00 }\n";
            const STREAM_REQUEST: &[u8] =
                b"{ \"stream\":true, \"input_ids\": [1,2], \"future\":1.00 }\n";
            let release = Arc::new(Notify::new());
            let closed = Arc::new(Notify::new());
            let engine_closed = closed.clone();
            let engine_release = release.clone();
            let first = Bytes::from_static(b": heartbeat\r\ndata: not-json\r\n\r\n");
            let last = Bytes::from_static(b"data: [DONE]\n\n");
            let unary = Bytes::from_static(b"{ \"future\":1e-05, \"engine\":\"overloaded\" }\n");
            let app = Router::new().route(
                "/generate",
                any(move |method: Method, headers: HeaderMap, body: Bytes| {
                    let release = engine_release.clone();
                    let closed = engine_closed.clone();
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
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let http = NativeHttp {
                transport: HttpTransport::new(
                    HttpEndpoint::parse(
                        &format!("http://{}", listener.local_addr().unwrap()),
                        "test",
                    )
                    .unwrap(),
                    Duration::from_secs(5),
                )
                .unwrap(),
            };
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
            let started = NativeHttpEndpoint::start(&primary, &http, CancellationToken::new())
                .await
                .unwrap();
            let worker = started.instance().id();
            let endpoint = primary
                .component()
                .endpoint(sglang_http::endpoint_name("generate"));
            let client = endpoint.client().await.unwrap();
            client.wait_for_instances().await.unwrap();
            let client = NativeGenerateClient::from_client(client, RouterMode::Direct)
                .await
                .unwrap();
            for mode in ["unary", "stream", "truncated", "cancel"] {
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
                    headers: sglang_http::encode_headers(headers),
                    body: Bytes::from_static(if mode == "unary" {
                        UNARY_REQUEST
                    } else {
                        STREAM_REQUEST
                    }),
                };
                let response = forward(&client, Context::new(request), worker)
                    .await
                    .unwrap();
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
}
