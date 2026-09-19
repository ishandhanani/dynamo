// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use dynamo_backend_common::sglang_http::lifecycle::{
    self, Descriptor, INCARNATION_HEADER, Operation, Request, Response, Snapshot,
};
use dynamo_runtime::{
    component::{Endpoint, StartedEndpoint},
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn,
        network::Ingress,
    },
    protocols::annotated::Annotated,
};
use futures::{StreamExt, stream};
use reqwest::{Method, StatusCode, header};
use tokio_util::sync::CancellationToken;

use super::{NativeHttp, transport::HttpTransport};

const CONTROL_TIMEOUT: Duration = Duration::from_secs(25);
const MAX_CONTROL_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone)]
pub(crate) struct LifecycleClient {
    transport: HttpTransport,
    descriptor: Descriptor,
}

impl LifecycleClient {
    pub(crate) async fn discover(http: &NativeHttp) -> anyhow::Result<Option<Self>> {
        tokio::time::timeout(CONTROL_TIMEOUT, async {
            let response = http
                .transport
                .send(
                    Method::GET,
                    "/server_info",
                    header::HeaderMap::new(),
                    Bytes::new(),
                )
                .await?;
            if response.status() == StatusCode::NOT_FOUND {
                return Ok(None);
            }
            let response = response.error_for_status()?;
            let info: serde_json::Value = serde_json::from_slice(&read_body(response).await?)?;
            let Some(descriptor) = info
                .get("request_lifecycle")
                .filter(|value| !value.is_null())
            else {
                return Ok(None);
            };
            let descriptor: Descriptor = serde_json::from_value(descriptor.clone())?;
            anyhow::ensure!(
                descriptor.version == 1,
                "unsupported SGLang lifecycle version"
            );
            anyhow::ensure!(
                descriptor.header_overrides,
                "SGLang lifecycle routing requires request header overrides"
            );
            anyhow::ensure!(
                lifecycle::is_valid_id(&descriptor.incarnation),
                "invalid SGLang lifecycle incarnation"
            );
            Ok(Some(Self {
                transport: http.transport.clone(),
                descriptor,
            }))
        })
        .await?
    }

    async fn request(&self, request: Request) -> anyhow::Result<Response> {
        let Request::Attempt {
            incarnation,
            attempt_id,
            operation,
        } = request
        else {
            let current = Self::discover(&NativeHttp {
                transport: self.transport.clone(),
            })
            .await?;
            return Ok(match current {
                Some(current) => Response::Descriptor(current.descriptor),
                None => Response::Rejected { status: 501 },
            });
        };
        anyhow::ensure!(
            lifecycle::is_valid_id(&incarnation),
            "invalid lifecycle incarnation"
        );
        anyhow::ensure!(
            lifecycle::is_valid_id(&attempt_id),
            "invalid lifecycle attempt ID"
        );
        let path = format!("/request_lifecycle/{attempt_id}");
        let headers = header::HeaderMap::from_iter([
            (
                header::HeaderName::from_static(INCARNATION_HEADER),
                header::HeaderValue::from_str(&incarnation)?,
            ),
            (
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("application/json"),
            ),
        ]);
        let acknowledged = matches!(operation, Operation::Acknowledge);
        let response = match operation {
            Operation::Snapshot { after } => {
                self.transport
                    .get_with_query(&path, &[("after", after.to_string())], headers)
                    .await?
            }
            operation => {
                let control = match operation {
                    Operation::Renew { lease_seconds } => {
                        anyhow::ensure!(
                            (1..=60).contains(&lease_seconds),
                            "invalid lifecycle lease"
                        );
                        serde_json::json!({"action": "renew", "lease_seconds": lease_seconds})
                    }
                    Operation::Cancel => serde_json::json!({"action": "cancel"}),
                    Operation::Acknowledge => serde_json::json!({"action": "acknowledge"}),
                    Operation::Snapshot { .. } => {
                        anyhow::bail!("invalid lifecycle control operation")
                    }
                };
                self.transport
                    .send(
                        Method::POST,
                        &path,
                        headers,
                        serde_json::to_vec(&control)?.into(),
                    )
                    .await?
            }
        };
        if !response.status().is_success() {
            return Ok(Response::Rejected {
                status: response.status().as_u16(),
            });
        }
        if acknowledged {
            anyhow::ensure!(
                response.status() == StatusCode::NO_CONTENT,
                "invalid lifecycle acknowledgement"
            );
            return Ok(Response::Acknowledged);
        }
        let snapshot: Snapshot = serde_json::from_slice(&read_body(response).await?)?;
        anyhow::ensure!(
            snapshot.incarnation == incarnation && snapshot.attempt_id == attempt_id,
            "lifecycle snapshot identity mismatch"
        );
        Ok(Response::Snapshot(snapshot))
    }

    pub(crate) async fn start(
        &self,
        primary: &Endpoint,
        cancel: CancellationToken,
    ) -> anyhow::Result<StartedEndpoint> {
        primary
            .component()
            .endpoint(lifecycle::endpoint_name(&primary.id().name))
            .endpoint_builder()
            .handler(Ingress::for_engine(Arc::new(LifecycleEndpoint {
                client: self.clone(),
                cancel,
            }))?)
            .start_with_registration()
            .await
    }
}

struct LifecycleEndpoint {
    client: LifecycleClient,
    cancel: CancellationToken,
}

#[async_trait]
impl AsyncEngine<SingleIn<Request>, ManyOut<Annotated<Response>>, anyhow::Error>
    for LifecycleEndpoint
{
    async fn generate(
        &self,
        input: SingleIn<Request>,
    ) -> anyhow::Result<ManyOut<Annotated<Response>>> {
        let (request, context) = input.into_parts();
        let context = context.context();
        let response = tokio::select! {
            biased;
            _ = self.cancel.cancelled() => anyhow::bail!("lifecycle endpoint shutting down"),
            _ = context.stopped() => anyhow::bail!("lifecycle query cancelled"),
            _ = context.killed() => anyhow::bail!("lifecycle query killed"),
            response = tokio::time::timeout(CONTROL_TIMEOUT, self.client.request(request)) => response??,
        };
        Ok(ResponseStream::new(
            Box::pin(stream::once(async move { Annotated::from_data(response) })),
            context,
        ))
    }
}

async fn read_body(response: reqwest::Response) -> anyhow::Result<Bytes> {
    let mut stream = response.bytes_stream();
    let mut body = BytesMut::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        anyhow::ensure!(
            chunk.len() <= MAX_CONTROL_BYTES.saturating_sub(body.len()),
            "SGLang lifecycle control response too large"
        );
        body.extend_from_slice(&chunk);
    }
    Ok(body.freeze())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Json, Router,
        extract::{Path, Query},
        routing::{get, post},
    };
    use dynamo_sidecar_common::HttpEndpoint;
    use std::{
        collections::HashMap,
        sync::atomic::{AtomicBool, Ordering},
    };
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn lifecycle_control_preserves_identity_and_never_invents_completion() {
        let restarted = Arc::new(AtomicBool::new(false));
        let server_restart = restarted.clone();
        let app = Router::new()
            .route(
                "/server_info",
                get(move || {
                    let incarnation = if server_restart.load(Ordering::SeqCst) {
                        "2"
                    } else {
                        "1"
                    }
                    .repeat(32);
                    async move {
                        Json(serde_json::json!({"request_lifecycle": {
                            "version": 1, "incarnation": incarnation, "header_overrides": true
                        }}))
                    }
                }),
            )
            .route(
                "/request_lifecycle/{id}",
                get(
                    |Path(id): Path<String>,
                     Query(query): Query<HashMap<String, String>>,
                     headers: header::HeaderMap| async move {
                        assert_eq!(query["after"], "7");
                        assert_eq!(headers[INCARNATION_HEADER], "1".repeat(32));
                        if id == "f".repeat(32) {
                            return (StatusCode::NOT_FOUND, Json(serde_json::Value::Null));
                        }
                        (
                            StatusCode::OK,
                            Json(serde_json::json!({
                                "incarnation": "1".repeat(32), "attempt_id": id,
                                "stage": "null", "version": 8, "sealed": true,
                                "cancel_requested": true, "terminal": false,
                                "children": [{"child_id": "3".repeat(32), "rid": "native",
                                    "kind": "sample", "dp_rank": 2, "dispatched": true,
                                    "prefill_complete": true, "terminal": false}]
                            })),
                        )
                    },
                )
                .merge(post(
                    |headers: header::HeaderMap, Json(body): Json<serde_json::Value>| async move {
                        assert_eq!(headers[INCARNATION_HEADER], "1".repeat(32));
                        if body["action"] == "acknowledge" {
                            StatusCode::NO_CONTENT
                        } else {
                            StatusCode::CONFLICT
                        }
                    },
                )),
            );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let http = NativeHttp {
            transport: HttpTransport::new(
                HttpEndpoint::parse(
                    &format!("http://{}", listener.local_addr().unwrap()),
                    "test",
                )
                .unwrap(),
                Duration::from_secs(2),
            )
            .unwrap(),
        };
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = LifecycleClient::discover(&http).await.unwrap().unwrap();
        let attempt = |id: String, operation| Request::Attempt {
            incarnation: "1".repeat(32),
            attempt_id: id,
            operation,
        };
        let response = client
            .request(attempt("0".repeat(32), Operation::Snapshot { after: 7 }))
            .await
            .unwrap();
        let Response::Snapshot(snapshot) = response else {
            panic!("expected snapshot")
        };
        assert!(snapshot.cancel_requested && snapshot.children[0].prefill_complete);
        assert!(!snapshot.terminal && !snapshot.children[0].terminal);
        assert_eq!(snapshot.children[0].dp_rank, Some(2));
        assert!(matches!(
            client
                .request(attempt("f".repeat(32), Operation::Snapshot { after: 7 }))
                .await
                .unwrap(),
            Response::Rejected { status: 404 }
        ));
        assert!(matches!(
            client
                .request(attempt("0".repeat(32), Operation::Cancel))
                .await
                .unwrap(),
            Response::Rejected { status: 409 }
        ));
        assert!(matches!(
            client
                .request(attempt("0".repeat(32), Operation::Acknowledge))
                .await
                .unwrap(),
            Response::Acknowledged
        ));
        assert!(
            client
                .request(attempt("../generate".into(), Operation::Cancel))
                .await
                .is_err()
        );
        restarted.store(true, Ordering::SeqCst);
        let Response::Descriptor(descriptor) = client.request(Request::Describe).await.unwrap()
        else {
            panic!("expected descriptor")
        };
        assert_eq!(descriptor.incarnation, "2".repeat(32));
        server.abort();
    }
}
