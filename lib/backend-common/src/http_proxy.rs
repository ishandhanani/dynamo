// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use crate::http::{self, MAX_BODY_CHUNK, Request, ResponseFrame};
use async_trait::async_trait;
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

pub struct HttpProxy {
    client: reqwest::Client,
    endpoint: reqwest::Url,
    cancel: CancellationToken,
    paths: &'static [&'static str],
}

impl HttpProxy {
    pub async fn start(
        primary: &Endpoint,
        endpoint: reqwest::Url,
        connect_timeout: Duration,
        paths: &'static [&'static str],
        cancel: CancellationToken,
    ) -> anyhow::Result<StartedEndpoint> {
        let handler = Arc::new(Self {
            client: reqwest::Client::builder()
                .connect_timeout(connect_timeout)
                .redirect(reqwest::redirect::Policy::none())
                .retry(reqwest::retry::never())
                .no_gzip()
                .no_brotli()
                .no_deflate()
                .no_zstd()
                .build()?,
            endpoint,
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
        let (path, query) = request
            .path
            .split_once('?')
            .map_or((request.path.as_str(), None), |(p, q)| (p, Some(q)));
        anyhow::ensure!(
            self.paths.contains(&path),
            "HTTP path is not enabled by this sidecar"
        );
        let mut headers = http::decode_headers(request.headers)?;
        // The HTTP client computes framing from the retained request bytes.
        headers.remove("content-length");
        let cancel = self.cancel.clone();
        let stream_context = context.clone();
        let cancelled = async move {
            tokio::select! {
                _ = cancel.cancelled() => {},
                _ = stream_context.stopped() => {},
                _ = stream_context.killed() => {},
            }
        };
        let client = self.client.clone();
        let mut url = self.endpoint.clone();
        url.set_path(path);
        url.set_query(query);
        let stream = async_stream::try_stream! {
            let response = client.request(method, url).headers(headers).body(request.body).send().await?;
            yield ResponseFrame::Head {
                status: response.status().as_u16(),
                headers: http::encode_headers(response.headers().clone()),
            };
            let mut upstream = response.bytes_stream();
            while let Some(chunk) = upstream.next().await {
                let mut bytes = chunk?;
                while !bytes.is_empty() {
                    yield ResponseFrame::Body(bytes.split_to(bytes.len().min(MAX_BODY_CHUNK)));
                }
            }
            yield ResponseFrame::End;
        };
        let stream = stream
            .take_until(cancelled)
            .map(|result: anyhow::Result<ResponseFrame>| match result {
                Ok(frame) => Annotated::from_data(frame),
                Err(error) => Annotated::from_error(error.to_string()),
            });
        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}
