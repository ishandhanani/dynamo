// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker catalog agent: register a worker with a Dynamo selection service over
//! HTTP, keep its lease alive, and deregister on shutdown.
//!
//! This is the runtime-free registration path (no etcd, no NATS). The
//! selection service's catalog record schema is the contract; this module
//! builds the `POST /workers` body from engine metadata and drives the
//! `POST /workers/{id}/heartbeat` / `DELETE /workers/{id}` lifecycle.

use std::collections::HashMap;
use std::time::Duration;

use dynamo_backend_common::DynamoError;
use serde::Serialize;
use tokio_util::sync::CancellationToken;

use crate::HttpEndpoint;

/// Registration body for the selection service (`POST /workers`).
#[derive(Debug, Clone, Serialize)]
pub struct CatalogRegistration {
    pub worker_id: u64,
    pub model_name: String,
    pub routing_group: String,
    /// Address a frontend dispatches to for this worker.
    pub endpoint: String,
    pub block_size: u32,
    pub data_parallel_start_rank: u32,
    pub data_parallel_size: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_num_batched_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_kv_blocks: Option<u64>,
    /// Engine KV-event ZMQ PUB endpoints by global DP rank, reachable from the
    /// selection service.
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub kv_events_endpoints: HashMap<u32, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay_endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router_hint_worker_type: Option<String>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub router_hint_source_control_endpoints: HashMap<u32, String>,
    /// Lease duration; the agent heartbeats at half this interval.
    pub ttl_secs: f64,
}

/// Configuration for the catalog agent.
#[derive(Debug, Clone)]
pub struct CatalogAgentConfig {
    pub catalog_url: HttpEndpoint,
    pub ttl: Duration,
}

impl CatalogAgentConfig {
    pub fn new(catalog_url: HttpEndpoint, ttl: Duration) -> Self {
        Self { catalog_url, ttl }
    }
}

/// HTTP client for one selection-service catalog.
#[derive(Debug, Clone)]
pub struct CatalogClient {
    base: HttpEndpoint,
    client: reqwest::Client,
}

fn catalog_error(message: impl Into<String>) -> DynamoError {
    crate::cannot_connect(message.into())
}

impl CatalogClient {
    pub fn new(base: HttpEndpoint) -> Result<Self, DynamoError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .map_err(|error| catalog_error(format!("catalog HTTP client: {error}")))?;
        Ok(Self { base, client })
    }

    pub fn base(&self) -> &HttpEndpoint {
        &self.base
    }

    /// `POST /workers`: create or replace the registration.
    pub async fn register(&self, registration: &CatalogRegistration) -> Result<(), DynamoError> {
        let url = self.base.with_path("/workers");
        let response = self
            .client
            .post(url.clone())
            .json(registration)
            .send()
            .await
            .map_err(|error| catalog_error(format!("register worker at {url}: {error}")))?;
        expect_success("register worker", response).await
    }

    /// `POST /workers/{id}/heartbeat`: renew the lease.
    pub async fn heartbeat(&self, worker_id: u64) -> Result<(), DynamoError> {
        let url = self
            .base
            .with_path(&format!("/workers/{worker_id}/heartbeat"));
        let response = self
            .client
            .post(url.clone())
            .json(&serde_json::json!({}))
            .send()
            .await
            .map_err(|error| catalog_error(format!("heartbeat at {url}: {error}")))?;
        expect_success("heartbeat", response).await
    }

    /// `DELETE /workers/{id}`: drain and deregister.
    pub async fn deregister(&self, worker_id: u64) -> Result<(), DynamoError> {
        let url = self.base.with_path(&format!("/workers/{worker_id}"));
        let response = self
            .client
            .delete(url.clone())
            .send()
            .await
            .map_err(|error| catalog_error(format!("deregister at {url}: {error}")))?;
        expect_success("deregister", response).await
    }
}

async fn expect_success(what: &str, response: reqwest::Response) -> Result<(), DynamoError> {
    let status = response.status();
    if status.is_success() {
        return Ok(());
    }
    let body = response.text().await.unwrap_or_default();
    Err(catalog_error(format!(
        "{what} failed with {status}: {}",
        body.trim()
    )))
}

/// A registered worker's lease keeper. Heartbeats at half the TTL until
/// cancelled, then deregisters. A heartbeat rejected with a not-found status
/// (the catalog restarted or expired the lease) re-registers.
pub struct CatalogAgent {
    client: CatalogClient,
    registration: CatalogRegistration,
}

impl CatalogAgent {
    /// Register `registration` with the catalog at `config.catalog_url`.
    pub async fn register(
        config: &CatalogAgentConfig,
        mut registration: CatalogRegistration,
    ) -> Result<Self, DynamoError> {
        registration.ttl_secs = config.ttl.as_secs_f64();
        let client = CatalogClient::new(config.catalog_url.clone())?;
        client.register(&registration).await?;
        tracing::info!(
            catalog = %config.catalog_url.as_str(),
            worker_id = registration.worker_id,
            model = %registration.model_name,
            routing_group = %registration.routing_group,
            ttl_secs = registration.ttl_secs,
            "registered worker with selection catalog"
        );
        Ok(Self {
            client,
            registration,
        })
    }

    pub fn registration(&self) -> &CatalogRegistration {
        &self.registration
    }

    /// Run the heartbeat loop until `cancel` fires, then deregister.
    pub async fn run(self, cancel: CancellationToken) {
        let interval = Duration::from_secs_f64((self.registration.ttl_secs / 2.0).max(0.05));
        let worker_id = self.registration.worker_id;
        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = tokio::time::sleep(interval) => {}
            }
            match self.client.heartbeat(worker_id).await {
                Ok(()) => {}
                Err(error) if error.to_string().contains("404") => {
                    tracing::warn!(worker_id, %error, "catalog lost this worker; re-registering");
                    if let Err(error) = self.client.register(&self.registration).await {
                        tracing::warn!(worker_id, %error, "re-registration failed; retrying next heartbeat");
                    }
                }
                Err(error) => {
                    tracing::warn!(worker_id, %error, "catalog heartbeat failed; retrying next heartbeat");
                }
            }
        }
        if let Err(error) = self.client.deregister(worker_id).await {
            tracing::warn!(worker_id, %error, "catalog deregistration failed");
        } else {
            tracing::info!(worker_id, "deregistered worker from selection catalog");
        }
    }
}
