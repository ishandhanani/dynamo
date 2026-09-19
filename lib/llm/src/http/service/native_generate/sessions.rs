// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine sessions have hard ownership. Only the dedicated routing-control
//! response is decoded here; public open/close/generation bytes remain opaque.

use dynamo_runtime::protocols::EndpointId;
use serde::Deserialize;

use super::{
    session_directory::{Directory, Owner, Phase, StoredOwner},
    *,
};

const SESSION_ID: &str = "x-sglang-session-id";
const SESSION_INCARNATION: &str = "x-sglang-session-incarnation";

#[derive(Deserialize)]
struct Snapshot {
    dp_rank: u32,
    session_incarnation: Option<String>,
    input_ids: Option<Vec<u32>>,
    #[serde(default)]
    engine_processed_input: bool,
    error: Option<String>,
}

/// This small projection also works for control bodies containing unknown
/// engine extensions which serde_json::Value cannot represent (e.g. 1e400).
pub(crate) fn request_session_id(
    body: &[u8],
    operation: http::Operation,
) -> anyhow::Result<Option<String>> {
    let fields: BTreeMap<String, &RawValue> = serde_json::from_slice(body)?;
    let value = if operation == http::Operation::Generate {
        let Some(params) = fields.get("session_params").filter(|v| v.get() != "null") else {
            return Ok(None);
        };
        let params: BTreeMap<String, &RawValue> = serde_json::from_str(params.get())?;
        params
            .get("id")
            .map(|raw| serde_json::from_str::<Option<String>>(raw.get()))
            .transpose()?
            .flatten()
    } else {
        fields
            .get("session_id")
            .map(|raw| serde_json::from_str::<Option<String>>(raw.get()))
            .transpose()?
            .flatten()
    };
    anyhow::ensure!(
        value.as_ref().is_none_or(|id| !id.is_empty()),
        "session ID must not be empty"
    );
    if operation != http::Operation::OpenSession {
        anyhow::ensure!(value.is_some(), "session request requires an ID");
    }
    Ok(value)
}

impl NativeGenerateBinding {
    fn session_directory(&self) -> anyhow::Result<&Directory> {
        anyhow::ensure!(
            self.prefill.is_none(),
            "native P/D sessions require engine history synchronization"
        );
        self.sessions
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("native session ownership requires etcd discovery"))
    }

    pub(crate) async fn session_endpoint(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Option<EndpointId>> {
        let owner = self
            .sessions
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("native session ownership requires etcd discovery"))?
            .get(session_id)
            .await?;
        Ok(owner.map(|stored| stored.owner.endpoint))
    }

    async fn describe_session_owner(&self, owner: &Owner) -> anyhow::Result<Descriptor> {
        anyhow::ensure!(
            owner.endpoint == self.endpoint
                && !self.cancellation.is_cancelled()
                && self.admitted_ids.borrow().contains(&owner.worker_id),
            "native session owner is not admitted"
        );
        let descriptor = NativeAttempt::describe(&self.control, owner.worker_id).await?;
        anyhow::ensure!(
            descriptor.supports_sessions(),
            "selected engine does not support native session control v1"
        );
        Ok(descriptor)
    }

    async fn check_session_owner(&self, owner: &Owner) -> anyhow::Result<()> {
        let descriptor = self.describe_session_owner(owner).await?;
        anyhow::ensure!(
            descriptor.incarnation == owner.worker_incarnation,
            "native session worker restarted"
        );
        Ok(())
    }

    async fn reclaim_absent_session(&self, id: &str) -> anyhow::Result<()> {
        let directory = self.session_directory()?;
        let Some(stored) = directory.get(id).await? else {
            return Ok(());
        };
        let descriptor = self.describe_session_owner(&stored.owner).await?;
        if descriptor.incarnation == stored.owner.worker_incarnation {
            anyhow::ensure!(
                stored.owner.phase != Phase::Opening,
                "native session open outcome is still unknown"
            );
            let snapshot = self
                .session_snapshot(&stored.owner, &serde_json::json!({"id": id}), None)
                .await?;
            anyhow::ensure!(
                snapshot.session_incarnation.as_deref() != Some(&stored.owner.session_incarnation),
                NativeRequestError(anyhow::anyhow!("native session is already open"))
            );
        }
        // A changed worker incarnation fences even a delayed Opening. In the
        // same incarnation, only a known completed open/close can be reclaimed.
        anyhow::ensure!(
            directory.remove(&stored).await?,
            "native session owner changed during cleanup"
        );
        Ok(())
    }

    async fn session_snapshot(
        &self,
        owner: &Owner,
        params: &Value,
        suffix: Option<&[u32]>,
    ) -> anyhow::Result<Snapshot> {
        let request = http::Request {
            method: "POST".into(),
            operation: http::Operation::SessionRouting,
            headers: vec![
                (
                    "content-type".into(),
                    Bytes::from_static(b"application/json"),
                ),
                (
                    http::lifecycle::INCARNATION_HEADER.into(),
                    owner.worker_incarnation.clone().into(),
                ),
            ],
            body: serde_json::to_vec(&serde_json::json!({
                "session_params": params, "dp_rank": owner.dp_rank, "input_ids": suffix,
            }))?
            .into(),
        };
        let response = tokio::time::timeout(Duration::from_secs(12), async {
            let response =
                super::super::forward(&self.client, Context::new(request), owner.worker_id).await?;
            let status = response.status();
            anyhow::ensure!(
                matches!(status.as_u16(), 200 | 404 | 409),
                "native session control returned {status}"
            );
            let body = axum::body::to_bytes(response.into_body(), 32 * 1024 * 1024).await?;
            let snapshot: Snapshot = serde_json::from_slice(&body)?;
            anyhow::ensure!(
                snapshot.dp_rank == owner.dp_rank,
                "native session control returned another rank"
            );
            anyhow::Ok(snapshot)
        })
        .await??;
        Ok(response)
    }

    async fn open_owner(&self, session_id: &str) -> anyhow::Result<StoredOwner> {
        let directory = self.session_directory()?;
        let owner = directory
            .get(session_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("native session has no live owner"))?;
        anyhow::ensure!(
            owner.owner.phase != Phase::Closing,
            "native session is closing"
        );
        self.check_session_owner(&owner.owner).await?;
        let snapshot = self
            .session_snapshot(&owner.owner, &serde_json::json!({"id":session_id}), None)
            .await?;
        if owner.owner.phase == Phase::Open
            && snapshot.session_incarnation.as_deref() != Some(&owner.owner.session_incarnation)
        {
            // Generation still fails; this only removes confirmed stale state
            // so an explicit later open can reuse the ID.
            directory.remove(&owner).await?;
        }
        anyhow::ensure!(
            snapshot.session_incarnation.as_deref() == Some(&owner.owner.session_incarnation),
            "native session incarnation is absent or changed"
        );
        if owner.owner.phase == Phase::Opening {
            // A different frontend can recover an accepted open without reading
            // its response. An absent Opening is ambiguous and is never replayed.
            directory.transition(owner, Phase::Open).await
        } else {
            Ok(owner)
        }
    }

    pub(super) async fn resolve_session(
        &self,
        projection: &Projection,
        children: &mut [Child],
        rank: Option<u32>,
    ) -> anyhow::Result<Option<Owner>> {
        let Some(params) = projection
            .value
            .get("session_params")
            .filter(|v| !v.is_null())
        else {
            return Ok(None);
        };
        let id = params
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("session request requires an ID"))?;
        let stored = self.open_owner(id).await?;
        let owner = stored.owner;
        anyhow::ensure!(
            rank.is_none_or(|rank| rank == owner.dp_rank),
            "requested rank conflicts with native session owner"
        );
        anyhow::ensure!(
            params
                .get("incarnation")
                .filter(|v| !v.is_null())
                .is_none_or(|v| v.as_str() == Some(&owner.session_incarnation)),
            "requested incarnation conflicts with native session owner"
        );
        if self.host.kv_router_if_enabled().is_some() {
            let mut prefixes = std::collections::HashMap::new();
            for child in children {
                if let Some(tokens) = prefixes.get(child.tokens.as_ref()) {
                    child.tokens = Arc::clone(tokens);
                    continue;
                }
                let snapshot = self
                    .session_snapshot(&owner, params, Some(child.tokens.as_slice()))
                    .await?;
                anyhow::ensure!(
                    snapshot.session_incarnation.as_deref() == Some(&owner.session_incarnation),
                    "native session incarnation changed"
                );
                anyhow::ensure!(
                    snapshot.error.is_none(),
                    "native session history is unavailable: {}",
                    snapshot.error.unwrap_or_default()
                );
                anyhow::ensure!(
                    !snapshot.engine_processed_input,
                    "native session KV routing requires engine-compatible multimodal history metadata"
                );
                let tokens = snapshot
                    .input_ids
                    .filter(|ids| !ids.is_empty())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "native KV routing requires nonempty effective prompt tokens"
                        )
                    })?;
                let tokens = Arc::new(tokens);
                prefixes.insert(child.tokens.as_ref().clone(), tokens.clone());
                child.tokens = tokens;
            }
        }
        Ok(Some(owner))
    }

    /// Run once, independent of the client connection. Losing the response to an
    /// open does not authorize another open or releasing its ownership claim.
    pub(crate) async fn session_control(
        self: Arc<Self>,
        operation: http::Operation,
        method: Method,
        headers: HeaderMap,
        body: Bytes,
    ) -> anyhow::Result<Response> {
        super::check_owned_headers(&headers).map_err(NativeRequestError)?;
        let id = request_session_id(&body, operation).map_err(NativeRequestError)?;
        self.session_directory()?;
        let (send, receive) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let result = tokio::time::timeout(
                Duration::from_secs(30),
                self.run_session_control(operation, method, headers, body, id),
            )
            .await
            .map_err(anyhow::Error::from)
            .and_then(|result| result);
            let _ = send.send(result);
        });
        receive.await?
    }

    async fn run_session_control(
        &self,
        operation: http::Operation,
        method: Method,
        mut headers: HeaderMap,
        body: Bytes,
        id: Option<String>,
    ) -> anyhow::Result<Response> {
        let directory = self.session_directory()?;
        let supplied_id = id.is_some();
        let id = id.unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
        let stored = if operation == http::Operation::OpenSession {
            self.reclaim_absent_session(&id).await?;
            let workers = self.admitted_ids.borrow().clone();
            anyhow::ensure!(
                !workers.is_empty() && !self.cancellation.is_cancelled(),
                "native WorkerSet is unavailable"
            );
            // Open has no prompt. Spread new ownership uniformly; generation
            // still uses the configured admission policy at this exact target.
            let worker_id = workers[rand::random_range(0..workers.len())];
            let config = &self.card.runtime_config;
            let rank = headers
                .get("x-data-parallel-rank")
                .map(|value| value.to_str()?.parse::<u32>().map_err(anyhow::Error::from))
                .transpose()?;
            let ranks = config.data_parallel_start_rank
                ..config
                    .data_parallel_start_rank
                    .saturating_add(config.data_parallel_size);
            anyhow::ensure!(
                !ranks.is_empty() && rank.is_none_or(|rank| ranks.contains(&rank)),
                "invalid session DP rank"
            );
            let descriptor = NativeAttempt::describe(&self.control, worker_id).await?;
            anyhow::ensure!(
                descriptor.supports_sessions(),
                "selected engine does not support native session control v1"
            );
            directory
                .claim(
                    &id,
                    Owner {
                        version: 1,
                        endpoint: self.endpoint.clone(),
                        worker_id,
                        dp_rank: rank.unwrap_or_else(|| rand::random_range(ranks)),
                        worker_incarnation: descriptor.incarnation,
                        session_incarnation: uuid::Uuid::new_v4().simple().to_string(),
                        phase: Phase::Opening,
                    },
                )
                .await?
        } else {
            anyhow::ensure!(
                operation == http::Operation::CloseSession,
                "invalid public session operation"
            );
            let stored = directory
                .get(&id)
                .await?
                .ok_or_else(|| anyhow::anyhow!("native session has no live owner"))?;
            let fields: BTreeMap<String, &RawValue> = serde_json::from_slice(&body)?;
            let requested_incarnation = fields
                .get("session_incarnation")
                .map(|value| serde_json::from_str::<Option<String>>(value.get()))
                .transpose()?
                .flatten();
            anyhow::ensure!(
                requested_incarnation
                    .as_ref()
                    .is_none_or(|value| value == &stored.owner.session_incarnation),
                NativeRequestError(anyhow::anyhow!(
                    "requested incarnation conflicts with native session owner"
                ))
            );
            self.check_session_owner(&stored.owner).await?;
            let stored = if stored.owner.phase == Phase::Opening {
                self.open_owner(&id).await?
            } else {
                stored
            };
            if stored.owner.phase == Phase::Open {
                directory.transition(stored, Phase::Closing).await?
            } else {
                stored
            }
        };
        let owner = &stored.owner;
        headers.insert(
            http::lifecycle::INCARNATION_HEADER,
            owner.worker_incarnation.parse()?,
        );
        headers.insert(SESSION_INCARNATION, owner.session_incarnation.parse()?);
        if operation == http::Operation::OpenSession && !supplied_id {
            headers.insert(SESSION_ID, id.parse()?);
        }
        let response = super::super::forward(
            &self.client,
            Context::new(http::Request {
                method: method.to_string(),
                headers: http::encode_headers(headers),
                body,
                operation,
            }),
            owner.worker_id,
        )
        .await?;
        let success = response.status().is_success();
        if operation == http::Operation::OpenSession {
            let snapshot = self
                .session_snapshot(owner, &serde_json::json!({"id":id}), None)
                .await?;
            if snapshot.session_incarnation.as_deref() == Some(&owner.session_incarnation) {
                directory.transition(stored, Phase::Open).await?;
            } else if !success {
                // The engine handler has returned a rejection. Its subsequent
                // scheduler query also proves this open did not install our
                // incarnation. Transport failures never reach this cleanup.
                directory.remove(&stored).await?;
            } else {
                anyhow::bail!("engine acknowledged open without the claimed session incarnation");
            }
        } else if success {
            // Native close acknowledges dispatch, not completion. Keep ownership
            // until the scheduler confirms this incarnation is gone.
            loop {
                let snapshot = self
                    .session_snapshot(owner, &serde_json::json!({"id":id}), None)
                    .await?;
                if snapshot.session_incarnation.as_deref() != Some(&owner.session_incarnation) {
                    directory.remove(&stored).await?;
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
        Ok(response)
    }
}
