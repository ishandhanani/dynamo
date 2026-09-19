// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native engine sessions have one authoritative owner, independent of frontends.
//! Records follow the selected worker's etcd lease, never the writing frontend's.

use dynamo_runtime::{
    protocols::EndpointId,
    storage::kv::Key,
    transports::etcd::{Client, CompareAndPutOutcome},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone)]
pub(super) struct Directory {
    client: Client,
    prefix: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum Phase {
    Opening,
    Open,
    Closing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct Owner {
    pub version: u32,
    pub endpoint: EndpointId,
    pub worker_id: u64,
    pub dp_rank: u32,
    pub worker_incarnation: String,
    pub session_incarnation: String,
    pub phase: Phase,
}

pub(super) struct StoredOwner {
    pub owner: Owner,
    key: String,
    bytes: Vec<u8>,
}

impl Directory {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            prefix: format!("v1/native_sessions/{}/", Key::from(model).url_safe(),),
        }
    }

    fn key(&self, session_id: &str) -> String {
        format!("{}{:x}", self.prefix, Sha256::digest(session_id.as_bytes()))
    }

    pub async fn get(&self, session_id: &str) -> anyhow::Result<Option<StoredOwner>> {
        self.get_key(self.key(session_id)).await
    }

    async fn get_key(&self, key: String) -> anyhow::Result<Option<StoredOwner>> {
        let values = self.client.kv_get(key.as_str(), None).await?;
        let Some(value) = values.first() else {
            return Ok(None);
        };
        let owner: Owner = serde_json::from_slice(value.value())?;
        anyhow::ensure!(
            owner.version == 1,
            "unsupported native session owner version"
        );
        Ok(Some(StoredOwner {
            owner,
            key,
            bytes: value.value().to_vec(),
        }))
    }

    /// Claim before sending open, including the incarnation chosen for that open.
    /// An ambiguous dispatch keeps this record; it must never be re-routed/replayed.
    pub async fn claim(&self, session_id: &str, owner: Owner) -> anyhow::Result<StoredOwner> {
        anyhow::ensure!(
            owner.version == 1 && owner.phase == Phase::Opening,
            "invalid new session owner"
        );
        let key = self.key(session_id);
        let bytes = serde_json::to_vec(&owner)?;
        anyhow::ensure!(
            self.client
                .kv_create(&key, bytes.clone(), Some(owner.worker_id))
                .await?
                .is_none(),
            "native session already has an owner"
        );
        Ok(StoredOwner { owner, key, bytes })
    }

    pub async fn transition(
        &self,
        previous: StoredOwner,
        phase: Phase,
    ) -> anyhow::Result<StoredOwner> {
        anyhow::ensure!(
            matches!(
                (previous.owner.phase, phase),
                (Phase::Opening, Phase::Open) | (Phase::Open, Phase::Closing)
            ),
            "invalid native session owner transition"
        );
        let mut owner = previous.owner.clone();
        owner.phase = phase;
        let bytes = serde_json::to_vec(&owner)?;
        match self
            .client
            .kv_compare_and_put(
                &previous.key,
                &previous.bytes,
                &bytes,
                Some(owner.worker_id),
            )
            .await?
        {
            CompareAndPutOutcome::Updated => Ok(StoredOwner {
                owner,
                key: previous.key,
                bytes,
            }),
            CompareAndPutOutcome::Conflict => {
                // Another frontend may have completed this same transition.
                // A different open or a close racing generation is never adopted.
                let current = self
                    .get_key(previous.key)
                    .await?
                    .ok_or_else(|| anyhow::anyhow!("native session owner disappeared"))?;
                anyhow::ensure!(current.owner == owner, "native session owner changed");
                Ok(current)
            }
            CompareAndPutOutcome::Missing => anyhow::bail!("native session owner disappeared"),
        }
    }

    /// Only after the engine proves this incarnation is absent, or a failed open
    /// is known to be complete. A delayed response cannot delete a newer owner.
    pub async fn remove(&self, previous: &StoredOwner) -> anyhow::Result<bool> {
        self.client
            .kv_compare_and_delete(&previous.key, &previous.bytes)
            .await
    }
}

#[cfg(all(test, feature = "integration"))]
mod tests {
    use super::*;
    use dynamo_runtime::{Runtime, transports::etcd::ClientOptions};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn native_session_directory_racing_frontends_cannot_steal_or_delete_reopened_owners() {
        let runtime = Runtime::from_current().unwrap();
        let first = Client::new(ClientOptions::default(), runtime.clone())
            .await
            .unwrap();
        let second = Client::new(ClientOptions::default(), runtime.clone())
            .await
            .unwrap();
        let model = uuid::Uuid::new_v4().to_string();
        let a = Directory::new(first.clone(), &model);
        let b = Directory::new(second.clone(), &model);
        let owner = Owner {
            version: 1,
            endpoint: EndpointId::from("test.worker.generate"),
            worker_id: first.lease_id(),
            dp_rank: 1,
            worker_incarnation: uuid::Uuid::new_v4().simple().to_string(),
            session_incarnation: uuid::Uuid::new_v4().simple().to_string(),
            phase: Phase::Opening,
        };
        let mut other = owner.clone();
        other.session_incarnation = uuid::Uuid::new_v4().simple().to_string();
        let (left, right) = tokio::join!(a.claim("s", owner), b.claim("s", other));
        assert_ne!(
            left.is_ok(),
            right.is_ok(),
            "exactly one concurrent open wins"
        );
        let opening = a.get("s").await.unwrap().unwrap();
        let recovered = b.get("s").await.unwrap().unwrap();
        assert_eq!(opening.owner, recovered.owner);
        let stale_opening = a.get("s").await.unwrap().unwrap();
        let (left, right) = tokio::join!(
            a.transition(opening, Phase::Open),
            b.transition(recovered, Phase::Open)
        );
        assert_eq!(left.unwrap().owner, right.unwrap().owner);
        let open = a.get("s").await.unwrap().unwrap();
        let closing = b.transition(open, Phase::Closing).await.unwrap();
        assert!(
            a.transition(stale_opening, Phase::Open).await.is_err(),
            "recovery cannot undo a close"
        );
        let stale_close = a.get("s").await.unwrap().unwrap();
        assert!(b.remove(&closing).await.unwrap());
        let mut new_owner = closing.owner;
        new_owner.phase = Phase::Opening;
        new_owner.session_incarnation = uuid::Uuid::new_v4().simple().to_string();
        let new_owner = a.claim("s", new_owner).await.unwrap();
        assert!(
            !b.remove(&stale_close).await.unwrap(),
            "late cleanup cannot remove a reopened owner"
        );
        assert_eq!(b.get("s").await.unwrap().unwrap().owner, new_owner.owner);
        assert!(
            b.transition(new_owner, Phase::Closing).await.is_err(),
            "an ambiguous open cannot be closed before acceptance is known"
        );
        let current = a.get("s").await.unwrap().unwrap();
        assert!(a.remove(&current).await.unwrap());
        runtime.shutdown();
        tokio::task::spawn_blocking(move || drop((first, second, a, b)))
            .await
            .unwrap();
    }
}
