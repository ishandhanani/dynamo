// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replicated session bindings: the one wire schema every host publishes and
//! applies, and the one applier. Transports are the hosts' (the selection
//! service's ZMQ peer mesh, the frontend's runtime event plane); each carries
//! [`AffinityBindingEvent`] and feeds received events to
//! [`SessionAffinity::apply_replica_event`].

use std::sync::Arc;

use tokio::sync::mpsc;

use super::{
    AffinityReplicaSink, AffinityTarget, AffinityVersion, ReplicaApplyOutcome, SessionAffinity,
};
use crate::identity::RoutingPartitionId;
pub use crate::services::common::replica_sync::AffinityBindingEvent;

impl AffinityBindingEvent {
    pub fn new(
        partition: &RoutingPartitionId,
        session_id: &str,
        target: AffinityTarget,
        version: AffinityVersion,
    ) -> Self {
        Self {
            partition: partition.clone(),
            session_id: session_id.to_string(),
            worker_id: target.worker_id,
            dp_rank: target.dp_rank,
            sequence: version.sequence,
            writer_id: version.writer_id,
        }
    }

    /// Queue a best-effort update, dropping it if the transport is full or closed.
    pub fn enqueue(self, tx: &mpsc::Sender<Self>) {
        if let Err(error) = tx.try_send(self) {
            tracing::trace!(%error, "dropping best-effort session affinity update");
        }
    }

    pub fn target(&self) -> AffinityTarget {
        AffinityTarget::new(self.worker_id, self.dp_rank)
    }

    pub fn version(&self) -> AffinityVersion {
        AffinityVersion {
            sequence: self.sequence,
            writer_id: self.writer_id,
        }
    }
}

/// Queue this partition's bindings for its transport, dropping updates when
/// the queue is full or closed. Replication is best effort.
pub fn replica_sink(
    partition: RoutingPartitionId,
    tx: mpsc::Sender<AffinityBindingEvent>,
) -> Arc<dyn AffinityReplicaSink> {
    Arc::new(move |session_id: &str, target, version| {
        AffinityBindingEvent::new(&partition, session_id, target, version).enqueue(&tx);
    })
}

impl SessionAffinity {
    /// Apply a binding another replica published, for the table serving
    /// `partition`. Every host's transport ends here and must exclude its own
    /// publications. The binding's writer may differ from the event sender:
    /// a replica refreshing an adopted binding retains the original version.
    pub fn apply_replica_event(
        &self,
        partition: &RoutingPartitionId,
        event: AffinityBindingEvent,
        is_live: impl Fn(AffinityTarget) -> bool,
    ) -> Option<ReplicaApplyOutcome> {
        if event.partition != *partition {
            return None;
        }
        self.observe_replica_sequence(event.sequence);
        let (target, version) = (event.target(), event.version());
        if !is_live(target) {
            tracing::trace!(
                key = %event.partition,
                worker_id = event.worker_id,
                "Dropping session affinity replica update: worker not schedulable here"
            );
            return None;
        }
        let outcome = self.apply_replica_update(event.session_id, target, version);
        tracing::trace!(
            worker_id = event.worker_id,
            ?outcome,
            "Applied session affinity replica update"
        );
        Some(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::super::SessionAffinityConfig;
    use super::*;
    use std::time::Duration;

    #[tokio::test(start_paused = true)]
    async fn peer_refresh_keeps_the_original_writers_binding_alive() {
        let partition = RoutingPartitionId::new("model", "group");
        let table = |writer| {
            let table =
                SessionAffinity::with_config(SessionAffinityConfig::new(Duration::from_secs(60)))
                    .unwrap();
            let (tx, rx) = mpsc::channel(4);
            assert!(table.enable_replication(writer, replica_sink(partition.clone(), tx)));
            (table, rx)
        };
        let (original, mut original_events) = table(100);
        let (peer, mut peer_events) = table(200);
        let target = AffinityTarget::new(1, Some(0));
        drop(
            original
                .commit(original.acquire("s", None).await.unwrap(), target)
                .unwrap(),
        );
        for _ in 0..2 {
            let event = original_events.try_recv().unwrap();
            assert_eq!(event.writer_id, 100);
            assert!(
                peer.apply_replica_event(&partition, event, |_| true)
                    .is_some()
            );
        }
        assert_eq!(peer.query_target("s", None).unwrap(), Some(target));
        tokio::time::advance(Duration::from_secs(30)).await;
        drop(
            peer.commit(peer.acquire("s", None).await.unwrap(), target)
                .unwrap(),
        );
        for _ in 0..2 {
            let event = peer_events.try_recv().unwrap();
            assert_eq!(
                event.writer_id, 100,
                "peer refresh preserves the original version"
            );
            assert_eq!(
                original.apply_replica_event(&partition, event, |_| true),
                Some(ReplicaApplyOutcome::Refreshed)
            );
        }
        assert!(original_events.is_empty(), "applying must not republish");
        assert!(peer_events.is_empty());
        for (elapsed, expected) in [(31, Some(target)), (30, None)] {
            tokio::time::advance(Duration::from_secs(elapsed)).await;
            for table in [&original, &peer] {
                assert_eq!(table.query_target("s", None).unwrap(), expected);
            }
        }
    }
}
