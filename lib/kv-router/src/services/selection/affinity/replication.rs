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
    /// `partition`. Every host's transport ends here.
    pub fn apply_replica_event(
        &self,
        partition: &RoutingPartitionId,
        event: AffinityBindingEvent,
        is_live: impl Fn(AffinityTarget) -> bool,
    ) -> Option<ReplicaApplyOutcome> {
        if event.writer_id == self.writer_id() {
            return None;
        }
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
    use std::time::Duration;

    use super::super::{AcquireStep, Hold, SessionAffinityConfig};
    use super::*;

    fn partition_id(name: &str) -> RoutingPartitionId {
        RoutingPartitionId::new(name, "default")
    }

    fn table() -> SessionAffinity {
        SessionAffinity::with_config(SessionAffinityConfig::new(Duration::from_secs(60)))
            .expect("table")
    }

    /// Install a sink that records every event the table publishes.
    fn record_events(
        table: &SessionAffinity,
        writer_id: u64,
        partition: &RoutingPartitionId,
    ) -> mpsc::Receiver<AffinityBindingEvent> {
        let (tx, rx) = mpsc::channel(16);
        assert!(table.enable_replication(writer_id, replica_sink(partition.clone(), tx)));
        rx
    }

    fn event(
        partition: &RoutingPartitionId,
        writer_id: u64,
        sequence: u64,
    ) -> AffinityBindingEvent {
        AffinityBindingEvent {
            partition: partition.clone(),
            session_id: "s".to_string(),
            worker_id: 1,
            dp_rank: Some(0),
            sequence,
            writer_id,
        }
    }

    #[test]
    fn schema_round_trips_with_the_partition_flattened() {
        let event = event(&partition_id("model"), 7, 42);
        let json = serde_json::to_value(&event).expect("encode");
        assert_eq!(json["model_name"], "model");
        assert_eq!(json["routing_group"], "default");
        assert_eq!(json["session_id"], "s");
        assert_eq!(json["writer_id"], 7);
        let decoded: AffinityBindingEvent = serde_json::from_value(json).expect("decode");
        assert_eq!(decoded, event);
        let bytes = rmp_serde::to_vec_named(&event).expect("msgpack encode");
        let decoded: AffinityBindingEvent = rmp_serde::from_slice(&bytes).expect("msgpack decode");
        assert_eq!(decoded, event);
    }

    #[tokio::test]
    async fn applier_scopes_by_partition_writer_and_liveness() {
        let partition = partition_id("model");
        let table = table();
        let _events = record_events(&table, 7, &partition);

        assert_eq!(
            table.apply_replica_event(&partition, event(&partition, 7, 1), |_| {
                panic!("own events must be rejected before liveness checking")
            }),
            None
        );
        assert_eq!(
            table.apply_replica_event(&partition, event(&partition_id("other"), 9, 2), |_| {
                panic!("other partitions must be rejected before liveness checking")
            }),
            None
        );
        let sequence = table.next_version().sequence + 10;
        let mut unknown = event(&partition, 9, sequence);
        unknown.worker_id = 5;
        assert_eq!(
            table.apply_replica_event(&partition, unknown, |target| target.worker_id == 1),
            None
        );
        assert_eq!(
            table.next_version().sequence,
            sequence + 1,
            "an unapplied event still advances the replica clock"
        );
        assert_eq!(table.query_target("s", None).expect("query"), None);
        let binding = event(&partition, 9, sequence + 2);
        assert_eq!(
            table.apply_replica_event(&partition, binding.clone(), |target| target.worker_id == 1),
            Some(ReplicaApplyOutcome::Inserted)
        );
        assert_eq!(
            table.apply_replica_event(&partition, binding, |_| true),
            Some(ReplicaApplyOutcome::Refreshed)
        );
        assert_eq!(
            table.query_target("s", None).expect("query"),
            Some(AffinityTarget::new(1, Some(0)))
        );
    }

    /// Publishing and applying a binding uses the same schema in both directions.
    #[tokio::test]
    async fn replicas_converge_through_the_shared_schema() {
        let partition = partition_id("model");
        let frontend = table();
        let service = table();
        let mut frontend_events = record_events(&frontend, 100, &partition);
        let mut service_events = record_events(&service, 200, &partition);

        // The frontend binds; the service applies what it published.
        bind(&frontend, AffinityTarget::new(1, Some(0)));
        assert!(!frontend_events.is_empty());
        while let Ok(event) = frontend_events.try_recv() {
            assert_eq!(event.partition, partition);
            assert_eq!(event.writer_id, 100);
            assert!(
                service
                    .apply_replica_event(&partition, event, |_| true)
                    .is_some()
            );
        }
        assert_eq!(
            service.query_target("s", None).expect("query"),
            Some(AffinityTarget::new(1, Some(0)))
        );

        // The service re-binds after the worker departs; the frontend follows.
        let AcquireStep::Held(hold) = service.try_acquire("s", None).expect("hold binding") else {
            panic!("expected a bound session");
        };
        hold.invalidate();
        bind(&service, AffinityTarget::new(2, Some(0)));
        assert!(!service_events.is_empty());
        while let Ok(event) = service_events.try_recv() {
            assert_eq!(event.writer_id, 200);
            frontend.apply_replica_event(&partition, event, |_| true);
        }
        assert_eq!(
            frontend.query_target("s", None).expect("query"),
            Some(AffinityTarget::new(2, Some(0))),
            "the frontend converges on the service's newer binding"
        );
    }

    fn bind(table: &SessionAffinity, target: AffinityTarget) {
        let AcquireStep::Held(Hold::Initialize(initialization)) =
            table.try_acquire("s", None).expect("acquire")
        else {
            panic!("expected a new session");
        };
        drop(initialization.commit(target).expect("commit"));
    }

    #[test]
    fn replica_sink_drops_updates_when_full_or_closed() {
        let partition = partition_id("model");
        let (tx, mut rx) = mpsc::channel(1);
        let sink = replica_sink(partition.clone(), tx);
        let target = AffinityTarget::new(1, Some(0));
        let version = AffinityVersion {
            sequence: 1,
            writer_id: 7,
        };
        sink.publish("first", target, version);
        sink.publish("second", target, version);
        assert_eq!(
            rx.try_recv().expect("first update"),
            AffinityBindingEvent::new(&partition, "first", target, version)
        );
        assert!(rx.try_recv().is_err());
        drop(rx);
        sink.publish("closed", target, version);
    }
}
