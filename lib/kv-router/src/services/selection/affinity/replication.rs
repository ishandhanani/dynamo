// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replicated session bindings: the one wire schema every host publishes and
//! applies, and the one applier. Transports are the hosts' (the selection
//! service's ZMQ peer mesh, the frontend's runtime event plane); each carries
//! [`AffinityBindingEvent`] and feeds received events to
//! [`AffinityResolver::apply_replica_event`].

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::resolver::AffinityResolver;
use super::table::{AffinityReplicaSink, AffinityTarget, AffinityVersion, ReplicaApplyOutcome};
use crate::identity::RoutingPartitionId;

/// One replicated session binding. The partition scopes the session id: a
/// binding for another partition is ignored on receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AffinityBindingEvent {
    #[serde(flatten)]
    pub partition: RoutingPartitionId,
    pub session_id: String,
    pub worker_id: u64,
    pub dp_rank: Option<u32>,
    pub sequence: u64,
    /// The publishing replica's writer id: its discovery instance id where
    /// the host has one (frontends), otherwise a random non-zero process id
    /// (the standalone service). The applier ignores its own writer id.
    pub writer_id: u64,
}

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

/// A table sink that turns each published binding into an
/// [`AffinityBindingEvent`] for `partition` and hands it to the transport.
/// Best effort: the transport drops what it cannot queue.
pub struct AffinityEventSink {
    partition: RoutingPartitionId,
    publish: Arc<dyn Fn(AffinityBindingEvent) + Send + Sync>,
}

impl AffinityEventSink {
    pub fn new(
        partition: RoutingPartitionId,
        publish: impl Fn(AffinityBindingEvent) + Send + Sync + 'static,
    ) -> Self {
        Self {
            partition,
            publish: Arc::new(publish),
        }
    }
}

impl AffinityReplicaSink for AffinityEventSink {
    fn publish(&self, session_id: &str, target: AffinityTarget, version: AffinityVersion) {
        (self.publish)(AffinityBindingEvent::new(
            &self.partition,
            session_id,
            target,
            version,
        ));
    }
}

/// What the applier did with a received binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicaEventDisposition {
    Applied(ReplicaApplyOutcome),
    /// Published by this replica.
    OwnWriter,
    /// For another partition than the receiving resolver's.
    OtherPartition,
    /// The host cannot schedule the bound worker; the version was still
    /// observed so this replica's own versions stay ahead.
    UnknownWorker,
}

impl AffinityResolver {
    /// Apply a binding another replica published, for the resolver serving
    /// `partition`. Every host's transport ends here.
    pub fn apply_replica_event(
        &self,
        partition: &RoutingPartitionId,
        event: AffinityBindingEvent,
    ) -> ReplicaEventDisposition {
        if event.writer_id == self.table().writer_id() {
            return ReplicaEventDisposition::OwnWriter;
        }
        if event.partition != *partition {
            return ReplicaEventDisposition::OtherPartition;
        }
        self.table().observe_replica_sequence(event.sequence);
        let (target, version) = (event.target(), event.version());
        if !self.is_schedulable(target) {
            tracing::trace!(
                key = %event.partition,
                worker_id = event.worker_id,
                "Dropping session affinity replica update: worker not schedulable here"
            );
            return ReplicaEventDisposition::UnknownWorker;
        }
        let outcome = self
            .table()
            .apply_replica_update(event.session_id, target, version);
        tracing::trace!(
            worker_id = event.worker_id,
            ?outcome,
            "Applied session affinity replica update"
        );
        ReplicaEventDisposition::Applied(outcome)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Mutex;
    use std::time::Duration;

    use super::super::resolver::TargetLiveness;
    use super::super::table::{SessionAffinity, SessionAffinityConfig};
    use super::*;
    use crate::protocols::WorkerId;

    struct Live(HashSet<WorkerId>);

    impl TargetLiveness for Live {
        fn is_schedulable(&self, target: AffinityTarget) -> bool {
            self.0.contains(&target.worker_id)
        }
    }

    fn partition_id(name: &str) -> RoutingPartitionId {
        RoutingPartitionId::new(name, "default")
    }

    fn resolver(live: &[WorkerId]) -> Arc<AffinityResolver> {
        let table =
            SessionAffinity::with_config(SessionAffinityConfig::new(Duration::from_secs(60)))
                .expect("table");
        Arc::new(AffinityResolver::new(
            table,
            Arc::new(Live(live.iter().copied().collect())),
        ))
    }

    /// Install a sink that records every event the table publishes.
    fn record_events(
        resolver: &AffinityResolver,
        writer_id: u64,
        partition: &RoutingPartitionId,
    ) -> Arc<Mutex<Vec<AffinityBindingEvent>>> {
        let events = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&events);
        assert!(resolver.table().enable_replication(
            writer_id,
            Arc::new(AffinityEventSink::new(partition.clone(), move |event| {
                recorded.lock().unwrap().push(event);
            })),
        ));
        events
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
        let resolver = resolver(&[1]);
        let _events = record_events(&resolver, 7, &partition);

        assert_eq!(
            resolver.apply_replica_event(&partition, event(&partition, 7, 1)),
            ReplicaEventDisposition::OwnWriter
        );
        assert_eq!(
            resolver.apply_replica_event(&partition, event(&partition_id("other"), 9, 2)),
            ReplicaEventDisposition::OtherPartition
        );
        let mut unknown = event(&partition, 9, 3);
        unknown.worker_id = 5;
        assert_eq!(
            resolver.apply_replica_event(&partition, unknown),
            ReplicaEventDisposition::UnknownWorker
        );
        assert!(
            resolver.table().next_version().sequence > 3,
            "an unapplied event still advances the replica clock"
        );
        assert_eq!(
            resolver.apply_replica_event(&partition, event(&partition, 9, 4)),
            ReplicaEventDisposition::Applied(ReplicaApplyOutcome::Inserted)
        );
        assert_eq!(
            resolver.table().query_target("s", None).expect("query"),
            Some(AffinityTarget::new(1, Some(0)))
        );
    }

    /// Two replicas with different liveness sources (a frontend and a
    /// selection service, in effect) converge on one binding through the
    /// shared schema, in both directions.
    #[tokio::test]
    async fn replicas_converge_through_the_shared_schema() {
        let partition = partition_id("model");
        let frontend = resolver(&[1, 2]);
        let service = resolver(&[1, 2]);
        let frontend_events = record_events(&frontend, 100, &partition);
        let service_events = record_events(&service, 200, &partition);

        // The frontend binds; the service applies what it published.
        let hold = frontend
            .resolve("s", None, std::future::pending())
            .await
            .expect("resolve")
            .hold
            .expect("hold");
        drop(
            frontend
                .commit(hold, AffinityTarget::new(1, Some(0)))
                .expect("commit"),
        );
        let published: Vec<_> = frontend_events.lock().unwrap().drain(..).collect();
        assert!(!published.is_empty());
        for event in published {
            assert_eq!(event.partition, partition);
            assert_eq!(event.writer_id, 100);
            assert!(matches!(
                service.apply_replica_event(&partition, event),
                ReplicaEventDisposition::Applied(_)
            ));
        }
        assert_eq!(
            service.table().query_target("s", None).expect("query"),
            Some(AffinityTarget::new(1, Some(0)))
        );

        // The service re-binds after the worker departs; the frontend follows.
        let hold = service
            .resolve("s", None, std::future::pending())
            .await
            .expect("resolve")
            .hold
            .expect("hold");
        let Err(_) = service.commit(hold, AffinityTarget::new(2, Some(0))) else {
            panic!("hard mismatch while the worker is live is rejected");
        };
        let hold = service
            .resolve("s", None, std::future::pending())
            .await
            .expect("resolve")
            .hold
            .expect("hold");
        drop(
            service
                .commit(hold, AffinityTarget::new(2, Some(0)))
                .expect("rebind"),
        );
        let published: Vec<_> = service_events.lock().unwrap().drain(..).collect();
        for event in published {
            assert_eq!(event.writer_id, 200);
            frontend.apply_replica_event(&partition, event);
        }
        assert_eq!(
            frontend.table().query_target("s", None).expect("query"),
            Some(AffinityTarget::new(2, Some(0))),
            "the frontend converges on the service's newer binding"
        );
    }
}
