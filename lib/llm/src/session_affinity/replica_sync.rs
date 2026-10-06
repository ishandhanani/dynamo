// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replication of session bindings between frontends over the runtime event
//! plane. The schema and the applier are the shared ones
//! (`AffinityBindingEvent`, `AffinityResolver::apply_replica_event`); this
//! module is the transport: publishers, subscribers, and the direct-ZMQ
//! fan-in when the deployment uses it.

use std::sync::{Arc, Weak};

use anyhow::{Context, Result};
use dynamo_runtime::{
    component::Client,
    discovery::EventTransportKind,
    traits::DistributedRuntimeProvider,
    transports::event_plane::{
        Codec, EventPublisher, EventSubscriber, ValidatedEnvelope, uses_direct_zmq,
    },
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tokio::{sync::mpsc, task::JoinHandle};
use tokio_util::sync::CancellationToken;

use crate::direct_zmq_fan_in::{
    ContinuityMode, FanInEvent, FanInObservation, start_direct_zmq_fan_in,
};
use dynamo_kv_router::RoutingPartitionId;
use dynamo_kv_router::services::selection::affinity::{
    AffinityBindingEvent, AffinityEventSink, AffinityReplicaSink, AffinityResolver,
};

/// The v1 subject carries [`SessionAffinityUpdate`], which names no partition;
/// the v2 subject carries the shared [`AffinityBindingEvent`]. Both are
/// published and applied for one release so mixed-version frontends stay
/// converged.
// Compatibility with v1.6 frontends during v1.7 rolling upgrades.
// TODO(v1.8): Publish and subscribe on `SESSION_AFFINITY_SUBJECT_V2` only.
pub(super) const SESSION_AFFINITY_SUBJECT: &str = "session_affinity_events";
pub(super) const SESSION_AFFINITY_SUBJECT_V2: &str = "session_affinity_events_v2";
const OUTBOUND_CHANNEL_CAPACITY: usize = 4_096;
const DIRECT_ZMQ_RCVHWM: i32 = 1_024;

/// The v1 wire payload: one binding without its partition. A receiver
/// assumes its own partition, which is what every v1.6 frontend did.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct SessionAffinityUpdate {
    pub session_id: String,
    pub worker_id: u64,
    pub dp_rank: Option<u32>,
    pub sequence: u64,
    pub writer_id: u64,
}

impl SessionAffinityUpdate {
    fn from_event(event: &AffinityBindingEvent) -> Self {
        Self {
            session_id: event.session_id.clone(),
            worker_id: event.worker_id,
            dp_rank: event.dp_rank,
            sequence: event.sequence,
            writer_id: event.writer_id,
        }
    }

    fn into_event(self, partition: &RoutingPartitionId) -> AffinityBindingEvent {
        AffinityBindingEvent {
            partition: partition.clone(),
            session_id: self.session_id,
            worker_id: self.worker_id,
            dp_rank: self.dp_rank,
            sequence: self.sequence,
            writer_id: self.writer_id,
        }
    }
}

/// Feeds received bindings to the resolver without keeping it alive.
#[derive(Clone)]
struct ReplicaUpdateApplier {
    partition: RoutingPartitionId,
    resolver: Weak<AffinityResolver>,
}

impl ReplicaUpdateApplier {
    /// `false` once the resolver is gone: the subscriber stops.
    fn apply(&self, event: AffinityBindingEvent) -> bool {
        let Some(resolver) = self.resolver.upgrade() else {
            return false;
        };
        let (worker_id, dp_rank) = (event.worker_id, event.dp_rank);
        let disposition = resolver.apply_replica_event(&self.partition, event);
        tracing::trace!(
            worker_id,
            ?dp_rank,
            ?disposition,
            "processed best-effort session affinity update"
        );
        true
    }
}

pub(super) struct ReplicaSyncRuntime {
    sink: Arc<dyn AffinityReplicaSink>,
    cancel: CancellationToken,
    publisher_task: Option<JoinHandle<()>>,
    subscriber_tasks: Vec<JoinHandle<()>>,
}

impl ReplicaSyncRuntime {
    /// Returns the runtime and this replica's writer id (its discovery instance id).
    pub(super) async fn start(
        client: Client,
        resolver: Weak<AffinityResolver>,
        partition: RoutingPartitionId,
    ) -> Result<(Self, u64)> {
        let endpoint = &client.endpoint;
        let router_id = endpoint.drt().discovery().instance_id();
        let transport_kind = endpoint.drt().default_event_transport_kind();
        let publisher_v1 = EventPublisher::for_endpoint_with_transport(
            endpoint,
            SESSION_AFFINITY_SUBJECT,
            transport_kind,
        )
        .await
        .context("create session affinity event publisher")?;
        let publisher_v2 = EventPublisher::for_endpoint_with_transport(
            endpoint,
            SESSION_AFFINITY_SUBJECT_V2,
            transport_kind,
        )
        .await
        .context("create session affinity v2 event publisher")?;
        let applier = ReplicaUpdateApplier {
            partition: partition.clone(),
            resolver,
        };

        let cancel = CancellationToken::new();
        let v2_applier = applier.clone();
        let v2_task = start_subscriber::<AffinityBindingEvent>(
            &client,
            SESSION_AFFINITY_SUBJECT_V2,
            transport_kind,
            publisher_v2.publisher_id(),
            cancel.clone(),
            move |event| v2_applier.apply(event),
        )
        .await?;
        let v1_applier = applier;
        let v1_partition = partition.clone();
        let v1_task = start_subscriber::<SessionAffinityUpdate>(
            &client,
            SESSION_AFFINITY_SUBJECT,
            transport_kind,
            publisher_v1.publisher_id(),
            cancel.clone(),
            move |update| v1_applier.apply(update.into_event(&v1_partition)),
        )
        .await?;

        let (tx, mut rx) = mpsc::channel::<AffinityBindingEvent>(OUTBOUND_CHANNEL_CAPACITY);
        let publisher_cancel = cancel.clone();
        let publisher_task = tokio::spawn(async move {
            loop {
                let event = tokio::select! {
                    _ = publisher_cancel.cancelled() => return,
                    event = rx.recv() => event,
                };
                let Some(event) = event else {
                    return;
                };
                let update = SessionAffinityUpdate::from_event(&event);
                if let Err(error) = publisher_v2.publish(&event).await {
                    tracing::trace!(
                        worker_id = event.worker_id,
                        dp_rank = ?event.dp_rank,
                        %error,
                        "failed to publish best-effort session affinity update"
                    );
                }
                if let Err(error) = publisher_v1.publish(&update).await {
                    tracing::trace!(
                        worker_id = update.worker_id,
                        dp_rank = ?update.dp_rank,
                        %error,
                        "failed to publish best-effort session affinity v1 update"
                    );
                }
            }
        });

        Ok((
            Self {
                sink: event_sink(partition, tx),
                cancel,
                publisher_task: Some(publisher_task),
                subscriber_tasks: vec![v2_task, v1_task],
            },
            router_id,
        ))
    }

    /// The sink the table publishes into.
    pub(super) fn sink(&self) -> Arc<dyn AffinityReplicaSink> {
        Arc::clone(&self.sink)
    }

    pub(super) fn shutdown_now(&mut self) {
        self.cancel.cancel();
        if let Some(task) = self.publisher_task.take() {
            task.abort();
        }
        self.subscriber_tasks.clear();
    }

    #[cfg(test)]
    pub(super) fn for_test(capacity: usize) -> (Self, mpsc::Receiver<AffinityBindingEvent>) {
        let (tx, rx) = mpsc::channel(capacity);
        (
            Self {
                sink: event_sink(crate::kv_router::embedded::embedded_partition_key(None), tx),
                cancel: CancellationToken::new(),
                publisher_task: None,
                subscriber_tasks: Vec::new(),
            },
            rx,
        )
    }
}

/// A sink that queues each binding for the publisher task; a full queue
/// drops the update (replication is best effort).
fn event_sink(
    partition: RoutingPartitionId,
    tx: mpsc::Sender<AffinityBindingEvent>,
) -> Arc<dyn AffinityReplicaSink> {
    Arc::new(AffinityEventSink::new(partition, move |event| {
        if let Err(error) = tx.try_send(event) {
            tracing::trace!(%error, "dropping best-effort session affinity update");
        }
    }))
}

/// Subscribe to `subject` on the deployment's event transport and feed each
/// decoded payload to `apply`, which returns `false` to stop.
async fn start_subscriber<T>(
    client: &Client,
    subject: &'static str,
    transport_kind: EventTransportKind,
    publisher_id: u64,
    cancel: CancellationToken,
    apply: impl Fn(T) -> bool + Clone + Send + Sync + 'static,
) -> Result<JoinHandle<()>>
where
    T: DeserializeOwned + Send + 'static,
{
    let endpoint = &client.endpoint;
    if should_use_direct_sync(transport_kind, uses_direct_zmq(transport_kind)) {
        let codec = Codec::default();
        let handler = move |envelope: ValidatedEnvelope| {
            let payload = codec
                .decode_payload::<T>(&envelope.payload)
                .context("decode session affinity update")?;
            apply(payload);
            Ok(())
        };
        let observer = move |observation: FanInObservation| match observation.event {
            FanInEvent::SequenceGap { missing } => tracing::warn!(
                subject,
                publisher_id = observation.publisher_id,
                generation = observation.generation,
                missing,
                "session affinity direct-ZMQ source skipped envelopes"
            ),
            FanInEvent::OutOfOrder => tracing::warn!(
                subject,
                publisher_id = observation.publisher_id,
                generation = observation.generation,
                "session affinity direct-ZMQ source received a non-increasing sequence"
            ),
            _ => {}
        };
        return start_direct_zmq_fan_in(
            endpoint.clone(),
            subject,
            DIRECT_ZMQ_RCVHWM,
            Some(publisher_id),
            ContinuityMode::TrackFromZero,
            cancel,
            handler,
            observer,
        )
        .await
        .context("start direct-ZMQ session affinity subscriber");
    }
    let mut subscriber =
        EventSubscriber::for_endpoint_with_transport(endpoint, subject, transport_kind)
            .await
            .context("create session affinity event subscriber")?
            .typed::<T>();
    Ok(tokio::spawn(async move {
        loop {
            let event = tokio::select! {
                _ = cancel.cancelled() => return,
                event = subscriber.next() => event,
            };
            let Some(event) = event else {
                return;
            };
            let payload = match event {
                Ok((_envelope, payload)) => payload,
                Err(error) => {
                    tracing::trace!(
                        %error,
                        "failed to receive best-effort session affinity update"
                    );
                    continue;
                }
            };
            if !apply(payload) {
                return;
            }
        }
    }))
}

impl Drop for ReplicaSyncRuntime {
    fn drop(&mut self) {
        self.shutdown_now();
    }
}

fn should_use_direct_sync(transport_kind: EventTransportKind, direct_zmq_topology: bool) -> bool {
    transport_kind == EventTransportKind::Zmq && direct_zmq_topology
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session_affinity::{AlwaysLive, HostAffinity, LiveWorkers};
    use dynamo_kv_router::services::selection::affinity::{
        AffinityTarget, AffinityVersion, ReplicaApplyOutcome, ReplicaEventDisposition,
        SessionAffinity, SessionAffinityConfig,
    };
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        discovery::{DiscoveryQuery, EventChannelQuery},
        distributed::DistributedConfig,
    };
    use std::time::Duration;

    #[test]
    fn direct_sync_is_selected_only_for_unbrokered_zmq() {
        assert!(should_use_direct_sync(EventTransportKind::Zmq, true));
        assert!(!should_use_direct_sync(EventTransportKind::Zmq, false));
        assert!(!should_use_direct_sync(EventTransportKind::Nats, false));
    }

    #[tokio::test]
    async fn replica_update_backpressure_is_nonfatal() {
        let (runtime, mut rx) = ReplicaSyncRuntime::for_test(1);
        let sink = runtime.sink();
        sink.publish(
            "first",
            AffinityTarget {
                worker_id: 10,
                dp_rank: Some(0),
            },
            AffinityVersion {
                sequence: 1,
                writer_id: 7,
            },
        );
        sink.publish(
            "second",
            AffinityTarget {
                worker_id: 11,
                dp_rank: Some(0),
            },
            AffinityVersion {
                sequence: 2,
                writer_id: 7,
            },
        );

        let update = rx.recv().await.unwrap();
        assert_eq!(update.session_id, "first");
        assert_eq!(update.writer_id, 7);
        assert_eq!(
            update.partition,
            crate::kv_router::embedded::embedded_partition_key(None)
        );
        assert!(rx.try_recv().is_err());
    }

    /// A v1 update (no partition) and a v2 event both apply to the receiving
    /// partition; a v2 event for another partition does not.
    #[tokio::test]
    async fn v1_and_v2_updates_both_apply_to_the_partition() {
        let partition = crate::kv_router::embedded::embedded_partition_key(Some("model"));
        let table =
            SessionAffinity::with_config(SessionAffinityConfig::new(Duration::from_secs(10)))
                .unwrap();
        let resolver = Arc::new(AffinityResolver::new(
            table,
            Arc::new(LiveWorkers([10, 11].into_iter().collect())),
        ));
        let applier = ReplicaUpdateApplier {
            partition: partition.clone(),
            resolver: Arc::downgrade(&resolver),
        };

        let v1 = SessionAffinityUpdate {
            session_id: "legacy".to_string(),
            worker_id: 10,
            dp_rank: Some(0),
            sequence: 5,
            writer_id: 9,
        };
        assert!(applier.apply(v1.into_event(&partition)));
        assert_eq!(
            resolver.table().query_target("legacy", None).unwrap(),
            Some(AffinityTarget::new(10, Some(0)))
        );

        let v2 = AffinityBindingEvent {
            partition: partition.clone(),
            session_id: "current".to_string(),
            worker_id: 11,
            dp_rank: None,
            sequence: 6,
            writer_id: 9,
        };
        assert!(applier.apply(v2));
        assert_eq!(
            resolver.table().query_target("current", None).unwrap(),
            Some(AffinityTarget::new(11, None))
        );

        let elsewhere = AffinityBindingEvent {
            partition: crate::kv_router::embedded::embedded_partition_key(Some("other")),
            session_id: "elsewhere".to_string(),
            worker_id: 10,
            dp_rank: None,
            sequence: 7,
            writer_id: 9,
        };
        assert_eq!(
            resolver.apply_replica_event(&partition, elsewhere),
            ReplicaEventDisposition::OtherPartition
        );
        assert_eq!(
            resolver.table().query_target("elsewhere", None).unwrap(),
            None
        );

        // The same binding on both subjects is idempotent.
        let again = SessionAffinityUpdate {
            session_id: "current".to_string(),
            worker_id: 11,
            dp_rank: None,
            sequence: 6,
            writer_id: 9,
        };
        assert_eq!(
            resolver.apply_replica_event(&partition, again.into_event(&partition)),
            ReplicaEventDisposition::Applied(ReplicaApplyOutcome::Refreshed)
        );

        drop(resolver);
        assert!(
            !applier.apply(
                SessionAffinityUpdate {
                    session_id: "late".to_string(),
                    worker_id: 10,
                    dp_rank: None,
                    sequence: 8,
                    writer_id: 9,
                }
                .into_event(&partition)
            )
        );
    }

    fn host_affinity() -> HostAffinity {
        HostAffinity::standalone(
            Duration::from_secs(10),
            crate::session_affinity::SessionAffinityMode::Hard,
            Arc::new(AlwaysLive),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn rejected_worker_update_still_advances_replica_clock() {
        let partition = crate::kv_router::embedded::embedded_partition_key(None);
        let table =
            SessionAffinity::with_config(SessionAffinityConfig::new(Duration::from_secs(10)))
                .unwrap();
        let resolver = Arc::new(AffinityResolver::new(
            table,
            Arc::new(LiveWorkers(Default::default())),
        ));
        let table = resolver.table();
        let baseline = table.next_version();
        let sequence = baseline.sequence.saturating_add(10);
        let applier = ReplicaUpdateApplier {
            partition: partition.clone(),
            resolver: Arc::downgrade(&resolver),
        };

        assert!(
            applier.apply(
                SessionAffinityUpdate {
                    session_id: "unknown-worker".to_string(),
                    worker_id: 10,
                    dp_rank: Some(0),
                    sequence,
                    writer_id: 9,
                }
                .into_event(&partition)
            )
        );

        assert_eq!(table.next_version().sequence, sequence.saturating_add(1));
    }

    #[tokio::test]
    async fn replica_runtime_drop_unregisters_before_replacement() {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap();
        let namespace_name = format!("session-affinity-replica-{}", uuid::Uuid::new_v4());
        let component_name = "workers";
        let component = drt
            .namespace(namespace_name.clone())
            .unwrap()
            .component(component_name)
            .unwrap();
        let endpoint = component.endpoint("generate");
        let query = DiscoveryQuery::EventChannels(EventChannelQuery::endpoint_topic(
            endpoint.id(),
            SESSION_AFFINITY_SUBJECT,
        ));
        let client = endpoint.client().await.unwrap();

        let original = host_affinity();
        let shared = original.clone();
        let (first, second) = tokio::join!(
            original.enable_replica_sync(client.clone()),
            shared.enable_replica_sync(client.clone()),
        );
        first.unwrap();
        second.unwrap();
        wait_for_registration_count(&drt, &query, 1).await;

        drop(original);
        shared.enable_replica_sync(client.clone()).await.unwrap();
        wait_for_registration_count(&drt, &query, 1).await;
        drop(shared);
        wait_for_registration_count(&drt, &query, 0).await;

        let replacement = host_affinity();
        replacement.enable_replica_sync(client).await.unwrap();
        wait_for_registration_count(&drt, &query, 1).await;

        drop(replacement);
        wait_for_registration_count(&drt, &query, 0).await;
    }

    async fn wait_for_registration_count(
        drt: &DistributedRuntime,
        query: &DiscoveryQuery,
        expected: usize,
    ) {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let registrations = drt.discovery().list(query.clone()).await.unwrap();
                if registrations.len() == expected {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("event registration count did not reach {expected}"));
    }
}
