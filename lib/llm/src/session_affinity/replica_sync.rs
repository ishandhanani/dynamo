// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replication of session bindings between frontends over the runtime event
//! plane. The schema and the applier are the shared ones
//! (`AffinityBindingEvent`,
//! [`SessionAffinity::apply_replica_event`](dynamo_kv_router::services::selection::affinity::SessionAffinity::apply_replica_event)); this
//! module is the transport: publishers, subscribers, and the direct-ZMQ
//! fan-in when the deployment uses it.

use std::sync::Arc;

use anyhow::{Context, Result};
use dynamo_runtime::{
    component::Client,
    discovery::EventTransportKind,
    traits::DistributedRuntimeProvider,
    transports::event_plane::{
        Codec, EventPublisher, EventSubscriber, ValidatedEnvelope, uses_direct_zmq,
    },
};
use tokio::{sync::mpsc, task::JoinHandle};
use tokio_util::sync::CancellationToken;

use crate::direct_zmq_fan_in::{
    ContinuityMode, FanInEvent, FanInObservation, start_direct_zmq_fan_in,
};
use dynamo_kv_router::RoutingPartitionId;
use dynamo_kv_router::services::selection::affinity::{
    AffinityBindingEvent, AffinityReplicaSink, AffinityTarget, AffinityVersion, SessionAffinity,
    WeakSessionAffinity,
};

/// Partition-scoped bindings use the same payload on every transport.
pub(super) const SESSION_AFFINITY_SUBJECT: &str = "session_affinity_events";
const OUTBOUND_CHANNEL_CAPACITY: usize = 4_096;
const DIRECT_ZMQ_RCVHWM: i32 = 1_024;

/// Returns `false` when the table is gone, so the subscriber can stop.
fn apply_update(
    table: &WeakSessionAffinity,
    partition: &RoutingPartitionId,
    event: AffinityBindingEvent,
    is_live: impl Fn(AffinityTarget) -> bool,
) -> bool {
    let Some(table) = table.upgrade() else {
        return false;
    };
    table.apply_replica_event(partition, event, is_live);
    true
}

pub(super) struct ReplicaSyncRuntime {
    partition: RoutingPartitionId,
    tx: mpsc::Sender<AffinityBindingEvent>,
    cancel: CancellationToken,
    publisher_task: Option<JoinHandle<()>>,
    subscriber_task: Option<JoinHandle<()>>,
}

impl ReplicaSyncRuntime {
    /// Returns the runtime and this replica's writer id (its discovery instance id).
    pub(super) async fn start(
        client: Client,
        table: WeakSessionAffinity,
        partition: RoutingPartitionId,
        is_live: impl Fn(AffinityTarget) -> bool + Clone + Send + Sync + 'static,
    ) -> Result<(Self, u64)> {
        let endpoint = &client.endpoint;
        let router_id = endpoint.drt().discovery().instance_id();
        let transport_kind = endpoint.drt().default_event_transport_kind();
        let publisher = EventPublisher::for_endpoint_with_transport(
            endpoint,
            SESSION_AFFINITY_SUBJECT,
            transport_kind,
        )
        .await
        .context("create session affinity event publisher")?;
        let cancel = CancellationToken::new();
        let receive_partition = partition.clone();
        let subscriber_task = start_subscriber(
            &client,
            transport_kind,
            publisher.publisher_id(),
            cancel.clone(),
            move |event| apply_update(&table, &receive_partition, event, &is_live),
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
                if let Err(error) = publisher.publish(&event).await {
                    tracing::trace!(
                        worker_id = event.worker_id,
                        dp_rank = ?event.dp_rank,
                        %error,
                        "failed to publish best-effort session affinity update"
                    );
                }
            }
        });

        Ok((
            Self {
                partition,
                tx,
                cancel,
                publisher_task: Some(publisher_task),
                subscriber_task: Some(subscriber_task),
            },
            router_id,
        ))
    }

    pub(super) fn shutdown_now(&mut self) {
        self.cancel.cancel();
        if let Some(task) = self.publisher_task.take() {
            task.abort();
        }
        if let Some(task) = self.subscriber_task.take() {
            task.abort();
        }
    }

    #[cfg(test)]
    pub(super) fn for_test(capacity: usize) -> (Arc<Self>, mpsc::Receiver<AffinityBindingEvent>) {
        let (tx, rx) = mpsc::channel(capacity);
        (
            Arc::new(Self {
                partition: crate::kv_router::embedded::embedded_partition_key(None),
                tx,
                cancel: CancellationToken::new(),
                publisher_task: None,
                subscriber_task: None,
            }),
            rx,
        )
    }
}

impl AffinityReplicaSink for ReplicaSyncRuntime {
    fn publish(&self, session_id: &str, target: AffinityTarget, version: AffinityVersion) {
        AffinityBindingEvent::new(&self.partition, session_id, target, version).enqueue(&self.tx);
    }
}

/// The table owns the transport through its sink. Subscribers keep only a weak
/// table handle, so dropping the last table also stops both transport tasks.
pub(crate) async fn enable_replica_sync(
    table: &SessionAffinity,
    partition: RoutingPartitionId,
    client: Client,
    workers: Option<crate::discovery::RuntimeConfigWatch>,
) -> Result<()> {
    let weak = table.downgrade();
    table
        .enable_replication_with(async move {
            let live_client = client.clone();
            let (runtime, writer_id) =
                ReplicaSyncRuntime::start(client, weak, partition, move |target| {
                    super::target_is_live(&live_client, workers.as_ref(), target)
                })
                .await?;
            Ok::<_, anyhow::Error>((writer_id, Arc::new(runtime) as Arc<dyn AffinityReplicaSink>))
        })
        .await
}

/// Subscribe on the deployment's event transport and feed each
/// decoded payload to `apply`, which returns `false` to stop.
async fn start_subscriber(
    client: &Client,
    transport_kind: EventTransportKind,
    publisher_id: u64,
    cancel: CancellationToken,
    apply: impl Fn(AffinityBindingEvent) -> bool + Clone + Send + Sync + 'static,
) -> Result<JoinHandle<()>> {
    let subject = SESSION_AFFINITY_SUBJECT;
    let endpoint = &client.endpoint;
    if should_use_direct_sync(transport_kind, uses_direct_zmq(transport_kind)) {
        let codec = Codec::default();
        let handler_cancel = cancel.clone();
        let handler = move |envelope: ValidatedEnvelope| {
            let payload = codec
                .decode_payload::<AffinityBindingEvent>(&envelope.payload)
                .context("decode session affinity update")?;
            if !apply(payload) {
                handler_cancel.cancel();
            }
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
            .typed::<AffinityBindingEvent>();
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
    use dynamo_kv_router::services::selection::affinity::{SessionAffinity, SessionAffinityConfig};
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
    async fn replica_subscriber_does_not_keep_table_alive() {
        let partition = crate::kv_router::embedded::embedded_partition_key(Some("model"));
        let table =
            SessionAffinity::with_config(SessionAffinityConfig::new(Duration::from_secs(10)))
                .unwrap();
        let weak = table.downgrade();
        let event = AffinityBindingEvent {
            partition: partition.clone(),
            session_id: "session".to_string(),
            worker_id: 10,
            dp_rank: Some(0),
            sequence: 5,
            writer_id: 9,
        };
        assert!(apply_update(&weak, &partition, event.clone(), |_| true));
        assert_eq!(
            table.query_target("session", None).unwrap(),
            Some(AffinityTarget::new(10, Some(0)))
        );
        drop(table);
        assert!(!apply_update(&weak, &partition, event, |_| true));
    }

    fn table() -> SessionAffinity {
        SessionAffinity::with_config(SessionAffinityConfig::new(Duration::from_secs(10))).unwrap()
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

        let partition = crate::kv_router::embedded::embedded_partition_key(None);
        let original = table();
        let shared = original.clone();
        let (first, second) = tokio::join!(
            enable_replica_sync(&original, partition.clone(), client.clone(), None),
            enable_replica_sync(&shared, partition.clone(), client.clone(), None),
        );
        first.unwrap();
        second.unwrap();
        wait_for_registration_count(&drt, &query, 1).await;

        drop(original);
        enable_replica_sync(&shared, partition.clone(), client.clone(), None)
            .await
            .unwrap();
        wait_for_registration_count(&drt, &query, 1).await;
        drop(shared);
        wait_for_registration_count(&drt, &query, 0).await;

        let replacement = table();
        enable_replica_sync(&replacement, partition, client, None)
            .await
            .unwrap();
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
