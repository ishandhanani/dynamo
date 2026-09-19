// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reconciles scheduler bookings from the engine control channel, never HTTP bytes.

use std::{collections::HashSet, sync::Arc, time::Duration};

use dynamo_runtime::pipeline::{Context, network::egress::push_router::PushRouter};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{StreamExt, future::BoxFuture, stream::FuturesUnordered};
use tokio::time::{Instant, MissedTickBehavior};
use tokio_util::sync::CancellationToken;

use crate::kv_router::native::NativeReservation;
use crate::protocols::sglang::http::lifecycle::{
    self, Child, ChildKind, Descriptor, Operation, Request, Response, Snapshot,
};

pub type LifecycleClient = PushRouter<Request, Annotated<Response>>;

const RPC_TIMEOUT: Duration = Duration::from_secs(25);
const MAINTENANCE_INTERVAL: Duration = Duration::from_secs(5);
const LOST_CONTROL_TIMEOUT: Duration = Duration::from_secs(30);
const CANCEL_TIMEOUT: Duration = Duration::from_secs(90);

/// The order is the engine's child-creation order: ordinary batch items, or all
/// sampling warmups followed by each input's samples. Every slot is booked
/// before dispatch; a sealed snapshot releases slots never created by the engine.
pub struct ReservedChild {
    pub kind: ChildKind,
    pub reservation: NativeReservation,
}

pub struct NativeAttempt {
    pub(super) attempt_id: String,
    pub(super) incarnation: String,
    worker_id: u64,
    pub(super) dp_rank: Option<u32>,
    stage: String,
    client: Arc<LifecycleClient>,
    reservations: Vec<Option<ReservedChild>>,
    slots: Vec<(ChildKind, Option<u32>)>,
    prefill_applied: HashSet<usize>,
    observed: Observed,
}

impl NativeAttempt {
    /// `client` must retain the same WorkerSet admission fence as the HTTP client.
    /// Refresh the descriptor on the selected worker before constructing an attempt.
    pub fn new(
        client: Arc<LifecycleClient>,
        descriptor: Descriptor,
        stage: String,
        reservations: Vec<ReservedChild>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            descriptor.version == 1 && descriptor.header_overrides,
            "native lifecycle is not supported"
        );
        anyhow::ensure!(
            lifecycle::is_valid_id(&descriptor.incarnation),
            "invalid worker incarnation"
        );
        anyhow::ensure!(
            matches!(stage.as_str(), "null" | "prefill" | "decode"),
            "invalid native stage"
        );
        let worker = reservations
            .first()
            .ok_or_else(|| anyhow::anyhow!("native attempt has no reservation"))?
            .reservation
            .target();
        anyhow::ensure!(
            reservations
                .iter()
                .all(|child| child.reservation.target() == worker),
            "native batch reservations must share a worker and DP rank"
        );
        Ok(Self {
            attempt_id: uuid::Uuid::new_v4().simple().to_string(),
            incarnation: descriptor.incarnation,
            worker_id: worker.worker_id,
            dp_rank: worker.dp_rank,
            stage,
            client,
            observed: Observed::default(),
            slots: reservations
                .iter()
                .map(|child| (child.kind, worker.dp_rank))
                .collect(),
            prefill_applied: HashSet::new(),
            reservations: reservations.into_iter().map(Some).collect(),
        })
    }

    pub fn worker_id(&self) -> u64 {
        self.worker_id
    }

    pub async fn describe(client: &LifecycleClient, worker_id: u64) -> anyhow::Result<Descriptor> {
        match rpc(client, worker_id, Request::Describe).await? {
            Response::Descriptor(descriptor) => Ok(descriptor),
            response => anyhow::bail!("native lifecycle discovery failed: {response:?}"),
        }
    }

    fn request(&self, operation: Operation) -> Request {
        Request::Attempt {
            incarnation: self.incarnation.clone(),
            attempt_id: self.attempt_id.clone(),
            operation,
        }
    }

    fn control(&self, operation: Operation) -> BoxFuture<'static, anyhow::Result<Response>> {
        let client = self.client.clone();
        let worker_id = self.worker_id;
        let request = self.request(operation);
        Box::pin(async move { rpc(&client, worker_id, request).await })
    }

    /// Ownership moves into this task before HTTP dispatch. Client cancellation
    /// requests an engine abort; it does not drop the scheduler reservations.
    pub(super) fn start(self, cancellation: CancellationToken) {
        tokio::spawn(async move {
            let attempt_id = self.attempt_id.clone();
            if let Err(error) = self.reconcile(cancellation).await {
                // The lease Drop path releases local bookings even when control is
                // lost. This is explicitly uncertain, never a fabricated engine ack.
                tracing::error!(%attempt_id, %error, "native generation accounting released without confirmed engine cleanup");
            }
        });
    }

    async fn reconcile(mut self, cancellation: CancellationToken) -> anyhow::Result<()> {
        let mut maintenance = tokio::time::interval(MAINTENANCE_INTERVAL);
        maintenance.set_missed_tick_behavior(MissedTickBehavior::Skip);
        // An RPC may take longer than the configured local expiry scan. Keep
        // local lease renewal independent of both long polls and control calls.
        let expiry = dynamo_kv_router::multi_worker_sequence::active_request_expiry_duration();
        let mut heartbeat = tokio::time::interval((expiry / 3).min(MAINTENANCE_INTERVAL));
        heartbeat.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let mut last_control = Instant::now();
        let mut cancel_started = None;
        let mut controls = FuturesUnordered::new();
        let mut poll = self.control(Operation::Snapshot { after: -1 });
        loop {
            let mut polled = false;
            let result = tokio::select! {
                _ = cancellation.cancelled(), if cancel_started.is_none() => {
                    cancel_started = Some(Instant::now());
                    // Cancel is retried by maintenance until terminal acknowledgement.
                    controls.push(self.control(Operation::Cancel));
                    continue;
                }
                _ = heartbeat.tick() => {
                    let mut expired = false;
                    for child in self.reservations.iter().flatten() {
                        expired |= child.reservation.touch().is_err();
                    }
                    if last_control.elapsed() >= LOST_CONTROL_TIMEOUT || expired {
                        cancel_started.get_or_insert_with(Instant::now);
                    }
                    if cancel_started.is_some_and(|start| start.elapsed() >= CANCEL_TIMEOUT) {
                        anyhow::bail!("engine cancellation acknowledgement timed out");
                    }
                    continue;
                }
                _ = maintenance.tick() => {
                    if controls.is_empty() {
                        let operation = if cancel_started.is_some() { Operation::Cancel }
                            else { Operation::Renew { lease_seconds: 30 } };
                        controls.push(self.control(operation));
                    }
                    continue;
                }
                Some(result) = controls.next(), if !controls.is_empty() => result,
                result = &mut poll => {
                    polled = true;
                    result
                }
            };
            match result {
                Ok(Response::Snapshot(snapshot)) => match self.apply(&snapshot).await {
                    Ok(true) => break,
                    Ok(false) => {
                        last_control = Instant::now();
                        if snapshot.cancel_requested {
                            cancel_started.get_or_insert_with(Instant::now);
                        }
                    }
                    Err(error) => {
                        tracing::warn!(attempt_id = %self.attempt_id, %error, "invalid native lifecycle snapshot; cancelling attempt");
                        cancel_started.get_or_insert_with(Instant::now);
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                },
                // Includes 404 before the HTTP handler claims the attempt,
                // worker restart, lease expiry and unavailable control transport.
                _ => tokio::time::sleep(Duration::from_millis(100)).await,
            }
            if polled {
                let after = self.observed.version.map_or(-1, |version| version as i64);
                poll = self.control(Operation::Snapshot { after });
            }
        }
        // All bookings are already released. Failure to reclaim the engine's
        // terminal snapshot is harmless: its bounded retention will reclaim it.
        let _ = self.control(Operation::Acknowledge).await;
        Ok(())
    }

    async fn apply(&mut self, snapshot: &Snapshot) -> anyhow::Result<bool> {
        anyhow::ensure!(
            snapshot.incarnation == self.incarnation
                && snapshot.attempt_id == self.attempt_id
                && snapshot.stage == self.stage,
            "native lifecycle identity mismatch"
        );
        if self
            .observed
            .version
            .is_some_and(|version| snapshot.version < version)
        {
            return Ok(false); // A renew reply can overtake an in-flight long poll.
        }
        self.observed.validate(snapshot, &self.slots)?;
        self.observed.version = Some(snapshot.version);
        self.observed.sealed = snapshot.sealed;
        self.observed.children = snapshot.children.clone();
        for (index, slot) in self.reservations.iter_mut().enumerate() {
            let Some(reserved) = slot.as_ref() else {
                continue;
            };
            let child = snapshot.children.get(index);
            if child.is_some_and(|child| child.terminal) || (child.is_none() && snapshot.sealed) {
                reserved.reservation.finish().await;
                *slot = None;
            } else if child.is_some_and(|child| child.prefill_complete)
                && !self.prefill_applied.contains(&index)
            {
                reserved.reservation.prefill_complete().await?;
                self.prefill_applied.insert(index);
            }
        }
        Ok(snapshot.terminal)
    }
}

#[derive(Default)]
struct Observed {
    version: Option<u64>,
    sealed: bool,
    children: Vec<Child>,
}

impl Observed {
    /// Validate the whole snapshot before releasing any booking. A malformed
    /// later child must not allow earlier slots to be freed speculatively.
    fn validate(
        &self,
        snapshot: &Snapshot,
        slots: &[(ChildKind, Option<u32>)],
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            snapshot.version <= i64::MAX as u64,
            "invalid lifecycle version"
        );
        anyhow::ensure!(
            snapshot.children.len() <= slots.len()
                && snapshot.children.len() >= self.children.len(),
            "native child count changed outside admitted bounds"
        );
        anyhow::ensure!(
            !self.sealed || (snapshot.sealed && snapshot.children.len() == self.children.len()),
            "sealed native attempt changed"
        );
        anyhow::ensure!(
            snapshot.terminal
                == (snapshot.sealed && snapshot.children.iter().all(|child| child.terminal)),
            "inconsistent native terminal state"
        );
        let mut ids = HashSet::new();
        for (index, child) in snapshot.children.iter().enumerate() {
            anyhow::ensure!(
                lifecycle::is_valid_id(&child.child_id) && ids.insert(&child.child_id),
                "invalid or duplicate native child identity"
            );
            anyhow::ensure!(
                !child.prefill_complete || child.dispatched,
                "undispatched child completed prefill"
            );
            anyhow::ensure!(
                !child.dispatched
                    || !(child.prefill_complete || child.terminal)
                    || child.dp_rank.is_some(),
                "scheduler acknowledgement has no DP rank"
            );
            let (kind, rank) = slots[index];
            anyhow::ensure!(
                child.kind == kind
                    && rank
                        .zip(child.dp_rank)
                        .is_none_or(|(expected, actual)| expected == actual),
                "native child does not match its reservation"
            );
            if let Some(previous) = self.children.get(index) {
                anyhow::ensure!(
                    previous.child_id == child.child_id
                        && previous.kind == child.kind
                        && previous.rid == child.rid
                        && previous
                            .dp_rank
                            .is_none_or(|rank| child.dp_rank == Some(rank))
                        && (!previous.dispatched || child.dispatched)
                        && (!previous.prefill_complete || child.prefill_complete)
                        && (!previous.terminal || child.terminal),
                    "native child identity or phase regressed"
                );
            }
        }
        Ok(())
    }
}

async fn rpc(
    client: &LifecycleClient,
    worker_id: u64,
    request: Request,
) -> anyhow::Result<Response> {
    tokio::time::timeout(RPC_TIMEOUT, async {
        let mut response = client.direct(Context::new(request), worker_id).await?;
        response
            .next()
            .await
            .ok_or_else(|| anyhow::anyhow!("empty lifecycle reply"))?
            .into_data()?
            .ok_or_else(|| anyhow::anyhow!("missing lifecycle reply"))
    })
    .await?
}

#[cfg(test)]
mod tests {
    use super::*;

    fn child(index: u8, kind: ChildKind) -> Child {
        Child {
            child_id: format!("{index:032x}"),
            rid: "overlapping-client-id".into(),
            kind,
            dp_rank: Some(3),
            dispatched: true,
            prefill_complete: false,
            terminal: false,
        }
    }

    fn snapshot(children: Vec<Child>, sealed: bool) -> Snapshot {
        Snapshot {
            incarnation: "a".repeat(32),
            attempt_id: "b".repeat(32),
            stage: "null".into(),
            version: 10,
            terminal: sealed && children.iter().all(|child| child.terminal),
            sealed,
            cancel_requested: false,
            children,
        }
    }

    #[test]
    fn recovered_fanout_preserves_identity_and_never_confuses_cancel_with_terminal() {
        let mut warmup = child(1, ChildKind::Warmup);
        warmup.terminal = true;
        let observed = Observed {
            version: Some(1),
            sealed: false,
            children: vec![warmup.clone()],
        };
        let slots = [
            (ChildKind::Warmup, Some(3)),
            (ChildKind::Sample, Some(3)),
            (ChildKind::Sample, Some(3)),
        ];
        let mut recovered = snapshot(
            vec![
                warmup,
                child(2, ChildKind::Sample),
                child(3, ChildKind::Sample),
            ],
            true,
        );
        recovered.cancel_requested = true;
        observed.validate(&recovered, &slots).unwrap();
        assert!(!recovered.terminal);
        // Missing every intermediate event is recoverable from one full snapshot.
        for child in &mut recovered.children {
            child.terminal = true;
        }
        recovered.terminal = true;
        observed.validate(&recovered, &slots).unwrap();

        recovered.children[2].dp_rank = Some(4);
        assert!(observed.validate(&recovered, &slots).is_err());
        recovered.children[2].dp_rank = Some(3);
        recovered.children[2].child_id = recovered.children[1].child_id.clone();
        assert!(observed.validate(&recovered, &slots).is_err());
        recovered.children.pop();
        recovered.children[0].terminal = false;
        recovered.terminal = false;
        assert!(observed.validate(&recovered, &slots).is_err());
    }

    #[test]
    fn native_delegated_rank_accepts_each_child_then_fences_its_observed_rank() {
        let slots = [(ChildKind::Sample, None); 2];
        let mut children = vec![child(1, ChildKind::Sample), child(2, ChildKind::Sample)];
        children[0].dp_rank = Some(0);
        children[1].dp_rank = Some(1);
        let mut state = snapshot(children, true);
        Observed::default().validate(&state, &slots).unwrap();
        assert!(
            Observed::default()
                .validate(&state, &[(ChildKind::Sample, Some(0)); 2])
                .is_err()
        );
        let observed = Observed {
            version: Some(state.version),
            sealed: state.sealed,
            children: state.children.clone(),
        };
        state.version += 1;
        state.children[1].dp_rank = Some(0);
        assert!(observed.validate(&state, &slots).is_err());
    }

    #[test]
    fn rejected_request_can_seal_empty_but_cannot_create_children_after_sealing() {
        let slots = [(ChildKind::Sample, Some(3))];
        let empty = snapshot(Vec::new(), true);
        Observed::default().validate(&empty, &slots).unwrap();
        let sealed = Observed {
            version: Some(10),
            sealed: true,
            children: Vec::new(),
        };
        assert!(
            sealed
                .validate(&snapshot(vec![child(1, ChildKind::Sample)], true), &slots)
                .is_err()
        );
        let mut unconfirmed = snapshot(vec![child(1, ChildKind::Sample)], true);
        unconfirmed.terminal = true;
        assert!(Observed::default().validate(&unconfirmed, &slots).is_err());
    }
}
