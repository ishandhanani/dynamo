// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use dynamo_kv_events::cursor::{CursorObservation, CursorState};
use dynamo_kv_events::zmq_wire::{RawKvEvent, ZmqEventNormalizer, decode_event_batch};
use dynamo_router_protocol::worker::{WorkerId, WorkerWithDpRank};

use super::backend::Indexer;
use super::registry::ListenerRecord;
use crate::services::common::zmq::{
    MultipartMessage, ZmqSocket, connect_dealer_socket, connect_sub_socket,
};

const WATERMARK_UNSET: u64 = u64::MAX;

/// Longest silence tolerated from the engine's replay socket per reply: a ROUTER that accepts
/// the connection but never answers would otherwise park the rank's listener for good. Same
/// bound as the worker-side recovery listener's inactivity timeout.
const REPLAY_INACTIVITY_TIMEOUT: Duration = Duration::from_secs(5);

fn cursor_from_watermark(watermark: u64) -> CursorState {
    if watermark == WATERMARK_UNSET {
        CursorState::Initial
    } else {
        CursorState::Live(watermark)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReplayRecoveryFailureReason {
    Empty,
    StartedAfterRequested,
    NonContiguous,
    EndedBeforeTarget,
}

impl ReplayRecoveryFailureReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::StartedAfterRequested => "started_after_requested",
            Self::NonContiguous => "non_contiguous",
            Self::EndedBeforeTarget => "ended_before_target",
        }
    }
}

struct ReplayRecoveryProgress {
    start_seq: u64,
    end_seq: u64,
    first_replayed: Option<u64>,
    last_replayed: Option<u64>,
    expected_next: u64,
    replayed: u64,
    non_contiguous: bool,
}

impl ReplayRecoveryProgress {
    fn new(start_seq: u64, end_seq: u64) -> Self {
        Self {
            start_seq,
            end_seq,
            first_replayed: None,
            last_replayed: None,
            expected_next: start_seq,
            replayed: 0,
            non_contiguous: false,
        }
    }

    fn record_batch(&mut self, seq: u64) {
        if self.first_replayed.is_none() {
            self.first_replayed = Some(seq);
        }
        if seq != self.expected_next {
            self.non_contiguous = true;
        }
        self.expected_next = seq.saturating_add(1);
        self.last_replayed = Some(seq);
        self.replayed += 1;
    }

    fn failure_reason(&self) -> Option<ReplayRecoveryFailureReason> {
        if self.start_seq >= self.end_seq {
            return None;
        }
        if self.replayed == 0 {
            return Some(ReplayRecoveryFailureReason::Empty);
        }
        if self
            .first_replayed
            .is_some_and(|first| first > self.start_seq)
        {
            return Some(ReplayRecoveryFailureReason::StartedAfterRequested);
        }
        if self.non_contiguous {
            return Some(ReplayRecoveryFailureReason::NonContiguous);
        }
        if self
            .last_replayed
            .is_some_and(|last| last < self.end_seq - 1)
        {
            return Some(ReplayRecoveryFailureReason::EndedBeforeTarget);
        }
        None
    }

    fn replayed(&self) -> u64 {
        self.replayed
    }

    fn warn_if_incomplete(&self, worker_id: WorkerId, dp_rank: u32) {
        let Some(reason) = self.failure_reason() else {
            return;
        };

        let reason = reason.as_str();
        let start_seq = self.start_seq;
        let end_seq = self.end_seq;
        let replayed = self.replayed;
        let first_replayed_display = self
            .first_replayed
            .map(|seq| seq.to_string())
            .unwrap_or_else(|| "none".to_string());
        let last_replayed_display = self
            .last_replayed
            .map(|seq| seq.to_string())
            .unwrap_or_else(|| "none".to_string());
        tracing::warn!(
            worker_id,
            dp_rank,
            requested_start = start_seq,
            requested_end = end_seq,
            first_replayed = ?self.first_replayed,
            last_replayed = ?self.last_replayed,
            replayed,
            reason,
            "Replay incomplete: requested=[{start_seq},{end_seq}), first={}, last={}, replayed={replayed}, reason={reason}",
            first_replayed_display,
            last_replayed_display,
        );
    }
}

struct ListenerLoop {
    worker_id: WorkerId,
    dp_rank: u32,
    indexer: Indexer,
    cancel: CancellationToken,
    live_socket: ZmqSocket,
    /// Dialed per replay: an idle DEALER per worker rank would hold a ZMQ socket for each
    /// source for its whole life, while replays only happen at startup and on gaps.
    replay_endpoint: Option<String>,
    watermark: Arc<AtomicU64>,
    /// The live seq that opened the last replay. Live frames queued behind it arrive after
    /// the replay already applied them, so a stale seq at or above it is a duplicate; a stale
    /// seq below it (or any stale seq once the stream moved past the replay) is an engine
    /// that restarted its sequence.
    replay_trigger: Option<u64>,
    normalizer: ZmqEventNormalizer,
    messages_processed: u64,
}

impl ListenerLoop {
    #[expect(clippy::too_many_arguments)]
    fn new(
        worker_id: WorkerId,
        dp_rank: u32,
        block_size: u32,
        engine_owned_mm_identity: bool,
        indexer: Indexer,
        cancel: CancellationToken,
        live_socket: ZmqSocket,
        replay_endpoint: Option<String>,
        watermark: Arc<AtomicU64>,
    ) -> Self {
        Self {
            worker_id,
            dp_rank,
            indexer,
            cancel,
            live_socket,
            replay_endpoint,
            watermark,
            replay_trigger: None,
            normalizer: ZmqEventNormalizer::new(block_size)
                .with_engine_owned_mm_identity(engine_owned_mm_identity),
            messages_processed: 0,
        }
    }

    fn cursor(&self) -> CursorState {
        cursor_from_watermark(self.watermark.load(Ordering::Acquire))
    }

    async fn replay_gap(&mut self, start_seq: u64, end_seq: u64) -> Result<u64, String> {
        tracing::info!(
            self.worker_id,
            self.dp_rank,
            start_seq,
            end_seq,
            "Requesting replay from engine"
        );

        let Some(replay_endpoint) = self.replay_endpoint.as_deref() else {
            tracing::warn!(
                self.worker_id,
                self.dp_rank,
                gap_size = end_seq.saturating_sub(start_seq),
                "No replay endpoint configured; batches lost"
            );
            return Ok(0);
        };
        if self.cancel.is_cancelled() {
            return Ok(0);
        }
        let mut replay_socket = match connect_dealer_socket(replay_endpoint) {
            Ok(socket) => socket,
            Err(error) => {
                tracing::error!(
                    self.worker_id,
                    self.dp_rank,
                    error = %error,
                    "Failed to connect replay socket to {replay_endpoint}"
                );
                return Ok(0);
            }
        };

        let worker_id = self.worker_id;
        let dp_rank = self.dp_rank;
        let indexer = &self.indexer;
        let watermark = &self.watermark;

        let req_frames = vec![Vec::new(), start_seq.to_be_bytes().to_vec()];
        if let Err(error) = replay_socket.send_multipart(req_frames).await {
            tracing::error!(worker_id, dp_rank, error = %error, "Failed to send replay request");
            return Ok(0);
        }

        let mut replay_progress = ReplayRecoveryProgress::new(start_seq, end_seq);
        loop {
            // A silent peer ends the replay like a recv error: the gap is logged as incomplete
            // below and the listener goes back to the live socket.
            let msg = tokio::select! {
                _ = self.cancel.cancelled() => break,
                result = tokio::time::timeout(REPLAY_INACTIVITY_TIMEOUT, replay_socket.recv_multipart()) => {
                    match result {
                        Ok(Ok(msg)) => msg,
                        Ok(Err(error)) => {
                            tracing::error!(worker_id, dp_rank, error = %error, "Replay recv error");
                            break;
                        }
                        Err(_) => {
                            tracing::warn!(
                                worker_id,
                                dp_rank,
                                timeout_ms = REPLAY_INACTIVITY_TIMEOUT.as_millis() as u64,
                                "Replay peer silent; giving up on this gap"
                            );
                            break;
                        }
                    }
                }
            };
            // DEALER strips the ROUTER identity. vLLM includes the PUB topic;
            // SGLang sends only the delimiter, sequence number, and payload.
            let (seq_bytes, payload) = match msg.as_slice() {
                [_, seq, payload] | [_, _, seq, payload] => (seq, payload),
                _ => {
                    tracing::warn!(
                        worker_id,
                        dp_rank,
                        "Unexpected replay frame count: {}",
                        msg.len()
                    );
                    break;
                }
            };
            if payload.is_empty() {
                break;
            }

            if seq_bytes.len() != 8 {
                tracing::warn!(
                    worker_id,
                    dp_rank,
                    "Invalid replay seq length: {}",
                    seq_bytes.len()
                );
                break;
            }
            let seq = u64::from_be_bytes(seq_bytes[..8].try_into().expect("length checked above"));

            let Ok(batch) = decode_event_batch(payload) else {
                tracing::warn!(worker_id, dp_rank, seq, "Failed to decode replayed batch");
                continue;
            };

            let effective_dp_rank = batch
                .data_parallel_rank
                .map_or(dp_rank, |rank| rank.cast_unsigned());
            for raw_event in batch.events {
                let Some(placement_event) = self.normalizer.normalize(
                    raw_event,
                    seq,
                    WorkerWithDpRank::new(worker_id, effective_dp_rank),
                ) else {
                    continue;
                };
                let router_event = placement_event
                    .into_router_event()
                    .expect("local worker placement must convert to router event");
                indexer
                    .apply_event_routed(router_event)
                    .await
                    .map_err(|error| {
                        format!(
                            "failed to apply replayed event for worker {worker_id} dp_rank {dp_rank}: {error}"
                        )
                    })?;
            }
            watermark.store(seq, Ordering::Release);
            replay_progress.record_batch(seq);
        }

        replay_progress.warn_if_incomplete(worker_id, dp_rank);

        let replayed = replay_progress.replayed();
        tracing::info!(worker_id, dp_rank, replayed, "Replay complete");
        Ok(replayed)
    }

    async fn handle_gap(&mut self, seq: u64) -> Result<(), String> {
        match self.cursor().observe(seq) {
            CursorObservation::Initial { got } if got > 0 => {
                tracing::warn!(
                    self.worker_id,
                    self.dp_rank,
                    expected = 0,
                    got,
                    "Gap detected: expected seq 0, got {got}"
                );
                self.replay_trigger = Some(got);
                self.replay_gap(0, got).await?;
            }
            CursorObservation::Gap { expected, got } => {
                tracing::warn!(
                    self.worker_id,
                    self.dp_rank,
                    expected,
                    got,
                    "Gap detected: expected seq {expected}, got {got}"
                );
                self.replay_trigger = Some(got);
                self.replay_gap(expected, got).await?;
            }
            CursorObservation::Initial { .. }
            | CursorObservation::Contiguous { .. }
            | CursorObservation::Stale { .. } => {}
        }
        Ok(())
    }

    /// The engine behind this rank restarted its sequence: drop the previous epoch's coverage
    /// and start over as on a cold start, so the restarted engine's history is replayed.
    async fn restart_epoch(&mut self, seq: u64) -> Result<(), String> {
        tracing::warn!(
            self.worker_id,
            self.dp_rank,
            got = seq,
            watermark = self.watermark.load(Ordering::Acquire),
            "Sequence regression: engine restarted; clearing the rank's coverage"
        );
        let worker = WorkerWithDpRank::new(self.worker_id, self.dp_rank);
        // Through the normalizer so its per-worker state restarts with the index.
        let clear = self
            .normalizer
            .normalize(
                RawKvEvent::AllBlocksCleared { ownership: None },
                seq,
                worker,
            )
            .expect("synthetic clear event is always accepted")
            .into_router_event()
            .expect("local worker placement must convert to router event");
        self.indexer
            .apply_event_routed(clear)
            .await
            .map_err(|error| {
                format!(
                    "failed to clear worker {} dp_rank {} after a sequence regression: {error}",
                    self.worker_id, self.dp_rank
                )
            })?;
        self.watermark.store(WATERMARK_UNSET, Ordering::Release);
        self.handle_gap(seq).await
    }

    async fn apply_live_batch(&mut self, seq: u64, payload: &[u8]) -> Result<(), String> {
        let batch = match decode_event_batch(payload) {
            Ok(batch) => batch,
            Err(error) => {
                tracing::warn!(
                    self.worker_id,
                    self.dp_rank,
                    "Failed to decode KvEventBatch: {error}"
                );
                return Ok(());
            }
        };

        let effective_dp_rank = batch
            .data_parallel_rank
            .map_or(self.dp_rank, |rank| rank.cast_unsigned());
        for raw_event in batch.events {
            let Some(placement_event) = self.normalizer.normalize(
                raw_event,
                seq,
                WorkerWithDpRank::new(self.worker_id, effective_dp_rank),
            ) else {
                continue;
            };
            let router_event = placement_event
                .into_router_event()
                .expect("local worker placement must convert to router event");
            self.indexer
                .apply_event_routed(router_event)
                .await
                .map_err(|error| {
                    format!(
                        "failed to apply live event for worker {} dp_rank {}: {error}",
                        self.worker_id, self.dp_rank
                    )
                })?;
            self.messages_processed += 1;
        }
        self.watermark.store(seq, Ordering::Release);
        Ok(())
    }

    async fn handle_message(&mut self, msg: MultipartMessage) -> Result<(), String> {
        if msg.len() != 3 {
            tracing::warn!(
                self.worker_id,
                self.dp_rank,
                "Unexpected ZMQ frame count: {}",
                msg.len()
            );
            return Ok(());
        }

        let seq_bytes = msg.get(1).expect("frame count checked above");
        if seq_bytes.len() != 8 {
            tracing::warn!(
                self.worker_id,
                self.dp_rank,
                "Invalid sequence number length: {}",
                seq_bytes.len()
            );
            return Ok(());
        }

        let seq = u64::from_be_bytes(seq_bytes[..8].try_into().expect("length checked above"));
        if matches!(self.cursor().observe(seq), CursorObservation::Stale { .. }) {
            if self.replay_trigger.is_some_and(|trigger| seq >= trigger) {
                return Ok(());
            }
            self.restart_epoch(seq).await?;
        } else {
            self.replay_trigger = None;
            self.handle_gap(seq).await?;
        }

        // The replay (this batch's own, or the restarted epoch's) may have covered this seq.
        if matches!(self.cursor().observe(seq), CursorObservation::Stale { .. }) {
            return Ok(());
        }

        let payload = msg.get(2).expect("frame count checked above");
        self.apply_live_batch(seq, payload).await
    }

    async fn run(mut self) -> Result<(), String> {
        loop {
            let msg = tokio::select! {
                biased;

                _ = self.cancel.cancelled() => {
                    tracing::info!(
                        self.worker_id,
                        self.dp_rank,
                        self.messages_processed,
                        "ZMQ listener exiting after cancellation"
                    );
                    return Ok(());
                }

                result = self.live_socket.recv_multipart() => {
                    match result {
                        Ok(msg) => msg,
                        Err(error) => {
                            return Err(format!(
                                "ZMQ recv failed for worker {} dp_rank {}: {error}",
                                self.worker_id,
                                self.dp_rank,
                            ));
                        }
                    }
                }
            };

            self.handle_message(msg).await?;
        }
    }
}

/// Spawn one listener attempt; the handle lets its deregistration wait for the task to stop
/// admitting events before the worker's removal is issued.
pub fn spawn_zmq_listener(
    worker_id: WorkerId,
    dp_rank: u32,
    record: Arc<ListenerRecord>,
    ready: watch::Receiver<bool>,
    generation: u64,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        if let Err(error) = run_listener(
            worker_id,
            dp_rank,
            record.clone(),
            ready,
            generation,
            cancel,
        )
        .await
        {
            tracing::error!(worker_id, dp_rank, error = %error, "ZMQ listener failed");
            record.try_mark_failed(generation, error);
        }
    })
}

async fn run_listener(
    worker_id: WorkerId,
    dp_rank: u32,
    record: Arc<ListenerRecord>,
    mut ready: watch::Receiver<bool>,
    generation: u64,
    cancel: CancellationToken,
) -> Result<(), String> {
    let endpoint = record.endpoint().to_string();
    let replay_endpoint = record.replay_endpoint().map(str::to_string);
    let block_size = record.block_size();
    let engine_owned_mm_identity = record.engine_owned_mm_identity();
    let indexer = record.indexer();
    let watermark = record.watermark();

    tracing::info!(worker_id, dp_rank, endpoint, "ZMQ listener starting");

    if cancel.is_cancelled() {
        return Ok(());
    }

    let socket = connect_sub_socket(&endpoint)
        .map_err(|e| format!("failed to connect ZMQ SUB socket to {endpoint}: {e}"))?;

    tokio::select! {
        _ = cancel.cancelled() => return Ok(()),
        result = ready.wait_for(|&value| value) => {
            result.map_err(|_| "ready channel closed before signaling".to_string())?;
        }
    }

    if !record.try_mark_active(generation) {
        tracing::debug!(
            worker_id,
            dp_rank,
            "Listener attempt is stale after readiness gate; exiting"
        );
        return Ok(());
    }

    tracing::info!(worker_id, dp_rank, "ZMQ listener ready, starting recv loop");

    if cancel.is_cancelled() || !record.is_current_attempt(generation) {
        return Ok(());
    }

    ListenerLoop::new(
        worker_id,
        dp_rank,
        block_size,
        engine_owned_mm_identity,
        indexer,
        cancel,
        socket,
        replay_endpoint,
        watermark,
    )
    .run()
    .await
}

/// Engine-framed batch payloads for the listener and registry tests.
#[cfg(test)]
pub(super) mod test_frames {
    /// One batch holding a `BlockStored` of `token` repeated over a block of four, chained
    /// under `parent` when given, in the positional layout the engines publish.
    pub fn store_payload(block_hash: u64, parent: Option<u64>, token: u32) -> Vec<u8> {
        let event = (
            "BlockStored",
            vec![block_hash],
            parent,
            vec![token; 4],
            4_usize,
            Option::<u64>::None,
            "GPU",
        );
        rmp_serde::to_vec(&(0.0_f64, vec![event], Some(0_i32))).unwrap()
    }

    /// One batch holding an `AllBlocksCleared`.
    pub fn clear_payload() -> Vec<u8> {
        rmp_serde::to_vec(&(0.0_f64, vec![("AllBlocksCleared",)], Some(0_i32))).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::test_frames::{clear_payload, store_payload};
    use super::*;
    use crate::services::common::zmq::create_sub_socket;
    use crate::services::indexer::backend::create_indexer;
    use dynamo_kv_hashing::routing::compute_block_hash_for_seq;

    fn frame(seq: u64, payload: Vec<u8>) -> MultipartMessage {
        vec![b"kv-events".to_vec(), seq.to_be_bytes().to_vec(), payload]
    }

    async fn score(indexer: &Indexer, tokens: &[u32]) -> Option<u32> {
        let hashes = compute_block_hash_for_seq(tokens, 4, Default::default());
        indexer
            .find_matches(hashes)
            .await
            .unwrap()
            .scores
            .get(&WorkerWithDpRank::new(1, 0))
            .copied()
    }

    #[rstest::rstest]
    #[case::sglang(false)]
    #[case::vllm(true)]
    #[tokio::test]
    async fn replay_accepts_engine_framing(#[case] include_topic: bool) {
        let context = zmq::Context::new();
        let router = context.socket(zmq::ROUTER).unwrap();
        router.set_linger(0).unwrap();
        router.set_rcvtimeo(5000).unwrap();
        router.set_sndtimeo(5000).unwrap();
        router.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = router.get_last_endpoint().unwrap().unwrap();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        let publisher = context.socket(zmq::PUB).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let live_socket =
            connect_sub_socket(&publisher.get_last_endpoint().unwrap().unwrap()).unwrap();

        let server = tokio::task::spawn_blocking(move || {
            let request = router.recv_multipart(0).unwrap();
            assert_eq!(&request[1..], &[vec![], 0_u64.to_be_bytes().to_vec()]);
            for seq in 0_u64..=2 {
                let mut frames = vec![request[0].clone(), vec![]];
                if include_topic {
                    frames.push(b"kv-events".to_vec());
                }
                if seq == 2 {
                    frames.extend([u64::MAX.to_be_bytes().to_vec(), vec![]]);
                } else {
                    let event = (
                        "BlockStored",
                        vec![seq + 100],
                        if seq == 0 { None } else { Some(100_u64) },
                        vec![1_u32 + seq as u32; 4],
                        4_usize,
                        Option::<u64>::None,
                        "GPU",
                    );
                    let payload = rmp_serde::to_vec(&(0.0_f64, vec![event], Some(0_i32))).unwrap();
                    frames.extend([seq.to_be_bytes().to_vec(), payload]);
                }
                router.send_multipart(frames, 0).unwrap();
            }
            // Keep the socket open until the client consumes the queued replies.
            done_rx.recv().unwrap();
        });
        let indexer = create_indexer(4, 1);
        let watermark = Arc::new(AtomicU64::new(WATERMARK_UNSET));
        let cancel = CancellationToken::new();
        let mut listener = ListenerLoop::new(
            1,
            0,
            4,
            false,
            indexer.clone(),
            cancel.clone(),
            live_socket,
            Some(endpoint),
            watermark.clone(),
        );
        let replayed = tokio::time::timeout(Duration::from_secs(5), listener.replay_gap(0, 2))
            .await
            .expect("replay completion marker")
            .unwrap();
        done_tx.send(()).unwrap();
        server.await.unwrap();
        assert_eq!(replayed, 2);
        assert_eq!(watermark.load(Ordering::Acquire), 1);
        indexer.dump_events().await.expect("flush indexer");
        let hashes = compute_block_hash_for_seq(&[1, 1, 1, 1, 2, 2, 2, 2], 4, Default::default());
        let matches = indexer.find_matches(hashes).await.unwrap();
        assert_eq!(matches.scores.get(&WorkerWithDpRank::new(1, 0)), Some(&2));
        cancel.cancel();
    }

    /// A replay ROUTER that accepts the request and never answers must not park the
    /// rank's listener: the live SUB socket would go unread while the status stays
    /// `Active`. The peer runs on a plain thread so the paused clock still auto-advances.
    #[tokio::test]
    async fn replay_gives_up_on_a_silent_peer() {
        let context = zmq::Context::new();
        let router = context.socket(zmq::ROUTER).unwrap();
        router.set_linger(0).unwrap();
        router.set_rcvtimeo(5000).unwrap();
        router.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = router.get_last_endpoint().unwrap().unwrap();
        let (got_request_tx, got_request_rx) = tokio::sync::oneshot::channel::<()>();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        let server = std::thread::spawn(move || {
            if router.recv_multipart(0).is_ok() {
                let _ = got_request_tx.send(());
            }
            // Hold the connection open without ever replying.
            let _ = done_rx.recv();
        });
        let indexer = create_indexer(4, 1);
        let watermark = Arc::new(AtomicU64::new(WATERMARK_UNSET));
        let cancel = CancellationToken::new();
        let mut listener = ListenerLoop::new(
            1,
            0,
            4,
            false,
            indexer,
            cancel.clone(),
            create_sub_socket(b"").unwrap(),
            Some(endpoint),
            watermark.clone(),
        );

        let replay = listener.replay_gap(0, 2);
        tokio::pin!(replay);
        tokio::select! {
            result = &mut replay => panic!("replay finished before the peer received the request: {result:?}"),
            received = got_request_rx => received.expect("replay peer should receive the request"),
        }
        // The peer holds the request and stays silent; the wait is virtual from here on.
        tokio::time::pause();
        let replayed = tokio::time::timeout(2 * REPLAY_INACTIVITY_TIMEOUT, replay)
            .await
            .expect("a silent replay peer must not park the listener")
            .unwrap();
        done_tx.send(()).unwrap();
        server.join().unwrap();
        assert_eq!(replayed, 0);
        assert_eq!(watermark.load(Ordering::Acquire), WATERMARK_UNSET);
        cancel.cancel();
    }

    /// An engine that restarts behind an unchanged registration counts from 0 again. Its
    /// batches must be indexed as a new epoch and the dead engine's blocks must go.
    #[tokio::test]
    async fn sequence_regression_starts_a_new_epoch() {
        let indexer = create_indexer(4, 1);
        let watermark = Arc::new(AtomicU64::new(WATERMARK_UNSET));
        let cancel = CancellationToken::new();
        let mut listener = ListenerLoop::new(
            1,
            0,
            4,
            false,
            indexer.clone(),
            cancel.clone(),
            create_sub_socket(b"").unwrap(),
            None,
            watermark.clone(),
        );

        listener
            .handle_message(frame(0, store_payload(100, None, 1)))
            .await
            .unwrap();
        listener
            .handle_message(frame(1, store_payload(101, Some(100), 2)))
            .await
            .unwrap();
        // The engine restarts: its publisher counts from 0 again and clears first.
        listener
            .handle_message(frame(0, clear_payload()))
            .await
            .unwrap();
        listener
            .handle_message(frame(1, store_payload(200, None, 3)))
            .await
            .unwrap();
        indexer.flush().await;

        assert_eq!(score(&indexer, &[3, 3, 3, 3]).await, Some(1));
        assert_eq!(score(&indexer, &[1, 1, 1, 1, 2, 2, 2, 2]).await, None);
        assert_eq!(watermark.load(Ordering::Acquire), 1);
        cancel.cancel();
    }

    /// Live frames queued behind the batch that opened a gap arrive after the replay already
    /// applied them. They are duplicates, not a restart: dropped, with the index untouched.
    #[tokio::test]
    async fn live_frames_the_replay_covered_are_dropped_not_reset() {
        let context = zmq::Context::new();
        let router = context.socket(zmq::ROUTER).unwrap();
        router.set_linger(0).unwrap();
        router.set_rcvtimeo(5000).unwrap();
        router.set_sndtimeo(5000).unwrap();
        router.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = router.get_last_endpoint().unwrap().unwrap();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        let server = std::thread::spawn(move || {
            let request = router.recv_multipart(0).unwrap();
            assert_eq!(&request[1..], &[vec![], 0_u64.to_be_bytes().to_vec()]);
            // Asked for [0, 2), the engine replays everything it has: seq 0 through 3.
            for seq in 0_u64..=3 {
                let mut frames = vec![request[0].clone(), vec![]];
                frames.extend([
                    seq.to_be_bytes().to_vec(),
                    store_payload(100 + seq, None, 10 + seq as u32),
                ]);
                router.send_multipart(frames, 0).unwrap();
            }
            router
                .send_multipart(
                    vec![
                        request[0].clone(),
                        vec![],
                        u64::MAX.to_be_bytes().to_vec(),
                        vec![],
                    ],
                    0,
                )
                .unwrap();
            let _ = done_rx.recv();
        });
        let indexer = create_indexer(4, 1);
        let watermark = Arc::new(AtomicU64::new(WATERMARK_UNSET));
        let cancel = CancellationToken::new();
        let mut listener = ListenerLoop::new(
            1,
            0,
            4,
            false,
            indexer.clone(),
            cancel.clone(),
            create_sub_socket(b"").unwrap(),
            Some(endpoint),
            watermark.clone(),
        );

        // Seq 2 opens the gap; the replay overshoots to 3, so the live 2 and 3 are duplicates
        // and seq 4 continues the stream.
        for (seq, payload) in [
            (2, store_payload(900, None, 20)),
            (3, store_payload(901, None, 21)),
            (4, store_payload(104, None, 14)),
        ] {
            tokio::time::timeout(
                Duration::from_secs(5),
                listener.handle_message(frame(seq, payload)),
            )
            .await
            .expect("replay completion marker")
            .unwrap();
        }
        done_tx.send(()).unwrap();
        server.join().unwrap();
        indexer.flush().await;

        for token in 10..=14 {
            assert_eq!(score(&indexer, &[token; 4]).await, Some(1), "token {token}");
        }
        assert_eq!(score(&indexer, &[20; 4]).await, None);
        assert_eq!(score(&indexer, &[21; 4]).await, None);
        assert_eq!(watermark.load(Ordering::Acquire), 4);
        cancel.cancel();
    }
}
