// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV cache hints from Dynamo to inference engine schedulers.
//!
//! [`HintController::prefetch`] asks a worker to load a token prefix's KV pages from
//! the shared store (L3) into device cache before the request reaches it. The call
//! enqueues onto a bounded channel and returns; a background task drives a
//! [`HintTransport`] with bounded concurrency. A full queue drops the hint with a
//! warning. Nothing here runs on a request's hot path.
//!
//! The production transport is [`ZmqHintTransport`]: a ZMQ DEALER per worker that
//! talks straight to the SGLang scheduler's inbound hint socket, the inverse of the
//! KV-event stream. The frame layout is in [`wire`].
//!
//! [`HintController::pin`] asks the shared store to keep a prefix's pages resident
//! for a while (for example while its session waits on a tool call), lower priority
//! evicted first; a pin for pages not written yet applies when they are. Pins have
//! their own queue and go through a [`PinSink`], [`ZmqStorePinSink`] in production.
//! Like prefetch, a pin returns at once and a full queue drops it.
//!
//! This crate reads no configuration itself: the host builds a [`HintConfig`] and a
//! [`HintEndpoints`] and passes them in.

pub mod transport;
pub mod wire;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::{Semaphore, mpsc};

pub use transport::{
    HintEndpointResolver, HintEndpoints, HintTransport, PinSink, PrefetchReply, ZmqHintTransport,
    ZmqStorePinSink,
};

/// Warn on the first dropped hint, then once per this many drops.
const DROP_WARN_EVERY: u64 = 1000;

/// The worker a prefetch hint targets. `worker_id` is the Dynamo instance id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerTarget {
    pub worker_id: u64,
    pub dp_rank: Option<u32>,
}

impl WorkerTarget {
    pub fn new(worker_id: u64, dp_rank: Option<u32>) -> Self {
        Self { worker_id, dp_rank }
    }
}

/// Bounds on the hint pipeline; see [`HintConfig::default`] for the defaults.
#[derive(Debug, Clone)]
pub struct HintConfig {
    /// Hints waiting for a sender slot before new ones are dropped.
    pub queue_depth: usize,
    /// Concurrent in-flight deliveries, also the per-connection pending bound.
    pub concurrency: usize,
    /// How long to wait for a scheduler reply before counting a hint as failed.
    pub timeout: Duration,
}

impl Default for HintConfig {
    fn default() -> Self {
        Self {
            queue_depth: 256,
            concurrency: 4,
            timeout: Duration::from_millis(2_000),
        }
    }
}

/// Hint outcome counters; read with [`HintController::counters`].
#[derive(Debug, Default)]
pub struct HintCounters {
    pub prefetch_sent: AtomicU64,
    pub prefetch_failed: AtomicU64,
    /// Hints that found the queue full.
    pub dropped: AtomicU64,
    /// Hints for a worker the transport cannot reach (no published endpoint).
    pub skipped_no_endpoint: AtomicU64,
    pub pin_sent: AtomicU64,
    pub pin_failed: AtomicU64,
    /// Pins that found their queue full.
    pub pin_dropped: AtomicU64,
    /// Pins on a controller built without a [`PinSink`].
    pub pin_skipped_no_sink: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HintCounterSnapshot {
    pub prefetch_sent: u64,
    pub prefetch_failed: u64,
    pub dropped: u64,
    pub skipped_no_endpoint: u64,
    pub pin_sent: u64,
    pub pin_failed: u64,
    pub pin_dropped: u64,
    pub pin_skipped_no_sink: u64,
}

struct Prefetch {
    worker: WorkerTarget,
    tokens: Arc<[u32]>,
}

/// Keep `keys` resident in the shared store for `ttl`; see [`HintController::pin`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinHint {
    pub tenant: String,
    /// Store object keys, in prefix order.
    pub keys: Arc<[String]>,
    pub ttl: Duration,
    /// Higher is kept longer when pinned pages must be evicted.
    pub priority: u32,
}

/// Fire-and-forget hint sender. Cheap to clone; all clones share one queue and task.
#[derive(Clone)]
pub struct HintController {
    tx: mpsc::Sender<Prefetch>,
    transport: Arc<dyn HintTransport>,
    config: Arc<HintConfig>,
    counters: Arc<HintCounters>,
    pins: Option<mpsc::Sender<PinHint>>,
}

impl HintController {
    /// Start the background sender on the current Tokio runtime.
    ///
    /// Panics outside a runtime context, like `tokio::spawn`.
    pub fn new(config: HintConfig, transport: Arc<dyn HintTransport>) -> Self {
        let (tx, rx) = mpsc::channel(config.queue_depth.max(1));
        let config = Arc::new(config);
        let counters = Arc::new(HintCounters::default());
        tokio::spawn(run_sender(
            rx,
            Arc::clone(&transport),
            config.concurrency.max(1),
            Arc::clone(&counters),
        ));
        Self {
            tx,
            transport,
            config,
            counters,
            pins: None,
        }
    }

    /// Send pins through `sink` on a queue of their own, `queue_depth` deep.
    ///
    /// Panics outside a runtime context, like `tokio::spawn`.
    pub fn with_pin_sink(mut self, sink: Arc<dyn PinSink>) -> Self {
        let (tx, rx) = mpsc::channel(self.config.queue_depth.max(1));
        tokio::spawn(run_pin_sender(rx, sink, Arc::clone(&self.counters)));
        self.pins = Some(tx);
        self
    }

    /// Ask the shared store to keep `hint.keys` resident for `hint.ttl`. Returns at
    /// once; the call never waits on the store.
    pub fn pin(&self, hint: PinHint) {
        if hint.keys.is_empty() {
            return;
        }
        let Some(pins) = &self.pins else {
            self.counters
                .pin_skipped_no_sink
                .fetch_add(1, Ordering::Relaxed);
            return;
        };
        tracing::debug!(
            keys = hint.keys.len(),
            ttl_ms = hint.ttl.as_millis() as u64,
            priority = hint.priority,
            "enqueue pin hint"
        );
        if pins.try_send(hint).is_err() {
            let dropped = self.counters.pin_dropped.fetch_add(1, Ordering::Relaxed) + 1;
            if dropped == 1 || dropped.is_multiple_of(DROP_WARN_EVERY) {
                tracing::warn!(
                    dropped_total = dropped,
                    queue_depth = self.config.queue_depth,
                    "KV pin queue full; dropping pin hint"
                );
            }
        }
    }

    /// Ask `worker` to load the KV pages for `tokens` from the shared store.
    pub fn prefetch(&self, worker: WorkerTarget, tokens: Arc<[u32]>) {
        if tokens.is_empty() {
            return;
        }
        if !self.transport.has_endpoint(worker) {
            self.counters
                .skipped_no_endpoint
                .fetch_add(1, Ordering::Relaxed);
            tracing::debug!(
                worker_id = worker.worker_id,
                "prefetch hint skipped: worker publishes no hint endpoint"
            );
            return;
        }
        tracing::debug!(
            worker_id = worker.worker_id,
            dp_rank = ?worker.dp_rank,
            tokens = tokens.len(),
            "enqueue prefetch hint"
        );
        if self.tx.try_send(Prefetch { worker, tokens }).is_err() {
            let dropped = self.counters.dropped.fetch_add(1, Ordering::Relaxed) + 1;
            if dropped == 1 || dropped.is_multiple_of(DROP_WARN_EVERY) {
                tracing::warn!(
                    dropped_total = dropped,
                    queue_depth = self.config.queue_depth,
                    "KV hint queue full; dropping prefetch hint"
                );
            }
        }
    }

    pub fn counters(&self) -> HintCounterSnapshot {
        HintCounterSnapshot {
            prefetch_sent: self.counters.prefetch_sent.load(Ordering::Relaxed),
            prefetch_failed: self.counters.prefetch_failed.load(Ordering::Relaxed),
            dropped: self.counters.dropped.load(Ordering::Relaxed),
            skipped_no_endpoint: self.counters.skipped_no_endpoint.load(Ordering::Relaxed),
            pin_sent: self.counters.pin_sent.load(Ordering::Relaxed),
            pin_failed: self.counters.pin_failed.load(Ordering::Relaxed),
            pin_dropped: self.counters.pin_dropped.load(Ordering::Relaxed),
            pin_skipped_no_sink: self.counters.pin_skipped_no_sink.load(Ordering::Relaxed),
        }
    }
}

/// Pins go one at a time: each is a few non-blocking socket sends.
async fn run_pin_sender(
    mut rx: mpsc::Receiver<PinHint>,
    sink: Arc<dyn PinSink>,
    counters: Arc<HintCounters>,
) {
    while let Some(hint) = rx.recv().await {
        match sink.pin(&hint).await {
            Ok(()) => {
                counters.pin_sent.fetch_add(1, Ordering::Relaxed);
                tracing::debug!(keys = hint.keys.len(), "pin hint delivered");
            }
            Err(error) => {
                let failed = counters.pin_failed.fetch_add(1, Ordering::Relaxed) + 1;
                if failed == 1 || failed.is_multiple_of(DROP_WARN_EVERY) {
                    tracing::warn!(sink = sink.name(), %error, failed_total = failed, "pin hint failed");
                }
            }
        }
    }
}

async fn run_sender(
    mut rx: mpsc::Receiver<Prefetch>,
    transport: Arc<dyn HintTransport>,
    concurrency: usize,
    counters: Arc<HintCounters>,
) {
    let permits = Arc::new(Semaphore::new(concurrency));
    while let Some(hint) = rx.recv().await {
        let Ok(permit) = Arc::clone(&permits).acquire_owned().await else {
            break;
        };
        let transport = Arc::clone(&transport);
        let counters = Arc::clone(&counters);
        tokio::spawn(async move {
            match transport.prefetch(hint.worker, hint.tokens).await {
                Ok(reply) => {
                    counters.prefetch_sent.fetch_add(1, Ordering::Relaxed);
                    tracing::debug!(
                        worker_id = hint.worker.worker_id,
                        matched_tokens = reply.matched_tokens,
                        loaded_tokens = reply.loaded_tokens,
                        elapsed_ms = reply.elapsed_ms,
                        "prefetch hint delivered"
                    );
                }
                Err(error) => {
                    counters.prefetch_failed.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!(
                        worker_id = hint.worker.worker_id,
                        transport = transport.name(),
                        %error,
                        "prefetch hint failed"
                    );
                }
            }
            drop(permit);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Recording {
        prefetch: Mutex<Vec<(WorkerTarget, Vec<u32>)>>,
        hang: bool,
        unreachable_worker: Option<u64>,
    }

    #[async_trait::async_trait]
    impl HintTransport for Recording {
        fn name(&self) -> &'static str {
            "recording"
        }

        fn has_endpoint(&self, worker: WorkerTarget) -> bool {
            self.unreachable_worker != Some(worker.worker_id)
        }

        async fn prefetch(
            &self,
            worker: WorkerTarget,
            tokens: Arc<[u32]>,
        ) -> anyhow::Result<PrefetchReply> {
            let count = tokens.len() as u64;
            self.prefetch
                .lock()
                .unwrap()
                .push((worker, tokens.to_vec()));
            if self.hang {
                std::future::pending::<()>().await;
            }
            Ok(PrefetchReply {
                rid: String::new(),
                matched_tokens: count,
                loaded_tokens: count,
                elapsed_ms: 0.0,
            })
        }
    }

    /// Yield until `check` holds; the paused clock advances only when every task idles.
    async fn wait_until(mut check: impl FnMut() -> bool) {
        for _ in 0..200 {
            if check() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("condition not met");
    }

    fn tokens(ids: &[u32]) -> Arc<[u32]> {
        Arc::from(ids)
    }

    #[tokio::test(start_paused = true)]
    async fn hints_reach_the_transport_with_their_arguments() {
        let transport = Arc::new(Recording {
            unreachable_worker: Some(8),
            ..Recording::default()
        });
        let controller = HintController::new(HintConfig::default(), transport.clone());

        controller.prefetch(WorkerTarget::new(7, Some(0)), tokens(&[1, 2, 3, 4]));
        // Empty hints never leave the caller.
        controller.prefetch(WorkerTarget::new(7, None), tokens(&[]));
        // Neither do hints for a worker the transport cannot reach.
        controller.prefetch(WorkerTarget::new(8, None), tokens(&[9]));

        wait_until(|| controller.counters().prefetch_sent == 1).await;
        assert_eq!(
            *transport.prefetch.lock().unwrap(),
            vec![(WorkerTarget::new(7, Some(0)), vec![1, 2, 3, 4])]
        );
        let counters = controller.counters();
        assert_eq!(counters.dropped, 0);
        assert_eq!(counters.skipped_no_endpoint, 1);
    }

    #[tokio::test(start_paused = true)]
    async fn full_queue_drops_with_counter() {
        let transport = Arc::new(Recording {
            hang: true,
            ..Recording::default()
        });
        let controller = HintController::new(
            HintConfig {
                queue_depth: 1,
                concurrency: 1,
                ..HintConfig::default()
            },
            transport.clone(),
        );
        // First hint: taken by the sender and parked in the hanging transport.
        controller.prefetch(WorkerTarget::new(1, None), tokens(&[1]));
        wait_until(|| transport.prefetch.lock().unwrap().len() == 1).await;
        // Second hint: received by the sender, which then parks on the single permit.
        controller.prefetch(WorkerTarget::new(1, None), tokens(&[2]));
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;
        // Third hint fills the one queue slot; the fourth finds it full and drops.
        controller.prefetch(WorkerTarget::new(1, None), tokens(&[3]));
        controller.prefetch(WorkerTarget::new(1, None), tokens(&[4]));
        assert_eq!(controller.counters().dropped, 1);
        assert_eq!(transport.prefetch.lock().unwrap().len(), 1);
    }

    #[derive(Default)]
    struct RecordingPins {
        pins: Mutex<Vec<PinHint>>,
    }

    #[async_trait::async_trait]
    impl PinSink for RecordingPins {
        fn name(&self) -> &'static str {
            "recording-pins"
        }

        async fn pin(&self, hint: &PinHint) -> anyhow::Result<()> {
            self.pins.lock().unwrap().push(hint.clone());
            Ok(())
        }
    }

    fn pin_hint(keys: &[&str]) -> PinHint {
        PinHint {
            tenant: "default".into(),
            keys: keys
                .iter()
                .map(|k| k.to_string())
                .collect::<Vec<_>>()
                .into(),
            ttl: Duration::from_secs(90),
            priority: 1,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn pins_reach_the_sink_and_need_one() {
        let transport = Arc::new(Recording::default());
        // Without a sink a pin is counted and goes nowhere.
        let bare = HintController::new(HintConfig::default(), transport.clone());
        bare.pin(pin_hint(&["a"]));
        assert_eq!(bare.counters().pin_skipped_no_sink, 1);

        let sink = Arc::new(RecordingPins::default());
        let controller =
            HintController::new(HintConfig::default(), transport).with_pin_sink(sink.clone());
        controller.pin(pin_hint(&["a", "b"]));
        // Empty pins never leave the caller.
        controller.pin(pin_hint(&[]));
        wait_until(|| controller.counters().pin_sent == 1).await;
        assert_eq!(*sink.pins.lock().unwrap(), vec![pin_hint(&["a", "b"])]);
        assert_eq!(controller.counters().prefetch_sent, 0);
    }
}
