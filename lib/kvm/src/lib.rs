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
//! Replies also feed a per-worker [`load::LoadModel`], so the host can ask
//! [`HintController::load_estimate_ms`] how long a hint of a given size will take to
//! land and decide whether a waiting request can still benefit from it.
//!
//! This crate reads no configuration itself: the host builds a [`HintConfig`] and a
//! [`HintEndpoints`] and passes them in.

pub mod load;
pub mod transport;
pub mod wire;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::{Semaphore, mpsc};

pub use load::{LoadFit, LoadModel};
pub use transport::{
    HintEndpointResolver, HintEndpoints, HintTransport, PrefetchReply, ZmqHintTransport,
};

/// Warn on the first dropped hint, then once per this many drops.
const DROP_WARN_EVERY: u64 = 1000;

/// The worker a prefetch hint targets. `worker_id` is the Dynamo instance id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    /// Tokens per KV page on the engine, the unit the load model counts in. Hosts pass
    /// the router's block size, which must equal the engine's page size.
    pub page_tokens: usize,
}

impl Default for HintConfig {
    fn default() -> Self {
        Self {
            queue_depth: 256,
            concurrency: 4,
            timeout: Duration::from_millis(2_000),
            page_tokens: 64,
        }
    }
}

/// Sees every successful scheduler reply, off the request path, after the load model
/// has learned from it. For metrics; implementations must not block.
pub trait HintReplyObserver: Send + Sync {
    /// `hint_tokens` is the length of the hint that was sent.
    fn on_reply(&self, worker: WorkerTarget, hint_tokens: usize, reply: &PrefetchReply);
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HintCounterSnapshot {
    pub prefetch_sent: u64,
    pub prefetch_failed: u64,
    pub dropped: u64,
    pub skipped_no_endpoint: u64,
}

struct Prefetch {
    worker: WorkerTarget,
    tokens: Arc<[u32]>,
}

/// Fire-and-forget hint sender. Cheap to clone; all clones share one queue and task.
#[derive(Clone)]
pub struct HintController {
    tx: mpsc::Sender<Prefetch>,
    transport: Arc<dyn HintTransport>,
    config: Arc<HintConfig>,
    counters: Arc<HintCounters>,
    loads: Arc<LoadModel>,
}

impl HintController {
    /// Start the background sender on the current Tokio runtime.
    ///
    /// Panics outside a runtime context, like `tokio::spawn`.
    pub fn new(config: HintConfig, transport: Arc<dyn HintTransport>) -> Self {
        Self::with_reply_observer(config, transport, None)
    }

    /// [`Self::new`], with `observer` told about every successful reply.
    pub fn with_reply_observer(
        config: HintConfig,
        transport: Arc<dyn HintTransport>,
        observer: Option<Arc<dyn HintReplyObserver>>,
    ) -> Self {
        let (tx, rx) = mpsc::channel(config.queue_depth.max(1));
        let config = Arc::new(config);
        let counters = Arc::new(HintCounters::default());
        let loads = Arc::new(LoadModel::new(config.page_tokens));
        tokio::spawn(run_sender(
            rx,
            Arc::clone(&transport),
            config.concurrency.max(1),
            Arc::clone(&counters),
            Arc::clone(&loads),
            observer,
        ));
        Self {
            tx,
            transport,
            config,
            counters,
            loads,
        }
    }

    /// Expected milliseconds for `worker` to load a `tokens`-long hint from the shared
    /// store: its fitted `L(n) = a + b * n` over pages, or the prior while it has fewer
    /// than [`load::MIN_SAMPLES`] replies.
    pub fn load_estimate_ms(&self, worker: WorkerTarget, tokens: usize) -> f64 {
        self.loads.estimate_ms(worker, tokens)
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
        }
    }
}

async fn run_sender(
    mut rx: mpsc::Receiver<Prefetch>,
    transport: Arc<dyn HintTransport>,
    concurrency: usize,
    counters: Arc<HintCounters>,
    loads: Arc<LoadModel>,
    observer: Option<Arc<dyn HintReplyObserver>>,
) {
    let permits = Arc::new(Semaphore::new(concurrency));
    while let Some(hint) = rx.recv().await {
        let Ok(permit) = Arc::clone(&permits).acquire_owned().await else {
            break;
        };
        let transport = Arc::clone(&transport);
        let counters = Arc::clone(&counters);
        let loads = Arc::clone(&loads);
        let observer = observer.clone();
        tokio::spawn(async move {
            let hint_tokens = hint.tokens.len();
            match transport.prefetch(hint.worker, hint.tokens).await {
                Ok(reply) => {
                    counters.prefetch_sent.fetch_add(1, Ordering::Relaxed);
                    loads.observe(hint.worker, reply.loaded_tokens, reply.elapsed_ms);
                    if let Some(observer) = &observer {
                        observer.on_reply(hint.worker, hint_tokens, &reply);
                    }
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
        /// Reported load time per reply.
        elapsed_ms: f64,
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
                elapsed_ms: self.elapsed_ms,
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

    #[tokio::test(start_paused = true)]
    async fn replies_train_the_load_model_and_reach_the_observer() {
        struct Replies(Mutex<Vec<(WorkerTarget, usize, u64)>>);
        impl HintReplyObserver for Replies {
            fn on_reply(&self, worker: WorkerTarget, hint_tokens: usize, reply: &PrefetchReply) {
                self.0
                    .lock()
                    .unwrap()
                    .push((worker, hint_tokens, reply.loaded_tokens));
            }
        }
        let transport = Arc::new(Recording {
            elapsed_ms: 400.0,
            ..Recording::default()
        });
        let replies = Arc::new(Replies(Mutex::new(Vec::new())));
        let controller = HintController::with_reply_observer(
            HintConfig {
                page_tokens: 16,
                ..HintConfig::default()
            },
            transport.clone(),
            Some(replies.clone()),
        );
        let worker = WorkerTarget::new(7, Some(0));
        let hint: Arc<[u32]> = Arc::from(vec![0u32; 128 * 16]);
        let prior = controller.load_estimate_ms(worker, hint.len());
        assert_eq!(prior, load::LoadFit::PRIOR.estimate_ms(128));

        for _ in 0..3 {
            controller.prefetch(worker, Arc::clone(&hint));
        }
        wait_until(|| controller.counters().prefetch_sent == 3).await;
        assert_eq!(replies.0.lock().unwrap().len(), 3);
        assert_eq!(replies.0.lock().unwrap()[0], (worker, 128 * 16, 128 * 16));
        // Three 400 ms replies for 128 pages pull the estimate well above the prior.
        let learned = controller.load_estimate_ms(worker, hint.len());
        assert!(learned > prior, "{learned} <= {prior}");
        // Another worker still answers with the prior.
        assert_eq!(
            controller.load_estimate_ms(WorkerTarget::new(8, Some(0)), hint.len()),
            prior
        );
    }
}
