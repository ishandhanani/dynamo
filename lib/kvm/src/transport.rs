// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How a hint reaches the engine scheduler.
//!
//! [`ZmqHintTransport`] is a ZMQ DEALER per scheduler endpoint connected to the
//! scheduler's inbound ROUTER (`--hint-zmq-endpoint` in SGLang). Each request is one
//! [`wire`](crate::wire) frame; the ROUTER strips the DEALER identity before the
//! handler sees it and addresses the reply back to the same identity, so a DEALER
//! receives the bare reply frame. A REQ-style peer may prepend an empty delimiter, so
//! decoding takes the last non-empty frame. Requests and replies are matched by `rid`.
//!
//! A hint is only useful now: the socket's send high-water mark is the in-flight
//! bound and `ZMQ_IMMEDIATE` is set, so a hint for a scheduler that is not connected
//! fails instead of queueing in ZMQ and bursting out stale on reconnect. A connection
//! that sees no traffic for [`IDLE_TIMEOUTS`] reply timeouts closes itself, and the
//! host calls [`ZmqHintTransport::forget`] when a worker leaves; the next hint for an
//! endpoint opens a fresh socket.
//!
//! [`ZmqStorePinSink`] carries pins to the shared store instead: a ZMQ PUSH to the
//! Mooncake master's `--hint_zmq_endpoint`, one [`StorePin`](crate::wire::StorePin)
//! frame per message and nothing back. Sends never block: a full or unconnected
//! socket fails the pin, which the controller counts and drops.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use tmq::{FromZmqSocket, Multipart};
use tokio::sync::{Mutex, mpsc, oneshot};

use crate::PinHint;
use crate::WorkerTarget;
use crate::wire::{
    PrefetchReqInput, STORE_PIN_MAX_KEYS, StorePin, decode_prefetch_reply, encode_prefetch_request,
    encode_store_pin,
};

/// Milliseconds a closing socket may linger; hints are advisory, never wait on them.
const ZMQ_LINGER_MS: i32 = 0;
/// Reply timeouts without any send or receive after which a connection closes.
pub const IDLE_TIMEOUTS: u32 = 8;

/// A successful scheduler answer to one prefetch.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefetchReply {
    pub rid: String,
    pub matched_tokens: u64,
    pub loaded_tokens: u64,
    pub elapsed_ms: f64,
}

/// Delivers one prefetch to the engine and waits for its reply. Implementations are
/// driven off the request path by [`HintController`](crate::HintController).
#[async_trait::async_trait]
pub trait HintTransport: Send + Sync {
    fn name(&self) -> &'static str;

    /// Whether `worker` can be reached at all; the controller skips hints for
    /// workers that cannot, before they take a queue slot.
    fn has_endpoint(&self, worker: WorkerTarget) -> bool {
        let _ = worker;
        true
    }

    async fn prefetch(
        &self,
        worker: WorkerTarget,
        tokens: Arc<[u32]>,
    ) -> anyhow::Result<PrefetchReply>;
}

/// Maps a worker instance id to its scheduler's hint endpoint (`tcp://host:port`),
/// or `None` when the worker publishes none.
pub type HintEndpointResolver = Arc<dyn Fn(u64) -> Option<String> + Send + Sync>;

/// Where a worker's hints go.
#[derive(Clone)]
pub enum HintEndpoints {
    /// Each worker's own published scheduler endpoint. The production path.
    PerWorker(HintEndpointResolver),
    /// One scheduler endpoint for every worker, whatever it publishes. A development
    /// override for single-worker setups whose runtime config lacks the endpoint.
    Fixed(String),
}

impl HintEndpoints {
    fn resolve(&self, worker_id: u64) -> Option<String> {
        match self {
            Self::PerWorker(resolver) => resolver(worker_id),
            Self::Fixed(endpoint) => Some(endpoint.clone()),
        }
    }
}

type ReplySender = oneshot::Sender<anyhow::Result<PrefetchReply>>;

struct Outbound {
    rid: String,
    frame: Vec<u8>,
    reply: ReplySender,
}

/// One lazily opened DEALER per endpoint. Each connection holds at most
/// `max_in_flight` unanswered requests, and a reply later than `timeout` is a failure.
pub struct ZmqHintTransport {
    endpoints: HintEndpoints,
    context: zmq::Context,
    connections: Mutex<HashMap<String, mpsc::Sender<Outbound>>>,
    timeout: Duration,
    max_in_flight: usize,
    next_rid: AtomicU64,
}

impl ZmqHintTransport {
    pub fn new(endpoints: HintEndpoints, timeout: Duration, max_in_flight: usize) -> Self {
        Self {
            endpoints,
            context: zmq::Context::new(),
            connections: Mutex::new(HashMap::new()),
            timeout,
            max_in_flight: max_in_flight.max(1),
            next_rid: AtomicU64::new(0),
        }
    }

    /// Close the connection to `endpoint`, if open. Call when the worker behind it
    /// is gone; a later hint for the same endpoint opens a new socket.
    pub async fn forget(&self, endpoint: &str) {
        if self.connections.lock().await.remove(endpoint).is_some() {
            tracing::debug!(endpoint, "closing KV hint socket: worker removed");
        }
    }

    /// Endpoints with an open connection task.
    pub async fn open_endpoints(&self) -> Vec<String> {
        let mut connections = self.connections.lock().await;
        connections.retain(|_, tx| !tx.is_closed());
        connections.keys().cloned().collect()
    }

    async fn connection(&self, endpoint: &str) -> anyhow::Result<mpsc::Sender<Outbound>> {
        let mut connections = self.connections.lock().await;
        if let Some(tx) = connections.get(endpoint)
            && !tx.is_closed()
        {
            return Ok(tx.clone());
        }
        let socket = self.context.socket(zmq::DEALER)?;
        socket.set_linger(ZMQ_LINGER_MS)?;
        // Hints are only useful now: bound what ZMQ may buffer to the in-flight
        // window and refuse to queue for a peer that is not connected.
        socket.set_sndhwm(self.max_in_flight as i32)?;
        socket.set_immediate(true)?;
        socket.connect(endpoint)?;
        let dealer = tmq::dealer::Dealer::from_zmq_socket(socket)?;
        let (tx, rx) = mpsc::channel(self.max_in_flight);
        tokio::spawn(run_connection(
            endpoint.to_string(),
            dealer,
            rx,
            self.timeout,
            self.max_in_flight,
        ));
        connections.insert(endpoint.to_string(), tx.clone());
        tracing::info!(endpoint, "opened KV hint socket");
        Ok(tx)
    }
}

#[async_trait::async_trait]
impl HintTransport for ZmqHintTransport {
    fn name(&self) -> &'static str {
        "zmq-scheduler"
    }

    fn has_endpoint(&self, worker: WorkerTarget) -> bool {
        self.endpoints.resolve(worker.worker_id).is_some()
    }

    async fn prefetch(
        &self,
        worker: WorkerTarget,
        tokens: Arc<[u32]>,
    ) -> anyhow::Result<PrefetchReply> {
        let Some(endpoint) = self.endpoints.resolve(worker.worker_id) else {
            anyhow::bail!("worker {} publishes no hint endpoint", worker.worker_id);
        };
        let rid = format!(
            "kvm-{}-{}",
            std::process::id(),
            self.next_rid.fetch_add(1, Ordering::Relaxed)
        );
        let frame = encode_prefetch_request(&PrefetchReqInput::new(
            rid.clone(),
            tokens.to_vec(),
            worker.dp_rank,
        ))?;
        let (reply_tx, reply_rx) = oneshot::channel();
        let outbound = Outbound {
            rid,
            frame,
            reply: reply_tx,
        };
        let connection = self.connection(&endpoint).await?;
        if connection.try_send(outbound).is_err() {
            anyhow::bail!("hint connection to {endpoint} is saturated or closed");
        }
        match tokio::time::timeout(self.timeout, reply_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => anyhow::bail!("hint connection to {endpoint} dropped the request"),
            Err(_) => anyhow::bail!(
                "no prefetch reply from {endpoint} within {} ms",
                self.timeout.as_millis()
            ),
        }
    }
}

async fn run_connection(
    endpoint: String,
    dealer: tmq::dealer::Dealer,
    mut rx: mpsc::Receiver<Outbound>,
    timeout: Duration,
    max_in_flight: usize,
) {
    let (mut sink, mut stream) = dealer.split();
    let mut pending: HashMap<String, ReplySender> = HashMap::new();
    let mut idle = tokio::time::interval(timeout);
    idle.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut idle_ticks: u32 = 0;
    let stop_reason = loop {
        tokio::select! {
            outbound = rx.recv() => {
                let Some(outbound) = outbound else { break Ok("closed by owner") };
                idle_ticks = 0;
                // Waiters that timed out free their slot right away.
                pending.retain(|_, tx| !tx.is_closed());
                if pending.len() >= max_in_flight {
                    let _ = outbound.reply.send(Err(anyhow::anyhow!(
                        "{max_in_flight} prefetches already in flight to {endpoint}"
                    )));
                    continue;
                }
                match sink.send(Multipart::from(vec![outbound.frame])).await {
                    Ok(()) => {
                        pending.insert(outbound.rid, outbound.reply);
                    }
                    Err(error) => {
                        let _ = outbound.reply.send(Err(anyhow::anyhow!("zmq send failed: {error}")));
                    }
                }
            }
            message = stream.next() => {
                let Some(message) = message else { break Ok("stream ended") };
                idle_ticks = 0;
                let multipart = match message {
                    Ok(multipart) => multipart,
                    Err(error) => break Err(error),
                };
                let Some(frame) = multipart.0.iter().rev().find(|frame| !frame.is_empty()) else {
                    continue;
                };
                let reply = match decode_prefetch_reply(frame) {
                    Ok(reply) => reply,
                    Err(error) => {
                        tracing::warn!(endpoint, %error, "undecodable prefetch reply");
                        continue;
                    }
                };
                let Some(tx) = pending.remove(&reply.rid) else {
                    tracing::debug!(endpoint, rid = %reply.rid, "late or unknown prefetch reply");
                    continue;
                };
                let result = match reply.error {
                    Some(error) => Err(anyhow::anyhow!("scheduler rejected prefetch: {error}")),
                    None => Ok(PrefetchReply {
                        rid: reply.rid,
                        matched_tokens: reply.matched_tokens,
                        loaded_tokens: reply.loaded_tokens,
                        elapsed_ms: reply.elapsed_ms,
                    }),
                };
                let _ = tx.send(result);
            }
            _ = idle.tick() => {
                pending.retain(|_, tx| !tx.is_closed());
                idle_ticks = idle_ticks.saturating_add(1);
                if pending.is_empty() && idle_ticks >= IDLE_TIMEOUTS {
                    break Ok("idle");
                }
            }
        }
    };
    // Dropping `rx` here marks the connection closed; the next hint reopens it.
    match stop_reason {
        Ok(reason) => tracing::debug!(endpoint, reason, "KV hint socket closed"),
        Err(error) => tracing::warn!(endpoint, %error, "KV hint socket receive failed; closed"),
    }
}

/// Where pins go: the shared store's hint socket.
#[async_trait::async_trait]
pub trait PinSink: Send + Sync {
    fn name(&self) -> &'static str;

    /// Deliver one pin; `Ok` means the store's socket took every frame.
    async fn pin(&self, hint: &PinHint) -> anyhow::Result<()>;
}

/// [`PinSink`] over a ZMQ PUSH to the Mooncake master's `--hint_zmq_endpoint`.
pub struct ZmqStorePinSink {
    endpoint: String,
    // zmq::Socket is Send but not Sync; a pin is a few non-blocking sends.
    socket: std::sync::Mutex<Option<zmq::Socket>>,
    context: zmq::Context,
    send_hwm: i32,
}

impl ZmqStorePinSink {
    /// `send_hwm` bounds the frames ZMQ buffers for a slow or absent store.
    pub fn new(endpoint: impl Into<String>, send_hwm: i32) -> Self {
        Self {
            endpoint: endpoint.into(),
            socket: std::sync::Mutex::new(None),
            context: zmq::Context::new(),
            send_hwm,
        }
    }

    fn open(&self) -> anyhow::Result<zmq::Socket> {
        let socket = self.context.socket(zmq::PUSH)?;
        socket.set_linger(ZMQ_LINGER_MS)?;
        socket.set_sndhwm(self.send_hwm)?;
        socket.set_immediate(true)?;
        socket.connect(&self.endpoint)?;
        Ok(socket)
    }
}

#[async_trait::async_trait]
impl PinSink for ZmqStorePinSink {
    fn name(&self) -> &'static str {
        "zmq-store-pin"
    }

    async fn pin(&self, hint: &PinHint) -> anyhow::Result<()> {
        let ttl_ms = u64::try_from(hint.ttl.as_millis()).unwrap_or(u64::MAX);
        let mut guard = self.socket.lock().unwrap();
        if guard.is_none() {
            *guard = Some(self.open()?);
        }
        let socket = guard.as_ref().unwrap();
        for keys in hint.keys.chunks(STORE_PIN_MAX_KEYS) {
            let frame = encode_store_pin(&StorePin::new(
                hint.tenant.clone(),
                keys.to_vec(),
                ttl_ms,
                hint.priority,
            ))?;
            socket.send(frame, zmq::DONTWAIT)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::wire::{PREFETCH_REPLY_TAG, PREFETCH_REQUEST_TAG, PrefetchReqOutput};

    fn free_tcp_endpoint() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        format!("tcp://127.0.0.1:{}", listener.local_addr().unwrap().port())
    }

    fn per_worker(endpoint: &str) -> HintEndpoints {
        let endpoint = endpoint.to_string();
        HintEndpoints::PerWorker(Arc::new(move |worker_id| {
            (worker_id == 7).then(|| endpoint.clone())
        }))
    }

    type Seen = Arc<std::sync::Mutex<Vec<(String, Vec<u32>, Option<u32>)>>>;

    #[derive(Clone, Copy)]
    enum Scheduler {
        Answer,
        Reject,
        Mute,
    }

    /// A stand-in scheduler: ROUTER that decodes the request and answers per `mode`.
    fn serve_router(endpoint: &str, mode: Scheduler) -> Seen {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let router = tmq::router(&tmq::Context::new()).bind(endpoint).unwrap();
        let seen_task = Arc::clone(&seen);
        tokio::spawn(async move {
            let (mut sink, mut stream) = router.split();
            while let Some(Ok(mut multipart)) = stream.next().await {
                let identity = multipart.pop_front().unwrap();
                let frame = multipart.pop_back().unwrap();
                let request: PrefetchReqInput = rmp_serde::from_slice(&frame).unwrap();
                assert_eq!(request.tag, PREFETCH_REQUEST_TAG);
                assert_eq!(request.http_worker_ipc, None);
                seen_task.lock().unwrap().push((
                    request.rid.clone(),
                    request.token_ids.clone(),
                    request.dp_rank,
                ));
                let reply = match mode {
                    Scheduler::Mute => continue,
                    Scheduler::Answer => PrefetchReqOutput {
                        tag: PREFETCH_REPLY_TAG.to_string(),
                        rid: request.rid,
                        http_worker_ipc: None,
                        matched_tokens: request.token_ids.len() as u64,
                        loaded_tokens: (request.token_ids.len() / 2) as u64,
                        elapsed_ms: 1.5,
                        error: None,
                    },
                    Scheduler::Reject => PrefetchReqOutput {
                        tag: PREFETCH_REPLY_TAG.to_string(),
                        rid: request.rid,
                        http_worker_ipc: None,
                        matched_tokens: 0,
                        loaded_tokens: 0,
                        elapsed_ms: 0.0,
                        error: Some("no external linker".to_string()),
                    },
                };
                let reply = rmp_serde::to_vec(&reply).unwrap();
                sink.send(Multipart::from(vec![identity.to_vec(), reply]))
                    .await
                    .unwrap();
            }
        });
        seen
    }

    fn tokens(ids: &[u32]) -> Arc<[u32]> {
        Arc::from(ids)
    }

    #[tokio::test]
    async fn prefetch_round_trips_msgpack_frames_to_the_scheduler_socket() {
        let endpoint = free_tcp_endpoint();
        let seen = serve_router(&endpoint, Scheduler::Answer);
        let transport = ZmqHintTransport::new(per_worker(&endpoint), Duration::from_secs(5), 4);

        let reply = transport
            .prefetch(WorkerTarget::new(7, Some(2)), tokens(&[1, 2, 3, 4]))
            .await
            .unwrap();
        assert_eq!(reply.matched_tokens, 4);
        assert_eq!(reply.loaded_tokens, 2);
        assert_eq!(reply.elapsed_ms, 1.5);
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0, reply.rid);
        assert_eq!(seen[0].1, vec![1, 2, 3, 4]);
        assert_eq!(seen[0].2, Some(2), "dp_rank rides the 5th element");
    }

    #[tokio::test]
    async fn fixed_endpoint_serves_every_worker() {
        let endpoint = free_tcp_endpoint();
        let seen = serve_router(&endpoint, Scheduler::Answer);
        let transport =
            ZmqHintTransport::new(HintEndpoints::Fixed(endpoint), Duration::from_secs(5), 4);
        assert!(transport.has_endpoint(WorkerTarget::new(9, None)));
        let reply = transport
            .prefetch(WorkerTarget::new(9, None), tokens(&[1, 2]))
            .await
            .unwrap();
        assert_eq!(reply.matched_tokens, 2);
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn scheduler_rejection_reaches_the_caller_before_the_timeout() {
        let endpoint = free_tcp_endpoint();
        let _seen = serve_router(&endpoint, Scheduler::Reject);
        let transport = ZmqHintTransport::new(per_worker(&endpoint), Duration::from_secs(5), 4);

        let started = Instant::now();
        let error = transport
            .prefetch(WorkerTarget::new(7, None), tokens(&[1]))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("no external linker"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[tokio::test]
    async fn unknown_worker_and_silent_scheduler_fail_without_blocking() {
        let endpoint = free_tcp_endpoint();
        let _seen = serve_router(&endpoint, Scheduler::Mute);
        let transport = ZmqHintTransport::new(per_worker(&endpoint), Duration::from_millis(200), 1);

        assert!(!transport.has_endpoint(WorkerTarget::new(9, None)));
        let error = transport
            .prefetch(WorkerTarget::new(9, None), tokens(&[1]))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("publishes no hint endpoint"));

        let started = Instant::now();
        let error = transport
            .prefetch(WorkerTarget::new(7, None), tokens(&[1]))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("no prefetch reply"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(2));

        // The timed-out waiter no longer occupies the single in-flight slot.
        let error = transport
            .prefetch(WorkerTarget::new(7, None), tokens(&[2]))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("no prefetch reply"), "{error}");
    }

    #[tokio::test]
    async fn forget_closes_the_connection_and_the_next_hint_reopens_it() {
        let endpoint = free_tcp_endpoint();
        let seen = serve_router(&endpoint, Scheduler::Answer);
        let transport = ZmqHintTransport::new(per_worker(&endpoint), Duration::from_secs(5), 4);

        transport
            .prefetch(WorkerTarget::new(7, None), tokens(&[1]))
            .await
            .unwrap();
        assert_eq!(transport.open_endpoints().await, vec![endpoint.clone()]);

        transport.forget(&endpoint).await;
        assert!(transport.open_endpoints().await.is_empty());

        transport
            .prefetch(WorkerTarget::new(7, None), tokens(&[2]))
            .await
            .unwrap();
        assert_eq!(transport.open_endpoints().await, vec![endpoint]);
        assert_eq!(seen.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn an_idle_connection_closes_itself() {
        let endpoint = free_tcp_endpoint();
        let _seen = serve_router(&endpoint, Scheduler::Answer);
        let timeout = Duration::from_millis(50);
        let transport = ZmqHintTransport::new(per_worker(&endpoint), timeout, 4);

        transport
            .prefetch(WorkerTarget::new(7, None), tokens(&[1]))
            .await
            .unwrap();
        assert_eq!(transport.open_endpoints().await.len(), 1);

        tokio::time::sleep(timeout * (IDLE_TIMEOUTS + 2)).await;
        assert!(
            transport.open_endpoints().await.is_empty(),
            "idle connection should have closed"
        );
    }

    #[tokio::test]
    async fn store_pin_sink_pushes_chunked_frames() {
        let endpoint = free_tcp_endpoint();
        let context = zmq::Context::new();
        let pull = context.socket(zmq::PULL).unwrap();
        pull.bind(&endpoint).unwrap();
        pull.set_rcvtimeo(5_000).unwrap();
        let sink = ZmqStorePinSink::new(endpoint, 1_000);
        let keys: Vec<String> = (0..STORE_PIN_MAX_KEYS + 1)
            .map(|i| format!("k{i}"))
            .collect();
        let hint = PinHint {
            tenant: "default".into(),
            keys: keys.clone().into(),
            ttl: Duration::from_millis(90_000),
            priority: 2,
        };
        // ZMQ_IMMEDIATE refuses frames until the connection is up.
        let deadline = Instant::now() + Duration::from_secs(5);
        while sink.pin(&hint).await.is_err() {
            assert!(Instant::now() < deadline, "store socket never connected");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let mut seen = Vec::new();
        for _ in 0..2 {
            let frame = pull.recv_bytes(0).unwrap();
            let pin: StorePin = rmp_serde::from_slice(&frame).unwrap();
            assert_eq!(
                (pin.tag.as_str(), pin.tenant_id.as_str()),
                ("Pin", "default")
            );
            assert_eq!((pin.ttl_ms, pin.priority), (90_000, 2));
            seen.extend(pin.keys);
        }
        assert_eq!(seen, keys);
    }
}
