// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The hint sockets (feature `zmq`). Every send is non-blocking and its failure only logged:
//! a hint never delays a request.
//!
//! - Prefetch: a PUSH per SGLang scheduler hint socket (`--hint-zmq-endpoint`), carrying a
//!   [`KvHint`] envelope as msgpack (a map, as SGLang's `KvHintsEnvelope` decodes it).
//! - Pin: a PUSH to the Mooncake master's `--hint_zmq_endpoint`, carrying
//!   `["Pin", tenant, [key, ...], ttl_ms, priority]`.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use crate::KvHint;

/// The store drops pin messages with more keys than this.
const MAX_PIN_KEYS: usize = 4096;
const SEND_HWM: i32 = 1_000;

pub struct HintSender {
    context: zmq::Context,
    schedulers: Mutex<HashMap<String, zmq::Socket>>,
    store: Option<Mutex<zmq::Socket>>,
}

impl HintSender {
    /// Pins go to `store_endpoint` when set.
    pub fn new(store_endpoint: Option<&str>) -> Self {
        let context = zmq::Context::new();
        let store = store_endpoint.and_then(|endpoint| {
            open(&context, zmq::PUSH, endpoint)
                .inspect_err(|error| tracing::warn!(endpoint, %error, "pin socket not opened"))
                .ok()
                .map(Mutex::new)
        });
        Self {
            context,
            schedulers: Mutex::new(HashMap::new()),
            store,
        }
    }

    /// Send `hint` to the scheduler hint socket at `endpoint`.
    pub fn send(&self, endpoint: &str, hint: &KvHint) {
        let mut schedulers = self.schedulers.lock().unwrap();
        if !schedulers.contains_key(endpoint) {
            match open(&self.context, zmq::PUSH, endpoint) {
                Ok(socket) => schedulers.insert(endpoint.to_string(), socket),
                Err(error) => return tracing::debug!(endpoint, %error, "hint not sent"),
            };
        }
        send_frame(
            &schedulers[endpoint],
            rmp_serde::to_vec_named(hint),
            "prefetch",
        );
    }

    /// Ask the store to keep `keys` resident for `ttl`, lower `priority` evicted first.
    pub fn pin(&self, tenant: &str, keys: &[String], ttl: Duration, priority: u32) {
        let Some(store) = &self.store else { return };
        let store = store.lock().unwrap();
        for keys in keys.chunks(MAX_PIN_KEYS) {
            let frame = ("Pin", tenant, keys, ttl.as_millis() as u64, priority);
            send_frame(&store, rmp_serde::to_vec(&frame), "pin");
        }
    }
}

fn open(context: &zmq::Context, kind: zmq::SocketType, endpoint: &str) -> zmq::Result<zmq::Socket> {
    let socket = context.socket(kind)?;
    socket.set_linger(0)?;
    socket.set_sndhwm(SEND_HWM)?;
    socket.connect(endpoint)?;
    Ok(socket)
}

fn send_frame(socket: &zmq::Socket, frame: Result<Vec<u8>, rmp_serde::encode::Error>, hint: &str) {
    let result = frame
        .map_err(|error| error.to_string())
        .and_then(|bytes| socket.send(bytes, zmq::DONTWAIT).map_err(|e| e.to_string()));
    if let Err(error) = result {
        tracing::debug!(hint, %error, "hint dropped");
    }
}
