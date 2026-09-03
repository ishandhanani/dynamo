// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline vocabulary plus the runtime's network transports. The
//! runtime-free core lives in `dynamo-pipeline` and is re-exported here at
//! its historical paths.

pub use dynamo_pipeline::pipeline::*;

/// `Sync` ownership cell for a non-`Sync` [`DataStream<T>`]. [`Self::take`]
/// moves the inner stream out; the mutex serialises concurrent attempts so
/// the first caller observes `Some(stream)` and all later callers see
/// `None`. Iteration happens on the returned `DataStream<T>`.
pub struct RequestStream<T: Data> {
    inner: std::sync::Mutex<Option<DataStream<T>>>,
}

impl<T: Data> RequestStream<T> {
    /// Wrap a [`DataStream<T>`] in a `Sync` ownership cell.
    pub fn new(stream: DataStream<T>) -> Self {
        Self {
            inner: std::sync::Mutex::new(Some(stream)),
        }
    }

    /// Atomically move the inner stream out. Returns `Some(stream)` exactly
    /// once across all threads racing on the same `RequestStream`; every
    /// subsequent call (on any thread) returns `None`. The returned stream
    /// is the unique owner; the wrapper retains nothing.
    pub fn take(&self) -> Option<DataStream<T>> {
        self.inner.lock().unwrap().take()
    }
}

impl<T: Data> std::fmt::Debug for RequestStream<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let taken = self.inner.lock().map(|g| g.is_none()).unwrap_or(true);
        f.debug_struct("RequestStream")
            .field("taken", &taken)
            .finish()
    }
}

/// Pipeline input for streaming-request engines: a [`RequestStream<T>`]
/// payload wrapped in a [`Context`], symmetric to [`SingleIn<T>`] for unary
/// inputs.
pub type ManyIn<T> = Context<RequestStream<T>>;

/// `ClientStreaming` Engine is a pipeline that takes multiple inputs and returns a single output
/// Typically the engine will consume the entire input stream; however, it can also decided to exit
/// early and emit a response without consuming the entire input stream.
pub type ClientStreamingEngine<T, U> = ServiceEngine<ManyIn<T>, SingleOut<U>>;

/// `BidirectionalStreaming` takes multiple inputs and returns multiple outputs. Input and output values
/// are considered independent of each other; however, they could be constrained to be related.
pub type BidirectionalStreamingEngine<T, U> = ServiceEngine<ManyIn<T>, ManyOut<U>>;

pub mod network;

pub use crate::routing_policy::{
    BuiltinRoutePicker, OccupancyReservation, OccupancySelection, RouteTarget,
    RoutingOccupancyState,
};
pub use network::egress::addressed_router::{
    AddressedPushRouter, AddressedRequest, StreamingDispatch, attach_first_response_guard,
    propagate_first_response_guard,
};
pub use network::egress::push_router::{
    MultimodalCacheIndex, MultimodalCacheKeyExtractor, PushRouter, RouterMode, WorkerLoadMonitor,
};
