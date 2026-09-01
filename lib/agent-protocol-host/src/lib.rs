// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal in-process contract between Dynamo and an external agent runtime.
//!
//! Dynamo implements inference, cancellation, routing, and public protocol
//! encoding. Runtime crates implement [`ProtocolInterceptor`] without taking a
//! dependency on Dynamo's frontend, router, engine, or Kubernetes internals.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use dynamo_protocols::types::anthropic::{
    AnthropicCreateMessageRequest, AnthropicMessageResponse, AnthropicStreamEvent, AnthropicUsage,
};
use dynamo_protocols::types::responses::{CreateResponse, Response, ResponseStreamEvent};
use futures::Stream;
use thiserror::Error;

pub type ProtocolFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A stream after public response headers have been committed.
///
/// Failures at this point must be represented by a native terminal event.
pub type ProtocolStream<T> = Pin<Box<dyn Stream<Item = T> + Send + 'static>>;

pub enum ProtocolOutput<T, E> {
    Unary(T),
    Streaming(ProtocolStream<E>),
}

pub type ResponsesProtocolOutput = ProtocolOutput<Response, ResponseStreamEvent>;
/// One typed Anthropic event plus Dynamo's wire-only usage extension.
///
/// `AnthropicStreamEvent::ContentBlockDelta` cannot represent the cumulative
/// usage Dynamo emits on token deltas, so the sidecar must survive runtime
/// observation until Dynamo serializes the public SSE frame.
#[derive(Debug, Clone)]
pub struct AnthropicStreamFrame {
    event: AnthropicStreamEvent,
    usage: Option<AnthropicUsage>,
}

impl AnthropicStreamFrame {
    pub fn new(event: AnthropicStreamEvent, usage: Option<AnthropicUsage>) -> Self {
        Self { event, usage }
    }

    pub fn event(&self) -> &AnthropicStreamEvent {
        &self.event
    }

    pub fn usage(&self) -> Option<&AnthropicUsage> {
        self.usage.as_ref()
    }

    pub fn into_event(self) -> AnthropicStreamEvent {
        self.event
    }
}

pub type AnthropicProtocolOutput = ProtocolOutput<AnthropicMessageResponse, AnthropicStreamFrame>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtocolExtensionErrorKind {
    BadRequest,
    Unauthorized,
    Forbidden,
    NotFound,
    Conflict,
    RateLimited,
    NotImplemented,
    Overloaded,
    Unavailable,
    Cancelled,
    Internal,
}

#[derive(Debug, Error)]
#[error("{message}")]
pub struct ProtocolExtensionError {
    kind: ProtocolExtensionErrorKind,
    message: String,
}

impl ProtocolExtensionError {
    pub fn new(kind: ProtocolExtensionErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn kind(&self) -> ProtocolExtensionErrorKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Dynamo-owned cancellation signal for one public request.
pub trait RequestCancellation: Send + Sync + 'static {
    fn is_cancelled(&self) -> bool;
    fn cancelled(&self) -> ProtocolFuture<'static, ()>;
}

/// Credential-free ingress state supplied to a runtime.
#[derive(Clone)]
pub struct ProtocolRequestContext {
    request_id: Arc<str>,
    idempotency_key: Option<Arc<str>>,
    cancellation: Arc<dyn RequestCancellation>,
}

impl ProtocolRequestContext {
    pub fn new(
        request_id: impl Into<Arc<str>>,
        idempotency_key: Option<Arc<str>>,
        cancellation: Arc<dyn RequestCancellation>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            idempotency_key,
            cancellation,
        }
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn idempotency_key(&self) -> Option<&str> {
        self.idempotency_key.as_deref()
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    pub fn cancelled(&self) -> ProtocolFuture<'static, ()> {
        self.cancellation.cancelled()
    }
}

type ProtocolInferenceFn<Request, Output> = dyn Fn(Request) -> ProtocolFuture<'static, Result<Output, ProtocolExtensionError>>
    + Send
    + Sync;

pub struct ProtocolInference<Request, Output> {
    invoke: Arc<ProtocolInferenceFn<Request, Output>>,
}

impl<Request, Output> Clone for ProtocolInference<Request, Output> {
    fn clone(&self) -> Self {
        Self {
            invoke: self.invoke.clone(),
        }
    }
}

impl<Request, Output> ProtocolInference<Request, Output>
where
    Request: 'static,
    Output: 'static,
{
    pub fn new<F>(invoke: F) -> Self
    where
        F: Fn(Request) -> ProtocolFuture<'static, Result<Output, ProtocolExtensionError>>
            + Send
            + Sync
            + 'static,
    {
        Self {
            invoke: Arc::new(invoke),
        }
    }

    pub fn infer(
        &self,
        request: Request,
    ) -> ProtocolFuture<'static, Result<Output, ProtocolExtensionError>> {
        (self.invoke)(request)
    }
}

pub type ResponsesInference = ProtocolInference<CreateResponse, ResponsesProtocolOutput>;
pub type AnthropicInference =
    ProtocolInference<AnthropicCreateMessageRequest, AnthropicProtocolOutput>;

/// Deployment-owned protocol runtime registered on Dynamo's HTTP service.
pub trait ProtocolInterceptor: Send + Sync + 'static {
    fn intercept_responses(
        &self,
        _request: &CreateResponse,
        _context: &ProtocolRequestContext,
    ) -> bool {
        false
    }

    fn intercept_anthropic(
        &self,
        _request: &AnthropicCreateMessageRequest,
        _context: &ProtocolRequestContext,
    ) -> bool {
        false
    }

    fn responses<'a>(
        &'a self,
        _request: CreateResponse,
        _context: ProtocolRequestContext,
        _inference: ResponsesInference,
    ) -> ProtocolFuture<'a, Result<ResponsesProtocolOutput, ProtocolExtensionError>> {
        Box::pin(async {
            Err(ProtocolExtensionError::new(
                ProtocolExtensionErrorKind::Internal,
                "Responses interception is not implemented",
            ))
        })
    }

    fn anthropic<'a>(
        &'a self,
        _request: AnthropicCreateMessageRequest,
        _context: ProtocolRequestContext,
        _inference: AnthropicInference,
    ) -> ProtocolFuture<'a, Result<AnthropicProtocolOutput, ProtocolExtensionError>> {
        Box::pin(async {
            Err(ProtocolExtensionError::new(
                ProtocolExtensionErrorKind::Internal,
                "Anthropic interception is not implemented",
            ))
        })
    }
}
