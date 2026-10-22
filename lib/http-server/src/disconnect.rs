// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client-disconnect monitoring for streaming responses.
//!
//! A request holds two [`ConnectionHandle`]s, one for the handler task and one for the
//! response body; when either is dropped without being disarmed, the monitor task kills
//! the engine context so the backend stops generating for a client that is gone.

use std::sync::Arc;

use dynamo_runtime_protocols::engine::AsyncEngineContext;

/// Callback run when a connection or stream closes unexpectedly.
pub type UnexpectedCloseHook = Arc<dyn Fn() + Send + Sync>;

#[derive(Clone, Copy)]
pub enum ConnectionStatus {
    Disabled,
    ClosedUnexpectedly,
    ClosedGracefully,
}

pub struct ConnectionHandle {
    sender: Option<tokio::sync::oneshot::Sender<ConnectionStatus>>,
    on_drop: ConnectionStatus,
}

impl ConnectionHandle {
    /// Handle which by default will issue a [`ConnectionStatus::ClosedGracefully`] signal when dropped.
    pub fn create_disarmed(sender: tokio::sync::oneshot::Sender<ConnectionStatus>) -> Self {
        Self {
            sender: Some(sender),
            on_drop: ConnectionStatus::ClosedGracefully,
        }
    }

    /// Handle which will issue a [`ConnectionStatus::ClosedUnexpectedly`] signal when dropped.
    pub fn create_armed(sender: tokio::sync::oneshot::Sender<ConnectionStatus>) -> Self {
        Self {
            sender: Some(sender),
            on_drop: ConnectionStatus::ClosedUnexpectedly,
        }
    }

    /// Handle which will not issue a signal when dropped.
    pub fn create_disabled(sender: tokio::sync::oneshot::Sender<ConnectionStatus>) -> Self {
        Self {
            sender: Some(sender),
            on_drop: ConnectionStatus::Disabled,
        }
    }

    /// Handle which will issue a [`ConnectionStatus::ClosedGracefully`] signal when dropped.
    pub fn disarm(&mut self) {
        self.on_drop = ConnectionStatus::ClosedGracefully;
    }

    /// Handle which will issue a [`ConnectionStatus::ClosedUnexpectedly`] signal when dropped.
    pub fn arm(&mut self) {
        self.on_drop = ConnectionStatus::ClosedUnexpectedly;
    }
}

impl Drop for ConnectionHandle {
    fn drop(&mut self) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(self.on_drop);
        }
    }
}

/// Creates a pair of handles which will monitor for disconnects from the client.
///
/// The first handle is armed and will issue a [`ConnectionStatus::ClosedUnexpectedly`] signal when dropped.
/// The second handle is disarmed and will issue a [`ConnectionStatus::ClosedGracefully`] signal when dropped.
///
/// The handles are returned in the order of the first being armed and the second being disarmed.
/// `on_unexpected_close` runs when either scope closes without a graceful signal, right
/// before the engine context is killed; the frontend uses it to count disconnects.
pub async fn create_connection_monitor(
    engine_context: Arc<dyn AsyncEngineContext>,
    on_unexpected_close: Option<UnexpectedCloseHook>,
) -> (ConnectionHandle, ConnectionHandle) {
    // these oneshot channels monitor possible disconnects from the client in two different scopes:
    // - the local task (connection_handle)
    // - an optionally streaming response (stream_handle)
    let (connection_tx, connection_rx) = tokio::sync::oneshot::channel();
    let (stream_tx, stream_rx) = tokio::sync::oneshot::channel();

    // detached task that will naturally close when both handles are dropped
    tokio::spawn(connection_monitor(
        engine_context.clone(),
        connection_rx,
        stream_rx,
        on_unexpected_close,
    ));

    // Two handles, the first is armed, the second is disarmed
    (
        ConnectionHandle::create_armed(connection_tx),
        ConnectionHandle::create_disabled(stream_tx),
    )
}

/// The monitor task behind [`create_connection_monitor`]: waits on both scopes in turn and
/// kills `engine_context` on an unexpected close.
#[tracing::instrument(level = "trace", skip_all, fields(request_id = %engine_context.id()))]
pub async fn connection_monitor(
    engine_context: Arc<dyn AsyncEngineContext>,
    connection_rx: tokio::sync::oneshot::Receiver<ConnectionStatus>,
    stream_rx: tokio::sync::oneshot::Receiver<ConnectionStatus>,
    on_unexpected_close: Option<UnexpectedCloseHook>,
) {
    match connection_rx.await {
        Err(_) | Ok(ConnectionStatus::ClosedUnexpectedly) => {
            // the client has disconnected, no need to gracefully cancel, just kill the context
            tracing::warn!("Connection closed unexpectedly; issuing cancellation");
            if let Some(hook) = &on_unexpected_close {
                hook();
            }
            engine_context.kill();
        }
        Ok(ConnectionStatus::ClosedGracefully) => {
            tracing::trace!("Connection closed gracefully");
        }
        Ok(ConnectionStatus::Disabled) => {}
    }

    match stream_rx.await {
        Err(_) | Ok(ConnectionStatus::ClosedUnexpectedly) => {
            tracing::warn!("Stream closed unexpectedly; issuing cancellation");
            if let Some(hook) = &on_unexpected_close {
                hook();
            }
            engine_context.kill();
        }
        Ok(ConnectionStatus::ClosedGracefully) => {
            tracing::trace!("Stream closed gracefully");
        }
        Ok(ConnectionStatus::Disabled) => {}
    }
}
