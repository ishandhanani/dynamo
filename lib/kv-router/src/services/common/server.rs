// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serving a standalone service's router on the shared Dynamo HTTP server.

use std::future::Future;
use std::time::Duration;

use axum::Router;
use dynamo_http_server::listener::{
    bind_listener, get_graceful_shutdown_timeout, listen_socket_addr, write_idle_timeout,
};
use dynamo_http_server::server::serve;

/// Serve `app` on all interfaces at `port` until `shutdown` resolves, then stop accepting
/// and give open connections the configured grace period
/// (`DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS`) to finish before they are closed.
pub(crate) async fn serve_until(
    port: u16,
    app: Router,
    shutdown: impl Future<Output = ()>,
) -> anyhow::Result<()> {
    let listener = bind_listener(listen_socket_addr("0.0.0.0", port)?)?;
    tracing::info!(address = %listener.local_addr()?, "HTTP server listening");
    serve(listener, None, app, write_idle_timeout(), async move {
        shutdown.await;
        tracing::info!("Received shutdown signal, stopping HTTP server");
        Duration::from_secs(get_graceful_shutdown_timeout() as u64)
    })
    .await;
    Ok(())
}
