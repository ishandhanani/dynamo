// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Listener setup and the environment-driven knobs of the HTTP server: bind address
//! parsing, the accept backlog, the write-idle timeout, and the graceful-shutdown budget.

use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

use anyhow::Result;
use dynamo_runtime_protocols::environment_names::llm as env_llm;

/// Seconds the server waits for inflight responses and open connections at shutdown
/// (`DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS`, default 5).
pub fn get_graceful_shutdown_timeout() -> usize {
    std::env::var(env_llm::DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(5)
}

/// How long a connection may stall mid-write before it is closed
/// (`DYN_HTTP_WRITE_IDLE_TIMEOUT_SECS`, default 120s).
pub fn write_idle_timeout() -> Duration {
    std::env::var(env_llm::DYN_HTTP_WRITE_IDLE_TIMEOUT_SECS)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .map(Duration::from_secs)
        .filter(|duration| tokio::time::Instant::now().checked_add(*duration).is_some())
        .unwrap_or(Duration::from_secs(120))
}

const DEFAULT_LISTEN_BACKLOG: u32 = 4096;

// `listen(2)` takes an `int`; anything above `i32::MAX` would go negative.
fn parse_listen_backlog(value: Result<String, std::env::VarError>) -> u32 {
    value
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|n| (1..=i32::MAX as u32).contains(n))
        .unwrap_or(DEFAULT_LISTEN_BACKLOG)
}

/// Accept backlog (`DYN_HTTP_LISTEN_BACKLOG`, default 4096).
pub fn listen_backlog() -> u32 {
    parse_listen_backlog(std::env::var(env_llm::DYN_HTTP_LISTEN_BACKLOG))
}

/// Build the listen address from an IP literal host. IPv6 brackets are optional,
/// so `::` and `[::]` are equivalent.
pub fn listen_socket_addr(host: &str, port: u16) -> Result<SocketAddr> {
    let literal = host.trim();
    let literal = literal
        .strip_prefix('[')
        .and_then(|inner| inner.strip_suffix(']'))
        .unwrap_or(literal);
    let ip: IpAddr = literal
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid address '{host}:{port}': {e}"))?;
    Ok(SocketAddr::new(ip, port))
}

/// `tokio::net::TcpListener::bind` listens with a backlog of 128. A few thousand
/// clients connecting within seconds overflow that, and overflowed connections
/// are delayed or, depending on host TCP settings, fail.
pub fn bind_listener(addr: SocketAddr) -> std::io::Result<tokio::net::TcpListener> {
    let socket = if addr.is_ipv4() {
        tokio::net::TcpSocket::new_v4()?
    } else {
        tokio::net::TcpSocket::new_v6()?
    };
    socket.set_reuseaddr(true)?;
    socket.bind(addr)?;
    socket.listen(listen_backlog())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_listen_backlog_env_var() {
        assert_eq!(
            parse_listen_backlog(Err(std::env::VarError::NotPresent)),
            DEFAULT_LISTEN_BACKLOG
        );
        assert_eq!(
            parse_listen_backlog(Ok("0".to_string())),
            DEFAULT_LISTEN_BACKLOG
        );
        assert_eq!(
            parse_listen_backlog(Ok("invalid".to_string())),
            DEFAULT_LISTEN_BACKLOG
        );
        assert_eq!(
            parse_listen_backlog(Ok((i32::MAX as u32 + 1).to_string())),
            DEFAULT_LISTEN_BACKLOG
        );
        assert_eq!(
            parse_listen_backlog(Ok(i32::MAX.to_string())),
            i32::MAX as u32
        );
        assert_eq!(parse_listen_backlog(Ok(" 8192 ".to_string())), 8192);
    }
}
