// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Listener setup and the environment-driven knobs of the HTTP server: bind address
//! parsing, the accept backlog, the write-idle timeout, the graceful-shutdown budget, and
//! a listener that rebinds its address when its socket dies.

use std::io;
use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

use anyhow::Result;
use dynamo_runtime_protocols::environment_names::llm as env_llm;
use tokio::net::{TcpListener, TcpStream};

use crate::server::{Accept, set_nodelay};

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
pub fn bind_listener(addr: SocketAddr) -> io::Result<TcpListener> {
    let socket = if addr.is_ipv4() {
        tokio::net::TcpSocket::new_v4()?
    } else {
        tokio::net::TcpSocket::new_v6()?
    };
    socket.set_reuseaddr(true)?;
    socket.bind(addr)?;
    socket.listen(listen_backlog())
}

/// A listener that survives a dead listening socket by rebinding its address.
///
/// [`Accept::accept_connection`] cannot fail, so recovery happens here. Errors that belong
/// to one queued connection, and process-wide resource shortages such as EMFILE, keep the
/// socket; any other accept error drops it, and the same concrete address (so an assigned
/// ephemeral port stays stable) is bound again with [`bind_listener`] after
/// `rebind_backoff`, retrying for as long as the bind fails.
pub struct RebindingTcpListener {
    /// Concrete address returned by the initial bind, including an assigned ephemeral port.
    address: SocketAddr,
    listener: Option<TcpListener>,
    rebind_backoff: Duration,
}

impl RebindingTcpListener {
    pub fn new(listener: TcpListener, address: SocketAddr, rebind_backoff: Duration) -> Self {
        Self {
            address,
            listener: Some(listener),
            rebind_backoff,
        }
    }

    /// The address every bind uses, i.e. the one the initial listener was bound to.
    pub fn local_addr(&self) -> SocketAddr {
        self.address
    }
}

impl Accept for RebindingTcpListener {
    async fn accept_connection(&mut self) -> TcpStream {
        let mut next_rebind = tokio::time::Instant::now();
        loop {
            let Some(listener) = self.listener.as_ref() else {
                tokio::time::sleep_until(next_rebind).await;
                next_rebind = tokio::time::Instant::now() + self.rebind_backoff;
                tracing::debug!(address = %self.address, "HTTP listener rebinding");
                match bind_listener(self.address) {
                    Ok(listener) => {
                        tracing::info!(address = %self.address, "HTTP listener rebound");
                        self.listener = Some(listener);
                    }
                    Err(error) => {
                        tracing::error!(
                            address = %self.address,
                            %error,
                            retry_after = ?self.rebind_backoff,
                            "HTTP listener failed to rebind"
                        );
                    }
                }
                continue;
            };

            match listener.accept().await {
                Ok((stream, _)) => {
                    set_nodelay(&stream);
                    return stream;
                }
                // Connection errors do not invalidate the listener.
                Err(error) if is_connection_error(&error) => {
                    tracing::trace!(
                        address = %self.address,
                        %error,
                        "HTTP connection failed before accept"
                    );
                }
                // Resource exhaustion does not invalidate the listener.
                Err(error) if is_resource_exhaustion_error(&error) => {
                    tracing::error!(
                        address = %self.address,
                        %error,
                        retry_after = ?self.rebind_backoff,
                        "HTTP listener lacks resources; retrying the same socket"
                    );
                    tokio::time::sleep(self.rebind_backoff).await;
                }
                Err(error) => {
                    tracing::error!(
                        address = %self.address,
                        %error,
                        "HTTP listener stopped accepting; rebinding"
                    );
                    // Drop before rebinding or the address remains in use.
                    self.listener = None;
                }
            }
        }
    }
}

fn is_connection_error(error: &io::Error) -> bool {
    if matches!(
        error.kind(),
        io::ErrorKind::ConnectionRefused
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionReset
    ) {
        return true;
    }

    // Linux accept(2) also reports pending network errors for the queued connection.
    #[cfg(target_os = "linux")]
    if let Some(code) = error.raw_os_error() {
        return matches!(
            code,
            libc::ENETDOWN
                | libc::EPROTO
                | libc::ENOPROTOOPT
                | libc::EHOSTDOWN
                | libc::ENONET
                | libc::EHOSTUNREACH
                | libc::EOPNOTSUPP
                | libc::ENETUNREACH
        );
    }

    false
}

/// EMFILE, ENFILE, and ENOBUFS require raw errno checks because Rust has no distinct kinds.
fn is_resource_exhaustion_error(error: &io::Error) -> bool {
    if error.kind() == io::ErrorKind::OutOfMemory {
        return true;
    }

    #[cfg(unix)]
    {
        if let Some(code) = error.raw_os_error() {
            return matches!(code, libc::ENFILE | libc::EMFILE | libc::ENOBUFS);
        }
    }

    false
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

    #[cfg(unix)]
    #[tokio::test]
    async fn rebinding_listener_serves_again_after_listener_shutdown() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();

        let shutdown_result = socket2::SockRef::from(&listener).shutdown(std::net::Shutdown::Both);

        let mut rebinding = RebindingTcpListener::new(listener, address, Duration::from_millis(10));

        if let Err(error) = shutdown_result {
            eprintln!("skipping: this platform refused shutdown on a listening socket: {error}");
            return;
        }

        let mut accepted = tokio::spawn(async move { rebinding.accept_connection().await });

        let mut clients = Vec::new();
        let peer = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let Ok(stream) = TcpStream::connect(address).await else {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    continue;
                };
                let client_addr = stream.local_addr().unwrap();
                clients.push((stream, client_addr));
                tokio::select! {
                    result = &mut accepted => {
                        let stream = result.expect("accept task should not panic");
                        return stream.peer_addr().unwrap();
                    }
                    _ = tokio::time::sleep(Duration::from_millis(50)) => {}
                }
            }
        })
        .await
        .expect("listener should rebind and accept connections again");

        assert!(
            clients.iter().any(|(_, client_addr)| *client_addr == peer),
            "accepted connection should be one we opened"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rebinding_listener_keeps_trying_after_a_rebind_fails() {
        let holder = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let contested_address = holder.local_addr().unwrap();

        let broken = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let shutdown_result = socket2::SockRef::from(&broken).shutdown(std::net::Shutdown::Both);

        let mut rebinding =
            RebindingTcpListener::new(broken, contested_address, Duration::from_millis(10));

        if let Err(error) = shutdown_result {
            eprintln!("skipping: this platform refused shutdown on a listening socket: {error}");
            return;
        }

        let accepted = tokio::spawn(async move { rebinding.accept_connection().await });

        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(holder);

        let client = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Ok(stream) = TcpStream::connect(contested_address).await {
                    return stream;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("listener should rebind once the address is free again");

        let stream = tokio::time::timeout(Duration::from_secs(5), accepted)
            .await
            .expect("accept should return once a later rebind succeeds")
            .expect("accept task should not panic");

        assert_eq!(
            stream.peer_addr().unwrap(),
            client.local_addr().unwrap(),
            "accepted connection should be the one we opened"
        );
    }

    #[cfg(unix)]
    #[test]
    fn resource_exhaustion_does_not_look_like_a_dead_listener() {
        for code in [libc::ENFILE, libc::ENOBUFS] {
            let error = io::Error::from_raw_os_error(code);
            assert!(
                is_resource_exhaustion_error(&error),
                "errno {code} should be treated as a process resource shortage, got {:?}",
                error.kind()
            );
            assert!(
                !is_connection_error(&error),
                "errno {code} is not a dead connection"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn pending_network_errors_leave_the_listener_usable() {
        for code in [
            libc::ENETDOWN,
            libc::EPROTO,
            libc::ENOPROTOOPT,
            libc::EHOSTDOWN,
            libc::ENONET,
            libc::EHOSTUNREACH,
            libc::EOPNOTSUPP,
            libc::ENETUNREACH,
        ] {
            let error = io::Error::from_raw_os_error(code);
            assert!(
                is_connection_error(&error),
                "errno {code} should retry accept"
            );
        }
    }

    #[tokio::test]
    async fn rebinding_listener_shutdown_interrupts_failed_rebinds() {
        let holder = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let listener = RebindingTcpListener {
            address: holder.local_addr().unwrap(),
            listener: None,
            rebind_backoff: Duration::from_secs(60),
        };
        let cancel_token = tokio_util::sync::CancellationToken::new();
        let observer = cancel_token.clone();
        let server = tokio::spawn(crate::server::serve(
            listener,
            None,
            axum::Router::new(),
            Duration::from_secs(30),
            async move {
                observer.cancelled_owned().await;
                Duration::ZERO
            },
        ));
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !server.is_finished(),
            "server should keep retrying the held address"
        );
        cancel_token.cancel();
        tokio::time::timeout(Duration::from_secs(1), server)
            .await
            .expect("shutdown should interrupt the 60-second rebind backoff")
            .expect("server task should not panic");
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn rebinding_listener_preserves_backlog_during_resource_exhaustion() {
        const CHILD_ENV: &str = "DYNAMO_TEST_LISTENER_EMFILE_CHILD";
        if std::env::var_os(CHILD_ENV).is_none() {
            // Descriptor limits are process-wide; isolate this test from sibling tests.
            // libtest names tests without the crate, so drop the first path segment.
            let module = module_path!()
                .split_once("::")
                .map_or(module_path!(), |(_, module)| module);
            let name = format!(
                "{module}::rebinding_listener_preserves_backlog_during_resource_exhaustion"
            );
            let output = tokio::time::timeout(
                Duration::from_secs(15),
                tokio::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", &name, "--nocapture"])
                    .env(CHILD_ENV, "1")
                    .kill_on_drop(true)
                    .output(),
            )
            .await
            .expect("descriptor-exhaustion child should finish promptly")
            .expect("test child should start");
            assert!(
                output.status.success(),
                "descriptor-exhaustion child failed:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
            return;
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let client = TcpStream::connect(address).await.unwrap();
        let peer = client.local_addr().unwrap();

        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: limit is writable and correctly aligned. Only this child is affected.
        assert_eq!(
            unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) },
            0
        );
        limit.rlim_cur = limit.rlim_cur.min(96);
        // SAFETY: limit is initialized; the soft limit is lowered without changing the hard limit.
        assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &limit) }, 0);

        let mut descriptors = Vec::new();
        loop {
            match std::fs::File::open("/dev/null") {
                Ok(file) => descriptors.push(file),
                Err(error) => {
                    assert_eq!(error.raw_os_error(), Some(libc::EMFILE));
                    break;
                }
            }
        }
        let error = listener.accept().await.unwrap_err();
        assert_eq!(error.raw_os_error(), Some(libc::EMFILE));

        let backoff = Duration::from_millis(200);
        let mut rebinding = RebindingTcpListener::new(listener, address, backoff);
        let started = tokio::time::Instant::now();
        let accepted = rebinding.accept_connection();
        tokio::pin!(accepted);
        assert!(
            tokio::time::timeout(Duration::from_millis(50), accepted.as_mut())
                .await
                .is_err(),
            "accept should wait during descriptor exhaustion"
        );
        drop(descriptors);

        // A fresh bind rules out lingering descriptor pressure as the cause of AddrInUse.
        let _control = TcpListener::bind("127.0.0.1:0").await.unwrap();
        assert_eq!(
            TcpListener::bind(address).await.unwrap_err().kind(),
            io::ErrorKind::AddrInUse,
            "the original listening address should remain held"
        );
        let stream = tokio::time::timeout(Duration::from_secs(2), accepted)
            .await
            .expect("the queued connection should survive the shortage");
        assert!(
            started.elapsed() >= backoff,
            "accept should finish its resource-shortage backoff before retrying"
        );
        assert_eq!(stream.peer_addr().unwrap(), peer);
    }
}
