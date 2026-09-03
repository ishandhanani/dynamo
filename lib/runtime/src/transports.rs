// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Transports module hosts all the network communication stacks used for talking
//! to services or moving data around the network.
//!
//! These are the low-level building blocks for the distributed system.

#[cfg(feature = "etcd")]
pub mod etcd;
pub mod event_plane;
#[cfg(feature = "nats")]
pub mod nats;

/// The NATS client handle threaded through the network manager. Without the
/// `nats` feature it is uninhabited, so every `Option` of it is `None`.
#[cfg(feature = "nats")]
pub type NatsClientHandle = async_nats::Client;
#[cfg(not(feature = "nats"))]
pub type NatsClientHandle = std::convert::Infallible;

/// NATS connection options carried by `DistributedConfig`; uninhabited without
/// the `nats` feature.
#[cfg(feature = "nats")]
pub type NatsClientOptions = nats::ClientOptions;
#[cfg(not(feature = "nats"))]
pub type NatsClientOptions = std::convert::Infallible;
pub mod tcp;
mod utils;
pub mod zmq;
