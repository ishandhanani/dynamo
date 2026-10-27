// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cache ingest: the in-process indexer mux (`backend`), lookup and recording helpers, session
//! updates, and, behind `ingest-zmq`, the worker registry with its ZMQ event listeners.

pub mod backend;
pub mod lookup;
pub mod recording;
pub mod session_updates;

#[cfg(feature = "ingest-zmq")]
pub mod listener;
#[cfg(feature = "ingest-zmq")]
pub mod metrics;
#[cfg(feature = "ingest-zmq")]
pub mod registry;
#[cfg(feature = "ingest-zmq")]
pub mod zmq;
