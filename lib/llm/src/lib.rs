// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo LLM
//!
//! The `dynamo.llm` crate is a Rust library that provides a set of traits and types for building
//! distributed LLM inference solutions.

pub mod common;
pub mod entrypoint;
mod frontend_load;
pub mod grpc;
pub mod http;
// pub mod key_value_store;
pub mod tokens;
pub mod utils;

#[cfg(feature = "block-manager")]
pub mod block_manager;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(test)]
#[path = "../../runtime/src/test_utils.rs"]
mod test_utils;
