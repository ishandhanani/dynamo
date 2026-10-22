// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine protocols. The wire types live in `dynamo_llm_protocol::protocols::common`
//! and are re-exported here at their historical paths; `input_trigger` and the
//! `*OptionsProvider` traits are frontend-side and stay in this crate.

pub use dynamo_llm_protocol::protocols::common::*;

pub mod input_trigger;

pub use super::providers::{
    OutputOptionsProvider, SamplingOptionsProvider, StopConditionsProvider,
};
