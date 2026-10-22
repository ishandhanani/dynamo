// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo LLM Protocol
//!
//! Wire contract between Dynamo frontends and workers: preprocessed requests,
//! engine outputs, the model deployment card and its runtime config, and the
//! worker/model/endpoint kinds that describe a deployment. `dynamo-llm`
//! re-exports every module here at its historical path, so existing
//! `dynamo_llm::{protocols::common, model_card, local_model::runtime_config,
//! model_type, worker_type, endpoint_type, common}` imports keep working.
//!
//! Everything in this crate crosses a process boundary and is subject to the
//! N-2 worker/frontend compatibility window described in `AGENTS.md`.

pub mod common;
pub mod endpoint_type;
pub mod local_model;
pub mod model_card;
pub mod model_type;
pub mod protocols;
pub mod worker_type;
