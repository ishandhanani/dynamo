// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Message formats exchanged between Dynamo frontends and workers.

pub mod agents;
pub mod common;
pub mod response_merge;

/// The token ID type
pub type TokenIdType = u32;
