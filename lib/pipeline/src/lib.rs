// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The pipeline vocabulary every Dynamo process speaks, with no dependency on
//! the distributed runtime: [`engine`] (the `AsyncEngine` traits and streams),
//! [`pipeline`] (request context, pipeline nodes, stream types, pipeline
//! errors), [`error`] (the `DynamoError` taxonomy migration and routing key
//! on), and [`protocols::annotated`] (the `Annotated` stream envelope).
//! `dynamo-runtime` re-exports all of it at its historical paths.

pub mod engine;
pub mod error;
pub mod pipeline;

pub mod protocols {
    pub mod annotated;
    pub mod maybe_error;
}
