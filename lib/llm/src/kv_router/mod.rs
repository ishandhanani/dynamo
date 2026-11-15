// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What dynamo-llm keeps of the KV router: the shared KV cache client. The request path
//! (`KvRouter`, `RoutingHost`, the prefill and encode hops) is the crate `dynamo-router-host`.
//! The worker-side KV event publisher is `dynamo_backend_common::kv_publisher`.

pub mod shared_cache;
