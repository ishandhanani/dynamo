// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang-native HTTP protocol types.

pub mod generate;
pub const HTTP_CAPABILITY: &str = "sglang_generate_http_v1";
pub(crate) mod stream;
