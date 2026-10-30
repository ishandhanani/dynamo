// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker selection for the Dynamo router: routing eligibility helpers, the selection plugin API
//! (filters, scorers, pickers), the composed selection policy, LoRA narrowing, conditional
//! disaggregation, session affinity, and the policies Dynamo ships (`builtin`), including the
//! runtime's stateless route pickers.

pub mod affinity;
pub mod builtin;
pub mod conditional_disagg;
pub mod lora;
pub mod plugin;
pub mod selector;

pub mod config;
