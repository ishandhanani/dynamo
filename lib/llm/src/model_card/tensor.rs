// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tensor model configuration as carried on the model deployment card.
//!
//! The request and response types of the tensor protocol live in
//! `crate::protocols::tensor`, which re-exports these.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use validator::Validate;

#[derive(Debug, Serialize, Clone, Eq, PartialEq, Deserialize)]
pub enum DataType {
    Bool,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    BFloat16,
    Float32,
    Float64,
    Bytes,
}

impl DataType {
    pub fn size(&self) -> usize {
        match self {
            DataType::Bool => size_of::<bool>(),
            DataType::Uint8 => size_of::<u8>(),
            DataType::Uint16 => size_of::<u16>(),
            DataType::Uint32 => size_of::<u32>(),
            DataType::Uint64 => size_of::<u64>(),
            DataType::Int8 => size_of::<i8>(),
            DataType::Int16 => size_of::<i16>(),
            DataType::Int32 => size_of::<i32>(),
            DataType::Int64 => size_of::<i64>(),
            // 16-bit floats; spelled out so this module does not depend on `half`.
            DataType::Float16 => 2,
            DataType::BFloat16 => 2,
            DataType::Float32 => size_of::<f32>(),
            DataType::Float64 => size_of::<f64>(),
            DataType::Bytes => 0, // variable length, return 0 as indicator
        }
    }
}

#[derive(Serialize, Deserialize, Validate, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TensorMetadata {
    pub name: String,
    pub data_type: DataType,
    pub shape: Vec<i64>,

    /// Optional parameters for this tensor
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub parameters: Parameters,
}

#[derive(Serialize, Deserialize, Validate, Debug, Clone, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct TensorModelConfig {
    pub name: String,
    pub inputs: Vec<TensorMetadata>,
    pub outputs: Vec<TensorMetadata>,
    // Optional Triton model config in serialized protobuf string,
    // if provided, it supersedes the basic model config defined above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub triton_model_config: Option<Vec<u8>>,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum ParameterValue {
    Bool(bool),
    Int64(i64),
    String(String),
    Double(f64),
    Uint64(u64),
}

pub type Parameters = HashMap<String, ParameterValue>;
