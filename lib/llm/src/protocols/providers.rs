// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-side adapter traits that extract the engine-facing option sets from a
//! request type. They are implemented by blanket impls over the OpenAI request
//! traits, so they belong to this crate rather than to the wire types they produce.

use anyhow::Result;

use super::common::{OutputOptions, SamplingOptions, StopConditions};

/// SamplingOptionsProvider is a trait that allows the caller to extract the sampling options from
/// the object that implements it. This will mutate the object.
pub trait SamplingOptionsProvider {
    fn extract_sampling_options(&self) -> Result<SamplingOptions>;
}

pub trait StopConditionsProvider {
    fn extract_stop_conditions(&self) -> Result<StopConditions>;
}

pub trait OutputOptionsProvider {
    fn extract_output_options(&self) -> Result<OutputOptions>;
}
