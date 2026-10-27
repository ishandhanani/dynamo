// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Classifier runtime and plugin API, owned by `dynamo-admission`; this module keeps the old paths.

pub(crate) use dynamo_admission::classifier::ClassificationOverrides;
pub use dynamo_admission::classifier::{
    AbortCause, ClassifierError, ClassifyEvent, ClassifyFuture, ClassifyRequest, RequestClassifier,
    RequestClassifierContext,
};
pub use dynamo_admission::runtime::{RequestClassifierRuntime, RequestLifecycle};
