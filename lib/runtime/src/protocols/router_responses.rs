// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `MaybeError` for the router's typed request-plane responses, so hosts can carry them over the
//! runtime's typed streams. The response types live in `dynamo-router-types`; the trait lives here.

use dynamo_router_types::cache::{
    IndexerQueryResponse, IndexerRecordRoutingDecisionResponse, WorkerKvQueryResponse,
};

use crate::error::DynamoError;
use crate::protocols::maybe_error::MaybeError;

impl MaybeError for WorkerKvQueryResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        WorkerKvQueryResponse::Error(err.to_string())
    }

    fn err(&self) -> Option<DynamoError> {
        match self {
            WorkerKvQueryResponse::Error(msg) => Some(DynamoError::msg(msg.clone())),
            _ => None,
        }
    }
}

impl MaybeError for IndexerQueryResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        IndexerQueryResponse::Error(err.to_string())
    }

    fn err(&self) -> Option<DynamoError> {
        match self {
            IndexerQueryResponse::Error(msg) => Some(DynamoError::msg(msg.clone())),
            _ => None,
        }
    }
}

impl MaybeError for IndexerRecordRoutingDecisionResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        IndexerRecordRoutingDecisionResponse::Error(err.to_string())
    }

    fn err(&self) -> Option<DynamoError> {
        match self {
            IndexerRecordRoutingDecisionResponse::Error(msg) => Some(DynamoError::msg(msg.clone())),
            _ => None,
        }
    }
}
