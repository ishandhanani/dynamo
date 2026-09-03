// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_sidecar_common::{HttpEndpoint, SidecarArgs};

fn parse_http_endpoint(raw: &str) -> Result<HttpEndpoint, String> {
    HttpEndpoint::parse(raw, "--vllm-http-endpoint").map_err(|error| error.to_string())
}

#[derive(clap::Parser, Clone, Debug)]
#[command(
    name = "dynamo-vllm-sidecar",
    about = "Run a Dynamo worker against vLLM's native gRPC service"
)]
pub(crate) struct Args {
    #[command(flatten)]
    pub sidecar: SidecarArgs,

    /// Optional controller-routable vLLM HTTP base URL for RL compatibility operations.
    #[arg(
        long,
        env = "VLLM_HTTP_ENDPOINT",
        value_parser = parse_http_endpoint
    )]
    pub vllm_http_endpoint: Option<HttpEndpoint>,

    /// Frontend-routable address of the vLLM gRPC service. When set, the worker
    /// advertises it so frontends with direct dispatch enabled send requests to
    /// vLLM directly instead of through this sidecar.
    #[arg(long, env = "DYN_ADVERTISE_GRPC_ENDPOINT")]
    pub advertise_grpc_endpoint: Option<String>,

    /// Base URL of a Dynamo selection service to register this worker with over
    /// HTTP (no etcd/NATS). Requires `--advertise-grpc-endpoint`; the worker
    /// heartbeats its lease and deregisters on shutdown.
    #[arg(long, env = "DYN_SELECTION_CATALOG_URL", value_parser = parse_catalog_endpoint)]
    pub selection_catalog_url: Option<HttpEndpoint>,

    /// Lease duration for the selection-catalog registration, in seconds.
    #[arg(long, env = "DYN_SELECTION_CATALOG_TTL_SECS", default_value_t = 30.0)]
    pub selection_catalog_ttl_secs: f64,

    /// Routing group under which the worker registers in the selection catalog.
    /// Defaults to the disaggregation role (`prefill`, `decode`) or `default`.
    #[arg(long, env = "DYN_SELECTION_CATALOG_ROUTING_GROUP")]
    pub selection_catalog_routing_group: Option<String>,
}

fn parse_catalog_endpoint(raw: &str) -> Result<HttpEndpoint, String> {
    HttpEndpoint::parse(raw, "--selection-catalog-url").map_err(|error| error.to_string())
}
