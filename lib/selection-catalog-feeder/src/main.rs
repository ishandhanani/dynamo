// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use dynamo_selection_catalog_feeder::{CatalogMirror, FeederConfig, parse_selector, run};
use dynamo_sidecar_common::{CatalogClient, HttpEndpoint};
use tokio_util::sync::CancellationToken;

/// Mirror Ready Kubernetes pods into a Dynamo selection-service catalog.
#[derive(Debug, Parser)]
#[command(name = "dynamo-selection-catalog-feeder")]
struct Args {
    /// Selection service base URL (its `/workers` API).
    #[arg(long, env = "DYN_SELECTION_CATALOG_URL")]
    catalog_url: String,
    /// Namespace to watch; defaults to the feeder's own pod namespace.
    #[arg(long, env = "POD_NAMESPACE")]
    namespace: String,
    /// Equality label selector, `k=v,k2=v2`.
    #[arg(long, env = "DYN_FEEDER_SELECTOR")]
    selector: String,
    /// Model name the workers serve.
    #[arg(long, env = "DYN_MODEL_NAME")]
    model_name: String,
    #[arg(long, env = "DYN_FEEDER_ROUTING_GROUP", default_value = "default")]
    routing_group: String,
    /// Port the frontend dispatches to on each pod.
    #[arg(long, env = "DYN_FEEDER_PORT")]
    port: u16,
    #[arg(long, env = "DYN_FEEDER_ENDPOINT_SCHEME", default_value = "http")]
    endpoint_scheme: String,
    /// KV block size; must equal the engine's.
    #[arg(long, env = "DYN_KV_CACHE_BLOCK_SIZE")]
    block_size: u32,
    #[arg(long, env = "DYN_FEEDER_DATA_PARALLEL_SIZE", default_value_t = 1)]
    data_parallel_size: u32,
    /// KV-event ZMQ port of DP rank 0. Omit to register no KV events.
    #[arg(long, env = "DYN_FEEDER_KV_EVENT_PORT")]
    kv_event_port: Option<u16>,
    #[arg(long, env = "DYN_FEEDER_KV_EVENT_PORT_STRIDE", default_value_t = 1)]
    kv_event_port_stride: u16,
    #[arg(long, env = "DYN_FEEDER_KV_EVENT_REPLAY_PORT")]
    replay_port: Option<u16>,
    #[arg(long, env = "DYN_FEEDER_TOTAL_KV_BLOCKS")]
    total_kv_blocks: Option<u64>,
    #[arg(long, env = "DYN_FEEDER_MAX_NUM_BATCHED_TOKENS")]
    max_num_batched_tokens: Option<u64>,
    /// Lease duration; the feeder heartbeats at a third of it.
    #[arg(long, env = "DYN_FEEDER_TTL_SECS", default_value_t = 30.0)]
    ttl_secs: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();
    if args.ttl_secs <= 0.0 {
        anyhow::bail!("--ttl-secs must be positive");
    }
    let selector = parse_selector(&args.selector)?;
    let catalog = HttpEndpoint::parse(&args.catalog_url, "--catalog-url")
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let client = CatalogClient::new(catalog).map_err(|error| anyhow::anyhow!("{error}"))?;
    let config = FeederConfig {
        model_name: args.model_name,
        routing_group: args.routing_group,
        endpoint_scheme: args.endpoint_scheme,
        port: args.port,
        block_size: args.block_size,
        data_parallel_size: args.data_parallel_size.max(1),
        kv_event_port: args.kv_event_port,
        kv_event_port_stride: args.kv_event_port_stride.max(1),
        replay_port: args.replay_port,
        total_kv_blocks: args.total_kv_blocks,
        max_num_batched_tokens: args.max_num_batched_tokens,
        ttl: Duration::from_secs_f64(args.ttl_secs),
    };
    let kube = kube::Client::try_default()
        .await
        .context("building Kubernetes client")?;

    let cancel = CancellationToken::new();
    let stop = cancel.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        stop.cancel();
    });
    run(
        kube,
        &args.namespace,
        selector,
        config,
        CatalogMirror::new(client),
        cancel,
    )
    .await
}
