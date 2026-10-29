// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The frontend's KV index: the shared `dynamo_kv_router` [`Indexer`] built
//! from router configuration with runtime-backed metrics and, when configured,
//! a request-plane remote primary.

use std::{sync::Arc, time::Duration};

use anyhow::Result;
use dynamo_kv_indexer::config::ApproximateCachePolicyKind;
use dynamo_kv_indexer::pruning::PruneConfig;
use dynamo_kv_indexer::{
    ApproximateRetentionConfig, ConcurrentRadixTreeCompressed, KvIndexer, KvIndexerMetrics,
    LowerTierIndexers, SessionPrefixIndexer, ThreadPoolIndexer,
};
use dynamo_kv_router::config::KvRouterConfig;

// Re-export tiered-match types so internal callers (`indexer::TieredMatchDetails`)
// keep working after these types moved to `dynamo-kv-router`.
pub(crate) use dynamo_kv_indexer::TieredMatchDetails;
#[allow(unused_imports)]
pub(crate) use dynamo_kv_indexer::WireTieredMatchDetails;
pub use dynamo_kv_indexer::ingest::backend::{Indexer, RemotePrimary, SideIndexer};
pub(crate) use dynamo_kv_indexer::ingest::recording::ApproximateRequestLease;
use dynamo_kv_indexer::ingest::session_updates::SessionUpdateSender;
use dynamo_runtime::component::Component;
pub(crate) use ingress::{RuntimeIngress, RuntimeIngressArgs};
use tokio_util::sync::CancellationToken;

mod embedding_cache;
mod ingress;
mod recovery;
pub mod remote;

pub use self::embedding_cache::{
    EmbeddingCacheIndexer, preprocessed_multimodal_cache_keys, try_build_cache_indexer,
};

use dynamo_kv_indexer::{IndexerCounterFamily, KvIndexerCounterCollector};
use dynamo_runtime::metrics::PrometheusMetric;

/// Component-registered indexer metrics, memoized per process so repeated indexer construction
/// never double-registers the counters.
pub trait KvIndexerMetricsExt {
    fn from_component(component: &Component) -> Arc<KvIndexerMetrics>;
}

const KV_CACHE_EVENTS_APPLIED_SUFFIX: &str = "kv_cache_events_applied";
const KV_CACHE_EVENT_WARNINGS_SUFFIX: &str = "kv_cache_event_warnings";
const CKF_MUTATION_SUFFIX: &str = "ckf_mutation_total";
static KV_INDEXER_METRICS: std::sync::OnceLock<Arc<KvIndexerMetrics>> = std::sync::OnceLock::new();

/// One indexer counter family as a component metric: the hierarchy names it
/// and injects its labels, the collector reads the indexer's atomics at scrape.
trait IndexerCounterMetric: PrometheusMetric {
    const FAMILY: IndexerCounterFamily;
    fn collector(&self) -> &KvIndexerCounterCollector;
}

macro_rules! indexer_counter_metric {
    ($name:ident, $family:expr) => {
        #[derive(Clone)]
        struct $name(KvIndexerCounterCollector);

        impl prometheus::core::Collector for $name {
            fn desc(&self) -> Vec<&prometheus::core::Desc> {
                self.0.desc()
            }

            fn collect(&self) -> Vec<prometheus::proto::MetricFamily> {
                self.0.collect()
            }
        }

        impl PrometheusMetric for $name {
            fn with_opts(opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
                KvIndexerCounterCollector::new($family, opts).map(Self)
            }
        }

        impl IndexerCounterMetric for $name {
            const FAMILY: IndexerCounterFamily = $family;

            fn collector(&self) -> &KvIndexerCounterCollector {
                &self.0
            }
        }
    };
}

indexer_counter_metric!(EventsAppliedMetric, IndexerCounterFamily::EventsApplied);
indexer_counter_metric!(EventWarningsMetric, IndexerCounterFamily::EventWarnings);
indexer_counter_metric!(CkfMutationsMetric, IndexerCounterFamily::CkfMutations);

/// Register `T` on the component and put `metrics` in its scrape. On failure
/// the counters still count; they are just not exported.
fn expose<T: IndexerCounterMetric>(
    component: &Component,
    suffix: &str,
    metrics: &KvIndexerMetrics,
) {
    match dynamo_runtime::metrics::create_metric::<T, _>(
        component,
        suffix,
        T::FAMILY.help(),
        &[],
        None,
        None,
    ) {
        Ok(metric) => metric.collector().attach(metrics),
        Err(e) => tracing::warn!(
            metric = suffix,
            "Failed to register kv indexer counters on the component; they stay unexposed: {e}"
        ),
    }
}

impl KvIndexerMetricsExt for KvIndexerMetrics {
    fn from_component(component: &Component) -> Arc<KvIndexerMetrics> {
        KV_INDEXER_METRICS
            .get_or_init(|| {
                let metrics = KvIndexerMetrics::new_unregistered();
                expose::<EventsAppliedMetric>(component, KV_CACHE_EVENTS_APPLIED_SUFFIX, &metrics);
                expose::<EventWarningsMetric>(component, KV_CACHE_EVENT_WARNINGS_SUFFIX, &metrics);
                expose::<CkfMutationsMetric>(component, CKF_MUTATION_SUFFIX, &metrics);
                Arc::new(metrics)
            })
            .clone()
    }
}
use self::remote::RemoteIndexer;
pub use self::remote::{ServedIndexerHandle, ServedIndexerMode, ensure_served_indexer_service};
#[cfg(feature = "ckf-diagnostics")]
pub(crate) use recovery::WorkerQueryHealthSnapshot;
pub(crate) use recovery::{
    DEFAULT_RECOVERY_ATTEMPT_TIMEOUT, KvEventSubscriptionHandle, RecoveryResetReason,
    RecoverySupervisor, RecoveryTarget, TargetFaultDisposition, start_target_subscriber,
};
pub(crate) use recovery::{
    KvStateAgentStatusCell, start_subscriber, start_worker_kv_query_endpoint,
    start_worker_kv_query_endpoint_with_status,
};
#[cfg(test)]
pub(crate) use recovery::{WorkerQueryClient, WorkerQueryTransport};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolvedApproximatePrimaryPolicy {
    Ttl,
    Lru,
    TtlRemoteFallback,
}

fn resolve_approximate_primary_policy(
    config: &KvRouterConfig,
) -> Result<ResolvedApproximatePrimaryPolicy> {
    if config.use_kv_events
        && config.router_approximate_cache_policy == ApproximateCachePolicyKind::Lru
    {
        anyhow::bail!(
            "router_approximate_cache_policy=lru requires use_kv_events=false; the local side indexer is TTL-only"
        );
    }
    if config.use_kv_events
        || config.router_approximate_cache_policy == ApproximateCachePolicyKind::Ttl
    {
        return Ok(ResolvedApproximatePrimaryPolicy::Ttl);
    }
    if config.use_remote_indexer || config.serve_indexer {
        return Ok(ResolvedApproximatePrimaryPolicy::TtlRemoteFallback);
    }
    Ok(ResolvedApproximatePrimaryPolicy::Lru)
}

/// Build the frontend router's index for `kv_router_config`.
pub(crate) async fn build(
    component: &Component,
    kv_router_config: &KvRouterConfig,
    block_size: u32,
    model_name: Option<&str>,
    cancellation_token: CancellationToken,
    session_prefix_index: Option<Arc<SessionPrefixIndexer>>,
) -> Result<Indexer> {
    let approximate_policy = resolve_approximate_primary_policy(kv_router_config)?;
    if approximate_policy == ResolvedApproximatePrimaryPolicy::TtlRemoteFallback {
        tracing::warn!(
            use_remote_indexer = kv_router_config.use_remote_indexer,
            serve_indexer = kv_router_config.serve_indexer,
            "Approximate LRU requires a router-local primary indexer; falling back to TTL"
        );
    }

    if kv_router_config.router_predicted_ttl_secs.is_some() && !kv_router_config.use_kv_events {
        anyhow::bail!(
            "router_predicted_ttl_secs requires use_kv_events=true; \
             do not combine a primary approximate indexer with a side approximate indexer"
        );
    }
    if kv_router_config.use_remote_indexer {
        let model_name = model_name
            .ok_or_else(|| {
                anyhow::anyhow!("model_name is required when use_remote_indexer is configured")
            })?
            .to_string();
        let indexer_component_name = component.name();
        tracing::info!(
            indexer_component = %indexer_component_name,
            model_name,
            "Using remote KV indexer"
        );
        let remote =
            RemoteIndexer::new(component, model_name, kv_router_config.use_kv_events).await?;
        let approx = predict_on_route_side_indexer(
            component,
            kv_router_config,
            block_size,
            cancellation_token.child_token(),
        );
        return Ok(Indexer::Remote {
            primary: Arc::new(remote),
            approx,
            primary_records_routing_decisions: !kv_router_config.use_kv_events,
        });
    }

    if !kv_router_config.use_kv_events {
        let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
        let prune_config = PruneConfig {
            ttl: Duration::from_secs_f64(kv_router_config.router_ttl_secs),
        };
        let retention = if approximate_policy == ResolvedApproximatePrimaryPolicy::Lru {
            tracing::info!(
                "Starting local primary approximate indexer with capacity-bounded LRU retention"
            );
            ApproximateRetentionConfig::Lru {
                fallback_ttl: prune_config,
            }
        } else {
            ApproximateRetentionConfig::Ttl(prune_config)
        };
        if kv_router_config.router_event_threads > 1 {
            let primary = Arc::new(
                ThreadPoolIndexer::new_with_metrics_and_approximate_retention(
                    ConcurrentRadixTreeCompressed::new(),
                    kv_router_config.router_event_threads as usize,
                    block_size,
                    Some(kv_indexer_metrics.clone()),
                    Some(retention),
                ),
            );
            let lower_tier = LowerTierIndexers::new_with_metrics(
                kv_router_config.router_event_threads as usize,
                block_size,
                Some(kv_indexer_metrics),
            );
            let session_updates = session_prefix_index.map(|index| {
                SessionUpdateSender::for_concurrent(index, Arc::clone(&primary), lower_tier.clone())
            });
            return Ok(Indexer::Concurrent {
                primary,
                lower_tier,
                approx: None,
                primary_records_routing_decisions: true,
                session_updates,
            });
        }

        let primary = KvIndexer::new_with_approximate_retention(
            cancellation_token.child_token(),
            block_size,
            kv_indexer_metrics.clone(),
            Some(retention),
        );
        let lower_tier =
            LowerTierIndexers::new_with_metrics(1, block_size, Some(kv_indexer_metrics));
        let session_updates = session_prefix_index.map(|index| {
            SessionUpdateSender::for_legacy(index, primary.clone(), lower_tier.clone())
        });
        return Ok(Indexer::Single {
            primary,
            lower_tier,
            approx: None,
            primary_records_routing_decisions: true,
            session_updates,
        });
    }

    let approx = predict_on_route_side_indexer(
        component,
        kv_router_config,
        block_size,
        cancellation_token.child_token(),
    );

    if kv_router_config.router_event_threads > 1 {
        let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
        let primary = Arc::new(ThreadPoolIndexer::new_with_metrics(
            ConcurrentRadixTreeCompressed::new(),
            kv_router_config.router_event_threads as usize,
            block_size,
            Some(kv_indexer_metrics.clone()),
        ));
        let lower_tier = LowerTierIndexers::new_with_metrics(
            kv_router_config.router_event_threads as usize,
            block_size,
            Some(kv_indexer_metrics),
        );
        let session_updates = session_prefix_index.map(|index| {
            SessionUpdateSender::for_concurrent(index, Arc::clone(&primary), lower_tier.clone())
        });
        return Ok(Indexer::Concurrent {
            primary,
            lower_tier,
            approx,
            primary_records_routing_decisions: false,
            session_updates,
        });
    }

    let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
    let primary = KvIndexer::new_with_pruning(
        cancellation_token.child_token(),
        block_size,
        kv_indexer_metrics.clone(),
        None,
    );
    let lower_tier = LowerTierIndexers::new_with_metrics(1, block_size, Some(kv_indexer_metrics));
    let session_updates = session_prefix_index
        .map(|index| SessionUpdateSender::for_legacy(index, primary.clone(), lower_tier.clone()));
    Ok(Indexer::Single {
        primary,
        lower_tier,
        approx,
        primary_records_routing_decisions: false,
        session_updates,
    })
}

/// Predict-on-route side indexer for `router_predicted_ttl_secs`, with the
/// component's indexer metrics.
fn predict_on_route_side_indexer(
    component: &Component,
    kv_router_config: &KvRouterConfig,
    block_size: u32,
    cancellation_token: CancellationToken,
) -> Option<SideIndexer> {
    let ttl_secs = kv_router_config.router_predicted_ttl_secs?;
    tracing::info!(
        ttl_secs,
        "Starting predict-on-route side indexer (short-TTL approximate)"
    );
    Some(SideIndexer::new(
        Duration::from_secs_f64(ttl_secs),
        block_size,
        kv_router_config.router_event_threads as usize,
        KvIndexerMetrics::from_component(component),
        cancellation_token,
    ))
}

#[cfg(test)]
pub(super) mod test_util {
    use dynamo_router_protocol::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, RouterEvent, StorageTier,
        compute_seq_hash_for_block,
    };

    pub(crate) fn store_event(
        worker_id: u64,
        dp_rank: u32,
        event_id: u64,
        prefix_hashes: &[u64],
        local_hashes: &[u64],
        storage_tier: StorageTier,
    ) -> RouterEvent {
        let prefix_block_hashes: Vec<LocalBlockHash> =
            prefix_hashes.iter().copied().map(LocalBlockHash).collect();
        let parent_hash = compute_seq_hash_for_block(&prefix_block_hashes)
            .last()
            .copied()
            .map(ExternalSequenceBlockHash);

        let full_hashes: Vec<LocalBlockHash> = prefix_hashes
            .iter()
            .chain(local_hashes.iter())
            .copied()
            .map(LocalBlockHash)
            .collect();
        let full_sequence_hashes = compute_seq_hash_for_block(&full_hashes);
        let new_sequence_hashes = &full_sequence_hashes[prefix_hashes.len()..];
        let blocks = local_hashes
            .iter()
            .zip(new_sequence_hashes.iter())
            .map(|(&local_hash, &sequence_hash)| KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(sequence_hash),
                tokens_hash: LocalBlockHash(local_hash),
                mm_extra_info: None,
            })
            .collect();

        RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash,
                    start_position: None,
                    blocks,
                }),
                dp_rank,
            },
            storage_tier,
        )
    }
}
