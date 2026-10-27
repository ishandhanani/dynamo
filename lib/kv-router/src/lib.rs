// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `dynamo-kv-router` is a facade over the router crates it was split into; it is removed in the
//! next step. Every path below resolves to its owner crate.

pub use dynamo_kv_indexer as indexer;
pub use dynamo_kv_indexer::{recovery, session_prefix_index, zmq_wire};
pub use dynamo_router_types::{identity, kv_hints, protocols, tracking_hash, worker_type};
pub use dynamo_slot_tracker as sequences;
pub use dynamo_worker_selection::conditional_disagg;

pub mod scheduling {
    pub use dynamo_admission::policy_queue;
    pub use dynamo_router_types::config;
    pub use dynamo_router_types::prefill_load;
    pub use dynamo_scheduler::scheduling_compat::*;
    pub use dynamo_scheduler::{local, overlap, overlap_refresh, queue};
    pub mod request_classifier {
        pub use dynamo_admission::classifier::*;
        pub use dynamo_admission::runtime::*;
    }
    pub use dynamo_worker_selection::lora as lora_filter;
    pub use dynamo_worker_selection::selector;
    pub mod policy_config {
        pub use dynamo_router_types::config::policy::*;
    }
    pub mod queue_admission {
        pub use dynamo_router_types::protocols::WorkerPlacement;
    }
}

pub mod plugins {
    pub use dynamo_scheduler::plugins::*;
    pub mod worker_selection {
        pub use dynamo_router_types::selection_inputs as inputs;
        pub use dynamo_scheduler::plugins::{
            WorkerSelectionPolicyParameters, WorkerSelectionPolicyProvider,
            WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistryError,
        };
        pub use dynamo_worker_selection::plugin::*;
    }
    pub mod request_classifier {
        pub use dynamo_admission::classifier::*;
    }
}

pub mod services {
    #[cfg(feature = "standalone-indexer")]
    pub mod indexer {
        pub use dynamo_kv_indexer::http::{
            IndexerConfig, logging, parse_workers, recovery, run_server, server,
        };
        pub use dynamo_kv_indexer::ingest::{
            backend, listener, lookup, metrics, recording, registry, session_updates,
        };
    }
    #[cfg(feature = "standalone-indexer")]
    pub use dynamo_kv_indexer::overlap;
    #[cfg(feature = "standalone-selection")]
    pub use dynamo_scheduler::selection;
    #[cfg(feature = "standalone-slot-tracker")]
    pub mod slot_tracker {
        pub use dynamo_slot_tracker::http::{SlotTrackerConfig, run_server, server};
        pub use dynamo_slot_tracker::registry;
    }
}

pub use indexer::concurrent_radix_tree_compressed;
pub use indexer::positional as nested_map;
pub use indexer::pruning as approx;
pub use indexer::radix_tree;

pub use scheduling::config;
pub use scheduling::queue;
pub use scheduling::selector;
pub use sequences::multi_worker as multi_worker_sequence;
pub use sequences::single as sequence;

#[cfg(any(test, feature = "bench"))]
pub use dynamo_scheduler::test_utils;

pub use self::multi_worker_sequence::{
    ActiveSequencesMultiWorker, NoopSequencePublisher, ReplicaWorkerPolicy, SequenceError,
    SequencePublisher, SequenceRequest, SequenceSubscriber,
};
pub use self::sequence::{ActiveSequences, RequestId};
pub use self::sequences::{PrefillTokenDeltas, WorkerLoadProjection};
pub use concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed;
pub use config::{
    ConditionalDisaggPolicyKind, KvRouterConfig, RouterConfigOverride, RouterPrefillLoadModel,
    RouterQueuePolicy, SharedCacheType,
};
pub use identity::{DEFAULT_ROUTING_GROUP, DcId, RoutingPartitionId, RoutingPartitionRef};
#[allow(deprecated)]
pub use indexer::{
    AnchorAwareBranchShardedIndexer, AnchorRef, AnchorTask, BranchShardedIndexer,
    LowerTierContinuation, LowerTierIndexer, MaybeError, SharedKvCache, SyncIndexer,
    ThreadPoolIndexer,
};
pub use nested_map::PositionalIndexer;
pub use plugins::worker_selection::WorkerSelectionPolicyFactory;
pub use plugins::worker_selection::{
    ScoredWorkerCandidate, WorkerCacheInput, WorkerCacheInputs, WorkerCandidate, WorkerCandidates,
    WorkerFilter, WorkerInputView, WorkerInputs, WorkerLoadInput, WorkerPicker, WorkerScorer,
    WorkerSelectionContext, WorkerSelectionPolicy,
};
pub use protocols::{
    KvCacheEventError, KvTransferEnforcement, LocalBlockHash, OverlapScores, RouterEvent,
    RouterEventSink, SharedCacheHits, WorkerConfigLike, WorkerId, compute_block_hash_for_seq,
};
pub use queue::SchedulerQueue;
pub use radix_tree::RadixTree;
pub use scheduling::LocalScheduler;
pub use scheduling::LoraWorkerFilter;
pub use scheduling::PrefillLoadEstimator;
pub use scheduling::{
    KvSchedulerError, PotentialLoad, SchedulingRequest, SchedulingResponse, SessionContext,
    WorkerSelectionInputTrigger, WorkerSelectionPolicyError,
};
#[cfg(any(test, feature = "bench"))]
pub use selector::DefaultWorkerSelector;
pub use selector::{WorkerSelectionInput, WorkerSelector};
pub use session_prefix_index::{
    LogicalNode, NodeId, SessionId, SessionPrefixIndexError, SessionPrefixIndexer,
};
pub use tracking_hash::{TrackingHashAlgorithm, TrackingHashContext, TrackingHashScope};
pub use worker_type::WorkerType;
