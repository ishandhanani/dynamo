---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Worker Selector Signal Reference
subtitle: Candidate columns and request fields exposed to runtime worker selector plugins
---

**Experimental.** Available since v1.3.0.

This reference maps the safe Rust worker selector API to the NVIDIA Dynamo KV router state used to populate it. See [Worker Selector Plugins](worker-selector-plugins.md) for a build and loading walkthrough.

## Request Signals

Request signals are available without declaring candidate inputs.

| `SelectionInput` accessor | Type | Dynamo value |
|---|---|---|
| `block_size()` | `u32` | KV cache block size in tokens |
| `isl_tokens()` | `u64` | Input sequence length in tokens |
| `expected_output_tokens()` | `Option<u64>` | Request-provided expected output length; `None` when absent |
| `tracks_prefill_tokens()` | `bool` | Whether this request uses active prefill-token tracking |
| `has_shared_cache_hits()` | `bool` | Whether the router received external shared-cache hit data for the request |
| `selection_mode()` | `SelectionMode` | `QueryOnly`, `Tracked`, or `TrackedWithAdmission` scheduling mode |
| `request_id()` | `Option<&str>` | Request ID when the scheduling mode carries one |
| `session_id()` | `Option<&str>` | Session ID when supplied by the caller |
| `candidate_inputs()` | `CandidateInputs` | Inputs materialized for this callback, including automatically added `IDENTITY` |
| `candidate_count()` | `usize` | Number of aligned entries in each provided candidate column |
| `is_empty()` | `bool` | Whether `candidate_count()` is zero |

## Candidate Set

For unpinned requests, Dynamo expands every eligible worker into its published data-parallel ranks. It removes unavailable workers, caller-disallowed workers, overloaded workers, and workers that fail required-taint constraints before building the columns. Pinned requests are resolved without calling the plugin.

Candidate order is not stable. Every provided column has `candidate_count()` entries and uses the same index. An index and all borrowed data remain valid only for the current `select` callback. Dynamo validates a selected worker and rank again before recording request accounting.

Any nonempty input declaration automatically includes `IDENTITY`. Declaring `NONE` avoids the plugin candidate scan and column materialization, so `candidate_count()` is zero. `Selection::UseDefault` can still invoke the built-in selector. Missing per-candidate map entries use the defaults listed below.

## Candidate Inputs

### Identity

| Accessor | Type | Dynamo source | Missing value |
|---|---|---|---|
| `worker_ids()` | `&[u64]` | Runtime worker ID from the eligible `WorkerWithDpRank` | Not applicable |
| `dp_ranks()` | `&[u32]` | Data-parallel rank expanded from the worker's published start rank and size | Not applicable |

Worker IDs identify the current runtime instance and can change after a restart. Use `ROUTING`'s stable ID for restart-stable hashing when available.

### Cached Tokens

| Accessor | Type | Dynamo source | Missing value |
|---|---|---|---|
| `cached_tokens()` | `Option<&[u64]>` | `SchedulingRequest::effective_cached_tokens_for`: effective overlap converted to tokens after configured lower-tier weights | `0` |

The accessor returns `None` when `CACHED_TOKENS` was not declared. Effective cached tokens describe router credit, not necessarily device-resident tokens. External shared-cache hits do not change this column; declare `CACHE_TIERS` to read them separately.

### Cache Tiers

`cache_tiers()` returns `Option<&[WorkerSelectorCacheTiersV1]>`.

| Field | Unit | Dynamo source | Missing value |
|---|---|---|---|
| `effective_overlap_blocks` | Weighted blocks, `f64` | `SchedulingRequest::effective_overlap_blocks_for`: device overlap plus configured lower-tier credit | `0.0` |
| `device_overlap_blocks` | Blocks, `u64` | Device-tier prefix overlap. When no tier maps exist for the request, Dynamo falls back to rounded nonnegative effective overlap | `0` |
| `host_pinned_overlap_blocks` | Blocks, `u64` | Host-pinned extension beyond the device prefix | `0` |
| `disk_overlap_blocks` | Blocks, `u64` | Disk and native external-tier extension beyond faster tiers | `0` |
| `shared_cache_beyond_device_blocks` | Blocks, `u64` | External shared-cache hits at positions beyond the device prefix | `0` |

Host-pinned and disk values are extensions, not cumulative prefix depths. Add them to preceding tiers only when the strategy needs a cumulative depth. `has_shared_cache_hits()` distinguishes an observed zero from unavailable external shared-cache data.

`effective_overlap_blocks` does not include `shared_cache_beyond_device_blocks`. Combine them explicitly when a strategy assigns shared-cache credit.

### Load

`loads()` returns `Option<&[WorkerSelectorLoadV1]>`. Each value is an ephemeral projection for the incoming request at selection time.

| Field | Unit | Dynamo source | Missing value |
|---|---|---|---|
| `active_prefill_tokens` | Tokens, `u64` | Currently tracked prefill work after any configured decay model | `0` |
| `active_decode_blocks` | Blocks, `u64` | Unique blocks used by active decode sequences | `0` |
| `additional_active_blocks` | Blocks, `u64` | Incoming request blocks not already shared with active sequences on this worker | `0` |

`additional_active_blocks` estimates added active footprint. It does not mean cache misses; those blocks can still exist in an inactive cache.

### Capacity

`capacities()` returns `Option<&[WorkerSelectorCapacityV1]>`.

| Method | Type | Dynamo source | Missing value |
|---|---|---|---|
| `total_kv_blocks()` | `Option<u64>` | Total KV block capacity published in the worker runtime configuration | `None` |
| `max_num_batched_tokens()` | `Option<u64>` | Maximum batched-token capacity published in the worker runtime configuration | `None` |

Use the methods on `WorkerSelectorCapacityV1` instead of interpreting the raw `CAPACITY_UNAVAILABLE` sentinel.

### Routing

`routing()` returns `Option<&[WorkerSelectorRoutingV1]>`.

| Field or helper | Type | Dynamo source | Missing value |
|---|---|---|---|
| `candidate_stable_routing_id(index)` | `Option<&str>` | Worker-published stable routing ID, such as a StatefulSet pod hostname | `None` |
| `preferred_taint_multiplier` | `f64` | Stock routing's multiplicative adjustment for request preferred taints matched by the worker | `1.0` |

Required taints are hard eligibility constraints and are applied before the plugin. Preferred taints remain soft input through `preferred_taint_multiplier`; values below `1.0` reduce stock cost, values above `1.0` increase it, and `1.0` is neutral.

### Derived Default-Cost Inputs

| Accessor | Type | Dynamo source |
|---|---|---|
| `default_costs()` | `Option<&[f64]>` | Complete configured worker cost, including prefill load, weighted KV overlap, decode load, and preferred-taint adjustment; lower is better |
| `default_kv_overlaps()` | `Option<&[f64]>` | Weighted KV overlap credit subtracted by the default cost, including configured device decay and host, disk, and shared-cache weights; higher is better |
| `default_decode_loads()` | `Option<&[u64]>` | Active decode blocks plus the incoming request's additional active blocks; lower is better |

`default_costs()` is evaluated before temperature sampling. These values come from the same calculation used by Dynamo's built-in selector rather than a plugin-side copy.

## Input Selection

Combine inputs with `|` in `required_candidate_inputs`:

```rust
fn required_candidate_inputs(&self) -> CandidateInputs {
    CandidateInputs::CACHED_TOKENS
        | CandidateInputs::LOAD
        | CandidateInputs::ROUTING
}
```

Dynamo calls this method once for each configured decode or prefill plugin state. The returned set cannot vary per request. Accessors for undeclared optional inputs return `None`.

## Selection Results

| Result | Host behavior |
|---|---|
| `Selection::Candidate(index)` | Maps the callback-local index to a worker and DP rank, rejects an out-of-range index, validates eligibility, then derives the accounting snapshot |
| `Selection::UseDefault` | Runs Dynamo's configured built-in worker selector for the same request |

Returning `Err(String)` fails selection for that request. Calls for one plugin state are serialized, and a mutable strategy can retain owned state between calls. Separate decode and prefill states can run on different threads.

The [plugin API source](https://github.com/ai-dynamo/dynamo/blob/main/lib/worker-selector-plugin-api/src/lib.rs) is authoritative for Rust types and ABI constants. The [runtime plugin host](https://github.com/ai-dynamo/dynamo/blob/main/lib/kv-router/src/scheduling/runtime_plugin.rs) is authoritative for field materialization.
