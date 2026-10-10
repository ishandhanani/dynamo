<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Selection Service

The public deployment and HTTP API contract is documented in
[Standalone Selection Service](../../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/router/standalone-selection.md).

This module composes the existing worker catalog, KV indexer, scheduler queue,
and active-sequence accounting. Keep these implementation invariants explicit:

- `/select` is query-only; `/select_and_reserve` books before returning.
- `/reservations` accepts `effective_prefill_tokens` as a direct
  `PrefillLoadHint` and rejects a value greater than normalized ISL.
- Mooncake overlap fields are raw matched-token observability. Effective
  prefill tokens use the scheduler's weighted cache credit and are not derived
  from `longest_matched`.
- Discovery-driven worker membership goes through `WorkerCatalogSource` and
  `CatalogReconciler` (`membership.rs`), not `upsert_worker`/`delete_worker`.
- A partition's KV index comes from `HostCache.index: KvIndexSource::Owned`. Its `KvEventIngress` builds and feeds it: `ZmqDirectIngress` here, `RuntimeIngress` in the frontend. The frontend ingress also supports Dynamo-native remote indexing. The `Indexer` type itself is shared with the frontend (`services::indexer::backend`).
- Each partition owns a `SessionAffinity` table with shared binding, resolution, and lease rules (see [Session Affinity](affinity/README.md)). Reservations own service leases; frontend streams own frontend leases. The KV frontend embeds `SelectionService` and shares its partition table. The table's replication sink retains the frontend transport runtime.
- Valid worker metadata updates preserve live bookings on surviving ranks and
  KV state from unchanged event sources. Catalog commits and ingress changes
  are serialized; partition policy factories can initialize independently.
- The frontend request lease manager owns expiry for embedded partitions.
  Standalone partitions use periodic request expiry.
- Selector replicas synchronize admission, prefill-complete, and free events.
- **NOTE:** Output-block updates remain local. They are deliberately excluded
  from replica sync because their frequency would consume disproportionate
  network bandwidth.
- Replica sync is best-effort and may delay, reorder, or drop events. Unknown
  catalog entries are dropped under `ReplicaWorkerPolicy::RequireRegistered`.
- Startup indexer recovery waits for replay submission, not a full processing
  barrier.
- Reservation IDs must be globally unique. Retry and idempotency behavior is
  the existing active-sequence behavior.
