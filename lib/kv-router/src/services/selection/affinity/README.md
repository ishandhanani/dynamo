<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Session affinity

Session affinity binds a session to the worker (and optionally the data-parallel rank) that served it, so later requests in the session reuse that worker's prefix cache. This directory holds the two shared layers; the routing hosts keep only what differs between them.

## Layers

| Layer | File | Owns |
|---|---|---|
| `SessionAffinity` | `table.rs` | Bindings, idle TTL, entry and session-id limits, `Hard`/`Soft` commit rules, `Hold` and `AffinityLease`, versioned `apply_replica_update`, the `AffinityReplicaSink` hook |
| `AffinityResolver` | `resolver.rs` | How one request drives the table: `resolve` (hold, liveness check, cancellation, full-table fallback), `query` (no hold), `commit` (bind, `Hard` rejection, departed-worker failover), and `release_filtered` |
| Pin rule | `scheduling/queue.rs` | How a binding constrains worker selection, for every policy |

The resolver takes a `TargetLiveness` from its host: whether a bound target can currently be scheduled. A binding whose target is not live is dropped before selection and the session re-initializes; a `Hard` commit rejected because the target departed after the hold re-binds the session to the dispatched worker, joining another request's initialization when one is in flight.

## The pin rule

`resolve` and `query` return an `AffinityRequirement { target, strength }` whose strength is the table's mode. The scheduler queue applies it when it selects a worker:

- `Hard`: selection is limited to the target while the target is eligible (known to the partition, inside the caller's allowed set, available, not overloaded). An ineligible target selects normally; `commit` then re-binds if the worker departed, or rejects if it is live but another worker was dispatched. Queue admission does not narrow: a hard-bound request waits in the router only when every eligible worker is busy, as any request does, and otherwise goes to its bound worker and queues there.
- `Soft`: the target reaches the selection policy as `WorkerSelectionContext::affinity_target()`. The builtin default policy stays on the bound worker while it is a candidate; a custom policy may select another worker, and `commit` rebinds the session to wherever the request ran.
- Explicit request pins (`pinned_worker`) are a separate, exact constraint with their own queue lane; they are not affinity.

## Hosts

| Host | Session key | Liveness | Admission | Commit point | Lease owner | Cancels a wait |
|---|---|---|---|---|---|---|
| Selection service (`core/run.rs`, `core/reservations.rs`) | `session_context.session_id` or legacy `session_id` on the wire request | partition catalog (`CatalogLiveness`) | `Book`: resolve, book, commit | after the booking lands, before the reservation is installed | the reservation row | core shutdown token |
| Standalone EPP (`deploy/inference-gateway/ext-proc/src/selector.rs`) | request session header | same as the selection service it embeds | `Book` | same | same | same |
| Frontend KV (`lib/llm/src/kv_router/routing_host`) | `x-dynamo-session-affinity-id` header, subagent group key | discovery plus DP-rank range | `Lease`; the host drives the shared table directly (`AffinityCoordinator`, `lib/llm/src/session_affinity`) and passes a `Hard` binding to the core as an exact pin, a `Soft` one as a target | after dispatch returns a response stream | the response stream | client disconnect (or the request's cleanup budget for a decode leg with staged KV) |
| Frontend builtin modes and prefill (`routing_host/builtin.rs`, `prefill_router`) | same as frontend KV | discovery plus DP-rank range | the host drives the shared table directly (`AffinityCoordinator`) | after dispatch | the response stream | client disconnect |
| Runtime EPP (`ext-proc/src/epp.rs`) | none | n/a | n/a | n/a | n/a | no decode session affinity |

## Errors

Hosts map `AffinityError` onto their own error types. The core maps `InvalidArgument` (the request contradicts the binding or exceeds the session-id limit) to `BadRequest`, `ResourceExhausted` to `NotReady`, `Cancelled` to scheduler shutdown, and `Dropped` to `Internal`. A full table never fails `resolve`: the request routes without affinity and `full_table_fallbacks` counts it (a plain counter on the resolver; not exported as a metric).
