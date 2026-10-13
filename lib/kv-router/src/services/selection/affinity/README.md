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
| Frontend KV (`lib/llm/src/kv_router/routing_host`) | `x-dynamo-session-affinity-id` header, subagent group key | discovery plus DP-rank range (`DiscoveryLiveness`, installed on the partition resolver) | `Lease`: core resolves and returns the hold in `Selected`; host runs `check_dispatch` before dispatching and commits after | after dispatch returns a response stream | the response stream | client disconnect drops the selection future (or the request's cleanup budget for a decode leg with staged KV) |
| Frontend builtin modes and builtin prefill (`routing_host/builtin.rs`, `prefill_router`) | same as frontend KV | discovery (`DiscoveryLiveness` without configs) | host resolves directly (`resolve_hosted_session`) over its own table (`HostAffinity::standalone`); `Hard` and explicit targets are exact, `Soft` is a preference except in Direct mode | after dispatch | the response stream | client disconnect (or the cleanup budget for a decode leg with staged KV) |
| Runtime EPP (`ext-proc/src/epp.rs`) | none | n/a | n/a | n/a | n/a | no decode session affinity |

Every host resolves through `AffinityResolver`. The frontend keeps `HostAffinity` (`lib/llm/src/session_affinity/host.rs`), which only holds the resolver and the event-plane replication attached to its table; the former `AffinityCoordinator` is gone. Prefill and decode hosts use separate tables on purpose: they bind to different worker pools and may carry different TTLs, so a session binds to a prefill worker and a decode worker independently.

Where the hosts differ on a live `Hard` mismatch (the bound worker is live but the request's own constraints steered selection elsewhere): the core rejects at commit and the table drops the binding, so the session re-binds on its next request; the frontend rejects before dispatch with `check_dispatch` and releases its hold without touching the binding, since nothing ran and the request, not the session, caused the mismatch.

## Replication

One wire schema, `AffinityBindingEvent` (`replication.rs`): the partition (flattened as `model_name` and `routing_group`), the session id, the target, and the version (`sequence`, `writer_id`). Every table publishes through `AffinityEventSink`, which builds the event for its partition and hands it to the host's transport; every transport feeds received events to `AffinityResolver::apply_replica_event`, the one applier. It ignores the replica's own writer id, ignores other partitions, advances the replica clock, and applies the binding only if the host can schedule its worker (`TargetLiveness`).

One writer-id rule: the id installed with the table's replication sink. Frontends use their discovery instance id (stable across restarts of the same instance); the standalone service uses a random non-zero process id, since it has no discovery.

Two transports stay: the selection service's ZMQ peer mesh (`services/common/replica_sync.rs`, topics `dynamo.session-affinity.v1` and `.v2`) and the frontend's runtime event plane (`lib/llm/src/session_affinity/replica_sync.rs`, subjects `session_affinity_events` for the old partition-less payload and `session_affinity_events_v2` for the shared schema). For one release both transports publish and apply both versions, so mixed-version replicas converge; applying the same binding twice is idempotent (the second apply refreshes the same version). The v1 forms carry a removal TODO.

## Configuration

`SessionAffinityConfig` (TTL, mode, entry limit, session-id limit) is the one configuration, and `SessionAffinityConfig::validate` the one validator; `ttl_from_secs_f64` is the one range check for a TTL given in seconds. The selection service builder, the frontend hosts, the Python bindings (`SelectionService`, `KvRouter`, and the `validate_session_affinity_ttl_secs` function the frontend's argument parser calls), and the EPP all go through it. The mode (`hard` or `soft`, `hard` by default) is settable on every host: `--router-session-affinity-mode` on the frontend, `--session-affinity-mode` and `SelectionService(session_affinity_mode=...)` on the standalone service, `DYN_EPP_SESSION_AFFINITY_MODE` on the EPP.

## Errors

Hosts map `AffinityError` onto their own error types. The core maps `InvalidArgument` (the request contradicts the binding or exceeds the session-id limit) to `BadRequest`, `ResourceExhausted` to `NotReady`, `Cancelled` to scheduler shutdown, and `Dropped` to `Internal`. A full table never fails `resolve`: the request routes without affinity and `full_table_fallbacks` counts it (a plain counter on the resolver; not exported as a metric).
