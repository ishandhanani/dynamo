---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Runtime-Optional Frontend (Design)
subtitle: One embedded selection core under the frontend, EPP, and sidecar, with the Dynamo runtime as a deployment option
---

## Summary

The frontend, the inference-gateway EPP, and the [standalone selection
service](../../modular-components/router/standalone-selection.md) each carry
their own wiring of the same scheduler, indexer, and active-sequence
accounting. This design converges them on one embedded `SelectionService`
instance per worker pool and moves transport (dispatch, retry, cancellation)
above a single selection seam. Once dispatch no longer needs the Dynamo request
plane, `dynamo-runtime` becomes deployment-optional: in-process Python backends
keep it, while [sidecar-backed](sidecar-backends.md) deployments run with no
etcd or NATS anywhere.

```text
                    frontend process (no runtime infra required)
  OpenAI HTTP - preprocess/tokenize - DisaggCoordinator
                                        |-- SelectionService (prefill pool)   embedded, in-process
                                        '-- SelectionService (decode pool)    embedded, in-process
                                        |     catalog + ZMQ KV events + per-pool replica sync
                                        v
                             direct dispatch (native gRPC / HTTP)
                                        v
       +---------------- worker pod / VM process ----------------+
       | sidecar: catalog client (HTTP) + event translator (ZMQ  |
       | PUB + replay) + health/heartbeat agent                  |
       | engine: stock vLLM / SGLang / TensorRT-LLM, native gRPC |
       +---------------------------------------------------------+
```

Aggregated deployments run one instance; disaggregated deployments run two
under a coordinator; encoder and fallback pools generalize as further
instances. The EPP hosts the same coordinator in standalone mode.

## Contracts

Each contract has exactly one definition and is consumed by the frontend, the
EPP, and the sidecar. Code anchors name where today's behavior lives.

### 1. Selection seam

Selection and booking live below the seam; transport and retry live above it.

- Booking ownership across a dispatch failure: the host that booked releases
  on failure, before the error reaches the retry manager, so a later attempt
  never overlaps stale cleanup (the ordering `lib/kv-router/src/scheduling/AGENTS.md`
  already requires).
- Release-on-redispatch: a migration retry frees the previous booking and
  excludes every worker the migration state machine has already failed.
- Prefill-complete arrives as a response-stream callback from the dispatcher,
  not from a worker-published event. This replaces the always-on
  `active_sequences_events` ingress
  (`ACTIVE_SEQUENCES_SUBJECT` in `lib/llm/src/kv_router.rs`) and resolves
  `TODO(epp-disconnect-semantics)` in
  `deploy/inference-gateway/ext-proc/src/server.rs`.

### 2. Routing inputs

The preprocessed bundle handed to selection: token ids or precomputed block and
sequence hashes, input sequence length, LoRA name, the structured session
context (`session_id`, `parent_session_id`, `session_final`, KV hints, input
trigger), priority, routing constraints, multimodal block hashes, and expected
output tokens. This is the `select` and `select_and_reserve` request body of
the selection service and makes `TODO(epp-request-routing)` in
`deploy/inference-gateway/ext-proc/src/epp.rs` explicit.

### 3. Dispatch

Native gRPC or HTTP transport to the engine with:

- cancellation propagation from the client stream to the engine;
- streaming and first-token signals surfaced to the host (first token doubles
  as prefill-complete for contract 1);
- an engine-error to migratable-error mapping that preserves the taxonomy in
  `lib/llm/src/migration.rs` (`is_migratable`: connect, disconnect, timeout,
  engine shutdown, incomplete stream, and single-worker overload migrate;
  cancellation and pool-wide exhaustion do not);
- the decode-must-proceed-on-cancel invariant from
  `lib/llm/src/kv_router/prefill_router/mod.rs`: once prefill may have
  completed, decode routing continues so an in-flight KV transfer has a
  receiver and its blocks are freed.

### 4. Registration and liveness

The worker catalog record documented in
[Worker Registration](../../modular-components/router/standalone-selection.md#worker-registration)
is the schema: endpoint, block size, data-parallel layout, capacity fields,
taints and topology domains, KV-transfer domain, KV-event endpoints per rank,
and router-hint metadata. Liveness is either a catalog heartbeat with a TTL or
an agent-owned health signal; a missed heartbeat drains, then deregisters.
Role (prefill, decode, encode, aggregated) selects the instance a record
belongs to.

### 5. Disaggregated coordination

Two embedded services are booked with a linked, non-atomic protocol:

1. Decode-anchored preview: a query-only `select` against the decode pool
   returns the candidate worker, `potential_decode_blocks`, and `decode_busy`.
2. Prefill busy read: an advisory `select` against the prefill pool returns
   `worker_load.prefill_busy` without queue admission.
3. Decision: the conditional-disaggregation policy
   (`lib/kv-router/src/conditional_disagg.rs`) chooses remote prefill or
   bypass.
4. Pinned commit: `select_and_reserve` on the decode pool with
   `pinned_worker`, then on the prefill pool constrained to the decode
   worker's `kv_transfer_domain`, each with its own `router_config_override`.

Compensation for every partial state:

| State | Action |
|---|---|
| Prefill booked, decode booking failed | Free prefill; retry with the decode worker excluded, or bypass to aggregated. |
| Decode booked, prefill booking failed | Bypass: run prefill on the decode worker, or free decode and retry. |
| Prefill completes | Free the prefill booking early on the prefill-complete signal (contract 1). |
| Either dispatch fails | Release that pool's booking before retry (contract 1). |

Cross-pool atomicity is explicitly waived: the two ledgers are per pool, and
the coordinator compensates rather than locks. Decode-first greedy pairing is
accepted; joint prefill/decode optimization is out of scope until a global
pair-cost scheduler becomes a requirement.

## Phases

| Phase | Scope | Exit |
|---|---|---|
| 0 | Selection-core parity in `lib/kv-router`: host hooks, shared-cache hits and full session context, LoRA narrowing, reservation index, advisory busy evaluation, approximate and remote indexers, router hints. No frontend behavior change. | Selection service is a strict input-superset of the frontend scheduler; EPP unaffected. |
| 1 | Re-host the aggregated `KvRouter` on an embedded `SelectionService` behind a construction flag. | Frontend benchmark A/B shows no routing-quality or latency regression; offline KV replay byte-parity; router end-to-end tests green on both paths; old wiring deleted. |
| 2 | Direct native-gRPC dispatch as a second transport beside the request plane; sidecar off the request path. | Sidecar-backed aggregated and disaggregated deployments serve via direct dispatch; worker-shutdown migration tests pass. |
| 3 | `DisaggCoordinator` on two embedded services (contract 5); encoder as a third instance; EPP standalone embeds the same coordinator and dynamo-mode is deprecated. | Disaggregated, conditional-disaggregated, and encoder parity in replay tests; one coordinator implementation shared by frontend and EPP. |
| 4 | Runtime-free sidecar: HTTP registration by role, own ZMQ PUB and replay, health-driven drain; Kubernetes and static-file catalog feeders; frontend assembly with no `DistributedRuntime`. | Full sidecar-backed deployment on a bare VM and on Kubernetes with zero etcd or NATS processes. |
| 5 (optional) | Compile-time optionality: pipeline traits in a leaf crate so `lib/llm` drops its unconditional `dynamo-runtime` edge. | Only with a concrete consumer such as Go or C bindings or a minimal image. |

```text
P0 --> P1 --> P3 --+
  '--> P2 --------+--> P4 --> (P5?)
```

## Risks

- Coordinator compensation bugs (phase 3): partial-booking cleanup across two
  ledgers. Mitigation: an explicit state machine, exhaustive transition tests,
  and the replay-parity disaggregated lifecycle campaign.
- Migration semantics over native gRPC (phase 2): a lossy error mapping
  silently turns migrations into client-visible failures. The taxonomy in
  contract 3 is the part to get right first.
- Hot-path regression (phase 1): a `RwLock` catalog and per-partition locking
  replace tuned paths. The benchmark gate runs before the old wiring is deleted.
- Feature narrowing over native contracts: tool-call and reasoning parsers are
  not supported over the vLLM gRPC path today. Contract 2 enumerates what is
  preserved.
- Best-effort replica sync: the frontend inherits the selection service's
  consistency invariants. Phase 1 must not route correctness through sync;
  sync carries load state only.
