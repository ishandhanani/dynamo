<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Router state machines

This file is the single inventory of the explicit and implicit state machines in the router split tree: the crates under `lib/router/**`, `lib/kv-events`, their hosts in `lib/llm/src/{kv_router,session_affinity,discovery,pool_relay}/**`, and the mocker replay driver. Every machine lists its location, its states with their concrete representation (enum variant, field combination, or which collection holds the item), the invariants the code relies on, and a transition table. A transition table has the columns `From | Event | Guard | To | Side effects | Where`: `From` and `To` are states named in the machine's state table, `Event` is the call, message, timer, or channel outcome that drives the step, `Guard` is the condition under which that row applies (`-` when unconditional), `Side effects` are the observable mutations outside the state field itself, and `Where` is a `path:line` anchor into the tree. Rows that name a timeout, retry, cancellation, shutdown, or concurrent interleaving are kept on purpose; they are the rows a fix is most likely to break. Each area ends with the list of interleavings that were traced and found sound, so a later reader knows which races were already checked. The final section lists the defects found while building the tables, with a status column that tracks fixes.

The tables describe revision `68005ade79` of the tree (the merge of upstream main `d9e5845dd7` into the router split v2 stack). Line numbers are exact for that revision and drift afterwards; the type and function names are the stable anchors. Keeping this file current is part of the change: a PR that changes any type, loop, channel, or timer listed here updates the corresponding table (states, invariants, rows) and the `Status` column of any finding it closes in the same PR. Findings marked `fixed` in the Findings table are fixed on the branch this file ships with, and their machines' tables describe the fixed behavior.

## Map

| Id | Machine | Owning crate and file | Purpose | Drives / driven by |
|---|---|---|---|---|
| KQ-1 | Request admission lifecycle | `dynamo-scheduler` `lib/router/scheduler/src/queue.rs` | one request from `Enqueue` to Booked, Selected, Refused, Expired, Retracted, or Shutdown-drained | drives AD-1, AD-2, ST-1, KQ-2, KQ-5, KQ-6; driven by KQ-9, KQ-10, PD-2, SA-6, SA-7 |
| KQ-2 | AdmissionLifecycleTransfer and RequestLifecycleLease | `dynamo-scheduler` `queue.rs` | caller-side RAII guard over a queued or booked request | drives KQ-4; driven by KQ-1, KQ-10 |
| KQ-3 | BookingHandle | `dynamo-scheduler` `queue.rs` | RAII guard over a committed booking | drives KQ-4, ST-1; driven by KQ-10, PD-5, SA-7 |
| KQ-4 | SchedulerBookingCleanup / AdmissionCleanup | `dynamo-scheduler` `queue.rs` | deferred and direct release protocol into the actor | drives ST-1, KQ-8; driven by KQ-2, KQ-3, KQ-10 |
| KQ-5 | Overlap refresh of an extracted candidate | `dynamo-scheduler` `queue.rs` | re-scores a held-out candidate under one 250 ms timeout | driven by KQ-1; drives KQ-6, KQ-8 |
| KQ-6 | Per-class expiry timers | `dynamo-scheduler` `queue.rs` | one armed deadline per policy class | drives AD-1; driven by KQ-1, KQ-5 |
| KQ-7 | Actor turn, prepared rows, worker-configuration generation | `dynamo-scheduler` `queue.rs` | caller-built selection rows adopted only when no input moved | driven by MB-4, MB-6, MB-7; drives KQ-1 |
| KQ-8 | Capacity updates and blocked-lane recheck | `dynamo-scheduler` `queue/capacity.rs` | wakes the actor when a worker freed load | drives AD-3; driven by KQ-4, ST-1 |
| KQ-9 | Actor command loop and shutdown | `dynamo-scheduler` `queue.rs` | the actor's turn structure and exit | drives KQ-1, KQ-5 |
| KQ-10 | Host-side lease consumers | `dynamo-llm` `kv_router/request_lease.rs`, `routing_host/request_guard.rs` | second-chance lease clock, request guard, cancellation policy | drives KQ-2, KQ-3, KQ-4, ST-1, AD-5 |
| AD-1 | Queue entry lifecycle | `dynamo-admission` `lib/router/admission/src/queue.rs` | which collections hold an entry; limits, lanes, expiry indexes | driven by KQ-1, KQ-6, mocker replay; drives AD-2, AD-3 |
| AD-2 | DRR credit cycle | `dynamo-admission` `queue.rs` | per-class deficit, carry class, round cursor | driven by KQ-1 |
| AD-3 | Per-worker lane blocking | `dynamo-admission` `queue.rs` | exact-lane heads blocked until a recheck | driven by KQ-8, AD-1 |
| AD-4 | Expiry indexes | `dynamo-admission` `queue.rs`, `profile.rs` | `expiring_roots` versus `due_entries` per class kind | drives KQ-6 |
| AD-5 | RequestLifecycle phases | `dynamo-admission` `classifier/runtime.rs` | Registered, Sent, Responding, Terminal per classified request | driven by KQ-10; drives AD-6 |
| AD-6 | Classifier runtime | `dynamo-admission` `classifier/runtime.rs` | per-id generation fence, terminal delivery, shutdown | driven by AD-5, KQ-1; drives SA-9 |
| AD-7 | Class resolution and deadline derivation | `dynamo-scheduler` `queue.rs` | override versus own class; `due_at` and `class_due_at` | drives AD-1, KQ-6 |
| AD-8 | Config resolution | `dynamo-kv-router` `scheduling/policy_config.rs` | YAML to `PolicyProfile` | drives AD-1 construction |
| ST-1 | Booking lifecycle | `dynamo-slot-tracker` `multi_worker.rs`, `single.rs`, `request_maps.rs` | index entry and per-rank state per request id | driven by KQ-1, KQ-4, KQ-10, ST-4, ST-5; drives ST-2, ST-6, ST-7 |
| ST-2 | Prefill load anchor | `dynamo-slot-tracker` `prefill_tracker.rs` | decaying anchored prefill load per rank | driven by ST-1 |
| ST-3 | Worker rank lifecycle and lazy slot registration | `dynamo-slot-tracker` `topology.rs`, `multi_worker.rs` | Unknown, Authoritative, Lazy, Removed per `(worker, dp_rank)` | driven by MB-6, ST-4, ST-5, KQ-1 |
| ST-4 | Replica sync | `dynamo-slot-tracker` `replica_sync.rs`; `dynamo-llm` `kv_router/sequence.rs` | mirroring bookings between routers | driven by ST-1; drives ST-1, KQ-10; transport TR-1 |
| ST-5 | Standalone slot-tracker HTTP service | `dynamo-kv-router` `services/slot_tracker/{registry,server}.rs` | per-partition tracker attach and detach | drives ST-1, ST-3 |
| ST-6 | Scheduler load publication | `dynamo-llm` `kv_router/routing_load.rs`; `dynamo-slot-tracker` `wire.rs` | booked load to the worker monitor and over the wire | driven by ST-1 |
| ST-7 | Fullness rule | `dynamo-slot-tracker` `busy.rs` | `has_room`, `FullRankCounts` per projection pass | driven by KQ-1, KQ-7 |
| ST-8 | Expiry timers | `dynamo-slot-tracker` `single.rs`, `multi_worker.rs`, `cleanup.rs` | force expiry, periodic sweep, trie cleanup | drives ST-1 |
| SA-1 | Table entry lifecycle | `dynamo-session-affinity` `lib.rs` | Absent, Initializing, Bound-Active, Bound-Idle, Bound-Expired, Bound-Joined | drives SA-2, SA-3; driven by SA-4, SA-5, SA-11 |
| SA-2 | AffinityInitialization handle | `dynamo-session-affinity` `lib.rs` | one initialization epoch | driven by SA-4, SA-7 |
| SA-3 | AffinityLease handle | `dynamo-session-affinity` `lib.rs` | current, stale, orphaned, released | driven by SA-4, SA-6, SA-7 |
| SA-4 | Per-request acquire protocol | `dynamo-session-affinity` `lib.rs`; `dynamo-llm` `session_affinity/coordinator.rs` | wait, hold Initialize or Bound, commit, release | drives SA-1; driven by SA-6, SA-7 |
| SA-5 | Replica agreement | `dynamo-session-affinity` `lib.rs`; `dynamo-llm` `session_affinity/replica_sync.rs`; `dynamo-scheduler` `replication.rs` | version ordering across frontends or cores | drives SA-1 |
| SA-6 | Frontend per-request affinity decision | `dynamo-llm` `kv_router/routing_host.rs` | binding key, pin versus advisory target, healing | drives SA-4, KQ-1, SA-8 |
| SA-7 | Core managed session | `dynamo-scheduler` `selection/core/run.rs` | hold, steer, commit under the booking; joined failover | drives SA-4, KQ-1, KQ-3 |
| SA-8 | Kernel eligibility: pinned worker versus affinity target | `dynamo-scheduler` `filter.rs`, `queue.rs` | two inputs with different force | driven by SA-6, SA-7, AD-7 |
| SA-9 | Admission override of the affinity target | `dynamo-admission` `classifier/mod.rs`; `dynamo-scheduler` `queue.rs` | a classifier may replace a Soft target | driven by AD-6; drives SA-8 |
| SA-10 | Subagent group keys | `dynamo-llm` `kv_router/routing_host.rs` | which key a subagent binds under per phase | drives SA-6 |
| SA-11 | Reaper and manual clock | `dynamo-session-affinity` `lib.rs` | lazy collection of expired bindings | drives SA-1 |
| KI-1 | CRTC node lifecycle | `dynamo-kv-indexer` `concurrent_radix_tree_compressed/{node,state,remove}.rs` | live, dead-linked, queued, unlinked, reused nodes | driven by KI-4; drives KI-2, KI-3 |
| KI-2 | Lane lookup holds | `dynamo-kv-indexer` `concurrent_radix_tree_compressed/lane_lookup.rs` | one `Arc` per node per lane | driven by KI-1 |
| KI-3 | Sweep schedule and maintenance thread | `dynamo-kv-indexer` `concurrent_radix_tree_compressed/mod.rs`, `thread_pool.rs` | when dead nodes are reclaimed | drives KI-1 |
| KI-4 | ThreadPoolIndexer lanes and two-phase removal | `dynamo-kv-indexer` `thread_pool.rs` | rank to lane assignment, `RemoveWorker` barrier | driven by KI-7, KI-8; drives KI-1 |
| KI-5 | KvIndexer actor | `dynamo-kv-indexer` `kv_indexer.rs` | single-threaded radix tree with side channels | driven by KI-7, KI-8 |
| KI-6 | Event cursor (ZMQ listener watermark; host RankState) | `dynamo-kv-events` `cursor.rs`; `dynamo-kv-router` `services/indexer/listener.rs`; `dynamo-llm` `indexer/recovery/worker_query_state.rs` | Initial, Live, Replaying; Idle, Recovering, Fenced | drives KI-4, KI-5; driven by KI-7, MB-5 |
| KI-7 | ZMQ listener attempt | `dynamo-kv-router` `services/indexer/registry.rs`, `listener.rs` | Pending, Active, Paused, Failed, Detached per `(worker, rank)` | driven by KI-8, MB-7; drives KI-6 |
| KI-8 | Worker registration and the index registry | `dynamo-kv-router` `services/indexer/registry.rs`; `dynamo-kv-indexer` `registry.rs` | registered workers, partition indexers, removal barrier | drives KI-4, KI-5, KI-7; driven by MB-7 |
| KI-9 | Peer recovery and dumps | `dynamo-kv-router` `services/indexer/recovery.rs`, `run.rs` | standalone indexer joins from a peer dump | drives KI-8 |
| KI-10 | LocalKvIndexer recovery server | `dynamo-kv-indexer` `local.rs` | worker-side replay buffer and cached dump | driven by TR-2; drives KI-6 |
| KI-11 | ZMQ normalizer per-worker filter | `dynamo-kv-events` `zmq_wire/mod.rs`, `zmq_wire/mm_exclusion.rs` | proven block size, multimodal exclusion, overflow latch | driven by KI-7, TR-3 |
| KI-12 | Residency snapshots and session residency | `dynamo-kv-indexer` `lower_tier_indexers.rs`, `lower_tier.rs`, `session_prefix_index.rs` | `ArcSwap` snapshots and per-session frontiers | driven by KI-4, KI-5 |
| PD-1 | PrefillRouter activation | `dynamo-llm` `kv_router/prefill_router/{mod,activation}.rs` | lifecycle x target x binding | driven by MB-1; drives PD-2, PD-6 |
| PD-2 | Disaggregated request lifecycle | `dynamo-llm` `prefill_router/mod.rs` | gate, conditional plan, prefill select, dispatch, handoff, decode | drives KQ-1, PD-3, PD-4, PD-5 |
| PD-3 | Prefill observer and decode merge | `dynamo-llm` `prefill_router/handoff.rs` | `PrefillTask::forward_decode` | driven by PD-2 |
| PD-4 | Client-cancellation link | `dynamo-llm` `prefill_router/cancellation.rs` | propagates stop and kill to the prefill context | driven by PD-2 |
| PD-5 | Prefill booking ownership | `dynamo-llm` `routing_host/request_guard.rs`, `request_lease.rs` | Planned, Committed, Dispatched, Released-early, Finished, Expired | drives KQ-3, KQ-10 |
| PD-6 | PrefillReservation | `dynamo-llm` `prefill_router/query.rs` | EPP query-path reservation | drives KQ-3 |
| PD-7 | Encode hop | `dynamo-llm` `kv_router/encoder_router.rs` | encode on an independent context, inline fallback | same shape as PD-1 |
| PD-8 | Embedded selection partition | `dynamo-llm` `kv_router/embedded.rs` | construction-time `OnceLock` and primed source | drives SA-6, MB-7 |
| MB-1 | Discovery group status and cohorts | `dynamo-llm` `discovery/controller.rs` | Queued, Building, Ready, Retrying, Blocked, BlockedReady; cohorts | drives MB-2, PD-1 |
| MB-2 | Admission channel and admitted Client view | `dynamo-llm` `discovery/worker_set.rs`; `dynamo-runtime` `component/client.rs` | Live, Emptied, Frozen view | driven by MB-1; drives MB-3, KQ-7 |
| MB-3 | Runtime Client routing instances | `dynamo-runtime` `component/client.rs` | discovered, routable, overloaded; availability initialized | driven by MB-2; drives KQ-7, SA-8 |
| MB-4 | RuntimeConfigWatch | `dynamo-llm` `discovery/runtime_configs.rs` | instances intersected with base cards | drives MB-5, MB-6, MB-7 |
| MB-5 | KV-source membership and coordinator | `dynamo-llm` `discovery/kv_source_membership.rs`, `kv_source_watch.rs` | per-rank source status; bound, retry, stopped coordinator | driven by MB-4; drives KI-6 |
| MB-6 | Kernel worker configuration and slot topology | `dynamo-scheduler` `local.rs` | configs map published to slots; `Rejected` publish | driven by MB-4, MB-7; drives ST-3 |
| MB-7 | Selection core catalog lifecycle | `dynamo-scheduler` `selection/{types,catalog,membership}.rs`, `selection/core/workers.rs` | Absent, Incomplete, Schedulable, Draining, Unschedulable; reconciler | driven by MB-4, HTTP, Python; drives MB-6, KI-7, KI-8 |
| TR-1 | Replica-sync mesh | `dynamo-kv-router` `services/common/replica_sync.rs`; `dynamo-scheduler` `replication.rs` | PUB/SUB between routers, peer set | carries ST-4, SA-5 |
| TR-2 | Worker-side KvEventPublisher lifecycle | `dynamo-llm` `kv_router/publisher/mod.rs` | creating, advertising, processing, stopping | drives TR-3, TR-4, TR-5, KI-10 |
| TR-3 | Publisher non-recovery ZMQ listener | `dynamo-llm` `publisher/zmq_listener.rs` | per-rank cursor, liveness clear | driven by TR-2; drives TR-5 |
| TR-4 | Publisher recovering listener | `dynamo-llm` `publisher/zmq_recovery.rs` | Recovering, Success, Failed, MissingHistory | driven by TR-2; drives TR-5 |
| TR-5 | Event processor batching | `dynamo-llm` `publisher/event_processor.rs`, `batching.rs` | idle, pending, flushing, barrier, fenced | driven by TR-3, TR-4 |
| TR-6 | Remote indexer client | `dynamo-llm` `indexer/remote.rs` | topology predicates per call, no transitions | driven by MB-3 |
| TR-7 | KV metrics subscriber and fan-in source | `dynamo-llm` `kv_router/metrics_subscriber.rs`, `direct_zmq_fan_in.rs` | mailbox, latched fault, connect backoff | drives the worker monitor |
| TR-8 | Pool relay registration and Global Discovery lease | `dynamo-llm` `pool_relay/routing_pool.rs`; `global-discovery-client` `heartbeat.rs` | registration watch, heartbeat lease | driven by MB-1 topology |
| TR-9 | Pool relay WAN publication | `dynamo-llm` `pool_relay/publication/{hub,stream,source}.rs`, `wan/grpc/service.rs` | hub, subscriber, stream, permits | driven by TR-10 |
| TR-10 | gRPC transport lifecycle | `dynamo-llm` `pool_relay/wan/grpc/server.rs` | bound, serving, not serving | drives TR-9 |

Machines documented by two analysts are merged under the owning crate: the kernel's drive of the policy queue (admission M5) is KQ-1; the kernel class timers (admission M4, second half) are KQ-6; prepared rows versus configuration generation (membership M7) is KQ-7; lazy slot registration (membership M6, kernel-queue M1, slot-tracker SM-3) is ST-3; the ZMQ listener attempt and cursor (transports T1, T2) are KI-7 and KI-6; the publisher event processor (kv-indexer 12) is TR-5; the kv-router service registries (membership M9) are KI-8 and ST-5.

## Kernel queue (`dynamo-scheduler`)

Abbreviations in this section: `Q:` = `lib/router/scheduler/src/queue.rs`, `L:` = `lib/router/scheduler/src/local.rs`, `A:` = `lib/router/admission/src/queue.rs`, `MW:` = `lib/router/slot-tracker/src/multi_worker.rs`, `RL:` = `lib/llm/src/kv_router/request_lease.rs`, `RG:` = `lib/llm/src/kv_router/routing_host/request_guard.rs`. One `SchedulerQueueActor` task per `SchedulerQueue` (`Q:756`, spawned `Q:1017`). Callers reach it through a bounded `mpsc<AdmissionCommand>` (`Q:389`, capacity 256 `Q:62`), the lock-free `AdmissionCleanup` side channel (`Q:536`), the `CapacityUpdates` side channel (`queue/capacity.rs:16`), and the synchronous `direct_release` closure (`Q:961-978`). Every mutation of the pending queue, class timers, counters, and DRR state happens on the actor task; slot-tracker bookings are mutated by the actor and by direct release.

### KQ-1 Request admission lifecycle

Location: `SchedulerQueueActor` `Q:756`; entry `SchedulerQueue::enqueue_admitted_with_block_hashes_and_lease` `Q:1110`; payload `QueuedRequest` `Q:88`, `AdmissionContext` `Q:253`. The admission analyst's M5 (arrival, queue, extraction, outcome) is this machine; AD-1 holds the queue-internal view.

| State | Representation |
|---|---|
| Submitted | `AdmissionCommand::Enqueue` in channel or being applied; `request.resp_tx.is_some()`; not in `pending` |
| Refused-at-ingress | `request.respond(Err)` before or without insertion (`Q:1119-1127`, `1134-1140`, `1143-1147`, `1712-1716`, `1760-1763`, `insert_arrival` refusals `1827-1830`, `1844-1868`) |
| Direct-admitting | `handle_enqueue` calls `admit_one` without insertion: empty worker set (`Q:1729-1731`) or `class.never_queues() && !should_queue` (`Q:1736-1756`) |
| Queued | `PolicyQueueEntry<QueuedRequest>` in `pending` (shared heap `class.pending` or exact lane `class.ready_by_worker[w]`, lane possibly in `class.blocked_workers`); counted in `pending_count`, `pending_isl_tokens`, `class_counters[c]` (`Q:1794-1797`) |
| Expired | removed by `take_expired` (`Q:1900`) or by `insert_arrival`'s own due check; answered `DeadlineExceeded` if receiver open (`Q:1875-1884`) |
| Retracted | removed by `drain_cleanup` `take_if_in_class` (`Q:2004-2010`), unanswered (`resp_tx` dropped with the entry) |
| Extracted | returned by `pending.extract_next` with an unsettled `DispatchCharge` (`Q:2125-2146`); counters already decremented (`Q:2148-2165`) |
| Refreshing | Extracted and inside `refresh_candidate` (`Q:2261`), see KQ-5 |
| Selected (untracked) | `!mode.is_tracked()` and `respond(Ok)` succeeded: `Admission::Selected` (`Q:2589`) |
| Booked | `slots.add_request_admitted` ok (`Q:2663`), transfer armed `Booking` (`Q:2673`), `attempt_tx` sent (`Q:2680`), `respond(Ok)` succeeded (`Q:2682`): `Admission::Booked` |
| Rolled back | booked but `respond(Ok)` failed: `free_if_booking` (`Q:2689-2695`): `Admission::Refused` |
| Refused-after-extraction | `Admission::Refused` from due time, closed receiver, selection error, booking error, or `RefreshEnd::Released` (`Q:2224-2242`) |
| Shutdown-drained | `pending.drain()` at actor exit, answered `SubscriberShutdown` (`Q:1582-1599`) |

Invariants:

- I1. `pending_count == pending.pending_count()` and `class_counters[c].pending_count == pending.class_stats(c).requests`; every removal subtracts (extraction `Q:2148-2165`, expiry `Q:1903-1904`, retraction `Q:2014-2016`, shutdown `Q:1585-1596`); the only add is `Q:1794-1797` after `Insertion::Queued`. `PolicyQueue::retain` and `pop_next` are not used by the actor.
- I2. `QueuedRequest.due_at == pending.enforced_due_at(metadata)` (`Q:1713`, `A:781-785`): explicit deadline, else class due only when the class has head expiry (`expires_class_budget` = controlled delay and EDF, `lib/router/admission/src/profile.rs:94-96`). Never changes after insertion.
- I3. A tracked request is booked only after `response_is_closed()` is false (`Q:2646`) and `due_at > admitted_at` (`Q:2655`); booking precedes delivery; a failed delivery rolls the booking back (`Q:2689-2695`).
- I4. `received_total` increments exactly once per applied `Enqueue` (`Q:1708-1710`); `rejected_*_total` at most once per request and never when the receiver is already closed at the check (`Q:1876-1878`, `1849-1851`).
- I5. The DRR charge of an extracted entry is settled exactly once (`Q:2243`) with `admitted = (admission != Refused)`; `Selected` pays, `Refused` does not.
- I6. Only tracked requests reach `book_and_respond`; `QueryOnly` passes the same queue and gate and is answered by `respond(Ok)` without booking (`Q:2581-2594`).
- I7. With queueing enabled on any class, a `TrackedWithLifecycle` request must carry a `RequestLifecycleLease` (`Q:1119-1127`); legacy `Tracked` never has one and cannot be retracted from the queue.
- Outcomes of a queued request: Booked or Selected (DRR charged); Refused (answered with an error, or dropped because the caller left; never re-queued; DRR not charged); Expired (`DeadlineExceeded`, `rejected_due_time_passed_total`); Retracted (lease dropped while `Pending`, unanswered); Shutdown-drained.
- Replay driver (`lib/mocker/src/replay/offline/extensions/kv_router/mod.rs`): `enqueue_with_due_at` with `due_at: None`, `class_due_at = slo_ttft_due_at(virtual now)`, `worker_count = workers * dp_size` (`:1058-1081`); `drain_pending` calls `recheck_all_workers` then `pop_next` (charges at once) (`:1525-1556`); cancellation uses `retain` (`:1153-1158`); no expiry.

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| (caller) | `enqueue_admitted_*` | queueing enabled, lifecycle mode, no lease | Refused-at-ingress | `respond(Err(BookingFailed))` | `Q:1119-1127` |
| (caller) | same | classification invalid, unknown class, or pinned worker not allowed | Refused-at-ingress | `respond(Err)` | `Q:1134-1147` |
| (caller) | same | channel closed | Refused-at-ingress | `respond(Err(SubscriberShutdown))` | `Q:1148-1151` |
| (caller) | same | channel slot reserved | Submitted | caller builds `PreparedAdmission` while holding the slot (`prepare_candidates`), sends `Enqueue`, awaits `ack_rx` | `Q:1152-1170` |
| Submitted | actor applies `Enqueue` | `drain_cleanup` due | Submitted | `drain_cleanup()` runs first so released entries leave before limits count | `Q:1642` |
| Submitted | `handle_enqueue` | `due_at <= now` | Refused-at-ingress | `reject_due_time_passed` | `Q:1712-1716` |
| Submitted | `handle_enqueue` | worker set empty | Direct-admitting | `admit_one` (selection yields `NoEndpoints`) | `Q:1729-1731` |
| Submitted | `handle_enqueue` | `class.never_queues()` and (`!queueing_enabled` or (no backlog and some eligible worker has room)) | Direct-admitting | `admit_one`; returns `== Booked` as drain hint | `Q:1736-1756` |
| Submitted | `handle_enqueue` | `never_queues` and should queue, `due_at <= limits_now` | Refused-at-ingress | `reject_due_time_passed` | `Q:1758-1763` |
| Submitted | `insert_arrival` | - | Submitted | `expire_due_classes(now)` first (KQ-6) | `Q:1818` |
| Submitted | `insert_arrival` loop | `due_at <= now` | Refused-at-ingress | counted `DeadlineExceeded` | `Q:1827-1830`, `1844-1847` |
| Submitted | `pending.enqueue_with_due_at` | class limits allow | Queued | `arm_if_earlier` class timer, `transfer.arm_pending()`, counters += | `Q:1831-1841`, `1788-1797` |
| Submitted | limit refusal | receiver closed | Refused-at-ingress (uncounted) | drop | `Q:1849-1851` |
| Submitted | limit refusal | class has an entry due by now | Submitted (retry) | `expire_class`; re-attempt insertion | `Q:1852-1863` |
| Submitted | limit refusal | no due entry | Refused-at-ingress | `rejected_queue_size_exceeded_total += 1`, `respond(Err(QueueRejected))` | `Q:1864-1868` |
| Queued | class timer fires or any drain start | entry due by `now` | Expired | `take_expired`, counters -=, `reject_due_time_passed` (counted iff receiver open), timer rearmed | `Q:1888-1910` |
| Queued | `drain_cleanup` with `AdmissionLifecycle` target in `Pending` | `tracked_request_id` in `unmanaged_request_ids` | Retracted | counters -=, lane unblocked if root, timer rearmed if armed, entry dropped | `Q:2001-2020`, `A:964-1035` |
| Queued | `drain_ready` then `extract_next` | class `!queueing_enabled`, or pinned worker invalid (treated as not full), or some eligible rank has room under `class.busy_thresholds` | Extracted | counters -=, charge taken | `Q:2121-2165`, gate `Q:2752-2780` |
| Queued | `extract_next` | gate false for a lane root | Queued (lane blocked) | lane in `blocked_workers`; rechecked only by `recheck_worker` or `recheck_all_workers` (AD-3, KQ-8) | `A:475-566` |
| Extracted | `should_refresh_overlap` | refresher set, `refresh_after` set, hashes present, waited >= `refresh_after` | Refreshing | pending `reply` sent first | `Q:2170-2185` |
| Extracted | `admit_one` | `due_at <= now` | Refused-after-extraction | counted `DeadlineExceeded` | `Q:2548-2554` |
| Extracted | `select_worker_for_request` | Err | Refused-after-extraction | `respond(Err)` | `Q:2557-2563` |
| Extracted | untracked | `due_at` passed | Refused-after-extraction | counted | `Q:2581-2588` |
| Extracted | untracked | `respond(Ok)` ok / failed | Selected / Refused | - | `Q:2589-2593` |
| Extracted | `book_and_respond` | receiver closed | Refused-after-extraction (uncounted) | debug log | `Q:2646-2652` |
| Extracted | `book_and_respond` | `due_at <= admitted_at` | Refused-after-extraction | counted | `Q:2654-2657` |
| Extracted | booking | selected worker still listed in `workers_with_configs` | Booked or Refused per the rows below | `add_request_admitted` (lazy registration, ST-3) | `Q` `book_and_respond` |
| Extracted | booking | selected worker no longer listed in `workers_with_configs` | Refused-after-extraction | `add_request_if_registered_admitted`; `Err(WorkerNotFound)` answered `respond(Err(NoEndpoints))` (the outcome a pinned worker missing from the config already gets); other errors `BookingFailed` (SM-10 fixed) | `Q` `book_and_respond` |
| Extracted | `add_request_*` | Err (`DuplicateRequest`) | Refused-after-extraction | `respond(Err(BookingFailed))` | `Q` `book_and_respond` |
| Extracted | booked | - | Booked | `transfer.arm_booking`, `attempt_tx.send`, `respond(Ok)`, observer dispatch | `Q:2671-2687` |
| Booked (undelivered) | `respond(Ok)` failed | - | Rolled back | `free_if_booking(request_id, worker, attempt_id)` | `Q:2689-2695` |
| Refreshing | `RefreshEnd::Finished(Some)` | - | Extracted (scores replaced) | `request.overlap` replaced, `kv_transfer_candidates` kept only if `retain_kv_transfer_chain`; prepared rows discarded | `Q:2186-2200`, `2217-2221` |
| Refreshing | `Finished(None)` / `TimedOut` / `Shutdown` | - | Extracted (queued scores) | `stopping` set on Shutdown | `Q:2201-2211` |
| Refreshing | `DueTimePassed` / `ResponseClosed` | - | Extracted, re-judged | `refreshing && due_at <= now`: counted rejection; closed receiver: `book_and_respond` refuses | `Q:2226-2230`, `2646` |
| Refreshing | `Released` | lifecycle transfer disarmed by cleanup | Refused-after-extraction (unanswered) | no booking, no rejection count | `Q:2224-2225`, `2338-2344` |
| any after extraction | `settle(charge, admitted)` | - | same | DRR deficit (AD-2); `rearm_class_timer` | `Q:2243-2244` |
| Extracted (Shutdown) | `stopping` | - | actor exits drain | `break` out of `drain_ready`; `run` sees `recv() == None` | `Q:2246-2248`, `1570` |
| Queued | actor exit | all command senders dropped | Shutdown-drained | `drain_cleanup()`, counters -=, `respond(Err(SubscriberShutdown))` | `Q:1579-1599` |

Residual window (SM-10, follow-up for the slot-tracker owner): a removal published and reconciled between the config check and the slot lookup inside the same booking still recreates a lazy slot; the airtight form evaluates the predicate under the slot table's `workers.write()` in `ensure_worker_registered_after_miss` with a predicate-taking `add_request_impl`. Blanket strict booking was rejected because a freshly published worker whose slot is not yet registered (config `send_if_modified` runs before `apply_published_worker_configs`; the `LocalScheduler` monitor registers after `changed()`) would get `NoEndpoints`.

### KQ-2 AdmissionLifecycleTransfer and RequestLifecycleLease

Location: `AdmissionLifecycleTransfer` `Q:466`, `AdmissionLifecycleState` `Q:458`, `RequestLifecycleLease` `Q:701`; created by `new_request_lifecycle_lease` `Q:1351` only for `ScheduleMode::TrackedWithLifecycle`. The `Arc<Transfer>` is shared by the lease (caller) and the `QueuedRequest` / `AdmissionContext` (actor); the state is a `Mutex<AdmissionLifecycleState>`.

| State | Representation |
|---|---|
| Unarmed | `Unarmed{request_id}`: lease built; actor has not queued or booked |
| Pending | `Pending{request_id}`: request is in `pending`; cleanup retracts by request id |
| Booking | `Booking(desc)`: request booked; cleanup calls `free_if_booking(desc)` |
| Disarmed | `Disarmed`: ownership moved on (`commit`) or cleanup consumed (`cleanup_target`) |

Invariants:

- The lease travels inside the `Enqueue` command and returns in the ack (`CommandReply::Enqueue`, `Q:1658`), so the caller never holds a lease whose request is still Submitted; at ack time the state is `Pending`, `Booking`, or `Unarmed` (already refused).
- `arm_booking` only from `Unarmed | Pending` (`Q:492-500`); `arm_pending` only from `Unarmed` (`Q:482-490`); `Disarmed` never re-arms.
- `commit` returns `Some(desc)` iff `Booking` (`Q:725-736`); `LocalScheduler` wraps it in a `BookingHandle` with no await in between (`L:418-425`): the booking is guarded by exactly one of lease or handle.
- `Drop` enqueues cleanup iff `is_armed()` (`Pending | Booking`) (`Q:738-754`); `is_armed()` reads without consuming, `cleanup_target()` consumes and sets `Disarmed` (`Q:506-521`).
- A refused, expired, or rejected request leaves the transfer `Pending` (nothing disarms it); its lease drop therefore always enqueues a `Pending` cleanup that finds nothing (NOTE kernel-queue-5 in Findings).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Unarmed | `insert_arrival` reaches Queued | - | Pending | - | `Q:1791-1793` |
| Unarmed / Pending | `book_and_respond` booked | - | Booking(desc) | - | `Q:2671-2677` |
| Booking | caller `lease.commit()` | - | Disarmed | returns desc, `BookingHandle` armed | `Q:725-736`, `L:421-424` |
| Pending / Booking | caller drops lease (cancel, error return, ack lost) | `is_armed()` | unchanged, plus cleanup entry `AdmissionLifecycle(transfer)` | `enqueue_cleanup_entry`: push to `dirty`, `try_send(Cleanup)` best effort | `Q:738-754`, `569-586` |
| Pending | actor `drain_cleanup` | - | Disarmed | `take_if_in_class` by `tracked_request_id`, counters -=, rearm timer | `Q:1932-1937`, `2001-2020` |
| Booking | actor `drain_cleanup` | - | Disarmed | `free_if_booking(desc)` (attempt-scoped) | `Q:1938-1948` |
| Unarmed / Disarmed | `drain_cleanup` | - | Disarmed | nothing | `Q:1950` |
| Pending (candidate in refresh) | any refresh event after cleanup consumed it | `!is_armed()` | `RefreshEnd::Released` | candidate refused, unanswered | `Q:2338-2344` |
| any | `RequestLifecycleLease::disarm` (crate-private) | - | Disarmed | lease stops guarding | `Q:717-723` |

### KQ-3 BookingHandle

Location: `BookingHandle` `Q:653`; built by `SchedulerQueue::booking_handle` `Q:1374` and `LocalScheduler::add_request_if_registered_guarded` `L:691-706`. States: `armed == true` (guarding), `armed == false` after `commit` (`Q:672-681`) or `release` (`Q:683-690`). Invariant: `release` has no await between `armed = false` and the synchronous release, so a dropped `release` future cannot strand the booking.

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| armed | `commit()` | - | committed | returns desc (request id moved); no cleanup | `Q:672-681` |
| armed | `release().await` | - | released | `direct_release(desc)` synchronously: `free_if_booking` plus `capacity_changed(worker)` if applied | `Q:683-690`, `961-978` |
| armed | drop | - | released (deferred) | `cleanup.enqueue(desc)`; actor frees at next `drain_cleanup` | `Q:692-698` |
| committed / released | drop | - | same | nothing | `Q:694` |

### KQ-4 SchedulerBookingCleanup / AdmissionCleanup protocol

Location: `AdmissionCleanup` `Q:536` (`dirty: SegQueue`, `pending: AtomicBool`), `SchedulerBookingCleanup` `Q:588`, targets `SchedulerCleanupTarget` `Q:451`, consumer `drain_cleanup` `Q:1921`. `enqueue_cleanup_entry` (`Q:569-586`): push, then `pending.swap(true)`; only the producer that flipped `false -> true` tries `try_send(Cleanup)`; `Full` is ignored (the drain cadence picks it up); `Closed` drains and drops (no actor left).

Invariants:

- All releases are attempt-scoped (`release_if_booking`, `MW:1013-1080`): a second release for the same descriptor is `NoChange`; a release after the worker was reconciled away removes only the index entry (`MW:1023-1033`).
- `made_ready` (`Drain::Full`) only when an outcome `is_applied()` or a ready head was retracted and some head is dispatchable (`Q:2021`).
- Direct release records a capacity hint only if applied (`Q:972-976`).
- `AdmissionCleanup::drain` (`Q:547-566`) drains to quiescence and re-sets `pending` if producers raced in, so a coalesced wake is never lost.
- Consumer cadence: every `Cleanup` command (`Q:1692`); any command or capacity wake when `commands.is_empty() || commands_since_cleanup == 256` (`Q:1552-1556`, `1572-1575`); every command and capacity wake handled inside a refresh (`Q:2321`, `2325`); once at actor exit (`Q:1579`).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| (producer) | `enqueue(desc)` | - | `Booking` target queued | deferred `free_if_booking` (publishes `Free`) | `Q:622-626` |
| (producer) | `enqueue_expired(desc)` | - | `ExpiredBooking` target queued | deferred `expire_if_booking` (local only) | `Q:628-634` |
| (producer) | `enqueue_acknowledged(desc)` / `enqueue_and_wait` | - | released now | `direct_release` runs on the caller; ack is the result | `Q:636-651` |
| (producer) | lease drop | `is_armed()` | `AdmissionLifecycle(transfer)` queued | deferred | `Q:738-754` |
| queued | `try_send(Cleanup)` | channel `Full` | queued | waits for `commands.is_empty()`, the 256-command counter, or a capacity wake; bound 256 commands (NOTE kernel-queue-10) | `Q:579-585`, `1552-1556`, `1572-1575` |
| queued | `try_send(Cleanup)` | channel `Closed` | dropped | drained and discarded | `Q:569-586` |
| queued | actor `drain_cleanup` | - | applied | per target: `free_if_booking` / `expire_if_booking` / retract-or-free (KQ-2); `made_ready` computed | `Q:1921-2022` |

### KQ-5 Overlap refresh of an extracted candidate

Location: `refresh_candidate` `Q:2261-2346`; ends `RefreshEnd` `Q:236`; events `RefreshEvent` `Q:225`; trait `OverlapScoresRefresh` `lib/router/scheduler/src/overlap_refresh.rs:85`; `OVERLAP_REFRESH_TIMEOUT = 250ms` `overlap_refresh.rs:143`.

| State | Representation |
|---|---|
| Awaiting | `refresh` future pinned under `tokio::time::timeout` (`Q:2268-2280`); candidate out of `pending`, charge unsettled, counters already decremented |
| Interrupted-by-event | one `select!` arm fired (`ClassTimer`, `Capacity`, `Command`); handled in a new actor turn (`Q:2315`), then loop |
| Ended | one of `Finished(Option)`, `DueTimePassed`, `ResponseClosed`, `Released`, `Shutdown`, `TimedOut` |

Invariants:

- One timeout for the whole refresh; events never restart it (`Q:2267`).
- `select!` is `biased` (`Q:2287`): a completed refresh beats a simultaneous due time, close, or timer; `tokio::time::timeout` polls the inner future before its deadline, so a refresh that completes at the timeout instant is `Finished`, not `TimedOut`.
- The candidate's class timer no longer covers it; its due time is raced by its own `sleep_until(due_at)` arm (`Q:2295-2299`).
- Commands applied during a refresh are answered at once (`Q:2332-2334`); the drain they ask for is the one that resumes (only `recheck_capacity_updates` runs, `Q:2326-2331`); the refresh is never abandoned by scheduler traffic (tests `commands_during_a_refresh_are_answered_without_abandoning_it`, `a_refresh_timeout_runs_from_the_refresh_start_through_scheduler_traffic`).
- `retain_kv_transfer_chain` is a query option with no retained state (`lib/router/kv-indexer/src/lower_tier_indexers.rs:364-379`); dropping the refresh future at timeout leaks nothing.
- `turn_prepared` is cleared by every event's `begin_turn` (`Q:2315`); a refreshed candidate never uses prepared rows (`Q:2217-2221`).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Awaiting | refresh future ready | `Ok(opt)` / `Err(Elapsed)` | Finished(opt) / TimedOut | return | `Q:2287-2289` |
| Awaiting | `resp_tx.closed()` | `resp_tx == None` completes immediately | ResponseClosed | return | `Q:2290-2294` |
| Awaiting | `sleep_until(due_at)` | `due_at.is_some()` | DueTimePassed | return | `Q:2295-2299` |
| Awaiting | class timer deadline | some class armed | Interrupted | `begin_turn`; `expire_due_classes(now)` | `Q:2300-2304`, `2317-2319` |
| Awaiting | `capacity_updates.notified()` | queueing enabled | Interrupted | `begin_turn`; `drain_cleanup()`; `recheck_capacity_updates(cleanup_ready)` | `Q:2305-2309`, `2320-2323` |
| Awaiting | command received | - | Interrupted | `begin_turn`; `apply_command(cmd, true)`; recheck per drain kind; reply sent | `Q:2310-2313`, `2324-2334` |
| Awaiting | `commands.recv() == None` | - | Shutdown | return; admission of the candidate still proceeds (NOTE kernel-queue-3) | `Q:2313` |
| Interrupted | after event | `lifecycle_transfer` present and `!is_armed()` | Released | return | `Q:2338-2344` |
| Interrupted | otherwise | - | Awaiting | loop (timer deadline re-read) | `Q:2282-2284` |

### KQ-6 Per-class expiry timers

Location: `ClassTimers` `Q:172-223`: `armed: Vec<Option<Instant>>` plus `by_deadline: BTreeSet<(Instant, class)>`; consumed by `run`'s `sleep_until(next_deadline)` arm (`Q:1541`, `1560-1567`) and the refresh loop (`Q:2283`, `2300-2304`). States per class: Disarmed (`armed[c] == None`) and Armed(d) (`armed[c] == Some(d)` and `(d, c)` in `by_deadline`). The admission analyst's M4 timer table is this machine; the per-class expiry indexes are AD-4.

Invariants:

- `armed[c] == Some(d)` iff `(d, c)` in `by_deadline` (`arm` maintains both, `Q:186-199`).
- Armed(d) implies `d == pending.next_due_at(c)` or earlier; an arrival can only advance (`arm_if_earlier`, `Q:201-205`); removals re-read `next_due_at` (`rearm_class_timer` only when armed, `Q:1914-1919`; `expire_class` always, `Q:1907-1908`).
- `next_due_at` is exact: `DueKey` stores whole seconds plus nanos and round-trips (`A:130-162`), so a timer at `next_due_at` always finds the entry expired when it fires, and the `insert_arrival` retry loop (`Q:1852-1863`) terminates.
- Head-expiry classes index every lane root, blocked lanes included, in `expiring_roots`; other classes index explicit deadlines in `due_entries` (`A:768-779`, `894-911`).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| any | `insert_arrival` reaches Queued with `due_at` | `due_at < armed[c]` or disarmed | Armed(due_at) | - | `Q:1788-1790` |
| Armed(d) | `sleep_until(d)` fires in `run` | - | popped | `handle_update(None)`: `expire_due_classes`, `pop_due` disarms, `expire_class` rearms at next due | `Q:1560-1567`, `1888-1910` |
| Armed(d) | drain start / capacity wake / enqueue / refresh timer arm | `d <= now` | rearmed or Disarmed | as above | `Q:2081`, `2092`, `2100`, `2117`, `2318` |
| Armed | extraction or retraction in the class | class lost an entry | rearmed at `next_due_at` or Disarmed | `rearm_class_timer` | `Q:2244`, `2017-2019` |
| Disarmed | extraction or retraction | - | Disarmed | no-op (an unarmed class had no expiring entry) | `Q:1914-1919` |
| any | limit refusal with `next_due_at <= now` | - | rearmed | `expire_class` then retry the insert | `Q:1852-1863` |

### KQ-7 Actor turn, prepared rows, and worker-configuration generation

Location: `turn`, `turn_prepared` `Q:777-781`; `workers_generation: AtomicU64` shared with `SchedulerQueue` (`Q:775`, `823`); `PreparedAdmission` `Q:108`; `prepare_candidates` `Q:1478` (caller task); `select_worker_for_request` `Q:2387` (actor); `refresh_workers_generation` `Q:2535`; `workers_with_configs: watch::Receiver` `Q:773`; `PreparedRows` `lib/router/worker-selection/src/policy.rs:812`; kernel retain `lib/router/scheduler/src/selector/mod.rs:759-781`. The membership analyst's M7 is this machine.

States of a prepared row set: Built-on-caller (generation `g`, availability pointer `a`, overload pointer `o`, config snapshot at build), Carried-in-command, Held-for-turn (`turn_prepared`, only for the request enqueued in the current turn), Adopted (`into_current` ok, rows swapped into the policy), Rebuilt (`into_current` None, `has_changed`, full ranks, refreshed, or stale turn), Dropped (`begin_turn`).

Invariants:

- `begin_turn` runs once per command or wake, including every event inside a refresh (`Q:1605-1608`, `1541`, `2315`); rows prepared in an earlier turn are dropped (`Q:2217-2221`).
- Guards at selection, all under the actor's `workers_with_configs.borrow()` (`Q:2416-2429`): (1) `!workers.has_changed()` relative to the actor receiver; (2) `workers_generation` equal (the actor does `borrow_and_update().has_changed()` then `fetch_add` before every prepared admission, `Q:2525-2527`, `2535-2540`; the caller loads the generation with Acquire before borrowing, `Q:1484-1486`); (3) `available_worker_ids` equal (pointer or value; `AvailableWorkerIds` caches one conversion per source `Arc`, `types.rs:49-76`); (4) `overloaded_worker_ids` equal by value; (5) `prepared_rows_usable` (full-rank, affinity, transfer); (6) `accepts_prepared` (same inputs, no filters).
- Soundness: for rows built from version V to be adopted while the actor reads V+1, guard (1) requires the actor to have already called `borrow_and_update` after V+1 was sent, which made guard (2) bump the generation (Release) after that read; the caller's Acquire load of the pre-bump generation orders its own `borrow()` before the actor's `borrow_and_update`, so the caller cannot have seen V+1 with the old generation, and a caller that saw V with the old generation fails (2). The Relaxed load at `Q:2425` is by the single writer thread. Any publish between prepare and admit rebuilds, even when rows would be identical.
- Rows are unusable when the projection found full ranks and the request has an exclusive affinity target or KV transfer candidates (`Q:2416-2419`). After adoption `PreparedRows::retain` drops workers whose every rank is full (`selector/mod.rs:772-781`) and the policy refills `LOAD` from the actor's projection (`policy.rs:577-620`).
- Runtime policy updates republish `candidate_preparer` (`Q:1675-1678`); they cannot change `uses_exclusive_affinity_target` or `required_worker_inputs` (`policy.rs:270-280`), so `SchedulerQueue.exclusive_affinity` cached at construction (`Q:985`) cannot go stale.
- Worker watch (`L:198-244`): a config publish is visible to the actor's next `borrow()` immediately; the slot-tracker topology (`reconcile_worker_configs`, `L:152-177`, outcomes `Unchanged | Applied | Rejected`) lags until the monitor task runs, then an `Update` command forces a full recheck (`L:238-242`). Slots never lead configs (MB-6).
- Membership also reaches selection through `WorkerAvailabilityProvider` (`types.rs:37`): `None` disables the hard filter, `Some(empty)` rejects everything (`filter.rs:116-120`, `:235-239`); the frontend's provider is the admitted `Client` view (`lib/llm/src/kv_router.rs:859-891`), LoRA requests get a per-request set; since SM-9's fix the frontend's config map is admission-filtered upstream of the kernel (MB-4), so a `None` provider no longer admits a non-admitted candidate.

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| (caller) | `prepare_candidates` | channel slot reserved | Built-on-caller | loads `workers_generation` (Acquire), borrows config, builds rows | `Q:1478-1500` |
| Built | `Enqueue` applied | same turn | Held-for-turn | `turn_prepared = Some` | `Q:777-781` |
| Held | `select_worker_for_request` | guards (1)-(6) hold | Adopted | rows swapped in; `retain` drops full workers | `Q:2416-2429`, `selector/mod.rs:772-781` |
| Held | `select_worker_for_request` | any guard fails | Rebuilt | rows rebuilt from current inputs | `Q:2416-2429` |
| Held | `begin_turn` (new command, wake, or refresh event) | - | Dropped | `turn_prepared = None` | `Q:1605-1608`, `2315` |
| any | config publish | `borrow_and_update().has_changed()` | generation + 1 | `fetch_add` (Release) | `Q:2535-2540` |

### KQ-8 Capacity updates and blocked-lane recheck

Location: `CapacityUpdates` `queue/capacity.rs:16-61` (`workers: ArrayQueue<256>`, `full_recheck: AtomicBool`, `notify: Notify`); actor side `recheck_capacity_updates` `Q:2055-2078`; producers `SchedulerQueue::capacity_changed` `Q:1436-1440`, `direct_release` `Q:972-976`, `LocalScheduler::free` / `mark_prefill_completed` / `free_if_worker` `L:708-830`. States: Idle (empty, flag false, no permit), Hinted(set of workers), Overflowed (`full_recheck` true), Signalled (`Notify` permit stored or waiter woken). Lane side is AD-3.

Invariants:

- `record` stores a permit when no waiter exists (`Notify::notify_one`), so a wake that loses the actor's `select!` race is kept for the next turn (`capacity.rs:33-38`; test `notifications_survive_idle_overflow_and_consumer_handoff`).
- `drain_into` clears the flag before draining and re-notifies if anything remains (`capacity.rs:46-60`).
- A hint for a worker no longer in the config promotes to a full recheck (`Q:2064-2070`), so lanes pinned to a removed worker are re-evaluated; the gate then reports such a lane dispatchable (`Q:2768-2770`) and selection answers `NoEndpoints` (`selector/mod.rs:241-251`).
- Hints are consumed even with no pending work (`Q:2061-2063`); nothing is lost because no lane exists to be stale.
- Slot-tracker request expiry (ST-8) and `add_request_local`'s drop of `expired_request_ids` free load without recording a hint (NOTE kernel-queue-8): bounded staleness for a blocked exact lane until a later hint, a config change, or the periodic `Update` (`L:270-287`).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Idle / Hinted | `record(Some(w))` | - | Hinted + Signalled | push, or overflow sets flag | `capacity.rs:33-38` |
| any | `record(None)` | - | Overflowed + Signalled | flag | `capacity.rs:34` |
| Signalled | actor `notified()` in `run` | - | draining | `handle_capacity_updates(drain_cleanup?)`: expire, maybe cleanup, `recheck_capacity_updates`, `drain_ready` if ready | `Q:1546-1559`, `2080-2089` |
| Signalled | actor `notified()` in refresh | - | draining | `drain_cleanup`, `recheck_capacity_updates(cleanup_ready)` | `Q:2320-2323` |
| Hinted | `recheck_capacity_updates(false)` | - | Idle | `recheck_worker(w)` per hinted lane (unblocks lane) | `Q:2075-2077` |
| Overflowed / removed worker / `Drain::Full` | `recheck_capacity_updates(true)` | - | Idle | `recheck_all_workers()` sets the class flag | `Q:2071-2073` |

### KQ-9 Actor command loop and shutdown

Location: `SchedulerQueueActor::run` `Q:1536-1600`. States: Idle-waiting (`select!` on command, capacity, timer, or plain `recv` when neither side channel applies, `Q:1543-1545`), Applying (`apply_command`, `Q:1623-1693`), Draining (`drain_after_command`, `Q:1610-1621`: `None` reply now; `Enqueued` `handle_enqueued`; `Full` `handle_update`), Refreshing (KQ-5), Exited (`recv()` returned `None`, `Q:1570`, then cleanup drain and pending drain).

Invariants:

- A command's reply is sent exactly once: at `Drain::None`, before the first refresh await, or at drain end (`Q:1612-1616`, `2178-2180`, `2250-2252`); a command applied inside a refresh is replied at once (`Q:2332-2334`).
- `drain_ready` never sends into the command channel (test `continuation_drain_does_not_self_send_into_saturated_actor_channel`).
- The channel stays open while any `SchedulerQueue`, `RequestLifecycleLease`, `SchedulerBookingCleanup`, or `BookingHandle` lives (each holds an `mpsc::Sender`); the host's `RequestLeaseManager` holds one `SchedulerBookingCleanup` (`RL:88-92`) and its replica observer is deliberately weak (`RL:256-262`), so the actor cannot outlive the manager through a cycle.
- `RuntimePolicyControl` holds a `WeakSender` (`Q:415-419`) and answers `Unavailable` once the actor is gone.

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Idle | command received | - | Applying | `begin_turn`; `drain_cleanup` if due | `Q:1546-1559`, `1605-1608` |
| Idle | capacity wake | queueing enabled | Draining | KQ-8 row | `Q:1546-1559` |
| Idle | class timer deadline | some class armed | Draining | `handle_update(None)` | `Q:1560-1567` |
| Applying | `apply_command` done | `Drain::None` | Idle | reply now | `Q:1612-1616` |
| Applying | `apply_command` done | `Drain::Enqueued` / `Drain::Full` | Draining | `handle_enqueued` / `handle_update` | `Q:1610-1621` |
| Draining | candidate needs refresh | KQ-5 guards | Refreshing | reply sent before the first refresh await | `Q:2170-2185` |
| Draining | drain end | - | Idle | reply if not yet sent | `Q:2250-2252` |
| Idle | `recv() == None` | all senders dropped | Exited | `drain_cleanup()`, `pending.drain()` answered `SubscriberShutdown` | `Q:1570`, `1579-1599` |
| Refreshing | `commands.recv() == None` | - | Exited after settling the candidate | `stopping` set; candidate admitted or refused and settled first | `Q:2313`, `2210`, `2246` |

### KQ-10 Host-side lease consumers

Location: `RequestLeaseManager` / `LeaseClock` `RL:26-545`; `KvRequestCleanup` and `RequestGuard` `RG:428-969`; cancellation policy `lib/llm/src/kv_router/routing_host/cancellation.rs:20-141`.

States per `RequestLeaseRecord.clock`: `TOUCHED` (new, or progressed since the last scan), `QUIET` (one scan without progress), `CLAIMED` (terminal: finished, completed, expired, or remote-completed). `reap` (`RL:48-70`) is second-chance: TOUCHED -> QUIET, QUIET -> CLAIMED plus expire. `touch` only QUIET -> TOUCHED; `claim_now` is a swap returning whether it won. Index: `by_attempt: HashMap<AttemptId, _>` and `current_by_request` (only `register_detached`) (`RL:95-99`); `insert` with an identical booking touches; a different booking under the same attempt id is logged and ignored (`RL:103-125`).

Invariants:

- Every scheduler release through the manager is attempt-scoped, so lease drop, reaper expiry, explicit finish, and attempt release compose idempotently (test `attempt_release_frees_booking_when_another_path_owns_cleanup`).
- `KvRequestCleanup::new` (`RG:440-468`) commits the `BookingHandle` and registers the lease synchronously; nothing awaits between `plan_kv_route`'s selection result and this call (`routing_host/kv.rs:281-296`).
- All retry paths release synchronously before re-selecting (tests `stream_failure_releases_booking_before_error_is_observable`, `worker_overload_stream_migration_releases_and_reselects`); `DuplicateRequest` at booking would otherwise become a non-migratable `BookingFailed` (NOTE kernel-queue-9).
- Cancellation policy (`cancellation.rs:84-141`): `CancelWhenStopped` drops the in-flight selection future (whose scheduler call then drops its lease and `resp_rx`, triggering KQ-2 cleanup); `DispatchWhenStopped` (decode with staged KV) runs it under the shared `CleanupBudget` (120 s, `cancellation.rs:20`).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| (none) | `register_local` / `register_detached` / `register_remote` | - | TOUCHED | insert | `RL:265-300`, `304` |
| TOUCHED / QUIET | `RequestAttemptLease::finish` / `finish_request` | `claim_now` wins | CLAIMED | remove; `enqueue_acknowledged` (synchronous direct release), then LRU finish | `RL:338-393` |
| TOUCHED / QUIET | `RequestAttemptLease` drop / armed `DetachedRequestLeaseEnrollment` drop | `claim_now` wins | CLAIMED | remove; `scheduler.enqueue(desc)` (deferred) | `RL:325-336`, `507-516`, `535-542` |
| QUIET | reaper tick | CAS QUIET -> CLAIMED | CLAIMED | remove; `enqueue_expired(desc)` (local only) | `RL:234-247`, `222-232` |
| TOUCHED / QUIET | replica `completed(desc)` | matching record, `claim_now` | CLAIMED | remove only (remote `Free` already mirrored) | `RL:313-320` |
| CLAIMED | `RequestAttemptRelease::release` | `finish` lost | CLAIMED | `release_booking` then `enqueue_and_wait` (attempt-scoped, idempotent) | `RL:400-416`, `480-486` |
| any | `touch_request` / `touch_booking` / `progressed` | record present | TOUCHED if QUIET | - | `RL:296-300`, `308-312` |
| guard live | `complete` | - | finished | classifier complete, metrics, `finish` | `RG:907-915` |
| guard live | `abort` | - | finished | `finish` | `RG:920-925` |
| guard live | `release_for_retry` | migratable | finished (classifier lifecycle parked) | `finish` | `RG:927-939` |
| guard live | `Drop` | - | finished (deferred) | fields drop, `RequestAttemptLease` drop, deferred release | `RG:963-969` |

Kernel-queue interleavings traced and sound:

- Worker removal while queued or booked, refresh timeout versus completion, controlled-delay expiry versus extraction, double release, counter drift, mid-selection cancellation, and shutdown with queued work all resolve to idempotent or doubly-checked paths with existing tests (`test_cancelled_pending_request_is_not_booked`, `cleanup_progresses_with_admission_commands_and_capacity_wakes`, `runtime_control_drains_cleanup_when_the_command_channel_was_full`, `test_pending_requests_receive_shutdown_on_queue_drop`).
- The busy gate and the selector judge eligibility differently on purpose: a head that passes the gate and fails selection is rejected, not re-queued (`Q:2125-2143`, `2752-2780`, `2399-2440`; tests `test_overloaded_provider_filters_at_admission`, `hard_availability_provider_filters_unpinned_and_pinned_selection`, `test_disallowed_worker_ids_fail_without_queueing`).
- `resp_tx == None` is read as "already closed" by both the refresh close arm (`Q:2290-2294`) and `response_is_closed()` (`types.rs:554`); unreachable for queued entries because `select_without_admission` requests (`L:506-515`) never enter the queue.

## Admission (`dynamo-admission`)

Abbreviations: `A:` = `lib/router/admission/src/queue.rs`, `Q:` = `lib/router/scheduler/src/queue.rs`, `R:` = `lib/router/admission/src/classifier/runtime.rs`, `P:` = `lib/router/admission/src/profile.rs`. Drivers: the kernel actor (one task owns the queue) and the offline replay driver `lib/mocker/src/replay/offline/extensions/kv_router/mod.rs`. Every queue machine is single-threaded (mutated only by the owning actor turn); AD-6 is the only concurrent machine. Analyst numbering: M1 -> AD-1, M2 -> AD-2, M3 -> AD-3, M4 -> AD-4 (timers in KQ-6), M5 -> KQ-1, M6 -> AD-5, M7 -> AD-6, M8 -> AD-7, M9 -> AD-8.

### AD-1 Queue entry lifecycle (`PolicyQueue<T>`, `PolicyClassQueue<T>`)

Location: `PolicyQueueEntry` `A:220`, `PolicyClassQueue` `A:330`, `PolicyQueue` `A:684`. Implicit machine: the state is which collections hold the entry's `enqueue_seq`.

| State | Representation |
|---|---|
| Absent | in no collection (before `enqueue_with_due_at`; after extract, expire, remove, drain) |
| Queued-shared | in `class.pending` (placement `Any`) |
| Queued-exact-head-candidate | `peek()` of `class.ready_by_worker[w]`, `w` not in `blocked_workers`, `WorkerLaneHead{w, priority, seq}` in `candidate_worker_heads` |
| Queued-exact-head-blocked | `peek()` of `ready_by_worker[w]`, `w` in `blocked_workers`, not in `candidate_worker_heads` |
| Queued-exact-behind | in `ready_by_worker[w]`, not its head |
| +Deadline-indexed | `(due, seq)` in `class.due_entries` iff `priority.due_low & EXPIRES` (explicit `due_at` in a class with `head_expiry == false`, `A:894`) |
| +Expiring-root | `LaneRoot{due, seq, lane}` in `class.expiring_roots` iff `head_expiry` and the entry is a lane head with finite due (`A:373-384`) |
| Extracted | returned by `extract_next` with a `DispatchCharge`; in no collection; stats and `pending_count` already decremented (`A:1174-1193`); no requeue API exists |

Invariants (checked only by the test hook `assert_indexes_match_entries` `A:1332-1351`):

- `pending_count == sum over classes of (|pending| + sum over w of |ready_by_worker[w]|)`; `class.stats.requests == |class entries|`; `raw_isl_tokens` and `cached_tokens` are sums of the stored `QueueEntrySnapshot`s; the same stored snapshot is added at `A:911` and subtracted at `A:954`, `A:1026`, `A:1182`.
- `blocked_workers` is a subset of `keys(ready_by_worker)` (`A:457` debug_assert; maintained at `A:589`, `A:662-663`, `A:1010-1021`). Every non-blocked lane's head, and nothing else, is in `candidate_worker_heads`.
- `head_expiry` implies `due_entries` empty (`A:894`, `A:1343`); `expiring_roots` equals the set of finite-due lane roots, blocked lanes included.
- `enqueue_seq` unique and increasing (`u64::wrapping_add`, `A:910`); equal `QueuePriority` orders lower seq first (`A:180`).
- EDF class: `strict_priority := 0`, `policy_score := 0.0` (`A:1218-1223`, `A:1246`); ordering due is `due_at.or(class_due_at)` (`A:888`). FCFS, LCFS, WSPT: ordering due is `due_at`; `class_due_at` dropped (`A:889`).
- `DueKey.low < 10^9 < 2^31`, so `EXPIRES` (bit 31, `A:124`) never orders; `DueKey::NONE` sorts after every finite key (`A:137-140`); all keys share `deadline_origin` fixed at construction; a due before it keys as `(0,0)` and decodes as the origin (`A:142-157`).
- `QueuePriority` is 24 bytes and the entry header at most 64 (`A:1359-1366`).
- `next_due_at(c)` reads `expiring_roots.first()` (head expiry, blocked roots included) or `due_entries.first()` (`A:767-777`); `enforced_due_at(meta) = due_at.or(class_due_at iff head_expiry)` (`A:781-785`) equals what `take_expired` enforces. An EDF entry with neither deadline keys `NONE`, sorts last in enqueue order, and never enters `expiring_roots` (`A:379`). Expiry cannot touch an Extracted entry; the kernel watches its due separately (KQ-5).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Absent | `enqueue_with_due_at(meta, worker_count, jump, strict, placement, T)` (`enqueue` `A:834` is the no-deadline form) | `queue_rejection` is `None`: for requests, raw, cached: `limit.is_none() \|\| current < limit * worker_count` (`worker_count == 0` makes every finite limit 0, so reject) | Queued-shared / Queued-exact-* | insert `due_entries` iff explicit deadline and no head expiry (`A:907`); `next_enqueue_seq += 1`; `add_stats`; `push_ready` (indexes head only if lane not blocked; replaces the lane's expiring root); `pending_count += 1` | `A:863-915`, `412-448`, `1255-1287` |
| Absent | same | limit reached | Absent | `Err((QueueRejection{policy_class, limit_kind, current, limit}, T))`; no seq consumed | `A:881-883` |
| exact-head-candidate | `extract_next` / `pop_next` via `next_dispatchable_worker`, `is_dispatchable(head)` false | - | exact-head-blocked | `candidate_worker_heads.pop_last`; `blocked_workers.insert(w)`; continue with the next-best lane | `A:538-543` |
| exact-head-blocked | `recheck_worker(w)` | `w` in `blocked_workers` | candidate | current head re-indexed | `A:643-654` |
| exact-head-blocked | `recheck_all_workers()` then the class's next worker-lane pass | `needs_blocked_worker_recheck` | candidate iff `is_dispatchable(current head)`, else blocked | flag cleared on that pass; the shared-only fast path (`ready_by_worker.is_empty()`) does not consume it | `A:656-658`, `480-503`, `456-469` |
| exact-head-blocked | `push_ready(Exact(w), e')`, `e'` outranks head | lane blocked | `e'` is head, lane stays blocked (NOTE admission-5) | `candidate_worker_heads` untouched; expiring root replaced | `A:424-447`, `417-421` |
| Queued-* (winning head) | `extract_next` | DRR affordability (AD-2) | Extracted | `round_cursor = class + 1`; `pop_lane`: head popped, next head indexed, empty lane removed, root replaced; `subtract_stats`; `pending_count -= 1`; `remove_due_time`; `deficit = 0` if class empty; returns `(entry, DispatchCharge{class, cost = scheduling_cost_tokens})` | `A:1174-1193`, `568-641` |
| Extracted | `settle(charge, true)` | - | Extracted (caller's) | `deficit -= cost` (saturating); `deficit = 0` if class empty, else `carry_class = Some(class)` iff `deficit > 0` | `A:1161-1172` |
| Extracted | `settle(charge, false)` | - | Extracted (caller's); no re-entry | deficit unchanged; carry as above | `A:1161-1172` |
| Extracted | `DispatchCharge` dropped unsettled | - | Extracted | class keeps the quantum that made the head affordable; `carry_class` stays `None` (`#[must_use]` only) | `A:677-682` |
| Queued-*, head-expiry class | `take_expired(class, now)` | `expiring_roots.first().due <= DueKey(now)` | Absent (returned `Vec`, due order) | `pop_expiring_root(lane)`: unblocks a blocked lane, indexes next head or drops the lane, replaces root; `subtract_stats`; `pending_count -= 1`; `deficit = 0` if class empty; no DRR charge | `A:949-961`, `584-614` |
| Queued-*, no head expiry | `take_expired(class, now)` | `due_entries.range(..=(now, MAX))` non-empty | Absent | `take_if_entry_in_class(seq in expired)`: one scan, heap rebuild of the class | `A:934-947` |
| Queued-* | `take_if_in_class(class, pred)` / `retain(keep)` | a payload matches | Absent | `removed_ready_head` computed; blocked lanes whose root is removed are unblocked (`A:992-997`); `pending` and every lane drained and rebuilt, empty lanes dropped; `rebuild_worker_heads` (prunes `blocked_workers` to existing lanes); `rebuild_expiring_roots`; per removed: `subtract_stats`, `pending_count -= 1`, `remove_due_time`; `deficit = 0` if class empty; returns `(removed, removed_ready_head)` | `A:964-1034`, `823-829` |
| Queued-* | `drain(self)` | - | Absent | consumes the queue; no accounting | `A:1149-1156` |

### AD-2 DRR credit cycle

Location: `extract_next` `A:1053-1147`, `settle` `A:1161-1172`, `extract_candidate` `A:1174-1193`. Per-class `deficit`; queue-level `round_cursor`, `carry_class`. Class states: Empty (`ready_is_empty()`, `deficit` forced to 0 at `A:959`, `1031`, `1074`, `1090`, `1167`, `1185`), Backlogged-dispatchable (`next_dispatchable` is `Some`), Backlogged-blocked (entries present, candidate `None`: every lane head failed `is_dispatchable`).

Invariants:

- `deficit` grows only by `quantum` on a ring visit with a dispatchable, unaffordable head (`A:1101`) or by `quantum * rounds` in the bulk fast-forward over dispatchable candidates (`A:1124-1134`); a blocked class never accrues (`A:1089-1094`, `1116-1122`).
- Credit is spent only by `settle(.., true)`.
- `rounds = min over dispatchable candidates c of ceil((cost_c - deficit_c) / quantum_c)` (`A:1112-1122`); `quantum == 0` divides by zero there (validated only by YAML, `lib/router/kv-router/src/scheduling/policy_config.rs:341`; `PolicyProfile::from_classes` does not check it, NOTE admission-2).
- A refused head leaves its quantum with the class and the successor runs next (`policy_queue_tests.rs:1005`). An all-blocked return drops only the carry preference, not credit (NOTE admission-8).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| any | `extract_next` | `pending_count == 0` | return `None` | `carry_class = None` | `A:1057-1060` |
| `carry = Some(c)` | carried pass | `candidate_c.cost <= deficit_c` | extract c | `carry_class` taken | `A:1064-1071` |
| `carry = Some(c)` | carried pass | unaffordable or blocked | ring (c visited last since `round_cursor == c + 1`) | candidate cached in `candidates[c]`; `deficit_c = 0` if empty | `A:1073-1076`, `1097-1098` |
| ring at class i | visit | candidate `None` | next | `deficit_i = 0` if empty; no quantum | `A:1089-1094` |
| ring at i | visit | `cost <= deficit_i` | extract i | - | `A:1096-1099` |
| ring at i | visit | `cost > deficit_i` | `deficit_i += quantum_i`; extract if affordable, else next | - | `A:1101-1105` |
| ring exhausted | fast-forward | no dispatchable candidate | return `None` | deficits unchanged; carry already `None` | `A:1112-1122` |
| ring exhausted | fast-forward | some candidate | `deficit_i += quantum_i * rounds` for each dispatchable i; extract first affordable from `round_cursor` | - | `A:1124-1144` |
| extracted c | `extract_candidate` | - | `round_cursor = c + 1` | `deficit_c = 0` if c empty | `A:1179`, `1185` |
| extracted c | `settle(admitted)` | - | `carry_class = Some(c)` iff c non-empty and `deficit_c > 0` | `deficit_c -= cost` iff admitted | `A:1161-1172` |

### AD-3 Per-worker lane blocking

Location: fields `A:335-338`. States per (class, w): NoLane (`w` not in `ready_by_worker`), Candidate (lane exists, head indexed), Blocked (lane exists, `w` in `blocked_workers`). Class flag `needs_blocked_worker_recheck`. Choice per class: best non-blocked lane head versus dispatchable shared head by `cmp_queue_order`; a tie goes to the exact lane (`A:550-564`). The shared `pending` heap is never blocked and is re-evaluated on every call (`A:505-515`). Wake-ups come from KQ-8.

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| NoLane | `push_lane(Exact(w))` | - | Candidate | lane created, head indexed | `A:427-444` |
| Candidate | head fails `is_dispatchable` | - | Blocked | head de-indexed | `A:538-542` |
| Blocked | `recheck_worker(w)` | - | Candidate | head indexed | `A:643-654` |
| Blocked | `recheck_all_workers()` then next worker-lane pass | head dispatchable | Candidate | else stays Blocked | `A:491-502` |
| Blocked | root expires (`pop_expiring_root`) | - | Candidate or NoLane | unblock, index next head or drop lane | `A:589-608` |
| Blocked | root removed by `take_if_entry_in_class` | - | Candidate or NoLane | `A:992` unblock; `rebuild_worker_heads` | `A:992-1022` |
| Candidate | head extracted (`pop_lane_head`) | `debug_assert !blocked` | Candidate (next head) or NoLane | - | `A:616-640` |
| Blocked | `push_lane(Exact(w), e')` | - | Blocked (head may become `e'`) | not re-evaluated until a recheck | `A:429-445` |

### AD-4 Expiry indexes

Location: `head_expiry = class_budget_expiry && controlled_delay && queue_policy == Edf`, fixed per class at construction (`A:720`, `P:94-96`). `PolicyQueue::new` (replay) never head-expires; `with_class_budget_expiry` (kernel, `Q:906`) does. The kernel's timers over these indexes are KQ-6.

| Class kind | Orders by | Expiry index | `take_expired` removes |
|---|---|---|---|
| EDF with `controlled_delay`, online | `due_at.or(class_due_at)`, then seq | `expiring_roots` (finite-due lane roots) | overdue prefix of every lane, blocked included |
| EDF without controlled delay, or any EDF under `PolicyQueue::new` | same | explicit `due_at` in `due_entries` | explicit deadlines only; class due never expires |
| FCFS / WSPT (LCFS only in the synthetic class) | strict, explicit `due_at`, score, seq | explicit `due_at` in `due_entries` | explicit deadlines only |

### AD-5 RequestLifecycle phases

Location: `RequestLifecycle` `R:387-523`. State: `phase` in {Registered, Sent, Responding, Terminal}, `worker: Option`, `context_tokens: Option<usize>`. Hosts: `lib/llm/src/kv_router/routing_host/kv.rs:147-173` (abort on selection error, `prepare_retry` when migratable, `selected`); `RG:744` `sent`, `RG:810` `responding`, `RG:909` `complete`, `RG:922` `abort`, `RG:934` `prepare_retry` and store for migration. Kernel side: `L:429-473` (`cached_overrides` gate, `classify_with`), `L:491-503` (`begin_request`).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Registered | `selected(w)` | - | Registered | `worker = Some(w)` | `R:414-418` |
| Registered | `sent(w)` | - | Sent | `worker = Some(w)`; send `Sent` | `R:420-430` |
| Sent / Responding / Terminal | `sent` | - | unchanged | ignored | `R:421` |
| Sent | `responding()` | `worker.is_some()` (debug_assert) | Responding | send `Responding` | `R:432-445` |
| non-Terminal | `observe_output_tokens` / `observe_context_tokens` | - | same | `context_tokens` high-water; `progress_updater.fetch_max` | `R:451-474` |
| non-Terminal | `prepare_retry()` | - | Registered | `worker`, `context_tokens` kept | `R:476-480` |
| non-Terminal | `complete()` | `worker.is_some()` | Terminal | `finish_request_and_send(Completed{worker, context_tokens})`; `request_id` taken | `R:482-500` |
| non-Terminal | `complete()` | `worker.is_none()` | Terminal via `abort(None)` | `Aborted{worker: None}` | `R:486-489` |
| non-Terminal | `abort(err)` / `Drop` | - | Terminal | `Aborted{worker (Some if selected or sent), error}` | `R:502-516`, `519-523` |
| Terminal | any | - | Terminal | no-op | - |

### AD-6 Classifier runtime

Location: `R:98-385`. Shared state: `live_requests: Mutex<HashMap<id, LiveRequest{generation, overrides: Option, progress}>>`; `pending_terminals{counts: Mutex<HashMap<id, usize>>, delivered: Notify}`; `classifier: AsyncMutex<Box<dyn RequestClassifier>>`; unbounded `events`; `shutdown: CancellationToken`; `delivery: JoinHandle`.

Per-id states: NotLive; Live{gen, overrides None} (`begin_request` `R:272-304`, `DuplicateRequestId` if live); Live{gen, overrides Some} (write-back `R:259-268`); NotLive+TerminalPending (`finish_request_and_send` `R:364-378`: remove, `reserve`, send, all under the live lock); NotLive (`release` after `on_event` returns, `R:168-170`).

Invariants:

- A reused id never reaches `classify` before the prior lifecycle's terminal was delivered; overrides cache only onto the lifecycle the classification started under; events are never dropped except at shutdown or drop.
- Delivery task (`R:134-172`): `select!{biased; cancelled => break; recv}`; per event `select!{biased; cancelled => break; lock + catch_unwind(on_event)}`; panic logged; terminal implies `release(id)`. Shutdown mid-`on_event` drops the callback future and skips `release`; waiters observe `shutdown` (`R:346-349`). `Drop for RequestClassifierRuntime` aborts the task (`R:381-385`). Terminal reservations made after shutdown are never released (NOTE admission-4; bounded by requests alive at shutdown).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| any | `classify_with` | shutdown cancelled | `Err(Shutdown)` | - | `R:195` |
| NotLive | `classify_with` with id | - | `Err(LifecycleEnded)` | - | `R:205` |
| Live, overrides Some | `classify_with` | - | `Ok(request with overrides)` | plugin not run | `R:209-212` |
| Live | `lock_classifier_for(id)` | pending terminals for id | wait outside the lock, re-check under it, loop; shutdown gives `Err(Shutdown)` | - | `R:321-335`, `348` |
| Live, locked | generation check | mismatch | `Err(LifecycleEnded)` | - | `R:226-234` |
| Live, locked | `catch_unwind(classify(..))` | panic | `Err(Panicked)` | lock released by scope | `R:235-236` |
| Live, future running | `biased` shutdown | shutdown | `Err(Shutdown)` | lock already released | `R:240-244` |
| Live, future done | future panics or returns `Err` | - | `Panicked` / `Failed` | - | `R:245-247` |
| Live, future done | `classification_id` differs | - | `Err(InvalidMetadata)` | - | `R:249-253` |
| Live, same generation | write-back | - | Live{overrides Some}, `Ok` | - | `R:261-262` |
| gone or re-registered | write-back | - | `Err(LifecycleEnded)` | nothing cached onto the new lifecycle (a plugin that re-ids the request lands here, NOTE admission-9) | `R:264-266` |
| Live | `finish_request_and_send` | - | NotLive+TerminalPending | remove, `reserve`, send under one lock | `R:364-378` |
| TerminalPending | delivery `on_event` returns | not cancelled | NotLive | `release(id)`; waiters notified | `R:168-170` |

### AD-7 Class resolution and deadline derivation

Location: `ClassifyRequest::policy_class()` `classifier/mod.rs:167-172`; kernel `validate_classification` `Q:1296-1349`; `queue_metadata` `Q:1260-1294`, `Q:1343-1347`.

| Source | Resolution | Unknown name |
|---|---|---|
| classifier override | `resolve_class_index_strict` | `InvalidClassificationMetadata("unknown policy class ..")`, rejected before queueing (`Q:1330-1338`) |
| request's own class, no override | `resolve_class_index(Some)` (`P:213-223`) | `UnknownPolicyClassError` to `KvSchedulerError`, rejected before queueing (`Q:1339-1341`) |
| neither | `default_class_index` | - |
| plugin `queue_thresholds()` for an unknown name | default class's thresholds (`mod.rs:132-143`) | the request still fails unless the plugin sets a known class |

Deadline derivation: `due_at` only from the classifier (`set_due_at`); `class_due_at = final_class.slo_ttft_due_at(ingress_at)` = ingress + `slo_ttft_ms` (`P:109-113`), counted from router ingress so classifier and channel waits spend it; a `scheduling_cost_tokens` override replaces the snapshot's (0 rejected, `Q:1309`); raw and cached tokens recomputed at enqueue. Enforced due per AD-4; ordering due per AD-1. An arrival whose enforced due already passed is rejected at `Q:1713` and at each insertion attempt (`Q:1827`, `Q:1844`).

### AD-8 Config resolution

Location: `lib/router/kv-router/src/scheduling/policy_config.rs`. YAML (`deny_unknown_fields`) to `resolve()` (`:216-279`); `models.<name>` replaces the root profile wholesale (`:168-181`); no profile gives `PolicyProfile::synthetic` (`P:177-201`: one class `default`, quantum 1, no budget; LCFS allowed only here). Per class (`:336-404`): name `[A-Za-z0-9_.-]+`, `quantum > 0`, `queue_policy` default EDF with LCFS refused, `controlled_delay` requires EDF, fractions finite and non-negative, `slo_ttft_ms` to `NonZeroU32`. `PolicyProfile::from_classes` (`P:140-175`) checks only non-empty, unique names, and a default when there is more than one class. Parsed once into a `OnceCell` (`config.rs:903-919`), validated at startup (`config.rs:867-869`).

Admission interleavings traced and sound:

- Refusal after extraction: no requeue path exists; the refused request is answered (`DeadlineExceeded`, selection error, `QueueRejected`) or dropped when its receiver closed, and `settle(.., false)` leaves credit with the class (`Q:2224-2244`, `A:1161-1172`). Tests `refused_extraction_spends_no_credit_and_its_successor_runs_next` (`policy_queue_tests.rs:1005`), `stalled_refresh_expires_its_candidate_without_release_or_drr_credit` (`Q:7167`).
- Expiry while extracted: the entry is in no index; its own due is watched by the refresh select (`Q:2295-2299`) and re-checked at `Q:2550` and `Q:2655`. Tests `deadline_expiry_during_refresh_is_counted_once` (`Q:4131`), `another_class_expires_while_a_candidate_refresh_stalls` (`Q:7228`).
- Blocked-only class and DRR: no quantum accrues (`A:1089-1094`, `1116-1122`). Tests `fully_blocked_ring_returns_without_accruing_deficit` (`policy_queue_tests.rs:1849`), `drr_weights_progress_and_skips_blocked_classes_without_credit` (`:1607`).
- Stats versus contents: the same stored snapshot is added and subtracted on every path (`A:911`, `954`, `1026`, `1182`); `assert_indexes_match_entries` checks `requests`, `due_entries`, `expiring_roots`, `pending_count`; the kernel mirrors with atomics on the same events (`Q:1794-1797`, `2047-2052`, `2154-2165`).
- Timer invariant `armed[c] == next_due_at(c)` holds through arrival, expiry, extraction, cleanup, and the limit-refusal retry loop. Tests `class_timer_follows_its_queue_head_through_cancellation_and_refill` (`Q:7083`), `next_due_at_follows_each_class_through_insert_remove_and_refill` (`policy_queue_tests.rs:957`).
- Classifier panic or return after abort or id reuse: generation fence (`R:226-234`, `259-268`). Tests `lifecycle_ended_during_classification_is_rejected` (`R:1180`), `stale_classification_does_not_cache_onto_a_reused_request_id` (`R:1053`), `reused_id_while_waiting_for_classifier_lock_rejects_stale_classification` (`R:1104`).
- Shutdown with queued entries: `run` drains every entry with `SubscriberShutdown` after the channel closes (`Q:1582-1600`); a candidate mid-refresh at shutdown is admitted or refused and settled before the loop stops (`Q:2210`, `2246`). Test `test_pending_requests_receive_shutdown_on_queue_drop` (`Q:5358`).
- Deadline derivation: enforced due equals the expiry index on every class kind (`A:781-785` versus `A:770-773`); an explicit deadline wins even when later than the class due. Tests `the_final_class_decides_controlled_delay` (`Q:8093`), `edf_class_budget_orders_by_original_ingress_and_explicit_deadlines_override` (`Q:5702`).
- Replay scales per-worker queue limits by ranks (`workers.len() * dp_size`, mocker `:1073-1075`) while the kernel scales by workers (`Q:1825`); a parity concern for `dynamo-kv-replay-parity`, not a queue defect (NOTE admission-6).

## Slot tracker (`dynamo-slot-tracker`)

Abbreviations: `MW:` = `lib/router/slot-tracker/src/multi_worker.rs`, `RS:` = `lib/router/slot-tracker/src/replica_sync.rs`, `S:` = `lib/router/slot-tracker/src/single.rs`, `T:` = `lib/router/slot-tracker/src/topology.rs`, `RL:` = `lib/llm/src/kv_router/request_lease.rs`. Analyst numbering SM-1..SM-8 maps to ST-1..ST-8.

Deployment facts that select which machine runs (verified at the construction sites):

| Host | Constructor | `replica_worker_policy` | Tracker expiry | Periodic force-expiry task | Who expires bookings |
|---|---|---|---|---|---|
| Embedded frontend (`lib/llm`) | `lib/router/scheduler/src/selection/core/workers.rs:296-318` with `HostReplication` from `lib/llm/src/kv_router/embedded.rs:290-294` | `LazyRegister` | `None` (`request_leases` is `Some`) | not started | host CLOCK reaper `RL:211-224,520-545` via `expire_if_booking` |
| Standalone selection service (`SelectionCore` default) | same site, `HostReplication::default()` `lib/router/scheduler/src/selection/core/mod.rs:251-257` | `RequireRegistered` | `Some(300 s)` | started | `ActiveSequences::force_expiry` |
| Standalone slot-tracker HTTP service | `lib/router/kv-router/src/services/slot_tracker/registry.rs:114-130` | `RequireRegistered` | `Some(300 s)` | started | `ActiveSequences::force_expiry` |

Topology drivers: kernel `LocalScheduler::reconcile_worker_configs` calls `slots.reconcile_workers` plus `set_worker_capacities` on every worker-config watch change (`lib/router/scheduler/src/local.rs:152-174`, monitor task `207-247`, synchronous `apply_published_worker_configs` `321-329`); the HTTP service calls `register_worker` / `unregister_worker` (`registry.rs:206,237`); the mocker uses `upsert_worker` / `unregister_worker`.

### ST-1 Booking lifecycle (`ActiveSequences`, `RequestIndex`, `ActiveSequencesMultiWorker`)

Location: `MW:377-392` (`workers: ShardedRwLock<WorkerTable>`, `request_index: RequestIndex`, `prompt_registry`). Per-rank state `S:124-132` (`requests: HashMap<RequestId, RequestState>`, `prefill: PrefillLoadTracker`, `prefill_requests: usize`, `blocks: BlockTracker`). Per-request index `lib/router/slot-tracker/src/request_maps.rs:31-35` (`request_to_booking: DashMap<RequestId, {worker, attempt_id, has_lora}>`, `request_to_lora`, `AttemptIdMinter`). States are implicit per request id: "I" = index entry, "S" = `requests` entry in the rank's `ActiveSequences`, "P" = `prefill.prefills` entry.

| State | Representation |
|---|---|
| Absent | no I, no S |
| Index-only (transient) | I present, no S; only between `try_insert_request` (`MW:1554`) and `add_request_with_prefill_tracking` (`MW:1561`), both under the same `workers.read()` guard, and inside replica add between `RS:189` and `RS:220` |
| Prefilling, loaded | I, S with `prefill_completed = false`, P present (`track_prefill_tokens && hint.tokens > 0`, `S:231-244`) |
| Prefilling, unloaded | I, S `prefill_completed = false`, no P (`S:682-724` names this "cached") |
| Decoding | I, S `prefill_completed = true`, no P |
| Orphan (invalid) | S present, no I; reachable only through SM-14 since SM-5's fix |
| Freed | Absent again; attempt id never reused (`AttemptIdMinter::mint`) |

Invariants (asserted only under `cfg(test, debug_assertions)` by `S:159-186` and `prefill_tracker.rs:327-412`):

- `prefill.prefills.keys()` is a subset of `{r in requests | !prefill_completed}`; `prefill_requests == |{r | !prefill_completed}|`.
- `prefill_full_tokens_sum == sum of initial_effective_prefill_tokens` over `prefills`; `anchored_prefill == prefill_order.front()`.
- Every `RequestBlockChain` terminal is owned by exactly one `RequestState` (chains are not `Clone`, `block_tracker.rs:38-41`).
- I implies S is relied on but not checked; `I.worker` must name a slot present in `WorkerTable.index` (violations repaired lazily by `stale_request_not_found` `MW:1586-1612`).
- An index entry is removed only by the path that removed the sequence state, or by topology removal and stale-slot repair where slot and state are already gone; an index-only booking (an add in flight) is left alone by `free`, matching the replica `Free` handler (`RS:263-265`) (restated by SM-5's fix).
- Every mutation of a rank's `ActiveSequences` and the matching `prompt_registry` delta or load update happen under that rank's `sequences.write()` (`prompt_registry.rs:284-288`); all sites comply (`MW:1560-1572`, `1632-1640`, `1037-1050`, `1445-1456`, `RS:215-234`, `259-276`, `304-315`).
- `enqueue_publish_event` runs after the local mutation and outside the sequence lock (`MW:579-586`): wire order between two overlapping lifecycle calls on one router is not the lock order.
- Publisher gauges (`observe_load`) refresh on every applied mutation; shared scheduler load is published only when shared state changed (`RS:329-333`, `MW:1119-1134`).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Absent | `add_request*` / `add_request_impl` | slot for W in table; else lazy path (`lazily_register_worker`) creates it once (`MW:1543-1551`), strict path returns `WorkerNotFound` | Prefilling (loaded or unloaded) | mint A; `force_expiry` runs first inside the rank (may free expired ids, `S:213`); membership delta plus load into registry; publish load snapshot; enqueue `AddRequest` if `replica_sync` | `MW:835-858`, `1523-1584` |
| any with I | `add_request*` same id | `try_insert_request` occupied | unchanged | `Err(DuplicateRequest{worker})` | `MW:1553-1559` |
| Prefilling | `mark_prefill_completed(id)` (unfenced) | I present; slot present | Decoding | `prefill_requests -= 1`; `LoadReleased` publishes load, `PhaseChanged` gauges only; returns `Applied` only for `LoadReleased` | `MW:1082-1116`, `S:261-281` |
| Prefilling | `mark_prefill_completed_if_booking(id, W, A)` | booking == (W, A) rechecked under seq lock | Decoding | as above, plus always enqueue `MarkPrefillCompleted` (even `PhaseChanged`) | `MW:1136-1180` |
| Decoding / Absent | either mark | `prefill_completed` already true, or no S | unchanged | `Unchanged` / `NoChange`; `if_booking` variant publishes nothing | `S:267-272` |
| any | `mark_prefill_completed(id)` | I present, slot missing | Absent (booking deleted) | `stale_request_not_found` removes I, `Err(RequestNotFound)` | `MW:1090-1098`, `1586-1612` |
| Prefilling / Decoding | `free(id)` (unfenced) | I present, slot present, `seq.free` Some | Absent | release chain (`block_tracker.rs:198-265`), `prefill.remove`, `prefill_requests -= 1` if still prefilling; registry delta plus load; publish load; `remove_request_if_worker`; enqueue `Free` | `MW:918-965` |
| Index-only | `free(id)` | `seq.free` None | unchanged (the in-flight add completes to Prefilling) | nothing published; returns `NoChange` (SM-5 fixed) | `MW` `free_on_worker` |
| any with I, slot missing | `free(id)` | `Err(RequestNotFound)` from mutate | Absent | I removed; returns `Applied`; no `Free` published, no load snapshot | `MW:942-944`, `1586-1612` |
| Absent | `free(id)` | `worker_for` None | Absent | `Ok(NoChange)` | `MW:923-925` |
| any | `free_if_worker(id, W)` | `worker_for == W` (worker-fenced, not attempt-fenced; no production caller, NOTE slot-tracker-8) | as `free` | - | `MW:967-978` |
| Prefilling / Decoding | `free_if_booking(id, W, A)` / `expire_if_booking` | booking == (W, A) rechecked under seq lock | Absent | `seq.free`; registry; `remove_request_if_booking`; publish load; `Free` enqueued only for `Publish` visibility | `MW:980-1080` |
| any with I, slot missing | `*_if_booking` | booking matched | Absent | I removed; no load snapshot, no `Free` | `MW:1024-1035`, `1146-1150`, `1254-1258` |
| Prefilling / Decoding | `add_output_block(s)` | I; slot; `num_blocks > 0` | same, more output blocks | random output hashes appended; fractional decay on unique suffix; gauges only (never replicated or published) | `MW:1206-1276`, `S:312-340` |
| any with S | `force_expiry` (timer) | `expiry_duration` Some; >= 30 s since last check; `started_at < now - expiry` | Absent | frees expired ids; caller removes them from I by id, unconditionally (`remove_requests`) | `S:344-379`, `MW:1440-1495`, `1575-1576`, `RS:238-239` |
| any with S on rank W | worker removal (unregister, reconcile, upsert shrink) | - | Absent | slot dropped with all S; `remove_worker_requests(W)` on I; registry membership and loads removed; no `Free` published (NOTE slot-tracker-3) | `MW:777-795`, `T:236-301` |
| Absent | replica `AddRequest` | `router_id != self`; `replica_sync`; `try_insert_request` vacant; slot present (`LazyRegister` creates it) | Prefilling | registry delta plus load (no cleanup); expired ids removed from I; `observer.admitted(booking)`; batch effects: publish load, cleanup trie | `RS:181-243`; lazy slot `RS:199-201` |
| any with I | replica `AddRequest` same id | occupied | unchanged | silently ignored, first owner kept (test `replica_duplicate_add_preserves_first_owner_without_side_effects`) | `RS:187-193` |
| Prefilling / Decoding | replica `Free` | I present and `I.worker == event.worker`; slot present; `seq.free` Some | Absent | registry; `remove_request_if_booking`; `observer.completed`; effects: publish load, wake scheduler, cleanup | `RS:244-287` |
| Index-only / Absent | replica `Free` | no I, worker mismatch, or `seq.free` None | unchanged | silent return; I is not removed when `seq.free` is None (the local path matches since SM-5's fix; slot-tracker-6 settled) | `RS:245-250`, `263-265` |
| Prefilling | replica `MarkPrefillCompleted` | I matches worker; slot; accepted even with `replica_sync = false` (worker-origin marks) | Decoding | registry load replace; gauges; `LoadReleased` gives `observer.progressed` plus scheduler wake | `RS:172-174`, `288-324` |
| any | publish queue full or closed | `enqueue_event` Err | unchanged locally | newest event dropped, rate-limited log; peers diverge until expiry | `MW:579-656` |

### ST-2 Prefill load anchor (`PrefillLoadTracker`)

Location: `lib/router/slot-tracker/src/prefill_tracker.rs:214-325`. States per rank: Empty (`anchored_prefill == None`, `prefill_order` empty); Anchored{front, anchored_since, all_modeled} where `all_modeled = unmodeled_prefill_count == 0`. Counters `prefill_full_tokens_sum`, `modeled_prefill_ms`, `unmodeled_prefill_count` use `checked_sub(..).expect` (`:271-286`); unreachable by the ST-1 invariants (every `remove` follows an `insert` of the same id, ids unique in `prefills`).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Empty | `insert(r, state, decay_now)` | - | Anchored{r, decay_now} | sums += tokens; modeled ms or unmodeled count | `:231-256` |
| Anchored | `insert` | - | same anchor | sums += | `:231-256` |
| Anchored{front = r} | `remove(r)` | - | Anchored{next front, now} or Empty | `set_anchor_to_front(decay_now)` resets decay to the removal time | `:258-301` |
| Anchored{front != r} | `remove(r)` | modeled | same front, `anchored_since += r.duration`, capped at `decay_now` | credits the anchor with the completed work | `:290-296`, `303-311` |
| Anchored | `remove(r)` | unmodeled, non-front | unchanged anchor | - | `:290-293` |
| any | `snapshot().active_tokens_at(now)` | all modeled | `sum - completed(anchor, elapsed)` saturating | - | `:61-77` |
| any | same | any unmodeled | `sum - anchor_full + ceil(anchor_full * remaining_frac)` (`checked_sub(..).expect`) | panics only if sum excludes the anchor (unreachable: anchor in prefills) | `:79-100` |

### ST-3 Worker rank lifecycle (`WorkerTable`) and lazy slot registration

Location: `T:116-331`; registry loads `prompt_registry.rs:231-335`; gauges via `observe_worker_registered` / `observe_worker_removed`; kernel lazy path `MW:1497-1521`, `1541-1551`; replica lazy path `RS:199-201`. This is the owning machine for lazy slot registration, which the kernel-queue (KQ-1 booking row) and membership (MB-6) sections reference. States per `(worker_id, dp_rank)` slot: Unknown (no slot, not in `worker_ranges`); Authoritative (slot, `worker_ranges[worker_id]` covers the rank); Lazy (slot via `ensure_worker`, worker id not in `worker_ranges` or rank outside its range); Removed (slot dropped). Loads-table membership mirrors slots except in the two-phase window below.

Invariants:

- Two-phase application: topology mutations hold `workers.write()` only for the table swap (`MW:690-693`, `728-731`, `745-748`); `request_index` and `prompt_registry` are updated afterwards without the lock (`MW:777-783`). Lifecycle calls landing in that window see a slot-less booking (handled by `stale_request_not_found`) or, with `LazyRegister`, can recreate the slot before the deferred phase runs (finding SM-14).
- `reconcile_workers` makes `index` keys equal the ranks of `worker_ranges` exactly (lazy ranks dropped) (`T:273-301`); `upsert_worker` / `register_worker` touch only that worker's ranks and preserve unrelated lazy ranks (`T:236-271`).
- Removing a rank removes its request-index entries (`MW:777-783`, `request_maps.rs:150-159`) and publishes `observe_worker_removed`; bookings on removed ranks are gone, a later `free` returns `RequestNotFound`.
- A dp change for an existing worker keeps slots (and bookings) for ranks present in both old and new ranges (`T:261-266`, `291-294`).
- `has_registered_workers()` / `worker_ranges()` ignore lazy ranks; `project_worker_loads`, `active_blocks`, `num_workers` include them.
- dp-rank arithmetic: `dp_start + dp_size` guarded by `checked_add` in `validate()` for register, upsert, reconcile; `WorkerTable::new_with_expiry` panics (`expect`) on an invalid initial map (`T:140-149`; production callers pass empty maps, the mocker passes its config); `set_worker_capacities` uses `saturating_add`. Lazy slots take the event's `dp_rank` verbatim and bypass `validate()` and `MAX_DATA_PARALLEL_RANKS_PER_WORKER` (NOTE slot-tracker-10).
- The strict entry point `add_request_if_registered*` exists "so a request racing worker removal cannot lazily recreate the removed worker/rank" (`MW:818-833`, `lib/router/scheduler/src/local.rs:688-706`); hosts call it, and since SM-10's fix the kernel calls it when the selected worker is no longer listed in `workers_with_configs` at booking time, using the lazy `add_request_admitted` otherwise (residual window under KQ-1).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Unknown / Lazy | `register_worker(range)` | `validate()` ok; worker id not in `worker_ranges`, else `DuplicateWorker` | Authoritative; Lazy slots of this worker inside the range are kept with their bookings, outside it dropped | `reconcile_worker_slots`; `apply_worker_topology_change` plus gauges | `T:169-182`, `236-271`; `MW:689-706` |
| any | `upsert_worker(range)` | `validate()`; no-op if identical | as register but tolerates existing | unrelated lazy slots preserved | `T:184-195` |
| Authoritative | `unregister_worker(id)` | id in `worker_ranges`, else `WorkerNotFound` | Removed (all ranks) | bookings dropped (ST-1 worker-removal row); WARN "Removing worker" | `T:197-206`; `MW:726-741` |
| any | `reconcile_workers(ranges)` | each validates; no duplicate ids | exactly the target set; retained slots keep state; lazy slots not in target dropped | - | `T:208-224`, `273-301`; `MW:743-763` |
| Registered(dp n) | `upsert_worker` / reconcile with dp m | - | ranks in both kept; new added; extra removed | - | `T:184-195`, `236-271` |
| any | `register_worker` existing id | - | unchanged | `Err(DuplicateWorker)` | `T:175-179` |
| Unknown | `add_request_admitted` (kernel booking) | index miss, worker still listed in the kernel's config at the check | Lazy plus booking | `ensure_worker_registered_after_miss`, `observe_worker_registered` | `MW:1541-1551`, `1508-1521` |
| Unknown | `add_request_if_registered` | index miss | Unknown | `Err(WorkerNotFound)` | `MW:1545-1546` |
| Unknown | replica `AddRequest` | `LazyRegister` | Lazy | bypasses `validate()`; gauge registered | `RS:199-201`; `T:313-330` |
| Lazy | next `reconcile_workers` | rank not in any range | Unknown | bookings on it dropped, `remove_worker_requests`; no `free` | `T:273-301` |
| any | `set_worker_capacities` | rank in loads table | capacity recorded | ranks not in the table ignored | `MW:1349-1363`, `prompt_registry.rs:321-328` |

### ST-4 Replica sync (event path)

Location: outbound `enqueue_publish_event` `MW:579-656` into the publisher implementation: `ScopedSequencePublisher::enqueue_event` `try_send` into `host_tx` (embedded, capacity 100 000, `lib/llm/src/kv_router/sequence.rs:46,383`) or the ZMQ `outbound_tx` (`lib/router/scheduler/src/replication.rs:295-340`); then `run_replica_singleton_publisher` / `run_replica_batch_publisher` (`sequence.rs:177-278`) or the ZMQ PUB task (TR-1). Inbound per partition: `PeerManager` SUB socket (`lib/router/kv-router/src/services/common/replica_sync.rs:264-341`) or runtime subscriber / direct-ZMQ fan-in (`sequence.rs:423-487`), then `SlotTrackerRegistry::dispatch_replica_event` (`registry.rs:407-428`: drops self events and unknown partitions), `ReplicaInbox::deliver` (`replication.rs:457-498`: drops block-size mismatch at `debug`, drops on a full channel with a rate-limited warn), `run_replica_sync` (`RS:80-158`).

Event states: Enqueued; Published; Dropped(Full) (newest dropped, rate-limited error log); Dropped(Closed{during_shutdown}); Dropped(encode or publish error) (logged, no retry). No acks, sequence numbers, or replay (NOTE slot-tracker-9, transports-6).

Invariants:

- `run_replica_sync` batching: Idle (awaiting `next_event`) to Draining{batch_events, batch_start}; a batch ends on `MAX_REPLICA_BATCH_EVENTS = 256`, 1 ms elapsed, `Pending`, stream end, or cancel; each batch flushes effects once (`RS:328-349`: per-rank latest load to gauges, published only if `publish`; one scheduler wake; one trie cleanup attempt), then `record_drain`. A receive error inside a batch flushes and continues (`RS:134-137`); an error from `next_event` is logged and skipped (`RS:99-105`).
- Per-event guards (`RS:160-174`): `router_id == self.router_id` drops (echo); `!replica_sync && !MarkPrefillCompleted` drops. Decay anchoring for mirrored loads uses local receive time (`RS:176-178`).
- Peer set mutated by `/replica_sync/register_peer|deregister_peer` through a command channel (`replica_sync.rs:343-367,398-433`); reconnect handled by libzmq (`zmq.rs:226-233`). A restarted peer learns only events published after it subscribed; a `Free` for an unknown id is ignored.
- Lease observer handoff (embedded host): `observer.admitted` registers a remote CLOCK record (`RL:305-308`); `progressed` touches (`RL:310-314`); `completed` claims and removes (`RL:316-323`). A mirrored booking whose `Free` is lost is reaped after 1-2 scans of `active_request_expiry_duration()` (300 s default) via `expire_if_booking`, which publishes nothing (`RL:199-209`).
- An unfenced `free` or `RouterRequest::MarkFree` on a router that only mirrors a booking frees it on the owner too (`kv_router.rs:1642-1647`, `2006-2014`; `MW:953-959`; `RS:244-287`): intended replica-group semantics, which makes a wrong-id `MarkFree` destructive across the group (NOTE slot-tracker-11).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| local mutation | `enqueue_publish_event` | queue has room | Enqueued | `try_send` | `MW:579-656` |
| Enqueued | publisher task | - | Published | batch or singleton publish | `sequence.rs:177-278`; `replica_sync.rs:146-198` |
| local mutation | `enqueue_publish_event` | queue full or closed | Dropped | rate-limited log; peers diverge until expiry | `MW:600-644`; `sequence.rs:91-106`; `replication.rs:295-340` |
| peer inbox | `dispatch_replica_event` | self event or unknown partition | dropped | - | `registry.rs:407-428` |
| peer inbox | `ReplicaInbox::deliver` | `block_size` mismatch | dropped | debug log (NOTE slot-tracker-9b) | `replication.rs:464-476` |
| peer inbox | `ReplicaInbox::deliver` | channel full | dropped | rate-limited warn | `replication.rs:477-490` |
| Idle | `next_event` ready | - | Draining | batch opened | `RS:80-158` |
| Draining | 256 events, 1 ms, `Pending`, stream end, or cancel | - | Idle | flush effects once; `record_drain` | `RS:328-349` |

### ST-5 Standalone slot-tracker HTTP service

Location: `lib/router/kv-router/src/services/slot_tracker/{registry,server}.rs`. `TrackerEntry` per `RoutingPartitionId` (`registry.rs:98-131`): Absent, Attached (created by the first `/register`, with its replica task and periodic expiry), Detached (last worker unregistered: removed from `trackers`, `cancel_token.cancel()`; the `Arc` may still be held by in-flight handlers). Guards: `lifecycle_lock` serializes register and unregister; `is_attached` plus `continue` restart the loop when the entry was detached under them (`registry.rs:184-210`, `212-249`). `add_request`, `free`, `mark_prefill_completed`, `potential_loads` take no lock and run on whatever entry `trackers.get` returned. Arrival order is the only ordering guarantee (`server.rs:174-175`).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Absent | `/register` | `lifecycle_lock`; duplicate id rejected, block-size mismatch rejected | Attached | tracker, replica task, periodic expiry created; `register_worker` | `registry.rs:164-210` |
| Attached | `/unregister` last worker | `lifecycle_lock` | Detached | removed from `trackers`; `cancel_token.cancel()` | `registry.rs:212-249` |
| Attached | `/add` | - | ST-1 add | `add_request_if_registered` (`WorkerNotFound` / `RequestNotFound` 404, `DuplicateRequest` 409) | `server.rs:164-216` |
| Attached | `/prefill_complete` | - | ST-1 mark | `request_booking` then `mark_prefill_completed_if_booking`; on `Unchanged` republish via `publish_prefill_completed_if_booking`; 404 only when the booking is gone | `registry.rs:311-337` |
| Attached | `/free` | - | ST-1 free (unfenced) | unknown id gives `NoChange` and 200 (NOTE slot-tracker-13); a `/free` racing an unreturned `/add` is 200 `NoChange` and the booking lands afterwards, releasable later, per the arrival-order contract (SM-5) | `registry.rs:339-345`; `server.rs:259-289` |
| Attached | concurrent teardown under a handler | `is_attached` false | loop restarts | - | `registry.rs:184-210` |

### ST-6 Scheduler load publication and the `SchedulerLoad` wire

Location: `lib/llm/src/kv_router/routing_load.rs`; wire `lib/router/slot-tracker/src/wire.rs:97-128`; publisher `lib/llm/src/kv_router/sequence.rs:489-603`; consumer `lib/llm/src/pool_relay/load.rs:296-425`; `ActiveLoad` `lib/llm/src/kv_router/active_load.rs`.

`SchedulerLoadSender` states: Disabled (`wake_tx == None`, `routing_load.rs:218-225`; encode contexts); Enabled; Closed (`wake_tx.is_closed()`; counted as unexpected unless the context token is cancelled, `:244-249`). `PendingSchedulerLoads` (`:96-153`): QueuedOnly (`queued.len() < capacity`, `overflow` empty); Overflowing on the first enqueue past capacity (later commands coalesce per rank into `overflow`, first-appearance order kept); back to QueuedOnly only after `pop` drains `queued` and then the whole overflow as one batch.

Invariants:

- Wake protocol: one-slot `mpsc<()>`; a wake is sent only when the queue was empty before the enqueue (`:169-173`, `251-262`); the receiver pops until empty before awaiting (`:277-284`). No lost wake-up: `is_empty` plus `enqueue` are under one mutex; a stale token costs one extra loop.
- Consumer `KvWorkerMonitor` receives `SchedulerLoadSnapshot`s (`LoadObservation::Scheduler`, `lib/llm/src/discovery/worker_monitor.rs:259-286`); when the context's busy thresholds are routed (`RoutableThresholds::route`, `routing_load.rs:229-233`) the queue judges booked load itself and scheduler snapshots no longer move the decode latch (`DecodeSignals::EngineOnly`, `worker_monitor.rs:210-216`).
- Wire: `SchedulerLoad{worker_id, dp_rank, active_decode_blocks, active_prefill_tokens, group}`; `SchedulerGroup::{Standalone{scheduler_id}, ReplicaGroup{group_id}, Unknown(#[serde(other)])}`. One batch per second per router from `service.loads(None, None)` summed across partitions per rank; group is `ReplicaGroup{endpoint id}` with replica sync on, else `Standalone{random}`. Consumer: per publisher keep the latest sequence; views expire after `SCHEDULER_FRESHNESS`; per rank `ReplicaGroup` views take the max, `Standalone` views the latest `published_at`; groups sum; `Unknown` ignored; total is `Complete` only when every discovered publisher reported every declared rank, else `Partial` / `Missing`.
- `ActiveLoad`: `active_decode_blocks` / `active_prefill_tokens` optional (scheduler-origin), `kv_used_blocks` optional (worker-origin); absence means "not reported", not zero.

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Enabled, queue empty | `enqueue(snapshot)` | - | QueuedOnly, wake sent | one `()` into the one-slot channel | `routing_load.rs:169-173`, `251-262` |
| QueuedOnly | `enqueue` | `queued.len() < capacity` | QueuedOnly | append | `:96-153` |
| QueuedOnly | `enqueue` | at capacity | Overflowing | coalesce per rank into `overflow` | `:96-153` |
| Overflowing | `pop` drains `queued` | - | QueuedOnly after the overflow batch | overflow emitted as one batch | `:96-153` |
| Enabled | `wake_tx.is_closed()` | context not cancelled | Closed | counted unexpected | `:244-249` |

### ST-7 Fullness rule (`BusyThresholds`, `FullRankCounts`)

Location: `lib/router/slot-tracker/src/busy.rs`. `has_room(load, capacity)` (`:61-79`): full when `tokens > prefill_busy_threshold`, or `tokens > frac * prefill_budget`, or `blocks > frac * total_kv_blocks` (strict `>`; a rank without `total_kv_blocks` skips the decode rule; an unconfigured rule never marks full). `load` is the booked projection (`active_decode_blocks`, not `potential_decode_blocks`). `FullRankCounts` (`:129-171`) is per projection pass: `reset(limits)` at pass start (`lib/router/scheduler/src/queue.rs:2350-2383`), `record` per rank, `worker_is_full(id, dp_size)` iff `count >= dp_size`; `clear()` when an in-place refresh diverged mid-pass (`prompt_registry.rs:425-437`). `all_eligible_workers_full` re-reads live loads through `WorkerLoadReader` (`queue.rs:2752-2780`). Thresholds come from `LoadThresholdConfig` (`worker_monitor.rs:720-731`) and can change between passes; `is_configured() == false` clears counts (`queue.rs:2365-2370`). A lazy rank outside a configured worker's range is counted toward `worker_is_full(id, dp_size)` (`:149-158`, `:166-170`; NOTE slot-tracker-10).

### ST-8 Expiry timers

| Timer | Period | Guard | Effect | Where |
|---|---|---|---|---|
| `ActiveSequences::force_expiry` | on every `add_request` and from the periodic task; evaluates at most every 30 s per rank | `expiry_duration` Some; `started_at < now - expiry` (absolute age, default 300 s, env `DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS`) | ST-1 `force_expiry` row; a live request older than the limit is dropped from accounting while it runs (NOTE slot-tracker-7) | `S:41-43,344-379`; `MW:59-91` |
| `start_periodic_force_expiry_across_all_workers` | 60 s, `MissedTickBehavior::Skip` | started only when the host has no request leases | sweeps every rank | `MW:1440-1495` |
| host CLOCK reaper | `active_request_expiry_duration()` scans | touched, quiet, claimed over two scans | `enqueue_expired` then `expire_if_booking` (local only) | `RL:26-73,211-224,518-545` |
| prompt-trie cleanup | >= 5 min since last completed sweep, one in flight | `try_schedule` CAS | unlink empty trie nodes | `lib/router/slot-tracker/src/cleanup.rs:33-116` |
| publish failure log limiter | 30 s per class | - | one log per interval with suppressed count | `MW:48-49,200-253` |

Slot-tracker interleavings traced and sound:

- Free for a never-added or already-freed id (local, fenced, replica, HTTP): quiet no-op on every path; attempt fencing prevents freeing a replacement (`stale_attempt_cleanup_cannot_free_reused_request_id`, `attempt_expiry_is_local_and_fenced_from_replacement_state`).
- `prefill_complete` after free: `Unchanged` / `NoChange` / 404 everywhere; `prefill_requests` cannot underflow (`S:270-273`, `294-297`).
- Replica add after free, duplicate add, free, and mark: `replica_duplicate_free_applies_side_effects_once` (`MW:3493`), `replica_duplicate_add_preserves_first_owner_without_side_effects` (`MW:3452`), `replica_sync_free_rejects_stale_worker`, `zmq_replica_sync_propagates_request_lifecycle` (`replica_sync.rs:588`). `Free` before `AddRequest` arose only from SM-5, now fixed.
- Shared-prefix accounting when one of two requests frees: `BlockTracker::release` releases only the dead suffix and reports `remove_from` (`block_tracker.rs:198-265`), `remove_path` truncates the worker's coverage there (`prompt_membership_trie.rs:348-402`); `randomized_lifecycle_matches_reference_model` (`block_tracker.rs:645`), `test_prompt_membership_delta_only_reports_first_add_and_last_remove` (`S:446`), `block_membership_index_matches_naive_loads_with_output_blocks_and_prefill_updates` (`MW:2749`). Trie and loads update as two steps, so a concurrent read can mix them (`prompt_registry.rs:242-249`).
- Potential versus actual load after rejections: the kernel books only after selection and rolls back an undelivered booking with the fenced release (`queue.rs:2632-2697`); projections are per request and never persisted (`prompt_registry.rs:27-33`).
- Counters checked for negative or saturating paths: `prefill_requests` (guarded by `prefill_completed`), `terminal_owners` (`checked_add` / `checked_sub` plus expect; chains single-owner), prefill sums (`checked_sub` plus expect), `prompt_total` / `output_total` f64 with `.round() as usize` (saturates at 0), `FullRankCounts` u32, `RateLimitedFailureLog` saturating, `SchedulerLoadSnapshot` `usize as u64`. None reachable to a wrong value through the public API.
- Scheduler-load wake protocol: no lost wake-up (ST-6); overflow coalescing keeps the latest snapshot per rank; tests `saturated_channel_*`.

## Session affinity (`dynamo-session-affinity`)

Abbreviations: `SA:` = `lib/router/session-affinity/src/lib.rs`, `RUN:` = `lib/router/scheduler/src/selection/core/run.rs`, `RH:` = `lib/llm/src/kv_router/routing_host.rs`, `CO:` = `lib/llm/src/session_affinity/coordinator.rs`. Analyst numbering M1..M11 maps to SA-1..SA-11. Two hosts drive the same table type with different contracts: the frontend `RoutingHost` has one `AffinityCoordinator` per routing host (prefill and decode hosts each have their own table and replication subject); with KV routing the coordinator wraps the embedded partition's table (`lib/llm/src/kv_router/embedded.rs:392-403`) and the core itself runs with `SessionBinding::None` (`lib/llm/src/kv_router.rs:1399`), so only the host holds a session. The standalone `SelectionCore` (router service and Python `KvRouter`) has one table per partition `SelectionEntry` (`lib/router/scheduler/src/selection/core/mod.rs:137`), held by the core on `select_and_reserve` / `create_reservation` (`SessionBinding::Managed`).

### SA-1 Table entry lifecycle (`AffinityEntry`)

Location: enum `SA:127-139`; `Inner.entries: DashMap<String, AffinityEntry>` `SA:174`; `Inner.entry_count` `SA:179`. Fields: `revision` identifies one initialization epoch (monotonic per table, `SA:436,472`); `version: AffinityVersion{sequence, writer_id}` identifies one binding for replication (`SA:752-757`); `target: WorkerAffinityTarget{worker_id, dp_rank: Option}`.

| State | Representation |
|---|---|
| Absent | no key in `entries` |
| Initializing | `AffinityEntry::Initializing{revision, notify}` |
| Bound-Active | `Bound{active_leases > 0, ..}` (never expires) |
| Bound-Idle | `Bound{active_leases == 0, idle_deadline > now}` |
| Bound-Expired | `Bound{active_leases == 0, idle_deadline <= now}`; still present and counted in `entry_count` until the reaper runs, but reclaimed by `expire_idle` before a capacity refusal (SM-7 fixed) |
| Bound-Joined (sub-state of Bound-Active) | `Bound` produced by `join_initializing`; `active_leases` includes one reserved use for the still-live `AffinityInitialization` of the same `revision` (`SA:551-558`) |

Invariants:

- `entry_count == entries.len()`: every insert is preceded by `reserve_entry`, every remove decrements (`SA:431,778` reserve; `SA:730,944,1080` decrement). In-place replacements (expired Bound to Initializing `SA:474`, Initializing to Bound `SA:551,896`, expired Bound to Bound `SA:799`) do not touch the count.
- An `Initializing` entry is removed only by its own `AffinityInitialization::drop` (`SA:926-945`); `expire_idle` and `invalidate` only remove `Bound` (`SA:724-726`, `1056-1062`).
- `active_leases` equals the number of live `AffinityLease` values with this `revision` plus one reserved use while a joined initializer is alive and uncommitted.
- Expiry is read-exact (`try_acquire` / `query_target` compare `idle_deadline <= now`, `SA:467,622`) but collection is lazy (`expire_idle` every `min(ttl, 30s)`, `SA:376`; manual clock `SA:353-360`).
- TTL is sliding: `idle_deadline = now + ttl` is written on commit (`SA:901`), join (`SA:557`), lease release when the lease's version is still current (`SA:1037-1040`), and replica Refreshed / ReplacedNewer / ReplacedExpired (`SA:804,815,826`); not on acquire of a Bound entry (`SA:489-515`) or `query_target`.
- No eviction exists; `max_entries` is admission-only (`reserve_entry`); a session with an in-flight request is never evicted by the limit. A failed `reserve_entry` runs `expire_idle(now)` after the DashMap shard guard drops and retries the reservation once, so only live entries fill the table (SM-7 fixed; cost is a full-table retain on the refusal path only); `try_acquire` wraps `try_acquire_at(.., now)` and `Inner::apply_replica_update` wraps `apply_replica_update_at`; reaper cadence unchanged.
- `validate_bound_target` (`SA:1097-1122`, acquire time): worker must match; `(Some(b), Some(r))` must be equal; `(None, Some(r))` rejected ("worker-only affinity cannot add DP rank"); `(Some, None)` allowed. `validate_dispatch_target` (`SA:1125-1152`, Hard commit): worker must match; a ranked binding requires the same rank and rejects a rank-less dispatch; a worker-only binding accepts any rank. `rebound_target` (`SA:1002-1009`, Soft commit): worker-only stays worker-only on the dispatched worker; ranked follows a ranked dispatch; ranked is unchanged by a rank-less dispatch, even to another worker (NOTE session-affinity-13).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Absent | `try_acquire` | `reserve_entry` ok | Initializing(rev = n) | count + 1; returns `Hold::Initialize` | `SA:430-452` |
| Absent | `try_acquire` | live entries >= max after `expire_idle(now)` and one retry of `reserve_entry` | Absent | `Err(ResourceExhausted)` | `SA` `try_acquire_at`, `reserve_entry` |
| Absent | `apply_replica_update` | reserve ok | Bound-Idle(version = v, leases = 0) | count + 1; `Inserted` | `SA:777-790` |
| Absent | `apply_replica_update` | live entries >= max after `expire_idle(now)` and one retry | Absent | `RejectedCapacity` (event lost) | `SA` `apply_replica_update_at` |
| Initializing | `try_acquire` (any requester) | - | Initializing | returns `AcquireStep::Wait(notified)`, registered before the shard lock drops | `SA:454-461` |
| Initializing(rev n) | `AffinityInitialization::commit(t)` | entry still `Initializing` with rev n; `validate_bound_target(t, requested)` ok | Bound-Active(leases = 1, version = new) | `notify_waiters`; publish `(t, version)` | `SA:885-914` |
| Initializing(rev n) | `join_initializing(t)` | entry `Initializing` | Bound-Joined(leases = 2, version = new) | `notify_waiters`; publish; returns the joiner's lease | `SA:539-570` |
| Initializing(rev n) | initializer dropped uncommitted | rev matches | Absent | count - 1; `notify_waiters` | `SA:926-946` |
| Initializing | `apply_replica_update` | - | Initializing | `IgnoredInitializing` (event lost) | `SA:792` |
| Initializing | `query_target` | - | Initializing | `Ok(None)` | `SA:613-621` |
| Bound-Joined(rev n) | initializer `commit(t)` | entry `Bound` rev n; Hard: `validate_dispatch_target(bound, t)` ok | Bound-Active (leases unchanged: the reserved use becomes the initializer's lease) | `SessionAffinity::commit(Hold::Bound)`: Soft may `rebind`; publish | `SA:858-884` |
| Bound-Joined(rev n) | initializer `commit(t)` | Hard and `validate_dispatch_target` fails | Bound (leases - 1 via drop) | `Err(InvalidArgument)`; drop decrements (`SA:937`) (finding SM-12) | `SA:868-869,918-948` |
| Bound-Joined(rev n) | initializer dropped | rev matches | Bound (leases - 1) | no notify of binding change; the decrement has no zero guard (NOTE session-affinity-7, unreachable) | `SA:929-939` |
| Bound-Idle / Bound-Active | `try_acquire(requested)` | `validate_bound_target(target, requested)` ok | Bound-Active(leases + 1) | returns `Hold::Bound{target, lease(rev, version)}` | `SA:489-515` |
| Bound | `try_acquire(requested)` | bound and requested disagree (worker, rank, or worker-only versus ranked) | unchanged | `Err(InvalidArgument)` | `SA:496,1097-1122` |
| Bound-Expired | `try_acquire` | - | Initializing(rev = n + 1) | in place, count unchanged; `Hold::Initialize` | `SA:463-488` |
| Bound-Expired | `query_target` | - | unchanged | `Ok(None)` | `SA:622-624` |
| Bound-Expired | `apply_replica_update(t, v)` | - | Bound-Idle(t, v, new rev) | `ReplacedExpired` (ignores version order) | `SA:793-807` |
| Bound-Expired | reaper tick / `advance_clock` reap | - | Absent | count - 1 | `SA:721-731` |
| Bound(t, v) | `apply_replica_update(t, v2)` | `v2 >= v`, same target | Bound(t, v2), deadline refreshed, leases kept | `Refreshed` | `SA:808-817` |
| Bound(t, v) | `apply_replica_update(t2, v2)` | `v2 > v` | Bound(t2, v2), deadline refreshed, leases kept, rev kept | `ReplacedNewer`; local leases at v become stale (SA-3) | `SA:818-828` |
| Bound(t, v) | `apply_replica_update(t2, v2)` | `v2 <= v`, target differs | unchanged | `IgnoredConflict` | `SA:829` |
| Bound-Active(rev n, v) | lease release | rev matches, leases > 0, `lease.version == v` | leases - 1; if 0 then Bound-Idle with refreshed deadline | publish `(target, v)` | `SA:1011-1044` |
| Bound-Active(rev n, v2) | stale lease (v != v2) release | rev matches | leases - 1, deadline not refreshed | no publish | `SA:1033-1039` |
| Bound(rev n, v) | `lease.invalidate()` | rev and version match | Absent | count - 1; other holders' leases become no-ops | `SA:1056-1081` |
| Bound(rev n, v2) | stale `lease.invalidate()` | version or rev mismatch | falls through to `release()` | - | `SA:1085` |
| Bound(rev n, v) | Soft `commit(Hold::Bound)` dispatched elsewhere | `rebind`: rev, version, current target unchanged since acquire | Bound(rebound target, new version) | publish new | `SA:583-590,973-997` |
| Bound | Hard `commit(Hold::Bound)` dispatched elsewhere | `validate_dispatch_target` fails | via `invalidate`: Absent (or release if stale) | `Err(InvalidArgument)` | `SA:592-596` |
| any | `Inner` dropped | last `SessionAffinity` clone gone | - | `cancel.cancel()` stops the reaper; handles' `Weak` fail (`Dropped`) | `SA:191-195` |

### SA-2 `AffinityInitialization` handle

Location: `SA:837-948`. States: `active == true` (armed), `active == false` (committed or handed over). Invariant: exactly one of commit-success, commit-failure-then-drop, or drop releases the reserved table slot or use.

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| armed | `commit(t)` | requested target mismatch | armed (then drop) | `Err`; drop removes Initializing or decrements the joined use | `SA:849` |
| armed | `commit(t)` | table dropped | armed | `Err(Dropped)`; drop is a no-op | `SA:850-852` |
| armed | `commit(t)` | entry missing or rev changed | armed | `Err("cancelled" / "changed")` (unreachable: nothing else removes Initializing) | `SA:853-857,885-894` |
| armed | `commit(t)` | entry Bound rev n (joined), Hard mismatch | armed | `Err`; drop: leases - 1 | `SA:868-869` |
| armed | `commit(t)` | entry Bound rev n (joined), ok | handed over (`active = false`) | returns `commit(Hold::Bound)` result; the lease owns the reserved use | `SA:871-883` |
| armed | `commit(t)` | entry Initializing rev n | committed | Bound-Active(leases = 1); notify; publish | `SA:895-914` |
| armed | drop | - | released | remove Initializing (count - 1) or leases - 1 on joined Bound; `notify_waiters` (skipped if table gone, `SA:923-925`) | `SA:918-948` |

### SA-3 `AffinityLease` handle

Location: `SA:952-1094`. Fields `revision`, `version` snapshot the entry at acquire or commit. States: active-current (entry version == lease version), active-stale (entry moved to a newer version via ReplacedNewer or a sibling's Soft rebind), active-orphaned (entry removed by a sibling's `invalidate` or re-initialized under a new revision), released.

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| active-current | drop / `release` | - | released | leases - 1; deadline refresh; publish | `SA:1011-1044` |
| active-stale | drop | rev matches | released | leases - 1 only | `SA:1033-1039` |
| active-orphaned | drop | entry missing, rev differs, or leases == 0 | released | nothing | `SA:1020-1035` |
| active-current | `invalidate` | - | released | entry removed, count - 1 | `SA:1056-1081` |
| active-stale / orphaned | `invalidate` | - | released | behaves as `release` | `SA:1085` |
| active (Soft commit) | `rebind(expected, t)` | CAS on (rev, version, target) | active-current at new version | entry target and version updated; publish by caller | `SA:973-997` |
| active (Hard commit of `Hold::Bound`) | `commit` | dispatch matches | active | publish `(bound, lease.version)` (stale version if ReplacedNewer landed in between: peers ignore it) | `SA:597` |

### SA-4 Per-request acquire protocol (`AcquireStep`, `Hold`)

Location: `SA:218-248`; `try_acquire` `SA:422-518`; `acquire` `SA:522-533`; frontend `acquire_with_context` `CO:119-144` (select on `stopped` / `killed`); frontend `acquire_affinity_slot` chooses between the cancellable wait and a budgeted uncancellable wait for staged-KV decode legs (`RH:586-613`). States per request: no-session (affinity disabled or no `SessionAffinityId` in context), waiting (`AcquireStep::Wait`), holding-Initialize, holding-Bound(target), committed (lease parked in `RequestGuard`, `request_guard.rs:751-753`), released. Waiters serialize behind the initializer for the initializer's whole select, queue, and dispatch (frontend) or select and booking (core); the wait is outside every deadline and queue budget (finding SM-15). The core's failover path is the only place that joins instead of waiting (`RUN:642-666`).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| start | `try_acquire` | entry Absent or Expired | holding-Initialize | - | `SA:430-452,463-488` |
| start | `try_acquire` | entry Initializing | waiting | - | `SA:454-461` |
| waiting | notify (commit, join, or initializer drop) | - | start (retry loop) | - | `SA:527-532`; `CO:126-143` |
| waiting | request stopped or killed (frontend, cancellable path) | - | `Err(Cancelled)` | - | `CO:134-137` |
| waiting | budget exhausted (frontend, staged-KV decode) | - | `Err` | - | `RH:604-612` |
| start | `try_acquire` | entry Bound live | holding-Bound | - | `SA:489-515` |
| holding-* | selection fails | - | released | drop: Initialize removed and waiters woken; Bound lease released, binding kept | `RH:680-695` |
| holding-Bound | host decides the target is invalid | - | `Hold::invalidate()` then re-acquire | - | `RH:666-678,683-694`; `RUN:575-615` |
| holding-* | dispatch accepted | - | committed | - | `RH:497-517`; `RUN:514-527` |
| committed | response stream ends or guard dropped | - | released | - | `request_guard.rs:963-968` |

### SA-5 Replica agreement

Location: version ordering `SA:103-108` (sequence, then writer_id); outcomes `SA:115-125`; `apply_replica_update` `SA:764-832`; sequence clock `SA:302-307,752-762`. Carriers: frontend `ReplicaSyncRuntime` (`lib/llm/src/session_affinity/replica_sync.rs`: sink `try_send` into a 4096 channel, drops on full `:48-64`; publisher task `:218-236`; applier skips own publisher id `:76-78`, observes the sequence `:83`, drops events whose worker is not in `client.instance_source` `:84-91`, applies `:93-100`; writer id = discovery instance id `:123`; installed once per coordinator `CO:76-98`). Core `AffinityMeshSink` (`lib/router/scheduler/src/replication.rs:37-100`, `try_send`, drop on full `:75-85`) into the peer mesh (TR-1; `services/common/replica_sync.rs:28,159-166,414-423`) or the host's `session_binding_updates` channel (`service.rs:170-172,506-535`); inbound `dispatch_affinity_event` skips own `process_id`, observes the sequence, drops events for workers outside the partition catalog (`mod.rs:469-511`).

States of one session across replicas: unknown, locally-initializing, bound(t, v) per replica. Agreement rule: highest `(sequence, writer_id)` wins; same target with equal or higher version refreshes; a replica that is initializing ignores the event. Two frontends binding one session concurrently: F1 commits (s, w1, v1) and publishes; F2 is Initializing and ignores it; F2 commits (s, w2, v2) with `v2 > v1` because `observe_replica_sequence` ran before the entry match (`SA:773`); F1 applies ReplacedNewer. Converges to w2; F1's in-flight request stays on w1 for its lifetime; F1's lease release at v1 neither refreshes nor republishes (`SA:1037-1039`). Replication is best effort; there is no unbind event (NOTE session-affinity-9).

| From (local) | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| any | local commit / join / Soft rebind / lease release | sink installed | same | publish `(session, target, version)` | `SA:913,568,588,1043` |
| any | sink channel full or closed | - | same | event dropped (best effort) | `replica_sync.rs:56-63`; `replication.rs:75-98` |
| unknown | peer event | worker discovered locally; capacity | bound(t, v) idle | - | `SA:777-790` |
| unknown | peer event | worker not discovered locally | unknown | event lost; sequence still observed | `replica_sync.rs:84-91`; `mod.rs:491-503` |
| locally-initializing | peer event | - | unchanged | lost | `SA:792` |
| bound(t, v) | peer (t, v' >= v) | - | refreshed | - | `SA:808-817` |
| bound(t, v) | peer (t', v' > v) | - | bound(t', v') | in-flight local leases become stale | `SA:818-828` |
| bound(t, v) | peer (t', v' <= v) | - | unchanged | - | `SA:829` |
| bound | local `invalidate` | - | removed locally only | no unbind event; peers self-heal on their next acquire | `SA:1048-1087` |

### SA-6 Frontend per-request affinity decision

Location: `select_with_session_affinity` `RH:615-696`; binding key `RH:470-490`; `accept_dispatch` `RH:497-517`; `affinity_target_is_valid` `RH:557-574`; explicit target `CO:282-305`. Binding key: an explicit per-request target (`nvext` worker id for the phase) forces the request's own `SessionAffinityId` (`RH:475-477`); otherwise a subagent (`agent_context.parent_session_id` present) binds under `subagent_group_affinity_id(parent)` for decode or aggregated and under the raw trimmed parent id for prefill (`RH:478-489`, SA-10); otherwise its own id.

| Policy | Mode | Hold target used as | Departed-worker handling | Where |
|---|---|---|---|---|
| KV | Hard | `pinned_target`: exact pin (`pinned_worker`, rank resolved); a worker-only binding narrows `allowed_worker_ids` to that worker and lets the scheduler pick the rank | selection error and `!affinity_target_is_valid`: `invalidate`, re-acquire, select once more | `kv.rs:39-45`; `kv_selection.rs:221-273`; `RH:683-694` |
| KV | Soft | `affinity_target` (advisory to the core; exclusive only for the builtin policy and only while eligible) | core steers elsewhere; commit rebinds | `kv.rs:43-46`; `queue.rs:135-158` |
| Hosted (round-robin, random, least-loaded) / DeviceAware | Hard | `target_constraint` then `dispatch_exact` (no transport fallback) | `!is_instance_discovered` after selection failure: invalidate plus re-acquire; discovered-but-unroutable fails the request | `hosted.rs:230-240,310-325`; test `builtin_hard_affinity_ignores_local_inhibition` |
| Hosted / DeviceAware | Soft | preferred worker if selectable, else picker; dispatch with transport fallback | before selection: `!is_instance_discovered` gives invalidate plus re-acquire (`RH:666-678`) | `hosted.rs:77-93,231-240` |
| Direct | Hard or Soft | `target_constraint` (the binding is the only target) | not healed before selection; Hard dispatch failure with an undiscovered worker invalidates; Soft keeps the binding and fails the request until TTL (NOTE session-affinity-11) | `hosted.rs:94-104,233,374-385` |

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| holding | worker accepted the send (`accept_dispatch`) | `commit(hold, dispatched)` ok | committed | Soft hosted dispatch derives the commit target from the final worker: rank kept only if the worker is the bound one (`hosted.rs:252-263`), so a fallback produces a rank-less dispatch and `rebound_target` leaves a ranked binding unchanged (test `builtin_soft_affinity_clears_dp_rank_after_worker_fallback`) | `RH:497-517` |
| holding | `accept_dispatch` | commit fails | failed | generation stopped, attempt aborted, request failed (test `affinity_commit_failure_after_the_send_aborts_the_attempt`) | `RH:505-516` |
| holding | dispatch failure before accept | Hard and `!affinity_target_is_valid(worker)` | invalidated | `hold.invalidate()` | `kv.rs:570-578`; `hosted.rs:374-385` |
| holding | dispatch failure before accept | otherwise | released | hold drops (Bound: release, binding kept) | same |
| holding-Bound (KV Hard) | pinned worker full or overloaded | - | failed, binding kept | `PinnedWorkerOverloaded` from the actor (`queue.rs:2557-2563`, `selector/mod.rs:240-253`); `affinity_target_is_valid` stays true, no re-target; migration retries cannot leave a Hard binding while the worker is discovered (test `migration_exclusion_preserves_hard_affinity_without_widening_or_escaping_hard_pins`) | `RH:683-694` |
| query-only (`query_instance_id`, `preview_kv_route`, `prefill_worker_busy`) | `query_target` | - | no hold | never heals a stale Hard binding (NOTE session-affinity-12); a conditional-disagg preview failing on it falls back to remote prefill where the non-query path heals (`prefill_router/mod.rs:540-548`) | `RH:638-641` |

### SA-7 Core managed session (`SessionBinding`)

Location: `lib/router/scheduler/src/selection/core/operation.rs:85-95`; `session_binding` `RUN:807-821`; `run_selection_inner` `RUN:263-567`; `hold_session` `RUN:575-615`; `commit_session` `RUN:620-671`; replay booking `reservations.rs:370-430`; lease storage `reservations.rs:42-47`; sweep `reservations.rs:553-583`. States per request: None (no session context, or an explicit `affinity_target` / `pinned_worker` makes it non-steerable, `RUN:77-78`), Query (`select` without booking: `query_target`, no hold), Managed (`select_and_reserve`, `create_reservation` replay: hold, steer, commit under the booking). Replay (`create_reservation` with a cached `select`) holds the session before booking (`reservations.rs:386-388`), books strictly, then commits; a Hard mismatch with the cached worker is `BadRequest` and frees the booking (test `two_phase_replay_rejects_a_worker_the_session_left`).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Managed, start | `hold_session` then `acquire` | Bound and `catalog.is_schedulable(target)` (lifecycle Schedulable, same partition, rank in range) | holding-Bound | `affinity_target = target` (exclusive for the builtin policy); overload, fullness, and taint are not checked here, the commit site handles them (SM-8 fixed) | `RUN:578-583,309-317` |
| Managed, start | `acquire` | Bound and not schedulable | loop: `invalidate`, re-acquire; yield every 32 | - | `RUN:579-603` |
| Managed, start | `acquire` | `ResourceExhausted` | no hold | routes unpinned, binds nothing (NOTE session-affinity-10) | `RUN:605-610` |
| holding | booking fails or future dropped | - | hold dropped (Bound: release) | - | `RUN:462-477` |
| holding-Initialize | booked on w | - | `commit(Initialize, w)` gives Bound-Active | lease into `Reservation._affinity_lease` | `RUN:514-527,629-631` |
| holding-Bound(t) | booked on w == t | - | lease kept | - | `RUN:630-631` |
| holding-Bound(t) | booked on w != t, Soft | - | rebind to w | - | `SA:583-590` |
| holding-Bound(t) | booked on w != t, Hard, t still schedulable | the request's own `allowed_worker_ids` (after LoRA narrowing) excludes t | `Err(BadRequest)` | `table.commit` Hard mismatch removes the binding; booking freed by the armed claim or handle (test `hard_mode_rejects_dispatch_away_from_a_live_binding`) | `RUN` `commit_session` call site; `SA:592-596` |
| holding-Bound(t) | booked on w != t, Hard, t still schedulable | t admitted by the request but shed by the host (overload provider, full ranks, unavailable, taint): `binding_allowed` true, `validate_dispatch_target` fails, `catalog.is_schedulable(t)` | `Err(Scheduler(PinnedWorkerOverloaded{worker_id: t}))` | binding kept (the hold drop releases the lease); booking freed by the armed handle; `table.commit` not called (SM-8 fixed) | `RUN` `commit_session` call site |
| holding-Bound(t) | booked on w != t, Hard, t departed since the hold | - | `try_acquire`: Held then validate (Hard) and commit; Wait then `join_initializing(w)`; up to 2 attempts, else `Ok(None)` unpinned | - | `RUN:638-668` |
| Bound (joined) | the initializer later books w' != w, Hard | - | `Err(BadRequest)` for the initializer; binding on w kept | reserved use released by the drop (finding SM-12; test `joined_failover_preserves_the_successful_bookings_binding`) | `SA:868-869` |
| lease in index | `free_reservation`, expiry sweep, peer displacement | - | lease dropped after the index lock is released | - | `reservations.rs:131-133,573-581` |

SM-8 fix notes: implemented at the `commit_session` call site (`binding_allowed` computed from the post-LoRA-narrowed set before the move); `commit_session` and the replay `finalize_reservation` are untouched, SA-6 and SA-8 unchanged. Passing the binding as `pinned_worker` was rejected because a pinned worker missing from the config fails fast with `NoEndpoints`, which would 503 the join and failover path.

### SA-8 Kernel eligibility: pinned worker versus affinity target

Location: `lib/router/scheduler/src/filter.rs:46-56` (`RoutingEligibility{pinned_worker, affinity_target, ..}`), `queue.rs:135-158` (`selection_eligibility`), `selector/mod.rs:74-80,240-253`, `reference.rs:424-443,483-485`, `lib/router/worker-selection/src/policy.rs:140,231-234,305-311`.

- `pinned_worker: WorkerWithDpRank` (explicit per-request pin, or the frontend's Hard binding): validated before selection (`validate_pinned_worker_allowed`, `validate_worker_rank`); not allowed gives `PinnedWorkerNotAllowed` / `NoEndpoints`; overloaded or full gives `PinnedWorkerOverloaded`; otherwise the policy sees exactly that rank. Overload bypass exists for pins only via `allows_worker_ignoring_overload`; hard availability still applies (`filter.rs:586-612` test).
- `affinity_target: WorkerAffinityTarget` (the core's Managed binding, the frontend's Soft binding, or a classifier override): applied as a constraint only when the policy declares `uses_exclusive_affinity_target()` and `affinity_target_is_eligible` (worker present, allowed, available, not overloaded, not full, taint-compatible, rank in range) (`queue.rs:150-157`, `filter.rs:357-372`); otherwise advisory context on `WorkerSelectionContext` (`selector/mod.rs:478-492`, `worker-selection/src/context.rs:196-203`).
- `exclusive_affinity` defaults to `false` on `WorkerSelectionPolicy` (`policy.rs:208,455`); the builtin default policy sets it (`lib/router-plugins/builtin/src/default.rs:85`); the test-only reference selector reports `true` (`policy.rs:306-309`). Hosted pickers never see affinity (the host applies it first; `policy.rs:395-401`). Hard mode with a non-exclusive policy fails every steered-away request on the core (NOTE session-affinity-6); nothing validates the mode and policy combination at startup (`services/selection/service.rs:198-215`).

### SA-9 Admission override of the affinity target

Location: `lib/router/admission/src/classifier/mod.rs:57-63,236-246,258-263`; applied in `queue.rs:1296-1323`. `ClassificationOverrides.worker_selection_target: Option<Option<WorkerAffinityTarget>>` (unset, set, cleared). Applied only when `request.pinned_worker.is_none()`, replacing `request.affinity_target` before the request is queued. Classification runs only for requests with a registered lifecycle (`lib/router/scheduler/src/local.rs:428-470`); only the frontend installs a classifier (`lib/llm/src/kv_router.rs:958-962`), and the frontend passes Hard bindings as `pinned_worker`, so today an override can displace only Soft affinity. The core passes a Hard managed binding as `affinity_target`, so a classifier wired into the standalone core could displace it (NOTE session-affinity-5, latent). The builtin `thunderagent` classifier calls `set_worker_selection_target` (`lib/router-plugins/builtin/src/thunderagent/request_classifier/mod.rs:320-323`).

### SA-10 Subagent group keys

Location: `SA:63-66`, `RH:470-490`, header mapping `lib/llm/src/protocols/agents.rs:159-175`. Prefill and decode hosts have separate tables, so the parent's prefill binding and the subagent group's decode binding never collide.

| Request | Phase | Binding key |
|---|---|---|
| explicit target present | any | own `SessionAffinityId` |
| no `agent_context` or no `parent_session_id` | any | own `SessionAffinityId` |
| subagent | Decode / Aggregated | `"\u{1}sg:" + blake3(parent_session_id)` (65 bytes, un-claimable from headers) |
| subagent, `trim(parent)` non-empty and != `trim(session_id)` | Prefill | raw trimmed `parent_session_id` (equals the parent's own key only when the affinity header maps to the agent session header; NOTE session-affinity-8) |
| subagent, parent == self or empty | Prefill | own `SessionAffinityId` |

### SA-11 Reaper and manual clock

Location: `SA:373-393` (tokio reaper, period `min(ttl, 30s)`, stops on `cancel` or when `Weak` fails); `SA:273-362` (manual clock: `advance_clock` rejects backwards time, reaps on the same cadence, lookups observe the new time immediately); `Inner::drop` cancels the reaper (`SA:191-195`). The frontend coordinator always uses the runtime clock; the manual clock is for synchronous hosts.

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| running | tick | `Weak` upgrades | running | `expire_idle(now)`: Bound-Expired entries removed, count decremented | `SA:373-393`, `721-731` |
| running | `cancel` or `Weak` fails | - | stopped | - | `SA:373-393` |
| manual | `advance_clock(t)` | `t >= now` and `t >= next_reap` | manual | reap, `next_reap += period` | `SA:353-360` |
| manual | `advance_clock(t)` | `t < now` | manual | rejected | `SA:273-362` |

Session-affinity interleavings traced and sound:

- Initialization is atomic and waiters wake on commit, join, or initializer drop; a cancelled waiter leaves cleanly (`session_affinity_initialization_is_atomic`, `session_affinity_initializer_cancellation_wakes_waiter`, `session_affinity_wait_stops_when_request_is_cancelled`, host tests `:130-208`).
- Replica ordering: only newer versions apply, a stale lease cannot invalidate or refresh a newer replica binding (`replica_applies_only_newer_affinity_versions`, `stale_lease_cannot_invalidate_or_refresh_newer_replica_binding`, host tests `:395-741`). `ReplacedExpired` ignores version order, but a late republish by a stale lease is blocked at `SA:1037-1039`, so no realistic resurrection trigger was found.
- Joined-initializer drop decrement (`SA:934-938`) has no zero guard; every path that could zero a joined binding's count while the initializer is alive was traced (joiner release, sibling leases, invalidate, re-initialization) and none reaches it: the reserved use keeps the count at least 1 and `invalidate` removes the entry rather than zeroing it.
- A Soft ranked binding is never widened or moved by a rank-less dispatch (`soft_ranked_binding_is_not_widened_by_worker_only_dispatch`); Hard dispatch away from a live binding is rejected (`hard_mode_rejects_dispatch_away_from_a_live_binding`); Direct Hard distinguishes unknown requests from stale affinity (`builtin_direct_distinguishes_unknown_requests_from_stale_affinity`).

## KV indexer (`dynamo-kv-indexer`, `dynamo-kv-events`, `dynamo-kv-router` indexer service)

Abbreviations: `CRTC:` = `lib/router/kv-indexer/src/concurrent_radix_tree_compressed`, `SVC:` = `lib/router/kv-router/src/services/indexer`, `REC:` = `lib/llm/src/kv_router/indexer/recovery`, `KVI:` = `lib/router/kv-indexer/src/kv_indexer.rs`, `TP:` = `lib/router/kv-indexer/src/thread_pool.rs`. Analyst numbering: 1 -> KI-1, 2 -> KI-2, 3 -> KI-3, 4 -> KI-4, 5 -> KI-5, 6/6a/6b -> KI-6 (transports T2 merged), 7 -> KI-7 (transports T1 merged), 8 -> KI-8, 9 -> KI-9, 10 -> KI-10, 11 -> KI-11, 12 -> TR-5, 13 -> KI-12.

### KI-1 CRTC node lifecycle

Location: `CRTC/node.rs:34-49` (`Node`: `shape_gate`, `shape_version`, sticky `internal`, `anchor`, `state`, `children`), `CRTC/state.rs:18-34` (`CrtcNodeState`: `edge`, `worker_cutoffs`, `full_edge_workers`, `parent_tail`), reclamation `CRTC/remove.rs`, sweep schedule `CRTC/mod.rs:63-91,195-226`.

| State | Representation |
|---|---|
| Live-leaf | `state.has_any_workers()`, `children.is_empty()`, `internal == false` |
| Live-internal | `internal == true` (sticky; set by `insert_child_if_still_missing` write path `node.rs:584` and `split_at_locked` `node.rs:601`; never cleared) |
| Dead-linked | `!has_any_workers() && children.is_empty()` and still in the parent map (`dead_leaf_key_and_tail` `node.rs:779-787`) |
| Queued | Dead-linked and a `CleanupEdge{parent, key, child}` (Weak refs) sits in some lane's `LaneLookup::pending_unlinks` (`lane_lookup.rs:226`) |
| Unlinked | removed from the parent map by `try_remove_stale_children` (`node.rs:810-841`); may still be referenced by a lane hold or a reader mid-walk; freed when the last `Arc` drops |
| Reused | Dead-linked node re-covered by a later store (`promote_to_full_with_version` / `cover_prefix_for_worker_with_version`) |
| Split-born-dead | suffix created by `split_off_suffix` with empty coverage and no children; queued by the splitting lane (`remove.rs:153-171`) |
| Anchor | `anchor == true`, one-block edge, lives in `anchor_nodes` for the tree's life (`mod.rs:298-311`); never extended, never unlinked |

Invariants:

- `worker_cutoffs[w] = k` implies `0 < k < edge.len()` and `w` not in `full_edge_workers` (`state.rs:25-29,117-154`).
- Edge positions never move; edges grow only at the tail, splits truncate (`state.rs:19-24`).
- `internal` is monotone; an internal node never takes `try_append_at_tail` / `try_extend_leaf_with_version` (`node.rs:487-489,518-524`).
- Unlink requires, under the parent's exclusive gate: still attached (`has_child` by identity), child gate `try_write` free, holder-less, childless, `Arc::strong_count == 2` (parent map plus unlinker) (`node.rs:816-835`).
- A lane holds exactly one `Arc` per node its entries name (`lane_lookup.rs:126-141,432-473`), so the strong-count test sees lane references.
- `shape_version` bumps on every shape change (split, extend, child insert on a non-internal node, child clear, unlink); plans carrying an older version are retried (`node.rs:129-145,558-589`).
- `deaths_since_sweep` is incremented only after the killer drops its own reference (`remove.rs:292-297`, `mod.rs:195-200`).
- Reader contract: `find_match_step` takes no shape gate (`node.rs:658-661`); it may observe adjacent shapes and undercount; it must never overcount except via the 2^-64 fingerprint collision of the equal-set skip (`node.rs:738-754`).
- Mid-chain removal truncates coverage on one node and does not cascade to children (`remove.rs:243-253`, documented `TODO(CORRECTNESS)`); `find_matches` is unaffected, `dump_events` may emit descendant stores whose parent is missing (NOTE kv-indexer-9).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| (none) | store, no child for first local | parent shape version unchanged | Live-leaf | `insert_if_absent`; parent becomes Live-internal if it was a leaf (write gate) | `node.rs:558-589`, `store.rs:449-461` |
| Live-leaf | store with parent == tail, worker covers tail | `!internal && !anchor` under write gate | Live-leaf (longer edge) | other full workers demoted to cutoff `old_len` | `node.rs:511-533`, `state.rs:205-231` |
| Live-* | store with parent mid-edge, diverging | plan version valid | prefix Live-internal plus suffix (Live or Split-born-dead) | children transferred to the suffix; lane lookups redirected for entries naming the prefix | `node.rs:591-604`, `mod.rs:324-348` |
| Live-* | remove run for worker w at min pos p | `p < current_cutoff(w)` | Live (cutoff p) or Dead-linked (`!has_any_workers`) | children cleared if no full worker remains; stale lookup hashes scrubbed | `node.rs:610-651`, `remove.rs:254-266` |
| Live-* | remove run | `p >= current_cutoff(w)` | unchanged | only the removed hash scrubbed from lookup | `node.rs:636-639` |
| Dead-linked | killer lane resolves parent (root, anchor, or `lookup.node(w, parent_tail)` verified by identity) | parent found and `has_child(key, node)` | Queued | pushed to `pending_unlinks`; drain of a quarter if len >= 4096 | `remove.rs:96-149` |
| Dead-linked | parent not resolvable | - | Dead-linked (left to sweep) | - | `remove.rs:144-148` |
| Queued | lane idle (`event_receiver.is_empty()`), `Flush`, or queue full | parent `try_write` ok, child `try_write` ok, holder-less, childless, count == 2 | Unlinked | parent version bump | `sync_impl.rs:157-170`, `remove.rs:177-194` |
| Queued | same, any guard false | - | Dead-linked (drops out of the queue, left to sweep) | - | `remove.rs:192` |
| Dead-linked | sweep (maintenance thread) | due: deaths >= max(1024, visited / 8) or 30 s | Unlinked | per parent: up to 4 `try_write` attempts with `yield_now`, deepest first | `mod.rs:204-226`, `remove.rs:27-92` |
| Dead-linked | sweep, parent gate busy on all 4 attempts | - | Dead-linked (until next sweep) | none (README: a parent whose gate stays taken keeps its dead children until the next sweep; SM-11 fixed the test contract, the sweep is unchanged) | `remove.rs:78-86` |
| Dead-linked | store reaching it (child lookup `Descend`) | version valid | Live | promoted; lane lookup gains entries (hold) | `store.rs:464-555` |
| Dead-linked | concurrent sweep and store | sweep checks `strong_count` under gates | exactly one of Unlinked / Live | a store holding the `Arc` (count 3) blocks unlink; a store blocked on the parent read gate while the sweep holds the write gate sees no child and creates a replacement | `node.rs:826-835`, `node.rs:540-556` |
| Live-internal with all children unlinked | store with parent == tail | - | Live-internal plus new child | `internal` sticky, never re-extended (test `test_cleanup_does_not_reopen_internal_node_for_extension`) | `node.rs:487-489` |
| any | `Cleared` for rank / `RemoveWorker{sweep_tree}` | - | coverage for the target removed on every node | whole-tree walk under each node's write gate; children cleared where no full worker remains | `remove.rs:318-351`, `node.rs:206-230` |
| any | `RemoveWorker{sweep_tree: false}` | - | lane lookup for the worker dropped, tree untouched | - | `remove.rs:324-330` |

### KI-2 Lane lookup holds

Location: `CRTC/lane_lookup.rs:126-216` (`NodeHolds`), `221-227` (`LaneLookup`). States per (node, lane): Unnamed (no hold); Held(refs = n >= 1). Invariant: a hold exists exactly while an entry names the node; `ids: address -> NodeId` is a bijection with live holds; freed ids recycle through `free` (`assert_invariants` `lane_lookup.rs:435-473`).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Unnamed | `upsert_all` / `insert` / `redirect(to)` | - | Held(1..) | `Arc` cloned once (strong + 1) | `lane_lookup.rs:154-174,301-349,385-405` |
| Held(n) | `remove` / `remove_all` / `remove_workers` / `redirect(from)` | - | Held(n - k) or Unnamed | `Arc` dropped when refs hit 0 (strong - 1) | `lane_lookup.rs:199-215,351-425` |
| Held | `redirect(from, to)` after split or repair | entry still names `from` | entry moves to `to` | entries naming any other node left alone | `lane_lookup.rs:385-405` |

### KI-3 Sweep schedule and maintenance thread

Location: `CRTC/mod.rs:80-91,204-226`; thread `TP:106-138`. States: NoThread (`tick == None`); Idle (ticking every 50 ms); Sweeping (holds the `last_sweep` mutex; only the maintenance thread sweeps). Counters `deaths_since_sweep`, `last_sweep{began, nodes}`. A death counted while a sweep runs rolls into the next sweep (`mod.rs:215`).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Idle | tick | `deaths >= max(after_deaths, nodes / 8) \|\| elapsed >= max_interval` | Sweeping | `deaths = 0`, `began = now` | `mod.rs:205-217` |
| Sweeping | walk done | - | Idle | `nodes = visited` | `mod.rs:219-225` |
| Idle | `stop` sender dropped or backend `Weak` dead | - | thread exits | joined by `stop_maintenance` / `Drop` | `TP:123-128,497-506` |

### KI-4 ThreadPoolIndexer lanes, worker assignment, two-phase removal

Location: `TP:58-104` (fields), `833-847` (`get_or_assign_thread_idx`), `849-907` (removals), `1309-1350` (interface impl); lane loop `CRTC/sync_impl.rs:13-175`. States per `(worker_id, dp_rank)`: Unassigned (no `worker_assignments` entry); Assigned(lane i) forever (entries never removed, `TP:62-68`). Per worker in a lane: Unknown (no `LaneLookup` worker); Known (`lookup.add_worker` on first store, `store.rs:37`); back to Unknown after `RemoveWorker` / `Cleared` (`lookup.remove_workers` drops the key).

Invariants:

- All events for one rank go through one lane FIFO; `RemoveWorker` is broadcast to every lane (a worker's ranks may sit on several lanes); the tree sweep for a removed worker runs once on lane 0 after every lane has dropped its lookup.
- Contract stated in code (`TP:892-894`): "The lifecycle caller stops this worker's source before removal ... events concurrently admitted before this barrier completes belong to the retiring source and may be removed." Events admitted after the barrier are indexed as a live worker; since SM-3's fix the registry joins the listener before the barrier, so only events from a source the lifecycle caller did not stop can arrive after it. With one worker on `Concurrent` the leak was invisible (lane 0 phase 2 cleans the lane lookup); the regression test seeds lane 0 with another worker.

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Unassigned | first event, anchor, or removal for a rank | - | Assigned(count++ % n) | `worker_assignment_count` is a reservation counter, not a live count | `TP:833-847` |
| any | `enqueue_event` | CacheOwner domain with `state_source` | sent to the owner-hashed lane only | - | `TP:685-711,728-743` |
| any | `enqueue_event`, `reset_scope == All` with `state_source` | owner lane != worker lane | sent to both lanes | event cloned | `TP:703-737` |
| Known | `Stored` with unknown parent for the worker | lookup miss and no anchor | unchanged | `ParentBlockNotFound` logged at warn, counted | `store.rs:168-203` |
| Unknown | `Removed{non-empty}` | `!lookup.contains_worker` | unchanged | `BlockNotFound` error, counted | `remove.rs:213-220` |
| Unknown | `Removed{empty}` | - | unchanged | no-op by design (normalizer id filler) | `remove.rs:213-216` |
| Known | `remove_worker(id)` | - | Unknown on every lane, then tree swept | phase 1: `RemoveWorker{sweep_tree: false}` to all lanes, await acks; phase 2: `RemoveWorker{sweep_tree: true}` to lane 0, await | `TP:870-907` |
| Known | `remove_worker_dp_rank` | LRU disabled | Unknown on the rank's lane plus tree swept from that lane | `RemoveWorkerDpRank{sweep_tree: true}`, fire and forget (no ack) | `TP:1325-1350`, `sync_impl.rs:130-141` |
| Known | `Cleared` for rank | - | Unknown on the lane plus tree swept | `erase_worker_coverage(DpRank, true)` | `CRTC/mod.rs:389-392` |
| any | `Flush` | - | same | drains the lane's pending unlinks fully, then acks | `sync_impl.rs:157-161` |
| any | `Terminate` | - | lane exits | - | `sync_impl.rs:162-164` |

### KI-5 KvIndexer actor (single-threaded RadixTree)

Location: `KVI:424-444` (channels), `474-633` (select loop), `255-293` (`drain_pending_mutations`), `137-148` (`MutationRequest`). States: Running; Cancelled (token). Channels: `mutation_rx` (cap 16384, FIFO: `Event`, `EventWithAck`, `ResetWorkerDpRank`, `RemoveWorker(WorkerId)`, `RemoveWorkerDpRank{worker_id, dp_rank}`), side channels `routing_rx`, `match_rx`, `dump_rx`, `flush_rx`; `remove_worker_sender()` returns the shared FIFO `mpsc::Sender<MutationRequest>` (public type). Invariant: a rank reset, a whole-worker removal, and a rank removal are FIFO barriers relative to events on `mutation_tx` (SM-2 fixed; the former `remove_worker_rx` / `remove_worker_dp_rank_rx` side channels no longer exist); events admitted after a removal are indexed as a live worker; the thread pool `Indexer::Concurrent` (KI-4) is unchanged.

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Running | poll | `biased`: cancel, `mutation_rx`, `match_rx`, ... | Running | mutations and removals applied in FIFO order | `KVI` select loop |
| Running | `ResetWorkerDpRank` / `RemoveWorker` / `RemoveWorkerDpRank` on the mutation queue | - | Running | rank or worker removed in FIFO position (ack sent for the reset); approximate-LRU lane state for the removed ranks forgotten | `KVI` `MutationRequest` handling |
| Running | dump / flush | - | Running | `drain_pending_mutations`: queued mutations and removals in FIFO order (approximate-LRU lane state for removed ranks forgotten), routing, prune; then dump or ack | `KVI` `drain_pending_mutations` |
| Running | cancel | - | exits | - | `KVI:478-481` |
| any | last `KvIndexer` clone dropped | `strong_count(_ref_count) == 1` | cancel | - | `KVI:1025-1032` |

### KI-6 Event cursor: ZMQ listener watermark and host `RankState`

Type: `lib/kv-events/src/cursor.rs:6-50`. States `Initial`, `Live(last)`. `observe(got)`: `Initial` -> `Initial{got}`; `got <= last` -> `Stale`; `got == last + 1` -> `Contiguous`; else `Gap{expected: last + 1, got}`. `advance_to(id)` is unconditional and the type has no regression state: a regression is `Stale`; the ZMQ listener below turns a stale seq outside its replay overlap window into a new epoch (SM-4 fixed), the host `RankState` still ignores it.

ZMQ listener cursor (transports T2 merged): `SVC/listener.rs:20-28` (`WATERMARK_UNSET = u64::MAX` is `Initial`), `187-189`, `318-345` (`handle_gap`), `389-420` (`handle_message`), `347-387` (`apply_live_batch`), `191-316` (`replay_gap`). The watermark is an `Arc<AtomicU64>` owned by `WorkerRegistry.watermarks[(instance, rank)]` (`SVC/registry.rs:455-459`), created `u64::MAX`, reused across pause and resume, removed on deregister, fresh on re-register (test `re_register_gets_fresh_watermark`). Granularity: one seq per ZMQ batch; every event in the batch gets `event_id = seq`; the watermark advances after the batch's events are enqueued, not applied (`listener.rs:385`). Invariants: a stale seq inside the replay overlap window (`ListenerLoop.replay_trigger`, the live seq that opened the last replay; the window closes on the next non-stale frame) is dropped as a duplicate, any other stale seq restarts the epoch (`restart_epoch`); replay is requested for `[expected, got)` before `got` is applied (`:412`); the live SUB socket is not drained while replaying (single task); replay waits are bounded by `REPLAY_INACTIVITY_TIMEOUT` (5 s, the same bound as the publisher's `zmq_recovery` `INACTIVITY_TIMEOUT`) and a replay that gives up does not mark the attempt Failed (Failed is terminal for the registry); the replay DEALER is dialed per gap and dropped after (`:212`); `connect_dealer_socket` sets `ZMQ_RCVTIMEO = 100 ms` but the async path uses `DONTWAIT` plus `AsyncFd` readiness, so the option never applies (`services/common/zmq.rs:163-190,252-263,303-311`).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Initial | batch seq 0 | - | Live(0) | apply batch | `listener.rs:340-343,419` |
| Initial | batch seq g > 0 | replay endpoint set | Replaying then Live(g) | `replay_gap(0, g)`; watermark stored per replayed batch; then the live batch applied | `listener.rs:320-329` |
| Initial | batch seq g > 0 | no replay endpoint | Live(g) | warn "batches lost"; history `[0, g)` never indexed (NOTE kv-indexer-11, transports-5, transports-12) | `listener.rs:200-208` |
| Live(n) | seq n + 1 | - | Live(n + 1) | apply | `listener.rs:341,419` |
| Live(n) | seq g > n + 1 | replay endpoint set | Replaying then Live(>= g - 1), then g applied if not Stale | DEALER connect, send `[empty, be64(n + 1)]`, apply replies until empty payload, error, or cancel; `warn_if_incomplete` (`ReplayRecoveryProgress`) | `listener.rs:330-339,84-143,191-316` |
| Live(n) | seq g > n + 1 | no replay endpoint | Live(g) | gap logged "batches lost", g applied, no residency clear | `listener.rs:200-208` |
| Live(n) | stale seq | seq >= `replay_trigger` (open overlap window) | Live(n) | dropped as a duplicate before decode | `listener.rs` `handle_gap` |
| Live(n) | stale seq | any other stale seq (engine restart) | Initial, then as a cold start | `restart_epoch`: warn, synthetic `AllBlocksCleared` for (worker, listener dp_rank) through the normalizer, watermark reset to unset, `handle_gap` re-run (replays `[0, seq)` or logs the lost prefix), then the batch applied (SM-4 fixed) | `listener.rs` `restart_epoch`, `handle_gap` |
| Replaying | reply frame malformed or decode error | - | continues or ends | frame-count or seq-length error ends replay (`break`), decode error skips the batch | `listener.rs:252-282` |
| Replaying | engine replies seq < n + 1 or > g | - | watermark follows every reply | re-applied in order (idempotent store or remove; `Cleared` re-applied) | `listener.rs:307` |
| Replaying | no reply within `REPLAY_INACTIVITY_TIMEOUT` (5 s) | - | Live(g) | warn; `warn_if_incomplete` logs the gap; the triggering live batch is applied; the listener continues live and the attempt is not marked Failed (SM-1 fixed) | `listener.rs` `replay_gap` |
| Replaying | cancel | - | task exit after `apply_live_batch` of the triggering batch | partial replay kept | `listener.rs:238-249,412-419` |
| Replaying | connect, send, recv, or framing failure, or empty reply | - | Live(g) | `replay_gap` returns `Ok(0)`, `warn_if_incomplete` only logs, g applied (NOTE transports-5) | `listener.rs:200-234,255-276,311` |
| any | `apply_event_routed` error in a batch (only `Cleared` reset failures: `IndexerOffline` / `DroppedRequest`) | - | listener task returns Err; record Failed (KI-7) | watermark not advanced for that seq | `listener.rs:374-382,456-478`; `lib/router/kv-indexer/src/backend.rs:339-420` |

Residual (SM-4): a restart whose first seen seq lands inside an open overlap window is treated as a duplicate; pre-existing and not addressed by the fix. `lib/kv-events/src/cursor.rs` is unchanged.

Host `RankState` cursor (worker-query recovery client): `REC/worker_query_state.rs:37-47` (`cursor`, `recovery_inflight`, `pending_live_events` <= 1024, `max_seen_live_id`), `66-126` (`observe_live_event`); consumers `REC/worker_query.rs:680-730` and `808-1000`. States: Idle(cursor); Recovering(cursor, buffer); Fenced (slot-level, `fence_for_reset`). Above 1024 buffered events the oldest is dropped and the cursor advances past the hole by documented contract (NOTE kv-indexer-6).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Idle(Initial) | live event, recoverable source | - | Recovering | buffer event; `Recover{None, None}` (full snapshot) | `worker_query_state.rs:96-103` |
| Idle(Initial / Live) | live event, not recoverable | Initial or Contiguous | Idle | `Apply`; cursor advanced after admission | `:119-124,128-131` |
| Idle(Live n) | gap, recoverable | - | Recovering | buffer; `Recover{Some(n + 1), None}` (server decides Events versus TreeDump) | `:104-117` |
| Idle(Live n) | gap, not recoverable | - | Idle | `ResetDegraded`: rank cleared then the event applied | `:118`, `worker_query.rs:707-727` |
| any | event with `reset_scope == All` | `id > last_admitted` | Idle | `Clear`: recovery cancelled, buffer discarded, clear admitted | `:73-81`, `worker_query.rs:683-700` |
| any | event id <= last admitted | - | unchanged | `Ignore` (includes restart from 0) | `:84` |
| Recovering | live event (any non-stale) | - | Recovering | buffered; oldest dropped above 1024 | `:85-95,196-203` |
| Recovering | `Events{events, last}` | source matches | Idle(cursor = max(last admitted, last), then last buffered) | snapshot events plus buffered tail (> recovered_through, sorted, dedup) admitted; holes warned, not re-queried | `worker_query.rs:833-847,933-970` |
| Recovering | `TreeDump{All}` | source matches | Idle | `replace_rank` transactional; cursor `Initial.advance_to(last)`; tail as above | `:848-888` |
| Recovering | `TreeDumpFailed` or non-authoritative error | - | Idle (cursor unchanged) | wait for the next live event | `:889-901,921-925` |
| Recovering | other error | - | Idle | degraded: buffered tail admitted, cursor = last buffered | `:926-931,979-1000` |
| Recovering | foreign-source response | - | Fenced | rank cleared, no replay of the buffered tail | `:823-832,1002-` |

### KI-7 ZMQ listener attempt (`ListenerRecord`)

Location: `SVC/registry.rs:137-152` (`ListenerRuntime{status, last_error, cancel_token, generation}`), `203-292`; task `SVC/listener.rs:456-540` (`run_listener`); driven by `services/selection/zmq_ingress.rs:78` (`reconcile`) and `:160` (`detach`). Transports T1 merged here. States: Pending(gen g, token); Active(g, token); Paused(no token); Failed(no token, `last_error`); Detached / Removed (record absent from `WorkerEntry.listeners`, token taken; watermark entry removed).

Invariants: a task is current iff `generation == runtime.generation && cancel_token.is_some()` (`registry.rs:269`; `is_current_attempt`, `try_mark_active`, `try_mark_failed`); exactly one token per attempt; a token is cancelled on pause, deregister, root cancel; `watermarks[(worker, rank)]` exists iff the record exists (removed in every deregister path `:514,560,609`; recreated `u64::MAX` by `register` `:455`); the readiness gate `ready_rx` (`:392`) must be `true` before an attempt becomes Active (`listener.rs:505-519`), and the SUB socket connects before the gate, so messages beyond RCVHWM are dropped until the loop starts (NOTE transports-12); a Failed or Paused record still answers `has_listener == true` (`:670`); `spawn_zmq_listener` returns its `JoinHandle`, kept per attempt by `ListenerRuntime` (`set_task` ignores a stale generation), and `ListenerRecord::stop()` (cancel plus join) is what every deregister path calls after the registry-visible changes and before the indexer removal (SM-3 fixed); the join is bounded by one batch because every wait in the task has a cancel arm, and no cancel check was added inside `handle_message`.

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Removed | `register` | worker key matches or new; rank not registered | Pending(g + 1) | claims the partition indexer, creates or reuses the watermark, spawns the task; waits on `ready` | `registry.rs:424-485`, `listener.rs:505-510` |
| Pending | `ready` true, `try_mark_active(g)` | g current | Active | recv loop starts | `listener.rs:512-539` |
| Pending | `ready` true but g stale (paused or replaced meanwhile) | `try_mark_active` false | unchanged | task exits silently | `listener.rs:512-519` |
| Pending / Active | `pause_listener` | token present | Paused | token cancelled; task exits at the next loop top | `registry.rs:216-243,621-641` |
| Paused / Failed | `resume_listener` | - | Pending(g + 1) | new token, new task, same watermark `Arc`; the stale task exits at its next guard | `registry.rs:245-267,643-663` |
| Pending / Active | task error (SUB connect, recv, `apply_event_routed` Err) | g current, token present | Failed | `last_error`, token dropped; no auto-restart (NOTE kv-indexer-8, transports-7) | `registry.rs:284-292`, `listener.rs:475-476` |
| Pending / Active | `deregister_dp_rank` / `deregister` / `deregister_all_routing_groups` | key matches | Removed | entry (`remove_if`) and watermark removed; token cancelled and task joined (`ListenerRecord::stop()`); then `remove_worker_dp_rank` / `remove_worker` enqueued; indexer released if unused | `registry.rs` deregister paths |
| Detached | task error | token none | Detached | `try_mark_failed` no-op | `registry.rs:286-288` |
| Active | cancel | observed at the `run` select (biased) | task exits | the current `handle_message` completes first | `listener.rs:422-453` |
| Active / Failed | `reconcile(previous, record)` same endpoint, replay, mm flags | `has_listener` | unchanged (retained; a Failed listener too) | none | `zmq_ingress.rs:106-119` |
| Active | `reconcile` with a changed endpoint | `has_listener` | Removed then Pending(1) on the new record | `deregister_dp_rank` then `register` | `zmq_ingress.rs:120-156` |
| any | root cancel | - | task exits | status unchanged | `registry.rs:333`, `listener.rs:427` |

### KI-8 Worker registration and the index registry

Location: `SVC/registry.rs:314-326` (`WorkerEntry{key, listeners}`, `watermarks`), `424-619`; `lib/router/kv-indexer/src/registry.rs:50-60,112-168,190-219` (`IndexRegistry`, lifecycle `tokio::Mutex`, `retain_empty_indexers`). States per instance id: Unknown; Registered(key, ranks non-empty); Mid-deregistration (ranks empty before `workers.remove`, filtered from `list`); Removed-pending-barrier (entry gone, `remove_worker` awaiting lanes); Unknown. States per partition: NoIndexer; Indexer(block_size, handle); Released (removed from the map; handles elsewhere keep the object alive).

Invariants: `claim` and `release_if_unused` serialize on the lifecycle lock; `in_use()` is evaluated under it (`registry.rs:159-168`); a block-size mismatch on `claim` is an error, on `get_or_create_indexer` only a warning; the scheduler core builds its registry `with_retained_indexers()` (`lib/router/scheduler/src/selection/core/mod.rs:349-353`), the standalone server does not (`SVC/registry.rs:363`); `deregister*` do not hold the lifecycle lock across the removal await (by design: test `registration_progresses_while_removal_is_backpressured`), and the thread-pool barrier removes by `WorkerId`, not by registration generation (finding SM-16); lock-free `get_or_create_indexer` is safe only because the core retains indexers and the standalone server recovers before serving (NOTE kv-indexer-7).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Unknown | `register` | key not registered under another partition; rank not registered | Registered | `claim` under the lifecycle lock; watermark `or_insert(u64::MAX)`; listener spawned after the lock drops | `registry.rs:424-485` |
| Registered | `register` same id, other partition | - | error | - | `:438-446` |
| Registered | `register` same id and rank | - | error "already registered" | - | `:447-449` |
| Registered | `deregister` | key matches | Removed-pending-barrier then Unknown | tokens cancelled and listener tasks joined; watermarks removed; `indexer.remove_worker(id).await`; `release_indexer_if_unused` | `registry.rs` `deregister` |
| Registered | `deregister_dp_rank`, not last | - | Registered(fewer ranks) | `remove_if` on the record, join the listener, then `remove_worker_dp_rank` (no ack on the thread pool) | `registry.rs` `deregister_dp_rank` |
| Registered | `deregister_dp_rank`, last | - | Unknown | as `deregister` | `:562-573` |
| Indexer | last worker leaves | `!retain_empty_indexers` and no `WorkerEntry` has the key | Released | `indexers.remove(key)` under the lock | `kv-indexer registry.rs:159-168` |
| NoIndexer | `get_or_create_indexer` (peer recovery, `ZmqDirectIngress::open`) | - | Indexer | lock-free `entry().or_insert_with` | `kv-indexer registry.rs:190-219` |
| Removed-pending-barrier | `register` same id (concurrent) | lifecycle lock free | Registered (new record, fresh watermark) while the old barrier is still in flight | the old phase 1 and 2 erase the new registration's early blocks (finding SM-16) | `registry.rs:436-452` versus `:519-522` |

### KI-9 Peer recovery (standalone indexer) and dumps

Location: `SVC/recovery.rs:20-88`; ordering `SVC/run.rs:184-196` (recover, `register_peer`, `wait_for_min_initial_workers`, `signal_ready`, HTTP serve); server dump `SVC/server.rs:489-523`; primary dumps `KVI:562-583` (drains queued mutations first) and `CRTC/sync_impl.rs:280-286` (explicitly not a consistent cut); tier retag `backend.rs:802-835`. Handshake states on the joining side: Idle, Fetching(peer i), Applying(partition), Done, NextPeer, GiveUp. Live events cannot arrive during dump application: listeners gate on `ready` (`listener.rs:505-510`), signalled after recovery. CRTC `dump_events` returns an empty vector on the lane path (`sync_impl.rs:142-144`) and a non-consistent tree walk on the interface path.

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Idle | start | peers non-empty | Fetching | 1 s sleep, 10 s HTTP timeout | `recovery.rs:20-27` |
| Fetching | 2xx JSON | - | Applying | per partition `get_or_create_indexer`, `apply_event_routed` each event (enqueue only) | `:61-84` |
| Applying | any apply error | - | NextPeer | partial state from this peer stays applied | `:78-81,34-37` |
| Fetching | status or parse error | - | NextPeer | - | `:57-62` |
| Applying | all events enqueued | - | Done (returns before lanes apply) | - | `:86-87` |
| Done | listeners start | ready signalled | each worker cursor `Initial` -> `Initial{got > 0}` -> full replay from 0 on top of the dump | duplicates counted as `DuplicateStore` warnings | `listener.rs:320-329` |

### KI-10 `LocalKvIndexer` recovery server (worker side)

Location: `lib/router/kv-indexer/src/local.rs:47-62` (`RecoveryState{buffer, max, generation, cached, building}`), `83-121` (`record`), `125-156` (`snapshot_expired`, `snapshot_tail`), `158-262` (`DumpPlan`, `DumpDecision`, `DumpBuild`), `402-412` (`apply_event_with_buffer`: record after admission, not application), `418-470` (`classify_query`). States: buffer Empty, Contiguous(first..last), Gapped (non-consecutive id recorded: error logged, generation bumped); snapshot None, Cached(base), Expired (appended > max / 2); build Idle, Building(notify, generation, base).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| any | `record(event)` | id == last + 1 and not Cleared | buffer grows (oldest popped above max) | expired snapshot freed | `local.rs:83-121` |
| any | `record` | gap or Cleared | generation + 1, snapshot dropped | error log on gap | `:87-96,106-109` |
| query | `start_id` None, buffer empty, or start older than the buffer | - | RequiresDump | - | `:445-460` |
| query | `start_id > last_buffered` | - | `TooNew` | - | `:462-470` |
| query | in range | - | `Events` through the current tail | may start at the last all-domain clear | `:338-351` |
| RequiresDump, Cached(base) | - | tail extends and not expired, tail has no Cleared | `TreeDump`-equivalent from cache plus tail | - | `:140-156` |
| RequiresDump, Building | - | - | Wait(notified) then re-check the cache | waiters never take the builder's response | `:55-61,160-165` |
| RequiresDump, Idle | - | - | Building then `finish` | cached only if the generation is unchanged and not expired; `Drop` releases the slot and wakes waiters | `:182-262` |

### KI-11 ZMQ normalizer per-worker filter state

Location: `lib/kv-events/src/zmq_wire/mod.rs:74-96` (`full_block_workers`, `cache_namespaces`, `group_metadata`, `engine_mm`), `207-299` (`preprocess_residency_with_reason`), `382-407` (`filter_partial_block`); `zmq_wire/mm_exclusion.rs:38-45` (`excluded`, `overflowed_workers`), `80-147` (`prepare_store`), `150-167` (`filter_remove`, `clear`), `169-180` (`exclude`). Per worker: Unproven (no full-size block yet), Proven (`full_block_workers` contains the worker), Unproven again on `AllBlocksCleared`. Per `(worker, tier, hash)`: Indexable, Excluded (mm cut or chained under an excluded parent); per worker: Normal, Overflowed (latched until clear; NOTE kv-indexer-10).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Unproven | `BlockStored` with `block_size == kv_block_size` | - | Proven | - | `mod.rs:387-391` |
| Proven | `BlockStored` smaller | - | Proven | filtered `PartialBlock` at debug; no placement event emitted | `:392-401` |
| Unproven | `BlockStored` smaller | - | Unproven | left to conversion, which warns | `:402-406` |
| any | `AllBlocksCleared` | - | Unproven; mm state cleared | - | `:291-293`, `mm_exclusion.rs:163-167` |
| Indexable | store whose block window holds an mm value | no duplicate hashes | that block and tail Excluded; store truncated | - | `mm_exclusion.rs:113-141` |
| Indexable | store with an Excluded parent | - | all blocks Excluded; store dropped | emitted as an empty removal to keep ids contiguous | `:102-111`, `mod.rs:59-70` |
| Normal | `exclude` at capacity (1 << 20) | key new | Overflowed | warn once; every store with a parent from this worker dropped | `:169-180,96-100` |
| Excluded | removal names it | - | Indexable (entry removed) | hash dropped from the removal | `:150-160` |
| Overflowed | any | not `clear` | Overflowed | removal of excluded keys frees capacity but never unlatches | `:163-167` |

### KI-12 Residency snapshots and session residency

- `LowerTierIndexers.routing_snapshot: Arc<ArcSwap<ResidencyRoutingSnapshot>>` replaced whole (`lower_tier_indexers.rs:40,80-81`); `OwnerSnapshot` append-only owner ids behind `ArcSwap` (`lower_tier.rs:237-277`). Readers pin one snapshot per lookup; tree reads are the best-effort reads of KI-1, so a lookup can see a store half applied across nodes (undercount only).
- `SessionPrefixIndexer.state: RwLock<IndexState>` (`session_prefix_index.rs:78-79`); mutations serialized through `SessionMutation` after the primary (`backend.rs:654-700`: `WorkerRemoved` / `Cleared` sent after the primary removal); LRU session eviction (`:610`). Per session: Unknown, Bound(frontiers per worker), Rebound on match or store, frontiers cleared on worker removal, evicted LRU.
- `kv_hints.rs` holds pure key functions and the `KvTransferCandidates` DTO; no state.

KV-indexer interleavings traced and sound:

- Dead-leaf reuse versus sweep for the node the store holds: a store holding the `Arc` (count 3) blocks unlink; a store blocked on the parent read gate while the sweep holds the write gate sees no child and inserts a replacement; a stale `shape_version` retries (`node.rs:826-835`, `540-556`, `564-575`). Tests `a_referenced_dead_leaf_is_left_for_the_sweep` (`CRTC/tests.rs:1904-1923`), `the_maintenance_thread_sweeps_a_leaf_the_lane_cannot_place` (`:2011-2031`).
- Rank reset, worker removal, and rank removal as FIFO barriers on the mutation queue (tests `post_reset_event_survives_fifo_rank_reset`, `remove_worker_does_not_overtake_events_admitted_before_it`, `remove_worker_dp_rank_does_not_overtake_events_admitted_before_it`).
- Generation fencing of listener attempts: a stale task cannot mark Active or Failed (`registry.rs:269-292`); `re_register_gets_fresh_watermark`, `deregister_dp_rank_removes_watermark`, `listener_cancelled_by_root` (`registry.rs:1040-1136`).
- Host recovery with a reused id or foreign source: fenced (`worker_query.rs:823-832`); peer recovery ordering (`run.rs:184-196`) excludes live events during dump application.
- `race_cleanup_with_dead_child_reuse_keeps_restored_child` was rewritten by SM-11's fix: it accepts `[2, 3]` or `[2, 2, 3]` after the race (the former `[5]` is unreachable because `internal` is sticky and both extension paths refuse an internal node), keeps `raw_child_edge_count == len` and score 5, then runs a second uncontended sweep and asserts exactly `[2, 3]`; measured 0/300 failures against 3/300 before. The sweep itself is unchanged: the skipped child has no lane hold and no children, so the next sweep reclaims it.

## Prefill and disaggregation host (`dynamo-llm`)

Abbreviations: `PR:` = `lib/llm/src/kv_router/prefill_router`, `RH:` = `lib/llm/src/kv_router/routing_host`. Analyst numbering M1..M8 maps to PD-1..PD-8. The kernel never decides prefill versus decode: `WorkerType` (`lib/router/protocol/src/worker_type.rs:95`) is a per-partition role fixed at construction (`embedded.rs:233,302`), `WorkerPlacement` (`lib/router/admission/src/placement.rs:71`) is only an admission-queue lane key (AD-3), and `prefill_load.rs` is two pure functions. The phase lives in the host: `RequestTracker.phase` (`lib/llm/src/protocols/common/timing.rs:389-403`), set by the `PrefillRouter`, read by the `RoutingHost` for cleanup policy and overrides.

### PD-1 PrefillRouter activation: lifecycle x target x binding

Location: `PR/mod.rs:264` (`PrefillRouter{lifecycle: AtomicU8, target: Mutex<Option<WorkerSetTargetId>>, binding: ArcSwapOption<PrefillBinding>, target_tx: watch}`); publishers `PR/activation.rs:604` (`set_target`) and `PR/activation.rs:474` (`drive_target`); reader `PR/activation.rs:639` (`activation_snapshot`). `L` = `lifecycle`, `T` = `*target.lock()`, `B` = `binding.load()`.

| State | L | T | B | Meaning |
|---|---|---|---|---|
| Untargeted-never | `Pending` | `None` | `None` | constructed, no target yet (`activation.rs:242`) |
| Pending(t) | `Pending` | `Some(t)` | `None` | target known, binding being built: fail closed |
| Active(t) | `Active` | `Some(t)` | `Some(b)`, `b.target_id == t` | serving through `b.router` |
| Unavailable-kept | `Unavailable` | `None` | `Some(b_old)` | target cleared, old binding retained for fast reactivation |
| Unavailable-bare | `Unavailable` | `None` | `None` | cleared before any build finished |

Invariants:

- `activation_snapshot` folds these to `Active(b) | Pending | Untargeted` (`activation.rs:639-654`): `Unavailable-*` and `Untargeted-never` are both `Untargeted` (passthrough); `Active` with `B == None` is treated as `Pending` or `Untargeted` by `T` (`activation.rs:644-647`).
- Both publishers store `B` and `L` while holding `target` (`activation.rs:558-566`, `604-624`); a snapshot taken under the lock never sees a half-published transition. The lock-free `lifecycle_state()` fast path at `PR/mod.rs:446` may lag, so `generate` re-resolves under the lock at `PR/mod.rs:447` and again at `PR/mod.rs:580-598` after the phase barrier.
- `L == Active` implies `B.is_some()`; `set_target(None)` never clears `B` (`activation.rs:616-623`); `set_target(Some(t'))` with `t' != B.target_id` clears it.
- A binding owns a `RoutingLoadContext` whose `Drop` cancels its token subtree (`kv_router/routing_load.rs:391-393`), stopping its `KvRouter`, load watchers, and admitted-instance follower (`lib/runtime/src/component/client.rs:704-730,937-947`); the per-build `parent_token` (`activation.rs:531`) is cancelled only by `PrefillRouter::drop` (`PR/mod.rs:400-405`).
- `WorkerSetTargetId::Committed{group, generation}`: `generation` is a controller-wide monotonic counter (`discovery/controller.rs:909-910`), so a re-added group never aliases a stale binding id.
- Only a typed `Decode` WorkerSet gets a `PrefillRouter` (`discovery/watcher.rs:666-668`); `reconcile_discovery_topology` sets the target to `Some` only when exactly one Prefill provider and exactly one Decode consumer exist in the namespace, else `None` (`discovery/model_manager.rs:2672-2691`), which turns a typed-Decode set into a passthrough (finding SM-18).
- Request-side reads (`PR/mod.rs:444-480`, `580-598`): `Active` gives the prefill hop; `Pending` and not query-only gives `WorkerUnavailable` (migratable 503); `Pending` and query-only, or `Untargeted`, passes through to the decode host. Admission read (`available_worker_ids_for`, `PR/mod.rs:340-378`): `None` if this router does not target `(group, endpoint)`; `Some(empty)` if cancelled or `L != Active`; else the binding client's available ids; the target match prefers `target_tx` over the binding, so a cleared target hides a retained binding.
- The `EncoderRouter` (`lib/llm/src/kv_router/encoder_router.rs:40-55,200-333`) is the same machine with the same five states and lock discipline, minus the legacy oneshot and `activation_snapshot` (its `generate` reads `L` then `B`, `encoder_router.rs:391-405`, and degrades to inline encoding when `B` is `None`).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| any | `set_target(Some(t))` | `T == Some(t)` | same | none (early return) | `activation.rs:607` |
| Untargeted-never / Pending(x) / Unavailable-bare | `set_target(Some(t))` | `B` is `None` or `B.target_id != t` | Pending(t) | `B := None`, `L := Pending`, `target_tx.send_replace(Some(t))` | `activation.rs:611-627` |
| Active(x) | `set_target(Some(t))`, `t != x` | - | Pending(t) | `B := None` (in-flight requests keep their `Arc<PrefillBinding>`); watch wakes `drive_target`, which drops any in-progress build | `activation.rs:621`, `545-550` |
| Unavailable-kept(b_old) | `set_target(Some(t))` | `b_old.target_id == t` | Active(t) | `L := Active`, no rebuild (`drive_target` sees `reuses_binding` and parks) | `activation.rs:611-619`, `501-518` |
| any with `T == Some` | `set_target(None)` | - | Unavailable-kept / bare | `L := Unavailable`, `B` untouched, watch makes `drive_target` cancel or skip the build | `activation.rs:616-617` |
| Pending(t) | `drive_target` build Ok | `T == Some(t)` under the lock | Active(t) | `B := Some(binding)`, `L := Active`, info log | `activation.rs:556-573` |
| Pending(t) | build Ok | `T != Some(t)` | unchanged | binding dropped (its `RoutingLoadContext` cancels) | `activation.rs:559-561` |
| Pending(t) | build Err | `T == Some(t)` | Pending(t) | error log; wait min(1 s, cancel, target change), then rebuild | `activation.rs:574-596` |
| Pending(t) | `target_rx.changed()` during build | - | per new T | build future dropped (`continue`) | `activation.rs:545-550` |
| any | `cancel_token.cancelled()` (router dropped, or parent) | - | terminal | `drive_target` returns; `available_worker_ids_for` reports the empty set | `activation.rs:487,510,544,588`; `PR/mod.rs:363` |
| Pending / Active | `activation_rx` (legacy oneshot) | first value | as `set_target(Some(Legacy(ep)))` | - | `activation.rs:279-286` |

### PD-2 Disaggregated request lifecycle in `PrefillRouter::generate`

Location: `PR/mod.rs:416-776`. One request, one task, plus the detached prefill observer (PD-3). Phases are encoded by position in the function plus `req.tracker.phase`, `prefill_completion: Option<PrefillTask>`, `PrefillOutcome`, and `decode_req.staged_kv_cleanup`.

| State | Representation |
|---|---|
| Gate | before `PR/mod.rs:482`; `tracker.phase` unchanged (`Aggregated` default) |
| ConditionalPlan | `conditional_disagg_policy.is_enabled()`, inside `plan_conditional_disagg_decode` (`PR/conditional_bypass.rs:162`) |
| Bypassed | `Ok(Some(decision))`: `tracker.phase = Decode`, `BYPASS_REMOTE_PREFILL_ANNOTATION` pushed, `dispatch_kv_plan` on the decode host; never touches the prefill hop or its queue (NOTE prefill-disagg-8) |
| PrefillSelecting | holds `prefill_phase_barrier` (`tracker.set_phase(Prefill)`, `PR/mod.rs:557`); inside `select_and_dispatch_prefill` before `prepare`; request context is the independent `prefill_context` (`PR/mod.rs:571,780-788`) |
| PrefillBooked | `plan_kv_route` returned (`RH/kv.rs:652-659`): `BookingHandle` armed inside the plan; `prepare` ran (`prepare_prefill_dispatch` plus `cancellation::arm_for`, `PR/mod.rs:619-636`) |
| PrefillDispatched-Bootstrap | `prepared.bootstrap_info.is_some()`: `spawn_prefill_task` owns stream and barrier; `PrefillOutcome::Bootstrap` (`PR/mod.rs:642-649`) |
| PrefillDispatched-Sync | no precomputed bootstrap: barrier dropped, `consume_prefill_stream` awaited inline (`PR/mod.rs:651-654`) |
| PrefillTerminal | `PrefillCompletion::Terminal` (EOS on the 1-token step): returns a one-item stream, no decode (`PR/mod.rs:708-715`) |
| PrefillHandoff | `Handoff{result, completion}`: `Bootstrap` if the params carry bootstrap keys, else `Completed` (`PR/mod.rs:656-678`) |
| DecodePhase | `tracker.set_phase(Decode)` permit acquired then dropped (`PR/mod.rs:737-739`); `into_decode_request` sets `staged_kv_cleanup = true`, `prefill_worker_id`, overrides `assume_kv_reuse = false`, `track_prefill_tokens = false`, `overlap_score_credit = 0` (normal disagg) (`PR/mod.rs:154-184,883-900`) |
| DecodeRouting | `next.generate(decode_req)` awaited, directly (`None` completion) or through `forward_decode` (PD-3) |
| Failed-prefill | `prefill_result` Err: classified Cancelled, AtCapacity, or Failed, returned (`PR/mod.rs:685-701`) |

Invariants:

- `prefill_phase_barrier` is a per-request semaphore permit; `set_phase(Decode)` cannot complete until it is dropped (spawned task start `PR/admission.rs:173`, or `PR/mod.rs:651`, or the async block completing on error).
- Client cancellation never stops `prefill_context` before `arm_for` runs (test `client_cancellation_does_not_cancel_prefill_context`, `PR/mod.rs:1208`); this still holds after SM-6's fix: a client stop during the queue hold ends the wait by dropping the plan future, the prefill context is not stopped. After `arm_for` a client stop reaches the prefill context only when both worker sets advertise `DISAGG_PREFILL_CANCEL_ANYTIME_V1` (`PR/cancellation.rs:15`); `prepare` and dispatch run on the prefill context alone once the plan returned, so PD-4 is unchanged.
- A stopped or killed client never aborts decode routing after prefill (`PR/mod.rs:719-730`); the decode host honours this via `StagedKv::Present -> DispatchWhenStopped`, bounded by one 120 s `CleanupBudget` (`RH/cancellation.rs:20,85-93,103-130`).
- Exactly one of {`PrefillOutcome::Terminal` returned, `into_decode_request` called} happens per attempt (`unreachable!` at `PR/mod.rs:179`).
- `prefill_release` is parked in `MigrationState` only after a successful dispatch (`PR/mod.rs:638-640`); a failed dispatch returns before the store.

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Gate | `lifecycle != Active` and snapshot `Pending`, not query-only | - | return `WorkerUnavailable` | - | `PR/mod.rs:460-472` |
| Gate | snapshot `Untargeted`, or query-only | - | passthrough `next.generate` | - | `PR/mod.rs:476-484` |
| Gate | conditional enabled | decode mode is KV, no preselected prefill worker, decode host installed, tokens non-empty | ConditionalPlan | `preview_kv_route(Decode)` (no admission), optional `prefill_worker_busy` probe | `PR/conditional_bypass.rs:169-226`, `RH/kv.rs:199-237,342-375` |
| ConditionalPlan | policy says bypass and the decode gate allows (`conditional_disagg_decode_busy_threshold`, `conditional_bypass.rs:232-242`) | - | Bypassed | `plan_kv_route_from_preview` books decode; `dispatch_kv_plan`; marker frame only if the client asked | `PR/conditional_bypass.rs:267-276`, `PR/mod.rs:496-527` |
| ConditionalPlan | `Ok(None)` | - | PrefillSelecting | - | `PR/mod.rs:528-530` |
| ConditionalPlan | `Err` Cancelled / InvalidArgument | - | return Err | parked classifier lifecycle handled in `select_with_request_lifecycle` | `PR/mod.rs:531-539`, `RH/kv.rs:144-170` |
| ConditionalPlan | other `Err` | - | PrefillSelecting | warn, fall back | `PR/mod.rs:540-547` |
| Bypassed | `dispatch_kv_plan` Err | - | return Err | booking released by `dispatch_selection` (`abort` or `release_for_retry`) | `RH/kv.rs:569-589` |
| PrefillSelecting | binding `None` after the barrier | snapshot Active | PrefillSelecting(new binding) | - | `PR/mod.rs:580-598` |
| PrefillSelecting | binding `None` | snapshot Pending / Untargeted | return `WorkerUnavailable` / passthrough | - | same |
| PrefillSelecting | prefill mode Direct, no `prefill_worker_id` | - | return Err | - | `PR/mod.rs:603-608` |
| PrefillSelecting | scheduler queue holds (class limit) | client context not stopped | PrefillSelecting | parked in the prefill partition queue (KQ-1 Queued); `select_and_dispatch_kv_prefill` awaits `plan_kv_route` under `cancel_on_stop(client_ctx)` (`engine_ctx` passed through `select_and_dispatch_prefill` and `PrefillRouter::generate`) | `RH/kv.rs` `select_and_dispatch_kv_prefill`, `RH/cancellation.rs:168-180` |
| PrefillSelecting | client context stops during the queue hold | - | Failed-prefill (`Cancelled`) | the plan future is dropped: a queued admission is retracted (KQ-2 lease drop) and an armed `BookingHandle` released; `prefill_context` itself is not stopped (SM-6 fixed) | `RH/kv.rs` `select_and_dispatch_kv_prefill` |
| PrefillSelecting | `plan_kv_route` Ok | - | PrefillBooked | `prepare`: bootstrap info, `arm_for` guard retained on `prefill_ctx` | `RH/kv.rs:652-666`, `PR/mod.rs:812-852` |
| PrefillBooked | `prepare` Err | - | Failed-prefill | `plan.abort` (lifecycle abort plus booking release) | `RH/kv.rs:660-665` |
| PrefillBooked | `dispatch_kv_plan` Err | - | Failed-prefill | guard `abort` / `release_for_retry`; no `store_prefill_release` | `RH/kv.rs:569-589`, `PR/mod.rs:637` |
| PrefillBooked | dispatch Ok, `bootstrap_info` precomputed | - | PrefillDispatched-Bootstrap | `store_prefill_release`; `spawn_prefill_task` (barrier moves into the task) | `PR/mod.rs:638-649`, `PR/admission.rs:164-191` |
| PrefillBooked | dispatch Ok, no precomputed bootstrap | - | PrefillDispatched-Sync | barrier dropped; `consume_prefill_stream` | `PR/mod.rs:651-654` |
| PrefillDispatched-Sync | first item Err, or stream empty | empty and `ctx.is_stopped()` gives `Cancelled` | Failed-prefill | - | `PR/admission.rs:27-45` |
| PrefillDispatched-Sync | first item has bootstrap keys | - | PrefillHandoff(Bootstrap) | spawns the tail observer (`completion: Some`) | `PR/admission.rs:84-105` |
| PrefillDispatched-Sync | first item terminal finish reason (not Length) | - | PrefillTerminal | `record_prefill_complete`; rest of the stream drained inline (booking released at EOF) | `PR/admission.rs:65-83,113-122` |
| PrefillDispatched-Sync | no data, no params, or TensorRT-LLM null `ctx_request_id` | - | Failed-prefill (`NoDisaggregatedParams`) | - | `PR/admission.rs:124-152` |
| PrefillDispatched-Sync | later item Err while draining | - | Failed-prefill | `record_prefill_complete` already called | `PR/admission.rs:69-73` |
| PrefillHandoff / PrefillDispatched-Bootstrap | - | - | DecodePhase | `set_phase(Decode)` waits for the barrier; `into_decode_request`; topology taints merged | `PR/mod.rs:737-755`, `902-920` |
| DecodePhase | `prefill_completion` is `None` | - | DecodeRouting | `next.generate(..).await` | `PR/mod.rs:766-773` |
| DecodePhase | `prefill_completion` is `Some` | - | PD-3 | `forward_decode(decode, engine_ctx, PrefillNeed, prefill_engine_ctx)` | `PR/mod.rs:767-772` |
| Failed-prefill | - | Cancelled / AtCapacity / Failed | return Err | log level by class; Migration may retry (`is_migratable`) | `PR/mod.rs:685-701` |

Behavior change (SM-6 fix 49c3ddcbe0): a client already stopped before the prefill hop now fails `Cancelled` immediately instead of running prefill plus a `DispatchWhenStopped` decode.

### PD-3 Prefill observer and decode merge (`PrefillTask::forward_decode`)

Location: `PR/handoff.rs:29` (`PrefillTask(JoinHandle<Result<PrefillOutput, PrefillError>>)`), `PR/handoff.rs:111-263`. States: `P` = prefill task {Running, Ok(output), Err(e)}; `D` = decode {Dispatching, Streaming, TerminalSeen, Ended, Err}; `mode` = `PrefillNeed::{None, BeforeLast, BeforeFirst}`; `buffered` = `first_decode_output` (BeforeFirst only).

Invariants:

- Dropping `PrefillTask` detaches, never aborts, the prefill task (`PR/handoff.rs:27-29`; test `disconnected_client_does_not_drop_prefill_transfer`).
- `completion` is polled to completion at most once per path (every arm that resolves it returns before any other await on it).
- Prefill is cancelled by a decode failure only for `ErrorClass::CapacityExhausted` anywhere in the chain or `UnknownPolicyClassRefusal` at dispatch time (`PR/handoff.rs:266-299`); every other decode error leaves prefill running for the in-flight KV transfer.
- A prefill failure always calls `request_context.stop_generating()` (and `stream_context` once a decode stream exists).
- Tail observer inside `consume_prefill_stream` (`PR/admission.rs:84-105`): bootstrap first item spawns the loop; each item through `check_output` (`Err` item gives `PrefillError`, `FinishReason::Error` gives error, `Cancelled` gives `PrefillError::Cancelled`); EOF gives `record_prefill_complete` and `Ok(last engine_data)`. An EOF caused by the egress ending a stopped stream is indistinguishable from success here (NOTE prefill-disagg-5).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| (P Running, D Dispatching) | P -> Err | `biased`, polled first | return Err | `request_context.stop_generating()`; decode future dropped (booking retracted or never made) | `handoff.rs:121-125` |
| (P Running, D Dispatching) | P -> Ok | - | await decode | if `emit`, prepend `output_item` to the stream; a `CapacityExhausted` decode refusal here cannot cancel the finished prefill (NOTE prefill-disagg-7) | `handoff.rs:126-131` |
| (P Running, D Dispatching) | D -> Err | - | return Err | `cancel_prefill_on_decode_error`: stop prefill only for capacity or unknown class | `handoff.rs:133-139` |
| (P Running, D Dispatching) | D -> Ok(stream) | - | Streaming | stream wraps `completion` | `handoff.rs:141-145` |
| BeforeFirst, Streaming | first item is an admission-rejection error | - | yield, end | prefill stopped | `handoff.rs:152-161` |
| BeforeFirst, Streaming | first item any other (including a non-admission error or EOF) | - | buffered; await P | holds the item until prefill resolves (NOTE prefill-disagg-4) | `handoff.rs:162-165` |
| BeforeFirst, buffered | P Ok | - | yield prefill item, yield buffered, drain | if decode already ended, do not re-poll | `handoff.rs:167-191` |
| BeforeFirst, buffered | P Err | - | yield error, end | both contexts `stop_generating` | `handoff.rs:173-178` |
| None / BeforeLast, Streaming | P resolves first | `biased` | drain decode | Ok: emit item if `emit`; Err: stop both, yield error, end | `handoff.rs:196-212` |
| None / BeforeLast, Streaming | decode item is error, `Error`, or `Cancelled` finish | - | yield, end | `cancel_prefill_on_decode_rejection` for error items | `handoff.rs:213-225` |
| None / BeforeLast, Streaming | terminal item | `emit` or EOF | await P before yielding | P Err gives the error yielded instead of the terminal | `handoff.rs:226-242` |
| None / BeforeLast, Streaming | terminal item | `!emit`, item present | yield, drain, then `let _ = completion.await` (late failure logged only, #17484) | - | `handoff.rs:243-257` |
| any Streaming | response stream dropped (client gone) | - | observer dropped | prefill continues; decode booking released by `monitor_response_stream` `stopped` arm | `RH/routing_host.rs:163-167,245-249` |

### PD-4 Client-cancellation link (`cancellation::arm_for`)

Location: `PR/cancellation.rs:9-37`; armed inside `prepare` (`PR/mod.rs:621-634`), guard retained on the prefill context. `decode_supports_prefill_cancellation` is a snapshot of the decode card at WorkerSet build (`discovery/watcher.rs:678-679`; NOTE prefill-disagg-6); `prefill_supports_cancellation` is per selected prefill worker (`PR/mod.rs:622-626`). HTTP disconnects arrive as `kill()` on the client context from a detached monitor (`http/service/disconnect.rs:321-359`); the engine future itself is spawned (`http/service/openai.rs:1358`) and is not dropped.

| State | Representation |
|---|---|
| Unlinked | before `prepare`, or `arm_for` returned `None` (either set lacks `DISAGG_PREFILL_CANCEL_ANYTIME_V1`) |
| Watching | task alive, `client.stopped()` pending |
| Stopped-propagated | `prefill.stop()` done, now awaiting `client.killed()` |
| Killed-propagated | `prefill.kill()` done, task returns |
| Detached | `prefill` `Weak` failed to upgrade, or guard dropped (`AbortOnDropHandle`) when the prefill context dies |

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Unlinked | `prepare` | both capabilities | Watching | task spawned, guard retained | `cancellation.rs:15-35`, `PR/mod.rs:627-634` |
| Watching | client stopped | `client.is_killed()` | Killed-propagated | `prefill.kill()` | `cancellation.rs:17-23` |
| Watching | client stopped | not killed | Stopped-propagated | `prefill.stop()` | `cancellation.rs:24-28` |
| Stopped-propagated | client killed | - | Killed-propagated | `prefill.kill()` | `cancellation.rs:30-33` |
| Watching | prefill context dropped | - | Detached | task aborted via the guard | `cancellation.rs:35` |

### PD-5 Prefill booking ownership

Location: `RH/request_guard.rs:428-485` (`KvRequestCleanup`), `RH/request_guard.rs:610` (`RequestGuard`), `kv_router/request_lease.rs:430-490`, `protocols/common/preprocessor.rs:230-248` (`MigrationState.prefill_release`), scheduler side `lib/router/scheduler/src/queue.rs:653-698` (KQ-3).

| State | Representation |
|---|---|
| Planned | `RoutePlan.cleanup` holds `BookingHandle{armed: true}` |
| Committed | `KvRequestCleanup::new`: `BookingHandle::commit` then `RequestAttemptLease` registered by attempt id (`request_guard.rs:446-459`) |
| Dispatched | `guard.on_dispatched()`; stream monitored by `monitor_response_stream` |
| Released-early | `RequestAttemptRelease::release()` from `Migration` before the retry dispatch (`migration.rs:665-672`) |
| Finished | `guard.complete()` / `guard.abort()` then `lease.finish()`; or `RequestAttemptLease::drop` then `enqueue_completion` |
| Expired | reaper `enqueue_expiry` (local only) |

Invariants: release is attempt-scoped and idempotent (`LeaseClock::claim_now` guards `finish`; a late `release()` after `finish` falls to `release_booking` then `free_if_booking(request_id, worker, attempt_id)`, a no-op for another attempt, `request_lease.rs:333-403`, `slot-tracker/src/request_maps.rs:120-133`); dropping an armed `BookingHandle` or a `RoutePlan` releases the booking (`queue.rs:692-698`, `RH/routing_host.rs:391-396`); dropping a `PrefillReservation` releases its booking (`PR/query.rs:13-24`); `store_prefill_release` overwrites, only `Migration`'s committed retry takes it (`migration.rs:664-672`), and an exhausted migration leaves the last prefill booked until its stream ends (test `migration_releases_prefill_booking_only_for_committed_retry`).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Planned | `prepare` Err | - | Finished | `plan.abort` | `RH/kv.rs:660-665` |
| Planned | `dispatch_kv_plan` | - | Committed | `track_planned_selection` builds the guard; `attempt_release` handle cloned first | `RH/kv.rs:667-668,379-411` |
| Committed | record routing decision Err | - | Finished | `guard.abort` | `RH/kv.rs:497-500` |
| Committed | dispatch Err | migratable with `migration_state` | Finished (classifier parked) | `release_for_retry` | `RH/kv.rs:583-586`, `request_guard.rs:927-938` |
| Committed | dispatch Err | otherwise | Finished | `guard.abort(cause)` | `RH/kv.rs:586` |
| Committed | dispatch Ok | - | Dispatched | `accept_dispatch`; `store_prefill_release` in the `PrefillRouter` | `RH/kv.rs:592-607`, `PR/mod.rs:638-640` |
| Dispatched | first tokens | - | Dispatched | `mark_prefill_completed_if_booking` once | `request_guard.rs:824-850` |
| Dispatched | EOF | - | Finished | `guard.complete` | `RH/routing_host.rs:245-246` |
| Dispatched | error item, stopped, or drain deadline | - | Finished | `guard.abort` (or `release_for_retry` for migratable error items) | `RH/routing_host.rs:163-249` |
| Dispatched | Migration retry under the same request id | - | Released-early | `take_prefill_release().release()` before re-booking; a later `finish` is a no-op | `migration.rs:665-672` |

### PD-6 `PrefillReservation` (EPP query path)

Location: `PR/query.rs:54-130`. States: Unreserved; Reserved{KV: `booking: Some`} or Reserved{builtin: advisory, `booking: None`}; Released (explicit `release().await`, acknowledged); Dropped (release enqueued). Guards: `reservation_id` non-empty; `L == Active` and `B.is_some()`, else `PrefillError::NotActivated`; `QueueRejected` bails with class and limit text. Invariant: the booking descriptor pins the exact scheduler attempt, so release after a rebind still frees the original partition (test `prefill_reservation_cleans_exact_bookings_after_rebind`). Builtin modes use `peek_next_worker` and never advance the RR cursor or acquire occupancy (tests at `query.rs:694-804`).

### PD-7 Encode hop

Location: `encoder_router.rs:385-429`. Passthrough unless `L == Active` and `should_encode` (`!is_probe && encoder_result.is_none() && has media`). Encode runs on an independent context (`encoder_router.rs:396-400`): Encoding -> {`Ok(result, link)`: attach to the request; `Err` (stream error item, no terminal, non-object result, `B == None`): log, fall back to inline encoding} -> `next.generate(original context)`. `consume_encode_stream` returns `Err` on any error item even after a terminal (`encoder_router.rs:355-358`). The downstream hop runs under `CancelWhenStopped` because `staged_kv_cleanup` is only set by `into_decode_request`, so an encode result produced after a client disconnect is never handed to a worker (finding SM-17).

### PD-8 Embedded selection partition

Location: `lib/llm/src/kv_router/embedded.rs`. Construction-time only: `RuntimeDiscoverySource.primed` (the first `next_snapshot` returns the current watch without waiting, `embedded.rs:442-462`); `affinity: OnceLock` (the first `affinity_coordinator` call wins its TTL and mode, later calls return the same coordinator regardless of arguments, `embedded.rs:392-404`); the catalog reconciler and queue-metrics updater are spawned with child tokens. No request-level state; the prefill versus decode distinction is only `worker_role` -> `WorkerType` -> partition role and the prefill-only `non_max_overlap_selection_observer` (`embedded.rs:333-341`).

Prefill-disagg coverage already in the tree (not re-derived):

- ai-dynamo/dynamo#18056 (draft): the per-class decode wait limit (`policy_classes[].decode_busy_threshold_frac`) is a no-op at the prefill hop and harmful at the decode hop in disaggregated serving; the fix makes the prefill queue the only gate and books every `RequestPhase::Decode` request directly; its known limits cover the conditional-disagg bypass skipping the prefill queue, classifiers running only on the decode router, and burst overshoot of the decode limit.
- #17455 prefill booking released on a committed migration retry (`migration_releases_prefill_booking_only_for_committed_retry`); #16766 / #14775 decode refusal and client cancellation reaching the remote prefill (`decode_rejection_reaches_remote_prefill_context`, `PR/cancellation.rs` tests, `unknown_class_decode_refusal_stops_pending_prefill`); #16736 fail closed while activating (`pending_target_fails_closed_instead_of_forwarding_to_decode`, `target_replaced_mid_flight_fails_closed_at_binding_load`); #17484 terminal decode no longer waits for prefill finish (`handoff.rs` tests); #13855 stopped decode legs still reach the worker (`RH/tests.rs:4194-4420`).
- Two-tier default notice (#17792): `entrypoint/input/http.rs:97-101` only warns when KV routing has no explicit policy instance; the prefill hop resolves its policy through the same `SelectionPolicySource` and `host_default_factory` (`kv_router/plugins.rs:20-22`, `activation.rs:372-387`), keyed by `WorkerType::Prefill`, so a future default flip applies to both hops at once and the notice has no state of its own.

## Membership (`dynamo-llm` discovery, `dynamo-runtime` client, `dynamo-scheduler` catalog)

Abbreviations: `CT:` = `lib/llm/src/discovery/controller.rs`, `CL:` = `lib/runtime/src/component/client.rs`, `W:` = `lib/router/scheduler/src/selection/core/workers.rs`, `MS:` = `lib/router/scheduler/src/selection/membership.rs`. Analyst numbering: M1 -> MB-1, M2 -> MB-2, M3 -> MB-3, M4 -> MB-4, M5 -> MB-5, M6 -> MB-6 (slot rows in ST-3), M7 -> KQ-7, M8 -> MB-7, M9 -> KI-8 and ST-5. Three membership planes exist and are joined only at the kernel: discovery admission (controller cohorts, admitted-id watch, admitted `Client` view), the runtime-config join (`RuntimeConfigWatch`, admission-filtered in the frontend since SM-9's fix, unfiltered for raw clients), and the selection-core catalog (`WorkerLifecycle`) that publishes `workers_with_configs` to the kernel and reconciles slots and index listeners.

```
discovery (AllModels)      -> MB-1 controller cohorts -> admission_tx (Vec<u64>) -> MB-2 admitted Client view -> available_instance_ids() -+
discovery (Endpoint)       -> raw Client instance_avail (MB-3) -+                                                                       |
discovery (EndpointModels) -> base cards -> MB-4 RuntimeConfigWatch (instances ∩ cards) -> MB-7 CatalogReconciler -> catalog -> workers_tx  |
                                                                                                                                  |     |
                                   kernel KQ-7: rows from workers_with_configs, filtered by availability  <-----------------------+-----+
                                   slots: reconcile_workers on every published map (MB-6, ST-3)
```

### MB-1 Discovery group status and cohorts

Location: `CT:156` (`GroupStatus`), `CT:200` (`DesiredGroup`), loop `CT:324` (`run`). Cohort sub-state: `cohorts: HashMap<checksum, BTreeSet<instance_key>>` plus `cohort_order: VecDeque<checksum>`; incumbent = `cohort_order.front()` (`CT:247`). `retained: Option<RetainedPipeline>` holds the previous admission sender while a replacement builds. `fingerprint = checksum [+ video contract]` (`CT:75`).

| State (per `GroupKey` = model name plus worker-set key) | Representation |
|---|---|
| Absent | no entry in `groups` |
| Idle | `GroupStatus::Idle` (transient inside `reconcile_group_inner`) |
| Queued | `Queued{mdc_checksum: fingerprint, committed_members}` |
| Building | `Building{fingerprint, generation, cancellation, committed_members}` |
| Ready | `Ready{fingerprint, committed_members}` |
| Retrying | `Retrying{fingerprint, deadline, committed_members}` (build failed) |
| Blocked | `Blocked{fingerprint, deadline, committed_members}` (commit or replace failed, commit withdrawn) |
| BlockedReady | `BlockedReady{fingerprint, committed_members, deadline}` (replace failed, old commit retained) |

Invariants:

- A materializing instance key is in exactly one cohort, keyed by the checksum stored in `desired[key]` (`CT:224`, `234`); `cohort_order` contains exactly the non-empty cohort checksums in first-observation order; an emptied cohort that returns joins the back (`CT:239-242`).
- `committed_members` is a subset of the incumbent cohort's members at commit time; while `retained.is_some()` the status fingerprint is the replacement's and `retained.contract` is the committed contract (`CT:191-198`).
- Build results are fenced by `(fingerprint, generation)` and `selected_checksum()` (`CT:975-992`); `next_build_generation` is unique, never 0 (`CT:910`). At most one `Building` per group; `active_builds <= max_concurrent_builds`.
- `admission_tx` of a group always carries ids within the incumbent members; the retained sender carries ids within the surviving committed members (`CT:670-683`).
- An instance key's `(group_key, mdc_checksum)` is immutable for its lifetime (`CT:439-451`); the discovery layer rejects non-taint same-id card mutations (`lib/runtime/src/discovery/mod.rs:1430-1469`), so a same-id change is only observable as Removed then Added.
- The first accepted configuration is frontend-local and sticky: a rejected cohort becomes eligible only when the incumbent cohort empties (`CT:247`, `589-654`); stated as intentional in `lib/llm/src/discovery/AGENTS.md` (NOTE membership-7).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| any | `Added(inst)`, key known, same checksum, projection, contract, limit | - | same | revision bump only | `CT:428-438` |
| any | `Added(inst)`, key known, materializing, group_key or checksum differs | - | same | ERROR log, revision bump, card ignored | `CT:439-451` |
| Absent / any | `Added(inst)` new key, materializing | `affects_selected` (checksum == front) | reconcile | cohort insert, `report_rejections` | `CT:460-476` |
| any | `Added(inst)` new key, materializing, not the incumbent checksum | - | same | cohort insert, ERROR "Rejected incompatible workers" once per checksum, observation publish | `CT:467,476,251-268` |
| any | `Removed(key)` | key in the incumbent cohort | reconcile | `desired.remove`, cohort remove | `CT:486-513` |
| any | `Removed(key)` of the last incumbent member | cohorts non-empty | reconcile with the new front | `fingerprint_changed` path | `CT:589-654` |
| any | `Removed(key)` of the last member overall | `cohorts.is_empty()` | Absent-like (group kept, `Idle`) | both admission senders get `[]`, cancel build, `remove_group` if committed | `CT:589-599` |
| Idle / Queued / Retrying / Blocked | reconcile, fingerprint unchanged | - | same kind (Retrying or Blocked only if `!desired_changed`, else Queued) | `admission_tx <- admitted(members)` | `CT:686-692,761-805` |
| Ready / BlockedReady | reconcile, fingerprint unchanged | `replace_group` Ok | Ready | admission narrowed to the intersection, then widened to the new members | `CT:694-724` |
| Ready / BlockedReady | reconcile, fingerprint unchanged | `replace_group` Err, members == committed | BlockedReady (deadline) | admission restored to old | `CT:725-740` |
| Ready / BlockedReady | reconcile, fingerprint unchanged | `replace_group` Err, members != committed | Blocked, `committed_members: None` | admission `[]`, `remove_group` | `CT:741-755` |
| any | reconcile, fingerprint changed, surviving commit | - | Queued (retained pipeline) | fresh `admission_tx`; the old sender moves to `retained` | `CT:621-640,797-805` |
| any | reconcile, fingerprint changed, no surviving commit | - | Queued, `committed_members: None` | old senders get `[]`, `remove_group`, fresh channel | `CT:641-652` |
| Queued | `start_queued_builds` | `active_builds < max` | Building(gen) | spawn `host.prepare(spec, admission_rx, cancel)` | `CT:863-959` |
| Building | result Prepared, current | `commit_group` / `replace_prepared_group` Ok | Ready | `admission_tx <- admitted(members)`, `retained = None` | `CT:1008-1034` |
| Building | result Prepared, current | commit Err | Blocked | admission `[]` | `CT:1035-1048` |
| Building | result Failed, current | - | Retrying (backoff 1, 2, 4, 8, 16, 30 s) | - | `CT:1051-1066,1330` |
| Building | result Cancelled, current | - | Queued | - | `CT:1067-1072` |
| any | result not current (fenced) | - | unchanged | `discard_prepared` | `CT:999-1005` |
| Retrying / Blocked | deadline due | - | Queued | - | `CT:1089-1126` |
| BlockedReady | deadline due | - | Ready, then reconcile(false) | retries `replace_group` | `CT:1105-1116,1130` |
| any | `Resync(instances)` | - | per-instance Added / Removed | `resync_revision` fence drops older periodic lists | `CT:407-421,1163-1228` |
| any | periodic list (`RECONCILIATION_INTERVAL`) | `revision >= resync_revision`, per-instance revision fence | per-instance Added / Removed | - | `CT:1152-1228` |
| any | stream `None` | - | frozen | WARN "retaining committed serving state"; `shutdown_builds` empties every admission sender | `CT:340-345,1135-1150` |

### MB-2 Admission channel, retained pipeline, admitted Client view

Location: `lib/llm/src/discovery/worker_set.rs:46` (`CommittedWorkerSetTarget`), `CL:690` (`with_admitted_instances_and_cancellation`), watcher wiring `lib/llm/src/discovery/watcher.rs:477-505`. States of the admitted view (per WorkerSet build): Live (updater task running; `instance_source` = discovered ∩ admitted); Emptied (`admitted_ids` sent `[]`, `filter_admitted_instances` returns `[]`, `CL:746-748`); Frozen (updater exited: `admitted_ids.changed()` Err after the sender was dropped, or `cancel_token`; the last value stays). Invariants: a view only publishes ids within its admission watch value; a retained pipeline's view keeps its own sender and never sees successor ids (`CT:631-640`); tokio delivers an unseen value before `Err(closed)` (`tokio-1.54.1/src/sync/watch.rs:979-999`), so `send_replace([])` followed by sender drop (`CT:642-651`) always empties before freezing. A view whose admitted ids were never discovered stays uninitialized (MB-3) and reports `None`, which the kernel reads as "no hard filter"; since SM-9's fix the candidate map itself is admission-filtered (MB-4), so this `None` is no longer load-bearing for admission. `Client::admitted_ids()` (additive field and accessor in `CL`) exposes the admission receiver the config wrapper consumes.

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Live | endpoint instances or admitted ids change | - | Live | recompute the intersection, send if changed | `CL:704-729` |
| Live | admission `[]` | - | Emptied | view routes nothing; `available_instance_ids()` is `Some(empty)` once initialized | `CL:746` |
| Live / Emptied | admission sender dropped or lifecycle cancel | - | Frozen | last value retained | `CL:707,714-717` |
| any | WorkerSet dropped | - | Frozen | `lifecycle_cancellation.cancel()` | `worker_set.rs:634-640` |

### MB-3 Runtime Client routing instances

Location: `CL:127` (`RoutingInstances`), `CL:357` (`RoutingInstancesState`), monitor `CL:904`. Fields per client: `discovered_ids`, `routable_ids` (discovered minus `report_instance_down`), `monitor_overloaded_ids` ∪ `request_overload_deadlines` = `overloaded_ids`, `availability_initialized`. Availability: Uninitialized (`availability_initialized == false`, never saw a non-empty discovered list) gives `available_instance_ids() == None` (no hard-availability filter); Initialized (first non-empty discovered list seen) gives `Some(routable_id_set)`, authoritative, possibly empty. Invariants: `routable` within `discovered`; `free = routable \ overloaded`; `overloaded` entries are dropped for ids that left discovery (`CL:240-245`); `instance_avail_tx` publishes `routable_ids` on every `update(.., true)`. Fault inhibition is cleared by any discovery change, not only after `reconcile_interval` (NOTE membership-5). The `Uninitialized -> None` semantics are untouched by SM-9's fix and no longer load-bearing for admission.

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| any | `reconcile_discovered(ids)` (instance source changed, or every `reconcile_interval`) | - | routable = discovered | clears all `report_instance_down` inhibition; prunes overload marks of departed ids; initialized if non-empty | `CL:237-255,918-953` |
| any | `report_instance_down(id)` | - | routable \ {id} | publish | `CL:257-272` |
| any | `mark_overloaded_until` / `set_overloaded_instances` | - | overloaded updated | request marks expire by deadline; a monitor update clears request marks | `CL:287-327` |
| any | `clear_overloaded_for_removed` (worker monitor) | - | same | - | `CL:329-341` |
| any | instance source sender dropped | - | monitor exits | `cancel_token.cancel()` | `CL:940-946` |

### MB-4 RuntimeConfigWatch (instances ∩ base cards)

Location: `lib/llm/src/discovery/runtime_configs.rs:30` (`base_runtime_config_watch`), `:129` (`runtime_config_watch`), cached per endpoint `model_manager.rs:2764`. Join task: Running, Stopped (primary cancel, `lifecycle` cancel, `tx.closed()`, either source closed); card task: Running, Stopped (lifecycle cancel, stream `None`, `tx.send` Err). Stopped is terminal: nothing re-subscribes; every consumer (`LocalScheduler` monitor, `CatalogReconciler::run`, approximate-LRU reconciler, worker monitor) exits on `changed()` Err and keeps its last map. Invariants: a worker appears in the map iff its id is in the raw client's `instance_avail` and a base card (no LoRA, no suffix) with a valid dp range was seen for it. `admitted_runtime_config_watch` (`runtime_configs.rs`) wraps that map for a router whose `Client` carries an admission receiver (`KvRouter::new_with_worker_role_and_scheduler_load`): configs intersected with `Client::admitted_ids()`, both watches tracked, bounded by the router token, frozen when the admission watch closes like the admitted view (SM-9 fixed; SM-19 covered by the same change). A raw client has no admission watch, so standalone, hosted, and legacy paths keep the unfiltered map and the availability provider's `None` still means no filter. Same-id card mutations other than taints never arrive (rejected at `lib/runtime/src/discovery/mod.rs:1442-1462` and `kv_store.rs:291-305`); `Resync` is ignored (`:99`), safe because the backend diff emits `Removed` for departed ids inside one stream (`mod.rs:1480-1485`).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Running | card `Added` (base, valid dp range) | - | Running | `configs.insert`; send if changed | `:47-74` |
| Running | card `Added` with an invalid dp range | - | Running | `configs.remove(id)` | `:65-71` |
| Running | `ModelTaintsUpdated` | - | Running | taints replaced in place (no slot churn downstream) | `:76-92` |
| Running | card `Removed` | - | Running | `configs.remove` | `:93-97` |
| Running | instance_avail change | - | Running | map recomputed, sent iff changed | `:157-189` |
| Running | stream `None` or cancel | - | Stopped | receivers see `changed()` Err | `:38-45,160-164` |
| Running (admitted wrapper) | base map or admitted ids change | `Client` has an admission receiver | Running | intersection recomputed, sent iff changed | `runtime_configs.rs` `admitted_runtime_config_watch` |
| Running (admitted wrapper) | admission watch closes or router token cancelled | - | Frozen | last intersection kept (like the admitted view, MB-2) | `runtime_configs.rs` `admitted_runtime_config_watch` |

### MB-5 KV-source membership and its coordinator

Location: `lib/llm/src/discovery/kv_source_membership.rs:101` (`KvSourceStatus`), `:242` (`KvSourceMembership`); coordinator `lib/llm/src/discovery/kv_source_watch.rs:135`. Per logical rank `(kv_state_endpoint, worker, dp_rank)`: Missing (no advertisement, or rank not in `runtime_configs`); ActiveRecoverable(S) (one incarnation, `recovery_target.is_some()`); ActiveLiveOnly(S); Ambiguous(Incarnations) (more than one publisher id advertises the key); Ambiguous(ConflictingDescriptor) (a publisher re-advertised different typed metadata); Ambiguous(EndpointMapping) (base cards disagree on the kv-state endpoint); Suppressed (a versioned source owns the rank). Coordinator: Bound(stream); Unbound+Retry (delay 100 ms to 5 s doubling, only when resolution is `Resolved`); Unbound (no retry, resolution `Ambiguous`); Stopped. Invariants: `publisher_sources[p] = source_id` iff `advertisements[source_id.key][p]` exists; removal is fenced by exact `(key, publisher_id)` (`:306-324`); a publisher never owns two keys (`:281-295`). Receiver `REC/worker_query.rs:234` (`reconcile_view`): a rank absent from the view triggers `remove_unexpected_key`, deactivate, and `reset_worker_dp_rank_and_wait` (`:365-381`, `target.rs:100-111`); a rank whose status lost its active source is reset (`:281-289`). This is the only path that clears a departed worker's blocks from the frontend's primary index when KV events are subscribed; the embedded `RuntimeIngress` has no `detach` or `reconcile` (`lib/llm/src/kv_router/indexer/ingress.rs:151-156`). Approximate-LRU mode additionally resets ranks via `reconcile_approximate_lru_snapshot` (`lib/llm/src/kv_router.rs:242-322`). A stream error or end resets every rank (NOTE membership-6, intended fail-closed; test `source_stream_error_clears_membership_and_requests_rebind`).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Missing | Added source (scoped, consistent) | - | Active* | - | `kv_source_watch.rs:258-289`, `membership.rs:266-304` |
| Active* | second incarnation Added | - | Ambiguous(Incarnations) | fails closed until one leaves | `membership.rs:353-358` |
| any | same publisher, different descriptor | - | Ambiguous(ConflictingDescriptor) | sticky until that publisher is removed | `membership.rs:289-294,327-331` |
| Active* / Ambiguous | Removed(publisher) | - | recomputed | - | `membership.rs:333-336` |
| Bound | runtime configs change, resolution unchanged | - | Bound | view republished (ranks that left configs vanish) | `kv_source_watch.rs:186-188` |
| Bound | resolution changes | - | rebind | membership reset to empty, stream dropped, rebound | `:177-185` |
| Bound | stream `Err` | - | Unbound+Retry | membership reset to empty (every rank Missing) | `:190-197,304-308` |
| Bound | stream `None` | - | Unbound+Retry | membership reset to empty | `:198-202` |
| Unbound+Retry | timer | - | Bound, or backoff x2 (cap 5 s) | - | `:203-210` |
| any | `Resync` event | - | unchanged | ignored (diff events carry the change) | `:300-303` |

### MB-6 Kernel worker configuration and slot topology

Location: `lib/router/scheduler/src/local.rs:152` (`reconcile_worker_configs`), monitor task `:212-257`, sync publish `:321`. Per worker in the kernel: In configs (key in `workers_with_configs`) or Not; the slot tracker is reconciled to the configs map on every published change (synchronously by the publisher through `apply_published_worker_configs` under `applied_worker_configs`, and again by the monitor task; both idempotent via `last_workers`). The slot-side states and rows (Unregistered, Registered rank, Lazy rank; `reconcile_workers`, `ensure_worker`, lazy booking) are ST-3; the frontend source converts the whole `workers_with_configs` map to `WorkerRequest`s with `endpoint = dyn://{id}`, `max_num_batched_tokens` defaulted, `is_eagle` from the router (`embedded.rs:468-`); in the frontend that map is the admitted intersection since SM-9's fix (`KvRouter::new_with_worker_role_and_scheduler_load` wraps `workers_with_configs` in `admitted_runtime_config_watch` when the router's `Client` carries an admission receiver, MB-4), so a non-admitted worker never enters slots or the catalog; raw clients (standalone, hosted, legacy paths) keep the unfiltered map.

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| any | configs map published (differs from `last_workers`) | `WorkerDpRange::validate` ok for all | slot ranks == configs | `reconcile_workers` plus `set_worker_capacities` (ST-3); in the frontend the published map is already the admitted intersection (MB-4, SM-9 fixed) | `local.rs:152-176` |
| any | configs map published | some range invalid | unchanged (`Rejected`) | ERROR; kernel rows and slots diverge until the next valid publish | `local.rs:162-165` |
| applied | monitor task wakes | map unchanged since the sync apply | unchanged | idempotent via `last_workers`; `Update` command forces a full recheck | `local.rs:212-257,238-242` |
| selection done, booking pending | config publish removes W between `select_worker_for_request` and the booking | removal visible in `workers_with_configs` at the booking check | Refused `NoEndpoints` via `add_request_if_registered_admitted` (SM-10 fixed) | no slot created | `queue.rs` `book_and_respond` |
| selection done, booking pending | removal published and reconciled between the config check and the slot lookup | - | Lazy rank for W plus booking (ST-3) | `observe_worker_registered(W)` after `removed(W)`; phantom slot until the next reconcile (SM-10 residual) | `multi_worker.rs:1541-1551` |

### MB-7 Selection core catalog lifecycle

Location: `lib/router/scheduler/src/selection/types.rs:38` (`WorkerLifecycle`), `W:14-226`, `selection/catalog.rs`, `MS:44` (`CatalogReconciler`), HTTP `lib/router/kv-router/src/services/selection/server.rs:79-116` (`POST /workers`, `PATCH /workers/{id}`, `DELETE /workers/{id}`), Python `lib/bindings/python/rust/llm/kv.rs:865-904`, frontend source `lib/llm/src/kv_router/embedded.rs:432-463`.

| State (per worker id, partition-independent) | Representation | In `scheduler_configs_for_key` |
|---|---|---|
| Absent | no record | no |
| Incomplete | `lifecycle == Incomplete`, `not_schedulable_reasons` non-empty (missing endpoint, block_size, dp_size, `max_num_batched_tokens` when queueing, ingress metadata, or `reconciliation failed`) | no |
| Schedulable | `Schedulable`, reasons empty, listeners registered per rank when the core listens | yes |
| Draining | `Draining` (transient during delete, partition move, or loss of schedulability; also left behind by a cancelled delete, NOTE membership-4) | no |
| Unschedulable | set just before `catalog.remove` in `delete_worker` | no |

Invariants:

- All catalog mutation (`upsert_workers`, `patch_worker`, `delete_worker`) is serialized by `catalog_updates` (async mutex) across the indexer awaits (`W:50,75,91`).
- The Draining publish (worker removed from the map) precedes indexer cleanup; the final publish follows the catalog replace (`W:165-171`); `publish_scheduler_config` sends only if the map changed and then reconciles slots synchronously (`W:413-437`).
- A previous `Draining` record is finished (indexer detach) before its id is reused (`W:178-185`).
- Strict booleans: `is_eagle` and `engine_owned_mm_identity` accept only JSON `true | false | null` on create and patch, including through pythonize (`types.rs:337,357,373,386,504-530`). `WorkerPatchRequest` fields are "absent = unchanged"; a scalar cannot be cleared to `None`; map and set fields can be emptied (`types.rs:391-455`; NOTE membership-8).
- dp range: `dp_start + dp_size` must not overflow and `dp_size <= 4096`, checked before any per-rank work (`W:131-143`).
- `CatalogReconciler`: `tracked` includes every id it introduced; `converged[id]` holds the desired record only while the core reported `Schedulable`; snapshots with duplicate ids are rejected before mutation (`MS:80-157`).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Absent / Incomplete | upsert, metadata complete, ingress reconcile Ok | - | Schedulable | partition ensured (`ensure_entry`), listeners registered (KI-7), publish | `W:34-68,172-226` |
| Absent / Incomplete | upsert, metadata missing or ingress Err | - | Incomplete | detach on Err; publish (no-op map) | `W:149-160,202-215` |
| Schedulable | upsert or patch same key, still schedulable | - | Schedulable | per-rank listener diff (`zmq_ingress.rs:78-158`), replace, publish; slots keep common ranks | `W:186-226` |
| Schedulable | upsert or patch, key changes or becomes non-schedulable | - | Draining then new record | immediate Draining publish, detach, then the new record and publish | `W:187-200` |
| Draining (left by a cancelled delete) | upsert | - | as Absent | detach first (test `cancelled_worker_delete_is_finished_before_re_registration`, `core/tests.rs:3847`) | `W:178-185` |
| Schedulable / Incomplete | delete | - | Draining, Unschedulable, Absent | publish, detach (await), publish, remove; a dropped handler future leaves Draining | `W:87-111` |
| Absent | delete or patch | - | Absent | `NotFound` (the reconciler treats it as success) | `W:80,92-95` |
| any | `ready()` | - | same | ready iff `schedulable_count > 0` and not cancelled | `W:121-129` |
| reconciler | snapshot | record == `converged[id]` | skip | - | `MS:96` |
| reconciler | snapshot | upsert returned non-Schedulable | retried on every later snapshot | WARN | `MS:117-128` |
| reconciler | snapshot | id tracked but absent | delete | `NotFound` tolerated; other errors keep it tracked | `MS:131-155` |
| reconciler | pass failed | - | pending retry 5 s unless a newer snapshot arrives | - | `MS:167-199` |
| reconciler | source closed | - | catalog frozen at the last state | debug log | `MS:173-176` |

Membership interleavings traced and sound:

- Prepared-rows generation guard (KQ-7): sound and conservative; tests `prepared_rows_are_rebuilt_when_selection_inputs_change`, `prepared_rows_recheck_config_under_selection_borrow` (NOTE membership-9).
- Indexer-service HTTP `deregister*` do not take `lock_lifecycle` while `register` does (`services/indexer/registry.rs:436,487-619`); because `deregister` removes the entry and cancels listeners synchronously and `indexer.remove_worker` is enqueued on first poll, a concurrent re-register for the same id orders its events after the removal unless the ingest queue is full; no block-losing sequence was completed (NOTE membership-10; the kv-indexer analyst's SM-16 is the queue-full variant). The standalone selection core and Python bindings serialize create, patch, and delete under `catalog_updates`; the embedded reconciler is the only writer in the frontend.
- Cohort succession and stickiness: `competing_cohorts_preserve_serving_until_complete_incumbent_drain`, `snapshots_preserve_priority_and_recreated_cohorts_join_the_end`, `frontends_keep_their_locally_first_configuration`; the ERROR log fires once per rejected checksum and again after succession (`CT:251-268,653`).
- Same-id dp changes are handled per rank with bookings kept on common ranks (`topology.rs:236-271`; `session_dp_rank_shrink_rebinds_only_removed_targets`, `overflowing_worker_dp_range_patch_preserves_live_worker`).
- `CatalogReconciler` re-upserts every non-schedulable worker on each snapshot (churn only, `MS:117-128`) and would not re-upsert a converged worker deleted out of band (no such deleter exists in the frontend).

## Transports (`dynamo-kv-router` services, `dynamo-llm` publisher and pool relay)

Abbreviations: `RSY:` = `lib/router/kv-router/src/services/common/replica_sync.rs`, `PUB:` = `lib/llm/src/kv_router/publisher`, `REL:` = `lib/llm/src/pool_relay`. Analyst numbering: T1 -> KI-7, T2 -> KI-6, T3 -> TR-1, T4 -> TR-2, T5 -> TR-3, T6 -> TR-4, T7 -> TR-5, T8 -> TR-6, T9 -> TR-7, T10 -> TR-8, T11 -> TR-9, T12 -> TR-10. There are no continuation tokens anywhere in this area: every WAN stream is server-push with `(base_sequence, sequence)` per frame, and the only resume primitive is "resubscribe and receive a fresh snapshot".

### TR-1 Replica-sync mesh (router to router)

Location: publisher task `RSY:103` (`start_replica_publisher`), runtime `RSY:31` (`ReplicaSyncRuntime`), peer set `RSY:203` (`PeerManager`), HTTP control `replica_sync_http.rs:80`; transport-free plumbing `lib/router/scheduler/src/replication.rs:128` (`ReplicaSyncConfig`), `:448` (`ReplicaInbox`); receiver ST-4. States that exist: publisher task `running | sequence_closed | affinity_closed | stopped` (two booleans, `RSY:121-139`); `PeerManager.peers: HashSet<String>` (endpoint known or unknown). States that do not exist: peer connected, caught up, behind, dead, replaced; there is no peer identity beyond the random `process_id` (`RSY:64`), used only to drop self-echo, and no sequence number, snapshot, or catch-up on the wire (`ScopedReplicaEvent` `replication.rs:104`; NOTE transports-6).

Invariants: one PUB socket per router bound at `tcp://*:port`, one SUB socket per router connected to every peer; topics `dynamo.slot-tracker.v1` and `dynamo.session-affinity.v1` are prefix-disjoint; every send is `try_send` into bounded channels (100k sequence, 4k affinity), drops counted and rate-logged; the PUB drops for a slow subscriber at libzmq HWM; stale remote bookings are reclaimed only by expiry (300 s, checked every 30 s, forced sweep every 60 s: ST-8).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| no peer | `POST /replica_sync/register_peer` | valid endpoint, not in the set | peer known | `socket.connect`, 201 | `RSY:361-374`, http `:33` |
| peer known | register again | - | same | 200, no-op | `RSY:362` |
| peer known | `POST /replica_sync/deregister_peer` | in the set | no peer | `socket.disconnect`, 200 | `RSY:375-388` |
| no peer | deregister | - | same | 404 | http `:53` |
| any | manager cancelled | - | task exit | commands answer 503 (`Unavailable`) | `RSY:273-277,306` |
| publisher running | `rx` closed | - | sequence_closed | keeps serving affinity | `RSY:126-132` |
| publisher running | cancel | - | stopped | queued-but-unsent events discarded (linger 0) | `RSY:125`, `zmq.rs:30` |
| receiver | peer P restarts (new `process_id`) | - | P's old bookings remain | reclaimed by expiry only | `single.rs:344-370` |
| receiver | `Free` / `MarkPrefillCompleted` for an unknown request | - | dropped | a late joiner never learns pre-existing requests | `slot-tracker replica_sync.rs:245-250,289-294` |

### TR-2 Worker-side KvEventPublisher lifecycle

Location: `PUB/mod.rs:275` (`KvEventPublisher`), `:115` (`KvEventSource::Zmq`), `:122` (`supervise_zmq_listener`), processor task `:462-572`. Fields: `source: Option<KvEventSource>`, `cancellation_token`, `startup: Option<oneshot::Receiver>`, `processor_handle: Option<JoinHandle>`; processor phases: creating event publisher, starting recovery endpoint, advertising discovery EventSource, processing, unregistering.

Invariants: the ZMQ listener is spawned before the processor task and buffers into an unbounded channel (`:433-441`, `:401`); `ready()` resolves only after the discovery advertisement succeeded (`:548`); any listener exit that is not a cancellation cancels the whole publisher (`:129-151`); `recovery` requires `enable_local_indexer` and excludes `liveness` (`:175-178`, `:420-430`); `shutdown()` aborts the listener and cancels; `stop_and_wait()` additionally joins supervisor then processor (`:737-758`); discovery is unregistered after the processor loop returns (`:559-566`); independent engine-publisher restart under the same publisher id is documented as unsupported (`:271-274`; finding SM-20).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| new | `EventPublisher::for_endpoint_id` fails | - | stopped | `rx` dropped; listener `tx.send` fails and exits; supervisor cancels; `ready()` gives Err | `:464-476` |
| new | recovery endpoint start fails | local indexer | processing (live-only source) | `recovery_target: None` advertised | `:479-503` |
| new | cancelled before advertising | - | stopped | recovery endpoint shut down, no discovery record | `:505-510` |
| advertising | discovery register fails | - | stopped | error logged, recovery endpoint shut down | `:537-546` |
| processing | listener task returns or aborts, token not cancelled | - | stopped | error log, cancel token, processor flushes the pending coalesced batch, publishes, unregisters | `:129-151`, `event_processor.rs:35-41` |
| processing | `shutdown()` / drop | - | stopping | listener aborted; events still queued in `rx` are not drained (NOTE transports-11) | `:737-745`, `event_processor.rs:34-41` |
| processing | Cleared reset fails locally | local indexer | stopped (fenced) | token cancelled, source withdrawn | `event_processor.rs:149-163` |
| processing | recovery barrier fails | - | stopped | `done.send(Err)`, loop breaks, token left alive so the listener reports | `event_processor.rs:52-76` |

### TR-3 Publisher non-recovery ZMQ listener

Location: `PUB/zmq_listener.rs:62` (`start_zmq_listener`), socket `lib/llm/src/utils/zmq.rs:57` (`connect_sub_socket`), monitor variant `:67`. States: connecting (tmq connect, no handshake wait); receiving with per-rank `last_source_cursors: HashMap<rank, seq>` and `expired: bool`; exited. Invariants: a rank-dedicated socket ignores batches without its rank (`:183-186`); discontinuity = `previous + 1 != seq` (`:189-191`); residency is cleared on a gap or heartbeat expiry only when `liveness` was negotiated (`:195`, `:132-152`); the first batch of a rank never counts as a gap.

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| connecting | connect error | - | exited | publisher stops (TR-2) | `:105-111` |
| receiving | batch seq == prev + 1 | - | receiving | normalize, send | `:189-226` |
| receiving | batch seq != prev + 1 | liveness | receiving | synthetic `AllBlocksCleared` then the batch | `:195-217` |
| receiving | batch seq != prev + 1 | no liveness | receiving | warn only; stale residency kept | `:196-204` |
| receiving | engine restarts at the same address (seq 0) | - | same as gap | with liveness: clear (correct); without: nothing (finding SM-20) | `:189-204` |
| receiving | no batch for `timeout` | liveness, `!expired` | receiving, `expired = true` | synthetic clear; timer disarmed | `:132-152` |
| expired | next batch | - | receiving | `expired = false`; batch applied on top of the cleared index | `:187-188` |
| receiving | recv Err, stream end, or `tx` closed | - | exited | publisher stops | `:157-161`, `:221-224` |

### TR-4 Publisher recovering listener (replay recovery)

Location: `PUB/zmq_recovery.rs:299` (`recovering_listener`), `:388` (`recovery_session`), `:45` (`ReplayCursor`), status `:117` (`KvStreamStatus`), command `:125` (`KvStreamCommand::Restart`). States: Recovering (round open, `deadline: Some`); Success (`deadline: None`, `is_success`); Failed{reason} (round exhausted: waits for `Restart`; or fatal: final); MissingHistory{expected, got} (final). Session sub-phases: connecting (both handshakes, 5 s), requesting, replaying (`is_replaying`, 5 s inactivity), live.

Invariants: cursor `next` advances only through `apply` in sequence order (`:263-295`); batches below `next` are ignored (`:65-73`); pending buffer bounded 4096 batches / 32 MiB, overflow is a `Retry` (`:77-81`); `Success` is set only after an index barrier (`:486-490`); a live gap after `Success` revokes it and re-requests (`:498-502`); a final status refuses `Restart` (`:374-380`); any exit path reports `Failed` unless already final (`ReportStop`, `:213-230`; test `stopped_listener_does_not_leave_success_behind`); only an exhausted round accepts `Restart`.

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| Recovering | handshakes ok, replay End, no gap, barrier ok | - | Success | `deadline = None`, status Success | `:468-491` |
| Recovering | reply starts past the cursor and the cursor is unchanged at End | - | MissingHistory (final) | sockets dropped | `:479-481,318-325` |
| Recovering | hole inside the reply, or gap remains at End | - | Recovering | re-request from the cursor | `:461,482-483` |
| Recovering | connect, replay, or live transport error, or deadline | within the round | Recovering | 500 ms backoff, new session, `clear_pending` | `:333-346,396` |
| Recovering | same, round exhausted | - | Failed (restartable) | status Failed; waits for `Restart` | `:347-370` |
| Failed (restartable) | `Restart{deadline}` | - | Recovering | `accepted` sent, new round | `:360-370` |
| Recovering | malformed replay envelope, `tx` closed, barrier error, sequence exhausted | - | Failed (final) | - | `:327-332,457,287,293,486-487` |
| Success | live seq > cursor | - | Recovering (new round) | `revoke`, replay requested | `:498-502` |
| Success | live seq < cursor (engine restarted at the same address) | - | Success | batch ignored (debug) (finding SM-20) | `:65-73` |
| Success | live or replay monitor event (DISCONNECTED) | - | Recovering | retry path | `:437-438` |
| any | cancel | - | exit | `ReportStop` gives Failed unless final | `:313,213-230` |

### TR-5 Event processor batching

Location: `PUB/event_processor.rs:20` (processor), `:31` (`last_raw_input_id`), `:82-105` (input gap detection), `:115-168` (Stored and Removed batched; Cleared flushes, clears the dedup domain, emits with `next_publish_id`, fences on local reset failure), `:52-76` (`Barrier` gives flush plus `flush_local_indexer`); coalescer `batching.rs:21`, `BatchingState` `batching.rs:113`. The kv-indexer analyst's section 12 is this machine. States: idle (`pending: None`); pending (one coalesced Stored or Removed event); flushing; barrier (recovery); fenced (token cancelled after a failed local reset).

Invariants: a source list is processed whole before returning to `select!` (`:80-82`); `Cleared` always flushes first, then passes through as its own barrier (`batching.rs:44-46`); a batch never exceeds `max_batch_blocks` (128); `next_publish_id` is monotonic and allocated at emit; the timer fires only when `timeout_ms` is set and something is pending (`:184-192`); raw-id gaps are logged and counted as "engine dropped events" (`:84-104`), which assumes one producer of raw ids although `next_event_id` is shared by the ZMQ listener and the `publish*` API (`mod.rs:289,729-731`, `zmq_listener.rs:270`; NOTE transports-8); a failed local `Cleared` fences the whole publisher (`:149-163`).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| idle | Stored / Removed input | - | pending | coalesce; timer armed if `timeout_ms` | `:115-168`, `:184-192` |
| pending | more input | batch under `max_batch_blocks` | pending | coalesce | `batching.rs:21-113` |
| pending | batch full, timer, or `Cleared` | - | flushing then idle | publish with `next_publish_id` | `:115-168` |
| any | `Cleared` | - | barrier | flush, clear the dedup domain, emit `Cleared`, `flush_local_indexer` | `batching.rs:44-46`, `:52-76` |
| barrier | local reset fails | local indexer | fenced | token cancelled, loop broken (TR-2 stopped) | `:149-163` |
| any | cancel | - | exit | only the pending coalesced event is flushed; `rx` not drained (NOTE transports-11) | `:34-41` |

### TR-6 Remote indexer client

Location: `lib/llm/src/kv_router/indexer/remote.rs:34` (`RemoteIndexer`), `:265` (`ServedIndexerService`), `:295` (`ensure_served_indexer_service`). There is no connection state machine: each call re-reads the cached instance sets (`:143-178`) and uses round-robin `PushRouter` with `fault_detection: false`. Topology predicates evaluated per call: no instances (allowed, passes); event-driven with a record endpoint present (error); approximate without exactly one query and one record instance from the same singleton (error). Served side: a process-wide singleton per `(connection_id, namespace, component)` with a mode fixed at first start (`:333-350`) and per-model bindings removed on handle drop (`:289-293`). No transition table: only guards.

### TR-7 KV metrics subscriber (direct ZMQ) and fan-in source

Location: `lib/llm/src/kv_router/metrics_subscriber.rs:38` (`KvMetricsSubscriber`), mailbox `:149`, fault latch `:217`; source loop `lib/llm/src/direct_zmq_fan_in.rs:482` (`run_source`), events `:36` (`FanInEvent`), `ContinuityMode::Disabled` for metrics (`metrics_subscriber.rs:388`). States: mailbox per worker `absent | pending(merged latest)`; `fault: Option<DirectKvMetricsFault>` (latched, delivered before any later load); fan-in source `connecting(backoff) | consuming | disconnected(backoff)` with `connected_once` deciding whether a connect counts as `Reconnect`. Invariants: one pending entry per worker rank, later fields overwrite (`:95-124`); capacity per shard, overflow drops the newest (`:116-118`, `:200-207`); `send_fault` holds every shard lock while clearing and latching so no load sent before the fault survives it and no load sent after is ordered before it (`:217-241`, `:272-274`); the consumer (`lib/llm/src/discovery/worker_monitor.rs:1082-1093`) logs a fault and continues; the previously applied `WorkerLoadState` is not reset, so threshold decisions run on the stale value until the next absolute report (NOTE transports-9).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| connecting | connect ok | - | consuming | `Reconnect` observed if not first | `fan_in.rs:502-524` |
| connecting | connect err | - | connecting | backoff doubles to `MAX_BACKOFF` | `:507-516` |
| consuming | receive error, stream end, or grouped `disconnected` | - | disconnected | `Disconnected` observed; metrics fault latched, pending loads cleared | `:553-559`, `metrics_subscriber.rs:76` |
| consuming | envelope decode error or identity mismatch | - | consuming | fault latched (`EnvelopeDecode` / `IdentityMismatch`), pending cleared | `:604-622`, `metrics_subscriber.rs:74-75` |
| consuming | payload decode error | - | consuming | `PayloadDecode` fault latched | `metrics_subscriber.rs:364-370` |
| any | mailbox consumer closed | - | source handler cancels the subscriber | `handler_cancel.cancel()` | `metrics_subscriber.rs:371-374` |
| any | supervisor discovery watch reset | - | same | `DiscoveryReset` fault | `fan_in.rs:405` |

### TR-8 Pool relay routing-pool registration and Global Discovery lease

Location: `REL/routing_pool.rs:23` (`RoutingPoolRegistration`), `:78` (`follow_topology`), `:133` (`select_model`); heartbeat `lib/global-discovery-client/src/heartbeat.rs:50` (`PoolHeartbeat`), `:120` (`keep_registered`), status `:31` (`RegistrationStatus`). States: registration watch `None(withheld reason) | Some(PoolRegistration{model})`; heartbeat `Unregistered | Registered(Lease) | Failing(reason)` with internal `renewed: Option<(Lease, started_at)>`, `retry` backoff, `lapse_logged`. Invariants: the registered model is sticky while it stays in the topology in any state (`routing_pool.rs:146-148`); exactly one call per heartbeat interval counted from call start (`heartbeat.rs:189`); a failed call retries at 1, 2, 4.. s capped by the interval (5 s before the first lease) with one extra retry 1 s before lease end (`:162-182`); stopping sends nothing, the service drops the pool when the lease passes (`:41-44`); a change of the registration value is picked up at the next call, not immediately, except `None -> Some` which is sent at once (`:131-140`; NOTE transports-10).

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| None | topology gains exactly one model (Ready preferred) | - | Some(model) | heartbeat sends at once | `routing_pool.rs:97-114`, `heartbeat.rs:135-140` |
| None | topology has more than one model and none registered | - | None(reason) | warn once per set | `:93-96,155-158` |
| Some(m) | topology still contains m (any readiness) | - | Some(m) | none | `:146-148` |
| Some(m) | m leaves the topology | - | Some(other) or None | sent at the next call | `:149-159` |
| Some | `follow_topology` ends (cancel, topology sender dropped) | - | None | withheld "the Relay stopped registering the pool"; no DELETE | `:123-125` |
| Unregistered | call ok | - | Registered(lease) | next call after `heartbeat_interval` | `heartbeat.rs:142-156` |
| Registered | call fails | - | Failing | backoff; lapse warning once `at.elapsed() >= lease` | `:157-184` |
| Failing | call ok | - | Registered | backoff reset | `:143-146` |
| any | cancel / drop | - | last status kept | task aborted | `:85-93` |

### TR-9 Pool relay WAN publication: hub, subscription, stream, gRPC

Location: hub `REL/publication/hub.rs:272` (`PublicationHub`), `:230` (`HubState`), `:183` (`HubSubscriber`), `:463` (`PublicationHubSubscription`), loop `:550` (`run_hub_loop`), `:762` (`recover_snapshot`), `:904` (`fan_out`); stream `REL/publication/stream.rs:45` (`open`), `:238` (`SnapshotBootstrap`); source `REL/publication/source.rs:267` (`subscribe_pool`), `:189` (permits), `:231` (`terminal_failure` fences `expected`'s generation only); gRPC `REL/wan/grpc/service.rs:207` (`subscribe_kv_pool`), `:375` (`pool_update_stream`), `:325` (`window_stream`), `:172` / `:247` (catalog, readiness); identity gate `wan/grpc/source.rs:53`.

States: hub `ready` (`is_ready && !is_idle_eviction_claimed`), `idle-eviction-claimed` (`is_ready = false, claimed = true`, no subscribers, bucket `Arc` unshared), `retired/stopped` (`is_ready = false`, `last_error`, `stopped` token); subscriber (per stream) `ACTIVE | LAGGED | RETIRED` in `close_reason: AtomicU8` plus membership in `state.subscribers` (removed on LAGGED, RETIRED, drop, or closed channel); stream `bootstrapping` (snapshot chunks from the blocking encoder via a 1-slot channel) -> `delta` -> `ended(Err)`; gRPC adds `heartbeat armed` once `snapshot_chunks_remaining == Some(0)`; admission: transport permit per stream kind (`service.rs:88`), publication permits `active_stream` (fail-fast) and `snapshot_encoding` (waits, cancelled by lifecycle or generation retirement) (`source.rs:189-229`).

Invariants: snapshot clone and queue registration happen under one `state` lock, and every `fan_out` happens under the same lock after the snapshot is advanced, so a subscriber's first delta always has `base_sequence == snapshot.sequence` (`hub.rs:343-382,617-760,873-902`); deltas are validated contiguous and identity-stable on the stream (`stream.rs:168-196`); a hub-internal lag or oversized delta replaces the snapshot from the actor and retires every subscriber (`hub.rs:762-782`); publication invariant failures fence the generation via `terminal_failure`, encoder cancellation and client-local progress timeout do not (`stream.rs:206-228`); the subscriber delta queue is bounded by `queue_capacity` 16 frames and 16 MiB and is fed from the moment of `subscribe()`, including during snapshot bootstrap (`hub.rs:22-23,343,904-917`; finding SM-21); the actor cuts a delta at 16 pending events or 1 ms after the first (`host.rs:91-92`, `actor.rs:1148-1152`); heartbeats start only after the last snapshot chunk and carry `base == sequence == current` (`service.rs:405-427,546-562`); window streams drop non-increasing `window_sequence` and end with `SUBSCRIBER_LAGGED` on broadcast lag (`service.rs:346-357`); identity in `subscribe_kv_pool` is checked twice (catalog key `wan/grpc/source.rs:58-73`, active generation `source.rs:276`), so a stale `expected` can only fence its own already-retired generation.

| From | Event | Guard | To | Side effects | Where |
|---|---|---|---|---|---|
| none | `SubscribeKvPool` | marker ok, id ok, pool in catalog, key matches | admitted | transport permit, active-stream permit (fail-fast `RESOURCE_EXHAUSTED`), encoding permit (waits) | `service.rs:207-245`, `source.rs:189-229` |
| admitted | hub ready | - | ACTIVE + bootstrapping | snapshot cloned, queue created | `hub.rs:343-382`, `stream.rs:54-75` |
| admitted | generation retired while waiting | - | Err UNAVAILABLE | permits released | `source.rs:217-221` |
| bootstrapping | client does not drain for `progress_timeout` (60 s) | - | Err `SNAPSHOT_PROGRESS_TIMEOUT` | encoder exits, encoding permit released, generation not fenced | `stream.rs:257-285,222-227` |
| bootstrapping or delta | more than 16 frames or 16 MiB queued | - | LAGGED, removed | next `ensure_active` / `recv` gives `SUBSCRIBER_LAGGED` | `hub.rs:904-917,504-512` |
| bootstrapping | snapshot frame identity, base, or sequence invalid | - | Err `INVALID_PUBLICATION` | generation fenced | `stream.rs:94-98,142-166` |
| delta | frame contiguous | - | delta | `current_sequence = frame.sequence` | `stream.rs:120-126` |
| delta | frame non-contiguous or identity drift | - | Err `INVALID_PUBLICATION` | generation fenced | `stream.rs:168-196` |
| ACTIVE | hub `retire()`, hub task stops, or `recover_snapshot` | - | RETIRED | `UNAVAILABLE` "requires a fresh snapshot" | `hub.rs:384-390,523-548,762-782` |
| ready | no subscribers, bucket `Arc` unshared, `try_begin_idle_eviction` | - | claimed | `evict_idle` retires the actor lease then joins the task | `hub.rs:392-439` |
| claimed | `subscribe()` | - | Err `Unavailable("stopped")` | the host restarts a hub for the same generation on the next subscribe | `hub.rs:344-351` |
| hub loop | actor broadcast `Lagged` or oversized delta | - | snapshot replaced | subscribers RETIRED | `hub.rs:742-752,762-782` |
| hub loop | `state.snapshot.sequence != base` after off-thread apply | - | hub stopped, generation fenced | `terminal_failure` | `hub.rs:682-692,730-740` |
| delta (gRPC) | heartbeat tick | `snapshot_chunks_remaining == Some(0)` | delta | `HEARTBEAT` frame | `service.rs:423-427` |
| window stream | broadcast `Lagged(n)` | - | Err `SUBSCRIBER_LAGGED` | - | `service.rs:352-357` |
| any stream | `cancel` (transport) | - | ended (clean) | permits dropped | `service.rs:191,399` |

### TR-10 gRPC transport lifecycle

Location: `REL/wan/grpc/server.rs:178` (`GrpcTransport::start`), health `:38`, supervisor `:303`. States: bound, not serving -> serving (`health.serving = true`, tonic health SERVING) -> not serving on cancel (`:274-280`). Every accepted connection is wrapped in `CancellableIo`, which fails all I/O once the child token is cancelled (`:84-93`), so shutdown does not wait for clients. The load publishers (`:286-300`) run under the same token; their failure is routed through `supervise_transport` to `fatal_cancel` (the host lifecycle), so a load-publisher failure stops the whole relay publication (`:301-309`).

Transports interleavings traced and sound:

- Forward-gap replay on the direct-ingress listener with a cooperative ROUTER (`replay_accepts_engine_framing`, `listener.rs:553`) and on the publisher recovery path (`live_gap_after_success_is_repaired_without_reapplying_history`, `zmq_recovery.rs:726`).
- Pool relay bounds tested one at a time: `subscriber_queue_enforces_byte_limit_before_message_capacity` (`hub.rs:1067`), `stalled_snapshot_bootstrap_requires_a_fresh_stream` (`stream.rs:447`); `wan/grpc/conformance.rs` was not read for a bootstrap-under-load case.
- Heartbeat lease semantics (immediate `None -> Some`, deferred `Some -> Some`, no withdrawal on stop) are pinned by `heartbeat.rs:305-359`.
- Metrics fault ordering: `send_fault` under all shard locks (`metrics_subscriber.rs:217-241`); loads are absolute snapshots, so the next report corrects a stale value.

## Findings

Labels: CONFIRMED = a complete reachable sequence was traced with the exact wrong outcome; SUSPECTED = a gap was seen but the sequence or outcome depends on something not verified. Impact classes follow the analysts' vocabulary. Rows are ordered CONFIRMED by impact, then SUSPECTED. `Status` is `open` until a fix lands, `fixed <sha> (<regression tests>)` once it has, or `open (design decision)` when the fix needs an owner's call; a fix PR updates this column and the machine tables it changes.

| Id | Area finding(s) | Label | Impact | Statement | Machines | Status |
|---|---|---|---|---|---|---|
| SM-1 | transports-1 | CONFIRMED | deadlock | Direct-ingress `replay_gap` awaits `recv_multipart` with no deadline (`services/indexer/listener.rs:238-249`; `ZMQ_RCVTIMEO` is inert on the async path, `services/common/zmq.rs:163-190,303-311`), so a silent replay peer freezes the rank's listener forever while status stays `Active`. | KI-6, KI-7 | fixed 6f1975d717 (`replay_gives_up_on_a_silent_peer`) |
| SM-2 | kv-indexer-2 | CONFIRMED | stale-routing | `KvIndexer::remove_worker` uses a side channel polled ahead of `mutation_rx` (`kv_indexer.rs:475-503`), so events admitted before `deregister(W)` re-create W after the removal; only `ResetWorkerDpRank` is a FIFO barrier (`:137-148,241-251`). | KI-5, KI-8 | fixed aed84c4ef9 (`remove_worker_does_not_overtake_events_admitted_before_it`, `remove_worker_dp_rank_does_not_overtake_events_admitted_before_it`) |
| SM-3 | kv-indexer-3, transports-2 | CONFIRMED | stale-routing | `WorkerRegistry::deregister*` cancels but never joins or drains the listener (`services/indexer/registry.rs:507-523,557-578`) and `handle_message` has no cancel check (`listener.rs:363-383,412-419`), so an in-flight batch is indexed after the removal barrier (`thread_pool.rs:870-907`) or ahead of the side-channel removal, and the removed `(worker, rank)` reappears. | KI-4, KI-5, KI-7, KI-8 | fixed a9db118e8d (`deregister_waits_for_the_listener_before_removing_the_worker`) |
| SM-4 | kv-indexer-4 (transports-3, direct-ingress half) | CONFIRMED | silent-drop, stale-routing | An engine sequence reset under an unchanged registration (static `--workers`, pause/resume sharing the watermark, unchanged catalog `reconcile`) is `Stale` forever (`lib/kv-events/src/cursor.rs:30-43`; `listener.rs:340-343,414-416`): every event including `AllBlocksCleared` is dropped without a log and the dead engine's blocks stay indexed. | KI-6 | fixed 601b44e3ee (`sequence_regression_starts_a_new_epoch`, `live_frames_the_replay_covered_are_dropped_not_reset`) |
| SM-5 | slot-tracker-1 (slot-tracker-12 folded) | CONFIRMED | leak, wrong-accounting | An unfenced `free` racing an in-flight `add_request` for the same id deletes the Index-only booking and publishes `Free` (`multi_worker.rs:944-964`); the add then inserts an orphan `RequestState` no fenced release can reach (`:1040-1042`); production path is `RouterRequest::New` cancelled during booking (`kv_router.rs:1944-1961,1642-1647`); permanent in the embedded host. | ST-1, KQ-1, ST-4 | fixed e28887935a (`unfenced_free_racing_in_flight_add_does_not_strand_sequence_state`) |
| SM-6 | prefill-disagg-1 | CONFIRMED | leak, stuck-request | A client disconnect during the prefill queue hold is unobserved because `arm_for` runs only inside `prepare`, after `plan_kv_route` returns (`prefill_router/mod.rs:619-636`; `routing_host/kv.rs:652-660`): the dead request keeps its class-queue slot, then books and runs prefill (and a staged-KV decode dispatch when cancel-anytime is not advertised by both sets). | PD-2, PD-4, KQ-1 | fixed 49c3ddcbe0 (`client_disconnect_during_prefill_queue_hold_retracts_the_request`) |
| SM-7 | session-affinity-1 | CONFIRMED | wrong-accounting | `reserve_entry` counts Bound-Expired entries against `max_entries` (`session-affinity/src/lib.rs:733-739`) until the lazy reaper runs (`:373-393,353-360`), so new sessions are refused (frontend `coordinator.rs:307-317`) or routed unbound (core `run.rs:605-610`) for up to `min(ttl, 30s)` after capacity freed. | SA-1, SA-11 | fixed d0650f8612 (`expired_binding_does_not_count_against_the_entry_limit`, `expired_binding_does_not_reject_a_replica_binding_for_capacity`) |
| SM-8 | session-affinity-2 | CONFIRMED | contract | Core Hard mode: `hold_session` checks only `is_schedulable` (`run.rs:309-317`), the actor drops an overloaded or full affinity target (`queue.rs:150-157`; `filter.rs:357-372`) and books elsewhere, and `commit_session` returns `BadRequest` and erases the binding (`run.rs:632-637`; `lib.rs:592-596`); the frontend returns `PinnedWorkerOverloaded` and keeps it (`routing_host.rs:683-694`). | SA-7, SA-8, SA-1 | fixed e79abbab35 (`hard_mode_keeps_the_binding_when_the_host_sheds_the_bound_worker`) |
| SM-9 | membership-2 | CONFIRMED | stale-routing | With an uninitialized admitted view the availability provider returns `None` (`client.rs:883-892,195-198`), read as "no hard filter" (`scheduler/src/types.rs:64-75`; `filter.rs:116-120`), while `workers_with_configs` is admission-agnostic (`runtime_configs.rs:136-177`): a rejected-cohort worker is booked and the dispatch fails `CannotConnect` (`push_router.rs:1796-1817`). | MB-2, MB-3, MB-4, KQ-7 | fixed 5b1a570dd1 (`admitted_view_without_discovered_members_yields_no_candidates`) |
| SM-10 | membership-1 (kernel-queue-1, same race labelled SUSPECTED there) | CONFIRMED | wrong-accounting | Nothing is held between selection (`queue.rs:2398-2484`) and `add_request_admitted` (`:2661-2664`), so a `reconcile_workers` in that window is undone by lazy registration (`multi_worker.rs:1541-1551,1508-1521`); the strict `add_request_if_registered_admitted` exists for this race (`:814-833`) but only hosts call it; the phantom slot and booking persist until the next reconcile. | KQ-1, ST-3, MB-6 | fixed 3aec16fdbf (`booking_does_not_recreate_a_worker_reconciled_away_after_selection`) |
| SM-11 | kv-indexer-1 (kv-indexer-12 folded) | CONFIRMED | contract | `race_cleanup_with_dead_child_reuse_keeps_restored_child` (`concurrent_radix_tree_compressed/tests.rs:636-697`) demands one-sweep completeness while a store holds the parent's gate, but the sweep gives up after 4 `try_write` attempts (`remove.rs:78-86`), yielding `[2, 2, 3]`; the accepted `[5]` is unreachable because `internal` is sticky (`node.rs:41-43,487-489`). | KI-1, KI-3 | fixed d22efeaed2 (test contract; `race_cleanup_with_dead_child_reuse_keeps_restored_child` rewritten) |
| SM-12 | session-affinity-3 | CONFIRMED (design decision) | contract | The initializer that loses a joined failover gets `BadRequest` when its own booking lands on a third worker (`lib.rs:858-869`; `run.rs:632-637`) although it carried no constraint; `joined_failover_preserves_the_successful_bookings_binding(initializer_selects_another_worker)` pins this as intended, so a fix reverses that assertion. | SA-2, SA-7 | open (design decision) |
| SM-13 | admission-1, kernel-queue-2 | SUSPECTED | silent-drop | `drain_cleanup` retracts queued entries by `tracked_request_id` rather than by the transfer `Arc` it holds (`queue.rs:1934-1937,2004-2010`), so a same-id entry enqueued while a `Pending` cleanup is undrained (channel `Full`, `:579-585`) is removed unanswered; whether an id can be reused in that window is unverified (every in-tree retry releases synchronously first). | KQ-2, KQ-4, AD-1 | open |
| SM-14 | slot-tracker-2 | CONFIRMED | wrong-accounting | Two-phase topology removal (`multi_worker.rs:745-748` swap, `:777-783` deferred cleanup) lets a `LazyRegister` replica `AddRequest` recreate the rank in the gap (`replica_sync.rs:199-201`; `topology.rs:313-330`) and then lose its index and loads entries. Captured by the slot agent: after the gap `num_workers = 1`, `ranges_empty = true`, `loads_entry = false`, `booking = None`; after `reconcile_workers([range])` the slot is reused with no loads entry; after one unrelated add plus free the orphan resurfaces as `blocks = {(1,0): 3}`, `counts = {(1,0): 1}`, unreleasable. SM-5's fix does not change this outcome. | ST-3, ST-4, ST-1 | open (design decision) |
| SM-15 | session-affinity-4 | SUSPECTED | stuck-request | Waiters on another request's initialization (`lib.rs:454-461`) are outside every deadline and queue budget (`run.rs:578`; `coordinator.rs:126-143`; `routing_host.rs:604-612`): a waiter's class deadline never fires while the initializer sits in the scheduler queue. | SA-4, SA-1 | open |
| SM-16 | kv-indexer-5 | SUSPECTED | silent-drop | `deregister(W)` releases the lifecycle lock before `remove_worker` (`services/indexer/registry.rs:507,436-483`), so a concurrent `register(W)` listener's early stores are erased by the old phase 1 and 2 (`thread_pool.rs:870-907`) and later chained stores are rejected `ParentBlockNotFound` while the watermark advances; no concurrent caller was confirmed. | KI-8, KI-4 | open |
| SM-17 | prefill-disagg-2 | SUSPECTED | contract | An encode result produced after a client disconnect is never handed downstream: `next.generate` runs on the killed context (`encoder_router.rs:411-428`) under `CancelWhenStopped` because `staged_kv_cleanup` is only set by `into_decode_request` (`routing_host/cancellation.rs:85-93`), contradicting the comment at `encoder_router.rs:413-415`; engine-side buffer reclamation not traced. | PD-7, KQ-10 | open |
| SM-18 | prefill-disagg-3 | SUSPECTED | contract | Two Prefill WorkerSets in one namespace clear the target (`discovery/model_manager.rs:2672-2691`), turning the typed-Decode set's `PrefillRouter` into a passthrough (`prefill_router/mod.rs:473-478`) that sends aggregated requests to decode-only engines (5xx, not 503); possibly operator error. | PD-1, MB-1 | open |
| SM-19 | membership-3 | SUSPECTED | wrong-accounting | Rejected-cohort workers reach every kernel-side structure (slots `local.rs:152-176`, catalog, KV listeners `recovery/worker_query.rs:247-249`, approximate-LRU registry `kv_router.rs:242-322`, classifier capacity `kv_router.rs:971-986`) because the runtime-config join is admission-agnostic; classifier impact not traced. | MB-4, MB-6, MB-7 | covered by SM-9 fix 5b1a570dd1 |
| SM-20 | transports-3 (publisher-side half; the direct-ingress half is SM-4) | SUSPECTED | stale-routing | The publisher recovery path drops every batch below the cursor after an engine sequence restart and stays `Success` (`publisher/zmq_recovery.rs:65-73`); the non-recovery listener without liveness only warns (`zmq_listener.rs:196-204`); independent engine restart is documented unsupported (`publisher/mod.rs:271-274`) and detection by the state agent is unverified. | TR-4, TR-3 | open |
| SM-21 | transports-4 | SUSPECTED | contract | The delta queue is registered at the snapshot cut (`publication/hub.rs:343-382`) and lags at 16 frames or 16 MiB (`:22-23,904-917`) before the stream ever polls `recv()` (`stream.rs:77-106`); under continuous traffic 16 cuts take about 16 ms (`host.rs:91-92`, `actor.rs:1148-1152`), so a slow bootstrap always ends `SUBSCRIBER_LAGGED`; real-deployment exposure not measured. | TR-9 | open |

Open design decisions: SM-14 has three candidate fixes, (1) run `apply_worker_topology_change` while still holding `workers.write()` (lock order workers then registry matches the add path; cost is the trie walk and index scan under the table write lock), (2) `LazyRegister` refuses ranks outside `worker_ranges` once any authoritative range exists, (3) scope the deferred cleanup to the dropped slot's own request ids with a slot generation guard; SM-12 is pinned as intended by `joined_failover_preserves_the_successful_bookings_binding`, so a fix reverses that test's assertion.

SM-19 is covered by the SM-9 fix (5b1a570dd1): the frontend's `workers_with_configs` is now the admitted intersection (MB-4), so a rejected-cohort worker never enters the slot tracker, the catalog, the KV listeners, or the classifier capacity; its label stays SUSPECTED because the classifier impact was never traced.

SUSPECTED items the analysts could not complete, with the missing piece: SM-13 (whether a scheduler id can be reused while a `Pending` cleanup is outstanding; router-generated ids make it unreachable); SM-15 (the "wrong" outcome is a missed deadline, serialization is documented); SM-16 (needs a test hook to park the deregister inside the primary barrier); SM-17 (engine-side reclamation of the staged encode buffer was not traced in `components/`); SM-18 (backend error text not traced; topology may be operator error); SM-19 (classifier behaviour on inflated capacity not traced); SM-20 (whether the state agent or catalog detects every engine restart was not verified; `Indexer::Concurrent` was not traced for transports-2 either, only `Indexer::Single`); SM-21 (`wan/grpc/conformance.rs` not read for a bootstrap-under-load case).

NOTE items flagged as worth acting on:

- kernel-queue-5: no rejection path moves a transfer `Pending -> Disarmed` (class expiry `queue.rs:1899-1910`, post-extraction refusals `:2548-2563,2646-2657,2664-2669`, `RefreshEnd::DueTimePassed`), so every refused or expired lifecycle request's lease drop schedules a `drain_cleanup` scan over every class (`:2001-2020`; `admission/queue.rs:964-979` full `entries()` scan per class) that finds nothing; cost scales with rejections x classes x queue length, which is the regime of a deadline storm. Fix: `transfer.disarm()` whenever the actor answers the request; pinning test: after `DeadlineExceeded`, `drop(lease)` must leave `cleanup.pending` false.
- slot-tracker-6: settled by SM-5's fix in the "leave the in-flight add alone" direction; the local `free_on_worker` now matches the replica `Free` handler (`replica_sync.rs:263-265`) and leaves an index-only booking untouched.
- slot-tracker-7: absolute-age `force_expiry` in the standalone services drops live requests older than 300 s from accounting and removes expired ids from the index unconditionally (`single.rs:34-43,344-379`; `multi_worker.rs:1575-1576,1462-1463`; `replica_sync.rs:238-239`); after SM-5 the unconditional removal can strike a different booking.
- slot-tracker-8: `free_if_worker` is worker-fenced, not attempt-fenced (`multi_worker.rs:967-978`), has no production caller (`local.rs:742-754` wrapper is uncalled), and is a candidate for removal.
- slot-tracker-9: three silent-drop points in replica sync (outbound queue full or closed `multi_worker.rs:600-644`; `ReplicaInbox::deliver` block-size mismatch at `debug` `replication.rs:464-476`; no catch-up for a restarted peer `replica_sync.rs:264-341`, `sequence.rs:423-487`); raise the block-size mismatch log to `warn`.
- slot-tracker-10: `LazyRegister` slots bypass `validate()` and `MAX_DATA_PARALLEL_RANKS_PER_WORKER` (`topology.rs:313-330`; `replica_sync.rs:199-201`; `multi_worker.rs:1543-1551`), so a peer can create unbounded slots until the next reconcile, and a lazy rank outside a configured range skews `FullRankCounts::worker_is_full` (`busy.rs:149-158,166-170`).
- session-affinity-8: the subagent prefill key is the raw trimmed parent id while decode uses the hashed group id (`routing_host.rs:478-489`): the prefill key equals the parent's own key only when the affinity header resolves to the agent session header (`protocols/agents.rs:159-175`); a parent id over `MAX_SESSION_AFFINITY_ID_BYTES` fails the prefill leg with `InvalidArgument` (`lib.rs:648-656`) while decode works; `" p"` and `"p"` share a prefill key but not a decode group.
- Membership negative results (recorded so nobody re-derives them): membership-9, the prepared-rows generation guard is sound and conservative (KQ-7); membership-10, the indexer-service HTTP `deregister*` versus `register` interleaving without a shared lock could not be completed into a block-losing sequence (the queue-full variant is SM-16); membership-7, no violation of the documented first-accepted-configuration rule was found.
- Remaining NOTE ids, recorded in the machine tables above and needing no action now: kernel-queue-3, -4, -6, -7, -8, -9, -10; admission-2 (quantum 0 reaches a division in `extract_next` only through a non-YAML `PolicyProfile::from_classes`), -3, -4, -5, -6, -7, -8, -9; slot-tracker-3, -4, -5, -11, -13; session-affinity-5, -6, -7, -9, -10, -11, -12, -13; kv-indexer-6, -7, -8, -9, -10, -11; prefill-disagg-4, -5, -6, -7, -8; membership-4, -5, -6, -8; transports-5, -6, -7, -8, -9, -10, -11, -12.
