# lib/router-host/src

This crate (`dynamo-router-host`) is the router's host layer around the `lib/router/kv-router`
selection core. Read [`README.md`](README.md) before changing selection inputs, booking
ownership, cancellation, prefill activation, or the encode and prefill hops; it explains the
layer map, per-stage roles, and the booking lifecycle these rules protect. What the crate may
depend on is in [`../AGENTS.md`](../AGENTS.md).

## Module map

- `lib.rs`: `KvRouter`. Estimates how much of a request each worker already has cached,
  then asks the scheduler to choose a worker. It selects and books, but never sends a request.
- `embedded.rs`: connects discovered workers and their load to one `SelectionService` partition
  from `lib/router/kv-router`, and runs each selection through `SelectionCore`.
- `metrics.rs`: Prometheus metrics of the request path.
- `managed.rs`: `ManagedKvRouter`, a `KvRouter` together with the load context that owns its
  load tasks.
- `request_lease.rs`: one request-liveness coordinator and reaper per `KvRouter`. Holds booking
  leases and releases them on completion or expiry.
- `routing_host.rs`: `RoutingHost`, the pipeline's terminal engine. Owns selection, dispatch,
  and response cleanup for one worker set.
  - `routing_host/kv.rs`: the KV path: preview, plan, select, track, dispatch.
  - `routing_host/kv_selection.rs`: turns a request's routing inputs into a `KvRouter` call.
  - `routing_host/hosted.rs`, `routing_host/occupancy.rs`: non-KV modes (random, round-robin,
    load-based) and Direct mode.
  - `routing_host/request_guard.rs`: tracks a dispatched request's progress and releases its
    booking.
  - `routing_host/cancellation.rs`: what a client disconnect cancels, and the cleanup budget.
- `prefill_router/`: `PrefillRouter`, the optional prefill hop in front of decode.
  - `activation.rs`: target and binding lifecycle; how the prefill hop's routing mode is chosen.
  - `admission.rs`: consumes the prefill stream, in the foreground or as a background task.
  - `handoff.rs`: keeps prefill running while decode runs and reports prefill failures.
  - `conditional_bypass.rs`: conditional disaggregation's decision to skip remote prefill.
  - `query.rs`: `reserve_prefill_worker`, the prefill-only booking used by the Runtime EPP.
  - `endpoints.rs`: `PrefillEndpointProvider`, the only way the prefill hop reaches the model
    manager: per-worker runtime-config lookups and the `KvRouter` of the prefill worker set.
- `encoder_router.rs`: `EncoderRouter`, the optional encode hop for multimodal inputs.
- `worker_set_target.rs`: the worker set an encode or prefill hop targets, and the id its
  binding is keyed on. Discovery chooses the target.
- `plugins.rs`: installs the linked router plugin registry.
- `session_affinity/`: `AffinityCoordinator`, which pins or prefers a session's worker, and its
  replica sync between routers.
- `migration.rs`, `migration/`: `Migration`, the pipeline operator that retries a failed
  request on another worker, and `migration/realtime/`, the replay journal of realtime sessions
  (called by the HTTP realtime service in `dynamo-llm`).
- `lora_routing.rs`, `routing_policy.rs`: the hosted candidate policy and the KV admission
  policy hooks, and their LoRA implementations.

Two neighbours live elsewhere: the worker-side KV event publisher is
`dynamo_backend_common::kv_publisher` (`lib/backend-common/src/kv_publisher/`), and
`lib/llm/src/kv_router/shared_cache.rs` in `dynamo-llm` is the HiCache shared KV cache client
(SGLang + Mooncake), which the model manager builds and hands `KvRouter` as a `SharedKvCache`.

The state these stages read is kept up to date by the crate `dynamo-sync`
([`lib/sync`](../../sync/AGENTS.md)), not by this crate:

- `indexer/`: tracks which KV-cache blocks each worker holds; lookups run in
  `lib/router/kv-router`.
- `sequence.rs`, `sequence/`: runtime glue for active-sequence tracking and its replica sync.
- `load/`: `ActiveLoad`, the load message workers publish on the event plane (written by
  the publisher, read by `metrics_subscriber.rs`, `worker_monitor.rs`, and `pool_relay/`); the
  subscription to it; the worker monitor; and `routing_load.rs`, the load context shared by a
  worker set's scheduler and its load metrics.
- `membership/`, `runtime_configs.rs`: KV event-source membership and the watch of per-worker
  runtime configs.
- `metrics.rs`: Prometheus metrics of the above.

`lib/llm/src/scheduler_load.rs` publishes the router's scheduler load for the Pool Relay.

## Guardrails

- `KvRouter` never dispatches. `RoutingHost` dispatches decode, aggregated, and prefill requests
  (the prefill hop through the prefill set's own `RoutingHost`); `EncoderRouter` dispatches
  encode requests through a round-robin `PushRouter`.
- Each booking has exactly one owner at a time (`RoutePlan`, then `RequestGuard`, or a detached
  lease) and is released exactly once.
- A preview never books capacity and never creates an affinity binding.
- Validate request-supplied worker pins only at initial selection. A pinned worker that
  disappears after a preview or at dispatch is a service failure, not a client error.
- Decode and aggregated routing own the classifier lifecycle. The prefill hop must not start
  one.
- Never drop a decode leg that carries staged KV before it reaches its worker, except when its
  cleanup budget runs out.
- Fail closed while the prefill target is `Pending`. Pass through only when `Untargeted`.
- Do not let clients request a conditional bypass.
- When you add a selection input, add it to the inputs table in `README.md` and decide whether
  the Runtime EPP (`deploy/inference-gateway/ext-proc/src/epp.rs`) must supply it too.
- The Runtime EPP builds its own decode scoring override (`decode_router_config_override` in
  `epp.rs`). Change it together with `build_decode_router_override` in
  `prefill_router/mod.rs`, or the two hosts score decode differently.
- Keep `README.md` in step with changes to the layer map, per-stage roles, booking lifecycle,
  cancellation, or prefill activation.
- `dynamo-sync` is below this crate and cannot import it. Pass what it needs in, as `KvRouter`
  does with the router-mode selectors of the remote indexer.
- This crate does not depend on `dynamo-llm`: not on `discovery`, `entrypoint`, `http`,
  `pool_relay`, `scheduler_load` or the publisher. Discovery builds the routers and passes in
  what they need from above: the model manager as `PrefillEndpointProvider`, and the Pool
  Relay's scheduler-load publisher as `SchedulerLoadPublisher`.

## Testing

- Unit tests live beside the code: `routing_host/tests.rs` and the tests in `prefill_router/`.
  Run `cargo test -p dynamo-router-host` (and `cargo test -p dynamo-sync` for the state sync).
- Some ZMQ, event-plane, and replica-sync tests bind local ports and can fail inside a sandbox
  or in a full parallel run. Rerun a failing one alone before treating it as a regression.
- End-to-end router tests are described in the
  [router testing guide](../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/router/router-testing.md).
