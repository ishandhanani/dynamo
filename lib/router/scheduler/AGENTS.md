# lib/router/scheduler

The router scheduler: the selection kernel (`local`, `queue`, `overlap`, `overlap_refresh`), the
plugin registry (`plugins`), `SelectionCore` / `SelectionService` / `SelectionServiceBuilder`
(`selection`), the selection HTTP routes behind `http`, and the nested router configuration
(`config`). Read `src/queue/AGENTS.md` before touching the queue actor and
`src/selection/README.md` before touching the service invariants.

## Guardrails

- Depend on the router crates (`dynamo-router-protocol`, `dynamo-kv-indexer`, `dynamo-slot-tracker`,
  `dynamo-admission`, `dynamo-worker-selection`) only. Never introduce `dynamo-runtime` or
  `lib/llm` dependencies, gated or ungated: the frontend, ext-proc, and the bindings are hosts
  on top of this crate.
- The standalone service (`http`) must stay usable by processes that run no other Dynamo
  component: brokerless transports (TCP, HTTP, ZMQ) only.
- The kernel owns one serializer per partition (the queue actor). Every booking goes through it
  except the documented reservation replay path; do not add a second writer.
- `config::RouterConfig::from_kv_router_config` is the only flat-to-nested conversion. Stage
  crates receive their slice; do not teach a stage to read `KvRouterConfig`.
- Keep `axum` default features disabled and enable only what the service uses.
- Use `FxHashMap` / `FxHashSet` for internal numeric keys and hot paths; never for text keys or
  externally controlled values such as `request_id`.
