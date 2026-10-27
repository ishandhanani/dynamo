# lib/router/kv-indexer

The KV cache index of the Dynamo router: every index tree, the KV event wire format
(`zmq_wire`), cursor recovery, the session prefix index, cache ingest, and, behind the
`http` feature, the standalone indexer service. Read the more specific `AGENTS.md` and
`README.md` files in subdirectories when one exists.

## Guardrails

- Depend only on `dynamo-router-types` for router vocabulary. The optional
  `dynamo-runtime` dependency exists for the `runtime-protocols` feature alone and is
  slated for removal; do not widen it.
- Never introduce `lib/llm` / `dynamo-llm` dependencies, gated or ungated.
- The standalone service (`http`) must stay usable by processes that run no other Dynamo
  component: brokerless transports (TCP, HTTP, ZMQ) only.
- Keep `axum` and `reqwest` default features disabled and enable only what the service
  uses.
- Use `FxHashMap` / `FxHashSet` for internal numeric keys and hot paths; never for text keys
  or externally controlled values.
- Within one indexer/hash domain, engine-published local block hashes and external
  sequence hashes are deterministic and consistent across workers. Treat a violation as a
  producer, configuration, or protocol error; do not revalidate it on lookup hot paths.
