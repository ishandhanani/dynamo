# lib/router/slot-tracker

The router's active-sequence write model and derived prompt-registry read model, the
replica mesh that keeps peers' load views in sync (`replica-sync-zmq`), and, behind the
`http` feature, the standalone slot-tracker service. Read `src/AGENTS.md` and
`src/README.md` before changing ownership boundaries.

## Guardrails

- Depend only on `dynamo-router-protocol` for router vocabulary and on `dynamo-kv-indexer`
  only for the ZMQ socket helper behind `replica-sync-zmq`. Never introduce
  `dynamo-runtime` or `lib/llm` / `dynamo-llm` dependencies, gated or ungated.
- The standalone service (`http`) must stay usable by processes that run no other Dynamo
  component: brokerless transports (TCP, HTTP, ZMQ) only.
- Keep request lifecycle operations inline on `ActiveSequencesMultiWorker`; do not add a
  write actor, lifecycle queue, batching layer, or service-level admission policy here.
- Treat load reads as advisory snapshots from the existing derived read model.
