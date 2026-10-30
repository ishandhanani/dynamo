# The Dynamo router crates

The router is seven crates under `lib/router/`: a protocol crate, four stages in pipeline
order, the kernel that runs them, and the composition crate hosts build a router from. Each
stage depends only on the protocol crate and the stages upstream of it; none depends on
`dynamo-runtime`, Prometheus, or an HTTP stack except inside an embedded standalone host behind
`http`. `scripts/router_layering_check.sh` checks these properties from `cargo tree`.

| Crate | Owns |
|---|---|
| `dynamo-router-protocol` | the vocabulary every stage shares: wire protocols and KV events, identities, tracking hashes, the per-request `RouterConfigOverride` and the `TrackingHashConfig` trio, scheduling request and response types, eligibility, overlap signals, busy thresholds, the per-rank load row, load slots, the indexer's query DTOs |
| `dynamo-kv-indexer` | every index tree, the KV event wire format, cursor recovery, the session prefix index, cache ingest (`ingest-zmq`), its config slice, and the standalone indexer service (`http`) |
| `dynamo-slot-tracker` | the active-sequence write model and the prompt-registry read model, the replica mesh (`replica-sync-zmq`), its config slice, and the standalone slot-tracker service (`http`) |
| `dynamo-worker-selection` | routing eligibility helpers, the selection plugin API and its candidate rows and views, the composed selection policy, LoRA narrowing, conditional disaggregation, session affinity, the policies Dynamo ships, and its config slice |
| `dynamo-admission` | the per-class policy queue, the queue profiles (`policy_config`), the request-classifier plugin API and runtime, and its config slice |
| `dynamo-scheduler` | the kernel: `SelectionCore`, `LocalScheduler` and the queue actor, the worker `Topology`, per-request `Constraints`, overlap analysis and refresh, and `RouterConfig`, the nested stage configuration it runs with |
| `dynamo-kv-router` | the composition: `SelectionService` and its builder, the plugin registry at the published `plugins::*` paths, the standalone selection HTTP service (`http`), and the flat `KvRouterConfig` with its environment and policy-file parsing, converted once into the stage slices |

Hosts (`dynamo-llm`, the Python bindings, ext-proc) compose these through `dynamo-kv-router`.
Read each crate's `AGENTS.md` before changing it.
