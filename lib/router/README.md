# The Dynamo router crates

The router is six crates under `lib/router/`, each a stage or a seam with a typed boundary, so
a stage can be swapped by anything that fits its input and output types:

| Crate | Owns |
|---|---|
| `dynamo-router-types` | the vocabulary every stage shares: wire protocols and KV events, identities, the flat `KvRouterConfig` wire shape and the policy document, scheduling request and response types, eligibility, overlap signals, busy thresholds, the indexer's query DTOs |
| `dynamo-kv-indexer` | every index tree, the KV event wire format, cursor recovery, the session prefix index, cache ingest (`ingest-zmq`), and the standalone indexer service (`http`) |
| `dynamo-slot-tracker` | the active-sequence write model and the prompt-registry read model, the replica mesh (`replica-sync-zmq`), and the standalone slot-tracker service (`http`) |
| `dynamo-admission` | the per-class policy queue, the request-classifier plugin API, and the classifier runtime |
| `dynamo-worker-selection` | routing eligibility helpers, the selection plugin API, the composed selection policy, LoRA narrowing, conditional disaggregation, session affinity, and the policies Dynamo ships (including the runtime's route pickers) |
| `dynamo-scheduler` | the kernel (`LocalScheduler` and the queue actor), the plugin registry, `SelectionCore` / `SelectionService` / `SelectionServiceBuilder`, the selection HTTP routes (`http`), and the nested `RouterConfig` |

Hosts (`dynamo-llm`, the Python bindings, ext-proc) compose these; `dynamo-kv-router`, the
crate that used to hold all of them, no longer exists. Read each crate's `AGENTS.md` before
changing it.
