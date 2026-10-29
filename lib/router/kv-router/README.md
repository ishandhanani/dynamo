# dynamo-kv-router

Composes a Dynamo router from the stage crates under `lib/router`.

One sentence: build and hold a router for a host. The kernel, `dynamo-scheduler`, runs the
stages atomically for one request; this crate is everything a host needs around it:

- `selection::SelectionService` and `SelectionServiceBuilder`: partitions, membership
  reconciliation, KV event ingress, replica sync, peer recovery, and the HTTP routes behind
  the `http` feature (the standalone selection service).
- `plugins`: the plugin API at its published paths (`plugins::worker_selection`,
  `plugins::request_classifier`), `RouterPluginRegistry`, and the resolved `RouterPlugins`.
- `config`: the nested router configuration assembled from the flat `KvRouterConfig`, plus
  the Dynamo environment reader.

Collaborators: `dynamo-scheduler` (the kernel), the four stage crates through it, and the
hosts above (`dynamo-llm`, the inference-gateway EPP, the Python bindings). Nothing in this
crate depends on `dynamo-runtime`.

## Features

- `http`: the standalone selection service routes and `run_server`
- `metrics`: forwards to the indexer's metrics until `dynamo-metrics` owns the collectors
- `bench`, `testing`: forwarded to the kernel; never enable `testing` in release

```bash
cargo test -p dynamo-kv-router --features http,testing
cargo clippy -p dynamo-kv-router --all-targets --features http,testing
```
