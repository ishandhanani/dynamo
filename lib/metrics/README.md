# dynamo-metrics

One sentence: name, label, create, and register Prometheus metrics the Dynamo way, and collect
stage counters at scrape time.

- `names`: every metric and label name constant plus the sanitizers (`sanitize_prometheus_name`,
  `sanitize_prometheus_label`, `build_metric_name`). `dynamo-runtime` re-exports this module as
  `dynamo_runtime::metrics::prometheus_names`. Changing a constant means regenerating the Python
  mirror: `cargo run -p dynamo-codegen --bin gen-python-prometheus-names`.
- `MetricsHierarchy`, `Metrics<H>`, `create_metric`: a hierarchy level (runtime, namespace,
  component, endpoint) names a metric `dynamo_component_<name>` and injects `dynamo_namespace`,
  `dynamo_component`, `dynamo_endpoint`, and `worker_id` labels; the runtime implements the
  trait for its types.
- `MetricsRegistry`: the registry tree a scrape walks, with update, exposition-text, and typed
  callbacks. `prom_typed` builds typed families for the OTLP exporter in the runtime.
- `collect::SourceSet`: the live-source set behind a scrape-time collector. A stage keeps atomics
  and knows nothing about Prometheus; a host collector attaches the stage handles it exports and
  reads every live one into fresh families per scrape.

Who depends on it: `dynamo-runtime`, `dynamo-http-server`, the router composition crate, and the
hosts. Stage crates under `lib/router` do not.

```bash
cargo test -p dynamo-metrics --features testing
cargo bench -p dynamo-metrics --bench gather
```
