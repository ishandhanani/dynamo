# lib/telemetry

`dynamo-telemetry` owns the trace and metric records Dynamo writes to sinks, and the sinks: the request trace (`request_trace`: record types, `DYN_REQUEST_TRACE` policy, the in-process bus, the JSONL, S3, NATS and OTLP sinks, payload capture, agent-context and tool-event records, the ZMQ tool relay, replay hashing), the forward-pass metrics trace (`fpm_trace`), the stream-to-JSONL `recorder`, response recording and logprob analysis (`perf`), and at the crate root the pieces they share (`jsonl_gz`, `jsonl`, `stream`, `bus`, `parse_sink_names`). It builds and writes records; it serves no requests and routes nothing.

- Depends on (Dynamo crates): `dynamo-runtime`, `dynamo-frontend-protocols`, `dynamo-model-card`, `dynamo-worker-selection` (the router selection trace record), `dynamo-kv-hashing`, `dynamo-tokens`, and the published `dynamo-protocols`.
- Must not depend on: `dynamo-llm` or any crate above it. The preprocessor, the HTTP services and the router call into this crate; state they own is passed in as arguments, not imported.
- The OpenTelemetry span and logging setup is not here: it lives in `dynamo-runtime` (`logging`, `telemetry`). This crate only exports request-trace records over OTLP.
- Records are read by tools at other versions (`benchmarks/request_trace`, `lib/data-gen`, the planner's retune): a field change is a wire change. Read `src/request_trace/AGENTS.md` before changing request tracing, and keep sink I/O off the request path.
- Hosts import this crate directly; `dynamo-llm` does not re-export it. Explicit log targets keep the name `dynamo_llm::request_trace` so existing log filters still match.
- Features: `request-trace-s3` compiles the S3 sink and its `object_store` client; `request-trace-bench` exports the hooks the `request_trace_finish_metadata` bench in `lib/llm/benches` drives. `dynamo-llm` forwards both.
- Tests: `perf::logprobs` reads a replay under `lib/llm/tests/data`. `tests/request_trace_nats_integration.rs` is ignored by default and needs NATS and etcd.
- Check: `cargo test -p dynamo-telemetry`, `cargo check -p dynamo-telemetry --all-targets --features request-trace-s3`, and `cargo tree -p dynamo-telemetry | grep dynamo-llm` must print nothing.
