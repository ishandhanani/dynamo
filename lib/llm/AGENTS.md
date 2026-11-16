# lib/llm

`dynamo-llm` is the frontend. It owns the OpenAI, Anthropic and System One HTTP service (`http`), the KServe gRPC service (`grpc`), the entrypoint that prepares an engine and runs an input (`entrypoint`), and the frontend serving-load publisher (`frontend_load`). `utils::lora_name_to_id` backs the Python binding of the same name.

The KV block manager (`block_manager`, `tokens`, `cuda`, `common::dtype`, `src/kernels`, behind the `block-manager` and `cuda` features) also still lives here. It is being removed separately; do not build on it.

Everything else that used to be in this crate is a crate of its own, and `dynamo-llm` re-exports none of it: import the owner. Dependencies point down this list.

| Crate | Path | Owns |
| --- | --- | --- |
| `dynamo-llm` | `lib/llm` | the frontend, as above |
| `dynamo-discovery` | `lib/discovery` | the model catalog (`ModelManager`, `ModelWatcher`, `WorkerSet`), the engines it registers, and the routed-pipeline builder |
| `dynamo-router-host` | `lib/router-host` | the router's request path: `KvRouter`, the prefill and encode hops, migration, session affinity |
| `dynamo-sync` | `lib/sync` | what keeps a router's view of its workers up to date: KV index ingest and recovery, active-sequence replica sync, worker load, KV event-source membership |
| `dynamo-preprocessor` | `lib/preprocessor` | the preprocessor and the detokenizer (`backend`) |
| `dynamo-telemetry` | `lib/telemetry` | request trace, sinks and recorders |
| `dynamo-generate-format` | `lib/generate-format` | the contract an engine-native `/generate` format implements (`GenerateFormat`) and the registry the frontend selects one from |
| `dynamo-frontend-protocols` | `lib/frontend-protocols` | the OpenAI, Anthropic, tensor and internal request and response types |
| `dynamo-model-card` | `lib/model-card` | the model deployment card, the runtime config, the local model and the model types |

- The chain is frontend -> discovery -> router host -> sync -> frontend-protocols -> model-card. `dynamo-preprocessor` and `dynamo-telemetry` sit beside `dynamo-sync`, above the protocols and the model card: discovery and the frontend depend on the preprocessor, the preprocessor and the router host on telemetry. No crate in the table depends on one above it.
- `dynamo-generate-format` sits beside `dynamo-sync` too, on `dynamo-frontend-protocols` and `dynamo-http-server`. This crate serves the installed formats (`http/service/format_generate.rs`); the format crates (`lib/generate-formats/*`) and their catalog depend on the contract and not on this crate, so the SGLang and vLLM sidecars that link a format do not link the frontend.
- Worker-side code is `dynamo-backend-common` (`lib/backend-common`): the admission gate, the first-token notifier, and the KV event and forward-pass-metrics publishers. It depends on `dynamo-sync` and what is below it, never on this crate; this crate depends on it for the worker entrypoint.
- The engine that serves a `dynamo-mocker` scheduler on the runtime is `dynamo-mocker-engine` (`lib/mocker/engine`), on `dynamo-backend-common`. This crate does not depend on it; the Python bindings call it.
- The Pool Relay is `dynamo-pool-relay` (`lib/pool-relay`), on `dynamo-sync`. This crate does not depend on it.
- Under all of them: `dynamo-http-server` (the HTTP server machinery and the request metrics) and `dynamo-error` (the error-class to HTTP status policy, feature `http`).
- Cargo features `media-ffmpeg` and `mm-routing` forward to `dynamo-preprocessor`, and `request-trace-s3` and `request-trace-bench` to `dynamo-telemetry`; they gate no code here.
- Log targets of moved code name the new crate (`dynamo_discovery::...`, `dynamo_router_host::...`); the v1.7.0 section of the deprecations page has the table.
- Check: `cargo check -p dynamo-llm --all-targets`, `cargo test -p dynamo-llm --lib` or `--test <name>`, and `cargo tree -p <crate> | grep dynamo-llm` must print nothing for every crate below this one in the table and for `dynamo-backend-common`, `dynamo-pool-relay` and `dynamo-mocker-engine`. The sidecars, the mocker servers, the format crates and their catalog may name this crate only in dev-dependencies: `cargo tree -e normal -i dynamo-llm -p dynamo-vllm-sidecar` must find nothing.

# Protocol Field Handling

Before changing request or response fields, validation, preprocessing, backend translation, or
response projection, read the
[protocol field handling contract](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/protocol-field-handling.md).
Identify every semantic owner and preserve the field's documented end-to-end behavior.

# N-2 Worker / Frontend Compatibility

Assume N-2 mixed-version operation between workers and frontends during rolling
updates. Frontends and workers from the current release and the two immediately
previous releases may coexist in any combination and must interoperate across
the worker/frontend boundary. This includes a single frontend concurrently
discovering worker cards from multiple supported releases for the same logical
deployment or worker set. Differences caused only by supported wire evolution
must remain decodable through the supported boundary normalization and
compatibility shims. Within one WorkerSet, admission requires equal MDC
checksums after that normalization and tokenizer overrides. The first accepted
configuration retains the set while any of its workers remain; a mismatched
newcomer receives no traffic and must not remove healthy serving state.
Equivalent serving behavior or supported wire representations do not grant a
separate exemption from checksum equality. See
[`lib/discovery/src/AGENTS.md`](../discovery/src/AGENTS.md) for the admission and
succession contract. N-3 and older combinations are unsupported unless a
narrower temporary exception is explicitly documented.

Treat model deployment cards (`dynamo-model-card`), discovery metadata
(`dynamo-discovery`), and worker/frontend wire formats
(`dynamo-frontend-protocols`) as cross-process compatibility surfaces. Consider
both age directions: an older worker with a newer frontend and a newer worker
with an older frontend, anywhere within the supported window.

- Normal, default deployment paths should remain operable across the N-2
  compatibility window. Unconditional parsing or discovery failures on those
  paths are generally compatibility bugs.
- Prefer tolerant readers and conservative writers. Accept known legacy fields
  when they are safe to interpret, and continue emitting required legacy fields
  for the supported compatibility window.
- Silent degradation may be acceptable when it is safe and bounded. It must not
  erase an essential capability, violate an invariant, or produce misleading
  state that makes an otherwise valid deployment unusable.
- Conditional failures may be acceptable for explicitly enabled features whose
  semantics cannot be represented safely by the other version. Prefer a
  targeted unsupported-feature error over a generic deserialization failure.

This expectation applies to cross-process worker/frontend wiring, including the
frontend's aggregation of mixed-version worker metadata. It does not establish
a compatibility window for direct worker-to-worker protocols or a general
stability promise for in-process Rust APIs; those are internal unless explicitly
documented otherwise.

## Compatibility Shims

Keep version-specific compatibility code narrow, close to the wire boundary,
and temporary. Translate legacy input into canonical state immediately, and
derive legacy output from canonical state. Do not plumb legacy fields through
internal APIs, make core logic branch on them, or rely on their presence after
deserialization. When both representations are present, the current
representation is authoritative unless a specific compatibility rule says
otherwise.

Every shim must name its compatibility window and removal condition, for
example:

```rust
// Compatibility with v1.2 workers and frontends during v1.4 rolling upgrades.
// TODO(v1.5): Remove when v1.2 falls outside the N-2 compatibility window.
```

Remove the shim when the corresponding version leaves the supported window;
do not carry compatibility branches forward indefinitely.

# Frontend Serving-Load Metrics

The frontend's request metrics (`http/service/metrics.rs`) also feed the frame
it publishes for the Pool Relay's `SubscribeServingLoad` (`frontend_load.rs`),
so `/metrics` and the Relay report the same numbers. Three are specific to that
use and are experimental (see the Metrics Catalog):

- `dynamo_frontend_awaiting_first_token_requests{model}` (gauge): admitted
  requests with no output token yet. Requests producing output are
  `clamp_min(dynamo_frontend_active_requests - dynamo_frontend_awaiting_first_token_requests, 0)`;
  the frame derives that value (saturating) rather than exporting a second gauge.
- `dynamo_frontend_inflight_input_tokens{model, state}` (gauge, `state` is
  `awaiting_first_token` or `generating`): known prompt tokens of in-flight
  requests. Sum over `state` for the total.
- `dynamo_frontend_input_tokens_total{model}` (counter): prompt tokens of every
  admitted request, counted at tokenization. Use `rate()`.

Keep them consistent: one request transition updates every family through the
request's shared `RequestMetrics` handle. Create a model's series when it
registers, never from the publish path, which only reads. Before adding or
changing a frontend metric, name its reader, check it is not a query over
existing metrics or a sibling that should be a label, write its series count in
the pull request, and add it to the Metrics Catalog.

## Decision Cleanup Diagnostics

Report decision cleanup overruns through the request-correlated warning, not a separate metric until a continuous consumer is identified. Preserve the current tracing span when spawning cleanup, and retain admission until local work finishes. Use the warning to investigate overdue cleanup; it does not prove physical GPU settlement. This diagnostic adds no Prometheus series.
