# lib/generate-format

`dynamo-generate-format` is the contract between the frontend and the engine-native `/generate` formats. A format implements `GenerateFormat` (`open` a `ClientRequest` into an `Opened` request, whose `bind` builds the routed `PreprocessedRequest` and a `GenerateExchange` once the `Worker` set is known) and decides its own client and worker errors (`ClientError`, `WorkerError`, `ErrorPhase`, `json_body_error`). A host registers formats in a `Registry` and calls `install` once; the frontend reads the installed set with `named`, and with `selected` for the one `DYN_GENERATE_FORMAT` names (default `sglang`).

The formats are `dynamo-sglang-generate` and `dynamo-vllm-generate` (`lib/generate-formats/*`), the default set is `dynamo-generate-format-catalog` (`lib/generate-format-catalog`), and the routes that serve them are in `dynamo-llm` (`http/service/format_generate.rs`).

- Depends on (Dynamo crates): `dynamo-frontend-protocols` (`PreprocessedRequest`, `LLMEngineOutput`), `dynamo-http-server` (the request-metric error types and classifiers in `metrics`), `dynamo-error` with `http` (sanitized public messages), `dynamo-runtime` (`DynamoError`), and `dynamo-model-card` only with the `testing` feature.
- Must not depend on: `dynamo-llm`, `dynamo-discovery`, `dynamo-router-host`, `dynamo-preprocessor`, or a format crate. The SGLang and vLLM sidecars link the format crates, so anything this crate depends on is linked into every sidecar.
- It could not go into `dynamo-http-server` (which `dynamo-frontend-protocols` depends on, so that would be a cycle) and does not belong in `dynamo-frontend-protocols` (wire types; it would gain `dynamo-error` and `mime`).
- Status codes, response bodies and metric categories decided here are the HTTP contract of `/generate` and `/inference/v1/generate`. A refactor must not change them, nor the `DYN_GENERATE_FORMAT` variable.
- Features: `testing` exports `test_formats`, the stand-in formats that `dynamo-llm`'s tests install because they cannot link the real format crates.
- No log statements. Check: `cargo check -p dynamo-generate-format --all-targets --features testing`, and `cargo tree -p dynamo-generate-format | grep 'dynamo-llm '` must print nothing.
