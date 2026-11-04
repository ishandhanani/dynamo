# lib/kvm

`dynamo-kvm` carries KV cache hints from Dynamo to an inference engine scheduler:
`HintController` (bounded queue, fire-and-forget, never on a request's hot path),
`transport` (the `HintTransport` trait and the ZMQ DEALER transport), and `wire`
(SGLang's `PrefetchReqInput` / `PrefetchReqOutput` msgpack layout).

- Dependencies: `tokio`, `tmq`, `zmq`, `rmp-serde`, `serde`, `futures`, `async-trait`,
  `anyhow`, `tracing`. No Dynamo crate, no environment reads: the host (`lib/llm`)
  resolves configuration and worker endpoints and passes them in.
- Forbidden: router policy, request routing, HTTP routes, worker discovery. Those
  live in `lib/kv-router` and `lib/llm`; this crate only delivers hints.
- The `wire` tests pin the exact bytes SGLang's `io_struct.msgpack_encode` produces.
  A change to the SGLang structs is a change to those bytes; update both together.
- Gate: `cargo test -p dynamo-kvm && cargo clippy -p dynamo-kvm --all-targets`.
