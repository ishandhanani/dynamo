# lib/kvm

`dynamo-kvm` carries KV cache hints from Dynamo to an inference engine scheduler:
`HintController` (bounded queues, fire-and-forget, never on a request's hot path),
`transport` (the `HintTransport` trait and the ZMQ DEALER transport to schedulers; the
`PinSink` trait and the ZMQ PUSH to the Mooncake master's hint socket), and `wire`
(SGLang's `PrefetchReqInput` / `PrefetchReqOutput` and the store's `Pin` msgpack layouts).

- Dependencies: `tokio`, `tmq`, `zmq`, `rmp-serde`, `serde`, `futures`, `async-trait`,
  `anyhow`, `tracing`. No Dynamo crate, no environment reads: the host (`lib/llm`)
  resolves configuration and worker endpoints and passes them in.
- Forbidden: router policy, request routing, HTTP routes, worker discovery. Those
  live in `lib/kv-router` and `lib/llm`; this crate only delivers hints.
- The `wire` tests pin the exact bytes SGLang's `io_struct.msgpack_encode` and the
  Mooncake master's pin decoder use. A change on either side is a change to those
  bytes; update both together.
- Gate: `cargo test -p dynamo-kvm && cargo clippy -p dynamo-kvm --all-targets`.
