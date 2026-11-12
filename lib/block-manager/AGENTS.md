# lib/block-manager

`dynamo-block-manager` owns the v1 KV block manager (`KvBlockManager`): block pools and state, layouts and storage (system, pinned, device, disk, NIXL), offload and onboard between tiers, the distributed leader/worker transfer path, the KV event consolidator, the controller endpoint and client, the standalone `kvbm_*` metrics listener, and the `tokens` and `dtype` types its API exposes. The `lib/kvbm-*` crates are the v2 rewrite and are not workspace members; this crate does not use them.

- Depends on (Dynamo crates): `dynamo-runtime`, `dynamo-memory`, `dynamo-kv-events`, `dynamo-kv-hashing`.
- Must not depend on: `dynamo-llm`, any router crate, `dynamo-http-server`. A host that links the router plugins passes them in (`ControlClient::new` takes the `PushRouterOptions`).
- `lib/bindings/kvbm` is the only in-repo consumer and imports `dynamo_block_manager` directly; `dynamo-llm` does not depend on this crate and keeps no re-export of it.
- `src/tokens.rs` is an older fork of `dynamo-tokens` (`lib/tokens`); do not grow it.
- Test: `cargo test -p dynamo-block-manager` runs the CPU-only tests. `--features testing-cuda` needs a GPU, `testing-nixl` needs NIXL, `testing-etcd` needs etcd.
