# lib/router/kv-router

The composition crate for the router: `SelectionService`, the plugin registry, router
configuration, and the standalone selection service. The kernel is `lib/router/scheduler`;
the stage crates are its siblings under `lib/router`.

- This crate may depend on every router crate and on axum; it must not depend on
  `dynamo-runtime`. Hosts inject runtime concerns (discovery, transport, registries) from
  above.
- The plugin API is re-exported here at its published paths. Changing a re-export is a
  plugin-API change; `tests/plugin_api_compat.rs` must keep compiling unchanged.
- Keep `axum` default features disabled and enable only what the routes use.
- When router configuration is serialized inside a model deployment card, it is part of the
  N-2 worker/frontend wire contract described in `lib/llm/AGENTS.md`.
