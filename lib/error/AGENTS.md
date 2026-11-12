# lib/error

`dynamo-error` is the error contract every Dynamo component classifies failures with (DEP #14354): `DynamoError`, `ErrorClass`, `ErrorReason`, `Diagnostic`, `PublicDetails`, the builder, and the reason catalog. The component closest to the cause classifies once; protocol boundaries render the class, and a status code is a function of the class, never of the reason.

- Depends on (Dynamo crates): none.
- Features: `http` adds the `http` module (class-to-status policy, overload status, sanitized response vocabulary) and the `http` crate for `StatusCode`; nothing else may be added behind it.
- Must not depend on: the runtime, tokio, any transport (zmq, axum, hyper, reqwest, prometheus), any router crate.
- Hosts import this crate directly; `dynamo-runtime` re-exports it as `dynamo_runtime::error`.
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`.
