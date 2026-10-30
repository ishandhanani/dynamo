# lib/router/admission

Admission for the Dynamo router: the per-class policy queue (`policy_queue`), the
request-classifier plugin API (`classifier`), and the classifier runtime that delivers lifecycle
events to a plugin (`runtime`).

## Guardrails

- Depend only on `dynamo-router-protocol` for router vocabulary; never introduce `dynamo-runtime`
  or `lib/llm` dependencies.
- The policy queue is on the admission hot path: keep push/pop allocation-free beyond the entry
  itself and gate changes on the `policy_queue` bench.
- Lifecycle events to a classifier are lossless and never block the scheduler; a plugin's
  panic or stall must not disable classification router-wide.
- The classifier traits are a public contract; widen them only with defaulted methods.
