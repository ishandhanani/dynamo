# lib/router/worker-selection

Worker selection for the Dynamo router: the selection plugin API (`plugin`), the composed
selection policy and the `WorkerSelector` trait (`selector`), LoRA narrowing, conditional
disaggregation, the session-affinity table (`affinity`), and the policies Dynamo ships
(`builtin`, including the runtime's stateless route pickers under `builtin::route`).

## Guardrails

- Depend only on `dynamo-router-types` for router vocabulary. `dynamo-runtime` depends on this
  crate for its route pickers; never introduce a dependency back on the runtime or on `lib/llm`.
- The selection hot path is per request: no allocation per candidate beyond the existing
  candidate rows, no `dyn` dispatch per rank, and every change to `selector` or `builtin`
  comes with before/after numbers from the `worker_selection` bench.
- Plugin traits (`WorkerFilter`, `WorkerScorer`, `WorkerPicker`) are a public contract; widen
  them only with defaulted methods.
- Test hooks for dependents live behind the `testing` feature; never enable it in release.
