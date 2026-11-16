# LLM Benchmarks

The tokenizer and media decode benchmarks live with the preprocessor: see
[`lib/preprocessor/benches`](../../preprocessor/benches/README.md).

## request_trace_finish_metadata

Measures request finish metadata overhead with tracing disabled/enabled, plus
tool-call metadata recording across increasing numbers of calls.

```bash
cargo bench --bench request_trace_finish_metadata -p dynamo-llm --features request-trace-bench
```
