---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Direct Engines
subtitle: Route to stock SGLang or vLLM gRPC servers with no Dynamo process beside the engine.
---

> [!WARNING]
> **Experimental.** Direct mode is new. Flags and behavior may change without notice.

Direct mode is the simplest way to put the Dynamo router in front of SGLang or vLLM. You
start stock engine servers with their gRPC port enabled, then start one frontend with the
engine URLs. The frontend probes each engine, learns its model, role, KV block size, and
KV-event sockets, and routes to it over gRPC. The KV-aware router, admission control,
prefill/decode orchestration, and request migration all run inside the frontend.

Compared with the [sidecar](../../developer-guide/knowledge-base/concepts/system-architecture/sidecar-backends.md),
direct mode gives up what a process beside the engine provides: dynamic discovery, index
recovery from a worker snapshot, and request-plane draining. In exchange it needs no etcd,
no NATS, and no sidecar container.

## Start the engines

**SGLang.** Each server needs `--grpc-port` and `--incremental-streaming-output`, the same
flags the sidecar requires. KV-aware routing also needs KV events:

```bash
python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 30000 --grpc-port 30001 --incremental-streaming-output \
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:5557","topic":"kv-events"}'
```

For disaggregated serving, start prefill and decode servers with `--disaggregation-mode`
as usual and list the prefill servers under `--prefill-engine-url`. SGLang reports its
role, and the frontend checks that the lists agree with it.

**vLLM.** Use the gRPC server shipped in the vLLM wheel, `vllm-rs serve`:

```bash
vllm-rs serve Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8100 --grpc-port 30001 \
  --reasoning-parser none -- \
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:5557","replay_endpoint":"tcp://*:5558","enable_kv_cache_events":true}'
```

For disaggregated serving, start the prefill engine with
`--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'` and the
decode engine with `kv_role: kv_consumer`, as the sidecar launch scripts do. vLLM does not
report its role over gRPC, so the frontend takes it from which list a URL is in:
`--prefill-engine-url` for prefill, `--engine-url` for decode (or aggregated when no
prefill engine is listed).

## Start the frontend

```bash
python3 -m dynamo.frontend --direct --direct-backend sglang \
  --engine-url http://127.0.0.1:30001 \
  --engine-url http://127.0.0.1:30011 \
  --router-mode kv --discovery-backend mem
```

`--direct-backend` is `sglang` (default) or `vllm`. `--engine-url` and
`--prefill-engine-url` may be repeated. `--model-path` is optional: the frontend uses the model
path or Hugging Face id each engine reports, and every engine must report the same one.
`--discovery-backend mem` keeps the run free of etcd; direct engines never use discovery.

Then send requests as usual:

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## What the frontend does with each URL

1. Connects, waits for `HealthCheck`, and calls `GetModelInfo` and `GetServerInfo`.
2. Derives the worker's role, KV block size, context length, data-parallel ranks,
   bootstrap host and port for prefill, and the ZMQ KV-event endpoints.
3. Registers the engine as a routable instance with a stable id derived from the URL,
   so every frontend replica agrees on worker ids.
4. Subscribes to the engine's KV events directly and feeds the router's index.
5. Dispatches each request over a gRPC `Generate` stream; a lost engine surfaces as a
   transport fault, so the router stops sending to it and in-flight requests migrate.

## Tool and reasoning parsers

The frontend parses tool calls and reasoning itself, with Dynamo's parsers. An SGLang engine
reports parser names Dynamo also knows, so the frontend adopts them. vLLM names its parsers
differently (`qwen3_xml` where Dynamo has `qwen3`), so a vLLM engine contributes none and
tool calls come back as plain text until you name the parsers:

```bash
python3 -m dynamo.frontend --direct --direct-backend vllm \
  --engine-url http://127.0.0.1:18701 \
  --dyn-tool-call-parser hermes --dyn-reasoning-parser qwen3 \
  --router-mode kv --discovery-backend mem
```

`--dyn-tool-call-parser` and `--dyn-reasoning-parser` take the same names as the sidecar flags
of the same name and override whatever an engine reported.

## Limitations

- SGLang and vLLM only; one engine family per frontend.
- vLLM dynamic LoRA loading, RL controls (pause, sleep, weight updates) and the encode role
  need the sidecar; they are hosted by its worker process.
- The HTTP frontend only. `--interactive` and `--kserve-grpc-server` reject `--direct`.
- The engine list is fixed at startup. Scale-out means restarting the frontend.
- KV-event replay recovers short gaps on vLLM (and on SGLang once it advertises a replay
  socket); a frontend restart warms the index from live events, with no snapshot of the
  engine's current cache.
