---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Direct Engines
subtitle: Route to stock SGLang gRPC servers with no Dynamo process beside the engine.
---

> [!WARNING]
> **Experimental.** Direct mode is new. Flags and behavior may change without notice.

Direct mode is the simplest way to put the Dynamo router in front of SGLang. You
start stock engine servers with their gRPC port enabled, then start one frontend with the
engine URLs. The frontend probes each engine, learns its model, role, KV block size, and
KV-event sockets, and routes to it over gRPC. The KV-aware router, admission control,
prefill/decode orchestration, and request migration all run inside the frontend.

Compared with the [sidecar](../../developer-guide/knowledge-base/concepts/system-architecture/sidecar-backends.md),
direct mode gives up what a process beside the engine provides: dynamic discovery, index
recovery from a worker snapshot, and request-plane draining. In exchange it needs no etcd,
no NATS, and no sidecar container.

## Start the engines

Each server needs `--grpc-port` and `--incremental-streaming-output`, the same
flags the sidecar requires. KV-aware routing also needs KV events:

```bash
python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 30000 --grpc-port 30001 --incremental-streaming-output \
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:5557","topic":"kv-events"}'
```

For disaggregated serving, start prefill and decode servers with `--disaggregation-mode`
as usual and list the prefill servers under `--prefill-engine-url`. SGLang reports its
role, and the frontend checks that the lists agree with it.

## Start the frontend

```bash
python3 -m dynamo.frontend --direct --direct-backend sglang \
  --engine-url http://127.0.0.1:30001 \
  --engine-url http://127.0.0.1:30011 \
  --router-mode kv --discovery-backend mem
```

`--direct-backend` is `sglang`. `--engine-url` and
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
reports parser names Dynamo also knows, so the frontend adopts them. `--dyn-tool-call-parser`
and `--dyn-reasoning-parser` take the same names as the sidecar flags of the same name and
override whatever the engine reported.

## Limitations

- SGLang only.
- The HTTP frontend only. `--interactive` and `--kserve-grpc-server` reject `--direct`.
- The engine list is fixed at startup. Scale-out means restarting the frontend.
- No KV-event replay: SGLang does not advertise a replay socket, so a frontend restart
  warms the index from live events, with no snapshot of the engine's current cache.
