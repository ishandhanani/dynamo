---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sidecar Backends
subtitle: Run Dynamo beside a stock inference engine through its native gRPC API.
---

> [!WARNING]
> **Experimental.** Sidecar packaging, launchers, and API coverage are still
> evolving. The sidecar path does not yet match every feature of the in-process
> backends.

A Dynamo sidecar runs beside the inference engine process. It registers the
engine with Dynamo discovery and forwards engine events into the Dynamo event
plane. Today, requests also pass through the sidecar. The target design routes
requests directly to the engine's native gRPC service.

## Design Goals

- Keep the upstream engine's native serve path and argument surface.
- Move toward public, versioned gRPC contracts with explicit backward
  compatibility instead of importing private engine APIs.
- Isolate Dynamo and engine dependencies in separate processes.
- Attribute failures through engine-specific and Dynamo-specific logs and health
  checks.
- Reuse Dynamo's frontend, routing, planning, and disaggregated-serving
  orchestration.

## Architecture

```mermaid
flowchart LR
  F[Dynamo Frontend]
  subgraph W[Same host or Kubernetes pod]
    direction TB
    S[Dynamo Sidecar] <-->|Native gRPC| E[Inference Engine]
  end
  S -->|Discovery and Event planes| F
  F -->|Request plane*<br/>Native gRPC| E
```

<sup>*</sup> The direct request path is the target design. Today, requests pass
through the sidecar.

In the target design, the frontend and router resolve the engine endpoint
through discovery, then send requests directly to the engine. The sidecar stays
off the request path and uses the engine's native gRPC service for metadata and
event integration with Dynamo's discovery and event planes.

## Target Responsibilities

| Layer | Responsibility |
|---|---|
| Dynamo frontend and router | OpenAI-compatible API, preprocessing, routing, and direct native gRPC requests to the engine |
| Dynamo sidecar | Engine registration and discovery, plus metadata and event forwarding |
| Inference engine | Native gRPC request serving, scheduling, sampling, token generation, KV cache, and GPU execution |

## Container Packaging

The sidecar Dockerfile builds all three engine-specific sidecar executables into
one CPU-only image. Deployments select an engine by setting the container
`command`:

| Engine | Container command |
|---|---|
| vLLM | `dynamo-vllm-sidecar` |
| SGLang | `dynamo-sglang-sidecar` |
| TensorRT-LLM | `dynamo-trtllm-sidecar` |

The image's default entrypoint, `dynamo-sidecar`, maps the short names `vllm`,
`sglang`, and `trtllm` onto those executables. It is a convenience for ad-hoc
`docker run`; the deployment manifests override it with `command`. The inference
engine remains in a separate GPU container, so the sidecar image does not
include vLLM, SGLang, TensorRT-LLM, CUDA, or engine-specific Python
dependencies.

No published sidecar image is available yet. Build the sidecar image from the
[sidecar Dockerfile](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/Dockerfile).

## Current Readiness

| Backend | Local launcher | Kubernetes example |
|---|---|---|
| [vLLM](../../modular-components/backends/vllm/sidecar.md) | Aggregated and disaggregated | Aggregated and disaggregated |
| [SGLang](../../modular-components/backends/sglang/sidecar.md) | Aggregated and disaggregated | Aggregated and disaggregated |
| [TensorRT-LLM](../../modular-components/backends/tensorrt-llm/sidecar.md) | Aggregated | Aggregated |

Disaggregated launch paths require multiple GPUs and use NIXL for KV transfer.
This table describes validated launch topologies, not feature parity with the
in-process backends.

## Direct Dispatch (Experimental)

The target design's direct request path is available for vLLM behind two
opt-ins, one on each side:

- The sidecar advertises a frontend-routable address for the engine's gRPC
  service with `--advertise-grpc-endpoint` (or `DYN_ADVERTISE_GRPC_ENDPOINT`).
  It is published in the worker's runtime config as `native_grpc_endpoint`,
  together with the role the engine serves (`native_grpc_mode`).
- The frontend process sets `DYN_ROUTER_DIRECT_DISPATCH=vllm`. Every KV routing
  host then connects to each advertising worker and sends admitted requests to
  vLLM directly. Workers that do not advertise an endpoint keep using the
  request plane, so a fleet can migrate one worker at a time.

The vLLM sidecar can also register the worker with a
[standalone selection service](../../modular-components/router/standalone-selection.md)
over HTTP (`--selection-catalog-url`, with a heartbeat-renewed lease and
deregistration on shutdown), publishing the advertised gRPC address as the
dispatch endpoint and vLLM's own KV-event ZMQ publishers as the event sources.
Together with direct dispatch this is the runtime-free registration path: no
etcd or NATS is needed between that worker and a selector.

Selection, booking, cancellation, and error semantics are unchanged: the direct
path uses the same request conversion, stream adapter, and gRPC-status error
mapping as the sidecar, so a dispatch failure releases the router's booking and
reaches the migration layer with the same error type it would have had over the
request plane. The sidecar stays in the pod for registration, KV events, and
health. Request-plane fault detection does not observe direct-dispatch
failures; worker liveness comes from discovery.

## Running Without etcd Or NATS

None of the three planes requires an infrastructure daemon on a single host or a
shared-filesystem VM group:

| Plane | Setting | Effect |
|---|---|---|
| Discovery | `DYN_DISCOVERY_BACKEND=file` (+ `DYN_FILE_KV=<shared dir>`) or `mem` | Registrations live in files (or in-process); no etcd. |
| Request | `DYN_REQUEST_PLANE=tcp` (default) | Frontend to worker over TCP; no NATS. With direct dispatch, straight to the engine's gRPC. |
| Event | `DYN_EVENT_PLANE=zmq` (default) | Workers publish KV events over ZMQ; the router subscribes directly. |
| Selection | `DYN_ROUTER_EMBEDDED_SELECTION=1` | Scheduling on the embedded selection partition, whose replica sync is ZMQ. |

Validated locally with `dynamo.frontend --router-mode kv` and two
`dynamo.mocker` workers under exactly these settings, with no etcd or NATS
process running: the model was served within ten seconds and chat completions
succeeded. Across Kubernetes pods, use `DYN_DISCOVERY_BACKEND=kubernetes`, or
the selection-catalog registration above, in place of the file backend.

See the
[sidecar Dockerfile, source, and engine-specific READMEs](https://github.com/ai-dynamo/dynamo/tree/main/lib/sidecar)
for implementation details.
