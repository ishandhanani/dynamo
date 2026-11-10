<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Direct SGLang engines on Kubernetes

> [!WARNING]
> **Experimental.** Direct mode is new. Flags, manifests, and behavior may change
> without notice.

These manifests put the Dynamo frontend in front of stock SGLang engine pods
with one `kubectl apply`. They need no Dynamo operator, no CRDs, no etcd, no
NATS, and no sidecar. The engine pods run the upstream SGLang image and
nothing else.

The frontend watches the pods that match a label selector. A pod joins once
it is `Ready` and leaves when it stops being `Ready` or is deleted, so you
scale the engine Deployment and the frontend follows. Routing, KV-aware
selection, and prefill/decode orchestration run in the frontend; see
[Direct Engines](../../../../../docs/fern/pages/cli/model-deployment/direct-engines.md).

| Manifest | Topology |
|----------|----------|
| [`agg.yaml`](agg.yaml) | Two aggregated engines behind a KV-aware frontend |
| [`disagg.yaml`](disagg.yaml) | One prefill and one decode engine, transferring KV over NIXL |

## Deploy

1. Build a frontend image from this branch and set it in the manifest, in place of
   `<your-registry>/dynamo-frontend:<tag>`.
2. For a gated model, create the token secret. The manifests treat it as optional:

   ```bash
   kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=<token>
   ```

3. Apply:

   ```bash
   kubectl apply -f agg.yaml
   kubectl rollout status deployment/dynamo-direct-frontend
   ```

4. Send a request:

   ```bash
   kubectl port-forward svc/dynamo-direct-frontend 8000:8000 &
   curl -s localhost:8000/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
   ```

5. Scale the engines; the frontend logs `engine pod serving` for each new pod:

   ```bash
   kubectl scale deployment/sglang-engine --replicas=4
   kubectl logs deployment/dynamo-direct-frontend | grep 'engine pod'
   ```

## How the frontend finds engines

The frontend runs with:

```bash
python3 -m dynamo.frontend --direct \
  --discovery-backend kubernetes-pods \
  --engine-selector app=sglang-engine \
  --engine-grpc-port 30001 \
  --router-mode kv
```

| Flag | Meaning |
|------|---------|
| `--discovery-backend kubernetes-pods` | Find engines from pods instead of `--engine-url`. Requires `--direct`. |
| `--engine-selector` | Label selector for the aggregated engines, or the decode engines when `--prefill-engine-selector` is set. |
| `--prefill-engine-selector` | Label selector for the prefill engines. |
| `--engine-namespace` | Namespace of the engine pods. Defaults to the frontend's own. |
| `--engine-grpc-port` | gRPC port of the engines. A pod's `nvidia.com/dynamo-engine-grpc-port` annotation overrides it. |

- An engine's URL is `http://<pod IP>:<port>`.
- Each pod is probed when it becomes `Ready`, retrying with backoff until its
  engine answers. The frontend then registers the engine in its own in-memory
  discovery, with the model card a sidecar would publish, and serves it as it
  serves sidecar workers. `/v1/models` is empty until the first engine
  registers.
- A pod that stops being `Ready` or is deleted is withdrawn from discovery.
- All engines must serve one model. A pod serving a different model is not
  registered.
- The frontend's service account needs `get`, `list`, and `watch` on pods in the
  engine namespace. The manifests grant exactly that with a `Role`.

## Limitations

- Engine ids derive from the pod name, as with the `kubernetes` discovery backend.
  A pod is withdrawn while it is not `Ready`, so its KV index entries start empty
  when it returns.
- A KV-aware frontend reaches each engine's KV-event socket on port 5557 of the pod
  IP. Network policies must allow it, and the gRPC and bootstrap ports.
- Several frontend replicas each route independently with their own KV index.
