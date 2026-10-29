#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Router crate layering, checked from `cargo tree` (normal dependencies, default features).
#
# 1. The protocol crate, the stage crates, and the kernel never depend on dynamo-runtime,
#    prometheus, dynamo-metrics, axum, reqwest, or hyper. Their embedded standalone hosts may,
#    behind `http` (and the indexer's `export`), which this script does not enable.
# 2. The composition crate (dynamo-kv-router) never depends on dynamo-runtime.
# 3. Dependency direction follows the pipeline: a stage depends only on the protocol crate and
#    the stages upstream of it; the kernel on every stage; the composition crate on the kernel.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
status=0
tree() { cargo tree -p "$1" -e normal --prefix none 2>/dev/null | awk '{print $1}' | sort -u; }
forbid() { # crate, forbidden-regex
  local hits; hits=$(tree "$1" | grep -E "$2" || true)
  if [ -n "$hits" ]; then echo "LAYERING: $1 depends on: $(echo "$hits" | tr '\n' ' ')"; status=1; fi
}
STAGE_FORBIDDEN='^(dynamo-runtime|dynamo-metrics|prometheus|axum|reqwest|hyper)$'
for c in dynamo-router-protocol dynamo-kv-indexer dynamo-slot-tracker dynamo-worker-selection dynamo-admission dynamo-scheduler; do
  forbid "$c" "$STAGE_FORBIDDEN"
done
forbid dynamo-kv-router '^dynamo-runtime$'
ROUTER='^dynamo-(router-protocol|kv-indexer|slot-tracker|worker-selection|admission|scheduler|kv-router)$'
allowed() { # crate, allowed-router-deps-regex
  local hits; hits=$(tree "$1" | grep -E "$ROUTER" | grep -vE "^$1$" | grep -vE "$2" || true)
  if [ -n "$hits" ]; then echo "LAYERING: $1 depends on router crates outside its chain: $(echo "$hits" | tr '\n' ' ')"; status=1; fi
}
allowed dynamo-router-protocol '^$'
allowed dynamo-kv-indexer '^dynamo-router-protocol$'
allowed dynamo-slot-tracker '^dynamo-(router-protocol|kv-indexer)$'
allowed dynamo-worker-selection '^dynamo-(router-protocol|kv-indexer|slot-tracker)$'
allowed dynamo-admission '^dynamo-(router-protocol|kv-indexer|slot-tracker|worker-selection)$'
allowed dynamo-scheduler '^dynamo-(router-protocol|kv-indexer|slot-tracker|worker-selection|admission)$'
allowed dynamo-kv-router '^dynamo-(router-protocol|kv-indexer|slot-tracker|worker-selection|admission|scheduler)$'
[ $status -eq 0 ] && echo "router layering: ok"
exit $status
