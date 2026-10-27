#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Count `dynamo_kv_router::` import sites outside the crates allowed to have them.

Used as a migration gate for the router re-architecture: the count must reach
zero once `dynamo-kv-router` is removed. Exit code 1 when any site remains
outside the allow-list.

Usage: scripts/router_path_census.py [--allow lib/kv-router ...] [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

PATTERN = re.compile(r"\bdynamo_kv_router::")
ROOTS = ("lib", "deploy", "examples", "components", "tests")
SKIP_DIRS = {"target", ".venv", "node_modules", ".git"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow", nargs="*", default=["lib/kv-router"], help="path prefixes allowed to contain the pattern")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    by_location: Counter[str] = Counter()
    lines = 0
    offenders: list[tuple[str, int, str]] = []
    for root in ROOTS:
        top = os.path.join(args.root, root)
        if not os.path.isdir(top):
            continue
        for dirpath, dirnames, filenames in os.walk(top):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if not fn.endswith(".rs"):
                    continue
                path = os.path.relpath(os.path.join(dirpath, fn), args.root)
                if any(path.startswith(a.rstrip("/") + "/") or path == a for a in args.allow):
                    continue
                try:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            if PATTERN.search(line):
                                lines += 1
                                parts = path.split("/")
                                loc = "/".join(parts[:3]) if parts[0] in ("lib", "deploy") else "/".join(parts[:2])
                                by_location[loc] += 1
                                offenders.append((path, i, line.rstrip()))
                except OSError:
                    continue

    if args.json:
        print(json.dumps({"lines": lines, "by_location": by_location, "offenders": offenders}, indent=1))
    else:
        print(f"dynamo_kv_router:: import lines outside allow-list {args.allow}: {lines}")
        for loc, n in by_location.most_common():
            print(f"  {n:5d}  {loc}")
    return 1 if lines else 0


if __name__ == "__main__":
    sys.exit(main())
