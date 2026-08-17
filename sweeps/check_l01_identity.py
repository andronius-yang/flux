#!/usr/bin/env python3
################################################################################
#
# Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
"""Capsule-level decomposition check for the combined l0+l1 pass (login-node
script: stdlib + csv only, no torch).

Given one capsule dir (sweeps/results/runs/<run_id>/) and three cells — a
layer0 isolated cell, a layer1 amortized (tmamo) isolated cell, and a combined
l01 isolated cell, all on the SAME matrix and build — assert the identity the
l01 bench (test/python/moe_combined/test_moe_l0l1_traffic.py) is built to
satisfy:

    mean_iters(max_ranks e2e_ms[l01])
      ~= mean_iters(max_ranks e2e_ms[l0])
       + mean_iters(max_ranks act_ms[l01])
       + mean_iters(max_ranks e2e_ms[l1 tmamo])

within --tolerance (relative, default 0.10). Aggregation mirrors
sweeps/SCHEMA.md's isolated-mode rule: per-iteration MAX across ranks, then
mean over iterations — computed here from raw metrics.csv rows (aggregation is
the summarizer's job; the capsule stays raw).

Row selection: impl=flux by default; --impl torch selects the unfused arms
instead (the torch l01 arm emits the same metric names). A torch LAYER0 cell
from the l0 traffic bench has no e2e_ms rows — for that case the per-(rank,
iter) e2e is reconstructed as comm_ms + scatter_ms + gemm_ms (the bench's
window is their concatenation).

Exit status: 0 when the identity holds, 1 on violation (or selection errors).
A violation is a FINDING about the combined pass (lost overlap, sync
amortization, schedule-inheritance mispricing) — rerun per protocol rule 2
before believing a wild residual.

Usage:
    python sweeps/check_l01_identity.py \\
        --capsule sweeps/results/runs/<run_id> \\
        --l0-cell hier_..._iso --l1-cell l1_hier_..._tmamo_iso --l01-cell l01_...

Cell selectors are exact cell_ids or unique substrings of them.
"""

import argparse
import csv
import os
import sys

# torch-l0 reconstruction set (see module docstring)
L0_TORCH_PHASES = ("comm_ms", "scatter_ms", "gemm_ms")


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_metrics(capsule):
    path = os.path.join(capsule, "metrics.csv")
    if not os.path.isfile(path):
        die(f"no metrics.csv under {capsule}")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def resolve_cell(rows, selector, label):
    """Exact cell_id match wins; otherwise the selector must be a substring of
    exactly one distinct cell_id."""
    cell_ids = sorted({r["cell_id"] for r in rows})
    if selector in cell_ids:
        return selector
    hits = [c for c in cell_ids if selector in c]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        die(f"{label}: selector {selector!r} matches no cell_id in this capsule")
    die(f"{label}: selector {selector!r} is ambiguous: {hits}")


def max_rank_mean(rows, cell_id, impl, metric, reconstruct_l0_torch=False):
    """SCHEMA isolated aggregation: per-iteration max across ranks of `metric`,
    then mean over iterations. With reconstruct_l0_torch, a (rank, iter) with
    no `metric` row falls back to the sum of the torch layer0 phase metrics."""
    per_ri = {}  # (rank, iter) -> value; phases summed on fallback
    have_direct = set()
    for r in rows:
        if r["cell_id"] != cell_id or r["impl"] != impl:
            continue
        key = (int(r["rank"]), int(r["iter"]))
        if r["metric"] == metric:
            per_ri[key] = float(r["value_ms"])
            have_direct.add(key)
        elif (
            reconstruct_l0_torch
            and r["metric"] in L0_TORCH_PHASES
            and key not in have_direct
        ):
            per_ri[key] = per_ri.get(key, 0.0) + float(r["value_ms"])
    if not per_ri:
        die(f"no {impl}/{metric} rows for cell {cell_id}")
    by_iter = {}
    for (rank, it), val in per_ri.items():
        by_iter[it] = max(by_iter.get(it, 0.0), val)
    vals = [by_iter[i] for i in sorted(by_iter)]
    return sum(vals) / len(vals), len(vals)


def cell_modes(rows, cell_id):
    return sorted({r["mode"] for r in rows if r["cell_id"] == cell_id})


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--capsule", required=True, help="sweeps/results/runs/<run_id> dir")
    p.add_argument("--l0-cell", required=True, help="layer0 isolated cell_id (or substring)")
    p.add_argument("--l1-cell", default=None,
                   help="layer1 tmamo isolated cell_id (or substring); omit "
                   "with --l1-from-l01 for arm families without a standalone "
                   "l1 cell (epic)")
    p.add_argument("--l1-from-l01", action="store_true",
                   help="source the l1 term from the l01 cell's own l1_ms "
                   "metric (the epic driver emits l1_ms = e2e - l0 - act, so "
                   "the WITHIN-cell identity is exact by construction; the "
                   "informative check is then l0-cell-vs-l01-cell l0_ms "
                   "agreement, reported alongside)")
    p.add_argument("--l01-cell", required=True, help="combined l01 cell_id (or substring)")
    p.add_argument(
        "--impl",
        default="flux",
        help="metrics.csv impl to aggregate (default flux; torch selects the unfused arms)",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.10,
        help="max |residual| / e2e[l01], relative (default 0.10)",
    )
    args = p.parse_args()

    rows = load_metrics(args.capsule)
    l0 = resolve_cell(rows, args.l0_cell, "--l0-cell")
    l01 = resolve_cell(rows, args.l01_cell, "--l01-cell")
    if args.l1_from_l01:
        e0, n0 = max_rank_mean(rows, l0, args.impl, "e2e_ms")
        e01, n01 = max_rank_mean(rows, l01, args.impl, "e2e_ms")
        act, _ = max_rank_mean(rows, l01, args.impl, "act_ms")
        l1_term, _ = max_rank_mean(rows, l01, args.impl, "l1_ms")
        l0_in_l01, _ = max_rank_mean(rows, l01, args.impl, "l0_ms")
        rhs = e0 + act + l1_term
        residual = e01 - rhs
        rel = residual / e01 if e01 else float("inf")
        print(f"capsule : {args.capsule}  (--l1-from-l01 mode)")
        print(f"impl    : {args.impl}")
        print(f"l0   e2e: {e0:9.3f} ms  [{l0}] ({n0} iters)")
        print(f"l01 l0  : {l0_in_l01:9.3f} ms  [{l01}]  <- l0-vs-l01 "
              f"agreement is the informative check "
              f"({(l0_in_l01 - e0) / e0 if e0 else 0:+.1%})")
        print(f"l01  act: {act:9.3f} ms  [{l01}]")
        print(f"l01  l1 : {l1_term:9.3f} ms  [{l01}]")
        print(f"sum     : {rhs:9.3f} ms")
        print(f"l01  e2e: {e01:9.3f} ms  [{l01}] ({n01} iters)")
        print(f"residual: {residual:+9.3f} ms  ({rel:+.1%}, tolerance "
              f"±{args.tolerance:.0%})")
        if abs(rel) > args.tolerance:
            print("FAIL: decomposition identity violated")
            sys.exit(1)
        print("OK")
        return
    if args.l1_cell is None:
        p.error("--l1-cell is required unless --l1-from-l01")
    l1 = resolve_cell(rows, args.l1_cell, "--l1-cell")

    # soft sanity: the identity is defined over isolated-discipline cells of
    # one capsule; mode mismatch or an l1 isolated-timing cell are warnings,
    # not errors — the caller may be probing exactly that.
    modes = {c: cell_modes(rows, c) for c in (l0, l1, l01)}
    if len({tuple(m) for m in modes.values()}) != 1:
        print(f"WARNING: cells span different modes: {modes} — the identity is "
              "only defined within one mode (SCHEMA never-mix rule)")
    if "_tmiso" in l1:
        print(f"WARNING: {l1} looks like timing_mode=isolated; the identity "
              "expects the amortized (_tmamo) l1 cell — the combined pass "
              "inherits its schedule from layer0")

    e0, n0 = max_rank_mean(rows, l0, args.impl, "e2e_ms", reconstruct_l0_torch=True)
    e1, n1 = max_rank_mean(rows, l1, args.impl, "e2e_ms")
    e01, n01 = max_rank_mean(rows, l01, args.impl, "e2e_ms")
    act, _ = max_rank_mean(rows, l01, args.impl, "act_ms")

    rhs = e0 + act + e1
    residual = e01 - rhs
    rel = residual / e01 if e01 else float("inf")

    print(f"capsule : {args.capsule}")
    print(f"impl    : {args.impl}")
    print(f"l0   e2e: {e0:9.3f} ms  [{l0}] ({n0} iters)")
    print(f"l01  act: {act:9.3f} ms  [{l01}]")
    print(f"l1   e2e: {e1:9.3f} ms  [{l1}] ({n1} iters)")
    print(f"sum     : {rhs:9.3f} ms")
    print(f"l01  e2e: {e01:9.3f} ms  [{l01}] ({n01} iters)")
    print(f"residual: {residual:+9.3f} ms  ({rel:+.1%} of l01 e2e, tolerance ±{args.tolerance:.0%})")

    if abs(rel) > args.tolerance:
        print("FAIL: decomposition identity violated")
        sys.exit(1)
    print("OK: decomposition identity holds")


if __name__ == "__main__":
    main()
