#!/usr/bin/env python3
"""Offline phase-split census for the NR-14 phase-ordered wire (E0).

Under the fused MoonEP path the proposed wire order is: (1) resident-destined
token rows first, (2) prefetch expert weights, (3) prefetch-only token rows.
Settled policy: a union row needed by BOTH a resident expert and a prefetch
slot on the same destination node travels in phase 1 — so phase 1 and phase 3
are a DISJOINT PARTITION of the lb_union node-level dedup union (no re-send).

This script prices that split with ZERO GPU time, for the same traffic inputs
the sweeps consume: per (source rank, dest node) it classifies every union row
as resident-needed (incl. shared) vs prefetch-only under the MoonEP plan's
virtual-expert mapping, and reports

  - phase-1 / phase-3 inter-node wire bytes and the prefetch-only fraction
    (the E3 go/no-go input: is there enough deferrable payload to matter?),
  - phase-2 weight-leg bytes (push_plan_stats census, direct and mcast),
  - per-round lb_union relay-chunk sizes if the union were cut per class
    (_chunk_bound applied to U_res and U_pref separately) — flags node pairs
    whose phase-3 round degenerates into latency-bound slivers,
  - per-rank resident vs prefetch-slot GEMM rows (E2 tail-exposure predictor).

Everything reuses the runtime replicas: matrix/routing loading from
predict_starvation.py (bit-equal dealer, real-trace sidecars), plan + mapping
from flux.testing.moonep_semantics / moonep_fused_map (pure CPU torch,
replicated). The partition invariant U_res + U_pref == U_mat is asserted
against build_fused_metadata's a2av_unique_counts.

Needs torch (the moonep modules are torch-based); runs on a login node.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_matrix  # noqa: E402
import predict_starvation  # noqa: E402  (deal_experts / load_routing reuse)

import torch  # noqa: E402


def load_moonep_modules():
    """flux.testing.{moonep_semantics,moonep_fused_map} — via the installed
    package when available, else file-loaded with stub package entries (the
    compiled flux extension is NOT needed; both modules import only
    dataclasses + torch)."""
    try:
        from flux.testing import moonep_fused_map as mf
        from flux.testing import moonep_semantics as ms

        return ms, mf
    except Exception:
        import importlib.util
        import types

        here = os.path.dirname(os.path.abspath(__file__))
        tdir = os.path.join(here, "..", "python", "flux", "testing")
        for name in ("flux", "flux.testing"):
            if name not in sys.modules:
                pkg = types.ModuleType(name)
                pkg.__path__ = []
                sys.modules[name] = pkg

        def _load(name):
            full = "flux.testing." + name
            if full in sys.modules:
                return sys.modules[full]
            spec = importlib.util.spec_from_file_location(
                full, os.path.join(tdir, name + ".py")
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[full] = mod
            spec.loader.exec_module(mod)
            setattr(sys.modules["flux.testing"], name, mod)
            return mod

        ms = _load("moonep_semantics")
        mf = _load("moonep_fused_map")
        return ms, mf


def classify_union(vmap, W, L):
    """(U_res, U_pref): [W, nn] node-level dedup union split by the phase
    policy. U_res[s, n] = unique tokens of source s needed by AT LEAST ONE
    resident expert on node n (phase 1, includes shared rows); U_pref[s, n] =
    unique tokens needed ONLY by prefetch slots there (phase 3)."""
    cfg = vmap.plan.cfg
    gpe, epn = vmap.gpe, cfg.epn
    ntokens = W * cfg.S
    nn = W // L
    vce = vmap.virtual_choosed.long()  # [ntokens, K]
    owner = vce // gpe
    node = owner // L
    is_res = (vce % gpe) < epn
    rows = torch.arange(ntokens).unsqueeze(1).expand_as(node)
    any_flags = torch.zeros(ntokens, nn, dtype=torch.bool)
    any_flags[rows.reshape(-1), node.reshape(-1)] = True
    res_flags = torch.zeros(ntokens, nn, dtype=torch.bool)
    res_flags[rows[is_res], node[is_res]] = True
    pref_only = any_flags & ~res_flags
    U_res = res_flags.view(W, cfg.S, nn).sum(1)
    U_pref = pref_only.view(W, cfg.S, nn).sum(1)
    return U_res, U_pref


def round_chunks(mf, U, L, n, m):
    """Per-relay chunk row counts of the (source node n -> dest node m)
    canonical stream under the balanced cut (moonep_fused_map._chunk_bound)."""
    cb = mf._chunk_bound
    return [cb(U, L, n, m, k + 1) - cb(U, L, n, m, k) for k in range(L)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matrix", help="path to <matrix_id>.txt (needs .meta.json)")
    ap.add_argument("--routing-file", help="<matrix_id>.routing.txt (trace family)")
    ap.add_argument("--family", help="gen_matrix family (alternative to --matrix)")
    ap.add_argument("--param", action="append", help="family param k=v")
    ap.add_argument("--W", type=int)
    ap.add_argument("--ranks-per-node", type=int)
    ap.add_argument("--budget-mib", type=int)
    ap.add_argument("--id", default="001")
    ap.add_argument("--topk", type=int, required=True)
    ap.add_argument("--G", type=int, default=128)
    ap.add_argument("--H", type=int, default=4096)
    ap.add_argument("--chunk-bytes", type=int, default=8192)
    ap.add_argument("--ffn-hidden", type=int, default=4096,
                    help="ffn shard rows per expert (EP=W in the driver, so"
                    " the shard is the full ffn_hidden)")
    ap.add_argument("--itemsize", type=int, default=2, help="dtype bytes (bf16=2)")
    ap.add_argument("--token-padding", type=int, default=128)
    ap.add_argument("--sliver-bytes", type=int, default=256 << 10,
                    help="flag phase-3 relay chunks smaller than this")
    ap.add_argument("--json", help="write the full census JSON here")
    args = ap.parse_args()

    if args.matrix:
        with open(args.matrix) as f:
            lines = f.read().split()
        W = int(lines[0])
        vals = [int(x) for x in lines[1:]]
        chunks = [[vals[s * W + d] // args.chunk_bytes for d in range(W)] for s in range(W)]
        with open(args.matrix.replace(".txt", ".meta.json")) as f:
            meta_json = json.load(f)
        T = meta_json["tokens_per_rank"]
        L = meta_json["ranks_per_node"]
        assert meta_json["topk"] == args.topk, "matrix topk != --topk"
    else:
        assert args.family and args.W and args.ranks_per_node and args.budget_mib
        params = dict(
            gen_matrix.FAMILY_DEFAULT_PARAMS[args.family],
            **gen_matrix.parse_params(args.param),
        )
        _, chunks, T = gen_matrix.generate(
            args.family, params, args.W, args.ranks_per_node,
            args.budget_mib, args.topk, args.chunk_bytes, args.id,
        )
        W, L = args.W, args.ranks_per_node

    if args.routing_file:
        assert args.matrix, "--routing-file requires --matrix"
        _, tok_experts = predict_starvation.load_routing(args.routing_file, W, T, args.topk)
    else:
        _, tok_experts = predict_starvation.deal_experts(chunks, T, args.topk, args.G)
    topk_all = torch.tensor(tok_experts, dtype=torch.int32)  # [W, T, K]

    ms, mf = load_moonep_modules()
    cfg = ms.MoonEPConfig(
        S=T, K=args.topk, E=args.G, R=W, H=args.H, token_padding=args.token_padding
    )
    plan = ms.compute_moonep_plan(cfg, topk_all)
    vmap = mf.build_virtual_map(plan, topk_all)
    meta = mf.build_fused_metadata(vmap, L)
    mf.preflight_metadata_checks(meta, W, L)

    nn = W // L
    U_res, U_pref = classify_union(vmap, W, L)
    U_mat = meta.a2av_unique_counts[:, W:].long()
    assert torch.equal(U_res + U_pref, U_mat), (
        "phase partition is not a disjoint cover of the union"
    )

    cb = args.chunk_bytes
    inter = torch.ones(W, nn, dtype=torch.bool)
    for s in range(W):
        inter[s, s // L] = False  # node-local traffic rides round 0, not the wire
    p1_rows = int(U_res[inter].sum())
    p3_rows = int(U_pref[inter].sum())
    p1_bytes, p3_bytes = p1_rows * cb, p3_rows * cb
    pref_frac = p3_rows / max(p1_rows + p3_rows, 1)

    pairs = mf.assign_gateways(plan, L)
    pstats = mf.push_plan_stats(pairs, L)
    ebytes = args.ffn_hidden * args.H * args.itemsize
    p2_direct = pstats["n_cross_legs"] * ebytes
    p2_mcast = pstats["n_cross_groups"] * ebytes

    # per-(node pair, class) balanced-cut chunk sizes for a would-be E3 split
    pair_rows = []
    slivers = 0
    for n in range(nn):
        for m in range(nn):
            if n == m:
                continue
            rc = round_chunks(mf, U_res, L, n, m)
            pc = round_chunks(mf, U_pref, L, n, m)
            slivers += sum(1 for c in pc if 0 < c * cb < args.sliver_bytes)
            pair_rows.append(dict(src_node=n, dst_node=m, res_chunks=rc, pref_chunks=pc))

    gpe, epn = vmap.gpe, cfg.epn
    splits = meta.splits.long().view(W, gpe)
    res_rows_rank = splits[:, :epn].sum(1)
    pref_rows_rank = splits[:, epn:].sum(1)
    pref_gemm_frac = (
        pref_rows_rank.double() / (res_rows_rank + pref_rows_rank).clamp(min=1).double()
    )

    mib = 1 << 20
    print(f"phase-split census: W={W} L={L} nn={nn} T={T} topk={args.topk} G={args.G}"
          f" gpe={gpe} (epn={epn} B={cfg.B})")
    print(f"  phase1 (resident+shared) inter-node: {p1_rows:8d} rows"
          f" {p1_bytes / mib:9.2f} MiB")
    print(f"  phase3 (prefetch-only)   inter-node: {p3_rows:8d} rows"
          f" {p3_bytes / mib:9.2f} MiB   pref-only frac {pref_frac:.4f}")
    print(f"  phase2 weights: direct {p2_direct / mib:.1f} MiB"
          f" ({pstats['n_cross_legs']} legs), mcast {p2_mcast / mib:.1f} MiB"
          f" ({pstats['n_cross_groups']} groups, multi={pstats['n_multi_groups']},"
          f" max_fan={pstats['max_fan']})")
    print(f"  weight:token wire ratio (direct): "
          f"{p2_direct / max(p1_bytes + p3_bytes, 1):.3f}")
    for pr in pair_rows:
        rc, pc = pr["res_chunks"], pr["pref_chunks"]
        print(f"  node {pr['src_node']}->{pr['dst_node']}:"
              f" res/relay {min(rc)}..{max(rc)} rows"
              f" ({min(rc) * cb / mib:.2f}..{max(rc) * cb / mib:.2f} MiB),"
              f" pref/relay {min(pc)}..{max(pc)} rows"
              f" ({min(pc) * cb / mib:.2f}..{max(pc) * cb / mib:.2f} MiB)")
    print(f"  phase-3 sliver chunks (<{args.sliver_bytes >> 10} KiB, >0): {slivers}")
    print("  per-rank prefetch-slot GEMM-row fraction (E2 tail exposure): "
          + " ".join(f"{f:.3f}" for f in pref_gemm_frac.tolist()))

    if args.json:
        out = dict(
            W=W, L=L, T=T, topk=args.topk, G=args.G, gpe=gpe, epn=epn, B=cfg.B,
            phase1_rows=p1_rows, phase1_bytes=p1_bytes,
            phase3_rows=p3_rows, phase3_bytes=p3_bytes,
            pref_only_frac=pref_frac,
            phase2_bytes_direct=p2_direct, phase2_bytes_mcast=p2_mcast,
            push_census=pstats, pair_rows=pair_rows,
            sliver_chunks=slivers,
            res_rows_per_rank=res_rows_rank.tolist(),
            pref_rows_per_rank=pref_rows_rank.tolist(),
            U_res=U_res.tolist(), U_pref=U_pref.tolist(),
        )
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
