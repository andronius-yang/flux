#!/usr/bin/env python3
"""Static (no GPU, no capsule) variation predictor for the motivation figure.

For a trace family + budget it ensures the matrix / routing / oracle-load
sidecars exactly as the sweep runner would (same generators, same ids), then
reports the per-rank spreads that drive each panel:

  compute   : routed rows per rank (ring / COMET GEMM is ~linear in rows;
              measured b16: GEMM ms ~= 0.5 + 0.114 * krows ring, 0.21 * krows COMET)
  dispatch  : inter-node bytes RECEIVED per rank (send is equal by budget)
  EPLB      : rows per rank AFTER the pool-oracle placement (compute balance),
              inter-node dispatch bytes per rank after placement, and the
              one-shot placement SEND ledger per home rank (puts, inter-node
              bytes) — the placement-phase proxy (measured placement span
              tracks inter-node placement bytes per home rank)

Runs on the login node: the EPLB planner is CPU torch; `flux` is stubbed so
its CUDA extension is never imported. Conda python (torch) required.

  python figs/motivation/predict_variation.py \\
      --family "trace:model=Kimi-K2;pools=livecodebench/execution;layer=5;sem=homog;dslots=64:32" \\
      --budgets 16,32 [--layers 5,10,20] [--pools a,b] [--measured phases.json ...]
"""
import argparse, importlib, json, os, statistics as st, sys, types

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "sweeps"))
sys.path.insert(0, os.path.join(REPO, "test", "python", "moe_ag_scatter"))
# stub the flux package: the planner modules only need torch/numpy
flux = types.ModuleType("flux"); flux.__path__ = [os.path.join(REPO, "python", "flux")]
testing = types.ModuleType("flux.testing"); testing.__path__ = [os.path.join(REPO, "python", "flux", "testing")]
sys.modules["flux"] = flux; sys.modules["flux.testing"] = testing
import torch  # noqa: E402
import gen_matrix, gen_trace_routing  # noqa: E402
tm = importlib.import_module("flux.testing.traffic_matrix")
es = importlib.import_module("flux.testing.eplb_semantics")
us = importlib.import_module("flux.testing.ultraep_semantics")
from eplb_oracle import rebalance_experts  # noqa: E402

PLAT = dict(matrices_root=os.path.expandvars("${PSCRATCH}/workspace/andrewy/a2av_test_matrices/generated"),
            traces_root=os.path.expandvars("${PSCRATCH}/workspace/andrewy/moe_traces"), ranks_per_node=4)
SHAPE = dict(topk=8, G=384, H=7168, chunk_bytes=14336, ffn=2048)   # K2


def spread(v):
    v = [float(x) for x in v]; m = st.mean(v)
    return dict(min=min(v), max=max(v), max_min=max(v) / min(v) if min(v) else float("inf"), max_mean=max(v) / m if m else 0)


def fmt(s, unit="", k=1.0):
    return f"{s['min'] / k:8.1f}–{s['max'] / k:8.1f}{unit}  max/min {s['max_min']:4.2f}x  max/mean {s['max_mean']:4.2f}x"


def ensure(family, budget, W, instance="001"):
    name, mparams = family.partition(":")[0], gen_matrix.parse_params(family.partition(":")[2].split(";"))
    params = dict(gen_matrix.FAMILY_DEFAULT_PARAMS["trace"], **mparams)
    mid, path, sha = gen_matrix.ensure_matrix(name, mparams, W, PLAT["ranks_per_node"], budget, SHAPE["topk"], SHAPE["chunk_bytes"],
                                              instance, PLAT["matrices_root"], nexperts=SHAPE["G"], traces_root=PLAT["traces_root"])
    args = (params, W, PLAT["ranks_per_node"], budget, SHAPE["topk"], SHAPE["chunk_bytes"], instance, PLAT["matrices_root"])
    if "dslots" in params:
        _, _, lpath, _ = gen_trace_routing.ensure_oracle_sidecars(*args, traces_root=PLAT["traces_root"], nexperts=SHAPE["G"])
    else:
        lpath, _ = gen_trace_routing.ensure_eplb_load(*args, traces_root=PLAT["traces_root"], nexperts=SHAPE["G"])
    return mid, path, gen_trace_routing.routing_path_of(path), lpath


def predict(family, budget, W=16, instance="001", redundant=2):
    mid, mpath, rpath, lpath = ensure(family, budget, W, instance)
    rpn = PLAT["ranks_per_node"]; node = lambda r: r // rpn
    matrix = tm.parse_traffic_matrix(mpath)
    G, K, cb = SHAPE["G"], SHAPE["topk"], SHAPE["chunk_bytes"]
    ce = tm.load_routing_file(rpath, G, K)
    S = ce.shape[0] // W
    rows = (matrix.sum(0) // cb).tolist()
    recv_inter = [int(sum(matrix[s, d] for s in range(W) if node(s) != node(d))) for d in range(W)]
    out = dict(matrix_id=mid, budget=budget, tokens_per_rank=S,
               rows=spread(rows), recv_inter_MB=spread([b / 2**20 for b in recv_inter]),
               gemm_ring_ms=spread([0.5 + 0.114 * r / 1000 for r in rows]), gemm_comet_ms=spread([0.21 * r / 1000 for r in rows]))
    # EPLB pool-oracle placement, CPU, identical call to the driver
    cfg = us.UltraEPConfig(S=S, K=K, G=G, R=W, H=SHAPE["H"], D=rpn, R_red=redundant, locality_aware=False, interleave=True)
    topk_all = ce.reshape(W, S, K).cpu().int()
    tpe = us.loads_from_topk(cfg, topk_all)
    pool_load = json.load(open(lpath))["load"]
    plan = es.build_eplb_plan(cfg, tpe, pool_load, "global", W // rpn, rebalance_experts, replica_select="quota")
    prow = plan.physical_rows_per_rank()
    wire = us.wire_matrix(cfg, plan, topk_all)
    e_recv = [sum(wire[s][d] for s in range(W) if node(s) != node(d)) * cb for d in range(W)]
    e_send = [sum(wire[s][d] for d in range(W) if node(s) != node(d)) * cb for s in range(W)]
    pairs = es.weight_placement_pairs(plan)
    per_expert = 2 * 2 * SHAPE["ffn"] * SHAPE["H"]   # fc1 + fc2 bf16
    sends = [0] * W; send_inter = [0] * W
    for host, b, l, home in pairs:
        sends[home] += 1
        if node(host) != node(home): send_inter[home] += per_expert
    out.update(eplb_rows=spread(prow), eplb_recv_inter_MB=spread([b / 2**20 for b in e_recv]), eplb_send_inter_MB=spread([b / 2**20 for b in e_send]),
               place_puts=spread(sends), place_inter_GB=spread([b / 2**30 for b in send_inter]), rehomed=len(pairs),
               place_inter_GB_by_rank=[round(b / 2**30, 2) for b in send_inter], rows_by_rank=rows)
    return out


def report(p):
    print(f"  {p['matrix_id']}  tokens/rank {p['tokens_per_rank']}")
    print(f"    compute  rows/rank        {fmt(p['rows'])}   -> GEMM ms ring {p['gemm_ring_ms']['min']:.2f}–{p['gemm_ring_ms']['max']:.2f} ({p['gemm_ring_ms']['max_min']:.2f}x), COMET {p['gemm_comet_ms']['min']:.2f}–{p['gemm_comet_ms']['max']:.2f} ({p['gemm_comet_ms']['max_min']:.2f}x)")
    print(f"    dispatch inter-node recv  {fmt(p['recv_inter_MB'], ' MB')}")
    print(f"    EPLB     rows after place {fmt(p['eplb_rows'])}   recv inter {fmt(p['eplb_recv_inter_MB'], ' MB')}")
    print(f"    EPLB     placement puts   {fmt(p['place_puts'])}   inter-node GB {fmt(p['place_inter_GB'], ' GB')}   re-homed {p['rehomed']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="trace:model=Kimi-K2;pools=livecodebench/execution;layer=5;sem=homog;dslots=64:32")
    ap.add_argument("--budgets", default="16"); ap.add_argument("--layers", default=""); ap.add_argument("--pools", default="")
    ap.add_argument("--instance", default="001"); ap.add_argument("--measured", nargs="*", default=[])
    a = ap.parse_args()
    fams = []
    base = a.family
    layers = [x for x in a.layers.split(",") if x] or [None]
    pools = [x for x in a.pools.split(",") if x] or [None]
    for L in layers:
        for P in pools:
            f = base
            if L: f = ";".join(("layer=" + L if kv.startswith("layer=") else kv) for kv in f.split(";"))
            if P: f = ";".join(("pools=" + P if kv.startswith("pools=") else kv) for kv in f.split(";"))
            fams.append(f)
    results = []
    for f in fams:
        for b in [int(x) for x in a.budgets.split(",")]:
            print(f"== {f}  b{b}")
            try:
                p = predict(f, b, instance=a.instance); report(p); results.append((f, b, p))
            except Exception as e:
                print("   FAILED:", str(e)[:200])
    if a.measured:
        # validate the placement proxy against measured placement spans (per rank)
        d = {"cells": {}}
        for j in a.measured: d["cells"].update(json.load(open(j))["cells"])
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import build_figure_options as B
        for f, b, p in results:
            for cid, c in d["cells"].items():
                if c["budget_mib"] != b or not c["variant"].startswith("eplb"): continue
                P = {r: B.place_metrics(c, r) for r in c["ranks"]}
                meas = [P[str(r)]["span"] for r in range(16)]; pred = p["place_inter_GB_by_rank"]
                mx, my = st.mean(pred), st.mean(meas)
                cov = sum((x - mx) * (y - my) for x, y in zip(pred, meas)); r = cov / (sum((x - mx) ** 2 for x in pred) ** .5 * sum((y - my) ** 2 for y in meas) ** .5)
                print(f"  validation {cid}: measured placement span {min(meas):.0f}–{max(meas):.0f} ms ({max(meas)/min(meas):.2f}x) vs predicted inter-node GB {min(pred):.2f}–{max(pred):.2f} ({max(pred)/min(pred):.2f}x); Pearson r = {r:.2f}")
