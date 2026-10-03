"""54_placement_fidelity.py -- handoff 54 (plan 10, S0 P3): does the serving placement reflect the requests' expert
popularity? Offline replay of a RECORDED routing (SGLang per_token expert-distribution dumps of the eval traffic)
through the lopep router under several placements, with the serving swap dynamics.

  PYTHONPATH=<lopep>/python python 54_placement_fidelity.py --eval <dumps> --mode decode \
      --calib lcbtd=<calib dir> --calib dec=<calib dir> [--history <dumps>] [--layers 0,4,8,...] [--C 0.25]

Placements: every --calib NAME=DIR (its lopep_config.json p2l per layer), "oracle" (the same solver as
lopep_sglang.calibrate on the eval rows themselves, 4096 rows per rank) and "rr" (home experts only, no replicas).
Per layer and recorded step (all ranks' rows of one forward pass, truncated to the smallest rank's row count):
  1. the serving swap decision on the step's expert loads (lopep.swap.decide_swaps, band C; the placement it
     returns persists into the next step, as in serving), unless --swap 0;
  2. the router's water-fill stage (lopep.routing.route_water_fill; the vacate pass moves copies between replicas of
     one expert and is not modelled) -> rows per GPU -> GPU max / mean and node max / mean.
Reports per placement the mean imbalance, the swap-trigger rate and moves per step, per layer and overall; with
--history, the per-layer correlation of the history and eval expert popularity.
"""
import argparse
import glob
import json
import os
import statistics as st

import numpy as np
import torch

from lopep.placement import Placement, demand_histogram, rebuild_l2p, solve_placement
from lopep.routing import c_rational, route_water_fill
from lopep.swap import decide_swaps


def load_steps(dumps, modes):
    """{forward_pass_id: {rank: int32 [L_model, T, K]}} of the kept forward modes"""
    steps = {}
    for f in sorted(glob.glob(os.path.join(dumps, "*.pt"))):
        rank = int(os.path.basename(f)[:-3].rsplit("_", 1)[1])
        for rec in torch.load(f, map_location="cpu", weights_only=False)["records"]:
            t = rec["topk_ids_of_layer"]
            if t.shape[1] == 0 or (modes is not None and rec["forward_mode"] not in modes):
                continue
            steps.setdefault(rec["forward_pass_id"], {})[rank] = t.int()
    return steps


def placement_of(p2l_list, G, R):
    p2l = torch.tensor(p2l_list, dtype=torch.int32)
    return Placement(p2l, rebuild_l2p(p2l, G, R), torch.bincount(p2l[p2l >= 0].long(), minlength=G).int(), {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--history", default="")
    ap.add_argument("--mode", default="decode", choices=["decode", "prefill", "all"])
    ap.add_argument("--calib", action="append", default=[], help="NAME=calibration dir (lopep_config.json)")
    ap.add_argument("--layers", default="", help="comma list of model layers (default every 4th)")
    ap.add_argument("--C", type=float, default=0.25)
    ap.add_argument("--swap", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()
    modes = {"decode": {2}, "prefill": {1}, "all": None}[a.mode]
    steps = load_steps(a.eval, modes)
    ids = sorted(steps)
    R = max(max(v) for v in steps.values()) + 1
    ids = [i for i in ids if len(steps[i]) == R]
    if a.max_steps:
        ids = ids[:a.max_steps]
    t0 = steps[ids[0]][0]
    n_model, K = t0.shape[0], t0.shape[2]
    calibs = {}
    for c in a.calib:
        name, d = c.split("=", 1)
        calibs[name] = json.load(open(os.path.join(d, "lopep_config.json") if os.path.isdir(d) else d))
    any_c = next(iter(calibs.values()))
    G, L = any_c["shape"]["num_experts"], any_c["ranks_per_node"]
    nlp = len(any_c["p2l"]["0"]) // R
    layers = [int(v) for v in a.layers.split(",")] if a.layers else list(range(0, n_model, 4))
    Cn, Cd = c_rational(a.C)
    hist_pop = None
    if a.history:
        hs = load_steps(a.history, modes)
        hist_pop = {l: torch.zeros(G, dtype=torch.float64) for l in layers}
        for by_rank in hs.values():
            for t in by_rank.values():
                for l in layers:
                    hist_pop[l] += torch.bincount(t[l].reshape(-1).long(), minlength=G).double()
    print(f"[fidelity] eval {a.eval} ({a.mode}): {len(ids)} steps x {R} ranks, layers {layers}, G {G} L {L} nlp {nlp} "
          f"C {a.C} swap {a.swap}")
    names = list(calibs) + ["oracle", "rr"]
    res = {n: dict(gi=[], ni=[], trig=0, moves=0, steps=0) for n in names}
    per_layer = {n: {} for n in names}
    for l in layers:
        rows = {r: torch.cat([steps[i][r][l] for i in ids], 0) for r in range(R)}
        g = torch.Generator().manual_seed(1000 + l)
        pool = torch.stack([rows[r][torch.randint(0, rows[r].shape[0], (4096,), generator=g)] for r in range(R)])
        pls = {n: placement_of(c["p2l"][str(l)], G, R) for n, c in calibs.items()}
        pls["oracle"] = solve_placement(demand_histogram(pool, L, G), L, nlp)
        rr = [((s // nlp) * (G // R) + s % nlp) if s % nlp < G // R else -1 for s in range(R * nlp)]
        pls["rr"] = placement_of(rr, G, R)
        eval_pop = torch.zeros(G, dtype=torch.float64)
        for n in names:
            pl = pls[n]
            gi_l, ni_l = [], []
            for i in ids:
                S = min(steps[i][r].shape[1] for r in range(R))
                topk = torch.stack([steps[i][r][l, :S] for r in range(R)])
                load_g = torch.bincount(topk.reshape(-1).long(), minlength=G)
                if n == names[0]:
                    eval_pop += load_g.double()
                if a.swap and n != "rr":
                    new_pl, _rounds = decide_swaps(load_g, pl, L, nlp, a.C)
                    res[n]["steps"] += 1
                    if new_pl is not None:
                        res[n]["trig"] += 1
                        res[n]["moves"] += int((new_pl.p2l != pl.p2l).sum())
                        pl = new_pl
                phys, _tab = route_water_fill(topk, pl.p2l, pl.l2p, pl.lcnts, nlp, L, Cn, Cd)
                gl = torch.bincount((phys.reshape(-1).long() // nlp), minlength=R).double()
                nl = gl.view(R // L, L).sum(1)
                gi_l.append(float(gl.max() / gl.mean()))
                ni_l.append(float(nl.max() / nl.mean()))
            res[n]["gi"] += gi_l
            res[n]["ni"] += ni_l
            per_layer[n][l] = (st.mean(gi_l), st.mean(ni_l))
        corr = ""
        if hist_pop is not None:
            h, e = hist_pop[l], eval_pop
            corr = f" hist-eval popularity corr {float(np.corrcoef(h.numpy(), e.numpy())[0, 1]):.3f}"
        print(f"[fidelity] layer {l:2d}: " + " | ".join(f"{n} gpu {per_layer[n][l][0]:.3f} node {per_layer[n][l][1]:.3f}"
                                                        for n in names) + corr, flush=True)
    print("[fidelity] overall (mean over layers x steps; GPU / node max over mean, swap trigger rate, moves per step):")
    for n in names:
        r = res[n]
        trig = f"trigger {r['trig'] / r['steps']:.3f} moves/step {r['moves'] / r['steps']:.2f}" if r["steps"] else "no swap"
        print(f"  {n:10s} gpu {st.mean(r['gi']):.3f} (p90 {np.percentile(r['gi'], 90):.3f})  node {st.mean(r['ni']):.3f}  {trig}")


if __name__ == "__main__":
    main()
