"""Per-tier fill of every expert copy for the plotted main-perf cells,
computed OFFLINE with the deterministic torch reference router
(loccap_route_sl) on the cells' real routing files and the pv2 placement
solved from their real oracle window — the same inputs the plotted arm
ran (eps 1/16, 2 spare slots per GPU). CPU only, torch-only imports.

Caveats: the reference resolves ticket races by token index (the kernel
uses relaxed atomics), and forced rows here are uncapped (f_cap None ->
fallback host) whereas the cells ran with the derived f_cap. Tier
tables are identical by contract; per-row splits can differ slightly.

Outputs (next to this file): tier_fill_cells.csv (one row per expert
copy), tier_fill_summary.md.
Usage: python figs/methodology/placement_routing/tier_fill_offline.py
"""
import csv
import importlib.util
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TESTING = os.path.join(ROOT, "python", "flux", "testing")
STAGING = "/pscratch/sd/y/yufeid/workspace/andrewy/sweep_data"
EPS = 0.0625
R_RED = 2
L = 4
SHAPE = {"K2": (384, 8), "Qwen": (128, 8)}


def load_by_path(name, path, patch=None):
    src = open(path).read()
    if patch:
        old, new = patch
        assert old in src, "patch anchor missing"
        src = src.replace(old, new)
    mod = type(sys)(name)
    mod.__file__ = path
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


pv2 = load_by_path("pv2_off", os.path.join(TESTING, "placement_v2.py"))
ANCHOR = ("    forced_budget_overflow = 0\n"
          "    rem = (phys_flat == UNASSIGNED).nonzero(as_tuple=True)[0]\n")
plg = load_by_path("plg_off", os.path.join(TESTING, "placelambda_gpu.py"),
                   patch=(ANCHOR, ANCHOR + "    FORCED_CAPTURE.append(rem.clone())\n"))
plg.FORCED_CAPTURE = []


def load_routing(path):
    with open(path) as f:
        n, k, g = (int(x) for x in f.readline().split())
        vals = torch.tensor([int(x) for x in f.read().split()], dtype=torch.int64)
    return vals.view(n, k), g, k


def cell_inputs(capsule, cell_id):
    rec = os.path.join(STAGING, capsule, "cells", cell_id, "records", "rank_000.jsonl")
    info = {}
    for line in open(rec):
        d = json.loads(line)
        if d.get("type") == "cell_info":
            info.update(d)
    ofile = info["epic_pll_oracle_file"]
    return ofile, ofile.replace(".oracle_routing.txt", ".routing.txt"), info


def analyze(nodes, model, budget, capsule, cell_id):
    G, K = SHAPE[model]
    W = nodes * L
    nlp = G // W + R_RED
    ofile, rfile, info = cell_inputs(capsule, cell_id)
    orc, g1, k1 = load_routing(ofile)
    bat, g2, k2 = load_routing(rfile)
    assert g1 == g2 == G and k1 == k2 == K
    S = bat.shape[0] // W
    tk = bat.view(W, S, K)
    node_of_tok = (torch.arange(W).repeat_interleave(orc.shape[0] // W)) // L
    hist = torch.zeros(nodes, G, dtype=torch.int64)
    hist.index_put_((node_of_tok.repeat_interleave(K), orc.reshape(-1)),
                    torch.ones(orc.numel(), dtype=torch.int64), accumulate=True)
    res = pv2.pv2_solve(hist, L, nlp)
    plg.FORCED_CAPTURE.clear()
    phys, st = plg.loccap_route_sl(tk, res["p2l"], res["l2p"], res["lcnts"],
                                   nlp, L, EPS, return_tables=True)
    forced_idx = plg.FORCED_CAPTURE[0] if plg.FORCED_CAPTURE else torch.zeros(0, dtype=torch.int64)
    phys = phys.long().reshape(-1)
    src = torch.arange(W).repeat_interleave(S * K)
    dst = phys // nlp
    tier = torch.full_like(phys, 3)
    tier[dst == src] = 1
    tier[(dst != src) & (dst // L == src // L)] = 2
    tier[forced_idx] = 4
    same_node_t3 = int(((tier == 3) & (dst // L == src // L)).sum())
    cap = st["cap"]
    p2l = res["p2l"].long()
    load = torch.bincount(phys, minlength=W * nlp)
    gpu_load = load.view(W, nlp).sum(1)
    rows = []
    for slot in range(W * nlp):
        g = int(p2l[slot])
        if g < 0:
            continue
        m = phys == slot
        t = torch.bincount(tier[m], minlength=5)
        rows.append(dict(nodes=nodes, model=model, budget_mib=budget, cell_id=cell_id,
                         expert=g, copies=int(res["lcnts"][g]), rank=slot // nlp,
                         node=slot // nlp // L, rows=int(m.sum()),
                         t1_self=int(t[1]), t2_node=int(t[2]), t3_remote=int(t[3]), t4_forced=int(t[4]),
                         gpu_load=int(gpu_load[slot // nlp]), cap=cap))
    tot = torch.bincount(tier, minlength=5)[1:].tolist()
    # ---- per-token remote-node incidence: actual cover vs even split ----
    tok = torch.arange(W * S).repeat_interleave(K)
    home = tok // S // L
    dnode = dst // L
    remote = dnode != home
    n_remote_rows = torch.bincount(tok[remote], minlength=W * S)
    keyn = tok * nodes + dnode
    act_nodes = torch.bincount(torch.unique(keyn[remote]) // nodes, minlength=W * S)
    ion = res["ion"]                                    # [G, NN]
    gen = torch.Generator().manual_seed(0)
    g_all = tk.reshape(-1)
    cand = ion[g_all].clone()                           # [rows, NN]
    cand[torch.arange(cand.shape[0]), home] = False     # home copies never remote
    has_home = ion[g_all, home]
    rnd = torch.rand(cand.shape, generator=gen) * cand
    pick = rnd.argmax(1)
    ev_remote = ~has_home
    keye = tok * nodes + pick
    ev_nodes = torch.bincount(torch.unique(keye[ev_remote]) // nodes, minlength=W * S)
    tokstat = dict(remote_rows_per_token=float(n_remote_rows.float().mean()),
                   remote_nodes_per_token_cover=float(act_nodes.float().mean()),
                   remote_nodes_per_token_evensplit=float(ev_nodes.float().mean()),
                   tokens_all_local_pct=100 * float((n_remote_rows == 0).float().mean()))
    summ = dict(nodes=nodes, model=model, budget_mib=budget, S=S, cap=cap, nlp=nlp,
                rows_total=int(phys.numel()), t1=tot[0], t2=tot[1], t3=tot[2], t4=tot[3],
                t3_same_node=same_node_t3, gpu_load_max=int(load.view(W, nlp).sum(1).max()),
                over_cap_rows=st["over_cap_rows"], copies=int(res["lcnts"].sum()),
                drift_ppm=info.get("epic_pll_oracle_drift_ppm"), **tokstat)
    return rows, summ


def main():
    win = [r for r in csv.DictReader(open(os.path.join(HERE, "main_perf_winners.csv")))
           if r["plotted"] == "1" and "pv2" in r["winner_arm"]]
    src = list(csv.DictReader(open(os.path.join(HERE, "..", "..", "main_perf", "figure_src.csv"))))
    all_rows, summs = [], []
    for w in win:
        s = [r for r in src if (r["nodes"], r["model"], r["budget_mib"]) ==
             (w["nodes"], w["model"], w["budget_mib"]) and r["row_id"] == w["winner_row_id"]][0]
        rows, summ = analyze(int(w["nodes"]), w["model"], int(w["budget_mib"]), s["capsule"], s["cell_id"])
        all_rows += rows
        summs.append(summ)
        print(summ, flush=True)
    with open(os.path.join(HERE, "tier_fill_cells.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wr.writeheader()
        wr.writerows(all_rows)
    with open(os.path.join(HERE, "tier_fill_summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summs[0].keys()))
        wr.writeheader()
        wr.writerows(summs)


if __name__ == "__main__":
    main()
