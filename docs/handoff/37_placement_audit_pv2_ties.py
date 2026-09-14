"""Replay the PV2 placement (python/flux/testing/placement_v2.py, commit
f11b5ac, the only version ever) on the EXACT oracle-window routing files
the main-perf OURS cells (ours_l01_s1_pv2_r2) solved on, and record for
every instance placement which key decided the node:

  affinity  - a unique node had the highest hist[node, expert]
  load      - >=2 nodes tied on affinity; lowest accumulated share won
  nodeid    - tied on affinity AND load; lowest node id won
  backfill  - leftover slot spent by the vectorized backfill (no affinity
              key at all: highest-share non-hosting expert per node)

Also computes two counterfactual node assignments on the same counts:
  aff_only   - affinity desc, node id asc (no load tie-break)
  load_first - accumulated share asc, affinity desc, node id asc
and their predicted remote rows (pv2_remote_rows) vs the real placement.

Cross-checks against the rank-0 stderr of each cell ("setup audit OK:
E_virt .. placement basis prev_batch drift .. ppm"): E_virt must equal
G + replicas and the drift must match.
"""
import csv
import glob
import importlib.util
import json
import os
import re
import sys

import torch


ROOT = "/global/u1/y/yufeid/workspace/changchen/andrewy/flux"
PS = os.environ["PSCRATCH"] + "/workspace/andrewy"
GEN = PS + "/a2av_test_matrices/generated"
SWEEP = PS + "/sweep_data"
OUT = PS + "/placement_audit/out"
L = 4  # ranks per node (perlmutter)

spec = importlib.util.spec_from_file_location(
    "pv2", ROOT + "/python/flux/testing/placement_v2.py")
pv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pv2)

CAPS = [
    "20260830-014319_perlmutter_bb7116fd", "20260829-145517_perlmutter_511b512c",
    "20260829-153207_perlmutter_df44245f", "20260829-143523_perlmutter_a147cb08",
    "20260829-153031_perlmutter_007a3f3b", "20260829-145836_perlmutter_1f6e0f92",
    "20260829-153941_perlmutter_a469f6ef", "20260829-144007_perlmutter_52fa3163",
    "20260829-153816_perlmutter_05b96d0b", "20260830-183022_perlmutter_bd8b7b04",
    "20260829-025435_perlmutter_091ef4bb", "20260829-025542_perlmutter_480051f9",
    "20260830-180110_perlmutter_e216d851",
]
FIG_BUDGETS = {1, 2, 4, 16, 64}


def load_routing(path, G, K):
    with open(path) as f:
        n, k, g = (int(x) for x in f.readline().split())
        assert k == K and g == G
        vals = [int(x) for x in f.read().split()]
    return torch.tensor(vals, dtype=torch.int64).view(n, K)


def demand_hist(topk_all, L, G):
    R, S, K = topk_all.shape
    NN = R // L
    ent_g = topk_all.reshape(-1)
    ent_tok = torch.arange(R * S).repeat_interleave(K)
    ent_node = ent_tok // (L * S)
    h = torch.zeros(NN * G, dtype=torch.int64)
    h.index_add_(0, ent_node * G + ent_g, torch.ones_like(ent_g))
    return h.view(NN, G)


def drift_ppm(now, ref):
    tot = int(ref.sum())
    return int((now - ref).abs().sum()) * 1_000_000 // (2 * tot)


def place_instrumented(hist, c, spn, rows, key, mode="real"):
    """Exact copy of pv2_place's sequential greedy with per-instance
    records. mode: real | aff_only | load_first."""
    NN, G = hist.shape
    load = hist.sum(0)
    share = (load << pv2.SHARE_BITS) // c
    order = torch.argsort(share * G + (G - 1 - torch.arange(G)),
                          descending=True, stable=True).tolist()
    aff_l = hist.t().tolist()
    c_l = c.tolist()
    share_l = share.tolist()
    free = [spn] * NN
    loadv = [0] * NN
    ion = torch.zeros(G, NN, dtype=torch.bool)
    primary = [-1] * G
    spilled = 0
    for g in order:
        affs = aff_l[g]
        sg = share_l[g]
        hosted = []
        for i in range(c_l[g]):
            cands = [u for u in range(NN) if free[u] > 0 and u not in hosted]
            if not cands:
                spilled += c_l[g] - i
                break
            if mode == "real":
                # identical to pv2_place's strict-compare loop
                bu, ba, bl = -1, -1, 0
                for u in cands:
                    a = affs[u]
                    if a > ba or (a == ba and (bu < 0 or loadv[u] < bl)):
                        bu, ba, bl = u, a, loadv[u]
            elif mode == "aff_only":
                bu = min(cands, key=lambda u: (-affs[u], u))
            elif mode == "load_first":
                bu = min(cands, key=lambda u: (loadv[u], -affs[u], u))
            if rows is not None:
                amax = max(affs[u] for u in cands)
                tied = [u for u in cands if affs[u] == amax]
                second = max([affs[u] for u in cands if affs[u] < amax],
                             default=-1)
                if len(tied) == 1:
                    dec = "affinity"
                    ltied = tied
                else:
                    lmin = min(loadv[u] for u in tied)
                    ltied = [u for u in tied if loadv[u] == lmin]
                    dec = "load" if len(ltied) == 1 else "nodeid"
                assert bu == ltied[0], (g, i, bu, ltied)
                rows.append(dict(
                    key, g=g, inst=i, is_primary=int(i == 0), c_g=c_l[g],
                    share=sg >> pv2.SHARE_BITS, ncand=len(cands),
                    amax=amax, second=second, n_aff_tied=len(tied),
                    n_load_tied=len(ltied), decided_by=dec, node=bu,
                    lowest_id_among_aff_tied=int(bu == min(tied)),
                    load_tiebreak_changed_node=int(dec != "affinity"
                                                   and bu != min(tied)),
                ))
            hosted.append(bu)
            free[bu] -= 1
            loadv[bu] += sg
        for u in hosted:
            ion[g, u] = True
        primary[g] = hosted[0] if hosted else -1
    n_greedy = int(ion.sum())
    # backfill (exact copy)
    if sum(free) > 0:
        node_free = torch.tensor(free)
        smax = int(share.clamp(max=1 << 32).max()) + 2
        share_c = share.clamp(max=1 << 32)
        cand = (~ion) & (node_free > 0).unsqueeze(0)
        gc_, uc_ = cand.nonzero(as_tuple=True)
        if gc_.numel():
            okey = (uc_ * smax + (smax - 1 - share_c[gc_])) * G + gc_
            o = torch.argsort(okey, stable=True)
            u_s = uc_[o]
            pre = pv2._seg_prefix(torch.ones_like(u_s), u_s)
            fit = pre < node_free[u_s]
            ion[gc_[o[fit]], u_s[fit]] = True
    n_backfill = int(ion.sum()) - n_greedy
    return ion, torch.tensor(primary), spilled, n_backfill, loadv


def find_log(cap, cell):
    pats = glob.glob(f"{SWEEP}/{cap}/cells/{cell}/torchrun/none_*/attempt_*/*/stdout.log")
    for p in sorted(pats):
        txt = open(p, errors="replace").read()
        m = re.search(r"setup audit OK: E_virt (\d+) .*?placement basis (\w+) drift (\d+) ppm", txt)
        if m:
            return int(m.group(1)), m.group(2), int(m.group(3))
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    seen = {}
    cells = []
    for cap in CAPS:
        txt = open(f"{ROOT}/sweeps/results/runs/{cap}/spec.yaml").read()
        fam = re.search(r'"(trace:[^"]+)"', txt).group(1)
        model = re.search(r"model=([^;]+)", fam).group(1)
        assert "pools=livecodebench/execution" in fam, fam
        for r in csv.DictReader(open(f"{ROOT}/sweeps/results/runs/{cap}/cells.csv")):
            if r["variant"] != "ours_l01_s1_pv2_r2" or r["status"] != "ok":
                continue
            mid = r["matrix_id"]
            W = int(re.match(r"w(\d+)x4_", mid).group(1))
            b = int(r["budget_mib"])
            if mid in seen:
                continue
            seen[mid] = 1
            cells.append(dict(capsule=cap, cell_id=r["cell_id"], matrix_id=mid,
                              model=model, nodes=W // L, budget_mib=b,
                              in_figure=int(b in FIG_BUDGETS), G=int(r["G"]),
                              topk=int(r["topk"]), fam=fam))
    cells.sort(key=lambda c: (c["model"], c["nodes"], c["budget_mib"]))
    inst_rows, summ = [], []
    for c in cells:
        G, K, W, NN = c["G"], c["topk"], c["nodes"] * L, c["nodes"]
        orc = load_routing(f"{GEN}/{c['matrix_id']}.oracle_routing.txt", G, K)
        evl = load_routing(f"{GEN}/{c['matrix_id']}.routing.txt", G, K)
        h_or = demand_hist(orc.view(W, -1, K), L, G)
        h_ev = demand_hist(evl.view(W, -1, K), L, G)
        nlp = -(-G // W) + 2  # ceil(G/W) + redundant_per_rank 2
        # verify nlp against the log's E_virt = W * nlp
        R = W
        cnt = pv2.pv2_counts(h_or.sum(0), NN, R * nlp)
        key = dict(model=c["model"], nodes=NN, budget_mib=c["budget_mib"],
                   matrix_id=c["matrix_id"])
        rows = []
        ion, prim, spilled, nbf, loadv = place_instrumented(h_or, cnt, L * nlp, rows, key)
        ref_ion, ref_prim, ref_sp = pv2.pv2_place(h_or, cnt, L * nlp)
        assert torch.equal(ion, ref_ion) and torch.equal(prim, ref_prim) and spilled == ref_sp
        ref = pv2.pv2_solve(h_or, L, nlp)
        assert torch.equal(ref["ion"], ion)
        ion_a, *_ = place_instrumented(h_or, cnt, L * nlp, None, key, "aff_only")
        ion_l, *_, loadv_l = place_instrumented(h_or, cnt, L * nlp, None, key, "load_first")
        inst_rows += rows
        n = len(rows)
        dec = {k: sum(r["decided_by"] == k for r in rows) for k in ("affinity", "load", "nodeid")}
        rep = [r for r in rows if not r["is_primary"]]
        decr = {k: sum(r["decided_by"] == k for r in rep) for k in ("affinity", "load", "nodeid")}
        zero_aff = sum(r["amax"] == 0 for r in rows)
        changed = sum(r["load_tiebreak_changed_node"] for r in rows)
        log = find_log(c["capsule"], c["cell_id"])
        e_virt = W * (nlp + 1)  # driver E_virt = W*gpe, gpe = nlp + 1 (verified against logs)
        assert R * nlp == G + ref["stats"]["replicas"], "slots not all spent"
        d = drift_ppm(h_ev, h_or)
        tot = int(h_or.sum())
        s = dict(**key, in_figure=c["in_figure"], capsule=c["capsule"][:15],
                 slots=R * nlp, replicas=ref["stats"]["replicas"],
                 c_max=ref["stats"]["c_max"], spilled=spilled,
                 greedy_instances=n, primaries=sum(r["is_primary"] for r in rows),
                 replica_instances=len(rep), backfill_instances=nbf,
                 dec_affinity=dec["affinity"], dec_load=dec["load"], dec_nodeid=dec["nodeid"],
                 rep_dec_affinity=decr["affinity"], rep_dec_load=decr["load"], rep_dec_nodeid=decr["nodeid"],
                 zero_affinity_instances=zero_aff,
                 load_tiebreak_changed_node=changed,
                 ion_differs_from_aff_only=int(not torch.equal(ion, ion_a)),
                 ion_differs_from_load_first=int(not torch.equal(ion, ion_l)),
                 remote_rows_real=pv2.pv2_remote_rows(h_or, ion),
                 remote_rows_aff_only=pv2.pv2_remote_rows(h_or, ion_a),
                 remote_rows_load_first=pv2.pv2_remote_rows(h_or, ion_l),
                 oracle_rows=tot,
                 node_share_max_over_mean_real=round(max(loadv) / (sum(loadv) / NN), 4),
                 node_share_max_over_mean_load_first=round(max(loadv_l) / (sum(loadv_l) / NN), 4),
                 log_E_virt=log[0] if log else "", log_basis=log[1] if log else "",
                 log_drift_ppm=log[2] if log else "", my_E_virt=e_virt, my_drift_ppm=d,
                 log_match=int(bool(log) and log[0] == e_virt and log[2] == d))
        summ.append(s)
        print(f"{s['model']:<9} {NN:>2}n b{s['budget_mib']:<3} inst {n:>4} aff {dec['affinity']:>4}"
              f" load {dec['load']:>3} id {dec['nodeid']:>3} bf {nbf:>3} zero {zero_aff:>3}"
              f" rr real/aff/load {s['remote_rows_real']}/{s['remote_rows_aff_only']}/{s['remote_rows_load_first']}"
              f" / {tot}  E_virt {e_virt} log {log} match {s['log_match']}", flush=True)
    with open(f"{OUT}/instances.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inst_rows[0].keys()))
        w.writeheader(); w.writerows(inst_rows)
    with open(f"{OUT}/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
