"""OURS intra-node swap unit gates (CPU, no GPU/flux import).

Gates: plan determinism, structural validity of table transpositions
(conservation, node-distinctness, l2p sort), pair-max gain law, tau
gating, orbit convergence + monotone max-load, and the SUB-MS decision
microbenchmark (user requirement: the decision is timed in total_ms).

Run: python3 test/python/moe_ag_scatter/test_ours_swap.py
"""

import importlib.util
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.normpath(os.path.join(_HERE, "..", "..", "..",
                                     "python", "flux", "testing"))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_PKG, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


oswap = _load("ours_swap")
pv2 = _load("placement_v2")


def check_tables(p2l, l2p, lcnts, G, R, L, nlp):
    n_inst = int((p2l >= 0).sum())
    assert int(lcnts.long().sum()) == n_inst
    node_cnt = torch.zeros(G, R // L, dtype=torch.int64)
    for s in range(R * nlp):
        e = int(p2l[s])
        if e >= 0:
            node_cnt[e, (s // nlp) // L] += 1
    assert int(node_cnt.max()) <= 1, "node-distinctness broken by swap"
    for g in range(G):
        cols = l2p[g][l2p[g] >= 0]
        assert cols.numel() == int(lcnts[g])
        assert bool((cols[1:] > cols[:-1]).all()), "l2p sort broken"
        for c in cols.tolist():
            assert int(p2l[c]) == g, "p2l/l2p inconsistent"


def band_gates():
    """Paper §4.3 band trigger (2026-09-23 reconciliation): the capped
    orbit under bal_C (a) never pairs a node that is inside
    (1 + C) * node-mean, (b) ends with every node in band OR with no
    strictly-improving intra-node exchange left in the out-of-band nodes,
    (c) is monotone (per-node max never rises), (d) needs no more swaps
    than the tau=1 fixpoint on the same basis, (e) is deterministic.
    Also checks the single-round swap_plan gate."""
    import numpy as np
    for (NN, L, G, nlp_extra, seed, C) in [
        (4, 4, 128, 2, 11, 0.25), (4, 4, 384, 2, 12, 0.0625),
        (8, 4, 384, 2, 13, 0.25), (16, 4, 128, 2, 14, 0.5),
        (16, 4, 384, 2, 15, 0.25), (4, 4, 384, 2, 16, 1.0),
    ]:
        R = NN * L
        nlp = G // R + nlp_extra
        g = torch.Generator().manual_seed(seed)
        hist = (torch.rand(NN, G, generator=g) ** 3 * 3000).long()
        res = pv2.pv2_solve(hist, L, nlp)
        p2l, l2p, lcnts = res["p2l"], res["l2p"], res["lcnts"]
        # perturb the demand so the basis is out of band on some nodes
        load_g = (hist.sum(0).double()
                  * (1 + 2 * torch.rand(G, generator=g))).long()
        Lr0 = oswap.rank_loads(load_g, p2l, lcnts, R, nlp).numpy()
        out0, r0 = oswap.node_band(Lr0, L, C)
        # (a) single round: every pair lies in an out-of-band node
        sw1, _ = oswap.swap_plan(load_g, p2l, lcnts, L, nlp, 1, bal_C=C)
        for (rh, _sh, _eh, rl, _sl, _el) in sw1:
            assert out0[rh // L] and rh // L == rl // L
        # (e) determinism
        sw1b, _ = oswap.swap_plan(load_g, p2l.clone(), lcnts.clone(), L,
                                  nlp, 1, bal_C=C)
        assert sw1 == sw1b
        # capped orbit, band vs tau=1 fixpoint on the same basis
        cap = 10 ** 6
        pb, lb, nb = oswap.swap_orbit_capped(load_g, p2l, l2p, lcnts, L,
                                             nlp, cap, bal_C=C)
        pt, lt, nt = oswap.swap_orbit_capped(load_g, p2l, l2p, lcnts, L,
                                             nlp, cap)
        if nb:
            check_tables(pb, lb, lcnts, G, R, L, nlp)
        mv_b = sum(len(m) for m in oswap.net_moves(p2l, pb, L, nlp)) // 2
        mv_t = sum(len(m) for m in oswap.net_moves(p2l, pt, L, nlp)) // 2
        assert mv_b <= mv_t, (mv_b, mv_t)                      # (d)
        Lrb = oswap.rank_loads(load_g, pb, lcnts, R, nlp).numpy()
        outb, rb = oswap.node_band(Lrb, L, C)
        # (c) monotone per node; untouched in-band nodes are bitwise equal
        for u in range(NN):
            assert Lrb[u * L:(u + 1) * L].max() <= Lr0[u * L:(u + 1) * L].max()
            if not out0[u]:
                assert (pb[u * L * nlp:(u + 1) * L * nlp]
                        == p2l[u * L * nlp:(u + 1) * L * nlp]).all()
        # (b) terminal: in band, or no improving exchange left
        if outb.any():
            sw_end, _ = oswap.swap_plan(load_g, pb, lcnts, L, nlp, 1,
                                        bal_C=C)
            assert sw_end == [], "band orbit stopped with swaps left"
        # (f) C = 0 ("perfect balance" constraint) reproduces the legacy tau=1
        # capped orbit bitwise: the paper-figure arms ARE the band trigger at C=0
        p0, l0, n0 = oswap.swap_orbit_capped(load_g, p2l, l2p, lcnts, L, nlp, 8)
        pz, lz, nz = oswap.swap_orbit_capped(load_g, p2l, l2p, lcnts, L, nlp, 8,
                                             bal_C=0.0)
        assert torch.equal(p0, pz) and n0 == nz, "band C=0 != legacy tau=1 orbit"
        print(f"  band ok NN={NN} G={G} C={C}: out-of-band"
              f" {int(out0.sum())}/{NN} (max/mean {r0.max():.3f}) ->"
              f" {int(outb.sum())}/{NN} ({rb.max():.3f}) in {nb} rounds,"
              f" moves band {mv_b} vs tau=1 fixpoint {mv_t} ({nt} rounds)")


def main():
    torch.manual_seed(0)
    band_gates()
    for (NN, L, G, nlp_extra, seed, tau) in [
        (4, 4, 128, 2, 1, 64), (4, 4, 384, 2, 2, 128),
        (16, 4, 384, 2, 3, 128), (16, 4, 128, 2, 4, 64),
        (8, 4, 384, 2, 5, 256), (4, 4, 384, 2, 6, 10**9),  # tau=inf twin
    ]:
        R = NN * L
        nlp = G // R + nlp_extra
        g = torch.Generator().manual_seed(seed)
        hist = (torch.rand(NN, G, generator=g) ** 3 * 3000).long()
        res = pv2.pv2_solve(hist, L, nlp)
        p2l, l2p, lcnts = res["p2l"], res["l2p"], res["lcnts"]
        load_g = hist.sum(0)

        s1, Lr1 = oswap.swap_plan(load_g, p2l, lcnts, L, nlp, tau)
        s2, _ = oswap.swap_plan(load_g, p2l.clone(), lcnts.clone(),
                                L, nlp, tau)
        assert s1 == s2, "swap plan nondeterministic"
        if tau >= 10**9:
            assert s1 == [], "tau=inf twin must never swap"
        # each rank in at most one swap; swaps intra-node; both slots real
        seen = set()
        for (rh, sh, eh, rl, sl, el) in s1:
            assert rh // L == rl // L and rh != rl
            for r in (rh, rl):
                assert r not in seen
                seen.add(r)
            assert int(p2l[sh]) == eh and int(p2l[sl]) == el
            # gain law: pair max strictly drops by >= tau
            w_h = int(load_g[eh]) // max(int(lcnts[eh]), 1)
            w_l = int(load_g[el]) // max(int(lcnts[el]), 1)
            base = max(int(Lr1[rh]), int(Lr1[rl]))
            new = max(int(Lr1[rh]) - w_h + w_l, int(Lr1[rl]) - w_l + w_h)
            assert base - new >= tau
        if s1:
            p2l_n, l2p_n = oswap.apply_swaps(p2l, l2p, s1)
            check_tables(p2l_n, l2p_n, lcnts, G, R, L, nlp)
            Lr2 = oswap.rank_loads(load_g, p2l_n, lcnts, R, nlp)
            # per swapped node the max rank load must not increase
            for (rh, _sh, _eh, rl, _sl, _el) in s1:
                u = rh // L
                pre = int(Lr1[u * L:(u + 1) * L].max())
                post = int(Lr2[u * L:(u + 1) * L].max())
                assert post <= pre
        orbit = oswap.swap_orbit(load_g, p2l, l2p, lcnts, L, nlp, tau)
        for (p2l_o, l2p_o) in orbit:
            check_tables(p2l_o, l2p_o, lcnts, G, R, L, nlp)
        # fixed point: no swaps on the last orbit element
        end_p2l = orbit[-1][0] if orbit else p2l
        s_end, _ = oswap.swap_plan(load_g, end_p2l, lcnts, L, nlp, tau)
        assert s_end == [], "orbit did not converge"
        print(f"  ok NN={NN} G={G} nlp={nlp} tau={tau} swaps={len(s1)} "
              f"orbit={len(orbit)}")

    # ---- sub-ms decision microbench on real traces (user requirement) --
    GEN = ("/pscratch/sd/y/yufeid/workspace/andrewy/a2av_test_matrices/"
           "generated")
    plfast = _load("placelambda_fast")
    for name, mid, R, nlp in [
        ("K2 16n b8", "w64x4_trace-6aa437_b8_k8_id001", 64, 8),
        ("Qwen 16n b8", "w64x4_trace-7ecc68_b8_k8_id001", 64, 4),
        ("K2 4n b64", "w16x4_trace-041f16_b64_k8_id001", 16, 26),
    ]:
        path = f"{GEN}/{mid}.routing.txt"
        if not os.path.exists(path):
            print(f"  skip {name}: no trace")
            continue
        with open(path) as f:
            nt, k, G = (int(x) for x in f.readline().split())
            vals = [int(x) for x in f.read().split()]
        topk = torch.tensor(vals, dtype=torch.int64).reshape(R, nt // R, k)
        hist = plfast.demand_hist(topk, 4, G).long()
        res = pv2.pv2_solve(hist, 4, nlp)
        load_g = hist.sum(0)
        for _ in range(3):
            oswap.swap_plan(load_g, res["p2l"], res["lcnts"], 4, nlp, 512)
        ts = []
        for _ in range(30):
            t0 = time.perf_counter()
            sw, _ = oswap.swap_plan(load_g, res["p2l"], res["lcnts"],
                                    4, nlp, 512)
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        ap = 0.0
        if sw:
            t0 = time.perf_counter()
            oswap.apply_swaps(res["p2l"], res["l2p"], sw)
            ap = (time.perf_counter() - t0) * 1e3
        print(f"  {name}: decide med {ts[15]:.3f} ms p90 {ts[27]:.3f} "
              f"(swaps {len(sw)}, apply {ap:.3f} ms)")
        assert ts[15] < 1.0, f"{name}: decision exceeds the sub-ms budget"
    print("OURS swap unit gates ALL OK")


if __name__ == "__main__":
    sys.exit(main())
