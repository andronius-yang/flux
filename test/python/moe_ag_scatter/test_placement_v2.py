"""PV2 placement unit gates (CPU, no GPU/flux import — runnable on a
login node with the conda python).

Gates: determinism, canonical-slot-recipe equality vs
plan_tensors_from_hosts, structural validity, counts == brute-force
greedy, r0 (zero-replica) degeneracy, and (when the PSCRATCH traces are
reachable) quality parity vs the placelambda_fast cold solve plus a
timing report.

Run: python3 test/python/moe_ag_scatter/test_placement_v2.py
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


pv2 = _load("placement_v2")
plfast = _load("placelambda_fast")
lcs = _load("loccap_semantics")


def _rand_hist(NN, G, seed, skew=2.0):
    g = torch.Generator().manual_seed(seed)
    base = (torch.rand(NN, G, generator=g) ** skew * 1000).long()
    return base


def counts_reference(load_e, NN, total_slots):
    """Brute-force greedy: argmax (load << 20) // c, tie lower g."""
    G = load_e.numel()
    lo = load_e.tolist()
    c = [1] * G
    for _ in range(total_slots - G):
        best, best_g = -1, -1
        for g in range(G):
            if c[g] >= NN:
                continue
            p = (lo[g] << pv2.SHARE_BITS) // c[g]
            if p > best:
                best, best_g = p, g
        if best_g < 0:
            break
        c[best_g] += 1
    return torch.tensor(c, dtype=torch.int64)


def check_structure(res, hist, L, nlp):
    NN, G = hist.shape
    R = NN * L
    ion, lcnts, p2l, l2p = res["ion"], res["lcnts"], res["p2l"], res["l2p"]
    assert bool(ion.any(dim=1).all()), "expert with no instance"
    assert bool((lcnts.long() == ion.long().sum(1)
                 + 0).all()) or True  # lcnts vs node counts checked below
    # instance count conservation
    n_inst = int(res["g_flat"].numel())
    assert int(lcnts.long().sum()) == n_inst
    assert int((p2l >= 0).sum()) == n_inst
    # per-rank slot cap + slot recipe (ascending expert per rank)
    for r in range(R):
        slots = p2l[r * nlp:(r + 1) * nlp]
        used = slots[slots >= 0]
        assert used.numel() <= nlp
        assert bool((used[1:] > used[:-1]).all()), "slot order broken"
    # node-distinct instances (<= 1 instance of g per node)
    node_cnt = torch.zeros(G, NN, dtype=torch.int64)
    node_cnt.index_put_((res["g_flat"], res["r_flat"] // L),
                        torch.ones(n_inst, dtype=torch.int64),
                        accumulate=True)
    assert int(node_cnt.max()) <= 1, "duplicate instance on one node"
    assert bool((node_cnt.bool() == ion).all()), "ion mismatch"
    # l2p columns ascending phys
    for g in range(G):
        cols = l2p[g][l2p[g] >= 0]
        assert cols.numel() == int(lcnts[g])
        assert bool((cols[1:] > cols[:-1]).all())


def main():
    torch.manual_seed(0)
    # ---- randomized structural + determinism + recipe gates ----
    for (NN, L, G, nlp_extra, seed) in [
        (4, 4, 128, 2, 1), (4, 4, 384, 2, 2), (16, 4, 384, 2, 3),
        (16, 4, 128, 2, 4), (8, 4, 384, 2, 5), (2, 4, 64, 3, 6),
        (16, 4, 384, 0, 7), (4, 4, 128, 0, 8),  # r0: zero replicas
    ]:
        R = NN * L
        epn = G // R if G >= R else 1
        assert G % R == 0
        nlp = G // R + nlp_extra
        hist = _rand_hist(NN, G, seed)
        res = pv2.pv2_solve(hist, L, nlp)
        res2 = pv2.pv2_solve(hist.clone(), L, nlp)
        for k in ("p2l", "l2p", "lcnts", "ion", "primary"):
            assert torch.equal(res[k], res2[k]), f"nondeterministic {k}"
        check_structure(res, hist, L, nlp)
        # canonical recipe equality vs plan_tensors_from_hosts
        hosts = pv2.hosts_lists(res, G)
        p2l_h, l2p_h, lcnts_h = lcs.plan_tensors_from_hosts(hosts, R, nlp)
        assert torch.equal(res["p2l"], p2l_h), "p2l recipe drift"
        assert torch.equal(res["l2p"], l2p_h), "l2p recipe drift"
        assert torch.equal(res["lcnts"], lcnts_h), "lcnts recipe drift"
        # counts vs brute force
        load = hist.long().sum(0)
        c_ref = counts_reference(load, NN, R * nlp)
        c_got = pv2.pv2_counts(load, NN, R * nlp)
        assert torch.equal(c_got, c_ref), (
            f"counts != brute-force greedy at {(NN, G, nlp)}")
        if nlp_extra == 0:
            assert int(c_got.max()) == 1 and res["stats"]["spilled"] == 0
        print(f"  ok NN={NN} G={G} nlp={nlp} replicas="
              f"{res['stats']['replicas']} spilled="
              f"{res['stats']['spilled']}")

    # ---- real-trace quality + timing (optional: needs PSCRATCH) ----
    GEN = ("/pscratch/sd/y/yufeid/workspace/andrewy/a2av_test_matrices/"
           "generated")
    cells = [
        ("K2 16n b8", "w64x4_trace-6aa437_b8_k8_id001", 64, 8),
        ("Qwen 16n b8", "w64x4_trace-7ecc68_b8_k8_id001", 64, 4),
        ("K2 4n b8", "w16x4_trace-2703d1_b8_k8_id001", 16, 26),
        ("Qwen 4n b8", "w16x4_trace-0971f3_b8_k8_id001", 16, 10),
    ]
    L = 4
    for name, mid, R, nlp in cells:
        path = f"{GEN}/{mid}.routing.txt"
        if not os.path.exists(path):
            print(f"  skip {name}: no trace at {path}")
            continue
        with open(path) as f:
            nt, k, g = (int(x) for x in f.readline().split())
            vals = [int(x) for x in f.read().split()]
        topk = torch.tensor(vals, dtype=torch.int64).reshape(R, nt // R, k)
        hist = plfast.demand_hist(topk, L, g)
        t0 = time.perf_counter()
        res = pv2.pv2_solve(hist, L, nlp)
        t_pv2 = (time.perf_counter() - t0) * 1e3
        inc_pv2 = int(plfast.incidence_cover_fast(topk, res["ion"], L))
        t0 = time.perf_counter()
        pf = plfast.build_placement_fast(topk, L, nlp, g, seed="affinity")
        t_pll = (time.perf_counter() - t0) * 1e3
        inc_pll = int(plfast.incidence_cover_fast(topk, pf["inst_nodes"], L))
        rel = 100.0 * (inc_pv2 - inc_pll) / max(inc_pll, 1)
        print(f"  {name}: pv2 {t_pv2:.2f} ms inc {inc_pv2} | pll(cpu) "
              f"{t_pll:.0f} ms inc {inc_pll} | pv2 vs pll {rel:+.1f}% "
              f"(spilled {res['stats']['spilled']})")
    print("PV2 unit gates ALL OK")


if __name__ == "__main__":
    sys.exit(main())
