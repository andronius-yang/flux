"""Offline quantifier for the minimal-move relay partition (handoff 42,
2026-09-23): on real K2 routing traces, place with pv2, route with the pv3
reference router (C = 1/4), derive the per-(source rank, remote node) union
counts U, and compare the intra-node rows moved by the LEGACY equal cut of
the source-ascending canonical stream against the WATER-FILL partition
(paper §4.2 eq. 3: Sum_k (V_k - cap_k)^+). CPU only, no flux import.

Run: python3 docs/handoff/42_minmove_offline.py
"""
import importlib.util
import os
import sys

import torch

_PKG = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", "python", "flux", "testing"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_PKG, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


pv2 = _load("placement_v2")
plfast = _load("placelambda_fast")
pv3r = _load("pv3_route")
GEN = "/pscratch/sd/y/yufeid/workspace/andrewy/a2av_test_matrices/generated"


def legacy_moved(V):
    L = len(V)
    tot = sum(V)
    c = [0]
    for v in V:
        c.append(c[-1] + v)
    b = [(tot // L) * k + min(k, tot % L) for k in range(L + 1)]
    keep = sum(max(0, min(b[k + 1], c[k + 1]) - max(b[k], c[k])) for k in range(L))
    return tot - keep


def minmove_moved(V):
    L = len(V)
    tot = sum(V)
    cap = [tot // L + (1 if k < tot % L else 0) for k in range(L)]
    return sum(max(0, V[k] - cap[k]) for k in range(L))


def main():
    for name, mid, R, nlp, L in [
        ("K2 4n b64", "w16x4_trace-041f16_b64_k8_id001", 16, 26, 4),
        ("K2 16n b8", "w64x4_trace-6aa437_b8_k8_id001", 64, 8, 4),
        ("Qwen 16n b8", "w64x4_trace-7ecc68_b8_k8_id001", 64, 4, 4),
    ]:
        path = f"{GEN}/{mid}.routing.txt"
        if not os.path.exists(path):
            print(f"skip {name}: no trace")
            continue
        with open(path) as f:
            nt, k, G = (int(x) for x in f.readline().split())
            vals = [int(x) for x in f.read().split()]
        topk = torch.tensor(vals, dtype=torch.int64).reshape(R, nt // R, k)
        NN = R // L
        hist = plfast.demand_hist(topk, L, G).long()
        res = pv2.pv2_solve(hist, L, nlp)
        phys, _ = pv3r.pv3_route(topk, res["p2l"], res["l2p"], res["lcnts"], nlp, L,
                                 C_num=1, C_den=4, return_tables=True)
        node = (phys.long() // nlp // L)                       # [R, S, K]
        U = torch.zeros(R, NN, dtype=torch.int64)
        for i in range(R):
            onehot = torch.zeros(node.shape[1], NN, dtype=torch.int64)
            onehot.scatter_(1, node[i], 1)                     # unique nodes per token
            U[i] = onehot.sum(0)
            U[i, i // L] = 0                                   # own node: not on the wire
        leg = mm = tot = 0
        worst = (0.0, None)
        for n in range(NN):
            for m in range(NN):
                if n == m:
                    continue
                V = [int(U[n * L + kk, m]) for kk in range(L)]
                a, b = legacy_moved(V), minmove_moved(V)
                leg += a
                mm += b
                tot += sum(V)
                if sum(V) and a - b > worst[0]:
                    worst = (a - b, (n, m, V, a, b))
        print(f"{name}: wire rows {tot}; intra-node moved rows legacy {leg}"
              f" ({100 * leg / max(tot, 1):.1f}% of wire) vs water-fill {mm}"
              f" ({100 * mm / max(tot, 1):.1f}%); saved {leg - mm}"
              f" ({100 * (leg - mm) / max(leg, 1):.1f}% of legacy movement);"
              f" worst pair n->m,V,legacy,minmove = {worst[1]}")


if __name__ == "__main__":
    main()
