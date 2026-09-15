"""Single-GPU parity + latency gate for the PV3 kernel (_pv3_ext.cu) vs
the torch reference (pv3_route.pv3_route). Run on a node with one GPU
(the extension builds/loads from $PSCRATCH, no flux import needed):

  python3 test/python/moe_ag_scatter/test_pv3_kernel.py [--real]

Checks (the relaxed-ticket contract): per-(source, expert, destination
rank) COUNTS equal the reference tables EXACTLY (bitwise counts; the
kernel is free only in which token takes which slot), conservation,
constraint 2 on the kernel's product, stats identically zero. Latency:
warm median of the single-rank kernel call (the per-iteration production
cost) at the paper shapes 4n/8n/16n K2+Qwen and K3 4n/32n.
"""
import importlib.util
import os
import statistics
import sys
import time

import torch

_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "..", "python", "flux", "testing")


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_BASE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PV3 = _load("pv3_route")
LC = _load("loccap_semantics")
EXT = _load("pv3_ext")
PV2 = _load("placement_v2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pv3_route import rand_hosts, rand_topk, slots_for  # noqa: E402


def kernel_all_ranks(ext, topk, l2p, lcnts, R, nlp, L, C):
    S, K = topk.shape[1], topk.shape[2]
    G = int(lcnts.numel())
    dev = "cuda"
    d = torch.zeros(R, G, dtype=torch.int32, device=dev)
    for r in range(R):
        d[r] = torch.bincount(topk[r].reshape(-1), minlength=G).int()
    l2p_d = l2p.int().to(dev).contiguous()
    lc_d = lcnts.int().to(dev).contiguous()
    tk_d = topk.int().to(dev)
    ws = torch.empty(ext.workspace_ints(G, R), dtype=torch.int32, device=dev)
    phys = torch.empty(R, S, K, dtype=torch.int32, device=dev)
    for r in range(R):
        ph, st = ext.route_pv3(tk_d[r].contiguous(), d, l2p_d, lc_d, r,
                               nlp, L, C[0], C[1], ws)
        assert int(st.sum()) == 0, ("kernel stats nonzero", st)
        phys[r] = ph
    return phys.cpu(), d, tk_d, l2p_d, lc_d, ws


def counts_gre(phys, topk, R, nlp, G):
    src = torch.arange(R).view(R, 1, 1).expand_as(phys).reshape(-1)
    g = topk.reshape(-1)
    dst = phys.long().reshape(-1) // nlp
    return torch.bincount(src * G * R + g * R + dst,
                          minlength=R * G * R).view(R, G, R)


def run_case(tag, ext, topk, hosts, R, L, nlp, C, iters=50):
    S, K = topk.shape[1], topk.shape[2]
    G = len(hosts)
    p2l, l2p, lcnts = LC.plan_tensors_from_hosts(hosts, R, nlp)
    phys_k, d, tk_d, l2p_d, lc_d, ws = kernel_all_ranks(
        ext, topk, l2p, lcnts, R, nlp, L, C)
    phys_r, st_r = PV3.pv3_route(topk, p2l, l2p, lcnts, nlp, L, C[0], C[1])
    assert bool(p2l.long()[phys_k.long()].eq(topk).all()), (tag, "conserv")
    ck = counts_gre(phys_k, topk, R, nlp, G)
    cr = counts_gre(phys_r.long(), topk, R, nlp, G)
    assert torch.equal(ck, cr), (tag, "kernel counts != reference counts",
                                 int((ck != cr).sum()))
    ipr = PV3.instance_phys_of_rank(l2p, lcnts, nlp, R)
    st_k = PV3.pv3_check(phys_k.long(), topk, ipr, nlp, L, C[0], C[1])
    assert st_k["c2_over_rows"] == 0 and st_k["c2_under_rows"] == 0, (
        tag, st_k)
    # latency: own-rank call, warm median (production per-iteration cost)
    r = 0
    tk0 = tk_d[r].contiguous()
    for _ in range(5):
        ext.route_pv3(tk0, d, l2p_d, lc_d, r, nlp, L, C[0], C[1], ws)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ext.route_pv3(tk0, d, l2p_d, lc_d, r, nlp, L, C[0], C[1], ws)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    # CUDA-graph replay cost (what the plan lane pays under PLAN_GRAPH)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ext.route_pv3(tk0, d, l2p_d, lc_d, r, nlp, L, C[0], C[1], ws)
    g.replay()
    torch.cuda.synchronize()
    tg = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        tg.append((time.perf_counter() - t0) * 1e3)
    inc, rem = PV3.incidence_remote(phys_k.long(), nlp, L)
    print(f"{tag:<28} R={R:>3} S={S:>5} K={K:>2} G={G:>3} nlp={nlp:>2} "
          f"kernel med {statistics.median(ts):.3f} ms (graph "
          f"{statistics.median(tg):.3f} ms) | ratio "
          f"[{st_k['replica_ratio_min']:.3f},{st_k['replica_ratio_max']:.3f}]"
          f" gpu [{st_k['gpu_ratio_min']:.3f},{st_k['gpu_ratio_max']:.3f}]"
          f" remote {rem} incid {inc}", flush=True)
    return statistics.median(ts)


def main():
    ext = EXT.load_ext()
    L = 4
    C = (1, 16)
    # parity on random placements (small, many)
    for NN, S, K, G in [(2, 16, 4, 32), (4, 48, 8, 96), (8, 32, 8, 128)]:
        R = NN * L
        nlp = slots_for(G, R, NN)
        for gen in range(3):
            hosts = rand_hosts(G, R, nlp, NN, L, gen)
            topk = rand_topk(R, S, K, G, gen + 100, 3.0)
            run_case(f"parity NN{NN} gen{gen}", ext, topk, hosts, R, L,
                     nlp, C, iters=10)
    print("parity OK")
    # latency at paper shapes: (NN, S per rank, K, G) — S from the byte
    # budgets used in the figures (b1/b4/b16 rows: K2 72/296/1168, Qwen
    # 128/512/2048), r2 slots
    shapes = [
        ("K2 4n b1", 4, 72, 8, 384), ("K2 4n b16", 4, 1168, 8, 384),
        ("Qwen 4n b16", 4, 2048, 8, 128),
        ("K2 8n b16", 8, 1168, 8, 384), ("K2 16n b16", 16, 1168, 8, 384),
        ("Qwen 16n b16", 16, 2048, 8, 128),
        ("K3 4n b16", 4, 1168, 16, 896), ("K3 16n b16", 16, 1168, 16, 896),
        ("K3 32n b16", 32, 1168, 16, 896), ("K3 32n b64", 32, 4672, 16, 896),
    ]
    for tag, NN, S, K, G in shapes:
        R = NN * L
        nlp = G // R + 2
        # a REAL pv2 placement (r2 slots) solved from a skewed synthetic
        # window, routed on a second draw of the same law (drift)
        topk_o = rand_topk(R, S, K, G, 7, 3.0)
        topk = rand_topk(R, S, K, G, 8, 3.0)
        hist = torch.zeros(NN, G, dtype=torch.int64)
        hist.index_put_(((torch.arange(R) // L).repeat_interleave(S * K),
                         topk_o.reshape(-1)),
                        torch.ones(R * S * K, dtype=torch.int64),
                        accumulate=True)
        res = PV2.pv2_solve(hist, L, nlp)
        hosts = PV2.hosts_lists(res, G)
        run_case(tag, ext, topk, hosts, R, L, nlp, C, iters=30)
    print("ALL PV3 KERNEL GATES OK")


if __name__ == "__main__":
    main()
