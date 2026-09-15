"""Single-GPU gate for the PV3C kernel (route_pv3c = pv3 + vacate pass) vs
the torch references: constraint 2 on the kernel product (exact integer
bounds), conservation, incidence <= the pv3 kernel's (monotone), and
within a band of the pv3c reference; latency at paper shapes for C in
{1/16, 1/4, 1/2}. Run on a node with one GPU:
  python3 test/python/moe_ag_scatter/test_pv3c_kernel.py
"""
import os, statistics, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pv3_kernel import _load, PV3, LC, EXT, PV2, rand_hosts, rand_topk, slots_for  # noqa


def run_all(ext, fn, wsz, topk, l2p, lcnts, R, nlp, L, C):
    S, K = topk.shape[1], topk.shape[2]
    G = int(lcnts.numel())
    d = torch.zeros(R, G, dtype=torch.int32, device="cuda")
    for r in range(R):
        d[r] = torch.bincount(topk[r].reshape(-1), minlength=G).int()
    l2p_d, lc_d = l2p.int().cuda().contiguous(), lcnts.int().cuda().contiguous()
    tk_d = topk.int().cuda()
    ws = torch.empty(wsz(G, R), dtype=torch.int32, device="cuda")
    phys = torch.empty(R, S, K, dtype=torch.int32, device="cuda")
    moved = 0
    for r in range(R):
        ph, st = fn(tk_d[r].contiguous(), d, l2p_d, lc_d, r, nlp, L, C[0], C[1], ws)
        assert int(st[2]) == 0 and int(st[0]) == 0, st
        moved += int(st[1])
        phys[r] = ph
    return phys.cpu(), moved, (d, tk_d, l2p_d, lc_d, ws)


def case(tag, ext, topk, hosts, R, L, nlp, C, iters=20, ref=True):
    S, K = topk.shape[1], topk.shape[2]
    p2l, l2p, lcnts = LC.plan_tensors_from_hosts(hosts, R, nlp)
    ipr = PV3.instance_phys_of_rank(l2p, lcnts, nlp, R)
    ph3, _, _ = run_all(ext, ext.route_pv3, ext.workspace_ints, topk, l2p, lcnts, R, nlp, L, C)
    phc, moved, (d, tk_d, l2p_d, lc_d, ws) = run_all(ext, ext.route_pv3c, ext.workspace_ints_c, topk, l2p, lcnts, R, nlp, L, C)
    assert bool(p2l.long()[phc.long()].eq(topk).all()), (tag, "conservation")
    st = PV3.pv3_check(phc.long(), topk, ipr, nlp, L, C[0], C[1])
    assert st["c2_over_rows"] == 0 and st["c2_under_rows"] == 0 and st["nonhost_rows"] == 0, (tag, st)
    i3, r3 = PV3.incidence_remote(ph3.long(), nlp, L)
    ic, rc = PV3.incidence_remote(phc.long(), nlp, L)
    assert ic <= i3 and rc <= r3, (tag, "vacate not monotone", i3, ic, r3, rc)
    line = f"{tag:<22} C={C[0]}/{C[1]:<3} incid pv3k {i3} -> pv3ck {ic} ({100*(ic/max(i3,1)-1):+.1f}%) moved {moved}"
    if ref:
        phr, str_ = PV3.pv3c_route(topk, p2l, l2p, lcnts, nlp, L, C[0], C[1])
        ir, _ = PV3.incidence_remote(phr.long(), nlp, L)
        line += f" ref pv3c {ir} ({100*(ic/max(ir,1)-1):+.1f}% vs ref)"
    # latency: own-rank call, warm median, eager + graph
    r = 0
    tk0 = tk_d[r].contiguous()
    for _ in range(3):
        ext.route_pv3c(tk0, d, l2p_d, lc_d, r, nlp, L, C[0], C[1], ws)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ext.route_pv3c(tk0, d, l2p_d, lc_d, r, nlp, L, C[0], C[1], ws)
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    print(line + f" | kernel {statistics.median(ts):.3f} ms", flush=True)


def main():
    ext = EXT.load_ext()
    L = 4
    for NN, S, K, G in [(2, 16, 8, 32), (4, 48, 8, 96), (8, 32, 8, 128), (4, 64, 16, 224)]:
        R = NN * L; nlp = slots_for(G, R, NN)
        for gen in range(2):
            hosts = rand_hosts(G, R, nlp, NN, L, gen)
            topk = rand_topk(R, S, K, G, gen + 100, 3.0)
            for C in [(1, 16), (1, 4), (1, 2)]:
                case(f"parity NN{NN} K{K} gen{gen}", ext, topk, hosts, R, L, nlp, C, iters=5)
    print("pv3c parity OK")
    shapes = [("K2 4n b16", 4, 1168, 8, 384), ("Qwen 4n b16", 4, 2048, 8, 128),
              ("K2 8n b16", 8, 1168, 8, 384), ("Qwen 16n b16", 16, 2048, 8, 128),
              ("K2 16n b16", 16, 1168, 8, 384), ("K3 16n b16", 16, 1168, 16, 896),
              ("K3 32n b16", 32, 1168, 16, 896)]
    for tag, NN, S, K, G in shapes:
        R = NN * L; nlp = G // R + 2
        topk_o = rand_topk(R, S, K, G, 7, 3.0); topk = rand_topk(R, S, K, G, 8, 3.0)
        hist = torch.zeros(NN, G, dtype=torch.int64)
        hist.index_put_(((torch.arange(R) // L).repeat_interleave(S * K), topk_o.reshape(-1)),
                        torch.ones(R * S * K, dtype=torch.int64), accumulate=True)
        res = PV2.pv2_solve(hist, L, nlp); hosts = PV2.hosts_lists(res, G)
        for C in [(1, 16), (1, 4), (1, 2)]:
            case(tag, ext, topk, hosts, R, L, nlp, C, iters=20, ref=(R <= 32))
    print("ALL PV3C KERNEL GATES OK")


if __name__ == "__main__" and "--h2h" not in sys.argv and "--h2h32" not in sys.argv:
    main()


def headtohead():
    """LocCap kernel (flux.placelambda_route_sl, the plotted router) vs pv3
    vs pv3c at the paper shapes on one GPU — same inputs, warm medians."""
    import flux
    ext = EXT.load_ext()
    L = 4
    shapes = [("K2 4n b16", 4, 1168, 8, 384), ("Qwen 4n b16", 4, 2048, 8, 128),
              ("K2 8n b16", 8, 1168, 8, 384), ("Qwen 8n b16", 8, 2048, 8, 128),
              ("K2 16n b16", 16, 1168, 8, 384), ("Qwen 16n b16", 16, 2048, 8, 128),
              ("K3 16n b16", 16, 1168, 16, 896), ("K3 32n b16", 32, 1168, 16, 896)]
    print(f"{'shape':<14} {'loccap(1/16)':>13} {'pv3(1/4)':>10} {'pv3c(1/4)':>10} {'pv3c(1/2)':>10}")
    for tag, NN, S, K, G in shapes:
        R = NN * L; nlp = G // R + 2
        topk_o = rand_topk(R, S, K, G, 7, 3.0); topk = rand_topk(R, S, K, G, 8, 3.0)
        hist = torch.zeros(NN, G, dtype=torch.int64)
        hist.index_put_(((torch.arange(R) // L).repeat_interleave(S * K), topk_o.reshape(-1)),
                        torch.ones(R * S * K, dtype=torch.int64), accumulate=True)
        res = PV2.pv2_solve(hist, L, nlp); hosts = PV2.hosts_lists(res, G)
        p2l, l2p, lcnts = LC.plan_tensors_from_hosts(hosts, R, nlp)
        d = torch.zeros(R, G, dtype=torch.int32, device="cuda")
        for r in range(R):
            d[r] = torch.bincount(topk[r].reshape(-1), minlength=G).int()
        l2p_d, lc_d = l2p.int().cuda().contiguous(), lcnts.int().cuda().contiguous()
        tk0 = topk[0].int().cuda().contiguous()
        wsc = torch.empty(ext.workspace_ints_c(G, R), dtype=torch.int32, device="cuda")
        def med(fn, iters=30):
            for _ in range(3): fn()
            torch.cuda.synchronize(); ts = []
            for _ in range(iters):
                torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
            return statistics.median(ts)
        t_lc = med(lambda: flux.placelambda_route_sl(tk0, d, l2p_d, lc_d, 0, nlp, L, 0.0625, 0))
        t_p3 = med(lambda: ext.route_pv3(tk0, d, l2p_d, lc_d, 0, nlp, L, 1, 4, wsc))
        t_c4 = med(lambda: ext.route_pv3c(tk0, d, l2p_d, lc_d, 0, nlp, L, 1, 4, wsc))
        t_c2 = med(lambda: ext.route_pv3c(tk0, d, l2p_d, lc_d, 0, nlp, L, 1, 2, wsc))
        print(f"{tag:<14} {t_lc:>10.3f} ms {t_p3:>7.3f} ms {t_c4:>7.3f} ms {t_c2:>7.3f} ms", flush=True)


def headtohead32():
    """The weak-scaling figure's 32n shapes (K2 b1/b16/b64; Qwen b16) —
    LocCap kernel vs pv3 vs pv3c on one GPU."""
    import flux
    ext = EXT.load_ext()
    L = 4
    shapes = [("K2 32n b1", 32, 72, 8, 384), ("K2 32n b4", 32, 296, 8, 384),
              ("K2 32n b16", 32, 1168, 8, 384), ("K2 32n b64", 32, 4672, 8, 384),
              ("Qwen 32n b16", 32, 2048, 8, 128), ("K2 16n b64", 16, 4672, 8, 384)]
    print(f"{'shape':<14} {'loccap(1/16)':>13} {'pv3(1/4)':>10} {'pv3c(1/4)':>10} {'pv3c(1/2)':>10}")
    for tag, NN, S, K, G in shapes:
        R = NN * L; nlp = G // R + 2
        topk_o = rand_topk(R, S, K, G, 7, 3.0); topk = rand_topk(R, S, K, G, 8, 3.0)
        hist = torch.zeros(NN, G, dtype=torch.int64)
        hist.index_put_(((torch.arange(R) // L).repeat_interleave(S * K), topk_o.reshape(-1)),
                        torch.ones(R * S * K, dtype=torch.int64), accumulate=True)
        res = PV2.pv2_solve(hist, L, nlp); hosts = PV2.hosts_lists(res, G)
        p2l, l2p, lcnts = LC.plan_tensors_from_hosts(hosts, R, nlp)
        d = torch.zeros(R, G, dtype=torch.int32, device="cuda")
        for r in range(R):
            d[r] = torch.bincount(topk[r].reshape(-1), minlength=G).int()
        l2p_d, lc_d = l2p.int().cuda().contiguous(), lcnts.int().cuda().contiguous()
        tk0 = topk[0].int().cuda().contiguous()
        wsc = torch.empty(ext.workspace_ints_c(G, R), dtype=torch.int32, device="cuda")
        def med(fn, iters=30):
            for _ in range(3): fn()
            torch.cuda.synchronize(); ts = []
            for _ in range(iters):
                torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
            return statistics.median(ts)
        t_lc = med(lambda: flux.placelambda_route_sl(tk0, d, l2p_d, lc_d, 0, nlp, L, 0.0625, 0))
        t_p3 = med(lambda: ext.route_pv3(tk0, d, l2p_d, lc_d, 0, nlp, L, 1, 4, wsc))
        t_c4 = med(lambda: ext.route_pv3c(tk0, d, l2p_d, lc_d, 0, nlp, L, 1, 4, wsc))
        t_c2 = med(lambda: ext.route_pv3c(tk0, d, l2p_d, lc_d, 0, nlp, L, 1, 2, wsc))
        print(f"{tag:<14} {t_lc:>10.3f} ms {t_p3:>7.3f} ms {t_c4:>7.3f} ms {t_c2:>7.3f} ms", flush=True)


if __name__ == "__main__" and "--h2h" in sys.argv:
    headtohead()
    sys.exit(0)
if __name__ == "__main__" and "--h2h32" in sys.argv:
    headtohead32()
    sys.exit(0)
