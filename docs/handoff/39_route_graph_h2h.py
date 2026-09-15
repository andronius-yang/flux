"""One-GPU timing of the pv3c route: eager launches vs CUDA-graph replay vs
the LocCap fused op, split into host enqueue time (perf_counter, no sync)
and GPU time (CUDA events). Same synthetic inputs as test_pv3c_kernel's
head-to-head. Usage: srun ... python3 docs/handoff/39_route_graph_h2h.py"""
import os, sys, time, statistics, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test", "python", "moe_ag_scatter"))
import test_pv3c_kernel as T
import flux
ext = T.EXT.load_ext()
L = 4
shapes = [("K2 4n b1", 4, 72, 8, 384), ("K2 16n b1", 16, 72, 8, 384), ("K2 32n b1", 32, 72, 8, 384),
          ("K2 32n b16", 32, 1168, 8, 384), ("Qwen 16n b16", 16, 2048, 8, 128)]
print(f"{'shape':<13} {'router':<12} {'host us':>8} {'gpu us':>8} {'wall us':>8}")
for tag, NN, S, K, G in shapes:
    R = NN * L; nlp = G // R + 2
    topk_o = T.rand_topk(R, S, K, G, 7, 3.0); topk = T.rand_topk(R, S, K, G, 8, 3.0)
    hist = torch.zeros(NN, G, dtype=torch.int64)
    hist.index_put_(((torch.arange(R) // L).repeat_interleave(S * K), topk_o.reshape(-1)),
                    torch.ones(R * S * K, dtype=torch.int64), accumulate=True)
    res = T.PV2.pv2_solve(hist, L, nlp); hosts = T.PV2.hosts_lists(res, G)
    p2l, l2p, lcnts = T.LC.plan_tensors_from_hosts(hosts, R, nlp)
    d = torch.zeros(R, G, dtype=torch.int32, device="cuda")
    for r in range(R):
        d[r] = torch.bincount(topk[r].reshape(-1), minlength=G).int()
    l2p_d, lc_d = l2p.int().cuda().contiguous(), lcnts.int().cuda().contiguous()
    tk0 = topk[0].int().cuda().contiguous()
    wsc = torch.empty(ext.workspace_ints_c(G, R), dtype=torch.int32, device="cuda")
    pinned = torch.zeros(4, dtype=torch.int64).pin_memory()
    send = torch.empty(S * K, dtype=torch.int32, device="cuda")

    def eager(cn, cd):
        ph, st = ext.route_pv3c(tk0, d, l2p_d, lc_d, 0, nlp, L, cn, cd, wsc)
        pinned.copy_(st, non_blocking=True)
        send.copy_(ph.view(-1))
        return ph, st

    def loccap():
        ph, st = flux.placelambda_route_sl(tk0, d, l2p_d, lc_d, 0, nlp, L, 0.0625, 0)
        pinned.copy_(st, non_blocking=True)
        send.copy_(ph.view(-1))

    for _ in range(3): eager(1, 2); loccap()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        keep = eager(1, 2)
    g.replay(); torch.cuda.synchronize()

    def timeit(fn, iters=50):
        for _ in range(5): fn()
        torch.cuda.synchronize()
        hs, gs, ws = [], [], []
        for _ in range(iters):
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter(); e0.record(); fn(); t1 = time.perf_counter(); e1.record()
            torch.cuda.synchronize(); t2 = time.perf_counter()
            hs.append((t1 - t0) * 1e6); gs.append(e0.elapsed_time(e1) * 1e3); ws.append((t2 - t0) * 1e6)
        return statistics.median(hs), statistics.median(gs), statistics.median(ws)
    for name, fn in (("loccap", loccap), ("pv3c eager", lambda: eager(1, 2)), ("pv3c graph", g.replay)):
        h, gp, w = timeit(fn)
        print(f"{tag:<13} {name:<12} {h:>8.0f} {gp:>8.0f} {w:>8.0f}", flush=True)
