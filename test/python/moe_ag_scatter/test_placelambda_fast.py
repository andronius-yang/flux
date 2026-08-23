"""Unit gates for placelambda_fast (single process; GPU if available).

Run: python3 test/python/moe_ag_scatter/test_placelambda_fast.py
     [--case qwen3_4n_b8] [--matrices <generated dir>]

Gates:
  1. invariants — every expert has a primary, inst covers primary, node
     count/slot caps hold, primary-count cap holds
  2. CPU == GPU bit-identity of the cold solve (the cross-device oracle
     precondition: exact-integer matmul + integer schedule)
  3. determinism — two identical solves are bit-identical
  4. quality — fast cover-incidence within +5% of the exact reference
     solver (build_placement_gpu) and always below fixed/d6
  5. warm no-op — warm re-solve on the same routing moves nothing and
     the decision reports zero gain / zero moves
  6. zero-D2H hot path — CUDA-graph capture of warm solve + decision
     succeeds and replays bit-stably (GPU only)
"""
import argparse
import importlib.util
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "python"))


def _fimport(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PLF = _fimport("placelambda_fast",
               os.path.join(_PYROOT, "flux", "testing",
                            "placelambda_fast.py"))
PLG = _fimport("placelambda_gpu",
               os.path.join(_PYROOT, "flux", "testing",
                            "placelambda_gpu.py"))

CASES = {
    "qwen3_4n_b8": ("w16x4_trace-0971f3_b8_k8_id001.routing.txt", 16, 4),
    "k3_4n_b7": ("w16x4_trace-04a502_b7_k16_id001.routing.txt", 16, 4),
}


def load_routing(gen, fname, W):
    with open(os.path.join(gen, fname)) as f:
        ntokens, topk, G = (int(x) for x in f.readline().split()[:3])
        rows = [[int(x) for x in line.split()] for line in f]
    arr = torch.tensor(rows, dtype=torch.int64)
    assert arr.shape[0] == ntokens and arr.shape[0] % W == 0
    return arr.reshape(W, ntokens // W, topk), G


def solve_sig(res):
    return (PLF.stats_host(res),
            res["primary"].cpu().tolist(),
            res["inst_nodes"].cpu().to(torch.int8).sum(1).tolist(),
            int(res["inst_nodes"].cpu().long().sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="qwen3_4n_b8", choices=CASES)
    ap.add_argument("--matrices", default=os.path.expandvars(
        "$PSCRATCH/workspace/andrewy/a2av_test_matrices/generated"))
    args = ap.parse_args()
    fname, W, L = CASES[args.case]
    topk_all, G = load_routing(args.matrices, fname, W)
    epn = G // W
    nlp = epn + 2
    NN = W // L
    has_gpu = torch.cuda.is_available()

    # --- 3 + 2: determinism and CPU==GPU ---------------------------------
    res_cpu = PLF.build_placement_fast(topk_all, L, nlp, G)
    res_cpu2 = PLF.build_placement_fast(topk_all, L, nlp, G)
    assert solve_sig(res_cpu) == solve_sig(res_cpu2), "CPU nondeterminism"
    if has_gpu:
        res_gpu = PLF.build_placement_fast(topk_all.cuda(), L, nlp, G)
        assert solve_sig(res_cpu) == solve_sig(res_gpu), (
            "CPU != GPU — cross-device oracle broken")
        hosts_cpu = PLF.finalize_hosts(res_cpu, W, L, nlp)
        hosts_gpu = PLF.finalize_hosts(res_gpu, W, L, nlp)
        assert hosts_cpu == hosts_gpu, "finalize CPU != GPU"
    print("OK determinism + cross-device identity")

    # --- 1: invariants ----------------------------------------------------
    res = res_gpu if has_gpu else res_cpu
    dev = res["primary"].device
    g_ar = torch.arange(G, device=dev)
    assert bool(res["inst_nodes"][g_ar, res["primary"]].all())
    prim_cnt = torch.bincount(res["primary"], minlength=NN)
    assert int(prim_cnt.max()) <= res["config"]["cnt_cap"]
    assert int(res["inst_nodes"].long().sum(0).max()) <= L * nlp
    hosts = PLF.finalize_hosts(res, W, L, nlp)
    per_rank = torch.zeros(W, dtype=torch.int64)
    for gexp, hs in enumerate(hosts):
        assert len(hs) == int(res["inst_nodes"][gexp].long().sum())
        assert len(set(hs)) == len(hs)
        for r in hs:
            per_rank[r] += 1
    assert int(per_rank.max()) <= nlp, "rank slot capacity exceeded"
    print("OK invariants")

    # --- 4: quality gate --------------------------------------------------
    tk = topk_all.cuda() if has_gpu else topk_all
    ex = PLG.build_placement_gpu(tk, L, nlp, G)
    ion_ex = PLF.hosts_to_ion(ex["hosts"], W, L, tk.device)
    lb_ex = int(PLF.incidence_cover_fast(tk, ion_ex, L))
    lb_f = int(PLF.incidence_cover_fast(tk, res["inst_nodes"].to(tk.device),
                                        L))
    fixed = [[gg // epn] for gg in range(G)]
    lb_fixed = int(PLF.incidence_cover_fast(
        tk, PLF.hosts_to_ion(fixed, W, L, tk.device), L))
    assert lb_f <= lb_ex * 1.05, (lb_f, lb_ex, "worse than exact +5%")
    assert lb_f < lb_fixed, (lb_f, lb_fixed, "no win over fixed")
    print(f"OK quality: fast {lb_f} exact {lb_ex} fixed {lb_fixed} "
          f"({(lb_f - lb_ex) / lb_ex * 100:+.2f}% vs exact)")

    # --- 5: warm no-op ----------------------------------------------------
    warm = PLF.build_placement_fast(
        tk, L, nlp, G, seed="warm", seed_primary=res["primary"].to(tk.device),
        seed_inst_nodes=res["inst_nodes"].to(tk.device),
        keep_bonus=PLF.LCM16 // 8, passes_a=2, passes_b=1, repair_passes=1)
    dec = PLF.place_decision_fast(tk, res["inst_nodes"].to(tk.device),
                                  warm, L)
    assert dec["moves_add"] == 0 and dec["moves_remove"] == 0, dec
    assert dec["gain_ppm"] == 0, dec
    print(f"OK warm no-op: {dec}")

    # --- 6: graph capture -------------------------------------------------
    if has_gpu:
        rp = res["primary"].clone()
        ri = res["inst_nodes"].clone()

        def hot():
            fr = PLF.build_placement_fast(
                tk, L, nlp, G, seed="warm", seed_primary=rp,
                seed_inst_nodes=ri, keep_bonus=PLF.LCM16 // 8,
                passes_a=2, passes_b=1, repair_passes=1)
            return PLF.place_decision_fast(tk, ri, fr, L, to_host=False)

        for _ in range(2):
            hot()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            packed = hot()
        gr.replay()
        torch.cuda.synchronize()
        first = packed.cpu().tolist()
        for _ in range(3):
            gr.replay()
        torch.cuda.synchronize()
        assert packed.cpu().tolist() == first, "graph replay unstable"
        print(f"OK graph capture + stable replay: packed={first}")
    print("ALL OK")


if __name__ == "__main__":
    main()
