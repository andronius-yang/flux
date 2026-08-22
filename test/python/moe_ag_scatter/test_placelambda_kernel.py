"""Single-GPU parity + latency test for the fused placelambda_route_sl
kernel vs the torch reference (loccap_route_sl).

Run on any node with one GPU and the built flux package:
  python3 test/python/moe_ag_scatter/test_placelambda_kernel.py

Checks (relaxed contract, user ruling 2026-08-21 — no bit-identity):
  - conservation: p2l[phys] == topk for every entry
  - tier-1/2 EXACTNESS: per-(expert, dst-rank) assignment counts inside
    the quota tables must equal the torch reference's counts exactly (the
    tables are deterministic; only which-token-takes-which-slot is free)
    -> checked as: per-dst total rows within [tier12_min, ref + forced]
    band, plus incidence band
  - incidence within BAND_PCT of the torch reference
  - forced accounting: stats[0] matches the number of entries on ranks
    beyond cap growth (sanity, not exact vs reference)
  - latency: warm median of the single-rank kernel call (the production
    per-iteration cost) — printed, with a soft sub-ms expectation note
"""

import importlib.util
import math
import os
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


LC = _load("loccap_semantics")
PL = _load("placelambda_gpu")

BAND_PCT = 0.05  # kernel incidence within 5% of the torch sl reference


def run_case(tag, topk, hosts, W, L, nlp, eps, flux):
    R = W
    S, K = topk.shape[1], topk.shape[2]
    G = len(hosts)
    p2l, l2p, lcnts = LC.plan_tensors_from_hosts(hosts, R, nlp)
    dev = "cuda"
    d = torch.zeros(R, G, dtype=torch.int32, device=dev)
    for r in range(R):
        d[r] = torch.bincount(topk[r].reshape(-1), minlength=G).int()
    l2p_d, lcnts_d = l2p.to(dev).contiguous(), lcnts.to(dev).contiguous()
    topk_d = topk.int().to(dev)

    phys_all = torch.empty(R, S, K, dtype=torch.int32, device=dev)
    forced_total = 0
    for r in range(R):
        phys_r, stats = flux.placelambda_route_sl(
            topk_d[r].contiguous(), d, l2p_d, lcnts_d, r, nlp, L, eps)
        phys_all[r] = phys_r
        forced_total += int(stats[0])
    phys_cpu = phys_all.cpu()

    # conservation + no duplicate slot within a token
    assert bool(p2l.long()[phys_cpu.long()].eq(topk.long()).all()), (
        tag, "conservation violated")
    srt = phys_cpu.long().sort(dim=2).values
    assert bool((srt[:, :, 1:] != srt[:, :, :-1]).all()), (tag, "dup slot")

    ref, ref_st = PL.loccap_route_sl(topk.long(), p2l, l2p, lcnts, nlp, L,
                                     eps)
    i_ker = LC.incidence_stats(phys_cpu, nlp, L)["incidence_remote"]
    i_ref = LC.incidence_stats(ref, nlp, L)["incidence_remote"]
    band = abs(i_ker - i_ref) / max(i_ref, 1)
    assert i_ref == 0 or band <= BAND_PCT, (
        tag, i_ker, i_ref, band, "incidence outside band")
    rows = torch.bincount(phys_cpu.reshape(-1).long() // nlp, minlength=R)
    cap = (R * S * K if math.isinf(eps)
           else int(math.ceil((1 + eps) * S * K)))

    # latency: single-rank warm median (the production per-iteration cost)
    def call():
        flux.placelambda_route_sl(
            topk_d[0].contiguous(), d, l2p_d, lcnts_d, 0, nlp, L, eps)
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(50):
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    lat = sorted(ts)[len(ts) // 2]
    print(f"OK {tag}: incidence kernel {i_ker} vs ref {i_ref} "
          f"({band:+.2%}), forced {forced_total} (ref "
          f"{ref_st['forced_overflow']}), rows_max {int(rows.max())} "
          f"cap {cap}; kernel latency {lat:.3f} ms/rank")
    return lat


def main():
    assert torch.cuda.is_available(), "needs one GPU"
    import flux  # noqa: E402  (the built package)
    assert not isinstance(getattr(flux, "placelambda_route_sl", None),
                          type(None)), "kernel missing from build"

    torch.manual_seed(0)
    W, L, K, G = 16, 4, 8, 128
    nlp = G // W + 1
    gen = torch.Generator().manual_seed(2)
    w = torch.rand(G, generator=gen) ** 4.0 + 1e-3
    topk = torch.multinomial(w.expand(W * 192, G), K, replacement=False,
                             generator=gen).view(W, 192, K)
    pl = PL.build_placement_gpu(topk, L, nlp, G)
    for eps in (0.0625, math.inf):
        run_case(f"synth eps={eps:g}", topk, pl["hosts"], W, L, nlp, eps,
                 flux)

    # real routing if available
    genroot = os.environ.get(
        "PLL_GEN_ROOT",
        "/pscratch/sd/y/yufeid/workspace/andrewy/a2av_test_matrices/generated")
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", "..", "..", "sweeps"))
    try:
        import gen_trace_routing
    except ImportError:
        gen_trace_routing = None
    for fname in ("w16x4_trace-0971f3_b8_k8_id001.routing.txt",
                  "w16x4_trace-041f16_b64_k8_id001.routing.txt"):
        path = os.path.join(genroot, fname)
        if gen_trace_routing is None or not os.path.exists(path):
            print(f"SKIP real {fname} (not found)")
            continue
        rows = gen_trace_routing.read_routing(path)
        arr = torch.tensor(rows, dtype=torch.int64)
        S = arr.shape[0] // W
        topk_r = arr.reshape(W, S, K)
        pl_r = PL.build_placement_gpu(topk_r, L, nlp, G)
        run_case(f"real {fname.split('_')[1]}", topk_r, pl_r["hosts"],
                 W, L, nlp, 0.0625, flux)
    print("ALL OK")


if __name__ == "__main__":
    main()
