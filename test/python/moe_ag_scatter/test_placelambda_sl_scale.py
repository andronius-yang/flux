"""Allocation-free scale solidity for the loccap_sl KERNEL at K3 semantics.

Single GPU (login A100), no torch.distributed: the sender-local design makes
every rank's row independently computable, so R ranks are emulated by
looping the kernel. Covers R in {16, 32, 64, 128} (4n/8n/16n/32n at L=4)
with the K3 canon shape (G=896, topk=16).

Per (R, routing, repeat):
  - conservation + dup-slot freedom on the assembled routing
  - PROVABLE bound compliance: per-rank recv <= recv_ub, per-(src,dst)
    rows <= pair_ub (loccap_sl_bounds from the deterministic tables) —
    checked on EVERY relaxed repeat (different atomic outcomes)
  - forced-budget overflow == 0 at the AUTO-derived f_cap (the sizing
    contract; f_cap = 2x the reference's max per-pair forced flow + 8)
  - incidence within BAND of the deterministic torch reference
  - reference self-consistency: the reference routing itself obeys the
    bounds
Kernel latency per R is printed (informational).

Run: python3 test/python/moe_ag_scatter/test_placelambda_sl_scale.py
Optionally set PLL_GEN_ROOT to pick up real K3-synth trace routings
(w{R//4}x4_trace-*_b*_k16_*.routing.txt) instead of synthetic skew.
"""

import glob
import importlib.util
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

G, K, L = 896, 16, 4          # K3 canon
EPS = 0.0625
BAND = 0.05
N_REP = 5


def synth_topk(R, S, seed, skew=3.0):
    gen = torch.Generator().manual_seed(seed)
    w = torch.rand(G, generator=gen) ** skew + 1e-3
    return torch.multinomial(w.expand(R * S, G), K, replacement=False,
                             generator=gen).view(R, S, K)


def real_topk(R, S_want):
    root = os.environ.get(
        "PLL_GEN_ROOT",
        "/pscratch/sd/y/yufeid/workspace/andrewy/a2av_test_matrices/generated")
    pat = os.path.join(root, f"w{R}x{L}_trace-*_k{K}_*.routing.txt")
    for path in sorted(glob.glob(pat)):
        sys.path.insert(0, os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "..", "..", "..", "sweeps"))
        import gen_trace_routing
        rows = gen_trace_routing.read_routing(path)
        arr = torch.tensor(rows, dtype=torch.int64)
        if arr.shape[0] % R or arr.shape[1] != K:
            continue
        if int(arr.max()) >= G:
            continue
        return arr.reshape(R, arr.shape[0] // R, K), os.path.basename(path)
    return None, None


def run_scale(R, topk, tag, flux):
    S = topk.shape[1]
    epn = G // R
    nlp = epn + 2
    pl = PL.build_placement_gpu(topk, L, nlp, G)
    p2l, l2p, lcnts = LC.plan_tensors_from_hosts(pl["hosts"], R, nlp)

    ref, aux = PL.loccap_route_sl(topk.long(), p2l, l2p, lcnts, nlp, L,
                                  EPS, return_tables=True)
    bounds = PL.loccap_sl_bounds(aux, R, None)  # auto f_cap
    f_cap = bounds["f_cap"]
    i_ref = LC.incidence_stats(ref, nlp, L)["incidence_remote"]

    def audit(phys, who):
        assert bool(p2l.long()[phys.long()].eq(topk.long()).all()), (
            tag, who, "conservation")
        srt = phys.long().sort(dim=2).values
        assert bool((srt[:, :, 1:] != srt[:, :, :-1]).all()), (
            tag, who, "dup slot")
        serve = phys.reshape(R, -1).long() // nlp
        recv = torch.bincount(serve.reshape(-1), minlength=R)
        pair = torch.bincount(
            (torch.arange(R, dtype=torch.int64).unsqueeze(1) * R
             + serve).reshape(-1), minlength=R * R).view(R, R)
        over_r = (recv - bounds["recv_ub"]).clamp(min=0)
        over_p = (pair - bounds["pair_ub"]).clamp(min=0)
        assert int(over_r.sum()) == 0, (tag, who, "recv bound",
                                        int(over_r.max()))
        assert int(over_p.sum()) == 0, (tag, who, "pair bound",
                                        int(over_p.max()))
        inc = LC.incidence_stats(phys, nlp, L)["incidence_remote"]
        if i_ref:
            assert abs(inc - i_ref) / i_ref <= BAND, (tag, who, inc, i_ref)
        return inc

    audit(ref, "reference")

    dev = "cuda"
    topk_d = topk.int().to(dev)
    d = torch.stack([torch.bincount(topk[r].reshape(-1), minlength=G)
                     for r in range(R)]).int().to(dev)
    l2p_d, lcnts_d = l2p.to(dev).contiguous(), lcnts.to(dev).contiguous()
    lat = []
    for rep in range(N_REP):
        phys = torch.empty(R, S, K, dtype=torch.int32, device=dev)
        t0 = time.perf_counter()
        for r in range(R):
            pr, st = flux.placelambda_route_sl(
                topk_d[r].contiguous(), d, l2p_d, lcnts_d, r, nlp, L, EPS,
                f_cap=f_cap)
            phys[r] = pr
            assert int(st[2]) == 0, (tag, r, "kernel forced-budget overflow")
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) / R * 1e3)
        inc = audit(phys.cpu(), f"kernel rep{rep}")
    print(f"OK {tag}: R={R} S={S} incidence ref {i_ref} "
          f"(last kernel {inc}); bounds recv_cap {bounds['recv_cap']} "
          f"pair_cap {bounds['pair_cap']} f_cap {f_cap}; "
          f"kernel {min(lat):.3f} ms/rank ({N_REP} relaxed repeats clean)")


def main():
    assert torch.cuda.is_available(), "needs one GPU"
    import flux
    torch.manual_seed(0)
    for R in (16, 32, 64, 128):
        S = 1024
        topk, name = real_topk(R, S)
        if topk is not None:
            run_scale(R, topk, f"real:{name}", flux)
        else:
            run_scale(R, synth_topk(R, S, seed=R), f"synth R{R}", flux)
    # one larger-S rung at 4n (b28-per-rank analog)
    run_scale(16, synth_topk(16, 4096, seed=99), "synth R16 S4096", flux)
    print("ALL OK")


if __name__ == "__main__":
    main()
