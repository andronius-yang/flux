"""Allocation-free planner-pipeline solidity for the loccap_sl arm.

Single GPU, NO torch.distributed init: EpicIterPlanner's exchange is
injected (exchange_fn), so R ranks' full per-iteration metadata pipelines
(kernel route -> assembled phys -> direct layout -> splits/segments) run
emulated at R in {16, 32, 128} with the K3 shape. This exercises, at scale:
  - UltraEPConfig.__post_init__ static asserts (R=128, G=896, K=16)
  - build_nodeaware_plan + plan_tensors_from_hosts at G=896
  - EpicIterPlanner(router="loccap_sl") ctor + derive + check_relaxed
  - conservation, bound compliance, and cross-rank splits reciprocity
    (ip_r.out_splits[s] == ip_s.in_splits[r]) on one shared assembled phys
  - derive_reference() bitwise stability (the deterministic contract side)

Run: python3 test/python/moe_ag_scatter/test_placelambda_sl_planner.py
(uses flux only for the kernel; the hc op layer is 4n-live-gated.)
"""

import importlib.util
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.join(_HERE, "..", "..", "..", "python", "flux", "testing")
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "python"))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_BASE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LC = _load("loccap_semantics")
PL = _load("placelambda_gpu")

G, K, L = 896, 16, 4
EPS = 0.0625
N_ITERS = 5


def run_planner_scale(R, S, flux, ES):
    torch.manual_seed(R)
    gen = torch.Generator().manual_seed(R)
    w = torch.rand(G, generator=gen) ** 3.0 + 1e-3
    topk = torch.multinomial(w.expand(R * S, G), K, replacement=False,
                             generator=gen).view(R, S, K)
    epn = G // R
    nlp = epn + 2

    pl = PL.build_placement_gpu(topk, L, nlp, G)
    cfg = ES.UltraEPConfig(S=S, K=K, G=G, R=R, H=3584, D=L, R_red=2)
    tpe = torch.stack([torch.bincount(topk[r].reshape(-1), minlength=G)
                       for r in range(R)]).int()
    pblob = {"version": 2, "G": G, "W": R, "nlp": cfg.nlp,
             "hosts": pl["hosts"]}
    plan = ES.build_nodeaware_plan(cfg, tpe, pblob)

    ref, aux = PL.loccap_route_sl(
        topk.long(), plan.p2l, plan.l2p, plan.lcnts, cfg.nlp, L, EPS,
        return_tables=True)
    plan.phys_override = ref
    bounds = PL.loccap_sl_bounds(aux, R, None)  # auto f_cap
    f_cap = bounds["f_cap"]
    ref_inc = LC.incidence_stats(ref, cfg.nlp, L)["incidence_remote"]

    dev = torch.device("cuda")
    topk_d = topk.int().to(dev)
    d_dev = tpe.to(dev).contiguous().view(-1)

    planners = []
    for r in range(R):
        p = ES.EpicIterPlanner(
            plan, r, dev, topk_all=topk, local_world_size=L,
            router="loccap_sl", eps=EPS, f_cap=f_cap,
            exchange_fn=lambda po, _r=r: run_planner_scale.assembled)
        p.relaxed_bounds = bounds
        p.ref_incidence = ref_inc
        planners.append(p)

    # derive_reference bitwise stability (deterministic contract side)
    a = planners[0].derive_reference()
    b = planners[0].derive_reference()
    assert a.seg_rows == b.seg_rows and a.seg_start == b.seg_start
    assert a.send_counts == b.send_counts and a.recv_counts == b.recv_counts
    assert torch.equal(a.send_row_index, b.send_row_index)
    assert a.n_recv == b.n_recv and a.max_pair_rows == b.max_pair_rows

    total = R * S * K
    for it in range(N_ITERS):
        # one shared assembled routing per emulated iteration (each row
        # authored once by its rank's relaxed kernel — the communication
        # model, minus the wire)
        assembled = torch.empty(total, dtype=torch.int32, device=dev)
        for r in range(R):
            pr, st = flux.placelambda_route_sl(
                topk_d[r].contiguous(), d_dev.view(R, G),
                plan.l2p.to(dev).contiguous(),
                plan.lcnts.to(dev).contiguous(), r, cfg.nlp, L, EPS,
                f_cap=f_cap)
            assembled[r * S * K:(r + 1) * S * K] = pr.view(-1)
            assert int(st[2]) == 0, (R, it, r, "forced-budget overflow")
        run_planner_scale.assembled = assembled

        ips = []
        for r in range(R):
            ip = planners[r].derive(d_dev)
            facts = planners[r].check_relaxed(
                ip, bounds, ref_incidence=ref_inc)
            ips.append(ip)
        assert sum(ip.n_recv for ip in ips) == total, (
            R, it, "recv rows don't partition the entry space")
        # cross-rank splits reciprocity on the shared assembled phys
        for r in (0, R // 2, R - 1):
            for s in (0, R // 2, R - 1):
                out_rs = int(ips[r].out_splits[s])
                in_sr = int(ips[s].in_splits[r])
                assert out_rs == in_sr, (R, it, r, s, out_rs, in_sr)
    print(f"OK planner R={R} S={S}: {N_ITERS} emulated iterations, "
          f"reciprocity + bounds + reference stability clean "
          f"(ref incidence {ref_inc}, recv_cap {bounds['recv_cap']})")


def main():
    assert torch.cuda.is_available(), "needs one GPU"
    import flux
    from flux.testing import epic_semantics as ES
    for R, S in ((16, 512), (32, 512), (128, 256)):
        run_planner_scale(R, S, flux, ES)
    print("ALL OK")


if __name__ == "__main__":
    main()
