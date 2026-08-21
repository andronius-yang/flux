"""CPU unit tests for the PLACE-lambda/LocCap GPU-portable module (no GPU,
no flux import).

Run directly: python3 test/python/moe_ag_scatter/test_placelambda_gpu.py

Covers: routing invariants (conservation, caps+forced accounting, dup-slot
freedom), determinism (double run), locality sanity (replicated placement
at eps=inf -> zero remote incidence), quality vs the exact python loccap
reference on the same inputs (bounded-round approximation must land within
QUALITY_TOL of the exact incidence and never lose to D6), and the placement
solver's structural + improvement guarantees. When CUDA is available the
same inputs are re-run on the GPU and asserted bit-identical to CPU (the
cross-device same-code oracle).
"""

import importlib.util
import math
import os

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

QUALITY_TOL = 0.10  # gpu-variant incidence within 10% of the exact loccap


def random_topk(R, S, K, G, seed, skew=None):
    gen = torch.Generator().manual_seed(seed)
    if skew is None:
        w = torch.ones(G)
    else:
        w = torch.rand(G, generator=gen) ** skew + 1e-3
    return torch.multinomial(w.expand(R * S, G), K, replacement=False,
                             generator=gen).view(R, S, K)


def check_route(topk, phys, p2l, nlp, cap, stats):
    R, S, K = topk.shape
    assert tuple(phys.shape) == (R, S, K)
    assert bool(p2l.long()[phys.long()].eq(topk.long()).all()), "conservation"
    srt = phys.long().sort(dim=2).values
    assert bool((srt[:, :, 1:] != srt[:, :, :-1]).all()), "dup slot in token"
    rows = torch.bincount(phys.reshape(-1).long() // nlp, minlength=R)
    assert int(rows.max()) == stats["rows_max"]
    assert int((rows - cap).clamp(min=0).sum()) == stats["over_cap_rows"]


def main():
    torch.manual_seed(0)
    R, S, K, G, L = 16, 192, 8, 128, 4
    nlp = G // R + 1

    # -- replicated placement, eps=inf -> pure locality, zero remote ------
    # (small G so every-node replication fits: 32 experts / 4 ranks = 8/rank)
    G2, nlp2 = 32, 8
    hosts_full = [sorted(n * L + (g % L) for n in range(R // L))
                  for g in range(G2)]
    p2l, l2p, lcnts = LC.plan_tensors_from_hosts(hosts_full, R, nlp2)
    topk = random_topk(R, S, K, G2, seed=1)
    phys, st = PL.loccap_route_gpu(topk, p2l, l2p, lcnts, nlp2, L, math.inf)
    check_route(topk, phys, p2l, nlp2, R * S * K, st)
    inc = LC.incidence_stats(phys, nlp2, L)
    assert inc["incidence_remote"] == 0, inc
    print(f"OK replicated/eps=inf: zero remote incidence, stats {st}")

    # -- fixed placement (single instance): route forced to the host ------
    epn = G // R
    hosts_fixed = [[g // epn] for g in range(G)]
    p2f, l2f, lcf = LC.plan_tensors_from_hosts(hosts_fixed, R, nlp)
    phys_f, st_f = PL.loccap_route_gpu(topk, p2f, l2f, lcf, nlp, L, 0.0)
    check_route(topk, phys_f, p2f, nlp, int(math.ceil(1.0 * S * K)), st_f)
    d6_f = LC.d6_route(topk, l2f, lcf)
    assert bool(phys_f.eq(d6_f).all()), "single-instance route must be forced"
    print(f"OK fixed placement: forced routing matches D6, stats {st_f}")

    # -- determinism: double run bit-identical ----------------------------
    hosts = None
    for skew in (4.0,):
        topk_s = random_topk(R, S, K, G, seed=2, skew=skew)
        pl = PL.build_placement_gpu(topk_s, L, nlp, G)
        hosts = pl["hosts"]
        pl2 = PL.build_placement_gpu(topk_s, L, nlp, G)
        assert pl == pl2, "placement solver not deterministic"
        for g, h in enumerate(hosts):
            assert len(h) >= 1 and len(set(h)) == len(h), (g, h)
        per_rank = torch.zeros(R, dtype=torch.int64)
        for h in hosts:
            for r in h:
                per_rank[r] += 1
        assert int(per_rank.max()) <= nlp, "rank slot capacity exceeded"
        print(f"OK placement solver: deterministic + structural "
              f"(fm_moves {pl['stats']['fm_moves']}, replicas "
              f"{pl['stats']['replica_slots_spent']})")

        p2n, l2n, lcn = LC.plan_tensors_from_hosts(hosts, R, nlp)
        for eps in (0.0, 0.0625, 0.25, math.inf):
            cap = (R * S * K if math.isinf(eps)
                   else int(math.ceil((1 + eps) * S * K)))
            a, sa = PL.loccap_route_gpu(topk_s, p2n, l2n, lcn, nlp, L, eps)
            b, sb = PL.loccap_route_gpu(topk_s, p2n, l2n, lcn, nlp, L, eps)
            assert bool(a.eq(b).all()) and sa == sb, "router not deterministic"
            check_route(topk_s, a, p2n, nlp, cap, sa)

            # quality vs the exact python reference on identical inputs
            ref = LC.loccap_route(topk_s, p2n, l2n, lcn, nlp, L, eps)
            i_gpu = LC.incidence_stats(a, nlp, L)["incidence_remote"]
            i_ref = LC.incidence_stats(ref, nlp, L)["incidence_remote"]
            d6 = LC.d6_route(topk_s, l2n, lcn)
            i_d6 = LC.incidence_stats(d6, nlp, L)["incidence_remote"]
            assert i_gpu <= i_d6, (eps, i_gpu, i_d6, "lost to D6")
            if i_ref > 0:
                excess = (i_gpu - i_ref) / i_ref
                assert excess <= QUALITY_TOL, (eps, i_gpu, i_ref, excess)
            print(f"OK eps={eps:g}: incidence gpu {i_gpu} vs exact {i_ref} "
                  f"(d6 {i_d6}); rows_max {sa['rows_max']} cap {cap} "
                  f"overflow {sa['over_cap_rows']}")

        # placement improvement: solved placement beats fixed on incidence
        ion_new = PL.hosts_to_inst_on_node(hosts, R, L)
        ion_fix = PL.hosts_to_inst_on_node(hosts_fixed, R, L)
        lb_new = PL.incidence_lb(topk_s, ion_new, L)
        lb_fix = PL.incidence_lb(topk_s, ion_fix, L)
        assert lb_new <= lb_fix, (lb_new, lb_fix)
        dec = PL.place_decision(topk_s, hosts_fixed, hosts, L)
        assert dec["lb_cur"] == lb_fix and dec["lb_new"] == lb_new
        assert dec["gain_ppm"] >= 0 and dec["moves_add"] > 0
        print(f"OK trigger: lb fixed {lb_fix} -> solved {lb_new} "
              f"(gain {dec['gain_ppm']/1e4:.1f}%, adds {dec['moves_add']}, "
              f"trigger {dec['trigger']})")

        # self-decision: fresh vs itself must never trigger
        dec0 = PL.place_decision(topk_s, hosts, hosts, L)
        assert dec0["gain_ppm"] == 0 and dec0["trigger"] == 0
        assert dec0["moves_add"] == 0 and dec0["moves_remove"] == 0

    # -- cross-device bit-identity (only when CUDA is present) ------------
    if torch.cuda.is_available():
        dev = "cuda"
        topk_s = random_topk(R, S, K, G, seed=2, skew=4.0)
        p2n, l2n, lcn = LC.plan_tensors_from_hosts(hosts, R, nlp)
        for eps in (0.0625, math.inf):
            a_cpu, _ = PL.loccap_route_gpu(topk_s, p2n, l2n, lcn, nlp, L, eps)
            a_gpu, _ = PL.loccap_route_gpu(topk_s.to(dev), p2n.to(dev),
                                           l2n.to(dev), lcn.to(dev),
                                           nlp, L, eps)
            assert bool(a_cpu.eq(a_gpu.cpu()).all()), (
                f"CPU/GPU routing mismatch at eps={eps}")
        pl_cpu = PL.build_placement_gpu(topk_s, L, nlp, G)
        pl_gpu = PL.build_placement_gpu(topk_s.to(dev), L, nlp, G)
        assert pl_cpu["hosts"] == pl_gpu["hosts"], "CPU/GPU placement mismatch"
        print("OK cross-device: CPU and GPU outputs bit-identical")
    else:
        print("SKIP cross-device check (no CUDA)")

    print("ALL OK")


if __name__ == "__main__":
    main()
