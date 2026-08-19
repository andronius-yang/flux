"""CPU integration test of the LocCap routing seam (no GPU, no flux libs).

Imports the flux.testing semantics modules via a package stub (their
top-level imports are torch/numpy/stdlib only), then checks:
  1. plan.phys_override derived from d6_route reproduces the D6
     reroute_expand exactly (the seam is order-compatible);
  2. loccap_route through the seam keeps the reroute contract and
     physical_rows_per_rank/plan_hash reflect the override;
  3. build_nodeaware_plan round-trips a placement blob through the shared
     slot recipe (p2l/l2p/lcnts equal the simulator's tensors).

Run: python3 test/python/moe_ag_scatter/test_loccap_seam.py
"""

import importlib
import math
import os
import sys
import types

import torch

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "..")
_PKG_DIR = os.path.join(_REPO, "python", "flux", "testing")

# package stub: flux.testing resolves to the source dir without running
# flux/__init__.py (which needs the CUDA libs)
flux_pkg = types.ModuleType("flux")
flux_pkg.__path__ = []
testing_pkg = types.ModuleType("flux.testing")
testing_pkg.__path__ = [_PKG_DIR]
sys.modules.setdefault("flux", flux_pkg)
sys.modules["flux.testing"] = testing_pkg

ue = importlib.import_module("flux.testing.ultraep_semantics")
es = importlib.import_module("flux.testing.epic_semantics")
lc = importlib.import_module("flux.testing.loccap_semantics")


def make_cfg(R=8, S=32, K=4, G=32, D=4, R_red=1):
    return ue.UltraEPConfig(S=S, K=K, G=G, R=R, H=64, D=D, R_red=R_red,
                            locality_aware=False, interleave=True)


def random_topk(R, S, K, G, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.stack([
        torch.stack([torch.randperm(G, generator=gen)[:K]
                     for _ in range(S)]) for _ in range(R)]).int()


def expand_all(cfg, plan, topk_all):
    """(entry_token, entry_phys) per src, normalized to (phys, token) order
    — the canonical form every downstream consumer sorts into."""
    out = []
    for src in range(cfg.R):
        t, p = ue.reroute_expand(cfg, plan, src, topk_all[src].long())
        order = torch.argsort(p * (cfg.S + 1) + t)
        out.append((t[order], p[order]))
    return out


def main():
    cfg = make_cfg(R_red=4)  # nlp=8: room for full per-node coverage
    topk_all = random_topk(cfg.R, cfg.S, cfg.K, cfg.G, seed=99)
    tpe = ue.loads_from_topk(cfg, topk_all)

    # --- 1. seam parity with D6 on a replicated placement -----------------
    # nodeaware blob: every expert on every node (round-robin rank in node)
    L = cfg.D
    NN = cfg.R // L
    epn = cfg.epn
    hosts = []
    for g in range(cfg.G):
        hs = {g // epn}
        for n in range(NN):
            if (g // epn) // L != n:
                hs.add(n * L + g % L)
        hosts.append(sorted(hs))
    blob = {"version": 1, "G": cfg.G, "W": cfg.R, "nlp": cfg.nlp,
            "hosts": hosts}
    plan = es.build_nodeaware_plan(cfg, tpe, blob)

    # shared slot recipe round-trip
    p2l2, l2p2, lcnts2 = lc.plan_tensors_from_hosts(hosts, cfg.R, cfg.nlp)
    assert torch.equal(plan.p2l, p2l2)
    assert torch.equal(plan.lcnts, lcnts2)
    assert torch.equal(plan.l2p[:, :l2p2.shape[1]], l2p2)

    base_hash = plan.plan_hash()
    base = expand_all(cfg, plan, topk_all)

    d6_phys = lc.d6_route(topk_all.long(), plan.l2p, plan.lcnts)
    plan.phys_override = d6_phys
    assert plan.plan_hash() != base_hash, "override must change plan_hash"
    via_seam = expand_all(cfg, plan, topk_all)
    for src in range(cfg.R):
        assert torch.equal(base[src][0], via_seam[src][0]), src
        assert torch.equal(base[src][1], via_seam[src][1]), src
    print("seam parity with D6: OK")

    # --- 2. loccap through the seam ---------------------------------------
    phys = lc.loccap_route(topk_all.long(), plan.p2l, plan.l2p, plan.lcnts,
                           cfg.nlp, L, 0.25)
    plan.phys_override = phys
    rows = plan.physical_rows_per_rank()
    st = lc.incidence_stats(phys, cfg.nlp, L)
    assert rows == st["rows_per_rank"]
    cap = int(math.ceil(1.25 * cfg.S * cfg.K))
    assert max(rows) <= cap, (max(rows), cap)
    ex = expand_all(cfg, plan, topk_all)
    n_entries = sum(t.numel() for t, _ in ex)
    assert n_entries == cfg.R * cfg.S * cfg.K
    # per-entry conservation is hard-asserted inside _expand_from_phys
    print(f"loccap through seam: OK (incidence_remote "
          f"{st['incidence_remote']}, rows max {max(rows)} <= cap {cap})")

    # --- 3. sidecar mismatch defends --------------------------------------
    bad = dict(blob, nlp=cfg.nlp + 1)
    try:
        es.build_nodeaware_plan(make_cfg(R_red=4), tpe, bad)
        raise SystemExit("expected nlp-mismatch assert")
    except AssertionError:
        print("sidecar nlp-mismatch assert: OK")

    print("OK: all seam integration tests passed")


if __name__ == "__main__":
    main()
