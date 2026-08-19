"""CPU unit tests for the LocCap router (no GPU, no flux import).

Run directly: python3 test/python/moe_ag_scatter/test_loccap_router.py
"""

import importlib.util
import math
import os

import torch

# File-path import (no `flux` package init, which needs the CUDA libs): the
# same loading contract sweeps/predict_placement.py uses, so this test also
# guards the simulator==runtime same-code guarantee.
_LOCCAP_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "python", "flux", "testing", "loccap_semantics.py")
_spec = importlib.util.spec_from_file_location("loccap_semantics", _LOCCAP_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
incidence_stats = _mod.incidence_stats
loccap_route = _mod.loccap_route
route_hash = _mod.route_hash


def make_fixed_placement(G, R, nlp):
    """Contiguous homing, no replicas (the --placement none shape)."""
    assert G % R == 0 and G // R <= nlp
    epn = G // R
    P = R * nlp
    p2l = torch.full((P,), -1, dtype=torch.int32)
    l2p = torch.full((G, R), -1, dtype=torch.int32)
    lcnts = torch.ones(G, dtype=torch.int32)
    for g in range(G):
        phys = (g // epn) * nlp + (g % epn)
        p2l[phys] = g
        l2p[g, 0] = phys
    return p2l, l2p, lcnts


def make_replicated_placement(G, R, nlp, ranks_per_node):
    """Every expert on every NODE once (round-robin rank within node), plus
    contiguous home — per-node-first coverage, the PLACE-lambda end state."""
    epn = G // R
    NN = R // ranks_per_node
    P = R * nlp
    p2l = torch.full((P,), -1, dtype=torch.int32)
    l2p = torch.full((G, R), -1, dtype=torch.int32)
    lcnts = torch.zeros(G, dtype=torch.int32)
    next_slot = [0] * R
    slots = [[] for _ in range(G)]
    for g in range(G):
        home_rank = g // epn
        for n in range(NN):
            if home_rank // ranks_per_node == n:
                r = home_rank
            else:
                r = n * ranks_per_node + (g % ranks_per_node)
            if any(s // nlp == r for s in slots[g]):
                continue
            phys = r * nlp + next_slot[r]
            assert next_slot[r] < nlp, "nlp too small for this test"
            next_slot[r] += 1
            p2l[phys] = g
            slots[g].append(phys)
    for g in range(G):
        ss = sorted(slots[g])
        lcnts[g] = len(ss)
        for j, phys in enumerate(ss):
            l2p[g, j] = phys
    return p2l, l2p, lcnts


def random_topk(R, S, K, G, seed):
    gen = torch.Generator().manual_seed(seed)
    out = torch.zeros(R, S, K, dtype=torch.int64)
    for r in range(R):
        for s in range(S):
            out[r, s] = torch.randperm(G, generator=gen)[:K]
    return out


def check_invariants(topk, phys, p2l, nlp, R, S, K, cap, allow_forced=False):
    assert tuple(phys.shape) == (R, S, K)
    assert bool(p2l.long()[phys.long()].eq(topk).all()), "conservation"
    # one instance per (token, logical): phys ids distinct within a token row
    for r in range(R):
        srt = phys[r].long().sort(dim=1).values
        assert bool((srt[:, 1:] != srt[:, :-1]).all()), "dup slot in a token"
    rows = torch.bincount(phys.long().reshape(-1) // nlp, minlength=R)
    if not allow_forced:
        assert int(rows.max()) <= cap, (int(rows.max()), cap)
    return rows


def main():
    R, S, K, G, nlp, L = 8, 64, 4, 32, 12, 4
    topk = random_topk(R, S, K, G, seed=1234)

    # --- degenerate: no replicas => routing is forced, incidence = fixed ---
    p2l, l2p, lcnts = make_fixed_placement(G, R, nlp)
    for eps in (0.0, 0.25, math.inf):
        phys = loccap_route(topk, p2l, l2p, lcnts, nlp, L, eps)
        cap = R * S * K if math.isinf(eps) else int(math.ceil((1 + eps) * S * K))
        check_invariants(topk, phys, p2l, nlp, R, S, K, cap, allow_forced=True)
        # with lcnts==1 the routing is unique: phys == the expert's only slot
        expect = l2p.long()[topk, 0]
        assert bool(phys.long().eq(expect).all())
    st_fixed = incidence_stats(phys, nlp, L)
    print("fixed placement:", st_fixed)

    # --- replicated: every expert on every node ---------------------------
    p2l, l2p, lcnts = make_replicated_placement(G, R, nlp, L)

    # eps=inf: pure locality => every entry served on the home node
    phys = loccap_route(topk, p2l, l2p, lcnts, nlp, L, math.inf)
    check_invariants(topk, phys, p2l, nlp, R, S, K, R * S * K)
    st = incidence_stats(phys, nlp, L)
    assert st["incidence_remote"] == 0, st
    print("all-replicated eps=inf:", st)

    # eps=0: hard balance => rows_per_rank <= ceil(S*K) exactly (no forced
    # demand possible: every expert has an instance on every node)
    phys0 = loccap_route(topk, p2l, l2p, lcnts, nlp, L, 0.0)
    rows0 = check_invariants(topk, phys0, p2l, nlp, R, S, K, S * K)
    st0 = incidence_stats(phys0, nlp, L)
    print("all-replicated eps=0:", st0, "rows:", rows0.tolist())

    # monotonicity: incidence_remote non-increasing in eps
    prev = None
    for eps in (0.0, 0.125, 0.25, 0.5, 1.0, math.inf):
        ph = loccap_route(topk, p2l, l2p, lcnts, nlp, L, eps)
        st_e = incidence_stats(ph, nlp, L)
        if prev is not None:
            assert st_e["incidence_remote"] <= prev + 1e-9, (eps, st_e, prev)
        prev = st_e["incidence_remote"]
        print(f"eps={eps}: remote={st_e['incidence_remote']} "
              f"imb={st_e['imbalance_max_over_mean']:.3f}")

    # bitwise determinism across calls
    a = loccap_route(topk, p2l, l2p, lcnts, nlp, L, 0.25)
    b = loccap_route(topk, p2l, l2p, lcnts, nlp, L, 0.25)
    assert bool(a.eq(b).all())
    assert route_hash(a) == route_hash(b)

    # skewed demand: one hot expert everywhere; caps must bind, forced
    # overflow must NOT trigger (hot expert has a replica on every node)
    hot = topk.clone()
    hot[:, :, 0] = 7
    # keep per-token distinctness
    for r in range(R):
        for s in range(S):
            row = hot[r, s].tolist()
            if len(set(row)) < K:
                pool = [g for g in range(G) if g not in row]
                seen = set()
                for k in range(K):
                    if row[k] in seen:
                        row[k] = pool.pop()
                    seen.add(row[k])
                hot[r, s] = torch.tensor(row)
    ph = loccap_route(hot, p2l, l2p, lcnts, nlp, L, 0.10)
    cap = int(math.ceil(1.10 * S * K))
    check_invariants(hot, ph, p2l, nlp, R, S, K, cap)
    st_h = incidence_stats(ph, nlp, L)
    print("hot-expert eps=0.10:", st_h)

    print("OK: all LocCap router unit tests passed")


if __name__ == "__main__":
    main()
