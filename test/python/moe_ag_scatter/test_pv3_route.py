"""CPU unit gates for the PV3 rotation water-fill router (branch pv3,
2026-09-14). torch-only, runs on a login node:

  python3 test/python/moe_ag_scatter/test_pv3_route.py

Gates:
  1. constraints: every routing satisfies the paper's constraint 2 in its
     integer form (floor((1-C)q) <= load <= ceil((1+C)q) per replica) and
     conservation, on random placements incl. the degenerate NO-replica
     placement, single-token experts, C = 0, C = 1 and skewed demand.
  2. totality: no stranded demand for any (placement, demand, C) — the
     §3 proof, exercised.
  3. determinism: bit-identical output across calls / devices-as-CPU and
     under a permutation-free re-run; route_hash stable.
  4. locality: with unlimited slack the router never sends a row off-node
     when the home node hosts the expert with room, and remote rows equal
     the locality lower bound max(0, node demand - node capacity) per
     (expert, node) when every replica cap is loose.
  5. aggregate: per-GPU load within (1 +- C) Q_j + n_j (rounded form).
"""
import importlib.util
import os
import random
import sys

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


def rand_hosts(G, R, nlp, NN, L, gen, max_c=None, no_replica=False):
    """Random placement: every expert >= 1 host, at most one instance per
    rank and per node (pv2's c <= NN), per-rank slots <= nlp. Replicas
    are dropped when the chosen node is full; the primary host falls back
    to the globally least-loaded rank (callers size nlp so it has room)."""
    max_c = max_c or NN
    slots = [0] * R
    hosts = []
    for g in range(G):
        c = 1 if no_replica else random.Random(gen + g).randint(1, max_c)
        nodes = random.Random(gen * 7 + g).sample(range(NN), c)
        h = []
        for u in nodes:
            ranks = list(range(u * L, (u + 1) * L))
            random.Random(gen * 13 + g + u).shuffle(ranks)
            for r in ranks:
                if slots[r] < nlp:
                    slots[r] += 1
                    h.append(r)
                    break
        if not h:
            r = min(range(R), key=lambda x: (slots[x], x))
            assert slots[r] < nlp, "test placement out of slots"
            slots[r] += 1
            h.append(r)
        hosts.append(sorted(h))
    return hosts


def slots_for(G, R, NN):
    """nlp large enough that rand_hosts never runs out of slots."""
    return -(-(G * NN) // R) + 1


def rand_topk(R, S, K, G, gen, skew=1.0):
    g = torch.Generator().manual_seed(gen)
    w = torch.rand(G, generator=g) ** skew + 1e-6
    out = torch.empty(R, S, K, dtype=torch.int64)
    for r in range(R):
        out[r] = torch.multinomial(w.expand(S, G), K, replacement=False,
                                   generator=g)
    return out


def check_case(tag, topk, hosts, R, L, nlp, C_num, C_den):
    G = len(hosts)
    p2l, l2p, lcnts = LC.plan_tensors_from_hosts(hosts, R, nlp)
    phys, st = PV3.pv3_route(topk, p2l, l2p, lcnts, nlp, L, C_num, C_den,
                             return_tables=True)
    assert st["c2_over_rows"] == 0 and st["c2_under_rows"] == 0, (tag, st)
    assert st["nonhost_rows"] == 0, (tag, st)
    assert st["c3_rounded_violations"] == 0, (tag, st)
    # conservation + distinct slots per token
    assert bool(p2l.long()[phys.long()].eq(topk).all()), tag
    srt = phys.long().sort(dim=2).values
    assert bool((srt[:, :, 1:] != srt[:, :, :-1]).all()), (tag, "dup slot")
    # determinism
    phys2, _ = PV3.pv3_route(topk, p2l, l2p, lcnts, nlp, L, C_num, C_den)
    assert torch.equal(phys, phys2), (tag, "nondeterministic")
    return phys, st


def test_random_family():
    L = 4
    for NN, S, K, G in [(2, 16, 4, 32), (4, 48, 8, 96),
                        (8, 32, 8, 128), (4, 64, 16, 224)]:
        R = NN * L
        nlp = slots_for(G, R, NN)
        for gen in range(3):
            hosts = rand_hosts(G, R, nlp, NN, L, gen)
            for skew in (1.0, 4.0):
                topk = rand_topk(R, S, K, G, gen + 100, skew)
                for C in [(0, 1), (1, 16), (1, 4), (1, 1)]:
                    check_case(f"rand NN{NN} gen{gen} skew{skew} C{C}",
                               topk, hosts, R, L, nlp, *C)
    print("random family OK")


def test_degenerate():
    L, NN, S, K, G = 4, 4, 40, 8, 64
    R = NN * L
    nlp = slots_for(G, R, NN)
    # no replicas at all: every expert exactly one host
    hosts = rand_hosts(G, R, nlp, NN, L, 5, no_replica=True)
    topk = rand_topk(R, S, K, G, 9, 6.0)
    phys, st = check_case("no-replica", topk, hosts, R, L, nlp, 1, 16)
    # with a single host per expert the routing is forced: load == D
    assert st["replica_ratio_min"] == 1.0 and st["replica_ratio_max"] == 1.0
    # single-token experts with two replicas at C = 0: rounding band
    hosts = rand_hosts(G, R, nlp, NN, L, 6)
    hosts[0] = [0, 4]                                  # e0: two replicas
    topk = torch.zeros(R, 1, 1, dtype=torch.int64)     # one token/rank, e0
    check_case("single-token e0 c2 C0", topk, hosts, R, L, nlp + 1, 0, 1)
    # C = 0 with divisible demand: exact balance (ratio == 1)
    topk = torch.zeros(R, 2, 1, dtype=torch.int64)     # 32 rows, 2 hosts
    _, st = check_case("e0 c2 C0 exact", topk, hosts, R, L, nlp + 1, 0, 1)
    assert st["replica_ratio_min"] == 1.0 == st["replica_ratio_max"], st
    print("degenerate OK")


def test_locality_bound():
    """With loose caps (C = 1 -> U = 2q, Lb = 0), remote rows per expert
    equal sum over nodes of max(0, node demand - capacity on that node),
    where capacity = (#replicas on node) * U — the locality-optimal count."""
    L, NN, S, K, G = 4, 4, 32, 8, 96
    R = NN * L
    nlp = slots_for(G, R, NN)
    hosts = rand_hosts(G, R, nlp, NN, L, 21)
    topk = rand_topk(R, S, K, G, 22, 3.0)
    phys, st = check_case("locality", topk, hosts, R, L, nlp, 1, 1)
    tab = st["tables"]
    U = tab["U"]
    d = st["d"]                                   # [R, G]
    dn = d.view(NN, L, G).sum(1)                  # [NN, G]
    rep = tab["rep"]
    hosted_n = torch.zeros(G, NN, dtype=torch.int64)
    for g in range(G):
        for j in rep[g].tolist():
            if j >= 0:
                hosted_n[g, j // L] += 1
    cap_n = hosted_n.t() * U.unsqueeze(0)         # [NN, G]
    lb = (dn - cap_n).clamp(min=0).sum()
    _, remote = PV3.incidence_remote(phys.long(), nlp, L)
    assert int(lb) <= remote, (int(lb), remote)
    # and no row leaves a node that still had room for its expert
    serve = phys.long() // nlp
    home = (torch.arange(R) // L).view(R, 1, 1)
    off = (serve // L != home)
    # per (home node, g): remote rows > 0 only if node capacity exhausted
    fill_n = torch.zeros(NN, G, dtype=torch.int64)
    for g in range(G):
        for jj, j in enumerate(rep[g].tolist()):
            if j >= 0:
                fill_n[j // L, g] += int(tab["fill"][g, jj])
    for u in range(NN):
        for g in range(G):
            n_off = int(((topk == g) & off & (home == u)).sum())
            if n_off > 0 and hosted_n[g, u] > 0:
                assert fill_n[u, g] == cap_n[u, g], (u, g, n_off)
    print("locality bound OK")


def test_route_hash_and_incidence():
    L, NN, S, K, G = 4, 4, 24, 8, 64
    R = NN * L
    nlp = slots_for(G, R, NN)
    hosts = rand_hosts(G, R, nlp, NN, L, 31)
    topk = rand_topk(R, S, K, G, 32, 2.0)
    phys, _ = check_case("hash", topk, hosts, R, L, nlp, 1, 16)
    h1 = PV3.route_hash(phys)
    phys2, _ = check_case("hash2", topk, hosts, R, L, nlp, 1, 16)
    assert h1 == PV3.route_hash(phys2)
    inc, rem = PV3.incidence_remote(phys.long(), nlp, L)
    ref = LC.incidence_stats(phys, nlp, L)
    assert inc == ref["incidence_remote"], (inc, ref)
    print("hash/incidence OK")


if __name__ == "__main__":
    test_random_family()
    test_degenerate()
    test_locality_bound()
    test_route_hash_and_incidence()
    print("ALL PV3 CPU GATES OK")
