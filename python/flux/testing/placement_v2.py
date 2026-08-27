"""PV2 — stateless node-aware greedy expert placement (branch pv2,
2026-08-27; the postdoc heuristic, assessed offline the same day).

The whole placement — replication counts, node assignment, rank
assignment, plan tensors — is a PURE FUNCTION of the per-node demand
histogram hist[NN, G] (derivable from the d[R, G] allgather the plan lane
already pays every iteration). Three consequences the PLL lane cannot
offer:

  1. No batch-size term anywhere: every stage is O(G log G + R*nlp*NN)
     host integer math (~1 ms), vs the PLL solve's Omega(R*S*K) one-hot
     GEMM engine (2-12 ms graphed, growing with budget).
  2. No warm state, no CUDA graph, no device solve: the per-iteration
     lane is one small D2H of d + host tensor math + the H2D of the plan
     tensors on adoption.
  3. SIZING CONTRACT RESTORED (handoff 22 §4): the runtime-adopted
     placement is bit-identical to the setup batch solve (stateless pure
     function of d, and d is fixed per cell), so the driver's
     {resident, batch} sizing envelope is exact again — the premise the
     PLL warm solve broke at r2.

Algorithm (per the 2026-08-27 offline assessment; the affinity variant is
mandatory — blind least-loaded spread loses 99% at qwen 4n):
  counts     EPLB-global greedy: replicate argmax(load/c), hard cap
             c[g] <= NN (one instance per node); integer priority
             (load << 20) // c, ties to lower g.
  placement  exact sequential greedy affinity spread: experts in (share
             desc, g asc) order; each instance to the best node by
             (hist[u, g] desc, accumulated share asc, u asc) among free
             non-hosting nodes; leftover slots backfilled to the
             highest-share non-hosting experts. Pure-python integer
             inner loop (~inst*NN compares) — GIL-bound, immune to the
             OMP-pool contention that makes tiny tensor ops jitter.
  ranks      within-node boustrophedon (snake) over instances sorted
             (share desc, g asc) — finalize_hosts snake semantics.
  tensors    the canonical slot recipe (loccap_semantics
             plan_tensors_from_hosts: each rank's hosted experts in
             ascending expert id occupy slots 0..n-1), produced directly
             from (g, rank) tensors — no hosts-list detour.

Determinism contract: integer arithmetic and stable sorts only — every
rank computes the identical placement from the identical histogram, no
cross-rank machinery. Module imports torch ONLY (file-path importable by
sweeps tooling, the placelambda_fast precedent).
"""

import heapq

import torch

SHARE_BITS = 20  # integer per-instance share scale: (load << 20) // c


def _seg_prefix(sorted_vals, sorted_keys):
    """Exclusive prefix sum of sorted_vals within equal-key runs of
    sorted_keys (both [N], keys sorted). placelambda_fast recipe."""
    csum = torch.cumsum(sorted_vals, 0) - sorted_vals
    N = sorted_vals.numel()
    if N == 0:
        return csum
    idx = torch.arange(N, dtype=torch.int64)
    newgrp = torch.ones(N, dtype=torch.bool)
    newgrp[1:] = sorted_keys[1:] != sorted_keys[:-1]
    base = torch.where(newgrp, csum, torch.zeros_like(csum))
    base = torch.cummax(base, 0).values
    return csum - base


def _run_ordinal(sorted_keys):
    """[N] int64 sorted -> position within its equal-key run (0-based)."""
    N = sorted_keys.numel()
    idx = torch.arange(N, dtype=torch.int64)
    if N == 0:
        return idx
    newgrp = torch.ones(N, dtype=torch.bool)
    newgrp[1:] = sorted_keys[1:] != sorted_keys[:-1]
    starts = torch.where(newgrp, idx, torch.zeros_like(idx))
    starts = torch.cummax(starts, 0).values
    return idx - starts


def pv2_counts(load_e, NN, total_slots):
    """Replication counts [G] int64: EPLB-global greedy argmax(load/c)
    with the node cap c <= NN. Integer priority (load << SHARE_BITS) // c,
    ties to lower g (the heap key embeds g). Exact greedy semantics; at
    G <= 896 the heap runs in well under a millisecond."""
    G = int(load_e.numel())
    assert total_slots >= G, "fewer slots than experts"
    lo = load_e.tolist()
    c = [1] * G
    cap = min(NN, total_slots)  # c can never exceed NN
    heap = [((-(lo[g] << SHARE_BITS), g)) for g in range(G)]
    heapq.heapify(heap)
    extra = total_slots - G
    while extra > 0 and heap:
        _, g = heapq.heappop(heap)
        if c[g] >= cap:
            continue  # saturated on every node: next most popular
        c[g] += 1
        extra -= 1
        if c[g] < cap:
            heapq.heappush(heap, (-((lo[g] << SHARE_BITS) // c[g]), g))
    return torch.tensor(c, dtype=torch.int64)


def pv2_place(hist, c, slots_per_node):
    """Node assignment. hist [NN, G] int64, c [G] int64 -> (ion [G, NN]
    bool, primary [G] int64, spilled int). Exact sequential greedy with
    affinity spread (the offline-assessed prototype semantics): experts
    in (share desc, g asc) order place their c instances one at a time,
    each on the best node by (residual affinity hist[u, g] desc,
    accumulated node share asc, node id asc) among free-slot nodes not
    yet hosting the expert. A final vectorized BACKFILL spends leftover
    slots on the highest-share non-hosting experts (slots are paid for —
    never waste them); `spilled` counts instances the greedy could not
    seat on a distinct node (diagnostic; conservation of the FIRST
    instance is guaranteed and asserted). First instance = primary."""
    NN, G = hist.shape
    load = hist.sum(0)                                      # [G]
    share = (load << SHARE_BITS) // c                       # [G]
    # exact sequential greedy in tuned pure python: ~G*NN + inst*NN plain
    # int compares under the GIL — immune to OMP-pool contention, and the
    # bit-exact semantics of the assessed prototype. Experts processed in
    # (share desc, g asc) order; each instance takes the best node by
    # (affinity desc, accumulated share asc, node id asc) among free
    # non-hosting nodes.
    order = torch.argsort(share * G + (G - 1
                          - torch.arange(G, dtype=torch.int64)),
                          descending=True, stable=True).tolist()
    aff_l = hist.t().tolist()                               # [G][NN]
    c_l = c.tolist()
    share_l = share.tolist()
    free = [slots_per_node] * NN
    loadv = [0] * NN
    pairs_g, pairs_u = [], []
    primary_l = [0] * G
    spilled = 0
    rng_nn = range(NN)
    for g in order:
        affs = aff_l[g]
        sg = share_l[g]
        hosted = []
        for _i in range(c_l[g]):
            bu = -1
            ba = -1
            bl = 0
            for u in rng_nn:
                if free[u] <= 0 or u in hosted:
                    continue
                a = affs[u]
                if a > ba or (a == ba and (bu < 0 or loadv[u] < bl)):
                    bu, ba, bl = u, a, loadv[u]
            if bu < 0:
                spilled += c_l[g] - _i
                break
            hosted.append(bu)
            free[bu] -= 1
            loadv[bu] += sg
        for u in hosted:
            pairs_g.append(g)
            pairs_u.append(u)
        primary_l[g] = hosted[0] if hosted else -1
    assert all(p >= 0 for p in primary_l), "expert with no instance"
    ion = torch.zeros(G, NN, dtype=torch.bool)
    ion[torch.tensor(pairs_g, dtype=torch.int64),
        torch.tensor(pairs_u, dtype=torch.int64)] = True
    primary = torch.tensor(primary_l, dtype=torch.int64)
    # ---- backfill: spend leftover slots (share desc, g asc per node) ----
    if sum(free) > 0:
        node_free = torch.tensor(free, dtype=torch.int64)
        smax = int(share.clamp(max=1 << 32).max()) + 2
        share_c = share.clamp(max=1 << 32)
        cand = (~ion) & (node_free > 0).unsqueeze(0)        # [G, NN]
        gc_, uc_ = cand.nonzero(as_tuple=True)
        if gc_.numel():
            okey = (uc_ * smax + (smax - 1 - share_c[gc_])) * G + gc_
            o = torch.argsort(okey, stable=True)
            u_s = uc_[o]
            pre = _seg_prefix(torch.ones_like(u_s), u_s)
            fit = pre < node_free[u_s]
            ion[gc_[o[fit]], u_s[fit]] = True
    assert bool((ion.any(dim=1)).all()), "expert with no instance"
    return ion, primary, spilled


def pv2_rank_assign(ion, load, c, L):
    """(g, node) instances -> (g_flat, r_flat) global-rank pairs via the
    within-node snake over (share desc, g asc) — finalize_hosts snake
    semantics, tensorized end to end."""
    G, NN = ion.shape
    gg, uu = ion.nonzero(as_tuple=True)
    share = (load << SHARE_BITS) // c                       # [G]
    smax = int(share.max()) + 2
    okey = (uu * smax + (smax - 1 - share[gg])) * G + gg
    order = torch.argsort(okey, stable=True)
    u_s = uu[order]
    pos = _run_ordinal(u_s)
    lane = pos % L
    down = (pos // L) % 2 == 1
    lane = torch.where(down, L - 1 - lane, lane)
    return gg[order], u_s * L + lane


def pv2_plan_tensors(g_flat, r_flat, G, R, nlp):
    """(g, rank) pairs -> (p2l [R*nlp] int32, l2p [G, R] int32, lcnts [G]
    int32) under THE canonical slot recipe (loccap_semantics
    plan_tensors_from_hosts): each rank's hosted experts in ascending
    expert id occupy its slots 0..n-1; l2p columns ascending phys slot."""
    N = int(g_flat.numel())
    p2l = torch.full((R * nlp,), -1, dtype=torch.int32)
    l2p = torch.full((G, R), -1, dtype=torch.int32)
    lcnts = torch.zeros(G, dtype=torch.int64)
    lcnts.index_add_(0, g_flat, torch.ones(N, dtype=torch.int64))
    key_gr = g_flat * R + r_flat
    key_s = torch.sort(key_gr).values
    assert N == 0 or bool((key_s[1:] != key_s[:-1]).all()), "duplicate host"
    idx = torch.arange(N, dtype=torch.int64)
    o_r = torch.argsort(r_flat * G + g_flat, stable=True)
    r_s = r_flat[o_r]
    ordn = _run_ordinal(r_s)
    assert N == 0 or bool((ordn < nlp).all()), "rank over nlp"
    phys_s = r_s * nlp + ordn
    p2l[phys_s] = g_flat[o_r].to(torch.int32)
    phys = torch.empty(N, dtype=torch.int64)
    phys[o_r] = phys_s
    o_g = torch.argsort(g_flat * (R * nlp) + phys, stable=True)
    g_s = g_flat[o_g]
    j = _run_ordinal(g_s)
    l2p[g_s, j] = phys[o_g].to(torch.int32)
    return p2l, l2p, lcnts.to(torch.int32)


def pv2_remote_rows(hist, ion):
    """Marginal wire proxy: rows demanded at nodes without a local
    instance (served remotely under home-if-hosted-else-primary).
    O(NN*G) integer; the pv2 gain/trigger statistic."""
    return int((hist * (~ion).t().long()).sum())


def pv2_solve(hist, L, nlp):
    """One-call PV2 solve from the demand histogram. hist [NN, G] int64
    (host) -> dict(p2l, l2p, lcnts, ion, primary, g_flat, r_flat, stats).
    Pure function — identical hist gives the bit-identical placement on
    every rank and every call."""
    NN, G = hist.shape
    R = NN * L
    hist = hist.long()
    load = hist.sum(0)
    c = pv2_counts(load, NN, R * nlp)
    ion, primary, spilled = pv2_place(hist, c, L * nlp)
    c_real = ion.long().sum(1)          # backfill may exceed intended c
    g_flat, r_flat = pv2_rank_assign(ion, load, c_real.clamp(min=1), L)
    p2l, l2p, lcnts = pv2_plan_tensors(g_flat, r_flat, G, R, nlp)
    return {
        "p2l": p2l, "l2p": l2p, "lcnts": lcnts,
        "ion": ion, "primary": primary,
        "g_flat": g_flat, "r_flat": r_flat,
        "stats": {
            "mode": "pv2",
            "replicas": int(c_real.sum()) - G,
            "spilled": spilled,
            "c_max": int(c_real.max()),
            "remote_rows_predicted": pv2_remote_rows(hist, ion),
        },
    }


def hosts_lists(res, G):
    """[G] sorted rank lists from a pv2_solve result — the pblob/sidecar
    format (COLD path: setup only)."""
    hosts = [[] for _ in range(G)]
    for g, r in zip(res["g_flat"].tolist(), res["r_flat"].tolist()):
        hosts[g].append(r)
    return [sorted(h) for h in hosts]
