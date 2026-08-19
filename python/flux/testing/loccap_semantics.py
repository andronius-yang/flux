"""LocCap: per-token replica-selection router under compute caps.

Chooses, for every (token, topk-entry) demand, WHICH physical replica serves
it, preferring (1) the token's home rank, then (2) its home node, then (3) a
minimum set of remote nodes (exact set cover), subject to a per-rank compute
cap kappa = ceil((1 + eps) * S * K) rows. eps is the balance-slack knob:
eps=0 is hard balance (locality only inside the fair share), eps=inf is pure
locality (spill only for experts with no local instance).

The objective this minimizes is token-node incidence: under the node-dedup
transports (hier_compress Mode-2, both directions) the inter-node wire bytes
of a token are proportional to the number of distinct non-home nodes serving
it, NOT the number of remote experts, so the tier-3 remainder is solved as a
per-token min-NODE-cover, never per-entry greedily.

Module contract: imports torch ONLY (no flux, no relative imports) so that
sweeps/predict_placement.py can load this file by path and the offline
simulator provably runs the exact code the GPU driver runs — predicted
incidence must equal realized incidence bit-for-bit on the same inputs.

All processing is in canonical orders (expert asc, source rank asc, token
asc); given identical inputs every rank computes the identical routing
(guarded by route_hash in the driver's plan-hash all-gather).

Capacity semantics: caps are soft only for FORCED demand — an entry whose
expert has no capacity-positive instance anywhere is assigned to the
globally least-loaded hosting rank and counted in forced_overflow (the
theory's forced-spill regime). rows_per_rank <= cap + forced overflow is a
hard post-assert.
"""

import hashlib
import math

import torch

REMOTE = -1  # tier-1/2 materialization sentinel: entry falls through to tier 3


def _instance_tables(p2l, l2p, lcnts, nlp, ranks_per_node, G, R):
    """inst_phys_of_rank [G, R] int64: the physical slot of expert g's
    instance on rank r, or -1 (an expert appears at most once per rank).
    inst_on_node [G, NN] bool."""
    NN = R // ranks_per_node
    inst_phys_of_rank = torch.full((G, R), -1, dtype=torch.int64)
    inst_on_node = torch.zeros(G, NN, dtype=torch.bool)
    l2p_l = l2p.long()
    for g in range(G):
        C = int(lcnts[g])
        for j in range(C):
            phys = int(l2p_l[g, j])
            r = phys // nlp
            assert inst_phys_of_rank[g, r] < 0, (
                f"expert {g}: two instances on rank {r} (phys "
                f"{int(inst_phys_of_rank[g, r])} and {phys})"
            )
            inst_phys_of_rank[g, r] = phys
            inst_on_node[g, r // ranks_per_node] = True
    return inst_phys_of_rank, inst_on_node


def _largest_remainder(want, budget):
    """Deterministic largest-remainder scale-down of integer demands `want`
    (list of int) to sum exactly `budget`. Ties -> lower index."""
    total = sum(want)
    assert total > budget >= 0
    scaled = [w * budget / total for w in want]
    base = [int(math.floor(s)) for s in scaled]
    rem = budget - sum(base)
    order = sorted(range(len(want)),
                   key=lambda i: (-(scaled[i] - base[i]), i))
    for i in order[:rem]:
        base[i] += 1
    assert sum(base) == budget
    return base


def loccap_route(topk_all, p2l, l2p, lcnts, nlp, ranks_per_node, eps):
    """[R, S, K] logical expert ids -> [R, S, K] int32 physical slot ids.

    See module docstring. Deterministic pure-host math; every rank must call
    with identical inputs.
    """
    R, S, K = topk_all.shape
    G = int(lcnts.numel())
    L = ranks_per_node
    assert R % L == 0
    NN = R // L
    topk = topk_all.cpu().long()
    lcnts_l = lcnts.cpu().long()
    inst_phys_of_rank, inst_on_node = _instance_tables(
        p2l.cpu(), l2p.cpu(), lcnts_l, nlp, L, G, R)
    node_of_rank = torch.arange(R, dtype=torch.int64) // L

    total_rows = R * S * K
    if math.isinf(eps):
        cap = total_rows  # unbounded: pure locality
    else:
        cap = int(math.ceil((1.0 + eps) * S * K))

    # Per-source demand histogram d[src, g].
    flat = topk.reshape(R, S * K)
    assert bool(((flat >= 0) & (flat < G)).all()), "expert ids out of range"
    d = torch.zeros(R, G, dtype=torch.int64)
    for src in range(R):
        d[src] = torch.bincount(flat[src], minlength=G)

    load = [0] * R                 # committed rows per rank
    forced_overflow = [0] * R      # rows above cap from forced demand

    # ---- Tier 1: own-rank quota (closed form per rank) -------------------
    # q1[src][g] for experts hosted on src itself.
    q1 = torch.zeros(R, G, dtype=torch.int64)
    for r in range(R):
        hosted = [g for g in range(G)
                  if int(inst_phys_of_rank[g, r]) >= 0 and int(d[r, g]) > 0]
        want = [int(d[r, g]) for g in hosted]
        H = sum(want)
        if H <= cap:
            got = want
        else:
            got = _largest_remainder(want, cap)
        for g, q in zip(hosted, got):
            q1[r, g] = q
        load[r] = sum(got)

    # ---- Tier 2: home-node water-fill (canonical g asc, src asc) ---------
    # q2[src][g] = list of (rank, count), ranks in the token's home node.
    q2 = [[None] * G for _ in range(R)]
    for u in range(NN):
        ranks_u = list(range(u * L, (u + 1) * L))
        for g in range(G):
            cand_all = [r for r in ranks_u if int(inst_phys_of_rank[g, r]) >= 0]
            if not cand_all:
                continue
            for src in ranks_u:
                rd = int(d[src, g]) - int(q1[src, g])
                if rd <= 0:
                    continue
                cand = [r for r in cand_all if r != src]
                alloc = []
                while rd > 0 and cand:
                    # least-loaded candidate, tie -> lower rank id
                    r_star = min(cand, key=lambda r: (load[r], r))
                    room = cap - load[r_star]
                    if room <= 0:
                        cand.remove(r_star)
                        continue
                    take = min(rd, room)
                    alloc.append((r_star, take))
                    load[r_star] += take
                    rd -= take
                    if load[r_star] >= cap:
                        cand.remove(r_star)
                if alloc:
                    q2[src][g] = alloc

    # ---- Materialization of tiers 1-2 (segmented prefix per (src, g)) ----
    # Tokens of (src, g) in ascending token order consume: own-rank quota,
    # then the node allocations in tier-2 order, then REMOTE.
    phys_all = torch.full((R, S, K), REMOTE, dtype=torch.int64)
    for src in range(R):
        g_of = flat[src]                          # [S*K]
        order = torch.argsort(g_of, stable=True)  # (g, then entry) canonical:
        # entries within one g arrive token-major because k varies fastest and
        # each token's experts are distinct -> per-g sequence is token asc.
        seg_sizes = d[src]
        seg_start = 0
        phys_flat = phys_all[src].reshape(-1)
        for g in range(G):
            n = int(seg_sizes[g])
            if n == 0:
                continue
            idx = order[seg_start:seg_start + n]
            seg_start += n
            targets = []
            if int(q1[src, g]) > 0:
                targets.append((src, int(q1[src, g])))
            if q2[src][g] is not None:
                targets.extend(q2[src][g])
            pos = 0
            for r, c in targets:
                phys_flat[idx[pos:pos + c]] = int(inst_phys_of_rank[g, r])
                pos += c
            # remainder stays REMOTE
        assert seg_start == S * K

    # ---- Tier 3: per-token exact min-node-cover for the remainder --------
    # Bucket unresolved tokens by (home node, frozen unresolved expert set):
    # the cover structure depends only on that signature; capacity is applied
    # chunkwise so the fill stays deterministic and approximately balanced.
    residual = [cap - load[r] for r in range(R)]

    def node_ranks_of(g, n):
        return [r for r in range(n * L, (n + 1) * L)
                if int(inst_phys_of_rank[g, r]) >= 0]

    # Vectorized bucket build (bit-identical to the per-token loop it
    # replaces: same membership, same within-bucket canonical (src, s)
    # order via stable sorts; bucket iteration order comes from
    # sorted(buckets) below either way).
    buckets = {}  # (home_node, tuple(experts)) -> list of (src, s)
    rem_mask = phys_all.eq(REMOTE)                    # [R, S, K]
    srcs, toks = rem_mask.any(dim=2).nonzero(as_tuple=True)
    if srcs.numel():
        sel = topk[srcs, toks]                        # [N, K]
        gsel = torch.where(rem_mask[srcs, toks], sel,
                           torch.full_like(sel, G))  # pad sorts LAST
        gsorted = gsel.sort(dim=1).values
        keyexp = torch.where(gsorted < G, gsorted + 1,
                             torch.zeros_like(gsorted))  # pad field = 0
        keys = torch.cat([node_of_rank[srcs].unsqueeze(1), keyexp], dim=1)
        order = torch.arange(srcs.numel())
        for col in range(K, -1, -1):                  # lexicographic stable
            order = order[torch.argsort(keys[order, col], stable=True)]
        ks = keys[order]
        newgrp = torch.ones(order.numel(), dtype=torch.bool)
        if order.numel() > 1:
            newgrp[1:] = (ks[1:] != ks[:-1]).any(dim=1)
        srcs_o, toks_o = srcs[order].tolist(), toks[order].tolist()
        starts = newgrp.nonzero(as_tuple=True)[0].tolist() + [order.numel()]
        for bi in range(len(starts) - 1):
            lo, hi = starts[bi], starts[bi + 1]
            row = ks[lo].tolist()
            gs = tuple(int(x) - 1 for x in row[1:] if int(x) > 0)
            buckets[(int(row[0]), gs)] = list(
                zip(srcs_o[lo:hi], toks_o[lo:hi]))

    def best_cover(home, gs):
        """Exact min cover of expert set gs by non-home nodes with currently
        capacity-positive instances; among min covers prefer (max min-residual,
        lexicographically smallest node tuple). Experts coverable by no
        capacity-positive node are returned as forced."""
        m = len(gs)
        cover_mask = []
        for n in range(NN):
            if n == home:
                cover_mask.append(0)
                continue
            bm = 0
            for i, g in enumerate(gs):
                if bool(inst_on_node[g, n]) and any(
                        residual[r] > 0 for r in node_ranks_of(g, n)):
                    bm |= 1 << i
            cover_mask.append(bm)
        reachable = 0
        for bm in cover_mask:
            reachable |= bm
        forced = [gs[i] for i in range(m) if not (reachable >> i) & 1]
        target = reachable
        if target == 0:
            return [], forced
        # BFS over subset masks: depth = cover size, so the first depth that
        # reaches `target` is the exact minimum cover.
        best = None
        # iterate masks in increasing popcount via BFS-like expansion
        frontier = {0: ()}
        seen = {0}
        depth = 0
        while best is None and depth <= NN:
            depth += 1
            nxt = {}
            for msk, path in frontier.items():
                for n in range(NN):
                    bm = cover_mask[n] & target
                    if bm == 0 or (bm | msk) == msk:
                        continue
                    nm = msk | bm
                    np_ = path + (n,)
                    if nm == target:
                        cand = np_
                        if best is None or _cover_pref(cand, gs, residual,
                                                      node_ranks_of) < \
                           _cover_pref(best, gs, residual, node_ranks_of):
                            best = cand
                    elif nm not in seen:
                        # keep lexicographically smallest path per mask
                        if nm not in nxt or np_ < nxt[nm]:
                            nxt[nm] = np_
            for nm in nxt:
                seen.add(nm)
            frontier = nxt
        assert best is not None
        return list(best), forced

    CHUNK = 256
    for key in sorted(buckets.keys()):
        home, gs = key
        entries = buckets[key]  # canonical: built in (src asc, token asc)
        pos = 0
        while pos < len(entries):
            chunk = entries[pos:pos + CHUNK]
            nodes, forced = best_cover(home, gs)
            # rank choice per expert: least-loaded hosting rank in its
            # chosen node (first covering node in `nodes` order), tie lower.
            rank_of_g = {}
            for g in gs:
                if g in forced:
                    hosts = [r for r in range(R)
                             if int(inst_phys_of_rank[g, r]) >= 0]
                    r_star = min(hosts, key=lambda r: (load[r], r))
                    rank_of_g[g] = (r_star, True)
                    continue
                placed = False
                for n in nodes:
                    cands = [r for r in node_ranks_of(g, n) if residual[r] > 0]
                    if cands:
                        r_star = min(cands, key=lambda r: (load[r], r))
                        rank_of_g[g] = (r_star, False)
                        placed = True
                        break
                assert placed, (g, nodes, home)
            # shrink chunk to the tightest capacity among chosen ranks
            need = {}
            for g in gs:
                r, is_forced = rank_of_g[g]
                if not is_forced:
                    need[r] = need.get(r, 0) + 1
            take = len(chunk)
            for r, per_tok in need.items():
                if per_tok > 0:
                    take = min(take, residual[r] // per_tok)
            take = max(take, 1)  # progress guarantee: first token may overflow
            for (src, s) in chunk[:take]:
                mask = phys_all[src, s].eq(REMOTE)
                for k in mask.nonzero(as_tuple=True)[0].tolist():
                    g = int(topk[src, s, k])
                    r, is_forced = rank_of_g[g]
                    phys_all[src, s, k] = int(inst_phys_of_rank[g, r])
                    load[r] += 1
                    if residual[r] > 0:
                        residual[r] -= 1
                    else:
                        forced_overflow[r] += 1
            pos += take

    # ---- Rebalancing repair pass -----------------------------------------
    # Greedy tier quotas can strand demand when capacity is tight (an
    # over-cap rank while an under-cap rank hosts the same expert). Move
    # entries from over-cap ranks to eligible under-cap ranks, preferring
    # moves that reduce (home node) or preserve (already-touched node)
    # incidence. Sweeps until no eligible move remains; leftover excess is
    # genuinely forced (theory's forced-spill regime).
    _repair_overflow(phys_all, topk, inst_phys_of_rank, nlp, L, cap, load, R)

    # ---- Post-asserts ----------------------------------------------------
    assert not bool(phys_all.eq(REMOTE).any()), "unrouted entries remain"
    p2l_l = p2l.cpu().long()
    assert bool(p2l_l[phys_all].eq(topk).all()), (
        "conservation violated: some entry routed to a slot of a different "
        "logical expert")
    rows = torch.bincount(phys_all.reshape(-1) // nlp, minlength=R)
    for r in range(R):
        assert int(rows[r]) == load[r], (r, int(rows[r]), load[r])
    return phys_all.to(torch.int32)


def _repair_overflow(phys_all, topk, inst_phys_of_rank, nlp, L, cap, load, R):
    hosts_of_g = {}

    def hosts(g):
        if g not in hosts_of_g:
            hosts_of_g[g] = [r for r in range(R)
                             if int(inst_phys_of_rank[g, r]) >= 0]
        return hosts_of_g[g]

    progress = True
    while progress:
        progress = False
        over = sorted((r for r in range(R) if load[r] > cap),
                      key=lambda r: (-load[r], r))
        if not over:
            break
        under = {r for r in range(R) if load[r] < cap}
        if not under:
            break
        serve_rank_pass = phys_all // nlp  # refreshed once per sweep; the
        # per-entry mutations below are tracked through `load` and the
        # entries list is re-derived next sweep
        for r in over:
            ent = (serve_rank_pass == r).nonzero()  # [(src,s,k)] canonical
            for src, s, k in ent.tolist():
                if load[r] <= cap:
                    break
                g = int(topk[src, s, k])
                cands = [r2 for r2 in hosts(g) if load[r2] < cap]
                if not cands:
                    continue
                tok_nodes = set((phys_all[src, s] // nlp // L).tolist())

                def pref(r2):
                    n2 = r2 // L
                    if n2 == src // L:
                        tier = 0          # home node: incidence improves
                    elif n2 in tok_nodes:
                        tier = 1          # already-touched node: no change
                    else:
                        tier = 2          # new node: incidence may grow
                    return (tier, load[r2], r2)

                r2 = min(cands, key=pref)
                phys_all[src, s, k] = int(inst_phys_of_rank[g, r2])
                load[r] -= 1
                load[r2] += 1
                progress = True
            if load[r] > cap:
                continue

    # Augmenting phase: single-hop moves cannot fix chains (over-cap rank's
    # experts hosted only on at-cap ranks that could themselves donate).
    # Unit-entry transportation is integrally feasible iff fractionally
    # feasible, so BFS augmenting paths over the rank graph (edge r_i->r_j:
    # some entry served on r_i has an instance on r_j) restore load <= cap
    # exactly whenever possible. Deterministic: canonical BFS order and
    # canonical witness entries.
    def _augment_once(r0):
        parent = {r0: None}       # rank -> (prev_rank, witness (src,s,k))
        frontier = [r0]
        goal = None
        # phys_all is immutable during the BFS (mutations happen only on
        # the walk-back below), so derive the serving map ONCE per call.
        # The per-expert first-witness extraction is vectorized
        # (bit-identical: flat nonzero order IS the canonical (src, s, k)
        # order; stable argsort keeps the first canonical entry per expert;
        # the unique-expert iteration is ascending = sorted(seen_g)).
        serve_rank = phys_all // nlp
        serve_flat = serve_rank.reshape(-1)
        topk_flat = topk.reshape(-1)
        S_, K_ = topk.shape[1], topk.shape[2]
        while frontier and goal is None:
            nxt = []
            for ri in frontier:
                flat = (serve_flat == ri).nonzero(as_tuple=True)[0]
                gs_all = topk_flat[flat]
                o = torch.argsort(gs_all, stable=True)
                gs_s = gs_all[o]
                first = torch.ones_like(gs_s, dtype=torch.bool)
                if gs_s.numel() > 1:
                    first[1:] = gs_s[1:] != gs_s[:-1]
                uniq = gs_s[first].tolist()
                wits = flat[o[first]].tolist()
                for g, wf in zip(uniq, wits):
                    witness = (wf // (S_ * K_), (wf // K_) % S_, wf % K_)
                    for rj in hosts(int(g)):
                        if rj in parent or rj == ri:
                            continue
                        parent[rj] = (ri, witness)
                        if load[rj] < cap:
                            goal = rj
                            break
                        nxt.append(rj)
                    if goal is not None:
                        break
                if goal is not None:
                    break
            frontier = nxt
        if goal is None:
            return False
        # walk back, moving one witness entry per hop (dst gains, src loses)
        rj = goal
        while parent[rj] is not None:
            ri, (src, s, k) = parent[rj]
            g = int(topk[src, s, k])
            assert phys_all[src, s, k] // nlp == ri
            phys_all[src, s, k] = int(inst_phys_of_rank[g, rj])
            load[ri] -= 1
            load[rj] += 1
            rj = ri
        return True

    stuck = set()
    while True:
        over = sorted((r for r in range(R)
                       if load[r] > cap and r not in stuck),
                      key=lambda r: (-load[r], r))
        if not over:
            break
        if not _augment_once(over[0]):
            stuck.add(over[0])   # genuinely forced excess


def _cover_pref(nodes, gs, residual, node_ranks_of):
    """Order key among equal-size covers: prefer larger worst-case residual
    (negated for min), then lexicographically smallest node tuple."""
    worst = None
    for n in nodes:
        for g in gs:
            rs = node_ranks_of(g, n)
            if rs:
                m = max(residual[r] for r in rs)
                worst = m if worst is None else min(worst, m)
    return (-(worst if worst is not None else 0), tuple(nodes))


def plan_tensors_from_hosts(hosts_of_expert, R, nlp):
    """[G] lists of host rank ids -> (p2l [R*nlp] int32, l2p [G, R] int32,
    lcnts [G] int32). THE slot recipe shared by the offline simulator and
    build_nodeaware_plan: each rank's hosted experts in ascending expert id
    occupy its slots 0..n-1; l2p columns in ascending physical slot id
    (no master semantics — the PINNED_MASTERS=False family)."""
    G = len(hosts_of_expert)
    p2l = torch.full((R * nlp,), -1, dtype=torch.int32)
    l2p = torch.full((G, R), -1, dtype=torch.int32)
    lcnts = torch.zeros(G, dtype=torch.int32)
    per_rank = [[] for _ in range(R)]
    for g, hosts in enumerate(hosts_of_expert):
        assert len(set(hosts)) == len(hosts), (g, hosts)
        for r in hosts:
            per_rank[r].append(g)
    slot_of = {}
    for r in range(R):
        gs = sorted(per_rank[r])
        assert len(gs) <= nlp, (r, len(gs), nlp)
        for j, g in enumerate(gs):
            phys = r * nlp + j
            p2l[phys] = g
            slot_of[(g, r)] = phys
    for g, hosts in enumerate(hosts_of_expert):
        slots = sorted(slot_of[(g, r)] for r in hosts)
        lcnts[g] = len(slots)
        for j, phys in enumerate(slots):
            l2p[g, j] = phys
    return p2l, l2p, lcnts


def d6_route(topk_all, l2p, lcnts):
    """The EPIC D6 baseline: source rank src sends ALL its tokens for
    expert g to instance src mod lcnts[g]. [R, S, K] -> [R, S, K] int32."""
    R, S, K = topk_all.shape
    topk = topk_all.cpu().long()
    l2p_l = l2p.cpu().long()
    lcnts_l = lcnts.cpu().long()
    phys = torch.zeros(R, S, K, dtype=torch.int64)
    for src in range(R):
        j = src % lcnts_l[topk[src]]          # [S, K]
        phys[src] = torch.gather(l2p_l[topk[src].reshape(-1)], 1,
                                 j.reshape(-1, 1)).reshape(S, K)
    return phys.to(torch.int32)


def evensplit_route(topk_all, l2p, lcnts):
    """The brute-force even-sharding baseline (user-specified 2026-08-19):
    per expert l, take its tokens in canonical global order (src-major,
    token asc — the same per-expert order reroute_expand produces) and
    split CONTIGUOUSLY into lcnts[l] equal chunks (largest-remainder: the
    first `rem` chunks get one extra), chunk j -> instance j (l2p column
    j). 'First half to replica 0, next half to replica 1.' Source- and
    token-oblivious; distinct from BOTH d6's per-source modding and
    UltraEP's per-(source,expert) quota interleave. [R, S, K] -> int32."""
    R, S, K = topk_all.shape
    topk = topk_all.cpu().long()
    l2p_l = l2p.cpu().long()
    lcnts_l = lcnts.cpu().long().tolist()
    G = int(lcnts.numel())
    phys_flat = torch.zeros(R * S * K, dtype=torch.int64)
    flat_g = topk.reshape(-1)
    for g in range(G):
        idx = (flat_g == g).nonzero(as_tuple=True)[0]  # canonical order
        n = idx.numel()
        if n == 0:
            continue
        C = int(lcnts_l[g])
        base, rem = divmod(n, C)
        pos = 0
        for j in range(C):
            c = base + (1 if j < rem else 0)
            if c:
                phys_flat[idx[pos:pos + c]] = int(l2p_l[g, j])
                pos += c
    return phys_flat.reshape(R, S, K).to(torch.int32)


def incidence_stats(phys_all, nlp, ranks_per_node):
    """Routing -> incidence + load facts (all derived from phys_all only).

    incidence_remote is THE objective: sum over tokens of distinct non-home
    serving nodes (= inter-node wire rows per direction under node-dedup
    transports). Token (src, s) is homed on rank src.
    """
    R, S, K = phys_all.shape
    L = ranks_per_node
    NN = R // L
    serve_rank = phys_all.long() // nlp                     # [R, S, K]
    serve_node = serve_rank // L
    on_node = torch.zeros(R, S, NN, dtype=torch.bool)
    on_node.scatter_(2, serve_node, True)
    nodes_per_token = on_node.sum(dim=2)                    # [R, S]
    home_node = (torch.arange(R) // L).view(R, 1, 1)
    remote_per_token = on_node.sum(dim=2) - on_node.gather(
        2, home_node.expand(R, S, 1)).squeeze(2).long()
    rows = torch.bincount(serve_rank.reshape(-1), minlength=R).tolist()
    mean_rows = sum(rows) / R
    return {
        "incidence_total": int(nodes_per_token.sum()),
        "incidence_remote": int(remote_per_token.sum()),
        "mean_nodes_per_token": float(remote_per_token.float().mean()),
        "rows_per_rank": rows,
        "imbalance_max_over_mean": (max(rows) / mean_rows) if mean_rows else 0.0,
    }


def route_hash(phys_all) -> int:
    """Order-stable digest of a routing (the plan_hash recipe)."""
    h = hashlib.sha256()
    h.update(phys_all.contiguous().to(torch.int32).numpy().tobytes())
    return int.from_bytes(h.digest()[:8], "little", signed=True)
