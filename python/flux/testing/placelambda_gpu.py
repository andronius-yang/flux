"""PLACE-lambda / LocCap — the GPU-portable planner+router (OUR optimization
family, distinct from the EPIC/EPLB baselines; the epic driver is only the
harness these arms run on).

This module is the device-executable redesign of the exact python reference
(`loccap_semantics.py` router, `sweeps/predict_placement.py` planner). It is
NOT a transliteration: the reference's exact min-node-cover (BFS over 2^NN
subset masks) and augmenting-path repair cannot run at the K3 showcase scale
(16-32 nodes), so the router here is a bounded-round vectorized algorithm —
a NEW arm (`loccap_gpu`) with its own sidecar entries, never silently
inheriting the python loccap's numbers. Quality is gated offline against the
exact reference (see test/python/moe_ag_scatter/test_placelambda_gpu.py).

Determinism contract (the cross-device same-code oracle): every function is
pure torch, device-agnostic, and uses INTEGER arithmetic only — stable
sorts, integer scatter/index_add (exact under atomics), and order-free
argmax/argmin via composite integer keys decoded from the reduced VALUE
(never from an argmax/argmin index, whose tie behavior is backend-defined).
The offline simulator (sweeps/predict_placement.py) runs this exact file on
CPU; the driver runs it on GPU; outputs must match bit-for-bit — the driver
hard-asserts route/placement hashes against the sidecar, and a mismatch is
a determinism bug, never noise.

Timing contract (SCHEMA rule 5 + the 2026-08-21 placement amendment):
  - loccap_route_gpu is TIMED, per iteration, in-window (plan_ms).
  - build_placement_gpu + placement_moves + place_decision form the
    DYNAMIC placement lane: timed in-window when the arm ablates placement
    ON (place_dynamic=dynamic); under the ideal-stale ablation
    (place_dynamic=static) the solver runs untimed at setup and is
    reported as place_solver_ms.

Module contract: imports torch ONLY (no flux, no relative imports) so
sweeps/predict_placement.py can load it by file path.
"""

import hashlib
import math

import torch

UNASSIGNED = -1
# proportional-fill weights are clamped so the largest-remainder integer
# products stay far from int64 overflow at every deployed scale
_W_CLAMP = 1 << 16


# --------------------------------------------------------------------------
# deterministic primitives
# --------------------------------------------------------------------------

def _largest_remainder_rows(want, budget):
    """Vectorized largest-remainder scale-down. want [N, M] int64 >= 0,
    budget [N] int64 >= 0 -> got [N, M] with row sums == min(row_sum,
    budget). Ties -> lower column (the reference rule). Rows whose sum is
    <= budget pass through unchanged."""
    N, M = want.shape
    tot = want.sum(1)
    over = tot > budget
    if not bool(over.any()):
        return want.clone()
    got = want.clone()
    w = want[over]
    b = budget[over]
    t = tot[over]
    base = w * b.unsqueeze(1) // t.unsqueeze(1)
    frac = w * b.unsqueeze(1) - base * t.unsqueeze(1)  # remainder numerator
    rem = b - base.sum(1)
    # rank columns by (frac desc, col asc): composite key is unique per row
    key = frac * M + (M - 1 - torch.arange(M, device=want.device,
                                           dtype=torch.int64))
    order = torch.argsort(key, dim=1, descending=True, stable=True)
    rankpos = torch.empty_like(order)
    rankpos.scatter_(1, order,
                     torch.arange(M, device=want.device,
                                  dtype=torch.int64).expand_as(order))
    got[over] = base + (rankpos < rem.unsqueeze(1)).long()
    return got


def _min_by_key(key, valid):
    """Order-free masked min over a [N, M] int64 key matrix -> (found [N]
    bool, best_value [N]). The column identity must already be embedded in
    the key's low bits so ties are impossible; callers decode from the
    value."""
    BIG = torch.iinfo(torch.int64).max
    masked = torch.where(valid, key, torch.full_like(key, BIG))
    best, _ = masked.min(dim=1)
    return best < BIG, best


def _run_ordinal(sorted_keys):
    """[E] int64 sorted -> position of each element within its equal-key
    run (0-based)."""
    E = sorted_keys.numel()
    idx = torch.arange(E, device=sorted_keys.device, dtype=torch.int64)
    if E == 0:
        return idx
    newgrp = torch.ones(E, dtype=torch.bool, device=sorted_keys.device)
    newgrp[1:] = sorted_keys[1:] != sorted_keys[:-1]
    starts = torch.where(newgrp, idx, torch.zeros_like(idx))
    starts = torch.cummax(starts, 0).values
    return idx - starts


def instance_tables_gpu(l2p, lcnts, nlp, ranks_per_node, R=None):
    """l2p [G, Cmax] int (Cmax = max instances, may differ from R), lcnts
    [G] int -> (inst_phys_of_rank [G, R] int64 (-1 = none), inst_on_node
    [G, NN] bool, hosted_rg [R, G] bool)."""
    G, Cmax = l2p.shape
    R = Cmax if R is None else R
    dev = l2p.device
    NN = R // ranks_per_node
    l2p_l = l2p.long()
    valid = (torch.arange(Cmax, device=dev).unsqueeze(0)
             < lcnts.long().unsqueeze(1)) & (l2p_l >= 0)
    g_idx, j_idx = valid.nonzero(as_tuple=True)
    phys = l2p_l[g_idx, j_idx]
    r_idx = phys // nlp
    ipr = torch.full((G, R), -1, dtype=torch.int64, device=dev)
    ipr[g_idx, r_idx] = phys
    pair_cnt = torch.zeros(G * R, dtype=torch.int64, device=dev)
    pair_cnt.index_add_(0, g_idx * R + r_idx, torch.ones_like(g_idx))
    assert int(pair_cnt.max()) <= 1, "expert has two instances on one rank"
    hosted_rg = (ipr >= 0).t().contiguous()
    ion = hosted_rg.view(NN, ranks_per_node, G).any(dim=1).t().contiguous()
    return ipr, ion, hosted_rg


# --------------------------------------------------------------------------
# LocCap GPU router (the TIMED per-iteration lane)
# --------------------------------------------------------------------------

def loccap_route_gpu(topk_all, p2l, l2p, lcnts, nlp, ranks_per_node, eps,
                     repair_rounds=2):
    """[R, S, K] logical expert ids -> ([R, S, K] int32 physical slots,
    stats dict). Bounded-round vectorized LocCap:

      tier 1  own-rank quota (largest remainder within cap)
      tier 2  home-node fill: node-aggregated demand distributed across
              hosting ranks proportional to residual capacity (largest
              remainder), one clip pass per rank, then split back per
              source by canonical prefix intervals
      tier 3  per-token GREEDY node cover in <= min(K, NN)+2 rounds
              (capacity-aware coverability; preference: max cover, then
              home node, then lower node id), rank choice within a node
              proportional to residual with a per-rank clip
      forced  leftovers to the globally least-loaded hosting rank
              (counted in forced_overflow / over_cap_rows)
      repair  `repair_rounds` vectorized passes moving over-cap entries to
              under-cap hosts (prefer the token's home node, then a node
              the token already touches, then least-loaded)

    cap = ceil((1+eps) * S * K); eps=inf -> pure locality (unbounded cap).
    Deterministic integer math throughout: identical inputs on any device
    produce the identical routing.
    """
    R, S, K = topk_all.shape
    dev = topk_all.device
    G = int(lcnts.numel())
    L = ranks_per_node
    assert R % L == 0
    NN = R // L
    topk = topk_all.long()
    ipr, ion, hosted_rg = instance_tables_gpu(l2p.to(dev), lcnts.to(dev),
                                              nlp, L, R=R)
    node_of_rank = torch.arange(R, device=dev, dtype=torch.int64) // L

    total_rows = R * S * K
    cap = total_rows if math.isinf(eps) else int(math.ceil((1.0 + eps) * S * K))
    cap_vec = torch.full((R,), cap, dtype=torch.int64, device=dev)

    flat = topk.reshape(R, S * K)
    assert bool(((flat >= 0) & (flat < G)).all()), "expert ids out of range"
    d = torch.bincount(
        (torch.arange(R, device=dev, dtype=torch.int64).unsqueeze(1) * G
         + flat).reshape(-1), minlength=R * G).view(R, G)

    # ---- tier 1: own-rank quota -----------------------------------------
    q1 = _largest_remainder_rows(d * hosted_rg.long(), cap_vec)
    load = q1.sum(1)

    # ---- tier 2: home-node proportional fill ----------------------------
    rd = d - q1                                             # [R, G]
    nh = ion.t()[node_of_rank]                              # [R, G]
    t2_want = rd * nh.long()
    t2_nlg = t2_want.view(NN, L, G)
    D = t2_nlg.sum(1)                                       # [NN, G]
    resid = (cap_vec - load).clamp(min=0)
    resid_node = resid.view(NN, L)
    hosts_nl = hosted_rg.view(NN, L, G).permute(0, 2, 1)    # [NN, G, L]
    w2 = (resid_node.clamp(max=_W_CLAMP).unsqueeze(1)
          * hosts_nl.long())                                # [NN, G, L]
    alloc = _largest_remainder_rows(
        (w2 * D.unsqueeze(-1)).reshape(NN * G, L),
        D.reshape(-1),
    ).view(NN, G, L)
    # clip each rank's total grant to its residual (largest remainder over
    # experts, tie lower g); freed demand falls through to tier 3
    alloc = _largest_remainder_rows(
        alloc.permute(0, 2, 1).reshape(NN * L, G),
        resid.reshape(-1),
    ).view(NN, L, G).permute(0, 2, 1).contiguous()          # [NN, G, L]

    # per-source split of the node grant (canonical src-ascending prefix)
    src_pref = torch.cumsum(t2_nlg, dim=1) - t2_nlg         # excl. prefix
    cumA = torch.cumsum(alloc, dim=-1)                      # [NN, G, L]
    lo_b = torch.cat([torch.zeros(NN, G, 1, dtype=torch.int64, device=dev),
                      cumA[..., :-1]], dim=-1)
    src_lo = src_pref.permute(0, 2, 1).unsqueeze(3)         # [NN,G,Lsrc,1]
    src_hi = ((src_pref + t2_nlg).permute(0, 2, 1).unsqueeze(3))
    A2 = (torch.minimum(src_hi, cumA.unsqueeze(2))
          - torch.maximum(src_lo, lo_b.unsqueeze(2))
          ).clamp(min=0)                                    # [NN,G,Ls,Ld]
    load = load + alloc.sum(1).reshape(-1)

    # ---- materialize tiers 1+2 (canonical entry order) ------------------
    # canonical order: src asc, expert asc, token asc (flat order within
    # one (src, g) is token-major because k varies fastest and a token's
    # experts are distinct — the reference's own argument).
    ent_src = (torch.arange(R, device=dev, dtype=torch.int64)
               .unsqueeze(1).expand(R, S * K).reshape(-1))
    ent_g = flat.reshape(-1)
    canon_key = ent_src * G + ent_g
    order = torch.argsort(canon_key, stable=True)
    ord_in_seg = _run_ordinal(canon_key[order])

    # per (src, g) target sequence: own rank (tier-1 quota + any tier-2
    # self grant, which is structurally 0) first, then the other home-node
    # ranks ascending
    src_local = torch.arange(R, device=dev, dtype=torch.int64) % L
    A2_src = A2.permute(0, 2, 1, 3).reshape(R, G, L)        # [Rsrc, G, Ld]
    own_extra = torch.gather(
        A2_src, 2, src_local.view(R, 1, 1).expand(R, G, 1)).squeeze(2)
    amounts = A2_src.scatter(
        2, src_local.view(R, 1, 1).expand(R, G, 1),
        torch.zeros(R, G, 1, dtype=torch.int64, device=dev))
    own_amt = q1 + own_extra                                # [R, G]
    node_base = (torch.arange(R, device=dev, dtype=torch.int64) // L) * L
    others = torch.argsort(
        (torch.arange(L, device=dev, dtype=torch.int64).unsqueeze(0)
         == src_local.unsqueeze(1)).long(), dim=1, stable=True)  # own last
    perm = torch.cat([src_local.unsqueeze(1), others[:, :L - 1]], dim=1)
    tgt_ranks = node_base.unsqueeze(1) + perm               # [R, L] global
    tgt_amounts = torch.cat(
        [own_amt.unsqueeze(2),
         torch.gather(amounts, 2,
                      perm[:, 1:].unsqueeze(1).expand(R, G, L - 1))],
        dim=2)                                              # [R, G, L]
    assigned_total = tgt_amounts.sum(2)                     # [R, G]

    seq_ranks = torch.repeat_interleave(
        tgt_ranks.unsqueeze(1).expand(R, G, L).reshape(-1),
        tgt_amounts.reshape(-1))
    in_quota = ord_in_seg < assigned_total.reshape(-1)[canon_key[order]]
    phys_flat = torch.full((R * S * K,), UNASSIGNED, dtype=torch.int64,
                           device=dev)
    ent_pos = order[in_quota]
    assert int(in_quota.sum()) == int(seq_ranks.numel())
    phys_flat[ent_pos] = ipr[ent_g[ent_pos], seq_ranks]

    # ---- tier 3: bounded-round greedy node cover ------------------------
    forced_overflow = 0
    cover_rounds = min(K, NN) + 2
    rounds_used = 0
    hp_r, hp_g = hosted_rg.nonzero(as_tuple=True)           # (rank, expert)
    resid = (cap_vec - load).clamp(min=0)
    tok_of_entry = (ent_src * S
                    + (torch.arange(R * S * K, device=dev,
                                    dtype=torch.int64) // K) % S)
    home_of_entry = node_of_rank[ent_src]
    nn_ar = torch.arange(NN, device=dev, dtype=torch.int64)
    for _round in range(cover_rounds):
        rem = (phys_flat == UNASSIGNED).nonzero(as_tuple=True)[0]
        if rem.numel() == 0:
            break
        rounds_used = _round + 1
        # capacity-aware coverability per (expert, node)
        hrm = torch.full((G * NN,), -1, dtype=torch.int64, device=dev)
        hrm.scatter_reduce_(0, hp_g * NN + node_of_rank[hp_r],
                            resid[hp_r], reduce="amax", include_self=True)
        coverable = (hrm.view(G, NN) > 0) & ion             # [G, NN]
        # per-token coverage counts over the remaining experts
        tok_ids = tok_of_entry[rem]
        uniq_tok, tok_inv = torch.unique(tok_ids, return_inverse=True)
        T = uniq_tok.numel()
        cnt = torch.zeros(T * NN, dtype=torch.int64, device=dev)
        cnt.index_add_(0, (tok_inv.unsqueeze(1) * NN + nn_ar).reshape(-1),
                       coverable[ent_g[rem]].long().reshape(-1))
        cnt = cnt.view(T, NN)
        home_t = node_of_rank[uniq_tok // S]
        is_home = nn_ar.unsqueeze(0) == home_t.unsqueeze(1)
        key = (cnt * 2 + is_home.long()) * NN + (NN - 1 - nn_ar)
        best, _ = key.max(dim=1)
        n_star = NN - 1 - (best % NN)
        covers = (best // NN) >= 2                          # cnt >= 1
        n_of_entry = n_star[tok_inv]
        sel = coverable[ent_g[rem], n_of_entry] & covers[tok_inv]
        e_sel = rem[sel]
        if e_sel.numel() == 0:
            break
        g_sel = ent_g[e_sel]
        n_sel = n_of_entry[sel]
        # per-(expert, node) demand -> hosting ranks proportional to
        # residual, then a per-rank clip; ungranted entries retry next round
        gn = g_sel * NN + n_sel
        dem = torch.bincount(gn, minlength=G * NN)
        hosts_gnl = hosted_rg.view(NN, L, G).permute(2, 0, 1)  # [G, NN, L]
        w3 = (resid.view(NN, L).clamp(max=_W_CLAMP).unsqueeze(0)
              * hosts_gnl.long())                           # [G, NN, L]
        a3 = _largest_remainder_rows(
            (w3.reshape(G * NN, L) * dem.reshape(-1, 1)), dem)
        a3 = _largest_remainder_rows(
            a3.view(G, NN, L).permute(1, 2, 0).reshape(NN * L, G),
            resid,
        ).view(NN, L, G).permute(2, 0, 1).reshape(G * NN, L)
        okey = gn * (total_rows + 1) + e_sel
        o3 = torch.argsort(okey, stable=True)
        ord3 = _run_ordinal(gn[o3])
        take = ord3 < a3.sum(1)[gn[o3]]
        glob_rank = ((torch.arange(G * NN, device=dev, dtype=torch.int64)
                      % NN).unsqueeze(1) * L
                     + torch.arange(L, device=dev,
                                    dtype=torch.int64).unsqueeze(0))
        seq3 = torch.repeat_interleave(glob_rank.reshape(-1),
                                       a3.reshape(-1))
        e_take = e_sel[o3[take]]
        assert int(take.sum()) == int(seq3.numel())
        phys_flat[e_take] = ipr[ent_g[e_take], seq3]
        load = load + torch.bincount(seq3, minlength=R)
        resid = (cap_vec - load).clamp(min=0)

    # ---- forced assignment ----------------------------------------------
    rem = (phys_flat == UNASSIGNED).nonzero(as_tuple=True)[0]
    if rem.numel():
        key = (load.unsqueeze(0) * R
               + torch.arange(R, device=dev,
                              dtype=torch.int64).unsqueeze(0)).expand(G, R)
        found, best = _min_by_key(key, ipr >= 0)
        g_rem = ent_g[rem]
        assert bool(found[g_rem].all()), "expert with no instance anywhere"
        r_forced = best[g_rem] % R
        phys_flat[rem] = ipr[g_rem, r_forced]
        add = torch.bincount(r_forced, minlength=R)
        over_add = ((load + add - cap_vec).clamp(min=0)
                    - (load - cap_vec).clamp(min=0))
        forced_overflow += int(over_add.sum())
        load = load + add

    # ---- repair rounds ---------------------------------------------------
    moved_total = 0
    for _rep in range(repair_rounds):
        if not bool((load > cap_vec).any()):
            break
        serve_rank = phys_flat // nlp
        excess = (load - cap_vec).clamp(min=0)
        over_mask = excess > 0
        eids = over_mask[serve_rank].nonzero(as_tuple=True)[0]
        if eids.numel() == 0:
            break
        rk = serve_rank[eids]
        o4 = torch.argsort(rk * (total_rows + 1) + eids, stable=True)
        ordr = _run_ordinal(rk[o4])
        seg_n = torch.bincount(rk, minlength=R)[rk[o4]]
        # move the LAST excess[r] entries per over-cap rank (canonical)
        pick = ordr >= (seg_n - excess[rk[o4]])
        cand = eids[o4[pick]]
        if cand.numel() == 0:
            break
        g_c = ent_g[cand]
        tok_c = tok_of_entry[cand]
        touched = torch.zeros(R * S, NN, dtype=torch.bool, device=dev)
        touched[tok_of_entry, serve_rank // L] = True
        host_valid = (ipr[g_c] >= 0) & (load < cap_vec).unsqueeze(0)
        r_grid = torch.arange(R, device=dev, dtype=torch.int64).unsqueeze(0)
        n_grid = r_grid // L
        tier = torch.full((cand.numel(), R), 2, dtype=torch.int64,
                          device=dev)
        tier[n_grid.expand_as(tier)
             == home_of_entry[cand].unsqueeze(1)] = 0
        tch = touched[tok_c].gather(1, n_grid.expand(cand.numel(), R))
        tier = torch.where((tier == 2) & tch, torch.ones_like(tier), tier)
        keym = (tier * (R * (total_rows + 1))
                + load.unsqueeze(0) * R + r_grid)
        found, best = _min_by_key(keym, host_valid)
        dst = best % R
        # destination capacity: first slack[dst] movers per dst (canonical)
        o5 = torch.argsort(dst * (total_rows + 1) + cand, stable=True)
        ordd = _run_ordinal(dst[o5])
        grant = found[o5] & (ordd < (cap_vec - load).clamp(min=0)[dst[o5]])
        mv = o5[grant]
        if mv.numel() == 0:
            break
        e_mv, d_mv = cand[mv], dst[mv]
        src_r = serve_rank[e_mv]
        phys_flat[e_mv] = ipr[ent_g[e_mv], d_mv]
        load = (load - torch.bincount(src_r, minlength=R)
                + torch.bincount(d_mv, minlength=R))
        moved_total += int(mv.numel())

    # ---- post-asserts ----------------------------------------------------
    assert not bool((phys_flat == UNASSIGNED).any())
    phys = phys_flat.view(R, S, K)
    p2l_l = p2l.to(dev).long()
    assert bool(p2l_l[phys].eq(topk).all()), "conservation violated"
    rows = torch.bincount(phys_flat // nlp, minlength=R)
    stats = {
        "cap": cap,
        "forced_overflow": forced_overflow,
        "cover_rounds": rounds_used,
        "repair_moved": moved_total,
        "rows_max": int(rows.max()),
        "over_cap_rows": int((rows - cap_vec).clamp(min=0).sum()),
    }
    return phys.to(torch.int32), stats


# --------------------------------------------------------------------------
# Sender-local LocCap (the kernel algorithm; user relaxation 2026-08-21)
# --------------------------------------------------------------------------

def loccap_route_sl(topk_all, p2l, l2p, lcnts, nlp, ranks_per_node, eps,
                    rounds=3):
    """Sender-LOCAL LocCap — the algorithm the fused CUDA kernel implements
    (user relaxation ruling 2026-08-21). [R, S, K] -> ([R, S, K] int32,
    stats).

    Contract: every shared quantity is a pure ORDER-INDEPENDENT integer
    function of the allgathered demand histogram d[R, G] (the plan_comm
    collective epic already pays) — quota and share tables agree across
    ranks with no determinism machinery, exactly like the transport's
    cumsum/splits. Every PER-ROW decision belongs to exactly one source
    rank: rank src assigns only its own S*K entries, consuming its
    pre-partitioned SHARES of each destination's capacity; receivers learn
    realized counts from the wire metadata as always. A production rank
    computes only its own row; this reference computes all rows at once
    (per-source math is independent, so the batch IS the per-rank result).

    Tiers 1-2 are identical to loccap_route_gpu (they were already pure
    functions of d). Tier 3 replaces live global capacity feedback with
    per-(source, destination-rank) shares of the post-tier-2 residual
    (largest-remainder proportional to each source's leftover hosted
    demand): per-token greedy min-node-cover unchanged — only the capacity
    state it consults is now the source's own shares. No repair pass: caps
    hold by construction of the shares; what the shares cannot place is
    forced (accounted). The known cost vs the global algorithm: no
    DONATION — one source's unused share cannot be claimed by another
    (the offline quality gate measures this gap).
    """
    R, S, K = topk_all.shape
    dev = topk_all.device
    G = int(lcnts.numel())
    L = ranks_per_node
    assert R % L == 0
    NN = R // L
    topk = topk_all.long()
    ipr, ion, hosted_rg = instance_tables_gpu(l2p.to(dev), lcnts.to(dev),
                                              nlp, L, R=R)
    node_of_rank = torch.arange(R, device=dev, dtype=torch.int64) // L

    total_rows = R * S * K
    cap = total_rows if math.isinf(eps) else int(math.ceil((1.0 + eps) * S * K))
    cap_vec = torch.full((R,), cap, dtype=torch.int64, device=dev)

    flat = topk.reshape(R, S * K)
    assert bool(((flat >= 0) & (flat < G)).all()), "expert ids out of range"
    d = torch.bincount(
        (torch.arange(R, device=dev, dtype=torch.int64).unsqueeze(1) * G
         + flat).reshape(-1), minlength=R * G).view(R, G)

    # ---- tiers 1+2: identical shared-table math to loccap_route_gpu -----
    q1 = _largest_remainder_rows(d * hosted_rg.long(), cap_vec)
    load = q1.sum(1)
    rd = d - q1
    nh = ion.t()[node_of_rank]
    t2_want = rd * nh.long()
    t2_nlg = t2_want.view(NN, L, G)
    D = t2_nlg.sum(1)
    resid = (cap_vec - load).clamp(min=0)
    hosts_nl = hosted_rg.view(NN, L, G).permute(0, 2, 1)
    w2 = (resid.view(NN, L).clamp(max=_W_CLAMP).unsqueeze(1)
          * hosts_nl.long())
    alloc = _largest_remainder_rows(
        (w2 * D.unsqueeze(-1)).reshape(NN * G, L),
        D.reshape(-1)).view(NN, G, L)
    alloc = _largest_remainder_rows(
        alloc.permute(0, 2, 1).reshape(NN * L, G),
        resid.reshape(-1),
    ).view(NN, L, G).permute(0, 2, 1).contiguous()
    src_pref = torch.cumsum(t2_nlg, dim=1) - t2_nlg
    cumA = torch.cumsum(alloc, dim=-1)
    lo_b = torch.cat([torch.zeros(NN, G, 1, dtype=torch.int64, device=dev),
                      cumA[..., :-1]], dim=-1)
    A2 = (torch.minimum(src_pref.permute(0, 2, 1).unsqueeze(3)
                        + t2_nlg.permute(0, 2, 1).unsqueeze(3),
                        cumA.unsqueeze(2))
          - torch.maximum(src_pref.permute(0, 2, 1).unsqueeze(3),
                          lo_b.unsqueeze(2))).clamp(min=0)
    load = load + alloc.sum(1).reshape(-1)

    # materialize tiers 1+2 (canonical entry order; same as the global fn)
    ent_src = (torch.arange(R, device=dev, dtype=torch.int64)
               .unsqueeze(1).expand(R, S * K).reshape(-1))
    ent_g = flat.reshape(-1)
    canon_key = ent_src * G + ent_g
    order = torch.argsort(canon_key, stable=True)
    ord_in_seg = _run_ordinal(canon_key[order])
    src_local = torch.arange(R, device=dev, dtype=torch.int64) % L
    A2_src = A2.permute(0, 2, 1, 3).reshape(R, G, L)
    own_extra = torch.gather(
        A2_src, 2, src_local.view(R, 1, 1).expand(R, G, 1)).squeeze(2)
    amounts = A2_src.scatter(
        2, src_local.view(R, 1, 1).expand(R, G, 1),
        torch.zeros(R, G, 1, dtype=torch.int64, device=dev))
    own_amt = q1 + own_extra
    node_base = (torch.arange(R, device=dev, dtype=torch.int64) // L) * L
    others = torch.argsort(
        (torch.arange(L, device=dev, dtype=torch.int64).unsqueeze(0)
         == src_local.unsqueeze(1)).long(), dim=1, stable=True)
    perm = torch.cat([src_local.unsqueeze(1), others[:, :L - 1]], dim=1)
    tgt_ranks = node_base.unsqueeze(1) + perm
    tgt_amounts = torch.cat(
        [own_amt.unsqueeze(2),
         torch.gather(amounts, 2,
                      perm[:, 1:].unsqueeze(1).expand(R, G, L - 1))],
        dim=2)
    granted2 = tgt_amounts.sum(2)
    seq_ranks = torch.repeat_interleave(
        tgt_ranks.unsqueeze(1).expand(R, G, L).reshape(-1),
        tgt_amounts.reshape(-1))
    in_quota = ord_in_seg < granted2.reshape(-1)[canon_key[order]]
    phys_flat = torch.full((R * S * K,), UNASSIGNED, dtype=torch.int64,
                           device=dev)
    ent_pos = order[in_quota]
    phys_flat[ent_pos] = ipr[ent_g[ent_pos], seq_ranks]

    # ---- tier-3 SHARE tables (pure functions of d) ----------------------
    resid3 = (cap_vec - load).clamp(min=0)                  # [R]
    leftover = (d - granted2).clamp(min=0)                  # [R, G]
    # source weight toward destination r = its leftover demand for experts
    # hosted on r (clamped; only proportions matter)
    # fp64 matmul: every partial sum is an exact integer < 2^53, so the
    # result is order-independent and bit-exact on any device (int64
    # matmul has no CUDA kernel)
    w3 = ((leftover.clamp(max=_W_CLAMP).double()
           @ hosted_rg.double().t()).long()).clamp(max=_W_CLAMP)
    myshare = _largest_remainder_rows(
        (w3 * resid3.unsqueeze(0)).t().contiguous(),        # rows = dst
        resid3,
    ).t().contiguous()                                      # [Rsrc, Rdst]

    # ---- tier 3: per-source greedy cover on own shares ------------------
    forced_overflow = 0
    rounds_used = 0
    tok_of_entry = (ent_src * S
                    + (torch.arange(R * S * K, device=dev,
                                    dtype=torch.int64) // K) % S)
    nn_ar = torch.arange(NN, device=dev, dtype=torch.int64)
    hosts_gnl = hosted_rg.view(NN, L, G).permute(2, 0, 1)   # [G, NN, L]
    for _round in range(rounds):
        rem = (phys_flat == UNASSIGNED).nonzero(as_tuple=True)[0]
        if rem.numel() == 0:
            break
        rounds_used = _round + 1
        # coverable[src, g, n]: expert g hosted on node n at a rank where
        # THIS source still holds share
        pos = (myshare > 0).view(R, NN, L)                  # [Rs, NN, L]
        covOK = (pos.unsqueeze(1)
                 & hosts_gnl.unsqueeze(0)).any(-1)          # [Rs, G, NN]
        # per-token cover counts over remaining experts
        tok_ids = tok_of_entry[rem]
        cnt = torch.zeros(R * S * NN, dtype=torch.int64, device=dev)
        cnt.index_add_(0, (tok_ids.unsqueeze(1) * NN + nn_ar).reshape(-1),
                       covOK[ent_src[rem], ent_g[rem]].long().reshape(-1))
        cnt = cnt.view(R * S, NN)
        home_t = (torch.arange(R * S, device=dev, dtype=torch.int64)
                  // (L * S))
        is_home = nn_ar.unsqueeze(0) == home_t.unsqueeze(1)
        key = (cnt * 2 + is_home.long()) * NN + (NN - 1 - nn_ar)
        best, _ = key.max(dim=1)
        n_star = NN - 1 - (best % NN)
        covers = (best // NN) >= 2
        n_of_entry = n_star[tok_of_entry[rem]]
        sel = (covOK[ent_src[rem], ent_g[rem], n_of_entry]
               & covers[tok_of_entry[rem]])
        e_sel = rem[sel]
        if e_sel.numel() == 0:
            break
        # rank within the chosen node: the source's max-share hosting rank
        # (tie lower); key embeds rank for order-free decode
        s_sel, g_sel, n_sel = ent_src[e_sel], ent_g[e_sel], n_of_entry[sel]
        l_ar = torch.arange(L, device=dev, dtype=torch.int64)
        sh_cand = myshare.view(R, NN, L)[s_sel, n_sel]      # [E, L]
        hosted_cand = hosts_gnl[g_sel, n_sel]               # [E, L]
        keyr = (sh_cand.clamp(max=_W_CLAMP) * L + (L - 1 - l_ar))
        found_r, best_r = _min_by_key(-keyr, hosted_cand & (sh_cand > 0))
        dst = n_sel * L + (L - 1 - ((-best_r) % L))
        ok = found_r
        # grant: first myshare[src, dst] candidates per (src, dst) in
        # canonical order (within one source — sender-local sequencing)
        e_ok = e_sel[ok]
        dst_ok = dst[ok]
        sd = ent_src[e_ok] * R + dst_ok
        o3 = torch.argsort(sd * (total_rows + 1) + e_ok, stable=True)
        ord3 = _run_ordinal(sd[o3])
        take = ord3 < myshare.reshape(-1)[sd[o3]]
        e_take = e_ok[o3[take]]
        d_take = dst_ok[o3[take]]
        if e_take.numel() == 0:
            break
        phys_flat[e_take] = ipr[ent_g[e_take], d_take]
        spent = torch.bincount(ent_src[e_take] * R + d_take,
                               minlength=R * R).view(R, R)
        myshare = myshare - spent
        load = load + spent.sum(0)

    # ---- forced: estimated least-loaded hosting rank --------------------
    rem = (phys_flat == UNASSIGNED).nonzero(as_tuple=True)[0]
    if rem.numel():
        keyf = (load.unsqueeze(0) * R
                + torch.arange(R, device=dev,
                               dtype=torch.int64).unsqueeze(0)).expand(G, R)
        found, best = _min_by_key(keyf, ipr >= 0)
        g_rem = ent_g[rem]
        assert bool(found[g_rem].all()), "expert with no instance anywhere"
        r_forced = best[g_rem] % R
        phys_flat[rem] = ipr[g_rem, r_forced]
        add = torch.bincount(r_forced, minlength=R)
        over_add = ((load + add - cap_vec).clamp(min=0)
                    - (load - cap_vec).clamp(min=0))
        forced_overflow += int(over_add.sum())
        load = load + add

    assert not bool((phys_flat == UNASSIGNED).any())
    phys = phys_flat.view(R, S, K)
    p2l_l = p2l.to(dev).long()
    assert bool(p2l_l[phys].eq(topk).all()), "conservation violated"
    rows = torch.bincount(phys_flat // nlp, minlength=R)
    stats = {
        "cap": cap,
        "forced_overflow": forced_overflow,
        "cover_rounds": rounds_used,
        "repair_moved": 0,
        "rows_max": int(rows.max()),
        "over_cap_rows": int((rows - cap_vec).clamp(min=0).sum()),
    }
    return phys.to(torch.int32), stats


# --------------------------------------------------------------------------
# PLACE-lambda placement solver (untimed-or-timed per the ablation toggle)
# --------------------------------------------------------------------------

def build_placement_gpu(topk_all, ranks_per_node, nlp, G,
                        balance_tol=0.20, max_moves=None):
    """Batch-observed PLACE-lambda: [R, S, K] routing -> hosts (per-expert
    sorted rank lists) + stats. Steepest-descent Stage A (single best move
    per step over the full [G, NN] integer delta table), bundle-credit
    Stage B with LCM-scaled integer scores (reference tie order: score
    desc, exact desc, then LOWER g / LOWER u), reference Stage C
    (host-side LPT; deterministic pure-python floats). Deterministic on any
    device.

    Input semantics (user design 2026-08-21): the in-pipeline solver
    observes ROUTING, not oracle pools — node u's token pool is its own
    ranks' batch tokens. The oracle-pool ceiling stays available offline
    through sweeps/predict_placement.py's original planner.
    """
    R, S, K = topk_all.shape
    dev = topk_all.device
    L = ranks_per_node
    NN = R // L
    epn = G // R
    topk = topk_all.long()
    LCM = 720720  # lcm(1..16)
    assert K <= 16, "bundle-credit LCM covers K <= 16"

    ent_g = topk.reshape(-1)                                # [R*S*K]
    E = ent_g.numel()
    ent_tok = torch.arange(R * S, device=dev,
                           dtype=torch.int64).repeat_interleave(K)
    ent_node = ent_tok // (L * S)
    load_e = torch.bincount(ent_g, minlength=G)
    g_ar = torch.arange(G, device=dev, dtype=torch.int64)
    nn_ar = torch.arange(NN, device=dev, dtype=torch.int64)
    e_ar = torch.arange(E, device=dev, dtype=torch.int64)

    # Stage A: steepest-descent rooted-lambda refinement ------------------
    primary = (g_ar // epn) // L
    cnt = torch.zeros(R * S, NN, dtype=torch.int64, device=dev)
    cnt.view(-1).index_add_(0, ent_tok * NN + primary[ent_g],
                            torch.ones_like(ent_tok))
    node_load = torch.zeros(NN, dtype=torch.int64, device=dev)
    node_load.index_add_(0, primary, load_e)
    node_cnt = torch.bincount(primary, minlength=NN)
    total = int(load_e.sum())
    load_cap = int(math.floor((1.0 + balance_tol) * total / NN))
    cnt_cap = L * epn + (L * (nlp - epn)) // 2

    if max_moves is None:
        max_moves = 8 * G
    moves = 0
    while moves < max_moves:
        z = (cnt[ent_tok] == 0).long()                      # [E, NN]
        one_at_a = (cnt[ent_tok, primary[ent_g]] == 1).long()
        pay = torch.zeros(G * NN, dtype=torch.int64, device=dev)
        pay.index_add_(0, (ent_g.unsqueeze(1) * NN + nn_ar).reshape(-1),
                       z.reshape(-1))
        # remove the b == home-pool contribution (a token never pays to
        # visit its own node)
        pay_excl = torch.zeros(G * NN, dtype=torch.int64, device=dev)
        pay_excl.index_add_(0, ent_g * NN + ent_node, z[e_ar, ent_node])
        pay = (pay - pay_excl).view(G, NN)
        gain_full = torch.zeros(G, dtype=torch.int64, device=dev)
        gain_full.index_add_(0, ent_g, one_at_a)
        gain_home = torch.zeros(G, dtype=torch.int64, device=dev)
        home_is_a = ent_node == primary[ent_g]
        gain_home.index_add_(0, ent_g[home_is_a], one_at_a[home_is_a])
        delta = pay - (gain_full - gain_home).unsqueeze(1)  # [G, NN]
        valid = nn_ar.unsqueeze(0) != primary.unsqueeze(1)
        valid &= (node_cnt + 1 <= cnt_cap).unsqueeze(0)
        valid &= (node_load.unsqueeze(0) + load_e.unsqueeze(1)) <= load_cap
        neg = valid & (delta < 0)
        if not bool(neg.any()):
            break
        flat_idx = g_ar.unsqueeze(1) * NN + nn_ar
        found, best = _min_by_key(
            (delta * (G * NN) + flat_idx).reshape(1, -1), neg.reshape(1, -1))
        gi = int(best[0]) % (G * NN)
        g_mv, b_mv = gi // NN, gi % NN
        a_mv = int(primary[g_mv])
        t_sel = ent_tok[ent_g == g_mv]
        cnt[t_sel, a_mv] -= 1
        cnt[t_sel, b_mv] += 1
        primary[g_mv] = b_mv
        node_load[a_mv] -= load_e[g_mv]
        node_load[b_mv] += load_e[g_mv]
        node_cnt[a_mv] -= 1
        node_cnt[b_mv] += 1
        moves += 1

    # Stage B: bundle-credit replication ----------------------------------
    inst_nodes = torch.zeros(G, NN, dtype=torch.bool, device=dev)
    inst_nodes[g_ar, primary] = True
    slots_left = (L * nlp - node_cnt).clone()
    spent = 0
    credit_c = torch.zeros(K + 1, dtype=torch.int64, device=dev)
    for c in range(2, K + 1):
        credit_c[c] = LCM // c
    while True:
        val = cnt[ent_tok, primary[ent_g]].clamp(max=K)
        H = torch.zeros(G * NN * (K + 1), dtype=torch.int64, device=dev)
        H.index_add_(0, (ent_g * NN + ent_node) * (K + 1) + val,
                     torch.ones_like(val))
        H = H.view(G, NN, K + 1)
        exact = H[:, :, 1]
        credit = (H[:, :, 2:] * credit_c[2:].view(1, 1, -1)).sum(-1)
        score = exact * LCM + credit                        # [G, NN]
        valid = (~inst_nodes) & (slots_left > 0).unsqueeze(0) & (score > 0)
        if not bool(valid.any()):
            break
        # tie order: score desc, exact desc, then lower g, lower u —
        # staged masked reductions (the ranges overflow one packed key)
        NEG = torch.full_like(score, -1)
        s_best = torch.where(valid, score, NEG).max()
        m2 = valid & (score == s_best)
        e_best = torch.where(m2, exact, NEG).max()
        m3 = m2 & (exact == e_best)
        flat_idx = g_ar.unsqueeze(1) * NN + nn_ar
        BIG = torch.full_like(flat_idx, G * NN)
        cand = int(torch.where(m3, flat_idx, BIG).min())
        g_r, u_r = cand // NN, cand % NN
        t_sel = ent_tok[(ent_g == g_r) & (ent_node == u_r)]
        p_r = int(primary[g_r])
        cnt[t_sel, p_r] -= 1
        cnt[t_sel, u_r] += 1
        inst_nodes[g_r, u_r] = True
        slots_left[u_r] -= 1
        spent += 1

    # Stage C: LPT rank assignment (host, reference logic) ----------------
    inst_nodes_h = inst_nodes.cpu()
    load_h = load_e.cpu().tolist()
    n_inst_of = inst_nodes_h.sum(1).tolist()
    per_node_inst = [[] for _ in range(NN)]
    for g in range(G):
        share = load_h[g] / n_inst_of[g]
        for u in range(NN):
            if bool(inst_nodes_h[g, u]):
                per_node_inst[u].append((g, share))
    hosts = [[] for _ in range(G)]
    for u in range(NN):
        ranks = list(range(u * L, (u + 1) * L))
        rload = {r: 0.0 for r in ranks}
        rslots = {r: 0 for r in ranks}
        for g, share in sorted(per_node_inst[u], key=lambda t: (-t[1], t[0])):
            cand = [r for r in ranks if rslots[r] < nlp and r not in hosts[g]]
            assert cand, (u, g, "no rank slot left — raise nlp")
            r = min(cand, key=lambda r: (rload[r], r))
            hosts[g].append(r)
            rload[r] += share
            rslots[r] += 1
    hosts = [sorted(h) for h in hosts]

    visited = cnt > 0
    home_col = (torch.arange(R * S, device=dev, dtype=torch.int64)
                // (L * S))
    incidence = int(visited.sum()) - int(
        visited.gather(1, home_col.unsqueeze(1)).sum())
    return {
        "hosts": hosts,
        "primary_node": primary.cpu().tolist(),
        "stats": {
            "mode": "nodeaware_gpu",
            "fm_moves": moves,
            "balance_tol": balance_tol,
            "replica_slots_spent": spent,
            "cnt_cap": cnt_cap,
            "incidence_remote_predicted": incidence,
        },
    }


# --------------------------------------------------------------------------
# Trigger lane: incidence bound + placement diff + decision
# --------------------------------------------------------------------------

def incidence_lb(topk_all, inst_on_node, ranks_per_node):
    """Capacity-free greedy-cover bound on token-node incidence under a
    placement. [R, S, K] + [G, NN] -> total remote node visits (int).
    <= min(K, NN) vectorized rounds; the cheap per-iteration trigger
    statistic."""
    R, S, K = topk_all.shape
    dev = topk_all.device
    L = ranks_per_node
    NN = inst_on_node.shape[1]
    topk = topk_all.long()
    ion = inst_on_node.to(dev)
    T = R * S
    tok = torch.arange(T, device=dev, dtype=torch.int64).repeat_interleave(K)
    g = topk.reshape(-1)
    home = tok // (L * S)
    rem = ~ion[g, home]                                     # home is free
    picks = torch.zeros(T, dtype=torch.int64, device=dev)
    nn_ar = torch.arange(NN, device=dev, dtype=torch.int64)
    for _ in range(min(K, NN)):
        idx = rem.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break
        cnt = torch.zeros(T * NN, dtype=torch.int64, device=dev)
        cnt.index_add_(0,
                       (tok[idx].unsqueeze(1) * NN + nn_ar).reshape(-1),
                       ion[g[idx]].long().reshape(-1))
        cnt = cnt.view(T, NN)
        key = cnt * NN + (NN - 1 - nn_ar)
        best, _ = key.max(dim=1)
        n_star = NN - 1 - (best % NN)
        has = (best // NN) > 0
        picks += has.long()
        served = ion[g[idx], n_star[tok[idx]]] & has[tok[idx]]
        rem[idx[served]] = False
    return int(picks.sum())


def placement_moves(hosts_cur, hosts_new):
    """Instance-level diff between two placements -> (adds, removes) as
    sorted (expert, rank) lists — the 'what has to move' calculation of the
    dynamic-placement decision (the weight dispatch itself is not modeled
    in this port)."""
    adds, removes = [], []
    assert len(hosts_new) == len(hosts_cur)
    for gexp in range(len(hosts_cur)):
        cur, new = set(hosts_cur[gexp]), set(hosts_new[gexp])
        adds.extend((gexp, r) for r in sorted(new - cur))
        removes.extend((gexp, r) for r in sorted(cur - new))
    return adds, removes


def hosts_to_inst_on_node(hosts, R, ranks_per_node, device="cpu"):
    ion = torch.zeros(len(hosts), R // ranks_per_node, dtype=torch.bool,
                      device=device)
    for gexp, hs in enumerate(hosts):
        for r in hs:
            ion[gexp, r // ranks_per_node] = True
    return ion


def place_decision(topk_all, hosts_cur, hosts_new, ranks_per_node,
                   gain_threshold_ppm=50_000):
    """The dynamic-placement decision (TIMED when the ablation is on):
    compare the incidence bound of the resident placement against the
    fresh solve on this batch; trigger when the fractional gain (in ppm)
    clears the threshold. Returns integer facts only."""
    R = topk_all.shape[0]
    dev = topk_all.device
    lb_cur = incidence_lb(
        topk_all, hosts_to_inst_on_node(hosts_cur, R, ranks_per_node, dev),
        ranks_per_node)
    lb_new = incidence_lb(
        topk_all, hosts_to_inst_on_node(hosts_new, R, ranks_per_node, dev),
        ranks_per_node)
    gain_ppm = 0 if lb_cur == 0 else (lb_cur - lb_new) * 1_000_000 // lb_cur
    adds, removes = placement_moves(hosts_cur, hosts_new)
    return {
        "lb_cur": lb_cur,
        "lb_new": lb_new,
        "gain_ppm": int(gain_ppm),
        "moves_add": len(adds),
        "moves_remove": len(removes),
        "trigger": int(gain_ppm >= gain_threshold_ppm),
    }


def placement_hash(hosts) -> int:
    """Order-stable digest of a placement (sidecar identity companion to
    route_hash)."""
    h = hashlib.sha256()
    h.update(repr([sorted(x) for x in hosts]).encode())
    return int.from_bytes(h.digest()[:8], "little", signed=True)
