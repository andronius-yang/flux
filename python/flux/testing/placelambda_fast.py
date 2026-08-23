"""PLACE-lambda FAST — the per-iteration expert-placement planner
(session 8.22.placefast; OUR optimization family, sits beside
placelambda_gpu.py which stays the exact reference).

Why this module exists: the exact batch-observed solver
(`placelambda_gpu.build_placement_gpu`) moves ONE expert per
steepest-descent step and spends ONE replica slot per histogram round,
with a host sync in every step — measured 179-336 ms at 4n and 1.6 s at
32n (login A100), plus a 126-1064 ms trigger decision, against a ~13 ms
layer0+1 latency. The objective math is right; the SCHEDULE is
undeployable per iteration.

This module keeps the exact solver's integer objective (the token-level
pay/gain delta table and the LCM bundle-credit score are lifted verbatim)
and replaces the schedule:

  Stage A  batched passes: one [G, NN] delta table per pass, then a
           deterministic capacity-checked batch admit (sorted prefix
           sums, destination caps frozen at pass start — conservative,
           optional best-delta damping against batch staleness), FIXED
           pass budget.
  Stage B  batched replica passes: one [G, NN, K+1] histogram per pass,
           per-node top-slots admit (two stable sorts give the reference
           tie order score desc / exact desc / lower g / lower u),
           FIXED pass budget.
  Stage C  OFF the per-iteration path: token-node incidence depends only
           on node-level placement, so the rank-level assignment runs
           only when a re-placement actually triggers (`finalize_hosts`,
           vectorized snake by default; the reference host LPT as an
           option).
  decision fixed-round masked greedy cover + a near-free demand-drift
           prefilter (`drift_ppm`).

ZERO-D2H CONTRACT (user directive 2026-08-22): the solve and the
decision statistics are fully shape-static device programs — no
`.item()`/`bool()`/`int()` on device tensors, no `nonzero()`, no
data-dependent shapes, fixed pass counts (an empty pass is a masked
no-op). Every op is CUDA-graph-capturable. The ONLY host readback is
the caller's final consumption of the decision verdict, which can be
deferred one iteration (`place_decision_fast(..., to_host=False)`).
Cold paths (affinity seed, finalize_hosts, host stats) may sync and say
so in their docstrings.

Seeds: "contig" (the reference seed), "affinity" (per-node demand
greedy balanced fill — cold path), or a WARM resident primary
(scenario 2: distribution drift, weights already resident). Hysteresis:
`move_margin` admits a move only when its delta clears the margin;
`keep_bonus` makes Stage B prefer replica instances already resident —
both bound weight-movement bytes.

Determinism contract: identical to placelambda_gpu — pure torch, integer
arithmetic, stable sorts, order-free composite-key argmin/argmax decoded
from values; identical inputs on any device produce identical placements.
Module imports torch ONLY (file-path importable by sweeps tooling).
"""

import math

import torch

UNASSIGNED = -1
LCM16 = 720720  # lcm(1..16) — bundle-credit integer scale, K <= 16
_I64MAX = torch.iinfo(torch.int64).max


# --------------------------------------------------------------------------
# deterministic primitives
# --------------------------------------------------------------------------

def _min_by_key(key, valid):
    """Order-free masked min over the last dim -> (found, best_value).
    Column identity must already be embedded in the key's low bits."""
    masked = torch.where(valid, key, torch.full_like(key, _I64MAX))
    best, _ = masked.min(dim=-1)
    return best < _I64MAX, best


def _run_ordinal(sorted_keys):
    """[N] int64 sorted -> position of each element within its equal-key
    run (0-based)."""
    N = sorted_keys.numel()
    idx = torch.arange(N, device=sorted_keys.device, dtype=torch.int64)
    if N == 0:
        return idx
    newgrp = torch.ones(N, dtype=torch.bool, device=sorted_keys.device)
    newgrp[1:] = sorted_keys[1:] != sorted_keys[:-1]
    starts = torch.where(newgrp, idx, torch.zeros_like(idx))
    starts = torch.cummax(starts, 0).values
    return idx - starts


def _seg_prefix(sorted_vals, sorted_keys):
    """Exclusive prefix sum of sorted_vals within equal-key runs of
    sorted_keys (both [N], keys sorted)."""
    csum = torch.cumsum(sorted_vals, 0) - sorted_vals   # exclusive global
    N = sorted_vals.numel()
    if N == 0:
        return csum
    idx = torch.arange(N, device=sorted_vals.device, dtype=torch.int64)
    newgrp = torch.ones(N, dtype=torch.bool, device=sorted_keys.device)
    newgrp[1:] = sorted_keys[1:] != sorted_keys[:-1]
    base = torch.where(newgrp, csum, torch.zeros_like(csum))
    base = torch.cummax(base, 0).values
    return csum - base


# --------------------------------------------------------------------------
# entry tensors (built once per solve)
# --------------------------------------------------------------------------

def _entries(topk_all, L):
    R, S, K = topk_all.shape
    dev = topk_all.device
    topk = topk_all.long()
    ent_g = topk.reshape(-1)                                # [E]
    ent_tok = torch.arange(R * S, device=dev,
                           dtype=torch.int64).repeat_interleave(K)
    ent_node = ent_tok // (L * S)
    return ent_g, ent_tok, ent_node


def demand_hist(topk_all, L, G):
    """Per-node expert demand [NN, G] — the drift/store statistic and the
    affinity-seed input. One index_add; sync-free."""
    R, S, K = topk_all.shape
    NN = R // L
    ent_g, _, ent_node = _entries(topk_all, L)
    h = torch.zeros(NN * G, dtype=torch.int64, device=topk_all.device)
    h.index_add_(0, ent_node * G + ent_g, torch.ones_like(ent_g))
    return h.view(NN, G)


def drift_ppm(hist_now, hist_ref, to_host=True):
    """L1 drift between two [NN, G] demand histograms, in ppm of total
    demand. Integer, order-free, O(NN*G): the quiet-iteration prefilter.
    to_host=False keeps the statistic on device (zero-D2H)."""
    tot = hist_ref.sum()
    l1 = (hist_now - hist_ref).abs().sum()
    ppm = torch.where(tot > 0, l1 * 1_000_000 // (2 * tot),
                      torch.full_like(tot, 1_000_000))
    return int(ppm) if to_host else ppm


# --------------------------------------------------------------------------
# seeds
# --------------------------------------------------------------------------

def seed_contig(G, R, L, device):
    epn = G // R
    g_ar = torch.arange(G, device=device, dtype=torch.int64)
    return (g_ar // epn) // L


def seed_affinity(hist, load_e, load_cap, cnt_cap, seed_cnt_cap=None):
    """COLD PATH (host syncs allowed). Greedy balanced best-fit from
    per-node demand marginals. hist [NN, G] -> primary [G]. Rounds over
    preference rank: in round c every unplaced expert bids for its c-th
    most-demanding node; bids are admitted in (demand desc, g asc) order
    under the load/count caps via sorted prefix sums. <= NN bounded
    rounds + least-loaded fallback (prefer under-cap)."""
    NN, G = hist.shape
    dev = hist.device
    # seed with a TIGHTER count budget than the hard cnt_cap: a seed that
    # packs nodes to cnt_cap leaves the repair/FM passes no destination
    # slots (observed: 3 nodes count-full, the 4th over load cap with 20
    # stranded heavy experts). ceil(G/NN)+1 keeps counts near-balanced
    # and leaves cnt_cap-headroom for every later move.
    if seed_cnt_cap is None:
        seed_cnt_cap = min(cnt_cap, -(-G // NN) + 1)
    cnt_cap = seed_cnt_cap
    aff = hist.t().contiguous()                             # [G, NN]
    pref = torch.argsort(aff * NN + (NN - 1 - torch.arange(
        NN, device=dev, dtype=torch.int64)).unsqueeze(0),
        dim=1, descending=True, stable=True)                # [G, NN]
    primary = torch.full((G,), -1, dtype=torch.int64, device=dev)
    node_load = torch.zeros(NN, dtype=torch.int64, device=dev)
    node_cnt = torch.zeros(NN, dtype=torch.int64, device=dev)
    g_ar = torch.arange(G, device=dev, dtype=torch.int64)
    vmax = int(aff.max()) + 2
    for c in range(NN):
        un = primary < 0
        if not bool(un.any()):
            break
        tgt = pref[:, c]
        val = aff[g_ar, tgt]
        # admit unplaced bids per target node in (val desc, g asc) order
        okey = (tgt * vmax + (vmax - 1 - val)) * G + g_ar
        okey = torch.where(un, okey, torch.full_like(okey, _I64MAX))
        order = torch.argsort(okey, stable=True)
        t_s = tgt[order]
        un_s = un[order]
        load_s = torch.where(un_s, load_e[order], torch.zeros_like(load_e))
        pre_load = _seg_prefix(load_s, t_s)
        pre_cnt = _seg_prefix(un_s.long(), t_s)
        fit = (un_s
               & (node_load[t_s] + pre_load + load_s <= load_cap)
               & (node_cnt[t_s] + pre_cnt + 1 <= cnt_cap))
        adm = order[fit]
        primary[adm] = tgt[adm]
        node_load.index_add_(0, tgt[adm], load_e[adm])
        node_cnt.index_add_(0, tgt[adm], torch.ones_like(adm))
    # vectorized fallback: leftover experts (load desc) round-robin over
    # free count-cap slots on load-ascending nodes; the solver's repair
    # passes clean up any residual load-cap breach with min-delta moves
    un = primary < 0
    if bool(un.any()):
        idx = un.nonzero(as_tuple=True)[0]
        lkey = (-load_e[idx]) * G + idx
        idx = idx[torch.argsort(lkey, stable=True)]         # load desc
        norder = torch.argsort(node_load * NN + torch.arange(
            NN, device=dev, dtype=torch.int64), stable=True)
        free = (cnt_cap - node_cnt).clamp(min=0)[norder]    # [NN]
        mf = int(free.max())
        grid_ok = (torch.arange(mf, device=dev,
                                dtype=torch.int64).unsqueeze(1)
                   < free.unsqueeze(0))                     # [mf, NN]
        slots = norder.unsqueeze(0).expand(mf, NN).reshape(-1)
        slots = slots[grid_ok.reshape(-1)]                  # round-robin
        assert int(slots.numel()) >= int(idx.numel()), \
            "cnt_cap slots exhausted — raise nlp"
        tgt = slots[:idx.numel()]
        primary[idx] = tgt
        node_load.index_add_(0, tgt, load_e[idx])
        node_cnt.index_add_(0, tgt, torch.ones_like(tgt))
    return primary


# --------------------------------------------------------------------------
# the fast solver — sync-free device program
# --------------------------------------------------------------------------

def build_placement_fast(topk_all, ranks_per_node, nlp, G,
                         balance_tol=0.20,
                         passes_a=4, passes_b=3, repair_passes=2,
                         max_moves_per_pass=0,
                         damp_num=0, damp_den=1,
                         seed="affinity",
                         seed_primary=None, seed_inst_nodes=None,
                         move_margin=0, keep_bonus=0,
                         min_gain_scaled=0):
    """Batched PLACE-lambda. [R, S, K] routing -> node-level placement
    (primary [G], inst_nodes [G, NN] bool) + device stats. Same integer
    objective as placelambda_gpu.build_placement_gpu; bounded-pass
    batched schedule; ZERO host syncs (cold seeds excepted).

    seed: "contig" | "affinity" (cold — syncs) | "warm" (seed_primary;
    Stage B prefers seed_inst_nodes replicas when keep_bonus > 0).
    move_margin: admit a Stage A move only if delta <= -1 - move_margin.
    damp_num/damp_den: admit only moves with delta <= best_delta *
    damp_num/damp_den (per pass, integer; 0 = off) — batch-staleness
    control.
    max_moves_per_pass: 0 = unlimited; else global per-pass admit budget
    in (delta asc, g asc) order.
    Rank-level hosts are NOT produced here — call finalize_hosts on
    trigger (Stage C off the per-iteration path). Stats are 0-dim device
    tensors; see stats_host()."""
    R, S, K = topk_all.shape
    dev = topk_all.device
    L = ranks_per_node
    NN = R // L
    assert K <= 16
    assert G % R == 0, "seed_contig needs G divisible by R"
    ent_g, ent_tok, ent_node = _entries(topk_all, L)
    E = ent_g.numel()
    load_e = torch.zeros(G, dtype=torch.int64, device=dev)
    load_e.index_add_(0, ent_g, torch.ones_like(ent_g))
    g_ar = torch.arange(G, device=dev, dtype=torch.int64)
    nn_ar = torch.arange(NN, device=dev, dtype=torch.int64)
    e_ar = torch.arange(E, device=dev, dtype=torch.int64)
    ones_e = torch.ones_like(ent_tok)
    total = load_e.sum()                                    # 0-dim tensor
    tol_ppm = int(round(balance_tol * 1_000_000))
    load_cap = total * (1_000_000 + tol_ppm) // (1_000_000 * NN)
    epn = G // R
    cnt_cap = L * epn + (L * (nlp - epn)) // 2              # host int

    hist = torch.zeros(NN * G, dtype=torch.int64, device=dev)
    hist.index_add_(0, ent_node * G + ent_g, ones_e)
    hist = hist.view(NN, G)

    # ---- seed ------------------------------------------------------------
    if seed == "warm":
        assert seed_primary is not None
        assert seed_primary.device == topk_all.device, (
            "warm seed must be device-resident (capture contract: no H2D"
            " inside the hot path)")
        primary = seed_primary.long().clone()
    elif seed == "affinity":
        primary = seed_affinity(hist, load_e, load_cap, cnt_cap)
    else:
        primary = seed_contig(G, R, L, dev)

    # ---- exact-integer matmul engine ------------------------------------
    # All the [E, NN]-scale reductions are GEMMs on the one-hot
    # token-expert matrix X [T, G]: every partial product is 0/1 and
    # every accumulated count is an exact integer < 2^24, so fp32
    # accumulation is bit-exact and order-free on any device (the
    # loccap_route_sl fp64-matmul precedent, on tensor cores). This
    # replaces 537 MB int64 intermediates per pass at 32n with ~2 GEMMs.
    T = R * S
    assert T < (1 << 24), "fp32-exact matmul bound"
    ent_pos = ent_tok * G + ent_g                           # fixed [E]
    ones_ef = torch.ones(E, dtype=torch.float32, device=dev)
    Xf = torch.zeros(T * G, dtype=torch.float32, device=dev)
    Xf.scatter_(0, ent_pos, ones_ef)                        # capture-safe
    Xf = Xf.view(T, G)
    home_tok = torch.arange(T, device=dev, dtype=torch.int64) // (L * S)
    Yf = torch.zeros(T * NN, dtype=torch.float32, device=dev)
    Yf.scatter_(0, torch.arange(T, device=dev, dtype=torch.int64) * NN
                + home_tok, torch.ones(T, dtype=torch.float32, device=dev))
    Yf = Yf.view(T, NN)

    ones_gf = torch.ones(G, dtype=torch.float32, device=dev)

    def _rebuild_cnt(primary):
        Pf = torch.zeros(G * NN, dtype=torch.float32, device=dev)
        Pf.scatter_(0, g_ar * NN + primary, ones_gf)
        return Xf @ Pf.view(G, NN)                          # [T, NN] fp32

    cnt = _rebuild_cnt(primary)
    node_load = torch.zeros(NN, dtype=torch.int64, device=dev)
    node_load.index_add_(0, primary, load_e)
    node_cnt = torch.zeros(NN, dtype=torch.int64, device=dev)
    node_cnt.index_add_(0, primary, torch.ones_like(primary))

    # ---- Stage A: batched delta passes (fixed count, masked no-ops) -----
    # The first `repair_passes` passes are BALANCE-REPAIR: forced
    # cheapest-delta evictions out of over-load-cap nodes (delta may be
    # positive — the explicit locality-for-balance trade a cold seed can
    # require; a no-op when nothing is over cap). Then improvement passes.
    moves_t = torch.zeros((), dtype=torch.int64, device=dev)
    OFFS = E + 1                                            # |delta| < E
    for _pass in range(repair_passes + passes_a):
        # delta table — the build_placement_gpu objective, as one GEMM:
        #   pay[g,b]   = #tokens containing g with cnt[t,b]==0
        #   excl[g,u]  = same, restricted to b == token home u
        #   gain_full[g] = #tokens containing g with cnt[t,p_g]==1
        #   gain_home[g] = same, restricted to tokens homed on p_g
        Z = (cnt == 0.0).float()                            # [T, NN]
        Wm = (cnt == 1.0).float()
        zh = Z.gather(1, home_tok.unsqueeze(1))             # [T, 1]
        wh = Wm.gather(1, home_tok.unsqueeze(1))
        Bmat = torch.cat([Z, Wm, zh * Yf, wh * Yf], dim=1)  # [T, 4NN]
        Sall = (Xf.t() @ Bmat).round().long()               # [G, 4NN]
        pay = Sall[:, :NN] - Sall[:, 2 * NN:3 * NN]
        M1 = Sall[:, NN:2 * NN]
        M2 = Sall[:, 3 * NN:]
        gain_full = M1[g_ar, primary]
        gain_home = M2[g_ar, primary]
        delta = pay - (gain_full - gain_home).unsqueeze(1)  # [G, NN]

        valid = nn_ar.unsqueeze(0) != primary.unsqueeze(1)
        valid &= (node_cnt + 1 <= cnt_cap).unsqueeze(0)
        valid &= (node_load.unsqueeze(0)
                  + load_e.unsqueeze(1)) <= load_cap
        key = delta * NN + nn_ar

        if _pass < repair_passes:
            over_amt = (node_load - load_cap).clamp(min=0)  # [NN]
            # ---- count-unblock swaps (review finding 1): when the
            # over-load node has count headroom but every under-load
            # node is count-full, evictions stall (no destination count
            # slot). Swap the lightest cheapest-delta expert OUT of each
            # blocked node INTO the hottest over-load node — the freed
            # count slots let the evictions below land. Masked no-op
            # when nothing is blocked. All 0-dim tensor gating —
            # sync-free.
            hotk = over_amt * NN + (NN - 1 - nn_ar)
            hot = NN - 1 - torch.remainder(hotk.max(), NN)  # 0-dim
            hot_over = over_amt.max() > 0
            hot_free = node_cnt.gather(0, hot.view(1)).squeeze(0) < cnt_cap
            src_blk = (node_cnt >= cnt_cap) & (node_load <= load_cap)
            cand_u = (src_blk[primary] & (primary != hot)
                      & hot_over & hot_free)
            d_hot = delta.gather(1, hot.view(1, 1).expand(G, 1)).squeeze(1)
            skey = (((primary * (E + 1) + load_e) * (2 * OFFS)
                     + (d_hot + OFFS)) * G + g_ar)
            skey = torch.where(cand_u, skey,
                               torch.full_like(skey, _I64MAX))
            o_u = torch.argsort(skey, stable=True)
            runk = torch.where(cand_u[o_u], primary[o_u],
                               torch.full_like(primary, NN)[o_u])
            first = (_run_ordinal(runk) == 0) & cand_u[o_u]
            csum = torch.cumsum(first.long(), 0) - 1
            headroom = (cnt_cap
                        - node_cnt.gather(0, hot.view(1))).squeeze(0)
            adm_u_s = first & (csum < headroom)
            adm_u = torch.zeros(G, dtype=torch.bool, device=dev)
            adm_u.scatter_(0, o_u, adm_u_s)
            adm_ul = adm_u.long()
            moves_t = moves_t + adm_ul.sum()
            node_load.index_add_(0, primary, -load_e * adm_ul)
            node_load.index_add_(0, hot.view(1),
                                 (load_e * adm_ul).sum().view(1))
            node_cnt.index_add_(0, primary, -adm_ul)
            node_cnt.index_add_(0, hot.view(1), adm_ul.sum().view(1))
            primary = torch.where(adm_u, hot.view(1).expand(G), primary)
            # delta/cnt stay stale for this pass (few swaps; the next
            # pass recomputes); caps below use the UPDATED loads/counts
            over_amt = (node_load - load_cap).clamp(min=0)
            valid = nn_ar.unsqueeze(0) != primary.unsqueeze(1)
            valid &= (node_cnt + 1 <= cnt_cap).unsqueeze(0)
            valid &= (node_load.unsqueeze(0)
                      + load_e.unsqueeze(1)) <= load_cap
            # ---- forced evictions: experts on over-cap nodes move to
            # their cheapest-delta feasible node; per-source shed quota =
            # overflow, then destination caps (leftovers retry next pass).
            # zero-load experts shed nothing — filtered (finding 3).
            found, best = _min_by_key(key, valid)
            found = found & (over_amt[primary] > 0) & (load_e > 0)
            b_star = torch.remainder(best, NN)
            d_star = best.div(NN, rounding_mode="floor")
            okey1 = ((primary * (2 * OFFS) + (d_star + OFFS)) * G + g_ar)
            okey1 = torch.where(found, okey1,
                                torch.full_like(okey1, _I64MAX))
            o1 = torch.argsort(okey1, stable=True)
            f1 = found[o1]
            shed = torch.where(f1, load_e[o1], torch.zeros_like(load_e))
            pre_shed = _seg_prefix(shed, primary[o1])
            sel1 = f1 & (pre_shed < over_amt[primary[o1]])
            m1 = torch.zeros(G, dtype=torch.bool, device=dev)
            m1.scatter_(0, o1, sel1)
            found = m1
        else:
            neg = valid & (delta <= -1 - move_margin)
            # best target per expert (order-free: b in the low bits)
            found, best = _min_by_key(key, neg)
            b_star = torch.remainder(best, NN)
            d_star = best.div(NN, rounding_mode="floor")
            if damp_num > 0:
                dmin = torch.where(found, d_star,
                                   torch.zeros_like(d_star)).min()
                thr = dmin * damp_num // damp_den           # <= 0
                found = found & (d_star <= thr)

        # canonical admit order: (dst node, delta asc, g asc); admit under
        # destination caps frozen at pass start (sources only shed —
        # conservative), optional global per-pass move budget
        okey = ((b_star * (2 * OFFS) + (d_star + OFFS)) * G + g_ar)
        okey = torch.where(found, okey, torch.full_like(okey, _I64MAX))
        order = torch.argsort(okey, stable=True)
        f_s = found[order]
        b_s = b_star[order]
        load_s = torch.where(f_s, load_e[order], torch.zeros_like(load_e))
        pre_load = _seg_prefix(load_s, b_s)
        pre_cnt = _seg_prefix(f_s.long(), b_s)
        fit = (f_s
               & (node_load[b_s] + pre_load + load_s <= load_cap)
               & (node_cnt[b_s] + pre_cnt + 1 <= cnt_cap))
        if max_moves_per_pass > 0 and _pass >= repair_passes:
            gkey = torch.where(found, (d_star + OFFS) * G + g_ar,
                               torch.full_like(g_ar, _I64MAX))
            grank = torch.argsort(torch.argsort(gkey, stable=True),
                                  stable=True)
            fit &= grank[order] < max_moves_per_pass
        adm_mask = torch.zeros(G, dtype=torch.bool, device=dev)
        adm_mask.scatter_(0, order, fit)
        moves_t = moves_t + adm_mask.long().sum()
        new_primary = torch.where(adm_mask, b_star, primary)
        adm_l = adm_mask.long()
        node_load.index_add_(0, primary, -load_e * adm_l)
        node_load.index_add_(0, b_star, load_e * adm_l)
        node_cnt.index_add_(0, primary, -adm_l)
        node_cnt.index_add_(0, b_star, adm_l)
        primary = new_primary
        cnt = _rebuild_cnt(primary)

    # ---- Stage B: batched replica passes (fixed count, masked no-ops) ---
    inst_nodes = torch.zeros(G * NN, dtype=torch.bool, device=dev)
    inst_nodes.scatter_(0, g_ar * NN + primary,
                        torch.ones(G, dtype=torch.bool, device=dev))
    inst_nodes = inst_nodes.view(G, NN)
    slots_left = torch.full((NN,), L * nlp, dtype=torch.int64,
                            device=dev) - node_cnt
    spent_t = torch.zeros((), dtype=torch.int64, device=dev)
    credit_c = torch.cat([
        torch.zeros(2, dtype=torch.int64, device=dev),
        LCM16 // torch.arange(2, K + 1, dtype=torch.int64, device=dev)])
    resident = None
    if keep_bonus > 0 and seed_inst_nodes is not None:
        resident = seed_inst_nodes.to(dev)
    flat_g = g_ar.unsqueeze(1).expand(G, NN).reshape(-1)
    flat_u = nn_ar.unsqueeze(0).expand(G, NN).reshape(-1)
    gn_ar = torch.arange(G * NN, device=dev, dtype=torch.int64)
    for _pass in range(passes_b):
        val = cnt[ent_tok, primary[ent_g]].long().clamp(min=0, max=K)
        H = torch.zeros(G * NN * (K + 1), dtype=torch.int64, device=dev)
        H.index_add_(0, (ent_g * NN + ent_node) * (K + 1) + val, ones_e)
        H = H.view(G, NN, K + 1)
        exact = H[:, :, 1]
        credit = (H[:, :, 2:] * credit_c[2:].view(1, 1, -1)).sum(-1)
        score = exact * LCM16 + credit                      # [G, NN]
        if resident is not None:
            score = score + resident.long() * keep_bonus
        valid = ((~inst_nodes) & (slots_left > 0).unsqueeze(0)
                 & (score > min_gain_scaled))
        # per-node admit order = reference tie order (score desc, exact
        # desc, g asc): two stable sorts compose the lexicographic key
        v_f = valid.reshape(-1)
        ex_f = torch.where(v_f, exact.reshape(-1),
                           torch.full_like(flat_g, -1))
        sc_f = torch.where(v_f, score.reshape(-1),
                           torch.full_like(flat_g, -1))
        o1 = torch.argsort((-ex_f) * G + flat_g, stable=True)
        o2 = torch.argsort(-sc_f[o1], stable=True)
        o12 = o1[o2]                                        # score,exact,g
        o3 = torch.argsort(flat_u[o12] * (G * NN + 1) + gn_ar,
                           stable=True)
        osel = o12[o3]                                      # by node, rank
        ordn = _run_ordinal(flat_u[osel])
        take = v_f[osel] & (ordn < slots_left[flat_u[osel]])
        adm_flat = torch.zeros(G * NN, dtype=torch.bool, device=dev)
        adm_flat.scatter_(0, osel, take)
        admitted = adm_flat.view(G, NN)
        spent_t = spent_t + adm_flat.long().sum()
        adm_e = admitted[ent_g, ent_node].float()           # [E] mask
        cnt.view(-1).index_add_(0, ent_tok * NN + primary[ent_g], -adm_e)
        cnt.view(-1).index_add_(0, ent_tok * NN + ent_node, adm_e)
        inst_nodes |= admitted
        slots_left = slots_left - admitted.long().sum(0)

    visited = cnt > 0
    home_col = (torch.arange(R * S, device=dev, dtype=torch.int64)
                // (L * S))
    incidence_t = (visited.long().sum()
                   - visited.gather(1, home_col.unsqueeze(1)).long().sum())
    return {
        "primary": primary,
        "inst_nodes": inst_nodes,
        "node_load": node_load,
        "hist": hist,
        "load_e": load_e,
        "_Xf": Xf,  # [T, G] one-hot; reusable as the cover workspace
        "stats_dev": {
            "fm_moves": moves_t,
            "replica_slots_spent": spent_t,
            "incidence_remote_predicted": incidence_t,
        },
        "config": {
            "mode": "placefast",
            "seed": seed,
            "passes_a": passes_a,
            "passes_b": passes_b,
            "repair_passes": repair_passes,
            "damp": (damp_num, damp_den),
            "max_moves_per_pass": max_moves_per_pass,
            "move_margin": move_margin,
            "keep_bonus": keep_bonus,
            "balance_tol": balance_tol,
            "cnt_cap": cnt_cap,
        },
    }


def stats_host(res):
    """COLD: one sync — device stats + config as a flat host dict."""
    out = dict(res["config"])
    keys = list(res["stats_dev"].keys())
    vals = torch.stack([res["stats_dev"][k] for k in keys]).cpu()
    out.update({k: int(v) for k, v in zip(keys, vals)})
    return out


# --------------------------------------------------------------------------
# Stage C — rank-level finalize (trigger-time only; COLD path)
# --------------------------------------------------------------------------

def finalize_hosts(res, R, ranks_per_node, nlp, method="snake"):
    """COLD PATH (syncs). Node-level placement -> per-expert sorted rank
    lists (python). snake: vectorized boustrophedon over per-node
    instances sorted by share desc — near-LPT, deterministic, one host
    sync. lpt: the reference host LPT (placelambda_gpu Stage C
    semantics)."""
    inst_nodes = res["inst_nodes"]
    load_e = res["load_e"]
    G, NN = inst_nodes.shape
    L = ranks_per_node
    n_inst = inst_nodes.long().sum(1)                       # [G]
    if method == "lpt":
        inst_h = inst_nodes.cpu()
        load_h = load_e.cpu().tolist()
        n_h = n_inst.cpu().tolist()
        per_node = [[] for _ in range(NN)]
        for g in range(G):
            share = load_h[g] / n_h[g]
            for u in range(NN):
                if bool(inst_h[g, u]):
                    per_node[u].append((g, share))
        hosts = [[] for _ in range(G)]
        for u in range(NN):
            ranks = list(range(u * L, (u + 1) * L))
            rload = {r: 0.0 for r in ranks}
            rslots = {r: 0 for r in ranks}
            for g, share in sorted(per_node[u],
                                   key=lambda t: (-t[1], t[0])):
                cand = [r for r in ranks
                        if rslots[r] < nlp and r not in hosts[g]]
                assert cand, (u, g, "no rank slot left — raise nlp")
                r = min(cand, key=lambda r: (rload[r], r))
                hosts[g].append(r)
                rload[r] += share
                rslots[r] += 1
        return [sorted(h) for h in hosts]
    # snake: instances sorted (share desc, g asc) per node, rank =
    # boustrophedon position
    gg, uu = inst_nodes.nonzero(as_tuple=True)
    share_q = (load_e[gg] << 20) // n_inst[gg]              # int key
    SQ = int(share_q.max()) + 1 if share_q.numel() else 1
    okey = (uu * SQ + (SQ - 1 - share_q)) * G + gg
    order = torch.argsort(okey, stable=True)
    u_s = uu[order]
    pos = _run_ordinal(u_s)
    lane = pos % L
    down = (pos // L) % 2 == 1
    lane = torch.where(down, L - 1 - lane, lane)
    rank = u_s * L + lane
    hosts = [[] for _ in range(G)]
    gg_h = gg[order].cpu().tolist()
    rk_h = rank.cpu().tolist()
    for g, r in zip(gg_h, rk_h):
        hosts[g].append(r)
    return [sorted(h) for h in hosts]


# --------------------------------------------------------------------------
# decision lane — sync-free trigger
# --------------------------------------------------------------------------

def incidence_cover_fast(topk_all, ion, ranks_per_node, workspace=None):
    """Fixed-round masked greedy node cover: total remote node visits
    under placement `ion` [G, NN], as a 0-dim device tensor. No
    data-dependent shapes, no host syncs — graph-capturable. Same greedy
    preference as placelambda_gpu.incidence_lb (max cover, lower node
    id; home free). Per round: one [E] scatter refreshes the masked
    one-hot X_rem, one exact-integer GEMM gives the [T, NN] cover
    counts (counts <= K — fp32-exact). workspace: an optional [T*G]
    fp32 buffer (zeroed at least once) shared across calls."""
    R, S, K = topk_all.shape
    dev = topk_all.device
    L = ranks_per_node
    G = ion.shape[0]
    NN = ion.shape[1]
    ion = ion.to(dev)
    T = R * S
    ent_g, ent_tok, ent_node = _entries(topk_all, L)
    ent_pos = ent_tok * G + ent_g
    rem = ~ion[ent_g, ent_node]                             # [E]
    if workspace is None:
        workspace = torch.zeros(T * G, dtype=torch.float32, device=dev)
    ion_f = ion.float()
    picks = torch.zeros((), dtype=torch.int64, device=dev)
    nn_ar = torch.arange(NN, device=dev, dtype=torch.int64)
    for _ in range(min(K, NN)):
        workspace.scatter_(0, ent_pos, rem.float())
        cnt = (workspace.view(T, G) @ ion_f).round().long() # [T, NN]
        key = cnt * NN + (NN - 1 - nn_ar)
        best, _ = key.max(dim=1)
        n_star = NN - 1 - torch.remainder(best, NN)
        has = best.div(NN, rounding_mode="floor") > 0
        picks = picks + has.long().sum()
        served = ion[ent_g, n_star[ent_tok]] & has[ent_tok]
        rem = rem & ~served
    return picks                                            # 0-dim tensor


def incidence_proxy(topk_all, primary, ion, ranks_per_node):
    """One-shot serving-node incidence, 0-dim device tensor: each entry
    is served at its HOME node when an instance is resident there, else
    at its primary node. An upper bound on the router's min-cover
    incidence (ignores consolidation onto shared non-home nodes) but
    computed identically for any placement — a fair, O(E) single-pass
    trigger statistic for large NN where the full greedy cover's
    min(K, NN) GEMM rounds dominate the hot path. Sync-free."""
    R, S, K = topk_all.shape
    dev = topk_all.device
    L = ranks_per_node
    NN = ion.shape[1]
    ion = ion.to(dev)
    primary = primary.to(dev)
    T = R * S
    ent_g, ent_tok, ent_node = _entries(topk_all, L)
    serve = torch.where(ion[ent_g, ent_node], ent_node, primary[ent_g])
    V = torch.zeros(T * NN, dtype=torch.bool, device=dev)
    V.scatter_(0, ent_tok * NN + serve,
               torch.ones_like(serve, dtype=torch.bool))
    V = V.view(T, NN)
    home_tok = torch.arange(T, device=dev, dtype=torch.int64) // (L * S)
    return (V.long().sum()
            - V.gather(1, home_tok.unsqueeze(1)).long().sum())


class PlacementStore:
    """Resident-placement state for the scenario-2 loop (weights stay
    where they are unless a trigger pays for the movement bytes).

    Holds: node-level placement (primary [G], ion [G, NN]), the demand
    histogram it was solved on ([NN, G] — the drift_ppm reference), and
    the finalized rank-level hosts (built lazily, only when a trigger
    commits the placement — Stage C off the hot path). All device
    tensors; adopting a fresh solve is a few tensor copies."""

    def __init__(self, res, R, ranks_per_node, nlp, hosts=None):
        self.R = R
        self.L = ranks_per_node
        self.nlp = nlp
        self.primary = res["primary"].clone()
        self.ion = res["inst_nodes"].clone()
        self.hist = res["hist"].clone()
        self.load_e = res["load_e"].clone()
        self.hosts = hosts                                  # lazy
        self.epoch = 0

    def moves_from(self, res_new):
        """Node-instance diff to a fresh solve -> (adds, removes) 0-dim
        device tensors — the weight-movement byte counter (bytes = adds *
        bytes_per_expert_instance; removes are free). Sync-free."""
        ion_new = res_new["inst_nodes"]
        return ((ion_new & ~self.ion).long().sum(),
                (self.ion & ~ion_new).long().sum())

    def adopt(self, res_new, finalize=True, method="snake"):
        """Commit a fresh solve as the resident placement (trigger fired
        and the movement was dispatched). finalize=True rebuilds hosts
        (COLD — syncs); the caller wires actual weight movement."""
        self.primary.copy_(res_new["primary"])
        self.ion.copy_(res_new["inst_nodes"])
        self.hist.copy_(res_new["hist"])
        self.load_e.copy_(res_new["load_e"])
        self.epoch += 1
        if finalize:
            self.hosts = finalize_hosts(res_new, self.R, self.L, self.nlp,
                                        method=method)
        return self.hosts


def place_decision_fast(topk_all, ion_cur, res_new, ranks_per_node,
                        gain_threshold_ppm=50_000, to_host=True,
                        mode="cover", primary_cur=None):
    """Tensorized trigger: incidence of the resident node-level
    placement vs the fresh solve, plus the node-instance move diff.
    mode="cover": fixed-round greedy cover (the exact-comparable
    statistic; min(K, NN) GEMM rounds per side — reuses the solver's Xf
    as workspace when present). mode="proxy": one-shot serving-node
    incidence for both sides (needs primary_cur; the O(E) large-NN
    statistic). to_host=True: one sync at the end returns the reference
    dict. to_host=False: ZERO-D2H — returns a [5] device tensor
    (lb_cur, lb_new, gain_ppm, adds, removes) the caller reads later
    (deferred-trigger pattern: consume at iteration i+1)."""
    R, S, K = topk_all.shape
    G = ion_cur.shape[0]
    if mode == "proxy":
        assert primary_cur is not None, "proxy trigger needs primary_cur"
        lb_cur = incidence_proxy(topk_all, primary_cur, ion_cur,
                                 ranks_per_node)
        lb_new = incidence_proxy(topk_all, res_new["primary"],
                                 res_new["inst_nodes"], ranks_per_node)
    else:
        ws = res_new.get("_Xf")
        ws = (ws.reshape(-1) if ws is not None
              and ws.numel() == R * S * G else
              torch.zeros(R * S * G, dtype=torch.float32,
                          device=topk_all.device))
        lb_cur = incidence_cover_fast(topk_all, ion_cur, ranks_per_node,
                                      workspace=ws)
        lb_new = incidence_cover_fast(topk_all, res_new["inst_nodes"],
                                      ranks_per_node, workspace=ws)
    ion_new = res_new["inst_nodes"]
    adds = (ion_new & ~ion_cur).long().sum()
    removes = (ion_cur & ~ion_new).long().sum()
    gain_ppm = torch.where(
        lb_cur > 0, (lb_cur - lb_new) * 1_000_000 // lb_cur,
        torch.zeros_like(lb_cur))
    packed = torch.stack([lb_cur, lb_new, gain_ppm, adds, removes])
    if not to_host:
        return packed
    lb_c, lb_n, gp, n_add, n_rem = (int(x) for x in packed.cpu())
    return {
        "lb_cur": lb_c,
        "lb_new": lb_n,
        "gain_ppm": gp,
        "moves_add": n_add,
        "moves_remove": n_rem,
        "trigger": int(gp >= gain_threshold_ppm),
    }


def hosts_to_ion(hosts, R, ranks_per_node, device="cpu"):
    ion = torch.zeros(len(hosts), R // ranks_per_node, dtype=torch.bool,
                      device=device)
    for gexp, hs in enumerate(hosts):
        for r in hs:
            ion[gexp, r // ranks_per_node] = True
    return ion
