"""Device-resident planner primitives for the per-iteration timed planning
phases of the expert-movement baseline drivers (SCHEMA.md protocol rule 5).

Every function here is a bit-exact vectorization of a scalar reference in
ultraep_semantics / eplb_semantics / epic_semantics, written as pure torch
ops with no data-dependent python control flow on tensor contents (the one
exception, the interleave coprime search, widens its candidate window by a
deterministic schedule). Results are therefore identical on any device;
the CPU parity suite (test_ep_gpu_plan.py) runs them with device="cpu"
against the scalar originals, and the drivers run them on CUDA inside the
timed `plan` event bracket.

Bit-exactness notes (the tie-breaks that matter):
  * largest-remainder bonus selection = repeated argmax with strict `>`
    (first max wins => lowest index j). The vectorized form uses a STABLE
    descending sort, whose order among equal remainders is ascending j —
    the identical selection.
  * searchsorted replica selection pads each prefix row monotonically via
    cummax; quota ranks are < total, so padded duplicates of the row max
    never alter the count of entries <= q.
"""

import torch

__all__ = [
    "largest_remainder_split",
    "rank_quota_prefix_nonlocal",
    "d6_rank_quota_prefix",
    "local_spread_rank_quota_prefix",
    "interleave_params_batched",
    "reroute_expand_gpu",
    "reroute_expand_all_gpu",
    "place_slots_from_locals",
    "comb_dst_slot_from_topk",
]


def _run_ordinal(sorted_keys: torch.Tensor) -> torch.Tensor:
    """Occurrence index within each run of equal consecutive keys.

    sorted_keys must be sorted ascending. Mirrors the run-start cummax
    recipe of ultraep_semantics.build_comm_layout (:830-841)."""
    n = sorted_keys.numel()
    dev = sorted_keys.device
    positions = torch.arange(n, device=dev, dtype=torch.int64)
    if n == 0:
        return positions
    is_start = torch.cat([
        torch.ones(1, dtype=torch.bool, device=dev),
        sorted_keys[1:] != sorted_keys[:-1],
    ])
    run_start = torch.where(is_start, positions,
                            torch.full_like(positions, -1))
    run_start = torch.cummax(run_start, dim=0).values
    return positions - run_start


def largest_remainder_split(loads_g: torch.Tensor, lcnts: torch.Tensor,
                            Cmax: int):
    """Equal split of per-expert loads across instances, largest-remainder
    with extras to the lowest instance index.

    Parity target: the quota loop of eplb_semantics.build_eplb_plan
    (:133-142). Returns (quota, quota_prefix), both [G, Cmax] int32 with
    zero tails beyond lcnts[l] (matching the CPU zeros-init layout).
    """
    dev = loads_g.device
    loads = loads_g.long()
    C = lcnts.long()
    j = torch.arange(Cmax, device=dev, dtype=torch.int64).unsqueeze(0)
    valid = j < C.unsqueeze(1)
    base = torch.div(loads, C, rounding_mode="floor").unsqueeze(1)
    rem = torch.remainder(loads, C).unsqueeze(1)
    quota = torch.where(valid, base + (j < rem).long(),
                        torch.zeros_like(base))
    prefix = torch.where(valid, quota.cumsum(dim=1),
                         torch.zeros_like(quota))
    return quota.to(torch.int32), prefix.to(torch.int32)


def rank_quota_prefix_nonlocal(tpe_all: torch.Tensor, quota: torch.Tensor,
                               lcnts: torch.Tensor) -> torch.Tensor:
    """[R, G, Cmax] int32 rank-quota prefix for ALL source ranks at once,
    non-locality path only.

    Parity target: ultraep_semantics._rank_quota_alloc_for_expert
    (:429-448, locality_aware=False) via build_rank_quota_prefix — the
    proportional floor split of each source's load over the quota row,
    then largest-remainder bonuses (ties -> lowest j via stable
    descending sort; see module docstring).
    """
    R, G = tpe_all.shape
    Cmax = quota.shape[1]
    dev = tpe_all.device
    my_load = tpe_all.long()                              # [R, G]
    q = quota.long()                                      # [G, Cmax]
    C = lcnts.long()
    j = torch.arange(Cmax, device=dev, dtype=torch.int64).unsqueeze(0)
    valid = (j < C.unsqueeze(1)).unsqueeze(0)             # [1, G, Cmax]
    total = q.sum(dim=1)                                  # [G] (tail is 0)

    active = ((my_load > 0) & (total.unsqueeze(0) > 0))   # [R, G]
    total_safe = torch.clamp(total, min=1).view(1, G, 1)
    scaled = my_load.unsqueeze(-1) * q.unsqueeze(0)       # [R, G, Cmax]
    alloc = torch.div(scaled, total_safe, rounding_mode="floor")
    remainders = torch.remainder(scaled, total_safe)
    alloc = torch.where(valid & active.unsqueeze(-1), alloc,
                        torch.zeros_like(alloc))
    remaining = torch.where(active, my_load - alloc.sum(dim=-1),
                            torch.zeros_like(my_load))    # [R, G]

    # Bonus: top-`remaining` valid entries by (remainder desc, j asc).
    # Invalid/inactive entries get key -1 (< any real remainder >= 0);
    # remaining < C guarantees they are never selected.
    keys = torch.where(valid & active.unsqueeze(-1), remainders,
                       torch.full_like(remainders, -1))
    _, idx = torch.sort(keys, dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(idx)
    ranks.scatter_(-1, idx, torch.arange(
        Cmax, device=dev, dtype=torch.int64).expand(R, G, Cmax))
    alloc = alloc + (ranks < remaining.unsqueeze(-1)).long()

    prefix = torch.where(valid, alloc.cumsum(dim=-1),
                         torch.zeros_like(alloc))
    return prefix.to(torch.int32)


def d6_rank_quota_prefix(tpe_all: torch.Tensor, lcnts: torch.Tensor,
                         Cmax: int) -> torch.Tensor:
    """[R, G, Cmax] int32 D6 prefix: source src sends ALL its tokens for
    expert l to instance j* = src mod lcnts[l].

    Parity target: epic_semantics.epic_rank_quota_prefix (:238-256) —
    step function, 0 below j*, tpe[src, l] at and above (within j < C).
    """
    R, G = tpe_all.shape
    dev = tpe_all.device
    C = lcnts.long().view(1, G, 1)
    j = torch.arange(Cmax, device=dev, dtype=torch.int64).view(1, 1, Cmax)
    src = torch.arange(R, device=dev, dtype=torch.int64).view(R, 1, 1)
    j_star = torch.remainder(src, C)
    step = (j >= j_star) & (j < C)
    out = torch.where(step, tpe_all.long().unsqueeze(-1),
                      torch.zeros(1, dtype=torch.int64, device=dev))
    return out.to(torch.int32)


def local_spread_rank_quota_prefix(tpe_all: torch.Tensor,
                                   lcnts: torch.Tensor,
                                   Cmax: int) -> torch.Tensor:
    """[R, G, Cmax] int32 rank-quota prefix for the `local_spread` replica
    rule: each SOURCE splits its own load for expert l equally over the
    expert's C_l instances by largest remainder (extras to the lowest
    instance index) — count-equivalent to token-ordinal round-robin
    replica selection (SGLang dynamic-dispatch analog), and fully
    sender-local: a pure function of tpe_all[src], no exchange.

    Parity target: largest_remainder_split applied per source row.
    Caveat (recorded in variants/handoff): count-equivalence, not
    token-identity — the coprime interleave permutes which tokens fill
    each instance's block.
    """
    R, G = tpe_all.shape
    dev = tpe_all.device
    loads = tpe_all.long().unsqueeze(-1)                    # [R, G, 1]
    C = lcnts.long().view(1, G, 1)
    j = torch.arange(Cmax, device=dev, dtype=torch.int64).view(1, 1, Cmax)
    valid = j < C
    base = torch.div(loads, C, rounding_mode="floor")
    rem = torch.remainder(loads, C)
    q = torch.where(valid, base + (j < rem).long(), torch.zeros_like(base))
    prefix = torch.where(valid, q.cumsum(dim=-1), torch.zeros_like(q))
    return prefix.to(torch.int32)


def interleave_params_batched(totals: torch.Tensor,
                              expert_ids: torch.Tensor):
    """(stride, offset) int64 tensors for the coprime-stride interleave.

    Parity target: ultraep_semantics._interleave_params (:606-617):
    stride0 = clamp(total//2 + 1), then increment (wrapping total -> 1)
    until gcd(stride, total) == 1. The wrap makes the candidate sequence
    the rotation c_i = ((stride0 - 1 + i) mod (total - 1)) + 1 for
    total >= 2; we scan it in deterministic windows (64, x4 growth).
    totals must be >= 1; rows with total == 1 return stride 1.
    """
    totals = totals.long()
    dev = totals.device
    n = totals.numel()
    offset = torch.remainder(expert_ids.long(), totals)
    stride = torch.ones_like(totals)
    multi = totals > 1
    if bool(multi.any()):
        t = totals[multi]
        s0 = torch.div(t, 2, rounding_mode="floor") + 1
        s0 = torch.where(s0 >= t, t - 1, s0)
        s0 = torch.clamp(s0, min=1)
        found = torch.zeros_like(t, dtype=torch.bool)
        res = s0.clone()
        base, width = 0, 64
        while not bool(found.all()):
            i = torch.arange(base, base + width, device=dev,
                             dtype=torch.int64).unsqueeze(0)
            cand = torch.remainder(s0.unsqueeze(1) - 1 + i,
                                   (t - 1).unsqueeze(1)) + 1
            ok = torch.gcd(cand, t.unsqueeze(1)) == 1
            first = torch.where(
                ok, i.expand_as(ok),
                torch.full_like(cand, base + width)).min(dim=1).values
            hit = (first < base + width) & ~found
            res = torch.where(
                hit,
                torch.remainder(s0 - 1 + first, t - 1) + 1,
                res)
            found = found | hit
            base += width
            width *= 4
        stride = torch.ones_like(totals)
        stride[multi] = res
    return stride, offset


def _reroute_core(es, ss, gs, rqp_rows, l2p, lcnts, counts_of_entry,
                  stride_of_entry, offset_of_entry, interleave: bool):
    """Shared replica-selection core; all inputs already entry-aligned and
    sorted (group-major, expert-major, token-ascending)."""
    ordinal = _run_ordinal(gs)
    totals = counts_of_entry
    if interleave:
        qr = torch.where(
            totals > 1,
            torch.remainder(ordinal * stride_of_entry + offset_of_entry,
                            torch.clamp(totals, min=1)),
            ordinal)
    else:
        qr = ordinal
    prm = rqp_rows.cummax(dim=1).values
    replica = torch.searchsorted(prm, qr.unsqueeze(1), right=True).squeeze(1)
    Ce = lcnts.long()[es]
    replica = torch.minimum(replica, torch.clamp(Ce - 1, min=0))
    phys = l2p.long()[es, replica]
    return ss, phys


def reroute_expand_gpu(rqp_src: torch.Tensor, l2p: torch.Tensor,
                       lcnts: torch.Tensor, topk_src: torch.Tensor,
                       interleave: bool):
    """Expand ONE source rank's routing logical -> physical on device.

    Parity target: ultraep_semantics.reroute_expand (:620-668). Returns
    (entry_token [N], entry_phys [N]) int64 in the kernel's implicit order
    (expert-major, token ascending within an expert), N == S*K.
    """
    S, K = topk_src.shape
    G = lcnts.numel()
    dev = topk_src.device
    e = topk_src.reshape(-1).long()
    s = torch.arange(S, device=dev, dtype=torch.int64).repeat_interleave(K)
    order = torch.argsort(e * S + s)          # unique keys -> deterministic
    es, ss = e[order], s[order]
    counts = torch.bincount(es, minlength=G)  # [G] == tpe of this source
    if interleave:
        stride, offset = interleave_params_batched(
            torch.clamp(counts, min=1),
            torch.arange(G, device=dev, dtype=torch.int64))
        stride_e, offset_e = stride[es], offset[es]
    else:
        stride_e = offset_e = None
    return _reroute_core(es, ss, es, rqp_src.long()[es], l2p, lcnts,
                         counts[es], stride_e, offset_e, interleave)


def reroute_expand_all_gpu(rqp_all: torch.Tensor, l2p: torch.Tensor,
                           lcnts: torch.Tensor, topk_all: torch.Tensor,
                           interleave: bool):
    """Batched reroute_expand for ALL R sources: returns (tok [R, N],
    phys [R, N]) int64, each row in the same order as the single-source
    expansion of that rank (source-major sort keys keep rows independent).
    """
    R, S, K = topk_all.shape
    G = lcnts.numel()
    N = S * K
    dev = topk_all.device
    e = topk_all.reshape(R, N).long()
    s = torch.arange(S, device=dev,
                     dtype=torch.int64).repeat_interleave(K).expand(R, N)
    r = torch.arange(R, device=dev, dtype=torch.int64).unsqueeze(1)
    key = (r * G + e) * S + s                 # src-major, expert, token
    order = torch.argsort(key.reshape(-1))
    ef = e.reshape(-1)[order]
    sf = s.reshape(-1)[order]
    rf = (r.expand(R, N).reshape(-1))[order]
    re = rf * G + ef                          # per-(src, expert) group id
    counts = torch.bincount(re, minlength=R * G)      # [R*G]
    if interleave:
        stride, offset = interleave_params_batched(
            torch.clamp(counts, min=1),
            torch.arange(G, device=dev, dtype=torch.int64).repeat(R))
        stride_e, offset_e = stride[re], offset[re]
    else:
        stride_e = offset_e = None
    rqp_rows = rqp_all.long().reshape(R * G, -1)[re]
    tok, phys = _reroute_core(ef, sf, re, rqp_rows, l2p, lcnts,
                              counts[re], stride_e, offset_e, interleave)
    return tok.reshape(R, N), phys.reshape(R, N)


def direct_layout_entries(ent_tok: torch.Tensor, ent_phys: torch.Tensor,
                          rank: int, nlp: int, R: int):
    """Shared device core of the per-iteration direct-wire layout, from the
    canonical per-source expansions (each row of ent_tok/ent_phys [R, N]
    already (phys, token)-sorted).

    Parity target: the sender/receiver derivation of
    ultraep_semantics.build_comm_layout (:810-841) /
    epic_semantics.build_epic_group_layouts m=1. Returns a dict of device
    tensors; the ragged boolean gather for the receiver rows is a known
    (honest, timed) sync."""
    dev = ent_tok.device
    my_tok = ent_tok[rank]
    my_phys = ent_phys[rank]
    in_splits = torch.bincount(my_phys // nlp, minlength=R).to(torch.int32)
    dest_all = ent_phys // nlp
    mine = dest_all == rank
    out_splits = mine.sum(dim=1).to(torch.int32)
    all_local = ent_phys[mine] - rank * nlp
    place_slots, seg_rows, seg_start = place_slots_from_locals(
        all_local, nlp)
    src_ids = torch.arange(R, device=dev, dtype=torch.int64).unsqueeze(1)
    pair_rows = torch.bincount((src_ids * R + dest_all).reshape(-1),
                               minlength=R * R)
    return dict(
        my_tok=my_tok, my_phys=my_phys, in_splits=in_splits,
        out_splits=out_splits, all_local=all_local,
        place_slots=place_slots, seg_rows=seg_rows, seg_start=seg_start,
        pair_max=pair_rows.max(),
    )


def place_slots_from_locals(all_local: torch.Tensor, nlp: int):
    """Receiver placement from the concatenated local-slot ids of arriving
    rows (src-major arrival order, (phys, token)-sorted within each src).

    Parity target: ultraep_semantics.build_comm_layout (:826-841).
    Returns (place_slots [n] int64, seg_rows [nlp] int64,
    seg_start [nlp] int64).
    """
    dev = all_local.device
    seg_rows = torch.bincount(all_local, minlength=nlp)
    seg_start = torch.zeros(nlp, dtype=torch.int64, device=dev)
    seg_start[1:] = torch.cumsum(seg_rows, dim=0)[:-1]
    order = torch.argsort(all_local, stable=True)
    occ_sorted = _run_ordinal(all_local[order])
    place_slots = torch.empty_like(all_local)
    place_slots[order] = seg_start[all_local[order]] + occ_sorted
    return place_slots, seg_rows, seg_start


def comb_dst_slot_from_topk(topk_src: torch.Tensor, ent_tok: torch.Tensor,
                            ent_logical: torch.Tensor,
                            G: int) -> torch.Tensor:
    """[n_send] int64 combine home slot tok*K + j, where j is the k-index
    of the entry's logical expert in its token's topk row.

    Parity target: the pos-table construction of
    epic_semantics.build_epic_group_layouts (:551-556, :616-626). The
    result is a permutation of [0, S*K) (asserted by the CPU tier, not on
    the hot path). G is passed explicitly to avoid a device sync.
    """
    S, K = topk_src.shape
    dev = topk_src.device
    pos = torch.zeros(S, G, dtype=torch.int64, device=dev)
    pos[torch.arange(S, device=dev).unsqueeze(1), topk_src.long()] = (
        torch.arange(K, device=dev, dtype=torch.int64).unsqueeze(0).expand(S, K)
    )
    j = pos[ent_tok, ent_logical]
    return ent_tok * K + j
