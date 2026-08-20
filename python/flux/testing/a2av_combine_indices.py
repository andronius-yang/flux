"""Python builders for the a2av layer1 combine indices (TopkReduceScatterOp
and GemmGroupedV2GatherRSOp amortized mode).

Moved verbatim from test/python/moe_gather_rs/test_moe_gather_rs_traffic.py
(which re-imports them) so non-test consumers — the EPIC baseline's
combine-only per-group TopkReduceScatterOp protocol — can import them from
flux.testing. The executable spec remains
test/python/moe_gather_rs/test_a2av_combine_sim.py (simulate_compress), and
FLUX_A2AV_RS_CHECK_IDENTITY=1 cross-checks the op's internal arithmetic
against this sort-based math.
"""

import torch


def build_a2av_combine_indices_dev(routing_idx, splits, rank, world_size,
                                   topk):
    """Device-agnostic twin of build_a2av_combine_indices: computes on the
    inputs' device and returns int32 tensors on that device (no forced
    .cpu()/.cuda() hops). Bit-parity with the CPU original is tested in
    test_epic_gpu_planner.py; used by the per-iteration timed planners
    (SCHEMA rule 5)."""
    routing_idx = routing_idx.long()
    dev = routing_idx.device
    m_full = routing_idx.numel()
    cpr = m_full // world_size
    splits = splits.long()
    n_experts_per_rank = splits.numel() // world_size
    ep_m_start = int(splits[: rank * n_experts_per_rank].sum())
    m_this_ep = int(
        splits[rank * n_experts_per_rank:(rank + 1) * n_experts_per_rank]
        .sum())
    iota_m = torch.arange(m_full, dtype=torch.long, device=dev)
    copy_of_row = torch.empty(m_full, dtype=torch.long,
                              device=dev).scatter_(0, routing_idx, iota_m)
    copy_of_row = copy_of_row[ep_m_start:ep_m_start + m_this_ep]
    home = copy_of_row // cpr
    pack_index = (home * m_this_ep + torch.arange(
        m_this_ep, dtype=torch.long, device=dev)).argsort()
    splits_cum = splits.cumsum(0)
    my_copies = routing_idx[rank * cpr:(rank + 1) * cpr]
    e_of = torch.searchsorted(splits_cum, my_copies, right=True)
    iota_c = torch.arange(cpr, dtype=torch.long, device=dev)
    perm = (e_of * cpr + iota_c).argsort()
    reduce_index = torch.empty(cpr, dtype=torch.long,
                               device=dev).scatter_(0, perm, iota_c)
    return pack_index.int(), reduce_index.int()


def build_a2av_unique_counts_dev(choosed_experts, world_size, nnodes,
                                 experts_per_rank):
    """Device-agnostic twin of build_a2av_unique_counts (same math, on the
    input's device, no .cpu())."""
    L = world_size // nnodes
    ntokens = choosed_experts.size(0)
    tokens_per_rank = ntokens // world_size
    owner = choosed_experts.long() // experts_per_rank
    on_node = torch.zeros(ntokens, nnodes, dtype=torch.bool,
                          device=owner.device)
    on_node.scatter_(1, owner // L, True)
    return on_node.view(world_size, tokens_per_rank, nnodes).sum(1).int()


def build_a2av_compress_indices_dev(
    routing_idx, splits, unique_counts, rank, world_size, nnodes, topk
):
    """Device-agnostic twin of build_a2av_compress_indices: identical math
    on the inputs' device; the two small per-rank scalar loops of the CPU
    original (Cp_col, rem_base) are vectorized. Returns int32 tensors on
    the compute device."""
    routing_idx = routing_idx.long()
    dev = routing_idx.device
    splits = splits.long().to(dev)
    U = unique_counts.long().to(dev)
    m_full = routing_idx.numel()
    W, NN = world_size, nnodes
    L = W // NN
    cpr = m_full // W
    ntok_local = cpr // topk
    ntokens = m_full // topk
    nex = splits.numel()
    E_loc = nex // W
    my_node, my_lr = rank // L, rank % L
    iota_m = torch.arange(m_full, dtype=torch.long, device=dev)
    splits_cum = splits.cumsum(0)
    e_of_copy = torch.searchsorted(splits_cum, routing_idx,
                                   right=True).clamp_max_(nex - 1)
    owner = e_of_copy // E_loc
    home = iota_m // cpr
    kmax = torch.iinfo(torch.long).max

    C = torch.zeros(W, W, dtype=torch.long, device=dev)
    C.index_put_((owner, home),
                 torch.ones(m_full, dtype=torch.long, device=dev),
                 accumulate=True)
    recv_off_C = torch.cat([
        torch.zeros(1, dtype=torch.long, device=dev),
        C[:, rank].cumsum(0)[:-1]])
    s_ids = torch.arange(W, dtype=torch.long, device=dev)
    Cp_col = torch.where(
        (s_ids // L) == my_node, C[:, rank],
        torch.where((s_ids % L) == my_lr, U[rank, s_ids // L],
                    torch.zeros_like(s_ids)))
    recv_off_Cp = torch.cat([
        torch.zeros(1, dtype=torch.long, device=dev),
        Cp_col.cumsum(0)[:-1]])

    owner_node, home_node = owner // L, home // L
    conv_mask = ((owner_node == my_node) & (home_node != my_node)
                 & (home % L == my_lr))
    conv_total = int(conv_mask.sum())
    if conv_total > 0:
        seg = home_node - (home_node > my_node).long()
        ls = owner % L
        conv_key = ((((seg * L + ls) * nex + e_of_copy) * m_full + iota_m)
                    .masked_fill(~conv_mask, kmax))
        conv_copy = conv_key.argsort()[:conv_total]
        wkey = seg[conv_copy] * ntokens + conv_copy // topk
        worder = wkey.argsort(stable=True)
        wire_copy = worder
        _, counts = torch.unique_consecutive(wkey[worder],
                                             return_counts=True)
        wire_ptr = torch.cat([
            torch.zeros(1, dtype=torch.long, device=dev),
            counts.cumsum(0)])
    else:
        wire_total = int(sum(
            int(U[tn * L + my_lr, my_node])
            for tn in range(NN) if tn != my_node
        ))
        assert wire_total == 0, (
            f"compress: conv_total == 0 but unique_counts claims "
            f"{wire_total} wire rows (inconsistent transposed U)"
        )
        wire_ptr = torch.zeros(1, dtype=torch.long, device=dev)
        wire_copy = torch.empty(0, dtype=torch.long, device=dev)

    iota_c = torch.arange(cpr, dtype=torch.long, device=dev)
    e_my = e_of_copy[rank * cpr:(rank + 1) * cpr]
    owner_my = owner[rank * cpr:(rank + 1) * cpr]
    perm = (e_my * cpr + iota_c).argsort()
    rows_C = torch.empty(cpr, dtype=torch.long,
                         device=dev).scatter_(0, perm, iota_c)
    rows_Cp = rows_C - recv_off_C[owner_my] + recv_off_Cp[owner_my]
    K = topk + NN + 1
    tl = iota_c // topk
    own_mask = owner_my // L == my_node
    own_total = int(own_mask.sum())
    key_own = (tl * K + (iota_c - tl * topk)).masked_fill(~own_mask, kmax)
    ord_own = key_own.argsort()
    own_rows = rows_Cp[ord_own][:own_total]
    own_keys = key_own[ord_own][:own_total]
    onode = owner_my // L
    flags = torch.zeros(ntok_local * NN, dtype=torch.long, device=dev)
    flags.scatter_(0, tl * NN + onode, 1)
    flags = flags.view(ntok_local, NN)
    flags[:, my_node] = 0
    rem_total = int(flags.sum())
    pos = flags.cumsum(0) - flags
    m_ids = torch.arange(NN, dtype=torch.long, device=dev)
    rem_base = torch.where(
        m_ids != my_node, recv_off_Cp[m_ids * L + my_lr],
        torch.zeros_like(m_ids))
    rem_rows2d = pos + rem_base.view(1, NN)
    tl_col = torch.arange(ntok_local, dtype=torch.long,
                          device=dev).view(-1, 1)
    m_row = m_ids.view(1, -1)
    key_rem = ((tl_col * K + topk + m_row)
               .masked_fill(flags.eq(0), kmax).reshape(-1))
    ord_rem = key_rem.argsort()
    rem_rows = rem_rows2d.reshape(-1)[ord_rem][:rem_total]
    rem_keys = key_rem[ord_rem][:rem_total]
    keys_all = torch.cat([own_keys, rem_keys])
    vals_all = torch.cat([own_rows, rem_rows])
    order = keys_all.argsort()
    red_row = vals_all[order]
    red_ptr = torch.cat([
        torch.zeros(1, dtype=torch.long, device=dev),
        torch.bincount(keys_all[order] // K,
                       minlength=ntok_local).cumsum(0)])
    return (
        wire_ptr.int(),
        wire_copy.int(),
        red_ptr.int(),
        red_row.int(),
    )


def build_a2av_combine_indices(routing_idx, split_cpu, rank, world_size, topk):
    """Mirror-layout routing plan for the a2av_hier combine, on CPU. Same ordering
    contract as layer0 a2av (copy-index tie-break); the op builds the identical
    tensors internally when these are not passed (FLUX_A2AV_RS_CHECK_IDENTITY=1
    cross-checks the op's arithmetic-identity path against this sort-based math).

    - pack_index[p]: gemm row at send-panel position p == this rank's gemm rows
      stably ordered by (token-home rank, row) -- within a home that is
      (expert, copy) order, matching layer0's recv layout.
    - reduce_index[t*topk+j]: recv-panel row of local copy (t, j) == inverse of
      the (expert, copy-index) sort of this rank's own copies (layer0's pack key).

    Delegates to the device-agnostic _dev twin (single source of truth;
    parity of the twin vs the pre-2026-08-20 body is pinned in
    test_epic_gpu_planner.py) and keeps the historical CPU-in/CUDA-out
    contract for existing callers."""
    pack_index, reduce_index = build_a2av_combine_indices_dev(
        routing_idx.long().cpu(), split_cpu.cpu(), rank, world_size, topk)
    return pack_index.cuda(), reduce_index.cuda()


def build_a2av_unique_counts(choosed_experts, world_size, nnodes, experts_per_rank):
    """Transposed-U dedup counts for the compress combine: U[d][n] = distinct
    tokens homed at rank d with >= 1 copy owned on node n (the layer0 U-matrix
    recipe consumed transposed). CPU host metadata, like splits_per_source
    (pre-rule-5 drivers computed it once at setup; new-accounting drivers
    recompute per iteration via the _dev twin)."""
    return build_a2av_unique_counts_dev(
        choosed_experts.cpu(), world_size, nnodes, experts_per_rank)


def build_a2av_compress_indices(
    routing_idx, split_cpu, unique_counts, rank, world_size, nnodes, topk
):
    """CPU twin of the op's build_a2av_compress_indices (executable spec:
    test_a2av_combine_sim.py::simulate_compress). Returns the compress CSRs a
    fused layer0+layer1 pipeline would hand over precomputed:
    wire_ptr/wire_copy (source side, wire row -> conv-panel rows) and
    red_ptr/red_row (destination side, local token -> C' recv rows).

    Delegates to the device-agnostic _dev twin (whose parity vs the
    pre-2026-08-20 body — including the vectorized Cp_col/rem_base loops —
    is pinned in test_epic_gpu_planner.py); keeps the historical
    CPU-in/CUDA-out contract for existing callers."""
    wp, wc, rp, rr = build_a2av_compress_indices_dev(
        routing_idx.long().cpu(), split_cpu.cpu(), unique_counts.cpu(),
        rank, world_size, nnodes, topk)
    return wp.cuda(), wc.cuda(), rp.cuda(), rr.cuda()
