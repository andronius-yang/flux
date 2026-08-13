"""MoonEP dispatch-algorithm semantics ported to plain PyTorch.

MoonEP (MoonshotAI/MoonEP) balances MoE dispatch with dynamic redundant
experts: a planning step rebalances hot experts onto under-loaded ranks so
every rank receives exactly S*K token rows regardless of routing skew, rows
land in a static expert-grouped [NvS, H] layout consumed directly by a
grouped GEMM, duplicate top-k entries of one token to the same rank cross
the wire once, and the B = E/R hottest remote expert segments per rank get
their weights prefetched from the home rank.

MoonEP's kernels are Hopper/NVLink-domain only (TMA bulk copies, NVSwitch
multicast), so this port re-implements the ALGORITHM, not the kernels:

  * The planner below is a vectorized re-implementation of MoonEP's own
    executable oracle (tests/planning_reference.py, vendored under
    test/python/moe_ag_scatter/moonep_oracle/). The fidelity contract is
    bit-identical plans: dst / cu_seqlens / experts_to_copy / remote_stats /
    zero_fill_ranges equal via torch.equal, dedup structures equal as group
    sets. test_moonep_planner.py enforces this on MoonEP's own test cases.
  * MoonEP's rank-0 planning + hardware-multicast broadcast is replaced by
    replicated planning: every rank all-gathers the [S, K] topk routing and
    runs this deterministic integer planner on identical inputs (the oracle
    itself already runs replicated in MoonEP's tests). The greedy loops are
    kept as near-verbatim ports of the oracle (same torch ops, same dtypes,
    same tie-breaks); only the O(N) per-token parts are vectorized.

The planner is pure CPU integer math: identical inputs give identical plans
on every rank, with no floating-point or device nondeterminism.
"""

from dataclasses import dataclass

import torch

INT32_MAX = 2**31 - 1


@dataclass
class MoonEPConfig:
    """Shape math of MoonEP's Buffer context (moonep/api.py _create_context)."""

    S: int              # tokens per rank
    K: int              # topk
    E: int              # global expert count
    R: int              # EP world size
    H: int = 0          # hidden dim (unused by the planner, used by the runner)
    B: int | None = None            # prefetch slots per rank; MoonEP default E/R
    token_padding: int = 128

    def __post_init__(self):
        assert self.E % self.R == 0, f"E ({self.E}) must be divisible by R ({self.R})"
        assert isinstance(self.token_padding, int) and self.token_padding > 0
        self.epn = self.E // self.R
        if self.B is None:
            self.B = self.epn
        assert isinstance(self.B, int) and self.B > 0
        self.N = self.S * self.K
        assert 0 < self.N < INT32_MAX
        self.NvS_capacity = self.N
        # Each destination rank receives tokens from at most one remote home
        # group under the constructive planner, so at most E/R remote + E/R
        # local non-empty segments each pad up by < token_padding rows.
        self.token_padding_extra = (self.token_padding - 1) * 2 * self.epn
        self.NvS = self.NvS_capacity + self.token_padding_extra
        # dst encodes dest * NvS + local offset in int32.
        assert self.R * self.NvS - 1 <= INT32_MAX, (
            f"dst encoding overflows int32: R={self.R} NvS={self.NvS}"
        )

    def oracle_ctx(self, rank: int) -> dict:
        """ctx dict accepted by the vendored planning_reference oracle."""
        return {
            "rank": rank,
            "R": self.R,
            "E": self.E,
            "B": self.B,
            "S": self.S,
            "K": self.K,
            "NvS_capacity": self.NvS_capacity,
            "NvS": self.NvS,
            "token_padding": self.token_padding,
        }


@dataclass(frozen=True)
class RankDedupPlan:
    """Duck-typed like the oracle's ReferenceDedupPlan for dedup_check."""

    dup_groups: torch.Tensor   # [NvS, 3] (primary_loff, dup_start, dup_n)
    dup_loffs: torch.Tensor    # [NvS]
    dup_counts: torch.Tensor   # [2] (n_groups, n_dup_loffs)


@dataclass
class MoonEPPlan:
    """Replicated global plan: every rank computes this identically."""

    cfg: MoonEPConfig
    tpe: torch.Tensor              # [R, E] int32 tokens per expert per source
    alloc: torch.Tensor            # [E, R] int32 tokens of e landing on dest d
    z: torch.Tensor                # [R, R] int32 home-group -> dest migrations
    cu_seqlens: torch.Tensor       # [R, E+B] int32 padded segment ends
    expert_off: torch.Tensor       # [R, E] int32 segment start of e on dest d
    experts_to_copy: torch.Tensor  # [R, B] int32 prefetch expert ids (-1 empty)
    remote_stats: torch.Tensor     # [R, 2] int32
    zero_fill_ranges: torch.Tensor  # [R, E+B, 2] int32 (pad_start, n_pad)
    dst: torch.Tensor              # [R, N] int32, negatives = dedup'd entries
    dup_groups: torch.Tensor       # [R, NvS, 3] int32
    dup_loffs: torch.Tensor        # [R, NvS] int32
    dup_counts: torch.Tensor       # [R, 2] int32

    def rank_dedup_plan(self, rank: int, device="cpu") -> RankDedupPlan:
        return RankDedupPlan(
            dup_groups=self.dup_groups[rank].to(device),
            dup_loffs=self.dup_loffs[rank].to(device),
            dup_counts=self.dup_counts[rank].to(device),
        )

    def plan_hash(self) -> int:
        """Order-stable digest for cross-rank determinism asserts."""
        import hashlib

        h = hashlib.sha256()
        for t in (self.tpe, self.alloc, self.z, self.cu_seqlens, self.expert_off,
                  self.experts_to_copy, self.remote_stats, self.zero_fill_ranges,
                  self.dst, self.dup_counts):
            h.update(t.contiguous().numpy().tobytes())
        return int.from_bytes(h.digest()[:8], "little", signed=True)


def compute_moonep_plan(cfg: MoonEPConfig, topk_all: torch.Tensor) -> MoonEPPlan:
    """Vectorized port of the vendored MoonEP planning oracle.

    topk_all: [R, S, K] integer expert ids (any device; computed on CPU).
    Returns the full replicated plan. Sections marked "oracle-verbatim" keep
    the oracle's exact torch ops/dtypes so tie-breaking matches bit-for-bit.
    """
    R, E, B, S, K, N = cfg.R, cfg.E, cfg.B, cfg.S, cfg.K, cfg.N
    epn = cfg.epn
    CAP = cfg.NvS_capacity
    NvS = cfg.NvS
    token_padding = cfg.token_padding

    assert tuple(topk_all.shape) == (R, S, K), (
        f"topk_all must be [R={R}, S={S}, K={K}], got {tuple(topk_all.shape)}"
    )
    topk = topk_all.cpu().long().reshape(R, N)
    assert bool(((topk >= 0) & (topk < E)).all()), "expert ids out of range"

    # tokens per expert per source rank, and source-rank prefix sums
    # (the global token sequence concatenates in source-rank order).
    tpe = torch.zeros(R, E, dtype=torch.int32)
    for r in range(R):
        tpe[r] = torch.bincount(topk[r], minlength=E).to(torch.int32)
    tpe_cumsum = tpe.cumsum(dim=0)
    expert_count = tpe_cumsum[R - 1]

    # ---- Part 1: greedy surplus/deficit rebalancing (oracle-verbatim) ----
    group_tokens = torch.zeros(R, dtype=expert_count.dtype)
    for h in range(R):
        group_tokens[h] = expert_count[h * epn:(h + 1) * epn].sum()
    balance = group_tokens - CAP

    alloc = torch.zeros(E, R, dtype=torch.int32)
    for e in range(E):
        alloc[e, e // epn] = expert_count[e]

    z = torch.zeros(R, R, dtype=torch.int32)
    while True:
        h = balance.argmax()
        u = balance.argmin()
        if balance[h] <= 0:
            break
        move = -balance[u]
        z[h, u] = move
        balance[h] -= move
        balance[u] = 0

    for h in range(R):
        expert_start = h * epn
        remaining = expert_count[expert_start:expert_start + epn].clone()
        quotas = z[h].clone()
        while True:
            d = quotas.argmax()
            quota = quotas[d]
            if quota <= 0:
                break
            local_e = remaining.argmax()
            e = expert_start + local_e
            rem = remaining[local_e]
            take = torch.minimum(rem, quota)
            alloc[e, d] += take
            alloc[e, h] -= take
            remaining[local_e] -= take
            quotas[d] -= take

    if not torch.equal(alloc.sum(dim=1), expert_count):
        raise AssertionError("moonep planner: per-expert token conservation failed")
    if bool((alloc.sum(dim=0) > CAP).any().item()):
        raise AssertionError("moonep planner: rank capacity exceeded")

    # ---- Physical layout per destination rank (oracle-verbatim) ----
    alloc_cumsum = alloc.cumsum(dim=1)

    expert_off = torch.zeros(R, E, dtype=torch.int32)
    cu_seqlens = torch.zeros(R, E + B, dtype=torch.int32)
    experts_to_copy = torch.full((R, B), -1, dtype=torch.int32)
    remote_stats_all = torch.zeros(R, 2, dtype=torch.int32)
    zero_fill_by_rank = torch.zeros(R, E + B, 2, dtype=torch.int32)

    for d in range(R):
        local_start = d * epn
        local_end = local_start + epn

        remote_experts = [
            e for e in range(E)
            if alloc[e, d].item() > 0 and not (local_start <= e < local_end)
        ]
        remote_experts.sort(key=lambda e: (alloc[e, d].item(), e), reverse=True)
        remote_stats_all[d, 0] = len(remote_experts)
        for b, e in enumerate(remote_experts[:B]):
            experts_to_copy[d, b] = e
            remote_stats_all[e // epn, 1] += 1

        experts_to_prefetch = set(remote_experts[:B])
        start = 0
        # Physical token order follows the VM group id: 0..E-1 are the global
        # expert groups (prefetch-selected experts skipped), E..E+B-1 the
        # prefetch slots of the selected remote experts.
        for g in range(E + B):
            cnt = 0
            expert_id = -1
            if g < E:
                if g not in experts_to_prefetch:
                    cnt = alloc[g, d].item()
                    expert_id = g
            else:
                b = g - E
                if experts_to_copy[d, b].item() >= 0:
                    expert_id = experts_to_copy[d, b].item()
                    cnt = alloc[expert_id, d].item()

            end = start + cnt
            if cnt > 0:
                padded = ((cnt + token_padding - 1) // token_padding) * token_padding
            else:
                padded = 0
            aligned_end = start + padded
            cu_seqlens[d, g] = aligned_end
            if cnt > 0:
                expert_off[d, expert_id] = start
                n_pad = padded - cnt
                if n_pad > 0:
                    zero_fill_by_rank[d, g, 0] = end
                    zero_fill_by_rank[d, g, 1] = n_pad
            start = aligned_end

        if cu_seqlens[d].max().item() > NvS:
            raise AssertionError("moonep planner: padded layout exceeds NvS")

    # ---- Part 2 (vectorized): per-entry canonical dst for every rank ----
    # local_cnt[i] = occurrence index of expert e=topk[r,i] among earlier
    # entries of the same rank; global position g = tpe_cumsum[r-1, e] +
    # local_cnt; dest = first rank whose alloc_cumsum[e] exceeds g.
    positions = torch.arange(N, dtype=torch.int64)
    ac64 = alloc_cumsum.to(torch.int64)          # [E, R]
    eo64 = expert_off.to(torch.int64)            # [R, E]
    dst_raw = torch.zeros(R, N, dtype=torch.int64)
    for r in range(R):
        e = topk[r]                              # [N] int64
        order = torch.argsort(e, stable=True)
        sorted_e = e[order]
        boundary = torch.ones(N, dtype=torch.bool)
        if N > 1:
            boundary[1:] = sorted_e[1:] != sorted_e[:-1]
        run_start = torch.where(boundary, positions, torch.full_like(positions, -1))
        run_start = torch.cummax(run_start, dim=0).values
        occ_sorted = positions - run_start
        local_cnt = torch.empty(N, dtype=torch.int64)
        local_cnt[order] = occ_sorted

        prev = torch.zeros(E, dtype=torch.int64)
        if r > 0:
            prev = tpe_cumsum[r - 1].to(torch.int64)
        g = prev[e] + local_cnt                  # [N]

        seq = ac64[e]                            # [N, R] rows non-decreasing
        dest = torch.searchsorted(seq, g.unsqueeze(1), right=True).squeeze(1)
        if bool((dest >= R).any()):
            raise AssertionError("moonep planner: no destination rank found")
        prev_alloc = torch.where(
            dest > 0,
            seq.gather(1, (dest - 1).clamp(min=0).unsqueeze(1)).squeeze(1),
            torch.zeros_like(g),
        )
        seg_pos = g - prev_alloc
        base_off = eo64[dest, e]
        dst_raw[r] = dest * NvS + base_off + seg_pos

    # ---- Part 3 (vectorized): wire dedup encoding + dedup structures ----
    # Within one token's top-K, only the first entry per destination rank
    # keeps its slot offset; later ones encode -raw - 1 (raw recoverable for
    # the per-entry weight scatter). Group order per destination rank is
    # (src_rank, token) row-major — identical to the oracle's append order,
    # since one token contributes at most one group per destination.
    d3 = torch.div(dst_raw, NvS, rounding_mode="floor").reshape(R, S, K)
    l3 = (dst_raw % NvS).reshape(R, S, K)
    dup_entry = torch.zeros(R, S, K, dtype=torch.bool)
    for k in range(1, K):
        dup_entry[:, :, k] = (d3[:, :, :k] == d3[:, :, k:k + 1]).any(dim=2)

    dst = torch.where(dup_entry.reshape(R, N), -dst_raw - 1, dst_raw).to(torch.int32)

    dup_groups = torch.zeros(R, NvS, 3, dtype=torch.int32)
    dup_loffs = torch.zeros(R, NvS, dtype=torch.int32)
    dup_counts = torch.zeros(R, 2, dtype=torch.int32)
    for dest in range(R):
        member = d3 == dest                              # [R, S, K]
        cnt = member.sum(dim=2)                          # [R, S]
        grp = cnt >= 2
        n_groups = int(grp.sum().item())
        if n_groups == 0:
            continue
        first_k = member.to(torch.int64).argmax(dim=2)   # first True per (src, s)
        primary = l3.gather(2, first_k.unsqueeze(2)).squeeze(2)   # [R, S]
        dup_sel = member.clone()
        dup_sel.scatter_(2, first_k.unsqueeze(2), False)          # drop firsts
        # boolean indexing flattens row-major (src, s, k): per-group segments
        # are contiguous and ordered by (src, s), dups within a group by k.
        loffs_flat = l3[dup_sel].to(torch.int32)
        dup_n = dup_sel.sum(dim=2)[grp].to(torch.int32)           # (src, s) order
        total = int(dup_n.sum().item())
        starts = torch.zeros(n_groups, dtype=torch.int32)
        if n_groups > 1:
            starts[1:] = torch.cumsum(dup_n, dim=0)[:-1]
        dup_groups[dest, :n_groups, 0] = primary[grp].to(torch.int32)
        dup_groups[dest, :n_groups, 1] = starts
        dup_groups[dest, :n_groups, 2] = dup_n
        dup_loffs[dest, :total] = loffs_flat
        dup_counts[dest, 0] = n_groups
        dup_counts[dest, 1] = total

    return MoonEPPlan(
        cfg=cfg,
        tpe=tpe,
        alloc=alloc,
        z=z,
        cu_seqlens=cu_seqlens,
        expert_off=expert_off,
        experts_to_copy=experts_to_copy,
        remote_stats=remote_stats_all,
        zero_fill_ranges=zero_fill_by_rank,
        dst=dst,
        dup_groups=dup_groups,
        dup_loffs=dup_loffs,
        dup_counts=dup_counts,
    )


# ---------------------------------------------------------------------------
# Dispatch transport layout + per-rank runner
# ---------------------------------------------------------------------------
#
# MoonEP's dispatch writes each representative row one-sided over NVLink
# directly into its final expert-grouped slot. Without a shared address space
# across Slingshot nodes, the port stages: pack representative rows sorted by
# destination -> all_to_all_single -> destination-local placement scatter to
# the plan-decided slots -> zero-fill -> local duplicate expansion. The plan
# is replicated, so the receiver derives every placement index locally (zero
# extra handshakes); wire rows and bytes equal MoonEP's dedup semantics
# exactly. Per-entry route weights (fp32, every top-k entry incl. dedup'd
# ones) ride a second small alltoallv, mirroring MoonEP's meta-channel
# weight transfer.


@dataclass
class RankCommLayout:
    """Index metadata for one rank, precomputed from the replicated plan
    (untimed-metadata contract: in a real system this is planner output)."""

    rank: int
    # sender side
    send_row_index: torch.Tensor    # [n_send] int64 local token row per packed row
    send_counts: list               # [R] rows this rank sends to each dest
    send_entry_index: torch.Tensor  # [n_ent_send] int64 entry idx per packed weight
    send_entry_counts: list         # [R] weight entries sent to each dest
    # receiver side
    recv_counts: list               # [R] rows received from each src
    recv_entry_counts: list         # [R] weight entries received from each src
    place_loffs: torch.Tensor       # [n_recv] int64 slot per received row
    weight_loffs: torch.Tensor      # [n_ent_recv] int64 slot per received weight
    zero_rows: torch.Tensor         # int64 pad rows to clear
    dup_primary: torch.Tensor       # [n_dup] int64 source row of each dup copy
    dup_target: torch.Tensor        # [n_dup] int64 dup slot to fill
    # GEMM segments: (vm_group, seg_start, seg_end_padded, expert_id)
    gemm_segments: list


def build_comm_layout(plan: MoonEPPlan, rank: int) -> RankCommLayout:
    cfg = plan.cfg
    R, K, E, B, NvS = cfg.R, cfg.K, cfg.E, cfg.B, cfg.NvS
    epn = cfg.epn

    enc = plan.dst.long()                     # [R, N]
    raw = torch.where(enc < 0, -enc - 1, enc)
    dest = torch.div(raw, NvS, rounding_mode="floor")
    loff = raw % NvS
    rep = enc >= 0

    # Sender: representative entries stable-sorted by destination. Stable
    # sort preserves entry order within a destination, which is exactly the
    # order the receiver reconstructs below.
    my_dest = dest[rank]
    my_rep = rep[rank]
    rep_idx = my_rep.nonzero(as_tuple=True)[0]
    order = torch.argsort(my_dest[rep_idx], stable=True)
    send_entries = rep_idx[order]
    send_row_index = torch.div(send_entries, K, rounding_mode="floor")
    send_counts = torch.bincount(my_dest[rep_idx], minlength=R).tolist()

    # Per-entry weights: ALL entries (dedup'd included) grouped by dest.
    all_idx = torch.arange(cfg.N, dtype=torch.int64)
    worder = torch.argsort(my_dest, stable=True)
    send_entry_index = all_idx[worder]
    send_entry_counts = torch.bincount(my_dest, minlength=R).tolist()

    # Receiver: rows arrive src-major, entry-order within src — derive the
    # slot of each incoming row/weight from the replicated plan.
    place, wplace, recv_counts, recv_entry_counts = [], [], [], []
    for src in range(R):
        m = (dest[src] == rank) & rep[src]
        place.append(loff[src][m])
        recv_counts.append(int(m.sum()))
        wm = dest[src] == rank
        wplace.append(loff[src][wm])
        recv_entry_counts.append(int(wm.sum()))
    place_loffs = torch.cat(place) if place else torch.zeros(0, dtype=torch.int64)
    weight_loffs = torch.cat(wplace) if wplace else torch.zeros(0, dtype=torch.int64)

    # Zero-fill rows from the plan's ranges.
    zf = plan.zero_fill_ranges[rank]
    zero_rows = torch.cat([
        torch.arange(int(s), int(s) + int(n), dtype=torch.int64)
        for s, n in zf.tolist() if int(n) > 0
    ]) if int(zf[:, 1].sum()) > 0 else torch.zeros(0, dtype=torch.int64)

    # Local duplicate expansion (epilogue): primary row -> each dup slot.
    ng = int(plan.dup_counts[rank, 0])
    ndup = int(plan.dup_counts[rank, 1])
    groups = plan.dup_groups[rank, :ng].long()
    dup_target = plan.dup_loffs[rank, :ndup].long()
    dup_primary = (
        torch.repeat_interleave(groups[:, 0], groups[:, 2])
        if ng > 0 else torch.zeros(0, dtype=torch.int64)
    )

    # GEMM segments over cu_seqlens (padded rows included, MoonEP contract).
    cu = plan.cu_seqlens[rank].tolist()
    e2c = plan.experts_to_copy[rank].tolist()
    segments = []
    prev = 0
    for g in range(E + B):
        end = cu[g]
        if end > prev:
            expert_id = g if g < E else e2c[g - E]
            segments.append((g, prev, end, expert_id))
        prev = end

    return RankCommLayout(
        rank=rank,
        send_row_index=send_row_index,
        send_counts=send_counts,
        send_entry_index=send_entry_index,
        send_entry_counts=send_entry_counts,
        recv_counts=recv_counts,
        recv_entry_counts=recv_entry_counts,
        place_loffs=place_loffs,
        weight_loffs=weight_loffs,
        zero_rows=zero_rows,
        dup_primary=dup_primary,
        dup_target=dup_target,
        gemm_segments=segments,
    )


class MoonEPLayer0Runner:
    """One rank's MoonEP-semantics layer0 pipeline on NCCL + local scatters.

    Phase methods are intentionally single-purpose so the driver can bracket
    each with CUDA events: pack() / a2av() / place_and_epilogue() (scatter +
    zero-fill + dup expansion) / prefetch() / gemm().
    """

    def __init__(self, plan: MoonEPPlan, rank: int, group, device,
                 dtype=torch.bfloat16, ffn_size_shard: int = 0):
        import torch.distributed as dist  # local import: keep module CPU-importable

        self.dist = dist
        self.plan = plan
        self.cfg = plan.cfg
        self.rank = rank
        self.group = group
        self.device = device
        self.dtype = dtype
        self.ffn_size_shard = ffn_size_shard
        # Constructor state only, not the port's default: the traffic driver
        # defaults to --transport nvshmem (fidelity-first, 2026-08-11) and
        # --prefetch_transport getmem (2026-08-12), calling enable_nvshmem()
        # / enable_getmem_prefetch() after construction; NCCL on either path
        # is the explicit opt-out the historical sweep arms pin.
        self.transport = "nccl"
        # Weight-movement path: "nccl" (batch_isend_irecv, declared port
        # artifact) until enable_getmem_prefetch() swaps in the one-sided
        # pull. weight_home is the symmetric-heap shard once enabled; the
        # GEMM's local-expert branch reads it in place of local_weights.
        self.prefetch_impl = "nccl"
        self.weight_home = None
        cfg = self.cfg

        lay = build_comm_layout(plan, rank)
        self.lay = lay
        dev = device
        self.send_row_index = lay.send_row_index.to(dev)
        self.send_entry_row = (lay.send_entry_index // cfg.K).to(dev)
        self.send_entry_k = (lay.send_entry_index % cfg.K).to(dev)
        self.place_loffs = lay.place_loffs.to(dev)
        self.weight_loffs = lay.weight_loffs.to(dev)
        self.zero_rows = lay.zero_rows.to(dev)
        self.dup_primary = lay.dup_primary.to(dev)
        self.dup_target = lay.dup_target.to(dev)

        self.n_send = len(lay.send_row_index)
        self.n_recv = sum(lay.recv_counts)
        self.n_ent_recv = sum(lay.recv_entry_counts)
        assert self.n_recv <= cfg.NvS_capacity

        H = cfg.H
        self.send_buf = torch.empty(self.n_send, H, dtype=dtype, device=dev)
        self.recv_buf = torch.empty(self.n_recv, H, dtype=dtype, device=dev)
        self.wsend_buf = torch.empty(cfg.N, dtype=torch.float32, device=dev)
        self.wrecv_buf = torch.empty(self.n_ent_recv, dtype=torch.float32, device=dev)
        self.hidden_buf = torch.zeros(cfg.NvS, H, dtype=dtype, device=dev)
        self.weights_buf = torch.zeros(cfg.NvS, dtype=torch.float32, device=dev)
        self.total_rows = int(plan.cu_seqlens[rank, -1])
        if ffn_size_shard:
            self.out_buf = torch.zeros(
                self.total_rows, ffn_size_shard, dtype=dtype, device=dev
            )
            self.prefetch_w = torch.zeros(
                cfg.B, ffn_size_shard, H, dtype=dtype, device=dev
            )

        # Every g < E segment must be a LOCAL expert: with MoonEP's default
        # B = E/R every remote expert with tokens gets a prefetch slot (each
        # dest receives from at most one home group). A smaller B would
        # leave token segments without weights — fail loudly.
        lo = cfg.epn * rank
        for g, _, _, e in lay.gemm_segments:
            if g < cfg.E:
                assert lo <= e < lo + cfg.epn, (
                    f"rank {rank}: non-prefetched remote expert segment "
                    f"(expert {e}, group {g}); use MoonEP default B=E/R"
                )

        # Prefetch transfer list (replicated plan -> every rank knows all
        # pairs): (dest, slot b, expert e, home rank).
        self.prefetch_pairs = []
        e2c = plan.experts_to_copy.tolist()
        for d in range(cfg.R):
            for b in range(cfg.B):
                e = e2c[d][b]
                if e >= 0:
                    self.prefetch_pairs.append((d, b, e, e // cfg.epn))

        # Wire accounting (bytes actually crossing between ranks).
        self.send_rows_by_dest = lay.send_counts
        self.wire_row_bytes = H * self.send_buf.element_size()

    # -- transport selection ----------------------------------------------

    def _pair_count_matrices(self):
        """[R, R] representative-row and all-entry counts from the plan
        (replicated, so identical on every rank)."""
        cfg = self.cfg
        enc = self.plan.dst.long()
        raw = torch.where(enc < 0, -enc - 1, enc)
        dest = torch.div(raw, cfg.NvS, rounding_mode="floor")
        rep = enc >= 0
        rows = torch.stack([
            torch.bincount(dest[s][rep[s]], minlength=cfg.R) for s in range(cfg.R)
        ])
        ents = torch.stack([
            torch.bincount(dest[s], minlength=cfg.R) for s in range(cfg.R)
        ])
        return rows, ents

    def enable_nvshmem(self, local_world_size: int, num_comm_sm: int = 8):
        """M4a: swap NCCL for flux's one-sided NVSHMEM All2AllSingle.

        Live path (a2a_single_kernel_v2; correction 2026-08-11 — the
        putmem_signal kernel in all2all_single_2d_impl.cu is dead code,
        never launched): cudaMemcpyAsync of the send buffer into symmetric
        staging, one nvshmemx_putmem_nbi_block per destination into the
        fixed slot max_split*n_dim*my_rank of the destination's staging, a
        scalar copy-out, and TWO nvshmemx_barrier_on_stream team barriers
        per forward. Put-then-barrier mirrors MoonEP's real dispatch (TMA
        S2G push + one system-scope cross_rank_barrier at kernel exit);
        still sender-driven, receiver-passive on the wire. The staged
        layout and every plan/placement index stay byte-identical to the
        NCCL arm. Requires flux.init_flux_shm(group) to have run. max_split
        (the symmetric per-(src,dst) slot size) is derived from the
        replicated plan, so all ranks construct identically-sized ops."""
        import flux  # GPU-side only

        rows, ents = self._pair_count_matrices()
        max_split_rows = max(int(rows.max()), 1)
        max_split_ents = max(int(ents.max()), 1)
        self._a2a_hidden = flux.All2AllSingle(
            self.group, max_split_rows, self.cfg.H, local_world_size,
            self.dtype,
        )
        self._a2a_weights = flux.All2AllSingle(
            self.group, max_split_ents, 1, local_world_size,
            torch.float32,
        )
        dev = self.device
        self._in_splits_h = torch.tensor(
            self.lay.send_counts, dtype=torch.int32, device=dev)
        self._out_splits_h = torch.tensor(
            self.lay.recv_counts, dtype=torch.int32, device=dev)
        self._in_splits_w = torch.tensor(
            self.lay.send_entry_counts, dtype=torch.int32, device=dev)
        self._out_splits_w = torch.tensor(
            self.lay.recv_entry_counts, dtype=torch.int32, device=dev)
        self._num_comm_sm = num_comm_sm
        self.transport = "nvshmem"

    def enable_getmem_prefetch(self, local_weights: torch.Tensor,
                               num_comm_sm: int = 8,
                               chunk_bytes: int = 4 << 20,
                               device_kernel: bool = True):
        """Swap the NCCL isend/irecv weight movement for the one-sided
        NVSHMEM getmem pull (flux.WeightPrefetchGetmem) — the authentic
        analog of MoonEP's destination-side prefetch (NR-12 fact 6: the
        receiver remote-READS the home rank's immutable weight rows; zero
        cross-rank signaling, so completion is locally observable and an
        event after prefetch() is a sufficient join). This rank's [epn,
        ffn_shard, H] home shard moves onto the symmetric heap ONCE here
        (upstream's memory model: weights always remotely readable);
        gemm()'s local branch reads it thereafter — residency moves, it
        does not grow. Cross-node pulls are proxy-mediated CXI on this
        fabric (extrapolation per NR-12 fact 7: MoonEP has no inter-node
        path; judged by preservation of intra-domain semantics). No team
        barriers anywhere, so this op is safe on an overlap side stream
        beside the token a2av and needs NO dedicated communicator.
        Requires flux.init_flux_shm(group); collective (all ranks must
        call, identical shapes)."""
        import flux  # GPU-side only

        cfg = self.cfg
        assert self.ffn_size_shard > 0, "getmem prefetch needs weight shapes"
        self._prefetch_op = flux.WeightPrefetchGetmem(
            self.group, cfg.epn, cfg.B, self.ffn_size_shard, cfg.H, self.dtype,
        )
        self.weight_home = self._prefetch_op.weight_home()
        self.weight_home.copy_(local_weights)
        # The op owns the prefetch slots ON the symmetric heap: a
        # proxy-mediated (cross-node) get segfaults into an ordinary
        # cudaMalloc destination (CXI needs the local buffer registered;
        # found 2026-08-11 by the a2av_comm_bench prefetch mode). Upstream's
        # slots are rows [E, E+B) of the same mapped weight tensor, so this
        # is the faithful layout anyway. Replaces the ordinary prefetch_w.
        self.prefetch_w = self._prefetch_op.prefetch_slots()
        # All of MY incoming pulls, local pairs included (getmem from the
        # own PE is a local SM copy — one uniform path, like upstream).
        pairs = [
            (b, home, e % cfg.epn)
            for d, b, e, home in self.prefetch_pairs
            if d == self.rank
        ]
        pairs_cpu = torch.tensor(pairs, dtype=torch.int32).reshape(-1, 3)
        self._prefetch_op.set_pairs(pairs_cpu)
        self._prefetch_num_comm_sm = num_comm_sm
        self._prefetch_chunk_bytes = chunk_bytes
        self._prefetch_device_kernel = device_kernel
        self.prefetch_impl = "getmem"

    # -- timed phases -----------------------------------------------------

    def pack(self, inputs_shard: torch.Tensor, route_weights: torch.Tensor):
        """Gather representative rows (dest-sorted) and per-entry weights."""
        torch.index_select(inputs_shard, 0, self.send_row_index, out=self.send_buf)
        w = route_weights[self.send_entry_row, self.send_entry_k]
        self.wsend_buf.copy_(w)

    def a2av(self):
        """Wire: representative rows + per-entry route weights."""
        if self.transport == "nvshmem":
            self._a2a_hidden.forward(
                self.send_buf, self.recv_buf,
                self._in_splits_h, self._out_splits_h, self._num_comm_sm,
            )
            self._a2a_weights.forward(
                self.wsend_buf.view(-1, 1), self.wrecv_buf.view(-1, 1),
                self._in_splits_w, self._out_splits_w, self._num_comm_sm,
            )
            return
        self.dist.all_to_all_single(
            self.recv_buf,
            self.send_buf,
            output_split_sizes=self.lay.recv_counts,
            input_split_sizes=self.lay.send_counts,
            group=self.group,
        )
        self.dist.all_to_all_single(
            self.wrecv_buf,
            self.wsend_buf,
            output_split_sizes=self.lay.recv_entry_counts,
            input_split_sizes=self.lay.send_entry_counts,
            group=self.group,
        )

    def place_and_epilogue(self):
        """Placement scatter + zero-fill (MoonEP zero warp) + dup expansion
        (MoonEP dispatch_epilogue, purely local)."""
        self.hidden_buf.index_copy_(0, self.place_loffs, self.recv_buf)
        self.weights_buf.index_copy_(0, self.weight_loffs, self.wrecv_buf)
        if self.zero_rows.numel():
            self.hidden_buf.index_fill_(0, self.zero_rows, 0)
        if self.dup_target.numel():
            self.hidden_buf.index_copy_(
                0, self.dup_target, self.hidden_buf.index_select(0, self.dup_primary)
            )

    def prefetch(self, local_weights: torch.Tensor, group=None):
        """Redundant-expert weight movement (MoonEP prefetch_weight):
        local_weights is this rank's [epn, ffn_shard, H] home shard. Pass a
        dedicated ProcessGroup (separate NCCL communicator) when running on
        an overlap stream, so these P2Ps run concurrently with the dispatch
        collectives instead of serializing on the same communicator. On the
        getmem path (enable_getmem_prefetch) local_weights and group are
        ignored: the pull reads the symmetric weight home, source ranks are
        passive, and no communicator exists to serialize on."""
        if self.prefetch_impl == "getmem":
            self._prefetch_op.forward(
                self.prefetch_w,
                self._prefetch_num_comm_sm,
                self._prefetch_chunk_bytes,
                self._prefetch_device_kernel,
            )
            return
        group = group if group is not None else self.group
        ops = []
        P2POp = self.dist.P2POp
        for d, b, e, home in self.prefetch_pairs:
            if d == self.rank and home == self.rank:
                self.prefetch_w[b].copy_(local_weights[e % self.cfg.epn])
            elif d == self.rank:
                ops.append(P2POp(self.dist.irecv, self.prefetch_w[b], peer=home,
                                 group=group))
            elif home == self.rank:
                ops.append(P2POp(self.dist.isend,
                                 local_weights[e % self.cfg.epn].contiguous(),
                                 peer=d, group=group))
        if ops:
            for req in self.dist.batch_isend_irecv(ops):
                req.wait()

    def gemm(self, gemm_only_op, local_weights: torch.Tensor):
        """Per-segment GEMM over cu_seqlens (padded rows computed, MoonEP
        contract). Local expert segments use the home shard, prefetch slots
        the prefetched weights. When the getmem path homed the weights on
        the symmetric heap, the local branch reads that tensor (symmetric
        memory is ordinary device memory to local kernels)."""
        cfg = self.cfg
        lo = cfg.epn * self.rank
        home = self.weight_home if self.weight_home is not None else local_weights
        for g, start, end, expert_id in self.lay.gemm_segments:
            w = (
                home[expert_id - lo]
                if g < cfg.E
                else self.prefetch_w[g - cfg.E]
            )
            gemm_only_op.forward(
                self.hidden_buf[start:end],
                w,
                output_buf=self.out_buf[start:end],
                fast_accum=False,
            )

    # -- accounting -------------------------------------------------------

    def wire_matrix_row(self) -> list:
        """Bytes this rank puts on the wire per destination (hidden rows)."""
        return [c * self.wire_row_bytes for c in self.send_rows_by_dest]

    def prefetch_recv_bytes(self) -> int:
        n = sum(
            1 for d, _, _, home in self.prefetch_pairs
            if d == self.rank and home != self.rank
        )
        return n * self.ffn_size_shard * self.cfg.H * self.prefetch_w.element_size()
