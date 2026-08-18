"""EPIC baseline semantics for the layer0 dispatch harness.

EPIC ("Balancing and Beyond: Communication-Centric Optimizations in Expert
Parallelism", Alibaba Cloud, SIGCOMM'26; flux/EPIC.pdf is ground truth — no
open-source release) is implemented here as a faithful, launch-granularity
baseline: §5.2 PEO (per-expert-group pipelining of dispatch -> grouped GEMM,
m in {1,2,4} groups), §4.2 EPIC-EPLB placement with replication (one-shot,
pool-oracle), and §4.3 dynamic intra-host expert migration. Layer0
(dispatch + expert GEMM) plus the --layers l01 full journey (GELU -> GEMM1
-> per-group combine -> terminal Sum), with the combine wire mirroring the
dispatch transport (v2).

Fidelity contract (decisions ledger in the 2026-08-16 plan):
  * PEO pipelines UNMODIFIED kernels at kernel-launch granularity — no flux
    GEMM-overlap machinery anywhere. Per group: comm-only a2av (the staged
    ultraep/eplb wire) then one un-overlapped flux.GemmGroupedV2 launch over
    that group's contiguous slot range.
  * Transports: hier_compress (the driver default) is EPIC's own Mode 2
    (§5.1 Fig 8(d), PXN relay + de-redundancy) — dispatch via the fused op's
    dispatch_only over the virtual slot space, l01 combine via per-group
    TopkReduceScatterOp. nvshmem is the Mode-1 (DeepEP-default) analog:
    per-entry wire, NO dedup — one row per (token, physical expert
    instance), matching both the paper's description of DeepEP's send
    behavior and the existing staged wire (ultraep_semantics: "There is NO
    wire dedup"). nccl is a debug/parity fallback, never a faithful arm.
  * Groups partition LOCAL PHYSICAL SLOTS [0, nlp) by index, identically on
    every rank, applied after placement and unchanged by migration
    (migration swaps slot CONTENTS). Under replication an expert's
    instances may fall in different groups on different ranks — accepted:
    the paper groups local experts by index, and with replication a GPU's
    local experts ARE its slots.
  * Placement input is the per-expert pool load vector `c` ONLY; the
    inter/intra-node split is ESTIMATED under an iid-source assumption
    (recorded); the true routing matrix is used post-hoc as a capsule
    diagnostic, never as planner input.
  * Replica selection (paper-silent, recorded assumption): a source rank
    sends ALL its tokens for expert l to instance `src mod lcnts[l]` —
    deterministic, one-shot-safe, even split in expectation. Expressed
    directly as a step-function rank_quota_prefix, so reroute_expand's
    conservation assert holds by construction and the coprime-stride
    interleave is a provable no-op (unit-tested).
  * Greedy tie-breaks throughout are lowest-index; the paper does not
    specify them (recorded assumption).

Everything downstream (reroute_expand, buffer layouts, canonical weight
generators, one-shot placement P2P) is reused from ultraep_semantics /
eplb_semantics so the epic and eplb arms stay bit-comparable.

Key layout theorem (what makes the m-sweep clean): the receiver-side
hidden/output buffer layout is IDENTICAL for every m. Groups are contiguous
slot ranges and hidden_buf is slot-major, so group g's rows are exactly the
contiguous span [seg_start[lo_g], seg_start[hi_g]) of the ungrouped layout,
with the same within-slot row order (group filtering preserves each slot's
(src, token) arrival order). Only the WIRE order differs (group-major).
Hence hidden_buf after all groups' scatters is bitwise equal to the
ungrouped scatter, and cross-m output identity is a hard invariant.
"""

import time
from dataclasses import dataclass, field

import torch

from .ultraep_semantics import (
    UltraEPConfig,
    UltraEPPlan,
    reroute_expand,
)
from .eplb_semantics import EPLBLayer0Runner

EPIC_GROUPS = (1, 2, 4)


# ---------------------------------------------------------------------------
# §4.2 EPIC-EPLB planner (redundancy -> GPU assign -> NIC/node positions)
# ---------------------------------------------------------------------------


def epic_redundancy_vector(pool_load: torch.Tensor, spare_slots: int,
                           replica_cap: int) -> torch.Tensor:
    """Greedy redundancy vector r minimizing max_i c_i/(1+r_i).

    Grants one replica slot at a time to the argmax of c_i/(1+r_i)
    (integer-safe cross-multiplied compare on float64 loads scaled to int;
    ties -> lowest expert id), capped at replica_cap extra copies per expert.
    Returns [G] int64 r (replica counts, EXcluding the master).
    """
    c = torch.as_tensor(pool_load, dtype=torch.float64).flatten()
    G = c.numel()
    assert bool((c >= 0).all()), "negative pool load"
    r = torch.zeros(G, dtype=torch.int64)
    # Work in exact integer space: scale float loads to integers only if
    # needed. Pool loads are token counts in practice; cast when lossless.
    ci = c.round().long() if bool((c == c.round()).all()) else None
    for _ in range(spare_slots):
        best = -1
        for i in range(G):
            if int(r[i]) >= replica_cap:
                continue
            if best < 0:
                best = i
                continue
            # c_i/(1+r_i) > c_best/(1+r_best)  <=>  cross-multiplied
            if ci is not None:
                lhs = int(ci[i]) * (1 + int(r[best]))
                rhs = int(ci[best]) * (1 + int(r[i]))
            else:
                lhs = float(c[i]) * (1 + int(r[best]))
                rhs = float(c[best]) * (1 + int(r[i]))
            if lhs > rhs:
                best = i
        if best < 0:
            break
        r[best] += 1
    return r


def epic_assign_gpus(pool_load: torch.Tensor, r: torch.Tensor, R: int,
                     nlp: int):
    """§4.2 stage 3: instances -> abstract GPUs, greedy by load.

    Instances (one per master+replica) sorted by descending effective load
    c_i/(1+r_i) (ties -> lower expert id, lower replica index); each goes to
    the least-loaded GPU that has a free slot and no instance of the same
    expert (ties -> lowest GPU id). Assignment order defines the LOCAL SLOT
    ORDER on each GPU (grouping-relevant; documented in the module
    docstring).

    Returns (gpu_lists, gpu_chat): gpu_lists[g] = ordered list of
    (logical, replica_idx) pairs; gpu_chat[g] = float sum of effective
    loads on g.
    """
    c = torch.as_tensor(pool_load, dtype=torch.float64).flatten()
    G = c.numel()
    instances = []
    for l in range(G):
        C = 1 + int(r[l])
        chat = float(c[l]) / C
        for j in range(C):
            instances.append((chat, l, j))
    assert len(instances) <= R * nlp, (
        f"{len(instances)} instances > {R * nlp} slots"
    )
    # Descending load; ties -> lower expert id, lower replica idx.
    instances.sort(key=lambda t: (-t[0], t[1], t[2]))

    gpu_lists = [[] for _ in range(R)]
    gpu_chat = [0.0] * R
    gpu_experts = [set() for _ in range(R)]
    for chat, l, j in instances:
        best = -1
        for g in range(R):
            if len(gpu_lists[g]) >= nlp or l in gpu_experts[g]:
                continue
            if best < 0 or gpu_chat[g] < gpu_chat[best]:
                best = g
        assert best >= 0, (
            f"no feasible GPU for expert {l} replica {j} "
            f"(cap: expert once per GPU; raise slots or lower redundancy)"
        )
        gpu_lists[best].append((l, j))
        gpu_chat[best] += chat
        gpu_experts[best].add(l)
    return gpu_lists, gpu_chat


def epic_node_positions(gpu_chat, D: int, num_nodes: int, K_pool: float):
    """§4.2 stage 4 (NIC stage, D5): abstract GPUs -> node positions.

    Per-GPU inter-node volume estimate under the iid-source assumption
    (pool units; ĉ = effective instance load, chat[g] = Σ ĉ on GPU g,
    total = Σ chat, R = D * num_nodes):
      recv(g) = chat[g] * (R - D) / R
      send(g) = (total / R) * (1 - node_chat(node(g)) / total)
    K_pool is accepted for interface completeness (the per-GPU emission in
    pool units is total/R regardless of K; K cancels because pool loads
    already count token-assignments).

    Greedy: GPUs in descending chat; each placed into the node with free
    capacity minimizing the resulting max over placed GPUs of send+recv
    (ties -> lowest node id). Position within a node = fill order.

    Returns (rank_of_gpu [R], est_send [R], est_recv [R]) with est_* indexed
    by ABSTRACT gpu id. On a single node (num_nodes == 1) the assignment is
    the identity-by-fill-order and all inter-node estimates are zero.
    """
    R = len(gpu_chat)
    assert R == D * num_nodes
    total = float(sum(gpu_chat))
    order = sorted(range(R), key=lambda g: (-gpu_chat[g], g))

    node_members = [[] for _ in range(num_nodes)]

    def objective(members):
        # max over placed GPUs of send+recv under the candidate membership.
        worst = 0.0
        for n, mem in enumerate(members):
            node_chat = sum(gpu_chat[g] for g in mem)
            for g in mem:
                recv = gpu_chat[g] * (R - D) / R if R > D else 0.0
                send = (
                    (total / R) * (1.0 - node_chat / total)
                    if total > 0 and R > D else 0.0
                )
                worst = max(worst, send + recv)
        return worst

    for g in order:
        best_n, best_obj = -1, None
        for n in range(num_nodes):
            if len(node_members[n]) >= D:
                continue
            node_members[n].append(g)
            obj = objective(node_members)
            node_members[n].pop()
            if best_n < 0 or obj < best_obj:
                best_n, best_obj = n, obj
        assert best_n >= 0
        node_members[best_n].append(g)

    rank_of_gpu = [0] * R
    for n in range(num_nodes):
        for pos, g in enumerate(node_members[n]):
            rank_of_gpu[g] = n * D + pos

    est_send = [0.0] * R
    est_recv = [0.0] * R
    if R > D and total > 0:
        for n in range(num_nodes):
            node_chat = sum(gpu_chat[g] for g in node_members[n])
            for g in node_members[n]:
                est_recv[g] = gpu_chat[g] * (R - D) / R
                est_send[g] = (total / R) * (1.0 - node_chat / total)
    return rank_of_gpu, est_send, est_recv


def epic_rank_quota_prefix(cfg: UltraEPConfig, tpe: torch.Tensor,
                           lcnts: torch.Tensor) -> torch.Tensor:
    """[R, G, max_replicas_dim] int32 D6 prefix: source src sends ALL its
    tokens for expert l to instance j* = src mod lcnts[l].

    Step function — 0 below j*, tpe[src, l] at and above — so
    prefix[C-1] == tpe[src, l] (reroute conservation) by construction, and
    searchsorted(prefix, q, right=True) == j* for every q in [0, load):
    the interleave permutation cannot change the chosen replica.
    """
    R, G, M = cfg.R, cfg.G, cfg.max_replicas_dim
    tpe_l = tpe.long()
    out = torch.zeros(R, G, M, dtype=torch.int32)
    for src in range(R):
        for l in range(G):
            C = int(lcnts[l])
            j_star = src % C
            out[src, l, j_star:C] = int(tpe_l[src, l])
    return out


def build_epic_plan(cfg: UltraEPConfig, tpe: torch.Tensor, pool_load,
                    num_nodes: int) -> UltraEPPlan:
    """EPIC §4.2 placement mapped onto the UltraEPPlan tensor layout.

    Same contract as build_eplb_plan: placement (p2l/l2p/lcnts) from the
    POOL load; quotas and rank_quota_prefix from the BATCH tpe (D6 rule);
    deterministic pure-integer/float64 host math, so every rank computes the
    identical plan (guarded by the driver's plan_hash all-gather).

    cfg is MUTATED: max_replicas_dim set (<= R: an expert appears at most
    once per GPU) BEFORE any tensor allocation. locality_aware quota paths
    are never used (D6 prefix is built directly).
    """
    assert tuple(tpe.shape) == (cfg.R, cfg.G)
    assert cfg.R % num_nodes == 0, (num_nodes, cfg.R)
    D = cfg.R // num_nodes
    pool = torch.as_tensor(pool_load, dtype=torch.float64).flatten()
    assert pool.numel() == cfg.G

    # Replica cap: one instance per GPU max => at most R instances.
    cfg.max_replicas_dim = cfg.R

    spare = cfg.R * cfg.R_red
    r = epic_redundancy_vector(pool, spare, replica_cap=cfg.R - 1)
    assert int(r.sum()) <= spare
    gpu_lists, gpu_chat = epic_assign_gpus(pool, r, cfg.R, cfg.nlp)
    rank_of_gpu, est_send, est_recv = epic_node_positions(
        gpu_chat, D, num_nodes, K_pool=float(cfg.K)
    )

    p2l = torch.full((cfg.P,), -1, dtype=torch.int32)
    lcnts = (1 + r).to(torch.int32)
    l2p = torch.full((cfg.G, cfg.max_replicas_dim), -1, dtype=torch.int32)
    # Physical slots in assignment order on each PHYSICAL rank; l2p columns
    # ordered by ascending physical slot id (deterministic; no master
    # semantics — the eplb PINNED_MASTERS=False family).
    slots_of_logical = [[] for _ in range(cfg.G)]
    est_send_by_rank = [0.0] * cfg.R
    est_recv_by_rank = [0.0] * cfg.R
    for g, lst in enumerate(gpu_lists):
        rank = rank_of_gpu[g]
        est_send_by_rank[rank] = est_send[g]
        est_recv_by_rank[rank] = est_recv[g]
        for slot_idx, (l, _j) in enumerate(lst):
            phys = rank * cfg.nlp + slot_idx
            p2l[phys] = l
            slots_of_logical[l].append(phys)
    for l in range(cfg.G):
        slots = sorted(slots_of_logical[l])
        assert len(slots) == int(lcnts[l]), (l, slots, int(lcnts[l]))
        for j, phys in enumerate(slots):
            l2p[l, j] = phys

    # Quota (informational parity with the eplb arm; the reroute table is
    # the D6 prefix, not this): equal split of the BATCH load.
    loads_g = tpe.long().sum(dim=0)
    quota = torch.zeros(cfg.G, cfg.max_replicas_dim, dtype=torch.int32)
    quota_prefix = torch.zeros(cfg.G, cfg.max_replicas_dim, dtype=torch.int32)
    for l in range(cfg.G):
        C = int(lcnts[l])
        base, rem = divmod(int(loads_g[l]), C)
        prefix = 0
        for j in range(C):
            q = base + (1 if j < rem else 0)
            quota[l, j] = q
            prefix += q
            quota_prefix[l, j] = prefix

    plan = UltraEPPlan(
        cfg=cfg, tpe=tpe.to(torch.int32), p2l=p2l, l2p=l2p, lcnts=lcnts,
        quota=quota, quota_prefix=quota_prefix,
        rank_quota_prefix=epic_rank_quota_prefix(cfg, tpe, lcnts),
        domain_solutions=[],
    )
    # D4/D5 diagnostic facts ride on the plan object (pool units).
    plan.epic_est_internode_send = est_send_by_rank
    plan.epic_est_internode_recv = est_recv_by_rank
    plan.epic_redundancy = r.tolist()
    return plan


def build_fixed_plan(cfg: UltraEPConfig, tpe: torch.Tensor) -> UltraEPPlan:
    """--placement none control: contiguous homing (expert l on rank
    l // epn, local slot l % epn), redundant slots unused (p2l = -1),
    lcnts = 1, D6 prefix (trivially single-column = tpe)."""
    assert tuple(tpe.shape) == (cfg.R, cfg.G)
    cfg.max_replicas_dim = cfg.R
    p2l = torch.full((cfg.P,), -1, dtype=torch.int32)
    l2p = torch.full((cfg.G, cfg.max_replicas_dim), -1, dtype=torch.int32)
    lcnts = torch.ones(cfg.G, dtype=torch.int32)
    for l in range(cfg.G):
        phys = (l // cfg.epn) * cfg.nlp + (l % cfg.epn)
        p2l[phys] = l
        l2p[l, 0] = phys
    loads_g = tpe.long().sum(dim=0)
    quota = torch.zeros(cfg.G, cfg.max_replicas_dim, dtype=torch.int32)
    quota_prefix = torch.zeros(cfg.G, cfg.max_replicas_dim, dtype=torch.int32)
    quota[:, 0] = loads_g.to(torch.int32)
    quota_prefix[:, 0] = loads_g.to(torch.int32)
    plan = UltraEPPlan(
        cfg=cfg, tpe=tpe.to(torch.int32), p2l=p2l, l2p=l2p, lcnts=lcnts,
        quota=quota, quota_prefix=quota_prefix,
        rank_quota_prefix=epic_rank_quota_prefix(cfg, tpe, lcnts),
        domain_solutions=[],
    )
    plan.epic_est_internode_send = [0.0] * cfg.R
    plan.epic_est_internode_recv = [0.0] * cfg.R
    plan.epic_redundancy = [0] * cfg.G
    return plan


# ---------------------------------------------------------------------------
# §5.2 PEO grouping + per-group comm layout
# ---------------------------------------------------------------------------


def group_partition(nlp: int, m: int):
    """m contiguous, near-equal partitions of local slots [0, nlp).

    Sizes nlp//m (+1 for the first nlp % m groups), identical on every rank.
    """
    assert 1 <= m <= nlp, (m, nlp)
    bounds = []
    lo = 0
    for g in range(m):
        size = nlp // m + (1 if g < nlp % m else 0)
        bounds.append((lo, lo + size))
        lo += size
    assert lo == nlp
    return bounds


@dataclass
class EpicGroupLayout:
    """One group's slice of the wire for one rank (host metadata)."""

    g: int
    slot_lo: int
    slot_hi: int
    # sender: entries destined to this group's slots on ANY rank,
    # (dest, slot, token) order — per-entry wire, no dedup.
    send_row_index: torch.Tensor      # [n_send_g] int64 local token row
    send_entry_logical: torch.Tensor  # [n_send_g] int64 logical expert
    send_counts: list                 # [W] rows to each dest
    # receiver: rows arrive src-major, (slot, token) order within src.
    recv_counts: list                 # [W] rows from each src
    place_slots: torch.Tensor         # [n_recv_g] int64 ABSOLUTE hidden_buf slot
    seg_rows: list                    # [slot_hi - slot_lo] rows per slot
    # combine (layer1): home staging slot t*K + j per SEND entry. By the
    # transposition theorem the combine recv stream on the home rank is,
    # position for position, this rank's dispatch send window (same order),
    # so this single tensor is the entire combine-side wire metadata.
    # j = position of the entry's logical expert in topk_all[rank, t] —
    # unique because reroute_expand hard-asserts distinct experts per token.
    comb_dst_slot: torch.Tensor = None  # [n_send_g] int64 in [0, S*K)


@dataclass
class EpicCommLayout:
    """All groups' wire metadata for one rank, plus shared-buffer offsets.

    The receiver hidden/output layout is the UNGROUPED slot-major layout
    (see module docstring theorem); only the wire order is group-major.
    """

    m: int
    groups: list                      # [m] EpicGroupLayout
    send_off: list                    # [m+1] row offsets into send_buf
    recv_off: list                    # [m+1] row offsets into recv_buf
    seg_start: list                   # [nlp] hidden_buf segment starts
    seg_rows: list                    # [nlp] rows per local slot
    gemm_segments: list               # [(slot, start, end, logical)] nonzero
    n_send: int
    n_recv: int
    # GLOBAL max over (src, group, dst) pair rows — identical on every rank
    # by construction (derived from the replicated plan + routing). MUST be
    # used for All2AllSingle max_split: the op computes remote staging
    # offsets with the SENDER's max_split while the receiver lays out its
    # staging with its OWN — per-rank-divergent values silently corrupt the
    # wire (found in bring-up 2026-08-16).
    max_pair_rows: int
    internode_send_rows: int          # this rank's rows to other nodes
    internode_recv_rows: int          # this rank's rows from other nodes


def build_epic_group_layouts(plan: UltraEPPlan, rank: int,
                             topk_all: torch.Tensor, m: int,
                             ranks_per_node: int = 0) -> EpicCommLayout:
    """Group-major generalization of build_comm_layout's (phys, token) sort:
    entries ordered by (group_of(slot), phys, token). Runs reroute_expand
    once per source (conservation assert reused)."""
    cfg = plan.cfg
    R, nlp = cfg.R, cfg.nlp
    bounds = group_partition(nlp, m)
    group_of_slot = torch.empty(nlp, dtype=torch.int64)
    for g, (lo, hi) in enumerate(bounds):
        group_of_slot[lo:hi] = g

    # Per-source expanded entries, canonical (phys, token) order.
    ent_tok, ent_phys = [], []
    for src in range(R):
        t, p = reroute_expand(cfg, plan, src, topk_all[src])
        order = torch.argsort(p * (cfg.S + 1) + t, stable=True)
        ent_tok.append(t[order])
        ent_phys.append(p[order])

    # Ungrouped receiver segment layout (shared across all m — the theorem).
    phys_base = rank * nlp
    recv_local_by_src = []
    for src in range(R):
        msk = (ent_phys[src] // nlp) == rank
        recv_local_by_src.append(ent_phys[src][msk] - phys_base)
    all_local = (
        torch.cat(recv_local_by_src) if recv_local_by_src
        else torch.zeros(0, dtype=torch.int64)
    )
    seg_rows = torch.bincount(all_local, minlength=nlp)
    seg_start = torch.zeros(nlp, dtype=torch.int64)
    seg_start[1:] = torch.cumsum(seg_rows, dim=0)[:-1]

    p2l_l = plan.p2l.long()
    gemm_segments = []
    for p in range(nlp):
        rows = int(seg_rows[p])
        if rows == 0:
            continue
        logical = int(p2l_l[phys_base + p])
        assert logical >= 0, f"rank {rank}: rows in unused slot {p}"
        start = int(seg_start[p])
        gemm_segments.append((p, start, start + rows, logical))

    groups = []
    send_off, recv_off = [0], [0]
    inter_send = inter_recv = 0
    my_node = rank // ranks_per_node if ranks_per_node else 0

    # Combine home-slot table pos[t, l] = j (position of logical l in this
    # rank's topk row for token t; unique — reroute_expand asserts distinct
    # experts per token).
    my_topk = topk_all[rank].cpu().long()                       # [S, K]
    pos = torch.full((cfg.S, cfg.G), -1, dtype=torch.int64)
    pos[torch.arange(cfg.S).unsqueeze(1), my_topk] = (
        torch.arange(cfg.K, dtype=torch.int64).expand(cfg.S, cfg.K)
    )

    # Replicated global pair-rows max (see EpicCommLayout.max_pair_rows):
    # every (src, group, dest) pair over ALL sources, not just this rank's.
    max_pair_rows = 0
    for src in range(R):
        src_grp = group_of_slot[ent_phys[src] % nlp]
        src_dest = ent_phys[src] // nlp
        pair_key = src_grp * R + src_dest
        counts = torch.bincount(pair_key, minlength=m * R)
        max_pair_rows = max(max_pair_rows, int(counts.max()))
    # Per-slot occurrence counters persist ACROSS groups per src? No —
    # a slot belongs to exactly one group, so per-group counting is exact.
    for g, (lo, hi) in enumerate(bounds):
        # -- sender ---------------------------------------------------------
        my_msk = (group_of_slot[ent_phys[rank] % nlp] == g)
        tok_g = ent_tok[rank][my_msk]
        phys_g = ent_phys[rank][my_msk]
        # already (phys, token)-sorted => (dest, slot, token) within group.
        dest_g = phys_g // nlp
        send_counts = torch.bincount(dest_g, minlength=R)
        send_entry_logical = p2l_l[phys_g]

        # -- receiver -------------------------------------------------------
        recv_counts = []
        place_chunks = []
        for src in range(R):
            loc = recv_local_by_src[src]
            in_g = loc[(group_of_slot[loc] == g)]
            recv_counts.append(int(in_g.numel()))
            place_chunks.append(in_g)
        grp_local = (
            torch.cat(place_chunks) if place_chunks
            else torch.zeros(0, dtype=torch.int64)
        )
        # slot of row i = seg_start[slot] + occurrence index in the group's
        # (src, entry) arrival order — identical to the ungrouped
        # within-slot order (group filtering preserves it).
        place_slots = torch.empty_like(grp_local)
        if grp_local.numel():
            order = torch.argsort(grp_local, stable=True)
            positions = torch.arange(grp_local.numel(), dtype=torch.int64)
            sorted_slots = grp_local[order]
            new_run = torch.cat([
                torch.ones(1, dtype=torch.bool),
                sorted_slots[1:] != sorted_slots[:-1],
            ])
            run_start = torch.where(
                new_run, positions, torch.full_like(positions, -1))
            run_start = torch.cummax(run_start, dim=0).values
            occ_sorted = positions - run_start
            place_slots[order] = seg_start[sorted_slots] + occ_sorted

        if ranks_per_node:
            for d in range(R):
                if d // ranks_per_node != my_node:
                    inter_send += int(send_counts[d])
            for s in range(R):
                if s // ranks_per_node != my_node:
                    inter_recv += recv_counts[s]

        j_g = pos[tok_g, send_entry_logical]
        assert bool((j_g >= 0).all()), "send entry expert not in token's topk"
        groups.append(EpicGroupLayout(
            g=g, slot_lo=lo, slot_hi=hi,
            send_row_index=tok_g,
            send_entry_logical=send_entry_logical,
            send_counts=send_counts.tolist(),
            recv_counts=recv_counts,
            place_slots=place_slots,
            seg_rows=seg_rows[lo:hi].tolist(),
            comb_dst_slot=tok_g * cfg.K + j_g,
        ))
        send_off.append(send_off[-1] + int(tok_g.numel()))
        recv_off.append(recv_off[-1] + sum(recv_counts))

    # Every (token, topk-slot) staging cell is owned by exactly one send
    # entry across all groups — the combine-staging correctness invariant.
    all_comb = torch.cat([grp.comb_dst_slot for grp in groups])
    assert bool(
        (torch.bincount(all_comb, minlength=cfg.S * cfg.K) == 1).all()
    ), "comb_dst_slot is not a permutation of [0, S*K)"

    return EpicCommLayout(
        m=m, groups=groups, send_off=send_off, recv_off=recv_off,
        seg_start=seg_start.tolist(), seg_rows=seg_rows.tolist(),
        gemm_segments=gemm_segments,
        n_send=send_off[-1], n_recv=recv_off[-1],
        max_pair_rows=max_pair_rows,
        internode_send_rows=inter_send,
        internode_recv_rows=inter_recv,
    )


def epic_dup_stats(lay: EpicCommLayout, R: int):
    """Wire-dedup counterfactual facts (capsule diagnostics, D3):
    dup_vs_nodedup   — rows a per-(token, dest rank) dedup ACROSS the whole
                       message would have saved (upper bound, Mode-2-like);
    dup_cross_group  — of those, the rows whose duplicates span different
                       groups (unrecoverable under within-group dedup)."""
    pairs_all = []
    per_group_savings = 0
    for grp in lay.groups:
        tok = grp.send_row_index
        if tok.numel() == 0:
            continue
        dest = torch.cat([
            torch.full((c,), d, dtype=torch.int64)
            for d, c in enumerate(grp.send_counts)
        ])
        pair = tok * R + dest
        per_group_savings += int(pair.numel() - torch.unique(pair).numel())
        pairs_all.append(pair)
    if not pairs_all:
        return {"dup_vs_nodedup": 0, "dup_cross_group": 0,
                "dup_within_group": 0}
    allp = torch.cat(pairs_all)
    total_savings = int(allp.numel() - torch.unique(allp).numel())
    return {
        "dup_vs_nodedup": total_savings,
        "dup_within_group": per_group_savings,
        "dup_cross_group": total_savings - per_group_savings,
    }


# ---------------------------------------------------------------------------
# hier_compress transport (EPIC Mode 2): per-group virtual bundles for the
# fused op's dispatch_only entry. Virtual expert space = physical slots + ONE
# PAD slot per rank (gpe = nlp + 1, pad local index = nlp, LAST in the rank
# block so pad GEMM rows are a contiguous dense-tail). Per-group fixed
# topk = K_g = max in-group entries of any token; pad entries self-route to
# the home rank's pad slot (self-copies, deduped by compress like any other
# same-rank duplicate). Metadata recipe/preflight/knob math reuse
# moonep_fused_map verbatim (FusedMeta is just the field bundle).
# ---------------------------------------------------------------------------


@dataclass
class EpicHcBundle:
    """One group's replicated host metadata for the hier_compress arm."""

    g: int
    K_g: int
    gpe: int                        # nlp + 1
    E_virt: int                     # R * gpe
    virtual_choosed: torch.Tensor   # [ntokens, K_g] int32 CPU
    meta: "object"                  # moonep_fused_map.FusedMeta
    pad_rows_per_rank: torch.Tensor  # [W] int64


def build_epic_hc_bundles(plan: UltraEPPlan, topk_all: torch.Tensor, m: int,
                          local_world_size: int, fixed_kg=None):
    """Per-group virtual routing bundles (replicated pure function of the
    plan + routing). Entry -> vslot mapping comes from the same
    reroute_expand expansion the direct layouts use, so bundles are
    consistent with EpicCommLayout by construction."""
    from .moonep_fused_map import (
        FusedMeta,
        _stable_scatter_index,
        preflight_metadata_checks,
        required_a2av_knobs,
    )

    cfg = plan.cfg
    R, S, nlp = cfg.R, cfg.S, cfg.nlp
    gpe = nlp + 1
    E_virt = R * gpe
    ntokens = R * S
    L = local_world_size
    bounds = group_partition(nlp, m)
    group_of_slot = torch.empty(nlp, dtype=torch.int64)
    for g, (lo, hi) in enumerate(bounds):
        group_of_slot[lo:hi] = g

    ent = []
    for src in range(R):
        t, p = reroute_expand(cfg, plan, src, topk_all[src])
        order = torch.argsort(p * (S + 1) + t, stable=True)
        ent.append((t[order], p[order]))

    home_of_token = torch.arange(ntokens, dtype=torch.int64) // S
    pad_vslot = home_of_token * gpe + nlp          # [ntokens]

    bundles = []
    for g in range(m):
        gts, vsl = [], []
        for src in range(R):
            t_all, p_all = ent[src]
            msk = group_of_slot[p_all % nlp] == g
            gts.append(src * S + t_all[msk])
            vsl.append((p_all[msk] // nlp) * gpe + (p_all[msk] % nlp))
        gts = torch.cat(gts)
        vsl = torch.cat(vsl)
        order = torch.argsort(gts, stable=True)
        gts_s, vsl_s = gts[order], vsl[order]
        # per-token occurrence index (column in virtual_choosed)
        positions = torch.arange(gts_s.numel(), dtype=torch.int64)
        if gts_s.numel():
            new_run = torch.cat([
                torch.ones(1, dtype=torch.bool), gts_s[1:] != gts_s[:-1]])
            run_start = torch.cummax(
                torch.where(new_run, positions,
                            torch.full_like(positions, -1)), 0).values
            occ = positions - run_start
            K_g = int(occ.max()) + 1
        else:
            occ = positions
            K_g = 1
        if fixed_kg is not None:
            # op instances freeze topk at ctor: migration may not widen a
            # group's per-token entry count past the frozen width
            assert K_g <= fixed_kg[g], (
                f"group {g}: post-migration K_g {K_g} exceeds the frozen op "
                f"topk {fixed_kg[g]} — hc arms cannot absorb this migration"
            )
            K_g = fixed_kg[g]
        vce = torch.full((ntokens, K_g), -1, dtype=torch.int64)
        vce[gts_s, occ] = vsl_s
        pad_mask = vce < 0
        pad_rows = torch.zeros(R, dtype=torch.int64).index_add_(
            0, home_of_token, pad_mask.sum(1))
        vce = torch.where(pad_mask, pad_vslot.unsqueeze(1).expand_as(vce), vce)
        vce = vce.int()

        # metadata recipe: build_fused_metadata with (S, K_g, gpe, E_virt)
        vce_l = vce.long()
        scatter_index = _stable_scatter_index(vce)
        splits = torch.bincount(vce_l.flatten(), minlength=E_virt).int()
        src_of_copy = home_of_token.repeat_interleave(K_g)
        splits_per_source = (
            torch.bincount(src_of_copy * E_virt + vce_l.flatten(),
                           minlength=R * E_virt)
            .view(R, E_virt).int().contiguous()
        )
        owner = vce_l // gpe
        flags = torch.zeros(ntokens, R, dtype=torch.bool)
        flags.scatter_(1, owner, True)
        u_mat = flags.view(R, S, R).sum(1)
        nn = R // L
        U_mat = flags.view(ntokens, nn, L).any(dim=2).view(R, S, nn).sum(1)
        a2av_unique_counts = torch.cat([u_mat, U_mat], dim=1).int().contiguous()
        m_per_rank = splits.long().view(R, gpe).sum(1)
        meta = FusedMeta(
            scatter_index=scatter_index,
            splits=splits,
            splits_per_source=splits_per_source,
            a2av_unique_counts=a2av_unique_counts,
            m_per_rank=m_per_rank,
        )
        preflight_metadata_checks(meta, R, L)
        bundles.append(EpicHcBundle(
            g=g, K_g=K_g, gpe=gpe, E_virt=E_virt,
            virtual_choosed=vce, meta=meta,
            pad_rows_per_rank=pad_rows,
        ))
    return bundles


def epic_hc_required_knobs(bundle: EpicHcBundle, W: int,
                           local_world_size: int) -> dict:
    """Exact FLUX_A2AV_MAX_{RECV,STAGE,RELAY}_NTOKENS for one group's op
    instance (moonep_fused_map.required_a2av_knobs on the group's meta)."""
    from .moonep_fused_map import required_a2av_knobs

    return required_a2av_knobs(bundle.meta, W, local_world_size)


# ---------------------------------------------------------------------------
# §4.3 dynamic intra-host expert migration
# ---------------------------------------------------------------------------


def slot_batch_loads(plan: UltraEPPlan) -> torch.Tensor:
    """[P] int64 batch tokens per physical slot under the D6 routing
    (host-independent: derived from rank_quota_prefix diffs, which never
    read instance hosts — migration does NOT change this tensor's values
    per logical expert/instance, only which slot holds them)."""
    cfg = plan.cfg
    rqp = plan.rank_quota_prefix.long()
    alloc = rqp.clone()
    alloc[:, :, 1:] -= rqp[:, :, :-1]
    inst_load = alloc.sum(dim=0)                      # [G, max_replicas_dim]
    out = torch.zeros(cfg.P, dtype=torch.int64)
    l2p = plan.l2p.long()
    for l in range(cfg.G):
        for j in range(int(plan.lcnts[l])):
            out[int(l2p[l, j])] = int(inst_load[l, j])
    return out


def gpu_batch_loads(plan: UltraEPPlan) -> torch.Tensor:
    """[R] int64 batch tokens per rank (sum of its slots)."""
    cfg = plan.cfg
    return slot_batch_loads(plan).reshape(cfg.R, cfg.nlp).sum(dim=1)


def plan_migration_swaps(plan: UltraEPPlan, tau_tokens: float,
                         ranks_per_node: int):
    """§4.3 greedy swap plan (D8), host-side, replicated.

    Per node: sort its D ranks by load, pair sorted extremes ((0, D-1),
    (1, D-2), ...), at most ONE swap per pair; the swap (slot a on the
    heavy rank <-> slot b on the light rank) maximizing the pair's
    max-load reduction, accepted iff gain > tau_tokens. Swaps that would
    co-locate two instances of one expert on a rank are filtered. Ties ->
    lowest (a, b). Returns list of (rank_h, slot_a, rank_l, slot_b, gain),
    globally ordered (node-major, pair order) so every rank derives the
    identical list.
    """
    cfg = plan.cfg
    D = ranks_per_node
    assert cfg.R % D == 0
    sl = slot_batch_loads(plan)
    gl = sl.reshape(cfg.R, cfg.nlp).sum(dim=1)
    p2l = plan.p2l.long()

    swaps = []
    for node in range(cfg.R // D):
        ranks = list(range(node * D, (node + 1) * D))
        order = sorted(ranks, key=lambda rk: (-int(gl[rk]), rk))
        for i in range(D // 2):
            rh, rl = order[i], order[D - 1 - i]
            Lh, Ll = int(gl[rh]), int(gl[rl])
            if Lh <= Ll:
                continue
            experts_h = set(
                int(p2l[rh * cfg.nlp + s]) for s in range(cfg.nlp)
                if int(p2l[rh * cfg.nlp + s]) >= 0
            )
            experts_l = set(
                int(p2l[rl * cfg.nlp + s]) for s in range(cfg.nlp)
                if int(p2l[rl * cfg.nlp + s]) >= 0
            )
            best = None
            for a in range(cfg.nlp):
                pa = rh * cfg.nlp + a
                la = int(p2l[pa])
                wa = int(sl[pa])
                for b in range(cfg.nlp):
                    pb = rl * cfg.nlp + b
                    lb = int(p2l[pb])
                    wb = int(sl[pb])
                    if la == lb:
                        continue
                    # Expert-duplication filter (a rank may hold an expert
                    # at most once; -1 empty slots always legal).
                    if la >= 0 and la in experts_l and lb != la:
                        continue
                    if lb >= 0 and lb in experts_h and la != lb:
                        continue
                    new_h = Lh - wa + wb
                    new_l = Ll - wb + wa
                    gain = max(Lh, Ll) - max(new_h, new_l)
                    if gain <= 0:
                        continue
                    if best is None or gain > best[0]:
                        best = (gain, a, b)
            if best is not None and best[0] > tau_tokens:
                swaps.append((rh, best[1], rl, best[2], int(best[0])))
    return swaps


def apply_swaps(plan: UltraEPPlan, swaps) -> dict:
    """Mutate the plan in place: swap p2l entries and patch the (<= 2)
    l2p entries per swap. quota/quota_prefix/rank_quota_prefix/lcnts are
    untouched (host-independent under locality_aware=False / D6).

    Returns an invariant report dict; raises on violation.
    """
    cfg = plan.cfg
    p2l = plan.p2l
    l2p = plan.l2p

    def patch_l2p(l, old_phys, new_phys):
        if l < 0:
            return
        C = int(plan.lcnts[l])
        row = l2p[l, :C]
        hits = (row == old_phys).nonzero(as_tuple=True)[0]
        assert hits.numel() == 1, (l, old_phys, row.tolist())
        row[hits[0]] = new_phys

    for rh, a, rl, b, _gain in swaps:
        pa = rh * cfg.nlp + a
        pb = rl * cfg.nlp + b
        la, lb = int(p2l[pa]), int(p2l[pb])
        patch_l2p(la, pa, pb)
        patch_l2p(lb, pb, pa)
        p2l[pa], p2l[pb] = lb, la

    # Invariants over the touched state (cheap full checks).
    for l in range(cfg.G):
        C = int(plan.lcnts[l])
        hosts = []
        for j in range(C):
            phys = int(l2p[l, j])
            assert phys >= 0 and int(p2l[phys]) == l, (l, j, phys)
            hosts.append(phys // cfg.nlp)
        assert len(set(hosts)) == C, f"expert {l} duplicated on a rank"
    return {"applied": len(swaps)}


def swap_weight_ops(swaps, rank: int, slot_fc1, slot_fc2, dist, group):
    """Batched P2P op list realizing the swaps' weight exchanges for THIS
    rank (EPLB place_weights discipline: all ranks walk the identical
    globally-ordered swap list; both directions staged through clones so a
    slot can send and receive in one batch). Returns (ops, keepalive,
    recv_bytes)."""
    ops, keepalive, recv_bytes = [], [], 0
    P2POp = dist.P2POp
    for rh, a, rl, b, _gain in swaps:
        for (my_r, my_slot, peer_r) in ((rh, a, rl), (rl, b, rh)):
            if my_r != rank:
                continue
            outgoing1 = slot_fc1[my_slot].clone().contiguous()
            keepalive.append(outgoing1)
            ops.append(P2POp(dist.isend, outgoing1, peer=peer_r, group=group))
            ops.append(P2POp(dist.irecv, slot_fc1[my_slot], peer=peer_r,
                             group=group))
            recv_bytes += slot_fc1[my_slot].numel() * slot_fc1.element_size()
            if slot_fc2 is not None:
                outgoing2 = slot_fc2[my_slot].clone().contiguous()
                keepalive.append(outgoing2)
                ops.append(P2POp(dist.isend, outgoing2, peer=peer_r,
                                 group=group))
                ops.append(P2POp(dist.irecv, slot_fc2[my_slot], peer=peer_r,
                                 group=group))
                recv_bytes += (slot_fc2[my_slot].numel()
                               * slot_fc2.element_size())
    return ops, keepalive, recv_bytes


# ---------------------------------------------------------------------------
# Runner: EPLB data plane + PEO group pipeline
# ---------------------------------------------------------------------------


class EpicLayer0Runner(EPLBLayer0Runner):
    """EPLB's slot-indexed data plane driven per PEO group.

    Inherits: slot_fc1/slot_fc2 (+canonical generators), place_weights()
    one-shot placement, out_buf, and all parent buffers — the parent ctor's
    ungrouped layout has the same n_send/n_recv totals and the IDENTICAL
    hidden_buf layout (module docstring theorem), so buffers are reused and
    only the wire indices are replaced by the group-major ones.

    Phase methods for the driver's two-stream pipeline: pack() (group-major),
    dispatch_group(g), scatter_group(g), gemm_group(g). GEMM backends:
    'grouped' = one flux.GemmGroupedV2 per group over slot_fc1[lo:hi]
    (launch-granularity faithful); 'gemmonly' = per-segment loop (debug /
    eplb-anchor parity).
    """

    def __init__(self, plan: UltraEPPlan, rank: int, group, device,
                 topk_all: torch.Tensor, m: int, dtype=torch.bfloat16,
                 ffn_size_shard: int = 0, place_fc2: bool = True,
                 ranks_per_node: int = 4, comm_group=None,
                 layers: str = "l0"):
        super().__init__(plan, rank, group, device, topk_all, dtype=dtype,
                         ffn_size_shard=ffn_size_shard, place_fc2=place_fc2)
        self.m = m
        # NCCL a2av calls run on the driver's comm stream; a DEDICATED
        # communicator avoids interleaving one communicator across two
        # streams (moonep_overlap precedent). Weight P2P and plan_comm stay
        # on `group`.
        self.comm_group = comm_group if comm_group is not None else group
        self.ranks_per_node = ranks_per_node
        assert layers in ("l0", "l01")
        if layers == "l01":
            assert place_fc2, "--layers l01 requires fc1fc2 weight placement"
        self.layers = layers
        self._gemm_backend = "grouped"
        self._grouped_ops = None
        self._grouped_ops_fc2 = None
        self.hc_enabled = False
        self.hcc_enabled = False
        self._last_inputs = None
        self.group_outputs = [None] * m
        self.group_act = [None] * m
        self.group1_outputs = [None] * m
        self.migration_swap_bytes = 0
        self._build_group_state(topk_all)
        if layers == "l01":
            H = self.cfg.H
            dev = self.device
            # expert-side packed combine rows (recv-capacity-sized, grows
            # with migration); home-side combine recv [S*K, H]; staging for
            # the terminal Sum; final [S, H] output.
            self.comb_send_buf = torch.empty(
                self.recv_buf.shape[0], H, dtype=dtype, device=dev)
            self.comb_recv_buf = torch.empty(
                self.n_send, H, dtype=dtype, device=dev)
            self.stage_buf = torch.zeros(
                self.n_send, H, dtype=dtype, device=dev)
            self.final_out = torch.zeros(self.cfg.S, H, dtype=dtype,
                                         device=dev)

    # -- layout (initial + post-migration rebuild) --------------------------

    def _build_group_state(self, topk_all: torch.Tensor):
        elay = build_epic_group_layouts(
            self.plan, self.rank, topk_all, self.m,
            ranks_per_node=self.ranks_per_node,
        )
        # n_send is invariant (S*K entries, per-entry wire); n_recv CHANGES
        # when migration moves instances between ranks — grow the recv-side
        # buffers when the new layout exceeds capacity (swap iterations
        # only; the driver syncs before applying a migration).
        assert elay.n_send == self.n_send, "send total must be S*K-invariant"
        if elay.n_recv > self.recv_buf.shape[0]:
            cap = int(elay.n_recv * 1.5)
            dev, dt = self.device, self.dtype
            H = self.cfg.H
            self.recv_buf = torch.empty(cap, H, dtype=dt, device=dev)
            self.wrecv_buf = torch.empty(cap, dtype=torch.float32, device=dev)
            self.hidden_buf = torch.zeros(cap, H, dtype=dt, device=dev)
            self.weights_buf = torch.zeros(cap, dtype=torch.float32,
                                           device=dev)
            self.out_buf = torch.zeros(cap, self.ffn_size_shard, dtype=dt,
                                       device=dev)
        self.n_recv = elay.n_recv
        self.elay = elay
        dev = self.device
        self.g_send_row_index = [
            grp.send_row_index.to(dev) for grp in elay.groups
        ]
        self.g_send_entry_logical = [
            grp.send_entry_logical.to(dev) for grp in elay.groups
        ]
        self.g_place_slots = [grp.place_slots.to(dev) for grp in elay.groups]
        if self.layers == "l01":
            self.g_comb_dst = [grp.comb_dst_slot.to(dev) for grp in elay.groups]
            self.g_comb_pack = [
                (grp.place_slots - elay.seg_start[grp.slot_lo]).to(dev)
                for grp in elay.groups
            ]
            if (hasattr(self, "comb_send_buf")
                    and elay.n_recv > self.comb_send_buf.shape[0]):
                self.comb_send_buf = torch.empty(
                    self.recv_buf.shape[0], self.cfg.H, dtype=self.dtype,
                    device=dev)
        if self.transport == "nvshmem":
            self._refresh_nvshmem_splits()
            assert elay.max_pair_rows <= self._epic_max_split, (
                f"post-migration pair rows {elay.max_pair_rows} exceed "
                f"All2AllSingle max_split {self._epic_max_split}; raise "
                f"--a2a_split_headroom"
            )
        # Grouped-GEMM splits (CPU int64, per group; zero-row slots legal).
        self._group_splits_cpu = [
            torch.tensor(grp.seg_rows, dtype=torch.int64)
            for grp in elay.groups
        ]
        self._topk_all = topk_all

    def rebuild_after_migration(self):
        """Group layouts + wire indices after apply_swaps mutated the plan;
        grouped-GEMM ops are rebuilt too (compacted weight copies and splits
        go stale when slot contents or per-slot rows change). Returns the
        host ms spent (book-keeping)."""
        t0 = time.perf_counter()
        self._build_group_state(self._topk_all)
        if self._gemm_backend == "grouped" and self._grouped_ops is not None:
            self._build_grouped_ops()
        if self.hc_enabled:
            self._rebuild_hc_bundles()
        return (time.perf_counter() - t0) * 1e3

    # -- transports ---------------------------------------------------------

    def enable_nvshmem(self, local_world_size: int, num_comm_sm: int = 8,
                       split_headroom: float = 2.0):
        """One All2AllSingle pair reused across all m group calls.

        max_split = headroom * current max per-(group, src->dst) pair rows
        (the op never validates per-call splits against max_split, so the
        runner asserts them itself on every layout rebuild — a hard failure
        beats silent staging overflow after migration reshapes the wire)."""
        import flux  # GPU-side only

        self._epic_max_split = max(
            1, int(self.elay.max_pair_rows * split_headroom))
        self._a2a_hidden = flux.All2AllSingle(
            self.group, self._epic_max_split, self.cfg.H, local_world_size,
            self.dtype,
        )
        self._a2a_probs = flux.All2AllSingle(
            self.group, self._epic_max_split, 1, local_world_size,
            torch.float32,
        )
        self._num_comm_sm = num_comm_sm
        self.transport = "nvshmem"
        self._refresh_nvshmem_splits()

    def _refresh_nvshmem_splits(self):
        dev = self.device
        self._g_in_splits = [
            torch.tensor(grp.send_counts, dtype=torch.int32, device=dev)
            for grp in self.elay.groups
        ]
        self._g_out_splits = [
            torch.tensor(grp.recv_counts, dtype=torch.int32, device=dev)
            for grp in self.elay.groups
        ]

    def enable_hier_compress(self, tp_env, local_world_size: int,
                             headroom: float = 1.5,
                             relay: str = "identity"):
        """EPIC Mode-2 dispatch transport: per-group GemmGroupedV2AGScatterOp
        instances (a2av_hier_compress) driven through dispatch_only over the
        virtual physical-slot expert space. relay='identity' is the faithful
        PXN shape (inter-node to the same-index GPU, NVLink forward =
        FLUX_A2AV_RELAY_IDENTITY); 'balanced' is our chunked-relay ablation.
        Requires enable_nvshmem() first (the per-entry probs side-wire stays
        on All2AllSingle — the fused op moves token rows only). Capacity env
        knobs are process-global ctor-reads: set per instance, in
        SPMD-identical group order, BEFORE each ctor."""
        import os

        import flux

        assert self.transport == "nvshmem", (
            "enable_hier_compress requires enable_nvshmem() first")
        assert relay in ("identity", "balanced")
        os.environ["FLUX_A2AV_RELAY_IDENTITY"] = (
            "1" if relay == "identity" else "0")
        os.environ.pop("FLUX_A2AV_LB_UNION", None)  # baseline: no lb_union
        self._hc_relay = relay
        self._hc_L = local_world_size
        self._hc_headroom = headroom
        self._hc_tp_env = tp_env
        self._hc_bundles = build_epic_hc_bundles(
            self.plan, self._topk_all, self.m, local_world_size)
        self._hc_kg = [b.K_g for b in self._hc_bundles]
        self._hc_ops = []
        self._hc_splits_gpu = []
        self._hc_scatter_gpu = []
        self._hc_caps = []
        ntokens = self.cfg.R * self.cfg.S
        for b in self._hc_bundles:
            knobs = epic_hc_required_knobs(b, self.cfg.R, local_world_size)
            caps = {k: int(int(v) * headroom) + 1 for k, v in knobs.items()}
            for k, v in caps.items():
                os.environ[k] = str(v)
            self._hc_caps.append(caps)
            moe_args = flux.MoeArguments(
                max_ntokens=ntokens,
                hidden=self.cfg.H,
                ffn_hidden=self.ffn_size_shard,
                nexperts=b.E_virt,
                topk=b.K_g,
                input_dtype=self.dtype,
                output_dtype=self.dtype,
            )
            self._hc_ops.append(flux.GemmGroupedV2AGScatterOp(
                tp_env=tp_env, moe_args=moe_args,
                a2av_dispatch=True, a2av_hier_compress=True))
            self._hc_splits_gpu.append(b.meta.splits.cuda())
            self._hc_scatter_gpu.append(b.meta.scatter_index.cuda())
        self.hc_enabled = True

    def _rebuild_hc_bundles(self):
        """Post-migration: bundles are pure functions of (plan, routing);
        op instances are frozen (capacities + topk) — hard-assert fit."""
        self._hc_bundles = build_epic_hc_bundles(
            self.plan, self._topk_all, self.m, self._hc_L,
            fixed_kg=self._hc_kg)
        for g, b in enumerate(self._hc_bundles):
            knobs = epic_hc_required_knobs(b, self.cfg.R, self._hc_L)
            for k, v in knobs.items():
                assert int(v) <= self._hc_caps[g][k], (
                    f"group {g}: post-migration {k} demand {v} exceeds the "
                    f"ctor capacity {self._hc_caps[g][k]}; raise headroom")
            self._hc_splits_gpu[g] = b.meta.splits.cuda()
            self._hc_scatter_gpu[g] = b.meta.scatter_index.cuda()

    def dispatch_group_hc(self, g: int):
        """hier_compress dispatch for group g: fused-op wire + dense
        materialization straight into the v1 hidden_buf segment (order
        theorem: fused dense order == v1 slot-major segment order; pad rows
        are the block tail and are dropped), plus the v1 per-entry probs
        side-wire."""
        b = self._hc_bundles[g]
        grp = self.elay.groups[g]
        base = self.elay.seg_start[grp.slot_lo]
        n_rows = sum(grp.seg_rows)
        dense, _ssi, _ssc, m_ep = self._hc_ops[g].dispatch_only(
            self._last_inputs, self._hc_splits_gpu[g],
            self._hc_scatter_gpu[g],
            b.meta.splits_per_source, b.meta.a2av_unique_counts,
        )
        n_real = int(m_ep) - int(b.pad_rows_per_rank[self.rank])
        assert n_real == n_rows, (
            f"group {g}: dense real rows {n_real} != layout rows {n_rows}")
        if n_rows:
            self.hidden_buf[base:base + n_rows].copy_(dense[:n_real])
        s_lo, s_hi = self.elay.send_off[g], self.elay.send_off[g + 1]
        r_lo, r_hi = self.elay.recv_off[g], self.elay.recv_off[g + 1]
        self._a2a_probs.forward(
            self.wsend_buf[s_lo:s_hi].view(-1, 1),
            self.wrecv_buf[r_lo:r_hi].view(-1, 1),
            self._g_in_splits[g], self._g_out_splits[g],
            self._num_comm_sm,
        )

    def enable_hc_combine(self, n_split: int = 4, pack_blocks: int = 3):
        """EPIC Mode-2 combine (S3): one flux.TopkReduceScatterOp per group
        over the group's virtual copy space (topk = K_g; pad rows carry
        zero data + zero vec_scale). Each group's run() returns the PARTIAL
        per-token sums [S, H]; final_out accumulates across groups
        (associative regrouping of the terminal Sum — allclose vs the
        direct arm, not bitwise). All indices come from the house python
        builders (flux.testing.a2av_combine_indices) applied verbatim in
        the virtual space. Compress silently degrades to plain hier at
        nnodes==1 (gather_rs cc:1504-1505) — 1n runs are plumbing smokes.
        NO C++: the combine-only op is already pybound."""
        import os

        import flux

        from .a2av_combine_indices import (
            build_a2av_combine_indices,
            build_a2av_compress_indices,
            build_a2av_unique_counts,
        )

        assert self.hc_enabled, "enable_hier_compress must run first"
        assert self.layers == "l01", "hc combine is an l01 phase"
        W = self.cfg.R
        L = self._hc_L
        nn = W // L
        self._hcc_nsplit = n_split
        self._hcc_pack_blocks = pack_blocks
        self._hcc_stream = torch.cuda.Stream(priority=-1)
        self._hcc_group_barrier = flux.GroupBarrier(self.group, False)
        self._hcc = []
        for b in self._hc_bundles:
            m_full = W * self.cfg.S * b.K_g
            cpr = m_full // W
            # Exact stage/conv/wire demands, replicating the op's collective
            # FLUX_CHECKs (gemm_grouped_v2_gather_rs.cc; same expressions as
            # sweeps/gen_matrix.a2av_rs_knob_demands) on THIS group's virtual
            # wire. cpr is NOT a valid conv/wire bound: conv aggregates a
            # source node's L ranks per remote dest LANE, so lane skew (EPIC
            # replica placement) can exceed S*K_g — bites at m=1 where the
            # whole batch shares one panel (m>=2 splits the skew).
            vc = b.virtual_choosed.long().cpu()
            ntok, kg = vc.shape
            tpr = ntok // W
            owner = (vc // b.gpe).flatten()
            home = (torch.arange(ntok, dtype=torch.long)
                    // tpr).repeat_interleave(kg)
            chunks = torch.zeros(W, W, dtype=torch.long)
            chunks.index_put_((home, owner),
                              torch.ones_like(home), accumulate=True)
            stage_d = conv_d = wire_d = cpr  # nn==1: plumbing-smoke fallback
            if nn > 1:
                Ug = build_a2av_unique_counts(
                    b.virtual_choosed, W, nn, b.gpe).long()
                stage_d = max(
                    int(chunks[gn * L:(gn + 1) * L,
                               [ns * L + gl for ns in range(nn)
                                if ns != gn]].sum())
                    for gn in range(nn) for gl in range(L))
                conv_d = max(
                    int(chunks[[tn * L + dl for tn in range(nn)
                                if tn != n2],
                               n2 * L:(n2 + 1) * L].sum())
                    for n2 in range(nn) for dl in range(L))
                wire_d = max(
                    int(Ug[[tn * L + dl for tn in range(nn)
                            if tn != n2], n2].sum())
                    for n2 in range(nn) for dl in range(L))
            os.environ["FLUX_A2AV_RS_MAX_SEND_ROWS"] = str(
                int(b.meta.m_per_rank.max()))
            os.environ["FLUX_A2AV_RS_MAX_STAGE_ROWS"] = str(max(stage_d, 1))
            os.environ["FLUX_A2AV_RS_MAX_CONV_ROWS"] = str(max(conv_d, 1))
            os.environ["FLUX_A2AV_RS_MAX_WIRE_ROWS"] = str(max(wire_d, 1))
            barriers = flux.create_tensor_list(
                (2 * n_split,), dtype=torch.int32, pg=self.group,
                ring_mode=True)
            op = flux.TopkReduceScatterOp(
                self.group, m_full, self.cfg.H, b.K_g, self.dtype,
                b.gpe,           # num_experts is PER-RANK (nex_total = *W)
                W, barriers, n_split,
                False,           # do_all_reduce
                False,           # use_read_mode
                nn,              # nnodes
                True,            # a2av_hier
                nn > 1,          # a2av_compress (degrades at 1n anyway)
            )
            routing_cpu = b.meta.scatter_index.flatten().cpu()
            splits_cpu = b.meta.splits.cpu()
            pack_idx, red_idx = build_a2av_combine_indices(
                routing_cpu, splits_cpu, self.rank, W, b.K_g)
            entry = {
                "op": op, "barriers": barriers,
                "routing": routing_cpu.cuda(),
                "pack": pack_idx, "red": red_idx,
                "uc": None, "wire": None, "redcsr": None,
                "m_this": int(b.meta.m_per_rank[self.rank]),
            }
            if nn > 1:
                uc = build_a2av_unique_counts(
                    b.virtual_choosed, W, nn, b.gpe)
                wp, wc, rp, rr = build_a2av_compress_indices(
                    routing_cpu, splits_cpu, uc, self.rank, W, nn, b.K_g)
                entry.update(uc=uc, wire=[wp, wc], redcsr=[rp, rr])
            entry["inbuf"] = torch.zeros(
                max(entry["m_this"], 1), self.cfg.H, dtype=self.dtype,
                device=self.device)
            entry["scale"] = torch.zeros(
                max(entry["m_this"], 1), dtype=torch.float32,
                device=self.device)
            entry["partial"] = torch.zeros(
                self.cfg.S, self.cfg.H, dtype=self.dtype,
                device=self.device)
            self._hcc.append(entry)
        self.hcc_enabled = True

    def enable_grouped_gemm(self, backend: str = "grouped"):
        """Build the per-group GEMM ops. 'grouped': one flux.GemmGroupedV2
        per group; 'gemmonly': per-segment flux.GemmOnly loop (parent
        semantics).

        Zero-row slots are legal: GemmGroupedV2's zero-split weight-pointer
        skew was FIXED upstream 2026-08-17 (gemm_grouped_v2.cc — the loop
        now advances the per-expert weight pointer past skipped experts),
        so all groups use storage-sharing slot views with full per-slot
        rows. Requires a binary at or after that fix; the sha-identity A/B
        against the old compacted-copy workaround proved equivalence before
        the workaround was removed. Splits refresh via
        rebuild_after_migration() (per-slot rows change when slots move)."""
        import flux

        assert backend in ("grouped", "gemmonly")
        self._gemm_backend = backend
        if backend == "gemmonly":
            self._gemm_only_op = flux.GemmOnly(
                self.dtype, self.dtype, self.dtype, use_fp8_gemm=False)
            return
        self._build_grouped_ops(flux)

    def _build_grouped_ops(self, flux_mod=None):
        if flux_mod is None:
            import flux as flux_mod
        # Storage-sharing slot views + full per-slot rows: zero-row slots are
        # legal since the 2026-08-17 GemmGroupedV2 zero-split fix (the
        # earlier compacted-weight-copy workaround was removed after the
        # same-binary sha-identity A/B proved the fix; requires a binary at
        # or after that fix). Views share storage with slot_fc1/2, so
        # migration weight swaps need no GEMM-op rebuild.
        self._grouped_ops = []
        self._grouped_splits = []
        if self.layers == "l01":
            self._grouped_ops_fc2 = []
        for g, grp in enumerate(self.elay.groups):
            rows = self._group_splits_cpu[g]
            w = self.slot_fc1[grp.slot_lo:grp.slot_hi]
            assert w.is_contiguous()
            self._grouped_ops.append(
                flux_mod.GemmGroupedV2(
                    w, int(w.shape[0]), self.dtype, self.dtype)
            )
            self._grouped_splits.append(rows)
            if self.layers == "l01":
                w2 = self.slot_fc2[grp.slot_lo:grp.slot_hi]
                assert w2.is_contiguous()
                self._grouped_ops_fc2.append(
                    flux_mod.GemmGroupedV2(
                        w2, int(w2.shape[0]), self.dtype, self.dtype)
                )

    # -- timed phase methods ------------------------------------------------

    def pack(self, inputs_shard: torch.Tensor, probs_shard: torch.Tensor):
        """Group-major pack of the whole send window (one launch pair per
        group keeps it simple; rows land at the group's send_buf offset).
        Under hier_compress the fused op packs hidden rows internally from
        inputs_shard, so only the probs side-wire is packed here."""
        self._last_inputs = inputs_shard
        off = self.elay.send_off
        for g in range(self.m):
            lo, hi = off[g], off[g + 1]
            if hi == lo:
                continue
            if not self.hc_enabled:
                torch.index_select(
                    inputs_shard, 0, self.g_send_row_index[g],
                    out=self.send_buf[lo:hi],
                )
            self.wsend_buf[lo:hi].copy_(
                probs_shard[self.g_send_row_index[g],
                            self.g_send_entry_logical[g]]
            )

    def dispatch_group(self, g: int):
        grp = self.elay.groups[g]
        s_lo, s_hi = self.elay.send_off[g], self.elay.send_off[g + 1]
        r_lo, r_hi = self.elay.recv_off[g], self.elay.recv_off[g + 1]
        if self.transport == "nvshmem":
            self._a2a_hidden.forward(
                self.send_buf[s_lo:s_hi], self.recv_buf[r_lo:r_hi],
                self._g_in_splits[g], self._g_out_splits[g],
                self._num_comm_sm,
            )
            self._a2a_probs.forward(
                self.wsend_buf[s_lo:s_hi].view(-1, 1),
                self.wrecv_buf[r_lo:r_hi].view(-1, 1),
                self._g_in_splits[g], self._g_out_splits[g],
                self._num_comm_sm,
            )
            return
        self.dist.all_to_all_single(
            self.recv_buf[r_lo:r_hi], self.send_buf[s_lo:s_hi],
            output_split_sizes=grp.recv_counts,
            input_split_sizes=grp.send_counts,
            group=self.comm_group,
        )
        self.dist.all_to_all_single(
            self.wrecv_buf[r_lo:r_hi], self.wsend_buf[s_lo:s_hi],
            output_split_sizes=grp.recv_counts,
            input_split_sizes=grp.send_counts,
            group=self.comm_group,
        )

    def scatter_group(self, g: int):
        r_lo, r_hi = self.elay.recv_off[g], self.elay.recv_off[g + 1]
        if r_hi == r_lo:
            return
        slots = self.g_place_slots[g]
        if not self.hc_enabled:
            # hier_compress places hidden rows in dispatch_group_hc
            self.hidden_buf.index_copy_(0, slots, self.recv_buf[r_lo:r_hi])
        self.weights_buf.index_copy_(0, slots, self.wrecv_buf[r_lo:r_hi])

    def gemm_group(self, g: int, sm_margin: int = 0):
        grp = self.elay.groups[g]
        seg_start = self.elay.seg_start
        lo_row = seg_start[grp.slot_lo]
        n_rows = sum(grp.seg_rows)
        if self._gemm_backend == "gemmonly":
            for p, start, end, _l in self.elay.gemm_segments:
                if not (grp.slot_lo <= p < grp.slot_hi):
                    continue
                self._gemm_only_op.forward(
                    self.hidden_buf[start:end], self.slot_fc1[p],
                    output_buf=self.out_buf[start:end],
                    fast_accum=False,
                )
            self.group_outputs[g] = self.out_buf[lo_row:lo_row + n_rows]
            return
        if n_rows == 0:
            self.group_outputs[g] = None
            return
        out = self._grouped_ops[g].forward(
            self.hidden_buf[lo_row:lo_row + n_rows],
            self._grouped_splits[g],
            sm_margin=sm_margin,
        )
        self.group_outputs[g] = out

    # -- layer1 phase methods (--layers l01) --------------------------------

    def act_group(self, g: int):
        """Activation between the two expert GEMMs: exact/erf GELU on the
        native-dtype GEMM0 output (the combined-driver convention,
        test_moe_l0l1_traffic.py:214)."""
        out = self.group_outputs[g]
        self.group_act[g] = (
            torch.nn.functional.gelu(out) if out is not None else None)

    def gemm1_group(self, g: int, sm_margin: int = 0):
        grp = self.elay.groups[g]
        n_rows = sum(grp.seg_rows)
        if n_rows == 0:
            self.group1_outputs[g] = None
            return
        if self._gemm_backend == "gemmonly":
            act = self.group_act[g]
            base = self.elay.seg_start[grp.slot_lo]
            out = torch.empty(n_rows, self.cfg.H, dtype=self.dtype,
                              device=self.device)
            for p, start, end, _l in self.elay.gemm_segments:
                if not (grp.slot_lo <= p < grp.slot_hi):
                    continue
                self._gemm_only_op.forward(
                    act[start - base:end - base], self.slot_fc2[p],
                    output_buf=out[start - base:end - base],
                    fast_accum=False,
                )
            self.group1_outputs[g] = out
            return
        self.group1_outputs[g] = self._grouped_ops_fc2[g].forward(
            self.group_act[g], self._grouped_splits[g], sm_margin=sm_margin)

    def combine_pack_group(self, g: int):
        """Expert-side combine pack: gather GEMM1 rows into recv-stream
        order (place_slots) and apply the per-row route-prob vec_scale in
        fp32 math (a2av_combine.cu:107-114 / moe_gather_rs_utils.py:96
        convention). weights_buf is slot-major = the scale per packed row.
        Under hc combine: assemble the op's expert-major input (GEMM1 rows
        + zero pad tail) and the per-row scales; the vec_scale multiply
        happens inside the op's pack kernel."""
        grp = self.elay.groups[g]
        n_rows = sum(grp.seg_rows)
        if getattr(self, "hcc_enabled", False):
            if n_rows == 0:
                return
            e = self._hcc[g]
            base = self.elay.seg_start[grp.slot_lo]
            e["inbuf"][:n_rows].copy_(self.group1_outputs[g])
            e["scale"][:n_rows].copy_(
                self.weights_buf[base:base + n_rows])
            return
        r_lo, r_hi = self.elay.recv_off[g], self.elay.recv_off[g + 1]
        if r_hi == r_lo:
            return
        rows = self.group1_outputs[g].index_select(0, self.g_comb_pack[g])
        scale = self.weights_buf.index_select(0, self.g_place_slots[g])
        self.comb_send_buf[r_lo:r_hi] = (
            rows.float() * scale.unsqueeze(1)).to(self.dtype)

    def combine_group(self, g: int):
        """Reverse wire: the SAME All2AllSingle pair (max_pair_rows is
        transpose-invariant) with swapped splits. Under hc combine: the
        per-group TopkReduceScatterOp run (triton-precedent protocol —
        barrier fill/zero bracket, GroupBarrier, side cp_stream)."""
        if getattr(self, "hcc_enabled", False):
            e = self._hcc[g]
            b = self._hc_bundles[g]
            stream = torch.cuda.current_stream()
            e["barriers"][self.rank % self._hc_L].fill_(1)
            self._hcc_group_barrier.barrier_all(stream.cuda_stream)
            e["op"].run(
                [e["inbuf"]], e["partial"],
                self.rank * b.gpe, b.gpe,
                self._hc_splits_gpu[g], e["routing"],
                [e["scale"]],
                self._hcc_pack_blocks,
                self._hcc_stream.cuda_stream,
                b.meta.splits_per_source,
                e["pack"], e["red"],
                e["uc"], e["wire"], e["redcsr"],
            )
            self._hcc_group_barrier.barrier_all(stream.cuda_stream)
            e["barriers"][self.rank % self._hc_L].zero_()
            e["op"].reset_buffer()
            return
        grp = self.elay.groups[g]
        r_lo, r_hi = self.elay.recv_off[g], self.elay.recv_off[g + 1]
        s_lo, s_hi = self.elay.send_off[g], self.elay.send_off[g + 1]
        if self.transport == "nvshmem":
            self._a2a_hidden.forward(
                self.comb_send_buf[r_lo:r_hi], self.comb_recv_buf[s_lo:s_hi],
                self._g_out_splits[g], self._g_in_splits[g],
                self._num_comm_sm,
            )
            return
        self.dist.all_to_all_single(
            self.comb_recv_buf[s_lo:s_hi], self.comb_send_buf[r_lo:r_hi],
            output_split_sizes=grp.send_counts,
            input_split_sizes=grp.recv_counts,
            group=self.comm_group,
        )

    def accumulate_group(self, g: int):
        """Deterministic home staging: each combine recv row owns a unique
        (token, topk-slot) cell (transposition theorem + the comb_dst_slot
        permutation assert). Under hc combine: accumulate the group's
        partial per-token sums."""
        if getattr(self, "hcc_enabled", False):
            if g == 0:
                self.final_out.zero_()
            self.final_out += self._hcc[g]["partial"]
            return
        s_lo, s_hi = self.elay.send_off[g], self.elay.send_off[g + 1]
        if s_hi == s_lo:
            return
        self.stage_buf.index_copy_(
            0, self.g_comb_dst[g], self.comb_recv_buf[s_lo:s_hi])

    def finalize_sum(self):
        """The single terminal Sum (EPIC Fig 10(b)): fixed (t, j) order ⇒
        bitwise m-invariant final output; matches the house
        view(S, K, H).sum(1) convention. Under hc combine the partials were
        already accumulated per group — no-op."""
        if getattr(self, "hcc_enabled", False):
            return
        self.final_out = self.stage_buf.view(
            self.cfg.S, self.cfg.K, self.cfg.H).sum(1)

    # -- parent overrides (guard against ungrouped-path misuse) -------------

    def a2av(self):
        raise RuntimeError("epic runner dispatches per group: dispatch_group")

    def place(self):
        raise RuntimeError("epic runner scatters per group: scatter_group")

    def gemm(self, *args, **kwargs):
        raise RuntimeError("epic runner GEMMs per group: gemm_group")

    # -- migration ----------------------------------------------------------

    def apply_migration(self, swaps):
        """Weight exchange + plan mutation + layout rebuild for a decided
        swap list. Returns (recv_bytes, relayout_ms)."""
        ops, keepalive, recv_bytes = swap_weight_ops(
            swaps, self.rank, self.slot_fc1,
            self.slot_fc2 if self.place_fc2 else None,
            self.dist, self.group,
        )
        if ops:
            for req in self.dist.batch_isend_irecv(ops):
                req.wait()
        del keepalive
        apply_swaps(self.plan, swaps)
        self.migration_swap_bytes += recv_bytes
        relayout_ms = self.rebuild_after_migration()
        return recv_bytes, relayout_ms

    # -- accounting ---------------------------------------------------------

    def dup_stats(self):
        return epic_dup_stats(self.elay, self.cfg.R)

    def internode_rows(self):
        return (self.elay.internode_send_rows, self.elay.internode_recv_rows)
