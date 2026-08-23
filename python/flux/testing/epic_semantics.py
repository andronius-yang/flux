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

import os
import time
from dataclasses import dataclass, field, replace as _dc_replace

import torch

from .ultraep_semantics import (
    UltraEPConfig,
    UltraEPPlan,
    reroute_expand,
)
from .eplb_semantics import EPLBLayer0Runner
from .loccap_semantics import plan_tensors_from_hosts

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
                    num_nodes: int,
                    replica_select: str = "local_static") -> UltraEPPlan:
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

    # Replica rule (campaign-2 knob): local_static == the paper's D6 rule
    # (default); local_spread == per-source largest-remainder equal split
    # (SGLang-dynamic analog, ablation). Both sender-local. The
    # ep_gpu_plan producer is device-agnostic (CPU tensors here) so the
    # setup reference stays bitwise-consistent with the timed planner.
    assert replica_select in ("local_static", "local_spread"), replica_select
    if replica_select == "local_spread":
        from .ep_gpu_plan import local_spread_rank_quota_prefix
        rqp = local_spread_rank_quota_prefix(tpe, lcnts,
                                             cfg.max_replicas_dim)
    else:
        rqp = epic_rank_quota_prefix(cfg, tpe, lcnts)
    plan = UltraEPPlan(
        cfg=cfg, tpe=tpe.to(torch.int32), p2l=p2l, l2p=l2p, lcnts=lcnts,
        quota=quota, quota_prefix=quota_prefix,
        rank_quota_prefix=rqp,
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


def build_nodeaware_plan(cfg: UltraEPConfig, tpe: torch.Tensor,
                         blob: dict) -> UltraEPPlan:
    """--placement nodeaware: PLACE-lambda placement from a
    <mid>.placement.json sidecar (sweeps/predict_placement.py — pool
    co-occurrence partition + per-node-first coverage replication).

    Same contract as build_epic_plan: placement from POOL statistics only
    (the sidecar), quotas informational, rank_quota_prefix = the D6 step
    function — so nodeaware composes with BOTH routers: --router d6 uses
    the prefix, --router loccap overrides per token via plan.phys_override.
    Slot recipe = loccap_semantics.plan_tensors_from_hosts, the EXACT
    recipe the offline simulator uses (predicted == realized incidence is a
    driver assert, not a hope). cfg is MUTATED (max_replicas_dim) like the
    sibling builders."""
    assert tuple(tpe.shape) == (cfg.R, cfg.G)
    assert int(blob.get("version", -1)) in (1, 2), "unknown placement version"
    for key, want in (("G", cfg.G), ("W", cfg.R), ("nlp", cfg.nlp)):
        got = int(blob[key])
        assert got == want, (
            f"placement sidecar {key}={got} != cfg {want} (check "
            f"--redundant_per_rank against the sidecar's redundant_per_rank)")
    hosts = blob["hosts"]
    assert len(hosts) == cfg.G
    cfg.max_replicas_dim = cfg.R
    p2l, l2p_small, lcnts = plan_tensors_from_hosts(hosts, cfg.R, cfg.nlp)
    l2p = torch.full((cfg.G, cfg.max_replicas_dim), -1, dtype=torch.int32)
    l2p[:, :l2p_small.shape[1]] = l2p_small

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
    plan.epic_est_internode_send = [0.0] * cfg.R
    plan.epic_est_internode_recv = [0.0] * cfg.R
    plan.epic_redundancy = (lcnts.long() - 1).tolist()
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
# Per-iteration timed GPU planning (SCHEMA rule 5), m=1 scope
# ---------------------------------------------------------------------------


def slot_loads_from_rqp(rqp_all: torch.Tensor, l2p: torch.Tensor,
                        lcnts: torch.Tensor, P: int) -> torch.Tensor:
    """[P] int64 batch tokens per physical slot from the iteration's
    rank-quota prefixes. Parity target: slot_batch_loads (:827-842) —
    identical values when rqp_all equals plan.rank_quota_prefix."""
    dev = rqp_all.device
    rqp = rqp_all.long()
    alloc = rqp.clone()
    alloc[:, :, 1:] -= rqp[:, :, :-1]
    inst = alloc.sum(dim=0)                          # [G, Cmax]
    Cmax = inst.shape[1]
    j = torch.arange(Cmax, device=dev, dtype=torch.int64).unsqueeze(0)
    valid = j < lcnts.long().unsqueeze(1)
    out = torch.zeros(P, dtype=torch.int64, device=dev)
    out[l2p.long()[valid]] = inst[valid]             # l2p injective on valid
    return out


def plan_migration_swaps_gpu(p2l: torch.Tensor, slot_loads: torch.Tensor,
                             tau_tokens: float, ranks_per_node: int,
                             nlp: int, G: int):
    """Device (or cpu-tensor) twin of plan_migration_swaps: identical swap
    list, tie-breaks reproduced exactly (stable descending rank sort; pair
    argmax key gain*nlp^2 + (nlp^2-1-flat) == strict-> scan, ties ->
    lowest (a, b)). One small D2H at the end; the tau gate is applied
    host-side on the exact integer gains.

    Consumes the ITERATION's slot loads (rule-5 fidelity fix: the decision
    reads the freshly derived loads, closing the never-read
    loads_gather_buf gap of the pre-2026-08-20 driver)."""
    dev = slot_loads.device
    P = slot_loads.numel()
    R = P // nlp
    D = ranks_per_node
    num_nodes = R // D
    p2l = p2l.long()
    sl = slot_loads.long()
    gl = sl.view(R, nlp).sum(dim=1)

    vals, idx = torch.sort(gl.view(num_nodes, D), dim=1, descending=True,
                           stable=True)
    order = idx + (torch.arange(num_nodes, device=dev,
                                dtype=torch.int64).unsqueeze(1) * D)
    half = D // 2
    if half == 0:
        return []
    rh = order[:, :half]                             # [nodes, half]
    rl = order.flip(1)[:, :half]
    Lh, Ll = gl[rh], gl[rl]

    has = torch.zeros(R, G, dtype=torch.bool, device=dev)
    valid = p2l >= 0
    rank_of_slot = torch.arange(P, device=dev, dtype=torch.int64) // nlp
    has[rank_of_slot[valid], p2l[valid]] = True

    slots = torch.arange(nlp, device=dev, dtype=torch.int64)
    pa = rh.unsqueeze(-1) * nlp + slots              # [nodes, half, nlp]
    pb = rl.unsqueeze(-1) * nlp + slots
    la, lb = p2l[pa], p2l[pb]
    wa, wb = sl[pa], sl[pb]
    in_l = has[rl.unsqueeze(-1).expand_as(la), la.clamp(min=0)] & (la >= 0)
    in_h = has[rh.unsqueeze(-1).expand_as(lb), lb.clamp(min=0)] & (lb >= 0)

    la_e, lb_e = la.unsqueeze(-1), lb.unsqueeze(-2)  # [nodes, half, nlp, nlp]
    wa_e, wb_e = wa.unsqueeze(-1), wb.unsqueeze(-2)
    Lh_e = Lh.unsqueeze(-1).unsqueeze(-1)
    Ll_e = Ll.unsqueeze(-1).unsqueeze(-1)
    gain = (torch.maximum(Lh_e, Ll_e)
            - torch.maximum(Lh_e - wa_e + wb_e, Ll_e - wb_e + wa_e))
    invalid = (
        (la_e == lb_e)
        | (in_l.unsqueeze(-1) & (lb_e != la_e))
        | (in_h.unsqueeze(-2) & (la_e != lb_e))
        | (gain <= 0)
        | (Lh_e <= Ll_e)
    )
    n2 = nlp * nlp
    flat_idx = torch.arange(n2, device=dev,
                            dtype=torch.int64).view(1, 1, nlp, nlp)
    key = (gain * n2 + (n2 - 1 - flat_idx)).masked_fill(invalid, -1)
    best_key, best_flat = key.view(num_nodes, half, n2).max(dim=-1)
    best_gain = torch.gather(gain.view(num_nodes, half, n2), -1,
                             best_flat.unsqueeze(-1)).squeeze(-1)

    blob = torch.stack([
        rh, rl, best_flat // nlp, best_flat % nlp, best_gain,
        (best_key >= 0).long(), (Lh > Ll).long(),
    ], dim=-1).reshape(-1, 7).cpu()                  # the ONE D2H
    swaps = []
    for row in blob.tolist():
        r_h, r_l, a, b, g_val, found, heavier = row
        if heavier and found and g_val > tau_tokens:
            swaps.append((r_h, a, r_l, b, int(g_val)))
    return swaps


@dataclass
class EpicIterPlan:
    """One iteration's routing-derived plan for the m=1 EPIC arms, produced
    on-device by EpicIterPlanner.derive inside the timed `plan` bracket."""

    # direct/probs-wire + gemm + l01 core (group 0 == everything at m=1)
    send_row_index: torch.Tensor
    send_entry_logical: torch.Tensor
    place_slots: torch.Tensor
    comb_dst_slot: torch.Tensor      # None on l0
    in_splits: torch.Tensor          # [R] int32 device
    out_splits: torch.Tensor
    send_counts: list
    recv_counts: list
    seg_rows: list
    seg_start: list
    gemm_segments: list
    n_recv: int
    max_pair_rows: int
    # migration decision input (device, from this iteration's rqp)
    slot_loads: torch.Tensor
    # hc transport (None-family when hc disabled)
    hc_splits: torch.Tensor          # [E_virt] int32 device
    hc_scatter: torch.Tensor         # [ntokens, K_g] int32 device
    hc_sps_cpu: torch.Tensor         # [R, E_virt] int32 CPU (op host arg)
    hc_uc_cpu: torch.Tensor          # [R, R+nn] int32 CPU (op host arg)
    hc_pad_mine: int
    hc_kg_iter: int
    hc_m_this: int
    # v2b in-window mode: ONLY the virtual routing ships from python; the
    # op derives splits/scatter/sps/uc itself (derive_routed_meta)
    hc_vce: torch.Tensor = None      # [ntokens, K_g] int32 device or None
    # hc combine (l01 x hc python-meta mode; None-family otherwise —
    # under in-window mode the combine op builds these itself)
    hcc_routing: torch.Tensor = None  # flattened scatter int32 device
    hcc_pack: torch.Tensor = None
    hcc_red: torch.Tensor = None
    hcc_uc: torch.Tensor = None       # CPU int32 (op host arg)
    hcc_wire: list = None             # [wire_ptr, wire_copy] device
    hcc_redcsr: list = None           # [red_ptr, red_row] device


def python_meta_from_vce(vce, R, S, gpe, nn, L):
    """Reference (splits, scatter_index, splits_per_source, unique_counts)
    from a GIVEN virtual routing vce [ntokens, kg] — the python side of the
    v2b in-window guard, decoupled from any particular entry-column order
    (the fast tail ships vce in topk column order; within a virtual slot
    the wire order depends only on (vslot, token), so any deterministic
    column order is contract-valid). Mirrors the legacy non-inwindow
    branch formulas verbatim."""
    dev = vce.device
    ntokens = R * S
    kg = vce.shape[1]
    E_virt = R * gpe
    vce_flat = vce.long().reshape(-1)
    scatter_index = (vce_flat.argsort(stable=True).argsort()
                     .int().view(ntokens, kg))
    splits = torch.bincount(vce_flat, minlength=E_virt).int()
    home = (torch.arange(ntokens, device=dev, dtype=torch.int64)
            // S)
    src_of_copy = home.repeat_interleave(kg)
    sps = torch.bincount(src_of_copy * E_virt + vce_flat,
                         minlength=R * E_virt).view(R, E_virt)
    owner = vce.long() // gpe
    flags = torch.zeros(ntokens, R, dtype=torch.bool, device=dev)
    flags.scatter_(1, owner, True)
    u_mat = flags.view(R, S, R).sum(1)
    U_mat = (flags.view(ntokens, nn, L).any(dim=2)
             .view(R, S, nn).sum(1))
    uc = torch.cat([u_mat, U_mat], dim=1)
    return splits, scatter_index, sps.int(), uc.int()


class EpicIterPlanner:
    """Per-iteration device planner for the EPIC m=1 arms (SCHEMA rule 5):
    quotas (D6), the reroute expansion, wire/scatter/segment indices, the
    hc virtual-space metadata, and the hc-combine index sets are all
    recomputed each iteration on device. One-shot ctor state: the (mutable,
    migration-refreshed) placement tensors and the replicated routing.

    Scope: m == 1 with the D6 router (the epic sweep arms) or the
    PLACE-lambda `loccap_gpu` router (router="loccap_gpu" + eps — OUR
    per-token replica-selection port, flux.testing.placelambda_gpu; the
    epic harness is only the vehicle). m > 1 / the exact python loccap
    stay on the legacy setup-time path (timing_accounting=
    legacy_untimed_plan)."""

    def __init__(self, plan: UltraEPPlan, rank: int, device,
                 topk_all: torch.Tensor, local_world_size: int,
                 l01: bool = False, hc: bool = False, hcc: bool = False,
                 kg_frozen: int = None,
                 replica_select: str = "local_static",
                 inwindow_meta: bool = False,
                 router: str = "d6", eps: float = None,
                 route_group=None, exchange_fn=None, f_cap: int = -1,
                 groups_meta=None):
        # local_static == the paper's own D6 rule (src mod lcnts) and
        # stays EPIC's default; local_spread is the SGLang-dynamic-analog
        # ablation (campaign-2 knob). No quota mode for epic.
        assert replica_select in ("local_static", "local_spread"), (
            replica_select)
        assert router in ("d6", "loccap_gpu", "loccap_sl"), router
        assert router == "d6" or eps is not None, (
            f"router {router} needs eps")
        self.router = router
        self.eps = eps
        self.route_group = route_group
        self.exchange_fn = exchange_fn
        self.f_cap = f_cap
        self.last_kernel_stats = None
        cfg = plan.cfg
        self.cfg = cfg
        self.plan = plan
        self.rank = rank
        self.device = device
        self.replica_select = replica_select
        # campaign-2 v2b: the op derives splits/scatter/sps/uc IN-WINDOW
        # (dispatch_only_routed / derive_routed_meta); the python planner
        # then only builds the direct-layout probs-wire indices + the
        # virtual routing (vce) + slot loads for the migration decision.
        self.inwindow_meta = inwindow_meta
        self.l01 = l01
        self.hc = hc
        self.hcc = hcc and not inwindow_meta
        self.L = local_world_size
        self.nn = cfg.R // local_world_size
        self.gpe = cfg.nlp + 1
        self.E_virt = cfg.R * self.gpe
        self.kg_frozen = kg_frozen
        self.topk_all = topk_all.long().to(device)
        ntokens = cfg.R * cfg.S
        self._home_of_token = (
            torch.arange(ntokens, device=device, dtype=torch.int64) // cfg.S)
        self._pad_vslot = self._home_of_token * self.gpe + cfg.nlp
        # multi-group rule-5 (8.22: EPIC m>1 fairness — the d6 fast tail):
        # groups_meta = [{slot_lo, slot_hi, gpe, kg}] per PEO group; the
        # slot->group lookup drives the group-major canonical orders and
        # every per-group slice. None/len==1 = the m=1 path.
        self.groups_meta = groups_meta
        self.m_groups = len(groups_meta) if groups_meta else 1
        if groups_meta:
            grp_of = torch.zeros(cfg.nlp, dtype=torch.int64, device=device)
            for gi, gm in enumerate(groups_meta):
                # per-group virtual space keeps the GLOBAL gpe (= nlp+1):
                # slots outside [slot_lo, slot_hi) are empty in that
                # group's routing; the pad slot stays gpe-1 == nlp
                assert gm["gpe"] == self.gpe, (gm, self.gpe)
                grp_of[gm["slot_lo"]:gm["slot_hi"]] = gi
            self._grp_of_slot = grp_of
            self._pad_vslot_g = [
                self._home_of_token * gm["gpe"] + (gm["gpe"] - 1)
                for gm in groups_meta]
        if router == "loccap_sl":
            # relaxed kernel arm: sender-local row + communicated agreement.
            # l01 combine IS supported since 8.22.route, but only through
            # the INWINDOW path (derive_routed_meta + derive_combine_meta
            # refresh the combine metadata per iteration; enable_hc_combine
            # m_capacity sizes the inbuf to the provable recv bound). The
            # LEGACY python hcc planner path stays excluded — the fast tail
            # ships vce only.
            assert not self.hcc, (
                "loccap_sl needs the inwindow meta path for l01 (legacy "
                "python hcc planning is not built from the fast tail); "
                "run with --hc_meta inwindow + combine capacity mode")
            self._topk_own_i32 = (topk_all[rank].int().contiguous()
                                  .to(device))
            self._phys_gather = torch.empty(
                cfg.R * cfg.S * cfg.K, dtype=torch.int32, device=device)
        self.refresh_placement()

    def refresh_placement(self):
        """(Re)upload the placement tensors — at ctor and after every
        applied migration (the swap mutates plan.p2l/l2p in place)."""
        dev = self.device
        self.l2p = self.plan.l2p.to(dev)
        self.lcnts = self.plan.lcnts.to(dev)
        self.p2l = self.plan.p2l.long().to(dev)
        self._p2l_host = self.plan.p2l.long().clone()

    def local_loads(self) -> torch.Tensor:
        return torch.bincount(self.topk_all[self.rank].reshape(-1),
                              minlength=self.cfg.G).to(torch.int32)

    def _tok_all(self):
        cfg = self.cfg
        return (torch.arange(cfg.S, device=self.device, dtype=torch.int64)
                .repeat_interleave(cfg.K).unsqueeze(0)
                .expand(cfg.R, cfg.S * cfg.K).contiguous())

    def _exchange(self, phys_own):
        """Sender-local row exchange: agreement across ranks by
        COMMUNICATION (the relaxed kernel's rows are authored once, never
        recomputed elsewhere). exchange_fn overrides for single-process
        R-emulation tests (no torch.distributed init needed)."""
        if self.exchange_fn is not None:
            return self.exchange_fn(phys_own)
        import torch.distributed as dist
        dist.all_gather_into_tensor(self._phys_gather,
                                    phys_own.reshape(-1).contiguous(),
                                    group=self.route_group)
        return self._phys_gather

    def derive(self, loads_gather_buf: torch.Tensor) -> EpicIterPlan:
        from .ep_gpu_plan import (
            d6_rank_quota_prefix,
            reroute_expand_all_gpu,
        )

        cfg = self.cfg
        R, G, S, K, nlp = cfg.R, cfg.G, cfg.S, cfg.K, cfg.nlp
        dev = self.device
        tpe_all = loads_gather_buf.view(R, G)
        kstats = None
        if (self.router == "loccap_sl"
                and int(os.getenv("FLUX_PLL_FORCE_REF", "0"))):
            # bisection hook: identical machinery (capacity mode, blob,
            # binding) but the ROUTING never changes across iterations —
            # discriminates capacity-mode bugs from changing-routing bugs
            return self.derive_reference()
        if self.router == "loccap_sl":
            # PLACE-lambda sender-local FUSED KERNEL (relaxed; the pll_*
            # kernel arm): own-row routing on device + the phys-row
            # allgather, all inside the timed plan bracket. The kernel's
            # shared tables derive from the same tpe allgather the epic
            # path already pays.
            if int(os.getenv("FLUX_PLL_SL_EXT", "0")):
                # patched-kernel A/B (templated register-mask tier-3 +
                # remote-cap flavor) via the standalone JIT extension —
                # no flux rebuild, main .so stays the binary elsewhere
                from .pll_sl_ext import load_ext
                phys_own, kstats = load_ext().route_sl(
                    self._topk_own_i32, tpe_all.int().contiguous(),
                    self.l2p.int(), self.lcnts.int(), self.rank, nlp,
                    self.L, self.eps, self.f_cap)
            else:
                import flux
                phys_own, kstats = flux.placelambda_route_sl(
                    self._topk_own_i32, tpe_all, self.l2p, self.lcnts,
                    self.rank, nlp, self.L, self.eps, f_cap=self.f_cap)
            phys_all = self._exchange(phys_own).long().view(R, S * K)
            tok_all = self._tok_all()
            rqp = None
        elif self.router == "loccap_gpu":
            # PLACE-lambda per-token replica selection, re-derived on
            # device per iteration (rule 5) — the deterministic torch arm.
            from .placelambda_gpu import loccap_route_gpu
            phys3, _ = loccap_route_gpu(
                self.topk_all.view(R, S, K), self.p2l.int(), self.l2p,
                self.lcnts, nlp, self.L, self.eps,
                remote_cap_only=bool(int(os.getenv(
                    "FLUX_LOCCAP_REMOTE_CAP_ONLY", "0"))))
            phys_all = phys3.long().reshape(R, S * K)
            tok_all = self._tok_all()
            rqp = None
        elif self.replica_select == "local_spread":
            from .ep_gpu_plan import local_spread_rank_quota_prefix
            rqp = local_spread_rank_quota_prefix(
                tpe_all, self.lcnts, cfg.max_replicas_dim)
        else:  # local_static == D6, the paper rule
            _d6_fast = (self.router == "d6"
                        and int(os.getenv("FLUX_PLL_FAST_TAIL", "1"))
                        and (not self.hc or self.inwindow_meta)
                        and not self.hcc)
            if _d6_fast:
                # local_static SHORTCUT (8.22 profiling): the general
                # replica-selection engine (full-E sort + run-ordinal +
                # [E, Cmax] cummax/searchsorted + bincount sync + the
                # host-looped coprime-interleave search) provably reduces
                # to ONE elementwise gather for this rule — the step
                # quota prefix sends EVERY ordinal of (src, expert) to
                # j* = src mod lcnts, interleave permutations included.
                # Bit-identical physical assignment, ~5 ms -> ~0.05 ms;
                # entry order differs (token-major vs expert-major) but
                # the fast tail re-sorts canonically, so every layout is
                # bitwise-equal (check_against-verified). This is a PORT
                # overhead removal, not a routing change — the same
                # fairness category as the tail itself.
                N = S * K
                topkf = self.topk_all.view(R, N)
                srcs = torch.arange(R, device=dev,
                                    dtype=torch.int64).unsqueeze(1)
                Ce = self.lcnts.long()[topkf]
                phys_all = self.l2p.long()[
                    topkf, torch.remainder(srcs, Ce)]
                tok_all = (torch.arange(S, device=dev, dtype=torch.int64)
                           .repeat_interleave(K).unsqueeze(0)
                           .expand(R, N))
                self._last_phys_all = phys_all
                ips = self._fast_tail_g(tok_all, phys_all)
                return ips[0] if self.m_groups == 1 else ips
            rqp = d6_rank_quota_prefix(tpe_all, self.lcnts,
                                       cfg.max_replicas_dim)
        if rqp is not None:
            tok_all, phys_all = reroute_expand_all_gpu(
                rqp, self.l2p, self.lcnts, self.topk_all, cfg.interleave)
        self._last_phys_all = phys_all
        if (self.router in ("loccap_sl", "loccap_gpu")
                and int(os.getenv("FLUX_PLL_FAST_TAIL", "1"))
                and (not self.hc or self.inwindow_meta)
                and not self.hcc):
            if int(os.getenv("FLUX_PLL_TAIL_GRAPH", "0")):
                ip = self._derive_fast_graphed(phys_all, kstats)
            else:
                ip = self._derive_from_phys_fast(phys_all, kstats)
        else:
            ip = self._derive_from_phys(tok_all, phys_all, rqp, kstats)
        if (self.router == "loccap_sl"
                and getattr(self, "_check_iters", False)):
            # validation runs (FLUX_PLL_CHECK_ITERS=1): audit EVERY
            # iteration's relaxed routing — perturbs timing, G1-gate only
            self.check_relaxed(ip, self.relaxed_bounds,
                               ref_incidence=getattr(
                                   self, "ref_incidence", None))
        return ip

    def derive_reference(self) -> EpicIterPlan:
        """Deterministic ip from the setup reference routing
        (plan.phys_override) — the bitwise-checkable side of the relaxed
        contract, used for the setup drift guard and the final
        deterministic correctness iteration."""
        ov = self.plan.phys_override
        assert ov is not None, "derive_reference needs plan.phys_override"
        cfg = self.cfg
        phys_all = ov.long().to(self.device).view(cfg.R, cfg.S * cfg.K)
        self._last_phys_all = phys_all
        if (self.router in ("loccap_sl", "loccap_gpu")
                and int(os.getenv("FLUX_PLL_FAST_TAIL", "1"))
                and (not self.hc or self.inwindow_meta)
                and not self.hcc):
            return self._derive_from_phys_fast(phys_all, None)
        return self._derive_from_phys(self._tok_all(), phys_all, None, None)

    def _recv_pad_capacity(self):
        """Fixed receive padding for the sync-free layout: the provable
        recv bound (loccap_sl capacity mode) or, for the deterministic
        loccap_gpu arm (static routing per cell), the exact setup count
        (computed once — a setup-time sync, not an iteration cost)."""
        b = getattr(self, "relaxed_bounds", None)
        if b is not None:
            return int(b["recv_cap"])
        if getattr(self, "_recv_exact", None) is None:
            ov = self.plan.phys_override
            assert ov is not None, (
                "fast tail needs relaxed_bounds (loccap_sl) or "
                "plan.phys_override (loccap_gpu)")
            serve = ov.reshape(-1).to(self.device) // self.cfg.nlp
            self._recv_exact = int((serve == self.rank).sum())
        return self._recv_exact

    def _derive_fast_graphed(self, phys_all, kstats=None) -> EpicIterPlan:
        """CUDA-graph wrapper around the fast tail: the device program is
        captured once (shapes are static by construction) and replayed;
        inputs are copied into static buffers, the blob D2H goes through
        a pinned staging copy after replay (the one host sync). Falls
        back to the eager fast tail if capture fails.
        FLUX_PLL_TAIL_GRAPH=1 enables."""
        if getattr(self, "_tail_graph_broken", False):
            return self._derive_from_phys_fast(phys_all, kstats)
        if getattr(self, "_tail_graph", None) is None:
            try:
                E = self.cfg.R * self.cfg.S * self.cfg.K
                self._g_phys_in = torch.empty(
                    self.cfg.R, self.cfg.S * self.cfg.K, dtype=torch.int64,
                    device=self.device)
                self._g_kstats_in = (torch.zeros(4, dtype=torch.int64,
                                                 device=self.device)
                                     if kstats is not None else None)
                self._g_phys_in.copy_(phys_all)
                if kstats is not None:
                    self._g_kstats_in.copy_(kstats.reshape(-1).long())
                for _ in range(2):  # warmup on the static buffers
                    self._fast_tail_device(self._g_phys_in,
                                           self._g_kstats_in)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._g_tail_out = self._fast_tail_device(
                        self._g_phys_in, self._g_kstats_in)
                torch.cuda.synchronize()
                self._tail_graph = g
                self._g_blob_pin = torch.empty_like(
                    self._g_tail_out["blob_dev"], device="cpu",
                    pin_memory=True)
            except Exception as e:  # noqa: BLE001 — eager fallback
                if self.rank == 0:
                    print(f"pll tail graph capture failed "
                          f"({type(e).__name__}: {e}); running eager",
                          flush=True)
                self._tail_graph_broken = True
                return self._derive_from_phys_fast(phys_all, kstats)
        else:
            self._g_phys_in.copy_(phys_all)
            if kstats is not None:
                self._g_kstats_in.copy_(kstats.reshape(-1).long())
        self._tail_graph.replay()
        # graph outputs live in the capture pool; downstream kernels read
        # the plan tensors on OTHER streams (comm / hcc side streams), so
        # handing pool memory across the boundary risks reuse races (seen
        # as CUTLASS internal errors / illegal access on the grouped
        # GEMM, data-timing dependent). Copy every consumed output into
        # persistent buffers on the current stream — a few ~us D2D
        # copies — and build the plan from those.
        if getattr(self, "_g_tail_persist", None) is None:
            self._g_tail_persist = {
                k: torch.empty_like(v)
                for k, v in self._g_tail_out.items()
                if torch.is_tensor(v)}
        for k, dst in self._g_tail_persist.items():
            dst.copy_(self._g_tail_out[k])
        outs = dict(self._g_tail_out)
        outs.update(self._g_tail_persist)
        self._g_blob_pin.copy_(self._g_tail_out["blob_dev"],
                               non_blocking=True)
        torch.cuda.synchronize()
        return self._fast_tail_host(outs, self._g_blob_pin,
                                    has_kstats=kstats is not None)

    def _fast_tail_g(self, tok_all, phys_all):
        """General fast tail for the D6 arms, m in {1, 2, 4} (8.22: EPIC
        rule-5 fairness — same sort-free/zero-D2H schedule class as the
        loccap tail, D6 routing untouched). Differences from the loccap
        tail: entry rows come from the reroute expansion in (expert,
        token) order (NOT topk order), so the per-group vce uses a
        token-occurrence scatter (one extra fixed-shape argsort); all
        canonical orders are GROUP-major ((grp, phys, tok) send wire,
        (grp, src, phys, tok) arrivals — group ranges are contiguous slot
        ranges, so every per-group quantity is a contiguous slice).
        Static-routing arms only (d6 loads are per-cell constants):
        capacity = the first-derive exact size. Returns a LIST of
        per-group EpicIterPlan (length m; m==1 callers take [0])."""
        # 8.23: _run_ordinal_fast — bit-identical on sorted keys, replaces
        # the torch.cummax spelling (~60x slower than the cub-scan class)
        from .ep_gpu_plan import comb_dst_slot_from_topk
        from .ep_gpu_plan import _run_ordinal_fast as _run_ordinal

        cfg = self.cfg
        R, G, S, K, nlp = cfg.R, cfg.G, cfg.S, cfg.K, cfg.nlp
        dev = self.device
        rank = self.rank
        m = self.m_groups
        metas = (self.groups_meta if self.groups_meta
                 else [dict(slot_lo=0, slot_hi=nlp, gpe=self.gpe,
                            kg=self.kg_frozen or K)])
        E = R * S * K
        assert S <= (1 << 13) and R * nlp <= (1 << 13) and R <= (1 << 7)
        grp_of = (self._grp_of_slot if self.groups_meta
                  else torch.zeros(nlp, dtype=torch.int64, device=dev))
        pads_g = (self._pad_vslot_g if self.groups_meta
                  else [self._pad_vslot])

        p_all = phys_all % nlp                              # [R, N] local slot
        g_all = grp_of[p_all]                               # entry group

        # ---- own row: (grp, phys, tok) send order ------------------------
        my_phys0 = phys_all[rank]
        my_tok0 = tok_all[rank]
        my_key = ((g_all[rank] << 26) | (my_phys0 << 13) | my_tok0)
        order_my = torch.argsort(my_key, stable=True)
        my_tok = my_tok0[order_my]
        my_phys = my_phys0[order_my]
        my_grp = g_all[rank][order_my]
        send_entry_logical = self.p2l[my_phys]
        comb_dst = (comb_dst_slot_from_topk(
            self.topk_all[rank], my_tok, send_entry_logical, G)
            if self.l01 else None)
        send_cnt_g = torch.zeros(m, dtype=torch.int64, device=dev)
        send_cnt_g.index_add_(0, my_grp, torch.ones_like(my_grp))
        in_gd = torch.zeros(m * R, dtype=torch.int64, device=dev)
        in_gd.index_add_(0, my_grp * R + my_phys // nlp,
                         torch.ones_like(my_grp))

        # ---- counts ------------------------------------------------------
        dest_all = phys_all // nlp
        mine = dest_all == rank
        src_ids = torch.arange(R, device=dev,
                               dtype=torch.int64).unsqueeze(1)
        out_gs = torch.zeros(m * R, dtype=torch.int64, device=dev)
        out_gs.index_add_(0, (g_all * R + src_ids)[mine].long()
                          if False else
                          torch.where(mine, g_all * R + src_ids,
                                      torch.zeros_like(g_all)).reshape(-1),
                          mine.reshape(-1).long())
        pair_flat = (src_ids * R + dest_all).reshape(-1)
        pair_rows = torch.zeros(R * R, dtype=torch.int64, device=dev)
        pair_rows.index_add_(0, pair_flat, torch.ones_like(pair_flat))
        slot_loads = torch.zeros(cfg.P, dtype=torch.int64, device=dev)
        slot_loads.index_add_(0, phys_all.reshape(-1),
                              torch.ones_like(pair_flat))
        # per-(grp, src) source entry counts (pad accounting)
        srccnt_gs = torch.zeros(m * R, dtype=torch.int64, device=dev)
        srccnt_gs.index_add_(0, (g_all * R + src_ids).reshape(-1),
                             torch.ones_like(pair_flat))

        # ---- arrivals: (grp, src, phys, tok) fixed-shape selection -------
        if getattr(self, "_recv_exact_g", None) is None:
            self._recv_exact_g = int(mine.sum())            # setup sync
        recv_cap = self._recv_exact_g
        key = (((~mine).long() << 36) | (g_all << 33)
               | (src_ids << 26) | (phys_all << 13)
               | tok_all).reshape(-1)
        arr = torch.argsort(key, stable=True)[:recv_cap]
        valid = mine.reshape(-1)[arr]
        slot_p = torch.where(valid, p_all.reshape(-1)[arr],
                             torch.full_like(arr, nlp))
        n_recv_g = torch.zeros(m, dtype=torch.int64, device=dev)
        n_recv_g.index_add_(0,
                            torch.where(valid, g_all.reshape(-1)[arr],
                                        torch.zeros_like(arr)),
                            valid.long())
        seg_ext = torch.zeros(nlp + 1, dtype=torch.int64, device=dev)
        seg_ext.index_add_(0, slot_p, torch.ones_like(slot_p))
        seg_rows = seg_ext[:nlp]
        seg_start = torch.zeros(nlp, dtype=torch.int64, device=dev)
        seg_start[1:] = torch.cumsum(seg_rows, dim=0)[:-1]
        o2 = torch.argsort(slot_p, stable=True)
        occ2 = _run_ordinal(slot_p[o2])
        seg_start_ext = torch.cat([seg_start, seg_start.new_zeros(1)])
        place_pad = torch.empty(recv_cap, dtype=torch.int64, device=dev)
        place_pad[o2] = seg_start_ext[slot_p[o2]] + occ2

        # ---- per-group vce: token-occurrence scatter ---------------------
        ntokens = R * S
        gtok = (src_ids * S + tok_all).reshape(-1)          # [E]
        key2 = (g_all.reshape(-1) << 38) | (gtok << 21) | \
            torch.arange(E, device=dev, dtype=torch.int64)
        o3 = torch.argsort(key2, stable=True)
        runk = (g_all.reshape(-1)[o3] * ntokens + gtok[o3])
        occ3 = torch.empty(E, dtype=torch.int64, device=dev)
        occ3[o3] = _run_ordinal(runk)
        vces = []
        overflow_t = torch.zeros((), dtype=torch.int64, device=dev)
        for gi, gm in enumerate(metas):
            kg = gm["kg"]
            gpe_g = gm["gpe"]
            in_g = g_all.reshape(-1) == gi
            ok = in_g & (occ3 < kg)
            overflow_t = overflow_t + (in_g & (occ3 >= kg)).long().sum()
            vslot = dest_all.reshape(-1) * gpe_g + p_all.reshape(-1)
            vflat = torch.empty(ntokens * kg + 1, dtype=torch.int64,
                                device=dev)
            vflat[:ntokens * kg] = pads_g[gi].repeat_interleave(kg) \
                if False else pads_g[gi].unsqueeze(1).expand(
                    ntokens, kg).reshape(-1)
            idx = torch.where(ok, gtok * kg + occ3,
                              torch.full_like(gtok, ntokens * kg))
            vflat.scatter_(0, idx, torch.where(ok, vslot,
                                               torch.zeros_like(vslot)))
            vces.append(vflat[:ntokens * kg].view(ntokens, kg)
                        .to(torch.int32).contiguous())

        # ---- ONE batched D2H --------------------------------------------
        d2h = [seg_rows, seg_start, in_gd, out_gs,
               pair_rows.max().reshape(1), n_recv_g, send_cnt_g,
               srccnt_gs, overflow_t.reshape(1)]
        blob = torch.cat([t.reshape(-1) for t in d2h]).cpu()
        off = 0

        def take(n):
            nonlocal off
            out = blob[off:off + n]
            off += n
            return out

        seg_rows_h = take(nlp).tolist()
        seg_start_h = take(nlp).tolist()
        in_gd_h = take(m * R).tolist()
        out_gs_h = take(m * R).tolist()
        max_pair_rows = int(take(1))
        n_recv_h = take(m).tolist()
        send_cnt_h = take(m).tolist()
        srccnt_h = take(m * R).tolist()
        assert int(take(1)) == 0, (
            "d6 fast tail: per-token group entries exceed the frozen K_g "
            "— bundle sizing drift (never noise)")

        ips = []
        s_off = 0
        r_off = 0
        for gi, gm in enumerate(metas):
            lo, hi, kg = gm["slot_lo"], gm["slot_hi"], gm["kg"]
            segments = []
            for p in range(lo, hi):
                rows = seg_rows_h[p]
                if rows == 0:
                    continue
                logical = int(self._p2l_host[rank * nlp + p])
                assert logical >= 0
                start = seg_start_h[p]
                segments.append((p, start, start + rows, logical))
            n_s = send_cnt_h[gi]
            n_r = n_recv_h[gi]
            pad_mine = kg * S - srccnt_h[gi * R + rank]
            ips.append(EpicIterPlan(
                send_row_index=my_tok[s_off:s_off + n_s],
                send_entry_logical=send_entry_logical[s_off:s_off + n_s],
                place_slots=place_pad[r_off:r_off + n_r],
                comb_dst_slot=(comb_dst[s_off:s_off + n_s]
                               if comb_dst is not None else None),
                in_splits=in_gd[gi * R:(gi + 1) * R].to(torch.int32),
                out_splits=out_gs[gi * R:(gi + 1) * R].to(torch.int32),
                send_counts=in_gd_h[gi * R:(gi + 1) * R],
                recv_counts=out_gs_h[gi * R:(gi + 1) * R],
                seg_rows=seg_rows_h[lo:hi],
                seg_start=seg_start_h[lo:hi],
                gemm_segments=segments,
                n_recv=n_r,
                max_pair_rows=max_pair_rows,
                slot_loads=slot_loads,
                hc_splits=None, hc_scatter=None, hc_sps_cpu=None,
                hc_uc_cpu=None, hc_pad_mine=pad_mine, hc_kg_iter=kg,
                hc_m_this=0,
                hc_vce=(vces[gi] if self.hc else None),
            ))
            s_off += n_s
            r_off += n_r
        return ips

    def _derive_from_phys_fast(self, phys_all, kstats=None) -> EpicIterPlan:
        """Sort-free, sync-free plan tail for the loccap routers
        (per-token entry count == K, m == 1). Replaces _derive_from_phys's
        full [R, S*K] canonical sort + ragged boolean gather (a mid-phase
        D2H sync) with:
          - one [S*K] sort of OWN row (the send-side wire order),
          - one [E] radix argsort keyed (not-mine, src, phys, tok) whose
            first recv_cap positions are my arrivals in canonical order
            (fixed-shape capacity padding instead of compaction),
          - vce built DIRECTLY from phys_all in topk column order — each
            token has exactly K entries, and within a virtual slot the
            wire/scatter order depends only on (vslot, global token), so
            the column order is contract-free: no gts sort, no occ,
            kg_iter == K,
          - counts via O(E) index_adds.
        The ONLY host readback is the single batched blob at the end
        (n_recv rides it); the boundary to the dispatch op stays device-
        resident (vce int32 + splits + place_slots). Env kill-switch:
        FLUX_PLL_FAST_TAIL=0 restores the legacy tail;
        FLUX_PLL_TAIL_GRAPH=1 CUDA-graphs the device half."""
        kd = kstats.reshape(-1).long() if kstats is not None else None
        outs = self._fast_tail_device(phys_all, kd)
        blob = outs["blob_dev"].cpu()
        return self._fast_tail_host(outs, blob,
                                    has_kstats=kstats is not None)

    def _fast_tail_device(self, phys_all, kstats_dev=None):
        """The shape-static device program of the fast tail (graph-
        capturable; no host syncs, no data-dependent shapes)."""
        cfg = self.cfg
        R, G, S, K, nlp = cfg.R, cfg.G, cfg.S, cfg.K, cfg.nlp
        dev = self.device
        rank = self.rank
        assert S <= (1 << 13) and R * nlp <= (1 << 13) and R <= (1 << 7), (
            "fast-tail sort key packing bounds exceeded")
        # 8.23: _run_ordinal_fast — bit-identical on sorted keys, replaces
        # the torch.cummax spelling (~60x slower than the cub-scan class)
        from .ep_gpu_plan import comb_dst_slot_from_topk
        from .ep_gpu_plan import _run_ordinal_fast as _run_ordinal

        # ---- own row: send-side canonical (phys, tok) order -------------
        tok_l = (torch.arange(S, device=dev, dtype=torch.int64)
                 .repeat_interleave(K))
        my_row = phys_all[rank]
        order_my = torch.argsort(my_row * (S + 1) + tok_l, stable=True)
        my_tok = tok_l[order_my]
        my_phys = my_row[order_my]
        send_entry_logical = self.p2l[my_phys]
        comb_dst = (comb_dst_slot_from_topk(
            self.topk_all[rank], my_tok, send_entry_logical, G)
            if self.l01 else None)
        # index_add instead of bincount throughout: torch.bincount can
        # issue a device sync to size its output — hidden latency eager,
        # capture-illegal under CUDA graphs
        in_splits = torch.zeros(R, dtype=torch.int64, device=dev)
        in_splits.index_add_(0, my_phys // nlp,
                             torch.ones_like(my_phys))
        in_splits = in_splits.to(torch.int32)

        # ---- counts (O(E), no sort) --------------------------------------
        dest_all = phys_all // nlp                          # [R, S*K]
        mine = dest_all == rank
        out_splits = mine.sum(dim=1).to(torch.int32)
        src_ids = torch.arange(R, device=dev,
                               dtype=torch.int64).unsqueeze(1)
        pair_flat = (src_ids * R + dest_all).reshape(-1)
        pair_rows = torch.zeros(R * R, dtype=torch.int64, device=dev)
        pair_rows.index_add_(0, pair_flat, torch.ones_like(pair_flat))
        n_recv_dev = mine.sum()
        slot_loads = torch.zeros(cfg.P, dtype=torch.int64, device=dev)
        slot_loads.index_add_(0, phys_all.reshape(-1),
                              torch.ones_like(pair_flat))

        # ---- arrivals: fixed-shape canonical selection -------------------
        recv_cap = self._recv_pad_capacity()
        key = (((~mine).long() << 34)
               | (src_ids.expand_as(phys_all) << 26)
               | (phys_all << 13)
               | tok_l.unsqueeze(0)).reshape(-1)
        arr = torch.argsort(key, stable=True)[:recv_cap]    # fixed slice
        valid = mine.reshape(-1)[arr]
        slot_p = torch.where(valid, phys_all.reshape(-1)[arr] - rank * nlp,
                             torch.full_like(arr, nlp))    # dummy bin nlp
        seg_ext = torch.zeros(nlp + 1, dtype=torch.int64, device=dev)
        seg_ext.index_add_(0, slot_p, torch.ones_like(slot_p))
        seg_rows = seg_ext[:nlp]
        seg_start = torch.zeros(nlp, dtype=torch.int64, device=dev)
        seg_start[1:] = torch.cumsum(seg_rows, dim=0)[:-1]
        o2 = torch.argsort(slot_p, stable=True)
        occ2 = _run_ordinal(slot_p[o2])
        seg_start_ext = torch.cat([seg_start, seg_start.new_zeros(1)])
        place_pad = torch.empty(recv_cap, dtype=torch.int64, device=dev)
        place_pad[o2] = seg_start_ext[slot_p[o2]] + occ2

        # ---- vce: topk column order, zero sorts --------------------------
        vce_i32 = None
        pad_mine = 0
        if self.hc:
            gpe = self.gpe
            kg = self.kg_frozen
            assert K <= kg, (K, kg)
            ntokens = R * S
            vce = ((dest_all * gpe) + (phys_all % nlp)).view(ntokens, K)
            if kg > K:
                pad_cols = (self._pad_vslot.unsqueeze(1)
                            .expand(ntokens, kg - K))
                vce = torch.cat([vce, pad_cols], dim=1)
            pad_mine = (kg - K) * S
            vce_i32 = vce.to(torch.int32).contiguous()

        # ---- the ONE batched D2H payload ---------------------------------
        d2h = [seg_rows, seg_start, in_splits.long(), out_splits.long(),
               pair_rows.max().reshape(1), n_recv_dev.reshape(1)]
        if kstats_dev is not None:
            d2h.append(kstats_dev)
        blob_dev = torch.cat([t.reshape(-1) for t in d2h])
        return dict(my_tok=my_tok, send_entry_logical=send_entry_logical,
                    comb_dst=comb_dst, in_splits=in_splits,
                    out_splits=out_splits, place_pad=place_pad,
                    slot_loads=slot_loads, vce=vce_i32,
                    pad_mine=pad_mine, blob_dev=blob_dev)

    def _fast_tail_host(self, outs, blob, has_kstats):
        """Host half: unpack the blob, build the segment list, construct
        the EpicIterPlan (tensor fields reference the device outputs)."""
        cfg = self.cfg
        R, K, nlp = cfg.R, cfg.K, cfg.nlp
        rank = self.rank
        off = 0

        def take(n):
            nonlocal off
            out = blob[off:off + n]
            off += n
            return out

        seg_rows_h = take(nlp).tolist()
        seg_start_h = take(nlp).tolist()
        send_counts = take(R).tolist()
        recv_counts = take(R).tolist()
        max_pair_rows = int(take(1))
        n_recv_h = int(take(1))
        if has_kstats:
            kstats_h = take(4).tolist()
            assert kstats_h[2] == 0, (
                f"loccap_sl forced budget exhausted ({kstats_h[2]} entries "
                "over the per-(src,dst) f_cap) — sizing-contract breach; "
                "raise --pll_f_cap")
            self.last_kernel_stats = kstats_h
        assert n_recv_h <= outs["place_pad"].numel(), (
            n_recv_h, outs["place_pad"].numel())

        segments = []
        base = rank * nlp
        for p in range(nlp):
            rows = seg_rows_h[p]
            if rows == 0:
                continue
            logical = int(self._p2l_host[base + p])
            assert logical >= 0, f"rank {rank}: rows in unused slot {p}"
            start = seg_start_h[p]
            segments.append((p, start, start + rows, logical))

        hc_kw = dict(hc_splits=None, hc_scatter=None, hc_sps_cpu=None,
                     hc_uc_cpu=None, hc_pad_mine=0, hc_kg_iter=0,
                     hc_m_this=0, hc_vce=None)
        if self.hc:
            hc_kw = dict(hc_splits=None, hc_scatter=None, hc_sps_cpu=None,
                         hc_uc_cpu=None, hc_pad_mine=outs["pad_mine"],
                         hc_kg_iter=cfg.K, hc_m_this=0,
                         hc_vce=outs["vce"])

        return EpicIterPlan(
            send_row_index=outs["my_tok"],
            send_entry_logical=outs["send_entry_logical"],
            place_slots=outs["place_pad"][:n_recv_h],
            comb_dst_slot=outs["comb_dst"],
            in_splits=outs["in_splits"],
            out_splits=outs["out_splits"],
            send_counts=send_counts,
            recv_counts=recv_counts,
            seg_rows=seg_rows_h,
            seg_start=seg_start_h,
            gemm_segments=segments,
            n_recv=n_recv_h,
            max_pair_rows=max_pair_rows,
            slot_loads=outs["slot_loads"],
            hcc_routing=None, hcc_pack=None, hcc_red=None,
            hcc_uc=None, hcc_wire=None, hcc_redcsr=None,
            **hc_kw,
        )

    def _derive_from_phys(self, tok_all, phys_all, rqp,
                          kstats=None) -> EpicIterPlan:
        from .ep_gpu_plan import (
            comb_dst_slot_from_topk,
            direct_layout_entries,
            _run_ordinal,
        )

        cfg = self.cfg
        R, G, S, K, nlp = cfg.R, cfg.G, cfg.S, cfg.K, cfg.nlp
        dev = self.device
        order = torch.argsort(phys_all * (S + 1) + tok_all, dim=1,
                              stable=True)
        ent_tok = torch.gather(tok_all, 1, order)
        ent_phys = torch.gather(phys_all, 1, order)

        lay = direct_layout_entries(ent_tok, ent_phys, self.rank, nlp, R)
        my_tok = lay["my_tok"]
        send_entry_logical = self.p2l[lay["my_phys"]]
        comb_dst = (
            comb_dst_slot_from_topk(self.topk_all[self.rank], my_tok,
                                    send_entry_logical, G)
            if self.l01 else None
        )
        slot_loads = (torch.bincount(phys_all.reshape(-1), minlength=cfg.P)
                      if rqp is None
                      else slot_loads_from_rqp(rqp, self.l2p, self.lcnts,
                                               cfg.P))

        d2h = [lay["seg_rows"], lay["seg_start"], lay["in_splits"].long(),
               lay["out_splits"].long(), lay["pair_max"].reshape(1)]

        hc_state = {}
        if self.hc:
            gpe, E_virt = self.gpe, self.E_virt
            ntokens = R * S
            kg = self.kg_frozen
            gts = (torch.arange(R, device=dev,
                                dtype=torch.int64).unsqueeze(1) * S
                   + ent_tok).reshape(-1)
            vsl = ((ent_phys // nlp) * gpe + ent_phys % nlp).reshape(-1)
            o2 = torch.argsort(gts, stable=True)
            gts_s, vsl_s = gts[o2], vsl[o2]
            occ = _run_ordinal(gts_s)
            kg_iter = occ.max() + 1              # device; asserted via D2H
            vce = torch.full((ntokens, kg), -1, dtype=torch.int64,
                             device=dev)
            vce[gts_s, occ.clamp(max=kg - 1)] = vsl_s
            pad_mask = vce < 0
            pad_rows = torch.zeros(R, dtype=torch.int64,
                                   device=dev).index_add_(
                0, self._home_of_token, pad_mask.sum(1))
            vce = torch.where(pad_mask,
                              self._pad_vslot.unsqueeze(1).expand_as(vce),
                              vce)
            if self.inwindow_meta:
                # v2b: the op derives splits/scatter/sps/uc in-window
                # (derive_routed_meta) — the planner ships ONLY the
                # virtual routing + pad accounting.
                d2h += [pad_rows, kg_iter.reshape(1)]
                hc_state = dict(vce=vce.to(torch.int32).contiguous())
            else:
                vce_flat = vce.reshape(-1)
                scatter_index = (vce_flat.argsort(stable=True).argsort()
                                 .int().view(ntokens, kg))
                splits = torch.bincount(vce_flat, minlength=E_virt).int()
                src_of_copy = self._home_of_token.repeat_interleave(kg)
                sps = torch.bincount(src_of_copy * E_virt + vce_flat,
                                     minlength=R * E_virt).view(R, E_virt)
                owner = vce // gpe
                flags = torch.zeros(ntokens, R, dtype=torch.bool,
                                    device=dev)
                flags.scatter_(1, owner, True)
                u_mat = flags.view(R, S, R).sum(1)
                U_mat = (flags.view(ntokens, self.nn, self.L).any(dim=2)
                         .view(R, S, self.nn).sum(1))
                uc = torch.cat([u_mat, U_mat], dim=1)
                m_per_rank = splits.long().view(R, gpe).sum(1)
                d2h += [sps.reshape(-1), uc.reshape(-1), pad_rows,
                        m_per_rank, kg_iter.reshape(1)]
                hc_state = dict(vce=vce, scatter_index=scatter_index,
                                splits=splits)

        hcc_state = {}
        if self.hcc:
            from .a2av_combine_indices import (
                build_a2av_combine_indices_dev,
                build_a2av_compress_indices_dev,
                build_a2av_unique_counts_dev,
            )
            routing = hc_state["scatter_index"].flatten()
            pack_idx, red_idx = build_a2av_combine_indices_dev(
                routing, hc_state["splits"], self.rank, R, self.kg_frozen)
            hcc_state = dict(routing=routing.int(), pack=pack_idx,
                             red=red_idx, uc=None, wire=None, redcsr=None)
            if self.nn > 1:
                uc_t = build_a2av_unique_counts_dev(
                    hc_state["vce"], R, self.nn, self.gpe)
                wp, wc, rp, rr = build_a2av_compress_indices_dev(
                    routing, hc_state["splits"], uc_t, self.rank, R,
                    self.nn, self.kg_frozen)
                hcc_state.update(uc=uc_t.cpu(), wire=[wp, wc],
                                 redcsr=[rp, rr])

        if kstats is not None:
            d2h.append(kstats.reshape(-1))  # kernel stats ride the one D2H
        # the batched D2H of the phase
        blob = torch.cat([t.reshape(-1) for t in d2h]).cpu()
        off = 0

        def take(n):
            nonlocal off
            out = blob[off:off + n]
            off += n
            return out

        seg_rows_h = take(nlp).tolist()
        seg_start_h = take(nlp).tolist()
        send_counts = take(R).tolist()
        recv_counts = take(R).tolist()
        max_pair_rows = int(take(1))
        hc_kw = dict(hc_splits=None, hc_scatter=None, hc_sps_cpu=None,
                     hc_uc_cpu=None, hc_pad_mine=0, hc_kg_iter=0,
                     hc_m_this=0, hc_vce=None)
        if self.hc and self.inwindow_meta:
            pad_rows_h = take(R)
            kg_iter_h = int(take(1))
            assert kg_iter_h <= self.kg_frozen, (
                f"iteration K_g {kg_iter_h} exceeds the frozen op topk "
                f"{self.kg_frozen} — hc arms cannot absorb this routing")
            hc_kw = dict(
                hc_splits=None, hc_scatter=None, hc_sps_cpu=None,
                hc_uc_cpu=None,
                hc_pad_mine=int(pad_rows_h[self.rank]),
                hc_kg_iter=kg_iter_h, hc_m_this=0,
                hc_vce=hc_state["vce"],
            )
        elif self.hc:
            sps_cpu = (take(R * self.E_virt).view(R, self.E_virt)
                       .int().contiguous())
            uc_cpu = (take(R * (R + self.nn)).view(R, R + self.nn)
                      .int().contiguous())
            pad_rows_h = take(R)
            m_per_rank_h = take(R)
            kg_iter_h = int(take(1))
            assert kg_iter_h <= self.kg_frozen, (
                f"iteration K_g {kg_iter_h} exceeds the frozen op topk "
                f"{self.kg_frozen} — hc arms cannot absorb this routing")
            hc_kw = dict(
                hc_splits=hc_state["splits"],
                hc_scatter=hc_state["scatter_index"],
                hc_sps_cpu=sps_cpu, hc_uc_cpu=uc_cpu,
                hc_pad_mine=int(pad_rows_h[self.rank]),
                hc_kg_iter=kg_iter_h,
                hc_m_this=int(m_per_rank_h[self.rank]),
                hc_vce=None,
            )

        if kstats is not None:
            kstats_h = take(4).tolist()
            assert kstats_h[2] == 0, (
                f"loccap_sl forced budget exhausted ({kstats_h[2]} entries "
                "over the per-(src,dst) f_cap) — sizing-contract breach; "
                "raise --pll_f_cap")
            self.last_kernel_stats = kstats_h

        segments = []
        base = self.rank * nlp
        for p in range(nlp):
            rows = seg_rows_h[p]
            if rows == 0:
                continue
            logical = int(self._p2l_host[base + p])
            assert logical >= 0, f"rank {self.rank}: rows in unused slot {p}"
            start = seg_start_h[p]
            segments.append((p, start, start + rows, logical))

        return EpicIterPlan(
            send_row_index=my_tok,
            send_entry_logical=send_entry_logical,
            place_slots=lay["place_slots"],
            comb_dst_slot=comb_dst,
            in_splits=lay["in_splits"],
            out_splits=lay["out_splits"],
            send_counts=send_counts,
            recv_counts=recv_counts,
            seg_rows=seg_rows_h,
            seg_start=seg_start_h,
            gemm_segments=segments,
            n_recv=int(lay["all_local"].numel()),
            max_pair_rows=max_pair_rows,
            slot_loads=slot_loads,
            hcc_routing=hcc_state.get("routing"),
            hcc_pack=hcc_state.get("pack"),
            hcc_red=hcc_state.get("red"),
            hcc_uc=hcc_state.get("uc"),
            hcc_wire=hcc_state.get("wire"),
            hcc_redcsr=hcc_state.get("redcsr"),
            **hc_kw,
        )

    def check_against(self, ip, runner, g: int = 0) -> None:
        """Loud setup-time drift guard vs the CPU reference state. Accepts
        the m=1 single plan or the multi-group list (per-group compare
        against the SETUP python layouts — routing is static, so equality
        is exact; vce compares as per-token multiset under the fast
        tail)."""
        if isinstance(ip, (list, tuple)):
            for gi, ipg in enumerate(ip):
                self._check_group_against(ipg, runner, gi)
            return
        grp = runner.elay.groups[0]
        assert torch.equal(ip.send_row_index.cpu(), grp.send_row_index)
        assert torch.equal(ip.send_entry_logical.cpu(),
                           grp.send_entry_logical)
        assert torch.equal(ip.place_slots.cpu(), grp.place_slots)
        assert ip.send_counts == grp.send_counts, "send splits drift"
        assert ip.recv_counts == grp.recv_counts, "recv splits drift"
        assert ip.seg_rows == runner.elay.seg_rows, "seg_rows drift"
        assert ip.seg_start == runner.elay.seg_start, "seg_start drift"
        assert ip.gemm_segments == runner.elay.gemm_segments
        assert ip.max_pair_rows == runner.elay.max_pair_rows
        if self.router in ("loccap_gpu", "loccap_sl"):
            # loccap_sl: check_against runs on derive_reference()'s ip
            # (deterministic side of the relaxed contract)
            ov = self.plan.phys_override
            ref_loads = torch.bincount(ov.long().reshape(-1),
                                       minlength=self.cfg.P)
            assert torch.equal(ip.slot_loads.cpu(), ref_loads), (
                f"{self.router} slot loads drift vs the setup CPU routing"
                " — cross-device determinism bug")
        else:
            assert torch.equal(ip.slot_loads.cpu(),
                               slot_batch_loads(self.plan))
        if self.l01:
            assert torch.equal(ip.comb_dst_slot.cpu(), grp.comb_dst_slot)
        if self.hc and self.inwindow_meta:
            b = runner._hc_bundles[0]
            if int(os.getenv("FLUX_PLL_FAST_TAIL", "1")):
                # fast tail (all rule-5 routers incl. d6) ships vce in a
                # virtual slot the wire order depends only on
                # (vslot, token), so the guard compares the per-token
                # MULTISET (column-order-free routing identity)
                a = ip.hc_vce.long().cpu().sort(dim=1).values
                bb = b.virtual_choosed.long().cpu().sort(dim=1).values
                assert torch.equal(a, bb), (
                    "in-window vce drift vs the setup bundle (multiset)")
            else:
                assert torch.equal(ip.hc_vce.cpu(), b.virtual_choosed), (
                    "in-window vce drift vs the setup bundle")
            return
        if self.hc:
            b = runner._hc_bundles[0]
            assert ip.hc_kg_iter == b.K_g or self.kg_frozen == b.K_g
            assert torch.equal(ip.hc_splits.cpu(), b.meta.splits)
            assert torch.equal(ip.hc_scatter.cpu(), b.meta.scatter_index)
            assert torch.equal(ip.hc_sps_cpu, b.meta.splits_per_source)
            assert torch.equal(ip.hc_uc_cpu, b.meta.a2av_unique_counts)
            assert ip.hc_pad_mine == int(b.pad_rows_per_rank[self.rank])
            assert ip.hc_m_this == int(b.meta.m_per_rank[self.rank])
        if self.hcc:
            e = runner._hcc[0]
            assert torch.equal(ip.hcc_pack.cpu(), e["pack"].cpu())
            assert torch.equal(ip.hcc_red.cpu(), e["red"].cpu())
            if self.nn > 1:
                assert torch.equal(ip.hcc_uc, e["uc"])
                for got, ref in zip(ip.hcc_wire + ip.hcc_redcsr,
                                    e["wire"] + e["redcsr"]):
                    assert torch.equal(got.cpu(), ref.cpu())

    def _check_group_against(self, ip: EpicIterPlan, runner, g: int):
        """Per-group setup drift guard for the multi-group rule-5 path."""
        grp = runner.elay.groups[g]
        assert torch.equal(ip.send_row_index.cpu(), grp.send_row_index), g
        assert torch.equal(ip.send_entry_logical.cpu(),
                           grp.send_entry_logical), g
        assert torch.equal(ip.place_slots.cpu(), grp.place_slots), g
        assert ip.send_counts == grp.send_counts, (g, "send splits drift")
        assert ip.recv_counts == grp.recv_counts, (g, "recv splits drift")
        assert ip.seg_rows == grp.seg_rows, (g, "seg_rows drift")
        lo, hi = grp.slot_lo, grp.slot_hi
        assert ip.seg_start == runner.elay.seg_start[lo:hi], g
        segs_ref = [s for s in runner.elay.gemm_segments
                    if lo <= s[0] < hi]
        assert ip.gemm_segments == segs_ref, (g, "gemm segments drift")
        if self.l01:
            assert torch.equal(ip.comb_dst_slot.cpu(), grp.comb_dst_slot), g
        if self.hc and self.inwindow_meta and ip.hc_vce is not None:
            b = runner._hc_bundles[g]
            a = ip.hc_vce.long().cpu().sort(dim=1).values
            bb = b.virtual_choosed.long().cpu().sort(dim=1).values
            assert torch.equal(a, bb), (
                g, "in-window vce drift vs the setup bundle (multiset)")

    def check_relaxed(self, ip: EpicIterPlan, bounds,
                      ref_incidence: int = None,
                      band: float = 0.05) -> dict:
        """Relaxed drift guard for the loccap_sl kernel arm (user ruling
        2026-08-21: invariants + bounds + incidence band replace bitwise
        identity). Audits the ASSEMBLED routing (every rank holds the full
        allgathered phys, so each rank verifies all ranks):
          1. conservation: every entry's slot maps back to its expert
          2. splits consistency: ip's in/out splits == phys bincounts
          3. sizing-bound compliance: per-rank recv <= recv_ub, per-pair
             rows <= pair_ub (elementwise) — the provable table bounds
          4. incidence within `band` of the setup reference (optional)
        Cheap (a few bincounts); run at setup always, per-iteration under
        FLUX_PLL_CHECK_ITERS=1. Returns audit facts."""
        cfg = self.cfg
        R, S, K, nlp = cfg.R, cfg.S, cfg.K, cfg.nlp
        phys = self._last_phys_all
        assert phys is not None
        assert bool(self.p2l[phys].eq(
            self.topk_all.view(R, S * K)).all()), (
            "loccap_sl conservation violated in the assembled routing")
        serve_rank = phys // nlp
        recv = torch.bincount(serve_rank.reshape(-1), minlength=R)
        pair = torch.bincount(
            (torch.arange(R, device=phys.device, dtype=torch.int64)
             .unsqueeze(1) * R + serve_rank).reshape(-1),
            minlength=R * R).view(R, R)
        in_mine = torch.bincount(serve_rank[self.rank], minlength=R)
        assert torch.equal(ip.in_splits.cpu().long(), in_mine.cpu()), (
            "in_splits != own-row bincount")
        assert sum(ip.recv_counts) == ip.n_recv
        assert int(recv[self.rank]) == ip.n_recv, (
            "assembled recv rows != ip.n_recv")
        ru = bounds["recv_ub"].to(recv.device)
        pu = bounds["pair_ub"].to(pair.device)
        assert bool((recv <= ru).all()), (
            "recv bound violated: "
            f"{(recv - ru).clamp(min=0).max().item()} rows over")
        assert bool((pair <= pu).all()), (
            "pair bound violated: "
            f"{(pair - pu).clamp(min=0).max().item()} rows over")
        facts = {"recv_max": int(recv.max()),
                 "pair_max": int(pair.max()),
                 "kernel_stats": self.last_kernel_stats}
        if ref_incidence is not None and ref_incidence > 0:
            node = serve_rank // self.L
            on = torch.zeros(R * S, self.nn, dtype=torch.bool,
                             device=phys.device)
            on.view(-1).scatter_(
                0, (torch.arange(R * S, device=phys.device,
                                 dtype=torch.int64)
                    .repeat_interleave(K) * self.nn
                    + node.reshape(-1)), True)
            home = (torch.arange(R * S, device=phys.device,
                                 dtype=torch.int64) // S) // self.L
            inc = int(on.sum()) - int(
                on.gather(1, home.unsqueeze(1)).sum())
            facts["incidence_remote"] = inc
            drift = abs(inc - ref_incidence) / ref_incidence
            assert drift <= band, (
                f"incidence {inc} outside the {band:.0%} band of the "
                f"reference {ref_incidence}")
        return facts


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
        if getattr(self, "hcc_enabled", False):
            # the combine entries (inbuf sizes, pack/red/compress indices)
            # are routing-dependent too — stale entries were the 2026-08-18
            # l01xhcxmig failure (inbuf mismatch / splits-vs-gemm-rows abort)
            self._rebuild_hc_combine()
        return (time.perf_counter() - t0) * 1e3

    # -- transports ---------------------------------------------------------

    def enable_nvshmem(self, local_world_size: int, num_comm_sm: int = 8,
                       split_headroom: float = 2.0,
                       max_split_floor: int = 0):
        """One All2AllSingle pair reused across all m group calls.

        max_split = headroom * current max per-(group, src->dst) pair rows
        (the op never validates per-call splits against max_split, so the
        runner asserts them itself on every layout rebuild — a hard failure
        beats silent staging overflow after migration reshapes the wire).
        max_split_floor: loccap_sl capacity mode — the PROVABLE per-pair
        bound (loccap_sl_bounds.pair_cap; replicated-deterministic, so the
        op's cross-rank max_split equality FLUX_CHECK holds)."""
        import flux  # GPU-side only

        self._epic_max_split = max(
            1, int(self.elay.max_pair_rows * split_headroom),
            int(max_split_floor))
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

    def reserve_recv_capacity(self, cap_rows: int):
        """loccap_sl capacity mode: grow the recv-side buffers to the
        PROVABLE recv bound (loccap_sl_bounds.recv_cap) so the relaxed
        kernel's per-iteration n_recv may vary underneath it. Call after
        the ctor and BEFORE enable_hier_compress/enable_grouped_gemm
        (bundles/backends snapshot buffer views). bind_iter_plan then
        tracks the exact per-iteration extent (recv_off/n_recv)."""
        cap_rows = int(cap_rows)
        self._recv_capacity = cap_rows
        if cap_rows > self.recv_buf.shape[0]:
            dev, dt = self.device, self.dtype
            H = self.cfg.H
            self.recv_buf = torch.empty(cap_rows, H, dtype=dt, device=dev)
            self.wrecv_buf = torch.empty(cap_rows, dtype=torch.float32,
                                         device=dev)
            self.hidden_buf = torch.zeros(cap_rows, H, dtype=dt, device=dev)
            self.weights_buf = torch.zeros(cap_rows, dtype=torch.float32,
                                           device=dev)
            self.out_buf = torch.zeros(cap_rows, self.ffn_size_shard,
                                       dtype=dt, device=dev)

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

    # -- per-iteration plan binding (SCHEMA rule 5, m=1 scope) --------------

    def _bind_iter_plan_multi(self, ips):
        """Multi-group bind (rule-5 EPIC m>1, 8.22): per-group plan fields
        into the per-group runner state; global elay assembled from the
        contiguous per-group slices. Static-routing arms — total recv must
        equal the setup sizing."""
        m = len(ips)
        assert m == self.m == len(self.elay.groups), (m, self.m)
        total_recv = sum(ip.n_recv for ip in ips)
        assert total_recv == self.n_recv, (total_recv, self.n_recv)
        new_groups = []
        recv_off = [0]
        seg_rows_all, seg_start_all, segments_all = [], [], []
        for g, ip in enumerate(ips):
            grp = self.elay.groups[g]
            self.g_send_row_index[g] = ip.send_row_index
            self.g_send_entry_logical[g] = ip.send_entry_logical
            self.g_place_slots[g] = ip.place_slots
            if self.layers == "l01":
                self.g_comb_dst[g] = ip.comb_dst_slot
                # comb_pack = group-relative positions (base = the
                # group's global segment offset; 0 at m=1)
                self.g_comb_pack[g] = ip.place_slots - ip.seg_start[0]
            new_groups.append(_dc_replace(
                grp, send_counts=ip.send_counts,
                recv_counts=ip.recv_counts, seg_rows=ip.seg_rows))
            recv_off.append(recv_off[-1] + ip.n_recv)
            seg_rows_all += ip.seg_rows
            seg_start_all += ip.seg_start
            segments_all += ip.gemm_segments
            self._group_splits_cpu[g] = torch.tensor(ip.seg_rows,
                                                     dtype=torch.int64)
            if getattr(self, "_grouped_splits", None):
                self._grouped_splits[g] = self._group_splits_cpu[g]
            if self.transport == "nvshmem":
                assert ip.max_pair_rows <= self._epic_max_split
                self._g_in_splits[g] = ip.in_splits
                self._g_out_splits[g] = ip.out_splits
        self.elay = _dc_replace(
            self.elay, groups=new_groups, seg_start=seg_start_all,
            seg_rows=seg_rows_all, gemm_segments=segments_all,
            recv_off=recv_off, n_recv=total_recv)
        if self.hc_enabled and ips[0].hc_vce is not None:
            self._iter_vce_g = [ip.hc_vce for ip in ips]
            self._iter_pad_g = [ip.hc_pad_mine for ip in ips]

    def bind_iter_plan(self, ip):
        """Swap the routing-derived index state for this iteration's
        EpicIterPlan (m=1). Called inside the timed `plan` bracket every
        iteration; on swap-applying iterations the driver re-derives after
        the (host) migration rebuild, so sizes here always match the
        current buffers — asserted, never silently grown.

        CAPACITY mode (loccap_sl, set via reserve_recv_capacity): the
        relaxed kernel's realized n_recv varies per iteration under the
        provable bound; buffers are capacity, ip carries the exact counts,
        and GEMM/scatter touch only [0, n_recv) (segment lists come from
        ip). Without capacity mode the historical exact-match assert
        stands (d6 / loccap_gpu arms — no behavior change)."""
        if isinstance(ip, (list, tuple)):
            return self._bind_iter_plan_multi(ip)
        assert self.m == 1, "per-iteration planning is scoped to m=1 arms"
        if getattr(self, "_recv_capacity", None) is not None:
            assert ip.n_recv <= self._recv_capacity, (
                f"iteration recv rows {ip.n_recv} exceed the reserved "
                f"capacity {self._recv_capacity} — provable-bound breach "
                "(sizing bug, never noise)")
            self.n_recv = ip.n_recv
        else:
            assert ip.n_recv == self.n_recv, (
                f"iteration recv rows {ip.n_recv} != current sizing "
                f"{self.n_recv} (derive must run AFTER any migration "
                "rebuild)"
            )
        dev = self.device
        self.g_send_row_index[0] = ip.send_row_index
        self.g_send_entry_logical[0] = ip.send_entry_logical
        self.g_place_slots[0] = ip.place_slots
        if self.layers == "l01":
            self.g_comb_dst[0] = ip.comb_dst_slot
            # m=1: seg_start[slot_lo] == 0, so comb_pack == place_slots
            self.g_comb_pack[0] = ip.place_slots
        grp = self.elay.groups[0]
        self.elay = _dc_replace(
            self.elay,
            groups=[_dc_replace(grp, send_counts=ip.send_counts,
                                recv_counts=ip.recv_counts,
                                seg_rows=ip.seg_rows)],
            seg_start=ip.seg_start, seg_rows=ip.seg_rows,
            gemm_segments=ip.gemm_segments,
            # capacity mode: track the iteration's EXACT recv extent —
            # dispatch/scatter/combine slice recv_buf by recv_off, and
            # leaving the stale value silently corrupts (plan Hole-1)
            recv_off=[0, ip.n_recv],
            n_recv=ip.n_recv,
        )
        self._group_splits_cpu[0] = torch.tensor(ip.seg_rows,
                                                 dtype=torch.int64)
        # 8.22 root-cause fix: enable_grouped_gemm captured a REFERENCE to
        # the setup splits tensor; rebinding _group_splits_cpu alone left
        # the grouped GEMM segmenting every iteration with SETUP sizes —
        # silently wrong under per-iteration routing variance (relaxed
        # loccap_sl), and an illegal access / CUTLASS internal error once
        # the deviation crosses the input slice (observed: qwen canon
        # oracle cells, rank 5). Keep the GEMM's splits in lockstep.
        if getattr(self, "_grouped_splits", None):
            self._grouped_splits[0] = self._group_splits_cpu[0]
        if self.transport == "nvshmem":
            assert ip.max_pair_rows <= self._epic_max_split, (
                f"pair rows {ip.max_pair_rows} exceed All2AllSingle "
                f"max_split {self._epic_max_split}; raise "
                f"--a2a_split_headroom")
            self._g_in_splits[0] = ip.in_splits
            self._g_out_splits[0] = ip.out_splits
        if self.hc_enabled and ip.hc_vce is not None:
            # v2b in-window mode: dispatch_group_hc derives the metadata
            # via the op (derive_routed_meta) from this vce each call.
            self._iter_vce = ip.hc_vce
            self._iter_pad_mine = ip.hc_pad_mine
            return
        if self.hc_enabled:
            self._hc_splits_gpu[0] = ip.hc_splits
            self._hc_scatter_gpu[0] = ip.hc_scatter
            # host-side op args + pad accounting for dispatch_group_hc
            self._iter_hc = dict(sps=ip.hc_sps_cpu, uc=ip.hc_uc_cpu,
                                 pad_mine=ip.hc_pad_mine)
        if getattr(self, "hcc_enabled", False):
            e = self._hcc[0]
            assert ip.hc_m_this == e["inbuf"].shape[0] or (
                ip.hc_m_this == 0 and e["inbuf"].shape[0] == 1), (
                f"iteration m_this {ip.hc_m_this} != combine inbuf rows "
                f"{e['inbuf'].shape[0]} (derive must follow the rebuild)")
            e.update(routing=ip.hcc_routing, pack=ip.hcc_pack,
                     red=ip.hcc_red, uc=ip.hcc_uc, wire=ip.hcc_wire,
                     redcsr=ip.hcc_redcsr, m_this=ip.hc_m_this)

    def enable_hier_compress(self, tp_env, local_world_size: int,
                             headroom: float = 1.5,
                             relay: str = "identity",
                             inkernel_swap: bool = False,
                             wire: str = "relay_identity",
                             cap_floors: dict = None,
                             fixed_kg: list = None):
        """EPIC Mode-2 dispatch transport: per-group GemmGroupedV2AGScatterOp
        instances (a2av_hier_compress) driven through dispatch_only over the
        virtual physical-slot expert space. relay='identity' is the faithful
        PXN shape (inter-node to the same-index GPU, NVLink forward =
        FLUX_A2AV_RELAY_IDENTITY); 'balanced' is our chunked-relay ablation.
        Requires enable_nvshmem() first (the per-entry probs side-wire stays
        on All2AllSingle — the fused op moves token rows only). Capacity env
        knobs are process-global ctor-reads: set per instance, in
        SPMD-identical group order, BEFORE each ctor.

        inkernel_swap=True (--migration inkernel): the GROUP-0 op is built
        with FLUX_A2AV_INKERNEL_SWAP = one expert's fc1(+fc2) bytes, so its
        dispatch_only can run EPIC §4.3's swap as the fused phase 0
        (symmetric scratch + flag are ctor-allocated, collectively)."""
        import os

        import flux

        assert self.transport == "nvshmem", (
            "enable_hier_compress requires enable_nvshmem() first")
        assert relay in ("identity", "balanced")
        assert wire in ("relay_identity", "lb_union")
        if wire == "lb_union":
            # Tier-B fused wire over the replicated virtual slot space —
            # valid because Tier-B gating is pure expert-id arithmetic on
            # any rank-blocked uniform-gpe space (dst_node = e // (E*Lb)).
            # RELAY_IDENTITY and LB_UNION are ctor-mutually-exclusive.
            # Pin the LB_UNION-conditioned defaults explicitly (setdefault:
            # an arm's env pin wins) so conn-rung clones measure one axis.
            assert not inkernel_swap, (
                "inkernel_swap x lb_union wire is untested — use the "
                "relay_identity wire for migration arms")
            os.environ["FLUX_A2AV_RELAY_IDENTITY"] = "0"
            os.environ["FLUX_A2AV_LB_UNION"] = "1"
            os.environ.pop("FLUX_A2AV_FANOUT", None)  # closed loser
            os.environ.setdefault("FLUX_A2AV_EARLY_LAUNCH", "1")
            os.environ.setdefault("FLUX_A2AV_FUSED_STAGE2", "1")
        else:
            os.environ["FLUX_A2AV_RELAY_IDENTITY"] = (
                "1" if relay == "identity" else "0")
            os.environ.pop("FLUX_A2AV_LB_UNION", None)  # baseline: no union
        self._hc_wire = wire
        self._hc_relay = relay
        self._hc_L = local_world_size
        self._hc_headroom = headroom
        self._hc_tp_env = tp_env
        # loccap_sl capacity mode: fixed_kg pins the vce width (K_g == K
        # analytically at m=1 — every router emits exactly K entries per
        # token, so this equals the organic value; the pin makes the
        # invariant explicit under per-iteration routing variance).
        # cap_floors raises the FLUX_A2AV_MAX_* ctor knobs to the PROVABLE
        # table bounds so no iteration can overflow the frozen panels.
        self._hc_cap_floors = dict(cap_floors or {})
        self._hc_bundles = build_epic_hc_bundles(
            self.plan, self._topk_all, self.m, local_world_size,
            fixed_kg=fixed_kg)
        self._hc_kg = [b.K_g for b in self._hc_bundles]
        self._hc_ops = []
        self._hc_splits_gpu = []
        self._hc_scatter_gpu = []
        self._hc_caps = []
        self._canon_checked = {}  # per-group one-shot injectivity guard
        self._inkernel_swap = inkernel_swap
        self._swap_seq = 0        # GLOBAL swap-round sequence (replicated)
        self._pending_swap = None
        swap_bytes = 0
        if inkernel_swap:
            swap_bytes = (self.slot_fc1[0].numel()
                          * self.slot_fc1.element_size())
            if self.place_fc2:
                swap_bytes += (self.slot_fc2[0].numel()
                               * self.slot_fc2.element_size())
        ntokens = self.cfg.R * self.cfg.S
        for gi, b in enumerate(self._hc_bundles):
            if inkernel_swap and gi == 0:
                os.environ["FLUX_A2AV_INKERNEL_SWAP"] = str(swap_bytes)
            else:
                os.environ.pop("FLUX_A2AV_INKERNEL_SWAP", None)
            knobs = epic_hc_required_knobs(b, self.cfg.R, local_world_size)
            caps = {k: max(int(int(v) * headroom) + 1,
                           int(self._hc_cap_floors.get(k, 0)))
                    for k, v in knobs.items()}
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
        os.environ.pop("FLUX_A2AV_INKERNEL_SWAP", None)
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
        swap_kw = {}
        if g == 0 and getattr(self, "_pending_swap", None) is not None:
            # EPIC §4.3 fused phase 0: the exchange kernel runs at the head
            # of THIS dispatch launch sequence (weights complete, then the
            # token wire — stream-ordered, no host sync between).
            peer, slot, seq = self._pending_swap
            swap_kw = dict(
                swap_fc1=self.slot_fc1[slot],
                swap_fc2=(self.slot_fc2[slot] if self.place_fc2 else None),
                swap_peer=peer, swap_epoch=seq)
            self._pending_swap = None
        import os as _os
        nan_canary = _os.environ.get("FLUX_EPIC_CANON_NAN", "0") == "1"
        if nan_canary:
            # D1 discriminator: prefill the dense output with NaN so a recv
            # row the wire never wrote survives as NaN (wire hole), while a
            # canon-permutation hole shows as a non-NaN wrong row — the two
            # 8n hypotheses separate in one run (diagnostic only).
            m_exp = int(b.meta.splits[
                self.rank * b.gpe:(self.rank + 1) * b.gpe].sum())
            swap_kw["dense_out"] = torch.full(
                (m_exp, self.cfg.H), float("nan"), dtype=self.dtype,
                device=self.device)
        # per-iteration planning (rule 5). v2b in-window mode: the op
        # itself derives splits/scatter/sps/uc from the bound virtual
        # routing (derive_routed_meta — kernels + one pinned D2H, all
        # inside this dispatch bracket); the derived tensors also refresh
        # the canon tail's splits and the combine entry's routing.
        ivce_g = getattr(self, "_iter_vce_g", None)
        ivce = (ivce_g[g] if ivce_g is not None
                else getattr(self, "_iter_vce", None))
        if ivce is not None:
            sd, scd, sps_cpu, uc_cpu = (
                self._hc_ops[g].derive_routed_meta(ivce))
            self._hc_splits_gpu[g] = sd
            self._hc_scatter_gpu[g] = scd
            _padm = (self._iter_pad_g[g] if ivce_g is not None
                     else self._iter_pad_mine)
            self._iter_hc = dict(sps=sps_cpu, uc=uc_cpu, pad_mine=_padm)
            if getattr(self, "hcc_enabled", False):
                # The combine plan in-window (rule 5). Default since
                # 2026-08-22 (plan eager-juggling-glacier 2b): ONE C++ call
                # TopkReduceScatterOp.derive_combine_meta (sort-free
                # compress builders, tag FLUX_A2AV_RS_DERIVE_COMBINE_RS_TAG)
                # inside this dispatch bracket, so run() no longer self-
                # builds with the slow sort-based builders on the combine
                # path. FLUX_EPIC_HCC_DERIVE=0 restores the in-op self-build
                # (ablation / bisection only — a rule-4 boundary).
                # unique_counts = the COMBINE U [W, NN] — the last NN
                # columns of the dispatch-side [W, W+NN] concat.
                W = self.cfg.R
                nn = W // self._hc_L
                routing_flat = scd.reshape(-1).contiguous()
                uc_comb = (uc_cpu[:, W:].contiguous() if nn > 1 else None)
                pack = red = wire = redcsr = None
                if getattr(self, "_hcc_derive_fused", True):
                    outs = self._hcc[g]["op"].derive_combine_meta(
                        sd, routing_flat, sps_cpu, uc_comb)
                    pack, red = outs[0], outs[1]
                    if len(outs) > 2:
                        wire = [outs[2], outs[3]]
                        redcsr = [outs[4], outs[5]]
                self._hcc[g].update(
                    routing=routing_flat, pack=pack, red=red,
                    uc=uc_comb, wire=wire, redcsr=redcsr, sps=sps_cpu)
                # lifetime: these are allocated on the current (comm) stream
                # and read by kernels on self._hcc_stream; pin them to that
                # stream so the caching allocator never recycles them early
                for _t in (pack, red, *(wire or ()), *(redcsr or ())):
                    if torch.is_tensor(_t):
                        _t.record_stream(self._hcc_stream)
        ihc = getattr(self, "_iter_hc", None)
        sps = ihc["sps"] if ihc is not None else b.meta.splits_per_source
        ucs = ihc["uc"] if ihc is not None else b.meta.a2av_unique_counts
        pad_mine = (ihc["pad_mine"] if ihc is not None
                    else int(b.pad_rows_per_rank[self.rank]))
        dense, ssi, _ssc, m_ep = self._hc_ops[g].dispatch_only(
            self._last_inputs, self._hc_splits_gpu[g],
            self._hc_scatter_gpu[g],
            sps, ucs,
            **swap_kw,
        )
        m = int(m_ep)
        if nan_canary:
            bad = torch.isnan(dense[:m]).any(dim=1)
            print(f"[canon-nan] rank {self.rank} g{g}: "
                  f"{int(bad.sum())}/{m} recv rows never written by the "
                  "wire", flush=True)
        n_real = m - pad_mine
        assert n_real == n_rows, (
            f"group {g}: dense real rows {n_real} != layout rows {n_rows}")
        if n_rows:
            # dense rows arrive segment-major in the WIRE's within-segment
            # order; ssi[i] is the row's SEGMENT-RELATIVE canonical index
            # ("sorted-D row -> per-expert D row": ascending under the
            # relay wire, a real permutation under lb_union's window
            # order — verified empirically 2026-08-18). canonical position
            # = segment base + ssi restores the v1 slot-major (src, token)
            # layout that the probs pairing and GEMM segments assume, for
            # any wire; a no-op permutation on the relay path.
            sl = self._hc_splits_gpu[g][
                self.rank * b.gpe:(self.rank + 1) * b.gpe].long()
            bounds = torch.cumsum(sl, 0)
            assert int(bounds[-1]) == m, (int(bounds[-1]), m)
            seg = torch.bucketize(
                torch.arange(m, device=dense.device), bounds, right=True)
            canon_pos = (bounds - sl)[seg] + ssi[:m].long()
            if not self._canon_checked.get(g, False):
                # W32 lesson (NR-16 amendment): index_copy_ into empty_like
                # silently corrupts if canon_pos is not a permutation
                # (allocator garbage in uncovered rows — zeros on fresh
                # blocks, stale bytes on recycled ones). Routing is static
                # per cell, so ONE injectivity check per group per cell is
                # a complete guard at zero steady-state cost.
                assert bool(torch.equal(
                    canon_pos.sort().values,
                    torch.arange(m, device=canon_pos.device))), (
                    f"group {g}: canon_pos is not a permutation of [0,{m}) "
                    "— ssi contract violated (see NR-16 8n amendment)")
                self._canon_checked[g] = True
            canon = torch.empty_like(dense[:m])
            canon.index_copy_(0, canon_pos, dense[:m])
            self.hidden_buf[base:base + n_rows].copy_(canon[:n_real])
        s_lo, s_hi = self.elay.send_off[g], self.elay.send_off[g + 1]
        r_lo, r_hi = self.elay.recv_off[g], self.elay.recv_off[g + 1]
        self._a2a_probs.forward(
            self.wsend_buf[s_lo:s_hi].view(-1, 1),
            self.wrecv_buf[r_lo:r_hi].view(-1, 1),
            self._g_in_splits[g], self._g_out_splits[g],
            self._num_comm_sm,
        )

    def enable_hc_combine(self, n_split: int = 4, pack_blocks: int = 3,
                          m_capacity: int = None):
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
        # a2av combine tiles need (H // n_split) % 8 == 0; the K3 shape
        # preset's n_split_l1=7 gives n_per=512 (2026-08-22: n_split was
        # never threaded to the expert-movement drivers — hardcoded 4 ->
        # n_per=896 at H=3584, legal for a2av but off the 512-tile lane)
        assert (self.cfg.H // n_split) % 8 == 0 and \
            self.cfg.H % n_split == 0, (self.cfg.H, n_split)
        W = self.cfg.R
        L = self._hc_L
        nn = W // L
        self._hcc_nsplit = n_split
        self._hcc_pack_blocks = pack_blocks
        # combine CAPACITY mode (8.22.route: l01 x relaxed loccap_sl —
        # per-iteration routing variance): inbuf/scale allocated at the
        # PROVABLE recv bound (loccap_sl_bounds.recv_cap, the same bound
        # the dispatch-side reserve_recv_capacity uses); each iteration
        # the pack slices [0, n_rows). None = legacy exact-size (static-
        # routing arms unchanged). The combine METADATA is already
        # per-iteration in-window (derive_routed_meta +
        # derive_combine_meta); the op ctor m_full is already worst-case.
        self._hcc_m_capacity = m_capacity
        self._hcc_stream = torch.cuda.Stream(priority=-1)
        import os as _os
        self._hcc_stream_edges = _os.environ.get(
            "FLUX_EPIC_HCC_STREAM_EDGES", "1") == "1"
        self._hcc_group_barrier = flux.GroupBarrier(self.group, False)
        self._hcc = []
        self._hcc_knob_caps = []
        for b in self._hc_bundles:
            m_full = W * self.cfg.S * b.K_g
            demands = self._hcc_rs_demands(b)
            # migration-proof headroom (same discipline as the dispatch-side
            # ctor caps): post-migration demands are hard-asserted against
            # these frozen values in _rebuild_hc_combine
            caps = {k: int(v * self._hc_headroom) + 1
                    for k, v in demands.items()}
            self._hcc_knob_caps.append(caps)
            os.environ["FLUX_A2AV_RS_MAX_SEND_ROWS"] = str(caps["send"])
            os.environ["FLUX_A2AV_RS_MAX_STAGE_ROWS"] = str(caps["stage"])
            os.environ["FLUX_A2AV_RS_MAX_CONV_ROWS"] = str(caps["conv"])
            os.environ["FLUX_A2AV_RS_MAX_WIRE_ROWS"] = str(caps["wire"])
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
            entry = {"op": op, "barriers": barriers}
            entry["partial"] = torch.zeros(
                self.cfg.S, self.cfg.H, dtype=self.dtype,
                device=self.device)
            self._refresh_hcc_entry(entry, b)
            self._hcc.append(entry)
        self.hcc_enabled = True
        # fused in-window combine derive (default ON; FLUX_EPIC_HCC_DERIVE=0
        # = in-op self-build ablation). Untimed one-shot drift guard: the C++
        # entry must reproduce the python builders bitwise on the setup
        # routing (same discipline as test_moe_l0l1_traffic's guard).
        self._hcc_derive_fused = bool(int(os.environ.get(
            "FLUX_EPIC_HCC_DERIVE", "1")))
        if self._hcc_derive_fused:
            for entry, b in zip(self._hcc, self._hc_bundles):
                uc_ref = (entry["uc"] if nn > 1 else None)
                outs = entry["op"].derive_combine_meta(
                    b.meta.splits.to(self.device).int().contiguous(),
                    entry["routing"].int().contiguous(),
                    b.meta.splits_per_source.int().contiguous(),
                    uc_ref)
                assert torch.equal(outs[0].cpu(), entry["pack"].cpu()), (
                    "derive_combine_meta pack_index != python builder")
                assert torch.equal(outs[1].cpu(), entry["red"].cpu()), (
                    "derive_combine_meta reduce_index != python builder")
                if nn > 1:
                    for got, ref in zip(outs[2:6],
                                        entry["wire"] + entry["redcsr"]):
                        assert torch.equal(got.cpu(), ref.cpu()), (
                            "derive_combine_meta compress CSR != python "
                            "builder")

    def _hcc_rs_demands(self, b):
        """Exact send/stage/conv/wire demands, replicating the op's
        collective FLUX_CHECKs (gemm_grouped_v2_gather_rs.cc; same
        expressions as sweeps/gen_matrix.a2av_rs_knob_demands) on THIS
        group's virtual wire. cpr = S*K_g is NOT a valid conv/wire bound:
        conv aggregates a source node's L ranks per remote dest LANE, so
        lane skew (EPIC replica placement) can exceed it — bites at m=1
        where the whole batch shares one panel (m>=2 splits the skew)."""
        from .a2av_combine_indices import build_a2av_unique_counts

        W = self.cfg.R
        L = self._hc_L
        nn = W // L
        cpr = self.cfg.S * b.K_g
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
        return {
            "send": max(int(b.meta.m_per_rank.max()), 1),
            "stage": max(stage_d, 1),
            "conv": max(conv_d, 1),
            "wire": max(wire_d, 1),
        }

    def _refresh_hcc_entry(self, entry, b):
        """(Re)build the routing-dependent fields of one combine entry —
        everything except the frozen op/barriers/partial. Called at enable
        time and again after every migration (per-rank rows and all combine
        indices change when instances move between ranks)."""
        from .a2av_combine_indices import (
            build_a2av_combine_indices,
            build_a2av_compress_indices,
            build_a2av_unique_counts,
        )

        W = self.cfg.R
        nn = W // self._hc_L
        routing_cpu = b.meta.scatter_index.flatten().cpu()
        splits_cpu = b.meta.splits.cpu()
        pack_idx, red_idx = build_a2av_combine_indices(
            routing_cpu, splits_cpu, self.rank, W, b.K_g)
        entry.update(
            routing=routing_cpu.cuda(), pack=pack_idx, red=red_idx,
            uc=None, wire=None, redcsr=None,
            m_this=int(b.meta.m_per_rank[self.rank]),
        )
        if nn > 1:
            uc = build_a2av_unique_counts(
                b.virtual_choosed, W, nn, b.gpe)
            wp, wc, rp, rr = build_a2av_compress_indices(
                routing_cpu, splits_cpu, uc, self.rank, W, nn, b.K_g)
            entry.update(uc=uc, wire=[wp, wc], redcsr=[rp, rr])
        cap_rows = getattr(self, "_hcc_m_capacity", None)
        rows = max(entry["m_this"], 1) if cap_rows is None else \
            max(cap_rows, entry["m_this"], 1)
        entry["inbuf"] = torch.zeros(
            rows, self.cfg.H, dtype=self.dtype, device=self.device)
        entry["scale"] = torch.zeros(
            rows, dtype=torch.float32, device=self.device)
        entry["m_use"] = max(entry["m_this"], 1)

    def _rebuild_hc_combine(self):
        """Post-migration combine refresh: the ops are frozen (m_full, K_g,
        gpe and the ctor RS capacities are migration-invariant or covered by
        the headroom caps — hard-asserted here); every routing-dependent
        entry field is rebuilt from the refreshed bundles."""
        for g, b in enumerate(self._hc_bundles):
            demands = self._hcc_rs_demands(b)
            for k, v in demands.items():
                assert v <= self._hcc_knob_caps[g][k], (
                    f"group {g}: post-migration combine {k} demand {v} "
                    f"exceeds the ctor capacity {self._hcc_knob_caps[g][k]};"
                    f" raise --hc_headroom")
            self._refresh_hcc_entry(self._hcc[g], b)

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
            # the op packs M_this_ep = inbuf-ARG.size(0) rows; combine_group
            # passes the [0, m_use) slice, so under capacity mode a varying
            # per-iteration row count is legal (stale tail rows sit beyond
            # the slice and are never packed)
            if getattr(self, "_hcc_m_capacity", None) is not None:
                assert n_rows <= e["inbuf"].size(0), (
                    f"group {g}: iteration combine rows {n_rows} exceed "
                    f"the reserved capacity {e['inbuf'].size(0)} — "
                    "provable-bound breach (sizing bug, never noise)")
                e["m_use"] = max(n_rows, 1)
                if n_rows == 0:
                    e["inbuf"][:1].zero_()
                    e["scale"][:1].zero_()
            else:
                # bundle m_this INCLUDES the zero pad tail (pad vslot =
                # last slot per rank block => pads sit AFTER the real
                # rows in expert-major order); the layout's n_rows counts
                # REAL rows only. Equality only holds at zero pads (all
                # prior l01 runs) — with pads (m>1 / K_g>K) the op must
                # pack the FULL m_this block, real rows + zero tail (the
                # buffer is zero-initialized and the real region is
                # layout-static here, so the tail stays zero).
                assert e["inbuf"].size(0) >= n_rows, (
                    f"group {g}: hcc inbuf rows {e['inbuf'].size(0)} < "
                    f"layout rows {n_rows} — bundle/layout drift (derive "
                    "must follow any rebuild)")
                e["m_use"] = max(e["inbuf"].size(0), 1)
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
            # 2026-08-22 fix (audit bug (a)): TopkReduceScatterOp.run lands ALL
            # its work on the private side stream self._hcc_stream and joins
            # its internal streams only into that stream. The two reference
            # callers of the same op (C++ forward_gather_rs_impl, triton
            # moe_gather_rs.py) bracket it with an IN edge (cp_stream waits the
            # caller) and an OUT edge (caller waits cp_stream); this runner had
            # neither, so combine_pack's inbuf/scale writes, the closing
            # barrier, the barrier flag zero_ and accumulate_group's read of
            # e["partial"] were unordered against the op -> per-lane
            # stale-by-one contributions under changing payloads (static
            # payloads masked it). FLUX_EPIC_HCC_STREAM_EDGES=0 restores the
            # unfenced form for bisection only.
            edges = getattr(self, "_hcc_stream_edges", True)
            if edges:
                self._hcc_stream.wait_stream(stream)
            e["barriers"][self.rank % self._hc_L].fill_(1)
            self._hcc_group_barrier.barrier_all(stream.cuda_stream)
            m_use = e.get("m_use", e["inbuf"].size(0))
            e["op"].run(
                [e["inbuf"][:m_use]], e["partial"],
                self.rank * b.gpe, b.gpe,
                self._hc_splits_gpu[g], e["routing"],
                [e["scale"][:m_use]],
                self._hcc_pack_blocks,
                self._hcc_stream.cuda_stream,
                # inwindow mode refreshes sps per iteration (migration can
                # reshape the wire); python mode keeps the plan's tensor
                e.get("sps", b.meta.splits_per_source),
                e["pack"], e["red"],
                e["uc"], e["wire"], e["redcsr"],
            )
            if edges:
                stream.wait_stream(self._hcc_stream)
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

    def apply_migration_inkernel(self, swaps):
        """EPIC §4.3 faithful path (--migration inkernel): host decision +
        plan/index rebuild only — NO NCCL, NO host sync. The weight exchange
        runs as the fused in-kernel phase 0 of the next group-0 hc dispatch
        (weights complete, then the token wire). Fidelity assumption
        (recorded as epic_swap_fused_path): the host-rebuilt post-swap
        indices ARE the 'updated placement' the dispatch routes with — the
        host-baked analog of the paper's device-resident map update.
        Returns (recv_bytes, relayout_ms) like apply_migration."""
        assert self._inkernel_swap and self.hc_enabled, (
            "inkernel migration needs enable_hier_compress(inkernel_swap=True)")
        assert self._pending_swap is None, (
            "previous swap descriptor never consumed by dispatch_group_hc(0)")
        apply_swaps(self.plan, swaps)
        relayout_ms = self.rebuild_after_migration()
        if swaps:
            # replicated: every rank bumps the round sequence, participant
            # or not (both pair members must agree on the flag epoch)
            self._swap_seq += 1
        mine = [s for s in swaps if self.rank in (s[0], s[2])]
        assert len(mine) <= 1, mine  # planner invariant sizes the scratch
        recv_bytes = 0
        if mine:
            rh, a, rl, b, _gain = mine[0]
            peer, slot = (rl, a) if self.rank == rh else (rh, b)
            self._pending_swap = (peer, slot, self._swap_seq)
            recv_bytes = (self.slot_fc1[slot].numel()
                          * self.slot_fc1.element_size())
            if self.place_fc2:
                recv_bytes += (self.slot_fc2[slot].numel()
                               * self.slot_fc2.element_size())
            self.migration_swap_bytes += recv_bytes
        return recv_bytes, relayout_ms

    # -- accounting ---------------------------------------------------------

    def dup_stats(self):
        return epic_dup_stats(self.elay, self.cfg.R)

    def internode_rows(self):
        return (self.elay.internode_send_rows, self.elay.internode_recv_rows)
