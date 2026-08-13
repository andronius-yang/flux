################################################################################
#
# Copyright 2026 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
"""MoonEP plan -> fused GemmGroupedV2AGScatterOp metadata (the VIRTUAL EXPERT
SPACE mapping).

The fused op has no index-injection API: its only levers are scatter_index,
splits_gpu / splits_per_source, and a2av_unique_counts, and its destination
rule is expert homing (dest = e // ep_nexperts). This module re-encodes a
replicated MoonEP plan as routing over R*(epn+B) VIRTUAL experts

    v = dest * gpe + g_local,   gpe = epn + B,
    g_local = e - dest*epn                 if dest is e's home rank
            = epn + b, experts_to_copy[dest][b] == e   otherwise,

so the UNMODIFIED fused op executes the plan's placement: v's home rank
(v // gpe) is the plan-decided destination, and the per-rank weight tensor is
the contiguous [epn + B, ffn_shard, H] home+slots layout (upstream MoonEP's
own [E+B] mapped-tensor shape).

Wire-semantics note: MoonEP's rank-level dedup (one representative row per
(token, dest rank), NR-12) is exactly compress's intra-node unique-token wire;
inter-node the union machinery dedups further to one copy per (token, dest
node). Dup expansion happens implicitly through sorted_gather_index; the
staged harness's token_padding / zero-fill / NvS layout have no fused
counterpart (grouped GEMM takes arbitrary segment sizes).

Order theorem (the linchpin of golden comparisons, unit-tested in
test_moonep_fused_map.py): virtual expert v = (d, e) holds exactly the
occurrence window [alloc_cumsum[e, d-1], alloc_cumsum[e, d]) of e's global
source-major occurrence sequence -- the same interior ordering as the fused
op's A rows -- so fused output row (v, i) corresponds to staged plan slot
(d, expert_off[d][e] + i).

Everything here is pure CPU torch on the replicated plan: every rank computes
identical outputs, no collectives, no GPU required (mirrors
moonep_semantics.py; runs wherever the planner test suite runs).
"""

from dataclasses import dataclass

import torch

from flux.testing.moonep_semantics import MoonEPPlan

__all__ = [
    "MoonEPVirtualMap",
    "FusedMeta",
    "build_virtual_map",
    "build_fused_metadata",
    "preflight_metadata_checks",
    "required_a2av_knobs",
    "fused_row_map",
    "assign_gateways",
]


@dataclass
class MoonEPVirtualMap:
    plan: MoonEPPlan
    gpe: int  # groups per rank = epn + B
    E_virt: int  # R * gpe
    dest: torch.Tensor  # [R, N] int64, decoded from plan.dst (raw // NvS)
    slot_lut: torch.Tensor  # [R, E] int64: local group of expert e on rank d, -1 if none
    virtual_choosed: torch.Tensor  # [R*S, K] int32: v = dest*gpe + slot_lut[dest, e]


@dataclass
class FusedMeta:
    scatter_index: torch.Tensor  # [R*S, K] int32 CPU (stable global expert-major position)
    splits: torch.Tensor  # [E_virt] int32
    splits_per_source: torch.Tensor  # [W, E_virt] int32 CPU contiguous
    a2av_unique_counts: torch.Tensor  # [W, W + nnodes] int32 CPU contiguous
    m_per_rank: torch.Tensor  # [W] int64: GEMM rows (copies) destined per rank


def _stable_scatter_index(virtual_choosed: torch.Tensor) -> torch.Tensor:
    # moe_utils.calc_scatter_index_stable, inlined to avoid moe_utils's
    # module-level flux/triton import chain.
    return (
        virtual_choosed.flatten().argsort(stable=True).argsort().int().view(virtual_choosed.shape)
    )


def build_virtual_map(plan: MoonEPPlan, topk_all: torch.Tensor) -> MoonEPVirtualMap:
    """Decode the plan's canonical dst into the virtual expert space.

    topk_all: [R, S, K] int expert ids (the same tensor compute_moonep_plan
    consumed). The plan's dst is the authority on destinations; the expert id
    is needed only to pick the local group, and the slot_lut assert is the
    loud failure for B < epn plans (a remote expert segment without a
    prefetch slot has no virtual expert -- same contract as the staged
    runner's "use MoonEP default B=E/R" assert).
    """
    cfg = plan.cfg
    R, E, epn, B, K = cfg.R, cfg.E, cfg.epn, cfg.B, cfg.K
    gpe = epn + B
    assert topk_all.shape == (R, cfg.S, K)

    slot_lut = torch.full((R, E), -1, dtype=torch.int64)
    home_cols = (
        torch.arange(R, dtype=torch.int64).unsqueeze(1) * epn
        + torch.arange(epn, dtype=torch.int64).unsqueeze(0)
    )  # [R, epn] expert ids homed on each rank
    slot_lut.scatter_(
        1, home_cols, torch.arange(epn, dtype=torch.int64).unsqueeze(0).expand(R, epn)
    )
    e2c = plan.experts_to_copy.long()  # [R, B], -1 = empty slot
    slot_vals = (torch.arange(B, dtype=torch.int64) + epn).unsqueeze(0).expand(R, B)
    valid = e2c >= 0
    # scatter only valid slots (index 0 placeholder for invalid, masked after)
    lut_rows = torch.arange(R).unsqueeze(1).expand(R, B)[valid]
    slot_lut[lut_rows, e2c[valid]] = slot_vals[valid]

    enc = plan.dst.long()  # [R, N]
    raw = torch.where(enc < 0, -enc - 1, enc)
    dest = torch.div(raw, cfg.NvS, rounding_mode="floor")  # [R, N]
    e_ent = topk_all.reshape(R, cfg.N).long()
    g_local = slot_lut[dest.flatten(), e_ent.flatten()].view(R, cfg.N)
    assert (g_local >= 0).all(), (
        "routed entry maps to a destination without a local group for its "
        "expert (B < epn plan?); the virtual-expert mapping requires MoonEP "
        "default B = E/R"
    )
    virtual_choosed = (dest * gpe + g_local).reshape(R * cfg.S, K).int()
    return MoonEPVirtualMap(
        plan=plan,
        gpe=gpe,
        E_virt=R * gpe,
        dest=dest,
        slot_lut=slot_lut,
        virtual_choosed=virtual_choosed,
    )


def build_fused_metadata(vmap: MoonEPVirtualMap, local_world_size: int) -> FusedMeta:
    """The test_moe_ag_traffic.py metadata recipe with G := E_virt.

    All CPU int32, replicated (pure function of the replicated plan).
    """
    cfg = vmap.plan.cfg
    W, gpe, E_virt = cfg.R, vmap.gpe, vmap.E_virt
    K = cfg.K
    ntokens = W * cfg.S
    L = local_world_size
    assert W % L == 0
    nn = W // L

    vce = vmap.virtual_choosed.long()  # [ntokens, K]
    scatter_index = _stable_scatter_index(vmap.virtual_choosed)

    splits = torch.bincount(vce.flatten(), minlength=E_virt).int()
    src_of_copy = (
        torch.arange(ntokens, dtype=torch.int64) // cfg.S
    ).repeat_interleave(K)
    splits_per_source = (
        torch.bincount(src_of_copy * E_virt + vce.flatten(), minlength=W * E_virt)
        .view(W, E_virt)
        .int()
        .contiguous()
    )

    owner = vce // gpe  # [ntokens, K] dest rank
    flags = torch.zeros(ntokens, W, dtype=torch.bool)
    flags.scatter_(1, owner, True)
    u_mat = flags.view(W, cfg.S, W).sum(1)  # [W, W] unique tokens s -> rank d
    U_mat = (
        flags.view(ntokens, nn, L).any(dim=2).view(W, cfg.S, nn).sum(1)
    )  # [W, nn] unique tokens s -> node union
    a2av_unique_counts = torch.cat([u_mat, U_mat], dim=1).int().contiguous()

    m_per_rank = splits.long().view(W, gpe).sum(1)
    return FusedMeta(
        scatter_index=scatter_index,
        splits=splits,
        splits_per_source=splits_per_source,
        a2av_unique_counts=a2av_unique_counts,
        m_per_rank=m_per_rank,
    )


def preflight_metadata_checks(meta: FusedMeta, W: int, local_world_size: int) -> None:
    """Python replica of the op's collective FLUX_CHECKs
    (gemm_grouped_v2_ag_scatter.cc:1226-1243): fail here, in one process,
    instead of as a collective abort under srun."""
    L = local_world_size
    nn = W // L
    E_virt = meta.splits.numel()
    gpe = E_virt // W
    cnt = meta.splits_per_source.long()
    chunks = cnt.view(W, W, gpe).sum(2)  # [W, W] copies s -> dest rank
    u = meta.a2av_unique_counts[:, :W].long()
    U = meta.a2av_unique_counts[:, W:].long()
    assert torch.equal(cnt.sum(0).int(), meta.splits), "splits_per_source column sums != splits"
    assert (u >= 0).all() and (u <= chunks).all(), "u out of [0, chunks]"
    assert torch.equal((u > 0), (chunks > 0)), "(u>0) must equal (chunks>0)"
    node_of_d = torch.arange(W) // L
    for n in range(nn):
        cols = node_of_d == n
        assert (u[:, cols].max(dim=1).values <= U[:, n]).all(), "u[s,d] > U[s,node(d)]"
        assert (U[:, n] <= u[:, cols].sum(dim=1)).all(), "U[s,n] > sum_d u[s,d]"


def _chunk_bound(U_mat: torch.Tensor, L: int, n: int, m: int, k: int) -> int:
    # canonical stream of source node n's L union segments toward node m,
    # cut into L near-equal contiguous chunks (relay rank k carries chunk k)
    total = int(U_mat[n * L : (n + 1) * L, m].sum())
    return (total // L) * k + min(k, total % L)


def required_a2av_knobs(meta: FusedMeta, W: int, local_world_size: int) -> dict:
    """Exact capacity requirements, replicating the op's checks
    (gemm_grouped_v2_ag_scatter.cc:1147, 1250-1263, 2315-2340) for the
    lb_union arm (union_bcast, balanced relay). Values are ROW counts.

    Under lb_union a remote-node source's recv region holds the whole node
    union U[s][node(d)], so the ctor default (2 * tokens_per_rank * topk) can
    overflow at low topk -- always set these in the environment BEFORE op
    construction (they are read at ctor time).
    """
    L = local_world_size
    nn = W // L
    u = meta.a2av_unique_counts[:, :W].long()
    U = meta.a2av_unique_counts[:, W:].long()

    def region_rows(s: int, d: int) -> int:
        if nn > 1 and s // L != d // L:
            return int(U[s, d // L])
        return int(u[s, d])

    max_recv = int(meta.m_per_rank.max())
    for d in range(W):
        col = sum(region_rows(s, d) for s in range(W))
        max_recv = max(max_recv, col)

    max_stage = 0
    max_relay = 0
    if nn > 1:
        for n in range(nn):
            for k in range(L):
                srows = sum(
                    _chunk_bound(U, L, ns, n, k + 1) - _chunk_bound(U, L, ns, n, k)
                    for ns in range(nn)
                    if ns != n
                )
                max_stage = max(max_stage, srows)
                rrows = sum(
                    _chunk_bound(U, L, n, (n - dn + nn) % nn, k + 1)
                    - _chunk_bound(U, L, n, (n - dn + nn) % nn, k)
                    for dn in range(1, nn)
                )
                max_relay = max(max_relay, rrows)

    return {
        "FLUX_A2AV_MAX_RECV_NTOKENS": str(max(max_recv, 1)),
        "FLUX_A2AV_MAX_STAGE_NTOKENS": str(max(max_stage, 1)),
        "FLUX_A2AV_MAX_RELAY_NTOKENS": str(max(max_relay, 1)),
    }


def fused_row_map(vmap: MoonEPVirtualMap, rank: int) -> tuple:
    """(fused_rows, plan_slots): 1:1 correspondence between this rank's fused
    output rows (virtual-expert-major, no padding) and the staged harness's
    padded out_buf rows (order theorem). Both int64 CPU."""
    plan = vmap.plan
    cfg = plan.cfg
    gpe, epn = vmap.gpe, cfg.epn
    splits = torch.bincount(
        vmap.virtual_choosed.long().flatten(), minlength=vmap.E_virt
    )  # recomputed locally to keep this function self-contained
    fused_rows = []
    plan_slots = []
    base = 0
    for g in range(gpe):
        v = rank * gpe + g
        cnt = int(splits[v])
        if cnt == 0:
            continue
        e = rank * epn + g if g < epn else int(plan.experts_to_copy[rank, g - epn])
        assert e >= 0
        start = int(plan.expert_off[rank, e])
        fused_rows.append(torch.arange(base, base + cnt, dtype=torch.int64))
        plan_slots.append(torch.arange(start, start + cnt, dtype=torch.int64))
        base += cnt
    if not fused_rows:
        return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)
    return torch.cat(fused_rows), torch.cat(plan_slots)


def assign_gateways(plan: MoonEPPlan, local_world_size: int) -> torch.Tensor:
    """Replicated, deterministic weight-push plan with node-level multicast
    gateways.

    Returns pairs_cpu int32 [n_pairs, 6] =
        (dst_rank, dst_slot, home_rank, src_row, gw_rank, gw_slot)
    gw_rank == -1 -> direct put home -> dst (single-node groups, the gateway
    member itself, and every intra-home-node destination). For a cross-node
    (expert, dest_node) group the gateway is the needy rank minimizing the
    running per-gateway forward-byte load (deterministic: groups visited in
    ascending (expert, dest_node); ties to lowest rank); the home rank sends
    ONE inter-node put to the gateway's own slot, the gateway forwards to the
    other needy ranks of its node over NVLink.
    """
    cfg = plan.cfg
    R, B, epn = cfg.R, cfg.B, cfg.epn
    L = local_world_size
    e2c = plan.experts_to_copy  # [R, B]
    # (expert, dest_node) -> [(dest_rank, slot), ...]
    groups: dict = {}
    for d in range(R):
        for b in range(B):
            e = int(e2c[d, b])
            if e < 0:
                continue
            groups.setdefault((e, d // L), []).append((d, b))
    fwd_load = [0] * R  # forwarded-expert count per gateway candidate
    pairs = []
    for (e, node), members in sorted(groups.items()):
        home = e // epn
        src_row = e % epn
        if node == home // L or len(members) == 1:
            # same node as home (NVLink direct) or singleton: all direct
            for d, b in members:
                pairs.append((d, b, home, src_row, -1, -1))
            continue
        gw_rank, gw_slot = min(members, key=lambda m: (fwd_load[m[0]], m[0]))
        fwd_load[gw_rank] += len(members) - 1
        for d, b in members:
            if d == gw_rank:
                pairs.append((d, b, home, src_row, -1, -1))  # the inter-node leg
            else:
                pairs.append((d, b, home, src_row, gw_rank, gw_slot))
    return torch.tensor(pairs, dtype=torch.int32).reshape(-1, 6)
