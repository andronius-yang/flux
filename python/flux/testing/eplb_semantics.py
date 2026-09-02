"""EPLB static-placement semantics for the layer0 dispatch harness.

EPLB (deepseek-ai/EPLB) is the predictive re-placement baseline MoonEP and
UltraEP define themselves against: compute ONE placement — a full
re-placement, masters move too, plus replicas in redundant slots — from a
*predicted* per-expert load vector, then pay zero per-batch solver and zero
per-batch weight movement. This module maps the vendored algorithm's output
(test/python/moe_ag_scatter/eplb_oracle/eplb.py, verbatim deepseek-ai/EPLB
@ d52c72d) onto the UltraEPPlan tensor layout so the whole staged data plane
(build_comm_layout / reroute_expand / UltraEPLayer0Runner) is reused
unchanged.

Semantics vs the UltraEP arm, in one paragraph: placement (`p2l/l2p/lcnts`)
comes from the POOL-predicted load and is static for the cell; quotas and
`rank_quota_prefix` come from the BATCH routing (so reroute conservation
holds exactly), split equally across an expert's instances by
largest-remainder — EPLB's own planning assumption — with the same
coprime-stride interleave as UltraEP's reroute. There is no per-batch
weight_sync phase: `EPLBLayer0Runner.place_weights` moves every re-homed
expert's weights ONCE at setup (book-kept, never timed in the phase loop).
Replicas are NOT NVL-domain-confined (global policy); slot budget matches
UltraEP exactly (nlp = epn + R_red physical slots per rank).

Hard constraints inherited from the shared machinery:
  * NEVER use the locality-aware quota path on an EPLB plan — it asserts the
    instance host lies in the source's NVL domain
    (ultraep_semantics._rank_quota_alloc_for_expert), which full re-placement
    violates. All quota decomposition here passes locality_aware=False, and
    remote-fraction facts must do the same.
  * EPLB may legally co-locate two instances of one expert on one rank and
    may place an expert's instances on any rank; do not port UltraEP's
    one-copy-per-rank / master-pinning invariants.
  * The global policy can create up to 1 + R*R_red instances of one expert,
    which exceeds UltraEPConfig.max_replicas_dim (= R); build_eplb_plan
    widens the config's max_replicas_dim accordingly before building the
    plan tensors.
"""

import os
import time
from dataclasses import dataclass, replace as _dc_replace

import torch

from .ep_gpu_plan import (
    comb_dst_slot_from_topk,
    d6_rank_quota_prefix,
    direct_layout_entries,
    direct_layout_entries_fast,
    largest_remainder_split,
    local_spread_rank_quota_prefix,
    rank_quota_prefix_nonlocal,
    reroute_expand_all_gpu,
    reroute_expand_all_gpu_fast,
)

# Replica-selection rules (2026-08-20 campaign-2 decision): sender-local
# modes only — `local_spread` (default; per-source largest-remainder equal
# split == token round-robin counts; SGLang dynamic-dispatch analog) and
# `local_static` (src mod C; SGLang static-map / EPIC D6 class). The
# global-quota split survives as `quota` for the retired staged arms'
# history and their parity tests, but is NOT a sweep knob value and gets
# no fused arm: it is the idealized global-sync ceiling that exists
# nowhere in production.
REPLICA_SELECT_MODES = ("local_spread", "local_static", "quota")
from .ultraep_semantics import (
    UltraEPConfig,
    UltraEPPlan,
    UltraEPLayer0Runner,
    build_rank_quota_prefix,
)

# Per-logical-expert canonical weight seeds: ANY rank can generate (and
# bitwise-verify) any expert's weights, which is what lets the one-time
# placement P2P be checked without broadcasts.
_FC1_SEED_BASE = 130003
_FC2_SEED_BASE = 260009

EPLB_POLICIES = ("global", "hier")


def build_eplb_plan(cfg: UltraEPConfig, tpe: torch.Tensor,
                    pool_load, policy: str, num_nodes: int,
                    rebalance_fn,
                    replica_select: str = "quota") -> UltraEPPlan:
    """Map one rebalance_experts() call onto the UltraEPPlan tensor layout.

    cfg:        shared shape config (D is irrelevant here beyond divisibility;
                the locality path is never used). MUTATED: max_replicas_dim
                is widened to fit the global policy's worst case.
    tpe:        [R, G] BATCH per-source-rank load histogram (loads_from_topk).
    pool_load:  length-G predicted per-expert load (the full trace pool
                histogram; any non-negative floats).
    policy:     "global" (canonical, DeepSeek's decode deployment) or "hier"
                (node-confined replicas; needs G/R/P divisible by num_nodes).
    rebalance_fn: the vendored eplb.rebalance_experts (injected so
                flux.testing never imports from the test tree).
    """
    assert policy in EPLB_POLICIES, policy
    assert tuple(tpe.shape) == (cfg.R, cfg.G)
    pool_load = torch.as_tensor(pool_load, dtype=torch.float64).flatten()
    assert pool_load.numel() == cfg.G, (
        f"pool_load has {pool_load.numel()} entries, expected G={cfg.G}"
    )
    assert bool((pool_load >= 0).all()), "negative pool load"

    # Widen the l2p/quota/rank_quota_prefix instance dimension: the global
    # policy may give one expert every redundant slot (1 + R*R_red instances).
    cfg.max_replicas_dim = max(cfg.R, 1 + cfg.R * cfg.R_red)

    weight = pool_load.float().reshape(1, cfg.G)
    if policy == "global":
        # eplb.py's own global spelling: hierarchical with 1 group / 1 node.
        phy2log, log2phy, logcnt = rebalance_fn(
            weight, cfg.P, num_groups=1, num_nodes=1, num_gpus=cfg.R
        )
    else:
        assert num_nodes >= 1
        assert cfg.G % num_nodes == 0 and cfg.R % num_nodes == 0, (
            f"hier policy needs G ({cfg.G}) and R ({cfg.R}) divisible by "
            f"num_nodes ({num_nodes})"
        )
        # Qwen3 has no expert groups: every expert is its own group, so
        # Step 1 (pack groups to nodes) becomes per-expert node packing.
        phy2log, log2phy, logcnt = rebalance_fn(
            weight, cfg.P, num_groups=cfg.G, num_nodes=num_nodes,
            num_gpus=cfg.R,
        )

    p2l = phy2log[0].to(torch.int32)
    lcnts = logcnt[0].to(torch.int32)
    assert int(lcnts.sum()) == cfg.P
    assert int(lcnts.min()) >= 1
    assert int(lcnts.max()) <= cfg.max_replicas_dim, (
        f"policy {policy}: expert with {int(lcnts.max())} instances exceeds "
        f"max_replicas_dim {cfg.max_replicas_dim}"
    )

    # l2p columns in replica-rank order: log2phy's column index IS the
    # replica rank (eplb.py scatters physical ids at column phyrank), so the
    # first lcnts[l] columns are exactly the valid entries.
    l2p = torch.full((cfg.G, cfg.max_replicas_dim), -1, dtype=torch.int32)
    src_width = log2phy.shape[-1]
    l2p[:, :src_width] = log2phy[0].to(torch.int32)
    for l in range(cfg.G):
        C = int(lcnts[l])
        assert bool((l2p[l, :C] >= 0).all()), f"expert {l}: hole in l2p"
        assert bool((l2p[l, C:] < 0).all()), f"expert {l}: stray l2p entry"

    # Quota: equal split of the BATCH load across instances,
    # largest-remainder (extras to the lowest instance index). Conservation
    # sum(quota[l,:C]) == batch load keeps reroute_expand's per-source
    # totals exact.
    loads_g = tpe.long().sum(dim=0)
    quota = torch.zeros(cfg.G, cfg.max_replicas_dim, dtype=torch.int32)
    quota_prefix = torch.zeros(cfg.G, cfg.max_replicas_dim, dtype=torch.int32)
    for l in range(cfg.G):
        C = int(lcnts[l])
        load = int(loads_g[l])
        base, rem = divmod(load, C)
        prefix = 0
        for j in range(C):
            q = base + (1 if j < rem else 0)
            quota[l, j] = q
            prefix += q
            quota_prefix[l, j] = prefix

    # Replica rule (see REPLICA_SELECT_MODES): the two local modes are
    # pure per-source functions (no global loads consumed); under them the
    # quota/quota_prefix fields above become informational only (the
    # build_epic_plan precedent). The ep_gpu_plan producers are
    # device-agnostic, so calling them on CPU tensors here keeps the
    # setup reference bitwise-consistent with the timed device planner.
    assert replica_select in REPLICA_SELECT_MODES, replica_select
    if replica_select == "quota":
        rank_quota_prefix = torch.stack([
            build_rank_quota_prefix(cfg, tpe, l2p, lcnts, quota, src,
                                    locality_aware=False)
            for src in range(cfg.R)
        ])
    elif replica_select == "local_spread":
        rank_quota_prefix = local_spread_rank_quota_prefix(
            tpe, lcnts, cfg.max_replicas_dim)
    else:  # local_static
        rank_quota_prefix = d6_rank_quota_prefix(
            tpe, lcnts, cfg.max_replicas_dim)

    return UltraEPPlan(
        cfg=cfg, tpe=tpe.to(torch.int32), p2l=p2l, l2p=l2p, lcnts=lcnts,
        quota=quota, quota_prefix=quota_prefix,
        rank_quota_prefix=rank_quota_prefix,
        domain_solutions=[],
    )


def fused_capacity_bounds(plan: UltraEPPlan, topk_all: torch.Tensor,
                          headroom: float = 1.25):
    """(max_rows_per_pair, max_recv_total) for the FusedEpDispatch ctor —
    deployment-scope allocation sizing (rule-5 legal one-shot) from the
    setup reference routing, GLOBAL over ranks so the collective geometry
    contract holds identically everywhere. Contents stay per-iteration;
    the op's in-kernel collective trap enforces the bounds at runtime."""
    from .ultraep_semantics import reroute_expand

    cfg = plan.cfg
    R, nlp = cfg.R, cfg.nlp
    P = R * nlp
    pair_max = 0
    recv_rows = torch.zeros(P, dtype=torch.int64)
    for src in range(R):
        _, phys = reroute_expand(cfg, plan, src, topk_all[src])
        per_slot = torch.bincount(phys, minlength=P)
        pair_max = max(pair_max, int(per_slot.max()))
        recv_rows += per_slot
    recv_max = int(recv_rows.view(R, nlp).sum(1).max())
    return (max(int(pair_max * headroom) + 1, 1),
            max(int(recv_max * headroom) + 1, 1))


def _flux():
    import flux  # GPU-side only (module import must stay CPU-clean)
    return flux


def weight_placement_pairs(plan: UltraEPPlan) -> list:
    """One-time weight movement list: (host, local_slot, logical, orig_home)
    for every physical slot whose logical expert's ORIGINAL contiguous home
    (e // epn, the fixed-placement baseline homing) is a different rank.

    Globally ordered by physical slot id, so every rank walks the identical
    sequence — the batched isend/irecv matching requirement.
    """
    cfg = plan.cfg
    p2l = plan.p2l.long()
    pairs = []
    for p in range(cfg.P):
        l = int(p2l[p])
        if l < 0:
            continue  # unused slot (plans that don't fill all P slots)
        host = p // cfg.nlp
        orig_home = l // cfg.epn
        if host != orig_home:
            pairs.append((host, p % cfg.nlp, l, orig_home))
    return pairs


def predicted_rows_per_rank(plan: UltraEPPlan, pool_load) -> list:
    """Per-rank load under the POOL prediction and equal instance split —
    EPLB's own packing objective (float rows; eplb_pred_imbalance =
    max/mean of this vector)."""
    cfg = plan.cfg
    pool_load = torch.as_tensor(pool_load, dtype=torch.float64).flatten()
    p2l = plan.p2l.long()
    lcnts = plan.lcnts.long()
    rows = [0.0] * cfg.R
    for p in range(cfg.P):
        l = int(p2l[p])
        rows[p // cfg.nlp] += float(pool_load[l]) / int(lcnts[l])
    return rows


@dataclass
class EplbIterPlan:
    """One iteration's routing-derived plan, produced on-device by
    EplbIterPlanner.derive inside the timed `plan` bracket (SCHEMA rule 5).
    Device tensors feed the phase methods directly; the host fields are the
    single batched D2H the phase pays (GemmOnly needs host segment bounds,
    the NCCL fallback needs host split lists)."""

    send_row_index: torch.Tensor      # [n_send] int64 device
    send_entry_logical: torch.Tensor  # [n_send] int64 device
    place_slots: torch.Tensor         # [n_recv] int64 device
    in_splits: torch.Tensor           # [R] int32 device
    out_splits: torch.Tensor          # [R] int32 device
    comb_dst_slot: torch.Tensor       # [n_send] int64 device (l01) or None
    send_counts: list
    recv_counts: list
    seg_rows: list
    seg_start: list
    gemm_segments: list
    n_recv: int
    max_pair_rows: int


class EplbIterPlanner:
    """Per-iteration GPU planner for the eplb arm (SCHEMA rule 5): every
    batch-derived quantity — quotas, rank-quota prefixes, the reroute
    expansion, splits, placement scatter indices, combine slots — is
    recomputed each iteration on device from the gathered loads and the
    routing, inside the timed `plan` event.

    One-shot ctor state is limited to what rule 5 exempts or the locked
    accounting boundary declares deployment-scope: the static placement
    tensors (p2l/l2p/lcnts — pool-derived, not batch-derived) and the
    replicated routing itself (gating metadata, the harness stand-in for
    the model's gate).

    Bit-parity contract: derive() reproduces build_comm_layout's fields
    exactly (check_against is the loud guard the driver runs at setup).
    Known in-phase syncs, both honest timed cost: the ragged boolean
    gather for the receiver rows, and the single batched D2H at the end.
    """

    def __init__(self, plan: UltraEPPlan, rank: int, device,
                 topk_all: torch.Tensor, want_comb: bool = False,
                 replica_select: str = "quota"):
        assert replica_select in REPLICA_SELECT_MODES, replica_select
        cfg = plan.cfg
        self.cfg = cfg
        self.rank = rank
        self.device = device
        self.want_comb = want_comb
        self.replica_select = replica_select
        # deployment-scope placement (rule-5 legal one-shot)
        self.l2p = plan.l2p.to(device)
        self.lcnts = plan.lcnts.to(device)
        self.p2l = plan.p2l.long().to(device)
        self._p2l_host = plan.p2l.long()
        # gating metadata (the rule-5 exempt input): replicated routing
        self.topk_all = topk_all.long().to(device)
        # fast-tail blob buffer (the phase's ONE D2H; size is cfg-static:
        # seg_rows + seg_start + in/out splits + pair_max + n_recv + ilv_ok)
        self._blob_pin = torch.empty(
            2 * cfg.nlp + 2 * cfg.R + 3, dtype=torch.int64,
            pin_memory=torch.cuda.is_available())

    def local_loads(self) -> torch.Tensor:
        """[G] int32 this-rank load histogram — the plan_comm payload,
        derived from routing per iteration (timed). Sync-free under the
        fast tail (torch.bincount hides an output-sizing D2H)."""
        ids = self.topk_all[self.rank].reshape(-1)
        if int(os.getenv("FLUX_PLL_FAST_TAIL", "1")):
            out = torch.zeros(self.cfg.G, dtype=torch.int64,
                              device=ids.device)
            out.index_add_(0, ids, torch.ones_like(ids))
            return out.to(torch.int32)
        return torch.bincount(ids, minlength=self.cfg.G).to(torch.int32)

    def derive(self, loads_gather_buf: torch.Tensor) -> EplbIterPlan:
        """Dispatch: the sync-free fast tail (default; 8.23 fairness pass,
        same accounting class as the epic fast tail — bit-identical plans,
        FLUX_PLL_FAST_TAIL=0 restores the legacy spelling)."""
        if int(os.getenv("FLUX_PLL_FAST_TAIL", "1")):
            return self._derive_fast(loads_gather_buf)
        return self._derive_legacy(loads_gather_buf)

    def _rqp_for(self, tpe_all: torch.Tensor) -> torch.Tensor:
        """The replica-rule branch (REPLICA_SELECT_MODES) — the AUTHENTIC
        planning math, shared verbatim by both derive spellings. All three
        rules are already sync-free batched torch."""
        cfg = self.cfg
        if self.replica_select == "quota":
            quota, _ = largest_remainder_split(
                tpe_all.long().sum(0), self.lcnts, cfg.max_replicas_dim)
            return rank_quota_prefix_nonlocal(tpe_all, quota, self.lcnts)
        if self.replica_select == "local_spread":
            return local_spread_rank_quota_prefix(
                tpe_all, self.lcnts, cfg.max_replicas_dim)
        return d6_rank_quota_prefix(
            tpe_all, self.lcnts, cfg.max_replicas_dim)

    def _derive_fast(self, loads_gather_buf: torch.Tensor) -> EplbIterPlan:
        """Sync-free derive twin: identical plan bits, zero hidden host
        syncs before the single pinned blob D2H. Removed taxes (all port
        overhead, never rule semantics): bincount output-sizing D2Hs, the
        host-looped coprime-interleave window scan, and the ragged boolean
        gather for receiver rows (now a fixed-shape sort, sliced [:n_recv]
        after the blob lands)."""
        cfg = self.cfg
        R, G, S, K, nlp = cfg.R, cfg.G, cfg.S, cfg.K, cfg.nlp
        tpe_all = loads_gather_buf.view(R, G)
        rqp = self._rqp_for(tpe_all)
        tok_all, phys_all, ilv_ok = reroute_expand_all_gpu_fast(
            rqp, self.l2p, self.lcnts, self.topk_all, cfg.interleave)

        # canonical (phys, token) order per source == dest-major for free
        order = torch.argsort(phys_all * (S + 1) + tok_all, dim=1,
                              stable=True)
        ent_tok = torch.gather(tok_all, 1, order)
        ent_phys = torch.gather(phys_all, 1, order)

        lay = direct_layout_entries_fast(ent_tok, ent_phys, self.rank,
                                         nlp, R)
        my_tok = lay["my_tok"]
        send_entry_logical = self.p2l[lay["my_phys"]]
        in_splits, out_splits = lay["in_splits"], lay["out_splits"]
        comb_dst = (
            comb_dst_slot_from_topk(self.topk_all[self.rank], my_tok,
                                    send_entry_logical, G)
            if self.want_comb else None
        )

        # the ONE batched (pinned) D2H of the phase
        blob = self._blob_pin
        blob.copy_(torch.cat([
            lay["seg_rows"], lay["seg_start"], in_splits.long(),
            out_splits.long(), lay["pair_max"].reshape(1),
            lay["n_recv_dev"].long(), ilv_ok.long(),
        ]))
        assert int(blob[-1]) == 1, "interleave 320-candidate window miss"
        seg_rows_h = blob[:nlp].tolist()
        seg_start_h = blob[nlp:2 * nlp].tolist()
        send_counts = blob[2 * nlp:2 * nlp + R].tolist()
        recv_counts = blob[2 * nlp + R:2 * nlp + 2 * R].tolist()
        max_pair_rows = int(blob[-3])
        n_recv = int(blob[-2])
        place_slots = lay["place_slots_pad"][:n_recv]

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

        return EplbIterPlan(
            send_row_index=my_tok,
            send_entry_logical=send_entry_logical,
            place_slots=place_slots,
            in_splits=in_splits,
            out_splits=out_splits,
            comb_dst_slot=comb_dst,
            send_counts=send_counts,
            recv_counts=recv_counts,
            seg_rows=seg_rows_h,
            seg_start=seg_start_h,
            gemm_segments=segments,
            n_recv=n_recv,
            max_pair_rows=max_pair_rows,
        )

    def _derive_legacy(self, loads_gather_buf: torch.Tensor) -> EplbIterPlan:
        cfg = self.cfg
        R, G, S, K, nlp = cfg.R, cfg.G, cfg.S, cfg.K, cfg.nlp
        tpe_all = loads_gather_buf.view(R, G)

        # Replica rule branch (REPLICA_SELECT_MODES): local modes are pure
        # per-source functions of tpe_all — no global information consumed.
        if self.replica_select == "quota":
            quota, _ = largest_remainder_split(
                tpe_all.long().sum(0), self.lcnts, cfg.max_replicas_dim)
            rqp = rank_quota_prefix_nonlocal(tpe_all, quota, self.lcnts)
        elif self.replica_select == "local_spread":
            rqp = local_spread_rank_quota_prefix(
                tpe_all, self.lcnts, cfg.max_replicas_dim)
        else:  # local_static
            rqp = d6_rank_quota_prefix(
                tpe_all, self.lcnts, cfg.max_replicas_dim)
        tok_all, phys_all = reroute_expand_all_gpu(
            rqp, self.l2p, self.lcnts, self.topk_all, cfg.interleave)

        # canonical (phys, token) order per source == dest-major for free
        order = torch.argsort(phys_all * (S + 1) + tok_all, dim=1,
                              stable=True)
        ent_tok = torch.gather(tok_all, 1, order)
        ent_phys = torch.gather(phys_all, 1, order)

        lay = direct_layout_entries(ent_tok, ent_phys, self.rank, nlp, R)
        my_tok = lay["my_tok"]
        send_entry_logical = self.p2l[lay["my_phys"]]
        in_splits, out_splits = lay["in_splits"], lay["out_splits"]
        all_local = lay["all_local"]
        place_slots = lay["place_slots"]
        seg_rows, seg_start = lay["seg_rows"], lay["seg_start"]
        comb_dst = (
            comb_dst_slot_from_topk(self.topk_all[self.rank], my_tok,
                                    send_entry_logical, G)
            if self.want_comb else None
        )

        # the ONE batched D2H of the phase
        blob = torch.cat([
            seg_rows, seg_start, in_splits.long(), out_splits.long(),
            lay["pair_max"].reshape(1),
        ]).cpu()
        seg_rows_h = blob[:nlp].tolist()
        seg_start_h = blob[nlp:2 * nlp].tolist()
        send_counts = blob[2 * nlp:2 * nlp + R].tolist()
        recv_counts = blob[2 * nlp + R:2 * nlp + 2 * R].tolist()
        max_pair_rows = int(blob[-1])

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

        return EplbIterPlan(
            send_row_index=my_tok,
            send_entry_logical=send_entry_logical,
            place_slots=place_slots,
            in_splits=in_splits,
            out_splits=out_splits,
            comb_dst_slot=comb_dst,
            send_counts=send_counts,
            recv_counts=recv_counts,
            seg_rows=seg_rows_h,
            seg_start=seg_start_h,
            gemm_segments=segments,
            n_recv=int(all_local.numel()),
            max_pair_rows=max_pair_rows,
        )

    def derive_fused(self, loads_gather_buf=None) -> torch.Tensor:
        """Campaign-2 fused path (planner v2a): the ENTIRE per-iteration
        plan is one [S, K] int32 device tensor of global physical slot
        ids — everything else (counts, layouts, offsets, combine handle)
        derives inside the fused dispatch launch. Local replica modes
        consume no exchange: tpe_all is a device histogram of the
        rule-5-exempt replicated routing. Returns dst_phys.
        """
        cfg = self.cfg
        R, G, S, K = cfg.R, cfg.G, cfg.S, cfg.K
        dev = self.device
        if self.replica_select == "quota":
            assert loads_gather_buf is not None, "quota mode needs loads"
            tpe_all = loads_gather_buf.view(R, G)
            quota, _ = largest_remainder_split(
                tpe_all.long().sum(0), self.lcnts, cfg.max_replicas_dim)
            rqp = rank_quota_prefix_nonlocal(tpe_all, quota, self.lcnts)
        else:
            src_base = torch.arange(R, device=dev,
                                    dtype=torch.int64).unsqueeze(1) * G
            tpe_all = torch.bincount(
                (src_base + self.topk_all.reshape(R, S * K)).reshape(-1),
                minlength=R * G).view(R, G).to(torch.int32)
            if self.replica_select == "local_spread":
                rqp = local_spread_rank_quota_prefix(
                    tpe_all, self.lcnts, cfg.max_replicas_dim)
            else:  # local_static
                rqp = d6_rank_quota_prefix(
                    tpe_all, self.lcnts, cfg.max_replicas_dim)
        from .ep_gpu_plan import reroute_expand_gpu
        tok, phys = reroute_expand_gpu(
            rqp[self.rank], self.l2p, self.lcnts, self.topk_all[self.rank],
            cfg.interleave)
        logical = self.p2l[phys]
        cell = comb_dst_slot_from_topk(self.topk_all[self.rank], tok,
                                       logical, G)
        dst_phys = torch.empty(S * K, dtype=torch.int64, device=dev)
        dst_phys[cell] = phys
        return dst_phys.view(S, K).to(torch.int32)

    def check_against(self, ip: EplbIterPlan, lay) -> None:
        """Loud bitwise drift guard vs the setup CPU reference layout
        (untimed; the driver runs it once at setup)."""
        assert torch.equal(ip.send_row_index.cpu(), lay.send_row_index)
        assert torch.equal(ip.send_entry_logical.cpu(),
                           lay.send_entry_logical)
        assert torch.equal(ip.place_slots.cpu(), lay.place_slots)
        assert ip.send_counts == lay.send_counts, "send splits drift"
        assert ip.recv_counts == lay.recv_counts, "recv splits drift"
        assert ip.seg_rows == lay.seg_rows, "seg_rows drift"
        assert ip.seg_start == lay.seg_start, "seg_start drift"
        assert ip.gemm_segments == lay.gemm_segments, "gemm segments drift"


class EPLBLayer0Runner(UltraEPLayer0Runner):
    """UltraEP staged data plane with EPLB's static full re-placement.

    Differences from the parent: weights live in ONE per-slot tensor
    (slot_fc1[nlp]) filled once by place_weights() — there is no
    master/replica split, no per-iteration weight_sync — and the GEMM
    indexes weights purely by local physical slot.
    """

    PINNED_MASTERS = False

    def __init__(self, plan: UltraEPPlan, rank: int, group, device,
                 topk_all: torch.Tensor, dtype=torch.bfloat16,
                 ffn_size_shard: int = 0, place_fc2: bool = True,
                 weight_place_wire: str = "nccl"):
        # ffn_size_shard=0 makes the parent skip out_buf/replica_fc1/
        # replica_fc2; slot-indexed buffers replace them below.
        super().__init__(plan, rank, group, device, topk_all, dtype=dtype,
                         ffn_size_shard=0, sync_fc2=False)
        assert ffn_size_shard > 0
        assert weight_place_wire in ("nccl", "nvshmem"), weight_place_wire
        self.ffn_size_shard = ffn_size_shard
        self.place_fc2 = place_fc2
        self.weight_place_wire = weight_place_wire
        cfg = self.cfg
        self.out_buf = torch.zeros(
            max(self.n_recv, 1), ffn_size_shard, dtype=dtype, device=device
        )
        if weight_place_wire == "nvshmem":
            # 2026-09-02 (motivation-figure lane): the one-time placement
            # rides one-sided NVSHMEM puts, so every put DESTINATION (the
            # slot panels) and SOURCE (one-expert staging block) must live
            # on the symmetric heap (CXI proxy rule). Collective alloc —
            # identical shapes on every PE; requires flux.init_flux_shm
            # before construction (the nvshmem/fused transports do).
            import flux  # GPU-side only
            self.slot_fc1 = flux.nvshmem_create_tensor(
                [cfg.nlp, ffn_size_shard, cfg.H], dtype)
            self._stage_fc1 = flux.nvshmem_create_tensor(
                [ffn_size_shard, cfg.H], dtype)
            if place_fc2:
                self.slot_fc2 = flux.nvshmem_create_tensor(
                    [cfg.nlp, cfg.H, ffn_size_shard], dtype)
                self._stage_fc2 = flux.nvshmem_create_tensor(
                    [cfg.H, ffn_size_shard], dtype)
        else:
            self.slot_fc1 = torch.zeros(
                cfg.nlp, ffn_size_shard, cfg.H, dtype=dtype, device=device
            )
            if place_fc2:
                # Layer0 consumes only fc1; fc2 rides along so the one-time
                # placement bytes are faithful to moving the full expert.
                self.slot_fc2 = torch.zeros(
                    cfg.nlp, cfg.H, ffn_size_shard, dtype=dtype, device=device
                )
        self.weight_place_bytes = 0
        self.weight_place_ms = 0.0
        self.layers = "l0"
        self.comb_dst_slot = None

    # -- canonical per-logical-expert weights (any rank can generate) -------

    def make_canonical_fc1(self, logical: int) -> torch.Tensor:
        gen = torch.Generator().manual_seed(_FC1_SEED_BASE + logical)
        w = torch.rand(self.ffn_size_shard, self.cfg.H,
                       dtype=torch.float32, generator=gen) * 0.01
        return w.to(self.dtype)

    def make_canonical_fc2(self, logical: int) -> torch.Tensor:
        gen = torch.Generator().manual_seed(_FC2_SEED_BASE + logical)
        w = torch.rand(self.cfg.H, self.ffn_size_shard,
                       dtype=torch.float32, generator=gen) * 0.01
        return w.to(self.dtype)

    # -- one-time weight placement (setup only, never in the phase loop) ----

    def place_weights(self, group=None):
        """Materialize the static placement ONCE: locally-homed slots are
        filled from the canonical generators; re-homed slots receive their
        expert over batched P2P from the expert's original contiguous home
        rank. Returns (recv_bytes, oneshot_ms); also stored on self."""
        group = group if group is not None else self.group
        cfg = self.cfg
        dist = self.dist
        dev = self.device
        p2l = self.plan.p2l.long()
        pairs = weight_placement_pairs(self.plan)

        self.slot_fc1.zero_()
        if self.place_fc2:
            self.slot_fc2.zero_()
        for b in range(cfg.nlp):
            l = int(p2l[self.rank * cfg.nlp + b])
            if l // cfg.epn == self.rank:
                self.slot_fc1[b].copy_(self.make_canonical_fc1(l))
                if self.place_fc2:
                    self.slot_fc2[b].copy_(self.make_canonical_fc2(l))

        if self.weight_place_wire == "nvshmem":
            return self._place_weights_nvshmem(pairs, group)

        dist.barrier(group=group)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ops, keepalive, recv_bytes = [], [], 0
        for host, b, l, home in pairs:
            if host == self.rank:
                ops.append(dist.P2POp(dist.irecv, self.slot_fc1[b],
                                      peer=home, group=group))
                recv_bytes += self.slot_fc1[b].numel() * self.slot_fc1.element_size()
                if self.place_fc2:
                    ops.append(dist.P2POp(dist.irecv, self.slot_fc2[b],
                                          peer=home, group=group))
                    recv_bytes += self.slot_fc2[b].numel() * self.slot_fc2.element_size()
            elif home == self.rank:
                w1 = self.make_canonical_fc1(l).to(dev).contiguous()
                keepalive.append(w1)
                ops.append(dist.P2POp(dist.isend, w1, peer=host, group=group))
                if self.place_fc2:
                    w2 = self.make_canonical_fc2(l).to(dev).contiguous()
                    keepalive.append(w2)
                    ops.append(dist.P2POp(dist.isend, w2, peer=host,
                                          group=group))
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        torch.cuda.synchronize()
        self.weight_place_ms = (time.perf_counter() - t0) * 1e3
        self.weight_place_bytes = recv_bytes
        return recv_bytes, self.weight_place_ms

    def _place_weights_nvshmem(self, pairs, group):
        """Same placement, one-sided wire (2026-09-02): the ORIGINAL HOME of
        every re-homed expert pushes it with one BLOCKING
        nvshmemx_putmem_on_stream per destination slot (intra-node = CE
        P2P copy, inter-node = proxy RMA kernel — both visible per put in
        nsys with their bytes), then one world barrier fences the panels
        (wire-ordering rule 6a: blocking puts, consumers gate on the
        barrier, never a per-put signal). Canonical weights are synthesized
        on the host BEFORE the timed/NVTX bracket (synthesis is a harness
        artifact and is never quoted); the bracket holds only device
        staging copies + puts + barrier. Receivers do nothing. Global
        pair order is walked on every rank, so put issue order is
        deterministic and identical to the NCCL twin's matching order."""
        dist = self.dist
        dev = self.device
        stream = torch.cuda.current_stream()
        fc1_bytes = self._stage_fc1.numel() * self._stage_fc1.element_size()
        fc2_bytes = (self._stage_fc2.numel() * self._stage_fc2.element_size()
                     if self.place_fc2 else 0)
        # my sends, in global pair order: (dest_pe, dest_slot, logical)
        my_sends = [(host, b, l) for host, b, l, home in pairs
                    if home == self.rank]
        recv_bytes = sum(fc1_bytes + fc2_bytes for host, _, _, _ in pairs
                         if host == self.rank)
        # untimed: host synthesis of every expert this rank pushes (each
        # logical once, device-resident, ordinary memory — put SOURCE is
        # the symmetric staging block below)
        synth = {}
        for _, _, l in my_sends:
            if l not in synth:
                w1 = self.make_canonical_fc1(l).to(dev)
                w2 = self.make_canonical_fc2(l).to(dev) if self.place_fc2 else None
                synth[l] = (w1, w2)
        self.weight_place_sends = [(host, l, fc1_bytes + fc2_bytes)
                                   for host, _, l in my_sends]

        dist.barrier(group=group)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.nvtx.range("eplb_place_weights"):
            last_l = None
            for host, b, l in my_sends:
                if l != last_l:
                    # stream-ordered: the previous (blocking) puts from the
                    # staging block have completed before this overwrite
                    self._stage_fc1.copy_(synth[l][0])
                    if self.place_fc2:
                        self._stage_fc2.copy_(synth[l][1])
                    last_l = l
                with torch.cuda.nvtx.range(
                        f"place_put e{l}->pe{host}.s{b} {fc1_bytes + fc2_bytes}B"):
                    flux_put = _flux().nvshmem_putmem_on_stream
                    flux_put(self.slot_fc1[b].data_ptr(),
                             self._stage_fc1.data_ptr(), fc1_bytes, host,
                             stream.cuda_stream)
                    if self.place_fc2:
                        flux_put(self.slot_fc2[b].data_ptr(),
                                 self._stage_fc2.data_ptr(), fc2_bytes, host,
                                 stream.cuda_stream)
            _flux().nvshmem_barrier_all_on_stream(stream.cuda_stream)
        torch.cuda.synchronize()
        self.weight_place_ms = (time.perf_counter() - t0) * 1e3
        self.weight_place_bytes = recv_bytes
        return recv_bytes, self.weight_place_ms

    # -- phase overrides ----------------------------------------------------

    def weight_sync(self, *args, **kwargs):
        raise RuntimeError(
            "eplb arm has no per-iteration weight_sync: placement is static "
            "(place_weights runs once at setup)"
        )

    def gemm(self, gemm_only_op, fc1_home: torch.Tensor = None):
        """Per-segment GEMM indexing weights by local physical slot (no
        master/replica arithmetic). fc1_home accepted for signature parity,
        ignored."""
        for p, start, end, _logical in self.lay.gemm_segments:
            gemm_only_op.forward(
                self.hidden_buf[start:end], self.slot_fc1[p],
                output_buf=self.out_buf[start:end],
                fast_accum=False,
            )

    # -- per-iteration plan binding (SCHEMA rule 5) -------------------------

    def bind_iter_plan(self, ip: EplbIterPlan):
        """Swap the routing-derived index state for this iteration's plan
        (called inside the timed `plan` bracket, right after
        EplbIterPlanner.derive). Buffer ALLOCATIONS stay ctor-sized — a
        declared memory-capacity convenience — so the static-routing
        contract is asserted loudly instead of silently overflowing."""
        assert ip.n_recv == self.n_recv, (
            f"iteration recv rows {ip.n_recv} != ctor sizing {self.n_recv} "
            "(dynamic routing needs re-sized buffers)"
        )
        self.send_row_index = ip.send_row_index
        self.send_entry_logical = ip.send_entry_logical
        self.place_slots = ip.place_slots
        self.lay = _dc_replace(
            self.lay,
            send_counts=ip.send_counts,
            recv_counts=ip.recv_counts,
            gemm_segments=ip.gemm_segments,
        )
        if self.transport == "nvshmem":
            assert ip.max_pair_rows <= self._a2a_max_split, (
                f"pair rows {ip.max_pair_rows} exceed All2AllSingle "
                f"max_split {self._a2a_max_split} (silent wire overflow)"
            )
            self._in_splits = ip.in_splits
            self._out_splits = ip.out_splits
        if ip.comb_dst_slot is not None:
            self.comb_dst_slot = ip.comb_dst_slot

    # -- campaign-2 fused wire (planner v2a, CANONICAL) ---------------------
    #
    # One FusedEpDispatch call replaces pack/a2av/place: the plan is just
    # dst_phys [S, K] (EplbIterPlanner.derive_fused); counts ride in-launch;
    # recv rows land slot-major/(src, stable-cell) ordered — which IS the
    # gemm segment order, so gemm/act/gemm2 read the op views directly and
    # the l01 combine consumes the recorded headers (no combine planning).

    def enable_fused_dispatch(self, local_world_size: int,
                              num_comm_sm: int = 8, m_groups: int = 1,
                              headroom: float = 1.25,
                              spin_limit: int = 0):
        import flux  # GPU-side only

        cfg = self.cfg
        mrp, mrt = fused_capacity_bounds(self.plan, self._topk_all,
                                         headroom)
        self._fused = flux.FusedEpDispatch(
            self.group, cfg.R // local_world_size, cfg.S, cfg.H, cfg.K,
            cfg.nlp, mrp, mrt, self.dtype, m_groups, spin_limit)
        self._fused_num_comm_sm = num_comm_sm
        self._fused_probs = None
        self._dst_phys = None
        self.transport = "fused"

    def set_fused_probs(self, probs_entry: torch.Tensor):
        """[S, K] fp32 device — the gate's native per-entry probs (the
        [S, G] -> [S, K] gather is a harness-form conversion of exempt
        gating metadata, done once at setup)."""
        assert probs_entry.shape == (self.cfg.S, self.cfg.K)
        self._fused_probs = probs_entry.contiguous()

    def bind_fused_plan(self, dst_phys: torch.Tensor):
        """The ENTIRE per-iteration plan of the fused path."""
        self._dst_phys = dst_phys

    def fused_dispatch(self, inputs_shard: torch.Tensor):
        cfg = self.cfg
        recv, w, seg = self._fused.dispatch(
            inputs_shard, self._dst_phys, self._fused_probs,
            self._fused_num_comm_sm)
        nlp = cfg.nlp
        seg_rows = seg[:nlp].tolist()
        seg_start = seg[nlp:].tolist()
        base = self.rank * nlp
        p2l = self.plan.p2l.long()
        segments = []
        for p in range(nlp):
            if seg_rows[p] == 0:
                continue
            logical = int(p2l[base + p])
            assert logical >= 0, f"rank {self.rank}: rows in unused slot {p}"
            segments.append((p, seg_start[p], seg_start[p] + seg_rows[p],
                             logical))
        self.lay = _dc_replace(self.lay, gemm_segments=segments,
                               seg_rows=seg_rows, seg_start=seg_start)
        self.n_recv = seg_start[-1] + seg_rows[-1]
        self.hidden_buf = recv
        self.weights_buf = w
        return recv

    def fused_combine(self):
        """Expert-side l01 combine: header-addressed row puts into every
        source's home staging (weights NOT applied here — receiver-side
        application is the DeepEP convention)."""
        self._fused.combine(self.comb_hidden_buf[:self.n_recv],
                            self._fused_num_comm_sm)

    def fused_combine_reduce(self):
        """Home-side gate + fp32 weighted reduce over the K cells."""
        cfg = self.cfg
        staging = self._fused.combine_gate(cfg.S, self._fused_num_comm_sm)
        w = self._fused_probs.view(cfg.S, cfg.K, 1)
        self.final_out.copy_(
            (staging.view(cfg.S, cfg.K, cfg.H).float() * w).sum(1)
            .to(self.dtype))
        return self.final_out

    # -- layer1 (gemm2 + combine), the dispatch mirror ----------------------
    #
    # The combine reverses the dispatch: gemm2 rows are gathered back to
    # recv-stream order and fp32-scaled by their route probs at the expert
    # side (EPIC non-hc convention, epic_semantics.combine_pack_group),
    # returned over the SAME All2AllSingle pair with swapped splits
    # (max_split is a global max => transpose-invariant; no new symmetric
    # memory), and accumulated at the token home deterministically via the
    # comb_dst_slot permutation (index_copy_ into [S*K, H] staging + one
    # terminal view(S, K, H).sum(1)). EPLB has NO dedup, so there is no
    # reverse-dedup partial-sum step (moonep's dup_primary/dup_target).

    def enable_layer1(self):
        """Allocate the gemm2/combine buffers. Requires the faithful
        full-expert placement (--weight_place fc1fc2)."""
        assert self.place_fc2, (
            "l01 needs slot_fc2 resident: run with --weight_place fc1fc2"
        )
        cfg = self.cfg
        dev, H = self.device, cfg.H
        self.layers = "l01"
        self.act_buf = torch.zeros(
            max(self.n_recv, 1), self.ffn_size_shard, dtype=self.dtype,
            device=dev)
        self.comb_hidden_buf = torch.zeros(
            max(self.n_recv, 1), H, dtype=self.dtype, device=dev)
        self.comb_send_buf = torch.empty(
            max(self.n_recv, 1), H, dtype=self.dtype, device=dev)
        self.comb_recv_buf = torch.empty(
            cfg.S * cfg.K, H, dtype=self.dtype, device=dev)
        self.stage_buf = torch.empty(
            cfg.S * cfg.K, H, dtype=self.dtype, device=dev)
        self.final_out = torch.zeros(cfg.S, H, dtype=self.dtype, device=dev)

    def act(self):
        """Exact GELU on the native-dtype GEMM0 output (house convention)."""
        self.act_buf[:self.n_recv].copy_(
            torch.nn.functional.gelu(self.out_buf[:self.n_recv]))

    def gemm2(self, gemm_only_op):
        """Down-projection over the same segments, weights by local slot."""
        for p, start, end, _logical in self.lay.gemm_segments:
            gemm_only_op.forward(
                self.act_buf[start:end], self.slot_fc2[p],
                output_buf=self.comb_hidden_buf[start:end],
                fast_accum=False,
            )

    def combine_pack(self):
        """Gather gemm2 rows back to recv-stream order and fp32-scale by
        the per-entry route probs (slot-major weights_buf indexed through
        place_slots — the EPIC non-hc pack)."""
        rows = self.comb_hidden_buf.index_select(0, self.place_slots)
        scale = self.weights_buf.index_select(0, self.place_slots)
        self.comb_send_buf[:self.n_recv] = (
            rows.float() * scale.unsqueeze(1)).to(self.dtype)

    def combine_a2av(self):
        """Reverse wire: dispatch splits swapped on the same op pair."""
        if self.transport == "nvshmem":
            self._a2a_hidden.forward(
                self.comb_send_buf, self.comb_recv_buf,
                self._out_splits, self._in_splits, self._num_comm_sm,
            )
            return
        self.dist.all_to_all_single(
            self.comb_recv_buf, self.comb_send_buf,
            output_split_sizes=self.lay.send_counts,
            input_split_sizes=self.lay.recv_counts,
            group=self.group,
        )

    def combine_place_reduce(self):
        """Deterministic home accumulation: comb_dst_slot is a permutation
        of [0, S*K) (every (token, k) cell written exactly once), then one
        terminal sum over the k axis — bitwise-stable ordering."""
        assert self.comb_dst_slot is not None, "bind_iter_plan(want_comb)"
        self.stage_buf.index_copy_(0, self.comb_dst_slot, self.comb_recv_buf)
        self.final_out.copy_(
            self.stage_buf.view(self.cfg.S, self.cfg.K, self.cfg.H).sum(1))
        return self.final_out
