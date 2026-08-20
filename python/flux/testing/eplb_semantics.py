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

import time
from dataclasses import dataclass, replace as _dc_replace

import torch

from .ep_gpu_plan import (
    comb_dst_slot_from_topk,
    direct_layout_entries,
    largest_remainder_split,
    rank_quota_prefix_nonlocal,
    reroute_expand_all_gpu,
)
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
                    rebalance_fn) -> UltraEPPlan:
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

    rank_quota_prefix = torch.stack([
        build_rank_quota_prefix(cfg, tpe, l2p, lcnts, quota, src,
                                locality_aware=False)
        for src in range(cfg.R)
    ])

    return UltraEPPlan(
        cfg=cfg, tpe=tpe.to(torch.int32), p2l=p2l, l2p=l2p, lcnts=lcnts,
        quota=quota, quota_prefix=quota_prefix,
        rank_quota_prefix=rank_quota_prefix,
        domain_solutions=[],
    )


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
                 topk_all: torch.Tensor, want_comb: bool = False):
        cfg = plan.cfg
        self.cfg = cfg
        self.rank = rank
        self.device = device
        self.want_comb = want_comb
        # deployment-scope placement (rule-5 legal one-shot)
        self.l2p = plan.l2p.to(device)
        self.lcnts = plan.lcnts.to(device)
        self.p2l = plan.p2l.long().to(device)
        self._p2l_host = plan.p2l.long()
        # gating metadata (the rule-5 exempt input): replicated routing
        self.topk_all = topk_all.long().to(device)

    def local_loads(self) -> torch.Tensor:
        """[G] int32 this-rank load histogram — the plan_comm payload,
        derived from routing per iteration (timed)."""
        return torch.bincount(self.topk_all[self.rank].reshape(-1),
                              minlength=self.cfg.G).to(torch.int32)

    def derive(self, loads_gather_buf: torch.Tensor) -> EplbIterPlan:
        cfg = self.cfg
        R, G, S, K, nlp = cfg.R, cfg.G, cfg.S, cfg.K, cfg.nlp
        tpe_all = loads_gather_buf.view(R, G)

        quota, _ = largest_remainder_split(
            tpe_all.long().sum(0), self.lcnts, cfg.max_replicas_dim)
        rqp = rank_quota_prefix_nonlocal(tpe_all, quota, self.lcnts)
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
                 ffn_size_shard: int = 0, place_fc2: bool = True):
        # ffn_size_shard=0 makes the parent skip out_buf/replica_fc1/
        # replica_fc2; slot-indexed buffers replace them below.
        super().__init__(plan, rank, group, device, topk_all, dtype=dtype,
                         ffn_size_shard=0, sync_fc2=False)
        assert ffn_size_shard > 0
        self.ffn_size_shard = ffn_size_shard
        self.place_fc2 = place_fc2
        cfg = self.cfg
        self.out_buf = torch.zeros(
            max(self.n_recv, 1), ffn_size_shard, dtype=dtype, device=device
        )
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
