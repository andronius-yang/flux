# OURS — the fused PLACE-lambda + LocCap + Slipstream-v2 MoE layer0+1 arm.
#
# One integrated algorithm (fresh arm, 2026-08-24 campaign):
#   placement  = PLACE-lambda FAST (placelambda_fast), solved from the
#                scenario-1 pre-batch oracle (untimed; pre-gating input),
#   routing    = LocCap sender-local fused kernel (flux.placelambda_route_sl,
#                relaxed-determinism contract) + ONE fused phys+probs
#                allgather, per iteration, timed,
#   dispatch   = Slipstream dispatch: a2av hier_compress + LB_UNION Tier-B
#                windowed union broadcast + FUSED_STAGE2 + EARLY_LAUNCH +
#                WAVE_PACK, tile-overlapped with the grouped GEMM0 via the
#                FUSED GemmGroupedV2AGScatterOp.forward (NOT dispatch_only),
#   combine    = Slipstream v2: GemmGroupedV2GatherRSOp.forward_gather_rs
#                with M-split destination waves + epilogue-fused pack +
#                completion-bucketed register receiver, n_split=1.
#
# The whole thing runs in a virtual-slot space (gpe = nlp + 1 slots per
# rank, slot v owned by rank v // gpe), so the fused ops' contiguous
# ownership arithmetic holds by construction and derive_routed_meta(vce)
# produces every metadata tensor both ops need, consistent by construction.
#
# OURS slot convention (differs from epic's pad-LAST): the PAD slot is
# LOCAL INDEX 0 (vslot = owner*gpe + 1 + phys%nlp; pad = owner*gpe, zero
# rows at m=1). Rationale (scenario 2): the fused op's weight gate spins
# tiles of local groups >= weight_gate_group_start and requires start > 0
# — pad-first makes start=1 gate exactly the real slots, with signal index
# (group-1) aligning to the weight op's slot ids.
#
# Timing contract (SCHEMA rule 5 / rule-10 s1 canon):
#   plan_comm = the per-iteration d[R, G] demand allgather (the recurring
#               exchange that lets every rank build the LocCap tables);
#   plan      = route kernel + fused phys+probs allgather + vce build +
#               l0_op.derive_routed_meta (its pinned D2H event sync is the
#               honest in-window host sync) [+ derive_combine_meta when the
#               plan-overlap knob is OFF];
#   e2e       = fused l0 forward -> GELU -> fused l1 forward; the combine
#               metadata derive may OVERLAP the l0 forward on a side stream
#               (FLUX_OURS_PLAN_OVERLAP=1) — its cost then hides under the
#               dispatch wire and any residue shows up in l1_ms, honestly.
#   placement: scenario 1 = static oracle solve at setup, untimed, reported
#              (rule-5 placement amendment, place_dynamic=static).
#
# Wire rule (SCHEMA rules 6/7): every inter-node put on both ops is the
# blocking putmem_signal (binary defaults); correctness cells randomize the
# payload per iteration; the final deterministic iteration binds the torch
# loccap_route_sl reference routing and validates the full journey.

import os

import torch
import torch.distributed as dist

__all__ = ["OursIterPlan", "OursIterPlanner", "OursRunner"]


class OursIterPlan:
    """One iteration's routing products (relaxed kernel lane)."""

    __slots__ = ("vce", "probs_all", "kstats_pinned")

    def __init__(self, vce, probs_all, kstats_pinned):
        self.vce = vce                  # [ntokens, K] int32 device
        self.probs_all = probs_all      # [ntokens, K] fp32 device
        self.kstats_pinned = kstats_pinned  # [4] int64 pinned (async copy)


class OursIterPlanner:
    """Slim per-iteration planner for the fused arm.

    The staged epic fast tail (send order, recv canonical order, gemm
    segment lists, probs-wire indices) is NOT needed on the fused path:
    the l0 op's derive_routed_meta(vce) produces splits/scatter/sps/uc and
    the ops schedule themselves. What remains here is exactly:
      route own rows (fused kernel, relaxed) ->
      ONE allgather of (phys_own int32 | probs_own fp32-bitcast) ->
      vce = (phys // nlp) * gpe + phys % nlp.
    """

    def __init__(self, plan, rank, device, topk_all, probs_all_setup,
                 local_world_size, eps, f_cap, route_group):
        cfg = plan.cfg
        self.cfg = cfg
        self.plan = plan
        self.rank = rank
        self.device = device
        self.L = local_world_size
        self.nn = cfg.R // local_world_size
        self.gpe = cfg.nlp + 1
        self.E_virt = cfg.R * self.gpe
        self.eps = eps
        self.f_cap = f_cap
        self.route_group = route_group
        self.last_kernel_stats = None

        S, K, R = cfg.S, cfg.K, cfg.R
        self.topk_all = topk_all.long().to(device)
        self._topk_own_i32 = topk_all[rank].int().contiguous().to(device)
        # own gate weights (harness gating output; the EXCHANGE is timed)
        self._probs_own = (probs_all_setup[rank * S:(rank + 1) * S]
                          .float().contiguous().to(device))
        # one-hot demand accumulator (index_add — bincount hides syncs)
        self._d_own = torch.zeros(cfg.G, dtype=torch.int32, device=device)
        self._ones = torch.ones(S * K, dtype=torch.int32, device=device)
        # fused exchange buffers: [phys(S*K) i32 | probs(S*K) i32-bitcast]
        self._xchg_send = torch.empty(2 * S * K, dtype=torch.int32,
                                      device=device)
        self._xchg_gather = torch.empty(R * 2 * S * K, dtype=torch.int32,
                                        device=device)
        self._kstats_pinned = torch.zeros(4, dtype=torch.int64,
                                          pin_memory=True)
        self.refresh_placement()

    def refresh_placement(self):
        dev = self.device
        self.l2p = self.plan.l2p.to(dev)
        self.lcnts = self.plan.lcnts.to(dev)
        self.p2l = self.plan.p2l.long().to(dev)

    # -- plan_comm bracket ---------------------------------------------------

    def local_loads(self) -> torch.Tensor:
        """d[G] for this rank, sync-free (index_add, not bincount)."""
        self._d_own.zero_()
        self._d_own.index_add_(0, self._topk_own_i32.reshape(-1).long(),
                               self._ones)
        return self._d_own

    # -- plan bracket ----------------------------------------------------------

    def derive(self, d_gather_buf: torch.Tensor) -> OursIterPlan:
        """Relaxed kernel route + fused phys/probs allgather + vce build."""
        import flux
        cfg = self.cfg
        S, K, R = cfg.S, cfg.K, cfg.R
        phys_own, kstats = flux.placelambda_route_sl(
            self._topk_own_i32, d_gather_buf, self.l2p, self.lcnts,
            self.rank, cfg.nlp, self.L, self.eps, self.f_cap)
        self._kstats_pinned.copy_(kstats, non_blocking=True)
        self.last_kernel_stats = self._kstats_pinned  # read post-sync
        # fused exchange: own phys rows + own probs (bitcast) in ONE gather
        self._xchg_send[:S * K].copy_(phys_own.view(-1))
        self._xchg_send[S * K:].copy_(self._probs_own.view(-1).view(torch.int32))
        dist.all_gather_into_tensor(self._xchg_gather, self._xchg_send,
                                    group=self.route_group)
        buf = self._xchg_gather.view(R, 2 * S * K)
        phys_all = buf[:, :S * K].reshape(R * S, K)
        probs_all = (buf[:, S * K:].contiguous().view(torch.float32)
                     .reshape(R * S, K))
        # pad-FIRST slot convention (see module header)
        vce = ((phys_all // cfg.nlp) * self.gpe + 1 + phys_all % cfg.nlp)
        return OursIterPlan(vce.int(), probs_all, self._kstats_pinned)

    def derive_reference(self) -> OursIterPlan:
        """Deterministic vce from the setup torch loccap_route_sl routing
        (plan.phys_override) — the final-iteration correctness routing."""
        cfg = self.cfg
        phys = self.plan.phys_override.view(cfg.R * cfg.S, cfg.K).long().to(
            self.device)
        vce = ((phys // cfg.nlp) * self.gpe + 1 + phys % cfg.nlp).int()
        probs_all = self._gather_probs_reference()
        return OursIterPlan(vce, probs_all, None)

    def _gather_probs_reference(self):
        cfg = self.cfg
        S, K, R = cfg.S, cfg.K, cfg.R
        out = torch.empty(R * S * K, dtype=torch.float32, device=self.device)
        dist.all_gather_into_tensor(
            out, self._probs_own.view(-1), group=self.route_group)
        return out.view(R * S, K)


class OursRunner:
    """Owns the two fused ops, the slot-space weights, and the per-iteration
    buffers. The driver owns event brackets; methods here are phase-pure."""

    def __init__(self, tp_group, ep_group, nnodes, local_world_size,
                 cfg, ffn, dtype, recv_cap, sm_margin=1,
                 plan_overlap=None):
        import flux
        self.tp_group = tp_group
        self.rank = tp_group.rank()
        self.W = tp_group.size()
        self.nnodes = nnodes
        self.L = local_world_size
        self.cfg = cfg
        self.gpe = cfg.nlp + 1
        self.E_virt = self.W * self.gpe
        self.H = cfg.H
        self.ffn = ffn
        self.dtype = dtype
        self.recv_cap = int(recv_cap)
        self.sm_margin = sm_margin
        self.ntokens = cfg.R * cfg.S
        self.K = cfg.K
        self.M_all = self.ntokens * cfg.K
        if plan_overlap is None:
            # default OFF until the A/B proves it at 4n AND 16n (critique H3)
            plan_overlap = bool(int(os.environ.get(
                "FLUX_OURS_PLAN_OVERLAP", "0")))
        self.plan_overlap = plan_overlap

        self.ep_start = self.rank * self.gpe

        # slot -> logical expert (pad slot = -1)
        p2l = None  # filled by set_weights

        # fused l0 op: Slipstream dispatch identity (LB_UNION Tier-B) by
        # default. setdefault: the relay-identity arm (ours_l01_s1_ri)
        # pins LB_UNION=0 — on LocCap-placed traffic the node-union
        # broadcast's dedup assumption weakens (copies are already
        # rank-consolidated), and the 16n b32+ crossover vs llc suggested
        # per-rank-exact delivery wins there. The knob must be IN the env
        # BEFORE the ctor (it sizes recv regions and picks the wire).
        os.environ.setdefault("FLUX_A2AV_LB_UNION", "1")
        tp_env = flux.DistEnvTPWithEP(tp_group=tp_group, nnodes=nnodes,
                                      ep_group=ep_group)
        moe_args = flux.MoeArguments(
            max_ntokens=self.ntokens,
            hidden=cfg.H,
            ffn_hidden=ffn,
            nexperts=self.E_virt,
            topk=cfg.K,
            input_dtype=dtype,
            output_dtype=dtype,
        )
        self.l0_op = flux.GemmGroupedV2AGScatterOp(
            tp_env=tp_env, moe_args=moe_args,
            a2av_dispatch=True, a2av_hier_compress=True)

        # fused l1 op: Slipstream v2 combine (msplit+fused_pack+bucket are
        # binary defaults under compress && ns1)
        self.l1_op = flux.GemmGroupedV2GatherRSOp(
            tp_group, self.E_virt, self.M_all, cfg.H, cfg.K, dtype,
            1, self.W, 1,
            nnodes=nnodes, n_split=1,
            do_all_reduce=False, use_read_mode=False,
            a2av_hier=False, a2av_hier_compress=True)

        # per-iteration buffers (capacity mode: recv_cap rows)
        self.out_buf = torch.zeros(self.recv_cap, ffn, dtype=dtype,
                                   device="cuda")
        self.scale_buf = torch.zeros(self.recv_cap + 1, dtype=torch.float32,
                                     device="cuda")
        self._iota_scale_src = None  # lazy

        # plan-overlap side stream + events
        self._meta_stream = torch.cuda.Stream()
        self._meta_ev = torch.cuda.Event()

        # stashed per-iteration metadata
        self._sd = self._scd = self._sps = self._uc = None
        self._l1_kwargs = None
        self._m_this = 0

    # -- setup -----------------------------------------------------------------

    def set_weights(self, w1_slot: torch.Tensor, w2_slot: torch.Tensor):
        """Per-slot GEMM weights, virtual-slot space:
        w1_slot [gpe, ffn, H], w2_slot [gpe, H, ffn]; pad slot rows are
        zeros (never referenced: the pad slot has zero split at m=1)."""
        assert w1_slot.shape == (self.gpe, self.ffn, self.H)
        assert w2_slot.shape == (self.gpe, self.H, self.ffn)
        self.w1 = w1_slot.contiguous()
        self.w2 = w2_slot.contiguous()

    # -- timed phases ----------------------------------------------------------

    def plan_meta(self, ip: OursIterPlan):
        """Blocking plan tail: in-op derive of every dispatch metadata
        tensor (+ combine meta inline when plan_overlap is off). The
        derive's pinned D2H event sync is the honest host sync; sps/uc are
        host-readable after it returns."""
        sd, scd, sps, uc = self.l0_op.derive_routed_meta(ip.vce)
        self._sd, self._scd, self._sps, self._uc = sd, scd, sps, uc
        # rows this rank computes = column-sum of my slot block
        self._m_this = int(sps[:, self.ep_start:self.ep_start + self.gpe]
                           .sum())
        assert self._m_this <= self.recv_cap, (
            f"recv overflow: m_this {self._m_this} > recv_cap "
            f"{self.recv_cap} (kernel drift past the sized bound)")
        if ip.kstats_pinned is not None:
            assert int(ip.kstats_pinned[2]) == 0, (
                "loccap_sl forced-budget overflow (kstats[2] != 0)")
        if not self.plan_overlap:
            self._derive_combine(ip)

    def issue_combine_meta(self, ip: OursIterPlan):
        """Plan-overlap lane: launch the combine-meta derive + scale build
        on the side stream, gated on the (already synced) derive results;
        the l1 forward waits self._meta_ev."""
        if not self.plan_overlap:
            return
        self._main_stream = torch.cuda.current_stream()
        self._meta_stream.wait_stream(self._main_stream)
        with torch.cuda.stream(self._meta_stream):
            self._derive_combine(ip)
        self._meta_ev.record(self._meta_stream)

    def _derive_combine(self, ip: OursIterPlan):
        uc_l1 = (self._uc[:, self.W:].contiguous()
                 if self.nnodes > 1 else None)
        meta = self.l1_op.derive_combine_meta(
            self._sd, self._scd.view(-1), self._sps,
            a2av_unique_counts=uc_l1)
        if self.plan_overlap:
            # side-stream allocations consumed on the main stream by the l1
            # forward: pin lifetimes for the caching allocator (critique H3)
            for t in meta:
                t.record_stream(self._main_stream)
        l1k = {
            "splits_per_source": self._sps,
            "a2av_pack_index": meta[0],
            "a2av_reduce_index": meta[1],
        }
        if uc_l1 is not None:
            l1k["a2av_wire_csr"] = [meta[2], meta[3]]
            l1k["a2av_reduce_csr"] = [meta[4], meta[5]]
            l1k["a2av_unique_counts"] = uc_l1
        self._l1_kwargs = l1k
        self._build_scale(ip)

    def _build_scale(self, ip: OursIterPlan):
        """output_vec_scale per gemm row of THIS rank, sync-free:
        row r in my segment gets the gate weight of the (token, k) copy
        whose scatter_index landed there. Invalid entries route to the
        guard slot recv_cap (sliced away)."""
        sd_long = self._sd.long()
        m_start = sd_long[:self.ep_start].sum()  # device scalar
        idx = self._scd.view(-1).long() - m_start
        valid = (idx >= 0) & (idx < self.recv_cap)
        idx = torch.where(valid, idx,
                          torch.full_like(idx, self.recv_cap))
        self.scale_buf.zero_()
        self.scale_buf.scatter_(0, idx, ip.probs_all.view(-1))

    def l0_forward(self, inputs_shard: torch.Tensor, gate_kwargs=None):
        self.l0_op.forward(
            inputs_shard=inputs_shard,
            weights=self.w1,
            splits_gpu=self._sd,
            scatter_index=self._scd,
            outputs_buf=self.out_buf[:self._m_this],
            fast_accum=False,
            sm_margin=self.sm_margin,
            splits_per_source=self._sps,
            a2av_unique_counts=self._uc,
            **(gate_kwargs or {}),
        )
        return self.out_buf[:self._m_this]

    def l1_forward(self, intermediate: torch.Tensor):
        if self.plan_overlap:
            torch.cuda.current_stream().wait_event(self._meta_ev)
        return self.l1_op.forward_gather_rs(
            intermediate,
            self.w2,
            self._sd,
            self._scd.view(-1),
            output_vec_scale=self.scale_buf[:self._m_this],
            fast_accum=False,
            sm_margin=self.sm_margin,
            bias=None,
            **self._l1_kwargs,
        )

    def prep(self):
        """Per-iteration hygiene OUTSIDE the timed window."""
        self.out_buf.zero_()
        self.l0_op.clear_buffers()
