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
#
# Plan-lane cost knobs (2026-08-25 16n plan-gap attack; EACH default OFF,
# rule-5 legal — nothing caches plan CONTENT, only buffers/programs):
#   FLUX_OURS_PLAN_XCHG_NARROW  0: int32 phys | fp32-bitcast probs wire
#                                  (8 B/route entry, legacy);
#                               1: int16 phys + fp32 probs bit-split
#                                  (6 B/entry, LOSSLESS, needs P<=32767);
#                               2: int16 phys + bf16 probs (4 B/entry —
#                                  llc wire parity; LOSSY probs rounding
#                                  ~2^-8 rel: allclose gates hold, out_sha
#                                  changes -> never-mix vs narrow<2).
#   FLUX_OURS_PLAN_PREALLOC     1: persistent probs/vce/phys tail buffers
#                                  + out=-form vce arithmetic (no per-iter
#                                  allocs in the derive tail). Implied by
#                                  NARROW>=1 and PLAN_GRAPH.
#   FLUX_OURS_PLAN_GRAPH        1: CUDA-graph the post-allgather derive
#                                  tail (probs recovery + vce build) —
#                                  capture-once/replay on the persistent
#                                  buffers (routing VALUES stream through
#                                  the allgather every iteration; the llc
#                                  FLUX_PLL_TAIL_GRAPH precedent), eager
#                                  fallback on capture failure.
#   FLUX_OURS_PLAN_SCALE_GRAPH  1: CUDA-graph the output_vec_scale build
#                                  (static shapes; inputs are the C++
#                                  persistent rt_* buffers + the planner
#                                  probs buffer — pointer-guarded replay,
#                                  eager for reference/setup-audit calls).

import os
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist

__all__ = ["OursIterPlan", "OursIterPlanner", "OursRunner"]

# FLUX_OURS_NVTX=1: sub-phase NVTX ranges inside the plan lane (host-side
# spans; nsys correlates the launched kernels/collectives). Diagnostic-only
# knob — default OFF so record capsules are untouched.
_NVTX = bool(int(os.environ.get("FLUX_OURS_NVTX", "0")))


@contextmanager
def _nvtx_range(tag):
    torch.cuda.nvtx.range_push(tag)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _nvtx(tag):
    return _nvtx_range(tag) if _NVTX else nullcontext()


class OursIterPlan:
    """One iteration's routing products (relaxed kernel lane)."""

    __slots__ = ("vce", "probs_all", "kstats_pinned", "stable")

    def __init__(self, vce, probs_all, kstats_pinned, stable=False):
        self.vce = vce                  # [ntokens, K] int32 device
        self.probs_all = probs_all      # [ntokens, K] fp32 device
        self.kstats_pinned = kstats_pinned  # [4] int64 pinned (async copy)
        # stable=True: vce/probs_all live in the planner's PERSISTENT
        # buffers (same pointers every iteration) — the scale-graph
        # capture/replay precondition. Reference/setup-audit plans are
        # stable=False and always take the eager scale path.
        self.stable = stable


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
                 local_world_size, eps, f_cap, route_group,
                 route_global=False):
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
        # r2/s2 contract fix (2026-08-26, handoff 22 §4): under live
        # re-placement the runtime-adopted placement's forced geometry can
        # exceed ANY setup-derived f_cap (the "oscillates between exactly
        # two placements" sizing premise is false at r2). The kernel's
        # forced_left tickets are per-call workspace ints (f_cap<=0 =>
        # INT_MAX/2), so a rank-LOCAL escalate-and-reroute BEFORE the
        # phys/probs exchange is sound: route is sender-local, nothing
        # persistent is sized by f_cap, and the recv_cap assert in
        # plan_meta stays the loud sizing backstop. Enabled by the driver
        # for s2 only (adds one early device sync per iteration).
        self.f_cap_retry = False
        self.f_cap_retries_total = 0
        self.f_cap_current = f_cap
        self.route_group = route_group
        self.last_kernel_stats = None
        # route-global (2026-08-29 restructure, handoff 26 §4): ONE
        # topk+probs allgather replaces the d-allgather + routed-decisions
        # allgather pair; every rank recomputes every rank's assignment
        # with the deterministic quota route (route_global_quota — the
        # relaxed kernel's atomic-ticket pairing cannot be replicated
        # cross-rank). Requires narrow==0 buffers and the prealloc tail.
        self.route_global = bool(route_global)

        S, K, R = cfg.S, cfg.K, cfg.R
        self.topk_all = topk_all.long().to(device)
        self._topk_own_i32 = topk_all[rank].int().contiguous().to(device)
        # own gate weights (harness gating output; the EXCHANGE is timed)
        self._probs_own = (probs_all_setup[rank * S:(rank + 1) * S]
                          .float().contiguous().to(device))
        # one-hot demand accumulator (index_add — bincount hides syncs)
        self._d_own = torch.zeros(cfg.G, dtype=torch.int32, device=device)
        self._ones = torch.ones(S * K, dtype=torch.int32, device=device)
        # plan-lane knobs (module header): all default OFF
        self.xchg_narrow = int(os.environ.get(
            "FLUX_OURS_PLAN_XCHG_NARROW", "0"))
        self.plan_graph = bool(int(os.environ.get(
            "FLUX_OURS_PLAN_GRAPH", "0")))
        self.plan_prealloc = (self.plan_graph or self.xchg_narrow > 0
                              or bool(int(os.environ.get(
                                  "FLUX_OURS_PLAN_PREALLOC", "0"))))
        if self.xchg_narrow:
            assert cfg.R * cfg.nlp <= 32767, (
                f"XCHG_NARROW needs P = R*nlp <= 32767 (int16 phys), got "
                f"{cfg.R * cfg.nlp}")
            # narrow 1: the strided [R, 2*S*K] i16 probs slice must view
            # as fp32 (row stride divisible by the dtype ratio)
            assert S * K % 2 == 0, "XCHG_NARROW=1 needs S*K even"
        # fused exchange buffers, ONE allgather either way:
        #   narrow 0: i32 [phys(S*K) | probs fp32-bitcast (S*K)]
        #   narrow 1: i16 [phys(S*K) | probs fp32 bit-split (2*S*K)]
        #   narrow 2: i16 [phys(S*K) | probs bf16-bitcast (S*K)]
        if self.xchg_narrow:
            per = 3 * S * K if self.xchg_narrow == 1 else 2 * S * K
            self._xchg_send = torch.empty(per, dtype=torch.int16,
                                          device=device)
            self._xchg_gather = torch.empty(R * per, dtype=torch.int16,
                                            device=device)
        else:
            self._xchg_send = torch.empty(2 * S * K, dtype=torch.int32,
                                          device=device)
            self._xchg_gather = torch.empty(R * 2 * S * K, dtype=torch.int32,
                                            device=device)
        if self.plan_prealloc:
            # persistent derive-tail buffers (buffer prealloc, not plan
            # caching: contents are fully rewritten every iteration)
            self._vce_buf = torch.empty(R * S, K, dtype=torch.int32,
                                        device=device)
            self._vce_tmp = torch.empty(R * S, K, dtype=torch.int32,
                                        device=device)
            self._probs_all_buf = torch.empty(R * S, K, dtype=torch.float32,
                                              device=device)
            self._phys_i32 = (torch.empty(R * S, K, dtype=torch.int32,
                                          device=device)
                              if self.xchg_narrow else None)
            # static-content index prep for local_loads (the topk trace is
            # the fixed per-cell gating input; the CAST is pure overhead)
            self._topk_own_flat_i64 = (self._topk_own_i32.reshape(-1)
                                       .long().contiguous())
        self._tail_graph = None
        self._tail_graph_broken = False
        self._kstats_pinned = torch.zeros(4, dtype=torch.int64,
                                          pin_memory=True)
        if self.route_global:
            assert self.xchg_narrow == 0, (
                "route_global v1 requires narrow==0 (topk rides the phys "
                "slot of the fused exchange)")
            if not self.plan_prealloc:
                self.plan_prealloc = True
                self._vce_buf = torch.empty(R * S, K, dtype=torch.int32,
                                            device=device)
                self._vce_tmp = torch.empty(R * S, K, dtype=torch.int32,
                                            device=device)
                self._probs_all_buf = torch.empty(
                    R * S, K, dtype=torch.float32, device=device)
                self._phys_i32 = None
                self._topk_own_flat_i64 = (self._topk_own_i32.reshape(-1)
                                           .long().contiguous())
            self._rg_stats = torch.zeros(4, dtype=torch.int64,
                                         device=device)
            # contiguous topk staging (the gathered buffer's topk slice is
            # strided; the kernel wants [R, S, K] contiguous)
            self._rg_topk_buf = torch.empty(R, S, K, dtype=torch.int32,
                                            device=device)
            self._rg_check = bool(int(os.environ.get(
                "FLUX_OURS_RG_CHECK", "0")))
            self._rg_graph = None
            self._rg_graph_broken = False
        self.refresh_placement()

    def refresh_placement(self):
        dev = self.device
        self.l2p = self.plan.l2p.to(dev)
        self.lcnts = self.plan.lcnts.to(dev)
        self.p2l = self.plan.p2l.long().to(dev)
        if self.route_global:
            from flux.testing.placelambda_gpu import instance_tables_gpu
            self._rg_tables = instance_tables_gpu(
                self.l2p, self.lcnts, self.cfg.nlp, self.L, R=self.cfg.R)

    # -- plan_comm bracket ---------------------------------------------------

    def local_loads(self) -> torch.Tensor:
        """d[G] for this rank, sync-free (index_add, not bincount)."""
        self._d_own.zero_()
        idx = (self._topk_own_flat_i64 if self.plan_prealloc
               else self._topk_own_i32.reshape(-1).long())
        self._d_own.index_add_(0, idx, self._ones)
        return self._d_own

    def rg_exchange(self):
        """Route-global plan_comm: the ONE recurring collective — raw
        topk + probs in the fused buffer (byte-identical layout to the
        legacy phys+probs exchange; topk rides the phys slot)."""
        S, K = self.cfg.S, self.cfg.K
        with _nvtx("plan.rg_exchange"):
            self._xchg_send[:S * K].copy_(self._topk_own_i32.view(-1))
            self._xchg_send[S * K:].copy_(
                self._probs_own.view(-1).view(torch.int32))
            dist.all_gather_into_tensor(self._xchg_gather, self._xchg_send,
                                        group=self.route_group)

    def _rg_compute(self):
        """Route-global derive: the deterministic quota route KERNEL
        (flux.placelambda_route_global — stable ordinals vs closed-form
        windows; bitwise spec = placelambda_gpu.route_global_quota) over
        ALL ranks from the gathered topk + vce/probs recovery, entirely
        into persistent buffers. Sync-free, static shapes —
        graph-capturable (outputs consumed only via persistent buffers,
        so the graph-pool pointers never escape)."""
        import flux
        cfg = self.cfg
        S, K, R = cfg.S, cfg.K, cfg.R
        n = S * K
        buf = self._xchg_gather.view(R, 2 * n)
        self._rg_topk_buf.view(R, n).copy_(buf[:, :n])
        self._probs_all_buf.view(R, n).copy_(
            buf[:, n:].view(torch.float32))
        phys, stats = flux.placelambda_route_global(
            self._rg_topk_buf, self.l2p, self.lcnts,
            int(self.lcnts.numel()), cfg.nlp, self.L, self.eps,
            self.f_cap_current if self.f_cap_current else 0)
        self._rg_stats.copy_(stats)
        phys = phys.view(R, n)
        torch.remainder(phys, cfg.nlp, out=self._vce_tmp.view(R, n))
        torch.div(phys, cfg.nlp, rounding_mode="floor",
                  out=self._vce_buf.view(R, n))
        self._vce_buf.mul_(self.gpe).add_(1).add_(self._vce_tmp)

    def _derive_rg(self) -> OursIterPlan:
        with _nvtx("plan.rg_route"):
            if self.plan_graph and not self._rg_graph_broken:
                if self._rg_graph is None:
                    try:
                        for _ in range(2):
                            self._rg_compute()
                        torch.cuda.synchronize()
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            self._rg_compute()
                        g.replay()
                        torch.cuda.synchronize()
                        self._rg_graph = g
                    except Exception as e:  # noqa: BLE001 — eager fallback
                        self._rg_graph = None
                        self._rg_graph_broken = True
                        if self.rank == 0:
                            print(f"[ours] rg graph capture failed "
                                  f"({type(e).__name__}: {e}); eager",
                                  flush=True)
                if self._rg_graph is not None:
                    self._rg_graph.replay()
                else:
                    self._rg_compute()
            else:
                self._rg_compute()
        if self._rg_check:
            # gate-only kernel-vs-spec identity (FLUX_OURS_RG_CHECK=1):
            # the torch quota route recomputed from the same gathered
            # buffer must produce the identical vce, every iteration
            from flux.testing.placelambda_gpu import route_global_quota
            cfg = self.cfg
            n = cfg.S * cfg.K
            phys_ref = route_global_quota(
                self._rg_topk_buf, self._rg_tables, cfg.nlp, self.L,
                self.eps, f_cap=self.f_cap_current or 0).view(cfg.R, n)
            vce_ref = ((phys_ref.long() // cfg.nlp) * self.gpe + 1
                       + phys_ref.long() % cfg.nlp).int()
            if not torch.equal(self._vce_buf.view(cfg.R, n), vce_ref):
                bad = (self._vce_buf.view(cfg.R, n) != vce_ref)
                nb = int(bad.sum())
                src, pos = [int(x) for x in bad.nonzero()[0]]
                g = int(self._rg_topk_buf.view(cfg.R, n)[src, pos])
                o_k = int(self._vce_buf.view(cfg.R, n)[src, pos])
                o_r = int(vce_ref[src, pos])
                bsrc = torch.unique(bad.nonzero()[:, 0]).tolist()[:8]
                dump = os.environ.get("FLUX_OURS_RG_DUMP")
                if dump and self.rank == 0:
                    torch.save(
                        {"topk": self._rg_topk_buf.cpu(),
                         "l2p": self.l2p.cpu(), "lcnts": self.lcnts.cpu(),
                         "nlp": cfg.nlp, "L": self.L, "eps": self.eps,
                         "f_cap": self.f_cap_current or 0}, dump)
                raise AssertionError(
                    f"route_global kernel != torch spec: {nb} bad; first "
                    f"(src {src}, pos {pos}, g {g}): kernel vce {o_k} vs "
                    f"ref {o_r}; bad srcs {bsrc}")
        self._kstats_pinned.copy_(self._rg_stats, non_blocking=True)
        self.last_kernel_stats = self._kstats_pinned
        return OursIterPlan(self._vce_buf, self._probs_all_buf,
                            self._kstats_pinned, stable=True)

    # -- plan bracket ----------------------------------------------------------

    def derive(self, d_gather_buf: torch.Tensor) -> OursIterPlan:
        """Relaxed kernel route + fused phys/probs allgather + vce build.
        Route-global mode: the exchange already ran (rg_exchange, in the
        plan_comm bracket) — derive is the deterministic global route."""
        if self.route_global:
            return self._derive_rg()
        import flux
        cfg = self.cfg
        S, K, R = cfg.S, cfg.K, cfg.R
        with _nvtx("plan.route"):
            phys_own, kstats = flux.placelambda_route_sl(
                self._topk_own_i32, d_gather_buf, self.l2p, self.lcnts,
                self.rank, cfg.nlp, self.L, self.eps, self.f_cap_current)
        if self.f_cap_retry:
            # forced-budget breach check + local escalate-and-reroute
            # (kstats[2] = forced_budget_overflow). Escalation ladder:
            # 4x, then uncapped (kernel INT_MAX tickets). The raised cap
            # STICKS (self.f_cap_current) — adopted-placement geometry is
            # persistent, re-deriving it every iteration would pay the
            # sync-retry twice for nothing.
            for esc in (4 * max(self.f_cap_current, 1), 0):
                with _nvtx("plan.kstats_sync"):
                    breach = int(kstats[2].item()) != 0
                if not breach:
                    break
                self.f_cap_current = esc
                self.f_cap_retries_total += 1
                phys_own, kstats = flux.placelambda_route_sl(
                    self._topk_own_i32, d_gather_buf, self.l2p,
                    self.lcnts, self.rank, cfg.nlp, self.L, self.eps,
                    self.f_cap_current)
        self._kstats_pinned.copy_(kstats, non_blocking=True)
        self.last_kernel_stats = self._kstats_pinned  # read post-sync
        # fused exchange: own phys rows + own probs (bitcast) in ONE gather
        with _nvtx("plan.xchg_pack"):
            self._xchg_send[:S * K].copy_(phys_own.view(-1))
            if self.xchg_narrow == 1:
                self._xchg_send[S * K:].copy_(
                    self._probs_own.view(-1).view(torch.int16))
            elif self.xchg_narrow == 2:
                self._xchg_send[S * K:].copy_(
                    self._probs_own.view(-1).to(torch.bfloat16)
                    .view(torch.int16))
            else:
                self._xchg_send[S * K:].copy_(
                    self._probs_own.view(-1).view(torch.int32))
        with _nvtx("plan.xchg_allgather"):
            if self.xchg_narrow:
                # NCCL rejects int16 (Short); ship the same bytes as int8
                # views
                dist.all_gather_into_tensor(
                    self._xchg_gather.view(torch.int8),
                    self._xchg_send.view(torch.int8),
                    group=self.route_group)
            else:
                dist.all_gather_into_tensor(self._xchg_gather,
                                            self._xchg_send,
                                            group=self.route_group)
        with _nvtx("plan.vce_tail"):
            if self.plan_prealloc:
                vce, probs_all = self._derive_tail()
                return OursIterPlan(vce, probs_all, self._kstats_pinned,
                                    stable=True)
            buf = self._xchg_gather.view(R, 2 * S * K)
            phys_all = buf[:, :S * K].reshape(R * S, K)
            probs_all = (buf[:, S * K:].contiguous().view(torch.float32)
                         .reshape(R * S, K))
            # pad-FIRST slot convention (see module header)
            vce = ((phys_all // cfg.nlp) * self.gpe + 1 + phys_all % cfg.nlp)
            return OursIterPlan(vce.int(), probs_all, self._kstats_pinned)

    # -- prealloc'd/graphable derive tail (post-allgather device program) ----

    def _tail_compute(self):
        """Probs recovery + pad-FIRST vce build, entirely into the
        persistent buffers (no allocations, no host syncs — the graphable
        device program). Bitwise-identical values to the legacy tail for
        narrow in {0, 1}; narrow==2 recovers bf16-rounded probs."""
        cfg = self.cfg
        S, K, R = cfg.S, cfg.K, cfg.R
        n = S * K
        buf = self._xchg_gather.view(R, -1)
        phys = buf[:, :n]
        if self.xchg_narrow:
            self._phys_i32.view(R, n).copy_(phys)  # i16 -> i32 upcast
            phys = self._phys_i32.view(R, n)
        if self.xchg_narrow == 2:
            self._probs_all_buf.view(R, n).copy_(
                buf[:, n:].view(torch.bfloat16))
        else:  # narrow 1 bit-splits fp32 across i16 pairs; 0 bitcasts i32
            self._probs_all_buf.view(R, n).copy_(
                buf[:, n:].view(torch.float32))
        torch.remainder(phys, cfg.nlp, out=self._vce_tmp.view(R, n))
        torch.div(phys, cfg.nlp, rounding_mode="floor",
                  out=self._vce_buf.view(R, n))
        self._vce_buf.mul_(self.gpe).add_(1).add_(self._vce_tmp)

    def _derive_tail(self):
        if self.plan_graph and not self._tail_graph_broken:
            if self._tail_graph is None:
                self._capture_tail_graph()
            if self._tail_graph is not None:
                self._tail_graph.replay()
                return self._vce_buf, self._probs_all_buf
        self._tail_compute()
        return self._vce_buf, self._probs_all_buf

    def _capture_tail_graph(self):
        """Capture-once on the persistent buffers (input _xchg_gather and
        every output pointer is stable by construction — in-place/out=
        only, so no graph-pool memory escapes to other streams). Eager
        fallback on any capture failure, llc tail-graph style."""
        try:
            for _ in range(2):  # warmup (lazy module loads, allocator)
                self._tail_compute()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._tail_compute()
            g.replay()
            torch.cuda.synchronize()
            self._tail_graph = g
        except Exception as e:  # noqa: BLE001 — eager fallback
            self._tail_graph = None
            self._tail_graph_broken = True
            if self.rank == 0:
                print(f"[ours] plan tail graph capture failed "
                      f"({type(e).__name__}: {e}); eager", flush=True)

    def prime_graphs(self):
        """Capture the plan tail graph at SETUP inside a rank-quiesced
        region (canon-regen 8/29 finding: lazy first-use capture
        mid-iteration SIGABRTs large-budget cells — torch's pre-capture
        sync waits on peers' in-flight NCCL/NVSHMEM work that needs this
        rank; b32/b64 volumes expose it). No-op when graphs are off or
        already captured; the capture records ops only, buffer VALUES are
        irrelevant."""
        if (self.plan_graph and self.plan_prealloc
                and not self._tail_graph_broken
                and self._tail_graph is None):
            self._capture_tail_graph()

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
            plan_overlap = int(os.environ.get(
                "FLUX_OURS_PLAN_OVERLAP", "0"))
        # 0 = inline (combine meta in the plan bracket); 1 = pre-l0 side
        # stream (8/25 structure — measured to relabel plan_ms into l0_ms);
        # 2 = late issue after the l0 enqueue (host meta work runs under
        # the executing GEMM, kernels on the sm_margin headroom)
        self.plan_overlap = int(plan_overlap)

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
        # FLUX_OURS_PLAN_SCALE_GRAPH (module header): CUDA-graph the scale
        # build. Static shapes; inputs are the C++ persistent rt_* buffers
        # (stable storage across derive_routed_meta calls) + the planner's
        # persistent probs buffer (ip.stable) — replay is pointer-guarded,
        # reference/setup-audit calls stay eager.
        self.scale_graph = bool(int(os.environ.get(
            "FLUX_OURS_PLAN_SCALE_GRAPH", "0")))
        self._scale_graph_obj = None
        self._scale_graph_broken = False
        self._scale_graph_key = None

        # plan-overlap side stream + events
        self._meta_stream = torch.cuda.Stream()
        self._meta_ev = torch.cuda.Event()
        self._plan_ev = torch.cuda.Event()  # derive-done marker (mode 2)

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
        with _nvtx("plan.derive_routed_meta"):
            sd, scd, sps, uc = self.l0_op.derive_routed_meta(ip.vce)
        self._sd, self._scd, self._sps, self._uc = sd, scd, sps, uc
        # rows this rank computes = column-sum of my slot block
        with _nvtx("plan.m_this_host"):
            self._m_this = int(sps[:, self.ep_start:self.ep_start
                                   + self.gpe].sum())
        assert self._m_this <= self.recv_cap, (
            f"recv overflow: m_this {self._m_this} > recv_cap "
            f"{self.recv_cap} (kernel drift past the sized bound)")
        if ip.kstats_pinned is not None:
            assert int(ip.kstats_pinned[2]) == 0, (
                "loccap_sl forced-budget overflow (kstats[2] != 0)")
        if self.plan_overlap == 2:
            # late-issue mode: mark derive-results readiness so the side
            # stream can start the combine meta DURING the l0 GEMM (waiting
            # this event, NOT wait_stream — wait_stream after the l0 enqueue
            # would order the meta kernels behind the whole GEMM)
            self._plan_ev.record(torch.cuda.current_stream())
        if not self.plan_overlap:
            self._derive_combine(ip)

    def issue_combine_meta(self, ip: OursIterPlan, late: bool = False):
        """Plan-overlap lane: launch the combine-meta derive + scale build
        on the side stream, gated on the (already synced) derive results;
        the l1 forward waits self._meta_ev.

        Mode 1 (late=False call site, pre-l0): side stream chains after the
        current stream — the 8/25 structure; host issue cost lands before
        the l0 launches (measured: relabels plan_ms into l0_ms).
        Mode 2 (late=True call site, post-l0-enqueue): the HOST meta work
        runs while the GPU executes l0; the side stream waits only the
        plan-derive event, so the meta kernels execute concurrently with
        the GEMM on the sm_margin headroom."""
        if self.plan_overlap == 2 and late:
            self._main_stream = torch.cuda.current_stream()
            self._meta_stream.wait_event(self._plan_ev)
            with torch.cuda.stream(self._meta_stream):
                self._derive_combine(ip)
            self._meta_ev.record(self._meta_stream)
            return
        if self.plan_overlap != 1 or late:
            return
        self._main_stream = torch.cuda.current_stream()
        self._meta_stream.wait_stream(self._main_stream)
        with torch.cuda.stream(self._meta_stream):
            self._derive_combine(ip)
        self._meta_ev.record(self._meta_stream)

    def _derive_combine(self, ip: OursIterPlan):
        uc_l1 = (self._uc[:, self.W:].contiguous()
                 if self.nnodes > 1 else None)
        with _nvtx("plan.combine_meta_op"):
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
        with _nvtx("plan.scale_build"):
            self._build_scale(ip)

    def _build_scale(self, ip: OursIterPlan):
        """output_vec_scale per gemm row of THIS rank, sync-free:
        row r in my segment gets the gate weight of the (token, k) copy
        whose scatter_index landed there. Invalid entries route to the
        guard slot recv_cap (sliced away)."""
        if self.scale_graph and not self._scale_graph_broken \
                and getattr(ip, "stable", False):
            key = (self._sd.data_ptr(), self._scd.data_ptr(),
                   ip.probs_all.data_ptr())
            if self._scale_graph_obj is None:
                self._capture_scale_graph(ip, key)
            if (self._scale_graph_obj is not None
                    and key == self._scale_graph_key):
                self._scale_graph_obj.replay()
                return
        self._scale_compute(ip)

    def _scale_compute(self, ip: OursIterPlan):
        """The scale-build device program (shape-static, sync-free,
        in-place into the persistent scale_buf — graph-capturable)."""
        sd_long = self._sd.long()
        m_start = sd_long[:self.ep_start].sum()  # device scalar
        idx = self._scd.view(-1).long() - m_start
        valid = (idx >= 0) & (idx < self.recv_cap)
        idx = torch.where(valid, idx,
                          torch.full_like(idx, self.recv_cap))
        self.scale_buf.zero_()
        self.scale_buf.scatter_(0, idx, ip.probs_all.view(-1))

    def _capture_scale_graph(self, ip: OursIterPlan, key):
        """Capture-once on the stable input pointers; the only output is
        the in-place persistent scale_buf, so no graph-pool memory escapes
        the bracket. Eager fallback on any capture failure."""
        try:
            for _ in range(2):  # warmup on the live buffers (idempotent)
                self._scale_compute(ip)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._scale_compute(ip)
            g.replay()
            torch.cuda.synchronize()
            self._scale_graph_obj = g
            self._scale_graph_key = key
        except Exception as e:  # noqa: BLE001 — eager fallback
            self._scale_graph_obj = None
            self._scale_graph_broken = True
            if self.rank == 0:
                print(f"[ours] scale graph capture failed "
                      f"({type(e).__name__}: {e}); eager", flush=True)

    def prime_scale_graph(self, planner):
        """Setup-time scale-graph capture (same quiesced-region rationale
        as OursIterPlanner.prime_graphs). Requires a prior plan_meta (the
        setup audit) so _sd/_scd point at the op's persistent buffers;
        the stub plan reuses the planner's persistent vce/probs buffers,
        so the pointer key matches every runtime stable plan."""
        if not (self.scale_graph and not self._scale_graph_broken
                and self._scale_graph_obj is None):
            return
        if getattr(self, "_sd", None) is None:
            return
        if not getattr(planner, "plan_prealloc", False):
            return
        ip = OursIterPlan(planner._vce_buf, planner._probs_all_buf, None,
                          stable=True)
        self._build_scale(ip)

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
