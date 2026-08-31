# OURS-DIRECT — transport ablation runner (2026-08-30, 16n low-budget
# diagnosis): the OURS plan lane UNCHANGED (PLACE/pv2 placement, LocCap
# routing, OursIterPlanner's fused phys+probs exchange), but the WIRE is
# the eplb_l01 arm's staged direct path — the transport class that owns
# 16n b1/b2 in the handoff-18 ledger (EPLB l0 3.33 ms vs ours 7.07 at K2
# 16n b1; the gap is the hier_compress/slipstream stage chain's fixed
# serialization cost, not bytes).
#
# Mechanism, verbatim from eplb_semantics/UltraEPLayer0Runner:
#   dispatch = pack (index_select rows to (dest,phys,token) wire order)
#              -> flux.All2AllSingle NVSHMEM one-sided a2av (hidden +
#                 one fp32 route-prob per row; one putmem_nbi_block per
#                 destination + two team barriers, NO signal gating —
#                 CLAUDE.md invariant 5 does not bite: nothing consumes
#                 a per-put signal)
#              -> place (index_copy into per-slot segments)
#              -> per-segment GemmOnly GEMM0 (un-overlapped, EPLB-class)
#   combine  = per-segment GemmOnly GEMM2 -> combine-pack (recv-stream
#              order, fp32 prob scale at the expert side) -> the SAME
#              All2AllSingle pair with swapped splits -> deterministic
#              home accumulation (comb_dst permutation + one K-axis sum).
#
# Plan adapter (timed, inside the driver's plan bracket — rule 5): the
# planner's vce [ntokens, K] is the ONLY input; phys recovery + the
# dest-major stable sort + direct_layout_entries_fast (the eplb fast-tail
# layout) + ONE batched pinned D2H produce splits / segments / placement
# scatter / combine slots. No plan content crosses iterations.
#
# Scope: scenario-1 only (the s2 movement lane is coupled to the fused
# ops' WPM symmetric buffers). BF16, nvshmem transport only.

import os

import torch

from .ep_gpu_plan import direct_layout_entries_fast

__all__ = ["OursDirectRunner", "OursDirectOverlapRunner"]


class OursDirectRunner:
    """Phase-pure runner with the OursRunner driver interface
    (plan_meta / issue_combine_meta / l0_forward / l1_forward / prep /
    prime_scale_graph / set_weights), so test_moe_ours_traffic's timed
    loop runs unmodified: l0_ms = pack+wire+place+GEMM0, l1_ms =
    GEMM2+cpack+comb+acc, plan_ms includes the adapter derive."""

    def __init__(self, tp_group, rank, local_world_size, cfg, ffn, dtype,
                 recv_cap, max_split, probs_own, num_comm_sm=None,
                 device="cuda"):
        import flux  # GPU-side only

        self.rank = rank
        self.L = local_world_size
        self.cfg = cfg
        S, K, R, H = cfg.S, cfg.K, cfg.R, cfg.H
        self.S, self.K, self.W, self.H = S, K, R, H
        self.nlp = cfg.nlp
        self.gpe = cfg.nlp + 1
        self.N = S * K            # every route entry is one wire row
        self.ffn = ffn
        self.dtype = dtype
        self.recv_cap = int(recv_cap)
        self.max_split = int(max_split)
        self.num_comm_sm = int(
            num_comm_sm if num_comm_sm is not None
            else os.environ.get("FLUX_OURS_DWIRE_COMM_SM", "8"))
        dev = torch.device(device)
        self.device = dev

        # the eplb staged wire: one op pair, ctor collective (setup only)
        self._a2a_hidden = flux.All2AllSingle(
            tp_group, self.max_split, H, local_world_size, dtype)
        self._a2a_probs = flux.All2AllSingle(
            tp_group, self.max_split, 1, local_world_size, torch.float32)
        self._gemm_only = flux.GemmOnly(dtype, dtype, dtype,
                                        use_fp8_gemm=False)

        # static templates for the adapter (gating-metadata shapes)
        self._tok_t = torch.arange(S, device=dev,
                                   dtype=torch.int64).repeat_interleave(K)
        self._k_t = torch.arange(K, device=dev, dtype=torch.int64).repeat(S)
        self._probs_own_flat = (probs_own.reshape(-1).float()
                                .contiguous().to(dev))
        assert self._probs_own_flat.numel() == self.N
        # ONE batched pinned D2H per plan (eplb fast-tail convention):
        # nlp seg_rows + nlp seg_start + R in_splits + R out_splits +
        # pair_max + n_recv
        self._blob_pin = torch.empty(
            2 * cfg.nlp + 2 * R + 2, dtype=torch.int64,
            pin_memory=torch.cuda.is_available())

        # buffers (capacity-mode recv side, exact send side)
        self.send_buf = torch.empty(self.N, H, dtype=dtype, device=dev)
        self.wsend_buf = torch.empty(self.N, dtype=torch.float32, device=dev)
        self.recv_buf = torch.empty(self.recv_cap, H, dtype=dtype,
                                    device=dev)
        self.wrecv_buf = torch.empty(self.recv_cap, dtype=torch.float32,
                                     device=dev)
        self.hidden_buf = torch.zeros(self.recv_cap, H, dtype=dtype,
                                      device=dev)
        self.weights_buf = torch.zeros(self.recv_cap, dtype=torch.float32,
                                       device=dev)
        self.out_buf = torch.zeros(self.recv_cap, ffn, dtype=dtype,
                                   device=dev)
        self.comb_hidden_buf = torch.zeros(self.recv_cap, H, dtype=dtype,
                                           device=dev)
        self.comb_send_buf = torch.empty(self.recv_cap, H, dtype=dtype,
                                         device=dev)
        self.comb_recv_buf = torch.empty(self.N, H, dtype=dtype, device=dev)
        self.stage_buf = torch.empty(self.N, H, dtype=dtype, device=dev)
        self.final_out = torch.zeros(S, H, dtype=dtype, device=dev)
        self._in_splits = torch.empty(R, dtype=torch.int32, device=dev)
        self._out_splits = torch.empty(R, dtype=torch.int32, device=dev)

        # per-iteration plan state (bound by plan_meta)
        self.n_recv = 0
        self._send_row_index = None   # my_tok, wire order
        self._place_slots = None
        self._comb_dst = None
        self._pentry = None
        self._segments = []           # [(local_slot p, start, end)]

        # sub-phase event ledger (diagnostic metrics; cuda only)
        self._sub_names = ("pack", "wire", "place", "gemm0",
                           "gemm2", "cpack", "comb", "acc")
        self._sub_ev = {n: [] for n in ("t0",) + self._sub_names}
        self._cuda = torch.cuda.is_available() and dev.type == "cuda"

    # -- driver-interface no-ops / hygiene ---------------------------------

    def set_weights(self, w1_slot, w2_slot):
        """Pad-FIRST slot weights (build_slot_weights): real local slot i
        lives at index 1 + i."""
        assert w1_slot.shape == (self.gpe, self.ffn, self.H)
        assert w2_slot.shape == (self.gpe, self.H, self.ffn)
        self.w1 = w1_slot.contiguous()
        self.w2 = w2_slot.contiguous()

    def prep(self):
        self.out_buf.zero_()

    def prime_scale_graph(self, planner):  # noqa: ARG002 — interface parity
        return

    def issue_combine_meta(self, ip, late=False):  # noqa: ARG002
        return

    # -- plan adapter (timed: driver plan bracket) --------------------------

    def plan_meta(self, ip):
        """vce -> eplb-class direct layout. One batched pinned D2H (the
        honest in-window host sync, same accounting class as the fused
        arm's derive_routed_meta event sync)."""
        S, K, R, nlp = self.S, self.K, self.W, self.nlp
        N = self.N
        vce = ip.vce.view(R, N).long()
        # ours pad-FIRST vce: vce = owner*gpe + 1 + phys%nlp (pad never
        # routed); recover global physical slot
        phys_all = (vce // self.gpe) * nlp + (vce % self.gpe) - 1
        tok_exp = self._tok_t.expand(R, N)
        k_exp = self._k_t.expand(R, N)
        # canonical (phys, token) order per source == dest-major for free
        order = torch.argsort(phys_all * (S + 1) + tok_exp, dim=1,
                              stable=True)
        ent_tok = torch.gather(tok_exp, 1, order)
        ent_phys = torch.gather(phys_all, 1, order)
        lay = direct_layout_entries_fast(ent_tok, ent_phys, self.rank,
                                         nlp, R)
        my_tok = lay["my_tok"]
        my_k = torch.gather(k_exp, 1, order)[self.rank]
        ent_flat = my_tok * K + my_k
        pentry = self._probs_own_flat[ent_flat]
        # deterministic home slot of every sent entry: the reverse a2av
        # returns rows in MY send order, so comb_dst is a permutation of
        # [0, S*K) by construction (each (token, k) sent exactly once)
        comb_dst = ent_flat

        blob = self._blob_pin
        blob.copy_(torch.cat([
            lay["seg_rows"], lay["seg_start"],
            lay["in_splits"].long(), lay["out_splits"].long(),
            lay["pair_max"].reshape(1), lay["n_recv_dev"].long(),
        ]))
        seg_rows_h = blob[:nlp].tolist()
        seg_start_h = blob[nlp:2 * nlp].tolist()
        pair_max = int(blob[-2])
        n_recv = int(blob[-1])
        assert pair_max <= self.max_split, (
            f"pair rows {pair_max} exceed All2AllSingle max_split "
            f"{self.max_split} (silent wire overflow)")
        assert n_recv <= self.recv_cap, (
            f"recv overflow: n_recv {n_recv} > recv_cap {self.recv_cap}")

        self.n_recv = n_recv
        self._send_row_index = my_tok
        self._place_slots = lay["place_slots_pad"][:n_recv]
        self._comb_dst = comb_dst
        self._pentry = pentry
        self._in_splits.copy_(lay["in_splits"])
        self._out_splits.copy_(lay["out_splits"])
        self._segments = [
            (p, seg_start_h[p], seg_start_h[p] + seg_rows_h[p])
            for p in range(nlp) if seg_rows_h[p] > 0
        ]

    # -- timed phases -------------------------------------------------------

    def _mark(self, name):
        if self._cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._sub_ev[name].append(e)

    def l0_forward(self, inputs_shard, gate_kwargs=None):  # noqa: ARG002
        self._mark("t0")
        torch.index_select(inputs_shard, 0, self._send_row_index,
                           out=self.send_buf)
        self.wsend_buf.copy_(self._pentry)
        self._mark("pack")
        n = self.n_recv
        self._a2a_hidden.forward(
            self.send_buf, self.recv_buf[:n],
            self._in_splits, self._out_splits, self.num_comm_sm)
        self._a2a_probs.forward(
            self.wsend_buf.view(-1, 1), self.wrecv_buf[:n].view(-1, 1),
            self._in_splits, self._out_splits, self.num_comm_sm)
        self._mark("wire")
        self.hidden_buf[:n].index_copy_(0, self._place_slots,
                                        self.recv_buf[:n])
        self.weights_buf[:n].index_copy_(0, self._place_slots,
                                         self.wrecv_buf[:n])
        self._mark("place")
        for p, start, end in self._segments:
            self._gemm_only.forward(
                self.hidden_buf[start:end], self.w1[1 + p],
                output_buf=self.out_buf[start:end], fast_accum=False)
        self._mark("gemm0")
        return self.out_buf[:n]

    def l1_forward(self, intermediate):
        n = self.n_recv
        for p, start, end in self._segments:
            self._gemm_only.forward(
                intermediate[start:end], self.w2[1 + p],
                output_buf=self.comb_hidden_buf[start:end],
                fast_accum=False)
        self._mark("gemm2")
        # combine-pack: recv-stream (wire arrival) order, expert-side fp32
        # prob scale (the eplb/EPIC non-hc convention)
        rows = self.comb_hidden_buf[:n].index_select(0, self._place_slots)
        scale = self.weights_buf[:n].index_select(0, self._place_slots)
        self.comb_send_buf[:n] = (rows.float()
                                  * scale.unsqueeze(1)).to(self.dtype)
        self._mark("cpack")
        # reverse wire: same op pair, swapped splits
        self._a2a_hidden.forward(
            self.comb_send_buf[:n], self.comb_recv_buf,
            self._out_splits, self._in_splits, self.num_comm_sm)
        self._mark("comb")
        # deterministic home accumulation: comb_dst is a permutation of
        # [0, S*K); one terminal K-axis sum (bitwise-stable ordering)
        self.stage_buf.index_copy_(0, self._comb_dst, self.comb_recv_buf)
        self.final_out.copy_(
            self.stage_buf.view(self.S, self.K, self.H).sum(1))
        self._mark("acc")
        return self.final_out

    # -- diagnostics --------------------------------------------------------

    def sub_times(self, warmup_iters):
        """Per-iteration sub-phase ms (post-warmup), dwire_* metric keys.
        Call after the timed loop and BEFORE any extra forward (the final
        deterministic iteration appends events past the loop)."""
        if not self._cuda or not self._sub_ev["t0"]:
            return {}
        n_iters = len(self._sub_ev["t0"])
        out = {f"dwire_{n}_ms": [] for n in self._sub_names}
        self._sub_ev["acc"][-1].synchronize()
        for i in range(warmup_iters, n_iters):
            prev = self._sub_ev["t0"][i]
            for name in self._sub_names:
                e = self._sub_ev[name][i]
                out[f"dwire_{name}_ms"].append(prev.elapsed_time(e))
                prev = e
        return out


class OursDirectOverlapRunner:
    """OURS-DOV — the direct-transport ablation with the dispatch leg
    OVERLAPPED (2026-08-30): same plan lane as every ours arm (pv2 +
    LocCap + r2, OursIterPlanner), same one-hop wire byte pattern and the
    same direct (un-staged) combine as the dwire arm, but layer0 runs as
    ONE GemmGroupedV2AGScatterOp forward in the FLAT a2av mode: pack ->
    ring-ordered direct puts (rank+1 first) with per-source signals ->
    the grouped GEMM launches immediately and its tiles are claimed
    bucket-by-bucket as source signals land (the existing dynamic-claimer
    tile spin). The wire honors the wire-ordering HARD RULE via
    FLUX_A2AV_FLAT_FENCED_SIG: intra-node sources ride nbi put_signal
    (NVLink store order, per-source release), remote data goes as
    CONCURRENT nbi puts ordered before their signal ops by ONE PE quiet
    (F2). No pack/place phases remain on the timed path: the pack is the
    op's own, and the GEMM writes its outputs directly into the dwire
    segment layout (stable expert-major scatter == (slot; source; token),
    the layout direct_layout_entries_fast produces), so the combine can
    stay byte-identical to the dwire arm.

    Plan bracket = derive_routed_meta (the fused arm's derive, one honest
    event-synced pinned D2H) + host prefix sums over sps + one [W, gpe]
    H2D + O(n_recv) index math + a LOCAL argsort of my own vce row. The
    probs side-wire is dropped: the combine scale comes from the
    planner's fused probs allgather through the scatter index (bitwise
    the same fp32 values the dwire arm sends over its probs a2av)."""

    def __init__(self, tp_group, ep_group, nnodes, rank, local_world_size,
                 cfg, ffn, dtype, recv_cap, max_split, sm_margin=1,
                 num_comm_sm=None, device="cuda"):
        import flux  # GPU-side only

        self.rank = rank
        self.L = local_world_size
        self.cfg = cfg
        S, K, R, H = cfg.S, cfg.K, cfg.R, cfg.H
        self.S, self.K, self.W, self.H = S, K, R, H
        self.nlp = cfg.nlp
        self.gpe = cfg.nlp + 1
        self.E_virt = R * self.gpe
        self.ep_start = rank * self.gpe
        self.N = S * K
        self.ffn = ffn
        self.dtype = dtype
        self.recv_cap = int(recv_cap)
        self.max_split = int(max_split)
        self.sm_margin = int(sm_margin)
        self.num_comm_sm = int(
            num_comm_sm if num_comm_sm is not None
            else os.environ.get("FLUX_OURS_DWIRE_COMM_SM", "8"))
        dev = torch.device(device)
        self.device = dev

        # ctor env contract (must be identical on every rank BEFORE the
        # collective op ctor): FLAT mode wants no union/compress knobs
        # (LB_UNION flips union_bcast_ and the early-launch default), the
        # fenced flat wire is the whole point of this arm, and early
        # launch is ctor-REQUIRED by it (the flat GEMM gate otherwise
        # waits full wire drain).
        os.environ["FLUX_A2AV_LB_UNION"] = "0"
        os.environ.setdefault("FLUX_A2AV_FLAT_FENCED_SIG", "1")
        os.environ.setdefault("FLUX_A2AV_EARLY_LAUNCH", "1")
        os.environ.setdefault("FLUX_A2AV_MAX_RECV_NTOKENS",
                              str(self.recv_cap))

        tp_env = flux.DistEnvTPWithEP(tp_group=tp_group, nnodes=nnodes,
                                      ep_group=ep_group)
        moe_args = flux.MoeArguments(
            max_ntokens=R * S,
            hidden=H,
            ffn_hidden=ffn,
            nexperts=self.E_virt,
            topk=K,
            input_dtype=dtype,
            output_dtype=dtype,
        )
        self.l0_op = flux.GemmGroupedV2AGScatterOp(
            tp_env=tp_env, moe_args=moe_args,
            a2av_dispatch=True, a2av_hier_compress=False)
        # combine wire: the dwire arm's staged a2av, hidden leg only
        self._a2a_hidden = flux.All2AllSingle(
            tp_group, self.max_split, H, local_world_size, dtype)
        self._gemm_only = flux.GemmOnly(dtype, dtype, dtype,
                                        use_fp8_gemm=False)

        # buffers
        self.out_buf = torch.zeros(self.recv_cap, ffn, dtype=dtype,
                                   device=dev)
        self.comb_hidden_buf = torch.zeros(self.recv_cap, H, dtype=dtype,
                                           device=dev)
        self.comb_send_buf = torch.empty(self.recv_cap, H, dtype=dtype,
                                         device=dev)
        self.comb_recv_buf = torch.empty(self.N, H, dtype=dtype, device=dev)
        self.stage_buf = torch.empty(self.N, H, dtype=dtype, device=dev)
        self.final_out = torch.zeros(S, H, dtype=dtype, device=dev)
        # +1 guard slot (see _build_scale in ours.py — copies whose
        # scatter row is outside my block land there and are sliced away)
        self.scale_buf = torch.zeros(self.recv_cap + 1, dtype=torch.float32,
                                     device=dev)
        self._in_splits = torch.empty(R, dtype=torch.int32, device=dev)
        self._out_splits = torch.empty(R, dtype=torch.int32, device=dev)
        self._cnt_pin = torch.empty(R, self.gpe, dtype=torch.int64,
                                    pin_memory=torch.cuda.is_available())
        self._cnt_dev = torch.empty(R, self.gpe, dtype=torch.int64,
                                    device=dev)
        self._arange_recv = torch.arange(self.recv_cap, device=dev,
                                         dtype=torch.int64)

        # per-iteration plan state
        self._sd = self._scd = self._sps = self._uc = None
        self._m_this = 0
        self.n_recv = 0            # alias for the harness audit
        self._place_slots = None   # arrival order -> out-layout row
        self._comb_dst = None
        self._segments = []        # [(vslot p, start, end)], p in [1, gpe)

        self._sub_names = ("l0", "gemm2", "cpack", "comb", "acc")
        self._sub_ev = {n: [] for n in ("t0",) + self._sub_names}
        self._cuda = torch.cuda.is_available() and dev.type == "cuda"

    # -- driver-interface no-ops / hygiene ---------------------------------

    def set_weights(self, w1_slot, w2_slot):
        """Pad-FIRST slot weights (build_slot_weights): w1 goes to the
        fused op whole ([gpe, ffn, H], pad slot zero-split); w2 is
        consumed per-slot by the segment GEMM2."""
        assert w1_slot.shape == (self.gpe, self.ffn, self.H)
        assert w2_slot.shape == (self.gpe, self.H, self.ffn)
        self.w1 = w1_slot.contiguous()
        self.w2 = w2_slot.contiguous()

    def prep(self):
        self.out_buf.zero_()
        self.l0_op.clear_buffers()

    def prime_scale_graph(self, planner):  # noqa: ARG002 — interface parity
        return

    def issue_combine_meta(self, ip, late=False):  # noqa: ARG002
        return

    # -- plan adapter (timed: driver plan bracket) --------------------------

    def plan_meta(self, ip):
        """derive_routed_meta (the op's own derive — sd/scd/sps/uc, one
        event-synced pinned D2H) + the combine metadata from sps: splits,
        segment table, the arrival->segment permutation, the scatter-based
        combine scale, and my local send order (comb_dst)."""
        R, gpe = self.W, self.gpe
        sd, scd, sps, uc = self.l0_op.derive_routed_meta(ip.vce)
        self._sd, self._scd, self._sps, self._uc = sd, scd, sps, uc

        # host math over the pinned sps (free after the derive's sync)
        sps_l = sps.long()
        cnt_ms = sps_l[:, self.ep_start:self.ep_start + gpe]  # [W, gpe]
        m_this = int(cnt_ms.sum())
        assert m_this <= self.recv_cap, (
            f"recv overflow: m_this {m_this} > recv_cap {self.recv_cap}")
        in_h = sps_l[self.rank].view(R, gpe).sum(1)   # my sends per dest
        out_h = cnt_ms.sum(1)                         # my recv per source
        pair_max = int(max(in_h.max(), out_h.max()))
        assert pair_max <= self.max_split, (
            f"pair rows {pair_max} exceed All2AllSingle max_split "
            f"{self.max_split} (silent wire overflow)")
        self._in_splits.copy_(in_h.to(torch.int32))
        self._out_splits.copy_(out_h.to(torch.int32))
        seg_rows = cnt_ms.sum(0)                      # [gpe], pad-first
        assert int(seg_rows[0]) == 0, "rows routed to the pad slot"
        seg_start = torch.zeros(gpe, dtype=torch.int64)
        seg_start[1:] = torch.cumsum(seg_rows, 0)[:-1]
        self._segments = [
            (p, int(seg_start[p]), int(seg_start[p] + seg_rows[p]))
            for p in range(gpe) if int(seg_rows[p]) > 0
        ]
        self._m_this = m_this
        self.n_recv = m_this

        # arrival (source-major) -> out layout (slot-major) permutation,
        # pure prefix-sum algebra over cnt_ms (both interiors are token-
        # ascending, so group offsets are the whole mapping)
        self._cnt_pin.copy_(cnt_ms)
        self._cnt_dev.copy_(self._cnt_pin, non_blocking=True)
        cnt = self._cnt_dev
        flat = cnt.reshape(-1)                        # (s, p) groups
        arr_start = flat.cumsum(0) - flat
        col = cnt.sum(0)
        seg_start_dev = col.cumsum(0) - col           # [gpe]
        out_start = (seg_start_dev.unsqueeze(0)
                     + (cnt.cumsum(0) - cnt)).reshape(-1)
        self._place_slots = (self._arange_recv[:m_this]
                             + torch.repeat_interleave(
                                 out_start - arr_start, flat))

        # combine scale per OUT row (ours.py _scale_compute, eager):
        # row r of my segment gets the gate weight of the copy whose
        # scatter index landed there; foreign copies -> guard slot
        sd_long = sd.long()
        m_start = sd_long[:self.ep_start].sum()
        idx = scd.view(-1).long() - m_start
        valid = (idx >= 0) & (idx < self.recv_cap)
        idx = torch.where(valid, idx,
                          torch.full_like(idx, self.recv_cap))
        self.scale_buf.zero_()
        self.scale_buf.scatter_(0, idx, ip.probs_all.view(-1))

        # my send order == the op's pack order (dest-major == vslot-major
        # by owner-major vslot numbering; token-ascending interiors): the
        # reverse wire returns my rows in exactly this order, and
        # arange(N)[order] == order makes comb_dst the sort permutation
        my_vce = ip.vce[self.rank * self.S:(self.rank + 1) * self.S]
        self._comb_dst = torch.argsort(my_vce.reshape(-1).long(),
                                       stable=True)

    # -- timed phases -------------------------------------------------------

    def _mark(self, name):
        if self._cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._sub_ev[name].append(e)

    def l0_forward(self, inputs_shard, gate_kwargs=None):
        assert not gate_kwargs, "dov is s1-only (no movement gates)"
        self._mark("t0")
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
        )
        self._mark("l0")
        return self.out_buf[:self._m_this]

    def l1_forward(self, intermediate):
        n = self._m_this
        for p, start, end in self._segments:
            self._gemm_only.forward(
                intermediate[start:end], self.w2[p],
                output_buf=self.comb_hidden_buf[start:end],
                fast_accum=False)
        self._mark("gemm2")
        # combine-pack in wire arrival order, expert-side fp32 scale —
        # byte-identical to the dwire combine
        rows = self.comb_hidden_buf[:n].index_select(0, self._place_slots)
        scale = self.scale_buf[:n].index_select(0, self._place_slots)
        self.comb_send_buf[:n] = (rows.float()
                                  * scale.unsqueeze(1)).to(self.dtype)
        self._mark("cpack")
        self._a2a_hidden.forward(
            self.comb_send_buf[:n], self.comb_recv_buf,
            self._out_splits, self._in_splits, self.num_comm_sm)
        self._mark("comb")
        self.stage_buf.index_copy_(0, self._comb_dst, self.comb_recv_buf)
        self.final_out.copy_(
            self.stage_buf.view(self.S, self.K, self.H).sum(1))
        self._mark("acc")
        return self.final_out

    # -- diagnostics --------------------------------------------------------

    def sub_times(self, warmup_iters):
        """Per-iteration sub-phase ms (post-warmup), dov_* metric keys."""
        if not self._cuda or not self._sub_ev["t0"]:
            return {}
        n_iters = len(self._sub_ev["t0"])
        out = {f"dov_{n}_ms": [] for n in self._sub_names}
        self._sub_ev["acc"][-1].synchronize()
        for i in range(warmup_iters, n_iters):
            prev = self._sub_ev["t0"][i]
            for name in self._sub_names:
                e = self._sub_ev[name][i]
                out[f"dov_{name}_ms"].append(prev.elapsed_time(e))
                prev = e
        return out
