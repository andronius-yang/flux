# OURS scenario 2 — live per-iteration re-placement with OVERLAPPED expert
# weight movement (the "expert dispatch" lane).
#
# Design (2026-08-25 fusion campaign, designer-B spec, critic-ratified):
#   placement lane (timed, place_ms, every iteration): drift prefilter →
#     warm PLACE-lambda-FAST solve (pA=2 pB=1 repair=1, warm seed =
#     resident) → decision statistic → gain trigger + runway-fit test →
#     move diff (adds/removes).
#   movement (on trigger; issued on a dedicated stream, NEVER summed into
#     phase columns): two WeightPushMulticast ops (w1 [ffn,H], w2 [H,ffn]),
#     multicast mode (ONE cross-node leg per (expert, dest node), NVLink CE
#     fan-out via gateways) + optional egress NIC-sharding
#     (plan_weight_shards: every NIC of both nodes carries a big leg).
#   consumption: the fused l0 GEMM's per-slot weight gate
#     (weight_signal=op_w1.signals()[...] with weight_gate_group_start=1 —
#     the pad-FIRST slot convention makes gated groups 1..nlp align with
#     WPM slots) — only MOVED slots' tiles ever spin, and only until their
#     push lands; w2 lands via op_w2.join() right before the l1 forward
#     (runway = plan + l0 + gelu, ~8-15 ms at 16n b8: effectively free).
#
# Epoch protocol per trigger iteration:
#   1. locally raise UNMOVED slots' signals to (epoch+1) (index_fill_);
#   2. set_plan(adds) + forward(multicast=True)  -> bumps epoch, issues legs;
#   3. gateway/egress/ingress/shard_join on the late-drained side stream;
#   4. the fused forward gates on weight_signal_epoch = the new epoch.
# Non-trigger iterations: no forward(), epoch unchanged, every signal
# already >= epoch — each gated tile pays one satisfied read.
#
# Wire-audit note (SCHEMA rule 6c): weight_push_multicast is wire-order
# UNAUDITED — every s2 gate cell runs the WEIGHT payload probe (per-epoch
# re-randomized home rows; a stale slot fails allclose on exactly the moved
# experts' rows) before any s2 number is quoted.

import os

import torch

from flux.testing.moonep_fused_map import plan_weight_shards

__all__ = ["OursMovementLane", "assign_gateways_sparse",
           "build_sched_order"]


def build_sched_order(gpe, moved_slots, device="cuda"):
    """Moved-last schedule encoding for the fused op (int32 [gpe]):
    front class (bit 30 clear) = pad slot 0 + unmoved slots in original
    index order, rank in [0, n_front); deferred class (bit 30 set) = the
    moved slots in ascending slot order (== WPM issue order), rank in
    [0, gpe - n_front). Returns (order tensor on `device`, n_front)."""
    moved = sorted({int(x) for x in moved_slots})
    assert 0 not in moved, "pad slot cannot move"
    order = [0] * gpe
    front = 0
    mset = set(moved)
    for e in range(gpe):
        if e not in mset:
            order[e] = front
            front += 1
    for i, e in enumerate(moved):
        order[e] = (1 << 30) | i
    t = torch.tensor(order, dtype=torch.int32, device=device)
    return t, front


def assign_gateways_sparse(adds, local_world_size):
    """Deterministic node-level multicast gateway assignment for a SPARSE
    add list (same policy as moonep_fused_map.assign_gateways, which takes
    a dense MoonEPPlan): adds = [(dst_rank, dst_slot, home_rank, src_row)].
    Returns [n, 6] rows (dst_rank, dst_slot, home_rank, src_row, gw_rank,
    gw_slot); gw_rank == -1 -> direct home->dst leg. For each cross-node
    (src_row@home, dest_node) group the gateway is the needy rank
    minimizing the running per-gateway forward-byte load (groups visited
    in ascending (home_rank, src_row, dest_node); ties to lowest rank);
    the home sends ONE inter-node put into the gateway's own slot and the
    gateway fans out over NVLink."""
    L = local_world_size
    groups = {}
    for (d, slot, h, sr) in adds:
        key = (h, sr, d // L)
        groups.setdefault(key, []).append((d, slot))
    gw_load = {}
    out = []
    for key in sorted(groups):
        h, sr, dn = key
        members = sorted(groups[key])
        if dn == h // L or len(members) == 1:
            # same-node dests (NVLink CE from home) and singletons: direct
            for d, slot in members:
                out.append((d, slot, h, sr, -1, -1))
            continue
        gw = min(members,
                 key=lambda m: (gw_load.get(m[0], 0), m[0]))
        gw_load[gw[0]] = gw_load.get(gw[0], 0) + len(members) - 1
        for d, slot in members:
            if (d, slot) == gw:
                out.append((d, slot, h, sr, -1, -1))  # the wire leg itself
            else:
                out.append((d, slot, h, sr, gw[0], gw[1]))
    return out


class OursMovementLane:
    """Owns the two weight ops, the resident-placement store, the trigger,
    and the movement streams. The driver owns event brackets."""

    def __init__(self, pg, rank, W, L, cfg, ffn, H, dtype,
                 gen_w1, gen_w2,
                 gain_threshold_ppm=50000,
                 weight_shard="auto",
                 shard_chunk_bytes=1 << 21):
        import flux
        self.pg = pg
        self.rank = rank
        self.W = W
        self.L = L
        self.cfg = cfg
        self.nlp = cfg.nlp
        self.gpe = cfg.nlp + 1
        self.G = cfg.G
        self.epn = cfg.G // W          # logical home shard per rank
        self.ffn = ffn
        self.H = H
        self.dtype = dtype
        self.gen_w1 = gen_w1
        self.gen_w2 = gen_w2
        self.gain_threshold_ppm = gain_threshold_ppm
        self.weight_shard = weight_shard
        self.shard_chunk_bytes = shard_chunk_bytes
        # FLUX_OURS_S2_MCAST=0: direct home->dst puts for every leg (no
        # gateway lane at all) — hang-triage / ablation knob
        self.multicast = bool(int(os.environ.get("FLUX_OURS_S2_MCAST",
                                                 "1")))
        # 2026-08-26 exposed-latency knobs (default OFF; arms set them):
        # moved-last GEMM schedule — defer THIS iteration's moved slots'
        # problems behind every resident problem in the static schedule,
        # so no CTA head-blocks on a weight spin while resident tiles
        # remain (release-time criterion: under s2 the moved class is
        # genuinely blocked until its push lands, unlike the NR-14
        # prefetch class whose weights were already resident).
        self.sched_moved_last = bool(int(os.environ.get(
            "FLUX_OURS_SCHED_MOVED_LAST", "0")))
        # late w2 issue — enqueue the l1 weight pushes AFTER the fused l0
        # forward (dispatch legs reach the proxy queue first; w2 runway =
        # l0 + gelu; join_w2 semantics unchanged).
        self.w2_late = bool(int(os.environ.get(
            "FLUX_OURS_S2_W2_LATE", "0")))

        # WPM slot space = gpe rows so prefetch_slots() IS the fused op's
        # [gpe, r0, r1] weights view; slot 0 = the pad slot (never pushed,
        # zeros). Gate alignment: weight_signal = signals()[1:] so gated
        # group j (j = 1..nlp) reads signal index j-1 -> WPM slot j. wait:
        # gate reads weight_signal_ptr[group - start] with start=1 => index
        # j-1; signals()[1:][j-1] = signals[j] = slot j's epoch. Correct.
        self.op_w1 = flux.WeightPushMulticast(
            pg, self.epn, self.gpe, ffn, H, dtype)
        self.op_w2 = flux.WeightPushMulticast(
            pg, self.epn, self.gpe, H, ffn, dtype)

        # home rows = the static logical shard (push sources)
        w1h = torch.stack([gen_w1(rank * self.epn + i)
                           for i in range(self.epn)]).to(dtype)
        w2h = torch.stack([gen_w2(rank * self.epn + i)
                           for i in range(self.epn)]).to(dtype)
        self.op_w1.weight_home().copy_(w1h.cuda())
        self.op_w2.weight_home().copy_(w2h.cuda())

        self.w_stream = torch.cuda.Stream()      # issue stream (movement)
        self.side_stream = torch.cuda.Stream()   # late-drained roles
        self.ev_move_start = torch.cuda.Event(enable_timing=True)
        self.ev_move_end = torch.cuda.Event(enable_timing=True)
        self.ev_issue_done = torch.cuda.Event()

        # resident placement (slot -> logical expert), device + host copies
        self.resident_p2l = None   # host long [W*nlp]
        self.moves_this_iter = 0
        self.move_bytes_this_iter = 0
        self.trigger_fired = 0
        self.last_gain_ppm = 0
        # moved-last schedule products (CURRENT iteration only)
        self._sched_order = None
        self._sched_n_front = 0
        # late-w2 pending state
        self._w2_pending = False
        self._w2_pairs = self._w2_shards = self._w2_keep = None
        self.ev_w2_issue_done = torch.cuda.Event()

    # -- setup ---------------------------------------------------------------

    def fill_slots_local(self, p2l_host):
        """Initial (oracle) placement slot fill by LOCAL deterministic
        generation — no wire at setup, forward() reserved for timed
        movement. prefetch_slots()[0] stays zeros (pad)."""
        self.resident_p2l = p2l_host.long().clone()
        s1 = self.op_w1.prefetch_slots()
        s2 = self.op_w2.prefetch_slots()
        s1.zero_()
        s2.zero_()
        for j in range(self.nlp):
            e = int(p2l_host[self.rank * self.nlp + j])
            s1[1 + j].copy_(self.gen_w1(e).to(self.dtype).cuda())
            s2[1 + j].copy_(self.gen_w2(e).to(self.dtype).cuda())
        torch.cuda.synchronize()

    def gemm_weights(self):
        """[gpe, r0, r1] views the fused ops consume directly."""
        return self.op_w1.prefetch_slots(), self.op_w2.prefetch_slots()

    def gate_kwargs(self):
        """weight-gate kwargs for the fused l0 forward (pad-first slot
        convention: gate groups 1..nlp <-> WPM slots 1..nlp)."""
        kw = dict(
            weight_signal=self.op_w1.signals()[1:],
            weight_signal_epoch=int(self.op_w1.epoch()),
            weight_gate_group_start=1,
        )
        if self._sched_order is not None:
            # moved-last schedule (this trigger iteration's moved set)
            kw["sched_expert_order"] = self._sched_order
            kw["sched_n_front"] = self._sched_n_front
        return kw

    def join_w2(self):
        """Zero-SM landing gate for the l1 weights, on the current stream."""
        if self._w2_pending:  # failsafe: driver skipped issue_w2_late
            self.issue_w2_late()
        self.op_w2.join()

    def join_w1(self):
        """End-of-iteration fabric drain for the l0 weight signals (2026-08-25
        K2-4n stale hang fix): the gated tiles consume only ROUTED slots, so
        a zero-row moved slot's epoch SET can still be in flight when the
        iteration boundary's device-sync + world barrier pass (one-sided
        remote writes are NOT bounded by the destination's sync). Under
        movement-every-iteration the late SET lands AFTER the next epoch
        raise and REGRESSES the flag -> next iteration's tile spins forever.
        join() waits GEQ epoch on ALL slots -> nothing crosses the boundary.
        Runs after e2e_end (untimed gap); satisfied reads when quiet."""
        self.op_w1.join()

    # -- per-iteration movement (called from the driver's place bracket) ----

    def apply_moves(self, new_p2l_host, gain_ppm, wprobe_cb=None):
        """Adopt `new_p2l_host` (host long [W*nlp]) if the trigger fires:
        issue delta weight pushes for every slot whose expert changed,
        multicast + optional NIC-shard, on w_stream; roles on side_stream.
        wprobe_cb(changed_experts) runs BEFORE any push is issued (the
        rule-6c weight payload probe re-randomizes home rows there).
        Returns True when movement was issued (the caller then routes on
        the NEW placement this iteration — eager adoption)."""
        self.moves_this_iter = 0
        self.move_bytes_this_iter = 0
        self.trigger_fired = 0
        self.last_gain_ppm = int(gain_ppm)
        self._sched_order = None
        self._sched_n_front = 0
        assert not self._w2_pending, "previous iteration's w2 never issued"
        if gain_ppm < self.gain_threshold_ppm:
            return False
        changed = (new_p2l_host != self.resident_p2l).nonzero(
            as_tuple=True)[0]
        if changed.numel() == 0:
            return False
        self.trigger_fired = 1
        self.moves_this_iter = int(changed.numel())
        if wprobe_cb is not None:
            wprobe_cb(sorted({int(new_p2l_host[i])
                              for i in changed.tolist()}),
                      set(changed.tolist()))

        # replicated pair list: (dst_rank, dst_slot, home_rank, src_row,
        # gw_rank, gw_slot); sparse gateway assignment (one cross-node leg
        # per (expert, dest node), NVLink fan-out)
        # vectorized adds build (was a ~300-iteration python loop)
        e_t = new_p2l_host[changed]
        adds = list(zip((changed // self.nlp).tolist(),
                        (changed % self.nlp + 1).tolist(),
                        (e_t // self.epn).tolist(),
                        (e_t % self.epn).tolist()))
        pairs = assign_gateways_sparse(adds, self.L)
        pairs_t = torch.tensor(pairs, dtype=torch.int32).reshape(-1, 6)
        per_expert_bytes = 2 * self.ffn * self.H * self.op_w1.weight_home().element_size()
        self.move_bytes_this_iter = self.moves_this_iter * per_expert_bytes

        cur = torch.cuda.current_stream()
        self.w_stream.wait_stream(cur)
        mine = changed[(changed // self.nlp) == self.rank]
        moved_slots = torch.unique(mine % self.nlp + 1)
        # keep mask + shard plan hoisted out of the op loop: identical
        # for w1/w2 (same slot space; same expert byte size ffn*H*es)
        mask = torch.ones(self.gpe, dtype=torch.bool)
        mask[moved_slots] = False
        keep = mask.nonzero(as_tuple=True)[0].cuda()
        shards = None
        if self.weight_shard != "off":
            shards = plan_weight_shards(
                pairs_t, self.L,
                self.ffn * self.H
                * self.op_w1.weight_home().element_size(),
                mode="mcast")
        if self.sched_moved_last and moved_slots.numel() > 0:
            # moved-last GEMM schedule encoding for the fused l0 forward:
            # front class = pad + unmoved slots (original order), deferred
            # class = MY moved slots ascending (== push issue order)
            self._sched_order, self._sched_n_front = build_sched_order(
                self.gpe, moved_slots.tolist())
        ops_now = ((self.op_w1,) if self.w2_late
                   else (self.op_w1, self.op_w2))
        with torch.cuda.stream(self.w_stream):
            self.ev_move_start.record()
            for op in ops_now:
                self._issue_op(op, pairs_t, shards, keep)
            self.ev_issue_done.record()
        self.side_stream.wait_event(self.ev_issue_done)
        with torch.cuda.stream(self.side_stream):
            for op in ops_now:
                self._issue_roles(op)
            self.ev_move_end.record()
        if self.w2_late:
            self._w2_pending = True
            self._w2_pairs, self._w2_shards, self._w2_keep = (
                pairs_t, shards, keep)

        self.resident_p2l = new_p2l_host.long().clone()
        return True

    def _issue_op(self, op, pairs_t, shards, keep):
        """Issue one weight op's movement legs on the CURRENT stream:
        raise unmoved slots to the new epoch (only moved slots gate),
        install the plan, forward. Epoch read per-op at issue time (w1/w2
        advance in lockstep, but late-w2 must not reuse a stale value)."""
        new_epoch = int(op.epoch()) + 1
        op.signals().index_fill_(0, keep, new_epoch)
        op.set_plan(pairs_t)
        if shards is not None:
            op.set_shard_plan(shards, self.shard_chunk_bytes, self.L)
        op.forward(self.multicast)

    def _issue_roles(self, op):
        """Late-drained relay roles for one op (side stream)."""
        if self.multicast:
            op.forward_gateway()
        if self.weight_shard != "off":
            op.forward_egress()
            op.forward_ingress()
            op.forward_shard_join()

    def issue_w2_late(self):
        """FLUX_OURS_S2_W2_LATE deferred l1-weight issue — called by the
        driver right after the fused l0 forward is ENQUEUED, so the
        dispatch wire's puts reach the proxy queue ahead of w2's. NO
        current-stream dependency is taken here (a wait_stream would park
        w2 behind the whole l0 GEMM, whose moved tiles spin on w1 signals
        -> the w2 legs would deadlock join_w2); w_stream's own FIFO
        already orders these puts after w1's. No-op unless a trigger
        iteration stashed pending w2 state."""
        if not self._w2_pending:
            return
        self._w2_pending = False
        with torch.cuda.stream(self.w_stream):
            self._issue_op(self.op_w2, self._w2_pairs, self._w2_shards,
                           self._w2_keep)
            self.ev_w2_issue_done.record()
        self.side_stream.wait_event(self.ev_w2_issue_done)
        with torch.cuda.stream(self.side_stream):
            self._issue_roles(self.op_w2)
            self.ev_move_end.record()
        self._w2_pairs = self._w2_shards = self._w2_keep = None

    def movement_ms(self):
        """Off-chain movement-stream span (never summed into phases)."""
        if not self.trigger_fired:
            return 0.0
        self.ev_move_end.synchronize()
        return self.ev_move_start.elapsed_time(self.ev_move_end)
