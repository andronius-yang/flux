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

__all__ = ["OursMovementLane", "assign_gateways_sparse"]


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
        return dict(
            weight_signal=self.op_w1.signals()[1:],
            weight_signal_epoch=int(self.op_w1.epoch()),
            weight_gate_group_start=1,
        )

    def join_w2(self):
        """Zero-SM landing gate for the l1 weights, on the current stream."""
        self.op_w2.join()

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
        adds = []
        for idx in changed.tolist():
            dst_rank, j = idx // self.nlp, idx % self.nlp
            e = int(new_p2l_host[idx])
            adds.append((dst_rank, 1 + j, e // self.epn, e % self.epn))
        pairs = assign_gateways_sparse(adds, self.L)
        pairs_t = torch.tensor(pairs, dtype=torch.int32).reshape(-1, 6)
        per_expert_bytes = 2 * self.ffn * self.H * self.op_w1.weight_home().element_size()
        self.move_bytes_this_iter = self.moves_this_iter * per_expert_bytes

        cur = torch.cuda.current_stream()
        self.w_stream.wait_stream(cur)
        with torch.cuda.stream(self.w_stream):
            self.ev_move_start.record()
            # raise unmoved slots to the NEW epoch so only moved slots gate
            new_epoch = int(self.op_w1.epoch()) + 1
            moved_slots = torch.unique(
                torch.tensor([1 + (i % self.nlp) for i in changed.tolist()
                              if i // self.nlp == self.rank],
                             dtype=torch.long))
            for op in (self.op_w1, self.op_w2):
                sig = op.signals()
                mask = torch.ones(self.gpe, dtype=torch.bool)
                mask[moved_slots] = False
                keep = mask.nonzero(as_tuple=True)[0].cuda()
                sig.index_fill_(0, keep, new_epoch)
                op.set_plan(pairs_t)
                if self.weight_shard != "off":
                    from flux.testing.moonep_fused_map import (
                        plan_weight_shards)
                    shards = plan_weight_shards(
                        pairs_t, self.L,
                        self.ffn * self.H
                        * op.weight_home().element_size(),
                        mode="multicast")
                    op.set_shard_plan(shards, self.shard_chunk_bytes,
                                      self.L)
                op.forward(True)
            self.ev_issue_done.record()
        self.side_stream.wait_event(self.ev_issue_done)
        with torch.cuda.stream(self.side_stream):
            for op in (self.op_w1, self.op_w2):
                op.forward_gateway()
                if self.weight_shard != "off":
                    op.forward_egress()
                    op.forward_ingress()
                    op.forward_shard_join()
            self.ev_move_end.record()

        self.resident_p2l = new_p2l_host.long().clone()
        return True

    def movement_ms(self):
        """Off-chain movement-stream span (never summed into phases)."""
        if not self.trigger_fired:
            return 0.0
        self.ev_move_end.synchronize()
        return self.ev_move_start.elapsed_time(self.ev_move_end)
