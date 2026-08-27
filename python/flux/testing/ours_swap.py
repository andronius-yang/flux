# OURS intra-node expert SWAP (branch pv2, 2026-08-27) — the EPIC §4.3
# real-time-rebalance analog, adapted to the OURS s2 stack and OVERLAPPED
# into the NVLink idle gaps between inter-node puts.
#
# Premise (user direction): inter-node wire is the scarce resource — NO
# cross-node expert movement per iteration. NVLink sits idle between the
# dispatch/combine inter-node puts (the redistribution round before egress
# and the forward round after ingress). Per-iteration INTRA-NODE expert
# swaps consume exactly that idle NVLink, rebalance GPU loads inside each
# node, and their benefit lands in the SAME iteration (the router runs
# after the swap-updated tables, EPIC's same-launch property).
#
# Decision (EPIC §4.3 greedy, ours from richer inputs): per-rank load
# L_r = sum over hosted instances of load[e] // lcnts[e] (global demand d
# from the plan_comm allgather AGAINST THE CURRENT placement p2l — not the
# static home map). Per node: sort ranks by L_r, pair heaviest<->lightest
# (then 2nd<->2nd...), and for each pair pick the single expert exchange
# minimizing the pair max load; accept iff gain >= tau_rows (the EPIC
# tau = t_swap/t_token movement-cost threshold, in rows). TIMED in the
# place bracket (counts toward total_ms): pure host integer math,
# sub-millisecond by construction (O(R*nlp) vector stats + <= NN * L/2 *
# nlp^2 scalar evaluations) — microbenched in test_ours_swap.py.
#
# Movement: the paired ranks exchange slot weights DIRECTLY over NVLink
# (NCCL P2P inside a per-node subgroup, issued on the movement stream,
# staged then copied into the WPM slot buffers). Signal protocol is
# LOCAL-ONLY: each rank raises its own slots' epoch signals — unmoved
# slots immediately, swapped slots after its recv+copy completes on the
# movement stream. No remote signal writes, no gateways, no NIC shards:
# the WPM mid-iteration remote-signal race class (handoff 20) cannot
# occur here by construction. The fused l0 GEMM's per-slot weight gate
# provides the overlap (only the swapped slots' tiles spin, and only
# until the NVLink exchange lands); l1 (w2) waits one movement-stream
# event before its forward (runway argument, ours_s2 header).
#
# Determinism: the plan is a pure integer function of (d, p2l) — every
# rank computes the identical swap list. With a fixed per-cell trace the
# runtime placement sequence is the setup-computable SWAP ORBIT (apply
# until fixed point), which the driver folds into the s2 sizing envelope
# exactly like the pv2 batch solve.

import torch

__all__ = ["swap_plan", "apply_swaps", "swap_orbit", "OursSwapLane"]


def rank_loads(load_g, p2l, lcnts, R, nlp):
    """[R] int64 per-rank load estimate: sum over hosted slots of
    load[e] // lcnts[e] (each instance carries an equal share). Host,
    vectorized, ~50 us."""
    p2l_l = p2l.long()
    valid = p2l_l >= 0
    e = p2l_l.clamp(min=0)
    share = torch.where(valid,
                        load_g.long()[e] // lcnts.long().clamp(min=1)[e],
                        torch.zeros_like(e))
    return share.view(R, nlp).sum(1)


def swap_plan(load_g, p2l, lcnts, L, nlp, tau_rows):
    """One round of EPIC-greedy intra-node pairing. Host integer, pure.
    Returns (swaps, L_r) where swaps = [(r_h, s_h, e_h, r_l, s_l, e_l)]
    (global slot ids; each rank appears in at most one swap).

    tau_rows < 0 = FORCE mode (the always-overlap probe): the pair-gap
    prefilter and the gain threshold are bypassed — every pair applies
    its best candidate exchange regardless of gain sign. At a balanced
    fixed point the best exchange has negative gain and the next
    iteration's best is its reversal, so movement OSCILLATES and NVLink
    traffic fires every iteration (worst-case overlap; the sizing orbit
    detects the cycle)."""
    R = p2l.numel() // nlp
    NN = R // L
    L_r = rank_loads(load_g, p2l, lcnts, R, nlp)
    lr = L_r.tolist()
    p2l_h = p2l.tolist()
    lg = load_g.tolist()
    lc = lcnts.tolist()
    force = tau_rows < 0
    swaps = []
    for u in range(NN):
        ranks = sorted(range(u * L, (u + 1) * L),
                       key=lambda r: (-lr[r], r))
        for i in range(L // 2):
            h, l = ranks[i], ranks[L - 1 - i]
            if not force and lr[h] - lr[l] <= tau_rows:
                continue
            hs = [(s, p2l_h[s]) for s in range(h * nlp, (h + 1) * nlp)
                  if p2l_h[s] >= 0]
            ls = [(s, p2l_h[s]) for s in range(l * nlp, (l + 1) * nlp)
                  if p2l_h[s] >= 0]
            h_set = {e for _, e in hs}
            l_set = {e for _, e in ls}
            base = max(lr[h], lr[l])
            best = None  # (gain, e_h, e_l, s_h, s_l)
            for s_h, e_h in hs:
                if e_h in l_set:
                    continue          # would duplicate on the light rank
                w_h = lg[e_h] // max(lc[e_h], 1)
                for s_l, e_l in ls:
                    if e_l in h_set or e_l == e_h:
                        continue
                    w_l = lg[e_l] // max(lc[e_l], 1)
                    if w_h <= w_l:
                        continue      # only heavy-for-light helps
                    new_max = max(lr[h] - w_h + w_l, lr[l] - w_l + w_h)
                    gain = base - new_max
                    if (force or gain >= tau_rows) and (
                            best is None or gain > best[0]
                            or (gain == best[0]
                                and (e_h, e_l) < (best[1], best[2]))):
                        best = (gain, e_h, e_l, s_h, s_l)
            if best is not None:
                _, e_h, e_l, s_h, s_l = best
                swaps.append((h, s_h, e_h, l, s_l, e_l))
    return swaps, L_r


def apply_swaps(p2l, l2p, swaps):
    """Apply a swap list to (p2l [R*nlp] i32, l2p [G, R] i32) -> new host
    tensors. A swap is a slot-content transposition: p2l[s_h] <-> p2l[s_l]
    and the two experts' l2p rows have those phys entries exchanged
    (columns re-sorted ascending, the canonical l2p order)."""
    p2l_n = p2l.clone()
    l2p_n = l2p.clone()
    for (_rh, s_h, e_h, _rl, s_l, e_l) in swaps:
        assert int(p2l_n[s_h]) == e_h and int(p2l_n[s_l]) == e_l
        p2l_n[s_h] = e_l
        p2l_n[s_l] = e_h
        for e, old_s, new_s in ((e_h, s_h, s_l), (e_l, s_l, s_h)):
            row = l2p_n[e]
            n = int((row >= 0).sum())
            vals = row[:n].tolist()
            vals[vals.index(old_s)] = new_s
            l2p_n[e, :n] = torch.tensor(sorted(vals), dtype=row.dtype)
    return p2l_n, l2p_n


def swap_orbit(load_g, p2l, l2p, lcnts, L, nlp, tau_rows, max_rounds=8):
    """Setup-side fixed-point iteration of the runtime swap sequence on a
    fixed demand vector (the sizing-envelope fold input). Returns the list
    of successive (p2l, l2p) placements AFTER each swapping iteration
    (empty if the start placement is already stable)."""
    out = []
    seen = {bytes(p2l.numpy().tobytes())}
    cur_p2l, cur_l2p = p2l, l2p
    for _ in range(max_rounds):
        swaps, _ = swap_plan(load_g, cur_p2l, lcnts, L, nlp, tau_rows)
        if not swaps:
            break
        cur_p2l, cur_l2p = apply_swaps(cur_p2l, cur_l2p, swaps)
        key = bytes(cur_p2l.numpy().tobytes())
        if key in seen:
            break                    # cycle (force-mode oscillation)
        seen.add(key)
        out.append((cur_p2l, cur_l2p))
    return out


class OursSwapLane:
    """Intra-node NVLink expert exchange, overlapped on the movement
    stream. Reuses an OursMovementLane's WPM slot storage + signals
    (never calls the WPM wire — no forward/multicast/shard). Epoch
    protocol: self.epoch is the gate epoch; unmoved slots are raised
    immediately, swapped slots after the exchange lands (LOCAL writes
    only)."""

    def __init__(self, lane, node_group, rank, L, nlp, ffn, H, dtype):
        self.lane = lane                    # OursMovementLane (storage)
        self.node_group = node_group        # this node's dist subgroup
        self.rank = rank
        self.L = L
        self.nlp = nlp
        self.epoch = int(lane.op_w1.epoch())
        self.w_stream = lane.w_stream
        self.ev_start = torch.cuda.Event(enable_timing=True)
        self.ev_end = torch.cuda.Event(enable_timing=True)
        self.ev_done = torch.cuda.Event()
        self._stag_w1 = torch.empty(ffn, H, dtype=dtype, device="cuda")
        self._stag_w2 = torch.empty(H, ffn, dtype=dtype, device="cuda")
        self._idx_all = torch.arange(1, nlp + 1, device="cuda")
        self.swaps_this_iter = 0
        self.move_bytes_this_iter = 0
        self._issued = False

    def issue(self, swaps):
        """Issue the exchange for this iteration's swap list (may be
        empty). Replicated plan: every rank calls this with the same
        list; only the two ranks of a pair move data. Enqueue-only —
        returns immediately."""
        import torch.distributed as dist
        self.swaps_this_iter = len(swaps)
        self.move_bytes_this_iter = 0
        self._issued = False
        mine = [sw for sw in swaps
                if sw[0] == self.rank or sw[3] == self.rank]
        assert len(mine) <= 1, "a rank joins at most one swap"
        if not swaps:
            return
        self.epoch += 1
        cur = torch.cuda.current_stream()
        self.w_stream.wait_stream(cur)
        with torch.cuda.stream(self.w_stream):
            self.ev_start.record()
            sig1 = self.lane.op_w1.signals()
            sig2 = self.lane.op_w2.signals()
            if not mine:
                # bystander: raise every real slot and be done
                sig1.index_fill_(0, self._idx_all, self.epoch)
                sig2.index_fill_(0, self._idx_all, self.epoch)
            else:
                (rh, s_h, e_h, rl, s_l, e_l) = mine[0]
                my_slot = (s_h if rh == self.rank else s_l) % self.nlp
                peer = rl if rh == self.rank else rh
                keep = self._idx_all[self._idx_all != (1 + my_slot)]
                sig1.index_fill_(0, keep, self.epoch)
                sig2.index_fill_(0, keep, self.epoch)
                slot_w1 = self.lane.op_w1.prefetch_slots()[1 + my_slot]
                slot_w2 = self.lane.op_w2.prefetch_slots()[1 + my_slot]
                ops = [dist.P2POp(dist.isend, slot_w1, peer,
                                  group=self.node_group),
                       dist.P2POp(dist.isend, slot_w2, peer,
                                  group=self.node_group),
                       dist.P2POp(dist.irecv, self._stag_w1, peer,
                                  group=self.node_group),
                       dist.P2POp(dist.irecv, self._stag_w2, peer,
                                  group=self.node_group)]
                for work in dist.batch_isend_irecv(ops):
                    work.wait()          # stream-orders w_stream on comms
                slot_w1.copy_(self._stag_w1)
                slot_w2.copy_(self._stag_w2)
                sig1.index_fill_(
                    0, torch.tensor([1 + my_slot], device="cuda"),
                    self.epoch)
                sig2.index_fill_(
                    0, torch.tensor([1 + my_slot], device="cuda"),
                    self.epoch)
                self.move_bytes_this_iter = 2 * (
                    slot_w1.numel() + slot_w2.numel()
                ) * slot_w1.element_size()
            self.ev_end.record()
            self.ev_done.record()
        self._issued = True

    def gate_kwargs(self):
        """Per-slot weight-gate kwargs for the fused l0 forward (pad-first
        convention, ours_s2 precedent)."""
        return dict(
            weight_signal=self.lane.op_w1.signals()[1:],
            weight_signal_epoch=self.epoch,
            weight_gate_group_start=1,
        )

    def l1_wait(self):
        """w2 landing gate: the current stream waits the movement-stream
        exchange (zero-cost when nothing was issued this iteration)."""
        if self._issued:
            torch.cuda.current_stream().wait_event(self.ev_done)

    def movement_ms(self):
        if not self._issued:
            return 0.0
        self.ev_end.synchronize()
        return self.ev_start.elapsed_time(self.ev_end)
