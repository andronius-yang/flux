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

import numpy as np
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

    Vectorized 2026-08-28 (numpy, every pair of every node at once —
    ~20 ops regardless of node count; bitwise the same output as the
    2026-08-27 loop incl. the top-8 prefilter set and the (gain, (e_h,
    e_l)) tie-break — 400-trial equivalence test). tau_rows < 0 = FORCE
    mode (the always-overlap probe): prefilter + gain threshold bypassed,
    every pair applies its best exchange regardless of gain sign, so
    movement oscillates at the fixed point and NVLink traffic fires every
    iteration (the sizing orbit detects the cycle)."""
    p2l_np = np.asarray(p2l.numpy(), dtype=np.int64)
    lg = np.asarray(load_g.numpy(), dtype=np.int64)
    lc = np.maximum(np.asarray(lcnts.numpy(), dtype=np.int64), 1)
    R = p2l_np.shape[0] // nlp
    NN = R // L
    G = lg.shape[0]
    force = tau_rows < 0
    valid = p2l_np >= 0
    e_all = np.where(valid, p2l_np, 0)
    w_slot = np.where(valid, lg[e_all] // lc[e_all], 0)          # [R*nlp]
    L_r = w_slot.reshape(R, nlp).sum(1)                             # [R]
    lr = L_r.reshape(NN, L)
    # per-node rank order by (-load, rank): distinct ranks -> one int key
    key = (-lr) * L + np.arange(L)[None, :]
    order = np.argsort(key, axis=1, kind="stable")                  # [NN, L]
    P = L // 2
    H = (np.arange(NN)[:, None] * L + order[:, :P]).reshape(-1)     # [Q]
    Lo = (np.arange(NN)[:, None] * L + order[:, L - 1:L - 1 - P:-1]).reshape(-1)
    Q = H.shape[0]
    lrH, lrL = L_r[H], L_r[Lo]
    pair_ok = np.ones(Q, dtype=bool) if force else (lrH - lrL > tau_rows)
    hs = H[:, None] * nlp + np.arange(nlp)[None, :]                 # [Q, nlp]
    ls = Lo[:, None] * nlp + np.arange(nlp)[None, :]
    e_hs, e_ls = p2l_np[hs], p2l_np[ls]
    v_hs, v_ls = e_hs >= 0, e_ls >= 0
    w_hs, w_ls = w_slot[hs], w_slot[ls]
    BIG = np.int64(1) << 60
    # prefilter order: hs by (-w, e) heaviest first, ls by (w, e) lightest
    # first; invalid slots sort last
    k_hs = np.lexsort((np.where(v_hs, e_hs, BIG), np.where(v_hs, -w_hs, BIG)), axis=1)
    k_ls = np.lexsort((np.where(v_ls, e_ls, BIG), np.where(v_ls, w_ls, BIG)), axis=1)
    K = min(8, nlp)
    qi = np.arange(Q)[:, None]
    hs8, ls8 = hs[qi, k_hs[:, :K]], ls[qi, k_ls[:, :K]]            # [Q, K]
    eh8, el8 = p2l_np[hs8], p2l_np[ls8]
    vh8, vl8 = eh8 >= 0, el8 >= 0
    wh8, wl8 = w_slot[hs8], w_slot[ls8]
    # membership constraints against the FULL slot sets of the pair
    eh_in_l = (eh8[:, :, None] == e_ls[:, None, :]).any(-1)         # [Q, K]
    el_in_h = (el8[:, :, None] == e_hs[:, None, :]).any(-1)
    new_max = np.maximum(lrH[:, None, None] - wh8[:, :, None] + wl8[:, None, :],
                         lrL[:, None, None] - wl8[:, None, :] + wh8[:, :, None])
    gain = np.maximum(lrH, lrL)[:, None, None] - new_max            # [Q, K, K]
    ok = (vh8[:, :, None] & vl8[:, None, :] & ~eh_in_l[:, :, None]
          & ~el_in_h[:, None, :] & (eh8[:, :, None] != el8[:, None, :])
          & (wh8[:, :, None] > wl8[:, None, :]) & pair_ok[:, None, None])
    if not force:
        ok &= gain >= tau_rows
    # maximize gain, tie -> smallest (e_h, e_l)
    sel = gain * (G * G + 1) - (eh8[:, :, None] * G + el8[:, None, :])
    sel = np.where(ok, sel, -BIG)
    flat = sel.reshape(Q, -1)
    best = flat.argmax(1)
    has = flat[np.arange(Q), best] > -BIG
    a, b = best // K, best % K
    swaps = []
    for q in np.nonzero(has)[0]:
        swaps.append((int(H[q]), int(hs8[q, a[q]]), int(eh8[q, a[q]]),
                      int(Lo[q]), int(ls8[q, b[q]]), int(el8[q, b[q]])))
    return swaps, torch.from_numpy(L_r)


def apply_swaps(p2l, l2p, swaps):
    """Apply a swap list to (p2l [R*nlp] i32, l2p [G, R] i32) -> new host
    tensors. A swap is a slot-content transposition: p2l[s_h] <-> p2l[s_l]
    and the two experts' l2p rows have those phys entries exchanged
    (columns re-sorted ascending, the canonical l2p order). numpy
    (2026-08-28): all transpositions compose as one slot relabeling —
    slots are distinct across swaps (a rank joins at most one); bitwise
    == the 2026-08-27 per-pair loop."""
    p2l_n = p2l.clone()
    l2p_n = l2p.clone()
    if not swaps:
        return p2l_n, l2p_n
    pa = p2l_n.numpy()
    la = l2p_n.numpy()
    ids = np.array([(sw[1], sw[4], sw[2], sw[5]) for sw in swaps],
                   dtype=np.int64)
    sh, sl, eh, el = ids[:, 0], ids[:, 1], ids[:, 2], ids[:, 3]
    assert (pa[sh] == eh).all() and (pa[sl] == el).all(), \
        "swap list stale vs p2l"
    pa[sh] = el
    pa[sl] = eh
    slot_map = np.arange(pa.shape[0], dtype=np.int64)
    slot_map[sh] = sl
    slot_map[sl] = sh
    experts = np.unique(np.concatenate([eh, el]))
    rows = la[experts].astype(np.int64)
    valid = rows >= 0
    big = pa.shape[0] + 1
    rows = np.where(valid, slot_map[np.maximum(rows, 0)], big)
    rows.sort(axis=1)
    la[experts] = np.where(rows == big, -1, rows).astype(la.dtype)
    return p2l_n, l2p_n


def swap_orbit(load_g, p2l, l2p, lcnts, L, nlp, tau_rows, max_rounds=8,
               return_cycle=False):
    """Setup-side fixed-point iteration of the runtime swap sequence on a
    fixed demand vector (the sizing-envelope fold input). Returns the list
    of successive (p2l, l2p) placements AFTER each swapping iteration
    (empty if the start placement is already stable). With
    return_cycle=True also returns the index into that list where the
    detected cycle begins (-1 = the cycle includes the start placement,
    None = no cycle: a fixed point or max_rounds reached)."""
    out = []
    keys = [bytes(p2l.numpy().tobytes())]
    cur_p2l, cur_l2p = p2l, l2p
    cycle = None
    for _ in range(max_rounds):
        swaps, _ = swap_plan(load_g, cur_p2l, lcnts, L, nlp, tau_rows)
        if not swaps:
            break
        cur_p2l, cur_l2p = apply_swaps(cur_p2l, cur_l2p, swaps)
        key = bytes(cur_p2l.numpy().tobytes())
        if key in keys:
            cycle = keys.index(key) - 1   # cycle (force-mode oscillation)
            break
        keys.append(key)
        out.append((cur_p2l, cur_l2p))
    return (out, cycle) if return_cycle else out


def swap_orbit_nodes(load_g, p2l, l2p, lcnts, L, nlp, tau_rows, max_rounds=64):
    """Force-mode orbit with PER-NODE cycle detection (2026-08-29). Each
    node's pair oscillation is independent, so the global placement only
    repeats when all node cycles realign (LCM of their periods — at 16n
    this exceeded the 8-round global search and the sizing fold covered
    the whole warmup transient, overflowing the 16G heap at b64). Returns
    (out, T, period): out = successive placements (out[i] = after round
    i+1), T = index into out of the first placement at which EVERY node is
    inside its cycle (-1 = the start placement already is), period = LCM
    of the node cycle lengths. T/period are None if some node has not
    cycled within max_rounds (caller falls back to the full fold)."""
    from math import gcd
    R = p2l.numel() // nlp
    NN = R // L
    span = L * nlp
    out = []
    node_keys = [[bytes(p2l[u * span:(u + 1) * span].numpy().tobytes())]
                 for u in range(NN)]
    node_entry = [None] * NN      # keys-index of the node's cycle entry
    node_len = [None] * NN
    cur_p2l, cur_l2p = p2l, l2p
    for _ in range(max_rounds):
        swaps, _ = swap_plan(load_g, cur_p2l, lcnts, L, nlp, tau_rows)
        if not swaps:
            break
        cur_p2l, cur_l2p = apply_swaps(cur_p2l, cur_l2p, swaps)
        out.append((cur_p2l, cur_l2p))
        for u in range(NN):
            if node_entry[u] is not None:
                continue
            k = bytes(cur_p2l[u * span:(u + 1) * span].numpy().tobytes())
            if k in node_keys[u]:
                node_entry[u] = node_keys[u].index(k)
                node_len[u] = len(node_keys[u]) - node_entry[u]
            else:
                node_keys[u].append(k)
        if all(e is not None for e in node_entry):
            break
    if any(e is None for e in node_entry):
        return out, None, None
    T = max(node_entry) - 1           # keys index -> out index (-1 = start)
    period = 1
    for n in node_len:
        period = period * n // gcd(period, n)
    return out, T, period


def _libcuda():
    """ctypes handle on the CUDA driver for the zero-SM stream memops the
    P2P exchange uses (same primitive WPM's C++ join uses on this heap)."""
    import ctypes
    lib = ctypes.CDLL("libcuda.so.1")
    fn = getattr(lib, "cuStreamWaitValue64_v2", None) or lib.cuStreamWaitValue64
    fn.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_ulonglong,
                   ctypes.c_uint]
    fn.restype = ctypes.c_int
    wr = getattr(lib, "cuStreamWriteValue64_v2", None) or lib.cuStreamWriteValue64
    wr.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_ulonglong,
                   ctypes.c_uint]
    wr.restype = ctypes.c_int
    return fn, wr


CU_STREAM_WAIT_VALUE_GEQ = 0x0
CU_STREAM_WAIT_VALUE_FLUSH = 0x4


class SwapTableSync:
    """Pushes the planner's HOST tables (plan.p2l, plan.l2p — updated by
    apply_swaps) into its existing DEVICE tables with two non-blocking
    copies from pinned mirrors: zero stream syncs, two launches (~40 us
    host). Replaces refresh_placement() on the fired-swap path (three
    pageable .to(dev) = three cudaStreamSynchronize, ~0.15-0.3 ms) and
    the first device-index attempt (12 launches, ~0.4 ms — slower).
    Pinned mirrors are double-buffered by parity so the host write of
    iteration i+1 can never race the in-flight H2D of iteration i.
    lcnts is invariant under a transposition."""

    def __init__(self, plan, device):
        self._pin_p2l = [torch.empty_like(plan.p2l, dtype=torch.long).pin_memory()
                         for _ in range(2)]
        self._pin_l2p = [torch.empty_like(plan.l2p).pin_memory()
                         for _ in range(2)]
        self._parity = 0

    def apply(self, planner, plan):
        k = self._parity
        self._parity ^= 1
        self._pin_p2l[k].copy_(plan.p2l)
        self._pin_l2p[k].copy_(plan.l2p)
        planner.p2l.copy_(self._pin_p2l[k], non_blocking=True)
        planner.l2p.copy_(self._pin_l2p[k], non_blocking=True)


def apply_swaps_tables_(p2l, l2p, sh, sl, eh, el, slot_map, arange, n_slots):
    """In-place transposition on (p2l long [R*nlp], l2p [G, R]) from id
    tensors on the SAME device — enqueue-only on CUDA. == apply_swaps."""
    p2l[sh] = el
    p2l[sl] = eh
    slot_map.copy_(arange)
    slot_map[sh] = sl
    slot_map[sl] = sh
    experts = torch.cat([eh, el])
    rows = l2p[experts].long()
    valid = rows >= 0
    rows = torch.where(valid, slot_map[rows.clamp(min=0)], rows)
    big = n_slots + 1
    srt, _ = torch.sort(torch.where(valid, rows, torch.full_like(rows, big)),
                        dim=1)
    l2p[experts] = torch.where(srt == big, torch.full_like(srt, -1),
                               srt).to(l2p.dtype)


class OursSwapLane:
    """Intra-node NVLink expert exchange, overlapped on the movement
    stream. Reuses an OursMovementLane's WPM slot storage + signals
    (never calls the WPM wire — no forward/multicast/shard). Epoch
    protocol: self.epoch is the gate epoch; unmoved slots are raised
    immediately, swapped slots after the exchange lands (LOCAL writes
    only).

    xport="nccl": torch batch_isend_irecv inside the per-node subgroup
    (the 2026-08-27 baseline; ~2.3 ms host enqueue per fired swap).
    xport="p2p": symmetric-heap staging with node-local peer views
    (flux.create_tensor_list -> nvshmem_ptr): each rank copies ITS slot
    into the PEER's staging (one contiguous cudaMemcpy over NVLink), then
    stores the epoch into the peer's landed-signal, waits ITS OWN
    landed-signal with a zero-SM cuStreamWaitValue64, copies staging ->
    slot and raises its own gate signal. Nobody writes anyone else's
    slot and nobody reads anyone else's staging, so the only cross-rank
    order is the landed-signal (peer's copy precedes its store in stream
    order). Host cost: ~8 enqueues per fired swap.

    issue="early": both matrices issued in the place bracket, right after
    the decision (the exchange leads the plan derive + dispatch, so the
    moved slot's gate is normally up before l0 starts).
    issue="late":  both matrices issued AFTER the fused l0 forward is
    enqueued (the moved slot's tiles spin until landing; the exchange
    visibly rides under dispatch/GEMM). issue="split": w1 early, w2 late
    (w2 is first needed by l1). The movement stream never takes a
    dependency on the current stream past the pre-l0 event (a wait_stream
    after l0 = deadlock against the gated GEMM)."""

    def __init__(self, lane, node_group, rank, L, nlp, ffn, H, dtype,
                 xport="nccl", issue="early", pg=None):
        assert xport in ("nccl", "p2p") and issue in ("early", "late",
                                                       "split")
        self.lane = lane                    # OursMovementLane (storage)
        self.node_group = node_group        # this node's dist subgroup
        self.rank = rank
        self.L = L
        self.nlp = nlp
        self.xport = xport
        self.issue_mode = issue
        self.epoch = int(lane.op_w1.epoch())
        self.w_stream = lane.w_stream
        self.ev_start = torch.cuda.Event(enable_timing=True)
        self.ev_end = torch.cuda.Event(enable_timing=True)
        self.ev_done = torch.cuda.Event()
        self.ev_pre = torch.cuda.Event()    # current-stream point the
        #                                     exchange may depend on
        self._idx_all = torch.arange(1, nlp + 1, device="cuda")
        self._slot_idx = [torch.tensor([1 + j], device="cuda")
                          for j in range(nlp)]
        self._keep_idx = [self._idx_all[self._idx_all != (1 + j)]
                          for j in range(nlp)]
        self.swaps_this_iter = 0
        self.move_bytes_this_iter = 0
        self._issued = False
        self._pending = None                # (my_slot, peer_rank)
        self._w_waited = False
        self._n_issued = 0
        if xport == "nccl":
            self._stag_w1 = torch.empty(ffn, H, dtype=dtype, device="cuda")
            self._stag_w2 = torch.empty(H, ffn, dtype=dtype, device="cuda")
        else:
            import flux
            assert pg is not None, "p2p xport needs the world pg"
            self.local_rank = rank % L
            self._stag_w1_all = flux.create_tensor_list([ffn, H], dtype, pg)
            self._stag_w2_all = flux.create_tensor_list([H, ffn], dtype, pg)
            self._xsig_all = flux.create_tensor_list([2], torch.int64, pg,
                                                     False, True)
            assert len(self._stag_w1_all) == L, (
                f"node-local view count {len(self._stag_w1_all)} != L {L}")
            self._stag_w1 = self._stag_w1_all[self.local_rank]
            self._stag_w2 = self._stag_w2_all[self.local_rank]
            self._xsig = self._xsig_all[self.local_rank]
            self._cu_wait, self._cu_write = _libcuda()
            self._wait_flags = CU_STREAM_WAIT_VALUE_GEQ | CU_STREAM_WAIT_VALUE_FLUSH
            # probe FLUSH support once (satisfied wait: value 0 >= 0)
            if self._cu_wait(torch.cuda.current_stream().cuda_stream,
                             self._xsig.data_ptr(), 0, self._wait_flags) != 0:
                self._wait_flags = CU_STREAM_WAIT_VALUE_GEQ
            # probe stream memops writes on the local heap AND on a peer
            # view (zero-SM signal path); fall back to fill_ kernels
            cs = torch.cuda.current_stream().cuda_stream
            peer_probe = self._xsig_all[(self.local_rank + 1) % L]
            self._write_ok = (
                self._cu_write(cs, self._xsig.data_ptr(), 0, 0) == 0
                and self._cu_write(cs, peer_probe.data_ptr(), 0, 0) == 0)
            torch.cuda.synchronize()

    # -- per-iteration protocol ---------------------------------------------

    def prepare(self, swaps):
        """Place-bracket part (current stream, host-cheap): epoch bump,
        raise every UNMOVED slot's gate signal, record the pre-l0 event the
        exchange may depend on. Replicated plan: every rank calls this
        with the same list; only the two ranks of a pair move data."""
        self.swaps_this_iter = len(swaps)
        self.move_bytes_this_iter = 0
        self._issued = False
        self._pending = None
        self._w_waited = False
        self._n_issued = 0
        if not swaps:
            return
        mine = [sw for sw in swaps
                if sw[0] == self.rank or sw[3] == self.rank]
        assert len(mine) <= 1, "a rank joins at most one swap"
        self.epoch += 1
        cur = torch.cuda.current_stream()
        self.ev_pre.record(cur)
        sig1 = self.lane.op_w1.signals()
        sig2 = self.lane.op_w2.signals()
        if not mine:
            # bystander: raise every real slot and be done
            sig1.index_fill_(0, self._idx_all, self.epoch)
            sig2.index_fill_(0, self._idx_all, self.epoch)
            return
        (rh, s_h, e_h, rl, s_l, e_l) = mine[0]
        my_slot = (s_h if rh == self.rank else s_l) % self.nlp
        peer = rl if rh == self.rank else rh
        sig1.index_fill_(0, self._keep_idx[my_slot], self.epoch)
        sig2.index_fill_(0, self._keep_idx[my_slot], self.epoch)
        self._pending = (my_slot, peer)

    def issue_early(self):
        if self._pending is None:
            return
        if self.issue_mode == "early":
            self._exchange(("w1", "w2"))
        elif self.issue_mode == "split":
            self._exchange(("w1",))

    def issue_late(self):
        """Call immediately AFTER the fused l0 forward is enqueued."""
        if self._pending is None:
            return
        if self.issue_mode == "late":
            self._exchange(("w1", "w2"))
        elif self.issue_mode == "split":
            self._exchange(("w2",))

    def _exchange(self, mats):
        import torch.distributed as dist
        my_slot, peer = self._pending
        with torch.cuda.stream(self.w_stream):
            if not self._w_waited:
                self.w_stream.wait_event(self.ev_pre)
                self.ev_start.record()
                self._w_waited = True
            for m in mats:
                k = 0 if m == "w1" else 1
                op = self.lane.op_w1 if k == 0 else self.lane.op_w2
                slot = op.prefetch_slots()[1 + my_slot]
                stag = self._stag_w1 if k == 0 else self._stag_w2
                if self.xport == "p2p":
                    peer_local = peer % self.L
                    peer_stag = (self._stag_w1_all if k == 0
                                 else self._stag_w2_all)[peer_local]
                    peer_stag.copy_(slot)             # NVLink P2P write
                    peer_sig = self._xsig_all[peer_local]
                    if self._write_ok:
                        rc = self._cu_write(self.w_stream.cuda_stream,
                                            peer_sig.data_ptr() + 8 * k,
                                            self.epoch, 0)
                        assert rc == 0, f"cuStreamWriteValue64 rc={rc}"
                    else:
                        peer_sig[k:k + 1].fill_(self.epoch)
                    rc = self._cu_wait(self.w_stream.cuda_stream,
                                       self._xsig.data_ptr() + 8 * k,
                                       self.epoch, self._wait_flags)
                    assert rc == 0, f"cuStreamWaitValue64 rc={rc}"
                else:
                    ops = [dist.P2POp(dist.isend, slot, peer,
                                      group=self.node_group),
                           dist.P2POp(dist.irecv, stag, peer,
                                      group=self.node_group)]
                    for work in dist.batch_isend_irecv(ops):
                        work.wait()      # stream-orders w_stream on comms
                slot.copy_(stag)
                if self.xport == "p2p" and self._write_ok:
                    rc = self._cu_write(self.w_stream.cuda_stream,
                                        op.signals().data_ptr()
                                        + 8 * (1 + my_slot), self.epoch, 0)
                    assert rc == 0, f"cuStreamWriteValue64 rc={rc}"
                else:
                    op.signals().index_fill_(0, self._slot_idx[my_slot],
                                             self.epoch)
                self.move_bytes_this_iter += (
                    2 * slot.numel() * slot.element_size())
                self._n_issued += 1
            if self._n_issued == 2:
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
