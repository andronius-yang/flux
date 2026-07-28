################################################################################
#
# Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
"""CPU-only simulation of the a2av_hier_compress balanced inter-node relay.

Runs without a GPU or the flux extension: it re-implements, in pure torch-CPU,
the exact host offset math and the exact ATen op sequences of
GemmGroupedV2AGScatterOpImpl::a2av_dispatch on the hier-relay-balance branch
(producer pack, balanced chunk partition, relay piece placement, wire chunks,
the generalized forward-index build, and gateway delivery), then checks that

  1. every destination's dedup recv buffer is byte-identical across
     (a) the balanced-relay wire, (b) the identity-relay wire (the original
     a2av_hier_compress scheme), and (c) a direct reference layout;
  2. every live recv row is written exactly once (the window cuts tile each
     (source, dest) region with no overlap and no gap);
  3. all forward indices stay inside their staging window (no garbage-slot
     bleed into live columns);
  4. per-round wire chunks are balanced to within one row, and a uniform-U
     routing relocates zero rows (the identity fast path);
  5. the three-stream schedule is deadlock-free under an event-driven
     executor with epoch-signal semantics, across two back-to-back epochs.

Usage:  python3 test/python/moe_ag_scatter/test_relay_balance_math.py
"""

import torch


# ---------------------------------------------------------------- metadata


def build_uc(choosed_experts, W, NN, L, T, E):
    """u[s][d], U[s][n] exactly as test_moe_ag_traffic.py builds them."""
    ntokens, _topk = choosed_experts.shape
    owner = choosed_experts.long() // E  # [ntokens, topk] destination rank
    flags = torch.zeros(ntokens, W, dtype=torch.bool)
    flags.scatter_(1, owner, True)  # token t needed by rank d (any expert)
    u_mat = flags.view(W, T, W).sum(1)  # [W, W]
    U_mat = flags.view(ntokens, NN, L).any(dim=2).view(W, T, NN).sum(1)  # [W, NN]
    return u_mat.long(), U_mat.long(), flags


def canon_start(U_mat, NN, L, n, sl, m):
    """Canonical start of source (n, sl)'s segment in the n -> m stream."""
    acc = 0
    for s2 in range(sl):
        acc += int(U_mat[n * L + s2, m])
    return acc


def chunk_bound(U_mat, NN, L, n, m, k):
    total = canon_start(U_mat, NN, L, n, L, m)
    return (total // L) * k + min(k, total % L)


def chunk_rows_of(U_mat, NN, L, n, m, k):
    return chunk_bound(U_mat, NN, L, n, m, k + 1) - chunk_bound(U_mat, NN, L, n, m, k)


def seg_offsets(u_mat, U_mat, W, NN, L, rank):
    """seg_off_h of the compressed send buffer for `rank` (nseg = L + NN - 1)."""
    my_node = rank // L
    nseg = L + NN - 1
    seg_off = [0] * (nseg + 1)
    seg = 0
    for n in range(NN):
        if n == my_node:
            for dl in range(L):
                seg_off[seg + 1] = seg_off[seg] + int(u_mat[rank, n * L + dl])
                seg += 1
        else:
            seg_off[seg + 1] = seg_off[seg] + int(U_mat[rank, n])
            seg += 1
    return seg_off


def recv_off_of_u(u_mat, s, d):
    return int(u_mat[:s, d].sum())


# ---------------------------------------------------------------- producer pack


def producer_pack(choosed_experts, rank, W, NN, L, T, E, topk, seg_off):
    """Exact ATen semantics of the compress pack (:954-971): returns the send
    buffer as GLOBAL TOKEN IDS, one row per (token, segment) pair."""
    my_node = rank // L
    nseg = L + NN - 1
    cpr = T * topk
    e_my = choosed_experts.view(-1)[rank * cpr : (rank + 1) * cpr].long()
    d64 = e_my.div(E, rounding_mode="floor")  # destination global rank
    nd = d64.div(L, rounding_mode="floor")  # destination node
    seg = torch.where(
        nd.eq(my_node),
        d64 - my_node * (L - 1),
        torch.where(nd.lt(my_node), nd, nd + (L - 1)),
    )
    tl = torch.arange(cpr).div(topk, rounding_mode="floor")
    pf = torch.zeros(T * nseg, dtype=torch.int32)
    pf.scatter_(0, tl * nseg + seg, 1)  # dup (token, seg) writes 1 again
    flag2d = pf.view(T, nseg)
    pos = flag2d.cumsum(0) - flag2d  # exclusive rank within the segment
    seg_off_dev = torch.tensor(seg_off[:nseg], dtype=torch.long)
    tgt = (pos + seg_off_dev.unsqueeze(0)).masked_fill_(flag2d.eq(0), cpr)  # garbage
    pack_gather = torch.zeros(cpr + 1, dtype=torch.long)
    tgrid = torch.arange(T * nseg).div(nseg, rounding_mode="floor")
    pack_gather.scatter_(0, tgt.reshape(-1), tgrid)
    total = seg_off[nseg]
    # payload = global token id (unique per token, enough to verify routing)
    inputs_shard = torch.arange(rank * T, (rank + 1) * T, dtype=torch.long)
    send_buf = inputs_shard.index_select(0, pack_gather[:total])
    # per-segment flag counts must reproduce the u/U-derived sizes
    seg_len = flag2d.sum(0)
    for i in range(nseg):
        assert int(seg_len[i]) == seg_off[i + 1] - seg_off[i], "pack segment mismatch"
    return send_buf


# ------------------------------------------------------- forward-index builds


def fwd_build_identity(choosed_experts, rank, W, NN, L, T, E, topk, u_mat, U_mat):
    """Exact ATen sequence of the identity (original) build (:1016-1034)."""
    my_node, my_lr = rank // L, rank % L
    R = NN - 1
    cpr = T * topk
    e_all = choosed_experts.view(-1).long()
    fwd_col_off = [0] * (R * L)
    facc = 0
    for dn in range(1, NN):
        s = ((my_node + dn) % NN) * L + my_lr
        for dl in range(L):
            fwd_col_off[(dn - 1) * L + dl] = facc
            facc += int(u_mat[s, my_node * L + dl])
    fwd_idx = torch.full((R * T * topk + 1,), -1, dtype=torch.long)
    fwd_garbage = fwd_idx.numel() - 1
    tl = torch.arange(cpr).div(topk, rounding_mode="floor")
    e_src = (
        e_all.view(NN, L, cpr).select(1, my_lr).roll(-(my_node + 1), 0).narrow(0, 0, R)
    )
    dl = e_src.div(E, rounding_mode="floor").sub_(my_node * L)
    off_node = dl.lt(0).logical_or_(dl.ge(L))
    r_base = torch.arange(R).view(R, 1) * (T * L)
    fp = (r_base + tl.unsqueeze(0) * L + dl).masked_fill_(off_node, R * T * L)
    ff = torch.zeros(R * T * L + 1, dtype=torch.int32)
    ff.scatter_(0, fp.reshape(-1), 1)
    flag3d = ff.narrow(0, 0, R * T * L).view(R, T, L)
    uni = flag3d.max(2).values
    posU = uni.cumsum(1) - uni
    pos = flag3d.cumsum(1) - flag3d
    fwd_col_off_dev = torch.tensor(fwd_col_off, dtype=torch.long)
    tgt = (pos + fwd_col_off_dev.view(R, 1, L)).masked_fill_(flag3d.eq(0), fwd_garbage)
    vals = posU.unsqueeze(2).expand(R, T, L)
    fwd_idx.scatter_(0, tgt.reshape(-1), vals.reshape(-1))
    return fwd_idx, fwd_col_off


def fwd_build_relay(choosed_experts, rank, W, NN, L, T, E, topk, u_mat, U_mat):
    """Exact ATen sequence of the generalized (balanced-relay) build."""
    my_node, my_lr = rank // L, rank % L
    R = NN - 1
    cpr = T * topk
    e_all = choosed_experts.view(-1).long()
    # host tables (as staged into the meta arena)
    fwd_col_off = [0] * (R * L * L)
    facc = 0
    for dn in range(1, NN):
        ns = (my_node + dn) % NN
        for sl in range(L):
            s = ns * L + sl
            for dl in range(L):
                fwd_col_off[((dn - 1) * L + sl) * L + dl] = facc
                facc += int(u_mat[s, my_node * L + dl])
    idx_cap = R * T * L * min(topk, L)
    assert facc < idx_cap + 1, "fwd_idx capacity overflow"
    recv_start = [0] * (R * L)
    win_a, win_b = [0] * R, [0] * R
    for dn in range(1, NN):
        ns = (my_node + dn) % NN
        for sl in range(L):
            recv_start[(dn - 1) * L + sl] = canon_start(U_mat, NN, L, ns, sl, my_node)
        win_a[dn - 1] = chunk_bound(U_mat, NN, L, ns, my_node, my_lr)
        win_b[dn - 1] = chunk_bound(U_mat, NN, L, ns, my_node, my_lr + 1)
    # device build
    fwd_idx = torch.full((idx_cap + 1,), -1, dtype=torch.long)
    fwd_garbage = fwd_idx.numel() - 1
    tl = torch.arange(cpr).div(topk, rounding_mode="floor")
    e_rounds = e_all.view(NN, L, cpr).roll(-(my_node + 1), 0).narrow(0, 0, R)
    dl = e_rounds.div(E, rounding_mode="floor").sub_(my_node * L)
    off_node = dl.lt(0).logical_or_(dl.ge(L))
    rsl_base = torch.arange(R * L).view(R, L, 1) * (T * L)
    fp = (rsl_base + tl.view(1, 1, cpr) * L + dl).masked_fill_(off_node, R * L * T * L)
    ff = torch.zeros(R * L * T * L + 1, dtype=torch.int32)
    ff.scatter_(0, fp.reshape(-1), 1)
    flag4d = ff.narrow(0, 0, R * L * T * L).view(R, L, T, L)
    uni = flag4d.max(3).values
    posU = uni.cumsum(2) - uni
    recv_start_dev = torch.tensor(recv_start, dtype=torch.long)
    win_a_dev = torch.tensor(win_a, dtype=torch.long)
    win_b_dev = torch.tensor(win_b, dtype=torch.long)
    canon = posU + recv_start_dev.view(R, L, 1)
    in_w = canon.ge(win_a_dev.view(R, 1, 1)).logical_and_(canon.lt(win_b_dev.view(R, 1, 1)))
    below = canon.lt(win_a_dev.view(R, 1, 1))
    valid = flag4d * in_w.unsqueeze(3)
    pos = valid.cumsum(2) - valid
    fwd_col_off_dev = torch.tensor(fwd_col_off, dtype=torch.long)
    tgt = (pos + fwd_col_off_dev.view(R, L, 1, L)).masked_fill_(valid.eq(0), fwd_garbage)
    vals = (canon - win_a_dev.view(R, 1, 1)).unsqueeze(3).expand(R, L, T, L)
    fwd_idx.scatter_(0, tgt.reshape(-1), vals.reshape(-1))
    cnt_in = valid.sum(2)  # [R, L, L]
    cnt_before = (flag4d * below.unsqueeze(3)).sum(2)  # [R, L, L]
    # the debug-check invariants from the C++ build
    cnt = flag4d.sum(2)
    ucnt = uni.sum(2)
    for r in range(R):
        ns = (my_node + r + 1) % NN
        for sl in range(L):
            s = ns * L + sl
            for dlv in range(L):
                assert int(cnt[r, sl, dlv]) == int(u_mat[s, my_node * L + dlv])
                assert int(cnt_before[r, sl, dlv] + cnt_in[r, sl, dlv]) <= int(
                    u_mat[s, my_node * L + dlv]
                )
            assert int(ucnt[r, sl]) == int(U_mat[s, my_node])
    return fwd_idx, fwd_col_off, cnt_in, cnt_before


# ---------------------------------------------------------------- wire sims


class Recv:
    """Dedup recv buffer of one destination, with a write-occupancy counter."""

    def __init__(self, u_mat, W, d):
        self.total = int(u_mat[:, d].sum())
        self.buf = torch.full((max(self.total, 1),), -1, dtype=torch.long)
        self.hits = torch.zeros(max(self.total, 1), dtype=torch.int32)

    def write(self, off, rows):
        n = rows.numel()
        assert off >= 0 and off + n <= self.total, "recv write out of range"
        self.buf[off : off + n] = rows
        self.hits[off : off + n] += 1


def deliver_intra_and_self(recvs, sends, seg_offs, u_mat, W, NN, L):
    """Round 0 + self-copy: identical in both wire modes."""
    for r in range(W):
        my_node, my_lr = r // L, r % L
        for dl in range(L):
            d = my_node * L + dl
            rows = int(u_mat[r, d])
            if rows == 0:
                continue
            seg = my_node + dl  # send_seg_off(dlg) == seg_off_h[my_node + dlg]
            src = sends[r][seg_offs[r][seg] : seg_offs[r][seg] + rows]
            recvs[d].write(recv_off_of_u(u_mat, r, d), src)


def wire_identity(recvs, sends, seg_offs, choosed, W, NN, L, T, E, topk, u_mat, U_mat):
    """The original scheme: same-lr union aggregate + gateway forward."""
    for g in range(W):  # g = gateway rank on the receiving side
        my_node, my_lr = g // L, g % L
        fwd_idx, fwd_col_off = fwd_build_identity(
            choosed, g, W, NN, L, T, E, topk, u_mat, U_mat
        )
        for dn in range(1, NN):
            ns = (my_node + dn) % NN
            s = ns * L + my_lr
            # the staged union segment == source s's send segment for my node
            seg = my_node if my_node < ns else my_node + L - 1
            union_rows = int(U_mat[s, my_node])
            stage_seg = sends[s][seg_offs[s][seg] : seg_offs[s][seg] + union_rows]
            for dl in range(L):
                dlg = (my_lr - dl + L) % L
                d = my_node * L + dlg
                rows = int(u_mat[s, d])
                if rows == 0:
                    continue
                off = fwd_col_off[(dn - 1) * L + dlg]
                idx = fwd_idx[off : off + rows]
                assert bool((idx >= 0).all()) and bool((idx < union_rows).all())
                recvs[d].write(recv_off_of_u(u_mat, s, d), stage_seg.index_select(0, idx))


def wire_relay(recvs, sends, seg_offs, choosed, W, NN, L, T, E, topk, u_mat, U_mat):
    """The balanced relay: pieces -> relay staging -> wire chunk -> gateway
    window -> per-(src_lr, dest) delivery at recv_off + cnt_before.
    Returns (moved_rows, wire_rows) for the relocation statistics."""
    moved = 0
    wire = 0
    # phase 1: piece puts into per-relay staging, mirroring the exact C++
    # offset arithmetic (relay_round_base + (lo - a_k); own_only skip)
    def relay_round_base(n, k, dn):
        acc = 0
        for d2 in range(1, dn):
            acc += chunk_rows_of(U_mat, NN, L, n, (n - d2 + NN) % NN, k)
        return acc

    relay_stage = {}  # rank -> staging buffer (all rounds packed by round)
    for r in range(W):
        n, _ = r // L, r % L
        cap = sum(
            chunk_rows_of(U_mat, NN, L, n, (n - dn + NN) % NN, r % L)
            for dn in range(1, NN)
        )
        relay_stage[r] = torch.full((max(cap, 1),), -1, dtype=torch.long)
    for r in range(W):  # sender
        n, my_lr = r // L, r % L
        for dn in range(1, NN):
            tn = (n - dn + NN) % NN
            sstart = canon_start(U_mat, NN, L, n, my_lr, tn)
            send_end = sstart + int(U_mat[r, tn])
            seg = tn if tn < n else tn + L - 1
            for k in range(L):
                a_k = chunk_bound(U_mat, NN, L, n, tn, k)
                b_k = chunk_bound(U_mat, NN, L, n, tn, k + 1)
                lo, hi = max(a_k, sstart), min(b_k, send_end)
                if hi <= lo:
                    continue
                if k == my_lr and a_k >= sstart and b_k <= send_end:
                    continue  # own_only fast path: wire puts from the send buffer
                dst = relay_stage[n * L + k]
                off = relay_round_base(n, k, dn) + (lo - a_k)
                src_off = seg_offs[r][seg] + (lo - sstart)
                dst[off : off + (hi - lo)] = sends[r][src_off : src_off + (hi - lo)]
                if k != my_lr:
                    moved += hi - lo
    # phase 2: wire chunks (relay staging, or the send buffer when own_only)
    gate_stage = {}
    for r in range(W):  # relay
        n, my_lr = r // L, r % L
        for dn in range(1, NN):
            tn = (n - dn + NN) % NN
            a_me = chunk_bound(U_mat, NN, L, n, tn, my_lr)
            b_me = chunk_bound(U_mat, NN, L, n, tn, my_lr + 1)
            rows = b_me - a_me
            wire += rows
            if rows == 0:
                gate_stage[(tn * L + my_lr, dn)] = torch.empty(0, dtype=torch.long)
                continue
            sstart = canon_start(U_mat, NN, L, n, my_lr, tn)
            send_end = sstart + int(U_mat[r, tn])
            if a_me >= sstart and b_me <= send_end:  # own_only
                seg = tn if tn < n else tn + L - 1
                src_off = seg_offs[r][seg] + (a_me - sstart)
                chunk = sends[r][src_off : src_off + rows].clone()
            else:
                base = relay_round_base(n, my_lr, dn)
                chunk = relay_stage[r][base : base + rows].clone()
            assert bool((chunk >= 0).all()), "wire chunk has an unwritten row"
            gate_stage[(tn * L + my_lr, dn)] = chunk
    # gateway delivery
    for g in range(W):
        my_node, my_lr = g // L, g % L
        fwd_idx, fwd_col_off, cnt_in, cnt_before = fwd_build_relay(
            choosed, g, W, NN, L, T, E, topk, u_mat, U_mat
        )
        for dn in range(1, NN):
            ns = (my_node + dn) % NN
            win = gate_stage[(g, dn)]
            win_rows = win.numel()
            assert win_rows == chunk_rows_of(U_mat, NN, L, ns, my_node, my_lr)
            for dl in range(L):
                dlg = (my_lr - dl + L) % L
                d = my_node * L + dlg
                for sl in range(L):
                    cnt = int(cnt_in[dn - 1, sl, dlg])
                    if cnt == 0:
                        continue
                    s = ns * L + sl
                    off = fwd_col_off[((dn - 1) * L + sl) * L + dlg]
                    idx = fwd_idx[off : off + cnt]
                    assert bool((idx >= 0).all()) and bool(
                        (idx < win_rows).all()
                    ), "fwd index outside the staging window"
                    dst_off = recv_off_of_u(u_mat, s, d) + int(cnt_before[dn - 1, sl, dlg])
                    recvs[d].write(dst_off, win.index_select(0, idx))
    return moved, wire


# ------------------------------------------------------------ deadlock model


def deadlock_check(W, NN, L, u_mat, U_mat, epochs=2):
    """Event-driven executor over the three per-rank streams with epoch-signal
    semantics. Global stall with nonempty queues = deadlock."""
    signals = {}  # (rank, name, slot) -> value, monotone, never reset

    def sig(rank, name, slot):
        return signals.get((rank, name, slot), 0)

    for run_id in range(1, epochs + 1):
        progs = {}  # (rank, stream) -> list of ops
        for r in range(W):
            n, my_lr = r // L, r % L
            inter, cp, sg = [], [], []
            # phase 1: all piece puts (relay-in signals), no waits
            for dn in range(1, NN):
                tn = (n - dn + NN) % NN
                sstart = canon_start(U_mat, NN, L, n, my_lr, tn)
                send_end = sstart + int(U_mat[r, tn])
                for k in range(L):
                    a_k = chunk_bound(U_mat, NN, L, n, tn, k)
                    b_k = chunk_bound(U_mat, NN, L, n, tn, k + 1)
                    if min(b_k, send_end) <= max(a_k, sstart):
                        continue
                    if k == my_lr:
                        continue  # local memcpy or own fast path: no signal
                    inter.append(("set", (n * L + k, "relay", (dn - 1) * L + my_lr)))
            # phase 2: wire rounds — waits on contributors, then node_sig set
            for dn in range(1, NN):
                tn = (n - dn + NN) % NN
                g = tn * L + my_lr
                a_me = chunk_bound(U_mat, NN, L, n, tn, my_lr)
                b_me = chunk_bound(U_mat, NN, L, n, tn, my_lr + 1)
                sstart = canon_start(U_mat, NN, L, n, my_lr, tn)
                send_end = sstart + int(U_mat[r, tn])
                own_only = a_me >= sstart and b_me <= send_end
                if b_me > a_me and not own_only:
                    for sl in range(L):
                        if sl == my_lr:
                            continue
                        c0 = canon_start(U_mat, NN, L, n, sl, tn)
                        c1 = c0 + int(U_mat[n * L + sl, tn])
                        if min(b_me, c1) > max(a_me, c0):
                            inter.append(("wait", (r, "relay", (dn - 1) * L + sl)))
                inter.append(("set", (g, "node", n)))
            # gateway rounds on cp_stream: wait node_sig, then gw signals
            for dn in range(1, NN):
                ns = (n + dn) % NN
                cp.append(("wait", (r, "node", ns)))
                for dl in range(L):
                    d = n * L + (my_lr - dl + L) % L
                    cp.append(("set", (d, "gw", (dn - 1) * L + my_lr)))
            # signal aggregation on cp_stream_signal
            for dn in range(1, NN):
                ns = (n + dn) % NN
                for gl in range(L):
                    sg.append(("wait", (r, "gw", (dn - 1) * L + gl)))
                for sl in range(L):
                    sg.append(("set", (r, "src", ns * L + sl)))
            progs[(r, "inter")] = inter
            progs[(r, "cp")] = cp
            progs[(r, "sig")] = sg
        pc = {k: 0 for k in progs}
        while True:
            progressed = False
            for key, ops in progs.items():
                while pc[key] < len(ops):
                    op, target = ops[pc[key]]
                    if op == "wait" and sig(*target) < run_id:
                        break
                    if op == "set":
                        signals[target] = run_id  # SET, monotone across epochs
                    pc[key] += 1
                    progressed = True
            if all(pc[k] == len(progs[k]) for k in progs):
                break
            if not progressed:
                stuck = {k: progs[k][pc[k]] for k in progs if pc[k] < len(progs[k])}
                raise AssertionError(f"deadlock at epoch {run_id}: {stuck}")
        # completeness: every remote source signal reached this epoch
        for d in range(W):
            for s in range(W):
                if s // L != d // L:
                    assert sig(d, "src", s) == run_id, "missing per-source signal"


# ------------------------------------------------------------------- driver


def reference_recv(choosed, W, NN, L, T, E, topk, u_mat):
    """Direct dedup layout: (s, d) region = ascending unique tokens of s
    needed by d (u/U tie every wire mode to this)."""
    ntokens = W * T
    owner = choosed.long() // E
    flags = torch.zeros(ntokens, W, dtype=torch.bool)
    flags.scatter_(1, owner, True)
    out = {}
    for d in range(W):
        parts = []
        for s in range(W):
            toks = torch.nonzero(flags[s * T : (s + 1) * T, d], as_tuple=False).view(-1)
            parts.append(toks + s * T)
        out[d] = torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)
        assert out[d].numel() == int(u_mat[:, d].sum())
    return out


def run_case(NN, L, T, E, topk, seed, skew):
    W = NN * L
    nexperts = W * E
    ntokens = W * T
    gen = torch.Generator().manual_seed(seed)
    if skew == "uniform_u":
        # every source needs every node equally: route copy (t, j) to expert
        # owned by rank (t + j) % W -> U is exactly uniform
        t = torch.arange(ntokens).view(-1, 1)
        j = torch.arange(topk).view(1, -1)
        choosed = (((t + j) % W) * E + (t % E)).int()
    elif skew == "hot":
        # concentrate on a few experts of one node
        hot = torch.randint(0, 2 * E, (ntokens, topk), generator=gen)
        cold = torch.randint(0, nexperts, (ntokens, topk), generator=gen)
        pick = torch.rand(ntokens, topk, generator=gen) < 0.7
        choosed = torch.where(pick, hot, cold).int()
    elif skew == "zero_source":
        # rank 0's tokens never leave node 0 -> U[0][n>0] == 0
        choosed = torch.randint(0, nexperts, (ntokens, topk), generator=gen).int()
        choosed[:T] = torch.randint(0, L * E, (T, topk), generator=gen).int()
    elif skew == "tiny":
        # nearly nothing crosses nodes: total < L for most rounds (zero chunks)
        choosed = (torch.arange(ntokens).view(-1, 1) % E).expand(-1, topk).clone().int()
        choosed = (choosed + (torch.arange(ntokens).view(-1, 1) // T) * L * E % nexperts).int()
        choosed[0, 0] = (nexperts - 1)  # a single cross-node token
    else:
        choosed = torch.randint(0, nexperts, (ntokens, topk), generator=gen).int()

    u_mat, U_mat, _ = build_uc(choosed, W, NN, L, T, E)
    seg_offs = [seg_offsets(u_mat, U_mat, W, NN, L, r) for r in range(W)]
    sends = [
        producer_pack(choosed, r, W, NN, L, T, E, topk, seg_offs[r]) for r in range(W)
    ]
    ref = reference_recv(choosed, W, NN, L, T, E, topk, u_mat)

    results = {}
    for mode in ("identity", "relay"):
        recvs = [Recv(u_mat, W, d) for d in range(W)]
        deliver_intra_and_self(recvs, sends, seg_offs, u_mat, W, NN, L)
        if mode == "identity":
            wire_identity(recvs, sends, seg_offs, choosed, W, NN, L, T, E, topk, u_mat, U_mat)
            moved = wire = None
        else:
            moved, wire = wire_relay(
                recvs, sends, seg_offs, choosed, W, NN, L, T, E, topk, u_mat, U_mat
            )
        for d in range(W):
            total = recvs[d].total
            assert bool(
                (recvs[d].hits[:total] == 1).all()
            ), f"{mode}: recv row written != once at dest {d}"
            assert torch.equal(
                recvs[d].buf[:total], ref[d]
            ), f"{mode}: recv mismatch at dest {d}"
        results[mode] = recvs

    # balance: within each (node, round), chunks differ by <= 1 row
    for n in range(NN):
        for dn in range(1, NN):
            tn = (n - dn + NN) % NN
            sizes = [chunk_rows_of(U_mat, NN, L, n, tn, k) for k in range(L)]
            assert max(sizes) - min(sizes) <= 1, "unbalanced chunks"
    if skew == "uniform_u":
        for n in range(NN):
            for m in range(NN):
                if m == n:
                    continue
                segs = [int(U_mat[n * L + sl, m]) for sl in range(L)]
                assert len(set(segs)) == 1, f"uniform_u construction not uniform: {segs}"
        assert moved == 0, f"uniform U must relocate zero rows, moved {moved}"

    deadlock_check(W, NN, L, u_mat, U_mat, epochs=2)
    return moved, wire


def main():
    topos = [(2, 2), (2, 4), (2, 8), (4, 2), (4, 4), (3, 4)]
    skews = ["rand", "hot", "uniform_u", "zero_source", "tiny"]
    seeds = [0, 1, 2, 3, 4]
    # T divisible by every W in the sweep so the uniform_u construction is
    # exactly uniform (dest ranks cycle through all residues mod W)
    T, E = 48, 3
    cases = 0
    for NN, L in topos:
        for topk in (2, 4):
            for skew in skews:
                for seed in seeds if skew in ("rand", "hot") else seeds[:1]:
                    moved, wire = run_case(NN, L, T, E, topk, seed, skew)
                    cases += 1
        print(f"topology NN={NN} L={L}: ok (last case moved {moved}/{wire} wire rows)")
    print(f"PASS: {cases} cases (recv identity, coverage, balance, deadlock-free x2 epochs)")


if __name__ == "__main__":
    main()
