"""Replicated multi-node port of the MoonEP fused planning kernel.

Vendored/adapted from MoonshotAI/MoonEP @ 7745ffa (moonep/planning.py, plus
the grid_sync barrier and its atomics from moonep/_common.py). ONE
cooperative CuTe-DSL kernel computes the FULL replicated plan —
dst_all[R,N], cu_all[R,E+B], etc_all[R,B], zfr_all[R,E+B,2],
stats_all[R,2] — from the globally-known routing (topk_all[R,N],
tpe_all[R,E]). This is the rule-5 per-iteration timed planner for the flux
staged MoonEP arm: authentic upstream planning math, launched inside the
driver's `plan` event bracket every iteration.

Deviations from upstream (2026-08-20 port decision: exact authenticity,
replace ONLY the cross-node-synchronization portions; upstream line refs):
  1. Inputs are the replicated topk_all/tpe_all (the driver's timed
     plan_comm allgather makes routing globally known), so the per-rank
     tpe/topk remote pushes (:603-608) and the rank-0 centralization guard
     (:610) are gone — Phase A runs identically on every rank over a LOCAL
     int32 scratch tensor (same PLAN-region sub-offsets, base 0).
  2. All three cross_rank_barriers deleted (:609, :979, :1077); grid_sync
     alone orders phases (nothing crosses ranks any more).
  3. The multimem plan broadcast (:955-960) deleted — it was the only
     multimem.st use (sm90-only).
  4. C1 runs in a source loop over a local order_all[R,N] scratch; the
     rank-1 offload of rank-0's C1 and its ORDER writeback (:961-970) are
     gone.
  5. C2 loops all sources and writes dst_all[r]; the src_info remote
     provenance writes (:1071-1073) are deleted — the flux port derives
     dedup pairs from the replicated dst (derive_moonep_layout_gpu), it
     has no in-kernel dispatch builder.
  6. Phase D (cp.async.bulk rank-slice staging — the second and last
     sm90-only dependency) deleted: Phase A writes the full [R,*] output
     tensors directly, so there is nothing to copy back.
  7. Dedup canonicalization grid-strides all R*S tokens (was: own S).

Everything else — the vblock histogram/scan machinery, the single-warp
balance and quota-alloc greedy loops, the top-B remote-expert selection,
the tie-break reductions, and the segment/padding math — is upstream code
verbatim. Surviving primitives (cooperative launch, match.any.sync,
redux.sync, shfl, gpu-scope atomics, st.global.v4) are all sm80-legal.
"""

import functools

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.runtime import make_ptr

# ============================================================
# Compile-time constants (upstream planning.py:93-96)
# ============================================================
BLOCK_SIZE_P2 = 2048
BLOCK_DIM_P2 = 512
ITEMS_PER_THREAD_P2 = BLOCK_SIZE_P2 // BLOCK_DIM_P2  # 4

GRID_SYNC_TAG = 0x80000000


def ceil_div(x, y):
    return (x + y - 1) // y


def align_up(x, alignment):
    return ceil_div(x, alignment) * alignment


def ceil_pow2(x):
    return 1 << max(x - 1, 0).bit_length()


def log2_r(R):
    return max(R.bit_length(), 1)


# ============================================================
# Low-level helpers (vendored: _common.py atomics + grid_sync,
# planning.py match_any/st_v4/scans)
# ============================================================


@dsl_user_op
def atom_add_release_gpu(ptr_i64, val, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(
        T.i32(), [ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "atom.add.release.gpu.global.s32 $0, [$1], $2;", "=r,l,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def ld_acquire_gpu_s32(ptr_i64, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(
        T.i32(), [ptr_i64],
        "ld.acquire.gpu.global.s32 $0, [$1];", "=r,l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@cute.jit
def grid_sync(bar_ptr, nsm: Int32, tid: Int32):
    """Self-resetting cooperative grid barrier (upstream _common.py:249)."""
    cute.arch.sync_threads()
    if tid == 0:
        b0 = bar_ptr.toint().ir_value()
        pid = cute.arch.block_idx()[0]
        inc = Int32(1)
        if pid == 0:
            inc = Int32(GRID_SYNC_TAG) - (nsm - 1)
        old = atom_add_release_gpu(b0, inc)
        done = cutlass.Boolean(False)
        while not done:
            new = ld_acquire_gpu_s32(b0)
            done = ((new ^ old) & GRID_SYNC_TAG) != 0
    cute.arch.sync_threads()


@dsl_user_op
def match_any_b32(val, *, loc=None, ip=None) -> Uint32:
    return Uint32(llvm.inline_asm(
        T.i32(), [Uint32(val).ir_value(loc=loc, ip=ip),
                  Uint32(0xFFFFFFFF).ir_value(loc=loc, ip=ip)],
        "match.any.sync.b32 $0, $1, $2;", "=r,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@cute.jit
def warp_inclusive_scan(v, lane):
    offset = 1
    for scan_step in cutlass.range_constexpr(5):
        y = cute.arch.shuffle_sync(v, lane - offset)
        if lane >= offset:
            v += y
        offset <<= 1
    return v


@cute.jit
def warp_exclusive_scan_e(s_hist, E: cutlass.Constexpr, tid):
    CHUNK = cutlass.const_expr(ceil_div(E, 32))
    if tid < 32:
        lane = tid
        csum = 0
        for j in cutlass.range_constexpr(CHUNK):
            idx = lane * CHUNK + j
            v = 0
            if idx < E:
                v = s_hist[idx]; s_hist[idx] = csum
            csum += v
        x = warp_inclusive_scan(csum, lane)
        offset = x - csum
        for j in cutlass.range_constexpr(CHUNK):
            idx = lane * CHUNK + j
            if idx < E: s_hist[idx] += offset
    cute.arch.barrier()


@cute.jit
def elem_ptr(tensor, coord):
    return tensor.iterator + tensor.layout(coord)


@cute.jit
def warp_argmax_min_idx(v, i):
    m = cute.arch.warp_redux_sync(v, "max")
    j = cute.arch.warp_redux_sync(i if v == m else 2147483647, "min")
    return m, j


@cute.jit
def warp_argmax_max_idx(v, i):
    m = cute.arch.warp_redux_sync(v, "max")
    j = cute.arch.warp_redux_sync(i if v == m else -1, "max")
    return m, j


@cute.jit
def warp_argmin_min_idx(v, i):
    m = cute.arch.warp_redux_sync(v, "min")
    j = cute.arch.warp_redux_sync(i if v == m else 2147483647, "min")
    return m, j


@cute.jit
def reg_scan_argmax_min_idx(reg, N: cutlass.Constexpr, lane):
    bv = -2147483648; bi = 2147483647
    for j in cutlass.range_constexpr(ceil_div(N, 32)):
        k = lane + j * 32
        if k < N and reg[j] > bv: bv = reg[j]; bi = k
    return warp_argmax_min_idx(bv, bi)


@cute.jit
def reg_scan_argmax_max_idx(reg, N: cutlass.Constexpr, lane):
    bv = 0; bi = -1
    CHUNK = cutlass.const_expr(ceil_div(N, 32))
    for j in cutlass.range(CHUNK, unroll_full=True):
        k = lane + j * 32
        if k < N and reg[j] >= bv: bv = reg[j]; bi = k
    return warp_argmax_max_idx(bv, bi)


@cute.jit
def reg_scan_argmin_min_idx(reg, N: cutlass.Constexpr, lane):
    bv = 2147483647; bi = 2147483647
    for j in cutlass.range_constexpr(ceil_div(N, 32)):
        k = lane + j * 32
        if k < N and reg[j] < bv: bv = reg[j]; bi = k
    return warp_argmin_min_idx(bv, bi)


class ReplicatedPlanningKernel:
    def __init__(self, R, E, B, S, K, NvS_capacity, NvS, num_vblocks,
                 token_padding, num_sms):
        self.R, self.E, self.B, self.S, self.K = R, E, B, S, K
        self.N = self.S * self.K
        self.NvS_capacity, self.NvS = NvS_capacity, NvS
        self.num_vblocks = num_vblocks
        self.token_padding, self.num_sms = token_padding, num_sms

    @cute.jit
    def __call__(self, tpe_all, topk_all, scratch, dst_all, cu_all,
                 etc_all, zfr_all, stats_all, alloc, group_tokens, z,
                 local_hist, order_all, bar, stream: cuda.CUstream):
        R = cutlass.const_expr(self.R)
        N = cutlass.const_expr(self.N)
        num_sms = cutlass.const_expr(self.num_sms)
        tpe_t = cute.make_tensor(tpe_all, cute.make_layout((R * self.E,)))
        topk_t = cute.make_tensor(topk_all, cute.make_layout((R * N,)))
        scr_t = cute.make_tensor(scratch,
                                 cute.make_layout((3 * self.E * R,)))
        dst_t = cute.make_tensor(dst_all, cute.make_layout((R * N,)))
        cu_t = cute.make_tensor(
            cu_all, cute.make_layout((R * (self.E + self.B),)))
        etc_t = cute.make_tensor(etc_all,
                                 cute.make_layout((R * self.B,)))
        zfr_t = cute.make_tensor(
            zfr_all, cute.make_layout((R * (self.E + self.B) * 2,)))
        stats_t = cute.make_tensor(stats_all,
                                   cute.make_layout((R * 2,)))
        alloc_t = cute.make_tensor(alloc,
                                   cute.make_layout((R * self.E,)))
        gt_t = cute.make_tensor(group_tokens, cute.make_layout((R,)))
        z_t = cute.make_tensor(z, cute.make_layout((R * R,)))
        lh_t = cute.make_tensor(
            local_hist, cute.make_layout((self.num_vblocks * self.E,)))
        ord_t = cute.make_tensor(order_all, cute.make_layout((R * N,)))
        bar_t = cute.make_tensor(bar, cute.make_layout((1,)))

        self.kernel(tpe_t, topk_t, scr_t, dst_t, cu_t, etc_t, zfr_t,
                    stats_t, alloc_t, gt_t, z_t, lh_t, ord_t,
                    bar_t).launch(
            grid=(num_sms, 1, 1), block=(BLOCK_DIM_P2, 1, 1),
            stream=stream, cooperative=True)

    # =========================================================
    # run_c1: upstream verbatim (planning.py:398-517) — 1a histogram /
    # 1b vblock prefix / expoff / passA scatter, per source
    # =========================================================
    @cute.jit
    def run_c1(self, topk_src, order_dst, tpe_src, local_hist, s_hist,
               s_bp, s_wcount, bar_ptr, num_sms, pid, tid):
        E = cutlass.const_expr(self.E)
        NUM_WARPS = cutlass.const_expr(BLOCK_DIM_P2 // 32)
        WST = cutlass.const_expr(NUM_WARPS + 1)
        IPT = cutlass.const_expr(ITEMS_PER_THREAD_P2)
        N = cutlass.const_expr(self.N)
        num_vblocks = cutlass.const_expr(self.num_vblocks)
        num_threads = BLOCK_DIM_P2
        warp = tid >> 5
        lane = tid & 31

        topk_in = cute.make_tensor(topk_src.iterator,
                                   cute.make_layout((N,)))
        order_out = cute.make_tensor(order_dst.iterator,
                                     cute.make_layout((N,)))
        tpe_counts = cute.make_tensor(tpe_src.iterator,
                                      cute.make_layout((E,)))
        vblocks_histogram = cute.make_tensor(
            local_hist.iterator,
            cute.make_layout((num_vblocks, E), stride=(E, 1)),
        )
        s_histogram = cute.make_tensor(s_hist.iterator,
                                       cute.make_layout((E,)))
        s_block_prefix = cute.make_tensor(s_bp.iterator,
                                          cute.make_layout((E,)))
        s_warp_counts = cute.make_tensor(
            s_wcount.iterator,
            cute.make_layout((E + 1, WST), stride=(WST, 1)),
        )

        # 1a
        for vb in cutlass.range(pid, num_vblocks, num_sms):
            for e in cutlass.range(tid, E, num_threads):
                s_histogram[e] = 0
            cute.arch.barrier()
            chunk = vb * BLOCK_SIZE_P2
            for p in cutlass.range(tid, BLOCK_SIZE_P2, num_threads):
                off = chunk + p
                if off < N:
                    expert = topk_in[off]
                    cute.arch.atomic_add(elem_ptr(s_histogram, expert), 1,
                                         scope="cta")
            cute.arch.barrier()
            for e in cutlass.range(tid, E, num_threads):
                vblocks_histogram[vb, e] = s_histogram[e]
            cute.arch.barrier()

        grid_sync(bar_ptr, num_sms, tid)
        E_SEG = 32
        seg_raw = cute.ceil_div(E, num_sms)
        experts_per_block = cute.round_up(seg_raw, E_SEG)
        e_lo = pid * experts_per_block
        e_hi = cutlass.min(e_lo + experts_per_block, E)
        for e in cutlass.range(e_lo + tid, e_hi, num_threads):
            cumsum = 0
            for vb in cutlass.range_constexpr(num_vblocks):
                v = vblocks_histogram[vb, e]
                vblocks_histogram[vb, e] = cumsum
                cumsum += v

        grid_sync(bar_ptr, num_sms, tid)
        for e in cutlass.range(tid, E, num_threads):
            s_histogram[e] = tpe_counts[e]
        cute.arch.barrier()
        warp_exclusive_scan_e(s_histogram, E, tid)
        lanes_lt = (Uint32(1) << lane) - Uint32(1)
        for vb in cutlass.range(pid, num_vblocks, num_sms):
            chunk = vb * BLOCK_SIZE_P2
            my_e = []; my_p = []
            for i in cutlass.range_constexpr(IPT):
                p = warp * (32 * IPT) + i * 32 + lane
                off = chunk + p
                my_p.append(p)
                ev = E
                if off < N:
                    ev = topk_in[off]
                my_e.append(ev)

            for idx in cutlass.range(tid, (E + 1) * WST, num_threads):
                expert_idx = idx // WST
                warp_slot = idx - expert_idx * WST
                s_warp_counts[expert_idx, warp_slot] = 0
            cute.arch.barrier()

            for e in cutlass.range(tid, E, num_threads):
                s_block_prefix[e] = vblocks_histogram[vb, e]

            ww = []
            for i in cutlass.range_constexpr(IPT):
                peers = match_any_b32(my_e[i])
                cell = elem_ptr(s_warp_counts, (my_e[i], warp))
                base = cute.arch.load(cell, Int32)
                ww.append(base + Int32(cute.arch.popc(peers & lanes_lt)))
                cute.arch.sync_warp()
                if (peers & lanes_lt) == Uint32(0):
                    cute.arch.store(cell,
                                    base + Int32(cute.arch.popc(peers)))
                cute.arch.sync_warp()
            cute.arch.barrier()

            for e in cutlass.range(tid, E, num_threads):
                cumsum = 0
                for w in cutlass.range_constexpr(NUM_WARPS):
                    c = s_warp_counts[e, w]
                    s_warp_counts[e, w] = cumsum
                    cumsum += c
            cute.arch.barrier()

            for i in cutlass.range_constexpr(IPT):
                ei = my_e[i]
                if ei < E:
                    within = s_warp_counts[ei, warp] + ww[i]
                    sp = s_histogram[ei] + s_block_prefix[ei] + within
                    order_out[sp] = chunk + my_p[i]
            cute.arch.barrier()

        grid_sync(bar_ptr, num_sms, tid)

    @cute.kernel
    def kernel(self, tpe_all, topk_all, scratch, dst_all, cu_all, etc_all,
               zfr_all, stats_all, alloc, group_tokens, z, lh, order_all,
               bar):
        R = cutlass.const_expr(self.R)
        E = cutlass.const_expr(self.E)
        B = cutlass.const_expr(self.B)
        S = cutlass.const_expr(self.S)
        K = cutlass.const_expr(self.K)
        epn = cutlass.const_expr(E // R)
        LOG2_R = cutlass.const_expr(log2_r(R))
        EB_PAD = cutlass.const_expr(ceil_pow2(E + B))
        IPT_EB = cutlass.const_expr(ceil_div(EB_PAD, BLOCK_DIM_P2))
        N = cutlass.const_expr(self.N)
        NvS = cutlass.const_expr(self.NvS)
        CAP = cutlass.const_expr(self.NvS_capacity)
        tp = cutlass.const_expr(self.token_padding)
        num_sms = cutlass.const_expr(self.num_sms)
        # local PLAN-region sub-offsets (upstream PB=PLAN_OFF, base 0 here)
        ALLOC_SUB = 0
        TPE_SUB = E * R
        EOFF_SUB = 2 * E * R
        num_threads = BLOCK_DIM_P2
        NUM_WARPS = cutlass.const_expr(BLOCK_DIM_P2 // 32)
        S1_TILE = 32
        S1_COLS = cutlass.const_expr(min(
            align_up((E + num_sms - 1) // num_sms, S1_TILE),
            BLOCK_DIM_P2,
        ))
        pid = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]

        smem = utils.SmemAllocator()

        def sa(n):
            align_elems = 16
            aligned_n = align_up(n, align_elems)
            return smem.allocate_tensor(Int32, cute.make_layout((aligned_n,)),
                                        byte_alignment=16)

        scratch_ints = cutlass.const_expr(max(
            R * S1_COLS,
            E + E // R + R,
            (E + 1) * (BLOCK_DIM_P2 // 32 + 1),
        ))
        s_scratch = sa(scratch_ints)
        s_hist = sa(E)
        s_bp = sa(E)
        s_col = sa(E)
        s_chosen = sa(B)
        s_wmax = sa(64)
        s_mask = sa(E)
        bar_p = bar.iterator

        # ---- Phase A (upstream :610-953, rank-0 guard removed: every
        # rank computes the identical full plan on local scratch) ----
        z_tensor = cute.make_tensor(
            z.iterator, cute.make_layout((R, R), stride=(R, 1)))
        for i in cutlass.range(
            pid * num_threads + tid, R * R, num_sms * num_threads
        ):
            z[i] = 0
        for i in cutlass.range(
            pid * num_threads + tid, R, num_sms * num_threads
        ):
            group_tokens[i] = 0
        grid_sync(bar_p, num_sms, tid)
        seg_raw = cute.ceil_div(E, num_sms)
        experts_per_block = cute.round_up(seg_raw, S1_TILE)
        start_idx = pid * experts_per_block
        end_idx = cutlass.min(start_idx + experts_per_block, E)
        tpe_gather = cute.make_tensor(
            tpe_all.iterator, cute.make_layout((R, E), stride=(E, 1)))
        tpe_cumsum = cute.make_tensor(
            scratch.iterator + TPE_SUB,
            cute.make_layout((R, E), stride=(E, 1)))
        s_tpe = cute.make_tensor(
            s_scratch.iterator,
            cute.make_layout((R, S1_COLS), stride=(S1_COLS, 1)))
        for e0 in cutlass.range(start_idx, end_idx, S1_COLS):
            for idx in cutlass.range(tid, R * S1_COLS, num_threads):
                r = idx // S1_COLS
                col = idx - r * S1_COLS
                expert_idx = e0 + col
                v = 0
                if expert_idx < end_idx:
                    v = tpe_gather[r, expert_idx]
                s_tpe[r, col] = v
            cute.arch.barrier()
            if tid < S1_COLS:
                expert_idx = e0 + tid
                if expert_idx < end_idx:
                    run = 0
                    for r in cutlass.range_constexpr(R):
                        run += s_tpe[r, tid]; s_tpe[r, tid] = run
                    cute.arch.atomic_add(
                        group_tokens.iterator + expert_idx // epn, run,
                        scope="gpu")
            cute.arch.barrier()
            for idx in cutlass.range(tid, R * S1_COLS, num_threads):
                r = idx // S1_COLS
                col = idx - r * S1_COLS
                expert_idx = e0 + col
                if expert_idx < end_idx:
                    tpe_cumsum[r, expert_idx] = s_tpe[r, col]
            cute.arch.barrier()
        grid_sync(bar_p, num_sms, tid)
        if pid == 0:
            if tid < 32:
                lane = tid
                CHUNK = cutlass.const_expr(ceil_div(R, 32))
                balance = cute.make_rmem_tensor(CHUNK, Int32)
                for j in cutlass.range_constexpr(CHUNK):
                    k = lane + j * 32
                    balance[j] = 0
                    if k < R: balance[j] = group_tokens[k] - CAP
                keep_balancing = True
                while keep_balancing:
                    surplus, surplus_rank = reg_scan_argmax_min_idx(
                        balance, R, lane)
                    deficit, deficit_rank = reg_scan_argmin_min_idx(
                        balance, R, lane)
                    if surplus <= 0 or deficit >= 0:
                        keep_balancing = False
                    else:
                        move_tokens = -deficit
                        for j in cutlass.range_constexpr(CHUNK):
                            k = lane + j * 32
                            if k == surplus_rank: balance[j] -= move_tokens
                            elif k == deficit_rank: balance[j] = 0
                        if lane == 0:
                            z_tensor[surplus_rank, deficit_rank] = move_tokens
                        cute.arch.sync_warp()
        grid_sync(bar_p, num_sms, tid)
        alloc_cumsum = cute.make_tensor(
            scratch.iterator + ALLOC_SUB,
            cute.make_layout((E, R), stride=(R, 1)))
        alloc_tensor = cute.make_tensor(
            alloc.iterator, cute.make_layout((R, E), stride=(E, 1)))
        s_alloc = cute.make_tensor(
            s_scratch.iterator,
            cute.make_layout((R, epn), stride=(epn, 1)))
        for owner_rank in cutlass.range(pid, R, num_sms):
            expert_base = owner_rank * epn
            for idx in cutlass.range(tid, epn * R, num_threads):
                local_expert_id = idx // R
                rank_idx = idx - local_expert_id * R
                global_expert = expert_base + local_expert_id
                s_alloc[rank_idx, local_expert_id] = (
                    tpe_cumsum[R - 1, global_expert]
                    if rank_idx == owner_rank else 0
                )
            cute.arch.barrier()
            if tid < 32:
                lane = tid
                R_CHUNK = cutlass.const_expr(ceil_div(R, 32))
                EPN_CHUNK = cutlass.const_expr(ceil_div(epn, 32))
                quotas = cute.make_rmem_tensor(R_CHUNK, Int32)
                owner_remaining = cute.make_rmem_tensor(EPN_CHUNK, Int32)
                for j in cutlass.range_constexpr(R_CHUNK):
                    rank_idx = lane + j * 32
                    quotas[j] = 0
                    if rank_idx < R:
                        quotas[j] = z_tensor[owner_rank, rank_idx]
                for j in cutlass.range_constexpr(EPN_CHUNK):
                    local_expert_id = lane + j * 32
                    owner_remaining[j] = 0
                    if local_expert_id < epn:
                        owner_remaining[j] = s_alloc[owner_rank,
                                                     local_expert_id]
                keep_balancing = cutlass.Boolean(True)
                while keep_balancing:
                    max_quota, target_rank = reg_scan_argmax_min_idx(
                        quotas, R, lane)
                    if max_quota <= 0:
                        keep_balancing = cutlass.Boolean(False)
                    else:
                        max_remaining, selected_expert_id = \
                            reg_scan_argmax_min_idx(owner_remaining, epn,
                                                    lane)
                        if max_remaining <= 0:
                            keep_balancing = cutlass.Boolean(False)
                        else:
                            take = cutlass.min(max_remaining, max_quota)
                            for j in cutlass.range_constexpr(R_CHUNK):
                                rank_idx = lane + j * 32
                                if rank_idx == target_rank:
                                    quotas[j] = max_quota - take
                            for j in cutlass.range_constexpr(EPN_CHUNK):
                                local_expert_id = lane + j * 32
                                if local_expert_id == selected_expert_id:
                                    owner_remaining[j] = max_remaining - take
                            if tid == 0:
                                s_alloc[target_rank,
                                        selected_expert_id] += take
                                s_alloc[owner_rank, selected_expert_id] = (
                                    max_remaining - take)
                            cute.arch.sync_warp()
            cute.arch.barrier()
            for idx in cutlass.range(tid, epn * R, num_threads):
                rank_idx = idx // epn
                local_expert_id = idx - rank_idx * epn
                global_expert = expert_base + local_expert_id
                alloc_tensor[rank_idx, global_expert] = (
                    s_alloc[rank_idx, local_expert_id])
            cute.arch.barrier()
            for local_expert_id in cutlass.range(tid, epn, num_threads):
                cum = 0
                for rank_idx in cutlass.range_constexpr(R):
                    cum += s_alloc[rank_idx, local_expert_id]
                    s_alloc[rank_idx, local_expert_id] = cum
            cute.arch.barrier()
            for idx in cutlass.range(tid, epn * R, num_threads):
                local_expert_id = idx // R
                rank_idx = idx - local_expert_id * R
                global_expert = expert_base + local_expert_id
                alloc_cumsum[global_expert, rank_idx] = (
                    s_alloc[rank_idx, local_expert_id])
            cute.arch.barrier()
        grid_sync(bar_p, num_sms, tid)
        expert_offsets = cute.make_tensor(
            scratch.iterator + EOFF_SUB,
            cute.make_layout((R, E), stride=(E, 1)))
        # Deviation 6: Phase A writes the FULL [R, *] outputs directly.
        all_cu_seqlens = cute.make_tensor(
            cu_all.iterator,
            cute.make_layout((R, E + B), stride=(E + B, 1)))
        zero_fill_start = cute.make_tensor(
            zfr_all.iterator,
            cute.make_layout((R, E + B), stride=((E + B) * 2, 2)))
        zero_fill_count = cute.make_tensor(
            zfr_all.iterator + 1,
            cute.make_layout((R, E + B), stride=((E + B) * 2, 2)))
        all_experts_to_copy = cute.make_tensor(
            etc_all.iterator, cute.make_layout((R, B), stride=(B, 1)))
        all_remote_stats = cute.make_tensor(
            stats_all.iterator, cute.make_layout((R, 2), stride=(2, 1)))
        s_expert_counts = cute.make_tensor(s_col.iterator,
                                           cute.make_layout((E,)))
        s_selected_experts = cute.make_tensor(s_chosen.iterator,
                                              cute.make_layout((B,)))
        s_selected_mask = cute.make_tensor(s_mask.iterator,
                                           cute.make_layout((E,)))
        s_scan_warp_prefix = cute.make_tensor(
            s_wmax.iterator, cute.make_layout((NUM_WARPS,)))
        for idx in cutlass.range(
            pid * num_threads + tid, R * 2, num_sms * num_threads
        ):
            stat_rank = idx // 2
            stat_idx = idx - stat_rank * 2
            all_remote_stats[stat_rank, stat_idx] = 0
        grid_sync(bar_p, num_sms, tid)
        for dest_rank in cutlass.range(pid, R, num_sms):
            local_start = dest_rank * epn
            local_end = local_start + epn
            for expert_idx in cutlass.range(tid, E, num_threads):
                cnt = alloc_tensor[dest_rank, expert_idx]
                s_expert_counts[expert_idx] = cnt
                s_selected_mask[expert_idx] = 0
            cute.arch.barrier()
            if tid < 32:
                lane = tid
                E_CHUNK = cutlass.const_expr(ceil_div(E, 32))
                remote_expert_counts = cute.make_rmem_tensor(E_CHUNK, Int32)
                for j in cutlass.range(E_CHUNK, unroll_full=True):
                    expert_idx = lane + j * 32
                    remote_expert_counts[j] = 0
                    if expert_idx < E:
                        cnt = s_expert_counts[expert_idx]
                        is_local = ((expert_idx >= local_start)
                                    & (expert_idx < local_end))
                        remote_expert_counts[j] = 0 if is_local else cnt
                remote_expert_count = 0
                for j in cutlass.range(E_CHUNK, unroll_full=True):
                    if remote_expert_counts[j] > 0:
                        remote_expert_count += 1
                remote_expert_count = cute.arch.warp_redux_sync(
                    remote_expert_count, "add")
                if tid == 0:
                    all_remote_stats[dest_rank, 0] = remote_expert_count
                for slot in cutlass.range_constexpr(B):
                    best_cnt, best_idx = reg_scan_argmax_max_idx(
                        remote_expert_counts, E, lane)
                    for j in cutlass.range(E_CHUNK, unroll_full=True):
                        expert_idx = lane + j * 32
                        if expert_idx == best_idx:
                            remote_expert_counts[j] = 0
                    if tid == 0:
                        expert_idx = best_idx if best_cnt > 0 else -1
                        s_selected_experts[slot] = expert_idx
                        all_experts_to_copy[dest_rank, slot] = expert_idx
                        if expert_idx >= 0:
                            owner_rank = expert_idx // epn
                            cute.arch.atomic_add(
                                elem_ptr(all_remote_stats,
                                         (owner_rank, 1)),
                                1, scope="gpu")
                            s_selected_mask[expert_idx] = 1
                    cute.arch.sync_warp()
            cute.arch.barrier()

            count_values = []
            expert_values = []
            padded_values = []
            for i in cutlass.range_constexpr(IPT_EB):
                group_idx = tid * IPT_EB + i
                token_count = 0
                expert_id = -1
                if group_idx < E + B:
                    if group_idx < E:
                        is_selected = s_selected_mask[group_idx] != 0
                        if ~is_selected:
                            token_count = s_expert_counts[group_idx]
                            expert_id = group_idx
                    else:
                        selected_expert = s_selected_experts[group_idx - E]
                        if selected_expert >= 0:
                            token_count = s_expert_counts[selected_expert]
                            expert_id = selected_expert
                padded_count = 0
                if token_count > 0:
                    if cutlass.const_expr(tp > 1):
                        padded_count = cute.round_up(token_count, tp)
                    else:
                        padded_count = token_count
                count_values.append(token_count)
                expert_values.append(expert_id)
                padded_values.append(padded_count)

            total_padded = 0
            for i in cutlass.range_constexpr(IPT_EB):
                total_padded += padded_values[i]
            lane = tid & 31
            warp_id = tid >> 5
            inclusive = warp_inclusive_scan(total_padded, lane)
            if lane == 31:
                s_scan_warp_prefix[warp_id] = inclusive
            cute.arch.barrier()
            if tid == 0:
                acc = 0
                for warp_idx in cutlass.range_constexpr(NUM_WARPS):
                    warp_total = s_scan_warp_prefix[warp_idx]
                    s_scan_warp_prefix[warp_idx] = acc
                    acc += warp_total
            cute.arch.barrier()
            base = s_scan_warp_prefix[warp_id] + inclusive - total_padded
            for i in cutlass.range_constexpr(IPT_EB):
                group_idx = tid * IPT_EB + i
                if group_idx < E + B:
                    padded_end = base + padded_values[i]
                    token_count = count_values[i]
                    expert_id = expert_values[i]
                    if token_count > 0:
                        expert_offsets[dest_rank, expert_id] = base
                    all_cu_seqlens[dest_rank, group_idx] = padded_end
                    pad_start = 0
                    pad_count = 0
                    if token_count > 0:
                        pad_extra = padded_values[i] - token_count
                        if pad_extra > 0:
                            pad_start = base + token_count
                            pad_count = pad_extra
                    zero_fill_start[dest_rank, group_idx] = pad_start
                    zero_fill_count[dest_rank, group_idx] = pad_count
                    base += padded_values[i]
            cute.arch.barrier()
        grid_sync(bar_p, num_sms, tid)
        # (Deviation 3: upstream's multimem broadcast of the plan region
        # stood here; the plan is already local on every rank.)

        # ---- C1 per source (deviation 4; runtime loop keeps the body
        # compiled once instead of R-times unrolled) ----
        for r in cutlass.range(R):
            topk_r = cute.make_tensor(topk_all.iterator + r * N,
                                      cute.make_layout((N,)))
            order_r = cute.make_tensor(order_all.iterator + r * N,
                                       cute.make_layout((N,)))
            tpe_r = cute.make_tensor(tpe_all.iterator + r * E,
                                     cute.make_layout((E,)))
            self.run_c1(topk_r, order_r, tpe_r, lh, s_hist, s_bp,
                        s_scratch, bar_p, num_sms, pid, tid)

        # ---- C2 per source (deviation 5: no src_info publication) ----
        s_expoff = cute.make_tensor(s_hist.iterator,
                                    cute.make_layout((E,)))
        alloc_cumsum_view = cute.make_tensor(
            scratch.iterator + ALLOC_SUB,
            cute.make_layout((E, R), stride=(R, 1)))
        tpe_cumsum_view = cute.make_tensor(
            scratch.iterator + TPE_SUB,
            cute.make_layout((R, E), stride=(E, 1)))
        expert_off_view = cute.make_tensor(
            scratch.iterator + EOFF_SUB,
            cute.make_layout((R, E), stride=(E, 1)))
        for r in cutlass.range(R):
            order_in = cute.make_tensor(order_all.iterator + r * N,
                                        cute.make_layout((N,)))
            topk_by_off = cute.make_tensor(topk_all.iterator + r * N,
                                           cute.make_layout((N,)))
            tpe_r = cute.make_tensor(tpe_all.iterator + r * E,
                                     cute.make_layout((E,)))
            dst_out = cute.make_tensor(dst_all.iterator + r * N,
                                       cute.make_layout((N,)))
            for e in cutlass.range(tid, E, num_threads):
                s_expoff[e] = tpe_r[e]
            cute.arch.barrier()
            warp_exclusive_scan_e(s_expoff, E, tid)
            seg = cute.ceil_div(N, num_sms)
            sbeg = pid * seg
            send = cutlass.min(sbeg + seg, N)
            for base in cutlass.range(sbeg + tid, send,
                                      num_threads * ITEMS_PER_THREAD_P2):
                for i in cutlass.range_constexpr(ITEMS_PER_THREAD_P2):
                    idx = base + i * BLOCK_DIM_P2
                    if idx < send:
                        offv = order_in[idx]
                        expert_idx = topk_by_off[offv]
                        prev = 0
                        if r > 0:
                            prev = tpe_cumsum_view[r - 1, expert_idx]
                        global_rank = prev + (idx - s_expoff[expert_idx])
                        lo = 0; hi = R; pc = 0
                        for bin_step in cutlass.range_constexpr(LOG2_R):
                            mid = (lo + hi) >> 1
                            ac = alloc_cumsum_view[expert_idx, mid]
                            if ac > global_rank: hi = mid
                            else: lo = mid + 1; pc = ac
                        bo = expert_off_view[lo, expert_idx]
                        dst_out[offv] = lo * NvS + bo + (global_rank - pc)
            # next source recomputes s_expoff: order the smem reuse
            cute.arch.barrier()
        grid_sync(bar_p, num_sms, tid)

        # ---- Dedup canonicalization over ALL R*S tokens (deviation 7,
        # upstream :1079-1113 register bitset logic verbatim) ----
        RS = cutlass.const_expr(self.R * S)
        seg_dst = cute.ceil_div(RS, num_sms)
        sbeg_dst = pid * seg_dst
        send_dst = cutlass.min(sbeg_dst + seg_dst, RS)
        dst_flat = cute.make_tensor(dst_all.iterator,
                                    cute.make_layout((R * N,)))
        for base in cutlass.range(sbeg_dst + tid, send_dst, num_threads):
            base_idx = base * K
            dst_vals = []
            dests = []
            for k in cutlass.range_constexpr(K):
                v = dst_flat[base_idx + k]
                d = v // NvS
                dst_vals.append(v)
                dests.append(d)
            seen_lo = Int64(0)
            seen_hi = Int64(0)
            for k in cutlass.range_constexpr(K):
                d = dests[k]
                dup = cutlass.Boolean(False)
                if d < Int32(64):
                    shift = Int64(d)
                    bit = Int64(1) << shift
                    dup = (seen_lo & bit) != Int64(0)
                    seen_lo = seen_lo | bit
                else:
                    shift = Int64(d - Int32(64))
                    bit = Int64(1) << shift
                    dup = (seen_hi & bit) != Int64(0)
                    seen_hi = seen_hi | bit
                if dup:
                    dst_flat[base_idx + k] = -(dst_vals[k]) - 1
        # grid_sync self-resets; no cleanup needed.


# ============================================================
# Host side: workspace + compile cache + launch
# ============================================================


@functools.lru_cache(maxsize=None)
def _get_compiled(R, E, B, S, K, NvS_capacity, NvS, num_vblocks,
                  token_padding, num_sms):
    k = ReplicatedPlanningKernel(R, E, B, S, K, NvS_capacity, NvS,
                                 num_vblocks, token_padding, num_sms)
    i32 = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    return cute.compile(k, i32, i32, i32, i32, i32, i32, i32, i32, i32,
                        i32, i32, i32, i32, i32, cuda.CUstream(0))


def _round4(n):
    return (n + 3) & ~3


class ReplicatedPlannerWorkspace:
    """Ctor-allocated (legal one-shot) device workspace + output tensors
    for the replicated planner. `launch()` runs the cooperative kernel on
    the current stream — the per-iteration timed call."""

    def __init__(self, cfg, device, num_sms: int = 32):
        R, E, B = cfg.R, cfg.E, cfg.B
        N = cfg.N
        self.cfg = cfg
        self.num_sms = num_sms
        self.num_vblocks = ceil_div(N, BLOCK_SIZE_P2)
        assert N <= cfg.NvS and R * cfg.NvS <= 2**31 - 1
        assert R <= 128 and cfg.K <= 32 and BLOCK_DIM_P2 >= R
        dev = device
        i32 = dict(dtype=torch.int32, device=dev)
        self.scratch = torch.zeros(3 * E * R, **i32)
        self.alloc = torch.zeros(R * E, **i32)
        self.group_tokens = torch.zeros(R, **i32)
        self.z = torch.zeros(R * R, **i32)
        self.local_hist = torch.zeros(self.num_vblocks * E, **i32)
        self.order_all = torch.zeros(R * N, **i32)
        self.grid_bar = torch.zeros(4, **i32)
        self.dst_all = torch.empty(_round4(R * N), **i32)[:R * N].view(R, N)
        self.cu_all = torch.empty(
            _round4(R * (E + B)), **i32)[:R * (E + B)].view(R, E + B)
        self.etc_all = torch.empty(_round4(R * B), **i32)[:R * B].view(R, B)
        self.zfr_all = torch.empty(
            _round4(R * (E + B) * 2), **i32)[:R * (E + B) * 2].view(
                R, E + B, 2)
        self.stats_all = torch.empty(_round4(R * 2), **i32)[:R * 2].view(
            R, 2)
        self._compiled = _get_compiled(
            R, E, B, cfg.S, cfg.K, cfg.NvS_capacity, cfg.NvS,
            self.num_vblocks, cfg.token_padding, num_sms)

    def launch(self, topk_all_dev: torch.Tensor, tpe_all_dev: torch.Tensor):
        """Run the planner on the current stream; results land in
        dst_all/cu_all/etc_all/zfr_all/stats_all. topk_all_dev [R, N] and
        tpe_all_dev [R, E] must be contiguous int32 CUDA tensors."""
        cfg = self.cfg
        assert topk_all_dev.dtype == torch.int32
        assert tpe_all_dev.dtype == torch.int32
        assert topk_all_dev.is_contiguous() and tpe_all_dev.is_contiguous()
        assert topk_all_dev.numel() == cfg.R * cfg.N
        assert tpe_all_dev.numel() == cfg.R * cfg.E

        def p16(t):
            return make_ptr(Int32, t.data_ptr(), cute.AddressSpace.gmem,
                            assumed_align=16)

        def p4(t):
            return make_ptr(Int32, t.data_ptr(), cute.AddressSpace.gmem,
                            assumed_align=4)

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self._compiled(
            p16(tpe_all_dev), p16(topk_all_dev), p16(self.scratch),
            p16(self.dst_all), p4(self.cu_all), p4(self.etc_all),
            p4(self.zfr_all), p4(self.stats_all), p16(self.alloc),
            p16(self.group_tokens), p16(self.z), p16(self.local_hist),
            p16(self.order_all), p16(self.grid_bar), stream,
        )
        return (self.dst_all, self.cu_all, self.etc_all, self.zfr_all,
                self.stats_all)
