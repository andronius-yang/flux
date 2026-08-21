//===- fused_ep_dispatch_impl.cu ---------------------------------- C++ ---===//
//
// Copyright 2026 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
#include <nvshmem.h>

#include "fused_ep_dispatch_impl.hpp"
#include "flux/flux.h"
#include "flux/cuda/cuda_common.h"

namespace bytedance {
namespace flux {

namespace {

constexpr int32_t kThreads = 256;
constexpr int32_t kHistTile = 2048;  // flat cells per pass-1 block

__device__ __forceinline__ uint64_t
ld_acquire_sys_u64(const uint64_t *ptr) {
  uint64_t v;
  asm volatile("ld.global.acquire.sys.b64 %0, [%1];\n" : "=l"(v) : "l"(ptr));
  return v;
}

// 16B-vectorized row copy, one warp per row segment.
__device__ __forceinline__ void
copy_row_warp(void *dst, const void *src, int64_t nbytes, int lane) {
  const uint4 *s = reinterpret_cast<const uint4 *>(src);
  uint4 *d = reinterpret_cast<uint4 *>(dst);
  int64_t n16 = nbytes / 16;
  for (int64_t i = lane; i < n16; i += 32) {
    d[i] = s[i];
  }
}

// ---------------------------------------------------------------------------
// Pack: deterministic counting sort of the S*K flat cells by dst_phys.
// pass 1: per-block histograms (+ global my_counts); scan: single block
// derives pack_base (exclusive over P) and block_offset; pass 2: stable
// emission in flat-cell order + row copy + header write.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(kThreads, 1)
fused_ep_pack_pass1_kernel(FusedEpDispatchParams p) {
  extern __shared__ int32_t s_hist[];
  const int32_t P = p.world_size * p.nlp;
  const int64_t N = (int64_t)p.S * p.K;
  for (int32_t e = threadIdx.x; e < P; e += blockDim.x) {
    s_hist[e] = 0;
  }
  __syncthreads();
  const int64_t lo = (int64_t)blockIdx.x * kHistTile;
  const int64_t hi = (lo + (int64_t)kHistTile < N) ? lo + (int64_t)kHistTile : N;
  for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    atomicAdd(&s_hist[p.dst_phys[i]], 1);
  }
  __syncthreads();
  for (int32_t e = threadIdx.x; e < P; e += blockDim.x) {
    int32_t c = s_hist[e];
    p.block_hist[(int64_t)blockIdx.x * P + e] = c;
    if (c) {
      atomicAdd(&p.my_counts[e], c);
    }
  }
}

__global__ void __launch_bounds__(kThreads, 1)
fused_ep_pack_scan_kernel(FusedEpDispatchParams p) {
  // single block: pack_base = exclusive scan of my_counts over P;
  // block_offset[b][e] = exclusive scan of block_hist over blocks per e.
  const int32_t P = p.world_size * p.nlp;
  for (int32_t e = threadIdx.x; e < P; e += blockDim.x) {
    int32_t run = 0;
    for (int32_t b = 0; b < p.num_hist_blocks; ++b) {
      int32_t c = p.block_hist[(int64_t)b * P + e];
      p.block_offset[(int64_t)b * P + e] = run;
      run += c;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int32_t run = 0;
    for (int32_t e = 0; e < P; ++e) {
      p.pack_base[e] = run;
      run += p.my_counts[e];
    }
    p.pack_base[P] = run;
  }
}

__global__ void __launch_bounds__(kThreads, 1)
fused_ep_pack_pass2_kernel(FusedEpDispatchParams p) {
  // Stable emission: block b re-walks its tile in flat order; a smem
  // running cursor per slot (thread-order walk => deterministic ranks).
  extern __shared__ int32_t s_cursor[];
  const int32_t P = p.world_size * p.nlp;
  const int64_t N = (int64_t)p.S * p.K;
  const int32_t b = blockIdx.x;
  for (int32_t e = threadIdx.x; e < P; e += blockDim.x) {
    s_cursor[e] = p.block_offset[(int64_t)b * P + e];
  }
  __syncthreads();
  const int64_t lo = (int64_t)b * kHistTile;
  const int64_t hi = (lo + (int64_t)kHistTile < N) ? lo + (int64_t)kHistTile : N;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int nwarp = blockDim.x / 32;
  // Stable rank assignment: thread 0 walks the tile strictly in flat-cell
  // order (deterministic by construction; the tile is small and the pack
  // cost is honest timed work — optimization is a named follow-up).
  __shared__ int32_t s_pos[kHistTile];
  if (threadIdx.x == 0) {
    for (int64_t i = lo; i < hi; ++i) {
      int32_t e = p.dst_phys[i];
      s_pos[i - lo] = p.pack_base[e] + s_cursor[e]++;
    }
  }
  __syncthreads();
  // row copies + headers: one warp per cell
  for (int64_t i = lo + warp; i < hi; i += nwarp) {
    int32_t pos = s_pos[i - lo];
    int64_t token = i / p.K;
    copy_row_warp(
        static_cast<char *>(p.pack_data) + (int64_t)pos * p.row_bytes,
        static_cast<const char *>(p.inputs_shard) + token * p.row_bytes,
        p.row_bytes, lane);
    if (lane == 0) {
      int32_t *h = p.pack_hdr + (int64_t)pos * 4;
      h[0] = (int32_t)i;                            // flat_cell = token*K+k
      h[1] = __float_as_int(p.probs[i]);            // route prob bits
      h[2] = p.rank;                                // src rank
      h[3] = 0;
    }
  }
}

// ---------------------------------------------------------------------------
// Counts push: my [P] counts row -> every peer's counts_sym[my_rank] +
// arrival signal (in-launch exchange; part of dispatch, lands in comm_ms).
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(kThreads, 1)
fused_ep_counts_push_kernel(FusedEpDispatchParams p) {
  const int32_t P = p.world_size * p.nlp;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  const int nwarp = (gridDim.x * blockDim.x) / 32;
  (void)lane;
  for (int32_t peer = warp; peer < p.world_size; peer += nwarp) {
    // fused put+signal: a device nvshmem_fence costs a full proxy quiet on
    // the CXI transport (~1 ms) — per-segment fences made the wire cost a
    // fixed ~180 ms at 16 ranks (2026-08-20 lo capsule). putmem_signal
    // orders its own payload before its own signal with no explicit fence
    // (probe form 2 validated on all transports).
    nvshmemx_putmem_signal_nbi_warp(
        p.counts_sym + (int64_t)p.rank * P, p.my_counts,
        (size_t)P * sizeof(int32_t), p.sig_counts + p.rank, 1,
        NVSHMEM_SIGNAL_ADD, peer);
  }
}

// ---------------------------------------------------------------------------
// Send: wait ALL peers' counts, single-block scan of the [R, P] matrix
// (remote bases + my recv layout + the collective overflow trap), then
// per-(dest, dest-slot) segment puts (data + header) + arrival signal.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(kThreads, 1)
fused_ep_scan_kernel(FusedEpDispatchParams p) {
  // block 0 waits counts arrivals, then derives every offset table.
  const int32_t R = p.world_size;
  const int32_t P = R * p.nlp;
  for (int32_t s = threadIdx.x; s < R; s += blockDim.x) {
    uint64_t spins = 0;
    while (ld_acquire_sys_u64(p.sig_counts + s) < p.run_id) {
      __nanosleep(200);
      if (++spins >= (uint64_t)250000000) {
        printf("[fused-ep] rank %d: counts from %d never arrived\n",
               p.rank, s);
        __trap();
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    // collective overflow trap: every rank evaluates the SAME expressions
    // over the SAME replicated counts matrix, so a violation traps on all
    // ranks together (the host FLUX_CHECK collective-failure property).
    int32_t max_pair = 0;
    for (int64_t i = 0; i < (int64_t)R * P; ++i) {
      max_pair = max(max_pair, p.counts_sym[i]);
    }
    if (max_pair > p.max_rows_per_pair) {
      printf("[fused-ep] rank %d: pair rows %d exceed bound %d\n",
             p.rank, max_pair, p.max_rows_per_pair);
      __trap();
    }
    // my recv layout: slots [rank*nlp, rank*nlp+nlp)
    int64_t run = 0;
    for (int32_t pl = 0; pl < p.nlp; ++pl) {
      int32_t col = p.rank * p.nlp + pl;
      p.seg_meta[p.nlp + pl] = (int32_t)run;   // seg_start
      int32_t rows = 0;
      for (int32_t s = 0; s < R; ++s) {
        p.recv_off[(int64_t)pl * R + s] = (int32_t)run + rows;
        rows += p.counts_sym[(int64_t)s * P + col];
      }
      p.seg_meta[pl] = rows;                   // seg_rows
      run += rows;
    }
    if (run > p.max_recv_total) {
      printf("[fused-ep] rank %d: recv rows %lld exceed bound %lld\n",
             p.rank, (long long)run, (long long)p.max_recv_total);
      __trap();
    }
    // my remote bases: for each dest slot e = d*nlp+pl, rows before mine
    // at d = (sum of d's earlier slots' columns) + earlier sources of e.
    for (int32_t d = 0; d < R; ++d) {
      int64_t drun = 0;
      for (int32_t pl = 0; pl < p.nlp; ++pl) {
        int32_t col = d * p.nlp + pl;
        int64_t before_me = 0;
        for (int32_t s = 0; s < R; ++s) {
          if (s < p.rank) {
            before_me += p.counts_sym[(int64_t)s * P + col];
          }
        }
        p.remote_base[col] = drun + before_me;
        for (int32_t s = 0; s < R; ++s) {
          drun += p.counts_sym[(int64_t)s * P + col];
        }
      }
    }
  }
}

__global__ void __launch_bounds__(kThreads, 1)
fused_ep_send_kernel(FusedEpDispatchParams p) {
  const int32_t P = p.world_size * p.nlp;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  const int nwarp = (gridDim.x * blockDim.x) / 32;
  // one warp per dest slot segment; ALWAYS signal, even zero-row segments.
  // No per-segment nvshmem_fence (a device fence = a full proxy quiet on
  // CXI, ~1 ms each — the 2026-08-20 fixed-180ms wire bug): each of the
  // two puts carries its own fused signal (+1), so the recv gate waits the
  // signal to reach 2 * run_id — both payloads delivered when it does.
  for (int32_t e = warp; e < P; e += nwarp) {
    const int32_t d = e / p.nlp;
    const int32_t pl = e % p.nlp;
    const int32_t rows = p.my_counts[e];
    uint64_t *sig = p.sig_data + (int64_t)pl * p.world_size + p.rank;
    if (rows > 0) {
      const int64_t src_row = p.pack_base[e];
      const int64_t dst_row = p.remote_base[e];
      nvshmemx_putmem_signal_nbi_warp(
          static_cast<char *>(p.recv_data_sym) + dst_row * p.row_bytes,
          static_cast<char *>(p.pack_data) + src_row * p.row_bytes,
          (size_t)rows * p.row_bytes, sig, 1, NVSHMEM_SIGNAL_ADD, d);
      nvshmemx_putmem_signal_nbi_warp(
          p.recv_hdr_sym + dst_row * 4, p.pack_hdr + src_row * 4,
          (size_t)rows * 4 * sizeof(int32_t), sig, 1, NVSHMEM_SIGNAL_ADD, d);
    } else if (lane == 0) {
      nvshmemx_signal_op(sig, 2, NVSHMEM_SIGNAL_ADD, d);
    }
  }
}

__global__ void __launch_bounds__(kThreads, 1)
fused_ep_recv_gate_kernel(FusedEpDispatchParams p, int32_t slot_begin,
                          int32_t slot_end) {
  const int32_t R = p.world_size;
  const int64_t total = (int64_t)(slot_end - slot_begin) * R;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t sig = (int64_t)slot_begin * R + i;
    uint64_t spins = 0;
    // 2 signal increments per (slot, src) per epoch — one per fused
    // put+signal (data, header)
    while (ld_acquire_sys_u64(p.sig_data + sig) < 2 * p.run_id) {
      __nanosleep(200);
      if (++spins >= (uint64_t)250000000) {
        printf("[fused-ep] rank %d: data signal %lld never arrived\n",
               p.rank, (long long)sig);
        __trap();
      }
    }
  }
}

__global__ void __launch_bounds__(kThreads, 1)
fused_ep_weights_extract_kernel(FusedEpDispatchParams p, float *weights_out) {
  // seg_meta[nlp-1 start] + rows gives n_recv; recv order headers -> probs
  const int64_t n_recv =
      (int64_t)p.seg_meta[p.nlp + p.nlp - 1] + p.seg_meta[p.nlp - 1];
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_recv;
       i += (int64_t)gridDim.x * blockDim.x) {
    weights_out[i] = __int_as_float(p.recv_hdr_sym[i * 4 + 1]);
  }
}

// ---------------------------------------------------------------------------
// Combine
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(kThreads, 1)
fused_ep_combine_put_kernel(FusedEpCombineParams p) {
  // one warp per (slot, src) recv segment: put each row into the SOURCE's
  // home staging at its flat cell (deterministic; each cell exactly once).
  const int32_t R = p.world_size;
  const int32_t P = R * p.nlp;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  const int nwarp = (gridDim.x * blockDim.x) / 32;
  for (int32_t seg = warp; seg < p.nlp * R; seg += nwarp) {
    const int32_t pl = seg / R;
    const int32_t s = seg % R;
    const int32_t col = p.rank * p.nlp + pl;
    const int32_t rows = p.counts_sym[(int64_t)s * P + col];
    const int32_t base = p.recv_off[(int64_t)pl * R + s];
    for (int32_t r = 0; r < rows; ++r) {
      const int64_t i = (int64_t)base + r;
      const int32_t flat_cell = p.recv_hdr_sym[i * 4 + 0];
      nvshmemx_putmem_nbi_warp(
          static_cast<char *>(p.comb_data_sym) +
              (int64_t)flat_cell * p.row_bytes,
          static_cast<const char *>(p.expert_rows) + i * p.row_bytes,
          (size_t)p.row_bytes, s);
    }
  }
}

__global__ void __launch_bounds__(kThreads, 1)
fused_ep_combine_signal_kernel(FusedEpCombineParams p) {
  // after quiet: ALWAYS-signal one ADD per (slot, src) pair -> home s
  // waits sig_comb[me] >= run_id * nlp. Launched post-quiet so delivery
  // of every put above is complete before any signal.
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  const int nwarp = (gridDim.x * blockDim.x) / 32;
  for (int32_t seg = warp; seg < p.nlp * p.world_size; seg += nwarp) {
    const int32_t s = seg % p.world_size;
    if (lane == 0) {
      nvshmemx_signal_op(p.sig_comb + p.rank, 1, NVSHMEM_SIGNAL_ADD, s);
    }
  }
}

__global__ void __launch_bounds__(kThreads, 1)
fused_ep_combine_gate_kernel(const uint64_t *sig_comb, int32_t R,
                             int32_t nlp, uint64_t run_id,
                             uint64_t spin_limit) {
  for (int32_t s = blockIdx.x * blockDim.x + threadIdx.x; s < R;
       s += gridDim.x * blockDim.x) {
    uint64_t spins = 0;
    while (ld_acquire_sys_u64(sig_comb + s) < run_id * (uint64_t)nlp) {
      __nanosleep(200);
      if (spin_limit != 0 && ++spins >= spin_limit) {
        printf("[fused-ep] combine gate: src %d stuck\n", s);
        __trap();
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Ordering probe (S4 gate)
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(kThreads, 1)
fused_ep_probe_writer_kernel(int32_t peer, int32_t *payload_sym,
                             int32_t words, uint64_t *sig_sym,
                             uint64_t epoch, int32_t form) {
  const int lane = threadIdx.x % 32;
  if (blockIdx.x == 0 && threadIdx.x < 32) {
    // local staging half sits after the probe half
    int32_t *stage = payload_sym + words;
    for (int32_t i = lane; i < words; i += 32) {
      stage[i] = (int32_t)epoch + i;
    }
    __syncwarp();
    if (form == 0) {
      nvshmemx_putmem_nbi_warp(payload_sym, stage,
                               (size_t)words * sizeof(int32_t), peer);
      __threadfence();
      nvshmem_fence();
      if (lane == 0) {
        nvshmemx_signal_op(sig_sym, 1, NVSHMEM_SIGNAL_ADD, peer);
      }
    } else {
      nvshmemx_putmem_signal_nbi_warp(
          payload_sym, stage, (size_t)words * sizeof(int32_t), sig_sym, 1,
          NVSHMEM_SIGNAL_ADD, peer);
    }
  }
}

__global__ void __launch_bounds__(kThreads, 1)
fused_ep_probe_reader_kernel(int32_t *payload_sym, int32_t words,
                             uint64_t *sig_sym, uint64_t epoch,
                             int32_t *err_count) {
  // EVERY block gates (__syncthreads is block-local — a single gating
  // block would let the others verify stale data, which is exactly the
  // false positive this probe's first version produced)
  if (threadIdx.x == 0) {
    uint64_t spins = 0;
    while (ld_acquire_sys_u64(sig_sym) < epoch) {
      __nanosleep(100);
      if (++spins >= (uint64_t)250000000) {
        printf("[fused-ep-probe] signal never arrived (epoch %llu)\n",
               (unsigned long long)epoch);
        __trap();
      }
    }
  }
  __syncthreads();
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
       i += gridDim.x * blockDim.x) {
    if (payload_sym[i] != (int32_t)epoch + i) {
      atomicAdd(err_count, 1);
    }
  }
}

}  // namespace

void
fused_ep_dispatch_impl(
    const FusedEpDispatchParams &params, float *weights_out,
    int32_t num_comm_sm, int32_t group_begin, int32_t group_end,
    cudaStream_t stream) {
  const int32_t P = params.world_size * params.nlp;
  const size_t hist_smem = (size_t)P * sizeof(int32_t);
  CUDA_CHECK(cudaMemsetAsync(params.my_counts, 0, P * sizeof(int32_t), stream));
  fused_ep_pack_pass1_kernel<<<params.num_hist_blocks, kThreads, hist_smem,
                               stream>>>(params);
  fused_ep_pack_scan_kernel<<<1, kThreads, 0, stream>>>(params);
  fused_ep_pack_pass2_kernel<<<params.num_hist_blocks, kThreads, hist_smem,
                               stream>>>(params);
  fused_ep_counts_push_kernel<<<1, kThreads, 0, stream>>>(params);
  fused_ep_scan_kernel<<<1, kThreads, 0, stream>>>(params);
  fused_ep_send_kernel<<<num_comm_sm, kThreads, 0, stream>>>(params);
  fused_ep_recv_gate_kernel<<<num_comm_sm, kThreads, 0, stream>>>(
      params, group_begin, group_end);
  fused_ep_weights_extract_kernel<<<num_comm_sm, kThreads, 0, stream>>>(
      params, weights_out);
}

void
fused_ep_recv_gate_only(
    const FusedEpDispatchParams &params, int32_t slot_begin,
    int32_t slot_end, int32_t num_comm_sm, cudaStream_t stream) {
  fused_ep_recv_gate_kernel<<<num_comm_sm, kThreads, 0, stream>>>(
      params, slot_begin, slot_end);
}

void
fused_ep_combine_impl(
    const FusedEpCombineParams &params, int32_t num_comm_sm,
    cudaStream_t stream) {
  fused_ep_combine_put_kernel<<<num_comm_sm, kThreads, 0, stream>>>(params);
  // drain every nbi put before ANY signal (stream-ordered quiet: the
  // cross-kernel analog of the in-warp fence-before-signal sequence)
  nvshmemx_quiet_on_stream(stream);
  fused_ep_combine_signal_kernel<<<1, kThreads, 0, stream>>>(params);
}

void
fused_ep_combine_gate_impl(
    const uint64_t *sig_comb, int32_t world_size, int32_t nlp,
    uint64_t run_id, uint64_t spin_limit, cudaStream_t stream) {
  fused_ep_combine_gate_kernel<<<1, kThreads, 0, stream>>>(
      sig_comb, world_size, nlp, run_id, spin_limit);
}

void
fused_ep_probe_impl(
    int32_t peer, int32_t *payload_sym, int32_t payload_words,
    uint64_t *sig_sym, uint64_t epoch, int32_t form, int32_t is_writer,
    int32_t *err_count, cudaStream_t stream) {
  if (is_writer) {
    fused_ep_probe_writer_kernel<<<1, kThreads, 0, stream>>>(
        peer, payload_sym, payload_words, sig_sym, epoch, form);
  } else {
    fused_ep_probe_reader_kernel<<<4, kThreads, 0, stream>>>(
        payload_sym, payload_words, sig_sym, epoch, err_count);
  }
}

void
fused_ep_dispatch_preload() {
  cudaFuncAttributes attr;
  CUDA_CHECK(cudaFuncGetAttributes(&attr, fused_ep_pack_pass1_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, fused_ep_pack_scan_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, fused_ep_pack_pass2_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, fused_ep_counts_push_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, fused_ep_scan_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, fused_ep_send_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, fused_ep_recv_gate_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, fused_ep_weights_extract_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, fused_ep_combine_put_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, fused_ep_combine_signal_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, fused_ep_combine_gate_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, fused_ep_probe_writer_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, fused_ep_probe_reader_kernel));
}

}  // namespace flux
}  // namespace bytedance
