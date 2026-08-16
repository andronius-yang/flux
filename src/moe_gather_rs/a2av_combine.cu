//===- a2av_combine.cu -------------------------------------------- C++ ---===//
//
// Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
// Kernels for the layer1 a2av_hier combine: a persistent pack kernel that turns
// the split-major GEMM output into destination-major send-panel chunks behind
// the per-split cascade flags, and a per-split topk reduce at the destination.
// All transport between the two is host-issued (copy engines / NIC, zero SMs).

#include <cutlass/barrier.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

#include "flux/args/moe_gather_rs.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_common_device.hpp"
#include "flux/flux.h"
#include "moe_gather_rs/topk_gather_rs.hpp"

namespace bytedance::flux {
namespace {

template <typename T>
union PackU {
  static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>);
  constexpr static int kElemsPerPack = sizeof(uint4) / sizeof(T);
  uint4 data;
  T elems[kElemsPerPack];
};

CUTLASS_DEVICE void
storePack(void *ptr, uint4 data) {
  asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
               :
               : "l"(ptr), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));
}

CUTLASS_DEVICE uint4
loadPack(void const *ptr) {
  uint4 data;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
               : "l"(ptr));
  return data;
}

template <typename T>
CUTLASS_DEVICE float
elem_to_float(T v) {
  if constexpr (std::is_same_v<T, __half>) {
    return __half2float(v);
  } else {
    return __bfloat162float(v);
  }
}

template <typename T>
CUTLASS_DEVICE T
float_to_elem(float f) {
  if constexpr (std::is_same_v<T, __half>) {
    return __float2half(f);
  } else {
    return __float2bfloat16(f);
  }
}

template <typename T, bool kHasVecScale>
__global__ void
__launch_bounds__(1024, 1) a2av_combine_pack_kernel(A2AVCombinePackArguments args) {
  using Barrier = cutlass::Barrier;
  constexpr int kElemsPerPack = PackU<T>::kElemsPerPack;
  const int64_t n_per = args.n_per;
  const int64_t packs_per_row = n_per / kElemsPerPack;
  CUTLASS_PRAGMA_NO_UNROLL
  for (int sid = 0; sid < args.n_split; sid++) {
    // the GEMM's tile->problem->split cascade releases this flag once every
    // expert's rows of column window sid are complete -- the minimal gate, since
    // any destination's rows interleave across all local experts
    Barrier::wait_eq(args.barrier, threadIdx.x, sid, 1);
    for (int gi = 0; gi < args.nnodes; gi++) {
      // remote-node chunks first so the NIC-bound flags flip earliest; the host
      // ladders consume flags in this same production order (no head-of-line)
      int g = (args.node_idx + 1 + gi) % args.nnodes;
      const int64_t row_lo = args.node_row_start[g];
      const int64_t total = (args.node_row_start[g + 1] - row_lo) * packs_per_row;
      for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < total;
           idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t p = row_lo + idx / packs_per_row;
        const int64_t col = (idx % packs_per_row) * kElemsPerPack;
        const int src_row = args.pack_index[p];
        PackU<T> pk;
        pk.data = loadPack(
            (T const *)args.gemm_out + (int64_t)src_row * args.n + (int64_t)sid * n_per + col);
        if constexpr (kHasVecScale) {
          const float s = args.vec_scale[src_row];
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < kElemsPerPack; i++) {
            pk.elems[i] = float_to_elem<T>(elem_to_float<T>(pk.elems[i]) * s);
          }
        }
        storePack(
            (T *)args.send_panel + ((int64_t)sid * args.panel_rows + p) * n_per + col, pk.data);
      }
      // publish the (g, sid) chunk to the host put ladders once every block is
      // done -- including g == node_idx: the intra-node ladder gates on it
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) {
        int done = atomicAdd(args.group_counters + g * args.n_split + sid, 1) + 1;
        if (done == gridDim.x) {
          atomic_store_release_sys(args.group_flags + g * args.n_split + sid, 1);
        }
      }
    }
  }
}

CUTLASS_DEVICE uint64_t
load_acquire_sys_u64(uint64_t const *ptr) {
  uint64_t v;
  asm volatile("ld.global.acquire.sys.b64 %0, [%1];\n" : "=l"(v) : "l"(ptr));
  return v;
}

// source lane of a recv-panel row: the unique s with cum[s] <= row < cum[s+1]
// (zero-row lanes have cum[s] == cum[s+1] and can never contain a row)
CUTLASS_DEVICE int
lane_of_row(int64_t const *cum, int world_size, int64_t row) {
  int lo = 0, hi = world_size - 1;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (row < cum[mid + 1]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

// Eager reduce: split-major outer loop like the pack kernel; per element the
// remaining-mask loop consumes contributions in arrival order. Summation order
// is arrival-dependent (like the dense ring path); correctness checks are
// tolerance-based. Resident on the reserved reduce SMs for the whole epoch.
// kCSR (compress): per-token contributions come from red_ptr/red_row instead
// of the fixed-topk reduce_index; the lane search runs over the C' image.
template <typename T, bool kCSR>
__global__ void
__launch_bounds__(512, 1) a2av_combine_eager_reduce_kernel(
    A2AVCombineEagerReduceArguments args) {
  constexpr int kElemsPerPack = PackU<T>::kElemsPerPack;
  const int64_t n_per = args.n_per;
  const int64_t packs_per_row = n_per / kElemsPerPack;
  const int64_t total = args.ntokens_local * packs_per_row;
  CUTLASS_PRAGMA_NO_UNROLL
  for (int sid = 0; sid < args.n_split; sid++) {
    T const *panel = (T const *)args.recv_panel + (int64_t)sid * args.panel_rows * n_per;
    for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
      const int64_t t = idx / packs_per_row;
      const int64_t col = (idx % packs_per_row) * kElemsPerPack;
      int64_t ent_base;
      int count;
      if constexpr (kCSR) {
        ent_base = args.red_ptr[t];
        count = args.red_ptr[t + 1] - (int32_t)ent_base;
      } else {
        ent_base = t * args.topk;
        count = args.topk;
      }
      float acc[kElemsPerPack];
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kElemsPerPack; i++) {
        acc[i] = 0.0f;
      }
      uint32_t remaining = (count >= 32) ? ~0u : ((1u << count) - 1u);
      while (remaining != 0) {
        bool made_progress = false;
        for (int j = 0; j < count; j++) {
          if ((remaining & (1u << j)) == 0) {
            continue;
          }
          const int64_t row =
              kCSR ? args.red_row[ent_base + j] : args.reduce_index[ent_base + j];
          const int lane = lane_of_row(args.recv_cum, args.world_size, row);
          if (load_acquire_sys_u64(
                  args.recv_signals + (int64_t)lane * args.n_split + sid) >= args.run_id) {
            PackU<T> pk;
            pk.data = loadPack(panel + row * n_per + col);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kElemsPerPack; i++) {
              acc[i] += elem_to_float<T>(pk.elems[i]);
            }
            remaining &= ~(1u << j);
            made_progress = true;
          }
        }
        if (!made_progress) {
          __nanosleep(200);
        }
      }
      PackU<T> out;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kElemsPerPack; i++) {
        out.elems[i] = float_to_elem<T>(acc[i]);
      }
      storePack((T *)args.output + t * args.n + (int64_t)sid * n_per + col, out.data);
    }
  }
}

// Source-side gateway pre-reduce for compress: per (split, target node in
// inter-ladder rotation order) thread 0 of each block spins on the L per-peer
// convergence signals, then the block strides the segment's wire rows, merging
// each row's CSR conv contributions in fp32; group completion flips the
// (tn, sid) wire flag via the pack kernel's counter handshake.
template <typename T>
__global__ void
__launch_bounds__(512, 1) a2av_combine_prereduce_kernel(A2AVCombinePreReduceArguments args) {
  constexpr int kElemsPerPack = PackU<T>::kElemsPerPack;
  const int64_t n_per = args.n_per;
  const int64_t packs_per_row = n_per / kElemsPerPack;
  const int NN = args.nnodes;
  const int L = args.local_world_size;
  CUTLASS_PRAGMA_NO_UNROLL
  for (int sid = 0; sid < args.n_split; sid++) {
    T const *conv = (T const *)args.conv_panel + (int64_t)sid * args.conv_rows * n_per;
    T *wire = (T *)args.wire_panel + (int64_t)sid * args.wire_rows * n_per;
    for (int gi = 0; gi < NN - 1; gi++) {
      const int tn = (args.node_idx + 1 + gi) % NN;
      const int seg = tn < args.node_idx ? tn : tn - 1;
      if (threadIdx.x == 0) {
        for (int ls = 0; ls < L; ls++) {
          uint64_t const *sig =
              args.conv_signals + ((int64_t)ls * NN + tn) * args.n_split + sid;
          while (load_acquire_sys_u64(sig) < args.run_id) {
            __nanosleep(200);
          }
        }
      }
      __syncthreads();
      const int64_t w_lo = args.wire_seg_start[seg];
      const int64_t total = (args.wire_seg_start[seg + 1] - w_lo) * packs_per_row;
      for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < total;
           idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t w = w_lo + idx / packs_per_row;
        const int64_t col = (idx % packs_per_row) * kElemsPerPack;
        float acc[kElemsPerPack];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElemsPerPack; i++) {
          acc[i] = 0.0f;
        }
        for (int32_t k = args.wire_ptr[w]; k < args.wire_ptr[w + 1]; k++) {
          PackU<T> pk;
          pk.data = loadPack(conv + (int64_t)args.wire_copy[k] * n_per + col);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < kElemsPerPack; i++) {
            acc[i] += elem_to_float<T>(pk.elems[i]);
          }
        }
        PackU<T> out;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElemsPerPack; i++) {
          out.elems[i] = float_to_elem<T>(acc[i]);
        }
        storePack(wire + w * n_per + col, out.data);
      }
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) {
        int done = atomicAdd(args.wire_counters + tn * args.n_split + sid, 1) + 1;
        if (done == gridDim.x) {
          atomic_store_release_sys(args.wire_flags + tn * args.n_split + sid, 1);
        }
      }
    }
  }
}

// Legacy-gate destination reduce under compress: CSR contributions per token
// (own-node copies + one merged row per contributing remote node)
template <typename T>
__global__ void
a2av_combine_csr_reduce_kernel(A2AVCombineCSRReduceArguments args) {
  constexpr int kElemsPerPack = PackU<T>::kElemsPerPack;
  const int64_t n_per = args.n_per;
  const int64_t packs_per_row = n_per / kElemsPerPack;
  const int64_t total = args.ntokens_local * packs_per_row;
  T const *panel = (T const *)args.recv_panel + (int64_t)args.sid * args.panel_rows * n_per;
  for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < total;
       idx += (int64_t)gridDim.x * blockDim.x) {
    const int64_t t = idx / packs_per_row;
    const int64_t col = (idx % packs_per_row) * kElemsPerPack;
    float acc[kElemsPerPack];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElemsPerPack; i++) {
      acc[i] = 0.0f;
    }
    for (int32_t k = args.red_ptr[t]; k < args.red_ptr[t + 1]; k++) {
      PackU<T> pk;
      pk.data = loadPack(panel + (int64_t)args.red_row[k] * n_per + col);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kElemsPerPack; i++) {
        acc[i] += elem_to_float<T>(pk.elems[i]);
      }
    }
    PackU<T> out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElemsPerPack; i++) {
      out.elems[i] = float_to_elem<T>(acc[i]);
    }
    storePack((T *)args.output + t * args.n + (int64_t)args.sid * n_per + col, out.data);
  }
}

template <typename T>
__global__ void
a2av_combine_reduce_kernel(A2AVCombineReduceArguments args) {
  constexpr int kElemsPerPack = PackU<T>::kElemsPerPack;
  const int64_t n_per = args.n_per;
  const int64_t packs_per_row = n_per / kElemsPerPack;
  const int64_t total = args.ntokens_local * packs_per_row;
  T const *panel = (T const *)args.recv_panel + (int64_t)args.sid * args.panel_rows * n_per;
  for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < total;
       idx += (int64_t)gridDim.x * blockDim.x) {
    const int64_t t = idx / packs_per_row;
    const int64_t col = (idx % packs_per_row) * kElemsPerPack;
    float acc[kElemsPerPack];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElemsPerPack; i++) {
      acc[i] = 0.0f;
    }
    for (int j = 0; j < args.topk; j++) {
      const int64_t row = args.reduce_index[t * args.topk + j];
      PackU<T> pk;
      pk.data = loadPack(panel + row * n_per + col);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kElemsPerPack; i++) {
        acc[i] += elem_to_float<T>(pk.elems[i]);
      }
    }
    PackU<T> out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElemsPerPack; i++) {
      out.elems[i] = float_to_elem<T>(acc[i]);
    }
    storePack((T *)args.output + t * args.n + (int64_t)args.sid * n_per + col, out.data);
  }
}

}  // namespace

// Force-load every combine kernel at construction time. Under
// CUDA_MODULE_LOADING=LAZY (the launch.sh default) a kernel's module is loaded
// at its FIRST launch; the eager / compress schedules put a persistent spin
// kernel (eager reduce / pre-reduce) on the device BEFORE the epoch's first
// launch of the remaining kernels, and a first-launch load that must complete
// behind a never-exiting resident kernel deadlocks the epoch (root cause of
// the 2026-08-16 two-node eager/compress hangs). Attribute queries are the
// documented cudart way to preload a kernel under lazy loading, and the ctor
// runs with an idle device, so every load here is trivial.
void
a2av_combine_preload(DataTypeEnum dtype) {
  cudaFuncAttributes attr;
  tuple_return_if(
      cute::make_tuple(_FP16{}, _BF16{}),
      [&](auto cdtype) { return cdtype == dtype; },
      [&](auto cdtype) {
        using T = decltype(to_cuda_dtype(cdtype));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, a2av_combine_pack_kernel<T, true>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, a2av_combine_pack_kernel<T, false>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, a2av_combine_eager_reduce_kernel<T, true>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, a2av_combine_eager_reduce_kernel<T, false>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, a2av_combine_prereduce_kernel<T>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, a2av_combine_csr_reduce_kernel<T>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, a2av_combine_reduce_kernel<T>));
      },
      [&]() { FLUX_CHECK(false) << "unsupported dtype for a2av combine preload: " << dtype; });
}

void
a2av_combine_pack(
    A2AVCombinePackArguments const &args, DataTypeEnum dtype, cudaStream_t stream) {
  constexpr int kThreads = 1024;
  FLUX_CHECK_LE(args.nnodes, kA2AVMaxNodes);
  FLUX_CHECK(args.n_per % 8 == 0) << "n/n_split must be a multiple of the 8-elem pack width";
  dim3 grid(args.threadblock_count), block(kThreads);
  const bool has_vec_scale = args.vec_scale != nullptr;
  tuple_return_if(
      tuple_cartesian_product(
          cute::make_tuple(_FP16{}, _BF16{}),
          cute::make_tuple(cute::true_type{}, cute::false_type{})),
      [&](auto tup) {
        auto [cdtype, has_vec_scale_] = tup;
        return cdtype == dtype && has_vec_scale_ == has_vec_scale;
      },
      [&](auto tup) {
        auto [cdtype, has_vec_scale_] = tup;
        using T = decltype(to_cuda_dtype(cdtype));
        constexpr bool kHasVecScale = decltype(has_vec_scale_){};
        a2av_combine_pack_kernel<T, kHasVecScale><<<grid, block, 0, stream>>>(args);
      },
      [&]() { FLUX_CHECK(false) << "unsupported dtype for a2av combine pack: " << dtype; });
  CUDA_CHECK(cudaGetLastError());
}

void
a2av_combine_eager_reduce(
    A2AVCombineEagerReduceArguments const &args, DataTypeEnum dtype, cudaStream_t stream) {
  constexpr int kThreads = 512;
  FLUX_CHECK(args.n_per % 8 == 0) << "n/n_split must be a multiple of the 8-elem pack width";
  FLUX_CHECK_LE(args.world_size, kA2AVMaxWorld);
  FLUX_CHECK_LE(args.topk, 31) << "eager reduce remaining-mask holds topk in 31 bits";
  const bool csr = args.red_ptr != nullptr;
  FLUX_CHECK(csr == (args.red_row != nullptr));
  dim3 grid(args.threadblock_count), block(kThreads);
  tuple_return_if(
      tuple_cartesian_product(
          cute::make_tuple(_FP16{}, _BF16{}),
          cute::make_tuple(cute::true_type{}, cute::false_type{})),
      [&](auto tup) {
        auto [cdtype, csr_] = tup;
        return cdtype == dtype && csr_ == csr;
      },
      [&](auto tup) {
        auto [cdtype, csr_] = tup;
        using T = decltype(to_cuda_dtype(cdtype));
        constexpr bool kCSR = decltype(csr_){};
        a2av_combine_eager_reduce_kernel<T, kCSR><<<grid, block, 0, stream>>>(args);
      },
      [&]() { FLUX_CHECK(false) << "unsupported dtype for a2av eager reduce: " << dtype; });
  CUDA_CHECK(cudaGetLastError());
}

void
a2av_combine_prereduce(
    A2AVCombinePreReduceArguments const &args, DataTypeEnum dtype, cudaStream_t stream) {
  constexpr int kThreads = 512;
  FLUX_CHECK(args.n_per % 8 == 0) << "n/n_split must be a multiple of the 8-elem pack width";
  FLUX_CHECK_LE(args.nnodes, kA2AVMaxNodes);
  FLUX_CHECK_GT(args.nnodes, 1) << "compress pre-reduce is a multi-node stage";
  dim3 grid(args.threadblock_count), block(kThreads);
  tuple_return_if(
      cute::make_tuple(_FP16{}, _BF16{}),
      [&](auto cdtype) { return cdtype == dtype; },
      [&](auto cdtype) {
        using T = decltype(to_cuda_dtype(cdtype));
        a2av_combine_prereduce_kernel<T><<<grid, block, 0, stream>>>(args);
      },
      [&]() { FLUX_CHECK(false) << "unsupported dtype for a2av pre-reduce: " << dtype; });
  CUDA_CHECK(cudaGetLastError());
}

void
a2av_combine_csr_reduce(
    A2AVCombineCSRReduceArguments const &args, DataTypeEnum dtype, cudaStream_t stream) {
  constexpr int kThreads = 512;
  FLUX_CHECK(args.n_per % 8 == 0) << "n/n_split must be a multiple of the 8-elem pack width";
  dim3 grid(args.threadblock_count), block(kThreads);
  tuple_return_if(
      cute::make_tuple(_FP16{}, _BF16{}),
      [&](auto cdtype) { return cdtype == dtype; },
      [&](auto cdtype) {
        using T = decltype(to_cuda_dtype(cdtype));
        a2av_combine_csr_reduce_kernel<T><<<grid, block, 0, stream>>>(args);
      },
      [&]() { FLUX_CHECK(false) << "unsupported dtype for a2av csr reduce: " << dtype; });
  CUDA_CHECK(cudaGetLastError());
}

void
a2av_combine_reduce(
    A2AVCombineReduceArguments const &args, DataTypeEnum dtype, cudaStream_t stream) {
  constexpr int kThreads = 512;
  FLUX_CHECK(args.n_per % 8 == 0) << "n/n_split must be a multiple of the 8-elem pack width";
  dim3 grid(args.threadblock_count), block(kThreads);
  tuple_return_if(
      cute::make_tuple(_FP16{}, _BF16{}),
      [&](auto cdtype) { return cdtype == dtype; },
      [&](auto cdtype) {
        using T = decltype(to_cuda_dtype(cdtype));
        a2av_combine_reduce_kernel<T><<<grid, block, 0, stream>>>(args);
      },
      [&]() { FLUX_CHECK(false) << "unsupported dtype for a2av combine reduce: " << dtype; });
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace bytedance::flux
