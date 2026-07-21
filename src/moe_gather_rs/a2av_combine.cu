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
