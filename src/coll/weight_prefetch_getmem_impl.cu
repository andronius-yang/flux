//===- weight_prefetch_getmem_impl.cu ----------------------------- C++ ---===//
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

#include "weight_prefetch_getmem_impl.hpp"
#include "flux/flux.h"
#include "flux/cuda/cuda_common.h"
namespace bytedance {
namespace flux {

// One (pair, chunk) work item per grid-stride step. Each block pulls one
// chunk of one expert's weights with a non-blocking get; nothing here or on
// the source rank signals — completion is drained by the host-side
// nvshmemx_quiet_on_stream after the launch, so an event recorded after that
// is a sufficient join. Intra-node PEs resolve to NVLink P2P copies,
// inter-node PEs go through the proxy (CXI on Slingshot).
__global__ void
__launch_bounds__(1024, 1) weight_prefetch_getmem_kernel(const WeightPrefetchGetmemParams params) {
  int32_t total = params.n_pairs * params.chunks_per_expert;
  for (int32_t work = blockIdx.x; work < total; work += gridDim.x) {
    int32_t pair = work / params.chunks_per_expert;
    int32_t chunk = work % params.chunks_per_expert;
    int32_t dst_slot = params.pairs[pair * 3 + 0];
    int32_t home_pe = params.pairs[pair * 3 + 1];
    int32_t src_row = params.pairs[pair * 3 + 2];
    int64_t offset = static_cast<int64_t>(chunk) * params.chunk_bytes;
    int64_t nbytes = params.expert_bytes - offset;
    if (nbytes <= 0) {
      continue;
    }
    if (nbytes > params.chunk_bytes) {
      nbytes = params.chunk_bytes;
    }
    char *dst = static_cast<char *>(params.dst_ptr) +
                static_cast<int64_t>(dst_slot) * params.expert_bytes + offset;
    const char *src = static_cast<const char *>(params.weight_home_ptr) +
                      static_cast<int64_t>(src_row) * params.expert_bytes + offset;
    nvshmemx_getmem_nbi_block(dst, src, nbytes, home_pe);
  }
}

void
weight_prefetch_getmem_impl(
    const WeightPrefetchGetmemParams params, int32_t num_comm_sm, cudaStream_t stream) {
  static constexpr int32_t kThreadsPerBlock = 1024;
  if (params.n_pairs > 0) {
    int32_t total = params.n_pairs * params.chunks_per_expert;
    int32_t nblocks = num_comm_sm < total ? num_comm_sm : total;
    weight_prefetch_getmem_kernel<<<nblocks, kThreadsPerBlock, 0, stream>>>(params);
  }
  nvshmemx_quiet_on_stream(stream);
}
}  // namespace flux
}  // namespace bytedance
