//===- weight_prefetch_getmem_impl.hpp ---------------------------- C++ ---===//
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

#pragma once
#include "flux/flux.h"
#include <nvshmemx.h>
#include <nvshmem.h>
namespace bytedance {
namespace flux {

// Destination-initiated pull of expert weight rows from remote ranks'
// symmetric weight-home shards (MoonEP prefetch semantics: the source rank is
// passive and never signaled — its weight home is immutable for the process
// lifetime, so a get is always valid). PE ids are global torch ranks
// (init_flux_shm asserts rank == PE). Byte-oriented: no dtype dispatch.
struct WeightPrefetchGetmemParams {
  void *weight_home_ptr;  // symm buf, [n_experts_local, expert_bytes]
  void *dst_ptr;          // normal buf, [n_slots, expert_bytes]
  const int32_t *pairs;   // [n_pairs, 3]: (dst_slot, home_pe, src_row)
  int32_t n_pairs;
  int32_t chunks_per_expert;
  int64_t expert_bytes;
  int64_t chunk_bytes;
};

void weight_prefetch_getmem_impl(
    const WeightPrefetchGetmemParams params,
    int32_t num_comm_sm,
    cudaStream_t stream);
}  // namespace flux
}  // namespace bytedance
