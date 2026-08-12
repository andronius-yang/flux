//===- weight_prefetch_getmem.h ----------------------------------- C++ ---===//
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
#include <c10/core/ScalarType.h>
#include <torch/all.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include "flux/ths_op/flux_shm.h"
namespace bytedance::flux::ths_op {

// Destination-initiated expert-weight prefetch (MoonEP pull semantics).
// The ctor collectively allocates a per-rank symmetric weight-home shard
// [n_experts_local, row_dim0, row_dim1]; the caller copies its local expert
// weights in once at setup (the shard is immutable afterwards — that
// immutability is why no signaling or barrier exists anywhere in this op).
class WeightPrefetchGetmem {
 public:
  WeightPrefetchGetmem(
      std::shared_ptr<Group> pg,
      int64_t n_experts_local,
      int64_t row_dim0,
      int64_t row_dim1,
      at::ScalarType dtype);

  ~WeightPrefetchGetmem();

  // My rank's symmetric weight-home shard (read by remote gets; also valid
  // as the local-expert GEMM operand — symmetric memory is ordinary device
  // memory to local kernels).
  torch::Tensor weight_home();

  // pairs_cpu: [n_pairs, 3] int32 CPU tensor of (dst_slot, home_pe, src_row)
  // for THIS rank's incoming pulls, derived from the replicated plan. Stored
  // once per plan (device copy for the kernel, host copy for the fallback).
  void set_pairs(torch::Tensor pairs_cpu);

  // Pull every pair's expert rows into dst ([n_slots, row_dim0, row_dim1],
  // ordinary memory) on the current torch stream. device_kernel=true issues
  // nvshmemx_getmem_nbi_block chunks from an SM kernel (authentic MoonEP
  // shape); false uses host nvshmemx_getmem_nbi_on_stream per pair (A/B
  // fallback). Both end with nvshmemx_quiet_on_stream: an event recorded
  // after forward() proves the weights landed.
  torch::Tensor forward(
      torch::Tensor dst, int32_t num_comm_sm, int64_t chunk_bytes, bool device_kernel);

 private:
  class WeightPrefetchGetmemImpl;
  WeightPrefetchGetmemImpl *impl_;
};

}  // namespace bytedance::flux::ths_op
