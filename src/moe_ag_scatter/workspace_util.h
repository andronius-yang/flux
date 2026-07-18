//===- workspace_util.h ---------------------------------------- C++ ------===//
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
#pragma once
#include "flux/args/moe_ag_scatter.h"
namespace bytedance::flux {

struct ProblemInfo {
  int32_t problem_idx;       // problem index
  int32_t problem_tile_idx;  // tile index inner problem. tile_idx_m = problem_tile_idx / tiled_n.
                             // tile_idx_n = problem_tile_idx % tiled_n
};

// save some memory
struct ProblemSchedV2 {
  int16_t problem_idx;
  int16_t tile_m_start;
  int16_t tile_m_size;
};

/* all ptrs pointer to device memory.
workspace structure:
  problem_sizes, cutlass::gemm::GemmCoord, problem_count
  ptr_A, void *, problem_count
  ptr_B, void *, problem_count
  ptr_C, void *, problem_count
  ptr_D, void *, problem_count
  scale_D, float *, problem_count
  lda, int64_t, problem_count
  ldb, int64_t, problem_count
  ldc, int64_t, problem_count
  ldd, int64_t, problem_count
  ldr, int64_t, problem_count
  gather_A, int *, problem_count
  scatter_D, int *, problem_count
  tile_count, int, 1
  problem_info, ProblemInfo, pad_to(num_tiles, threadblock_count)
*/
struct MoeAgScatterWorkspaceArgumements {
  void *problem_sizes;
  void **ptr_A;
  void **ptr_B;
  void **ptr_C;
  void **ptr_D;
  float **scale_D;
  int64_t *lda;
  int64_t *ldb;
  int64_t *ldc;
  int64_t *ldd;
  int64_t *ldr;
  int32_t **gather_A;
  int32_t **scatter_D;
  int *tile_count;  // keep it in device to avoid device sync
  ProblemInfo *problem_info;
};

// a2av dispatch mode: extra workspace regions appended after the dense layout,
// holding the dynamically-claimed tile schedule. Pointers are computed host-side
// only (get_a2av_schedule_workspace) and passed into the prepare kernel, so
// there is no device-side offset mirror to keep in sync.
struct A2AVScheduleWorkspace {
  ProblemInfo *bucket_tiles = nullptr;  // [max_tiles], grouped by source bucket
  int *bucket_offsets = nullptr;        // [world_size + 2] prefix offsets
  int *bucket_cursors = nullptr;        // [world_size + 1] claim cursors
  uint64_t *multi_masks = nullptr;      // [max_tiles] source masks (bucket W only)
  size_t bytes_end = 0;                 // total workspace bytes incl. these regions
};

inline A2AVScheduleWorkspace
get_a2av_schedule_workspace(
    void *workspace_base, size_t dense_bytes, int max_tiles, int world_size) {
  auto pad128 = [](size_t x) { return (x + 127) / 128 * 128; };
  auto at = [&](size_t off) { return (void *)((char *)workspace_base + off); };
  A2AVScheduleWorkspace w;
  size_t off = pad128(dense_bytes);
  w.bucket_tiles = (ProblemInfo *)at(off);
  off = pad128(off + sizeof(ProblemInfo) * max_tiles);
  w.bucket_offsets = (int *)at(off);
  off = pad128(off + sizeof(int) * (world_size + 2));
  w.bucket_cursors = (int *)at(off);
  off = pad128(off + sizeof(int) * (world_size + 1));
  w.multi_masks = (uint64_t *)at(off);
  off = pad128(off + sizeof(uint64_t) * max_tiles);
  w.bytes_end = off;
  return w;
}

void make_workspace_async(
    const GemmGroupedV2AGScatterArguments &args,
    GemmLayoutEnum layout,
    int input_elem_size,
    int output_elem_size,
    int threadblock_count,
    void *workspace,
    cudaStream_t stream,
    A2AVScheduleWorkspace a2av_ws = {});

/**
 * @brief Get the sorted problem schedule cuda v2 object
 *
 * @param splits
 * @param rank
 * @param tp_size
 * @param cumsum_per_rank_ptr
 * @param ep_start
 * @param ep_nexperts
 * @param tiled_m_size
 * @param num_weight_groups
 * @return std::vector<ProblemSchedule>
 */
void get_sorted_problem_schedule_cuda_v2(
    const int32_t *const splits,
    int rank,
    int tp_size,
    const int *cumsum_per_rank_ptr,
    const int ep_start,
    const int ep_nexperts,
    const int tiled_m_size,
    const int num_weight_groups,
    ProblemSchedV2 *problem_schedules,
    cudaStream_t stream);

}  // namespace bytedance::flux
