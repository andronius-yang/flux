
//===- workspace_helper.h -------------------------------------- C++ ------===//
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
#include <cutlass/gemm_coord.h>
#include <cutlass/layout/matrix.h>

#include "flux/args/moe_gather_rs.h"
namespace bytedance::flux {

struct MoeGatherRSWorkspaceArgs {
  int num_groups;
  int N_split;
  int ep_start;
  int ep_nexperts;
  int N, K;
  int32_t *splits_gpu;
  void *input[kMaxNumGroups];
  void *weights[kMaxNumGroups];
  void *output[kMaxNumGroups];
  float *input_scales[kMaxNumGroups];
  float *weight_scales[kMaxNumGroups];
  // M-split waves (Slipstream v2, FLUX_A2AV_RS_MSPLIT): problems become
  // (ring wave of dest nodes, expert) ROW segments (full N) instead of
  // (split, expert) column windows. Tables are device int32, built by the host
  // per iteration from splits_per_source (pinned-arena async H2D on the same
  // stream). msplit == 0 leaves the legacy construction bit-exact.
  int msplit = 0;
  int n_waves = 0;
  const int32_t *wave_M = nullptr;              // [n_waves * ep_nexperts] segment rows
  const int32_t *wave_off = nullptr;            // [n_waves * ep_nexperts] rows into the expert block
  const int32_t *non_empty_per_wave = nullptr;  // [n_waves]
  // v2 chunked combine (FLUX_A2AV_RS_CHUNK_E): explicit problem -> expert map
  // for chunk-ordered problem lists; nullptr = legacy i % ep_nexperts.
  const int32_t *prob_eid = nullptr;  // [n_waves * ep_nexperts]
  int *barrier = nullptr;  // preset zero-target wave flags to 1 (no producer exists)
  // gen-8c epilogue-fused pack: the ScatterD indices per problem. Legacy:
  // scatter_D_ptr[i] = iota (relative identity, ptr_D keeps its M_acc base).
  // Fused: scatter_D_ptr[i] = inv_pack + problem's global row base, and ptr_D
  // becomes the send-panel base (absolute panel rows) — the GEMM writes the
  // dest-major panel directly and the pack kernel degenerates to a flag relay.
  int *iota = nullptr;             // [max rows] identity indices (shared)
  int *inv_pack = nullptr;         // [M_this_ep] gemm row -> send-panel row
  void *send_panel = nullptr;      // fused: D base override (row stride == N)
  int fused_pack = 0;
};

/**

 workspace architecture

problem_sizes, cutlass::gemm::GemmCoord *, problem_count
ptr_A, void *, problem_count
ptr_B, void *, problem_count
ptr_C, void *, problem_count
ptr_D, void *, problem_count
lda, int64_t, problem_count
ldb, int64_t, problem_count
ldc, int64_t, problem_count
ldd, int64_t, problem_count
scale_A, float *, problem_count
scale_B, float *, problem_count
non_empty_problem_count, int, 1
 */

void make_workspace(
    const MoeGatherRSWorkspaceArgs &args,
    GemmLayoutEnum layout,
    int input_elem_size,
    int output_elem_size,
    void *workspace,
    cudaStream_t stream);
}  // namespace bytedance::flux
