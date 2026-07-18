//===- moe_ag_scatter.h ------------------------------------------- C++ ---===//
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
#include "./comm_none.h"
#include "flux/utils.h"

namespace bytedance::flux {
constexpr int kMaxNumGroups = 2;

struct GemmGroupedAgScatterArguments : GemmGroupedV3Arguments {
  DistEnv dist_env;
  int ntokens;
  int h;
  void *nvshmem_input_buffer;
  int32_t const **gather_A;
  int32_t const **scatter_D;
  void const *problem_schedule;
  void *barrier_ptr = nullptr;
  int sm_margin = 0;
};

struct GemmGroupedV2AGScatterArguments {
  int rank = 0;
  int world_size = 1;
  // node-aware rank ordering for the tile schedule; nnodes==1 reduces to legacy behavior
  DistEnv dist_env;
  int sm_margin = 0;

  int num_groups = 0;  // make sure num_groups <= kMaxNumGroups
  int ep_start = 0;
  int ep_nexperts = 0;
  void *input = nullptr;                 // before gather_A
  void *weight[kMaxNumGroups] = {};      // with groups
  void *output[kMaxNumGroups] = {};      // with groups
  // FP8 arguments
  float *scaleD[kMaxNumGroups] = {};  // with groups
  int M_this_ep = 0, N = 0, K = 0;
  int lda = 0, ldb = 0, ldc = 0, ldd = 0;
  int *splits = nullptr;
  // calculated on prepare workspace
  int32_t *gather_A = nullptr;   // on device memory expected
  int32_t *scatter_D = nullptr;  // on device memory expected
  void *problem_schedules = nullptr;
  int num_problem_schedules = 0;
  int *accum_per_rank_ptr = nullptr;  // on device memory expected
  int tile_size_m = 0, tile_size_n = 0;
  int *barrier_ptr = nullptr;
  // a2av dispatch mode (raw alltoallv): per-source-rank NVSHMEM signals replace
  // barrier_ptr, compared against the run-id epoch. nullptr == legacy dense mode.
  uint64_t *signal_ptr = nullptr;
  uint64_t signal_expected = 0;
  // a2av_ring mode: sends follow the reverse hierarchical ring, so the dense
  // static problem schedule is used instead of the dynamic tile-claimer buckets.
  bool a2av_ring_schedule = false;
  // fill inside op. only Op has the information
  float alpha = 1.f;
  float beta = 0.f;
};

}  // namespace bytedance::flux
