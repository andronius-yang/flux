//===- a2av_tile_claimer.hpp ---------------------------------- C++ ------===//
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
// Dynamic tile claiming for the a2av dispatch mode of the sm80 AG-Scatter
// grouped GEMM. Tiles are pre-bucketed by the single source rank they depend
// on (bucket s in [0, W)), with multi-source tiles in bucket W. Persistent
// CTAs claim tiles via per-bucket atomic cursors, preferring buckets whose
// source signal has already arrived, so no CTA blocks behind a slow source
// while other sources' tiles are ready. Signal reads here are only a
// scheduling heuristic; the per-tile acquire spin in the GEMM kernel is the
// correctness backstop.

#pragma once

#include "cutlass/cutlass.h"

namespace cutlass {
namespace gemm {
namespace kernel {

struct A2AVTileClaimer {
  int const *bucket_offsets;    // [world_size + 2] prefix offsets into bucket_tiles
  int *bucket_cursors;          // [world_size + 1] claim cursors, zeroed per launch
  uint64_t const *multi_masks;  // [num multi tiles] source mask per bucket-W tile
  uint64_t const *signal_ptr;   // [world_size] per-source epoch signals
  uint64_t signal_expected;     // current run-id epoch
  int world_size;

  CUTLASS_DEVICE
  A2AVTileClaimer(
      int const *bucket_offsets_,
      int *bucket_cursors_,
      uint64_t const *multi_masks_,
      uint64_t const *signal_ptr_,
      uint64_t signal_expected_,
      int world_size_)
      : bucket_offsets(bucket_offsets_),
        bucket_cursors(bucket_cursors_),
        multi_masks(multi_masks_),
        signal_ptr(signal_ptr_),
        signal_expected(signal_expected_),
        world_size(world_size_) {}

  // Claim one tile. Returns the claimed index into bucket_tiles, or -1 when
  // every bucket is drained. Called by thread 0 only; the caller broadcasts
  // the result through shared memory.
  CUTLASS_DEVICE
  int
  claim() {
    int const W = world_size;
    while (true) {
      bool any_remaining = false;
      uint64_t arrived = 0;
      // single-source buckets, staggered by CTA id to spread contention
      for (int i = 0; i < W; ++i) {
        int s = (i + blockIdx.x) % W;
        bool ready =
            (*reinterpret_cast<uint64_t const volatile *>(signal_ptr + s)) >= signal_expected;
        if (ready) {
          arrived |= (uint64_t(1) << s);
        }
        int size = bucket_offsets[s + 1] - bucket_offsets[s];
        if (*reinterpret_cast<int const volatile *>(bucket_cursors + s) >= size) {
          continue;
        }
        any_remaining = true;
        if (!ready) {
          continue;
        }
        int idx = atomicAdd(bucket_cursors + s, 1);
        if (idx < size) {
          return bucket_offsets[s] + idx;
        }
      }
      // multi-source bucket: claim when the peeked tile's full source set has
      // arrived, or when no single-source work is left anywhere (the per-tile
      // spin then blocks, which is optimal — nothing else to run).
      int msize = bucket_offsets[W + 1] - bucket_offsets[W];
      int mcur = *reinterpret_cast<int const volatile *>(bucket_cursors + W);
      if (mcur < msize) {
        uint64_t need = multi_masks[mcur];  // peek; races are benign (backstop spin)
        if ((need & ~arrived) == 0 || !any_remaining) {
          any_remaining = true;
          int idx = atomicAdd(bucket_cursors + W, 1);
          if (idx < msize) {
            return bucket_offsets[W] + idx;
          }
        } else {
          any_remaining = true;
        }
      }
      if (!any_remaining) {
        return -1;
      }
#if __CUDA_ARCH__ >= 700
      __nanosleep(200);
#endif
    }
  }
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
