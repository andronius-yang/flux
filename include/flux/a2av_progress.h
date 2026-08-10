//===- a2av_progress.h ---------------------------------------- C++ ------===//
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
// FLUX_A2AV_NVTX_PROXY=1: per-source-bucket tile-progress slots living in
// host-mapped pinned memory. The a2av GEMM kernel (tile claimer + persistent
// loop) publishes coarse progress; a host poller thread (a2av_nvtx_proxy.hpp)
// converts the transitions into NVTX ranges so per-source wait/compute shows
// up inside the otherwise-opaque grouped-GEMM span on an nsys timeline.
//
// Race-freedom by monotonicity: no slot is ever reset. Epoch-tagged slots
// ("_seq") carry the NVSHMEM run-id, which increases every iteration, and the
// tile counters only ever increment; the poller snapshots per-epoch baselines
// instead of anyone zeroing memory. This means enqueue-ahead host code never
// races a still-running iteration's device writes.

#pragma once

#include <cstdint>

namespace bytedance::flux {

struct A2AVProgressSlots {
  // world_size is bounded by the 64-bit multi-source masks; +1 for bucket W
  // (multi-source tiles), which reports as its own pseudo-source.
  static constexpr int kMaxBuckets = 65;

  // -- device -> host ------------------------------------------------------
  // written once per epoch by the first CTA observing the source's signal
  uint64_t ready_seq[kMaxBuckets];
  // monotonic tile counters (device atomics). claimed[s] - baseline is
  // also the bucket cursor value, i.e. exactly tiles [0, k) of s have fired.
  uint32_t claimed[kMaxBuckets];
  uint32_t completed[kMaxBuckets];
  // per-bucket tile totals for the epoch in expected_seq (CTA 0, launch start)
  uint64_t expected_seq;
  uint32_t expected[kMaxBuckets];
  uint32_t num_buckets;  // world_size + 1

  // -- layer C: device-clock stamps + tile-trace cursor --------------------
  // kernel-entry %globaltimer (CTA 0), tagged by t0_seq; every _rel time in
  // the tile trace and every arrival payload is relative to t0_gt
  uint64_t t0_gt;
  uint64_t t0_seq;
  // %globaltimer at the same observation that writes ready_seq[s]
  uint64_t arrival_gt[kMaxBuckets];
  // monotonic append cursor into the tile-trace buffer (never reset; the
  // poller snapshots per-epoch baselines) — also an exact "tiles fired" count
  uint32_t trace_cursor;
};

// One fired tile (layer C trace). Written in three touches so only the record
// index stays live across the GEMM mainloop (register discipline). Times are
// the LOW 32 BITS of absolute %globaltimer ns — the hot path never reads
// t0_gt (CTA 0 may stamp it after other CTAs already fired); the analysis
// script rebases with wrap-safe u32 subtraction against low32(t0_gt), exact
// for any span < 4.3 s.
struct A2AVTileRecord {
  uint32_t tile;     // problem_idx (10b) << 22 | problem-local tile idx (22b)
  uint32_t meta;     // smid (8b) << 24 | seg_start (6b) << 18 | seg_end (6b) << 12 | cta (12b)
  uint32_t t_enter;  // low32 %globaltimer: CTA reached the tile (spin entry)
  uint32_t t_fire;   // low32 %globaltimer: spin exit / mainloop start
  uint32_t t_done;   // low32 %globaltimer: tile retired (epilogue done)
  uint32_t pad;
};

#if defined(__CUDACC__)
// dependency-free %globaltimer read for the instrumentation sites (this header
// is included from both host-only TUs and cutlass kernel headers)
__device__ __forceinline__ uint64_t
a2av_globaltimer() {
  uint64_t v;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
  return v;
}
#endif

}  // namespace bytedance::flux
