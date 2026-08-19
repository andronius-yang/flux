//===----------------------------------------------------------------------===//
//
// EPIC §4.3 in-kernel expert swap (intra-node, NVLink peer-VA exchange).
//
// The swap DECISION is host-side (paper: overlapped with the pre-dispatch
// cast); this kernel only APPLIES one planned swap: it exchanges the weight
// bytes of one expert slot between the two paired GPUs of a node, as the
// FIRST device op of the dispatch launch sequence ("the kernel first runs a
// short rebalance phase to apply expert swaps, then enters the dispatch
// phase"). Strictly sequential with the token wire — no overlap by design.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace bytedance::flux {

struct EpicSwapParams {
  // this rank's weight slot storage (plain device memory; never read remotely)
  void *my_fc1_slot;
  void *my_fc2_slot;  // nullptr when fc2 is not placed
  int64_t fc1_bytes;
  int64_t fc2_bytes;  // 0 when fc2 is not placed
  // exchange staging: my_scratch is symmetric; peer_scratch is the
  // nvshmem_ptr NVLink VA of the PEER's scratch (same-node peer only)
  void *my_scratch;
  const void *peer_scratch;
  // symmetric u64 epoch flags (never memset; monotone SET / GEQ discipline)
  uint64_t *my_flag;
  const uint64_t *peer_flag;
  uint64_t epoch;  // replicated swap-sequence number, starts at 1
  // LOCAL arrival counter (u64, never reset). arrive_base = the counter's
  // value before this launch (caller advances it by 2*grid per launch,
  // using the grid the launcher returns); the kernel derives its goals as
  // base + gridDim.x (flag release) and base + 2*gridDim.x (t3 stamp).
  unsigned long long *arrive;
  unsigned long long arrive_base;
  // LOCAL u64[4] %globaltimer stamps {entry, snapshot_done, peer_observed,
  // pull_done}; nullptr disables (FLUX_A2AV_TIMING off)
  uint64_t *stamps;
};

// Launch the exchange kernel on `stream` with at most `num_sm` blocks.
// Returns the grid size actually used (the caller advances its arrival
// goals by 2*grid per launch).
int epic_swap_exchange(const EpicSwapParams &params, int num_sm, cudaStream_t stream);

// cudaFuncGetAttributes preload (CUDA_MODULE_LOADING=LAZY hang class —
// call once from the op ctor on an idle device).
void epic_swap_preload();

}  // namespace bytedance::flux
