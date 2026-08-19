//===----------------------------------------------------------------------===//
//
// EPIC §4.3 in-kernel expert swap — see epic_swap.hpp for the contract.
//
// Exchange protocol (race-freedom): each rank writes only its OWN slot and
// reads only the peer's epoch-immutable scratch. The snapshot decouples the
// send-side data from the receive-side write target, so the bidirectional
// swap needs exactly one flag per rank and no ordering between the two
// pulls. Scratch reuse across epochs is fenced by dispatch_only's two
// end-of-call nvshmemx_barrier_all_on_stream: the peer's NEXT snapshot is
// stream-ordered behind its epoch-close barrier, which cannot complete
// until this rank (whose pull precedes its own barrier arrival) arrives.
//
// Spin safety: this kernel is the FIRST device op of the iteration on both
// pair members; the swap plan is replicated and pair-symmetric, so the peer
// provably launches the same-epoch kernel with no dependency on this rank
// before its release store — the wait is bounded by correctness, not a
// watchdog.
//
//===----------------------------------------------------------------------===//
#include "epic_swap.hpp"

#include <algorithm>

#include "flux/cuda/cuda_common.h"
#include "flux/a2av_progress.h"

namespace bytedance::flux {

namespace {

__device__ __forceinline__ uint64_t
ld_acquire_sys_u64(const uint64_t *ptr) {
  uint64_t v;
  asm volatile("ld.global.acquire.sys.b64 %0, [%1];\n" : "=l"(v) : "l"(ptr));
  return v;
}

__device__ __forceinline__ void
st_release_sys_u64(uint64_t *ptr, uint64_t v) {
  asm volatile("st.global.release.sys.b64 [%0], %1;\n" ::"l"(ptr), "l"(v));
}

// flat 16B-granule view over {fc1, fc2}: granule i < n1 lives in fc1
__device__ __forceinline__ uint4 *
granule(void *fc1, void *fc2, int64_t n1, int64_t i) {
  return (i < n1) ? reinterpret_cast<uint4 *>(fc1) + i
                  : reinterpret_cast<uint4 *>(fc2) + (i - n1);
}

__global__ __launch_bounds__(1024, 1) void epic_swap_exchange_kernel(EpicSwapParams p) {
  const int64_t n1 = p.fc1_bytes / 16;
  const int64_t n = (p.fc1_bytes + p.fc2_bytes) / 16;
  const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t nthreads = (int64_t)gridDim.x * blockDim.x;
  if (p.stamps != nullptr && blockIdx.x == 0 && threadIdx.x == 0) {
    p.stamps[0] = a2av_globaltimer();
  }
  // phase 1: SNAPSHOT — my slot(s) -> my symmetric scratch (local HBM)
  uint4 *scratch = reinterpret_cast<uint4 *>(p.my_scratch);
  for (int64_t i = tid; i < n; i += nthreads) {
    scratch[i] = *granule(p.my_fc1_slot, p.my_fc2_slot, n1, i);
  }
  // phase 2: system-wide arrival; last block publishes the epoch flag
  __syncthreads();
  __threadfence_system();
  if (threadIdx.x == 0) {
    unsigned long long old = atomicAdd(p.arrive, 1ULL);
    if (old + 1 == p.arrive_base + gridDim.x) {
      st_release_sys_u64(p.my_flag, p.epoch);
      if (p.stamps != nullptr) {
        p.stamps[1] = a2av_globaltimer();
      }
    }
  }
  // phase 3: SPIN on the peer's flag (GEQ, monotone — never reset)
  if (threadIdx.x == 0) {
    while (ld_acquire_sys_u64(p.peer_flag) < p.epoch) {
    }
    if (p.stamps != nullptr && blockIdx.x == 0) {
      p.stamps[2] = a2av_globaltimer();
    }
  }
  __syncthreads();
  // phase 4: PULL — peer scratch (direct NVLink loads) -> my slot(s)
  const uint4 *peer = reinterpret_cast<const uint4 *>(p.peer_scratch);
  for (int64_t i = tid; i < n; i += nthreads) {
    *granule(p.my_fc1_slot, p.my_fc2_slot, n1, i) = peer[i];
  }
  __syncthreads();
  __threadfence_system();
  if (threadIdx.x == 0) {
    unsigned long long old = atomicAdd(p.arrive, 1ULL);
    if (old + 1 == p.arrive_base + 2ULL * gridDim.x && p.stamps != nullptr) {
      p.stamps[3] = a2av_globaltimer();
    }
  }
}

}  // namespace

int
epic_swap_exchange(const EpicSwapParams &params, int num_sm, cudaStream_t stream) {
  const int64_t n = (params.fc1_bytes + params.fc2_bytes) / 16;
  int grid = (int)std::min<int64_t>(std::max(num_sm, 1), (n + 1023) / 1024);
  epic_swap_exchange_kernel<<<grid, 1024, 0, stream>>>(params);
  CUDA_CHECK(cudaGetLastError());
  return grid;
}

void
epic_swap_preload() {
  cudaFuncAttributes attr;
  CUDA_CHECK(cudaFuncGetAttributes(&attr, epic_swap_exchange_kernel));
}

}  // namespace bytedance::flux
