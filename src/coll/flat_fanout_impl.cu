//===- flat_fanout_impl.cu ---------------------------------------- C++ ---===//
//
// Copyright 2026 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License").
//
//===----------------------------------------------------------------------===//
// See flat_fanout.h. Pattern precedent: all2all_impl.cu (per-block
// nvshmemx_putmem_nbi_block fan-out) + the F2 quiet-then-signal ordering of
// the host fenced wire it replaces.
#include "coll/flat_fanout.h"

#include <nvshmem.h>
#include <nvshmemx.h>

namespace bytedance {
namespace flux {

namespace {

constexpr int kThreads = 256;

__global__ void __launch_bounds__(kThreads, 1)
flat_fanout_put_kernel(const FlatFanoutParams p) {
  const int W = p.world_size;
  const int d = (p.rank + 1 + blockIdx.x) % W;
  const int64_t b = p.bytes[d];
  if (b == 0) {
    if (threadIdx.x == 0) {
      nvshmemx_signal_op(p.signal + p.rank, p.signal_value, NVSHMEM_SIGNAL_SET, d);
    }
    return;
  }
  char *dst = p.recv_base + p.dst_off_bytes[d];
  const char *src = p.send_base + p.src_off_bytes[d];
  if (p.is_intra[d]) {
    nvshmemx_putmem_signal_nbi_block(
        dst, src, (size_t)b, p.signal + p.rank, p.signal_value, NVSHMEM_SIGNAL_SET, d);
  } else {
    nvshmemx_putmem_nbi_block(dst, src, (size_t)b, d);
  }
}

__global__ void __launch_bounds__(32, 1)
flat_fanout_quiet_signal_kernel(const FlatFanoutParams p) {
  if (threadIdx.x != 0) return;
  const int W = p.world_size;
  // PE-wide completion of every nbi put issued by this PE (all blocks of the
  // put kernel, stream-ordered before us), then the remote signals in ring
  // order. Intra-node destinations were signalled by their put_signal.
  nvshmem_quiet();
  for (int i = 1; i < W; i++) {
    const int d = (p.rank + i) % W;
    if (p.bytes[d] > 0 && !p.is_intra[d]) {
      nvshmemx_signal_op(p.signal + p.rank, p.signal_value, NVSHMEM_SIGNAL_SET, d);
    }
  }
}

}  // namespace

void
flat_fanout_put(const FlatFanoutParams &p, cudaStream_t stream) {
  const int n = p.world_size - 1;
  if (n <= 0) return;
  flat_fanout_put_kernel<<<n, kThreads, 0, stream>>>(p);
}

void
flat_fanout_quiet_signal(const FlatFanoutParams &p, cudaStream_t stream) {
  flat_fanout_quiet_signal_kernel<<<1, 32, 0, stream>>>(p);
}

}  // namespace flux
}  // namespace bytedance
