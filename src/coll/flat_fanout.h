//===- flat_fanout.h ---------------------------------------------- C++ ---===//
//
// Copyright 2026 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License").
//
//===----------------------------------------------------------------------===//
// Device-issued fenced fan-out for the FLAT a2av dispatch wire (dov arm,
// FLUX_A2AV_FLAT_FENCED_DEV=1, 2026-09-17). Replaces the host-issued
// per-destination stream ops of FLUX_A2AV_FLAT_FENCED_SIG (W-1 nbi puts +
// quiet + W-1 signal ops, ~2W launches whose issue cost is what made dov
// lose at 8n/16n 1 MiB — handoff 31 §4b) with TWO kernels on the wire
// stream:
//   1. flat_fanout_put: one block per destination in ring order (rank+1
//      first). Remote: nvshmemx_putmem_nbi_block (concurrent, the NIC
//      pipelines the fan-out). Intra-node: nvshmemx_putmem_signal_nbi_block
//      (NVLink P2P store order — the audited intra configuration) so near
//      sources release per source. Zero payload: signal only.
//   2. flat_fanout_quiet_signal: ONE thread does nvshmem_quiet() (the same
//      PE-wide proxy quiet the on-stream quiet performs), then emits the
//      remote destinations' signal ops in ring order — the F2
//      quiet-then-signal pattern (wire-ordering HARD RULE: never an nbi
//      put_signal gate on the CXI wire).
// Per-destination tables live in device memory the caller fills (one small
// H2D per forward from a pinned staging buffer).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace bytedance {
namespace flux {

struct FlatFanoutParams {
  const int64_t *dst_off_bytes;  // [W] byte offset into recv_base at destination d
  const int64_t *src_off_bytes;  // [W] byte offset into send_base for destination d
  const int64_t *bytes;          // [W] payload bytes to destination d (0 = signal only)
  const int32_t *is_intra;       // [W] 1 = same node (P2P put_signal), 0 = remote
  char *recv_base;               // symmetric recv region (remote address base)
  const char *send_base;         // local send staging
  uint64_t *signal;              // symmetric per-source signal slots (index = source rank)
  uint64_t signal_value;         // epoch to SET
  int rank;
  int world_size;
};

// grid = world_size - 1 blocks (block i -> destination (rank + 1 + i) % W)
void flat_fanout_put(const FlatFanoutParams &p, cudaStream_t stream);
// grid = 1 block: PE quiet, then remote signal ops in ring order
void flat_fanout_quiet_signal(const FlatFanoutParams &p, cudaStream_t stream);

}  // namespace flux
}  // namespace bytedance
