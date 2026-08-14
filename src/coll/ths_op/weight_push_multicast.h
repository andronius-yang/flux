//===- weight_push_multicast.h ------------------------------------ C++ ---===//
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

// Source-initiated expert-weight PUSH with optional node-level multicast —
// the concurrent-flow counterpart of WeightPrefetchGetmem (scenario 2 of the
// moonep-fused ablations). Home ranks push expert rows from their symmetric
// weight tensor straight into destination ranks' symmetric prefetch-slot
// rows via HOST-ISSUED CE nvshmemx_putmem_signal (no SM kernels — NR-02
// Class A safe beside a resident GEMM; the a2av_comm_bench measured issue
// path as bandwidth-irrelevant, the CXI proxy is the limiter). Per-slot
// symmetric u64 signals carry a monotonic run_id epoch (SET + GEQ, never
// memset). NO team collectives, barriers, or communicators anywhere, so the
// op is safe on a side stream CONCURRENT with the token a2av (same
// clearance as the getmem pull, NR-12 fact 8 / NR-02 Class B: every wait's
// writer is a remote rank).
//
// Multicast mode: one inter-node put per (expert, dest node) lands in a
// plan-chosen gateway's own slot; the gateway forwards to the other needy
// ranks of its node over NVLink CE puts, paced by a zero-SM
// CUStreamWaitValue64 on its own slot signal. Direct mode pushes every pair
// point-to-point (the M3 baseline the multicast is ablated against).
class WeightPushMulticast {
 public:
  // Collective ctor: allocates the contiguous symmetric
  // [n_experts_local + n_slots, row_dim0, row_dim1] weight tensor (home
  // rows [0, epn), slots [epn, epn+n_slots) — upstream MoonEP's [E+B]
  // mapped layout) and the u64[n_slots] signal array.
  WeightPushMulticast(
      std::shared_ptr<Group> pg,
      int64_t n_experts_local,
      int64_t n_slots,
      int64_t row_dim0,
      int64_t row_dim1,
      at::ScalarType dtype);

  ~WeightPushMulticast();

  torch::Tensor weight_full();      // [epn + n_slots, r0, r1] symmetric
  torch::Tensor weight_home();      // narrow view [0, epn)
  torch::Tensor prefetch_slots();   // narrow view [epn, epn + n_slots)
  torch::Tensor signals();          // symmetric int64[n_slots] epoch signals

  // Replicated plan, the FULL pair list on every rank (identical —
  // assign_gateways in flux.testing.moonep_fused_map): int32 CPU
  // [n_pairs, 6] = (dst_rank, dst_slot, home_rank, src_row, gw_rank,
  // gw_slot); gw_rank == -1 => direct home -> dst leg (also the gateway
  // member's own inter-node leg). Direct-mode forward ignores gw fields.
  void set_plan(torch::Tensor pairs_cpu);

  // Issue this rank's HOME-role pushes on the current torch stream (direct
  // mode: every pair; multicast mode: the gw<0 legs — intra-home-node
  // dests, singletons, and each cross-node group's single inter-node leg
  // into the gateway's own slot). Increments and returns the epoch. NO
  // waits of any kind (NR-13 F-B: an event recorded after this call means
  // "issued", never "arrived", so it can gate a forward launch safely).
  // All puts are nbi CE puts from symmetric memory; completion is observed
  // via the per-slot signals (join()/the fused GEMM's weight gate), and the
  // token a2av's end-of-iteration barrier_all quiets the nbi tails.
  int64_t forward(bool multicast);

  // GATEWAY role, issued separately so its slot-arrival wait NEVER sits in
  // the launch path (NR-13 fact 3: the wait in the issue window held the
  // gateway's token wire hostage): per gateway slot one zero-SM
  // CUStreamWaitValue64 (GEQ current epoch; writer is a remote home rank —
  // NR-02 Class-B safe), then NVLink CE putmem_signal fan-out to the needy
  // same-node peers. No-op for non-gateway ranks and in direct mode. Run it
  // on a stream whose completion is event-joined into the iteration only
  // AFTER the fused forward launch (the fan-out then overlaps the forward;
  // receivers' weight-gated tiles absorb any residual arrival latency).
  void forward_gateway();

  // Destination-side landing gate: zero-SM stream waits (GEQ current epoch)
  // on each of MY incoming slots, on the current torch stream.
  void join();

  int64_t epoch() const;

 private:
  class WeightPushMulticastImpl;
  WeightPushMulticastImpl *impl_;
};

}  // namespace bytedance::flux::ths_op
