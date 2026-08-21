//===- fused_ep_dispatch.h ---------------------------------------- C++ ---===//
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

// DeepEP-lineage fused expert dispatch/combine (campaign-2 planner v2a,
// CANONICAL for the eplb arm and epic's direct wire — replaces the staged
// All2AllSingle path). No planning step exists: the caller provides a
// sender-local [S, K] physical-slot routing (replica selection is a local
// rule), a tiny per-slot counts vector rides INSIDE the dispatch launch
// (the DeepEP internode.cu notify_dispatch analog — zero host
// collectives), token rows land at exact deterministic remote offsets
// (local-slot-major, then source, then stable flat-cell order), and the
// per-row int4 header {flat_cell, prob_bits, src, 0} doubles as the
// combine handle. Arrival ordering is put_nbi -> fence -> signal_op(ADD)
// (the IBGDA same-QP trick of upstream does not port to Slingshot);
// probe_signal_ordering() validates the sequence on the live transport
// and MUST pass before any bring-up (campaign S4 hard gate).
//
// Signals are epoch-monotone u64 (calloc once, never memset): after epoch
// k, sig_counts/sig_data slots hold k and sig_comb slots hold k * nlp;
// every wait is GEQ. One dispatch() (and at most one combine()) per
// epoch; run_id advances inside dispatch().
class FusedEpDispatch {
 public:
  // Collective ctor (identical arguments on every rank — enforced by an
  // all_gather_cpu contract check): symmetric recv/staging/signal
  // allocation, NVSHMEM + kernel priming (on a dedicated prime signal so
  // the monotone real signals stay untouched), and a
  // CUDA_DEVICE_MAX_CONNECTIONS > 1 check (a resident spin gate ahead of
  // a later-enqueued transport kernel deadlocks at conn=1).
  //   nlp:               local physical slots per rank (P = world * nlp)
  //   max_rows_per_pair: per-(slot, src) recv bound (collective trap)
  //   max_recv_total:    total recv rows bound (sizes recv buffers)
  //   m_groups:          1 = gate everything inside dispatch();  > 1 =
  //                      dispatch() skips the recv gate and the caller
  //                      pipelines wait_group(g) per contiguous slot group
  FusedEpDispatch(
      std::shared_ptr<Group> pg,
      int64_t nnodes,
      int64_t s_max,
      int64_t hidden,
      int64_t topk,
      int64_t nlp,
      int64_t max_rows_per_pair,
      int64_t max_recv_total,
      at::ScalarType dtype,
      int64_t m_groups,
      int64_t spin_limit);

  ~FusedEpDispatch();

  // The whole fused journey on the current torch stream: pack (stable
  // counting sort by dst_phys) -> in-launch counts exchange -> exact-
  // offset segment puts + per-(slot, src) arrival signals -> recv gate
  // (m_groups == 1) -> weights extract -> ONE pinned D2H of the [2*nlp]
  // seg metadata (event-synced: the phase's honest host sync).
  // Returns {recv_rows [n_recv, hidden] view, weights [n_recv] f32 view,
  // seg_meta [2*nlp] int32 CPU (seg_rows | seg_start)}.
  std::vector<torch::Tensor> dispatch(
      torch::Tensor inputs_shard,   // [S, hidden] dtype device
      torch::Tensor dst_phys,       // [S, K] int32 device, global slot ids
      torch::Tensor probs,          // [S, K] f32 device
      int64_t num_comm_sm);

  // Per-group recv gate for m_groups > 1 (contiguous local-slot range).
  void wait_group(int64_t g, int64_t num_comm_sm);

  // l01 combine, expert side: put each gemm2 row into the SOURCE's home
  // staging cell (header-addressed; deterministic, each (token, k) cell
  // written exactly once), quiet, then always-signal nlp ADDs per source.
  void combine(torch::Tensor expert_rows, int64_t num_comm_sm);

  // l01 combine, home side: wait sig_comb[s] >= run_id * nlp for all s,
  // then return the [S*K, hidden] staging view (the caller applies its
  // home-local route probs in fp32 and reduces over K — receiver-side
  // weight application, the DeepEP combine convention).
  torch::Tensor combine_gate(int64_t s_tokens, int64_t num_comm_sm);

  // Standalone S4 ordering probe: `iters` epochs of put->fence->signal
  // (form 0) and putmem_signal (form 1) against peer (rank+1)%world, both
  // directions, payload verified word-by-word. Hard-fails on any stale
  // read. Collective.
  void probe_signal_ordering(int64_t iters);

  torch::Tensor recv_rows();   // full-capacity [max_recv_total, hidden]
  torch::Tensor headers();     // [max_recv_total, 4] int32

 private:
  class FusedEpDispatchImpl;
  FusedEpDispatchImpl *impl_;
};

}  // namespace bytedance::flux::ths_op
