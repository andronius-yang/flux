//===- gemm_grouped_v2_ag_scatter.h ------------------------------- C++ ---===//
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
#include <c10/core/ScalarType.h>
#include <torch/all.h>
#include "coll/ths_op/all_gather_types.h"
#include "flux/ths_op/ths_op.h"

namespace bytedance::flux::ths_op {
std::tuple<
    int,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
prepare_moe_ag_scatter_args(
    torch::Tensor splits_gpu,
    torch::Tensor scatter_index,
    int ntokens,
    int topk,
    int num_weights_group,
    int ep_start,
    int ep_nexperts,
    int rank,
    int world_size,
    int tile_size_m,
    intptr_t stream_);

class GemmGroupedV2AGScatterOp {
 public:
  GemmGroupedV2AGScatterOp(
      std::shared_ptr<Group> tp_group,
      int ep_size,
      int nnodes,
      int max_ntokens,
      int ffn_hidden,  // before TP shard
      int hidden,
      int num_experts,
      int topk,
      at::ScalarType input_dtype,
      at::ScalarType output_dtype,
      bool a2av_dispatch = false,
      bool a2av_ring = false,
      bool a2av_hier = false,
      bool a2av_hier_compress = false);
  ~GemmGroupedV2AGScatterOp();
  void clear_buffers();
  torch::Tensor forward(
      torch::Tensor inputs_shard,
      torch::Tensor weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      c10::optional<torch::Tensor> outputs_buf,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin,
      AllGatherOptionWithOptional ag_option,
      // metadata-exchange result: int32 CPU [world_size, nexperts] per-source
      // per-expert copy counts; splits is its column sum. nullopt = derive
      // everything from splits/scatter_index as before.
      c10::optional<torch::Tensor> splits_per_source = c10::nullopt,
      // a2av_hier_compress metadata: int32 CPU [world_size, world_size + nnodes]
      // dedup counts — cols [0, W) = unique tokens source s -> rank d, cols
      // [W, W + nnodes) = unique tokens source s -> node-n union. Identical on
      // all ranks; required (with splits_per_source) in compress mode.
      c10::optional<torch::Tensor> a2av_unique_counts = c10::nullopt,
      // weight-gated tiles (moonep_fused scenario 2): CUDA u64/int64 epoch
      // signals (>= n_slots elements, e.g. WeightPushMulticast.signals());
      // problems with local group >= weight_gate_group_start spin on
      // weight_signal[group - start] >= weight_signal_epoch. a2av static-
      // schedule modes only. nullopt/-1 = no weight gating.
      c10::optional<torch::Tensor> weight_signal = c10::nullopt,
      int64_t weight_signal_epoch = 0,
      int64_t weight_gate_group_start = -1);
  // Dispatch-only entry (EPIC baseline, a2av modes): runs the dispatch wire
  // WITHOUT the fused GEMM and materializes the received rows densely.
  // Returns (dense_rows [M_this_ep, hidden], sorted_scatter_index,
  // sorted_splits_cumsum [ep_nexperts, world_size], M_this_ep).
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t> dispatch_only(
      torch::Tensor inputs_shard,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<torch::Tensor> splits_per_source = c10::nullopt,
      c10::optional<torch::Tensor> a2av_unique_counts = c10::nullopt,
      c10::optional<torch::Tensor> dense_out = c10::nullopt,
      // EPIC §4.3 in-kernel swap (phase 0, sequential with the wire): this
      // rank's slot storage views (contiguous; fc2 optional) and the
      // same-node peer + monotone swap epoch. swap_peer = -1 disables.
      // Requires FLUX_A2AV_INKERNEL_SWAP=<scratch bytes> at ctor time.
      c10::optional<torch::Tensor> swap_fc1 = c10::nullopt,
      c10::optional<torch::Tensor> swap_fc2 = c10::nullopt,
      int64_t swap_peer = -1,
      int64_t swap_epoch = 0);
  // Drain the always-on swap-phase timing events (per-launch ms, launch
  // order); call once after the timed loop.
  std::vector<double> collect_swap_times();
  torch::Tensor forward_triton_aot(
      torch::Tensor inputs_shard,
      torch::Tensor weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      c10::optional<torch::Tensor> outputs_buf,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin,
      AllGatherOptionWithOptional ag_option);
  std::vector<torch::Tensor> forward_multiple_weights(
      torch::Tensor inputs_shard,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<std::vector<torch::Tensor>> bias,
      c10::optional<std::vector<torch::Tensor>> input_scale,
      c10::optional<std::vector<torch::Tensor>> weight_scale,
      c10::optional<std::vector<torch::Tensor>> output_scale,
      c10::optional<std::vector<torch::Tensor>> outputs_buf,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin,
      AllGatherOptionWithOptional ag_option,
      c10::optional<torch::Tensor> splits_per_source = c10::nullopt,
      c10::optional<torch::Tensor> a2av_unique_counts = c10::nullopt);
  std::vector<torch::Tensor> profiling(
      torch::Tensor inputs_shard,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<std::vector<torch::Tensor>> input_scale,
      c10::optional<std::vector<torch::Tensor>> weight_scale,
      c10::optional<std::vector<torch::Tensor>> output_scale,
      c10::optional<std::vector<torch::Tensor>> outputs_buf,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin,
      AllGatherOptionWithOptional ag_option,
      c10::intrusive_ptr<ProfilingContext> opt_ctx);

 private:
  class GemmGroupedV2AGScatterOpImpl;
  GemmGroupedV2AGScatterOpImpl *impl_ = nullptr;
};

}  // namespace bytedance::flux::ths_op
