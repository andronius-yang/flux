//===- gemm_grouped_v2_gather_rs.h -------------------------------- C++ ---===//
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

#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_op.h"

namespace bytedance::flux::ths_op {
class TopkReduceScatterOp {
 public:
  TopkReduceScatterOp(
      std::shared_ptr<Group> tp_group_,
      int max_m,
      int n_dim,
      int topk,
      at::ScalarType output_dtype,
      int ep_nexperts,
      int ep_world_size,
      std::vector<torch::Tensor> barriers,
      int n_split,
      bool do_all_reduce = false,
      bool use_read_mode = false,
      int nnodes = 1,
      bool a2av_hier = false,
      bool a2av_compress = false);
  ~TopkReduceScatterOp();
  void reset_buffer();
  // M-split waves (Slipstream v2, FLUX_A2AV_RS_MSPLIT): arm the combine's
  // per-schedule-step wave gates and destination order for the NEXT run().
  // node_order[i] = i-th dest node in production order (ring or size-sorted);
  // wave_of_node[i] = its cascade flag. Per-iteration state; n_waves == 0
  // disarms (legacy single-split gate + ring order).
  void set_msplit_waves(
      std::vector<int> const &wave_of_node,
      std::vector<int> const &node_order,
      int n_waves);
  // gen-8c epilogue-fused pack: the send panel the GEMM scatters into
  void *send_panel_ptr();
  int64_t send_panel_rows();
  torch::Tensor run(
      std::vector<torch::Tensor> gemm_outs,  // of group_size
      c10::optional<torch::Tensor> output_,
      int ep_start,
      int ep_nexperts,
      torch::Tensor splits,
      torch::Tensor routing_idx,
      c10::optional<std::vector<torch::Tensor>> output_vec_scales,
      int num_thread_blocks,
      intptr_t cp_stream,
      // a2av_hier mode only: the [W, nexperts] splits_per_source metadata (int32
      // CPU) and the mirror-layout pack/reduce gather indices (int32 CUDA).
      // Compress adds the transposed-U dedup counts and the wire/reduce CSRs.
      c10::optional<torch::Tensor> splits_per_source = c10::nullopt,
      c10::optional<torch::Tensor> pack_index = c10::nullopt,
      c10::optional<torch::Tensor> reduce_index = c10::nullopt,
      c10::optional<torch::Tensor> unique_counts = c10::nullopt,
      c10::optional<std::vector<torch::Tensor>> wire_csr = c10::nullopt,
      c10::optional<std::vector<torch::Tensor>> reduce_csr = c10::nullopt);
  // One-call combine/compress plan for the a2av_hier path (rule-5 plan
  // bracket; outputs feed run()'s pack/reduce/wire_csr/reduce_csr).
  std::vector<torch::Tensor> derive_combine_meta(
      torch::Tensor splits_gpu,
      torch::Tensor routing_idx,
      torch::Tensor splits_per_source,
      c10::optional<torch::Tensor> a2av_unique_counts = c10::nullopt);

 private:
  class TopkReduceScatterOpImpl;
  TopkReduceScatterOpImpl *impl_;
};

class GemmGroupedV2GatherRSOp {
 public:
  GemmGroupedV2GatherRSOp(
      std::shared_ptr<Group> tp_group_,
      int64_t total_num_experts,
      int64_t max_m,
      int64_t n_dim,
      int64_t topk,
      at::ScalarType output_dtype,
      int64_t tp_world_size,
      int64_t ep_world_size,
      int64_t max_input_groups,
      int64_t n_split,
      bool do_all_reduce = false,
      bool use_read_mode = false,
      int64_t nnodes = 1,
      bool a2av_hier = false,
      bool a2av_hier_compress = false);
  ~GemmGroupedV2GatherRSOp();
  torch::Tensor forward_gather_rs(
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync,
      // a2av_hier mode only: splits_per_source is REQUIRED ([W, nexperts] int32
      // CPU); the index tensors are optional precomputed routing-plan inputs (a
      // fused layer0+layer1 pipeline passes layer0's, paying the index math once).
      // Compress (dedup) plan: a2av_unique_counts is the transposed-U dedup count
      // matrix ([W, nnodes] int32 CPU, required whenever compress is on);
      // a2av_wire_csr = [wire_ptr, wire_copy] and a2av_reduce_csr =
      // [red_ptr, red_row] are the optional precomputed compress CSRs
      // (all-or-none as a pair; built in-forward when absent).
      c10::optional<torch::Tensor> splits_per_source = c10::nullopt,
      c10::optional<torch::Tensor> a2av_pack_index = c10::nullopt,
      c10::optional<torch::Tensor> a2av_reduce_index = c10::nullopt,
      c10::optional<torch::Tensor> a2av_unique_counts = c10::nullopt,
      c10::optional<std::vector<torch::Tensor>> a2av_wire_csr = c10::nullopt,
      c10::optional<std::vector<torch::Tensor>> a2av_reduce_csr = c10::nullopt);
  // Rule-5 per-iteration combine/compress index derivation as one host call
  // (2026-08-21): returns [pack_index, reduce_index] for a2av_hier, plus
  // [wire_ptr, wire_copy, red_ptr, red_row] when compress. splits_gpu int32
  // CUDA [nexperts], routing_idx int32 CUDA [m_full], splits_per_source int32
  // CPU [W, nexperts]; a2av_unique_counts int32 CPU [W, nnodes] (compress).
  std::vector<torch::Tensor> derive_combine_meta(
      torch::Tensor splits_gpu,
      torch::Tensor routing_idx,
      torch::Tensor splits_per_source,
      c10::optional<torch::Tensor> a2av_unique_counts = c10::nullopt);
  torch::Tensor forward_gather_rs_triton_aot(
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor splits,
      torch::Tensor routing_idx,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync);
  torch::Tensor profiling(
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync,
      c10::intrusive_ptr<ProfilingContext> opt_ctx);
  torch::Tensor forward_gather_rs_multiple(
      std::vector<torch::Tensor> inputs,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      c10::optional<std::vector<torch::Tensor>> bias,
      c10::optional<std::vector<torch::Tensor>> input_scale,
      c10::optional<std::vector<torch::Tensor>> weight_scale,
      c10::optional<std::vector<torch::Tensor>> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync);

 private:
  class GemmGroupedV2GatherRSOpImpl;
  GemmGroupedV2GatherRSOpImpl *impl_;
};

}  // namespace bytedance::flux::ths_op
