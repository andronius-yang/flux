//===- gemm_grouped_v2_ag_scatter.cc ------------------------------ C++ ---===//
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
#include "moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.h"
#include "coll/ths_op/all_gather_op.h"
#include "coll/ths_op/all_gather_types.h"
#include "cute/tensor.hpp"
#include "cutlass/util/device_memory.h"
#include "flux/args/moe_ag_scatter.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/flux.h"
#include "flux/utils.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/topo_utils.h"
#include "flux/ths_op/util.h"
#include "host/nvshmem_api.h"
#include "host/nvshmemx_api.h"
#include "moe_ag_scatter/sort_util.h"
#include "moe_ag_scatter/triton_util.h"
#include "moe_ag_scatter/workspace_util.h"
#include <nvshmemx.h>
#include <chrono>
#include <limits>
#include <optional>
#include <ATen/core/jit_type.h>
#include <ATen/core/List.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ops/empty.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/Logging.h>
#include <c10/util/Optional.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <utility>
#include <torch/cuda.h>
#include <torch/types.h>
#if defined(FLUX_WITH_TRITON_AOT)
#include "triton_aot_generated/flux_triton_aot.h"
#endif

namespace {
c10::optional<std::vector<torch::Tensor>>
as_optional_vec(c10::optional<torch::Tensor> &t) {
  if (t.has_value()) {
    return c10::optional<std::vector<torch::Tensor>>{{t.value()}};
  }
  return {};
}
}  // namespace

namespace bytedance::flux::ths_op {

/**
 * @return M_this_ep, M_this_ep_pad, gather_A_index, scatter_D_index, expert_idx, rank_start_idx,
 * rank_end_idx
 */
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
    intptr_t stream_) {
  cudaStream_t stream = (cudaStream_t)stream_;
  int nexperts = splits_gpu.numel();  // TODO(houqi.1993) no drop tokens?

  // should be M_this_ep, but never mind gather_index takes little memory
  torch::Tensor gather_index = empty_with_uninitialized_data(
      std::vector<int64_t>{ntokens * topk},
      torch::TensorOptions(torch::kCUDA).dtype(at::ScalarType::Int));
  torch::Tensor sorted_gather_index = empty_with_uninitialized_data(
      std::vector<int64_t>{ntokens * topk},
      torch::TensorOptions(torch::kCUDA).dtype(at::ScalarType::Int));
  torch::Tensor sorted_scatter_index = empty_with_uninitialized_data(
      std::vector<int64_t>{ntokens * topk},
      torch::TensorOptions(torch::kCUDA).dtype(at::ScalarType::Int));
  torch::Tensor M_this_ep_holder = empty_with_uninitialized_data(
      std::vector<int64_t>{1},
      torch::TensorOptions(torch::kCPU).dtype(at::ScalarType::Int).pinned_memory(true));
  torch::Tensor sorted_splits = empty_with_uninitialized_data(
      std::vector<int64_t>{ep_nexperts * world_size},
      torch::TensorOptions(torch::kCUDA).dtype(at::ScalarType::Int));
  torch::Tensor sorted_splits_cumsum = empty_with_uninitialized_data(
      std::vector<int64_t>{ep_nexperts * world_size},
      torch::TensorOptions(torch::kCUDA).dtype(at::ScalarType::Int));
  calc_gather_index_impl(
      nexperts,
      ntokens,
      topk,
      ep_start,
      ep_start + ep_nexperts,
      splits_gpu.data_ptr<int32_t>(),
      scatter_index.data_ptr<int32_t>(),
      gather_index.data_ptr<int32_t>(),
      M_this_ep_holder.data_ptr<int>(),
      stream);

  AGScatterSortOpArgumentsV2 args = {
      rank,
      world_size,
      ntokens,
      ep_nexperts,
      splits_gpu.data_ptr<int32_t>() + ep_start,
      gather_index.data_ptr<int32_t>(),
      sorted_splits.data_ptr<int32_t>(),
      sorted_splits_cumsum.data_ptr<int32_t>(),
      sorted_scatter_index.data_ptr<int32_t>(),
      sorted_gather_index.data_ptr<int32_t>(),
  };
  ag_scatter_sort_impl_v2(args, stream);

  int M_this_ep = scatter_index.numel();  // for EP=1, M_this_ep is always M_full
  if (ep_nexperts != nexperts) {
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
    M_this_ep = *M_this_ep_holder.data_ptr<int32_t>();
  }

  int max_problem_schedule_size = world_size * ep_nexperts * num_weights_group;
  torch::Tensor problem_schedules_gpu = empty_with_uninitialized_data(
      std::vector<int64_t>{(int64_t)(max_problem_schedule_size * sizeof(ProblemSchedV2))},
      torch::TensorOptions(torch::kByte).device(at::kCUDA));

  get_sorted_problem_schedule_cuda_v2(
      splits_gpu.data_ptr<int32_t>(),
      rank,
      world_size,
      sorted_splits_cumsum.data_ptr<int32_t>(),
      ep_start,
      ep_nexperts,
      tile_size_m,
      num_weights_group,
      (ProblemSchedV2 *)problem_schedules_gpu.data_ptr(),
      stream);

  // maybe larger than needed, but never mind the waste, just too little
  int m_pad = pad_to(M_this_ep, tile_size_m) + ep_nexperts * tile_size_m;
  int num_tiles_pad = m_pad / tile_size_m;

  auto option = torch::TensorOptions(torch::kInt32).device(torch::kCUDA);
  torch::Tensor m_pad_holder = empty_with_uninitialized_data(std::vector<int64_t>{1}, option);
  torch::Tensor gather_a_index =
      empty_with_uninitialized_data(std::vector<int64_t>{m_pad}, option);
  torch::Tensor scatter_d_index =
      empty_with_uninitialized_data(std::vector<int64_t>{m_pad}, option);
  torch::Tensor expert_index =
      empty_with_uninitialized_data(std::vector<int64_t>{num_tiles_pad}, option);
  torch::Tensor rank_start_index =
      empty_with_uninitialized_data(std::vector<int64_t>{num_tiles_pad}, option);
  torch::Tensor rank_end_index =
      empty_with_uninitialized_data(std::vector<int64_t>{num_tiles_pad}, option);

  get_moe_ag_scatter_args(
      splits_gpu.data_ptr<int>(),
      sorted_splits_cumsum.data_ptr<int>(),
      problem_schedules_gpu.data_ptr(),
      max_problem_schedule_size,
      sorted_gather_index.data_ptr<int>(),
      sorted_scatter_index.data_ptr<int>(),
      ep_start,
      ep_nexperts,
      world_size,
      M_this_ep,
      tile_size_m,
      m_pad_holder.data_ptr<int>(),
      gather_a_index.data_ptr<int32_t>(),
      scatter_d_index.data_ptr<int32_t>(),
      expert_index.data_ptr<int32_t>(),
      rank_start_index.data_ptr<int32_t>(),
      rank_end_index.data_ptr<int32_t>(),
      stream);
  return std::tuple(
      M_this_ep,
      m_pad_holder,
      gather_a_index,
      scatter_d_index,
      expert_index,
      rank_start_index,
      rank_end_index);
}

class GemmGroupedV2AGScatterOp::GemmGroupedV2AGScatterOpImpl {
 private:
  std::shared_ptr<Group> tp_group;
  const int rank;
  const int world_size;
  const int ep_size;
  const int nnodes;
  const DistEnv dist_env;
  const int ffn_tp_size;
  const int ep_rank;
  const int ffn_tp_rank;
  const int max_ntokens;
  const int N;
  const int hidden;
  const int nexperts;
  const int topk;
  at::ScalarType input_dtype;
  at::ScalarType output_dtype;
  const int32_t ep_nexperts;
  const int32_t ep_start;

  torch::Tensor workspace_buffer;

  c10::cuda::CUDAStream cp_stream;             // intra-node copies (V3: cp_stream_intra_node)
  c10::cuda::CUDAStream cp_stream_inter_node;  // remote-node fetches, used iff nnodes > 1
  cudaEvent_t ready_event;
  cudaEvent_t fetch_remote_event;
  cudaEvent_t all_gather_event;

  // nnodes == 1: intra-node all-gather over CUDA-IPC buffers (legacy path)
  std::optional<AllGatherOp> ag_op;
  // nnodes > 1: V3-style all-gather over NVSHMEM symmetric memory
  torch::Tensor input_buffer;
  // we use cutlass::DeviceAllocation instead of pytorch tensor here,
  // because if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set,
  // pytorch tensor .data_ptr() will return a virtual address which is invalid
  // for cuStreamWriteValue32_v2
  cutlass::DeviceAllocation<uint8_t> barrier_block;
  GroupBarrier group_barrier;

  // a2av dispatch mode (raw alltoallv): each (token, topk-slot) copy goes
  // directly producer -> expert-owner rank; wire bytes follow the routing.
  const bool a2av_dispatch_;
  // a2av_ring: puts follow the reverse hierarchical ring (mirror of the dense
  // stage order), and the GEMM keeps the dense static problem schedule.
  const bool a2av_ring_;
  uint64_t run_id_ = 0;             // epoch value carried by the NVSHMEM signals
  int64_t max_recv_ntokens_ = 0;    // rows of the symmetric recv buffer
  torch::Tensor a2av_send_buffer;   // symmetric [tokens_per_rank_max * topk, hidden]
  torch::Tensor a2av_recv_buffer;   // symmetric [max_recv_ntokens_, hidden]
  torch::Tensor a2av_signal_buffer; // symmetric uint64[world_size], never memset
  // one-shot dispatch scratch: allocated once (setup), contents rebuilt every
  // iteration — routing is never cached across forwards
  torch::Tensor a2av_arange_i64_;   // [n_copies_max] iota, routing-independent
  torch::Tensor a2av_chunks_cpu_;   // pinned int32 [W * W] chunk-count matrix
  torch::Tensor a2av_e_all_;        // i64 [n_copies_max] fused-kernel outputs...
  torch::Tensor a2av_s_all_buf_;    // i64 [n_copies_max]
  torch::Tensor a2av_flat_dst_;     // i64 [n_copies_max]
  torch::Tensor a2av_not_mine_;     // bool [n_copies_max]
  torch::Tensor a2av_expert_base_;  // i64 [nexperts]
  torch::Tensor a2av_chunks_gpu_;   // i32 [W * W]
  torch::Tensor a2av_pack_key_;     // i64 [copies_per_rank_max]
  // splits_per_source (metadata) path: host-computed group tables staged in one
  // pinned buffer and uploaded with a single H2D into a device arena per
  // iteration. Layout: cumA/offA/offR_of_A i64 [G], expert_base i64 [nexperts],
  // sorted_splits_cumsum i32 [G], with G = ep_nexperts * world_size.
  torch::Tensor a2av_meta_pinned_;  // pinned bytes
  torch::Tensor a2av_meta_dev_;     // device bytes, same layout
  cudaEvent_t counts_event_ = nullptr;  // gates the put loop on the 1 KB counts D2H
  // FLUX_A2AV_TIMING=1 diagnostics: per-forward segment boundaries on the main stream
  static constexpr int kNumTimingEvents = 6;
  cudaEvent_t timing_events_[kNumTimingEvents] = {};
  static constexpr int kNumStage2Events = 11;
  cudaEvent_t stage2_events_[kNumStage2Events] = {};

 private:
  c10::cuda::CUDAStream
  create_cp_stream() const {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, CU_STREAM_NON_BLOCKING));
    return at::cuda::getStreamFromExternal(stream, at::cuda::current_device());
  }

  void
  _ensure_topo_initialized() {
    if (!topo_utils::is_topo_initialized()) {
      topo_utils::initialize_topo(this->tp_group.get());
    }
  }

  AllGatherOption
  materialize(const AllGatherOptionWithOptional opt, bool with_input_scale) {
    return AllGatherOption{
        .input_buffer_copied = opt.input_buffer_copied.value_or(false),
        .use_cuda_core_local = opt.use_cuda_core_local.value_or(with_input_scale),
        .use_cuda_core_ag = opt.use_cuda_core_ag.value_or(with_input_scale),
        .fuse_sync = opt.fuse_sync.value_or(with_input_scale),
        .use_read = opt.use_read.value_or(false),
        .mode = opt.mode.value_or(get_default_ag_ring_mode()),
    };
  }

 public:
  GemmGroupedV2AGScatterOpImpl(
      std::shared_ptr<Group> tp_group,
      int ep_size,
      int nnodes,
      int max_ntokens,
      int ffn_hidden,  // before TP shard
      int hidden,
      int nexperts,
      int topk,
      at::ScalarType input_dtype,
      at::ScalarType output_dtype,
      bool a2av_dispatch = false,
      bool a2av_ring = false)
      : tp_group(tp_group),
        world_size(tp_group->get_size()),
        ep_size(ep_size),
        nnodes(nnodes),
        dist_env(tp_group->get_rank(), tp_group->get_size(), nnodes),
        ffn_tp_size(world_size / ep_size),
        rank(tp_group->get_rank()),
        ffn_tp_rank(rank % ffn_tp_size),
        ep_rank(rank / ffn_tp_size),
        max_ntokens(max_ntokens),
        N(ffn_hidden / ffn_tp_size),
        hidden(hidden),
        nexperts(nexperts),
        topk(topk),
        input_dtype(input_dtype),
        output_dtype(output_dtype),
        ep_nexperts(nexperts / ep_size),
        ep_start(this->ep_nexperts * ep_rank),
        cp_stream(create_cp_stream()),
        cp_stream_inter_node(create_cp_stream()),
        a2av_dispatch_(a2av_dispatch),
        a2av_ring_(a2av_ring),
        // ring_mode barriers are CUDA-IPC based and intra-node only; multi-node
        // must take the NVSHMEM barrier (ring_mode = false)
        group_barrier(this->tp_group, nnodes == 1 && this->tp_group->get_size() > 8) {
    _ensure_topo_initialized();
    CHECK_DIV(nexperts, ep_size);
    CHECK_DIV(ffn_hidden, ffn_tp_size);
    FLUX_CHECK_GE(nnodes, 1);
    CHECK_DIV(world_size, nnodes);
    FLUX_CHECK(!a2av_ring || a2av_dispatch) << "a2av_ring requires a2av_dispatch";
    if (a2av_dispatch) {
      FLUX_CHECK_EQ(this->ffn_tp_size, 1) << "a2av dispatch requires ep_size == world_size";
      FLUX_CHECK(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE) == dist_env.local_rank);
      int64_t tokens_per_rank_max = (max_ntokens + world_size - 1) / world_size;
      // default recv capacity: 2x the balanced per-rank load (capped at the total);
      // very skewed routings need FLUX_A2AV_MAX_RECV_NTOKENS
      this->max_recv_ntokens_ = get_int_from_env(
          "FLUX_A2AV_MAX_RECV_NTOKENS",
          (int)std::min<int64_t>((int64_t)max_ntokens * topk, tokens_per_rank_max * topk * 2));
      this->a2av_send_buffer =
          nvshmem_create_tensor({tokens_per_rank_max * topk, hidden}, input_dtype);
      this->a2av_recv_buffer =
          nvshmem_create_tensor({this->max_recv_ntokens_, hidden}, input_dtype);
      this->a2av_signal_buffer =
          nvshmem_create_tensor({world_size}, at::ScalarType::Long, /*init_zero=*/true);
      const int64_t n_copies_max = tokens_per_rank_max * (int64_t)topk * world_size;
      auto opt_cuda_i64 = torch::TensorOptions(torch::kCUDA).dtype(torch::kLong);
      this->a2av_arange_i64_ = torch::arange(n_copies_max, opt_cuda_i64);
      this->a2av_chunks_cpu_ = torch::empty(
          {(int64_t)world_size * world_size},
          torch::TensorOptions(torch::kCPU).dtype(torch::kInt).pinned_memory(true));
      this->a2av_e_all_ = torch::empty({n_copies_max}, opt_cuda_i64);
      this->a2av_s_all_buf_ = torch::empty({n_copies_max}, opt_cuda_i64);
      this->a2av_flat_dst_ = torch::empty({n_copies_max}, opt_cuda_i64);
      this->a2av_not_mine_ = torch::empty(
          {n_copies_max}, torch::TensorOptions(torch::kCUDA).dtype(torch::kBool));
      this->a2av_expert_base_ = torch::empty({nexperts}, opt_cuda_i64);
      this->a2av_chunks_gpu_ = torch::empty(
          {(int64_t)world_size * world_size},
          torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));
      this->a2av_pack_key_ = torch::empty({tokens_per_rank_max * (int64_t)topk}, opt_cuda_i64);
      const int64_t meta_groups = (int64_t)this->ep_nexperts * world_size;
      const int64_t meta_bytes =
          3 * meta_groups * sizeof(int64_t) + nexperts * sizeof(int64_t) +
          meta_groups * sizeof(int32_t);
      this->a2av_meta_pinned_ = torch::empty(
          {meta_bytes},
          torch::TensorOptions(torch::kCPU).dtype(torch::kByte).pinned_memory(true));
      this->a2av_meta_dev_ = torch::empty(
          {meta_bytes}, torch::TensorOptions(torch::kCUDA).dtype(torch::kByte));
      if (rank == 0) {
        double sym_mb = (this->a2av_send_buffer.nbytes() + this->a2av_recv_buffer.nbytes() +
                         this->a2av_signal_buffer.nbytes()) /
                        1024.0 / 1024.0;
        fprintf(
            stderr,
            "[flux a2av] recv rows %ld send rows %ld -> %.0f MiB symmetric heap per rank\n",
            (long)this->max_recv_ntokens_,
            (long)(tokens_per_rank_max * topk),
            sym_mb);
      }
    } else if (nnodes == 1) {
      ag_op.emplace(this->tp_group, 1, max_ntokens, hidden, input_dtype);
    } else {
      FLUX_CHECK(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE) == dist_env.local_rank);
      this->input_buffer = nvshmem_create_tensor({max_ntokens, hidden}, input_dtype);
      this->barrier_block.reset(pad_to(world_size * (int64_t)sizeof(int), (int64_t)128));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&this->ready_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->fetch_remote_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->all_gather_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->counts_event_, cudaEventDisableTiming));
    for (int i = 0; i < kNumTimingEvents; i++) {
      CUDA_CHECK(cudaEventCreate(&this->timing_events_[i]));  // timing-capable
    }
    for (int i = 0; i < kNumStage2Events; i++) {
      CUDA_CHECK(cudaEventCreate(&this->stage2_events_[i]));
    }
  }

  ~GemmGroupedV2AGScatterOpImpl() {
    for (int i = 0; i < kNumTimingEvents; i++) {
      CUDA_CHECK(cudaEventDestroy(this->timing_events_[i]));
    }
    for (int i = 0; i < kNumStage2Events; i++) {
      CUDA_CHECK(cudaEventDestroy(this->stage2_events_[i]));
    }
    CUDA_CHECK(cudaEventDestroy(this->counts_event_));
    CUDA_CHECK(cudaEventDestroy(this->all_gather_event));
    CUDA_CHECK(cudaEventDestroy(this->fetch_remote_event));
    CUDA_CHECK(cudaEventDestroy(this->ready_event));
    CUDA_CHECK(cudaStreamDestroy(this->cp_stream));
    CUDA_CHECK(cudaStreamDestroy(this->cp_stream_inter_node));
  }

 protected:
  auto
  get_gemm_meta(bool fast_accum) const {
    auto arch = get_arch();
    auto sm_core = get_sm_core();
    auto gemm_layout = _RCR{};  // TODO(houqi.1993) only RCR supported
    auto input_dtype = from_torch_dtype(this->input_dtype);
    auto output_dtype = from_torch_dtype(this->output_dtype);
    auto dt_conf = make_gemm_dtype_config(input_dtype, input_dtype, output_dtype, output_dtype);
    auto v2_meta = make_gemm_v2_meta(fast_accum && dt_conf.is_input_fp8());
    auto meta = make_gemm_meta(
        dt_conf, arch, sm_core, _AGScatter{}, gemm_layout, _GemmGroupedV2{}, v2_meta);
    return meta;
  }

  auto
  get_rt_conf() const {
    return make_runtime_config(512, this->N, this->hidden);
  }

  // ported from GemmGroupedV3AGScatterOpImpl::all_gather_all2all: all-gather the
  // input shards of all ranks into the NVSHMEM symmetric `input_buffer`, writing a
  // per-source-rank ready flag (value 1) into `barrier_block` as each shard lands.
  // used only when nnodes > 1.
  void
  all_gather_all2all(torch::Tensor const &inputs_shard) {
    using namespace cute;

    int ntokens_shard = inputs_shard.size(0);
    Tensor full_input = make_tensor(
        static_cast<uint8_t *>(input_buffer.data_ptr()),
        make_shape(
            make_shape(c10::elementSize(this->input_dtype), this->hidden),
            ntokens_shard,
            dist_env.world_size));

    // fetch data from other ranks and write the flag to mark the data ready
    // outer loop iterating the node_idx, processing the current node first then others
    // inner loop iterating the local_rank, use all2all for communication
    for (int node_idx = dist_env.node_idx, i = 0; i < dist_env.nnodes;
         ++i, node_idx = (node_idx + 1) % dist_env.nnodes) {
      if (node_idx == dist_env.node_idx) {
        auto main_stream = c10::cuda::getCurrentCUDAStream();
        auto shard_input = full_input(_, _, dist_env.rank);
        CUDA_CHECK(cudaMemcpyAsync(
            shard_input.data(),
            inputs_shard.data_ptr(),
            shard_input.size(),
            cudaMemcpyDeviceToDevice,
            main_stream));
        nvshmemx_barrier_all_on_stream(main_stream);
        CUDA_CHECK(cudaEventRecord(this->ready_event, main_stream));
        CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream, this->ready_event));
      } else {
        if (i == 1) {
          // the first remote fetch wait for data ready
          CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream_inter_node, this->ready_event));
        }
        int src_rank = dist_env.local_rank_to_global_rank(dist_env.local_rank, node_idx);
        auto shard_input = full_input(_, _, src_rank);
        nvshmemx_getmem_on_stream(
            shard_input.data(),
            shard_input.data(),
            shard_input.size(),
            src_rank,
            this->cp_stream_inter_node);
        nvshmemx_barrier_on_stream(NVSHMEMX_TEAM_NODE, this->cp_stream_inter_node);
        CUDA_CHECK(cudaEventRecord(this->fetch_remote_event, this->cp_stream_inter_node));
        CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream, this->fetch_remote_event));
      }
      for (int local_rank = dist_env.local_rank, j = 0; j < dist_env.local_world_size;
           ++j, local_rank = (local_rank + 1) % dist_env.local_world_size) {
        int src_rank = dist_env.local_rank_to_global_rank(local_rank, node_idx);
        int local_rank_global = dist_env.local_rank_to_global_rank(local_rank);
        if (local_rank != dist_env.local_rank) {
          auto shard_input = full_input(_, _, src_rank);
          nvshmemx_getmem_on_stream(
              shard_input.data(),
              shard_input.data(),
              shard_input.size(),
              local_rank_global,
              this->cp_stream);
        }
        CU_CHECK(CUStreamWriteValue(
            this->cp_stream,
            (CUdeviceptr)(ptr_offset(barrier_block.get(), src_rank * sizeof(int))),
            1,
            CU_STREAM_WRITE_VALUE_DEFAULT));
      }
    }
    CUDA_CHECK(cudaEventRecord(this->all_gather_event, this->cp_stream));
  }

  struct A2AVDispatchState {
    // index tensors are allocated at full n_copies size (fixed shapes keep the
    // build sync-free); only the first M_this_ep rows are valid and the GEMM
    // reads exactly that many via data_ptr + M_this_ep
    torch::Tensor sorted_gather_index;   // int32: sorted-A row -> recv-buffer row
    torch::Tensor sorted_scatter_index;  // int32: sorted-D row -> per-expert D row
    torch::Tensor sorted_splits_cumsum;  // int32 [ep_nexperts, world_size]
    int M_this_ep = 0;
  };

  // Raw alltoallv dispatch: pack my (token, topk-slot) copies destination-major
  // into the symmetric send buffer, then one putmem_signal per destination rank
  // into its recv buffer. The recv layout is (source, expert, dst_row)-ordered so
  // each src->dst message is a single contiguous put; the GEMM reads rows through
  // sorted_gather_index, so no unpack kernel is needed.
  // NOTE: a real system would exchange per-(source, expert) counts first; in this
  // harness every rank holds the identical global scatter_index, so all offsets
  // are computed locally.
  A2AVDispatchState
  a2av_dispatch(
      torch::Tensor const &inputs_shard,
      torch::Tensor const &splits_gpu,
      torch::Tensor const &scatter_index,
      const int32_t *cnt_host,  // [W, nexperts] splits_per_source, or nullptr
      cudaStream_t stream) {
    const int W = this->world_size;
    const int64_t E = this->ep_nexperts;
    const int tokens_per_rank = inputs_shard.size(0);
    const int64_t copies_per_rank = (int64_t)tokens_per_rank * topk;
    const int64_t n_copies = copies_per_rank * W;
    const bool use_meta = cnt_host != nullptr;
    this->run_id_ += 1;
    static const bool kTiming = get_int_from_env("FLUX_A2AV_TIMING", 0) != 0;
    if (kTiming) {
      CUDA_CHECK(cudaEventRecord(this->timing_events_[0], stream));
    }
    auto host_now = []() { return std::chrono::steady_clock::now(); };
    auto host_us = [](auto a, auto b) {
      return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    auto h0 = host_now();

    // ---- metadata path (splits_per_source provided): everything the wire and
    // the schedule need is derived on the HOST from cnt[s][e] before any device
    // work — the counts kernel/histogram, the 1 KB D2H, and the counts-event
    // wait all disappear from the timed path. Group tables for stage 2 are
    // staged into pinned memory and uploaded with one async H2D.
    const int64_t nexG = E * W;  // number of (expert_loc, source) groups
    int64_t M_this_ep = 0;
    std::vector<int64_t> chunks64((size_t)W * W, 0);
    torch::Tensor cumA_dev, offA_dev, offR_of_A_dev, expert_base_dev, ssc_dev;
    if (use_meta) {
      // guard pinned-staging reuse: the previous iteration's H2D must be done
      // (counts_event_ doubles as the H2D-completion event on this path; it is
      // long finished by now, so this returns immediately)
      CUDA_CHECK(cudaEventSynchronize(this->counts_event_));
      const int64_t nex = this->nexperts;
      auto cnt_at = [&](int s, int64_t e) -> int64_t { return cnt_host[s * nex + e]; };
      for (int s = 0; s < W; s++) {
        for (int d = 0; d < W; d++) {
          int64_t acc = 0;
          for (int64_t e = (int64_t)d * E; e < (int64_t)(d + 1) * E; e++) {
            acc += cnt_at(s, e);
          }
          chunks64[s * W + d] = acc;
        }
      }
      for (int s = 0; s < W; s++) {
        M_this_ep += chunks64[s * W + rank];
      }
      FLUX_CHECK_LE(M_this_ep, this->max_recv_ntokens_)
          << "a2av recv buffer overflow; raise FLUX_A2AV_MAX_RECV_NTOKENS";
      // staging layout: cumA/offA/offR_of_A i64 [nexG], expert_base i64
      // [nexperts], sorted_splits_cumsum i32 [nexG] (row-major [E, W])
      char *stage = reinterpret_cast<char *>(this->a2av_meta_pinned_.data_ptr());
      int64_t *cumA_h = reinterpret_cast<int64_t *>(stage);
      int64_t *offA_h = cumA_h + nexG;
      int64_t *offR_of_A_h = offA_h + nexG;
      int64_t *expert_base_h = offR_of_A_h + nexG;
      int32_t *ssc_h = reinterpret_cast<int32_t *>(expert_base_h + nex);
      // A-order groups g = e_loc*W + s (size = cnt[s][ep_start + e_loc])
      int64_t acc = 0;
      for (int64_t e_loc = 0; e_loc < E; e_loc++) {
        for (int s = 0; s < W; s++) {
          int64_t g = e_loc * W + s;
          offA_h[g] = acc;
          acc += cnt_at(s, ep_start + e_loc);
          cumA_h[g] = acc;
        }
      }
      // recv-order groups h = s*E + e_loc; offR_of_A maps A-group -> recv offset
      std::vector<int64_t> offR((size_t)nexG, 0);
      acc = 0;
      for (int s = 0; s < W; s++) {
        for (int64_t e_loc = 0; e_loc < E; e_loc++) {
          offR[s * E + e_loc] = acc;
          acc += cnt_at(s, ep_start + e_loc);
        }
      }
      for (int64_t e_loc = 0; e_loc < E; e_loc++) {
        for (int s = 0; s < W; s++) {
          offR_of_A_h[e_loc * W + s] = offR[s * E + e_loc];
        }
      }
      // sorted_splits_cumsum [E, W]: inclusive cumsum over sources per expert
      for (int64_t e_loc = 0; e_loc < E; e_loc++) {
        int32_t c = 0;
        for (int s = 0; s < W; s++) {
          c += (int32_t)cnt_at(s, ep_start + e_loc);
          ssc_h[e_loc * W + s] = c;
        }
      }
      // expert_base[e] = prefix sum of column sums (== prefix of splits)
      int64_t base = 0;
      for (int64_t e = 0; e < nex; e++) {
        expert_base_h[e] = base;
        for (int s = 0; s < W; s++) {
          base += cnt_at(s, e);
        }
      }
      CUDA_CHECK(cudaMemcpyAsync(
          this->a2av_meta_dev_.data_ptr(),
          stage,
          this->a2av_meta_pinned_.nbytes(),
          cudaMemcpyHostToDevice,
          stream));
      CUDA_CHECK(cudaEventRecord(this->counts_event_, stream));
      auto opt_dev_i64 = torch::TensorOptions(torch::kCUDA).dtype(torch::kLong);
      auto opt_dev_i32 = torch::TensorOptions(torch::kCUDA).dtype(torch::kInt);
      char *dev = reinterpret_cast<char *>(this->a2av_meta_dev_.data_ptr());
      cumA_dev = torch::from_blob(dev, {nexG}, opt_dev_i64);
      offA_dev = torch::from_blob(dev + nexG * 8, {nexG}, opt_dev_i64);
      offR_of_A_dev = torch::from_blob(dev + 2 * nexG * 8, {nexG}, opt_dev_i64);
      expert_base_dev = torch::from_blob(dev + 3 * nexG * 8, {nex}, opt_dev_i64);
      ssc_dev = torch::from_blob(dev + 3 * nexG * 8 + nex * 8, {E, (int64_t)W}, opt_dev_i32);
    }

    auto opt_i64 = torch::TensorOptions(torch::kCUDA)
                       .dtype(torch::kLong)
                       .device_index(at::cuda::current_device());
    constexpr int64_t kShift = int64_t(1) << 32;
    auto iota = this->a2av_arange_i64_.narrow(0, 0, n_copies);

    // ---- stage 1 (pre-wire, minimal): one fused kernel decodes every copy,
    // fills the [W,W] chunk counts and all stage-2 inputs, and emits the pack
    // keys; then the tiny D2H and the producer pack. No host sync of any kind
    // in this stage — CUDA bincount/nonzero are banned (both hide a full
    // stream drain).
    auto e_all = this->a2av_e_all_.narrow(0, 0, n_copies);
    auto s_all = this->a2av_s_all_buf_.narrow(0, 0, n_copies);
    auto flat_dst = this->a2av_flat_dst_.narrow(0, 0, n_copies);
    auto not_mine = this->a2av_not_mine_.narrow(0, 0, n_copies);
    auto chunks_full = this->a2av_chunks_gpu_;
    if (!use_meta) {
      CUDA_CHECK(cudaMemsetAsync(chunks_full.data_ptr(), 0, chunks_full.nbytes(), stream));
    }
    a2av_stage1_impl(
        A2AVStage1Arguments{
            .scatter_index = scatter_index.data_ptr<int32_t>(),
            .splits = splits_gpu.data_ptr<int32_t>(),
            .nexperts = this->nexperts,
            .ep_nexperts = (int)E,
            .world_size = W,
            .rank = rank,
            .copies_per_rank = copies_per_rank,
            .n_copies = n_copies,
            .e_all = e_all.data_ptr<int64_t>(),
            .s_all = s_all.data_ptr<int64_t>(),
            .flat_dst = flat_dst.data_ptr<int64_t>(),
            .not_mine = not_mine.data_ptr<bool>(),
            // metadata path: counts + expert_base come from the host tables
            .expert_base = use_meta ? nullptr : this->a2av_expert_base_.data_ptr<int64_t>(),
            .chunks = use_meta ? nullptr : chunks_full.data_ptr<int32_t>(),
            .pack_key = this->a2av_pack_key_.data_ptr<int64_t>()},
        stream);
    if (!use_meta) {
      this->a2av_chunks_cpu_.copy_(chunks_full, /*non_blocking=*/true);  // 1 KB into pinned
      CUDA_CHECK(cudaEventRecord(this->counts_event_, stream));
    }

    // producer pack: my copies only, destination-major. pack_key = e * cpr +
    // local_p, so ascending order is (destination, expert, copy index) — the
    // copy-index tie-break is mirrored by the consumer keys in stage 2.
    auto perm_s = this->a2av_pack_key_.narrow(0, 0, copies_per_rank).argsort();
    auto send_gather_index = perm_s.div((int64_t)topk, "floor");
    auto send_view = this->a2av_send_buffer.narrow(0, 0, copies_per_rank);
    at::index_select_out(send_view, inputs_shard, 0, send_gather_index);
    CUDA_CHECK(cudaEventRecord(this->ready_event, stream));
    if (kTiming) {
      CUDA_CHECK(cudaEventRecord(this->timing_events_[1], stream));
    }

    // ---- stage 2 (overlaps the wire): consumer indices, enqueued BEFORE the
    // host waits on the counts event, so these kernels run while the puts fly.
    // Everything is fixed-shape at n_copies: owned copies sort first (their keys
    // are unique — scatter_index is a permutation), rows past M_this_ep are
    // in-bounds garbage the GEMM never reads (it consumes data_ptr + M_this_ep).
    torch::Tensor sorted_gather_index, sorted_scatter_index, sorted_splits_cumsum;
    auto build_stage2 = [&]() {
      auto mark = [&](int i) {
        if (kTiming) {
          CUDA_CHECK(cudaEventRecord(this->stage2_events_[i], stream));
        }
      };
      mark(0);
      auto e_loc = e_all.sub((int64_t)ep_start);  // negative for foreign copies (masked below)
      mark(1);
      constexpr int64_t kMax64 = std::numeric_limits<int64_t>::max();
      // sorted mat-A order: (expert, source, copy); recv order: (source, expert,
      // copy). The copy-index (iota) tie-break matches the producer pack_key, so
      // every s->d message's interior order equals its recv region's.
      auto key_r = ((s_all * E + e_loc) * kShift + iota).masked_fill_(not_mine, kMax64);
      mark(2);
      if (use_meta) {
        // one sort + arithmetic identity: within any (s, e) group both the
        // A-order and the recv order sort by the same copy index, so the A->recv
        // map is fully determined by the host-derived group offset tables.
        auto order_r = key_r.argsort();
        mark(3);
        auto g = torch::searchsorted(cumA_dev, iota, /*out_int32=*/false, /*right=*/true)
                     .clamp_max_((int64_t)nexG - 1);
        // tail rows (>= M_this_ep) are unread garbage but must stay in-bounds
        // for the index_selects below, hence the clamp
        auto sgi64 = (offR_of_A_dev.index_select(0, g) + iota - offA_dev.index_select(0, g))
                         .clamp_(0, n_copies - 1);
        mark(4);
        sorted_gather_index = sgi64.to(torch::kInt);
        auto scatter_val = flat_dst - expert_base_dev.index_select(0, e_all);
        // A-pos i -> recv-pos sgi64[i] -> copy order_r[sgi64[i]] (== old perm_a)
        sorted_scatter_index =
            scatter_val.index_select(0, order_r).index_select(0, sgi64).to(torch::kInt);
        mark(5);
        sorted_splits_cumsum = ssc_dev;  // uploaded, exact [E, W] semantics
        mark(6);
        mark(7);
        mark(8);
        mark(9);
        mark(10);
        static const bool kCheckIdentity =
            get_int_from_env("FLUX_A2AV_CHECK_IDENTITY", 0) != 0;
        if (kCheckIdentity) {
          auto key_a = ((e_loc * W + s_all) * kShift + iota).masked_fill_(not_mine, kMax64);
          auto perm_a = key_a.argsort();
          auto recv_pos = torch::empty({n_copies}, opt_i64).scatter_(0, order_r, iota);
          auto ref_gather = recv_pos.index_select(0, perm_a).to(torch::kInt);
          auto ref_scatter = scatter_val.index_select(0, perm_a).to(torch::kInt);
          FLUX_CHECK(torch::equal(
              sorted_gather_index.narrow(0, 0, M_this_ep), ref_gather.narrow(0, 0, M_this_ep)))
              << "a2av metadata identity mismatch (gather)";
          FLUX_CHECK(torch::equal(
              sorted_scatter_index.narrow(0, 0, M_this_ep), ref_scatter.narrow(0, 0, M_this_ep)))
              << "a2av metadata identity mismatch (scatter)";
        }
        return;
      }
      auto key_a = ((e_loc * W + s_all) * kShift + iota).masked_fill_(not_mine, kMax64);
      auto perm_a = key_a.argsort();
      mark(3);
      mark(4);
      // sort (values + indices) instead of argsort: the sorted keys also yield the
      // per-(source, expert) group boundaries below
      auto sorted_r = key_r.sort(0);
      auto key_r_sorted = std::get<0>(sorted_r);
      auto order_r = std::get<1>(sorted_r);
      mark(5);
      // inverse permutation by scatter-of-iota (one sort cheaper than argsort().argsort())
      auto recv_pos = torch::empty({n_copies}, opt_i64).scatter_(0, order_r, iota);
      mark(6);
      sorted_gather_index = recv_pos.index_select(0, perm_a).to(torch::kInt);
      mark(7);
      auto scatter_val = flat_dst - this->a2av_expert_base_.index_select(0, e_all);
      sorted_scatter_index = scatter_val.index_select(0, perm_a).to(torch::kInt);
      mark(8);
      // per-(source, expert) counts WITHOUT atomics: W*E binary searches for the
      // group ends in the sorted recv keys (foreign keys sort past every
      // threshold). The scatter_add alternative floods one address with ~n_copies
      // same-bin atomicAdds and cost ~14 ms in-pipeline at 131k copies.
      auto thresholds = (this->a2av_arange_i64_.narrow(0, 0, (int64_t)W * E) + 1) * kShift;
      auto ends = torch::searchsorted(key_r_sorted, thresholds, /*out_int32=*/false, /*right=*/false);
      mark(9);
      auto cnt_flat = at::diff(ends, 1, 0, torch::zeros({1}, opt_i64));
      sorted_splits_cumsum = cnt_flat.view({W, E}).cumsum(0).t().contiguous().to(torch::kInt);
      mark(10);
    };
    // perf-diagnostic knobs (default off): reorder stage 2 after the put issuance /
    // drain the stream before issuing puts, to bisect overlap effects under
    // CUDA_DEVICE_MAX_CONNECTIONS=1
    static const bool kStage2AfterPuts =
        get_int_from_env("FLUX_A2AV_STAGE2_AFTER_PUTS", 0) != 0;
    static const bool kSyncBeforePuts =
        get_int_from_env("FLUX_A2AV_SYNC_BEFORE_PUTS", 0) != 0;
    auto h1 = host_now();
    if (!kStage2AfterPuts) {
      build_stage2();
    }
    auto h2 = host_now();
    if (kTiming) {
      CUDA_CHECK(cudaEventRecord(this->timing_events_[2], stream));
    }

    // ---- host: in the derive path, wait only for stage 1's counts D2H (the
    // event precedes the stage-2 enqueues in stream order, so none of the sorts
    // gate the wire). In the metadata path everything is already known.
    if (!use_meta) {
      CUDA_CHECK(cudaEventSynchronize(this->counts_event_));
    }
    if (kSyncBeforePuts) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    auto h3 = host_now();
    if (kTiming) {
      fprintf(
          stderr,
          "[a2av-host] rank %d enq_stage1 %ld us enq_stage2 %ld us counts_wait %ld us\n",
          rank,
          (long)host_us(h0, h1),
          (long)host_us(h1, h2),
          (long)host_us(h2, h3));
    }
    if (!use_meta) {
      const int32_t *chunks_host = this->a2av_chunks_cpu_.data_ptr<int32_t>();
      for (int i = 0; i < W * W; i++) {
        chunks64[i] = chunks_host[i];
      }
      for (int s = 0; s < W; s++) {
        M_this_ep += chunks64[s * W + rank];
      }
      FLUX_CHECK_LE(M_this_ep, this->max_recv_ntokens_)
          << "a2av recv buffer overflow; raise FLUX_A2AV_MAX_RECV_NTOKENS";
    }
    auto chunk_at = [&](int s, int d) -> int64_t { return chunks64[s * W + d]; };

    CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream, this->ready_event));
    CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream_inter_node, this->ready_event));

    const int64_t row_bytes = (int64_t)hidden * c10::elementSize(input_dtype);
    uint64_t *signal_base = reinterpret_cast<uint64_t *>(this->a2av_signal_buffer.data_ptr());
    char *send_base = reinterpret_cast<char *>(this->a2av_send_buffer.data_ptr());
    char *recv_base = reinterpret_cast<char *>(this->a2av_recv_buffer.data_ptr());
    // 16x16-scale prefix sums on the host: my send-segment offsets and, per
    // destination, my exclusive offset RO[rank][d] into its recv region
    std::vector<int64_t> send_off(W, 0), recv_off(W, 0);
    for (int d = 0, acc = 0; d < W; d++) {
      send_off[d] = acc;
      acc += chunk_at(rank, d);
    }
    for (int d = 0; d < W; d++) {
      int64_t acc = 0;
      for (int s = 0; s < rank; s++) {
        acc += chunk_at(s, d);
      }
      recv_off[d] = acc;
    }
    // self-delivery on cp_stream: the send segment's interior order equals the
    // recv region's interior order, so one contiguous local copy suffices
    if (chunk_at(rank, rank) > 0) {
      CUDA_CHECK(cudaMemcpyAsync(
          recv_base + recv_off[rank] * row_bytes,
          send_base + send_off[rank] * row_bytes,
          chunk_at(rank, rank) * row_bytes,
          cudaMemcpyDeviceToDevice,
          this->cp_stream));
    }
    nvshmemx_signal_op_on_stream(
        signal_base + rank, this->run_id_, NVSHMEM_SIGNAL_SET, rank, this->cp_stream);
    // zero-payload destinations still get the signal (the GEMM waits on every source)
    auto issue_put = [&](int d, cudaStream_t put_stream) {
      int64_t bytes = chunk_at(rank, d) * row_bytes;
      if (bytes > 0) {
        nvshmemx_putmem_signal_nbi_on_stream(
            recv_base + recv_off[d] * row_bytes,
            send_base + send_off[d] * row_bytes,
            bytes,
            signal_base + rank,
            this->run_id_,
            NVSHMEM_SIGNAL_SET,
            d,
            put_stream);
      } else {
        nvshmemx_signal_op_on_stream(
            signal_base + rank, this->run_id_, NVSHMEM_SIGNAL_SET, d, put_stream);
      }
    };
    if (!a2av_ring_) {
      // remote puts, ring order starting at rank+1 to avoid incast
      for (int i = 1; i < W; i++) {
        issue_put((rank + i) % W, this->cp_stream_inter_node);
      }
    } else {
      // scheduled mode: reverse hierarchical ring — the mirror of the receivers'
      // stage order (shift_rank_to_order), so destination d sees our chunk at
      // exactly the stage the dense problem schedule expects source `rank` at.
      // Each slot k is a bijection source->destination, so no incast. Intra-node
      // puts ride cp_stream; inter-node puts start concurrently on
      // cp_stream_inter_node (their tiles are scheduled last anyway).
      const int L = dist_env.local_world_size;
      const int NN = dist_env.nnodes;
      for (int k = 1; k < W; k++) {
        int dn = k / L, dl = k % L;
        int d = dist_env.local_rank_to_global_rank(
            (dist_env.local_rank - dl + L) % L, (dist_env.node_idx - dn + NN) % NN);
        issue_put(d, dn == 0 ? this->cp_stream : this->cp_stream_inter_node);
      }
    }
    CUDA_CHECK(cudaEventRecord(this->fetch_remote_event, this->cp_stream_inter_node));
    CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream, this->fetch_remote_event));
    CUDA_CHECK(cudaEventRecord(this->all_gather_event, this->cp_stream));

    if (kStage2AfterPuts) {
      build_stage2();
    }

    return A2AVDispatchState{
        sorted_gather_index, sorted_scatter_index, sorted_splits_cumsum, (int)M_this_ep};
  }

  std::vector<torch::Tensor>
  forward_impl(
      torch::Tensor inputs_shard,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<std::vector<torch::Tensor>> input_scales,
      c10::optional<std::vector<torch::Tensor>> weight_scales,
      c10::optional<std::vector<torch::Tensor>> output_scales,
      c10::optional<std::vector<torch::Tensor>> outputs_buf,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin,
      const AllGatherOption &opt,
      c10::optional<torch::Tensor> splits_per_source,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    FLUX_CHECK(
#if TORCH_SUPPORT_FP8
        inputs_shard.scalar_type() == at::ScalarType::Float8_e5m2 ||
        inputs_shard.scalar_type() == at::ScalarType::Float8_e4m3fn ||
#endif
        inputs_shard.scalar_type() == at::ScalarType::BFloat16 ||
        inputs_shard.scalar_type() == at::ScalarType::Half)
        << inputs_shard.scalar_type();
    // Step 0. do some shape checks
    int const N = this->N;
    int const K = hidden;
    // doing shape CHECK
    CHECK_INPUT(inputs_shard, this->input_dtype);
    CHECK_NDIM(inputs_shard, 2);
    const int tokens_per_rank = inputs_shard.size(0);
    CHECK_2D(inputs_shard, tokens_per_rank, K);

    const int ntokens = tokens_per_rank * world_size;

    const std::size_t num_weights_group = weights.size();
    for (std::size_t i = 0; i < num_weights_group; ++i) {
      CHECK_INPUT(weights[i], this->input_dtype);
      CHECK_3D(weights[i], this->ep_nexperts, N, K);
    }

    CHECK_INPUT(splits_gpu, torch::kInt32);
    CHECK_NDIM(splits_gpu, 1);
    FLUX_CHECK_LE(this->nexperts, splits_gpu.size(0));

    CHECK_INPUT(scatter_index, torch::kInt32);
    CHECK_2D(scatter_index, ntokens, this->topk);

    // metadata-exchange result: per-source per-expert copy counts, host-side.
    // splits[e] is its column sum; every rank holds the identical matrix.
    const int32_t *cnt_host = nullptr;
    if (splits_per_source.has_value()) {
      auto const &cnt = splits_per_source.value();
      FLUX_CHECK(cnt.device().is_cpu()) << "splits_per_source must be a CPU tensor";
      FLUX_CHECK(cnt.scalar_type() == torch::kInt32);
      FLUX_CHECK(cnt.is_contiguous());
      CHECK_2D(cnt, world_size, this->nexperts);
      cnt_host = cnt.data_ptr<int32_t>();
    }

    FLUX_CHECK(!input_scales.has_value());
    FLUX_CHECK(!weight_scales.has_value());
    if (output_scales.has_value()) {
      TORCH_CHECK_EQ(output_scales->size(), num_weights_group);
      for (std::size_t i = 0; i < num_weights_group; ++i) {
        CHECK_INPUT(output_scales->at(i), torch::kFloat32);
        CHECK_1D(output_scales->at(i), this->ep_nexperts);
      }
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Step 1: get op. and prepare op buffers
    auto meta = this->get_gemm_meta(fast_accum);
    auto rt_conf = this->get_rt_conf();
    OpRegistry::OpPtr op;
    if (hparams.has_value()) {
      op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      op = OpRegistry::instance().get_op(meta, rt_conf);
    }
    const auto tile_shape = op->get_runtime_gemm_hparams().tile_shape();
    const int tile_M = cute::get<0>(tile_shape);
    const int tile_N = cute::get<1>(tile_shape);

    // Step 2: Launch AG comm as early as possible
    bool is_s8_gemm = is_s8_torch_dtype(inputs_shard.scalar_type());
    FLUX_CHECK(!is_s8_gemm) << "not support INT8 MOE AG+Scatter yet";

    int topk = this->topk;
    int ep_nexperts = this->ep_nexperts;
    int nexperts = this->nexperts;
    int ep_start = this->ep_start;
    torch::Tensor sorted_gather_index, sorted_scatter_index, sorted_splits_cumsum;
    torch::Tensor problem_schedules_gpu;
    int num_problem_schedules = 0;
    int M_this_ep = 0;

    if (a2av_dispatch_) {
      FLUX_CHECK_EQ((int)num_weights_group, 1) << "a2av mode supports a single weight group";
      FLUX_CHECK(!allgather_output.has_value()) << "a2av mode has no dense gathered buffer";
      FLUX_CHECK_EQ((int)splits_gpu.size(0), nexperts) << "drop-token unsupported in a2av mode";
      A2AVDispatchState a2av_state =
          this->a2av_dispatch(inputs_shard, splits_gpu, scatter_index, cnt_host, stream);
      sorted_gather_index = a2av_state.sorted_gather_index;
      sorted_scatter_index = a2av_state.sorted_scatter_index;
      sorted_splits_cumsum = a2av_state.sorted_splits_cumsum;
      M_this_ep = a2av_state.M_this_ep;
      if (a2av_ring_) {
        // static ring schedule: the prepare kernel takes the dense branch and
        // writes ProblemSchedV2 into this buffer (bucket workspace is skipped)
        num_problem_schedules = ep_nexperts * world_size * num_weights_group;
        problem_schedules_gpu = empty_with_uninitialized_data(
            std::vector<int64_t>{num_problem_schedules * (int64_t)sizeof(ProblemSchedule)},
            torch::TensorOptions(torch::kInt8).device(torch::kCUDA));
      }
    } else {
    if (nnodes == 1) {
      CUDA_CHECK(cudaEventRecord(this->ready_event, stream));
      CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream, this->ready_event));
      ag_op->run(inputs_shard, c10::nullopt, opt, this->cp_stream);
    } else {
      // reset the per-source-rank ready flags on the main stream: ordered before the
      // local shard copy (and hence before ready_event / any flag write) inside
      // all_gather_all2all
      CUDA_CHECK(cudaMemsetAsync(barrier_block.get(), 0, barrier_block.bytes(), stream));
      this->all_gather_all2all(inputs_shard);
    }

    // Step 3: helper kernels. for preparing gather_index & sort tokens & outputs
    // should be M_this_ep, but never mind gather_index takes little memory
    auto opt_i32d = torch::TensorOptions(torch::kCUDA)
                        .dtype(at::ScalarType::Int)
                        .device_index(at::cuda::current_device());
    auto opt_i32h =
        torch::TensorOptions(torch::kCPU).dtype(at::ScalarType::Int).pinned_memory(true);
    torch::Tensor gather_index =
        empty_with_uninitialized_data(std::vector<int64_t>{ntokens * topk}, opt_i32d);
    sorted_gather_index =
        empty_with_uninitialized_data(std::vector<int64_t>{ntokens * topk}, opt_i32d);
    sorted_scatter_index =
        empty_with_uninitialized_data(std::vector<int64_t>{ntokens * topk}, opt_i32d);
    torch::Tensor M_this_ep_holder =
        empty_with_uninitialized_data(std::vector<int64_t>{1}, opt_i32h);
    torch::Tensor sorted_splits =
        empty_with_uninitialized_data(std::vector<int64_t>{ep_nexperts * world_size}, opt_i32d);
    sorted_splits_cumsum =
        empty_with_uninitialized_data(std::vector<int64_t>{ep_nexperts * world_size}, opt_i32d);
    calc_gather_index_impl(
        nexperts,
        ntokens,
        topk,
        ep_start,
        ep_start + ep_nexperts,
        splits_gpu.data_ptr<int32_t>(),
        scatter_index.data_ptr<int32_t>(),
        gather_index.data_ptr<int32_t>(),
        M_this_ep_holder.data_ptr<int>(),
        stream);

    AGScatterSortOpArgumentsV2 moe_sort_args = {
        rank,
        world_size,
        ntokens,
        ep_nexperts,
        splits_gpu.data_ptr<int32_t>() + ep_start,
        gather_index.data_ptr<int32_t>(),
        sorted_splits.data_ptr<int32_t>(),
        sorted_splits_cumsum.data_ptr<int32_t>(),
        sorted_scatter_index.data_ptr<int32_t>(),
        sorted_gather_index.data_ptr<int32_t>(),
    };
    ag_scatter_sort_impl_v2(moe_sort_args, stream);

    sort_scatter_index_to_per_expert(
        sorted_scatter_index.data_ptr<int>(),
        splits_gpu.data_ptr<int>(),
        ep_start,
        ep_nexperts,
        stream);

    M_this_ep = scatter_index.numel();  // for EP=1, M_this_ep is always M_full
    if (ep_nexperts != nexperts) {
      if (cnt_host != nullptr) {
        // metadata shortcut: sum my experts' columns on the host, skipping the
        // dense path's only per-iteration device sync (the pinned readback)
        int64_t m = 0;
        for (int s = 0; s < world_size; s++) {
          for (int e = ep_start; e < ep_start + ep_nexperts; e++) {
            m += cnt_host[s * this->nexperts + e];
          }
        }
        M_this_ep = (int)m;
      } else {
        CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
        M_this_ep = *M_this_ep_holder.data_ptr<int32_t>();
      }
    }

    num_problem_schedules = ep_nexperts * world_size * num_weights_group;
    problem_schedules_gpu = empty_with_uninitialized_data(
        std::vector<int64_t>{num_problem_schedules * (int64_t)sizeof(ProblemSchedule)},
        torch::TensorOptions(torch::kInt8).device(torch::kCUDA));
    }  // end dense (non-a2av) path
    // Step 4: prepare GEMM args
    torch::Tensor barrier;  // engaged iff nnodes == 1 (dense path)
    int32_t *barrier_ptr = nullptr;
    torch::Tensor input_buffer;
    if (a2av_dispatch_) {
      // rows are addressed through sorted_gather_index; signals replace the barrier
      input_buffer = this->a2av_recv_buffer;
    } else if (nnodes == 1) {
      barrier = ag_op->local_barrier_buffer();
      barrier_ptr = barrier.data_ptr<int32_t>();
      input_buffer = ag_op->local_input_buffer().slice(0, 0, ntokens);
    } else {
      barrier_ptr = reinterpret_cast<int32_t *>(barrier_block.get());
      input_buffer = this->input_buffer.slice(0, 0, ntokens);
    }

    // shapes check
    std::vector<torch::Tensor> outputs = outputs_buf.value_or([&]() {
      std::vector<torch::Tensor> outputs;
      for (std::size_t i = 0; i < num_weights_group; ++i) {
        outputs.emplace_back(empty_with_uninitialized_data(
            std::vector<int64_t>{M_this_ep, N}, inputs_shard.options()));
      };
      return outputs;
    }());

    TORCH_CHECK_EQ(outputs.size(), num_weights_group);
    for (std::size_t i = 0; i < num_weights_group; ++i) {
      CHECK_INPUT(outputs[i], this->output_dtype);
      CHECK_2D(outputs[i], M_this_ep, N);
    }

    // set the output type here accordlingly
    auto args = GemmGroupedV2AGScatterArguments{
        .rank = rank,
        .world_size = world_size,
        .dist_env = dist_env,
        .sm_margin = sm_margin,
        .num_groups = (int)num_weights_group,
        .ep_start = ep_start,
        .ep_nexperts = ep_nexperts,
        .input = input_buffer.data_ptr(),
        .M_this_ep = M_this_ep,
        .N = N,
        .K = K,
        .splits = splits_gpu.data_ptr<int>(),
        .gather_A = sorted_gather_index.data_ptr<int32_t>(),
        .scatter_D = sorted_scatter_index.data_ptr<int32_t>(),
        .problem_schedules =
            problem_schedules_gpu.defined() ? problem_schedules_gpu.data_ptr() : nullptr,
        .num_problem_schedules = num_problem_schedules,
        .accum_per_rank_ptr = sorted_splits_cumsum.data_ptr<int32_t>(),
        .tile_size_m = tile_M,
        .tile_size_n = tile_N,
        .barrier_ptr = barrier_ptr};
    if (a2av_dispatch_) {
      args.signal_ptr = reinterpret_cast<uint64_t *>(this->a2av_signal_buffer.data_ptr());
      args.signal_expected = this->run_id_;
      args.a2av_ring_schedule = a2av_ring_;
    }
    for (int gid = 0; gid < num_weights_group; gid++) {
      args.weight[gid] = weights[gid].data_ptr();
      args.output[gid] = outputs[gid].data_ptr();
      args.scaleD[gid] =
          output_scales.has_value() ? output_scales->at(gid).data_ptr<float>() : nullptr;
    }

    static const bool kA2avTiming = get_int_from_env("FLUX_A2AV_TIMING", 0) != 0;
    if (a2av_dispatch_) {
      // do not start the (SM-occupying) GEMM before all puts/signals are issued;
      // in ring mode the intra-node puts live on cp_stream, which all_gather_event
      // covers (it is recorded after cp_stream waits on fetch_remote_event)
      CUDA_CHECK(
          cudaStreamWaitEvent(stream, a2av_ring_ ? this->all_gather_event : this->fetch_remote_event));
      if (kA2avTiming) {
        CUDA_CHECK(cudaEventRecord(this->timing_events_[3], stream));
      }
    } else if (nnodes == 1) {
      CUDA_CHECK(cudaStreamWaitEvent(stream, ag_op->get_local_prepare_event()));
    } else {
      // do not start the (SM-occupying) GEMM before the remote fetches are issued
      CUDA_CHECK(cudaStreamWaitEvent(stream, this->fetch_remote_event));
    }
    if (M_this_ep > 0) {
      int64_t workspace_size = op->get_workspace_size(args);
      lazy_init_buffer_tensor(&this->workspace_buffer, workspace_size);

      // Step 5: launch GEMM
      op->run(args, workspace_size ? this->workspace_buffer.data_ptr() : nullptr, stream);
    }
    if (a2av_dispatch_ && kA2avTiming) {
      CUDA_CHECK(cudaEventRecord(this->timing_events_[4], stream));
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, this->all_gather_event));
    if (nnodes > 1 || a2av_dispatch_) {
      // ensure that when the next time each rank copy data to itself's shard in the
      // input_buffer, all ranks have already finished allgather so that we can
      // safely modify input_buffer. In a2av mode this barrier additionally quiets
      // our outstanding nbi puts and keeps iteration n+1 puts from racing
      // iteration n's GEMM reads of the recv buffer.
      nvshmemx_barrier_all_on_stream(stream);
    }
    if (a2av_dispatch_ && kA2avTiming) {
      CUDA_CHECK(cudaEventRecord(this->timing_events_[5], stream));
      CUDA_CHECK(cudaEventSynchronize(this->timing_events_[5]));
      float seg[kNumTimingEvents - 1];
      for (int i = 0; i < kNumTimingEvents - 1; i++) {
        CUDA_CHECK(
            cudaEventElapsedTime(&seg[i], this->timing_events_[i], this->timing_events_[i + 1]));
      }
      fprintf(
          stderr,
          "[a2av-timing] rank %d stage1 %.3f stage2 %.3f gemmgate %.3f gemm %.3f barrier %.3f ms\n",
          rank,
          seg[0],
          seg[1],
          seg[2],
          seg[3],
          seg[4]);
      float s2[kNumStage2Events - 1];
      for (int i = 0; i < kNumStage2Events - 1; i++) {
        CUDA_CHECK(
            cudaEventElapsedTime(&s2[i], this->stage2_events_[i], this->stage2_events_[i + 1]));
      }
      fprintf(
          stderr,
          "[a2av-stage2] rank %d mask %.3f keyA %.3f sortA %.3f keyR %.3f sortR %.3f inv %.3f "
          "gather %.3f scatter %.3f cnt %.3f cumsum %.3f ms\n",
          rank,
          s2[0],
          s2[1],
          s2[2],
          s2[3],
          s2[4],
          s2[5],
          s2[6],
          s2[7],
          s2[8],
          s2[9]);
    }

    if (allgather_output.has_value()) {
      CHECK_INPUT(allgather_output.value(), this->input_dtype);
      CHECK_2D(allgather_output.value(), ntokens, K);
      CUDA_CHECK(cudaMemcpyAsync(
          allgather_output->data_ptr(),
          input_buffer.data_ptr(),
          allgather_output->nbytes(),
          cudaMemcpyDeviceToDevice,
          stream));
    }

    return outputs;
  }

#if defined(FLUX_WITH_TRITON_AOT)
  using FuncType = decltype(moe_ag_scatter_grouped_gemm_s8_ex);
  moe_ag_scatter_grouped_gemm_kernel__triton_algo_info_t
  get_default_triton_algo_info(at::ScalarType input_dtype, bool has_bias) {
    moe_ag_scatter_grouped_gemm_kernel__triton_algo_info_t algo_info;
    bool is_s8_gemm = is_s8_torch_dtype(input_dtype);
    if (is_s8_gemm) {
      algo_info = moe_ag_scatter_grouped_gemm_kernel__triton_algo_info_t{
          .WITH_BIAS = has_bias,
          .BLOCK_SIZE_M = 64,
          .BLOCK_SIZE_N = 128,
          .BLOCK_SIZE_K = 64,
          .GROUP_SIZE_M = 4,
          .num_warps = 4,
          .num_stages = 4};
    } else if (input_dtype == torch::kHalf) {
      algo_info = moe_ag_scatter_grouped_gemm_kernel__triton_algo_info_t{
          .WITH_BIAS = has_bias,
          .BLOCK_SIZE_M = 128,
          .BLOCK_SIZE_N = 128,
          .BLOCK_SIZE_K = 32,
          .GROUP_SIZE_M = 8,
          .num_warps = 4,
          .num_stages = 3};
    } else if (input_dtype == torch::kBFloat16) {
      algo_info = moe_ag_scatter_grouped_gemm_kernel__triton_algo_info_t{
          .WITH_BIAS = has_bias,
          .BLOCK_SIZE_M = 128,
          .BLOCK_SIZE_N = 128,
          .BLOCK_SIZE_K = 32,
          .GROUP_SIZE_M = 8,
          .num_warps = 4,
          .num_stages = 3};
    } else {
      FLUX_CHECK(false) << "unsupported dtype " << input_dtype;
    }
    return algo_info;
  }
  FuncType *
  get_triton_aot_func(at::ScalarType input_dtype) {
    if (input_dtype == torch::kInt8) {
      return moe_ag_scatter_grouped_gemm_s8_ex;
    } else if (input_dtype == torch::kHalf) {
      return moe_ag_scatter_grouped_gemm_fp16_ex;
    } else if (input_dtype == torch::kBFloat16) {
      return moe_ag_scatter_grouped_gemm_bf16_ex;
    } else {
      FLUX_CHECK(false) << "unsupported dtype " << input_dtype;
      return nullptr;
    }
  }
#endif
  std::vector<torch::Tensor>
  forward_triton_aot_impl(
      torch::Tensor inputs_shard,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<std::vector<torch::Tensor>> biases,
      c10::optional<std::vector<torch::Tensor>> input_scales,
      c10::optional<std::vector<torch::Tensor>> weight_scales,
      c10::optional<std::vector<torch::Tensor>> output_scales,
      c10::optional<std::vector<torch::Tensor>> outputs_bufs,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin,
      const AllGatherOption &opt) {
#if defined(FLUX_WITH_TRITON_AOT)
    FLUX_CHECK(nnodes == 1) << "moe_ag_scatter triton path is single-node only";
    FLUX_CHECK(!a2av_dispatch_) << "a2av dispatch mode does not support the triton path";
    FLUX_CHECK(weights.size() == 1);
    bool is_fp8_gemm = is_fp8_torch_dtype(inputs_shard.scalar_type());
    bool is_s8_gemm = is_s8_torch_dtype(inputs_shard.scalar_type());
    FLUX_CHECK(!is_fp8_gemm) << "not support INT8 MOE AG+Scatter yet";
    // Step 0. do some shape checks
    int const N = this->N;
    int const K = this->hidden;
    // doing shape CHECK
    CHECK_INPUT(inputs_shard, this->input_dtype);
    CHECK_NDIM(inputs_shard, 2);
    const int tokens_per_rank = inputs_shard.size(0);
    CHECK_2D(inputs_shard, tokens_per_rank, K);

    const int ntokens = tokens_per_rank * world_size;

    const std::size_t num_weights_group = weights.size();
    for (std::size_t i = 0; i < num_weights_group; ++i) {
      CHECK_INPUT(weights[i], this->input_dtype);
      CHECK_3D(weights[i], this->ep_nexperts, N, K);  // RCR layout
    }

    CHECK_INPUT(splits_gpu, torch::kInt32);
    CHECK_NDIM(splits_gpu, 1);
    FLUX_CHECK_LE(this->nexperts, splits_gpu.size(0));

    CHECK_INPUT(scatter_index, torch::kInt32);
    CHECK_2D(scatter_index, ntokens, this->topk);

    if (is_s8_gemm) {
      FLUX_CHECK(biases.has_value());
      FLUX_CHECK(input_scales.has_value());
      FLUX_CHECK(weight_scales.has_value());
    } else {
      FLUX_CHECK(!biases.has_value());
      FLUX_CHECK(!input_scales.has_value());
      FLUX_CHECK(!weight_scales.has_value());
    }
    if (biases.has_value()) {
      FLUX_CHECK_EQ(biases->size(), num_weights_group);
      for (int i = 0; i < num_weights_group; i++) {
        CHECK_INPUT(biases->at(i), torch::kFloat32);
        CHECK_3D(biases->at(i), this->ep_nexperts, 1, N);
      }
    }
    if (input_scales.has_value()) {
      FLUX_CHECK_EQ(input_scales->size(), num_weights_group);
      for (int i = 0; i < num_weights_group; i++) {
        CHECK_INPUT(input_scales->at(i), torch::kFloat32);
        CHECK_1D(input_scales->at(i), tokens_per_rank);
      }
    }
    if (weight_scales.has_value()) {
      for (int i = 0; i < num_weights_group; i++) {
        CHECK_INPUT(weight_scales->at(i), torch::kFloat32);
        CHECK_3D(weight_scales->at(i), this->ep_nexperts, 1, N);
      }
    }
    if (output_scales.has_value()) {
      TORCH_CHECK_EQ(output_scales->size(), num_weights_group);
      for (std::size_t i = 0; i < num_weights_group; ++i) {
        CHECK_INPUT(output_scales->at(i), torch::kFloat32);
        CHECK_1D(output_scales->at(i), this->ep_nexperts);
      }
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Step 2: Launch AG comm as early as possible
    bool allgather_input_scale = input_scales.has_value() && is_s8_gemm;

    CUDA_CHECK(cudaEventRecord(this->ready_event, stream));
    CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream, this->ready_event));
    ag_op->run(
        inputs_shard,
        allgather_input_scale ? c10::optional<torch::Tensor>{input_scales->at(0)} : c10::nullopt,
        opt,
        this->cp_stream);

    // Step 3: helper kernels. for preparing gather_index & sort tokens & outputs
    // should be M_this_ep, but never mind gather_index takes little memory
    int M_this_ep;
    torch::Tensor m_pad_holder;
    torch::Tensor gather_a_index;
    torch::Tensor scatter_d_index;
    torch::Tensor expert_index;
    torch::Tensor rank_start_index;
    torch::Tensor rank_end_index;

    FuncType *moe_ag_scatter_grouped_gemm = get_triton_aot_func(inputs_shard.scalar_type());
    auto algo_info = get_default_triton_algo_info(inputs_shard.scalar_type(), biases.has_value());
    std::tie(
        M_this_ep,
        m_pad_holder,
        gather_a_index,
        scatter_d_index,
        expert_index,
        rank_start_index,
        rank_end_index) =
        prepare_moe_ag_scatter_args(
            splits_gpu,
            scatter_index,
            ntokens,
            topk,
            1,
            ep_start,
            ep_nexperts,
            rank,
            world_size,
            algo_info.BLOCK_SIZE_M,
            (intptr_t)stream);

    // Step 4: prepare GEMM args
    torch::Tensor barrier = ag_op->local_barrier_buffer();
    torch::Tensor input_buffer = ag_op->local_input_buffer().slice(0, 0, ntokens);
    c10::optional<torch::Tensor> input_scale_tensor =
        allgather_input_scale
            ? c10::optional<torch::Tensor>{ag_op->local_input_scale_buffer().slice(0, 0, ntokens)}
            : c10::nullopt;

    // shapes check
    std::vector<torch::Tensor> outputs = outputs_bufs.value_or([&]() {
      std::vector<torch::Tensor> outputs;
      auto option = at::TensorOptions(this->output_dtype).device(torch::kCUDA);
      for (std::size_t i = 0; i < num_weights_group; ++i) {
        outputs.emplace_back(
            empty_with_uninitialized_data(std::vector<int64_t>{M_this_ep, N}, option));
      };
      return outputs;
    }());

    TORCH_CHECK_EQ(outputs.size(), num_weights_group);
    for (std::size_t i = 0; i < num_weights_group; ++i) {
      CHECK_INPUT(outputs[i], this->output_dtype);
      CHECK_2D(outputs[i], M_this_ep, N);
    }

    FLUX_CHECK(input_scales.has_value());
    FLUX_CHECK(weight_scales.has_value());

    if (M_this_ep > 0) {
      CUDA_CHECK(cudaStreamWaitEvent(stream, ag_op->get_local_prepare_event()));
      auto rtn = moe_ag_scatter_grouped_gemm(
          (CUstream)stream,
          (CUdeviceptr)input_buffer.data_ptr(),
          (CUdeviceptr)weights[0].data_ptr(),
          (CUdeviceptr)outputs[0].data_ptr(),
          (CUdeviceptr)(biases.has_value() ? biases->at(0).data_ptr() : nullptr),  // bias
          (CUdeviceptr)input_scale_tensor->data_ptr(),                             // input_scale
          (CUdeviceptr)(weight_scales.has_value() ? weight_scales->at(0).data_ptr()
                                                  : nullptr),  // weight_scale
          (CUdeviceptr)(output_scales.has_value() ? output_scales->at(0).data_ptr()
                                                  : nullptr),  // output_scale
          (CUdeviceptr)gather_a_index.data_ptr(),
          (CUdeviceptr)scatter_d_index.data_ptr(),
          (CUdeviceptr)expert_index.data_ptr(),
          (CUdeviceptr)rank_start_index.data_ptr(),
          (CUdeviceptr)rank_end_index.data_ptr(),
          (CUdeviceptr)m_pad_holder.data_ptr(),
          N,
          K,
          ep_nexperts,
          M_this_ep,
          input_buffer.stride(0),
          input_buffer.stride(1),
          weights[0].stride(0),
          weights[0].stride(2),
          weights[0].stride(1),  // transpose_weight
          outputs[0].stride(0),
          outputs[0].stride(1),
          (CUdeviceptr)barrier.data_ptr(),
          algo_info);
      CU_CHECK(rtn);
    }

    CUDA_CHECK(cudaStreamWaitEvent(stream, this->all_gather_event));

    if (allgather_output.has_value()) {
      CHECK_INPUT(allgather_output.value(), this->input_dtype);
      CHECK_2D(allgather_output.value(), ntokens, K);
      CUDA_CHECK(cudaMemcpyAsync(
          allgather_output->data_ptr(),
          input_buffer.data_ptr(),
          allgather_output->nbytes(),
          cudaMemcpyDeviceToDevice,
          stream));
    }

    return outputs;
#else
    FLUX_CHECK(false) << "please compile with --triton-aot option.";
#endif
  }

 public:
  std::vector<torch::Tensor>
  forward_multiple_weights(
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
      c10::optional<torch::Tensor> splits_per_source) {
    bool is_s8_gemm = inputs_shard.scalar_type() == torch::kInt8;
    AllGatherOption option = materialize(ag_option, is_s8_gemm && input_scale.has_value());
    return forward_impl(
        std::move(inputs_shard),
        std::move(weights),
        std::move(splits_gpu),
        std::move(scatter_index),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        std::move(outputs_buf),
        std::move(allgather_output),
        fast_accum,
        sm_margin,
        option,
        std::move(splits_per_source),
        c10::nullopt);
  }

  void
  clear_buffers() {
    if (nnodes > 1 && this->input_buffer.defined()) {
      this->input_buffer.zero_();
    }
    // a2av signal buffer is deliberately NOT cleared: the epoch scheme relies on
    // monotonically increasing signal values and clearing would corrupt in-flight
    // iterations. Data buffers need no clearing (rows fully overwritten per use).
  }

  torch::Tensor
  forward(
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
      c10::optional<torch::Tensor> splits_per_source) {
    if (inputs_shard.scalar_type() == torch::kInt8) {
      return forward_triton_aot(
          inputs_shard,
          weights,
          splits_gpu,
          scatter_index,
          bias,
          input_scale,
          weight_scale,
          output_scale,
          outputs_buf,
          allgather_output,
          fast_accum,
          sm_margin,
          ag_option);
    }
    FLUX_CHECK(!bias.has_value());
    bool is_s8_gemm = inputs_shard.scalar_type() == torch::kInt8;
    AllGatherOption option = materialize(ag_option, is_s8_gemm && input_scale.has_value());
    auto outputs = forward_impl(
        std::move(inputs_shard),
        {weights},
        std::move(splits_gpu),
        std::move(scatter_index),
        as_optional_vec(input_scale),
        as_optional_vec(weight_scale),
        as_optional_vec(output_scale),
        as_optional_vec(outputs_buf),
        std::move(allgather_output),
        fast_accum,
        sm_margin,
        option,
        std::move(splits_per_source),
        c10::nullopt);
    return outputs[0];
  }

  torch::Tensor
  forward_triton_aot(
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
      AllGatherOptionWithOptional ag_option) {
    bool is_s8_gemm = inputs_shard.scalar_type() == torch::kInt8;
    AllGatherOption option = materialize(ag_option, is_s8_gemm && input_scale.has_value());
    auto outputs = forward_triton_aot_impl(
        std::move(inputs_shard),
        {weights},
        std::move(splits_gpu),
        std::move(scatter_index),
        as_optional_vec(bias),
        as_optional_vec(input_scale),
        as_optional_vec(weight_scale),
        as_optional_vec(output_scale),
        as_optional_vec(outputs_buf),
        std::move(allgather_output),
        fast_accum,
        sm_margin,
        option);
    return outputs[0];
  }

  std::vector<torch::Tensor>
  profiling(
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
      c10::intrusive_ptr<ProfilingContext> opt_ctx) {
    bool is_s8_gemm = inputs_shard.scalar_type() == torch::kInt8;
    AllGatherOption option = materialize(ag_option, is_s8_gemm && input_scale.has_value());
    auto meta = unify_type(this->get_gemm_meta(fast_accum));
    auto rt_conf = this->get_rt_conf();
    ProfilingContext tmp_ctx("__tmp__");
    ProfilingContext *ctx = opt_ctx == nullptr ? &tmp_ctx : opt_ctx.get();

    auto elapsed_tensor = torch::empty({}, inputs_shard.options().dtype(c10::ScalarType::Float));
    auto reduced_elapsed_tensor = elapsed_tensor.clone();

    OpRegistry::instance().visit_hparams(
        [&](UnifiedGemmHParams const &hparams) {
          // filter non-consistent hparams
          constexpr int warm_iters = 5;
          constexpr int iters = 10;
          float total_elapsed = 0;

          auto stream = c10::cuda::getCurrentCUDAStream();
          group_barrier.barrier_all(stream);
          c10::cuda::stream_synchronize(stream);
          auto cp_hparams = hparams;
          for (int iter = 0; iter < warm_iters + iters; ++iter) {
            GpuTimer timer;
            timer.start(stream);
            auto output [[maybe_unused]] = this->forward_impl(
                inputs_shard,
                weights,
                splits_gpu,
                scatter_index,
                input_scale,
                weight_scale,
                output_scale,
                outputs_buf,
                allgather_output,
                fast_accum,
                sm_margin,
                option,
                c10::nullopt,
                cp_hparams);
            timer.stop();
            if (iter >= warm_iters) {
              total_elapsed += timer.elapsed_millis();
            }
          }

          // Avoid GPU frequency adjustment
          group_barrier.barrier_all(stream);
          c10::cuda::stream_synchronize(stream);
          sleep(1);
          float avg_elapsed = int(total_elapsed / iters * 1000) / 1000.0;
          float reduce_elapsed = all_reduce_max_float(this->tp_group.get(), avg_elapsed);
          ctx->add(meta, rt_conf, hparams, reduce_elapsed);
        },
        meta);

    auto best_hparams = ctx->record_best(meta, rt_conf);
    return this->forward_impl(
        std::move(inputs_shard),
        std::move(weights),
        std::move(splits_gpu),
        std::move(scatter_index),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        std::move(outputs_buf),
        std::move(allgather_output),
        fast_accum,
        sm_margin,
        option,
        c10::nullopt,
        std::move(best_hparams));
  }
};

GemmGroupedV2AGScatterOp::GemmGroupedV2AGScatterOp(
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
    bool a2av_dispatch,
    bool a2av_ring)
    : impl_(new GemmGroupedV2AGScatterOpImpl(
          tp_group,
          ep_size,
          nnodes,
          max_ntokens,
          ffn_hidden,  // before TP shard
          hidden,
          num_experts,
          topk,
          input_dtype,
          output_dtype,
          a2av_dispatch,
          a2av_ring)) {}
GemmGroupedV2AGScatterOp::~GemmGroupedV2AGScatterOp() { delete impl_; }

void
GemmGroupedV2AGScatterOp::clear_buffers() {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2AGScatterOp is not initialized";
  impl_->clear_buffers();
}
torch::Tensor
GemmGroupedV2AGScatterOp::forward(
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
    c10::optional<torch::Tensor> splits_per_source) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2AGScatterOp is not initialized";
  return impl_->forward(
      std::move(inputs_shard),
      std::move(weights),
      std::move(splits_gpu),
      std::move(scatter_index),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      std::move(outputs_buf),
      std::move(allgather_output),
      fast_accum,
      sm_margin,
      ag_option,
      std::move(splits_per_source));
}
torch::Tensor
GemmGroupedV2AGScatterOp::forward_triton_aot(
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
    AllGatherOptionWithOptional ag_option) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2AGScatterOp is not initialized";
  return impl_->forward_triton_aot(
      std::move(inputs_shard),
      std::move(weights),
      std::move(splits_gpu),
      std::move(scatter_index),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      std::move(outputs_buf),
      std::move(allgather_output),
      fast_accum,
      sm_margin,
      ag_option);
}
std::vector<torch::Tensor>
GemmGroupedV2AGScatterOp::forward_multiple_weights(
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
    c10::optional<torch::Tensor> splits_per_source) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2AGScatterOp is not initialized";
  return impl_->forward_multiple_weights(
      std::move(inputs_shard),
      std::move(weights),
      std::move(splits_gpu),
      std::move(scatter_index),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      std::move(outputs_buf),
      std::move(allgather_output),
      fast_accum,
      sm_margin,
      ag_option,
      std::move(splits_per_source));
}
std::vector<torch::Tensor>
GemmGroupedV2AGScatterOp::profiling(
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
    c10::intrusive_ptr<ProfilingContext> opt_ctx) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2AGScatterOp is not initialized";
  return impl_->profiling(
      std::move(inputs_shard),
      std::move(weights),
      std::move(splits_gpu),
      std::move(scatter_index),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      std::move(outputs_buf),
      std::move(allgather_output),
      fast_accum,
      sm_margin,
      ag_option,
      std::move(opt_ctx));
}

}  // namespace bytedance::flux::ths_op
