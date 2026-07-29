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
#include <c10/cuda/CUDAGuard.h>
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
  // balanced-relay signal aggregation only (compress, nnodes > 1): pure
  // front-end memops (CUStreamWaitValue64/CUStreamWriteValue64, zero SMs).
  // Must not ride cp_stream (would couple gateway rounds across local ranks)
  // nor cp_stream_inter_node (would poison fetch_remote_event, which gates the
  // GEMM launch).
  c10::cuda::CUDAStream cp_stream_signal;
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
  // a2av_hier: hierarchical dispatch mirroring all_gather_all2all — intra-node
  // traffic goes direct (ring dn==0 slots); inter-node traffic travels as ONE
  // aggregated message per peer node to the same-local-rank "gateway", which
  // then forwards each destination's sub-chunk intra-node. The GEMM keeps the
  // dense static problem schedule (rounds land in receiver stage order).
  const bool a2av_hier_;
  // a2av_hier_compress: hierarchical dispatch with token-dedup wire semantics.
  // The traffic matrix / splits / scatter_index stay LOGICAL (GEMM problem
  // sizes and schedule unchanged), but each token crosses the wire at most once
  // per destination RANK (intra-node: dedup across that rank's experts) and at
  // most once per destination NODE (inter-node: one union aggregate to the
  // gateway, which gathers each local rank's exact subset and forwards it).
  // Receiver-side duplication is free: multiple GEMM A rows alias one recv row
  // through sorted_gather_index (gather_A is read-only in the kernel).
  const bool a2av_hier_compress_;
  // FLUX_A2AV_RELAY_IDENTITY=1 (compress, nnodes > 1): disable the balanced
  // inter-node relay and keep the fixed relay = self assignment (the original
  // a2av_hier_compress wire, byte-identical) for A/B. The flag changes the
  // WIRE layout, so it must be set identically on every rank.
  const bool relay_identity_;
  uint64_t run_id_ = 0;             // epoch value carried by the NVSHMEM signals
  int64_t max_recv_ntokens_ = 0;    // rows of the symmetric recv buffer
  int64_t max_stage_ntokens_ = 0;   // rows of the symmetric gateway staging buffer
  torch::Tensor a2av_send_buffer;   // symmetric [tokens_per_rank_max * topk, hidden]
  torch::Tensor a2av_recv_buffer;   // symmetric [max_recv_ntokens_, hidden]
  torch::Tensor a2av_signal_buffer; // symmetric uint64[world_size], never memset
  // a2av_hier only (nnodes > 1): staging area for inbound node-aggregated
  // payloads, plus per-source-node arrival signals (epoch discipline, never memset)
  torch::Tensor a2av_stage_buffer_;       // symmetric [max_stage_ntokens_, hidden]
  torch::Tensor a2av_node_signal_buffer_; // symmetric uint64[nnodes]
  cudaEvent_t hier_dispatch_event_ = nullptr;  // round-0 intra puts issued (GEMM gate)
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
  // a2av_hier_compress only: gateway forward-pack scratch (each local
  // destination's subset is gathered here out of the staging union before its
  // put; plain device memory — only the put DESTINATION must be symmetric),
  // plus fixed-shape index scratch for the sync-free pack/consumer/forward
  // index builds (garbage-slot idiom, no nonzero/masked_select)
  torch::Tensor a2av_fwd_scratch_;  // [tokens_per_rank_max * topk, hidden], nnodes > 1
  torch::Tensor a2av_mine_token_;   // i32 [max_ntokens + 1] (+1 = garbage slot)
  torch::Tensor a2av_pack_flag_;    // i32 [tokens_per_rank_max * (L + NN - 1)]
  torch::Tensor a2av_pack_gather_;  // i64 [tokens_per_rank_max * topk + 1]
  torch::Tensor a2av_fwd_flag_;     // i32 [(NN-1) * tokens_per_rank_max * L + 1], nnodes > 1
  torch::Tensor a2av_fwd_idx_;      // i64 [(NN-1) * tokens_per_rank_max * topk + 1]
  int64_t compress_meta_off_ = 0;   // 8-aligned offset of the compress fields in the meta arena
  cudaEvent_t fwd_index_event_ = nullptr;  // forward-index build done (gateway gathers gate)
  // balanced inter-node relay (compress, nnodes > 1, !relay_identity_): each
  // round's canonical stream (the node's L union segments, ascending source
  // local rank) is cut into L near-equal chunks; local relay rank k stages and
  // wire-puts chunk k to the same-local-rank gateway on the target node. All
  // chunk boundaries derive from the replicated U matrix, so sender / relay /
  // gateway / destination agree with zero extra metadata.
  torch::Tensor a2av_relay_stage_;    // symmetric [max_relay_ntokens_, hidden]
  torch::Tensor a2av_relay_sig_;      // symmetric u64[(NN-1)*L], slot (dn-1)*L + src_lr
  torch::Tensor a2av_gw_round_sig_;   // symmetric u64[(NN-1)*L], slot (dn-1)*L + gw_lr
  torch::Tensor a2av_fwd_cnt_pinned_; // pinned i32 [2, NN-1, L, L]: cnt_in / cnt_before
  int64_t max_relay_ntokens_ = 0;     // rows of the symmetric relay staging buffer
  cudaEvent_t fwd_cnt_event_ = nullptr;     // cnt_in/cnt_before D2H done (host reads)
  cudaEvent_t relay_send_event_ = nullptr;  // relay piece puts issued (GEMM gate)
  cudaEvent_t signal_done_event_ = nullptr; // dest-side signal aggregation issued
  // FLUX_A2AV_TIMING=1 diagnostics: per-forward segment boundaries on the main stream
  static constexpr int kNumTimingEvents = 6;
  cudaEvent_t timing_events_[kNumTimingEvents] = {};
  static constexpr int kNumStage2Events = 11;
  cudaEvent_t stage2_events_[kNumStage2Events] = {};
  // FLUX_A2AV_TIMING=1, balanced-relay fwd-index build only: op-group boundaries
  static constexpr int kNumRelayFwdEvents = 12;
  cudaEvent_t relay_fwd_events_[kNumRelayFwdEvents] = {};

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
      bool a2av_ring = false,
      bool a2av_hier = false,
      bool a2av_hier_compress = false)
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
        cp_stream_signal(create_cp_stream()),
        a2av_dispatch_(a2av_dispatch),
        a2av_ring_(a2av_ring),
        a2av_hier_(a2av_hier),
        a2av_hier_compress_(a2av_hier_compress),
        relay_identity_(get_int_from_env("FLUX_A2AV_RELAY_IDENTITY", 0) != 0),
        // ring_mode barriers are CUDA-IPC based and intra-node only; multi-node
        // must take the NVSHMEM barrier (ring_mode = false)
        group_barrier(this->tp_group, nnodes == 1 && this->tp_group->get_size() > 8) {
    _ensure_topo_initialized();
    CHECK_DIV(nexperts, ep_size);
    CHECK_DIV(ffn_hidden, ffn_tp_size);
    FLUX_CHECK_GE(nnodes, 1);
    CHECK_DIV(world_size, nnodes);
    FLUX_CHECK(!a2av_ring || a2av_dispatch) << "a2av_ring requires a2av_dispatch";
    FLUX_CHECK(!a2av_hier || a2av_dispatch) << "a2av_hier requires a2av_dispatch";
    FLUX_CHECK(!(a2av_hier && a2av_ring)) << "a2av_hier and a2av_ring are mutually exclusive";
    FLUX_CHECK(!a2av_hier_compress || a2av_dispatch)
        << "a2av_hier_compress requires a2av_dispatch";
    FLUX_CHECK(!(a2av_hier_compress && a2av_ring))
        << "a2av_hier_compress and a2av_ring are mutually exclusive";
    FLUX_CHECK(!(a2av_hier_compress && a2av_hier))
        << "a2av_hier_compress and a2av_hier are mutually exclusive";
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
      if ((a2av_hier || a2av_hier_compress) && nnodes > 1) {
        // gateway staging: holds the node-aggregated inbound payloads from the
        // nnodes-1 same-local-rank peers; expected load ~= one rank's recv (the
        // node's inbound traffic splits across L gateways by source local rank)
        this->max_stage_ntokens_ = get_int_from_env(
            "FLUX_A2AV_MAX_STAGE_NTOKENS",
            (int)std::min<int64_t>(
                (int64_t)max_ntokens * topk, tokens_per_rank_max * topk * 2));
        this->a2av_stage_buffer_ =
            nvshmem_create_tensor({this->max_stage_ntokens_, hidden}, input_dtype);
        this->a2av_node_signal_buffer_ =
            nvshmem_create_tensor({nnodes}, at::ScalarType::Long, /*init_zero=*/true);
        if (a2av_hier_compress && !this->relay_identity_) {
          // balanced-relay staging: holds MY wire chunks for all NN-1 rounds at
          // once (piece puts for every round are issued before the first wire
          // wait — see the deadlock note in a2av_dispatch). Expected load
          // ~= the node's total outbound / L, i.e. one balanced rank's share.
          const int64_t L = world_size / nnodes;
          this->max_relay_ntokens_ = get_int_from_env(
              "FLUX_A2AV_MAX_RELAY_NTOKENS",
              (int)std::min<int64_t>(
                  (int64_t)max_ntokens * topk, tokens_per_rank_max * topk * 2));
          this->a2av_relay_stage_ =
              nvshmem_create_tensor({this->max_relay_ntokens_, hidden}, input_dtype);
          this->a2av_relay_sig_ = nvshmem_create_tensor(
              {(int64_t)(nnodes - 1) * L}, at::ScalarType::Long, /*init_zero=*/true);
          this->a2av_gw_round_sig_ = nvshmem_create_tensor(
              {(int64_t)(nnodes - 1) * L}, at::ScalarType::Long, /*init_zero=*/true);
          this->a2av_fwd_cnt_pinned_ = torch::empty(
              {2, (int64_t)nnodes - 1, L, L},
              torch::TensorOptions(torch::kCPU).dtype(torch::kInt).pinned_memory(true));
        }
      }
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
      // compress appends (8-aligned): send-segment offsets i64[nseg + 1] with
      // nseg = L + NN - 1, plus the gateway forward-index column offsets and,
      // in balanced-relay mode, the canonical source starts and my per-round
      // window bounds; covered by the same single H2D upload.
      //   identity: fwd_col_off i64[(NN-1) * L]         (per (round, dst_lr))
      //   relay:    fwd_col_off i64[(NN-1) * L * L]     (per (round, src_lr, dst_lr))
      //             + recv_start i64[(NN-1) * L] + win_a i64[NN-1] + win_b i64[NN-1]
      int64_t total_meta_bytes = meta_bytes;
      if (a2av_hier_compress) {
        const int64_t L = world_size / nnodes;
        const int64_t nseg = L + nnodes - 1;
        const int64_t R = nnodes - 1;
        const int64_t extra = this->relay_identity_ ? R * L : R * L * L + R * L + 2 * R;
        this->compress_meta_off_ = pad_to(meta_bytes, (int64_t)8);
        total_meta_bytes =
            this->compress_meta_off_ + (nseg + 1 + extra) * (int64_t)sizeof(int64_t);
      }
      this->a2av_meta_pinned_ = torch::empty(
          {total_meta_bytes},
          torch::TensorOptions(torch::kCPU).dtype(torch::kByte).pinned_memory(true));
      this->a2av_meta_dev_ = torch::empty(
          {total_meta_bytes}, torch::TensorOptions(torch::kCUDA).dtype(torch::kByte));
      if (a2av_hier_compress) {
        const int64_t L = world_size / nnodes;
        const int64_t nseg = L + nnodes - 1;
        auto opt_cuda_i32 = torch::TensorOptions(torch::kCUDA).dtype(torch::kInt);
        this->a2av_mine_token_ = torch::empty({(int64_t)max_ntokens + 1}, opt_cuda_i32);
        this->a2av_pack_flag_ = torch::empty({tokens_per_rank_max * nseg}, opt_cuda_i32);
        this->a2av_pack_gather_ =
            torch::empty({tokens_per_rank_max * (int64_t)topk + 1}, opt_cuda_i64);
        if (nnodes > 1) {
          // relay mode grows the index scratch by the extra source-lr axis:
          //   flag: (round, src_lr, token, dst_lr); idx columns capacity
          //   Sum_{round, src_lr, dst_lr} u <= (NN-1) * L * T * min(topk, L)
          // (per-round scratch capacity is unchanged: in-window forwarded rows
          //  <= window_rows * min(topk, L) <= T * topk)
          const int64_t src_lrs = this->relay_identity_ ? 1 : L;
          const int64_t idx_cap = this->relay_identity_
                                      ? (nnodes - 1) * tokens_per_rank_max * (int64_t)topk
                                      : (nnodes - 1) * tokens_per_rank_max * L *
                                            std::min<int64_t>(topk, L);
          this->a2av_fwd_scratch_ = torch::empty(
              {tokens_per_rank_max * (int64_t)topk, hidden},
              torch::TensorOptions(torch::kCUDA).dtype(input_dtype));
          this->a2av_fwd_flag_ = torch::empty(
              {(nnodes - 1) * tokens_per_rank_max * src_lrs * L + 1}, opt_cuda_i32);
          this->a2av_fwd_idx_ = torch::empty({idx_cap + 1}, opt_cuda_i64);
        }
      }
      if (rank == 0) {
        double sym_mb = (this->a2av_send_buffer.nbytes() + this->a2av_recv_buffer.nbytes() +
                         this->a2av_signal_buffer.nbytes()) /
                        1024.0 / 1024.0;
        if (this->a2av_stage_buffer_.defined()) {
          sym_mb += (this->a2av_stage_buffer_.nbytes() +
                     this->a2av_node_signal_buffer_.nbytes()) /
                    1024.0 / 1024.0;
        }
        if (this->a2av_relay_stage_.defined()) {
          sym_mb += (this->a2av_relay_stage_.nbytes() + this->a2av_relay_sig_.nbytes() +
                     this->a2av_gw_round_sig_.nbytes()) /
                    1024.0 / 1024.0;
        }
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
    CUDA_CHECK(cudaEventCreateWithFlags(&this->hier_dispatch_event_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->fwd_index_event_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->fwd_cnt_event_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->relay_send_event_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->signal_done_event_, cudaEventDisableTiming));
    for (int i = 0; i < kNumTimingEvents; i++) {
      CUDA_CHECK(cudaEventCreate(&this->timing_events_[i]));  // timing-capable
    }
    for (int i = 0; i < kNumStage2Events; i++) {
      CUDA_CHECK(cudaEventCreate(&this->stage2_events_[i]));
    }
    for (int i = 0; i < kNumRelayFwdEvents; i++) {
      CUDA_CHECK(cudaEventCreate(&this->relay_fwd_events_[i]));  // timing-capable
    }
  }

  ~GemmGroupedV2AGScatterOpImpl() {
    for (int i = 0; i < kNumTimingEvents; i++) {
      CUDA_CHECK(cudaEventDestroy(this->timing_events_[i]));
    }
    for (int i = 0; i < kNumStage2Events; i++) {
      CUDA_CHECK(cudaEventDestroy(this->stage2_events_[i]));
    }
    for (int i = 0; i < kNumRelayFwdEvents; i++) {
      CUDA_CHECK(cudaEventDestroy(this->relay_fwd_events_[i]));
    }
    CUDA_CHECK(cudaEventDestroy(this->counts_event_));
    CUDA_CHECK(cudaEventDestroy(this->hier_dispatch_event_));
    CUDA_CHECK(cudaEventDestroy(this->fwd_index_event_));
    CUDA_CHECK(cudaEventDestroy(this->fwd_cnt_event_));
    CUDA_CHECK(cudaEventDestroy(this->relay_send_event_));
    CUDA_CHECK(cudaEventDestroy(this->signal_done_event_));
    CUDA_CHECK(cudaEventDestroy(this->all_gather_event));
    CUDA_CHECK(cudaEventDestroy(this->fetch_remote_event));
    CUDA_CHECK(cudaEventDestroy(this->ready_event));
    CUDA_CHECK(cudaStreamDestroy(this->cp_stream));
    CUDA_CHECK(cudaStreamDestroy(this->cp_stream_inter_node));
    CUDA_CHECK(cudaStreamDestroy(this->cp_stream_signal));
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
      const int32_t *uc_host,   // [W, W + nnodes] a2av_unique_counts, or nullptr
      cudaStream_t stream) {
    const int W = this->world_size;
    const int64_t E = this->ep_nexperts;
    const int tokens_per_rank = inputs_shard.size(0);
    const int64_t copies_per_rank = (int64_t)tokens_per_rank * topk;
    const int64_t n_copies = copies_per_rank * W;
    const bool use_meta = cnt_host != nullptr;
    // compress (a2av_hier_compress_) working set, filled in the use_meta block:
    // u_mat[s*W+d]  = unique tokens source s must deliver to RANK d
    // U_mat[s*NN+n] = unique tokens source s must deliver to NODE n (union)
    const bool compress = this->a2av_hier_compress_;
    FLUX_CHECK(!compress || (use_meta && uc_host != nullptr))
        << "a2av_hier_compress requires splits_per_source AND a2av_unique_counts";
    std::vector<int64_t> u_mat, U_mat, recv_off_u, seg_off_h, fwd_col_off_h;
    // balanced-relay only: canonical starts of the inbound sources per round,
    // and my per-round staging window [win_a, win_b)
    std::vector<int64_t> recv_start_h, win_a_h, win_b_h;
    int64_t total_send_rows = 0;
    torch::Tensor seg_off_dev, fwd_col_off_dev, recv_start_dev, win_a_dev, win_b_dev;
    // ---- balanced-relay partition (compress && !relay_identity_): per round,
    // the L union segments of a (source node n -> target node m) transfer form
    // ONE canonical stream (ascending source local rank, token-ascending
    // interiors — exactly the pack order); it is cut into L near-equal
    // contiguous chunks and local relay rank k carries chunk k. These lambdas
    // are the SINGLE source of truth for the partition: sender pieces, relay
    // wire puts, gateway windows and every capacity check must all derive from
    // them (any divergent re-derivation silently corrupts wire offsets).
    // All lazy over U_mat (filled in the compress metadata block below).
    auto U_of = [&](int n, int sl, int m) -> int64_t {
      return U_mat[(int64_t)dist_env.local_rank_to_global_rank(sl, n) * dist_env.nnodes + m];
    };
    // canonical start of source (n, sl)'s segment in the n -> m stream;
    // sl == L yields the stream total
    auto canon_start = [&](int n, int sl, int m) -> int64_t {
      int64_t acc = 0;
      for (int s2 = 0; s2 < sl; s2++) {
        acc += U_of(n, s2, m);
      }
      return acc;
    };
    // balanced chunk boundary k (k in [0, L]) of the n -> m stream: the first
    // (total mod L) chunks get one extra row
    auto chunk_bound = [&](int n, int m, int k) -> int64_t {
      const int64_t Lb = dist_env.local_world_size;
      const int64_t total = canon_start(n, (int)Lb, m);
      return (total / Lb) * k + std::min<int64_t>((int64_t)k, total % Lb);
    };
    auto chunk_rows_of = [&](int n, int m, int k) -> int64_t {
      return chunk_bound(n, m, k + 1) - chunk_bound(n, m, k);
    };
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
      // ---- compress metadata: the unique-token counts drive the WIRE layout
      // only; everything logical above (chunks64, M_this_ep, group tables)
      // stays untouched. seg_off (compressed send segments) and fwd_col_off
      // (gateway forward-index columns) ride the same pinned arena and single
      // H2D below.
      if (compress) {
        const int L = dist_env.local_world_size;
        const int NN = dist_env.nnodes;
        const int my_node = dist_env.node_idx;
        const int my_lr = dist_env.local_rank;
        const int ucols = W + NN;
        const int64_t nseg = L + NN - 1;
        u_mat.assign((size_t)W * W, 0);
        U_mat.assign((size_t)W * NN, 0);
        for (int s = 0; s < W; s++) {
          for (int d = 0; d < W; d++) {
            u_mat[s * W + d] = uc_host[s * ucols + d];
          }
          for (int n = 0; n < NN; n++) {
            U_mat[s * NN + n] = uc_host[s * ucols + W + n];
          }
        }
        // host sanity: dedup counts must be consistent with the logical matrix
        for (int s = 0; s < W; s++) {
          for (int d = 0; d < W; d++) {
            int64_t uv = u_mat[s * W + d];
            int64_t cv = chunks64[s * W + d];
            FLUX_CHECK(uv >= 0 && uv <= cv && (uv > 0) == (cv > 0))
                << "a2av_unique_counts inconsistent with splits_per_source at (" << s << ", "
                << d << ")";
            FLUX_CHECK_LE(uv, U_mat[s * NN + d / L]);
          }
          for (int n = 0; n < NN; n++) {
            int64_t su = 0;
            for (int d = n * L; d < (n + 1) * L; d++) {
              su += u_mat[s * W + d];
            }
            FLUX_CHECK_LE(U_mat[s * NN + n], su)
                << "a2av_unique_counts node union out of range at (" << s << ", " << n << ")";
          }
        }
        // dedup recv layout: source-major regions of u[s][d] rows. Overflow
        // check takes the max over ALL destinations — the same expression on
        // every rank, so a failure is collective (no one-rank-throws hang).
        int64_t max_col = 0;
        for (int d = 0; d < W; d++) {
          int64_t col = 0;
          for (int s = 0; s < W; s++) {
            col += u_mat[s * W + d];
          }
          max_col = std::max(max_col, col);
        }
        FLUX_CHECK_LE(max_col, this->max_recv_ntokens_)
            << "a2av compress recv overflow; raise FLUX_A2AV_MAX_RECV_NTOKENS";
        recv_off_u.assign(W, 0);
        for (int s = 1; s < W; s++) {
          recv_off_u[s] = recv_off_u[s - 1] + u_mat[(int64_t)(s - 1) * W + rank];
        }
        // compressed send segments: nodes ascending; my node expanded into L
        // per-destination-rank segments (ascending global rank); each remote
        // node is ONE union segment. Interiors are ascending token index.
        seg_off_h.assign(nseg + 1, 0);
        for (int n = 0, seg = 0; n < NN; n++) {
          if (n == my_node) {
            for (int dl = 0; dl < L; dl++, seg++) {
              seg_off_h[seg + 1] = seg_off_h[seg] + u_mat[(int64_t)rank * W + n * L + dl];
            }
          } else {
            seg_off_h[seg + 1] = seg_off_h[seg] + U_mat[(int64_t)rank * NN + n];
            seg++;
          }
        }
        total_send_rows = seg_off_h[nseg];
        FLUX_CHECK_LE(total_send_rows, copies_per_rank);
        // gateway forward-index columns: exact-packed, absolute offsets into
        // a2av_fwd_idx_; the scratch buffer reuses per-round-relative (identity)
        // or host-running-packed (relay) offsets each round.
        //   identity: one column per (round dn, local dest dl) — the round's
        //             source is the single same-local-rank rank
        //   relay:    one column per (round dn, source lr sl, local dest dl) —
        //             my staging window spans every source lr of the round
        if (this->relay_identity_ || NN == 1) {
          fwd_col_off_h.assign((size_t)std::max(NN - 1, 0) * L, 0);
          int64_t facc = 0;
          for (int dn = 1; dn < NN; dn++) {
            int s = dist_env.local_rank_to_global_rank(my_lr, (my_node + dn) % NN);
            for (int dl = 0; dl < L; dl++) {
              fwd_col_off_h[(dn - 1) * L + dl] = facc;
              facc += u_mat[(int64_t)s * W + my_node * L + dl];
            }
          }
          if (NN > 1) {
            FLUX_CHECK_LT(facc, this->a2av_fwd_idx_.numel());  // last slot = garbage
          }
        } else {
          const int R = NN - 1;
          fwd_col_off_h.assign((size_t)R * L * L, 0);
          int64_t facc = 0;
          for (int dn = 1; dn < NN; dn++) {
            int ns = (my_node + dn) % NN;
            for (int sl = 0; sl < L; sl++) {
              int s = dist_env.local_rank_to_global_rank(sl, ns);
              for (int dl = 0; dl < L; dl++) {
                fwd_col_off_h[((size_t)(dn - 1) * L + sl) * L + dl] = facc;
                facc += u_mat[(int64_t)s * W + my_node * L + dl];
              }
            }
          }
          FLUX_CHECK_LT(facc, this->a2av_fwd_idx_.numel());  // last slot = garbage
          // canonical source starts + my staging window per round, for the
          // window mask of the generalized forward-index build
          recv_start_h.assign((size_t)R * L, 0);
          win_a_h.assign(R, 0);
          win_b_h.assign(R, 0);
          for (int dn = 1; dn < NN; dn++) {
            int ns = (my_node + dn) % NN;
            for (int sl = 0; sl < L; sl++) {
              recv_start_h[(size_t)(dn - 1) * L + sl] = canon_start(ns, sl, my_node);
            }
            win_a_h[dn - 1] = chunk_bound(ns, my_node, my_lr);
            win_b_h[dn - 1] = chunk_bound(ns, my_node, my_lr + 1);
          }
        }
        // stage seg_off + the forward tables into the pinned arena (same H2D
        // below); the relay-only vectors are empty in identity mode
        int64_t *cmp_h = reinterpret_cast<int64_t *>(stage + this->compress_meta_off_);
        for (int64_t i = 0; i <= nseg; i++) {
          cmp_h[i] = seg_off_h[i];
        }
        size_t coff = (size_t)nseg + 1;
        for (size_t i = 0; i < fwd_col_off_h.size(); i++) {
          cmp_h[coff++] = fwd_col_off_h[i];
        }
        for (size_t i = 0; i < recv_start_h.size(); i++) {
          cmp_h[coff++] = recv_start_h[i];
        }
        for (size_t i = 0; i < win_a_h.size(); i++) {
          cmp_h[coff++] = win_a_h[i];
        }
        for (size_t i = 0; i < win_b_h.size(); i++) {
          cmp_h[coff++] = win_b_h[i];
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
      if (compress) {
        const int64_t L = dist_env.local_world_size;
        const int64_t NN = dist_env.nnodes;
        const int64_t nseg = L + NN - 1;
        seg_off_dev =
            torch::from_blob(dev + this->compress_meta_off_, {nseg}, opt_dev_i64);
        if (NN > 1) {
          char *cbase = dev + this->compress_meta_off_ + (nseg + 1) * 8;
          if (this->relay_identity_) {
            fwd_col_off_dev = torch::from_blob(cbase, {(NN - 1) * L}, opt_dev_i64);
          } else {
            const int64_t R = NN - 1;
            fwd_col_off_dev = torch::from_blob(cbase, {R * L * L}, opt_dev_i64);
            recv_start_dev = torch::from_blob(cbase + R * L * L * 8, {R * L}, opt_dev_i64);
            win_a_dev =
                torch::from_blob(cbase + (R * L * L + R * L) * 8, {R}, opt_dev_i64);
            win_b_dev =
                torch::from_blob(cbase + (R * L * L + R * L + R) * 8, {R}, opt_dev_i64);
          }
        }
      }
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

    if (compress) {
      // compressed producer pack: ONE send row per (token, segment) pair.
      // Sync-free build (garbage-slot scatter + cumsum + index_select): flag
      // each (local token, segment) once, per-segment exclusive cumsum over
      // ascending token index yields the row within the segment, then one
      // index_select gathers the rows from inputs_shard.
      const int64_t Lc = dist_env.local_world_size;
      const int64_t NNc = dist_env.nnodes;
      const int64_t my_node = dist_env.node_idx;
      const int64_t nseg = Lc + NNc - 1;
      auto e_my = e_all.narrow(0, (int64_t)rank * copies_per_rank, copies_per_rank);
      auto d64 = e_my.div(E, "floor");   // destination global rank per copy
      auto nd = d64.div(Lc, "floor");    // destination node
      // segment id: remote node n -> n (n < my_node) or n + L - 1 (n > my_node);
      // my node local rank dl -> my_node + dl == d - my_node * (L - 1)
      auto seg = torch::where(
          nd.eq(my_node),
          d64 - my_node * (Lc - 1),
          torch::where(nd.lt(my_node), nd, nd + (Lc - 1)));
      auto tl = iota.narrow(0, 0, copies_per_rank).div((int64_t)topk, "floor");
      auto pf = this->a2av_pack_flag_.narrow(0, 0, (int64_t)tokens_per_rank * nseg);
      pf.zero_();
      pf.scatter_(0, tl * nseg + seg, 1);  // dup (token, seg) hits write 1 again
      auto flag2d = pf.view({(int64_t)tokens_per_rank, nseg});
      auto pos = flag2d.cumsum(0) - flag2d;  // exclusive rank within the segment
      auto tgt = (pos + seg_off_dev.unsqueeze(0))
                     .masked_fill_(flag2d.eq(0), (int64_t)copies_per_rank);  // garbage slot
      auto tgrid = iota.narrow(0, 0, (int64_t)tokens_per_rank * nseg).div(nseg, "floor");
      this->a2av_pack_gather_.scatter_(0, tgt.reshape(-1), tgrid);
      if (total_send_rows > 0) {
        auto send_view = this->a2av_send_buffer.narrow(0, 0, total_send_rows);
        at::index_select_out(
            send_view, inputs_shard, 0, this->a2av_pack_gather_.narrow(0, 0, total_send_rows));
      }
      static const bool kCheckCompressPack =
          get_int_from_env("FLUX_A2AV_CHECK_COMPRESS", 0) != 0;
      if (kCheckCompressPack) {
        // debug only (may sync): per-segment flag counts must reproduce the
        // u/U-derived segment sizes, else pack rows spill into the next segment
        auto seg_len = flag2d.sum(0).cpu();  // [nseg]
        for (int64_t i = 0; i < nseg; i++) {
          FLUX_CHECK_EQ(seg_len[i].item<int64_t>(), seg_off_h[i + 1] - seg_off_h[i])
              << "a2av compress pack-flag/segment-size mismatch at segment " << i;
        }
      }
    } else {
      // producer pack: my copies only, destination-major. pack_key = e * cpr +
      // local_p, so ascending order is (destination, expert, copy index) — the
      // copy-index tie-break is mirrored by the consumer keys in stage 2.
      auto perm_s = this->a2av_pack_key_.narrow(0, 0, copies_per_rank).argsort();
      auto send_gather_index = perm_s.div((int64_t)topk, "floor");
      auto send_view = this->a2av_send_buffer.narrow(0, 0, copies_per_rank);
      at::index_select_out(send_view, inputs_shard, 0, send_gather_index);
    }
    CUDA_CHECK(cudaEventRecord(this->ready_event, stream));
    if (kTiming) {
      CUDA_CHECK(cudaEventRecord(this->timing_events_[1], stream));
    }

    // ---- compress gateway forward indices (main stream, overlaps the wire):
    // ONE batched build over all NN-1 rounds (round r's source = my local rank
    // on node (my_node + r + 1) % NN): for each LOCAL destination, its rows out
    // of that source's staged union segment. Stored value is posU = the token's
    // exclusive rank within the union (ascending token index) == its row in the
    // staged segment; column interiors are ascending token, matching the
    // producer's segment order.
    if (compress && dist_env.nnodes > 1 && this->relay_identity_) {
      const int64_t Lc = dist_env.local_world_size;
      const int64_t NNc = dist_env.nnodes;
      const int64_t R = NNc - 1;
      const int64_t T = tokens_per_rank;
      const int64_t my_node = dist_env.node_idx;
      const int64_t fwd_garbage = this->a2av_fwd_idx_.numel() - 1;
      auto tl = iota.narrow(0, 0, copies_per_rank).div((int64_t)topk, "floor");
      // [R, cpr] copies of the same-local-rank sources in round order: view
      // e_all as [NN, L, cpr], pick my local rank's row, roll node my_node + 1
      // to the front, drop my own node (roll materializes the strided view)
      auto e_src = e_all.view({NNc, Lc, copies_per_rank})
                       .select(1, dist_env.local_rank)
                       .roll(-(my_node + 1), 0)
                       .narrow(0, 0, R);
      auto dl = e_src.div(E, "floor").sub_(my_node * Lc);  // local dest, or off-node
      auto off_node = dl.lt(0).logical_or_(dl.ge(Lc));
      auto r_base = this->a2av_arange_i64_.narrow(0, 0, R).view({R, 1}) * (T * Lc);
      auto fp = (r_base + tl.unsqueeze(0) * Lc + dl).masked_fill_(off_node, R * T * Lc);
      auto ff = this->a2av_fwd_flag_.narrow(0, 0, R * T * Lc + 1);
      ff.zero_();
      ff.scatter_(0, fp.reshape(-1), 1);  // garbage slot at R * T * Lc
      auto flag3d = ff.narrow(0, 0, R * T * Lc).view({R, T, Lc});
      auto uni = std::get<0>(flag3d.max(2));  // union flag per (round, token)
      auto posU = uni.cumsum(1) - uni;        // the token's row in the staged union
      auto pos = flag3d.cumsum(1) - flag3d;   // per-dest exclusive rank
      auto tgt =
          (pos + fwd_col_off_dev.view({R, 1, Lc})).masked_fill_(flag3d.eq(0), fwd_garbage);
      auto vals = posU.unsqueeze(2).expand({R, T, Lc});
      this->a2av_fwd_idx_.scatter_(0, tgt.reshape(-1), vals.reshape(-1));
      static const bool kCheckCompressFwd =
          get_int_from_env("FLUX_A2AV_CHECK_COMPRESS", 0) != 0;
      if (kCheckCompressFwd) {
        // debug only (may sync): the on-device flag counts must reproduce the
        // host metadata driving the wire offsets, else the gathers read skew
        auto cnt = flag3d.sum(1).cpu();  // [R, Lc] == u[s_r][my node dests]
        auto ucnt = uni.sum(1).cpu();    // [R]     == U[s_r][my_node]
        for (int64_t r = 0; r < R; r++) {
          int s = dist_env.local_rank_to_global_rank(
              dist_env.local_rank, (int)((my_node + r + 1) % NNc));
          for (int64_t dlv = 0; dlv < Lc; dlv++) {
            FLUX_CHECK_EQ(
                cnt[r][dlv].item<int64_t>(), u_mat[(int64_t)s * W + my_node * Lc + dlv])
                << "a2av compress fwd-flag/u mismatch at (" << s << ", " << dlv << ")";
          }
          FLUX_CHECK_EQ(ucnt[r].item<int64_t>(), U_mat[(int64_t)s * NNc + my_node])
              << "a2av compress fwd-union/U mismatch at source " << s;
        }
      }
      CUDA_CHECK(cudaEventRecord(this->fwd_index_event_, stream));
    } else if (compress && dist_env.nnodes > 1) {
      // ---- balanced relay: generalized forward-index build. My staging
      // window [win_a, win_b) of round dn holds a contiguous slice of the
      // canonical ns -> my_node stream, spanning MULTIPLE source local ranks.
      // Flags live on (round, src_lr, token, dst_lr); the union position plus
      // the host canonical start yields each row's canonical position, the
      // window mask selects my slice, and the stored value is the row's
      // window-relative staging position. cnt_in / cnt_before (the in-window
      // and before-window row counts per (round, src_lr, dst_lr)) ride a tiny
      // async D2H — the host needs them to address the forward puts, since a
      // window cut inside a source's segment is token-level information.
      const int64_t Lc = dist_env.local_world_size;
      const int64_t NNc = dist_env.nnodes;
      const int64_t R = NNc - 1;
      const int64_t T = tokens_per_rank;
      const int64_t my_node = dist_env.node_idx;
      const int64_t fwd_garbage = this->a2av_fwd_idx_.numel() - 1;
      auto rmark = [&](int i) {
        if (kTiming) {
          CUDA_CHECK(cudaEventRecord(this->relay_fwd_events_[i], stream));
        }
      };
      rmark(0);
      auto tl = iota.narrow(0, 0, copies_per_rank).div((int64_t)topk, "floor");
      // [R, L, cpr]: ALL source local ranks, rounds in gateway arrival order
      // ns = (my_node + dn) % NN (roll materializes the strided view)
      auto e_rounds = e_all.view({NNc, Lc, copies_per_rank})
                          .roll(-(my_node + 1), 0)
                          .narrow(0, 0, R);
      auto dl = e_rounds.div(E, "floor").sub_(my_node * Lc);  // local dest, or off-node
      auto off_node = dl.lt(0).logical_or_(dl.ge(Lc));
      rmark(1);
      // flat flag position ((r * L + sl) * T + t) * L + dl
      auto rsl_base =
          this->a2av_arange_i64_.narrow(0, 0, R * Lc).view({R, Lc, 1}) * (T * Lc);
      auto fp = (rsl_base + tl.view({1, 1, copies_per_rank}) * Lc + dl)
                    .masked_fill_(off_node, R * Lc * T * Lc);
      auto ff = this->a2av_fwd_flag_.narrow(0, 0, R * Lc * T * Lc + 1);
      ff.zero_();
      ff.scatter_(0, fp.reshape(-1), 1);  // garbage slot at R * L * T * L
      auto flag4d = ff.narrow(0, 0, R * Lc * T * Lc).view({R, Lc, T, Lc});
      rmark(2);
      auto uni = std::get<0>(flag4d.max(3));  // union flag per (round, src_lr, token)
      auto posU = uni.cumsum(2) - uni;        // row within (r, sl)'s union segment
      auto canon = posU + recv_start_dev.view({R, Lc, 1});  // canonical position
      rmark(3);
      auto in_w = canon.ge(win_a_dev.view({R, 1, 1}))
                      .logical_and_(canon.lt(win_b_dev.view({R, 1, 1})));
      auto below = canon.lt(win_a_dev.view({R, 1, 1}));
      rmark(4);
      auto valid = flag4d * in_w.unsqueeze(3);  // needed by dl AND inside my window
      rmark(5);
      auto pos = valid.cumsum(2) - valid;       // in-window rank within (r, sl, dl)
      rmark(6);
      auto tgt = (pos + fwd_col_off_dev.view({R, Lc, 1, Lc}))
                     .masked_fill_(valid.eq(0), fwd_garbage);
      rmark(7);
      // stored value = window-relative staging row
      auto vals = (canon - win_a_dev.view({R, 1, 1})).unsqueeze(3).expand({R, Lc, T, Lc});
      auto tgt_flat = tgt.reshape(-1);
      auto vals_flat = vals.reshape(-1);
      rmark(8);
      this->a2av_fwd_idx_.scatter_(0, tgt_flat, vals_flat);
      rmark(9);
      auto cnt_in = valid.sum(2);                          // [R, L, L]
      auto cnt_before = (flag4d * below.unsqueeze(3)).sum(2);  // [R, L, L]
      rmark(10);
      this->a2av_fwd_cnt_pinned_.select(0, 0).copy_(cnt_in.to(torch::kInt), true);
      this->a2av_fwd_cnt_pinned_.select(0, 1).copy_(cnt_before.to(torch::kInt), true);
      rmark(11);
      CUDA_CHECK(cudaEventRecord(this->fwd_cnt_event_, stream));
      static const bool kCheckCompressFwd =
          get_int_from_env("FLUX_A2AV_CHECK_COMPRESS", 0) != 0;
      if (kCheckCompressFwd) {
        // debug only (may sync): flag counts must reproduce the host metadata,
        // and the window split must tile each u column exactly
        auto cnt = flag4d.sum(2).cpu();  // [R, L, L] == u[(ns, sl)][my node dests]
        auto ucnt = uni.sum(2).cpu();    // [R, L]    == U[(ns, sl)][my_node]
        auto cin = cnt_in.cpu();
        auto cbef = cnt_before.cpu();
        for (int64_t r = 0; r < R; r++) {
          int ns = (int)((my_node + r + 1) % NNc);
          for (int64_t sl = 0; sl < Lc; sl++) {
            int s = dist_env.local_rank_to_global_rank((int)sl, ns);
            for (int64_t dlv = 0; dlv < Lc; dlv++) {
              int64_t uv = u_mat[(int64_t)s * W + my_node * Lc + dlv];
              FLUX_CHECK_EQ(cnt[r][sl][dlv].item<int64_t>(), uv)
                  << "a2av relay fwd-flag/u mismatch at (" << s << ", " << dlv << ")";
              FLUX_CHECK_LE(
                  cbef[r][sl][dlv].item<int64_t>() + cin[r][sl][dlv].item<int64_t>(), uv)
                  << "a2av relay window split out of range at (" << s << ", " << dlv << ")";
            }
            FLUX_CHECK_EQ(ucnt[r][sl].item<int64_t>(), U_mat[(int64_t)s * NNc + my_node])
                << "a2av relay fwd-union/U mismatch at source " << s;
          }
        }
      }
      CUDA_CHECK(cudaEventRecord(this->fwd_index_event_, stream));
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
      if (compress) {
        // dedup consumer: my recv buffer holds each token ONCE per source
        // region (interior ascending token index). One-cumsum identity: with
        // mine_token[t] = 1 iff any copy of global token t routes to my
        // experts, and C = its exclusive prefix sum, EVERY copy of t reads
        // recv row C[t] — tokens are source-contiguous, so recv_off_dedup[s] +
        // rank-of-t-within-source == C[t] exactly. scatter_D and the splits
        // cumsum keep their LOGICAL semantics; the GEMM problem sizes and the
        // dense static schedule are unchanged, multiple A rows just alias one
        // recv row (gather_A is read-only in the kernel).
        auto key_a = ((e_loc * W + s_all) * kShift + iota).masked_fill_(not_mine, kMax64);
        mark(2);
        auto perm_a = key_a.argsort();
        mark(3);
        const int64_t ntokens = (int64_t)tokens_per_rank * W;
        auto flat_token = iota.div((int64_t)topk, "floor");  // global token of copy p
        auto mine_n = this->a2av_mine_token_.narrow(0, 0, ntokens + 1);
        mine_n.zero_();
        mine_n.scatter_(0, flat_token.masked_fill(not_mine, ntokens), 1);
        auto c_excl = mine_n.cumsum(0) - mine_n;  // C[t], i64 [ntokens + 1]
        mark(4);
        auto gidx = c_excl.index_select(0, flat_token);
        // tail rows (>= M_this_ep) are unread garbage; clamp is pure hygiene
        sorted_gather_index = gidx.index_select(0, perm_a)
                                  .clamp_(0, this->max_recv_ntokens_ - 1)
                                  .to(torch::kInt);
        auto scatter_val = flat_dst - expert_base_dev.index_select(0, e_all);
        sorted_scatter_index = scatter_val.index_select(0, perm_a).to(torch::kInt);
        mark(5);
        sorted_splits_cumsum = ssc_dev;  // uploaded, exact LOGICAL [E, W] semantics
        // stages 6-10 don't exist in the compress consumer; mark them anyway so
        // the FLUX_A2AV_TIMING readout keeps one fixed index layout
        for (int i = 6; i <= 10; i++) {
          mark(i);
        }
        static const bool kCheckCompress =
            get_int_from_env("FLUX_A2AV_CHECK_COMPRESS", 0) != 0;
        if (kCheckCompress) {
          // debug only (may sync): invert C (recv row -> token) and assert
          // every A row's dedup recv row carries that row's own token
          auto tok_of_row = torch::full({this->max_recv_ntokens_ + 1}, -1, opt_i64);
          auto row_slot =
              c_excl.narrow(0, 0, ntokens)
                  .masked_fill(mine_n.narrow(0, 0, ntokens).eq(0), this->max_recv_ntokens_);
          tok_of_row.scatter_(0, row_slot, this->a2av_arange_i64_.narrow(0, 0, ntokens));
          auto a_tok = flat_token.index_select(0, perm_a).narrow(0, 0, M_this_ep);
          auto got = tok_of_row.index_select(
              0, sorted_gather_index.narrow(0, 0, M_this_ep).to(torch::kLong));
          FLUX_CHECK(torch::equal(got, a_tok)) << "a2av compress consumer identity mismatch";
        }
        return;
      }
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
    // compress only (u_mat/U_mat empty otherwise; lambdas are lazy)
    auto u_at = [&](int s, int d) -> int64_t { return u_mat[(int64_t)s * W + d]; };
    auto U_at = [&](int s, int n) -> int64_t {
      return U_mat[(int64_t)s * dist_env.nnodes + n];
    };

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
    // recv region's interior order, so one contiguous local copy suffices.
    // compress: dedup segment (my-node segment my_node + my_lr) -> dedup region
    const int64_t self_rows = compress ? u_at(rank, rank) : chunk_at(rank, rank);
    const int64_t self_send_off =
        compress ? seg_off_h[dist_env.node_idx + dist_env.local_rank] : send_off[rank];
    const int64_t self_recv_off = compress ? recv_off_u[rank] : recv_off[rank];
    if (self_rows > 0) {
      CUDA_CHECK(cudaMemcpyAsync(
          recv_base + self_recv_off * row_bytes,
          send_base + self_send_off * row_bytes,
          self_rows * row_bytes,
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
    if (a2av_hier_compress_) {
      // hierarchical dispatch with token-dedup wire semantics. Same stream /
      // signal / round discipline as a2av_hier below, but: intra-node puts send
      // each destination's UNIQUE tokens once; each remote node receives ONE
      // union aggregate at the gateway, which gathers each local rank's exact
      // subset (index_select, needs SMs -> sm_margin >= 1 enforced upstream)
      // into scratch and forwards it with a NON-nbi put (local completion
      // before the next round's gather reuses the scratch).
      const int L = dist_env.local_world_size;
      const int NN = dist_env.nnodes;
      const int my_node = dist_env.node_idx;
      const int my_lr = dist_env.local_rank;
      // staging offset of source node ns's union segment at gateway (gnode, glr):
      // segments exact-packed ascending by source node, gateway's own node skipped
      auto stage_off_u = [&](int gnode, int glr, int ns) -> int64_t {
        int64_t acc = 0;
        for (int n = 0; n < ns; n++) {
          if (n == gnode) {
            continue;
          }
          acc += U_at(dist_env.local_rank_to_global_rank(glr, n), gnode);
        }
        return acc;
      };
      // dedup recv offset of source s's region at destination d
      auto recv_off_of_u = [&](int s, int d) -> int64_t {
        int64_t acc = 0;
        for (int s2 = 0; s2 < s; s2++) {
          acc += u_at(s2, d);
        }
        return acc;
      };
      // my-node send segment for local destination dlg (see the pack)
      auto send_seg_off = [&](int dlg) -> int64_t { return seg_off_h[my_node + dlg]; };
      if (NN > 1 && this->relay_identity_) {
        // staging overflow check before any inter-node wire; same expression on
        // every rank -> collective failure (no one-rank hang)
        int64_t max_stage_rows = 0;
        for (int gn = 0; gn < NN; gn++) {
          for (int gl = 0; gl < L; gl++) {
            int64_t rows = 0;
            for (int ns = 0; ns < NN; ns++) {
              if (ns == gn) {
                continue;
              }
              rows += U_at(dist_env.local_rank_to_global_rank(gl, ns), gn);
            }
            max_stage_rows = std::max(max_stage_rows, rows);
          }
        }
        FLUX_CHECK_LE(max_stage_rows, this->max_stage_ntokens_)
            << "a2av_hier_compress staging overflow; raise FLUX_A2AV_MAX_STAGE_NTOKENS";
      } else if (NN > 1) {
        // balanced relay: gateway staging holds balanced chunks and the relay
        // staging holds my chunks of every outbound round; both bounds are
        // pure functions of the replicated U matrix -> collective failure
        int64_t max_stage_rows = 0, max_relay_rows = 0;
        for (int n = 0; n < NN; n++) {
          for (int k = 0; k < L; k++) {
            int64_t srows = 0;
            for (int ns = 0; ns < NN; ns++) {
              if (ns == n) {
                continue;
              }
              srows += chunk_rows_of(ns, n, k);
            }
            max_stage_rows = std::max(max_stage_rows, srows);
            int64_t rrows = 0;
            for (int dn = 1; dn < NN; dn++) {
              rrows += chunk_rows_of(n, (n - dn + NN) % NN, k);
            }
            max_relay_rows = std::max(max_relay_rows, rrows);
          }
        }
        FLUX_CHECK_LE(max_stage_rows, this->max_stage_ntokens_)
            << "a2av_hier_compress staging overflow; raise FLUX_A2AV_MAX_STAGE_NTOKENS";
        FLUX_CHECK_LE(max_relay_rows, this->max_relay_ntokens_)
            << "a2av relay staging overflow; raise FLUX_A2AV_MAX_RELAY_NTOKENS";
      }
      // round 0: intra-node direct puts of the dedup segments, mirror local order
      for (int dl = 1; dl < L; dl++) {
        int dlg = (my_lr - dl + L) % L;
        int d = dist_env.local_rank_to_global_rank(dlg, my_node);
        int64_t rows = u_at(rank, d);
        if (rows > 0) {
          nvshmemx_putmem_signal_nbi_on_stream(
              recv_base + recv_off_of_u(rank, d) * row_bytes,
              send_base + send_seg_off(dlg) * row_bytes,
              rows * row_bytes,
              signal_base + rank,
              this->run_id_,
              NVSHMEM_SIGNAL_SET,
              d,
              this->cp_stream);
        } else {
          nvshmemx_signal_op_on_stream(
              signal_base + rank, this->run_id_, NVSHMEM_SIGNAL_SET, d, this->cp_stream);
        }
      }
      CUDA_CHECK(cudaEventRecord(this->hier_dispatch_event_, this->cp_stream));
      if (NN > 1) {
        uint64_t *node_sig =
            reinterpret_cast<uint64_t *>(this->a2av_node_signal_buffer_.data_ptr());
        char *stage_base = reinterpret_cast<char *>(this->a2av_stage_buffer_.data_ptr());
        char *scratch_base = reinterpret_cast<char *>(this->a2av_fwd_scratch_.data_ptr());
        if (this->relay_identity_) {
          // inter-node union aggregates, mirror node order; arrival signal slot =
          // source node, value = epoch. Empty aggregates still signal.
          for (int dn = 1; dn < NN; dn++) {
            int tn = (my_node - dn + NN) % NN;
            int g = dist_env.local_rank_to_global_rank(my_lr, tn);
            int64_t rows = U_at(rank, tn);
            int seg = tn < my_node ? tn : tn + L - 1;
            if (rows > 0) {
              nvshmemx_putmem_signal_nbi_on_stream(
                  stage_base + stage_off_u(tn, my_lr, my_node) * row_bytes,
                  send_base + seg_off_h[seg] * row_bytes,
                  rows * row_bytes,
                  node_sig + my_node,
                  this->run_id_,
                  NVSHMEM_SIGNAL_SET,
                  g,
                  this->cp_stream_inter_node);
            } else {
              nvshmemx_signal_op_on_stream(
                  node_sig + my_node,
                  this->run_id_,
                  NVSHMEM_SIGNAL_SET,
                  g,
                  this->cp_stream_inter_node);
            }
          }
          // gateway rounds: per-round front-end wait (cuStreamWaitValue64, zero
          // SMs), then gather each local destination's exact subset out of the
          // staged union. The index_selects run on cp_stream (stream guard); they
          // wait on fwd_index_event_ so the index build (main stream) is done.
          CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream, this->fwd_index_event_));
          {
            c10::cuda::CUDAStreamGuard guard(this->cp_stream);
            auto opt_in = torch::TensorOptions(torch::kCUDA).dtype(this->input_dtype);
            for (int dn = 1; dn < NN; dn++) {
              int ns = (my_node + dn) % NN;
              int s = dist_env.local_rank_to_global_rank(my_lr, ns);
              CU_CHECK(CUStreamWaitValue64(
                  this->cp_stream,
                  reinterpret_cast<CUdeviceptr>(node_sig + ns),
                  this->run_id_,
                  CU_STREAM_WAIT_VALUE_GEQ));
              int64_t union_rows = U_at(s, my_node);
              auto stage_seg = torch::from_blob(
                  stage_base + stage_off_u(my_node, my_lr, ns) * row_bytes,
                  {std::max<int64_t>(union_rows, 1), (int64_t)hidden},
                  opt_in);
              const int64_t round_base = fwd_col_off_h[(dn - 1) * L];
              // forward in mirror local order (same stage slots as a2av_hier)
              for (int dl = 0; dl < L; dl++) {
                int dlg = (my_lr - dl + L) % L;
                int d = dist_env.local_rank_to_global_rank(dlg, my_node);
                int64_t rows = u_at(s, d);
                if (rows == 0) {
                  nvshmemx_signal_op_on_stream(
                      signal_base + s, this->run_id_, NVSHMEM_SIGNAL_SET, d, this->cp_stream);
                  continue;
                }
                auto idx = this->a2av_fwd_idx_.narrow(0, fwd_col_off_h[(dn - 1) * L + dlg], rows);
                if (d == rank) {
                  // gateway's own subset: gather straight into the recv region
                  auto dst = torch::from_blob(
                      recv_base + recv_off_of_u(s, rank) * row_bytes,
                      {rows, (int64_t)hidden},
                      opt_in);
                  at::index_select_out(dst, stage_seg, 0, idx);
                  nvshmemx_signal_op_on_stream(
                      signal_base + s, this->run_id_, NVSHMEM_SIGNAL_SET, rank, this->cp_stream);
                } else {
                  // round-relative offsets: the gateway's own (d == rank) column
                  // leaves a hole in the scratch — harmless capacity slack, the
                  // per-round total is still <= copies_per_rank
                  const int64_t scratch_off = fwd_col_off_h[(dn - 1) * L + dlg] - round_base;
                  auto dst = torch::from_blob(
                      scratch_base + scratch_off * row_bytes, {rows, (int64_t)hidden}, opt_in);
                  at::index_select_out(dst, stage_seg, 0, idx);
                  // NON-nbi on purpose: the scratch is refilled next round, and
                  // nbi gives no local-completion guarantee
                  nvshmemx_putmem_signal_on_stream(
                      recv_base + recv_off_of_u(s, d) * row_bytes,
                      scratch_base + scratch_off * row_bytes,
                      rows * row_bytes,
                      signal_base + s,
                      this->run_id_,
                      NVSHMEM_SIGNAL_SET,
                      d,
                      this->cp_stream);
                }
              }
            }
          }
        } else {
          // ==== balanced inter-node relay ====
          // Per round, the node's L union segments form one canonical stream
          // (ascending source local rank; token-ascending interiors) cut into
          // L near-equal chunks by chunk_bound(). Relay rank k stages chunk k
          // (intra-node pieces pushed by the owning sources) and wire-puts it
          // to the same-local-rank gateway on the target node, so every rank's
          // per-round wire bytes are ceil(total / L) instead of U[rank][tn].
          char *relay_base = reinterpret_cast<char *>(this->a2av_relay_stage_.data_ptr());
          uint64_t *relay_sig = reinterpret_cast<uint64_t *>(this->a2av_relay_sig_.data_ptr());
          uint64_t *gw_sig = reinterpret_cast<uint64_t *>(this->a2av_gw_round_sig_.data_ptr());
          // staging offset of the round chunk from source node ns at gateway
          // (gnode, glr): chunks exact-packed ascending by source node,
          // gateway's own node skipped (chunk analogue of stage_off_u)
          auto stage_off_chunk = [&](int gnode, int glr, int ns) -> int64_t {
            int64_t acc = 0;
            for (int n = 0; n < ns; n++) {
              if (n == gnode) {
                continue;
              }
              acc += chunk_rows_of(n, gnode, glr);
            }
            return acc;
          };
          // relay rank k's staging base for round dn: my chunks packed
          // ascending by round (ALL rounds staged at once, see below)
          auto relay_round_base = [&](int k, int dn) -> int64_t {
            int64_t acc = 0;
            for (int d2 = 1; d2 < dn; d2++) {
              acc += chunk_rows_of(my_node, (my_node - d2 + NN) % NN, k);
            }
            return acc;
          };
          // my segment's canonical range and send-buffer base in round dn
          auto my_seg_base = [&](int tn) -> int64_t {
            int seg = tn < my_node ? tn : tn + L - 1;
            return seg_off_h[seg];
          };

          // ---- phase 1: ALL piece transfers for ALL rounds, before any wire
          // wait. DEADLOCK RULE: piece puts depend only on the local pack
          // (ready_event), never on a wait — interleaving them with the wire
          // rounds would create a cross-rank wait cycle. Zero-overlap pairs
          // are skipped entirely on both sides (no signal, no wait).
          for (int dn = 1; dn < NN; dn++) {
            int tn = (my_node - dn + NN) % NN;
            const int64_t sstart = canon_start(my_node, my_lr, tn);
            const int64_t send = sstart + U_at(rank, tn);
            for (int k = 0; k < L; k++) {
              const int64_t a_k = chunk_bound(my_node, tn, k);
              const int64_t b_k = chunk_bound(my_node, tn, k + 1);
              const int64_t lo = std::max(a_k, sstart);
              const int64_t hi = std::min(b_k, send);
              if (hi <= lo) {
                continue;
              }
              if (k == my_lr && a_k >= sstart && b_k <= send) {
                // single-source fast path: my chunk lives entirely in my own
                // segment — the wire loop puts it straight from the send
                // buffer, no staging hop (must mirror own_only below)
                continue;
              }
              char *src = send_base + (my_seg_base(tn) + (lo - sstart)) * row_bytes;
              if (k == my_lr) {
                CUDA_CHECK(cudaMemcpyAsync(
                    relay_base + (relay_round_base(k, dn) + (lo - a_k)) * row_bytes,
                    src,
                    (hi - lo) * row_bytes,
                    cudaMemcpyDeviceToDevice,
                    this->cp_stream_inter_node));
              } else {
                int rk = dist_env.local_rank_to_global_rank(k, my_node);
                nvshmemx_putmem_signal_nbi_on_stream(
                    relay_base + (relay_round_base(k, dn) + (lo - a_k)) * row_bytes,
                    src,
                    (hi - lo) * row_bytes,
                    relay_sig + (dn - 1) * L + my_lr,
                    this->run_id_,
                    NVSHMEM_SIGNAL_SET,
                    rk,
                    this->cp_stream_inter_node);
              }
            }
          }
          // GEMM gate: piece puts are issued (pure local nbi work); the wire
          // loop below contains cross-rank front-end waits and must NOT gate
          // the GEMM launch (forward_impl waits on this instead of
          // fetch_remote_event in relay mode)
          CUDA_CHECK(cudaEventRecord(this->relay_send_event_, this->cp_stream_inter_node));

          // ---- phase 2: wire loop, mirror node order. One contiguous put of
          // my chunk per round; node_sig keeps its single-writer-per-slot
          // semantics (the round's chunk k comes only from relay (ns, k)).
          for (int dn = 1; dn < NN; dn++) {
            int tn = (my_node - dn + NN) % NN;
            int g = dist_env.local_rank_to_global_rank(my_lr, tn);
            const int64_t a_me = chunk_bound(my_node, tn, my_lr);
            const int64_t b_me = chunk_bound(my_node, tn, my_lr + 1);
            const int64_t rows = b_me - a_me;
            if (rows == 0) {
              nvshmemx_signal_op_on_stream(
                  node_sig + my_node,
                  this->run_id_,
                  NVSHMEM_SIGNAL_SET,
                  g,
                  this->cp_stream_inter_node);
              continue;
            }
            const int64_t sstart = canon_start(my_node, my_lr, tn);
            const int64_t send = sstart + U_at(rank, tn);
            const bool own_only = a_me >= sstart && b_me <= send;
            char *wire_src;
            if (own_only) {
              wire_src = send_base + (my_seg_base(tn) + (a_me - sstart)) * row_bytes;
            } else {
              // front-end waits (zero SMs) for every remote contributor of my
              // chunk; the host knows the contributor set from U
              for (int sl = 0; sl < L; sl++) {
                if (sl == my_lr) {
                  continue;
                }
                const int64_t c0 = canon_start(my_node, sl, tn);
                const int64_t c1 = c0 + U_of(my_node, sl, tn);
                if (std::min(b_me, c1) > std::max(a_me, c0)) {
                  CU_CHECK(CUStreamWaitValue64(
                      this->cp_stream_inter_node,
                      reinterpret_cast<CUdeviceptr>(relay_sig + (dn - 1) * L + sl),
                      this->run_id_,
                      CU_STREAM_WAIT_VALUE_GEQ));
                }
              }
              wire_src = relay_base + relay_round_base(my_lr, dn) * row_bytes;
            }
            nvshmemx_putmem_signal_nbi_on_stream(
                stage_base + stage_off_chunk(tn, my_lr, my_node) * row_bytes,
                wire_src,
                rows * row_bytes,
                node_sig + my_node,
                this->run_id_,
                NVSHMEM_SIGNAL_SET,
                g,
                this->cp_stream_inter_node);
          }

          // ---- host sync on the tiny cnt_in/cnt_before D2H: placed AFTER the
          // wire issue and immediately before its only consumer (the gateway
          // loop). The fwd build precedes stage 2 on the main stream, so in
          // steady state this returns almost immediately (precedent: the
          // no-metadata counts sync above).
          CUDA_CHECK(cudaEventSynchronize(this->fwd_cnt_event_));
          const int32_t *cnt_in_h = this->a2av_fwd_cnt_pinned_.data_ptr<int32_t>();
          const int32_t *cnt_bef_h = cnt_in_h + (int64_t)(NN - 1) * L * L;

          // ---- gateway rounds: per-round front-end wait, then gather each
          // (source lr, local dest) piece of my staging window. A window cut
          // inside a source's segment splits its (s, d) recv region across
          // gateways: my slice is contiguous and lands at
          // recv_off_of_u(s, d) + cnt_before (both host-known after the sync).
          CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream, this->fwd_index_event_));
          {
            c10::cuda::CUDAStreamGuard guard(this->cp_stream);
            auto opt_in = torch::TensorOptions(torch::kCUDA).dtype(this->input_dtype);
            for (int dn = 1; dn < NN; dn++) {
              int ns = (my_node + dn) % NN;
              CU_CHECK(CUStreamWaitValue64(
                  this->cp_stream,
                  reinterpret_cast<CUdeviceptr>(node_sig + ns),
                  this->run_id_,
                  CU_STREAM_WAIT_VALUE_GEQ));
              const int64_t win_rows = chunk_rows_of(ns, my_node, my_lr);
              auto stage_seg = torch::from_blob(
                  stage_base + stage_off_chunk(my_node, my_lr, ns) * row_bytes,
                  {std::max<int64_t>(win_rows, 1), (int64_t)hidden},
                  opt_in);
              int64_t sc = 0;  // per-round scratch rows, host running-packed
              for (int dl = 0; dl < L; dl++) {
                int dlg = (my_lr - dl + L) % L;
                int d = dist_env.local_rank_to_global_rank(dlg, my_node);
                int last_sl = -1;
                for (int sl = 0; sl < L; sl++) {
                  if (cnt_in_h[((int64_t)(dn - 1) * L + sl) * L + dlg] > 0) {
                    last_sl = sl;
                  }
                }
                if (last_sl < 0) {
                  // nothing of my window goes to d this round; still signal
                  // (the destination aggregation waits on every gateway slot)
                  nvshmemx_signal_op_on_stream(
                      gw_sig + (dn - 1) * L + my_lr,
                      this->run_id_,
                      NVSHMEM_SIGNAL_SET,
                      d,
                      this->cp_stream);
                  continue;
                }
                for (int sl = 0; sl < L; sl++) {
                  const int64_t cnt = cnt_in_h[((int64_t)(dn - 1) * L + sl) * L + dlg];
                  if (cnt == 0) {
                    continue;
                  }
                  int s = dist_env.local_rank_to_global_rank(sl, ns);
                  auto idx = this->a2av_fwd_idx_.narrow(
                      0, fwd_col_off_h[((size_t)(dn - 1) * L + sl) * L + dlg], cnt);
                  const int64_t dst_off = recv_off_of_u(s, d) +
                                          cnt_bef_h[((int64_t)(dn - 1) * L + sl) * L + dlg];
                  if (d == rank) {
                    auto dst = torch::from_blob(
                        recv_base + dst_off * row_bytes, {cnt, (int64_t)hidden}, opt_in);
                    at::index_select_out(dst, stage_seg, 0, idx);
                  } else {
                    FLUX_CHECK_LE(sc + cnt, copies_per_rank)
                        << "a2av relay forward scratch overflow";
                    auto dst = torch::from_blob(
                        scratch_base + sc * row_bytes, {cnt, (int64_t)hidden}, opt_in);
                    at::index_select_out(dst, stage_seg, 0, idx);
                    if (sl == last_sl) {
                      // NON-nbi (the scratch is refilled next round) with the
                      // per-round gateway signal fused on the LAST piece:
                      // intra-node on-stream puts to the same peer land in
                      // stream order (P2P copies), so the signal covers the
                      // earlier pieces too
                      nvshmemx_putmem_signal_on_stream(
                          recv_base + dst_off * row_bytes,
                          scratch_base + sc * row_bytes,
                          cnt * row_bytes,
                          gw_sig + (dn - 1) * L + my_lr,
                          this->run_id_,
                          NVSHMEM_SIGNAL_SET,
                          d,
                          this->cp_stream);
                    } else {
                      nvshmemx_putmem_on_stream(
                          recv_base + dst_off * row_bytes,
                          scratch_base + sc * row_bytes,
                          cnt * row_bytes,
                          d,
                          this->cp_stream);
                    }
                    sc += cnt;
                  }
                }
                if (d == rank) {
                  nvshmemx_signal_op_on_stream(
                      gw_sig + (dn - 1) * L + my_lr,
                      this->run_id_,
                      NVSHMEM_SIGNAL_SET,
                      rank,
                      this->cp_stream);
                }
              }
            }
          }

          // ---- destination-side signal aggregation (cp_stream_signal, pure
          // front-end memops): a source's (s, d) rows may now arrive via
          // several gateways, so the per-source epoch signals the GEMM spins
          // on get ONE writer again — me. Once all L gateway slots of a round
          // reach the epoch, every source of that node is fully delivered
          // (putmem_signal orders payload before signal), so write signal[s]
          // for ALL its sources, zero-traffic ones included.
          for (int dn = 1; dn < NN; dn++) {
            int ns = (my_node + dn) % NN;
            for (int gl = 0; gl < L; gl++) {
              CU_CHECK(CUStreamWaitValue64(
                  this->cp_stream_signal,
                  reinterpret_cast<CUdeviceptr>(gw_sig + (dn - 1) * L + gl),
                  this->run_id_,
                  CU_STREAM_WAIT_VALUE_GEQ));
            }
            for (int sl = 0; sl < L; sl++) {
              int s = dist_env.local_rank_to_global_rank(sl, ns);
              CU_CHECK(CUStreamWriteValue64(
                  this->cp_stream_signal,
                  reinterpret_cast<CUdeviceptr>(signal_base + s),
                  this->run_id_,
                  CU_STREAM_WRITE_VALUE_DEFAULT));
            }
          }
          CUDA_CHECK(cudaEventRecord(this->signal_done_event_, this->cp_stream_signal));
        }
      }
    } else if (a2av_hier_) {
      // hierarchical dispatch, mirroring all_gather_all2all: intra-node traffic
      // is delivered directly (the ring dn==0 slots below); inter-node traffic
      // travels as ONE aggregated putmem_signal per peer node, addressed to the
      // same-local-rank "gateway" there, which forwards each destination's
      // sub-chunk intra-node once the round's arrival signal shows up. The send
      // buffer is destination-major in GLOBAL rank order and a node's ranks are
      // globally contiguous, so a node's aggregate is a contiguous slice; the
      // forwarded sub-chunks are internally (expert, copy)-ordered and land
      // bit-identically to direct puts — recv layout, stage-2 index math and
      // the dense static problem schedule are all unchanged.
      const int L = dist_env.local_world_size;
      const int NN = dist_env.nnodes;
      const int my_node = dist_env.node_idx;
      const int my_lr = dist_env.local_rank;
      auto node_chunk = [&](int s, int n) -> int64_t {
        int64_t acc = 0;
        for (int d = n * L; d < (n + 1) * L; d++) {
          acc += chunk_at(s, d);
        }
        return acc;
      };
      // staging offset of source node ns's segment at gateway (gnode, glr):
      // segments exact-packed ascending by source node, gateway's own node skipped
      auto seg_off = [&](int gnode, int glr, int ns) -> int64_t {
        int64_t acc = 0;
        for (int n = 0; n < ns; n++) {
          if (n == gnode) {
            continue;
          }
          acc += node_chunk(dist_env.local_rank_to_global_rank(glr, n), gnode);
        }
        return acc;
      };
      // generalizes recv_off[] (the s == rank column) to any source
      auto recv_off_of = [&](int s, int d) -> int64_t {
        int64_t acc = 0;
        for (int s2 = 0; s2 < s; s2++) {
          acc += chunk_at(s2, d);
        }
        return acc;
      };
      if (NN > 1) {
        // staging overflow check before any inter-node wire; every rank
        // evaluates the same expression, so failure is collective (no
        // one-rank-throws-while-others-wait-in-the-barrier hang)
        int64_t max_stage_rows = 0;
        for (int gn = 0; gn < NN; gn++) {
          for (int gl = 0; gl < L; gl++) {
            int64_t rows = 0;
            for (int ns = 0; ns < NN; ns++) {
              if (ns == gn) {
                continue;
              }
              rows += node_chunk(dist_env.local_rank_to_global_rank(gl, ns), gn);
            }
            max_stage_rows = std::max(max_stage_rows, rows);
          }
        }
        FLUX_CHECK_LE(max_stage_rows, this->max_stage_ntokens_)
            << "a2av_hier staging overflow; raise FLUX_A2AV_MAX_STAGE_NTOKENS";
      }
      // round 0: intra-node direct puts, mirror local order (== ring dn==0 slots)
      for (int dl = 1; dl < L; dl++) {
        int d = dist_env.local_rank_to_global_rank((my_lr - dl + L) % L, my_node);
        issue_put(d, this->cp_stream);
      }
      CUDA_CHECK(cudaEventRecord(this->hier_dispatch_event_, this->cp_stream));
      if (NN > 1) {
        uint64_t *node_sig =
            reinterpret_cast<uint64_t *>(this->a2av_node_signal_buffer_.data_ptr());
        char *stage_base = reinterpret_cast<char *>(this->a2av_stage_buffer_.data_ptr());
        // inter-node aggregated sends, mirror node order (receivers see node
        // my_node at stage block (my_node - node_d) mod NN, as the dense
        // schedule expects); arrival signal slot = source node, value = epoch.
        // Empty aggregates still signal — gateways wait on every round.
        for (int dn = 1; dn < NN; dn++) {
          int tn = (my_node - dn + NN) % NN;
          int g = dist_env.local_rank_to_global_rank(my_lr, tn);
          int64_t rows = node_chunk(rank, tn);
          if (rows > 0) {
            nvshmemx_putmem_signal_nbi_on_stream(
                stage_base + seg_off(tn, my_lr, my_node) * row_bytes,
                send_base + send_off[tn * L] * row_bytes,
                rows * row_bytes,
                node_sig + my_node,
                this->run_id_,
                NVSHMEM_SIGNAL_SET,
                g,
                this->cp_stream_inter_node);
          } else {
            nvshmemx_signal_op_on_stream(
                node_sig + my_node,
                this->run_id_,
                NVSHMEM_SIGNAL_SET,
                g,
                this->cp_stream_inter_node);
          }
        }
        // gateway forwarding, rounds ascending; the front-end stream wait
        // (cuStreamWaitValue64, zero SMs — cannot deadlock against the
        // spinning GEMM) is the inter-node-arrival -> intra-node-forward
        // dependency, the a2av analogue of the allgather's fetch_remote_event
        for (int dn = 1; dn < NN; dn++) {
          int ns = (my_node + dn) % NN;
          int s = dist_env.local_rank_to_global_rank(my_lr, ns);
          CU_CHECK(CUStreamWaitValue64(
              this->cp_stream,
              reinterpret_cast<CUdeviceptr>(node_sig + ns),
              this->run_id_,
              CU_STREAM_WAIT_VALUE_GEQ));
          char *seg = stage_base + seg_off(my_node, my_lr, ns) * row_bytes;
          // forward in mirror local order so receiver d sees source s at stage
          // L*dn + ((lr_s - lr_d) mod L); the segment interior is ascending
          // global d, hence the within-segment prefix sum
          for (int dl = 0; dl < L; dl++) {
            int d = dist_env.local_rank_to_global_rank((my_lr - dl + L) % L, my_node);
            int64_t sub_rows = chunk_at(s, d);
            int64_t within = 0;
            for (int d2 = my_node * L; d2 < d; d2++) {
              within += chunk_at(s, d2);
            }
            if (d == rank) {
              // gateway's own sub-chunk: local copy + local signal
              if (sub_rows > 0) {
                CUDA_CHECK(cudaMemcpyAsync(
                    recv_base + recv_off_of(s, rank) * row_bytes,
                    seg + within * row_bytes,
                    sub_rows * row_bytes,
                    cudaMemcpyDeviceToDevice,
                    this->cp_stream));
              }
              nvshmemx_signal_op_on_stream(
                  signal_base + s, this->run_id_, NVSHMEM_SIGNAL_SET, rank, this->cp_stream);
            } else if (sub_rows > 0) {
              nvshmemx_putmem_signal_nbi_on_stream(
                  recv_base + recv_off_of(s, d) * row_bytes,
                  seg + within * row_bytes,
                  sub_rows * row_bytes,
                  signal_base + s,
                  this->run_id_,
                  NVSHMEM_SIGNAL_SET,
                  d,
                  this->cp_stream);
            } else {
              nvshmemx_signal_op_on_stream(
                  signal_base + s, this->run_id_, NVSHMEM_SIGNAL_SET, d, this->cp_stream);
            }
          }
        }
      }
    } else if (!a2av_ring_) {
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
      c10::optional<torch::Tensor> a2av_unique_counts,
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

    // compress dedup counts: cols [0, W) = u[s][d] (unique tokens s -> rank d),
    // cols [W, W + nnodes) = U[s][n] (unique tokens s -> node-n union);
    // identical on all ranks, host-side, untimed metadata (extension of the
    // splits_per_source contract — NOT derivable from cnt, depends on overlap)
    const int32_t *uc_host = nullptr;
    if (a2av_unique_counts.has_value()) {
      auto const &uc = a2av_unique_counts.value();
      FLUX_CHECK(uc.device().is_cpu()) << "a2av_unique_counts must be a CPU tensor";
      FLUX_CHECK(uc.scalar_type() == torch::kInt32);
      FLUX_CHECK(uc.is_contiguous());
      CHECK_2D(uc, world_size, world_size + this->nnodes);
      uc_host = uc.data_ptr<int32_t>();
    }
    if (a2av_hier_compress_) {
      FLUX_CHECK(cnt_host != nullptr && uc_host != nullptr)
          << "a2av_hier_compress requires splits_per_source AND a2av_unique_counts";
      // the gateway gathers (index_select) need SMs while the GEMM tiles spin
      // on signals only those gathers can produce — a full-occupancy GEMM
      // would deadlock. Reserve at least one SM for the copy engine work.
      FLUX_CHECK(sm_margin >= 1 || nnodes == 1)
          << "a2av_hier_compress with nnodes > 1 requires sm_margin >= 1";
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
          this->a2av_dispatch(inputs_shard, splits_gpu, scatter_index, cnt_host, uc_host, stream);
      sorted_gather_index = a2av_state.sorted_gather_index;
      sorted_scatter_index = a2av_state.sorted_scatter_index;
      sorted_splits_cumsum = a2av_state.sorted_splits_cumsum;
      M_this_ep = a2av_state.M_this_ep;
      if (a2av_ring_ || a2av_hier_ || a2av_hier_compress_) {
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
      args.a2av_ring_schedule = a2av_ring_ || a2av_hier_ || a2av_hier_compress_;
    }
    for (int gid = 0; gid < num_weights_group; gid++) {
      args.weight[gid] = weights[gid].data_ptr();
      args.output[gid] = outputs[gid].data_ptr();
      args.scaleD[gid] =
          output_scales.has_value() ? output_scales->at(gid).data_ptr<float>() : nullptr;
    }

    static const bool kA2avTiming = get_int_from_env("FLUX_A2AV_TIMING", 0) != 0;
    if (a2av_dispatch_) {
      if (a2av_hier_ || a2av_hier_compress_) {
        // gate only on round-0 intra puts + inter-node sends being ISSUED; the
        // gateway forwarding proceeds concurrently with the GEMM, whose tiles
        // spin on the per-source signals. hier forwards with front-end waits +
        // CE puts (zero SMs); compress additionally runs index_select gathers,
        // which need the SMs kept free by the enforced sm_margin >= 1
        CUDA_CHECK(cudaStreamWaitEvent(stream, this->hier_dispatch_event_));
        if (a2av_hier_compress_ && nnodes > 1 && !relay_identity_) {
          // balanced relay: the wire loop carries cross-rank front-end waits,
          // so fetch_remote_event would gate the GEMM on peers' packs; gate on
          // the relay piece puts instead (pure local nbi work)
          CUDA_CHECK(cudaStreamWaitEvent(stream, this->relay_send_event_));
        } else {
          CUDA_CHECK(cudaStreamWaitEvent(stream, this->fetch_remote_event));
        }
      } else {
        // do not start the (SM-occupying) GEMM before all puts/signals are issued;
        // in ring mode the intra-node puts live on cp_stream, which all_gather_event
        // covers (it is recorded after cp_stream waits on fetch_remote_event)
        CUDA_CHECK(cudaStreamWaitEvent(
            stream, a2av_ring_ ? this->all_gather_event : this->fetch_remote_event));
      }
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
    if (a2av_hier_compress_ && nnodes > 1 && !relay_identity_) {
      // fold the signal-aggregation stream into the epoch: its front-end ops
      // must complete before the barrier closes this iteration (the GEMM
      // transitively drains it, but keep the ordering explicit)
      CUDA_CHECK(cudaStreamWaitEvent(stream, this->signal_done_event_));
    }
    if (nnodes > 1 || a2av_dispatch_) {
      // ensure that when the next time each rank copy data to itself's shard in the
      // input_buffer, all ranks have already finished allgather so that we can
      // safely modify input_buffer. In a2av mode this barrier additionally quiets
      // our outstanding nbi puts and keeps iteration n+1 puts from racing
      // iteration n's GEMM reads of the recv buffer. In hier mode the same
      // argument covers the staging buffer and node arrival signals: all
      // forwarding reads are enqueued before all_gather_event (waited above),
      // and iteration n+1 sends wait on ready_event, recorded after this barrier.
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
      if (a2av_hier_compress_ && nnodes > 1 && !relay_identity_) {
        float rf[kNumRelayFwdEvents - 1];
        for (int i = 0; i < kNumRelayFwdEvents - 1; i++) {
          CUDA_CHECK(cudaEventElapsedTime(
              &rf[i], this->relay_fwd_events_[i], this->relay_fwd_events_[i + 1]));
        }
        fprintf(
            stderr,
            "[a2av-relayfwd] rank %d dl %.3f flag %.3f canon %.3f mask %.3f valid %.3f "
            "cumsum %.3f tgt %.3f flatten %.3f scatter %.3f cnts %.3f d2h %.3f ms\n",
            rank,
            rf[0],
            rf[1],
            rf[2],
            rf[3],
            rf[4],
            rf[5],
            rf[6],
            rf[7],
            rf[8],
            rf[9],
            rf[10]);
      }
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
      c10::optional<torch::Tensor> splits_per_source,
      c10::optional<torch::Tensor> a2av_unique_counts) {
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
        std::move(a2av_unique_counts),
        c10::nullopt);
  }

  void
  clear_buffers() {
    if (nnodes > 1 && this->input_buffer.defined()) {
      this->input_buffer.zero_();
    }
    // a2av signal buffer (and the hier node arrival signals, the relay-in
    // signals and the per-round gateway signals) are deliberately NOT cleared:
    // the epoch scheme relies on monotonically increasing signal values and
    // clearing would corrupt in-flight iterations. Data buffers need no
    // clearing (rows fully overwritten per use).
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
      c10::optional<torch::Tensor> splits_per_source,
      c10::optional<torch::Tensor> a2av_unique_counts) {
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
        std::move(a2av_unique_counts),
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
    bool a2av_ring,
    bool a2av_hier,
    bool a2av_hier_compress)
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
          a2av_ring,
          a2av_hier,
          a2av_hier_compress)) {}
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
    c10::optional<torch::Tensor> splits_per_source,
    c10::optional<torch::Tensor> a2av_unique_counts) {
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
      std::move(splits_per_source),
      std::move(a2av_unique_counts));
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
    c10::optional<torch::Tensor> splits_per_source,
    c10::optional<torch::Tensor> a2av_unique_counts) {
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
      std::move(splits_per_source),
      std::move(a2av_unique_counts));
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
