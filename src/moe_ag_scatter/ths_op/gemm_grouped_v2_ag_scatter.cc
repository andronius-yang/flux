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
#include "moe_ag_scatter/epic_swap.hpp"
#include "moe_ag_scatter/sort_util.h"
#include "moe_ag_scatter/triton_util.h"
#include "moe_ag_scatter/workspace_util.h"
#include "moe_ag_scatter/ths_op/a2av_nvtx_proxy.hpp"
#include "flux/a2av_progress.h"
#include <nvshmemx.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
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
  // FLUX_A2AV_PACK_OVERLAP=1 (compress only): iteration n+1's producer pack
  // (meta H2D + stage1 + pack scan + send gather) runs here so it overlaps
  // iteration n's GEMM on the main stream. Requires parity double-buffering
  // of the send buffer and meta arena (run_id_ & 1). The 2x symmetric send
  // allocation is collective, so the env must be set identically on every
  // rank. Wall-clock no-op when CUDA_DEVICE_MAX_CONNECTIONS=1 (single hw
  // queue) or FLUX_A2AV_TIMING=1 (per-iteration host sync); correctness is
  // unaffected either way. Contract: forward() inputs must be device-ready
  // when called (this stream does not join the caller stream's tail).
  c10::cuda::CUDAStream pack_stream_;
  // FLUX_A2AV_NVTX_PROXY only: carries the 1-block progress-mirror kernel. A
  // dedicated stream because the mirror stays resident for the whole GEMM —
  // on any comm stream it would serialize that stream's memops behind itself
  // (deadlock against relay signal chains).
  c10::cuda::CUDAStream nvtx_proxy_stream_;
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
  // FLUX_A2AV_UNION_BCAST=1 (compress, nnodes > 1): the gateway forwards the
  // WHOLE staged union to every local rank (contiguous nbi puts straight from
  // the symmetric staging buffer — no gather, no scratch, no sm_margin
  // requirement) and each receiver aliases its subset out of the U-sized
  // region through the consumer index. Implies the identity wire (ORs into
  // relay_identity_). Changes the RECV layout (remote-source regions hold
  // U[s][n] rows, not u[s][d]), so it must be set identically on every rank.
  const bool union_bcast_;
  // FLUX_A2AV_PACK_OVERLAP=1 (compress): see pack_stream_. Must be set
  // identically on every rank (collective 2x symmetric send allocation).
  const bool pack_overlap_;
  // FLUX_A2AV_EARLY_LAUNCH=1 (any a2av variant; since 2026-08-16 the default
  // is ON under FLUX_A2AV_LB_UNION=1 when CUDA_DEVICE_MAX_CONNECTIONS > 1 —
  // see the ctor initializer): issue the inter-node sends
  // first, launch the GEMM right after stage 2, and defer the SM-free
  // cp_stream wire sequence (self copy, round-0 intra puts, hier/union
  // gateway forwarding) until after the launch; cp_stream FIFO order is
  // preserved by deferring the whole sequence, never a prefix. On
  // hier/union/flat/ring the deferred ops are strictly CE memcpys + front-end
  // memops + nbi CE puts, legal even at CUDA_DEVICE_MAX_CONNECTIONS=1. The
  // compress gather/relay arms' index_select tails are NOT deferred — a
  // kernel enqueued after the persistent GEMM blankets every SM can starve at
  // dispatch forever — but issued inline on the idle pack stream behind
  // front-end waits (pre-launch enqueue, concurrent execution: the proven
  // non-early configuration). Those arms require CUDA_DEVICE_MAX_CONNECTIONS
  // > 1 (ctor-checked) so their pre-launch front-end waits cannot serialize
  // the GEMM launch behind full wire delivery in a single channel.
  const bool early_launch_;
  // FLUX_A2AV_BLOCKING_WIRE=1 (instrumented only): inter-node puts (hier /
  // compress aggregates, relay phase-2 wire, flat per-dest remote puts) use
  // the blocking-local put, whose proxy entrypoint kernel spans until local
  // completion — the wire becomes a visible device span. Under
  // CUDA_DEVICE_MAX_CONNECTIONS=1 that kernel serializes ahead of the GEMM;
  // for overlap visualization raise the env (launch.sh default is :-1).
  const bool blocking_wire_;
  // 2026-08-22 relay-pull ordering fix + diagnostics (cross-iteration stale
  // delivery under per-call-changing metadata; plan eager-juggling-glacier
  // Stage 1). relay_fence_: quiet on cp_stream_inter_node between the
  // relay phase-1 nbi gets and the phase-2 wire put (the F1 fix).
  // relay_poison_/relay_blocking_pull_/epoch_quiet_: mechanism-isolation
  // toggles (T2/T3/T4), never set in campaign specs.
  const bool relay_fence_;
  const bool relay_poison_;
  const bool relay_blocking_pull_;
  const bool epoch_quiet_;
  const bool wire_sig_fence_;
  const bool wait_flush_;  // F3 (2026-08-22): CU_STREAM_WAIT_VALUE_FLUSH on every
  // front-end signal wait. GPUDirect-RDMA writes that reached the device
  // before the flag are NOT guaranteed visible to downstream device work
  // without the flush (cuStreamWaitValue64 driver doc); the raw GEQ poll on
  // node_sig let the gateway forward the previous epoch's stage.
  const bool nvshmem_wait_;  // F4 (2026-08-22): gate RDMA-delivered node_sig with
  // nvshmemx_signal_wait_until_on_stream (NVSHMEM enforces GPUDirect-RDMA
  // consistency via its proxy flush) instead of a raw CUStreamWaitValue64
  // poll — the pattern gather_rs already uses for its inter-node signals.
  // A100@Perlmutter reports cudaDevAttrCanFlushRemoteWrites=0, so the
  // CU_STREAM_WAIT_VALUE_FLUSH route (wait_flush_) is unavailable here.
  unsigned int
  a2av_wait_flags() const {
    return CU_STREAM_WAIT_VALUE_GEQ | (wait_flush_ ? CU_STREAM_WAIT_VALUE_FLUSH : 0u);
  }  // F2 (2026-08-22): wire data as putmem_nbi, ONE
  // quiet, THEN the node_sig signal ops — never rely on put_signal
  // data-before-signal ordering on the libfabric/CXI wire (one-epoch-stale
  // gateway forwards under per-iteration payload change)
  // FLUX_A2AV_FANOUT=1 (lb_union Tier B only; NR-06 re-check 2026-08-07):
  // eager per-round gateway forwards — round dn's node_sig wait + window puts
  // enqueue on fanout_streams_[dn-1] instead of the single tail stream, so a
  // late round never head-of-line blocks a later round whose relay chunk
  // already landed. Rounds are re-joined into the tail stream via
  // fanout_events_ right after the loop, so the existing all_gather_event /
  // iteration-barrier structure covers the eager arm unchanged. Knob off
  // (default) keeps the shipped ring order byte-identical.
  const bool fanout_eager_;
  std::vector<c10::cuda::CUDAStream> fanout_streams_;  // NN-1, knob on && nnodes > 1
  std::vector<cudaEvent_t> fanout_events_;             // NN-1, DisableTiming
  // FLUX_A2AV_FUSED_STAGE2=1 (compress only; since 2026-08-16 the default is
  // ON under FLUX_A2AV_LB_UNION=1 — see the ctor initializer): replace the
  // ATen consumer-build
  // chain (key/argsort/index_select + Tier B gating searchsorted) with the
  // fused kernels in sort_util — A rows assigned per (expert, source) group
  // via the host offA table + an atomic in-group rank (interior order is
  // arbitrary; no consumer observes it), gather = dedup recv row from the
  // mine-token cumsum, gating lanes histogrammed sort-free. Cuts the
  // launch-blocking stage-2 chain to ~3 kernels + 1 cumsum.
  const bool fused_stage2_;
  // FLUX_A2AV_SEG_GATE_BALLOT=1: legacy two-ballot (W<=64) process_tile
  // segment gate instead of the default W-unbounded predicate gate (A/B knob)
  const bool seg_gate_ballot_;
  uint64_t run_id_ = 0;              // epoch value carried by the NVSHMEM signals
  int64_t max_recv_ntokens_ = 0;     // rows of the symmetric recv buffer
  int64_t max_stage_ntokens_ = 0;    // rows of the symmetric gateway staging buffer
  torch::Tensor a2av_send_buffer;    // symmetric [tokens_per_rank_max * topk, hidden]
  torch::Tensor a2av_recv_buffer;    // symmetric [max_recv_ntokens_, hidden]
  torch::Tensor a2av_signal_buffer;  // symmetric uint64[world_size], never memset
  // a2av_hier only (nnodes > 1): staging area for inbound node-aggregated
  // payloads, plus per-source-node arrival signals (epoch discipline, never memset)
  torch::Tensor a2av_stage_buffer_;            // symmetric [max_stage_ntokens_, hidden]
  torch::Tensor a2av_node_signal_buffer_;      // symmetric uint64[nnodes]
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
  torch::Tensor a2av_meta_pinned_;  // pinned bytes (x2 stride when pack_overlap_)
  torch::Tensor a2av_meta_dev_;     // device bytes, same layout
  int64_t meta_stride_ = 0;         // bytes of one meta arena parity slice
  int64_t send_half_rows_ = 0;      // rows of one send-buffer parity half
  // gates the put loop on the 1 KB counts D2H / the meta H2D; indexed by the
  // pack parity when pack_overlap_ (index 0 otherwise)
  cudaEvent_t counts_event_[2] = {};
  // pack overlap only: pack inputs (e_all & friends) free for the next
  // iteration (recorded after build_stage2), and end-of-iteration barrier
  // done (carries the cross-iteration put ordering the main-stream program
  // order used to provide)
  cudaEvent_t pack_inputs_free_ = nullptr;
  cudaEvent_t barrier_done_event_ = nullptr;
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
  torch::Tensor a2av_relay_stage_;           // symmetric [max_relay_ntokens_, hidden]
  torch::Tensor a2av_gw_round_sig_;          // symmetric u64[(NN-1)*L], slot (dn-1)*L + gw_lr
  // balanced relay phase 1 is PULL (A/B vs serial/push 2026-08-03, capsule
  // 20260803-150832: pull wins isolated max-rank e2e by 3.7%, significant):
  // symmetric u64[L], slot = source local rank, SET to run_id_ right after
  // the pack — the only cross-rank dependency of a relay's gets
  torch::Tensor a2av_pack_ready_sig_;
  // Tier B (lb_union): per-(expert, gating-lane) inclusive cumsum [E, W]
  // where the remote node's lanes are WINDOW boundaries (data-dependent under
  // dedup, hence device-computed) and local lanes keep source boundaries.
  // Fed to args.accum_per_rank_ptr so the bucket build and the per-tile spin
  // ballot are window-keyed with zero kernel changes.
  torch::Tensor a2av_gating_cumsum_;
  // fused_stage2_ only: persistent consumer-build outputs and the per-group
  // atomic rank counters (+ gating histogram in the second half)
  torch::Tensor a2av_sorted_gather_;   // i32 [n_copies_max]
  torch::Tensor a2av_sorted_scatter_;  // i32 [n_copies_max]
  torch::Tensor a2av_blk_cnt_;         // i32 [2 * E * W]: blk_cnt | gate_hist
  // Tier B sidecar ground truth: per-slot window rows (remote slots only),
  // filled in the meta block where the chunk lambdas are in scope
  std::vector<uint32_t> nvtx_window_rows_;
  torch::Tensor a2av_fwd_cnt_pinned_;        // pinned i32 [2, NN-1, L, L]: cnt_in / cnt_before
  int64_t max_relay_ntokens_ = 0;            // rows of the symmetric relay staging buffer
  cudaEvent_t fwd_cnt_event_ = nullptr;      // cnt_in/cnt_before D2H done (host reads)
  cudaEvent_t relay_send_event_ = nullptr;   // relay piece puts issued (GEMM gate)
  cudaEvent_t signal_done_event_ = nullptr;  // dest-side signal aggregation issued
  // EPIC §4.3 in-kernel expert swap (dispatch_only only, sequential phase 0):
  // FLUX_A2AV_INKERNEL_SWAP = exchange-scratch BYTES (fc1+fc2 of one expert
  // slot; 0 = disabled; read in the ctor, set SPMD-identically). One swap per
  // rank per call max (the EPIC planner's heaviest/lightest pairing
  // invariant). Scratch/flag are symmetric (ctor-allocated — nvshmem_malloc
  // is collective); the arrival counter and stamps are LOCAL. The swap epoch
  // is its own monotone sequence (caller-supplied, checked here), distinct
  // from run_id_. Scratch reuse across epochs is fenced by dispatch_only's
  // two end-of-call barrier_all (see epic_swap.cu header comment).
  int64_t inkernel_swap_bytes_ = 0;
  torch::Tensor swap_scratch_;        // symmetric byte[inkernel_swap_bytes_]
  torch::Tensor swap_flag_;           // symmetric u64[1], zero-init, never memset
  torch::Tensor swap_arrive_;         // LOCAL u64[1], zeroed once, never reset
  torch::Tensor swap_stamps_;         // LOCAL u64[4] globaltimer stamps
  torch::Tensor swap_stamps_pinned_;  // pinned mirror for the timing readback
  uint64_t swap_epoch_seen_ = 0;
  unsigned long long swap_arrive_base_ = 0;
  // always-on swap timing: event pairs recorded around each swap launch;
  // collect_swap_times() drains them after the timed loop
  static constexpr int kSwapEventPool = 256;
  std::vector<cudaEvent_t> swap_events_;  // 2 * kSwapEventPool, lazy-none when disabled
  int swap_events_used_ = 0;
  // FLUX_A2AV_TIMING=1 diagnostics: per-forward segment boundaries on the main stream
  static constexpr int kNumTimingEvents = 6;
  cudaEvent_t timing_events_[kNumTimingEvents] = {};
  static constexpr int kNumStage2Events = 11;
  cudaEvent_t stage2_events_[kNumStage2Events] = {};
  // FLUX_A2AV_TIMING=1, balanced-relay fwd-index build only: op-group boundaries
  static constexpr int kNumRelayFwdEvents = 12;
  cudaEvent_t relay_fwd_events_[kNumRelayFwdEvents] = {};
  // FLUX_A2AV_NVTX_PROXY=1: host-mapped per-source tile-progress slots and the
  // poller thread that renders them as NVTX ranges (a2av_nvtx_proxy.hpp).
  // Progress writes are read-mostly additions; the data path is unchanged.
  bool nvtx_proxy_enabled_ = false;
  A2AVProgressSlots *progress_slots_ = nullptr;  // pinned host copy the poller reads
  // kernel-written slots live in device memory (cheap L2 atomics); the poller
  // refreshes the pinned copy with ~800 B CE memcpys on nvtx_proxy_stream_.
  // GEMM CTAs never touch PCIe, and no SM-resident helper exists — a mirror
  // kernel would deadlock under CUDA_DEVICE_MAX_CONNECTIONS=1 (single compute
  // queue: resident helper blocks the GEMM whose completion would retire it).
  A2AVProgressSlots *progress_slots_dev_ = nullptr;
  // layer C per-tile trace ring (device; lazily sized at first launch when N /
  // tile shape are known). Extraction and sidecar writing live in the poller.
  A2AVTileRecord *tile_trace_dev_ = nullptr;
  uint32_t tile_trace_capacity_ = 0;
  std::unique_ptr<A2AVNvtxProxy> nvtx_proxy_;
  // meta-path [E, W] logical splits cumsum snapshot; drives the dense-schedule
  // per-source expected-tile totals (empty when the meta path didn't run)
  std::vector<int32_t> nvtx_ssc_;

  // FLUX_A2AV_EARLY_LAUNCH: the cp_stream wire sequence, materialized as
  // plain descriptors at dispatch time (all offset math is host data) and
  // issued after the GEMM launch. Descriptors, not closures: the dispatch
  // helpers capture stack state by reference and would dangle.
  struct DeferredWireOp {
    // Deliberately SM-free kinds only (CE copies, nbi CE puts, front-end
    // memops, event records): deferred ops are enqueued AFTER the persistent
    // GEMM launch, and a kernel enqueued once the GEMM has blanketed every SM
    // can starve at dispatch forever (observed on the relay gateway gathers —
    // see the t_* tail ops in a2av_dispatch, which are issued inline on the
    // pack stream instead).
    enum Kind : uint8_t { kSelfCopy, kSignal, kPut, kWait64, kRecordHierEvent };
    Kind kind;
    void *dst = nullptr;
    const void *src = nullptr;
    int64_t bytes = 0;
    uint64_t *sig = nullptr;
    uint64_t val = 0;
    int pe = 0;
  };
  std::vector<DeferredWireOp> deferred_wire_;
  bool deferred_wire_armed_ = false;

  // Execute the deferred sequence on cp_stream (original FIFO order), then
  // re-record the tail events the iteration barrier depends on (they must
  // cover the deferred puts, so they cannot stay at their dispatch-time site).
  void
  issue_deferred_wire() {
    if (!this->deferred_wire_armed_) {
      return;
    }
    // keep the poller's stream syncs out of this window (see set_paused)
    if (this->nvtx_proxy_) {
      this->nvtx_proxy_->set_paused(true);
    }
    for (const DeferredWireOp &op : this->deferred_wire_) {
      switch (op.kind) {
        case DeferredWireOp::kSelfCopy:
          CUDA_CHECK(cudaMemcpyAsync(
              op.dst, op.src, op.bytes, cudaMemcpyDeviceToDevice, this->cp_stream));
          break;
        case DeferredWireOp::kSignal:
          nvshmemx_signal_op_on_stream(op.sig, op.val, NVSHMEM_SIGNAL_SET, op.pe, this->cp_stream);
          break;
        case DeferredWireOp::kPut:
          nvshmemx_putmem_signal_nbi_on_stream(
              op.dst,
              op.src,
              op.bytes,
              op.sig,
              op.val,
              NVSHMEM_SIGNAL_SET,
              op.pe,
              this->cp_stream);
          break;
        case DeferredWireOp::kWait64:
          CU_CHECK(CUStreamWaitValue64(
              this->cp_stream,
              reinterpret_cast<CUdeviceptr>(op.sig),
              op.val,
              this->a2av_wait_flags()));
          break;
        case DeferredWireOp::kRecordHierEvent:
          CUDA_CHECK(cudaEventRecord(this->hier_dispatch_event_, this->cp_stream));
          break;
      }
    }
    CUDA_CHECK(cudaEventRecord(this->fetch_remote_event, this->cp_stream_inter_node));
    CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream, this->fetch_remote_event));
    CUDA_CHECK(cudaEventRecord(this->all_gather_event, this->cp_stream));
    this->deferred_wire_.clear();
    this->deferred_wire_armed_ = false;
    if (this->nvtx_proxy_) {
      this->nvtx_proxy_->set_paused(false);
    }
  }

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
        pack_stream_(create_cp_stream()),
        nvtx_proxy_stream_(create_cp_stream()),
        a2av_dispatch_(a2av_dispatch),
        a2av_ring_(a2av_ring),
        a2av_hier_(a2av_hier),
        a2av_hier_compress_(a2av_hier_compress),
        relay_identity_(
            get_int_from_env("FLUX_A2AV_RELAY_IDENTITY", 0) != 0 ||
            get_int_from_env("FLUX_A2AV_UNION_BCAST", 0) != 0),
        union_bcast_(
            get_int_from_env("FLUX_A2AV_UNION_BCAST", 0) != 0 ||
            get_int_from_env("FLUX_A2AV_LB_UNION", 0) != 0),
        pack_overlap_(get_int_from_env("FLUX_A2AV_PACK_OVERLAP", 0) != 0),
        // CANONICALIZED 2026-08-16 (layer-axis campaign, handoff 07 §2): under
        // FLUX_A2AV_LB_UNION=1 the factorial winners EARLY_LAUNCH and
        // FUSED_STAGE2 default ON (three-run sign agreement: E -0.3..-1.8 ms,
        // F -0.2..-0.7 ms on real 4n trace routing). An explicit env setting
        // always wins over the lb_union default. EARLY_LAUNCH's default
        // additionally requires CUDA_DEVICE_MAX_CONNECTIONS > 1 so a bare
        // launch.sh run (conn=1 default) keeps its historical behavior instead
        // of tripping the compress-path conn>1 ctor check below.
        early_launch_(
            get_int_from_env(
                "FLUX_A2AV_EARLY_LAUNCH",
                (get_int_from_env("FLUX_A2AV_LB_UNION", 0) != 0 &&
                 get_int_from_env("CUDA_DEVICE_MAX_CONNECTIONS", 1) > 1)
                    ? 1
                    : 0) != 0 &&
            a2av_dispatch),
        blocking_wire_(get_int_from_env("FLUX_A2AV_BLOCKING_WIRE", 0) != 0),
        relay_fence_(get_int_from_env("FLUX_A2AV_RELAY_FENCE", 0) != 0),
        relay_poison_(get_int_from_env("FLUX_A2AV_RELAY_POISON", 0) != 0),
        relay_blocking_pull_(get_int_from_env("FLUX_A2AV_RELAY_BLOCKING_PULL", 0) != 0),
        epoch_quiet_(get_int_from_env("FLUX_A2AV_EPOCH_QUIET", 0) != 0),
        wire_sig_fence_(get_int_from_env("FLUX_A2AV_WIRE_SIGNAL_FENCE", 0) != 0),
        wait_flush_(get_int_from_env("FLUX_A2AV_WAIT_FLUSH", 0) != 0),
        nvshmem_wait_(get_int_from_env("FLUX_A2AV_NVSHMEM_WAIT", 0) != 0),
        fanout_eager_(get_int_from_env("FLUX_A2AV_FANOUT", 0) != 0),
        fused_stage2_(
            get_int_from_env(
                "FLUX_A2AV_FUSED_STAGE2", get_int_from_env("FLUX_A2AV_LB_UNION", 0) != 0 ? 1 : 0) !=
                0 &&
            a2av_hier_compress),
        seg_gate_ballot_(get_int_from_env("FLUX_A2AV_SEG_GATE_BALLOT", 0) != 0),
        // ring_mode barriers are CUDA-IPC based and intra-node only; multi-node
        // must take the NVSHMEM barrier (ring_mode = false)
        group_barrier(this->tp_group, nnodes == 1 && this->tp_group->get_size() > 8) {
    _ensure_topo_initialized();
    CHECK_DIV(nexperts, ep_size);
    CHECK_DIV(ffn_hidden, ffn_tp_size);
    FLUX_CHECK_GE(nnodes, 1);
    CHECK_DIV(world_size, nnodes);
    // the default predicate segment gate is W-unbounded; the remaining caps
    // are the progress/trace instrumentation buckets (kMaxBuckets = 129 →
    // W<=128) and, per mode, the legacy ballot gate (FLUX_A2AV_SEG_GATE_BALLOT=1,
    // W<=64) and the dynamic claimer's 64-bit multi-source masks (W<=64).
    if (seg_gate_ballot_) {
      FLUX_CHECK_LE(world_size, 64) << "ballot segment gate caps world_size at 64";
    } else {
      FLUX_CHECK_LE(world_size, 128) << "progress buckets cap world_size at 128";
    }
    if (a2av_dispatch && !(a2av_ring || a2av_hier || a2av_hier_compress)) {
      FLUX_CHECK_LE(world_size, 64) << "dynamic-claimer multi-source masks cap world_size at 64";
    }
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
      // FLUX_A2AV_LB_UNION: balanced chunked wire + union-broadcast gateway
      // (union_bcast_ true, relay_identity_ false). The other two knobs pick
      // fixed points of that pair — combining them would silently collapse
      // lb_union into plain union; hard-error instead.
      const bool lb_union = get_int_from_env("FLUX_A2AV_LB_UNION", 0) != 0;
      FLUX_CHECK(!(lb_union && get_int_from_env("FLUX_A2AV_RELAY_IDENTITY", 0) != 0))
          << "FLUX_A2AV_LB_UNION and FLUX_A2AV_RELAY_IDENTITY are mutually exclusive";
      FLUX_CHECK(!(lb_union && get_int_from_env("FLUX_A2AV_UNION_BCAST", 0) != 0))
          << "FLUX_A2AV_LB_UNION and FLUX_A2AV_UNION_BCAST are mutually exclusive";
      FLUX_CHECK(!(lb_union && this->pack_overlap_))
          << "FLUX_A2AV_LB_UNION + FLUX_A2AV_PACK_OVERLAP is unsupported";
      FLUX_CHECK(!this->fanout_eager_ || lb_union)
          << "FLUX_A2AV_FANOUT requires FLUX_A2AV_LB_UNION (eager Tier B gateway forward)";
      if (this->fanout_eager_ && nnodes > 1) {
        for (int i = 0; i < nnodes - 1; i++) {
          this->fanout_streams_.push_back(create_cp_stream());
          cudaEvent_t ev = nullptr;
          CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
          this->fanout_events_.push_back(ev);
        }
      }
      FLUX_CHECK(!(this->early_launch_ && this->pack_overlap_))
          << "FLUX_A2AV_EARLY_LAUNCH + FLUX_A2AV_PACK_OVERLAP is untested; unset one";
      if (this->early_launch_ && a2av_hier_compress &&
          !(this->relay_identity_ && this->union_bcast_) && nnodes > 1) {
        // the gather/relay tails are issued inline (pack stream) behind
        // front-end waits BEFORE the GEMM launch; in a single hardware
        // channel those pending waits would serialize the GEMM launch behind
        // full wire delivery, defeating the reorder
        FLUX_CHECK(get_int_from_env("CUDA_DEVICE_MAX_CONNECTIONS", 1) > 1)
            << "FLUX_A2AV_EARLY_LAUNCH on the compress gather/relay paths requires "
               "CUDA_DEVICE_MAX_CONNECTIONS > 1";
      }
      FLUX_CHECK_EQ(this->ffn_tp_size, 1) << "a2av dispatch requires ep_size == world_size";
      FLUX_CHECK(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE) == dist_env.local_rank);
      FLUX_CHECK(!this->pack_overlap_ || a2av_hier_compress)
          << "FLUX_A2AV_PACK_OVERLAP requires a2av_hier_compress";
      // STAGE2_AFTER_PUTS moves build_stage2 behind the puts, which would
      // release the pack inputs after the next iteration's pack already ran
      FLUX_CHECK(!(this->pack_overlap_ && get_int_from_env("FLUX_A2AV_STAGE2_AFTER_PUTS", 0) != 0))
          << "FLUX_A2AV_PACK_OVERLAP and FLUX_A2AV_STAGE2_AFTER_PUTS are mutually exclusive";
      int64_t tokens_per_rank_max = (max_ntokens + world_size - 1) / world_size;
      // default recv capacity: 2x the balanced per-rank load (capped at the total);
      // very skewed routings need FLUX_A2AV_MAX_RECV_NTOKENS
      this->max_recv_ntokens_ = get_int_from_env(
          "FLUX_A2AV_MAX_RECV_NTOKENS",
          (int)std::min<int64_t>((int64_t)max_ntokens * topk, tokens_per_rank_max * topk * 2));
      // pack_overlap: two parity halves so iteration n+1's pack can write
      // while iteration n's puts still read (quieted only by the barrier)
      this->send_half_rows_ = tokens_per_rank_max * topk;
      this->a2av_send_buffer = nvshmem_create_tensor(
          {this->send_half_rows_ * (this->pack_overlap_ ? 2 : 1), hidden}, input_dtype);
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
            (int)std::min<int64_t>((int64_t)max_ntokens * topk, tokens_per_rank_max * topk * 2));
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
              (int)std::min<int64_t>((int64_t)max_ntokens * topk, tokens_per_rank_max * topk * 2));
          this->a2av_relay_stage_ =
              nvshmem_create_tensor({this->max_relay_ntokens_, hidden}, input_dtype);
          this->a2av_gw_round_sig_ = nvshmem_create_tensor(
              {(int64_t)(nnodes - 1) * L}, at::ScalarType::Long, /*init_zero=*/true);
          if (!this->union_bcast_) {
            // gather gateway only: lb_union forwards whole windows, host-addressed
            this->a2av_fwd_cnt_pinned_ = torch::empty(
                {2, (int64_t)nnodes - 1, L, L},
                torch::TensorOptions(torch::kCPU).dtype(torch::kInt).pinned_memory(true));
          }
          this->a2av_pack_ready_sig_ =
              nvshmem_create_tensor({L}, at::ScalarType::Long, /*init_zero=*/true);
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
      this->a2av_not_mine_ =
          torch::empty({n_copies_max}, torch::TensorOptions(torch::kCUDA).dtype(torch::kBool));
      this->a2av_expert_base_ = torch::empty({nexperts}, opt_cuda_i64);
      this->a2av_chunks_gpu_ = torch::empty(
          {(int64_t)world_size * world_size},
          torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));
      this->a2av_pack_key_ = torch::empty({tokens_per_rank_max * (int64_t)topk}, opt_cuda_i64);
      const int64_t meta_groups = (int64_t)this->ep_nexperts * world_size;
      const int64_t meta_bytes = 3 * meta_groups * sizeof(int64_t) + nexperts * sizeof(int64_t) +
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
        // union bcast forwards whole unions: no forward-index tables; the Tier B
        // gating searchsorted queries i64[E * (W + 1)] ride the upload instead
        const int64_t extra = this->union_bcast_
                                  ? (int64_t)this->ep_nexperts * (world_size + 1)
                                  : (this->relay_identity_ ? R * L : R * L * L + R * L + 2 * R);
        this->compress_meta_off_ = pad_to(meta_bytes, (int64_t)8);
        total_meta_bytes =
            this->compress_meta_off_ + (nseg + 1 + extra) * (int64_t)sizeof(int64_t);
      }
      // pack_overlap: two parity slices — the GEMM reads its slice (ssc_dev /
      // accum_per_rank_ptr) for its whole runtime while the next pack uploads
      this->meta_stride_ = total_meta_bytes;
      const int64_t meta_par = this->pack_overlap_ ? 2 : 1;
      this->a2av_meta_pinned_ = torch::empty(
          {total_meta_bytes * meta_par},
          torch::TensorOptions(torch::kCPU).dtype(torch::kByte).pinned_memory(true));
      this->a2av_meta_dev_ = torch::empty(
          {total_meta_bytes * meta_par}, torch::TensorOptions(torch::kCUDA).dtype(torch::kByte));
      if (a2av_hier_compress) {
        const int64_t L = world_size / nnodes;
        const int64_t nseg = L + nnodes - 1;
        auto opt_cuda_i32 = torch::TensorOptions(torch::kCUDA).dtype(torch::kInt);
        this->a2av_mine_token_ = torch::empty({(int64_t)max_ntokens + 1}, opt_cuda_i32);
        this->a2av_pack_flag_ = torch::empty({tokens_per_rank_max * nseg}, opt_cuda_i32);
        if (this->fused_stage2_) {
          this->a2av_sorted_gather_ = torch::empty({n_copies_max}, opt_cuda_i32);
          this->a2av_sorted_scatter_ = torch::empty({n_copies_max}, opt_cuda_i32);
          this->a2av_blk_cnt_ = torch::empty({2 * meta_groups}, opt_cuda_i32);
        }
        this->a2av_pack_gather_ =
            torch::empty({tokens_per_rank_max * (int64_t)topk + 1}, opt_cuda_i64);
        if (nnodes > 1 && !this->union_bcast_) {
          // relay mode grows the index scratch by the extra source-lr axis:
          //   flag: (round, src_lr, token, dst_lr); idx columns capacity
          //   Sum_{round, src_lr, dst_lr} u <= (NN-1) * L * T * min(topk, L)
          // (per-round scratch capacity is unchanged: in-window forwarded rows
          //  <= window_rows * min(topk, L) <= T * topk)
          // (union bcast skips all of this: whole-union forwards need no
          //  gather scratch, flags, or index columns)
          const int64_t src_lrs = this->relay_identity_ ? 1 : L;
          const int64_t idx_cap =
              this->relay_identity_
                  ? (nnodes - 1) * tokens_per_rank_max * (int64_t)topk
                  : (nnodes - 1) * tokens_per_rank_max * L * std::min<int64_t>(topk, L);
          this->a2av_fwd_scratch_ = torch::empty(
              {tokens_per_rank_max * (int64_t)topk, hidden},
              torch::TensorOptions(torch::kCUDA).dtype(input_dtype));
          this->a2av_fwd_flag_ =
              torch::empty({(nnodes - 1) * tokens_per_rank_max * src_lrs * L + 1}, opt_cuda_i32);
          this->a2av_fwd_idx_ = torch::empty({idx_cap + 1}, opt_cuda_i64);
        }
      }
      if (rank == 0) {
        double sym_mb = (this->a2av_send_buffer.nbytes() + this->a2av_recv_buffer.nbytes() +
                         this->a2av_signal_buffer.nbytes()) /
                        1024.0 / 1024.0;
        if (this->a2av_stage_buffer_.defined()) {
          sym_mb += (this->a2av_stage_buffer_.nbytes() + this->a2av_node_signal_buffer_.nbytes()) /
                    1024.0 / 1024.0;
        }
        if (this->a2av_relay_stage_.defined()) {
          sym_mb += (this->a2av_relay_stage_.nbytes() + this->a2av_pack_ready_sig_.nbytes() +
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
      {
        // NVSHMEM transport-kernel priming (the gather_rs 1550b67 fix,
        // audited across to the dispatch op 2026-08-17): under
        // CUDA_MODULE_LOADING=LAZY a transport kernel's module loads at its
        // FIRST launch; the persistent GEMM (and EARLY_LAUNCH's deferred
        // wire replayed behind it) can otherwise strand that first load
        // behind resident spin kernels. One real op per transport path —
        // self signal, P2P put+signal, P2P bare signal, inter-node
        // put+signal, inter-node bare signal (the zero-row-lane primitive)
        // — SET 0 over a zero-initialized slot, idle device, collective.
        cudaStream_t pstream = c10::cuda::getCurrentCUDAStream();
        uint64_t *psig = reinterpret_cast<uint64_t *>(this->a2av_signal_buffer.data_ptr());
        nvshmemx_signal_op_on_stream(psig, 0, NVSHMEM_SIGNAL_SET, this->rank, pstream);
        if (dist_env.local_world_size > 1) {
          int peer = dist_env.node_idx * dist_env.local_world_size +
                     (dist_env.local_rank + 1) % dist_env.local_world_size;
          nvshmemx_putmem_signal_nbi_on_stream(
              psig, psig, sizeof(uint64_t), psig + 1, 0, NVSHMEM_SIGNAL_SET, peer, pstream);
          nvshmemx_signal_op_on_stream(psig, 0, NVSHMEM_SIGNAL_SET, peer, pstream);
        }
        if (nnodes > 1) {
          int peer = ((dist_env.node_idx + 1) % nnodes) * dist_env.local_world_size +
                     dist_env.local_rank;
          nvshmemx_putmem_signal_nbi_on_stream(
              psig, psig, sizeof(uint64_t), psig + 1, 0, NVSHMEM_SIGNAL_SET, peer, pstream);
          nvshmemx_signal_op_on_stream(psig, 0, NVSHMEM_SIGNAL_SET, peer, pstream);
        }
        nvshmemx_quiet_on_stream(pstream);
        CUDA_CHECK(cudaStreamSynchronize(pstream));
        nvshmem_barrier_all();
      }
      // EPIC §4.3 in-kernel swap state. The knob value is the scratch byte
      // count (one expert's fc1+fc2); it must be set SPMD-identically before
      // construction — nvshmem_create_tensor is collective, so lazy first-use
      // allocation would deadlock (non-participating ranks skip swap calls).
      this->inkernel_swap_bytes_ = get_int_from_env("FLUX_A2AV_INKERNEL_SWAP", 0);
      if (this->inkernel_swap_bytes_ > 0) {
        FLUX_CHECK_EQ(this->inkernel_swap_bytes_ % 16, 0)
            << "swap scratch must be 16B-granular";
        this->swap_scratch_ =
            nvshmem_create_tensor({this->inkernel_swap_bytes_}, at::ScalarType::Byte);
        this->swap_flag_ =
            nvshmem_create_tensor({1}, at::ScalarType::Long, /*init_zero=*/true);
        this->swap_arrive_ =
            torch::zeros({1}, torch::TensorOptions(torch::kCUDA).dtype(torch::kLong));
        this->swap_stamps_ =
            torch::zeros({4}, torch::TensorOptions(torch::kCUDA).dtype(torch::kLong));
        this->swap_stamps_pinned_ = torch::empty(
            {4}, torch::TensorOptions(torch::kCPU).dtype(torch::kLong).pinned_memory(true));
        this->swap_events_.resize(2 * kSwapEventPool);
        for (auto &ev : this->swap_events_) {
          CUDA_CHECK(cudaEventCreate(&ev));
        }
        // the swap kernel is a module of its own — preload it (LAZY-loading
        // hang class; a2av_combine_preload precedent)
        epic_swap_preload();
        nvshmem_barrier_all();
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
    CUDA_CHECK(cudaEventCreateWithFlags(&this->counts_event_[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->counts_event_[1], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->pack_inputs_free_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->barrier_done_event_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->hier_dispatch_event_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->fwd_index_event_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->fwd_cnt_event_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->relay_send_event_, cudaEventDisableTiming));
    if (this->wait_flush_) {
      int dev = 0, can_flush = 0;
      CUDA_CHECK(cudaGetDevice(&dev));
      CUDA_CHECK(cudaDeviceGetAttribute(&can_flush, cudaDevAttrCanFlushRemoteWrites, dev));
      FLUX_CHECK(can_flush) << "FLUX_A2AV_WAIT_FLUSH=1 but the device cannot flush remote "
                               "writes (cudaDevAttrCanFlushRemoteWrites=0)";
    }
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
    this->nvtx_proxy_enabled_ =
        this->a2av_dispatch_ && get_int_from_env("FLUX_A2AV_NVTX_PROXY", 0) != 0;
    if (this->nvtx_proxy_enabled_) {
      CUDA_CHECK(cudaHostAlloc(
          (void **)&this->progress_slots_, sizeof(A2AVProgressSlots), cudaHostAllocDefault));
      memset(this->progress_slots_, 0, sizeof(A2AVProgressSlots));
      CUDA_CHECK(cudaMalloc((void **)&this->progress_slots_dev_, sizeof(A2AVProgressSlots)));
      CUDA_CHECK(cudaMemset(this->progress_slots_dev_, 0, sizeof(A2AVProgressSlots)));
      this->nvtx_proxy_ = std::make_unique<A2AVNvtxProxy>(
          this->progress_slots_,
          this->progress_slots_dev_,
          (cudaStream_t)this->nvtx_proxy_stream_,
          this->rank,
          this->world_size,
          this->nnodes);
    }
  }

  ~GemmGroupedV2AGScatterOpImpl() {
    this->nvtx_proxy_.reset();  // join the poller before freeing its slots
    if (this->progress_slots_ != nullptr) {
      CUDA_CHECK(cudaFreeHost(this->progress_slots_));
      CUDA_CHECK(cudaFree(this->progress_slots_dev_));
      if (this->tile_trace_dev_ != nullptr) {
        CUDA_CHECK(cudaFree(this->tile_trace_dev_));
      }
    }
    for (int i = 0; i < kNumTimingEvents; i++) {
      CUDA_CHECK(cudaEventDestroy(this->timing_events_[i]));
    }
    for (int i = 0; i < kNumStage2Events; i++) {
      CUDA_CHECK(cudaEventDestroy(this->stage2_events_[i]));
    }
    for (int i = 0; i < kNumRelayFwdEvents; i++) {
      CUDA_CHECK(cudaEventDestroy(this->relay_fwd_events_[i]));
    }
    CUDA_CHECK(cudaEventDestroy(this->counts_event_[0]));
    CUDA_CHECK(cudaEventDestroy(this->counts_event_[1]));
    CUDA_CHECK(cudaEventDestroy(this->pack_inputs_free_));
    CUDA_CHECK(cudaEventDestroy(this->barrier_done_event_));
    CUDA_CHECK(cudaEventDestroy(this->hier_dispatch_event_));
    CUDA_CHECK(cudaEventDestroy(this->fwd_index_event_));
    CUDA_CHECK(cudaEventDestroy(this->fwd_cnt_event_));
    CUDA_CHECK(cudaEventDestroy(this->relay_send_event_));
    CUDA_CHECK(cudaEventDestroy(this->signal_done_event_));
    CUDA_CHECK(cudaEventDestroy(this->all_gather_event));
    CUDA_CHECK(cudaEventDestroy(this->fetch_remote_event));
    CUDA_CHECK(cudaEventDestroy(this->ready_event));
    for (auto &ev : this->fanout_events_) {
      CUDA_CHECK(cudaEventDestroy(ev));
    }
    for (auto &ev : this->swap_events_) {
      CUDA_CHECK(cudaEventDestroy(ev));
    }
    for (auto &s : this->fanout_streams_) {
      CUDA_CHECK(cudaStreamDestroy(s));
    }
    CUDA_CHECK(cudaStreamDestroy(this->cp_stream));
    CUDA_CHECK(cudaStreamDestroy(this->cp_stream_inter_node));
    CUDA_CHECK(cudaStreamDestroy(this->cp_stream_signal));
    CUDA_CHECK(cudaStreamDestroy(this->pack_stream_));
    CUDA_CHECK(cudaStreamDestroy(this->nvtx_proxy_stream_));
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
      cudaStream_t stream,
      // wire-deferral (FLUX_A2AV_EARLY_LAUNCH replay) is a caller decision:
      // forward_impl passes early_launch_; dispatch_only passes false (no
      // GEMM to reorder around — this also keeps the gather/relay tail on
      // cp_stream instead of the never-joined pack_stream_).
      bool defer_wire_arg) {
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
    std::vector<int64_t> recv_start_h, win_a_h, win_b_h, gate_q_h;
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
    if (this->nvtx_proxy_enabled_) {
      this->nvtx_ssc_.clear();  // repopulated iff the meta path runs below
    }
    // pack overlap: parity slice of the send buffer / meta arena, and the
    // stream the pack runs on (the main stream when the knob is off)
    const bool pack_ov = this->pack_overlap_ && this->a2av_hier_compress_;
    const int par = pack_ov ? (int)(this->run_id_ & 1) : 0;
    cudaStream_t pack_str = pack_ov ? (cudaStream_t)this->pack_stream_ : stream;
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
    torch::Tensor cumA_dev, offA_dev, offR_of_A_dev, expert_base_dev, ssc_dev, gate_q_dev;
    if (use_meta) {
      // guard pinned-staging reuse: the previous iteration's H2D must be done
      // (counts_event_ doubles as the H2D-completion event on this path; it is
      // long finished by now, so this returns immediately)
      CUDA_CHECK(cudaEventSynchronize(this->counts_event_[par]));
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
      if (!compress) {
        // recv holds raw copies: gate on the max copies column over ALL
        // destinations — the same expression on every rank, so a failure is
        // collective. The old per-rank M_this_ep form let one skewed rank
        // throw while the rest entered the wire and spun on its signals until
        // an external watchdog killed them. Under compress the recv layout is
        // deduped and the max_col dedup check below is the true (also
        // collective) recv gate — M_this_ep there counts GEMM A-row copies,
        // which alias recv rows and may legitimately exceed recv capacity
        // (index scratch is ctor-sized to n_copies_max, so no separate A-row
        // capacity check is needed).
        int64_t copies_max_col = 0;
        for (int d = 0; d < W; d++) {
          int64_t col = 0;
          for (int s = 0; s < W; s++) {
            col += chunks64[s * W + d];
          }
          copies_max_col = std::max(copies_max_col, col);
        }
        FLUX_CHECK_LE(copies_max_col, this->max_recv_ntokens_)
            << "a2av recv buffer overflow; raise FLUX_A2AV_MAX_RECV_NTOKENS";
      }
      // staging layout: cumA/offA/offR_of_A i64 [nexG], expert_base i64
      // [nexperts], sorted_splits_cumsum i32 [nexG] (row-major [E, W])
      char *stage = reinterpret_cast<char *>(this->a2av_meta_pinned_.data_ptr()) +
                    (int64_t)par * this->meta_stride_;
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
      if (this->nvtx_proxy_enabled_) {
        // snapshot for the dense-schedule expected-tile totals (forward_impl);
        // pinned arena parity slices get reused, this copy doesn't
        this->nvtx_ssc_.assign(ssc_h, ssc_h + nexG);
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
                << "a2av_unique_counts inconsistent with splits_per_source at (" << s << ", " << d
                << ")";
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
        // dedup recv layout: source-major regions of u[s][d] rows — except in
        // union-bcast mode, where a REMOTE-node source's region holds the whole
        // U[s][node(d)]-row union (the gateway forwards it verbatim; consumers
        // alias their subset). Overflow check takes the max over ALL
        // destinations — the same expression on every rank, so a failure is
        // collective (no one-rank-throws hang).
        auto region_rows = [&](int s, int d) -> int64_t {
          return (this->union_bcast_ && s / L != d / L) ? U_mat[(int64_t)s * NN + d / L]
                                                        : u_mat[(int64_t)s * W + d];
        };
        int64_t max_col = 0;
        for (int d = 0; d < W; d++) {
          int64_t col = 0;
          for (int s = 0; s < W; s++) {
            col += region_rows(s, d);
          }
          max_col = std::max(max_col, col);
        }
        FLUX_CHECK_LE(max_col, this->max_recv_ntokens_)
            << "a2av compress recv overflow; raise FLUX_A2AV_MAX_RECV_NTOKENS";
        recv_off_u.assign(W, 0);
        for (int s = 1; s < W; s++) {
          recv_off_u[s] = recv_off_u[s - 1] + region_rows(s - 1, rank);
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
        if (this->union_bcast_) {
          // union bcast: whole-union forwards need no per-destination columns
          fwd_col_off_h.clear();
          gate_q_h.clear();
          if (NN > 1 && !this->relay_identity_) {
            // Tier B gating queries, per expert e: [e * R | e * R + end(w)...]
            // with R = max_recv_ntokens_ and end(w) the recv-row end of lane w
            // in global source order — local lanes end at their union region
            // end, remote lanes at their delivering window's end. Consumed by
            // one searchsorted over the (expert, dedup recv row) composite key
            // in build_stage2 (per-expert A-row counts, right-open at ends).
            const int64_t R_key = this->max_recv_ntokens_;
            const int64_t E_loc = this->ep_nexperts;
            std::vector<int64_t> lane_end(W, 0);
            this->nvtx_window_rows_.assign(W, 0);
            for (int s = 0; s < W; s++) {
              if (s / L == my_node) {
                lane_end[s] = recv_off_u[s] + region_rows(s, rank);
              } else {
                const int ns = s / (int)L, gl = s % (int)L;
                lane_end[s] = recv_off_u[ns * L] + chunk_bound(ns, my_node, gl + 1);
                this->nvtx_window_rows_[s] = (uint32_t)chunk_rows_of(ns, my_node, gl);
              }
            }
            gate_q_h.assign((size_t)E_loc * (W + 1), 0);
            for (int64_t e = 0; e < E_loc; e++) {
              gate_q_h[e * (W + 1)] = e * R_key;
              for (int s = 0; s < W; s++) {
                gate_q_h[e * (W + 1) + 1 + s] = e * R_key + lane_end[s];
              }
            }
          }
        } else if (this->relay_identity_ || NN == 1) {
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
        for (size_t i = 0; i < gate_q_h.size(); i++) {
          cmp_h[coff++] = gate_q_h[i];
        }
      }
      if (pack_ov) {
        // GEMM n-1 reads this parity's meta slice (ssc_dev) until it finishes;
        // pack_inputs_free_ (recorded after build_stage2 n-1, i.e. before GEMM
        // n-1 starts on the main stream) also transitively orders us behind
        // everything that reads the pack scratch this stream is about to write
        CUDA_CHECK(cudaStreamWaitEvent(pack_str, this->pack_inputs_free_));
      }
      CUDA_CHECK(cudaMemcpyAsync(
          reinterpret_cast<char *>(this->a2av_meta_dev_.data_ptr()) +
              (int64_t)par * this->meta_stride_,
          stage,
          this->meta_stride_,
          cudaMemcpyHostToDevice,
          pack_str));
      CUDA_CHECK(cudaEventRecord(this->counts_event_[par], pack_str));
      auto opt_dev_i64 = torch::TensorOptions(torch::kCUDA).dtype(torch::kLong);
      auto opt_dev_i32 = torch::TensorOptions(torch::kCUDA).dtype(torch::kInt);
      char *dev = reinterpret_cast<char *>(this->a2av_meta_dev_.data_ptr()) +
                  (int64_t)par * this->meta_stride_;
      cumA_dev = torch::from_blob(dev, {nexG}, opt_dev_i64);
      offA_dev = torch::from_blob(dev + nexG * 8, {nexG}, opt_dev_i64);
      offR_of_A_dev = torch::from_blob(dev + 2 * nexG * 8, {nexG}, opt_dev_i64);
      expert_base_dev = torch::from_blob(dev + 3 * nexG * 8, {nex}, opt_dev_i64);
      ssc_dev = torch::from_blob(dev + 3 * nexG * 8 + nex * 8, {E, (int64_t)W}, opt_dev_i32);
      if (compress) {
        const int64_t L = dist_env.local_world_size;
        const int64_t NN = dist_env.nnodes;
        const int64_t nseg = L + NN - 1;
        seg_off_dev = torch::from_blob(dev + this->compress_meta_off_, {nseg}, opt_dev_i64);
        if (NN > 1 && this->union_bcast_ && !this->relay_identity_) {
          gate_q_dev = torch::from_blob(
              dev + this->compress_meta_off_ + (nseg + 1) * 8,
              {(int64_t)this->ep_nexperts * ((int64_t)W + 1)},
              opt_dev_i64);
        }
        if (NN > 1 && !this->union_bcast_) {
          char *cbase = dev + this->compress_meta_off_ + (nseg + 1) * 8;
          if (this->relay_identity_) {
            fwd_col_off_dev = torch::from_blob(cbase, {(NN - 1) * L}, opt_dev_i64);
          } else {
            const int64_t R = NN - 1;
            fwd_col_off_dev = torch::from_blob(cbase, {R * L * L}, opt_dev_i64);
            recv_start_dev = torch::from_blob(cbase + R * L * L * 8, {R * L}, opt_dev_i64);
            win_a_dev = torch::from_blob(cbase + (R * L * L + R * L) * 8, {R}, opt_dev_i64);
            win_b_dev = torch::from_blob(cbase + (R * L * L + R * L + R) * 8, {R}, opt_dev_i64);
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
    const int64_t nseg_c = compress ? (int64_t)dist_env.local_world_size + dist_env.nnodes - 1 : 0;
    if (compress) {
      // pre-zero the seg-major [nseg, tokens] pack flags the fused stage1
      // kernel writes (the pack scan below turns them into pack_gather)
      CUDA_CHECK(cudaMemsetAsync(
          this->a2av_pack_flag_.data_ptr(),
          0,
          (size_t)(nseg_c * tokens_per_rank) * sizeof(int32_t),
          pack_str));
      if (this->fused_stage2_) {
        // fused consumer build: stage 1 writes the per-token keep flags (the
        // legacy chain's mine_n scatter), incl. the +1 garbage slot
        CUDA_CHECK(cudaMemsetAsync(
            this->a2av_mine_token_.data_ptr(),
            0,
            (size_t)((int64_t)tokens_per_rank * W + 1) * sizeof(int32_t),
            pack_str));
      }
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
            .pack_key = this->a2av_pack_key_.data_ptr<int64_t>(),
            .pack_flag = compress ? this->a2av_pack_flag_.data_ptr<int32_t>() : nullptr,
            .topk = (int)topk,
            .local_world_size = dist_env.local_world_size,
            .node_idx = dist_env.node_idx,
            .mine_token = (compress && this->fused_stage2_)
                              ? this->a2av_mine_token_.data_ptr<int32_t>()
                              : nullptr,
            .union_bcast = this->union_bcast_ && dist_env.nnodes > 1},
        pack_str);
    if (!use_meta) {
      this->a2av_chunks_cpu_.copy_(chunks_full, /*non_blocking=*/true);  // 1 KB into pinned
      CUDA_CHECK(cudaEventRecord(this->counts_event_[par], stream));
    }

    if (compress) {
      // compressed producer pack, fused: the stage1 kernel already wrote the
      // seg-major [nseg, tokens] flags; ONE scan kernel assigns each flagged
      // token its exclusive rank within its segment and builds pack_gather,
      // then one index_select gathers the rows from inputs_shard. (Replaces
      // the previous ~12-launch ATen scatter/cumsum/scatter chain — the cost
      // was launch count on the single hardware queue, not bandwidth.)
      a2av_pack_scan_impl(
          A2AVPackScanArguments{
              .pack_flag = this->a2av_pack_flag_.data_ptr<int32_t>(),
              .seg_off = seg_off_dev.data_ptr<int64_t>(),
              .pack_gather = this->a2av_pack_gather_.data_ptr<int64_t>(),
              .tokens = (int64_t)tokens_per_rank,
              .nseg = (int)nseg_c},
          pack_str);
      if (total_send_rows > 0) {
        // pack overlap: the gather runs on pack_stream_ (stream guard) and
        // writes this iteration's parity half of the send buffer
        c10::optional<c10::cuda::CUDAStreamGuard> guard;
        if (pack_ov) {
          guard.emplace(this->pack_stream_);
        }
        auto send_view = this->a2av_send_buffer.narrow(
            0, (int64_t)par * this->send_half_rows_, total_send_rows);
        at::index_select_out(
            send_view, inputs_shard, 0, this->a2av_pack_gather_.narrow(0, 0, total_send_rows));
      }
      // pack overlap lifetime contract (repo convention, cf. the commented
      // record_stream note in all_to_all_transpose_gemm_kernel.cc): the CALLER
      // must keep inputs_shard / scatter_index / splits (and not mutate them)
      // until the next forward() or a device sync — pack_stream_ reads them
      // without allocator bookkeeping (record_stream on an external stream
      // aborts at interpreter teardown: allocator event queries outlive the
      // CUDA context).
      static const bool kCheckCompressPack = get_int_from_env("FLUX_A2AV_CHECK_COMPRESS", 0) != 0;
      if (kCheckCompressPack) {
        // debug only (may sync): per-segment flag counts must reproduce the
        // u/U-derived segment sizes, else pack rows spill into the next segment
        c10::optional<c10::cuda::CUDAStreamGuard> guard;
        if (pack_ov) {
          guard.emplace(this->pack_stream_);  // flags were written on pack_stream_
        }
        auto seg_len = this->a2av_pack_flag_.narrow(0, 0, nseg_c * (int64_t)tokens_per_rank)
                           .view({nseg_c, (int64_t)tokens_per_rank})
                           .sum(1)
                           .cpu();  // [nseg]
        for (int64_t i = 0; i < nseg_c; i++) {
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
    CUDA_CHECK(cudaEventRecord(this->ready_event, pack_str));
    if (pack_ov) {
      // main stream consumes the pack outputs (fwd-index builds + consumer
      // build read e_all & the meta views) — join before either runs
      CUDA_CHECK(cudaStreamWaitEvent(stream, this->ready_event));
    }
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
    if (compress && dist_env.nnodes > 1 && this->relay_identity_ && !this->union_bcast_) {
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
      auto tgt = (pos + fwd_col_off_dev.view({R, 1, Lc})).masked_fill_(flag3d.eq(0), fwd_garbage);
      auto vals = posU.unsqueeze(2).expand({R, T, Lc});
      this->a2av_fwd_idx_.scatter_(0, tgt.reshape(-1), vals.reshape(-1));
      static const bool kCheckCompressFwd = get_int_from_env("FLUX_A2AV_CHECK_COMPRESS", 0) != 0;
      if (kCheckCompressFwd) {
        // debug only (may sync): the on-device flag counts must reproduce the
        // host metadata driving the wire offsets, else the gathers read skew
        auto cnt = flag3d.sum(1).cpu();  // [R, Lc] == u[s_r][my node dests]
        auto ucnt = uni.sum(1).cpu();    // [R]     == U[s_r][my_node]
        for (int64_t r = 0; r < R; r++) {
          int s = dist_env.local_rank_to_global_rank(
              dist_env.local_rank, (int)((my_node + r + 1) % NNc));
          for (int64_t dlv = 0; dlv < Lc; dlv++) {
            FLUX_CHECK_EQ(cnt[r][dlv].item<int64_t>(), u_mat[(int64_t)s * W + my_node * Lc + dlv])
                << "a2av compress fwd-flag/u mismatch at (" << s << ", " << dlv << ")";
          }
          FLUX_CHECK_EQ(ucnt[r].item<int64_t>(), U_mat[(int64_t)s * NNc + my_node])
              << "a2av compress fwd-union/U mismatch at source " << s;
        }
      }
      CUDA_CHECK(cudaEventRecord(this->fwd_index_event_, stream));
    } else if (compress && dist_env.nnodes > 1 && !this->union_bcast_) {
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
      auto e_rounds =
          e_all.view({NNc, Lc, copies_per_rank}).roll(-(my_node + 1), 0).narrow(0, 0, R);
      auto dl = e_rounds.div(E, "floor").sub_(my_node * Lc);  // local dest, or off-node
      auto off_node = dl.lt(0).logical_or_(dl.ge(Lc));
      rmark(1);
      // flat flag position ((r * L + sl) * T + t) * L + dl
      auto rsl_base = this->a2av_arange_i64_.narrow(0, 0, R * Lc).view({R, Lc, 1}) * (T * Lc);
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
      auto in_w =
          canon.ge(win_a_dev.view({R, 1, 1})).logical_and_(canon.lt(win_b_dev.view({R, 1, 1})));
      auto below = canon.lt(win_a_dev.view({R, 1, 1}));
      rmark(4);
      auto valid = flag4d * in_w.unsqueeze(3);  // needed by dl AND inside my window
      rmark(5);
      auto pos = valid.cumsum(2) - valid;  // in-window rank within (r, sl, dl)
      rmark(6);
      auto tgt =
          (pos + fwd_col_off_dev.view({R, Lc, 1, Lc})).masked_fill_(valid.eq(0), fwd_garbage);
      rmark(7);
      // stored value = window-relative staging row
      auto vals = (canon - win_a_dev.view({R, 1, 1})).unsqueeze(3).expand({R, Lc, T, Lc});
      auto tgt_flat = tgt.reshape(-1);
      auto vals_flat = vals.reshape(-1);
      rmark(8);
      this->a2av_fwd_idx_.scatter_(0, tgt_flat, vals_flat);
      rmark(9);
      auto cnt_in = valid.sum(2);                              // [R, L, L]
      auto cnt_before = (flag4d * below.unsqueeze(3)).sum(2);  // [R, L, L]
      rmark(10);
      this->a2av_fwd_cnt_pinned_.select(0, 0).copy_(cnt_in.to(torch::kInt), true);
      this->a2av_fwd_cnt_pinned_.select(0, 1).copy_(cnt_before.to(torch::kInt), true);
      rmark(11);
      CUDA_CHECK(cudaEventRecord(this->fwd_cnt_event_, stream));
      static const bool kCheckCompressFwd = get_int_from_env("FLUX_A2AV_CHECK_COMPRESS", 0) != 0;
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
              FLUX_CHECK_LE(cbef[r][sl][dlv].item<int64_t>() + cin[r][sl][dlv].item<int64_t>(), uv)
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
      torch::Tensor e_loc;  // negative for foreign copies (masked below)
      if (!(compress && this->fused_stage2_)) {
        e_loc = e_all.sub((int64_t)ep_start);
      }
      mark(1);
      constexpr int64_t kMax64 = std::numeric_limits<int64_t>::max();
      if (compress && this->fused_stage2_) {
        // ---- fused consumer build (FLUX_A2AV_FUSED_STAGE2): the ATen
        // key/argsort/index_select chain and the Tier B gating searchsorted
        // collapse into one cumsum + two kernels. Stage 1 already wrote the
        // per-token keep flags; A rows are assigned per (expert, source)
        // group from the host offA table plus an atomic in-group rank —
        // interior order is arbitrary, which no consumer observes (gather /
        // scatter are per-row indirections, the tile gating compares only
        // group boundaries). Timing-mark remap under this path:
        //   keyA = scratch memset, sortA = mine cumsum,
        //   keyR = consumer build kernel, sortR = Tier B gating cumsum.
        const int64_t ntokens = (int64_t)tokens_per_rank * W;
        auto mine_n = this->a2av_mine_token_.narrow(0, 0, ntokens + 1);
        CUDA_CHECK(cudaMemsetAsync(
            this->a2av_blk_cnt_.data_ptr(), 0, this->a2av_blk_cnt_.nbytes(), stream));
        mark(2);
        auto c_excl = mine_n.cumsum(0) - mine_n;  // C[t], i64 [ntokens + 1]
        mark(3);
        const bool tier_b = this->union_bcast_ && !this->relay_identity_ && dist_env.nnodes > 1;
        int32_t *blk_cnt = this->a2av_blk_cnt_.data_ptr<int32_t>();
        a2av_consumer_build_impl(
            A2AVConsumerBuildArguments{
                .n_copies = n_copies,
                .topk = (int)topk,
                .ep_start = (int)ep_start,
                .ep_nexperts = (int)E,
                .world_size = W,
                .e_all = e_all.data_ptr<int64_t>(),
                .s_all = s_all.data_ptr<int64_t>(),
                .flat_dst = flat_dst.data_ptr<int64_t>(),
                .not_mine = not_mine.data_ptr<bool>(),
                .c_excl = c_excl.data_ptr<int64_t>(),
                .offA = offA_dev.data_ptr<int64_t>(),
                .expert_base = expert_base_dev.data_ptr<int64_t>(),
                .blk_cnt = blk_cnt,
                .gather = this->a2av_sorted_gather_.data_ptr<int32_t>(),
                .scatter = this->a2av_sorted_scatter_.data_ptr<int32_t>(),
                // gate_q row 0 is [0, end(0), ..., end(W-1)]: skip the base
                .lane_end = tier_b ? gate_q_dev.data_ptr<int64_t>() + 1 : nullptr,
                .gate_hist = tier_b ? blk_cnt + nexG : nullptr},
            stream);
        mark(4);
        if (tier_b) {
          if (!this->a2av_gating_cumsum_.defined()) {
            this->a2av_gating_cumsum_ = torch::empty(
                {(int64_t)E, (int64_t)W}, torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));
          }
          a2av_gating_cumsum_impl(
              A2AVGatingCumsumArguments{
                  .ep_nexperts = (int)E,
                  .world_size = W,
                  .gate_hist = blk_cnt + nexG,
                  .gating_cumsum = this->a2av_gating_cumsum_.data_ptr<int32_t>()},
              stream);
        }
        mark(5);
        sorted_gather_index = this->a2av_sorted_gather_.narrow(0, 0, n_copies);
        sorted_scatter_index = this->a2av_sorted_scatter_.narrow(0, 0, n_copies);
        sorted_splits_cumsum = ssc_dev;  // uploaded, exact LOGICAL [E, W] semantics
        for (int i = 6; i <= 10; i++) {
          mark(i);
        }
        static const bool kCheckFused = get_int_from_env("FLUX_A2AV_CHECK_COMPRESS", 0) != 0;
        if (kCheckFused && M_this_ep > 0) {
          // debug only (may sync): order-free inversion. scatter uniquely
          // identifies each row's copy (flat_dst is a global permutation), so
          // recover p per A-row and assert (a) group membership matches the
          // offA nesting, (b) gather is that copy's dedup recv row, (c) the
          // keep-flag total matches the host recv layout (legacy check #2).
          auto iota_m = iota.narrow(0, 0, M_this_ep);
          auto inv = torch::empty({n_copies}, opt_i64).scatter_(0, flat_dst, iota);
          auto g_of = torch::searchsorted(cumA_dev, iota_m, /*out_int32=*/false, /*right=*/true);
          auto e_row = g_of.div((int64_t)W, "floor").add((int64_t)ep_start);
          auto s_row = g_of.remainder((int64_t)W);
          auto flat_of_row = sorted_scatter_index.narrow(0, 0, M_this_ep).to(torch::kLong) +
                             expert_base_dev.index_select(0, e_row);
          auto p_of_row = inv.index_select(0, flat_of_row);
          FLUX_CHECK(torch::equal(e_all.index_select(0, p_of_row), e_row))
              << "a2av fused consumer: expert-group membership mismatch";
          FLUX_CHECK(torch::equal(s_all.index_select(0, p_of_row), s_row))
              << "a2av fused consumer: source-group membership mismatch";
          // row_of_tok[t] = the dedup recv row of kept token t (a row's copy is
          // always kept, so the dropped-token fill never reaches `want`)
          auto row_of_tok = c_excl.narrow(0, 0, ntokens)
                                .masked_fill(mine_n.narrow(0, 0, ntokens).eq(0), ntokens);
          auto want = row_of_tok.index_select(0, p_of_row.div((int64_t)topk, "floor"));
          auto got = sorted_gather_index.narrow(0, 0, M_this_ep).to(torch::kLong);
          FLUX_CHECK(torch::equal(got, want)) << "a2av fused consumer: gather/recv-row mismatch";
          if (this->union_bcast_ && dist_env.nnodes > 1 && !u_mat.empty()) {
            const int64_t Lb = dist_env.local_world_size;
            const int64_t last = W - 1;
            const int64_t last_rows = (last / Lb != rank / Lb)
                                          ? U_mat[last * dist_env.nnodes + rank / Lb]
                                          : u_mat[last * W + rank];
            FLUX_CHECK_EQ(c_excl.index({ntokens}).item<int64_t>(), recv_off_u[last] + last_rows)
                << "a2av fused consumer flag total != host recv layout";
          }
        }
        return;
      }
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
        // union-bcast recv regions for REMOTE-node sources hold the whole node
        // union, so their keep-flag is "needed by my node"; same-node sources
        // keep the exact "needed by me" flag. Tokens stay source-contiguous, so
        // the one-cumsum identity below holds verbatim over the mixed regions.
        if (this->union_bcast_ && dist_env.nnodes > 1) {
          const int64_t Lb = dist_env.local_world_size;
          const int64_t my_node_b = dist_env.node_idx;
          auto src_node = s_all.div(Lb, "floor");
          auto dst_node = e_all.div((int64_t)E * Lb, "floor");
          auto drop = torch::where(src_node.eq(my_node_b), not_mine, dst_node.ne(my_node_b));
          mine_n.scatter_(0, flat_token.masked_fill(drop, ntokens), 1);
        } else {
          mine_n.scatter_(0, flat_token.masked_fill(not_mine, ntokens), 1);
        }
        auto c_excl = mine_n.cumsum(0) - mine_n;  // C[t], i64 [ntokens + 1]
        mark(4);
        auto gidx = c_excl.index_select(0, flat_token);
        // tail rows (>= M_this_ep) are unread garbage; clamp is pure hygiene
        auto sorted_gidx = gidx.index_select(0, perm_a).clamp_(0, this->max_recv_ntokens_ - 1);
        sorted_gather_index = sorted_gidx.to(torch::kInt);
        auto scatter_val = flat_dst - expert_base_dev.index_select(0, e_all);
        sorted_scatter_index = scatter_val.index_select(0, perm_a).to(torch::kInt);
        mark(5);
        sorted_splits_cumsum = ssc_dev;  // uploaded, exact LOGICAL [E, W] semantics
        if (this->union_bcast_ && !this->relay_identity_ && dist_env.nnodes > 1) {
          // Tier B gating cumsum via one searchsorted: rows sorted by key_a are
          // (expert, source, copy)-ordered and the dedup recv row is monotone
          // in that order, so (expert, recv row) composited as e * R + row is a
          // globally sorted key. Lane ends (local = union region ends, remote =
          // window ends, both host constants) ride the meta arena as queries
          // e * R and e * R + end(w); gating[e][w] = bisect_left(end) -
          // bisect_left(base). Lane id == signal slot id (the window slot of
          // remote rank (ns, gl)). No H2D and no hidden sync on this path.
          if (!this->a2av_gating_cumsum_.defined()) {
            this->a2av_gating_cumsum_ = torch::empty(
                {(int64_t)E, (int64_t)W}, torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));
          }
          auto e2 = e_loc.index_select(0, perm_a).narrow(0, 0, M_this_ep);
          auto key2 = torch::add(
              sorted_gidx.narrow(0, 0, M_this_ep), e2, (int64_t)this->max_recv_ntokens_);
          auto res = torch::searchsorted(key2, gate_q_dev, /*out_int32=*/true, /*right=*/false)
                         .view({(int64_t)E, (int64_t)W + 1});
          this->a2av_gating_cumsum_.copy_(
              res.narrow(1, 1, W) - res.narrow(1, 0, 1), /*non_blocking=*/true);
        }
        // stages 6-10 don't exist in the compress consumer; mark them anyway so
        // the FLUX_A2AV_TIMING readout keeps one fixed index layout
        for (int i = 6; i <= 10; i++) {
          mark(i);
        }
        static const bool kCheckCompress = get_int_from_env("FLUX_A2AV_CHECK_COMPRESS", 0) != 0;
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
          if (this->union_bcast_ && dist_env.nnodes > 1 && !u_mat.empty()) {
            // the inversion above can't see flag <-> wire-offset divergence
            // (both sides use the same C); assert the flag total against the
            // host recv layout the gateway puts are addressed with
            const int64_t Lb = dist_env.local_world_size;
            const int64_t last = W - 1;
            const int64_t last_rows = (last / Lb != rank / Lb)
                                          ? U_mat[last * dist_env.nnodes + rank / Lb]
                                          : u_mat[last * W + rank];
            FLUX_CHECK_EQ(c_excl.index({ntokens}).item<int64_t>(), recv_off_u[last] + last_rows)
                << "a2av bcast consumer flag total != host recv layout";
          }
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
        static const bool kCheckIdentity = get_int_from_env("FLUX_A2AV_CHECK_IDENTITY", 0) != 0;
        if (kCheckIdentity) {
          auto key_a = ((e_loc * W + s_all) * kShift + iota).masked_fill_(not_mine, kMax64);
          auto perm_a = key_a.argsort();
          auto recv_pos = torch::empty({n_copies}, opt_i64).scatter_(0, order_r, iota);
          auto ref_gather = recv_pos.index_select(0, perm_a).to(torch::kInt);
          auto ref_scatter = scatter_val.index_select(0, perm_a).to(torch::kInt);
          FLUX_CHECK(
              torch::equal(
                  sorted_gather_index.narrow(0, 0, M_this_ep), ref_gather.narrow(0, 0, M_this_ep)))
              << "a2av metadata identity mismatch (gather)";
          FLUX_CHECK(
              torch::equal(
                  sorted_scatter_index.narrow(0, 0, M_this_ep),
                  ref_scatter.narrow(0, 0, M_this_ep)))
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
      auto ends =
          torch::searchsorted(key_r_sorted, thresholds, /*out_int32=*/false, /*right=*/false);
      mark(9);
      auto cnt_flat = at::diff(ends, 1, 0, torch::zeros({1}, opt_i64));
      sorted_splits_cumsum = cnt_flat.view({W, E}).cumsum(0).t().contiguous().to(torch::kInt);
      mark(10);
    };
    // perf-diagnostic knobs (default off): reorder stage 2 after the put issuance /
    // drain the stream before issuing puts, to bisect overlap effects under
    // CUDA_DEVICE_MAX_CONNECTIONS=1
    static const bool kStage2AfterPuts = get_int_from_env("FLUX_A2AV_STAGE2_AFTER_PUTS", 0) != 0;
    static const bool kSyncBeforePuts = get_int_from_env("FLUX_A2AV_SYNC_BEFORE_PUTS", 0) != 0;
    auto h1 = host_now();
    if (!kStage2AfterPuts) {
      build_stage2();
    }
    if (pack_ov) {
      // pack scratch (e_all & friends) has no reader past this point; release
      // it to the NEXT iteration's pack (which overlaps this iteration's GEMM).
      // kStage2AfterPuts is FLUX_CHECKed incompatible with pack_overlap_.
      CUDA_CHECK(cudaEventRecord(this->pack_inputs_free_, stream));
    }
    auto h2 = host_now();
    if (kTiming) {
      CUDA_CHECK(cudaEventRecord(this->timing_events_[2], stream));
    }

    // ---- host: in the derive path, wait only for stage 1's counts D2H (the
    // event precedes the stage-2 enqueues in stream order, so none of the sorts
    // gate the wire). In the metadata path everything is already known.
    if (!use_meta) {
      CUDA_CHECK(cudaEventSynchronize(this->counts_event_[par]));
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
      // never compress on this path (compress requires use_meta); the counts
      // are a D2H of an integer histogram over replicated splits/scatter_index,
      // so the max-column expression is identical on every rank and a failure
      // is collective (see the use_meta site above for why not per-rank).
      int64_t copies_max_col = 0;
      for (int d = 0; d < W; d++) {
        int64_t col = 0;
        for (int s = 0; s < W; s++) {
          col += chunks64[s * W + d];
        }
        copies_max_col = std::max(copies_max_col, col);
      }
      FLUX_CHECK_LE(copies_max_col, this->max_recv_ntokens_)
          << "a2av recv buffer overflow; raise FLUX_A2AV_MAX_RECV_NTOKENS";
    }
    auto chunk_at = [&](int s, int d) -> int64_t { return chunks64[s * W + d]; };
    // compress only (u_mat/U_mat empty otherwise; lambdas are lazy)
    auto u_at = [&](int s, int d) -> int64_t { return u_mat[(int64_t)s * W + d]; };
    auto U_at = [&](int s, int n) -> int64_t { return U_mat[(int64_t)s * dist_env.nnodes + n]; };

    CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream, this->ready_event));
    CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream_inter_node, this->ready_event));
    if (pack_ov) {
      // with the pack off the main stream, ready_event no longer implies the
      // previous iteration's end-of-epoch barrier (recv/stage buffer reuse and
      // the GEQ signal epoch discipline both depend on it) — wait explicitly
      CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream, this->barrier_done_event_));
      CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream_inter_node, this->barrier_done_event_));
    }

    const int64_t row_bytes = (int64_t)hidden * c10::elementSize(input_dtype);
    uint64_t *signal_base = reinterpret_cast<uint64_t *>(this->a2av_signal_buffer.data_ptr());
    char *send_base = reinterpret_cast<char *>(this->a2av_send_buffer.data_ptr()) +
                      (int64_t)par * this->send_half_rows_ * row_bytes;
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
    // FLUX_A2AV_EARLY_LAUNCH: cp_stream wire ops are recorded as descriptors
    // instead of issued; forward_impl replays them (issue_deferred_wire) right
    // after the GEMM launch. The WHOLE cp_stream sequence defers or none of it
    // — cp_stream is FIFO and a partial deferral would reorder delivery.
    const bool defer_wire = defer_wire_arg;
    if (defer_wire) {
      this->deferred_wire_.clear();
      this->deferred_wire_armed_ = true;
    }
    auto emit_self_copy = [&](void *dst, const void *src, int64_t bytes) {
      if (defer_wire) {
        this->deferred_wire_.push_back(
            {DeferredWireOp::kSelfCopy, dst, src, bytes, nullptr, 0, 0});
      } else {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, this->cp_stream));
      }
    };
    auto emit_signal = [&](uint64_t *sig, int pe) {
      if (defer_wire) {
        this->deferred_wire_.push_back(
            {DeferredWireOp::kSignal, nullptr, nullptr, 0, sig, this->run_id_, pe});
      } else {
        nvshmemx_signal_op_on_stream(sig, this->run_id_, NVSHMEM_SIGNAL_SET, pe, this->cp_stream);
      }
    };
    auto emit_put = [&](void *dst, const void *src, int64_t bytes, uint64_t *sig, int pe) {
      if (defer_wire) {
        this->deferred_wire_.push_back(
            {DeferredWireOp::kPut, dst, src, bytes, sig, this->run_id_, pe});
      } else {
        nvshmemx_putmem_signal_nbi_on_stream(
            dst, src, bytes, sig, this->run_id_, NVSHMEM_SIGNAL_SET, pe, this->cp_stream);
      }
    };
    auto emit_wait = [&](uint64_t *ptr) {
      if (defer_wire) {
        this->deferred_wire_.push_back(
            {DeferredWireOp::kWait64, nullptr, nullptr, 0, ptr, this->run_id_, 0});
      } else {
        CU_CHECK(CUStreamWaitValue64(
            this->cp_stream,
            reinterpret_cast<CUdeviceptr>(ptr),
            this->run_id_,
            this->a2av_wait_flags()));
      }
    };
    auto emit_hier_event = [&]() {
      if (defer_wire) {
        this->deferred_wire_.push_back(
            {DeferredWireOp::kRecordHierEvent, nullptr, nullptr, 0, nullptr, 0, 0});
      } else {
        CUDA_CHECK(cudaEventRecord(this->hier_dispatch_event_, this->cp_stream));
      }
    };
    // ---- gather/relay tail ops (t_*): the compress gather arm and the relay
    // gateway loop carry index_select SM kernels. These are NEVER deferred:
    // kernels enqueued after the persistent GEMM has blanketed the SMs can
    // starve at dispatch (observed as a permanent mid-tail cp_stream stall on
    // the relay — the racy loser of the GEMM ramp). Instead, under early
    // launch the tail is issued INLINE on the otherwise-idle pack stream
    // (pre-launch enqueue behind the node_sig front-end wait is exactly the
    // proven non-early configuration); without early launch it stays on
    // cp_stream as before. pack_stream_ is free here: pack_overlap_ is
    // FLUX_CHECKed incompatible with early_launch_.
    const cudaStream_t tail_stream =
        defer_wire ? (cudaStream_t)this->pack_stream_ : (cudaStream_t)this->cp_stream;
    auto t_wait_event = [&](cudaEvent_t ev) { CUDA_CHECK(cudaStreamWaitEvent(tail_stream, ev)); };
    auto t_wait = [&](uint64_t *ptr) {
      if (this->nvshmem_wait_) {
        nvshmemx_signal_wait_until_on_stream(ptr, NVSHMEM_CMP_GE, this->run_id_, tail_stream);
        return;
      }
      CU_CHECK(CUStreamWaitValue64(
          tail_stream,
          reinterpret_cast<CUdeviceptr>(ptr),
          this->run_id_,
          this->a2av_wait_flags()));
    };
    auto t_signal = [&](uint64_t *sig, int pe) {
      nvshmemx_signal_op_on_stream(sig, this->run_id_, NVSHMEM_SIGNAL_SET, pe, tail_stream);
    };
    auto t_blocking_put = [&](void *dst, const void *src, int64_t bytes, uint64_t *sig, int pe) {
      nvshmemx_putmem_signal_on_stream(
          dst, src, bytes, sig, this->run_id_, NVSHMEM_SIGNAL_SET, pe, tail_stream);
    };
    auto t_put_nosig = [&](void *dst, const void *src, int64_t bytes, int pe) {
      nvshmemx_putmem_on_stream(dst, src, bytes, pe, tail_stream);
    };
    // dst <- stage_seg[idx[idx_off : idx_off + rows]] via at::index_select_out
    auto t_index_select =
        [&](void *dst, const void *stage_ptr, int64_t stage_rows, int64_t idx_off, int64_t rows) {
          c10::cuda::CUDAStreamGuard guard(
              at::cuda::getStreamFromExternal(tail_stream, at::cuda::current_device()));
          auto opt_in = torch::TensorOptions(torch::kCUDA).dtype(this->input_dtype);
          auto stage_seg = torch::from_blob(
              const_cast<void *>(stage_ptr), {stage_rows, (int64_t)hidden}, opt_in);
          auto dst_t = torch::from_blob(dst, {rows, (int64_t)hidden}, opt_in);
          auto idx = this->a2av_fwd_idx_.narrow(0, idx_off, rows);
          at::index_select_out(dst_t, stage_seg, 0, idx);
        };
    if (self_rows > 0) {
      emit_self_copy(
          recv_base + self_recv_off * row_bytes,
          send_base + self_send_off * row_bytes,
          self_rows * row_bytes);
    }
    emit_signal(signal_base + rank, rank);
    // zero-payload destinations still get the signal (the GEMM waits on every source)
    auto issue_put = [&](int d, cudaStream_t put_stream) {
      int64_t bytes = chunk_at(rank, d) * row_bytes;
      if (put_stream == (cudaStream_t)this->cp_stream) {
        // intra path: goes through the emitters so early-launch can defer it
        if (bytes > 0) {
          emit_put(
              recv_base + recv_off[d] * row_bytes,
              send_base + send_off[d] * row_bytes,
              bytes,
              signal_base + rank,
              d);
        } else {
          emit_signal(signal_base + rank, d);
        }
        return;
      }
      if (bytes > 0) {
        // blocking_wire_ (instrumented): inter-node puts only — blocking the
        // intra-node P2P puts would serialize the whole flat fan-out
        if (this->blocking_wire_ && d / dist_env.local_world_size != dist_env.node_idx) {
          nvshmemx_putmem_signal_on_stream(
              recv_base + recv_off[d] * row_bytes,
              send_base + send_off[d] * row_bytes,
              bytes,
              signal_base + rank,
              this->run_id_,
              NVSHMEM_SIGNAL_SET,
              d,
              put_stream);
        } else {
          nvshmemx_putmem_signal_nbi_on_stream(
              recv_base + recv_off[d] * row_bytes,
              send_base + send_off[d] * row_bytes,
              bytes,
              signal_base + rank,
              this->run_id_,
              NVSHMEM_SIGNAL_SET,
              d,
              put_stream);
        }
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
      // dedup recv offset of source s's region at destination d (bcast mode:
      // remote-node source regions are U-sized unions, mirroring recv_off_u)
      auto recv_off_of_u = [&](int s, int d) -> int64_t {
        int64_t acc = 0;
        for (int s2 = 0; s2 < s; s2++) {
          acc += (this->union_bcast_ && s2 / L != d / L) ? U_at(s2, d / L) : u_at(s2, d);
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
          emit_put(
              recv_base + recv_off_of_u(rank, d) * row_bytes,
              send_base + send_seg_off(dlg) * row_bytes,
              rows * row_bytes,
              signal_base + rank,
              d);
        } else {
          emit_signal(signal_base + rank, d);
        }
      }
      emit_hier_event();
      if (NN > 1) {
        uint64_t *node_sig =
            reinterpret_cast<uint64_t *>(this->a2av_node_signal_buffer_.data_ptr());
        char *stage_base = reinterpret_cast<char *>(this->a2av_stage_buffer_.data_ptr());
        // undefined in union-bcast mode (whole-union forwards need no scratch)
        char *scratch_base = this->a2av_fwd_scratch_.defined()
                                 ? reinterpret_cast<char *>(this->a2av_fwd_scratch_.data_ptr())
                                 : nullptr;
        if (this->relay_identity_) {
          // inter-node union aggregates, mirror node order; arrival signal slot =
          // source node, value = epoch. Empty aggregates still signal.
          for (int dn = 1; dn < NN; dn++) {
            int tn = (my_node - dn + NN) % NN;
            int g = dist_env.local_rank_to_global_rank(my_lr, tn);
            int64_t rows = U_at(rank, tn);
            int seg = tn < my_node ? tn : tn + L - 1;
            if (rows > 0) {
              // blocking_wire_ (instrumented): local-completion put whose proxy
              // entrypoint kernel spans the wire drain on the timeline
              if (this->blocking_wire_) {
                nvshmemx_putmem_signal_on_stream(
                    stage_base + stage_off_u(tn, my_lr, my_node) * row_bytes,
                    send_base + seg_off_h[seg] * row_bytes,
                    rows * row_bytes,
                    node_sig + my_node,
                    this->run_id_,
                    NVSHMEM_SIGNAL_SET,
                    g,
                    this->cp_stream_inter_node);
              } else {
                nvshmemx_putmem_signal_nbi_on_stream(
                    stage_base + stage_off_u(tn, my_lr, my_node) * row_bytes,
                    send_base + seg_off_h[seg] * row_bytes,
                    rows * row_bytes,
                    node_sig + my_node,
                    this->run_id_,
                    NVSHMEM_SIGNAL_SET,
                    g,
                    this->cp_stream_inter_node);
              }
            } else {
              nvshmemx_signal_op_on_stream(
                  node_sig + my_node,
                  this->run_id_,
                  NVSHMEM_SIGNAL_SET,
                  g,
                  this->cp_stream_inter_node);
            }
          }
          if (this->union_bcast_) {
            // union broadcast: per-round front-end wait, then forward the WHOLE
            // staged union to every local rank (mirror order, self included via
            // loopback put) as ONE contiguous put per destination — the exact
            // forward a2av_hier does, just with the union payload. Pure copy
            // engine: no index build, no gather, no scratch, no SMs. nbi is
            // safe here because the put source is the symmetric staging buffer,
            // untouched until the end-of-iteration barrier quiets these puts
            // (the gather arm's non-nbi constraint is a property of its reused
            // scratch, which this arm does not have).
            for (int dn = 1; dn < NN; dn++) {
              int ns = (my_node + dn) % NN;
              int s = dist_env.local_rank_to_global_rank(my_lr, ns);
              emit_wait(node_sig + ns);
              const int64_t union_rows = U_at(s, my_node);
              char *ustage = stage_base + stage_off_u(my_node, my_lr, ns) * row_bytes;
              for (int dl = 0; dl < L; dl++) {
                int dlg = (my_lr - dl + L) % L;
                int d = dist_env.local_rank_to_global_rank(dlg, my_node);
                if (union_rows > 0) {
                  emit_put(
                      recv_base + recv_off_of_u(s, d) * row_bytes,
                      ustage,
                      union_rows * row_bytes,
                      signal_base + s,
                      d);
                } else {
                  emit_signal(signal_base + s, d);
                }
              }
            }
          } else {
            // gateway rounds: per-round front-end wait (cuStreamWaitValue64, zero
            // SMs), then gather each local destination's exact subset out of the
            // staged union. The index_selects wait on fwd_index_event_ so the
            // index build (main stream) is done. Issued via t_* — inline on the
            // pack stream under early launch (never deferred: see tail_stream).
            t_wait_event(this->fwd_index_event_);
            for (int dn = 1; dn < NN; dn++) {
              int ns = (my_node + dn) % NN;
              int s = dist_env.local_rank_to_global_rank(my_lr, ns);
              t_wait(node_sig + ns);
              int64_t union_rows = U_at(s, my_node);
              char *ustage = stage_base + stage_off_u(my_node, my_lr, ns) * row_bytes;
              const int64_t stage_rows = std::max<int64_t>(union_rows, 1);
              const int64_t round_base = fwd_col_off_h[(dn - 1) * L];
              // forward in mirror local order (same stage slots as a2av_hier)
              for (int dl = 0; dl < L; dl++) {
                int dlg = (my_lr - dl + L) % L;
                int d = dist_env.local_rank_to_global_rank(dlg, my_node);
                int64_t rows = u_at(s, d);
                if (rows == 0) {
                  t_signal(signal_base + s, d);
                  continue;
                }
                const int64_t idx_off = fwd_col_off_h[(dn - 1) * L + dlg];
                if (d == rank) {
                  // gateway's own subset: gather straight into the recv region
                  t_index_select(
                      recv_base + recv_off_of_u(s, rank) * row_bytes,
                      ustage,
                      stage_rows,
                      idx_off,
                      rows);
                  t_signal(signal_base + s, rank);
                } else {
                  // round-relative offsets: the gateway's own (d == rank) column
                  // leaves a hole in the scratch — harmless capacity slack, the
                  // per-round total is still <= copies_per_rank
                  const int64_t scratch_off = idx_off - round_base;
                  t_index_select(
                      scratch_base + scratch_off * row_bytes, ustage, stage_rows, idx_off, rows);
                  // NON-nbi on purpose: the scratch is refilled next round, and
                  // nbi gives no local-completion guarantee
                  t_blocking_put(
                      recv_base + recv_off_of_u(s, d) * row_bytes,
                      scratch_base + scratch_off * row_bytes,
                      rows * row_bytes,
                      signal_base + s,
                      d);
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

          // ---- phase 1 (PULL): each relay gathers its own chunk with
          // intra-node gets on its own wire stream, for ALL rounds, before
          // any wire wait — 8 relays run on 8 copy engines instead of
          // chaining on each source's piece-put FIFO (the retired serial
          // design; A/B capsule 20260803-150832). DEADLOCK RULE: a get waits
          // only on the peer's pack-ready flag, which depends only on that
          // peer's local pack — acyclic. Announce my pack FIRST (my writers
          // precede any of my waits: no same-rank wait inversion). Zero-
          // overlap pairs are skipped on both sides (no signal, no wait).
          uint64_t *pack_ready =
              reinterpret_cast<uint64_t *>(this->a2av_pack_ready_sig_.data_ptr());
          for (int dl = 1; dl < L; dl++) {
            int d = dist_env.local_rank_to_global_rank((my_lr + dl) % L, my_node);
            nvshmemx_signal_op_on_stream(
                pack_ready + my_lr,
                this->run_id_,
                NVSHMEM_SIGNAL_SET,
                d,
                this->cp_stream_inter_node);
          }
          // peer send-segment base for (peer local rank, target node): the
          // U/u rows are replicated, so every peer's send layout is
          // host-computable (generalizes seg_off_h / my_seg_base)
          auto peer_seg_base = [&](int plr, int tn) -> int64_t {
            int prank = dist_env.local_rank_to_global_rank(plr, my_node);
            int64_t acc = 0;
            for (int n = 0; n < tn; n++) {
              if (n == my_node) {
                for (int dl = 0; dl < L; dl++) {
                  acc += u_at(prank, n * L + dl);
                }
              } else {
                acc += U_at(prank, n);
              }
            }
            return acc;
          };
          if (this->relay_poison_) {
            // T2 diagnostic: sentinel-fill the relay panel so any row the
            // wire ships before its pull landed is unmistakable (0xA5)
            CUDA_CHECK(cudaMemsetAsync(
                relay_base,
                0xA5,
                (size_t)this->a2av_relay_stage_.numel() *
                    this->a2av_relay_stage_.element_size(),
                this->cp_stream_inter_node));
          }
          bool pulled_peer = false;
          for (int dn = 1; dn < NN; dn++) {
            int tn = (my_node - dn + NN) % NN;
            const int64_t a_me = chunk_bound(my_node, tn, my_lr);
            const int64_t b_me = chunk_bound(my_node, tn, my_lr + 1);
            for (int sl = 0; sl < L; sl++) {
              const int64_t s0 = canon_start(my_node, sl, tn);
              const int64_t s1 = s0 + U_of(my_node, sl, tn);
              const int64_t lo = std::max(a_me, s0);
              const int64_t hi = std::min(b_me, s1);
              if (hi <= lo) {
                continue;
              }
              if (sl == my_lr) {
                if (a_me >= s0 && b_me <= s1) {
                  // single-source fast path (must mirror own_only below)
                  continue;
                }
                CUDA_CHECK(cudaMemcpyAsync(
                    relay_base + (relay_round_base(my_lr, dn) + (lo - a_me)) * row_bytes,
                    send_base + (my_seg_base(tn) + (lo - s0)) * row_bytes,
                    (hi - lo) * row_bytes,
                    cudaMemcpyDeviceToDevice,
                    this->cp_stream_inter_node));
                continue;
              }
              int prank = dist_env.local_rank_to_global_rank(sl, my_node);
              CU_CHECK(CUStreamWaitValue64(
                  this->cp_stream_inter_node,
                  reinterpret_cast<CUdeviceptr>(pack_ready + sl),
                  this->run_id_,
                  this->a2av_wait_flags()));
              pulled_peer = true;
              if (this->relay_blocking_pull_) {
                // T3 diagnostic: blocking get (local completion at return)
                nvshmemx_getmem_on_stream(
                    relay_base + (relay_round_base(my_lr, dn) + (lo - a_me)) * row_bytes,
                    send_base + (peer_seg_base(sl, tn) + (lo - s0)) * row_bytes,
                    (hi - lo) * row_bytes,
                    prank,
                    this->cp_stream_inter_node);
              } else {
                nvshmemx_getmem_nbi_on_stream(
                    relay_base + (relay_round_base(my_lr, dn) + (lo - a_me)) * row_bytes,
                    send_base + (peer_seg_base(sl, tn) + (lo - s0)) * row_bytes,
                    (hi - lo) * row_bytes,
                    prank,
                    this->cp_stream_inter_node);
              }
            }
          }
          // GEMM gate: piece transfers are issued (local nbi work; pull adds
          // pack-ready waits, which are acyclic); the wire loop below contains
          // cross-rank front-end waits and must NOT gate the GEMM launch
          // (forward_impl waits on this instead of fetch_remote_event in
          // relay mode)
          CUDA_CHECK(cudaEventRecord(this->relay_send_event_, this->cp_stream_inter_node));
          if (this->relay_fence_ && pulled_peer) {
            // F1: complete the phase-1 nbi gets BEFORE the wire reads
            // relay_base. Stream FIFO gives issue order only; a proxy-
            // lowered get can land after the put has read its source, which
            // ships the PREVIOUS epoch's relay bytes — invisible while
            // routing metadata and payloads repeat, corrupting under
            // per-iteration change. Own-only / self-copy rounds never set
            // pulled_peer and pay nothing.
            nvshmemx_quiet_on_stream(this->cp_stream_inter_node);
          }

          // ---- phase 2: wire loop, mirror node order. One contiguous put of
          // my chunk per round; node_sig keeps its single-writer-per-slot
          // semantics (the round's chunk k comes only from relay (ns, k)).
          std::vector<int> fenced_sig_targets;  // F2: signal after the quiet
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
              // pull phase 1 assembled my chunk with gets that are FIFO-
              // ordered before this wire put on the same stream — no inbound
              // signal to wait for (relay_sig retired with the push design)
              wire_src = relay_base + relay_round_base(my_lr, dn) * row_bytes;
            }
            if (this->wire_sig_fence_) {
              // F2: data only; the round's node_sig is SET after a PE-wide
              // quiet below, so the gateway's GEQ wait can never observe the
              // signal ahead of (or reordered across epochs relative to) the
              // chunk bytes it certifies.
              nvshmemx_putmem_nbi_on_stream(
                  stage_base + stage_off_chunk(tn, my_lr, my_node) * row_bytes,
                  wire_src,
                  rows * row_bytes,
                  g,
                  this->cp_stream_inter_node);
              fenced_sig_targets.push_back(g);
              continue;
            }
            // blocking_wire_ (instrumented): local-completion put whose proxy
            // entrypoint kernel spans the wire drain on the timeline
            if (this->blocking_wire_) {
              nvshmemx_putmem_signal_on_stream(
                  stage_base + stage_off_chunk(tn, my_lr, my_node) * row_bytes,
                  wire_src,
                  rows * row_bytes,
                  node_sig + my_node,
                  this->run_id_,
                  NVSHMEM_SIGNAL_SET,
                  g,
                  this->cp_stream_inter_node);
            } else {
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
          }
          if (this->wire_sig_fence_ && !fenced_sig_targets.empty()) {
            nvshmemx_quiet_on_stream(this->cp_stream_inter_node);
            for (int g : fenced_sig_targets) {
              nvshmemx_signal_op_on_stream(
                  node_sig + my_node,
                  this->run_id_,
                  NVSHMEM_SIGNAL_SET,
                  g,
                  this->cp_stream_inter_node);
            }
          }

          if (this->union_bcast_) {
            // ==== lb_union gateway forward (Tier B window gating): the
            // window is CONTIGUOUS in every destination's recv image (windows
            // are chunk_bound cuts of the canonical stream that the
            // source-major union regions concatenate into), so each round is
            // ONE contiguous put per destination, its fused signal flipping
            // the WINDOW's gating slot — the a2av_signal_buffer slot of
            // remote rank (ns, my_lr), re-keyed from source-identity to
            // delivering-window identity (the gating cumsum re-keys the
            // claimer + tile spin to match). Tiles gated on this window
            // unblock as it lands; the destination-side aggregation is gone.
            // Issued via t_* (inline; pack stream under early launch), NEVER
            // deferred: the GEMM reads these slots from launch, so writers
            // must be pre-launch-enqueued (channel wait-order-inversion rule).
            for (int dn = 1; dn < NN; dn++) {
              int ns = (my_node + dn) % NN;
              // FLUX_A2AV_FANOUT=1 (eager, NR-06 re-check): this round's wait
              // + puts enqueue on their own stream, so a late round's
              // node_sig wait (and its local-completion puts) never
              // head-of-line block a later round whose relay chunk already
              // landed. Knob off: rs == tail_stream, byte-identical to the
              // shipped ring order.
              const cudaStream_t rs = this->fanout_eager_
                                          ? (cudaStream_t)this->fanout_streams_[dn - 1]
                                          : tail_stream;
              if (this->nvshmem_wait_) {
                // F4: NVSHMEM-owned wait = consistency-enforced observation of
                // the RDMA-written chunk (one writer: relay (ns, my_lr))
                nvshmemx_signal_wait_until_on_stream(
                    node_sig + ns, NVSHMEM_CMP_GE, this->run_id_, rs);
              } else {
                CU_CHECK(CUStreamWaitValue64(
                    rs,
                    reinterpret_cast<CUdeviceptr>(node_sig + ns),
                    this->run_id_,
                    this->a2av_wait_flags()));  // one writer: relay (ns, my_lr)
              }
              const int64_t win_a = chunk_bound(ns, my_node, my_lr);
              const int64_t win_b = chunk_bound(ns, my_node, my_lr + 1);
              char *wstage = stage_base + stage_off_chunk(my_node, my_lr, ns) * row_bytes;
              uint64_t *wslot = signal_base + (int64_t)ns * L + my_lr;
              for (int dl = 0; dl < L; dl++) {
                // ring rotation: dest order staggered per (gateway, round) so
                // every destination receives its windows in a different
                // sequence — spreads first-window arrival across destinations
                // at zero extra resources (A/B/C verdict: capsule
                // 20260804-043026, significant at b2, never worse)
                int dlg = (my_lr + 1 + dn + dl) % L;
                int d = dist_env.local_rank_to_global_rank(dlg, my_node);
                char *dst = recv_base + (recv_off_of_u(ns * L, d) + win_a) * row_bytes;
                if (win_b <= win_a) {
                  nvshmemx_signal_op_on_stream(wslot, this->run_id_, NVSHMEM_SIGNAL_SET, d, rs);
                  continue;
                }
                nvshmemx_putmem_signal_on_stream(
                    dst,
                    wstage,
                    (win_b - win_a) * row_bytes,
                    wslot,
                    this->run_id_,
                    NVSHMEM_SIGNAL_SET,
                    d,
                    rs);
              }
              if (this->fanout_eager_) {
                CUDA_CHECK(cudaEventRecord(this->fanout_events_[dn - 1], rs));
              }
            }
            if (this->fanout_eager_) {
              // re-join the fan-out into the tail stream so all_gather_event
              // (and the pre-barrier wait at forward_impl) covers the eager
              // rounds exactly as it covers the ring order — the staging /
              // recv reuse invariant is unchanged
              for (int dn = 1; dn < NN; dn++) {
                CUDA_CHECK(cudaStreamWaitEvent(tail_stream, this->fanout_events_[dn - 1]));
              }
            }
          } else {
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
            // Issued via t_* — inline on the pack stream under early launch
            // (never deferred: see tail_stream above).
            t_wait_event(this->fwd_index_event_);
            for (int dn = 1; dn < NN; dn++) {
              int ns = (my_node + dn) % NN;
              t_wait(node_sig + ns);
              const int64_t win_rows = chunk_rows_of(ns, my_node, my_lr);
              char *wstage = stage_base + stage_off_chunk(my_node, my_lr, ns) * row_bytes;
              const int64_t stage_rows = std::max<int64_t>(win_rows, 1);
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
                  t_signal(gw_sig + (dn - 1) * L + my_lr, d);
                  continue;
                }
                for (int sl = 0; sl < L; sl++) {
                  const int64_t cnt = cnt_in_h[((int64_t)(dn - 1) * L + sl) * L + dlg];
                  if (cnt == 0) {
                    continue;
                  }
                  int s = dist_env.local_rank_to_global_rank(sl, ns);
                  const int64_t idx_off = fwd_col_off_h[((size_t)(dn - 1) * L + sl) * L + dlg];
                  const int64_t dst_off =
                      recv_off_of_u(s, d) + cnt_bef_h[((int64_t)(dn - 1) * L + sl) * L + dlg];
                  if (d == rank) {
                    t_index_select(
                        recv_base + dst_off * row_bytes, wstage, stage_rows, idx_off, cnt);
                  } else {
                    FLUX_CHECK_LE(sc + cnt, copies_per_rank)
                        << "a2av relay forward scratch overflow";
                    t_index_select(
                        scratch_base + sc * row_bytes, wstage, stage_rows, idx_off, cnt);
                    if (sl == last_sl) {
                      // NON-nbi (the scratch is refilled next round) with the
                      // per-round gateway signal fused on the LAST piece:
                      // intra-node on-stream puts to the same peer land in
                      // stream order (P2P copies), so the signal covers the
                      // earlier pieces too
                      t_blocking_put(
                          recv_base + dst_off * row_bytes,
                          scratch_base + sc * row_bytes,
                          cnt * row_bytes,
                          gw_sig + (dn - 1) * L + my_lr,
                          d);
                    } else {
                      t_put_nosig(
                          recv_base + dst_off * row_bytes,
                          scratch_base + sc * row_bytes,
                          cnt * row_bytes,
                          d);
                    }
                    sc += cnt;
                  }
                }
                if (d == rank) {
                  t_signal(gw_sig + (dn - 1) * L + my_lr, rank);
                }
              }
            }
          }  // !union_bcast_ (gather gateway)

          if (!this->union_bcast_) {
            // Tier B: lb_union gateways signal window slots directly; only the
            // gather arm still needs the per-source aggregation below
            // ---- destination-side signal aggregation (cp_stream_signal, pure
            // front-end memops): a source's (s, d) rows may now arrive via
            // several gateways, so the per-source epoch signals the GEMM spins
            // on get ONE writer again — me. Once all L gateway slots of a round
            // reach the epoch, every source of that node is fully delivered
            // (putmem_signal orders payload before signal), so write signal[s]
            // for ALL its sources, zero-traffic ones included.
            // Issued inline in every mode: pure front-end memops whose gw_sig
            // dependencies are themselves inline-issued (t_* tail above), so
            // pre-launch enqueue is the proven non-early configuration.
            for (int dn = 1; dn < NN; dn++) {
              int ns = (my_node + dn) % NN;
              for (int gl = 0; gl < L; gl++) {
                CU_CHECK(CUStreamWaitValue64(
                    this->cp_stream_signal,
                    reinterpret_cast<CUdeviceptr>(gw_sig + (dn - 1) * L + gl),
                    this->run_id_,
                    this->a2av_wait_flags()));
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
      emit_hier_event();
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
            if (this->blocking_wire_) {
              nvshmemx_putmem_signal_on_stream(
                  stage_base + seg_off(tn, my_lr, my_node) * row_bytes,
                  send_base + send_off[tn * L] * row_bytes,
                  rows * row_bytes,
                  node_sig + my_node,
                  this->run_id_,
                  NVSHMEM_SIGNAL_SET,
                  g,
                  this->cp_stream_inter_node);
            } else {
              nvshmemx_putmem_signal_nbi_on_stream(
                  stage_base + seg_off(tn, my_lr, my_node) * row_bytes,
                  send_base + send_off[tn * L] * row_bytes,
                  rows * row_bytes,
                  node_sig + my_node,
                  this->run_id_,
                  NVSHMEM_SIGNAL_SET,
                  g,
                  this->cp_stream_inter_node);
            }
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
          emit_wait(node_sig + ns);
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
                emit_self_copy(
                    recv_base + recv_off_of(s, rank) * row_bytes,
                    seg + within * row_bytes,
                    sub_rows * row_bytes);
              }
              emit_signal(signal_base + s, rank);
            } else if (sub_rows > 0) {
              emit_put(
                  recv_base + recv_off_of(s, d) * row_bytes,
                  seg + within * row_bytes,
                  sub_rows * row_bytes,
                  signal_base + s,
                  d);
            } else {
              emit_signal(signal_base + s, d);
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
    if (!defer_wire) {
      CUDA_CHECK(cudaEventRecord(this->fetch_remote_event, this->cp_stream_inter_node));
      CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream, this->fetch_remote_event));
      CUDA_CHECK(cudaEventRecord(this->all_gather_event, this->cp_stream));
    }
    // defer_wire: issue_deferred_wire() records these after replaying the
    // deferred cp_stream ops, so the iteration barrier still covers them

    if (kStage2AfterPuts) {
      build_stage2();
    }

    return A2AVDispatchState{
        sorted_gather_index, sorted_scatter_index, sorted_splits_cumsum, (int)M_this_ep};
  }

 public:
  // Dispatch-only entry for externally-composed pipelines (the EPIC
  // baseline): run the a2av wire WITHOUT the fused GEMM and materialize the
  // received rows into a dense [M_this_ep, hidden] tensor an external
  // grouped GEMM can consume. Completion gating is barrier-based (simple
  // and provably covers empty-window signal slots): the first
  // nvshmemx_barrier_all quiesces every rank's outstanding puts before the
  // gather reads the recv buffer; the second is the standard epoch-close
  // barrier (next iteration's remote puts must not race this epoch's
  // reads). EARLY_LAUNCH deferral is neutralized via defer_wire_arg=false.
  // Reads FLUX_A2AV_DISPATCH_ONLY_TAG (a no-op knob whose literal string in
  // the built .so is the sweep runner's capability probe for this method).
  // EPIC §4.3 phase 0 (in-kernel expert swap): launch the exchange kernel as
  // the FIRST device op of the dispatch launch sequence. In the dispatch_only
  // configuration (pack_overlap rejected, defer_wire=false) every subsequent
  // dispatch op runs on `stream` or on a cp-stream event-gated behind it, so
  // stream order alone gives the paper's "rebalance phase first, then the
  // dispatch phase" — sequential, one launch window, zero host sync between.
  // swap_fc1/swap_fc2 are THIS rank's slot storage views (contiguous; fc2
  // optional); the peer must be a same-node NVLink peer. One swap per rank
  // per call (the EPIC planner's pairing invariant sizes the scratch).
  void
  maybe_launch_inkernel_swap(
      c10::optional<torch::Tensor> const &swap_fc1,
      c10::optional<torch::Tensor> const &swap_fc2,
      int64_t swap_peer,
      int64_t swap_epoch,
      cudaStream_t stream) {
    if (swap_peer < 0) {
      return;
    }
    FLUX_CHECK(this->inkernel_swap_bytes_ > 0)
        << "in-kernel swap needs FLUX_A2AV_INKERNEL_SWAP=<scratch bytes> at ctor time";
    FLUX_CHECK(swap_fc1.has_value()) << "swap_peer set but no swap_fc1 slot view";
    const int L = world_size / nnodes;
    FLUX_CHECK(swap_peer != this->rank && swap_peer / L == this->rank / L)
        << "EPIC §4.3 swaps are strictly intra-node: peer " << swap_peer << " rank "
        << this->rank;
    // the epoch is the GLOBAL swap-round sequence (replicated; bumped every
    // round that has any swaps), shared by both pair members for the GEQ
    // handshake — a rank may skip rounds it does not participate in, so
    // monotonicity is strictly-greater, not +1
    FLUX_CHECK_GT(swap_epoch, (int64_t)this->swap_epoch_seen_)
        << "swap epoch must be monotone";
    auto const &fc1 = swap_fc1.value();
    FLUX_CHECK(fc1.is_cuda() && fc1.is_contiguous());
    int64_t fc1_bytes = (int64_t)fc1.nbytes();
    int64_t fc2_bytes = 0;
    void *fc2_ptr = nullptr;
    if (swap_fc2.has_value()) {
      auto const &fc2 = swap_fc2.value();
      FLUX_CHECK(fc2.is_cuda() && fc2.is_contiguous());
      fc2_bytes = (int64_t)fc2.nbytes();
      fc2_ptr = fc2.data_ptr();
    }
    FLUX_CHECK_EQ(fc1_bytes % 16, 0);
    FLUX_CHECK_EQ(fc2_bytes % 16, 0);
    FLUX_CHECK_LE(fc1_bytes + fc2_bytes, this->inkernel_swap_bytes_)
        << "swap payload exceeds the ctor-sized scratch";
    void *peer_scratch = nvshmem_ptr(this->swap_scratch_.data_ptr(), (int)swap_peer);
    uint64_t *peer_flag =
        reinterpret_cast<uint64_t *>(nvshmem_ptr(this->swap_flag_.data_ptr(), (int)swap_peer));
    FLUX_CHECK(peer_scratch != nullptr && peer_flag != nullptr)
        << "peer " << swap_peer << " not NVLink-reachable (nvshmem_ptr null)";
    static const bool kSwapTiming = get_int_from_env("FLUX_A2AV_TIMING", 0) != 0;
    EpicSwapParams p;
    p.my_fc1_slot = fc1.data_ptr();
    p.my_fc2_slot = fc2_ptr;
    p.fc1_bytes = fc1_bytes;
    p.fc2_bytes = fc2_bytes;
    p.my_scratch = this->swap_scratch_.data_ptr();
    p.peer_scratch = peer_scratch;
    p.my_flag = reinterpret_cast<uint64_t *>(this->swap_flag_.data_ptr());
    p.peer_flag = peer_flag;
    p.epoch = (uint64_t)swap_epoch;
    p.arrive = reinterpret_cast<unsigned long long *>(this->swap_arrive_.data_ptr());
    p.arrive_base = this->swap_arrive_base_;
    p.stamps =
        kSwapTiming ? reinterpret_cast<uint64_t *>(this->swap_stamps_.data_ptr()) : nullptr;
    FLUX_CHECK(this->swap_events_used_ < kSwapEventPool) << "swap event pool exhausted";
    CUDA_CHECK(cudaEventRecord(this->swap_events_[2 * this->swap_events_used_], stream));
    int num_sm = get_int_from_env("FLUX_A2AV_SWAP_NUM_SM", 16);
    int grid = epic_swap_exchange(p, num_sm, stream);
    CUDA_CHECK(cudaEventRecord(this->swap_events_[2 * this->swap_events_used_ + 1], stream));
    this->swap_events_used_ += 1;
    this->swap_arrive_base_ += 2ULL * (unsigned long long)grid;
    this->swap_epoch_seen_ = (uint64_t)swap_epoch;
    if (kSwapTiming) {
      // instrumented mode only (never compared against clean cells): sync the
      // stamps back and split the phase into snapshot / peer-wait / pull
      CUDA_CHECK(cudaMemcpyAsync(
          this->swap_stamps_pinned_.data_ptr(),
          this->swap_stamps_.data_ptr(),
          4 * sizeof(uint64_t),
          cudaMemcpyDeviceToHost,
          stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      auto const *s = reinterpret_cast<const uint64_t *>(this->swap_stamps_pinned_.data_ptr());
      // t1 (my release, last-arriving block) and t2 (block 0's peer-flag
      // observation) are unordered across blocks: when the peer released
      // first, t2 < t1 and the true exposed wait is zero — clamp instead of
      // wrapping. pull is t3 - max(t1, t2) (phase 4 starts after both).
      int64_t snap = (int64_t)(s[1] - s[0]);
      int64_t wait = (int64_t)s[2] - (int64_t)s[1];
      int64_t pull = (int64_t)s[3] - (int64_t)std::max(s[1], s[2]);
      fprintf(
          stderr,
          "[a2av-swap] rank %d epoch %ld snapshot %.3f wait %.3f pull %.3f ms\n",
          this->rank,
          (long)swap_epoch,
          snap / 1e6,
          std::max<int64_t>(wait, 0) / 1e6,
          pull / 1e6);
    }
  }

  // Drain the always-on swap timing events. Call once AFTER the timed loop
  // (values are per-launch, in launch order; same-stream event-elapsed =
  // kernel residency = snapshot + peer-wait + pull, which under the
  // sequential-phases design IS the exposed cost).
  std::vector<double>
  collect_swap_times() {
    std::vector<double> out;
    out.reserve(this->swap_events_used_);
    for (int i = 0; i < this->swap_events_used_; i++) {
      CUDA_CHECK(cudaEventSynchronize(this->swap_events_[2 * i + 1]));
      float ms = 0.f;
      CUDA_CHECK(
          cudaEventElapsedTime(&ms, this->swap_events_[2 * i], this->swap_events_[2 * i + 1]));
      out.push_back((double)ms);
    }
    return out;
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t>
  dispatch_only(
      torch::Tensor inputs_shard,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<torch::Tensor> splits_per_source,
      c10::optional<torch::Tensor> a2av_unique_counts,
      c10::optional<torch::Tensor> dense_out,
      c10::optional<torch::Tensor> swap_fc1,
      c10::optional<torch::Tensor> swap_fc2,
      int64_t swap_peer,
      int64_t swap_epoch) {
    (void)get_int_from_env("FLUX_A2AV_DISPATCH_ONLY_TAG", 0);
    (void)get_int_from_env("FLUX_A2AV_INKERNEL_SWAP_TAG", 0);
    (void)get_int_from_env("FLUX_A2AV_DELIVERY_GATE_TAG", 0);  // NR-16 D1 fix
    // 2026-08-22 wire-ordering finding: the nbi wire put_signal lets the
    // gateway observe node_sig before the chunk bytes (one-epoch-stale
    // forwards under per-iteration payload change; invisible with static
    // payloads). Verified fix = FLUX_A2AV_BLOCKING_WIRE=1; candidate
    // FLUX_A2AV_WIRE_SIGNAL_FENCE=1 (data nbi -> quiet -> signal). Binary tag:
    (void)get_int_from_env("FLUX_A2AV_WIRE_SIGNAL_FENCE_TAG", 0);
    (void)get_int_from_env("FLUX_A2AV_WAIT_FLUSH_TAG", 0);
    (void)get_int_from_env("FLUX_A2AV_NVSHMEM_WAIT_TAG", 0);
    FLUX_CHECK(a2av_dispatch_) << "dispatch_only requires an a2av-mode op";
    FLUX_CHECK(!pack_overlap_)
        << "dispatch_only is untested with FLUX_A2AV_PACK_OVERLAP";
    CHECK_INPUT(inputs_shard, this->input_dtype);
    CHECK_NDIM(inputs_shard, 2);
    CHECK_INPUT(splits_gpu, torch::kInt32);
    CHECK_NDIM(splits_gpu, 1);
    FLUX_CHECK_LE(this->nexperts, splits_gpu.size(0));
    CHECK_INPUT(scatter_index, torch::kInt32);
    CHECK_NDIM(scatter_index, 2);
    FLUX_CHECK_EQ(scatter_index.size(1), this->topk);

    const int32_t *cnt_host = nullptr;
    if (splits_per_source.has_value()) {
      auto const &cnt = splits_per_source.value();
      FLUX_CHECK(cnt.device().is_cpu()) << "splits_per_source must be a CPU tensor";
      FLUX_CHECK(cnt.scalar_type() == torch::kInt32);
      FLUX_CHECK(cnt.is_contiguous());
      CHECK_2D(cnt, world_size, this->nexperts);
      cnt_host = cnt.data_ptr<int32_t>();
    }
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
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // EPIC §4.3 phase 0: the swap kernel is the first device op of this
    // epoch's launch sequence; the token wire below is stream-ordered
    // strictly behind it (weights complete, THEN tokens move).
    this->maybe_launch_inkernel_swap(swap_fc1, swap_fc2, swap_peer, swap_epoch, stream);
    auto st = this->a2av_dispatch(
        inputs_shard, splits_gpu, scatter_index, cnt_host, uc_host, stream,
        /*defer_wire_arg=*/false);

    CUDA_CHECK(cudaStreamWaitEvent(stream, this->all_gather_event));
    if (a2av_hier_compress_ && nnodes > 1 && !relay_identity_ && !union_bcast_) {
      CUDA_CHECK(cudaStreamWaitEvent(stream, this->signal_done_event_));
    }
    // NR-16 D1 fix (2026-08-19): dispatch_only had NO receiver-side
    // delivery gate — the barrier alone raced at W32 (per-epoch transient
    // missing rows; masked by static payloads in multi-epoch runs, so 4n
    // always looked clean). Gate exactly like the fused path's proven 8n
    // contract: wait every source slot's epoch signal (the always-signal
    // invariant covers zero-row lanes on every compress wire) as zero-SM
    // front-end memops. The preceding quiet folds this PE's outstanding
    // nbi tails into the proxy first — all_gather_event already ordered us
    // after both cp streams' issue points — which also closes the
    // epoch-close reuse hazard (H-a) for the next iteration's staging.
    if (a2av_hier_compress_ && nnodes > 1) {
      nvshmemx_quiet_on_stream(stream);
      uint64_t *sig =
          reinterpret_cast<uint64_t *>(this->a2av_signal_buffer.data_ptr());
      for (int s = 0; s < this->world_size; s++) {
        CU_CHECK(CUStreamWaitValue64(
            stream,
            reinterpret_cast<CUdeviceptr>(sig + s),
            this->run_id_,
            this->a2av_wait_flags()));
      }
    }
    // delivery barrier: after this, every rank's epoch-n puts have landed
    nvshmemx_barrier_all_on_stream(stream);

    auto gidx = st.sorted_gather_index.narrow(0, 0, st.M_this_ep);
    torch::Tensor dense;
    if (dense_out.has_value()) {
      dense = dense_out.value();
      CHECK_INPUT(dense, this->input_dtype);
      CHECK_2D(dense, st.M_this_ep, this->hidden);
      at::index_select_out(dense, this->a2av_recv_buffer, 0, gidx);
    } else {
      dense = at::index_select(this->a2av_recv_buffer, 0, gidx);
    }

    // epoch-close barrier: iteration n+1 remote puts must not race the
    // gather above (dense is private memory past this point)
    if (this->epoch_quiet_) {
      // T4 diagnostic (only if T1 fails): explicit PE-wide drains on every
      // issuing stream before the barrier — a fix here would indicate the
      // platform barrier lacks full quiet semantics for proxy ops
      nvshmemx_quiet_on_stream(this->cp_stream);
      nvshmemx_quiet_on_stream(this->cp_stream_inter_node);
      nvshmemx_quiet_on_stream(stream);
    }
    nvshmemx_barrier_all_on_stream(stream);
    return {dense, st.sorted_scatter_index, st.sorted_splits_cumsum,
            (int64_t)st.M_this_ep};
  }

  // ---- campaign-2 v2b: in-window hc metadata (rule 5) -------------------
  //
  // dispatch_only_routed derives splits / scatter_index /
  // splits_per_source / a2av_unique_counts ON DEVICE from the raw
  // replicated routing, D2H's the two small host-consumed matrices into
  // pinned staging (event-synced — an honest in-window sync, precedents
  // at the non-meta chunks path and the relay cnt_in sync), and delegates
  // to dispatch_only unchanged. Capacity knobs (FLUX_A2AV_MAX_*) remain
  // setup-computed ctor state: allocation is deployment scope, CONTENTS
  // are per-iteration. Determinism note: the stable scatter index is
  // bit-identical to python argsort(stable).argsort() — replicated
  // cross-rank data must never come from the non-deterministic
  // calc_scatter_index.

  void
  ensure_routed_meta() {
    if (routed_meta_ready_) {
      return;
    }
    const int W = this->world_size;
    const int64_t E = this->nexperts;
    const int64_t n_copies = (int64_t)this->max_ntokens * this->topk;
    const int32_t nblocks =
        (int32_t)((n_copies + kA2AVMetaTile - 1) / kA2AVMetaTile);
    auto dev = torch::TensorOptions(torch::kCUDA).dtype(torch::kInt32);
    auto pin = torch::TensorOptions(torch::kCPU)
                   .dtype(torch::kInt32)
                   .pinned_memory(true);
    rt_splits_dev_ = torch::zeros({E}, dev);
    rt_scatter_dev_ = torch::zeros({(int64_t)this->max_ntokens, this->topk},
                                   dev);
    rt_sps_dev_ = torch::zeros({(int64_t)W, E}, dev);
    rt_uc_dev_ = torch::zeros({(int64_t)W, W + this->nnodes}, dev);
    rt_sps_cpu_ = torch::zeros({(int64_t)W, E}, pin);
    rt_uc_cpu_ = torch::zeros({(int64_t)W, W + this->nnodes}, pin);
    rt_block_hist_ = torch::zeros({(int64_t)nblocks, E}, dev);
    rt_block_offset_ = torch::zeros({(int64_t)nblocks, E}, dev);
    rt_expert_base_ = torch::zeros({E + 1}, dev);
    CUDA_CHECK(cudaEventCreateWithFlags(&rt_meta_event_,
                                        cudaEventDisableTiming));
    routed_meta_ready_ = true;
  }

  std::vector<torch::Tensor>
  derive_routed_meta(torch::Tensor topk_ids) {
    (void)get_int_from_env("FLUX_A2AV_INWINDOW_META_TAG", 0);
    CHECK_INPUT(topk_ids, torch::kInt32);
    CHECK_NDIM(topk_ids, 2);
    const int W = this->world_size;
    const int64_t ntok = topk_ids.size(0);
    FLUX_CHECK_EQ(topk_ids.size(1), this->topk);
    FLUX_CHECK(ntok % W == 0);
    FLUX_CHECK_LE(ntok, this->max_ntokens);
    ensure_routed_meta();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int64_t E = this->nexperts;
    const int64_t n_copies = ntok * this->topk;
    CUDA_CHECK(cudaMemsetAsync(rt_splits_dev_.data_ptr(), 0,
                               E * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(rt_sps_dev_.data_ptr(), 0,
                               (size_t)W * E * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        rt_uc_dev_.data_ptr(), 0,
        (size_t)W * (W + this->nnodes) * sizeof(int32_t), stream));
    A2AVMetaCountsArguments margs{
        topk_ids.data_ptr<int32_t>(),
        ntok,
        (int32_t)this->topk,
        (int32_t)E,
        (int32_t)this->ep_nexperts,
        (int32_t)W,
        (int32_t)this->nnodes,
        (int32_t)(W / this->nnodes),
        ntok / W,
        rt_splits_dev_.data_ptr<int32_t>(),
        rt_sps_dev_.data_ptr<int32_t>(),
        rt_uc_dev_.data_ptr<int32_t>()};
    a2av_meta_counts_impl(margs, stream);
    A2AVStableScatterArguments sargs{
        topk_ids.data_ptr<int32_t>(),
        n_copies,
        (int32_t)E,
        rt_block_hist_.data_ptr<int32_t>(),
        rt_block_offset_.data_ptr<int32_t>(),
        rt_expert_base_.data_ptr<int32_t>(),
        rt_scatter_dev_.data_ptr<int32_t>(),
        (int32_t)((n_copies + kA2AVMetaTile - 1) / kA2AVMetaTile)};
    a2av_stable_scatter_index_impl(sargs, stream);
    CUDA_CHECK(cudaMemcpyAsync(rt_sps_cpu_.data_ptr(), rt_sps_dev_.data_ptr(),
                               (size_t)W * E * sizeof(int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        rt_uc_cpu_.data_ptr(), rt_uc_dev_.data_ptr(),
        (size_t)W * (W + this->nnodes) * sizeof(int32_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(rt_meta_event_, stream));
    CUDA_CHECK(cudaEventSynchronize(rt_meta_event_));
    return {rt_splits_dev_,
            rt_scatter_dev_.narrow(0, 0, ntok),
            rt_sps_cpu_, rt_uc_cpu_};
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t>
  dispatch_only_routed(
      torch::Tensor inputs_shard,
      torch::Tensor topk_ids,
      c10::optional<torch::Tensor> dense_out,
      c10::optional<torch::Tensor> swap_fc1,
      c10::optional<torch::Tensor> swap_fc2,
      int64_t swap_peer,
      int64_t swap_epoch) {
    auto meta = derive_routed_meta(topk_ids);
    return dispatch_only(
        inputs_shard, meta[0], meta[1],
        c10::optional<torch::Tensor>(rt_sps_cpu_),
        c10::optional<torch::Tensor>(rt_uc_cpu_), dense_out, swap_fc1,
        swap_fc2, swap_peer, swap_epoch);
  }

 private:
  bool routed_meta_ready_ = false;
  torch::Tensor rt_splits_dev_, rt_scatter_dev_, rt_sps_dev_, rt_uc_dev_;
  torch::Tensor rt_sps_cpu_, rt_uc_cpu_;
  torch::Tensor rt_block_hist_, rt_block_offset_, rt_expert_base_;
  cudaEvent_t rt_meta_event_ = nullptr;

 protected:
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
      c10::optional<torch::Tensor> weight_signal,
      int64_t weight_signal_epoch,
      int64_t weight_gate_group_start,
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
      // Union bcast is exempt: its forwards are pure copy-engine puts.
      FLUX_CHECK(sm_margin >= 1 || nnodes == 1 || union_bcast_)
          << "a2av_hier_compress with nnodes > 1 requires sm_margin >= 1 "
             "(unless FLUX_A2AV_UNION_BCAST=1)";
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
          this->a2av_dispatch(
              inputs_shard, splits_gpu, scatter_index, cnt_host, uc_host, stream,
              /*defer_wire_arg=*/this->early_launch_);
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
        .accum_per_rank_ptr =
            (this->union_bcast_ && !this->relay_identity_ && this->a2av_gating_cumsum_.defined())
                ? this->a2av_gating_cumsum_.data_ptr<int32_t>()
                : sorted_splits_cumsum.data_ptr<int32_t>(),
        .tile_size_m = tile_M,
        .tile_size_n = tile_N,
        .barrier_ptr = barrier_ptr};
    args.seg_gate_ballot = this->seg_gate_ballot_;
    if (a2av_dispatch_) {
      args.signal_ptr = reinterpret_cast<uint64_t *>(this->a2av_signal_buffer.data_ptr());
      args.signal_expected = this->run_id_;
      // Tier B: lb_union runs the claimer over window-keyed buckets (its
      // gating cumsum + window-slot signals re-key the whole path); the other
      // hier modes keep the dense static schedule
      // Tier B candidate D: dense static schedule + window-keyed per-tile
      // spin (fine-grained release, zero claimer overhead at G=128 geometry)
      args.a2av_ring_schedule = a2av_ring_ || a2av_hier_ || a2av_hier_compress_;
      if (this->nvtx_proxy_enabled_) {
        args.progress_slots = this->progress_slots_dev_;
      }
    }
    if (weight_signal.has_value() && weight_gate_group_start >= 0) {
      // weight-gated tiles (moonep_fused scenario 2): prefetch-slot problems
      // (local group >= start) spin on their slot's weight epoch signal;
      // local-expert problems never wait. Requires the a2av static schedule
      // (the dynamic claimer's buckets are token-keyed only).
      FLUX_CHECK(a2av_dispatch_ && args.a2av_ring_schedule)
          << "weight-gated tiles require an a2av static-schedule mode";
      FLUX_CHECK(weight_signal->is_cuda());
      FLUX_CHECK(weight_signal->scalar_type() == at::ScalarType::Long)
          << "weight_signal must be int64 (u64 epoch signals)";
      FLUX_CHECK(weight_gate_group_start > 0 && weight_gate_group_start < ep_nexperts)
          << "weight_gate_group_start " << weight_gate_group_start << " out of (0, "
          << ep_nexperts << ")";
      FLUX_CHECK(weight_signal->numel() >= ep_nexperts - weight_gate_group_start)
          << "weight_signal too small for the prefetch-slot groups";
      args.weight_signal_ptr = reinterpret_cast<uint64_t *>(weight_signal->data_ptr());
      args.weight_signal_expected = static_cast<uint64_t>(weight_signal_epoch);
      args.weight_gate_group_start = static_cast<int>(weight_gate_group_start);
      // F-D (NR-13 / NR-14): schedule every prefetch-slot problem after every
      // resident problem so the weight spin lands at the tail of the
      // wavefront, overlapped with resident compute. Scoped to the weight-gate
      // branch: without a gate boundary there is no class to reorder.
      // DEFAULT OFF. Canonicalization was attempted on 2026-08-15 and REVERTED:
      // the -15% b64 "win" of capsule 20260814-145605 did not survive an
      // order-controlled repeat. Six capsules (20260815-124733/-124853/-125007
      // forward order, -125313/-125432/-125546 reversed, 20 iters each) put
      // slot-last at 27.9-29.1 ms vs 22.1-25.2 ms interleaved at b64 — a ~29%
      // REGRESSION that follows the arm, not the run position. Measured
      // mechanism (tile-trace capsule 20260815-132439, NR-14 amendment
      // 2026-08-15b): NOT weight-gate spin — the dense static schedule is
      // head-of-line, and deferring the slot class thins the schedule prefix
      // ahead of the first inter-node token segment (own-node resident work
      // ~0.5 ms vs ~6.6 ms arrival on the worst rank), parking the entire
      // wavefront at zero compute for 5-6 ms every iteration. Interleaved
      // slot tiles are the ballast that keeps the sweep behind the arrival
      // front; the weight wait they'd defer is negligible at b64.
      static const bool kSchedPrefetchLast =
          get_int_from_env("FLUX_A2AV_SCHED_PREFETCH_LAST", 0) != 0;
      args.sched_prefetch_last = kSchedPrefetchLast;
      // one-shot audit line (rank 0): the timing capsules cannot distinguish
      // "the flag flipped" from "a whole-cell transient", so make the RESOLVED
      // value observable in every run's log rather than inferred from ms.
      static const bool kAuditOnce = [&] {
        if (get_rank_from_env() == 0) {
          fprintf(
              stderr,
              "[a2av] sched_prefetch_last=%d (gate_start=%d, ep_nexperts=%d)\n",
              static_cast<int>(kSchedPrefetchLast),
              args.weight_gate_group_start,
              ep_nexperts);
        }
        return true;
      }();
      (void)kAuditOnce;
    }
    for (int gid = 0; gid < num_weights_group; gid++) {
      args.weight[gid] = weights[gid].data_ptr();
      args.output[gid] = outputs[gid].data_ptr();
      args.scaleD[gid] =
          output_scales.has_value() ? output_scales->at(gid).data_ptr<float>() : nullptr;
    }

    static const bool kA2avTiming = get_int_from_env("FLUX_A2AV_TIMING", 0) != 0;
    // FLUX_A2AV_NO_GEMM_GATE=1 (a2av only): launch the GEMM without waiting
    // for the dispatch events. Correctness is carried by the per-tile signal
    // spins; the CUDA_DEVICE_MAX_CONNECTIONS=1 invariant (every kernel the
    // comm path needs precedes the GEMM in the single queue) still holds
    // because removing the WAITS does not change ENQUEUE order. Note the
    // events these waits target are stream-ordered AFTER the intra put
    // payloads (putmem_signal orders signal-after-payload), so the gate
    // actually waits for this rank's own intra CE copies to COMPLETE, not
    // merely for their issue — skipping it converts that wait into in-kernel
    // spin overlapped with compute.
    static const bool kNoGemmGate = get_int_from_env("FLUX_A2AV_NO_GEMM_GATE", 0) != 0;
    if (a2av_dispatch_) {
      if (kNoGemmGate || this->early_launch_) {
        // no event waits: GEMM starts as soon as stage-2 outputs are ready
        // (early_launch_: the intra wire has not even been issued yet)
      } else if (a2av_hier_ || a2av_hier_compress_) {
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

      if (this->nvtx_proxy_) {
        if (this->tile_trace_dev_ == nullptr) {
          // worst-case tiles per iteration: recv-buffer row capacity plus one
          // partial tile per expert, x tile columns x weight groups
          const int grid_n = (N + tile_N - 1) / tile_N;
          const int64_t tiled_m_max = this->max_recv_ntokens_ / tile_M + ep_nexperts;
          // x FLUX_A2AV_TRACE_EPOCHS iterations of headroom: the sidecar
          // drain is decoupled from the stream (snapshot ring), so the
          // device ring must survive the drain lagging a few epochs behind
          const int64_t trace_epochs =
              std::max<int64_t>(1, get_int_from_env("FLUX_A2AV_TRACE_EPOCHS", 4));
          this->tile_trace_capacity_ = (uint32_t)std::min<int64_t>(
              tiled_m_max * grid_n * (int64_t)num_weights_group * trace_epochs, 0x7FFFFFFF);
          CUDA_CHECK(cudaMalloc(
              (void **)&this->tile_trace_dev_,
              sizeof(A2AVTileRecord) * (size_t)this->tile_trace_capacity_));
          this->nvtx_proxy_->set_tile_trace(this->tile_trace_dev_, this->tile_trace_capacity_);
        }
        args.tile_trace = this->tile_trace_dev_;
        args.tile_trace_capacity = this->tile_trace_capacity_;
        // dense-schedule expected tiles per gating source: a tile counts for
        // its segment_end (last source its M rows span), mirroring the
        // kernel's ballot in process_tile; x tile columns x weight groups.
        // rows[] is the per-source ROW ground truth (sum over experts) — the
        // sidecar carries it so analyses can tell small segments from empty
        // ones (tile attribution alone cannot).
        std::vector<uint32_t> expected, src_rows;
        if (this->union_bcast_ && !this->relay_identity_ && nnodes > 1) {
          // Tier B: buckets are window-keyed and the claimer publishes exact
          // bucket totals from the device — host expected[] would be
          // source-keyed and SHADOW them in the sidecar, so leave it empty.
          // src_rows: local sources keep u rows; remote slots carry the
          // WINDOW's row count (host-known from chunk bounds).
          const int Lw = world_size / nnodes;
          const int my_node_w = rank / Lw;
          src_rows.assign(world_size + 1, 0);
          for (int s = 0; s < world_size; ++s) {
            if (s / Lw == my_node_w) {
              if (!this->nvtx_ssc_.empty()) {
                uint32_t acc = 0;
                for (int e = 0; e < ep_nexperts; ++e) {
                  const int32_t *row = this->nvtx_ssc_.data() + (int64_t)e * world_size;
                  acc += (uint32_t)(row[s] - (s ? row[s - 1] : 0));
                }
                src_rows[s] = acc;
              }
            } else {
              // window rows (remote slot = delivering window), staged by the
              // meta block where the chunk lambdas are in scope
              src_rows[s] =
                  s < (int)this->nvtx_window_rows_.size() ? this->nvtx_window_rows_[s] : 0u;
            }
          }
        } else if (!this->nvtx_ssc_.empty()) {
          const int W = world_size;
          const int grid_n = (N + tile_N - 1) / tile_N;
          expected.assign(W + 1, 0);
          src_rows.assign(W + 1, 0);
          for (int e = 0; e < ep_nexperts; ++e) {
            const int32_t *row = this->nvtx_ssc_.data() + (int64_t)e * W;
            const int32_t M_e = row[W - 1];
            for (int s = 0; s < W; ++s) {
              src_rows[s] += (uint32_t)(row[s] - (s ? row[s - 1] : 0));
            }
            for (int64_t t = 0; t * tile_M < M_e; ++t) {
              int32_t last = (int32_t)std::min<int64_t>((t + 1) * tile_M, M_e) - 1;
              int seg_end = int(std::upper_bound(row, row + W, last) - row);
              expected[seg_end] += (uint32_t)(grid_n * num_weights_group);
            }
          }
        }
        // stream-ordered iteration bracket: runs after the GEMM gate, right
        // before the kernel (and its close after), so the poller needs no sync
        CUDA_CHECK(cudaLaunchHostFunc(
            stream,
            A2AVNvtxProxy::iter_start_cb,
            new A2AVNvtxProxy::IterStart{
                this->nvtx_proxy_.get(),
                this->run_id_,
                std::move(expected),
                std::move(src_rows)}));
      }
      // Step 5: launch GEMM
      op->run(args, workspace_size ? this->workspace_buffer.data_ptr() : nullptr, stream);
    }
    // FLUX_A2AV_EARLY_LAUNCH: replay the deferred intra wire now that the GEMM
    // is launched (or skipped: M_this_ep == 0 still owes the peers its wire)
    this->issue_deferred_wire();
    if (this->nvtx_proxy_ && M_this_ep > 0) {
      // enqueued AFTER the deferred wire in HOST order: under
      // CUDA_DEVICE_MAX_CONNECTIONS=1 a pending host function blocks the
      // single channel, and wire ops enqueued behind it would deadlock
      // against the spinning GEMM the callback waits on (observed hang).
      // Stream position is unchanged (still right after the GEMM): the D2H
      // snapshot + publish callback capture this epoch's final slot state in
      // stream order, so the sidecar cannot lose epochs to poller stalls.
      this->nvtx_proxy_->enqueue_epoch_snapshot(stream, this->run_id_);
    }
    if (a2av_dispatch_ && kA2avTiming) {
      CUDA_CHECK(cudaEventRecord(this->timing_events_[4], stream));
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, this->all_gather_event));
    if (a2av_hier_compress_ && nnodes > 1 && !relay_identity_ && !union_bcast_) {
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
      if (this->pack_overlap_ && a2av_hier_compress_) {
        // pack overlap: ready_event moves to pack_stream_, so the "sends n+1
        // behind barrier n" ordering above is carried by this event instead
        // (waited by both cp streams at the top of the next dispatch)
        CUDA_CHECK(cudaEventRecord(this->barrier_done_event_, stream));
      }
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
          "[a2av-timing] rank %d stage1 %.3f stage2 %.3f gemmgate %.3f gemm %.3f barrier %.3f "
          "ms\n",
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
      // balanced gather only: lb_union (union_bcast_) runs no fwd-index build,
      // so its relay_fwd_events_ are never recorded (ElapsedTime would throw)
      if (a2av_hier_compress_ && nnodes > 1 && !relay_identity_ && !union_bcast_) {
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
        c10::nullopt,
        0,
        -1,
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
      c10::optional<torch::Tensor> a2av_unique_counts,
      c10::optional<torch::Tensor> weight_signal,
      int64_t weight_signal_epoch,
      int64_t weight_gate_group_start) {
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
        std::move(weight_signal),
        weight_signal_epoch,
        weight_gate_group_start,
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
                c10::nullopt,
                0,
                -1,
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
        c10::nullopt,
        0,
        -1,
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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t>
GemmGroupedV2AGScatterOp::dispatch_only(
    torch::Tensor inputs_shard,
    torch::Tensor splits_gpu,
    torch::Tensor scatter_index,
    c10::optional<torch::Tensor> splits_per_source,
    c10::optional<torch::Tensor> a2av_unique_counts,
    c10::optional<torch::Tensor> dense_out,
    c10::optional<torch::Tensor> swap_fc1,
    c10::optional<torch::Tensor> swap_fc2,
    int64_t swap_peer,
    int64_t swap_epoch) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2AGScatterOp is not initialized";
  return impl_->dispatch_only(
      std::move(inputs_shard),
      std::move(splits_gpu),
      std::move(scatter_index),
      std::move(splits_per_source),
      std::move(a2av_unique_counts),
      std::move(dense_out),
      std::move(swap_fc1),
      std::move(swap_fc2),
      swap_peer,
      swap_epoch);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t>
GemmGroupedV2AGScatterOp::dispatch_only_routed(
    torch::Tensor inputs_shard,
    torch::Tensor topk_ids,
    c10::optional<torch::Tensor> dense_out,
    c10::optional<torch::Tensor> swap_fc1,
    c10::optional<torch::Tensor> swap_fc2,
    int64_t swap_peer,
    int64_t swap_epoch) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2AGScatterOp is not initialized";
  return impl_->dispatch_only_routed(
      std::move(inputs_shard),
      std::move(topk_ids),
      std::move(dense_out),
      std::move(swap_fc1),
      std::move(swap_fc2),
      swap_peer,
      swap_epoch);
}

std::vector<torch::Tensor>
GemmGroupedV2AGScatterOp::derive_routed_meta(torch::Tensor topk_ids) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2AGScatterOp is not initialized";
  return impl_->derive_routed_meta(std::move(topk_ids));
}

std::vector<double>
GemmGroupedV2AGScatterOp::collect_swap_times() {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2AGScatterOp is not initialized";
  return impl_->collect_swap_times();
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
    c10::optional<torch::Tensor> a2av_unique_counts,
    c10::optional<torch::Tensor> weight_signal,
    int64_t weight_signal_epoch,
    int64_t weight_gate_group_start) {
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
      std::move(a2av_unique_counts),
      std::move(weight_signal),
      weight_signal_epoch,
      weight_gate_group_start);
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
