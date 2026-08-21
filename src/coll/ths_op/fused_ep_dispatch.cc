//===- fused_ep_dispatch.cc --------------------------------------- C++ ---===//
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
#include "coll/ths_op/fused_ep_dispatch.h"

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/all.h>

#include <cstdlib>
#include <vector>

#include "coll/fused_ep_dispatch_impl.hpp"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/utils.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"

namespace bytedance {
namespace flux {
namespace ths_op {

using torch::Tensor;

namespace {
constexpr int32_t kHistTileHost = 2048;  // must match kHistTile in the .cu
}

class FusedEpDispatch::FusedEpDispatchImpl {
 private:
  std::shared_ptr<Group> pg_;
  int32_t rank_;
  int32_t world_size_;
  int64_t nnodes_;
  int64_t s_max_, hidden_, topk_, nlp_;
  int64_t max_rows_per_pair_, max_recv_total_;
  at::ScalarType dtype_;
  int64_t m_groups_;
  int64_t spin_limit_;
  int64_t row_bytes_;
  int32_t num_hist_blocks_max_;
  uint64_t run_id_ = 0;
  int64_t last_s_ = 0;

  // symmetric (put sources included — proxied transports require it)
  torch::Tensor counts_sym_, recv_data_sym_, recv_hdr_sym_, comb_data_sym_;
  torch::Tensor sig_counts_, sig_data_, sig_comb_;
  torch::Tensor probe_payload_, probe_sig_, prime_buf_, prime_sig_;
  torch::Tensor my_counts_, pack_data_, pack_hdr_, comb_stage_sym_;
  // device scratch
  torch::Tensor block_hist_, block_offset_, pack_base_;
  torch::Tensor remote_base_, recv_off_, seg_meta_;
  torch::Tensor weights_out_, probe_err_;
  // pinned + event for the seg D2H
  torch::Tensor seg_pinned_;
  cudaEvent_t seg_event_;

  FusedEpDispatchParams
  make_params(int64_t s_now) {
    FusedEpDispatchParams p{};
    p.rank = rank_;
    p.world_size = world_size_;
    p.nlp = (int32_t)nlp_;
    p.S = (int32_t)s_now;
    p.K = (int32_t)topk_;
    p.row_bytes = row_bytes_;
    p.my_counts = my_counts_.data_ptr<int32_t>();
    p.block_hist = block_hist_.data_ptr<int32_t>();
    p.pack_base = pack_base_.data_ptr<int32_t>();
    p.block_offset = block_offset_.data_ptr<int32_t>();
    p.pack_data = pack_data_.data_ptr();
    p.pack_hdr = pack_hdr_.data_ptr<int32_t>();
    p.remote_base = remote_base_.data_ptr<int64_t>();
    p.seg_meta = seg_meta_.data_ptr<int32_t>();
    p.recv_off = recv_off_.data_ptr<int32_t>();
    p.counts_sym = counts_sym_.data_ptr<int32_t>();
    p.recv_data_sym = recv_data_sym_.data_ptr();
    p.recv_hdr_sym = recv_hdr_sym_.data_ptr<int32_t>();
    p.sig_counts = reinterpret_cast<uint64_t *>(sig_counts_.data_ptr());
    p.sig_data = reinterpret_cast<uint64_t *>(sig_data_.data_ptr());
    p.max_rows_per_pair = (int32_t)max_rows_per_pair_;
    p.max_recv_total = max_recv_total_;
    p.run_id = run_id_;
    p.num_hist_blocks =
        (int32_t)((s_now * topk_ + kHistTileHost - 1) / kHistTileHost);
    return p;
  }

 public:
  FusedEpDispatchImpl(
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
      int64_t spin_limit)
      : pg_(pg),
        rank_(pg->get_rank()),
        world_size_(pg->get_size()),
        nnodes_(nnodes),
        s_max_(s_max),
        hidden_(hidden),
        topk_(topk),
        nlp_(nlp),
        max_rows_per_pair_(max_rows_per_pair),
        max_recv_total_(max_recv_total),
        dtype_(dtype),
        m_groups_(m_groups),
        spin_limit_(spin_limit) {
    // capability tag (sweep probe greps the raw .so bytes for this string)
    (void)get_int_from_env("FLUX_FUSED_EP_DISPATCH_TAG", 0);
    FLUX_CHECK(s_max > 0 && hidden > 0 && topk > 0 && nlp > 0);
    FLUX_CHECK(m_groups >= 1 && nlp % m_groups == 0);
    row_bytes_ = hidden * c10::elementSize(dtype);
    FLUX_CHECK(row_bytes_ % 16 == 0) << "row_bytes must be 16B aligned";
    // conn=1 would deadlock a resident recv gate against a later-enqueued
    // transport kernel (NR-15 lesson); refuse loudly.
    const char *conn = std::getenv("CUDA_DEVICE_MAX_CONNECTIONS");
    FLUX_CHECK(conn != nullptr && std::atoi(conn) > 1)
        << "FusedEpDispatch requires CUDA_DEVICE_MAX_CONNECTIONS > 1";

    // collective geometry contract (silent wire corruption otherwise)
    struct Cfg {
      int64_t s_max, hidden, topk, nlp, mrpp, mrt, mg;
      int32_t dtype;
    } mine{s_max, hidden, topk, nlp, max_rows_per_pair, max_recv_total,
           m_groups, (int32_t)dtype};
    std::vector<Cfg> all(world_size_);
    pg_->all_gather_cpu(&mine, all.data(), sizeof(Cfg));
    for (int r = 0; r < world_size_; ++r) {
      FLUX_CHECK(
          all[r].s_max == mine.s_max && all[r].hidden == mine.hidden &&
          all[r].topk == mine.topk && all[r].nlp == mine.nlp &&
          all[r].mrpp == mine.mrpp && all[r].mrt == mine.mrt &&
          all[r].mg == mine.mg && all[r].dtype == mine.dtype)
          << "FusedEpDispatch ctor config diverges across ranks (rank " << r
          << " vs " << rank_ << ") — exact-offset addressing requires "
             "identical geometry everywhere";
    }

    const int64_t P = (int64_t)world_size_ * nlp_;
    // symmetric (collective, identical order on every rank; signals are
    // calloc'd u64 and NEVER memset afterwards — epoch-monotone)
    counts_sym_ = nvshmem_create_tensor({(int64_t)world_size_, P},
                                        at::ScalarType::Int, true);
    recv_data_sym_ = nvshmem_create_tensor({max_recv_total_, hidden_},
                                           dtype_, true);
    recv_hdr_sym_ = nvshmem_create_tensor({max_recv_total_, 4},
                                          at::ScalarType::Int, true);
    comb_data_sym_ = nvshmem_create_tensor({s_max_ * topk_, hidden_},
                                           dtype_, true);
    sig_counts_ = nvshmem_create_tensor({(int64_t)world_size_},
                                        at::ScalarType::Long, true);
    sig_data_ = nvshmem_create_tensor({nlp_ * world_size_},
                                      at::ScalarType::Long, true);
    sig_comb_ = nvshmem_create_tensor({(int64_t)world_size_},
                                      at::ScalarType::Long, true);
    probe_payload_ = nvshmem_create_tensor({2 * 4096},
                                           at::ScalarType::Int, true);
    probe_sig_ = nvshmem_create_tensor({1}, at::ScalarType::Long, true);
    prime_buf_ = nvshmem_create_tensor({16}, at::ScalarType::Int, true);
    prime_sig_ = nvshmem_create_tensor({1}, at::ScalarType::Long, true);
    // put SOURCES must also live on the symmetric heap: on the proxied
    // transports (libfabric/CXI here) the proxy resolves the local side of
    // an RMA through the registered symmetric segment — an ordinary CUDA
    // tensor as source works over NVLink P2P but segfaults the host proxy
    // thread at nnodes > 1 (nvshmemt_libfabric_rma, NULL mr deref;
    // root-caused 2026-08-20)
    my_counts_ = nvshmem_create_tensor({P}, at::ScalarType::Int, true);
    pack_data_ = nvshmem_create_tensor({s_max_ * topk_, hidden_}, dtype_,
                                       true);
    pack_hdr_ = nvshmem_create_tensor({s_max_ * topk_, 4},
                                      at::ScalarType::Int, true);
    comb_stage_sym_ = nvshmem_create_tensor({max_recv_total_, hidden_},
                                            dtype_, true);

    // local scratch
    auto dev = recv_data_sym_.device();
    auto i32 = torch::TensorOptions(dev).dtype(at::ScalarType::Int);
    num_hist_blocks_max_ =
        (int32_t)((s_max_ * topk_ + kHistTileHost - 1) / kHistTileHost);
    block_hist_ = torch::zeros({(int64_t)num_hist_blocks_max_, P}, i32);
    block_offset_ = torch::zeros({(int64_t)num_hist_blocks_max_, P}, i32);
    pack_base_ = torch::zeros({P + 1}, i32);
    remote_base_ = torch::zeros({P},
                                torch::TensorOptions(dev).dtype(at::kLong));
    recv_off_ = torch::zeros({nlp_ * world_size_}, i32);
    seg_meta_ = torch::zeros({2 * nlp_}, i32);
    weights_out_ = torch::zeros({max_recv_total_},
                                torch::TensorOptions(dev).dtype(at::kFloat));
    probe_err_ = torch::zeros({1}, i32);
    seg_pinned_ = torch::zeros(
        {2 * nlp_},
        torch::TensorOptions(at::kCPU).dtype(at::ScalarType::Int)
            .pinned_memory(true));
    CUDA_CHECK(cudaEventCreateWithFlags(&seg_event_,
                                        cudaEventDisableTiming));

    // priming (lazy-load hang class): every transport kernel class this op
    // uses, against dedicated scratch — the real signals stay untouched.
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    char *pb = static_cast<char *>(prime_buf_.data_ptr());
    uint64_t *ps = reinterpret_cast<uint64_t *>(prime_sig_.data_ptr());
    const int64_t L = world_size_ / (nnodes_ > 0 ? nnodes_ : 1);
    std::vector<int> peers{rank_};
    if (L > 1) {
      peers.push_back((int)((rank_ / L) * L + (rank_ + 1) % L));
    }
    if (nnodes_ > 1) {
      peers.push_back((int)((rank_ + L) % world_size_));
    }
    for (int peer : peers) {
      nvshmemx_putmem_nbi_on_stream(pb, pb, 16, peer, stream);
      nvshmemx_putmem_signal_nbi_on_stream(
          pb, pb, 16, ps, 0, NVSHMEM_SIGNAL_ADD, peer, stream);
      nvshmemx_signal_op_on_stream(ps, 0, NVSHMEM_SIGNAL_SET, peer, stream);
    }
    nvshmemx_quiet_on_stream(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fused_ep_dispatch_preload();
    nvshmem_barrier_all();
  }

  ~FusedEpDispatchImpl() { cudaEventDestroy(seg_event_); }

  std::vector<Tensor>
  dispatch(Tensor inputs_shard, Tensor dst_phys, Tensor probs,
           int64_t num_comm_sm) {
    (void)get_int_from_env("FLUX_FUSED_EP_DISPATCH_TAG", 0);
    CHECK_NDIM(inputs_shard, 2);
    CHECK_NDIM(dst_phys, 2);
    CHECK_NDIM(probs, 2);
    FLUX_CHECK(inputs_shard.dtype() == dtype_ &&
               inputs_shard.is_contiguous() &&
               inputs_shard.device().is_cuda());
    FLUX_CHECK(dst_phys.dtype() == at::ScalarType::Int &&
               dst_phys.is_contiguous() && dst_phys.device().is_cuda());
    FLUX_CHECK(probs.dtype() == at::ScalarType::Float &&
               probs.is_contiguous() && probs.device().is_cuda());
    const int64_t S = inputs_shard.size(0);
    FLUX_CHECK(S <= s_max_ && inputs_shard.size(1) == hidden_);
    FLUX_CHECK(dst_phys.size(0) == S && dst_phys.size(1) == topk_);
    FLUX_CHECK(probs.size(0) == S && probs.size(1) == topk_);

    run_id_ += 1;
    last_s_ = S;
    auto p = make_params(S);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    p.inputs_shard = inputs_shard.data_ptr();
    p.dst_phys = dst_phys.data_ptr<int32_t>();
    p.probs = probs.data_ptr<float>();
    const int32_t gate_end = (m_groups_ == 1) ? (int32_t)nlp_ : 0;
    fused_ep_dispatch_impl(p, weights_out_.data_ptr<float>(),
                           (int32_t)num_comm_sm, 0, gate_end, stream);

    // the phase's single host sync: [2*nlp] seg metadata
    CUDA_CHECK(cudaMemcpyAsync(
        seg_pinned_.data_ptr(), seg_meta_.data_ptr(),
        2 * nlp_ * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(seg_event_, stream));
    CUDA_CHECK(cudaEventSynchronize(seg_event_));
    const int32_t *sm = seg_pinned_.data_ptr<int32_t>();
    const int64_t n_recv =
        (int64_t)sm[2 * nlp_ - 1] + sm[nlp_ - 1];  // last start + rows
    return {recv_data_sym_.narrow(0, 0, std::max<int64_t>(n_recv, 1)),
            weights_out_.narrow(0, 0, std::max<int64_t>(n_recv, 1)),
            seg_pinned_.clone()};
  }

  void
  wait_group(int64_t g, int64_t num_comm_sm) {
    FLUX_CHECK(m_groups_ > 1 && g >= 0 && g < m_groups_);
    const int32_t gsz = (int32_t)(nlp_ / m_groups_);
    auto p = make_params(last_s_);
    fused_ep_recv_gate_only(p, (int32_t)(g * gsz), (int32_t)((g + 1) * gsz),
                            (int32_t)num_comm_sm,
                            c10::cuda::getCurrentCUDAStream());
  }

  void
  combine(Tensor expert_rows, int64_t num_comm_sm) {
    CHECK_NDIM(expert_rows, 2);
    FLUX_CHECK(expert_rows.dtype() == dtype_ &&
               expert_rows.is_contiguous() &&
               expert_rows.device().is_cuda());
    FLUX_CHECK(expert_rows.size(1) == hidden_);
    FusedEpCombineParams c{};
    c.rank = rank_;
    c.world_size = world_size_;
    c.nlp = (int32_t)nlp_;
    c.S = (int32_t)last_s_;
    c.K = (int32_t)topk_;
    c.row_bytes = row_bytes_;
    c.n_recv = (int32_t)expert_rows.size(0);
    // stage through the symmetric heap: the caller's gemm output is an
    // ordinary CUDA tensor, which a proxied transport cannot use as a put
    // source (same constraint as the pack buffers — see the ctor comment).
    // The D2D copy is part of the combine bracket, honestly timed.
    FLUX_CHECK(expert_rows.size(0) <= max_recv_total_);
    cudaStream_t stream_ = c10::cuda::getCurrentCUDAStream();
    CUDA_CHECK(cudaMemcpyAsync(
        comb_stage_sym_.data_ptr(), expert_rows.data_ptr(),
        (size_t)expert_rows.size(0) * row_bytes_, cudaMemcpyDeviceToDevice,
        stream_));
    c.expert_rows = comb_stage_sym_.data_ptr();
    c.recv_hdr_sym = recv_hdr_sym_.data_ptr<int32_t>();
    c.recv_off = recv_off_.data_ptr<int32_t>();
    c.counts_sym = counts_sym_.data_ptr<int32_t>();
    c.comb_data_sym = comb_data_sym_.data_ptr();
    c.sig_comb = reinterpret_cast<uint64_t *>(sig_comb_.data_ptr());
    c.run_id = run_id_;
    fused_ep_combine_impl(c, (int32_t)num_comm_sm,
                          c10::cuda::getCurrentCUDAStream());
  }

  Tensor
  combine_gate(int64_t s_tokens, int64_t num_comm_sm) {
    fused_ep_combine_gate_impl(
        reinterpret_cast<const uint64_t *>(sig_comb_.data_ptr()),
        world_size_, (int32_t)nlp_, run_id_, (uint64_t)spin_limit_,
        c10::cuda::getCurrentCUDAStream());
    return comb_data_sym_.narrow(0, 0, s_tokens * topk_);
  }

  void
  probe_signal_ordering(int64_t iters) {
    FLUX_CHECK(world_size_ >= 2) << "ordering probe needs >= 2 ranks";
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // ring: every rank WRITES to (rank+1) and READS from (rank-1) each
    // step, so every rank's signal advances by exactly 1 per (it, form)
    // and the GEQ expectation stays a simple monotone counter.
    const int next_peer = (rank_ + 1) % world_size_;
    const int32_t words = 4096;
    int32_t *payload = probe_payload_.data_ptr<int32_t>();
    uint64_t *sig = reinterpret_cast<uint64_t *>(probe_sig_.data_ptr());
    for (int64_t it = 1; it <= iters; ++it) {
      for (int32_t form = 0; form <= 1; ++form) {
        const uint64_t sig_epoch = (uint64_t)((it - 1) * 2 + form + 1);
        // writer first (issues nbi + signal, retires), then the reader
        // spin on the SAME stream — no dependency cycle: the writer
        // kernel has completed before the reader launches.
        fused_ep_probe_impl(next_peer, payload, words, sig, sig_epoch,
                            form, /*is_writer=*/1,
                            probe_err_.data_ptr<int32_t>(), stream);
        fused_ep_probe_impl(next_peer, payload, words, sig, sig_epoch,
                            form, /*is_writer=*/0,
                            probe_err_.data_ptr<int32_t>(), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // epoch fence: without this, a fast peer's NEXT-epoch put lands
        // in our payload mid-verification and reads as "stale" (a probe
        // artifact, not a transport violation)
        pg_->sync();
      }
    }
    auto err = probe_err_.cpu();
    FLUX_CHECK(err.item<int32_t>() == 0)
        << "put->signal ordering VIOLATED on this transport: "
        << err.item<int32_t>() << " stale words — do NOT bring up "
           "FusedEpDispatch (campaign S4 gate)";
    nvshmem_barrier_all();
  }

  Tensor recv_rows() { return recv_data_sym_; }
  Tensor headers() { return recv_hdr_sym_; }
};

FusedEpDispatch::FusedEpDispatch(
    std::shared_ptr<Group> pg, int64_t nnodes, int64_t s_max, int64_t hidden,
    int64_t topk, int64_t nlp, int64_t max_rows_per_pair,
    int64_t max_recv_total, at::ScalarType dtype, int64_t m_groups,
    int64_t spin_limit)
    : impl_(new FusedEpDispatchImpl(
          pg, nnodes, s_max, hidden, topk, nlp, max_rows_per_pair,
          max_recv_total, dtype, m_groups, spin_limit)) {}
FusedEpDispatch::~FusedEpDispatch() { delete impl_; }
std::vector<torch::Tensor>
FusedEpDispatch::dispatch(torch::Tensor a, torch::Tensor b, torch::Tensor c,
                          int64_t n) {
  FLUX_CHECK(impl_ != nullptr);
  return impl_->dispatch(a, b, c, n);
}
void
FusedEpDispatch::wait_group(int64_t g, int64_t n) {
  FLUX_CHECK(impl_ != nullptr);
  impl_->wait_group(g, n);
}
void
FusedEpDispatch::combine(torch::Tensor a, int64_t n) {
  FLUX_CHECK(impl_ != nullptr);
  impl_->combine(a, n);
}
torch::Tensor
FusedEpDispatch::combine_gate(int64_t s, int64_t n) {
  FLUX_CHECK(impl_ != nullptr);
  return impl_->combine_gate(s, n);
}
void
FusedEpDispatch::probe_signal_ordering(int64_t iters) {
  FLUX_CHECK(impl_ != nullptr);
  impl_->probe_signal_ordering(iters);
}
torch::Tensor
FusedEpDispatch::recv_rows() {
  FLUX_CHECK(impl_ != nullptr);
  return impl_->recv_rows();
}
torch::Tensor
FusedEpDispatch::headers() {
  FLUX_CHECK(impl_ != nullptr);
  return impl_->headers();
}

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
