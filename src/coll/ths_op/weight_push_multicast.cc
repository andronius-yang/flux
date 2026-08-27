//===- weight_push_multicast.cc ----------------------------------- C++ ---===//
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
#include "coll/ths_op/weight_push_multicast.h"

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/all.h>

#include <algorithm>
#include <vector>

#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/flux.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"

namespace bytedance {
namespace flux {
namespace ths_op {

using torch::Tensor;

namespace {

// 2026-08-22 wire-ordering HARD RULE (CLAUDE.md invariant 5 / SCHEMA rule 6):
// on libfabric/CXI the nbi put_signal exposes the flag before the data, so
// every weight-push put a consumer gates on is BLOCKING by default.
// FLUX_WPM_BLOCKING_WIRE=0 restores the refuted nbi wire (ablation only).
static inline bool
flux_wpm_blocking_wire() {
  static const bool v =
      bytedance::flux::get_int_from_env("FLUX_WPM_BLOCKING_WIRE", 1) != 0;
  return v;
}
static inline void
flux_wpm_put_signal(
    void *dst,
    const void *src,
    size_t bytes,
    uint64_t *sig,
    uint64_t val,
    int sig_op,
    int pe,
    cudaStream_t stream) {
  if (flux_wpm_blocking_wire()) {
    nvshmemx_putmem_signal_on_stream(dst, src, bytes, sig, val, sig_op, pe, stream);
  } else {
    nvshmemx_putmem_signal_nbi_on_stream(dst, src, bytes, sig, val, sig_op, pe, stream);
  }
}

struct PushLeg {
  int32_t dst_rank;
  int32_t dst_slot;
  int32_t src_row;  // row in MY weight home (home legs) — see users
};
struct PullLeg {
  int32_t slot;   // MY slot to fill
  int32_t home;   // expert's home rank (get source PE)
  int32_t src;    // row in the home's weight_home
};
struct FwdLeg {
  int32_t gw_slot;   // MY slot to wait on + forward from
  int32_t dst_rank;  // same-node peer
  int32_t dst_slot;
};
struct ShardLeg {
  int32_t home_rank;
  int32_t dst_rank;
  int32_t dst_slot;
  int32_t shard_idx;
  int32_t egress_rank;
  int32_t ingress_rank;
  int64_t byte_off;
  int64_t byte_len;
  // staging positions on the egress/ingress rank, derived by every rank from
  // the replicated table scan (writer and owner agree with no exchange)
  int32_t eg_slot_idx = -1;
  int32_t in_slot_idx = -1;
  int32_t src_row = -1;  // home legs only: joined from the pair plan
};
}  // namespace

class WeightPushMulticast::WeightPushMulticastImpl {
 private:
  std::shared_ptr<Group> pg_;
  int64_t n_experts_local_;
  int64_t n_slots_;
  int64_t expert_bytes_;
  int32_t rank_;
  int32_t world_size_;
  at::ScalarType dtype_;
  int64_t run_id_ = 0;

  torch::Tensor weight_full_;
  torch::Tensor weight_home_;
  torch::Tensor prefetch_slots_;
  torch::Tensor signals_;  // symmetric int64[n_slots], epoch, never memset

  // partitions of the replicated plan (see set_plan)
  std::vector<PushLeg> direct_all_;   // home == me: every pair, direct mode
  std::vector<PushLeg> mcast_out_;    // home == me && gw < 0: multicast legs
  std::vector<FwdLeg> mcast_fwd_;     // gw == me: NVLink forwards
  std::vector<int32_t> my_in_slots_;  // dst == me: slots join() waits on
  std::vector<PullLeg> my_pull_;      // dst == me: round-4 pull legs

  // egress NIC-sharding state (see set_shard_plan; empty => disabled)
  std::vector<ShardLeg> shard_home_;     // home == me: wait-free stage/push
  std::vector<ShardLeg> shard_egress_;   // egress == me != home
  std::vector<ShardLeg> shard_ingress_;  // ingress == me != dst
  std::vector<uint64_t> sharded_out_keys_;  // (dst<<32)|slot of MY sharded legs
  std::vector<int32_t> my_shard_slots_;     // dst == me: sharded in-slots
  std::vector<uint64_t> arrive_quota_;      // [n_slots] chunk adds per forward
  std::vector<uint64_t> expected_arrive_;   // [n_slots] cumulative host mirror
  int64_t shard_chunk_bytes_ = 0;
  int64_t shard_maxc_ = 0;        // max chunks per shard (sig stride)
  int64_t max_shard_bytes_ = 0;   // staging row pitch
  torch::Tensor egress_stage_;    // symmetric [cap_e, max_shard_bytes_] bytes
  torch::Tensor ingress_stage_;   // symmetric [cap_i, max_shard_bytes_] bytes
  torch::Tensor eg_sig_;          // symmetric u64[cap_e * shard_maxc_]
  torch::Tensor in_sig_;          // symmetric u64[cap_i * shard_maxc_]
  torch::Tensor shard_arrive_;    // symmetric u64[n_slots], ADD, never reset
  torch::Tensor prime_buf_;       // symmetric scratch for kernel priming
  torch::Tensor prime_sig_;       // symmetric u64[1] priming signal (stays 0)

  void
  prime_pull(int64_t local_world_size) {
    // 2026-08-27 pull-mode priming (lazy-load class, both gates wedged at
    // i0 l0): getmem_on_stream is a NEW device-kernel class never primed
    // by the put-side paths, and under tokens-first the gets' first
    // launch happens while the gated GEMM already spins on the very
    // signals those gets deliver. Prime BOTH get classes against the
    // priming scratch while the GPU is idle: an INTRANODE peer (P2P get
    // kernel) and, when one exists, an INTERNODE peer (proxy get), plus
    // the local signal SET forward_pull uses. Lane calls this once at
    // setup (the ctor lacks local_world_size).
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int64_t L = local_world_size;
    char *buf = static_cast<char *>(this->prime_buf_.data_ptr());
    int p2p_peer = static_cast<int>((this->rank_ / L) * L + (this->rank_ + 1) % L);
    nvshmemx_getmem_on_stream(buf, buf, 16, p2p_peer, stream);
    if (this->world_size_ > L) {
      int far_peer = static_cast<int>((this->rank_ + L) % this->world_size_);
      nvshmemx_getmem_on_stream(buf, buf, 16, far_peer, stream);
    }
    uint64_t *sig = reinterpret_cast<uint64_t *>(this->prime_sig_.data_ptr());
    nvshmemx_signal_op_on_stream(sig, 0, NVSHMEM_SIGNAL_SET, this->rank_, stream);
    nvshmemx_quiet_on_stream(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void
  prime_shard_kernels(int64_t local_world_size) {
    // 2026-08-16 lazy-load lesson (ctor-priming fix 1550b67): NVSHMEM
    // on-stream signal/put ops to P2P peers are DEVICE KERNELS whose first
    // launch must never happen under a resident spinning GEMM
    // (CUDA_MODULE_LOADING=LAZY module loads deadlock behind persistent
    // kernels). Under weights_first issue order the shard chain enqueues
    // before the GEMM, hiding the first launch; under tokens_first the GEMM
    // is already resident and spinning on the very signals these kernels
    // deliver (observed 2n hang, 2026-08-17). Prime every kernel class the
    // shard chain uses — P2P putmem_signal SET, P2P putmem_signal ADD (add
    // value 0), local signal_op SET — against dedicated scratch on an
    // intranode peer, then synchronize. Values are chosen so all signals
    // stay 0; the scratch payload is meaningless by construction.
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int64_t L = local_world_size;
    int peer = static_cast<int>((this->rank_ / L) * L + (this->rank_ + 1) % L);
    char *buf = static_cast<char *>(this->prime_buf_.data_ptr());
    uint64_t *sig = reinterpret_cast<uint64_t *>(this->prime_sig_.data_ptr());
    flux_wpm_put_signal(
        buf, buf, 16, sig, 0, NVSHMEM_SIGNAL_SET, peer, stream);
    flux_wpm_put_signal(
        buf, buf, 16, sig, 0, NVSHMEM_SIGNAL_ADD, peer, stream);
    nvshmemx_signal_op_on_stream(sig, 0, NVSHMEM_SIGNAL_SET, this->rank_, stream);
    nvshmemx_quiet_on_stream(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  bool
  is_sharded_out(int32_t d, int32_t b) const {
    uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(d)) << 32) |
                   static_cast<uint32_t>(b);
    return std::binary_search(this->sharded_out_keys_.begin(), this->sharded_out_keys_.end(), key);
  }

  int64_t
  chunks_of(int64_t byte_len) const {
    return (byte_len + this->shard_chunk_bytes_ - 1) / this->shard_chunk_bytes_;
  }

 public:
  WeightPushMulticastImpl(
      std::shared_ptr<Group> pg,
      int64_t n_experts_local,
      int64_t n_slots,
      int64_t row_dim0,
      int64_t row_dim1,
      at::ScalarType dtype)
      : pg_(pg),
        n_experts_local_(n_experts_local),
        n_slots_(n_slots),
        rank_(pg->get_rank()),
        world_size_(pg->get_size()),
        dtype_(dtype) {
    (void)bytedance::flux::get_int_from_env("FLUX_WPM_BLOCKING_WIRE_DEFAULT_TAG", 0);  // 2026-08-22
    FLUX_CHECK(n_experts_local > 0 && n_slots > 0 && row_dim0 > 0 && row_dim1 > 0);
    // Collective, uniform across ranks. Contiguous [epn + B] layout: the
    // single weights operand the fused a2av GEMM takes, and both the put
    // SOURCE (home rows) and DESTINATION (slot rows) stay on the symmetric
    // heap (proxy transfers want registered memory at both ends).
    this->weight_full_ =
        nvshmem_create_tensor({n_experts_local + n_slots, row_dim0, row_dim1}, dtype, true);
    this->weight_home_ = this->weight_full_.narrow(0, 0, n_experts_local);
    this->prefetch_slots_ = this->weight_full_.narrow(0, n_experts_local, n_slots);
    this->signals_ = nvshmem_create_tensor({n_slots}, at::ScalarType::Long, true);
    // multi-writer arrival counter for sharded legs (SIGNAL_ADD): its
    // correctness depends on an exact zero start, so zero it explicitly
    // (local device memset, uniform across ranks, before any use).
    this->shard_arrive_ = nvshmem_create_tensor({n_slots}, at::ScalarType::Long, true);
    this->shard_arrive_.zero_();
    // kernel-priming scratch (see prime_shard_kernels)
    this->prime_buf_ = nvshmem_create_tensor({16}, at::ScalarType::Byte, true);
    this->prime_sig_ = nvshmem_create_tensor({1}, at::ScalarType::Long, true);
    this->prime_sig_.zero_();
    this->expected_arrive_.assign(n_slots, 0);
    this->arrive_quota_.assign(n_slots, 0);
    this->expert_bytes_ = row_dim0 * row_dim1 * this->weight_full_.element_size();
  }

  torch::Tensor
  weight_full() {
    return this->weight_full_;
  }
  torch::Tensor
  weight_home() {
    return this->weight_home_;
  }
  torch::Tensor
  prefetch_slots() {
    return this->prefetch_slots_;
  }
  torch::Tensor
  signals() {
    return this->signals_;
  }

  void
  set_plan(torch::Tensor pairs_cpu) {
    CHECK_NDIM(pairs_cpu, 2);
    FLUX_CHECK(pairs_cpu.size(1) == 6);
    FLUX_CHECK(pairs_cpu.dtype() == at::ScalarType::Int);
    FLUX_CHECK(pairs_cpu.device().is_cpu());
    FLUX_CHECK(pairs_cpu.is_contiguous());
    this->direct_all_.clear();
    this->mcast_out_.clear();
    this->mcast_fwd_.clear();
    this->my_in_slots_.clear();
    this->my_pull_.clear();
    const int32_t *p = pairs_cpu.data_ptr<int32_t>();
    int64_t n = pairs_cpu.size(0);
    for (int64_t i = 0; i < n; ++i) {
      int32_t d = p[i * 6 + 0], b = p[i * 6 + 1], home = p[i * 6 + 2];
      int32_t src = p[i * 6 + 3], gw = p[i * 6 + 4], gws = p[i * 6 + 5];
      FLUX_CHECK(d >= 0 && d < this->world_size_) << "dst_rank out of range at pair " << i;
      FLUX_CHECK(b >= 0 && b < this->n_slots_) << "dst_slot out of range at pair " << i;
      FLUX_CHECK(home >= 0 && home < this->world_size_) << "home_rank out of range at " << i;
      FLUX_CHECK(src >= 0 && src < this->n_experts_local_) << "src_row out of range at " << i;
      FLUX_CHECK(gw >= -1 && gw < this->world_size_) << "gw_rank out of range at pair " << i;
      FLUX_CHECK(gw < 0 || (gws >= 0 && gws < this->n_slots_))
          << "gw_slot out of range at pair " << i;
      FLUX_CHECK(gw != d) << "a gateway never forwards to itself (pair " << i << ")";
      if (home == this->rank_) {
        this->direct_all_.push_back({d, b, src});
        if (gw < 0) {
          this->mcast_out_.push_back({d, b, src});
        }
      }
      if (gw == this->rank_) {
        this->mcast_fwd_.push_back({gws, d, b});
      }
      if (d == this->rank_) {
        this->my_in_slots_.push_back(b);
        this->my_pull_.push_back({b, home, src});
      }
    }
    // one CUStreamWaitValue64 per gateway slot serves all its forwards
    std::stable_sort(
        this->mcast_fwd_.begin(),
        this->mcast_fwd_.end(),
        [](const FwdLeg &a, const FwdLeg &b) { return a.gw_slot < b.gw_slot; });
  }

  void
  set_shard_plan(torch::Tensor shards_cpu, int64_t chunk_bytes, int64_t local_world_size) {
    CHECK_NDIM(shards_cpu, 2);
    FLUX_CHECK(shards_cpu.size(1) == 8);
    FLUX_CHECK(shards_cpu.dtype() == at::ScalarType::Int);
    FLUX_CHECK(shards_cpu.device().is_cpu());
    FLUX_CHECK(shards_cpu.is_contiguous());
    const int64_t L = local_world_size;
    FLUX_CHECK(L > 0 && this->world_size_ % L == 0);
    this->shard_home_.clear();
    this->shard_egress_.clear();
    this->shard_ingress_.clear();
    this->sharded_out_keys_.clear();
    this->my_shard_slots_.clear();
    this->arrive_quota_.assign(this->n_slots_, 0);
    const int32_t *p = shards_cpu.data_ptr<int32_t>();
    const int64_t n = shards_cpu.size(0);
    this->max_shard_bytes_ = 0;
    for (int64_t i = 0; i < n; ++i) {
      this->max_shard_bytes_ = std::max(this->max_shard_bytes_, (int64_t)p[i * 8 + 7]);
    }
    this->shard_chunk_bytes_ =
        (chunk_bytes > 0 && chunk_bytes < this->max_shard_bytes_) ? chunk_bytes
                                                                  : this->max_shard_bytes_;
    this->shard_maxc_ = this->max_shard_bytes_ > 0 ? this->chunks_of(this->max_shard_bytes_) : 0;
    // one replicated scan: every rank derives the same staging slot indices
    // for every row, so writer and owner agree with zero metadata exchange
    std::vector<int32_t> eg_count(this->world_size_, 0), in_count(this->world_size_, 0);
    for (int64_t i = 0; i < n; ++i) {
      ShardLeg s;
      s.home_rank = p[i * 8 + 0];
      s.dst_rank = p[i * 8 + 1];
      s.dst_slot = p[i * 8 + 2];
      s.shard_idx = p[i * 8 + 3];
      s.egress_rank = p[i * 8 + 4];
      s.ingress_rank = p[i * 8 + 5];
      s.byte_off = p[i * 8 + 6];
      s.byte_len = p[i * 8 + 7];
      FLUX_CHECK(s.home_rank >= 0 && s.home_rank < this->world_size_) << "row " << i;
      FLUX_CHECK(s.dst_rank >= 0 && s.dst_rank < this->world_size_) << "row " << i;
      FLUX_CHECK(s.dst_slot >= 0 && s.dst_slot < this->n_slots_) << "row " << i;
      FLUX_CHECK(s.home_rank / L != s.dst_rank / L) << "only cross-node legs shard, row " << i;
      FLUX_CHECK(s.shard_idx >= 0 && s.shard_idx < L) << "row " << i;
      FLUX_CHECK(s.egress_rank == (s.home_rank / L) * L + s.shard_idx) << "row " << i;
      FLUX_CHECK(s.ingress_rank == (s.dst_rank / L) * L + s.shard_idx) << "row " << i;
      FLUX_CHECK(s.byte_off >= 0 && s.byte_len > 0) << "row " << i;
      FLUX_CHECK(s.byte_off + s.byte_len <= this->expert_bytes_) << "row " << i;
      if (s.egress_rank != s.home_rank) {
        s.eg_slot_idx = eg_count[s.egress_rank]++;
      }
      if (s.ingress_rank != s.dst_rank) {
        s.in_slot_idx = in_count[s.ingress_rank]++;
      }
      if (s.home_rank == this->rank_) {
        // join src_row from the pair plan (set_plan must have run): the
        // shard table carries only the leg identity (dst_rank, dst_slot)
        s.src_row = -1;
        for (const auto &leg : this->direct_all_) {
          if (leg.dst_rank == s.dst_rank && leg.dst_slot == s.dst_slot) {
            s.src_row = leg.src_row;
            break;
          }
        }
        FLUX_CHECK(s.src_row >= 0)
            << "shard row " << i << " has no matching pair in set_plan (call set_plan first)";
        this->shard_home_.push_back(s);
        this->sharded_out_keys_.push_back(
            (static_cast<uint64_t>(static_cast<uint32_t>(s.dst_rank)) << 32) |
            static_cast<uint32_t>(s.dst_slot));
      }
      if (s.egress_rank == this->rank_ && s.egress_rank != s.home_rank) {
        this->shard_egress_.push_back(s);
      }
      if (s.ingress_rank == this->rank_ && s.ingress_rank != s.dst_rank) {
        this->shard_ingress_.push_back(s);
      }
      if (s.dst_rank == this->rank_) {
        if (this->arrive_quota_[s.dst_slot] == 0) {
          this->my_shard_slots_.push_back(s.dst_slot);
        }
        this->arrive_quota_[s.dst_slot] += static_cast<uint64_t>(this->chunks_of(s.byte_len));
      }
    }
    std::sort(this->sharded_out_keys_.begin(), this->sharded_out_keys_.end());
    // staging capacity = replicated max over ranks => identical symmetric
    // allocs on every rank (collective safety with no exchange). Fresh sig
    // arrays start at 0; SET-epoch/GEQ stays correct since epochs only grow.
    int64_t cap_e = n ? *std::max_element(eg_count.begin(), eg_count.end()) : 0;
    int64_t cap_i = n ? *std::max_element(in_count.begin(), in_count.end()) : 0;
    if (cap_e > 0) {
      this->egress_stage_ =
          nvshmem_create_tensor({cap_e, this->max_shard_bytes_}, at::ScalarType::Byte, true);
      this->eg_sig_ =
          nvshmem_create_tensor({cap_e * this->shard_maxc_}, at::ScalarType::Long, true);
      this->eg_sig_.zero_();
    }
    if (cap_i > 0) {
      this->ingress_stage_ =
          nvshmem_create_tensor({cap_i, this->max_shard_bytes_}, at::ScalarType::Byte, true);
      this->in_sig_ =
          nvshmem_create_tensor({cap_i * this->shard_maxc_}, at::ScalarType::Long, true);
      this->in_sig_.zero_();
    }
    if (n > 0) {
      this->prime_shard_kernels(L);
    }
  }

  int64_t
  forward(bool multicast) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    this->run_id_ += 1;
    const uint64_t epoch = static_cast<uint64_t>(this->run_id_);
    char *home_base = static_cast<char *>(this->weight_home_.data_ptr());
    char *slots_base = static_cast<char *>(this->prefetch_slots_.data_ptr());
    uint64_t *sig_base = reinterpret_cast<uint64_t *>(this->signals_.data_ptr());
    auto emit_home_leg = [&](const PushLeg &leg) {
      char *dst = slots_base + static_cast<int64_t>(leg.dst_slot) * this->expert_bytes_;
      char *src = home_base + static_cast<int64_t>(leg.src_row) * this->expert_bytes_;
      if (leg.dst_rank == this->rank_) {
        // self leg: local CE copy + bare signal (uniform consumer wait)
        CUDA_CHECK(cudaMemcpyAsync(
            dst, src, this->expert_bytes_, cudaMemcpyDeviceToDevice, stream));
        nvshmemx_signal_op_on_stream(
            sig_base + leg.dst_slot, epoch, NVSHMEM_SIGNAL_SET, this->rank_, stream);
      } else {
        // nbi CE put from the immutable symmetric home; signal-after-payload
        // is the putmem_signal contract. The nbi tail is quieted by the
        // token a2av's end-of-iteration barrier_all (every iteration runs
        // one), and slot rows are rewritten only after that barrier.
        flux_wpm_put_signal(
            dst,
            src,
            this->expert_bytes_,
            sig_base + leg.dst_slot,
            epoch,
            NVSHMEM_SIGNAL_SET,
            leg.dst_rank,
            stream);
      }
    };
    // Sharded legs leave the single-NIC path entirely: their home emission
    // here is the wait-free NVLink staging (or the shard-idx==home_lr NIC
    // fast path); the NIC/reassembly hops live in forward_egress/ingress.
    // The dest-side expectation accrues HERE so forward_shard_join() of the
    // same iteration waits the exact cumulative count (host mirror of the
    // never-reset device SIGNAL_ADD counter).
    const bool sharding = !this->shard_home_.empty() || !this->shard_egress_.empty() ||
                          !this->shard_ingress_.empty() || !this->my_shard_slots_.empty();
    if (sharding) {
      for (int32_t b : this->my_shard_slots_) {
        this->expected_arrive_[b] += this->arrive_quota_[b];
      }
      char *eg_stage_base = this->egress_stage_.defined()
                                ? static_cast<char *>(this->egress_stage_.data_ptr())
                                : nullptr;
      char *in_stage_base = this->ingress_stage_.defined()
                                ? static_cast<char *>(this->ingress_stage_.data_ptr())
                                : nullptr;
      uint64_t *eg_sig_base = this->eg_sig_.defined()
                                  ? reinterpret_cast<uint64_t *>(this->eg_sig_.data_ptr())
                                  : nullptr;
      uint64_t *arrive_base = reinterpret_cast<uint64_t *>(this->shard_arrive_.data_ptr());
      for (const auto &s : this->shard_home_) {
        char *src_row_base = home_base + static_cast<int64_t>(s.src_row) * this->expert_bytes_;
        const int64_t nch = this->chunks_of(s.byte_len);
        for (int64_t c = 0; c < nch; ++c) {
          const int64_t coff = c * this->shard_chunk_bytes_;
          const int64_t b = std::min(this->shard_chunk_bytes_, s.byte_len - coff);
          char *src = src_row_base + s.byte_off + coff;
          if (s.egress_rank == this->rank_) {
            // fast path shard_idx == home_lr: my own NIC carries this shard
            if (s.ingress_rank == s.dst_rank) {
              // ...and it lands directly in the final slot (dst_lr match)
              char *dst = slots_base + static_cast<int64_t>(s.dst_slot) * this->expert_bytes_ +
                          s.byte_off + coff;
              flux_wpm_put_signal(
                  dst, src, b, arrive_base + s.dst_slot, 1, NVSHMEM_SIGNAL_ADD, s.dst_rank,
                  stream);
            } else {
              char *dst = in_stage_base +
                          static_cast<int64_t>(s.in_slot_idx) * this->max_shard_bytes_ + coff;
              flux_wpm_put_signal(
                  dst, src, b,
                  reinterpret_cast<uint64_t *>(this->in_sig_.data_ptr()) +
                      s.in_slot_idx * this->shard_maxc_ + c,
                  epoch, NVSHMEM_SIGNAL_SET, s.ingress_rank, stream);
            }
          } else {
            // NVLink CE stage to the node-mate egress rank, per-chunk signal
            // so its NIC push of chunk c overlaps my stage of chunk c+1
            char *dst = eg_stage_base +
                        static_cast<int64_t>(s.eg_slot_idx) * this->max_shard_bytes_ + coff;
            flux_wpm_put_signal(
                dst, src, b, eg_sig_base + s.eg_slot_idx * this->shard_maxc_ + c, epoch,
                NVSHMEM_SIGNAL_SET, s.egress_rank, stream);
          }
        }
      }
    }
    if (!multicast) {
      for (const auto &leg : this->direct_all_) {
        if (sharding && this->is_sharded_out(leg.dst_rank, leg.dst_slot)) {
          continue;
        }
        emit_home_leg(leg);
      }
      return this->run_id_;
    }
    // M4 multicast HOME role only: mcast_out_ = the gw<0 pairs —
    // intra-home-node destinations, singleton groups, and each cross-node
    // group's single inter-node leg (into the gateway's own slot). The
    // gateway fan-out moved to forward_gateway() (NR-13 F-B): no waits here.
    for (const auto &leg : this->mcast_out_) {
      if (sharding && this->is_sharded_out(leg.dst_rank, leg.dst_slot)) {
        continue;
      }
      emit_home_leg(leg);
    }
    return this->run_id_;
  }

  int64_t
  forward_pull() {
    // Round-4 PULL movement (2026-08-27): destination-side getmem per MY
    // moved slot + a LOCAL epoch signal, both on the CURRENT stream. The
    // signal is stream-ordered behind its own payload, so the CXI
    // signal-before-data hazard (SCHEMA rule 6) cannot occur here: no
    // remote signals, no gateway/shard chain, no blocking-wire dependency.
    // Caller contract (tokens-first): enqueue AFTER the fused l0 forward's
    // dispatch legs; w1's op before w2's. Homes' weight_home_ is immutable
    // within an iteration (gate-mode wprobe synchronizes per iteration),
    // and weight_full_ is symmetric, so the local home pointer addresses
    // the remote PE's copy under NVSHMEM symmetric addressing.
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    this->run_id_ += 1;
    const uint64_t epoch = static_cast<uint64_t>(this->run_id_);
    char *home_base = static_cast<char *>(this->weight_home_.data_ptr());
    char *slots_base = static_cast<char *>(this->prefetch_slots_.data_ptr());
    uint64_t *sig_base = reinterpret_cast<uint64_t *>(this->signals_.data_ptr());
    for (const auto &leg : this->my_pull_) {
      char *dst = slots_base + static_cast<int64_t>(leg.slot) * this->expert_bytes_;
      char *src = home_base + static_cast<int64_t>(leg.src) * this->expert_bytes_;
      if (leg.home == this->rank_) {
        CUDA_CHECK(cudaMemcpyAsync(
            dst, src, this->expert_bytes_, cudaMemcpyDeviceToDevice, stream));
      } else {
        nvshmemx_getmem_on_stream(dst, src, this->expert_bytes_, leg.home, stream);
      }
      // per-slot landing: SET right after THIS slot's payload in FIFO order
      nvshmemx_signal_op_on_stream(
          sig_base + leg.slot, epoch, NVSHMEM_SIGNAL_SET, this->rank_, stream);
    }
    return this->run_id_;
  }

  void
  forward_gateway() {
    // GATEWAY role: one zero-SM wait on my landed slot per gateway slot,
    // then one NVLink CE putmem_signal per needy same-node peer, sourced
    // from the slot row. Every wait's satisfying writer is a REMOTE home
    // rank (NR-02 Class B safe: nothing later on my own channels releases
    // it); waits are issued in ascending gw_slot order, an executable order
    // on one stream since each depends only on remote arrivals. Every leg
    // carries a full expert row — this op has no zero-payload destinations
    // by construction (a prefetch pair exists only for alloc > 0). Uses the
    // CURRENT epoch (call after forward() in the same iteration).
    if (this->mcast_fwd_.empty()) {
      return;
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const uint64_t epoch = static_cast<uint64_t>(this->run_id_);
    char *slots_base = static_cast<char *>(this->prefetch_slots_.data_ptr());
    uint64_t *sig_base = reinterpret_cast<uint64_t *>(this->signals_.data_ptr());
    int32_t cur_slot = -1;
    for (const auto &f : this->mcast_fwd_) {
      if (f.gw_slot != cur_slot) {
        CU_CHECK(CUStreamWaitValue64(
            stream,
            reinterpret_cast<CUdeviceptr>(sig_base + f.gw_slot),
            epoch,
            CU_STREAM_WAIT_VALUE_GEQ));
        cur_slot = f.gw_slot;
      }
      char *dst = slots_base + static_cast<int64_t>(f.dst_slot) * this->expert_bytes_;
      char *src = slots_base + static_cast<int64_t>(f.gw_slot) * this->expert_bytes_;
      flux_wpm_put_signal(
          dst,
          src,
          this->expert_bytes_,
          sig_base + f.dst_slot,
          epoch,
          NVSHMEM_SIGNAL_SET,
          f.dst_rank,
          stream);
    }
  }

  void
  forward_egress() {
    // EGRESS role: per staged chunk one zero-SM wait (writer = the home
    // rank's wait-free NVLink stage — remote, NR-02 Class-B safe), then the
    // NIC push over my same-local-rank wire. Chunks of one shard are issued
    // in order; different legs' shards share the stream (same NIC anyway).
    if (this->shard_egress_.empty()) {
      return;
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const uint64_t epoch = static_cast<uint64_t>(this->run_id_);
    char *eg_stage_base = static_cast<char *>(this->egress_stage_.data_ptr());
    uint64_t *eg_sig_base = reinterpret_cast<uint64_t *>(this->eg_sig_.data_ptr());
    uint64_t *arrive_base = reinterpret_cast<uint64_t *>(this->shard_arrive_.data_ptr());
    char *slots_base = static_cast<char *>(this->prefetch_slots_.data_ptr());
    char *in_stage_base = this->ingress_stage_.defined()
                              ? static_cast<char *>(this->ingress_stage_.data_ptr())
                              : nullptr;
    for (const auto &s : this->shard_egress_) {
      char *my_stage = eg_stage_base + static_cast<int64_t>(s.eg_slot_idx) * this->max_shard_bytes_;
      const int64_t nch = this->chunks_of(s.byte_len);
      for (int64_t c = 0; c < nch; ++c) {
        const int64_t coff = c * this->shard_chunk_bytes_;
        const int64_t b = std::min(this->shard_chunk_bytes_, s.byte_len - coff);
        CU_CHECK(CUStreamWaitValue64(
            stream,
            reinterpret_cast<CUdeviceptr>(eg_sig_base + s.eg_slot_idx * this->shard_maxc_ + c),
            epoch,
            CU_STREAM_WAIT_VALUE_GEQ));
        if (s.ingress_rank == s.dst_rank) {
          // fast path shard_idx == dst_lr: land directly in the final slot
          char *dst = slots_base + static_cast<int64_t>(s.dst_slot) * this->expert_bytes_ +
                      s.byte_off + coff;
          flux_wpm_put_signal(
              dst, my_stage + coff, b, arrive_base + s.dst_slot, 1, NVSHMEM_SIGNAL_ADD,
              s.dst_rank, stream);
        } else {
          char *dst = in_stage_base +
                      static_cast<int64_t>(s.in_slot_idx) * this->max_shard_bytes_ + coff;
          flux_wpm_put_signal(
              dst, my_stage + coff, b,
              reinterpret_cast<uint64_t *>(this->in_sig_.data_ptr()) +
                  s.in_slot_idx * this->shard_maxc_ + c,
              epoch, NVSHMEM_SIGNAL_SET, s.ingress_rank, stream);
        }
      }
    }
  }

  void
  forward_ingress() {
    // INGRESS role: per landed chunk one zero-SM wait (writer = the previous
    // node's egress rank — remote), then the NVLink CE reassembly copy into
    // the dest rank's slot at the shard's byte offset, +1 on its arrive
    // counter. Rows with ingress == dst never reach here (the writer lands
    // them in the final slot directly).
    if (this->shard_ingress_.empty()) {
      return;
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const uint64_t epoch = static_cast<uint64_t>(this->run_id_);
    char *in_stage_base = static_cast<char *>(this->ingress_stage_.data_ptr());
    uint64_t *in_sig_base = reinterpret_cast<uint64_t *>(this->in_sig_.data_ptr());
    uint64_t *arrive_base = reinterpret_cast<uint64_t *>(this->shard_arrive_.data_ptr());
    char *slots_base = static_cast<char *>(this->prefetch_slots_.data_ptr());
    for (const auto &s : this->shard_ingress_) {
      char *my_stage = in_stage_base + static_cast<int64_t>(s.in_slot_idx) * this->max_shard_bytes_;
      const int64_t nch = this->chunks_of(s.byte_len);
      for (int64_t c = 0; c < nch; ++c) {
        const int64_t coff = c * this->shard_chunk_bytes_;
        const int64_t b = std::min(this->shard_chunk_bytes_, s.byte_len - coff);
        CU_CHECK(CUStreamWaitValue64(
            stream,
            reinterpret_cast<CUdeviceptr>(in_sig_base + s.in_slot_idx * this->shard_maxc_ + c),
            epoch,
            CU_STREAM_WAIT_VALUE_GEQ));
        char *dst = slots_base + static_cast<int64_t>(s.dst_slot) * this->expert_bytes_ +
                    s.byte_off + coff;
        flux_wpm_put_signal(
            dst, my_stage + coff, b, arrive_base + s.dst_slot, 1, NVSHMEM_SIGNAL_ADD, s.dst_rank,
            stream);
      }
    }
  }

  void
  forward_shard_join() {
    // DEST role finalize: wait the cumulative arrive count (multi-writer
    // SIGNAL_ADD — the L reassembly writers are different ranks, so a
    // last-writer SET is impossible without cross-rank ordering), then
    // publish the ordinary epoch SET on signals_ so join()/tile gates see
    // sharded and unsharded slots identically.
    if (this->my_shard_slots_.empty()) {
      return;
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    uint64_t *arrive_base = reinterpret_cast<uint64_t *>(this->shard_arrive_.data_ptr());
    uint64_t *sig_base = reinterpret_cast<uint64_t *>(this->signals_.data_ptr());
    const uint64_t epoch = static_cast<uint64_t>(this->run_id_);
    for (int32_t b : this->my_shard_slots_) {
      CU_CHECK(CUStreamWaitValue64(
          stream,
          reinterpret_cast<CUdeviceptr>(arrive_base + b),
          this->expected_arrive_[b],
          CU_STREAM_WAIT_VALUE_GEQ));
      nvshmemx_signal_op_on_stream(sig_base + b, epoch, NVSHMEM_SIGNAL_SET, this->rank_, stream);
    }
  }

  void
  join() {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    uint64_t *sig_base = reinterpret_cast<uint64_t *>(this->signals_.data_ptr());
    for (int32_t b : this->my_in_slots_) {
      CU_CHECK(CUStreamWaitValue64(
          stream,
          reinterpret_cast<CUdeviceptr>(sig_base + b),
          static_cast<uint64_t>(this->run_id_),
          CU_STREAM_WAIT_VALUE_GEQ));
    }
  }

  int64_t
  epoch() const {
    return this->run_id_;
  }
};

WeightPushMulticast::WeightPushMulticast(
    std::shared_ptr<Group> pg,
    int64_t n_experts_local,
    int64_t n_slots,
    int64_t row_dim0,
    int64_t row_dim1,
    at::ScalarType dtype)
    : impl_(new WeightPushMulticastImpl(
          pg, n_experts_local, n_slots, row_dim0, row_dim1, dtype)) {}

WeightPushMulticast::~WeightPushMulticast() { delete impl_; }

torch::Tensor
WeightPushMulticast::weight_full() {
  FLUX_CHECK(impl_ != nullptr) << "WeightPushMulticast is not initialized!";
  return impl_->weight_full();
}

torch::Tensor
WeightPushMulticast::weight_home() {
  FLUX_CHECK(impl_ != nullptr) << "WeightPushMulticast is not initialized!";
  return impl_->weight_home();
}

torch::Tensor
WeightPushMulticast::prefetch_slots() {
  FLUX_CHECK(impl_ != nullptr) << "WeightPushMulticast is not initialized!";
  return impl_->prefetch_slots();
}

torch::Tensor
WeightPushMulticast::signals() {
  FLUX_CHECK(impl_ != nullptr) << "WeightPushMulticast is not initialized!";
  return impl_->signals();
}

void
WeightPushMulticast::set_plan(torch::Tensor pairs_cpu) {
  FLUX_CHECK(impl_ != nullptr) << "WeightPushMulticast is not initialized!";
  impl_->set_plan(pairs_cpu);
}

void
WeightPushMulticast::set_shard_plan(
    torch::Tensor shards_cpu, int64_t chunk_bytes, int64_t local_world_size) {
  FLUX_CHECK(impl_ != nullptr) << "WeightPushMulticast is not initialized!";
  impl_->set_shard_plan(shards_cpu, chunk_bytes, local_world_size);
}

void
WeightPushMulticast::forward_egress() {
  FLUX_CHECK(impl_ != nullptr) << "WeightPushMulticast is not initialized!";
  impl_->forward_egress();
}

void
WeightPushMulticast::forward_ingress() {
  FLUX_CHECK(impl_ != nullptr) << "WeightPushMulticast is not initialized!";
  impl_->forward_ingress();
}

void
WeightPushMulticast::forward_shard_join() {
  FLUX_CHECK(impl_ != nullptr) << "WeightPushMulticast is not initialized!";
  impl_->forward_shard_join();
}

int64_t
WeightPushMulticast::forward(bool multicast) {
  FLUX_CHECK(impl_ != nullptr) << "WeightPushMulticast is not initialized!";
  return impl_->forward(multicast);
}

void
WeightPushMulticast::forward_gateway() {
  FLUX_CHECK(impl_ != nullptr) << "WeightPushMulticast is not initialized!";
  impl_->forward_gateway();
}
void
WeightPushMulticast::prime_pull(int64_t local_world_size) {
  FLUX_CHECK(impl_ != nullptr) << "WeightPushMulticast not initialized";
  impl_->prime_pull(local_world_size);
}
int64_t
WeightPushMulticast::forward_pull() {
  FLUX_CHECK(impl_ != nullptr) << "WeightPushMulticast not initialized";
  return impl_->forward_pull();
}

void
WeightPushMulticast::join() {
  FLUX_CHECK(impl_ != nullptr) << "WeightPushMulticast is not initialized!";
  impl_->join();
}

int64_t
WeightPushMulticast::epoch() const {
  return impl_->epoch();
}

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
