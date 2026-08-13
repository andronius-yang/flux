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
struct PushLeg {
  int32_t dst_rank;
  int32_t dst_slot;
  int32_t src_row;  // row in MY weight home (home legs) — see users
};
struct FwdLeg {
  int32_t gw_slot;   // MY slot to wait on + forward from
  int32_t dst_rank;  // same-node peer
  int32_t dst_slot;
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
      }
    }
    // one CUStreamWaitValue64 per gateway slot serves all its forwards
    std::stable_sort(
        this->mcast_fwd_.begin(),
        this->mcast_fwd_.end(),
        [](const FwdLeg &a, const FwdLeg &b) { return a.gw_slot < b.gw_slot; });
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
        nvshmemx_putmem_signal_nbi_on_stream(
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
    if (!multicast) {
      for (const auto &leg : this->direct_all_) {
        emit_home_leg(leg);
      }
      return this->run_id_;
    }
    // M4 multicast. HOME role: mcast_out_ = the gw<0 pairs — intra-home-node
    // destinations, singleton groups, and each cross-node group's single
    // inter-node leg (into the gateway's own slot).
    for (const auto &leg : this->mcast_out_) {
      emit_home_leg(leg);
    }
    // GATEWAY role: one zero-SM wait on my landed slot per gateway slot,
    // then one NVLink CE putmem_signal per needy same-node peer, sourced
    // from the slot row. Every wait's satisfying writer is a REMOTE home
    // rank (NR-02 Class B safe: nothing later on my own channels releases
    // it); waits are issued in ascending gw_slot order, an executable order
    // on one stream since each depends only on remote arrivals. Every leg
    // carries a full expert row — this op has no zero-payload destinations
    // by construction (a prefetch pair exists only for alloc > 0).
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
      nvshmemx_putmem_signal_nbi_on_stream(
          dst,
          src,
          this->expert_bytes_,
          sig_base + f.dst_slot,
          epoch,
          NVSHMEM_SIGNAL_SET,
          f.dst_rank,
          stream);
    }
    return this->run_id_;
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

int64_t
WeightPushMulticast::forward(bool multicast) {
  FLUX_CHECK(impl_ != nullptr) << "WeightPushMulticast is not initialized!";
  return impl_->forward(multicast);
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
