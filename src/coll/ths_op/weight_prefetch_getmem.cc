//===- weight_prefetch_getmem.cc ---------------------------------- C++ ---===//
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
#include "coll/ths_op/weight_prefetch_getmem.h"

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <cuda_runtime_api.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/all.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <vector>

#include "coll/weight_prefetch_getmem_impl.hpp"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"

namespace bytedance {
namespace flux {
namespace ths_op {

using torch::Tensor;

class WeightPrefetchGetmem::WeightPrefetchGetmemImpl {
 private:
  std::shared_ptr<Group> pg_;
  int64_t n_experts_local_;
  int64_t expert_bytes_;
  int32_t world_size_;
  at::ScalarType dtype_;

  torch::Tensor weight_home_;
  torch::Tensor pairs_dev_;
  std::vector<int32_t> pairs_host_;

 public:
  WeightPrefetchGetmemImpl(
      std::shared_ptr<Group> pg,
      int64_t n_experts_local,
      int64_t row_dim0,
      int64_t row_dim1,
      at::ScalarType dtype)
      : pg_(pg),
        n_experts_local_(n_experts_local),
        world_size_(pg->get_size()),
        dtype_(dtype) {
    FLUX_CHECK(n_experts_local > 0 && row_dim0 > 0 && row_dim1 > 0);
    // Collective, uniform across ranks by construction (every rank homes
    // n_experts_local experts of identical shape).
    this->weight_home_ =
        nvshmem_create_tensor({n_experts_local, row_dim0, row_dim1}, dtype, false);
    this->expert_bytes_ = row_dim0 * row_dim1 * this->weight_home_.element_size();
  }

  torch::Tensor
  weight_home() {
    return this->weight_home_;
  }

  void
  set_pairs(torch::Tensor pairs_cpu) {
    CHECK_NDIM(pairs_cpu, 2);
    FLUX_CHECK(pairs_cpu.size(1) == 3);
    FLUX_CHECK(pairs_cpu.dtype() == at::ScalarType::Int);
    FLUX_CHECK(pairs_cpu.device().is_cpu());
    FLUX_CHECK(pairs_cpu.is_contiguous());
    const int32_t *p = pairs_cpu.data_ptr<int32_t>();
    int64_t n = pairs_cpu.size(0);
    for (int64_t i = 0; i < n; ++i) {
      FLUX_CHECK(p[i * 3 + 1] >= 0 && p[i * 3 + 1] < this->world_size_)
          << "home_pe out of range at pair " << i;
      FLUX_CHECK(p[i * 3 + 2] >= 0 && p[i * 3 + 2] < this->n_experts_local_)
          << "src_row out of range at pair " << i;
    }
    this->pairs_host_.assign(p, p + n * 3);
    this->pairs_dev_ = pairs_cpu.to(this->weight_home_.device());
  }

  torch::Tensor
  forward(torch::Tensor dst, int32_t num_comm_sm, int64_t chunk_bytes, bool device_kernel) {
    FLUX_CHECK(dst.device().is_cuda());
    FLUX_CHECK(dst.is_contiguous());
    FLUX_CHECK(dst.dtype() == this->dtype_);
    FLUX_CHECK(num_comm_sm > 0);
    FLUX_CHECK(chunk_bytes > 0);
    int64_t n_pairs = static_cast<int64_t>(this->pairs_host_.size()) / 3;
    for (int64_t i = 0; i < n_pairs; ++i) {
      int64_t slot = this->pairs_host_[i * 3 + 0];
      FLUX_CHECK((slot + 1) * this->expert_bytes_ <= static_cast<int64_t>(dst.nbytes()))
          << "dst_slot " << slot << " exceeds dst capacity";
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    if (device_kernel) {
      WeightPrefetchGetmemParams params{
          .weight_home_ptr = this->weight_home_.data_ptr(),
          .dst_ptr = dst.data_ptr(),
          .pairs = n_pairs > 0 ? this->pairs_dev_.data_ptr<int32_t>() : nullptr,
          .n_pairs = static_cast<int32_t>(n_pairs),
          .chunks_per_expert =
              static_cast<int32_t>((this->expert_bytes_ + chunk_bytes - 1) / chunk_bytes),
          .expert_bytes = this->expert_bytes_,
          .chunk_bytes = chunk_bytes};
      weight_prefetch_getmem_impl(params, num_comm_sm, stream);
      return dst;
    }
    char *dst_base = static_cast<char *>(dst.data_ptr());
    const char *home_base = static_cast<const char *>(this->weight_home_.data_ptr());
    for (int64_t i = 0; i < n_pairs; ++i) {
      int32_t slot = this->pairs_host_[i * 3 + 0];
      int32_t home_pe = this->pairs_host_[i * 3 + 1];
      int32_t src_row = this->pairs_host_[i * 3 + 2];
      nvshmemx_getmem_nbi_on_stream(
          dst_base + static_cast<int64_t>(slot) * this->expert_bytes_,
          home_base + static_cast<int64_t>(src_row) * this->expert_bytes_,
          this->expert_bytes_,
          home_pe,
          stream);
    }
    nvshmemx_quiet_on_stream(stream);
    return dst;
  }
};

WeightPrefetchGetmem::WeightPrefetchGetmem(
    std::shared_ptr<Group> pg,
    int64_t n_experts_local,
    int64_t row_dim0,
    int64_t row_dim1,
    at::ScalarType dtype)
    : impl_(new WeightPrefetchGetmemImpl(pg, n_experts_local, row_dim0, row_dim1, dtype)) {}

WeightPrefetchGetmem::~WeightPrefetchGetmem() { delete impl_; }

torch::Tensor
WeightPrefetchGetmem::weight_home() {
  FLUX_CHECK(impl_ != nullptr) << "WeightPrefetchGetmem is not initialized!";
  return impl_->weight_home();
}

void
WeightPrefetchGetmem::set_pairs(torch::Tensor pairs_cpu) {
  FLUX_CHECK(impl_ != nullptr) << "WeightPrefetchGetmem is not initialized!";
  impl_->set_pairs(pairs_cpu);
}

torch::Tensor
WeightPrefetchGetmem::forward(
    torch::Tensor dst, int32_t num_comm_sm, int64_t chunk_bytes, bool device_kernel) {
  FLUX_CHECK(impl_ != nullptr) << "WeightPrefetchGetmem is not initialized!";
  return impl_->forward(dst, num_comm_sm, chunk_bytes, device_kernel);
}

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
