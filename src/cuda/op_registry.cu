//===- op_registry.cu --------------------------------------------- C++ ---===//
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

#include "flux/gemm_hparams.h"
#include "flux/op_registry_proto_utils.h"
#include "flux/flux.h"
#include "flux/utils.h"
#include "flux/op_registry.h"
#include <mutex>

namespace bytedance {
namespace flux {

namespace {
std::once_flag init_flag;
ArchEnum arch;
SMCoreEnum sm_core;

void
init_device_properties() {
  int major, minor;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
  int arch_num = major * 10 + minor;
  // H100/ALPS port (2026-09-06, docs/handoff/35_h100_alps_weak_scaling.md):
  // FLUX_ARCH_OVERRIDE=80 makes an sm90 device select the Sm80-tagged kernel
  // space (the V2 ops that carry the a2av/OURS work are generated for
  // Sm80/Sm89 only; their CUTLASS-2.x kernels compile natively for sm_90a,
  // see CMake GEN_CUDAARCHS). Unset = stock behaviour.
  if (const char *ov = getenv("FLUX_ARCH_OVERRIDE"); ov != nullptr && ov[0] != '\0') {
    int forced = atoi(ov);
    FLUX_CHECK(forced == 80 || forced == 89 || forced == 90)
        << "FLUX_ARCH_OVERRIDE must be 80/89/90, got " << ov;
    FLUX_CHECK(forced <= arch_num) << "FLUX_ARCH_OVERRIDE=" << forced
                                   << " cannot exceed the device arch " << arch_num;
    arch_num = forced;
  }
  FLUX_CHECK(arch_num == 80 || arch_num == 89 || arch_num == 90)
      << "unsupported arch: " << arch_num;
  arch = ArchEnum{arch_num};

  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
  // FLUX_SM_CORE_OVERRIDE=<92|108|78|132>: registry-key only (the GEMM grid
  // still sizes from the real device SM count via get_sm_count()).
  if (const char *ov = getenv("FLUX_SM_CORE_OVERRIDE"); ov != nullptr && ov[0] != '\0') {
    sm_count = atoi(ov);
  }

  switch (sm_count) {
    case 92: sm_core = SMCoreEnum::L20; break;
    case 108: sm_core = SMCoreEnum::A100; break;
    case 78: sm_core = SMCoreEnum::H20; break;
    case 132: sm_core = SMCoreEnum::H800; break;
    default: FLUX_CHECK(false) << "Unsupported SM count for SMCoreEnum: " << sm_count; break;
  }
}
}  // namespace

ArchEnum
get_arch() {
  std::call_once(init_flag, init_device_properties);
  return arch;
}

SMCoreEnum
get_sm_core() {
  std::call_once(init_flag, init_device_properties);
  return sm_core;
}

TuningConfigRegistry &
TuningConfigRegistry::instance() {
  static TuningConfigRegistry inst;
  const char *env = getenv("FLUX_TUNE_CONFIG_FILE");
  if (env != nullptr) {
    static std::once_flag flag;
    std::call_once(flag, load_tune_config_from_file, inst, env);
  } else {
#if defined(FLUX_DEBUG)
    if (get_int_from_env("RANK", 0) == 0) {
      std::cerr << "FLUX_TUNE_CONFIG_FILE not set. no tune config file specified, using default "
                   "configs\n";
    }
#endif
  }
  return inst;
}

OpRegistry &
OpRegistry::instance() {
  static OpRegistry inst;
  return inst;
}

bool
OpRegistry::check_heuristic_rule(
    const UnifiedGemmMeta &meta, const UnifiedGemmHParams &hparams, const RuntimeConfig &rt_conf) {
  if (meta.impl() == _GemmV3{}) {
    if (rt_conf.m() < 2048) {
      auto const &v3_hparams = std::get<unified_type_t<GemmV3HParams>>(hparams.impl_spec());
      return cute::get<0>(v3_hparams.cluster_shape()) == 1;
    }
  }
  return true;
}

}  // namespace flux
}  // namespace bytedance
