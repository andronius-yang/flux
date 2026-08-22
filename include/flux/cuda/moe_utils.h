//===- moe_utils.h ----------------------------------------------- C++ ---===//
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

#include <cuda_runtime_api.h>

namespace bytedance::flux {
/**
 * @brief a none-deterministic way to calculate scatter_index from choosed_experts.
 *
 * @param[in] choosed_experts : of topk * ntokens
 * @param[in] count : count of per experts.
 * @param[out] scatter_index : of topk * ntokens
 * @param[in] total_num : topk * ntokens
 * @param[in] expert_num
 * @param[in] stream
 */
void calc_scatter_index(
    const int *choosed_experts,  // of total_num
    const int *count,            // of expert_num
    int *scatter_index,          // of total_num
    const int total_num,         // topk * ntokens
    int expert_num,
    cudaStream_t stream);

/**
 * PLACE-lambda sender-local LocCap router (pll_* arms): route this rank's
 * [S, K] gating entries to physical expert slots. Shared quota/share
 * tables are order-independent functions of the allgathered demand d;
 * per-entry assignment uses relaxed atomic tickets (no bit-determinism —
 * see moe_utils.cu preamble). stats[4] int64 (pre-zeroed by caller):
 * [0] forced entries, [1] tier-3 entries, [2] forced-budget overflow
 * (must stay 0 under a positive f_cap — sizing-contract breach otherwise).
 * cap64 = ceil((1+eps)*S*K) (huge value = pure locality). f_cap = the
 * per-(src, dst) forced-admission budget (<= 0 unlimited) — the one clamp
 * that makes the pair/recv sizing bounds provable.
 */
void placelambda_route_sl(
    const int *topk_own,  // [S, K]
    const int *d,         // [R, G]
    const int *l2p,       // [G, Cmax]
    const int *lcnts,     // [G]
    int *phys_own,        // [S, K] out
    long long *stats,     // [4] out, pre-zeroed
    void *workspace,      // >= placelambda_route_sl_workspace_bytes(...)
    int S, int K, int G, int R, int Cmax, int nlp, int ranks_per_node,
    int my_rank, long long cap64, int f_cap, cudaStream_t stream);

size_t placelambda_route_sl_workspace_bytes(int G, int R, int ranks_per_node);

}  // namespace bytedance::flux
