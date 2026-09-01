//===- sort_util.h --------------------------------------------- C++ ------===//
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
#include "flux/utils.h"

namespace bytedance::flux {

// given scatter_index, compute gather_index.
// scatter_index maps [ntokens] to [ntokens*topk], gather_index is the reverse operation.
// consider ep, we only need the gather_index for partial experts,
// we just store the gather_index for rows whose index is in [rows_offset, rows_offset+total_rows)
void calc_gather_index_impl(
    int32_t nexperts,
    int32_t ntokens,
    int32_t topk,
    int32_t expert_idx_start,
    int32_t expert_idx_end,
    int32_t const *splits,
    int32_t const *scatter_index,
    int32_t *gather_index_ep,
    int32_t *total_nrows_ep_gpu,  // scalar
    cudaStream_t stream);

void calc_gather_index_impl_v2(
    int32_t nexperts,
    int32_t ntokens,
    int32_t topk,
    int32_t rows_start,  // let's hope this won't overflow: not as many as 2**31 tokens
    int32_t rows_end,
    int32_t const *scatter_index,
    int32_t *gather_index_ep,
    cudaStream_t stream);

// The original computing flow:
//   input (shard) -> (ag) -> input (full) -> (scatter) -> mat A -> (gemm) -> mat D
// We sort matrix A so that the dependant data from
// input (shard) is as contiguous as possible.
// The new flow is:
//   input (shard) -> (ag) -> input (full) -> (scatter&sort) -> sorted mat A
//    -> (gemm) -> sorted mat D -> (scatter) -> mat D
// The original gemm is #nexperts problems, sort the tokens by a paired key:
// (the rank it is from, expert id), constructing #nexperts * #tp_size new problems.
// This is used to make overlapingg all-gather possible. By overlapping computing tokens
// from a rank whose data is ready, with fetching data from the next rank.
//
// splits: [nexperts]
// gather_index: [ntokens*topk]
// scatter_index: [ntokens*topk]
// sorted_splits: [nexperts*tp_size]
// sorted_gather_index: [ntokens*topk]
//   row index of `sorted mat A` -> row index of `input (full)`
// sorted_scatter_index: [nexperts*tp_size]
//   row index of `sorted mat D` -> row index of `mat D`
struct AGScatterSortOpArguments {
  DistEnv dist_env;
  int32_t ntokens;
  int32_t nexperts_ep;
  int32_t const *splits_ep;
  int32_t const *gather_index_ep;
  int32_t *sorted_splits;
  int32_t *sorted_scatter_index;
  int32_t *sorted_gather_index;
};

void ag_scatter_sort_impl(AGScatterSortOpArguments const &args, cudaStream_t stream);

struct AGScatterSortOpArgumentsV2 {
  int rank;  // not used.
  int world_size;
  int32_t ntokens;
  int32_t nexperts_ep;
  int32_t const *splits_ep;
  int32_t const *gather_index_ep;
  int32_t *sorted_splits;
  int32_t *sorted_splits_cumsum;
  int32_t *sorted_scatter_index;
  int32_t *sorted_gather_index;
};
void ag_scatter_sort_impl_v2(AGScatterSortOpArgumentsV2 const &args, cudaStream_t stream);

// a2av dispatch stage 1, fused: one grid-stride pass over all global copies
// decodes expert/source/owner, produces every tensor stage 2 needs, the [W,W]
// chunk-count matrix (must be pre-zeroed), and the producer pack keys
// (e * copies_per_rank + local_p — pack tie-break is the global copy index).
struct A2AVStage1Arguments {
  int32_t const *scatter_index;  // [n_copies] global dst rows
  int32_t const *splits;         // [nexperts]
  int nexperts;
  int ep_nexperts;
  int world_size;
  int rank;
  int64_t copies_per_rank;
  int64_t n_copies;
  int64_t *e_all;        // [n_copies] global expert id
  int64_t *s_all;        // [n_copies] source rank
  int64_t *flat_dst;     // [n_copies] dst row (int64 copy of scatter_index)
  bool *not_mine;        // [n_copies] owner != rank
  int64_t *expert_base;  // [nexperts] exclusive row base per expert; nullptr = skip
  int32_t *chunks;       // [world_size * world_size], pre-zeroed; nullptr = skip
                         // counting (metadata path derives it host-side)
  int64_t *pack_key;     // [copies_per_rank]
  // compress pack fusion: per-(segment, local token) flags, SEG-MAJOR
  // [nseg, tokens_per_rank] (contiguous per segment for the pack scan),
  // pre-zeroed; nullptr = skip. topk/local_world_size/node_idx only read
  // when pack_flag != nullptr.
  int32_t *pack_flag;
  int topk;
  int local_world_size;
  int node_idx;
  // fused consumer build: per-global-token keep flag (dedup recv rows are its
  // exclusive cumsum), pre-zeroed incl. the +1 garbage slot; nullptr = skip.
  // Keep rule mirrors the compress consumer: same-node source -> owner == rank;
  // remote source -> union_bcast ? dst node == my node : owner == rank.
  int32_t *mine_token;
  bool union_bcast;
};
void a2av_stage1_impl(A2AVStage1Arguments const &args, cudaStream_t stream);

// a2av compress consumer build, fused (replaces the ATen key/argsort/
// index_select chain): one grid-stride pass over all copies assigns each kept
// copy its A row via the host block-start table offA (A-order groups
// g = e_loc * W + s) plus a per-group atomic rank — interior order within a
// group is arbitrary, which no consumer observes (gather/scatter are per-row
// indirections and the tile gating compares only group boundaries; same
// design as AgScatterSortOpV2). Writes gather_A[row] = c_excl[token] (the
// dedup recv row) and scatter_D[row] = flat_dst - expert_base[e]. Rows past
// M_this_ep are never read by the GEMM and stay untouched. When lane_end !=
// nullptr (Tier B / lb_union) it also histograms rows into gating lanes by
// upper-bounding the recv row in lane_end[0..W-1]; a second tiny kernel turns
// the histogram into the inclusive per-expert lane cumsum the claimer reads.
// blk_cnt and gate_hist are [E * W] i32 scratch, pre-zeroed.
struct A2AVConsumerBuildArguments {
  int64_t n_copies;
  int topk;
  int ep_start;
  int ep_nexperts;            // E
  int world_size;             // W
  int64_t const *e_all;       // [n_copies] global expert id
  int64_t const *s_all;       // [n_copies] source rank
  int64_t const *flat_dst;    // [n_copies]
  bool const *not_mine;       // [n_copies]
  int64_t const *c_excl;      // [ntokens + 1] exclusive cumsum of mine_token
  int64_t const *offA;        // [E * W] A-order group starts (meta arena)
  int64_t const *expert_base; // [nexperts] (meta arena)
  int32_t *blk_cnt;           // [E * W], pre-zeroed
  int32_t *gather;            // [>= M_this_ep] out: A row -> dedup recv row
  int32_t *scatter;           // [>= M_this_ep] out: A row -> per-expert D row
  // Tier B gating (nullptr = skip): lane_end = gate_q row 0 shifted by one
  // (i.e. end(0..W-1)); gate_hist accumulates per-(e_loc, lane) row counts
  int64_t const *lane_end;    // [W]
  int32_t *gate_hist;         // [E * W], pre-zeroed
  // 2026-08-22 LANE-KEYED A order (Tier B): the tile gate partitions an
  // expert's A rows by LANE (window) via gating_cumsum, so the A order must be
  // lane-monotone within each expert; source-keyed groups (offA) are NOT —
  // windows cut through source regions (audit bug (b): one torn row per
  // rank under changing payloads). Two-pass protocol:
  //   hist_only = true            -> only gate_hist is accumulated (pass 1)
  //   offA_lane != nullptr        -> row = offA_lane[e_loc*W + lane] + atomic
  //                                  in-lane rank (pass 3; gate_hist untouched)
  //   both unset                  -> legacy source-keyed single pass
  bool hist_only;
  int64_t const *offA_lane;   // [E * W] lane-keyed A-order group starts
};
void a2av_consumer_build_impl(A2AVConsumerBuildArguments const &args, cudaStream_t stream);

// Tier B finalize: gating_cumsum[e][w] = sum_{w' <= w} gate_hist[e][w']
// (inclusive over lanes; empty lanes repeat the previous value)
struct A2AVGatingCumsumArguments {
  int ep_nexperts;             // E
  int world_size;              // W
  int32_t const *gate_hist;    // [E * W]
  int32_t *gating_cumsum;      // [E * W] out
  // optional lane-keyed A offsets: offA_lane[e*W + w] = offA[e*W] (the
  // expert's first A row, source-keyed table) + exclusive lane prefix
  int64_t const *offA;         // [E * W] or nullptr
  int64_t *offA_lane;          // [E * W] out or nullptr
};
void a2av_gating_cumsum_impl(A2AVGatingCumsumArguments const &args, cudaStream_t stream);

// compress pack fusion stage 2: one block per send segment; a multi-tile
// block-wide exclusive scan over the segment's flag row assigns each flagged
// token its exclusive rank, writing pack_gather[seg_off[seg] + rank] = token.
// Replaces the ATen scatter/cumsum/scatter chain (launch-count, not
// bandwidth: total work is nseg * tokens_per_rank int32).
struct A2AVPackScanArguments {
  int32_t const *pack_flag;  // [nseg, tokens] seg-major (from a2av_stage1)
  int64_t const *seg_off;    // [nseg] exclusive send-segment row offsets
  int64_t *pack_gather;      // [copies_per_rank + 1] send row -> local token
  int64_t tokens;            // tokens_per_rank
  int nseg;
};
void a2av_pack_scan_impl(A2AVPackScanArguments const &args, cudaStream_t stream);

void sort_scatter_index_to_per_expert(
    int *sorted_scatter_index,
    int *splits_gpu,
    int ep_start,
    int ep_nexperts,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// In-window hc metadata derivation (campaign-2 planner v2b, rule 5): the
// dispatch_only_routed entry derives splits / splits_per_source /
// a2av_unique_counts / a stable scatter_index ON DEVICE from the raw
// replicated topk routing, inside the timed window — replacing the python
// setup-time metadata of the pre-v2 epic hc arms.
// ---------------------------------------------------------------------------

// Per-copy histograms + per-token dedup counts from the replicated
// [ntokens_global, topk] routing. All outputs pre-zeroed by the caller.
// Sums of nonnegative ints are order-independent => deterministic and
// bitwise rank-identical (the replicated-data requirement).
struct A2AVMetaCountsArguments {
  int32_t const *topk_ids;  // [ntokens_global, topk] global expert ids
  int64_t ntokens;          // ntokens_global (= tokens_per_rank * W)
  int32_t topk;
  int32_t nexperts;         // E_virt
  int32_t ep_nexperts;      // experts per rank (owner = e / ep_nexperts)
  int32_t world_size;       // W (<= 128: per-token owner bitmask in 2x u64)
  int32_t nnodes;
  int32_t local_world;
  int64_t tokens_per_rank;
  int32_t *splits;          // [nexperts] out
  int32_t *sps;             // [W, nexperts] out (splits_per_source)
  int32_t *uc;              // [W, W + nnodes] out (u_mat | U_mat)
};
void a2av_meta_counts_impl(A2AVMetaCountsArguments const &args, cudaStream_t stream);

// DETERMINISTIC counting-sort scatter index: bit-identical to the python
// argsort(stable).argsort() reference. (calc_scatter_index in
// src/cuda/moe_utils.cu is explicitly non-deterministic and must NEVER
// produce replicated cross-rank data.) Three launches: per-tile smem
// histograms -> single-block scans (expert_base + per-block offsets) ->
// stable emission in flat (token, k) order.
struct A2AVStableScatterArguments {
  int32_t const *topk_ids;  // [n_copies] flat
  int64_t n_copies;
  int32_t nexperts;
  int32_t *block_hist;      // [num_blocks, nexperts] scratch
  int32_t *block_offset;    // [num_blocks, nexperts] scratch
  int32_t *expert_base;     // [nexperts + 1] scratch (exclusive scan)
  int32_t *scatter_index;   // [n_copies] out
  int32_t num_blocks;       // = ceil(n_copies / kA2AVMetaTile)
};
constexpr int32_t kA2AVMetaTile = 2048;
void a2av_stable_scatter_index_impl(
    A2AVStableScatterArguments const &args, cudaStream_t stream);

struct ProblemSchedule {
  int32_t expert_id;
  int32_t m_start;
  int32_t m_end;
  int32_t source_rank_start;
  int32_t source_rank_end;

  friend std::ostream &
  operator<<(std::ostream &os, ProblemSchedule const &sched) {
    os << "expert_id:" << sched.expert_id;
    os << ",m_start:" << sched.m_start;
    os << ",m_end:" << sched.m_end;
    os << ",source_rank_start:" << sched.source_rank_start;
    os << ",source_rank_end:" << sched.source_rank_end;
    return os;
  }
};

// Obtain the ordering of tiles in the m dimension such that the number of ranks
// dependent on data from other ranks is minimized for the same amount of computation.
//
// tile_size: tile_M
// ntiles: how many tiles we prefer to choose in a split
// Returns: <expert_id, m_start, m_end>
std::vector<ProblemSchedule> get_sorted_problem_schedule(
    std::vector<int32_t> const &sorted_splits_cpu,
    DistEnv const &dist_env,
    int32_t nexperts_ep,
    int32_t tile_size,
    int32_t ntiles = 4);

std::vector<ProblemSchedule> get_sorted_problem_schedule_v2(
    const int32_t *const splits,
    int rank,
    int tp_size,
    const int *cumsum_per_rank_ptr,
    const int ep_start,
    const int ep_nexperts,
    const int tiled_m_size,
    const int num_weight_groups);

std::vector<ProblemSchedule> get_relax_sorted_problem_schedule_v2(
    std::vector<int32_t> const &splits,
    int rank,
    int tp_size,
    const int *split_accum_per_rank_ptr,
    const int expert_idx_offset,
    const int nexperts_ep,
    const int tiled_m_size,
    const int num_weight_groups,
    const int nfold);

std::vector<ProblemSchedule> get_sorted_problem_schedule_v2_with_ntiles_limit(
    std::vector<int32_t> const &splits,
    int rank,
    int tp_size,
    const int *split_accum_per_rank_ptr,
    const int expert_idx_offset,
    const int nexperts_ep,
    const int tiled_m_size,
    const int num_weight_groups,
    const int ntiles_limit);

// we shift the computation rank order to comply with the order of gathering data.
// for ranks of the same nodes, circular shift the ranks to make the current local rank
// to be processed first. for different nodes, do the same shifting strategy as the local ranks.
// e.g. two nodes with ranks [0,1,2,3], [4,5,6,7]:
//  for rank #1, the order is: (1,2,3,0,5,6,7,4)
//  for rank #6, the order is: (6,7,4,5,2,3,0,1)
CUTLASS_HOST_DEVICE
int
shift_rank_to_order(int rank, DistEnv const &dist_env) {
  auto [node_idx, local_rank] = dist_env.global_rank_to_node_idx_local_rank(rank);
  int node_idx_shift = (node_idx - dist_env.node_idx + dist_env.nnodes) % dist_env.nnodes;
  int local_rank_shift =
      (local_rank - dist_env.local_rank + dist_env.local_world_size) % dist_env.local_world_size;
  return dist_env.local_rank_to_global_rank(local_rank_shift, node_idx_shift);
}

CUTLASS_HOST_DEVICE
int
revert_order_to_rank(int order, DistEnv const &dist_env) {
  auto [node_idx, local_rank] = dist_env.global_rank_to_node_idx_local_rank(order);
  int node_idx_origin = (node_idx + dist_env.node_idx) % dist_env.nnodes;
  int local_rank_origin = (local_rank + dist_env.local_rank) % dist_env.local_world_size;
  return dist_env.local_rank_to_global_rank(local_rank_origin, node_idx_origin);
}

}  // namespace bytedance::flux
