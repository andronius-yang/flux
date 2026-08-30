//===- moe_gather_rs.h -------------------------------------------- C++ ---===//
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

#include "flux/flux.h"

#pragma once
namespace bytedance::flux {

constexpr int kMaxNumGroups = 2;
constexpr int kMaxExpertCount = 1024;

struct GemmGatherArguments {
  int m;
  int n;
  int k;
  float alpha;
  float beta;
  void const *input;
  void const *weight;
  void const *bias;
  void *output;
  void const *gather_index;
  void const *gather_weight;
  void *gather_output;
};

struct GemmGatherStoreArguments {
  int m;
  int n;
  int k;
  int rank;
  int world_size;
  float alpha;
  float beta;
  const void *input;
  const void *weight;
  const void *bias;
  void *gemm_output;
  const index_t *gather_index;
  const void *gather_weight;
  void *gather_outputs;
  void **rs_outputs_ptrs;
};

struct GemmGatherRsArguments {
  int m;
  int n;
  int k;
  int rank;
  int world_size;
  float alpha;
  float beta;
  const void *input;
  const void *weight;
  const void *bias;
  void *gemm_output;
  const index_t *gather_index;
  const void *gather_weight;
  void *gather_outputs;
  const bool *finish_gather;
  void **rs_outputs_ptrs;
};

struct GemmGroupedV3GatherRSArguments {
  void *problem_sizes_device;
  int problem_count;
  float alpha;
  float beta;
  const void **ptr_A;
  const void **ptr_B;
  const void **ptr_C;
  void **ptr_D;
  void *lda;
  void *ldb;
  void *ldc;
  void *ldd;
  void *problem_sizes_host;
  int32_t rank;
  int32_t world_size;
  void **output_scatter_ptrs;
  void **inter_Ds;
  int32_t topk;
  int32_t *barrier;
  int32_t *routing_idx;
  int32_t SPLITS;
  int32_t totalM;
  int32_t n_dim;
  // following args are for expert parallel
  int32_t tp_world_size;
  int32_t ep_world_size;
  int32_t globalM;
  int32_t max_token_per_rank;
  int32_t ep_m_start;
  int32_t ep_m_end;
  float **input_scale_ptr;
  float **weight_scale_ptr_array;
  float **output_vec_scale_ptr;
  int32_t sm_margin;
  int32_t input_groups;
  int32_t *ep_pos_filtered;
  int32_t *ep_token_idx_filtered;
  int32_t *ep_total_token_acc;
};

struct GemmGroupedV2GatherRSArguments {
  void *problem_sizes;  // cutlass::gemm::GemmCoord*
  int problem_count;
  int *non_empty_problem_count;  // a pointer in GPU memory
  float alpha;
  float beta;
  void **ptr_A;
  void **ptr_B;
  void **ptr_C;
  void **ptr_D;
  void *lda;  // to support split_n
  void *ldb;
  void *ldc;
  void *ldd;
  void *ldr;
  // for FP8 arguments
  void **ptr_Aux = nullptr;     // m * n
  void **ptr_Vector = nullptr;  // bias: 1 * n
  float *abs_max_Aux = nullptr;
  float *abs_max_D = nullptr;
  // scaling tensors
  float const **scaleA = nullptr;
  float const **scaleB = nullptr;
  float const *scaleC = nullptr;
  float const *scaleD = nullptr;    // require if D is fp8
  float const *scaleAux = nullptr;  // require if Aux is fp8
  int32_t topk;
  int32_t *barrier;
  int32_t *routing_idx;
  int32_t n_split;
  // following args are for expert parallel
  int sm_margin;
  // M-split waves (Slipstream v2): device [n_waves] non-empty problem count per
  // cascade group (a wave can lack rows for an expert that is non-empty
  // elsewhere, so the uniform division is wrong there). nullptr = legacy
  // uniform per-split division (bit-exact column-split behavior).
  int const *non_empty_per_group = nullptr;
  // v2 chunked combine (FLUX_A2AV_RS_CHUNK_E): device [problem_count]
  // problem -> cascade group (wave) map for chunk-ordered problem lists;
  // nullptr = legacy uniform division.
  int const *prob_group_map = nullptr;
  // gen-8c epilogue-fused pack: per-problem D scatter-index pointers (built by
  // make_workspace; identity iota when fused pack is off)
  int **scatter_D_ptr = nullptr;
};

struct TopKReduceGatherRSArguments {
  int32_t rank;
  int32_t world_size;
  void **output_scatter_ptrs;
  void *inter_D;
  int32_t topk;
  int32_t *barrier;
  int32_t *routing_idx;
  int32_t SPLITS;
  int32_t totalM;  // M for the current group gemm
  int32_t n_dim;
  int32_t n_tb_blocks;
  int32_t tp_world_size;
  int32_t ep_world_size;
  // M for all the experts, should be equal to totalM when expert parallel is not enabled
  int32_t globalM;
  float *input_scale_ptr;
  float *output_vec_scale_ptr;
};

struct TopKReduceGatherRSV2Arguments {
  void *input_ptrs[kMaxNumGroups];  // [input_groups, m_this_ep * n] of output_dtype
  void *output_ptr;                 // [world_size, ntokens * n] of output_dtype
  float *output_vec_scale_ptrs[kMaxNumGroups];
  int *splits;
  int *routing_idx;  // [m_full = ntokens * topk]
  int m_full;  // M for all the experts, m_full = ntokens * topk; m_full == m_this_ep for EP=1
  int n;
  int nexperts;
  int topk;
  int input_groups;
  bool do_all_reduce = false;
  bool use_read_mode = false;

  int threadblock_count;
  int tile_size_m;  // tile shape m. 128 (the only instantiation).
  int tile_size_n;  // tile shape n: 1024 when (n / n_split) is 1024-aligned,
                    // else 512 (2026-08-21, K3 H=3584). (n / n_split) %
                    // tile_size_n == 0 required; see combine_tile_n().
  // for reduce_scatter
  int rank;
  int world_size;
  int n_split;
  // peer pointer arrays below are local (intra-node) scoped: [local_world_size] entries,
  // indexed by local rank. for nnodes == 1 local == global.
  int **barrier;            // [local_world_size][n_split * 2]
  void **reduce_ptrs;       // [local_world_size][ntokens * n] of output_dtype
  int **tile_barrier_ptrs;  // [local_world_size][num_tiles]
  // hierarchical multi-node reduce-scatter (nnodes > 1). identity values for nnodes == 1.
  int nnodes = 1;
  int node_idx = 0;
  int local_rank = 0;        // == rank when nnodes == 1
  int local_world_size = 1;  // must be set to world_size when nnodes == 1
  int staging_rows = 0;      // max token rows per (node, split) staging slot: max_m/topk/world_size
  // [nnodes, n_split, staging_rows, n/n_split] on the NVSHMEM symmetric heap; the accumulated
  // per-node partial for remote-owned segments is staged here for host-issued putmem_signal
  void *staging_send = nullptr;
  int *group_flags = nullptr;     // [nnodes * n_split] kernel -> host chunk-ready flags
  int *group_counters = nullptr;  // [nnodes * n_split] per-block completion counters
};

// ---- a2av_hier combine (layer1 alltoallv): pack + reduce kernel arguments ----

constexpr int kA2AVMaxNodes = 16;

// Persistent pack kernel: split-major outer loop gated on the GEMM's per-split
// ready flag; per split it gathers each outgoing copy's n_per column window from
// gemm_out into the symmetric send panel in (home_rank, expert, copy) order
// (== layer0 a2av's recv layout), applying output_vec_scale per source row.
// Chunk completion per (dest_node, sid) is published to the host put ladders via
// the group_counters/group_flags handshake -- including the OWN node's chunk
// (the intra-node ladder gates on it), unlike the dense ring kernel.
struct A2AVCombinePackArguments {
  void const *gemm_out;         // [m_this_ep, n] of dtype, expert-major rows
  float const *vec_scale;       // [m_this_ep] per-row topk weight, nullptr if absent
  int32_t const *pack_index;    // [m_this_ep]: send-panel row -> gemm_out row
  void *send_panel;             // [n_split, panel_rows, n_per] symmetric
  int *barrier;                 // local per-split GEMM ready flags (cascade output)
  int *group_flags;             // [nnodes * n_split] kernel -> host chunk-ready flags
  int *group_counters;          // [nnodes * n_split] per-block completion counters
  int64_t node_row_start[kA2AVMaxNodes + 1];  // dest-node row ranges in the send panel
  int64_t panel_rows;           // send panel row capacity per split
  int n;
  int n_per;                    // n / n_split
  int n_split;
  int nnodes;
  int node_idx;
  int threadblock_count;
  // M-split waves (Slipstream v2, FLUX_A2AV_RS_MSPLIT): the GEMM completes in
  // destination-wave order (ring waves of dest-node row segments, n_split==1),
  // and the pack gates PER RING STEP on the wave's cascade flag instead of the
  // single split flag — the whole per-node ladder then pipelines under the
  // remaining GEMM. msplit == 0 keeps the legacy per-split gate bit-exact.
  int msplit;                       // 0 = legacy
  int wave_of_node[kA2AVMaxNodes];  // schedule step -> barrier wave-flag index
  int node_order[kA2AVMaxNodes];    // schedule step -> dest node (ring or size-sorted)
  // gen-8c epilogue-fused pack: the GEMM already wrote the send panel via
  // ScatterD, so the pack degenerates to a pure FLAG RELAY — wait each wave
  // flag and flip the per-node chunk flags, moving no data.
  int relay_only;
  // v2 M2 pieces: number of per-expert-chunk cascade flags to wait at entry
  // (the no-split build fires barrier[0..n) per chunk; wave_of_node stays 0).
  // 0 = legacy wave gating.
  int n_chunk_flags;
};

// gen-8c: invert an int32 permutation-ish map (out[idx[p]] = p) — builds the
// pack inverse (gemm row -> send-panel row) from pack_index at plan time.
struct A2AVInvertIndexArguments {
  int32_t const *idx;  // [n]
  int32_t *out;        // [n]
  int64_t n;
};
void a2av_invert_index(A2AVInvertIndexArguments const &args, cudaStream_t stream);

// Per-split topk reduce at the destination: launched once per split after all W
// per-source recv signals for that split have fired; sums each local token's topk
// recv-panel rows in fp32 and writes the [:, sid*n_per : (sid+1)*n_per] window of
// the output shard.
struct A2AVCombineReduceArguments {
  void const *recv_panel;       // [n_split, panel_rows, n_per] symmetric
  int32_t const *reduce_index;  // [ntokens_local * topk]: local copy -> recv-panel row
  void *output;                 // [ntokens_local, n]
  int64_t panel_rows;           // recv panel row capacity per split
  int64_t ntokens_local;
  int n;
  int n_per;
  int topk;
  int sid;
  int threadblock_count;
};

// Sort-free compress-plan derivation (2026-08-21): every ordering in the
// compress CSRs is arithmetic on the layer0 stable scatter_index plus host
// cnt/U prefix tables (the conv panel is the destination A-order restricted
// per (segment, owner-lane) block; wire/red groupings are per-token O(topk)
// ranks). 4 kernels, no sorts, deterministic direct writes. Replaces the
// argsort-based build_a2av_compress_indices on the derive path (which stays
// as the FLUX_A2AV_RS_CHECK_IDENTITY reference).
struct A2AVCompressPlanArguments {
  // per-copy inputs (device)
  int32_t const *scatter_index;  // [m_full] global A-order position per copy
  int32_t const *e_of_copy;      // [m_full] expert of each copy
  // prefix tables (device; built host-side from cnt/U, no device sync)
  int32_t const *home_base;    // [nex * W] exclusive per-expert home prefix
  int64_t const *expert_base;  // [nex] exclusive A-order expert base
  int64_t const *conv_base;    // [(NN-1) * L * E_loc] conv bucket bases
  int64_t const *my_cnt_cum;   // [nex] exclusive prefix of cnt[rank][e]
  int64_t const *recv_off_C;   // [W]
  int64_t const *recv_off_Cp;  // [W]
  int64_t const *rem_base;     // [NN]
  // scratch (device, zeroed by the launcher where required)
  int32_t *conv_count;   // [(NN-1) * tokens_per_rank] conv copies per (seg, t)
  int32_t *wire_row_of;  // [(NN-1) * tokens_per_rank] (seg, t) -> wire row, -1 if none
  int32_t *red_flags;    // [ntok_local * NN] token contributes from node m
  int32_t *rem_pos;      // [ntok_local * NN] column-exclusive one-cumsum
  // outputs (device int32, sized by the host from cnt/U totals)
  int32_t *wire_ptr;   // [wire_total + 1]
  int32_t *wire_copy;  // [conv_total]
  int32_t *red_ptr;    // [ntok_local + 1]
  int32_t *red_row;    // [own_total + rem_total]
  // geometry
  int64_t m_full;
  int topk;
  int world_size;
  int nnodes;
  int local_world_size;
  int rank;
  int64_t nexperts;
  int64_t ep_nexperts;  // experts per owner rank
};

void a2av_compress_plan(A2AVCompressPlanArguments const &args, cudaStream_t stream);

// Kernel-side a2av combine pack/reduce index build (2026-08-29 plan-lane
// de-serialization): the sort-free identities of build_a2av_combine_indices
// as two direct-write kernels over host prefix tables — replaces the ~15-op
// torch dispatcher chain (arange/searchsorted/index_select/scatter + two
// pageable H2Ds) whose host serialization dominated derive_combine_meta
// (~0.64 ms host for ~0.03 ms GPU, step-0 nsys 20260829-081712). The torch
// chain stays as the FLUX_A2AV_RS_CHECK_IDENTITY reference.
struct A2AVCombinePlanArguments {
  int32_t const *routing_idx;  // [m_full] layer0 stable scatter index (device)
  // prefix tables (device; host-built from cnt, ONE pinned async H2D)
  int64_t const *cumA;       // [nexG] inclusive A-order (e_loc, h) group cum
  int64_t const *offA;       // [nexG] exclusive A-order group base
  int64_t const *offR_of_A;  // [nexG] recv-panel base of A-order group
  int64_t const *expert_cum;  // [nex] inclusive global per-expert row cum
  int64_t const *my_cum;      // [nex] exclusive prefix of cnt[rank][e]
  int64_t const *h_base;      // [nex] rank's exclusive home base within e
  // outputs (device int32)
  int32_t *pack_index;    // [m_this_ep]
  int32_t *reduce_index;  // [cpr]
  int64_t m_this_ep;
  int64_t cpr;   // copies per rank (m_full / W)
  int64_t row0;  // rank * cpr slice offset into routing_idx
  int64_t nexG;  // E_loc * W A-order groups
  int64_t nex;   // total experts
};

void a2av_combine_plan(A2AVCombinePlanArguments const &args, cudaStream_t stream);

constexpr int kA2AVMaxWorld = 64;

// Eager (arrival-order) destination reduce: ONE persistent kernel per forward,
// launched with no front-end waits. Per output element it keeps a remaining-mask
// over the token's topk recv rows and folds in any row whose source lane's
// per-split recv signal has already fired (64-bit acquire poll), spinning only
// when no remaining lane has arrived — the minimal real dependency, replacing
// the host-side wait-all-W gate per split. The source lane of a recv row is
// recovered by binary search over recv_cum (per-source rows are contiguous in
// the recv panel, and split slices columns, so the prefix is split-invariant).
// ---- compress (dedup) combine: pre-reduce + CSR reduce kernel arguments ----

// Source-side gateway pre-reduce (persistent, one launch per forward): per
// (split, target node in inter-ladder rotation order) it spins on the L
// per-peer convergence signals, merges each wire row's contributing conv-panel
// rows (CSR) in fp32, writes the wire panel, and flips the (tn, sid) wire flag
// the host inter ladder gates on -- the pack kernel's counter/flag handshake.
struct A2AVCombinePreReduceArguments {
  void const *conv_panel;        // [n_split, conv_rows, n_per] symmetric
  void *wire_panel;              // [n_split, wire_rows, n_per] symmetric
  int32_t const *wire_ptr;       // [wire_rows_local + 1] CSR offsets
  int32_t const *wire_copy;      // [conv_rows_local] wire row -> conv-panel rows
  uint64_t const *conv_signals;  // [(L * nnodes) * n_split], slot (ls*NN+tn)*n_split+sid
  uint64_t run_id;
  int *wire_flags;               // [nnodes * n_split] kernel -> host wire-ready flags
  int *wire_counters;            // [nnodes * n_split] per-block completion counters
  int node_order[kA2AVMaxNodes];  // schedule step -> remote target node (ring default)
  int64_t wire_seg_start[kA2AVMaxNodes + 1];  // wire-row start per segment (tn asc skip own)
  int64_t conv_rows;             // conv panel row capacity per split
  int64_t wire_rows;             // wire panel row capacity per split
  int n_per;
  int n_split;
  int nnodes;
  int node_idx;
  int local_world_size;
  int threadblock_count;
  // 0 = unlimited (default). >0: trap after this many no-progress sleep
  // iterations in the conv-signal wait — converts a missing-signal bug into
  // a loud abort instead of a hang (FLUX_A2AV_RS_SPIN_LIMIT). LAST field
  // with default: launch sites use positional aggregate init.
  uint64_t spin_limit = 0;
};

// Legacy-gate destination reduce under compress: per-token contribution count
// varies (own-node copies + one merged row per contributing remote node), so
// the fixed-topk reduce_index becomes the red_ptr/red_row CSR.
struct A2AVCombineCSRReduceArguments {
  void const *recv_panel;    // [n_split, panel_rows, n_per] symmetric (C' image)
  int32_t const *red_ptr;    // [ntokens_local + 1]
  int32_t const *red_row;    // [red_total]: token contributions, recv-panel rows
  void *output;              // [ntokens_local, n]
  int64_t panel_rows;
  int64_t ntokens_local;
  int n;
  int n_per;
  int sid;
  int threadblock_count;
};

struct A2AVCombineEagerReduceArguments {
  void const *recv_panel;        // [n_split, panel_rows, n_per] symmetric
  int32_t const *reduce_index;   // [ntokens_local * topk]: local copy -> recv-panel row
  // compress: variable per-token contributions replace the fixed-topk stride;
  // when red_ptr != nullptr the kernel walks red_ptr/red_row and reduce_index
  // is ignored. recv_cum then describes the C' image (zero-width lanes for
  // non-materialized remote ranks -- they can never contain a row).
  int32_t const *red_ptr = nullptr;  // [ntokens_local + 1]
  int32_t const *red_row = nullptr;  // [red_total]
  void *output;                  // [ntokens_local, n]
  uint64_t const *recv_signals;  // [world_size * n_split] epoch signals, never reset
  uint64_t run_id;               // this epoch's expected signal value (GEQ)
  int64_t recv_cum[kA2AVMaxWorld + 1];  // recv-row prefix by source rank; [W] = image rows
  int64_t panel_rows;            // recv panel row capacity per split
  int64_t ntokens_local;
  int world_size;
  int n;
  int n_per;
  int n_split;
  int topk;
  int threadblock_count;
  // 0 = unlimited (default). >0: trap after this many consecutive
  // no-progress sleeps (FLUX_A2AV_RS_SPIN_LIMIT). LAST field with default:
  // launch sites use positional aggregate init.
  uint64_t spin_limit = 0;
};

// ---- lane-chain receiver (Slipstream v2b, FLUX_A2AV_RS_LANE_CHAIN) --------
// Replaces per-element signal polling (eager) / host wait-all (legacy) with a
// per-LANE chain in EXPECTED arrival order: one front-end wait per lane
// releases a tiny scatter-add of just that lane's recv rows into an fp32
// accumulator; a finalize cast runs after the last lane. Waiting machinery is
// O(W) front-end waits instead of O(elements x polls); the reduce drips in
// behind each arrival so the post-last-arrival tail is one lane + the cast.

// Invert the per-token reduce CSR into a recv-row -> token map (plan-time
// arithmetic on tensors the plan already holds; launched on the reduce stream
// before the chain, ~tens of us).
struct A2AVLaneTokenMapArguments {
  int32_t const *red_ptr;  // [ntokens + 1]
  int32_t const *red_row;  // [red_total] recv-panel rows per token
  int64_t ntokens;
  int32_t *token_of;       // [image_rows] out: recv row -> local token
  // optional (arrival-dynamic receiver): per-token outstanding-contribution
  // counters, remain[t] = red_ptr[t + 1] - red_ptr[t]. LAST field with
  // default: existing launch sites use positional aggregate init.
  int32_t *remain = nullptr;  // [ntokens] out, skipped when nullptr
};

struct A2AVLaneReduceArguments {
  void const *recv_panel;      // [image_rows, n] (n_split == 1)
  int32_t const *token_of;     // recv row -> local token
  float *scratch;              // [ntokens, n] fp32 accumulator
  int64_t row_lo;              // lane's first recv row
  int64_t nrows;               // lane's row count
  int n;
  int threadblock_count;
  // Lanes are chained serially on one stream, so races exist only WITHIN a
  // lane. Remote C' lanes carry one merged row per token (collision-free ->
  // plain adds); own-node lanes can carry several copies of a token
  // (multiple local experts -> atomicAdd). 0 = plain, 1 = atomic.
  int use_atomic;
};

struct A2AVFinalizeArguments {
  float const *scratch;  // [ntokens, n]
  void *output;          // [ntokens, n] of dtype
  int64_t ntokens;
  int n;
  int threadblock_count;
};

void a2av_lane_token_map(A2AVLaneTokenMapArguments const &args, cudaStream_t stream);
void a2av_combine_lane_reduce(
    A2AVLaneReduceArguments const &args, DataTypeEnum dtype, cudaStream_t stream);
void a2av_combine_finalize(
    A2AVFinalizeArguments const &args, DataTypeEnum dtype, cudaStream_t stream);

// ---- completion-bucketed register receiver (Slipstream gen-10,
// FLUX_A2AV_RS_BUCKET) -------------------------------------------------------
// Arrival-order folding at wait-all's 1x bytes: tokens bucket by the chain
// position of their LAST-arriving contribution lane (plan-time, on-stream,
// ~us); each front-end lane wait then releases a register-CSR fold of exactly
// the tokens that lane completes. No fp32 scratch RMW (lane-chain's 4-5x byte
// amplification), no atomics in the fold, no finalize: every token is read
// once and written once, as early as legality allows. The post-last-arrival
// tail is one bucket instead of the whole reduce.
struct A2AVBucketMapArguments {
  int32_t const *red_ptr;    // [ntokens + 1]
  int32_t const *red_row;    // [red_total] recv-panel rows per token
  int32_t const *lane_off;   // [world_size + 1] C' recv-row prefix by source rank
  int32_t const *chain_pos;  // [world_size] source rank -> chain position
  int world_size;
  int n_chain;               // chain length S (= L + NN - 1 materializing lanes)
  int64_t ntokens;
  int32_t *comp;             // [ntokens] out: completion chain position
  int32_t *bucket_cnt;       // [n_chain] out: bucket sizes (pre-zeroed)
};
struct A2AVBucketScanArguments {
  int32_t const *bucket_cnt;  // [n_chain]
  int n_chain;
  int32_t *bucket_ptr;        // [n_chain + 1] out: exclusive prefix
  int32_t *bucket_cur;        // [n_chain] out: zeroed scatter cursors
};
struct A2AVBucketScatterArguments {
  int32_t const *comp;        // [ntokens]
  int32_t const *bucket_ptr;  // [n_chain + 1]
  int64_t ntokens;
  int32_t *bucket_cur;        // [n_chain] scatter cursors
  int32_t *bucket_tok;        // [ntokens] out: tokens grouped by completion bucket
};
struct A2AVBucketReduceArguments {
  void const *recv_panel;     // [n_split, panel_rows, n_per] symmetric (C' image)
  int32_t const *red_ptr;     // [ntokens_local + 1]
  int32_t const *red_row;     // [red_total]
  int32_t const *bucket_ptr;  // [n_chain + 1] (device; sizes unknown to host)
  int32_t const *bucket_tok;  // [ntokens_local]
  int bucket;                 // which completion bucket this launch folds
  void *output;               // [ntokens_local, n]
  int64_t panel_rows;
  int n;
  int n_per;
  int sid;
  int threadblock_count;
};
void a2av_bucket_map(A2AVBucketMapArguments const &args, cudaStream_t stream);
void a2av_bucket_scan(A2AVBucketScanArguments const &args, cudaStream_t stream);
void a2av_bucket_scatter(A2AVBucketScatterArguments const &args, cudaStream_t stream);
void a2av_combine_bucket_reduce(
    A2AVBucketReduceArguments const &args, DataTypeEnum dtype, cudaStream_t stream);

// ---- arrival-dynamic receiver (H4, FLUX_A2AV_RS_RECV_DYN) ------------------
// The bucket receiver's host wait chain consumes lanes in EXPECTED arrival
// order: one out-of-order arrival head-of-line blocks the folds of every lane
// already landed behind it (S = L + NN - 1 sequential front-end waits). This
// ONE persistent kernel replaces the chain: warps poll the S per-lane epoch
// signals directly, claim row chunks of any ARRIVED lane via per-lane atomic
// cursors, and decrement each row's token counter; the warp whose decrement
// completes a token folds it immediately (register CSR walk in red_ptr order,
// read once / written once -- the bucket fold's exact arithmetic, so the
// output stays BITWISE-identical to the wait-all reduce; only the schedule is
// arrival-dependent). The post-last-arrival tail is that lane's rows plus the
// tokens it completes -- the bucket receiver's ideal tail, robust to any
// arrival permutation. Zero-row lanes are excluded host-side (they still
// signal; no token can complete there).
struct A2AVDynReduceArguments {
  void const *recv_panel;        // [image_rows, n] (n_split == 1, sid 0)
  int32_t const *red_ptr;        // [ntokens_local + 1]
  int32_t const *red_row;        // [red_total] recv-panel rows per token
  int32_t const *token_of;       // [image_rows] recv row -> local token
  int32_t *remain;               // [ntokens_local] outstanding contributions (pre-filled)
  // remain[] extent. v2 slack-row hardening (16n b32+ livelock fix): token_of
  // is -1-filled before the map kernel, and the kernel skips any row whose
  // token falls outside [0, ntokens_local) — uc-derived lane extents may
  // exceed the reduce CSR's coverage, and such slack rows must be claimed
  // (so lanes exhaust) but never decrement remain or fold.
  int64_t ntokens_local;
  int32_t *lane_cursor;          // [n_lanes] row-claim cursors (pre-zeroed)
  void *output;                  // [ntokens_local, n]
  uint64_t const *recv_signals;  // [world_size * n_split] epoch signals, never reset
  uint64_t run_id;               // this epoch's expected signal value (GEQ)
  int32_t lane_sig[kA2AVMaxWorld];     // lane -> recv_signals slot (rank * n_split + sid)
  int64_t lane_row_lo[kA2AVMaxWorld];  // lane -> first recv row of its C' block
  int32_t lane_rows[kA2AVMaxWorld];    // lane -> row count (> 0)
  int n_lanes;                   // materializing lanes with rows (<= L + NN - 1)
  int n;                         // row width (== n / n_split, n_split == 1)
  int chunk_rows;                // rows claimed per cursor bump
  int threadblock_count;
  // 0 = unlimited (default). >0: trap after this many consecutive
  // no-progress sleeps (FLUX_A2AV_RS_SPIN_LIMIT). LAST field with default:
  // launch sites use positional aggregate init.
  uint64_t spin_limit = 0;
};
void a2av_combine_dyn_reduce(
    A2AVDynReduceArguments const &args, DataTypeEnum dtype, cudaStream_t stream);

}  // namespace bytedance::flux
