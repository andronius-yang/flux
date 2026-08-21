//===- fused_ep_dispatch_impl.hpp --------------------------------- C++ ---===//
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

#pragma once
#include "flux/flux.h"
#include <nvshmem.h>
#include <nvshmemx.h>
namespace bytedance {
namespace flux {

// DeepEP-lineage fused expert dispatch (campaign-2 planner v2a): planning
// is NOT a separate step — destinations are sender-local arithmetic over a
// [S, K] physical-slot routing, a tiny per-(dest slot) counts vector rides
// INSIDE the dispatch launch (the internode.cu notify_dispatch analog:
// zero host collectives, exact deterministic layouts), token rows land at
// exact remote offsets with per-(dest slot, src) arrival signals, and the
// combine handle (source flat cell + route prob) rides as a per-row int4
// header. The IBGDA same-QP put/atomic ordering trick of upstream DeepEP
// does NOT port to Slingshot; ordering here is the repo-standard
// putmem_nbi -> __threadfence -> nvshmem_fence -> signal_op(ADD) sequence
// (dis_scatter_forward_impl.cu precedent), validated by the standalone
// probe kernels below before any op bring-up.
//
// Byte-oriented rows (row_bytes % 16 == 0); PE ids are global torch ranks
// (init_flux_shm asserts rank == PE). All signals are u64, calloc-init,
// NEVER memset: value after epoch k is k (or k*expected for aggregated
// slots); waits are GEQ run_id-scaled expectations.

struct FusedEpDispatchParams {
  // geometry
  int32_t rank;
  int32_t world_size;   // R
  int32_t nlp;          // local physical slots per rank; P = R * nlp
  int32_t S;            // tokens on this rank THIS call (<= S_max)
  int32_t K;            // entries per token
  int64_t row_bytes;    // H * sizeof(dtype), multiple of 16
  // inputs (device)
  const void *inputs_shard;   // [S, row_bytes]
  const int32_t *dst_phys;    // [S, K] global physical slot ids
  const float *probs;         // [S, K]
  // local scratch (device, non-symmetric)
  int32_t *my_counts;         // [P] send counts (zeroed per call)
  int32_t *block_hist;        // [num_hist_blocks, P] stable-sort histograms
  int32_t *pack_base;         // [P + 1] exclusive scan of my_counts
  int32_t *block_offset;      // [num_hist_blocks, P] scan output
  void *pack_data;            // [S_max*K, row_bytes] send staging
  int32_t *pack_hdr;          // [S_max*K, 4] {flat_cell, prob_bits, src, 0}
  int64_t *remote_base;       // [P] my segment's row base at each dest slot
  int32_t *seg_meta;          // [2*nlp] recv seg_rows | seg_start (D2H src)
  int32_t *recv_off;          // [nlp, R] recv row offset per (slot, src)
  // symmetric buffers
  int32_t *counts_sym;        // [R, P]: row s = rank s's send counts
  void *recv_data_sym;        // [max_recv_total, row_bytes]
  int32_t *recv_hdr_sym;      // [max_recv_total, 4]
  uint64_t *sig_counts;       // [R] counts arrival (per source)
  uint64_t *sig_data;         // [nlp, R] data arrival (per slot, src)
  // capacity (collective-trap bounds; identical on every rank)
  int32_t max_rows_per_pair;
  int64_t max_recv_total;
  // epoch
  uint64_t run_id;
  int32_t num_hist_blocks;
};

struct FusedEpCombineParams {
  int32_t rank;
  int32_t world_size;
  int32_t nlp;
  int32_t S;            // tokens per rank (uniform, combine cell grid)
  int32_t K;
  int64_t row_bytes;
  int32_t n_recv;               // rows this rank received at dispatch
  const void *expert_rows;      // [n_recv, row_bytes] gemm2 output
  const int32_t *recv_hdr_sym;  // headers recorded at dispatch
  const int32_t *recv_off;      // [nlp, R] segment offsets (from dispatch)
  const int32_t *counts_sym;    // [R, P] counts matrix (from dispatch)
  void *comb_data_sym;          // [S*K, row_bytes] home staging
  uint64_t *sig_comb;           // [R]; home waits GEQ run_id * nlp per src
  uint64_t run_id;
};

// dispatch launch sequence (all on `stream`): pack pass1 -> scan -> pack
// pass2 -> counts push -> send (waits counts, scans matrix, puts+signals)
// -> recv gate (+ weights extract into weights_out) -> seg_meta ready for
// the caller's single pinned D2H.
void fused_ep_dispatch_impl(
    const FusedEpDispatchParams &params,
    float *weights_out,           // [max_recv_total] fp32 (probs per row)
    int32_t num_comm_sm,
    int32_t group_begin,          // recv-gate slot range [begin, end)
    int32_t group_end,
    cudaStream_t stream);

// standalone recv-gate launch for m_groups > 1 pipelining (contiguous
// local-slot range [slot_begin, slot_end)).
void fused_ep_recv_gate_only(
    const FusedEpDispatchParams &params, int32_t slot_begin,
    int32_t slot_end, int32_t num_comm_sm, cudaStream_t stream);

// combine launch sequence: per-(slot, src) segment row puts into the home
// staging (cell-addressed via headers) -> quiet -> per-home always-signal.
void fused_ep_combine_impl(
    const FusedEpCombineParams &params,
    int32_t num_comm_sm,
    cudaStream_t stream);

// home-side combine gate: wait sig_comb[s] >= run_id * nlp for all s.
void fused_ep_combine_gate_impl(
    const uint64_t *sig_comb, int32_t world_size, int32_t nlp,
    uint64_t run_id, uint64_t spin_limit, cudaStream_t stream);

// standalone put->signal ordering probe (S4 hard gate): writer fills a
// payload with the epoch, puts it, fences, signals; reader waits GEQ then
// verifies every payload word equals the epoch. err_count accumulates
// mismatches. Exercises BOTH the fence+signal_op and the putmem_signal
// forms (form = 0 / 1).
void fused_ep_probe_impl(
    int32_t peer, int32_t *payload_sym, int32_t payload_words,
    uint64_t *sig_sym, uint64_t epoch, int32_t form, int32_t is_writer,
    int32_t *err_count, cudaStream_t stream);

// attribute-query preloads (lazy-load hang class): call at ctor.
void fused_ep_dispatch_preload();

}  // namespace flux
}  // namespace bytedance
