//===- gemm_grouped_v2_gather_rs.cc -------------------------------------------- C++ ---===//
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

#include "moe_gather_rs/ths_op/gemm_grouped_v2_gather_rs.h"

#include <ATen/core/List.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/ivalue.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <cuda_runtime_api.h>
#include <cutlass/fast_math.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <torch/all.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cutlass/util/device_memory.h>
#include <cutlass/util/packed_stride.hpp>
#include <iostream>
#include <optional>
#include <nvshmemx.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <utility>
#include <vector>

#include "host/nvshmem_api.h"
#include "host/nvshmemx_api.h"

#include "flux/args/moe_gather_rs.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"
#include "flux/utils.h"
#include "moe_gather_rs/topk_gather_rs.hpp"
#include "moe_gather_rs/workspace_helper.h"
#if defined(FLUX_WITH_TRITON_AOT)
#include "moe_utils.h"
#include "triton_aot_generated/flux_triton_aot.h"
#endif

namespace {
// the copy tile size for TopkReduceScatterOp. has nothing to do with the GEMM tile_size_m
static constexpr int kTileSizeM = 128, kTileSizeN = 1024;
// 2026-08-21 (K3 H=3584 = 7*512): the dense-combine tile N drops to 512 when
// n_per_split is not 1024-aligned. Single policy point — MUST agree with the
// kernel-side dispatch in topk_gather_rs_v2.cu (tile-barrier sizing and
// args.tile_size_n are derived from this). 1024-aligned shapes select the
// same instantiation as before this change.
static constexpr int kTileSizeNMin = 512;
static inline int
combine_tile_n(int n_dim, int n_split) {
  return (n_dim / n_split) % kTileSizeN == 0 ? kTileSizeN : kTileSizeNMin;
}
long
get_args_workspace_size(int problem_count) {
  using bytedance::flux::pad_to;
  constexpr int kAlignment = 128;
  // the workspace size
  int bytes =
      pad_to(sizeof(cutlass::gemm::GemmCoord) * problem_count, kAlignment) * 1  // problem_sizes
      + pad_to(sizeof(void *) * problem_count, kAlignment) * 4   // ptr_A/ptr_B/ptr_C/ptr_D
      + pad_to(sizeof(int64_t) * problem_count, kAlignment) * 5  // lda/ldb/ldc/ldd/ldr
      + pad_to(sizeof(float *) * problem_count, kAlignment) * 2  // scale_A/scale_B
      + pad_to(sizeof(int *) * problem_count, kAlignment) * 1    // scatter_D (gen-8c)
      + pad_to(sizeof(int) * 1, kAlignment) * 1;                 // non_empty_problem_count
  return bytes;
}
c10::optional<std::vector<torch::Tensor>>
as_optional_vec(c10::optional<torch::Tensor> &t) {
  if (t.has_value()) {
    return c10::optional<std::vector<torch::Tensor>>{{t.value()}};
  }
  return {};
}

// 2026-08-22 wire-ordering HARD RULE (CLAUDE.md invariant 5 / SCHEMA rule 6):
// on libfabric/CXI the nbi put_signal exposes the flag before the data, so
// every combine-wire put a consumer gates on is BLOCKING by default.
// FLUX_A2AV_RS_BLOCKING_WIRE=0 restores the refuted nbi wire (ablation only).
static inline bool
flux_rs_blocking_wire() {
  static const bool v =
      bytedance::flux::get_int_from_env("FLUX_A2AV_RS_BLOCKING_WIRE", 1) != 0;
  return v;
}
static inline void
flux_rs_put_signal(
    void *dst,
    const void *src,
    size_t bytes,
    uint64_t *sig,
    uint64_t val,
    int sig_op,
    int pe,
    cudaStream_t stream,
    int local_world_size,
    int my_node) {
  // INTER-NODE only: the ordering hazard is the libfabric/CXI proxy path.
  // Intra-node (P2P) puts stay nbi — CE-ordered, and the blocking on-stream
  // variant runs a device kernel that dereferences host-staged sources
  // (unspecified launch failure, 2026-08-22 audit).
  const bool inter_node = (pe / local_world_size) != my_node;
  if (inter_node && flux_rs_blocking_wire()) {
    nvshmemx_putmem_signal_on_stream(dst, src, bytes, sig, val, sig_op, pe, stream);
  } else {
    nvshmemx_putmem_signal_nbi_on_stream(dst, src, bytes, sig, val, sig_op, pe, stream);
  }
}


void *
data_ptr_or(c10::optional<torch::Tensor> &t, void *other) {
  return t.has_value() ? t->data_ptr() : other;
}
int
get_rs_threadblock_count() {
  static int rs_num_blocks = bytedance::flux::get_int_from_env("FLUX_RS_BLOCKS", 3);
  return rs_num_blocks;
}
// a2av_hier combine: SM budget of the pack / reduce kernels. Both are reserved
// out of the GEMM via sm_margin (the pack kernel is persistent and the per-split
// reduce must find free SMs while the GEMM still spins on later splits).
int
get_a2av_pack_blocks() {
  // Slipstream canonicalization (2026-08-23): 10/8/6 replaces 3/3/2 (specs had
  // pinned 6/6/4). The ladders do payload-proportional work behind fixed
  // grids and were CTA-starved at high budgets: 10/8/6 cuts l1 by
  // 1.9-2.1 ms at b64 (monotone 4->6->10 dose-response, both canon models)
  // and ties at b2-b8 — the extra GEMM reservation never bites where the
  // phase is comm-bound. Rule-4 DEFAULT tag: FLUX_A2AV_RS_CTA_1086_TAG.
  static int v = bytedance::flux::get_int_from_env("FLUX_A2AV_RS_PACK_BLOCKS", 10);
  return v;
}
int
get_a2av_reduce_blocks() {
  static int v = bytedance::flux::get_int_from_env("FLUX_A2AV_RS_REDUCE_BLOCKS", 8);
  return v;
}
int
get_a2av_prered_blocks() {
  (void)bytedance::flux::get_int_from_env("FLUX_A2AV_RS_CTA_1086_TAG", 0);
  static int v = bytedance::flux::get_int_from_env("FLUX_A2AV_RS_PRERED_BLOCKS", 6);
  return v;
}
// Slipstream v2 (2026-08-24): M-split destination-wave combine. The GEMM is
// decomposed into (ring wave of dest nodes, expert) ROW sub-problems (full N;
// requires n_split == 1) whose cascade flags release the pack -> conv ->
// prered -> wire ladder PER TARGET NODE while the GEMM still runs: put count
// stays at the ns1 minimum (NN-1 blocking puts/rank) with M-axis pipelining —
// the resolution of the "n_split multiplies the proxy-bound put count"
// tension (handoff 16 §1). Default OFF; own rule-4 never-mix boundary.
int
get_a2av_rs_msplit() {
  (void)bytedance::flux::get_int_from_env("FLUX_A2AV_RS_MSPLIT_TAG", 0);
  // Slipstream supersession (2026-08-24, user decision): msplit + fused pack
  // + wave pack + bucket are the official Slipstream defaults (binary tag
  // FLUX_A2AV_SLIPSTREAM2_TAG; SCHEMA rule 13). Explicit env still wins;
  // activation stays guarded by a2av_hier && nnodes > 1 (&& compress for the
  // receiver), so dense / single-node paths are untouched.
  (void)bytedance::flux::get_int_from_env("FLUX_A2AV_SLIPSTREAM2_TAG", 0);
  static int v = bytedance::flux::get_int_from_env("FLUX_A2AV_RS_MSPLIT", 1);
  return v;
}
// Ring-consecutive dest nodes per wave (tile-quantization dial; waves never
// cross the ring wrap). 1 = per-node waves (finest pipelining; default — tile
// waste hides under the wire in every wire-bound cell per the 8.24 model).
int
get_a2av_rs_wave_nodes() {
  static int v =
      std::max<int>(1, bytedance::flux::get_int_from_env("FLUX_A2AV_RS_WAVE_NODES", 1));
  return v;
}
// Lane-chain receiver (Slipstream v2b, 2026-08-24): per-LANE front-end waits
// chained in EXPECTED arrival order (descending ring, own node last) release
// tiny per-lane scatter-adds into an fp32 accumulator + one finalize cast.
// O(W) waits replace the eager kernel's O(elements x polls) system-scope
// acquire polling and the legacy wait-all's zero receive-overlap. Requires
// the compress C' image and n_split == 1. Own never-mix boundary.
int
get_a2av_rs_lane_chain() {
  (void)bytedance::flux::get_int_from_env("FLUX_A2AV_RS_LANE_CHAIN_TAG", 0);
  static int v = bytedance::flux::get_int_from_env("FLUX_A2AV_RS_LANE_CHAIN", 0);
  return v;
}
// Size-ordered waves (Slipstream gen-8a, user-directed 2026-08-24): 0 = ring
// order (canonical), 1 = remote waves sorted by DESCENDING segment size (ties
// by ring position; own node stays last). The largest put starts earliest and
// the hot receiver's inflows begin first (NR-03). Requires WAVE_NODES == 1.
// Pre-registered falsifier: size-order correlates senders toward hot
// destinations — if inbound bunching (incast) at the hot node exceeds the
// early-start win, ring stays canon. Every schedule consumer (pack, prered,
// conv/wire ladders, GEMM wave order) follows ONE host-built order array, so
// ring mode is bit-identical to the pre-knob binary.
int
get_a2av_rs_wave_order() {
  (void)bytedance::flux::get_int_from_env("FLUX_A2AV_RS_WAVE_ORDER_TAG", 0);
  static int v = bytedance::flux::get_int_from_env("FLUX_A2AV_RS_WAVE_ORDER", 0);
  return v;
}
// Epilogue-fused pack (gen-8c, user-directed 2026-08-24): the l1 GEMM writes
// the dest-major send panel DIRECTLY through a ScatterD epilogue (pack inverse
// indices) with the topk gate coefficients pre-folded into the intermediate on
// the K side (3.5x cheaper than the pack's N-side pass; mathematically
// identical). Removes the pack kernel's full M x N read+write round-trip and
// one gating hop per wave (the pack degenerates to a 1-effective-CTA flag
// relay). Requires msplit. Own never-mix boundary.
int
get_a2av_rs_fused_pack() {
  (void)bytedance::flux::get_int_from_env("FLUX_A2AV_RS_FUSED_PACK_TAG", 0);
  (void)bytedance::flux::get_int_from_env("FLUX_A2AV_SLIPSTREAM2_TAG", 0);
  static int v = bytedance::flux::get_int_from_env("FLUX_A2AV_RS_FUSED_PACK", 1);
  return v;
}
// OWN_WAVE=first (gen-9 receiver study, 2026-08-24): compute the OWN-node
// wave FIRST instead of last. Costs ~1/NN of the GEMM in wire-start delay;
// buys the receiver early own-node contributions, so tokens complete at
// their last REMOTE arrival (spread over the drain) instead of at the final
// own-node fold — the structural precondition for any arrival-order
// receiver (lanechain/ordered) to beat wait-all. 0 = own last (canon).
int
get_a2av_rs_own_wave_first() {
  (void)bytedance::flux::get_int_from_env("FLUX_A2AV_RS_OWN_FIRST_TAG", 0);
  static int v = bytedance::flux::get_int_from_env("FLUX_A2AV_RS_OWN_FIRST", 0);
  return v;
}
// Completion-bucketed register receiver (Slipstream gen-10, 2026-08-24):
// arrival-order folding at wait-all's 1x bytes. Tokens bucket by the chain
// position of their LAST contribution's lane (own-node lanes chained FIRST
// for CONSUMPTION only -- the wire keeps the canon own-last production
// order); each lane wait releases a register-CSR fold of exactly the tokens
// it completes. No scratch RMW (lane-chain's 4-5x amplification), no
// finalize; the exposed tail is one bucket. Requires compress C' and
// n_split == 1; exclusive with LANE_CHAIN. Own never-mix boundary.
// Supersession-default resolution helper (SCHEMA rule 13): a knob that is ON
// by DEFAULT must silently demote in configs it cannot run in (the v1 ns2
// ablation arm, the lane-chain diagnostics); an EXPLICIT env=1 keeps the loud
// FLUX_CHECK contract.
static bool
a2av_env_explicit(const char *key) {
  return std::getenv(key) != nullptr;
}
int
get_a2av_rs_bucket() {
  (void)bytedance::flux::get_int_from_env("FLUX_A2AV_RS_BUCKET_TAG", 0);
  (void)bytedance::flux::get_int_from_env("FLUX_A2AV_SLIPSTREAM2_TAG", 0);
  static int v = bytedance::flux::get_int_from_env("FLUX_A2AV_RS_BUCKET", 1);
  return v;
}
// Arrival-dynamic bucket receiver (H4, 2026-08-25): replace the bucket
// receiver's S = L + NN - 1 sequential front-end waits (EXPECTED arrival
// order -- one inverted arrival head-of-line blocks every already-landed
// lane behind it; at 16n the tail is the max of 15 remote arrivals) with ONE
// persistent kernel on the reserved reduce SMs that claims row chunks of any
// ARRIVED lane and folds each token at its true completion (per-token
// outstanding-contribution counters). Fold arithmetic is the bucket fold's
// (fp32 CSR-order accumulate, read once / written once), so the output stays
// BITWISE-identical to the wait-all reduce; only the schedule is
// arrival-dependent. Requires the bucket receiver's plan (compress C',
// n_split == 1). Default OFF -- knob-off is bit-identical enqueue-for-enqueue
// to the pre-knob binary. Own never-mix boundary.
// v2 (2026-08-25): slack-row sentinel hardening — token_of -1-fill + in-kernel
// range guard closes the 16n b32+ garbage-t OOB-atomicSub livelock; tag bumped
// to V2 so the runner can tell fixed builds from v1 (v1 dyn arms never-mix).
int
get_a2av_rs_recv_dyn() {
  (void)bytedance::flux::get_int_from_env("FLUX_A2AV_RS_RECV_DYN_V2_TAG", 0);
  static int v = bytedance::flux::get_int_from_env("FLUX_A2AV_RS_RECV_DYN", 0);
  return v;
}
// Rows claimed per cursor bump in the arrival-dynamic receiver (fold spread
// vs cursor-atomic traffic dial; folds trigger inside row processing, so a
// smaller chunk spreads token folds across more warps).
int
get_a2av_rs_recv_dyn_chunk() {
  static int v =
      std::max(1, bytedance::flux::get_int_from_env("FLUX_A2AV_RS_RECV_DYN_CHUNK", 4));
  return v;
}
// Debug watchdog for the combine's device spin loops: 0 (default) =
// unlimited, historical behavior. >0 = trap after N no-progress sleeps
// (~200ns each) so a missing-signal bug aborts loudly instead of hanging
// (the 2026-08-17 epic_l01_hc_m4 b2 hang class).
uint64_t
get_a2av_spin_limit() {
  static uint64_t v =
      (uint64_t)bytedance::flux::get_int_from_env("FLUX_A2AV_RS_SPIN_LIMIT", 0);
  return v;
}
}  // namespace

namespace bytedance::flux::ths_op {

using torch::Tensor;

// The two a2av_hier routing-plan index builders are free functions so both
// the gather-rs op and the standalone TopkReduceScatterOp combine entry can
// self-build absent indices on the timed critical path (v2b in-window).
namespace {
  // Builds the mirror-layout gather indices for the a2av_hier combine, sharing
  // layer0 a2av's exact ordering contract (same (.., expert, dst_row) keys, same
  // copy-index tie-break):
  // - pack_index [M_this_ep]: send-panel row -> gemm row. The send panel is
  //   (home_rank, expert, copy)-ordered == layer0's recv layout on this rank, so
  //   this is the inverse of layer0's sorted_gather_index arithmetic identity --
  //   derived from the same offA/cumA/offR_of_A host tables, NO sort.
  // - reduce_index [cpr]: local copy (t_local * topk + j) -> recv-panel row. The
  //   recv panel is (owner_rank, expert, copy)-ordered == layer0's send-buffer
  //   layout on this rank, i.e. globally (expert, copy)-sorted -- the inverse of
  //   layer0's pack permutation: ONE argsort of the layer0 pack key + a scatter.
  // A fused layer0+layer1 pipeline can pass layer0's tensors instead and pay the
  // index math once (see the forward kwargs).
  std::pair<torch::Tensor, torch::Tensor>
  build_a2av_combine_indices(
      torch::Tensor const &routing_idx,
      torch::Tensor const &splits_gpu,
      torch::Tensor const &splits_per_source,
      int64_t M_this_ep,
      int64_t m_full,
      int world_size,
      int rank,
      int64_t total_num_experts,
      int64_t ep_nexperts,   // experts per owner rank (tp == 1)
      int64_t ep_start) {
    const int W = world_size;
    const int64_t nex = total_num_experts;
    const int64_t E_loc = ep_nexperts;  // experts per owner rank (tp == 1)
    const int64_t nexG = E_loc * W;
    const int64_t cpr = m_full / W;
    const int32_t *cnt = splits_per_source.data_ptr<int32_t>();
    auto cnt_at = [&](int h, int64_t e) -> int64_t { return cnt[h * nex + e]; };
    auto opt_i64 = torch::TensorOptions(torch::kCUDA).dtype(torch::kLong);

    // host tables -- formulas identical to layer0's metadata collapse (dispatch
    // source == combine home, both index cnt rows). A-order groups g = e_loc*W+h.
    torch::Tensor tables = torch::empty({3, nexG}, torch::TensorOptions().dtype(torch::kLong));
    int64_t *cumA_h = tables[0].data_ptr<int64_t>();
    int64_t *offA_h = tables[1].data_ptr<int64_t>();
    int64_t *offR_of_A_h = tables[2].data_ptr<int64_t>();
    int64_t acc = 0;
    for (int64_t e_loc = 0; e_loc < E_loc; e_loc++) {
      for (int h = 0; h < W; h++) {
        int64_t g = e_loc * W + h;
        offA_h[g] = acc;
        acc += cnt_at(h, ep_start + e_loc);
        cumA_h[g] = acc;
      }
    }
    FLUX_CHECK_EQ(acc, M_this_ep) << "splits_per_source disagrees with gemm rows";
    std::vector<int64_t> offR((size_t)nexG, 0);
    acc = 0;
    for (int h = 0; h < W; h++) {
      for (int64_t e_loc = 0; e_loc < E_loc; e_loc++) {
        offR[h * E_loc + e_loc] = acc;
        acc += cnt_at(h, ep_start + e_loc);
      }
    }
    for (int64_t e_loc = 0; e_loc < E_loc; e_loc++) {
      for (int h = 0; h < W; h++) {
        offR_of_A_h[e_loc * W + h] = offR[h * E_loc + e_loc];
      }
    }
    auto tables_dev = tables.to(torch::kCUDA);
    auto cumA = tables_dev[0];
    auto offA = tables_dev[1];
    auto offR_of_A = tables_dev[2];

    torch::Tensor pack_index;
    auto iota = torch::arange(M_this_ep, opt_i64);
    if (M_this_ep > 0) {
      auto g = torch::searchsorted(cumA, iota, /*out_int32=*/false, /*right=*/true)
                   .clamp_max_(nexG - 1);
      auto sgi = offR_of_A.index_select(0, g) + iota - offA.index_select(0, g);
      pack_index =
          torch::empty({M_this_ep}, opt_i64).scatter_(0, sgi, iota).to(torch::kInt);
    } else {
      pack_index = torch::empty({0}, torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));
    }

    // reduce index: my copies sorted by (expert, copy index) == recv-panel
    // order. SORT-FREE since 2026-08-22: the rank of a local copy among my
    // copies in (e, copy) order is pure scd arithmetic — my exclusive
    // per-expert prefix + the copy's rank inside its (e, home==me) A-order
    // sub-block (scd - expert_base - home_base(e, me)). Same identity family
    // as the pack side above; bitwise == the old argsort (unique keys).
    auto routing_slice =
        routing_idx.narrow(0, (int64_t)rank * cpr, cpr).to(torch::kLong);
    auto splits_cum = splits_gpu.to(torch::kLong).cumsum(0);
    auto e_of = torch::searchsorted(splits_cum, routing_slice, /*out_int32=*/false, /*right=*/true)
                    .clamp_max_(nex - 1);
    torch::Tensor rtab = torch::empty({3, nex}, torch::TensorOptions().dtype(torch::kLong));
    {
      int64_t *my_cum = rtab[0].data_ptr<int64_t>();
      int64_t *e_base = rtab[1].data_ptr<int64_t>();
      int64_t *h_base = rtab[2].data_ptr<int64_t>();
      int64_t acc_my = 0;
      int64_t acc_e = 0;
      for (int64_t e = 0; e < nex; e++) {
        my_cum[e] = acc_my;
        e_base[e] = acc_e;
        int64_t hb = 0;
        for (int h = 0; h < W; h++) {
          if (h == rank) {
            h_base[e] = hb;
          }
          hb += cnt_at(h, e);
        }
        acc_my += cnt_at(rank, e);
        acc_e += hb;
      }
    }
    auto rtab_dev = rtab.to(torch::kCUDA, /*non_blocking=*/true);
    auto reduce_index = (rtab_dev[0].index_select(0, e_of) + routing_slice -
                         rtab_dev[1].index_select(0, e_of) - rtab_dev[2].index_select(0, e_of))
                            .to(torch::kInt);

    static const bool kCheckIdentity =
        get_int_from_env("FLUX_A2AV_RS_CHECK_IDENTITY", 0) != 0;
    if (kCheckIdentity && M_this_ep > 0) {
      // brute-force reference for the arithmetic pack identity: recover each gemm
      // row's global copy index from routing_idx, sort rows by (home, row)
      int64_t ep_m_start = 0;
      for (int64_t e = 0; e < ep_start; e++) {
        for (int h = 0; h < W; h++) {
          ep_m_start += cnt_at(h, e);
        }
      }
      auto iota_m = torch::arange((int64_t)m_full, opt_i64);
      auto copy_of_row = torch::empty({(int64_t)m_full}, opt_i64)
                             .scatter_(0, routing_idx.to(torch::kLong), iota_m)
                             .narrow(0, ep_m_start, M_this_ep);
      auto h_of = copy_of_row.div(cpr, "floor");
      auto perm_ref = (h_of * M_this_ep + iota).argsort();
      FLUX_CHECK(torch::equal(pack_index, perm_ref.to(torch::kInt)))
          << "a2av_hier pack-index identity mismatch";
    }
    return {pack_index, reduce_index};
  }

  // Compress (dedup) metadata for the a2av_hier combine. Executable spec:
  // test_a2av_combine_sim.py::simulate_compress. One partial per (token,
  // source node) crosses the wire; source rank (n, lr) owns all wire rows to
  // rank (tn, lr) (same-lr end-to-end); the node's copies converge on it and
  // are merged per token before the put. Everything below is derived from the
  // globally replicated routing metadata, no exchange.
  //
  // Outputs (int32 CUDA):
  // - wire_ptr [wire_rows + 1], wire_copy [conv_rows]: source-side CSR, wire
  //   row -> contributing conv-panel rows. Conv panel on this rank: segments
  //   ordered (dest node tn ascending SKIPPING own, source local rank ls
  //   ascending), each holding peer (n, ls)'s send-panel slice for dest rank
  //   (tn, my_lr) in the peer's (expert, copy) order. Wire rows are
  //   token-ascending per tn segment; per-segment row count must equal the
  //   transposed dedup count U[(tn, my_lr)][n] (FLUX_CHECKed).
  // - red_ptr [ntokens_local + 1], red_row [red_total]: destination-side CSR,
  //   local token -> its recv-panel rows under the compress chunk matrix C'
  //   (own-node lanes keep per-rank chunks; the remote lane materializes only
  //   at the same-lr source rank with U[me][m] rows). Per token: own-node
  //   copies individually in copy-j order, then one merged row per
  //   contributing remote node ascending. Remote merged row position is the
  //   transposed one-cumsum: exclusive count of earlier home tokens with a
  //   copy on that node.
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  build_a2av_compress_indices(
      torch::Tensor const &routing_idx,
      torch::Tensor const &splits_gpu,
      torch::Tensor const &splits_per_source,  // CPU int32 [W, nex]
      torch::Tensor const &unique_counts,      // CPU int32 [W, NN]: U[home][src_node]
      int64_t m_full,
      int world_size,
      int nnodes,
      int local_world_size,
      int rank,
      int64_t total_num_experts,
      int64_t ep_nexperts,   // experts per owner rank (tp == 1)
      int topk) {
    const int W = world_size;
    const int NN = nnodes;
    const int L = local_world_size;
    const int my_node = rank / L;
    const int my_lr = rank % L;
    const int64_t nex = total_num_experts;
    const int64_t E_loc = ep_nexperts;
    const int64_t cpr = m_full / W;
    const int64_t ntok_local = cpr / topk;
    const int64_t ntokens = m_full / topk;
    auto opt_i64 = torch::TensorOptions(torch::kCUDA).dtype(torch::kLong);
    auto opt_i32 = torch::TensorOptions(torch::kCUDA).dtype(torch::kInt);
    constexpr int64_t kMax64 = std::numeric_limits<int64_t>::max();

    FLUX_CHECK(unique_counts.device().is_cpu()) << "a2av_unique_counts must be CPU";
    CHECK_2D(unique_counts, W, NN);
    FLUX_CHECK(unique_counts.is_contiguous());
    FLUX_CHECK(unique_counts.scalar_type() == at::ScalarType::Int);
    const int32_t *U = unique_counts.data_ptr<int32_t>();
    const int32_t *cnt = splits_per_source.data_ptr<int32_t>();

    // host chunk matrix C[s][d] and the C/C' recv prefixes of MY column
    std::vector<int64_t> C((size_t)W * W, 0);
    for (int s = 0; s < W; s++) {
      for (int d = 0; d < W; d++) {
        int64_t acc = 0;
        for (int64_t e = s * E_loc; e < (s + 1) * E_loc; e++) {
          acc += cnt[d * nex + e];
        }
        C[s * W + d] = acc;
      }
    }
    torch::Tensor lane_tables = torch::empty({2, W}, torch::TensorOptions().dtype(torch::kLong));
    int64_t *recv_off_C = lane_tables[0].data_ptr<int64_t>();
    int64_t *recv_off_Cp = lane_tables[1].data_ptr<int64_t>();
    int64_t own_total = 0;
    for (int s = 0, accC = 0, accCp = 0; s < W; s++) {
      recv_off_C[s] = accC;
      recv_off_Cp[s] = accCp;
      accC += C[s * W + rank];
      const bool same_node = s / L == my_node;
      if (same_node) {
        accCp += C[s * W + rank];
        own_total += C[s * W + rank];
      } else if (s % L == my_lr) {
        accCp += U[rank * NN + s / L];
      }
    }
    auto lane_dev = lane_tables.to(torch::kCUDA);
    auto recv_off_C_dev = lane_dev[0];
    auto recv_off_Cp_dev = lane_dev[1];

    // ---- global per-copy attributes (every rank holds the full metadata)
    auto iota_m = torch::arange(m_full, opt_i64);
    auto splits_cum = splits_gpu.to(torch::kLong).cumsum(0);
    auto e_of_copy =
        torch::searchsorted(splits_cum, routing_idx.to(torch::kLong), false, /*right=*/true)
            .clamp_max_(nex - 1);
    auto owner = e_of_copy.div(E_loc, "floor");
    auto home = iota_m.div(cpr, "floor");

    // ---- source side: conv-panel order + wire CSR
    torch::Tensor wire_ptr, wire_copy;
    int64_t conv_total = 0, wire_total = 0;
    for (int tn = 0; tn < NN; tn++) {
      if (tn == my_node) {
        continue;
      }
      wire_total += U[(tn * L + my_lr) * NN + my_node];
      for (int ls = 0; ls < L; ls++) {
        conv_total += C[(my_node * L + ls) * W + (tn * L + my_lr)];
      }
    }
    if (conv_total > 0) {
      auto owner_node = owner.div((int64_t)L, "floor");
      auto home_node = home.div((int64_t)L, "floor");
      auto conv_mask = (owner_node == my_node) & (home_node != my_node) &
                       (home - home_node * L == my_lr);
      auto seg = home_node - (home_node > my_node).to(torch::kLong);
      auto ls = owner - owner_node * L;
      auto conv_key = (((seg * L + ls) * nex + e_of_copy) * m_full + iota_m)
                          .masked_fill_(~conv_mask, kMax64);
      auto conv_copy = conv_key.argsort().narrow(0, 0, conv_total);
      // wire grouping: (segment, token), token-ascending inside each segment
      auto wkey = seg.index_select(0, conv_copy) * ntokens +
                  conv_copy.div((int64_t)topk, "floor");
      auto worder = wkey.argsort(/*stable=*/true, /*dim=*/-1, /*descending=*/false);
      wire_copy = worder.to(torch::kInt);
      auto counts = std::get<2>(torch::unique_consecutive(
          wkey.index_select(0, worder), /*return_inverse=*/false, /*return_counts=*/true));
      FLUX_CHECK_EQ(counts.numel(), wire_total)
          << "compress wire rows disagree with a2av_unique_counts (transposed U)";
      wire_ptr = torch::cat({torch::zeros({1}, opt_i64), counts.cumsum(0)}).to(torch::kInt);
    } else {
      // conv_total == 0 mathematically implies wire_total == 0; assert it so
      // an inconsistent externally-supplied U aborts here instead of driving
      // the pre-reduce kernel past wire_ptr's single element (2026-08-17
      // hardening, epic combine-only entry).
      FLUX_CHECK_EQ(wire_total, 0)
          << "compress: conv_total == 0 but a2av_unique_counts claims "
          << wire_total << " wire rows (inconsistent transposed U)";
      wire_ptr = torch::zeros({1}, opt_i32);
      wire_copy = torch::empty({0}, opt_i32);
    }

    // ---- destination side: red CSR under C'
    auto iota_c = torch::arange(cpr, opt_i64);
    auto e_my = e_of_copy.narrow(0, (int64_t)rank * cpr, cpr);
    auto owner_my = owner.narrow(0, (int64_t)rank * cpr, cpr);
    // today's recv row (C layout) via the reduce-index formula, then C' remap:
    // intra-lane position is preserved, only the lane base changes
    auto perm = (e_my * cpr + iota_c).argsort();
    auto rows_C = torch::empty({cpr}, opt_i64).scatter_(0, perm, iota_c);
    auto rows_Cp = rows_C - recv_off_C_dev.index_select(0, owner_my) +
                   recv_off_Cp_dev.index_select(0, owner_my);
    const int64_t K = (int64_t)topk + NN + 1;  // per-token entry order key base
    auto tl = iota_c.div((int64_t)topk, "floor");
    auto own_mask = owner_my.div((int64_t)L, "floor") == my_node;
    auto key_own =
        (tl * K + (iota_c - tl * topk)).masked_fill_(~own_mask, kMax64);
    auto ord_own = key_own.argsort();
    auto own_rows = rows_Cp.index_select(0, ord_own).narrow(0, 0, own_total);
    auto own_keys = key_own.index_select(0, ord_own).narrow(0, 0, own_total);
    // remote merged rows: flags [ntok_local, NN] -> per-column exclusive cumsum
    int64_t rem_total = 0;
    torch::Tensor rem_base = torch::zeros({NN}, torch::TensorOptions().dtype(torch::kLong));
    int64_t *rem_base_p = rem_base.data_ptr<int64_t>();
    for (int m = 0; m < NN; m++) {
      if (m == my_node) {
        continue;
      }
      rem_base_p[m] = recv_off_Cp[m * L + my_lr];
      rem_total += U[rank * NN + m];
    }
    torch::Tensor red_ptr, red_row;
    auto onode = owner_my.div((int64_t)L, "floor");
    auto flags = torch::zeros({ntok_local * NN}, opt_i64)
                     .scatter_(0, tl * NN + onode, 1)
                     .view({ntok_local, (int64_t)NN});
    flags.select(1, my_node).zero_();
    auto pos = flags.cumsum(0) - flags;
    auto rem_rows2d = pos + rem_base.to(torch::kCUDA).view({1, (int64_t)NN});
    auto tl_col = torch::arange(ntok_local, opt_i64).view({ntok_local, 1});
    auto m_row = torch::arange((int64_t)NN, opt_i64).view({1, (int64_t)NN});
    auto key_rem = (tl_col * K + topk + m_row)
                       .masked_fill_(flags.eq(0), kMax64)
                       .reshape({-1});
    auto ord_rem = key_rem.argsort();
    auto rem_rows = rem_rows2d.reshape({-1}).index_select(0, ord_rem).narrow(0, 0, rem_total);
    auto rem_keys = key_rem.index_select(0, ord_rem).narrow(0, 0, rem_total);
    auto keys_all = torch::cat({own_keys, rem_keys});
    auto vals_all = torch::cat({own_rows, rem_rows});
    auto ord = keys_all.argsort();
    red_row = vals_all.index_select(0, ord).to(torch::kInt);
    auto tl_sorted = keys_all.index_select(0, ord).div(K, "floor");
    red_ptr = torch::cat(
                  {torch::zeros({1}, opt_i64),
                   torch::bincount(tl_sorted, {}, ntok_local).cumsum(0)})
                  .to(torch::kInt);

    static const bool kCheckIdentity =
        get_int_from_env("FLUX_A2AV_RS_CHECK_IDENTITY", 0) != 0;
    if (kCheckIdentity) {
      // every red entry lands inside the C' recv image, every token has at
      // least one contribution, totals match the host tables
      FLUX_CHECK_EQ(own_total + rem_total, red_row.size(0));
      if (red_row.size(0) > 0) {
        FLUX_CHECK_GE(red_row.min().item<int64_t>(), 0);
        int64_t cpr_cp = 0;
        for (int s = 0; s < W; s++) {
          cpr_cp += (s / L == my_node)               ? C[s * W + rank]
                    : (s % L == my_lr)               ? U[rank * NN + s / L]
                                                     : 0;
        }
        FLUX_CHECK_LT(red_row.max().item<int64_t>(), cpr_cp);
      }
    }
    return {wire_ptr, wire_copy, red_ptr, red_row};
  }

  // Sort-free compress-plan builder (2026-08-21): identical outputs to
  // build_a2av_compress_indices above, but every ordering is arithmetic on
  // the layer0 stable scatter_index + host cnt/U prefix tables — 4 kernels
  // (a2av_compress_plan), no radix sorts, deterministic. The sort-based
  // sibling stays as the FLUX_A2AV_RS_CHECK_IDENTITY reference.
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  build_a2av_compress_indices_fast(
      torch::Tensor const &routing_idx,
      torch::Tensor const &splits_gpu,
      torch::Tensor const &splits_per_source,  // CPU int32 [W, nex]
      torch::Tensor const &unique_counts,      // CPU int32 [W, NN]
      int64_t m_full,
      int world_size,
      int nnodes,
      int local_world_size,
      int rank,
      int64_t total_num_experts,
      int64_t ep_nexperts,
      int topk) {
    const int W = world_size;
    const int NN = nnodes;
    const int L = local_world_size;
    const int my_node = rank / L;
    const int my_lr = rank % L;
    const int64_t nex = total_num_experts;
    const int64_t E_loc = ep_nexperts;
    const int64_t cpr = m_full / W;
    const int64_t ntok_local = cpr / topk;
    const int64_t tpr = ntok_local;  // tokens per rank (uniform homing)
    auto opt_i32 = torch::TensorOptions(torch::kCUDA).dtype(torch::kInt);
    const int32_t *U = unique_counts.data_ptr<int32_t>();
    const int32_t *cnt = splits_per_source.data_ptr<int32_t>();

    // ---- host tables (all from cnt/U; no device sync) ----
    std::vector<int64_t> C((size_t)W * W, 0);
    for (int s = 0; s < W; s++) {
      for (int d = 0; d < W; d++) {
        int64_t acc = 0;
        for (int64_t e = s * E_loc; e < (s + 1) * E_loc; e++) {
          acc += cnt[d * nex + e];
        }
        C[s * W + d] = acc;
      }
    }
    const int64_t n_i64 = nex /*expert_base*/ + nex /*my_cnt_cum*/ +
                          (int64_t)(NN - 1) * L * E_loc /*conv_base*/ + W /*recv_off_C*/ +
                          W /*recv_off_Cp*/ + NN /*rem_base*/;
    torch::Tensor t64 = torch::empty({n_i64}, torch::TensorOptions().dtype(torch::kLong));
    int64_t *p64 = t64.data_ptr<int64_t>();
    int64_t *expert_base = p64;
    int64_t *my_cnt_cum = expert_base + nex;
    int64_t *conv_base = my_cnt_cum + nex;
    int64_t *recv_off_C = conv_base + (int64_t)(NN - 1) * L * E_loc;
    int64_t *recv_off_Cp = recv_off_C + W;
    int64_t *rem_base = recv_off_Cp + W;
    torch::Tensor t32 = torch::empty({nex * W}, torch::TensorOptions().dtype(torch::kInt));
    int32_t *home_base = t32.data_ptr<int32_t>();
    {
      int64_t acc_e = 0;
      int64_t acc_my = 0;
      for (int64_t e = 0; e < nex; e++) {
        expert_base[e] = acc_e;
        my_cnt_cum[e] = acc_my;
        int64_t hb = 0;
        for (int h = 0; h < W; h++) {
          home_base[e * W + h] = (int32_t)hb;
          hb += cnt[h * nex + e];
        }
        acc_e += hb;
        acc_my += cnt[rank * nex + e];
      }
    }
    int64_t own_total = 0;
    {
      int64_t accC = 0;
      int64_t accCp = 0;
      for (int s = 0; s < W; s++) {
        recv_off_C[s] = accC;
        recv_off_Cp[s] = accCp;
        accC += C[s * W + rank];
        if (s / L == my_node) {
          accCp += C[s * W + rank];
          own_total += C[s * W + rank];
        } else if (s % L == my_lr) {
          accCp += U[rank * NN + s / L];
        }
      }
    }
    int64_t conv_total = 0;
    int64_t wire_total = 0;
    {
      const int64_t node_e0 = (int64_t)my_node * L * E_loc;
      int64_t acc = 0;
      for (int tn = 0, seg = 0; tn < NN; tn++) {
        if (tn == my_node) {
          continue;
        }
        const int h = tn * L + my_lr;
        wire_total += U[h * NN + my_node];
        for (int64_t j = 0; j < L * E_loc; j++) {
          conv_base[(int64_t)seg * L * E_loc + j] = acc;
          acc += cnt[h * nex + node_e0 + j];
        }
        seg++;
      }
      conv_total = acc;
    }
    int64_t rem_total = 0;
    for (int m = 0; m < NN; m++) {
      rem_base[m] = 0;
      if (m == my_node) {
        continue;
      }
      rem_base[m] = recv_off_Cp[m * L + my_lr];
      rem_total += U[rank * NN + m];
    }
    if (conv_total == 0) {
      FLUX_CHECK_EQ(wire_total, 0)
          << "compress: conv_total == 0 but a2av_unique_counts claims " << wire_total
          << " wire rows (inconsistent transposed U)";
    }

    // ---- device inputs / scratch / outputs ----
    auto splits_cum = splits_gpu.to(torch::kLong).cumsum(0);
    auto e_of = torch::searchsorted(
                    splits_cum, routing_idx.to(torch::kLong), false, /*right=*/true)
                    .clamp_max_(nex - 1)
                    .to(torch::kInt);
    auto t64_dev = t64.to(torch::kCUDA, /*non_blocking=*/true);
    auto t32_dev = t32.to(torch::kCUDA, /*non_blocking=*/true);
    const int64_t seg_tokens = (int64_t)(NN - 1) * tpr;
    auto scratch = torch::empty({2 * seg_tokens + 2 * tpr * NN}, opt_i32);
    auto wire_ptr = torch::empty({wire_total + 1}, opt_i32);
    auto wire_copy = torch::empty({conv_total}, opt_i32);
    auto red_ptr = torch::empty({ntok_local + 1}, opt_i32);
    auto red_row = torch::empty({own_total + rem_total}, opt_i32);

    int64_t *d64 = t64_dev.data_ptr<int64_t>();
    int32_t *scr = scratch.data_ptr<int32_t>();
    A2AVCompressPlanArguments args{
        .scatter_index = routing_idx.data_ptr<int32_t>(),
        .e_of_copy = e_of.data_ptr<int32_t>(),
        .home_base = t32_dev.data_ptr<int32_t>(),
        .expert_base = d64,
        .conv_base = d64 + 2 * nex,
        .my_cnt_cum = d64 + nex,
        .recv_off_C = d64 + 2 * nex + (int64_t)(NN - 1) * L * E_loc,
        .recv_off_Cp = d64 + 2 * nex + (int64_t)(NN - 1) * L * E_loc + W,
        .rem_base = d64 + 2 * nex + (int64_t)(NN - 1) * L * E_loc + 2 * W,
        .conv_count = scr,
        .wire_row_of = scr + seg_tokens,
        .red_flags = scr + 2 * seg_tokens,
        .rem_pos = scr + 2 * seg_tokens + tpr * NN,
        .wire_ptr = wire_ptr.data_ptr<int32_t>(),
        .wire_copy = wire_copy.data_ptr<int32_t>(),
        .red_ptr = red_ptr.data_ptr<int32_t>(),
        .red_row = red_row.data_ptr<int32_t>(),
        .m_full = m_full,
        .topk = topk,
        .world_size = W,
        .nnodes = NN,
        .local_world_size = L,
        .rank = rank,
        .nexperts = nex,
        .ep_nexperts = E_loc};
    a2av_compress_plan(args, c10::cuda::getCurrentCUDAStream());
    return {wire_ptr, wire_copy, red_ptr, red_row};
  }
}  // namespace

class TopkReduceScatterOp::TopkReduceScatterOpImpl {
 private:
  std::shared_ptr<Group> tp_group;
  int32_t rank;
  int32_t world_size;  // the total world size
  const int nnodes;
  const int node_idx;
  const int local_rank;
  const int local_world_size;
  int32_t max_m;
  int32_t n_dim;
  int32_t topk;
  at::ScalarType output_dtype;
  const int ep_nexperts;
  const int ep_world_size;  // the world size of expert parallel
  const bool do_all_reduce;
  const bool use_read_mode;
  const int n_split;

  // intra-node buffers: tensor lists / pointer arrays are [local_world_size],
  // indexed by local rank (== global rank when nnodes == 1)
  torch::Tensor reduce_buffer;
  std::vector<torch::Tensor> reduce_buffers;
  torch::Tensor reduce_buffer_dptrs;
  torch::Tensor tile_barrier;
  std::vector<torch::Tensor> tile_barriers;
  torch::Tensor tile_barrier_dptrs;
  torch::Tensor barrier;
  std::vector<torch::Tensor> barriers;
  int **barrier_dev_ptrs = nullptr;

  // inter-node staging (nnodes > 1 only): symmetric-heap send/recv buffers, one
  // [staging_rows, n/n_split] slot per (node, split); host-issued putmem_signal
  // moves each finished slot to the peer with the same local rank on the owner node
  torch::Tensor staging_send;
  torch::Tensor staging_recv;
  torch::Tensor internode_signals;  // [nnodes * n_split] uint64 signal targets
  // cuStreamWriteValue/WaitValue32 need real device addresses, so use
  // cutlass::DeviceAllocation, not torch tensors (expandable_segments VA issue)
  cutlass::DeviceAllocation<int> group_flags;     // [nnodes * n_split]
  cutlass::DeviceAllocation<int> group_counters;  // [nnodes * n_split]
  c10::cuda::CUDAStream internode_stream;
  // FLUX_A2AV_RS_WIRE_STREAMS=2 (2026-08-23 M4-C10): the compress wire
  // ladder's blocking puts for (sid, tn) cells are pairwise independent
  // (distinct wire-panel segments, destinations, and per-dest recv_sig
  // copies) yet serialize on one stream; parity-split them over a second
  // internode stream. DEFAULT 2 since the Slipstream canonicalization
  // (2026-08-23); =1 is the single-stream ablation. At conn=1 the split
  // preserves the ladder's enqueue order exactly (still an executable
  // schedule), it just buys nothing.
  int rs_wire_streams_ = 1;
  std::vector<c10::cuda::CUDAStream> internode_streams2_;   // rs_wire_streams_ - 1 extras
  cudaEvent_t a2av_inter2_done_ = nullptr;
  cudaEvent_t staging_reset_event;
  uint64_t run_id_ = 0;
  // M-split waves (Slipstream v2): per-iteration pack-gate state, armed by
  // set_msplit_waves() before each run(); 0 = legacy single-split gate
  int msplit_n_waves_ = 0;
  int msplit_wave_of_node_[kA2AVMaxNodes] = {};
  int msplit_node_order_[kA2AVMaxNodes] = {};

  // a2av_hier combine state. Layouts mirror layer0's a2av dispatch exactly:
  // the send panel is (home_rank, expert, copy)-ordered (== layer0's recv
  // layout), the recv panel is (owner_rank, expert, copy)-ordered (== layer0's
  // send layout), so every copy lands back at its layer0 pack position and the
  // pack/reduce gather indices are the inverses of layer0's index math.
  const bool a2av_hier;
  // FLUX_A2AV_RS_EAGER: replace the per-split host wait-all-W reduce gates with
  // one persistent arrival-order reduce kernel (variant-selection ctor boolean,
  // knob-off leaves the shipped schedule untouched)
  const bool a2av_eager_;
  // a2av_hier_compress: one partial per (token, source node) crosses the wire.
  // Source rank (n, lr) owns all wire rows to rank (tn, lr): the node's copies
  // converge on it (conv panel, NVLink), a persistent pre-reduce kernel merges
  // them per token into the wire panel, and the inter ladder puts straight into
  // the destination's recv panel (C' image) -- no destination gateway hop.
  // False when nnodes == 1 (degrades to plain a2av_hier: zero wire savings).
  const bool a2av_compress_;
  // FLUX_A2AV_RS_LANE_CHAIN: per-lane chained receiver in expected arrival
  // order (supersedes eager/legacy receive paths when set; compress + ns1)
  const bool a2av_lane_chain_;
  // FLUX_A2AV_RS_BUCKET (gen-10): completion-bucketed register receiver --
  // arrival-order folding at 1x bytes (exclusive with lane-chain)
  const bool a2av_bucket_;
  // FLUX_A2AV_RS_RECV_DYN (H4): consume the bucket receiver's lanes in TRUE
  // arrival order via one persistent kernel (per-token completion counters)
  // instead of the expected-order host wait chain; bitwise-identical output
  const bool a2av_recv_dyn_;
  // FLUX_A2AV_RS_FUSED_PACK (gen-8c): the GEMM scatters the send panel
  // directly; the pack kernel runs as a flag relay and applies NO vec_scale
  // (pre-folded into the intermediate by the caller)
  const bool a2av_fused_pack_;
  int64_t a2av_send_rows_ = 0;   // send panel row capacity per split (routing-dependent load)
  int64_t a2av_recv_rows_ = 0;   // recv panel rows per split: exactly max_m / world_size
  int64_t a2av_stage_rows_ = 0;  // gateway staging row capacity per split
  int64_t a2av_conv_rows_ = 0;   // compress: convergence panel row capacity per split
  int64_t a2av_wire_rows_ = 0;   // compress: wire panel row capacity per split
  torch::Tensor a2av_send_panel_;       // [n_split, a2av_send_rows_, n_per] symmetric
  torch::Tensor a2av_recv_panel_;       // [n_split, a2av_recv_rows_, n_per] symmetric
  torch::Tensor a2av_stage_panel_;      // [n_split, a2av_stage_rows_, n_per] symmetric (nnodes>1)
  torch::Tensor a2av_conv_panel_;       // compress: [n_split, conv_rows, n_per] symmetric
  torch::Tensor a2av_wire_panel_;       // compress: [n_split, wire_rows, n_per] symmetric
  torch::Tensor a2av_recv_signals_;     // uint64 [world_size * n_split], epoch, never reset
  torch::Tensor a2av_arrival_signals_;  // uint64 [nnodes * n_split], epoch, never reset
  torch::Tensor a2av_conv_signals_;     // compress: uint64 [L * NN * n_split], epoch, never reset
  // compress: pre-reduce kernel -> host wire-ready flags, memset per run under
  // the staging_reset_event discipline (same as group_flags)
  cutlass::DeviceAllocation<int> wire_flags_;     // [nnodes * n_split]
  cutlass::DeviceAllocation<int> wire_counters_;  // [nnodes * n_split]
  // lane-chain receiver buffers (a2av_lane_chain_ only; plain device memory)
  torch::Tensor a2av_scratch_fp32_;   // [max ntokens_local, n] fp32 accumulator
  torch::Tensor a2av_token_of_row_;   // [a2av_recv_rows_] int32 recv row -> token
  // bucket receiver buffers (a2av_bucket_ only): plain device memory for the
  // per-iteration completion map, plus a run-parity double-buffered pinned ->
  // device pair for the tiny lane tables (lane_off [W+1] | chain_pos [W])
  torch::Tensor a2av_bucket_comp_;    // [ntok_max] int32 completion position
  torch::Tensor a2av_bucket_tok_;     // [ntok_max] int32 tokens by bucket
  torch::Tensor a2av_bucket_meta_;    // [3*kA2AVMaxWorld + 2] int32: cnt | cur | ptr
  torch::Tensor a2av_bucket_lanes_h_; // pinned [2 * 2*(W+1)] int32 (parity halves)
  torch::Tensor a2av_bucket_lanes_d_; // device twin
  std::optional<c10::cuda::CUDAStream> a2av_intra_stream_;    // intra-node put ladder (CEs)
  std::optional<c10::cuda::CUDAStream> a2av_gateway_stream_;  // gateway forward ladder
  std::optional<c10::cuda::CUDAStream> a2av_reduce_stream_;   // signal waits + per-split reduce
  std::optional<c10::cuda::CUDAStream> a2av_conv_stream_;     // compress: convergence put ladder
  std::optional<c10::cuda::CUDAStream> a2av_prered_stream_;   // compress: resident pre-reduce kernel
  cudaEvent_t a2av_intra_done_ = nullptr;
  cudaEvent_t a2av_inter_done_ = nullptr;
  cudaEvent_t a2av_gateway_done_ = nullptr;
  cudaEvent_t a2av_reduce_done_ = nullptr;
  cudaEvent_t a2av_conv_done_ = nullptr;
  cudaEvent_t a2av_prered_done_ = nullptr;

  bool buffer_initialized = false;

 private:
  void
  init_buffer_once(at::ScalarType dtype) {
    if (this->buffer_initialized)
      return;
    if (this->a2av_hier) {
      // a2av mode skips every dense-only buffer (ring reduce buffers, tile
      // barriers, dense staging, internode signals): peers never write partials,
      // only whole copies into the recv panel. The ctor flag is uniform across
      // ranks, so skipping the collective allocations is collectively consistent.
      const int64_t n_per = this->n_dim / this->n_split;
      this->a2av_recv_rows_ = this->max_m / this->world_size;  // exact: topk copies per token
      this->a2av_send_rows_ = get_int_from_env(
          "FLUX_A2AV_RS_MAX_SEND_ROWS",
          std::min<int64_t>((int64_t)this->max_m, 2 * this->a2av_recv_rows_));
      this->a2av_send_panel_ =
          nvshmem_create_tensor({this->n_split, this->a2av_send_rows_, n_per}, dtype);
      this->a2av_recv_panel_ =
          nvshmem_create_tensor({this->n_split, this->a2av_recv_rows_, n_per}, dtype);
      this->a2av_recv_signals_ = nvshmem_create_tensor(
          {(int64_t)this->world_size * this->n_split}, at::ScalarType::Long, true);
      if (this->nnodes > 1 && !this->a2av_compress_) {
        this->a2av_stage_rows_ = get_int_from_env(
            "FLUX_A2AV_RS_MAX_STAGE_ROWS",
            std::min<int64_t>((int64_t)this->max_m, 2 * this->a2av_recv_rows_));
        this->a2av_stage_panel_ =
            nvshmem_create_tensor({this->n_split, this->a2av_stage_rows_, n_per}, dtype);
        this->a2av_arrival_signals_ = nvshmem_create_tensor(
            {(int64_t)this->nnodes * this->n_split}, at::ScalarType::Long, true);
      }
      if (this->a2av_compress_) {
        // compress replaces the destination-side staging/gateway machinery with
        // source-side convergence + wire panels; the flag is a uniform ctor
        // input so the collective allocation swap is consistent across ranks
        this->a2av_conv_rows_ = get_int_from_env(
            "FLUX_A2AV_RS_MAX_CONV_ROWS",
            std::min<int64_t>((int64_t)this->max_m, 2 * this->a2av_recv_rows_));
        this->a2av_wire_rows_ = get_int_from_env(
            "FLUX_A2AV_RS_MAX_WIRE_ROWS",
            std::min<int64_t>((int64_t)this->max_m, 2 * this->a2av_recv_rows_));
        this->a2av_conv_panel_ =
            nvshmem_create_tensor({this->n_split, this->a2av_conv_rows_, n_per}, dtype);
        this->a2av_wire_panel_ =
            nvshmem_create_tensor({this->n_split, this->a2av_wire_rows_, n_per}, dtype);
        this->a2av_conv_signals_ = nvshmem_create_tensor(
            {(int64_t)this->local_world_size * this->nnodes * this->n_split},
            at::ScalarType::Long,
            true);
        this->wire_flags_.reset(this->nnodes * this->n_split);
        this->wire_counters_.reset(this->nnodes * this->n_split);
        CUDA_CHECK(
            cudaMemset(this->wire_flags_.get(), 0, sizeof(int) * this->nnodes * this->n_split));
        CUDA_CHECK(cudaMemset(
            this->wire_counters_.get(), 0, sizeof(int) * this->nnodes * this->n_split));
        if (this->a2av_lane_chain_) {
          // fp32 accumulator (plain device memory; the chain's scatter-adds
          // and finalize are the only readers/writers)
          const int64_t ntok_max = (int64_t)this->max_m / this->topk / this->world_size;
          this->a2av_scratch_fp32_ = empty_with_uninitialized_data(
              std::vector<int64_t>{ntok_max, (int64_t)this->n_dim},
              at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Float));
        }
        if (this->a2av_lane_chain_ || this->a2av_recv_dyn_) {
          // recv-row -> token map (lane-chain scatter-adds; arrival-dynamic
          // receiver's per-row completion decrements)
          this->a2av_token_of_row_ = empty_with_uninitialized_data(
              std::vector<int64_t>{this->a2av_recv_rows_},
              at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Int));
        }
        if (this->a2av_bucket_) {
          const int64_t ntok_max = (int64_t)this->max_m / this->topk / this->world_size;
          auto opt_i32 = at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Int);
          this->a2av_bucket_comp_ =
              empty_with_uninitialized_data(std::vector<int64_t>{ntok_max}, opt_i32);
          this->a2av_bucket_tok_ =
              empty_with_uninitialized_data(std::vector<int64_t>{ntok_max}, opt_i32);
          this->a2av_bucket_meta_ = empty_with_uninitialized_data(
              std::vector<int64_t>{3 * kA2AVMaxWorld + 2}, opt_i32);
          this->a2av_bucket_lanes_h_ = torch::empty(
              {2 * 2 * ((int64_t)this->world_size + 1)},
              at::TensorOptions(at::kCPU).dtype(at::ScalarType::Int).pinned_memory(true));
          this->a2av_bucket_lanes_d_ = empty_with_uninitialized_data(
              std::vector<int64_t>{2 * 2 * ((int64_t)this->world_size + 1)}, opt_i32);
        }
      }
      // chunk-ready flags per (dest_node, sid) -- allocated for nnodes == 1 too:
      // the intra-node ladder gates on the own-node flag
      this->group_flags.reset(this->nnodes * this->n_split);
      this->group_counters.reset(this->nnodes * this->n_split);
      CUDA_CHECK(
          cudaMemset(this->group_flags.get(), 0, sizeof(int) * this->nnodes * this->n_split));
      CUDA_CHECK(
          cudaMemset(this->group_counters.get(), 0, sizeof(int) * this->nnodes * this->n_split));
      // Preload every kernel this data path launches: ours (attribute queries
      // force the module loads) plus NVSHMEM's on-stream transfer/signal
      // kernels (primed by issuing one real op per transport path). Under
      // CUDA_MODULE_LOADING=LAZY a kernel's module is loaded at its FIRST
      // launch, and the eager / compress schedules put a persistent spin
      // kernel on the device BEFORE the epoch's first NVSHMEM on-stream call:
      // that first-launch load never completes behind the never-exiting
      // resident kernel and the epoch deadlocks (2-node eager/compress hang,
      // root-caused 2026-08-16; legacy survives only because its lone spin
      // kernel, the pack, drains once the GEMM finishes). The ctor runs with
      // an idle device, so every load below is trivial. The priming ops write
      // SET 0 over zero-initialized signal slots (a no-op value), and
      // nvshmem_barrier_all() orders their remote delivery before any epoch.
      a2av_combine_preload(from_torch_dtype(dtype));
      {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        uint64_t *sig = (uint64_t *)this->a2av_recv_signals_.data_ptr();
        nvshmemx_signal_op_on_stream(sig, 0, NVSHMEM_SIGNAL_SET, this->rank, stream);
        if (this->local_world_size > 1) {
          int peer = this->node_idx * this->local_world_size +
                     (this->local_rank + 1) % this->local_world_size;
          flux_rs_put_signal(
              sig, sig, sizeof(uint64_t), sig + 1, 0, NVSHMEM_SIGNAL_SET, peer, stream, this->local_world_size, this->node_idx);
          // BARE signal_op to a P2P peer is a DISTINCT transport kernel from
          // both the self signal_op and putmem_signal — it is what the
          // ladders emit for ZERO-ROW intra-node lanes (always-signal
          // invariant). Unprimed, its first launch deadlocks behind the
          // resident pack/pre-reduce spin kernels (the 2026-08-16 class;
          // recurred 2026-08-17 as the epic_l01_hc_m4 b2 hang).
          nvshmemx_signal_op_on_stream(sig, 0, NVSHMEM_SIGNAL_SET, peer, stream);
        }
        if (this->nnodes > 1) {
          int peer = ((this->node_idx + 1) % this->nnodes) * this->local_world_size +
                     this->local_rank;
          flux_rs_put_signal(
              sig, sig, sizeof(uint64_t), sig + 1, 0, NVSHMEM_SIGNAL_SET, peer, stream, this->local_world_size, this->node_idx);
          // Same for the INTER-NODE bare signal_op (NIC/proxy transport):
          // emitted iff a remote lane has zero rows — U[d][n] == 0 in the
          // compress wire ladder, node_chunk == 0 in plain hier — which is
          // exactly the small-budget small-K_g regime. THE 20260817 fix.
          nvshmemx_signal_op_on_stream(sig, 0, NVSHMEM_SIGNAL_SET, peer, stream);
        }
        nvshmemx_quiet_on_stream(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        nvshmem_barrier_all();
      }
      torch::cuda::synchronize();
      this->buffer_initialized = true;
      return;
    }
    std::vector<void *> hptrs(this->local_world_size, nullptr);
    const int ptr_bytes = sizeof(void *) * this->local_world_size;
    // initialize the output buffer
    this->reduce_buffers = flux_create_tensor_list(
        {this->max_m / this->topk, this->n_dim}, dtype, this->tp_group.get());
    FLUX_CHECK_EQ((int)this->reduce_buffers.size(), this->local_world_size);
    this->reduce_buffer = this->reduce_buffers[this->local_rank];
    for (int i = 0; i < this->local_world_size; ++i) {
      hptrs[i] = reduce_buffers[i].data_ptr();
    }
    CHECK(!reduce_buffer_dptrs.defined());
    this->reduce_buffer_dptrs =
        torch::empty({ptr_bytes}, at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Byte));
    CUDA_CHECK(cudaMemcpy(
        this->reduce_buffer_dptrs.data_ptr(), hptrs.data(), ptr_bytes, cudaMemcpyHostToDevice));
    if (this->nnodes > 1) {
      const int64_t staging_rows = this->max_m / this->topk / this->world_size;
      const int64_t n_per = this->n_dim / this->n_split;
      this->staging_send =
          nvshmem_create_tensor({this->nnodes, this->n_split, staging_rows, n_per}, dtype);
      this->staging_recv =
          nvshmem_create_tensor({this->nnodes, this->n_split, staging_rows, n_per}, dtype);
      this->internode_signals =
          nvshmem_create_tensor({this->nnodes * this->n_split}, at::ScalarType::Long, true);
      this->group_flags.reset(this->nnodes * this->n_split);
      this->group_counters.reset(this->nnodes * this->n_split);
      CUDA_CHECK(cudaMemset(this->group_flags.get(), 0, sizeof(int) * this->nnodes * this->n_split));
      CUDA_CHECK(
          cudaMemset(this->group_counters.get(), 0, sizeof(int) * this->nnodes * this->n_split));
    }
    torch::cuda::synchronize();
    this->buffer_initialized = true;
  }
  int
  get_tile_barrier_size(int num_tiles) const {
    return num_tiles;
  }

  void
  create_rs_barrier() {
    int m_tiles_at_most = (this->max_m + kTileSizeM - 1) / kTileSizeM + this->ep_nexperts;
    const int tile_n = combine_tile_n(this->n_dim, this->n_split);
    int n_tiles = (this->n_dim + tile_n - 1) / tile_n;
    int num_tiles = m_tiles_at_most * n_tiles;

    int tile_barrier_size = get_tile_barrier_size(num_tiles);
    if (!this->tile_barrier.defined() || this->tile_barrier.numel() < tile_barrier_size) {
      // initialize the tile_barrier
      this->tile_barriers =
          flux_create_tensor_list({tile_barrier_size}, at::ScalarType::Int, this->tp_group.get());
      FLUX_CHECK_EQ((int)this->tile_barriers.size(), this->local_world_size);
      this->tile_barrier = this->tile_barriers[this->local_rank];
      std::vector<int *> hptrs(this->local_world_size, nullptr);
      const int ptr_bytes = sizeof(int *) * this->local_world_size;
      for (int i = 0; i < this->local_world_size; ++i) {
        hptrs[i] = (int *)this->tile_barriers[i].data_ptr();
      }
      CHECK(!tile_barrier_dptrs.defined());
      this->tile_barrier_dptrs =
          torch::empty({ptr_bytes}, at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Byte));
      CUDA_CHECK(cudaMemcpy(
          this->tile_barrier_dptrs.data_ptr(), hptrs.data(), ptr_bytes, cudaMemcpyHostToDevice));
    }
  }

  c10::cuda::CUDAStream
  create_internode_stream() const {
    at::cuda::CUDAGuard guard(at::cuda::current_device());
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    return at::cuda::getStreamFromExternal(stream, at::cuda::current_device());
  }

 public:
  void *
  send_panel_ptr() {
    this->init_buffer_once(this->output_dtype);  // idempotent
    return this->a2av_send_panel_.data_ptr();
  }
  int64_t
  send_panel_rows() {
    return this->a2av_send_rows_;
  }
  void
  ensure_buffers(at::ScalarType dtype) {
    this->init_buffer_once(dtype);
  }

  void
  set_msplit_waves(
      std::vector<int> const &wave_of_node, std::vector<int> const &node_order, int n_waves) {
    if (n_waves <= 0) {
      this->msplit_n_waves_ = 0;
      return;
    }
    FLUX_CHECK(this->a2av_hier) << "msplit waves require the a2av_hier combine";
    FLUX_CHECK_EQ(this->n_split, 1) << "msplit requires n_split == 1";
    FLUX_CHECK_EQ((int)wave_of_node.size(), this->nnodes);
    FLUX_CHECK_EQ((int)node_order.size(), this->nnodes);
    FLUX_CHECK_LE(this->nnodes, kA2AVMaxNodes);
    const int my_node2 = this->rank / this->local_world_size;
    FLUX_CHECK(
        node_order[this->nnodes - 1] == my_node2 || node_order[0] == my_node2)
        << "own node must be the first or final schedule step";
    this->msplit_n_waves_ = n_waves;
    for (int i = 0; i < this->nnodes; i++) {
      FLUX_CHECK_LT(wave_of_node[i], n_waves);
      this->msplit_wave_of_node_[i] = wave_of_node[i];
      this->msplit_node_order_[i] = node_order[i];
    }
  }

  TopkReduceScatterOpImpl(
      std::shared_ptr<Group> tp_group_,
      int max_m,
      int n_dim,
      int topk,
      at::ScalarType output_dtype,
      int ep_nexperts,
      int ep_world_size,
      const std::vector<torch::Tensor> &barriers,
      int n_split_,
      bool do_all_reduce_ = false,
      bool use_read_mode_ = false,
      int nnodes_ = 1,
      bool a2av_hier_ = false,
      bool a2av_compress = false)
      : tp_group(tp_group_),
        rank(tp_group_->get_rank()),
        world_size(tp_group_->get_size()),
        nnodes(nnodes_),
        node_idx(DistEnv(tp_group_->get_rank(), tp_group_->get_size(), nnodes_).node_idx),
        local_rank(DistEnv(tp_group_->get_rank(), tp_group_->get_size(), nnodes_).local_rank),
        local_world_size(tp_group_->get_size() / nnodes_),
        max_m(max_m),
        n_dim(n_dim),
        topk(topk),
        output_dtype(output_dtype),
        ep_nexperts(ep_nexperts),
        ep_world_size(ep_world_size),
        do_all_reduce(do_all_reduce_),
        use_read_mode(use_read_mode_),
        n_split(n_split_),
        internode_stream(create_internode_stream()),
        a2av_hier(a2av_hier_),
        // eager (arrival-order) receiver reduce defaults ON under msplit —
        // wave arrivals are spread across the GEMM span and the wait-all gate
        // would forfeit the receive-side overlap; explicit env still wins
        a2av_eager_(
            a2av_hier_ &&
            get_int_from_env(
                "FLUX_A2AV_RS_EAGER",
                (nnodes_ > 1 && get_a2av_rs_msplit() != 0 && get_a2av_rs_bucket() == 0)
                    ? 1
                    : 0) != 0),
        a2av_compress_(a2av_compress && nnodes_ > 1),
        a2av_lane_chain_(
            a2av_compress && nnodes_ > 1 && get_a2av_rs_lane_chain() != 0),
        a2av_bucket_(
            a2av_compress && nnodes_ > 1 && get_a2av_rs_bucket() != 0 &&
            (a2av_env_explicit("FLUX_A2AV_RS_BUCKET") ||
             (n_split_ == 1 && get_a2av_rs_lane_chain() == 0))),
        a2av_recv_dyn_(a2av_bucket_ && get_a2av_rs_recv_dyn() != 0),
        a2av_fused_pack_(
            a2av_hier_ && nnodes_ > 1 && get_a2av_rs_fused_pack() != 0),
        barriers(barriers) {
    if (this->a2av_lane_chain_) {
      FLUX_CHECK_EQ(n_split_, 1) << "FLUX_A2AV_RS_LANE_CHAIN requires n_split == 1";
    }
    if (this->a2av_bucket_) {
      FLUX_CHECK_EQ(n_split_, 1) << "FLUX_A2AV_RS_BUCKET requires n_split == 1";
      FLUX_CHECK(!this->a2av_lane_chain_)
          << "FLUX_A2AV_RS_BUCKET and FLUX_A2AV_RS_LANE_CHAIN are exclusive receivers";
    }
    if (a2av_env_explicit("FLUX_A2AV_RS_RECV_DYN") && get_a2av_rs_recv_dyn() != 0) {
      // explicit env keeps the loud contract (the default silently rides the
      // bucket flag, so demoted-bucket configs stay untouched)
      FLUX_CHECK(this->a2av_bucket_)
          << "FLUX_A2AV_RS_RECV_DYN requires the bucket receiver's plan "
             "(compress C', n_split == 1, FLUX_A2AV_RS_BUCKET on)";
    }
    FLUX_CHECK_GE(nnodes, 1);
    FLUX_CHECK_DIV(world_size, nnodes);
    if (nnodes > 1) {
      FLUX_CHECK(!do_all_reduce) << "do_all_reduce not supported with nnodes > 1";
      FLUX_CHECK(!use_read_mode) << "use_read_mode not supported with nnodes > 1";
      FLUX_CHECK_DIV(max_m / topk, world_size);
      FLUX_CHECK(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE) == local_rank)
          << "rank layout must be node-contiguous (rank = node_idx * local_world_size + "
             "local_rank)";
    }
    FLUX_CHECK(!a2av_compress || a2av_hier_) << "a2av_hier_compress implies the a2av data path";
    if (a2av_compress && nnodes_ == 1 && this->rank == 0) {
      FLUX_LOG_FIRST_N(INFO, 1)
          << "a2av_hier_compress on a single node degrades to plain a2av_hier "
             "(node-level dedup saves zero wire bytes)\n";
    }
    if (this->a2av_hier) {
      FLUX_CHECK(!do_all_reduce) << "do_all_reduce not supported with a2av_hier";
      FLUX_CHECK(!use_read_mode) << "use_read_mode not supported with a2av_hier";
      FLUX_CHECK_EQ(ep_world_size, world_size) << "a2av_hier requires EP == world (tp == 1)";
      FLUX_CHECK_DIV(max_m, world_size);
      FLUX_CHECK(
          output_dtype == at::ScalarType::Half || output_dtype == at::ScalarType::BFloat16)
          << "a2av_hier supports fp16/bf16 only";
      FLUX_CHECK(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE) == local_rank)
          << "rank layout must be node-contiguous (rank = node_idx * local_world_size + "
             "local_rank)";
      this->a2av_intra_stream_ = create_internode_stream();
      this->a2av_reduce_stream_ = create_internode_stream();
      if (nnodes > 1 && !this->a2av_compress_) {
        this->a2av_gateway_stream_ = create_internode_stream();
      }
      if (this->a2av_compress_) {
        this->a2av_conv_stream_ = create_internode_stream();
        this->a2av_prered_stream_ = create_internode_stream();
        CUDA_CHECK(cudaEventCreateWithFlags(&this->a2av_conv_done_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&this->a2av_prered_done_, cudaEventDisableTiming));
      }
      CUDA_CHECK(cudaEventCreateWithFlags(&this->a2av_intra_done_, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&this->a2av_inter_done_, cudaEventDisableTiming));
      // Slipstream canonicalization (2026-08-23, M4 verdict capsules): the
      // split wire ladder is the DEFAULT (l1 win sign-stable 16/16 fwd+rev
      // x both canon models x b2-b64; wire rule intact — every inter-node
      // put stays blocking putmem_signal). =1 is the single-stream
      // ablation. Rule-4 DEFAULT tag below (string literal probed by the
      // sweep runner's capability check).
      (void)get_int_from_env("FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT2_TAG", 0);
      // 2026-08-24 canonicalization (user decision): DEFAULT 16 wire lanes
      // (~= node count at 16n; knee is ~14-15 total lanes and flat past it
      // at both measured scales, so 16 sits on the plateau; trivially
      // adjustable for 32n). Rule-4 DEFAULT tag below.
      (void)get_int_from_env("FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT16_TAG", 0);
      // 2026-08-23 (l1 combine 16n campaign): the knob is now a true ladder —
      // S streams round-robin the (sid, tn) cells (gi % S), so at large NN
      // the per-target blocking puts overlap instead of ~ (NN-1)/2 of them
      // serializing per stream. S=2 reproduces the canonical parity split
      // bit-identically ((gi % 2) == (gi & 1)); capped at 16. Binaries with
      // the general ladder carry FLUX_A2AV_RS_WIRE_NSTREAMS_TAG.
      (void)get_int_from_env("FLUX_A2AV_RS_WIRE_NSTREAMS_TAG", 0);
      (void)get_int_from_env("FLUX_A2AV_RS_WIRE_XSPREAD_TAG", 0);
      this->rs_wire_streams_ = get_int_from_env("FLUX_A2AV_RS_WIRE_STREAMS", 16);
      if (this->rs_wire_streams_ < 1) {
        this->rs_wire_streams_ = 1;
      }
      if (this->rs_wire_streams_ > 32) {
        this->rs_wire_streams_ = 32;
      }
      for (int i = 1; i < this->rs_wire_streams_; i++) {
        this->internode_streams2_.push_back(create_internode_stream());
      }
      if (this->rs_wire_streams_ > 1) {
        CUDA_CHECK(
            cudaEventCreateWithFlags(&this->a2av_inter2_done_, cudaEventDisableTiming));
      }
      CUDA_CHECK(cudaEventCreateWithFlags(&this->a2av_gateway_done_, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&this->a2av_reduce_done_, cudaEventDisableTiming));
    }
    if (!this->a2av_hier && nnodes > 1) {
      // 2026-08-23 (l1 combine 16n campaign, 8n-agent diagnosis): the DENSE
      // multi-node sender serialized n_split*(NN-1) blocking putmem_signal on
      // ONE internode stream (~pure ladder-overhead-bound at low budgets:
      // K2 b1 8n l1 ~= 49 puts x ~0.24 ms proxy RTT with near-zero payload).
      // FLUX_RS_WIRE_STREAMS (default 1 = shipped schedule, opt-in ladder,
      // clamp 32) spreads the (sid, gi) cells over S lanes exactly like the
      // a2av XSPREAD mapping. Every put stays a BLOCKING putmem_signal.
      // Binaries with the dense ladder carry FLUX_RS_WIRE_NSTREAMS_TAG.
      (void)get_int_from_env("FLUX_RS_WIRE_NSTREAMS_TAG", 0);
      // 2026-08-24 canonicalization (user decision): dense default 16 lanes,
      // same rationale as the a2av knob. Rule-4 DEFAULT tag below.
      (void)get_int_from_env("FLUX_RS_WIRE_STREAMS_DEFAULT16_TAG", 0);
      this->rs_wire_streams_ = get_int_from_env("FLUX_RS_WIRE_STREAMS", 16);
      if (this->rs_wire_streams_ < 1) {
        this->rs_wire_streams_ = 1;
      }
      if (this->rs_wire_streams_ > 32) {
        this->rs_wire_streams_ = 32;
      }
      for (int i = 1; i < this->rs_wire_streams_; i++) {
        this->internode_streams2_.push_back(create_internode_stream());
      }
      if (this->rs_wire_streams_ > 1) {
        CUDA_CHECK(
            cudaEventCreateWithFlags(&this->a2av_inter2_done_, cudaEventDisableTiming));
      }
    }
    this->init_buffer_once(output_dtype);
    if (!this->a2av_hier) {
      this->create_rs_barrier();
    }

    std::vector<void *> barrier_ptrs(this->local_world_size, nullptr);
    FLUX_CHECK_EQ((int)this->barriers.size(), this->local_world_size);
    for (int i = 0; i < this->local_world_size; i++) {
      barrier_ptrs[i] = this->barriers[i].data_ptr();
    }
    CUDA_CHECK(cudaMalloc(&this->barrier_dev_ptrs, this->local_world_size * sizeof(void *)));
    CUDA_CHECK(cudaMemcpy(
        this->barrier_dev_ptrs,
        barrier_ptrs.data(),
        this->local_world_size * sizeof(void *),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->staging_reset_event, cudaEventDisableTiming));
    torch::cuda::synchronize();  // we don't assume create/run on the same stream so sync is safe
    this->barrier = this->barriers[this->local_rank];
  }

  ~TopkReduceScatterOpImpl() {
    CUDA_CHECK(cudaEventDestroy(this->staging_reset_event));
    CUDA_CHECK(cudaStreamDestroy(this->internode_stream));
    for (auto &s : {this->a2av_intra_stream_, this->a2av_gateway_stream_, this->a2av_reduce_stream_,
                    this->a2av_conv_stream_, this->a2av_prered_stream_}) {
      if (s.has_value()) {
        CUDA_CHECK(cudaStreamDestroy(s.value()));
      }
    }
    for (auto &s2 : this->internode_streams2_) {
      CUDA_CHECK(cudaStreamDestroy(s2));
    }
    if (this->a2av_inter2_done_ != nullptr) {
      CUDA_CHECK(cudaEventDestroy(this->a2av_inter2_done_));
    }
    for (auto e : {this->a2av_intra_done_, this->a2av_inter_done_, this->a2av_gateway_done_,
                   this->a2av_reduce_done_, this->a2av_conv_done_, this->a2av_prered_done_}) {
      if (e != nullptr) {
        CUDA_CHECK(cudaEventDestroy(e));
      }
    }
    if (this->barrier_dev_ptrs != nullptr) {
      CUDA_CHECK(cudaFree(this->barrier_dev_ptrs));
    }
  }

  // a2av_hier combine: pack (persistent kernel, split-major, behind the GEMM
  // cascade flags) -> host put ladders on copy engines / NIC (intra direct,
  // inter-node aggregated via same-local-rank gateways, gateway forwards paced
  // by zero-SM cuStreamWaitValue64) -> per-split destination topk reduce once
  // that split's W per-source recv signals have fired. All host waits are
  // enqueued AFTER the pack kernel launch and in dependency order (intra/inter
  // ladders, then gateway, then reduce): under CUDA_DEVICE_MAX_CONNECTIONS=1
  // every enqueued front-end wait can block later ops in the shared channel, so
  // enqueue order must be an executable schedule.
  torch::Tensor
  run_a2av_hier(
      torch::Tensor gemm_out,
      torch::Tensor output,
      torch::Tensor const &splits_per_source,
      torch::Tensor const &pack_index,
      torch::Tensor const &reduce_index,
      c10::optional<torch::Tensor> const &unique_counts,
      c10::optional<std::vector<torch::Tensor>> const &wire_csr,
      c10::optional<std::vector<torch::Tensor>> const &reduce_csr,
      c10::optional<std::vector<torch::Tensor>> const &output_vec_scales,
      int m_full,
      int num_thread_blocks,
      cudaStream_t stream_raw) {
    const int W = this->world_size;
    const int L = this->local_world_size;
    const int NN = this->nnodes;
    const int my_node = this->node_idx;
    const int my_lr = this->local_rank;
    DistEnv dist_env(this->rank, W, NN);
    const int nex_total = this->ep_nexperts * this->ep_world_size;
    const int E_loc = nex_total / W;  // experts per owner rank (EP == world)
    const int64_t cpr = (int64_t)m_full / W;  // copies homed on each rank
    const int64_t M_this_ep = gemm_out.size(0);
    const int64_t n_per = this->n_dim / this->n_split;
    auto dtype = gemm_out.scalar_type();
    const int64_t row_bytes = n_per * c10::elementSize(dtype);

    FLUX_CHECK(splits_per_source.device().is_cpu()) << "splits_per_source must be a CPU tensor";
    CHECK_2D(splits_per_source, W, nex_total);
    FLUX_CHECK(splits_per_source.is_contiguous());
    FLUX_CHECK(splits_per_source.scalar_type() == at::ScalarType::Int);
    CHECK_INPUT(pack_index, at::ScalarType::Int);
    CHECK_INPUT(reduce_index, at::ScalarType::Int);
    CHECK_1D(pack_index, M_this_ep);
    CHECK_1D(reduce_index, cpr);
    FLUX_CHECK_LE(cpr, this->a2av_recv_rows_);

    // combine chunk matrix: C[s][d] = copies expert-owner s returns to home d
    // = sum over s's experts of cnt[d][e] -- the transpose-aggregate of the
    // dispatch chunk matrix, from the same metadata-exchange input.
    const int32_t *cnt = splits_per_source.data_ptr<int32_t>();
    std::vector<int64_t> chunks64((size_t)W * W, 0);
    for (int s = 0; s < W; s++) {
      for (int d = 0; d < W; d++) {
        int64_t acc = 0;
        for (int e = s * E_loc; e < (s + 1) * E_loc; e++) {
          acc += cnt[d * nex_total + e];
        }
        chunks64[s * W + d] = acc;
      }
    }
    auto chunk_at = [&](int s, int d) -> int64_t { return chunks64[s * W + d]; };
    // sanity: my outbound rows == gemm rows; every home receives exactly cpr copies
    {
      int64_t my_rows = 0;
      for (int d = 0; d < W; d++) {
        my_rows += chunk_at(this->rank, d);
      }
      FLUX_CHECK_EQ(my_rows, M_this_ep) << "splits_per_source disagrees with gemm rows";
      for (int d = 0; d < W; d++) {
        int64_t col = 0;
        for (int s = 0; s < W; s++) {
          col += chunk_at(s, d);
        }
        FLUX_CHECK_EQ(col, cpr) << "chunk matrix column " << d << " != ntokens_local * topk";
      }
    }
    // send-panel overflow check, evaluated identically on ALL ranks (max over
    // every rank's outbound rows) so failure is collective, never a hang
    {
      int64_t max_send_rows = 0;
      for (int s = 0; s < W; s++) {
        int64_t rows = 0;
        for (int d = 0; d < W; d++) {
          rows += chunk_at(s, d);
        }
        max_send_rows = std::max(max_send_rows, rows);
      }
      FLUX_CHECK_LE(max_send_rows, this->a2av_send_rows_)
          << "a2av_hier send panel overflow; raise FLUX_A2AV_RS_MAX_SEND_ROWS";
    }
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
    auto recv_off_of = [&](int s, int d) -> int64_t {
      int64_t acc = 0;
      for (int s2 = 0; s2 < s; s2++) {
        acc += chunk_at(s2, d);
      }
      return acc;
    };
    if (NN > 1 && !this->a2av_compress_) {
      // gateway staging overflow, same collective-evaluation discipline
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
      FLUX_CHECK_LE(max_stage_rows, this->a2av_stage_rows_)
          << "a2av_hier staging overflow; raise FLUX_A2AV_RS_MAX_STAGE_ROWS";
    }
    std::vector<int64_t> send_off(W, 0);
    for (int d = 0, acc = 0; d < W; d++) {
      send_off[d] = acc;
      acc += chunk_at(this->rank, d);
    }

    // ---- compress host tables: C' recv layout, conv/wire offsets, checks ----
    const int32_t *U = nullptr;
    if (this->a2av_compress_) {
      FLUX_CHECK(unique_counts.has_value())
          << "a2av_hier_compress requires a2av_unique_counts ([W, nnodes] int32 CPU)";
      FLUX_CHECK(wire_csr.has_value() && reduce_csr.has_value())
          << "a2av_hier_compress requires the wire/reduce CSRs (built by the gather-rs "
             "op or passed as precomputed routing-plan inputs)";
      U = unique_counts->data_ptr<int32_t>();
    }
    // C'[s][d]: own-node lanes keep per-rank chunks; the remote lane
    // materializes only at the same-lr source rank with U[d][node(s)] rows
    auto chunk_cp = [&](int s, int d) -> int64_t {
      if (s / L == d / L) {
        return chunk_at(s, d);
      }
      if (s % L == d % L) {
        return U[d * NN + s / L];
      }
      return 0;
    };
    auto recv_off_cp = [&](int s, int d) -> int64_t {
      int64_t acc = 0;
      for (int s2 = 0; s2 < s; s2++) {
        acc += chunk_cp(s2, d);
      }
      return acc;
    };
    // active recv layout: C' under compress, C otherwise (intra puts + eager
    // lane prefixes must agree with what the wire actually delivers)
    auto recv_off_active = [&](int s, int d) -> int64_t {
      return this->a2av_compress_ ? recv_off_cp(s, d) : recv_off_of(s, d);
    };
    // conv panel offset at gateway (my_node, dl): segments (tn asc skip own,
    // ls asc), each C[(my_node, ls)][(tn, dl)] rows in the peer's panel order
    auto conv_off = [&](int dl, int tn, int ls) -> int64_t {
      int64_t acc = 0;
      for (int t2 = 0; t2 < NN; t2++) {
        if (t2 == my_node) {
          continue;
        }
        if (t2 == tn) {
          break;
        }
        for (int l2 = 0; l2 < L; l2++) {
          acc += chunk_at(my_node * L + l2, t2 * L + dl);
        }
      }
      for (int l2 = 0; l2 < ls; l2++) {
        acc += chunk_at(my_node * L + l2, tn * L + dl);
      }
      return acc;
    };
    // my wire panel: segments (tn asc skip own), U[(tn, my_lr)][my_node] rows
    auto wire_seg_off = [&](int tn) -> int64_t {
      int64_t acc = 0;
      for (int t2 = 0; t2 < NN; t2++) {
        if (t2 == my_node || t2 >= tn) {
          continue;
        }
        acc += U[(t2 * L + my_lr) * NN + my_node];
      }
      return acc;
    };
    if (this->a2av_compress_) {
      // conv/wire overflow, collective-evaluation discipline (identical
      // expressions on every rank, so failure aborts everywhere, never hangs)
      int64_t max_conv = 0, max_wire = 0;
      for (int n2 = 0; n2 < NN; n2++) {
        for (int dl = 0; dl < L; dl++) {
          int64_t conv_rows = 0, wire_rows = 0;
          for (int tn = 0; tn < NN; tn++) {
            if (tn == n2) {
              continue;
            }
            for (int ls = 0; ls < L; ls++) {
              conv_rows += chunk_at(n2 * L + ls, tn * L + dl);
            }
            wire_rows += U[(tn * L + dl) * NN + n2];
          }
          max_conv = std::max(max_conv, conv_rows);
          max_wire = std::max(max_wire, wire_rows);
        }
      }
      FLUX_CHECK_LE(max_conv, this->a2av_conv_rows_)
          << "a2av_hier_compress conv panel overflow; raise FLUX_A2AV_RS_MAX_CONV_ROWS";
      FLUX_CHECK_LE(max_wire, this->a2av_wire_rows_)
          << "a2av_hier_compress wire panel overflow; raise FLUX_A2AV_RS_MAX_WIRE_ROWS";
      FLUX_CHECK_LE(recv_off_cp(W, this->rank), cpr) << "C' image exceeds recv panel";
    }

    // per-run epoch + chunk-flag reset, published to the ladder streams before
    // their first CUStreamWaitValue can observe them
    this->run_id_ += 1;
    const size_t flag_bytes = sizeof(int) * NN * this->n_split;
    CUDA_CHECK(cudaMemsetAsync(this->group_flags.get(), 0, flag_bytes, stream_raw));
    CUDA_CHECK(cudaMemsetAsync(this->group_counters.get(), 0, flag_bytes, stream_raw));
    if (this->a2av_compress_) {
      // wire flags/counters join the same reset + event-publication discipline:
      // any stream observing them must first wait staging_reset_event, so a
      // stale 1 from the previous run can never release a wire put early
      CUDA_CHECK(cudaMemsetAsync(this->wire_flags_.get(), 0, flag_bytes, stream_raw));
      CUDA_CHECK(cudaMemsetAsync(this->wire_counters_.get(), 0, flag_bytes, stream_raw));
    }
    CUDA_CHECK(cudaEventRecord(this->staging_reset_event, stream_raw));
    cudaStream_t intra_stream = this->a2av_intra_stream_.value();
    cudaStream_t reduce_stream = this->a2av_reduce_stream_.value();
    CUDA_CHECK(cudaStreamWaitEvent(intra_stream, this->staging_reset_event));
    if (NN > 1) {
      CUDA_CHECK(cudaStreamWaitEvent(this->internode_stream, this->staging_reset_event));
      for (auto &s2 : this->internode_streams2_) {
        CUDA_CHECK(cudaStreamWaitEvent(s2, this->staging_reset_event));
      }
    }
    if (this->a2av_compress_) {
      CUDA_CHECK(
          cudaStreamWaitEvent(this->a2av_conv_stream_.value(), this->staging_reset_event));
      CUDA_CHECK(
          cudaStreamWaitEvent(this->a2av_prered_stream_.value(), this->staging_reset_event));
    }

    // pack kernel FIRST -- every host wait below is enqueued after it, so the
    // shared front-end channel always has the flag producer ahead of its consumers
    // fused pack (gen-8c): the GEMM already scattered the panel with the
    // coefficients pre-folded — the pack is a flag relay, scale must be null
    const bool fused_relay = this->a2av_fused_pack_ && this->msplit_n_waves_ > 0;
    A2AVCombinePackArguments pack_args{
        .gemm_out = gemm_out.data_ptr(),
        .vec_scale = (!fused_relay && output_vec_scales.has_value())
                         ? (float const *)output_vec_scales->at(0).data_ptr()
                         : nullptr,
        .pack_index = pack_index.data_ptr<int32_t>(),
        .send_panel = this->a2av_send_panel_.data_ptr(),
        .barrier = this->barrier.data_ptr<int>(),
        .group_flags = this->group_flags.get(),
        .group_counters = this->group_counters.get(),
        .node_row_start = {},
        .panel_rows = this->a2av_send_rows_,
        .n = this->n_dim,
        .n_per = (int)n_per,
        .n_split = this->n_split,
        .nnodes = NN,
        .node_idx = my_node,
        .threadblock_count = num_thread_blocks};
    for (int n = 0; n < NN; n++) {
      pack_args.node_row_start[n] = send_off[n * L];
    }
    pack_args.node_row_start[NN] = M_this_ep;
    // M-split waves: gate the pack per schedule step on the wave's cascade flag
    pack_args.relay_only = fused_relay ? 1 : 0;
    pack_args.msplit = this->msplit_n_waves_ > 0 ? 1 : 0;
    if (pack_args.msplit) {
      FLUX_CHECK_EQ(this->n_split, 1);
      for (int gi = 0; gi < NN; gi++) {
        pack_args.wave_of_node[gi] = this->msplit_wave_of_node_[gi];
        pack_args.node_order[gi] = this->msplit_node_order_[gi];
      }
    }
    // remote-node production schedule for the conv/prered/wire consumers:
    // the msplit order when armed (own node filtered out — it may sit first
    // under OWN_FIRST or last under the canon), ring otherwise — every
    // consumer follows this ONE array
    int sched_remote[kA2AVMaxNodes];
    if (pack_args.msplit) {
      int w = 0;
      for (int gi = 0; gi < NN; gi++) {
        if (this->msplit_node_order_[gi] != my_node) {
          sched_remote[w++] = this->msplit_node_order_[gi];
        }
      }
      FLUX_CHECK_EQ(w, NN - 1);
    } else {
      for (int gi = 0; gi < NN - 1; gi++) {
        sched_remote[gi] = (my_node + 1 + gi) % NN;
      }
    }
    auto flux_dtype = from_torch_dtype(dtype);
    a2av_combine_pack(pack_args, flux_dtype, stream_raw);

    char *send_base = (char *)this->a2av_send_panel_.data_ptr();
    char *recv_base = (char *)this->a2av_recv_panel_.data_ptr();
    uint64_t *recv_sig = (uint64_t *)this->a2av_recv_signals_.data_ptr();
    auto send_ptr = [&](int sid, int64_t row) -> char * {
      return send_base + ((int64_t)sid * this->a2av_send_rows_ + row) * row_bytes;
    };
    auto recv_ptr = [&](int sid, int64_t row) -> char * {
      return recv_base + ((int64_t)sid * this->a2av_recv_rows_ + row) * row_bytes;
    };

    const bool gw_path = NN > 1 && !this->a2av_compress_;
    char *stage_base = gw_path ? (char *)this->a2av_stage_panel_.data_ptr() : nullptr;
    uint64_t *arrival_sig =
        gw_path ? (uint64_t *)this->a2av_arrival_signals_.data_ptr() : nullptr;
    auto stage_ptr = [&](int sid, int64_t row) -> char * {
      return stage_base + ((int64_t)sid * this->a2av_stage_rows_ + row) * row_bytes;
    };
    cudaStream_t gateway_stream = gw_path ? (cudaStream_t)this->a2av_gateway_stream_.value()
                                          : (cudaStream_t) nullptr;

    // compress: launch the persistent pre-reduce kernel right after the pack
    // kernel, before any host wait reaches the conn=1 channel (a blocked wait
    // ahead of a kernel launch could park it forever). It spins on the conv
    // signals per (tn, sid) and flips the wire flags the inter ladder gates on.
    char *conv_base = nullptr, *wire_base = nullptr;
    uint64_t *conv_sig = nullptr;
    if (this->a2av_compress_) {
      conv_base = (char *)this->a2av_conv_panel_.data_ptr();
      wire_base = (char *)this->a2av_wire_panel_.data_ptr();
      conv_sig = (uint64_t *)this->a2av_conv_signals_.data_ptr();
      A2AVCombinePreReduceArguments prered_args{
          .conv_panel = conv_base,
          .wire_panel = wire_base,
          .wire_ptr = wire_csr->at(0).data_ptr<int32_t>(),
          .wire_copy = wire_csr->at(1).data_ptr<int32_t>(),
          .conv_signals = conv_sig,
          .run_id = this->run_id_,
          .wire_flags = this->wire_flags_.get(),
          .wire_counters = this->wire_counters_.get(),
          .wire_seg_start = {},
          .conv_rows = this->a2av_conv_rows_,
          .wire_rows = this->a2av_wire_rows_,
          .n_per = (int)n_per,
          .n_split = this->n_split,
          .nnodes = NN,
          .node_idx = my_node,
          .local_world_size = L,
          .threadblock_count = get_a2av_prered_blocks(),
          .spin_limit = get_a2av_spin_limit()};
      for (int tn = 0, seg = 0; tn < NN; tn++) {
        if (tn == my_node) {
          continue;
        }
        prered_args.wire_seg_start[seg] = wire_seg_off(tn);
        seg++;
      }
      {
        int64_t total_wire = 0;
        for (int tn = 0; tn < NN; tn++) {
          if (tn != my_node) {
            total_wire += U[(tn * L + my_lr) * NN + my_node];
          }
        }
        prered_args.wire_seg_start[NN - 1] = total_wire;  // segment-array end
      }
      // production schedule (ring by default, msplit order when armed) — must
      // ALWAYS be filled: the kernel visits args.node_order unconditionally
      for (int gi = 0; gi < NN - 1; gi++) {
        prered_args.node_order[gi] = sched_remote[gi];
      }
      a2av_combine_prereduce(prered_args, flux_dtype, this->a2av_prered_stream_.value());
    }
    auto conv_ptr = [&](int sid, int64_t row) -> char * {
      return conv_base + ((int64_t)sid * this->a2av_conv_rows_ + row) * row_bytes;
    };
    auto wire_ptr_at = [&](int sid, int64_t row) -> char * {
      return wire_base + ((int64_t)sid * this->a2av_wire_rows_ + row) * row_bytes;
    };
    A2AVCombineReduceArguments reduce_args{
        .recv_panel = this->a2av_recv_panel_.data_ptr(),
        .reduce_index = reduce_index.data_ptr<int32_t>(),
        .output = output.data_ptr(),
        .panel_rows = this->a2av_recv_rows_,
        .ntokens_local = cpr / this->topk,
        .n = this->n_dim,
        .n_per = (int)n_per,
        .topk = this->topk,
        .sid = 0,
        .threadblock_count = get_a2av_reduce_blocks()};

    if (this->a2av_eager_ && !this->a2av_lane_chain_ && !this->a2av_bucket_) {
      // eager arrival-order reduce: ONE persistent kernel for all splits,
      // enqueued while the front-end channel still holds no host wait (conn=1:
      // a blocked wait ahead of a kernel launch could park it forever). It
      // polls the epoch-valued recv signals directly, so it needs neither the
      // flag-reset event nor any per-split gate below.
      A2AVCombineEagerReduceArguments eager_args{
          .recv_panel = this->a2av_recv_panel_.data_ptr(),
          .reduce_index = reduce_index.data_ptr<int32_t>(),
          .red_ptr = this->a2av_compress_ ? reduce_csr->at(0).data_ptr<int32_t>() : nullptr,
          .red_row = this->a2av_compress_ ? reduce_csr->at(1).data_ptr<int32_t>() : nullptr,
          .output = output.data_ptr(),
          .recv_signals = recv_sig,
          .run_id = this->run_id_,
          .recv_cum = {},
          .panel_rows = this->a2av_recv_rows_,
          .ntokens_local = cpr / this->topk,
          .world_size = W,
          .n = this->n_dim,
          .n_per = (int)n_per,
          .n_split = this->n_split,
          .topk = this->topk,
          .threadblock_count = get_a2av_reduce_blocks(),
          .spin_limit = get_a2av_spin_limit()};
      if (this->a2av_compress_) {
        // per-token contributions <= topk own-node copies + NN-1 merged rows
        FLUX_CHECK_LE(this->topk + NN - 1, 31)
            << "eager compress remaining-mask holds topk + nnodes - 1 in 31 bits";
      }
      for (int s = 0; s < W; s++) {
        eager_args.recv_cum[s] = recv_off_active(s, this->rank);
      }
      eager_args.recv_cum[W] = recv_off_active(W, this->rank);
      a2av_combine_eager_reduce(eager_args, flux_dtype, reduce_stream);
    }

    if (this->a2av_recv_dyn_) {
      // ARRIVAL-DYNAMIC receiver (H4, FLUX_A2AV_RS_RECV_DYN): the bucket
      // receiver's per-lane waits in EXPECTED order become one persistent
      // kernel consuming the same S lanes in TRUE arrival order (per-token
      // completion counters; fold arithmetic identical to the bucket fold, so
      // the output is bitwise-equal to wait-all). Enqueued HERE -- with the
      // eager kernel's discipline, before any host wait reaches the conn=1
      // front-end channel -- because a persistent kernel launch parked behind
      // a blocked wait never starts. It polls the epoch-valued recv signals
      // directly, so it needs neither the flag-reset event nor any per-split
      // gate below; the signal-trust contract is unchanged (a lane's rows are
      // touched only after its signal acquires >= run_id).
      FLUX_CHECK(this->a2av_compress_);
      FLUX_CHECK_EQ(this->n_split, 1);
      const int64_t ntok_local = cpr / this->topk;
      A2AVDynReduceArguments dyn_args{
          .recv_panel = this->a2av_recv_panel_.data_ptr(),
          .red_ptr = reduce_csr->at(0).data_ptr<int32_t>(),
          .red_row = reduce_csr->at(1).data_ptr<int32_t>(),
          .token_of = this->a2av_token_of_row_.data_ptr<int32_t>(),
          // remain reuses the bucket completion map, lane cursors the bucket
          // meta buffer: the bucket receiver's own launches are skipped below
          .remain = this->a2av_bucket_comp_.data_ptr<int32_t>(),
          .ntokens_local = ntok_local,
          .lane_cursor = this->a2av_bucket_meta_.data_ptr<int32_t>(),
          .output = output.data_ptr(),
          .recv_signals = recv_sig,
          .run_id = this->run_id_,
          .lane_sig = {},
          .lane_row_lo = {},
          .lane_rows = {},
          .n_lanes = 0,
          .n = (int)n_per,
          .chunk_rows = get_a2av_rs_recv_dyn_chunk(),
          .threadblock_count = get_a2av_reduce_blocks(),
          .spin_limit = get_a2av_spin_limit()};
      int Sd = 0;
      auto add_lane = [&](int s2) {
        const int64_t rows = chunk_cp(s2, this->rank);
        if (rows <= 0) {
          return;  // zero-row lanes still signal; nothing can complete there
        }
        dyn_args.lane_sig[Sd] = s2 * this->n_split;  // sid == 0
        dyn_args.lane_row_lo[Sd] = recv_off_cp(s2, this->rank);
        dyn_args.lane_rows[Sd] = (int32_t)rows;
        Sd++;
      };
      // same S materializing lanes as the bucket chain (order is irrelevant
      // to correctness here; kept for the staggered poll start heuristic)
      add_lane(this->rank);
      for (int dl = 1; dl < L; dl++) {
        add_lane(my_node * L + (my_lr - dl + L) % L);
      }
      for (int gi = 1; gi < NN; gi++) {
        add_lane(((my_node - gi + NN) % NN) * L + my_lr);
      }
      dyn_args.n_lanes = Sd;
      FLUX_CHECK_GT(Sd, 0) << "arrival-dynamic receiver: empty C' image";
      CUDA_CHECK(cudaMemsetAsync(
          dyn_args.lane_cursor, 0, sizeof(int32_t) * (size_t)Sd, reduce_stream));
      // v2 slack-row hardening (16n b32+ livelock fix): -1-fill the WHOLE
      // token_of buffer (a2av_recv_rows_ rows >= cpr >= image rows >= every
      // lane extent — the FLUX_CHECK chain above) BEFORE the map kernel
      // writes the CSR-covered rows, so any slack row the kernel walks reads
      // the sentinel and is skipped instead of aliasing a token (the
      // garbage-t OOB-atomicSub livelock class). 0xFF bytes == int32 -1.
      CUDA_CHECK(cudaMemsetAsync(
          this->a2av_token_of_row_.data_ptr(),
          0xFF,
          this->a2av_token_of_row_.nbytes(),
          reduce_stream));
      // token map + remain counters ride the existing plan kernel, ordered
      // before the persistent kernel on the same stream
      a2av_lane_token_map(
          A2AVLaneTokenMapArguments{
              reduce_csr->at(0).data_ptr<int32_t>(),
              reduce_csr->at(1).data_ptr<int32_t>(),
              ntok_local,
              this->a2av_token_of_row_.data_ptr<int32_t>(),
              dyn_args.remain},
          reduce_stream);
      a2av_combine_dyn_reduce(dyn_args, flux_dtype, reduce_stream);
    }

    // The ladders are enqueued INTERLEAVED PER SPLIT, in dependency order
    // (inter -> intra -> gateway -> reduce): under CUDA_DEVICE_MAX_CONNECTIONS=1
    // all streams multiplex one front-end channel and a pending wait can block
    // later-enqueued ops, so the enqueue order must itself be an executable
    // pipelined schedule. Within a split the pack kernel flips remote-node flags
    // first (production order) and the own-node flag last, so the inter waits
    // sit ahead of the intra wait; the reduce waits depend on this rank's own
    // gateway forwards, which are enqueued just before them.
    for (int sid = 0; sid < this->n_split; sid++) {
      if (this->a2av_compress_) {
        // conv ladder: behind the pack chunk flag per target node (produced
        // remote-first). My sub-chunk for dest (tn, dl) converges on local
        // gateway (my_node, dl): self sub-chunk is a CE memcpy into my own
        // conv panel, peers get one contiguous putmem_signal each (NVLink CE).
        // Every (peer, tn) pair signals every split, payload or not.
        cudaStream_t conv_stream = this->a2av_conv_stream_.value();
        for (int gi = 0; gi < NN - 1; gi++) {
          int tn = sched_remote[gi];  // production schedule (ring / size-sorted)
          CU_CHECK(CUStreamWaitValue(
              conv_stream,
              (CUdeviceptr)(this->group_flags.get() + tn * this->n_split + sid),
              1,
              CU_STREAM_WAIT_VALUE_GEQ));
          for (int di = 0; di < L; di++) {
            int dl = (my_lr + di) % L;  // self first, then rotation (no incast)
            int d = tn * L + dl;
            int gw = my_node * L + dl;
            int64_t rows = chunk_at(this->rank, d);
            int64_t coff = conv_off(dl, tn, my_lr);
            uint64_t *slot =
                conv_sig + ((int64_t)my_lr * NN + tn) * this->n_split + sid;
            if (gw == this->rank) {
              if (rows > 0) {
                CUDA_CHECK(cudaMemcpyAsync(
                    conv_ptr(sid, coff),
                    send_ptr(sid, send_off[d]),
                    rows * row_bytes,
                    cudaMemcpyDeviceToDevice,
                    conv_stream));
              }
              nvshmemx_signal_op_on_stream(
                  slot, this->run_id_, NVSHMEM_SIGNAL_SET, gw, conv_stream);
            } else if (rows > 0) {
              flux_rs_put_signal(
                  conv_ptr(sid, coff),
                  send_ptr(sid, send_off[d]),
                  rows * row_bytes,
                  slot,
                  this->run_id_,
                  NVSHMEM_SIGNAL_SET,
                  gw,
                  conv_stream, this->local_world_size, this->node_idx);
            } else {
              nvshmemx_signal_op_on_stream(
                  slot, this->run_id_, NVSHMEM_SIGNAL_SET, gw, conv_stream);
            }
          }
        }
        // wire ladder: behind the pre-reduce kernel's (tn, sid) wire flag, one
        // direct putmem_signal per remote node into the same-lr destination's
        // recv panel (C' image) -- no destination gateway hop. The (rank, sid)
        // slot at the destination keeps exactly one writer: me.
        for (int gi = 0; gi < NN - 1; gi++) {
          int tn = sched_remote[gi];  // production schedule (ring / size-sorted)
          int d = tn * L + my_lr;
          // S <= NN-1: per-split round-robin (gi % S) — bit-identical to the
          // canonical parity split at S=2 and to the first nstreams ladder.
          // S > NN-1: the split loop only has NN-1 puts, so extra lanes are
          // dead under gi % S; spread by GLOBAL cell index (sid*(NN-1)+gi)
          // instead, so puts from different splits round-robin over all S
          // lanes (knee-finding past NN-1; 2026-08-23 l1 16n campaign,
          // binaries carry FLUX_A2AV_RS_WIRE_XSPREAD_TAG).
          int wire_lane = 0;
          if (this->rs_wire_streams_ > 1) {
            wire_lane =
                (this->rs_wire_streams_ <= NN - 1)
                    ? (gi % this->rs_wire_streams_)
                    : (int)(((int64_t)sid * (NN - 1) + gi) % this->rs_wire_streams_);
          }
          cudaStream_t wstream =
              (wire_lane > 0) ? (cudaStream_t)this->internode_streams2_[wire_lane - 1]
                              : (cudaStream_t)this->internode_stream;
          CU_CHECK(CUStreamWaitValue(
              wstream,
              (CUdeviceptr)(this->wire_flags_.get() + tn * this->n_split + sid),
              1,
              CU_STREAM_WAIT_VALUE_GEQ));
          int64_t rows = U[d * NN + my_node];
          if (rows > 0) {
            flux_rs_put_signal(
                recv_ptr(sid, recv_off_cp(this->rank, d)),
                wire_ptr_at(sid, wire_seg_off(tn)),
                rows * row_bytes,
                recv_sig + this->rank * this->n_split + sid,
                this->run_id_,
                NVSHMEM_SIGNAL_SET,
                d,
                wstream, this->local_world_size, this->node_idx);
          } else {
            nvshmemx_signal_op_on_stream(
                recv_sig + this->rank * this->n_split + sid,
                this->run_id_,
                NVSHMEM_SIGNAL_SET,
                d,
                wstream);
          }
        }
      } else if (NN > 1) {
        // inter-node: ONE aggregated put per (remote node, split) to the
        // same-local-rank gateway there, consumed in the pack kernel's chunk
        // production order (node_idx+1 ascending -- no consumer schedule exists
        // in the combine, and matching production order avoids head-of-line
        // blocking; the rotation staggers sources across gateways, no incast).
        for (int gi = 0; gi < NN - 1; gi++) {
          int tn = (my_node + 1 + gi) % NN;
          int g = dist_env.local_rank_to_global_rank(my_lr, tn);
          CU_CHECK(CUStreamWaitValue(
              this->internode_stream,
              (CUdeviceptr)(this->group_flags.get() + tn * this->n_split + sid),
              1,
              CU_STREAM_WAIT_VALUE_GEQ));
          int64_t rows = node_chunk(this->rank, tn);
          if (rows > 0) {
            flux_rs_put_signal(
                stage_ptr(sid, seg_off(tn, my_lr, my_node)),
                send_ptr(sid, send_off[tn * L]),
                rows * row_bytes,
                arrival_sig + my_node * this->n_split + sid,
                this->run_id_,
                NVSHMEM_SIGNAL_SET,
                g,
                this->internode_stream, this->local_world_size, this->node_idx);
          } else {
            nvshmemx_signal_op_on_stream(
                arrival_sig + my_node * this->n_split + sid,
                this->run_id_,
                NVSHMEM_SIGNAL_SET,
                g,
                this->internode_stream);
          }
        }
      }
      // intra-node: behind the own-node chunk flag; self chunk is a local CE
      // copy, peers get one contiguous putmem_signal each (CE over NVLink for
      // same-node PEs). Every pair signals every split, payload or not.
      CU_CHECK(CUStreamWaitValue(
          intra_stream,
          (CUdeviceptr)(this->group_flags.get() + my_node * this->n_split + sid),
          1,
          CU_STREAM_WAIT_VALUE_GEQ));
      if (chunk_at(this->rank, this->rank) > 0) {
        CUDA_CHECK(cudaMemcpyAsync(
            recv_ptr(sid, recv_off_active(this->rank, this->rank)),
            send_ptr(sid, send_off[this->rank]),
            chunk_at(this->rank, this->rank) * row_bytes,
            cudaMemcpyDeviceToDevice,
            intra_stream));
      }
      nvshmemx_signal_op_on_stream(
          recv_sig + this->rank * this->n_split + sid,
          this->run_id_,
          NVSHMEM_SIGNAL_SET,
          this->rank,
          intra_stream);
      for (int dl = 1; dl < L; dl++) {
        int d = dist_env.local_rank_to_global_rank((my_lr - dl + L) % L, my_node);
        int64_t rows = chunk_at(this->rank, d);
        if (rows > 0) {
          flux_rs_put_signal(
              recv_ptr(sid, recv_off_active(this->rank, d)),
              send_ptr(sid, send_off[d]),
              rows * row_bytes,
              recv_sig + this->rank * this->n_split + sid,
              this->run_id_,
              NVSHMEM_SIGNAL_SET,
              d,
              intra_stream, this->local_world_size, this->node_idx);
        } else {
          nvshmemx_signal_op_on_stream(
              recv_sig + this->rank * this->n_split + sid,
              this->run_id_,
              NVSHMEM_SIGNAL_SET,
              d,
              intra_stream);
        }
      }
      if (gw_path) {
        // gateway forwards: per source node behind the arrival signal (zero-SM
        // front-end wait, cannot deadlock against the spinning GEMM); forwarded
        // sub-chunks are indistinguishable from direct puts at the destination.
        // Own sub-chunk is a local CE copy + self signal.
        for (int dn = 1; dn < NN; dn++) {
          int ns = (my_node + dn) % NN;
          int s = dist_env.local_rank_to_global_rank(my_lr, ns);
          CU_CHECK(CUStreamWaitValue64(
              gateway_stream,
              (CUdeviceptr)(arrival_sig + ns * this->n_split + sid),
              this->run_id_,
              CU_STREAM_WAIT_VALUE_GEQ));
          const int64_t seg = seg_off(my_node, my_lr, ns);
          for (int dl = 0; dl < L; dl++) {
            int d = dist_env.local_rank_to_global_rank((my_lr - dl + L) % L, my_node);
            int64_t sub_rows = chunk_at(s, d);
            int64_t within = 0;
            for (int d2 = my_node * L; d2 < d; d2++) {
              within += chunk_at(s, d2);
            }
            if (d == this->rank) {
              if (sub_rows > 0) {
                CUDA_CHECK(cudaMemcpyAsync(
                    recv_ptr(sid, recv_off_of(s, this->rank)),
                    stage_ptr(sid, seg + within),
                    sub_rows * row_bytes,
                    cudaMemcpyDeviceToDevice,
                    gateway_stream));
              }
              nvshmemx_signal_op_on_stream(
                  recv_sig + s * this->n_split + sid,
                  this->run_id_,
                  NVSHMEM_SIGNAL_SET,
                  this->rank,
                  gateway_stream);
            } else if (sub_rows > 0) {
              flux_rs_put_signal(
                  recv_ptr(sid, recv_off_of(s, d)),
                  stage_ptr(sid, seg + within),
                  sub_rows * row_bytes,
                  recv_sig + s * this->n_split + sid,
                  this->run_id_,
                  NVSHMEM_SIGNAL_SET,
                  d,
                  gateway_stream, this->local_world_size, this->node_idx);
            } else {
              nvshmemx_signal_op_on_stream(
                  recv_sig + s * this->n_split + sid,
                  this->run_id_,
                  NVSHMEM_SIGNAL_SET,
                  d,
                  gateway_stream);
            }
          }
        }
      }
      // per-split reduce (legacy gate, a2av_eager_ off): gate on the per-source
      // recv signals of the split, then one memory-bound kernel folds them into
      // the output column window. Under compress only the C' image's
      // L + NN - 1 lanes materialize (own-node ranks + the same-lr rank of
      // each remote node) -- waiting a lane that never signals would hang.
      // With a2av_eager_ the persistent kernel launched above already consumes
      // the signals in arrival order.
      if (this->a2av_lane_chain_) {
        // LANE-CHAIN receiver (Slipstream v2b): per-lane front-end waits in
        // EXPECTED arrival order release per-lane scatter-adds into the fp32
        // accumulator; finalize casts after the last lane. Waits' writers are
        // remote ranks (NIC writes) or the intra ladder enqueued above --
        // NR-02 Class-B safe. The reduce work drips in behind each arrival;
        // the post-last-arrival tail is one lane's fold + the cast.
        FLUX_CHECK(this->a2av_compress_);
        FLUX_CHECK_EQ(this->n_split, 1);
        const int64_t ntok_local = cpr / this->topk;
        CUDA_CHECK(cudaStreamWaitEvent(reduce_stream, this->staging_reset_event));
        CUDA_CHECK(cudaMemsetAsync(
            this->a2av_scratch_fp32_.data_ptr(),
            0,
            (size_t)ntok_local * this->n_dim * sizeof(float),
            reduce_stream));
        a2av_lane_token_map(
            A2AVLaneTokenMapArguments{
                reduce_csr->at(0).data_ptr<int32_t>(),
                reduce_csr->at(1).data_ptr<int32_t>(),
                ntok_local,
                this->a2av_token_of_row_.data_ptr<int32_t>()},
            reduce_stream);
        auto chain_lane = [&](int s) {
          const int64_t rows = chunk_cp(s, this->rank);
          if (rows <= 0) {
            return;  // zero-row lanes still signal; nothing to read or wait for
          }
          CU_CHECK(CUStreamWaitValue64(
              reduce_stream,
              (CUdeviceptr)(recv_sig + s * this->n_split + sid),
              this->run_id_,
              CU_STREAM_WAIT_VALUE_GEQ));
          a2av_combine_lane_reduce(
              A2AVLaneReduceArguments{
                  this->a2av_recv_panel_.data_ptr(),
                  this->a2av_token_of_row_.data_ptr<int32_t>(),
                  (float *)this->a2av_scratch_fp32_.data_ptr(),
                  recv_off_cp(s, this->rank),
                  rows,
                  (int)this->n_dim,
                  get_a2av_reduce_blocks(),
                  /*use_atomic=*/(s / L == my_node) ? 1 : 0},
              flux_dtype,
              reduce_stream);
        };
        // expected arrival order: sender node n2 reaches me at ITS ring step
        // (my_node - n2 - 1) mod NN, so the nearest ring predecessor lands
        // first -> remote same-lr lanes in descending ring. Own-node lanes
        // follow the wave policy: LAST under the canon (own wave at GEMM
        // end), FIRST under FLUX_A2AV_RS_OWN_FIRST (own wave computed first,
        // intra arrives early — the receiver-overlap precondition).
        const bool own_first_chain = get_a2av_rs_own_wave_first() != 0;
        auto chain_own = [&]() {
          chain_lane(this->rank);
          for (int dl = 1; dl < L; dl++) {
            chain_lane(my_node * L + (my_lr - dl + L) % L);
          }
        };
        if (own_first_chain) {
          chain_own();
        }
        for (int gi = 1; gi < NN; gi++) {
          const int n2 = (my_node - gi + NN) % NN;
          chain_lane(n2 * L + my_lr);
        }
        if (!own_first_chain) {
          chain_own();
        }
        a2av_combine_finalize(
            A2AVFinalizeArguments{
                (float const *)this->a2av_scratch_fp32_.data_ptr(),
                output.data_ptr(),
                ntok_local,
                (int)this->n_dim,
                get_a2av_reduce_blocks()},
            flux_dtype,
            reduce_stream);
      } else if (this->a2av_bucket_ && !this->a2av_recv_dyn_) {
        // COMPLETION-BUCKETED receiver (Slipstream gen-10): a ~us plan-time
        // bucket sort of the reduce CSR by completion chain position (on the
        // reduce stream, inside the timed bracket), then per-lane front-end
        // waits each releasing a register-CSR fold of exactly the tokens
        // that lane completes. (Under FLUX_A2AV_RS_RECV_DYN the persistent
        // arrival-dynamic kernel launched above consumes the same lanes and
        // this whole branch is skipped.) Own-node lanes chain FIRST -- consumption
        // order only, the wire keeps the canon own-last production: own rows
        // are ready at GEMM end, which precedes the remote drain, so tokens
        // complete at their last REMOTE arrival and the fold spreads over
        // the window at wait-all's 1x byte budget. Sequential waits on one
        // stream give the bucket-prefix guarantee; skipping zero-row lanes
        // is safe (no CSR entry, so no token can complete there).
        FLUX_CHECK(this->a2av_compress_);
        FLUX_CHECK_EQ(this->n_split, 1);
        const int64_t ntok_local = cpr / this->topk;
        int chain_lane_of[kA2AVMaxWorld];
        int S = 0;
        chain_lane_of[S++] = this->rank;
        for (int dl = 1; dl < L; dl++) {
          chain_lane_of[S++] = my_node * L + (my_lr - dl + L) % L;
        }
        for (int gi = 1; gi < NN; gi++) {
          const int n2 = (my_node - gi + NN) % NN;
          chain_lane_of[S++] = n2 * L + my_lr;
        }
        FLUX_CHECK_LE(S, kA2AVMaxWorld);
        // per-iteration lane tables, run-parity double-buffered against the
        // pinned buffer being rewritten under a still-in-flight async copy
        const int64_t lane_words = 2 * ((int64_t)W + 1);
        const int64_t lane_base = (int64_t)(this->run_id_ & 1) * lane_words;
        int32_t *lanes_h = this->a2av_bucket_lanes_h_.data_ptr<int32_t>() + lane_base;
        int32_t *lanes_d = this->a2av_bucket_lanes_d_.data_ptr<int32_t>() + lane_base;
        int32_t *lane_off_h = lanes_h;             // [W + 1]
        int32_t *chain_pos_h = lanes_h + (W + 1);  // [W]
        for (int s2 = 0; s2 <= W; s2++) {
          lane_off_h[s2] = (int32_t)recv_off_cp(s2, this->rank);
        }
        for (int s2 = 0; s2 < W; s2++) {
          chain_pos_h[s2] = 0;
        }
        for (int k = 0; k < S; k++) {
          chain_pos_h[chain_lane_of[k]] = k;
        }
        CUDA_CHECK(cudaMemcpyAsync(
            lanes_d,
            lanes_h,
            sizeof(int32_t) * (size_t)(2 * W + 1),
            cudaMemcpyHostToDevice,
            reduce_stream));
        int32_t *bucket_meta = this->a2av_bucket_meta_.data_ptr<int32_t>();
        int32_t *bucket_cnt = bucket_meta;
        int32_t *bucket_cur = bucket_meta + kA2AVMaxWorld;
        int32_t *bucket_ptr_d = bucket_meta + 2 * kA2AVMaxWorld;
        CUDA_CHECK(
            cudaMemsetAsync(bucket_cnt, 0, sizeof(int32_t) * (size_t)S, reduce_stream));
        a2av_bucket_map(
            A2AVBucketMapArguments{
                reduce_csr->at(0).data_ptr<int32_t>(),
                reduce_csr->at(1).data_ptr<int32_t>(),
                lanes_d,
                lanes_d + (W + 1),
                W,
                S,
                ntok_local,
                this->a2av_bucket_comp_.data_ptr<int32_t>(),
                bucket_cnt},
            reduce_stream);
        a2av_bucket_scan(
            A2AVBucketScanArguments{bucket_cnt, S, bucket_ptr_d, bucket_cur},
            reduce_stream);
        a2av_bucket_scatter(
            A2AVBucketScatterArguments{
                this->a2av_bucket_comp_.data_ptr<int32_t>(),
                bucket_ptr_d,
                ntok_local,
                bucket_cur,
                this->a2av_bucket_tok_.data_ptr<int32_t>()},
            reduce_stream);
        for (int k = 0; k < S; k++) {
          const int s2 = chain_lane_of[k];
          if (chunk_cp(s2, this->rank) <= 0) {
            continue;  // zero-row lane: still signals, but its bucket is empty
          }
          CU_CHECK(CUStreamWaitValue64(
              reduce_stream,
              (CUdeviceptr)(recv_sig + s2 * this->n_split + sid),
              this->run_id_,
              CU_STREAM_WAIT_VALUE_GEQ));
          a2av_combine_bucket_reduce(
              A2AVBucketReduceArguments{
                  this->a2av_recv_panel_.data_ptr(),
                  reduce_csr->at(0).data_ptr<int32_t>(),
                  reduce_csr->at(1).data_ptr<int32_t>(),
                  bucket_ptr_d,
                  this->a2av_bucket_tok_.data_ptr<int32_t>(),
                  k,
                  output.data_ptr(),
                  this->a2av_recv_rows_,
                  (int)this->n_dim,
                  (int)n_per,
                  sid,
                  get_a2av_reduce_blocks()},
              flux_dtype,
              reduce_stream);
        }
      } else if (!this->a2av_eager_ && !this->a2av_recv_dyn_) {
        for (int s = 0; s < W; s++) {
          if (this->a2av_compress_ && s / L != my_node && s % L != my_lr) {
            continue;  // lane never materializes under C'
          }
          CU_CHECK(CUStreamWaitValue64(
              reduce_stream,
              (CUdeviceptr)(recv_sig + s * this->n_split + sid),
              this->run_id_,
              CU_STREAM_WAIT_VALUE_GEQ));
        }
        if (this->a2av_compress_) {
          A2AVCombineCSRReduceArguments csr_args{
              .recv_panel = this->a2av_recv_panel_.data_ptr(),
              .red_ptr = reduce_csr->at(0).data_ptr<int32_t>(),
              .red_row = reduce_csr->at(1).data_ptr<int32_t>(),
              .output = output.data_ptr(),
              .panel_rows = this->a2av_recv_rows_,
              .ntokens_local = cpr / this->topk,
              .n = this->n_dim,
              .n_per = (int)n_per,
              .sid = sid,
              .threadblock_count = get_a2av_reduce_blocks()};
          a2av_combine_csr_reduce(csr_args, flux_dtype, reduce_stream);
        } else {
          reduce_args.sid = sid;
          a2av_combine_reduce(reduce_args, flux_dtype, reduce_stream);
        }
      }
    }

    // tail joins: everything the epoch produced must reach the gather-rs stream
    // before the caller's gather_rs_done_event / closing barrier, covering
    // panel + staging reuse in the next iteration
    CUDA_CHECK(cudaEventRecord(this->a2av_intra_done_, intra_stream));
    CUDA_CHECK(cudaStreamWaitEvent(stream_raw, this->a2av_intra_done_));
    if (NN > 1) {
      CUDA_CHECK(cudaEventRecord(this->a2av_inter_done_, this->internode_stream));
      CUDA_CHECK(cudaStreamWaitEvent(stream_raw, this->a2av_inter_done_));
      for (auto &s2 : this->internode_streams2_) {
        CUDA_CHECK(cudaEventRecord(this->a2av_inter2_done_, s2));
        CUDA_CHECK(cudaStreamWaitEvent(stream_raw, this->a2av_inter2_done_));
      }
    }
    if (gw_path) {
      CUDA_CHECK(cudaEventRecord(this->a2av_gateway_done_, gateway_stream));
      CUDA_CHECK(cudaStreamWaitEvent(stream_raw, this->a2av_gateway_done_));
    }
    if (this->a2av_compress_) {
      CUDA_CHECK(cudaEventRecord(this->a2av_conv_done_, this->a2av_conv_stream_.value()));
      CUDA_CHECK(cudaStreamWaitEvent(stream_raw, this->a2av_conv_done_));
      CUDA_CHECK(cudaEventRecord(this->a2av_prered_done_, this->a2av_prered_stream_.value()));
      CUDA_CHECK(cudaStreamWaitEvent(stream_raw, this->a2av_prered_done_));
    }
    CUDA_CHECK(cudaEventRecord(this->a2av_reduce_done_, reduce_stream));
    CUDA_CHECK(cudaStreamWaitEvent(stream_raw, this->a2av_reduce_done_));
    return output;
  }

  // Combine/compress plan as ONE C++ call for the a2av_hier TopkReduceScatter
  // path (2026-08-22, plan eager-juggling-glacier Stage 2b; mirrors
  // GemmGroupedV2GatherRSOp::derive_combine_meta): pack/reduce gather
  // indices (+ the SORT-FREE compress wire/reduce CSRs) derived from this
  // iteration's splits/routing/splits_per_source — the epic/pll l01
  // harnesses call it inside the rule-5 plan bracket instead of letting
  // run() self-build with the slow sort-based builder on the timed path.
  // ep geometry follows run(): nex_total = ep_nexperts * ep_world_size,
  // e_loc = nex_total / world_size, ep_start = rank * e_loc.
  std::vector<torch::Tensor>
  derive_combine_meta(
      torch::Tensor splits_gpu,
      torch::Tensor routing_idx,
      torch::Tensor splits_per_source,
      c10::optional<torch::Tensor> a2av_unique_counts) {
    (void)get_int_from_env("FLUX_A2AV_RS_DERIVE_COMBINE_RS_TAG", 0);
    (void)get_int_from_env("FLUX_A2AV_RS_BLOCKING_WIRE_DEFAULT_TAG", 0);  // 2026-08-22
    FLUX_CHECK(this->a2av_hier)
        << "TopkReduceScatterOp::derive_combine_meta requires the a2av_hier ctor flag";
    CHECK_INPUT(routing_idx, at::ScalarType::Int);
    CHECK_INPUT(splits_gpu, at::ScalarType::Int);
    const int64_t nex_total = (int64_t)this->ep_nexperts * this->ep_world_size;
    const int64_t e_loc = nex_total / this->world_size;
    CHECK_1D(splits_gpu, nex_total);
    FLUX_CHECK(splits_per_source.device().is_cpu()) << "splits_per_source must be CPU";
    CHECK_2D(splits_per_source, this->world_size, nex_total);
    FLUX_CHECK(splits_per_source.scalar_type() == at::ScalarType::Int);
    FLUX_CHECK(splits_per_source.is_contiguous());
    const int64_t m_full = routing_idx.numel();
    FLUX_CHECK_DIV(m_full, (int64_t)this->world_size * this->topk);
    const int32_t *cnt = splits_per_source.data_ptr<int32_t>();
    const int64_t ep_start = (int64_t)this->rank * e_loc;
    int64_t m_this_ep = 0;
    for (int h = 0; h < this->world_size; h++) {
      for (int64_t e = ep_start; e < ep_start + e_loc; e++) {
        m_this_ep += cnt[h * nex_total + e];
      }
    }
    auto [pack_idx, reduce_idx] = build_a2av_combine_indices(
        routing_idx,
        splits_gpu,
        splits_per_source,
        m_this_ep,
        m_full,
        this->world_size,
        this->rank,
        nex_total,
        e_loc,
        ep_start);
    std::vector<torch::Tensor> out{pack_idx, reduce_idx};
    if (this->a2av_compress_) {
      FLUX_CHECK(a2av_unique_counts.has_value())
          << "a2av_compress derive requires a2av_unique_counts ([W, nnodes] int32 CPU)";
      auto [wp, wc, rp, rr] = build_a2av_compress_indices_fast(
          routing_idx,
          splits_gpu,
          splits_per_source,
          a2av_unique_counts.value(),
          m_full,
          this->world_size,
          this->nnodes,
          this->local_world_size,
          this->rank,
          nex_total,
          e_loc,
          this->topk);
      static const bool kCheckIdentity =
          get_int_from_env("FLUX_A2AV_RS_CHECK_IDENTITY", 0) != 0;
      if (kCheckIdentity) {
        auto [wp_ref, wc_ref, rp_ref, rr_ref] = build_a2av_compress_indices(
            routing_idx,
            splits_gpu,
            splits_per_source,
            a2av_unique_counts.value(),
            m_full,
            this->world_size,
            this->nnodes,
            this->local_world_size,
            this->rank,
            nex_total,
            e_loc,
            this->topk);
        FLUX_CHECK(torch::equal(wp, wp_ref)) << "RS compress plan wire_ptr identity mismatch";
        FLUX_CHECK(torch::equal(wc, wc_ref)) << "RS compress plan wire_copy identity mismatch";
        FLUX_CHECK(torch::equal(rp, rp_ref)) << "RS compress plan red_ptr identity mismatch";
        FLUX_CHECK(torch::equal(rr, rr_ref)) << "RS compress plan red_row identity mismatch";
      }
      out.push_back(wp);
      out.push_back(wc);
      out.push_back(rp);
      out.push_back(rr);
    }
    return out;
  }
  torch::Tensor
  run(std::vector<torch::Tensor> gemm_outs,  // of group_size
      c10::optional<torch::Tensor> output_,
      int ep_start,
      int ep_nexperts,
      torch::Tensor splits,
      torch::Tensor routing_idx,
      c10::optional<std::vector<torch::Tensor>> output_vec_scales,
      int num_thread_blocks,
      intptr_t cp_stream,
      c10::optional<torch::Tensor> splits_per_source = c10::nullopt,
      c10::optional<torch::Tensor> pack_index = c10::nullopt,
      c10::optional<torch::Tensor> reduce_index = c10::nullopt,
      c10::optional<torch::Tensor> unique_counts = c10::nullopt,
      c10::optional<std::vector<torch::Tensor>> wire_csr = c10::nullopt,
      c10::optional<std::vector<torch::Tensor>> reduce_csr = c10::nullopt) {
    at::cuda::CUDAStream stream =
        at::cuda::getStreamFromExternal((cudaStream_t)cp_stream, at::cuda::current_device());
    at::cuda::CUDAStreamGuard _(stream);
    CHECK_INPUT(routing_idx, at::ScalarType::Int);
    CHECK_INPUT(splits, at::ScalarType::Int);
    int N = this->n_dim;
    int m_full = routing_idx.size(0);
    int ntokens = m_full / this->topk;
    int ntokens_per_rank = ntokens / this->world_size;
    int ntokens_out = this->do_all_reduce ? ntokens : ntokens_per_rank;
    FLUX_CHECK_GE(gemm_outs.size(), 1);
    FLUX_CHECK_LE(gemm_outs.size(), kMaxNumGroups);
    auto dtype = gemm_outs[0].scalar_type();

    auto output = output_.value_or(empty_with_uninitialized_data(
        std::vector<int64_t>{ntokens_out, N}, gemm_outs[0].options()));
    CHECK_TYPE(output, dtype);
    CHECK_2D(output, ntokens_out, N);

    if (this->a2av_hier) {
      FLUX_CHECK_EQ((int)gemm_outs.size(), 1) << "a2av_hier supports a single weight group";
      FLUX_CHECK(splits_per_source.has_value())
          << "a2av_hier requires splits_per_source ([W, nexperts] int32 CPU)";
      auto const &cnt_t = splits_per_source.value();
      // v2b in-window: absent routing-plan indices self-build here, on the
      // timed critical path (same v1 placement as the gather_rs entry above --
      // per-iteration routing means these can never be assumed precomputed)
      const int64_t nex_total = (int64_t)this->ep_nexperts * this->ep_world_size;
      const int64_t e_loc = nex_total / this->world_size;
      if (this->a2av_compress_ && !wire_csr.has_value()) {
        FLUX_CHECK(unique_counts.has_value())
            << "a2av_hier_compress in-window build requires unique_counts ([W, nnodes] int32 CPU)";
        auto [wp, wc, rp, rr] = build_a2av_compress_indices(
            routing_idx, splits, cnt_t, unique_counts.value(), m_full,
            this->world_size, this->nnodes, this->local_world_size, this->rank,
            nex_total, e_loc, this->topk);
        wire_csr = std::vector<torch::Tensor>{wp, wc};
        reduce_csr = std::vector<torch::Tensor>{rp, rr};
      }
      torch::Tensor pack_idx_t, reduce_idx_t;
      if (pack_index.has_value() || reduce_index.has_value()) {
        FLUX_CHECK(pack_index.has_value() && reduce_index.has_value())
            << "pass both pack_index and reduce_index or neither";
        pack_idx_t = pack_index.value();
        reduce_idx_t = reduce_index.value();
      } else {
        std::tie(pack_idx_t, reduce_idx_t) = build_a2av_combine_indices(
            routing_idx, splits, cnt_t, gemm_outs[0].size(0), m_full,
            this->world_size, this->rank, nex_total, e_loc,
            (int64_t)this->rank * e_loc);
      }
      return run_a2av_hier(
          gemm_outs[0],
          output,
          splits_per_source.value(),
          pack_idx_t,
          reduce_idx_t,
          unique_counts,
          wire_csr,
          reduce_csr,
          output_vec_scales,
          m_full,
          num_thread_blocks,
          (cudaStream_t)cp_stream);
    }

    TopKReduceGatherRSV2Arguments args{
        .output_ptr = (void *)output.data_ptr(),
        .splits = splits.data_ptr<int>(),
        .routing_idx = routing_idx.data_ptr<int>(),
        .m_full = m_full,
        .n = N,
        .nexperts = ep_nexperts * this->ep_world_size,
        .topk = this->topk,
        .input_groups = (int)gemm_outs.size(),
        .do_all_reduce = this->do_all_reduce,
        .use_read_mode = this->use_read_mode,
        .threadblock_count = num_thread_blocks,
        .tile_size_m = kTileSizeM,
        .tile_size_n = combine_tile_n(this->n_dim, this->n_split),
        .rank = this->rank,
        .world_size = this->world_size,
        .n_split = this->n_split,
        .barrier = this->barrier_dev_ptrs,
        .reduce_ptrs = (void **)this->reduce_buffer_dptrs.data_ptr(),
        .tile_barrier_ptrs = (int **)this->tile_barrier_dptrs.data_ptr(),
        .nnodes = this->nnodes,
        .node_idx = this->node_idx,
        .local_rank = this->local_rank,
        .local_world_size = this->local_world_size,
        .staging_rows = (int)(this->max_m / this->topk / this->world_size),
        .staging_send = this->nnodes > 1 ? this->staging_send.data_ptr() : nullptr,
        .group_flags = this->nnodes > 1 ? this->group_flags.get() : nullptr,
        .group_counters = this->nnodes > 1 ? this->group_counters.get() : nullptr,
    };
    for (int i = 0; i < gemm_outs.size(); i++) {
      args.input_ptrs[i] = (void *)gemm_outs[i].data_ptr();
      args.output_vec_scale_ptrs[i] =
          output_vec_scales.has_value() ? (float *)output_vec_scales->at(i).data_ptr() : nullptr;
    }
    cudaStream_t stream_raw = (cudaStream_t)cp_stream;
    if (this->nnodes > 1) {
      // per-run reset of the kernel->host chunk-ready flags/counters, published to the
      // internode stream via an event so its waits cannot observe stale values
      this->run_id_ += 1;
      const size_t flag_bytes = sizeof(int) * this->nnodes * this->n_split;
      CUDA_CHECK(cudaMemsetAsync(this->group_flags.get(), 0, flag_bytes, stream_raw));
      CUDA_CHECK(cudaMemsetAsync(this->group_counters.get(), 0, flag_bytes, stream_raw));
      CUDA_CHECK(cudaEventRecord(this->staging_reset_event, stream_raw));
      CUDA_CHECK(cudaStreamWaitEvent(this->internode_stream, this->staging_reset_event));
      for (auto &s2 : this->internode_streams2_) {
        CUDA_CHECK(cudaStreamWaitEvent(s2, this->staging_reset_event));
      }
    }
    auto output_dtype = from_torch_dtype(dtype);
    if (this->ep_world_size == 1) {
      topk_gather_rs_v2(args, output_dtype, (cudaStream_t)cp_stream);
    } else {
      ep_topk_gather_rs_v2(args, output_dtype, ep_start, ep_nexperts, (cudaStream_t)cp_stream);
    }
    if (this->nnodes > 1) {
      const int64_t rows = ntokens / this->world_size;  // runtime token rows per rank
      const int64_t n_per = N / this->n_split;
      const int64_t slot_bytes = (int64_t)args.staging_rows * n_per * output.element_size();
      const int64_t chunk_bytes = rows * n_per * output.element_size();
      char *send_base = (char *)this->staging_send.data_ptr();
      char *recv_base = (char *)this->staging_recv.data_ptr();
      uint64_t *sig_base = (uint64_t *)this->internode_signals.data_ptr();
      // sender side: as the kernel finishes staging each (remote node, split) chunk, push it
      // with one contiguous putmem_signal to the rank with the same local rank on that node.
      // same (sid, group) order as the kernel produces the chunks.
      for (int sid = 0; sid < this->n_split; sid++) {
        for (int gi = 0; gi < this->nnodes - 1; gi++) {
          int g = (this->node_idx + 1 + gi) % this->nnodes;
          int idx = g * this->n_split + sid;
          // FLUX_RS_WIRE_STREAMS lane spread (global (sid, gi) cell index,
          // same mapping as the a2av XSPREAD ladder); lane 0 = the original
          // internode stream, S=1 = the shipped single-stream schedule
          int wire_lane = 0;
          if (this->rs_wire_streams_ > 1) {
            wire_lane =
                (int)(((int64_t)sid * (this->nnodes - 1) + gi) % this->rs_wire_streams_);
          }
          cudaStream_t wstream =
              (wire_lane > 0) ? (cudaStream_t)this->internode_streams2_[wire_lane - 1]
                              : (cudaStream_t)this->internode_stream;
          CU_CHECK(CUStreamWaitValue(
              wstream,
              (CUdeviceptr)(this->group_flags.get() + idx),
              1,
              CU_STREAM_WAIT_VALUE_GEQ));
          flux_rs_put_signal(
              recv_base + (int64_t)(this->node_idx * this->n_split + sid) * slot_bytes,
              send_base + (int64_t)idx * slot_bytes,
              chunk_bytes,
              sig_base + this->node_idx * this->n_split + sid,
              this->run_id_,
              NVSHMEM_SIGNAL_SET,
              /*pe=*/g * this->local_world_size + this->local_rank,
              wstream, this->local_world_size, this->node_idx);
        }
      }
      // join the extra lanes into the original internode stream so no lane
      // is ever less-ordered than the shipped single-stream schedule
      for (auto &s2 : this->internode_streams2_) {
        CUDA_CHECK(cudaEventRecord(this->a2av_inter2_done_, s2));
        CUDA_CHECK(cudaStreamWaitEvent(this->internode_stream, this->a2av_inter2_done_));
      }
      // receiver side: wait for every remote node's partial of my token shard, then
      // accumulate it into the output (own-node contribution was written by the kernel)
      for (int sid = 0; sid < this->n_split; sid++) {
        for (int m = 0; m < this->nnodes; m++) {
          if (m == this->node_idx) {
            continue;
          }
          nvshmemx_signal_wait_until_on_stream(
              sig_base + m * this->n_split + sid, NVSHMEM_CMP_GE, this->run_id_, stream_raw);
        }
        internode_reduce_gather_rs(
            output.data_ptr(),
            this->staging_recv.data_ptr(),
            output_dtype,
            this->nnodes,
            this->node_idx,
            this->n_split,
            sid,
            rows,
            n_per,
            N,
            args.staging_rows,
            stream_raw);
      }
    }
    if (this->do_all_reduce) {
      cudaMemcpyAsync(
          output.data_ptr(),
          this->reduce_buffer.data_ptr(),
          ntokens * this->n_dim * output.element_size(),
          cudaMemcpyDeviceToDevice,
          (cudaStream_t)cp_stream);
    }
    return output;
  }

  void
  reset_buffer() {
    if (this->tile_barrier.defined()) {
      this->tile_barrier.zero_();
    }
  }
};

/// This class only runs the basic grouped_gemm, it is mainly used for testing
class GemmGroupedV2GatherRSOp::GemmGroupedV2GatherRSOpImpl {
 private:
  std::shared_ptr<Group> tp_group;
  int32_t ep_nexperts;
  int32_t ep_start;
  const int32_t total_num_experts;
  int32_t max_m;
  int32_t n_dim;
  int32_t topk;
  at::ScalarType output_dtype;
  int32_t max_input_groups;
  int32_t rank;
  int32_t world_size;     // the total world size
  int32_t tp_world_size;  // the world size of tensor parallel
  int32_t ep_world_size;  // the world size of expert parallel
  int32_t nnodes;
  int32_t local_rank;        // == rank when nnodes == 1
  int32_t local_world_size;  // == world_size when nnodes == 1
  int n_split;
  bool do_all_reduce;
  bool a2av_hier;
  bool a2av_compress;  // a2av_hier_compress ctor flag; false when nnodes == 1
  torch::Tensor barrier;
  std::vector<torch::Tensor> barriers;  // [local_world_size], indexed by local rank
  std::unique_ptr<TopkReduceScatterOp> topk_reduce_scatter_op = nullptr;
  // M-split waves (Slipstream v2, FLUX_A2AV_RS_MSPLIT): per-iteration wave
  // tables (pinned host arena -> device, one async H2D per forward)
  bool msplit_ = false;
  bool fused_pack_ = false;    // gen-8c: GEMM scatters the send panel directly
  int msplit_wave_nodes_ = 1;
  torch::Tensor msplit_host_;  // pinned int32 [2 * NN * E + NN]
  torch::Tensor msplit_dev_;   // device int32, same capacity
  torch::Tensor msplit_iota_;      // int32 [max_m] shared identity indices
  torch::Tensor msplit_inv_pack_;  // int32 [max_m] gemm row -> panel row
  cudaEvent_t msplit_h2d_event_ = nullptr;
  std::vector<int> msplit_wave_of_node_;  // schedule position -> cascade flag
  std::vector<int> msplit_node_order_;    // schedule position -> dest node

  torch::Tensor workspace;
  cudaEvent_t gemm_start_event;
  cudaEvent_t gather_rs_done_event;
  cudaStream_t gather_rs_stream;
  GroupBarrier group_barrier;

  int
  get_barrier_size(int problem_count) const {
    return pad_to(this->n_split, 128) * 2  // 1st: ready flag per tile, 2nd: counter per split
           + pad_to(problem_count, 128);   // counter for each problem gemm done tiles
  }

  void
  create_barriers() {
    // M-split waves need tile counters for up to nnodes * ep_nexperts problems.
    // The wave-flag/problem-counter regions reuse the legacy 128-padded slots
    // (n_waves <= nnodes <= 128, checked in the ctor), so the legacy layout is
    // byte-identical for everything it addresses; this is a superset sizing.
    const int problem_count = std::max<int>(
        this->n_split * this->ep_nexperts * this->max_input_groups,
        this->msplit_ ? this->nnodes * this->ep_nexperts * this->max_input_groups : 0);
    const int barrier_size = get_barrier_size(problem_count);
    if (this->barriers.empty()) {
      this->barriers = flux_create_tensor_list(
          std::vector<int64_t>{barrier_size}, at::ScalarType::Int, this->tp_group.get(), true);
      FLUX_CHECK_EQ((int)this->barriers.size(), this->local_world_size);
      this->barrier = this->barriers[this->local_rank];
    }
  }

 public:
  // Rule-5 in-window planning entry (2026-08-21): the combine (and compress)
  // index build as ONE host call — the same internal arithmetic-identity
  // builders the isolated-mode in-forward path uses. cnt/U are host metadata
  // the caller already holds (e.g. from the layer0 derive_routed_meta), so
  // the host offset tables cost no D2H sync; the chain launches back-to-back
  // with no interpreter gaps. Returns {pack_index, reduce_index} for
  // a2av_hier, plus {wire_ptr, wire_copy, red_ptr, red_row} when compress.
  std::vector<torch::Tensor>
  derive_combine_meta(
      torch::Tensor splits_gpu,
      torch::Tensor routing_idx,
      torch::Tensor splits_per_source,
      c10::optional<torch::Tensor> a2av_unique_counts) {
    (void)get_int_from_env("FLUX_A2AV_RS_DERIVE_COMBINE_TAG", 0);
    FLUX_CHECK(this->a2av_hier)
        << "derive_combine_meta requires the a2av_hier/a2av_hier_compress ctor flag";
    CHECK_INPUT(routing_idx, at::ScalarType::Int);
    CHECK_INPUT(splits_gpu, at::ScalarType::Int);
    CHECK_1D(splits_gpu, this->total_num_experts);
    FLUX_CHECK(splits_per_source.device().is_cpu()) << "splits_per_source must be CPU";
    CHECK_2D(splits_per_source, this->world_size, this->total_num_experts);
    FLUX_CHECK(splits_per_source.scalar_type() == at::ScalarType::Int);
    FLUX_CHECK(splits_per_source.is_contiguous());
    const int64_t m_full = routing_idx.numel();
    FLUX_CHECK_DIV(m_full, (int64_t)this->world_size * this->topk);
    // gemm rows of this rank's EP slice, from the host cnt (no device sync)
    const int32_t *cnt = splits_per_source.data_ptr<int32_t>();
    int64_t m_this_ep = 0;
    for (int h = 0; h < this->world_size; h++) {
      for (int64_t e = this->ep_start; e < this->ep_start + this->ep_nexperts; e++) {
        m_this_ep += cnt[h * this->total_num_experts + e];
      }
    }
    auto [pack_idx, reduce_idx] = build_a2av_combine_indices(
        routing_idx,
        splits_gpu,
        splits_per_source,
        m_this_ep,
        m_full,
        this->world_size,
        this->rank,
        this->total_num_experts,
        this->ep_nexperts,
        this->ep_start);
    std::vector<torch::Tensor> out{pack_idx, reduce_idx};
    if (this->a2av_compress) {
      FLUX_CHECK(a2av_unique_counts.has_value())
          << "a2av_hier_compress derive requires a2av_unique_counts ([W, nnodes] int32 CPU)";
      // sort-free path (2026-08-21): scd-arithmetic kernels, no radix sorts
      auto [wp, wc, rp, rr] = build_a2av_compress_indices_fast(
          routing_idx,
          splits_gpu,
          splits_per_source,
          a2av_unique_counts.value(),
          m_full,
          this->world_size,
          this->nnodes,
          this->local_world_size,
          this->rank,
          this->total_num_experts,
          this->ep_nexperts,
          this->topk);
      static const bool kCheckIdentity =
          get_int_from_env("FLUX_A2AV_RS_CHECK_IDENTITY", 0) != 0;
      if (kCheckIdentity) {
        auto [wp_ref, wc_ref, rp_ref, rr_ref] = build_a2av_compress_indices(
            routing_idx,
            splits_gpu,
            splits_per_source,
            a2av_unique_counts.value(),
            m_full,
            this->world_size,
            this->nnodes,
            this->local_world_size,
            this->rank,
            this->total_num_experts,
            this->ep_nexperts,
            this->topk);
        FLUX_CHECK(torch::equal(wp, wp_ref)) << "compress plan wire_ptr identity mismatch";
        FLUX_CHECK(torch::equal(wc, wc_ref)) << "compress plan wire_copy identity mismatch";
        FLUX_CHECK(torch::equal(rp, rp_ref)) << "compress plan red_ptr identity mismatch";
        FLUX_CHECK(torch::equal(rr, rr_ref)) << "compress plan red_row identity mismatch";
      }
      out.push_back(wp);
      out.push_back(wc);
      out.push_back(rp);
      out.push_back(rr);
    }
    return out;
  }

 private:

  void
  create_workspace_or_expand(int64_t workspace_size) {
    if (workspace_size <= 0)
      return;
    workspace_size = pad_to(workspace_size, 128);
    if (!this->workspace.defined() || workspace_size > this->workspace.numel()) {
      this->workspace = torch::empty(
          {workspace_size}, at::TensorOptions().dtype(at::ScalarType::Byte).device(at::kCUDA));
    }
  }

  c10::cuda::CUDAStream
  CreateReduceScatterStream() {
    at::cuda::CUDAGuard guard(at::cuda::current_device());
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithPriority(
        &stream, cudaStreamNonBlocking, get_highest_cuda_stream_priority()));
    return at::cuda::getStreamFromExternal(stream, at::cuda::current_device());
  }

  // 2026-08-21: tile-aware + a2av-aware. Legacy branches FIRST so every
  // previously-constructible config (dense AND a2av) returns exactly what it
  // returned before this change (rule-4: 1024-aligned shapes are untouched).
  static int
  n_split_fixed(int n_split, int n_dim, bool a2av) {
    const int n_per = n_dim / n_split;
    if (n_per % kTileSizeN == 0) {
      return n_split;  // legacy accept
    }
    if (a2av) {
      // 2026-08-24 (l1 combine campaign, FLUX_A2AV_NSPLIT_HONOR_TAG): the
      // a2av combine only needs 8-elem pack alignment, so HONOR the
      // requested n_split instead of falling into the legacy demotion below
      // — previously K2 (n=7168) a2av n_split in {2..6} silently demoted to
      // 7 (n_per not 1024-aligned but n 1024-aligned), which poisoned every
      // labeled-ns2 K2 a2av cell. Behavior change is a2av-only and its own
      // never-mix boundary (1024-aligned n_per configs are byte-unchanged
      // via the legacy accept above).
      (void)bytedance::flux::get_int_from_env("FLUX_A2AV_NSPLIT_HONOR_TAG", 0);
      FLUX_CHECK(n_dim % n_split == 0 && (n_dim / n_split) % 8 == 0)
          << "a2av: n (" << n_dim << ") / n_split (" << n_split
          << ") must be a multiple of 8";
      return n_split;
    }
    if (n_dim % n_split == 0 && n_per % kTileSizeNMin == 0) {
      // 2026-08-24 canonicalization (FLUX_RS_NSPLIT_512_TAG): the 512-tile
      // dense lane accepts this split, so HONOR it instead of the legacy
      // demotion below — previously K2 dense n_split in {2..6} silently
      // demoted to 7 exactly like the a2av case (own never-mix boundary;
      // 1024-aligned n_per configs unchanged via the legacy accept above).
      (void)bytedance::flux::get_int_from_env("FLUX_RS_NSPLIT_512_TAG", 0);
      return n_split;
    }
    if (n_dim % kTileSizeN == 0) {
      return n_dim / kTileSizeN;  // legacy demotion (dense lanes only)
    }
    // 512-tile dense lane (K3 H=3584 = 7*512)
    if (n_dim % n_split == 0 && n_per % kTileSizeNMin == 0) {
      return n_split;
    }
    FLUX_CHECK_DIV(n_dim, kTileSizeNMin);
    return n_dim / kTileSizeNMin;
  }

 public:
  GemmGroupedV2GatherRSOpImpl(
      std::shared_ptr<Group> tp_group_,
      int64_t total_num_experts,
      int64_t max_m,
      int64_t n_dim,
      int64_t topk,
      at::ScalarType output_dtype,
      int64_t tp_world_size,
      int64_t ep_world_size,
      int64_t max_input_groups,
      int64_t n_split_,
      bool do_all_reduce_ = false,
      bool use_read_mode = false,
      int64_t nnodes_ = 1,
      bool a2av_hier_ = false,
      bool a2av_hier_compress_ = false)
      : tp_group(tp_group_),
        total_num_experts(total_num_experts),
        max_m(max_m),
        n_dim(n_dim),
        topk(topk),
        output_dtype(output_dtype),
        max_input_groups(max_input_groups),
        rank(tp_group_->get_rank()),
        world_size(tp_group_->get_size()),
        tp_world_size(tp_world_size),
        ep_world_size(ep_world_size),
        nnodes(nnodes_),
        local_rank(DistEnv(tp_group_->get_rank(), tp_group_->get_size(), nnodes_).local_rank),
        local_world_size(tp_group_->get_size() / nnodes_),
        n_split(n_split_fixed(n_split_, n_dim, a2av_hier_ || a2av_hier_compress_)),
        do_all_reduce(do_all_reduce_),
        a2av_hier(a2av_hier_ || a2av_hier_compress_),
        a2av_compress(a2av_hier_compress_ && nnodes_ > 1),
        group_barrier(tp_group_, false) {
    if (this->n_split != n_split_) {
      FLUX_LOG_FIRST_N(WARN, 1) << "warning: (n / split_n) not aligned to the combine tile ("
                                << combine_tile_n(n_dim, this->n_split)
                                << "), set split_n=" << this->n_split << "\n";
    }
    FLUX_CHECK(!(a2av_hier_ && a2av_hier_compress_))
        << "pass a2av_hier or a2av_hier_compress, not both";
    FLUX_CHECK_EQ(this->tp_world_size * this->ep_world_size, this->world_size);
    FLUX_CHECK_DIV(this->total_num_experts, this->ep_world_size);
    FLUX_CHECK_LE(max_input_groups, kMaxNumGroups);
    FLUX_CHECK_GE(this->nnodes, 1);
    FLUX_CHECK_DIV(this->world_size, this->nnodes);
    if (this->a2av_hier) {
      // v1 scope: complete [1, hidden] GEMM rows (with tp > 1 each copy would be
      // a K-partial and a2av-of-copies would not apply), single weight group
      FLUX_CHECK_EQ(this->tp_world_size, 1) << "a2av_hier requires tp_world_size == 1";
      FLUX_CHECK_EQ(this->max_input_groups, 1) << "a2av_hier requires max_input_groups == 1";
      FLUX_CHECK(!do_all_reduce_) << "a2av_hier does not support do_all_reduce";
      FLUX_CHECK(!use_read_mode) << "a2av_hier does not support use_read_mode";
      FLUX_CHECK_DIV(this->max_m, this->world_size);
    }
    this->ep_nexperts = this->total_num_experts / this->ep_world_size;
    int ep_rank = this->rank / this->tp_world_size;
    this->ep_start = this->ep_nexperts * ep_rank;
    this->gather_rs_stream = CreateReduceScatterStream();
    CUDA_CHECK(cudaEventCreateWithFlags(&this->gemm_start_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->gather_rs_done_event, cudaEventDisableTiming));
    // M-split waves (Slipstream v2): ctor-scoped mode resolution (allocation
    // implications: barrier sizing below + the wave-table arena)
    this->msplit_ = this->a2av_hier && this->nnodes > 1 && get_a2av_rs_msplit() != 0 &&
                    (a2av_env_explicit("FLUX_A2AV_RS_MSPLIT") || this->n_split == 1);
    if (this->msplit_) {
      FLUX_CHECK_EQ(this->n_split, 1) << "FLUX_A2AV_RS_MSPLIT requires n_split == 1";
      FLUX_CHECK_LE(this->nnodes, kA2AVMaxNodes);
      FLUX_CHECK_LE(this->nnodes, 128)
          << "msplit wave flags must fit the 128-padded barrier flag region";
      this->msplit_wave_nodes_ = get_a2av_rs_wave_nodes();
      const int64_t tbl = (int64_t)2 * this->nnodes * this->ep_nexperts + this->nnodes;
      this->msplit_host_ = torch::empty(
          {tbl},
          at::TensorOptions(at::kCPU).dtype(at::ScalarType::Int).pinned_memory(true));
      this->msplit_dev_ = empty_with_uninitialized_data(
          std::vector<int64_t>{tbl}, at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Int));
      CUDA_CHECK(cudaEventCreateWithFlags(&this->msplit_h2d_event_, cudaEventDisableTiming));
      this->msplit_wave_of_node_.assign(this->nnodes, 0);
      this->msplit_node_order_.assign(this->nnodes, 0);
      this->fused_pack_ = get_a2av_rs_fused_pack() != 0;
      if (this->fused_pack_) {
        this->msplit_inv_pack_ = empty_with_uninitialized_data(
            std::vector<int64_t>{this->max_m},
            at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Int));
      }
    }
    // shared identity indices: the ScatterD iterator always reads an index
    // array, so every non-fused problem points at this relative iota
    this->msplit_iota_ = torch::arange(
        this->max_m, at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Int));
    create_barriers();
    topk_reduce_scatter_op = std::make_unique<TopkReduceScatterOp>(
        tp_group_,
        max_m,
        n_dim,
        topk,
        output_dtype,
        total_num_experts / ep_world_size,
        ep_world_size,
        this->barriers,
        this->n_split,
        do_all_reduce_,
        use_read_mode,
        nnodes_,
        this->a2av_hier,
        this->a2av_compress);
  }


  torch::Tensor
  forward_gather_rs_impl(
      std::vector<torch::Tensor> inputs,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits,
      torch::Tensor routing_idx,
      c10::optional<std::vector<torch::Tensor>> bias,
      c10::optional<std::vector<torch::Tensor>> input_scales,
      c10::optional<std::vector<torch::Tensor>> weight_scales,
      c10::optional<std::vector<torch::Tensor>> output_vec_scales,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync,
      c10::optional<UnifiedGemmHParams> const &hparams,
      c10::optional<torch::Tensor> splits_per_source = c10::nullopt,
      c10::optional<torch::Tensor> a2av_pack_index = c10::nullopt,
      c10::optional<torch::Tensor> a2av_reduce_index = c10::nullopt,
      c10::optional<torch::Tensor> a2av_unique_counts = c10::nullopt,
      c10::optional<std::vector<torch::Tensor>> a2av_wire_csr = c10::nullopt,
      c10::optional<std::vector<torch::Tensor>> a2av_reduce_csr = c10::nullopt) {
    /*
      Note: When expert parallel is enabled, the inputs/weights tensor should be
      the partial the current expert parallel rank. But the splits_cpu and routing
      idx should be global no matter whether expert parallel is enabled, which means the
      splits_cpu/routing_idx should contains all the experts / tokens no matter whether expert
      parallel is enabled.
    */
    FLUX_CHECK(!bias.has_value());
    FLUX_CHECK_LE(inputs.size(), this->max_input_groups);
    int num_groups = inputs.size();
    FLUX_CHECK_LE(num_groups, this->max_input_groups);
    FLUX_CHECK_EQ(num_groups, weights.size());

    at::ScalarType input_torch_type = weights[0].scalar_type();
    FLUX_CHECK(input_torch_type != at::ScalarType::Char)
        << "Moe AG+Scatter INT8 not supported yet";
    bool is_fp8 = is_fp8_torch_dtype(input_torch_type);
    // if the dtype of input is fp8, use bfloat16 as the output dtype
    at::ScalarType output_torch_type = is_fp8 ? at::ScalarType::BFloat16 : input_torch_type;
    DataTypeEnum output_type = from_torch_dtype(output_torch_type);
    int m_full = routing_idx.size(0);
    int ntokens = m_full / this->topk;
    int n_tokens_per_rank = ntokens / this->world_size;
    int M_this_ep = inputs[0].size(0);
    int K = inputs[0].size(1);
    int E = weights[0].size(0);
    int N = weights[0].size(1);
    // check input/weight
    for (int i = 0; i < num_groups; i++) {
      CHECK_3D(weights[i], this->ep_nexperts, N, K);  // only RCR layout supported
      CHECK_INPUT(weights[i], input_torch_type);
      CHECK_2D(inputs[i], M_this_ep, K);
      CHECK_INPUT(inputs[i], input_torch_type);
    }
    // check input_scale/weight_scale/output_vec_scale
    if (input_scales.has_value()) {
      FLUX_CHECK_EQ(input_scales->size(), num_groups);
      for (auto &input_scale : input_scales.value()) {
        CHECK_1D(input_scale, 1);
        CHECK_INPUT(input_scale, at::ScalarType::Float);
      }
    }
    if (weight_scales.has_value()) {
      FLUX_CHECK_EQ(weight_scales->size(), num_groups);
      for (auto &weight_scale : weight_scales.value()) {
        CHECK_1D(weight_scale, E);
        CHECK_INPUT(weight_scale, at::ScalarType::Float);
      }
    }
    if (output_vec_scales.has_value()) {
      FLUX_CHECK_EQ(output_vec_scales->size(), num_groups);
      for (auto &output_vec_scale : output_vec_scales.value()) {
        CHECK_1D(output_vec_scale, M_this_ep);
        CHECK_INPUT(output_vec_scale, at::ScalarType::Float);
      }
    }

    CHECK_INPUT(routing_idx, at::ScalarType::Int);
    if (this->ep_world_size == 1) {
      FLUX_CHECK_EQ(M_this_ep, m_full);
    } else {
      FLUX_CHECK_LE(M_this_ep, m_full) << "input.size(0) larger than routing_idx.size(0)";
    }
    FLUX_CHECK_DIV(m_full, this->world_size * this->topk);
    FLUX_CHECK_LE(m_full, this->max_m) << "input.size(0) " << M_this_ep << " larger than max_m\n";
    FLUX_CHECK_EQ(N, this->n_dim);

    FLUX_CHECK_GE(N, 8) << "N must be greater than or equal 8 for cutlass grouped gemm.";
    FLUX_CHECK_GE(K, 8) << "K must be greater than or equal 8 for cutlass grouped gemm.";
    torch::Tensor splits_gpu;
    if (!splits.is_cuda()) {
      splits_gpu = empty_with_uninitialized_data(
          splits.sizes(), at::TensorOptions(c10::kCUDA).dtype(at::ScalarType::Int));
      splits_gpu.copy_(splits, true);
    } else {
      splits_gpu = splits;
    }
    CHECK_INPUT(splits_gpu, at::ScalarType::Int);
    CHECK_1D(splits_gpu, this->total_num_experts);

    torch::Tensor a2av_pack_idx_t, a2av_reduce_idx_t;
    if (this->a2av_hier) {
      FLUX_CHECK_EQ(num_groups, 1) << "a2av_hier supports a single weight group";
      FLUX_CHECK(!is_fp8) << "a2av_hier supports fp16/bf16 only";
      FLUX_CHECK(splits_per_source.has_value())
          << "a2av_hier requires splits_per_source ([W, nexperts] int32 CPU)";
      auto const &cnt_t = splits_per_source.value();
      FLUX_CHECK(cnt_t.device().is_cpu()) << "splits_per_source must be a CPU tensor";
      CHECK_2D(cnt_t, this->world_size, this->total_num_experts);
      FLUX_CHECK(cnt_t.scalar_type() == at::ScalarType::Int);
      FLUX_CHECK(cnt_t.is_contiguous());
      // compress (dedup) routing-plan tensors: shape-validated here, consumed
      // once the a2av_hier_compress data path lands. All-or-none per pair; the
      // U matrix rides alone (isolated mode passes only it, like
      // splits_per_source, and the op builds the CSRs in-forward).
      FLUX_CHECK(a2av_wire_csr.has_value() == a2av_reduce_csr.has_value())
          << "pass both a2av_wire_csr and a2av_reduce_csr or neither";
      if (a2av_wire_csr.has_value()) {
        FLUX_CHECK(a2av_unique_counts.has_value())
            << "compress CSRs require a2av_unique_counts ([W, nnodes] int32 CPU)";
        FLUX_CHECK_EQ((int)a2av_wire_csr->size(), 2) << "a2av_wire_csr = [wire_ptr, wire_copy]";
        FLUX_CHECK_EQ((int)a2av_reduce_csr->size(), 2)
            << "a2av_reduce_csr = [red_ptr, red_row]";
        for (auto const &t : *a2av_wire_csr) {
          CHECK_INPUT(t, at::ScalarType::Int);
        }
        for (auto const &t : *a2av_reduce_csr) {
          CHECK_INPUT(t, at::ScalarType::Int);
        }
      }
      if (a2av_unique_counts.has_value()) {
        auto const &u_t = a2av_unique_counts.value();
        FLUX_CHECK(u_t.device().is_cpu()) << "a2av_unique_counts must be a CPU tensor";
        CHECK_2D(u_t, this->world_size, this->nnodes);
        FLUX_CHECK(u_t.scalar_type() == at::ScalarType::Int);
        FLUX_CHECK(u_t.is_contiguous());
      }
      if (this->a2av_compress) {
        FLUX_CHECK(a2av_unique_counts.has_value())
            << "a2av_hier_compress requires a2av_unique_counts ([W, nnodes] int32 CPU) -- "
               "untimed host metadata, like splits_per_source";
        if (!a2av_wire_csr.has_value()) {
          // isolated mode: compress CSR build on the timed critical path, same
          // v1 placement as the pack/reduce index build below
          auto [wp, wc, rp, rr] = build_a2av_compress_indices(
              routing_idx, splits_gpu, cnt_t, a2av_unique_counts.value(), m_full,
              this->world_size, this->nnodes, this->local_world_size, this->rank,
              this->total_num_experts, this->ep_nexperts, this->topk);
          a2av_wire_csr = std::vector<torch::Tensor>{wp, wc};
          a2av_reduce_csr = std::vector<torch::Tensor>{rp, rr};
        }
      } else {
        FLUX_CHECK(!a2av_unique_counts.has_value() && !a2av_wire_csr.has_value())
            << "compress plan tensors require the a2av_hier_compress ctor flag";
      }
      if (a2av_pack_index.has_value() || a2av_reduce_index.has_value()) {
        FLUX_CHECK(a2av_pack_index.has_value() && a2av_reduce_index.has_value())
            << "pass both a2av_pack_index and a2av_reduce_index or neither";
        a2av_pack_idx_t = a2av_pack_index.value();
        a2av_reduce_idx_t = a2av_reduce_index.value();
      } else {
        // v1 placement: index math on the main stream before the GEMM launch --
        // ordering to the pack kernel comes free via gemm_start_event. A fused
        // layer0+layer1 pipeline passes layer0's tensors and skips this entirely.
        std::tie(a2av_pack_idx_t, a2av_reduce_idx_t) = build_a2av_combine_indices(
            routing_idx, splits_gpu, cnt_t, M_this_ep, m_full,
            this->world_size, this->rank, this->total_num_experts,
            this->ep_nexperts, this->ep_start);
      }
    }

    auto stream = c10::cuda::getCurrentCUDAStream();

    ArchEnum arch = get_arch();
    SMCoreEnum sm_core = get_sm_core();
    auto input_type = from_torch_dtype(input_torch_type);
    auto dt_conf = to_gemm_dtype_config(
        make_gemm_dtype_config(input_type, input_type, output_type, output_type));
    auto impl_spec = make_gemm_v2_meta(fast_accum and dt_conf.is_input_fp8());
    // always use topk=1 impl: to save some compile time
    auto comm_spec = make_gather_rs_meta(1);
    auto meta = make_gemm_meta(
        dt_conf, arch, sm_core, _GatherRS{}, _RCR{}, _GemmGroupedV2{}(), impl_spec, comm_spec);
    auto rt_conf = make_runtime_config(N, cute::ceil_div(m_full, this->ep_nexperts), K);
    OpRegistry::OpPtr gemm_op;
    if (hparams.has_value()) {
      gemm_op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      gemm_op = OpRegistry::instance().get_op(meta, rt_conf);
    }

    std::vector<torch::Tensor> gemm_outs;
    for (int i = 0; i < num_groups; i++) {
      gemm_outs.push_back(empty_with_uninitialized_data(
          std::vector<int64_t>{M_this_ep, N},
          at::TensorOptions(at::kCUDA).dtype(output_torch_type)));
    }
    torch::Tensor output = empty_with_uninitialized_data(
        std::vector<int64_t>{this->do_all_reduce ? ntokens : n_tokens_per_rank, N},
        at::TensorOptions(at::kCUDA).dtype(output_torch_type));

    // ---- M-split waves (Slipstream v2): destination-wave GEMM sub-problems --
    // Host-built per iteration from splits_per_source (rule 5: in the timed
    // bracket, ~NN*E adds), staged through the pinned arena with ONE async H2D
    // on the forward stream (NR-09: a pageable H2D here would hide a sync).
    int msplit_n_waves = 0;
    const int32_t *msplit_wave_M_dev = nullptr;
    const int32_t *msplit_wave_off_dev = nullptr;
    const int32_t *msplit_ne_wave_dev = nullptr;
    if (this->msplit_) {
      FLUX_CHECK(splits_per_source.has_value());
      const int NN = this->nnodes;
      const int E = this->ep_nexperts;
      const int L = this->local_world_size;
      const int my_node = this->rank / L;
      const int NG = this->msplit_wave_nodes_;
      const int32_t *cnt = splits_per_source->data_ptr<int32_t>();  // [W, total_experts]
      // Ring order own-node-last; waves = NG ring-consecutive nodes, never
      // crossing the my_node wrap (the intermediate's within-expert rows are
      // home-node ASCENDING — the layer0 stable scatter order — so a wave must
      // be one contiguous ascending node range).
      std::vector<std::pair<int, int>> waves;  // ascending node range [a, b)
      const bool size_order = get_a2av_rs_wave_order() != 0;
      if (size_order) {
        // gen-8a: remote per-node waves ordered by DESCENDING segment size
        // (ties by ring position), own node last. Requires NG == 1: a wave
        // must stay one contiguous ascending node range.
        FLUX_CHECK_EQ(NG, 1)
            << "FLUX_A2AV_RS_WAVE_ORDER=size requires FLUX_A2AV_RS_WAVE_NODES=1";
        std::vector<std::pair<int64_t, int>> rem;  // (-rows, ring_pos) -> node
        for (int gi = 0; gi < NN - 1; gi++) {
          const int node = (my_node + 1 + gi) % NN;
          int64_t rows = 0;
          for (int lr2 = 0; lr2 < L; lr2++) {
            for (int e = 0; e < E; e++) {
              rows += cnt[(int64_t)(node * L + lr2) * this->total_num_experts +
                          this->ep_start + e];
            }
          }
          rem.emplace_back(-(rows << 8) + gi, node);  // size desc, ring tiebreak
        }
        std::sort(rem.begin(), rem.end());
        for (auto &p : rem) {
          waves.emplace_back(p.second, p.second + 1);
        }
        waves.emplace_back(my_node, my_node + 1);  // own node = final wave
      } else {
        const bool own_first = get_a2av_rs_own_wave_first() != 0;
        if (own_first) {
          waves.emplace_back(my_node, my_node + 1);  // own node = FIRST wave
        }
        const int r1 = NN - 1 - my_node;  // ring run 1: nodes my+1 .. NN-1
        for (int s = 0; s < r1; s += NG) {
          waves.emplace_back(my_node + 1 + s, std::min(my_node + 1 + s + NG, NN));
        }
        for (int s = 0; s < my_node; s += NG) {  // ring run 2: nodes 0 .. my-1
          waves.emplace_back(s, std::min(s + NG, my_node));
        }
        if (!own_first) {
          waves.emplace_back(my_node, my_node + 1);  // own node = final wave
        }
      }
      msplit_n_waves = (int)waves.size();
      // schedule arrays, ONE source of truth for every consumer:
      // node_order[i] = i-th node in production order; wave_of_order[i] = its
      // cascade flag (== position of its wave). Ring mode reproduces the
      // legacy (my_node+1+gi)%NN sequence exactly.
      {
        int pos = 0;
        for (int w = 0; w < msplit_n_waves; w++) {
          for (int node = waves[w].first; node < waves[w].second; node++, pos++) {
            this->msplit_node_order_[pos] = node;
            this->msplit_wave_of_node_[pos] = w;
          }
        }
        FLUX_CHECK_EQ(pos, NN);
        const int own_pos = get_a2av_rs_own_wave_first() != 0 ? 0 : NN - 1;
        FLUX_CHECK_EQ(this->msplit_node_order_[own_pos], my_node);
      }
      // guard the pinned arena against a still-in-flight previous H2D
      CUDA_CHECK(cudaEventSynchronize(this->msplit_h2d_event_));
      int32_t *hp = this->msplit_host_.data_ptr<int32_t>();
      int32_t *h_wave_M = hp;
      int32_t *h_wave_off = hp + (int64_t)msplit_n_waves * E;
      int32_t *h_ne = hp + (int64_t)2 * msplit_n_waves * E;
      for (int w = 0; w < msplit_n_waves; w++) {
        h_ne[w] = 0;
      }
      for (int e = 0; e < E; e++) {
        const int64_t ge = (int64_t)this->ep_start + e;  // global expert id
        // plain-ascending home prefix — the same shape as the combine
        // builders' home_base cumsum (their FLUX_CHECKs guard consistency)
        int64_t acc = 0;
        int64_t node_base[kA2AVMaxNodes + 1];
        for (int n2 = 0; n2 < NN; n2++) {
          node_base[n2] = acc;
          for (int lr2 = 0; lr2 < L; lr2++) {
            acc += cnt[(int64_t)(n2 * L + lr2) * this->total_num_experts + ge];
          }
        }
        node_base[NN] = acc;
        for (int w = 0; w < msplit_n_waves; w++) {
          const int64_t off = node_base[waves[w].first];
          const int64_t rows = node_base[waves[w].second] - off;
          FLUX_CHECK_LE(rows, (int64_t)2147483647);
          h_wave_M[(int64_t)w * E + e] = (int32_t)rows;
          h_wave_off[(int64_t)w * E + e] = (int32_t)off;
          if (rows > 0) {
            h_ne[w] += 1;
          }
        }
      }
      const int64_t used = (int64_t)2 * msplit_n_waves * E + msplit_n_waves;
      CUDA_CHECK(cudaMemcpyAsync(
          this->msplit_dev_.data_ptr(),
          hp,
          used * sizeof(int32_t),
          cudaMemcpyHostToDevice,
          stream));
      CUDA_CHECK(cudaEventRecord(this->msplit_h2d_event_, stream));
      const int32_t *dp = this->msplit_dev_.data_ptr<int32_t>();
      msplit_wave_M_dev = dp;
      msplit_wave_off_dev = dp + (int64_t)msplit_n_waves * E;
      msplit_ne_wave_dev = dp + (int64_t)2 * msplit_n_waves * E;
    }

    MoeGatherRSWorkspaceArgs ws_args{
        .num_groups = num_groups,
        .N_split = this->n_split,
        .ep_start = this->ep_start,
        .ep_nexperts = this->ep_nexperts,
        .N = N,
        .K = K,
        .splits_gpu = splits_gpu.data_ptr<int>()};
    for (int i = 0; i < num_groups; i++) {
      ws_args.input[i] = inputs[i].data_ptr();
      ws_args.weights[i] = weights[i].data_ptr();
      ws_args.output[i] = gemm_outs[i].data_ptr();
      ws_args.input_scales[i] =
          input_scales.has_value() ? input_scales->at(i).data_ptr<float>() : nullptr;
      ws_args.weight_scales[i] =
          weight_scales.has_value() ? weight_scales->at(i).data_ptr<float>() : nullptr;
    }

    if (this->msplit_) {
      ws_args.msplit = 1;
      ws_args.n_waves = msplit_n_waves;
      ws_args.wave_M = msplit_wave_M_dev;
      ws_args.wave_off = msplit_wave_off_dev;
      ws_args.non_empty_per_wave = msplit_ne_wave_dev;
      ws_args.barrier = this->barrier.data_ptr<int>();
    }
    ws_args.iota = this->msplit_iota_.data_ptr<int32_t>();
    if (this->msplit_ && this->fused_pack_) {
      // gen-8c epilogue-fused pack: fold the gate coefficients into the
      // intermediate on the K side (mathematically identical: sum_j w_j (A_j B)
      // == sum_j (w_j A_j) B), build the pack inverse, and point D at the
      // dest-major send panel
      if (output_vec_scales.has_value()) {
        inputs[0].mul_(output_vec_scales->at(0).unsqueeze(1));
      }
      a2av_invert_index(
          A2AVInvertIndexArguments{
              a2av_pack_idx_t.data_ptr<int32_t>(),
              this->msplit_inv_pack_.data_ptr<int32_t>(),
              (int64_t)M_this_ep},
          stream);
      FLUX_CHECK_LE(
          (int64_t)M_this_ep, this->topk_reduce_scatter_op->send_panel_rows())
          << "fused pack: send panel capacity (raise FLUX_A2AV_RS_MAX_SEND_ROWS)";
      ws_args.fused_pack = 1;
      ws_args.inv_pack = this->msplit_inv_pack_.data_ptr<int32_t>();
      ws_args.send_panel = this->topk_reduce_scatter_op->send_panel_ptr();
    }
    int problem_count = this->msplit_
                            ? msplit_n_waves * ws_args.ep_nexperts
                            : ws_args.N_split * ws_args.num_groups * ws_args.ep_nexperts;
    torch::Tensor workspace_gpu = empty_with_uninitialized_data(
        std::vector<int64_t>{get_args_workspace_size(problem_count)},
        at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Char));
    void *workspace = workspace_gpu.data_ptr();
    make_workspace(
        ws_args,
        GemmLayoutEnum::RCR,
        c10::elementSize(input_torch_type),
        c10::elementSize(output_torch_type),
        workspace,
        stream);

    constexpr int kAlignment = 128;

    // the offsets
    int offset_problem_sizes = 0;
    int offset_ptr_A = pad_to(
        offset_problem_sizes + problem_count * sizeof(cutlass::gemm::GemmCoord), kAlignment);
    int offset_ptr_B = pad_to(offset_ptr_A + problem_count * sizeof(void *), kAlignment);
    int offset_ptr_C = pad_to(offset_ptr_B + problem_count * sizeof(void *), kAlignment);
    int offset_ptr_D = pad_to(offset_ptr_C + problem_count * sizeof(void *), kAlignment);
    int offset_lda = pad_to(offset_ptr_D + problem_count * sizeof(void *), kAlignment);
    int offset_ldb = pad_to(offset_lda + problem_count * sizeof(int64_t), kAlignment);
    int offset_ldc = pad_to(offset_ldb + problem_count * sizeof(int64_t), kAlignment);
    int offset_ldd = pad_to(offset_ldc + problem_count * sizeof(int64_t), kAlignment);
    int offset_ldr = pad_to(offset_ldd + problem_count * sizeof(int64_t), kAlignment);
    int offset_scale_A = pad_to(offset_ldr + problem_count * sizeof(int64_t), kAlignment);
    int offset_scale_B = pad_to(offset_scale_A + problem_count * sizeof(float *), kAlignment);
    int offset_scatter_D = pad_to(offset_scale_B + problem_count * sizeof(float *), kAlignment);
    int offset_non_empty_problem_count =
        pad_to(offset_scatter_D + problem_count * sizeof(int *), kAlignment);
    // the ptrs
    cutlass::gemm::GemmCoord *problem_sizes =
        (cutlass::gemm::GemmCoord *)((char *)workspace + offset_problem_sizes);
    void **ptr_A = (void **)((char *)workspace + offset_ptr_A);
    void **ptr_B = (void **)((char *)workspace + offset_ptr_B);
    void **ptr_C = (void **)((char *)workspace + offset_ptr_C);
    void **ptr_D = (void **)((char *)workspace + offset_ptr_D);
    int64_t *lda = (int64_t *)((char *)workspace + offset_lda);
    int64_t *ldb = (int64_t *)((char *)workspace + offset_ldb);
    int64_t *ldc = (int64_t *)((char *)workspace + offset_ldc);
    int64_t *ldd = (int64_t *)((char *)workspace + offset_ldd);
    int64_t *ldr = (int64_t *)((char *)workspace + offset_ldr);
    float **scale_A = (float **)((char *)workspace + offset_scale_A);
    float **scale_B = (float **)((char *)workspace + offset_scale_B);
    int **scatter_D_ptr_ws = (int **)((char *)workspace + offset_scatter_D);
    int *non_empty_problem_count = (int *)((char *)workspace + offset_non_empty_problem_count);

    float alpha = 1.0, beta = 0.0;

    GemmGroupedV2GatherRSArguments args{
        .problem_sizes = problem_sizes,
        .problem_count = problem_count,
        .non_empty_problem_count = non_empty_problem_count,
        .alpha = alpha,
        .beta = beta,
        .ptr_A = ptr_A,
        .ptr_B = ptr_B,
        .ptr_C = ptr_C,
        .ptr_D = ptr_D,
        .lda = lda,
        .ldb = ldb,
        .ldc = ldc,
        .ldd = ldd,
        .ldr = ldr,
        .scaleA = (float const **)scale_A,
        .scaleB = (float const **)scale_B,
        .topk = this->topk,
        .barrier = this->barrier.data_ptr<int>(),
        .routing_idx = routing_idx.data_ptr<int32_t>(),
        // msplit: the cascade's group axis is the WAVE (n_split stays 1 for
        // panels/columns — full-N sub-problems)
        .n_split = this->msplit_ ? msplit_n_waves : n_split,
        .sm_margin = sm_margin + (this->a2av_hier
                                      ? get_a2av_pack_blocks() + get_a2av_reduce_blocks() +
                                            (this->a2av_compress ? get_a2av_prered_blocks() : 0)
                                      : get_rs_threadblock_count()),
        .non_empty_per_group = this->msplit_ ? msplit_ne_wave_dev : nullptr,
        .scatter_D_ptr = scatter_D_ptr_ws};

    int64_t workspace_size = gemm_op->get_workspace_size(args);
    this->create_workspace_or_expand(workspace_size);

    group_barrier.barrier_all(stream);

    // ensure barrier initialized correctly
    CUDA_CHECK(cudaEventRecord(this->gemm_start_event, stream));
    CUDA_CHECK(cudaStreamWaitEvent(gather_rs_stream, this->gemm_start_event));

    if (M_this_ep > 0) {
      gemm_op->run(args, this->workspace.defined() ? this->workspace.data_ptr() : nullptr, stream);
    } else {
      this->barrier.fill_(1);
    }
    topk_reduce_scatter_op->set_msplit_waves(
        this->msplit_wave_of_node_,
        this->msplit_node_order_,
        this->msplit_ ? msplit_n_waves : 0);
    output = topk_reduce_scatter_op->run(
        gemm_outs,
        output,
        this->ep_start,
        this->ep_nexperts,
        splits_gpu,
        routing_idx,
        output_vec_scales,
        this->a2av_hier ? get_a2av_pack_blocks() : get_rs_threadblock_count(),
        (intptr_t)gather_rs_stream,
        splits_per_source,
        this->a2av_hier ? c10::optional<torch::Tensor>(a2av_pack_idx_t) : c10::nullopt,
        this->a2av_hier ? c10::optional<torch::Tensor>(a2av_reduce_idx_t) : c10::nullopt,
        this->a2av_compress ? a2av_unique_counts : c10::nullopt,
        this->a2av_compress ? a2av_wire_csr : c10::nullopt,
        this->a2av_compress ? a2av_reduce_csr : c10::nullopt);
    CUDA_CHECK(cudaEventRecord(this->gather_rs_done_event, gather_rs_stream));
    CUDA_CHECK(cudaStreamWaitEvent(stream, this->gather_rs_done_event));

    group_barrier.barrier_all(stream);
    this->barrier.zero_();
    this->topk_reduce_scatter_op->reset_buffer();
    return output;
  }

  torch::Tensor
  forward_gather_rs(
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync,
      c10::optional<torch::Tensor> splits_per_source = c10::nullopt,
      c10::optional<torch::Tensor> a2av_pack_index = c10::nullopt,
      c10::optional<torch::Tensor> a2av_reduce_index = c10::nullopt,
      c10::optional<torch::Tensor> a2av_unique_counts = c10::nullopt,
      c10::optional<std::vector<torch::Tensor>> a2av_wire_csr = c10::nullopt,
      c10::optional<std::vector<torch::Tensor>> a2av_reduce_csr = c10::nullopt) {
    if (input.scalar_type() == torch::kInt8) {
      FLUX_CHECK(!this->a2av_hier) << "a2av_hier does not support the int8/triton path";
      return forward_gather_rs_triton_aot(
          input,
          weight,
          splits_cpu,
          routing_idx,
          c10::nullopt,
          input_scale,
          weight_scale,
          output_vec_scale,
          fast_accum,
          sm_margin,
          with_stream_sync);
    }
    return forward_gather_rs_impl(
        {std::move(input)},
        {std::move(weight)},
        std::move(splits_cpu),
        std::move(routing_idx),
        as_optional_vec(bias),
        as_optional_vec(input_scale),
        as_optional_vec(weight_scale),
        as_optional_vec(output_vec_scale),
        fast_accum,
        sm_margin,
        with_stream_sync,
        c10::nullopt,
        std::move(splits_per_source),
        std::move(a2av_pack_index),
        std::move(a2av_reduce_index),
        std::move(a2av_unique_counts),
        std::move(a2av_wire_csr),
        std::move(a2av_reduce_csr));
  }

  torch::Tensor
  forward_gather_rs_triton_aot(
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor splits,
      torch::Tensor routing_idx,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync) {
#if defined(FLUX_WITH_TRITON_AOT)
    FLUX_CHECK(this->nnodes == 1) << "moe_gather_rs triton path is single-node only";
    int M_this_ep = input.size(0);
    int K = input.size(1);
    int E = weight.size(0);
    int N = weight.size(1);
    int m_full = routing_idx.size(0);
    int ntokens = m_full / this->topk;
    int n_tokens_per_rank = ntokens / this->world_size;
    at::ScalarType input_dtype = weight.scalar_type();
    bool is_fp8 = is_fp8_torch_dtype(input_dtype);
    bool is_s8_gemm = input_dtype == at::ScalarType::Char;
    CHECK_INPUT(input, input_dtype);
    CHECK_INPUT(weight, input_dtype);

    // check input_scale/weight_scale/output_vec_scale
    if (input_scale.has_value()) {
      if (is_s8_gemm) {
        FLUX_CHECK_EQ(input_scale->numel(), M_this_ep);
      } else {
        CHECK_1D(input_scale.value(), 1);
      }
      CHECK_INPUT(input_scale.value(), at::ScalarType::Float);
    }
    FLUX_CHECK(weight_scale.has_value());
    if (weight_scale.has_value()) {
      if (is_s8_gemm) {
        CHECK_2D(weight_scale.value(), E, N);
      } else {
        CHECK_1D(weight_scale.value(), E);
      }
      CHECK_INPUT(weight_scale.value(), at::ScalarType::Float);
    }
    FLUX_CHECK(output_vec_scale.has_value());
    if (output_vec_scale.has_value()) {
      FLUX_CHECK_EQ(output_vec_scale->numel(), M_this_ep);
      CHECK_INPUT(output_vec_scale.value(), at::ScalarType::Float);
    }

    CHECK_INPUT(routing_idx, at::ScalarType::Int);
    if (this->ep_world_size == 1) {
      FLUX_CHECK_EQ(M_this_ep, m_full);
    } else {
      FLUX_CHECK_LE(M_this_ep, m_full) << "input.size(0) larger than routing_idx.size(0)";
    }
    FLUX_CHECK_DIV(m_full, this->world_size * this->topk);
    FLUX_CHECK_LE(m_full, this->max_m) << "input.size(0) " << M_this_ep << " larger than max_m\n";
    FLUX_CHECK_EQ(N, this->n_dim);
    FLUX_CHECK_DIV(N, 16) << "N % 16 == 0 expected for triton grouped gemm.";
    FLUX_CHECK_DIV(K, 16) << "K % 16 == 0 expected for triton grouped gemm.";

    torch::Tensor splits_cpu, splits_gpu;
    auto option_cpu = at::TensorOptions(at::kCPU).pinned_memory(true).dtype(at::ScalarType::Int);
    auto option_gpu = at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Int);
    if (splits.is_cuda()) {
      splits_gpu = splits;
      splits_cpu = empty_with_uninitialized_data(splits.sizes(), option_cpu);
      splits_cpu.copy_(splits, false);  // non-blocking copy
    } else {
      splits_cpu = splits;
      splits_gpu = empty_with_uninitialized_data(splits.sizes(), option_gpu);
      auto splits_pin = empty_with_uninitialized_data(splits.sizes(), option_cpu);
      splits_pin.copy_(splits, true);      // async copy
      splits_gpu.copy_(splits_pin, true);  // async copy
    }
    at::ScalarType output_dtype = is_fp8 || is_s8_gemm ? at::ScalarType::BFloat16 : input_dtype;

    using FuncType = decltype(moe_gather_rs_grouped_gemm_s8_ex);
    FuncType *grouped_gemm_func = nullptr;
    moe_gather_rs_grouped_gemm_kernel__triton_algo_info_t algo_info;
    if (is_s8_gemm) {
      grouped_gemm_func = moe_gather_rs_grouped_gemm_s8_ex;
      algo_info = moe_gather_rs_grouped_gemm_kernel__triton_algo_info_t{
          .N_SPLIT = n_split,
          .BLOCK_SIZE_M = 64,
          .BLOCK_SIZE_N = 128,
          .BLOCK_SIZE_K = 64,
          .num_warps = 4,
          .num_stages = 4};
    } else if (input_dtype == torch::kHalf) {
      grouped_gemm_func = moe_gather_rs_grouped_gemm_fp16_ex;
      algo_info = moe_gather_rs_grouped_gemm_kernel__triton_algo_info_t{
          .N_SPLIT = n_split,
          .BLOCK_SIZE_M = 128,
          .BLOCK_SIZE_N = 128,
          .BLOCK_SIZE_K = 64,
          .num_warps = 4,
          .num_stages = 4};
    } else if (input_dtype == torch::kBFloat16) {
      grouped_gemm_func = moe_gather_rs_grouped_gemm_bf16_ex;
      algo_info = moe_gather_rs_grouped_gemm_kernel__triton_algo_info_t{
          .N_SPLIT = n_split,
          .BLOCK_SIZE_M = 128,
          .BLOCK_SIZE_N = 128,
          .BLOCK_SIZE_K = 64,
          .num_warps = 4,
          .num_stages = 4};
    } else {
      FLUX_CHECK(false) << "unsupported dtype " << input_dtype;
    }

    int *splits_ptr = splits_cpu.data_ptr<int>() + this->ep_start;
    int blocked_m_tiles = 0;
    int tile_size_m = algo_info.BLOCK_SIZE_M;
    for (int i = 0; i < this->ep_nexperts; i++) {
      blocked_m_tiles += (splits_ptr[i] + tile_size_m - 1) / tile_size_m;
    }
    torch::Tensor gather_a_index = empty_with_uninitialized_data(
        std::vector<int64_t>{tile_size_m * blocked_m_tiles}, option_gpu);
    torch::Tensor expert_index =
        empty_with_uninitialized_data(std::vector<int64_t>{blocked_m_tiles}, option_gpu);
    auto stream = at::cuda::getCurrentCUDAStream();
    calc_moe_triton_blocked_gather_a(
        splits_gpu.data_ptr<int>(),
        this->ep_start,
        this->ep_nexperts,
        tile_size_m,
        gather_a_index.data_ptr<int>(),
        expert_index.data_ptr<int>(),
        ep_nexperts,
        1024,
        stream);
    torch::Tensor gemm_out = empty_with_uninitialized_data(
        std::vector<int64_t>{M_this_ep, N}, option_gpu.dtype(output_dtype));

    torch::Tensor output = empty_with_uninitialized_data(
        std::vector<int64_t>{this->do_all_reduce ? ntokens : n_tokens_per_rank, N},
        at::TensorOptions(at::kCUDA).dtype(output_dtype));

    group_barrier.barrier_all(stream);

    // ensure barrier initialized correctly
    CUDA_CHECK(cudaEventRecord(this->gemm_start_event, stream));

    if (M_this_ep == 0) {
      this->barrier.fill_(1);
    } else {
      auto rtn = grouped_gemm_func(
          (CUstream)stream,
          (CUdeviceptr)input.data_ptr(),
          (CUdeviceptr)weight.data_ptr(),
          (CUdeviceptr)gemm_out.data_ptr(),
          (CUdeviceptr)data_ptr_or(input_scale, nullptr),       // input_scale
          (CUdeviceptr)data_ptr_or(weight_scale, nullptr),      // weight_scale
          (CUdeviceptr)data_ptr_or(output_vec_scale, nullptr),  // output_scale
          (CUdeviceptr)gather_a_index.data_ptr(),
          (CUdeviceptr)expert_index.data_ptr(),
          blocked_m_tiles * tile_size_m,
          N,
          K,
          ep_nexperts,
          M_this_ep,
          input.stride(0),
          input.stride(1),
          weight.stride(0),
          weight.stride(2),
          weight.stride(1),  // transpose_weight
          gemm_out.stride(0),
          gemm_out.stride(1),
          (CUdeviceptr)barrier.data_ptr(),
          algo_info);
      CU_CHECK(rtn);
    }

    // ensure barrier initialized correctly
    CUDA_CHECK(cudaStreamWaitEvent(gather_rs_stream, this->gemm_start_event));
    output = this->topk_reduce_scatter_op->run(
        {gemm_out},
        output,
        this->ep_start,
        this->ep_nexperts,
        splits_gpu,
        routing_idx,
        c10::nullopt,
        get_rs_threadblock_count(),
        (intptr_t)gather_rs_stream);
    CUDA_CHECK(cudaEventRecord(this->gather_rs_done_event, gather_rs_stream));
    CUDA_CHECK(cudaStreamWaitEvent(stream, this->gather_rs_done_event));

    group_barrier.barrier_all(stream);
    this->barrier.zero_();
    this->topk_reduce_scatter_op->reset_buffer();
    return output;
#else
    FLUX_CHECK(false) << "please compile with --triton-aot option.";
#endif
  }

  torch::Tensor
  profiling(
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync,
      c10::intrusive_ptr<ProfilingContext> opt_ctx) {
    int full_m = routing_idx.size(0);
    int K = input.size(1);
    int E = weight.size(0);
    int N = weight.size(1);
    ArchEnum arch = get_arch();
    SMCoreEnum sm_core = get_sm_core();
    auto weight_dtype = weight.scalar_type();
    auto dtype = from_torch_dtype(weight_dtype);
    bool is_fp8 = (dtype == _E4M3{}) || (dtype == _E5M2{});
    // if the dtype of input is fp8, use bfloat16 as the output dtype
    DataTypeEnum output_type = is_fp8 ? dtype : _BF16{};
    auto dt_conf =
        to_gemm_dtype_config(make_gemm_dtype_config(dtype, dtype, output_type, output_type));
    auto impl_spec = make_gemm_v2_meta(fast_accum and dt_conf.is_input_fp8());
    // always use topk=1 impl: to save some compile time
    auto comm_spec = make_gather_rs_meta(1);
    auto meta = unify_type(make_gemm_meta(
        dt_conf, arch, sm_core, _GatherRS{}, _RCR{}, _GemmGroupedV2{}(), impl_spec, comm_spec));
    auto rt_conf = make_runtime_config(cute::ceil_div(full_m, this->ep_nexperts), N, K);
    ProfilingContext tmp_ctx("__tmp__");
    ProfilingContext *ctx = opt_ctx == nullptr ? &tmp_ctx : opt_ctx.get();

    OpRegistry::instance().visit_hparams(
        [&](UnifiedGemmHParams const &hparams) {
          constexpr int warm_iters = 5;
          constexpr int iters = 10;
          float total_elapsed = 0;
          auto cp_hparams = hparams;
          auto comm_params = std::get<unified_type_t<GatherRSHParams>>(cp_hparams.comm_spec());
          if (comm_params.n_dim_per_split() != N / this->n_split) {
            return;
          }
          auto stream = c10::cuda::getCurrentCUDAStream();
          for (int iter = 0; iter < warm_iters + iters; ++iter) {
            GpuTimer timer;
            timer.start(stream);
            auto output [[maybe_unused]] = this->forward_gather_rs_impl(
                {input},
                {weight},
                splits_cpu,
                routing_idx,
                as_optional_vec(bias),
                as_optional_vec(input_scale),
                as_optional_vec(weight_scale),
                as_optional_vec(output_vec_scale),
                fast_accum,
                sm_margin,
                false,  // whether with stream sync
                cp_hparams);
            timer.stop();
            if (iter >= warm_iters) {
              total_elapsed += timer.elapsed_millis();
            }
          }

          float avg_elapsed = int(total_elapsed / iters * 1000) / 1000.0;
          ctx->add(meta, rt_conf, hparams, avg_elapsed);
        },
        meta);

    auto best_hparams = ctx->record_best(meta, rt_conf);

    return this->forward_gather_rs_impl(
        {input},
        {weight},
        splits_cpu,
        routing_idx,
        as_optional_vec(bias),
        as_optional_vec(input_scale),
        as_optional_vec(weight_scale),
        as_optional_vec(output_vec_scale),
        fast_accum,
        sm_margin,
        false,  // whether with stream sync
        std::move(best_hparams));
  }

  torch::Tensor
  forward_gather_rs_multiple(
      std::vector<torch::Tensor> inputs,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      c10::optional<std::vector<torch::Tensor>> bias,
      c10::optional<std::vector<torch::Tensor>> input_scale,
      c10::optional<std::vector<torch::Tensor>> weight_scale,
      c10::optional<std::vector<torch::Tensor>> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync) {
    // all the inputs && weights share the same splits_cpu and routing_idx
    CHECK(inputs.size() == weights.size());
    return forward_gather_rs_impl(
        std::move(inputs),
        std::move(weights),
        std::move(splits_cpu),
        std::move(routing_idx),
        std::move(bias),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_vec_scale),
        fast_accum,
        sm_margin,
        with_stream_sync,
        c10::nullopt);
  }
  std::tuple<int64_t, int64_t, int64_t>
  get_pickle_info() const {
    return std::make_tuple(this->max_m, this->n_dim, this->ep_nexperts);
  }
};

TopkReduceScatterOp::TopkReduceScatterOp(
    std::shared_ptr<Group> tp_group,
    int max_m,
    int n_dim,
    int topk,
    at::ScalarType output_dtype,
    int ep_nexperts,
    int ep_world_size,
    std::vector<torch::Tensor> barriers,
    int n_split,
    bool do_all_reduce,
    bool use_read_mode,
    int nnodes,
    bool a2av_hier,
    bool a2av_compress)
    : impl_(new TopkReduceScatterOpImpl(
          tp_group,
          max_m,
          n_dim,
          topk,
          output_dtype,
          ep_nexperts,
          ep_world_size,
          barriers,
          n_split,
          do_all_reduce,
          use_read_mode,
          nnodes,
          a2av_hier,
          a2av_compress)) {}
TopkReduceScatterOp::~TopkReduceScatterOp() { delete impl_; }
void
TopkReduceScatterOp::set_msplit_waves(
    std::vector<int> const &wave_of_node, std::vector<int> const &node_order, int n_waves) {
  FLUX_CHECK(impl_ != nullptr) << "TopkReduceScatterOp is not initialized";
  impl_->set_msplit_waves(wave_of_node, node_order, n_waves);
}
void *
TopkReduceScatterOp::send_panel_ptr() {
  FLUX_CHECK(impl_ != nullptr) << "TopkReduceScatterOp is not initialized";
  return impl_->send_panel_ptr();
}
int64_t
TopkReduceScatterOp::send_panel_rows() {
  FLUX_CHECK(impl_ != nullptr) << "TopkReduceScatterOp is not initialized";
  return impl_->send_panel_rows();
}
void
TopkReduceScatterOp::reset_buffer() {
  FLUX_CHECK(impl_ != nullptr) << "TopkReduceScatterOp not initialized";
  impl_->reset_buffer();
}
torch::Tensor
TopkReduceScatterOp::run(
    std::vector<torch::Tensor> gemm_outs,  // of group_size
    c10::optional<torch::Tensor> output,
    int ep_start,
    int ep_nexperts,
    torch::Tensor splits,
    torch::Tensor routing_idx,
    c10::optional<std::vector<torch::Tensor>> output_vec_scales,
    int num_thread_blocks,
    intptr_t cp_stream,
    c10::optional<torch::Tensor> splits_per_source,
    c10::optional<torch::Tensor> pack_index,
    c10::optional<torch::Tensor> reduce_index,
    c10::optional<torch::Tensor> unique_counts,
    c10::optional<std::vector<torch::Tensor>> wire_csr,
    c10::optional<std::vector<torch::Tensor>> reduce_csr) {
  FLUX_CHECK(impl_ != nullptr) << "TopkReduceScatterOp not initialized";
  return impl_->run(
      std::move(gemm_outs),
      std::move(output),
      ep_start,
      ep_nexperts,
      std::move(splits),
      std::move(routing_idx),
      std::move(output_vec_scales),
      num_thread_blocks,
      cp_stream,
      std::move(splits_per_source),
      std::move(pack_index),
      std::move(reduce_index),
      std::move(unique_counts),
      std::move(wire_csr),
      std::move(reduce_csr));
}

std::vector<torch::Tensor>
TopkReduceScatterOp::derive_combine_meta(
    torch::Tensor splits_gpu,
    torch::Tensor routing_idx,
    torch::Tensor splits_per_source,
    c10::optional<torch::Tensor> a2av_unique_counts) {
  FLUX_CHECK(impl_ != nullptr) << "TopkReduceScatterOp not initialized";
  return impl_->derive_combine_meta(
      std::move(splits_gpu),
      std::move(routing_idx),
      std::move(splits_per_source),
      std::move(a2av_unique_counts));
}

GemmGroupedV2GatherRSOp::GemmGroupedV2GatherRSOp(
    std::shared_ptr<Group> tp_group_,
    int64_t total_num_experts,
    int64_t max_m,
    int64_t n_dim,
    int64_t topk,
    at::ScalarType output_dtype,
    int64_t tp_world_size,
    int64_t ep_world_size,
    int64_t max_input_groups,
    int64_t n_split_,
    bool do_all_reduce,
    bool use_read_mode,
    int64_t nnodes,
    bool a2av_hier,
    bool a2av_hier_compress)
    : impl_(new GemmGroupedV2GatherRSOpImpl(
          tp_group_,
          total_num_experts,
          max_m,
          n_dim,
          topk,
          output_dtype,
          tp_world_size,
          ep_world_size,
          max_input_groups,
          n_split_,
          do_all_reduce,
          use_read_mode,
          nnodes,
          a2av_hier,
          a2av_hier_compress)) {}

GemmGroupedV2GatherRSOp::~GemmGroupedV2GatherRSOp() { delete impl_; }
torch::Tensor
GemmGroupedV2GatherRSOp::forward_gather_rs(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor splits_cpu,
    torch::Tensor routing_idx,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_vec_scale,
    bool fast_accum,
    int sm_margin,
    bool with_stream_sync,
    c10::optional<torch::Tensor> splits_per_source,
    c10::optional<torch::Tensor> a2av_pack_index,
    c10::optional<torch::Tensor> a2av_reduce_index,
    c10::optional<torch::Tensor> a2av_unique_counts,
    c10::optional<std::vector<torch::Tensor>> a2av_wire_csr,
    c10::optional<std::vector<torch::Tensor>> a2av_reduce_csr) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2GatherRSOp not initialized";
  return impl_->forward_gather_rs(
      std::move(input),
      std::move(weight),
      std::move(splits_cpu),
      std::move(routing_idx),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_vec_scale),
      fast_accum,
      sm_margin,
      with_stream_sync,
      std::move(splits_per_source),
      std::move(a2av_pack_index),
      std::move(a2av_reduce_index),
      std::move(a2av_unique_counts),
      std::move(a2av_wire_csr),
      std::move(a2av_reduce_csr));
}
std::vector<torch::Tensor>
GemmGroupedV2GatherRSOp::derive_combine_meta(
    torch::Tensor splits_gpu,
    torch::Tensor routing_idx,
    torch::Tensor splits_per_source,
    c10::optional<torch::Tensor> a2av_unique_counts) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2GatherRSOp not initialized";
  return impl_->derive_combine_meta(
      std::move(splits_gpu),
      std::move(routing_idx),
      std::move(splits_per_source),
      std::move(a2av_unique_counts));
}
torch::Tensor
GemmGroupedV2GatherRSOp::forward_gather_rs_triton_aot(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor splits,
    torch::Tensor routing_idx,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_vec_scale,
    bool fast_accum,
    int sm_margin,
    bool with_stream_sync) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2GatherRSOp not initialized";
  return impl_->forward_gather_rs_triton_aot(
      std::move(input),
      std::move(weight),
      std::move(splits),
      std::move(routing_idx),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_vec_scale),
      fast_accum,
      sm_margin,
      with_stream_sync);
}
torch::Tensor
GemmGroupedV2GatherRSOp::profiling(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor splits_cpu,
    torch::Tensor routing_idx,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_vec_scale,
    bool fast_accum,
    int sm_margin,
    bool with_stream_sync,
    c10::intrusive_ptr<ProfilingContext> opt_ctx) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2GatherRSOp not initialized";
  return impl_->profiling(
      std::move(input),
      std::move(weight),
      std::move(splits_cpu),
      std::move(routing_idx),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_vec_scale),
      fast_accum,
      sm_margin,
      with_stream_sync,
      std::move(opt_ctx));
}
torch::Tensor
GemmGroupedV2GatherRSOp::forward_gather_rs_multiple(
    std::vector<torch::Tensor> inputs,
    std::vector<torch::Tensor> weights,
    torch::Tensor splits_cpu,
    torch::Tensor routing_idx,
    c10::optional<std::vector<torch::Tensor>> bias,
    c10::optional<std::vector<torch::Tensor>> input_scale,
    c10::optional<std::vector<torch::Tensor>> weight_scale,
    c10::optional<std::vector<torch::Tensor>> output_vec_scale,
    bool fast_accum,
    int sm_margin,
    bool with_stream_sync) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2GatherRSOp not initialized";
  return impl_->forward_gather_rs_multiple(
      std::move(inputs),
      std::move(weights),
      std::move(splits_cpu),
      std::move(routing_idx),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_vec_scale),
      fast_accum,
      sm_margin,
      with_stream_sync);
}

}  // namespace bytedance::flux::ths_op
