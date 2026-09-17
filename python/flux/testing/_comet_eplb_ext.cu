// Copyright 2026 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License").
//
// COMET+EPLB fused router: ONE launch mapping this rank's LOGICAL routing
// [S, K] to PHYSICAL slots [S, K] under the eplb arm's sender-local replica
// rules, bit-exact with EplbIterPlanner.derive_fused (flux.testing
// eplb_semantics / ep_gpu_plan: local_spread_rank_quota_prefix,
// d6_rank_quota_prefix, reroute_expand_gpu + _interleave_params,
// comb_dst_slot_from_topk). Replaces ~8 torch ops with 3 device syncs
// (~2 ms/iter in the plan_comm bracket) by a single kernel.
//
// Semantics per logical expert l (one thread block each):
//   n        = number of this rank's entries routed to l (dense routing
//              contract: at most one entry per token per expert, so the
//              expert-major, token-ascending order of the reference equals
//              the flat (t, k) scan order restricted to l)
//   ordinal  = position of the entry in that order, 0..n-1
//   interleave (n > 1): stride = first coprime of n scanning from
//              clamp(n/2+1, 1, n-1) with wrap to 1; offset = l % n;
//              qr = (ordinal*stride + offset) % n; else qr = ordinal
//   local_spread: instance j gets q_j = n/C + (j < n%C); prefix_j =
//              cumsum; replica = #{j < C : prefix_j <= qr} (clamped C-1)
//   local_static: replica = src % C
//   out[t, k] = l2p[l, replica]
// Grid = G blocks; each block scans the N = S*K entries twice (count, then
// assign) with a block-wide ballot prefix, so ordinals are deterministic.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ int gcd_dev(int a, int b) {
  while (b != 0) {
    int t = a % b;
    a = b;
    b = t;
  }
  return a;
}

__device__ __forceinline__ int interleave_stride(int n) {
  // ultraep_semantics._interleave_params (reroute.cu:189-209 port)
  int stride = n / 2 + 1;
  if (stride >= n) stride = n - 1;
  if (stride < 1) stride = 1;
  while (gcd_dev(stride, n) != 1) {
    stride += 1;
    if (stride >= n) stride = 1;
  }
  return stride;
}

__global__ void route_local_kernel(const int* __restrict__ topk,   // [N]
                                   const int* __restrict__ l2p,    // [G, Cmax]
                                   const int* __restrict__ lcnts,  // [G]
                                   int* __restrict__ out,          // [N]
                                   int N, int Cmax, int src, int mode,
                                   int interleave) {
  const int l = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  __shared__ int s_warp[kWarps];
  __shared__ int s_n;
  __shared__ int s_stride;
  __shared__ int s_offset;
  __shared__ int s_running;

  // pass 1: n = count of entries routed to l
  int cnt = 0;
  for (int i = tid; i < N; i += kThreads) cnt += (topk[i] == l);
  for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
  if (lane == 0) s_warp[warp] = cnt;
  __syncthreads();
  if (tid == 0) {
    int n = 0;
    for (int w = 0; w < kWarps; ++w) n += s_warp[w];
    s_n = n;
    s_running = 0;
    if (interleave && n > 1) {
      s_stride = interleave_stride(n);
      s_offset = l % n;
    } else {
      s_stride = 1;
      s_offset = 0;
    }
  }
  __syncthreads();
  const int n = s_n;
  if (n == 0) return;
  const int C = lcnts[l];
  const int stride = s_stride;
  const int offset = s_offset;
  const int base_q = n / C;
  const int rem_q = n % C;
  const int j_static = src % C;

  // pass 2: deterministic ordinals via block-wide ballot prefix, in order
  for (int chunk = 0; chunk < N; chunk += kThreads) {
    const int i = chunk + tid;
    const bool match = (i < N) && (topk[i] == l);
    const unsigned ballot = __ballot_sync(0xffffffffu, match);
    const int warp_cnt = __popc(ballot);
    const int in_warp = __popc(ballot & ((1u << lane) - 1u));
    if (lane == 0) s_warp[warp] = warp_cnt;
    __syncthreads();
    int before = 0;
    for (int w = 0; w < warp; ++w) before += s_warp[w];
    if (match) {
      const int ordinal = s_running + before + in_warp;
      int qr = ordinal;
      if (interleave && n > 1) qr = (ordinal * stride + offset) % n;
      int replica;
      if (mode == 1) {
        replica = j_static;
      } else {
        // #{j : prefix_j <= qr}, prefix_j = (j+1)*base + min(j+1, rem)
        replica = 0;
        for (int j = 0; j < C; ++j) {
          const int pre = (j + 1) * base_q + min(j + 1, rem_q);
          replica += (pre <= qr);
        }
        if (replica > C - 1) replica = C - 1;
      }
      out[i] = l2p[l * Cmax + replica];
    }
    __syncthreads();
    if (tid == 0) {
      int tot = 0;
      for (int w = 0; w < kWarps; ++w) tot += s_warp[w];
      s_running += tot;
    }
    __syncthreads();
  }
}

}  // namespace

torch::Tensor route_local(torch::Tensor topk, torch::Tensor l2p,
                          torch::Tensor lcnts, int64_t src, int64_t mode,
                          bool interleave) {
  TORCH_CHECK(topk.is_cuda() && l2p.is_cuda() && lcnts.is_cuda(), "cuda tensors");
  TORCH_CHECK(topk.dtype() == torch::kInt32 && l2p.dtype() == torch::kInt32 &&
                  lcnts.dtype() == torch::kInt32,
              "int32 tensors");
  TORCH_CHECK(topk.dim() == 2 && l2p.dim() == 2 && lcnts.dim() == 1, "shapes");
  TORCH_CHECK(mode == 0 || mode == 1, "mode: 0 = local_spread, 1 = local_static");
  auto topk_c = topk.contiguous();
  auto l2p_c = l2p.contiguous();
  auto lcnts_c = lcnts.contiguous();
  const int G = static_cast<int>(lcnts_c.size(0));
  TORCH_CHECK(l2p_c.size(0) == G, "l2p rows != G");
  const int Cmax = static_cast<int>(l2p_c.size(1));
  const int N = static_cast<int>(topk_c.numel());
  auto out = torch::empty_like(topk_c);
  if (N == 0 || G == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream();
  route_local_kernel<<<G, kThreads, 0, stream>>>(
      topk_c.data_ptr<int>(), l2p_c.data_ptr<int>(), lcnts_c.data_ptr<int>(),
      out.data_ptr<int>(), N, Cmax, static_cast<int>(src),
      static_cast<int>(mode), interleave ? 1 : 0);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("route_local", &route_local,
        "COMET+EPLB fused sender-local router: logical [S,K] -> physical [S,K]");
}
