// Standalone JIT build of the PLACE-lambda sender-local router
// (8.22.route): the PATCHED kernel — templated register-mask tier-3
// cover + FLUX_LOCCAP_REMOTE_CAP_ONLY flavor — compiled independently of
// libflux_cuda.so so the kernel A/B needs no full flux rebuild. Source
// of truth: src/cuda/moe_utils.cu (this file is extracted from it by
// tooling; edit there).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <tuple>

#define CUDA_CHECK(expr)                                        \
  do {                                                          \
    cudaError_t _e = (expr);                                    \
    TORCH_CHECK(_e == cudaSuccess, "CUDA error: ",              \
                cudaGetErrorString(_e));                        \
  } while (0)

namespace pll_ext {
// PLACE-lambda sender-local LocCap router (placelambda_route_sl).
//
// Fused per-iteration replica-selection for the pll_* arms: each rank
// routes ONLY its own [S, K] gating entries. All shared tables (own-rank
// quotas, home-node grants, tier-3 capacity shares) are pure
// order-independent integer functions of the allgathered demand histogram
// d[R, G], so every rank derives identical tables with no cross-rank
// coordination; the per-entry assignment then uses RELAXED atomic tickets
// (user ruling 2026-08-21: no bit-determinism required — agreement across
// ranks comes from the phys-row allgather / counts exchange, never from
// replaying each other's decisions). Reference algorithm and offline
// simulator: python/flux/testing/placelambda_gpu.py::loccap_route_sl
// (tables must match it exactly; per-entry choices are free within the
// table quotas).
//
// Tier 1  own-rank quota (largest remainder within cap)
// Tier 2  home-node grant: node demand -> hosting ranks proportional to
//         residual capacity, per-rank clip, per-source prefix split
// Tier 3  per-token greedy node-cover on 32-bit node masks, consuming
//         this rank's pre-partitioned shares via atomic tickets
// forced  leftovers to a precomputed least-loaded hosting rank
//
// Largest remainder uses O(n^2) fraction-ranking (n <= 1024) instead of a
// sort — deterministic, tiny at these sizes.
//===--------------------------------------------------------------------===//

namespace {

constexpr int kPllThreads = 256;
constexpr int kWClamp = 1 << 16;  // proportional-weight clamp (overflow guard)

// rank of frac[i] under (frac desc, index asc); frac in shared memory
__device__ __forceinline__ int
pll_frac_rank(const long long *frac, int n, int i) {
  int r = 0;
  long long fi = frac[i];
  for (int j = 0; j < n; ++j) {
    long long fj = frac[j];
    r += (fj > fi) || (fj == fi && j < i);
  }
  return r;
}

// ipr[g*R + r] = physical slot of g's instance on rank r (-1 none);
// covmask[g] = bitmask of nodes hosting g
__global__ void
pll_ipr_kernel(
    const int *l2p, const int *lcnts, int *ipr, unsigned *covmask,
    int G, int R, int Cmax, int nlp, int L) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= G) return;
  unsigned cm = 0;
  int C = lcnts[g];
  for (int j = 0; j < C; ++j) {
    int phys = l2p[g * Cmax + j];
    if (phys < 0) continue;
    int r = phys / nlp;
    ipr[g * R + r] = phys;
    cm |= 1u << (r / L);
  }
  covmask[g] = cm;
}

// per-source own-rank quota (tier 1); one block per source rank
__global__ void
pll_q1_kernel(
    const int *d, const int *ipr, int *q1, int *load,
    int G, int R, long long cap) {
  extern __shared__ long long s_frac[];  // [G]
  int src = blockIdx.x;
  __shared__ long long s_tot, s_base;
  if (threadIdx.x == 0) s_tot = s_base = 0;
  __syncthreads();
  long long tot = 0;
  for (int g = threadIdx.x; g < G; g += blockDim.x) {
    long long w = (ipr[g * R + src] >= 0) ? d[src * G + g] : 0;
    s_frac[g] = w;  // stage the hosted demand
    tot += w;
  }
  atomicAdd((unsigned long long *)&s_tot, (unsigned long long)tot);
  __syncthreads();
  tot = s_tot;
  if (tot <= cap) {
    for (int g = threadIdx.x; g < G; g += blockDim.x)
      q1[src * G + g] = (int)s_frac[g];
    if (threadIdx.x == 0) load[src] = (int)tot;
    return;
  }
  long long sum_base = 0;
  for (int g = threadIdx.x; g < G; g += blockDim.x) {
    long long w = s_frac[g];
    long long base = w * cap / tot;
    q1[src * G + g] = (int)base;
    sum_base += base;
  }
  atomicAdd((unsigned long long *)&s_base, (unsigned long long)sum_base);
  __syncthreads();
  // second pass: frac keys, then rank-and-bump
  for (int g = threadIdx.x; g < G; g += blockDim.x) {
    long long w = s_frac[g];
    s_frac[g] = w * cap - (long long)q1[src * G + g] * tot;
  }
  __syncthreads();
  int rem = (int)(cap - s_base);
  for (int g = threadIdx.x; g < G; g += blockDim.x)
    if (pll_frac_rank(s_frac, G, g) < rem) q1[src * G + g] += 1;
  if (threadIdx.x == 0) load[src] = (int)cap;
}

// tier-2 node grant, proportional to residual (one thread per (node, g))
__global__ void
pll_t2alloc_kernel(
    const int *d, const int *q1, const int *ipr, const unsigned *covmask,
    const int *load, int *allocT,
    int G, int R, int L, int NN, int cap) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= NN * G) return;
  int u = idx / G, g = idx % G;
  bool hosted_here = (covmask[g] >> u) & 1u;
  long long D = 0;
  if (hosted_here)
    for (int l = 0; l < L; ++l) {
      int src = u * L + l;
      D += d[src * G + g] - q1[src * G + g];
    }
  long long w[32], toti = 0;
  for (int l = 0; l < L; ++l) {
    int r = u * L + l;
    long long resid = max(0, cap - load[r]);
    w[l] = (ipr[g * R + r] >= 0) ? min(resid, (long long)kWClamp) : 0;
    toti += w[l];
  }
  int *out = allocT + (u * G + g) * L;
  if (D == 0 || toti == 0) {
    for (int l = 0; l < L; ++l) out[l] = 0;
    return;
  }
  // largest remainder of D over w (n = L, in registers)
  long long base[32], frac[32], sb = 0;
  for (int l = 0; l < L; ++l) {
    base[l] = w[l] * D / toti;
    frac[l] = w[l] * D - base[l] * toti;
    sb += base[l];
  }
  int rem = (int)(D - sb);
  for (int l = 0; l < L; ++l) {
    int rk = 0;
    for (int j = 0; j < L; ++j)
      rk += (frac[j] > frac[l]) || (frac[j] == frac[l] && j < l);
    out[l] = (int)(base[l] + (rk < rem ? 1 : 0));
  }
}

// per-destination clip of the tier-2 grant to the residual (block per rank)
__global__ void
pll_t2clip_kernel(
    int *allocT, int *load, int G, int L, int cap) {
  extern __shared__ long long s_frac[];  // [G]
  int r = blockIdx.x;
  int u = r / L, l = r % L;
  __shared__ long long s_tot, s_base;
  if (threadIdx.x == 0) s_tot = s_base = 0;
  __syncthreads();
  long long resid = max(0, cap - load[r]);
  long long tot = 0;
  for (int g = threadIdx.x; g < G; g += blockDim.x) {
    long long w = allocT[(u * G + g) * L + l];
    s_frac[g] = w;
    tot += w;
  }
  atomicAdd((unsigned long long *)&s_tot, (unsigned long long)tot);
  __syncthreads();
  tot = s_tot;
  if (tot <= resid) {
    if (threadIdx.x == 0) load[r] += (int)tot;
    return;
  }
  long long sum_base = 0;
  for (int g = threadIdx.x; g < G; g += blockDim.x) {
    long long w = s_frac[g];
    long long base = w * resid / tot;
    allocT[(u * G + g) * L + l] = (int)base;
    sum_base += base;
  }
  atomicAdd((unsigned long long *)&s_base, (unsigned long long)sum_base);
  __syncthreads();
  for (int g = threadIdx.x; g < G; g += blockDim.x) {
    long long w = s_frac[g];
    s_frac[g] = w * resid - (long long)allocT[(u * G + g) * L + l] * tot;
  }
  __syncthreads();
  int rem = (int)(resid - s_base);
  for (int g = threadIdx.x; g < G; g += blockDim.x)
    if (pll_frac_rank(s_frac, G, g) < rem)
      allocT[(u * G + g) * L + l] += 1;
  if (threadIdx.x == 0) load[r] += (int)resid;
}

// per-source split of the node grant (prefix intervals) -> granted2 for
// every source; cumulative boundaries + target ranks for src == my_rank
__global__ void
pll_t2split_kernel(
    const int *d, const int *q1, const int *allocT, int *granted2,
    int *bound, int *tgt, int my_rank,
    int G, int R, int L, int NN) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= R * G) return;
  int src = idx / G, g = idx % G;
  int u = src / L, lsrc = src % L;
  long long pref = 0, want = 0, cum = 0, Gug = 0;
  for (int l = 0; l < L; ++l) {
    int s2 = u * L + l;
    long long rdv = d[s2 * G + g] - q1[s2 * G + g];
    // t2_want is nonzero only when the node hosts g; allocT rows are zero
    // otherwise, so the interval math self-gates
    if (l < lsrc) pref += rdv;
    if (l == lsrc) want = rdv;
    Gug += allocT[(u * G + g) * L + l];
  }
  if (Gug == 0 || want == 0) {
    granted2[src * G + g] = q1[src * G + g];
    if (src == my_rank) {
      for (int j = 0; j < L; ++j) {
        bound[g * L + j] = q1[src * G + g];
        tgt[g * L + j] = u * L + ((j == 0) ? lsrc : (j <= lsrc ? j - 1 : j));
      }
    }
    return;
  }
  long long lo = min(pref, Gug), hi = min(pref + want, Gug);
  granted2[src * G + g] = q1[src * G + g] + (int)(hi - lo);
  if (src != my_rank) return;
  // per-destination overlap for my row; targets own-first then ascending
  long long own_extra = 0, amt[32];
  cum = 0;
  for (int l = 0; l < L; ++l) {
    long long c0 = cum;
    cum += allocT[(u * G + g) * L + l];
    long long ov = min(hi, cum) - max(lo, c0);
    amt[l] = ov > 0 ? ov : 0;
  }
  own_extra = amt[lsrc];
  int j = 0;
  long long acc = q1[src * G + g] + own_extra;
  bound[g * L + 0] = (int)acc;
  tgt[g * L + 0] = src;
  j = 1;
  for (int l = 0; l < L; ++l) {
    if (l == lsrc) continue;
    acc += amt[l];
    bound[g * L + j] = (int)acc;
    tgt[g * L + j] = u * L + l;
    ++j;
  }
}

// tier-3 share weights w3[s, r] (thread per (src, dst), loop over G)
__global__ void
pll_w3_kernel(
    const int *d, const int *granted2, const int *ipr, long long *w3,
    int G, int R) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= R * R) return;
  int s = idx / R, r = idx % R;
  long long acc = 0;
  for (int g = 0; g < G; ++g) {
    if (ipr[g * R + r] < 0) continue;
    int lo = d[s * G + g] - granted2[s * G + g];
    acc += min(max(lo, 0), kWClamp);
  }
  w3[s * R + r] = min(acc, (long long)kWClamp);
}

// tier-3 shares: partition each destination's residual over sources
// (block per destination rank); also emits my share row and the forced
// fallback host per expert
__global__ void
pll_shares_kernel(
    const long long *w3, const int *load, const int *ipr, int *share_my,
    int *fallback, int *forced_left, int f_cap, int my_rank, int G, int R,
    int cap, int remote_cap_only) {
  extern __shared__ long long s_frac[];  // [R]
  int r = blockIdx.x;
  // remote_cap_only (FLUX_LOCCAP_REMOTE_CAP_ONLY): tiers 1+2 are
  // intra-node (zero wire bytes) and ran uncapped, so the eps budget
  // applies in FULL to the tier-3 cross-node shares; the fallback choice
  // below still uses the real load (least-loaded quality preserved)
  long long resid = remote_cap_only ? (long long)cap
                                    : max(0, cap - load[r]);
  __shared__ long long s_tot, s_base;
  if (threadIdx.x == 0) s_tot = s_base = 0;
  __syncthreads();
  long long tot = 0;
  for (int s = threadIdx.x; s < R; s += blockDim.x) {
    s_frac[s] = w3[s * R + r];
    tot += s_frac[s];
  }
  atomicAdd((unsigned long long *)&s_tot, (unsigned long long)tot);
  __syncthreads();
  tot = s_tot;
  int mine = 0;
  if (tot <= resid) {
    if (threadIdx.x == 0) mine = (int)s_frac[my_rank];
  } else {
    long long sum_base = 0;
    __shared__ int s_bs[1024];
    for (int s = threadIdx.x; s < R; s += blockDim.x) {
      long long w = s_frac[s];
      long long base = w * resid / tot;
      s_bs[s] = (int)base;
      sum_base += base;
    }
    atomicAdd((unsigned long long *)&s_base, (unsigned long long)sum_base);
    __syncthreads();
    for (int s = threadIdx.x; s < R; s += blockDim.x) {
      long long w = s_frac[s];
      s_frac[s] = w * resid - (long long)s_bs[s] * tot;
    }
    __syncthreads();
    int rem = (int)(resid - s_base);
    if (threadIdx.x == 0) {
      int rk = pll_frac_rank(s_frac, R, my_rank);
      mine = s_bs[my_rank] + (rk < rem ? 1 : 0);
    }
  }
  if (threadIdx.x == 0) share_my[r] = mine;
  // per-(src, dst) forced-admission budget (the ONE sizing clamp — makes
  // pair_ub/recv_ub provable; <=0 means unlimited)
  if (threadIdx.x == 0)
    forced_left[r] = (f_cap <= 0) ? (INT_MAX / 2) : f_cap;
  // forced fallback per expert: least-loaded hosting rank (grid-stride on
  // block 0 only, once)
  if (r == 0) {
    for (int g = threadIdx.x; g < G; g += blockDim.x) {
      long long best = LLONG_MAX;
      int bestr = -1;
      for (int rr = 0; rr < R; ++rr) {
        if (ipr[g * R + rr] < 0) continue;
        long long key = (long long)load[rr] * R + rr;
        if (key < best) { best = key; bestr = rr; }
      }
      fallback[g] = bestr;
    }
  }
}

// tiers 1+2 assignment via relaxed tickets (thread per entry)
__global__ void
pll_route12_kernel(
    const int *topk_own, const int *bound, const int *tgt, const int *ipr,
    int *cnt, int *phys_own, long long *stats,
    int S, int K, int G, int R, int L) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= S * K) return;
  int g = topk_own[e];
  int t = atomicAdd(&cnt[g], 1);
  int total = bound[g * L + (L - 1)];
  if (t < total) {
    int j = 0;
    while (bound[g * L + j] <= t) ++j;  // L <= 32, linear
    phys_own[e] = ipr[g * R + tgt[g * L + j]];
  } else {
    phys_own[e] = -1;
    atomicAdd((unsigned long long *)&stats[1], 1ull);
  }
}

// tier-3 greedy node cover on this rank's shares (thread per token)
__global__ void
pll_route3_kernel(
    const int *topk_own, const unsigned *covmask, const int *ipr,
    const int *fallback, int *share_my, int *forced_left, int *phys_own,
    long long *stats, int S, int K, int G, int R, int L, int NN,
    int home_node) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= S) return;
  int gs[32];
  int nrem = 0;
  for (int k = 0; k < K; ++k)
    if (phys_own[s * K + k] < 0) gs[nrem++] = k;
  if (nrem == 0) return;
  unsigned dead_nodes = 0;
  int attempts = 0;
  while (nrem > 0 && attempts < NN + 32) {
    ++attempts;
    // pick the node covering most remaining experts (home wins ties, then
    // lower node id)
    int best_cnt = 0, best_n = -1;
    for (int n = 0; n < NN; ++n) {
      if ((dead_nodes >> n) & 1u) continue;
      int c = 0;
      for (int i = 0; i < nrem; ++i) {
        int g = topk_own[s * K + gs[i]];
        c += (covmask[g] >> n) & 1u;
      }
      bool better = c > best_cnt ||
                    (c == best_cnt && c > 0 &&
                     (n == home_node && best_n != home_node));
      if (better) { best_cnt = c; best_n = n; }
    }
    if (best_cnt == 0) break;
    int n = best_n;
    bool claimed_any = false;
    for (int i = 0; i < nrem;) {
      int k = gs[i];
      int g = topk_own[s * K + k];
      if (!((covmask[g] >> n) & 1u)) { ++i; continue; }
      bool done = false;
      for (int l = 0; l < L; ++l) {
        int r = n * L + l;
        if (ipr[g * R + r] < 0) continue;
        int old = atomicSub(&share_my[r], 1);
        if (old > 0) {
          phys_own[s * K + k] = ipr[g * R + r];
          done = true;
          break;
        }
        atomicAdd(&share_my[r], 1);
      }
      if (done) {
        claimed_any = true;
        gs[i] = gs[--nrem];
      } else {
        ++i;
      }
    }
    if (!claimed_any) dead_nodes |= 1u << n;  // node exhausted for me
  }
  for (int i = 0; i < nrem; ++i) {
    int k = gs[i];
    int g = topk_own[s * K + k];
    // forced admission under the per-(src, dst) budget: fallback first,
    // then the other hosting ranks ascending; a fully exhausted budget
    // assigns the fallback anyway and bumps the LOUD overflow counter
    // stats[2] (the driver asserts it stays 0 — sizing-contract breach)
    int r = fallback[g];
    int old = atomicSub(&forced_left[r], 1);
    if (old <= 0) {
      atomicAdd(&forced_left[r], 1);
      bool ok = false;
      for (int rr = 0; rr < R && !ok; ++rr) {
        if (rr == r || ipr[g * R + rr] < 0) continue;
        int o2 = atomicSub(&forced_left[rr], 1);
        if (o2 > 0) {
          r = rr;
          ok = true;
        } else {
          atomicAdd(&forced_left[rr], 1);
        }
      }
      if (!ok) atomicAdd((unsigned long long *)&stats[2], 1ull);
    }
    phys_own[s * K + k] = ipr[g * R + r];
    atomicAdd((unsigned long long *)&stats[0], 1ull);
  }
}

// templated tier-3 cover: per-entry coverage masks live in REGISTERS
// (full unroll over KT) and the remaining-set is a KT-bit word — the
// generic version's gs[32] local array spills to local memory and its
// O(NN * nrem) rescan dominates the kernel at R=128 (measured 3.2 ms;
// the [32]-array suspicion from the scale suite). Same relaxed-ticket
// semantics; K in {8, 16} dispatched, generic fallback otherwise.
template <int KT>
__global__ void
pll_route3_kernel_t(
    const int *topk_own, const unsigned *covmask, const int *ipr,
    const int *fallback, int *share_my, int *forced_left, int *phys_own,
    long long *stats, int S, int G, int R, int L, int NN,
    int home_node) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= S) return;
  unsigned rem = 0;
  unsigned m[KT];
#pragma unroll
  for (int k = 0; k < KT; ++k) {
    bool need = phys_own[s * KT + k] < 0;
    m[k] = need ? covmask[topk_own[s * KT + k]] : 0u;
    rem |= need ? (1u << k) : 0u;
  }
  if (!rem) return;
  unsigned dead_nodes = 0;
  int attempts = 0;
  while (rem && attempts < NN + 32) {
    ++attempts;
    int best_cnt = 0, best_n = -1;
    for (int n = 0; n < NN; ++n) {
      if ((dead_nodes >> n) & 1u) continue;
      int c = 0;
#pragma unroll
      for (int k = 0; k < KT; ++k)
        c += (int)(((rem >> k) & 1u) & ((m[k] >> n) & 1u));
      bool better = c > best_cnt ||
                    (c == best_cnt && c > 0 &&
                     (n == home_node && best_n != home_node));
      if (better) { best_cnt = c; best_n = n; }
    }
    if (best_cnt == 0) break;
    int n = best_n;
    bool claimed_any = false;
#pragma unroll
    for (int k = 0; k < KT; ++k) {
      if (!((rem >> k) & 1u) || !((m[k] >> n) & 1u)) continue;
      int g = topk_own[s * KT + k];
      bool done = false;
      for (int l = 0; l < L; ++l) {
        int r = n * L + l;
        if (ipr[g * R + r] < 0) continue;
        int old = atomicSub(&share_my[r], 1);
        if (old > 0) {
          phys_own[s * KT + k] = ipr[g * R + r];
          done = true;
          break;
        }
        atomicAdd(&share_my[r], 1);
      }
      if (done) {
        claimed_any = true;
        rem &= ~(1u << k);
      }
    }
    if (!claimed_any) dead_nodes |= 1u << n;
  }
#pragma unroll
  for (int k = 0; k < KT; ++k) {
    if (!((rem >> k) & 1u)) continue;
    int g = topk_own[s * KT + k];
    int r = fallback[g];
    int old = atomicSub(&forced_left[r], 1);
    if (old <= 0) {
      atomicAdd(&forced_left[r], 1);
      bool ok = false;
      for (int rr = 0; rr < R && !ok; ++rr) {
        if (rr == r || ipr[g * R + rr] < 0) continue;
        int o2 = atomicSub(&forced_left[rr], 1);
        if (o2 > 0) {
          r = rr;
          ok = true;
        } else {
          atomicAdd(&forced_left[rr], 1);
        }
      }
      if (!ok) atomicAdd((unsigned long long *)&stats[2], 1ull);
    }
    phys_own[s * KT + k] = ipr[g * R + r];
    atomicAdd((unsigned long long *)&stats[0], 1ull);
  }
}

}  // namespace

void
placelambda_route_sl(
    const int *topk_own,   // [S, K] this rank's gating
    const int *d,          // [R, G] allgathered demand histograms
    const int *l2p,        // [G, Cmax]
    const int *lcnts,      // [G]
    int *phys_own,         // [S, K] out
    long long *stats,      // [4] out (zeroed): forced, tier3_entries,
                           //                   forced_budget_overflow
    void *workspace,       // placelambda_route_sl_workspace_bytes(...)
    int S, int K, int G, int R, int Cmax, int nlp, int ranks_per_node,
    int my_rank, long long cap64, int f_cap, cudaStream_t stream) {
  // capability tag: the literal below in the built .so is the sweep
  // runner's probe for this kernel (never remove)
  static const char *kTag = "FLUX_PLACELAMBDA_ROUTE_SL_TAG";
  (void)std::getenv(kTag);
  int L = ranks_per_node;
  int NN = R / L;
  int cap = (int)std::min(cap64, (long long)S * K * R);
  // remote-only-cap flavor (FLUX_LOCCAP_REMOTE_CAP_ONLY=1): tiers 1+2
  // uncapped (intra-node, zero wire bytes), the eps cap budgets only the
  // tier-3/forced cross-node residue — must mirror the torch reference
  // (placelambda_gpu.loccap_route_sl remote_cap_only=True) table-exactly
  const char *rco_env = std::getenv("FLUX_LOCCAP_REMOTE_CAP_ONLY");
  const int rco = (rco_env != nullptr && rco_env[0] == '1') ? 1 : 0;
  const int cap12 = rco ? (int)std::min((long long)S * K * R,
                                        (long long)INT_MAX / 2)
                        : cap;
  char *ws = (char *)workspace;
  int *ipr = (int *)ws;                 ws += sizeof(int) * G * R;
  unsigned *covmask = (unsigned *)ws;   ws += sizeof(int) * G;
  int *q1 = (int *)ws;                  ws += sizeof(int) * R * G;
  int *load = (int *)ws;                ws += sizeof(int) * R;
  int *allocT = (int *)ws;              ws += sizeof(int) * NN * G * L;
  int *granted2 = (int *)ws;            ws += sizeof(int) * R * G;
  int *bound = (int *)ws;               ws += sizeof(int) * G * L;
  int *tgt = (int *)ws;                 ws += sizeof(int) * G * L;
  long long *w3 = (long long *)ws;      ws += sizeof(long long) * R * R;
  int *share_my = (int *)ws;            ws += sizeof(int) * R;
  int *fallback = (int *)ws;            ws += sizeof(int) * G;
  int *cnt = (int *)ws;                 ws += sizeof(int) * G;
  int *forced_left = (int *)ws;         ws += sizeof(int) * R;

  CUDA_CHECK(cudaMemsetAsync(ipr, 0xFF, sizeof(int) * G * R, stream));
  CUDA_CHECK(cudaMemsetAsync(covmask, 0, sizeof(int) * G, stream));
  CUDA_CHECK(cudaMemsetAsync(cnt, 0, sizeof(int) * G, stream));
  int tb = kPllThreads;
  auto blocks = [tb](int n) { return (n + tb - 1) / tb; };
  size_t smemG = sizeof(long long) * G;
  size_t smemR = sizeof(long long) * R;
  pll_ipr_kernel<<<blocks(G), tb, 0, stream>>>(
      l2p, lcnts, ipr, covmask, G, R, Cmax, nlp, L);
  pll_q1_kernel<<<R, tb, smemG, stream>>>(d, ipr, q1, load, G, R, cap12);
  pll_t2alloc_kernel<<<blocks(NN * G), tb, 0, stream>>>(
      d, q1, ipr, covmask, load, allocT, G, R, L, NN, cap12);
  pll_t2clip_kernel<<<R, tb, smemG, stream>>>(allocT, load, G, L, cap12);
  pll_t2split_kernel<<<blocks(R * G), tb, 0, stream>>>(
      d, q1, allocT, granted2, bound, tgt, my_rank, G, R, L, NN);
  pll_w3_kernel<<<blocks(R * R), tb, 0, stream>>>(
      d, granted2, ipr, w3, G, R);
  pll_shares_kernel<<<R, tb, smemR, stream>>>(
      w3, load, ipr, share_my, fallback, forced_left, f_cap, my_rank, G, R,
      cap, rco);
  pll_route12_kernel<<<blocks(S * K), tb, 0, stream>>>(
      topk_own, bound, tgt, ipr, cnt, phys_own, stats, S, K, G, R, L);
  if (K == 16) {
    pll_route3_kernel_t<16><<<blocks(S), tb, 0, stream>>>(
        topk_own, covmask, ipr, fallback, share_my, forced_left, phys_own,
        stats, S, G, R, L, NN, my_rank / L);
  } else if (K == 8) {
    pll_route3_kernel_t<8><<<blocks(S), tb, 0, stream>>>(
        topk_own, covmask, ipr, fallback, share_my, forced_left, phys_own,
        stats, S, G, R, L, NN, my_rank / L);
  } else {
    pll_route3_kernel<<<blocks(S), tb, 0, stream>>>(
        topk_own, covmask, ipr, fallback, share_my, forced_left, phys_own,
        stats, S, K, G, R, L, NN, my_rank / L);
  }
  CUDA_CHECK(cudaGetLastError());
}

size_t
placelambda_route_sl_workspace_bytes(int G, int R, int ranks_per_node) {
  int L = ranks_per_node;
  int NN = R / L;
  size_t b = 0;
  b += sizeof(int) * G * R;            // ipr
  b += sizeof(int) * G;                // covmask
  b += sizeof(int) * R * G;            // q1
  b += sizeof(int) * R;                // load
  b += sizeof(int) * NN * G * L;       // allocT
  b += sizeof(int) * R * G;            // granted2
  b += sizeof(int) * G * L * 2;        // bound + tgt
  b += sizeof(long long) * R * R;      // w3
  b += sizeof(int) * R;                // share_my
  b += sizeof(int) * G;                // fallback
  b += sizeof(int) * G;                // cnt
  b += sizeof(int) * R;                // forced_left (f_cap tickets)
  return b;
}

}  // namespace pll_ext

#define CHECK_INPUT(t, tp)                                              \
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() &&                       \
              t.scalar_type() == tp, #t " check failed")

std::tuple<torch::Tensor, torch::Tensor>
route_sl(const torch::Tensor topk_own, const torch::Tensor d,
         const torch::Tensor l2p, const torch::Tensor lcnts,
         int64_t my_rank, int64_t nlp, int64_t ranks_per_node,
         double eps, int64_t f_cap) {
  CHECK_INPUT(topk_own, at::ScalarType::Int);
  CHECK_INPUT(d, at::ScalarType::Int);
  CHECK_INPUT(l2p, at::ScalarType::Int);
  CHECK_INPUT(lcnts, at::ScalarType::Int);
  int S = topk_own.size(0), K = topk_own.size(1);
  int R = d.size(0), G = d.size(1);
  int Cmax = l2p.size(1);
  TORCH_CHECK(lcnts.size(0) == G && l2p.size(0) == G);
  TORCH_CHECK(G <= 4096 && R <= 1024, "route_sl size guard");
  long long cap64 = std::isinf(eps)
                        ? (long long)S * K * R
                        : (long long)std::ceil((1.0 + eps) * S * K);
  torch::Tensor phys = at::empty_like(topk_own);
  torch::Tensor stats =
      at::zeros({4}, topk_own.options().dtype(at::ScalarType::Long));
  torch::Tensor ws = at::empty(
      {(int64_t)pll_ext::placelambda_route_sl_workspace_bytes(
          G, R, ranks_per_node)},
      topk_own.options().dtype(at::ScalarType::Byte));
  auto stream = at::cuda::getCurrentCUDAStream();
  pll_ext::placelambda_route_sl(
      topk_own.data_ptr<int>(), d.data_ptr<int>(), l2p.data_ptr<int>(),
      lcnts.data_ptr<int>(), phys.data_ptr<int>(),
      (long long *)stats.data_ptr<int64_t>(), ws.data_ptr(),
      S, K, G, R, Cmax, (int)nlp, (int)ranks_per_node, (int)my_rank,
      cap64, (int)f_cap, stream);
  return {phys, stats};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("route_sl", &route_sl, "patched placelambda_route_sl");
}
