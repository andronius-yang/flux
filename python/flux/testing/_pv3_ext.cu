// PV3 rotation water-fill router — standalone JIT extension (branch pv3,
// 2026-09-14). Reference algorithm and constraint proof:
// python/flux/testing/pv3_route.py (pv3_tables / pv3_route) and
// docs/handoff/38_pv3_routing.md. Loader: pv3_ext.py.
//
// Two launches per iteration (+ one memset):
//   pv3_tables_kernel   one THREAD per expert: builds the expert's replica
//                       list from l2p/lcnts, plays the R-round rotation
//                       water-fill over all sources (pure integer function
//                       of the allgathered d[R, G] — identical on every
//                       rank), and emits THIS rank's segment table: the
//                       replicas in the order this rank visits them with
//                       cumulative row counts.
//   pv3_route_kernel    one thread per own entry: relaxed atomic ticket
//                       per expert -> segment lookup -> physical slot
//                       (the LocCap kernel's ticket contract: counts per
//                       (source, replica) are exact = the reference
//                       tables; which token takes which slot is free).
// No forced regime, no f_cap, stats identically zero.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <climits>
#include <tuple>

#define CUDA_CHECK(expr)                                        \
  do {                                                          \
    cudaError_t _e = (expr);                                    \
    TORCH_CHECK(_e == cudaSuccess, "CUDA error: ",              \
                cudaGetErrorString(_e));                        \
  } while (0)

namespace pv3_ext {

constexpr int kMaxRep = 32;      // replicas per expert (pv2: <= NN <= 32).
                                 // 2026-09-15: 64 -> 32 — the tables kernel's
                                 // per-thread local arrays (STACK 2048 B at 64)
                                 // are reserved for every resident thread on
                                 // the GPU at first launch (~450 MB); Qwen 16n
                                 // b16's fullest GPU could not provide it
                                 // (illegal address on local rank 3).
constexpr int kMaxNN = 32;       // nodes (node_jj table width)
constexpr int kThreads = 128;

__device__ __forceinline__ long long
ceil_div_ll(long long a, long long b) { return (a + b - 1) / b; }

// One thread per expert. left[] is the per-source remaining demand, kept in
// a global workspace row (R ints per expert; L1-resident after first
// touch). Every (source, replica) pair is visited exactly once, so the
// take is written once and never re-read.
// largest-remainder share of `total` for source `me` with integer weights
// w(i) (functor), deterministic (frac desc, index asc) — O(R)
template <typename W>
__device__ __forceinline__ int
lr_share(long long total, int R, int me, W w) {
  if (total <= 0) return 0;
  long long Wsum = 0;
  for (int i = 0; i < R; ++i) Wsum += w(i);
  if (Wsum <= 0) return 0;
  long long sum_base = 0, base_me = 0, frac_me = 0;
  for (int i = 0; i < R; ++i) {
    long long wi = w(i);
    long long b = total * wi / Wsum;
    sum_base += b;
    if (i == me) { base_me = b; frac_me = total * wi - b * Wsum; }
  }
  long long rem = total - sum_base;
  int rank = 0;
  for (int i = 0; i < R; ++i) {
    if (i == me) continue;
    long long wi = w(i);
    long long b = total * wi / Wsum;
    long long f = total * wi - b * Wsum;
    rank += (f > frac_me) || (f == frac_me && i < me);
  }
  return (int)(base_me + (rank < rem ? 1 : 0));
}

__global__ void
pv3_tables_kernel(
    const int *__restrict__ d,        // [R, G] allgathered demand
    const int *__restrict__ l2p,      // [G, Cmax]
    const int *__restrict__ lcnts,    // [G]
    int *__restrict__ left_ws,        // [G, R] workspace
    int *__restrict__ seg_phys,       // [G, kMaxRep] out (own visit order)
    int *__restrict__ seg_bound,      // [G, kMaxRep] out (cumulative)
    int *__restrict__ seg_cnt,        // [G] out (#segments)
    int G, int R, int Cmax, int nlp, int L, int NN, int my_rank,
    int C_num, int C_den,
    // pv3c (cover) outputs — null when cover == 0
    int cover, int *__restrict__ mtab,      // [G, kMaxRep, R] takes
    int *__restrict__ rep_phys,             // [G, kMaxRep] phys per replica
    int *__restrict__ rep_cnt,              // [G]
    int *__restrict__ rel_me,               // [G, kMaxRep] release budget
    int *__restrict__ ext_me,               // [G, kMaxRep] extra budget
    int *__restrict__ node_jj) {            // [G, kMaxNN] node -> replica
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= G) return;
  int rep[kMaxRep], phys_of[kMaxRep];
  int fill[kMaxRep], take_me[kMaxRep];
  int c = min(lcnts[g], kMaxRep);
  for (int jj = 0; jj < c; ++jj) {
    int ph = l2p[g * Cmax + jj];
    phys_of[jj] = ph;
    rep[jj] = ph / nlp;
    fill[jj] = 0;
    take_me[jj] = 0;
  }
  // ascending host rank order (l2p columns are ascending phys slot, hence
  // ascending rank already; insertion sort keeps the contract explicit)
  for (int a = 1; a < c; ++a) {
    int ra = rep[a], pa = phys_of[a];
    int b = a - 1;
    while (b >= 0 && rep[b] > ra) {
      rep[b + 1] = rep[b]; phys_of[b + 1] = phys_of[b]; --b;
    }
    rep[b + 1] = ra; phys_of[b + 1] = pa;
  }
  long long D = 0;
  int *left = left_ws + (size_t)g * R;
  for (int i = 0; i < R; ++i) {
    int v = d[i * G + g];
    left[i] = v;
    D += v;
  }
  if (cover) {
    rep_cnt[g] = c;
    for (int m = 0; m < NN; ++m) node_jj[g * kMaxNN + m] = -1;
    for (int jj = 0; jj < c; ++jj) {
      rep_phys[g * kMaxRep + jj] = phys_of[jj];
      rel_me[g * kMaxRep + jj] = 0;
      ext_me[g * kMaxRep + jj] = 0;
      int m = rep[jj] / L;                 // O(1) node -> replica lookup
      node_jj[g * kMaxNN + m] = (node_jj[g * kMaxNN + m] == -1) ? jj : -2;
    }
  }
  if (c == 0 || D == 0) {
    seg_cnt[g] = 0;
    if (cover) {                      // budgets read these: make them 0
      left[0] = 0;
      left[1] = 0;
    }
    return;
  }
  long long U = ceil_div_ll((long long)(C_den + C_num) * D,
                            (long long)C_den * c);
  long long Lb = ((long long)(C_den - C_num) * D) / ((long long)C_den * c);
  long long unassigned = D;
  // 2026-09-15 (handoff 39 §10 #1): the visit loop was 821 us at 32n K2
  // b1 on real placements (c up to 13) — a c^2 deficit rescan over
  // local-memory arrays plus five int div/mods per visit. Same arithmetic,
  // restructured: per-replica (node, local rank) precomputed, the round's
  // (du, dl) carried incrementally, and def_tot = sum_k max(0, Lb -
  // fill[k]) kept as a running sum (fill starts at 0 => c * Lb).
  unsigned char u_rep[kMaxRep], l_rep[kMaxRep];   // node < 32, local rank < L
  for (int jj = 0; jj < c; ++jj) {
    u_rep[jj] = (unsigned char)(rep[jj] / L);
    l_rep[jj] = (unsigned char)(rep[jj] % L);
  }
  long long def_tot = (Lb > 0) ? Lb * (long long)c : 0;
  int du = 0, dl = 0;
  for (int p = 0; p < R; ++p) {
    for (int jj = 0; jj < c; ++jj) {
      int u_i = (int)u_rep[jj] - du; if (u_i < 0) u_i += NN;
      int l_i = (int)l_rep[jj] - dl; if (l_i < 0) l_i += L;
      int i = u_i * L + l_i;
      long long want = left[i];
      long long take = 0;
      if (want > 0) {
        long long f = fill[jj];
        long long def_j = Lb - f; if (def_j < 0) def_j = 0;
        long long room = U - f;
        long long reserve = unassigned - (def_tot - def_j);
        take = want;
        if (room < take) take = room;
        if (reserve < take) take = reserve;
        if (take < 0) take = 0;
        if (take > 0) {
          long long nf = f + take;
          long long def_n = Lb - nf; if (def_n < 0) def_n = 0;
          def_tot += def_n - def_j;
          fill[jj] = (int)nf;
          left[i] -= (int)take;
          unassigned -= take;
          if (i == my_rank) take_me[jj] = (int)take;
        }
      }
      // every (source, replica) pair is visited exactly once: the write
      // below fully defines the table (pv3c needs all sources' takes)
      if (cover) mtab[((size_t)g * kMaxRep + jj) * R + i] = (int)take;
    }
    if (++dl == L) { dl = 0; ++du; }
  }
  if (cover) {
    // pv3c: publish the per-replica fill and the expert bounds; the
    // budget shares are computed by pv3c_budget_kernel (a block per
    // (expert, replica), threads over sources — the serial per-expert
    // largest-remainder loops were the 16n cost)
    for (int jj = 0; jj < c; ++jj) {
      rel_me[g * kMaxRep + jj] = fill[jj];       // fill, consumed below
      ext_me[g * kMaxRep + jj] = take_me[jj];    // my take, consumed below
    }
    left[0] = (int)Lb;                           // left[] is spent: reuse
    left[1] = (int)U;                            // (R >= 2 always)
  }
  // own visit order: sort replicas by the round in which my_rank visits
  // them (p = ((u_j - u_me) mod NN) * L + ((l_j - l_me) mod L))
  int u_me = my_rank / L, l_me = my_rank % L;
  // visit key of replica jj for my rank (computed on the fly: c <= 32)
  auto vkey = [&](int jj) {
    int du2 = (int)u_rep[jj] - u_me; if (du2 < 0) du2 += NN;
    int dl2 = (int)l_rep[jj] - l_me; if (dl2 < 0) dl2 += L;
    return du2 * L + dl2;
  };
  unsigned char ord[kMaxRep];
  for (int jj = 0; jj < c; ++jj) ord[jj] = (unsigned char)jj;
  for (int a = 1; a < c; ++a) {
    unsigned char oa = ord[a]; int ka = vkey(oa);
    int b = a - 1;
    while (b >= 0 && vkey(ord[b]) > ka) { ord[b + 1] = ord[b]; --b; }
    ord[b + 1] = oa;
  }
  int n = 0, acc = 0;
  for (int a = 0; a < c; ++a) {
    int jj = ord[a];
    if (take_me[jj] == 0) continue;
    acc += take_me[jj];
    seg_phys[g * kMaxRep + n] = phys_of[jj];
    seg_bound[g * kMaxRep + n] = acc;
    ++n;
  }
  seg_cnt[g] = n;
}

// One thread per own entry: relaxed ticket -> segment -> phys slot.
__global__ void
pv3_route_kernel(
    const int *__restrict__ topk_own,   // [S*K]
    const int *__restrict__ seg_phys, const int *__restrict__ seg_bound,
    const int *__restrict__ seg_cnt, int *__restrict__ cnt,
    int *__restrict__ phys_own, long long *__restrict__ stats,
    int n_entries) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_entries) return;
  int g = topk_own[e];
  int t = atomicAdd(&cnt[g], 1);
  int n = seg_cnt[g];
  int jj = 0;
  while (jj < n && seg_bound[g * kMaxRep + jj] <= t) ++jj;
  if (jj >= n) {
    // impossible by construction (tables cover exactly d[me, g] rows);
    // loud counter + fallback to the last segment keeps routing total
    atomicAdd((unsigned long long *)&stats[2], 1ull);
    jj = n > 0 ? n - 1 : 0;
  }
  phys_own[e] = seg_phys[g * kMaxRep + jj];
}

// pv3c budget shares (largest remainder, deterministic: frac desc, index
// asc) — one block per (expert, replica), threads stride over sources.
//   release[me] = my share of (fill - Lb), weights = the sources' takes
//   extra[me]   = my share of (U - fill),  weights = the sources' demand
// Inputs come from the tables kernel: rel_me holds fill, ext_me holds my
// take, seg_bound[.., kMaxRep-1] holds U, left_ws[g*R] holds Lb.
constexpr int kBudgetThreads = 256;

__device__ __forceinline__ long long
block_sum_ll(long long v, long long *sh) {
  __syncthreads();
  sh[threadIdx.x] = v;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
    __syncthreads();
  }
  long long r = sh[0];
  __syncthreads();
  return r;
}

template <typename W>
__device__ __forceinline__ int
lr_share_block(long long total, int R, int me, W w, long long *sh) {
  if (total <= 0) return 0;
  long long ws = 0;
  for (int i = threadIdx.x; i < R; i += blockDim.x) ws += w(i);
  long long Wsum = block_sum_ll(ws, sh);
  if (Wsum <= 0) return 0;
  long long sb = 0, my_base = 0, my_frac = -1;
  for (int i = threadIdx.x; i < R; i += blockDim.x) {
    long long wi = w(i);
    long long b = total * wi / Wsum;
    sb += b;
    if (i == me) { my_base = b; my_frac = total * wi - b * Wsum; }
  }
  long long sum_base = block_sum_ll(sb, sh);
  // broadcast me's base/frac (exactly one thread owns i == me)
  __shared__ long long s_me[2];
  if (my_frac >= 0) { s_me[0] = my_base; s_me[1] = my_frac; }
  __syncthreads();
  long long base_me = s_me[0], frac_me = s_me[1];
  long long rank = 0;
  for (int i = threadIdx.x; i < R; i += blockDim.x) {
    if (i == me) continue;
    long long wi = w(i);
    long long b = total * wi / Wsum;
    long long f = total * wi - b * Wsum;
    rank += (f > frac_me) || (f == frac_me && i < me);
  }
  long long rank_me = block_sum_ll(rank, sh);
  long long rem = total - sum_base;
  return (int)(base_me + (rank_me < rem ? 1 : 0));
}

__global__ void
pv3c_budget_kernel(
    const int *__restrict__ d, const int *__restrict__ mtab,
    const int *__restrict__ rep_cnt, const int *__restrict__ seg_bound,
    const int *__restrict__ left_ws, int *__restrict__ rel_me,
    int *__restrict__ ext_me, int G, int R, int my_rank) {
  __shared__ long long sh[kBudgetThreads];
  int g = blockIdx.x / kMaxRep, jj = blockIdx.x % kMaxRep;
  if (g >= G || jj >= rep_cnt[g]) return;
  long long fill = rel_me[g * kMaxRep + jj];
  int take_me = ext_me[g * kMaxRep + jj];
  long long U = left_ws[(size_t)g * R + 1];
  long long Lb = left_ws[(size_t)g * R];
  const int *row = mtab + ((size_t)g * kMaxRep + jj) * R;
  int rel = lr_share_block(fill - Lb, R, my_rank,
                           [row](int i) { return (long long)row[i]; }, sh);
  if (rel > take_me) rel = take_me;
  int ext = lr_share_block(U - fill, R, my_rank,
                           [d, G, g](int i) { return (long long)d[i * G + g]; },
                           sh);
  __syncthreads();
  if (threadIdx.x == 0) {
    rel_me[g * kMaxRep + jj] = rel;
    ext_me[g * kMaxRep + jj] = ext;
  }
}

// pv3c VACATE pass: one thread per own token. A remote node is vacated
// when every entry of the token served there can move — release ticket
// at its current replica, extra ticket at a replica on a node the token
// already touches (home first) — with relaxed atomic tickets; freed
// tickets return to the pool. Monotone: distinct remote nodes per token
// never increase; both bounds hold (every move is paid from slack).
template <int KT>
__global__ void
pv3c_vacate_kernel(
    const int *__restrict__ topk_own, const int *__restrict__ rep_phys,
    const int *__restrict__ rep_cnt, int *__restrict__ rel,
    int *__restrict__ ext, int *__restrict__ phys_own,
    long long *__restrict__ stats, int S, int nlp, int L, int NN,
    int home, const int *__restrict__ node_jj) {
  // 2026-09-15 cooperative edition (handoff 39 §10 #1): one KT-lane group
  // per token, a lane per (token, expert) entry. The serial per-token loop
  // (368 us at 32n b1: 72 tokens = 3 warps of dependent global chains) is
  // replaced by group-wide node selection (match/ballot/shfl) and a
  // parallel all-or-nothing acquire: every lane on the chosen node takes
  // its release + extra tickets at once; a group vote commits or rolls all
  // of them back. Same policy as the serial kernel — least-touched remote
  // node first (ties by id), targets home first then touched nodes in
  // replica order, bounds paid from the budgets on both sides — so the
  // vacate stays monotone and constraint-clean; only the transient
  // ordering of ticket probes differs (relaxed contract, as before).
  constexpr int TPW = 32 / KT;
  static_assert(KT == 8 || KT == 16, "KT");
  int lane = threadIdx.x & 31;
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int grp = lane / KT, k = lane % KT;
  int s = warp * TPW + grp;
  bool active = s < S;
  unsigned gmask = ((KT == 32) ? 0xffffffffu : ((1u << KT) - 1u)) << (grp * KT);
  int ph = active ? phys_own[s * KT + k] : 0;
  int nd = active ? (ph / nlp) / L : home;
  int g = active ? topk_own[s * KT + k] : 0;
  int moved_total = 0;
  if (active) {
    for (int it = 0; it < NN; ++it) {
      bool remote = nd != home;
      unsigned rmask = __ballot_sync(gmask, remote);
      if (rmask == 0) break;
      unsigned tb = remote ? (1u << nd) : 0u;
#pragma unroll
      for (int o = KT / 2; o > 0; o >>= 1) tb |= __shfl_xor_sync(gmask, tb, o);
      const unsigned touched = tb;               // group-uniform
      unsigned tried = 0;
      bool moved = false;
      while (!moved) {
        // candidate = least-touched untried remote node (ties by id)
        int key = 0x7fffffff;
        unsigned same = __match_any_sync(gmask, nd);   // all group lanes
        if (remote && !((tried >> nd) & 1u))
          key = __popc(same & rmask) * 64 + nd;
#pragma unroll
        for (int o = KT / 2; o > 0; o >>= 1)
          key = min(key, __shfl_xor_sync(gmask, key, o));
        if (key == 0x7fffffff) break;
        int n = key & 63;
        tried |= 1u << n;
        // parallel acquire (no warp intrinsics inside): lanes on node n
        // take a release ticket at their current replica and an extra
        // ticket at a target replica (home first, then touched nodes)
        bool mine = remote && nd == n;
        bool ok = true;
        int jo = -1, jt = -1;
        if (mine) {
          int c = rep_cnt[g];
          int cur = node_jj[g * kMaxNN + n];
          if (cur == -2) {
            cur = -1;
            for (int jj = 0; jj < c; ++jj)
              if (rep_phys[g * kMaxRep + jj] == ph) { cur = jj; break; }
          }
          if (cur < 0 || rel[g * kMaxRep + cur] <= 0) {
            ok = false;
          } else {
            int old = atomicSub(&rel[g * kMaxRep + cur], 1);
            if (old <= 0) { atomicAdd(&rel[g * kMaxRep + cur], 1); ok = false; }
            else jo = cur;
          }
          if (ok) {
            for (int pass = 0; pass < 2 && jt < 0; ++pass) {
              for (int jj = 0; jj < c && jt < 0; ++jj) {
                int m = (rep_phys[g * kMaxRep + jj] / nlp) / L;
                bool want = (pass == 0) ? (m == home)
                                        : (m != n && m != home && ((touched >> m) & 1u));
                if (!want || ext[g * kMaxRep + jj] <= 0) continue;
                int o2 = atomicSub(&ext[g * kMaxRep + jj], 1);
                if (o2 > 0) jt = jj;
                else atomicAdd(&ext[g * kMaxRep + jj], 1);
              }
            }
            if (jt < 0) { atomicAdd(&rel[g * kMaxRep + jo], 1); jo = -1; ok = false; }
          }
        }
        bool all_ok = __all_sync(gmask, ok);
        if (all_ok) {
          if (mine) {
            // commit: the freed room at the source, the moved row's release
            atomicAdd(&ext[g * kMaxRep + jo], 1);
            atomicAdd(&rel[g * kMaxRep + jt], 1);
            ph = rep_phys[g * kMaxRep + jt];
            nd = (ph / nlp) / L;
            ++moved_total;
          }
          moved = true;
        } else if (mine && jt >= 0) {
          // rollback this lane's fully acquired pair
          atomicAdd(&rel[g * kMaxRep + jo], 1);
          atomicAdd(&ext[g * kMaxRep + jt], 1);
        }
      }
      if (!moved) break;
    }
  }
  if (active) phys_own[s * KT + k] = ph;
  int tot = moved_total;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
  if (lane == 0 && tot)
    atomicAdd((unsigned long long *)&stats[1], (unsigned long long)tot);
}

}  // namespace pv3_ext

#define CHECK_INPUT(t, tp)                                              \
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() &&                       \
              t.scalar_type() == tp, #t " check failed")

// Persistent workspace: left [G, R] + seg tables [G, kMaxRep] x2 + seg_cnt
// [G] + cnt [G] ints.
int64_t workspace_ints(int64_t G, int64_t R) {
  return G * R + 2 * G * pv3_ext::kMaxRep + 2 * G;
}

// pv3c workspace: pv3's + mtab [G, kMaxRep, R] + rep_phys/rel/ext
// [G, kMaxRep] + rep_cnt [G]
int64_t workspace_ints_c(int64_t G, int64_t R) {
  return workspace_ints(G, R) + G * pv3_ext::kMaxRep * R
         + 3 * G * pv3_ext::kMaxRep + G + G * pv3_ext::kMaxNN;
}

std::tuple<torch::Tensor, torch::Tensor>
route_pv3(const torch::Tensor topk_own, const torch::Tensor d,
          const torch::Tensor l2p, const torch::Tensor lcnts,
          int64_t my_rank, int64_t nlp, int64_t ranks_per_node,
          int64_t C_num, int64_t C_den, torch::Tensor ws) {
  CHECK_INPUT(topk_own, at::ScalarType::Int);
  CHECK_INPUT(d, at::ScalarType::Int);
  CHECK_INPUT(l2p, at::ScalarType::Int);
  CHECK_INPUT(lcnts, at::ScalarType::Int);
  CHECK_INPUT(ws, at::ScalarType::Int);
  int S = topk_own.size(0), K = topk_own.size(1);
  int R = d.size(0), G = d.size(1);
  int Cmax = l2p.size(1);
  int L = (int)ranks_per_node, NN = R / L;
  TORCH_CHECK(lcnts.size(0) == G && l2p.size(0) == G);
  TORCH_CHECK(R % L == 0 && NN >= 1);
  // Cmax is the l2p column STRIDE (plan_tensors_from_hosts emits [G, R]);
  // the replica count that matters is lcnts[g] <= kMaxRep (pv2: <= NN),
  // clamped inside the table kernel (a rank-count guard here would need a
  // device sync).
  TORCH_CHECK(ws.numel() >= workspace_ints(G, R), "pv3 workspace too small");
  TORCH_CHECK(C_den > 0 && C_num >= 0 && C_num <= C_den);
  torch::Tensor phys = at::empty_like(topk_own);
  torch::Tensor stats =
      at::zeros({4}, topk_own.options().dtype(at::ScalarType::Long));
  int *w = ws.data_ptr<int>();
  int *left_ws = w;                 w += (size_t)G * R;
  int *seg_phys = w;                w += (size_t)G * pv3_ext::kMaxRep;
  int *seg_bound = w;               w += (size_t)G * pv3_ext::kMaxRep;
  int *seg_cnt = w;                 w += G;
  int *cnt = w;
  auto stream = at::cuda::getCurrentCUDAStream();
  CUDA_CHECK(cudaMemsetAsync(cnt, 0, sizeof(int) * G, stream));
  int tb = pv3_ext::kThreads;
  pv3_ext::pv3_tables_kernel<<<(G + tb - 1) / tb, tb, 0, stream>>>(
      d.data_ptr<int>(), l2p.data_ptr<int>(), lcnts.data_ptr<int>(),
      left_ws, seg_phys, seg_bound, seg_cnt, G, R, Cmax, (int)nlp, L, NN,
      (int)my_rank, (int)C_num, (int)C_den,
      0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
  int n = S * K;
  pv3_ext::pv3_route_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      topk_own.data_ptr<int>(), seg_phys, seg_bound, seg_cnt, cnt,
      phys.data_ptr<int>(), (long long *)stats.data_ptr<int64_t>(), n);
  CUDA_CHECK(cudaGetLastError());
  return {phys, stats};
}

// pv3c: pv3 + the per-token vacate pass on the release/extra budgets.
std::tuple<torch::Tensor, torch::Tensor>
route_pv3c(const torch::Tensor topk_own, const torch::Tensor d,
           const torch::Tensor l2p, const torch::Tensor lcnts,
           int64_t my_rank, int64_t nlp, int64_t ranks_per_node,
           int64_t C_num, int64_t C_den, torch::Tensor ws) {
  CHECK_INPUT(topk_own, at::ScalarType::Int);
  CHECK_INPUT(d, at::ScalarType::Int);
  CHECK_INPUT(l2p, at::ScalarType::Int);
  CHECK_INPUT(lcnts, at::ScalarType::Int);
  CHECK_INPUT(ws, at::ScalarType::Int);
  int S = topk_own.size(0), K = topk_own.size(1);
  int R = d.size(0), G = d.size(1);
  int Cmax = l2p.size(1);
  int L = (int)ranks_per_node, NN = R / L;
  TORCH_CHECK(lcnts.size(0) == G && l2p.size(0) == G);
  TORCH_CHECK(R % L == 0 && NN >= 1 && NN <= 32, "pv3c: NN <= 32");
  TORCH_CHECK(K == 8 || K == 16, "pv3c vacate kernel: K in {8, 16}");
  TORCH_CHECK(ws.numel() >= workspace_ints_c(G, R), "pv3c workspace");
  torch::Tensor phys = at::empty_like(topk_own);
  torch::Tensor stats =
      at::zeros({4}, topk_own.options().dtype(at::ScalarType::Long));
  int *w = ws.data_ptr<int>();
  int *left_ws = w;                 w += (size_t)G * R;
  int *seg_phys = w;                w += (size_t)G * pv3_ext::kMaxRep;
  int *seg_bound = w;               w += (size_t)G * pv3_ext::kMaxRep;
  int *seg_cnt = w;                 w += G;
  int *cnt = w;                     w += G;
  int *mtab = w;                    w += (size_t)G * pv3_ext::kMaxRep * R;
  int *rep_phys = w;                w += (size_t)G * pv3_ext::kMaxRep;
  int *rel = w;                     w += (size_t)G * pv3_ext::kMaxRep;
  int *ext = w;                     w += (size_t)G * pv3_ext::kMaxRep;
  int *rep_cnt = w;                 w += G;
  int *node_jj = w;
  auto stream = at::cuda::getCurrentCUDAStream();
  CUDA_CHECK(cudaMemsetAsync(cnt, 0, sizeof(int) * G, stream));
  int tb = pv3_ext::kThreads;
  pv3_ext::pv3_tables_kernel<<<(G + tb - 1) / tb, tb, 0, stream>>>(
      d.data_ptr<int>(), l2p.data_ptr<int>(), lcnts.data_ptr<int>(),
      left_ws, seg_phys, seg_bound, seg_cnt, G, R, Cmax, (int)nlp, L, NN,
      (int)my_rank, (int)C_num, (int)C_den,
      1, mtab, rep_phys, rep_cnt, rel, ext, node_jj);
  pv3_ext::pv3c_budget_kernel<<<G * pv3_ext::kMaxRep, pv3_ext::kBudgetThreads,
                                0, stream>>>(
      d.data_ptr<int>(), mtab, rep_cnt, seg_bound, left_ws, rel, ext, G, R,
      (int)my_rank);
  int n = S * K;
  pv3_ext::pv3_route_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      topk_own.data_ptr<int>(), seg_phys, seg_bound, seg_cnt, cnt,
      phys.data_ptr<int>(), (long long *)stats.data_ptr<int64_t>(), n);
  int home = (int)my_rank / L;
  int vth = S * K;                       // one lane per (token, entry)
  if (K == 16) {
    pv3_ext::pv3c_vacate_kernel<16><<<(vth + 127) / 128, 128, 0, stream>>>(
        topk_own.data_ptr<int>(), rep_phys, rep_cnt, rel, ext,
        phys.data_ptr<int>(), (long long *)stats.data_ptr<int64_t>(), S,
        (int)nlp, L, NN, home, node_jj);
  } else {
    pv3_ext::pv3c_vacate_kernel<8><<<(vth + 127) / 128, 128, 0, stream>>>(
        topk_own.data_ptr<int>(), rep_phys, rep_cnt, rel, ext,
        phys.data_ptr<int>(), (long long *)stats.data_ptr<int64_t>(), S,
        (int)nlp, L, NN, home, node_jj);
  }
  CUDA_CHECK(cudaGetLastError());
  return {phys, stats};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("route_pv3", &route_pv3, "pv3 rotation water-fill route (own rank)");
  m.def("workspace_ints", &workspace_ints, "pv3 workspace size in ints");
  m.def("route_pv3c", &route_pv3c, "pv3c: pv3 + per-token vacate pass");
  m.def("workspace_ints_c", &workspace_ints_c, "pv3c workspace size");
}
