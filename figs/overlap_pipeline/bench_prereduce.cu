// Standalone microbenchmark of the combine-side gateway pre-reduce (Σ) stage.
//
// Question (2026-09-10): is the canon a2av_combine_prereduce_kernel slow because
// it is CTA-starved (6 CTAs) or because it is written latency-bound?  And does a
// dense padded-slot layout (the "union window" analog of the dispatch fan-out)
// make the reduce cheap at 6-12 CTAs?
//
// Kernels (bf16, n = 7168, one row = 14336 B):
//   k0_csr_ref     : the shipped structure — one thread per 16 B pack, serial
//                    CSR walk (wire_ptr / wire_copy), fp32 accumulate.
//   k1_csr_ilp     : same CSR, but each thread owns P=4 consecutive packs and
//                    prefetches up to KMAX contribution indices first, so all
//                    P*cnt 16 B loads are independent and in flight together.
//   k2_dense_zero  : dense layout [slot][wire_row][n], S slots, zero-filled;
//                    unconditional S-way sum (reads zeros).
//   k3_dense_mask  : same dense layout + per-row presence bitmask; loads only
//                    present slots (no zero traffic), P packs per thread.
// Each kernel is timed at grid = 6, 12, 24, 48 CTAs x 512 threads over one
// "wave" (wire_rows rows, mean contributions per row = ~2).
//
// Build:  nvcc -O3 -arch=sm_80 -std=c++17 bench_prereduce.cu -o bench_prereduce
// Run:    ./bench_prereduce [wire_rows=6000] [mean_contrib=2.0] [slots=4]
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

constexpr int N = 7168;
constexpr int PACK = 8;                 // bf16 per 16 B
constexpr int PPR = N / PACK;           // packs per row = 896
constexpr int THREADS = 512;
constexpr int KMAX = 8;                 // topk bound on contributions per wire row

__device__ __forceinline__ uint4 ldg16(const void *p) {
  return __ldg(reinterpret_cast<const uint4 *>(p));
}
__device__ __forceinline__ void st16(void *p, uint4 v) {
  *reinterpret_cast<uint4 *>(p) = v;
}
__device__ __forceinline__ void acc_pack(float *acc, uint4 v) {
  const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&v);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 f = __bfloat1622float2(h[i]);
    acc[2 * i] += f.x;
    acc[2 * i + 1] += f.y;
  }
}
__device__ __forceinline__ uint4 pack_out(const float *acc) {
  uint4 v;
  __nv_bfloat162 *h = reinterpret_cast<__nv_bfloat162 *>(&v);
#pragma unroll
  for (int i = 0; i < 4; i++) h[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
  return v;
}

// ---------------------------------------------------------------- k0: shipped structure
__global__ void __launch_bounds__(THREADS, 1)
k0_csr_ref(const __nv_bfloat16 *conv, __nv_bfloat16 *wire, const int *wire_ptr,
           const int *wire_copy, int wire_rows) {
  const long long total = (long long)wire_rows * PPR;
  for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x; idx < total;
       idx += (long long)gridDim.x * blockDim.x) {
    const int w = (int)(idx / PPR);
    const int col = (int)(idx % PPR) * PACK;
    float acc[PACK] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = wire_ptr[w]; k < wire_ptr[w + 1]; k++) {
      uint4 v = ldg16(conv + (long long)wire_copy[k] * N + col);
      acc_pack(acc, v);
    }
    st16(wire + (long long)w * N + col, pack_out(acc));
  }
}

// ---------------------------------------------------------------- k1: CSR with ILP
template <int P>
__global__ void __launch_bounds__(THREADS, 1)
k1_csr_ilp(const __nv_bfloat16 *conv, __nv_bfloat16 *wire, const int *wire_ptr,
           const int *wire_copy, int wire_rows) {
  constexpr int GROUPS = PPR / P;  // thread-groups of P packs per row
  const long long total = (long long)wire_rows * GROUPS;
  for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x; idx < total;
       idx += (long long)gridDim.x * blockDim.x) {
    const int w = (int)(idx / GROUPS);
    const int col = (int)(idx % GROUPS) * (P * PACK);
    const int k0 = wire_ptr[w], k1 = wire_ptr[w + 1];
    const int cnt = min(k1 - k0, KMAX);
    int rows[KMAX];
#pragma unroll
    for (int k = 0; k < KMAX; k++) rows[k] = (k < cnt) ? wire_copy[k0 + k] : 0;
    float acc[P][PACK];
#pragma unroll
    for (int p = 0; p < P; p++)
#pragma unroll
      for (int i = 0; i < PACK; i++) acc[p][i] = 0.f;
    // issue every load before accumulating: P*cnt independent 16 B loads
    uint4 v[KMAX][P];
#pragma unroll
    for (int k = 0; k < KMAX; k++) {
      if (k < cnt) {
        const __nv_bfloat16 *src = conv + (long long)rows[k] * N + col;
#pragma unroll
        for (int p = 0; p < P; p++) v[k][p] = ldg16(src + p * PACK);
      }
    }
#pragma unroll
    for (int k = 0; k < KMAX; k++) {
      if (k < cnt) {
#pragma unroll
        for (int p = 0; p < P; p++) acc_pack(acc[p], v[k][p]);
      }
    }
    __nv_bfloat16 *dst = wire + (long long)w * N + col;
#pragma unroll
    for (int p = 0; p < P; p++) st16(dst + p * PACK, pack_out(acc[p]));
  }
}

// ---------------------------------------------------------------- k2: dense zero-filled slots
template <int P>
__global__ void __launch_bounds__(THREADS, 1)
k2_dense_zero(const __nv_bfloat16 *dense, __nv_bfloat16 *wire, int wire_rows, int S) {
  constexpr int GROUPS = PPR / P;
  const long long total = (long long)wire_rows * GROUPS;
  const long long slot_stride = (long long)wire_rows * N;
  for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x; idx < total;
       idx += (long long)gridDim.x * blockDim.x) {
    const int w = (int)(idx / GROUPS);
    const int col = (int)(idx % GROUPS) * (P * PACK);
    float acc[P][PACK];
#pragma unroll
    for (int p = 0; p < P; p++)
#pragma unroll
      for (int i = 0; i < PACK; i++) acc[p][i] = 0.f;
    uint4 v[KMAX][P];
#pragma unroll
    for (int s = 0; s < KMAX; s++) {
      if (s < S) {
        const __nv_bfloat16 *src = dense + s * slot_stride + (long long)w * N + col;
#pragma unroll
        for (int p = 0; p < P; p++) v[s][p] = ldg16(src + p * PACK);
      }
    }
#pragma unroll
    for (int s = 0; s < KMAX; s++)
      if (s < S)
#pragma unroll
        for (int p = 0; p < P; p++) acc_pack(acc[p], v[s][p]);
    __nv_bfloat16 *dst = wire + (long long)w * N + col;
#pragma unroll
    for (int p = 0; p < P; p++) st16(dst + p * PACK, pack_out(acc[p]));
  }
}

// ---------------------------------------------------------------- k3: dense + presence mask
template <int P>
__global__ void __launch_bounds__(THREADS, 1)
k3_dense_mask(const __nv_bfloat16 *dense, __nv_bfloat16 *wire, const unsigned char *mask,
              int wire_rows, int S) {
  constexpr int GROUPS = PPR / P;
  const long long total = (long long)wire_rows * GROUPS;
  const long long slot_stride = (long long)wire_rows * N;
  for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x; idx < total;
       idx += (long long)gridDim.x * blockDim.x) {
    const int w = (int)(idx / GROUPS);
    const int col = (int)(idx % GROUPS) * (P * PACK);
    const unsigned m = mask[w];
    float acc[P][PACK];
#pragma unroll
    for (int p = 0; p < P; p++)
#pragma unroll
      for (int i = 0; i < PACK; i++) acc[p][i] = 0.f;
    uint4 v[KMAX][P];
#pragma unroll
    for (int s = 0; s < KMAX; s++) {
      if (s < S && (m >> s) & 1u) {
        const __nv_bfloat16 *src = dense + s * slot_stride + (long long)w * N + col;
#pragma unroll
        for (int p = 0; p < P; p++) v[s][p] = ldg16(src + p * PACK);
      }
    }
#pragma unroll
    for (int s = 0; s < KMAX; s++)
      if (s < S && (m >> s) & 1u)
#pragma unroll
        for (int p = 0; p < P; p++) acc_pack(acc[p], v[s][p]);
    __nv_bfloat16 *dst = wire + (long long)w * N + col;
#pragma unroll
    for (int p = 0; p < P; p++) st16(dst + p * PACK, pack_out(acc[p]));
  }
}


// ---------------------------------------------------------------- k4: cp.async streaming reducer
// Each CTA owns a contiguous range of wire rows and streams their conv rows through
// a STAGES-deep shared-memory ring with cp.async (16 B per thread per instruction,
// two instructions per 14336 B row), so STAGES rows (~14 KB each) are in flight per
// SM regardless of how many contributions a wire row has.  Accumulation happens
// from shared memory into registers; one wire row = 896 packs = 512 + 384 threads.
#include <cuda_pipeline.h>
constexpr int STAGES = 6;
constexpr int ROW_BYTES = N * 2;
__global__ void __launch_bounds__(THREADS, 1)
k4_csr_stream(const __nv_bfloat16 *conv, __nv_bfloat16 *wire, const int *wire_ptr,
              const int *wire_copy, int wire_rows) {
  extern __shared__ __align__(16) unsigned char smem[];
  // CTA's wire-row range
  const int per = (wire_rows + gridDim.x - 1) / gridDim.x;
  const int w0 = blockIdx.x * per, w1 = min(wire_rows, w0 + per);
  if (w0 >= w1) return;
  const int k0 = wire_ptr[w0], k1 = wire_ptr[w1];   // conv-row stream for this CTA
  const int nrows = k1 - k0;
  const int t = threadIdx.x;
  const bool two = (t + THREADS) < PPR;              // this thread owns packs t and t+512
  auto issue = [&](int j) {                          // stage j % STAGES <- conv row wire_copy[k0 + j]
    if (j < nrows) {
      const int r = wire_copy[k0 + j];
      unsigned char *dst = smem + (size_t)(j % STAGES) * ROW_BYTES;
      const unsigned char *src = reinterpret_cast<const unsigned char *>(conv + (long long)r * N);
      __pipeline_memcpy_async(dst + t * 16, src + t * 16, 16);
      if (two) __pipeline_memcpy_async(dst + (t + THREADS) * 16, src + (t + THREADS) * 16, 16);
    }
    __pipeline_commit();
  };
  for (int j = 0; j < STAGES - 1; j++) issue(j);
  float acc0[PACK], acc1[PACK];
  int w = w0, kend = wire_ptr[w + 1];
#pragma unroll
  for (int i = 0; i < PACK; i++) acc0[i] = acc1[i] = 0.f;
  for (int j = 0; j < nrows; j++) {
    issue(j + STAGES - 1);
    __pipeline_wait_prior(STAGES - 1);
    __syncthreads();
    const unsigned char *st = smem + (size_t)(j % STAGES) * ROW_BYTES;
    acc_pack(acc0, *reinterpret_cast<const uint4 *>(st + t * 16));
    if (two) acc_pack(acc1, *reinterpret_cast<const uint4 *>(st + (t + THREADS) * 16));
    if (k0 + j + 1 == kend) {                        // wire row w complete
      __nv_bfloat16 *dst = wire + (long long)w * N;
      st16(dst + t * PACK, pack_out(acc0));
      if (two) st16(dst + (t + THREADS) * PACK, pack_out(acc1));
#pragma unroll
      for (int i = 0; i < PACK; i++) acc0[i] = acc1[i] = 0.f;
      w++;
      kend = wire_ptr[w + 1];
    }
    __syncthreads();                                  // stage j may be refilled next iteration
  }
}

// ---------------------------------------------------------------- host

int main(int argc, char **argv) {
  const int wire_rows = argc > 1 ? atoi(argv[1]) : 6000;
  const double mean_c = argc > 2 ? atof(argv[2]) : 2.0;
  const int S = argc > 3 ? atoi(argv[3]) : 4;
  printf("wire_rows=%d mean_contrib=%.2f slots=%d n=%d row=%d B\n", wire_rows, mean_c, S, N, N * 2);

  // ---- synthetic plan: contributions per wire row, 1..min(S,KMAX), mean ~ mean_c;
  //      conv rows laid out in (slot, token) order (peers' slices), wire rows token-ascending
  std::mt19937 rng(42);
  std::vector<int> cnt(wire_rows);
  {
    // geometric-ish: P(c) ∝ (mean-1)^(c-1) clipped to [1, S]
    std::vector<double> pw(S);
    double q = std::max(0.05, std::min(0.95, (mean_c - 1.0) / mean_c));
    for (int c = 1; c <= S; c++) pw[c - 1] = std::pow(q, c - 1);
    std::discrete_distribution<int> dd(pw.begin(), pw.end());
    for (int w = 0; w < wire_rows; w++) cnt[w] = 1 + dd(rng);
  }
  // choose which slots each wire row uses
  std::vector<unsigned char> mask(wire_rows);
  std::vector<std::vector<int>> slots_of(wire_rows);
  for (int w = 0; w < wire_rows; w++) {
    std::vector<int> s(S);
    std::iota(s.begin(), s.end(), 0);
    std::shuffle(s.begin(), s.end(), rng);
    s.resize(cnt[w]);
    std::sort(s.begin(), s.end());
    slots_of[w] = s;
    unsigned m = 0;
    for (int x : s) m |= 1u << x;
    mask[w] = (unsigned char)m;
  }
  // conv panel (CSR layout): segment per slot, rows token-ascending inside the segment
  std::vector<int> seg_off(S + 1, 0);
  for (int w = 0; w < wire_rows; w++)
    for (int x : slots_of[w]) seg_off[x + 1]++;
  for (int s = 0; s < S; s++) seg_off[s + 1] += seg_off[s];
  const int conv_rows = seg_off[S];
  std::vector<int> wire_ptr(wire_rows + 1, 0), wire_copy(conv_rows);
  {
    std::vector<int> cur(seg_off.begin(), seg_off.end() - 1);
    int k = 0;
    for (int w = 0; w < wire_rows; w++) {
      wire_ptr[w] = k;
      for (int x : slots_of[w]) wire_copy[k++] = cur[x]++;
    }
    wire_ptr[wire_rows] = k;
  }
  const double actual_c = (double)conv_rows / wire_rows;
  printf("conv_rows=%d (actual mean contrib %.2f); CSR bytes in+out = %.1f MB; dense-zero bytes = %.1f MB\n",
         conv_rows, actual_c, (conv_rows + wire_rows) * (double)N * 2 / 1e6,
         ((double)S + 1) * wire_rows * N * 2 / 1e6);

  // ---- device buffers
  const size_t row_b = (size_t)N * sizeof(__nv_bfloat16);
  __nv_bfloat16 *d_conv, *d_dense, *d_wire, *d_ref;
  int *d_ptr, *d_copy;
  unsigned char *d_mask;
  CK(cudaMalloc(&d_conv, (size_t)conv_rows * row_b));
  CK(cudaMalloc(&d_dense, (size_t)S * wire_rows * row_b));
  CK(cudaMalloc(&d_wire, (size_t)wire_rows * row_b));
  CK(cudaMalloc(&d_ref, (size_t)wire_rows * row_b));
  CK(cudaMalloc(&d_ptr, (wire_rows + 1) * sizeof(int)));
  CK(cudaMalloc(&d_copy, conv_rows * sizeof(int)));
  CK(cudaMalloc(&d_mask, wire_rows));
  // fill conv rows with small deterministic values; dense = same rows placed in slots, zeros elsewhere
  {
    std::vector<__nv_bfloat16> h_conv((size_t)conv_rows * N), h_dense((size_t)S * wire_rows * N, __float2bfloat16(0.f));
    for (int w = 0; w < wire_rows; w++) {
      for (int k = wire_ptr[w]; k < wire_ptr[w + 1]; k++) {
        const int r = wire_copy[k];
        const int slot = slots_of[w][k - wire_ptr[w]];
        for (int j = 0; j < N; j++) {
          float val = 0.001f * ((r * 7 + j) % 97) - 0.05f;
          h_conv[(size_t)r * N + j] = __float2bfloat16(val);
          h_dense[((size_t)slot * wire_rows + w) * N + j] = __float2bfloat16(val);
        }
      }
    }
    CK(cudaMemcpy(d_conv, h_conv.data(), h_conv.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dense, h_dense.data(), h_dense.size() * 2, cudaMemcpyHostToDevice));
  }
  CK(cudaMemcpy(d_ptr, wire_ptr.data(), (wire_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_copy, wire_copy.data(), conv_rows * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_mask, mask.data(), wire_rows, cudaMemcpyHostToDevice));

  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  const int grids[] = {6, 12, 24, 48};
  const double csr_mb = (conv_rows + wire_rows) * (double)N * 2 / 1e6;
  const double dense_mb = ((double)S + 1) * wire_rows * N * 2 / 1e6;
  std::vector<__nv_bfloat16> h_ref((size_t)wire_rows * N), h_out((size_t)wire_rows * N);

  auto check = [&](const char *name) {
    CK(cudaMemcpy(h_out.data(), d_wire, h_out.size() * 2, cudaMemcpyDeviceToHost));
    double maxd = 0;
    for (size_t i = 0; i < h_out.size(); i++)
      maxd = std::max(maxd, (double)fabsf(__bfloat162float(h_out[i]) - __bfloat162float(h_ref[i])));
    if (maxd > 1e-2) printf("  !! %s max |diff| vs ref = %.4g\n", name, maxd);
  };
  auto timeit = [&](const char *name, double mb, auto launch) {
    for (int g : grids) {
      launch(g);  // warm
      CK(cudaDeviceSynchronize());
      const int reps = 10;
      CK(cudaEventRecord(e0));
      for (int r = 0; r < reps; r++) launch(g);
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
      float ms;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      ms /= reps;
      printf("  %-14s grid=%2d  %7.3f ms  %6.1f GB/s (%.0f MB moved)\n", name, g, ms, mb / ms, mb);
      check(name);
    }
  };

  // reference from k0 at 48 CTAs
  k0_csr_ref<<<48, THREADS>>>(d_conv, d_wire, d_ptr, d_copy, wire_rows);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h_ref.data(), d_wire, h_ref.size() * 2, cudaMemcpyDeviceToHost));

  printf("--- CSR layout (shipped conv panel)\n");
  timeit("k0_csr_ref", csr_mb, [&](int g) { k0_csr_ref<<<g, THREADS>>>(d_conv, d_wire, d_ptr, d_copy, wire_rows); });
  timeit("k1_csr_ilp<2>", csr_mb, [&](int g) { k1_csr_ilp<2><<<g, THREADS>>>(d_conv, d_wire, d_ptr, d_copy, wire_rows); });
  timeit("k1_csr_ilp<4>", csr_mb, [&](int g) { k1_csr_ilp<4><<<g, THREADS>>>(d_conv, d_wire, d_ptr, d_copy, wire_rows); });
  timeit("k1_csr_ilp<8>", csr_mb, [&](int g) { k1_csr_ilp<8><<<g, THREADS>>>(d_conv, d_wire, d_ptr, d_copy, wire_rows); });
  {
    const int smem_b = STAGES * ROW_BYTES;
    CK(cudaFuncSetAttribute(k4_csr_stream, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_b));
    timeit("k4_csr_stream", csr_mb, [&](int g) { k4_csr_stream<<<g, THREADS, smem_b>>>(d_conv, d_wire, d_ptr, d_copy, wire_rows); });
  }
  printf("--- dense padded-slot layout (S=%d slots per wire row)\n", S);
  timeit("k2_dense_zero<4>", dense_mb, [&](int g) { k2_dense_zero<4><<<g, THREADS>>>(d_dense, d_wire, wire_rows, S); });
  timeit("k3_dense_mask<4>", csr_mb, [&](int g) { k3_dense_mask<4><<<g, THREADS>>>(d_dense, d_wire, d_mask, wire_rows, S); });
  timeit("k3_dense_mask<8>", csr_mb, [&](int g) { k3_dense_mask<8><<<g, THREADS>>>(d_dense, d_wire, d_mask, wire_rows, S); });
  printf("done\n");
  return 0;
}
