// p9_push.cu -- handoff 53 probe P9 (plan 9): NVLink delivery of packed rows to the 3 node peers, today's path vs
// the fused device push, all 4 GPUs of a node at once (one process, peer access), with background GPU load.
//
//   ce   : gather kernel (rows by index into a local send buffer, k blocks) -> per peer cudaMemcpyAsync (copy engine)
//          -> per peer cuStreamWriteValue64 signal to the peer-mapped word        (today's dispatch round 0 / combine
//          intra lanes, issued from the host)
//   push : ONE kernel: gather rows by index and store them straight into each peer's receive region (16-byte
//          vectors), per-destination __threadfence_system + last-block counter, the last block release-stores the
//          peer's signal                                                     (plan 9 C1 / D1)
// Background on every GPU during the measurement: none | occ (GEMM1-shaped persistent occupant: 200 blocks x 128
// threads, 66560 B dynamic smem, spinning until released) | gemm (cuBLAS bf16 GEMM 2048 x 1536 x 2048 in a loop on
// another stream: real SM + HBM contention).
// Rows are 4096 B (hidden 2048 bf16); per-peer bytes = rows * 4096. Metric: GPU time from the first op to the last
// delivery op on each source GPU (CUDA events on its stream), max over the 4 GPUs; median / p90 over iterations;
// payload checked on the receivers after every cell (bad rows must be 0).
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                                         \
  do {                                                                                                \
    cudaError_t e_ = (x);                                                                             \
    if (e_ != cudaSuccess) {                                                                          \
      fprintf(stderr, "CUDA error %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      _exit(1);                                                                                       \
    }                                                                                                 \
  } while (0)
#define CUK(x)                                                                       \
  do {                                                                               \
    CUresult r_ = (x);                                                               \
    if (r_ != CUDA_SUCCESS) {                                                        \
      fprintf(stderr, "CU error %s:%d %s -> %d\n", __FILE__, __LINE__, #x, (int)r_); \
      _exit(1);                                                                      \
    }                                                                                \
  } while (0)

constexpr int kG = 4;            // GPUs per node
constexpr int kRowBytes = 4096;  // hidden 2048 x bf16
constexpr int kTokens = 8192;    // source token rows per GPU

struct PushArgs {
  const uint4 *src;          // [kTokens][kRowBytes/16]
  const int *gather;         // [3][rows] token index per row, per peer
  uint4 *dst[3];             // peer receive region (peer-mapped, UVA)
  unsigned long long *sig[3];  // peer signal word for this source (peer-mapped)
  unsigned *counter;         // [3] local last-block counters (self-resetting)
  long long rows;            // rows per peer
  unsigned long long epoch;
};

__global__ void __launch_bounds__(512, 2) push_kernel(PushArgs a) {
  constexpr int vpr = kRowBytes / 16;
  for (int p = 0; p < 3; ++p) {  // destinations outer, grid-stride inner
    const long long total = a.rows * vpr;
    const int *gi = a.gather + (long long)p * a.rows;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
      const long long r = i / vpr, v = i - r * vpr;
      a.dst[p][r * vpr + v] = a.src[(long long)gi[r] * vpr + v];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      const unsigned prev = atomicInc(a.counter + p, gridDim.x - 1);  // wraps to 0 on the last arrival
      if (prev == gridDim.x - 1) {
        __threadfence_system();
        asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(a.sig[p]), "l"(a.epoch) : "memory");
      }
    }
  }
}
// today's local pack: gather rows by index into the local send buffer (3 contiguous segments)
__global__ void __launch_bounds__(512, 2) gather_kernel(const uint4 *src, const int *gather, uint4 *send, long long rows) {
  constexpr int vpr = kRowBytes / 16;
  const long long total = 3 * rows * vpr;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
    const long long r = i / vpr, v = i - r * vpr;
    send[r * vpr + v] = src[(long long)gather[r] * vpr + v];
  }
}
// GEMM1-shaped persistent spinner: every thread spins with system-scope acquire loads on a DEVICE word (the GEMM1
// tile gate's spin on its arrival signals); released by a stream write-value memop (no SM needed)
__global__ void __launch_bounds__(128, 1) occupant_kernel(const unsigned long long *release, float *sink) {
  extern __shared__ float sm[];
  sm[threadIdx.x] = threadIdx.x;
  unsigned long long v = 0;
  do {
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(release) : "memory");
  } while (v == 0);
  if (sm[threadIdx.x] == -1.f) sink[0] = 1.f;
}
__global__ void fill_src(uint4 *src, int dev) {
  const long long total = (long long)kTokens * kRowBytes / 16;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x)
    src[i] = make_uint4((unsigned)i, (unsigned)(i >> 32), (unsigned)dev, 0xC0FFEEu);
}
// receiver check: region from source s must hold src_s rows gather_s[p][r]
__global__ void check_kernel(const uint4 *dst, const int *gather, long long rows, int src_dev, unsigned long long *bad) {
  constexpr int vpr = kRowBytes / 16;
  const long long total = rows * vpr;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
    const long long r = i / vpr, v = i - r * vpr;
    const long long si = (long long)gather[r] * vpr + v;
    const uint4 x = dst[i];
    if (x.x != (unsigned)si || x.y != (unsigned)(si >> 32) || x.z != (unsigned)src_dev || x.w != 0xC0FFEEu)
      atomicAdd(bad, 1ull);
  }
}

int main(int argc, char **argv) {
  std::string bg = argc > 1 ? argv[1] : "none";
  int iters = 50;
  for (int i = 2; i < argc; ++i)
    if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
  int ndev = 0;
  CK(cudaGetDeviceCount(&ndev));
  if (ndev < kG) {
    fprintf(stderr, "need %d GPUs\n", kG);
    return 1;
  }
  const long long max_rows = 4096;  // up to 16 MiB per peer
  struct Dev {
    uint4 *src, *send, *recv;            // recv: [kG sources][max_rows rows]
    int *gather;                         // [3][max_rows]
    unsigned long long *sig;             // [kG]
    unsigned *counter;
    unsigned long long *bad;
    cudaStream_t s, bgs;
    cudaEvent_t e0, e1;
    float *sink;
    cublasHandle_t cb;
    unsigned long long *rel;
    cudaStream_t rels;
    __nv_bfloat16 *A, *B, *C;
  } D[kG];
  unsigned *rel_h = nullptr;
  CK(cudaHostAlloc((void **)&rel_h, 64, cudaHostAllocMapped | cudaHostAllocPortable));
  memset(rel_h, 0, 64);
  for (int d = 0; d < kG; ++d) {
    CK(cudaSetDevice(d));
    for (int p = 0; p < kG; ++p)
      if (p != d) {
        cudaError_t e = cudaDeviceEnablePeerAccess(p, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CK(e);
      }
    Dev &x = D[d];
    CK(cudaMalloc(&x.src, (size_t)kTokens * kRowBytes));
    CK(cudaMalloc(&x.send, (size_t)3 * max_rows * kRowBytes));
    CK(cudaMalloc(&x.recv, (size_t)kG * max_rows * kRowBytes));
    CK(cudaMalloc(&x.gather, (size_t)3 * max_rows * sizeof(int)));
    CK(cudaMalloc(&x.sig, kG * 8));
    CK(cudaMalloc(&x.counter, 16));
    CK(cudaMalloc(&x.bad, 8));
    CK(cudaMalloc(&x.sink, 64));
    CK(cudaMalloc(&x.rel, 8));
    CK(cudaStreamCreateWithFlags(&x.rels, cudaStreamNonBlocking));
    CK(cudaMemset(x.sig, 0, kG * 8));
    CK(cudaMemset(x.counter, 0, 16));
    CK(cudaStreamCreateWithFlags(&x.s, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&x.bgs, cudaStreamNonBlocking));
    CK(cudaEventCreate(&x.e0));
    CK(cudaEventCreate(&x.e1));
    fill_src<<<256, 256>>>(x.src, d);
    std::vector<int> g(3 * max_rows);
    srand(1234 + d);
    for (auto &v : g) v = rand() % kTokens;
    CK(cudaMemcpy(x.gather, g.data(), g.size() * sizeof(int), cudaMemcpyHostToDevice));
    if (bg == "gemm") {
      if (cublasCreate(&x.cb) != CUBLAS_STATUS_SUCCESS) return 1;
      cublasSetStream(x.cb, x.bgs);
      CK(cudaMalloc(&x.A, 2048ull * 2048 * 2));
      CK(cudaMalloc(&x.B, 2048ull * 1536 * 2));
      CK(cudaMalloc(&x.C, 2048ull * 1536 * 2));
    }
    CK(cudaFuncSetAttribute(occupant_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 66560));
    CK(cudaDeviceSynchronize());
  }
  {
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, push_kernel));
    int occ_push_beside = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_push_beside, push_kernel, 512, 0));
    cudaFuncAttributes fo;
    CK(cudaFuncGetAttributes(&fo, occupant_kernel));
    printf("INFO push_kernel regs=%d (512 thr: %d regs/block, max blocks/SM alone %d); occupant regs=%d smem=66560\n",
           fa.numRegs, fa.numRegs * 512, occ_push_beside, fo.numRegs);
  }
  auto start_bg = [&]() {
    for (int d = 0; d < kG; ++d) {
      CK(cudaSetDevice(d));
      CK(cudaMemset(D[d].rel, 0, 8));
      CK(cudaDeviceSynchronize());
      if (bg == "occ") occupant_kernel<<<200, 128, 66560, D[d].bgs>>>(D[d].rel, D[d].sink);
    }
    if (bg == "occ") usleep(20000);
  };
  auto gemm_bg = [&](int d) {  // a few GEMMs queued on the background stream per iteration
    const float alpha = 1.f, beta = 0.f;
    for (int k = 0; k < 3; ++k)
      cublasGemmEx(D[d].cb, CUBLAS_OP_N, CUBLAS_OP_N, 1536, 2048, 2048, &alpha, D[d].B, CUDA_R_16BF, 1536, D[d].A,
                   CUDA_R_16BF, 2048, &beta, D[d].C, CUDA_R_16BF, 1536, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  };
  auto stop_bg = [&]() {
    for (int d = 0; d < kG; ++d) {
      CK(cudaSetDevice(d));
      CUK(cuStreamWriteValue64((CUstream)D[d].rels, (CUdeviceptr)D[d].rel, 1, CU_STREAM_WRITE_VALUE_DEFAULT));
    }
    for (int d = 0; d < kG; ++d) {
      CK(cudaSetDevice(d));
      CK(cudaDeviceSynchronize());
    }
  };
  const long long rows_list[] = {128, 256, 512, 1024};  // 0.5 / 1 / 2 / 4 MiB per peer
  const int blocks_list[] = {8, 16, 32, 64};
  unsigned long long epoch = 0;
  for (long long rows : rows_list) {
    for (int mode = 0; mode < 2; ++mode) {  // 0 = ce, 1 = push
      for (int bi = 0; bi < 4; ++bi) {
        const int blocks = blocks_list[bi];
        std::vector<float> tms;
        start_bg();
        for (int it = 0; it < iters; ++it) {
          ++epoch;
          for (int d = 0; d < kG; ++d) {
            CK(cudaSetDevice(d));
            Dev &x = D[d];
            if (bg == "gemm") gemm_bg(d);
            CK(cudaEventRecord(x.e0, x.s));
            int pi = 0;
            uint4 *dsts[3];
            unsigned long long *sigs[3];
            for (int p = 0; p < kG; ++p) {
              if (p == d) continue;
              dsts[pi] = D[p].recv + (long long)d * max_rows * (kRowBytes / 16);
              sigs[pi] = D[p].sig + d;
              ++pi;
            }
            if (mode == 1) {
              PushArgs a{x.src, x.gather, {dsts[0], dsts[1], dsts[2]}, {sigs[0], sigs[1], sigs[2]}, x.counter, rows, epoch};
              push_kernel<<<blocks, 512, 0, x.s>>>(a);
            } else {
              gather_kernel<<<blocks, 512, 0, x.s>>>(x.src, x.gather, x.send, rows);
              for (int k = 0; k < 3; ++k) {
                CK(cudaMemcpyAsync(dsts[k], x.send + (long long)k * rows * (kRowBytes / 16), rows * kRowBytes,
                                   cudaMemcpyDeviceToDevice, x.s));
                CUK(cuStreamWriteValue64((CUstream)x.s, (CUdeviceptr)sigs[k], epoch, CU_STREAM_WRITE_VALUE_DEFAULT));
              }
            }
            CK(cudaEventRecord(x.e1, x.s));
          }
          float worst = 0;
          for (int d = 0; d < kG; ++d) {
            CK(cudaSetDevice(d));
            CK(cudaEventSynchronize(D[d].e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, D[d].e0, D[d].e1));
            worst = std::max(worst, ms);
          }
          if (it >= 3) tms.push_back(worst * 1000.f);
        }
        stop_bg();
        // payload check (the last iteration's data; gather indices: ce copies segment k = the k-th peer's rows)
        unsigned long long bad_total = 0;
        for (int d = 0; d < kG; ++d) {
          CK(cudaSetDevice(d));
          for (int s = 0; s < kG; ++s) {
            if (s == d) continue;
            const int k = d < s ? d : d - 1;  // d is the k-th peer of source s
            CK(cudaMemset(D[d].bad, 0, 8));
            // the gather table of source s lives on device s: peer read
            check_kernel<<<64, 256>>>(D[d].recv + (long long)s * max_rows * (kRowBytes / 16), D[s].gather + k * rows,
                                      rows, s, D[d].bad);
            unsigned long long b = 0;
            CK(cudaMemcpy(&b, D[d].bad, 8, cudaMemcpyDeviceToHost));
            bad_total += b;
          }
        }
        std::sort(tms.begin(), tms.end());
        printf("P9 bg=%-4s mode=%-4s per_peer_MiB=%.2f blocks=%2d us med=%7.1f p90=%7.1f bad=%llu\n", bg.c_str(),
               mode ? "push" : "ce", rows * kRowBytes / 1048576.0, blocks, tms[tms.size() / 2], tms[tms.size() * 9 / 10],
               bad_total);
      }
    }
  }
  return 0;
}
