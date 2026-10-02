// p10_dev_put.cu -- handoff 53 probe P10 (plan 9): device-initiated NVSHMEM inter-node puts on Slingshot / CXI
// (libfabric transport: every remote RMA is executed by NVSHMEM's CPU proxy).  Does the signal ever become visible
// before the data (CLAUDE.md rule 5), what does one put cost, and how do concurrent puts couple (the device quiet is
// GPU-wide)?  Also P11c: are the host on-stream put and nvshmemx_barrier_all_on_stream capturable in a CUDA graph?
//
// One PE per GPU, MPI bootstrap, partner = same local rank on the next node (a wire lane): rank r puts to
// (r + L) % n and receives from (r - L) % n.  Per iteration `it` (a device counter bumped by a head kernel; nothing
// host-valued is baked into the graph):
//   fill    : src[k][i] = pattern(it, k, i) (fresh payload every iteration)
//   wire    : wait ack >= it-1 (the receiver checked the previous iteration), then put src[k] -> partner dst[k] with
//             signal sig[k] = it, for k = 0..conc-1, using the variant under test
//   check   : wait sig[k] >= it, then IMMEDIATELY read the last word of dst[k] and then every word; any word not equal
//             to pattern(it, k, i) is an ordering violation; then signal ack = it to the sender
// Variants: blk (nvshmem_putmem_signal, 1 thread per put, blocking), warp (nvshmemx_putmem_signal_warp),
// block (nvshmemx_putmem_signal_block), nqs (nvshmem_putmem_nbi; nvshmem_quiet; nvshmemx_signal_op SET),
// nbi (nvshmem_putmem_signal_nbi: the form rule 5 forbids; control), os (host nvshmemx_putmem_signal_on_stream,
// today's wire; per put issued in order).  Modes: graph (default) or eager.  --barrier adds
// nvshmemx_barrier_all_on_stream at the end of each iteration (P11c).
// Output: P10 lines per (variant, size, conc): violations, per-put device time (globaltimer, wire kernel), host
// iteration time; rank 0 prints the max over ranks.
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_rank = 0, g_size = 1;
#define CK(x)                                                                                                \
  do {                                                                                                       \
    cudaError_t e_ = (x);                                                                                    \
    if (e_ != cudaSuccess) {                                                                                 \
      fprintf(stderr, "[r%d] CUDA error %s:%d %s -> %s\n", g_rank, __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                                          \
    }                                                                                                        \
  } while (0)

enum Var { kBlk = 0, kWarp, kBlock, kNqs, kNbi, kOs };
static const char *kVarName[] = {"blk", "warp", "block", "nqs", "nbi", "os"};
constexpr int kMaxConc = 8;

__device__ __forceinline__ uint64_t gtimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint64_t pattern(uint64_t it, int k, size_t i) {
  return (it * 0x9E3779B97F4A7C15ull) ^ ((uint64_t)k << 56) ^ (i * 0xBF58476D1CE4E5B9ull);
}
__device__ __forceinline__ uint64_t ld_acquire(const uint64_t *p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

struct Bufs {
  uint64_t *src, *dst, *sig, *ack;  // symmetric: src/dst [kMaxConc][words], sig [kMaxConc], ack [1]
  uint64_t *it;                     // local device iteration counter
  unsigned long long *viol;         // [2]: last-word violations, any-word violations
  unsigned long long *tput;         // [2]: sum of per-put device ns, count
  volatile unsigned *kill;          // host-mapped kill word
};

__global__ void head_kernel(Bufs b) { *b.it += 1; }
__global__ void fill_kernel(Bufs b, size_t words, int conc) {
  const uint64_t it = *b.it;
  for (int k = 0; k < conc; ++k) {
    uint64_t *s = b.src + (size_t)k * words;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < words; i += (size_t)gridDim.x * blockDim.x)
      s[i] = pattern(it, k, i);
  }
}
// one block; thread / warp / block forms per variant; puts k = 0..conc-1 (warp w issues put w for the warp form)
__global__ void wire_kernel(Bufs b, size_t words, int conc, int var, int partner, int sender) {
  const uint64_t it = *b.it;
  const size_t bytes = words * 8;
  if (threadIdx.x == 0) {  // flow control: the receiver checked the previous iteration
    while (ld_acquire(b.ack) + 1 < it && *b.kill == 0) __nanosleep(200);
  }
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  uint64_t t0 = gtimer();
  if (var == kBlk || var == kNbi || var == kNqs) {
    if (threadIdx.x < conc) {
      const int k = threadIdx.x;
      uint64_t *d = b.dst + (size_t)k * words, *s = b.src + (size_t)k * words;
      if (var == kBlk) {
        nvshmem_putmem_signal(d, s, bytes, b.sig + k, it, NVSHMEM_SIGNAL_SET, partner);
      } else if (var == kNbi) {
        nvshmem_putmem_signal_nbi(d, s, bytes, b.sig + k, it, NVSHMEM_SIGNAL_SET, partner);
      } else {
        nvshmem_putmem_nbi(d, s, bytes, partner);
      }
    }
    if (var == kNqs) {
      __syncthreads();
      if (threadIdx.x == 0) nvshmem_quiet();
      __syncthreads();
      if (threadIdx.x < conc) nvshmemx_signal_op(b.sig + threadIdx.x, it, NVSHMEM_SIGNAL_SET, partner);
    }
  } else if (var == kWarp) {
    if (warp < conc) {
      nvshmemx_putmem_signal_warp(b.dst + (size_t)warp * words, b.src + (size_t)warp * words, bytes, b.sig + warp,
                                  it, NVSHMEM_SIGNAL_SET, partner);
    }
  } else if (var == kBlock) {
    for (int k = 0; k < conc; ++k)
      nvshmemx_putmem_signal_block(b.dst + (size_t)k * words, b.src + (size_t)k * words, bytes, b.sig + k, it,
                                   NVSHMEM_SIGNAL_SET, partner);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(b.tput, gtimer() - t0);
    atomicAdd(b.tput + 1, 1ull);
  }
  (void)lane;
}
// grid kernel: block 0 thread 0 waits for every signal (then the grid checks); the last-word check runs first
__global__ void check_kernel(Bufs b, size_t words, int conc, int sender) {
  const uint64_t it = *b.it;
  __shared__ int ok;
  if (threadIdx.x == 0) {
    ok = 1;
    for (int k = 0; k < conc; ++k) {
      while (ld_acquire(b.sig + k) < it) {
        if (*b.kill) { ok = 0; break; }
        __nanosleep(100);
      }
      // the moment the signal is visible: the last word of the payload must already be the new one
      const uint64_t *d = b.dst + (size_t)k * words;
      if (ok && blockIdx.x == 0 && *(volatile const uint64_t *)(d + words - 1) != pattern(it, k, words - 1))
        atomicAdd(b.viol, 1ull);
    }
  }
  __syncthreads();
  if (!ok) return;
  for (int k = 0; k < conc; ++k) {
    const uint64_t *d = b.dst + (size_t)k * words;
    unsigned long long bad = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < words; i += (size_t)gridDim.x * blockDim.x)
      bad += (*(volatile const uint64_t *)(d + i) != pattern(it, k, i));
    if (bad) atomicAdd(b.viol + 1, bad);
  }
}
// os variant: the same flow control as the wire kernel, before the host on-stream puts
__global__ void ackwait_kernel(Bufs b) {
  const uint64_t it = *b.it;
  while (ld_acquire(b.ack) + 1 < it && *b.kill == 0) __nanosleep(200);
}
// ack after the whole check grid finished (separate kernel, stream-ordered)
__global__ void ack_kernel(Bufs b, int sender) {
  nvshmemx_signal_op(b.ack, *b.it, NVSHMEM_SIGNAL_SET, sender);
}

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_size);
  int iters = 300, L = 4;
  bool eager = false, barrier = false;
  std::vector<std::string> vars = {"blk", "warp", "block", "nqs", "nbi", "os"};
  std::vector<size_t> sizes = {4096, 65536, 1u << 20, 4u << 20};
  std::vector<int> concs = {1, 4};
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--iters") iters = atoi(argv[++i]);
    else if (a == "--eager") eager = true;
    else if (a == "--barrier") barrier = true;
    else if (a == "--vars") {
      vars.clear();
      std::string s = argv[++i];
      size_t p = 0;
      while (p != std::string::npos) {
        size_t q = s.find(',', p);
        vars.push_back(s.substr(p, q == std::string::npos ? q : q - p));
        p = q == std::string::npos ? q : q + 1;
      }
    } else if (a == "--conc") {
      concs = {atoi(argv[++i])};
    }
  }
  int ndev = 0;
  CK(cudaGetDeviceCount(&ndev));
  CK(cudaSetDevice(g_rank % ndev));
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  MPI_Comm comm = MPI_COMM_WORLD;
  attr.mpi_comm = &comm;
  if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr) != 0) {
    fprintf(stderr, "nvshmemx_init_attr failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  const int partner = (g_rank + L) % g_size, sender = (g_rank - L + g_size) % g_size;
  const size_t max_words = (4u << 20) / 8;
  Bufs b{};
  b.src = (uint64_t *)nvshmem_malloc(max_words * 8 * kMaxConc);
  b.dst = (uint64_t *)nvshmem_malloc(max_words * 8 * kMaxConc);
  b.sig = (uint64_t *)nvshmem_malloc(64 * 8);
  b.ack = (uint64_t *)nvshmem_malloc(64 * 8);
  if (!b.src || !b.dst || !b.sig || !b.ack) {
    fprintf(stderr, "nvshmem_malloc failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  CK(cudaMalloc(&b.it, 8));
  CK(cudaMalloc(&b.viol, 16));
  CK(cudaMalloc(&b.tput, 16));
  unsigned *kill_h = nullptr, *kill_d = nullptr;
  CK(cudaHostAlloc((void **)&kill_h, 64, cudaHostAllocMapped));
  memset(kill_h, 0, 64);
  CK(cudaHostGetDevicePointer((void **)&kill_d, kill_h, 0));
  b.kill = kill_d;
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  if (g_rank == 0)
    printf("INFO P10 ranks=%d L=%d partner(0)=%d mode=%s barrier=%d iters=%d\n", g_size, L, partner,
           eager ? "eager" : "graph", (int)barrier, iters);
  for (const std::string &vn : vars) {
    int var = -1;
    for (int v = 0; v < 6; ++v)
      if (vn == kVarName[v]) var = v;
    if (var < 0) continue;
    for (size_t bytes : sizes) {
      for (int conc : concs) {
        const size_t words = bytes / 8;
        // fresh state for every cell: all signals / acks / counters zero, every PE synchronized
        CK(cudaMemset(b.sig, 0, 64 * 8));
        CK(cudaMemset(b.ack, 0, 64 * 8));
        CK(cudaMemset(b.it, 0, 8));
        CK(cudaMemset(b.viol, 0, 16));
        CK(cudaMemset(b.tput, 0, 16));
        CK(cudaDeviceSynchronize());
        nvshmem_barrier_all();
        auto body = [&](cudaStream_t s) {
          head_kernel<<<1, 1, 0, s>>>(b);
          fill_kernel<<<64, 256, 0, s>>>(b, words, conc);
          if (var == kOs) {
            // today's wire: host on-stream blocking put per lane (the value is the iteration as a host constant is
            // not possible in a graph -> signal value = a constant that only increases per cell: use it via a
            // device word is impossible for the host API, so this variant runs eager only)
            ackwait_kernel<<<1, 1, 0, s>>>(b);
            for (int k = 0; k < conc; ++k)
              nvshmemx_putmem_signal_on_stream(b.dst + (size_t)k * words, b.src + (size_t)k * words, bytes,
                                               b.sig + k, 1, NVSHMEM_SIGNAL_ADD, partner, s);
          } else {
            const int threads = var == kWarp ? 32 * conc : (var == kBlock ? 256 : 32);
            wire_kernel<<<1, threads, 0, s>>>(b, words, conc, var, partner, sender);
          }
          check_kernel<<<16, 256, 0, s>>>(b, words, conc, sender);
          ack_kernel<<<1, 1, 0, s>>>(b, sender);
          if (barrier) nvshmemx_barrier_all_on_stream(s);
        };
        cudaGraphExec_t ge = nullptr;
        const bool use_graph = !eager && var != kOs;
        if (use_graph) {
          cudaGraph_t g;
          CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
          body(st);
          CK(cudaStreamEndCapture(st, &g));
          CK(cudaGraphInstantiate(&ge, g, 0));
        }
        std::vector<double> host_us;
        bool hung = false;
        for (int it = 0; it < iters && !hung; ++it) {
          auto t0 = std::chrono::steady_clock::now();
          if (var == kOs) {
            // os: signal ADD 1 per iteration so the receiver's `sig >= it` holds (SET of a host constant is not
            // expressible per iteration without baking); data ordering is what is tested
            body(st);
          } else if (use_graph) {
            CK(cudaGraphLaunch(ge, st));
          } else {
            body(st);
          }
          while (true) {
            cudaError_t q = cudaStreamQuery(st);
            if (q == cudaSuccess) break;
            if (q != cudaErrorNotReady) CK(q);
            if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10)) {
              hung = true;
              kill_h[0] = 1;
              break;
            }
          }
          host_us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        }
        CK(cudaStreamSynchronize(st));
        kill_h[0] = 0;
        unsigned long long v[2], tp[2];
        CK(cudaMemcpy(v, b.viol, 16, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(tp, b.tput, 16, cudaMemcpyDeviceToHost));
        std::sort(host_us.begin(), host_us.end());
        double med = host_us.empty() ? 0 : host_us[host_us.size() / 2];
        double p90 = host_us.empty() ? 0 : host_us[host_us.size() * 9 / 10];
        double put_us = tp[1] ? (double)tp[0] / tp[1] / 1000.0 : 0;
        // max over ranks
        double loc[5] = {(double)v[0], (double)v[1], med, p90, put_us}, mx[5];
        int h = hung ? 1 : 0, hmx = 0;
        MPI_Reduce(loc, mx, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&h, &hmx, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (g_rank == 0)
          printf("P10 var=%-5s mode=%-5s bytes=%8zu conc=%d iters=%d viol_lastword=%.0f viol_any=%.0f iter_us med=%.1f "
                 "p90=%.1f put_dev_us=%.1f hung=%d -> %s\n",
                 kVarName[var], use_graph ? "graph" : "eager", bytes, conc, iters, mx[0], mx[1], mx[2], mx[3], mx[4],
                 hmx, hmx ? "HUNG" : (mx[0] + mx[1] > 0 ? "VIOLATIONS" : "ok"));
        if (ge) CK(cudaGraphExecDestroy(ge));
        if (hmx) {  // a hung cell leaves PEs out of step: stop the run
          MPI_Barrier(MPI_COMM_WORLD);
          nvshmem_finalize();
          MPI_Finalize();
          return 0;
        }
        CK(cudaDeviceSynchronize());
        nvshmem_barrier_all();
      }
    }
  }
  nvshmem_finalize();
  MPI_Finalize();
  return 0;
}
