// p11a_graph_k2.cu -- handoff 53 probe P11a (plan 9, deadlock class K2): inside ONE CUDA graph, can a node that
// depends on a spinning kernel be submitted ahead of that spinner's producer in a shared hardware queue (the P8 shape,
// handoff 52), so that the producer never runs?  Single GPU, stream capture from two streams, replayed many times.
//
//   S = spinner (waits until flag >= epoch; GEMM-shaped occupant with --full: 200 blocks x 128 threads, 66 KB smem)
//   D = a kernel that depends on S (the "successor": activation / join after the dispatch GEMM)
//   P = the producer of the flag (1-thread kernel, or a captured cuStreamWriteValue64 memop with --memop)
// Variants (capture order on streams A and B):
//   unsafe_after : A: S, D        then B: P   (P captured after D; P independent of S and D)
//   unsafe_before: B: P first, then A: S, D   (P captured before S; still independent)
//   safe         : A: S; B: P; join B -> A; A: D   (plan-9 rule K2: every producer of a spinner's word is an ancestor
//                  of every successor of that spinner)
// The epoch is a device counter bumped by a 1-thread head kernel at the start of each replay (no host value baked).
// A replay that does not finish within 2 s is a deadlock: the host raises the kill word (host-mapped) and counts it.
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include <chrono>
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
#define CUK(x)                                                                         \
  do {                                                                                 \
    CUresult r_ = (x);                                                                 \
    if (r_ != CUDA_SUCCESS) {                                                          \
      fprintf(stderr, "CU error %s:%d %s -> %d\n", __FILE__, __LINE__, #x, (int)r_);   \
      _exit(1);                                                                        \
    }                                                                                  \
  } while (0)

__global__ void head_kernel(unsigned long long *epoch) { *epoch += 1; }
__global__ void spin_kernel(const unsigned long long *epoch, volatile unsigned long long *flag,
                            volatile unsigned *kill, int smem_touch) {
  extern __shared__ char sm[];
  if (smem_touch) sm[threadIdx.x] = 0;
  if (threadIdx.x == 0) {
    const unsigned long long e = *epoch;
    while (*flag < e && *kill == 0) __nanosleep(500);
  }
  __syncthreads();
}
__global__ void dep_kernel(int *out) {
  if (threadIdx.x == 0) out[0] += 1;
}
__global__ void prod_kernel(const unsigned long long *epoch, unsigned long long *flag) {
  __threadfence_system();
  *(volatile unsigned long long *)flag = *epoch;
}

int main(int argc, char **argv) {
  std::string variant = argc > 1 ? argv[1] : "unsafe_after";
  bool full = false, memop = false;
  int iters = 200, pool = 0;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--full") full = true;
    else if (a == "--memop") memop = true;
    else if (a == "--iters") iters = atoi(argv[++i]);
    else if (a == "--pool") pool = atoi(argv[++i]);
  }
  CK(cudaSetDevice(0));
  CK(cudaFree(0));
  for (int i = 0; i < pool; i++) {  // perturb the stream -> hardware queue assignment like a serving process
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  }
  unsigned long long *epoch = nullptr, *flag = nullptr;
  int *out = nullptr;
  CK(cudaMalloc(&epoch, 8));
  CK(cudaMalloc(&flag, 8));
  CK(cudaMalloc(&out, 64));
  CK(cudaMemset(epoch, 0, 8));
  CK(cudaMemset(flag, 0, 8));
  unsigned *kill_h = nullptr, *kill_d = nullptr;
  CK(cudaHostAlloc((void **)&kill_h, 64, cudaHostAllocMapped));
  memset(kill_h, 0, 64);
  CK(cudaHostGetDevicePointer((void **)&kill_d, kill_h, 0));
  const int grid = full ? 200 : 1, threads = 128, smem = full ? 66560 : 0;
  if (full) CK(cudaFuncSetAttribute(spin_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cudaStream_t A, B;
  CK(cudaStreamCreateWithFlags(&A, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&B, cudaStreamNonBlocking));
  cudaEvent_t fork, join;
  CK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
  // warm every kernel (no lazy load inside the capture)
  head_kernel<<<1, 1, 0, A>>>(epoch);
  prod_kernel<<<1, 1, 0, A>>>(epoch, flag);
  spin_kernel<<<grid, threads, smem, A>>>(epoch, flag, kill_d, full);
  dep_kernel<<<1, 32, 0, A>>>(out);
  CK(cudaStreamSynchronize(A));
  auto P = [&](cudaStream_t s) {
    if (memop) {
      // value = a large constant >= any epoch reached here (a memop cannot read the device epoch)
      CUK(cuStreamWriteValue64((CUstream)s, (CUdeviceptr)flag, 1ull << 40, CU_STREAM_WRITE_VALUE_DEFAULT));
    } else {
      prod_kernel<<<1, 1, 0, s>>>(epoch, flag);
    }
  };
  auto S = [&](cudaStream_t s) { spin_kernel<<<grid, threads, smem, s>>>(epoch, flag, kill_d, full); };
  auto D = [&](cudaStream_t s) { dep_kernel<<<1, 32, 0, s>>>(out); };
  cudaGraph_t g;
  CK(cudaStreamBeginCapture(A, cudaStreamCaptureModeThreadLocal));
  head_kernel<<<1, 1, 0, A>>>(epoch);
  CK(cudaEventRecord(fork, A));
  CK(cudaStreamWaitEvent(B, fork, 0));
  if (variant == "unsafe_after") {
    S(A); D(A); P(B);
    CK(cudaEventRecord(join, B));
    CK(cudaStreamWaitEvent(A, join, 0));
  } else if (variant == "unsafe_before") {
    P(B); S(A); D(A);
    CK(cudaEventRecord(join, B));
    CK(cudaStreamWaitEvent(A, join, 0));
  } else if (variant == "unsafe_deep" || variant == "safe_deep") {
    // the producer sits DEEP in its branch (a chain of independent kernels before it), while the spinner's successor
    // D is shallow: a depth-ordered submission would enqueue D before P (the plan-9 layer: C3 forward after C2 relay
    // after the pack, vs the activation right after GEMM1)
    S(A);
    if (variant == "unsafe_deep") D(A);
    for (int k = 0; k < 6; ++k) dep_kernel<<<1, 32, 0, B>>>(out + 8);
    P(B);
    CK(cudaEventRecord(join, B));
    CK(cudaStreamWaitEvent(A, join, 0));
    if (variant == "safe_deep") D(A);
  } else if (variant == "safe") {
    S(A); P(B);
    CK(cudaEventRecord(join, B));
    CK(cudaStreamWaitEvent(A, join, 0));
    D(A);
  } else {
    fprintf(stderr, "unknown variant %s\n", variant.c_str());
    _exit(2);
  }
  CK(cudaStreamEndCapture(A, &g));
  cudaGraphExec_t ge;
  CK(cudaGraphInstantiate(&ge, g, 0));
  size_t nn = 0;
  CK(cudaGraphGetNodes(g, nullptr, &nn));
  int hangs = 0, done = 0;
  double tmax = 0, tsum = 0;
  for (int it = 0; it < iters; it++) {
    if (memop) CK(cudaMemsetAsync(flag, 0, 8, A));  // memop variant: re-arm (the head bumps the epoch anyway)
    auto t0 = std::chrono::steady_clock::now();
    CK(cudaGraphLaunch(ge, A));
    bool ok = false;
    while (true) {
      cudaError_t q = cudaStreamQuery(A);
      if (q == cudaSuccess) { ok = true; break; }
      if (q != cudaErrorNotReady) CK(q);
      if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(2)) break;
    }
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) {
      hangs++;
      kill_h[0] = 1;
      CK(cudaStreamSynchronize(A));
      kill_h[0] = 0;
      if (hangs >= 3) break;
    } else {
      done++;
      tsum += us;
      tmax = std::max(tmax, us);
    }
  }
  const char *cdmc = getenv("CUDA_DEVICE_MAX_CONNECTIONS");
  printf("P11a cdmc=%s pool=%d variant=%-14s producer=%s spinner=%s nodes=%zu replays_ok=%d hangs=%d mean_us=%.1f max_us=%.1f -> %s\n",
         cdmc ? cdmc : "default", pool, variant.c_str(), memop ? "memop" : "kernel", full ? "full" : "1blk", nn, done,
         hangs, done ? tsum / done : 0.0, tmax, hangs ? "DEADLOCK" : "ok");
  return 0;
}
