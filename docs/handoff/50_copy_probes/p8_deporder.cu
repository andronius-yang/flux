// p8_deporder.cu -- handoff 52 probe P8: with a kernel spinning on the device, does an op enqueued on the SAME
// stream behind the spinner (a dependent kernel launch, an event record) block LATER work of OTHER streams when the
// streams share a hardware queue (CUDA_DEVICE_MAX_CONNECTIONS=1)? This is the C4 no-sync shape: the main thread
// launches the dispatch GEMM (spins on wire the proxy issues), then keeps enqueueing GEMM-dependent work on the main
// stream, and the proxy issues the GEMM's producers (a D2D copy + a signal write on cp_stream) only later.
//
// Every case: spinner on `main` (waits for a host-mapped flag F, or a kill flag after 3 s), then the case's ops,
// then on `cp`: [event e_pre] [D2D copy 4 MiB] [stream write F = 1] [event e_post]. Reported after 2 s: whether
// e_pre / e_post completed and whether F landed. Then the kill flag frees the spinner.
//   base        : nothing between the spinner and the cp ops
//   dep_kernel  : a trivial kernel on main (stream-ordered after the spinner)
//   dep_event   : cudaEventRecord on main
//   dep_wgeq    : a one-warp spin kernel on main waiting for F (the wait_geq_kernel join shape)
//   fe_wait     : cudaStreamWaitEvent(other, event recorded on main after the spinner) (front-end wait, NR-02 B)
//   pre12_dep   : [cp: e_pre] first, THEN the dependent kernel on main, then the rest of the cp ops
//   write_only  : like dep_kernel but cp has no copy (signal write only)
//   full_<case> : the spinner fills every SM (one 1024-thread block per SM minus 8), as the persistent GEMM does
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#define CK(x)                                                                                         \
  do {                                                                                                \
    cudaError_t e_ = (x);                                                                             \
    if (e_ != cudaSuccess) {                                                                          \
      fprintf(stderr, "CUDA error %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      _exit(1);                                                                                       \
    }                                                                                                 \
  } while (0)
#define CUK(x)                                                              \
  do {                                                                      \
    CUresult r_ = (x);                                                      \
    if (r_ != CUDA_SUCCESS) {                                               \
      fprintf(stderr, "CU error %s:%d %s -> %d\n", __FILE__, __LINE__, #x, (int)r_); \
      _exit(1);                                                             \
    }                                                                       \
  } while (0)

__global__ void spin_kernel(volatile unsigned *flag, volatile unsigned *kill) {
  if (threadIdx.x == 0) {
    while (*flag == 0 && *kill == 0) {
      __nanosleep(500);
    }
  }
  __syncthreads();
}
__global__ void trivial_kernel(int *out) {
  if (threadIdx.x == 0) out[0] += 1;
}
__global__ void wgeq_kernel(volatile unsigned *flag, volatile unsigned *kill) {
  if (threadIdx.x == 0) {
    while (*flag == 0 && *kill == 0) {
      __nanosleep(500);
    }
  }
}

static const char *q(cudaEvent_t e) {
  cudaError_t r = cudaEventQuery(e);
  return r == cudaSuccess ? "ok" : (r == cudaErrorNotReady ? "PEND" : "ERR");
}

int main(int argc, char **argv) {
  std::string mode = argc > 1 ? argv[1] : "base";
  const bool full = mode.rfind("full_", 0) == 0;
  const std::string c = full ? mode.substr(5) : mode;
  CK(cudaSetDevice(0));
  CK(cudaFree(0));
  int nsm = 0;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  unsigned *flag_h = nullptr, *kill_h = nullptr;
  CK(cudaHostAlloc((void **)&flag_h, 64, cudaHostAllocMapped));
  CK(cudaHostAlloc((void **)&kill_h, 64, cudaHostAllocMapped));
  memset(flag_h, 0, 64);
  memset(kill_h, 0, 64);
  unsigned *flag_d = nullptr, *kill_d = nullptr;
  CK(cudaHostGetDevicePointer((void **)&flag_d, flag_h, 0));
  CK(cudaHostGetDevicePointer((void **)&kill_d, kill_h, 0));
  const size_t nbytes = 4u << 20;
  void *a = nullptr, *b = nullptr;
  int *scratch = nullptr;
  CK(cudaMalloc(&a, nbytes));
  CK(cudaMalloc(&b, nbytes));
  CK(cudaMalloc((void **)&scratch, 64));
  // P8_PRIO=hp: cp and other at the highest priority (main stays default); P8_NSTREAMS=n: n default and n high
  // priority streams created first (the pools torch and NCCL hold in a serving process)
  int lo = 0, hi = 0;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  const bool hp = getenv("P8_PRIO") != nullptr && std::string(getenv("P8_PRIO")) == "hp";
  const int npool = getenv("P8_NSTREAMS") ? atoi(getenv("P8_NSTREAMS")) : 0;
  for (int i = 0; i < npool; i++) {
    cudaStream_t s1, s2;
    CK(cudaStreamCreateWithPriority(&s1, cudaStreamNonBlocking, lo));
    CK(cudaStreamCreateWithPriority(&s2, cudaStreamNonBlocking, hi));
  }
  cudaStream_t main_s, cp, other;
  CK(cudaStreamCreateWithPriority(&main_s, cudaStreamNonBlocking, lo));
  CK(cudaStreamCreateWithPriority(&cp, cudaStreamNonBlocking, hp ? hi : lo));
  CK(cudaStreamCreateWithPriority(&other, cudaStreamNonBlocking, hp ? hi : lo));
  cudaEvent_t e_pre, e_post, e_main;
  CK(cudaEventCreateWithFlags(&e_pre, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&e_post, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&e_main, cudaEventDisableTiming));
  // warm every kernel and op once (no lazy load inside the measurement)
  spin_kernel<<<1, 32, 0, main_s>>>(flag_d, kill_d);
  trivial_kernel<<<1, 32, 0, main_s>>>(scratch);
  kill_h[0] = 1;
  wgeq_kernel<<<1, 32, 0, main_s>>>(flag_d, kill_d);
  CK(cudaMemcpyAsync(b, a, nbytes, cudaMemcpyDeviceToDevice, cp));
  CUK(cuStreamWriteValue32((CUstream)cp, (CUdeviceptr)(flag_d + 1), 1, 0));
  CK(cudaDeviceSynchronize());
  kill_h[0] = 0;
  flag_h[0] = 0;

  const int grid = full ? (nsm - 8) : 1;
  const int threads = full ? 1024 : 32;
  spin_kernel<<<grid, threads, 0, main_s>>>(flag_d, kill_d);
  usleep(20000);  // the spinner is resident
  auto cp_ops = [&](bool with_pre, bool with_copy) {
    if (with_pre) CK(cudaEventRecord(e_pre, cp));
    if (with_copy) CK(cudaMemcpyAsync(b, a, nbytes, cudaMemcpyDeviceToDevice, cp));
    CUK(cuStreamWriteValue32((CUstream)cp, (CUdeviceptr)flag_d, 1, 0));
    CK(cudaEventRecord(e_post, cp));
  };
  if (c == "base") {
    cp_ops(true, true);
  } else if (c == "dep_kernel") {
    trivial_kernel<<<1, 32, 0, main_s>>>(scratch);
    cp_ops(true, true);
  } else if (c == "dep_event") {
    CK(cudaEventRecord(e_main, main_s));
    cp_ops(true, true);
  } else if (c == "dep_wgeq") {
    wgeq_kernel<<<1, 32, 0, main_s>>>(flag_d, kill_d);
    cp_ops(true, true);
  } else if (c == "fe_wait") {
    CK(cudaEventRecord(e_main, main_s));
    CK(cudaStreamWaitEvent(other, e_main, 0));
    trivial_kernel<<<1, 32, 0, other>>>(scratch);
    cp_ops(true, true);
  } else if (c == "pre12_dep") {
    CK(cudaEventRecord(e_pre, cp));
    trivial_kernel<<<1, 32, 0, main_s>>>(scratch);
    cp_ops(false, true);
  } else if (c == "write_only") {
    trivial_kernel<<<1, 32, 0, main_s>>>(scratch);
    cp_ops(true, false);
  } else {
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    _exit(2);
  }
  usleep(2000000);
  const char *cdmc = getenv("CUDA_DEVICE_MAX_CONNECTIONS");
  printf("P8 prio=%s pool=%d cdmc=%s mode=%-16s e_pre=%-4s flag=%u e_post=%-4s -> %s\n", hp ? "hp" : "def", npool, cdmc ? cdmc : "default", mode.c_str(),
         c == "pre12_dep" ? q(e_pre) : q(e_pre), flag_h[0], q(e_post),
         flag_h[0] ? "released" : "BLOCKED (deadlock without the kill flag)");
  fflush(stdout);
  kill_h[0] = 1;
  CK(cudaDeviceSynchronize());
  return 0;
}
