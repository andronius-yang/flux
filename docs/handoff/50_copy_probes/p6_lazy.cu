// p6_lazy.cu -- handoff 51 probe P6: while the main thread is inside the FIRST launch of a not-yet-loaded kernel
// (CUDA_MODULE_LOADING=LAZY) and another kernel spins on the device, can a second host thread still release the
// spinner, and with which kind of call?  One process, one GPU, one mode per process (a first launch happens once).
//
//   mode rt    : thread B releases with a runtime call  (cudaMemcpyAsync H2D of the flag on its own stream)
//   mode drv   : thread B releases with a driver call   (cuStreamWriteValue32 on its own stream)
//   mode host  : thread B releases with a plain store to host-mapped pinned memory (no API call)
//   mode late  : like host, but B waits 3 s first: does the first launch wait for the device (returns after ~3 s)
//                or not (returns at once while the spinner still runs)?
// Timeline printed in ms from the moment A enters the cold launch. A watchdog ends the process after 12 s and
// reports which calls never returned (a deadlock).
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include <atomic>
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

__global__ void spin_kernel(volatile int *dflag, volatile int *hflag, int *done) {
  if (threadIdx.x == 0) {
    while (*dflag == 0 && *hflag == 0) {
      __nanosleep(1000);
    }
    *done = 1;
  }
}
#ifdef COLD_SO
extern "C" void p6_launch_cold(cudaStream_t s, int *out);  // p6_cold.cu, a separate library (module-level load)
#endif
// never launched before the measured call: its first launch loads it under LAZY loading
__global__ void cold_kernel(int *out) {
  if (threadIdx.x == 0) {
    out[1] = 42;
  }
}

using clk = std::chrono::steady_clock;
static clk::time_point t0;
static double ms() { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }
static std::atomic<double> a_enter{-1}, a_return{-1}, b_call{-1}, b_return{-1};

int main(int argc, char **argv) {
  std::string mode = argc > 1 ? argv[1] : "drv";
  // "<mode>_nocold": control without the cold launch (does the release path itself work?)
  const bool nocold = mode.size() > 7 && mode.compare(mode.size() - 7, 7, "_nocold") == 0;
  if (nocold) {
    mode = mode.substr(0, mode.size() - 7);
  }
  const char *lz = getenv("CUDA_MODULE_LOADING");
#ifdef COLD_SO
  setvbuf(stdout, nullptr, _IONBF, 0);
  printf("P6 (cold kernel in a separate library) mode %s, CUDA_MODULE_LOADING=%s\n", mode.c_str(), lz ? lz : "(unset)");
#else
  printf("P6 (cold kernel in the same module) mode %s, CUDA_MODULE_LOADING=%s\n", mode.c_str(), lz ? lz : "(unset)");
#endif
  CK(cudaSetDevice(0));
  int *dflag, *done, *out, *hflag_h, *hflag_d, *one_h;
  CK(cudaMalloc(&dflag, 4));
  CK(cudaMalloc(&done, 4));
  CK(cudaMalloc(&out, 8));
  CK(cudaHostAlloc(&hflag_h, 4, cudaHostAllocMapped));
  CK(cudaHostGetDevicePointer(&hflag_d, hflag_h, 0));
  CK(cudaHostAlloc(&one_h, 4, cudaHostAllocDefault));
  *one_h = 1;
  cudaStream_t s_spin, s_b, s_cold;
  CK(cudaStreamCreateWithFlags(&s_spin, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&s_b, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&s_cold, cudaStreamNonBlocking));
  // warm the spinner (load it) with the flag already set, and warm the driver write path
  CK(cudaMemset(dflag, 0, 4));
  *hflag_h = 1;
  spin_kernel<<<1, 32, 0, s_spin>>>(dflag, hflag_d, done);
  CK(cudaStreamSynchronize(s_spin));
  if (cuStreamWriteValue32((CUstream)s_b, (CUdeviceptr)out, 0, CU_STREAM_WRITE_VALUE_DEFAULT) != CUDA_SUCCESS) {
    printf("cuStreamWriteValue32 unsupported\n");
    return 1;
  }
  CK(cudaMemcpyAsync(out, one_h, 4, cudaMemcpyHostToDevice, s_b));
  CK(cudaStreamSynchronize(s_b));
  CK(cudaMemset(dflag, 0, 4));
  CK(cudaMemset(done, 0, 4));
  *hflag_h = 0;
  CK(cudaDeviceSynchronize());

  // the spinner now runs until released
  spin_kernel<<<1, 32, 0, s_spin>>>(dflag, hflag_d, done);
  usleep(50000);
  t0 = clk::now();
  std::thread watchdog([&] {
    for (int i = 0; i < 120; i++) {
      usleep(100000);
      if (a_return.load() >= 0 && (b_return.load() >= 0 || mode == "host" || mode == "late")) {
        return;
      }
    }
    printf("RESULT mode %s DEADLOCK after 12 s: A cold launch %s (entered %.1f), B call %s (called %.1f)\n",
           mode.c_str(), a_return.load() >= 0 ? "returned" : "NEVER RETURNED", a_enter.load(),
           b_return.load() >= 0 ? "returned" : "never returned", b_call.load());
    fflush(stdout);
    _exit(3);
  });
  std::thread b([&] {
    while (a_enter.load() < 0) {
      std::this_thread::yield();
    }
    usleep(mode == "late" ? 3000000 : 200000);  // A is inside the cold launch by now
    b_call = ms();
    if (mode == "rt") {
      CK(cudaMemcpyAsync((void *)dflag, one_h, 4, cudaMemcpyHostToDevice, s_b));
    } else if (mode == "drv") {
      CUresult r = cuStreamWriteValue32((CUstream)s_b, (CUdeviceptr)dflag, 1, CU_STREAM_WRITE_VALUE_DEFAULT);
      if (r != CUDA_SUCCESS) {
        printf("cuStreamWriteValue32 failed %d\n", (int)r);
      }
    } else {
      *(volatile int *)hflag_h = 1;  // host / late: no API call at all
    }
    b_return = ms();
  });
  a_enter = ms();
  if (nocold) {
    a_return = ms();
    b.join();
    CK(cudaStreamSynchronize(s_spin));
    printf("RESULT mode %s_nocold OK: B called %.1f returned %.1f, spinner done at %.1f\n", mode.c_str(),
           b_call.load(), b_return.load(), ms());
    fflush(stdout);
    _exit(0);
  }
#ifdef COLD_SO
  p6_launch_cold(s_cold, out);  // first launch: lazy load of the library's module
#else
  cold_kernel<<<1, 32, 0, s_cold>>>(out);  // first launch: lazy load of cold_kernel
#endif
  a_return = ms();
  b.join();
  CK(cudaDeviceSynchronize());
  int done_h = 0;
  CK(cudaMemcpy(&done_h, done, 4, cudaMemcpyDeviceToHost));
  printf("RESULT mode %s OK: A entered cold launch %.1f ms, returned %.1f ms; B called %.1f, returned %.1f; "
         "spinner released %s\n",
         mode.c_str(), a_enter.load(), a_return.load(), b_call.load(), b_return.load(), done_h ? "yes" : "no");
  fflush(stdout);
  _exit(0);
}
