// cores.cu: can a 1024-thread block co-run with a resident "busy" grid? (single GPU, quick check)
#include <cuda_runtime.h>
#include <cstdio>
__global__ void __launch_bounds__(1024, 2) busy(volatile int* f) { float a = threadIdx.x; while (*f == 0) { for (int j = 0; j < 4096; ++j) a = fmaf(a, 1.0001f, 0.5f); } if (a == 1.f) f[1] = 1; }
__global__ void __launch_bounds__(1024, 2) work(float* p, int n) { for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] += 1.f; }
int main() {
  int* f; cudaHostAlloc(&f, 8, cudaHostAllocMapped); f[0] = 0; int* fd; cudaHostGetDevicePointer(&fd, f, 0);
  float* p; cudaMalloc(&p, 1 << 24);
  cudaStream_t sb, sw; cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking); cudaStreamCreateWithFlags(&sw, cudaStreamNonBlocking);
  cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, busy); printf("busy regs=%d\n", fa.numRegs); cudaFuncGetAttributes(&fa, work); printf("work regs=%d\n", fa.numRegs);
  int bps; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, busy, 1024, 0); printf("busy blocks/SM=%d\n", bps);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  int grids[] = {0, 108, 212, 214, 216};
  int tb[] = {1024, 256};
  for (int g : grids) for (int t : tb) {
    f[0] = 0;
    if (g) { busy<<<g, 1024, 0, sb>>>(fd); }
    { struct timespec ts = {0, 20000000}; nanosleep(&ts, 0); }
    cudaEventRecord(e0, sw); work<<<4, t, 0, sw>>>(p, 1 << 20); cudaEventRecord(e1, sw);
    // wait up to 2 s
    cudaError_t q; double waited = 0; while ((q = cudaEventQuery(e1)) == cudaErrorNotReady && waited < 2.0) { waited += 0.001; struct timespec ts = {0, 1000000}; nanosleep(&ts, 0); }
    float ms = -1; if (q == cudaSuccess) cudaEventElapsedTime(&ms, e0, e1);
    printf("busy grid %3d, work block %4d: %s (%.3f ms)\n", g, t, q == cudaSuccess ? "RAN" : "STARVED", ms);
    f[0] = 1; cudaDeviceSynchronize();
  }
  return 0;
}
