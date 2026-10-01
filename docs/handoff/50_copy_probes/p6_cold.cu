// p6_cold.cu -- the cold kernel of probe P6 in its own shared library: its first launch loads a whole module
// (library-level lazy loading, the case seen in the C2b gate hang), not just a function of a loaded module.
#include <cuda_runtime.h>
__global__ void cold_so_kernel(int *out) {
  if (threadIdx.x == 0) {
    out[1] = 43;
  }
}
extern "C" void p6_launch_cold(cudaStream_t s, int *out) { cold_so_kernel<<<1, 32, 0, s>>>(out); }
