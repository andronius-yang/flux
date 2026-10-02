// p12_barrier.cu -- handoff 53 probe P12: latency of nvshmemx_barrier_all_on_stream on Perlmutter (NVSHMEM 3.2.5,
// libfabric / CXI) at the node count of the allocation (one PE per GPU), idle GPUs, eager and inside a CUDA graph
// (the layer graphs' end-of-layer barrier), plus the same barrier after a skew injected on one PE (a spin kernel
// of --skew-us on PE 0 before its barrier) to separate barrier latency from waiting for the slowest PE.
// Output (PE 0): per mode the median / p90 time per barrier from CUDA events over --iters barriers in a row.
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e = (x);                                                               \
    if (e != cudaSuccess) {                                                            \
      fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
      exit(1);                                                                         \
    }                                                                                  \
  } while (0)

__global__ void spin_us(long long us) {
  long long t0 = clock64();
  // ~1.41 GHz A100 SM clock
  while (clock64() - t0 < us * 1410) {
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int iters = 200, skew_us = 100;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--iters") iters = atoi(argv[++i]);
    if (a == "--skew-us") skew_us = atoi(argv[++i]);
  }
  int wr, ws;
  MPI_Comm_rank(MPI_COMM_WORLD, &wr);
  MPI_Comm_size(MPI_COMM_WORLD, &ws);
  int ndev = 0;
  CK(cudaGetDeviceCount(&ndev));
  CK(cudaSetDevice(wr % ndev));
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  MPI_Comm comm = MPI_COMM_WORLD;
  attr.mpi_comm = &comm;
  if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr) != 0) {
    fprintf(stderr, "nvshmemx_init_attr failed\n");
    return 1;
  }
  const int me = nvshmem_my_pe(), npes = nvshmem_n_pes();
  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  auto report = [&](const char *mode, std::vector<float> &v) {
    std::sort(v.begin(), v.end());
    float med = v[v.size() / 2], p90 = v[v.size() * 9 / 10], mn = v[0];
    float g[3] = {med, p90, mn}, gm[3];
    MPI_Reduce(g, gm, 3, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (me == 0) {
      printf("P12 pes=%d mode=%s us_per_barrier med=%.1f p90=%.1f min=%.1f (max over PEs)\n", npes, mode, gm[0],
             gm[1], gm[2]);
      fflush(stdout);
    }
  };
  // warm-up
  for (int i = 0; i < 20; i++) nvshmemx_barrier_all_on_stream(s);
  CK(cudaStreamSynchronize(s));
  nvshmem_barrier_all();
  // eager: one barrier per timed interval (events around it; the stream otherwise idle)
  {
    std::vector<float> v;
    for (int i = 0; i < iters; i++) {
      CK(cudaEventRecord(e0, s));
      nvshmemx_barrier_all_on_stream(s);
      CK(cudaEventRecord(e1, s));
      CK(cudaEventSynchronize(e1));
      float ms;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      v.push_back(ms * 1000.f);
    }
    report("eager_single", v);
  }
  nvshmem_barrier_all();
  // eager back-to-back: 10 barriers per interval
  {
    std::vector<float> v;
    for (int i = 0; i < iters / 10 + 1; i++) {
      CK(cudaEventRecord(e0, s));
      for (int k = 0; k < 10; k++) nvshmemx_barrier_all_on_stream(s);
      CK(cudaEventRecord(e1, s));
      CK(cudaEventSynchronize(e1));
      float ms;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      v.push_back(ms * 100.f);
    }
    report("eager_b2b", v);
  }
  nvshmem_barrier_all();
  // graph: 10 barriers captured, replayed
  {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int k = 0; k < 10; k++) nvshmemx_barrier_all_on_stream(s);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphLaunch(ge, s));
    CK(cudaStreamSynchronize(s));
    nvshmem_barrier_all();
    std::vector<float> v;
    for (int i = 0; i < iters / 10 + 1; i++) {
      CK(cudaEventRecord(e0, s));
      CK(cudaGraphLaunch(ge, s));
      CK(cudaEventRecord(e1, s));
      CK(cudaEventSynchronize(e1));
      float ms;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      v.push_back(ms * 100.f);
    }
    report("graph_b2b", v);
  }
  nvshmem_barrier_all();
  // skew: PE 0 spins skew_us before its barrier; the others time their barrier (= skew + latency)
  {
    std::vector<float> v;
    for (int i = 0; i < iters; i++) {
      if (me == 0) spin_us<<<1, 1, 0, s>>>(skew_us);
      CK(cudaEventRecord(e0, s));
      nvshmemx_barrier_all_on_stream(s);
      CK(cudaEventRecord(e1, s));
      CK(cudaEventSynchronize(e1));
      float ms;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      if (me != 0) v.push_back(ms * 1000.f);
      MPI_Barrier(MPI_COMM_WORLD);
    }
    if (me == 0) v.push_back(0.f);
    report("skew_others", v);
  }
  nvshmem_barrier_all();
  nvshmem_finalize();
  MPI_Finalize();
  return 0;
}
