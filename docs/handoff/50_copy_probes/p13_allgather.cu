// p13_allgather.cu -- handoff 53 probe P13 (E3): the layer's two small all-gathers (loads: G int32 per rank = 512 B;
// routing: 2 * S * K int32 per rank = 16 KiB at S = 256, K = 8) on Perlmutter at the allocation's node count, one
// PE / rank per GPU: NCCL all-gather (a communicator of all ranks, ring / LL as NCCL picks) vs NVSHMEM fcollect
// (nvshmemx_int32_fcollect_on_stream on NVSHMEM_TEAM_WORLD, symmetric buffers), eager back-to-back and inside a CUDA
// graph, payload checked every iteration (each rank writes rank * 1000003 + iteration).
// Output (rank 0): per (op, size, mode) the median / p90 time per collective (CUDA events over 10 in a row, max
// over ranks), and the mismatch count.
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
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
#define NK(x)                                                                         \
  do {                                                                                \
    ncclResult_t r = (x);                                                             \
    if (r != ncclSuccess) {                                                           \
      fprintf(stderr, "NCCL %s at %s:%d\n", ncclGetErrorString(r), __FILE__, __LINE__); \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

__global__ void fill(int *p, int n, int v) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = v;
}
__global__ void check(const int *p, int n, int npes, int iter, unsigned long long *bad) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * npes; i += gridDim.x * blockDim.x) {
    const int r = i / n;
    if (p[i] != r * 1000003 + iter) atomicAdd(bad, 1ULL);
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int iters = 200;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--iters") iters = atoi(argv[++i]);
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
  ncclUniqueId id;
  if (wr == 0) NK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t nc;
  NK(ncclCommInitRank(&nc, ws, id, wr));
  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  unsigned long long *bad;
  CK(cudaMalloc(&bad, sizeof(unsigned long long)));
  const int sizes[2] = {128, 4096};  // int32 elements per rank: 512 B (loads), 16 KiB (routing at S = 256)
  for (int si = 0; si < 2; si++) {
    const int n = sizes[si];
    int *src = (int *)nvshmem_malloc(sizeof(int) * n);
    int *dst = (int *)nvshmem_malloc(sizeof(int) * n * ws);
    for (int op = 0; op < 2; op++) {        // 0 NCCL, 1 NVSHMEM fcollect
      for (int mode = 0; mode < 2; mode++) {  // 0 eager, 1 graph
        auto one = [&]() {
          if (op == 0) {
            NK(ncclAllGather(src, dst, n, ncclInt32, nc, s));
          } else {
            nvshmemx_int32_fcollect_on_stream(NVSHMEM_TEAM_WORLD, dst, src, n, s);
          }
        };
        cudaGraphExec_t ge = nullptr;
        if (mode == 1) {
          cudaGraph_t g;
          CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
          for (int k = 0; k < 10; k++) one();
          CK(cudaStreamEndCapture(s, &g));
          CK(cudaGraphInstantiate(&ge, g, 0));
        }
        CK(cudaMemsetAsync(bad, 0, sizeof(unsigned long long), s));
        std::vector<float> t;
        for (int it = 0; it < iters / 10 + 3; it++) {
          fill<<<8, 256, 0, s>>>(src, n, wr * 1000003 + it);
          CK(cudaStreamSynchronize(s));
          MPI_Barrier(MPI_COMM_WORLD);
          CK(cudaEventRecord(e0, s));
          if (mode == 1) {
            CK(cudaGraphLaunch(ge, s));
          } else {
            for (int k = 0; k < 10; k++) one();
          }
          CK(cudaEventRecord(e1, s));
          check<<<16, 256, 0, s>>>(dst, n, ws, it, bad);
          CK(cudaEventSynchronize(e1));
          float ms;
          CK(cudaEventElapsedTime(&ms, e0, e1));
          if (it >= 3) t.push_back(ms * 100.f);  // us per collective
        }
        CK(cudaStreamSynchronize(s));
        unsigned long long hb;
        CK(cudaMemcpy(&hb, bad, sizeof(hb), cudaMemcpyDeviceToHost));
        std::sort(t.begin(), t.end());
        float g[2] = {t[t.size() / 2], t[t.size() * 9 / 10]}, gm[2];
        MPI_Reduce(g, gm, 2, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        unsigned long long tb;
        MPI_Reduce(&hb, &tb, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (wr == 0) {
          printf("P13 ranks=%d op=%s bytes_per_rank=%d mode=%s us_per_collective med=%.1f p90=%.1f mismatches=%llu\n", ws,
                 op == 0 ? "nccl_allgather" : "nvshmem_fcollect", n * 4, mode == 0 ? "eager" : "graph", gm[0], gm[1],
                 tb);
          fflush(stdout);
        }
        if (ge) CK(cudaGraphExecDestroy(ge));
      }
    }
    nvshmem_barrier_all();
    nvshmem_free(src);
    nvshmem_free(dst);
  }
  ncclCommDestroy(nc);
  nvshmem_finalize();
  MPI_Finalize();
  return 0;
}
