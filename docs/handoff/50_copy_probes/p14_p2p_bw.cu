// p14_p2p_bw.cu -- handoff 53 probe P14: inter-node point-to-point bandwidth on the plan-9 wire's pairs (rank r ->
// the same local rank on the next node, every GPU of a node sending at once), NCCL send/recv vs NVSHMEM device
// put-with-signal (one blocking put by one thread, the wire warp's form) vs NVSHMEM host on-stream put, sizes
// 256 KiB .. 8 MiB. Times per transfer from CUDA events over 20 in a row (max over ranks), GB/s = bytes / time.
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

__global__ void dev_put(char *dst, const char *src, size_t bytes, uint64_t *sig, uint64_t v, int pe, int reps) {
  if (threadIdx.x == 0) {
    for (int k = 0; k < reps; k++) nvshmem_putmem_signal(dst, src, bytes, sig, v + k, NVSHMEM_SIGNAL_SET, pe);
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int wr, ws;
  MPI_Comm_rank(MPI_COMM_WORLD, &wr);
  MPI_Comm_size(MPI_COMM_WORLD, &ws);
  int ndev = 0;
  CK(cudaGetDeviceCount(&ndev));
  CK(cudaSetDevice(wr % ndev));
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  MPI_Comm comm = MPI_COMM_WORLD;
  attr.mpi_comm = &comm;
  if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr) != 0) return 1;
  ncclUniqueId id;
  if (wr == 0) NK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t nc;
  NK(ncclCommInitRank(&nc, ws, id, wr));
  const int L = ndev, nn = ws / L;
  const int to = (wr + L) % ws, from = (wr - L + ws) % ws;  // same local rank, next / previous node
  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  const size_t maxb = 8u << 20;
  char *sbuf = (char *)nvshmem_malloc(maxb), *rbuf = (char *)nvshmem_malloc(maxb);
  uint64_t *sig = (uint64_t *)nvshmem_calloc(1, sizeof(uint64_t));
  const int reps = 20;
  for (size_t b = 256u << 10; b <= maxb; b <<= 1) {
    for (int op = 0; op < 3; op++) {
      std::vector<float> t;
      for (int it = 0; it < 6; it++) {
        CK(cudaStreamSynchronize(s));
        MPI_Barrier(MPI_COMM_WORLD);
        CK(cudaEventRecord(e0, s));
        if (op == 0) {
          for (int k = 0; k < reps; k++) {
            NK(ncclGroupStart());
            NK(ncclSend(sbuf, b, ncclChar, to, nc, s));
            NK(ncclRecv(rbuf, b, ncclChar, from, nc, s));
            NK(ncclGroupEnd());
          }
        } else if (op == 1) {
          dev_put<<<1, 32, 0, s>>>(rbuf, sbuf, b, sig, 1, to, reps);
        } else {
          for (int k = 0; k < reps; k++) nvshmemx_putmem_signal_on_stream(rbuf, sbuf, b, sig, k + 1, NVSHMEM_SIGNAL_SET, to, s);
        }
        CK(cudaEventRecord(e1, s));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        if (it >= 1) t.push_back(ms * 1000.f / reps);
        nvshmemx_barrier_all_on_stream(s);
      }
      std::sort(t.begin(), t.end());
      float g = t[t.size() / 2], gm;
      MPI_Reduce(&g, &gm, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
      if (wr == 0) {
        printf("P14 ranks=%d nodes=%d op=%s bytes=%zu us_per_transfer=%.1f GB/s=%.2f\n", ws, nn,
               op == 0 ? "nccl_sendrecv" : (op == 1 ? "nvshmem_dev_put_signal" : "nvshmem_host_put_signal"), b, gm,
               b / (gm * 1e3));
        fflush(stdout);
      }
    }
  }
  ncclCommDestroy(nc);
  nvshmem_finalize();
  MPI_Finalize();
  return 0;
}
