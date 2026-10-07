// Device dual3 swap lane: the kernels the research tree adds to the serving runtime's staged lane (case study,
// 2026-10-06). Loader and lane: serving_dual3.py.
//
// dual3 schedule on the device: W1 of every moved slot is pushed by its sender under the sender's dispatch GEMM
// (the existing lane_push kernel on a side stream behind the dispatch GEMM-start mark); W2 is PULLED by its receiver
// under the receiver's combine GEMM (pull_w2 below, on a side stream behind the combine GEMM-start mark), so the
// combine GEMM's tile gates depend only on the receiver's own progress (a sender-side W2 push after the sender's
// dispatch GEMM would couple the receiver's resident gated GEMM to the sender's data-dependent waits: the serving
// W2 cycle of 2026-10-01). The sender's W2 slot is overwritten only by its own commit after its combine GEMM, which
// first waits for every receiver's acknowledgment (wait_acks below).
//
// No stream waits and no event joins on side streams (2026-10-06, first attempt hung): the serving runtime has more
// streams than hardware queues, so a stream memop wait parked on a side stream can sit at the head of the queue
// that also carries the forward stream's later GEMM-start mark write (a deadlock). Instead the phase kernels are
// launched behind a fork of already-enqueued forward-stream work and spin on the mark themselves, and the joins are
// device words: the W1 commit first waits until this rank's pushes raised the receivers' gates (wait_pushed), the
// W2 commit waits on the gates its own pull raised (lane_commit) after the receivers' acknowledgments (wait_acks).
//
// The kernels may start beside a spinning GEMM block (rule R5): push / pull are 512 threads at <= 64 registers
// (launch bounds), the waits one warp. Every kernel returns at once when the decision block reports no swap.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define D3_CUDA_CHECK(expr)                                     \
  do {                                                          \
    cudaError_t _e = (expr);                                    \
    TORCH_CHECK(_e == cudaSuccess, "CUDA error: ",              \
                cudaGetErrorString(_e));                        \
  } while (0)

namespace d3 {

constexpr int kBlocks = 32;
constexpr int kThreads = 512;

// the swap_decide result block: [0] rounds, [1] total, [2] err, [3, 3+R) moves per rank, then R x cap x 4 moves
// (dst slot, src rank, src slot, expert) sorted by destination slot
__device__ __forceinline__ const long long *moves_of(const long long *blk, int R, int cap, int r) {
  return blk + 3 + R + (long long)r * cap * 4;
}

// receiver: W2 of every incoming slot, from the sender's slot (a node peer's symmetric slot storage) into this
// rank's W2 staging entry m (the entry the combine GEMM reads through the weight pointer override); the last
// block raises this rank's W2 gate words of the incoming slots and acknowledges each sender
__global__ void __launch_bounds__(kThreads, 2)
pull_w2_kernel(const long long *blk, int R, int L, int rank, int nlp, int cap, const long long *peer_w2,
               long long slot_bytes, char *stag, long long *gate, const long long *peer_ack, const long long *mark,
               long long epoch, unsigned int *counter) {
  if (blk[0] == 0) return;
  const int n = (int)blk[3 + rank];
  if (n == 0) return;
  if (threadIdx.x == 0) {                           // under the GEMM: wait for its start mark (one-shot epoch word)
    volatile const long long *mk = reinterpret_cast<volatile const long long *>(mark);
    while (*mk < epoch) __nanosleep(256);
  }
  __syncthreads();
  const long long *mv = moves_of(blk, R, cap, rank);
  const long long vpm = slot_bytes / 16;
  const long long total = (long long)n * vpm;
  for (long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x; v < total;
       v += (long long)gridDim.x * blockDim.x) {
    const int m = (int)(v / vpm);
    const long long o = v - (long long)m * vpm;
    const int sr = (int)mv[m * 4 + 1], ss = (int)mv[m * 4 + 2];
    const int4 *src = reinterpret_cast<const int4 *>(reinterpret_cast<const char *>(peer_w2[sr % L]) +
                                                     (long long)(1 + ss) * slot_bytes) + o;
    int4 *dst = reinterpret_cast<int4 *>(stag + (long long)m * slot_bytes) + o;
    *dst = *src;
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned int prev = atomicAdd(counter, 1u);
    if (prev == gridDim.x - 1) {                    // every block's copies are fenced: raise gates, acknowledge
      __threadfence_system();
      for (int m = 0; m < n; ++m) {
        const int dj = (int)mv[m * 4], sr = (int)mv[m * 4 + 1];
        reinterpret_cast<volatile long long *>(gate)[(nlp + 1) + 1 + dj] = epoch;     // matrix 1 gate word
        reinterpret_cast<volatile long long *>(peer_ack[sr % L])[(rank % L) * cap + m] = epoch;
      }
      __threadfence_system();
      *counter = 0u;
    }
  }
}

// sender, W1: every slot this rank gives to a node peer, into the peer's W1 staging entry (the receiver's move index),
// under the sender's dispatch GEMM (spin on its start mark); the last block raises the peers' W1 gate words
__global__ void __launch_bounds__(kThreads, 2)
push_w1_kernel(const long long *blk, int R, int L, int rank, int nlp, int cap, const char *slots, long long slot_bytes,
               const long long *peer_stag, const long long *peer_gate, const long long *mark, long long epoch,
               unsigned int *counter) {
  constexpr int kMaxOut = 64;
  __shared__ int s_n, s_ss[kMaxOut], s_dl[kMaxOut], s_idx[kMaxOut], s_dj[kMaxOut];
  if (blk[0] == 0) return;
  if (threadIdx.x == 0) {
    int n = 0;
    const int node = rank / L;
    for (int r = node * L; r < (node + 1) * L; ++r) {
      const int c = (int)blk[3 + r];
      const long long *mv = moves_of(blk, R, cap, r);
      for (int i = 0; i < c; ++i) {
        if ((int)mv[i * 4 + 1] == rank && n < kMaxOut) {
          s_dj[n] = (int)mv[i * 4];
          s_ss[n] = (int)mv[i * 4 + 2];
          s_dl[n] = r % L;
          s_idx[n] = i;
          ++n;
        }
      }
    }
    s_n = n;
  }
  __syncthreads();
  const int n = s_n;
  if (n == 0) return;
  if (threadIdx.x == 0) {                           // under the GEMM: wait for its start mark (one-shot epoch word)
    volatile const long long *mk = reinterpret_cast<volatile const long long *>(mark);
    while (*mk < epoch) __nanosleep(256);
  }
  __syncthreads();
  const long long vpm = slot_bytes / 16;
  const long long total = (long long)n * vpm;
  for (long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x; v < total;
       v += (long long)gridDim.x * blockDim.x) {
    const int m = (int)(v / vpm);
    const long long o = v - (long long)m * vpm;
    const int4 *src = reinterpret_cast<const int4 *>(slots + (long long)(1 + s_ss[m]) * slot_bytes) + o;
    int4 *dst = reinterpret_cast<int4 *>(reinterpret_cast<char *>(peer_stag[s_dl[m]]) +
                                         (long long)s_idx[m] * slot_bytes) + o;
    *dst = *src;
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned int prev = atomicAdd(counter, 1u);
    if (prev == gridDim.x - 1) {
      __threadfence_system();
      for (int m = 0; m < n; ++m)
        reinterpret_cast<volatile long long *>(peer_gate[s_dl[m]])[1 + s_dj[m]] = epoch;   // matrix 0 gate word
      __threadfence_system();
      *counter = 0u;
    }
  }
}

// sender, before its W1 commit overwrites the slots it gave away: wait until its pushes raised the receivers' gates
__global__ void wait_pushed_kernel(const long long *blk, int R, int L, int rank, int cap, const long long *peer_gate,
                                   long long epoch) {
  if (blk[0] == 0) return;
  const int t = threadIdx.x;
  if (t < L) {
    const int r = (rank / L) * L + t;
    const int c = (int)blk[3 + r];
    const long long *mv = moves_of(blk, R, cap, r);
    for (int i = 0; i < c; ++i) {
      if ((int)mv[i * 4 + 1] != rank) continue;
      volatile const long long *g = reinterpret_cast<volatile const long long *>(peer_gate[t]) + 1 + mv[i * 4];
      while (*g < epoch) __nanosleep(256);
    }
  }
  __syncwarp();
  __threadfence_system();
}

// sender: before its W2 commit overwrites the slots it gave away, wait until every receiver in the node has
// pulled them (ack[receiver local rank * cap + the receiver's move index] >= epoch)
__global__ void wait_acks_kernel(const long long *blk, int R, int L, int rank, int cap, const long long *ack,
                                 long long epoch) {
  if (blk[0] == 0) return;
  const int t = threadIdx.x;
  if (t < L) {
    const int r = (rank / L) * L + t;
    const int c = (int)blk[3 + r];
    const long long *mv = moves_of(blk, R, cap, r);
    for (int i = 0; i < c; ++i) {
      if ((int)mv[i * 4 + 1] != rank) continue;
      volatile const long long *a = reinterpret_cast<volatile const long long *>(ack) + (long long)t * cap + i;
      while (*a < epoch) __nanosleep(256);
    }
  }
  __syncwarp();
  __threadfence_system();
}

}  // namespace d3

void pull_w2(const torch::Tensor blk, int64_t R, int64_t L, int64_t rank, int64_t nlp, int64_t cap,
             const torch::Tensor peer_w2, int64_t slot_bytes, int64_t stag_addr, torch::Tensor gate,
             const torch::Tensor peer_ack, int64_t mark_addr, int64_t epoch, torch::Tensor counter) {
  TORCH_CHECK(blk.is_cuda() && blk.scalar_type() == torch::kLong, "pull_w2: blk int64 cuda");
  TORCH_CHECK(peer_w2.is_cuda() && peer_w2.numel() == L && peer_ack.is_cuda() && peer_ack.numel() == L, "pull_w2: [L]");
  TORCH_CHECK(slot_bytes % 16 == 0 && gate.scalar_type() == torch::kLong, "pull_w2: 16-byte slots, int64 gates");
  d3::pull_w2_kernel<<<d3::kBlocks, d3::kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const long long *>(blk.data_ptr<int64_t>()), (int)R, (int)L, (int)rank, (int)nlp, (int)cap,
      reinterpret_cast<const long long *>(peer_w2.data_ptr<int64_t>()), (long long)slot_bytes,
      reinterpret_cast<char *>(stag_addr), reinterpret_cast<long long *>(gate.data_ptr<int64_t>()),
      reinterpret_cast<const long long *>(peer_ack.data_ptr<int64_t>()), reinterpret_cast<const long long *>(mark_addr),
      (long long)epoch, reinterpret_cast<unsigned int *>(counter.data_ptr<int>()));
  D3_CUDA_CHECK(cudaGetLastError());
}

void wait_acks(const torch::Tensor blk, int64_t R, int64_t L, int64_t rank, int64_t cap, const torch::Tensor ack,
               int64_t epoch) {
  TORCH_CHECK(ack.is_cuda() && ack.scalar_type() == torch::kLong && ack.numel() >= L * cap, "wait_acks: ack [L*cap]");
  TORCH_CHECK(L <= 32, "wait_acks: one warp covers the node");
  d3::wait_acks_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const long long *>(blk.data_ptr<int64_t>()), (int)R, (int)L, (int)rank, (int)cap,
      reinterpret_cast<const long long *>(ack.data_ptr<int64_t>()), (long long)epoch);
  D3_CUDA_CHECK(cudaGetLastError());
}

void push_w1(const torch::Tensor blk, int64_t R, int64_t L, int64_t rank, int64_t nlp, int64_t cap,
             const torch::Tensor slots, int64_t slot_bytes, const torch::Tensor peer_stag, const torch::Tensor peer_gate,
             int64_t mark_addr, int64_t epoch, torch::Tensor counter) {
  TORCH_CHECK(slot_bytes % 16 == 0 && slots.is_contiguous(), "push_w1: 16-byte slots");
  TORCH_CHECK(peer_stag.numel() == L && peer_gate.numel() == L, "push_w1: [L] peer tables");
  d3::push_w1_kernel<<<d3::kBlocks, d3::kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const long long *>(blk.data_ptr<int64_t>()), (int)R, (int)L, (int)rank, (int)nlp, (int)cap,
      static_cast<const char *>(slots.data_ptr()), (long long)slot_bytes,
      reinterpret_cast<const long long *>(peer_stag.data_ptr<int64_t>()),
      reinterpret_cast<const long long *>(peer_gate.data_ptr<int64_t>()), reinterpret_cast<const long long *>(mark_addr),
      (long long)epoch, reinterpret_cast<unsigned int *>(counter.data_ptr<int>()));
  D3_CUDA_CHECK(cudaGetLastError());
}

void wait_pushed(const torch::Tensor blk, int64_t R, int64_t L, int64_t rank, int64_t cap, const torch::Tensor peer_gate,
                 int64_t epoch) {
  TORCH_CHECK(L <= 32 && peer_gate.numel() == L, "wait_pushed: one warp, [L] gates");
  d3::wait_pushed_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const long long *>(blk.data_ptr<int64_t>()), (int)R, (int)L, (int)rank, (int)cap,
      reinterpret_cast<const long long *>(peer_gate.data_ptr<int64_t>()), (long long)epoch);
  D3_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("push_w1", &push_w1, "sender-side W1 push under the sender's dispatch GEMM (dual3 phase 0)");
  m.def("wait_pushed", &wait_pushed, "sender-side wait until its W1 pushes raised the receivers' gates");
  m.def("pull_w2", &pull_w2, "receiver-side W2 pull of the incoming slots (dual3 phase 1)");
  m.def("wait_acks", &wait_acks, "sender-side wait for the receivers' W2 pull acknowledgments");
}
