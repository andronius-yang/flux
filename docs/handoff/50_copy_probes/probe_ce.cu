// probe_ce.cu -- handoff 50 stage 0 (plan 7): are intra-node copies issued without the host main thread
// independent of the SMs?  One process per GPU (4 per Perlmutter node), MPI bootstrap, CUDA-IPC peer
// mappings (what NVSHMEM uses intra-node with NVSHMEM_DISABLE_CUDA_VMM=1) and, with -DUSE_NVSHMEM,
// NVSHMEM symmetric buffers.  Every rank copies to peer (rank+1)%n; rank 0 also fans out to 3 peers.
//
//   P0  host cudaMemcpyAsync (copy-engine baseline) + SM copy kernel with k blocks
//   P1  device-launched graph (cudaGraphLaunch from a kernel, fire-and-forget) holding one peer memcpy node
//   P2  host-launched graph: planning kernel sets a conditional (IF chain / SWITCH) choosing a size bucket
//   P3  host proxy thread: a kernel writes descriptors to pinned memory, a CPU thread issues the copies
//       (loop of cudaMemcpyAsync, and cudaMemcpyBatchAsync)
//   P4  device-side cudaMemcpyAsync (CDP2)
//   P5  cuStreamWriteValue64 to a peer-mapped signal word after the data copy; receiver kernel acquires
//       the signal and verifies the payload (payload = epoch, changes every iteration); fallback = 8-byte memcpy
//   busy = none | spin | stream : a persistent kernel occupying every SM slot but two (spin: ALU loop;
//       stream: HBM read-modify-write over 1 GB) runs during the measurement.
// Output: CSV rows  rank,probe,mech,buf,busy,bytes,n,metric,median_us,p90_us,min_us  (plus INFO lines).
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <sched.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#ifdef USE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

static int g_rank = -1, g_size = 0, g_dev = 0, g_ndev = 0;
#define CK(x)                                                                                      \
  do {                                                                                             \
    cudaError_t e_ = (x);                                                                          \
    if (e_ != cudaSuccess) {                                                                       \
      fprintf(stderr, "[r%d] CUDA error %s:%d %s -> %s\n", g_rank, __FILE__, __LINE__, #x,         \
              cudaGetErrorString(e_));                                                             \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                                \
    }                                                                                              \
  } while (0)

static const size_t kSizes[] = {524288, 1048576, 2516480, 16777216};  // 0.5 / 1 / 2.4 / 16 MB
static const size_t kRegion = 32u << 20;                               // per-source inbox region
static const int kMaxPeers = 8;

// ---------------------------------------------------------------- device code
__device__ __forceinline__ uint64_t gtimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__global__ void stamp_kernel(uint64_t* slot) { *slot = gtimer(); }
__global__ void fill_kernel(uint64_t* p, size_t n, uint64_t v) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = v;
}
__global__ void __launch_bounds__(1024, 2) copy_kernel(const int4* __restrict__ src, int4* __restrict__ dst, size_t n16) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) dst[i] = src[i];
}
__global__ void __launch_bounds__(1024, 2) busy_kernel(volatile int* flag, int mode, int4* buf, size_t n16) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
  float a = tid, b = 1.0001f;
  while (__shfl_sync(0xffffffffu, (threadIdx.x & 31) == 0 ? *flag : 0, 0) == 0) {
    if (mode == 0) {
      // GEMM-like occupancy: every slot held, issue slots mostly free; each step is a dependent
      // L2 load (one hot line, no HBM traffic) followed by a short FMA chain. A pure FMA spin
      // starves co-resident warps outright (oldest-first issue, a busy warp is always ready).
      // (hammering one L2 line from every warp, or a pure FMA chain, both starve co-resident
      // warps; a short FMA burst + nanosleep holds every slot like a memory-stalled GEMM)
      for (int j = 0; j < 64; ++j) {
        for (int k = 0; k < 64; ++k) a = fmaf(a, b, 0.5f);
        __nanosleep(2000);
      }
    } else {
      for (size_t j = tid; j < n16; j += stride) {
        int4 v = buf[j];
        v.x += 1;
        buf[j] = v;
      }
    }
  }
  if (a == 12345.f) buf[0].x = 1;  // keep the ALU loop alive
}
// P1: launch a device graph from a kernel
__global__ void launcher_kernel(cudaGraphExec_t ge, uint64_t* slotL, int* err) {
  *slotL = gtimer();
  cudaError_t e = cudaGraphLaunch(ge, cudaStreamGraphFireAndForget);
  if (e != cudaSuccess) *err = (int)e;
}
// P2: planning kernel sets the conditionals
__global__ void plan_if_kernel(cudaGraphConditionalHandle h0, cudaGraphConditionalHandle h1,
                               cudaGraphConditionalHandle h2, cudaGraphConditionalHandle h3, int nb,
                               const int* chosen, uint64_t* slotP) {
  int c = *chosen;
  *slotP = gtimer();
  cudaGraphSetConditional(h0, c == 0);
  if (nb > 1) cudaGraphSetConditional(h1, c == 1);
  if (nb > 2) cudaGraphSetConditional(h2, c == 2);
  if (nb > 3) cudaGraphSetConditional(h3, c == 3);
}
__global__ void plan_switch_kernel(cudaGraphConditionalHandle h, const int* chosen, uint64_t* slotP) {
  int c = *chosen;
  *slotP = gtimer();
  cudaGraphSetConditional(h, (unsigned)c);
}
// P3: descriptor ring in pinned mapped memory
struct Desc {
  volatile uint64_t epoch;
  int n;
  int bytes[64];
  int src_off[64];
  int dst_off[64];
  int peer[64];
  uint64_t t_flag;
};
__global__ void desc_kernel(Desc* d, uint64_t epoch, int n, int bytes, int peer_mode) {
  d->n = n;
  for (int i = 0; i < n; ++i) {
    d->bytes[i] = bytes;
    d->src_off[i] = i * bytes;
    d->dst_off[i] = i * bytes;
    d->peer[i] = peer_mode < 0 ? (i % 3) : peer_mode;  // peer_mode<0: spread over 3 peers
  }
  __threadfence_system();
  d->t_flag = gtimer();
  __threadfence_system();
  d->epoch = epoch;
  __threadfence_system();
}
// P4: CDP2 device-side memcpy
__global__ void cdp_copy_kernel(void* dst, const void* src, size_t bytes, uint64_t* slotL, uint64_t* slotE, int* err) {
  *slotL = gtimer();
  cudaError_t e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, cudaStreamTailLaunch);
  if (e != cudaSuccess) *err = (int)e;
  stamp_kernel<<<1, 1, 0, cudaStreamTailLaunch>>>(slotE);
}
// P5: receiver acquires the signal, then verifies the payload (every word == epoch)
__device__ __forceinline__ uint64_t ld_acquire_sys(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__global__ void wait_verify_kernel(const uint64_t* sig, uint64_t epoch, const uint64_t* data, size_t nwords, uint64_t* out) {
  __shared__ int timed_out;
  if (threadIdx.x == 0) {
    timed_out = 0;
    uint64_t t0 = gtimer();
    while (ld_acquire_sys(sig) < epoch) {
      if (gtimer() - t0 > 3000000000ull) { timed_out = 1; break; }
    }
    out[2] = gtimer();
  }
  __syncthreads();
  if (timed_out) { if (threadIdx.x == 0) out[0] = 1; return; }
  unsigned bad = 0;
  for (size_t i = threadIdx.x; i < nwords; i += blockDim.x) if (ld_acquire_sys(data + i) != epoch) ++bad;
  atomicAdd((unsigned*)&out[1], bad);
  if (threadIdx.x == 0) out[0] = 0;
}

// ---------------------------------------------------------------- host state
struct Buffers {
  const char* name;
  uint64_t* src;                 // local source (kRegion bytes)
  uint64_t* inbox;               // local inbox, g_size regions of kRegion
  uint64_t* sig;                 // local signal words (64)
  uint64_t* peer_inbox[kMaxPeers];  // peer p's inbox region reserved for me
  uint64_t* peer_sig[kMaxPeers];    // peer p's signal word for me
};
static Buffers g_ipc, g_nvs;
static uint64_t* g_stamps = nullptr;  // mapped pinned, 256 slots
static uint64_t* g_stamps_d = nullptr;
static int* g_err_d = nullptr;
static int* g_err_h = nullptr;
static cudaStream_t g_s, g_busy_s;
static int* g_busy_flag_d = nullptr;  // device memory: a host-mapped flag polled by every thread clogs the memory system
static cudaStream_t g_ctl;
static int4* g_busy_buf = nullptr;
static int g_num_sms = 0, g_blocks_per_sm = 0;
static std::string g_busy = "none";
static int g_iters = 25;
static int g_free = 4;      // block slots left free by the busy kernel (issue kernels need one)
static int g_onepersm = 0;  // busy grid = one 1024-thread block per SM (all SMs busy, half the slots)
static std::string g_only = "";
static bool g_quiet = false;  // warm-up pass: no rows

static bool want(const char* p) { return g_only.empty() || g_only.find(p) != std::string::npos; }
static double us(uint64_t a, uint64_t b) { return (double)(b - a) / 1000.0; }
struct Stat { double med, p90, mn; };
static Stat stat(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  Stat s;
  s.med = v[v.size() / 2];
  s.p90 = v[std::min(v.size() - 1, (size_t)(v.size() * 9 / 10))];
  s.mn = v[0];
  return s;
}
static void row(const char* probe, const char* mech, const char* buf, size_t bytes, int n, const char* metric, const std::vector<double>& v) {
  if (v.empty() || g_quiet) return;
  Stat s = stat(v);
  printf("ROW,%d,%s,%s,%s,%s,%zu,%d,%s,%.2f,%.2f,%.2f\n", g_rank, probe, mech, buf, g_busy.c_str(), bytes, n, metric, s.med, s.p90, s.mn);
}
static void info(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  printf("INFO,%d,", g_rank);
  vprintf(fmt, ap);
  printf("\n");
  va_end(ap);
}

static void busy_start() {
  if (g_busy == "none") return;
  CK(cudaMemsetAsync(g_busy_flag_d, 0, sizeof(int), g_ctl));
  CK(cudaStreamSynchronize(g_ctl));
  int mode = g_busy.rfind("spin", 0) == 0 ? 0 : 1;  // the label carries a suffix
  int grid = g_onepersm ? g_num_sms : g_blocks_per_sm * g_num_sms - g_free;
  busy_kernel<<<grid, 1024, 0, g_busy_s>>>(g_busy_flag_d, mode, g_busy_buf, (size_t)(1u << 30) / 16);
  CK(cudaGetLastError());
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
}
static void busy_stop() {
  if (g_busy == "none") return;
  static int one = 1;
  CK(cudaMemcpyAsync(g_busy_flag_d, &one, sizeof(int), cudaMemcpyHostToDevice, g_ctl));
  CK(cudaStreamSynchronize(g_ctl));
  CK(cudaStreamSynchronize(g_busy_s));
}

// ------------------------------------------------------------ P0 host memcpy / SM copy
static void p0(const Buffers& B, int peer) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (size_t bytes : kSizes) {
    std::vector<double> gpu, host;
    for (int it = 0; it < g_iters + 3; ++it) {
      auto h0 = std::chrono::steady_clock::now();
      CK(cudaEventRecord(e0, g_s));
      CK(cudaMemcpyAsync(B.peer_inbox[peer], B.src, bytes, cudaMemcpyDefault, g_s));
      CK(cudaEventRecord(e1, g_s));
      auto h1 = std::chrono::steady_clock::now();
      CK(cudaStreamSynchronize(g_s));
      float ms;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      if (it >= 3) { gpu.push_back(ms * 1000.0); host.push_back(std::chrono::duration<double, std::micro>(h1 - h0).count()); }
    }
    row("P0", "host_memcpy", B.name, bytes, 1, "gpu_us", gpu);
    row("P0", "host_memcpy", B.name, bytes, 1, "issue_us", host);
    Stat s = stat(gpu);
    row("P0", "host_memcpy", B.name, bytes, 1, "GBps", {bytes / s.med / 1e3});
    // SM copy kernel with k blocks
    for (int k : {1, 2, 4, 8, 16, 32}) {
      std::vector<double> g2;
      for (int it = 0; it < g_iters + 3; ++it) {
        CK(cudaEventRecord(e0, g_s));
        copy_kernel<<<k, 1024, 0, g_s>>>((const int4*)B.src, (int4*)B.peer_inbox[peer], bytes / 16);
        CK(cudaEventRecord(e1, g_s));
        CK(cudaStreamSynchronize(g_s));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        if (it >= 3) g2.push_back(ms * 1000.0);
      }
      row("P0", "sm_copy", B.name, bytes, k, "gpu_us", g2);
      Stat s2 = stat(g2);
      row("P0", "sm_copy", B.name, bytes, k, "GBps", {bytes / s2.med / 1e3});
    }
  }
  // concurrency: rank 0 to 3 peers at once on 3 streams, 16 MB each
  if (g_rank == 0 && g_size >= 4) {
    cudaStream_t ss[3];
    cudaEvent_t ea[3], eb[3];
    for (int i = 0; i < 3; ++i) { CK(cudaStreamCreateWithFlags(&ss[i], cudaStreamNonBlocking)); CK(cudaEventCreate(&ea[i])); CK(cudaEventCreate(&eb[i])); }
    size_t bytes = 16777216;
    std::vector<double> agg, single;
    for (int it = 0; it < g_iters + 3; ++it) {
      cudaEvent_t start;
      CK(cudaEventCreate(&start));
      CK(cudaEventRecord(start, g_s));
      for (int i = 0; i < 3; ++i) { CK(cudaStreamWaitEvent(ss[i], start, 0)); CK(cudaEventRecord(ea[i], ss[i])); CK(cudaMemcpyAsync(B.peer_inbox[i + 1], B.src, bytes, cudaMemcpyDefault, ss[i])); CK(cudaEventRecord(eb[i], ss[i])); }
      for (int i = 0; i < 3; ++i) CK(cudaStreamSynchronize(ss[i]));
      float mx = 0, ms;
      for (int i = 0; i < 3; ++i) { CK(cudaEventElapsedTime(&ms, start, eb[i])); mx = std::max(mx, ms); CK(cudaEventElapsedTime(&ms, ea[i], eb[i])); if (it >= 3) single.push_back(ms * 1000.0); }
      if (it >= 3) agg.push_back(mx * 1000.0);
      CK(cudaEventDestroy(start));
    }
    row("P0", "conc3_memcpy", B.name, bytes, 3, "gpu_us_all3", agg);
    row("P0", "conc3_memcpy", B.name, bytes, 3, "gpu_us_each", single);
    Stat s = stat(agg);
    row("P0", "conc3_memcpy", B.name, bytes, 3, "GBps_aggregate", {3.0 * bytes / s.med / 1e3});
  }
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
}

// ------------------------------------------------------------ P1 device-launched graph
static void p1(const Buffers& B, int peer) {
  for (size_t bytes : kSizes) {
    cudaGraph_t g;
    CK(cudaGraphCreate(&g, 0));
    cudaGraphNode_t nS, nM, nE;
    uint64_t* slotS = g_stamps_d + 1;
    uint64_t* slotE = g_stamps_d + 2;
    uint64_t* slotL = g_stamps_d + 0;
    cudaKernelNodeParams kp = {};
    kp.func = (void*)stamp_kernel;
    kp.gridDim = dim3(1);
    kp.blockDim = dim3(1);
    void* argsS[] = {&slotS};
    kp.kernelParams = argsS;
    CK(cudaGraphAddKernelNode(&nS, g, nullptr, 0, &kp));
    CK(cudaGraphAddMemcpyNode1D(&nM, g, &nS, 1, B.peer_inbox[peer], B.src, bytes, cudaMemcpyDefault));
    void* argsE[] = {&slotE};
    kp.kernelParams = argsE;
    CK(cudaGraphAddKernelNode(&nE, g, &nM, 1, &kp));
    cudaGraphExec_t ge;
    cudaError_t ie = cudaGraphInstantiateWithFlags(&ge, g, cudaGraphInstantiateFlagDeviceLaunch);
    if (ie != cudaSuccess) { info("P1 instantiate(DeviceLaunch) failed: %s", cudaGetErrorString(ie)); cudaGetLastError(); return; }
    CK(cudaGraphUpload(ge, g_s));
    CK(cudaStreamSynchronize(g_s));
    // reference: the same graph launched from the host
    cudaGraphExec_t gh;
    CK(cudaGraphInstantiate(&gh, g, 0));
    std::vector<double> l2s, s2e, l2e, hl2s, hs2e, host_issue;
    int nerr = 0;
    for (int it = 0; it < g_iters + 3; ++it) {
      g_stamps[0] = g_stamps[1] = g_stamps[2] = 0;
      *((volatile int*)g_err_h) = 0;  // mapped
      launcher_kernel<<<1, 1, 0, g_s>>>(ge, slotL, g_err_d);
      { cudaError_t le = cudaGetLastError(); if (le != cudaSuccess) { info("P1 launcher kernel launch failed: %s (device graph launch unsupported here)", cudaGetErrorString(le)); cudaDeviceSynchronize(); cudaGetLastError(); CK(cudaGraphExecDestroy(ge)); CK(cudaGraphExecDestroy(gh)); CK(cudaGraphDestroy(g)); return; } }
      auto t0 = std::chrono::steady_clock::now();
      while (*((volatile uint64_t*)g_stamps + 2) == 0) {
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > 5.0) break;
      }
      CK(cudaStreamSynchronize(g_s));
      CK(cudaDeviceSynchronize());
      if (g_stamps[2] == 0) { ++nerr; continue; }
      if (it >= 3) { l2s.push_back(us(g_stamps[0], g_stamps[1])); s2e.push_back(us(g_stamps[1], g_stamps[2])); l2e.push_back(us(g_stamps[0], g_stamps[2])); }
      // host-launched reference with stamps around
      g_stamps[0] = g_stamps[1] = g_stamps[2] = 0;
      auto h0 = std::chrono::steady_clock::now();
      stamp_kernel<<<1, 1, 0, g_s>>>(slotL);
      CK(cudaGraphLaunch(gh, g_s));
      auto h1 = std::chrono::steady_clock::now();
      CK(cudaStreamSynchronize(g_s));
      if (it >= 3) { hl2s.push_back(us(g_stamps[0], g_stamps[1])); hs2e.push_back(us(g_stamps[1], g_stamps[2])); host_issue.push_back(std::chrono::duration<double, std::micro>(h1 - h0).count()); }
    }
    int derr = *((volatile int*)g_err_h);
    if (derr) info("P1 device cudaGraphLaunch error code %d (%s)", derr, cudaGetErrorString((cudaError_t)derr));
    if (nerr) info("P1 bytes=%zu: %d iterations never completed (stamp E stayed 0)", bytes, nerr);
    row("P1", "device_graph", B.name, bytes, 1, "launch_to_copystart_us", l2s);
    row("P1", "device_graph", B.name, bytes, 1, "copystart_to_end_us", s2e);
    row("P1", "device_graph", B.name, bytes, 1, "launch_to_end_us", l2e);
    if (!s2e.empty()) { Stat s = stat(s2e); row("P1", "device_graph", B.name, bytes, 1, "GBps", {bytes / s.med / 1e3}); }
    row("P1", "host_graph", B.name, bytes, 1, "launch_to_copystart_us", hl2s);
    row("P1", "host_graph", B.name, bytes, 1, "copystart_to_end_us", hs2e);
    row("P1", "host_graph", B.name, bytes, 1, "issue_us", host_issue);
    CK(cudaGraphExecDestroy(ge));
    CK(cudaGraphExecDestroy(gh));
    CK(cudaGraphDestroy(g));
  }
}

// ------------------------------------------------------------ P2 conditional nodes (size buckets)
static void p2(const Buffers& B, int peer) {
  int* chosen_d;
  CK(cudaMalloc(&chosen_d, sizeof(int)));
  uint64_t* slotP = g_stamps_d + 10;
  uint64_t* slotS = g_stamps_d + 11;
  uint64_t* slotE = g_stamps_d + 12;
  for (int mode = 0; mode < 2; ++mode) {  // 0 = IF chain, 1 = SWITCH
    for (int nb : {1, 4}) {
      if (mode == 1 && nb == 1) continue;
      cudaGraph_t g;
      CK(cudaGraphCreate(&g, 0));
      cudaGraphConditionalHandle h[4] = {};
      int nh = mode == 0 ? nb : 1;
      for (int i = 0; i < nh; ++i) {
        cudaError_t e = cudaGraphConditionalHandleCreate(&h[i], g, 0, cudaGraphCondAssignDefault);
        if (e != cudaSuccess) { info("P2 cudaGraphConditionalHandleCreate failed: %s", cudaGetErrorString(e)); cudaGetLastError(); return; }
      }
      cudaGraphNode_t nPlan;
      cudaKernelNodeParams kp = {};
      kp.gridDim = dim3(1);
      kp.blockDim = dim3(1);
      void* argsIf[] = {&h[0], &h[1], &h[2], &h[3], &nb, &chosen_d, &slotP};
      void* argsSw[] = {&h[0], &chosen_d, &slotP};
      if (mode == 0) { kp.func = (void*)plan_if_kernel; kp.kernelParams = argsIf; } else { kp.func = (void*)plan_switch_kernel; kp.kernelParams = argsSw; }
      CK(cudaGraphAddKernelNode(&nPlan, g, nullptr, 0, &kp));
      // bodies: bucket b copies kSizes[b] (b < 4)
      auto add_body = [&](cudaGraph_t body, size_t bytes) {
        cudaGraphNode_t bS, bM, bE;
        cudaKernelNodeParams k2 = {};
        k2.func = (void*)stamp_kernel;
        k2.gridDim = dim3(1);
        k2.blockDim = dim3(1);
        void* aS[] = {&slotS};
        k2.kernelParams = aS;
        CK(cudaGraphAddKernelNode(&bS, body, nullptr, 0, &k2));
        CK(cudaGraphAddMemcpyNode1D(&bM, body, &bS, 1, B.peer_inbox[peer], B.src, bytes, cudaMemcpyDefault));
        void* aE[] = {&slotE};
        k2.kernelParams = aE;
        CK(cudaGraphAddKernelNode(&bE, body, &bM, 1, &k2));
      };
      if (mode == 0) {
        for (int b = 0; b < nb; ++b) {
          cudaGraphNodeParams cp = {};
          cp.type = cudaGraphNodeTypeConditional;
          cp.conditional.handle = h[b];
          cp.conditional.type = cudaGraphCondTypeIf;
          cp.conditional.size = 1;
          cudaGraphNode_t nC;
          cudaError_t e = cudaGraphAddNode(&nC, g, &nPlan, 1, &cp);
          if (e != cudaSuccess) { info("P2 IF node add failed: %s", cudaGetErrorString(e)); cudaGetLastError(); return; }
          add_body(cp.conditional.phGraph_out[0], kSizes[b]);
        }
      } else {
        cudaGraphNodeParams cp = {};
        cp.type = cudaGraphNodeTypeConditional;
        cp.conditional.handle = h[0];
        cp.conditional.type = cudaGraphCondTypeSwitch;
        cp.conditional.size = nb;
        cudaGraphNode_t nC;
        cudaError_t e = cudaGraphAddNode(&nC, g, &nPlan, 1, &cp);
        if (e != cudaSuccess) { info("P2 SWITCH node add failed: %s", cudaGetErrorString(e)); cudaGetLastError(); return; }
        for (int b = 0; b < nb; ++b) add_body(cp.conditional.phGraph_out[b], kSizes[b]);
      }
      cudaGraphExec_t ge;
      cudaError_t ie = cudaGraphInstantiate(&ge, g, 0);
      if (ie != cudaSuccess) { info("P2 instantiate failed: %s", cudaGetErrorString(ie)); cudaGetLastError(); return; }
      const char* mech = mode == 0 ? "if_chain" : "switch";
      for (int b = 0; b < nb; ++b) {
        std::vector<double> p2s, s2e, issue;
        for (int it = 0; it < g_iters + 3; ++it) {
          g_stamps[10] = g_stamps[11] = g_stamps[12] = 0;
          CK(cudaMemcpyAsync(chosen_d, &b, sizeof(int), cudaMemcpyHostToDevice, g_s));
          auto h0 = std::chrono::steady_clock::now();
          CK(cudaGraphLaunch(ge, g_s));
          auto h1 = std::chrono::steady_clock::now();
          CK(cudaStreamSynchronize(g_s));
          if (g_stamps[12] == 0) { info("P2 %s nb=%d bucket %d: body did not run", mech, nb, b); break; }
          if (it >= 3) { p2s.push_back(us(g_stamps[10], g_stamps[11])); s2e.push_back(us(g_stamps[11], g_stamps[12])); issue.push_back(std::chrono::duration<double, std::micro>(h1 - h0).count()); }
        }
        row("P2", mech, B.name, kSizes[b], nb, "cond_to_copystart_us", p2s);
        row("P2", mech, B.name, kSizes[b], nb, "copystart_to_end_us", s2e);
        row("P2", mech, B.name, kSizes[b], nb, "issue_us", issue);
      }
      CK(cudaGraphExecDestroy(ge));
      CK(cudaGraphDestroy(g));
    }
  }
  CK(cudaFree(chosen_d));
}

// ------------------------------------------------------------ P3 host proxy thread
struct Proxy {
  Desc* ring;         // mapped pinned
  Desc* ring_d;
  cudaStream_t ps;
  std::thread th;
  std::atomic<int> stop{0};
  std::atomic<uint64_t> expect{0};
  std::atomic<int> done{0};
  int use_batch = 0;
  const Buffers* B = nullptr;
  double t_detect = 0, t_issued = 0;  // steady_clock us
  uint64_t* slotE;
  int err = 0;
};
static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void proxy_main(Proxy* P) {
  CK(cudaSetDevice(g_dev));
  cpu_set_t cs;
  CPU_ZERO(&cs);
  sched_getaffinity(0, sizeof(cs), &cs);
  int last = -1;
  for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &cs)) last = c;
  if (last >= 0) { cpu_set_t one; CPU_ZERO(&one); CPU_SET(last, &one); sched_setaffinity(0, sizeof(one), &one); }
  while (!P->stop.load()) {
    uint64_t want = P->expect.load();
    if (want == 0 || P->ring->epoch != want) { continue; }
    P->t_detect = now_us();
    int n = P->ring->n;
    const Buffers& B = *P->B;
    if (P->use_batch) {
#if CUDART_VERSION >= 12080
      void* dsts[64]; void* srcs[64]; size_t sizes[64];
      for (int i = 0; i < n; ++i) { dsts[i] = (char*)B.peer_inbox[P->ring->peer[i]] + P->ring->dst_off[i]; srcs[i] = (char*)B.src + P->ring->src_off[i]; sizes[i] = P->ring->bytes[i]; }
      cudaMemcpyAttributes at = {};
      at.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
      size_t idx = 0, fail = 0;
      cudaError_t e = cudaMemcpyBatchAsync(dsts, srcs, sizes, n, &at, &idx, 1, &fail, P->ps);
      if (e != cudaSuccess) { P->err = (int)e; cudaGetLastError(); }
#else
      P->err = -1;
#endif
    } else {
      for (int i = 0; i < n; ++i)
        CK(cudaMemcpyAsync((char*)B.peer_inbox[P->ring->peer[i]] + P->ring->dst_off[i], (char*)B.src + P->ring->src_off[i], P->ring->bytes[i], cudaMemcpyDefault, P->ps));
    }
    stamp_kernel<<<1, 1, 0, P->ps>>>(P->slotE);
    P->t_issued = now_us();
    P->expect.store(0);
    P->done.store(1);
  }
}
static void p3(const Buffers& B, int peer) {
  // CPU<->GPU clock offset (GPU globaltimer ns - CPU steady ns), +- a launch latency
  std::vector<double> offs;
  for (int i = 0; i < 11; ++i) {
    double h0 = now_us();
    stamp_kernel<<<1, 1, 0, g_s>>>(g_stamps_d + 20);
    CK(cudaStreamSynchronize(g_s));
    offs.push_back(g_stamps[20] / 1000.0 - h0);
  }
  double off = stat(offs).med;
  Proxy P;
  CK(cudaHostAlloc(&P.ring, sizeof(Desc), cudaHostAllocMapped));
  memset(P.ring, 0, sizeof(Desc));
  CK(cudaHostGetDevicePointer(&P.ring_d, P.ring, 0));
  CK(cudaStreamCreateWithFlags(&P.ps, cudaStreamNonBlocking));
  P.B = &B;
  P.slotE = g_stamps_d + 21;
  P.th = std::thread(proxy_main, &P);
  uint64_t epoch = 1;
  for (int use_batch : {0, 1}) {
    P.use_batch = use_batch;
    const char* mech = use_batch ? "proxy_batch" : "proxy_loop";
    for (int n : {1, 12, 37}) {
      for (size_t bytes : {kSizes[0], kSizes[2]}) {
        if ((size_t)n * bytes > kRegion) continue;
        std::vector<double> flag2E, detect, issue, flag2E_dev;
        int errs = 0;
        for (int it = 0; it < g_iters + 3; ++it) {
          g_stamps[21] = 0;
          P.done.store(0);
          P.err = 0;
          ++epoch;
          P.expect.store(epoch);
          desc_kernel<<<1, 1, 0, g_s>>>(P.ring_d, epoch, n, (int)bytes, n == 1 ? peer : (g_size >= 4 ? -1 : peer));
          CK(cudaGetLastError());
          double t0 = now_us();
          while (!P.done.load()) { if (now_us() - t0 > 5e6) break; }
          if (!P.done.load()) { info("P3 %s n=%d: proxy never fired", mech, n); break; }
          CK(cudaStreamSynchronize(g_s));
          CK(cudaStreamSynchronize(P.ps));
          if (P.err) { ++errs; continue; }
          if (it >= 3) {
            flag2E_dev.push_back(us(P.ring->t_flag, g_stamps[21]));
            detect.push_back(P.t_detect + off - P.ring->t_flag / 1000.0);
            issue.push_back(P.t_issued - P.t_detect);
          }
        }
        if (errs) info("P3 %s: %d iterations returned error %d (%s)", mech, errs, P.err, P.err > 0 ? cudaGetErrorString((cudaError_t)P.err) : "not compiled");
        row("P3", mech, B.name, bytes, n, "flag_to_done_us", flag2E_dev);
        row("P3", mech, B.name, bytes, n, "flag_to_detect_us(+-launch)", detect);
        row("P3", mech, B.name, bytes, n, "proxy_issue_us", issue);
        if (!flag2E_dev.empty()) { Stat s = stat(flag2E_dev); row("P3", mech, B.name, bytes, n, "GBps_incl_latency", {(double)n * bytes / s.med / 1e3}); }
      }
    }
  }
  P.stop.store(1);
  P.th.join();
  CK(cudaStreamDestroy(P.ps));
  CK(cudaFreeHost(P.ring));
  info("P3 clock offset gpu-cpu = %.1f us (median of 11; uncertainty ~ one launch latency)", off);
}

// ------------------------------------------------------------ P4 CDP2 device memcpy
static void p4(const Buffers& B, int peer) {
  for (size_t bytes : kSizes) {
    std::vector<double> l2e;
    int errs = 0;
    for (int it = 0; it < g_iters + 3; ++it) {
      g_stamps[30] = g_stamps[31] = 0;
      *((volatile int*)g_err_h) = 0;
      cdp_copy_kernel<<<1, 1, 0, g_s>>>(B.peer_inbox[peer], B.src, bytes, g_stamps_d + 30, g_stamps_d + 31, g_err_d);
      cudaError_t le = cudaGetLastError();
      if (le != cudaSuccess) { info("P4 launch failed: %s", cudaGetErrorString(le)); cudaDeviceSynchronize(); cudaGetLastError(); return; }
      CK(cudaStreamSynchronize(g_s));
      CK(cudaDeviceSynchronize());
      int derr = *((volatile int*)g_err_h);
      if (derr) { ++errs; if (errs == 1) info("P4 device cudaMemcpyAsync error %d (%s)", derr, cudaGetErrorString((cudaError_t)derr)); continue; }
      if (g_stamps[31] == 0) { ++errs; continue; }
      if (it >= 3) l2e.push_back(us(g_stamps[30], g_stamps[31]));
    }
    row("P4", "cdp_memcpy", B.name, bytes, 1, "launch_to_end_us", l2e);
    if (!l2e.empty()) { Stat s = stat(l2e); row("P4", "cdp_memcpy", B.name, bytes, 1, "GBps_incl_latency", {bytes / s.med / 1e3}); }
    if (errs) info("P4 bytes=%zu: %d iterations failed", bytes, errs);
  }
}

// ------------------------------------------------------------ P5 signal visibility + ordering
static void p5(const Buffers& B, int peer, int from) {
  // receiver: my inbox region [from], my sig word [from]; sender: peer's inbox region [me], peer's sig [me]
  uint64_t* out_d;
  uint64_t* out;
  CK(cudaHostAlloc(&out, 4 * sizeof(uint64_t), cudaHostAllocMapped));
  CK(cudaHostGetDevicePointer(&out_d, out, 0));
  uint64_t* epoch_d;
  CK(cudaMalloc(&epoch_d, sizeof(uint64_t)));
  CUstream cs = (CUstream)g_s;
  static uint64_t epoch = 1000;
  for (int mech = 0; mech < 2; ++mech) {  // 0 = cuStreamWriteValue64, 1 = 8-byte memcpy from a device word
    const char* mname = mech == 0 ? "streamwrite64" : "memcpy8";
    for (size_t bytes : {kSizes[0], kSizes[2], kSizes[3]}) {
      int viol = 0, timeouts = 0, api_err = 0, iters = 60;
      std::vector<double> sig2seen;
      const uint64_t* data_rx = (const uint64_t*)((char*)B.inbox + (size_t)from * kRegion);
      for (int it = 0; it < iters; ++it) {
        ++epoch;
        // receiver arms first
        out[0] = out[1] = out[2] = 0;
        wait_verify_kernel<<<1, 256, 0, g_s>>>(B.sig + from, epoch, data_rx, bytes / 8, out_d);
        CK(cudaGetLastError());
        MPI_Barrier(MPI_COMM_WORLD);
        // sender: payload = epoch, then data copy, then signal
        cudaStream_t ss;
        CK(cudaStreamCreateWithFlags(&ss, cudaStreamNonBlocking));
        fill_kernel<<<64, 256, 0, ss>>>(B.src, bytes / 8, epoch);
        CK(cudaMemcpyAsync(B.peer_inbox[peer], B.src, bytes, cudaMemcpyDefault, ss));
        if (mech == 0) {
          CUresult r = cuStreamWriteValue64((CUstream)ss, (CUdeviceptr)B.peer_sig[peer], epoch, CU_STREAM_WRITE_VALUE_DEFAULT);
          if (r != CUDA_SUCCESS) { ++api_err; CK(cudaMemcpyAsync(epoch_d, &epoch, 8, cudaMemcpyHostToDevice, ss)); CK(cudaMemcpyAsync(B.peer_sig[peer], epoch_d, 8, cudaMemcpyDefault, ss)); }
        } else {
          CK(cudaMemcpyAsync(epoch_d, &epoch, 8, cudaMemcpyHostToDevice, ss));
          CK(cudaMemcpyAsync(B.peer_sig[peer], epoch_d, 8, cudaMemcpyDefault, ss));
        }
        CK(cudaStreamSynchronize(ss));
        CK(cudaStreamSynchronize(g_s));
        CK(cudaStreamDestroy(ss));
        if (out[0]) ++timeouts; else if (out[1]) ++viol;
        MPI_Barrier(MPI_COMM_WORLD);
      }
      info("P5 %s %s bytes=%zu: iters=%d timeouts=%d payload_violations=%d api_errors=%d", mname, B.name, bytes, iters, timeouts, viol, api_err);
      row("P5", mname, B.name, bytes, 1, "timeouts", {(double)timeouts});
      row("P5", mname, B.name, bytes, 1, "violations", {(double)viol});
      row("P5", mname, B.name, bytes, 1, "api_errors", {(double)api_err});
    }
  }
  CK(cudaFreeHost(out));
  CK(cudaFree(epoch_d));
}

// ------------------------------------------------------------ setup
static void setup_ipc(Buffers& B) {
  B.name = "ipc";
  CK(cudaMalloc(&B.src, kRegion));
  CK(cudaMalloc(&B.inbox, kRegion * g_size));
  CK(cudaMalloc(&B.sig, 64 * sizeof(uint64_t)));
  CK(cudaMemset(B.sig, 0, 64 * sizeof(uint64_t)));
  CK(cudaMemset(B.inbox, 0, kRegion * g_size));
  cudaIpcMemHandle_t hi, hs;
  CK(cudaIpcGetMemHandle(&hi, B.inbox));
  CK(cudaIpcGetMemHandle(&hs, B.sig));
  std::vector<cudaIpcMemHandle_t> all_i(g_size), all_s(g_size);
  MPI_Allgather(&hi, sizeof(hi), MPI_BYTE, all_i.data(), sizeof(hi), MPI_BYTE, MPI_COMM_WORLD);
  MPI_Allgather(&hs, sizeof(hs), MPI_BYTE, all_s.data(), sizeof(hs), MPI_BYTE, MPI_COMM_WORLD);
  for (int p = 0; p < g_size; ++p) {
    if (p == g_rank) { B.peer_inbox[p] = (uint64_t*)((char*)B.inbox + (size_t)p * kRegion); B.peer_sig[p] = B.sig + p; continue; }
    void* pi; void* ps;
    CK(cudaIpcOpenMemHandle(&pi, all_i[p], cudaIpcMemLazyEnablePeerAccess));
    CK(cudaIpcOpenMemHandle(&ps, all_s[p], cudaIpcMemLazyEnablePeerAccess));
    B.peer_inbox[p] = (uint64_t*)((char*)pi + (size_t)g_rank * kRegion);
    B.peer_sig[p] = (uint64_t*)ps + g_rank;
  }
  fill_kernel<<<256, 256>>>(B.src, kRegion / 8, 7);
  CK(cudaDeviceSynchronize());
}
#ifdef USE_NVSHMEM
static bool setup_nvshmem(Buffers& B) {
  B.name = "nvshmem";
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  MPI_Comm comm = MPI_COMM_WORLD;
  attr.mpi_comm = &comm;
  if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr) != 0) { info("nvshmemx_init_attr failed"); return false; }
  B.src = (uint64_t*)nvshmem_malloc(kRegion);
  B.inbox = (uint64_t*)nvshmem_malloc(kRegion * g_size);
  B.sig = (uint64_t*)nvshmem_malloc(64 * sizeof(uint64_t));
  if (!B.src || !B.inbox || !B.sig) { info("nvshmem_malloc failed (NVSHMEM_SYMMETRIC_SIZE?)"); return false; }
  CK(cudaMemset(B.sig, 0, 64 * sizeof(uint64_t)));
  CK(cudaMemset(B.inbox, 0, kRegion * g_size));
  for (int p = 0; p < g_size; ++p) {
    char* pi = (char*)nvshmem_ptr(B.inbox, p);
    uint64_t* ps = (uint64_t*)nvshmem_ptr(B.sig, p);
    if (!pi || !ps) { info("nvshmem_ptr(peer %d) is NULL (no P2P mapping)", p); return false; }
    B.peer_inbox[p] = (uint64_t*)(pi + (size_t)g_rank * kRegion);
    B.peer_sig[p] = ps + g_rank;
  }
  fill_kernel<<<256, 256>>>(B.src, kRegion / 8, 7);
  CK(cudaDeviceSynchronize());
  nvshmem_barrier_all();
  return true;
}
#endif

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_size);
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--busy") && i + 1 < argc) g_busy = argv[++i];
    else if (!strcmp(argv[i], "--only") && i + 1 < argc) g_only = argv[++i];
    else if (!strcmp(argv[i], "--iters") && i + 1 < argc) g_iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--free") && i + 1 < argc) g_free = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--onepersm")) g_onepersm = 1;
  }
  CK(cudaGetDeviceCount(&g_ndev));
  g_dev = g_rank % g_ndev;
  CK(cudaSetDevice(g_dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, g_dev));
  g_num_sms = prop.multiProcessorCount;
  int ce = 0, drv = 0, rt = 0;
  CK(cudaDeviceGetAttribute(&ce, cudaDevAttrAsyncEngineCount, g_dev));
  CK(cudaDriverGetVersion(&drv));
  CK(cudaRuntimeGetVersion(&rt));
  for (int p = 0; p < g_ndev; ++p) {
    if (p == g_dev) continue;
    int can = 0;
    CK(cudaDeviceCanAccessPeer(&can, g_dev, p));
    if (can) { cudaError_t e = cudaDeviceEnablePeerAccess(p, 0); if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CK(e); cudaGetLastError(); }
  }
  if (g_rank == 0) info("device=%s SMs=%d asyncEngineCount=%d driver=%d runtime=%d busy=%s iters=%d ranks=%d", prop.name, g_num_sms, ce, drv, rt, g_busy.c_str(), g_iters, g_size);
  CK(cudaStreamCreateWithFlags(&g_s, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&g_busy_s, cudaStreamNonBlocking));
  CK(cudaHostAlloc(&g_stamps, 256 * sizeof(uint64_t), cudaHostAllocMapped));
  memset(g_stamps, 0, 256 * sizeof(uint64_t));
  CK(cudaHostGetDevicePointer(&g_stamps_d, g_stamps, 0));
  int* err_h;
  CK(cudaHostAlloc(&err_h, sizeof(int), cudaHostAllocMapped));
  *err_h = 0;
  CK(cudaHostGetDevicePointer(&g_err_d, err_h, 0));
  g_err_h = err_h;
  CK(cudaMalloc(&g_busy_flag_d, sizeof(int)));
  CK(cudaMemset(g_busy_flag_d, 0, sizeof(int)));
  CK(cudaStreamCreateWithFlags(&g_ctl, cudaStreamNonBlocking));
  CK(cudaMalloc(&g_busy_buf, 1u << 30));
  if (g_busy != "none") g_busy += g_onepersm ? "_1persm" : ("_free" + std::to_string(g_free));
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&g_blocks_per_sm, busy_kernel, 1024, 0));
  if (g_rank == 0) info("busy grid = %d blocks x 1024 (blocks/SM %d, SMs %d, free slots %d, onepersm %d)", g_onepersm ? g_num_sms : g_blocks_per_sm * g_num_sms - g_free, g_blocks_per_sm, g_num_sms, g_free, g_onepersm);

  int peer = (g_rank + 1) % g_size, from = (g_rank + g_size - 1) % g_size;
  setup_ipc(g_ipc);
  std::vector<Buffers*> bufs = {&g_ipc};
#ifdef USE_NVSHMEM
  bool nv_ok = setup_nvshmem(g_nvs);
  if (nv_ok) bufs.push_back(&g_nvs);
#endif
  MPI_Barrier(MPI_COMM_WORLD);
  for (Buffers* Bp : bufs) {
    const Buffers& B = *Bp;
    if (g_busy != "none") {
      // warm-up with idle SMs: every module / graph type loads now (a lazily loaded module waits for
      // an idle GPU, and the busy kernel never ends -> deadlock otherwise)
      const int it = g_iters;
      g_iters = 1;
      g_quiet = true;
      if (want("p0")) p0(B, peer);
      if (want("p2")) p2(B, peer);
      if (want("p3")) p3(B, peer);
      if (want("p4")) p4(B, peer);
      g_quiet = false;
      g_iters = it;
      CK(cudaDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);
    }
    busy_start();
    if (want("p0")) p0(B, peer);
    if (want("p1")) p1(B, peer);
    if (want("p2")) p2(B, peer);
    if (want("p3")) p3(B, peer);
    if (want("p4")) p4(B, peer);
    busy_stop();
    MPI_Barrier(MPI_COMM_WORLD);
    if (want("p5")) { busy_start(); p5(B, peer, from); busy_stop(); }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#ifdef USE_NVSHMEM
  if (nv_ok) nvshmem_finalize();
#endif
  MPI_Finalize();
  return 0;
}
