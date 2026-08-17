// Comm-only benchmark of the two layer0 dispatch transports AS IMPLEMENTED in
// this repo (src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc):
//
//   a2av  — a2av_ring dispatch wire path: self cudaMemcpyAsync + self signal on
//           cp_stream; one putmem_signal_nbi per destination in the reverse
//           hierarchical ring, intra-node puts on cp_stream, inter-node puts on
//           cp_stream_inter_node; window ends when all W per-source epoch
//           signals have arrived (the condition flux's GEMM tiles spin on).
//   ag    — all_gather_all2all two-level allgather: self shard copy + global
//           barrier on the main stream; per remote node, getmem of the
//           same-local-rank shard on cp_stream_inter_node + node-team barrier;
//           NVLink fan-out getmems of the remaining shards from node-mates on
//           cp_stream; window ends when every shard is resident.
//
//   prefetch — MoonEP-style weight pull (flux WeightPrefetchGetmem shape):
//           every rank simultaneously getmem-pulls <msg_bytes> of expert
//           weights from a symmetric source on the NEXT node's same-local
//           rank (cross-node, 1 ingress + 1 egress per NIC — the link
//           saturation case; PREFETCH_INTRA=1 pulls from a node-mate for
//           the NVLink baseline). Source is symmetric and immutable, dst is
//           ordinary cudaMalloc, no signaling — completion via
//           quiet_on_stream, as in the real op. Knobs:
//           PREFETCH_IMPL=kernel|stream, PREFETCH_CHUNK_BYTES (default =
//           msg, i.e. one chunk), PREFETCH_NBLOCKS (kernel impl, default 8).
//
//   egress_shard — NIC-sharded weight push (the WeightPushMulticast egress
//           sharding design): per node, SHARD_NHOMES "hot expert home" ranks
//           (lr < NHOMES) each own one <msg_bytes> cross-node leg to the
//           same-lr rank on the NEXT node. SHARD_DIRECT=1 pushes the whole
//           leg over the home's single NIC (putmem_signal SET — the
//           baseline). Otherwise the leg is byte-split into SHARD_L
//           near-equal shards riding same-local-rank wires:
//             home --NVLink CE--> egress (node, i) staging  [signal SET]
//             egress --NIC--> ingress (node+1, i) staging   [signal SET]
//             ingress --NVLink--> final slot @byte offset   [SIGNAL_ADD +1]
//           Fast path i == home_lr: home NIC-pushes its own shard straight
//           into the final slot (egress == home, ingress == dest). The dest
//           waits arrive >= cumulative expected chunk count — the same wait
//           the production forward_shard_join() performs before re-emitting
//           the epoch SET. SHARD_CHUNK_BYTES < shard pipelines the NVLink
//           staging against the NIC push (per-chunk signals). Knobs:
//           SHARD_L (default GPUS_PER_NODE), SHARD_NHOMES (default 1),
//           SHARD_CHUNK_BYTES (default = whole shard), SHARD_DIRECT.
//
// No GEMM, no index math: offsets precomputed outside the timed window.
// Bootstrap is NVSHMEM unique-ID via a shared file (as flux's launch.sh).
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CUCHECK(cmd)                                                         \
  do {                                                                       \
    cudaError_t e = (cmd);                                                   \
    if (e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

static int
env_int(const char *k, int d) {
  const char *v = getenv(k);
  return v ? atoi(v) : d;
}

static long long
env_ll(const char *k, long long d) {
  const char *v = getenv(k);
  return v ? atoll(v) : d;
}

// Mirror of flux's weight_prefetch_getmem_kernel: grid-stride over chunks,
// one nvshmemx_getmem_nbi_block per chunk, drained by quiet_on_stream on the
// host side (zero signaling — the source is immutable).
__global__ void __launch_bounds__(1024, 1)
prefetch_getmem_kernel(char *dst, const char *src, long long msg, long long chunk, int pe) {
  long long nchunks = (msg + chunk - 1) / chunk;
  for (long long c = blockIdx.x; c < nchunks; c += gridDim.x) {
    long long off = c * chunk;
    long long b = msg - off;
    if (b > chunk) b = chunk;
    nvshmemx_getmem_nbi_block(dst + off, src + off, b, pe);
  }
}

int
main(int argc, char **argv) {
  int rank = env_int("SLURM_PROCID", 0);
  int W = env_int("SLURM_NTASKS", 1);
  int local_rank = env_int("SLURM_LOCALID", 0);
  CUCHECK(cudaSetDevice(local_rank));

  if (argc < 4) {
    if (rank == 0)
      fprintf(stderr, "usage: %s <uidfile> a2av <matrix.txt> [iters]\n"
                      "       %s <uidfile> ag <shard_bytes> [iters]\n"
                      "       %s <uidfile> prefetch <msg_bytes> [iters]\n"
                      "       %s <uidfile> egress_shard <msg_bytes> [iters]\n",
              argv[0], argv[0], argv[0], argv[0]);
    return 1;
  }
  const char *uidfile = argv[1];
  std::string mode = argv[2];
  std::string arg = argv[3];
  int iters = argc > 4 ? atoi(argv[4]) : 20;
  int warmup = 5;

  // ---- UID bootstrap via shared file ----
  nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;
  if (rank == 0) {
    nvshmemx_get_uniqueid(&uid);
    std::string tmp = std::string(uidfile) + ".tmp";
    FILE *f = fopen(tmp.c_str(), "wb");
    fwrite(&uid, sizeof(uid), 1, f);
    fclose(f);
    rename(tmp.c_str(), uidfile);
  } else {
    FILE *f = nullptr;
    for (int i = 0; i < 3000 && !f; i++) {
      f = fopen(uidfile, "rb");
      if (!f) usleep(10000);
    }
    if (!f || fread(&uid, sizeof(uid), 1, f) != 1) {
      fprintf(stderr, "rank %d: uid file read failed\n", rank);
      return 1;
    }
    fclose(f);
  }
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  nvshmemx_set_attr_uniqueid_args(rank, W, &uid, &attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
  if (nvshmem_my_pe() != rank || nvshmem_n_pes() != W) {
    fprintf(stderr, "rank %d: nvshmem pe mismatch\n", rank);
    return 1;
  }
  if (rank == 0) fprintf(stderr, "[bench] nvshmem up: %d pes, mode=%s\n", W, mode.c_str());

  int L = env_int("GPUS_PER_NODE", 4);
  int NN = W / L, node = rank / L, lr = rank % L;

  // flux stream layout: main + cp_stream + cp_stream_inter_node
  cudaStream_t main_s, cp, cp_inter;
  CUCHECK(cudaStreamCreateWithFlags(&main_s, cudaStreamNonBlocking));
  CUCHECK(cudaStreamCreateWithFlags(&cp, cudaStreamNonBlocking));
  CUCHECK(cudaStreamCreateWithFlags(&cp_inter, cudaStreamNonBlocking));
  cudaEvent_t t0, t1, ready_event, fetch_remote_event;
  CUCHECK(cudaEventCreate(&t0));
  CUCHECK(cudaEventCreate(&t1));
  CUCHECK(cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming));
  CUCHECK(cudaEventCreateWithFlags(&fetch_remote_event, cudaEventDisableTiming));

  std::vector<float> t;
  long long wire_bytes = 0;  // this rank's non-self send bytes on the wire

  if (mode == "a2av") {
    // ---- traffic matrix, offsets (mirrors a2av_dispatch stage-1 results) ----
    std::vector<long long> M((size_t)W * W, 0);
    std::ifstream f(arg);
    int n;
    f >> n;
    if (n != W) {
      if (rank == 0) fprintf(stderr, "matrix nranks %d != world %d\n", n, W);
      return 1;
    }
    for (size_t i = 0; i < (size_t)W * W; i++) f >> M[i];
    long long send_total = 0, recv_total = 0;
    std::vector<long long> send_off(W, 0), recv_off(W, 0);
    for (int d = 0; d < W; d++) {
      send_off[d] = send_total;
      send_total += M[(size_t)rank * W + d];
    }
    for (int d = 0; d < W; d++) {
      long long acc = 0;
      for (int s = 0; s < rank; s++) acc += M[(size_t)s * W + d];
      recv_off[d] = acc;
    }
    for (int s = 0; s < W; s++) recv_total += M[(size_t)s * W + rank];
    long long max_send = 1, max_recv = 1;
    for (int r = 0; r < W; r++) {
      long long st = 0, rt = 0;
      for (int d = 0; d < W; d++) st += M[(size_t)r * W + d];
      for (int s = 0; s < W; s++) rt += M[(size_t)s * W + r];
      max_send = std::max(max_send, st);
      max_recv = std::max(max_recv, rt);
    }
    wire_bytes = send_total - M[(size_t)rank * W + rank];

    char *sendbuf = (char *)nvshmem_malloc(max_send);
    char *recvbuf = (char *)nvshmem_malloc(max_recv);
    uint64_t *sig = (uint64_t *)nvshmem_calloc(W, sizeof(uint64_t));
    CUCHECK(cudaMemset(sendbuf, 1, send_total));

    // reverse hierarchical ring: slot k -> intra-node first, then previous nodes
    // (mirror of shift_rank_to_order; see a2av_dispatch)
    std::vector<int> order, dn_of;
    for (int k = 1; k < W; k++) {
      int dn = k / L, dl = k % L;
      order.push_back(((node - dn + NN) % NN) * L + ((lr - dl + L) % L));
      dn_of.push_back(dn);
    }

    uint64_t epoch = 0;
    for (int it = 0; it < warmup + iters; it++) {
      CUCHECK(cudaDeviceSynchronize());
      nvshmem_barrier_all();
      epoch++;
      // window start = pack complete / ready_event in the real dispatch
      CUCHECK(cudaEventRecord(t0, main_s));
      CUCHECK(cudaEventRecord(ready_event, main_s));
      CUCHECK(cudaStreamWaitEvent(cp, ready_event, 0));
      CUCHECK(cudaStreamWaitEvent(cp_inter, ready_event, 0));
      long long self_b = M[(size_t)rank * W + rank];
      if (self_b > 0) {
        CUCHECK(cudaMemcpyAsync(
            recvbuf + recv_off[rank], sendbuf + send_off[rank], self_b,
            cudaMemcpyDeviceToDevice, cp));
      }
      nvshmemx_signal_op_on_stream(sig + rank, epoch, NVSHMEM_SIGNAL_SET, rank, cp);
      for (size_t k = 0; k < order.size(); k++) {
        int d = order[k];
        cudaStream_t put_s = dn_of[k] == 0 ? cp : cp_inter;  // stream split as in src
        long long b = M[(size_t)rank * W + d];
        if (b > 0) {
          nvshmemx_putmem_signal_nbi_on_stream(
              recvbuf + recv_off[d], sendbuf + send_off[d], b, sig + rank, epoch,
              NVSHMEM_SIGNAL_SET, d, put_s);
        } else {
          nvshmemx_signal_op_on_stream(sig + rank, epoch, NVSHMEM_SIGNAL_SET, d, put_s);
        }
      }
      CUCHECK(cudaEventRecord(fetch_remote_event, cp_inter));
      CUCHECK(cudaStreamWaitEvent(cp, fetch_remote_event, 0));
      // input fully resident == every source's signal arrived (GEMM spin condition)
      for (int s = 0; s < W; s++) {
        nvshmemx_signal_wait_until_on_stream(sig + s, NVSHMEM_CMP_GE, epoch, cp);
      }
      CUCHECK(cudaEventRecord(t1, cp));
      nvshmemx_quiet_on_stream(cp_inter);  // outgoing drain outside the window
      CUCHECK(cudaStreamSynchronize(cp));
      CUCHECK(cudaStreamSynchronize(cp_inter));
      float ms;
      CUCHECK(cudaEventElapsedTime(&ms, t0, t1));
      if (it >= warmup) t.push_back(ms);
    }
  } else if (mode == "ag") {
    // ---- two-level allgather exactly as all_gather_all2all ----
    long long shard = atoll(arg.c_str());
    wire_bytes = (long long)(NN - 1) * shard;  // NIC bytes; NVLink fan-out is (W - NN) * shard? per-rank intra pulls: (L-1)*NN
    char *input_buffer = (char *)nvshmem_malloc((size_t)W * shard);  // replicated [W, shard]
    char *inputs_shard;
    CUCHECK(cudaMalloc(&inputs_shard, shard));  // private shard, as in flux
    CUCHECK(cudaMemset(inputs_shard, 1, shard));

    for (int it = 0; it < warmup + iters; it++) {
      CUCHECK(cudaDeviceSynchronize());
      nvshmem_barrier_all();
      CUCHECK(cudaEventRecord(t0, main_s));
      for (int node_idx = node, i = 0; i < NN; ++i, node_idx = (node_idx + 1) % NN) {
        if (node_idx == node) {
          // own shard into my replicated slot, then global barrier so every
          // peer's slot is populated before anyone pulls it (as in src)
          CUCHECK(cudaMemcpyAsync(
              input_buffer + (size_t)rank * shard, inputs_shard, shard,
              cudaMemcpyDeviceToDevice, main_s));
          nvshmemx_barrier_all_on_stream(main_s);
          CUCHECK(cudaEventRecord(ready_event, main_s));
          CUCHECK(cudaStreamWaitEvent(cp, ready_event, 0));
        } else {
          if (i == 1) {
            CUCHECK(cudaStreamWaitEvent(cp_inter, ready_event, 0));
          }
          // inter-node: fetch ONLY the same-local-rank shard of that node
          int src_rank = node_idx * L + lr;
          nvshmemx_getmem_on_stream(
              input_buffer + (size_t)src_rank * shard,
              input_buffer + (size_t)src_rank * shard, shard, src_rank, cp_inter);
          nvshmemx_barrier_on_stream(NVSHMEMX_TEAM_NODE, cp_inter);
          CUCHECK(cudaEventRecord(fetch_remote_event, cp_inter));
          CUCHECK(cudaStreamWaitEvent(cp, fetch_remote_event, 0));
        }
        // intra-node fan-out: pull this node's other shards from node-mates
        for (int l = lr, j = 0; j < L; ++j, l = (l + 1) % L) {
          if (l != lr) {
            int src_rank = node_idx * L + l;
            int peer = node * L + l;  // node-mate holding that shard
            nvshmemx_getmem_on_stream(
                input_buffer + (size_t)src_rank * shard,
                input_buffer + (size_t)src_rank * shard, shard, peer, cp);
          }
        }
      }
      CUCHECK(cudaEventRecord(t1, cp));  // all shards resident (all_gather_event)
      CUCHECK(cudaStreamSynchronize(cp));
      CUCHECK(cudaStreamSynchronize(cp_inter));
      CUCHECK(cudaStreamSynchronize(main_s));
      float ms;
      CUCHECK(cudaEventElapsedTime(&ms, t0, t1));
      if (it >= warmup) t.push_back(ms);
    }
  } else if (mode == "prefetch") {
    // ---- MoonEP weight pull, one expert matrix per rank per iteration ----
    long long msg = atoll(arg.c_str());
    long long chunk = env_ll("PREFETCH_CHUNK_BYTES", msg);
    int nblocks = env_int("PREFETCH_NBLOCKS", 8);
    int intra = env_int("PREFETCH_INTRA", 0);
    const char *impl_s = getenv("PREFETCH_IMPL");
    bool kernel_impl = !impl_s || strcmp(impl_s, "kernel") == 0;
    // cross-node: same local rank on the next node (every NIC serves
    // exactly one ingress and one egress pull — the saturation case);
    // intra: next local rank on the same node (NVLink baseline)
    int home = intra ? node * L + (lr + 1) % L : ((node + 1) % NN) * L + lr;
    wire_bytes = intra ? 0 : msg;
    long long nchunks = (msg + chunk - 1) / chunk;
    if (nblocks > nchunks) nblocks = (int)nchunks;

    int dst_sym = env_int("PREFETCH_DST_SYM", 1);
    char *weight_home = (char *)nvshmem_malloc(msg);  // symmetric source
    char *dst;
    if (dst_sym) {
      // proxy-mediated (cross-node) gets need the LOCAL destination
      // registered with the provider too — ordinary cudaMalloc dst
      // segfaults on CXI (found 2026-08-11; intra-node P2P gets don't
      // care). Upstream's prefetch slots are mapped memory as well.
      dst = (char *)nvshmem_malloc(msg);
    } else {
      CUCHECK(cudaMalloc(&dst, msg));  // the crashing config, kept for A/B
    }
    CUCHECK(cudaMemset(weight_home, 1, msg));
    if (rank == 0)
      fprintf(stderr,
              "[bench] prefetch msg=%lld chunk=%lld nchunks=%lld nblocks=%d "
              "impl=%s intra=%d (rank->home example: 0->%d)\n",
              msg, chunk, nchunks, nblocks, kernel_impl ? "kernel" : "stream",
              intra, home);

    for (int it = 0; it < warmup + iters; it++) {
      CUCHECK(cudaDeviceSynchronize());
      nvshmem_barrier_all();
      CUCHECK(cudaEventRecord(t0, main_s));
      if (kernel_impl) {
        prefetch_getmem_kernel<<<nblocks, 1024, 0, main_s>>>(
            dst, weight_home, msg, chunk, home);
      } else {
        for (long long c = 0; c < nchunks; c++) {
          long long off = c * chunk;
          long long b = std::min(chunk, msg - off);
          nvshmemx_getmem_nbi_on_stream(dst + off, weight_home + off, b, home, main_s);
        }
      }
      // the join: local completion of my own pulls — nothing else, exactly
      // as WeightPrefetchGetmem::forward
      nvshmemx_quiet_on_stream(main_s);
      CUCHECK(cudaEventRecord(t1, main_s));
      CUCHECK(cudaStreamSynchronize(main_s));
      float ms;
      CUCHECK(cudaEventElapsedTime(&ms, t0, t1));
      if (it >= warmup) t.push_back(ms);
    }
  } else if (mode == "egress_shard") {
    // ---- NIC-sharded weight push (see header). One leg per home rank per
    // node; every leg's shard geometry is identical, so all indices are
    // derivable on every rank with zero metadata exchange (the production
    // scheme's replicated-plan property).
    long long msg = atoll(arg.c_str());
    int SL = env_int("SHARD_L", L);
    int NHOMES = env_int("SHARD_NHOMES", 1);
    int direct = env_int("SHARD_DIRECT", 0);
    if (SL < 1) SL = 1;
    if (SL > L) SL = L;
    if (NHOMES < 1) NHOMES = 1;
    if (NHOMES > L) NHOMES = L;
    if (NN < 2) {
      if (rank == 0) fprintf(stderr, "egress_shard needs >= 2 nodes\n");
      return 1;
    }
    // near-equal byte split of [0, msg) into SL shards (production cut rule)
    std::vector<long long> cut(SL + 1, 0), len(SL, 0);
    {
      long long base = msg / SL, rem = msg % SL;
      for (int i = 0; i < SL; i++) {
        len[i] = base + (i < rem ? 1 : 0);
        cut[i + 1] = cut[i] + len[i];
      }
    }
    long long shard_max = len.empty() ? 1 : len[0];  // shard 0 is the longest
    long long chunk = env_ll("SHARD_CHUNK_BYTES", shard_max);
    if (chunk < 1 || chunk > shard_max) chunk = shard_max;
    long long MAXC = (shard_max + chunk - 1) / chunk;
    std::vector<long long> nch(SL, 0);
    long long total_chunks = 0;  // per leg == the dest's per-iter arrive quota
    for (int i = 0; i < SL; i++) {
      nch[i] = len[i] > 0 ? (len[i] + chunk - 1) / chunk : 0;
      total_chunks += nch[i];
    }
    wire_bytes = direct ? (lr < NHOMES ? msg : 0)
                        : (lr < SL ? (long long)NHOMES * len[lr] : 0);

    char *weight_home = (char *)nvshmem_malloc(msg);   // leg source (home ranks)
    char *final_slot = (char *)nvshmem_malloc(msg);    // leg dest (dest ranks)
    char *eg_stage = (char *)nvshmem_malloc((size_t)NHOMES * shard_max);
    char *in_stage = (char *)nvshmem_malloc((size_t)NHOMES * shard_max);
    uint64_t *eg_sig = (uint64_t *)nvshmem_calloc((size_t)NHOMES * MAXC, sizeof(uint64_t));
    uint64_t *in_sig = (uint64_t *)nvshmem_calloc((size_t)NHOMES * MAXC, sizeof(uint64_t));
    uint64_t *arrive = (uint64_t *)nvshmem_calloc(1, sizeof(uint64_t));
    CUCHECK(cudaMemset(weight_home, 1, msg));
    if (rank == 0)
      fprintf(stderr,
              "[bench] egress_shard msg=%lld SL=%d nhomes=%d chunk=%lld "
              "chunks/leg=%lld direct=%d\n",
              msg, SL, NHOMES, chunk, total_chunks, direct);

    uint64_t epoch = 0, expected = 0;
    int next_node = (node + 1) % NN;
    for (int it = 0; it < warmup + iters; it++) {
      CUCHECK(cudaDeviceSynchronize());
      nvshmem_barrier_all();
      epoch++;
      CUCHECK(cudaEventRecord(t0, main_s));
      CUCHECK(cudaEventRecord(ready_event, main_s));
      CUCHECK(cudaStreamWaitEvent(cp, ready_event, 0));
      CUCHECK(cudaStreamWaitEvent(cp_inter, ready_event, 0));
      if (direct) {
        if (lr < NHOMES) {  // single-NIC baseline: whole leg from the home
          nvshmemx_putmem_signal_nbi_on_stream(
              final_slot, weight_home, msg, arrive, epoch, NVSHMEM_SIGNAL_SET,
              next_node * L + lr, cp_inter);
        }
      } else {
        // HOME role: stage shards to node-mates (NVLink CE, wait-free); fast
        // path i == h goes NIC-direct into the final slot.
        if (lr < NHOMES) {
          int h = lr;
          for (int i2 = 0; i2 < SL; i2++) {
            for (long long c = 0; c < nch[i2]; c++) {
              long long soff = c * chunk;
              long long b = std::min(chunk, len[i2] - soff);
              long long goff = cut[i2] + soff;
              if (i2 == h) {
                nvshmemx_putmem_signal_nbi_on_stream(
                    final_slot + goff, weight_home + goff, b, arrive, 1,
                    NVSHMEM_SIGNAL_ADD, next_node * L + h, cp_inter);
              } else {
                nvshmemx_putmem_signal_nbi_on_stream(
                    eg_stage + (size_t)h * shard_max + soff, weight_home + goff,
                    b, eg_sig + h * MAXC + c, epoch, NVSHMEM_SIGNAL_SET,
                    node * L + i2, main_s);
              }
            }
          }
        }
        // EGRESS role (my shard index == lr): per staged chunk, NIC-push to
        // the same-lr ingress rank on the next node.
        if (lr < SL) {
          for (int h = 0; h < NHOMES; h++) {
            if (h == lr) continue;  // that leg's shard lr went NIC-direct
            for (long long c = 0; c < nch[lr]; c++) {
              long long soff = c * chunk;
              long long b = std::min(chunk, len[lr] - soff);
              nvshmemx_signal_wait_until_on_stream(
                  eg_sig + h * MAXC + c, NVSHMEM_CMP_GE, epoch, cp_inter);
              nvshmemx_putmem_signal_nbi_on_stream(
                  in_stage + (size_t)h * shard_max + soff,
                  eg_stage + (size_t)h * shard_max + soff, b,
                  in_sig + h * MAXC + c, epoch, NVSHMEM_SIGNAL_SET,
                  next_node * L + lr, cp_inter);
            }
          }
        }
        // INGRESS role: legs landing on my node (home (node-1, h) ->
        // dest (node, h)); NVLink-copy each chunk into the dest's slot at its
        // global byte offset, +1 on the dest's arrive counter.
        if (lr < SL) {
          for (int h = 0; h < NHOMES; h++) {
            if (h == lr) continue;
            for (long long c = 0; c < nch[lr]; c++) {
              long long soff = c * chunk;
              long long b = std::min(chunk, len[lr] - soff);
              nvshmemx_signal_wait_until_on_stream(
                  in_sig + h * MAXC + c, NVSHMEM_CMP_GE, epoch, cp);
              nvshmemx_putmem_signal_nbi_on_stream(
                  final_slot + cut[lr] + soff,
                  in_stage + (size_t)h * shard_max + soff, b, arrive, 1,
                  NVSHMEM_SIGNAL_ADD, node * L + h, cp);
            }
          }
        }
      }
      // drain my NIC pushes inside the window, then the dest arrival wait
      nvshmemx_quiet_on_stream(cp_inter);
      CUCHECK(cudaEventRecord(fetch_remote_event, cp_inter));
      CUCHECK(cudaStreamWaitEvent(cp, fetch_remote_event, 0));
      if (lr < NHOMES) {  // DEST of the leg arriving from the previous node
        if (direct) {
          nvshmemx_signal_wait_until_on_stream(arrive, NVSHMEM_CMP_GE, epoch, cp);
        } else {
          expected += (uint64_t)total_chunks;  // cumulative, never reset
          nvshmemx_signal_wait_until_on_stream(arrive, NVSHMEM_CMP_GE, expected, cp);
        }
      }
      CUCHECK(cudaEventRecord(t1, cp));
      CUCHECK(cudaStreamSynchronize(cp));
      CUCHECK(cudaStreamSynchronize(cp_inter));
      CUCHECK(cudaStreamSynchronize(main_s));
      float ms;
      CUCHECK(cudaEventElapsedTime(&ms, t0, t1));
      if (it >= warmup) t.push_back(ms);
    }
  } else {
    if (rank == 0) fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
  }

  std::sort(t.begin(), t.end());
  double mean = 0;
  for (float x : t) mean += x;
  mean /= t.size();
  printf(
      "RESULT mode=%s rank=%d med_ms=%.3f mean_ms=%.3f p95_ms=%.3f wire_MB=%.2f\n",
      mode.c_str(), rank, t[t.size() / 2], mean, t[(size_t)(t.size() * 0.95)],
      wire_bytes / 1e6);
  fflush(stdout);
  nvshmem_finalize();
  return 0;
}
