# Comm-Only Benchmark: a2av_ring dispatch vs two-level allgather

This standalone benchmark isolates the **wire latency** of the two layer0
dispatch transports **as implemented in this repo** — no GEMM, no overlap, no
index math in the measured window. Both modes are line-by-line transcripts of
the production choreography in
`src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc`, driven by the same
traffic matrices the flux harness consumes.

## The two modes

**`a2av` — the `a2av_ring` dispatch wire path** (`a2av_dispatch()`):
- three streams as in flux: main, `cp_stream`, `cp_stream_inter_node`;
- window opens at `ready_event` (in production: recorded right after the pack;
  here the send buffer is pre-packed since prep is excluded by design);
- self-delivery: `cudaMemcpyAsync` D2D + self `signal_op` on `cp_stream`;
- one `nvshmemx_putmem_signal_nbi_on_stream` per destination carrying exactly
  `M[rank][d]` bytes to offset `RO[rank][d] = Σ_{s<rank} M[s][d]`, issued in
  the **reverse hierarchical ring** (mirror of `shift_rank_to_order`):
  intra-node slots on `cp_stream`, inter-node slots on `cp_stream_inter_node`;
  zero-byte pairs still signal — every source signals every destination;
- `fetch_remote_event` on the inter stream → `cp_stream` waits it, then the
  window closes when **all W per-source epoch signals have arrived** on
  `cp_stream` — the exact condition flux's GEMM tiles spin on
  (`signal[s] >= run_id`, monotonic, never reset);
- outgoing-put drainage (`quiet`) happens outside the window, matching the
  trailing `nvshmem_barrier_all` after the GEMM in production.

**`ag` — the two-level allgather** (`all_gather_all2all()`):
- own shard D2D into the replicated buffer on the main stream, then
  `nvshmemx_barrier_all_on_stream` (so every peer's slot is populated), as in
  src — this barrier is **inside** the window, it is part of the algorithm;
- outer loop over nodes in ring order (own node first). Per remote node:
  **one** `getmem_on_stream` of the *same-local-rank* shard on
  `cp_stream_inter_node`, then `nvshmemx_barrier_on_stream(NVSHMEMX_TEAM_NODE)`
  and `fetch_remote_event` → `cp_stream` waits;
- inner loop: NVLink fan-out — `getmem` the node's remaining shards **from
  node-mates** on `cp_stream`;
- window closes when all W shards are resident on `cp_stream` (the
  `all_gather_event` position in src).

Key structural point: the two-level AG is **node-deduplicated** — each shard
crosses each node boundary once and fans out over NVLink. At 4 nodes ×
16 MB shards its NIC traffic is 48 MB/rank (uniform), while its NVLink traffic
is 192 MB/rank. A naive flat pairwise "allgather" would push 192 MB/rank
through the NIC and measures ~3x slower — do not use flat patterns as an AG
proxy.

## Methodology

- **Bootstrap**: NVSHMEM unique-ID (`NVSHMEMX_INIT_WITH_UNIQUEID`), rank/size
  from `SLURM_PROCID`/`SLURM_NTASKS`, UID broadcast through a `$PSCRATCH`
  file. MPI is avoided deliberately: binaries outside the site MPI plumbing
  fall back to singleton, and the Cray `CC` wrapper drags in a GTL library
  with an unsatisfiable cudart-11 dependency. UID is also what flux itself
  uses (`NVSHMEM_BOOTSTRAP=UID` in `launch.sh`).
- **Alignment**: each iteration starts with `cudaDeviceSynchronize()` + host
  `nvshmem_barrier_all()`; reported latency therefore includes straggler skew
  caused by the traffic itself — the quantity that gates a synchronized step.
- **Warmup/iters**: 5 warmup + 20 timed (warmup covers transport setup only,
  per the project's one-shot-profiling convention; this bench contains no
  routing work at all).
- **Statistics**: each rank reports median, mean, and p95 of its window.
  The headline number per case is the **slowest rank by median**, reported
  with that rank's own median/mean/p95 — a synchronized step completes when
  its slowest rank does, and quoting one rank's full stats avoids mixing
  distributions across ranks.
- **Environment**: Slingshot-11, `NVSHMEM_REMOTE_TRANSPORT=libfabric`,
  `NVSHMEM_LIBFABRIC_PROVIDER=cxi`, `NVSHMEM_DISABLE_CUDA_VMM=1` (same as
  `launch.sh`). Intra-node NVSHMEM ops ride NVLink P2P; inter-node ops go
  through the proxy (no GPU-initiated NIC path on this fabric).

## Results (Perlmutter, 4 nodes x 4 A100, dist_001 matrices, 2026-07-17)

Slowest rank by median; that rank's median / mean / p95 in ms:

| budget | a2av_ring (matrix bytes) | two-level AG (full shards) | faster |
|---|---|---|---|
| 2 MiB  | rank 5: **0.51** / 0.51 / 0.57 | rank 13: 0.77 / 0.77 / 0.83 | a2av 1.5x |
| 16 MiB | rank 14: 1.76 / 1.77 / 1.84 | rank 10: **1.57** / 1.57 / 1.70 | AG 1.12x |
| 64 MiB | rank 12: 6.95 / 6.92 / 7.22 | rank 7: **4.75** / 4.86 / 5.76 | AG 1.46x |

Takeaways:

1. **Small messages: alltoallv wins on latency** — the AG pays two in-window
   barriers and a two-hop store-and-forward, which dominate when payloads are
   tiny.
2. **Large skewed traffic: the two-level AG's wire is genuinely faster**
   (4.75 vs 6.95 ms at 64 MiB) despite moving 4x the total bytes. Node-level
   dedup gives it a uniform 48 MB/rank NIC load, immune to routing skew; the
   192 MB/rank NVLink fan-out costs ~1 ms. Raw a2av concentrates the matrix's
   hot column on one node — at 64 MiB the slowest a2av rank is 12, not the
   hot-column rank 14: the whole hot *node* (ranks 12-15 receive ~124 MB
   aggregate vs ~50 MB on other nodes) slows together, i.e. the binding
   resource is per-node NIC ingress, and "15x fewer wire bytes" counts NVLink
   and NIC bytes as equal when they are not.
3. These floors are consistent with the end-to-end harness: dense-AG forward
   ~8.4 ms over a ~4.8 ms comm floor (overlap hides most of it behind the
   GEMM); a2av_ring forward ~9.0 ms over a ~7.0 ms floor (little left to
   hide). At 64 MiB **no amount of dispatch-prep or tile-schedule
   optimization closes the gap — the transport itself needs node-level
   dedup/forwarding** (each token crossing to a node once, then
   NVLink-scatter to its expert owners), which would be a new comm_pattern
   since it changes the wire-bytes-equal-matrix harness contract.

## Usage

```bash
source ./module.sh
bash a2av_comm_bench/build.sh
salloc --qos interactive -C gpu --account m4243_g -N 4 --gpus-per-node=4 \
  bash a2av_comm_bench/run.sh
# aggregate: grep RESULT, reduce med_ms per (budget, mode) by max/mean
```
