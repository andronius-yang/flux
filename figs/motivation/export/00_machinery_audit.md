# Motivation figure ("imbalance shifts") — machinery audit, 2026-09-02

Scope: postdoc's three-panel motivation diagram (NVSHMEM a2a+GEMM / EPLB /
COMET), 4–6 ranks per panel, drawn from real nsys timelines with per-rank
token counts and bytes. No OURS arm anywhere. This file records whether the
existing harness matches the proposed theory and whether the durations can be
captured and quantified. Numbers below are from existing capsules, not new runs.

## 1. Arm mapping (same arms as figs/main_perf)

| Panel | Theory | Arm (variants.py) | Driver | nsys-capable |
|---|---|---|---|---|
| 1 | a2a dispatch then GEMM; inter-node wire imbalance + compute imbalance | `l01_nvshmem` (blocking `putmem_on_stream` ring, world barrier, unfused GemmGroupedV2) | test_moe_combined/test_moe_l0l1_traffic.py `--impl nvshmem` | yes (no capture exists yet) |
| 2 | expert placement (hot experts replicated) then dispatch then balanced GEMM | `eplb_l01` (pool-oracle EPLB placement, staged All2AllSingle wire) | test_moe_ag_scatter/test_moe_eplb_traffic.py | yes (no capture exists yet) |
| 3 | COMET: intra-node overlap, inter-node exposed, compute imbalanced | `l01_allgather_dense` (stock sm80 port of upstream V3 allgather) | l0l1 driver `--impl flux --l0_comm_pattern allgather` | yes (4n K2 b8/b32 captures exist: motif capsule 20260830-132308, worktree binary) |

## 2. What the timeline machinery gives per rank (verified on the COMET b32 capture)

nsys mode = one `.nsys-rep` per node, all 4 local ranks in one timebase.
Exported with `nsys export --type sqlite` (0.3 s) and read with
`figs/motivation/nsys_rank_phases.py`. Two facts that matter for extraction:

- The `iterN` NVTX range is host-side and closes at enqueue (1.4 ms) while the
  device work runs 30.9 ms; attribute kernels/memcpys to an iteration by
  joining on `correlationId` to the runtime launch inside the range (the
  script does this), never by timestamp.
- Kernel names separate the phases cleanly: inter-node NVSHMEM =
  `nvshmemi_proxy_rma_entrypoint_blocking` (one per remote node, on the
  inter-node stream), node barrier = `barrier_on_stream_kernel_threadgroup`,
  intra-node = CUPTI memcpy copyKind 10 (P2P, 32 MB each), grouped GEMM =
  `Kernel` (2 launches), combine = `ep_topk_gather_rs_kernel_v2` +
  `nvshmemi_proxy_rma_signal_entrypoint_blocking`.

COMET K2 lcb b32 4n, node 0, iteration 5 (ms from iteration start):

| rank | routing allgather | inter-node fetch (3 x 32 MB) | intra P2P (12 x 32 MB) | GEMM0 start..end | GEMM0 ms | rows |
|---|---|---|---|---|---|---|
| 0 | 0.09–0.35 | 1.10–7.23 | 1.11–8.72 | 7.66–21.64 | 13.37 | 29207 |
| 1 | 0.08–0.37 | 1.10–7.45 | 1.11–8.60 | 7.65–19.52 | 9.10 | 19399 |
| 2 | 0.11–0.38 | 1.11–7.23 | 1.11–8.64 | 7.63–18.69 | 7.16 | 14850 |
| 3 | 0.08–0.38 | 1.08–7.26 | 1.09–8.66 | 7.66–19.63 | 9.38 | 20026 |

GEMM ms tracks matrix column sums almost linearly (rows from the capsule's
traffic matrix; all 16 ranks: 10072–29639 rows, max/mean 1.58). So
"tokens assigned" and "bytes per transfer" are all available offline from the
capsule (matrix_path, chunk_bytes, tokens_per_rank, H) and need no new
instrumentation.

## 3. Theory vs machinery — where they disagree

1. **COMET's inter-node wire is exposed, but it is not imbalanced.** The
   allgather fetches the same fixed 96 MB/rank (b32) from remote nodes on
   every rank; measured fetch spans differ by ~9% across ranks (5.2–5.7 ms).
   What the trace shows is a serial gate: the GEMM starts only after
   `fetch_remote_event` (7.6 ms) on every rank. Draw "exposed" not
   "imbalanced" for the inter-node bar; the imbalance in panel 3 is compute
   (13.4 vs 7.2 ms on one node, 1.87x).
2. **COMET's intra-node overlap with compute is nominal at nnodes>1.** The
   P2P copies (1.1–8.7 ms) overlap the inter-node fetch, not the GEMM
   (starts 7.6 ms); about 1 ms of the 8 ms intra wire overlaps compute. The
   "intra-node links overlapped with compute" premise holds for the
   single-node CUDA-IPC path and for the no-gate ablation
   (FLUX_A2AV_DENSE_NO_GEMM_GATE + conn=8, worktree branch motif-tiletrace),
   not for stock multi-node COMET. Provenance: the gate ships verbatim in
   upstream db4ffe0 (sm90 V3), our sm80 port is line-for-line.
3. **Sender-side dispatch bytes are equal on every rank by harness invariant 1**
   (matrix row sums = budget x topk). In the b32 K2 matrix, inter-node send
   MB/rank is 184–200 (8% spread). Per-rank byte imbalance exists on the
   RECEIVE side (column sums 3x) and therefore in compute. For panel 1 the
   inter-node wire imbalance will appear as incast stalls of blocking puts
   to hot receivers, not as unequal send volume. Caption accordingly.
4. **EPLB's placement phase is one-shot at setup, not per step.** The arm
   places once per cell (batched NCCL `isend/irecv`, `place_weights` in
   python/flux/testing/eplb_semantics.py) and never moves weights per
   iteration. It IS in the nsys capture (ncclDevKernel_SendRecv per rank,
   before the first `iter0_warmup` range) but carries no NVTX label, and the
   recorded `eplb_weight_place_ms_oneshot` (5181 ms, K2 b16 4n) is host wall
   time that includes CPU generation of the canonical weights inside the
   timed region, so it is not a transfer time. Per-rank recv bytes are
   recorded on every rank (1.23–1.53 GB, K2 b16 4n, 390 re-homed slots);
   per-rank SEND counts (the "hot expert home sends to many ranks" bar)
   are derivable offline from `weight_placement_pairs(plan)` (plan is
   deterministic from the cell's `.eplb_load.json`). Panel 2 must be captioned
   as periodic re-placement (DeepSeek re-solves every N steps), and the bar
   should come from the nsys NCCL kernel span, not the recorded ms.
5. **EPLB compute balance is real and quantified.** K2 b16 4n:
   gemm rows/rank 8656–10551 (imbalance 1.57 before → 1.13 after; pool
   prediction 1.00). Dispatch: remote_frac 0.937 (global policy re-homes
   nearly everything off-node), per-rank [W,W] bytes recorded as
   `eplb_wire_bytes`. Its wire is one All2AllSingle kernel (per-block nbi
   puts + signal waits), so inter- vs intra-node time is NOT separable
   inside it; only bytes are. Panel 2's dispatch bar = whole-kernel span
   per rank + inter-node byte annotation.
6. **Recorded per-rank brackets hide the imbalance; nsys is mandatory.**
   All three drivers barrier inside the timed bracket, so l0_ms spreads only
   1.02–1.13x across ranks while GEMM rows spread 3x (K2 b16 4n, records
   from capsules 20260831-051230 / 20260824-153542 / 20260824-154143). The
   `phases` mode is excluded for all three drivers. The per-rank wait shows
   up in nsys as barrier-kernel duration (COMET node 0: 1.7–6.0 ms per rank).

## 4a. EPLB placement wire port (2026-09-02, user-directed)

Ported the one-time placement from batched NCCL isend/irecv to blocking
NVSHMEM puts, opt-in via `--weight_place_wire nvshmem` (driver), runner
kwarg `weight_place_wire`, variant twin `eplb_l01_nvplace`, heap sizing in
`eplb_place_sym_bytes` (sweep.py). Clean and small (python only, no rebuild):

- Slot panels (`slot_fc1/slot_fc2`) and a one-expert staging pair become
  symmetric-heap tensors (`flux.nvshmem_create_tensor`, collective, identical
  shapes) — required because on CXI every put source AND destination must be
  symmetric. Adds nlp x (fc1+fc2) + 1 expert to the heap (K2: +1.5 GB).
- Each expert's ORIGINAL home walks the global pair list and issues one
  BLOCKING `nvshmem_putmem_on_stream` per (dest PE, slot) for fc1 and fc2,
  then one world barrier (wire rule 6a; receivers do nothing). Host weight
  synthesis happens before the bracket and is not in the NVTX range.
- nsys: outer range `eplb_place_weights`; one range per put
  `place_put e<l>->pe<host>.s<slot> <bytes>B`, so every put's device span
  (intra-node = P2P memcpy with bytes, inter-node = proxy RMA kernel) is
  attributable by correlation to destination and bytes. Per-rank send ledger
  is recorded as `eplb_weight_place_sends` in the rank JSONL.
- Correctness: the existing bitwise slot check (every re-homed slot equals
  the canonical weights) is the proof of the wire. Smoke spec:
  `sweeps/specs/motivation_smoke_2n_eplb_nvplace.yaml` (nvplace vs nccl twin).
- SMOKE GREEN (2n K2 b4, job 57859784, capsule 20260902-132036_perlmutter_e7e16490,
  correctness on, 2/2 ok): nvplace and NCCL twin place the identical 359
  re-homed slots (imbalance after 1.124 both); per-rank recv 2.35–2.69 GB
  identical across wires; new send ledger: 42–50 puts/rank, inter-node
  send 0.95–1.74 GB per rank (1.8x spread — the hot-home imbalance the
  panel wants). Bracket 209 ms (puts + barrier only) vs 8459 ms NCCL twin
  (includes host synthesis — never quote either as latency; the figure
  uses the nsys spans).
- Capture spec: `sweeps/specs/motivation_4n_k2_nsys.yaml` (3 arms, nsys,
  isolated discipline, K2 lcb b16 + b32; eplb b32 heap clamps at the 16G cap).

## 4. Gaps to close before capture

- No nsys capture exists for `l01_nvshmem` or `eplb_l01`; COMET's existing
  captures are on the worktree binary. Plan: ONE 4n capsule, three arms,
  `modes: [nsys]` with `extra_env: FLUX_SWEEP_ISOLATED_ITERS=1` (barrier before
  every window gives a common per-iteration origin across nodes), K2 lcb
  homog, b16 (the main figure's budget) and b32, `profile_iters 3`.
- Add `torch.cuda.nvtx.range("eplb_place_weights")` around the P2P batch in
  `place_weights` (timing-neutral) so panel 2's placement bar is addressable.
- Optional: a `disp_end` event in the nvshmem arm is NOT needed (nsys
  separates puts from GEMM); leave the driver alone.
- Cross-node rank selection: ranks on different nodes live in different reps
  with different clocks; align each rank at its `iterN` range start (post
  barrier, isolated mode). Same-node quads need no alignment.
- Tile-level in-GEMM overlap (option-B raster) needs the
  FLUX_A2AV_TILE_TRACE_DENSE port (branch motif-tiletrace, 12 commits, merges
  clean into main, requires libflux rebuild). Not needed for phase-level bars.

## 5. Capture + figure options (2026-09-02, same day)

- Capsule `20260902-133340_perlmutter_b27f2040` (job 57859970, 6/6 ok, nsys +
  isolated, K2 lcb b16 + b32, arms l01_nvshmem / eplb_l01_nvplace /
  l01_allgather_dense, one binary). UNCOMMITTED — runner's git add line is in
  the run log.
- Pipeline: `extract_phases.py` (sqlite export cached per CELL — rep basenames
  repeat across cells; correlation-id attribution) -> `phases_<capsule>.json`
  -> `build_figure_options.py --budget 16 --appendix-budget 32` ->
  `imbalance_shifts_options.html` (self-contained, both themes, three options
  A strips / B lanes / C ledger + all-16-rank tables + rank-selection rule).
- Headline numbers (b16, middle timed window, layer-0 window per rank):
  ring: inter-node wire occupancy 6.45–12.03 ms (1.32x), wait before GEMM
  0.02–6.12 ms (complementary), GEMM0 1.06–2.18 ms (1.50x); dispatch span is
  barrier-equalized (12.6–13.1). EPLB: placement span 61–152 ms per rank
  (1.27x max/mean, 20–29 puts, 0.71–1.37 GB inter-node per rank), a2a+barrier
  13.05–13.34 (equalized; a2a kernel only issues nbi puts — completion is in
  the barrier quiet, so NIC/NVLink are inseparable on this arm), GEMM 1.97–2.31
  (1.07x, rows 8.7k–10.6k). COMET: inter-node fetch 2.37–3.43 ms (byte-fixed
  96 MB/rank... at b16 48 MB), NVLink 2.2 ms flat, GEMM 1.41–3.48 (1.43x),
  wait AFTER GEMM 1.35–3.81 ms (the end-of-op barrier absorbs the spread).
- Rank rule: extremes of GEMM, extremes of inter-node wire (EPLB: of
  placement and dispatch), + median GEMM; 4–6 ranks, nodes mixed.

## 6. EPLB exposed dispatch wire (2026-09-02, user-directed side lane)

- `--dispatch_wire blocking_ring` (driver) / `enable_blocking_ring_wire` +
  `_a2av_blocking_ring` (EPLBLayer0Runner) / variant `eplb_l01_nvplace_bwire`
  / heap add in eplb_sym_size / spec `motivation_4n_k2_nsys_bwire.yaml`.
  Pack, placement, place, GEMM byte-identical; only the wire changes: one
  blocking putmem_on_stream per destination in ring order (hidden rows +
  fp32 probs) from/into symmetric panels laid out source-major like the a2a
  op, self block = device copy, ONE world barrier. NVTX `disp_put pe<d> <B>`
  per put. Instrumented lane — never a latency arm.
- Capsule `20260902-140327_perlmutter_7fd98ac6` (job 57860364, 2/2 ok,
  correctness on -> bitwise dispatch rows match). b16 per rank: inter-node
  wire occupancy 6.90–11.26 ms (1.24x) with 92–98 MB sent per rank (equal
  by construction) and 88–108 MB received; barrier wait before GEMM
  0.02–4.56 ms (complementary); GEMM 1.97–2.32 (1.08x, balanced). b32:
  14.8–22.7 ms (1.26x), wait 0.02–7.80. The staged a2a arm hides all of
  this inside its barrier quiet (13.05–13.34 equalized).
- Page rebuilt from both capsules (EPLB panel = exposed wire; section
  "Same EPLB step, two dispatch wires" shows the a2a twin on the same ranks).
