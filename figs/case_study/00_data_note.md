# Case-study figure lane — data note (2026-09-05)

Two 4n case studies of OURS (one-round overlapped expert swap,
`ours_l01_s2_swap_p2p_t1_r2`) showing where GPU compute, NVLink and NIC
RDMA time went, plus two twins per case: `_noov` (sequential swap = overlap
effect) and `_esplit` (locality-oblivious equal-split routing = local-routing
effect). All from ONE capsule / ONE binary / ONE process per arm:

* capsule `sweeps/results/runs/20260905-121506_perlmutter_7a584851` (6/6 ok,
  nsys mode, K2 b64, 16 ranks). nsys mode = per-iteration synced timelines
  and byte ledgers only — NEVER latency (SCHEMA rule 3).
* spec `sweeps/specs/casestudy_sc_d4_k2_4n_nsys.yaml`: family 1 = the S-C
  dwell-4 schedule of handoff 34 (LOO 7-pool basis, 8 topics, placement
  carried over, 32+5 iterations all profiled); family 2 = plain lcb homog
  (main-figure pool, fresh oracle placement, 3+3 iterations).
* iteration -> topic (family 1, run index incl. 5 warmups; NVTX label
  `iterN`): proLaw 0-3 (warmup), lcb 4-7, clinical 8-11, colmath 12-15,
  elecEng 16-19, hsPsych 20-23, hsWorldHist 24-27, philosophy 28-31,
  proLaw 32-35, lcb 36. Sidecar epoch index k == run iteration k.

Selected cells / iterations:

| case | cell | iteration | why |
|---|---|---|---|
| EFFICIENT | plain lcb `..._trace-610042_b64_k8_nsys` | `iter4` (2nd timed) | fresh oracle placement, 46.4 ms, rows/rank spread 1.17x, no swap fires |
| SKEWED | schedule `..._trace-b549f7_b64_k8_nsys` | `iter33` (2nd of the proLaw block; `iter32` = block entry) | LOO basis, 57.1 ms, rows/rank spread 1.53x, 14/16 ranks swap 58 MB every iteration |

Derived data (PSCRATCH, regenerate with the two scripts here):
`$PSCRATCH/workspace/andrewy/figs_data/case_study/{timeline_20260905-121506.json,
timeline_summary_20260905-121506.csv, byte_ledger_20260905-121506.csv, sqlite/}`
— `extract_timeline.py` (per rank x iteration device events classed into
GPU / NIC / NVLink / wait lanes + host `plan.*`/`swap.*` ranges + a2av proxy
per-source wait/compute ranges), `byte_ledger.py` (receive-side rows/bytes
per rank per epoch split self / NVLink / NIC from the tile-trace sidecar).

Code changes for this lane (ALL case-study-only, default-off, no effect on
existing arms/specs): `--route_rule equal_split` (driver + planner +
`equal_split_route_all` sizing fold), gated `swap.*` NVTX ranges
(`FLUX_OURS_NVTX=1`), runner keeps spec iters for schedule families under
profiling modes, arm `ablation_l01_s2_swap_t1_esplit_p2p_r2`.

## Capture 2 (2026-09-05 pm) — overlapped COMET baseline, one rebuilt binary

User ruling: the figure needs an "overlapped" worse-performing baseline to
compare the OURS timeline against = COMET with its pre-GEMM completion gate
dropped and `CUDA_DEVICE_MAX_CONNECTIONS=8` (`l01_allgather_dense_nogate_c8`,
handoff 30 rebuttal cell). The knob lived only on the motif branch, so
`36dfd8d` (TILE_TRACE_DENSE) and `af76971` (DENSE_NO_GEMM_GATE + arms +
specs) were cherry-picked onto main as `b2e9998` / `bb4a637` and the .so
rebuilt (9/5 07:06, tags present, python/flux/lib + lib64 synced).
**Capsule 20260905-121506 is the pre-knob binary — never mix.**

* capsule `sweeps/results/runs/20260905-141104_perlmutter_fdd1c4af` (20/20
  ok): spec `casestudy2_baseline_sc_d4_k2_4n_nsys.yaml` = the same two
  families x 5 arms (COMET no-gate c8, COMET gated, OURS ovl / noov /
  equal-split) x modes [nsys, isolated]. THE FIGURE IS NOW DRAWN FROM THIS
  CAPSULE (all rows one binary); the isolated cells answer the latency
  question on the same binary.
* Isolated, max rank per iteration, median over block, total / l0 / l1 ms:

| arm | plain lcb (efficient) | sched proLaw block (skewed) | sched whole |
|---|---|---|---|
| COMET overlapped (no-gate c8) | 55.2 / 21.7 / 32.6 | 58.2 / 23.6 / 33.7 | 54.0 / 20.9 / 31.3 |
| COMET gated | 60.2 / 29.2 / 29.4 | 64.7 / 33.6 / 30.1 | 58.0 / 29.2 / 27.9 |
| OURS overlapped swap | 46.7 / 19.0 / 25.0 | 56.6 / 23.1 / 29.8 | 49.6 / 20.7 / 25.9 |
| OURS sequential swap | 46.3 / 19.3 / 24.1 | 57.2 / 22.8 / 30.6 | 51.6 / 20.7 / 26.8 |
| OURS equal-split routing | 50.0 / 21.2 / 26.2 | 57.2 / 23.1 / 30.2 | 52.2 / 22.0 / 26.8 |

  Reading: dropping COMET's gate buys 7-10 ms of layer 0 but costs 3-4 ms of
  layer 1 (8 connections hurt the dense reduce-scatter); at b64 4n its l0 does
  NOT beat OURS in the efficient case (21.7 vs 19.0) and ties in the skewed
  block (23.6 vs 23.1). On the schedule's lcb block right after proLaw
  (stale carried placement) overlapped COMET is the faster arm (51.0 vs 53.5
  total) — quote with that caveat only.
* Derived data: `timeline_20260905-141104.json`, `timeline_summary_20260905-141104.csv`,
  `byte_ledger_20260905-141104.csv`. CAVEAT for the COMET rows: the sidecar's
  per-source row counts are the rows this rank COMPUTES from each source
  (post-scatter GEMM rows), which for the dense allgather is not what
  crossed the wire (the allgather moves the fixed W x budget shard set,
  ~6.4 GB NIC total, spread ~1.0x by construction). So `bytes_inter` for
  COMET = compute rows attributed to remote sources, and `rows_total` spread
  = COMET's expert-compute imbalance (3.0x efficient, 3.8x skewed vs OURS
  1.17x / 1.53x). The byte table in the review page labels COMET rows
  accordingly.
* Extractor classes added for the dense path: `nvshmemi_proxy_rma_entrypoint_blocking`
  (getmem fetch) -> NIC token comm; `ep_topk_gather_rs_kernel_v2` +
  `internode_reduce_kernel` -> Top-k Reduce (GPU side lane);
  `nvshmemi_signal_wait_until_on_stream_kernel` -> wait; index/workspace
  kernels -> plan compute. 67 MB P2P copies on NVLink = the intra-node allgather.

## Capture 4 (2026-09-09, 3D scheduling — the v2 figure source; capsule 20260909-124809_perlmutter_0e64e3cc, 20/20)

Worktree `flux-3dsched` (docs/handoff/36_3d_scheduling.md is the authority), one
binary (ths_op d3bb40c7: layer-1 per-problem weight gate + in-wave moved-last,
GEMM-start marks in both fused ops). Lane = the composed 8-slot intra-node
exchange, RESET-EVERY (the oracle-basis placement is restored before every timed
iteration, so every iteration carries the full orbit; 253 global swaps/iter on
the proLaw block) — the 9/5 one-slot `t1` lane moved ~1 slot/rank and is NOT
comparable row-for-row. Arms (rows of `case_study_v2`):

| row | arm | issue point of the exchange |
|---|---|---|
| COMET, overlapped | l01_allgather_dense_nogate_c8 | — |
| swap in host gap | ablation_l01_s2_swapall_rst_3d_early_str4_p2p_r2 | place bracket (9/1 ablation; lands in the plan gap) |
| sequential swap | ablation_l01_s2_swapall_rst_3d_noov_str4_p2p_r2 | place bracket, stream waits for landing |
| 3D-scheduled swap | ablation_l01_s2_swapall_rst_3d_dual3_str4_p2p_r2 | w1 phase starts WITH the l0 GEMM (GEMM-start mark), w2 phase WITH the l1 GEMM; per-tile / per-problem gates absorb the landing |

Isolated latency, same capsule (rank-max per iteration; S-C median / mean /
proLaw block; plain median):

| row | S-C med | S-C mean | proLaw | plain |
|---|---|---|---|---|
| COMET overlapped | 52.21 | 52.63 | 55.4 | 54.91 |
| COMET gated | 55.04 | 56.37 | 63.2 | 60.09 |
| swap in host gap | 52.99 | 53.62 | 59.9 | 47.95 |
| sequential swap | 54.09 | 56.69 | 65.6 | 48.11 |
| **3D-scheduled swap** | **52.25** | **52.80** | 61.5 | **47.54** |

total_ms check vs the pre-3D point (host gap): -0.7 (S-C med), -0.8 (mean),
-0.4 (plain) — the scheme costs nothing; COMET rows within 1 ms of captures
2/3 (no drift). Timelines: `extract_timeline.py 20260909-124809 ...` ->
PSCRATCH figs_data/case_study/timeline_20260909-124809.json; swap copies are
tagged with their issue phase (early / late = w1 / l1 = w2) and the summary
carries `_swap_under_gemm_ms` / `_swap_under_nic_ms`. Figure:
`build_case_study.py <json> --rows cs3 --out figs/case_study/case_study_v2`
(the 9/5 `case_study.*` = v1, untouched).

## Scenario naming (2026-09-13 ruling) — Predictable / Drift

Figure labels: row 1 (`Efficient` in the builder) = **Predictable**, row 2
(`Skewed`) = **Drift**. Earlier labels `Predictable/Shifting` (CS_v2 9/12)
and the one-day `Specialization/Drift` (6f359ee) are retired for this lane;
the ablation keeps `Specialization/Drift` because its first scenario is a
DIFFERENT workload (below).

Definitions (postdoc wording, adopted):

* **Predictable** — placement history and evaluation come from the same
  dataset, so historical expert demand is a useful basis for placement. The
  predictability comes from that correspondence, not from the absence of a
  dataset mixture. This IS the main-experiment setup (§5.1): the figure shows
  the mechanisms under relatively balanced demand.
* **Drift** — evaluation encounters demand absent from the placement
  history, exposing imbalance and triggering expert movement. Call it a
  workload *shift*; "swap" is reserved for the system's corrective expert
  movement.

Verification (checked against the capsule spec / cells / summary, not the
labels; capsule `20260909-124809_perlmutter_0e64e3cc`, capture 4):

| | ablation "Specialization" (S-A) | case-study row 1 "Predictable" | main experiment (K2 4n b64) |
|---|---|---|---|
| family / cell id | `pools=professional_law; opool=8-pool mix` (`ablcycle_sa_prolaw_k2_4n`) | `pools=livecodebench/execution; dslots=64:32` → `trace-610042` | `trace-610042` (identical family hash on every `figs/main_perf` 4n K2 row) |
| placement history | equal-weight mix of 8 pools incl. professional_law, same layer/window | lcb decode slots [32,64) (rule-10 previous window, `oracle_slots`) | same as case-study row 1 |
| evaluation | professional_law, slots [64,96) | lcb, slots [64,96) | same |
| arm | `swapall_rp4` (reset to basis every 4th timed iteration, dwell-4 proxy) | `swapall_rst_3d_dual3_str4` (reset to basis before EVERY timed iteration, full capped orbit) | `ours_l01_s1_pv2_r2` (placement solved once at setup, no movement) |
| statistic | mean over 16 timed iters, 4 reps | one iteration (`iter4`), nsys mode | iter-max median, isolated mode |

So the postdoc's reading is correct: the case-study first row is the
main-experiment workload, not the ablation's S-A. Row 2 shares the ablation's
S-C family (7-pool history excluding professional_law, 8-topic schedule,
dwell 4; `iter33` = 2nd iteration of the professional_law block), which is
why "Drift" is the same word in both figures.

Caveat the caption must carry: the case-study arm is the reset-every proxy,
so expert movement fires in BOTH rows by construction — Predictable `iter4`:
10/16 ranks move 56–112 MB, GEMM spread 1.13x (27.1 vs 23.9 ms);
Drift `iter33`: 16/16 ranks move 56–448 MB, GEMM spread 1.91x (44.5 vs
23.3 ms); `iter5`/`iter32` reproduce both. Under the main-experiment arm
(`s1_pv2`, no swap) the Predictable row would carry no expert-comm blocks.
The green blocks in the Predictable row therefore show the swap mechanism
idling cheaply under balanced demand, not a claim that the main experiment
moves experts.

## CS_v3 (2026-09-13) — evidence fixes from the postdoc review of CS_v2

Same data as CS_v2 (capture 4, `timeline_20260909-124809.json`, rows/ranks/
iterations unchanged; `CS_v3_ranks.csv` is byte-identical to `CS_v2_ranks.csv`).
Style = the user's hand-adjusted `CS_v2_hand.drawio` (diffed against the
generated CS_v2: identical except the legend label `Plan / Meta` ->
`Plan / Metadata`; its scenario label predates the Predictable ruling).
Build: `build_case_study.py <json> --rows cs3 --template cs_v3 --out figs/case_study/CS_v3`.

1. **No device-to-device copies drawn.** The GPU lane aggregates every CUDA
   stream, so a `copy.d2d` painted over the GEMM never showed an interrupted
   GEMM (the GEMM runs on stream 7/17; the copies sit on the a2av receive
   streams and on the swap stream). The extractor classes every d2d memcpy
   as `copy.d2d` -> Token Comm. blue, which also swept up the local copies
   that install received expert weights (Predictable r1 `iter4`: two d2d on
   stream 36 at 8.18 ms right after the 58 MB `nvlink.swap` on the same
   stream). Counts in the drawn ranks: r8 10 copies / 0.94 ms (6 inside a
   GEMM), r1 14 / 1.31 ms (8 inside a GEMM). The figure claims no on-GPU
   memory-movement resource, so all of them are omitted (`DROP_D2D`); the
   extractor is unchanged and still reports them in the summary.
2. **Host bands.** The opening GPU gap is now hatched grey = **Host**:
   intervals where no device kernel or copy runs on any stream AND a host
   NVTX range (`plan.*` / `swap.*`) is open, merged when closer than 0.06 ms
   (`host_bands`). In every drawn rank the first band is
   `swap.d2h + swap.decide + swap.apply_tables + swap.prepare`
   (Predictable 0.3–1.5 ms, Drift 0.3–2.5 ms), followed by `plan.route`,
   the kstats/xchg allgather gap, and `derive_routed_meta / m_this_host /
   combine_meta_op` slivers; Drift r9 also shows two `swap.issue_l1` bands
   just before the l0 GEMM. GPU-idle gaps with no host range stay white.
   Legend gains a hatched `Host` swatch after `Wait`.

## CS_v4 (2026-09-13) — one Plan / Metadata block, and the NVLink concurrency question

Build: `build_case_study.py <json> --rows cs3 --template cs_v4 --out figs/case_study/CS_v4`
(= cs_v3 + `PLAN_MERGE`). Plan-coloured GPU-lane kernels closer than 3 ms are
drawn as one block: the pre-GEMM plan phase becomes a single bracket
(Predictable 0.14–6.28 ms, 14 kernels; Drift r9 0.12–8.18 ms, 17 kernels) and
the l0->l1 combine-plan pair a second short block (0.5–1.4 ms). Only Host
bands >= 0.3 ms are hatched on top of it, which leaves exactly one per rank:
`swap.d2h/decide/apply_tables/prepare` (Predictable 0.3–1.5 ms, Drift
0.3–2.5 ms). The merged block is a phase bracket, not kernel-busy time: it
spans the sub-0.3 ms host slivers and the NCCL allgather that runs on its
own stream. Ranks csv identical to CS_v2.

**Is the close blue/green alternation on the Drift NVLink lane serial?**
Mostly yes. Measured on the drawn ranks (`nvlink.token` >= 0.05 ms vs
`nvlink.swap`):

| rank | swap copies (streams) | swap ∩ token overlap | swap self-overlap | token Σ vs union |
|---|---|---|---|---|
| Drift r9 `iter33` | 16 on 4 streams (8 dispatch-side, 8 combine-side) | 1.76 ms over 4 pairs, of 11.9 ms swap | depth 2 in the second dispatch wave (14.7–15.8 ms); first wave back-to-back 8.76→12.23 | 24.2 vs 21.3 ms (depth 2) |
| Drift r12 `iter33` | 6 on 3 streams | 0 | depth 2 (7.68–8.43 ∥ 7.69–8.33) | 12.4 vs 10.1 ms (depth 3) |
| Predictable r1 `iter4` | 2 on 1 stream | 0 | 1 | 15.1 vs 13.6 ms (depth 2) |
| Predictable r8 `iter4` | 0 | — | — | 10.6 vs 7.1 ms (depth 3) |

So the swap copies are issued on 4 streams but the first dispatch-side wave
lands back-to-back (copy-engine / NVLink serialization), the second wave has
2-deep overlap, and only 1.8 ms of swap time on r9 coincides with a token
copy (drawn green-on-top, so that blue is hidden). The larger masking is
within blue: token copies from 3–4 receive streams overlap 2–3 deep, and the
single lane shows their union (2.3–3.5 ms less than the summed durations).
The lane is a resource-busy view, not a stream view; the caption should say
"NVLink busy" rather than imply serial issue.

## CS_v5 (2026-09-15): pv3c routing data, rank pick among swapping ranks

CS_v5 = the cs_v4 template rebuilt from the pv3c (paper-constraint router)
nsys capture, capsule `20260915-072722` (handoff 39 §13), legend "Expert
Swap". The swap decision runs on the demand histogram before routing, so
the per-rank swap copies are identical to the 9/9 capture: on the plain-lcb
(Predictable) cell ranks 1, 3, 4, 6, 9, 10, 11, 12, 13, 14 copy 2-4 expert
slots every iteration and ranks 0, 2, 5, 7, 8, 15 copy none. The
longest/shortest-GEMM pick is blind to that: CS_v4 landed on r8 (silent) +
r1 (2 copies); the first CS_v5 cut landed on r5 + r7 (both silent), which
is why no green appeared there. `--prefer-swapping` restricts the pick to
ranks that swapped in the drawn iteration: Predictable = r9 / r1, Skewed =
r9 / r13 (unchanged; iter33 r9 copies 16 slots, 10.8 ms). Build:
`build_case_study.py <json> --out figs/case_study/CS_v5 --rows cs3
--template cs_v4 --variant-suffix _pv3c_eps025 --prefer-swapping`.

### CS_v5 row labels (2026-09-15, user ruling)

The vertical row labels name the traffic, matching the ablation's group
labels: top = **LiveCodeBench** (the plain steady workload, `trace-610042`,
iter4; the main-perf workload), bottom = **Prof. law rotating** (the
professional-law block of the 8-topic rotation, `trace-b549f7`, iter33), set
on two centred lines. Predictable / Drift (2026-09-13) remain the prose
names for the two scenarios. Same data, ranks and geometry as the CS_v5
cut above; `SCENARIO_NAME` in the builder, which now accepts `\n` in a
vertical label (SVG tspans, draw.io `<br>` with `align=center`).
