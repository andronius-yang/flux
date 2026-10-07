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

### CS_v5 row order (2026-09-15, user ruling)

The Prof. law rotating row is drawn ABOVE the LiveCodeBench row
(`--skewed-first`): the text discusses the drift case first and closes on
the balanced case. Ranks, data and geometry unchanged. The section text that
goes with the figure is drafted in `case_study_section_draft.md`. Build:
`build_case_study.py <json> --out figs/case_study/CS_v5 --rows cs3
--template cs_v4 --variant-suffix _pv3c_eps025 --prefer-swapping --skewed-first`.

## CS_v6 (2026-10-06): the swap decision on the GPU (no host wait)

User directive: the paper figure must not show a host wait; follow the serving path's planning
changes, no mechanism change, minimal work. Measured on CS_v5's data first: the device was idle
2.1–3.1 ms before the layer-0 GEMM on every drawn rank, of which 1.2–2.2 ms was ONE gap = the host
swap chain (`swap.d2h` + numpy orbit + table upload + prepare), 0.2 ms the metadata sync
(`derive_routed_meta`), the rest 0.06–0.08 ms slivers. So the minimal port is the decision only.

**Change (branch `cs6-devswap`, worktree `$PSCRATCH/workspace/andrewy/flux-cs6`, libflux unchanged
= the flux-pv3 build 4ab20ff4/ddc8682e):** `python/flux/testing/_swap_decide_ext.cu` = the serving
path's `swap_decide_kernel` (Zepp sglang-dev `src/planner/swap_decide.cu`) verbatim as a JIT
extension (registry/verdict glue removed, pad correction a no-op with ntok = S). Runner
`--swap_decide device`: after the loads all-gather the kernel rewrites the planner's p2l / l2p in
place (the router runs on the swapped tables in the same iteration), its result block rides a
non-blocking D2H that the existing planning sync completes, and the unchanged 3D-scheduled lane
(dual3: w1 under the l0 GEMM, w2 under the l1 GEMM, 4 streams) is armed from the pull lists after
that sync. Swap policy = CS_v5's (user ruling 10-06: "force" so both phases show): reset to the
oracle placement before every timed iteration and NO band test (C = -1 puts every node out of band
= the tau=1 orbit). Under the paper's band rule at C = 0.25 LiveCodeBench never swaps (worst node
max/mean 1.09); professional law does (node 2 at 1.98). Arms `..._pv3c_eps025_dsd` (+ `_dsd_gate`),
specs `cs6_*.yaml`.

**Gate (capsule 20261007-023406):** the device decision equals the host orbit bit for bit in every
checked iteration (moves, p2l, device l2p / p2l; LiveCodeBench 10/10 iterations, 12 moves each;
schedule 2/2). LiveCodeBench torch-reference check 160/160 rank-iterations OK. The schedule cell
hit 1 bad row on ranks 8 and 13 in WARM-UP iteration 1 (carried swaps across a topic switch; the
host-decision arm was never gated on this family). Note: outputs are ~1e-3 vs atol 1e-2, so this
gate only catches garbage rows, and under a topic schedule its reference used topic 0 (fixed 10-06:
reference = the iteration's topic). Attribution run: capsule 20261007-032214 (host vs device,
non-asserting) — RESOLVED: with the topic-correct reference BOTH arms pass 160/160 rank-iterations
on the schedule (max |out - ref| 6.1e-5 vs |ref| <= 7.0e-3 = bf16 rounding), device decision 10/10
equal to the host orbit. The 023406 bad row was the old wrong-topic reference: outputs reach ~7e-3,
so two opposite-sign outputs can differ by more than atol 1e-2 (rare, one row).

**Capture (capsule 20261007-025748, 8/8 ok, one binary):** `_dsd` and the host-decision arm, nsys +
isolated, CS_v5 recipe. Isolated (median over iterations of the max over ranks, ms):

| workload | decision | plan_comm | place | plan | e2e | total |
|---|---|---|---|---|---|---|
| LiveCodeBench (32 it) | host | 0.18 | 1.22 | 1.96 | 44.91 | 48.16 |
| LiveCodeBench (32 it) | device | 0.19 | 0.11 | 1.91 | 45.31 | 47.42 |
| prof. law block (4 it) | host | 0.18 | 2.44 | 2.20 | 55.20 | 59.70 |
| prof. law block (4 it) | device | 0.18 | 0.22 | 2.00 | 55.61 | 57.94 |

nsys (drawn ranks): device idle before the l0 GEMM 0.85–0.92 ms (was 2.1–3.1 on CS_v5): 0.20 ms
metadata sync, ~0.27 ms `swap.prepare` (pull-list parse + lane arming, now after that sync), rest
slivers; none >= 0.3 ms, so no Host band is drawn and the legend drops "Host". The l0 GEMM starts at
4.9–5.8 ms (device) vs 6.1–8.1 ms (host) in every captured iteration; iteration END varies 2–3 ms
with the combine-egress tail independently of the decision.

Builds (from the flux-cs6 tree; JSON = `figs_data/case_study/timeline_20261007-025748_perlmutter_a694f55d.json`):
- `CS_v6`: `--rows cs3 --template cs_v4 --variant-suffix _pv3c_eps025_dsd --prefer-swapping --skewed-first`
  (iter33 / iter4 as CS_v5; drawn ranks prof. law r11 / r12, LiveCodeBench r4 / r3). Iter33 is the
  slowest iteration of its block on this arm (61.2 ms vs 57.7–58.4 for iter32/34/35).
- `CS_v6_law35`: same + `--skew-iter iter35` (57.7 ms, closest to the isolated block median 57.9).
- `CS_v6_hostdecide`: `--variant-suffix _pv3c_eps025` = the same-binary host-decision "before".

## CS_v7 (2026-10-06): one MoE layer on the serving path (full device offload)

User ruling 10-06: the case study shows ONE MoE layer as it runs in serving, with every planning step and
every communication issue on the device (no host read of a routing count inside the forward); the case-study
text names it "the case study of one MoE layer". Research-tree only: the library is used as an engine, nothing
is committed to the release repository.

**Engine:** the private lopep `p10-f` build (`$PSCRATCH/workspace/andrewy/lopep_p10f`, 3d83066 = the serving
defaults Zepp `sglang-dev` was pruned from) with its debug timestamp instruments on (`LOPEP_WIRE_TRACE=1`,
`LOPEP_PACK_TRACE=1`; Zepp's release build has none). Per step on device: step head, loads all-gather, swap
decision, lane arm, pad rebuild, fused router, routing all-gather, planner tail, fused metadata front, dispatch
plan block, demands, arena; dispatch = pack-push kernel (own-node rows into the node peers over NVLink), relay
kernel, one-warp wire kernel (non-blocking puts in groups of 2 + quiet + signals), gateway forward kernel;
staged swap lane (push W1 / W2 into the destination's staging over NVLink, the GEMMs read the staging behind
their gates, commit staging -> slot after each GEMM); combine pack / pre-reduce / device combine wire /
bucket reduce; the forward's capacity verdict = one NCCL all-reduce at its end (drawn as Wait). Eager steps
(the 4680-token bucket is above the graph limit).

**Driver** `serving_case_study.py` (+ inputs from `export_serving_inputs.py`: the research oracle placement,
the routing files and topic schedule, gate weights; K2 shape with the case-study arm's one-matrix GELU expert,
ffn 2048): one layer per forward (begin / step / end, verdict after the `iter<i>` NVTX range), isolated
(sync + barrier), swap forced as CS_v5 / CS_v6 (decision band -1 after the serving warm-up, oracle placement +
slot weights reset before every timed iteration), capacities = provable bounds over every topic and its forced
orbit (scale 1; an x2 run died on GPU memory), heap 12 GiB (S-C) / 8 GiB (LCB).
Capture `$PSCRATCH/workspace/andrewy/sweep_data/cs7_20261006-213442` (job 59468509, 4n): `sc_check` (2+8
iterations, PyTorch reference every iteration: 0 bad rows, 22-43 moves per iteration, no verdict abort),
`sc_nsys` (5+32) and `lcb_nsys` (3+3) with the traces, `sc_time` / `lcb_time` (5+32, no profiler, no traces).

**Lanes** (`extract_serving_timeline.py`): NIC = wire-trace windows (dispatch round: put issued -> its group's
quiet + signals returned; combine position: pre-reduce flag seen -> returned); NVLink = pack-push spans,
gateway-forward windows (block 0), the combine pack's per-block push windows, swap pushes (`lane_push_kernel`
>= 0.05 ms; shorter launches are ranks with nothing to send); GPU = kernels by name. NOT drawn: the relay's
NVLink pulls (no timestamps; its span is mostly waiting) and every spin-dominated span (wire, combine wire,
relay, forward, lane wait, lane commit, whose copy is local). Stamps aligned to nsys per (rank, iteration) by
the stamped kernels' entry stamps: the two anchors of one GPU agree to 0.43 us median (28 us max).

**Timings** (no profiler, step = begin_forward -> end_forward event bracket incl. the verdict all-reduce, median
over iterations of the max over ranks): LiveCodeBench 48.23 ms; S-C schedule 52.61 ms, prof. law block 59.59 ms.
nsys spans of the drawn iterations: prof. law iter32-35 55.3-57.7 ms (iter33 55.9), LiveCodeBench iter3-5
48.4-48.7 (iter4 48.4); the layer-0 GEMM starts at 3.2-5.4 ms (CS_v6 4.9-5.8, CS_v5 6-8).

Build: `build_case_study.py figs_data/case_study/timeline_cs7.json --out figs/case_study/CS_v7 --rows cs3
--template cs_v4 --variant-suffix _serving --prefer-swapping --skewed-first` (drawn ranks: prof. law r8 / r15,
LiveCodeBench r9 / r10). Text changes vs CS_v6 for the section: the swap shows as the staged lane's pushes of
both matrices early in the step (the commits are local and not drawn); the dispatch wire is issued by kernels;
the iteration ends with the forward's capacity verdict (Wait).

## CS_v8 (2026-10-06): the serving path with a device dual3 lane (research tree only)

User directive: "for the research tree only, do a dual3 implementation, but ALL ON GPU, with no additional
latency". CS_v7's serving path swaps through the staged lane (both matrices pushed before the dispatch GEMM);
CS_v8 replaces it with a device-driven 3D (dual3) schedule, installed into the unmodified lopep p10-f runtime by
the driver (`--lane dual3`):

- `python/flux/testing/serving_dual3.py` + `_dual3_ext.cu`: phase 0 = `push_w1_kernel`, the sender pushes W1
  of every moved slot into the receiver's staging under its own dispatch GEMM; phase 1 = `pull_w2_kernel`, the
  receiver pulls W2 from the sender's slot (W2 slots allocated on the symmetric heap) into its staging under its
  own combine GEMM. The GEMMs read the moved experts from the staging behind the per-slot gate words with the
  moved-last schedule (the staged lane's `lane_arm` / overrides / `lane_commit` are reused); the W1 commit waits
  until its pushes raised the receivers' gates (`wait_pushed`), the W2 commit until the receivers acknowledged
  their pulls (`wait_acks`). All driven by the device decision block: no host read, no host-issued copy.
- Why a pull for W2: a sender-side W2 push gated on the sender's combine GEMM makes the receiver's gated combine
  GEMM depend on the sender's dispatch-phase waits (the serving W2 cycle of 2026-10-01, which W2_EARLY fixed).
- Hang found and fixed on the way: the first version parked `cuStreamWaitValue64` waits on the GEMM-start marks
  on side streams; the runtime has more streams than hardware queues, so a parked wait can block the queue that
  carries the forward stream's later mark write (both first runs hung in the first step). Now the phase kernels
  spin on the marks themselves and every join is a device word (no side-stream waits or event joins).
- Every kernel launched beside a spinning GEMM fits rule R5 (push_w1 40 / pull_w2 38 registers x 512 threads,
  waits one warp).

Capture `$PSCRATCH/workspace/andrewy/sweep_data/cs8_20261006-222934` (job 59470763, 4n): `d3_sc_check` (1+3,
PyTorch reference every iteration: 64/64 rank-iterations OK, max |out - ref| 6.1e-5 vs |ref| <= 7.5e-3, 43/43/43/22
moves), `sc_nsys` / `lcb_nsys` (dual3 + wire / pack traces), and same-allocation timing (no profiler, median over
iterations of the max over ranks):

| workload | staged | dual3 |
|---|---|---|
| S-C schedule, 32 iterations | 50.50 ms | 49.43 ms |
| topics 0-6 (light swaps) | 49.4-52.4 | 47.9-52.4 (equal or lower) |
| prof. law block (4 it, 43 moves) | 61.94 (63.35 63.76 60.52 56.69) | 63.07 (63.11 66.19 63.03 57.47) |
| LiveCodeBench, 32 iterations | 47.13 ms | 45.71 ms |

The prof. law block difference (+1.1 ms median) is inside its 7-10 ms iteration spread; everywhere else dual3 is
equal or faster. Extraction: `push_w1_kernel` / `pull_w2_kernel` = Expert Swap, their spans trimmed to start at
the GEMM they wait for (the kernels are launched before it and spin on its start mark); the waits are not drawn.
Build: `build_case_study.py figs_data/case_study/timeline_cs8.json --out figs/case_study/CS_v8 --rows cs3
--template cs_v4 --variant-suffix _serving --prefer-swapping --skewed-first` (prof. law iter33 = 65.8 ms, its block
64.9-65.8 under nsys; LiveCodeBench iter4 47.0; ranks r9 / r0, r13 / r3): green at both GEMM starts (W1 under the
dispatch GEMM, W2 under the combine GEMM) on every drawn rank.
