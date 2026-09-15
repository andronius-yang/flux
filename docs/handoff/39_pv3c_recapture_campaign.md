# Handoff 39 — pv3c recapture campaign brief (2026-09-15)

User directive: recapture EVERY graphed cell that the routing change
affects — main-perf "Ours" rows, weak-scaling Ours points, the ablation
figure's OURS arms, the cycling-ablation OURS arms, the case-study OURS
arms — under the paper-constraint router **pv3c** (handoff 38), NOT the
baselines. At 8n/16n/32n the slack C is an experiment (C = 1/4 and 1/2
both run; the orchestrator picks per topology). Results are NEW data
("pv3c fixed-constraint data"), never overwriting `figure_src.csv`.

Authority for what pv3c is: `docs/handoff/38_pv3_routing.md`.
Worktree (ALL lanes run from here, nothing else): 
`/pscratch/sd/y/yufeid/workspace/andrewy/flux-pv3` (branch pv3).
Env (source before anything): `$PSCRATCH/workspace/andrewy/pv3_env.sh`
(cd's into the worktree, module.sh + CUDA 12.4 pin + PYTHONPATH).
Binary: main's libs linked into the worktree (`python/flux/lib/*.so`,
sha16 a0c60c75 / d3bb40c7) + the pv3 JIT extension already built at
`$PSCRATCH/workspace/andrewy/pv3_ext_build` (compute ranks load it; it
must never be rebuilt during the campaign).
Status bus (append one line per event, lane-prefixed):
`$PSCRATCH/workspace/andrewy/logs/pv3/recapture_status.txt`
Run logs: `$PSCRATCH/workspace/andrewy/logs/pv3/run_<spec>.log`
Capsules land in `sweeps/results/runs/<run_id>/` (NEVER commit).

## 1. Lanes and their specs (all under `sweeps/specs/`)

| lane | nodes | specs (run in this order) | est. |
|---|---|---|---|
| lane8 | 8 | `pv3c_mp_8n_k2`, `pv3c_mp_8n_qwen`, `pv3c_weak_8n_k2_b1`, `pv3c_weak_8n_k2_b64` | 27+27+6+3 cells, ~50 min |
| lane16 | 16 | `pv3c_mp_16n_k2`, `pv3c_mp_16n_qwen`, `pv3c_weak_16n_k2_b1`, `pv3c_weak_16n_k2_b64` | 36+36+6+3 cells, ~70 min |
| lane32 | 32 | `sbatch $PSCRATCH/workspace/andrewy/sbatch_pv3c_weak32.sh` (runs `pv3c_weak_32n_k2_b1`, `pv3c_weak_32n_k2_b64` inside the batch job) | 9 cells, queue-bound |
| lane4A | 4 (+2) | `pv3c_abl_loo_k2_4n` x5 runs, `pv3c_abl_matched_k2_4n` x5 runs, then `pv3c_weak_2n_k2` on a 2-node grant | 60 + 6 cells, ~2 h |
| lane4B | 4 | `pv3c_cyc_sa_k2_4n` (112 cells, iters 16), then `pv3c_cyc_sc_d4_k2_4n` x3 runs | ~3.5 h |
| lane4C | 4 | `pv3c_cs_nsys_k2_4n` (nsys+isolated), `pv3c_cs_iso_twins_k2_4n` | 18 cells, ~1 h |

Every spec pairs each OURS arm with its `_pv3c_eps025` twin (and
`_pv3c_eps05` at 8n/16n/32n) inside ONE capsule — the LocCap twin is
the in-capsule comparator (it is an "Ours" arm, not a baseline).

## 2. Invocation (exactly this, nothing else)

```
source $PSCRATCH/workspace/andrewy/pv3_env.sh
python3 sweeps/sweep.py run --spec sweeps/specs/<spec>.yaml --jobid <JOB> \
    > $PSCRATCH/workspace/andrewy/logs/pv3/run_<spec>[_repN].log 2>&1
```
Always `--jobid` explicitly (the runner's autodetect can grab another
session's job). For repeated runs (reps) just rerun the same spec; each
run is its own capsule. `--dry-run` first if anything looks off.

## 3. Allocation policy

- 8n: `salloc -q regular -C gpu -A m5424_g -N 8 --gpus-per-node=4 --no-shell -t 01:30:00`
  (fallback if regular sits > 60 min: `-q debug -t 00:30:00` and run one
  spec per debug grant, 8 nodes max on debug). 
- 16n: `salloc -q regular -C gpu -A m5424_g -N 16 --gpus-per-node=4 --no-shell -t 02:00:00`.
- 32n: sbatch only (user ruling for 32n), the script above; then
  `squeue --me -n pv3c_weak32` to watch; its .out lands in
  `$PSCRATCH/workspace/andrewy/logs/slurm_32n/`.
- 4n / 2n: `salloc -q interactive -C gpu -A m5424_g -N 4 --gpus-per-node=4 --no-shell -t 03:30:00`
  (2n: `-N 2 -t 00:40:00`). Interactive allows 2 concurrent 4-node jobs
  per user: if salloc is rejected for the cap, wait 5 min and retry.
- Accounts: `m5424_g` (m5350_g user balance is 0; m4243_g exhausted).
- Run salloc in the background with its stdout to a file and poll for
  `Granted job allocation <id>` — take the jobid ONLY from that file.
  `until grep -q "ready for job" <file>; do sleep 10; done` before srun.
- **scancel your job the moment your last spec ends** (idle GPU burn is
  the cardinal sin here); never scancel a job you did not grant.

## 4. Hard rules

1. No code edits, no rebuilds, no commits, no git operations in the
   worktree. If a cell fails, record it and move on; do NOT debug.
2. Do not run anything on login nodes except the runner itself.
3. Never `pkill -f` / `pgrep -f` with a pattern that appears in your own
   command line (it kills your own shell). Kill by pid from `pgrep -u
   $USER -f "specs/<spec-name-with-[b]racket-trick>"` only.
4. Watch budget: a cell has ~60 s per iteration; if a run log is silent
   for longer than the spec's `idle_timeout_s`, the runner reaps it — do
   not add your own kills unless the runner itself hangs.
5. Never touch jobs named mig5n*, agl_*, muxopd or any job you did not
   create (other projects on the same account).
6. Status bus line at EVERY event: `<lane> GRANT <jobid> <nodes>n`,
   `<lane> START <spec>`, `<lane> END <spec> rc=<rc> <cells: a/b ok>`,
   `<lane> CELL <cell_id> <status> <secs>` for non-ok cells,
   `<lane> RELEASE <jobid>`, `<lane> DONE`.
7. All persistent logs under `$PSCRATCH/workspace/andrewy/logs/pv3/`.

## 5. Sanity anchors (same statistic: per-iteration max over ranks, median over iterations)

4n K2 (capsules 20260915-053358/-055811): LocCap 4.02/6.29/14.70 ms at
b1/4/16, pv3c C=1/4 4.08/6.11/14.47. 4n Qwen: 3.04/4.75/12.12 vs
3.08/4.63/12.07. Existing 8n/16n Ours (figure_src): K2 ours12 8n
5.86/8.37/19.56, 16n 12.61/15.63/27.92; Qwen 8n 5.41/7.38/17.18, 16n
12.33/14.68/26.06. A pv3c cell more than ~5% above its in-capsule LocCap
twin is worth a `<lane> NOTE` line but is still a valid data point.

## 6. Known risks

- 8n/16n Qwen llc: LocCap once hit "recv bound violated" at 16n Qwen
  (handoff 15); pv3c sizes provably — if a pv3c cell dies at startup on
  FLUX_CHECK / recv overflow, record it (capsule status `failed`).
- dwire b64 and 16n b64 in general: symmetric heap (16G) — the specs
  avoid dwire b64; `pv3c_weak_16n_k2_b64` / `_32n_` may die at
  NVSHMEM_MALLOC on the pv3c arms (larger provable recv caps): record it.
- dual3 b16 wedge (main-branch arm bug) — not in these specs (b64 only).
- 32n: the LocCap dwire b1 cell previously "hung at teardown" after a
  complete timed loop (status timeout, numbers valid) — expect the same.
- Any `stuck` cell holds the nodes until the runner reaps it (timeout_s).

## 7. Lane final report (post to the orchestrator; also append `<lane> DONE`)

For each spec: run_id, cells ok/total, list of non-ok cells with the
one-line reason from the rank log, jobid(s) with granted/used minutes.
Nothing else — the orchestrator does all aggregation and the on-par
verdicts.

## 8. Campaign log (orchestrator)

- 00:26 six lanes spawned; 4A/4C granted at once, 4B waited on the 2-job
  interactive cap, 8n/16n regular + 32n sbatch queued on priority.
- lane4C DONE 01:11 — case study 12/12 nsys+isolated + 6/6 LocCap twins
  (capsules 20260915-072722, -080408), job 58342303 used 44/120 min.
- lane4A DONE 01:25 — ablation LOO 5 reps + matched 5 reps (60/60,
  capsules -072710…-081441) + weak 2n 6/6 (-082011); jobs 58342302 (53/180
  min), 58344044 (5/40 min).
- lane8 DONE 02:15 — regular never granted in 60 min; ran on two debug
  grants (58344290 21.7/30 min, 58345222 13.5/30 min): mp_8n_k2 27/27
  (-084021), mp_8n_qwen 27/27 (-090203), weak_8n b1 6/6 (-085521), b64
  3/3 (-085807), llc b1 repeat 3/3 (-085958; the -084021 C=1/4 llc b1 cell
  read +24.6% with equal lane brackets = stall, superseded by the repeat).
- Aggregator: `39_pv3c_recapture_aggregate.py` → figure_src_pv3c.csv
  (main_perf, weak_scaling), ablation_*_pv3c.csv, results_tidy_pv3c.csv
  (cycling), case_study/pv3c_capsules.csv, and
  `39_pv3c_recapture_verdict.md` (Δ vs in-capsule twin and vs the plotted
  figure value per cell; C choice per topology by mean Δ vs twin).
- lane16 (job 58342301, regular, granted 02:21): mp_16n_k2 35/36
  (-092116; dwire C=1/2 b16 NVSHMEM_MALLOC — the 16G heap class), mp_16n_qwen
  35/36 (-094347; same dwire C=1/2 b16 failure). 16n verdict: K2 within a few
  % (pv3c trails +2..+6% at b16 on fused/s2/llc; dwire, the plotted 16n b1
  winner, +2.4% b1 / −7.1% b16 at C=1/4); Qwen b16 NOT on par at C=1/4
  (s1 +13.9%, llc +11.5%), C=1/2 partial (llc +6.6%), and two cells hit the
  known ~350 ms Qwen-16n-b16 stall (LocCap s2, pv3c s1 C=1/2). Added the
  `_pv3c_eps1` twins + `pv3c_mp_16n_qwen_b16_ladder.yaml` (LocCap, 1/2, 1 on
  s1/s2/llc at b16) to lane16's queue behind the weak specs.
- lane16 cont. (03:10-03:20): weak_16n_k2_b1 6/6 (-100611), weak_16n_k2_b64
  3/3 (-101015: pv3c C=1/4 +10.5% / C=1/2 +5.1% vs twin — l0 and l1 wire),
  Qwen b16 ladder 9/9 ok (-101317) but 3 of 9 cells in the ~350 ms l1 stall
  class (LocCap s1 80 ms, pv3c C=1 s1 362 ms, pv3c C=1/2 s2 192 ms); clean
  cells: s1 C=1/2 28.17 (+3.3% vs the -094347 twin), s2 C=1 28.64 (+5.3%),
  llc C=1/2 35.69 (+4.3%), llc C=1 37.64 (+10%: C=1 lets the combine
  imbalance grow, not monotone). Aggregator now: `_pv3c_eps1` parsed as C=1;
  stall rule (iter-max median > 2x the plotted value -> excluded + listed);
  twin fallback to the campaign's clean LocCap cells of the same cell
  (flagged `xcapN`); C rule PRE-REGISTERED 03:30 before the K2 b16 ladder:
  smallest mean Δ vs twin, ties within 1 pp broken by the smaller worst case;
  `39_pv3c_chosen_dataset.csv` = one value per plotted cell at the chosen C.
  Follow-up (NOT changed mid-campaign): in-driver plan bracket at 16n is
  +0.2-0.5 ms under pv3c (b1: 1.01 -> 1.25 C=1/4 / 1.53 C=1/2) although the
  kernel h2h was par — host side of route_pv3c at 16n to be profiled.
- lane16 DONE 03:28 (job 58342301 used 68/120 min): K2 b16 ladder 9/9 ok
  (-102031): at b16 C=1 beats C=1/2 on all three lanes (s1 +1.7 vs +2.7%,
  s2 -1.0 vs +1.5%, llc -1.6 vs +0.8% vs twin) but C=1 covers only 5/27
  16n cells and is the weakest bound ([0, 2q]) -> not a candidate.
  C RULE (refined 03:40 for coverage, paired comparison as originally
  intended): 2n/4n/8n -> C=1/4; 16n -> C=1/2 (paired mean +2.5% / worst
  +7.9% vs C=1/4 +3.5% / +16.1%). 16n VERDICT: pv3c C=1/2 trails LocCap by
  +2.5% on average (fused b1..b16 -0.1..+3.5%, Qwen dispatch b16 +6.4%, llc
  K2 b1 +7.9%); dwire on par or better (b16 -7..-9%). NOT strictly on par
  at 16n; 2n-8n on par or better. In-driver plan bracket overhead: 8n
  +0.03..0.14 ms, 16n +0.25..0.5 ms (follow-up).

- lane4B DONE 04:06 — S-A 112/112 (-081501), S-C dwell-4 reps 1-3 36/36
  (-093801, -100726, -103719); jobs 58343908 (84/230 min) + 58346876
  (89/150 min) released. All 4n/8n/16n lanes complete; only lane32
  (sbatch 58342300, PD Priority since 00:30) outstanding.

- lane32 DONE 06:27 (sbatch 58342300: queued 349 min, ran 7 min): weak_32n
  b1 4/6 (-131546; both dwire pv3c cells NVSHMEM_MALLOC on the 4G dwire heap),
  weak_32n b64 3/3 (-131929). 32n VERDICT: fused b1 on par (+1.2/+1.3% vs
  twin, -1.4% vs plotted; the +0.8-1.1 ms is the plan bracket, l0/l1 par),
  b64 NOT on par (+9.3% C=1/2, +10.1% C=1/4 vs twin; +5.9% vs plotted; l0
  +5.5-8.5 ms, l1 +2.3-5.2 ms wire). C rule -> 32n C=1/2.
  RCA dwire 32n failure (offline replay + recorded `dwire_max_split`): the
  test's pv3c pair cap = pair_ub.max (42) + 8W (1024) and the shared cushion
  adds ext_dst (335/586) + 8W again -> All2AllSingle max_split 2537/2678 vs
  LocCap's 1535 (recorded) on the same 4G heap -> alloc-only sizing artifact
  of the test at W=128 (fix post-campaign: pair cap = pair_ub + ext_dst +
  small constant). Rerun submitted 06:52 as sbatch 58357412
  (`pv3c_weak_32n_k2_b1_dwire_rerun.yaml`: dwire LocCap + pv3c 1/4 + 1/2,
  `sym_size_override_g: 12`, 25 min wall) — status bus prefix `lane32r`.

- lane32r DONE 07:41 (sbatch 58357412: queued 47 min, ran 2 min): dwire b1
  rerun 3/3 ok (-143919) with the 12G heap; recorded max_split LocCap 1535 /
  pv3c 2425 (C=1/4) / 2678 (C=1/2) confirms the sizing RCA. dwire 32n b1:
  pv3c +6.8% (C=1/4) / +9.5% (C=1/2) vs twin — ENTIRELY the plan bracket
  (1.71 -> 2.44 / 2.90 ms); l0 and l1 are lower under pv3c (4.35 -> 4.09,
  2.60 -> 2.49). Same on the fused 32n b1 cell (+0.8-1.1 ms plan). The
  in-driver plan overhead grows with W (8n +0.03-0.14, 16n +0.25-0.5, 32n
  +0.7-1.2 ms) while the kernel h2h said +0.15 ms at 32n -> the first
  post-campaign item. ALL LANES COMPLETE 07:41; 57 capsules; no jobs held.

## 9. Figure readiness — the pv3c fixed-constraint datasets (2026-09-15 07:45, final)

Nothing below overrides a plotted dataset: every file is a `*_pv3c.*` twin
next to the plotted one, carrying `capsule`/`cell_id` provenance and the same
statistic (per-iteration max across ranks, median over the 10 timed
isolated iterations; ablation rows keep their it0 / rest-mean convention;
cycling rows keep the per-cell median + full series). Router = pv3c with
C = 1/4 at 2/4/8 nodes and C = 1/2 at 16 and 32 nodes (rule in §8); 16n/32n rows re-measured on kernel v4/v4.1 (§12). Comparator for "on par" = the LocCap twin
measured in the same capsule on the same binary (Δtwin); Δ vs the plotted
value (Δfig) carries binary drift and is informational.

| figure | pv3c dataset | cells | verdict (Δtwin, chosen C) | status |
|---|---|---|---|---|
| main_perf (rows ours12, ours12_dispatch, ours2_nooverlap, ours2_direct) | `figs/main_perf/figure_src_pv3c.csv` (all C's) + `docs/handoff/39_pv3c_chosen_dataset.csv` (one value per plotted cell at the chosen C) | 4n 24/24, 8n 19/19 (dwire only b1 at 8n, as plotted), 16n 24/24 (dwire b16 filled from C=1/4: the C=1/2 arm dies at NVSHMEM_MALLOC) | 4n mean −0.9% (−5.0..+4.4), 8n −0.9% (−6.3..+4.9), 16n +0.1% on kernel v4 (26/27 cells v4; −12.7..+7.0; only Qwen b16 fused +7.0 (campaign row: the v4 attempt hit the ~350 ms combine stall) and Qwen b16 swap +5.7 (v4, l0/l1 wire) above +4%) | READY 4n/8n/16n (Qwen 16n b16 = wire caveat) |
| weak_scaling (ours + dwire, K2, b1/b64) | `figs/weak_scaling/figure_src_pv3c.csv` + chosen dataset | 2n 2/2, 4n 3/3, 8n 3/3, 16n 3/3, 32n 3/3 (dwire b1 from the 12G-heap rerun -143919) | 2n −1.1%, 4n +0.5%, 8n −1.4%, 16n +0.2% on v4 (b1 −1.7/−1.5, b64 +3.7), 32n +1.5% on v4 (ours b1 −0.6, dwire b1 −3.4, ours b64 +8.6 = wire) | READY 2–32n (b64 at 16–32n = wire caveat) |
| ablation (LOO + matched, K2 4n b64, 3 arms × 5 reps) | `figs/ablation/ablation_iter_tidy_pv3c.csv`, `figs/ablation/ablation_tables_pv3c.csv` | 60/60 | it0 / rest-mean within ±1 ms of LocCap on every arm (LOO placement_swap_seq it0 67.8 vs 71.2 — pv3c lower, sd 1.0 vs 4.3) | READY |
| ablation_cycling (S-A seen-8 per topic, 7 arms; S-C LOO-proLaw dwell-4, 6 arms × 3 reps) | `figs/ablation_cycling/results_tidy_pv3c.csv` | S-A 112/112; S-C 36/36 (3 reps) | S-A per-arm mean over topics −0.9..+1.3%; S-C 3-rep means per arm −1.4..+1.0% (rep-3 slow-wall cell's timed iterations in line with reps 1–2) | READY |
| case_study (dual3 / early / noov 3D-str4 arms, plain lcb + S-C dwell-4, nsys + isolated) | `figs/case_study/pv3c_capsules.csv` (nsys paths + isolated twins) | 12 nsys+iso pv3c cells + 6 LocCap iso twins | isolated totals within ±1 ms of the LocCap twins (dual3 plain 49.4 vs 48.7; S-C 52.2 vs 51.8) | READY (extractor re-run on the pv3c nsys reps is the figure lane's step) |

Excluded/failed cells (recorded, not silently dropped): 16n dwire pv3c C=1/2
b16 (K2 + Qwen) NVSHMEM_MALLOC; stall-class cells listed in the verdict
file (8n llc b1 C=1/4 in -084021; 16n Qwen b16 in -094347 and -101317);
dual3 b16 wedge is a pre-existing arm bug (not in these specs).

## 10. Post-campaign follow-ups (code changes deferred so the 57 capsules share one tree)

1. **In-driver plan bracket under pv3c grows with W** (8n +0.03..0.14 ms,
   16n +0.25..0.5, 32n +0.7..1.2 ms; C=1/2 slower than C=1/4) although the
   1-GPU kernel head-to-head read +0.15 ms at 32n. The plan graph
   (FLUX_OURS_PLAN_GRAPH) captures only the post-allgather tail for BOTH
   routers, so the route kernels run eagerly in both; the difference is in
   the five pv3c launches themselves (tables kernel = one thread per expert
   walking R=W rounds; budget kernel = G x kMaxRep(64) blocks; vacate over
   S tokens x NN nodes) under real placements. Profile `plan.route_pv3` with
   nsys at 16n/32n; candidates: warp-per-expert tables kernel, budget grid
   = G x c_e, vacate early-out for tokens with <= 1 remote node. This is the
   whole 32n b1 gap (dwire +0.74 ms, fused +0.8 ms; l0/l1 are par or better).
2. **Test pair cap for pv3 routes** (`test_moe_ours_traffic.py` pv3 sizing
   block): `pair_cap = pair_ub + 8W` and `cushion = ext_dst + 8W` stack two
   8W terms (2048 rows at W=128) into the dwire All2AllSingle width; use
   pair_ub + ext_dst + a small constant (the realized pair max is 40-44 rows
   at 32n b1). Until then dwire pv3c cells at 32n need
   `sym_size_override_g: 12`.
3. **16n/32n large-budget wire gap** (b16/b64: +2..+9% vs LocCap at C=1/2)
   is structural token-node incidence under the per-replica band; C=1 does
   not close it (llc worse). Options if the paper needs parity there: a
   node-cover-aware take order inside the rotation (handoff 38 §8), or
   quoting pv3c with the band as the constraint-faithful router and LocCap
   as the unconstrained upper bound.
4. Two Qwen-16n-b16 stall classes remain (LocCap and pv3c alike, l1 ~340
   ms): unrelated to routing, pre-existing (handoff 30 open item).

## 11. Follow-up 1 in progress — the pv3c route CUDA graph (2026-09-15 08:20)

Implementation (`python/flux/testing/ours.py`, knob `FLUX_OURS_ROUTE_GRAPH`,
default ON when `FLUX_OURS_PLAN_GRAPH=1` and the route rule is pv3/pv3c):
`derive()` is split into `_route_and_pack` (route kernels + kstats D2H +
exchange pack, all on persistent buffers) and `_exchange_and_tail`;
`prime_graphs(d_gather_buf)` fills the demand buffer with a valid histogram
from the setup topk and captures `_route_and_pack` once (graph-pool
phys/kstats kept alive; consumers read `_xchg_send`/`_kstats_pinned`);
`derive()` replays when the buffer identity matches; `refresh_placement`
updates l2p/lcnts/p2l IN PLACE while a graph holds them (SwapTableSync
already did). LocCap is never graphed (f_cap-retry host sync). Test:
`planner.prime_graphs(d_gather_buf)`. Arms `*_pv3c_eps025_nrg` /
`*_pv3c_eps05_nrg` (s1 + dwire) pin the graph OFF for A/Bs.

Gate (job 58362597, capsule 20260915-151133): s1 / dwire / s2 forced-swap
pv3c C=1/4 at 4n K2 b16 — 3/3 ok, every iteration "gate OK (0 bad rows)",
route graph captured on every rank (rank-0 print), swaps fired under it.
Next: 4n b1 A/B (`pv3c_rgraph_ab_4n_k2_b1`) on the same grant, then the
32n A/B `pv3c_rgraph_32n_k2_b1` (sbatch 58362936: LocCap, pv3c C=1/2
graph, `_nrg` eager, fused + dwire, heap 12G).

### 11.1 The route graph was not the fix — the kernels were (09:10)

One-GPU replay on the REAL campaign inputs (`39_route_graph_h2h_real.py`,
32n/16n K2 b1 matrices + pv2 placements; the synthetic h2h had hidden this
because real placements carry up to 13 replicas per expert):

| cell | LocCap | pv3c (campaign ext) | pv3c graph | pv3c kernel v4 |
|---|---|---|---|---|
| 32n K2 b1 | 564 us | 1243 us | 1244 us | **500 us** |
| 16n K2 b1 | 401 us | 530 us | 552 us | **245 us** |
| 32n K2 b64 | 649 us | 1301 us | 1302 us | **555 us** |

Host enqueue is 55-90 us for both routers (graph: 25 us) — the driver gap
was GPU time, so the graph alone changes nothing (kept: it is free and
removes the launch path). nsys on the campaign ext (32n b1): tables kernel
821 us, vacate 368 us, budget 31, route 3 vs LocCap's route3 415 + shares
57 + w3 34. Kernel v4 (`_pv3_ext.cu`, campaign source/binary archived as
`logs/pv3/_pv3_ext_campaign_20260915.cu` / `pv3_ext_campaign_20260915.so`):

1. tables kernel: the c^2 deficit rescan (local-memory arrays) per visit
   replaced by a running deficit sum; per-replica (node, local rank)
   precomputed; the round's (du, dl) carried incrementally (no int div/mod
   per visit); a `node_jj[G, 32]` node -> replica table for the vacate
   pass. Same arithmetic: `test_pv3_kernel.py` parity (bitwise counts vs
   the torch reference) green. 821 -> 369 us.
2. vacate kernel: cooperative — one KT-lane group per token, a lane per
   entry; least-touched-node selection by match/ballot/shfl within the
   group, parallel all-or-nothing acquire (release + extra tickets per lane,
   group vote commits or rolls back), plain-load pre-checks before the
   atomics, O(1) source-replica lookup via node_jj. Same policy and budgets:
   `test_pv3c_kernel.py` green with identical incidence/moves (K2 8n +0.0%
   vs the python reference). 368 -> 78 us. (An intermediate per-node target
   loop was 2x slower than the replica scan — c is 1-2 on average — and a
   divergent `__match_any_sync` hung one gate; both fixed.)

Kernel-test times (synthetic): K2 16n b16 C=1/2 0.586 -> 0.198 ms; Qwen 16n
0.657 -> 0.286; K3 32n 2.9 -> 0.34 ms. Next: 4n driver gate on the rebuilt
ext (job 58365092), then in-driver A/Bs at 32n (sbatch, b1) and 16n (b1/b4).

### 11.2 Driver gate + 4n A/B on kernel v4 (09:20)

Gate (job 58365092, capsule 20260915-161133-ish `run_pv3c_rgraph_gate_4n_k2_v4.log`):
3/3 ok, 128/128 "gate OK (0 bad rows)" per cell, route graph captured on
every lane (fused s1, dwire, s2 forced swaps). 4n K2 b1 A/B (-161623):
plan bracket LocCap 0.644 -> pv3c eager 0.581 / graph 0.563 (fused), dwire
1.328 -> 1.248 / 1.145 — pv3c's route is now the cheaper one at 4n too.
BUT the graphed FUSED arm read plan_comm 0.53 ms on 14/16 ranks vs 0.14
for the eager twin (same in the first A/B, -151806: 0.30 vs 0.16; the
graphed dwire arm was clean) — untraced interaction with the fused arm's
plan_comm bracket. Decision: FLUX_OURS_ROUTE_GRAPH default 0 (opt-in);
`_rg` twins added for A/Bs. In-driver A/Bs in flight: 32n b1 (sbatch
58365589: LocCap, pv3c C=1/2 eager, `_rg`, fused + dwire) and 16n b1/b4
(salloc regular 58365322, chain `chain_rgraph16n.sh`).

### 11.3 32n in-driver A/B on kernel v4 (10:54, capsule 20260915-175055)

K2 b1, C=1/2: fused LocCap 37.996 -> pv3c 37.712 ms (-0.7%; plan 1.878 ->
1.440), dwire 8.865 -> 8.658 (-2.3%; plan 1.686 -> 1.756). Campaign-time
(old kernels): +1.2% / +9.5% with plan 2.43 / 2.90 ms. The `_rg` graph twin
was no better than eager (fused plan 1.861) — default-off confirmed. Short
walltimes (12 min sbatch) were granted in 54 min vs 5.8 h for the 90-min
request. 16n b1/b4 A/B (20-min salloc 58367556) pending.

### 11.4 16n in-driver A/B on kernel v4 (11:01, capsule 20260915-175557) — follow-up 1 CLOSED

K2, C=1/2, Δ vs the in-capsule LocCap twin (campaign-time value in
brackets): b1 fused +0.3% [+2.7%] (plan 0.990 -> 1.008 ms, was 1.384),
b1 dwire -1.7% [+3.7%], b4 fused -1.6% [+0.6%] (plan 1.20 -> 1.08),
b4 dwire -4.1% [+0.9%]. With 32n b1 (§11.3) every small-budget 16n/32n
cell that trailed in the campaign is now on par or better; the b16/b64
wire gap (incidence) is untouched by this fix. The `_rg` graph twin was the
best fused b1 cell here (12.409, plan 0.895) with a normal plan_comm, so
the 4n plan_comm inflation is not graph-specific jitter-proof evidence
either way — the graph stays opt-in. Datasets in §9 for 16n/32n were
measured on the campaign kernels; a refresh on v4 (16n main perf K2+Qwen
LocCap + C=1/2 ~8 nh, 16n weak ~1.5 nh, 32n weak b1/b64 ~4 nh in short
batch jobs) is the user's call. Follow-up cost: 8 short grants, 5.5 nh (sacct).

### 11.5 Kernel v4 constraint audit on the real inputs (11:20)

`39_pv3c_v4_constraint_audit.py` -> `.txt` / `.csv` (job 58372843, 1 GPU):
the v4 kernels run as the driver runs them (once per rank, every rank) on
14 campaign cells (4/8/16/32n, K2 + Qwen, b1 + b16/b64) × C ∈ {1/4, 1/2}
× {pv3, pv3c} = 56 routings, scored with `pv3_check`: constraint 2 over/
under rows 0, non-host rows 0, constraint 3 (floor/ceil form) 0,
conservation True, kernel loud counters 0 — in all 56. Un-rounded
constraint 3: 0 for every pv3c routing (1-3 GPUs on three pv3-only
routings = the integer-rounding case). Realised replica ratio pv3c:
C=1/4 [0.746, 1.254], C=1/2 [0.516, 1.493]. Kernel incidence +0.4..+1.3%
above the deterministic reference (relaxed order). Handoff 40 §6-§7 now
carry the v4 guarantee argument (bit-exact tables + order-independent
ticket invariants fill − rel = floor, fill + ext = ceiling) and a
paper-ready routing paragraph.

## 12. Kernel-v4 refresh of the 16n/32n datasets (launched 2026-09-15 11:35, user go)

Scope: the 16n and 32n cells of the pv3c datasets, re-measured on kernel v4
at the campaign's chosen C = 1/2 with in-capsule LocCap twins (dwire also at
C = 1/4 for the b16 fill). Short walltimes for backfill. Lanes (status-bus
prefixes): `refresh16k` = salloc regular 16n 35 min ->
`pv3c_v4_mp_16n_k2` (27 cells) + `pv3c_v4_weak_16n_k2_b1` (4) +
`pv3c_v4_weak_16n_k2_b64` (2); `refresh16q` = 16n 30 min ->
`pv3c_v4_mp_16n_qwen` (27); `refresh32` = sbatch 15 min ->
`pv3c_v4_weak_32n_k2_b1` (4, heap 12G) + `pv3c_v4_weak_32n_k2_b64` (2).
Aggregator: capsules with run id >= 20260915-1611 are `kernel=v4`; the
chosen dataset prefers v4 rows per cell; the C rule stays on the campaign
rows; `_rg`/`_nrg` twins are diagnostics and dropped. The v4 A/B capsules
(-175055 32n b1, -175557 16n b1/b4) already count as v4 rows.

### 12.1 Refresh incident: Qwen 16n b16 pv3c cells died at first launch — kernel v4.1 (12:35)

Capsule -184102 (16n Qwen): s1 and s2 pv3c C=1/2 at b16 failed on every
attempt with `CUDA error: an illegal memory access`, always on a local-rank-3
process, 17-27 s after start (setup records only, no iteration metrics);
every other v4 refresh cell passed (K2 16n 26/27, weak 16n/32n all ok).
Ruled out: (a) the route product — the v4 audit on this cell's inputs is
clean for all 64 ranks; (b) sizing — the realised v4 route's a2av region
requirements (l0 recv/stage/relay, l1 send/stage/conv/wire) are within the
driver's reference+cushion caps and within 0.2% of the old kernel's
(`logs/pv3/sizing_qwen16.py`); (c) CUDA_LAUNCH_BLOCKING localisation —
deadlocks the fused spin kernels (LocCap twin stuck too). Cause: the v4
tables kernel's per-thread LOCAL memory (cuobjdump STACK 2048 B vs 1536 B
for the campaign build: two more kMaxRep=64 arrays). CUDA reserves local
memory for every resident thread at first launch (2048 threads x 108 SMs x
2 KB = ~450 MB); Qwen b16's fullest GPU (local rank 3 carries the largest
gateway panels) could not provide it -> illegal address at the first route
launch. Fix v4.1: kMaxRep 64 -> 32 (replicas <= nodes; audit max 13;
one-time guard `lcnts.max() <= 32` in both planners), U moved into the
spent `left[1]` slot, byte-sized node/rank/order arrays, visit keys
computed inline -> STACK 480 B (~106 MB reservation), vacate 0 B. Gates
green (parity + pv3c), 56-routing audit clean, real-input route time 32n
b1 441 us / 16n b1 216 us (LocCap 565 / 387). Rerun of the two cells +
LocCap twins: `pv3c_v4_qwen16_b16_rerun.yaml` (lane refresh16qb).

### 12.2 Refresh complete (14:15) — final verdict

Rerun -210903 (v4.1, Qwen 16n b16): 4/4 ok; pv3c fused hit the known combine
stall (82 ms, excluded by the stall rule -> the cell keeps its campaign row,
+7.0% vs the pooled twins), pv3c swap 28.35 vs 26.82 (+5.7%, plan 2.40 vs
3.12: all of it l0/l1 wire). Route-graph A/B capsules -151806/-161623
excluded from the datasets (their default arm ran with the graph ON).
Chosen dataset, Δ vs in-capsule LocCap twin: 2n −1.4%, 4n −0.9%, 8n −0.9%,
16n +0.1% (26/27 rows on v4), 32n +0.1% (all v4). Cells above +4%: 4n Qwen
dwire b16 +4.4, 8n Qwen fused b4 +4.9, 16n Qwen fused b16 +7.0 (campaign),
16n Qwen swap b16 +5.7, 32n fused b64 +8.6 — the last three are incidence
(wire) at large budgets; nothing is plan bracket any more. Refresh cost:
16n K2 35-min grant used 17 min, 16n Qwen 30-min used 15, 32n batch 4 min,
Qwen b16 rerun 3 min (+ the debug/sanitizer grants) ≈ 10.5 nh. Capsules
-183514, -184102, -184912, -185111, -184537, -184739, -210903 (+ the v4
A/Bs -175055/-175557 and gates).
