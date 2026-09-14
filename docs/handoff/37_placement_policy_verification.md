# 37 — Placement policy verification: affinity-first node assignment, tie-breaker usage on LiveCodeBench

Date: 2026-09-14. Question from the paper draft: the sentence
"we place each replica on nodes with higher historical demand for its expert"
implies the node-assignment stage uses the per-node demand histogram (which
node asked for which expert), not just per-expert load, and uses it at higher
priority than compute balance. Is that what the code does, and in the
LiveCodeBench main-perf cells how often did the compute tie-breaker decide?

## Verdict

1. **Yes, node assignment is affinity-first.** In `pv2_place`
   (`python/flux/testing/placement_v2.py`, the only version ever committed,
   f11b5ac 2026-08-27, unchanged since), every instance of an expert is seated on
   the node with the highest `hist[node, expert]` among nodes that still have a
   free slot and do not already host that expert. Accumulated node compute share
   is consulted **only** when two or more candidate nodes tie on that histogram
   value; node id breaks a residual tie. There is no compute-balance term in the
   primary key and no balance constraint other than the fixed slot count per node
   (`L * nlp`).
2. **The tie-breaker is rare, and essentially never touches replicas.** Over the
   six distinct placements behind the main-perf OURS arm (2 models x 4/8/16
   nodes; every budget of a (model, nodes) pair solves on the same oracle window
   and yields the bit-identical placement), 92–95% of instances were decided by a
   unique affinity maximum. Replicas specifically: 100% affinity-decided at 4n;
   at 8n/16n 3–9 of 64–128 replicas fell to the load tie-breaker. Most ties are
   near-dead experts (window demand 0–10 rows) whose primary instance had to go
   somewhere.
3. **Compute balance is achieved by replication, not by node assignment.** The
   affinity placement leaves per-node share imbalance of 1.08–1.38 (max/mean),
   where a load-first counterfactual gives 1.00; in exchange it cuts predicted
   remote rows by 5–7% vs load-first. This is the intended design (the docstring:
   "blind least-loaded spread loses 99% at qwen 4n").
4. **"Historical" is exactly right for the s1 arm.** The histogram is built from
   the oracle sidecar `<mid>.oracle_routing.txt` = the PREVIOUS decode window of
   the same pool (dslots 64:32 → eval slots [64, 96), oracle [32, 64), gap 0).
   Every main-perf OURS rank log records `placement basis prev_batch`, the
   placement is solved once at setup, and the s1 arm never re-solves
   (`place 0.00x ms` per iteration). The evaluated batch never influences it.

## Code trace (what runs, in order)

`test_moe_ours_traffic.py` (`--place_solver pv2 --redundant_per_rank 2
--oracle_routing_file …`):

- `tk_solve` = the oracle routing file (previous window), not the eval batch;
  `oracle_basis = "prev_batch"`.
- `hist = demand_hist(tk_solve, L, G)` — `[NN, G]` count of top-k entries per
  (source node, expert). Token t is homed on rank `t // S`, node `rank // L`.
- `pv2_solve(hist, L, nlp)`:
  1. `pv2_counts(load = hist.sum(0), NN, R*nlp)` — EPLB-global greedy: repeatedly
     give one more instance to `argmax(load / c)`, cap `c[g] <= NN`. Uses
     **per-expert load only**. This is the compute-balancing stage.
  2. `pv2_place(hist, c, L*nlp)` — experts in (share desc, id asc) order; for each
     of its `c[g]` instances pick the node by
     `(hist[u, g] desc, accumulated node share asc, u asc)` among free,
     non-hosting nodes. First instance placed = primary. Leftover slots are
     backfilled by share (no affinity key) — **zero backfill occurred** in all
     six placements (counts spend every slot exactly).
  3. `pv2_rank_assign` — within-node snake over (share desc, id asc); rank
     choice is not demand-aware.
- s1 never calls the runtime pv2 lane; s2 arms re-solve per iteration on the
  live `d` histogram (same function, so the same priority order applies there).

Note: the `pv2_place` docstring says "residual affinity"; the code uses the raw
`hist[u, g]` for every instance (nodes already hosting the expert are simply
excluded). Doc wording only, no behaviour difference.

## Measurement (LiveCodeBench/execution, layer 5, main-perf OURS cells)

Replay: `docs/handoff/37_placement_audit_pv2_ties.py` (working copy and full
per-instance CSV under `$PSCRATCH/workspace/andrewy/placement_audit/`). It
imports the shipped `placement_v2.py`, rebuilds `hist` from each cell's
`*.oracle_routing.txt`, reruns the greedy with per-instance logging, asserts the
instrumented placement equals `pv2_place`/`pv2_solve` bit-exactly, and
cross-checks each cell's rank-0 `setup audit OK` line (E_virt and drift ppm
matched for all six; see `log_match`). Cells: the 42 ok `ours_l01_s1_pv2_r2`
cells in the 13 capsules cited by `figs/main_perf/figure_src.csv`
(20260829-143523 … 20260830-183022), b1–b64.

| model | nodes | slots | replicas | instances | affinity-decided | load tie-break | node-id tie-break | replicas: affinity / load / id | zero-demand instances | node share max/mean (real vs load-first) | remote rows real / aff-only / load-first (of window rows) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Kimi-K2 | 4 | 416 | 32 | 416 | 390 (93.8%) | 26 | 0 | 32 / 0 / 0 | 6 | 1.079 vs 1.000 | 70162 / 70162 / 74599 (122112) |
| Kimi-K2 | 8 | 448 | 64 | 448 | 409 (91.3%) | 39 | 0 | 61 / 3 / 0 | 7 | 1.114 vs 1.000 | 86539 / 86564 / 92852 (122112) |
| Kimi-K2 | 16 | 512 | 128 | 512 | 452 (88.3%) | 60 | 0 | 119 / 9 / 0 | 11 | 1.380 vs 1.000 | 96081 / 96117 / 103091 (121856) |
| Qwen3-235B | 4 | 160 | 32 | 160 | 151 (94.4%) | 9 | 0 | 32 / 0 / 0 | 7 | 1.215 vs 1.000 | 65684 / 65714 / 70045 (122624) |
| Qwen3-235B | 8 | 192 | 64 | 192 | 179 (93.2%) | 12 | 1 | 60 / 4 / 0 | 7 | 1.340 vs 1.003 | 81074 / 81079 / 85739 (122624) |
| Qwen3-235B | 16 | 256 | 128 | 256 | 232 (90.6%) | 23 | 1 | 119 / 8 / 1 | 9 | 1.254 vs 1.003 | 89688 / 89683 / 95270 (122368) |

Reading the tie-breaks:

- Ties are dominated by **primaries of cold experts**: at 4n every tie is a
  primary; the tied affinity values are mostly 0–10 rows in a ~122k-row window
  (K2 4n tie values: 0,0,1,2,2,2,2,4,5,6,6,10,…). 2–8 instances per placement
  had zero demand on every node.
- **Replicas** (the objects of the paper sentence) are affinity-decided in
  100% (4n), 94–95% (8n), 93% (16n) of cases; the affinity value that decided
  a replica has median 81–553 rows, i.e. real per-node demand, not noise.
- Load tie-breaks changed the outcome relative to plain "lowest node id" in
  0–33 instances per placement (`load_tiebreak_changed_node`), and the resulting
  placement differs from the affinity-only counterfactual by at most ±36
  predicted remote rows (<0.04% of the window) — the tie-breaker is immaterial
  to the wire.
- Load-first (balance-priority) would have added 4.4k–7k predicted remote
  rows (+6–7%) at every scale.

Full per-cell numbers: `docs/handoff/37_placement_tiebreak_summary.csv`
(one row per (model, nodes); the `budgets_mib_sharing_this_placement` column
lists the budgets that share that identical placement).

## Suggested paper wording

"Replication counts follow per-expert load (EPLB-style greedy). Each instance
is then assigned to the node with the highest demand for that expert in the
preceding window; node compute share breaks ties only. On LiveCodeBench, >88%
of instances and >93% of replicas are placed by a unique demand maximum; ties
arise almost only for near-idle experts."

## Caveats

- Applies to the pv2 solver (`--place_solver pv2`, every OURS arm since
  2026-08-27). The older PLACE-lambda solver (`placelambda_fast`, ablation-only
  after 8/27) uses an affinity *seed* plus repair passes and was not audited.
- s2 arms use the same function on the live histogram per iteration; the
  priority order is identical but tie statistics were not re-measured there.
- Per-node share imbalance quoted is the solver's own integer share proxy
  (`load << 20 // c`), not measured GEMM time.
