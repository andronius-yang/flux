# Handoff 42 — Reconciling the implementation with the submitted paper (2026-09-23)

Tree: `$PSCRATCH/workspace/andrewy/flux-pv3` (branch pv3, the live tree the
conda editable install points at). Paper: `$PSCRATCH/workspace/andrewy/
LibraX_NSDI_27.pdf` (LibraX, NSDI '27 submission). User rule for this lane:
**every reconciliation must respect the measured `total_ms` — no new source of
latency** (memory `reconciliation-no-new-latency-rule`). Both changes below are
strict reductions (fewer swaps / fewer NVLink bytes) with sub-ms host cost and
ship with a same-binary twin knob for the in-capsule A/B (SCHEMA rule 4).

## 0. The three divergences (user list)

| # | paper says | code did | action |
|---|---|---|---|
| 1 | §4.3 eq. (4): swap only when the GPU-load imbalance exceeds the configured load constraint; accept only max-reducing swaps; stop when back under | composed-orbit lane ran the tau=1 greedy to its FIXPOINT every iteration (`swapall`, cap 8 slots/rank); the one-slot lane used a rows threshold or FORCE (tau=-1) | **band trigger** on the placement reference loads, `--swap_trigger band` (§1) — on the dual3 / dual3+moved-last arms |
| 2 | §4.2 eq. (2)/(3): NIC-balanced split moving the MINIMUM intra-node bytes, Σ_i (V_i − V/G)^+ | lb_union relay cut the source-ascending canonical stream into L equal contiguous chunks — balanced, but NOT minimal (excess cascades) | **water-fill partition**, `FLUX_A2AV_LB_MINMOVE` (default ON under LB_UNION) (§2) |
| 3 | Algorithm 1: redistribution "in rounds across node pairs" | one up-front redistribution (all rounds' pieces pulled before the first wire wait; relay staging holds NN−1 rounds) | **left as is** (user: lower priority, no narrative conflict) — §3 records the memory cost |

## 1. Swap trigger (paper §4.3) — `--swap_trigger band`

**Why the old trigger was wrong under pv3c.** The router (pv3/pv3c, handoff 38/40)
holds every replica's load inside `[(1−C)q, (1+C)q]` with `q = D_e / c_e`, so a
GPU's realised load is inside `(1 ± C) · Q_j`, `Q_j = Σ_{slots} D_e // c_e`
(exactly `ours_swap.rank_loads`). Routing therefore can never push a GPU
outside that band, and it never rebalances *across* GPUs either — the ±C slack
is spent on locality. The only imbalance swaps must repair is the reference
`Q_j` itself, which is fixed by placement × demand. "Estimated GPU load exceeds
the allowed imbalance" is thus a statement about `Q_j`, and because swaps are
intra-node (the node total Σ_j Q_j is invariant) the constraint is per node:

    node u out of band  ⇔  max_{j∈u} Q_j  >  (1 + C) · mean_{j∈u} Q_j        (node_band)

with C the same relaxation ratio as the router's (`--eps`; one configured
load constraint, as the paper has one C). Integer-exact form used:
`max · L > (1 + C) · Σ`.

**Algorithm (ours_swap.py).** `_swap_plan_np(..., bal_C)`: the per-node
heaviest↔lightest pairing is unchanged; a pair is eligible only if its node
is out of band; every exchange must strictly reduce the pair max (tau ≥ 1);
in the multi-pick pass a node that the picks so far brought back in band takes
no further exchange. `swap_orbit_capped(..., bal_C)` (the composed
dual3/swapall lane) therefore runs rounds only while some node is out of
band and stops at the first placement satisfying the constraint everywhere
(or when no improving intra-node exchange exists in the out-of-band nodes) —
the minimal-swap reading of "perform it until we are back under the
balance". Staging cap (8 slots/rank) unchanged. Cost: one extra `[R]` numpy
reduction per round (microseconds); the decision stays sub-ms (unit gate).

**Driver / arms.** `--swap_trigger {tau,band}` (default tau = legacy),
`--swap_bal_C` (default −1 → `--eps`). Arms (sweeps/variants.py):
`ablation_l01_s2_swapall_{nr,rst}_3d_dual3{_str4,_ml_str4}_bal_p2p_r2` (+ `_gate`,
+ the auto-derived `_pv3c_eps025` twins, C = 1/4 for router AND trigger).
`[swap-band]` (setup) and `band(after)` (per-iteration, check mode) log lines
report out-of-band nodes and max/mean before/after.

**Unit gates** (`test/python/moe_ag_scatter/test_ours_swap.py::band_gates`, CPU):
in-band nodes are never touched (bitwise), per-node max monotone, terminal
state in band or no improving swap left, band swaps ≤ tau=1 fixpoint swaps,
deterministic. Passed 2026-09-23 (6 configs).

## 2. Minimal-move relay partition (paper §4.2 eq. 2/3) — `FLUX_A2AV_LB_MINMOVE`

**Mechanism (unchanged parts).** Per (source node n → target node m) the L
union segments (`U[n·L+k][m]` rows each, one per source local rank k) form one
stream; relay rank k stages "its chunk" (intra-node pulls), puts it once over
its NIC to gateway (m, k), which forwards the whole window to every local
rank; the window is one gating lane (signal per (round, k)).

**Legacy partition = equal cut of the source-ascending canonical stream**:
chunk k = rows `[kV/L, (k+1)V/L)` of `seg_0 ‖ seg_1 ‖ … ‖ seg_{L−1}`. NIC-exact,
but the intra-node movement is `V − Σ_k |chunk_k ∩ seg_k|`, and a heavy rank's
excess shifts every later rank's rows off their own NIC: `V = (3, 3, 0)`,
cap 2 → legacy moves 3 rows, the minimum (eq. 3) is 2; `V = (6, 2, 2, 2)`,
cap 3 → legacy 6, minimum 3. No contiguous equal cut of ANY segment order
attains eq. (3) in general (the excess can only flow to the next chunk), so
the fix must be piecewise.

**New partition = water-fill (two-pointer), O(L) per pair, ≤ 2L−1 pieces:**

    cap_k  = ⌊V/L⌋ + [k < V mod L]                (C_NIC = 0, the legacy split of the remainder)
    keep_k = min(V_k, cap_k)                      relay k keeps the FIRST keep_k rows of its own segment
    excess of donors (V_k > cap_k) → room of importers (cap_k − V_k), donors and
    importers walked ascending; moved rows = Σ_k (V_k − cap_k)^+  = eq. (3)

Chunk k on the wire/receiver = `[kept prefix of k][imported pieces in donor
order]`, size ≤ cap_k (NIC balance unchanged). The remote union region on the
receiver is CHUNK-major: window k = `[B_k, B_k + size_k)`, `B_k = Σ_{j<k} size_j`;
`chunk_bound()/chunk_rows_of()` now return the plan's `B_k`/`size_k`, so the
gateway forwards, lane ends (`gate_q`), capacity checks and the wire put are
untouched in shape. Two places consume the piece list: (a) relay phase 1
(`round_pieces`: kept prefix self-copy + one `getmem` per imported piece;
`own_only` = no imports → the put reads the send segment directly), (b) the
fused consumer build (`sort_util.cu`): the canonical dedup row (source-ascending
`mine_token` cumsum) is remapped through per-source-node piece tables
`(lo, hi, dst)` sorted by canonical start (binary search over ≤ 2L entries per
copy; tables ride the existing single meta H2D, `mm_words` i64). The
`FLUX_A2AV_CHECK_COMPRESS` audit applies the same remap on the host.
Ctor-checked prerequisites: union_bcast (Tier B) and fused stage 2 (the ATen
consumer chain is not remapped); `FLUX_A2AV_LB_MINMOVE=0` = legacy equal cut on
the same binary (arms `<arm>_mm0`, `requires` the knob string in the .so).

**Offline magnitude** (`docs/handoff/42_minmove_offline.py`: real K2/Qwen
routing traces, pv2 placement, pv3 reference route C = 1/4):

| trace | wire rows | moved rows legacy | moved rows water-fill | saved |
|---|---|---|---|---|
| K2 4n b64 (id001) | 2640 | 721 (27.3 % of wire) | 381 (14.4 %) | 47.2 % of the movement |
| K2 16n b8 | 196903 | 7261 (3.7 %) | 4592 (2.3 %) | 36.8 % |
| Qwen 16n b8 | 296861 | 14521 (4.9 %) | 9213 (3.1 %) | 36.6 % |

Worst single pair (K2 4n): `V = (103, 135, 210, 196)` → legacy 177 vs 84 rows.

## 3. Per-round vs one-shot redistribution (left as is)

Algorithm 1 issues the intra-node redistribution per round; the code pulls
every round's pieces up front (deadlock rule: gets wait only on peers'
pack-ready flags, acyclic) into a relay staging that holds all NN−1 rounds
(`FLUX_A2AV_MAX_RELAY_NTOKENS`, ≈ one balanced rank's share of the node's
outbound). Memory: (NN−1) rounds staged at once instead of 1 — the sizing in
`sweep.py` already covers it (union column max). Not a narrative conflict
(the paper's rounds describe the wire order; the redistribution is
dependency-free work that the schedule may hoist). Revisit only if heap
pressure at 32n makes per-round staging worth its extra pull→put edges.

## 4. Validation plan / results

Chain `$PSCRATCH/workspace/andrewy/logs/pv3/chain_recon.sh` (status
`recon_status.txt`): build on one node of a 4n interactive allocation →
`recon_gate_4n_k2` (proLaw severe reset-every cell, `--check_iters 1`, arms:
s1 fused pv3c C=1/4 minmove ON/OFF, dual3_str4_bal, dual3_ml_str4_bal) →
`recon_ab_4n_k2` (isolated, K2 4n b8/b64, proLaw-severe + plain lcb: minmove
on vs `_mm0` on s1 and dual3_bal; band vs tau=1 on dual3; dual3_ml_bal).

### 4.1 Gate — capsule `20260923-195048_perlmutter_fcc1b53d`, 4/4 ok

proLaw severe reset-every, K2 4n b64, `--check_iters 1`, payload probe ON: fused s1
pv3c C=1/4 with the minimal-move partition (9/9 iterations, 0 bad rows), its
legacy-cut twin `_mm0` (9/9), dual3_str4 band (9/9), dual3_ml_str4 band (9/9).
Band trigger on the real drift event (rank-0 log): basis 3/4 nodes out of band
(max/mean 1.883 at C = 1/4) → 8 band rounds → 1/4 nodes (1.255; the 8-slot
staging cap stops the rest), 43 slot moves per event, decision 2.3–2.6 ms in the
place bracket (handoff 36 recorded 2.1 ms for the tau=1 orbit); on the quiet
lcb cell the basis is in band (max/mean 1.09) and the lane moves nothing.

### 4.2 4n A/B — capsules `20260923-201035_perlmutter_0a530626` (b64, 12/12) and `20260923-201930_perlmutter_14e2acd6` (b8 s1, 4/4)

Isolated, rank-max per iteration, median over 10 timed iterations (ms).
`Δ` = vs the first row of the block. NOTE: dual3 arms run at b64 only — the
composed lane wedges below b64 in perf mode (pre-existing: 9/15 capsule
20260915-051044 b16 dual3_str4 stuck / `_pv3` failed, before this lane); the
9/23 b8 attempt reproduced it (stuck 372 s) and was re-scoped.

| cell | arm | total | Δ | l0 | l1 | place |
|---|---|---|---|---|---|---|
| lcb b64 | s1 minmove ON | 46.08 | — | 19.21 | 23.75 | 0 |
| | s1 `_mm0` | 46.18 | +0.2 % | 18.75 | 24.90 | 0 |
| | dual3_str4 **band** | 47.42 | — | 19.72 | 25.04 | 0.55 |
| | dual3_str4 tau=1 (legacy) | 48.27 | +1.8 % | 19.11 | 25.67 | 1.18 |
| | dual3_str4 band `_mm0` | 46.51 | −1.9 % | 19.02 | 24.51 | 0.62 |
| | dual3_ml_str4 band | 46.72 | −1.5 % | 18.86 | 24.73 | 0.58 |
| proLaw severe b64 | s1 minmove ON | 69.40 | — | 26.01 | 40.63 | 0 |
| | s1 `_mm0` | 65.23 | −6.0 % | 25.84 | 37.24 | 0 |
| | dual3_str4 **band** | 60.48 | — | 24.84 | 31.10 | 2.48 |
| | dual3_str4 tau=1 (legacy) | 61.05 | +0.9 % | 24.74 | 31.54 | 2.32 |
| | dual3_str4 band `_mm0` | 60.87 | +0.6 % | 23.46 | 32.65 | 2.47 |
| | dual3_ml_str4 band | 65.87 | +8.9 % | 28.56 | 31.70 | 2.45 |
| lcb b8 | s1 minmove ON | 8.94 | — | 3.49 | 4.54 | 0 |
| | s1 `_mm0` | 8.92 | −0.2 % | 3.43 | 4.54 | 0 |
| proLaw b8 | s1 minmove ON | 11.08 | — | 3.90 | 6.12 | 0 |
| | s1 `_mm0` | 10.73 | −3.2 % | 3.75 | 6.14 | 0 |

Reading (user rule: no new latency):
- **Band trigger (dual3):** total ≤ the tau=1 orbit on both cells (−1.8 % lcb,
  −0.9 % proLaw); place bracket halves on the quiet cell (0.55 vs 1.18 ms: the
  in-band node never pays the exchange). Adopt.
- **Minimal-move partition, l0 (the layer it touches):** parity within noise on
  every pair (per-iteration ranges overlap: proLaw b64 25.2–26.6 vs 25.4–26.7,
  lcb b64 18.2–20.0 vs 18.6–20.1). No mechanism in l0 got slower.
- **Open item — l1 on the s1 proLaw pair:** minmove ON reads l1 37.2–42.0 (8/10
  iterations ≥ 38.8) vs `_mm0` 37.0–38.1, i.e. +3.4 ms median, while the same
  knob reads −1.5 ms on the dual3 pair of the same cell and −1.1 ms on the lcb
  s1 pair. l1 does not consume the l0 partition (its wire and A-row order are
  its own), and the severe cell's l1 is the known high-variance phase, so this
  is being re-measured (3 fresh capsules × 20 iterations,
  `recon_rep_4n_k2_s1_b64`) before any verdict. dual3_ml on proLaw (+8.9 %,
  l0 28.56) is the known moved-last l0 cost (handoff 36 §4: "l0 moved-last
  costs ~6 ms on the 8-slot lane"), unrelated to this lane.

### 4.2b Repeat of the open item — 3 fresh 4n capsules × 20 iterations (`recon_rep_4n_k2_s1_b64`)

`20260923-202503_perlmutter_b47f1661`, `20260923-202720_perlmutter_6729853f`,
`20260923-202838_perlmutter_333bf215` (proLaw severe, K2 4n b64, s1 fused pv3c C=1/4):

| capsule | `_mm0` total / l0 / l1 | minmove ON total / l0 / l1 | Δ total |
|---|---|---|---|
| b47f1661 | 66.63 / 27.56 / 36.83 | 65.44 / 25.79 / 37.07 | −1.8 % |
| 6729853f | 67.03 / 27.88 / 36.80 | 66.89 / 26.94 / 37.17 | −0.2 % |
| 333bf215 | 66.15 / 26.83 / 36.69 | 65.84 / 26.57 / 37.02 | −0.5 % |

The +3.4 ms l1 reading of §4.2 was cell-to-cell variance: with fresh processes
the partition is at or below the legacy cut on total in 3/3 capsules, l0 is
−0.3…−1.8 ms and l1 +0.2…+0.3 ms (noise). **Verdict 4n: no new latency;
adopt as default.**

### 4.4 8n (debug QOS, user request) — `20260923-202517_perlmutter_aac67e6e` (b64), `20260923-203004_perlmutter_423948e6` (b8)

proLaw severe, K2 8n, isolated, 8 timed iterations:

| cell | arm | total | Δ | l0 | l1 |
|---|---|---|---|---|---|
| b8 | s1 minmove ON | 13.90 | — | 5.22 | 7.81 |
| | s1 `_mm0` | 13.98 | +0.6 % | 5.13 | 7.75 |
| b64 | s1 minmove ON | 88.15 | — | 37.87 | 47.47 |
| | s1 `_mm0` | 85.45 | −3.1 % | 36.90 | 45.70 |

b8: parity. b64: a single-capsule +3.1 % (l0 +1.0, l1 +1.8 ms) of the same size
and shape as the 4n reading that three repeats dissolved (per-iteration ranges
overlap: l0 37.2–42.6 vs 36.2–40.2); an 8n triple repeat of the pair
(`recon_rep_8n_k2_s1_b64`, 3 × 20 iterations) is the deciding measurement —
results in §4.5 (verdict: constant). **dual3 at 8n b64 could not be measured**: all three dual3 arms
(band, band `_mm0`, and the LEGACY tau=1 arm alike) died at NVSHMEM init with
`flux_shm.cc:117 Check failed: ptr != nullptr` — the composed swap lane's
loose-bound sizing clamps at the 16 G symmetric-heap cap and the 8n b64 demand
exceeds it (sweep.py WARNING "clamped AT the platform cap"). Pre-existing swap-
lane capacity limit (handoff 36 measured dual3 at 4n only), independent of this
lane; the band trigger at 8n therefore rests on the 4n A/B (§4.2) and on the
unit gates.

### 4.5 8n triple repeat and pooled verdict (`recon_rep_8n_k2_s1_b64`, debug QOS job 58798505)

`20260923-205119_perlmutter_5b748ca7`, `20260923-205247_perlmutter_05500e1d`,
`20260923-205410_perlmutter_2f070a67` (proLaw severe, K2 8n b64, 20 iterations):

| capsule | `_mm0` total / l0 / l1 | minmove ON total / l0 / l1 | Δ total |
|---|---|---|---|
| first pass aac67e6e (8 it) | 85.45 / 36.90 / 45.70 | 88.15 / 37.87 / 47.47 | +3.1 % |
| 5b748ca7 | 87.34 / 36.96 / 46.60 | 88.77 / 37.67 / 47.85 | +1.6 % |
| 05500e1d | 90.26 / 38.47 / 47.71 | 91.10 / 38.29 / 47.32 | +0.9 % |
| 2f070a67 | 88.84 / 36.38 / 46.78 | 88.05 / 37.59 / 47.49 | −0.9 % |

Pooled per-iteration rank-max samples (bootstrap 95 % CI of the median
difference, `docs/handoff/42_recon_summary.py` companion computation):

| topology | total ON vs `_mm0` | l0 | l1 |
|---|---|---|---|
| 4n b64 (45 samples/arm) | −0.33 ms (−0.5 %), CI [−1.12, +0.70] | −0.56 ms, CI [−1.68, +0.14] | +0.29 ms, CI [+0.14, +0.55] |
| 8n b64 (48 samples/arm) | +0.18 ms (+0.2 %), CI [−1.44, +2.01] | +0.67 ms, CI [−0.99, +2.05] | +0.62 ms, CI [−0.28, +1.63] |
| 8n b8 (1 capsule) | −0.08 ms (−0.6 %) | | |

**Verdict (user rule): total_ms is constant within measurement noise at 4n and
8n** (both CIs straddle zero; the capsule-to-capsule spread of the severe cell,
±1.5 ms, is larger than any partition effect). The one resolvable component is
a +0.3 ms l1 at 4n (CI excludes zero; 0.8 % of l1), offset by l0 — plausibly
the combine's gather locality over the chunk-major A order; it is not a new
serial stage. The paper's eq. (3) minimum is realised (offline: −37…−47 % of
the intra-node relay movement, §2) at no measured cost. Default stays ON;
`_mm0` remains the same-binary twin for any future check.

### 4.3 Setup-time breakdown (why a cell takes 35–140 s; `[setup-trace]` stamps, s1 b8 4n)

| stage | elapsed (rank 0) |
|---|---|
| process start → dist env up (torch import + torchrun rendezvous, 16 ranks) | 9.8–10.3 s |
| node-major check (first collective) | +3.8 s |
| flux shm / NVSHMEM init (libfabric/CXI bootstrap) | +3.1 s |
| placement + reference route + sizing collective + op construction | +0.6 s |
| setup audit (torch two-layer reference on GPU) | +6.1 s |
| **timed loop start** | **23.5–24.0 s** |
| 15 iterations (5 warmup) at b8 | < 1 s |
| result records + process exit + runner bookkeeping | ~10 s |

Perf cells: 33–38 s at b8, 37–53 s at b64 (the dual3 arms add ~10 s: up to nine
CPU pv3c reference routes for the swap-orbit sizing placements, 1.2 s each).
Gate cells: 130–190 s — `--check_iters 1` runs the torch reference every
iteration (iso_sync 1.0 s per iteration vs 4–10 ms in perf mode) plus the final
correctness pass. The runner adds 3–8 s per cell (staging dirs, srun launch,
record parsing). Integrity-safe savings: more timed iterations per cell are
essentially free (setup is ≥ 95 % of a perf cell); the swap-orbit sizing routes
could use the GPU pv3c kernel (−10 s on swap arms); the swap arms' loose-bound
16 G heap could be sized exactly (NVSHMEM registers the whole heap). Process
reuse across cells is NOT acceptable (fresh heap / sizing / no cached schedules
are part of the one-shot semantics).

## 5. The sub-b64 dual3 wedge — ROOT-CAUSED and FIXED (2026-09-23 evening)

**Symptom (recurring since 9/9):** the composed swap lane with a post-l0 issue
point (`dual3`, `late3`) parks every rank in perf mode at b8/b16 (9/15 capsule
20260915-051044 b16 stuck; 9/23 b8 stuck 372 s) while b64 and every
`--check_iters` cell pass.

**Evidence (job 58815215, 4n K2 proLaw severe b8, `dual3_str4_bal`):**
1. `CUDA_MODULE_LOADING=EAGER` twin of the wedging cell: ok in 57 s
   (capsule 20260924-062717) → the known lazy-load-behind-spin-kernel class
   (memory `l1-a2av-lazy-load-hang`, handoff 8/16).
2. Attach to the parked LAZY run (`$PSCRATCH/workspace/andrewy/logs/pv3/
   wedge_attach.log`): on all four local ranks py-spy shows the host thread
   inside `l1_op.derive_combine_meta` ← `OursRunner.issue_combine_meta(ip,
   late=True)` (driver line 1873); cuda-gdb `info cuda kernels` shows exactly
   ONE resident kernel per GPU: the l0 `agscatter gemmgroupedv2 streamk` GEMM,
   spinning on its per-slot weight gate.

**Mechanism.** In the `late*/dual*` issue modes the driver enqueued, after the
l0 forward: (1) the late plan-overlap combine-meta kernels, then (2) the swap
exchange (`issue_late`). The gated l0 GEMM is resident and spinning on the
swapped slots until (2)'s pushes/pulls land. Under LAZY module loading the
FIRST launch of a `derive_combine_meta` kernel in the process blocks the host
until its module loads, and that load never completes behind the resident
spinning GEMM (the 8/16 class) — so (2) is never issued: circular wait, host
blocked, device spinning. At b64 the l0 op's ~2 ms metadata prologue keeps the
GEMM off the device until the host is past (1); at b8/b16 the prologue is
short and the GEMM is resident when (1) launches. `--check_iters` never
wedges because its per-iteration syncs put the first launches of (1) in
iteration 0 before any gate... (perturbed timing). The 9/9 "wait-before-write"
fix of handoff 36 §13 changed timing, not the cause.

**Fix (driver only, `test_moe_ours_traffic.py`):** `swap_lane.issue_late()`
is now the very next enqueue after the l0 forward, BEFORE
`issue_combine_meta(late=True)`. Rule (also written at the site): *the release
of a device gate must never depend on host progress past another kernel
launch.* The l1 side already follows it (`issue_l1_post` immediately after the
l1 forward). Verified under default LAZY on the same allocation: b8
`dual3_str4_bal` ok 45 s (capsule 20260924-064111), b16 LEGACY tau `dual3_str4`
(the 9/15 cell) ok 46 s (20260924-064157), b8 `dual3_str4_bal_gate` 6/6
iterations 0 bad rows (20260924-064244). Hardening option not taken (no need
shown): prime `derive_combine_meta` at setup, or trap the weight-gate spin.

## 6. Narrative checks for the band trigger (2026-09-24, job 58815666, 4n K2 b64) — FINAL

User requirements: (1) the ablation must keep its ordering (the overlapped-swap
step still yields a speedup); (2) the case study must still show visible NVLink
expert movement. Arms: legacy tau=1 orbit vs `_bal` (band, C = the router's 1/4)
vs `_balc16` (band, C = 1/16). Tool: `docs/handoff/42_narr_summary.py`
(rank-max total_ms; it0 = the drift-event iteration after the post-warmup
reset; rest = mean of the other timed iterations; slot moves from the records,
one move = w1 + w2 ≈ 2 × 29.4 MB). CORRECTION 9/24: the first cut of these
tables (and the §4.2/§4.4 medians) filtered `iter >= warmup_iters` on the
metrics, which already index TIMED iterations only — i.e. dropped the first
five timed iterations. Scripts fixed; all numbers below use every timed
iteration; §4 conclusions unchanged (see 6.4).

### 6.0 The key identity: band trigger at C = 0 ≡ the legacy tau=1 orbit

`node_band(C=0)` is "max > mean", i.e. any node not perfectly balanced is out
of band; every strictly max-reducing exchange is then eligible and the greedy
stops only at the fixpoint — exactly `swap_orbit_capped` without the band.
Verified bitwise on 40 random (placement, demand) cases incl. the 8-slot cap
(`test_ours_swap.py::band_gates` (f)). **So the paper-figure arms (ablation
`swapall_pw`, cycling `swapall`, case study `dual3_str4`) already ARE the
paper's §4.3 algorithm with the configured load constraint set to 0 (perfect
balance) and the 8-slot staging cap: trigger only when imbalanced, accept only
max-reducing exchanges, stop when no exchange reduces the max.** The
reconciliation adds the explicit constraint dial; the figures need no
recapture for it. What the dial buys at C > 0 is measured below.

### 6.1 Case study (capsule 20260924-065407, dual3_str4 reset-every, 32 iterations)

| row | trigger | total median | it0 | moves / iteration |
|---|---|---|---|---|
| Predictable (lcb) | C=0 (legacy) | 48.82 | 48.26 | 12.0 |
| | band C=1/4 | 47.83 | 47.68 | **0.0** |
| | band C=1/16 | 48.45 | 47.83 | 6.0 |
| Drift (S-C schedule; proLaw block = drawn iteration) | C=0 (legacy) | 51.64 | 58.81 | 28.6 |
| | band C=1/4 | 51.55 | 55.70 | 2.9 (proLaw block only) |
| | band C=1/16 | 51.72 | 55.52 | 17.4 |

C=1/4 removes the Predictable row's expert-swap blocks (its basis sits at
max/mean 1.09); C=1/16 keeps movement on both rows (half / 60 % of the
volume) at equal total.

### 6.2 Ablation, 3 reps (matched 20260924-{070659,072147,073628}; LOO 20260924-{071357,072843,074330})

it0 = drift-event iteration (mean over reps, sd ≤ 1.8), rest = other iterations, moves at it0:

| study | arm | C=0 (legacy) it0 / rest / moves | band 1/4 | band 1/16 |
|---|---|---|---|---|
| matched | placement+routing+swap seq (pr) | 54.81 / 48.93 / 27 | 50.14 / 49.48 / 0 | 49.81 / 48.49 / 20 |
| matched | full stack, seq swap | 53.42 / 48.14 / 27 | 48.83 / 48.86 / 0 | 49.10 / 48.48 / 20 |
| matched | full stack, overlapped swap | **49.00** / 48.05 / 27 | 49.96 / 48.92 / 0 | 49.96 / 47.82 / 20 |
| LOO | placement+routing+swap seq (pr) | 68.81 / 56.30 / 59 | 62.34 / 56.88 / 15 | 69.09 / 56.16 / 53 |
| LOO | full stack, seq swap | 67.67 / 55.83 / 59 | 61.03 / 56.07 / 15 | 68.68 / 55.95 / 53 |
| LOO | full stack, overlapped swap | **59.98** / 55.17 / 59 | 56.87 / 56.06 / 15 | **59.85** / 55.33 / 53 |

Reading. The overlapped-swap step (ovl vs seq at it0) is 4.4 ms matched /
7.7 ms LOO at C=0 — today's legacy arms reproduce the 9/15 figure data
(LOO seq 66–69 vs ovl 58–61). At C=1/16 the LOO step is intact (8.8 ms, same
moves ±10 %) but the matched step vanishes (the trigger moves 20 of 27 slots
and the sequential exchange is then only ~0.6 ms exposed, so there is little
left to overlap); at C=1/4 the matched study never swaps and LOO moves a
quarter of the slots — the step shrinks to 4 ms and every steady-state
iteration reads ~1 ms slower (uncorrected imbalance below 1.25×). Steady-state
(rest) totals: C=1/16 ≤ C=0 in 5 of 6 arms; C=1/4 ≥ C=0 in 6 of 6.

### 6.3 Verdict and recommendation

- **Both narrative requirements hold with C_swap = 0**, which is bitwise the
  arms the figures were drawn from. No figure needs recapture for the swap
  reconciliation; the paper's "configured load constraint" is 0 (perfect
  balance) in those experiments and its "minimal number of swaps" is the
  fixpoint under the 8-slot staging cap.
- **C_swap = 1/16** is the useful non-zero setting: 25–50 % fewer moves at equal
  or lower steady-state total, both case-study rows still move, the LOO
  ablation step intact; but it erases the matched-study step, so it must not
  replace the figure arms.
- **C_swap = 1/4 (the router's C) is too coarse** for the swap trigger: it
  breaks both narratives. The routing relaxation and the swap constraint are
  separate dials in the code (`--eps`, `--swap_bal_C`).
- Main perf: the `ours12_dispatch` row (one exchange per pair, FORCE) remains
  the odd arm out; the reconciled arm for a recapture is the composed lane
  (band C=0 or 1/16, dual3, cap 8), now runnable at every budget (§5), pending
  the 8n b64 heap sizing of the swap lane.

### 6.4 Corrected §4 numbers (all timed iterations)

4n b64 A/B (20260923-201035): lcb s1 46.07 vs `_mm0` 45.99 (+0.2 %); dual3 band
47.30 vs tau 48.55 (−2.6 %); proLaw s1 69.27 vs 65.83 (the +5 % that the
repeats dissolved); dual3 band 60.49 vs tau 61.31 (−1.3 %). Pooled partition
verdict: 4n b64 total −0.36 ms (−0.5 %) CI [−1.07, +0.68] (l0 −0.76 CI
[−1.69, −0.12]; l1 +0.35 CI [+0.21, +0.57]); 8n b64 total +0.44 ms (+0.5 %) CI
[−0.96, +1.96]; 8n b8 13.90 vs 13.90. Constant within noise at both
topologies, as before.

### 6.5 Why the harness setup time is NOT MoE-layer latency (user question 9/24)

The per-cell setup (24 s fixed + up to minutes for swap arms) is the test
harness sizing the NVSHMEM symmetric heap and receive/staging buffers with
PROVABLE per-cell bounds: it folds every placement the run will visit (the
swap orbit on the cell's fixed demand; per-topic orbits on schedule families)
and routes each with the CPU reference router (1.2 s per route) before
allocation, so a cell can run at the heap cap and detect overflow instead of
corrupting silently. The tau=1 orbit produces many placements (309 s setup on
the schedule cell), the band trigger few (95 s). None of this is per-batch work
a serving system would do: buffers are allocated once at model load with
capacity bounds (the pv3c caps are O(G) per iteration and ARE computed inside
the timed bracket), the placement basis comes from history, and every
per-iteration decision — plan-comm allgather, swap decision + table apply,
routing, metadata, the exchange issue — is inside `total_ms` (SCHEMA rule 5:
only the initial gating-metadata exchange is untimed). The fixed 24 s is
process start (torch import, 16-rank rendezvous, NVSHMEM bootstrap, heap
registration), paid once per deployment.

## 7. Per-round relay staging (`FLUX_A2AV_RELAY_PER_ROUND`, 2026-09-24) — the paper's per-round redistribution, for memory

**Why.** The 8n b64 swap arms died at NVSHMEM init (§4.4): the fused ops'
capacity buffers at the s2 ceilings (recv 222k rows = 3.2 GB, gateway stage
112k = 1.6 GB, relay 106k = 1.5 GB, send, plus the combine's panels) left no
room on the 16 G heap for the composed swap lane's 470 MB staging
(`ours_swap.py:842`). The relay staging held MY wire chunks for ALL NN−1
rounds at once (§3), i.e. Σ_rounds chunk; the paper's Algorithm 1 stages per
round.

**Design (C++ `gemm_grouped_v2_ag_scatter.cc`, default ON under LB_UNION).**
Two round slots (double buffer) of `max_relay_ntokens_ / 2` rows each; round
dn is pulled into slot `(dn−1)&1` and put from it, so the buffer is
`2 × max_round chunk` instead of `Σ_rounds chunk`: (NN−1)/2 × smaller at equal
rounds (3.5× at 8n, 7.5× at 16n, 15.5× at 32n). Synchronisation is one
intra-rank device edge per round and nothing else:
- the pull of round dn (dn ≥ 3) may reuse the slot only after the put of round
  dn−2 has READ it. On the default single stream (pulls and puts both on
  `cp_stream_inter_node`) the blocking put's stream completion already
  guarantees that by FIFO — zero extra operations; with
  `FLUX_A2AV_RELAY_PULL_STREAM=1` the pull stream waits the round's put event
  (`relay_put_events_`, recorded on the wire stream after each put);
- no host sync, no remote dependency: the put is local-completion, the peer
  gets wait only on peers' pack flags (as before) — the chain pull(dn) →
  put(dn) → pull(dn+2) is intra-rank and acyclic;
- host enqueue order interleaves pull(dn)/put(dn) per round (`pull_round` /
  `put_round` lambdas; legacy order kept for the knob-off path); the GEMM
  gate `relay_send_event_` is recorded after the first two rounds' pulls (the
  only ones that never wait on a put), so the GEMM launch point is unchanged;
- pipeline depth 2: the pull of round dn+2 waits for put(dn); put(dn+1) is
  already staged, and an NVLink pull (~10× the per-NIC rate) finishes long
  before put(dn+1) does, so the NIC never idles — expected latency effect 0.
- Requires the BLOCKING wire (`FLUX_A2AV_BLOCKING_WIRE=1`, the 8/22 hard-rule
  default; nbi / fenced puts do not certify the source is read at stream
  completion) — ctor-checked. `_rpr0` twins pin the legacy all-rounds staging
  on the same binary.

**Sizing paths changed to match** (the buffer only shrinks if the knob asks
for less): `moonep_fused_map.required_a2av_knobs` (the OURS driver's exact
sizing; `relay_per_round_enabled()` mirrors the ctor default),
`gen_matrix.a2av_knob_demands` (new `relay_lb_pr`), `sweep.exact_scale_knobs`
(flux / l01 drivers; picks the per-round bound from the variant env). The
runtime FLUX_CHECK mirrors: `2 × max_round ≤ FLUX_A2AV_MAX_RELAY_NTOKENS`.

**Not done (next candidate): the gateway staging** (inbound chunks of all
NN−1 rounds, 1.6 GB here) could be made per-round the same way, but its
reuse needs the SENDER to wait on the RECEIVER's "forwarded" ack (a remote
flow-control signal per round) — a cross-rank dependency the wire currently
does not have; it couples the sender's put to the receiver's forward
progress. Evaluate only if the relay saving is insufficient.

### 7.1 Results, 4n (job 58817882 round 1, job 58818300 round 2)

**Round 1 (single wire stream, capsules 20260924-081756 gate 6/6, -083151 A/B):**
correct (per-iteration checks at b8/b64, both arms), but fused s1 on the severe
cell at b64 read l0 +6 ms with per-round ON: on the single stream the blocking
put of round dn serialized the pull of round dn+1 behind that round's whole
wire transfer — the double buffer had degraded to a depth-1 pipeline. Two
cells also failed at the recorder flush (a `_drift` name collision my sizing
patch introduced; renamed). Fixes: per-round staging always allocates the
dedicated pull stream and the per-round event edges (depth-2 pipeline as
designed); the relay cushion is the node-pair drift over the L relays (§7)
instead of the whole-node all-rounds slack.

**Round 2 (capsules 20260924-085122 gate 6/6, 20260924-090506 A/B 16/16):**

| cell | per-round ON total (l0 / l1) | legacy `_rpr0` | Δ |
|---|---|---|---|
| lcb b8, s1 | 8.94 (3.50 / 4.56) | 8.89 (3.56 / 4.52) | +0.6 % |
| lcb b64, s1 | 45.72 (19.37 / 24.33) | 45.10 (19.15 / 24.20) | +1.4 % |
| severe b8, s1 | 10.81 (3.95 / 6.11) | 10.96 (3.94 / 6.15) | −1.4 % |
| severe b64, s1 | 69.86 (26.76 / 40.37) | 68.01 (26.10 / 39.53) | +2.7 % |
| lcb b64, dual3 band | 47.35 | 47.35 | 0 |
| severe b64, dual3 band | 59.86 (24.23 / 31.34) | 58.95 (23.97 / 29.98) | +1.5 % |

Relay buffer (rows, from the sizing lines): b64 21340 vs 64248 (**−67 %**,
0.9 → 0.3 GB per rank at K2), b8 2794 vs 8003; the cushion fell from 51845 to
12184 rows (s1) / 17638 (dual3) at b64. The ratio improves with node count
((NN−1)/2 on the exact part; the cushion is now node-count independent).

Latency: within the 3 % parity threshold on every cell (automatic check
passed → 8n/16n queued), but the b64 s1 deltas are both positive (+0.6 /
+1.85 ms, l0 +0.2 / +0.66). Single capsule; the same-size readings in §4.2
dissolved under repeats, and the severe cell's l1 carries ±1.5 ms of
cell-to-cell variance, so this is not yet a verdict either way. If a small
positive l0 cost persists across the 8n/16n reads and a 4n repeat, the lever
is the slot count (a 3-slot buffer keeps most of the saving and adds a round
of pull slack), to be added as `FLUX_A2AV_RELAY_SLOTS`.

### 7.2 8n / 16n reads of round 2 (capsules 20260924-093206 8n b64, -100626 16n b64, -101147 16n b16)

| cell | per-round ON (l0 / l1) | legacy `_rpr0` (l0 / l1) | Δ total | relay rows ON / legacy |
|---|---|---|---|---|
| 8n b64 severe, s1 | 91.62 (42.91 / 45.49) | 89.86 (38.57 / 47.99) | +2.0 % | 15934 / 105688 (−85 %) |
| 8n b64 severe, dual3 band | 82.68 | (heap-capped before) | ran | 16202 / — |
| 8n b64 severe, dual3 tau | 86.99 | (heap-capped before) | ran | 16202 / — |
| 16n b16 severe, s1 | 44.14 (21.61 / 20.72) | 40.16 (17.07 / 21.02) | **+9.9 %** | 4198 / 54473 (−92 %) |
| 16n b64 severe, all four arms | NVSHMEM_MALLOC fail | NVSHMEM_MALLOC fail (legacy too) | — | 15206 / 216320 |

**Heap goal met at 8n**: the dual3 arms that died on the 16 G heap now run at
8n b64 (relay 1.5 GB → 0.23 GB per rank). At 16n b64 on the severe family the
legacy arm dies too: the gateway stage (246k rows = 3.5 GB) and the recv
region dominate — the receiver-side per-round staging (§7, "not done") is
what that cell needs.

**Latency goal NOT met at scale**: l0 +4.3 ms at 8n b64 (every iteration
40.5–45.0 vs 33.8–41.2) and +4.5 ms at 16n b16 — about +0.3 ms per round,
growing with node count. Two candidate mechanisms, both changed for round 3
and isolated by same-binary twins (`rpr3_ab_{4,8}n_k2`):
1. the relay's peer pulls are `nvshmemx_getmem_nbi_on_stream`, which on this
   deployment is proxy-lowered — in the legacy order all gets ran BEFORE the
   wire and were hidden; in the per-round order they interleave with the puts
   on the single proxy thread and steal wire bandwidth → `FLUX_A2AV_RELAY_P2P_PULL`
   (default ON): copy-engine `cudaMemcpyAsync` over the P2P-mapped symmetric
   address (`nvshmem_ptr`, as `flux_shm` tensor lists), no proxy;
2. pipeline depth 2 → `FLUX_A2AV_RELAY_SLOTS` (default 2; twin 3).
Arms: `_rpr0_nop2p` (= the old binary), `_rpr0` (legacy staging + P2P pulls),
base (per-round 2 slots + P2P), `_rprs3`, `_rpr_nop2p` (round-2 config).

### 7.3 Round 3 isolation — VERDICT (capsules 20260924-102746 gate 8/8, -104607 4n, -105850 8n)

8n severe, fused s1, rank-max medians (l0 / total):

| arm | b16 | b64 |
|---|---|---|
| `_rpr0_nop2p` legacy staging + getmem (= old binary) | 9.56 / 23.40 | 38.58 / 89.33 |
| `_rpr_nop2p` per-round 2 slots + getmem (round 2) | 10.64 / 24.58 (+5.0 %) | 41.41 / 90.59 (+1.4 %) |
| **per-round 2 slots + P2P pulls (new default)** | 9.32 / 23.33 (−0.3 %) | 37.27 / 87.25 (−2.3 %) |
| `_rprs3` per-round 3 slots + P2P | 9.39 / 23.47 (+0.3 %) | 36.97 / 86.88 (−2.7 %) |
| `_rpr0` legacy staging + P2P | 9.95 / 23.82 (+1.8 %) | 38.27 / 89.65 (+0.4 %) |

The round-2 cost reproduces exactly in the getmem twin and vanishes with the
copy-engine pulls; slot depth 2 vs 3 makes no difference. Mechanism confirmed:
the proxy-lowered `getmem_nbi` pulls, interleaved per round, shared the proxy
thread with the wire puts. 4n five-arm A/B: all within ±1.8 % (noise). Gate
8/8 with per-iteration output checks incl. the P2P pulls.

**Verdict: per-round relay staging (2 slots) with copy-engine P2P pulls is the
default — paper Algorithm 1's per-round redistribution, at parity or better
on total_ms at 4n and 8n, relay buffer −67 % (4n) / −85 % (8n) / −92 % (16n),
and the 8n b64 swap arms fit the heap again.** The 16n confirmation on the
new binary (b16 severe five arms; b64 on the lcb family since the severe
family exhausts the heap at 16n b64 for every arm) is queued as
`rpr3_16n_k2_*` (§7.4). Receiver-side (gateway stage) per-round staging
remains the next lever for 16n b64 on severe placements.

### 7.4 16n confirmation on the round-3 binary (job 58820895; capsules 20260924-130036 b16 severe, -130341 b64 lcb)

| cell | arm | l0 | total | Δ |
|---|---|---|---|---|
| 16n b16 severe | `_rpr0_nop2p` (old binary) | 16.20 | 39.93 | — |
| | `_rpr_nop2p` (round 2: per-round + getmem) | 21.48 | 44.42 | **+11.3 %** (reproduces §7.2) |
| | **per-round 2 slots + P2P (default)** | 15.54 | 38.35 | **−3.9 %** |
| | `_rprs3` 3 slots + P2P | 16.17 | 39.28 | −1.6 % |
| | `_rpr0` legacy staging + P2P | 16.44 | 43.18 | +8.2 % (l0/l1 at parity; the total carries a stall outside l0/l1 — single capsule, not reproduced elsewhere) |
| 16n b64 lcb (main-perf family) | `_rpr0_nop2p` | 40.91 | 88.08 | — |
| | **per-round 2 slots + P2P** | 37.37 | 83.14 | **−5.6 %** |

Relay buffer at 16n: 4198 vs 54473 rows (b16), 15206 vs 216320 (b64) — −92 %.

**Closed.** The paper's per-round redistribution (Algorithm 1) is now the
implementation's default with the relay buffer at 2 × max-round-chunk, at
parity or better on total_ms at 4n, 8n and 16n (the user rule), because the
pulls no longer touch the NVSHMEM proxy. What remains open for memory is the
receiver side (gateway stage, all NN−1 rounds resident), which is what 16n b64
on severe placements needs; it requires a remote per-round acknowledgement and
is a separate decision.

## 8. Main-perf-conditions check (SCHEMA rule 17, 2026-09-24) — capsule 20260924-153937 (4n, 24/24)

User rule: headline comparisons run under the Figure-9 conditions (lcb family,
the figure's Ours arm `ours_l01_s1_pv2_r2_pv3c_eps025`, b1/b4/b16/b64); b64-only
and severe-family reads are ablation discussion. `_legacy` = all three knobs
(minmove partition, per-round staging, P2P pulls) pinned to the old behaviour
on the same binary.

| 4n K2 lcb, fused s1 | figure value | new defaults | `_legacy` twin | new vs twin |
|---|---|---|---|---|
| b1 | 4.03 (9/15 binary) | 3.76 | 3.86 | −2.8 % |
| b4 | 6.11 (9/15) | 5.82 | 5.85 | −0.5 % |
| b16 | 14.27 (9/15) | 14.28 | 14.25 | +0.2 % |
| b64 | 46.67 (8/29) | 45.72 / 46.07 (two capsules) | 45.10 / 45.99 | +1.4 % / +0.2 % |

No increased latency on the main-perf cells at 4n; the small budgets favour
the new defaults. Band-dual3 on lcb: new −7 % (b1), −3 % (b4), −3 % (b16) vs
its twin. Severe family (ablation discussion only): b1/b4 parity; b16 +5.6 %
single capsule (same class as the §4.2 reading that dissolved under repeats;
unrepeated). 8n and 16n under the same conditions: `recon_smallb_{8,16}n_k2`.

### 8.1 8n under main-perf conditions (capsule 20260924-163137, 6/6; debug job 58828609)

| 8n K2 lcb, fused s1 | figure value (9/15 binary) | new defaults | `_legacy` twin | new vs twin |
|---|---|---|---|---|
| b1 | 5.84 | 4.29 (l0 1.82 / l1 1.67) | 4.18 (1.83 / 1.57) | +2.5 % (+0.11 ms, in l1) |
| b4 | 8.60 | 7.46 (3.21 / 3.22) | 7.30 (3.13 / 3.27) | +2.1 % (+0.16 ms) |
| b16 | 19.16 | 17.52 (7.89 / 8.27) | 17.50 (8.29 / 7.96) | −0.1 % |
| b64 (§7.2/7.3, severe family only) | — | — | — | see §7.3 |

Well below the plotted values at every budget (build drift since 9/15) and
within ±2.5 % (≤ 0.16 ms) of the same-binary twin; the small deltas sit in l1
at b1, which the changed paths do not touch (noise class). 16n under the same
conditions queued (`recon_smallb_16n_k2`).

### 8.2 16n under main-perf conditions (capsule 20260924-170433, 6/6; regular job 58830206) — CLOSED

| 16n K2 lcb, fused s1 | figure value (9/15 binary, C=1/2 arm) | new defaults | `_legacy` twin | new vs twin |
|---|---|---|---|---|
| b1 | 12.63 | 6.81 (l0 2.61) | 7.80 (l0 2.95) | −12.7 % |
| b4 | 15.34 | 10.58 | 10.43 | +1.4 % (+0.15 ms) |
| b16 | 29.31 | 25.27 | 25.85 | −2.2 % |
| b64 (§7.4, same family) | 85.79 | 83.14 | 88.08 | −5.6 % |

All three topologies now read at or below the plotted values and within noise
of (mostly below) the same-binary twin at every main-perf budget. The
reconciliation set — minimal-move partition, per-round sender redistribution
with copy-engine pulls, band-triggered swaps — is latency-neutral or better
under the Figure-9 conditions at 4n, 8n and 16n. Nothing outstanding on this
lane except the receiver-side staging decision (§7, deferred by design).

## 9. FINAL recapture on the reconciled defaults (2026-09-24 pm; SCHEMA rules 17/18) — in progress

Binary: round-3 build (per-round staging + P2P pulls + minmove + band trigger,
commit df726f9+). Old router retired: every Ours cell is pv3c (C=1/4 at 2–8n,
1/2 at 16n) at ALL budgets incl. b2/b64. `_legacy` = the three comm-side
knobs pinned to the pre-reconciliation behaviour on the same binary.

**Step 1 — gate ladder** (20260924-193531): 6/6, per-iteration output checks at
b1/b4/b16, fused arm + band dual3 arm.

**Step 2 — case study** (20260924-194756, 12/12 isolated + nsys, K2 4n b64):
the drawn arm (dual3 reset-every) moves 12 slots/iteration on the Predictable
row and 28.6 on the drift schedule; nsys reports for both rows captured on the
final binary (`sweep_data/20260924-194756.../nsys/`) for the figure regenerate.

**Step 3 — ablation, 3 reps, final binary** (matched 20260924-{202339,202841,
203340}; LOO -{202602,203102,203600}); it0 = drift iteration:

| study | pr (place+route+swap seq) | full stack seq swap | full stack overlapped swap |
|---|---|---|---|
| matched, per rep | 50.5 / 66.8* / 50.5 | 51.5 / 49.5 / 53.0 | 47.2 / 65.1* / 49.2 |
| matched, figure (9/15) | 54.0 | 53.1 | 50.2 |
| LOO, mean (sd) | 68.69 (0.5) | 69.37 (0.8) | **61.55 (0.1)** |
| LOO, figure (9/15) | 67.8 | 67.7 | 59.1 |

\* rep 2 carried a stall spike in two arms (the class figs/ablation/README
already notes: show reps or a box). Excluding it, the overlapped-swap step is
3.9–4.3 ms matched and 7.8 ms LOO — the plotted ordering and magnitudes hold.

**Steps 4–6 — main perf, new defaults vs `_legacy` twin (total ms):**

| cell | b1 | b2 | b4 | b16 | b64 |
|---|---|---|---|---|---|
| Qwen 4n (20260924-203838) | 2.79 / 2.86 | 3.34 / 3.42 | 4.45 / 4.50 | 11.67 / 11.77 | 41.68 / 40.18 (+3.6 %) |
| Qwen 4n figure | 3.03 | 3.79 | 4.67 | 11.81 | 40.13 |
| K2 4n b2/b64 (-204348) | | 4.40 / 4.36 | | | 47.46 / 46.73 (+1.5 %) |
| K2 4n figure | 4.03 | 4.68 | 6.11 | 14.27 | 46.67 |
| K2 2n (-204615) | 4.18 / 4.20 | 4.63 / 4.69 | 5.57 / 5.57 | 12.01 / 11.65 | 36.95 / 36.46 |
| Qwen 2n (-205256) | 2.36 / 2.37 | 2.79 / 2.76 | 3.44 / 3.46 | 8.54 / 8.52 | 31.32 / 30.29 (+3.3 %) |
| Qwen 8n (-205126) | 3.62 / 3.60 | 4.35 / 4.36 | 6.04 / 5.93 | 15.19 / 15.80 | 54.20 / 55.30 (−2.0 %) |
| Qwen 8n figure | 5.27 | 5.87 | 7.74 | 16.74 | 58.37 |

b1–b16: at or below the plotted values everywhere and within ±3 % of the
twin. **b64 at 2n/4n reads +1.3…+3.6 % above the twin** (and Qwen 4n b64 above
its plotted 40.13), while 8n/16n b64 read −2…−6 %. Which of the three defaults
carries the low-node-count b64 cost is being isolated (`final_b64iso_{2,4}n_*`,
2 reps: `_mm0`, `_rpr0`, `_rpr0_nop2p`, `_rpr_nop2p`, `_legacy` twins).
Pending: K2 8n b2/b64, 16n K2 + Qwen (queued).

### 9.1 Remaining main-perf rows + the b64 low-node-count isolation

**K2 8n b2/b64 pv3c rows** (20260924-211139): b2 5.16 vs twin 5.43 (figure 6.48);
b64 59.28 vs 59.60 (figure 62.64).

**K2 16n, pv3c C=1/2 at every budget** (20260924-225643, 10/10; b64 ran at the
runner's 13G heap — the pv3c arm now covers the b64 row that was LocCap):

| b1 | b2 | b4 | b16 | b64 |
|---|---|---|---|---|
| 6.09 / 6.28 (fig 12.63) | 7.59 / 7.56 (13.69) | 9.89 / 9.87 (15.34) | 24.42 / 24.39 (29.31) | 82.15 / 83.70 (85.79) |

(new defaults / `_legacy` twin (plotted value)) — at or below both everywhere.

**b64 isolation at 2n/4n, 2 reps each** (K2 4n 20260924-{210852,212321}; Qwen
4n -{211247,212707}; K2 2n -{211608,213024}; Qwen 2n -{212010,213427}), total
vs the all-legacy twin:

| arm | 4n K2 | 4n Qwen | 2n K2 | 2n Qwen |
|---|---|---|---|---|
| minmove only (legacy staging, getmem) | −1.3 % | +2.9 % | +0.4 % | +2.7 % |
| minmove + P2P pulls (legacy staging) | +1.7 % | +4.9 % | +1.8 % | +3.9 % |
| per-round + P2P, minmove off | −0.6 % | −1.3 % | −0.5 % | +3.5 % |
| minmove + per-round, getmem | −2.6 % | +3.0 % | −0.1 % | +2.9 % |
| all new defaults | +2.2 % | +2.8 % | +1.8 % | +4.6 % |

Two ~1–3 % effects, present only at b64 with 1–3 rounds: (a) copy-engine
pulls cost at 2n/4n (+1.2…+3.0 points over the matching getmem arm in all
four columns) while they are the large win at 8n/16n (proxy contention grows
with rounds); (b) the minimal-move partition costs on Qwen (+2.7…+2.9) but not
on K2. Per-round staging itself is free at every node count once its pulls are
off the proxy. Recommendation pending the user's ruling: node-count-aware pull
transport (CE pulls at ≥ 8 nodes, getmem below); Qwen b64 repeat at 8n/16n
before deciding on a node-count-aware partition default. 16n Qwen queued.

### 9.2 Qwen 16n, pv3c C=1/2 at every budget (20260925-010102, 10/10; b64 at 13G)

| | b1 | b2 | b4 | b16 | b64 |
|---|---|---|---|---|---|
| new defaults | 6.11 | 7.08 | 9.08 | 397.0* | 80.52 |
| `_legacy` twin | 6.46 | 6.87 | 9.04 | 80.1* | 83.01 |
| plotted (9/15 b1–b16; 8/29 LocCap b64) | 12.32 | 13.02 | 14.79 | 28.17 | 76.56 |

\* **b16 = the known intermittent 16n-Qwen combine stall** (handoff 39 §9:
"~350 ms stall, l1 340 ms with equal lane brackets"): per-iteration l1 is
bimodal in BOTH arms — clean iterations 10.1–10.9 ms (total 22–25, below the
plotted 28.17), stalled iterations 340–491 ms (new defaults 6/10 iterations,
twin 4/10). It is independent of the reconciliation (the twin stalls too) and
was handled in the 9/15 campaign by a b16 repeat ladder; a b16/b64 repeat
window (`final_mp_16n_qwen_b16b64_rep`, 2 reps) is queued. b64: the new
defaults beat the twin by 3.1 % but both read above the 8/29 plotted LocCap
value (76.56) — a cross-build/day offset on this cell (the same binary's twin
is +8.4 %); the repeat window re-reads it.

User ruling (9/24 pm): keep ALL new defaults (no node-count-aware transport);
the 2n/4n b64 deltas of §9.1 are accepted as ablation-side discussion; weak
scaling not re-run (32n margin), cycling ablation not re-run (its swap
decisions are the C_swap=0 orbit, unchanged).

### 9.3 Qwen 16n b16/b64 repeat (20260925-013234, -013531; 2 × 4 cells) — recapture CLOSED

| cell | new defaults | `_legacy` twin |
|---|---|---|
| b16, stalled iterations / 10 | 5, 7 | 5, 6 |
| b16, clean-iteration median | 23.5, 24.2 | 23.9, 24.1 (plotted 28.17) |
| b16, stalled-iteration l1 | 350–500 ms (quantised ~350 / ~500) | same |
| b64 median | 79.5, 79.7 | 83.0, 83.1 (plotted 76.56, LocCap 8/29) |

The 16n-Qwen b16 combine stall is present in every capsule today (3 of 3),
in both arms, at identical rates and identical clean-iteration latency: a
pre-existing cell property (handoff 39 §9 called it intermittent; on 9/15 a
repeat ladder found clean capsules), not a reconciliation effect. The
quantised 350 / 500 ms stall lengths point at a fabric-level retry/timeout in
the combine wire, which is a separate root-cause lane. For the figure the
comparable number is the clean-iteration median (23.5–24.2 vs 28.17 plotted;
new = twin). b64: the new defaults beat the twin by 4 % in both reps; both sit
above the 8/29 LocCap plot (cross-build offset on this cell only; 16n K2 b64
sits below its plot).

**Final recapture verdict (all figures, new defaults = minmove partition +
per-round staging with CE pulls + band trigger C_swap=0, pv3c everywhere):**
gate ladder green; case study recaptured with movement on both rows; ablation
ordering and magnitudes hold (3 reps); main perf at or below every plotted
value at b1/b2/b4/b16 for K2 and Qwen at 2n/4n/8n/16n and at b64 for 8n/16n
K2; the accepted exceptions are b64 at 2n/4n (+1.3…+4.6 % vs the same-binary
twin, user ruling) and the pre-existing 16n-Qwen b16 stall class.

## 10. The 16n-Qwen-b16 combine stall — root cause (2026-09-25)

History: flagged open in handoff 30 (8/30), reproduced in 31 and 39, handled by excluding
>2x-plot cells and re-running; 9/24 final recapture: 3/3 capsules, 50-70 % of iterations,
BOTH arms, l1 340-490 ms (quantised ~340 / ~430 / ~485), clean iterations 22-25 ms.

**Shape** (capsule 20260925-013531): on every stalled iteration all 64 ranks report the
same l1 within 0.4 ms (one dependency holds the collective); l0 is clean (10-11 ms) on
the same iterations; no libfabric/NVSHMEM warnings at default logging.

**Window 1** (job 58854731, capsules 20260925-072918 base, -073007 hybrid, -073049
software, -073130 reqbuf, -073209 timing) — fabric hypothesis REJECTED:

| arm | stalled / 10 |
|---|---|
| base (FI_LOG_LEVEL=warn FI_LOG_PROV=cxi) | 6 |
| FI_CXI_RX_MATCH_MODE=hybrid | 7 |
| FI_CXI_RX_MATCH_MODE=software | 6 |
| hybrid + FI_CXI_REQ_BUF_SIZE=16M, MIN_POSTED=8 | 6 |
| FLUX_A2AV_TIMING=1 | **0** |

libfabric warn logs: only benign init lines (av insert, hmem dlopen), no flow-control /
LE / retry events in any arm. FLUX_A2AV_TIMING is read by the DISPATCH op only; its
material effect is a `cudaEventSynchronize` at the end of the dispatch forward
(ag_scatter.cc ~5145): the host cannot run ahead into the combine while l0 executes
(the isolated driver `test_moe_l0l1_traffic.py` syncs + barriers before each iteration
only; plan/l0/act/l1 are enqueued back-to-back).

**Window 2** (job 58857561, capsules 20260925-081526 base, -081612 timing, -081648
conn8, -081934 spintrap; ws2/ws1/nsys cells failed with illegal memory accesses in
knob-specific paths, unrelated):

| arm | stalled | l0 med | l1 clean med | total clean med |
|---|---|---|---|---|
| base | 6/10 | 10.89 | 11.01 | 24.12 |
| FLUX_A2AV_TIMING=1 | 0/20 | 11.83 | 11.26 | 25.26 |
| CUDA_DEVICE_MAX_CONNECTIONS=8 | **0/10** | 12.02 | 10.50 | 25.17 |
| FLUX_A2AV_RS_SPIN_LIMIT=1e6 | 7/10, trap never fired | | | |

**What the two windows establish.** The stall is a GPU-side scheduling interaction in
the combine, not a fabric event, and it depends on the CUDA hardware-channel count:
production cells run with `CUDA_DEVICE_MAX_CONNECTIONS=32` (recorded in every capsule's
env; the dispatch FLUX_CHECKs > 1 under FLUX_A2AV_EARLY_LAUNCH, so `launch.sh`'s `:-1`
default never applies to the ours arm) and stall 6/10; at 8 channels the same cell ran
0/10 with a better clean l1 (10.50 vs 11.01) but a slower l0 (12.02 vs 10.89) in that
single read. A host wait at the end of the dispatch (FLUX_A2AV_TIMING) also removes it
(0/20). The quantised ~340/430/485 ms lengths are unexplained; no CUPTI activity longer
than 50 ms appears in the (warmup-only) nsys capture, so the stall is idle time.

**Refuted fix (commits d1b978f/f232029, REVERTED in a83665c/7f473da).**
`FLUX_A2AV_RS_WAIT_KERNEL=1` replaced the combine's 12 zero-SM `cuStreamWaitValue` GEQ
waits with one-thread spin kernels. Every cell DEADLOCKED (4n gates K2/Qwen b4/b16, 16n
b16): the persistent pack/reduce kernels saturate the SMs while they wait for data, a
spin-kernel wait cannot be scheduled, and the puts stream-ordered behind it never issue.
The zero-SM waits are load-bearing. Side effect: a deadlocked cell leaves the
allocation's nodes "busy" (step creation disabled) — release and re-allocate.

**Channel ladder — RESULT (2026-09-25 05:41; same binary 03:28 build; 4n capsules
20260925-1030xx..1111xx, 16n 20260925-1233xx..1239xx, host enqueue timers on):**

| 16n Qwen | stalled iters | b16 total (clean) | b64 total |
|---|---|---|---|
| c32 (production pin) | 14 / 20 | 24.21 | 78.72 |
| c16 | **0 / 20** | 23.32 (−3.7 %) | 78.54 (−0.2 %) |
| c8 | 0 / 30 | 24.76 (+2.3 %) | 83.26 (+5.8 %) |
| c4 | 0 / 20 | 27.94 (+15 %) | 92.07 (+17 %) |
| c2 | 0 / 20 | 26.66 (+10 %) | 94.00 (+19 %) |

| 4n Qwen total vs c32 | b1 | b4 | b16 | b64 |
|---|---|---|---|---|
| c16 (2 reps) | +2.4 % | +3.4 % | +1.4 % | +1.6 % |
| c8 (4 reps) | −0.9 % | +0.1 % | +6.6 % | +11.6 % |
| c4 / c2 | +1 / +2 % | +5 / +9 % | +8 / +11 % | +5 / +9 % |

The stall exists ONLY at 32 connections (the hardware maximum). The host-side enqueue
of the combine is ~1 ms in every iteration including stalled ones (`l1_enq_ms`, new
driver metric), so the wait is on the device; the host-queue hypothesis is refuted.
Mechanism at exactly 32 is still unnamed (the nsys mode crashes at 16n), but the
lever is: **c16 removes the stall and is at parity or better at 16n; at 4n it reads
+1.4…+3.4 % on 2 reps (within twin noise, but one-signed).** Candidate ruling: amend
SCHEMA rule 14's OURS pin 32 → 16. Validation queued (specs `cpin_*`): 4n K2 twins +
2 more Qwen reps, s2 correctness gates at c16 vs c32 (rule 14's conn≤8 deadlock /
torn-row class: K2 + Qwen b16/b32/b64, random payload), 8n K2/Qwen twins (debug QOS),
16n K2 twins + Qwen reps (regular).
