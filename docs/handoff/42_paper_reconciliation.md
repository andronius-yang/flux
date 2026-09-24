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
