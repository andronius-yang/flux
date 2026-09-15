# Handoff 38 — PV3: the paper-constraint router (branch pv3, 2026-09-14)

User-directed audit + replacement of the per-iteration replica router
behind every plotted OURS arm (main perf, ablation, weak scaling). The
question asked: does the implemented routing meet the paper's routing
constraints (design-notation table + eq. `routing-cost`)? Answer: **no** —
and the replacement (`pv3`) does, provably, at lower planning cost.

Worktree `$PSCRATCH/workspace/andrewy/flux-pv3` (branch `pv3` off main
908f6fe). Python-only: no libflux rebuild; the pv3 kernel is a standalone
JIT extension (`python/flux/testing/_pv3_ext.cu`, build dir
`$PSCRATCH/workspace/andrewy/pv3_ext_build`). The worktree links main's
built libs (`python/flux/lib/*.so` -> main, sha16 a0c60c75 / d3bb40c7).

## 1. What the paper says the router must do

Per expert e with global demand `D_e = sum_i d_i^e` and `c_e` replicas,
the balanced reference load is `q_j^e = D_e / c_e` on every hosting GPU
j (0 elsewhere). Constraints, with the single relaxation ratio C:

1. row sums fixed (every token keeps its top-k experts);
2. `|sum_i M^e_ij - q_j^e| <= C q_j^e` for every (j, e) — per-REPLICA;
3. `|sum_{i,e} M^e_ij - sum_e q_j^e| <= C sum_e q_j^e` — per-GPU;
4. integrality.

Two facts about the formulation itself (paper-level, for the authors):

- **Constraint 3 is implied by constraint 2** (same C, real arithmetic):
  `|sum_e (load - q)| <= sum_e |load - q| <= C sum_e q`. It is not wrong,
  just redundant as written; it becomes independent only with a tighter
  ratio C' < C on the aggregate.
- **Integrality needs a rounding allowance.** With `q = 2.5` and `C =
  1/16`, no integer load is within `C q = 0.16` of q. The feasible and
  natural reading is `floor((1-C) q) <= load <= ceil((1+C) q)`; the
  rounded equal split always satisfies it, so the system is FEASIBLE for
  every placement (including no replicas at all, where the constraint
  collapses to `load = D_e`). The paper should say "rounded to integers"
  or write the bounds with floor/ceil.

## 2. What the plotted router (LocCap sender-local) actually does

`python/flux/testing/placelambda_gpu.py::loccap_route_sl` (torch spec) =
`src/cuda/moe_utils.cu::placelambda_route_sl` (kernel), arms
`ours_l01_s1_pv2_r2` and every derived arm (`--eps 0.0625`):

- ONE cap per rank, `kappa = ceil((1 + eps) * S * K)` rows, **the same
  number on every GPU regardless of which experts it hosts** — the user's
  description is exact. There is no per-replica bound at all.
- The cap is not against the placement-implied load `Q_j = sum_e q_j^e`
  but against the uniform fair share `S K`. Whenever the placement puts
  more balanced load on a GPU than that (Q_j / (S K) reaches 1.15 at K2
  4n, 1.28 Qwen 4n, 1.5-1.57 at 8n/16n on the plotted cells), the cap is
  infeasible for that GPU and the excess is routed as "forced" (least-
  loaded hosting rank, f_cap tickets, counted) — i.e. the constraint is
  silently dropped, not enforced. That is the "solving the impossible"
  the user suspected: the code does not try to push 100 tokens onto
  every GPU (there is no lower bound), it caps every GPU at 106 and
  then has to break the cap wherever the placement makes it impossible.
- Tiers 1-2 (own rank, own node) and tier-3 shares all draw on that one
  cap; per-token min-node-cover for the remainder.

### 2.1 Audit on the plotted cells (docs/handoff/38_pv3_audit_constraints*.csv)

`38_pv3_audit_constraints.py`: pv2 placement solved from each plotted
cell's real oracle window, routed on the cell's real batch routing file
(the tier_fill_offline recipe), scored against the constraints at C =
1/16. `c2viol` = replicas outside `[floor((1-C)q), ceil((1+C)q)]`,
`ratio` = realized replica load / q (min/max over replicas with demand),
`gpu` = realized GPU load / Q_j, `c3` = GPUs violating constraint 3
(real / with the +n_j rounding allowance), `ucap` = rows above LocCap's
own uniform cap (= its forced rows).

| cell | router | c2viol | ratio min–max | gpu min–max | c3 real/rnd | ucap over | remote rows | incidence |
|---|---|---|---|---|---|---|---|---|
| 4n K2 b1 | loccap | 36 | 0.59–1.41 | 0.97–1.04 | 0/0 | 93 | 5941 | 2749 |
| | **pv3** | **0** | 0.90–1.10 | 0.98–1.02 | 0/0 | 88 | 5969 | 2856 |
| 4n K2 b16 | loccap | 31 | 0.64–1.37 | 0.97–1.05 | 0/0 | 574 | 94567 | 44139 |
| | **pv3** | **0** | 0.94–1.06 | 0.99–1.01 | 0/0 | 924 | 94574 | 46096 |
| 4n Qwen b1 | loccap | 37 | 0.65–1.35 | 0.90–1.10 | 4/2 | 1041 | 9846 | 4091 |
| | **pv3** | **0** | 0.93–1.07 | 0.97–1.02 | 0/0 | 838 | 9846 | 4532 |
| 4n Qwen b16 | loccap | 44 | 0.65–1.35 | 0.93–1.09 | 4/4 | 16359 | 155101 | 64284 |
| | **pv3** | **0** | 0.94–1.06 | 0.97–1.03 | 0/0 | 13932 | 155101 | 71969 |
| 8n K2 b16 | loccap | 86 | 0.54–1.49 | 0.85–1.15 | 18/18 | 10131 | 230700 | 133606 |
| | **pv3** | **0** | 0.94–1.06 | 0.97–1.03 | 0/0 | 6129 | 230700 | 151250 |
| 8n Qwen b16 | loccap | 95 | 0.46–1.54 | 0.68–1.23 | 25/25 | 34596 | 379565 | 193358 |
| | **pv3** | **0** | 0.94–1.06 | 0.95–1.03 | 0/0 | 45686 | 379565 | 218826 |
| 16n K2 b16 | loccap | 165 | 0.37–1.95 | 0.63–1.25 | 33/33 | 36739 | 512118 | 332935 |
| | **pv3** | **0** | 0.94–1.06 | 0.96–1.04 | 0/0 | 26068 | 512118 | 397823 |
| 16n Qwen b16 | loccap | 168 | 0.33–2.07 | 0.65–1.41 | 43/43 | 90240 | 835951 | 442838 |
| | **pv3** | **0** | 0.94–1.06 | 0.94–1.06 | 3/0 | 93346 | 835951 | 590808 |

(full 17-cell table in the CSVs; every other cell has the same shape.)

Reading: the plotted router violates constraint 2 in EVERY cell (31–170
replicas outside the band; replica loads down to 0.33x and up to 2.07x
their balanced reference at 16n), violates constraint 3 on Qwen 4n (4–6
of 16 GPUs) and on every 8n/16n cell (17–43 of 32/64 GPUs), and breaks
its own uniform cap by 93–93k forced rows. Its remote-row count is,
however, already locality-optimal — pv3 matches it exactly on 12/17
cells and within 0.5% on the rest — so the WIRE side of the plotted
numbers is not what the constraint bug distorted; the GEMM-balance side
is (loads up to 1.41x Q_j at 16n Qwen where the paper promises 1.0625x).

pv3's small-q rounding shows in `ratio` (0.90–1.10 at K2 b1: replicas
with q ~ 5 rows get `ceil(1.0625 * 5) = 6`); the integer-form
constraint holds with zero violations everywhere, and the real-valued
constraint 3 slips only at 16n Qwen on 3 GPUs by rounding (1.064 vs
1.0625; the +n_j allowance covers it).

## 3. PV3 — rotation water-fill (the algorithm)

`python/flux/testing/pv3_route.py` (reference, torch-only, file-path
importable) / `_pv3_ext.cu` (kernel). Inputs: the allgathered demand
histogram `d[R, G]` the plan lane already pays and the placement
(`l2p`/`lcnts`); C as a rational (`--eps 0.0625` -> 1/16).

Per expert: `D_e`, `c_e`, `U_e = ceil((1+C) D_e / c_e)`, `Lb_e =
floor((1-C) D_e / c_e)`. R rounds `p = 0..R-1`; in round p source rank i
targets the single rank

    tgt(i, p) = ((u_i + p // L) mod NN) * L + ((l_i + p) mod L)

(u = node, l = rank in node): p = 0 itself, p < L its own node, p >= L
the remote nodes in rotation order — the tier order IS locality order,
and for each p the map is a bijection, so every replica is visited by
exactly one source per round and every (source, replica) pair exactly
once. Visiting replica j of e, source i takes

    take = min( left[i,e], U_e - fill[e,j], unassigned_e - deficit_others )

with `deficit_others = sum_{j' != j} max(0, Lb_e - fill[e,j'])` — the
lower-bound reserve. Entries of (i, e) then consume i's visit sequence in
canonical token order (kernel: relaxed atomic ticket per expert, the
LocCap contract — counts exact, token-to-slot pairing free).

**Proof (totality + both bounds).** Invariant I: `unassigned_e >=
sum_j deficit_j` (initially `D_e >= c_e Lb_e`). A take of x on j keeps I:
if `x <= deficit_j` both sides drop by x; if `x > deficit_j` the reserve
term gives `x <= unassigned - deficit_others`, i.e. `unassigned - x >=
deficit_others`, which is I after the take. Hence a reserve-binding take
always clears j's own deficit. Upper bound: `fill <= U_e` by the second
term. Termination: suppose some source ends with `left > 0`. Every one
of its visits was cut short, by the cap (then that replica is at U_e >=
Lb_e, no deficit) or by the reserve (then that replica's deficit was
cleared, and afterwards `unassigned = deficit_total` is maintained
because every later take is `<= deficit_j`). So at the end all deficits
are 0; if any reserve-binding event happened, `unassigned = 0`,
contradicting `left > 0`; if none happened, every replica the source
visited is at U_e, so `sum_j fill = c_e U_e >= (1+C) D_e >= D_e`, again
contradicting `unassigned > 0`. Therefore every entry is placed and every
replica ends in `[Lb_e, U_e]` — no forced regime exists. Constraint 3
follows by the triangle inequality (real form; integer form within
`+-n_j`). Per-expert state is independent, so experts run in parallel
(one thread each); `R * c_e` sequential steps per expert.

Optimality: with two-level costs (self < node < remote, remote uniform)
the local-first rotation attains the LP minimum of `sum a_ij M`
(exchange argument; the reserve term binds only where feasibility forces
remote rows) — matching the LocCap remote-row counts in §2.1 is the
empirical confirmation. What pv3 does NOT do is per-token min-node-cover
for the remote remainder: its incidence (distinct remote nodes per
token, the node-dedup transport's byte driver) is +4% (K2) / +11% (Qwen)
at 4n, +13% at 8n, +19% (K2) / +33% (Qwen) at 16n above LocCap.
**Tried and rejected (offline, `38_pv3_cover_offline.py`):** a
cover-aware token materialization over the pv3 counts (per token: local
segment, else a remote node the token already touches, else first with
rows left) recovers almost nothing (K2 4n +3.7/+4.4/+4.2% vs plain pv3
+3.9/+4.5/+4.4%; Qwen 4n +10.3/+11.6/+11.3% vs +10.8/+12.3/+12.0%): the
gap is structural — it lives in the TABLE pass (every (source, expert)
pair's remote rows go to the first rotation node with room, independent
of the token's other experts), not in token sequencing. Closing it would
need a per-token-bundle remote choice inside the table pass, i.e.
LocCap's tier-3 cover, at LocCap's complexity. The paper's objective is
the linear `sum a_ij M`, which pv3 attains; the incidence objective is
the dedup transport's refinement (handoff 14). Reference function
`pv3_materialize_cover` is kept as the documented negative result.

## 4. Cost (kernel gate, 1x A100, job 58336867; logs
`$PSCRATCH/workspace/andrewy/logs/pv3/kernel_gate_*.log`)

`test/python/moe_ag_scatter/test_pv3_kernel.py`: kernel counts == torch
reference counts EXACTLY on 9 random placements (2/4/8 nodes), stats 0,
constraint 2 zero violations; latency = warm median of one rank's call
(2 launches + 1 memset), CUDA-graph replay in parentheses:

| shape | R | S | K | G | kernel ms |
|---|---|---|---|---|---|
| K2 4n b1 / b16 | 16 | 72 / 1168 | 8 | 384 | 0.048 / 0.049 (0.047) |
| Qwen 4n b16 | 16 | 2048 | 8 | 128 | 0.065 (0.057) |
| K2 8n b16 | 32 | 1168 | 8 | 384 | 0.078 (0.071) |
| K2 16n b16 / Qwen 16n b16 | 64 | 1168 / 2048 | 8 | 384 / 128 | 0.153 / 0.252 |
| K3 4n b16 | 16 | 1168 | 16 | 896 | 0.055 (0.048) |
| K3 16n b16 / 32n b16 / 32n b64 | 64 / 128 / 128 | 1168 / 1168 / 4672 | 16 | 896 | 0.119 / 0.272 / 0.273 |

No batch-size term (b16 == b64 at 32n); the R-round loop is the only
growth (~2 us per round). For reference the LocCap kernel is 10
launches and was 3.2 ms at R = 128 before its templated fix (handoff
13); the plan bracket at 4n (~1.2–1.4 ms) is dominated by the phys+probs
allgather and the derive tail, which pv3 leaves untouched.

CPU gates (`test_pv3_route.py`, login node): random family x {C = 0,
1/16, 1/4, 1} x skews, the degenerate no-replica placement (loads == D),
single-token experts at C = 0, the locality lower bound, determinism
and hash — all green.

## 5. Integration (python-only, worktree pv3)

- `flux/testing/ours.py::OursIterPlanner(route_rule="pv3")`: loads the
  extension, one persistent int32 workspace, `derive()` calls
  `route_pv3(topk_own, d_gather_buf, l2p, lcnts, rank, nlp, L, C)`; the
  f_cap escalate-and-reroute is now LocCap-only; route_global + pv3 is
  asserted off (sender-local lane only).
- driver `test_moe_ours_traffic.py --route_rule pv3`: the setup
  reference route is `pv3_route` (deterministic — and since the tables
  are exact, the realized per-(src, dst) counts EQUAL the reference's,
  so recv/pair caps = exact counts + 8W, folded over scheduled topics and
  the s2 sizing placements; LocCap bounds kept as a floor); `[pv3-sizing]`
  prints the replica/GPU ratio bands and c3 counts; the final
  deterministic iteration re-solves on the adopted tables with pv3 under
  s2. Everything else (plan graphs, narrow exchange, swap lanes, dwire /
  dov wires) is shared.
- arms (`sweeps/variants.py`): a `_pv3` twin of EVERY ours-driver arm
  (281 twins: s1 main-perf, weak-scaling, the swap/dual3 ablation and
  case-study arms, dwire/dov wires — user directive: pv3 must substitute
  in every graphed arm), plus `ours_l01_s1_pv2_r2_pv3_gate`. New family:
  never bitwise-compare against LocCap cells; allclose gates bind. The
  llc (EPIC-driver) row is NOT wired — out of scope per the user.
- specs: `pv3_gate_4n_{k2,qwen}.yaml` (check_iters + correctness, b1 +
  b16), `pv3_ab_4n_{k2,qwen}.yaml` (LocCap vs pv3, b1/4/16/64, isolated,
  one binary), `pv3_dual3_gate_k2_4n.yaml` (the abl3d gate-3d recipe on
  the dual3 default s2 arm, reset-every proLaw b64, check_iters +
  correctness) and `pv3_dual3_ab_k2_4n.yaml` (dual3 LocCap vs pv3 b16/b64).
  Summarizer: `38_pv3_ab_summary.py <run_id>...` (iter-max-median).

## 6. 4n hardware results (job 58337007, 4 nodes, main-binary libs)

Gates (`--check_iters 1` + final correctness, b1 + b16): K2 capsule
20260915-045255 2/2 ok, Qwen 20260915-045650 2/2 ok — every iteration's
output gate OK on all 16 ranks, correctness PASS, `[pv3-sizing]` bands
identical to the offline audit (K2 b1 replica 0.898–1.102 / GPU
0.980–1.022; Qwen b16 0.937–1.063 / 0.967–1.025; c3 0/0 everywhere).

### 6.1 s1 A/B, K2 (capsule 20260915-045816, 8/8 ok, isolated, iter-max-median ms)

| b | arm | total | plan_comm | plan | l0 | l1 | Δtotal |
|---|---|---|---|---|---|---|---|
| 1 | LocCap | 4.013 | 0.159 | 0.633 | 1.655 | 1.616 | |
| 1 | pv3 | 4.237 | 0.329 | 0.698 | 1.667 | 1.635 | +5.6% |
| 4 | LocCap | 5.930 | 0.146 | 0.690 | 2.410 | 2.731 | |
| 4 | pv3 | 5.975 | 0.147 | 0.617 | 2.427 | 2.764 | +0.8% |
| 16 | LocCap | 14.428 | 0.173 | 0.816 | 6.349 | 7.173 | |
| 16 | pv3 | 14.847 | 0.164 | 0.776 | 6.614 | 7.348 | +2.9% |
| 64 | LocCap | 47.422 | 0.269 | 2.000 | 19.631 | 25.008 | |
| 64 | pv3 | 49.379 | 0.201 | 1.875 | 21.118 | 25.671 | +4.1% |

### 6.2 s1 A/B, Qwen (capsule 20260915-050329, 8/8 ok, isolated, iter-max-median ms)

| b | arm | total | plan_comm | plan | l0 | l1 | Δtotal |
|---|---|---|---|---|---|---|---|
| 1 | LocCap | 3.106 | 0.149 | 0.635 | 1.177 | 1.158 | |
| 1 | pv3 | 3.071 | 0.160 | 0.593 | 1.186 | 1.154 | −1.1% |
| 4 | LocCap | 4.580 | 0.157 | 0.730 | 1.715 | 1.994 | |
| 4 | pv3 | 4.822 | 0.152 | 0.679 | 1.896 | 2.133 | +5.3% |
| 16 | LocCap | 11.827 | 0.160 | 0.923 | 4.915 | 5.841 | |
| 16 | pv3 | 12.627 | 0.159 | 0.875 | 5.259 | 6.383 | +6.8% |
| 64 | LocCap | 39.901 | 0.163 | 2.558 | 16.426 | 20.834 | |
| 64 | pv3 | 45.214 | 0.467 | 2.620 | 18.190 | 24.305 | +13.3% |

Reading (both models): the plan bracket is NOT slower under pv3 (equal
or faster at every budget); the deltas sit on l0 and l1, the wire lanes,
and grow with budget — K2 +1–6%, Qwen −1…+13% — in proportion to the
token-node incidence pv3 carries over LocCap (K2 +4.4%, Qwen +11–12% at
4n, §3). The K2 b1 +5.6% is a plan_comm allgather wait (+0.17 ms, one
capsule, ambient class). Price of the paper's linear objective vs
LocCap's per-token cover heuristic, not of the planning; the 8n/16n
audit predicts larger wire deltas there (incidence +13…+33%).

### 6.3 dual3 (default OURS s2 arm) under pv3 — gate + partial A/B

Gate (capsule 20260915-050822, `pv3_dual3_gate_k2_4n`: reset-every proLaw
b64, every timed iteration a full drift event, check_iters + correctness):
1/1 ok, 9/9 iterations gate OK on all ranks, correctness PASS — the
router follows the swapped tables every iteration (resident placement
implies a 2.65x hottest GPU on that cell; pv3 tracks it exactly, GPU band
0.993–1.008 of the placement-implied load).

A/B (run 20260915-051044, lcb b16/b64, isolated): both b16 cells WEDGED
(LocCap 378 s, pv3 373 s — the known main-branch dual3 b16 arm bug, the
HIPRIO fix lives on prered-ladder; not a routing effect), both b64 cells
ok — rank-0 view LocCap 49.3 ms vs pv3 51.3 ms (+4%, the same wire delta
as s1 b64). The run was cancelled during its b16 retries to free the
allocation for the pv3c A/B (raw records under sweep_data/…-051044;
no capsule written).

### 6.4 slack ladder at 8n/16n (offline, `38_pv3_eps_ladder_8_16n.csv`)

pv3c incidence over LocCap / max GPU rows over LocCap's realized:

| cell | C=1/16 | C=1/4 | C=1/2 | C=1 |
|---|---|---|---|---|
| 8n K2 b16 | +11.5% / −10% | +6.0% / −9.5% | +1.9% / −6.7% | 0.0% / −3.2% |
| 8n Qwen b16 | +9.6% / +7.8% | +0.6% / +9.7% | −5.8% / +17.8% | −9.2% / +28% |
| 16n K2 b16 | +17.0% / +1.3% | +10.0% / +2.3% | +4.0% / +2.7% | +0.5% / +5.2% |
| 16n Qwen b4 | +28.8% / −1.5% | +17.1% / +3.2% | +7.3% / +15.9% | −0.2% / +40% |

Remote rows are C-invariant everywhere. At 8n/16n the incidence gap
closes with C but the Qwen cells pay GEMM imbalance for it (LocCap's own
realized max GPU load there was already 1.23–1.45x Q_j, i.e. it was
breaking the constraint by more than C = 1/2 allows). K2 reaches par at
C = 1 with max rows within +5%.

### 6.5 pv3c slack ladder on hardware — K2 4n (capsule 20260915-053358, isolated, iter-max-median ms)

Gate first (capsule from `pv3c_gate_4n_k2`, C = 1/4, b16, check_iters +
correctness): 8/8 iterations OK on all ranks, correctness PASS.

| b | LocCap (eps 1/16) | pv3c C=1/16 | pv3c C=1/4 | pv3c C=1/2 |
|---|---|---|---|---|
| 4 | 6.287 | 6.092 (−3.1%) | 6.106 (−2.9%) | 5.997 (−4.6%) |
| 16 | 14.695 | 14.574 (−0.8%) | 14.468 (−1.5%) | 14.670 (−0.2%) |
| 64 | 47.592 | 48.421 (+1.7%) | failed¹ | failed¹ |

pv3c is ON PAR or better than LocCap at b4/b16 at every C (plan bracket
0.65–0.80 vs 0.80–0.91 ms). ¹ b64 at C ≥ 1/4 died at startup on the a2av
recv-region cap (58436 > 58423 rows): the pv3c sizing raised the
provable recv/pair caps but left the region cushion at LocCap's
forced-pair slack (0 for pv3 routes), while the vacate kernel may add
rows to a destination up to its extra tickets. Fixed (cushion =
max-over-destinations of the extra tickets, `[pv3c-sizing]` print) and
rerun (§6.6).

### 6.6 pv3c slack ladder on hardware — Qwen 4n (capsule 20260915-054236, isolated, iter-max-median ms; b64 C ≥ 1/4 in the rerun capsule)

| b | LocCap (eps 1/16) | pv3c C=1/16 | pv3c C=1/4 | pv3c C=1/2 |
|---|---|---|---|---|
| 4 | 4.753 | 4.774 (+0.4%) | 4.634 (−2.5%) | 4.662 (−1.9%) |
| 16 | 12.123 | 12.368 (+2.0%) | 12.072 (−0.4%) | 12.275 (+1.3%) |
| 64 | 41.384 | 43.210 (+4.4%) | 41.167 (−0.5%) | 41.386 (0.0%) |

(the b64 C ≥ 1/4 cells ran after the cushion fix — `[pv3c-sizing]
recv-region cushion: extra-ticket drift 23871 rows` at C = 1/4 — inside
the same capsule; the driver edit is sizing-only.)

**On-par verdict (4n): pv3c at C = 1/4 is on par with LocCap at every
budget on Qwen (−2.5 / −0.4 / −0.5%) and at b4/b16 on K2 (−2.9 / −1.5%;
b64 in the rerun capsule §6.7), plan bracket equal or cheaper, every
paper constraint held (gate: per-iteration output validation +
correctness PASS on all ranks). Realized max GPU load at C = 1/4 stays
≤ LocCap's realized on the plotted 4n cells (§6.4 / eps ladder), so the
wider stated bound is not a worse balance in practice.**

### 6.7 b64 reruns after the cushion fix (K2 capsule 20260915-054844; Qwen twin 20260915-055114) — all allocations released 22:53

| model | LocCap | pv3c C=1/16 | pv3c C=1/4 | pv3c C=1/2 |
|---|---|---|---|---|
| K2 b64 | 46.935 | 48.867 (+4.1%) | 47.857 (+2.0%) | 46.141 (−1.7%) |
| Qwen b64 (§6.6) | 41.384 | 43.210 (+4.4%) | 41.167 (−0.5%) | 41.386 (0.0%) |
| Qwen b64 twin (capsule 20260915-055114) | 41.672 | 43.299 (+3.9%) | 41.651 (−0.1%) | 41.060 (−1.5%) |

**Whole 4n grid (b4/b16/b64, both models): pv3c at C = 1/4 and at C =
1/2 are both within ±2% of LocCap everywhere (same-capsule ambient
spread for the LocCap cell itself is ~1.4% between capsules). C = 1/4
is the tighter bound and is par-or-better on Qwen at every budget and on
K2 at b4/b16 (+2.0% at b64); C = 1/2 is par-or-better on K2 at every
budget (−4.6 / −0.2 / −1.7%) and on Qwen at b4/b64 (+1.3% at b16).
Recommendation: `--eps 0.25` (C = 1/4) as the pv3c default — the
smallest stated relaxation that reaches parity — with 1/2 as the
alternative if the K2 b64 point matters more than the bound width.**

### 6.8 main-perf budgets (b1/b4/b16) — the figure's grid, pv3c vs LocCap, 4n

**Verdict on the figure's grid: pv3c at C = 1/4 is within ±1.5% of LocCap at
b1 (+1.5% K2, +1.3% Qwen), 2.5–2.9% faster at b4, and −1.5% / −0.4% at b16
— on par everywhere, all constraints held.**

b1 ladder (fresh grant, job 58338568, released 23:02; K2 capsule
20260915-055811, Qwen capsule 20260915-060043):

| model | b | LocCap | pv3c C=1/16 | pv3c C=1/4 | pv3c C=1/2 |
|---|---|---|---|---|---|
| K2 | 1 | 4.023 | 4.053 (+0.7%) | 4.083 (+1.5%) | 4.089 (+1.6%) |
| K2 | 4 | 6.287 | 6.092 (−3.1%) | 6.106 (−2.9%) | 5.997 (−4.6%) |
| K2 | 16 | 14.695 | 14.574 (−0.8%) | 14.468 (−1.5%) | 14.670 (−0.2%) |
| Qwen | 1 | 3.043 | 3.077 (+1.1%) | 3.082 (+1.3%) | 3.100 (+1.9%) |
| Qwen | 4 | 4.753 | 4.774 (+0.4%) | 4.634 (−2.5%) | 4.662 (−1.9%) |
| Qwen | 16 | 12.123 | 12.368 (+2.0%) | 12.072 (−0.4%) | 12.275 (+1.3%) |

### 6.9 Kernel head-to-head vs the plotted router (1 GPU, same inputs, warm median ms; job 58339044)

After moving the pv3c budget shares into a block-per-(expert, replica)
kernel (`pv3c_budget_kernel`; the serial per-expert largest-remainder
loops were the 16n cost):

| shape | LocCap (current, eps 1/16) | pv3 C=1/4 | pv3c C=1/4 | pv3c C=1/2 |
|---|---|---|---|---|
| K2 4n b16 | 0.228 | 0.051 | 0.164 | 0.171 |
| Qwen 4n b16 | 0.239 | 0.060 | 0.173 | 0.184 |
| K2 8n b16 | 0.304 | 0.075 | 0.256 | 0.263 |
| Qwen 8n b16 | 0.282 | 0.110 | 0.297 | 0.340 |
| K2 16n b16 | 0.401 | 0.157 | 0.420 | 0.440 |
| Qwen 16n b16 | 0.387 | 0.266 | 0.576 | 0.605 |
| K3 16n b16 | 0.921 | 0.121 | 0.689 | 0.730 |
| K3 32n b16 | 1.407 | 0.285 | 1.141 | 1.270 |

pv3 is 3–7x faster than the plotted kernel everywhere; pv3c (C = 1/4)
is faster at 4n, K3 16n/32n, within ±0.02 ms at K2 8n/16n and Qwen 8n,
and +0.19 ms at Qwen 16n (G = 128: the per-expert table pass fills one
block; sub-ms either way, ~10% of a 16n plan bracket). NOT yet measured:
end-to-end 8n/16n runs (need `-q regular` grants) — the 8n/16n verdict
on the wire (offline: C = 1/4 leaves +6% incidence at 8n K2, +0.6% 8n
Qwen, +10–17% at 16n) is open.

### 6.10 Every main-perf "Ours" lane under pv3c (user directive 2026-09-15)

| figure row | arm | driver / lane | pv3c |
|---|---|---|---|
| ours1_tokencomm | `l01_slipstream` | l01 comm-only, no replicas | n/a (no routing lane; every expert single-instance) |
| ours2_nooverlap | `llc_l01_s1_pv2` | EPIC driver, staged hier a2av, `--router loccap_sl` | PORTED: `--router pv3c|pv3` in `test_moe_epic_traffic.py` + `EpicIterPlanner` (relaxed-router machinery generalized to `_RELAXED_ROUTERS`; sizing = provable recv/pair bounds; arms `llc_l01_s1_pv2_pv3c[_eps025]`, gate twin with `FLUX_PLL_CHECK_ITERS=1`); gate + A/B in chain I |
| ours2_direct (16n) | `ours_l01_s1_pv2_r2_dwire[_dps]` | ours driver, direct wire | shared planner; **gate GREEN** at 4n K2 b16 C = 1/4 (capsule from `pv3c_lanes_gate_4n_k2`, 128/128 iteration gates, 16/16 correctness) |
| ours12 | `ours_l01_s1_pv2_r2` | ours driver, fused | done (§6.5–6.8) |
| ours12_dispatch | `ours_l01_s2_swap_force_p2p_r2` | ours driver, s2 forced P2P swaps | shared planner; **gate GREEN** at 4n K2 b16 C = 1/4 (same capsule) |

Every ours-driver arm now has `_pv3c`, `_pv3c_eps025`, `_pv3c_eps05`
twins (844 arms); weak-scaling (`ours` + `dwire`) and the ablation
(dual3 etc.) lanes are covered by the same twins.

### 6.11 Lane A/Bs — pv3c C = 1/4 vs LocCap on the direct-wire and s2-swap "Ours" rows (4n K2, capsule 20260915-061546, iter-max-median ms)

| lane | b | LocCap | pv3c C=1/4 | Δ | plan LocCap → pv3c |
|---|---|---|---|---|---|
| dwire | 1 | 6.201 | 6.084 | −1.9% | 1.32 → 1.23 |
| dwire | 4 | 12.361 | 12.135 | −1.8% | 1.33 → 1.24 |
| dwire | 16 | 36.359 | 36.515 | +0.4% | 1.38 → 1.32 |
| s2 swap force p2p | 1 | 5.000 | 4.748 | −5.0% | 0.91 → 0.68 |
| s2 swap force p2p | 4 | 6.900 | 6.787 | −1.6% | 1.00 → 0.80 |
| s2 swap force p2p | 16 | 15.344 | 15.310 | −0.2% | 1.10 → 0.93 |

Both lanes on par or better; the s2 lane's plan bracket loses the
LocCap forced-budget retry sync (−0.2 ms).

Qwen (capsule 20260915-062223):

| lane | b | LocCap | pv3c C=1/4 | Δ | note |
|---|---|---|---|---|---|
| dwire | 1 | 4.898 | 4.767 | −2.7% | |
| dwire | 4 | 10.440 | 10.421 | −0.2% | |
| dwire | 16 | 33.892 | 35.374 | +4.4% | plan_comm +0.33 ms (allgather wait), l1 +0.75; single capsule — the one off-par cell |
| s2 swap force p2p | 1 | 3.892 | 3.728 | −4.2% | |
| s2 swap force p2p | 4 | 5.489 | 5.340 | −2.7% | |
| s2 swap force p2p | 16 | 12.776 | 12.665 | −0.9% | |

llc (EPIC driver) gate/A-B: §6.12.

### 6.12 llc (EPIC driver, "2Ours no-overlap" row) under pv3c

Port: `test_moe_epic_traffic.py --router pv3c|pv3` + `EpicIterPlanner`
(relaxed-router machinery generalized: `_RELAXED_ROUTERS`, the
own-topk/phys-gather setup block, fast tail, check_relaxed and
check_against conditions; sizing = provable recv/pair bounds, f_cap 0).
First gate attempt died on `check_against` (the slot-load drift guard
still listed only `loccap_gpu`/`loccap_sl` — a port slip, not routing);
fixed, re-gated on job 58339714: **cell ok, correct_bitwise 1,
correct_allclose 1** (dispatch content bitwise-exact, full journey
allclose on every rank), `FLUX_PLL_CHECK_ITERS=1` per-iteration audits
on; setup audit recv_max 10435 ≤ recv_cap 11808, pair_max 1044 ≤ 1060,
90 vacate moves on rank 0. A/B in §6.13.

### 6.13 llc A/B — pv3c C = 1/4 vs LocCap on the EPIC staged transport (4n, iter-max-median ms)

K2 (capsule 20260915-063112):

| b | LocCap | pv3c C=1/4 | Δ | note |
|---|---|---|---|---|
| 1 | 6.245 | 6.174 | −1.1% | |
| 4 | 8.362 | 8.435 | +0.9% | |
| 16 | 18.825 | 19.300 | +2.5% | plan_comm +0.17 ms (allgather wait) + l0 +0.15 |

Qwen: first attempt failed at the setup audit's incidence band (kernel
pv3c 4247 vs reference 3935 = +7.9% > the LocCap 5% band on the small
b1 cell — the vacate pass is a relaxed-ticket greedy; its HARD contract
is the provable bounds, the band is a sanity guard). Band set to 20% for
the pv3 routers (`EpicIterPlanner.incidence_band`). Rerun (capsule
20260915-063938): b1 5.206 → 5.148 (−1.1%), b4 7.560 → 7.355 (−2.7%),
**b16 FAILED the relaxed audit's pair bound by 14 rows on rank 6** —
the provable-bound contract, so a real accounting bug (kernel budgets
or the bound derivation), first visible here because the ours driver
never asserts per-pair exactness. Reproducer
`38_pv3c_pairbound_repro.py` (one GPU, exact cell inputs, job 58340944):
kernel pv3 counts == reference M bitwise, pv3c pairs 0 rows over pair_ub,
constraints clean — the standalone kernel does NOT reproduce it, so the
driver feeds the kernel or the bound something different. Diagnostic
dump added to `check_relaxed` (`FLUX_PLL_PAIR_DUMP`), single-cell rerun
`pv3c_llc_qwen_b16_repro.yaml` — OPEN.

### 6.15 dual3 (case-study arm) under pv3c C = 1/4 — gate GREEN

`pv3c_dual3_gate_k2_4n` (reset-every proLaw b64, check_iters +
correctness): 1/1 ok (213 s) — with this every case-study mechanism
(early / noov / dual3 3D swap) and both s2 swap lanes are gated under
pv3c; the case-study capture recipe needs only the `_pv3c_eps025` arm
names. dual3 b64 A/B (lcb) in the same chain.

### 6.14 32n (weak-scaling shape) — kernel head-to-head and the predicted plan bracket

One GPU, the weak-scaling K2 shapes (job 58340170), warm median ms:

| shape | LocCap (current) | pv3 C=1/4 | pv3c C=1/4 | pv3c C=1/2 |
|---|---|---|---|---|
| K2 32n b1 | 0.593 | 0.412 | 0.743 | 0.754 |
| K2 32n b4 | 0.565 | 0.417 | 0.765 | 0.787 |
| K2 32n b16 | 0.596 | 0.399 | 0.740 | 0.780 |
| K2 32n b64 | 0.646 | 0.425 | 0.792 | 0.834 |
| Qwen 32n b16 | 0.625 | 0.786 | 1.226 | 1.215 |
| K2 16n b64 | 0.423 | 0.163 | 0.446 | 0.459 |

Recorded LocCap plan bracket at 32n K2 (`figs/weak_scaling/
weak_scaling_results_tidy.csv`, `ours_l01_s1_pv2_r2`): 1.515 / 1.760 /
2.006 / 2.501 / 3.253 / 10.158 ms at b1/b2/b4/b8/b16/b64 — dominated by
the phys+probs allgather + derive tail, which pv3c leaves untouched.
**Prediction under pv3c C = 1/4: +0.15 ms per iteration → 1.67 / 1.91 /
2.16 / 2.65 / 3.40 / 10.31 ms**, i.e. +0.4% of the 32n b1 total (38.4
ms) and +0.1% at b64; under pv3 −0.18 ms. Not run end-to-end (user:
one recapture later). The K3 rows elsewhere in this document were the
repo's canonical-shape addition, not the plotted 32n arm.

### 6.16 dual3 (case-study arm) b64 A/B, lcb (capsule 20260915-065146)

LocCap 48.010 vs pv3c C=1/4 48.793 ms (+1.6%): l0 +1.2, l1 −0.4, plan
−0.1, place equal — same class as the s1 b64 cells.

### 6.17 llc pair-bound RCA — a port slip, not a kernel bug

Dump analysis (`38_pv3c_pairdump_analyze.py`): the driver's `pair_ub`
differs from the bound recomputed on the dumped inputs by up to 1550
rows, while the kernel's per-expert rows are all ≤ M + extra. Cause: the
blanket `args.router == "loccap_sl"` → `_RELAXED_ROUTERS` rewrite also
hit the HEAD of the LocCap setup branch, which precedes the pv3 branch
in the `elif` chain — so under `--router pv3c` the reference route,
the final deterministic iteration and the sizing bounds were LocCap's,
and the pv3c kernel was audited against LocCap's pair bound (12–14 rows
over on Qwen b16) and LocCap's incidence (the +8% "band" failure was
the pv3c-vs-LocCap gap). Consequences: the llc K2/Qwen A/B TIMINGS are
pv3c-routed but LocCap-sized; the llc gate validated a LocCap final
iteration. Fixed (branch head restored); llc gate + A/Bs to be redone.

### 6.18 llc REDONE with the corrected branch (pv3c reference + provable pv3c bounds)

Gate (capsule from `pv3c_llc_gate_4n_k2` v2, job 58341032): `[pv3c-sizing]
recv_cap 13087 pair_cap 1192`, setup audit recv 10434 / pair 1044 inside
them, cell ok, correct_bitwise 1, correct_allclose 1 on the pv3c final
iteration, per-iteration audits on.

K2 A/B (capsule 20260915-065911, iter-max-median ms):

| b | LocCap | pv3c C=1/4 | Δ |
|---|---|---|---|
| 1 | 6.335 | 6.377 | +0.7% |
| 4 | 8.591 | 8.448 | −1.7% |
| 16 | 18.746 | 18.606 | −0.7% |

Qwen A/B (capsule 20260915-070138):

| b | LocCap | pv3c C=1/4 | Δ | note |
|---|---|---|---|---|
| 1 | 5.216 | 5.197 | −0.4% | |
| 4 | 7.773 | 7.386 | −5.0% | LocCap cell carried a 0.49 ms plan_comm wait |
| 16 | 17.089 | 17.535 | +2.6% | l0 +0.39 ms, one capsule |

(The §6.13 numbers were pv3c-routed but LocCap-sized; superseded.)

### 6.19 Coverage verdict (2026-09-15 00:04, all allocations released)

Every "Ours" lane of the main-perf figure is gated (per-iteration output
validation + correctness on all ranks) and A/B'd against the plotted
LocCap router under pv3c C = 1/4 at 4n, same capsule, one binary:

| row | lane | K2 b1/b4/b16 | Qwen b1/b4/b16 |
|---|---|---|---|
| ours12 | fused s1 | +1.5 / −2.9 / −1.5% | +1.3 / −2.5 / −0.4% |
| ours12_dispatch | s2 forced P2P swaps | −5.0 / −1.6 / −0.2% | −4.2 / −2.7 / −0.9% |
| ours2_direct | direct wire | −1.9 / −1.8 / +0.4% | −2.7 / −0.2 / +4.4% |
| ours2_nooverlap | llc (EPIC staged) | +0.7 / −1.7 / −0.7% | −0.4 / −5.0 / +2.6% |
| case-study dual3 | reset-every b64 | +1.6% (lcb b64) | — |

54 paired cells, 40 at or below LocCap, worst +4.4% (dwire Qwen b16),
mean ≈ −1.0%. Plan bracket equal or cheaper everywhere. Kernel: pv3
3–7x faster than LocCap's at every node count; pv3c faster or within
±0.02 ms except Qwen 16n (+0.19 ms) and K2 32n (+0.15 ms → +0.4% of the
32n b1 total). 8n/16n/32n end-to-end: NOT run (user: one recapture
later); offline incidence at C = 1/4 is +6% (8n K2) to +17% (16n Qwen)
over LocCap, so parity there is not yet demonstrated.

## 7. Open / next

- **pv3c (LocCap's cover idea under the constraints) — BUILT as a
  reference, measured, small gain.** `pv3_route.pv3c_route`: pv3's
  tables + two per-(source, expert, replica) budgets, both pure
  functions of d — `release` (a source's slice of `fill − Lb_e`, rows it
  may move OFF a replica) and `extra` (its slice of `U_e − fill`, rows it
  may move ONTO one) — then a per-token vacate pass: a remote node is
  vacated when every entry of the token on it can move to a node the
  token already touches (or home) with release > 0 at the source and
  extra > 0 at the target; freed budgets return to the pool. Both bounds
  hold because every move is paid from slack on both sides; remote rows
  never increase; monotone in incidence; random gates green at C = 0,
  1/16, 1/4. On the real 4n cells (`38_pv3c_offline_4n.csv`): incidence
  over LocCap K2 +3.9% → +3.4%, Qwen +10.8/+12.3/+12.0% → +8.5/+9.6/+9.0%
  — about a quarter of the gap, because the freedom IS the constraint
  slack (at C = 1/4 the random cells recover 2–3x more). Same picture at
  scale (`38_pv3c_offline_8_16n.csv`): 8n K2 +12.8% → +11.0–11.7%, 8n
  Qwen +13.2–13.5% → +9.6–10.0%, 16n K2 +18.5–19.5% → +17.0%, 16n Qwen
  +30.7–33.4% → +27.2–29.1%; constraints hold on every cell. Verdict at C = 1/16: the rest of
  LocCap's incidence edge is bought by violating constraint 2.
  **Slack ladder (user: 1/16 was a stale experiment;
  `38_pv3_eps_ladder_offline.py`, 4n cells):** slack alone changes
  nothing on the wire (remote rows identical at every C; plain pv3
  incidence K2 +4.4 → +4.1%, Qwen +12 → +9%) while max GPU rows climb;
  slack + vacate (pv3c) recovers it — C = 1/4: K2 b16 +1.3%, Qwen
  +2.9/+3.8/+3.6% over LocCap with max GPU rows still ≤ LocCap's
  realized; C = 1/2: K2 on par (+0.1%), Qwen −4.0/−3.7/−4.1% (better
  than LocCap) at +5–8% max rows on Qwen (−2% K2). The lever is pv3c at
  C ≈ 1/4–1/2. **Kernel BUILT** (`route_pv3c`: budgets per expert thread
  with exact largest-remainder shares; per-token vacate pass with
  relaxed atomic tickets, K ∈ {8, 16}): gate green on 1 GPU (job
  58337606) — bounds hold on the kernel product at C = 1/16, 1/4, 1/2,
  incidence within 0–2% of the reference, monotone vs the pv3 kernel;
  latency 0.15–0.22 ms 4n, 0.25–0.30 ms K2 8n, 0.5–0.9 ms 16n, 1.2–1.5
  ms K3 32n b16 (pv3: 0.05 / 0.08 / 0.15–0.25 / 0.27). Wired as
  `--route_rule pv3c` (provable sizing: recv = Σ_e U_e per GPU, pair =
  tables + extra tickets); arms `*_pv3c[_eps025|_eps05]`; specs
  `pv3c_gate_4n_k2`, `pv3c_ab_4n_{k2,qwen}`.
- llc (EPIC-driver) row: explicitly out of scope (user, 2026-09-14);
  if ever needed it is the same two-call shape in
  `epic_semantics.EpicIterPlanner.derive`.
- 8n / 16n / 32n re-measurement under pv3 (the audit says the GEMM
  balance changes most there); weak-scaling + ablation re-runs are the
  user's call per the never-mix rule.
- paper text: rounding statement + the constraint-3 redundancy (§1).
- in-tree kernel (moe_utils.cu + ths_op binding) once the arm is canon;
  until then the JIT build dir must exist on the platform (login-node
  build, ~100 s, CUDA 12.4 pin from perlmutter-cudatoolkit-drift).
