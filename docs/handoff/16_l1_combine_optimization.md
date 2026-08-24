# Handoff 16 — layer1 combine optimization at scale (2026-08-23/24)

Two-lane campaign (8n + 16n peers, handoff-15 §3 format) attacking the 16n
ranking inversion found by the datacamp (handoff 15 §2): the hcc combine's l1
exploded at W=64 (39.2 ms vs EPLB's direct-a2av 12.5 at K2 b8). Mission: (1)
make Slipstream the best arm at 8n/16n, K2+Qwen b1–b64; (2) make COMET beat
Torch+GEMM. Cost: **24.3 node-hours** (8n 11.3, 16n 13.0), 22 record capsules,
3 binary generations, 15/15 random-payload correctness gates green.

## 1. The mechanism (nsys-established at both scales)

The combine wire drains in **end-synchronized PROXY ROUNDS**: each stream lane
serializes its blocking `putmem_signal`s through the single NVSHMEM host proxy
(~0.2 ms/put service solo, ~340–590 µs under concurrency, plus a payload term:
~3.5 GB/s proxy throughput). At 16n S=2 the capture shows 71 rounds × 583 µs ≈
the whole 43.6 ms wall — **l1 ≈ put_count × proxy_service, payload-independent
at small budgets**. Consequences:
- **Stream-lane knee at ~14–15 TOTAL in-flight lanes** (S=14 at 8n, S=15 at
  16n) = proxy concurrency, NOT lanes-per-target; flat past the knee.
- **n_split is the dominant lever** (it multiplies put count): K2 ns1+S15 at
  16n cuts l1 39.5→11.67 (−70%; b1 −87%) — one round of NN−1 puts.
- `CUDA_DEVICE_MAX_CONNECTIONS` is a NON-lever (8==32 everywhere; conn=1
  INVALID for Slipstream — EARLY_LAUNCH FLUX_CHECK at ag_scatter.cc:800).
- CTA budgets 10/8/6 and dense `FLUX_RS_BLOCKS=20` stay (re-ladders flat at
  the winners; RS_BLOCKS default 3 is CTA-starved — 24.5 vs 15.9 l1 at 8n).
- **Dense/COMET floor** = the `ep_topk_gather_rs_kernel_v2` production
  kernel's per-(node,split) tile-barrier waves (45% of dense l1 at 16n b8);
  receiver signal-waits are ~2 µs. ns1 collapses waves AND puts.
- Refuted on evidence (patches avoided): conv-stream spreading (CE streams
  not starved), cross-split put aggregation (ns1 achieves it in env).

## 2. Code changes (all in `src/moe_gather_rs/ths_op/gemm_grouped_v2_gather_rs.cc`, uncommitted)

| Gen | ths_op sha | Tag | Change |
|---|---|---|---|
| 1 | 3bdc6601 | `FLUX_A2AV_RS_WIRE_NSTREAMS_TAG` | RS_WIRE_STREAMS a true ladder (S−1 extra streams, `gi % S`; S=2 bit-identical to canonical parity) |
| 2 | efb01ae4 | `FLUX_A2AV_RS_WIRE_XSPREAD_TAG` | S>NN−1 cross-split spread `(sid·(NN−1)+gi) % S` (S≤NN−1 unchanged); cap 32 |
| 3 | b2bf765a | `FLUX_RS_WIRE_NSTREAMS_TAG` | DENSE sender honors `FLUX_RS_WIRE_STREAMS` (default 1 = shipped schedule bit-unchanged; XSPREAD mapping; lanes join back) |
| 5 | e3dabfe372e9ccdd | `FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT16_TAG` + `FLUX_RS_WIRE_STREAMS_DEFAULT16_TAG` + `FLUX_RS_NSPLIT_512_TAG` | **CANONICALIZATION (2026-08-24, user decision; SCHEMA rule 12)**: a2av + dense wire-stream defaults 16 (plateau; = nodes at 16n, easy 32n bump); dense honors 512-tile-acceptable splits (K2 dense ns2 constructible — previously demoted to 7, same class as §4) |
| 4 | 2ad1176c | `FLUX_A2AV_NSPLIT_HONOR_TAG` | a2av skips the legacy n_split demotion (see §4) — BUILT 2026-08-24 for the split-overlap verification lane (K2 ns2..6 constructible for the first time; ns1/ns7 + qwen ns2/ns4 byte-unchanged via legacy accept) |

Every inter-node put remains a blocking `putmem_signal` (SCHEMA rule 5
untouched). Tags live in **libflux_cuda_ths_op.so** (not libflux_cuda.so).
lib64→lib copy required after every build.

## 3. Recommended configs (validated, gated, capsule-recorded)

| Scale | Arm | Config |
|---|---|---|
| 16n | Slipstream (both models) | `n_split_l1=1` + `FLUX_A2AV_RS_WIRE_STREAMS=15` (single config; b64 tie-in-band vs ns-canon, documented) |
| 16n | COMET (both models) | `n_split_l1=1` + `FLUX_RS_WIRE_STREAMS=15` + `FLUX_RS_BLOCKS=20` |
| 8n | Slipstream K2 | **REVISED (§4b, gen-4, gated)**: ns1+S14 for b1–b4; **ns2+S14 for b8–b64** (ns2 bridges the interior: wins b16–b32, ties ns7 at b64). Gen-3 capsule record (ns1+S7 / ns7+S14) remains the pre-gen-4 reference |
| 8n | Slipstream qwen | ns2+S14 (ns1 does NOT win at 8n qwen — n_split policy is scale+model dependent; S≈14 is the portable knob) |
| 8n | COMET (both) | ns1 + `FLUX_RS_WIRE_STREAMS=7` + RS_BLOCKS 20 |

Specs: `sweeps/specs/l1opt16n_{k2,qwen}_16n_win.yaml` (K2 spec drops `shape:`
to clear the apply_shape n_split pin — comment in-file),
`l1opt_slipstream_{k2,qwen,k2v2}_8n.yaml`, `l1opt_comet_{k2,qwen,k2v2,qwenv2}_8n.yaml`.
**CANONICALIZED 2026-08-24 (user decision; SCHEMA rule 12 = authority):**
binary defaults S=16 both wire knobs (gen-5); per-arm n_split pinned in
variants.py — Slipstream `--n_split 2`, COMET `--n_split 2`, EPIC/PLL-LLC
`--l1_n_split 1` (staged l1: ns1 forfeits nothing); sweep.py defers to
variant-pinned `--l1_n_split`; torch stays on spec presets (untouched NCCL
baseline). RATIONALE (user, 2026-08-24): the n_split canon encodes each
arm's ARCHITECTURAL IDENTITY, not the per-cell perf optimum — **COMET ns2
is the authentic interpretation of the paper's algorithm** (column-wise
splits + overlap are part of the baseline's identity; ns1 would strip the
paper's mechanism and make it a tuned variant), Slipstream ns2 keeps our
overlap mechanism live, and staged EPIC/PLL take ns1 because they have no
overlap to express. The validation ns1-vs-ns2 COMET A/B therefore
QUANTIFIES the cost of authenticity (it does not challenge the choice).
Remaining known perf deltas accepted with the canon: 16n qwen slipstream
(ns1 measured better at b8: 11.64 vs 13.40) and K2 b1–b4 slipstream
(ns1 better by ~0.9 ms at b1).
**VALIDATED (canonval, gen-5, 8n debug window): 6/6 gates PASS** —
slipstream+COMET K2/qwen 32/32 "flux and torch matches" (incl. the
first-ever K2 dense ns2 construct), EPIC+LLC K2 32/32 ranks "dispatch
bitwise-exact, full journey allclose", all random payload. **COMET
authenticity cost (ns2 vs ns1, K2)**: +0.32 ms (~2%) at b8 — and ns2
WINS b64 by 5.4 ms (113.9 vs 119.2): the paper's column-split overlap
pays at large budgets on dense too.

## 4. The K2 n_split demotion artifact (correctness-of-labels hazard)

`n_split_fixed()` (~:2403) runs legacy demotion before the a2av branch: **K2
a2av n_split∈{2..6} silently runs ns7** while labeled as requested (n_per 3584
not 1024-aligned, n 7168 is). Any historical/future capsule labeled K2-a2av-ns2..6
on pre-gen-4 binaries measured ns7. K2 ns1 and ns7 are real (legacy accept);
qwen ns2/ns4 unaffected. Gen-4 fix BUILT 2026-08-24 (2ad1176c; a2av honors
requested n_split, legacy-accept/dense byte-unchanged, own never-mix tag) —
anchors reproduced gen-3 within band. NOTE: K2 ns3 is NOT constructible
(7168 % 3 != 0, driver assert); the valid K2 split set with n_per%512==0 is
{1, 2, 7, 14}, so the b32–b64 bridge candidate is ns2 (ns14 = fine-split rung).

## 4b. Split-overlap verification lane (l1split8n, 2026-08-24, gen-4 2ad1176c)

User-directed follow-up testing the hypothesis "ns1 sacrifices the COMET-paper
column-split overlap". All confirmed:
- **Split pipelining is SOUND** (nsys, K2 ns7+S14): first wire put starts 2.6 ms
  into a 25.5 ms GEMM2 (= end of split-0's column window), clean ~3.5 ms
  per-split cadence, 31–34/49 puts start before GEMM2 ends; wire-under-GEMM
  overlap fraction 47–55% at b64 (b8 ~35% — GEMM too short to hide more).
  Architecture verified in source: Slipstream's fused op gates the SAME
  TopkReduceScatterOp on per-(split, expert-tile) barriers written
  progressively by the GEMM (:2816–2824); the epic/llc staged runner
  pre-fills those barriers to 1 and event-gates the combine strictly after
  full GEMM2 (test_moe_epic_traffic.py:363, epic_semantics.py:3339) —
  column-split GEMM overlap architecturally present in Slipstream, absent in
  EPIC/PLL, exactly as required.
- **True K2 split ladder (first ever, gen-4), S=14, e2e ms**: b1 ns1 4.75 <
  ns2 5.65 « ns7 9.75 « ns14 15.46; b8 ns1 14.19 ≈ ns2 (band); b16 ns2 23.57 <
  ns1 24.39 < ns7 25.70; b32 ns2 43.54 < ns1 46.07; b64 ns2 82.39 ≈ ns7 83.21
  < ns1 86.72. **ns2 bridges the entire interior**; ns14 loses everywhere
  (put-count penalty). qwen: b8 ns2 12.96 < ns1 13.55 < ns4 14.11; b64 ns4
  82.50 < ns2 85.11 < ns1 85.95 — the optimum split grows FINER with budget
  (overhead↔overlap curve), model-dependent rails.
- **Winner-env transfer to the STAGED arms** (gen-4 baselines reproduce gen-0
  within band; ns1+S14, K2/qwen b8 totals): EPIC 25.20→17.43 (l1 13.71→6.27,
  −54%) / 20.49→18.47 (l1 9.58→6.92); PLL/LLC 22.79→15.74 (l1 12.99→6.13,
  −53%) / 17.58→14.62 (l1 8.36→5.31). Pure win — staged arms forfeit no
  overlap at any budget, so ns1 is their config everywhere.
- **REVISED 8n Slipstream recommendation**: K2 = ns1+S14 for b1–b4, **ns2+S14
  for b8–b64** (replaces the ns1/ns7 budget split); qwen = ns2+S14 (ns4 rail
  at b64 within reach). K2 ns2+S14 correctness gate: **PASS 32/32 ranks
  all-close** (random payload, gen-4, full-stdout rerun on 57510019 —
  thresholds 0.02/0.02, every rank "flux and torch matches").
- Ops note: the lane was killed mid-close by a model-usage limit; the
  orchestrator adopted the remainder (9 cells, one debug window). New
  mistake for the ledger: FLUX_SWEEP_RECORD_DIR must be a SHARED-FS path —
  a login-node /tmp path silently discards records (compute-local /tmp).

## 5. Verdicts (same-binary, same-allocation, plan-inclusive totals)

**16n** (winners vs refs, K2/qwen): Slipstream best at K2 b4–b64 (b8 24.1 vs
torch 32.2 / EPLB 30.4; b64 139.1 vs EPLB 211.0) and qwen b8–b64 (b4 within
0.6 of EPLB) → **goal 1 achieved b4/b8-up; the datacamp inversion is reversed
everywhere l1 caused it**. COMET beats torch at qwen b8+ and K2 b16+ (K2 b8
band-tie) and owns b32/b64 (torch OOM class) → **goal 2 achieved b8/b16-up**.
**8n**: Slipstream best at EVERY K2 budget (b1 5.79 vs torch 6.93); qwen best
b2–b16. COMET beats torch at every K2 budget and qwen b2–b64.
**Honest boundaries**: b1(–b2 at 16n) belongs to torch — l0 DISPATCH (~6.5 ms)
dominates Slipstream there now, a dispatch problem outside this campaign;
qwen b32–b64 at 8n belongs to PLL/LLC (placement traffic reduction — a
different lever). Torch b32+ = its gathered-bytes OOM class (handoff 15).

Capsules: 16n winners `20260824-023050_…{8dc058d3,318e86f7,49831467,60a2022f}`
+ refs `…{6e5e833c,0e0a3d5c,242b990d,7557d2a7}`; 8n set of 14 (run_ids in the
8n verdict table, `scratchpad/l1opt8n/verdict_table.txt`). All uncommitted.

## 6. Structural tickets (user decisions, both lanes concur)

1. **NVSHMEM proxy bypass** — the single proxy thread (~340–450 µs/put even
   with 15 concurrent lanes) is the remaining wire floor; sub-7 ms 16n combine
   wire is out of env/schedule scope. (Candidate directions: IBGDA-style
   device-initiated puts if CXI ever supports it, multi-PE proxy, or NCCL for
   the wire leg.)
2. **Dense production-kernel restructure** — COMET's remaining b8 floor
   (17.06 l1 at 16n) is `ep_topk_gather_rs_kernel_v2`'s fixed wave structure.
3. Gen-4 BUILT (2ad1176c); the 8n K2 b32–b64 bridge is ns2 (ns3 not constructible: 7168 % 3 != 0).

## 7. Orchestration lessons (delta over handoff 15)

- **Discriminator-before-patch works**: two patches were avoided outright
  (conv-streams refuted by nsys; aggregation mooted by ns1-in-env) and the
  demotion fix was correctly scoped by source verification before any build.
- **Rebuild cadence**: never rebuild while any lane's batch can be mid-flight
  (a .so swap splits a batch across binaries); announce PENDING/complete in
  the status log; keep S=2/default-path bit-identity so anchors survive
  generations; verify tags in the RIGHT library (ths_op).
- Peer-lane cross-relay through the orchestrator (round model, knee, conn
  guard, demotion warning) repeatedly saved allocation minutes; the 16n lane
  burning zero cells on poisoned K2 ns2 rungs paid for the whole relay
  overhead.
- Driver-direct A/B + capsule-only records (one allocation per verdict set)
  kept the whole optimization at 24.3 nh.
