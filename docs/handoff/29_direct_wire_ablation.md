# Handoff 29 — direct-wire transport ablation: why EPLB owns 16n b1 (2026-08-30)

**One-line:** the 16n low-budget loss of every fused arm is the
hier_compress/slipstream stage chain's hop-count-scaling fixed cost, not
planning and not placement; a new ablation arm drives OUR plan lane
(pv2 + LocCap + r2) over the exact staged All2AllSingle wire the eplb_l01
arm uses, and at 4n it reproduces EPLB's transport bit-for-bit while
beating EPLB on plan cost alone.

## 1. Diagnosis (question: why does EPLB crush us at 16n b1?)

Measured ledger, K2 b1 isolated (handoff-18 capsules + 8/29 pv2 regen):

| arm | e2e | l0 | l1 | plan+comm | total |
|---|---|---|---|---|---|
| EPLB staged a2av (16n) | 5.53 | 3.33 (wire 2.86) | 2.19 (comb 1.70) | 4.21 | 9.74 |
| Torch+GEMM (16n) | 6.79 | 2.18 | 4.37 | 0.75 | 7.49 |
| ours pv2_r2 (16n, 8/29) | 12.01 | 7.07 | 4.93 | 2.08 | 14.11 |

The deficit is entirely the transport chain. The user's "extra jumps"
hypothesis holds with one refinement: at b1 the extra *copies* are
microseconds — what costs milliseconds is the extra stages as
**serialization** (gateway relay put rounds at ~340–450 µs/proxy-put,
per-stage signal/barrier chains, the combine wave/receiver structure),
which scales with NN:

| component | 4n → 8n → 16n (K2 b1) |
|---|---|
| fused l0 (ours) | 1.65 → 2.67 → 7.07 |
| fused l1 (ours) | 2.79 → 3.11 → 4.93 |
| staged direct wire (eplb comm) | 1.52 → 2.05 → 2.86 |
| staged comb (eplb) | 1.07 → 1.52 → 1.70 |

Direct wire = one staging memcpy + W−1 *concurrent* nbi_block puts + two
team barriers → NN-flat-ish. Compression/dedup buys bytes that are worth
nothing at b1 while paying that latency structure. (Handoff 16 already
flagged the boundary: "b1(–b2 at 16n) belongs to torch".)

## 2. The ablation arm

`--wire direct` on `test_moe_ours_traffic.py` (s1-only):
- plan lane UNCHANGED: pv2 placement, LocCap routed kernel, fused
  phys+probs allgather, OursIterPlanner (graphs stay on).
- wire = `flux/testing/ours_direct.py` (`OursDirectRunner`, driver
  interface parity): vce → phys → dest-major stable sort →
  `direct_layout_entries_fast` (the eplb fast-tail layout, one batched
  pinned D2H, timed in the plan bracket) → pack → `flux.All2AllSingle`
  NVSHMEM a2av (hidden + fp32 probs) → place → per-segment `GemmOnly`;
  combine mirrored on the same op pair with swapped splits +
  deterministic comb_dst home reduce. No signal gating (put+barrier
  publication — invariant-5 clean); payload randomized per iteration.
- arms: `ours_l01_s1_pv2_r2_dwire` (+`_gate`), conn pinned 8; heap via
  the eplb row-sum bound (sweep.py ours-branch override).
- diagnostic metrics: `dwire_{pack,wire,place,gemm0,gemm2,cpack,comb,acc}_ms`.
- unit test: `test_ours_direct_plan.py` (CPU, adapter vs brute force).

Gates 4/4 green (4n K2+Qwen b1+b8, per-iter output checks + correctness,
capsules `20260830-073401_8ec57462` / `-073752_250b7c8b`). Gotcha: gate
cells' per-iteration reference inflates wire ~2x — never quote gate
timing (the first gate attempt also crash-taught: the setup audit and
`emit_info` had fused-only attrs; both now branch on the wire mode).

## 3. 4n A/B verdict (capsules `20260830-082548_d687da8e` K2 / `-083150_3a993069` qwen, one binary, committed)

e2e (total) at 4n:

| arm | K2 b1 | K2 b8 | qwen b1 | qwen b8 |
|---|---|---|---|---|
| ours fused | **3.31 (4.13)** | **8.22 (9.12)** | **2.24 (3.04)** | **5.80 (6.88)** |
| ours dwire | 4.74 (6.18) | 18.48 (20.04) | 3.41 (4.85) | 17.08 (18.71) |
| eplb_l01 | 4.82 (7.98) | 18.36 (21.59) | 3.38 (6.48) | 17.29 (20.46) |

1. **Port faithful**: dwire e2e ≡ eplb_l01 at every budget/model; wire
   and comb sub-phases match ±0.03 ms.
2. **dwire beats EPLB by 1.6–1.8 ms total at every cell — purely plan**
   (ours 1.1–1.3 + 0.2 vs EPLB 2.7–2.9 + 0.3). This margin is what our
   placement/routing is worth on EPLB's own transport.
3. **Fused wins 4n everywhere** (few hops → small stage-chain cost; the
   un-overlapped wire+GEMM serialization costs dwire ~1.4 ms at b1 and
   grows with bytes) — consistent with, and predicted by, the diagnosis.
4. dwire l1 beats fused l1 at K2 b1 (2.09 vs 2.79): single w2 pass per
   expert, no per-wave re-read (the handoff-26 floor sidestepped).

## 4. 16n (the actual question) — HANDED OFF, specs ready, prediction registered

`sweeps/specs/dwire_ab_16n_{k2,qwen}.yaml`: pv2_r2 vs dwire vs eplb_l01,
**b1/2/4/16/64** (full ladder, user 2026-08-30; budget-major cell order
so low budgets land first if a window cuts the tail), isolated, one
capsule per model. Prediction registered BEFORE any 16n run:

- dwire ≈ 7.5–8.5 ms total at K2 b1 (transport ~5.5 e2e + our plan
  ~1.5–2.1) vs fused 14.1 and EPLB ~9.5–10 → **flip vs EPLB at b1/b2 in
  our favor, ~85–90% confidence**; flip vs our own fused arm >95%
  (fused l0 alone is 7.07 at 16n b1).
- Mechanism the numbers must show: dwire e2e ≡ eplb_l01 e2e (same
  transport), dwire total = EPLB total − (plan-lane gap ~2 ms ± wire
  delta from pv2 node-awareness).
- Falsifiers: dwire wire ≫ eplb wire in-capsule (r2/loccap pair-pattern
  skew at 64 ranks), or EPLB's planner much cheaper than its 8/24 16n
  reading (partially defused: the 4n in-capsule plan gap was measured on
  THIS binary).

The 8/30 origin session queued 16n three times (gpu_regular congested,
2h+ pending, no backfill estimate at -t 45 or -t 30) and CANCELLED its
final request (57735507) when the user reassigned the run to another
session's 16n sweep — see §5.

## 5. Takeover run-book (for the session running the 16n sweep)

Everything is committed on `pv2` — code `8ef4406`, spec ladder + this
section in the follow-up commit; capsules `33f2148` (gates) and
`786fb5c` (4n A/B). **All changes are python/spec-level — no C++
touched, no rebuild needed**; the editable install picks up
`flux/testing/ours_direct.py` automatically. `git pull` on the same
checkout is sufficient.

How to run (either form is fine):
1. As-is: `python3 sweeps/sweep.py run --spec sweeps/specs/dwire_ab_16n_k2.yaml
   --jobid <id>` then the qwen twin. 15 cells each, all three arms
   in-capsule (rule 4 satisfied per model).
2. Folded into your own 16n specs: add
   `ours_l01_s1_pv2_r2_dwire` and `eplb_l01` to your `variants:` list.
   Keep `ours_l01_s1_pv2_r2` in the same capsule — it is the comparator.
   Do NOT hand the dwire arm extra env: its variant already pins conn=8
   and the sweep sizes its heap via the eplb row-sum bound
   (`sweep.py` ours-branch override — verify dry-run shows the eplb-style
   2/4/9/16/16G ladder for dwire, NOT the fused arm's flat 6G).

Time budget (confidence for one-go): measured 4n cell rates + 8/24 16n
history give **K2 ≈ 25–35 min, qwen ≈ 20–30 min** — b1–b4 cells are
40–90 s, b16 ~2–3 min, b64 the long pole (eplb_l01 K2 16n b64 ran e2e
~211 ms/iter on 8/24; expect 4–6 min/cell incl. the first-eplb-cell
load-sidecar generation). **One allocation of -t 75–90 covers both
models comfortably; -t 30 covers roughly one model's b1–b16.** Cell
order is budget-major, so a cut window still yields the b1/b2 verdict.

Per-budget confidence the cells run clean: b1–b4 ~95% (gates + 4n A/B
green, sizing verified in dry-run), b16 ~90%, b64 ~75–80% — the 16G
at-cap heap class (same as eplb_l01, which DID run 16n b64 on 8/24;
dwire's real staging is max_split-based, far under the row-sum prior).
Failure modes are all loud/clean: nvshmem_malloc failure or the
per-iteration `max_split`/`recv_cap` asserts → `failed` cell, no hang
class known. Never quote `_gate` cells for latency (their per-iteration
reference inflates the wire ~2x — measured).

After the run: check dwire e2e ≡ eplb_l01 e2e per cell (transport
identity — if it drifts, something is wrong with the fold, stop and
compare `dwire_wire_ms` vs eplb `comm_ms`); then judge §4's prediction
on total_ms. Update this handoff §4 with the verdict and flip the
memory entry `dwire-transport-ablation` to COMPLETE.

## 6. 16n VERDICT (2026-08-30, campaign handoff 30, binary 1b5593d4)

§4 prediction **CONFIRMED on both models**. K2: dwire b1 total 7.488 (predicted
7.5-8.5), EPLB 9.989 (predicted 9.5-10, exact), fused ours 12.625 (predicted ~14);
**flip vs EPLB b1 -25.0%, b2 -21.4%**; fused reclaims from b4+ (b8 20.5 vs 27.9).
Qwen: identity tight (-0.4..-5.0%), dwire wins b1 -18.2% / b2 -13.9%. Wire legs
identical (K2 b8 13.134 vs 13.171); dwire overall e2e 7-10% faster than eplb at
b4-b16 — the delta sits outside the wire leg (recorded, not debugged). Falsifiers
did not fire. New: dwire 16n b32 = wedge-class kill (exit 143, both tries, both
models); b64 = NVSHMEM_MALLOC heap wall as §5 priced (~75-80% confidence borne out).
Full dataset: handoff 30.
