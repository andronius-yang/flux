# 07 — Comm-only layer-axis campaign (2026-08-16)

The first sweep campaign with a **layer dimension**: layer0 (dispatch), layer1
(combine), and the combined continuous layer0+GELU+layer1 pass, all on real
Qwen3-235B trace routing (`mmlu/high_school_world_history`, layer 92, homog,
topk=8, G=128, H=4096, bf16), budgets 1–64 MiB, isolated mode, Perlmutter.
Comm-only: no MoonEP/UltraEP/EPLB arms. This doc is the campaign authority;
`sweeps/SCHEMA.md` governs interpretation. Every number below is
capsule-traceable; W=8 (2n) and W=16 (4n) results are never mixed.

## Executive summary

1. **Layer0 factorial verdicts (W=16, verdict-grade — three independent runs,
   sign agreement)**: on the lb_union base, `FLUX_A2AV_FUSED_STAGE2` **wins**
   (−0.2…−0.7 ms), `FLUX_A2AV_EARLY_LAUNCH` **wins** (−0.3…−1.8 ms, grows
   with budget), `FLUX_A2AV_FANOUT` **loses** (+0.05…+0.6 ms — the
   `hier_compress_lb_union_eager` A/B arm finally has its verdict: delete).
   **Canonical layer0 config: `lb_union + FUSED_STAGE2 + EARLY_LAUNCH`** —
   beats stock flux (allgather) by ~25–27% at b1–b8, tapering to −23…−24%
   (b16), −19% (b32), −15% (b64); beats unfused torch by up to −64% (b16)
   and FAST at every budget.
2. **Layer1 verdicts (W=8, binary C)**: plain `a2av_hier` is the winner —
   ~2× faster than dense ring-RS everywhere, 1.4–2.2× faster than unfused
   torch (b1–b16). **The eager arrival-order reduce is a 15–35% REGRESSION**
   on trace routing; compress ≈ hier at W=8 (dedup should grow with node
   count — 4n data pending). Schedule inheritance (tmiso→tmamo) is worth a
   flat ~0.3 ms on hier, 1.5–3.8 ms on compress.
3. **Combined pass (W=8, binary C)**: best pairing =
   `lb_union(F+E) + plain hier`: **9.93 ms at b8 vs 15.87 stock and 17.50
   torch (−37% / −43%), and −37…−51% vs stock across b1–b16**, decomposition
   identity clean (−1%). The original
   eager pairing violated the identity (+18% at b8): ablation attributed the
   entire penalty to the l1 eager persistent reduce (early-launch
   exonerated) — eager is doubly disqualified (slow alone AND composes
   badly).
4. **Two real bugs found, root-caused, and fixed** by this campaign's
   bring-up discipline (details §5): the lazy-load spin-kernel deadlock
   (binary B fix) and the empty-expert split-cascade hang (binary C fix) —
   the latter explains why layer1 had never survived real trace routing at
   any scale.
5. Baseline feasibility boundaries confirmed: unfused torch reference OOMs
   at b32+ (its curve legitimately ends at b16); FAST hits the 16G
   symmetric-heap ceiling at b64 on both layers (documented loss).

## 1. Capsule ledger (all committed; one build per comparison set)

**Binary A** (pre-fix, sha `d69ceee…`/ths-op `355de11b…`):
- P2 bring-up 2n: `20260816-123413` (l1 identity check), `20260816-125015`
  (clean arms), `20260816-131627` (factorial 8-corner smoke, 8/8 green).
- P3 layer0 4n: `20260816-132227` (blow fwd 45/45), `20260816-133612`
  (bhigh fwd 18/18), rev twins ×2 drivers: `20260816-161551` + `-161758`
  (45/45 each), `20260816-162940` + `-163134` (18/18 each) — the duplicate
  rev pairs are two independent drivers of the same specs on one binary
  (bonus ordering repeats).
- P3 fast: l0 `20260816-134133` (5/5) + `-134308` (b32 ok, b64 heap-ceiling
  fail); l1 `20260816-163504` (5/5) + `-163640` (b32 ok, b64 fail).
- Documented negatives: `20260816-134402` (l1 4n, 20/20 stuck — later
  root-caused as the empty-expert bug, NOT node count), `20260816-170140`
  (binary-B 4n smoke still stuck — the discriminator that separated bug #2
  from bug #1).
- The one-shot BvN `schedule_ms` in fast cells ≈ 0.9 ms flat on Perlmutter —
  floors its small-budget cells (quote alongside b1/b2 comparisons).
- b1–b16 layer0 cells carry the un-overlapped torch reference as
  `impl=torch` rows (comm+scatter+gemm; OOM-bounded above b16).

**Binary C** (both fixes, 2n sub-campaign, W=8):
- l1 five arms fwd+rev: `20260816-172012` (50/50; provenance: launched on
  binary B, every trace cell stuck, ALL data-producing attempts post-swap on
  C — audited via start_ts) + `20260816-185918` (bhigh 20/20);
  rev `20260816-191205` (50/50) + `20260816-192706` (20/20).
- l0 companion (identity siblings): `20260816-190517` (10/10).
- l01 combined (eager pairing) fwd+rev: `20260816-190804` + `20260816-193310`
  (15/15 each).
- l01 corrected pairing (hier) fwd+rev: the two capsules from specs
  `commsweep2n_l01hier_blow{,_rev}` (5/5 ok each, allclose, budget order
  reversed between them).
- 4n binary-C l1+l01 campaign: parallel session's lane, in flight.

## 2. Layer0 (W=16, binary A) — the factorial

Isolated max-rank e2e_ms (fwd / revA / revB), selected budgets:

| arm | b2 | b8 | b32 | b64 |
|---|---|---|---|---|
| torch unfused (impl rows) | 3.34 | 12.78 | OOM | OOM |
| fast (dealer arm; +0.9 ms BvN) | 5.09 | 11.62 | 40.1 | heap-fail |
| allgather (stock flux) | 2.30/2.29/2.27 | 6.11/6.65/6.26 | 23.8/24.1/23.8 | 52.2/51.4/52.0 |
| lb_union (base) | 2.00 | 4.72 | 19.7 | 45.7 |
| **+F+E (fused_early)** | **1.67** | **4.47/4.67/4.43** | 19.2–19.4 | 44.4–44.7 |
| +F+N+E (triple) | 1.72–1.81 | 4.44–4.46 | **18.7–19.0** | 43.9–44.5 |

torch/fast quoted from their own rows/capsules (single driver each; torch
b1–b16 only). Baseline shape: torch beats fast below b8; fast crosses over
at b8 (11.6 vs 12.8) and holds to b32; the winner config dominates both
everywhere (e.g. b16: 8.9–9.2 vs torch 25.5, fast 20.7, stock 11.5–11.8).

(Full 9-arm × 7-budget × 3-run table: `factorial_analysis.py` output in the
capsules; base numbers quoted from fwd, spreads across runs ≤ ~2% except
flagged single-cell transients.) Main effects with three-run sign agreement:
F −0.19…−0.71 (agree b1,b2,b4,b8,b32); E −0.06…−1.8 (agree b2,b16,b32,b64);
N +0.04…+0.56 (agree b2–b16). The triple's occasional edge over fused_early
rides an N×E interaction — not verdict-grade since N loses alone. NR-14
lesson held: two arms showed 3–6 ms single-cell outliers at b16/b64 in one
run each; only the three-run agreement rule kept them out of the verdicts.

**Canonicalization recommendation**: default-on `FLUX_A2AV_FUSED_STAGE2` and
`FLUX_A2AV_EARLY_LAUNCH` for lb_union; delete the `_eager` (FANOUT) A/B arm.

## 3. Layer1 (W=8, binary C)

Isolated max-rank e2e_ms, trace, fwd/rev agree ≤2% (quoting fwd):

| arm (tm=iso) | b1 | b8 | b16 | b64 |
|---|---|---|---|---|
| torch (unfused) | 3.44 | 9.47 | 16.5 | OOM |
| l1_dense (stock ring-RS) | 3.63 | 11.07 | 22.6 | 90.0 |
| **l1_hier** | **1.60** | **6.00** | **11.9** | **50.2** |
| l1_hier_eager | 1.86 | 7.82 | 15.9 | 67.3 |
| l1_compress | 2.78 | 7.47 | 13.6 | 53.5 |

`l1_fast` is excluded from this table — its capsules (`163504/163640`) are
**W=16**: 3.96/5.13/6.27/11.11/21.0/42.4 ms at b1–b32 (b64 heap-fail),
dealer arm, e2e≡iso. It currently stands alone at W=16 (the flux l1 4n
cells were the binary-A negatives); the pending 4n binary-C lane makes it
comparable. With inherited indices (tmamo),
hier drops to 1.30/5.67/11.6/49.7 and compress to 1.29/5.69/11.5/49.7 —
compress's in-forward CSR build is its main iso-mode cost at W=8; its wire
dedup breaks even here (2 nodes = little inter-node fan-in to dedup).

## 4. Combined layer0+1 (W=8, binary C)

Isolated max-rank e2e_ms of the full pass (l0 → GELU → l1), fwd/rev:

| pairing | b1 | b2 | b4 | b8 | b16 |
|---|---|---|---|---|---|
| torch+torch | 6.52 | 6.96 | 11.4 | 17.5 | 32.1 |
| allgather+dense (stock) | 5.06 | 5.73 | 9.90 | 15.9 | 32.0 |
| lbunion(F+E)+hier_eager | 2.83 | 4.58 | 7.79 | 14.6 | 24.0 |
| **lbunion(F+E)+hier** | **2.47** | **3.46** | **5.51** | **9.93** | **20.1** |

(hier-pairing capsules: `commsweep2n_l01hier_blow{,_rev}` runs, fwd/rev
within 0.5% at every budget; vs stock −51/−40/−44/−37/−37%; identity
residual −1.2% at b8, ~0% at b16.)

Decomposition identity (combined ≈ l0-isolated + GELU + l1-amortized,
tolerance ±10%): torch +0.9%, stock −1.0%, eager pairing **+18.3% at b8**
(FAIL — grows b1→b8, gone at b16), hier pairing −1% (ablation). Attribution
ablation at b8: eager-off = 9.95 ms, early-launch-off = no change ⇒ the l1
eager persistent reduce kernel is the entire composition penalty (it
occupies SMs/issue slots while the l0 stage of the window still needs the
device; fixed-cost signature explains the vanishing residual at b16).
`l1_index_build_ms` (one-shot python builders, untimed by decision) is
reported in every l01 cell's info record.

## 5. The two bugs (both fixed; full narratives in the memory ledger)

1. **Lazy-load spin deadlock** (binary B fix, `worktree-l1-hang-debug`
   1550b67): `CUDA_MODULE_LOADING=LAZY` first-launch module loads enqueue
   behind the never-exiting persistent spin kernels that l1 eager and
   compress introduce; NVSHMEM 3.2.5 delivers every on-stream signal via a
   device kernel, so the signal producer can never load ⇒ circular wait.
   Fix: ctor-time `cudaFuncGetAttributes` preload + NVSHMEM transport
   priming. Found because compress's first-ever GPU run was gated by the
   bring-up ladder.
2. **Empty-expert split-cascade hang** (binary C fix, `worktree-l1-nn4-debug`
   c9b82b6): `set_barrier_ptr`'s split cascade divides the full problem list
   by the non-empty problem count — any local expert with zero routed rows
   mis-buckets completions and the per-split barrier never fires. Hangs
   EVERY l1 arm (dense included) on any matrix containing empty experts —
   i.e. all real-trace matrices at these shapes — at any node count. This
   masqueraded as an "NN>2 bug" because synthetic bring-up matrices have no
   empty experts. Audit rule: when trace cells hang, `bincount` the routing
   for empty experts FIRST.

## 6. Open items

- 4n binary-C l1 + l01 campaign (parallel session, in flight) — completes
  the W=16 layer1/combined story, incl. whether compress's dedup pays at
  larger node counts (the formalization doc's open question 6).
- `l01_fast` (fast+fast combined) blocked on the FAST credit-reset question;
  `l1_fast` b64 heap ceiling documented.
- Canonicalization patches (knob defaults F/E on; delete FANOUT arm) not yet
  applied to variants/kernel defaults — do after the 4n lane lands so the
  whole campaign shares one variant table.
- gemm_rs sibling `set_barrier_ptr` (same pattern as bug #2) unaudited.
- 16n W64 closure cells (pre-campaign debt) remain unrun — separate campaign.
