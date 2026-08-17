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
2. **Layer1 verdicts (binary C)**: plain `a2av_hier` is the standalone
   winner at small/mid budgets — ~2× faster than dense ring-RS everywhere
   (W=8 and W=16), 1.4–2.2× faster than unfused torch. **The eager
   arrival-order reduce is a REGRESSION at both scales** (15–35% at W=8;
   +2%→+11% at W=16, growing with budget). **Compress crosses over at
   W=16**: loses iso at b1–b4 (in-forward CSR build), wins −4/−6% at
   b32/b64 (−17/−14% with inherited indices) — dedup-merge on the combine
   wire pays at scale under real routing (§3.1). Schedule inheritance
   (tmiso→tmamo) is worth a flat ~0.3 ms on hier, 1.5–5 ms on compress.
3. **Combined pass (binary C)**: at W=8 the corrected pairing
   `lb_union(F+E) + plain hier` gave 9.93 ms at b8 (−37% vs stock), after
   the eager pairing violated the decomposition identity (+18% at b8;
   ablation pinned the entire penalty on the l1 eager persistent reduce).
   **At W=16 the best-pairing A/B crowns `lb_union(F+E) + compress` at
   EVERY budget** (fwd/rev agreement): the combined window inherits the
   CSRs, so compress's build penalty vanishes and the wire dedup remains —
   **10.6 ms at b8 vs 21.9 stock (−52%) and 24.8 torch (−57%)**; −45..−52%
   vs stock across b2–b64 (§4.1).
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

### 3.1 Layer1 at W=16 (4n, binary C — capsules 96f807d1/9725d5e0 fwd, a13c43be/0c055e74 rev; all allclose)

Isolated max-rank e2e_ms (fwd; rev agrees ≤2% except noted; tmamo in parens):

| arm (iso) | b1 | b4 | b8 | b16 | b32 | b64 |
|---|---|---|---|---|---|---|
| torch (unfused) | 2.86 | 6.62 | 11.9 | 22.3 | OOM | OOM |
| l1_dense (stock) | 5.19 | 10.1 | 15.7 | 31.4 | 59.8 | 122.5 |
| **l1_hier** | **1.95** (1.64) | **4.25** (3.92) | **7.73** (7.42) | **14.6** (14.4) | 29.4 (30.0) | 61.4 (61.5) |
| l1_hier_eager | 1.95 | 4.44 | 7.91 | 15.9 | 32.9 | 68.0 |
| l1_compress | 3.05 | 5.03 | 7.93 | 14.4† | **28.6** (24.9) | **58.1** (53.2) |
| l1_compress_eager | 3.03 | 5.68 | 9.79 | 18.1 | 34.6 | 70.1 |
| l1_fast (W=16, dealer) | 3.96 | 6.27 | 11.1 | 21.0 | 42.4 | heap-fail |

W=16 verdicts (fwd/rev sign agreement vs l1_hier):
- **eager: LOSS at scale too** — +2% (b8) growing to **+11% (b32/b64)**;
  the 2n verdict generalizes.
- **compress CROSSES OVER**: +54/+19% at b1/b4 (in-forward CSR build), ~par
  at b8 (†b16 fwd favors compress −0.28, rev +0.68 — no verdict), then
  **WINS −4% (b32) and −6% (b64)**; in tmamo (inherited indices) the win is
  −17% (b32) and **−13.6% (b64)** — dedup-merge on the combine wire pays at
  W=16 large budgets under real routing (formalization open question 6:
  answered for the combine direction).
- dense ≈ 2× hier everywhere; torch loses to hier from b1.

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

### 4.1 Combined at W=16 (4n, binary C)

Eager pairing + baselines (capsules 537522b3/5282d34f fwd, 9bd629be/b002806a
rev; all allclose where correctness ON):

| pairing | b1 | b2 | b4 | b8 | b16 | b32 | b64 |
|---|---|---|---|---|---|---|---|
| torch+torch | 4.96 | 7.49 | 13.4 | 24.8 | 48.3 | OOM | OOM |
| allgather+dense (stock) | 6.84 | 7.66 | 13.5 | 21.9 | 43.1 | 85.2 | 177.3 |
| lbunion(F+E)+hier_eager | 3.54 | 4.64 | 8.00 | 15.2 | 25.1 | 53.9 | 113.2 |

Best-pairing A/B (capsules 9378fed5 fwd / 397ac0fa rev; compress-pairing
correctness gated at 4n first):

| pairing | b1 | b2 | b4 | b8 | b16 | b32 | b64 |
|---|---|---|---|---|---|---|---|
| lbunion(F+E)+hier | 2.86/2.91 | 4.81*/4.06 | 6.87/6.66 | 12.5/12.5 | 24.9/24.8 | 52.0/51.8 | 112.4/111.7 |
| **lbunion(F+E)+compress** | **2.76/2.70** | **3.63/3.66** | **5.89/5.86** | **10.6/10.6** | **21.2/21.2** | **46.2/45.3** | **99.6/100.4** |

(*single-iteration transient in the fwd cell, max 11.5 ms; the rev twin is
clean.) **The compress pairing wins at EVERY budget with fwd/rev agreement**
— −10..−15% vs the hier pairing at b2–b16, −11..−12% at b32/b64, and even
b1: the combined window inherits the CSR indices (amortized semantics), so
compress's isolated-mode build penalty vanishes and the wire-dedup win
remains. Note the W=8 combined table above predates the W=16 compress
verdict; at W=8 combined, hier vs compress was not measured — the W=16 A/B
is the authority for the pairing choice.

**Campaign combined headline (W=16)**: `lb_union(F+E) + compress` = **10.6 ms
at b8 vs 21.9 stock flux (−52%) and 24.8 torch (−57%)**; −45..−52% vs stock
across b2–b64.

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

- ~~4n binary-C l1 + l01 campaign~~ LANDED (§3.1/§4.1): W=16 five-arm l1
  pairs, l01 eager-pairing + best-pairing A/B, l1_fast completion — compress
  dedup pays at W=16 (l1 iso b32/64; combined at every budget).
- `l01_fast` (fast+fast combined) blocked on the FAST credit-reset question;
  `l1_fast` b64 heap ceiling documented.
- ~~Canonicalization patches~~ LANDED 2026-08-17 (commit 58efc02 + smoke
  capsule 20260817-025014): F+E default-ON under LB_UNION in the binary
  (E gated on conn>1 so bare launch.sh conn=1 runs keep historical
  behavior); FANOUT and RS_EAGER arms KEPT but marked CLOSED-LOSER (user
  directive: mechanisms retained, case closed, ablation-only);
  explicit-off ablation arms added (_nofused/_noearly/_nofused_noearly);
  `l01_lbunion_compress` promoted to reference combined config. 2n smoke
  (binary D, ths-op sha a690c9e6): both arms ok+allclose, base (silent F+E)
  1.241 ms vs explicit-off 1.542 ms at b2 uniform — defaults demonstrably
  engaged. env_json boundary note in variants.py/SCHEMA.md/SKILL.md.
- ~~gemm_rs sibling `set_barrier_ptr` unaudited~~ AUDITED 2026-08-17:
  **UNREACHABLE on every default/deployed path** — the MoE trigger (one
  empty segment among nonempty) is structurally impossible in gemm_rs
  (`m % world_size` FLUX_CHECKed, reduce_scatter_kernel.hpp:2190), and the
  PCIe swizzle's genuinely-empty segments are explicitly skipped. ONE latent
  exact analog of the c9b82b6 mismatch exists at
  `sm89/gemm_with_absmax.h:843-848` + `epilogue_evt.hpp:358-363` (counter
  indexed by tile ROW, target counts a whole RANK SEGMENT — missing
  `/ segments_per_rank`), reachable only via the non-default, warned-against
  `use_gemmk=true` + `per_tile_flags=false` combo in kPcieMode (never
  entered on Perlmutter/AWS NVLink) with `m_per_rank > kM`; minimal fix =
  `seg = segment_idx / segments_per_rank` in both files if PCIe ever
  matters. Adjacent notes: m=0 hangs the PCIe waiters (no `m > 0` guard);
  `tiled_m < world_size` floor-division hazards at epilogue_evt.hpp:397 and
  sm89:827 (currently un-dereferenced / guarded only on the ws==8 queue
  path). Not fixed — unvalidatable on our platforms; recorded here so the
  audit stops being listed.
- 16n W64 closure cells (pre-campaign debt) remain unrun — separate campaign.
