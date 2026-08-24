# 18 — Authentic l01 data-point campaign (2026-08-24)

**One-line:** 7 baselines x {K2, Qwen} x {4n, 8n, 16n} x b1–b64, isolated-mode
dispatch+combine (l01), one binary campaign-wide (the 08-24 07:01 build of
17cb7bb: Slipstream v2 canon + ns canonicalizations). **294 cells: 285 ok, 0
failed, 9 pre-skipped** (known OOM classes, reasons in the tidy CSV).
Sibling: `18_authentic_l01_results_tidy.csv` (one row per cell).

## What this campaign is
User-directed successor to datacamp (handoff 15) after three changes landed:
1. **Slipstream v2** is the official Slipstream (SCHEMA rule 13) — M-split
   destination-wave combine + fpwp + bucketed receiver. `l01_slipstream` runs
   v2; every capsule requires `FLUX_A2AV_SLIPSTREAM2_TAG`.
2. **ns canonicalizations (2026-08-24 user decisions):** COMET
   (`l01_allgather_dense`) dense l1 ns=2 (`FLUX_RS_NSPLIT_512_TAG`); EPIC
   (`epic_l01_hc_m1`) and PLL (`llc_l01_s1`) staged l1 ns=1.
3. **LLC recv-bound fix** (commit 3e6e8ed) in the binary.
Same s1-canon inputs as datacamp (rule 10: dslots 64:32, oracle g=0, same
content-addressed matrices/routing/oracles) — torch and MoonEP arms reproduce
datacamp anchors within +/-10% at every topology (gate table in the capsules'
status log). NEVER-MIX vs datacamp: COMET/EPIC/PLL (ns flips), Slipstream
(v1 -> v2), llc at forced-heavy cells (recv-bound fix).

## Aggregation conventions
Same as handoff 15: e2e/plan/plan_comm/total = per-iter MAX across ranks,
MEDIAN of 10 iters; l0/act/l1 = the critical rank's spans (rank with max e2e
that iter), so l0+act+l1 == e2e. MoonEP emits no e2e/l0/l1: e2e = total −
plan_comm − plan; l0 = pack+comm+scatter+prefetch+gemm; l1 =
gemm2+cpack+comb+acc. FAST excluded (broken, left for its own session).

## Headline findings
1. **The datacamp 16n inversion is CLOSED by Slipstream v2.** Datacamp 16n b8
   K2/Qwen had Slipstream v1 at 52.1/38.3 total (beaten by Torch+GEMM);
   today v2 posts e2e 21.6/21.5 — fastest or tied-fastest arm at every
   topology and budget, −38%/−39% vs COMET at 16n b8.
2. **COMET ns=2 halves its small-budget l1 at 4n** (K2 b1 e2e 6.8 -> 3.6;
   l1 5.1 -> 1.8) and wins 4n b8 K2 (8.53 vs Slipstream 8.92), but scales
   worst of the fused arms (16n b8 35.1/34.7; 16n b64 ~230 both models).
3. **PLL (ns=1 + recv-bound fix) is the fastest placement arm everywhere**
   (16n b8 21.43/21.26 — statistically tied with Slipstream) and its Qwen
   16n b4–b32 cells — the datacamp recv-bound failures — now pass 6/6
   (validation point for 3e6e8ed).
4. Placement ranking PLL < EPIC < EPLB < MoonEP holds at 4n/8n; at 16n EPLB
   closes on EPIC (27.1 vs 23.2 b8 K2) but no longer beats the comm arms
   (datacamp's EPLB-first 16n ranking was a v1-Slipstream artifact).

## Capsules (42, uncommitted by the lanes; commit with the tidy CSV)
4n (jobid 57529115): ad0c03ab 2397fca5 506ad7af 0de1bd1d 3e868187 63c27792
3681d4a5 618cfaac c69340fb 5260755d 41746609 c8dfe3ed d47c15f2 7ce25917
8n (57530880/57531861/57533162): 4b4467e8 aaeb2b14 7e54e5ec 625fe535 b61c1890
5b578c99 0d5e000e 901b5618 095c1643 16f98932 062d03f9 a78bf560 f1b1c303 472b596a
16n (57529125): 8dc95910 995de4bd b1bac7cc 4adf476a 21645f00 2b0189fb 0b408e25
89f73bcc 5948c807 26f674a7 bb526f7f 38fe5d32 b7468c6e f4796901
(all prefixed 20260824-*_perlmutter_.)

## Cost + incidents
| Job | QOS | Nodes | Elapsed | nh |
|---|---|---|---|---|
| 57529115 | interactive | 4 | 00:50:39 | 3.38 |
| 57529131 | debug | 8 | 00:30:03 | **4.01 wasted** (window granted, lane missed the wake, idled to timeout) |
| 57530880 | debug | 8 | 00:19:24 | 2.59 |
| 57531861 | debug | 8 | 00:25:08 | 3.35 |
| 57533162 | debug | 8 | 00:08:01 | 1.07 |
| 57529125 | regular | 16 | 01:27:13 | 23.26 (incl. **~7.7 wasted**: inv-1 completion missed, idle 08:51–09:20) |
**Total 37.66 nh (25.9 productive, ~11.7 idle-burn across the two incidents).**
Process lesson for the next campaign brief (extends handoff 15 §5): lanes must
NEVER park on background-notification waits for salloc grants or runner
completions — poll `squeue`/foreground the runner; both incidents were
missed-wake idle burns, caught only by the orchestrator's cross-checks.
