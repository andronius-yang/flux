# Main performance figure — `figure_src.csv` provenance notes

Generated 2026-09-01 by locating every manually recorded total_ms value in the
sweep capsules. Grid: {4n, 16n} x {Qwen, K2} x figure rows x budgets
{1, 2, 4, 16, 64} MiB. 208 recorded values were checked; **205 matched a
capsule statistic within <= 0.006 ms**, the other 3 are in the discrepancy
ledger below, and the 2 recorded-NA cells were found and filled.

**v2 addendum (2026-09-03): 8-node rows added** — the postdoc's offline 8n
grid (10 rows x 5 budgets x 2 models = 100 values) was resolved the same
way: **100/100 matched a capsule statistic exactly** (90 on `main`, the 10
FAST cells on `fast-split`), no discrepancies. Row-label note: the 8n grid
calls its reference row "Torch a2av (not ag) + GEMM"; every one of those
cells matched `l01_nvshmem` (the blocking-put NVSHMEM ring a2av +
un-overlapped GEMM — the arm that took over the legacy "Torch+GEMM" slot
on 8/31), i.e. the SAME arm as the 4n/16n "NVSHMEM a2av + GEMM" row, so it
carries `row_id = nvshmem_gemm` (flag `label_torch_a2av_is_l01_nvshmem`).
The 8n "2Ours(expert balance + Routing)" row is `llc_l01_s1_pv2` (hier
a2av, no overlap) like the other topologies; there is no direct-a2av row
at 8n. Capsule mix per row is recorded per cell; 8n nvshmem cells are
the runner-console mean (`iter_max_mean`), everything else the campaign
median, as at 4n/16n.

## Columns

- `total_ms` — the authoritative value to plot (3 decimals, recomputed from the
  capsule's raw `metrics.csv`; equals the recorded value at its precision
  except for the flagged corrections).
- `recorded_ms` — the manually recorded value, verbatim.
- `stat` — how the raw per-(rank, iteration) rows aggregate:
  `iter_max_median` = per-iteration MAX across ranks, MEDIAN over iterations
  (the datapoint-campaign convention, handoff 15/18);
  `iter_max_mean` = same but MEAN over iterations (the runner's console
  summary — what was on screen when those rows were recorded);
  `composed` = no in-harness cell exists, see ledger item 2.
- `capsule` / `cell_id` / `branch` — where the raw rows live:
  `sweeps/results/runs/<capsule>/metrics.csv` on `main`, except the FAST rows
  whose capsules are committed on the **`fast-split`** branch only.
- `flag` — non-empty on every cell that is anything other than a clean match.

All values are `total_ms` = plan_comm + plan + e2e window (SCHEMA protocol
rule 5, planning-inclusive), `isolated` mode (per-iteration sync+barrier,
inference semantics) — except FAST, whose harness is a separate process and
records under mode `e2e` (comm and GEMM un-overlapped by construction; its
per-iteration numbers are the quotable ones per handoff 24 on `fast-split`).

## Figure row -> arm mapping

The CSV row labels are figure-facing names, NOT arm keys. The mapping
(`row_id` -> `arm_variant`):

| row_id | figure label | arm (cell_id prefix) | what it is |
|---|---|---|---|
| nvshmem_gemm | NVSHMEM a2av + GEMM | `l01_nvshmem` | primitive blocking-put ring a2av + un-overlapped grouped GEMM (the 8/31 reference-arm flip: this now runs the legacy "Torch+GEMM" slot) |
| fast_gemm | FAST + GEMM | `l01_fast` | FAST BvN-scheduled alltoallv (3rdparty/FAST) + un-overlapped grouped GEMM; v4 wire binary for b1–b32 rows, see ledger for b64 |
| comet | Comet | `l01_allgather_dense` | Comet/Flux fused op, dense allgather dispatch (both layers) |
| moonep | MoonEP | `moonep_l01_nvshmem_getmem` | MoonEP faithful defaults (nvshmem dispatch + getmem weight prefetch) |
| eplb | EPLB / EPLB (direct a2av) | `eplb_l01` | EPLB pool-oracle static placement over direct a2av dispatch (same arm at 4n and 16n; the 16n label just spells out the wire) |
| epic | EPIC | `epic_l01_hc_m1` | EPIC hier-compress M1 |
| ours1_tokencomm | 1Ours(Token Comm) | `l01_slipstream` | Slipstream token-comm optimizations only (msplit+fp+wp+bucket, no placement) |
| ours2_nooverlap | 2Ours(... hier a2av (no overlap)) [4n] / 2Ours(... no overlap) [16n] | `llc_l01_s1_pv2` | PLACE-lambda+LocCap s1 placement + routing on the hier a2av, no compute/comm overlap (pv2 planner). One arm, two label spellings — both run hier a2av |
| ours2_direct | 2Ours (... direct a2av) [16n only] | `ours_l01_s1_pv2_r2_dwire` (b1–b16), `..._dwire_dps` (b64) | s1 placement + routing over the DIRECT wire (transport ablation, handoff 29/30) |
| ours12 | 1+2 | `ours_l01_s1_pv2_r2` | full system: slipstream overlap + s1 placement/routing, r2 slack parity |
| ours12_dispatch | 1+2 + expert dispatch (re-solve every iteration no expert movement) | `ours_l01_s2_swap_force_p2p_r2` | s2 methodology arm: placement re-solved every iteration, forced overlapped P2P expert swaps (handoff 25/27) |

## Discrepancy ledger (recorded grid vs. capsule authority)

1. **4n K2 FAST b16: recorded 16.13 is a transcription error** (duplicate of
   the b4 value). Authority (handoff 24 final FAST table + capsule
   `20260829-133740_perlmutter_ff3902f4`) = **35.680**. `figure_src.csv`
   carries 35.680.
2. **16n b64 FAST: the recorded grid swapped the two models.** These two cells
   never ran in-harness (Qwen = 40 GB memory-wall pre-skip; K2 = CXI wedge,
   FAST-traffic x NCCL interaction). They are COMPOSED numbers — native v6
   wire + plan + GEMM measured in separate processes (composition closed
   within 8% on qwen 8n b64): **K2 = 330.0, Qwen = 348.0**
   (`fast-split:docs/handoff/24_fast_split_regen.md`, final table). The
   recorded grid had 330 under Qwen and 348 under K2 — swapped.
   `figure_src.csv` carries the correct assignment. Figure should footnote
   the composition.
3. **16n direct-a2av b64 (recorded "NA — find if we ever did this"): yes, it
   ran green on 8/31** with the pair-cushion arm `ours_l01_s1_pv2_r2_dwire_dps`
   (the fix for the plain-dwire b32 wedge; dps only ran b32/b64):
   K2 = 199.759 (`20260831-011130_perlmutter_45dacd5f`),
   Qwen = 196.718 (`20260831-011223_perlmutter_93360895`). Filled in
   `figure_src.csv` with flag `filled_was_NA;dps_arm`. Note the b1–b16 cells
   of this row are the plain `dwire` arm — same transport, dps adds the
   receive-side pair cushion.

## Caveats to keep in mind when styling/quoting

- **Cross-capsule, cross-binary mixing.** The picked values intentionally span
  many capsules and several builds (SCHEMA protocol rule 4 says compare inside
  one capsule; this figure is a curated exception). Per-cell `capsule` +
  `git_sha` are recorded so any cell can be audited or re-run.
- **`llc_l01_s1_pv2` self-oracle caveat**: the 2Ours-no-overlap capsules here
  (8/27–8/29) predate the 8/31 oracle-basis fix (`98eb2c8`) — they ran
  SELF-oracle routing rather than the shared oracle basis. Annotate or regen
  if oracle-basis parity across arms matters for the final figure.
- **stat mixes median and mean across rows** (whatever was recorded was
  matched — the nvshmem rows and a handful of single cells were recorded off
  the runner's console mean; everything else is the campaign median). The
  `stat` column is per-cell truth. If uniform aggregation is wanted for the
  figure, regenerate every cell as `iter_max_median` from the same capsules
  (sub-0.5 ms shifts).
- **FAST provenance**: capsules only on the `fast-split` branch; 16n rows ran
  the 16-server-limits binary (`MAX_SERVER_NUM` fix), b1–b32 rows the v4 wire.
  FAST numbers are mode `e2e` from its standalone harness — un-overlapped by
  construction, so the comparison against fused/isolated arms is the intended
  apples-to-apples for this figure but the mode label differs.
- 4n K2 FAST b1 comes from the gate capsule `20260829-114622_perlmutter_e7ef30e3`
  (the full-ladder capsule's b1 median is 10.76 — 0.01 lower); this mirrors
  handoff 24's own table.

## How this was built

`~scratchpad/build_agg.py` recomputed iter-max mean+median for every
`total_ms`/`e2e_ms` in all 966 capsules on `main` plus the 44 `fast-split`
capsules; `resolve.py` matched each recorded value at its recorded precision
(tolerance = half an ulp of the last digit) constrained to the mapped arm,
topology, model, and budget. Scripts are session-scratch; the attribution
lives in this CSV.
