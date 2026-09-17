# main_perf_v3 — main-perf figure + the COMET+EPLB bar (opened 2026-09-17)

**Provisional ADDITION lane, same standing as main_perf_v2**: nothing here
overwrites `figs/main_perf` (still the paper authority) until the user rules.
v3 = REV 3.1 of `figs/main_perf` with ONE change: an 8th bar, **COMET+EPLB**
(`row_id comet_eplb`), placed between COMET and Ours. Sizing (7.0 x 2.47 in,
2100x740 px at 300 dpi), fonts, the seven original colors and hatches, the
ceiling/truncation and speedup rules are unchanged; bars are 7/8 as wide
because eight now share the same group width.

## The new bar

- Arm `l01_allgather_dense_eplb` (branch pv3, commit 859af86; handoff 41):
  the COMET dense fused ops executing the `eplb_l01` arm's static
  pool-oracle placement (same vendored EPLB, same G/W+2 slot budget, same
  load sidecar) through the physical-slot space; one fused router kernel
  per iteration. "Overlap + placement", the reviewer-requested baseline.
- Color `#7a3b7a` (deep plum), hatch `++`. Re-validated with the six-checks
  math (Python twin of the dataviz validator; the 7-color baseline
  reproduces SPEC §5's recorded 16.1 / 18.6 exactly): adjacent CVD worst
  10.7 (deutan, vs Ours), normal-vision worst 17.2, lightness band and
  chroma PASS, contrast relief unchanged (sand/teal/olive, edge stroke +
  hatch). Deep purple candidates closer to Ours' steel blue FAIL the CVD
  gate (#7b5ea7: dE 2.0) — never lighten this color.
- Values: `figure_src.csv` rows `comet_eplb` = the 2026-09-17 main-perf
  capsules (071106/071304 4n, 071215/071417 8n, 073316/073515 16n),
  `iter_max_median`, same statistic as every other campaign row.

## Two renders — read this before quoting

- `main_perf_v3.*` — the official rows + the new bar. **Caveat:** the
  official COMET row is the 2026-08-24 binary; the same arm measured in the
  2026-09-17 capsules (same binary as COMET+EPLB) is 4-47% FASTER at 1-4 MiB
  (16n b1: 15.3 -> 8.1 ms). In this render part of the COMET -> COMET+EPLB
  gap is binary/day drift, not placement (SCHEMA rule 4).
- `main_perf_v3_samebinary.*` — identical, except the COMET bar is fed from
  `row_id comet_v3anchor` (the `l01_allgather_dense` cells of the same six
  capsules). This is the honest COMET-vs-COMET+EPLB comparison: tie at K2
  1 MiB, -6..-12% at 4/16 MiB, -1..-16% on Qwen. Every other bar is
  unchanged between the two renders. Which one (or a full COMET re-measure)
  goes to the paper is the user's ruling.

Generator: `make_figure.py` (CONFIG knobs `ROW_SOURCE`, `VARIANTS`); data
`figure_src.csv` (v1 rows verbatim + `comet_eplb` + `comet_v3anchor`);
provenance addendum at the end of `figure_src.md`.
