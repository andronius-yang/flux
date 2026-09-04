# Weak-scaling figure — aesthetic & architecture spec

Status: REV 3.1 (2026-09-04; postdoc: speedup labels ALWAYS above the bar,
bars in the evaluation-wide speedup color — applied to EVERY render, so the
REV 1.2/2.x outputs are regenerated, no longer byte-stable). Data authority: `figure_src.csv` (+
`figure_src.md`). Generator: `make_figure.py` (matplotlib; one `CONFIG`
block, every **[knob]** below lives there). Two versions from one script:
**verA** (latency) and **verB** (throughput).

## 1. Purpose and rulings (user, 2026-09-02)

- Single-column NSDI figure: width = `\columnwidth` = **3.33 in**;
  figure + caption **<= 1/5 of the 9 in text height** (1.8 in) -> the
  figure PDF is **1.45 in** tall, leaving ~0.35 in for a caption.
- X axis = node count **2, 4, 8, 16, 32** (equal spacing, i.e. log2).
- Left y axis: **verA** total latency (ms); **verB** synthesized MoE-layer
  throughput (Mtok/s) = pre-topk tokens/rank × ranks ÷ latency.
- Two lines, COMET vs Ours at the 64 MiB per-rank budget, different colors
  AND different marker shapes.
- Right y axis hosts an overlaid **speedup bar chart**: one bar per node
  count, Ours vs COMET, value printed on top; bars end at 16n because COMET
  does not run at 32n (§4). Same bars in verA and verB (tokens cancel).
- Legend on top; the same parameterized-script discipline as
  `figs/main_perf`.
- Design note: this is deliberately a dual-axis chart (the one form the
  viz method flags as easy to misread). Mitigations: bars are recessive
  gray behind the lines, the right axis label/ticks are gray to match the
  bars, and the left axis keeps the primary ink.

## 2. Marks

- **Lines** **[knob: SERIES]**: COMET olive `#999933`, square markers;
  Ours steel blue `#4878b0`, circle markers — the same hues these systems
  carry in the main figure, so identity is consistent across figures.
  Line 1.1 pt, marker 3.6 pt, **no edge stroke** (REV 1.1: the white
  marker edge painted over neighbouring elements).
- **Speedup bars** **[knob: BARS]** (REV 3.1, postdoc): the speedup family
  is the main figure's Ours-speedup red — fill `#d7616c` (muted red-coral,
  65% of `#c1121f` on white), edge `#c1121f` 0.5 pt, width 0.55 of the
  slot, drawn *behind* lines (zorder). This is the ONE speedup color for
  every evaluation figure (main figure's speedup annotations use the same
  `#c1121f`). Right y-limit = ceil_nice(max speedup × 1.5) **[knob:
  BAR_HEADROOM]**. Value label `"{:.2f}×"` **always directly above its bar**
  (postdoc ruling — even where it crosses a line), 5.8 pt bold `#c1121f`
  with a 1.3 pt white halo so it stays legible over line segments; drawn on
  the upper (line) axes so it sits above both lines and markers. The
  earlier grey fill / auto-base placement is retired.
- **COMET missing at 32n** **[knob: FAIL_NOTE]** (REV 1.2, postdoc): an **✗**
  in the COMET color sits directly **on the x axis** at the 32n slot, with
  the short note **OOM** just above it (5.5 pt, COMET color). No dashed
  tail, nothing floating in the plot area. The caption carries the reason
  (§4).

## 3. Layout & chrome

- Figure 3.33 × 1.45 in **[knob: FIG_W, FIG_H]**; margins left 0.135 /
  right 0.86 / top 0.80 / bottom 0.21 **[knob: MARGINS]**.
- Left ylim = ceil_nice(1.12 × max plotted) **[knob: HEADROOM]**;
  3–4 ticks; light horizontal gridlines at the left axis's tick positions,
  drawn on the *lower* (bar) axes so they render beneath the bars and their
  value labels (REV 1.1 — on the upper axes they crossed the labels).
- X ticks `2 4 8 16 32`, axis title "Nodes" **[knob: X_LABEL]**.
- Left axis label **[knob: Y_LABELS]**: verA "Latency (ms)", verB
  "Throughput (Mtok/s)". Right axis label "Speedup vs COMET" in gray.
- Legend: one row of three (COMET, Ours, Speedup vs COMET), 6.5 pt, no
  frame, flush to the top edge **[knob: LEGEND]**.
- Typography **[knob: FONT_FAMILY, FONT_SIZES]**: sans (Helvetica/Arial/
  DejaVu fallback), legend 6.5, axis labels 6.5, ticks 6, bar labels 5.8,
  fail note 5.5; `pdf.fonttype 42`.

## 4. Data contract

- Reads `figure_src.csv`; verA plots `total_ms`, verB `throughput_mtok_s`;
  bars use `speedup_vs_comet` where present. Missing COMET cells are
  rendered as the fail note, never as zero. Asserts every expected
  (nodes, system) row exists.
- COMET 32n at 64 MiB: **OOM on A100-40GB** (verified by session 78f1b4cd,
  2026-09-03, after the upstream 64-rank sort-table cap was raised — COMET
  runs at 32n for 1–16 MiB). The dense gathered input is ~8.6 GB/rank at
  128 ranks; symmetric-heap demand (>17 G) + torch (16.1 GiB) + overhead
  (6.1 GiB) exceeds 39.5 GB. Full record in `figure_src.md`. Caption
  wording suggestion: *"COMET's dense all-gather does not fit a 40 GB A100
  at 32 nodes × 64 MiB (OOM)."*
- Caveats: 4/8/16n cells are the handoff-30 same-binary set (differ from
  the main figure's capsules by up to ~5% for the same nominal cells —
  user ruling: keep); 2n/32n are the cap-fix binary; NVSHMEM ring omitted
  (user ruling: bars vs COMET only).

## 4b. Baseline switch (REV 2.0, user 2026-09-03)

- `python3 make_figure.py --baseline nvshmem` renders the same two versions
  against the **NVSHMEM+GEMM ring** (`l01_nvshmem`, the main figure's
  speedup reference) instead of COMET: series `nvshmem` = sand `#ddaa33`
  (the main figure's nvshmem_gemm hue), triangle markers **[knob: SERIES]**;
  bars = `speedup_vs_nvshmem`, right label + legend entry "Speedup vs NVSHMEM" (the full
  "…+GEMM" form overflowed the column width), legend colspacing 0.7;
  outputs `weak_scaling_nvshmem_ver{A,B}.{pdf,png}` **[knob: BASELINES]**.
  The ring runs at every node count incl. 32n, so all five bars are drawn
  and no fail note appears. 2n ring point: capsule `20260904-004243` (ring +
  Ours in one capsule; the Ours row keeps the COMET-capsule value 36.092 ms,
  the ring capsule's 36.097 ms agrees within 0.01%).
- The default (no flag) still produces the COMET renders byte-identically.
- **Budget switch (REV 2.1)** **[knob: BUDGET]**: `--budget 1` plots the
  1 MiB rows of `figure_src.csv` (72 pre-topk tokens/rank; verB throughput
  scales accordingly); outputs gain a `_b1` infix
  (`weak_scaling_nvshmem_b1_ver{A,B}.{pdf,png}`). 64 MiB stays the
  unsuffixed canon. The 1 MiB story is the fixed-cost regime: the ring
  speedup is 1.18× at 2n and grows with node count (1.33 / 1.62 / 1.92 /
  2.08×) — the opposite shape from 64 MiB, where it is ~2.4–3.1× flat.

## 4c. ver4: stacked two-budget figure (REV 3.0, user 2026-09-04)

- `python3 make_figure.py --baseline nvshmem --stacked` -> one single-column
  figure, **two panels stacked** (top 1 MiB, bottom 64 MiB), latency lines
  + speedup bars in each, **one shared legend** on top and **one shared
  right-axis label** ("Speedup vs NVSHMEM", centered on the stack — per-panel
  labels collided); right-axis SCALES stay per panel (a shared scale
  flattened the 64 MiB bars once the 1 MiB panel reached ~9x — each panel
  prints its own ticks). Bar value labels use `label_pos="above_all"`: above
  the bar top AND above every marker at that x, never at the base (the 1 MiB
  panel's small bars and lines both hug the baseline). Per-panel
  labels collided at this height) **[knob: STACKED]**. Height 1.8 in total
  (= 1/5 of the 9 in text height, "compact for now"; caption not budgeted),
  `hspace` 0.22, x tick labels + "Nodes" only on the bottom panel, a small
  budget tag ("1 MiB" / "64 MiB") top-left inside each panel.
  Outputs `weak_scaling_nvshmem_stacked.{pdf,png}`.
- **Ours = min over arms** **[knob: OURS_ARMS]** (main-figure convention):
  at each node count the "Ours" line takes the fastest of `ours` (s1) and
  `dwire` (placement + routing over a direct staged a2av) rows present in
  `figure_src.csv`; the winning arm is recorded on the merged row. At
  64 MiB s1 wins everywhere (dwire rows absent/slower) so REV 2.x renders
  are byte-identical; at 1 MiB dwire wins from 16n (7.20 vs 12.77 ms at
  16n; 9.25 vs 39.19 ms at 32n — the 32n dwire cell's timed loop is complete
  but its process hung at teardown, see `figure_src.md`). 1 MiB 16n rows are the same-binary one-capsule set
  `20260904-120809` (dwire + s1 + ring).

## 5. Output & checks

- `python3 make_figure.py` -> `weak_scaling_verA.{pdf,png}` and
  `weak_scaling_verB.{pdf,png}` (300 dpi previews), deterministic PDFs.
- Render checklist: no bar label collides with a marker or the fail note
  (generator audit with rendered extents); markers legible over bars;
  grayscale export still separates COMET (square) from Ours (circle);
  fonts embedded; PDF height 1.45 in.
