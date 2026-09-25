# main_perf_v4 — main-perf figure on the consolidated implementation (2026-09-25)

**Status: new version by user ruling (2026-09-25).** v4 = v3 (REV 3.1 aesthetics, eight bars
incl. COMET+EPLB, both renders) with ONE change: **the Ours rows are replaced by the final
candidate campaign on the paper-consistent implementation**, and the Ours bar is the min over
exactly three candidates. Baseline rows are v3's, unchanged (their capsules and dates are in
`figure_src.csv`; the `_samebinary` render feeds COMET from its 2026-09-17 anchor as in v3).

## The Ours rows (handoff 42 §12; capsules 20260925-200150/-200638 4n, -203721/-204147 8n, -204837/-205308 16n)

Consolidated implementation: pv3c router at the figure's C (1/4 at ≤ 8n, 1/2 at 16n), minimal-move
relay partition (paper §4.2 eq. 2/3), per-round relay staging + P2P pulls, band swap trigger
(paper §4.3), `CUDA_DEVICE_MAX_CONNECTIONS=24` (SCHEMA rule 14 amendment; the direct-wire arm keeps
its canonical 8). Binary = the 2026-09-25 03:28 build of pv3 (the reconciled tree; the knob code of
the same morning reverted). Statistic `iter_max_median`, 10 timed iterations, correctness off
(gated the same morning on the new pin, 10/10 with per-iteration random payload).

| row_id | candidate | arm |
|---|---|---|
| `ours2_direct` | placement + routing, direct a2av wire (no comm/comp overlap) | `ours_l01_s1_pv2_r2_dwire_pv3c_eps{025,05}` |
| `ours12` | everything except expert swap (fused overlap) | `ours_l01_s1_pv2_r2_pv3c_eps{025,05}` |
| `ours12_dispatch` | everything on: fused + 3D-scheduled band-triggered expert swap (the case-study arm without the drift replay) | `ablation_l01_s2_swapall_nr_3d_dual3_str4_bal_p2p_r2_pv3c_eps{025,05}` |

Plotted Ours = min of the three per (nodes, model, budget). Winners: fused in 13 of 16 decided
cells; 3D swap at 8n K2 b16, 8n Qwen b1, 16n K2 b16 (1–6 %, its decision tax is the only
difference); direct wire at 16n b1 for both models (3–8 %). The two 16n direct-wire b16 cells
failed at setup (the direct arm's symmetric-heap sizing) and are absent by user ruling; they
cannot move any min (direct loses every b4/b16 by ≥ 45 %).

Budgets: b1/b4/b16 are the plotted groups. b64 `ours12` rows come from the same-day conn=24
fused capsules (mean over reps, flag `b64_fused_conn24_20260925…`, not plotted); b2 has no
candidate-campaign row (not plotted). The legacy forced-swap arm (v3's `ours12_dispatch`) is
retired from main perf; `ours1_tokencomm` and `ours2_nooverlap` are not candidates in v4.

## Files

- `build_figure_src_v4.py` — regenerates `figure_src.csv` from v3 + the capsules (run from anywhere).
- `make_figure.py` — v3's generator with `OURS_CANDIDATES` = the three rows and v4 output names.
- `main_perf_v4.{pdf,png}`, `main_perf_v4_samebinary.{pdf,png}` — the two renders.
- `SPEC.md` — v3's REV 3.1 spec (aesthetics unchanged).
