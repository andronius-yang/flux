# main_perf_v6 — Ours with gpu-plan on, COMET+EPLB plotted (2026-10-07)

**Status: ADDITION to v5 (never an overwrite).** Two changes from v5:

1. **Ours rows (b1 / b4 / b16) re-measured with `gpu-plan` ON**: the layer planned and issued on the GPU
   (device router, swap decision, metadata, kernel-issued dispatch and combine wire, per-layer CUDA graphs, device
   dual3 swap lane). Release candidate: clone `$PSCRATCH/workspace/andrewy/zepp_gpuplan`, branch `gpuplan`
   `d0ab91f` (= Zepp `sglang-dev` + the gpu-plan option), `libzepp_cuda.so` sha256 `beb332a88464f324...`.
   Protocol unchanged: `scripts/reproduce.sh --gpu-plan 1` (isolated, 5 warm-up + 10 timed iterations,
   per-iteration MAX over ranks, MEDIAN over iterations, the bit-identical trace batches of the baseline capsules).
   Allocations (2026-10-06/07): 4n interactive 59473401, 8n debug 59472597, 16n regular 59472596.
2. **COMET+EPLB (`comet_eplb`, the 2026-09-17 capsules, unchanged since v3) is plotted**, as the bar next to Ours
   (`INCLUDE_COMET_EPLB=True`; plum `#7a3b7a`, hatch `++`, the v3-validated style). Its in-bar ratio labels are
   white (`dark_text_L` 0.40 -> 0.48; only the plum fill is below 0.48).

Style edit (2026-10-07, user, re-rendered in place): the COMET bar gets its own hatch, vertical `||||` (was none),
the same in the ablation figure (`figs/ablation_cycling`).

Baseline rows are v5's, byte-identical. v5's Ours rows at b2 / b64 are carried (`carried_from_v5`). The plotted Ours
bar is the min over `ours12` / `ours12_dispatch` / `ours2_direct` as in v4/v5 (direct is host-planned: gpu-plan
applies to the overlap strategy only).

Validation of the gpu-plan numbers (2026-10-07): timing window, isolation and statistic audited equivalent to the
research harness (`test_moe_l0l1_traffic.py` `perf_combined`); strict correctness check 12/12 PASS on 4 nodes
(relative error ~4e-3 vs the torch reference, rotating routing batches, negative controls, 0 capacity re-runs,
expert moves in flight at b1/b4) — `gpuplan_measured/strict_check*.{py,log,txt}`.

Files: `build_figure_src_v6.py` (v5 figure_src + `gpuplan_measured/on`), `figure_src.csv`, `make_figure.py`
(v5's + the two changes), `main_perf_v6.{pdf,png}` (official COMET row) and `main_perf_v6_samebinary.{pdf,png}`
(COMET on its same-capsule anchor), `gpuplan_measured/{on,off}/main_perf_{4,8,16}n.csv.gz` + `compare_*.txt`
(`off` = the same-day gpu-plan OFF grids, not plotted). Render: `module load python; python make_figure.py`.
