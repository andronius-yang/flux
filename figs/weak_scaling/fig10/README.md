# Paper Figure 10 (weak scaling, (a) A100 over (b) H100) with the updated A100 1 MB row (2026-10-07)

`weak_scaling_fig10.{pdf,png}` = Figure 10 of the 2026-09-18 paper build, same format, with the A100 1 MB panel
from `../figure_src.csv` (main_perf_v6 values at 4 / 8 / 16 nodes, the 2-node release-build run; speedups
1.53 / 2.10 / 2.96 / 5.77 / 9.49x). The A100 64 MB panel and both H100 panels are unchanged.

- Layout: `make_figure_fig10.py` places four panels at the exact figure size and axes rectangles of the PDF's
  Figure 10 (measured from its vector geometry) and draws each with `../make_figure.py`'s `draw_panel` (same
  styling, ylims, ticks and labels). Exactness check: `--original` renders the 09-18 data
  (`weak_scaling_fig10_original.{pdf,png}`); every text label and every vector element matches the paper's figure
  within 0.2 pt (lines, bars and axes within 0.03 pt).
- H100 data (`h100/figure_src.csv`): the H100 runs (CSCS ALPS, handoff 35) were never committed to this
  repository, so the values are recovered from the paper PDF (`LibraX_NSDI_27.pdf`, Figure 10(b)): latencies from
  the vertices of the plotted A2AV+GEMM / Ours lines and the axis geometry. Validated on the A100 panels of the
  same figure against the known data: every point within 0.0007 %. The recovered speedups equal the printed
  labels and the bar heights.
- `a100_pre_gpuplan/figure_src.csv` = the A100 data as plotted on 09-18 (for `--original` only).

Render: `module load python; python make_figure_fig10.py` (add `--original` for the check).
