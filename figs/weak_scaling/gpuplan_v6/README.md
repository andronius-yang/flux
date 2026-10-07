# Weak scaling, A100, with the main_perf_v6 1 MiB values (2026-10-07)

Addition to the canonical `../figure_src.csv` (never an overwrite). One change: at 1 MiB and 4 / 8 / 16 nodes, the
A2AV+GEMM and Ours values are the K2 1 MiB cells of `figs/main_perf_v6/figure_src.csv`, so the speedups match
Figure 9 (A2AV+GEMM = its `nvshmem_gemm` rows; Ours = its best of `ours12` / `ours12_dispatch` / `ours2_direct`,
gpu-plan on, Zepp release candidate `zepp_gpuplan` d0ab91f). At 1 MiB and 2 nodes, Ours (3.260 ms, overlap, swap
off; direct 7.720, swap on 3.564) was measured on 2026-10-07 with the same build (its tree is Zepp `a622091`) and the
same protocol (5 warm-up + 10 timed iterations, router C 0.25, job 59494219; `k2_b1_2n.csv.gz`); A2AV+GEMM at
2 nodes is the canonical value. Every other cell (32 nodes at 1 MiB, the whole 64 MiB row) is the canonical
dataset, unchanged.

Render: `module load python; python make_figure.py --baseline nvshmem --stacked --src-dir gpuplan_v6 --out-dir gpuplan_v6`
-> `weak_scaling_nvshmem_stacked.{pdf,png}`.
