# Weak scaling, A100, with the main_perf_v6 1 MiB values (2026-10-07)

Addition to the canonical `../figure_src.csv` (never an overwrite). One change: at 1 MiB and 4 / 8 / 16 nodes, the
A2AV+GEMM and Ours values are the K2 1 MiB cells of `figs/main_perf_v6/figure_src.csv`, so the speedups match
Figure 9 (A2AV+GEMM = its `nvshmem_gemm` rows; Ours = its best of `ours12` / `ours12_dispatch` / `ours2_direct`,
gpu-plan on, Zepp release candidate `zepp_gpuplan` d0ab91f). Every other cell (2 and 32 nodes at 1 MiB, the whole
64 MiB row) is the canonical dataset, unchanged.

Render: `module load python; python make_figure.py --baseline nvshmem --stacked --src-dir gpuplan_v6 --out-dir gpuplan_v6`
-> `weak_scaling_nvshmem_stacked.{pdf,png}`.
