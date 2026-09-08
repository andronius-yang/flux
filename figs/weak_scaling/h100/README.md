# figs/weak_scaling/h100 — the H100 (CSCS ALPS) twin of the weak-scaling figure

Authority: `docs/handoff/35_h100_alps_weak_scaling.md` (goal, precautions,
setup, run ladder, figure). Everything the H100 campaign produces or needs
at run time lives in THIS folder so the collaborator's cluster workspace is
never touched:

| path | what | committed? |
|---|---|---|
| `data/` | **`h100-weak-scaling` branch only**: the routing pool (xz tar in sub-50 MiB parts) + the 10 matrix sets + `SHA256SUMS.txt`; unpack per `data/README.md` / handoff 35 §4.1 | yes, on that branch (never on main) |
| `traces/` | the Kimi-K2 `livecodebench/execution` routing pool, unpacked from `data/` | no (gitignored) |
| `matrices/` | traffic matrices + routing/oracle sidecars, unpacked from `data/` (the runner regenerates identical ids from `traces/` if absent) | no |
| `raw/` | sweep data root: per-cell rank JSONLs, torchrun logs (`sweeps/platforms/alps.yaml data_root`) | no |
| `logs/` | build logs, salloc/sbatch stdout, session scratch | no |
| `venv/` | the python environment, if a venv is used | no |
| `SESSION_LOG.md` | append-only ledger of everything the session did (handoff 35 §7) | YES |
| `build_figure_src.py` | capsules -> `figure_src.csv` (same statistic + columns as the Perlmutter figure) | YES |
| `figure_src.csv`, `figure_src.md` | the H100 dataset + its notes (written by the session) | YES |
| `weak_scaling_nvshmem_stacked.{pdf,png}` (+ verA/verB) | the H100 renders from `../make_figure.py --src-dir figs/weak_scaling/h100` | YES |

Capsules go where the runner puts them (`sweeps/results/runs/<run_id>_alps_<digest>/`)
and are committed like every other capsule.
