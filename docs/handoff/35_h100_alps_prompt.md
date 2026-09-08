# Paste-in prompt for the ALPS/H100 session (companion to handoff 35)

Operator: fill every `<...>` before pasting (or answer them when the session asks — it
will not build anything until it has them). Paste the block below as the first message
of a fresh Claude Code session started in the root of the cloned repository.

---

You are starting a measurement campaign on the CSCS ALPS cluster (GH200 nodes: Grace aarch64 CPUs + NVIDIA H100 GPUs, Slingshot-11/CXI network). This repository was developed and measured on NERSC Perlmutter (A100); you have no memory files from that work — the handoff documents in `docs/handoff/` are your memory.

Your single task: reproduce the A2AV+GEMM weak-scaling figure on H100 at 2, 4, 8, 16 and 32 nodes, exactly as specified in `docs/handoff/35_h100_alps_weak_scaling.md`. Nothing more, nothing less: no extra arms, budgets, shapes, modes, tuning or optimization.

Before doing anything else:
1. Read `docs/handoff/35_h100_alps_weak_scaling.md` completely, then `CLAUDE.md`, `sweeps/SCHEMA.md` (section "Protocol rules"), `figs/weak_scaling/SPEC.md` (sections 1, 4b, 4c) and `.claude/skills/sweep/SKILL.md`.
2. Write back, in your own words, a short plan: the deliverable, the 30 measured cells, the gate ladder, the one-binary rule, where every run-time artifact must live (`figs/weak_scaling/h100/` inside this clone — never my home or scratch outside the clone), and the list of things you still need from me. Wait for my confirmation before running the discovery step or building.

Facts about this site (fill in / correct):
- vCluster / login host: <e.g. daint.alps.cscs.ch>
- Slurm account: <account>
- Partition / QOS for GPU nodes, max walltime, and whether interactive `salloc` at 32 nodes is acceptable or `sbatch` is expected: <...>
- Reservation (if any): <none>
- This clone's absolute path (visible from compute nodes): <path>
- Software provisioning: <uenv image + view | modules | other>; NVSHMEM with the libfabric transport is <available at <path> | not available (build it per handoff 35 §4.2)>
- The shipped data (`traces_k2_lcb_execution.tar`, `matrices/`, `SHA256SUMS.txt`) is at: <path>
- Push rights to the git remote: <yes | no — hand back a git bundle>
- Anything I know is unusual about this cluster: <...>

Working rules for this session (they are also in the handoff):
- Keep every log, build output, matrix, trace, venv and raw sweep artifact under `figs/weak_scaling/h100/` in this clone; commit only capsules, the figure dataset, renders and `SESSION_LOG.md`.
- Append to `figs/weak_scaling/h100/SESSION_LOG.md` at every event (discovery results, build launch/finish with the binary sha, each gate verdict, each capsule, each incident). Tell me at every such event too, not only at the end.
- Gate-first: stop at the first red gate and root-cause it; never widen timeouts to "wait longer"; never rebuild between rungs (one binary for the whole figure).
- Release every allocation as soon as its capsule is written; never let GPU nodes idle.
- Never run filesystem-wide searches; bound every search to this clone or a path I gave you.
- Do not modify arm definitions, sizing formulas or kernels; the point is to measure the same code on H100.

End state: the seven capsules committed, `figs/weak_scaling/h100/figure_src.csv` + `figure_src.md`, the renders `weak_scaling_nvshmem_stacked.{pdf,png}` (+ verA/verB), the session log, all pushed on branch `h100-alps-weak-scaling`, and a final summary listing the binary sha, GPU/memory, NVSHMEM source, which build path was used, every deviation from the handoff, and the ten (nodes, budget) speedups.
