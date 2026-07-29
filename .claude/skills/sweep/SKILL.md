---
name: sweep
description: Run a reproducible perf sweep of the layer0 a2av dispatch variants and persist the results capsule. Use whenever the user asks for a sweep, latency comparison, or perf numbers across budgets/skews/variants.
---

# Running a sweep

The sweep system lives in `sweeps/` — read `sweeps/SCHEMA.md` before
interpreting results. One invocation = one immutable capsule under
`sweeps/results/runs/<run_id>/` (commit it) + raw logs at the platform data
root (leave them).

## Constants (do NOT vary these without an explicit user ask + spec file)

- `H=4096`, `chunk_bytes=8192` (one bf16 token), `dtype=bfloat16`,
  `ffn_hidden=4096`, EP = world size.
- **Budget is STRICTLY the pre-topk send budget** per source rank:
  `tokens_per_rank = budget_mib*2^20/8192`, matrix row sums =
  `budget_mib * 2^20 * topk`. Never quote budgets in any other framing.
- Perf protocol: the runner exports `FLUX_TEST_DETERMINISTIC=0` itself and the
  capsule's `deterministic` column audits it — do not hand-run perf outside
  the runner without that env, and never trust a perf number whose
  deterministic column is 1.
- Knob scaling (`FLUX_A2AV_MAX_*_NTOKENS`, `NVSHMEM_SYMMETRIC_SIZE`) is
  formula-driven (SCHEMA.md §knobs) — do not hand-tune per cell.

## Variables (the axes the user picks)

- `--variants` — canonical names from `sweeps/variants.py`:
  `hier` (baseline), `hier_compress` (dedup + balanced relay),
  `hier_compress_identity`, `hier_compress_union` (gateway bcast),
  `hier_compress_pack` (bcast + pack overlap), plus `allgather`/`a2av`/
  `a2av_ring`.
- `--families` — traffic distribution: `uniform`, `hotcol[:frac=..]`,
  `nodeskew[:frac=..]`, `remotefrac` (per-rank inter-node skew, the relay's
  target case). Distribution identity = matrix_id (deterministic; generated
  on demand).
- `--budgets-mib` — pre-topk budgets, typically `2,4,8,16,32,64`.
- `--topk`, `--G` — routing shape (G % world_size == 0; the generator
  pre-checks routing feasibility and tells you the minimum G).
- `--modes` — `e2e` (clean latency), `phases` (FLUX_A2AV_TIMING breakdown —
  perturbed, never compare its e2e against clean cells), `torchprof`, `nsys`.
- `--iters/--warmup-iters` (default 10/5), `--skip-correctness` (large
  budgets; correctness columns go empty), `--matrix-instance` (new random
  instance of a family).

## Procedure

1. Environment: `source ./env_aws.sh` (AWS) or `source ./module.sh`
   (Perlmutter). Check an allocation exists: `squeue -u $USER` — if not:
   `salloc --partition=a100 --nodes=2 --exclusive --no-shell` (AWS) /
   `salloc --qos interactive -C gpu --account m4243_g -N 2 --no-shell`.
2. Always `--dry-run` first; sanity-check the cell list and env lines.
3. Run. Watch for `skipped_capability` warnings (build lacks a variant's env
   knob = stale build — rebuild before believing anything).
4. After: check `cells.csv` statuses; `ok` + `correct_allclose=1` (or empty
   when skipped) is the bar. Commit with the printed `git add ... && git
   commit ...` command.
5. Report results from the capsule (max-rank and mean over ranks of e2e_ms
   per cell), never from eyeballing stdout.

## Gotchas

- EFA transients inflate whole cells ~3x occasionally — rerun outliers
  (`sweep.py rerun <capsule>`) before drawing conclusions.
- Recv-overflow FLUX_CHECK is per-rank data-dependent: some ranks throw,
  others spin at 100% GPU forever. The cell timeout (900 s) reaps these;
  a `timeout` status usually means undersized MAX_RECV knobs (raise budget
  anchors only via SCHEMA.md, not ad hoc).
- `hier_compress*` multi-node needs `sm_margin >= 1` (runner auto-bumps,
  except union-bcast which legitimately allows 0).
- nsys may capture no kernels on this stack — the runner flags it; use
  `torchprof` mode as the reliable timeline.
- Single-node runs can't use `remotefrac`/`nodeskew` (no remote ranks);
  use `uniform`/`hotcol`.
