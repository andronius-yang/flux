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
  `hier_compress_lb_union` (balanced wire + gateway window-bcast; like the
  union modes it forwards with pure CE puts, so sm_margin 0 is legitimate),
  `hier_compress_pack` (bcast + pack overlap), plus `allgather`/`a2av`/
  `a2av_ring`, and `fast` (the FAST load-balancing alltoallv + un-overlapped
  GemmGroupedV2 baseline). `fast` constraints: needs `3rdparty/FAST/nvidia/
  libflash.so` (build once per checkout: `srun ... ./scripts/build_fast.sh`
  on a compute node, after sourcing the platform env), needs >= 2 nodes,
  expands in `e2e` mode only (its phase breakdown — pack/schedule/fill/wire/
  unpack/gemm — is captured structurally, no phases cell), and its `e2e_ms`
  includes the flat BvN `schedule_ms` recompute (~4.4 ms/iter on p4d) — call
  that out in small-budget comparisons. EP-semantics arms (driver-swapped
  tests, six aliased phase names, no phases cells — SCHEMA.md is the
  authority): `moonep[_overlap|_nvshmem|...]` (per-batch global token
  migration), `ultraep[_domain16|_overlap|_nvshmem|...]` (per-batch
  NVL-confined expert replication), and `eplb` (STATIC placement from the
  full-pool predicted load — trace-family cells only for headline numbers;
  the runner auto-generates the `<mid>.eplb_load.json` prediction sidecar
  from the cell's exact pools).
- `--families` — traffic distribution: `uniform`, `hotcol[:frac=..]`,
  `nodeskew[:frac=..]`, `remotefrac` (per-rank inter-node skew, the relay's
  target case), `fanoutskew[:nodefracs=..]` (per-NODE exporter skew, the
  starvation-campaign family), `trace` (REAL Qwen3-235B routing sampled from
  the Patterns-behind-Chaos pools; needs `pools=...;layer=..;sem=..` params,
  fetched pools under the platform `traces_root`, and emits a
  `.routing.txt` sidecar the bench consumes via `--routing_file` — see
  SCHEMA.md). Distribution identity = matrix_id (deterministic; generated
  on demand).
- `--budgets-mib` — pre-topk budgets, typically `2,4,8,16,32,64`.
- `--topk`, `--G` — routing shape (G % world_size == 0; the generator
  pre-checks routing feasibility and tells you the minimum G).
- `--modes` — `isolated` (**the default for latency claims**:
  FLUX_SWEEP_ISOLATED_ITERS=1 syncs + barriers before every timed window, so
  each iteration is one isolated layer execution — inference semantics),
  `e2e` (clean, back-to-back — the *throughput* view; never compare against
  `isolated`), `phases` (FLUX_A2AV_TIMING breakdown —
  perturbed, never compare its e2e against clean cells), `nsys` (primary
  timeline: per-node Nsight capture, CE/P2P visibility, NVTX iter
  ranges), `torchprof` (secondary timeline: Python-op attribution). For
  overlap investigations run `--modes e2e,nsys`; start analysis with
  `nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum <rep>`.
  Export `FLUX_A2AV_NVTX_PROXY=1` with an nsys cell to add per-source
  wait/pending/compute ranges inside the a2av GEMM span (NVTX domain
  "a2av"; see sweeps/SCHEMA.md). `FLUX_A2AV_EARLY_LAUNCH=1` (any a2av
  variant) reorders GEMM-before-intra-wire — treat as its own configuration;
  `FLUX_A2AV_BLOCKING_WIRE=1` + `CUDA_DEVICE_MAX_CONNECTIONS=8` is the
  overlap-visualization recipe (instrumented only).
  **Default for profiling captures (user directive 2026-08-05): nsys/timeline
  cells carry `FLUX_SWEEP_ISOLATED_ITERS: "1"` and `FLUX_A2AV_EARLY_LAUNCH:
  "1"` in `extra_env`** — plain nsys mode runs profiled iterations
  back-to-back (per-iter sync+barrier exists only under the isolated knob),
  so without it iter-n wire tails contaminate peer ranks' iter-n+1 on the
  timeline; early launch makes the GEMM/comm overlap visible. Template:
  `sweeps/specs/lb_union_pm4n_trace_b32skew_blocking_iso.yaml`. Still
  instrumented — never a latency source.
- `--profile-iters` — iters AND warmup for torchprof/nsys cells (default 3).
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
5. Report results from the capsule. For latency claims quote `isolated`-mode
   cells: mean over iterations of per-iteration max-across-ranks of e2e_ms
   (the runner prints it; recompute from metrics.csv when needed). Pipelined
   `e2e` cells are the throughput view (max-rank and mean over ranks of e2e_ms
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
- An `nsys_empty` cell status means the capture was lost (killed nsys /
  empty trace) — the rep is missing or undersized; rerun the cell rather
  than trusting an ok-looking srun.log.
- Single-node runs can't use `remotefrac`/`nodeskew` (no remote ranks);
  use `uniform`/`hotcol`.
