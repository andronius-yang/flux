# Sweep data contract

This file is the authority for what sweep results mean. If code and this file
disagree, fix one of them in the same commit.

## Budget semantics (read this first)

**`budget_mib` is STRICTLY the pre-topk send budget per source rank** — the
bytes of *unique tokens homed on the rank* before topk fan-out:

- `tokens_per_rank = budget_mib * 2^20 / chunk_bytes` (independent of topk)
- matrix **row sums** = `budget_mib * 2^20 * topk` bytes (post-fanout wire rows)

A "16 MiB budget at topk 8" therefore moves up to 128 MiB of logical wire rows
per rank. Never quote a budget number without this framing; sweeps before
2026-07-29 used both framings inconsistently, which is why this rule exists.

## Run capsule

One runner invocation = one **immutable** capsule:

```
sweeps/results/runs/<run_id>/     # committed to git (small)
  manifest.json    # provenance: git sha+dirty, .so hashes, capability probe,
                   # platform, per-artifact {path, sha256, bytes}
  spec.yaml        # the fully-resolved spec — `sweep.py rerun <capsule>` re-executes it
  metrics.csv      # per-(rank, iteration, metric) rows — never aggregated
  cells.csv        # one row per cell, including skipped/failed ones
<data_root>/<run_id>/cells/<cell_id>/   # platform-local staging (NOT committed)
  records/rank_NNN.jsonl   # per-rank recorder output (hashed in manifest)
  torchrun/...             # per-rank stdout/stderr ([a2av-*] marks live here)
  srun.log  nsys/  prof/
```

Capsules are never edited; a re-run mints a new run_id. Raw staging can be
deleted once a capsule is committed — the manifest hashes record what existed.

- `run_id` = `<UTC yyyymmdd-HHMMSS>_<platform>_<hex8>` (hex8 = sha256 of
  spec+git+host+ns).
- `cell_id` = `{variant}_{family_slug}_b{budget}_k{topk}_{mode}`.
- `matrix_id` = `w{W}x{L}_{family}[-p]_b{budget}_k{topk}_id{instance}` — a pure
  function of the generator inputs; the FNV-1a seed derives from the same
  canonical string, so any platform regenerates a byte-identical file.

## metrics.csv

One row per (rank, iteration, metric). Aggregation (max-rank, mean, p50) is
the summarizer's job, never stored here.

| column | meaning |
|---|---|
| run_id, cell_id | capsule + cell identity (join to cells.csv) |
| mode | `e2e` \| `phases` \| `torchprof` \| `nsys` — see mode rules below |
| impl | `flux` (the op under test) or `torch` (dense reference) |
| rank | global rank (0-based) |
| iter | post-warmup iteration index (0-based) |
| metric | see below |
| value_ms | milliseconds, always |
| source | `recorder` (CUDA-event, in-process) or `stderr` (parsed [a2av-*] marks) |

Metrics from the recorder: `e2e_ms` (flux, per-iteration op.forward wall);
`comm_ms`, `scatter_ms`, `gemm_ms` (torch reference phases); FAST baseline
(`impl=fast`, per iteration): `e2e_ms, pack_ms, schedule_ms, fill_ms, wire_ms,
unpack_ms, gemm_ms` (components of the e2e window), `reset_ms` (inter-iteration
hygiene OUTSIDE the window — never add it to e2e), `host_e2e_ms` (host-wall
cross-check). Metric meaning is scoped by `impl` — `gemm_ms` under torch, fast,
and (phases) flux are three different measurements by design.

Metrics parsed from stderr in `phases` mode (per iteration, per rank):
`stage1_ms, stage2_ms, gemmgate_ms, a2av_gemm_ms, barrier_ms` ([a2av-timing]),
`stage2_*_ms` ([a2av-stage2] sub-marks), `host_enq_stage1_ms,
host_enq_stage2_ms, host_counts_wait_ms` ([a2av-host], µs→ms), and
`relayfwd_*_ms` ([a2av-relayfwd], balanced-relay gather builds only — the
`hier_compress_lb_union` variant runs no forward-index build and never emits
this mark, like the union modes).

## cells.csv

One row per cell, including `skipped_capability` / `failed` / `timeout` ones.
Highlights (full list = header row):

- `status` — `ok`, `failed`, `timeout`, `skipped_capability`.
- `deterministic` — AND over all ranks' recorded
  `torch.are_deterministic_algorithms_enabled()`. Perf cells must be 0; this
  column is the audit that they were (never assume — the 2026-07-29 root-cause
  showed deterministic scatter_ inflating compress paths ~500x).
- `env_json` — the full env delta the runner constructed (knob scaling,
  variant env, mode env). What you'd need to reproduce by hand.
- `wire_ratio`, `relay_ident_bytes`, `relay_balanced_bytes` — compress-only
  cell facts from the pre-run analysis block.
- `correct_bitwise`, `correct_allclose` — AND over ranks; empty when
  `skip_correctness` ran. Note bitwise may legitimately be 0 with determinism
  off; allclose is the correctness verdict.
- `matrix_sha256`, `git_sha`, `git_dirty` — provenance; treat `git_dirty=1`
  results as unreproducible.

## Modes — the never-mix rule

- `e2e` — clean run, no instrumentation, back-to-back iterations (pipelined:
  the host runs ~1 iteration ahead; consecutive iterations hide parts of each
  other's wire tails). This is the **throughput view**; its `e2e_ms` may be
  quoted as pipelined steady-state latency only, and never compared against
  `isolated` cells.
- `isolated` — `FLUX_SWEEP_ISOLATED_ITERS=1` (set by the runner): the harness
  runs `torch.cuda.synchronize()` + `torch.distributed.barrier()` before EVERY
  timed window, so each iteration measures one isolated layer execution —
  inference semantics (routing changes per activation; no cross-iteration
  pipelining, no launch-skew echo). **This is the default mode for latency
  claims.** The quoted number is the mean over iterations of the
  per-iteration MAX across ranks of `e2e_ms` (the runner prints it at cell
  finish; metrics.csv stays raw per the aggregation rule). `iso_sync_ms` (per
  rank, per iteration) is the host wall time of the sync+barrier pair — a
  straggler-skew indicator, not a latency component. Never compare isolated
  vs `e2e` cells: they measure different regimes. The knob may also ride any
  other mode via spec `extra_env` (e.g. an nsys capture under isolated
  discipline); audit via `env_json` / recorder `meta.env`.
- `phases` — `FLUX_A2AV_TIMING=1`. The C++ marks force a per-iteration device
  sync, so **e2e numbers from phases cells are perturbed and must never be
  compared against e2e-mode cells**. Use them for the phase breakdown only.
- `nsys` — Nsight Systems capture, **the primary timeline mode**: one
  `.nsys-rep` per node (all torchrun ranks followed; validated on the AWS
  p4d stack 2026-07-31), showing CE vs SM attribution, P2P memcpy sizes and
  rates, and per-iteration NVTX ranges (`iterN` / `iterN_warmup`). Trace set
  is `cuda,nvtx` only — osrt is deliberately off (the NVSHMEM/EFA proxy
  thread busy-polls, and osrt-tracing it emits ~18 GB/min, filling the
  node-local disk); proxy-thread cost shows up indirectly as wire latency.
  The runner enforces one full-size rep per node — a missing/undersized
  capture demotes the cell to status `nsys_empty`. Exporting
  `FLUX_A2AV_NVTX_PROXY=1` adds live per-source ranges inside the a2av GEMM
  span (NVTX domain "a2av": `src<s>.wait/pending/compute` + completion
  quartiles + `intra/inter_epoch` envelopes, emitted by a host poller fed
  from device progress counters; edges lag device truth by ~5–30 µs).
  Timeline aid only — it adds no metrics.csv columns and its cells follow
  the same never-compare rule as any instrumented mode. The same gate also
  writes per-rank tile-trace sidecars (`a2av_tile_trace_r<rank>.bin`) into
  the cell's records dir at the platform data root (raw, never committed);
  analyze with `sweeps/plot_a2av_trace.py` (cohort table, fired/in-flight
  regime plot, Perfetto per-SM Gantt). Related knobs:
  `FLUX_A2AV_EARLY_LAUNCH=1` (any a2av variant since 2026-08-02: GEMM
  launches right after stage 2, intra wire issued after it — a functional
  reorder, so e2e cells with it on are a separate variant configuration, not
  comparable to gate-order cells; at the variants' default CONNECTIONS=8
  union measured -12% vs the old conn=1 baseline. On the compress
  gather/relay arms the index_select tails are issued inline on the pack
  stream (post-launch-enqueued kernels can starve at dispatch behind the
  blanketing GEMM) and the ctor requires CONNECTIONS > 1 — this includes
  `hier_compress_lb_union`, whose phase-2 wire carries inline pre-launch
  front-end waits even though its gateway forward is pure-CE — visibility
  mode, not a perf configuration) and `FLUX_A2AV_BLOCKING_WIRE=1` (instrumented
  only: inter-node puts — hier/compress aggregates, relay phase-2, flat
  remote per-dest — become visible device spans; under
  CUDA_DEVICE_MAX_CONNECTIONS=1 they serialize ahead of the GEMM — the
  three-way overlap capture recipe is EARLY_LAUNCH=1 BLOCKING_WIRE=1
  CUDA_DEVICE_MAX_CONNECTIONS=8, nsys mode).
- `torchprof` — `--profile` chrome traces; secondary timeline mode, kept for
  Python-op/shape attribution (which ATen call launched which kernel).

Timings from profiling modes are for inspection, not comparison.

The `mode` column rides along in metrics.csv precisely so no join is needed to
filter clean numbers.

**`impl=fast` rows appear only in `mode=e2e`.** The FAST baseline's phase
decomposition (`pack/schedule/fill/wire/unpack/gemm_ms`) is captured
structurally — its alltoallv is host-blocking and the e2e window syncs every
iteration by construction (the un-overlapped one-shot methodology) — so phase
capture does not perturb `e2e_ms` and there is no separate `phases` cell for
fast. The never-mix rule above governs `FLUX_A2AV_TIMING` instrumentation of
flux cells only. Semantics: fast `e2e_ms` is the comm-start → gemm-finish
CUDA-event window ≈ pack + schedule + fill + wire + unpack + gemm; the flat
BvN `schedule_ms` recompute (~4.4 ms/iter on p4d CPUs, ~0.9 on Perlmutter) is
INSIDE it and floors small budgets — quote it alongside small-budget
comparisons. Comparable to flux `e2e_ms` (both windows include their
pack-equivalents).

## Comm/comp attribution

Phase wall-times from `phases` mode inherently conflate overlap (comm hidden
under the GEMM tile-spins shows up as gemm time, not comm time). The honest
decomposition protocol when needed: pair an `e2e` cell with a comm-only or
comp-only ablation and difference the results in the summarizer — never store
derived "exposed comm" as raw truth. Timeline modes are the ground truth for
"was it actually overlapped" — nsys first (sees CE memcpys, proxy threads,
and cross-process node timelines that the torch profiler cannot), torchprof
when Python-op attribution is needed.

## Knob scaling (validated anchors)

`scale_knobs` in sweep.py, overridable per-platform:

- `FLUX_A2AV_MAX_{RECV,STAGE,RELAY}_NTOKENS = max(163840, 4 * row_chunks)`
  (rounded up to 8192), where `row_chunks = budget_mib * 2^20 * topk /
  chunk_bytes`. Anchors: post-topk ≤256 MiB → 163840; 512 → 262144; 1024 →
  524288 (topk16 sweeps, 2026-07-29).
- `NVSHMEM_SYMMETRIC_SIZE`: 6G up to post-topk 256 MiB, linear to 16G at
  1024 MiB, capped by platform `sym_size_max_g`.

Under-sizing fails loudly (FLUX_CHECK recv-overflow / NVSHMEM init) — but note
the recv-overflow check is per-rank data-dependent: some ranks throw while the
rest spin at 100% GPU. That is what the cell timeout is for.

## Protocol rules

1. Perf cells always run with `FLUX_TEST_DETERMINISTIC=0` (runner-enforced,
   recorder-audited via the `deterministic` column).
2. EFA transients can inflate a whole cell ~3x (validated 2026-07-29). If a
   cell is a wild outlier, rerun the sweep (`sweep.py rerun`) and compare
   capsules before believing it.
3. `--skip_correctness` is for large budgets where the torch reference
   dominates or OOMs; its cells carry empty correctness columns — visible,
   not silent.
4. **Prefer paired arms inside ONE capsule.** All defensible results in this
   project are within-capsule A/Bs. Comparisons across capsules must match on:
   platform, world_size, topk, G, matrix_id, mode, deterministic — **and on
   the build**, i.e. the `flux_libs` sha256 in `manifest.json`. `git_sha` is
   NOT a build identity: 124 capsules span only 5 shas but **28 distinct
   `libflux_cuda_ths_op.so` builds**, and 308 of 405 cells were produced from
   a dirty tree. Measured spreads for one unchanged configuration: ~0.3–1.7%
   across capsules at b8/b32/b64 on the *same* build (up to ~4.8% at b2), but
   **6–33% across different builds**. Every headline claim in this project is
   ≤7%, i.e. below the cross-build spread — so a cross-build comparison can
   manufacture a result of either sign. If you must compare across capsules,
   state the build hashes and treat anything below the same-build spread as
   noise. See `docs/handoff/04_build_ledger.md`.
