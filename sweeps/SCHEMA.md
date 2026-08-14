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
  For the `trace` family the canonical params include an injected `poolsha`
  (content fingerprint of the referenced trace pools' `pool.manifest.json`
  shas), so identity is a pure function of the actual trace bytes: re-fetched
  or changed traces mint a new matrix_id rather than silently regenerating a
  different file under the old one.

### The `trace` family (real MoE routing; gen_trace_routing.py)

Batches sampled from real expert-selection traces (the "Patterns behind
Chaos" dataset, arXiv 2510.05497; fetched by `fetch_traces.py`, analyzed
offline by `trace_analysis.py`). Differences from the synthetic families:

- **Two artifacts** per matrix_id under `matrices_root`: the `[W][W]` byte
  matrix `.txt` (derived) and `.routing.txt` — the per-token expert ids that
  produced it (line 1 `ntokens topk G`, then one token per line). The bench
  consumes the routing via `--routing_file`, which bypasses the synthetic
  dealer. This matters because the dealer is the MAXIMUM-dedup assignment:
  feeding trace-derived bytes through it overstates every dedup/union win and
  the closed-form `dedup_round_stats` law is INVALID for real routing. A
  spec arm with `dealer=1` in the family params deliberately runs the same
  bytes through the dealer (identical matrix_id, distinct cell) as the
  token-overlap counterfactual; `cells.csv` records `routing_mode`
  (`real`/`dealer`), `routing_path`, `routing_sha256` (empty for synthetic
  families).
- **Nonzero diagonal**: real tokens route to experts owned by their home rank.
  Row sums still satisfy the budget invariant (`budget_mib*2^20*topk`), but
  wire bytes per row = row_sum − diag, so trace arms move fewer wire bytes
  than a synthetic family at the same budget. Compare lb vs union (or any
  variants) on the SAME trace matrix; never quote trace-vs-synthetic latency
  as a family effect.
- Params: `pools` (`+`-joined `bench/subject` list; ORDER is semantic for
  `sem=pernode` — node i samples pools[i]; canonical-sorted for `mixed`),
  `layer` (int), `sem` (`homog`/`pernode`/`mixed`), `pool`
  (`decode`/`prefill`/`all`), `model`. Sampling is with replacement (the pool
  is a distribution, not a corpus); `sample_multiplicity` in the meta records
  the draws/pool-size ratio. Requires the platform yaml key `traces_root`.

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

MoonEP-semantics arm (`impl=moonep`, variant `moonep`, driver-swapped test
`test_moe_moonep_traffic.py` — a semantic port of MoonshotAI/MoonEP
redundant-expert dispatch; plan bit-equality vs the vendored MoonEP oracle is
enforced by `test_moonep_planner.py`): per iteration, in every mode,
`plan_comm_ms` (topk allgather — the replicated-planning wire; excluded from
`comm_ms`), `pack_ms` (dest-sorted send-buffer gather — a port-added local
copy MoonEP's one-sided writes don't have), `comm_ms` (NCCL alltoallv of
dedup'd representative rows + per-entry fp32 route weights — the pure wire
number; wire rows/bytes equal MoonEP's dedup semantics), `scatter_ms`
(placement scatter + zero-fill + local duplicate expansion; contains the
second port-added copy), `prefetch_ms` (redundant expert-weight movement,
home rank -> prefetch slots — weight traffic, reported separately so token
comparisons vs other arms stay apples-to-apples; layer0 moves 1 weight
matrix where MoonEP training moves 3), `gemm_ms` (per-segment GemmOnly over
`cu_seqlens[E+B]`, padded rows computed per the MoonEP contract), `total_ms`.
The plan itself is deterministic integer host math computed identically on
every rank at setup (untimed-metadata contract, like `splits_per_source`;
reported as cell_info `moonep_plan_host_ms`).

M4 arms (same driver, same plan and correctness gates — the grid isolates
transport and overlap with semantics held fixed; cell facts
`moonep_transport` / `moonep_overlap_prefetch` / `moonep_shared_comm_stream`
audit which ran):
`moonep_nvshmem[_overlap]` swaps the dispatch a2av for flux's one-sided
NVSHMEM `All2AllSingle`. Live path (correction 2026-08-11): `putmem_nbi`
per destination into fixed per-source slots of symmetric staging + two
team stream barriers per call — put-then-barrier, the same publication
shape as MoonEP's real dispatch (TMA push + one system-scope exit
barrier); the file's `putmem_signal` kernel is dead code, never launched
(NR-12 fact 8). Still sender-driven and receiver-passive on the wire;
needs `NVSHMEM_SYMMETRIC_SIZE`, sized by the runner from the matrix.
`moonep_overlap` / `moonep_nvshmem_overlap` run prefetch on a dedicated
high-priority stream + separate NCCL communicator, event-joined before GEMM
(MoonEP `async_finish` comm-stream semantics): `prefetch_ms` becomes the
prefetch STREAM duration (fork after plan_comm -> done) and a new
`prefetch_wait_ms` is the exposed stall the main stream paid at the join —
the overlap-adjusted layer time is `total_ms`; never sum phase columns on
overlap arms (phases run concurrently, and contention legitimately inflates
`pack_ms`/`comm_ms` there). Cell facts: `moonep_z_matrix`
(home-group -> dest migrations), `moonep_wire_bytes` (realized dedup'd wire
matrix — differs from the input matrix BY DESIGN, MoonEP rebalances),
`moonep_nvs`, `moonep_prefetch_recv_bytes`, and `gemm_rows_per_rank`, which
is constant (S*K + padding) across ranks — the balance fingerprint.
No `phases` cells (the breakdown arrives free in every mode); no NVSHMEM
heap and no FLUX_A2AV_* knobs (pure NCCL + local scatters).

UltraEP-semantics arm (`impl=ultraep`, variants `ultraep` /
`ultraep_domain16`, driver-swapped test `test_moe_ultraep_traffic.py` — a
semantic port of Dots-Infra/UltraEP replicated-expert balancing; solver +
reroute BIT-equality vs UltraEP's real kernels is enforced by
`test_ultraep_planner.py` against the SM80 oracle build and the vendored
goldens under `ultraep_oracle/`). Same six phase names as moonep so arms
compare inside one capsule, with aliased meanings: `plan_comm_ms` is the
[W, G] int32 LOADS allgather (UltraEP's metadata fcollect, ~8 KiB — NOT
moonep's [S, K] topk allgather; `ultraep_plan_comm_bytes` audits the
asymmetry), `comm_ms` carries NO dedup (one wire row per (token, physical
expert), faithful to UltraEP/Megatron dispatch; `ultraep_dup_rows` counts
what dedup would save), `prefetch_ms` is UltraEP `weight_sync` (`direct`
plan; master -> replica-slot copies, intra-NVLink-domain by construction at
D=4; fc1 feeds the GEMM, fc2 rides along bitwise-verified so bytes match
UltraEP's full-expert sync — `--weight_sync fc1` gives moonep-comparable
bytes, `ultraep_weight_sync` audits), and `gemm_ms` runs UNPADDED
per-physical-expert segments. CRITICAL contrast to moonep:
`gemm_rows_per_rank` is NOT constant — replication is confined to the
NVLink domain (one independent solve per node), so cross-node imbalance is
untouched by design and the residual spread IS the measurement. Cell facts:
`ultraep_imbalance_before/after` (rank max/mean), `ultraep_lb_floor` (max
domain-mean / global-mean — the reachable floor), `ultraep_threshold_T` +
`ultraep_solver_path` per domain (fast/precheck/bisect audit),
`ultraep_replicas_total/_max_per_expert`, `ultraep_remote_frac_with/
without_locality`, `ultraep_wire_bytes`, `ultraep_weight_sync_recv_bytes`,
`ultraep_nvl_domain_size`, `ultraep_plan_host_ms` (untimed-metadata
contract). `ultraep_domain16` treats the whole EP16 group as one domain
(rack-scale counterfactual: weight_sync crosses nodes, LB floor -> 1.0);
it prices the fabric assumption and is NOT a faithful Perlmutter
deployment. `ultraep_overlap[_joingemm]` run weight_sync on a dedicated
high-priority stream + separate NCCL communicator forked after plan_comm
(facts `ultraep_overlap_ws`, `ultraep_ws_join`; join semantics and the
prefetch_ms/prefetch_wait_ms rule in the fidelity paragraph below).
`ultraep_nvshmem[_overlap]` swaps the token a2av for the one-sided
All2AllSingle (putmem_nbi + 2 team barriers; fact `ultraep_transport`;
nvshmem overlap is join=dispatch ONLY — nvshmem+joingemm is forbidden,
NR-02 Class-B surface); weight_sync stays NCCL P2P in every arm (declared
port artifact, NR-12 fact 8). No `phases` cells; no FLUX_A2AV_* knobs;
NVSHMEM heap only for the nvshmem arms (`ultraep_sym_size`: no-dedup,
domain-bounded — reroute confines redirection to the logical target's NVL
domain).

EPLB arm (`impl=eplb`, variant `eplb`, driver-swapped test
`test_moe_eplb_traffic.py` — the VENDORED deepseek-ai/EPLB @ d52c72d under
`eplb_oracle/` is the production algorithm itself, mapped onto the shared
UltraEPPlan machinery by `eplb_semantics.py`; invariants + the DeepSeek
README-example pin enforced by `test_eplb_planner.py`). Static
predicted-load placement: ONE placement per cell — full re-placement,
masters move too, global policy, same nlp = G/W + 2 slot budget as ultraep
— computed from the FULL pool histogram sidecar `<mid>.eplb_load.json`
(generated by the runner from the cell's exact pools; the oracle-ceiling
prediction; `eplb_load_sha` is the provenance fact — matrix identity is
UNCHANGED, the load file is placement input, not routing). Same six phase
names, with these meanings: `plan_comm_ms` is the same [W, G] loads
allgather as ultraep and is KEPT per iteration — recv splits still need the
counts exchange; a zero plan_comm claim is invalid. `comm_ms`/`scatter_ms`
are ultraep-identical (no dedup, `eplb_dup_rows` audits; staged pack ->
a2av -> place). `prefetch_ms` is EMPTY (~0 by construction): there is no
per-batch weight movement — this near-zero column vs moonep prefetch /
ultraep weight_sync IS the arm's recurring-cost advantage, and the one-time
placement it buys is book-kept as `eplb_weight_place_bytes` (per rank) +
`eplb_weight_place_ms_oneshot` (never entered into latency). `gemm_ms` runs
unpadded per-physical-expert segments; `gemm_rows_per_rank` spread is the
residual imbalance of the STATIC placement on each sampled batch — the
measurement (between moonep's constant S*K and the flux fixed-placement
skew, exactly as good as the pool predicts the batch; the iid trace sampler
is EPLB's best case, so read `eplb_imbalance_after` as a CEILING). Token ->
physical split: equal split per expert across instances
(largest-remainder), per-source decomposition via the shared NON-locality
rank-quota path + coprime interleave. The locality-aware path is
structurally forbidden on EPLB plans (replicas are not NVL-confined);
`eplb_remote_frac` is the no-locality number ONLY. Cell facts:
`eplb_policy`, `eplb_imbalance_before/after`, `eplb_pred_imbalance` (on the
pool load — the packing objective), `eplb_replicas_total/_max_per_expert`,
`eplb_rehomed_slots`, `eplb_remote_frac`, `eplb_wire_bytes`,
`eplb_load_source` (`pool` for headline cells; `batch` = self-oracle
fallback, smoke only), `eplb_load_sha`, `eplb_plan_host_ms`
(untimed-metadata contract), `eplb_transport`. Canonical transport is the
one-sided All2AllSingle (NCCL remains only for plan_comm and the one-time
setup P2P); heap via `eplb_sym_size` — the ultraep domain bound is UNSAFE
under global re-homing, so the full row-sum bound is used. No `phases`
cells; no FLUX_A2AV_* knobs.

Overlap and transport fidelity (moonep + ultraep arms — canonical; cited by
insight-ledger NR-12, read that entry before proposing overlap work): the
serialized arms are deliberately-pessimistic ports — BOTH upstreams overlap
their weight movement (UltraEP: async weight_sync on a dedicated
high-priority comm stream, event-joined by the integration; MoonEP: async
dispatch+prefetch on ONE shared comm stream, serialized with each other,
overlapping only main-stream compute — so `moonep_overlap` is FINER than
upstream and `moonep_overlap_shared` is the authentic-serialization
counterpart). Join points are NOT symmetric: MoonEP consumes prefetch at
GEMM, so joining before GEMM is authentic for moonep; UltraEP's reference
integration joins weight_sync BEFORE token dispatch, and that join is the
PUBLICATION MECHANISM, not a preference — UltraEP's direct mode pushes to
peer VAs with no receiver-side signaling at all, so the dispatch
collective's completion is the only cross-rank happens-before (NR-12 fact
5). The ultraep overlap arms expose this via `--ws_join`: `dispatch` =
authentic (window = pack only; weight_sync mostly exposed, matching the
paper's own accounting), `gemm` = counterfactual (window =
pack+comm+scatter; sound ONLY because this NCCL port is two-sided — irecv
completion rides the ws event; label like `ultraep_domain16`).
Overlap-arm metric rule (all overlap arms, both drivers): `prefetch_ms` =
ws/prefetch STREAM duration fork→done (off-chain; never sum phase columns
on overlap arms), `prefetch_wait_ms` = exposed stall from the event
preceding the join to `join`, and the phase after the join measures from
`join`, so the six phase columns always partition [start, gemm].
The separate NCCL communicator the concurrent-overlap arms need is PORT
MACHINERY (NCCL serializes per communicator) — neither upstream has a
communicator in its weight path (UltraEP: raw NVLink ld/st via
`nvshmem_ptr` peer VAs from its own SM kernel; MoonEP: destination-side
TMA remote-read PULL). Transport authenticity ladder for weight movement
(corrected 2026-08-11 — put+signal is NOT what either upstream does):
custom NVSHMEM kernel matching the real upstream shape (bare `putmem_nbi`
push published by the subsequent collective join for UltraEP direct /
`getmem` pull for MoonEP prefetch) > CUDA-IPC peer copies > NCCL P2P
(two-sided rendezvous + own protocol optimizations — declared port
artifact, NR-12 fact 8). The MoonEP rung at the top of that ladder is
IMPLEMENTED as of 2026-08-11 (`flux.WeightPrefetchGetmem`; arms
`moonep_getmem`, `moonep_getmem_overlap`, `moonep_nvshmem_getmem` — NR-12
facts 9-11): SM-kernel getmem_nbi_block pulls from a symmetric weight
home into symmetric prefetch slots (BOTH must be symmetric — CXI proxy
gets segfault into unregistered local memory, fact 10), joined by one
quiet_on_stream; the getmem overlap arm needs NO second communicator.
The `moonep`/`moonep_overlap`/`moonep_nvshmem` arms keep NCCL weights for
capsule comparability with pre-M4d history.

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
