# Sweep data contract

This file is the authority for what sweep results mean. If code and this file
disagree, fix one of them in the same commit.

## Budget semantics (read this first)

**`budget_mib` is STRICTLY the pre-topk send budget per source rank** — the
bytes of *unique tokens homed on the rank* before topk fan-out:

- `tokens_per_rank = round(budget_mib * 2^20 / chunk_bytes)`, rounded to
  the nearest MULTIPLE OF TOPK (the layer1/l01 lane requires
  `ntokens % (W*topk) == 0`; the invariant is global)
- matrix **row sums** = `tokens_per_rank * chunk_bytes * topk` bytes
  (post-fanout wire rows)

Nominal byte ladder (2026-08-21): budget labels are the power-of-two
ladder b1/2/4/8/16/32/64 for every shape. The rounding is EXACT whenever
`chunk_bytes` divides the budget — every pre-2026-08-21 label (chunk 8192:
tokens = 128*budget, always topk-divisible) — and the realized per-rank
budget `tokens_per_rank * chunk_bytes` is recorded as
`effective_budget_bytes` in meta.json and cells.csv (K3 worst labeling
error ~1.6% at b1/b2, <=0.2% from b8 up — quote effective bytes when it
matters). The interim K3 7-MiB-multiple labels (b7/b28, capsules
20260821-104600/-105153 only) were byte-exact under chunk 7168; never
conflate the labels.

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
  families). Since 2026-08-21 the layer0 `fast` driver also consumes
  `--routing_file` on real cells (FAST is a comm-phase substitute: matrix
  AND per-expert gemm loads are both trace-derived; the test asserts the
  routing realizes the matrix), as does the combined `l01_fast` driver.
  `dealer=1` remains an ABLATION arm only. The standalone layer1 `l1_fast`
  driver still cannot consume routing (unchanged).
- **Trace semantics canon (user decision 2026-08-21): `sem=homog` for ALL
  future realistic-traffic sweeps** — pernode is retired (only 8 K3 pools
  exist; homog scales to any node count; the 20260821-104600/-105153
  capsules are the last pernode ones). **Canonical homog pool:
  `mmlu/professional_law`** (the topic-concentrated hot-expert workload
  used across prior campaigns); `livecodebench/execution` is the
  designated ALTERNATE for topic-shift/domain ablations. Canonical K3
  family string: `trace:model=Kimi-K3-synth;pools=mmlu/professional_law;
  layer=92;sem=homog`.
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
- **`model=Kimi-K3-synth` (2026-08-20): synthesized-empirical pools.** No
  captured K3 routing exists; these pools are generated by
  `synthesize_k3_traces.py` — real Kimi-K2 per-topic per-layer hotness
  marginals, upscaled 384→896 by exact hot-split slot expansion (hottest
  128 K2 experts → 3 child slots, rest → 2; per-topic marginals mass-exact,
  cross-topic divergences preserved exactly), sampled at topk 16
  (provenance, seeds, and the K2→K3 layer relabeling in each
  `pool.manifest.json`). Two-sided reading, both halves binding:
  (1) the routing is DERIVED, never presentable as captured K3 traces, and
  within-token/cross-token co-occurrence is not modeled — placement/dedup
  wins that depend on co-occurrence structure are NOT demonstrable on these
  pools; (2) the perf numbers are CONCRETE — real kernels, real wire, real
  placement decisions, measured under a distribution fitted from real K2
  routing. Caveat direction is pre-registered: K2-derived curves likely
  OVERSTATE K3 imbalance (K3 trains with Quantile Balancing; the
  measured sparsity→flatness trend), so expert-movement/placement wins on
  these pools are upper bounds.

## metrics.csv

One row per (rank, iteration, metric). Aggregation (max-rank, mean, p50) is
the summarizer's job, never stored here.

| column | meaning |
|---|---|
| run_id, cell_id | capsule + cell identity (join to cells.csv) |
| mode | `e2e` \| `phases` \| `torchprof` \| `nsys` — see mode rules below |
| impl | `flux` (the op under test), `torch` (unfused reference — BOTH layers emit it whenever correctness is on, so the un-overlapped torch baseline rides every flux cell for free), `fast`, or an EP-arm name. NOTE: the runner's isolated console summary aggregates `impl=flux` only — summarizer scripts comparing baselines must also read the `torch`/`fast` rows |
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

Rule-5 conversion of the flux and fast drivers (2026-08-21, protocol rule
5): `impl=flux`, `impl=torch`, and `impl=fast` rows additionally carry
`plan_comm_ms` (the per-iteration [ntokens, topk] routing allgather — the
recurring exchange that makes routing globally known) and `plan_ms` (the
per-iteration on-GPU derivation of ALL routing-derived metadata: flux/torch
via the op's fused `derive_routed_meta` — splits, stable scatter_index,
splits_per_source, compress unique counts, plus the torch reference's
gather_index reconstruction and splits D2H; fast via the vectorized
torch-GPU `derive_fast_meta_gpu` — pack/unpack indices, gemm splits, and
the BvN input matrix, since the FAST process owns the only NVSHMEM init
and constructs no flux op), and `total_ms = plan_comm + plan + e2e-window`.
`e2e_ms` keeps its historical meaning (the op.forward / comm-start→gemm
window; anchors unchanged). **Quote `total_ms`** for isolated latency on
these drivers; capsules stamped `timing_accounting=per_iter_gpu`. Setup
still computes the same metadata once, untimed, purely as a bitwise drift
guard on the in-window derivation. Never compare planning-inclusive totals
against pre-2026-08-21 flux/fast capsules (the never-mix boundary is the
driver change; pre-boundary capsules simply lack these metric rows).

`plan_ms` (2026-08-20, protocol rule 5; drivers eplb/epic/moonep): the
per-iteration on-device plan derivation inside the timed bracket —
plan_comm-end to plan-end. It is the visibility guard against
planning-dominated totals: quote plan_ms/total_ms per arm in capsule notes,
and if it exceeds ~1-2% of the isolated total for an arm, investigate
before quoting the arm's verdicts (the number stays honest either way; the
cell facts `timing_accounting` and `moonep_planner_impl` carry the
caveat). `plan_ms` sits inside `total_ms` but OUTSIDE `e2e_ms` (the
comm-start anchor is unchanged).

Planner v2 (2026-08-20, campaign 2 — `planner_impl=fused_dispatch`): on the
fused-canonical arms (`eplb_fused*`, `epic_*_mig_fused`) there is no separate
planner op at all — planning is fused into the dispatch, DeepEP-lineage:
sender-local replica selection + an in-launch counts exchange that derives
exact recv layouts inside the dispatch kernel chain. `plan_ms` on these arms
is structurally near-zero (only the [S,K]→physical-slot map derivation
remains outside the op) and the counts exchange + arrival gating are
dispatch wire time, charged to `comm_ms`. EPLB fused runs NO pre-dispatch
collective in either replica mode (`plan_comm_bytes=0`); EPIC always keeps
its timed per-iteration loads allgather + migration decision (paper order:
gate → counts sync → migrate → dispatch), so its `plan_comm` stays nonzero.

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
comparisons vs other arms stay apples-to-apples; layer0-only runs move 1
weight matrix where MoonEP moves 3, `--layers l01` runs move 2 — w1+w2 in
ONE prefetch phase under one join, the upstream one-pass contract), `gemm_ms`
(per-segment GemmOnly over `cu_seqlens[E+B]`, padded rows computed per the
MoonEP contract), `total_ms`. Under `--layers l01` (arm
`moonep_l01_nvshmem_getmem`, 2026-08-17) five more brackets extend the chain
after `gemm_ms`: `act_ms` (gelu), `gemm2_ms` (down-projection over the same
segments), `cpack_ms` (route-weight scale + reverse-dedup partial sums + pack
— one row per (token, dest) returns, wire symmetric with dispatch),
`comb_ms` (the DIRECT a2av transpose back), `acc_ms` (home-side index_add
top-k completion); `total_ms` then ends at `acc`. Keep `prefetch_ms` a
separately-quoted column in every l01 comparison — it is the always-rent
baseline any future persistent-experts (keep-stale) arm is judged against.
Since 2026-08-20 (protocol rule 5) the moonep driver plans PER ITERATION
on device — the authentic ported planner kernel + layout derivation,
reported as `plan_ms` (`timing_accounting=per_iter_gpu`). The setup CPU
plan survives as sizing/reference/drift-guard only; `moonep_plan_host_ms`
is that setup reference build, not the timed planner.

Fused-arm brackets (`driver=moonep_fused`): `e2e_ms`, `prefetch_ms` (the
weight ISSUE window), `gate_ms` (issue -> forward launch; 0 by definition
under tokens_first), `fused_ms` (the fused dispatch+GEMM window). Sharded
arms (`--weight_shard`, 2026-08-17) add `shard_ms` (pref_end -> gw_end: the
egress/reassembly/finalize/gateway chain on its dedicated stream — overlaps
`fused_ms` under the tiles gate, serial ahead of `gate_ms` under join) plus
the `wshard_*` cell facts (requested/resolved, leg count, per-rank NIC
egress bytes pre/post shard). `--layers l01` narrows `fused_ms` to the l0
window and appends `act_ms`, `l1_join_ms` (the explicit w2 landing gate),
`l1_ms` (the fused gather-rs combine with INHERITED metadata); `e2e_ms` then
spans through l1. The standalone virtual-space layer1 arms
(`driver=moonep_l1`, `moonep_l1_{hier,compress}`) reuse the gather_rs metric
shape (`e2e_ms` + `iso_sync_ms`, timing_mode axis) with cell facts
`moonep_gpe` / `moonep_E_virt` / `moonep_empty_slots_rank0` /
`l1_index_build_ms`; their combine-copy reduction vs the dispatch matrix is
printed per run (replication makes slot rows combine locally).

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
`ultraep_nvl_domain_size`, `ultraep_plan_host_ms` (setup-time planning —
pre-rule-5 `legacy_untimed_plan` accounting until this driver is next
touched; see protocol rule 5). `ultraep_domain16` treats the whole EP16
group as one domain
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
fallback, smoke only), `eplb_load_sha`, `eplb_plan_host_ms` (the setup
reference build; the timed planner is the per-iteration `plan_ms`,
protocol rule 5), `eplb_transport`. Canonical transport is the
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

- `status` — `ok`, `failed`, `timeout`, `stuck`, `skipped_capability`,
  `skipped_capacity` (exact knob demand exceeds the platform symmetric-heap
  cap; recorded up front, never launched).
- `deterministic` — AND over all ranks' recorded
  `torch.are_deterministic_algorithms_enabled()`. Perf cells must be 0; this
  column is the audit that they were (never assume — the 2026-07-29 root-cause
  showed deterministic scatter_ inflating compress paths ~500x).
- `env_json` — the full env delta the runner constructed (knob scaling,
  variant env, mode env). What you'd need to reproduce by hand.
  **DEFAULT-FLIP BOUNDARY (2026-08-16)**: binaries built on/after this date
  default `FLUX_A2AV_FUSED_STAGE2` and `FLUX_A2AV_EARLY_LAUNCH` ON under
  `FLUX_A2AV_LB_UNION=1` (E only when `CUDA_DEVICE_MAX_CONNECTIONS > 1`).
  An absent key in `env_json` therefore means a DIFFERENT configuration on
  either side of the boundary — never byte-compare `env_json` across it;
  identify the binary by the manifest `flux_libs` sha (protocol rule 4:
  git_sha is not a build identity).
- `wire_ratio`, `relay_ident_bytes`, `relay_balanced_bytes` — compress-only
  cell facts from the pre-run analysis block.
- `router`, `eps`, `placement_sha`, `incidence_remote`,
  `mean_nodes_per_token` — appended 2026-08-18 (nodeaware/LocCap campaign;
  epic-driver arms only, empty elsewhere). `router` ∈ {`d6`, `loccap`};
  `eps` = the LocCap balance slack (caps `(1+eps)*S*K` rows/rank; `inf` =
  pure locality). `incidence_remote` = Σ over tokens of distinct non-home
  serving nodes — under the node-dedup transports this IS the inter-node
  wire rows per direction, the campaign's objective. Placement comes from
  the `<mid>.placement_<mode>_r<red>.json` sidecar
  (`sweeps/predict_placement.py`, PLACE-lambda: pool co-occurrence
  partition + per-node-first coverage replication; `rankconc` = the
  equal-slot concentration ablation). Same contract as `.eplb_load.json`:
  placement INPUT only, matrix identity unchanged, never-overwrite, sha
  recorded here. The sidecar carries the pre-registered simulated
  incidence per (router, eps); the driver hard-asserts the realized
  `route_hash`/`incidence_remote` equal the simulation (same code by
  file-path import — a mismatch is a determinism bug, not noise). The D6
  router is re-derived per iteration on device (rule 5); the
  loccap/evensplit python routers still run once per cell
  (`legacy_untimed_plan` accounting, `epic_loccap_plan_host_ms` fact) and
  are NOT quotable in new-accounting capsules. **The GPU port landed
  2026-08-21 as a NEW arm family (`pll_*`, router `loccap_gpu`, placement
  `placelambda_gpu` — flux.testing.placelambda_gpu):** a bounded-round
  vectorized approximation (greedy node cover + clipped proportional
  fills + bounded repair; integer-deterministic, CPU==GPU bit-identical),
  NOT the exact python loccap — its incidence lands within ~10% of exact
  (offline gate) but its balance can exceed cap where only multi-hop
  augmenting chains could repair (reported in the router stats). Never
  compare `loccap_gpu` cells against exact-`loccap` cells as if the same
  router. pll arms run rule-5 (`per_iter_gpu`, router in `plan_ms`);
  placement accounting per the rule-5 placement amendment
  (`place_dynamic` toggle, `place_ms` metric, `epic_place_*` facts;
  sidecar mode `placement_placelambda_gpu`, batch-observed — the sidecar
  is the offline CPU solve of the same module and the driver hard-asserts
  its on-device solve matches it). eps default 0.0625 = the working
  default from the confirmed flat basin [0, 0.125], NOT canonicalized.
- `correct_bitwise`, `correct_allclose` — AND over ranks; empty when
  `skip_correctness` ran. Note bitwise may legitimately be 0 with determinism
  off; allclose is the correctness verdict.
- `matrix_sha256`, `git_sha`, `git_dirty` — provenance; treat `git_dirty=1`
  results as unreproducible.
- `layer` (appended 2026-08-16) — `l0` (dispatch, the historical default;
  older capsules simply lack the column), `l1` (gather-rs combine,
  driver=gather_rs / fast_gather_rs), `l01` (combined continuous pass,
  drivers l01 + l01_fast; RULE-5 CONVERTED 2026-08-21,
  timing_accounting=per_iter_gpu: one timed window per iteration = routing
  allgather (`plan_comm_ms`) -> ALL routing-derived metadata for BOTH
  layers on GPU (`plan_ms`: the l0 op's fused derive_routed_meta seeding
  the vectorized _dev builders for the l1 pack/reduce indices + compress
  CSRs; the CPU builders survive as untimed drift guards, their wall
  reported as `l1_index_build_ms`) -> layer0 forward (stage2 scheduling
  still in-window) -> GELU -> layer1 forward. `total_ms = plan_comm +
  plan + e2e`; e2e/l0/act/l1 anchors unchanged vs pre-conversion l01
  capsules — never compare planning-inclusive totals across the boundary.
  2026-08-21 canonicalizations on this lane: the compress-arm plan uses the
  SORT-FREE scd-arithmetic kernels (a2av_compress_plan; bitwise-identical
  to the sort formulation, FLUX_A2AV_RS_CHECK_IDENTITY cross-checks both);
  l01_allgather_dense carries FLUX_RS_BLOCKS=20 (the dense combine was
  CTA-starved at topk-16: K3 b32 e2e 104 -> 60 ms over the {3..24} knob
  sweep, knee ~20) — env flip = never-byte-compare boundary. The a2av
  combine was starved the SAME way: 2026-08-22 both a2av l01 arms carry
  FLUX_A2AV_RS_{PACK,REDUCE,PRERED}_BLOCKS = 6/6/4 (K3 b32 compress e2e
  61.9 -> 51.0; 12/12/8 regresses — margin starves the GEMMs). The
  compress plan is additionally fully sort-free since 2026-08-22
  (reduce_index by the same scd arithmetic).
  Driver l01_fast (--impl fast, 2026-08-21): the FAST+FAST combined
  unfused baseline on REAL routing — dispatch alltoallv -> grouped GEMM0
  -> GELU -> grouped GEMM1 -> combine alltoallv (transposed matrix) ->
  home topk-reduce; TWO flash_comm_t instances (one per direction,
  vendored patch scripts/fast_two_instance.patch) so both credit resets
  stay OUTSIDE the window (`reset_ms`); e2e-only, >= 2 nodes; extra
  metrics gemm2/cpack/comb_*/acc mirror the moonep l01 chain. No
  timing_mode axis, no phases cells. **Reference combined configuration
  since 2026-08-16: `l01_lbunion_compress`** — see variants.py for the
  14/14-budget verdict; the standalone-l1 verdict differs at small budgets
  and must never be conflated with the combined one).
  Lives on the VARIANT (sweeps/variants.py `layer` field), not as a spec
  axis — an l1 measurement is a different arm, not a different mode of the
  same arm.
- `timing_mode` (appended 2026-08-16) — l1 flux cells only: `isolated` (the
  op builds pack/reduce indices + compress CSRs in-forward, on the timed
  path — what layer1-alone must pay) or `amortized` (the harness precomputes
  everything a fused layer0+layer1 pipeline would inherit from layer0's
  in-window stage2 and passes it in untimed — the combined-pass proxy).
  Cell_id carries `_tmiso`/`_tmamo`. **Never compare across timing_mode**:
  they measure different regimes by construction. Empty for l0 cells and
  for `l1_fast` (its index metadata is setup-time — pre-rule-5
  `legacy_untimed_plan` accounting; the BvN schedule recompute stays
  in-window per the one-shot rule).
- `timing_accounting` (appended 2026-08-20, protocol rule 5) —
  `per_iter_gpu`: every batch-derived plan quantity is recomputed per
  iteration on device inside the timed bracket (a `plan_ms` metric column
  reports the planner's share; quote plan_ms/total_ms per arm).
  `legacy_untimed_plan`: setup-time planning (all pre-2026-08-20 capsules,
  which simply lack the column, plus arms the per-iteration path does not
  cover yet — epic m>1, loccap routers, moonep_fused, ultraep,
  standalone l1/gather_rs drivers). The layer0 **flux and fast drivers
  were converted 2026-08-21** (routing allgather as plan_comm +
  derive_routed_meta / derive_fast_meta_gpu as plan; see the metrics
  note), and the **combined l01 + l01_fast drivers on the same day** (see
  the layer=l01 note) — their pre-boundary capsules are legacy. **Never
  compare planning-inclusive totals across the two accountings** — same
  never-mix logic as protocol rule 4.
- `torch_ref_impl` (appended 2026-08-21) — `local_slice_scatter`: the torch
  unfused reference scatters ONLY this rank's EP slice ([nrows_ep, H]
  staging) instead of the global [ntokens*topk, H]; the gemm loop never
  read the other W-1 slices, so the old global materialization was W-fold
  extraneous memory (7 GiB at K3 b28 — the large-budget OOM class) AND
  W-fold extraneous scatter work inside `scatter_ms`. Column absent =
  pre-fix global-scatter reference; **`impl=torch` rows never compare
  across the flip** (comm_ms/gemm_ms are unaffected; scatter_ms and totals
  drop by design).
- `planner_impl` (appended 2026-08-20, campaign 2) — `fused_dispatch`:
  planning fused into the dispatch op (FusedEpDispatch for eplb, the
  `dispatch_only_routed` in-window derive for epic hc); `torch_gpu`: the
  campaign-1 per-iteration torch-op planner (retired from specs, kept for
  history); `legacy`: setup-time planning. Reused arm names across a
  planner_impl change are a rule-4 boundary (see the capability tags note).
- `replica_select` (appended 2026-08-20, campaign 2) — replica-choice rule
  for redundant experts, sender-local in both modes: `local_spread`
  (DEFAULT; per-source largest-remainder prefix ≡ round-robin/equal token
  split, the SGLang dynamic-dispatch load distribution), `local_static`
  (src mod C, the EPIC D6 rule / SGLang static-map class), `quota` (the
  retired staged-arm rule; needed a pre-dispatch loads allgather).
- `plan_comm_bytes` (appended 2026-08-20, campaign 2) — payload of the
  pre-dispatch planning collective, from the driver facts. 0 on
  sender-local eplb fused arms (no such collective exists); W*G*4 on epic
  (the per-iteration loads allgather is part of EPIC's algorithm and always
  timed) and on the retired quota arms.
- `shape`, `ffn_hidden` (appended 2026-08-20, qwen3 validation lane) —
  `shape` is the `SHAPE_PRESETS` key the spec used (`k3` = Kimi-K3 canon
  G896/k16/H3584/ffn3072/chunk7168 on the synthesized pools; `qwen3` =
  authentic Qwen3-235B-A22B G128/k8/H4096/**ffn1536**/chunk8192 on captured
  Qwen3 routing — the fast 4n lane, since K3 topk-16 incidence saturates at
  4 nodes); empty = raw-field spec. The runner (`apply_shape`) hard-errors
  when a preset spec explicitly contradicts a pinned field or names a trace
  family whose model doesn't match the preset (authentic capture vs
  synthesized pools is part of the shape). **ffn correction, a rule-4-style
  never-mix boundary:** `ffn_hidden` was previously unrecorded, and every
  pre-2026-08-20 Qwen3-shape capsule ran ffn_hidden=4096 — wrong for Qwen3
  (per-expert `moe_intermediate_size` is 1536; ~2.7x per-expert GEMM
  inflation). Capsules lacking the `ffn_hidden` column therefore carry
  invalid absolute GEMM-inclusive numbers for the Qwen3 shape; never
  compare them against ffn_hidden=1536 cells. Comm-only/wire facts
  (bytes, incidence, dedup ratios) are ffn-independent and remain valid.
- `place_dynamic`, `place_solver_ms` (appended 2026-08-21, pll_* arms) —
  the PLACE-lambda placement-ablation toggle and the untimed setup
  solve's measured latency (see the rule-5 placement amendment). Empty /
  0 on non-pll arms. The timed dynamic-lane quantity is the `place_ms`
  metric in metrics.csv; per-cell trigger facts (`epic_place_lb_cur`,
  `lb_new`, `gain_ppm`, `moves_add/remove`, `trigger`) live in the
  records artifacts.

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

**fast e2e ≡ isolated (documented equivalence, 2026-08-16).** The fast
drivers (layer0 `test_moe_ag_fast_baseline.py`, layer1
`test_moe_gather_rs_fast_baseline.py`) execute an explicit
`torch.cuda.synchronize()` before every iteration's window
(test_moe_ag_fast_baseline.py:194) followed by the HOST-BLOCKING
`comm.alltoallv` (:200-201), and the tail NVSHMEM barrier doubles as the
per-iteration credit reset — no cross-iteration pipelining is physically
possible, so a fast `e2e` cell already measures one isolated execution per
iteration. Fast `e2e_ms` may therefore be compared against other arms'
`isolated` cells (this equivalence, not a mode match, is the license); the
general isolated-vs-e2e never-mix rule is unchanged for every other arm.

Layer1 (`driver=gather_rs`) cells never expand in `phases` mode — the
gather-rs op has no `FLUX_A2AV_TIMING` stderr marks; the runner prints a
NOTE and skips, mirroring the EP arms. l1 flux cells instead carry the
`timing_mode` axis (see cells.csv) — isolated-vs-amortized within one arm is
the schedule-inheritance decomposition, obtained without instrumentation.

## Comm/comp attribution

Phase wall-times from `phases` mode inherently conflate overlap (comm hidden
under the GEMM tile-spins shows up as gemm time, not comm time). The honest
decomposition protocol when needed: pair an `e2e` cell with a comm-only or
comp-only ablation and difference the results in the summarizer — never store
derived "exposed comm" as raw truth. Timeline modes are the ground truth for
"was it actually overlapped" — nsys first (sees CE memcpys, proxy threads,
and cross-process node timelines that the torch profiler cannot), torchprof
when Python-op attribution is needed.

## Knob scaling (exact per-cell computation, 2026-08-15)

`exact_scale_knobs` in sweep.py: each cell's `FLUX_A2AV_MAX_{RECV,STAGE,RELAY}_NTOKENS`
is computed from the SAME expressions the runtime FLUX_CHECKs, evaluated on the cell's
on-disk matrix (+ routing file when `routing_mode=real`; dealer closed form otherwise)
by `gen_matrix.a2av_knob_demands`. Per knob:

- `RECV = max(copies column max, union-bcast dedup column max)` — covers the
  non-compress recv gate and the lb_union region layout.
- `STAGE = max(hier copies staging, identity-relay U staging, balanced-relay chunks)`;
  `RELAY = balanced-relay chunks`. These are far below RECV (~19-40% at the
  scales measured), so they are sized independently.
- Each demand is rounded up to 8192 rows and floored at 163840: every cell whose
  demands fit the old floor keeps a **byte-identical** `env_json`.
- `NVSHMEM_SYMMETRIC_SIZE` = component sum of the ctor's symmetric buffers
  (2 send halves + RECV + STAGE + RELAY, or the dense W*T gathered input,
  whichever is larger) + 1G overhead, floor 6G, capped by platform
  `sym_size_max_g`. If the *uncapped* requirement exceeds the cap the runner
  records the cell as `skipped_capacity` instead of launching it.

Parity with the torch reference (`flux.testing.moonep_fused_map.required_a2av_knobs`)
and the dealer closed form are unit-tested in `sweeps/test_knob_demands.py`.

History: the previous anchor formula (`max(163840, 4*row_chunks)` uniform across the
three knobs; anchors from topk16 sweeps 2026-07-29) had no world-size term, while the
lb_union recv demand grows ~W*T. It under-provisioned exactly at b64/W32 (271685 needed
vs 262144, one-rank FLUX_CHECK -> apparent hang) and b32+b64/W64 (259218/518504 vs
163840/262144, collective fast-fail), 2026-08-15. Capsules recorded before this change
carry the old env values — knobs are capacity-only, but do not byte-diff `env_json`
across the policy boundary.

Under-sizing still fails loudly (FLUX_CHECK recv-overflow / NVSHMEM init); as of
2026-08-15 the recv-overflow checks are collective (same expression on every rank), so
a capacity failure aborts all ranks instead of leaving the fleet spinning for the
watchdog.

### Layer1 (gather-rs) knobs — `exact_rs_scale_knobs` (2026-08-16)

l1 flux cells get `FLUX_A2AV_RS_MAX_{SEND,STAGE,CONV,WIRE}_ROWS` +
`NVSHMEM_SYMMETRIC_SIZE` from `gen_matrix.a2av_rs_knob_demands`, replicating
the gather-rs op's collective FLUX_CHECKs (gemm_grouped_v2_gather_rs.cc
:562-571 send, :600-615 stage, :687-710 conv/wire). Inputs stay in DISPATCH
orientation — the layer1 wire is the matrix transpose, and the transposition
lives inside the demand function (the C++ `chunk_at(s, d)` == dispatch
`chunks[d][s]`):

- `SEND = max dispatch column sum` (each owner's outbound rows — numerically
  identical to the layer0 recv_copies bound).
- `STAGE` (non-compress): per-(gateway node, lane) staging of remote-homed
  copies. `CONV`/`WIRE` (compress): per-(owner node, dest lane) convergence
  rows and U-deduped wire partials.
- The recv panel is knob-free (`max_m / world_size` exact, cc :233).
- Rounded up to 8192 rows, NO legacy floor (new axis — no historical
  env_json to keep byte-identical). Heap = send + recv(cpr) +
  (stage | conv+wire) rows x chunk_bytes + 1G, floor 6G, same
  `skipped_capacity` contract. `dense` cells reuse the conservative
  hier-shaped bound (its ring/staging buffers are not audited here —
  verify at the 2n bring-up smoke).
- All four RS knobs are always exported; the op reads only the panels its
  branch allocates. All RS checks are COLLECTIVE — an undersized l1 cell
  aborts everywhere, it never hangs (unlike the historical layer0 per-rank
  recv gate).
- Parity: `test_rs_demands_brute` in `sweeps/test_knob_demands.py`
  brute-forces the C++ loops in wire orientation against the
  dispatch-orientation implementation (plus the manual 2n demand-minus-one
  GPU probe documented in that test's docstring).

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
   noise. See `docs/handoff/04_build_ledger.md`. The 2026-08-20 campaign-2
   binaries carry two new capability tag strings, grep-able from the
   installed `.so` bytes: `FLUX_FUSED_EP_DISPATCH_TAG` (the FusedEpDispatch
   op replacing EPLB's staged wire / EPIC's direct wire) and
   `FLUX_A2AV_INWINDOW_META_TAG` (`dispatch_only_routed` in-window hc
   metadata). They are a rule-4 boundary: arm names reused across the flip
   (`epic_hc_m1_mig*`) measure different code — never compare a cell from a
   binary with a tag against one without it.
5. **Timing accounting: gating metadata is the ONLY untimed work
   (user directive, 2026-08-20).** The EXCLUSIVE untimed exemption is the
   one initial step where the harness provides/exchanges the gating
   metadata (the routing, i.e. the gate's output) so every rank can compute
   identical plans — it stands in for the model's own gate, which is not
   the system under test. **Every calculation downstream of gating must be
   inside the timed region, per iteration, no exceptions**: plan
   derivation, replication/balancing decisions, dedup and slot maps, split
   transposes, combine-plan construction, pack metadata. Nothing derived
   from gating may be cached across iterations or precomputed at setup:
   static per-cell routing is a harness convenience for re-timing one MoE
   layer — in production the gating changes every iteration, so a plan
   computed once at setup is an illegal amortization. (The
   "`plan_host_ms` untimed-metadata contract" in pre-2026-08-20 drivers is
   exactly this mistake; totals from those drivers under-charge planning.
   Driver docstrings are corrected as each driver is touched.) The
   recurring wire cost of making routing globally known (`plan_comm`) is
   and stays timed. Implementation corollary: raw python planning must not
   be what gets timed — it overcharges by orders of magnitude vs
   production implementations (MoonEP upstream fuses its entire per-step
   planner into ONE cooperative GPU kernel); plan derivation is
   implemented on-GPU, fused/grouped as far as possible, and THAT is
   timed. Baselines with an authentic upstream planner (MoonEP) time the
   authentic planning cost un-separated if upstream fuses it. Capsules
   produced under the old accounting must never be compared against
   new-accounting capsules on any total that contains planning (same
   never-mix logic as rule 4; the boundary is the driver change, cite it
   when quoting). Fused-canonical arms (`planner_impl=fused_dispatch`,
   2026-08-20) satisfy this rule structurally: there is no planner to
   mis-time — planning is part of the dispatch launch and lands in
   `comm_ms` (see the Planner v2 paragraph under `plan_ms`). The layer0
   comm drivers (flux: all a2av arms + stock Comet + the torch reference;
   fast) were converted 2026-08-21 — timed per-iteration routing allgather
   (`plan_comm_ms`) + fused/vectorized on-GPU derivation (`plan_ms`),
   setup metadata demoted to a bitwise drift guard; the flux test's
   `--no_metadata_cnt` flag is ABLATION ONLY under the new accounting
   (derive still runs and is timed; the flag only withholds cnt from
   forward) and never appears in campaign specs.

   **Placement amendment (user directive, 2026-08-21 — PLACE-lambda
   arms).** EXPERT PLACEMENT (which experts' weights reside where) is a
   different timescale than batch planning: re-placement implies weight
   movement that production amortizes across iterations. Its accounting
   is therefore an EXPLICIT ABLATION TOGGLE, never an implicit
   amortization:
   - `place_dynamic=static` (the ideal-stale arm): the placement is an
     input; one setup solve runs untimed on device and is REPORTED as the
     `epic_place_solver_ms` fact (visible, uncharged). Nothing
     placement-related runs in-window.
   - `place_dynamic=dynamic` (placement is part of the optimization under
     test): the per-iteration solve + instance move-diff + trigger
     threshold run INSIDE the timed bracket, reported as the `place_ms`
     metric (a zero-width event on static arms). Weight dispatch itself is
     not yet modeled (queued for the fusion pass); the timed quantity is
     the full decision apparatus.
   The ROUTER lane has no toggle: per-token replica selection
   (`loccap_gpu`) is batch-plan work and is ALWAYS timed per iteration
   in-window (`plan_ms`), like every rule-5 planner. The dynamic-arm
   trigger facts are `epic_place_lb_cur/lb_new/gain_ppm/moves_add/`
   `moves_remove/trigger` (identical across iterations under static
   per-cell routing — asserted, recorded once).
