# 09 — Comm-only drivers: Tier-1/Tier-2 timing restructure (queued campaign)

> **STATUS 2026-08-20**: QUEUED, not started. Written at the close of the
> campaign-2 fused-canonical planner work (8.20.expert-sweep session) so a
> session WITHOUT that context can run this as its own campaign. The
> expert-movement baselines (eplb/epic/moonep) are DONE and rule-5-clean —
> this queue covers the remaining **comm-only** drivers (FAST, comet/flux
> a2av, hier/lb_union arms) whose routing metadata is still built at setup
> time (`timing_accounting=legacy_untimed_plan`). LocCap is explicitly
> deferred until its GPU router port lands (separate effort, parallel
> session).

Read FIRST: `sweeps/SCHEMA.md` protocol rule 5 (the user's hard directive
defining the tiers, 2026-08-20) and the `plan_ms` + "Planner v2" paragraphs
there; then this doc. The reusable implementation templates live in the
campaign-2 code (see §3). Environment: `source ./module.sh`; account
m5350_g; salloc+srun only, never sbatch; scancel the moment jobs finish;
NERSC cudatoolkit module drifted to 12.9 mid-2026-08-20 — pin the 24.5/12.4
toolchain per the `perlmutter-cudatoolkit-drift` memory before ANY rebuild.

## The two tiers (rule 5 restated operationally)

- **Tier 1 — gating metadata (the ONLY untimed thing)**: the initial step
  where the harness provides/exchanges the gate's output (the routing:
  `choosed_experts` / topk ids) so every rank holds identical replicated
  routing. It stands in for the model's gate, which is not the system under
  test. Every baseline receives the SAME Tier-1 data, so exempting it is
  comparison-neutral.
- **Tier 2 — everything derived from it (ALL timed, per iteration)**:
  splits, scatter/sort indices, splits_per_source, dedup/unique counts,
  pack/reduce/combine indices, CSR wires, capacity-independent layout
  tables. No setup caching, no cross-iteration reuse — production gating
  changes every step. Allocation SIZING may stay setup-time (one-shot
  deployment scope); tensor CONTENTS may not.

Litmus test: "would this tensor's VALUES change if the gate emitted a
different batch?" → Tier 2, timed.

## 1. Inventory — where each comm-only driver builds Tier-2 data at setup

1. **flux a2av arms (comet layer0; `test_moe_ag_traffic.py`)**
   - `MoeMlp1Ctx` takes `gating.scatter_index` at setup
     (`python/flux/testing/moe_ag_scatter_utils.py:107`) and the driver
     passes splits/scatter_index into forward untimed.
   - `splits_per_source_cpu` + compress dedup counts built at setup
     (`test_moe_ag_traffic.py:495-502`), passed as "untimed setup"
     (`:226-230`). Note the existing `--no_metadata_cnt` flag (`:417`)
     already re-derives SOME metadata in-window — it is a partial
     precedent, not the fix.
   - Fix: route the flux arms through **`dispatch_only_routed` /
     `derive_routed_meta`** (the v2b entry,
     `src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc`, pybind
     `dispatch_only_routed`): pass the Tier-1 topk ids, let the op derive
     splits/stable-scatter/sps/uc on device inside the timed bracket. For
     the fused (dispatch+GEMM) forward path, add a sibling `forward_routed`
     with the same `ensure_routed_meta` prologue.
2. **hier / hier_compress / lb_union arms (same test file)**: same fix —
   they are `dispatch_only`/forward consumers of the same metadata bundle.
   The layer1 (gather_rs) side already self-builds ALL combine indices
   in-window when not passed (campaign-2: `run`'s a2av_hier branch and the
   gather_rs entry both call `build_a2av_combine_indices` /
   `build_a2av_compress_indices` on the timed critical path —
   `src/moe_gather_rs/ths_op/gemm_grouped_v2_gather_rs.cc`, free functions
   near the top of the file). For l01 arms, simply STOP passing the
   python-built pack/red/wire/redcsr and let the op build (the epic
   `hc_meta=inwindow` wiring in
   `python/flux/testing/epic_semantics.py::dispatch_group_hc` is the
   worked example, including the uc `[W, W+NN] -> [:, W:]` slice gotcha).
3. **FAST baseline (`test_moe_ag_fast_baseline.py`)**: host index math
   marked "untimed metadata" at `:356` — Tier 2 by the litmus test. Port
   the index build to device (the `a2av_meta_counts_impl` /
   `a2av_stable_scatter_index_impl` kernels in
   `src/moe_ag_scatter/sort_util.cu` are the primitives) or time the
   existing build per iteration as an honest interim (facts must then say
   `planner_impl=torch_gpu`, and the overcharge caveat from rule 5's
   "implementation corollary" applies). The BvN schedule recompute is
   already in-window (one-shot rule) — keep it.
4. **LocCap router arms**: DEFERRED. Still a once-per-cell python port —
   `timing_accounting=legacy_untimed_plan`, not quotable in new-accounting
   capsules until the GPU router port lands (tracked by the parallel
   LocCap session). Do not block this campaign on it.

## 2. Accounting contract for the restructured arms

- The in-window derive lands INSIDE the existing timed bracket of the
  phase that consumes it (dispatch bracket for layer0 metadata, combine
  bracket for layer1 indices) — same placement as epic v2b. No new metric
  columns needed; `plan_ms` stays a driver concept for arms that have a
  distinct planning phase.
- Facts: set `timing_accounting=per_iter_gpu` + `planner_impl` per arm
  once converted; capsule notes must quote the before/after totals as a
  narrative ONLY (rule-4/rule-5 never-mix: legacy capsules never compare
  against new ones on planning-inclusive totals).
- SCHEMA edits in the SAME commit as each driver conversion: extend the
  `timing_accounting` fact doc's "arms the per-iteration path does not
  cover yet" list (remove converted arms), and note the driver boundary
  with a dated comment in `sweeps/variants.py`.
- A converted arm on the same binary is still a NEW rule-4 comparison
  boundary if the op entry changed (new capability tag if C++ changed —
  follow the `FLUX_A2AV_INWINDOW_META_TAG` pattern, greppable string in
  the installed `.so`).

## 3. Reusable templates from campaign 2 (all landed, all validated 4n)

- **In-window metadata derive**: `derive_routed_meta` /
  `ensure_routed_meta` / `dispatch_only_routed`
  (`gemm_grouped_v2_ag_scatter.{h,cc}`; kernels in `sort_util.cu` — the
  stable counting-sort scatter index is bitwise-equal to python
  `argsort(stable).argsort()`; the old `calc_scatter_index` is
  NON-deterministic and FORBIDDEN for replicated data).
- **In-op combine-index self-build**: absent pack/red/wire/redcsr →
  the op builds them on the timed path (`gemm_grouped_v2_gather_rs.cc`,
  both the gather_rs entry and TopkReduceScatterOp::run).
- **Standalone bitwise test**: `test/python/moe_ag_scatter/`
  `test_a2av_inwindow_meta.py` (4 adversarial routings × run-twice
  determinism vs the python reference) — clone it for any new derive.
- **NVSHMEM multi-node gotcha (2026-08-20, cost a debugging session)**:
  on the proxied transports (libfabric/CXI at nnodes>1) EVERY put SOURCE
  must live on the symmetric heap — an ordinary CUDA tensor as source
  works over NVLink P2P and SEGFAULTS THE HOST PROXY THREAD
  (`nvshmemt_libfabric_rma`, NULL mr deref at offset 0x30) at 2n+. Stage
  through a symmetric buffer (see `comb_stage_sym_` in
  `src/coll/ths_op/fused_ep_dispatch.cc`). The 2-form ordering probe
  (`probe_fused_ep.py`) does NOT catch this class — its sources are
  already symmetric; smoke at 2n before believing any new put path.

## 4. Validation ladder

1. CPU: pytest the new derive vs python reference (clone
   `test_a2av_inwindow_meta.py`).
2. 1n 4-GPU: driver smoke correctness ON, run-twice bitwise; then the
   skewed matrix (`$PSCRATCH/a2av_test_matrices/matrix_4r_skewed.txt`).
3. 2n then 4n: the SAME smokes (the proxy-segv class above only exists at
   nnodes>1; 2n uniform matrix `uniform_2n_8r.txt`, 4n
   `4n_16r/*_dist_001.txt`).
4. Capsule: paired legacy-vs-converted arms in ONE capsule on ONE binary
   for the narrative; converted-only arms thereafter.

## 5. Changed columns / facts summary

| fact | value on converted arms |
|---|---|
| `timing_accounting` | `per_iter_gpu` |
| `planner_impl` | `fused_dispatch` (in-op derive) or `torch_gpu` (interim timed torch) |
| `plan_comm_bytes` | 0 (comm-only arms have no planning collective) |
| capability tag | new tag string if C++ entries changed |
