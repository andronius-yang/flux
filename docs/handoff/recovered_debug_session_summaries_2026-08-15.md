

# ===== MAIN SESSION =====


## [2026-08-15T14:20:40.606Z]

Worktree `debug` is created (branch `worktree-debug`, based on `realistic-moe-input` @ `4678551`, which already contains the sweep-campaign commits including the `19ce9f9` `FLUX_DEBUG_DUMP_DIR` fix) and the session is now working inside it. As requested, I've done nothing else yet — here's my read of the debug handoff (`docs/handoff/06_realistic_moe_sweep_debug_todo.md`) and how I'd attack it.

**The four queued items, as I understand them:**

1. **`moonep_fused_push_auto_gated` crashes at every budget at 16n/W64** — a real crash (fails even with `--skip-correctness`), fast (~11s), so there should be a clean traceback. Zero latency data exists for the headline MoonEP arm at scale. The handoff explicitly says the raw stderr from the failed 16n cells was *never inspected* — so my first move is to read the existing raw rank logs at the Perlmutter platform data root from capsule `20260815-105733_perlmutter_244825e3`. **That needs no GPU allocation at all.**

2. **`allgather` + `moonep_fused_push_auto_gated` fail `torch_allclose` deterministically at 16n** — arm-specific (three other arms pass the same check on the same matrix instance), so a genuine W=64 correctness bug, and it taints the already-reported 16n latency numbers for those two arms. The fix path is a single-budget rerun with correctness ON and actually diffing the dumped `x`/`y`/`moe_ctx` tensors (dumps now go to `FLUX_DEBUG_DUMP_DIR`, which I'll point at scratch).

3. **`hier_compress_lb_union`: hangs at b64/W32 (8n), fails fast at b32+b64/W64 (16n)** — flagship arm, scale-dependent. The handoff's own instruction: before chasing a logic bug, check the `scale_knobs()` / `FLUX_A2AV_MAX_RECV_NTOKENS` formula (anchored at L=8, predicted to under-provision at higher W). That's a static code check — also **no allocation needed** for the first step.

4. **`fast` fails at b64 at 4n/8n** — documentation-only, known-fragile arm; I'll leave it unless you say otherwise.

**Proposed sequence:** (a) mine the existing rank logs for #1's traceback and #2's failure fingerprints — free; (b) statically audit `scale_knobs()` against W=32/64 for #3 — free; (c) only then take **one** targeted allocation to recreate the cheapest reproducible failure with heavy logging into scratch (`/tmp/claude-…/scratchpad` per your instruction), and `scancel` immediately after. No sweeps — direct `srun ./launch.sh` cells cloned from the capsules' `spec.yaml`.

**Two things I want your input on before allocating:**

- **Scale/QOS cost:** #1 and #2 live at 16n and #3's hang at 8n — both exceed the 4-node interactive QOS cap, so recreating them needs `-q regular` on **m5350_g** (m4243_g is exhausted). Are you OK with a short regular-QOS 8-node (and possibly 16-node) allocation, or should I first see how far I can get from logs + a 4n-max interactive probe (e.g. checking whether #2's mismatch already shows at W=16 with the right knobs)?
- **Priority confirmation:** handoff says #1, #2, then #3. Fine to follow that, or do you want the flagship `hier_compress_lb_union` breakdown (#3) bumped up?

My default if you just say "go": log mining + static knob audit first, then a 4-node interactive probe for anything that reproduces there, escalating to `-q regular` 8n/16n only for failures that genuinely require the scale.


## [2026-08-15T14:25:45.136Z]

I'll explore the failure evidence and repro machinery first, then draft the plan. Launching three parallel Explore agents: one to mine the existing capsules/raw logs for the actual failure fingerprints, one to audit the `scale_knobs()` formula, and one to map the single-cell repro machinery (variants, knobs, debug dumps).


## [2026-08-15T14:32:17.172Z]

Allocation strategy locked in: both regular-QOS allocations at 45 minutes (so every cell gets pre-scripted before landing — the windows are pure execution), 16n serviced first on overlap, plus a 4n interactive probe. The plan draft is updated.

The three exploration agents are still mining the failed-cell logs, auditing the `scale_knobs()` arithmetic, and mapping the manual repro machinery. I'll fold their findings into the per-bug debug procedures (Phase C) as they report back, then present the finalized plan for your approval.


## [2026-08-15T14:33:58.400Z]

Folded in. Key updates to the plan: item #1 is a hard `W <= 32` assert (single-warp gating ballot) in the moonep_fused driver — a designed precondition, not a compute bug — and the deterministic 16n allclose failure is actually `allgather`-only; the 8n lb_union "hang" is the predicted `FLUX_A2AV_MAX_RECV_NTOKENS` under-provisioning (271685 needed vs 262144 provisioned, one rank aborts and 31 spin). Waiting on the remaining two agents (raw-log excerpts for the 16n fast-fail and `fast` b64, and the `scale_knobs()` arithmetic audit) before finalizing the plan.


## [2026-08-15T14:41:45.571Z]

Both Plan agents are running. They'll design the `scale_knobs()`/check-site/routing-anomaly fixes and scope the moonep_fused W≤32 constraint plus the ranked hypothesis list for the allgather W=64 bug. I'll fold their designs into the plan file when they report, then present the final plan for approval.


## [2026-08-15T14:51:48.141Z]

The kernel-side scoping just collapsed the two headline bugs into one root cause: the single-warp source-segment gating ballot in `process_tile` — at W=64, segments 32–63 are never waited on, so allgather tiles read zeroed unarrived rows (the exact-zero bands), and moonep_fused's `W <= 32` assert exists precisely to avoid this cliff. One ~50-line kernel fix addresses both. And H1 can be confirmed GPU-free by predicting the mismatch bands from the routing files. Let me now rewrite the plan file into its final form.


## [2026-08-15T15:22:48.941Z]

Status while the build runs — a lot has landed this session:

**Root causes (all four handoff items resolved on paper, two GPU-proven so far):**
1. **One kernel bug explained both 16n headline failures.** The `process_tile` source-segment gate used a single-warp 32-lane ballot; at W=64 any tile whose rows extend past `split_accum[31]` got an empty ballot and **waited for nothing** — even its sources <32. I confirmed this GPU-free by mapping the recorded mismatch rows to source ranks via the routing files: the zero rows are exactly the last-arriving fetches (node 1 for rank 11, the fetch frontier for rank 3). moonep_fused's `assert W<=32` guarded the same cliff — its "crash" was the designed precondition, and its allclose "failure" was a mischaracterization (it never reached the checker; only `allgather` genuinely failed).
2. **lb_union's 8n hang and 16n fast-fail are one provisioning bug** — `scale_knobs()` had no W term. My new exact computation reproduces all three failure-log demands digit-for-digit (271685/259218/518504). **GPU-proven at 8n**: the formerly-hanging b64/W32 cell ran green with the raised RECV knob, in both dealer and real-routing forms.
3. **Bonus find: the sweep retry path silently dropped `routing_mode`** — the failing cells actually ran dealer routing mislabeled as real Qwen3 trace. Any capsule with `family=trace AND routing_mode=''` is tainted.
4. `fast` b64 @4n is an NVSHMEM heap-ceiling init failure (my rerun hung silently in init); the "8n b64" claim doesn't exist in the record.

**Fixes committed on `worktree-debug`** (4 commits): W≤64 segment gating in the kernel + host guard; collective/unit-correct capacity checks (kills the one-rank-abort→31-spin "hang" mode); exact per-cell knobs + `skipped_capacity` + retry fix + unit tests (all passing, incl. torch parity); SCHEMA.md + handoff 06 + memory updated.

**In flight:** full worktree rebuild (~30% done); 16n allocation still queued (its 45-min verification script is pre-staged, with a no-rebuild fallback list). 8n and 4n allocations already used and released.


## [2026-08-15T15:35:36.395Z]

**W16 regression gate is green on the rebuilt binary**: all 16 ranks pass `torch_allclose` for both `hier_compress_lb_union` and `allgather` (correctness ON), and moonep_fused reports bitwise+allclose OK — the generalized segment gate is behavior-neutral at W≤32, as designed. The 4n allocation is released.

Everything is now staged for the final step: the 16n window (still queued, `dbg16n`, 45-min request). When it lands, the pre-staged script runs, in order: lb_union b32/b64 with exact knobs (closing #3 at 16n), allgather b1/b8/b16 correctness ON (closing #2), moonep_fused b8 correctness ON at W64 (closing #1), plus a compress-path W64 regression cell — then I release the allocation.


# ===== SUBAGENT agent-a413be646b1dd4c6a.jsonl =====


## [2026-08-15T14:51:12.332Z]

The Write tool is not available in this read-only session, so I will deliver the full plan directly as my final output.

---

# Findings & Plan

The two tasks turn out to share a single root: **the only W<=32 dependence anywhere in the V2 AG-Scatter family is the per-tile source-segment gating in `process_tile`** of the sm80 grouped-GEMM kernel. It is the "fused gating ballot" the moonep test assert names, and it is (with near-certainty) the cause of the W=64 allgather zero-band failures — the allgather path simply has no guard, so it silently runs off the 32-lane cliff.

## TASK 1 — moonep_fused W<=32 constraint

### (a) The actual single-warp dependence

`/global/u1/y/yufeid/workspace/changchen/andrewy/flux/.claude/worktrees/debug/src/moe_ag_scatter/cutlass_impls/ag_scatter_gemm_grouped_with_absmax.h:592-613`:

```c++
int lane_idx = threadIdx.x % 32;
...
int * split_accum = params.split_tp_accum_ptr + params.world_size * (problem_idx % params.nexperts_ep);
int segment_start =
    __ffs(__ballot_sync(0xffffffff, lane_idx < params.world_size ? (m_start < split_accum[lane_idx]) : false)) - 1;
int segment_end =
    __ffs(__ballot_sync(0xffffffff, lane_idx < params.world_size ? (m_end < split_accum[lane_idx]) : false)) - 1;
...
int seg_rows_acc = lane_idx < params.world_size ? split_accum[lane_idx] : 0;
int seg_prev_acc = __shfl_up_sync(0xffffffff, seg_rows_acc, 1);
bool lane_nonempty = seg_rows_acc > (lane_idx == 0 ? 0 : seg_prev_acc);
if (lane_idx >= segment_start && lane_idx <= segment_end && lane_nonempty) {
  // lines 614-638: lane spins on signal_ptr[lane_idx] (a2av) or barrier_ptr[lane_idx] (dense)
```

The design is **one warp lane per source segment**: `split_accum` is the per-expert inclusive cumsum over `world_size` source lanes (`accum_per_rank_ptr`, set at `src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc:3332-3335`; in lb_union it is the window-keyed `a2av_gating_cumsum_`, still shape [E, W] per `.cc:427-432`). With `lane_idx` in [0,32), segments 32-63 can never appear in the ballot, and no lane ever spins on `signal_ptr[32..63]` / `barrier_ptr[32..63]`. Worse, when a tile's m-range lies wholly beyond `split_accum[31]`, both ballots are 0, `segment_start = segment_end = -1`, and the tile waits on **nothing at all**.

There is no rank-bitmask data structure problem elsewhere — everything adjacent is already W<=64-clean:
- Tile claimer masks are uint64: `a2av_tile_claimer.hpp:39,74,104-105` (`uint64_t(1) << s`, `need & ~arrived`).
- Bucket build masks are uint64: `workspace_util.cu:224-236` (`uint64_t(1) << sid`, `63 - __clzll(mask)`).
- Dense schedule builder is loop-based, not lane-based: `workspace_util.cu:35-115` (`calc_sorted_problem_schedule_v2`, strided loops over `tp_size`).
- Epilogue retire recomputes the segment scalar-ly (already W=64-safe loop): `ag_scatter_gemm_grouped_with_absmax.h:851-856`.
- Trace meta packs segment in 6 bits (`& 0x3F`, line 674) — holds 0..63.
- `DistEnv` is all int32 (`include/flux/utils.h:52-105`); ring-order helpers are modular arithmetic (`sort_util.h:270-287`).

### (b) How hard is W=64?

Localized: **one block of ~25 lines in one header**, device-side only. Sketch (replacing lines 596-638):

```c++
uint32_t bs0 = __ballot_sync(0xffffffff, lane_idx < W && m_start < split_accum[lane_idx]);
uint32_t be0 = __ballot_sync(0xffffffff, lane_idx < W && m_end   < split_accum[lane_idx]);
uint64_t ms = bs0, me = be0;
if (W > 32) {
  ms |= uint64_t(__ballot_sync(0xffffffff, lane_idx + 32 < W && m_start < split_accum[lane_idx + 32])) << 32;
  me |= uint64_t(__ballot_sync(0xffffffff, lane_idx + 32 < W && m_end   < split_accum[lane_idx + 32])) << 32;
}
int segment_start = __ffsll(ms) - 1;
int segment_end   = __ffsll(me) - 1;
for (int s = lane_idx; s < W; s += 32) {           // lane covers s and s+32
  bool nonempty = split_accum[s] > (s ? split_accum[s - 1] : 0);
  if (s >= segment_start && s <= segment_end && nonempty) { /* existing spin, indexed by s */ }
}
```

Notes: the `__shfl_up_sync` nonempty trick must become a direct `split_accum[s-1]` read for the s>=32 half (cheap, L2-resident, once per tile); the spin body (`signal_ptr[s]` / `barrier_ptr[s]` / `progress_slots` stores) just re-indexes `lane_idx` → `s`. The progress/trace side (`arrival_gt[lane_idx]` under `threadIdx.x < 32`, line 622-630) needs the same `s` re-index. Everything else in the tile (weight-gate wait 647-655 — indexed by local group, not rank; retire 851-856; claimer; bucket build) is untouched. Host side needs only a `FLUX_CHECK(world_size <= 64)` guard (none exists today — grep found no world-size cap in the .cc).

Two-warp ballot + cross-warp reduction, per-node hierarchical gating, etc. are all unnecessary — the whole gate executes redundantly per warp already, so keeping it warp-local with a 2x strided loop is the minimal, schedule-neutral extension.

### (c) Scope of the constraint

Ballot only. The virtual-expert mapping is pure replicated CPU torch with no bitmasks (`python/flux/testing/moonep_fused_map.py` — int32/int64 index tensors throughout; `preflight_metadata_checks`/`required_a2av_knobs` at lines 196-270 are loop-based over W and nn=W/L). `WeightPushMulticast` gating is per-(slot) epoch signals, not per-rank masks. `assert args.G % W == 0` holds at G=128, W=64.

### Recommendation: implement W<=64 (Option A)

Reasoning: Task 2's allgather bug is the *same* code path (barrier arm of the same spin, see below), so the kernel fix is needed anyway; doing it once removes the moonep_fused structural exclusion for free. Graceful-unsupported would leave the sweep's 16-node arm permanently dark *and* leave the allgather W=64 correctness bug in place.

- **Option A diff**: ~40-60 lines in `ag_scatter_gemm_grouped_with_absmax.h` (process_tile), ~3 lines host guard in `gemm_grouped_v2_ag_scatter.cc`, 1-line assert change to `W <= 64` at `test_moe_moonep_fused_traffic.py:427`. **Rebuild scope**: the header is included via `src/moe_ag_scatter/gemm_grouped_v2_ag_scatter.hpp` into the generator-emitted `GemmGroupedV2AGScatter` instantiation space (`src/generator/gen_moe_ag_scatter.cc`) — a full CUTLASS re-instantiation of the V2 AG-scatter op family only (tens of minutes parallel; other op families cached).
- **Option B diff** (fallback if a 16-node window can't be gotten for validation): ~10 lines in the test — replace the assert with a distinctive marker (e.g. `[moonep-fused] UNSUPPORTED: W=%d > 32 (single-warp gating ballot)` on stderr + `RECORDER.emit_info(status="unsupported_world_size")` + `sys.exit(96)`), plus ~5 lines in `sweeps/sweep.py:821-823` mapping exit code 96 → `status="unsupported"` (today any nonzero exit → `"failed"`). No rebuild.

## TASK 2 — allgather W=64 zero-band: ranked hypotheses

Path taken at W=64/nnodes=16 (dense, multi-node): `gemm_grouped_v2_ag_scatter.cc:3193-3204` → `all_gather_all2all` (`:957-1023`, ring-ordered `nvshmemx_getmem` fetches, per-source flag via `CUStreamWriteValue` into `barrier_block[src_rank]` at `:1015-1019`) → GEMM gated only on `fetch_remote_event` = fetches *issued*, not completed (`:3457-3459`) → in-kernel per-segment barrier spin is the sole correctness backstop (`:3282-3295` selects `barrier_block` as `barrier_ptr`; kernel spin at header `:634-637`).

**H1 (dominant, ~certain): the single-warp segment ballot silently drops segments 32-63.**
- Evidence: `ag_scatter_gemm_grouped_with_absmax.h:592-638` as above — at W=64 no tile ever waits for source ranks 32-63; tiles fully beyond `split_accum[31]` wait for nothing.
- Why rows are *exactly zero*: `clear_buffers()` zeroes `input_buffer` every iteration when nnodes>1 (`gemm_grouped_v2_ag_scatter.cc:3974-3977`), and the test calls `op.clear_buffers()` per iteration (`test_moe_ag_traffic.py:255`). A tile that reads unarrived gathered rows reads exact zeros → output rows = 0·B = exactly 0. Sub-tile zero granularity is expected because the corruption is in gathered **A rows**, not in tile scheduling.
- Fit to the observed signature: b8 `[21,4096]` rows 1-2 — the whole 21-row problem is a *single* tile (tile_M>=128), so a scheduling/truncation bug would zero all 21 rows; only per-row A corruption can zero rows 1-2. b1 `[647,4096]` rows 320-641: 647×(32/64) ≈ 323 — the band starts almost exactly at the source-32 cumsum boundary and spans ≈ sources 32-63. Failing ranks differ only in which unguarded sources lose the copy race (ring fetch order `:971-1013` starts at own node/local_rank, making the loser set rank-position-dependent — consistent with the local_rank-3 pattern, e.g. NIC assignment on Perlmutter).
- W=16/32 pass because the ballot then covers all of `world_size`.

**H2 (low): reference path wrong.** `MoeAgScatterWithTorch` (`python/flux/testing/moe_ag_scatter_utils.py:207-275`) is plain `all_gather_into_tensor` + `index_select` + per-expert matmul, shares `ctx.gather_index`/`splits_cpu` with flux, no W-dependent constructs. Already contradicted by the zero-rows signature (a wrong reference would make *torch* rows differ nonzero-vs-nonzero).

**H3 (low): ProblemSchedV2 int16 truncation.** `workspace_util.h:29-31` (`int16_t problem_idx/tile_m_start/tile_m_size`, `static_assert(sizeof==6)` at `workspace_util.cu:33`). Values at the failing shapes are tiny (problem_idx < ep_nexperts·groups = 2; tile counts < 32k); would produce tile_M-aligned whole-tile holes — contradicted by rows 1-2 of 21.

**H4 (low): barrier flag lifecycle race.** memset of `barrier_block` on main stream (`:3202`) vs flag writes on cp_stream — but cp_stream is fenced behind `ready_event` recorded after `nvshmemx_barrier_all_on_stream` (`:982-984`), so ordered. Would anyway zero *all* of one source's rows across guarded segments too.

**H5 (note, not these budgets): cute int32 size overflow in `all_gather_all2all`'s `full_input`** (`:961-966`) — `elemsize·hidden·ntokens_shard·W` exceeds 2^31 at b64 (8192·4096·2·64 ≈ 4.3e9). Irrelevant at b1/b8/b16 but worth a guard while in there.

### Dump discrimination plan (offline, no GPU)

From the dumped flux/torch outputs + moe_ctx (splits, gating, scatter_index):
1. Build per-expert per-source cumsum: `splits_per_source[s][e]` = bincount of source-rank-s tokens choosing expert e (source rank = gathered row // tokens_per_rank). Map each zero row → (expert, offset) → source segment s.
2. **H1 predicts**: every zero row has s >= 32; band edges land on segment boundaries (± segment-size fuzz), never on tile_M multiples; the b1 band start reproduces `cumsum[e][31]`.
3. **H3 predicts**: bands are tile_M-aligned whole tiles (already falsified by b8).
4. **H2 predicts**: flux rows nonzero; a CPU fp64 recompute from moe_ctx sides against torch (already falsified by zeros).
5. If any zero row maps to s < 32 → H1 wrong; go to H4 and dump `barrier_block` semantics next.
6. Corroboration: the dumped `gathered_input` (`allgather_output`, copied after `all_gather_event` at `:3566/3655`) should be fully correct under H1 — data arrived, GEMM read early.

### Fix = the Task 1 Option A kernel change (dense/barrier arm re-indexed identically).

## GPU verification needs

- **Static / no GPU**: dump analysis above; `test_moonep_fused_map.py` (pure CPU); compile of the re-instantiated op family; code review of the s-indexed spin against `a2av_tile_claimer.hpp` mask conventions.
- **Single node / W<=4-16**: regression of allgather (nnodes==1 branch, different barrier source), a2av/a2av_ring/lb_union arms at W<=16 to prove the generalized ballot is schedule- and perf-neutral where it used to pass; optionally a tiny standalone kernel test of the 64-segment gate with a faked `split_accum`/barrier array (fits `src/moe_ag_scatter/test/` alongside `test_sort_utils.cc`, single GPU).
- **16-node W=64 window (must run)**: `test_moe_ag_traffic.py --comm_pattern allgather` at the failing cells (b1/b8/b16, same routing files) expecting torch_allclose pass; one a2av-family W=64 run (claimer + window-keyed gating at 64 lanes); `test_moe_moonep_fused_traffic.py` moonep_fused_push_auto_gated smoke with the assert relaxed to W<=64. Before the fix lands, one diagnostic W=64 run with the dump enabled is enough to close the H1 confirmation if the existing dumps are insufficient.

### Critical Files for Implementation
- /global/u1/y/yufeid/workspace/changchen/andrewy/flux/.claude/worktrees/debug/src/moe_ag_scatter/cutlass_impls/ag_scatter_gemm_grouped_with_absmax.h (process_tile:592-638 — the fix site)
- /global/u1/y/yufeid/workspace/changchen/andrewy/flux/.claude/worktrees/debug/src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc (barrier/gating wiring :957-1023, :3282-3338, :3974-3983; host W<=64 guard)
- /global/u1/y/yufeid/workspace/changchen/andrewy/flux/.claude/worktrees/debug/test/python/moe_ag_scatter/test_moe_moonep_fused_traffic.py (assert :427 → W<=64, or Option B graceful exit)
- /global/u1/y/yufeid/workspace/changchen/andrewy/flux/.claude/worktrees/debug/test/python/moe_ag_scatter/test_moe_ag_traffic.py (W=64 verification driver, reference path)
- /global/u1/y/yufeid/workspace/changchen/andrewy/flux/.claude/worktrees/debug/src/moe_ag_scatter/workspace_util.cu (already-W64-safe schedule/bucket build — reference for mask conventions; int16 ProblemSchedV2 note)


# ===== SUBAGENT agent-a98acac7eaa73b6c4.jsonl =====


## [2026-08-15T14:49:28.668Z]

I have completed the exploration and root-caused all three issues, including finding the exact code path for the routing-mode anomaly (C). Full implementation plan follows.

---

# Implementation Plan: knob sizing, C++ check semantics, and the routing-drop bug

## Part C first — the routing-mode anomaly (root cause FOUND, smallest fix, gates data validity)

### Root cause
`sweep.py`'s one-shot retry pass rebuilds the cell dict from a key whitelist that **omits `routing_mode`** (`/global/u1/y/yufeid/workspace/changchen/andrewy/flux/.claude/worktrees/debug/sweeps/sweep.py:1289-1300`):

```python
fresh = {
    k: old[k]
    for k in ("cell_id", "variant", "mode", "family", "family_params",
              "budget_mib", "world_size")
}
redo = run_cell(spec, plat, fresh, jobid, matrices[old["cell_id"]], ...)
```

Chain: the `:1147` overflow at large budgets kills one rank → 31 ranks spin → watchdog returns `"stuck"` (`:879-895`) → `"stuck" in RETRY_STATUSES` (`:82`) → retry runs with `routing_mode` missing → `run_cell:773` (`routing_path=matrix.get("routing") if cell.get("routing_mode") == "real" else None`) passes `None` → no `--routing_file` → `finalize:970` records `routing_mode=cell.get("routing_mode","")` = `''`. This exactly reproduces the observation: only the budgets that failed once (8n/b64, 16n/b32, 16n/b64) show `''`, siblings show `real` — per-budget variation inside a per-family setting, via the retry path. The retried cell silently measured **dealer** routing on a trace matrix (mislabeled data).

### Fix (sweep.py only)
1. Replace whitelist reconstruction with a pristine-cell lookup: build `by_id = {c["cell_id"]: c for c in runnable}` before the run loop; `fresh = dict(by_id[old["cell_id"]])`. Kills the whole dropped-key bug class (any future cell key survives retries). Factor into a small function `_fresh_retry_cell(old, by_id)` so it is unit-testable.
2. Fail loudly instead of degrading, in `run_cell` (before `build_cell_cmd`):
   - `if cell["family"] == "trace" and cell.get("routing_mode") not in ("real", "dealer"): raise SystemExit(...)`
   - `if cell.get("routing_mode") == "real" and not matrix.get("routing"): raise SystemExit(...)` (today a missing `matrix["routing"]` silently drops the flag too).
3. Keep `cell_note` provenance as-is (`recovered on retry ...`), which lets you audit affected past capsules.

### Verification (no GPU)
- New pure-python test (e.g. `sweeps/` test or extend dry-run): assert `_fresh_retry_cell` preserves `routing_mode` and every key of `expand_cells` output; assert the loud-fail guards trigger.
- Audit existing capsules: `cells.csv` rows with `family=trace AND routing_mode='' ` are invalid measurements (dealer bytes labeled trace) — plan a note in the capsule ledger; the 8n/16n GPU windows re-run those cells.

## Part A — `scale_knobs()` fix (sweep.py:262-280)

### Recommended approach: exact per-cell computation (not a W-aware closed form)
Reasons:
- Both gate quantities are already computable exactly, in pure python, from artifacts that exist on disk **before any cell launches**: the matrix (`<mid>.txt`, bytes; copies = bytes/chunk_bytes) and, for `routing_mode=real`, `<mid>.routing.txt`. `gen_trace_routing.real_dedup_stats(routing, W, L, T, G)` already produces exactly the `u`/`U` the runtime derives (`:276-291`); the dealer closed form lives in `gen_matrix.dedup_round_stats` (`:236-256`).
- A closed form is structurally impossible to make tight: the union recv demand `max_d Σ_s region_rows(s,d)` depends on real token-overlap (measured 518504 at 16n/b64 vs the W·T ceiling 524288 — nearly saturated there, but overlap-dependent elsewhere), and the copies-column-max (the non-compress recv demand, 271685 at 8n/b64) depends on family skew, not just budget.
- Precedent: the `moonep_fused` driver already sets EXACT knobs from the plan (`build_cell_env` comment at `sweep.py:588-592`).

**On reusing `moonep_fused_map`:** `required_a2av_knobs(meta: FusedMeta, W, local_world_size)` (`python/flux/testing/moonep_fused_map.py:225-272`) does **not** need a plan — it reads only `meta.a2av_unique_counts` (`[W, W+nn]` = u‖U) and `meta.m_per_rank` (`[W]` copies column sums). But it needs torch, and `sweep.py` runs on the login node and deliberately never imports flux/torch (`probe_capabilities` comment `:363-371`). So: **port the ~35 lines of arithmetic** (`region_rows` max-col, `_chunk_bound` stage/relay cuts, `moonep_fused_map.py:218-266`) into a pure-python helper in `sweeps/gen_matrix.py` (next to `dedup_stats_from_U`), e.g.:

```
def a2av_knob_demands(chunks_copies, u, U, W, L) -> dict
  # returns {"recv_union": ..., "recv_copies": ..., "stage_ident": ...,
  #          "stage_lb": ..., "relay_lb": ..., "stage_hier_copies": ...}
```

and add a parity unit test against the torch implementation (below).

### Per-knob sizing (one value per matrix, shared by all a2av variants — preserves cross-variant comparability inside a capsule)
- `RECV = max(recv_copies_max_col, recv_union_max_col)` — covers non-compress arms (`a2av_dispatch/ring/hier`, recv holds copies), compress non-union arms (u-regions ≤ union), and lb_union (`region_rows` col max, the `:1262` gate). Including copies-col-max also makes the sweep robust even before the C++ `:1147` fix ships (no rebuild needed to unblock).
- `STAGE = max(stage_hier_copies, stage_identity_U, stage_lb_chunks)` — hier (non-compress) stages copies (`:2905-2918` expression), identity stages `Σ_remote U` per gateway (`:2300-2313`), lb stages balanced chunk cuts (`:2319-2337`). All are closed functions of chunks/u/U.
- `RELAY = relay_lb` (`:2330-2339` expression); non-lb arms don't allocate it.
- **u/U source per cell:** `real` → parse routing file (memoize per matrix id; b64/W64 is ~4M ints, fine on a login node); `dealer`/synthetic families → closed form `U[s][n] = min(Σ_d∈n copies[s][d], T)`, `u[s][d] = min(copies[s][d], T)` per the sorted column-major dealer run argument (`gen_matrix.py:237-240`). **Must be validated** against `traffic_matrix_to_choosed_experts` (`python/flux/testing/traffic_matrix.py:62`, pure CPU torch — load the module file standalone via importlib to avoid the flux package `__init__`).
- **Margin policy:** demands are exact replicas of the runtime's own check expressions computed from the same inputs, so margin = rounding only: `cap = max(163840, ceil(demand/8192)*8192)`. The 163840 floor keeps every currently-passing small-budget cell **byte-identical** in `env_json` (comparability). No multiplicative margin — it would just waste symmetric heap at the 16G ceiling.

### Symmetric-heap sizing (`sym_g`)
Replace the post-topk anchor interpolation with a component sum mirroring the ctor (`gemm_grouped_v2_ag_scatter.cc:694-842`):
`heap_bytes = (send_rows*(2 if pack_overlap variant present else 1) + RECV + STAGE + RELAY) * chunk_bytes` with `send_rows = T*topk`; `sym_g = max(6, ceil(heap_bytes/1G + overhead_g))` with `overhead_g = 1` (calibrated: legacy anchor 163840×4×8KiB = 5G → "6G" validated point). Cap at `plat["sym_size_max_g"]` as today (`:621-623`).

### Unsupported-by-capacity handling
Compute knobs in `cmd_run` right after `ensure_matrix` (attach to `matrices[cell_id]["knobs"]`, memoized per matrix id). If the **uncapped** `sym_g` exceeds `sym_size_max_g`, do not launch: append the cell with a new status `skipped_capacity` (mirror the `skipped_capability` block at `:1240-1253`) and a note carrying demand-vs-cap numbers. Check downstream status consumers (`trace_analysis.py`, summarizers) tolerate the new status string — flag in review.

### Docs
`sweeps/SCHEMA.md` §"Knob scaling (validated anchors)" (`~:396-410`): replace the anchor description with the exact-computation rule, note (a) the date + rationale (lb_union union recv grows ~W·T and broke every W-blind anchor), (b) small-budget values unchanged (floor), (c) protocol note: capsules straddling the policy change differ in `env_json` — knobs are capacity-only, but do not byte-diff `env_json` across the change; also update the "Under-sizing fails loudly ... per-rank" paragraph once B lands.

### Verification
- **No GPU:** (1) unit test `a2av_knob_demands` == torch `required_a2av_knobs` on random routings (build a `FusedMeta(scatter_index=None, splits=None, splits_per_source=None, a2av_unique_counts=…, m_per_rank=…)` — dataclass accepts it — or synthesize via the existing `_build` path in `test/python/moe_ag_scatter/test_moonep_fused_map.py`); (2) regression fixture asserting the new caps reproduce the measured demands: W32/b64→recv ≥ 271685 (copies) and ≥ 238956 (union), W64/b32 ≥ 259218, W64/b64 ≥ 518504, and that small-budget cells emit today's exact values; (3) `sweep.py run --dry-run` prints env per cell — eyeball 8n/16n specs; (4) dealer-u closed-form vs `traffic_matrix_to_choosed_experts` CPU test.
- **GPU window (8n/16n):** re-run the three failing lb_union cells + one small-budget control; confirm `[flux a2av] recv rows ...` ctor log matches, no overflow, heap ≤ 16G.

## Part B — C++ check fixes (`src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc`, surgical)

Facts established:
- `M_this_ep` (copies column-sum for my rank) is used downstream only as the **GEMM row count** (output alloc `:3302`, gemm args `:3323/:3045`, debug narrows `:1810-1990`). The A rows are never materialized in a dedicated M-sized buffer: gather is read-only aliasing into the recv buffer (`:300-303`), and all index scratch (`a2av_sorted_gather_/scatter_`, `a2av_arange_i64_`, `a2av_e_all_`, …) is ctor-sized to `n_copies_max = tokens_per_rank_max * topk * W` (`:740-798`) ≥ any column sum. **No separate A-row capacity check is needed** — say so in a comment.
- `:2069` can never be on the compress path (`:1061` `FLUX_CHECK(!compress || use_meta)`), so its recv-copies semantics are already correct there.

Changes:
1. **`:1147`:** wrap in `if (!compress)` (the `:1262` collective union check is the true recv gate under compress), and change the expression from per-rank `M_this_ep` to the collective `max_d Σ_s chunks64[s*W+d]` (chunks64 is fully populated from replicated `cnt_host` metadata at `:1135-1143` — same expression on every rank, mirrors the `:1262`/`:2313`/`:2337` pattern; strictly stronger than today, and any run it newly fails was doomed to a hang anyway).
2. **`:2069`:** same max-over-destinations rewrite for collectivity — *conditional on verifying* `a2av_chunks_cpu_` (D2H of the derived histogram) is bit-identical across ranks (it is computed from replicated `splits_gpu`/`scatter_index`; verify the counts kernel). If not provable, leave per-rank and extend the comment to state it, since the sweep-side fix (A) now guarantees capacity host-side before launch — that *is* the "pre-validate on host from replicated metadata" option, already how the moonep_fused driver works.
3. Optional/flagged, not in the minimal diff: `:3470` `tiled_m_max = max_recv_ntokens_/tile_M + ep_nexperts` under-counts worst-case tiles under compress (tiles follow M_this_ep copies, which may exceed recv rows). nvtx-proxy-only, ring-buffered — note it; fix later if tile traces at big budgets look truncated.
4. Leave `:1283` (`total_send_rows <= copies_per_rank`, per-rank u/U rows) as-is with a comment — documented per-rank, small blast radius, out of scope.

**Risk/ledger:** this file backs 28 distinct builds in the ledger; any change forces a rebuild and a new `flux_libs` sha in future manifests. Fix A alone unblocks the failing sweeps with zero rebuild; land B on the next planned rebuild. Verify with: compile + the existing python preflight mirror tests (`test_moonep_fused_map.py::test_knob_formulas` semantics unchanged); GPU window: 2-node smoke with deliberately undersized `FLUX_A2AV_MAX_RECV_NTOKENS` on lb_union → expect **all** ranks to abort with the `:1262` message (no watchdog kill), and a non-compress run undersized → all ranks abort via the new collective `:1147`.

## Sequencing
1. C (retry fix + loud guards) — smallest, protects data validity immediately.
2. A (exact knobs + sym_g + `skipped_capacity` + SCHEMA.md) — unblocks 8n/16n without rebuild.
3. B (C++ collective/unit-correct checks) — next rebuild.
4. GPU windows: re-run invalidated trace cells and the three overflow cells.

### Critical Files for Implementation
- /global/u1/y/yufeid/workspace/changchen/andrewy/flux/.claude/worktrees/debug/sweeps/sweep.py
- /global/u1/y/yufeid/workspace/changchen/andrewy/flux/.claude/worktrees/debug/sweeps/gen_matrix.py
- /global/u1/y/yufeid/workspace/changchen/andrewy/flux/.claude/worktrees/debug/src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc
- /global/u1/y/yufeid/workspace/changchen/andrewy/flux/.claude/worktrees/debug/python/flux/testing/moonep_fused_map.py
- /global/u1/y/yufeid/workspace/changchen/andrewy/flux/.claude/worktrees/debug/sweeps/SCHEMA.md


# ===== SUBAGENT agent-a9c35b552fbfe38b2.jsonl =====


## [2026-08-15T14:33:23.810Z]

## Summary

I traced the full sweep → variant → test-script → env chain, pulled the exact env/matrix identities out of the three failing capsules, and read the actual rank logs on `$PSCRATCH`. Two of the three failures already have confirmed root causes visible in the logs (see §6).

---

## 1. Variant → test script / args / env mapping

Sources: `/global/u1/y/yufeid/workspace/changchen/andrewy/flux/.claude/worktrees/debug/sweeps/variants.py` (table) and `/global/u1/y/yufeid/workspace/changchen/andrewy/flux/.claude/worktrees/debug/sweeps/sweep.py` lines 39–44 (test constants), 581–635 (`build_cell_env`), 638–761 (`build_cell_cmd`).

| variant | `driver` | test script (sweep.py const) | launcher | variant-specific CLI | variant `env` | heap sizer |
|---|---|---|---|---|---|---|
| `allgather` | flux (default) | `test/python/moe_ag_scatter/test_moe_ag_traffic.py` | `./launch.sh` | `--comm_pattern allgather` | *(none)* | `scale_knobs()` → `FLUX_A2AV_MAX_{RECV,STAGE,RELAY}_NTOKENS` + `NVSHMEM_SYMMETRIC_SIZE` |
| `hier_compress_lb_union` | flux | same as above | `./launch.sh` | `--comm_pattern a2av_hier_compress` | `FLUX_A2AV_LB_UNION=1`, `CUDA_DEVICE_MAX_CONNECTIONS=8` | `scale_knobs()` |
| `moonep_nvshmem_getmem` | `moonep` | `test/python/moe_ag_scatter/test_moe_moonep_traffic.py` | `./launch.sh` | `--transport nvshmem --prefetch_transport getmem` | `CUDA_DEVICE_MAX_CONNECTIONS=8` | `moonep_getmem_sym_size()` — **no** `FLUX_A2AV_*` knobs |
| `eplb` | `eplb` | `test/python/moe_ag_scatter/test_moe_eplb_traffic.py` | `./launch.sh` | `--transport nvshmem` + `--eplb_load_file <mid>.eplb_load.json` | `CUDA_DEVICE_MAX_CONNECTIONS=8` | `eplb_sym_size()` — no `FLUX_A2AV_*` |
| `moonep_fused_push_auto_gated` | `moonep_fused` | `test/python/moe_ag_scatter/test_moe_moonep_fused_traffic.py` | `./launch.sh` | `--weight_path push --weight_push_mode auto --weight_gate tiles` | `FLUX_A2AV_LB_UNION=1`, `CUDA_DEVICE_MAX_CONNECTIONS=8` | `moonep_fused_sym_size()`; **the driver itself computes the exact `FLUX_A2AV_MAX_*` via `os.environ.setdefault` from the plan** (test file line 475–477) — sweep deliberately does not pre-set them |
| `fast` | `fast` | `test/python/moe_ag_scatter/test_moe_ag_fast_baseline.py` | **`./launch_fast.sh`** | *(none; `--comm_pattern` never passed)* | *(none)* | `fast_sym_size()`; requires `3rdparty/FAST/nvidia/libflash.so`, `nodes>=2`, e2e mode only, **cannot take `--routing_file`** (must use the `dealer=1` trace arm) |

Common CLI block appended to every non-fast driver (`sweep.py:709-732`), in this exact order:
```
--topk <topk> --G <G> --H <H> --chunk_bytes <cb> --ffn_hidden_size <ffn>
--dtype <dtype> --iters <n> --warmup_iters <n> --sm_margin <n>
[--routing_file <path>]  [--skip_correctness]  [--profile if mode=torchprof]
```

Mode/runner env added on top for every cell (`build_cell_env`):
- `mode=isolated` → `FLUX_SWEEP_ISOLATED_ITERS=1`
- `mode=phases` → `FLUX_A2AV_TIMING=1`
- always → `FLUX_TEST_DETERMINISTIC=0`, `FLUX_SWEEP_RECORD_DIR=<staging>/records`, `FLUX_EXTRA_TORCHRUN_ARGS="--redirects 3 --log-dir <staging>/torchrun"`
- plus `spec.extra_env` last (so a spec can override anything).

srun prefix (`sweep.py:645-650`): `srun --jobid=<id> --nodes=<N> --ntasks-per-node=1 --gpus-per-node=4` (the `--gpus-per-node=4` comes from `srun_extra` in `sweeps/platforms/perlmutter.yaml`, added 2026-08-15 because srun steps stopped inheriting the GRES).

### Where the traffic matrices come from

- Roots (`sweeps/platforms/perlmutter.yaml`): `matrices_root=$PSCRATCH/workspace/andrewy/a2av_test_matrices/generated`, `traces_root=$PSCRATCH/workspace/andrewy/moe_traces`, `data_root=$PSCRATCH/workspace/andrewy/sweep_data`.
- Trace pool on disk: `/pscratch/sd/y/yufeid/workspace/andrewy/moe_traces/Qwen/Qwen3-235B-A22B-FP8/mmlu/high_school_world_history/` (also `mmlu/college_mathematics`, `mmlu/philosophy`, `mmlu_ZH_CN/*`), fetched by `sweeps/fetch_traces.py`.
- Selection per cell: family string `trace:pools=mmlu/high_school_world_history;layer=92;sem=homog` → `sweeps/gen_trace_routing.py:ensure_trace_matrix()` samples `T = budget_mib*2^20/chunk_bytes` tokens per rank with replacement from the layer-92 decode pool, seeded by `fnv1a(canonical_string(...))`. Matrix id = `w{W}x{L}_trace-{hash}_b{budget}_k{topk}_id{instance}`. Two artifacts land side by side: `<mid>.txt` (the `[W][W]` byte matrix) and `<mid>.routing.txt` (`ntokens topk G` header + one line of `topk` expert ids per token). `sweep.py:1180-1184` passes the `.routing.txt` as `--routing_file` whenever `routing_mode == "real"` (the default for `trace`; `dealer=1` suppresses it).
- The tests hard-verify the pairing: `assert torch.equal(got * chunk_bytes, matrix)` — a routing file that doesn't realize the matrix aborts.

---

## 2. Two concrete manual repro command lines

Prereqs for both: `source ./module.sh` first (sets `NVSHMEM_HOME`, conda env, CUDA 12.4 / nvshmem 3.2.5 / nccl 2.24.3); `salloc --qos interactive -C gpu --account m4243_g -N <8|16> --gpus-per-node=4 -t 30 --no-shell`; run from the repo root.

### A. `hier_compress_lb_union`, trace b64, W=32 (8 nodes × 4), correctness ON

```bash
JOBID=<your jobid>
M=/pscratch/sd/y/yufeid/workspace/andrewy/a2av_test_matrices/generated/w32x4_trace-1a6a90_b64_k8_id001
OUT=/pscratch/sd/y/yufeid/workspace/andrewy/manual/lbunion_b64_w32

FLUX_A2AV_LB_UNION=1 \
CUDA_DEVICE_MAX_CONNECTIONS=8 \
FLUX_A2AV_MAX_RECV_NTOKENS=262144 \
FLUX_A2AV_MAX_STAGE_NTOKENS=262144 \
FLUX_A2AV_MAX_RELAY_NTOKENS=262144 \
NVSHMEM_SYMMETRIC_SIZE=10G \
FLUX_SWEEP_ISOLATED_ITERS=1 \
FLUX_TEST_DETERMINISTIC=0 \
FLUX_SWEEP_RECORD_DIR=$OUT/records \
FLUX_DEBUG_DUMP_DIR=$OUT/dump \
FLUX_EXTRA_TORCHRUN_ARGS="--redirects 3 --log-dir $OUT/torchrun" \
srun --jobid=$JOBID --nodes=8 --ntasks-per-node=1 --gpus-per-node=4 \
  ./launch.sh test/python/moe_ag_scatter/test_moe_ag_traffic.py \
    --traffic_matrix $M.txt \
    --comm_pattern a2av_hier_compress \
    --topk 8 --G 128 --H 4096 --chunk_bytes 8192 --ffn_hidden_size 4096 \
    --dtype bfloat16 --iters 10 --warmup_iters 5 --sm_margin 8 \
    --routing_file $M.routing.txt \
  2>&1 | tee /tmp/lbunion_b64_w32.log
```
This is byte-for-byte the cell `hier_compress_lb_union_trace-37d38e_b64_k8_isolated` from capsules `20260815-074430_perlmutter_3ccbcb47` and `20260815-081416_perlmutter_bb881f76` (the latter added `--skip_correctness`). It will appear to hang for ~4 min — see §6; there is no timeout inside the test, so wrap with `timeout 400` or rely on `srun --time`.

**Root-cause fix to try in the same shell:** raise the three NTOKENS knobs (the log shows 271685 needed vs 262144 provided) and enlarge the heap accordingly:
`FLUX_A2AV_MAX_{RECV,STAGE,RELAY}_NTOKENS=327680 NVSHMEM_SYMMETRIC_SIZE=14G` (recv/stage/relay are each `NTOKENS × H × 2B` symmetric tensors ≈ 2.5 GiB each at 327680, plus a 0.5 GiB send buffer; the platform cap is `sym_size_max_g: 16`).

### B. `moonep_fused_push_auto_gated`, trace b8, W=64 (16 nodes × 4), correctness ON

```bash
JOBID=<your jobid>
M=/pscratch/sd/y/yufeid/workspace/andrewy/a2av_test_matrices/generated/w64x4_trace-3e8e1d_b8_k8_id002
OUT=/pscratch/sd/y/yufeid/workspace/andrewy/manual/moonepfused_b8_w64

FLUX_A2AV_LB_UNION=1 \
CUDA_DEVICE_MAX_CONNECTIONS=8 \
NVSHMEM_SYMMETRIC_SIZE=2G \
FLUX_SWEEP_ISOLATED_ITERS=1 \
FLUX_TEST_DETERMINISTIC=0 \
FLUX_SWEEP_RECORD_DIR=$OUT/records \
FLUX_EXTRA_TORCHRUN_ARGS="--redirects 3 --log-dir $OUT/torchrun" \
srun --jobid=$JOBID --nodes=16 --ntasks-per-node=1 --gpus-per-node=4 \
  ./launch.sh test/python/moe_ag_scatter/test_moe_moonep_fused_traffic.py \
    --traffic_matrix $M.txt \
    --weight_path push --weight_push_mode auto --weight_gate tiles \
    --topk 8 --G 128 --H 4096 --chunk_bytes 8192 --ffn_hidden_size 4096 \
    --dtype bfloat16 --iters 10 --warmup_iters 5 --sm_margin 8 \
    --routing_file $M.routing.txt \
  2>&1 | tee /tmp/moonepfused_b8_w64.log
```
Do **not** set `FLUX_A2AV_MAX_*` here — the driver computes exact values from the plan via `os.environ.setdefault` and any explicit value silently wins.

Deliberately omitted `FLUX_DEBUG_DUMP_DIR`: this test has no dump path (see §3). It will abort in ~11 s on a hard Python assert (see §6).

---

## 3. `FLUX_DEBUG_DUMP_DIR` mechanism (commit `19ce9f9`)

**Only one file implements it**: `/global/u1/y/yufeid/workspace/changchen/andrewy/flux/.claude/worktrees/debug/test/python/moe_ag_scatter/test_moe_ag_traffic.py`, lines 699–724 (`check_result`).

- **Trigger**: `flux.torch_allclose(x, y, atol, rtol)` raises (`python/flux/util.py:38`, plain `RuntimeError` after printing the diff table: shapes, `x-y`, mismatched elements, diff count/%, max diff, diff index list to stderr). `atol/rtol = 1e-2/1.5e-2` for bf16, `1e-2/1e-3` for fp16. Bitwise mismatch alone does **not** trigger a dump — it only sets `bitwise_all=False` and prints; only the allclose failure dumps and re-raises.
- **Files written**, into `os.environ.get("FLUX_DEBUG_DUMP_DIR", "/tmp")` (dir `makedirs(exist_ok=True)`), one set per failing rank:
  - `flux_<rank>.pt` — the flux output tensor `x` (`torch.save`)
  - `torch_<rank>.pt` — the torch-reference output `y`
  - `moe_ctx_<rank>.pt` — the whole `MoeMlp1Ctx` object (inputs, weights, splits, gating args) — this is the multi-GB one
  
  Names come from `check_result(perf_result_flux, perf_result_torch, "flux", "torch")` at line 728, so `name_x="flux"`, `name_y="torch"`.
- **Format**: raw `torch.save` pickles (CUDA tensors — load with `map_location="cpu"`).
- **Practical gotchas**:
  1. The check runs under `flux.exec_in_rank_order` (`python/flux/dist_utils.py:85`) — ranks check one at a time with a barrier between. The first failing rank raises and never reaches its barrier, so **in practice only that one rank's dump lands**; the rest hang until torchrun tears down.
  2. Default `/tmp` on Perlmutter compute nodes is **node-local and discarded at job end** (CLAUDE.md:64-68). For a manual repro you must point `FLUX_DEBUG_DUMP_DIR` at `$PSCRATCH` or the dumps evaporate. The commit's motivation was only to stop polluting the repo tree.
  3. `RECORDER.emit_correctness(bitwise, allclose=False)` fires before the re-raise, so `cells.csv` `correct_allclose=0` is written even on the crash path.
- Sibling tests that still dump into cwd (not converted): `test_moe_ag.py:557-559`, `examples/moe_layer0.py:357-359`, `test/python/stress/*`.
- **The other four drivers have no dump machinery at all.** `test_moe_moonep_fused_traffic.py:377-415` catches `RuntimeError` from `torch_allclose(verbose=False)`, prints `rank N: V3b virtual expert V (group G) mismatch`, and ends with `assert ok_bitwise and ok_allclose`. Same shape in `test_moe_moonep_traffic.py:285`, `test_moe_ultraep_traffic.py:371`, `test_moe_eplb_traffic.py:285`.

---

## 4. Debug / tracing knob list

**a2av instrumentation** (all read via `get_int_from_env` in `src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc`):

| knob | line | effect |
|---|---|---|
| `FLUX_A2AV_TIMING=1` | 1109, 3414 | per-forward stage marks to stderr: `[a2av-timing] rank R stage1 … stage2 … gemmgate … gemm … barrier … ms`, plus `[a2av-stage2]`, `[a2av-host]`, `[a2av-relayfwd]` lines (regexes in `sweep.py:85-143`). **Forces a per-iteration device sync — never compare its e2e against clean cells.** |
| `FLUX_A2AV_NVTX_PROXY=1` | 872, 3471 | live NVTX domain "a2av" per-source ranges inside the GEMM span; also writes per-rank tile-trace sidecars `a2av_tile_trace_r<rank>.bin` into `FLUX_SWEEP_RECORD_DIR`. Analyze with `sweeps/plot_a2av_trace.py`. |
| `FLUX_A2AV_TRACE_EPOCHS` | 3475 | default 4 — iterations of tile-trace ring capacity |
| `FLUX_A2AV_CHECK_COMPRESS=1` | 1542, 1614, 1701, 1809, 1915 | in-kernel verification of the compress pack/forward/fused paths. This is the recommended first correctness probe per `docs/handoff/01_perlmutter_bringup.md` §4. |
| `FLUX_A2AV_CHECK_IDENTITY=1` | 1973 | verify the identity-relay wire |
| `FLUX_A2AV_BLOCKING_WIRE=1` | 629 | instrumented only: makes inter-node puts visible device spans |
| `FLUX_A2AV_EARLY_LAUNCH=1` | 628 | functional reorder (GEMM issued right after stage 2); visibility mode, not a perf config; requires `CUDA_DEVICE_MAX_CONNECTIONS>1` |
| `FLUX_A2AV_NO_GEMM_GATE=1` | 3425 | launch GEMM without waiting (ablation) |
| `FLUX_A2AV_SYNC_BEFORE_PUTS`, `FLUX_A2AV_STAGE2_AFTER_PUTS` | 2025-2026 | ordering ablations |
| `FLUX_A2AV_SCHED_PREFETCH_LAST` | — | slot-last tile schedule, default OFF (NR-14 retraction) |
| `FLUX_A2AV_MAX_{RECV,STAGE,RELAY}_NTOKENS` | 694/710/723 | symmetric buffer rows — the capacity knobs that under-provision at W=32 b64 |
| `FLUX_DEBUG_DUMP_DIR` | test-side | correctness-mismatch tensor dump dir |
| `FLUX_TEST_DETERMINISTIC` | test-side | must be `0` for perf; `1` is ~500x slower on compress paths |
| `FLUX_SWEEP_ISOLATED_ITERS=1` / `FLUX_SWEEP_RECORD_DIR` | test-side | isolated-iteration discipline / per-rank JSONL recorder (`python/flux/testing/recorder.py`, writes `rank_NNN.jsonl`) |

**Three-way overlap capture recipe** (SCHEMA.md:362): `FLUX_A2AV_EARLY_LAUNCH=1 FLUX_A2AV_BLOCKING_WIRE=1 CUDA_DEVICE_MAX_CONNECTIONS=8` in nsys mode.

**Watchdog — sweep-runner side only.** `sweep.py:879-895` `_wait_with_watchdog(proc, staging, start, timeout_s, idle_timeout_s)`: absolute `timeout_s=900` → status `timeout`; `idle_timeout_s` (SPEC_DEFAULTS `180`, and `180` in all three failing specs — the handoff doc's "250s" is approximate wall-clock, not a configured value) polls `_latest_mtime(staging)` and kills the process **group** with SIGINT(30s)→SIGTERM(15s)→SIGKILL(10s) → status `stuck`. **There is no timeout inside any test script**, so a hand-launched srun will spin forever; use `timeout <s> srun ...` or `srun --time=…`.

**NVSHMEM env in this repo**: `launch.sh` sets `NVSHMEM_BOOTSTRAP=UID`, `NVSHMEM_DISABLE_CUDA_VMM=1`, and (multi-node only) `NVSHMEM_REMOTE_TRANSPORT=libfabric`, `NVSHMEM_LIBFABRIC_PROVIDER=cxi`. `NVSHMEM_SYMMETRIC_SIZE` is commented out in launch.sh (line 29) — the sweep injects it. `launch_fast.sh` additionally sets `NVSHMEM_IB_ENABLE_IBGDA=0`, `NVSHMEM_DISABLE_GDRCOPY=1`, `SLURM_MPI_TYPE=cray_shasta`, `MPICH_GPU_SUPPORT_ENABLED=0`, default heap `4G`, and `LD_PRELOAD`s `libnvshmem_host.so.3`. **No `NVSHMEM_DEBUG`/`NVSHMEM_INFO` is set anywhere in-repo** — for verbose transport tracing you'd add `NVSHMEM_DEBUG=INFO NVSHMEM_DEBUG_SUBSYS=ALL` yourself (plus `NCCL_DEBUG=INFO`).

**Per-rank logs when hand-launching**: keep the sweep's convention —
`FLUX_EXTRA_TORCHRUN_ARGS="--redirects 3 --log-dir <shared dir>"` (`--redirects 3` = stdout+stderr). torchrun writes `<log-dir>/<run_id>/attempt_0/<local_rank>/{stdout,stderr}.log`; because `TORCHELASTIC_RUN_ID` is `none` under launch.sh, each node gets its own `none_<random>` subdir (so 16 dirs of 4 local ranks at W=64). The log dir **must be on `$PSCRATCH`**, not node-local `/tmp`. Redirect srun's own aggregate output on the login side (`| tee /tmp/…`).

---

## 5. `launch.sh` multi-node behavior + constraints a manual repro must respect

`launch.sh` (49 lines) consumes: `SLURM_NNODES` → `--nnodes`, `SLURM_NODEID` → `--node_rank`, `SLURM_JOB_NODELIST` → master addr via `scontrol show hostnames | head -n1`. `MASTER_PORT` defaults to `23456`; `NPROC_PER_NODE` defaults to `nvidia-smi --list-gpus | wc -l` (4 on Perlmutter). It must be launched **one task per node**: `srun --nodes=N --ntasks-per-node=1 ./launch.sh <script> <args>`.

Constraints:
- **`--gpus-per-node=4` on the srun step is mandatory** (perlmutter.yaml comment, 2026-08-15): without it srun steps see "No devices found" and `nproc_per_node` resolves to 0.
- `NVSHMEM_SYMMETRIC_SIZE` is *not* set by launch.sh — you must export it yourself, using the same sizer the sweep used (values reproduced in §2 and §6 from `env_json`). Platform cap is 16G.
- `MASTER_PORT` collides if you run two steps concurrently in one allocation — override it for the second.
- Divisibility / shape asserts (`test_moe_ag_traffic.py:436-448`): `H * dtype.itemsize == chunk_bytes` (4096 × 2 = 8192); `G % WORLD_SIZE == 0` (128 % 64 = 0 ✓); `matrix.shape[0] == WORLD_SIZE` — **the matrix filename encodes W (`w32x4` / `w64x4`), so you cannot reuse a W=32 matrix at 16 nodes**; `choosed_experts.shape[0] % WORLD_SIZE == 0`; and the routing file must exactly realize the matrix (`torch.equal(got*chunk_bytes, matrix)`).
- `test_moe_moonep_fused_traffic.py:427`: **`assert W <= 32`**.
- `a2av_hier_compress` multi-node needs `--sm_margin >= 1`; `sweep.py:678-686` forces `max(1, sm_margin)` for gather-gateway compress arms, but exempts `UNION_BCAST`/`LB_UNION` (pure-CE forwards). At the spec's `sm_margin=8` this is moot.
- `fast`: `W > L` (≥2 nodes), no `--routing_file`, e2e only, needs `3rdparty/FAST/nvidia/libflash.so`.
- Token count is derived, not passed: `tokens_per_rank = budget_mib*2^20/chunk_bytes` (b64 → 8192/rank → ntokens 262144 at W=32; b8 → 1024/rank → 65536 at W=64).

---

## 6. Bonus: the actual failure causes are already in the rank logs

I read the preserved staging dirs. Two of the three open items are answered:

**(a) `moonep_fused_push_auto_gated` at W=64 is not a correctness bug — it's a hard precondition.** Every rank in `…/20260815-105733_perlmutter_244825e3/cells/moonep_fused_push_auto_gated_trace-37d38e_b8_k8_isolated/torchrun/*/attempt_0/*/stderr.log` shows:
```
[rank40]: AssertionError: fused gating ballot is single-warp (W <= 32)
[rank40]:     assert W <= 32, "fused gating ballot is single-warp (W <= 32)"
```
That is `test/python/moe_ag_scatter/test_moe_moonep_fused_traffic.py:427`. It explains both handoff item #1 (fails in ~11 s even with `--skip-correctness`) *and* this arm's half of item #2 (it never reaches `torch_allclose`). The `torch_allclose` failure at W=64 is **`allgather`-only** — that cell's logs do show `❌ flux check failed` / `RuntimeError` from `python/flux/util.py:66`, on a `[21, 4096]` output tile.

**(b) `hier_compress_lb_union` b64/W32 "hang" is the predicted recv under-provisioning**, exactly as `docs/handoff/01_perlmutter_bringup.md` §8 warned:
```
src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc:1147 Check failed:
271685((M_this_ep)) <= 262144((this->max_recv_ntokens_))
a2av recv buffer overflow; raise FLUX_A2AV_MAX_RECV_NTOKENS
```
(rank 16, from `…/20260815-081416_perlmutter_bb881f76/cells/…/torchrun/`). The check is per-rank and data-dependent, so one rank throws while the other 31 spin at 100% GPU → the runner's idle watchdog kills it and records `stuck`. `scale_knobs()` returned 262144 (`sweep.py:262-278`); the skewed rank needs 271685.

Note the failing runs were executed from a **different worktree** — the paths in those logs read `…/.claude/worktrees/realistic-moe-sweep-campaign/…`, not the `debug` worktree you're in. If the `.so` there differs from the one this worktree resolves, the repro is not identical; check `sha256sum python/flux/lib/libflux_cuda_ths_op.so` before trusting a negative result.


# ===== SUBAGENT agent-a9f37a674f8b51a67.jsonl =====


## [2026-08-15T14:40:11.738Z]

I have everything I need — including the actual failure logs, which confirm the arithmetic exactly. Here is the report.

## 1. `scale_knobs()` — `sweeps/sweep.py:262-280` (called at `sweeps/sweep.py:620`)

```python
def scale_knobs(budget_mib, topk, chunk_bytes):
    """MAX_RECV/STAGE/RELAY + NVSHMEM_SYMMETRIC_SIZE from the validated anchor
    points (topk16 sweeps: post-topk 2..256 MiB -> 163840/6G, 512 -> 262144/10G,
    1024 -> 524288/16G). See SCHEMA.md §knobs; platform yaml may override."""
    row_chunks = budget_mib * (1 << 20) * topk // chunk_bytes
    cap = max(163840, math.ceil(4 * row_chunks / 8192) * 8192)
    post_mib = budget_mib * topk
    if post_mib <= 256:
        sym_g = 6
    else:
        sym_g = min(16, math.ceil(6 + (post_mib - 256) * 10 / 768))
    return {
        "FLUX_A2AV_MAX_RECV_NTOKENS": str(cap),
        "FLUX_A2AV_MAX_STAGE_NTOKENS": str(cap),
        "FLUX_A2AV_MAX_RELAY_NTOKENS": str(cap),
        "NVSHMEM_SYMMETRIC_SIZE": f"{sym_g}G",
    }
```

**The formula contains no `W`, no `L`, no `nnodes`.** All three NTOKENS knobs get the same `cap`. `chunk_bytes = 8192` (`sweep.py:65`), `topk = 8`, so with `T = tokens_per_rank = budget_mib*128`:

| budget | `row_chunks` = T·topk | `cap` (all 3 knobs) | `NVSHMEM_SYMMETRIC_SIZE` |
|---|---|---|---|
| b32 | 32768 | `max(163840, 131072)` = **163840** | 6G |
| b64 | 65536 | `max(163840, 262144)` = **262144** | 10G |

Identical at W=16, W=32, W=64. Verified against the recorded `env_json` of the hung 8n/b64 cell (`sweeps/results/runs/20260815-081416_perlmutter_bb881f76/cells.csv`): `MAX_RECV=MAX_STAGE=MAX_RELAY=262144`, `NVSHMEM_SYMMETRIC_SIZE=10G`. Perlmutter's `sym_size_max_g: 16` (`sweeps/platforms/perlmutter.yaml:20`) does not clamp either value.

## 2. The documented concern

- `sweeps/SCHEMA.md:396-410` (§"Knob scaling (validated anchors)") states the formula and adds: *"Under-sizing fails loudly (FLUX_CHECK recv-overflow / NVSHMEM init) — but note the recv-overflow check is per-rank data-dependent: some ranks throw while the rest spin at 100% GPU. That is what the cell timeout is for."*
- `docs/handoff/01_perlmutter_bringup.md:208-227` (§8, "[HYPOTHESIS-PM]") predicts the risk concentrates at **b32/b64** where `cap` rises above the 163840 floor, and predicts the fingerprint: *"a couple of ranks throwing while the rest spin at 100% GPU. Do not misdiagnose this as a hang."*

§8's stated *mechanism* (L=4 doubles each relay's share) turns out to be the wrong lever — relay/stage are the least-loaded buffers. But its *conclusion* (under-provisioning at b32/b64) is exactly right, for a different reason: the knob is flat in **W**, while the recv column scales roughly linearly in **W**.

## 3. Consumers of `FLUX_A2AV_MAX_RECV_NTOKENS` and overflow behavior

All C++ sites are in `src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc`:

| site | what it does | overflow behavior |
|---|---|---|
| `:693-704` | reads knob; default `min(max_ntokens*topk, ceil(max_ntokens/W)*topk*2)`; allocates `a2av_recv_buffer = nvshmem_create_tensor({max_recv_ntokens_, hidden})` | n/a (allocation) |
| **`:1147`** | `FLUX_CHECK_LE(M_this_ep, max_recv_ntokens_)` — `M_this_ep = Σ_s chunks64[s][rank]`, the **pre-dedup** post-topk copy count for *this* rank | **per-rank, data-dependent → non-collective.** One rank throws, the rest proceed → 100%-GPU hang |
| **`:1262`** | `FLUX_CHECK_LE(max_col, max_recv_ntokens_)` — `max_col = max_d Σ_s region_rows(s,d)`, the **deduped/union** recv rows; comment at `:1246-1249` explicitly notes *"the same expression on every rank, so a failure is collective (no one-rank-throws hang)"* | **collective → all ranks abort → fast fail** |
| `:2069` | same check as 1147 on the non-`use_meta` (device-derived counts) path | per-rank → hang |
| `:1302` (`R_key`) | stride of the Tier-B gating composite key `e*R + row` | correctness depends on recv rows < R; guarded by 1262 |
| `:1883` | `sorted_gidx.clamp_(0, max_recv_ntokens_-1)` — comment says *"tail rows are unread garbage; clamp is pure hygiene"* | silent clamp (not a guard) |
| `:1919-1922`, `:3470` | debug `tok_of_row` buffer; nvtx tile-trace ring capacity | debug paths only |

Sibling knobs: `MAX_STAGE` sizes `a2av_stage_buffer_` (`:710-714`), checked at `:2313`, `:2337`, `:2918`; `MAX_RELAY` sizes `a2av_relay_stage_` (`:723-727`), checked at `:2339`. Both those checks are **collective** (pure functions of the replicated `U_mat`).

Python consumers: `python/flux/testing/moonep_fused_map.py:206-271` computes the **exact** knobs from the plan (the `moonep_fused` driver is deliberately not `scale_knobs`'d — see `sweeps/variants.py:153-159`), and its formula is the authoritative reference:
```python
max_recv = int(meta.m_per_rank.max())          # the :1147 quantity (pre-dedup)
for d in range(W):
    max_recv = max(max_recv, sum(region_rows(s, d) for s in range(W)))  # the :1262 quantity
```
Asserted in `test/python/moe_ag_scatter/test_moonep_fused_map.py:224`.

**Worst-case recv rows for `hier_compress_lb_union`** (`FLUX_A2AV_LB_UNION=1` ⇒ `union_bcast_=true`, `relay_identity_=false`, `sweeps/variants.py:460-464`): remote-source regions hold the whole **node union**, not this rank's subset (`:310-317`, `:1244-1252`):

```
recv_rows(d) = Σ_{s in node(d)} u[s][d]  +  Σ_{s outside} U[s][node(d)]
             ≤ L·T + (W−L)·T = W·T          (hard bound, T = budget_mib·128)
```
vs. `cap ≈ 4·T·topk = 32·T`. So the union arm's headroom runs out around **W ≈ 32-64** — precisely where it broke.

## 4. Arithmetic verdict per cell (exact, computed from the real matrices/routing)

Required = `max(max_d Σ_s c[s][d]  (the :1147 gate),  max_d Σ_s region_rows(s,d)  (the :1262 gate))`.

| W (nodes) | b | `cap` | pre-dedup max col (`:1147`) | union recv max col (`:1262`) | verdict |
|---|---|---|---|---|---|
| 16 (4n) | 32 | 163840 | 101,433 | 64,835 | OK (62%) |
| 16 (4n) | 64 | 262144 | 144,036 | 122,587 | OK (55%) |
| 32 (8n) | 32 | 163840 | 135,230 | 119,255 | OK (83%) |
| **32 (8n)** | **64** | **262144** | **271,685 → OVERFLOW ×1.04** | 238,956 (real routing, passes) | **UNDER-PROVISIONED** |
| **64 (16n)** | **32** | **163840** | **215,747 → OVERFLOW ×1.32** | **259,218 → OVERFLOW ×1.58** | **UNDER-PROVISIONED** |
| **64 (16n)** | **64** | **262144** | **431,792 → OVERFLOW ×1.65** | **518,504 → OVERFLOW ×1.98** | **UNDER-PROVISIONED** |

`MAX_STAGE` / `MAX_RELAY` are **not** the problem anywhere: worst observed is stage 106,122 / relay 48,595 at W=64/b64 — 40% and 19% of `cap`. §8's L=4 relay hypothesis is disconfirmed.

**These are not estimates — the exact numbers appear in the failure logs:**

```
.../gemm_grouped_v2_ag_scatter.cc:1147 Check failed: 271685((M_this_ep)) <= 262144((this->max_recv_ntokens_))
   [rank16 only, 8n/b64]   -> /pscratch/.../20260815-081416_perlmutter_bb881f76/cells/
                              hier_compress_lb_union_trace-37d38e_b64_k8_isolated/torchrun/none_zi60amav/attempt_0/0/stderr.log
.../gemm_grouped_v2_ag_scatter.cc:1262 Check failed: 259218((max_col)) <= 163840(...)   [every rank, 16n/b32]
.../gemm_grouped_v2_ag_scatter.cc:1262 Check failed: 518504((max_col)) <= 262144(...)   [every rank, 16n/b64]
   -> /pscratch/.../20260815-111907_perlmutter_007e9586/cells/hier_compress_lb_union_trace-37d38e_b{32,64}_k8_isolated/
```

## 5. Why 8n hangs and 16n fails fast — same overflow, two different checks

`:1147` executes strictly before `:1262` (both inside the `use_meta` block; the compress metadata sub-block starts at `:1207`).

- **8n / b64 (hang):** only the `:1147` **per-rank** check trips, and on **exactly 1 of 32 ranks** (rank 16, whose column sum is 271,685; the next-largest is 209,103). The other 31 ranks evaluate the collective `:1262` check, find `max_col ≤ 262144`, pass it, and go on to the wire — then spin forever on epoch signals from a rank that is already dead (the GEMM tile spin is described at `:2829-2831`; the front-end waits are the `CUStreamWaitValue64` calls at `:2152`, `:2181`, `:2593`, `:2693`, `:2840`). Grepping every rank's stderr in that cell yields **only** the rank-16 `:1147` message — no `:1262` failure anywhere. That is the §8-predicted "looks like a hang; is not" fingerprint, and the 250s watchdog kill is what `cells.csv` records as `stuck`.
- **16n / b32 + b64 (fast fail):** the overflow is large enough that the **collective** `:1262` check trips too, with the identical value on every rank (259,218 / 518,504), so all 64 ranks abort in ~11s with a clean traceback. (1 rank at b32 and 3 ranks at b64 additionally trip `:1147` first — irrelevant, everyone dies either way.)

So: **one root cause, two check sites, two symptoms.** No logic bug is required to explain item #3.

## Two secondary findings worth carrying into the fix

1. **`:1147` is unit-mismatched for the compress path.** It compares `M_this_ep` (the rank's **A-row / pre-dedup copy** count) against `max_recv_ntokens_` (a **recv-row** capacity). Under `a2av_hier_compress` those are different quantities — the code's own comment at `:1206-1210` says `M_this_ep` "stays untouched"/logical, and `:300-303` says multiple GEMM A rows alias one recv row via `sorted_gather_index`. At 8n/b64 the *actual* recv requirement (238,956) fits in 262,144; the run was killed by a check on a number that never lands in the buffer. Raising the knob fixes it either way, but this is worth deciding deliberately (things like the `e*R_key` gating stride at `:1302` and the clamp at `:1883` are recv-row-based, so the recv-row check at `:1262` is the one that must stay).
2. **The failing cells did not run the routing they are documented as running.** `srun.log` shows the 8n/b64 and 16n/b32+b64 commands were issued **without `--routing_file`** (and with `--skip_correctness`), i.e. synthetic max-dedup dealer routing, while b1-b32 at 8n were issued **with** it. `cells.csv` records `routing_mode=''` for exactly those cells while sibling cells in the same capsule record `real`. Under current `sweep.py` (`:305-307`, `:773`, `:1180`) that should be impossible — `routing_mode` is per-family, not per-budget. This also explains why `max_col` at 16n (259,218/518,504) exceeds what the real trace routing would give (216,317/432,684): the dealer dedups less. Worth root-causing before anyone quotes those cells as "real Qwen3 routing".

**Suggested knob for a re-run** (not applied — read-only): `extra_env` with `FLUX_A2AV_MAX_RECV_NTOKENS` at least 288K for 8n/b64, 288K for 16n/b32, and 576K for 16n/b64 (round to 8192). Note 16n/b64 at 576K rows × 8192 B = 4.5 GiB for the recv buffer alone; with stage+relay left at `cap` the symmetric heap will need to go above the current 10G (Perlmutter's `sym_size_max_g: 16` gives room, but the three knobs should be sized separately rather than all set to one `cap` — stage/relay need only ~40%/20% of it).


# ===== SUBAGENT agent-ad781d2f6fd71d3bc.jsonl =====


## [2026-08-15T14:35:02.148Z]

Here is the full evidence dossier.

---

# 1. Resolved data root

`sweeps/platforms/perlmutter.yaml` → `data_root: ${PSCRATCH}/workspace/andrewy/sweep_data`, with `$PSCRATCH = /pscratch/sd/y/yufeid`.

**Resolved: `/pscratch/sd/y/yufeid/workspace/andrewy/sweep_data`**

Related roots from the same yaml:
- matrices: `/pscratch/sd/y/yufeid/workspace/andrewy/a2av_test_matrices/generated`
- traces: `/pscratch/sd/y/yufeid/workspace/andrewy/moe_traces`
- python env: `/pscratch/sd/y/yufeid/conda_envs/andrewy-comet`

# 2. Log directory naming convention (task 4)

Per `sweeps/SCHEMA.md` (lines 20–40) plus what is actually on disk:

```
<data_root>/<run_id>/cells/<cell_id>/
    srun.log                       # line 1 = "+ srun ... ./launch.sh <script> <args>"
                                   # line 2 = "+ env {json}"   <-- exact repro command + env
                                   # then all torchrun lines, then the ChildFailedError summary
    records/rank_NNN.jsonl         # recorder output (hashed into manifest.json)
    torchrun/none_<8char>/attempt_0/<local_rank 0..3>/{stdout,stderr}.log
    nsys/  prof/
<data_root>/<run_id>/cells/<cell_id>.attempt0/   # sibling dir holding the FIRST attempt when retried
```

- `run_id` = `<UTC yyyymmdd-HHMMSS>_<platform>_<hex8>`; `cell_id` = `{variant}_{family_slug}_b{budget}_k{topk}_{mode}`.
- **Gotcha:** the `none_<8char>` component is a random torchrun rdzv id **per node**, so you cannot map node_rank → directory by name. The subdirectory under `attempt_0/` is the **local** rank (0–3); the global rank only appears inside the file as the `[rankNN]:` prefix. Practical recipe: `grep -rl "Traceback\|Check failed\|diff count" <cell>/torchrun/`.
- Global-rank → node mapping is recoverable from `srun.log`'s ChildFailedError block (`rank : 40 (local_rank: 0)`, `host : nid002585-hsn0`).
- `cells.csv` column `log_dir` gives the cell dir directly; `env_json` gives the env; `matrix_path` gives the traffic matrix.
- **Important:** not every run has a committed capsule. The 16n b32/b64 data lives **only** in the data root under `20260815-111907_perlmutter_007e9586` (no `spec.yaml`/`cells.csv`/`manifest.json` in `sweeps/results/runs/`). Same for `20260815-105146_perlmutter_fe2062c5` (an earlier partial 16n attempt).

---

# 3. Failure (a) — `moonep_fused_push_auto_gated` crash at 16n. ROOT CAUSE FOUND.

**It is not a CUDA/NVSHMEM crash. It is a hard Python `assert` that W ≤ 32.** The arm is structurally incapable of running at W=64; there is no memory/fabric bug to chase.

Failing cells (capsule `20260815-105733_perlmutter_244825e3`, 16n / W=64, correctness ON):
`moonep_fused_push_auto_gated_trace-37d38e_b{1,2,4,8,16}_k8_isolated` — all `failed`, exit 143 (b16: exit 1), `correct_allclose` **blank**.

Log file (b1, global rank 40 = root cause rank from `srun.log`):
`/pscratch/sd/y/yufeid/workspace/andrewy/sweep_data/20260815-105733_perlmutter_244825e3/cells/moonep_fused_push_auto_gated_trace-37d38e_b1_k8_isolated/torchrun/none_dw8yuams/attempt_0/0/stderr.log`

VERBATIM (whole file):
```
[rank40]: Traceback (most recent call last):
[rank40]:   File "/global/u1/y/yufeid/workspace/changchen/andrewy/flux/.claude/worktrees/realistic-moe-sweep-campaign/test/python/moe_ag_scatter/test_moe_moonep_fused_traffic.py", line 363, in <module>
[rank40]:     assert W <= 32, "fused gating ballot is single-warp (W <= 32)"
[rank40]:            ^^^^^^^
[rank40]: AssertionError: fused gating ballot is single-warp (W <= 32)
[rank40]:[W815 04:14:11.787031671 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
```

All 16 node-leader ranks emit it identically (ranks 0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60 — plus ranks 4–7 and 19,43 on some nodes). Same at **every** budget; e.g. b16:
```
[rank56]: AssertionError: fused gating ballot is single-warp (W <= 32)
[rank32]: AssertionError: fused gating ballot is single-warp (W <= 32)
... (16 ranks) ...
```

`srun.log` root-cause block:
```
test/python/moe_ag_scatter/test_moe_moonep_fused_traffic.py FAILED
Root Cause (first observed failure):
[0]:
  time      : 2026-08-15_04:14:11
  host      : nid002585-hsn0
  rank      : 40 (local_rank: 0)
  exitcode  : 1 (pid: 1366501)
```

**Exact repro (from `srun.log` lines 1–2, cell b1):**
```
srun --jobid=57010416 --nodes=16 --ntasks-per-node=1 --gpus-per-node=4 ./launch.sh \
  test/python/moe_ag_scatter/test_moe_moonep_fused_traffic.py \
  --traffic_matrix /pscratch/sd/y/yufeid/workspace/andrewy/a2av_test_matrices/generated/w64x4_trace-17f12e_b1_k8_id002.txt \
  --weight_path push --weight_push_mode auto --weight_gate tiles \
  --topk 8 --G 128 --H 4096 --chunk_bytes 8192 --ffn_hidden_size 4096 \
  --dtype bfloat16 --iters 10 --warmup_iters 5 --sm_margin 8
```
env:
```json
{"CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_LB_UNION": "1",
 "FLUX_EXTRA_TORCHRUN_ARGS": "--redirects 3 --log-dir <cell>/torchrun",
 "FLUX_SWEEP_ISOLATED_ITERS": "1", "FLUX_SWEEP_RECORD_DIR": "<cell>/records",
 "FLUX_TEST_DETERMINISTIC": "0", "NVSHMEM_SYMMETRIC_SIZE": "2G"}
```
(cells.csv also records `FLUX_A2AV_MAX_RECV_NTOKENS/RELAY/STAGE = 163840`, `NVSHMEM_SYMMETRIC_SIZE 6G` for the allgather cells; the moonep cells use 2G.)

Source: assert is at **line 427** in the current debug worktree — `/global/u1/y/yufeid/workspace/changchen/andrewy/flux/.claude/worktrees/debug/test/python/moe_ag_scatter/test_moe_moonep_fused_traffic.py:427` (it was line 363 in the now-deleted `realistic-moe-sweep-campaign` worktree the sweep ran from, so the file has drifted).

**This also invalidates half of handoff item #2.** `moonep_fused_push_auto_gated` does **not** "fail `torch_allclose` at 16n" — it never reaches the checker. Grep for `diff count`/`torch_allclose` across every 16n moonep cell returns **zero hits**, and `correct_allclose` is blank (not `0`) in `cells.csv`. Only `allgather` genuinely fails the allclose check at 16n.

---

# 4. Failure (b) — `allgather` `torch_allclose` at 16n. Mismatch reports extracted.

The tensors are `x = perf_result_flux`, `y = perf_result_torch` (from `check_result(perf_result_flux, perf_result_torch, "flux", "torch")`), i.e. **the fused MoE output**, shape `[rows_on_this_rank, 4096]`, bf16, checked at `atol=1e-2, rtol=1.5e-2`. Failing rank reported is **rank 11** (b8/b16) / **rank 3** (b1) on `cuda:3`.

### b8 — the cleanest signal: flux emits **all-zero rows**
File: `/pscratch/sd/y/yufeid/workspace/andrewy/sweep_data/20260815-105733_perlmutter_244825e3/cells/allgather_trace-37d38e_b8_k8_isolated/torchrun/none_jy70zcgw/attempt_0/3/stderr.log`

VERBATIM (excerpt):
```
x:
tensor([[-0.00473022, -0.00173950, -0.00154114,  ...,  0.01495361,
         -0.00405884,  0.00151062],
        [ 0.00000000,  0.00000000,  0.00000000,  ...,  0.00000000,
          0.00000000,  0.00000000],
        [ 0.00000000,  0.00000000,  0.00000000,  ...,  0.00000000,
          0.00000000,  0.00000000],
        ...
y:
tensor([[-0.00473022, -0.00173950, -0.00154877,  ...,  0.01495361,
         -0.00405884,  0.00151062],
        [-0.09375000,  0.14257812, -0.08007812,  ...,  0.06884766,
          0.04638672,  0.02355957],
        [-0.08398438,  0.06005859, -0.01831055,  ..., -0.29687500,
         -0.03027344, -0.01409912],
        ...
x diff:
tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:3', dtype=torch.bfloat16)
y diff:
tensor([-0.09375000,  0.14257812, -0.08007812,  ..., -0.29687500,
        -0.03027344, -0.01409912], device='cuda:3', dtype=torch.bfloat16)
diff count: 7425 (8.632%), [21, 4096]
diff max: 0.37109375, atol: 0.01, rtol_abs: 2.240994945168495e-09
diff locations:
tensor([[   1,    0],
        [   1,    1],
        [   1,    2],
        ...,
        [   2, 4093],
        [   2, 4094],
        [   2, 4095]], device='cuda:3')
--------------------------------------------------------------

[rank11]: Traceback (most recent call last):
[rank11]:   File ".../test/python/moe_ag_scatter/test_moe_ag_traffic.py", line 725, in <module>
[rank11]:     flux.exec_in_rank_order(
[rank11]:   File ".../python/flux/dist_utils.py", line 88, in exec_in_rank_order
[rank11]:     func()
[rank11]:   File ".../test/python/moe_ag_scatter/test_moe_ag_traffic.py", line 726, in <lambda>
[rank11]:     TP_GROUP, lambda: check_result(perf_result_flux, perf_result_torch, "flux", "torch")
[rank11]:   File ".../test/python/moe_ag_scatter/test_moe_ag_traffic.py", line 719, in check_result
[rank11]:     raise e
[rank11]:   File ".../test/python/moe_ag_scatter/test_moe_ag_traffic.py", line 712, in check_result
[rank11]:     flux.torch_allclose(x, y, atol=atol, rtol=rtol)
[rank11]:   File ".../python/flux/util.py", line 66, in torch_allclose
[rank11]:     raise RuntimeError
[rank11]: RuntimeError
```
**Reading:** `diff locations` span exactly rows **1 and 2** of a 21-row tensor, and `x diff` (the flux side of the mismatched elements) is **all zeros**. This is 2 output rows never written — dropped/unscattered tokens, not numerical drift. Deterministic: the retry (`allgather_trace-37d38e_b8_k8_isolated.attempt0`) reports `diff count: 7437 (8.646%), [21, 4096]` — same 2 rows, ±12 elements.

### Other budgets, same cell family (all rank 11 / rank 3, `[rows, 4096]`)
| budget | `diff count` | shape | `diff max` | mismatched row band |
|---|---|---|---|---|
| b1 | `106823 (4.031%)` | `[647, 4096]` | `0.703125` | rows 320 → 641 |
| b2 | `46586 (2.788%)` | `[408, 4096]` | `1.0` | — |
| b4 | `21016 (0.563%)` | `[912, 4096]` | `3.140625` | — |
| b8 | `7425 (8.632%)` | `[21, 4096]` | `0.37109375` | rows 1–2 (x = 0) |
| b16 | `10095 (6.486%)` | `[38, 4096]` | `0.353515625` | rows 3 → 31 |

b16 file: `.../cells/allgather_trace-37d38e_b16_k8_isolated/torchrun/none_yk6j9ohh/attempt_0/3/stderr.log`
```
diff count: 10095 (6.486%), [38, 4096]
diff max: 0.353515625, atol: 0.01, rtol_abs: 2.342858351767063e-09
diff locations:
tensor([[   3,    0],
        [   3,    1],
        [   3,    2],
        ...,
        [  31, 4093],
        [  31, 4094],
        [  31, 4095]], device='cuda:3')
```
b1 file: `.../cells/allgather_trace-37d38e_b1_k8_isolated/torchrun/none_v9c0ndeo/attempt_0/3/stderr.log`. Independent earlier 16n capsule `20260815-105146_perlmutter_fe2062c5` reproduces at b8: `diff count: 4046 (6.174%), [16, 4096]`.

In every case the mismatch is a **contiguous band of rows**, never scattered — consistent with a rank/slot indexing bug at W=64, not fp noise.

### Bonus: the debug-dump crash (handoff item #2's "crashing on repo-tree disk quota")
Verbatim from the same b1 file — this is *why* no `.pt` dumps exist:
```
[rank3]: During handling of the above exception, another exception occurred:
[rank3]: Traceback (most recent call last):
[rank3]:   File ".../torch/serialization.py", line 944, in save
[rank3]:     _save(
[rank3]:   File ".../torch/serialization.py", line 1216, in _save
[rank3]:     zip_file.write_record(name, storage, num_bytes)
[rank3]: RuntimeError: [enforce fail at inline_container.cc:815] . PytorchStreamWriter failed writing file data/0: file write failed
...
[rank3]:   File ".../test/python/moe_ag_scatter/test_moe_ag_traffic.py", line 714, in check_result
[rank3]:     torch.save(x, f"{name_x}_{TP_GROUP.rank()}.pt")
[rank3]: RuntimeError: [enforce fail at inline_container.cc:626] . unexpected pos 448 vs 342
```

**Repro (b8, 16n):**
```
srun --jobid=57010416 --nodes=16 --ntasks-per-node=1 --gpus-per-node=4 ./launch.sh \
  test/python/moe_ag_scatter/test_moe_ag_traffic.py \
  --traffic_matrix /pscratch/sd/y/yufeid/workspace/andrewy/a2av_test_matrices/generated/w64x4_trace-<...>_b8_k8_id002.txt \
  --comm_pattern <allgather> --topk 8 --G 128 --H 4096 --chunk_bytes 8192 \
  --ffn_hidden_size 4096 --dtype bfloat16 --iters 10 --warmup_iters 5 --sm_margin 8
```
env (from cells.csv, b1 row; identical shape at other budgets):
```json
{"FLUX_A2AV_MAX_RECV_NTOKENS": "163840", "FLUX_A2AV_MAX_RELAY_NTOKENS": "163840",
 "FLUX_A2AV_MAX_STAGE_NTOKENS": "163840", "FLUX_SWEEP_ISOLATED_ITERS": "1",
 "FLUX_TEST_DETERMINISTIC": "0", "NVSHMEM_SYMMETRIC_SIZE": "6G"}
```
Matrix b1: `w64x4_trace-17f12e_b1_k8_id002.txt` (+ `.routing.txt`), sha256 `30e456b6…`/`8571f6a9…`. git sha `b17d826586c52f8452d59f2cd472972994306722` (dirty=1).

---

# 5. Failure (c) — 16n `hier_compress_lb_union` b32/b64 fast-fail. ROOT CAUSE FOUND.

**It is exactly the predicted `FLUX_A2AV_MAX_RECV_NTOKENS` under-provisioning. Not a logic bug.**

Data lives **only** at `/pscratch/sd/y/yufeid/workspace/andrewy/sweep_data/20260815-111907_perlmutter_007e9586/` (uncommitted 16n b1..b64 `--skip_correctness` run).

### b32 (cap = 163840)
File: `.../cells/hier_compress_lb_union_trace-37d38e_b32_k8_isolated/torchrun/none_qj0hx5vw/attempt_0/0/stderr.log`
```
/global/.../worktrees/realistic-moe-sweep-campaign/src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc:1147 Check failed: 215747((M_this_ep)) <= 163840((this->max_recv_ntokens_))a2av recv buffer overflow; raise FLUX_A2AV_MAX_RECV_NTOKENS
[rank32]: Traceback (most recent call last):
[rank32]:   File ".../test/python/moe_ag_scatter/test_moe_ag_traffic.py", line 660, in <module>
[rank32]:     perf_result_flux = perf_flux(
[rank32]:   File ".../torch/utils/_contextlib.py", line 116, in decorate_context
[rank32]:     return func(*args, **kwargs)
[rank32]:   File ".../test/python/moe_ag_scatter/test_moe_ag_traffic.py", line 266, in perf_flux
[rank32]:     op.forward(
[rank32]: RuntimeError: .../gemm_grouped_v2_ag_scatter.cc:1147 Check failed: 215747((M_this_ep)) <= 163840((this->max_recv_ntokens_))a2av recv buffer overflow; raise FLUX_A2AV_MAX_RECV_NTOKENS
```
The other 63 ranks hit a **second, different** check site:
```
.../gemm_grouped_v2_ag_scatter.cc:1262 Check failed: 259218((max_col)) <= 163840((this->max_recv_ntokens_))a2av compress recv overflow; raise FLUX_A2AV_MAX_RECV_NTOKENS
```
Repro:
```
srun --jobid=57010416 --nodes=16 --ntasks-per-node=1 --gpus-per-node=4 ./launch.sh \
  test/python/moe_ag_scatter/test_moe_ag_traffic.py \
  --traffic_matrix /pscratch/sd/y/yufeid/workspace/andrewy/a2av_test_matrices/generated/w64x4_trace-3ca604_b32_k8_idrealmoesweep16n.txt \
  --comm_pattern a2av_hier_compress --topk 8 --G 128 --H 4096 --chunk_bytes 8192 \
  --ffn_hidden_size 4096 --dtype bfloat16 --iters 10 --warmup_iters 5 --sm_margin 8 --skip_correctness
env {"CUDA_DEVICE_MAX_CONNECTIONS":"8","FLUX_A2AV_LB_UNION":"1",
     "FLUX_A2AV_MAX_RECV_NTOKENS":"163840","FLUX_A2AV_MAX_RELAY_NTOKENS":"163840",
     "FLUX_A2AV_MAX_STAGE_NTOKENS":"163840","FLUX_SWEEP_ISOLATED_ITERS":"1",
     "FLUX_TEST_DETERMINISTIC":"0","NVSHMEM_SYMMETRIC_SIZE":"6G"}
```

### b64 (cap = 262144)
Cell `.../cells/hier_compress_lb_union_trace-37d38e_b64_k8_isolated/`. Distinct check failures:
```
.../gemm_grouped_v2_ag_scatter.cc:1262 Check failed: 518504((max_col))    <= 262144((this->max_recv_ntokens_))a2av compress recv overflow; raise FLUX_A2AV_MAX_RECV_NTOKENS   [ranks 0,1,3..17,19..31,33..63]
.../gemm_grouped_v2_ag_scatter.cc:1147 Check failed: 431792((M_this_ep))  <= 262144((this->max_recv_ntokens_))a2av recv buffer overflow                                        [rank32]
.../gemm_grouped_v2_ag_scatter.cc:1147 Check failed: 298838((M_this_ep))  <= 262144((this->max_recv_ntokens_))a2av recv buffer overflow                                        [rank2]
.../gemm_grouped_v2_ag_scatter.cc:1147 Check failed: 295600((M_this_ep))  <= 262144((this->max_recv_ntokens_))a2av recv buffer overflow                                        [rank18]
```
Repro (same as above with `--traffic_matrix .../w64x4_trace-b3d85b_b64_k8_idrealmoesweep16n.txt`), env identical except `FLUX_A2AV_MAX_{RECV,RELAY,STAGE}_NTOKENS = 262144` and `NVSHMEM_SYMMETRIC_SIZE = 10G`.

### The formula, confirmed
`sweeps/sweep.py:262` `scale_knobs()` — **no `world_size`/`nnodes` term at all**:
```python
row_chunks = budget_mib * (1 << 20) * topk // chunk_bytes
cap = max(163840, math.ceil(4 * row_chunks / 8192) * 8192)
```
b32/k8/8192 → `row_chunks=32768`, `4*row_chunks=131072` → floored to **163840**; observed demand **259218** (1.58×).
b64/k8/8192 → `row_chunks=65536` → **262144**; observed demand **518504** (1.98×).
The fixed `4×` skew allowance is a W-independent constant, so headroom erodes as W grows. C++ check sites in the current debug worktree: `src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc:1147`, `:1262` (and a third at `:2069`).

---

# 6. Failure (d) — 8n `hier_compress_lb_union` b64 "hang". **It is NOT a hang — it is the same overflow, with a deadlocked teardown.**

Capsule `20260815-081416_perlmutter_bb881f76`, cell `hier_compress_lb_union_trace-37d38e_b64_k8_isolated`, status `stuck`, `notes = "stuck, retry 1: stuck"`, no exit code, 8n / W=32, `--skip_correctness`.

**Rank 16 died with the identical C++ check**, in *both* attempts:
`.../cells/hier_compress_lb_union_trace-37d38e_b64_k8_isolated/torchrun/none_zi60amav/attempt_0/0/stderr.log`
```
/global/.../src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc:1147 Check failed: 271685((M_this_ep)) <= 262144((this->max_recv_ntokens_))a2av recv buffer overflow; raise FLUX_A2AV_MAX_RECV_NTOKENS
[rank16]: Traceback (most recent call last):
[rank16]:   File ".../test/python/moe_ag_scatter/test_moe_ag_traffic.py", line 660, in <module>
[rank16]:     perf_result_flux = perf_flux(
[rank16]:   File ".../torch/utils/_contextlib.py", line 116, in decorate_context
[rank16]:     return func(*args, **kwargs)
[rank16]:   File ".../test/python/moe_ag_scatter/test_moe_ag_traffic.py", line 266, in perf_flux
[rank16]:     op.forward(
[rank16]: RuntimeError: .../gemm_grouped_v2_ag_scatter.cc:1147 Check failed: 271685((M_this_ep)) <= 262144((this->max_recv_ntokens_))a2av recv buffer overflow; raise FLUX_A2AV_MAX_RECV_NTOKENS
```
The `.attempt0` sibling dir contains the **byte-identical** message (`271685 <= 262144`). Only 271685/262144 = **1.036×** over — which is why 8n manifests marginally (one rank) while 16n blows up on nearly all ranks.

**Last lines before the watchdog kill** — the surviving 31 ranks are stuck in NVSHMEM/NCCL collectives; the only rank that got past setup logged this and stopped (`.../none_63z444gu/attempt_0/0/stdout.log`, last line, no further output for 250 s):
```
Per-rank gemm rows: [77476, 169544, 69220, 58408, 95647, 17948, 54039, 34085, 22845, 209103, 43878, 55779, 12788, 72540, 11630, 26968, 271685, 16284, 172867, 73404, 91796, 13402, 20188, 12664, 62890, 63827, 31942, 63611, 47465, 40922, 18381, 63926]
...
a2av relay balance: sum of per-round max sender bytes identity 2962432000 -> balanced 2943901696 (0.994x)
```
Note element **[16] = 271685** — exactly the value in the check. All other ranks' `stdout.log` stop at `after flux_shm initialization` (61 bytes).

`srun.log` tail (watchdog kill):
```
[2026-08-15T08:22:20.530] error: Failed to destroy CXI Service ID 5 (cxi0): Device or resource busy
srun: error: nid002096: task 1: Killed
srun: Terminating StepId=57008900.1
srun: error: nid008528: task 7: Killed
srun: error: nid002405: task 2: Killed
srun: error: nid001856: task 0: Killed
srun: error: nid003249: task 3: Killed
srun: error: nid003873: task 4: Killed
srun: error: nid008525: task 6: Killed
srun: error: nid008228: task 5: Killed

+ killed by runner: stuck
```
Start/end 08:18:26Z → 08:22:21Z (235 s, consistent with the 250 s idle watchdog).

**Repro:**
```
srun --jobid=57008900 --nodes=8 --ntasks-per-node=1 --gpus-per-node=4 ./launch.sh \
  test/python/moe_ag_scatter/test_moe_ag_traffic.py \
  --traffic_matrix /pscratch/sd/y/yufeid/workspace/andrewy/a2av_test_matrices/generated/w32x4_trace-1a6a90_b64_k8_id001.txt \
  --comm_pattern a2av_hier_compress --topk 8 --G 128 --H 4096 --chunk_bytes 8192 \
  --ffn_hidden_size 4096 --dtype bfloat16 --iters 10 --warmup_iters 5 --sm_margin 8 --skip_correctness
env {"CUDA_DEVICE_MAX_CONNECTIONS":"8","FLUX_A2AV_LB_UNION":"1",
     "FLUX_A2AV_MAX_RECV_NTOKENS":"262144","FLUX_A2AV_MAX_RELAY_NTOKENS":"262144",
     "FLUX_A2AV_MAX_STAGE_NTOKENS":"262144","FLUX_SWEEP_ISOLATED_ITERS":"1",
     "FLUX_TEST_DETERMINISTIC":"0","NVSHMEM_SYMMETRIC_SIZE":"10G"}
```
git sha `17bb16ef289995e9f3a3bbba804f8255ff8f7884`.

**So (c) and (d) are one bug**, and the secondary bug is that a `FLUX_CHECK` abort on one rank does not tear down the job — it deadlocks the other 31.

---

# 7. Failure (e) — `fast` at b64. Two different things, and one claim I could not substantiate.

### 4n `fast` b64 — NVSHMEM team-creation failure (documented)
Capsule `20260815-012649_perlmutter_1dee50c2`, cell `fast_trace-0c9698_b64_k8_e2e`, `failed`, exit 143, 4n / W=16. All 4 node-leader ranks, file e.g.
`/pscratch/sd/y/yufeid/workspace/andrewy/sweep_data/20260815-012649_perlmutter_1dee50c2/cells/fast_trace-0c9698_b64_k8_e2e/torchrun/none_n4qj_7_f/attempt_0/0/stdout.log`
— the **entire file is one line**, `stderr.log` is empty:
```
Failed, NCCL error /tmp/nvshmem.rMeZStM/nvshmem_src/src/host/team/team_internal.cpp:531 'unhandled cuda error (run with NCCL_DEBUG=INFO for details)'
```
(same line in `none_ptjttaqa`, `none_3_s0yc7e`, `none_1kiepk2f`). `srun.log` root cause: `rank : 4 (local_rank: 0)`, `host : nid001124-hsn0`, `exitcode : 1`. Note `NVSHMEM_SYMMETRIC_SIZE=16G` — b64×k8 = 512 post-topk MiB, which pins `sym_g` at the platform's `sym_size_max_g: 16` ceiling; on a 40 GB A100 that is where NVSHMEM team creation falls over.

Repro (note the different launcher, `launch_fast.sh`):
```
srun --jobid=56981710 --nodes=4 --ntasks-per-node=1 --gpus-per-node=4 ./launch_fast.sh \
  test/python/moe_ag_scatter/test_moe_ag_fast_baseline.py \
  --traffic_matrix /pscratch/sd/y/yufeid/workspace/andrewy/a2av_test_matrices/generated/w16x4_trace-041f16_b64_k8_id001.txt \
  --topk 8 --G 128 --H 4096 --chunk_bytes 8192 --ffn_hidden_size 4096 \
  --dtype bfloat16 --iters 10 --warmup_iters 5 --sm_margin 8
env {"FLUX_SWEEP_RECORD_DIR":"<cell>/records","FLUX_TEST_DETERMINISTIC":"0","NVSHMEM_SYMMETRIC_SIZE":"16G"}
```

### 8n `fast` b64 — **no such cell exists.** Handoff item #4 is unsupported for 8n.
I scanned `variant == fast && status != ok` across **every** committed capsule and every `fast_*` cell dir under the data root. Results:
- 4n/W=16, b64: the one failure above (`20260815-012649`).
- 16n/W=64: 9+9 failures on 2026-08-05 (`20260805-141821_perlmutter_932a88e2`, `20260805-144344_perlmutter_db245058`) — the previously-known 16n instability, budgets 2/8/32 only.
- 8n/W=32: only two `stuck` cells at **b2** and **b8** on 2026-08-05 (`20260805-154528_perlmutter_b0b6dadb` `fast_uniform_b2_k8_e2e`, `20260805-161216_perlmutter_db958e5a` `fast_fanoutskew-349186_b8_k4_e2e`).
- **No 8n `fast` cell at b64 anywhere**, in any capsule or under `<data_root>/*/cells/`. No 8n capsule from 2026-08-15 (`074430`, `075822`, `081416`) includes the `fast` variant at all.

Treat "fast fails at b64 at 8n" as unverified in the record.

---

# 8. Two extra findings not in the handoff

**8a. The 4n and 8n `b64` failures of `allgather` and `hier_compress_lb_union` (correctness ON) are CUDA OOM, not the recv-overflow.** These are the cells in `20260815-011216_perlmutter_1a0e0037` (4n) and `20260815-074430_perlmutter_3ccbcb47` (8n) marked `failed`. Verbatim (8n allgather b64, rank 0):
```
[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 GiB. GPU 0 has a total capacity of 39.49 GiB of which 4.29 GiB is free. Process 1056878 has 418.00 MiB memory in use. Including non-PyTorch memory, this process has 33.96 GiB memory in use. ... Of the allocated memory 18.09 GiB is allocated by PyTorch, and 32.00 MiB is reserved by PyTorch but unallocated.
```
4n variant is the same with `Tried to allocate 8.00 GiB` / `11.22 GiB is allocated by PyTorch, and 6.88 GiB is reserved by PyTorch but unallocated`. This is the torch **reference** path allocating the full unsharded tensor — which is precisely why the same cells pass under `--skip_correctness` in `20260815-012546` / `20260815-075822`. Distinct from the 16n recv-overflow.

**8b. `hier_compress_lb_union` passes allclose at 16n but is never bitwise-exact.** In `20260815-105733` every `hier_compress_lb_union` cell has `correct_allclose=1, correct_bitwise=0`, while `eplb` and `moonep_nvshmem_getmem` are `1/1`. Possibly expected (different reduction order), but worth confirming it is intentional before leaning on it as the "clean" control arm in handoff item #2's argument.

---

# 9. Quick-reference index of every file cited

| What | Path |
|---|---|
| Platform yaml | `/global/u1/y/yufeid/workspace/changchen/andrewy/flux/.claude/worktrees/debug/sweeps/platforms/perlmutter.yaml` |
| Knob formula | `.../worktrees/debug/sweeps/sweep.py:262` (`scale_knobs`) |
| C++ check sites | `.../worktrees/debug/src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc:1147, :1262, :2069` |
| W≤32 assert | `.../worktrees/debug/test/python/moe_ag_scatter/test_moe_moonep_fused_traffic.py:427` |
| 16n capsule (b1–b16, corr ON) | `.../worktrees/debug/sweeps/results/runs/20260815-105733_perlmutter_244825e3/` |
| 16n b32/b64 raw logs (**uncommitted**) | `/pscratch/sd/y/yufeid/workspace/andrewy/sweep_data/20260815-111907_perlmutter_007e9586/cells/` |
| 16n earlier partial (**uncommitted**) | `/pscratch/sd/y/yufeid/workspace/andrewy/sweep_data/20260815-105146_perlmutter_fe2062c5/cells/` |
| moonep 16n assert | `.../20260815-105733_perlmutter_244825e3/cells/moonep_fused_push_auto_gated_trace-37d38e_b1_k8_isolated/torchrun/none_dw8yuams/attempt_0/0/stderr.log` |
| allgather 16n b8 mismatch | `.../20260815-105733_perlmutter_244825e3/cells/allgather_trace-37d38e_b8_k8_isolated/torchrun/none_jy70zcgw/attempt_0/3/stderr.log` |
| allgather 16n b1 mismatch + dump crash | `.../20260815-105733_perlmutter_244825e3/cells/allgather_trace-37d38e_b1_k8_isolated/torchrun/none_v9c0ndeo/attempt_0/3/stderr.log` |
| allgather 16n b16 mismatch | `.../20260815-105733_perlmutter_244825e3/cells/allgather_trace-37d38e_b16_k8_isolated/torchrun/none_yk6j9ohh/attempt_0/3/stderr.log` |
| hier 16n b32 overflow | `.../20260815-111907_perlmutter_007e9586/cells/hier_compress_lb_union_trace-37d38e_b32_k8_isolated/torchrun/none_qj0hx5vw/attempt_0/0/stderr.log` |
| hier 16n b64 overflow | `.../20260815-111907_perlmutter_007e9586/cells/hier_compress_lb_union_trace-37d38e_b64_k8_isolated/` |
| hier 8n b64 "hang" | `.../20260815-081416_perlmutter_bb881f76/cells/hier_compress_lb_union_trace-37d38e_b64_k8_isolated/torchrun/none_zi60amav/attempt_0/0/stderr.log` (+ `none_63z444gu/attempt_0/0/stdout.log` for last-lines, + `.attempt0/` sibling) |
| fast 4n b64 | `.../20260815-012649_perlmutter_1dee50c2/cells/fast_trace-0c9698_b64_k8_e2e/torchrun/none_n4qj_7_f/attempt_0/0/stdout.log` |

**Revised priority suggestion:** #1 and #3/#4-of-the-handoff are now solved on paper (a W≤32 assert; a W-independent `scale_knobs` cap). The only remaining genuine unknown is the `allgather` W=64 correctness bug in §4 — contiguous bands of output rows, sometimes exactly zero. That is now the sole item that needs new experiments.
