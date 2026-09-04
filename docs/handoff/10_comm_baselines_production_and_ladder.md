# 10 — Comm-only baselines: production b1–b64 sweeps + the 8n/16n/32n ladder

Handoff written 2026-08-22 (session: rule-5 comm-driver conversion → K3 l01
bring-up → optimization/fairness pass). Read `00_START_HERE.md` first if you
are new to the repo. Everything below is committed on **local main** (never
pushed): code through `cec4053`, capsules through `8774048`. The installed
`.so` matches main as of `cec4053` + the sort-free-plan build (rebuild anyway
before any capsule — rule 4).

---

## Part A — Running properly validated comm-only sweeps (K3 + qwen, b1–b64)

### A.1 State: what is already validated

All five comm-only arms are **rule-5 converted** (`timing_accounting=
per_iter_gpu`): the per-iteration timed window contains the routing allgather
(`plan_comm_ms`) + ALL routing-derived metadata derivation on GPU (`plan_ms`)
+ the op window (`e2e_ms`); `total_ms = plan_comm + plan + e2e` is the
quotable isolated number. Setup computes the same metadata once, untimed,
purely as bitwise drift guards — every launch self-checks.

| arm | driver | notes |
|---|---|---|
| `allgather` / `l01_allgather_dense` | flux/l01 | stock Comet; `FLUX_RS_BLOCKS=20` canonicalized (env boundary) |
| `hier` / `l01_lbunion_hier` | flux/l01 | a2av 6/6/4 CTA env canonicalized |
| `hier_compress_lb_union` / `l01_lbunion_compress` | flux/l01 | fully sort-free plan (scd arithmetic); a2av 6/6/4 |
| torch reference (rides flux cells) / `l01_torch` | — | `torch_ref_impl=local_slice_scatter`; index_add reference; heartbeats |
| `fast` / `l01_fast` | fast/l01_fast | real routing since 08-21; two flash_comm_t (vendored refcount patch `scripts/fast_two_instance.patch`, applied by `scripts/build_fast.sh`); e2e-only, >=2 nodes |

Reference capsules (all committed): layer0 `20260821-104600`/`-105153`;
l01 `20260821-133153` (b8) + `-161617`/`-163902` (b32); post-tuning
`20260822-005041` (the current standings source); nsys forensics
`20260821-235942`, `20260822-003304`. Current K3 4n isolated totals
(b8/b32): compress 14.72/55.88 < hier 17.31/67.60, dense 18.46/62.52;
torch 35.24/89.37; fast e2e 37.65/133.71.

### A.2 The production grid

Shapes via `--shape` presets (`sweep.py SHAPE_PRESETS`): `k3` (G896/topk16/
H3584/ffn3072/chunk7168, n_split_l1=7, trace model Kimi-K3-synth) and
`qwen3` (G128/topk8/H4096/ffn1536/chunk8192 — the fast 4n-validation lane).
Budgets are the **nominal byte ladder b1,2,4,8,16,32,64** (2026-08-21
canonicalization: `gen_matrix.budget_tokens` rounds to topk-multiple tokens;
historical labels bit-exact; K3 label error <=1.6% at b1/b2, recorded as
`effective_budget_bytes` in meta.json + cells.csv). Trace semantics:
**`sem=homog`, canonical pool `mmlu/professional_law`** (pernode retired —
capsules before 08-21 evening are the last pernode ones);
`livecodebench/execution` is the designated topic-shift alternate.

Template specs: `sweeps/specs/k3_rule5_smoke.yaml` (layer0 arms),
`sweeps/specs/k3_l01_smoke.yaml` (l01 arms; timeout 2400 / idle 900).
Extend `budgets_mib` to the full ladder and bump `timeout_s` headroom for
b64 first-light. Run flow (per CLAUDE.md): `source ./module.sh`, rebuild
main first with the **cudatoolkit 24.5/12.4 pins** (memory:
perlmutter-cudatoolkit-drift) and copy `python/flux/lib64/* -> lib/`
(lib64-shadows-lib gotcha), then `salloc --qos interactive -C gpu --account
m5350_g -N 4 --no-shell` (m4243_g is exhausted) and
`python sweeps/sweep.py run --spec ... --jobid ...`; scancel immediately
after; the runner prints the capsule commit command.

### A.3 Per-arm operational caveats (all learned the hard way)

- **fast**: e2e mode only (isolated cells silently skipped); needs >=2
  nodes; consumes REAL trace routing since 08-21 (dealer=1 is
  ablation-only). At b32+ its correctness reference OOMs beside the FAST
  heap — run large-budget fast cells with `skip_correctness` (the
  wire-layout drift guards still run every launch; full correctness is
  proven at b8). `l01_fast` heap = 2x
  `fast_sym_size` clamped; when clamped the argv builder auto-shrinks
  `--capacity_mib` to clamp/16 (both formulas must stay in sync — they live
  in `build_cell_env` and `build_cell_cmd`).
- **torch**: never silent anymore (per-window heartbeats) — but at b64 its
  reference cost is first-light; if it exceeds ~20 min/cell, invoke the
  standing user directive: drop torch+gemm from the campaign specs rather
  than burning node-hours.
- **b64 first-light** (never run anywhere): check `fast_sym_size` vs the
  `sym_size_max_g: 16` platform clamp; dense/torch memory fits per the
  local_slice_scatter math; expect zero-split experts at b1/b2 on real
  traces (fixed code path, but preflight-audit `bincount` of the routing
  anyway — l1-cascade lesson).
- **Plan-share footnotes**: quote `plan_comm_ms + plan_ms` share per arm
  next to totals (SCHEMA rule); compress is ~5–10% (documented residual:
  l0 derive + e_of searchsorted + small kernels), everything else <=6.5%.

### A.4 Never-mix registry (cite the boundary when quoting)

1. `timing_accounting=per_iter_gpu` — flux+fast drivers 2026-08-21, l01
   same day. Pre-boundary capsules are legacy.
2. `torch_ref_impl=local_slice_scatter` — torch rows never compare across.
3. Nominal byte ladder — b7/b28 interim K3 labels (capsules 104600/105153)
   vs b8/b32 nominal: different matrices, never conflate.
4. Env canonicalizations — `FLUX_RS_BLOCKS=20` (dense),
   `FLUX_A2AV_RS_{PACK,REDUCE,PRERED}_BLOCKS=6/6/4` (a2av arms): never
   byte-compare env_json across the flips.
5. Rule 4 as always: one binary per capsule; rebuild main before campaigns.
6. **Method rule (user directive)**: when tuning any arm's CTA/SM knob,
   sweep the sibling arms' equivalent knobs before quoting standings — the
   dense FLUX_RS_BLOCKS win briefly and falsely flipped b32 leadership
   while compress sat on starved defaults.

### A.5 Interpretive guardrails for the 4n ledger

- K3 at 4n is the **incidence-saturation regime** (topk-16: ~every token
  touches every node) — dedup/placement mechanisms are structurally
  understated; 4n supports baseline characterization and bring-up, the
  showcase claims belong to 8n+.
- FAST characterization (timeline-verified, capsule 163902 + $PSCRATCH
  .../fast_nsys): its intra-then-inter node rebalancing IS active (heavy
  senders finish early; peers carry forwarded load); per-direction floor =
  max-NODE inter bytes / 4 NICs @ ~26 GB/s (~15–16.5 ms at b32); measured
  1.9x (dispatch) and ~3.3x (combine) above floor = staging + synchronous
  round barriers + host-proxy issue, worse on the combine because
  node-level expert skew makes whole nodes stragglers (what expert
  placement exists to fix). NCCL's pipelined dense allgather beats even
  FAST's floor at 4n; prediction: gap narrows at higher NN.

---

## Part B — Warnings and lessons for 8n/16n/32n allocations

### B.1 Slurm/allocation discipline

- Interactive QOS caps at **4 nodes/4 h**; 8n+ needs `-q regular`. Never
  sbatch (user preference): `salloc -q regular -N <n> --no-shell` submitted
  in the background (it queues; the grant arrives as a task notification),
  then a **pre-baked unattended driver script** runs the entire rung in one
  session. One queue wait per rung.
- Take jobids ONLY from your own salloc output (parallel-session scancel
  collisions, 2026-08-14).
- **Check `squeue --me` after ANY interruption**: killing a runner task
  also kills its chained `scancel`, orphaning the allocation (55 idle
  minutes / ~3.7 node-hours lost this session).
- scancel the moment the intended jobs finish. Budget estimate: ~12
  node-hours per 8n rung, ~25 for 16n.

### B.2 Before burning any at-scale hours (Tier 0/1, no or 4n GPU)

- **CPU geometry matrix** (login, free): the plan builders are
  device-agnostic — run the `_dev` twins + CPU builders + the
  `derive_fast_meta` simulation at W=32/64/128, NN=8/16/32. This class of
  test would have caught the int32-offset cliff without a single GPU.
- **Static limits audit**: `include/flux/args/moe_gather_rs.h` has
  `kA2AVMaxWorld = 64` and `kA2AVMaxNodes = 16`. 16n (W=64, NN=16) sits
  exactly AT both; **32n (W=128, NN=32) exceeds both** — raise + audit
  before any 32n attempt.
- **Per-rank geometry twins at 4n**: replicate 16n/32n experts-per-rank
  (14, 7) via G-scaling in the unit tests; force zero-splits with
  `--dist random_with_first_k_experts` (empties are fixed but re-verify at
  each geometry — the wrong-output class hid at eid≈97, a NON-round
  boundary; bisect empirically before reading code, and remember non-round
  boundaries smell like integer overflow).
- **Loud-deadlock burn-in**: set `FLUX_A2AV_RS_SPIN_LIMIT>0` on all
  bring-up/smoke cells — converts silent device spins into immediate
  aborts (epic-v3 watchdog).

### B.3 Hang semantics (hard-won this session)

- The runner's idle-kill watches **staging file mtimes**. Arms that are
  silent while healthy get killed as "stuck": the torch arm needed
  per-window heartbeats (a silent-but-healthy b32 cell ran 16+ min; no
  timeout value distinguishes that from a hang — emit progress instead).
- "stuck" is often **one rank raising inside `exec_in_rank_order` while
  the others block** — read the per-cell torchrun logs in staging (they
  survive; node-local `/tmp` debug dumps do NOT — always set
  `FLUX_DEBUG_DUMP_DIR` to $PSCRATCH).
- **Killed steps can wedge nodes** (CXI "Device or resource busy") and
  poison subsequent cells on the same allocation with instant failures.
  Mitigations: order risky/slow arms LAST in `--variants`; treat a burst
  of 0-second failures as a dead allocation (also check whether the wall
  simply expired — "Slurm job has expired" in srun.log).
- Cell timeouts: `timeout_s`/`idle_timeout_s` contain hangs at ~6 min
  each including retry; budget wall time as (worst-case cells x
  timeouts), not (expected runtime).

### B.4 Scale-dependent tuning (do NOT reuse 4n constants)

- **SM-budget knobs are shape- AND scale-dependent** (see memory
  `overlap-sm-budget-tuning`): `FLUX_RS_BLOCKS` (dense) and
  `FLUX_A2AV_RS_{PACK,REDUCE,PRERED}_BLOCKS` (a2av) are bandwidth-engine
  CTA budgets. PACK/PRERED work grows with remote-node count (up to
  min(topk, NN-1) segments/token); the 4n knees (20 and 6/6/4) will move
  right at every rung. Probe knee±1 predicted from the traffic model at
  each rung's bring-up — a 2–3 point confirm, not a full sweep — and
  re-canonicalize per-N (env flips = never-mix boundaries).
- NVSHMEM heap sizing is formula-driven per cell (never hand-tune); at
  scale re-verify `fast_sym_size`/l01 merges against the 16G clamp.

### B.5 Rung-specific gates

- **8n**: no known blockers. First rung; also the first place dedup
  headroom returns (incidence de-saturation begins).
- **16n**: (a) **FAST 64-rank segfault** (l0algos, un-root-caused) —
  triage FIRST on the 16n allocation with the segv backtrace shim at
  `$PSCRATCH/workspace/andrewy/debug/segvbt.so`, dispatch-only smoke before any fast grid cell;
  if libflash is at fault, patch via `scripts/fast_two_instance.patch`-
  style vendored patches (FAST is a submodule — patches must live in the
  parent repo and be applied by `scripts/build_fast.sh`); (b) the W64
  ballot-cliff fix has NEVER been closure-tested at 16n — the 16n smoke
  IS the closure run.
- **32n**: blocked on the B.2 static limits + everything above.

### B.6 Debug method rules that paid off

1. Bisect the failing axis empirically before reading code; non-round
   boundaries = overflow.
2. When a "hang" appears: check sacct step durations first (0-second
   failures = environment; N-minute = idle-kill; check for wall expiry).
3. Every derivation change ships with a bitwise drift guard against the
   previous formulation (the `_dev` twins / CHECK_IDENTITY pattern) —
   the sort-free plan landed with zero correctness risk because both
   formulations ran side-by-side under `FLUX_A2AV_RS_CHECK_IDENTITY=1`
   in validation (and NEVER in quoted capsules — verifiable in env_json).
4. Per-rank metric rows in capsules are a timeline substitute: the FAST
   hot-sender mechanism fell out of per-rank `comb_wire_ms` before any
   nsys was needed.
