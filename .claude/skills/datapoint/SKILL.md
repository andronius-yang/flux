---
name: datapoint
description: Run an authentic l01 data-point collection campaign (7 baselines x K2/Qwen x 4n/8n/16n x b1-b64, isolated dispatch+combine) with per-topology lane subagents, and produce the plan-inclusive tidy dataset. Use when the user asks for an authentic data-point campaign / "datapoint gather" across baselines, topologies, and token budgets.
---

# Authentic l01 data-point campaign

Reproduces the 2026-08-24 campaign (docs/handoff/18_authentic_l01_campaign.md
= authority; datacamp handoff 15 = parent protocol). One campaign = 42
capsules: 7 baselines x {k2, qwen} x {4n, 8n, 16n}, budgets b1-b64, isolated
mode ONLY, 5 warmup + 10 timed iters, s1-canon inputs (SCHEMA rule 10). The
ONLY axes that vary are baseline / model / topology / budget — no arms, no
front/back, no extra knobs.

## The 7 lanes (exact --variants keys; run torch first per model as anchor, moonep last)

| Report name | variant key | note |
|---|---|---|
| Torch+GEMM | `l01_torch` | anchor arm |
| COMET | `l01_allgather_dense` | dense l1 ns=2 (canon 8/24) |
| Slipstream | `l01_slipstream` | v2 canon; requires FLUX_A2AV_SLIPSTREAM2_TAG |
| EPLB | `eplb_l01` | |
| EPIC | `epic_l01_hc_m1` | staged l1 ns=1 (canon 8/24) |
| MoonEP | `moonep_l01_nvshmem_getmem` | slowest; K2 ~70-85 s/cell |
| PLL | `llc_l01_s1` | ns=1; needs the recv-bound fix (3e6e8ed+) |

FAST is EXCLUDED (broken as of 8/24; do not touch 3rdparty/FAST or its files).

## Pre-flight (all BEFORE any salloc)

1. **Rule-4 binary audit**: `strings python/flux/lib/libflux_cuda_ths_op.so`
   must contain every tag the variants `require` (SLIPSTREAM2, RS_MSPLIT,
   RS_BUCKET, WAVE_PACK, RS_NSPLIT_512, NSPLIT_HONOR, RS_CTA_1086,
   DISPATCH_ONLY, RS_WIRE_STREAMS_DEFAULT16); verify no src file is newer
   than the .so (`find src -newer <so>`), and confirm the tree is
   BUILD-FROZEN for the campaign (ask the user if another session is active).
   One binary serves all capsules — never rebuild mid-campaign.
2. **Dry-run capacity audit** (handoff 17): `source ./module.sh` (system
   python3.6 cannot run sweep.py), then for all 6 specs
   `datacamp_{k2,qwen}_{4n,8n,16n}.yaml`:
   `python sweeps/sweep.py run --spec <s> --variants <all 7> --dry-run`,
   grep `skipped_capability` and the `_A2AV_SYM_G_AT_CAP` WARNINGs. At-cap
   warnings on epic/llc/eplb are the accepted class (priors ~4x realized);
   anything NEW stops the launch.
3. **Account**: confirm with the user (m5350_g as of 8/24; m4243_g exhausted).
4. **Known pre-skips** (fill cells as pre-skipped WITH REASON, never attempt):
   - torch 8n b64, both models (gathered-bytes OOM/kill, datacamp exit 143)
   - torch 16n b32+b64, both models (gathered-bytes CUDA OOM on 40G)
   - llc/PLL 16n b64, both models (authentic >16G symmetric-heap, hc ctor)
   - eplb 16n qwen b64 (planner [E,129] int64 OOM, non-authentic — fix is a
     rule-4 USER decision; default = pre-skip)
   Implement via `--budgets-mib` subsets per invocation.

## Execution format (handoff 15 §3 + the 8/24 corrections)

One subagent per topology, each owning its allocations; orchestrator owns
cross-checks + aggregation. Shared status file, one line at EVERY event
(user rule: status-update cadence). Lane briefs must include: exact
invocations, allocation policy + fallbacks, anchors, pre-skips, hard rules,
report format — lanes can never be given new invocation kinds later.

Per-invocation (one capsule per model x variant, 14 per topology):
`python sweeps/sweep.py run --spec sweeps/specs/datacamp_<model>_<N>n.yaml
 --variants <key> --jobid <OWN jobid> [--budgets-mib <subset>]`

Allocation policy (validated on a 100%-full machine):
- 4n: `salloc --qos interactive -C gpu -N4 -t 150 --no-shell -I120`;
  regular fallback after ~20 min of rejects. (Interactive caps: 4 nodes/4h,
  two concurrent 4n jobs per user.)
- 8n: debug QOS 30-min windows (`-t 30`) — grants in seconds; walltime gate
  before each invocation (>=12 min moonep-K2, >=8 min others); scancel the
  window when the next invocation won't fit; loop. Debug REJECTS N=16.
- 16n: regular ONLY; submit IMMEDIATELY at lane start (`-t 110`, backfill-
  friendly), 15-45 min queue age typical; walltime gate >=20/10 min;
  resubmit for any remainder.

### MISTAKES LEDGER — every one of these actually happened; enforce them
- **NEVER park a lane on a background-notification wait for a salloc grant
  or a runner completion.** Both 8/24 incidents (11.7 nh idle-burn) were
  missed wakes: a granted debug window idled 30 min to TIMEOUT, and a
  finished 16n invocation left 16 nodes idle 29 min. Run invocations as
  FOREGROUND commands; verify grants by polling `squeue -j <id> -h -o %T`.
  Orchestrator cross-check: if a lane is silent, compare `sacct -j <id>`
  step activity against the status log before believing "still running".
- jobid ONLY from your own salloc output, never `squeue --me` (8/23
  cross-session scancel collisions; parallel lanes + other sessions exist).
- scancel the instant the last capsule lands — idle grants are the #1
  node-hour hazard (8/23 stayed at 27 nh only through release discipline).
- Unique lane-prefixed filenames in any shared scratch dir (8/23 collision).
- Watchdogs keyed on jobid go stale across invocations and fire false
  WATCHDOG_HANG on finished logs — key them on the live invocation.
- No rebuilds, no repo edits, no debugging of failed cells (capture reason,
  tail the cell's srun.log/torchrun rank logs, move on; spec retries=1).
- Gate-first: first invocation is l01_torch on K2; compare b8 vs the anchor
  table (+/-25%; MoonEP sanity +/-30% on total_ms) BEFORE fanning out.
  Anchors (b8, K2/Qwen, e2e_ms): 4n 14.8/12.9, 8n 19.7/19.2, 16n 31.0/32.8
  (from handoff 18 tidy; torch and moonep are the stable-arm anchors).

## Aggregation + reporting (the part users care about — get it right)

Conventions (handoff 15, implemented in `aggregate.py` NEXT TO THIS SKILL —
run `python3 .claude/skills/datapoint/aggregate.py <run_ids...>` from repo
root; plain python3 suffices):
- e2e/plan/plan_comm/total = per-iter MAX across ranks, MEDIAN of 10 iters.
- l0/act/l1 = the CRITICAL rank's spans (rank with max e2e that iter), so
  l0+act+l1 == e2e. Never sum per-span maxes (over-counts on MoonEP).
- MoonEP emits no e2e/l0/l1: e2e = total - plan_comm - plan;
  l0 = pack+comm+scatter+prefetch+gemm; l1 = gemm2+cpack+comb+acc.
- **HEADLINE NUMBERS ARE PLAN-INCLUSIVE `total_ms`** — the realistic
  post-gating duration (SCHEMA rule 5: only the gating-metadata exchange is
  untimed; plan derivation is per-batch and cannot be cached). Keep e2e_ms
  as the mechanism column, but any table answering "how fast is arm X" uses
  total_ms. This shifted 8/24 rankings: EPLB/EPIC carry 2.5-4.6 ms plan,
  and the 16n PLL~Slipstream e2e tie tips slightly to Slipstream on total.
- Inject the pre-skipped cells as rows (status=pre-skipped + reason) so the
  dataset is exactly |baselines| x |models| x |topos| x 7 budgets.
- Deliverables: THREE files as the next docs/handoff/NN set — the campaign
  md, the tidy CSV, AND the figure-tables CSV generated by
  `python3 .claude/skills/datapoint/figure_tables.py <tidy.csv> <out.csv>`
  (blocks topology-major Qwen-then-K2, one row per figure arm in the paper's
  fixed order incl. placeholder-empty rows for FAST and the ours-combined
  arms, column groups TOTAL|PLAN|LAYER0|LAYER1 with TOTAL leftmost).
  Capsules staged; a human commits (print the git command) unless the user
  authorizes the commit. Report total node-hours with any waste itemized.

Never-mix reminders when comparing to older data: COMET/EPIC/PLL ns flips +
Slipstream v1->v2 + llc recv-bound fix are all 8/24 boundaries vs datacamp;
rule 4 (same binary) governs every in-table comparison.

## SLACK PARITY RULE (2026-08-25, user directive)

All expert-replication-based baselines MUST have their redundant slots
per rank EXPLICITLY SPECIFIED on the command line — never rely on driver
defaults. The campaign default is **2** for every replication baseline
(EPIC, EPLB, UltraEP, llc, OURS); MoonEP keeps its own default selection
(B = E/R prefetch slots — different semantics, do not force to 2). The 2026-08-25 campaign found
OURS was the only arm at R_red=0 (a structural handicap worth 16-24% at
8n); never let this drift again. Exact commands per baseline:

- EPIC / llc arms (epic driver `test_moe_epic_traffic.py`): pass
  `--redundant_per_rank 2` explicitly (driver default IS 2 — pin it anyway
  so a default change can never silently unlevel the field).
- EPLB (`test_moe_eplb_traffic.py`): pass `--redundant_per_rank 2`
  (default 2 — pin).
- UltraEP: `UltraEPConfig.R_red` default 2 — pin via the driver flag where
  exposed.
- OURS (`test_moe_ours_traffic.py`): pass `--redundant_per_rank 2` — i.e.
  use the `_r2` arms (`ours_l01_s1_r2`, `ours_l01_s1_wb_r2`). The bare
  `ours_l01_s1` runs R_red=0 (pre-parity identity, kept for history);
  it is NOT slack-comparable to the other movement baselines.
- MoonEP: its slack is the prefetch slot set `B` (default = E/R, a FULL
  extra set, refilled per iteration) — structurally different semantics;
  do NOT equate B to 2. Record B in the capsule notes and annotate any
  cross-baseline slot-budget comparison.

NEVER-MIX: slack-2 cells vs R_red=0 cells is a hard boundary (2026-08-25);
headline tables must hold slack constant across arms.
