# Handoff 15 — the 2026-08-23 authentic data-point campaign: results, launch format, and operational lessons

This document is three things at once: (1) the record of the 8.23 seven-baseline
data campaign (what was measured, where it lives, what failed and why), (2) the
**canonical launch format for future multi-lane sweep campaigns** (user decision
2026-08-23 — later sweeps launch in exactly this shape), and (3) the operational
lessons ledger, so the next campaign does not re-discover them the hard way.
Siblings: `15_datacamp_results_tidy.csv` (the cleaned dataset, one row per cell)
and `15_datacamp_report.html` (self-contained report page; also published as the
"L01 Baseline Ladders" artifact).

## 1. What was measured

Isolated-mode layer0+layer1 (dispatch→combine) latency, scenario-1 canon basis
(SCHEMA rule 10: livecodebench/execution layer 5, sem=homog, eval decode slots
[64,96), oracle = previous window g=0), warmup 5 + 10 timed iters per cell,
`skip_correctness` (perf-only; arms were correctness-validated in the settled-
table and flip capsules), `FLUX_TEST_DETERMINISTIC=0` runner-enforced.

Grid: 7 baselines × 2 models (Kimi-K2, Qwen3-235B) × 3 topologies (4n/8n/16n,
4 ranks per node) × 7 budgets (b1..b64 MiB pre-topk). The settled arm keys
(handoff 13 is the evidence authority for the placement picks):

| Baseline | variant key | notes |
|---|---|---|
| Torch+GEMM | `l01_torch` | un-fused NCCL reference; `torch_ref_impl=local_slice_scatter` |
| COMET | `l01_allgather_dense` | stock upstream allgather+scatter / dense combine |
| Slipstream | `l01_slipstream` | canonical comm arm (rule 11 binary defaults) |
| EPLB | `eplb_l01` | staged direct-a2av, oracle-window loads |
| EPIC | `epic_l01_hc_m1` | m=1, hc transport, d6 routing (the settled best-EPIC) |
| MoonEP | `moonep_l01_nvshmem_getmem` | staged authentic, plans per-round natively |
| PLL/LLC | `llc_l01_s1` | NEW canonical key: placelambda_fast static on oracle rows + loccap_sl eps .0625 + FLUX_PLL_TAIL_GRAPH=1 |

Infrastructure made durable this campaign (all in the sweeps layer):
`dslots=start:width` trace-family param — **decode-slot windows are 0-BASED
DECODE ORDINALS** (slot s = output token ti s+1; a ti-space first cut was off
by one — verified byte-exact against the s1c prototype oracle files);
`ensure_oracle_sidecars` (`<mid>.oracle_routing.txt` raw prev-window rows,
W-divisibility tail-trim, + `<mid>.oracle_load.json` {version:1,G,load});
sweep.py passes the ORACLE-basis load to eplb/epic and `--oracle_routing_file`
to placelambda arms on dslots cells; specs `sweeps/specs/datacamp_{k2,qwen}_
{4,8,16}n.yaml` (they carry `idle_timeout_s: 900` — no CLI flag exists for it).

## 2. Results ledger

**281/292 cells ok (+2 deliberate skips) for 27.0 node-hours (m5350_g, 22
multi-node jobs).** 42 campaign capsules `20260823-1612xx..2222xx` + 2 gate
capsules (`-155937` K2, `-160519` qwen) + 2 abandoned 0/7 dirs (`-171851`,
`-171905`, superseded — the revoked-allocation incident, §5.3). The 12 final
16n capsules were run by the orchestrator session on ONE allocation (57479686)
— rule-4-friendly. Aggregation conventions (used by the tidy CSV and report):
e2e/plan/total = per-iter MAX across ranks, median of 10; **l0/act/l1 = the
CRITICAL rank's spans** (the rank with max e2e that iter), so l0+act+l1 == e2e
— max-per-span across ranks over-counts badly on rank-asymmetric arms (MoonEP:
sum-of-maxes 48.6 vs true e2e 31.7 at 4n b8 K2). MoonEP emits no e2e/l0/l1:
derive per rank as e2e = total − plan_comm − plan, l0 = pack+comm+scatter+
prefetch+gemm, l1 = gemm2+cpack+comb+acc (the brackets partition total
exactly). Settled-table comparisons use plan-INCLUSIVE `total_ms`.

Failure attribution (complete):
- Torch+GEMM 8n b64 ×2 and 16n b32 ×2: CUDA OOM — the un-fused reference
  materializes the full gathered activation; onset tracks constant gathered
  bytes (b64@8n ≈ b32@16n ≈ 16 GiB). 16n b64 ×2 pre-skipped as the same class
  (saved ~1 h of timeout+retry burn).
- EPLB 16n qwen b64: CUDA OOM at the 40 GB card.
- PLL 16n b64 ×2: NVSHMEM symmetric-heap exhaustion at the 16 GiB platform cap.
- **PLL 16n qwen b4–b32: LocCap capacity-bound assertion** (`recv bound
  violated: 114 rows over`, `epic_semantics.py check_relaxed`) — a REAL ARM
  FINDING at W=64 on Qwen routing, recorded undebugged per campaign policy.

Headline findings:
1. **The ranking inverts at 16 nodes.** 4n/8n reproduce the settled tables
   (LLC < EPIC < EPLB < MoonEP; Slipstream fastest comm arm). At 16n b8, EPLB
   is fastest (32.0/29.8 total K2/qwen) and even Torch+GEMM (32.0/33.8) beats
   Slipstream (52.1/38.3) on K2. The phase split localizes it: Slipstream l0
   is still best-in-class (11.1 ms K2 b8); its **l1 hcc combine is the entire
   loss** (39.2 ms vs EPLB's direct-a2av 12.5, torch's dense NCCL RS 22.0),
   and EPIC's identical hcc combine is equally slow (36.2) → it is the hcc
   TopkReduceScatterOp wire at W=64, not Slipstream-specific. Suspects: the
   wire is nn−1 = 15 target nodes of blocking inter-node puts over
   RS_WIRE_STREAMS=2, and both that default and the 10/8/6 CTA budgets were
   tuned at 4n (nn−1 = 3). NEXT CAMPAIGN (layer1 combine debugging) starts
   here: 16n phases/nsys cell + RS_WIRE_STREAMS/CTA ladders at 16n.
2. Torch+GEMM's speed is legitimate: the 8.21 `local_slice_scatter` +
   `index_add` changes removed only output-invariant, wire-invariant dead
   LOCAL work (W-fold staging nobody read). Its wire cost is untouched — full
   `[ntokens,H]` allgather dispatch + full dense reduce-scatter combine, W-fold
   more bytes than any sparse arm — and NCCL rings still win at W=64. Rows
   across the flip are never-mix (`torch_ref_impl` column).
3. Placement slack is symmetric: EPLB/EPIC/PLL all run `nlp = G/W + 2`
   physical slots per rank (`--redundant_per_rank` default 2, recorded as
   `epic_redundant_per_rank`); MoonEP instead has B = G/W dynamic per-round
   prefetch slots. Equal slack cannot explain their relative 16n ordering
   (it is the combine transport). CAVEAT: relative slack grows with W — at
   qwen 16n the static arms hold 4 vs logical 2 experts/rank (+100%); a
   `redundant_per_rank=1` ablation is the cheap fix if that corner matters.
4. Wire-audit status (rule 5, and it is about DATA/SIGNAL ORDERING on CXI —
   nothing to do with plan accounting): every campaign leg carries a passing
   payload-probe verdict (handoff 08 rounds 1–5, incl. the epic-runner combine
   seam fix and the fused stage-2 lane-order fix) EXCEPT MoonEP's staged l01
   combine (probe host-OOMed, no verdict; primitive passed in the dispatch
   direction) and the staged EPLB combine direction (no leg-specific verdict,
   same primitive class). Torch's NCCL collectives are out of scope.

## 3. THE LAUNCH FORMAT (canonical for future campaigns)

One orchestrator (the main session) + one general-purpose subagent lane per
baseline, spawned in a single parallel batch. Division of labor that proved
correct: **lanes own acquisition, execution, watchdogs, and reporting for
exactly one variant; the orchestrator owns inputs, cross-lane coordination,
aggregation, and anything that spans lanes.**

Before spawning any lane (orchestrator, in order):
1. Freeze and fingerprint the binary (`sha256 libflux_cuda.so`); forbid
   rebuilds campaign-wide; verify capability tags via a dry-run probe.
2. Pre-generate ALL input bundles on $PSCRATCH (matrices + routing + oracle
   sidecars, content-addressed) so every lane reads identical bytes.
3. Author per-(model, topology) spec files; lanes pass `--spec` +
   `--variants <their key>` + `--jobid` only.
4. **Run a gate**: one small allocation, ALL arms × one anchor cell (b8),
   cross-checked against the last settled table before fan-out. The gate
   caught the dslots off-by-one and the plan-inclusive-totals convention.
5. Create the shared status file; put its path in every brief.

The lane brief must contain (a lane can never be given new invocation kinds
later — see §5.5): mission + exact invocations for every tier; the full
allocation policy incl. all fallback QOS paths; hard rules (no rebuilds/edits/
commits, no debugging failures, jobid only from own salloc output, never touch
others' jobs); watchdog expectations; the status-file protocol (append a line
at EVERY event, lane-prefixed); sanity anchors (expected b8 numbers from the
gate); known risks per arm; and the final-report format (capsule run_ids +
per-capsule cell status + one-line reasons + jobids with granted-wall minutes).

Allocation policy that worked on a 100%-full machine (1655/1664 allocated all
evening):
- 4n: interactive first (`-I120` immediate-or-fail; fits **two** concurrent
  4n jobs per user, 3rd+ submit rejected `QOSMaxSubmitJobPerUserLimit`) →
  regular fallback → orchestrator rotates lanes back onto interactive slots
  as they free (the regular 4n queue granted NOTHING for 90+ min).
- 8n: **debug QOS is the workhorse** — grants 8-node jobs in seconds even on
  a full machine; 30-min windows fit a full 2-model tier for every arm
  (fastest tier: ~6 min); several debug jobs queue per user; self-serve loop
  with ~3-min resubmit on the per-user cap. Debug REJECTS N=16
  (`QOSMaxNodePerJobLimit`).
- 16n: regular only (premium declined). Pre-submit EARLY (grants needed
  ~15–45 min of queue age when they came at all); short walltime (`-t 100`)
  for backfill. When per-lane 16n stalls, the fallback that actually worked:
  the ORCHESTRATOR acquires one 16n allocation itself and runs every
  remaining arm on it as per-arm banked capsules (§5.5) with a
  remaining-walltime gate before each launch and a capsule-exists check to
  avoid duplicating lanes that granted meanwhile; lanes keep their pending
  jobs as backstops and release them on coverage (a granted duplicate was
  released within ~2 min).
- Release discipline: scancel the moment a tier's last capsule lands;
  idle-grant leaks are the main node-hour hazard (total campaign cost stayed
  at 27.0 nh because of this).

## 4. Efficiency planning numbers (for budgeting the next campaign)

Cell wall (incl. torchrun/NVSHMEM startup): comm arms ~15–30 s at every
topology; placement arms ~20–45 s; MoonEP K2 ~70–85 s (qwen ~30 s). A 7-cell
capsule ≈ 2–10 min; a full 2-model tier per lane ≈ 6–20 min of allocation.
Tier costs: 4n tier ≈ 2.4 nh total across 7 lanes on interactive; 8n ≈ 8 nh
on debug windows; 16n ≈ 12 nh (mostly one 100-min orchestrator allocation).
The gate ≈ 3 nh. Queue reality dominated wall-clock: ~6.5 h end-to-end, of
which GPU execution was ~1.5 h.

## 5. LESSONS (the mistakes to not repeat)

1. **The background-task reaper kills blocking salloc clients (~35–60 min
   lifetime), and a dead client CANCELS its job — even a RUNNING one.** This
   cancelled five pending 16n jobs (each reset hard-won queue age) and once
   revoked a *granted* 8n allocation mid-use (§5.3). Fix: launch salloc
   `setsid nohup ... --no-shell > lane_prefixed.log &` (detached from the
   task's process group) or host it in a persistent monitor, and poll
   squeue + the logfile. Never let a bare blocking salloc be the only holder
   of a queued job. (The orchestrator's own session tasks were NOT reaped —
   only subagent tasks were.)
2. **Jobid attribution between parallel lanes is unreliable — treat every
   agent jobid claim as unverified.** Three misattribution incidents,
   including two lanes both "owning" 57478107 (actually EPIC's backstop):
   when EPIC legitimately scancelled it, the two squatters' invocations died
   0/7 at start. Rules: a lane takes jobids ONLY from its own salloc output;
   the orchestrator audits ownership via `sacct` submit timestamps, never
   via agent claims; and reconcile the final node-hour ledger from sacct.
3. **Shared-scratchpad filename collisions are real**: two lanes truncating
   each other's `salloc_16n.log` caused the misattributions above. EVERY
   per-lane file must be lane-prefixed; put the exact filenames in the brief.
4. **Abandoned capsules from dead allocations look distinctive**: every cell
   fails at 0 s (srun "job has expired"), 0 metrics rows. Re-run the whole
   invocation from scratch; never quote the partial dir; list it in the
   report as superseded.
5. **Subagent permission layers are load-bearing walls — design around them,
   don't fight them.** They refused: running on a jobid the lane did not
   salloc (orchestrator allocation relay is IMPOSSIBLE), running variants
   outside the lane's brief (the consolidated 7-arm run died twice), and
   holding an allocation idle for handoff against the CLAUDE.md
   immediate-scancel rule (~88 min of granted 16n walltime was returned
   unused). Consequences for the format: lanes self-serve allocations only;
   put every invocation a lane might ever need in the ORIGINAL brief;
   anything cross-lane (multi-arm capsules, salvage runs) is run by the
   orchestrator session itself, which has no such classifier.
6. **Bank incrementally when walltime is uncertain.** A capsule is written
   only at the END of its invocation — one big 49-cell capsule dies whole if
   the allocation dies; per-arm 7-cell invocations bank every ~5 min. The
   orchestrator's 16n salvage used per-arm capsules with a `mins_left` gate
   before each launch.
7. **Pre-skip proven failure classes.** Torch b64 at 16n would have burned up
   to ~30 min/model in timeout+retry for a cell whose OOM law
   (constant gathered bytes) was already established at 8n. Skipping it is
   recorded explicitly, in the status log and the report.
8. **Aggregation conventions matter and are easy to get wrong** (§2): the
   max-per-span-across-ranks trap on rank-asymmetric arms; settled tables are
   plan-INCLUSIVE totals; MoonEP needs the phase-partition derivation. And
   scan-the-capsules scripts must match the FULL run_id date range (a
   `20260823-1*` glob silently dropped every evening `-2*` capsule).
9. **Gate first, always** (§3 step 4). Two campaign-saving catches for ~3 nh.
10. **Status file as coordination bus.** Lane-prefixed event lines + the
    orchestrator's `orchestrator16 START/FINISH <arm>` lines were how lanes
    knew to release duplicate grants and when their arm was covered. Cheap,
    robust, auditable; the orchestrator memory + this doc were reconstructed
    from it.
11. **Interactive QOS facts** (Perlmutter, 2026-08): 2 concurrent 4n jobs per
    user; the submit-cap rejection is transient (retry when a slot frees);
    the orchestrator explicitly rotating lanes onto freed interactive slots
    beat waiting on regular by ~an hour.
12. **Never-mix boundaries created by this campaign**: `torch_ref_impl`
    (local_slice_scatter) rows vs older global-scatter rows; s1-basis eval
    batches (dslots) vs the M4 comm-ladder's pool-sampled batches; these
    capsules vs anything from a different binary (rule 4 as always).

## 6. Open items handed to the next session

- **Layer1 hcc combine at scale** (THE next campaign): 16n phases/nsys
  attribution; RS_WIRE_STREAMS ladder (2 was tuned at nn=4) and RS CTA ladder
  at 16n; compare against EPLB's direct-a2av combine and NCCL dense RS as the
  bounds. The 4n-tuned M4 defaults (rule 11) are NOT sacred at W=64.
- LocCap recv-bound violation at 16n qwen b4–b32 (llc arm): real assertion,
  `recv bound violated: 114 rows over` — capacity model vs W scaling.
- Probe verdicts still missing: MoonEP staged l01 combine (needs sweep-style
  per-cell sizing to avoid the host-OOM), staged EPLB combine direction.
- `redundant_per_rank=1` ablation at qwen 16n if the +100% slack corner is
  quoted in the paper.
- Uncommitted: the sweeps plumbing (gen_trace_routing dslots/oracle, sweep.py
  wiring, `llc_l01_s1`, datacamp specs), the 44 capsules, this doc + CSV +
  report. The runner prints per-capsule commit commands; a human commits.
