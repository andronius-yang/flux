# START HERE — layer0 a2av, AWS → Perlmutter handoff (2026-08-04)

You are picking up a **measurement project with a working implementation**. Your output is
capsules, not code. The implementation landed; what is unfinished is knowing how it behaves
on a shape we never ran.

**Where work stopped.** Tier B window gating for `hier_compress_lb_union` landed 2026-08-04
(commits `7d4b3b9`, `232f371`, `8549311`) on AWS, 2 nodes × 8× A100, EFA. That cluster is
**terminated** — nothing can be re-measured there. 124 capsules survive under
`sweeps/results/runs/`.

**Where you are.** NERSC Perlmutter: 4 GPUs/node (L=4, not 8), Slingshot/CXI (not EFA),
and ≥4 nodes allocatable — so **NN>2 is reachable for the first time**. NVSHMEM is 3.2.5-1
here vs 3.3.9 on AWS. The CUDA 12.4 pin holds on both.

**The question this work exists to answer:** how do the current layer0 variants perform at
**NN>2 and L=4**? Both axes are untested. Several measured results are predicted to move in
a specific direction on this shape — `02` states them as falsifiable predictions.

---

## Four known blockers, in the order you will hit them

1. `module.sh:31` hardcodes a stale `FLUX_ROOT="$HOME/workspace/changchen/andrewy/flux"`.
   It does not match this checkout and will silently misdirect `CPATH`.
2. `module.sh:27` activates `$PSCRATCH/conda_envs/andrewy-comet`. `$PSCRATCH` is purged
   periodically at NERSC; assume this env may be gone. `01` §2 has the rebuild contract.
3. `sweeps/platforms/perlmutter.yaml` is **entirely untested hypothesis** — no `nsys_bin`,
   empty `srun_extra` and `env`, `sym_size_max_g` copied from 8-rank AWS, `$PSCRATCH` roots
   that `sweep.py:230-236` hard-fails on if unresolved.
4. **Zero of the 124 capsules were produced on Perlmutter.** The sweep harness is 100%
   AWS-validated. Expect first-run breakage in the `srun` prefix and path expansion.

## The first four commands

```bash
salloc --qos interactive -C gpu --account m4243_g -N 2 --gpus-per-node=4 -t 60 --no-shell
source ./module.sh                     # fix FLUX_ROOT first (blocker 1)
srun --jobid=<id> --nodes=1 --ntasks-per-node=1 bash -lc \
  'source ./module.sh && nproc=16 ./build.sh --arch 80 --sm-cores 108 --nvshmem --no_test --jobs 16'
srun --nodes=2 --ntasks-per-node=1 ./launch.sh test/python/moe_ag_scatter/test_moe_ag.py
```

**Stop at the first failure and fix it there.** Do not proceed past a warning — `01` is a
gated ladder for exactly this reason.

## Three rules that decide whether a number is real

1. The `deterministic` column in `cells.csv` must read `0`. If it reads `1`, the number is
   garbage (deterministic `scatter_` is ~500× slower and the compress paths are
   `scatter_`-heavy). The runner enforces this; verify anyway.
2. **Never compare instrumented cells against clean ones.** `phases` (`FLUX_A2AV_TIMING`)
   and `nsys` force per-iteration syncs. They are for breakdowns, never for latency.
3. Quote **`isolated`** mode for latency (mean over iterations of the per-iteration max
   across ranks) and `e2e` for pipelined throughput. They measure different regimes.

A fourth rule, learned the hard way and now in `SCHEMA.md`: **compare arms inside one
capsule built from one binary.** Across builds the same configuration moved by 6–33%, which
is larger than every headline claim in this project. See `04`.

`sweeps/SCHEMA.md` is the authority on all of this and is current.

---

## Reading order

| Read | When |
|---|---|
| `01_perlmutter_bringup.md` | Now. Execute it; it is a gated ladder. |
| `sweeps/SCHEMA.md` + `.claude/skills/sweep/SKILL.md` | Before your first sweep, not after. |
| `02_algorithm_state_and_next_moves.md` | Once you have one green capsule. This is the daily driver. |
| `03_insight_ledger.md` | **Before you propose any optimization.** |
| `04_build_ledger.md` | When interpreting any pre-existing capsule. |
| `docs/qa_walkthroughs/layer0_a2av_walkthrough.md`, `docs/launch/comet_traffic_matrix_a2av.md` | Only when you need mechanism. Still accurate, platform-neutral. |
| `09_comm_only_tier12_restructure.md` | Before touching the comm-only drivers' timing (FAST/comet/hier arms): the queued Tier-1/Tier-2 (rule-5) restructure campaign, self-contained for a fresh session. |

`03` exists to stop you re-deriving things that cost days. Read it before you build, not
after — including the idea you are most likely to propose in your first hour, that **saving
wire bytes wins**. It doesn't, and `NR-01` explains why.

Scope note: this handoff was written **layer0-only**. The layer1 work
(split-pipelined hierarchical alltoallv combine, `a2av_hier[_compress]` on
`GemmGroupedV2GatherRSOp`, + eager arrival-order reduce) has since MERGED to
main (2026-08-11, f97c628 era) and is sweep-integrated as the `l1_*` variants
(2026-08-16, layer-axis campaign — see `sweeps/SCHEMA.md` §layer/timing_mode).
Layer1 compress is CPU-sim validated; its first GPU run is the campaign's 2n
bring-up ladder.

**Layer-axis campaign authority (2026-08-16): `07_comm_only_layer_axis_campaign.md`**
— verdicts, capsule ledger, and both hang root-causes for the comm-only
layer0+layer1 campaign live there (`07_dashboard.html` is the visual summary);
`08_followup_queue.md` is the post-campaign work queue. Canonicalization of the
campaign winners (F+E default-ON under LB_UNION; `l01_lbunion_compress` as the
reference combined config; FANOUT and RS_EAGER closed as losers) landed
2026-08-16 — `sweeps/variants.py` notes are the arm-level authority.

**Comm-only production readiness (2026-08-22): `10_comm_baselines_production_and_ladder.md`**
— the rule-5 conversion of all comm-only arms (per-iteration timed planning,
K3 enablement incl. the 512-tile dense combine + topk-16, the FAST combined
arm on real routing, nominal byte ladder, homog canon pool), the validated
b1–b64 sweep procedure with the never-mix registry, the current K3 4n
standings, and the warnings/gates for 8n/16n/32n allocations (FAST 64-rank
segfault, W64 closure, kA2AVMaxWorld/Nodes limits blocking 32n, SM-budget
re-probing per rung). Supersedes `09` for the comm-only lane.
