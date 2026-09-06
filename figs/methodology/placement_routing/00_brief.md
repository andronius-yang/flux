# Expert placement & routing — methodology figure brief (2026-09-05, REV 0.0)

Discussion record for the first diagram of the methodology subsection.
Spec (`SPEC.md`), options page and generator follow once the rulings in
§7 land. Ground truth here is traced from code and capsules; every claim
names its source so a later session can re-verify.

## 1. Directive 1 applied: WHICH arm this figure explains

The main performance figure plots, per (topology, model, budget) group,
the fastest of five OURS candidates (SPEC §2.2). Resolving that rule
against `figs/main_perf/figure_src.csv` (script `main_perf_winners.py`,
ledger `main_perf_winners.csv`) over the 18 plotted groups (3 topologies
x 2 models x {1, 4, 16} MiB):

| winner (figure row -> arm) | plotted groups won | where |
|---|---|---|
| `ours12` -> `ours_l01_s1_pv2_r2` | **15 / 18** | everything except the three below |
| `ours2_direct` -> `ours_l01_s1_pv2_r2_dwire` | 2 | 16n 1 MiB, both models (same plan lane, direct wire) |
| `ours1_tokencomm` -> `l01_slipstream` | 1 | 8n Qwen 1 MiB (5.16 vs 5.41 ms, 4.5% margin; no placement/routing at all) |
| `ours12_dispatch` -> `ours_l01_s2_swap_force_p2p_r2` | **0** | wins only unplotted 64 MiB groups (4n K2, 16n both) by 0.03-0.5% |

So the placement-and-routing story the figure must tell is the **s1
pv2 r2 plan lane**, which all 17 placement-carrying winners share
(`sweeps/variants.py` pv2 arm loop; `ours_l01_s1_pv2_r2_dwire` is
defined as the same `test_args` with only the transport swapped). What
that lane is, per `docs/handoff/27_arms_canon_and_trunk.md` §2 and the
driver `test/python/moe_ag_scatter/test_moe_ours_traffic.py`:

- **placement: pv2, solved ONCE at setup, untimed**, from the scenario-1
  oracle window (`--oracle_routing_file`, basis `prev_batch` = the
  previous decode window, gap 0 — SCHEMA rule 10), with
  `--redundant_per_rank 2` (r2: two replica slots per rank beyond the
  G/W home slots — the "slack parity" ruling, handoff 21);
- **routing: LocCap sender-local, planned and TIMED every iteration**
  (`plan_comm` = top-k all-gather, `plan` = on-GPU derive under CUDA
  graphs, `--plan_overlap 2` late combine-meta issue) with slack
  `--eps 0.0625`;
- **no per-iteration placement work**: no re-solve, no trigger, no
  weight movement (`--scenario s1`; handoff 27 table: "per-iter
  placement work: none").

**Implication (must be stated to the user):** the diagram may NOT show
a per-iteration placement re-solve, a trigger decision, or expert
weights moving — those are s2 machinery, which wins no plotted group.
The system-overview brief (`figs/system_overview/00_brief.md` §2) drew
the s2 chain ("placement re-solve + swap decision" in a `place` bracket);
under directive 1 that table needs the same correction (§6).

## 2. Placement mechanism (what pv2 does) — source of truth

`python/flux/testing/placement_v2.py` (handoff 23 §1; SCHEMA rule 15
made it the canonical solver). Input: the per-node demand histogram
`hist[NN, G]` of the oracle window (NN nodes, G experts) — nothing
else. Output: replication counts, node assignment, rank assignment,
plan tensors. Four stages, all integer, all deterministic (every rank
computes the identical placement, no cross-rank exchange):

1. **Counts** — EPLB-style global greedy: start with one instance per
   expert, then repeatedly give an extra instance to the expert with the
   largest per-instance share `load/c`, integer priority
   `(load << 20) // c`, ties to the lower expert id, **hard cap
   `c[g] <= NN`** (at most one instance of an expert per node — the
   node-dedup transport makes a second in-node copy worthless for the
   wire). Total slots = `W * (G/W + 2)` under r2.
2. **Node assignment — affinity spread**: experts in (share desc, id
   asc) order seat their instances one at a time; each instance goes to
   the free-slot node not yet hosting that expert with the largest
   residual demand `hist[u, g]` (ties: lower accumulated node share,
   then lower node id). Leftover slots are backfilled to the
   highest-share non-hosting experts (slots are paid for, never
   wasted). The first instance placed is the "primary".
3. **Rank assignment** — within each node, a boustrophedon (snake) over
   its instances sorted by (share desc, id asc): balances per-rank
   compute load without any batch-size term.
4. **Plan tensors** — the canonical slot recipe (each rank's hosted
   experts in ascending id occupy its slots 0..n-1), produced directly
   from the (expert, rank) tensors.

Cost: 0.9-1.9 ms host, flat in batch (handoff 23 §4) — irrelevant to
the s1 arm's timing (setup, untimed) but it is why the same solver can
be re-run per iteration in the s2 ablations.

What the OFFLINE assessment established (memory
`greedy-place-heuristic-assessment`, handoff 23): the affinity variant
is mandatory — a blind least-loaded spread loses 99% at Qwen 4n; with
affinity, pv2 ties or beats PLACE-lambda at 16n and is at quality
parity +/-10% on hardware (handoff 23 §4 static s1 rows).

## 3. Routing mechanism (what LocCap sender-local does) — source of truth

Reference: `python/flux/testing/loccap_semantics.py` (docstring) and
the sender-local relaxation `loccap_route_sl` in
`python/flux/testing/placelambda_gpu.py` (docstring; the fused CUDA
kernel `pll_route3_kernel_t` in `src/cuda/moe_utils.cu` implements it,
handoff 13). Per iteration, every rank decides for each of its own
(token, top-k entry) pairs **which physical replica** of the chosen
expert serves it:

- **Objective = token-node incidence**, not remote-expert count: under
  the node-dedup transport a token's inter-node bytes are proportional
  to the number of distinct non-home nodes that serve it (its latent
  row travels once per node, handoff 14 / system-overview "travel
  once"). Hence the remainder is solved as a per-token **minimum node
  cover**, never greedily per entry.
- **Three tiers, in order**: (1) an instance on the token's **home
  rank** (zero movement); (2) an instance on its **home node** (NVLink
  only, zero wire bytes); (3) a minimum set of **remote nodes** covering
  the token's remaining experts.
- **Capacity cap** per rank `kappa = ceil((1 + eps) * S * K)` rows with
  eps = 0.0625: tiers 1-2 consume a rank's fair share via closed-form
  quotas and a home-node water-fill (largest-remainder integer
  rounding); tier 3 consumes per-(source, destination) **shares** of the
  post-tier-2 residual capacity, pre-partitioned proportionally, so the
  decision is **sender-local**: rank r assigns only its own S*K entries
  from tables that are pure integer functions of the all-gathered demand
  histogram `d[R, G]`. No live global capacity feedback, no repair pass,
  no donation of unused shares (the accepted cost vs the global
  algorithm; offline gate measures the gap). Demand no share can place
  is "forced" to the least-loaded hosting rank and counted.
- **Timing**: the demand histogram comes from the top-k all-gather that
  is `plan_comm`; routing + plan-tensor derive is `plan` (handoff 13:
  1.2-1.4 ms at 4n after the sort-free tail + CUDA graph).

Everything shared across ranks is a deterministic function of `d`, so
sender and receiver agree on splits with no extra exchange — this is
the design point that lets routing be re-planned every iteration inside
the timed window at ~1 ms.

## 4. Key insights the diagram must carry

1. **Two cadences, one plan lane.** Placement is decided once from the
   previous window's demand (what is hot, how many copies, which nodes);
   routing is decided every iteration from the current batch's demand.
   Drift between the two is absorbed by routing choosing among replicas,
   not by moving weights.
2. **Placement is node-first.** The `c <= NN` cap and the affinity
   spread put each replica on the node that demanded the expert most,
   because with node-dedup only *which nodes* host an expert changes the
   wire; the intra-node rank choice is a compute-balance afterthought
   (snake).
3. **Routing minimizes distinct remote nodes per token, under a compute
   cap.** The tier order is exactly locality order (rank, node, remote
   cover), and eps is the single knob trading GEMM-side balance for
   wire locality.
4. **Sender-local = no coordination round.** Every rank derives the
   same quota/share tables from the one all-gather it already pays;
   nothing is negotiated, so the per-iteration cost is a small GPU
   derive rather than a collective.
5. **Replica headroom (r2) is what makes 1-3 pay**: two extra slots per
   rank are the only room the counts stage has to split hot experts
   (handoff 21 parity finding: OURS with R_red=0 had nowhere to split
   the ~1 GB hot-expert load at Qwen 16n).

## 5. Candidate compositions (for the options page)

**A. Two-panel "decide once / route every step".** Left: a small cluster
(2 nodes x 2 GPUs, 6-8 experts) with the oracle demand histogram as
tiny bars per node and the resulting placement (replicas of the hot
expert on the two nodes that wanted it, cap 1/node). Right: one batch's
tokens on one source rank, each token's top-k fanning out through the
three tiers to replicas, with the cover for a two-expert token landing
on ONE remote node instead of two. A thin strip beneath marks the
cadence (setup vs every iteration).

**B. Single "worked example" panel.** One 2x2-GPU cluster; the
histogram, placement and one iteration's routing drawn on the same
picture with numbered badges 1-3 (counts, spread, cover). Densest but
one register, and it reuses the system-overview drawing vocabulary
(`figs/system_overview` REV 0.3 glyphs: travel-once, ghost reroute).

**C. Algorithm-box pair.** Two compact pseudo-code / flow boxes
(placement stages 1-4; routing tiers 1-3 with the cap) plus a tiny
example. Cleanest for a methodology reader, least visual.

Working recommendation: **A**, sharing glyphs with the system overview
so the reader carries the picture across the paper.

## 6. Reconciliation items raised by directive 1

- `figs/system_overview/00_brief.md` §2 describes the per-iteration
  chain of "the methodology arm (OURS s2 = always-solve + overlapped
  swaps)", including a timed `place` bracket. Under directive 1 the
  overview should draw the s1 chain (no `place` bracket, no weight
  movement) — or the directive needs a carve-out for the overview.
  Not edited; user ruling needed.
- The third methodology mechanism, "expert-dispatch overlap", is by
  construction an s2-only mechanism, and no s2 arm wins a plotted
  main-perf group (they win three unplotted 64 MiB groups by <0.5%).
  The cycling-ablation figure (`figs/ablation_cycling`) is where swaps
  pay. Directive 1 as stated would leave that figure without a
  main-perf arm to describe; user ruling needed on whether the
  directive scopes the ablation-backed mechanism differently.
- `docs/handoff/27_arms_canon_and_trunk.md` §3 still calls a planned
  s2-combined arm "THE s2 of the methodology section". That plan
  predates today's directive and is superseded by it for the figures.

## 7. Open rulings for the user

1. Composition A / B / C (§5), or another.
2. Example size: 2 nodes x 2 GPUs (system-overview scale) vs 2 x 4
   (Perlmutter-authentic).
3. Whether to show eps/capacity at all, or only the three tiers.
4. Height budget and tooling (draw.io like the system overview, or a
   generator like main_perf).
5. The two reconciliation items in §6.
