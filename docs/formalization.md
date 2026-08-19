# Formalization: primitives, flows, and scheduling for MoE layer0

**Status: working theory, not settled.** This is the running record of the
framework we intend to use as the paper's Section 3 ("why these optimizations
are one thing, not a pile of tricks"). It exists to be recalled in future
paper-writing / brainstorming sessions. It records the framing, the empirical
evidence extracted from existing capsules, **and the objections and open
questions raised against it** — those are as important as the claims.

Started 2026-08-15. Companion authorities: `sweeps/SCHEMA.md` (what results
mean), `docs/handoff/00_START_HERE.md` (campaign state), `docs/design.md`
(kernel design).

---

## 0. Motivation

We had been describing the work as "we categorized three consumers of resources
in MoE — expert GEMM, expert dispatch, token dispatch — and existing systems
each address some subset":

| system | addresses | leaves alone |
|---|---|---|
| FAST | token dispatch | expert GEMM overlap, balance |
| Comet | overlaps expert GEMM with token dispatch | balances neither token nor expert dispatch |
| MoonEP | shifts imbalance from token dispatch to expert dispatch | overlaps nothing |
| EPLB | rebalances expert placement | no overlap, no wire reshaping |

A systems paper cannot present this as "we did a bunch of overlapping and
reduction." We need primitives that let us reason about flow of data and usage
of compute. The starting pair was **merge** (aggregate so data crosses a domain
boundary once, possibly via a gateway) and **split** (subdivide a large object
so multiple NICs / paths carry it).

This doc argues those two are real but incomplete, and proposes what completes
them.

---

## 1. The core formulation: one assignment variable generates all three costs

The three "resource consumers" are not independent — they are three shadows of
a single choice:

> For every (token *t*, expert *e*) pair the router demands, choose the rank
> **r(t,e)** where that product is computed.

That choice induces all three costs at once:

- *t* must be present at `r(t,e)` → **token dispatch**
- `W_e` must be present at `r(t,e)` → **expert dispatch**
- `r(t,e)` pays the FLOPs → **compute, and its imbalance**

Every system is a corner of this one problem:

| system | r(t,e) | consequence |
|---|---|---|
| Comet / classic EP | `home(e)` | tokens move; imbalance = routing skew |
| MoonEP | `home(t)` | weights move; imbalance moves to which experts get pulled |
| EPLB | `home(e)`, but *changes* `home` | same flow shape, flatter load, memory cost |
| ours (virtual-expert lb_union) | mixed, per-group | the interpolation nobody else does |

### 1.1 The crossover is a bytes ratio

Per expert *e*, with *h* = bytes per token activation and `W_e` = bytes of e's
layer0 weights:

- **pull tokens to e:** `h · (#remote tokens routed to e)`
- **push e's weights to the tokens:** `W_e` (once, under multicast)

→ **push iff remote-token-count for *e* exceeds `W_e / h`.**

Illustrative (H=7168, I=2048, gate+up, bf16): `W_e ≈ 59 MB`, `h ≈ 14 KB`,
threshold ≈ **4k tokens** (≈2k with fp8 weights).

Consequences worth stating as a result rather than a heuristic:

- Below threshold tokens win — which is *why* classic EP is the sensible default.
- Above it (long prefill; high skew concentrating tokens on one expert) weights win.
- **The direction of optimal flow flips with batch × skew, and it flips per
  expert, not per layer.** This is the justification for a framework instead of
  a point design, and it explains why MoonEP looks good only on some shapes.

---

## 2. The cost model and the lower bound

Resources are richer than (network, compute). The real vector is:

**R = { NIC (inter-node), NVLink (intra-node), CE (copy engine), SM, HBM }**

For a placement π, with `L_r(π)` the load induced on resource *r*:

```
T  ≥  max_r  L_r(π) / rate_r
```

plus a critical-path floor (the ingredient bytes + FLOPs of any single tile).

This gives the paper's spine:

> **Placement moves the lower bound. The other primitives close the gap to it.**

- MoonEP lowers `L_SM`(imbalance), raises `L_NIC`.
- EPLB lowers `L_SM` at fixed `L_NIC`, paying memory.
- **Comet does not move the bound at all** — it only closes the gap. Which is
  exactly why overlap alone collapses under routing skew.

Prior work does one or the other. The reason both are needed jointly:
**the placement that minimizes the bound is not the one that is easiest to
overlap.**

---

## 3. The primitive algebra

### 3.1 Merge and split are the *spatial* half of a 2×2

Both share a signature — *amortize* (fewer, bigger) vs *subdivide* (more,
smaller) — and that signature also exists on the time axis:

|  | **Amortize** | **Subdivide** |
|---|---|---|
| **Space** | **Merge** — one crossing serves many destinations | **Split** — many paths serve one object |
| **Time** | **Cache / Replicate** — one transfer serves many uses | **Pipeline** — many stages serve one object |

Every cell pays the same currency:

- **amortization** buys byte/fixed-cost savings with **latency and coupling**
  (you wait for the slowest contributor);
- **subdivision** buys parallelism and earliness with **fixed cost and
  coordination**.

Occupancy of the cells: Comet is bottom-right only. EPLB is bottom-left (+
Place). DeepEP is the top row. Nobody occupies all four — that is the shape of
our contribution.

### 3.2 Merge is TWO primitives, not one

This is the sharpening the data forced (§5), and it is probably the single most
defensible novel claim in the framework:

- **Dedup-merge** — *same bytes* to many destinations cross once (node-level
  token compression; weight multicast). Saves **bytes**. Opportunity depends on
  *collisions* (a token's topk landing on one node).
- **Coalesce-merge** — *different bytes* to the same destination are bundled.
  Saves **messages** (per-put fixed cost). Opportunity depends on
  *fragmentation* (peer count × small transfers).

They have different cost models and, empirically, **opposite scaling on both
node count and budget.**

Symmetrically, split is two things: *bandwidth split* (parallel paths, same
time) and *pipeline split* (sequential chunks, earliness — the temporal cell).

### 3.3 Merge relocates work; it never removes it

Merge introduces a **gateway**, which is a new resource that can bind. It is
profitable on bytes essentially always (NVLink ~600 GB/s vs NIC ~25 GB/s), so
**the real cost of merge is latency coupling, not bytes.**

This reframes our window gating: **`lb` vs `union` is not a hack, it is the
continuous knob on merge** — how much to coalesce before firing, trading
coupling against amortization. The starvation campaign's numbers (lb deficit
0.29–0.455 vs union 0.07–0.11) are a direct measurement of over-merging's
coupling cost.

### 3.4 Overlap is not a fourth cell — it converts Σ into max

Overlap does not reshape a flow. It changes how resource loads *combine*:

```
without overlap:   T = Σ_r  L_r / rate_r
with perfect overlap: T = max_r  L_r / rate_r
```

> **Overlap's entire contribution is converting a sum over resources into a max
> over resources.**

This makes it first-class instead of an afterthought, and it says exactly when
it is worthless: when one resource dominates, `Σ ≈ max` already and there is
nothing to hide. Hence Comet-style overlap collapsing under skew.

Define **overlap efficiency**:

```
η = (Σ − T) / (Σ − max)   ∈ [0,1]
```

**η is measurable from capsules we already have**: `phases` mode force-syncs per
iteration, so it *is* a measurement of Σ. `η = (Σ_phases − T_isolated) /
(Σ_phases − max_phase)`. Caveat: the syncs inflate Σ, so η computed this way is
a **conservative lower bound**. Nobody in this literature reports an overlap
efficiency number; we could.

### 3.5 The engine axis: primitives are not implementation-neutral

Because CE and SM are distinct resources, **which engine implements a primitive
decides whether it competes with the compute it is supposed to hide behind.**

> A spatial primitive's value depends on its implementing engine.

A merge staged through SM-driven packing/index-building destroys overlap with
the GEMM even while saving the same wire bytes; a merge staged through CE-driven
bulk copies preserves it. This is why `sm_margin` and
`CUDA_DEVICE_MAX_CONNECTIONS` are first-order in our results and not tuning
noise. §5.3 has the measurement.

### 3.6 Merge *creates* the overlap opportunity

Not orthogonal — they compose. Flat a2av uses one engine for cross-node
traffic. Hierarchy decomposes each transfer into
**NVLink-gather → NIC-cross → NVLink-scatter**: three stages over two disjoint
engines, i.e. a pipeline that did not previously exist.

> **Merge manufactures the stages; overlap fills them.**

So "overlapping inter-node RDMA with intra-node NVLink" is not a separate
optimization from hierarchy — it is the payoff of hierarchy, and it is why
hierarchy's win grows with node count (§5.1).

### 3.7 Summary: the three-part structure

1. **Place** sets the loads `L_r` → *sets the bound*.
2. **Merge / Split** move load between resources → *reshape the bound*.
3. **Overlap / Pipeline** make `max` achievable instead of `Σ`, conditional on
   engine disjointness → *close the gap*.

---

## 4. The scheduling problem

Layer0 is a **two-ingredient assembly problem**: tile (t,e) becomes eligible at
`max(arrival(t), arrival(W_e))`, and ingredients are *shared* — one weight
arrival unlocks many tiles, one token arrival unlocks topk.

The natural greedy is **density**: send next whatever maximizes *unlocked FLOPs
per byte on the binding resource*.

### 4.1 Why naive greedy stalls, and the fix

The dependency is an **AND**, so the unlock function is **not submodular** — a
weight arriving alone unlocks nothing. Greedy stalls. This is precisely why the
current staged scheme exists (prefetch expert weights scheduled last, dispatch
tokens sent early): it hand-resolves the conjunction.

> **The current 2-round scheme is the two-point discretization of the density
> rule.** Tokens are small and unlock topk tiles each (high density); weight
> blocks are huge with zero density until tokens land.

**The fix — split both flows fine enough that short prefixes already unlock a
real tile.** For weights that means splitting along the **N (intermediate)
dimension**: `W[:, 0:I/c]` plus any tokens is a valid GEMM tile.

> Comet reschedules layer0 along **Dim-M** to overlap the *token* flow. The dual
> is **Dim-N** to overlap the *weight* flow.

With both, the unlock function becomes near-linear in prefix, density is
well-defined incrementally and **dynamic** (weight-chunk density jumps the
instant the first token chunk lands), and the two rounds dissolve into one
density-ordered stream. A static 2-round schedule cannot exploit that jump.
Bonus: column blocks are discardable after use → memory-bounded weight
streaming → more resident experts.

### 4.2 Hardness

Exact makespan is NP-hard: the placement half is a partition problem; the
ordering half is min-sum-set-cover-like (greedy is 4-approx in the OR case).
The split-to-linearize trick is what buys back the regime where greedy has
teeth.

The placement stage itself is cheap enough for one-shot per-layer use: binary
search on T with a greedy feasibility check over E×R counts (256×64 = 16k) is
microseconds on host — compatible with the no-schedule-caching constraint.

### 4.3 Balance vs locality: the central tension

Dedup-merge pays off when a token's topk experts **collide on one node**.
Load balancing spreads a hot expert's replicas across nodes, which spreads each
token's destination set and **destroys collisions**.

> **Load balance and co-activation locality are opposed placement objectives.**

(Recorded but *deprioritized* — see §6, the user prefers a MoonEP-style
per-batch direction over an EPLB successor. The tension itself remains true and
should still be stated in the paper; it is what makes joint scheduling
necessary. Supporting evidence already in hand: mismatched-topic load made
placement *worse* than fixed, +31–43% e2e, because a load-only objective is
fragile to distribution shift.)

### 4.4 Re-push vs keep-stale is ski rental

MoonEP re-prefetches every round — no persistent replication. The question
"replicate only when enough has changed" is **not orthogonal to e2e latency**;
it is the temporal-amortization cell of the 2×2, trading expert-dispatch bytes
(`L_NIC`) against compute imbalance (`L_SM`) — the same objective, evaluated on
the time axis.

> Keeping a stale placement costs an imbalance penalty **every batch**;
> re-pushing costs a large **one-time** transfer. Rent repeatedly, or buy once.

This is **ski rental**: deterministic 2-competitive, `e/(e−1)` randomized, with
**no prediction of future batches required** — which matters given
no-schedule-caching. Concrete rule: accumulate the imbalance penalty since the
last re-push; re-push when it reaches the weight-transfer cost.

MoonEP is the degenerate *always-rent* strategy. The claim this buys us against
it is not "we overlapped more" but: **same placement quality, asymptotically
bounded fraction of the weight traffic.**

---

## 5. Empirical evidence (from existing capsules, 2026-08-15)

Offline re-analysis of all 175 capsules; **no new GPU time**. Every delta
computed **within a capsule** (same binary — SCHEMA invariant 4), same `mode`,
same `matrix_id`, topk=8, median `e2e_ms`. Cell counts are small (2–7); the
trends are consistent across 4 node counts × 3 budgets, but individual cells are
suggestive, not definitive.

### 5.1 Coalesce-merge: `hier` vs flat `a2av`

Delta of a2av vs hier (positive = a2av slower, i.e. coalescing wins):

| nodes | b=2MiB | b=8MiB | b=32MiB |
|---|---|---|---|
| 2 | −5.8 | +69.0 | +0.4 |
| 4 | +67.2 | +68.9 | +64.4 |
| 8 | +147.0 | +74.6 | +47.7 |
| 16 | **+238.0** | +148.4 | · |

**Grows with N; grows as budget shrinks.** At b=2MiB/16 nodes flat a2av is
**3.4× slower**. Low budget × many nodes is exactly where fragmented small puts
dominate — the per-put fixed cost term.

### 5.2 Dedup-merge: `hier_compress_union` vs `hier`

| nodes | b=2MiB | b=8MiB | b=32MiB |
|---|---|---|---|
| 2 | −36.4 | −53.9 | **−61.2** |
| 4 | −18.5 | −36.9 | −41.9 |
| 8 | +2.6 | −5.4 | −0.5 |
| 16 | +1.2 | −1.1 | · |

**Decays with N — dead by n=8, crosses zero — and grows with budget.**

### 5.3 The two anti-correlate on both axes

Dedup-merge saves *bytes* → matters when bytes bind → large budget; opportunity
shrinks with N (fewer collisions). Coalesce-merge saves *messages* → matters
when fixed cost binds → small budget, many peers; opportunity grows with N.

This is the cleanest available validation of §3.2. **Paper figure.**

### 5.4 RETRACTED — there was no coupling term; the opportunity was never there

**This section previously claimed:** the dedup byte saving at n=8/k=8 is still
~34% analytically (from the iid reference `(N−1)(1 − (1−1/N)^k)` vs `k(N−1)/N`
crossings) yet measured latency benefit is zero, therefore the cost model needs
a *gateway coupling term growing with N* that cancels it.

**That was wrong, and the error was using an iid reference for a non-iid
generator.** Measured directly (2026-08-15, `sweeps/dedup_factor.py`, offline,
no GPU) the true cross-node dedup ceiling of the synthetic families is:

| nodes | uniform | fanoutskew | iid reference (what §5.4 assumed) |
|---|---|---|---|
| 4 | 53.1% | 52.5% | 55.0% |
| 8 | **3.1%** | **2.5%** | 34.4% |
| 16 | **0.0%** | **0.0%** | 19.3% |

There is essentially **no dedup opportunity at n≥8 and literally none at n=16**.
Dedup-merge shows zero latency benefit there because **there is nothing to
dedup** — no coupling term is needed to explain §5.2 at all.

**The mechanism is exact and closed-form.** The sorted column-major dealer gives
`U[s][n] = min(copies_to_node, T)` (`gen_matrix.dedup_round_stats`), and
copies-per-node ≈ `topk·T/nn`, so

> **synthetic dedup ceiling = max(0, 1 − nn/topk)** — nonzero only while
> **nodes < topk**.

Check: n=2 → 75%, n=4 → 50% (measured 53.1%), n=8 → 0% (measured 3.1%, the
residual is skew/rounding), n=16 → 0%. And it upper-bounds the measured
latencies of §5.2 (−54.8%, −26.5%, ~0, ~0) exactly as a ceiling should.

### 5.4b The retraction's real consequence: §5.2 is a generator artifact

Real routing does **not** collapse. Re-mapping the stored Qwen3 routing files
across node counts (same real co-activation structure, only the
expert→rank→node map changes as `epr = G//W`, tokens-per-rank held constant):

| nodes | real trace | synthetic dealer | iid reference |
|---|---|---|---|
| 2 | 75.1% | 75.0% | 75.0% |
| 4 | 53.3% | 50.0% | 55.0% |
| 8 | **33.6%** | **0.0%** | 34.4% |
| 16 | **17.5%** | **0.0%** | 19.3% |

Confirmed exactly, not just by re-mapping, on the *actually generated* 8-node
trace matrix `w32x4_trace-2d1cf0_b8_k8_id001`: **35.7%** ceiling
(copies=229047, unique=147201).

**Real Qwen3 routing behaves like iid for dedup purposes; the synthetic dealer
destroys all dedup opportunity once `nn ≥ topk`.** Since every n=8 and n=16
measurement in §5.1–§5.3 used synthetic families, and every trace capsule in the
repo is 4-node:

> **§5.2's "dedup-merge dies at n≥8" is an artifact of the matrix generator, not
> a property of MoE dispatch. We have no real-routing measurement of dedup-merge
> above 4 nodes.** The headline claim of §3.2/§5.3 — that dedup and coalesce
> anti-correlate in N — survives only on the budget axis until a trace run at
> n≥8 exists.

This is why the 8n/16n eager specs carry a `trace` arm: the synthetic families
there can test eager's round de-serialization, but they cannot test dedup at
all, because their ceiling is exactly zero.

*Methodological lesson worth keeping:* an analytic reference is only a reference
for the generator that produced the data. The iid formula was right for real
routing and wrong for the dealer, and using it as a stand-in manufactured a
"missing coupling term" that did not exist.

### 5.5 The engine axis, measured

At n=4 / topk=8, vs `hier`:

| variant | delta | engine |
|---|---|---|
| `hier_compress` | −0.1% | SM index build |
| `hier_compress_union` | **−26.5%** | pure-CE puts |
| `hier_compress_lb_union` | −24.7% | pure-CE puts |

**All three dedup the same bytes.** The ~26-point spread is purely the SM cost
of the index build versus CE puts. Same primitive, same byte saving, different
engine, 26 points. Direct support for §3.5.

### 5.6 Terminology caution — `wire_ratio` is NOT the dedup factor

`wire_ratio` in `cells.csv` is `headroom` from
`sweeps/gen_matrix.py:dedup_stats_from_U` = `relay_balanced_bytes /
relay_ident_bytes`, a **relay load-balance** metric. The true dedup factor is
`pair[(sn,dn)] = (copies, unique, ratio)` from the same function, and it is
**computable offline** over the 200 stored matrices at
`$PSCRATCH/.../a2av_test_matrices/generated/` (23 with real-trace routing, where
the closed form does not hold and U comes from measured routing).

**DONE 2026-08-15** — `sweeps/dedup_factor.py`, zero GPU cost. It did not
quantify a coupling term; it showed there was none to quantify, and retracted
§5.4. See §5.4/§5.4b. The script computes both the synthetic ceiling (regenerate
chunks deterministically → `dedup_round_stats`) and the real-routing ceiling
from a `<mid>.routing.txt`.

### 5.7 SCOPE CAVEAT — every result above is from a near-compute-balanced regime

**Expert placement in the a2av harness is static, uniform, and contiguous**:
`owner_rank = expert_id // experts_per_rank`, `experts_per_rank = G // W`
(`python/flux/testing/traffic_matrix.py:62`). No replication, no balancing. On
top of that the harness levels load twice more:

1. **Row sums are asserted equal** (`traffic_matrix.py:91-93`) — invariant 1.
   Per-rank *send* volume is structurally identical. (This is physically correct
   for a uniform batch split: every rank holds `tokens_per_rank` tokens, each
   picking exactly topk experts. It is not an artifact — but it does mean ragged
   / variable-tokens-per-rank batches cannot be expressed at all.)
2. **Within a destination rank, copies are dealt round-robin over that rank's
   experts** (`base = chunks[s] // experts_per_rank` + remainder spread), so
   per-expert load inside a rank is as even as arithmetic allows.

The matrix families then skew the *wire pattern* (fan-out, remote fraction, node
skew) — not compute. Measured over the 200 stored matrices:

| family | W | row imb (send) | col imb (**compute**) med | max |
|---|---|---|---|---|
| uniform | 16/32/64 | 1.0000 | 1.00 | 1.05 |
| remotefrac | 16/32/64 | 1.0000 | 1.22 | 1.25 |
| fanoutskew | 16/32/64 | 1.0000 | 1.43–1.53 | 1.54 |
| **trace (real Qwen3)** | **16** | 1.0000 | **2.03** | **3.10** |
| hotcol | 16 | 1.0000 | 7.50 | 7.50 |

→ **`L_SM` was pinned near-constant while `L_NIC` was varied.** The synthetic
sweeps are, by design, a pure comm benchmark — which is exactly the regime where
placement has nothing to fix and flow primitives are the only lever. That is why
placement did not appear to matter before it was introduced as a variable.

Placement became relevant precisely when the campaign moved to real traces: the
trace family's 2.03 median imbalance is the same ballpark as the EPLB arm's
reported 1.852 → 1.014.

**The gap that matters for the framework:** there are **no trace matrices above
W=16** (22 at W=16, 1 at W=4; none at W=32/64). So every node-count scaling
result in §5.1–§5.4 — including the dedup/coalesce crossover and dedup's death
at n=8 — comes from configurations with essentially no compute imbalance
(1.00–1.53). **Prediction:** at 8n/16n *with real traces*, `L_SM` starts to bind,
and per §3.4 (Σ→max) shaving comm further stops paying — the crossover node
count should move, and placement should overtake flow primitives as the lever.
Until real-trace matrices exist at W=32/64, the scaling story is only established
for the comm-bound regime.

---

### 5.8 CONFOUND — the "lb wins at 16 nodes" result is a BLOCKING_WIRE artifact

Recalled from an earlier campaign: at 16 nodes `hier_compress_lb_union` beat
`hier_compress_union`, reversing the penalty seen at lower N. **The result is
real but confounded**, and the confound inverts the conclusion.

The only 16-node lb-vs-union data in the whole capsule set is
`20260805-135758_perlmutter_25308597` (topk=4, G=128) and
`20260805-142025_perlmutter_44fa16c5` (topk=8, G=192) — same day, `git_sha
a25eda6e`, `.so 23035ab8f8f3`, isolated, `sm_margin=8`, `EARLY_LAUNCH=1`,
**`FLUX_A2AV_BLOCKING_WIRE=1`**, and **no `CUDA_DEVICE_MAX_CONNECTIONS` key at
all** (i.e. conn=1 from `launch.sh`'s default).

Splitting every lb-vs-union pair in the capsule set by that knob:

| nodes | wire | conn | cells | lb vs union |
|---|---|---|---|---|
| 2 | nonblock | 8 | 47 | +2.0 |
| 4 | nonblock | 8 | 61 | +2.8 |
| 8 | nonblock | 8 | 15 | +4.2 |
| 8 | nonblock | 32 | 15 | +4.5 |
| 4 | **BLOCK** | 1 | 18 | +6.3 |
| 8 | **BLOCK** | 1 | 18 | +2.2 |
| 16 | **BLOCK** | 1 | 10 | **−1.7** |
| 16 | **BLOCK** | 8 | 5 | +0.5 |

> **The N-trend reverses sign with the knob.** Blocking: +6.3 → +2.2 → −1.7, lb
> improving with N to an outright win. Non-blocking: +2.0 → +2.8 → +4.2, lb
> getting *worse* with N. These are 47/61/15/18-cell medians, not noise.

`FLUX_A2AV_BLOCKING_WIRE` is documented measurement-only: it serializes the GEMM
behind the wire (§9 "laws", `layer0_a2av_walkthrough.md`), and the
realistic-trace campaign already recorded that the lb/union gap **vanishes**
under it. So blocking mode suppresses precisely the mechanism that makes lb
worse — the 16n win is most plausibly that suppression rather than a genuine
crossover. Note also that even at 16n, moving from conn=1 to conn=8 *within*
blocking mode takes lb from −1.7% back to +0.5%.

**There is no non-blocking 16-node lb-vs-union measurement in existence.** The
pending A1/A2 capsules supply the first one, and the two regimes make opposite
predictions there — which makes that run considerably more valuable than the
eager verdict alone (§7.5b) suggested.

A second axis is worth probing rather than assuming: lb's penalty *does* shrink
with budget (topk8 at 4n: +10.4 at b=1 → +2.3 at b=8), and the 16n wins sit at
the largest budgets tested (−1.4 at b=8 topk8; −2.2 at b=32 topk4). If lb ever
wins non-blocking, high N with a large budget is where. `b=64` has never run
above 2 nodes and `b=32`/topk8 at 16n has zero ok cells historically (all exit
143, timeout-reaped) — so
`sweeps/specs/lbunion_pm16n_iso_k8_bigbudget_probe.yaml` isolates it as a
separate 8-cell capsule where a timeout costs one cell rather than the campaign.
Matrix generation at b=64/64-rank is confirmed working (sym 10G, cap 262144), so
the only live risk there is runtime harness memory.

**RESOLVED 2026-08-15 — capsule `20260815-124911_perlmutter_71b93a5d`.** The
first non-blocking 16-node lb-vs-union measurement. 18/18 ok, isolated, conn=8,
topk=8, G=192, instance 003, build `31fc91f44af8` (median stat):

| family | b=2 | b=8 | b=16 |
|---|---|---|---|
| trace | +6.2 | +6.4 | +2.5 |
| fanoutskew | +7.8 | +3.9 | **−8.3** |

**lb loses in 5 of 6 cells.** At the two budgets that the blocking capsules
also covered, the contrast is direct:

| budget | BLOCKING conn=1 | NON-BLOCKING conn=8 (this run) |
|---|---|---|
| b=2 | **+0.3** | +6.2 / +7.8 |
| b=8 | **−1.4** | +6.4 / +3.9 |

> **The 16-node lb win does not survive the removal of `BLOCKING_WIRE`.** It was
> the knob, as §5.8 predicted. The non-blocking series therefore reads
> +2.0 (2n) → +2.8 (4n) → +4.2 (8n) → ≈+5 (16n): **lb_union gets monotonically
> worse than union as N grows, with no crossover.**

Two honest qualifications:

1. **Build-confounded across N.** This capsule ran on `31fc91f44af8`
   (phase-ordered-wire), not the `ba9e91096019` used for 4n/8n, because the
   conda editable install was repointed mid-campaign. The *within-capsule*
   lb-vs-union comparison is valid (SCHEMA rule 4); splicing this point onto the
   2n/4n/8n series is not, and the build ledger records configs moving 6–33%
   across builds. The qualitative conclusion (lb loses, no crossover) is robust
   because it is a within-capsule sign, not a magnitude.
2. **Noisy at n=1.** Mean and median disagree materially here (fanoutskew b=8:
   +3.9 median vs −9.2 mean), so no single cell is trustworthy. The 5/6 sign
   pattern is the result, not the numbers.

The one cell that goes the other way — **fanoutskew b=16, −8.3%** — is the
largest budget tested and is the only support in this campaign for the
"lb wins at high N once the budget is large" hypothesis. It is a single noisy
cell and b=32/b=64 are unreachable at 16n (zero ok cells historically; lb_union
reproducibly hangs at b=64, `b17d826`), so the budget axis cannot currently be
pushed further at this node count. Treat it as unresolved, not as support.

Eager at 16n, same capsule, vs union: +11.7 / +10.1 / +21.8 (trace) and
+13.8 / +25.3 / −4.2 (fanoutskew) — consistent with §7.5b's verdict, and no
crossover appears at 16 nodes either.

*Lesson, same shape as §5.4:* a knob that exists "for measurement" silently
became part of a headline comparison. Always split historical pairs by
`env_json` before trusting a remembered trend.

---

## 6. Feedback, objections, and questions raised (recorded verbatim in substance)

These are the user's, and they shaped the framework above. Keep them; they are
the reviewer questions we will face.

**F1 — "The formalization doesn't account for overlapping."**
Overlapping inter/intra RDMA/NVLink, computation with communication, expert
weights — arguably the most important optimization we did, and it was missing
from the first framing. → Resolved into §3.4–§3.6: overlap is Σ→max, it is
coupled to merge through the engine axis, and merge is what creates the stages
it fills. *This resolution is new and should be checked against more data.*

**F2 — "Low budget, comm-dominated batch: reduce small fragmented puts; comm can
eat some skew and still show minimal latency."**
→ **Confirmed** by §5.1, and specifically it is the *coalesce* half, not the
dedup half: at b=2MiB, n≥8, dedup contributes nothing (+2.6, +1.2) while
coalescing contributes +147% / +238%.

**F3 — "The machinery might not be the cleanest code; is the perspective
actually new?"**
Assessment: the individual primitives are not new (hierarchical a2a = DeepEP,
overlap = Comet, placement = EPLB). What is not seen elsewhere:
(a) **weights as a first-class flow subject to the same primitives as tokens**
(weight multicast *is* dedup-merge; split weight prefetch across NICs *is*
split); (b) **weight-gated tiles** — Comet gates tiles on token arrival, gating
on *weight* arrival is a different dependency; (c) **interpolating the
placement** (virtual-expert mapping driving a flux-native fused path on a MoonEP
plan) instead of picking a corner; (d) the §5.3 dedup/coalesce separation.
Code quality does not affect novelty but does affect whether reviewers believe
the numbers and how artifact evaluation goes; the capsule discipline is the
mitigation and is unusually strong.

**F4 — "Not an EPLB-style successor; MoonEP is more relevant and newer, and has
per-batch balancing. But MoonEP re-prefetches every round — no persistent
replication — so maybe replicate only when enough changes arrive. This seems
orthogonal to e2e latency though."**
→ Direction accepted (per-batch, not EPLB successor). **Disputed as
orthogonal**: §4.4 shows it is the temporal-amortization cell of the same
`max(L_NIC, L_SM)` objective, and it has a clean 2-competitive ski-rental
solution requiring no prediction. §4.3's co-activation tension is retained as
theory but deprioritized as a build direction.

**F5 — "What is eager's actual mechanism? If it removes a false synchronization
dependency and increases overlap, shouldn't it be the default?"**
→ See §7. The intuition is right; the recalled mechanism was not.

**F6 — "What formalization does eager try to test?"**
→ See §7.1. It is a pure probe of the coupling term, and the cleanest
single-axis experiment we have.

---

## 7. `FLUX_A2AV_FANOUT` ("eager") — actual mechanism and why it is not default

Source: `src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc` ~L2682–2737.

The lb_union gateway forward loops `dn = 1 .. NN-1`, one round per remote source
node. Each round does `CUStreamWaitValue64(node_sig + ns)` — wait for that
node's relay chunk to land — then issues L `putmem_signal` forwards.

- **Default (knob off):** every round shares `tail_stream`. Streams execute in
  issue order, so round 1's *wait* blocks rounds 2..NN−1 even if their relay
  chunks have already landed. **Head-of-line blocking across gateway rounds.**
- **Eager (`FLUX_A2AV_FANOUT=1`, requires `FLUX_A2AV_LB_UNION`):** each round
  gets its own stream `fanout_streams_[dn-1]`; rounds fire in **arrival order**
  rather than fixed ascending-round order. They re-join `tail_stream` via
  `fanout_events_` so `all_gather_event` / the pre-barrier wait and the
  staging/recv reuse invariant are unchanged.

So the mechanism **is** removal of a false serialization — the user's intuition
was correct. It is **not** "launch the GEMM first, then start transfers"; that is
a *different* knob, `FLUX_A2AV_EARLY_LAUNCH` (same file, ~L321/L2113), which
defers the cp_stream wire ops as descriptors so the GEMM is enqueued first and
tiles spin on signals. (There is also an unrelated `FLUX_A2AV_RS_EAGER` in
layer1 `moe_gather_rs`.) **Three distinct knobs, easily conflated.**

### Why it is not the default

1. **The verdict was "not clearly better," and it was never re-run at scale.**
   Only capsules `20260807-055012` (isolated) and `20260807-055227` (nsys), both
   n=4 / b32 / k8 / conn=8. Eager 16.544 / 28.765 vs ring lb_union 17.188 /
   28.924 → **−3.7% and −0.5%** — and plain `hier_compress_union` (16.470 /
   28.226) was as good or better than eager in both. Marginal win over its own
   base, no win over the simpler arm. The code comment says to canonicalize or
   delete "once the eager-vs-ring capsule verdict is in the ledger"; that verdict
   was never produced.
2. **n=4 is the worst possible place to test it.** The head-of-line opportunity
   is `NN−1` rounds = **3**. At n=16 it is **15** — the mechanism should matter
   far more there, and it was never run there.
3. **It may self-defeat at scale via connection oversubscription.** Eager
   allocates NN−1 streams. The a2av family pins
   `CUDA_DEVICE_MAX_CONNECTIONS=8`; at n=16 that is 15 streams over 8 hardware
   channels, so streams multiplex onto shared channels and **re-serialize —
   reinstating the head-of-line blocking it removed, while adding overhead.**
   Newer (moonep) grids run `conn=32`, which would lift that constraint.
4. **It helps least where lb_union works best.** The benefit is a function of
   *arrival skew across source nodes*; the balanced wire exists precisely to make
   arrivals uniform. Also, the existing ring rotation
   (`dlg = (my_lr + 1 + dn + dl) % L`) already spreads first-window arrival at
   zero cost, targeting a related goal more cheaply.

### 7.1 What formalization eager tests

Eager is **not a merge variant** — it is the overlap primitive (§3.4) applied at
a new scope: *inside* the merge, across gateway rounds, instead of between comm
and compute. It is a **pure probe of the coupling term** (§3.3, §5.4).

What makes it unusually clean: eager is byte-identical to the ring order, issues
the same L puts across the same NN−1 rounds, uses the same engine, and changes
neither placement nor dedup opportunity. **Every `L_r` in the cost model is
unchanged.** The only variable is the ordering constraint among rounds. Most
sweep arms confound bytes, messages, and engine at once; eager varies one axis.

It therefore tests three nested claims:

1. **§3.3 — merge's true price is coupling, not bytes.** If coupling is the
   price, relieving it must pay.
2. **The coupling term grows with N.** Eager's benefit should be an
   increasing function of N, since it de-serializes N−1 rounds. Flat-in-N
   benefit falsifies the round-serialization microfoundation.
   *(This claim originally cited §5.4, which is now retracted — §5.4's evidence
   for a coupling term was an artifact. Eager is therefore the **only**
   remaining evidence for or against a coupling term at all, which raises rather
   than lowers the value of the n≥8 run.)*
3. **§3.6 — merge manufactures the stages that overlap fills.** The gateway's
   rounds *are* merge-created stages. If they cannot be filled, §3.6 is in
   trouble, since it is what makes merge and overlap non-orthogonal.

### 7.2 Why a null result is still informative: it discriminates two couplings

There are two distinct microfoundations for the coupling cost, and they demand
different primitives:

- **Inter-round coupling** — round 1's wait blocks rounds 2..NN−1 on the shared
  stream. Scales with NN−1. **This is what eager removes.**
- **Intra-round coupling** — a round still waits for its own relay chunk, which
  waits for the slowest of the L contributing local ranks. **Eager cannot touch
  this by construction.**

So: eager wins at n=16 ⟹ coupling is inter-round, it is reversible, and the
paper's claim upgrades from "dedup dies above n=8" to "dedup dies above n=8
*unless the gateway is decoupled*." Eager flat at n=16 *with adequate
connections* ⟹ coupling is intra-round, and the fix is not more streams but
**splitting the window** so a round forwards partial arrivals instead of waiting
for its whole relay chunk — a different primitive we would only know to reach
for because eager came back flat.

### 7.3 Eager would force a resource into the model

Eager's entire cost is NN−1 streams — nothing else. The model's resource vector
(§2) is currently `{NIC, NVLink, CE, SM, HBM}`, which does not include issue
channels. If eager wins at conn=32 and loses at conn=8, then:

> **Issue channels / hardware connections are a resource in R**, not an
> implementation detail.

Eager is the one primitive whose cost lands *only* on that resource, making it
the cleanest available probe for whether the resource vector is complete.

**Corollary for prioritization: eager's value is as an instrument, not as an
optimization.** Its mediocre n=4 perf verdict is not a reason to skip it; a
single-axis A/B is worth disproportionately more for validating a cost model
than for making anything faster.

### The experiment that would settle it

**Eager × {conn=8, conn=32} × {n=8, n=16} × {b=2, b=8}.** This is the untested
cell, it is cheap, and the framework predicts a specific result: eager's value
should grow with N (more rounds to de-serialize) **but only once connections
exceed NN−1**. If eager wins at n=16/conn=32 and loses at n=16/conn=8, that
confirms both the head-of-line mechanism and the channel-oversubscription
counter-mechanism in one capsule.

*(Amended 2026-08-15: this originally also promised to test "whether de-coupling
the gateway recovers the dedup benefit §5.4 shows is cancelled at n≥8". §5.4 is
retracted — on the synthetic families there is no cancelled benefit to recover,
because the ceiling is exactly zero at n≥8. The dedup question is only
answerable on the `trace` arm, which is why all four 8n/16n specs now carry
one.)*

### 7.4 Results — the N=4 anchor (capsules C1/C2, 2026-08-15)

First two capsules of the N-scaling campaign. Both from build
`ba9e91096019dfc0` (the twin gate passes, so they are comparable), isolated,
topk=8, G=128, 40 cells each, all `ok`. C1 = `20260815-044144_perlmutter_22befac3`
(conn=8), C2 = `20260815-045429_perlmutter_b10fc8dd` (conn=32). One matrix
instance per cell, so each number is a single measurement — read sign
consistency across cells, not individual magnitudes.

**Eager vs its own base `hier_compress_lb_union` (the single-axis A/B):**

| conn | family | b=1 | b=2 | b=8 | b=16 | b=32 |
|---|---|---|---|---|---|---|
| 8 | fanoutskew | +4.7 | +10.3 | +2.5 | +3.7 | +1.9 |
| 8 | uniform | +1.1 | +0.6 | +2.1 | +5.1 | +5.6 |
| 32 | fanoutskew | −0.2 | −1.1 | −2.6 | +3.7 | −1.2 |
| 32 | uniform | −16.5 | −4.4 | −1.4 | +3.2 | +4.3 |

At conn=8 eager is worse in **10/10 cells**. This is the *expected* low end of
an increasing-in-N curve: `NN−1 = 3` rounds is the least head-of-line blocking
there is to remove, and 3 extra streams + 3 event records + 3 join waits cost
more than they save. The anchor is doing its job, not failing.

**The conn control is the real result.** Comparing each variant against itself
across the twins (conn=32 vs conn=8, negative = conn32 faster):

| variant | extra streams | fanoutskew | uniform |
|---|---|---|---|
| `hier` | 0 | +2.4 +1.6 −12.5 +0.6 +0.1 | +0.2 −6.8 −2.4 −0.3 +0.4 |
| `hier_compress_lb_union` | 0 | +0.9 +4.4 +4.3 −0.3 +1.5 | +19.3 +3.3 −0.5 +1.0 +0.8 |
| `hier_compress_lb_union_eager` | NN−1 | **−3.8 −6.4 −1.0 −0.3 −1.6** | **−1.5 −1.9 −3.9 −0.8 −0.5** |
| `hier_compress_union` | 0 | +0.5 +0.0 +0.3 +0.1 −0.1 | +15.6 −0.9 +0.3 −0.5 +0.9 |

**Eager is the only arm that allocates streams and the only arm that
systematically benefits from more connections — 10/10 negative.** The arms that
allocate none are ~0 or noise. That is about as clean a confirmation of the
channel-oversubscription mechanism as this apparatus can produce, and it
supports promoting issue channels into the §2 resource vector (§7.3).

**Refinement it forces on the prediction.** §7's "only once connections exceed
NN−1" is *wrong as stated*: at n=4 eager needs only 3 streams, comfortably under
conn=8, yet still gains from conn=32. Eager's streams are not the only channel
consumers — the main stream, `cp_stream_inter_node`, the pack stream and the
GEMM all compete. So the binding comparison is **total concurrent stream demand
vs conn**, not `NN−1` vs conn. This makes the n=16 prediction *stronger*: 15
extra streams at conn=8 should be severely oversubscribed.

**Eager still never beats plain `hier_compress_union`**, at either conn (union
is −1.6 to −14.6% vs lb_union across both capsules). Consistent with the n=4
verdict from 2026-08-07. Eager's value remains as an instrument (§7.1), not as
a shipping default.

**Two side results, both new:**

1. **Dedup-merge goes negative at small budget.** vs `hier` at n=4/conn=8,
   `hier_compress_union` is −20.2% (fanoutskew) / −33.2% (uniform) at b=8 but
   **+2.4% / −0.0% at b=1**, and `lb_union` is outright **+10.3% / +12.8%** at
   b=1. §3.2 predicts this — dedup saves *bytes*, so where bytes do not bind
   (b=1 is the fixed-cost regime) its index/staging overhead is pure loss.
   **§5.2 should therefore claim decay on both axes, not only in N**: dedup dies
   as N grows *and* as budget shrinks. The b=1 point did not exist before this
   campaign.
2. **Dedup helps *less* under skew.** At b≥8, union vs hier is ≈−33% under
   `uniform` but only ≈−20% under `fanoutskew`. **RESOLVED — not a
   contradiction of §4.3.** Measured remote-traffic fraction over the stored
   matrices: `uniform` W=16 is **0.80**, `fanoutskew` is **0.50**. There is
   simply 60% more cross-node traffic to dedup under `uniform`. Moreover
   `fanoutskew` does not skew *expert popularity* at all — it skews how much
   each source node sends remotely and then spreads that remote traffic
   **uniformly over all remote ranks** (`w[d] = p / (W - L)`,
   `gen_matrix.py:_row_weights`), so collision probability *per remote byte* is
   identical to `uniform`. §4.3's claim concerns co-activation / expert-
   popularity concentration, which **no synthetic family here produces** — only
   the trace family could, and it has no instances above W=16 (§5.7).
   Side note for §5.1–§5.2: `uniform`'s remote fraction itself rises with W
   (0.80 → 0.90 → 0.95 at W=16/32/64), a confound to keep in mind, though it
   pushes both merges the same direction and so does not explain their opposite
   scaling.

### 7.5 Results — n=8 (capsules B1/B2, 2026-08-15)

`20260815-082516_perlmutter_45bda348` (conn=8) and
`20260815-084642_perlmutter_ca5a1fbf` (conn=32), 60/60 ok each, same build
`ba9e91096019dfc0`, isolated, topk=8, G=128, now including the **trace** arm.
Numbers below use `--stat schema-median` (§ analyzer) unless noted.

**Noise control first.** Cells are n=1 (one launch each), and 8n is much noisier
than 4n. Comparing mean-over-iterations against median-over-iterations separates
within-launch transients from real effects at zero cost: the eye-catching
**−29.7%** eager "win" at uniform/b2 collapses to **−3.7%** under the median (it
was inflated iterations in the *baseline* cell), while the large fanoutskew
penalties are stable (+25.8→+27.6, +15.6→+17.5, +16.2→+16.6). Quote the median.

**The conn discriminator — channel oversubscription confirmed, decisively.**
Eager vs `hier_compress_lb_union`, fanoutskew, 8 nodes:

| conn | b=1 | b=2 | b=8 | b=16 | b=32 |
|---|---|---|---|---|---|
| 8 | −2.9 | +0.5 | **+27.6** | **+17.5** | **+16.6** |
| 32 | −4.5 | −4.9 | **+2.0** | **−4.3** | **−1.6** |

The penalty **vanishes** with more channels. At n=8 eager wants `NN−1 = 7`
streams, which against conn=8 collides with the main stream,
`cp_stream_inter_node`, the pack stream and the GEMM. A competing hypothesis —
that eager converts serialization into *NIC bandwidth sharing*, delaying the
heavy round that sets the makespan under skew — is **falsified**: bandwidth
sharing would not care about `CUDA_DEVICE_MAX_CONNECTIONS`. §7.3 is upheld:
**issue channels belong in the §2 resource vector, and eager's entire cost is
denominated in them.**

**The N-curve, both signs of it.** Eager vs its base, median stat:

| conn | n=4 (mean of cells) | n=8 (mean of cells) |
|---|---|---|
| 32 (adequate) | ≈ **+0.2%** | ≈ **−3.3%** |
| 8 (starved) | ≈ **+3.7%** | ≈ +2.6% (fanoutskew alone: **+11.8%**) |

> **With adequate channels eager's *benefit* grows with N (≈0% → ≈−3%),
> confirming §7.1 claim 2 and that inter-round serialization is real. With
> starved channels its *penalty* grows with N, because its cost is denominated
> in a resource whose demand is `NN−1`.**

So eager is a legitimate default **conditional on channel headroom**, not
unconditionally — and the n=4/conn=8 verdict that shelved it in 2026-08-07 was
measured in exactly the starved corner.

**Dedup at 8 nodes under REAL routing (the §5.4b question).**
`hier_compress_union` vs `hier`, trace arm:

| conn | b=1 | b=2 | b=8 | b=16 | b=32 |
|---|---|---|---|---|---|
| 8 | +1.9 | +0.6 | −4.6 | −5.5 | −4.5 |
| 32 | +1.8 | +0.1 | −4.0 | −6.1 | −2.0 |

**These are two independent capsules agreeing cell-by-cell** — a replication,
which matters far more than n=1 per cell suggests. Meanwhile the *synthetic*
families disagree wildly between the same two capsules (fanoutskew b=1: −4.0 vs
+4.9; uniform b=2: +8.8 vs −6.7). That contrast is itself the confirmation of
§5.4b: where the dedup ceiling is 0%, `union` vs `hier` measures nothing but
gateway mechanics and behaves like noise; where it is 35.7%, the signal
replicates.

Three conclusions:

1. **Dedup-merge does pay at n=8 under real routing** (≈−4 to −6% at b≥8),
   contradicting §5.2's synthetic-derived "dies at n≥8". §5.4b's retraction is
   now confirmed *experimentally*, not just analytically.
2. **The budget axis survives.** Dedup pays only at b≥8 and is mildly harmful at
   b=1–2 (+1.8/+1.9 and +0.1/+0.6, also replicated), matching §3.2: dedup buys
   bytes, so it loses where fixed cost binds.
3. **Conversion is poor and that is the new live question.** A 35.7% byte
   ceiling yields ~5% latency. The byte saving is real, available, and largely
   *not* converting — which is where a coupling/overhead term genuinely belongs,
   unlike the retracted §5.4 where the opportunity never existed at all.

### 7.5b VERDICT — eager is a net loss; every earlier delta was against the wrong baseline

§7.5 quotes eager against its own base `hier_compress_lb_union`. That flatters
it, because `lb_union` is not the arm you would ship — plain
`hier_compress_union` is. Against **that** baseline (median stat):

| config | eager vs `union`, mean | cells lost |
|---|---|---|
| 4n conn=32 | **+8.1%** | 10/10 |
| 8n conn=8 | **+8.5%** | 14/15 |
| 8n conn=32 | **+1.5%** | 9/15 |

**Eager does not beat the best arm at any node count measured.** It repairs a
deficiency in an arm that is already the weaker design. As a shipping default:
no.

What survives, and it is the part worth keeping:

1. **The gap closes monotonically with N** at adequate channels: +8.1% (4n) →
   +1.5% (8n), and at 8n/conn=32 eager already wins 6/15 cells (−8.0, −8.5,
   −7.0, all at larger budgets). Extrapolation puts a crossover near 16 nodes —
   the pending A1/A2 point.
2. **The mechanism is confirmed and quantified** regardless of the verdict:
   inter-round serialization is real, and eager's cost is denominated in CUDA
   issue channels (+27.6% → +2.0% from `conn` alone, §7.5).
3. **§7.1 called this in advance** — eager's value is as an *instrument* for the
   coupling term, not as an optimization. It told us the resource vector was
   missing channels while losing on wall-clock. Both halves held.

*Statistical honesty:* cells are n=1 and these are single-digit effects, so the
weight is on sign counts (10/10, 14/15), not per-cell magnitudes. +1.5% at
8n/conn=32 is small enough that the defensible claim is **"eager ≈ union at 8
nodes given channel headroom"**, not "eager loses there".

### 7.6 The dedup ceiling is independent of budget — so budget acts on conversion

Exact ceilings on the *generated* trace matrices (`sweeps/dedup_factor.py`
methodology, computed directly from each `<mid>.routing.txt`):

| nodes | b=1 | b=2 | b=4 | b=8 | b=16 | b=32 | b=64 |
|---|---|---|---|---|---|---|---|
| 8 | 35.8% | 35.8% | 35.5% | 35.7% | 35.6% | 35.6% | 35.6% |
| 16 | 26.0% | 25.8% | · | 25.8% | 25.9% | · | · |

**Flat in budget, to within 0.3 points.** This decomposes §7.5's budget effect
cleanly: dedup-merge's *opportunity* does not vary with budget at all, so the
fact that it is mildly harmful at b=1–2 and worth ~−5% at b≥8 is entirely a
**conversion** effect — its fixed-cost overhead is constant while the bytes it
saves scale with the budget. That is §3.2's cost model confirmed on the
opportunity/realisation split rather than inferred from latency alone.

It also means **budget and node count act on different terms**: N sets the
opportunity (35.7% → 26.0% from 8 to 16 nodes), budget sets how much of it
survives overhead. The §3.2 claim that dedup and coalesce "anti-correlate on
both axes" was conflating these.

**Pre-registered prediction for A1/A2 (n=16), recorded before the run:** the
n=8 trace arm converted a 35.7% ceiling into ≈5% latency (~14% conversion). If
conversion is roughly N-invariant, n=16's 26.0% ceiling should yield **≈3–4% at
b≥8, and ≥0 (mildly harmful) at b=1–2**. A materially larger gain would mean
conversion *improves* with N; a null would mean the fixed overhead grows with N
fast enough to eat a still-substantial 26% opportunity — which would finally be
genuine evidence for a coupling term that grows in N, this time with the
opportunity actually verified present.

---

### 7.7 Pre-registration: node-aware placement + LocCap router (recorded before the run)

Campaign 8.19.theory (plan: node-aware PLACE-λ placement + per-token LocCap
replica routing + hier_compress/lb_union transports; implementation commits
71cdbd6..53ae4ad, all Python, binary identity preserved). Everything below
was computed offline (`sweeps/predict_placement.py`, zero GPU) and committed
BEFORE the first allocation. The simulator file-path-imports the exact
router module the GPU driver runs; the driver hard-asserts realized
`route_hash`/`incidence_remote` equal these simulations, so predicted ≠
realized is a determinism bug, never noise. Placement sidecars
(`<mid>.placement_{nodeaware,rankconc}_r2.json`, never-overwrite) carry the
per-(router, ε) numbers; `placement_sha` is the per-cell provenance fact.

**κ_conv calibration** (byte→latency conversion, `Δlat% / Δwirebytes%`,
from the committed same-capsule pairs hier vs hier_compress_union, 8n trace,
capsules 20260815-082516/-084642, iso mean-of-max, wire Δ = 33.3%):

| budget | κ (capsule 1) | κ (capsule 2) |
|---|---|---|
| b1 | −0.068 | −0.052 |
| b2 | −0.018 | −0.030 |
| b8 | +0.144 | +0.120 |
| b16 | +0.171 | +0.173 |
| b32 | +0.161 | +0.054 |

κ(b64) is EXTRAPOLATED ≥ 0.17 (monotone trend; Capsule B measures it
directly). Key structural reading: κ stays ≤ 0.17 even at b32 — latency is
never wire-dominated on this fabric — so the compute term stays binding and
the predicted ε* sits at the TIGHT end of the ladder (0.0625–0.125) at all
budgets, not rising with budget as the naive max-form model suggested.

**Predicted incidence tables** (internode dedup rows vs the fixed/d6
baseline; imb = max/mean rank rows; sidecars carry full ladders):

4n (W16, r2): baseline rows 11350 / 45510 / 363732 at b2 / b8 / b64; fixed
imb 1.87 / 1.852 / 1.86 (the b8 value reproduces the EPLB capsule's
1.852-before exactly — same matrix, independent code path).

| arm | b2 | b8 | b64 | imb (b8) |
|---|---|---|---|---|
| nodeaware / d6 | −35.7% | −35.4% | −35.4% | 1.30 |
| nodeaware / loccap ε=0.0625 | −44.5% | −43.5% | −43.6% | 1.06 |
| nodeaware / loccap ε=0.125 | −47.7% | −47.0% | −47.4% | 1.13 |
| nodeaware / loccap ε=0.25 | −54.3% | −52.8% | −53.6% | 1.25 |
| nodeaware / loccap ε=∞ | −60.8% | −60.6% | −60.6% | 1.60 |
| rankconc (any router) | −27.0% | −26.6% | −26.8% | 1.21 |

8n (W32, 8-entry interleaved pernode pools, r2): baseline rows 152534 at
b8; fixed imb 2.42. nodeaware/d6 −34.5% (imb 1.31); loccap ε=0.0625
**−48.4% at imb 1.062**; ε=0.25 −58.7%; ε=∞ −59.7% (imb 1.44); rankconc
−34.4% flat. Hot-node egress −41% at ε=0.0625. (8n b64 block appended in a
follow-up commit before the 8n capsule runs — its own pre-registration
precedes its own measurement.)

Structure is budget-flat (f_loc varies < 1pp across b2→b64 at 4n), matching
§7.6's ceiling-flatness: locality is a property of the routing, budget only
sets conversion.

**Falsifiable predictions** (all iso, same-capsule, twin-confirmed):

- **P1** (Capsule B): latency monotone in measured `incidence_remote` at
  fixed budget/transport; slope ≈ κ(b): ~0 at b2, 0.12–0.14 at b8,
  ≥ 0.17 at b64.
- **P2** (A2/B): realized byte reductions equal the table above EXACTLY
  (deterministic math; the in-driver assert enforces it cell by cell).
- **P3** (A2): ε ladder valley at 0.0625–0.125 for BOTH b8 and b64 (the
  κ-stays-low prediction); `lcinf` strictly worse than the valley at b64.
- **P4** (A1, 8n): nodeaware beats rankconc at equal slots through the
  loccap arms (predicted byte gap ~17pp at 4n, ~25pp at 8n); at 8n the
  gap opens ONLY through the router (d6 coverage ≈ concentration, 34.5 vs
  34.4%) — coverage without per-token routing is nearly worthless at 8n.
- **P5** (A1/B): `wire_ratio` on hc cells improves under nodeaware
  placement by ≈ the d6 rows above (−35% class) vs fixed/epic placement.
- **P6** (A1/B, b2): the largest byte-shift arm moves latency by less than
  the within-capsule spread (κ(b2) ≈ 0).
- **Magnitudes**: vs fixed-placement hc at ε*: ≈ −5.7% latency at b8
  (κ·44%), ≈ −7.4% at b64; vs nodeaware/d6 (router-only effect): ≈ −1% at
  b8 via κ, PLUS the balance channel (imb 1.29 → 1.08) that κ does not
  price — the two-channel decomposition is exactly what A1 vs A2 separates.

**NR-01 statement (which serialized work is removed):** placement is
one-shot/static (EPLB-class; `place_weights` bracket, ~0 recurring);
LocCap removes inter-node wire rows from the NR-14 head-of-line prefix,
reduces per-destination fan-out (the eager +27.6%@conn8 → +2%@conn32
resource), and cuts the hot node's max egress (−33..−41% predicted — the
max-rank latency setter). LocCap's own cost: the python port runs
once-per-cell under the untimed-metadata contract
(`epic_loccap_plan_host_ms` cell fact; measured 0.1–1 s at 4n scales —
same class as the other EP planner ports); the production implementation
is a reroute.cu-class GPU kernel (~0.1–0.5 ms), and this projection is
part of what P1–P6 must justify before any kernel is built.

## 8. Open questions

1. ~~**Does de-coupling the gateway recover the dead dedup benefit at n≥8?**~~
   **ANSWERED / VOID (2026-08-15).** The premise was false: on the synthetic
   families the n≥8 dedup ceiling is exactly 0%, so there is no cancelled
   benefit to recover and no coupling term is implied. See §5.4. The live
   replacement question is (1b).
1b. **Does dedup-merge still pay at n≥8 under REAL routing?** Real traces retain
   33.6% (n=8) and 17.5% (n=16) dedup headroom where synthetic has none, and no
   trace capsule above 4 nodes exists. This is now the single highest-value
   measurement in the campaign, and the `trace` arm of the 8n/16n specs is
   aimed at it.
2. **Is dedup actively harmful at high N?** At n=16/b=2MiB,
   `hier_compress_union` is **+1.2%** — slightly worse than no dedup. With the
   ceiling now known to be 0% there, this is dedup's *pure overhead* with zero
   offsetting benefit, which makes it a clean measurement of the cost side
   rather than a puzzle. Re-ask it on the trace arm, where the benefit is real.
3. ~~**Measure the real dedup factor offline.**~~ **DONE** — `dedup_factor.py`;
   it retracted §5.4 rather than confirming it (§5.4b).
3b. ~~**Why does dedup help less under skew than under uniform?**~~ **RESOLVED**
   in §7.4 side result 2: it is a remote-fraction effect (`uniform` 0.80 vs
   `fanoutskew` 0.50), and `fanoutskew` does not skew expert *popularity* at
   all. Note how this compounds with §5.4b: **no synthetic family produces
   co-activation concentration, and none has dedup headroom at nn ≥ topk** — so
   the `trace` arm is the only way to test either §4.3 or dedup-at-scale, and
   until this campaign no trace matrix above W=16 existed at all.
4. **Compute η (§3.4) across the existing phases capsules.** Gives a per-config
   overlap-efficiency number, conservative but reportable.
5. **Does the Dim-N weight split (§4.1) actually pay?** It is the framework's
   main *prescription* and is so far untested.
6. **Where does layer1 fit?** Its merge is a *reduce* (partial sums combined
   before crossing), a `topk→1` byte reduction that is associative and therefore
   unconditional. **Scatter-merge is conditional; gather-merge is
   unconditional.** Worth one line in the paper for symmetry.
7. **A slot exists for compress/quantize** (reduce bytes at compute/accuracy
   cost) in the same algebra; probably out of scope, but the framework should
   name the slot.

---

## 9. One-sentence thesis candidates

- "Every optimization here transfers work from a saturated resource to a slack
  one; the primitives are the legal transfers, and the scheduler picks the
  sequence."
- "Placement moves the lower bound; merge and split reshape it; overlap makes it
  achievable."
- "Prior work either moves the bound or closes the gap. A good scheduler must do
  both, because the placement that minimizes the bound is not the one that is
  easiest to overlap."
