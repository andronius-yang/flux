# MoonEP and UltraEP — Balancing the Computation, and Why Balanced Traffic Falls Out of It

**Scope**: `python/flux/testing/moonep_semantics.py` and
`python/flux/testing/ultraep_semantics.py` (the Perlmutter semantic ports, branch
`realistic-moe-input`), their drivers
`test/python/moe_ag_scatter/test_moe_{moonep,ultraep}_traffic.py`, the vendored oracles
under `test/python/moe_ag_scatter/{moonep,ultraep}_oracle/`, and the upstream sources they
are faithful to: the MoonEP checkout (`MoonshotAI/MoonEP` @ `7745ffa`, sibling directory
`../MoonEP/`) and the UltraEP checkout (`Dots-Infra/UltraEP` v1.0.0 @ `94cab09`,
sm80-oracle branch, sibling directory `../UltraEP/`), plus the UltraEP paper
(arXiv:2606.04101).

Companion docs:
[`comet_layer0_communication_patterns.md`](comet_layer0_communication_patterns.md) (the
flux layer0 this document constantly refers back to),
[`comet_layer1_communication_patterns.md`](comet_layer1_communication_patterns.md) (the
combine side), and
[`ep_semantics_vs_flux.md`](ep_semantics_vs_flux.md) (the measured comparison against the
flux/Comet arms, grounded in the four `20260808-*` sweep capsules).

This walkthrough is written beginner-first and progressively deeper: every mechanism is
introduced in plain language, then worked by hand on one small toy example that is reused
through the whole document, then stated in its real form with `file:line` citations, and
finally shown at production size using numbers from the committed sweep capsules. Nothing
technical is omitted — it is only sequenced after the intuition. Scope restriction: this
document explains **inference/forward semantics** (the lens the sweep harness measures —
one-shot, re-planned per activation). Training-side machinery (gradient buffers,
`reduce_grad`, backward dispatch/combine) is mentioned only where it explains a design
constraint, never elaborated.

---

## 1. The coordinate system: dispatch and combine, as flux already does them

### 1.1 What an MoE layer must do, in this repo's own vocabulary

A Mixture-of-Experts (MoE) layer replaces one big MLP with `E` small expert MLPs, and a
**router** (gating network) sends each token to its top-`K` experts. Under **expert
parallelism (EP)**, the experts are divided among `R` GPU ranks — each rank *homes*
`E/R` experts — so a token computed on rank 0 whose router picked an expert homed on
rank 5 must physically travel to rank 5, be multiplied by that expert's weights, and
travel back. That gives every MoE layer exactly two collectives, and this repo implements
both as fused Comet kernels:

- **layer0 / the dispatch side** (`src/moe_ag_scatter`): move every token's hidden vector
  to the rank(s) that will run its experts, land the rows *grouped by expert* so a grouped
  GEMM can consume them segment by segment, and run that GEMM
  (all-gather + scatter + grouped GEMM — see
  [`comet_layer0_communication_patterns.md`](comet_layer0_communication_patterns.md)).
- **layer1 / the combine side** (`src/moe_gather_rs`): run the second GEMM, gather each
  token's `K` expert outputs back to the token's home rank, and reduce them with the
  router's per-entry weights
  (grouped GEMM + gather + topk-reduce + reduce-scatter — see
  [`comet_layer1_communication_patterns.md`](comet_layer1_communication_patterns.md)).

Everything in this document is an alternative answer to those same two collectives.
MoonEP literally names its two public operations `dispatch` and `combine`
(`MoonEP/README.md:83-152`); UltraEP leaves dispatch/combine to the framework and inserts
itself *in front of* dispatch (`UltraEP/README.md:126-151`). Keep the flux picture in your
head as the fixed reference frame.

### 1.2 The straggler: why fixed expert placement makes skew expensive

With fixed placement, the router alone decides how many rows each rank's grouped GEMM
receives. Notation used throughout (MoonEP's, `MoonEP/README.md:43`): `S` tokens per rank,
`K` routed experts per token, so each rank emits exactly `S*K` routed entries and the
*average* rank receives `S*K` rows — but nothing forces the actual distribution to be
flat. If one rank's experts are "hot" this microbatch, that rank receives more rows, its
GEMM runs longer, and — because layer1 cannot combine until every expert output exists —
**the whole layer waits for the slowest rank**. That is the straggler problem. Two things
go wrong at once, and it is worth separating them now because the rest of this document
hinges on the distinction:

1. **Computation imbalance**: the hot rank's grouped GEMM has more rows (the straggler).
2. **Traffic imbalance**: the hot rank also *receives* more wire traffic (an incast).

Overlap-style systems (flux/Comet) attack the *visibility* of (2) by hiding communication
behind computation, but (1) survives overlap by construction: no amount of overlap makes a
14,497-row GEMM finish when an 8,832-row GEMM does. MoonEP and UltraEP instead attack (1)
directly — they change *where the expert computation runs* — and, as we will see in §3,
(2) then improves as a **side effect** of the relocation, without ever being the
optimization objective.

### 1.3 The toy example used for the rest of this document

Four ranks, eight experts (two homed per rank), four tokens per rank, top-2 routing:

```
R = 4 ranks       homing: rank0 {e0,e1}  rank1 {e2,e3}  rank2 {e4,e5}  rank3 {e6,e7}
S = 4 tokens/rank, K = 2   =>  S*K = 8 routed entries per rank, 32 globally
```

Routing this microbatch (each token lists its top-2 experts):

```
rank0:  t0 (e2,e0)   t1 (e2,e1)   t2 (e6,e7)   t3 (e2,e1)
rank1:  t4 (e2,e3)   t5 (e2,e3)   t6 (e2,e0)   t7 (e3,e2)
rank2:  t8 (e2,e4)   t9 (e2,e5)   t10 (e4,e5)  t11 (e2,e6)
rank3:  t12 (e2,e7)  t13 (e6,e7)  t14 (e2,e6)  t15 (e6,e2)
```

Per-expert global token counts (add up the mentions):

```
e0: 2   e1: 2   e2: 13   e3: 3   e4: 2   e5: 2   e6: 5   e7: 3     (total 32 ✓)
```

Expert `e2` is hot — 13 of 32 entries, over 5× its fair share. Per-rank *received* rows
under fixed placement (each rank gets the tokens of its two homed experts):

```
rank0 (e0,e1):  4      rank1 (e2,e3): 16      rank2 (e4,e5):  4      rank3 (e6,e7):  8
```

Rank 1 receives 16 rows where the fair share is 8: its grouped GEMM does 2× the average
work, and the layer runs at rank 1's speed. Imbalance (max/mean) = 16/8 = **2.0**. This
toy is not exaggerated: the real Qwen3-235B decode trace at layer 92 used by the sweep
capsules shows imbalance 1.85 at EP16 (§4.2, §5.4), and the UltraEP paper reports
1.30–4.01 in production workloads (arXiv:2606.04101, abstract).

> **Conceptual anchor.** Under fixed expert placement, the router *is* the load balancer,
> and it is not a good one on any single microbatch. Every system in this document is a
> different answer to the question: *who gets to override the router's placement decision,
> when, and at what price?*

---

## 2. Why routing skew is the enemy, and why prediction is not enough

### 2.1 The ideal–reality gap

Frameworks are usually benchmarked under force-balanced routing — every rank receives a
near-identical row count — because that is the throughput ceiling of the hardware. Real
routing does not cooperate: even with algorithm-side aux losses keeping expert usage
balanced *statistically*, the load within any single microbatch swings hard, and the gap
between force-balanced ideal and achieved throughput reaches **up to 2×**
(`UltraEP/docs/blog_v1.md:13`). The finer-grained the MoE (hundreds of small experts, few
homed per rank), the fewer experts each rank holds and the less the per-rank sum
self-averages — large EP *amplifies* skew rather than smoothing it
(`UltraEP/docs/blog_v1.md:29`).

This is also the scale regime that motivated MoonEP. Kimi K3 (Moonshot AI, July 2026) is
a 2.8T-parameter MoE activating 104B parameters — **16 of 896 experts** per token, plus 2
shared experts — and its technical report credits its "Stable LatentMoE" architecture plus
infrastructure work for ≈2.5× scaling efficiency over Kimi K2
([github.com/MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3),
`k3_tech_report.pdf`). MoonEP is the EP communication library Moonshot open-sourced
alongside K3 to keep expert-parallel dispatch efficient *under* imbalance; the K3 report
itself says little about MoonEP's internals — the algorithm detail lives in
`MoonEP/README.md` and the executable planning oracle vendored into this repo (§4).

### 2.2 Predictive balancing, and why both libraries reject it

The established fix is **predictive re-placement**: EPLB (DeepSeek, 2025) periodically
re-shuffles which rank homes which expert based on a *history window* of routing
(`UltraEP/docs/blog_v1.md:36`). That works exactly as well as expert load is stationary —
and for fine-grained MoE under real traffic it often is not, so the hot/cold prediction
misses and the balancing operation loses its effect (`UltraEP/docs/blog_v1.md:38`). LPLB
(DeepSeek, 2025) patches reroute on top of EPLB with linear programming, but inherits
EPLB's placement bias and hits solvability limits (`UltraEP/docs/blog_v1.md:113`).

MoonEP and UltraEP make the same, more aggressive choice: **plan from the exact
post-gating load of the current microbatch, every layer, every microbatch** — zero
prediction error, but the planning and the weight movement land on the critical path and
must be driven to near-zero cost (`UltraEP/docs/blog_v1.md:42-44`;
`MoonEP/README.md:7-8`, "planned online from the current router outputs"). This repo's
sweep harness honors the same discipline: routing schedules are never cached across
iterations — inference re-plans per activation (see the no-schedule-caching rule the
sweep protocol enforces; `docs/handoff/02_algorithm_state_and_next_moves.md:359-360`,
"plan-reuse was dropped: inference re-plans per activation").

Where they diverge is *what* they re-plan (§3), and *how far* the fix is allowed to reach:
UltraEP deliberately confines expert replication to the NVLink scale-up domain — its paper
frames the whole system around **rack-scale nodes** where an entire EP group fits in one
high-bandwidth domain (arXiv:2606.04101; `UltraEP/docs/blog_v1.md:53`) — while MoonEP's
planner is global across the EP group (its kernels, however, are NVLink/NVSwitch-domain
implementations; `python/flux/testing/moonep_semantics.py:11-12`).

---

## 3. The shared move: relocate the expert kernels; balanced traffic falls out

### 3.1 One idea, two mechanisms

Both systems change **where the expert GEMM for a hot expert runs**:

- **MoonEP** migrates *tokens*: the planner gives under-loaded ranks a byte-exact quota of
  the hot expert's tokens, ships those tokens there instead of to the home rank, and
  prefetches the expert's *weights* to the receiving ranks so the GEMM can run where the
  tokens landed. Result: every rank computes exactly `S*K` rows
  (`MoonEP/README.md:7`).
- **UltraEP** replicates *experts*: the solver instantiates replicas of hot experts in
  spare "redundant slots" on under-loaded ranks, assigns each instance a token **quota**,
  syncs the master's weights to the replica slots, and reroutes tokens
  logical→physical (`python/flux/testing/ultraep_semantics.py:3-8`).

These are two parameterizations of the same underlying move — put the kernel where the
capacity is — differing in the constraint they hold fixed: MoonEP holds the *row count*
fixed (exactly `S*K` everywhere, perfect balance by construction), UltraEP holds the
*expert placement machinery* fixed (masters never move; only bounded replicas appear) and
drives the row count toward a solved threshold.

### 3.2 Watch the wire matrix change as a consequence

Here is the punchline the section title promises, worked on the toy. Count wire **rows**
(off-diagonal `src→dest` hidden-vector transfers) under all three regimes. Under fixed
flux placement, every `e2` entry flows to rank 1:

```
flux (fixed placement)          MoonEP (token migration)        UltraEP (replication+quota)
   to: r0 r1 r2 r3                 to: r0 r1 r2 r3                 to: r0 r1 r2 r3
from r0  – 3  0  1*            from r0  –  0  0  1*           from r0  –  0  0  2
from r1  1  –  0  0            from r1  2  –  0  0            from r1  1  –  0  0
from r2  0  3  –  1            from r2  0  2  –  1            from r2  1  0  –  1
from r3  0  3  0  –            from r3  0  0  3  –            from r3  2  1  0  –
recv:    1  9  0  2            recv:    2  2  3  2            recv:    4  1  0  3
total wire rows: 12            total: 9                       total: 8
```

(`*` = token `t2 (e6,e7)`: both entries land on rank 3, so MoonEP and flux's dedup send
**one** row for two entries; UltraEP sends two — see §4.5 and §5.5. The UltraEP column
uses the locality-aware quota split of §5.5, which is what keeps rank 0/1/2's own hot
tokens local.)

Read the `recv` row: flux funnels 9 of 12 wire rows into rank 1 (traffic incast mirrors
the compute hotspot); MoonEP's receive vector is [2,2,3,2]; UltraEP's is [4,1,0,3]. Nobody
optimized the wire matrix — MoonEP's planner objective is "every rank receives exactly
`S*K` rows" and UltraEP's is "minimize the load threshold τ" — yet the traffic flattened
(and even *shrank*, because relocation lets hot tokens stay closer to their source).
**Balanced traffic is the shadow cast by balanced computation.** The sweep harness
records this honestly: `moonep_wire_bytes` "differs from the input matrix BY DESIGN,
MoonEP rebalances" (`sweeps/SCHEMA.md:139-140`).

The price of the move is a new, third kind of traffic that flux does not have at all:
**weight movement** (MoonEP prefetch, §4.6; UltraEP weight_sync, §5.6). The entire
engineering of both libraries is about making that price small; the entire fairness
question when comparing them to flux is about where that price shows up (see
[`ep_semantics_vs_flux.md`](ep_semantics_vs_flux.md)).

---

## 4. MoonEP's dispatch: every rank receives exactly S·K rows

### 4.1 The contract, stated against flux layer0

Flux layer0's contract is: deliver every routed token row to the rank homing its expert,
expert-grouped, and let the GEMM start on each segment as it arrives. MoonEP's dispatch
keeps the "expert-grouped rows ready for a grouped GEMM" half and *replaces the
destination rule*: destinations come from a per-microbatch **plan**, not from homing.
Three properties are the contract (`MoonEP/README.md:7-9`):

1. **Perfect balance** — every rank receives exactly `S*K` rows regardless of skew.
2. **Online planning** — the plan is computed from the current router output.
3. **Zero copy and static shapes** — rows land directly in a fixed `[NvS, H]`
   expert-grouped buffer whose views feed the GEMM; the consumer needs only
   `cu_seqlens[E+B]` to know which segments are live. Statically-known shapes also
   eliminate the per-layer host synchronization flux-style dynamic dispatch needs
   (`MoonEP/README.md:9`), and — one sentence on the training benefit — fixed activation
   shapes are what let MoonEP's end-to-end training survive high imbalance without memory
   fragmentation and OOM (`MoonEP/README.md:31-32`).

The port's planner is a vectorized re-implementation of MoonEP's own executable oracle,
vendored verbatim at `test/python/moe_ag_scatter/moonep_oracle/planning_reference.py`,
with a **bit-identical fidelity contract** enforced by `test_moonep_planner.py`
(`python/flux/testing/moonep_semantics.py:14-19`). Everything below is therefore not "the
port's interpretation" — it is the algorithm MoonEP's production kernel is itself tested
against.

### 4.2 How hot ranks are found: the balance vector and the z matrix

**Plain language.** A rank is "hot" when the tokens routed to its homed experts exceed
its fair share. MoonEP defines fair share exactly: `S*K` rows (what a rank would receive
under perfect balance, and — not a coincidence — the size of its receive buffer).

**Toy, by hand.** Per home-group token counts from §1.3:

```
group_tokens = [4, 16, 4, 8]        CAP = S*K = 8
balance      = group_tokens − CAP = [−4, +8, −4, 0]        (sums to 0, always)
```

Positive balance ⇒ overloaded home group ("hot rank"); negative ⇒ that rank has room.
The sum is zero because every source rank emits exactly `S*K` entries
(`moonep_oracle/planning_reference.py:72-75`). Now the greedy pairing
(`python/flux/testing/moonep_semantics.py:169-177`, oracle-verbatim):

```python
while True:
    h = balance.argmax()      # most overloaded home group
    u = balance.argmin()      # roomiest destination rank
    if balance[h] <= 0: break
    move = -balance[u]        # fill the receiver COMPLETELY
    z[h, u] = move
    balance[h] -= move
    balance[u] = 0
```

Trace it: `h=1 (+8), u=0 (−4)` → `z[1,0]=4`, balance `[0,+4,−4,0]`; then `h=1, u=2` →
`z[1,2]=4`, balance all zero; loop exits. The **migration matrix**:

```
z[1] = [4, 0, 4, 0]      "home group 1 exports 4 rows to rank 0 and 4 rows to rank 2"
```

Note the policy: receivers are filled *fully* (`balance[u] = 0`), so each round retires
one receiver; the loop runs at most `R` times. Ties break toward the lowest index
(`argmax`/`argmin` semantics), which is part of the bit-fidelity contract.

**Real instance.** The layer-92 Qwen3 trace at EP16 (capsule
`20260808-011748`, cell fact `moonep_z_matrix`) produces exactly this shape at scale — a
mostly-zero matrix with a few large migrations:

```
z[2]  = [.., 3116 → rank3, .., 3347 → rank12, ..]     z[6]  = [.., 2230 → rank4, ..]
z[5]  = [818 → rank0, 158 → rank2, ..]                z[15] = [2464 → rank1, 163 → rank6, 4355 → rank7]
```

Seven of sixteen home groups export; the hottest (group 15, +6982 over CAP) fills three
receivers. Compare the *before* picture — the flux arms' received-row counts on the same
routing ranged 3,837–15,172 — with the *after*: §4.3's fingerprint.

### 4.3 How hot experts are chosen inside a hot group, and the alloc table

**Plain language.** `z` says how many rows group `h` must shed and to whom — but not
*which experts'* rows. MoonEP answers: always shed from the expert with the most
remaining tokens (the hottest one), because fewer distinct migrated experts means fewer
weight prefetches later.

**Toy.** Home group 1 holds `e2` (13 tokens) and `e3` (3). Quotas `[4,0,4,0]`
(`python/flux/testing/moonep_semantics.py:179-195`): each round picks the largest
remaining quota and the largest remaining local expert — `d=0`: take 4 from `e2`;
`d=2`: take 4 from `e2`. Result, as the **allocation table** `alloc[e, d]` (tokens of
expert `e` computed on rank `d`):

```
alloc[e2] = [4, 5, 4, 0]        (13 conserved: 4 migrated to r0, 4 to r2, 5 stay home)
alloc[e3] = [0, 3, 0, 0]        (untouched — e2 alone covered the quota)
```

Per-rank received rows: r0 = 2+2+4 = **8**, r1 = 5+3 = **8**, r2 = 2+2+4 = **8**,
r3 = 5+3 = **8**. Perfect balance, from a 2.0-imbalanced routing. Two invariants are
asserted after the loops (`moonep_semantics.py:197-200`): per-expert conservation
(`alloc.sum(1) == expert_count`) and capacity (`alloc.sum(0) <= CAP`).

**Real instance.** The balance fingerprint at EP16, layer 92:
`gemm_rows_per_rank = [8832, 8832, 8960, 8832, 8576, ...]` — every rank within one
padding quantum of 8192 = `S*K` real rows (§4.4 explains the +pad). The flux arms on the
identical routing: `[7374, 5728, 14497, ..., 15172]`. That flat-vs-jagged pair of vectors
*is* MoonEP working (`sweeps/SCHEMA.md:141-142`).

### 4.4 The physical receive layout: VM groups, cu_seqlens, and the dst encoding

**Plain language.** Knowing *how many* rows each rank receives is not enough — a grouped
GEMM needs rows *contiguous per expert*, and MoonEP additionally promises a *static*
buffer with *no receiver-side index computation at dispatch time*. So the planner
pre-computes, for every row of every rank, the exact slot it will occupy.

**The layout.** Each destination rank's `[NvS, H]` buffer is a sequence of **VM groups**:
group ids `0..E-1` are the global experts (only this rank's *local* experts will be
non-empty here), and ids `E..E+B-1` are **prefetch slots** holding the segments of
migrated remote experts (`moonep_semantics.py:227-256`; `B` = prefetch slots per rank,
default `E/R`). Every non-empty segment is padded up to `token_padding` (default 128) so
segment starts stay aligned for the GEMM; `cu_seqlens[E+B]` records padded ends,
`zero_fill_ranges` records which pad rows must be zeroed
(`moonep_oracle/planning_reference.py:145-151` — and note the master behavior: the tail
beyond the last padded end is *undefined*, consumers only ever read via `cu_seqlens`).

The buffer height is `NvS = S*K + (token_padding−1)·2·(E/R)` — capacity plus worst-case
padding for at most `E/R` local + `E/R` remote non-empty segments
(`moonep_semantics.py:59-64`). At the capsule shape (S=1024, K=8, E=128, R=16, pad 128):
`NvS = 8192 + 127·2·8 = 10224` — the static layout costs ~25% extra rows over the real
8192 (cell fact `moonep_nvs = 10224`), and padded rows are *computed* by the GEMM (the
MoonEP contract; `moonep_semantics.py:691-693`) — the price of never re-shaping.

**Toy slot assignment for `e2`.** Every routed entry gets a destination by pure prefix
arithmetic (`moonep_semantics.py:262-264`; oracle loop at
`planning_reference.py:211-241`): concatenate all ranks' `e2` entries in source-rank
order into one global sequence `g = 0..12`; the destination is the first rank whose
`alloc_cumsum[e2] = [4, 9, 13, 13]` exceeds `g`:

```
g:      0  1  2 | 3  4  5  6 | 7  8  9 | 10 11 12      (r0's 3, r1's 4, r2's 3, r3's 3)
dest:   0  0  0 | 0  1  1  1 | 1  1  2 |  2  2  2
```

Notice two things worth pausing on. First, rank 0's own `e2` tokens go to *rank 0's*
prefetch slot — the migration quota is partly satisfied by tokens that never move
(accidental locality). Second, rank 1 *exports* its first `e2` token to rank 0 while
simultaneously *importing* `e2` tokens from rank 2 — the contiguous-slice rule is blind to
sources. That is a deliberate trade: destinations computable from two prefix sums and one
`searchsorted`, no per-token negotiation, at the cost of a few avoidable wire rows.

**The encoding.** Each entry's slot is packed into one int32:
`dst = dest · NvS + expert_off[dest, e] + position_within_segment`
(`moonep_semantics.py:298`; overflow guard at `:66-68`). High "digits" say *which rank*,
low digits say *which row of its buffer* — the entire dispatch, on every rank, is then
just "scatter row `i` to `dst[i]`". This is the static-shape, zero-index-computation
promise made concrete.

### 4.5 Wire dedup: one row per (token, destination rank)

**Plain language.** If both top-2 entries of a token land on the same destination rank
(likelier than you'd think — hot experts co-occur), sending the hidden vector twice is
pure waste: the receiver can copy it locally.

**Toy.** Token `t2 (e6,e7)` from rank 0: both experts live on rank 3. One row crosses the
wire; rank 3's *duplicate expansion* copies it from the `e6` segment slot into the `e7`
segment slot afterwards.

**Mechanism.** Within one token's top-K, only the first entry per destination keeps its
slot offset; later ones are encoded `−raw − 1` (`moonep_semantics.py:300-312`). Why not
just drop them? Because every top-k entry carries its *own route weight* which must reach
the combine's weighted reduce, so the raw slot must stay recoverable — the `−1` merely
guarantees the encoded form is negative (`moonep_oracle/planning_reference.py:250-255`).
The weights of *all* entries (dedup'd included) travel on a separate small fp32 channel
(`moonep_semantics.py:370-373`), and the receiver's dedup structures
(`dup_groups`/`dup_loffs`) drive the local copy-out
(`moonep_semantics.py:314-341`, `:447-455`).

One subtlety the harness is careful about: dedup savings depend on *real token overlap*.
The sweep's synthetic matrix dealer maximizes overlap by construction, so headline
numbers only come from real routing traces (both EP drivers print a warning otherwise;
see `sweeps/gen_trace_routing.py` doc-comment). On the real layer-92 trace the realized
dedup'd wire matrix is the `moonep_wire_bytes` cell fact.

### 4.6 Weight prefetch: the price of moving the kernel

**Plain language.** Rank 0 received 4 tokens of `e2` — but rank 0 does not *have* `e2`'s
weights. Before its GEMM reaches the prefetch slot, the weights must arrive from the home
rank. This is the new traffic class that token migration creates.

**Which experts get prefetched.** For each destination, the remote experts with tokens
are sorted by token count descending (ties → higher id) and the top `B` get prefetch
slots (`moonep_semantics.py:215-224`) — the **B hottest remote segments**. With MoonEP's
default `B = E/R` this covers *every* migrated expert, because the constructive planner
sends each destination tokens from at most one home group (≤ `E/R` distinct experts) —
the port asserts exactly this (`moonep_semantics.py:542-552`). For **inference** MoonEP
recommends `B = 3–4`: any overflow expert's weights are simply read *through NVLink
symmetric memory* from the home rank during the GEMM — slower, still correct
(`MoonEP/README.md:56-59`). (`B = E/R` is *mandatory* only in training, so that every
touched expert is local for the gradient path — the one sentence of training detail we
need.)

**Toy.** `experts_to_copy`: rank 0 → `[e2, −]`, rank 2 → `[e2, −]`; home rank 1 serves
both transfers. **Real instance:** at the capsule shape each rank prefetches up to 8
experts' fc1 shards; rank 0 received `moonep_prefetch_recv_bytes = 33,554,432` (32 MiB)
per activation, and the prefetch phase costs ~4.7 ms serialized — or mostly hides under
dispatch on a separate stream (`moonep_overlap`: 8.89 ms total vs 10.48 serialized,
capsule `20260808-032217`; details in
[`ep_semantics_vs_flux.md`](ep_semantics_vs_flux.md) §3).

### 4.7 MoonEP's combine, against flux layer1

Flux layer1 gathers each token's `K` expert outputs back to the token's home rank and
reduces them with the route weights. MoonEP's `combine` is the same collective run off
the *saved plan*: outputs leave in `[NvS, H]` VM-group order, each representative row
travels back to the source, and the top-k weighted sum lands token-major `[S, H]`
(`MoonEP/README.md:130-140`). Dedup runs in reverse for free: a dedup'd entry never had a
distinct row — its contribution is reconstructed at the source from the representative
row and the per-entry weight that rode the meta channel (which is why the raw `dst` had
to stay recoverable, §4.5). The flux port implements and measures only the dispatch side
(layer0); combine's semantics are fixed by the same plan, and porting it is listed as an
open next move in [`ep_semantics_vs_flux.md`](ep_semantics_vs_flux.md) §5.

### 4.8 Memory consumption (inference)

Everything is static; nothing is allocated per step. Per rank:

| Buffer | Shape | Toy | Capsule shape (S=1024, K=8, E=128, R=16, H=4096, bf16) |
|---|---|---|---|
| Token receive buffer | `[NvS, H]` | 20×H | 10224 × 4096 × 2 B ≈ **80 MiB** (25% of it padding) |
| Route weights | `[NvS]` fp32 | — | 40 KiB |
| Weight tensor, per projection | `[E+B, H, H']` contiguous | `[10, H, H']` | rows `[0,E)` = every rank's local experts via symmetric memory; rows `[E, E+B)` = prefetch slots (`MoonEP/README.md:51-54`) |
| Prefetch pool | `B` experts **total, process-global** | — | shared across *all layers* — `B` extra expert weights per projection in total, not per layer (`MoonEP/README.md:54`) |

The `[E+B, H, H']` contiguity is a hard requirement — the grouped GEMM addresses experts
purely by row index (`MoonEP/README.md:51`) — and it is what makes the prefetch-slot
trick free at GEMM time: a migrated expert's segment just points at row `E+b` instead of
row `e` via `cu_seqlens`. The port mirrors the shapes it needs
(`moonep_semantics.py:527-540`: `hidden_buf [NvS, H]`, `prefetch_w [B, ffn_shard, H]`,
staging `send_buf`/`recv_buf` sized to the plan's exact counts). The headline memory
story vs flux-style dynamic dispatch: ~25% row padding and one pooled set of prefetch
weights, bought back by *zero* dynamic allocation, no fragmentation, and no host sync on
shapes.

---

## 5. UltraEP's dispatch: replicate the expert, split its tokens by quota

### 5.1 The contract, stated against flux layer0

UltraEP does not replace the dispatch collective at all — it is a *balancing layer in
front of* an ordinary dispatcher (DeepEP-style in production, the same staged a2av as the
MoonEP port here). What it changes is the **routing map** the dispatcher sees. Each rank
gets `nlp = E/R + R_red` **physical expert slots**: its `E/R` immutable *masters* plus
`R_red` (default 2) *redundant slots* that any hot expert's replica may occupy this
microbatch (`ultraep_semantics.py:85-87`; `UltraEP/docs/blog_v1.md:70`). Per layer, per
microbatch (`ultraep_semantics.py:3-8`):

1. all-gather the exact post-gating load table `tpe[R, G]` (~8 KiB — *not* the full
   routing),
2. solve which experts get replicas where, and each instance's token **quota**,
3. sync master weights into the replica slots (NVLink-confined),
4. **reroute** every token logical→physical, then dispatch as usual.

No dedup, no static token buffer, unpadded segments: the receive side is exactly as big
as the solved load, and the residual imbalance *is* the measurement
(`ultraep_semantics.py:967-969`). Fidelity here is even stronger than MoonEP's: the port
is bit-identical to UltraEP's **real CUDA kernels** (`quota_placement_solve_kernel`,
`dense_quota_reroute_scatter_kernel`), enforced on the SM80 oracle build and on frozen
goldens under `test/python/moe_ag_scatter/ultraep_oracle/`
(`ultraep_semantics.py:14-32`).

### 5.2 The problem, formalized (from the paper)

The paper (arXiv:2606.04101 §4.3) formalizes what §3 said in prose. Variables: the load
matrix `Λ = {λ_{r,e}}` (tokens from source rank `r` to logical expert `e` — exactly the
port's `tpe`), the **quota table** `U = {u_{e,r}}` (`u_{e,r} > 0` iff rank `r` hosts an
instance of `e`, and says how many tokens that instance serves), the slot assignment `X`,
and the reroute split `Q = {q_{r,e,t}}`. The forward objective (Eq. 1) is the critical
path `T_solve + max(T_reroute, T_weight_distr) + T_a2a + T_moe`, with the two loads that
matter modeled as (Eqs. 3–4):

```
T_moe  ∝  max_r Σ_e u_{e,r}                      — the busiest rank's GEMM rows
T_a2a  ∝  max_r max( Σ_e λ_{r,e},  Σ_e u_{e,r} ) — busiest sender or receiver
```

— i.e. **minimizing the max post-reroute rank load simultaneously attacks compute and
traffic**, which is the formal version of "balanced traffic falls out". Constraints:
masters immutable, `≤ R_red` replicas per rank, no expert twice on one rank, every new
replica's quota ≥ `u_min` (default 1024 — a replica must be *worth its weight sync*).
The solver seeks the minimum **threshold τ** such that, defining per-rank
`exc_r(τ) = max(ℓ_r − τ, 0)` and `slk_r(τ) = max(τ − ℓ_r, 0)`, all excess can be
reassigned into slack without violating the constraints. The key design idea
(`UltraEP/docs/blog_v1.md:115`): don't solve placement then reroute — solve the **quota**
directly, coupling both; each accepted step simultaneously instantiates a replica and
fixes its final load.

### 5.3 The solver: a feasibility oracle inside a threshold search

**Plain language.** "Can everyone get under load τ?" is easy to check greedily; the
smallest feasible τ is then found by searching over τ. UltraEP wraps one greedy
**export oracle** in a three-stage search tuned to exit early on easy instances.

**Toy, by hand** (one 4-rank domain, `R_red = 1`, toy `u_min = 1`; port:
`_solve_domain`, `ultraep_semantics.py:310-408`). Rank loads `[4, 16, 4, 8]`, mean 8.

*Bounds*: `lo = ceilf(32/4 · 1.0) = 8`, `hi = max-load = 16`.

*Stage 1 — fast-eps probe* (`:345-354`): try `τ = ceilf(lo·1.01) = 9` up to three times,
stepping by `(hi+99)//100`. Run the oracle at τ=9:

- `excess = [0, 7, 0, 0]`, `slack = [5, 0, 5, 1]` — necessary condition 7 ≤ 11 holds.
- Visit **source ranks in descending load** (`[r1, r3, r0, r2]` — this is how "hot
  ranks" are ordered here, the analogue of MoonEP's `argmax`), and within a rank its
  **experts in descending load** (`:320-326`): rank 1 must shed 7; its hottest expert is
  `e2` (13).
- The fast path uses a **dynamic quota floor** (`:250-261`): with 3 admissible slots
  (r0, r2, r3), the floor becomes `ceil(7/3) = 3` — spread the excess rather than dump
  it.
- **Target choice** = admissible rank with **maximum remaining slack**, ties → lowest
  rank (`:263-277`): r0 (slack 5). Quota `q = min(need 7, slack 5, available 12) = 5` →
  replica `(e2 → r0, q=5)`. (`available` keeps ≥1 token on the master unless
  `allow_zero_master_quota`; `:239-246`.)
- Next round: floor `ceil(2/2)=1`, target r2 → `(e2 → r2, q=2)`. Need met. **Feasible at
  τ=9, path="fast"** — the search never runs.

Solved placement: `e2` master on r1 keeps quota 13−7 = 6; replicas on r0 (q=5, in r0's
redundant slot) and r2 (q=2). Solved rank loads `[9, 9, 8, 8]` — imbalance 2.0 → 1.125.
Not MoonEP's exact `[8,8,8,8]`: replication is quantized by experts, slots, and the
quota floor. That gap is intrinsic, and the real solver reports which path produced each
domain's plan so you can audit it (`DomainSolution.path`, `:96-104`).

**The full ladder, for hard instances** (`:356-408`): if all fast probes fail —
(2) a coarse *capacity bisection* using only the necessary condition
`Σ excess ≤ Σ slack` narrows `[lo, hi)` cheaply; (3) a full-oracle *precheck* at that
bound often terminates ("precheck" path); (4) otherwise a **4-probe parallel bisection**
— the CUDA kernel's four warps each test a probe at `lo + range·(w+1)/5`, the first
feasible lane tightens both bounds at once ("bisect" path). On GPU the whole solve is one
cooperative thread block per domain, load table in shared memory, and stays ~100 µs at
EP64 (arXiv:2606.04101 §5.3; `UltraEP/docs/blog_v1.md:117`; measured
`update_placement` 0.067–0.078 ms, `UltraEP/README.md:171-180`).

**Real instance** (capsule `20260808-090536`, layer 92, EP16, D=4): four independent
domain solves, thresholds `T = [8251, 7365, 9903, 7692]`, all four via the `"fast"` path
— on real routing the 1%-over-ideal probe almost always succeeds immediately, which is
exactly why the fast path exists.

### 5.4 NVLink-domain confinement: the reachable floor, and the rack-scale thesis

**Plain language.** A replica is only worth creating if copying the expert's weights to
it is nearly free — so UltraEP only ever places a replica inside the *NVLink scale-up
domain* of its master (`UltraEP/docs/blog_v1.md:53`). On Perlmutter that domain is 4
GPUs (one node): each node solves *its own* balance problem, and **cross-node imbalance
is untouched by design** (`ultraep_semantics.py:34-38`).

**The floor.** If node A's four ranks are collectively hotter than average, no
intra-node shuffling helps; the best any per-domain balancer can reach is

```
nvl_domain_lower_bound = max over domains(domain mean rank load) / global mean rank load
```

(`ultraep_semantics.py:180-190`). **Real instance:** layer 92 has `lb_floor = 1.183` —
one node's pools are simply hotter — and UltraEP lands at imbalance 1.209, essentially on
its floor (from 1.852). The `ultraep_domain16` counterfactual arm treats all 16 ranks as
one domain (what a rack-scale NVL72-class node would allow): floor 1.0, achieved 1.029 —
but its weight_sync then crosses Slingshot, which is priced honestly in
[`ep_semantics_vs_flux.md`](ep_semantics_vs_flux.md) §3 (it is *not* a faithful
Perlmutter deployment; `sweeps/variants.py:98-102`). This pair of arms is the paper's
rack-scale-node thesis (arXiv:2606.04101, title) turned into a measurable A/B.

The port asserts confinement structurally: layout construction fails on any
`"replica escaped NVL domain"` (`ultraep_semantics.py:821`).

### 5.5 Reroute: locality-aware quota split, and the interleave

**Plain language.** The solver fixed *instance quotas globally*; each source rank must
now decide, for each of its own tokens, *which instance* gets it — identically on every
rank, with no negotiation. And here is where the traffic side effect is actively helped
along: **prefer the instance on the token's own rank**.

**Toy.** `e2` has instances (master r1 q=6, replica r0 q=5, replica r2 q=2); per-source
`e2` loads `[3, 4, 3, 3]`. The locality-aware split
(`_rank_quota_alloc_for_expert`, `ultraep_semantics.py:441-484`) first lets every
instance absorb tokens from its *own host rank*: r0's 3 tokens → its local replica; r1's
4 → its master; r2's 2 (of 3) → its local replica. Only the residue (r3's 3 tokens plus
r2's 1 leftover) is apportioned over the remaining remote capacity by **largest-remainder
rounding** (ties → lower host rank, then lower slot). Compare
`remote_token_fraction` with locality on/off — the cell facts on the real trace read
0.914 vs 0.932: locality claws back ~2% of wire rows *for free*, from the same placement.

One honest subtlety the port surfaces (`ultraep_semantics.py:141-144`): each source
rounds independently, so realized per-instance rows can deviate from the solved quota by
±(sources−1) — in the toy, the master ends at 5 rows against quota 6. `gemm_rows_per_rank`
is therefore computed from the *realized* allocation, never from `quota`.

**Determinism without coordination.** Within one (source, expert), token `j` of `n` is
mapped to an instance via the **rank-quota prefix** (a per-source cumulative table,
`build_rank_quota_prefix`, `:487-515`) — instance = `searchsorted(prefix, position)`.
Position is not `j` itself but a **coprime-stride interleave**
`(j·stride + expert) mod n` with `stride = n//2 + 1` bumped to the next value coprime
with `n` (`:597-608`, porting `reroute.cu:189-209`): consecutive tokens fan out across
instances instead of marching through them in blocks, which decongests the subsequent
dispatch (knob `ULTRA_EP_QUOTA_REROUTE_INTERLEAVE`, `UltraEP/README.md:203`). In the toy,
r3's three `e2` tokens land replica-r0, replica-r0, master — spread, not clumped.

**No dedup — a real semantic contrast.** One wire row per (token, physical expert)
entry, faithful to UltraEP/Megatron dispatch (`ultraep_semantics.py:722-725`). Toy token
`t2` costs 2 rows to rank 3 where MoonEP paid 1. On the real trace the delta is small but
nonzero: `ultraep_dup_rows = 1347` rows (~1% of 131k) that MoonEP-style dedup would have
saved — quantified, per the harness rule that semantic differences become cell facts
rather than silent noise.

### 5.6 Weight sync: direct copies, and the relay idea for hot multicast

The chosen placement is materialized by copying each replicated master's weights (fc1
*and* fc2 — the full expert) into the replica slots, master → replica, always
intra-domain. At NVLink-domain ≤ 8 UltraEP forces the **direct** plan — every replica
pulls straight from its master (`ultraep_semantics.py:39-41`; port:
`weight_sync`, `:939-965`). The interesting engineering appears at rack scale, where one
scorching expert may need many replicas: the source rank's outbound multicast becomes the
new hotspot, so UltraEP builds a **chunk streaming relay** — a two-stage relay tree,
chosen at runtime from the traffic distribution, in which low-traffic ranks forward
chunks (groups of tiles) onward while the source streams ahead, no global barriers
(`UltraEP/docs/blog_v1.md:125`; eligibility: ≥4 replicas of one expert, knobs at
`UltraEP/README.md:190-193`). On Perlmutter D=4 the relay can never trigger (3 peers <
4-replica minimum) — a semantics note the port records explicitly
(`ultraep_semantics.py:39-41`).

The combine side needs no UltraEP changes at all: reroute is invisible to the token's
home rank — outputs come back from wherever the instance ran, and the weighted reduce
uses the same per-entry probs that rode along with dispatch (the `reroute` returns
adjusted `probs, routing_map` and the framework's own dispatch/combine run unchanged;
`UltraEP/README.md:139-148`).

### 5.7 Memory consumption (inference)

| Item | Cost | Notes |
|---|---|---|
| Redundant weight slots | `R_red` experts per rank, **reused across all layers** | the headline number: one slot on Qwen3-235B costs 108 MB with cross-layer reuse vs 9.9 GB without (`UltraEP/docs/blog_v1.md:72`); no optimizer state for replicas — that stays with masters |
| Token buffers (port) | sized to the *actual* receive count | no `NvS`-style static over-allocation, no padding — but the size is skewed exactly as the residual imbalance is (`ultraep_semantics.py:872-877`) |
| Plan tensors | `rank_quota_prefix [R, G, R]` dominates | 16·128·16·4 B = 128 KiB at capsule shape — trivial on-device, but building it host-side is R·G Python loops: `ultraep_plan_host_ms` ≈ 86 ms (D=4) / 142 ms (D=16), which is why it lives under the untimed-metadata contract while the *real* kernel solves in ~0.1 ms (§5.3) |
| Plan wire | `[R, G]` int32 all-gather | 8192 B at capsule shape (`ultraep_plan_comm_bytes`) — vs MoonEP's full `[S, K]` topk all-gather; UltraEP's planner needs only *counts*, MoonEP's dst assignment needs *per-entry order* (§4.4) |

No dynamic allocation at runtime is a stated design principle
(`UltraEP/docs/blog_v1.md:64`), same as MoonEP — both libraries treat allocator behavior
as part of the performance contract, not an afterthought.

---

## 6. The flux ports: same plans, staged transport, six timed phases

### 6.1 Why these are semantic ports, and what "bit-identical" buys

Neither library's kernels run here: MoonEP's are Hopper/NVLink-domain-only (TMA bulk
copies, NVSwitch multicast — `moonep_semantics.py:11-12`), UltraEP's are SM90+/NVSHMEM
(`ultraep_semantics.py:10-11`); Perlmutter is sm80 A100 with 4-GPU NVLink islands on
Slingshot. So the ports re-implement the **algorithms**, and pin their meaning with
fidelity contracts you can re-run:

- MoonEP: plans bit-identical (`torch.equal` per tensor, dedup structures as group sets)
  to the vendored oracle — which is what MoonEP's own production kernel is tested
  against — over MoonEP's 18 planning cases, R=16 Perlmutter-shaped cases, and lognormal
  fuzz (`test_moonep_planner.py`; CPU-only, runs on a login node).
- UltraEP: all solver tensors and the reroute expansion bit-identical to the **real
  kernels** via the SM80 oracle build, plus 16 frozen golden cases for machines without
  it (`test_ultraep_planner.py`; goldens double-run for determinism before freezing,
  `ultraep_oracle/dump_goldens.py`).

A plan-hash all-gather asserts every rank computed the identical plan before any timed
iteration (both drivers; e.g. `test_moe_moonep_traffic.py:372-375`).

### 6.2 One structural substitution, applied twice

Both upstream data planes assume a shared address space (one-sided writes into the final
slot). Across Slingshot nodes there is none, so both ports stage identically
(`moonep_semantics.py:363-373`; `ultraep_semantics.py:713-725`):

```
pack rows sorted by destination → all_to_all_single → destination-local placement
scatter to plan-decided slots → (zero-fill / dup-expand | nothing) → per-segment GEMM
```

The plan being replicated is what makes this cheap: the receiver derives every placement
index locally, zero extra handshakes. Wire rows and bytes equal the upstream semantics
exactly; the two port-added local copies are measured as their own phases so the wire
number stays pure. Replicated planning also substitutes for MoonEP's rank-0-plans-then-
hardware-multicasts scheme — every rank all-gathers the routing and runs the
deterministic integer planner itself (`moonep_semantics.py:20-25`).

### 6.3 The six phases, and the phase-name aliasing trick

Both drivers emit the *same six phase names* so arms compare inside one capsule, with
per-arm meanings documented in `sweeps/SCHEMA.md:105-176`:

| Phase | MoonEP arm | UltraEP arm |
|---|---|---|
| `plan_comm` | `[S,K]` topk all-gather (replicated-planning wire) | `[R,G]` loads all-gather (~8 KiB) |
| `pack` | dest-sorted representative-row gather (port-added copy) | (phys,token)-sorted rerouted-row gather |
| `comm` | a2av of dedup'd rows + per-entry fp32 weights | a2av of all rows + probs — **no dedup** |
| `scatter` | placement + zero-fill + duplicate expansion | placement only |
| `prefetch` | weight prefetch, home → prefetch slots | **aliased**: `weight_sync`, master → replica slots |
| `gemm` | per-segment over `cu_seqlens[E+B]`, padded rows computed | per-segment, unpadded — residual imbalance is the measurement |

The planner itself runs once per cell under the **untimed-metadata contract** (routing is
static per cell, so timing per-iteration host planning would measure redundant Python,
not the algorithm — the recurring per-layer *wire* cost `plan_comm` is still timed every
iteration; `test_moe_moonep_traffic.py:23-29`). The real planners cost ~0.1 ms on-GPU
(§5.3, `MoonEP/README.md:8`); the ports' host reimplementations cost 72–142 ms and are
reported as `*_plan_host_ms` cell facts, never entered into latency.

### 6.4 The disclosed deviations, and which way they bias

From the handoff ledger (`docs/handoff/02_algorithm_state_and_next_moves.md:346-353`),
every deviation is disclosed and biased **against** the ports: two-sided staged a2av plus
two local copies instead of one-sided direct-into-slot writes; replicated planning pays a
visible `plan_comm`; prefetch serialized in the base arm (MoonEP overlaps it — the
`moonep_overlap` arm restores this and is the honest best configuration); the
port moves 2 weight matrices (w1+w2, both in one prefetch phase under one
join) where MoonEP moves 3 — the port models an ungated FFN, so gate/up
collapse to one matrix (narrowed 2026-08-17 from the original 1-of-3 when the
`--layers l01` journey landed; layer0-only runs still move just w1). The M4 arms then close the
transport-fidelity gap one axis at a time — `moonep_nvshmem` swaps in flux's one-sided
NVSHMEM `All2AllSingle` (sender-driven, receiver-passive, `moonep_semantics.py:586-618`;
`putmem_nbi` + two team barriers per call — put-then-barrier like MoonEP's real dispatch;
the implementation's `putmem_signal` kernel is dead code — corrected 2026-08-11, NR-12)
and produced bitwise-identical results on first bring-up, proving the semantics are
transport-invariant. The measured story of all four arms, and the
`CUDA_DEVICE_MAX_CONNECTIONS` lesson learned on the way, live in
[`ep_semantics_vs_flux.md`](ep_semantics_vs_flux.md) §3.

---

## 7. Side by side

| Axis | MoonEP | UltraEP |
|---|---|---|
| Mechanism | migrate hot experts' *tokens* to under-loaded ranks (`z`, `alloc`) | *replicate* hot experts into redundant slots, split load by quota |
| Hot-rank detection | `balance = group_tokens − S*K`; `argmax` overloaded / `argmin` roomiest, receivers filled fully | threshold-τ feasibility search; oracle visits source ranks by descending load, target = max remaining slack |
| Hot-expert selection | most-remaining-tokens within the hot home group | descending expert load within each overloaded rank |
| Balance achieved | exact: every rank `S*K` rows (+padding) | near-optimal down to `nvl_domain_lower_bound`; solver-quantized |
| Scope of the fix | global across the EP group | one independent solve per NVLink domain (cross-node skew untouched) |
| Wire semantics | dedup: one row per (token, dest rank); `−raw−1` encoding keeps per-entry weights | no dedup: one row per (token, physical expert); `ultraep_dup_rows` audits the delta |
| GEMM layout | static `[NvS,H]`, `token_padding`-aligned, padded rows computed | unpadded per-physical-expert segments, sized to realized load |
| Plan wire | `[S,K]` topk all-gather (needs per-entry order) | `[R,G]` loads all-gather, ~8 KiB (needs only counts) |
| Weight movement | prefetch B hottest remote segments (B=3–4 inference, symmetric-memory overflow) | weight_sync master→replica, intra-domain; chunk-streaming relay at rack scale |
| Solver style | two greedy loops, O(R) rounds | fast-eps probe → capacity bisection → precheck → 4-probe parallel bisection ladder |
| Memory posture | `[E+B,H,H']` contiguous weights + ~25% row padding, process-global prefetch pool | `R_red` cross-layer-reused replica slots (9.9 GB → 108 MB on Qwen3-235B), no token over-allocation |
| Fidelity anchor here | bit-identical to vendored executable oracle | bit-identical to real kernels (SM80 oracle build) + frozen goldens |

The deepest similarity is easy to miss under the table: **both planners are pure
deterministic integer math on replicated inputs** — identical plans on every rank with no
broadcast, no floats (UltraEP: exactly two float32 `ceilf`s, faithfully reproduced —
`ultraep_semantics.py:22-24`). That single property is what lets a plan *be* the
communication protocol: every sender and receiver derives every index locally.

---

## 8. TL;DR mapping

| Concept | Where it lives |
|---|---|
| MoonEP balance vector / z / alloc greedy | `python/flux/testing/moonep_semantics.py:158-200`; oracle `moonep_oracle/planning_reference.py:61-131` |
| MoonEP VM layout, `cu_seqlens`, padding, prefetch pick | `moonep_semantics.py:202-259` |
| MoonEP `dst` encoding + dedup `−raw−1` | `moonep_semantics.py:261-341`; rationale `planning_reference.py:250-255` |
| MoonEP staged transport + phases | `moonep_semantics.py:360-720` (`MoonEPLayer0Runner`) |
| MoonEP contract, B policy, weight buffer | `MoonEP/README.md:7-9, 43-59` |
| UltraEP formalization (Λ, U, Q, τ, exc/slk) | arXiv:2606.04101 §4.3 |
| UltraEP solver ladder + export oracle | `python/flux/testing/ultraep_semantics.py:198-408`; arXiv:2606.04101 §5.1, Alg. 1 |
| UltraEP locality quota split + interleave | `ultraep_semantics.py:411-515, 597-656`; arXiv:2606.04101 §5.2 |
| UltraEP domain floor / rack-scale counterfactual | `ultraep_semantics.py:180-190`; `sweeps/variants.py:98-109` |
| UltraEP weight_sync + relay | `ultraep_semantics.py:939-965`; `UltraEP/docs/blog_v1.md:119-132` |
| Phase/metric contracts for both arms | `sweeps/SCHEMA.md:105-176` |
| Deviations ledger + M4 history | `docs/handoff/02_algorithm_state_and_next_moves.md:321-392` |
| Measured capsules | `sweeps/results/runs/20260808-{011748,015920,032217,090536}_perlmutter_*` |

The measured comparison against the flux/Comet overlap arms — including where the toy
intuition survives contact with real hardware and where it does not — continues in
[`ep_semantics_vs_flux.md`](ep_semantics_vs_flux.md).
