# Routing — toy walkthrough of the four steps + determinism (2026-09-05)

> NOTE 2026-09-05: `SUMMARY.md` supersedes the placement table below (it re-derives the toy from an oracle histogram; the routing toy there uses the solver's actual output). The step rationales here remain valid.

Companion to `01_decision_logic.md` §B. Source of truth: the kernel
contract comment in `src/cuda/moe_utils.cu` (tiers 1/2/3/forced, relaxed
tickets) and `python/flux/testing/ours.py` (per-iteration plan lane).

## Setup

3 nodes x 2 GPUs, 8 experts A..H, 2 slots per GPU (12 slots = 8 homes +
4 copies). Placement (already decided at setup):

| node | GPU | hosts |
|---|---|---|
| N0 | G0 | A, B |
| N0 | G1 | C, D |
| N1 | G2 | E, A' |
| N1 | G3 | F, B' |
| N2 | G4 | G, E' |
| N2 | G5 | H, A'' |

Batch: 4 tokens per GPU, top-2, so 8 rows per GPU. Cap per GPU =
ceil((1 + 1/16) * 8) = 9 rows. We follow source GPU G0 (home node N0):

| token | experts |
|---|---|
| t0 | A, B |
| t1 | A, C |
| t2 | E, G |
| t3 | D, H |

Before any decision every GPU all-gathers its demand histogram (how many
rows it wants of each expert) — this is `plan_comm`. From that one
table every GPU computes, identically, the quotas and shares used below.

## Step 1 — own GPU

G0 hosts A and B. Its own demand for them: A x2 (t0, t1), B x1 (t0) =
3 rows. 3 <= 9, so all three stay on G0. (If G0's self-hosted demand
were 12 rows, it would keep 9, scaled down expert-by-expert by largest
remainder, and the other 3 would fall through to the next tiers.)
G0 load: 3.

## Step 2 — own node

Remaining demand for experts hosted elsewhere in N0: C (t1) and D (t3),
both on G1. Suppose G1's own step-1 load is 5 (its tokens wanted C x3,
D x2), so G1's residual capacity is 9 - 5 = 4. The node's outside demand
on G1 is 2 rows (G0's C and D); 2 <= 4, both granted. G0 sends C and D
to G1 over NVLink. Zero wire bytes so far. G1 load: 7.

Had N0 wanted 6 rows of C from G1 (from G0 and G1 together beyond G1's
own), the 4 residual rows would be split across the node's hosting
GPUs in proportion to their residual capacity (only G1 here) and the
sources served in GPU order; the unserved rows fall to step 3.

## Step 3 — remote nodes: fewest nodes per token

Left: t2 (E, G) and t3's H. E is hosted on N1 (G2) and N2 (G4); G only
on N2 (G4); H only on N2 (G5).

Shares first. Each remote GPU's residual capacity is pre-split among
source GPUs in proportion to how much leftover demand each source has
for experts hosted there:

- G4 (hosts G, E'): residual 4. Leftover demand for G/E: G0 wants 2,
  G2 wants 1, G3 wants 1. Shares 2 : 1 : 1 -> G0 gets 2 rows on G4.
- G5 (hosts H, A''): residual 1. Leftover demand: G0 wants 1, G2 wants
  4, G3 wants 3. One row split 1 : 4 : 3 by largest remainder -> G2
  gets it. **G0's share on G5 is 0.**

Now the per-token cover, for G0's tokens:

- t2 (E, G): which single remote node covers the most of {E, G} using
  GPUs where G0 still holds share? N1 covers E (via G2). N2 covers E
  and G (both on G4, share 2). N2 wins. Both rows go to G4; G0 spends
  its 2 tickets there. **One remote node touched.** An even split of E
  across its two copies would have sent half of E to N1 and touched two
  nodes for this token — that is exactly what the objective forbids.
- t3's H: hosted only on G5, where G0 holds no share, so no node can
  cover it. After 3 rounds it is still unassigned.

## Step 4 — forced

H goes to the least-loaded GPU that hosts it — G5, the only host — and
is counted as a forced row. G5 ends over its cap. The cap is soft only
for this residue; it exists so routing is total (every row has a
destination) even when the shares run out.

Result for G0's 8 rows: 3 stayed home, 2 crossed NVLink, 2 went to one
remote node, 1 was forced to another. Every other GPU did the same for
its own rows at the same time, with no coordination.

## Is the routing deterministic?

Two layers, and the answer differs:

- **The tables are deterministic.** Own-GPU quotas, node grants and
  remote shares are pure integer functions of the all-gathered
  histogram. Every GPU computes bit-identical tables; the offline torch
  reference and the CUDA kernel must agree on them exactly (unit-gated).
- **The per-row choices are not.** Inside the kernel, rows claim
  tickets from the quotas and shares with relaxed atomics (user ruling
  2026-08-21: no bit-determinism required). Which of two equal rows
  gets the last ticket on a GPU can differ run to run. Every choice
  stays within the tables, so per-destination row counts are bounded
  the same way every run, but the exact row-to-replica map is not
  reproducible and cannot be recomputed by another GPU.

That is why agreement across GPUs does not come from replaying each
other's decisions. Each GPU routes only its own rows, then **one fused
all-gather of (chosen physical slot | gate prob) per row** publishes the
decisions; every GPU builds its send/receive splits from that. So the
plotted arm pays two all-gathers per iteration: the demand histogram
before routing (`plan_comm`) and the decisions after routing (`plan`).
The alternative `--route_global 1` (one all-gather of top-k + probs,
every GPU deterministically recomputes every GPU's routing with a quota
rule) exists but is not the plotted arm.

## Step 3 rationale — why "which remote node" is decided the way it is

**The cost fact.** The dispatch transport deduplicates per node: when a
token needs experts on a remote node, its hidden row crosses the wire
ONCE to that node, no matter how many of its experts live there (the
node fans it out over NVLink). So a token's wire cost is the number of
distinct remote nodes it touches — not the number of remote experts,
not the number of rows. Two experts on one node cost the same as one.

**Therefore the decision is per token, not per expert.** Choosing each
expert's "best" copy independently can touch three nodes for a token
whose experts could all have been served by one. The right question is:
what is the smallest set of remote nodes that, between them, host every
expert this token still needs? That is a set cover; the kernel solves
it greedily per token.

**The greedy.** For one token with a set of still-unserved experts:
1. Score every node by how many of those experts it hosts on a GPU
   where this source still holds tickets.
2. Take the highest-scoring node (ties: home node first, then lower
   node id — a fixed rule so ties never need coordination).
3. Send ALL of that token's experts hosted there to that node; inside
   the node each expert goes to its hosting GPU (if the expert has two
   hosting GPUs in the node, the one where the source has more
   tickets).
4. Remove those experts, repeat for what is left (up to 3 rounds).

Example: a token still needs X, Y, Z. X is on N1 and N2, Y on N2, Z on
N1 and N3. Scores: N1 = 2 (X, Z), N2 = 2 (X, Y), N3 = 1. Tie -> N1:
X and Z go to N1. Round 2: Y -> N2. Two nodes. Per-expert "nearest
copy" could have chosen N2 for X, N2 for Y, N3 for Z — also two nodes
here, but with X on N1 only it would touch three; the cover never does
worse than the per-expert choice and often better.

**Why tickets, and why proportional.** Locality alone would pile every
token onto the node holding the most copies, blowing its compute cap.
The global algorithm fixed that with live capacity feedback, which
needs coordination every row. The sender-local version instead splits
each remote GPU's residual capacity up front into per-source SHARES
(tickets) that every GPU can compute from the histogram alone. Shares
are proportional to each source's leftover demand for that GPU's
experts, so a source that needs a GPU more gets more of it; integer
rounding is by largest remainder so the shares sum exactly to the
residual.

Tickets are what make the cover respect balance without any exchange:
a node only counts toward a token's cover if the source still holds
tickets on a hosting GPU there. As a source spends its tickets on a
popular GPU, that node drops out of its covers and the greedy naturally
shifts later tokens to the next-best node. What no ticket can absorb
is forced (step 4). The accepted price of going sender-local: one
source's unused tickets cannot be lent to another (no donation).

## Step 4 rationale — forced rows

**Who ends up here.** A row whose expert no node can cover for this
source: every hosting GPU of that expert is one where the source's
tickets are spent, or where it was handed zero tickets in the first
place (a hot GPU whose small residual went to sources with more
demand). After the three cover rounds these rows are still unassigned.
In the toy, G0's H: hosted only on G5, share 0.

**Where they go.** Every GPU computes, once per iteration and
identically, a per-expert fallback: the hosting GPU with the least load
AFTER tiers 1-2 (a frozen table — keying it on live tier-3 load made the
kernel and the torch reference disagree and broke the receive-buffer
sizing, handoff 17). The row is sent to the fallback. In the toy,
fallback[H] = G5, its only host.

**Budget.** Forced traffic is bounded per (source, destination) pair by
`f_cap` rows, derived at setup from the reference routing tables
(`loccap_sl_bounds`) so that receive buffers are provably large enough.
A forced row takes a forced-ticket on the fallback; if the fallback's
budget for this source is spent, it tries the expert's other hosting
GPUs in ascending id, each with its own budget; if all are spent it is
sent to the fallback anyway and a loud overflow counter is bumped. In
the plotted s1 arm that counter must stay zero (sizing-contract
assert); in the s2 arms the planner re-routes with 4x, then unbounded,
budget (handoff 22b addendum).

**Why it exists.** Routing must be total — every row needs a
destination before the all-gather publishes decisions. Caps are hard
for tiers 1-3 by construction of the quotas and shares; they are soft
only for this residue, which is the compute-imbalance the system
accepts rather than negotiate. The residue is a consequence of "no
donation": a source cannot borrow another source's unused tickets, so
a hot expert with few hosts can strand demand even when global
capacity remains.

**Cost of the forced path.** None beyond the row's normal wire cost:
the fallback table is one tiny kernel over G experts, and admission is
one atomic per forced row. The forced rows themselves are the ones
that break the compute cap and, being routed by load rather than
locality, may add a remote node to their token's cover.

## Was the router's cost ever measured precisely?

Yes, at three levels; none is a full topology x budget kernel-only
bracket, so treat the first two as anchors and the third as the
quotable per-cell number.

1. **Kernel alone** (handoff 13, 8/22): the tier-3 cover kernel
   `pll_route3` after the register-mask rewrite — R=128 (32 nodes)
   3.37 -> 1.65 ms offline; "4n parity". Handoff 12 quotes the whole
   sender-local kernel at ~0.4 ms at 4n, replacing an 18.5 ms torch
   router.
2. **Inside the plan lane** (handoff 26 §1(b), 8/29, 4n b1, host NVTX
   medians, model-independent): route kernel 0.11 ms, phys+probs
   all-gather 0.11, vce tail 0.21, derive_routed_meta 0.26, scale build
   0.23, derive_combine_meta 0.64. Routing is ~7% of the plan lane; the
   lane is ~70% dispatch/combine metadata derivation. COMET's whole
   plan is 0.26 ms.
3. **Per plotted main-perf cell** (`plan_bracket_main_perf.txt`, from
   the winner capsules): `plan` = route kernel + decisions all-gather +
   vce + derive, i.e. the router's upper bound. 0.62-0.93 ms at 4n,
   0.79-1.24 at 8n, 1.17-3.40 at 16n; 5-24% of total, largest share at
   1 MiB and at 16n. The budget dependence (b1 -> b16 adds 0.2-1.8 ms)
   is the derive/all-gather side growing with rows; the cover kernel is
   thread-per-entry with a per-token loop over NN nodes, so its own
   growth is with rows x nodes.

Not measured anywhere: the route kernel isolated at 8n/16n on the
plotted binary, or its split between tiers. If the figure needs a
sentence like "the cover costs X ms at 16 nodes", it needs one nsys
capture of a 16n cell (kernel name `pll_route3_kernel_t`).
