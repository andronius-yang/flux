# Handoff 40 — The capacity-constrained router (pv3c), explained from scratch

2026-09-15. Written for the paper authors. Everything here is traced from
`python/flux/testing/pv3_route.py` (reference + checks), `_pv3_ext.cu`
(kernels) and the driver hooks; handoff 38 has the audits and proofs,
handoff 39 the recapture campaign and its datasets.

## 0. One-paragraph summary

The router decides, for every (token, expert) pair, which *copy* of that
expert does the work. The postdoc's formulation says: each copy must
receive about its fair share of that expert's traffic, within a slack C,
and among all such assignments prefer the one that moves the fewest bytes.
The old router did not implement this; it enforced one per-GPU cap that
ignored which experts the GPU hosts, and silently broke it when the
placement made it impossible ("forced" rows). pv3c implements the
formulation exactly: a rotation "water-fill" that fills copies nearest
first up to the band ceiling while reserving enough for every copy's
floor, followed by a "vacate" pass that spends the leftover slack to make
each token's remote entries land on as few nodes as possible (which is
what the transport actually bills). It is provably feasible, needs no
forced regime, and is on par with the old router through 8 nodes; at
16–32 nodes it trails by a few percent on large budgets (wire) and on
small budgets only through a plan-bracket overhead that is being fixed.

## 1. The formulation, in plain words

Every batch, each GPU has S tokens and each token has picked its top-k
experts. Some experts have been copied onto several GPUs ("replicas" or
"copies"). Notation: expert e has demand D_e this batch (how many
(token, e) entries exist) and c_e copies. The rules:

1. **Rows are fixed.** A token must go to the experts it picked. Routing
   never changes the model's decision, only which copy does the work.
2. **Per-copy fair share, ± C.** Copy j of expert e is supposed to receive
   q = D_e / c_e entries. It may receive anything in
   [(1 − C)·q, (1 + C)·q]. A GPU with no copy of e has q = 0, so it can
   never receive e's entries.
3. **Per-GPU total, ± C.** A GPU's total load stays within ±C of the sum
   of the fair shares of the copies it hosts.
4. **Whole tokens.**

Objective: among assignments that satisfy 1–4, minimise transport cost
(cheapest: token stays on its own GPU; next: another GPU in the same node
over NVLink; most expensive: another node over the network).

Two facts about the rule set that the paper text should absorb:

- **Rule 3 follows from rule 2** with the same C: a sum of quantities that
  are each within ±C of their reference is within ±C of the sum of the
  references. It is not wrong, just implied. (The audit confirms it: no
  route ever violated 3 without violating 2.)
- **Whole tokens need floor/ceil wording.** "Within ±C·q" is impossible
  when q is small: q = 0.5 and C = 1/4 allow 0.4..0.6, which no integer
  hits. The implementable form is
  floor((1 − C)·q) ≤ load ≤ ceil((1 + C)·q). With that reading a valid
  assignment always exists (the rounded equal split).

## 2. Why this formulation works — the intuition

**What the old router got wrong.** It had one number per GPU: "no GPU
takes more than (1 + eps) × its equal share of the batch", whatever it
hosts. That sounds like balance, but it fights the placement. Example: the
placement put a hot expert alone on GPU 3, so GPU 3 *must* take all of that
expert's entries; the uniform cap is impossible there and the code gave up
on it ("forced" rows). Meanwhile a GPU hosting only cold experts was
allowed a load it could never receive. The measured result (handoff 38 §2)
on the plotted cells: replica loads between 0.33× and 2.07× of fair share
at 16 nodes, i.e. the stated constraint was never held.

**What the new one does.** It moves the reference from "equal share of the
batch" to "fair share of what this copy exists for". Think of each expert
as a job with D_e units of work and c_e workers; every worker should get
about D_e / c_e units, ±C. That is always achievable (split evenly), it
matches what the placement intended (the placement chose c_e so that
D_e / c_e is a sensible load), and it still leaves freedom: the ±C band is
exactly the room the router has to prefer nearby copies. Balance is
guaranteed by construction; locality is what the router optimises inside
the band.

**Worked example.** Expert e has 300 entries this batch and copies on GPU
A (node 1) and GPU B (node 2); q = 150, C = 1/4, so each copy may take
113..188. Node 1 holds 220 of the 300 entries. The router lets A take up
to 188 of them (local, cheap); the remaining 32 plus the 80 from node 2
go to B (112 — one below the floor, so the ceiling on A is reduced by one
during the fill; see the reserve below). Without the band A would take
all 220 (best locality, worst balance); with the old uniform cap A might
have been forbidden from taking even its fair 150 if it also hosted other
busy experts.

## 3. Local phase, then remote phase — and why the remote phase needs the band

Locality is a strict hierarchy: own GPU costs nothing, own node costs an
NVLink hop, another node costs the network. So every router fills in that
order: first each source keeps what it can on its own copies, then on its
node's copies, and only the overflow goes to remote nodes.

In the **old router** the remote phase (its "tier 3") had to spread the
overflow under the per-GPU cap, and because that cap did not know which
experts a GPU hosts, the phase regularly found no feasible target and fell
into the forced regime. The remote phase *depended* on the imbalance cap
in the sense that the cap was the only thing stopping a hot node from
dumping everything on one remote GPU — and it was the wrong cap.

In **pv3c** the band plays that role, and it makes the local phase safe
and the remote phase both necessary and well defined:

- **Ceiling (1 + C)·q stops hoarding.** Without it, the local phase would
  let a node keep its hot experts entirely on its own copies, starving the
  remote copies and overloading local GPUs.
- **Floor (1 − C)·q forces the local phase to leave enough behind.** A
  remote copy must not end up idle while its siblings run at 1 + C. So a
  source may take from a copy only what remains after subtracting every
  *other* copy's outstanding deficit to its floor. This is the "reserve"
  term in the code: take = min(what I still have, room under the ceiling,
  unassigned − deficit of the other copies).
- **The overflow is exactly what must cross nodes under the band.** The
  remote phase places it on remote copies that still have room, nearest
  first. A short proof (handoff 38 §3) shows it always finishes: the
  reserve guarantees the floors, the ceilings guarantee nobody over-fills,
  and because total capacity Σ ceil((1 + C)·q) ≥ D_e nothing is ever
  stranded. There is no forced regime any more.

**The covering refinement (the "c" in pv3c).** The fused transport
de-duplicates per node: a token whose remote entries all land on the same
remote node is shipped once. So after the per-copy counts are fixed, it
pays to make each token's remote entries *cluster* on as few nodes as
possible. The band gives the freedom for that too: within the slack, a
token's entry can move from the copy on node X to the copy on node Y if X
has released room (it is above its floor) and Y has spare room (below its
ceiling). That is what the vacate pass does, and it is the only place the
cap enters the covering step: every move is paid from the ±C slack on both
sides, so the bounds cannot be violated by covering, no matter how many
moves it makes. Worked example: token t has entries on nodes X (copy of
e1) and Y (copies of e2, e3). If Y also hosts a copy of e1 with spare
room, and X's copy of e1 is above its floor, move t's e1 entry to Y: t now
touches one remote node instead of two, both copies stay inside their
bands, and the two tickets that paid for the move go back to the pool so
another token can use the room X just released.

## 4. How it is implemented, end to end

Inputs each iteration (exchanged in the plan-comm bracket, as before): the
demand histogram d[rank, expert] (how many entries each GPU has for each
expert) and the placement tables (which physical slot holds which copy).
Everything below runs on every rank identically and produces identical
tables, so no rank has to trust another.

### 4.1 Tables pass — one GPU thread per expert (`pv3_tables_kernel`)

For its expert the thread computes D_e, c_e, the ceiling
U = ceil((1 + C)·D_e / c_e) and the floor Lb = floor((1 − C)·D_e / c_e),
then plays R "rounds" (R = number of GPUs). In round p every source GPU i
visits exactly one GPU,

    target(i, p) = node (u_i + p div L) mod NN, rank (l_i + p) mod L

which walks: itself (p = 0), the other GPUs of its node (p = 1..L−1), then
the other nodes in rotation order. For a fixed p this map is a
permutation, so each copy is visited by exactly one source per round and
each (source, copy) pair exactly once overall — that is why one thread can
do the whole expert sequentially with no atomics. At a visit the source
takes

    min( its remaining entries for e,
         U − fill of that copy,
         unassigned_e − deficit of the other copies )

(the third term is the reserve of §3). The takes for the thread's own rank
are kept as a short segment list: copy A gets n_A of my entries, then copy
B gets n_B, in visit order.

### 4.2 Materialisation — one thread per (token, expert) entry (`pv3_route_kernel`)

Each entry draws a ticket for its expert (an atomic counter) and the
ticket number selects the segment: tickets 0..n_A−1 go to copy A, and so
on. Counts per (source, copy) are therefore exactly the table; only which
particular token got which slot is left to the hardware — the same
"relaxed ticket" contract the old router used and the transport already
tolerates.

### 4.3 Slack budgets — one block per (expert, copy) (`pv3c_budget_kernel`)

For the vacate pass each source gets a slice of two per-copy slacks:
release = my share of (fill − Lb) at that copy (entries I may move away)
and extra = my share of (U − fill) (entries I may move in). Shares are a
deterministic largest-remainder split weighted by each source's own rows
at that copy, computed by threads striding over sources with block
reductions. Because shares are computed from the tables, every rank knows
every budget without another exchange.

### 4.4 Vacate pass — one thread per token (`pv3c_vacate_kernel`)

The thread lists the remote nodes its token touches. Starting with the
node it touches least, it tries to move *every* entry it has there onto a
node it already touches (home first). Each move needs one release ticket
at the old copy and one extra ticket at the new copy, taken with atomics;
if any entry of that node cannot move, all tickets of the attempt are
rolled back and the next node is tried. On success the freed tickets go
back to the pool. Each success removes one node from the token's remote
set, so the loop ends after at most NN steps. Bounds hold because every
move is paid from slack on both sides; remote rows never increase because
targets are only nodes the token already uses, or home.

### 4.5 Sizing and the driver

The transports pre-allocate receive regions, so the driver needs bounds it
can trust before the first iteration. For pv3 the realised per-pair counts
equal the tables, so the reference route *is* the sizing. For pv3c the
vacate pass can add at most the extra tickets a destination holds, so the
caps are provable: per GPU Σ ceil((1 + C)·q) over its copies, per
(source, destination) table + extra, and the receive-region cushion is the
extra tickets pointing at that destination. One hook serves every OURS
lane (fused, direct wire, s2 swaps, dual3) through the shared planner, and
the EPIC-driver llc row through its planner; both are gated by
per-iteration output validation against a routing-independent reference
plus a final deterministic iteration on the reference routing.

### 4.6 What C means

C is the only knob. C = 1/16 is already on par with the old router in
most cells but loses a few percent on the wire at large budgets; C = 1/4
(±25 % per copy) reaches parity across the 4-node grid on both models;
C = 1/2 is better at 16 and 32 nodes. The realised max GPU load under
C = 1/4 stays at or below what the old router actually produced, so the
wider *stated* bound is not a worse balance in practice — the old router
simply never held the bound it stated.

## 5. What the recapture measured (handoff 39 §9, same statistic as the figures)

Δ = pv3c vs the old router measured in the same capsule on the same
binary; C chosen per topology by a fixed rule (handoff 39 §8).

| topology | C | mean Δ | worst cell | verdict |
|---|---|---|---|---|
| 2n | 1/4 | −1.4 % | −0.1 % | on par / better |
| 4n (24 main-perf cells + weak) | 1/4 | −1.1 % | +4.4 % | on par |
| 8n | 1/4 | −0.8 % | +4.9 % | on par |
| 16n | 1/2 | +1.5 % | +7.9 % | close |
| 32n | 1/2 | +6.1 % | +9.5 % | trails |

Ablation (LOO + matched, 5 reps), cycling ablation (S-A 8 topics, S-C 3
reps) and the case-study arms are all within ±1.5 % of the old router.

Where the 16–32n residual comes from, and what it is not:

- **Large budgets (b16/b64): wire incidence.** pv3c's per-token node
  count is a few percent above the old router's at 16+ nodes because the
  cover is done inside the band; the old router had no band to respect.
  C = 1 does not close it (the combine gets worse). Structural.
- **Small budgets (b1/b4): the plan bracket, not the route.** At 32n the
  routed l0 and l1 phases are *lower* under pv3c; the loss is +0.7–1.2 ms
  in the plan bracket while the single-GPU kernel head-to-head predicted
  +0.15 ms. This is the driver-side launch path (five launches through
  the extension vs one fused op, amplified by max-over-128-ranks) and is
  the follow-up in progress.
- **Not a constraint violation anywhere:** every pv3c cell holds the band
  by construction (checked on every reference route; the kernel's drift
  is provably bounded and sized for).

## 6. The fast kernels (v4) and why they still guarantee the constraints

After the recapture, the router's two heavy kernels were rewritten for
speed (handoff 39 §11: 32n plan bracket 2.4 -> 1.4 ms, now cheaper than the
old router; v4.1 additionally shrinks the tables kernel's per-thread local
memory 2048 -> 480 B after a first-launch memory failure on the fullest GPU,
handoff 39 §12.1). Speed changes *how* the numbers are computed, never *which*
numbers, and the two halves are guaranteed by two different arguments.

**Half 1 — the tables (what each copy receives) are exact.** The band is
enforced entirely by the per-(source, copy) count table of §4.1, which is
built by one thread per expert. v4 keeps the same sequence of visits and
the same three-way minimum at every visit; it only (a) carries the
"deficit of the other copies" as a running sum instead of re-adding it
from scratch at every visit, and (b) precomputes each copy's node and
local rank so the visit loop does no integer division. Because the
arithmetic is the same, the result is the same *bit for bit*: the parity
test (`test_pv3_kernel.py`) compares every (source, expert, destination)
count of the kernel against the torch reference implementation and the
reference is the one the proof in handoff 38 §3 is written for
(ceilings: a take is never more than the room under the ceiling; floors:
the reserve keeps enough unassigned rows for every other copy's floor;
totality: Σ ceil((1 + C)·q) ≥ D_e, so the fill always completes). The
materialisation step (tickets → segments) is unchanged.

**Half 2 — the vacate pass cannot leave the band, in any order.** This is
the half that became parallel (one 8-lane group per token instead of one
thread), so the argument must not depend on the order in which entries
move. It does not. For each copy j keep two counters that the moves
consume and refill: release tickets rel_j and extra tickets ext_j,
initialised to fill_j − floor_j and ceiling_j − fill_j (both ≥ 0 after
Half 1; the per-source shares only partition them further). A move of one
entry from copy j to copy j' of the *same expert* does exactly four
things: take one rel_j (only if rel_j > 0, atomically), take one ext_j'
(only if ext_j' > 0, atomically), then, on commit, put one ticket back on
each side (ext_j += 1: the vacated slot is room now; rel_j' += 1: the
moved row may move again). If the group cannot complete all the moves it
needs for a node, every ticket it took is returned. Two invariants
therefore hold after every commit or rollback, whatever the interleaving
across tokens and lanes:

    fill_j − rel_j = floor_j        (departure: both −1; arrival: both +1)
    fill_j + ext_j = ceiling_j      (departure: fill −1, ext +1; arrival: fill +1, ext −1)

and since tickets are only ever taken when positive, rel_j ≥ 0 and
ext_j ≥ 0, hence floor_j ≤ fill_j ≤ ceiling_j at all times. Rows stay
fixed because a move only ever targets a copy of the entry's own expert
(the lookup table maps a node to *that expert's* copy there), so a GPU
without a copy can never receive the entry, and conservation (every entry
lands on a copy of the expert it picked) holds trivially. The cooperative
kernel changes only who takes the tickets and when; it cannot change what
the counters allow.

**What the audit checked, on the real inputs.** The reference tests run
on synthetic traffic; the closing audit ran the v4 kernels exactly as the
driver calls them — once per rank, for every rank — on the campaign's own
routing matrices and placements (4/8/16/32 nodes, K2 and Qwen, b1 and
b16/b64, C = 1/4 and 1/2) and scored the assembled routing with the same
checker used for the paper's constraints (integer form). Results are in
`39_pv3c_v4_constraint_audit.txt` next to this file and summarised here:

| check (56 routings: 14 cells × C ∈ {1/4, 1/2} × {pv3, pv3c}) | result |
|---|---|
| rule 1 / conservation: every entry lands on a copy of the expert it picked | all 56 pass |
| entries on GPUs that host no copy of the expert | 0 rows in all 56 |
| rule 2, integer form: copy load outside [⌊(1−C)q⌋, ⌈(1+C)q⌉] | 0 rows over, 0 rows under, in all 56 |
| rule 3, integer (floor/ceil) form: GPU load outside its band | 0 in all 56 |
| rule 3, un-rounded real form | 0 for every pv3c routing; 1–3 GPUs in three pv3-only routings (the integer-rounding case of §1) |
| realised copy load / fair share, pv3c | C = 1/4: [0.746, 1.254]; C = 1/2: [0.516, 1.493] (the bands are [0.75, 1.25] and [0.5, 1.5]; the 0.746 is a floor on a small q) |
| realised GPU load / Σ fair shares, pv3c | C = 1/4: [0.795, 1.165]; C = 1/2: [0.561, 1.320] |
| kernel's loud counters (ticket overflow, forced rows) | 0 in all 56 |
| network incidence, kernel vs the deterministic reference (12 b1 cells) | kernel +0.4 % … +1.3 % above the reference (relaxed ticket order), never below the pv3 tables |

For comparison, the old router's realised copy loads on the same plotted
cells ranged from 0.33× to 2.07× the fair share at 16 nodes (handoff 38
§2): the band is not a formality.

## 7. A paragraph for the paper (the routing problem and how it is solved)

> **Routing.** After the gating histogram d (rows per GPU per expert) is
> exchanged, every GPU computes the same routing tables. For each expert
> e with D_e rows and c_e replicas, each replica must receive between
> ⌊(1 − C)·D_e/c_e⌋ and ⌈(1 + C)·D_e/c_e⌉ rows — the balance band, our
> only constraint; rows never change expert. Tables are filled greedily
> by a rotation water-fill: in round p every source GPU visits one GPU
> (itself, then its node-mates over NVLink, then the other nodes in
> rotation order), assigning as many of its rows for e as the visited
> replica's remaining ceiling allows while reserving enough unassigned
> rows for every other replica's floor. This ordering realises the
> locality preference (own GPU, then node, then network) and the
> reserve guarantees every floor is reached; because the ceilings sum to
> at least D_e, the fill always completes and no exception path is
> needed. A second, order-independent pass then spends the remaining
> slack on network traffic: a token's remote rows are moved, one node at
> a time, onto nodes it already sends to, each move paying one release
> ticket at the source replica (available only above its floor) and one
> extra ticket at the target (available only below its ceiling), so
> every routing it produces is inside the band by construction and the
> number of distinct nodes each token reaches — the quantity the
> de-duplicating transport bills — can only decrease. Both passes are
> single GPU kernels (a thread per expert; an 8-lane group per token)
> that cost 0.25–0.5 ms at 16–32 nodes, less than the heuristic router
> they replace, and produce byte-identical tables on every rank without
> further communication.

## 8. Two questions to check the idea landed

1. Suppose the placement gives expert e three copies, and one of them sits
   on a node that sends *no* entries to e at all this batch. With C = 1/4,
   what is the minimum number of entries that must cross the network to
   that copy, and which term in the take rule (§4.1) is responsible for
   making the local phase leave them behind?
2. The vacate pass only moves an entry to a node the token *already*
   touches. If we allowed it to move an entry to a node the token does not
   yet touch whenever that node has spare room, would the per-copy band
   still hold, and would the transport bill go down or up? Explain why the
   two answers differ.
