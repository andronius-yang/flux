# Expert placement & routing — the complete decision walkthrough (2026-09-05)

Every decision the plotted arm (`ours_l01_s1_pv2_r2`) makes, in order,
each with its rule and its rationale. Facts marked [capsule] were read
from the winner cells' own records (4n K2 1 MiB, capsule
20260829-143523; 16n K2 4 MiB, capsule 20260830-180110); the command
line in those cells' srun.log carries `--eps 0.0625 --place_solver pv2
--redundant_per_rank 2`, so eps IS in the canon data.

## 0. Inputs and cadence

| what | value | why |
|---|---|---|
| Placement basis | the PREVIOUS decode window's routing (oracle basis `prev_batch`, gap 0) [capsule] | the only information an inference server has before the batch arrives; self-oracle is a ceiling ablation, never the arm |
| How stale that basis is | drift 46% of demand at 4n b1, 22% at 16n b4 [capsule `epic_pll_oracle_drift_ppm`] | this is the gap routing must absorb every iteration, since placement does not move |
| Placement cadence | once, at setup, untimed | s1 = static placement; per-iteration re-placement is the s2 ablation family, which wins no plotted group |
| Routing cadence | every iteration, timed (`plan_comm` + `plan`) | the batch's top-k is new each iteration |
| Slots | G/R home slots + 2 spare per GPU (r2): 32 copies at 4n, 128 at 16n [capsule `pv2_replicas`] | spare slots are the only room the counts stage has to split hot experts; R_red=0 had none (handoff 21 parity) |
| Balance slack | eps = 1/16: each GPU may take 6.25% more rows than an even split [capsule cmdline] | the single knob trading GEMM-side balance for wire locality |

## A. Placement (three decisions, sequential)

### P1 — how many copies of each expert
Rule: start every expert at one copy; hand out the spare copies one at
a time, each to the expert with the largest per-copy load
`load[g] / c[g]` (ties: lower id); an expert already on every node
(`c = NN`) is retired.
Rationale: per-copy load, not raw load, is what a GPU will actually
see, so splitting the hottest expert first is the greedy that lowers
the maximum expected GEMM load fastest. The `c <= NN` cap exists
because the transport deduplicates per node: a second copy inside a
node cannot reduce wire bytes, only the node choice can.
[capsule] 4n: `c_max = 4 = NN` (the hottest expert reached the cap);
16n: `c_max = 8 < 16` (no expert needed every node). No spill either.

### P2 — which node hosts each copy
Rule: experts in descending per-copy load; each copy goes to the node
with the largest oracle demand for that expert among nodes that have a
free slot and do not already host it (ties: smaller accumulated
per-copy load on the node, then lower id). Leftover slots are backfilled
with the highest-load experts missing from that node.
Rationale: with node-dedup, wire bytes depend only on WHICH NODES host
an expert, so the copy belongs where the demand is (affinity). Node
load is a tie-break, not the objective — the offline assessment showed
a load-first spread loses 99% at Qwen 4n. Highest-load experts choose
first because they have the most rows to keep local.

### P3 — which GPU inside the node
Rule: per node, sort its copies by per-copy load and deal them across
the L GPUs in a snake (0..L-1, L-1..0, ...).
Rationale: any GPU in a node is equivalent for the wire, so this stage
only levels expected compute. A snake over a sorted list is the
classic near-LPT balance in one pass, and it is deterministic.

Output: for every expert, the set of (node, GPU) hosting it. Nothing
else about the placement is used later — routing does not consult the
per-copy load.

## B. Routing (every iteration, every GPU for its own rows)

### R0 — publish demand
Rule: each GPU counts, per expert, how many of its top-k rows want it,
and all-gathers that histogram (`plan_comm`).
Rationale: every later table is a pure integer function of this
histogram, so after one small collective every GPU holds identical
quotas and shares with no further coordination.

### R1 — own GPU first
Rule: a GPU serves its own demand for experts it hosts, up to the cap
`ceil((1+eps) * S * K)`; if over, scaled down per expert by largest
remainder, the excess falling through.
Rationale: zero bytes moved, zero latency; the cap keeps a GPU that
hosts the hot expert from swallowing the whole batch's compute.

### R2 — own node next
Rule: remaining demand for experts hosted on other GPUs of the same
node is granted to those GPUs in proportion to their residual capacity
(cap minus R1 load), each GPU's intake clipped at its residual, sources
served in GPU order.
Rationale: NVLink traffic costs no wire bytes, so it is the next
cheapest place; proportional-to-residual keeps the node's GPUs level
without knowing the batch in advance.

### R3 — remote nodes: cover, not spread
Rule (two parts).
(a) Shares: each remote GPU's residual capacity is pre-partitioned into
per-source tickets proportional to each source's leftover demand for
that GPU's experts (largest-remainder integers).
(b) Cover: per token, score each node by how many of the token's
unserved experts it hosts on a GPU where the source still holds
tickets; send all those experts to the top node (ties: home node, then
lower id), each to its hosting GPU with the most tickets; repeat for
the remainder, three rounds.
Rationale: the wire cost of a token is the number of distinct remote
nodes it touches (its row travels once per node), so the objective is
fewest nodes per token — a set cover, solved greedily per token. An
even split across an expert's copies would raise that count. Tickets
replace live global capacity feedback: they let every GPU respect the
caps using only the histogram, and as a source spends tickets on a
popular GPU that node drops out of its later covers, shifting later
tokens elsewhere on its own. Price: no donation between sources.

### R4 — forced residue
Rule: a row no node can cover for this source goes to a per-expert
fallback = the hosting GPU with the least load after R1-R2 (frozen
table). Forced traffic per (source, destination) is budgeted by `f_cap`
(24 rows at 4n b1, 64 at 16n b4 [capsule]); a spent budget tries the
expert's other hosts, then takes the fallback anyway and raises an
overflow counter that s1 asserts is zero.
Rationale: routing must be total before decisions are published; the
cap is hard for R1-R3 by construction and soft only here. The residue
is the accepted compute imbalance created by no-donation and by hot
experts with few hosts. The budget is what makes receive buffers
provably sized at setup.

### R5 — publish decisions
Rule: one fused all-gather of (chosen physical slot | gate prob) per
row; every GPU derives its send/receive splits from it.
Rationale: R3's per-row choices are made with relaxed atomic tickets
(2026-08-21 ruling), so they are not bit-reproducible and cannot be
recomputed by another GPU; publishing them is cheaper than a
deterministic global recompute (0.11 ms at 4n b1, handoff 26).

## C. What is and is not deterministic

| layer | deterministic? |
|---|---|
| placement (P1-P3) | yes — integer, stable sorts, identical on every GPU |
| routing tables (R1-R3a quotas, grants, shares; R4 fallback) | yes — pure functions of the R0 histogram, kernel unit-gated equal to the torch reference |
| routing per-row choices (R3b cover tickets, R4 admission) | no — relaxed atomics; counts per destination bounded, exact map not reproducible |

## D. What the capsules record and what they do not

Recorded per cell: eps, solver, replica count, c_max, spill, f_cap,
recv cap, oracle basis + drift, predicted remote rows, out_sha.
NOT recorded: forced-row counts, per-tier row split, realized
token-node incidence. The kernel returns those stats every iteration
(`kstats`) but the driver only asserts on the overflow entry. If the
figure wants "x% of rows stayed on-node", it needs one re-run with the
stats logged (python-only change) — no binary rebuild.

## E. Q&A (2026-09-05, user questions)

**R2 — who decides how much intra-node traffic lands where?**
Correction to the R2 rule as first written: because placement never
puts two copies of one expert in the same node (P1 cap, and backfill is
node-level too), every expert has exactly ONE hosting GPU per node. So
the "proportional to hosting GPUs' residual" clause is degenerate in
the canon data. What actually happens on a host GPU h with residual
`resid[h]` (cap minus its R1 load):
1. The node's outside demand for each expert h hosts is summed
   (`D[node, g]`, over the node's other GPUs).
2. If the total exceeds `resid[h]`, the residual is split ACROSS
   EXPERTS in proportion to their node demand (largest remainder).
3. Within one expert, the granted rows are handed to the demanding
   GPUs in ascending GPU id (a prefix split), NOT proportionally to
   their demand. GPU 1 is served before GPU 3.
So: if GPU 3 demands a lot of an expert on GPU 0, GPU 3 does send more
to GPU 0 — up to the expert's grant — but it is last in line; when the
grant is short, the lower-id GPUs are served first and GPU 3's excess
leaks to R3. Example: GPU 0 hosts C with residual 4; GPUs 1, 2, 3 want
C: 1, 1, 3. Grant 4 → GPU 1: 1, GPU 2: 1, GPU 3: 2, and 1 row of GPU 3's
leaks to R3. If GPU 0 also hosted D demanded 2 by GPU 2, the residual
4 splits C:D ≈ 3:1 first, then C's 3 rows go 1, 1, 1 (GPU 3 leaks 2)
and D's 1 row to GPU 2 (leaks 1).

**R3 — greedy cover vs first-come-first-served?** Both, at different
levels. Across GPUs there is no contest at all: each source holds its
own pre-partitioned tickets on every remote GPU, so sources never
compete. Within one source, all its tokens are routed in parallel by
the kernel (a thread per token). Each thread runs its own greedy
(best-covering node first, claim tickets for every expert it can place
there, then the next node for what remains). When two tokens of the
same source want the last ticket on the same GPU, whichever thread's
atomic lands first wins — that is the first-come-first-served part, and
it is exactly the relaxed, non-reproducible piece. In the torch
reference the same contest is resolved by token index ascending.

**R4 — "fill it to whichever expert needs it"?** Not quite: routing
never changes WHICH expert a row uses (that is the gate's top-k); it
only chooses WHICH COPY. A forced row goes to the least-loaded GPU that
hosts that row's expert (load frozen after R2), within the per-pair
forced budget; if that host's budget is spent, the expert's other hosts
in ascending id; if all spent, the least-loaded host anyway plus the
overflow counter.
