# Expert placement & routing — summary walkthrough with toy examples (2026-09-05)

Self-contained. Supersedes the toy placement table in
`02_routing_toy_example.md` (that one was hand-picked; this one is the
solver's actual output on the toy histogram, so the two diagrams chain).
Mechanism = the arm plotted as "Ours" in the main-perf figure
(`ours_l01_s1_pv2_r2`): eps 1/16, pv2 placement, 2 spare slots per GPU,
all confirmed from the winner cells' launch logs.

## 0. The setting

- Cluster: NN nodes x L GPUs. Toy: 3 nodes x 2 GPUs (N0 = G0,G1;
  N1 = G2,G3; N2 = G4,G5).
- Experts: G. Toy: 8 experts A..H. Each GPU has G/R = 8/6 home slots
  rounded to the plan's `nlp`; toy: 1 home slot + 1 spare = 2 slots per
  GPU, 12 slots, so 4 extra copies.
- Two cadences: **placement once, at setup, from the previous window**
  (the oracle); **routing every iteration, from the live batch**.
  Placement never moves in this arm; routing absorbs the drift (46% of
  demand at 4n, 22% at 16n in the real cells).
- One balance knob: **eps = 1/16**. Every GPU may take at most
  `cap = ceil((1 + eps) * S * K)` rows per iteration (S tokens per GPU,
  K = top-k), i.e. 6.25% over an even split.

## A. Placement — three sequential decisions

Input: the oracle histogram `hist[node, expert]` = rows each node
demanded of each expert in the previous window. Toy:

| node | A | B | C | D | E | F | G | H | sum |
|---|---|---|---|---|---|---|---|---|---|
| N0 | 30 | 14 | 8 | 6 | 2 | 2 | 2 | 2 | 66 |
| N1 | 24 | 10 | 2 | 2 | 14 | 8 | 2 | 4 | 66 |
| N2 | 18 | 4 | 2 | 2 | 14 | 2 | 8 | 6 | 56 |
| load | 72 | 28 | 12 | 10 | 30 | 12 | 12 | 12 | |

### P1 — how many copies (4 spare copies to hand out)
Rule: every expert starts at 1 copy. Repeat per spare copy: give it to
the expert with the largest **per-copy load** `load / copies` (ties:
lower id); an expert already on every node (copies = NN) is retired.

| step | per-copy loads (top 3) | pick | result |
|---|---|---|---|
| 1 | A 72, E 30, B 28 | A | A -> 2 copies, per-copy 36 |
| 2 | A 36, E 30, B 28 | A | A -> 3 copies, per-copy 24 (= NN, retired) |
| 3 | E 30, B 28, A 24 | E | E -> 2 copies, per-copy 15 |
| 4 | B 28, A 24, E 15 | B | B -> 2 copies, per-copy 14 |

Final per-copy loads: A 24, E 15, B 14, C 12, F 12, G 12, H 12, D 10.
Rationale: per-copy load is what one GPU will actually see; splitting
the hottest first lowers the expected maximum fastest. The one-copy-
per-node cap: the transport deduplicates per node, so a second copy in
the same node cannot save wire bytes — only the node choice can.

### P2 — which node hosts each copy (4 slots per node)
Rule: experts in descending per-copy load (ties: lower id). Each copy
goes to the node with the **largest oracle demand for that expert**
among nodes with a free slot that do not already host it (ties: lower
accumulated per-copy load on the node, then lower id).

| expert (per-copy) | copies | demand by node | placed on | free slots after (N0,N1,N2) |
|---|---|---|---|---|
| A (24) | 3 | 30 / 24 / 18 | N0, N1, N2 | 3,3,3 |
| E (15) | 2 | 2 / 14 / 14 | N1 (tie -> lower id), N2 | 3,2,2 |
| B (14) | 2 | 14 / 10 / 4 | N0, N1 | 2,1,2 |
| C (12) | 1 | 8 / 2 / 2 | N0 | 1,1,2 |
| F (12) | 1 | 2 / 8 / 2 | N1 | 1,0,2 |
| G (12) | 1 | 2 / 2 / 8 | N2 | 1,0,1 |
| H (12) | 1 | 2 / 4 / 6 | N2 | 1,0,0 |
| D (10) | 1 | 6 / 2 / 2 | N0 | 0,0,0 |

Node sets: N0 {A, B, C, D}, N1 {A, E, B, F}, N2 {A, E, G, H}. All slots
used; nothing to backfill.
Rationale: with node-dedup, wire bytes depend only on WHICH NODES host
an expert, so a copy belongs where the demand is. Node load is a
tie-break only (a load-first spread loses 99% at Qwen 4n offline).
Hot experts choose first because they have the most rows to keep local.

### P3 — which GPU inside the node
Rule: per node, sort its copies by per-copy load (ties: lower id) and
deal them across the GPUs in a snake: GPU 0, 1, ..., L-1, L-1, ..., 1, 0.

| node | sorted copies | snake (L = 2) | result |
|---|---|---|---|
| N0 | A 24, B 14, C 12, D 10 | G0, G1, G1, G0 | G0 {A, D}, G1 {B, C} |
| N1 | A 24, E 15, B 14, F 12 | G2, G3, G3, G2 | G2 {A, F}, G3 {E, B} |
| N2 | A 24, E 15, G 12, H 12 | G4, G5, G5, G4 | G4 {A, H}, G5 {E, G} |

Rationale: any GPU in a node is equivalent for the wire, so this stage
only levels expected compute; a snake over a sorted list is one-pass
near-LPT balance, deterministic. Nothing else about the placement is
used later — routing never consults the per-copy loads.

## B. Routing — every iteration, each GPU for its own rows

Toy batch: 4 tokens per GPU, top-2, so 8 rows per GPU; cap =
ceil(1.0625 * 8) = 9. We follow source **G0 (home node N0; hosts A, D)**:

| token | experts | hosts of each |
|---|---|---|
| t0 | A, D | A: G0 (self) ; D: G0 (self) |
| t1 | A, C | A: G0 (self) ; C: G1 (home node) |
| t2 | E, G | E: G3 (N1), G5 (N2) ; G: G5 (N2) |
| t3 | B, H | B: G1 (home node), G3 (N1) ; H: G4 (N2) |

### R0 — publish demand
Every GPU counts its rows per expert and all-gathers the histogram
(`plan_comm`). Every table below is an integer function of it, so all
GPUs hold identical quotas/shares with no further exchange.

### R1 — own GPU, up to the cap
G0's self-hosted demand: A x2 (t0, t1) + D x1 (t0) = 3 rows <= 9. All
stay. (At 12 rows it would keep 9, scaled down per expert by largest
remainder, and 3 would fall through.) G0 load 3.

### R2 — own node, from the host's residual
C (t1) and B (t3) are on G1. Say G1 kept 5 own rows in R1, residual 4.
Node demand on G1 from the other GPU: C 1, B 1 = 2 <= 4: both granted.
Zero wire bytes so far. G1 load 7.
How a short residual is split (the actual rule; every expert has ONE
host per node, so it is never split across hosts): first ACROSS
EXPERTS in proportion to node demand, then within an expert to the
demanding GPUs **in ascending GPU id** (prefix), the last GPU's excess
leaking to R3. Example: G1 residual 4; GPUs 0, 2, 3 want C: 1, 1, 3 ->
GPU 0 gets 1, GPU 2 gets 1, GPU 3 gets 2 and leaks 1.
(Toy has 2 GPUs per node, so only G0 asks G1; the prefix rule shows
with L = 4.)

### R3 — remote nodes: tickets, then fewest-nodes-per-token
(a) Tickets. Each remote GPU's residual (cap minus post-R2 load) is
pre-split among source GPUs in proportion to each source's leftover
demand for that GPU's experts (largest remainder):
- G5 (hosts E, G): residual 4; leftover demand for E/G from G0: 2,
  G2: 1, G3: 1 -> tickets 2 / 1 / 1. **G0 holds 2 on G5.**
- G4 (hosts A, H): residual 1; leftover from G0: 1, G2: 4, G3: 3 ->
  the single row goes to G2. **G0 holds 0 on G4.**
- G3 (hosts E, B): G0's remaining B is served in R2 already; E could go
  here; say G0 holds 1 ticket on G3.
(b) Cover, per token (all of G0's tokens in parallel, one thread each):
score each node by how many of the token's unserved experts it hosts on
a GPU where G0 still holds tickets; send all of them to the top node
(ties: home node, then lower id); repeat for the rest (3 rounds).
- t2 {E, G}: N1 covers E (G3, 1 ticket). N2 covers E AND G (G5,
  2 tickets). N2 wins: both rows -> G5, G0 spends its 2 tickets there.
  **One remote node touched.** An even split of E over its copies would
  have sent half to N1 and touched two nodes.
- t3's H: only on G4, where G0 holds no ticket -> no node covers it ->
  left unassigned.
Rationale: a token's wire cost is the number of distinct remote nodes
it touches (its row travels once per node), so the objective is fewest
nodes per token — a set cover, greedy per token. Tickets replace live
global capacity feedback: they let every GPU respect the caps from the
histogram alone, and as a source spends tickets on a popular GPU that
node drops out of its later covers, shifting later tokens elsewhere
with no exchange. Within one source, two tokens racing for the last
ticket on a GPU are resolved first-come-first-served (relaxed atomics).
Price of sender-local: no donation of unused tickets between sources.

### R4 — forced residue
H -> the least-loaded GPU hosting H (load frozen after R2) = G4, its
only host, inside the per-(source, destination) forced budget `f_cap`
(24 rows at 4n b1, 64 at 16n b4 in the real cells). Budget spent ->
try H's other hosts ascending; all spent -> G4 anyway + overflow
counter (must stay 0 in this arm). Routing never changes the EXPERT a
row uses, only the copy.
Rationale: routing must be total before decisions are published; the
cap is hard for R1-R3 by construction and soft only here. The residue
is the compute imbalance the design accepts, created by no-donation
and by hot experts with few hosts.

### R5 — publish decisions
One fused all-gather of (chosen slot | gate prob) per row; every GPU
derives its send/receive splits from it. Needed because R3/R4 per-row
choices are not reproducible by other GPUs.

**G0's 8 rows:** 3 stayed home (A, A, D), 2 crossed NVLink (C, B -> G1),
2 went to one remote node (E, G -> G5), 1 was forced (H -> G4).

## C. Determinism, in one table

| layer | deterministic across GPUs / runs? |
|---|---|
| placement P1-P3 | yes (integers, stable sorts) |
| routing tables: R1 quotas, R2 grants, R3 tickets, R4 fallback | yes (pure functions of the R0 histogram; kernel unit-gated equal to the torch reference) |
| routing per-row choices: R3 cover ticket claims, R4 admission | no (relaxed atomics, 2026-08-21 ruling); per-destination counts bounded, exact map not reproducible -> hence R5 |

## D. Numbers that may be quoted with the figure

| quantity | value | source |
|---|---|---|
| eps | 1/16 | winner cells' launch logs |
| spare slots | 2 per GPU (32 copies at 4n, 128 at 16n) | winner cells |
| oracle drift absorbed by routing | 46% (4n b1), 22% (16n b4) | winner cells |
| route kernel | 0.11 ms at 4n b1 (7% of the plan lane) | handoff 26 |
| plan bracket (router upper bound) | 0.6-0.9 ms 4n, 0.8-1.2 8n, 1.2-3.4 16n | `plan_bracket_main_perf.txt` |
| decisions all-gather | 0.11 ms at 4n b1 | handoff 26 |
| NOT recorded | forced-row counts, per-tier split, realized incidence | would need one logged re-run (python-only) |

## E. Drawing notes

- Draw two cadences explicitly (setup vs every iteration); never a
  re-solve loop, trigger, or moving weights — none exist in this arm.
- Placement panel: the histogram (tiny bars per node), P1 as a
  "split the tallest bar" sequence, P2 as arrows copy -> node labelled
  by the node's demand, P3 as the snake deal inside one node.
- Routing panel: one source GPU, its 4 tokens, the four tiers as
  concentric locality rings (self, node, remote cover, forced); the
  t2 cover is the key glyph (two experts, one node). Show the cap as a
  fill level on each GPU and the tickets as small stubs a source holds
  on remote GPUs.
- Vocabulary: "expert placement & routing" (ablation legend, verbatim).
