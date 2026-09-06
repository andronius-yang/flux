# Routing figure — expert-centric fill + one-token view, with measured proportions (2026-09-05)

User's diagram plan (9/5 pm): one topology, two registers — (1) an
expert-centric rectangle = one copy's compute load, shaded by where its
rows came from; (2) one token on another node, showing how its top-k set
shapes the decisions. Numbers below are computed OFFLINE by
`tier_fill_offline.py`: the deterministic torch reference router on the
plotted cells' REAL routing files, with the pv2 placement solved from
their REAL oracle windows (eps 1/16, 2 spare slots) — i.e. the plotted
arm's own inputs. Files: `tier_fill_cells.csv` (one row per expert
copy), `tier_fill_summary.csv` (per cell). Caveat: forced rows are
uncapped here (reference f_cap None) and ticket races resolve by token
index instead of atomics — counts per tier are contract-identical, the
exact row map is not.

## 1. Corrections to the plan

| plan | verdict |
|---|---|
| shade 1 = "node local ... originate from the same GPU" | rename: shade 1 = **GPU-local** (source GPU hosts the expert, tier 1); shade 2 = **node-local** (tier 2) |
| shade 2 "will take the majority" | **No.** Remote (tier 3) is the majority everywhere: 52-66% of all rows. GPU+node local together = 36% at 4n, 22-27% at 8n, 14-20% at 16n (table §2) |
| R3 picks NODE destinations | Yes. The cover chooses the node; the GPU is then fixed by placement (one copy per node) |
| merge R3 and R4? | Keep separate: R4 is the only shade allowed ABOVE the cap line. Draw the cap on the rectangle; forced rows are the sliver spilling over it |
| R4 "whichever token still has remaining experts, pick those" | Yes, but the destination is the least-loaded host OF THAT EXPERT (frozen post-R2 load), never "whichever expert" |

## 2. Global tier split (% of all routed rows), plotted cells

| nodes | model | 1 MiB | 4 MiB | 16 MiB |
|---|---|---|---|---|
| 4 | K2 | 8.8 / 26.7 / 63.0 / 1.5 | 9.1 / 27.7 / 62.3 / 1.0 | 9.2 / 27.5 / 62.4 / 0.9 |
| 4 | Qwen | 9.5 / 30.4 / 53.5 / 6.6 | 10.3 / 30.8 / 52.6 / 6.3 | 10.2 / 30.6 / 52.7 / 6.4 |
| 8 | K2 | 5.4 / 17.1 / 65.2 / 12.3 | 5.6 / 16.9 / 66.0 / 11.5 | 5.7 / 17.2 / 65.7 / 11.4 |
| 8 | Qwen | — | 6.8 / 20.5 / 63.1 / 9.6 | 6.9 / 20.7 / 62.9 / 9.5 |
| 16 | K2 | 3.6 / 10.9 / 62.7 / 22.7 | 3.6 / 10.7 / 63.4 / 22.2 | 3.6 / 10.8 / 63.3 / 22.4 |
| 16 | Qwen | 5.0 / 15.1 / 64.5 / 15.4 | 5.1 / 15.2 / 64.5 / 15.2 | 5.1 / 15.2 / 64.5 / 15.2 |

Cell = GPU-local / node-local / remote-cover / forced. Budget barely
matters (the split is a property of placement x routing, not of batch
size); node count does: local shrinks ~1/NN, forced grows with NN
because tickets get scarcer (cap is only 6.25% above the mean load).

Why local is not the majority: with G/W home slots + 2 spares per GPU,
only 21 (K2 4n) to 81 (K2 16n) experts have more than one copy, so a
typical expert lives on ONE node and 1 - 1/NN of its demand is remote
by construction. Replication is what raises the local share of the
hot experts (see §3).

## 3. Per-copy proportions to draw (4 nodes, K2, 4 MiB; cap 2516 rows/GPU)

Across all 416 copies: median self 6.0%, node 19.5%, remote 73.5%,
forced 0% (p75 also 0 — forced is concentrated on a few copies).

| expert | copies | copy on | rows | self | node | remote | forced |
|---|---|---|---|---|---|---|---|
| 110 (hottest, on every node) | 4 | N0 GPU 0 | 237 | 28% | 72% | 0% | 0% |
| 110 | 4 | N2 GPU 8 | 253 | 22% | 78% | 0% | 0% |
| 119 | 3 | N0 GPU 1 | 328 | 13% | 46% | 40% | 0% |
| 119 | 3 | N3 GPU 13 | 243 | 20% | 68% | 12% | 0% |
| 18 (single copy) | 1 | N2 GPU 8 | 64 | 8% | 20% | 72% | 0% |

The picture this gives: a fully replicated expert is served entirely
inside each node (self + node = 100%, zero wire); a 3-copy expert takes
remote rows only from the one node without a copy; a single-copy
expert is ~3/4 remote. That IS the placement lever, visible in the fill.

GPU-level fill (rectangle = one GPU's compute, cap line at 1+eps):
min 87%, median 93%, max 106% of cap; 3 of 16 GPUs over the cap, all by
forced rows (over-cap rows = 0.6% of all rows). A GPU rectangle
example, GPU 0 (hosts 26 experts, 2348/2516 rows), its six largest
segments (rows self/node/remote/forced): 247 (2 copies) 287 =
27/90/170/0; 11 (2) 239 = 22/83/134/0; 110 (4) 237 = 67/170/0/0; 326 (2)
171 = 20/62/89/0; 9 (4) 158 = 41/117/0/0; 303 (1) 158 = 8/33/117/0.

At 16 nodes (K2, 4 MiB, same cap) the same rectangles read: median copy
self 2%, node 6%, remote 63%, forced 26% (p25 0, p75 43%); GPU fill
min 50%, median 96%, max 142%; 26 of 64 GPUs over cap (5.9% of rows
over cap). Forced sits on single-copy experts of GPUs whose residual
was consumed by their other experts — e.g. GPU 0: expert 255 (1 copy)
385 rows = 6/19/205/155.

## 4. The token panel — measured incidence

Per token (top-k 8), mean over all tokens of the cell: remote rows =
rows served outside the home node; "nodes (cover)" = distinct remote
nodes the token actually touches under the greedy cover; "nodes
(even)" = what the SAME placement would give if each remote row picked
one of its expert's copies uniformly at random (the "spread evenly
across replicas" alternative; caps ignored, so it flatters even-split).

| nodes | model | remote rows / token | nodes touched: cover | nodes touched: even split | tokens fully local |
|---|---|---|---|---|---|
| 4 | K2 | 5.1 | 2.36 | 2.47 | 0.1% |
| 4 | Qwen | 4.7 | 1.96 | 2.27 | 0.7% |
| 8 | K2 | 6.2 | 3.58 | 4.12 | 0.0% |
| 8 | Qwen | 5.8 | 2.95 | 3.70 | 0.1% |
| 16 | K2 | 6.9 | 4.46 | 5.50 | 0.0% |
| 16 | Qwen | 6.4 | 3.39 | 5.01 | 0.0% |

(4 MiB rows; 1 and 16 MiB are within 0.03.) Reading: a K2 token at 16n
has ~7 of its 8 experts off-node but its row travels to only ~4.5
nodes; even-split would send it to ~5.5 (+23%); Qwen 16n: 3.4 vs 5.0
(+48%). Upper bound if every remote row went to a different node =
the remote-rows column. This is the number the token panel can carry:
"8 experts, ~7 remote, 4.5 nodes".

## 5. Drawing guidance (routing panel)

- Register 1, expert-centric: draw THREE copies of one expert side by
  side (fully replicated / partially / single) so the fill shades tell
  the replication story; or one GPU rectangle with expert segments and
  the cap line. Use §3 proportions verbatim (cite cell 4n K2 4 MiB).
- Shades: GPU-local (darkest) -> node-local -> remote (lightest);
  forced hatched, above the cap line only.
- Register 2, one token on another node: 8 top-k stubs; 1 served on its
  own GPU, 1-2 on the node, the rest grouped by the cover into ~2.4
  remote nodes at 4n (draw 2 remote node boxes each receiving several
  stubs — the "many experts, one node" glyph), one stub forced with a
  dashed arrow to the least-loaded host.
- Vocabulary: keep "expert placement & routing"; name tiers GPU-local /
  node-local / remote / forced (not R1-R4) in the figure.
