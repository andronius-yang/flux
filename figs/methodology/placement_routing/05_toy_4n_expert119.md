# 4-node toy for the routing figure: expert 119 + token 1296 (2026-09-05)

Real data, not invented: cell `ours_l01_s1_pv2_r2_trace-610042_b4_k8_isolated`
(capsule 20260829-143523, K2, 4 nodes x 4 GPUs, 4 MiB, top-k 8), routed
offline by the reference router on the cell's own inputs
(`toy_4n_k2_b4_expert119.txt` is the raw dump). Cap = 2516 rows/GPU.

## Register 1 — expert 119, expert-centric

Placement facts (why it looks like this):
- Oracle demand by node: N0 533, N1 510, N2 527, N3 532 (load 2102).
  P1 gave it 3 copies; P2 put them on the three highest-demand nodes
  N0, N3, N2 — N1 (510, lowest) got none. Copies sit on GPU 1 (N0),
  GPU 8 (N2), GPU 13 (N3).
- Batch demand by node: N0 196, N1 204, N2 192, N3 213.

Fill of each copy (rows = self / node-local / remote / forced):

| copy | rows | GPU-local | node-local | remote | forced | GPU total / cap |
|---|---|---|---|---|---|---|
| N0, GPU 1 | 328 | 44 (13%) | 152 (46%) | 132 (40%), all from N1 | 0 | 2211 / 2516 = 88% |
| N2, GPU 8 | 234 | 49 (21%) | 143 (61%) | 42 (18%), all from N1 | 0 | 2299 / 2516 = 91% |
| N3, GPU 13 | 243 | 49 (20%) | 164 (67%) | 30 (12%), all from N1 | 0 | 2549 / 2516 = 101% (over by forced rows of OTHER experts) |

What the three rectangles say:
- Every node with a copy serves its own demand entirely locally
  (self + node-local = 100% of that node's 119 rows); zero wire.
- The only remote rows are N1's 204, and they split 132 / 42 / 30
  across the three copies — NOT evenly. The split is a by-product of
  N1's tokens' covers (register 2): N0 hosts the most of what N1's
  tokens need, so 119 rides along to N0 most of the time.
- GPU 13 is over the cap line by 33 rows although expert 119 itself has
  no forced rows: the cap is per GPU, and the overflow belongs to other
  experts on GPU 13. Draw the cap on the GPU, not on the expert.

## Register 2 — token 1296 on N1 (GPU 4), the node with no copy of 119

Its 8 experts, their copies, and the decision:

| expert | copies on nodes (GPU) | tier | destination |
|---|---|---|---|
| 110 | N0, N1, N2, N3 (0, 4, 8, 12) | GPU-local | GPU 4 (itself) |
| 127 | N0, N1, N2 (2, 6, 10) | node-local | GPU 6 |
| 119 | N0, N2, N3 (1, 8, 13) | remote, cover round 1 | N0 -> GPU 1 |
| 219 | N0 (3) | remote, cover round 1 | N0 -> GPU 3 |
| 304 | N0 (3) | remote, cover round 1 | N0 -> GPU 3 |
| 312 | N0 (3) | remote, cover round 1 | N0 -> GPU 3 |
| 334 | N0 (2) | remote, cover round 1 | N0 -> GPU 2 |
| 311 | N3 (15) | remote, cover round 2 | N3 -> GPU 15 |

Walk:
1. GPU-local: 110 is on GPU 4 itself (it is the everywhere-replicated
   hottest expert) -> stays.
2. Node-local: 127 has a copy on GPU 6, same node -> NVLink.
3. Remote cover: six experts remain {119, 219, 304, 312, 334, 311}.
   Node scores = how many of the six each node hosts (on GPUs where
   GPU 4 still holds tickets): N0 = 5, N1 = 0, N2 = 1, N3 = 2.
   Round 1: N0 wins -> 119, 219, 304, 312, 334 all go to N0 (three of
   them land on the same GPU 3). Round 2: 311 remains, only N3 hosts
   it -> N3. **Two remote nodes for six remote experts.**
4. Forced: none for this token (4n K2 forces ~1% of rows; a forced
   row is the exception here, not the rule — at 16n it is 22%).

The glyph: expert 119 has three possible copies (N0, N2, N3). Nothing
about 119 chose N0 — the other four remaining experts did, and 119
rode along at zero extra node visits. An "even split across 119's
copies" would have sent it to N2 or N3 two times in three, adding a
third node to this token's trip.

Cover vs even split for the whole cell (mean per token): 2.35 vs 2.46
nodes touched, with 5.1 of 8 experts remote.

## Drawing notes
- Register 1: three copy rectangles for 119 in one row, ordered N0,
  N2, N3, each with the same three shades bottom-up (GPU-local
  darkest, node-local, remote lightest) at the proportions above;
  a small "N1" tag on every remote slice (all remote rows are N1's).
  Optionally a thin GPU bar next to GPU 13 with the cap line crossed.
- Register 2: token 1296 as a card on N1/GPU 4 with 8 stubs; arrows:
  one loop to itself (110), one to GPU 6 (127), a bundle of five to N0
  (fan-in to GPUs 1, 2, 3), one to N3 (311). Node scores 5 / 0 / 1 / 2
  written next to the nodes. Ghost the two unused copies of 119 on N2
  and N3 (the "not even split" point).
- Both registers share the same 4-node picture; 119's copies are the
  common element (register 1's rectangles ARE the boxes register 2's
  arrow lands on at GPU 1).

## Why N0's copy gets 132 of N1's 204 rows (measured, `toy_4n_k2_b4_why_n0.txt`)

Not a global bias toward N0. Checked on the same cell:
- N1's remote rows overall split N0 1947 / N2 2078 / N3 1858 — near even.
- N1's demand for experts hosted only on N0 / N2 / N3: 1564 / 1830 / 1712
  rows — N0 hosts the LEAST of what N1 needs, not the most.
- N1's tickets on N0 / N2 / N3: 2281 / 2411 / 2005; residual capacity
  6692 / 6647 / 6500 — even.

It is specific to the 204 N1 tokens that request 119. Their round-1
node scores (hosts among the remaining experts): N0 strictly best in
80, N2 in 32, N3 in 30, and 62 ties — 52 of which include N0. The
cover's tie rule is "home node, then LOWER node id", so N0 takes all 52.
80 + 52 = 132 exactly. Two causes, then:
1. Co-occurrence (80 of 132): tokens that want 119 tend to also want
   single-copy experts that placement put on N0 — a property of this
   routing trace, not of the algorithm.
2. Tie-break (52 of 132): the deterministic lower-id rule hands every
   tie to N0. Systematic, but washed out at the global level (the
   near-even split above) because strict wins dominate overall.
Figure note: if the diagram annotates the 132/42/30 split, call it
"decided by the tokens' other experts (and ties to the lowest node
id)", never "N0 has more capacity" or "N0 is closer".

## Is the lower-id tie rule a flaw? (offline ablation, `tiebreak_ablation_offline.py`)

Swapped the tier-3 tie rule in the reference router for "home node, then
the node where the source holds the MOST remaining tickets, then lower
id" and re-routed the same cells (K2, 4 MiB):

| cell | rule | forced rows beyond shares | over-cap rows | remote nodes / token | max GPU load / cap |
|---|---|---|---|---|---|
| 4n | canon lower-id | 360 (1.0%) | 213 | 2.349 | 2668 / 2516 |
| 4n | ticket-aware | 372 (1.0%) | 200 | 2.349 | 2636 / 2516 |
| 16n | canon lower-id | 12433 (8.2%) | 8893 | 4.457 | 3582 / 2516 |
| 16n | ticket-aware | 10520 (6.9%) | 5756 | 4.409 | 3357 / 2516 |

Verdict: not a flaw in the wire objective — a tie means the token
touches the same number of nodes either way, and nodes/token is
unchanged at 4n, −1% at 16n. It is a load-balance simplification whose
cost appears at scale: at 16n the ticket-aware rule cuts over-cap rows
by 35% and the worst GPU's overload from +42% to +33% of cap. At 4n it
is harmless. Cause: ties spend the lowest-id node's tickets on rows
that did not need them, starving later strict covers there.
The fix is one term in the cover key (kernel `pll_route3_kernel_t` and
the reference), untested on hardware — a NEW arm if ever measured
(never-mix). For the paper: describe the rule honestly as
"deterministic tie-break", not as a design choice with a rationale.
