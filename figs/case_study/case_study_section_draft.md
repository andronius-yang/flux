# Case study section — draft (2026-09-15)

Figure: `figs/case_study/CS_v5.{pdf,svg}` (rows: Prof. law rotating on top,
LiveCodeBench below). Data: pv3c capsule `20260915-072722`, 4 nodes x 4 A100,
64 MiB send budget, top-k 8, Qwen3 routing; OURS arm = placement + pv3c routing
+ overlapping scheduling + 3D-scheduled expert swap. Every number below is
from the drawn iterations (`CS_v5_ranks.csv`, `00_data_note.md`) or from
`figs/ablation_cycling/{figure_src.csv,sc_d4_iteration_series.csv}`.
Bracketed notes are for the authors, not for the paper.

---

## Case study

Figure X shows one iteration of a 4-node MoE layer (64 MiB dispatched per
GPU) on two workloads, as a per-rank timeline with three lanes: NIC RDMA,
NVLink, and GPU compute (with host phases hatched). For each workload we draw
the two ranks with the longest and the shortest expert GEMM in that
iteration, so the vertical distance between the two GPU lanes is the compute
imbalance the system had to absorb. The vertical bar is the collective end
of the iteration, which all sixteen ranks reach within 0.1 ms.

**Under drift (top).** The upper rows come from the professional-law block of
an 8-topic rotation, two iterations after the topic switched. The expert
placement was derived without this topic, so its hot experts are
under-replicated and no routing inside the load band can equalise the work:
expert-GEMM time spans 22.5 to 40.9 ms across ranks (1.82x), and the drawn
ranks are those two extremes. The imbalance is spatial as well as per-rank:
one node holds the four longest layer-1 GEMMs.

The expert-swap decision reacts to this in the same iteration. It runs on the
demand histogram before routing and fires on all sixteen ranks; the hottest
rank pulls sixteen expert slots over NVLink (10.8 ms of copy time), the
coldest two. The 3D schedule splits the hot rank's copies into a burst in the
host gap and under the layer-0 GEMM (5.0 ms) and a second burst under the
layer-1 GEMM (5.2 ms). Both bursts sit entirely beneath compute (green blocks
under orange), so the movement adds nothing to the rank's span even at this
volume.

What does set the iteration length is combine egress. The hot node's layer-1
GEMMs end late, its combine puts start late and serialise on the NIC (the
drawn hot rank's three puts run from 48.5 to 59.5 ms), and its per-rank NIC
occupancy is the highest in the system (put time spans 24.6 to 53.8 ms
across ranks, 2.19x). The cold rank finishes sending at 48.6 ms and idles for
12.9 ms waiting for the hot node's partials; the hot rank finishes receiving
at 55.9 ms and waits for its own egress to drain. The grey blocks on both
rows are the same imbalance seen from its two ends.

This frame is one point on the trajectory the rotation ablation (Figure Y,
right panel) averages over. On the same four-iteration dwell, the
professional-law block without expert movement stays at 71.5 to 74.5 ms
across all four iterations: the mismatch is permanent. With the overlapped
swap the same block reads 60.1, 57.9, 55.9, 54.9 ms, descending as each
iteration moves hot experts toward their demand [the drawn iteration is the
second of the block, i.e. mid-convergence]. Over the whole schedule this is
the difference between the "+ overlapping scheduling" and "+ expert swap
overlap" bars (block mean 72.8 to 57.2 ms; schedule speedup 1.07x to 1.11x):
every topic switch pays the convergence cost once per dwell, and the swap is
what makes the cost decay instead of persist.

**Under a matched placement (bottom).** The lower rows come from the steady
LiveCodeBench workload, where the placement was derived from the same
distribution it serves. Here the routing constraint is feasible and the
timeline shows the intended steady state.

*Compute balance.* Expert-GEMM time spans 24.2 to 27.1 ms across ranks
(1.12x); the drawn extremes are 26.9 and 24.4 ms. Every replica's load sits
within the (1 +- C) band of its fair share, and the placement supplies enough
replicas of the hot experts for that band to be feasible, so the GEMM
lengths, which are the per-rank sums of those loads, are bounded by
construction.

*Dispatch hidden under compute.* Each rank issues exactly three NIC puts, one
per remote node, because every token crosses the network once per
destination node; the fan-out to the four GPUs of a node is twelve NVLink
copies (three direct to node mates, nine gateway forwards of the three
remote windows), identical on every rank. All of it finishes inside the
layer-0 GEMM (6.0 to 21.5 ms on the hot rank), which starts before any
remote data has landed and consumes windows in arrival order.

*Combine hidden under compute.* The layer-1 GEMM (23.3 to 34.6 ms) releases
the combine wave by wave: the four intra-node pre-reduce rounds start at
25.8, 28.8, 32.5 and 34.8 ms and the three NIC puts at 32.4 and 35.3 ms,
while the GEMM is still running; the top-k reduce is the receiving side of
the same waves. Only the last 2 to 5 ms of the iteration is exposed on any
rank.

*Minimal expert swap.* The decision sees a histogram that barely moved, so
barely anything moves: ten of sixteen ranks copy two to four slots, the
drawn ranks two each (0.75 ms per copy), one in the host gap and one under
layer 1. The iteration ends at 49.1 ms against 63.0 ms under drift, and the
residual wait on the hot rank (5.5 ms) is again the combine egress of the
heaviest sender elsewhere rather than its own compute.

[Author notes. (1) The plan/metadata block, about 5 ms in both rows, is on
the critical path; own it in one sentence in the text as the price of
per-iteration constraint-faithful planning. (2) NIC egress imbalance (1.86x
steady, 2.19x drift) is the honest residual in both rows and motivates NIC
balancing as an optimization distinct from routing and placement. (3) No
COMET comparison is made in this section by design; the COMET timelines in
the data note are from a different capsule.]
