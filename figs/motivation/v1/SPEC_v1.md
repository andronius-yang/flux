# Motivation figure — three-lane timeline, SPEC v1 (2026-09-03)

Status: v1, awaiting postdoc review. Every ruling below came from the user
on 2026-09-02/03; knobs are marked. Generator: `build_v1_lanes.py`
(inputs: the committed phase JSONs of capsules 20260902-133340 and
20260902-140327; outputs: `lanes_v1.svg` review render, `lanes_v1.drawio`
editable twin, `lanes_v1_ranks.csv` rank ledger).

## Format
- NSDI single column: 241 pt (3.33 in) wide; height follows content
  (~150 pt). Fonts: Helvetica 7 pt labels, 6 pt annotations, ink #17191c.
  Single light look (paper figure), no theme switching.
- Three baselines stacked vertically, in this order: NVSHMEM a2av + GEMM,
  EPLB (exposed dispatch wire lane), COMET. Padding between baselines
  10 pt; inside a baseline the lanes are near-adjacent (0.5 pt between
  lanes of one rank, 2 pt between ranks).
- Per rank three lanes, top to bottom: NIC (inter-node RDMA), NVL
  (intra-node P2P), SM (expert GEMM). Lane height 3.2 pt.
- Left gutter: rank id (r<N>). Right gutter: routed rows and inter-node
  dispatch bytes for that rank (from the capsule matrix / recorded wire
  matrix), 6 pt.

## Horizontal structure (user ruling 2026-09-03)
- **Expert placement column** (EPLB only; ring and COMET rows are empty
  with a small "no placement" note): RELATIVE scale — the longest of the
  four drawn ranks fills the column, the others scale to it. Bars = the
  rank's placement puts (NIC lane = inter-node puts, NVL lane = intra-node
  puts). Column width is a KNOB `placement_frac` (default 1/3 of the
  drawable width; 0 removes the column).
- **Token dispatch + compute**: ONE ABSOLUTE ms scale shared by all
  three baselines (KNOB `absolute=True`; `False` = per-baseline relative).
  Each rank's time origin is its first dispatch wire event (routing
  all-gather / plan / pack are excluded — they are prep, stated in the
  caption). The window ends at the rank's own layer-0 end (`l0_end`).
- **Barrier glyph**: for the un-overlapped variants (ring, EPLB) the
  dispatch barrier is drawn as a double vertical line `||` spanning the
  rank's three lanes at the barrier RELEASE time; the empty NIC lane
  between the last put and `||` is the rank's wait. No hatching. Legend
  entry: "|| barrier". COMET (overlapped) gets no glyph; its post-GEMM
  barrier is left as empty SM lane and explained in the caption.

## Rank rule (4 per baseline, user ruling)
In order, skipping ranks already chosen: (1) longest layer-0 total,
(2) shortest layer-0 total, (3) longest inter-node wire occupancy
(EPLB: longest placement span instead), (4) longest expert GEMM;
fallbacks (5) shortest inter-node wire, (6) shortest GEMM. Total = first
wire event to `l0_end` on that rank. Ranks may come from any node.

## Data
- Budget b16, middle timed window (iter4), both capsules on the same
  python-only binary; nsys under isolated discipline; correctness on.
- Classes: NIC = NVSHMEM proxy RMA kernels (ring puts, COMET fetches,
  EPLB exposed-wire puts); NVL = CUDA P2P copies; SM = grouped-GEMM
  launches; placement = the puts inside `eplb_place_weights`.

## Caption facts to carry
- Send bytes are equal on every rank by construction (same token budget);
  the wire spread is receive-side incast and ring position.
- EPLB panel runs the exposed-wire lane (blocking put per destination),
  not the staged a2a kernel of the main results; same bytes, same rows.
- COMET's NVLink copies overlap its inter-node fetch, not its GEMM; the
  GEMM waits for the last remote fetch.

## Knobs
`placement_frac` (default 0.333), `absolute` (default True),
`ranks_per_variant` (default 4), `budget` (default 16), `iteration`
(default middle timed), `lane_h`/`gaps` (see generator header).
