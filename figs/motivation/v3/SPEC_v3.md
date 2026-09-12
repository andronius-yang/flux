# Motivation figure — SPEC v3 (2026-09-12, user instructions of 2026-09-12)

Source of rulings: the user's written instructions of 2026-09-12 on top of
SPEC v2 (postdoc XML). Generator: `build_v3_lanes.py` (inputs: phase JSONs
of capsule 20260904-123815 for panels 1–2 and the MoonEP capture for
panel 3). Outputs: `lanes_v3.svg` (points), `lanes_v3.drawio` (editable
twin, layers background / bars / glyphs / axes / labels), `lanes_v3.png`,
`lanes_v3.pdf`, `lanes_v3_ranks.csv`.

## What changed from v2
- **Panel 3 = authentic MoonEP, not EPLB.** Arm `moonep_l01_nvshmem_getmem`
  (per-batch redundant-expert plan, destination-side getmem pull of w1+w2
  serialized before the GEMM, exactly upstream's shared-comm-stream
  order) drawn through its exposed-wire twin
  `moonep_l01_nvshmem_getmem_bwire` (blocking ring puts + world barrier in
  place of the staged a2a kernel; plan/placement/prefetch/GEMM/combine
  byte-identical — `sweeps/variants.py`). Per-batch expert movement is the
  story: it is drawn INSIDE the timeline (green, between the dispatch
  barrier and the GEMM) on the lane its home ranks live on (NIC when the
  pulled experts are homed off-node, NVLink when on-node — from the
  recorded `moonep_prefetch_pairs`), not as v2's relative block.
- **Titles = problem / solution statements** (two lines, split at "+"):
  1. "Inter-node communication bottleneck + Computation Imbalance" — all red
  2. "Communication Overlap + Computation Imbalance" — red = Computation Imbalance
  3. "Expert communication bottleneck + Computation Balance" — red = Expert
     communication bottleneck
  Red `#c0392b` marks the problem the subdiagram still has.
- **Arrows** (`#8a2e24`, block head) from a red problem to the subdiagram
  that resolves it: panel-1 line 1 → panel 2 (horizontal, title row);
  panel-2 line 2 → panel 3 (horizontal); panel-1 line 2 → panel 3 (arc over
  panel 2's title; knob `ARC_H` reserves the room above the titles).
- Height 27 % of width (v2: 25 %) for the arc; everything else (lanes,
  colors, background patterns, per-subdiagram ms axes, legend) as v2.
- Rank rule for panel 3: longest expert pull, shortest on-node (NVLink)
  expert pull (user ruling 2026-09-12: a real pull, not the no-pull rank),
  then longest / shortest inter-node token wire, fallbacks median pull /
  total extremes (4 ranks). Panels 1–2 keep the v1.1 rule.

## Data (2026-09-12 capture)
Panels 1–2: K2 mmlu/professional_law layer 18, b32, middle timed window
(iter4), capsule 20260904-123815 (unchanged from v2). Panel 3: same routing
and budget, capsule `20260912-221306_perlmutter_363dd11a` (job 58247219,
4/4 ok, bitwise + allclose green on all 16 ranks, deterministic 0; binary
ths_op 6d3261d1 — NOT the v2 capsules' 505e4bed, main was rebuilt since
2026-09-04, so panels 1–2 and panel 3 come from two builds; each panel has
its own axis and no cross-panel latency is quoted — SCHEMA rule 4), cell
`moonep_l01_nvshmem_getmem_bwire_trace-2ce8ee_b32_k8_nsys`, iter4. Ranks
r3 r12 r13 r14 (longest pull / shortest inter-node wire / median pull /
one-expert pull over NVLink). Time origin per rank = first dispatch wire event.

All-16-rank b32 numbers (ms; `nsys_sqlite` cache under
`$PSCRATCH/workspace/andrewy/figs_data/motivation/`): expert pull 0.04
(r4, no expert assigned) / 0.75 (r5 r14 r15, one expert over NVLink) /
2.8–8 (one or two experts from a lightly loaded off-node home) / 24.2–25.0
(r0 r3 r6 r7 r10 r11, one expert each from the hot home rank 12, seven
pullers sharing its NIC); GEMM 2.86–3.33 (1.07x); token wire NIC 11.2–18.6,
barrier at 19.0–19.7. The authentic a2a arm's recorded prefetch_ms agrees
with the twin within 2.1 ms on every rank at b16 and b32 (same plan, same
pulls: a2a b32 = 25.1 / 4.9 / 17.2 / 25.2 / 0.0 / 0.8 / 24.9 / 24.7 / 5.5 /
4.3 / 24.8 / 24.7 / 2.9 / 7.8 / 0.8 / 0.8 ms for r0..r15).

## Caveats (caption material)
- MoonEP's prefetch is exposed by design (upstream shares one comm stream
  between dispatch and prefetch — `moonep_overlap_shared` is the upstream
  order; the authentic arm serializes prefetch after the dispatch barrier).
- The expert-comm span is the getmem kernel + its quiet; NIC/NVLink lane
  is assigned from the plan (pull sets are homogeneous per rank on these
  routings). Its length is set by the hot home's egress fan-out, not by
  the puller's bytes (0 / 1 / 2 experts, 58.7 MB each).
- The exposed-wire twin is instrumented — never a latency arm.

## Knobs
`HEIGHT_FRAC` (0.27), `ARC_H` (9), `RED`, `ARROW`, `COL`, `PANELS`,
`ARROWS`, `--budget`, `--ranks`, `--moonep-arm` (bwire | a2a), `--rule`.
