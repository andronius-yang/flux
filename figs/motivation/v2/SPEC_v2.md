# Motivation figure — SPEC v2 (2026-09-04, postdoc edits via lanes_v2_postdoc.drawio)

Source of rulings: the postdoc's edited XML (`lanes_v2_postdoc.drawio`) +
the user's written instructions of 2026-09-04. Generator: `build_v2_lanes.py`
(inputs: phase JSONs of capsules 20260904-123815 and 20260904-132050).
Outputs: `lanes_v2.svg` (points), `lanes_v2.drawio` (editable twin, layers
background / bars / glyphs / axes / labels), `lanes_v2.png`, `lanes_v2_ranks.csv`.

## Format
- Cross-column NSDI figure: width = \textwidth = 504 pt (7.0 in); height ≈ 25 %
  of width = 126 pt (knob `HEIGHT_FRAC`). Single light look, Helvetica.
- Three subdiagrams side by side, LEFT TO RIGHT in this order, titles on top
  exactly as in the XML:
  1. "Computation Imbalance + Communication Imbalance" — NVSHMEM a2av + GEMM
  2. "Token Comm. Balanced + Expert Comp. Imbalance" — COMET
  3. "Expert Comp. Balanced → Comm. Imbalance" — EPLB (exposed dispatch wire,
     ring-order placement)
  Equal widths, a gap between subdiagrams; the lane rows are vertically
  aligned across subdiagrams but every subdiagram has its own lines (no
  continuous line across the gap).
- Per rank three lanes, top to bottom: NIC RDMA, NVLink, GPU (labels once, at
  the left of the first rank row, as in the XML). Lanes are thicker than v1
  (~5 pt) and near-adjacent; 4 ranks per subdiagram (v1.1 rule; ids omitted).
- No rank ids, no row/byte stats. The ONLY numbers: per-subdiagram ms axis at
  the bottom, and the placement span in ms next to each EPLB placement bar.

## Color = task (not resource)
- blue `#2a78d6` Token Comm. (dispatch puts / fetches on NIC, P2P copies on NVLink)
- green `#1baf7a` Expert Comm. (one-shot placement puts, NIC and NVLink)
- yellow `#eda100` Expert Comp. (expert GEMM on GPU) — the XML used orange
  `#eb6834`; the written instruction says yellow, which wins (knob `COL`).
- `||` barrier release (black), un-overlapped variants only.

## Background resource lines (bottom layer)
Darker than v1 (`#7d8289`), one dash pattern per resource so the resource is
readable even where a bar hides the line: NIC RDMA solid, NVLink dashed,
GPU dotted. Drawn first so bars overlap and hide them. A pattern ledger at
the bottom names the three resources; the color legend sits beside it.

## Scales
- No unified scale. Each subdiagram's dispatch→GEMM timeline gets its own
  0..T axis (T = longest of its four ranks), so each imbalance fills its panel.
- EPLB subdiagram: the expert-comm (placement) block takes 2/5 of the
  subdiagram width (relative scale, longest rank fills it, ms label per bar);
  the remaining 3/5 is the regular dispatch→GEMM timeline with its own axis.
- Time origin per rank = its first dispatch wire event (prep excluded).

## Data (recommended scenario, 2026-09-04)
K2 mmlu/professional_law layer 18, b32, middle timed window: ring + COMET
from capsule 20260904-123815; EPLB from the ring-order placement cell of
capsule 20260904-132050. Ranks per panel (v1.1 rule): ring r7 r11 r12 r15;
COMET r4 r8 r12 r15; EPLB r7 r11 r12 r13 (see lanes_v2_ranks.csv).

## Knobs
`HEIGHT_FRAC` (0.25), `PLACE_FRAC` (0.4), `COL`, `RANKS` (4), `--budget`,
`--rule`, `--eplb-arm` (ringplace | bwire).
