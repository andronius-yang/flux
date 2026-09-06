# System overview diagram — brief and discussion (2026-09-05, REV 0.0)

Lane for the **main cross-column figure of the implementation / system
design section**. This file is the discussion record; the drawing spec
(`SPEC.md`), options page and generator come later, following the lane
conventions of `figs/main_perf`, `figs/ablation_cycling` and
`figs/motivation` (every aesthetic value a `[knob]`, rulings dated,
never-mix with earlier revisions).

## 1. What this figure is for (user brief, 2026-09-05)

- One figure that **introduces the whole system** and gives the reader a
  map of where the three mechanisms sit in an MoE layer's iteration.
- It **does not** explain any mechanism. Each has its own diagram in its
  own subsection:
  1. token comm/computation overlap (the merge/split of token traffic —
     Slipstream fused dispatch + msplit/fused-pack/bucket combine);
  2. expert placement + routing (pv2 placement, LocCap sender-local
     routing);
  3. expert-dispatch overlap (the intra-node expert swap, P2P weight
     exchange overlapped with the host planning gap / GEMM).
- Job of the overview: name the three, show *when in the iteration* and
  *on which resource* (host, NVLink, inter-node NIC, SM) each acts, and
  show how they hand off to each other. Nothing finer than a box per
  mechanism.

Names must match the ablation figure legend verbatim (`figs/ablation_cycling`
REV 0.3): "token comm comp overlap", "expert placement & routing",
"expert-dispatch overlap". If the paper text uses other section titles,
the ablation lane changes too — one vocabulary.

## 2. Ground truth to draw from (what one iteration actually does)

Per-iteration chain of the methodology arm (OURS s2 = always-solve +
overlapped swaps, handoff 27 §3; component facts from handoffs 20/25/26),
in issue order on the host:

| # | step | resource | timed bracket (SCHEMA rule 5) | mechanism it belongs to |
|---|---|---|---|---|
| 0 | gating-metadata exchange (initial only) | NIC | untimed (the one exception) | — |
| 1 | top-k routing all-gather | NIC | `plan_comm` | placement & routing |
| 2 | plan derivation on GPU: LocCap sender-local routing, replica choice, plan graphs (CUDA-graph replay) | SM (small) | `plan` | placement & routing |
| 3 | place bracket: placement re-solve (quiet always-solve) + swap decision (greedy pair+swap, host integer); early issue of the NVLink P2P weight exchange into the peer's symmetric staging | host + NVLink | `place` | expert-dispatch overlap |
| 4 | layer-0 fused dispatch + grouped GEMM: LB_UNION dispatch, wave-adapt 48, tiles spin on arrival; inter-node puts on the blocking wire | NIC + NVLink + SM | `e2e` (l0) | token comm comp overlap |
| 5 | layer-1 grouped GEMM + combine: msplit / fused-pack / bucket receiver, combine-idx kernel, late combine-meta overlap (≤16 MiB) | SM + NIC | `e2e` (l1) | token comm comp overlap |
| 6 | end-of-iteration fabric drain (join_w1 wait-all) | — | untimed gap | — |

Facts that constrain the picture (do not draw what the machinery does
not do):

- The swap's weight movement is fully hidden (handoff 25 §0.1): it sits
  in the host planning gap under early issue. The overview may show it
  as a bar *under* the planning bracket, not as a bar of its own on the
  critical path.
- Placement is re-solved every iteration but only the *intra-node* swap
  moves weights every iteration in the methodology arm; cross-node
  migration (s2-pv2) is an ablation arm. The overview should not
  suggest weights cross the NIC per step.
- There is **no schedule caching** across iterations (memory rule): every
  iteration re-plans. Drawing a "plan once, reuse" loop would be wrong.
- The plan is *not* overlapped with the previous iteration (plan_overlap
  removed, handoff 20 FINAL CANON). Drawing the plan bracket overlapped
  with the previous combine would be wrong.
- Inter-node token wire is the blocking put (wire-ordering hard rule);
  overlap with compute comes from tile-level spinning on arrival inside
  the GEMM, not from a non-blocking wire.

## 3. Format constraints

- `\begin{figure*}`, NSDI two-column: width 7.0 in (`\textwidth`), placed
  at `\linewidth`, exported at final size, fonts true points
  (`pdf.fonttype 42`) — same regime as `figs/main_perf/SPEC.md` §1.
- Height budget: not yet ruled. The main-perf lane holds 1/4 page incl.
  caption (2.25 in); a system diagram in a design section usually gets
  less. Proposed starting point: **1.6–1.9 in** tall, caption ≤ 3 lines.
- Palette: the muted main-figure palette (terracotta / sand / green /
  teal / wine / olive + steel-blue "Ours", no pink/purple, CVD-checked).
  Suggest one hue per mechanism, reused by that mechanism's own detailed
  figure later, so colour = mechanism across the whole paper.
- Fonts/stroke: inherit main-figure rulings (edge stroke, label sizes).
- Tooling: not decided. Options are a matplotlib/SVG generator (like the
  other lanes, keeps knobs in code) or a hand-drawn vector (draw.io /
  TikZ). Prior lanes are all generated; a schematic with text-heavy boxes
  is the one case where TikZ may be cleaner. **Open question for the
  user.**

## 4. Candidate compositions (to be turned into an options page)

**A. Timeline swim-lanes (one iteration, left to right).** Rows = resources
(host, NVLink/intra-node, NIC/inter-node, SMs). Boxes = the steps of §2,
coloured by mechanism, with the overlap relationships shown by vertical
alignment (swap copies under the plan gap; dispatch wire under the l0
GEMM; combine under the l1 GEMM). Three call-out labels name the
mechanisms and point to their sections. *Pro:* shows "when" and "where"
at once, matches how the ablation and motivation figures already think
(brackets, resources). *Con:* looks like a profile, not an architecture;
risk of readers expecting the boxes to be to scale.

**B. Layered stack (control plane over data plane).** Top band: the
per-iteration planner (routing all-gather → plan → place/swap decision),
bottom band: the fused layer-0/layer-1 kernels with token flow drawn as
arrows between GPUs of two nodes. The three mechanisms are three labelled
regions: the planner band (placement & routing), the NVLink weight
exchange arrow between two GPUs of one node (expert-dispatch overlap),
and the fused kernel boxes (token comm comp overlap). *Pro:* reads as a
system, cleanly separates "decide" from "move". *Con:* loses the
overlap-in-time story, which is the paper's point.

**C. Hybrid: two nodes drawn physically, one iteration's arrows numbered
in order.** A small cluster picture (2 nodes × 2 GPUs shown, "…" for the
rest) with numbered arrows 1–5 following §2, and a thin timeline strip
underneath showing the same numbers as brackets to convey the overlap.
*Pro:* both "where" and "when". *Con:* densest; 7 in wide helps but
1.8 in tall is tight for two registers.

Working recommendation: **A or C**. B is what a generic MoE-system paper
draws; the overlap story is what distinguishes this system, and the two
detailed comm/swap figures will themselves be timelines, so the overview
should set that frame.

## 5. Open questions for the user

1. Height budget (1/5 or 1/4 page incl. caption?) and whether the
   figure sits at the top of the design section or mid-section.
2. Vocabulary check: section titles for the three mechanisms in the
   current paper draft (must equal the ablation legend).
3. Tooling: generated (matplotlib/SVG) vs hand-drawn vector (TikZ /
   draw.io). Affects how the options page is produced.
4. Which scale is depicted: 2 nodes × 2 GPUs schematic, or 2 nodes × 4
   GPUs (Perlmutter-authentic)? Schematic favoured; the figure carries no
   numbers.
5. Whether the layer-1 combine is drawn as its own step or folded into
   "token comm comp overlap" (the ablation figure folds it).
6. Whether the initial one-shot placement (pv2 at setup, oracle basis)
   appears at all, e.g. as a greyed "setup" prelude, or is left to the
   placement section.

## 6. Non-goals (do not let these creep in)

- No merge/split token-traffic detail (its own figure).
- No placement/routing algorithm detail (replica counts, LocCap caps).
- No swap-decision detail (pair+swap heuristic, tau).
- No numbers, no measured durations, no speedups. Boxes are not to scale
  and the caption says so.
- No baseline (COMET/EPLB/NVSHMEM) in this figure; contrast lives in the
  motivation figure.

## 7. Next steps in this lane

1. User rulings on §5 → REV 0.1 of this brief.
2. Options page `overview_options.html` with A / B / C sketched at the
   real aspect ratio.
3. `SPEC.md` for the chosen option; generator or vector source; PDF/PNG.

## 8. The existing draw.io figure (`moe.drawio` / `moe.pdf`, uploaded 2026-09-05)

What it is: a dataflow schematic of one MoE layer on 2 nodes x 2 GPUs,
one token per GPU (A–D, colour = token), top-3 routing, 2 experts per
GPU (expert parallel), three phases left to right: Dispatch, Compute
(w1 -> Activation -> w2, the expert boxes drawn twice), Combine+Reduce
(three copies fan into one output token). Line pattern = physical path:
fine dotted (draw.io `dashPattern=0.7 1`, width 1) = intra-node NVLink,
dashed (`4 2`, width 1.2) = inter-node through the NIC. Local copies
(token to an expert on its own GPU) draw no line. Exported PDF is
224.9 x 133.9 pt = **3.12 x 1.86 in** (single-column size, aspect 1.68);
draw.io unit = 0.75 pt, so `fontSize=10` expert labels are 7.5 pt,
token letters 8.25 pt, "Activation" 4.5 pt (below the 6 pt floor).

Routing table it encodes (worth keeping — the imbalance is already in it):

| token (home) | experts hit | drawn lines (dispatch) |
|---|---|---|
| A (GPU0) | E0 local, E2 intra, E4 inter | 2 |
| B (GPU1) | E1 intra, E4 + E5 inter (same destination GPU) | 3 |
| C (GPU2) | E4 local, E0 inter, E6 intra | 2 |
| D (GPU3) | E6 + E7 local, E4 intra | 1 |

Expert 4 receives all four tokens (the hot expert); Expert 3 receives
none (the idle slot). Drawn edges today: 8 dispatch + 8 combine = 16.

### 8.1 How the three mechanisms land on this canvas

- **Expert placement & routing.** The idle Expert 3 slot on GPU1 is the
  natural home for a *replica of Expert 4*. With the replica, A's inter
  copy (GPU0 -> GPU2) and B's E4 copy become intra-node hops to GPU1: a
  dashed line turns dotted. That single visual (dashed -> dotted, hot
  expert with two boxes) is the whole introduction; the replica-count
  and LocCap cap detail stays in the placement section.
- **Expert-dispatch overlap.** A weight arrow (third line pattern, e.g.
  solid grey with an open head) from the Expert 4 box on GPU2 to the
  replica slot on GPU1, drawn *during* Dispatch (same column as the token
  lines), says "weights move on NVLink while tokens are in flight". It
  must stay intra-node (methodology arm swaps within a node only).
- **Token comm comp overlap.** Hardest on a dataflow canvas, because the
  phase columns say "sequential". Two candidate devices: (a) let the
  dispatch lines end *inside* the Compute column at staggered depths
  (early rows of the expert box already computing while later copies
  arrive), or (b) a thin time strip under the three phase labels showing
  Dispatch/Compute/Combine brackets overlapping instead of abutting. The
  merge/split itself (B's two copies to GPU2 sharing one wire crossing)
  is at most a hint: draw the two copies as one bundled dashed line that
  forks at the destination, no annotation.

### 8.2 Two tokens per rank (user's proposal)

Edge counts (non-local copies drawn, roughly 2/3 of copies):

| tokens/rank | top-k | copies | drawn dispatch lines | + combine |
|---|---|---|---|---|
| 1 (today) | 3 | 12 | 8 | 16 |
| 2 | 2 | 16 | ~11 | ~22 |
| 2 | 3 | 24 | ~16 | ~32 |

Why two tokens per rank helps: same-rank tokens to the same destination
GPU are the natural merge motif, and the second token per rank lets one
token pair show placement/routing (replica hop) while the other shows
plain traffic, without changing the top-k. Why it costs: the dispatch
column already carries 8 vertical runs across 4 GPU rows; ~11–16 runs
need a wider dispatch gutter, which the cross-column width supplies but
the vertical span does not. Recommendation: **2 tokens x top-2** (22
edges), and drop the second (w2) expert-box column, merging the two GEMMs
into one box with the activation implied, to buy the horizontal room.

### 8.3 Aspect problem for a cross-column figure

The canvas is 1.68:1; a `figure*` at 7.0 in wide and ~1.9 in tall is
3.7:1. Scaling the current canvas to 7 in gives 4.2 in tall — not
acceptable. Three ways to use the width:

1. **Before / after pair** — left panel = this figure (plain MoE, all
   traffic exposed), right panel = the same canvas with the three
   mechanisms drawn and labelled (1)(2)(3) with section refs. Each panel
   3.4 in wide -> 2.0 in tall at the current aspect, 8.2 pt expert
   labels. Slightly over budget in height; fixable by dropping the w2
   column (canvas 297 -> ~240 units wide, height unchanged) or one
   GPU row per node.
2. **Single wide canvas, two tokens per rank, mechanisms annotated in
   place** — widen the Dispatch gutter and the Combine gutter (that is
   where the width goes), keep 4 GPU rows; height stays ~185 units so
   the figure is ~1.9 in tall only if the width scale is 0.0103 in/unit,
   i.e. the canvas grows to ~680 units wide. Fonts then land at ~7.4 pt.
   Feasible, and it is the "one integrated diagram" the user asked for.
3. **Three same-base panels, one mechanism highlighted each** — cleanest
   introduction but 2.33 in per panel drives labels to ~5.6 pt; needs
   "E0" style labels and no w2 column. Kept as fallback.

Working recommendation: **option 2**, with option 1 as the alternative
if the annotations crowd the in-place canvas.

### 8.4 Small fixes to carry over regardless

- "Activation" at 4.5 pt is unreadable in print; either drop the second
  GEMM column (recommended) or raise to 6 pt minimum.
- Legend "Inter-node traffic routed through NICs" -> "Inter-node (NIC)"
  and "Intra-node Traffic" -> "Intra-node (NVLink)"; add the weight-move
  pattern once mechanism (2) is drawn.
- Token colours (yellow/red/blue/light-blue) are fine for tokens but
  clash with the mechanism-per-hue plan in §3; mechanisms should be
  labelled by number and outline, not by fill hue, on this canvas.

## 9. REV 0.1 — user rulings 2026-09-05 (after seeing §8)

Rulings, verbatim in intent:

1. **Three expert slots per GPU**: two resident experts plus one free
   replica slot. Show the hot expert replicated and shedding load.
2. **No before/after pair.** One canvas. The dispatch side must show the
   **cross-node merge** and the **pack-forward** (gateway forward).
3. **Combine side must show the row mechanism**: partial rows for the
   same token converge within the serving node and are **pre-reduced
   before the NIC crossing**, even with only two nodes.
4. **Expert swap = one arrow only**, meaning "adjusted per demand so GPU
   compute is more balanced". No swap-decision detail.

Mechanism facts the drawing must respect (SCHEMA rules 13, `incidence_remote`
definition; handoffs 25, 28):

- Dispatch under the node-dedup transport: **one wire row per (token,
  remote node)** — a token routed to two GPUs of one remote node crosses
  the NIC once, lands on a gateway GPU of that node (one of its serving
  GPUs) and is forwarded to the sibling over NVLink. Rows from one source
  bound for the same remote node are packed into one message
  (wave-pack). Cross-node merge = this packing; pack-forward = the
  gateway's NVLink forward.
- Combine is the mirror: partials for a remote-homed token produced on
  two GPUs of one node **converge on one GPU over NVLink, are pre-reduced
  there, and cross the NIC once** (conv rows -> prereduce -> one blocking
  put per destination). Local-node partials fold at the home GPU.
- The swap in the methodology arm is **intra-node** (NVLink P2P weight
  exchange between two GPUs of one node), issued early and hidden. It
  balances GPU compute *within* a node. Balance *across* nodes comes from
  the per-iteration placement re-solve (cross-node migration is the
  ablation arm s2-pv2, not the methodology arm). **Open question for the
  user (ruling 4 says "inter-node GPU compute"):** the swap arrow should
  join two GPUs of the same node; is that what was meant, or should the
  figure also imply cross-node rebalancing (which would be the re-solve,
  not the swap)?

### 9.1 Proposed routing table for the extended canvas (2 tokens/GPU, top-2)

Slots: GPU0 {E0, E1, free}, GPU1 {E2, E3, free}, GPU2 {E4, E5, free},
GPU3 {E6, E7, free}. Hot expert = E4 on GPU2. Replica E4' goes into a
free slot — **two candidate homes, user to pick**:

- (i) GPU3 (same node, ruling 1 read literally): sheds GPU2's compute
  onto GPU3; NIC traffic unchanged.
- (ii) GPU1 (other node): sheds GPU2's compute AND turns Node0's E4
  copies from dashed to dotted — this is the node-aware placement +
  sender-local routing win the paper claims. **Recommended.**

Table assumes (ii); under (i) swap the E4' rows to GPU3 and the A1/B2
lines become the ones that stay dashed.

| token (home) | expert 1 | expert 2 | dispatch path drawn | motif carried |
|---|---|---|---|---|
| A1 (GPU0) | E0 local | E4' GPU1 | dotted (was dashed to GPU2 pre-replica) | placement & routing |
| A2 (GPU0) | E1 local | E7 GPU3 | dashed to gateway GPU2, dotted forward to GPU3 | pack-forward |
| B1 (GPU1) | E4 GPU2 | E6 GPU3 | ONE dashed run GPU1->GPU2, dotted forward GPU2->GPU3 | cross-node merge (1 wire row for 2 GPUs) + pack-forward |
| B2 (GPU1) | E2 local | E5 GPU2 | packed into the same GPU1->Node1 message as B1 | cross-node merge (packing) |
| C1 (GPU2) | E4 local | E6 GPU3 | dotted | plain intra |
| C2 (GPU2) | E5 local | E1 GPU0 | dashed | plain inter |
| D1 (GPU3) | E6 local | E4 GPU2 | dotted | plain intra (hot expert still loaded) |
| D2 (GPU3) | E7 local | E3 GPU1 | dashed to gateway GPU1 (its owner) | plain inter |

Combine motifs (mirror of dispatch, same token letters so the reader
follows one token both ways):

- **B1'**: partials on GPU2 (E4) and GPU3 (E6) — dotted GPU3->GPU2
  converge, a small "Σ" pre-reduce box on GPU2, ONE dashed put to GPU1.
  This is ruling 3.
- **A2'**: partial on GPU3 (E7) only — one dashed put home (no
  pre-reduce needed; shows the plain path next to the merged one).
- **A1'**: E0 local + E4' on GPU1 — dotted home. The replica's benefit
  shows in both directions.
- Local-node partials (C, D) fold at the home GPU as today's fan-in
  trapezoid.

Loads after placement (rows per GPU, copies): GPU0 4, GPU1 5 (incl.
E4' serving A1, B... ) — to be tallied in the draft; the intent is GPU2
drops from the hottest to near the mean and no GPU idles.

Swap arrow (ruling 4): one double-headed arrow between the E5 box on
GPU2 and the E7 box on GPU3 (same node), labelled "swap per demand", in
the third line pattern (solid, open heads). Drawn once, no timeline.

### 9.2 Canvas plan

- Keep the four GPU rows and the three phase columns; drop the second
  (w2) expert column, one expert box per slot, activation implied.
- Width goes into the Dispatch gutter (needs ~11 vertical runs + the
  packed bundle) and the Combine gutter (converge + pre-reduce + put).
- Line patterns: dotted = intra-node NVLink, dashed = inter-node NIC,
  solid open-head = weight movement (swap / replica fill). Legend gets
  the third entry.
- Mechanism callouts: three numbered tags (1) placement & routing at the
  E4' slot, (2) token comm comp overlap at the packed bundle + Σ box,
  (3) expert-dispatch overlap at the swap arrow — the numbers, not hues,
  carry the mapping to the subsections.
- Target: 7.0 in wide, ~1.9–2.1 in tall; draw.io canvas ~680 x 190 units
  so that fonts land at 7–8 pt at export.

Next: user picks replica home (i)/(ii) and answers the swap-scope
question; then a first `moe_overview.drawio` draft is produced from
`moe.drawio` by extending the cell set (same ids/styles) for review.

## 10. REV 0.2 — ground truth = what the main-perf "1+2 + expert dispatch" arm ran (user ruling 2026-09-05: draw what ran, not what was documented)

Arm: `ours_l01_s2_swap_force_p2p_r2` (figs/main_perf row `ours12_dispatch`).
Definition: `sweeps/variants.py` `_OURS_ENV` + `_SWAP_ARGS` + `--swap_xport p2p
--swap_issue early --swap_tau_rows -1`. Driver: `test/python/moe_ag_scatter/
test_moe_ours_traffic.py`. Code-verified per-iteration sequence:

| # | what runs | where (code) | resource |
|---|---|---|---|
| 1 | all-gather of per-rank expert loads `d` (`plan_comm`) | driver plan_comm bracket | NIC |
| 2 | **swap lane** (`place`): D2H of `d`; per node pair heaviest<->lightest GPU, pick the one slot exchange that minimises the pair max; `tau=-1` => fires EVERY iteration; table transposition; w1+w2 exchanged over NVLink P2P (each rank cudaMemcpy's its slot into the peer's symmetric staging, landed-signal, copy in), issued EARLY on the movement stream. **No cross-node movement. No pv2 re-solve** — the driver's `if s2_swap: ... elif use_pv2:` skips the re-solve lane in this arm (driver :1370 / :1430). | `ours_swap.py`, driver :1368-1430 | host + NVLink |
| 3 | LocCap sender-local routing on the swapped tables (replica instance per copy, local-first under cap `(1+eps)·S·K`, eps 0.0625), plan graphs | `loccap_semantics.py`, `ours.py` | SM (small) |
| 4 | **layer-0 fused dispatch + grouped GEMM**, `a2av_hier_compress` + `FLUX_A2AV_LB_UNION`: intra-node copies go direct over NVLink (dedup per destination rank); inter-node = ONE union row per (token, destination node). Per (source node -> dest node) round the node's L union segments are concatenated and cut into L near-equal chunks; **relay rank k stages chunk k (NVLink gather from its node-mates) and wire-puts it, blocking, to the same-local-rank gateway on the target node; the gateway forwards each destination's subset over NVLink**. GEMM launched early; tiles gate on segment arrival (fused stage 2) and on per-slot weight signals (only the swapped slots' tiles spin, until the exchange lands). | `gemm_grouped_v2_ag_scatter.cc` :293-302, :478-483, :788 | NVLink + NIC + SM |
| 5 | **layer-1 grouped GEMM + Slipstream combine**: GEMM decomposed into destination-node ROW waves (msplit; remote nodes first, own node last); epilogue-fused pack writes the send panel with top-k coefficients folded; per wave, partials for tokens homed at rank (tn, lr) **converge over NVLink onto the local rank with the same lr (conv panel), a persistent pre-reduce kernel merges them per token, ONE blocking put per destination rank lands straight in the home's recv panel (no destination gateway hop)**; the bucketed receiver folds each token at its last lane's arrival. Own-node partials: per-rank chunks over NVLink, folded at home. | `gemm_grouped_v2_gather_rs.cc` :177-184, :592-600, :1130-1135 | SM + NVLink + NIC |
| 6 | end-of-iteration join / drain (untimed) | driver | — |

Setup (untimed, once per cell): pv2 placement on the oracle histogram —
EPLB-global replica counts capped at one instance per node, node-aware
affinity spread, within-node snake; `--redundant_per_rank 2` => each rank
holds `G/W + 2` slots (the figure's 2 + 1 free slot is the same idea at
schematic scale).

### 10.1 Deltas against §2 and §9 (these override)

- **Placement per iteration = intra-node swap only.** The global
  placement (replicas, node assignment) is fixed at setup. The
  "expert placement & routing" tag on the figure therefore means: replica
  instances placed node-aware at setup + LocCap local-first routing every
  iteration. The "expert-dispatch overlap" tag = the every-iteration
  intra-node swap, exchange hidden. Ruling 4's arrow joins two GPUs of ONE
  node; nothing rebalances across nodes per iteration in this arm.
- **Dispatch inter-node path has two NVLink hops around one NIC hop**:
  egress relay (node-level pack, cut into L chunks so both NICs of the
  node carry an equal share) -> NIC to the same-local-rank gateway ->
  ingress forward. Ruling 2's "cross-node merge" = the union row per
  (token, node) plus that node-level pack; "pack-forward" = the gateway's
  forward. In the 2x2 schematic the same-local-rank pairs are GPU0<->GPU2
  and GPU1<->GPU3.
- **Combine converge target is the same-local-rank GPU, for every remote
  partial, even a single one.** A token homed on GPU1 (lr 1) and served
  anywhere in Node1 has its partials converge on GPU3 (lr 1), pre-reduce
  there, one put to GPU1. A token homed on GPU0 converges on GPU2. The put
  lands directly in the home's recv panel.
- **Combine overlaps the l1 GEMM by destination waves**: the rows bound
  for the remote node are computed first and leave while the own-node
  rows are still computing. This is the combine-side half of "token comm
  comp overlap"; the dispatch-side half is tiles gating on segment arrival.

### 10.2 Routing table, corrected (2 tokens/GPU, top-2, replica E4' on GPU1)

| token (home) | copies | dispatch path as run | combine path as run |
|---|---|---|---|
| A1 (GPU0) | E0 local, E4' GPU1 | dotted GPU0->GPU1 | dotted GPU1->GPU0, fold at home |
| A2 (GPU0) | E1 local, E7 GPU3 | union row in Node0's pack to Node1; chunk relayed via GPU0 or GPU1 NIC to its same-lr gateway; forward to GPU3 | partial on GPU3 -> converge on GPU2 (lr 0) -> Σ -> one dashed put to GPU0 |
| B1 (GPU1) | E4 GPU2, E6 GPU3 | ONE union row for Node1 (merge); forwarded to both GPU2 and GPU3 by the gateway | partials on GPU2 + GPU3 -> converge on GPU3 (lr 1) -> Σ -> one dashed put to GPU1 (ruling 3 motif) |
| B2 (GPU1) | E2 local, E5 GPU2 | packed with A2/B1 into Node0's message to Node1 | partial on GPU2 -> converge on GPU3 (lr 1) -> rides the same put batch to GPU1 as B1' (separate row, no reduce with B1) |
| C1 (GPU2) | E4 local, E6 GPU3 | dotted | dotted back |
| C2 (GPU2) | E5 local, E1 GPU0 | Node1's pack to Node0 -> gateway GPU0/GPU1 -> forward | partial on GPU0 -> converge on GPU0 (lr 0, itself) -> put to GPU2 |
| D1 (GPU3) | E6 local, E4 GPU2 | dotted | dotted back |
| D2 (GPU3) | E7 local, E3 GPU1 | Node1's pack to Node0 -> forward to GPU1 | partial on GPU1 -> converge on GPU1 (lr 1, itself) -> put to GPU3 |

Drawing rule that follows: draw the Node0 -> Node1 traffic as ONE packed
bundle leaving Node0 that splits into two dashed NIC runs (GPU0->GPU2,
GPU1->GPU3), each fanning out dotted at the gateway. Mirror on the
combine side: dotted converge runs into the same-lr GPU, a Σ box, one
dashed run home. Two tokens per rank is what makes the bundle visible.

### 10.3 Flag for the main_perf lane (not fixed here)

`figs/main_perf/figure_src.md` labels row `ours12_dispatch` "re-solve
every iteration, no expert movement". The driver shows the swap arm
skips the pv2 re-solve (`elif use_pv2`) and DOES move weights (intra-node
NVLink swaps every iteration under `tau=-1`). The label should read
"setup placement + overlapped intra-node expert swaps every iteration".

## 11. REV 0.2 draft (2026-09-05) — `moe_overview.drawio`

User rulings folded in: replica E4' on GPU1; swap arrow intra-node
(E5 on GPU2 <-> E6 on GPU3); components must be professional, succinct,
research-grade.

Files: `make_overview_drawio.py` (generator; all geometry in its KNOBS
block, routing tables `IN/OUT/PART` + the `lane_edge` calls), output
`moe_overview.drawio` (uncompressed XML, opens in draw.io, every element
hand-editable), `moe_overview_preview.png/.pdf` (matplotlib render of the
same primitives — review only; the draw.io export is the figure).
Run with `$PSCRATCH/conda_envs/andrewy-comet/bin/python
figs/system_overview/make_overview_drawio.py`.

Routing drawn (2 tokens per GPU, top-2; every expert stack receives at
most ONE token by wire so no line crosses a box):

| token | experts | dispatch drawn | combine drawn |
|---|---|---|---|
| A1 | E0, E4' (GPU1) | dotted to GPU1 | dotted back |
| A2 | E1, E7 (GPU3) | dashed to gateway GPU2, dotted forward to GPU3 | partial converges on GPU2 (lr 0), Σ, dashed home |
| B1 | E4, E5 (both GPU2) | dashed to gateway GPU3, forwarded to GPU2 (one wire row for two experts) | both partials converge on GPU3 (lr 1), Σ, ONE dashed row home (B1 receives a single pre-reduced partial) |
| B2 | E2, E4' | local | local |
| C1 | E4, E6 (GPU3) | dotted | dotted back |
| C2 | E5, E1 (GPU0) | dashed to gateway GPU0 | dashed home |
| D1 | E6, E7 | local | local |
| D2 | E7, E3 (GPU1) | dashed to gateway GPU1 | dashed home |

Design decisions in the draft (each a knob):
- Free slots: empty dashed boxes, no text. Replica E4' = green bold
  outline, badge ①. Badges are 9-unit black discs with a white ring,
  straddling the element they mark (① E4' right edge, ② below gateway
  GPU2 and above Σ on GPU3, ③ on the swap bracket).
- Swap = ⊐-shaped double-headed open arrow hugging the right side of the
  expert column (never crosses the free slot); setup replica fill =
  single open arrow E4 -> E4' across the node boundary; both grey.
- Line colour = darker shade of the token fill (A gold, B coral, C
  indigo, D steel blue); dotted 0.7/1 = NVLink, dashed 4/2 = NIC.
- Gateway = small grey square on every GPU (each GPU is the same-lr
  gateway of its peer); Σ only on Node1 (the direction with merged
  partials), Node0's single partials go straight (pass-through).
- Legend: five samples on one line + the three mechanism names on a
  second line (`legt_tags`; move to the caption if height must drop).
- Canvas 654 x 215 units = 6.81 x 2.24 in at draw.io scale 1 (fonts:
  experts 6.75 pt, tokens 6 pt, legend 6 pt). Export at 100 % and place
  at `\linewidth` in `figure*`.

Not drawn (recorded, deliberate): the egress relay hop (with 2 wire rows
per node the LB_UNION chunk cut falls on the segment boundary, so each
source relays its own chunk); the destination-wave order of the l1
GEMM; the per-slot weight gate.

## 12. User review of the REV 0.2 draft (2026-09-05, from draw.io) — assessment

Suggestions: (1) a visible "travel once" notion on dispatch AND combine;
(2) LocCap reroute drawn as grey "would-take" path + actual path with a
highlight ring; (3) pre-combine shown explicitly: partials travel
intra-node, pre-reduce, cross the wire exactly once. Use the gutters
between expert ingress/egress stacks and the home tokens.

Assessment: all three accepted. Correction they expose: the draft's B1
had both experts on GPU2 and drew TWO forwards out of gateway GPU3;
the code does one forward per destination rank and aliases the row to
both experts locally — so the draft showed rank-dedup, drawn wrong.

Plan (REV 0.3, pending user confirmation of the routing):
- Routing: A1 E0+E4'(GPU1) [reroute, ghost to E4@GPU2]; A2 E5@GPU2 +
  E7@GPU3 [one NIC row -> GW2, consumed at E5, forwarded to E7]; B1
  E4@GPU2 + E6@GPU3 [mirror via GW3]; B2 E2+E4' local; C1 E4+E5 local;
  D1 E6+E7 local; C2 E5 + E1@GPU0; D2 E7 + E3@GPU1. E4 demand = 4 (2 at
  E4', 2 at E4). Node1 loses its plain intra-node dispatch line (every
  Node1 expert already carries one wire token) — accepted.
- "Travel once": exactly one token glyph per NIC crossing, placed on the
  dashed line where it straddles the node boundary (dispatch gutter and
  combine gutter); gateway square = the visible fan-out point (one dashed
  in, one short dotted to own stack, one dotted forward out).
- Ghost path: grey dashed A1 -> grey-outlined ghost slot leftmost in E4's
  stack on GPU2; badge ① moves to the fork after A1's home token with a
  thin highlight ring; dispatch side only.
- Pre-combine: Σ boxes move left next to the egress stacks; each Σ takes
  one own-GPU partial (short dotted) + one sibling-GPU partial (dotted
  converge lane) and emits one dashed row carrying the in-flight glyph.
- Canvas, legend, badges ②③ unchanged.

## 13. REV 0.3 implemented (2026-09-05) — routing confirmed by the user

`make_overview_drawio.py` rewritten; `moe_overview.drawio` + preview
regenerated. What changed on the canvas (all knobs):

- Routing per §12 (A2 = E5@GPU2 + E7@GPU3 via gateway GPU2; B1 = E4@GPU2
  + E6@GPU3 via gateway GPU3; C1/D1 local; C2/D2 single NIC rows).
- "Travel once": one in-flight token glyph per NIC crossing, straddling
  the Node0/Node1 boundary on its lane — dispatch A2, B1, C2, D2 and
  combine A2', B1', C2', D2'; a grey ghost "A1" glyph marks the crossing
  the replica avoided. Gateways show one dashed line in, one short dotted
  consume, one dotted forward.
- LocCap reroute: grey dashed "route w/o replica" from A1 to a grey
  ghost slot on Expert 4's row (GPU2); black ring around the fork + badge
  ①. Legend gained "route w/o replica".
- Pre-combine: Σ boxes next to the egress stacks (GPU2 x=396, GPU3
  x=414); own-GPU partial enters the left edge, the sibling-GPU partial
  enters from below (Σ2) / above (Σ3); one dashed row leaves with its
  glyph. A2 and B1 each receive a single pre-reduced partial at home.
- Home partial stacks: wire-arriving partial is now leftmost (fixes a
  REV 0.2 defect where the line ran under the first box).
- Badges: ① at the fork ring, ② below gateway GPU2 and right-below Σ2,
  ③ on the swap bracket. Canvas unchanged: 654 x 215 units (6.81 x 2.24 in).

## 14. User's V3 canvas (`user_sys.drawio`, 2026-09-06) — token-block edit

The user's heavily edited version: paths removed, experts collapsed to one
"Experts" box per phase per GPU, three grey section panels (§3.1 placement
+ routing over the compute column, §3.3 dispatch / combine overlap panels,
§3.4 three-dim scheduling strip at the bottom), GPU0's source token
replaced by five 5x5 blocks. Requested edit (token blocks only, nothing
else moved): five blocks per rank at the source, histogram-style token
blocks in front of each expert inside the §3.1 panel, mirrored on the
combine side.

Delivered as `user_sys_tokens.drawio` (original left untouched):
- Source rows for GPU1/2/3: 5 x (5x5) blocks at x=63.5, y=row+20, same
  group/style as GPU0's; the old single B/C/D tokens removed.
- Per-expert histograms: one group per expert box, rows A/B/C/D top to
  bottom (fixed row order so a colour sits at the same height on every
  GPU), bar length = tokens from that source rank. Dispatch bars are
  right-aligned 1 unit before the expert box (x <= 150, panel left edge
  132.3), combine bars left-aligned 1 unit after it (x >= 253.5, panel
  right edge 271.7). Rows centred in the 30-unit box (y = box + 5 .. + 25).
- Counts (A,B,C,D): GPU0 2,3,2,1 = 8 (hot); GPU1 1,1,0,1 = 3; GPU2
  1,1,3,1 = 6; GPU3 1,0,0,2 = 3. Each source rank sums to 5, total 20.
  Zero-count rows leave a gap by design.
- The lettered A/B/C/D destination tokens are kept: with unlabeled source
  blocks they are the only colour-to-rank key in the figure.
`user_sys_tokens_preview.png` is a geometry-only check render.
