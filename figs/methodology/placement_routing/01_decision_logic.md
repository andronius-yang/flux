# Placement & routing — the decision logic to diagram (2026-09-05)

Verified against the arm the main-perf figure plots (`ours_l01_s1_pv2_r2`:
`python/flux/testing/placement_v2.py` for placement,
`loccap_route_sl` in `python/flux/testing/placelambda_gpu.py` for routing).
This file is the logical description; implementation detail stays out.

Notation: G experts, NN nodes, L GPUs per node, R = NN*L GPUs.
Oracle input = `hist[node, expert]`: how many top-k demands each node
raised for each expert in the previous window. `load[g]` = its sum over
nodes. Slots: every GPU holds G/R "home" experts plus 2 spare slots
(r2), so there are 2*R extra copies to hand out.

## A. Placement (once, at setup, from the oracle)

Three questions, answered one after another — not interleaved per copy.

**A1. How many copies of each expert?** (the only place "hottest" and
"divide the load" appear)

    c[g] = 1 for all g
    repeat 2*R times:
        pick g maximizing load[g] / c[g]      # per-COPY load, not raw load
              (ties: lower expert id; skip experts already at NN copies)
        c[g] += 1                             # its per-copy load becomes load[g]/c[g]
    share[g] = load[g] / c[g]                 # the expected balanced load of one copy

Rules the diagram must respect: the pick is by per-copy load, so an
expert stops looking hot as soon as it is split; an expert may never
have two copies on one node (cap c <= NN) because the transport
deduplicates per node — a second in-node copy buys no wire.

**A2. Which node gets each copy?** (locality, not load)

    for experts in descending share (ties: lower id):
        for each of its c[g] copies:
            node = the node with the LARGEST hist[node, g]
                   among nodes with a free slot that do not already host g
                   (ties: lower accumulated node share, then lower node id)
            free[node] -= 1 ; nodeshare[node] += share[g]
    leftover slots (from copies that found no distinct node): filled with
    the highest-share experts not yet on that node.

So "which node?" = the node that demanded this expert the most in the
oracle. Node load only breaks ties; it never overrides affinity.

**A3. Which GPU inside the node?** (compute balance only; wire-neutral)

    per node: sort its copies by share descending,
              deal them across the L GPUs in a snake
              (GPU 0,1,..,L-1, L-1,..,1,0, 0,1,...)

Not "the lowest-loaded GPU per copy": a snake deal over the sorted list.
Any GPU in the node is equivalent for the wire, so this stage only
levels the expected GEMM load. Every GPU in the node sees the copy as
"home node".

## B. Routing (every iteration, from the current batch)

The binding quantity is **not** an expert's balanced share. It is a
per-GPU compute cap:

    cap = (1 + eps) * S * K rows per GPU,  eps = 1/16

i.e. each GPU may take at most 6.25% more rows than a perfectly even
split of the batch. The expected balanced share from A1 is never
consulted by the router. Each source GPU decides, for its own tokens'
(token, expert) demands, which copy serves each — in three locality
tiers, all computed from the all-gathered demand histogram, no
negotiation:

**B1. Own GPU.** All demand for experts the source hosts itself goes to
itself, scaled down proportionally only if it would exceed the cap.

**B2. Own node.** Remaining demand for experts hosted elsewhere in the
node: the node's total demand for expert g is split across the node's
GPUs hosting g in proportion to their remaining capacity, each GPU's
intake capped at its remaining capacity (a node-local water-fill;
sources fill in GPU order). Zero wire bytes so far.

**B3. Remote nodes — fewest nodes per token, not an even spread.**
Each remote GPU's remaining capacity is pre-partitioned into shares per
source GPU, proportional to how much leftover demand that source has
for experts hosted there (sources that need a GPU more get more of it).
Then, per token, the source picks the ONE remote node that covers the
most of that token's still-unserved experts using nodes where it still
holds share (ties: prefer home node, then lower id), sends all those
experts to that node (within the node: the hosting GPU where the source
has the most share), and repeats for what is left (3 rounds).

**B4. Forced.** Demand that no share can absorb goes to the least-loaded
hosting GPU and is counted; caps are soft only for this residue.

## C. Verdict on the argument as stated

| your statement | holds? | correction |
|---|---|---|
| pick hottest expert, replicate it | partly | "hottest" = largest per-copy load load/c, so the loop naturally spreads copies; cap one copy per node |
| pick a node it is not on — which node? | yes | the node with the highest oracle demand for that expert (affinity), tie → lower accumulated share |
| place on a rank — which rank? lowest weighted? | no | rank chosen last, per node, by dealing the node's copies (sorted by share) in a snake across its GPUs; wire-neutral, compute balance only |
| divide and adjust expected load | yes | per-copy load = load/c; it drives A1 picks and A2 tie-breaks, and is never used by routing |
| interleaved loop (replicate → node → rank → adjust) | no | three sequential decisions: all counts first, then all node choices in share order, then ranks |
| routing: given an expert's expected balanced load ... | no | the router's constraint is a per-GPU compute cap (1+eps)·S·K over all experts, not a per-expert quota |
| local nodes rush to fill that quota | yes | own GPU first, then own node, each up to the cap |
| distribute the rest evenly across remote replicas | no | per token, the fewest remote nodes that cover its remaining experts; capacity shares are demand-proportional, not even. Evenness would raise token-node incidence, which is the objective being minimized |

The strongest single line for the figure: **placement decides where
copies live by who asked for them; routing decides which copy serves a
token by how few nodes it has to touch, under an even-compute cap.**
