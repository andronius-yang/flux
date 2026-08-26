# Handoff 21 — Combine-side transport TODOs (s1 + s2)

Status: **proposed 2026-08-25 pm** (user-approved TODO order). Both tickets
target the same measured mechanism; both apply to s1 and s2 identically
(the combine transport is scenario-independent — s2 adds movement on top).

## The mechanism (nsys-established, 16n qwen b64, capsule 183714 pair)

The NVSHMEM host proxy is a PER-RANK serial resource (~7.6 GB/s drain on
CXI; blocking puts complete in FIFO cohorts at drain instants). Our
combine's hierarchical MERGE (prered: one row per token per source node)
minimizes bytes (~462 MB/node vs llc's ~600) but the per-rank SEND duty
follows expert hosting, so placement skew concentrated 272/121/53/15 MB
onto the four local proxies — the node drains at the pace of ONE proxy
(35.8 ms wire) while llc's blunter per-rank sends (~150 MB/rank spread)
drain in 25.0 ms. Receiver-side, the final-lane wait (~10 ms) is
arrival-limited by that slowest sender proxy. MERGE-CONCENTRATION
TRADEOFF: merged-but-serial loses to unmerged-but-parallel.

r2 (replica parity, canonical since 2026-08-25) splits HOT-EXPERT load
across instances and recovers most of the balance where replicas exist —
but qwen@16n-class shapes (G = W·epn slot-full at r0; single hot experts
irreducible at whole-expert granularity, 97% of the spread survives the
provably optimal pairing) and any future drift need TRANSPORT-level
fixes. Placement permutation (wirebal) is proven empty (handoff/report
ledger); these two tickets are the levers.

## TODO 1 (first): combine wave-panel redistribution across local proxies

Distribute the merged per-destination wave panels across the L local
ranks' proxies instead of letting send duty follow expert hosting:
dest-node wave ns -> sending rank lr = ns mod L (or plan-computed
balanced assignment from the known per-wave byte counts — the host
already builds per-iteration wave tables msplit_host_/msplit_node_order_,
gemm_grouped_v2_gather_rs.cc ~:2839-2851). Requires: an intra-node
NVLink hop moving each wave's pre-reduced panel to its sender rank
(CE-copy, overlappable under GEMM1 waves) + signal bookkeeping so the
receiver's per-source-node lane semantics are unchanged (additive
signals already support multiple contributors). Expected: node outbound
462 MB drains through 4 proxies ~= 15 ms vs measured 35.8 (bound), even
with ZERO placement freedom. Works identically under s2 (movement does
not touch the combine wire path). Never-mix: new wire schedule -> new
binary tag.

## TODO 2: dispatch CE-parcel / union closure (design complete, unbuilt)

Sender orders each destination-NODE parcel [excl(lr0)|...|excl(lrL-1)|
shared-tail] so the gateway forwards per-dest CONTIGUOUS ranges with
copy engines only (keeps the :4007 no-SM-gather exemption); recv drops
from L*U to Sum(excl)+L*shared. FULL DESIGN + the blocking assumption
(one-cumsum consumer identity, sort_util.cu:675 + ATen twin :2191-2231;
4 coordinated sites across sender pack scan / boundary formulas / uc
contract / plan side) in 21_assets/cpp_h4_h1_scoping.md. GATE before
building: measure the shared-multiplicity histogram from a captured
trace (if shared multiplicity ~2 dominates, the L+1-group variant saves
only ~10-15% — decide variant by data). Grows mandatory toward 32n
(union surplus -> ~Lx).

## Order + verdict criteria

1 then 2 (user-approved). Ticket 1 verdict cell: 16n qwen b64 combine
wire span (nsys) < 20 ms and l1 gap vs llc closed to <= 0. Ticket 2
verdict: recv/needed ratio at 16n b64 -> ~1.0-1.3 with l0 unchanged or
better. Both must pass random-payload gates (rule 6) and the b1-b8
small-budget range (the 2026-08-25 wedge taught: test small budgets at
scale FIRST).

## Context pointers

- Capsules: quad/ledger twins (20260826-0122xx..0153xx), nsys pair
  (20260825-183714), record baselines (20260825-182557/183241).
- Related open defects (NOT these tickets): dyn receiver livelock
  (dropped; RCA in campaign notes), 16n first-touch wedge / PSCRATCH
  degradation ops class (canary-first rule).
