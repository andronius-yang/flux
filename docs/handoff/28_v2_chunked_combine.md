# Handoff 28 — v2 chunked/pieces combine: why overlap-without-rereads
# lost to the wave dial (2026-08-29 night → 08-30 morning)

Campaign: implement and test handoff 26 §3's "v2" combine (GEMM<->wire
overlap WITHOUT the msplit weight-reread tax) and compare l1 + total vs
the canon arms at 4n AND 8n, b{1,2,4,16,64}, K2 + Qwen. Built in
worktree branch `v2-combine` (since merged: pv2 @ 9a582e5, pushed to
fork; worktree pruned, branch kept). Code = default-OFF knobs
(`FLUX_A2AV_RS_CHUNK_E`, `FLUX_A2AV_RS_PIECES`, tags `..._CHUNK_TAG`,
`..._PIECES_TAG`). **VERDICT (user ruling 8/30): the ORIGINAL canon
strategy stands** — wave-adapt 48 keeps the record at every measured
cell at both scales; everything here is gated ablation infrastructure
plus a set of measured negative results. Results ledger with full
tables: `docs/worknotes/v2_chunked_combine_design.md`. Memory:
v2-chunked-combine-campaign.

## 1. Starting model (handoff 26) and what this campaign corrected

Handoff 26 modeled the wave path's cost as (n_waves-1) full HBM weight
passes (~1.5 ms at 4n K2) and proposed chunk-ordered sub-problems +
progressive per-(chunk, wave) release as a strict win. Three model
errors surfaced when we measured:

1. **The reread is NOT ~1.5 ms of exposed time.** nsys l1-GEMM spans
   (capsule 20260830-015422, K2 b16 4n): msp0 2.32 / chunk 3.44 / wa0
   3.87 ms. The TRUE marginal of HBM-vs-L2 rereads is wa0-chunk =
   **0.43 ms** — the extra passes largely hide under real MMA once
   budgets carry real flops. The bandwidth model triple-counted.
2. **The sub-problem SPLIT itself is the bigger tax.** chunk-msp0 =
   +1.12 ms = tile padding (~0.5: n_waves x E padded 128-row tiles) +
   re-walking the panels from L2 (~0.46, 3 x 764 MB at L2 bandwidth,
   cached != free). At b1 the "reread" cost handoff 26 attributed to
   HBM passes is mostly PADDING FLOPS (~104 padded tiles ~1.9 ms).
3. **The overlap is worth more than everything above.** wa0 vs msp0:
   0.94 (b16) → 4.26 ms (b64) at 4n K2; 2.07 → 7.26 ms at 8n. Any
   scheme that trades full overlap for weight locality starts in a
   hole deeper than the tax it removes.

## 2. Attempt M1 — chunk-ordered waves (FLUX_A2AV_RS_CHUNK_E)

Same (wave, expert) sub-problems, emitted expert-chunk-outer/
wave-inner via a problem->group map through the cascade (prob_eid +
prob_group_map; ~100-line diff). Gates green (checked twins,
per-iteration torch-reference output validation + payload probe).

**Why it did not work:** M1 keeps the wave flags' *semantics* but under
chunk order they all fire near GEMM end — the wire serializes after the
GEMM (collapse timing) while the split's padding remains. So M1 = the
collapse's comm with MORE compute than the collapse: it lost to wa0 at
every budget (K2 b16 8.94 vs 7.00) and to msp0 by the split residual.
Its value was diagnostic: it isolated the reread marginal (0.43) from
the split residual (1.12) and proved L2 reuse works on hardware (the
reuse-failed signature, +2.4, was excluded). Capsules
20260830-013149 (K2), -014319 (Qwen), -015422 (nsys).

## 3. Attempt M2 — no-split + progressive piece release
## (FLUX_A2AV_RS_PIECES)

The M1 data forced a redesign: drop the split entirely. One full-row
problem per expert (no padding, ~1 weight pass by construction),
per-expert-chunk cascade flags (the existing per-problem counters with
group = chunk — zero new GEMM kernel code), then a piece-granular comm
chain: pack piece relay -> piece-sliced conv puts -> per-(tn, piece)
prereduce -> one blocking wire put per (dest, piece) at equal
sender/dest offsets -> bucket P-slot lane waits. Enablers that are
correct and retained:

- **Two-sided ready-piece plan order**: wire rows renumber
  (seg, ready_piece, token), ready_piece = max contributor chunk —
  deterministic from splits, derived independently on both sides (no
  new exchange). P-pass warp scans; red_flags = piece+1; P=1 bitwise
  legacy.
- **Per-(lane, piece) epoch-SET signal slots** (depth-8 dedicated
  tensors): keeps the run_id trust contract — no signal-ADD, no
  resets, wire rule 6a untouched (every put blocking).
- **Piece-outer enqueue/visit order** everywhere (conv ladder,
  prereduce): tn-outer would head-of-line-park every later dest's
  early pieces behind the first dest's LAST piece on the single conv
  stream. (Caught in review, not on hardware.)

Gates green FIRST EXECUTION at 4n and 8n, both models, per-iteration
reference checks + payload probe (capsules 20260830-0227xx, -053349).
The implementation is believed correct; the losses below are physics.

## 4. Why M2 did not work — three independent walls

l1_ms, K2 (capsules 20260830-023028 4n, -053552 8n):

| scale | b | wa0 | canon | msp0 | pieces P4 | P2 |
|---|---|---|---|---|---|---|
| 4n | 1 | 2.82 | **1.61** | 1.67 | 2.15 | 1.75 |
| 4n | 16 | **7.06** | 7.19 | 8.15 | 7.44 | 7.35 |
| 4n | 64 | **24.53** | 25.05 | 29.87 | 25.40 | 26.45 |
| 8n | 1 | 3.35 | 2.28 | **2.18** | 4.96 | — |
| 8n | 16 | **9.21** | 9.34 | 11.28 | 11.43 | — |
| 8n | 64 | **32.34** | 33.10 | 39.60 | 37.38 | — |

**Wall 1 — last-contributor skew (4n, the dedup's price).** A compress
wire row (one per (dest token, tn)) is sendable only when its LAST
contributing expert chunk finishes: P(ready by chunk c) ~ (c/n_chunks)^k
with k ~ 4 node-hits/token at K2 — most deduped bytes mature in the
final chunks, so early pieces carry little mass. Pieces recovered
70-84% of the wave overlap (msp0 8.15 -> 7.44 at b16; 29.87 -> 25.40 at
b64) but the exposed tail costs more than the reread+padding the waves
pay. This is intrinsic to keeping the dedup: full early release of a
destination REQUIRES computing that destination everywhere first —
which is exactly the wave schedule and its reread.

**Wall 2 — put fragmentation (8n, pre-registered by the user).** Piece
puts multiply the serial ~120 us/put CXI proxy constant by
(NN-1) x P: at 8n P4 that is 28 blocking wire puts (~3.4 ms of
constants) + 196 piece conv puts. Pieces fell behind even the COLLAPSE
at 8n b16 (11.43 vs 11.28) and its b1 rent exploded (+2.8 vs msp0).
P must shrink with NN toward the P=1 collapse fixed point — at which
point the mechanism buys nothing the wave-adapt dial doesn't already.

**Wall 3 — nothing to save on small-weight models.** Qwen (12.6 MB
panels): the reread the whole design exists to remove is ~0.15 ms, so
pieces is pure rent at every budget (b16 6.42-6.70 vs canon 5.83,
capsule -024455). Any engage rule collapses to "pieces off for
Qwen-class models" — the same byte-rule shape as wave-adapt, guarding
a mechanism that then never engages at either measured scale.

Also confirmed: the piece machinery's floor rent at b1 (+0.08 P2 /
+0.48 P4 at 4n) — the user's pre-registration that splitting the
inter-node puts nets nothing at b1/b2 held at every cell.

## 5. Net assessment

The wave path wins because its two costs (rereads that hide under MMA,
padding that amortizes at large budgets) are cheaper than what any
dedup-preserving progressive release gives up (the skewed tail) plus
what it must spend (per-piece put constants). The canon wave-adapt
dial — collapse when reread+padding can't pay, waves otherwise — is
the right production schedule at both scales; this campaign moves it
from "best measured" to "defended against its strongest challenger."

If ever reopened, the prerequisites are known and recorded: (a) P
shrinks with NN (auto rule), (b) skew-aware piece boundaries (cut
chunk index at n_chunks*(p/P)^(1/4) to balance wire mass), (c) a
conv-only piece variant (piece-slice the cheap NVLink hops, keep ONE
wire put per dest), (d) sort-reference piece twin for CHECK_IDENTITY
(compare currently skipped under pieces; product gate covers outputs),
(e) receiver piece-progressive folding (bucket sub-lanes). None of
these lifts Wall 1's ceiling above ~the wave path at 4n; (a)+(c) are
what could make 8n competitive.

## 6. Ledger

Code: pv2 9a582e5 (merge), knobs default OFF, staged arms untouched.
Arms: ours_l01_s1_pv2_r2_{chunk,wa0,pieces,pieces2,+_gate twins}
(never headline). Capsules (committed a678216 + the merge): M1 smokes
012542/012726, gates 012835/013030, A/B 013149/014319, nsys 015422;
M2 gates 022702/022909, A/B 023028/024455; 8n gate 053349, A/B 053552.
Qwen b2 canon cell in 024455 = CXI transient (void). Node-hours: ~2.5h
4n interactive (two allocations, both scancelled on completion) + 30
min 8n DEBUG QOS (user call — granted in minutes where -q regular sat
behind ~8.4k pending jobs; use debug for short 8n lanes). Worktree
flux-v2-combine pruned post-merge (rebuild from pv2 to rerun pieces
arms); flux-fast-split kept (unmerged + uncommitted capsules).
