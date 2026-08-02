# Layer0 a2av and a2av_hier — data-movement timelines

Continuation of `layer0_baseline_annotated.md` (read that first — the
constants, the event-snapshot rule, and the "regime A/B" SM discipline are
all assumed here). Same constants: `L` local ranks, `N` nodes, `R = N·L`
ranks (`W` in code), `T` tokens per shard, `H` hidden, `k` topk. New here:
`chunk(s, d)` = rows source rank `s` must deliver to destination rank `d`
(the routed (token, topk-slot) copies — this is where the sweep's pre-topk
budget × `k` actually becomes wire bytes). All code references are
`gemm_grouped_v2_ag_scatter.cc` unless noted.

The one-sentence contrast with the baseline: **the baseline pulls everything
and filters at compute time; a2av pushes only what routing demands, and every
arrival is announced by an epoch-stamped point-to-point signal instead of
barriers and flags.**

---

## 0. Machinery shared by both variants

**Host-side offsets, no exchange.** Every rank holds the identical global
`scatter_index`/`splits` (harness property), so all send/recv offsets — the
`W×W` chunk matrix prefix sums — are computed on the host with no
communication (lines 1687-1700).

**Stage 1 + pack (producer side).** One fused kernel
(`a2av_stage1_impl`, `sort_util.cu:525`) decodes every (token, topk-slot)
copy, fills the `[W,W]` chunk counts, and emits pack keys; then a single
`index_select` gathers my rows **destination-major** into the symmetric send
buffer (line 1273). Destination-major means each destination's rows are one
contiguous slice → each src→dst transfer is ONE contiguous put.
`ready_event` (`R`) is recorded after the pack (1275).

**Stage 2 (consumer-side indices).** Pure torch index ops on the main stream
build `sorted_gather_index` / `sorted_scatter_index` / per-(expert, source)
cumsums describing the recv buffer. The recv layout is (source, expert,
dst_row)-ordered, so incoming puts land ready-to-use — **there is no unpack
kernel**; the GEMM reads rows through `sorted_gather_index` exactly as in the
baseline.

**Signals replace flags.** The consumer-side announcement is
`a2av_signal_buffer`: a symmetric `uint64[W]`, slot `s` owned by source `s`.
Delivery is `nvshmemx_putmem_signal_nbi_on_stream`: payload into my recv
buffer, then `signal[s] = run_id` — NVSHMEM guarantees the signal is ordered
**after** the payload. GEMM threadblocks spin with an acquire load until
`sig[s] >= signal_expected` (`ag_scatter_gemm_grouped_with_absmax.h:533`).
Two differences from the baseline's flags:

1. **Epoch discipline.** The buffer is *never memset* (a memset would race
   with a peer's early next-iteration put). Instead `run_id_` increments per
   iteration and the compare is GEQ. This is the push-design answer to "the
   receiver can't know when a reset is safe": don't reset — version.
2. **Set by the wire, not by my streams.** In the baseline I wrote my own
   flags after my own pulls. Here, `signal[s]` on my GPU is written by the
   NIC/peer as the last stage of source `s`'s put. My recv timeline is paced
   by 15 other ranks' send streams — nothing I enqueue locally tracks it.

**Zero-payload rule.** A destination owed nothing still gets a bare
`signal_op` (lines 1731-1734): the GEMM waits on every source column, so
every source must always announce, even "nothing for you this round."

**No open barrier.** Push needs no publish guarantee — the sender owns its
data. The **close** barrier survives (line 2800): it quiets outstanding nbi
puts and fences recv/send/stage buffer reuse for iteration t+1 (and the
epoch rule above covers the signal buffer).

---

## 1. Raw a2av (`a2av_dispatch`, flat alltoallv)

Wire program (after `R`): self-delivery is a local `cudaMemcpyAsync` +
local `signal_op` on `cp_stream` (1709-1717); then the W−1 remote puts are
issued back-to-back on `cp_stream_inter_node` in ring order `rank+1,
rank+2, …` (each step is a source→dest bijection across ranks — no incast;
lines 2412-2415). All puts are **nbi**: the stream op completes when the
transfer is *handed to the NIC/CE*, not when it is delivered. Nothing ever
waits for delivery locally — delivery is observed only by the destination's
spinning tiles.

`I` = `fetch_remote_event` (2432) therefore means "**all puts issued**", and
that — not completion — is the GEMM gate (2764): pure regime B.

```
cp                   ┌─[self copy · sig[me]=run_id]················(wait I)──AG
                     │
main   [stage1][pack]R─[stage2: build sorted indices]··············(wait I)─►[ GEMM ▓▓▓▓▓ tiles spin: sig[s] ≥ run_id ▓▓▓▓▓ ]──(wait AG)──[█ close barrier █]
                     │                                                      │
inter                └─[put+sig→d₁][put+sig→d₂][put+sig→d₃]····[put+sig→d₁₅]I

wire   ····[payload from source s ⇒ my recv rows][sig[s]=run_id]····[payload from s′ ⇒ recv][sig[s′]=run_id]····
       (the wire lane is NOT one of my streams — inbound puts land whenever each source's send stream and the NIC deliver them;
        putmem_signal guarantees each sig lands after its payload)
```

Legend (code names in `gemm_grouped_v2_ag_scatter.cc` unless noted):

| symbol | code | line |
|---|---|---|
| `[stage1]` | `a2av_stage1_impl` — fused decode + `[W,W]` chunk counts + pack keys | 1185 (kernel `sort_util.cu:525`) |
| `[pack]` | `index_select` of my rows, destination-major, into the symmetric send buffer | 1273 |
| `R` | `ready_event`, recorded after the pack; both copy streams wait on it | 1275, 1672-1673 |
| `[stage2]` | torch index ops → `sorted_gather_index` / `sorted_scatter_index` / cumsums | 1284-1632 |
| `[self copy · sig[me]]` | `cudaMemcpyAsync` send→recv + local `nvshmemx_signal_op` | 1709-1717 |
| `[put+sig→dᵢ]` | `nvshmemx_putmem_signal_nbi_on_stream`, ring order `rank+1, rank+2, …` (bijection per step, no incast); zero-payload destination → bare `signal_op` | 2413-2415, 1719-1734 |
| `I` | `fetch_remote_event` = all puts **ISSUED** (nbi: handed to NIC/CE, not delivered); waited by the GEMM gate and by `cp_stream` | 2432-2433, 2764 |
| `AG` | `all_gather_event` at the end of `cp_stream`; main waits it before the close barrier | 2434, 2784 |
| `[ GEMM ]` | `op->run`; each tile acquire-spins `sig[s] ≥ run_id` for every source its M-range spans | 2778, `absmax.h:533` |
| `[█ close barrier █]` | `nvshmemx_barrier_all_on_stream` — quiets outstanding nbi puts, fences send/recv buffer reuse for iteration t+1 | 2800 |
| `sig[s]`, `run_id` | `a2av_signal_buffer` — symmetric `uint64[W]`, slot `s` owned by source `s`; never memset, epoch increments, GEQ compare | 316, 860 |

What overlaps what: the GEMM starts as soon as stage 2 is done and the puts
are issued; **all** communication — outbound drain and inbound arrivals —
runs under it. Sources unlock incrementally as their signals land. The cost
raw a2av still pays at scale: every rank sends `R−L` separate inter-node
messages (one per remote rank), each carrying only that destination's rows —
message-count and per-message overhead scale with R, and every copy of a
token crosses the network once per remote destination.

---

## 2. a2av_hier (hierarchical dispatch, `nnodes > 1`, lines 2260-2434)

Mirrors the baseline's hierarchy — intra-node direct, inter-node via
same-local-rank gateways — but push-based and barrier-free. Three pieces:

**Round 0, intra-node (on `cp_stream`).** Self memcpy + signal, then L−1
direct `putmem_signal_nbi` over NVLink in *mirror local order*
(`d = (my_lr − dl) mod L`, line 2323-2326) — each slot a bijection, no
incast. `H` = `hier_dispatch_event_` recorded here (2327): "round-0 puts
issued."

**Inter-node aggregates (on `cp_stream_inter_node`).** The send buffer is
destination-major in *global* rank order and a node's ranks are globally
contiguous — so all rows for node `m`'s L ranks are ONE contiguous slice.
Each remote node gets a single aggregated `putmem_signal_nbi` (2341),
addressed to my same-local-rank **gateway** there, landing in its symmetric
`a2av_stage_buffer_`, with the arrival announced in
`a2av_node_signal_buffer_[my_node] = run_id`. N−1 sends, issued
back-to-back, nbi, no barriers, mirror node order. `I` =
`fetch_remote_event` (2432) = "aggregates issued."

**Gateway forwarding (on `cp_stream`, concurrent with the GEMM).** For each
source node `ns` in ascending round order:
`CUStreamWaitValue64(cp_stream, node_sig[ns], run_id, GEQ)` (2366) — a
**front-end memop wait, zero SMs, cannot deadlock against the spinning
GEMM** — then L sub-chunk deliveries in mirror local order: NVLink
`putmem_signal_nbi` from the staged segment into each local peer's recv
buffer + `signal[s]` (2395), and a local memcpy + local `signal_op` for the
gateway's own sub-chunk (2385-2393). This wait is the exact a2av analogue of
the baseline's `[b]`+`fetch_remote_event` — with the barrier *kernel*
replaced by a doorbell in symmetric memory plus a front-end wait (the
"regime B redesign exercise", implemented).

The GEMM gate (2748-2764) waits `H` **and** `I` — both "issued" events — so
it launches right after stage 2, and the entire inter-node phase plus all
forwarding runs underneath it:

```
cp                   ┌─[self copy · sig[me]][put+sig→p₁][put+sig→p₂][put+sig→p₃]H····[w₁][fwd n+1: L× put+sig, src=(n+1,ℓ)][w₂][fwd n+2: same, src=(n+2,ℓ)][w₃][fwd n+3: …](wait I)AG
                     │                                                          │
main   [stage1][pack]R─[stage2: build sorted indices]···················(wait H)·············(wait I)─►[ GEMM ▓▓▓ tiles spin: sig[s] ≥ run_id ▓▓▓ ]──(wait AG)──[█ close barrier █]
                     │                                                                              │
inter                └─[agg put+nodesig→G@(n+1,ℓ)][agg put+nodesig→G@(n+2,ℓ)][agg put+nodesig→G@(n+3,ℓ)]I

wire   ····[agg payload from (n+i,ℓ) ⇒ my stage buffer][node_sig[n+i]=run_id]····        ← each arrival releases my [wᵢ]
       ····[intra put payloads from my L−1 node peers ⇒ recv rows][sig[peer]=run_id]····  ← unlocks intra-source tiles directly
       (wire = not my streams; my whole inter row is nbi ISSUE work — microseconds — so I fires early and the gate is
        effectively stage2-bound; the [wᵢ] waits on cp are where real inter-node latency is absorbed, under the GEMM)
```

Legend (code names in `gemm_grouped_v2_ag_scatter.cc`; `R`, `AG`, `sig[s]`,
`run_id`, close barrier as in the raw-a2av legend above):

| symbol | code | line |
|---|---|---|
| `[put+sig→pᵢ]` | round-0 intra-node `putmem_signal_nbi` over NVLink, mirror local order `p = (my_lr − i) mod L` — bijection per slot, no incast | 2322-2326 |
| `H` | `hier_dispatch_event_` = round-0 intra puts issued; first half of the GEMM gate | 2327, 2756 |
| `[agg put+nodesig→G@(n+i,ℓ)]` | ONE aggregated `putmem_signal_nbi` per remote node — the node's L destination chunks are one contiguous send-buffer slice — into gateway `(n+i, ℓ)`'s symmetric `a2av_stage_buffer_`; doorbell `node_sig[my_node] = run_id`; empty aggregate → bare `signal_op` | 2336-2357 |
| `I` | `fetch_remote_event` = aggregates **issued**; second half of the GEMM gate; `cp_stream` waits it before `AG` | 2432-2433, 2764 |
| `[wᵢ]` | `CUStreamWaitValue64(cp_stream, node_sig[n+i], run_id, GEQ)` — front-end memop, **0 SMs, cannot deadlock against the spinning GEMM** | 2366-2370 |
| `[fwd n+i]` | forward the staged segment of source `(n+i, ℓ)`: L−1 NVLink `putmem_signal_nbi` sub-chunks into local peers' recv buffers + self memcpy/`signal_op`, mirror local order | 2375-2408 |
| `G@(n+i,ℓ)` | the same-local-rank gateway on node `n+i` (each rank is also gateway `ℓ` of its own node — the wire lane's stage-buffer arrivals) | 2338 |
| `node_sig` | `a2av_node_signal_buffer_` — symmetric `uint64[N]`, one slot per source node, epoch discipline, never memset | 320, 512 |
| GEMM gate | `cudaStreamWaitEvent(stream, H)` then `cudaStreamWaitEvent(stream, I)` — both "issued" events | 2748-2764 |

There is **no open barrier and no barrier kernel anywhere on the dispatch
path** — every cross-rank dependency is a payload-ordered signal plus a
front-end wait.

Token arrival order at a rank: self → intra-node peers (direct puts) →
remote node `n+1, n+2, …` as its own gateway forwards each staged round —
the same receiver stage order as the baseline, which is why the dense static
problem schedule carries over unchanged (mirror orders are chosen precisely
so "receiver `d` sees source `s` at the stage the schedule expects").

**What hier buys and doesn't buy.** Inter-node message count per rank drops
from `R−L` to `N−1`; each message is big and contiguous (NIC-friendly, no
incast by construction). Inter-node *bytes* do **not** drop: the aggregate
is the concatenation of all L destinations' chunks, so a token going to
several ranks of one remote node still crosses the network once per
destination copy. Deduplicating those copies (send the union once, re-fan
at the gateway with `index_select`) is exactly `a2av_hier_compress` — which
is also why compress needs SMs at the gateway and forces `sm_margin ≥ 1`,
while plain hier's forwarding is pure CE + front-end memops.

**Sweep `phases` mode mapping** (`FLUX_A2AV_TIMING`, events at 868, 1282,
1632, 2766, 2781, 2809): `stage1` = decode+pack, `stage2` = index build,
`gemmgate` = the H/I waits, `gemm` = kernel span (includes any tile
spinning!), `barrier` = AG wait + close barrier. In the diagrams: `stage1` +
`stage2` span the main stream up to the gate; `gemm` spans the GEMM bar;
comm hides inside `gemm` unless it outlives it — then it surfaces in
`barrier`.

---

## 3. Q&A round — the producer pack (2026-07-31)

**Q: baseline pack is just "move shard to symmetric buffer"; a2av needs
per-destination contiguous chunks. How is the send heap laid out, and what
kernel math achieves it?**

Confirmed: baseline's pack is one `cudaMemcpyAsync` of the whole shard into
the symmetric input buffer (line 736) — every rank wants all of it, order
irrelevant. a2av sends a *different subset* to each destination, and each
subset must be contiguous to leave as ONE `putmem_signal_nbi`.

**Layout.** `a2av_send_buffer` is symmetric `[tokens_per_rank_max * topk,
hidden]` (line 496) — sized in **copies**, not tokens (a token routed to k
experts occupies k rows; raw a2av does not dedup). Rows are my copies sorted
by `(destination rank d, expert e, copy index lp)`; destination d's chunk
occupies rows `[send_off[d], send_off[d] + chunk(me,d))` with
`send_off[d] = Σ_{d'<d} chunk(me,d')` (host prefix sum, ~line 1692).

**Kernel math** (`a2av_stage1_kernel`, sort_util.cu:525):

1. *Decode*: for each copy p globally, `d = scatter_index[p]` is its row in
   the expert-sorted global layout; binary search over the smem inclusive
   expert prefix `splits_cum` finds expert `e` (first `e` with
   `splits_cum[e] > d`); `owner = e / ep_nexperts`. Experts are
   **block-assigned** to ranks, so expert-ascending ⇒ owner-ascending.
2. *Pack key* (line 576, my copies only): `pack_key[lp] = e*copies_per_rank
   + lp`. One integer encodes the 3-level sort: high part majors by expert
   (hence destination), `+lp` breaks ties by copy index without cross-expert
   collision. Keys unique ⇒ argsort trivially stable. The SAME `(e, lp)`
   tie-break builds the consumer keys in stage 2 — sender order and receiver
   expectation agree with no negotiation.
3. *Gather* (lines 1266-1273): `perm_s = pack_key.argsort();
   send_gather_index = perm_s / topk` (recovers the token row, since
   `lp = token*topk + slot`); one `index_select_out` writes all rows.
4. *Chunk counts*: same kernel histograms `(s = p/copies_per_rank, owner)`
   into smem `[W,W]`, flushed by atomicAdd (571-598); ~1 KB D2H to pinned
   (1209); host waits only on `counts_event_` (1639) — stage-2 sorts keep
   running — then prefix-sums it into `send_off` (my row) and `recv_off`
   (column d, sources before me = my exclusive offset into d's recv region).

**Why no unpack:** all senders share the discipline, so every destination's
recv buffer lands in (source, expert, copy) order — exactly what the stage-2
`sorted_gather_index` expects. The pack `index_select` is the only
data-rearranging kernel in the dispatch.

## 4. Q&A round — recv layout, signals, and the tile gate

**Q: is the recv buffer `copies_per_rank * W`, with each source at a fixed
stride? How do recv offsets work, and how do signals unblock the GEMM?**

**Sizing — exact-packed, not strided.** The plausible `cpr * W` fixed-stride
layout is NOT what's implemented. `a2av_recv_buffer` is
`[max_recv_ntokens_, hidden]` with default
`max_recv_ntokens_ = min(total_copies, 2 * tokens_per_rank_max * topk)`
(lines 490-492) — 2x the *balanced* per-rank load, override
`FLUX_A2AV_MAX_RECV_NTOKENS`, runtime-guarded `FLUX_CHECK(M_this_ep <=
max_recv_ntokens_)` (1662). Exact packing works because **every rank runs
stage1 over ALL W*cpr copies** (`scatter_index`/`splits` are global routing
metadata — the pre-known gating output), so every rank locally derives the
full `[W,W]` chunks matrix. Sender s computes its exclusive offset into
destination d's recv region — `RO[s][d] = Σ_{s'<s} chunk(s',d)` — with zero
communication. Fixed strides would also address fine but cost W× the
symmetric heap (which `NVSHMEM_SYMMETRIC_SIZE` caps) and would scatter the
GEMM's rows sparsely; exact packing keeps rows dense in `[0, M_this_ep)`.

**Ordering — one global rule, not arrival order.** Physical recv layout at
every destination: source-major regions s = 0..W-1 (region sizes differ per
destination, rule doesn't), each interior in (expert, copy-index) order ==
the producer's chunk order. So physical = **(source, expert, copy)**.

**Physical vs GEMM order.** The grouped GEMM's logical mat-A order is
**(expert, source, copy)** — expert-major because there is one GEMM problem
per expert, with per-source segments inside each problem
(`sorted_splits_cumsum [E, W]`). The permutation between the two orders IS
`sorted_gather_index` — the "unpack" folded into the GEMM's gather. Because
both orders sort by the same copy-index tie-break within any (s,e) group,
the map is pure group-offset arithmetic (meta path: `searchsorted` +
`offR_of_A[g] + iota - offA[g]`, ~1545-1555).

**Signals → tile gate.** `a2av_signal_buffer` = symmetric `uint64[W]`, never
memset (GEQ epoch: `sig[s] >= run_id`). Source s sets slot s at destination
d via `putmem_signal_nbi` (payload-before-signal), a bare `signal_op` when
the chunk is empty, and `memcpy + signal_op` for itself. The gate
(`ag_scatter_gemm_grouped_with_absmax.h:513-541`): each tile computes its
row span `[m_start, m_end]` inside its expert problem; a warp ballot +
`__ffs` over `split_accum` (the per-expert source cumsum) finds which source
segments the tile straddles; one lane per contributing source spins on a
system-scope acquire load `while (sig[lane] < run_id)`; then
`__syncthreads()`. **Granularity: per tile x contributing sources** — a tile
wholly inside source 3's segment waits only for `sig[3]`; a straddling tile
waits for both its sources; tiles over early arrivals compute while late
sources' bytes are still on the wire. That per-tile self-pacing (instead of
the baseline's launch-gate-on-last-event) is the overlap mechanism.

**Put streams** (raw path, lines 2411-2415): default = ring order
`rank+1..rank+W-1`, ALL on `cp_stream_inter_node` (intra-node destinations
included). Scheduled mode (`a2av_ring_`): reverse hierarchical ring — intra
puts on `cp_stream`, inter puts concurrently on `cp_stream_inter_node`, slot
k a source→destination bijection (no incast), mirroring the receivers'
stage order. Then `fetch_remote_event` on inter → cp waits →
`all_gather_event` (2432-2434; baseline event names reused).

---

## 5. Q&A round — split-put signaling and minimal metadata (answers)

**Split-chunk signaling.** Signaling `run_id` after the first half releases
ALL lanes waiting on that source — second-half tiles read stale garbage
(buffer never memset ⇒ last iteration's tokens: plausible wrong numbers,
not NaNs). Early per-half firing needs BOTH ends refined: signal space
(slot per (source, half)) and gate geometry (`split_accum` at half
boundaries). Single-slot alternative: `SIGNAL_ADD` 1 per half, wait
`sig[s] >= 2*run_id` (epoch discipline survives — the counter accumulates
exactly 2/iteration) — but that only fixes last-half waiting, not early
firing. **Hidden subtlety:** even "two puts, signal on the second" is
broken — putmem_signal's payload-before-signal covers only ITS OWN payload;
separate nbi puts to the same PE are unordered without `nvshmem_fence`.
This is why one-put-per-destination is a correctness design, not just a
bandwidth nicety: a single put+signal carries the whole ordering argument.

**Minimal metadata AG.** Minimum = per-token topk expert ids (W·T·k small
ints). Every rank deterministically recomputes `splits` (histogram) and
`scatter_index` (prefix + stable rank-within-expert) from identical input.
Symmetry: stage1's binary search exists only to invert scatter_index
(d → e); shipping expert ids makes the decode free but requires
constructing `flat_dst` (e → d: histogram + prefix + stable rank) — the
work changes direction, not size (~one extra stage1-sized pass). The
harness precomputing scatter_index outside the timed region is a real but
bounded simplification.

---

## 5b. Q&A round — what unblocks the puts, and what run_id really is

**Two events, two consumers.** `counts_event_` (after stage1's 1 KB counts
D2H, line 1210) gates the HOST (`cudaEventSynchronize`, 1639): it cannot
compose the put calls without `send_off`/`recv_off`/lengths, all host
prefix sums over the [W,W] counts. `ready_event` (after the pack
index_select, 1275) gates the DEVICE: the cp streams wait on it
(1672-1673), so no bytes leave before the send buffer is contiguous. The
host therefore enqueues puts while the pack may still be running — host
launch latency hides under the pack; correctness rides on the stream wait,
never on host timing.

**run_id.** Incremented once per forward (line 860); every put in one
iteration carries the SAME value, written to the sender's slot `sig[me]`
at each destination. (Diagram note: `p+s→+k` means putmem_signal to
destination `(me+k) mod W` — ring order — NOT a signal value.)
Single-shot production call: run_id = 1 over an all-zero buffer —
effectively an on/off flag. The epoch (GEQ, never memset) exists to
eliminate the flag-reset problem across iterations: a reset would have to
land after every tile read but before any of W remote writers' next
signals — a distributed race. With GEQ, iteration n's value can't satisfy
iteration n+1's test, so stale signals are inert and no one ever writes a
zero.

### Timeline — raw a2av, stage1/pack split, host lane explicit

```
host   ····································┌─(wait C)──[send_off/recv_off]──[enqueue: self · p+s→+1 … p+s→+15]   «CPU: enqueue only — device order held by R»
                                           │
main   [stage1: decode+counts]──[1KB D2H]──C──[argsort]──[pack: index_select]──R──[stage2: consumer indices]························(wait I)┌►[ GEMM ▓ tiles spin: sig[s] ≥ run_id ▓ ]──(wait AG)──[█ close barrier █]
                                                                               │                                                            │                                          │
cp     ········································································├─[self copy · sig[me]=run_id]·······················(wait I)┼─AG╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┘
                                                                               │                                                            │
inter  ········································································└─[p+s→+1][p+s→+2][p+s→+3]··························[p+s→+15]I
```

C = counts_event_ (1210, host sync 1639: metadata gate, CPU only). R =
ready_event (1275, stream waits 1672-1673: payload gate, device). p+s→+k =
putmem_signal_nbi to rank (me+k) mod W, same run_id on every put (issue_put
1719, ring loop 2412-2415). I = fetch_remote_event 2432 (puts issued; GEMM
launch gate 2748-2764; cp joins 2433). AG = all_gather_event 2434 (main
joins at 2784 before the close barrier 2800). Tile spin: absmax.h 513-541.

### Timeline — a2av_hier (N=4, L=4; G@(n,l) = gateway rank l on node n)

```
host   ·····················┌─(wait C)──[u/U offsets]──[enqueue: intra p+s · agg p+s · w+fwd rounds]   «CPU: enqueue only»
                            │
main   [stage1]──[1KB D2H]──C──[argsort]──[pack]──R──[stage2: consumer indices]·········(wait H)······································································(wait I)┌►[ GEMM ▓ tiles spin: sig[s] ≥ run_id ▓ ]──(wait AG)──[█ close barrier █]
                                                  │                                            │                                                                              │                                          │
cp     ···········································├─[self copy·sig[me]][p+s→ℓ₁][p+s→ℓ₂][p+s→ℓ₃]H──[w1][fwd1: p+s→ℓ₀…ℓ₃]──[w2][fwd2: p+s→ℓ₀…ℓ₃]──[w3][fwd3: p+s→ℓ₀…ℓ₃]─(wait I)┼─AG╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┘
                                                  │                                                │                      │                      │                            │
wire   ···········································│··············································[ns1]··················[ns2]··················[ns3]··························│
                                                  │                                                                                                                           │
inter  ···········································└─[agg p+s→G@(n+1,me)][agg p+s→G@(n+2,me)][agg p+s→G@(n+3,me)]······························································I
```

Deltas vs raw (shared machinery — C, R, stage1/pack, tile spin — identical):
`[p+s→ℓᵢ]` = round-0 intra puts to my L-1 local peers, mirror order,
signal `sig[me]` (2322-2326). `H` = hier_dispatch_event_, "intra issued"
(2327). `[agg p+s→G@(n+i,me)]` = ONE aggregated put per remote node (all L
chunks concatenated) into my same-local-rank gateway's stage buffer; its
signal is the `node_sig` doorbell (2336-2357). `[nsᵢ]` = INCOMING doorbell
from remote node n+i — timing set by the remote sender, no local lane
produces it. `[wᵢ]` = CUStreamWaitValue64 on that doorbell: front-end
wait, zero SMs (2366-2370). `[fwdᵢ]` = gateway forwarding: slice source
(n+i, my_lr)'s staged aggregate, NVLink p+s per local rank, signaling
`sig[(n+i,my_lr)]` at each (2375-2408). GEMM gate = (wait H)+(wait I),
both issued-semantics (2748-2764) — forward completion is deliberately NOT
in the gate (it would chain my GEMM launch to remote nodes' send timing);
arrival truth lives in the per-source signals the tiles spin on.

Key reading: the cp lane switches roles at H — sender (intra dispatch)
before, receiver-side gateway (forwarding) after. Hier cuts inter-node
MESSAGES per rank from R-L to N-1; BYTES are unchanged (a token wanted by
3 ranks of node m crosses the wire 3 times inside the aggregate) — dedup
of those copies is hier_compress.

---

## 5c. Q&A round — diagram semantics: verticals, the host horizon, and R

**Vertical convention (now explicit): verticals are DEVICE-side release
edges** — cudaEvent waits and CUStreamWaitValue waits, i.e. who executes
before whom on the GPU. Host enqueueing is a precondition, not a sync
edge: every box on every lane was enqueued by the host, but enqueue order
only puts work in queues, it never times execution. The host appears as
an edge exactly once — C — the only steady-state point where information
flows device→host and the CPU blocks. Everywhere else the host is a
moving horizon that stays ahead of device execution.

**R's consumers are the ├/└ corners**: they ARE
`cudaStreamWaitEvent(cp/inter, ready_event)` (1672-1673) — the first
executed op on each put stream, enqueued early and parked until the pack
finishes. Without R the puts race the pack and ship a half-written send
buffer; R is the payload gate that makes the pack's contiguity guarantee
mean anything.

**Hidden stall the diagram can't draw**: since enqueue is a precondition,
a slow host can leave cp idle AFTER R fires — waiting on unenqueued work,
not on any event. This is why hidden host syncs are banned in the
dispatch path (see the "CUDA bincount/nonzero are banned" comment, 1163):
each would pin the host horizon to device progress and surface exactly
this stall.

**Two-hop chain, completed** (hier): hop 1 = aggregated putmem_signal:
payload into the gateway stage buffer BEFORE the `node_sig` doorbell
flips (doorbell is a separate word from the per-source `sig[]` array).
Front-end CUStreamWaitValue64 releases the forward round. Hop 2 = the
forward's putmem_signal: chunk at the destination BEFORE `sig[s]` flips;
the tile's acquire load orders A-reads after that. With putmem_nbi + a
separate signal_op on a different stream, hop 2's payload-before-signal
breaks: nothing orders the bare signal after the nbi delivery, so tiles
can release on a signal that never certified the data — acquire orders
you after the SIGNAL WRITE, not after the payload.

---

## 5d. Q&A round — barriers deleted, certification per payload

One-sentence summary of a2av vs baseline: **every coordination barrier is
replaced by per-payload certification.** Three confirmed claims, sharpened:

1. **No pacing barriers.** No barrier between rounds/nodes/teams during
   dispatch; the ring offset (rank+1, rank+2, ...) is the only incast
   mitigation. Sole survivor: the close `barrier_all` (2800) — an
   epoch/reuse fence (quiets nbi puts, certifies buffer reuse), nothing
   inside the iteration waits on it. Caveat: baseline's per-round barriers
   made its ring collision-free BY CONSTRUCTION; a2av's ring is only
   STATISTICALLY incast-free (assumes ranks start their put trains at
   similar times — true since packs finish on similar clocks; drift can
   transiently collide slots, nothing prevents it).

2. **No node-team barrier, and who "triggers" is inverted.** Baseline
   PULLS: a getmem must not read until the remote certifies its buffer,
   and barriers are how pullers learn that. a2av PUSHES: put+signal is
   self-certifying. Puts are issued unconditionally by the sender (gated
   only by local R) — consumers trigger on arrival: GEMM tiles on
   `sig[s]`, and the one arrival-triggered SEND in the system, the hier
   gateway forward (released by the `node_sig` doorbell — an 8-byte
   front-end wait doing the job baseline needed an L-rank barrier for).

3. **Tiles: arrival-driven, no consumption order — with two structures.**
   Each tile spins on exactly the sources its `split_accum` span
   straddles. But arrivals have a predictable shape (self memcpy first,
   intra NVLink next, inter EFA last) — that skew IS the overlap. And
   `a2av_ring` deliberately re-imposes a pattern: the reverse hierarchical
   ring delivers my chunk to destination d at exactly the stage the dense
   STATIC tile schedule expects it (2418-2423) — a2av wire married to
   baseline tile order.

Closes the original baseline-notes question ("where is the nvshmem
barrier installed?"): baseline had open world + per-round TEAM_NODE +
close world; a2av deleted the first two, kept only the close.

---

## 6. Verified: intra-node put+signal rides the copy engine, not SMs

Concern raised 2026-07-31: host-issued `nvshmemx_putmem_signal_nbi_on_stream`
to NVLink peers might secretly use SM transport kernels, competing with the
GEMM (would invalidate the `sm_margin=0` assumption for raw a2av / hier).
The NVSHMEM pip wheel is binary-only, so this was settled empirically: nsys
capture of a single-node 8-rank raw a2av run (uniform b2/k8/G128 matrix,
p4d, `sweeps` data root `nsys_ce_check/a2av_ce_31.nsys-rep`).

Verdict — **copy engine, zero SMs**:

- Per-destination puts appear as `[CUDA memcpy Peer-to-Peer]` (~2.4 MB each,
  exactly budget*topk/W = 2 MiB * 8 / 8 per chunk; 150-185 GB/s, NVLink CE
  rates) on the cp/inter streams.
- The ONLY `nvshmemi_*` GPU kernels in the whole trace: symmetric-heap
  `init_array` kernels (one-time, at startup) and
  `barrier_on_stream_kernel_threadgroup` (the world barrier — per-iteration
  close barrier, the SM cost we already knew about).
- No put/transfer/proxy kernels exist. Signal writes (8 B) are front-end /
  CE ops, invisible as kernels.

So post-issue, raw a2av's intra-node data movement consumes no SMs; the SM
budget question only arises for compress's gateway `index_select`
(sm_margin >= 1) and the close barrier. Remote EFA puts go through the CPU
proxy thread (also zero SMs), not exercised in this single-node capture.

---

## 7. stage1 / pack / stage2 demystified — worked example

**Taxonomy** (three pieces, commonly conflated):
stage1 (sort_util.cu:525) = producer METADATA: decode every copy
(e/s/d/not_mine), [W,W] counts, pack keys. pack (1270-1273) = producer
PAYLOAD: argsort + index_select populating the symmetric send buffer —
this is what R certifies, and it can never move after the puts. stage2
(build_stage2, 1443+) = consumer METADATA: sorted_gather/scatter_index +
sorted_splits_cumsum, consumed only by the GEMM — which is why
FLUX_A2AV_STAGE2_AFTER_PUTS may move its ENQUEUES after the put loop
(host horizon reaches the puts sooner when host-bound; no-op when the
pack/R is binding; device edges unchanged — stage2 stays main-stream
program order, no event involves it).

**Example**: W=2, E=2/rank (e0,e1 on r0; e2,e3 on r1), T=2, topk=2,
cpr=4. Routing t0→{e0,e2}, t1→{e1,e2} (r0); t2→{e1,e3}, t3→{e0,e2} (r1).

Global expert-sort: e0 rows 0-1 (copies p0,p6), e1 rows 2-3 (p2,p4), e2
rows 4-6 (p1,p3,p7), e3 row 7 (p5) → scatter_index = [0,4,2,5,3,7,1,6],
splits = [2,2,3,1], splits_cum = [2,4,7,8].

stage1 decode = inverting that sort: p1 has d=4; first expert with
splits_cum > 4 is e2; owner = 2/2 = r1. Histogram falls out free:
chunks = [[2,2],[2,2]].

Pack keys on r0 (key = e*cpr + lp): lp0(t0,e0)=0, lp1(t0,e2)=9,
lp2(t1,e1)=6, lp3(t1,e2)=11. argsort=[0,2,1,3], /topk=[t0,t1,t0,t1]:

    send r0: [t0(e0), t1(e1) | t0(e2), t1(e2)]
              └─self, off 0─┘ └─→r1, off 2: ONE put─┘

The whole trick in one line: sorting by e gives destination-contiguity
BECAUSE experts are block-assigned (e ascending ⇒ owner ascending); lp is
a collision-free tie-break. One multiply-add per copy plus a sort.

Recv at r1 (r0's put at recv_off[0]=0; self chunk at recv_off[1]=2):
[t0(e2,s0), t1(e2,s0) | t3(e2,s1), t2(e3,s1)] — (source, expert, copy).

stage2 output: GEMM problems e2=[t0(s0),t1(s0),t3(s1)], e3=[t2(s1)];
sorted_gather_index maps that (expert,source,copy) order onto the
physical (source,expert,copy) rows; sorted_splits_cumsum records the
(expert, source) segment boundaries — the very split_accum table the tile
gate ballots over to pick which sig[s] to spin on.

---

## 8. FLUX_A2AV_NVTX_PROXY — per-source ranges inside the GEMM blob

Device code cannot emit NVTX, and all a2av waiting hides inside the single
grouped-GEMM kernel, so an nsys capture shows one opaque span. The proxy
(`src/moe_ag_scatter/ths_op/a2av_nvtx_proxy.hpp` + `include/flux/a2av_progress.h`)
splits that span per source:

- **Device → host**: the kernel publishes monotonic per-source-bucket counters
  with cheap L2 atomics into a device-memory slot block — `ready_seq[s]`
  (signal observed arrived), `claimed[s]` (tiles fired; dense schedule
  attributes a tile to its `segment_end`, the source that gates it; the
  flat-variant claimer counts at claim with exact bucket attribution),
  `completed[s]` (tiles retired). The poller thread refreshes a pinned host
  copy with an ~800 B CE memcpy per tick — the GEMM CTAs never touch PCIe
  (per-tile system atomics cost ~+45% e2e and were rejected), and no
  SM-resident helper exists (under `CUDA_DEVICE_MAX_CONNECTIONS=1` a resident
  mirror kernel serializes ahead of the GEMM on the single compute queue and
  deadlocks; CE copies interleave with a running kernel — the same property
  the hier wire itself relies on). Nothing is ever reset — epoch tags are the
  run-id, so enqueue-ahead host code can't race a running iteration.
- **Host**: two `cudaLaunchHostFunc` callbacks bracket the kernel in stream
  order (no syncs); a ~10 µs poller thread renders NVTX domain "a2av":
  `i<run_id>.src<s>.wait` (iteration start → arrival), `.pending` (arrival →
  first tile fired = claimer/SM saturation, distinct from wire latency),
  `.compute` with completion-quartile sub-ranges (`c0-25`…`c75-100`; a long
  tail sub-range = straggler tiles), plus `intra_epoch` / `inter_epoch`
  envelopes whose offset/overlap is the internode skew at a glance.
- **Honesty limits**: edges are poller observations (~5–30 µs lag); tile
  *starts* are exact prefixes (monotonic per-bucket cursors ⇒ `claimed[s]=k`
  means tiles `0..k-1` fired), tile *finishes* are counts, not identities;
  dense-mode `ready` is "first observed by a gated tile", an upper bound on
  arrival. Timeline aid only — never quote its cells as latency
  (sweeps/SCHEMA.md never-mix rule).

Usage: `FLUX_A2AV_NVTX_PROXY=1` + an nsys-mode sweep cell (trace set already
includes nvtx). Zero cost when unset (null-pointer gate in the kernel).

### Layer C: per-tile trace (same trigger, no extra knobs)

With the proxy enabled, the kernel also appends one 24 B record per fired tile
(`{problem/tile id, smid/cta/source-span, t_enter, t_fire, t_done}`, low-32
`%globaltimer` ns) into a device ring via one L2 cursor atomic, written in
three touches so only the record index crosses the mainloop (register-neutral,
verified via `cuobjdump` diff: REG identical, stack smaller). Per-source device
arrival stamps (`arrival_gt[s]`) are taken on the backstop-spin slow path —
their *absence* over a whole iteration is itself a result: no tile ever
visibly blocked, i.e. the iteration was SM-limited, not comm-limited. The
poller CE-copies each iteration's records at the kernel-end callback and
appends a self-contained block to `a2av_tile_trace_r<rank>.bin` under
`FLUX_SWEEP_RECORD_DIR` (raw platform data, never committed), and emits
`i<e>.src<s>.arrival` NVTX marks whose payload is the device-precise arrival
(range edges stay observational — NVTX cannot be backdated).

Analysis: `python sweeps/plot_a2av_trace.py <records_dir> --rank R
[--curves out.png] [--gantt out.json] [--align nsys.sqlite]` — prints the
per-arrival cohort table (dynamic last-arriving-source attribution; falls back
to the static `segment_end` label when nothing blocked), renders the
fired/in-flight regime plot (in-flight pinned at capacity = SM-limited;
snapping up at an arrival = comm-limited), and writes a Perfetto per-SM Gantt
(gray spin slice + compute slice per tile; open in ui.perfetto.dev).

---

## 9. FLUX_A2AV_EARLY_LAUNCH — the reorder, and three MAX_CONNECTIONS=1 laws

The gate experiments showed the GEMM launch was delayed not by the dispatch
event waits (removing them moved nothing) but by ~0.35 ms of serial host-side
put issuance between stage 2 and `op->run`. `FLUX_A2AV_EARLY_LAUNCH=1`
(hier and compress-union only) restructures the host program: inter-node
aggregates issue first (their proxy submission kernels must precede the GEMM
in the single compute queue), the GEMM launches immediately after stage 2
(`gemmgate_ms` 0.35 → 0.006), and the ENTIRE cp_stream wire sequence — self
copy, round-0 intra puts, gateway forwarding — is materialized as plain
descriptors and replayed after the launch (whole-sequence deferral: cp_stream
is FIFO, a partial deferral reorders delivery). Verified: intra CE copies
(~54 MB P2P + self DtoD) now execute inside the GEMM span. e2e is neutral at
b8 — the head tiles depend on the deferred copies, so earlier launch buys
spin, not compute; converting it requires guaranteed head work (self-source
first in the stage order), a separate change.

Laws of CUDA_DEVICE_MAX_CONNECTIONS=1 learned the hard way, all reproduced:
1. A kernel enqueued after the persistent GEMM serializes behind it (mirror
   kernel deadlock, §8) — hence inter (kernel-shaped) issues pre-launch and
   only CE/memops may follow.
2. Blocking NVSHMEM ops ahead of the GEMM serialize the GEMM behind the wire
   (`FLUX_A2AV_BLOCKING_WIRE` exists for measurement, not production; its
   `nvshmemi_*_signal_entrypoint_blocking` kernel spans the wire drain —
   ~3 ms at b8 — and is the visible inter-node put; capture with
   CUDA_DEVICE_MAX_CONNECTIONS=8 to see it beside the GEMM).
3. A pending `cudaLaunchHostFunc` in the channel dams everything enqueued
   after it in HOST order: wire ops queued behind a hostfn that fires at GEMM
   completion deadlock against the GEMM's own signal spins (observed as a
   16-rank first-forward hang; fixed by issuing the deferred wire before
   enqueueing the iter-end callback).

Why native flux never needed any of this: static pattern (near-zero
issuance), self-shard pre-placed (guaranteed head work), all movement CE +
flags (kernel-free — the single queue never bites).

---

## 10. Minimal-wait spin: no waiting on empty segments (default contract)

The per-tile gate now requires a lane's segment to be non-empty
(`split_accum[s] > split_accum[s-1]`, one `__shfl_up_sync` + compare on
values the ballot already loaded — register-neutral, ns-scale, verified via
cuobjdump + SASS). Boundary lanes are non-empty by ballot construction, so
this skips exactly EMPTY INTERIOR segments (flat cumsum spots inside a
tile's span); skipped lanes have no rows to acquire-order, so memory
semantics are unchanged. The claimer's multi-source masks already filtered
empty sources at prepare time (`fill_problem_info_a2av`) — the backstop
spin now agrees with the prepare kernel. Zero-payload signals are STILL
SENT; the contract is: never wait on them, never stop sending them.

Two diagnostic lessons from validating this (2026-08-01, rank-9 case):
- The stall that motivated the change turned out to be a GENUINE
  dependency: the gating source's segment was merely SHORT (< 128 rows,
  swallowed inside straddling tiles across experts, so it never owns a
  tile's m_end). Layer-C `expected[s]` counts tiles attributed by seg_end —
  `expected[s] == 0` does NOT imply zero rows. Ground truth for emptiness
  is the row cumsum, not the tile attribution. (Sidecar follow-up: include
  per-source row counts in the header so analyses can tell these apart.)
- The real lever for such stalls is sender-side: a tiny segment's
  putmem_signal orders behind the source's larger puts on its wire stream
  (amplified under blocking-wire), so hundreds of consumer tiles gate on a
  signal delayed by unrelated bytes. Candidate: issue small per-destination
  segments before large ones in the put loops.

The empty-skip therefore fires rarely on dense-ish matrices; it is
insurance that the wait set is minimal by construction, priced at ~nothing,
and it matters increasingly with scale/sparsity (more zero pairs).

---

## 11. Early launch + blocking wire on ALL a2av variants (2026-08-02)

Both instrumentation modes were generalized from {hier, compress-union} to the
whole family, so every variant's nsys timeline is complete and comparable:

- **Blocking wire** now also covers the flat variant's per-destination puts
  (inter-node destinations only — blocking the 7 intra P2P puts would
  serialize the fan-out) and the balanced relay's phase-2 wire put. The relay
  first hop stays nbi: it is intra-node (pieces to local relay peers), like
  every other intra put.
- **Early launch** now covers flat/ring (their deferred cp_stream set was
  already emitter-based — the old exclusion comment was stale) and the two
  compress arms. The SM-free wire (self copy, round-0 intra puts, hier/union
  forwarding) defers exactly as before; the gather-arm / relay-gateway tails
  (index_select SM kernels + forwards) are **issued inline on the idle pack
  stream** behind their front-end waits, never deferred.
- **A fourth MAX_CONNECTIONS-adjacent law, discovered the hard way**: a
  kernel enqueued AFTER the persistent GEMM has blanketed every SM can
  starve at dispatch *forever* — not merely run margin-throttled. The first
  implementation deferred the relay's gathers behind the GEMM launch; the
  16-rank first forward wedged permanently, and per-op replay events +
  `cuda-gdb info cuda kernels` showed each rank's cp_stream dying at an
  arbitrary depth in its gather sequence with ONLY the GEMM resident (SMs
  mask fff…f = all 108) and the stalled `index_select` never dispatched.
  Whether a post-blanket kernel dispatches is a race against the GEMM ramp
  (identity's single-hop wire won it; the relay's two-hop `node_sig` always
  lost). Pre-launch enqueue behind a front-end wait — the non-early
  configuration — dispatches reliably; that is what the inline pack-stream
  tail restores. The ctor still FLUX_CHECKs `CUDA_DEVICE_MAX_CONNECTIONS >
  1` for these arms: at conn=1 the tail's pending pre-launch waits would
  serialize the GEMM launch behind full wire delivery in the single channel.
  This remains a **visibility mode**, not a perf claim — compare
  instrumented cells only against equally-instrumented cells (SCHEMA
  invariant 3's analog).

Analysis side: `plot_a2av_trace.py --scan-ranks` ranks all ranks of a capture
by starvation (longest contiguous window below half the rank's peak in-flight,
plus a normalized deficit integral) — it independently re-finds rank 9 on the
§10 capture — and `--compare LABEL=DIR[:RANK]` renders worst ranks of several
runs on one shared time axis.

---

## Comprehension questions

1. **The two-hop signal chain.** In hier, destination `d` (not a gateway for
   source `s`) never communicates with `s` directly, yet its tiles safely
   spin on `sig[s]`. Trace the chain that guarantees `d`'s recv rows exist
   before `sig[s] = run_id` becomes visible at `d` — there are two
   payload-before-signal hops and one front-end wait linking them. Which
   single link would break if the gateway's forward used `putmem_nbi` + a
   separate `signal_op` on a *different* stream?

2. **Bytes vs messages.** A token on rank `(n, ℓ)` is routed to all L ranks
   of remote node `m`. Count its inter-node crossings and the number of
   inter-node messages it rides in, for raw a2av vs a2av_hier vs (expected)
   hier_compress. Which resource does each step of the hierarchy save, and
   at what new cost on the gateway?
