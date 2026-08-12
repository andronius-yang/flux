# Dense Allgather vs A2AV: Who Needs a "Global Copy Id" — and Why

A companion to `a2av_packing_walkthrough.md`. That doc explained *how* the
a2av path packs and reorders. This one answers a sharper question:

> The dense allgather path has worked for years **without** any "global copy
> id". The a2av path cannot live without one. What changed?

Short version: the two paths use different **addressing schemes** for rows.

- Dense AG names a row **by address** — "token row 21" means the same
  physical location on every rank, because the gathered buffer has a fixed,
  static layout.
- A2AV names a row **by position in a queue** — "the 2nd row of the chunk
  source 5 sent me". There is no fixed layout; offsets are *derived from
  counts*. A positional name only works if sender and receiver agree on the
  queue order **without talking to each other** — and the global copy id is
  exactly that agreement.

Code references: `src/moe_ag_scatter/sort_util.cu` and
`src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc`.

---

## 0. The cast of characters

| Symbol | Meaning | Value in our example |
|---|---|---|
| `W` | world size (number of GPUs) | **8** (r0..r7) |
| nodes | 2 nodes × 4 GPUs | node 0 = {r0..r3}, node 1 = {r4..r7} |
| `nexperts` | total experts | **8** (e0..e7) |
| `E` | experts per rank (`nexperts / W`) | **1** — rank r owns expert **e_r**, full weights (EP = world size) |
| `tokens_per_rank` | tokens per shard | **4** (t0..t3 on each rank) |
| `topk` | experts per token | **2** |
| `cpr` | copies per rank = `tokens_per_rank · topk` | **8** |

Two ways to *name* things, which the two paths use differently:

- **token row** = `s · tokens_per_rank + n`
  — where token n of rank s lives in the dense all-gathered buffer.
  Rank 5's t1 → token row `5·4 + 1 = 21`. An **address**.
- **global copy id (gc)** = `s · cpr + n · topk + k`
  — the flattened position of the (token n, slot k) entry in the global
  `scatter_index` tensor, enumerated rank 0's tokens first, then rank 1's, …
  Rank 5's t1, slot 1 → `5·8 + 1·2 + 1 = 43`. A **name for one copy**,
  computable by *every* rank from the replicated `scatter_index`, with no
  communication (`s = gc / cpr`, plain integer division).

---

## 1. The running example: one expert, six rows

We only need to watch **one expert**: e2, owned by rank 2 (node 0). Say the
router sent it six copies:

| Source | Token | Its topk | e2 is slot | token row | gc |
|---|---|---|---|---|---|
| r0 | t0 | {e2, e5} | k=0 | 0  | **0**  |
| r0 | t3 | {e1, e2} | k=1 | 3  | **7**  |
| r2 | t2 | {e2, e6} | k=0 | 10 | **20** |
| r5 | t1 | {e0, e2} | k=1 | 21 | **43** |
| r5 | t2 | {e2, e7} | k=0 | 22 | **44** |
| r7 | t0 | {e2, e3} | k=0 | 28 | **56** |

Note two sources (r0 and r5) contribute **two rows each**. Those multi-row
segments are exactly where the two paths visibly differ.

Both paths must end at the same place: rank 2's grouped GEMM wants e2's six
rows as one contiguous block of matrix A, grouped by source, so the tile
scheduler knows which rows depend on which source's arrival.

---

## 2. Dense AG path: address-based, and happily nondeterministic

### Step 1 — the allgather fixes the geometry

After the hierarchical allgather, **every** rank holds all 32 token rows at
**fixed offsets**: token row 21 is at byte `21 · row_bytes` on every GPU,
whether or not that rank needs it. No counts, no cumsums — the layout is
static.

### Step 2 — receiver-side sort, with a race that doesn't matter

Rank 2 builds its e2 block with the `AgScatterSortOp` kernel
(`sort_util.cu:286-291`). Two things happen:

**(a) Sources are ordered receiver-relatively.** Buckets are arranged by
`shift_rank_to_order` *from rank 2's perspective* — own rank first, then
around the local node, then the remote node the same way:

```
stage order at rank 2:   r2, r3, r0, r1,   r6, r7, r4, r5
                         ── node 0 ─────   ── node 1 ─────
```

(At rank 5 the order would be `r5, r6, r7, r4, r1, r2, r3, r0` — there is
no single global order; each receiver arranges sources by *its own* data
arrival schedule.)

**(b) Rows inside a bucket get their slot from an `atomicAdd`** — i.e., GPU
thread-arrival order, which can differ run to run. Two equally valid runs:

```
              stage:    0    |  2          |  5    |  7
              source:   r2      r0            r7      r5
Run A  token row:      10  |   0     3   |  28   |  21     22
Run B  token row:      10  |   3     0   |  28   |  22     21
                                ↑ race                ↑ race
```

### Why Run B is perfectly fine

Look at what the sorted index actually stores: `sorted_gather_index[i]` = an
**absolute token row**. In Run B, A-row 1 says "fetch token row 3" — and
that is exactly what gets fetched, from the same fixed offset as always. The
paired `sorted_scatter_index` was built in the *same kernel pass*, so A-row
1's output also goes to t3's router-assigned output position. The
permutation changed; every (input row → output position) *pairing* is
intact.

The deep reason: **each index entry carries a complete address.** The order
in which entries sit in the list is bookkeeping, not meaning. And since
rank 2's GEMM is the only consumer of rank 2's ordering, no other rank ever
needs to reproduce it. Nondeterminism is free.

> The dense path therefore has **no notion of a global copy id** — it never
> needed one. Copies exist only as multiple index entries pointing at the
> same token row.

---

## 3. A2AV path: position-based, so order *is* the address

Now delete the dense buffer. In a2av mode, only the *needed* rows move, and
they land tightly packed:

```
rank 2's recv buffer:  [ chunk from r0 | chunk from r1 | ... | chunk from r7 ]
```

Where does the chunk from r5 start? At `recv_off[5]` = the cumsum of the
chunk sizes of r0..r4 — a number **derived from counts**, different every
iteration. And where inside rank 2's buffer is "r5's token t1"? There is no
address for it. The only possible name is *positional*:

```
"the 1st row of the (source 5 → expert e2) segment"
```

### The trap

Positional names are computed **independently on both ends of the wire**:

- **Sender r5** packs its chunk in some order and pushes the bytes.
- **Receiver r2** computes `sorted_gather_index` with pure offset
  arithmetic — it never inspects the arriving bytes.

If r5 packed `(t2, t1)` while r2's arithmetic assumed `(t1, t2)`:

- A-row "r5-segment, position 0" fetches `recv_off[5] + 0` → physically
  **t2's bytes**,
- but its output is scattered to **t1's** output row.

No out-of-bounds access, no failed check — a **silent row swap** in the MoE
output. This is precisely the failure the dense path is immune to, because
there an index entry *is* an address, and here it is only a position.

### The fix: manufacture one canonical order, everywhere

Both sides sort their view of the world by the **same composite key**:

```
( block id ) · 2³²  +  gc          block = (source, expert) group
```

- The high part groups rows into (source, expert) blocks.
- The low part — the **global copy id** — freezes the block's interior in
  ascending-gc order.

Since `scatter_index` is replicated, *every* rank can compute *every* copy's
gc locally. So sender and receiver arrive at the same interior order without
exchanging a byte, and the offset arithmetic is safe. For our e2 block:

```
A row:   0        1     |  2     |  3        4     |  5
source:  ── r0 ──────      r2       ── r5 ──────      r7
token:   t0       t3       t2       t1       t2       t0
gc:      0        7        20       43       44       56
```

Byte-for-byte identical on every run — the `atomicAdd` race of the dense
sort is *banished by construction*, because keys are unique (the gc makes
them so) and unique keys have exactly one sorted order.

Two smaller contrasts, visible in the same picture:

| | Dense AG | A2AV |
|---|---|---|
| Source order within an expert | receiver-relative (`shift_rank_to_order`: r2, r3, r0, r1, r6, r7, r4, r5) | absolute (r0..r7); the *ring put schedule* handles arrival order instead |
| Segment interiors | atomicAdd arrival order (racy, harmless) | ascending gc (deterministic, load-bearing) |

---

## 4. Recap card

| | **Dense AG path** | **A2AV path** |
|---|---|---|
| Buffer layout | static, rank-major, fixed offsets | packed, offsets = cumsums of this iteration's counts |
| A row is named by | **address**: absolute token row | **position**: "k-th row of segment (s, e)" |
| What an index entry stores | the full address of its data | an offset that is only right if both ends agree on order |
| Interior order of a segment | whatever the atomics produced — meaningless | ascending **global copy id** — it *is* the addressing |
| Needs cross-rank agreement? | no (each rank's order is private) | yes (sender pack and receiver arithmetic must match) |
| Cost of a reorder race | none (pairings stay intact) | silent row swap in the output |

**One-line takeaway:** dense AG can be sloppy about order because its
geometry is fixed and its indices carry addresses; a2av gave up the fixed
geometry to move less data, so order became the only address — and the
global copy id is the one order every rank can compute alone.
