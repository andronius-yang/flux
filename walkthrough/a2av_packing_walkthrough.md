# A2AV Dispatch: Data Packing & Reordering, Step by Step

A hand-holding walkthrough of how the raw-alltoallv dispatch mode of
`GemmGroupedV2AGScatterOp` turns an arbitrarily-ordered token shard into
(1) contiguous per-destination wire messages and (2) an expert-major layout
the grouped GEMM can consume — using nothing but **one sort key** and **a
handful of prefix sums** over a single count table.

All code references are to
`src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc` unless noted.

---

## 0. The cast of characters

| Symbol | Meaning | Value in our running example |
|---|---|---|
| `W` | world size = number of GPUs | **2** (rank 0, rank 1) |
| `nexperts` | total experts in the model | **4** (e0, e1, e2, e3) |
| `E` (`ep_nexperts`) | experts *per rank* = `nexperts / W` | **2** — rank 0 owns {e0, e1}, rank 1 owns {e2, e3} |
| `ep_start` | first expert id I own | 0 on rank 0, 2 on rank 1 |
| `topk` | experts each token is routed to | **2** |
| `tokens_per_rank` | tokens in my shard `T_s` | **2** |
| `cpr` (`copies_per_rank`) | `tokens_per_rank × topk` = row-copies I send out | **4** |
| `s` | a **source** rank (who produced a token) | 0 or 1 |
| `d` | a **destination** rank (who owns the expert); `d = e / E` | e0, e1 → 0; e2, e3 → 1 |
| `e_loc` | expert id local to its owner: `e − ep_start` | e2 is `e_loc = 0` on rank 1 |

**Ground-truth routing.** Rank 0 has tokens **a0, a1**; rank 1 has **b0, b1**:

```
a0 → {e1, e2}      b0 → {e2, e0}
a1 → {e0, e1}      b1 → {e1, e3}
```

A **copy** is one (token, topk-slot) pair. In token order
(copy index `c = token·topk + slot`), each rank's four copies are:

```
rank 0:  c0=(a0,e1)  c1=(a0,e2)  c2=(a1,e0)  c3=(a1,e1)
rank 1:  c0=(b0,e2)  c1=(b0,e0)  c2=(b1,e1)  c3=(b1,e3)
```

The shard really is in "arbitrary" order — e1, e2, e0, e1 interleaved. That
mess must be organized **twice**: once for the *wire* (group by destination
GPU) and once for the *GEMM* (group by expert).

Two count tables fall straight out of the routing; *everything* else in this
walkthrough is a prefix sum over them.

**`cnt[s][e]`** — copies from source `s` for expert `e` (`splits_per_source`):

|        | e0 | e1 | e2 | e3 |
|--------|----|----|----|----|
| s = 0  | 1  | 2  | 1  | 0  |
| s = 1  | 1  | 1  | 1  | 1  |

**`C[s][d]`** — the chunk matrix: collapse `cnt` over each owner's experts:

|        | d = 0 | d = 1 |
|--------|-------|-------|
| s = 0  | 3     | 1     |
| s = 1  | 2     | 2     |

Column sums of `cnt` are the classic `splits = [2, 3, 2, 1]`, and their
running total `expert_base = [0, 2, 5, 7]` says where each expert's block
starts in the *global* expert-sorted layout (8 rows total).

### The three row orders

Every piece of cumsum/argsort machinery in this file is a conversion between
two of exactly three orders:

1. **Token order** — `T_s` as the router produced it; experts interleaved.
2. **Wire order** — destination-major, so each `s→d` message is contiguous.
3. **GEMM order** — expert-major, so each grouped-GEMM problem is contiguous.

---

## 1. What `scatter_index` already gives you for free

The router sorted all 8 global copies by expert (ties broken by global copy
id). That global expert-sorted order is:

```
row: 0    1    2    3    4    5    6    7
     a1   b0 | a0   a1   b1 | a0   b0 | b1
     ─ e0 ──   ──── e1 ───   ── e2 ─   e3
```

`scatter_index[token][slot]` stores each copy's row number in that layout
(its `flat_dst`):

```
a0 = [2, 5]    a1 = [0, 3]    b0 = [6, 1]    b1 = [4, 7]
```

**Step 1 — decode (`a2av_stage1_impl`, `:698`)** just reads this backwards,
per copy, in one elementwise kernel — *no sorting*:

- `e` = which expert block does my `flat_dst` fall into? (compare against
  `expert_base`)
- `d = e / E`
- `s` = global copy id `/ cpr`

Example: rank 1's copy c2 has `flat_dst = 4` → falls in `[2, 5)` → e1 →
`d = 0` → "this copy goes to the other GPU."

---

## 2. Sender pack: one sort key, one gather

Take rank 1. Its copies must be grouped **destination-major** so each `s→d`
message is one contiguous put. Build one integer per copy
(`pack_key = e·cpr + c`, emitted by stage 1, sorted at `:725`):

```
c0 (e2): 2·4+0 =  8        sorted keys:   1, 6, 8, 15
c1 (e0): 0·4+1 =  1    →   sorted copies: c1(e0,b0), c2(e1,b1), c0(e2,b0), c3(e3,b1)
c2 (e1): 1·4+2 =  6
c3 (e3): 3·4+3 = 15
```

Why this key works:

- **high part `e`** — experts become the major criterion, and since each
  owner's experts are contiguous ids, ascending expert ⇒ ascending
  destination;
- **low part `c`** — ties break in **copy-index order**. *Remember this
  tie-break; it is the keystone of step 4.*

One `argsort` + one `index_select` later, rank 1's send buffer is:

```
[ b0, b1 | b0, b1 ]
  ── d=0 ─  ── d=1 ─      (chunk C[1][0]: e0-row, e1-row; chunk C[1][1]: e2-row, e3-row)
```

Where each chunk starts = **row-wise cumsum of `C[1][·]`** →
`send_off = [0, 2]` (`:884-887`).

> This is the **only physical data reorder the sender ever performs**.

---

## 3. The wire: chunks land at precomputed offsets, nothing is reordered

Every rank knows the full `C[s][d]` matrix, so receiver offsets need no
negotiation: the chunk from source `s` lands at
`recv_off[s]` = **column-wise cumsum of `C[·][d]`** (`:888-894`), delivered
as one contiguous `nvshmemx_putmem_signal` per destination.

Rank 0 (owns e0, e1) receives `C[0][0] = 3` rows from itself and
`C[1][0] = 2` rows from rank 1, so `recv_off = [0, 3]` and its 5-row recv
buffer ends up:

```
recv row:  0        1        2      | 3        4
           a1(e0)   a0(e1)   a1(e1) | b0(e0)   b1(e1)
           ────── from s = 0 ──────   ─ from s = 1 ─
```

Read the structure: **source-major outside, expert-order inside** each chunk
— because that is exactly how each sender packed it. Sorted, but the wrong
way around for compute.

---

## 4. GEMM side: a *virtual* transpose via two cumsum tables

The grouped GEMM wants one contiguous row block per expert:
**expert-major outside, source-order inside**:

```
A row:   0        1      | 2        3        4
         a1(s0)   b0(s1) | a0(s0)   a1(s0)   b1(s1)
         ────── e0 ─────   ───────── e1 ──────────
```

Compare with the recv buffer: **same 5 rows, blockwise-transposed** (blocks =
`(s, e_loc)` groups of varying size). Instead of moving data, we build
`sorted_gather_index[i]` = "A-row `i` lives at recv-row …", here
`[0, 3, 1, 2, 4]` — the GEMM's A-loader gathers through it.

### The keystone invariant

Both layouts sort *within* a block `(s, e_loc)` by the **same copy index** —
the sender's pack key used `c`, and the A-order key uses `c` too. Look at
block `(s0, e1)`: it is `a0 then a1` in the recv buffer **and** `a0 then a1`
in A-order. So the two layouts differ only in *where blocks start*, never in
interior order — and block starts are just the same counts cumsum'd in two
different nesting orders:

| Table | Nesting | Block starts (example) |
|---|---|---|
| `offA[g]` | A-order: **e_loc outer, s inner** → (e0,s0), (e0,s1), (e1,s0), (e1,s1) | `[0, 1, 2, 4]` |
| `offR[h]` | recv-order: **s outer, e_loc inner** → (s0,e0), (s0,e1), (s1,e0), (s1,e1) | `[0, 1, 3, 4]` |
| `offR_of_A[g]` | `offR` re-indexed into A's block order | `[0, 3, 1, 4]` |

Then for any A-row `i`: find its block `g` with one `searchsorted` into the
inclusive cumsum `cumA = [1, 2, 4, 5]`, and

```
gather[i] = offR_of_A[g] + (i − offA[g])
```

Check `i = 3` (second e1 row from s0): `g = 2`, gather = `1 + (3 − 2)` =
**2** → recv row 2 = a1(e1) ✓.
Check `i = 1`: `g = 1`, gather = `3 + (1 − 1)` = **3** → b0(e0) ✓.

That is the whole "complicated math":

- **metadata path** (`:761-766`): the three tables are built by tiny host
  loops (`:623-645`) from `cnt[s][e]`; the transpose costs **zero sorts**.
- **derive path** (fallback, `:797-824`): the same mapping is derived on-GPU
  with two argsorts over packed keys `(block_id)·2³² + copy_id` — same idea,
  keys instead of tables.

---

## 5. Two leftovers, briefly

- **`sorted_scatter_index[i]`** — where GEMM output row `i` goes in the final
  expert-sorted output. Reuses the router's work: it is just
  `flat_dst − expert_base[ep_start]` (`:769`). No new sort.
- **`sorted_splits_cumsum[e_loc][s]`** (`:647-653`) — per-expert running
  totals over sources (e1: s0 ends at 2, s1 at 3). Tells the tile scheduler
  which A-rows of each expert depend on which source's arrival — consumed by
  the static ring schedule and the per-tile signal spin.

---

## Recap card

| Table | Built by | Answers |
|---|---|---|
| `expert_base` | cumsum of `splits` | which expert does `flat_dst` belong to; where does e's global block start |
| `send_off` | row-cumsum of `C[s][·]` | where does my chunk for destination `d` start in my send buffer |
| `recv_off` | column-cumsum of `C[·][d]` | where does source `s`'s chunk land in my recv buffer |
| `offA` / `cumA` | cumsum of `cnt` blocks, (e, s)-nested | where does block (e, s) start in GEMM order |
| `offR_of_A` | cumsum of `cnt` blocks, (s, e)-nested, re-indexed | where does that same block start in recv order |

**One real sort** (sender pack). **One shared tie-break** (copy index) that
freezes block interiors. Everything else is the same little count table
cumsum'd in different nesting orders.
