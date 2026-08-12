# COMET MoE Layer 1 on SM80 — A Worked Example: Pure Expert Parallelism, 2 Nodes × 4 GPUs

**Scope**: a concrete, end-to-end trace of `src/moe_gather_rs` in the configuration that
mirrors the layer-0 a2av dispatch work — `W=8` across 2 nodes, one expert per rank,
`ffn_tp_size = 1`, `topk = 2`. Every number below is worked through by hand; the second
half is a deep dive on **scheduling**.

This doc is the *conceptual* companion to the code-based walkthroughs. Read it first if you
want the shape of the thing; read the others for the mechanics:

- [`comet_layer1_communication_patterns.md`](comet_layer1_communication_patterns.md) — the
  full low-level mechanics (buffers, flags, barriers, transport)
- [`comet_layer1_decompose_reschedule.md`](comet_layer1_decompose_reschedule.md) — the
  paper's *decompose*/*reschedule* as it appears in code
- [`comet_layer1_multinode_design_provenance.md`](comet_layer1_multinode_design_provenance.md)
  — where each piece of the multi-node design came from
- [`a2av_ag_walkthrough.md`](a2av_ag_walkthrough.md) — the layer-0 config this mirrors

> **The punchline up front.** Everything in this configuration is correct and well
> overlapped, but the dense reduce-scatter is solving a denser problem than we actually
> have: **75% of the bytes it moves are zeros** (§2), and the ring pays `L−1` hops to avoid
> an incast of depth `topk = 2` (§6). This is the layer-1 mirror of the gap the a2av
> dispatch closed on layer 0.

---

## 0. The configuration

| | |
|---|---|
| world size `W` | **8** — 2 nodes × 4 A100s |
| node layout | node 0 = r0–r3, node 1 = r4–r7 (contiguous, required by same-local-rank pairing) |
| experts `G` | **8**, one per rank — rank *r* owns expert *e_r*, full weights |
| `ffn_tp_size` | **1** — no K sharding at all |
| `ep_world_size` | **8** = `W` |
| topk | **2** |
| tokens | 16 for the trace below (2 per rank). Real run: `-M 40960` → 20480 tokens, 2560 per rank |

Because `T = 1`, every row the grouped GEMM emits is a **finished dot product**. There is no
K-partial anything — the first of the three reductions described in
[`comet_layer1_communication_patterns.md` §1](comet_layer1_communication_patterns.md#1-the-goal-of-the-data-movement)
is simply absent. And because each rank holds exactly one expert, `ep_nexperts = 1`, which
has real consequences for the scheduling machinery (§5.1).

What remains to be done by communication:

1. **topk combine** — token *t* has 2 expert copies. With one expert per rank, two distinct
   experts always means two *distinct ranks*, so these can never be folded locally.
2. **scatter to owner** — unchanged, and still routing-independent.

---

## 1. Where the rows are

Rank *r* owns tokens `2r, 2r+1`; expert `e_j` lives on rank *j*. Say the router produced:

| token | experts | contributions live on | owner |
|---|---|---|---|
| t0 | e1, e5 | r1 (n0), r5 (n1) | r0 |
| t1 | e0, e3 | r0, r3 (both n0) | r0 |
| t2 | e1, e2 | r1, r2 (n0) | r1 |
| t3 | e5, e6 | r5, r6 (both n1) | r1 |
| t4 | e0, e7 | r0 (n0), r7 (n1) | r2 |
| t5 | e1, e4 | r1 (n0), r4 (n1) | r2 |
| t6 | e2, e3 | r2, r3 (n0) | r3 |
| t7 | e0, e5 | r0 (n0), r5 (n1) | r3 |
| t8 | e1, e6 | r1 (n0), r6 (n1) | r4 |
| t9 | e4, e7 | r4, r7 (both n1) | r4 |
| t10 | e0, e2 | r0, r2 (both n0) | r5 |
| t11 | e5, e7 | r5, r7 (n1) | r5 |
| t12 | e1, e3 | r1, r3 (n0) | r6 |
| t13 | e4, e6 | r4, r6 (n1) | r6 |
| t14 | e2, e5 | r2 (n0), r5 (n1) | r7 |
| t15 | e0, e1 | r0, r1 (both n0) | r7 |

`splits = [5, 6, 4, 3, 3, 5, 3, 3]` — so `M_this_ep` per rank is 5, 6, 4, 3, 3, 5, 3, 3.
**A 2× spread**, and with one expert per rank there is nothing to average it out. This is the
dominant scheduling problem in this config (§5.7).

Cases worth noting in the table:

- **t3, t9** — both experts on the same *node*, so those tokens need no network hop at all.
- **t10** — both contributions on node 0, but the owner is r5 on node 1: must cross the
  network, but its two pieces can be summed before they leave.
- **t1, t2, t6, t9, t11, t13** — the owner is itself a contributor.

---

## 2. The local fold

Each rank asks "what can I contribute to each of the 16 tokens?" With one expert, the answer
is: a real row for the tokens routed to my expert, zero for everything else.

```
        t0  t1  t2  t3  t4  t5  t6  t7  t8  t9 t10 t11 t12 t13 t14 t15
C_0(e0)  .   ●   .   .   ●   .   .   ●   .   .   ●   .   .   .   .   ●     5 of 16
C_1(e1)  ●   .   ●   .   .   ●   .   .   ●   .   .   .   ●   .   .   ●     6
C_2(e2)  .   .   ●   .   .   .   ●   .   .   .   ●   .   .   .   ●   .     4
C_3(e3)  .   ●   .   .   .   .   ●   .   .   .   .   .   ●   .   .   .     3
C_4(e4)  .   .   .   .   .   ●   .   .   .   ●   .   .   .   ●   .   .     3
C_5(e5)  ●   .   .   ●   .   .   .   ●   .   .   .   ●   .   .   ●   .     5
C_6(e6)  .   .   .   ●   .   .   .   .   ●   .   .   .   .   ●   .   .     3
C_7(e7)  .   .   .   .   ●   .   .   .   .   ●   .   ●   .   .   .   .     3
```

32 filled cells out of 128. That is the **useful fraction**, and with one expert per rank it
has a closed form with no dependence on the routing distribution at all:

```
useful fraction = topk / W = 2 / 8 = 25%
```

(Distinct experts ⇒ distinct ranks ⇒ every token generates exactly `topk` nonzero rows, with
zero variance. This sits exactly at the structural ceiling `min(topk, E)/E`, so no scheduling
or placement change recovers it — only a sparse transport.)

**Three quarters of what the ring is about to move are those dots.**

The answer is `final = C_0 + … + C_7`, with rows `2s, 2s+1` going to rank *s*. Same-shaped
matrix on every rank, elementwise sum, each rank keeps a row block: a **reduce-scatter**.
Nothing is materialized — the fold is `routing_idx` lookups plus the `ep_m_start/ep_m_end`
range test happening inline at load time
(`topk_gather_rs_v2.cu:847-850`; see
[communication patterns §5.1](comet_layer1_communication_patterns.md#51-why-a-ring-and-why-this-segment-rotation)).

---

## 3. The hierarchical ring

Segments: `seg_s = {t_2s, t_2s+1}`, owned by rank *s*. **Node 0 owns segs 0–3 (group g=0),
node 1 owns segs 4–7 (group g=1).**

The ring is `L = 4` stages over *local* ranks, and it runs **once per segment group**, so
twice. Node 0 does g=1 first (remote), then g=0.

Node 0's ring for **group 1** (node 1's segments), `segment = 4 + (stage + local_rank + 1) % 4`,
pushing to `(local_rank − 1 + 4) % 4`:

```
            stage 0        stage 1        stage 2        stage 3
r0:      seg5  → r3     seg6  → r3     seg7  → r3     seg4 → staging
r1:      seg6  → r0     seg7  → r0     seg4  → r0     seg5 → staging
r2:      seg7  → r1     seg4  → r1     seg5  → r1     seg6 → staging
r3:      seg4  → r2     seg5  → r2     seg6  → r2     seg7 → staging
```

Follow **seg4** (tokens 8, 9): `r3 → r2 → r1 → r0`, each hop adding its own contribution.
Node 0's contributions to seg4 are: t8 gets `e1` from r1; t9 gets *nothing* (both its experts
are on node 1). So the chunk r0 stages is `[e1(t8), 0]` — half zeros, shipped anyway.

Two structural points visible in that table:

- **Read any column**: at each stage the four ranks work four *different* segments, so all
  four NVLink directions carry traffic simultaneously. That is what the `+ local_rank` term
  in the rotation buys.
- **At stage 3, rank `l` holds the node's total for seg `4+l`.** The put target is
  `pe = g·L + local_rank = 4 + l`, which is exactly the rank that *owns* seg `4+l`. The
  same-local-rank pairing is not merely NIC-rail spreading — it lands each chunk on the
  precise GPU that needs it, with no intra-node forwarding on the receive side. The rotation
  formula and the pairing rule have to be chosen together for this to line up.

Node 0 then runs group 0 identically for its own segments, except stage 3 writes straight
into `output` (`last_round_dst_ptr`, `topk_gather_rs_v2.cu:297-315`).

---

## 4. The cross-node hop and final assembly

Node 1 runs g=0 first (remote), then g=1. Its own-group ring produces, for seg4:
`e6(t8)` from r6, and `e4(t9) + e7(t9)` from r4 and r7 — landing in r4's `output`.

Final assembly on r4: `output` already holds `[e6(t8), e4+e7(t9)]` from the local ring;
`internode_reduce_gather_rs` adds the arrived `[e1(t8), 0]` from node 0. Result:

```
t8 = e1 + e6      ✓   (t8 → e1, e6)
t9 = e4 + e7      ✓   (t9 → e4, e7 — both node-local, but a zero chunk shipped anyway)
```

**Traffic per rank**, real numbers (20480 tokens, N=5120, fp16, `n_split=5`):

| path | per split | total |
|---|---|---|
| NVLink (6 peer pushes: `L−1` per group × 2 groups) | 6 × 5.24 MB = 31.4 MB | **157 MB** |
| network (1 put per remote node per split) | 5.24 MB | **26 MB** |

Of that 157 MB of NVLink traffic, only ~39 MB carries nonzero data.

---

## 5. Scheduling

This is the part that determines whether any of the above overlaps, and this config stresses
it in specific ways.

### 5.1 The GEMM reschedule and its degeneracy at one expert per rank

`make_workspace_kernel` (`workspace_helper.cu:80-102`) expands the expert GEMMs into
`n_split ×` as many column-window subproblems, indexed **split-major**:

```cpp
int sid = i / problem_per_split;    // problems of split 0 come first
int sr  = i % problem_per_split;
int eid = sr % args.ep_nexperts;
problem_sizes[i] = {Mi, new_N, K};                     // new_N = N / n_split
ptr_D[i] = ptr_with_offset(output[gid], (M_acc*N + sid*new_N) * elem);
ldd[i]   = LayoutD::packed({Mi, N}).stride(0);         // stride = FULL N
```

The CUTLASS visitor is built `kDeviceOnly`, so it walks problems in array order — meaning
**the GEMM finishes all of split 0 before starting split 1**, which is exactly the order the
consumer and the network ladder want. (Full treatment:
[decompose/reschedule §3.1](comet_layer1_decompose_reschedule.md#31-problem-order--split-major).)

Now the config-specific part: `problem_per_split = ep_nexperts × num_groups = 1 × 1 = 1`.
**Each split is a single subproblem.** Consequences:

- The three-level completion cascade (tile → subproblem → split,
  [communication patterns §4.2](comet_layer1_communication_patterns.md#42-completion-detection-tile--problem--split-in-three-atomics))
  collapses. The middle counter needs exactly one increment, so in practice the last tile of
  the split's single GEMM flips the split flag directly.
- The `non_empty_problem_count` device-side computation — which exists so zero-token experts
  do not leave a flag permanently one short — becomes a binary question: does *this rank's
  one expert* have any tokens? If not, `M_this_ep == 0`, the op takes the `barrier.fill_(1)`
  escape (`gemm_grouped_v2_gather_rs.cc:822`) and releases the consumer manually.
  **At `W=8` with 8 experts this is a live scenario** — one unpopular expert idles an entire
  GPU's GEMM. It is handled, but it essentially never fires in the TP config the code was
  designed for.
- Each subproblem is `[M_this_ep, 1024, K]` with `M_this_ep` as small as 3 rows in the toy
  example. Real runs are larger, but the point stands: **skinny, imbalanced GEMMs**, a poor
  fit for a persistent grouped kernel sized for 105 SMs.

### 5.2 The resource partition

Overlap here is SMs-against-SMs, split statically:

```cpp
int rs_num_blocks = get_int_from_env("FLUX_RS_BLOCKS", 3);   // :99-103
...
.sm_margin = sm_margin + get_rs_threadblock_count();          // :811 — GEMM yields 3 SMs
```

Three streams, each owning a distinct resource:

- **main stream** — grouped GEMM on 105 of 108 SMs
- **`gather_rs_stream`** — the consumer, 3 blocks × 800 threads (768 workers + a 32-thread
  sync warp)
- **`internode_stream`** — the host put ladder, no SMs at all

Bracketed by `group_barrier.barrier_all` before and after, with `gemm_start_event` gating the
consumer (it is a spin-waiting kernel, so it must not launch before the flags it polls are in
a defined state) and `gather_rs_done_event` gating the close.

The static partition matters more than it looks: the consumer is spin-waiting on flags that a
*peer's* consumer progress depends on. If the GEMM were allowed to occupy all 108 SMs, the
consumer could be descheduled while a peer waits on its ring push — a distributed stall caused
by a purely local scheduling decision. (See
[decompose/reschedule §4](comet_layer1_decompose_reschedule.md#4-horizontal-fusion-how-producer-and-consumer-share-the-gpu).)

### 5.3 The consumer loop nest

```cpp
for (int sid = 0; sid < n_split; sid++) {
  Barrier::wait_eq(barrier[local_rank], threadIdx.x, sid, 1);      // producer gate
  for (int g_iter = 0; g_iter < nnodes; g_iter++) {
    int g = (node_idx + 1 + g_iter) % nnodes;                      // REMOTE FIRST
    for (int stage = 0; stage < local_world_size; stage++) {
      for (int blk = blockIdx.x; blk < m_tiles*n_tiles; blk += gridDim.x) { ... }
    }
    // last block of this (g,sid) flips group_flags → releases a network put
  }
}
```

Every level of that nest is a deliberate scheduling choice:

- **`sid` outermost** — matches producer emission order, so the consumer never waits on a
  split the GEMM has not prioritized.
- **One `wait_eq` per split, on one local word.** The consumer touches the producer exactly
  `n_split` times total. Everything else is peer synchronization.
- **`g` remote-first** — the network is the long pole, so the chunk that needs a NIC round
  trip is produced first and its put overlaps the own-node group's ring work. This is the
  *opposite* rotation from layer 0's all-gather, which walks own-node first because there the
  consumer wants local data earliest. Same loop, inverted phase, each matching its layer's
  overlap direction (provenance:
  [§5.3](comet_layer1_multinode_design_provenance.md#53-node-rotation-outer-loop-and-same-local-rank-pairing--mirrored-from-the-layer-0-v3-all-gather-already-ported-to-layer-0-v2)).
- **`stage` inside `g`** — all 4 ranks advance stages in lockstep on different segments, so
  all 4 NVLink directions stay busy.
- **tiles innermost, grid-strided** — with 20480 tokens, `m_tiles = 2560/128 = 20` and
  `n_tiles = 1024/1024 = 1`, so 20 tiles spread over 3 blocks (~7 each) per stage. Readiness
  is checked **per tile**, not per stage:
  `WorkerBarrier::wait_eq(tile_barrier_ptr, tid, tile_idx, 1)`. A rank starts consuming
  whichever tiles have landed instead of waiting on a whole-segment barrier — this is the
  mechanism that absorbs producer skew, and in this config it is doing heavy lifting (§5.7).
- **Flag-setting is offloaded to the sync warp** (`topk_gather_rs_v2.cu:455-462`): workers hit
  `FullBarSync` after their stores, then thread 0 of the 32-thread warp does the system-scope
  release on the *destination* rank's array. Workers proceed to the next tile without waiting
  for that release to retire.

### 5.4 The pre-enqueued network ladder

```cpp
for (int sid = 0; sid < n_split; sid++)
  for (int gi = 0; gi < nnodes - 1; gi++) {
    int g = (node_idx + 1 + gi) % nnodes;                 // same order the kernel produces
    CUStreamWaitValue(internode_stream, group_flags + idx, 1, GEQ);
    nvshmemx_putmem_signal_nbi_on_stream(..., pe = g*L + local_rank, internode_stream);
  }
```

All of this is queued **before the consumer kernel produces its first chunk**. No host polling
thread, no device-side verb (Slingshot is host-proxied — a device put would detour through the
same proxy while burning consumer threadblocks). Each put fires the instant its flag flips.

The ordering constraint is the subtle part: **ladder order must equal kernel production
order.** Both are remote-groups-first, ascending split. If they diverged, a wait on a
not-yet-ready chunk would head-of-line-block a chunk that *was* ready. On 2 nodes there is
only one remote group so the inner loop is trivial, but the invariant is what makes it extend
to 4+ nodes.

Receive side stays on the consumer stream: `signal_wait_until(CMP_GE, run_id_)` per split, then
an accumulate kernel. Signals are monotonic and never reset — resetting a symmetric word would
need its own cross-node rendezvous (provenance:
[§5.6](comet_layer1_multinode_design_provenance.md#56-monotonic-run_id-epoch-signals--new-shared-with-the-layer-0-a2av-dispatch-same-commit)).

### 5.5 The complete timeline

```
main stream (105 SM)      gather_rs_stream (3 SM)          internode_stream
────────────────────      ───────────────────────          ────────────────
barrier_all; events ────► (wait gemm_start_event)          (wait staging_reset)
GEMM split 0              wait_eq flag[0] … spin
  cascade ⇒ flag[0] ────► ring g=1, split 0 (4 stages)
GEMM split 1                └ group_flags[1,0]=1 ────────► put chunk(1,0) ⇒ signal
                          ring g=0, split 0 → output
  cascade ⇒ flag[1] ────► ring g=1, split 1 ─────────────► put chunk(1,1) ⇒ signal
GEMM split 2              ring g=0, split 1
  …                       signal_wait(node1, split 0) ◄──── remote put lands
                          internode_reduce += → output
gather_rs_done ◄───────── done
barrier_all; flags.zero_()
```

Three overlaps stacked:

1. split *s*'s ring hides behind split *s+1*'s GEMM;
2. the remote group's put hides behind the own group's ring;
3. split *s*'s network arrival hides behind split *s+1*'s ring.

### 5.6 Tuning knobs

**`n_split`** is clamped, not free (`gemm_grouped_v2_gather_rs.cc:527-534`):

```cpp
int n_split_fixed(int n_split, int n_dim) {
  if (n_dim / n_split % kTileSizeN != 0) { FLUX_CHECK_DIV(n_dim, kTileSizeN);
                                           n_split = n_dim / kTileSizeN; }
  return n_split;
}
```

`kTileSizeN = 1024` (`topk_gather_rs_v2.cu:700-703`, alongside `kTiledM = 128`). So for
`N = 5120`, `N/n_split` must be a multiple of 1024 → only **`n_split ∈ {1, 5}`** survive;
anything else is silently snapped to 5. Worth knowing before spending time sweeping it.
Raising `n_split` starts communication earlier and shortens the drain tail, but narrows each
GEMM subproblem toward `[M_this_ep, 1024, K]` — and with `ep_nexperts = 1` and small
`M_this_ep`, those are already thin.

**`FLUX_RS_BLOCKS`** (default 3) trades GEMM throughput against ring drain latency. The
consumer is bandwidth-bound, so 3 blocks usually saturate NVLink — but in *this* config the
consumer's work is fixed while the GEMM's is small and lopsided, which shifts the balance. If
the ring is on the critical path rather than the GEMM, this is the first thing to move.

### 5.7 What pure EP does to the schedule

Two effects specific to pure EP that do not appear in the TP config the scheduler was tuned
for:

**Producer skew is maximal, consumer load is uniform.** `M_this_ep` ranges 3–6 in the toy
(2×); real routing can be worse. But the ring's per-stage work is
`ntokens_per_rank × N_split` — **identical on every rank, independent of routing**. So ranks
reach `flag[sid]` at very different times while all needing to do the same amount of ring
work. Since the ring is a chain (`r_l` pushes to `r_{l−1}`, and every rank sits in every
segment's chain at some stage), a slow producer injects a bubble into every segment, just at a
different stage each time. Per-tile flags and per-split gating limit the damage to one column
strip at a time, but they do not eliminate it. This is also why the empty-expert path
(`M_this_ep == 0` → `barrier.fill_(1)`) exists and why it is reachable here.

**The ring's premise is weakest here** — see §6.

---

## 6. Why the ring is the wrong shape here

The ring's two justifications
([communication patterns §5.1](comet_layer1_communication_patterns.md#51-why-a-ring-and-why-this-segment-rotation))
are: (a) a naive gather-to-owner creates a `W−1`-deep **incast** on the owner GPU, and (b) the
owner alone would perform all `W−1` additions.

With `topk = 2` and one expert per rank, **there is no incast to avoid.** Token *t* has
exactly 2 contributions. Its owner receives at most 2 messages — not 7. Total addition work is
1 add per token, not 7. The schedule is doing 6 NVLink pushes and 8 stage-synchronizations per
split to solve a problem whose natural depth is 2, and 75% of the bytes it moves are zeros.

Stated as a scaling law: the dense ring moves `ntokens × N × (W−1)` bytes regardless of
routing, while the genuinely-needed traffic is `ntokens × topk × N × (W−1)/W`. The ratio is
`topk/W`, so **the overhead grows linearly with world size while the useful content stays
fixed** — 4× excess at 2 nodes, 8× at 4 nodes.

The dense reduce-scatter is the *right* algorithm for tensor parallelism, where every rank
genuinely contributes to every token (useful fraction 100%, incast depth genuinely `W`). Pure
EP with `topk ≪ W` is its maximally adversarial input.

The mirror-image fix is an **a2av combine**: each rank sends only its nonzero contribution
rows, directly to the owning rank; each owner adds the `≤ topk` arrivals locally. Depth 1
instead of `W−1`, `topk/W` of the bytes, same number of adds. Two notes on feasibility:

- **Easier than layer 0 in one respect.** The op contract already requires `routing_idx` and
  `splits` to be **global on every rank** regardless of EP
  (`gemm_grouped_v2_gather_rs.cc:615-619`). Every rank can therefore compute, with no
  communication, which tokens it contributes to and who owns each — so send/recv counts and
  offsets are derivable locally. No `cnt[s][e]` metadata-exchange phase would be needed.
- **Harder in the respects already known.** Variable-length messages break the fixed-size
  staging slots and the fully pre-enqueued put ladder (§5.4), and would need the mirror of the
  "global copy id" agreement from
  [`a2av_ag_walkthrough.md`](a2av_ag_walkthrough.md) — sender and receiver agreeing on queue
  order without talking. That doc's answer transfers directly, since it is the same replicated
  routing table read from the other end.

---

## Run it

```bash
source ./module.sh
salloc -A m4243_g -q interactive -C gpu -N 2 --gpus-per-node=4 -t 30
export NVSHMEM_SYMMETRIC_SIZE=4G   # staging/signals live on the symmetric heap
srun --nodes=2 --ntasks-per-node=1 ./launch.sh \
    test/python/moe_gather_rs/test_moe_gather_rs.py -M 40960 -T 1 -E 8 -G 8 --topk 2
```

Constraints the ctor enforces: token counts divisible by `world_size · topk`, `max_m/topk`
divisible by `world_size`, node-contiguous rank layout, `T · E == world_size`.
`do_all_reduce` and `use_read_mode` are single-node only.
