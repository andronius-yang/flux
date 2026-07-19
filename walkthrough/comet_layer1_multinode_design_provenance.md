# COMET MoE Layer 1 on SM80 — Multi-Node Design Provenance: What Existed, What Was Built, and What Was Mirrored From Where

**Scope**: the multi-node (`nnodes > 1`) path of `src/moe_gather_rs` — the hierarchical
intra-node ring + host-staged NVSHMEM `putmem_signal` design added to
`GemmGroupedV2GatherRSOp` in commit `7a1c4ac`. Companion docs:
[`comet_layer1_communication_patterns.md`](comet_layer1_communication_patterns.md)
(the full low-level mechanics of the layer-1 communication and reduction, §5–§6 of which
describe the multi-node hops in detail) and
[`comet_layer0_communication_patterns.md`](comet_layer0_communication_patterns.md)
(layer 0, whose §4.4 documents the multi-node all-gather this design partially mirrors).

This doc answers a different question than the companions: **not "how does it work" but
"where did each piece come from."** Upstream Flux/Comet shipped *no* multi-node
implementation of the MoE layer-1 reduce-scatter — on any architecture. The design that
now runs on Perlmutter was therefore assembled: some elements are direct ports, some are
upstream idioms re-hosted in a new position, and some are new. Knowing which is which
tells you where the design's assumptions were inherited (and validated elsewhere) versus
where they were chosen for this machine specifically.

---

## 1. The starting inventory: what upstream actually provided

The honest baseline, verifiable at the pre-port commit (`git show 7a1c4ac^:<file>`):

| Upstream component | Multi-node? | What it contributes |
|---|---|---|
| `moe_gather_rs` V2/sm80 (`topk_gather_rs_v2.cu`) | **No** — flat ring over `world_size`, P2P load/store only | the intra-node ring mechanics, per-tile flags, producer flag cascade |
| `moe_gather_rs` V3/sm90 (`gemm_grouped_v3_gather_rs.cc`) | **No** — peer pointers via `nvshmem_ptr` (`:199,211`), which is NULL across nodes | nothing portable (see §2) |
| `moe_ag_scatter` V3/sm90 all-gather (`gemm_grouped_v3_ag_scatter.cc:172`) | **Yes** — hierarchical node-rotation loop | the node-rotation outer loop, same-local-rank pairing, symmetric-heap staging (already ported to layer 0 V2) |
| dense `gemm_rs` sm80 (`reduce_scatter_kernel.hpp`) | **Yes** — but a very different design (see §6) | the "reduce inside the node first, then one inter-node exchange" hierarchy; host-side `cuStreamWaitValue32` gating (`:2096,2231`) |
| `src/coll` DisScatter / all2all (`all2all_impl.cu:99`) | sm90-oriented | the `putmem_signal` payload+doorbell fusion idiom (device-issued there) |
| `src/coll/all_gather_op.cc:522` | intra-node | `CUStreamWaitValue` as a stream-side wait on a device flag |

Two pieces of upstream evidence that the layer-1 multi-node gap was real and known:

- The V2/V3 ops had **no `nnodes` parameter at all** — `grep -rn nnodes src/moe_gather_rs`
  at `7a1c4ac^` returns nothing. Layer 0's V3 op, by contrast, carried a working
  `tp_env.nnodes` loop.
- `epilogue_gather.hpp:602-603` contains a **commented-out** `nvshmem_putmem_nbi` in the
  V3 gather epilogue — upstream tried (or at least sketched) pushing layer-1 results to
  remote peers directly from the epilogue, and abandoned it. The shipped V3 path writes
  through `nvshmem_ptr`-mapped peer pointers instead, which only exist where P2P
  load/store exists: inside a node.

So the porting situation was asymmetric between the two layers:

> **Layer 0**: a correct multi-node all-gather existed in the same op family (V3), one
> architecture up. Porting = re-hosting the V3 host-side loop into the V2 op.
> **Layer 1**: no multi-node reduce-scatter existed in the op family at all. Porting was
> impossible; the design had to be composed from adjacent precedents.

---

## 2. Why nothing could simply be ported

**V3 layer 1 was not a multi-node design one architecture up.** Its communication layer is
a pointer table: `output_scatter_ptrs[i] = nvshmem_ptr(output_buffer.data_ptr(), i)`
(`gemm_grouped_v3_gather_rs.cc:211`). `nvshmem_ptr` returns a usable address only for PEs
reachable by load/store (NVLink peers); for a remote-node PE it returns NULL. The entire
V3 consume path — epilogue threads storing reduced packs straight into peer memory —
therefore has no cross-node analogue. There was nothing to port down.

**Layer 0's ported answer solves the wrong problem.** The layer-0 multi-node AG
(walkthrough §4.4) is a *replication* collective: the same bytes must land everywhere, so
pull-based `getmem` with same-local-rank pairing and NVLink redistribution is natural, and
the consumer (the GEMM) only ever *waits* for data. Layer 1 is a *reduction* collective:
every rank holds different partial sums that must be combined, the bytes shrink as they
move (aggregation), and the "transport" must interleave with arithmetic — each hop's
payload is *produced* by the previous hop's adds. A pull model would make every consumer
wait on remote arithmetic it cannot see; the data dependencies force a producer-driven,
push-shaped pipeline. Direction, operator, and driver all invert.

**The dense `gemm_rs` multi-node design does not fit the MoE consumer.** It exists
(sm80, `nnodes > 1`), and its top-level shape — reduce within the node, then exactly one
inter-node exchange per segment — is the right shape (and is retained, §5). But its
mechanics assume the dense case: output tile ↔ communication tile is 1:1, so the whole
reduce-scatter runs as one cooperative-launch kernel spanning the GPU, with per-segment
`gemm_done`/`copy_done` flag arrays, and the inter-node hop is a **blocking, device-issued
`nvshmem_getmem` inside the kernel** (`reduce_scatter_kernel.hpp:2023`) doorbelled by
`nvshmem_int_atomic_set` (`:2009`). In the MoE op the consumer must run *concurrently*
with a persistent grouped GEMM on 3 reserved SMs (`FLUX_RS_BLOCKS`) — it cannot be a
whole-GPU cooperative kernel, and a device-side blocking pull inside a 3-block kernel
would stall the only threadblocks doing reduction work while the NIC round-trips.
On Perlmutter there is a second, machine-level reason to keep NVSHMEM verbs off the
device entirely: Slingshot's libfabric transport is **host-proxied** (no IBGDA), so a
device-issued put/get is just a slower detour through the same host proxy that a
host-issued `_on_stream` verb reaches directly.

---

## 3. The design, in one pass

(Full mechanics with code excerpts: companion doc §5–§6. This is the shape, so the
provenance table in §5 has referents.)

```
per split sid (producer flag cascade releases splits in order):
  for g in (node_idx+1, node_idx+2, …, node_idx) mod nnodes:      # remote groups FIRST
    L-stage intra-node ring over local ranks (L=4):
      stage 0: gather(routing_idx) + topk-sum my contribution, push to ring predecessor
      stage k: wait per-tile flag; add my contribution onto arrived partial; push on
      last stage, g == node_idx:  write straight into `output` (scatter implicit)
      last stage, g != node_idx:  write into staging_send[g][sid]   (node-level sum done)
    kernel→host: counter cascade; last block sets group_flags[g][sid]  (release)
  host ladder (internode_stream, pre-enqueued):
      CUStreamWaitValue(group_flags[g][sid]) →
      nvshmemx_putmem_signal_nbi_on_stream(staging_recv slot on node g, signal := run_id)
      to the same-local-rank PE on node g
  receive (consumer stream): signal_wait_until ≥ run_id per remote node, then
      internode_reduce adds staging_recv[m][sid] into `output`
```

Key numbers: cross-node bytes per output byte = `(nnodes−1)/nnodes` of one copy — each
node ships one *already-summed* contribution per remote segment group, so wire traffic is
cut by `L=4×` versus shipping per-GPU partials, and the flat ring's `world_size−1`
network-latency hops per segment are collapsed to exactly one put per (node pair, split).

---

## 4. What changed in the code (the port, mechanically)

All in `7a1c4ac`, ~430 lines across 8 files:

- **Kernel** (`topk_gather_rs_v2.cu`): the flat ring became hierarchical. Upstream:
  `segment = (stage + rank + 1) % world_size`, push to `(rank−1) % world_size`. Port
  (`:370-372`): `segment = group_idx·L + (stage + local_rank + 1) % L`, push to
  `(local_rank−1+L) % L` — same rotation, re-indexed from the world to the node, wrapped
  in a new outer loop over segment groups walked remote-first (`:650`). New:
  `last_round_dst_ptr` (`:297-315`) forking the final hop between `output` and
  `staging_send`, and the kernel→host `group_counters`/`group_flags` handshake
  (`:667-677`).
- **Op** (`ths_op/gemm_grouped_v2_gather_rs.cc`): `nnodes` ctor kwarg with node-contiguous
  rank-layout and divisibility checks (`:269-274`); symmetric-heap `staging_send/recv`
  `[nnodes, n_split, staging_rows, n_per]` and uint64 `internode_signals` (`:180-185`);
  private `group_flags/group_counters` as `cutlass::DeviceAllocation` (`:186-190`); a
  dedicated `internode_stream` carrying the pre-enqueued `CUStreamWaitValue` →
  `putmem_signal` ladder (`:394-413`); the receive-side `signal_wait_until` + accumulate
  (`:416-437`); monotonic `run_id_` epochs (`:370`).
- **Plumbing**: `nnodes` through `include/flux/args/moe_gather_rs.h`, pybind
  (`python/flux/cpp_mod.py`), tests/examples via `flux.testing.NNODES()`; Slurm-aware
  `launch.sh` selecting the libfabric/CXI transport.

The single-node path is byte-for-byte the upstream behavior: with `nnodes == 1`,
`group_idx` takes one value, `L == world_size`, `last_round_dst_ptr` always resolves to
`output`, and no staging/ladder/signals are allocated.

---

## 5. Provenance, element by element

The core of this doc. For each design element: where it came from, what changed in
transit, and why that source was the right one to mirror.

### 5.1 Intra-node ring with rotated segments — *inherited from upstream V2, re-indexed*

The push-mode ring, the `(stage + rank + 1)` segment rotation (all four ranks on four
different segments each stage, one flow per NVLink direction), the per-tile
`tile_barriers` set by a dedicated sync warp, the gather+topk folded into every stage's
"my contribution" — all of this is the upstream single-node kernel, untouched in
mechanics. The only change is the index space: `rank/world_size` → `local_rank/L`. This
was the highest-value inheritance: the ring is the part of the design where correctness
is subtle (per-tile release/acquire across peer memory), and it arrived pre-validated.

### 5.2 Hierarchy boundary at the node, "reduce locally, ship once" — *mirrored from dense `gemm_rs` sm80 (shape only)*

Upstream's dense multi-node reduce-scatter already embodies the right theorem for a
bandwidth-cliff fabric: complete the reduction within the NVLink island first, so the
inter-node fabric carries each byte exactly once, already aggregated. The port keeps that
shape and discards the mechanics (cooperative whole-GPU kernel, device-side blocking
`getmem`, per-segment flag matrix) for the reasons in §2. This is provenance at the level
of *invariant*, not code: "cross-node traffic = one pre-reduced contribution per (node
pair, segment)" is the property both designs share.

### 5.3 Node-rotation outer loop and same-local-rank pairing — *mirrored from the layer-0 V3 all-gather (already ported to layer-0 V2)*

The loop `g = (node_idx + 1 + g_iter) % nnodes` and the peer choice
`pe = g·L + local_rank` are the layer-0 multi-node AG's signature moves
(`gemm_grouped_v3_ag_scatter.cc:172`; V2 port documented in the layer-0 walkthrough
§4.4): every rank talks to its same-local-rank counterpart, so a node's inter-node flows
are one-per-GPU, spread across NIC rails, and delivered to the GPU that will consume them
without an intra-node forwarding hop.

One deliberate **inversion** in transit: layer 0 walks *own node first* (its consumer
wants local data earliest — arrival order becomes the compute schedule), while layer 1
walks *remote groups first* (`topk_gather_rs_v2.cu:650`) — the network transfer for
remote groups is the longest-latency path, so it is launched while the kernel still has
the own-node group's work to hide it behind. Same loop, opposite rotation phase, each
matching its layer's overlap direction.

### 5.4 `putmem_signal` payload+doorbell fusion — *idiom from `src/coll`, re-hosted from device to host*

`nvshmemx_putmem_signal_nbi_block` is how upstream's DisScatter/all2all kernels move
tokens cross-node (`all2all_impl.cu:99`): one verb carries the payload and sets the
receiver's signal word after it, eliminating a separate flag round-trip. The port keeps
the verb but moves the call site: `nvshmemx_putmem_signal_nbi_on_stream`, issued from the
**host** ladder rather than from kernel threads. Two reasons, one per layer of the stack:
the consumer kernel owns 3 SMs and must never block on NIC latency (§2), and on
Slingshot/libfabric device-issued verbs are host-proxied anyway — the `_on_stream` form
reaches the proxy without burning device cycles to get there.

### 5.5 `CUStreamWaitValue`-gated host ladder — *idiom from dense `gemm_rs` and `all_gather_op`, new composition*

Stream-side waits on device flag words exist upstream in two places: the ring-push AG
(`all_gather_op.cc:522`) and the dense reduce-scatter's per-segment host driver
(`reduce_scatter_kernel.hpp:2096,2231`). New here is the *composition*: the entire
ladder — every `(g, sid)` wait and its put — is **pre-enqueued** on a dedicated
`internode_stream` before the consumer kernel produces its first chunk, in exactly the
order the kernel produces chunks (remote-first groups, ascending splits). The ladder is
therefore a passive pipeline: no host polling thread, no kernel involvement, each put
fires the moment its `group_flags` word flips, and head-of-line blocking cannot occur
because ladder order equals production order. The kernel side of the handshake
(`__threadfence_system` + counter, last block flips the flag, `:667-677`) is the same
last-writer-flips-the-flag pattern as the op's own producer cascade — provenance §5.7.

### 5.6 Monotonic `run_id` epoch signals — *new (shared with the layer-0 a2av dispatch, same commit)*

Upstream resets its flags every iteration inside barrier-protected windows. That works
for intra-node flags (a `barrier_all` proves nobody still polls), but a cross-node
symmetric signal word would need a cross-node rendezvous to reset safely — "has the peer
consumed it yet?" is itself a communication. The port sidesteps reset entirely: signals
are `uint64` words that only grow; the sender `SET`s the current `run_id_`, the receiver
waits `CMP_GE run_id_`, and no reset ever happens (allocation is zero-initialized once).
The local, private `group_flags/group_counters` — which *can* be reset cheaply — are
`cudaMemsetAsync`-ed per run and fenced to the internode stream by `staging_reset_event`.
This split (monotonic epochs across the network, reset flags within the GPU) has no
upstream precedent in this repo; the same idiom was introduced simultaneously for the
layer-0 a2av dispatch's per-source signals.

### 5.7 Kernel→host chunk-ready cascade — *mirrored from the op's own producer flag cascade*

The GEMM→consumer handshake upstream already solved "many unordered writers, one release
flag" with a tile→problem→split counter cascade
(`gather_rs_gemm_grouped_with_absmax.h:550-576`). The kernel→host handshake reuses that
solution one level down: `gridDim.x` (=3) blocks finish a `(g, sid)` group in arbitrary
order; an `atomicAdd` counter elects the last one, which alone does the system-scope
release store that the host ladder's `CUStreamWaitValue` observes. Same pattern, smaller
writer set, different consumer (a stream wait instead of a spinning kernel).

### 5.8 Symmetric-heap staging + `DeviceAllocation` flags — *mirrored from the layer-0 V2 multi-node port*

`staging_send/recv` and `internode_signals` on the NVSHMEM symmetric heap (the only
cross-node-addressable memory), while the kernel→host flags are private
`cutlass::DeviceAllocation` — plain `cudaMalloc`, not torch tensors, because
`cuStreamWriteValue/WaitValue32` reject the virtual addresses torch's
`expandable_segments` allocator hands out. That VA lesson was learned in the layer-0
multi-node port (`gemm_grouped_v2_ag_scatter.cc:248-252`) and applied here directly
(comment at `gemm_grouped_v2_gather_rs.cc:147-148`).

### Summary table

| Design element | Provenance | Changed in transit |
|---|---|---|
| intra-node push ring, rotated segments, per-tile flags, sync warp | upstream V2 single-node kernel | index space `world` → `node`; outer group loop added |
| node-boundary hierarchy, aggregate-before-transmit | dense `gemm_rs` sm80 multi-node (shape only) | mechanics replaced wholesale (§2) |
| node-rotation loop, same-local-rank pairing | layer-0 V3 AG (via layer-0 V2 port) | rotation phase inverted: remote-first, not own-first |
| `putmem_signal` fusion | `src/coll` DisScatter kernels | device-issued → host-issued `_on_stream` |
| stream-wait-gated comm ladder | `gemm_rs` per-segment driver, `all_gather_op` ring | pre-enqueued full pipeline on a dedicated stream |
| monotonic epoch signals, never reset | **new** (concurrent with a2av dispatch) | — |
| last-writer counter cascade (kernel→host) | this op's own producer cascade | writer set = 3 blocks; consumer = `CUStreamWaitValue` |
| symmetric staging + private `DeviceAllocation` flags | layer-0 V2 multi-node port | — |
| receive-side per-split accumulate kernel | analogue of `gemm_rs`'s post-transfer `add_continous_kernel` | runs on the consumer stream, split-pipelined |

---

## 6. Roads not taken

- **Flat `world_size` ring across nodes.** The single-node ring generalizes syntactically
  (just keep `% world_size`), but every hop crossing the node boundary would need a
  network verb instead of an NVLink store: `world_size−1` serialized network latencies
  per segment, with hop k+1 unable to start until hop k's payload *and* its remote
  arithmetic finish. The hierarchy pays `L−1` NVLink hops (cheap, overlapped) plus
  exactly **one** network transfer per (node pair, split).
- **Port V3 layer 1.** Nothing to port — `nvshmem_ptr` is NULL across nodes (§2), and
  upstream's own commented-out `putmem_nbi` in the epilogue (`epilogue_gather.hpp:602`)
  marks where that road was abandoned upstream.
- **Adopt dense `gemm_rs` mechanics.** Cooperative whole-GPU launch conflicts with the
  3-SM concurrent-consumer architecture; 1:1 tile↔comm mapping doesn't survive the
  gather+topk permutation; device-side blocking `getmem` stalls the reduction pipeline on
  NIC round-trips (§2).
- **All-gather partials, reduce locally.** Ship every rank's `[M, N]` partials everywhere
  and let each rank sum: `world_size×` the wire bytes of the hierarchical scheme, on the
  layer whose whole point is that reduction *shrinks* data as it moves.
- **Device-issued puts from the consumer kernel** (DisScatter-style). No IBGDA on
  Slingshot: the device verb funnels through the host proxy regardless, so it inherits
  all the latency and additionally spends the consumer's scarce threadblocks issuing it.

---

## 7. Validation and running it

Validated on Perlmutter, 2 nodes × 4 A100s, against the single-process torch reference
(`allclose`), with the constraints the ctor enforces: token counts divisible by
`world_size·topk`, `max_m/topk` divisible by `world_size` (fixed staging slots),
node-contiguous rank layout (same-local-rank pairing depends on it), and
`do_all_reduce`/`use_read_mode` single-node-only.

```bash
source ./module.sh
salloc -A m4243_g -q interactive -C gpu -N 2 --gpus-per-node=4 -t 30
export NVSHMEM_SYMMETRIC_SIZE=4G   # staging/signals live on the symmetric heap
srun --nodes=2 --ntasks-per-node=1 ./launch.sh \
    test/python/moe_gather_rs/test_moe_gather_rs.py -M 40960 -T 8 -E 1
```
