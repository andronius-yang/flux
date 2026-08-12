# Insight ledger — settled questions and hazards

**Read this before you propose an optimization, not before you build.** Everything here cost
days to establish. Entries are ordered by *relitigation risk* — how likely a fresh agent is
to independently re-propose the dead idea — not by how interesting they are.

Each entry has a stable ID (`NR-nn`). Cite them from code and capsule notes (`// see NR-02`)
the way the source already cites decision capsules.

Every entry carries a **confidence** label, and they are not equal:

- **measured** — a paired A/B inside one capsule on one build.
- **mechanism** — reasoned from how the hardware or stack works; consistent with data but
  not isolated by an experiment.
- **hunch** — judgment from watching many runs. Useful, not evidence.

And a **falsifier**: the observation that reopens it. An entry without a falsifier is either
re-litigated forever or over-trusted forever; both are failures.

---

## NR-01 — "Saving wire bytes wins" is false unless the machinery is off the critical path

**Relitigation risk: highest.** This is the most seductive wrong idea in the project, and a
fresh agent will re-propose it within an hour of reading the dedup design.

**Question.** Does reducing inter-node bytes via token dedup reduce layer0 latency?

**Verdict.** No, not on its own. Dedup saved **27–42%** of inter-node bytes and *lost* to the
`hier` baseline at every budget tested. The byte saving was worth ~0.3 ms on the fabric while
the dedup machinery (serialized pack gathers, non-`nbi` gateway forwards) cost ~4.4 ms on the
critical path. It only won after the machinery was moved off that path:
`dedup-gather 11.31 → union-bcast 10.54 → +fused pack 9.48 → +pack overlap 8.69` versus
`hier ≈ 9.9` ms (64 MiB rows, topk=8).

**Scope.** Measured at L=8/NN=2 over EFA. The *ordering* is fabric-independent reasoning; the
crossover budget is not.

**What it means now.** Wire-byte reduction is a *permission slip*, not a win. Any new
proposal must state which existing serialized work it removes, not how many bytes it saves.

**Falsifier.** A fabric where wire time dominates end-to-end by enough that a 30% byte cut
exceeds the machinery cost. Slingshot is faster than EFA's emulated RMA (NR-04), which makes
this *less* likely here, not more.

**Confidence: measured.** **Cost to re-test:** ~6 cells.

---

## NR-02 — Two hazard classes produce an identical silent hang. Learn the discriminator.

**Relitigation risk: highest cost.** Neither ever raises an error. Both are rank-dependent
and intermittent. An agent without this entry will tune knobs for a day.

**Symptom in both cases:** the job produces no output, no error, some or all ranks pinned at
100% GPU, reproduces intermittently and not on every rank.

| | **Class A — dispatch starvation** | **Class B — channel wait-order inversion** |
|---|---|---|
| Cause | A kernel enqueued *after* the persistent 200-CTA agscatter GEMM has spread over all 108 SMs may **never dispatch**. It is a race against the GEMM ramp, not throttling. | A pre-launch-enqueued `CUStreamWaitValue` whose satisfying writer is **same-rank and enqueued later**. Streams multiplex onto `CUDA_DEVICE_MAX_CONNECTIONS` hardware channels; a channel executes in **host enqueue order**, so a blocking wait at the channel head blocks the very put that would release it. |
| Fingerprint | `cuda-gdb` `info cuda kernels` shows the GEMM resident with an SM mask of all-`f`; the starved kernel never appears. Per-op replay events show `cp_stream` dying mid-sequence at an arbitrary depth. | Waits and writers are both present in the enqueue record, but the writer sits *behind* the wait on the same channel. Whether it hangs depends on that run's stream→channel hash — hence the flakiness. |
| Rule | **Never defer SM-kernel-bearing work past the GEMM launch.** Only SM-free ops (CE copies, `nbi` CE puts, memops, event records) may live in the deferred batch. | **For every pre-launch wait, its writers must be either remote ranks (their puts arrive as NIC/P2P memory writes, never through my channels) or same-rank work enqueued *before* the wait.** |

**Negative controls — these do NOT rescue either class. Do not retry them:**
`CUDA_DEVICE_MAX_CONNECTIONS=32`, `sm_margin=16`. Class A is dispatch, class B is queue
order; neither is an occupancy problem.

**The debug recipe that cracked it.** `FLUX_A2AV_DEBUG_HANG` was **stripped from the tree and
never committed** — it cannot be recovered from git. Reconstruct it:

1. Per-replay-op `cudaEvent`s, queried on the **next** iteration (not the current one — the
   current one is hung).
2. Device signal-buffer readback on a **fresh** stream, so the read is not itself queued
   behind the hang.
3. `cuda-gdb` resident-kernel dump (`info cuda kernels`) for the SM mask.

Reading: if the signal buffer shows the awaited value **never written**, and the writer's
kernel never appears resident → Class A. If the awaited value is written *later* than the
waiter's enqueue position on a shared channel → Class B.

**Scope.** Both are properties of CUDA stream/channel semantics and the persistent-GEMM
design. **Neither is AWS-specific — both transfer to Perlmutter unchanged.**

**Falsifier.** None expected; these are architectural. **Confidence: measured** (both were
diagnosed to root cause and fixed). **Cost to re-test:** don't.

---

## NR-03 — What killed `hier` was recv-side skew, not send bytes

**Question.** Why did the non-compress a2av family underperform?

**Verdict.** Receive-side hot-rank skew. In the 2026-07-29 full sweep the barrier phase was
**44–47 ms of 64 ms** end-to-end at b64 for `a2av`/`hier`, while the compress family's
dedup'd recv shrank it to ~3 ms. Most striking: a **dense allgather beat the entire
non-compress a2av family** (0.42–0.63× `hier`), despite moving far more data.

**What it means now.** Send-side balancing (the whole relay line of work) cannot touch this.
A hot receiving rank has more GEMM rows and there is no send-side fix. When diagnosing a
regression, check recv skew before wire bytes.

**Scope.** `remotefrac` columns are deliberately skewed; the effect size is matrix-dependent.

**Falsifier.** A traffic matrix with balanced recv where `hier` still loses.

**Confidence: measured.** **Cost to re-test:** it is a by-product of any full sweep.

---

## NR-04 — The EFA landing ladder: closed, AWS-only, and probably void here

**Question.** Why did inter-node payloads land in a staggered "ladder" (0.27–2.2 ms spread)?

**Verdict.** Not payload size (union dedup compresses 18× row skew to ~3× bytes, and the
0.90-remote-fraction sender landed *first*), and not launch skew (≤324 µs). Landing order
equals **sender-side egress drain order**, exactly. Root cause: NVSHMEM's libfabric transport
selects a fabric whose `fi_write` is **software-emulated** (`FI_PROTO_EFA`, raw max message
8928 B, proxy-thread serviced), giving 3–9 GB/s per NIC against a ~24 GB/s node egress line.

**Scope. This is a p4d/EFA hardware story** — Nitro v3 supports RDMA READ but not WRITE.
**Slingshot/CXI is a real RMA provider, so this entry is expected to be void on Perlmutter.**

**What it means now.** Do not carry the ladder's conclusions here. Do re-measure landing
spread, because Tier B's payoff scales with it (`02` §4, M4).

**Falsifier / reopener.** If CXI *also* shows a landing ladder, the receiver-initiated
("get-based") union wire — where the gateway pulls from the source's symmetric buffer after a
tiny RTS signal, the inversion `aws-ofi-nccl` uses — becomes worth building. It was designed
but never implemented. Validate with `nvshmem perftest` `shmem_get_bw` vs `shmem_put_bw`
first; sources are at `sw/src/nvshmem_src/perftest/` on the old cluster (gone) or from the
NVSHMEM tarball.

**Confidence: mechanism**, corroborated by gateway landing markers. **Cost to re-test:** one nsys pair.

---

## NR-05 — `FI_EFA_USE_DEVICE_RDMA=1` is a no-op; `efa-direct` is a dead end

**Question.** Can NVSHMEM be pushed onto native device RDMA instead of the emulated path?

**Verdict.** No, on both routes. (a) `FI_EFA_USE_DEVICE_RDMA=1` made no difference to NVSHMEM
CUDA-memory puts — last drain 2159 µs vs 2138 µs baseline. (b) The `efa-direct` fabric that
flag reveals advertises **no RMA capabilities at all** (`FI_MSG`/`SEND`/`RECV` only, 8928 B
max), so selecting it would break NVSHMEM's `fi_write` put path outright.

**Evidence handle — read this carefully.** The surviving committed arm is a single **`nsys`**
cell (`20260803-033303`); its intended baseline partner `20260803-031503` has
`status=stuck` and was deliberately left uncommitted (`68409d4`). Under this project's own
never-mix rule an `nsys` cell **cannot support a latency verdict**. So (a) is
*weakly measured*; (b) is a capability read from `fi_info` and is solid.

**Scope.** AWS/EFA only. Irrelevant on Perlmutter except as a cautionary tale about
`fi_info`-driven optimism.

**Falsifier.** Not applicable off AWS. **Confidence: (a) mechanism, weakly measured; (b) measured.**

---

## NR-06 — Parallel fan-out lanes lose to ring rotation

**Question.** Should the gateway's per-destination forwards go out on parallel streams
(`fanpar`) rather than a ring-rotated order on one (`fanring`/canonical)?

**Verdict.** No. A/B/C at b2/b8/b32, capsule `20260804-043026`: ring rotation is significant
at b2 (−69 µs trimmed), ties at b8/b32, and is never worse. Parallel lanes were **worse at
b8 (+72 µs)**. The `FLUX_A2AV_FANOUT` knob, the `fanpar`/`fanring` variants and the fan-out
stream pool were all deleted at canonicalization.

**Scope.** L=8. Ring rotation is *defined over L*, so the balance it creates changes at L=4.
Worth one cheap re-check, not a redesign.

**Falsifier.** A shape where per-destination forwards are large enough that stream-level
parallelism beats queue ordering.

**Confidence: measured.** **Cost to re-test:** 9 cells (the original A/B/C spec).

**Re-check 2026-08-07 (L=4/CXI/realistic traces): the ring verdict does NOT transfer
to the starving regime.** After the realistic-trace campaign showed lb_union deficits
0.41–0.80 with windows parked behind the ascending-round forward order, the eager arm
was reimplemented as `FLUX_A2AV_FANOUT=1` (per-ROUND fan-out streams for the Tier-B
gateway forward + event re-join into the tail stream; variant
`hier_compress_lb_union_eager`). Capsules `20260807-055012_perlmutter_d5f3160a` (iso)
+ `20260807-055227_perlmutter_b9a6fc40` (nsys), b32 k8 EARLY_LAUNCH, Qwen3 trace arms:
isolated latency ring 17.26 / eager 16.57 / union 16.55 ms (pernode — eager recovers
~97% of the gap) and 29.00 / 28.61 / 28.22 (homogZH — ~50%). Correctness clean both
knob states. Tile-trace deficits were INCONCLUSIVE in that capture: run variance on
unchanged code paths (union worst-rank 0.238–0.466 vs 0.079–0.094 in capsule
1120f2ea, different build/nodes) exceeded the variant deltas — do not quote E2
deficits. Eager is a canonicalization candidate (make default for lb_union or fold
into a redesign); the original b2 −69 µs ring win was a non-starving-regime result.

---

## NR-07 — The tile claimer is not worth it for lb_union

**Question.** Should lb_union use the dynamic tile claimer instead of the dense static schedule?

**Verdict.** No. At G=128 the claimer costs **~0.8 ms** on its own — larger than the entire
effect being chased. lb_union takes the dense static schedule with a window-keyed per-tile
spin ("candidate D"). Two related fixes were also rejected: **claimer lookahead** (Fix A) and
**coarser gating lanes** (Fix C).

**Note on a confusing artifact.** The `seg_end`-lane bucketing change (Fix B) *did* ship in
`workspace_util.cu` and is correct — but it is **inert for lb_union**, which builds no
buckets. It benefits only the plain a2av dispatch mode. Older notes saying "tried, no help"
and the commit message saying "shipped" are both true, in different scopes.

**Scope.** G=128. The claimer's cost scales with G; at much smaller G the trade could differ.

**Falsifier.** A configuration where claimer overhead is small relative to the imbalance it
removes — i.e. small G with high per-tile variance.

**Confidence: measured.**

---

## NR-08 — Never cache the gating cumsum across iterations

**Question.** The per-(expert, window) gating cumsum is recomputed every forward. Cache it?

**Verdict.** No — and this is a **methodological** rejection, which makes it permanent rather
than platform-dependent. In deployment, routing changes with every activation, so a cached
cumsum would be valid only in a benchmark where the same routing repeats. Caching would make
the benchmark faster and the system no faster. That is faking the measurement.

The same honesty principle produced `isolated` mode (per-iteration sync + barrier, so no
cross-iteration pipelining can hide a wire tail) and the `FLUX_TEST_DETERMINISTIC` audit.

**What was done instead.** Collapse the cumsum to **one `searchsorted` with zero H2D**,
riding the existing pinned meta arena. See NR-09.

**Falsifier.** None. If someone proposes caching, the answer is "then measure with routing
that changes per iteration, and show it still helps."

**Confidence: mechanism (design principle).** In-code at
`gemm_grouped_v2_ag_scatter.cc` ("routing is never cached across forwards").

---

## NR-09 — A 3.58 → 4.33 ms regression was host dispatch overhead, not the algorithm

**Question.** Tier B regressed by ~21%. Was the window-gating design wrong?

**Verdict.** No. The cost was **~380–450 µs/iter of host-blocking time** in the per-iteration
gating-cumsum torch block: ~120–165 µs from two **pageable** `from_blob(...).to(device)` H2D
copies, each forcing a hidden `cudaStreamSynchronize` mid-dispatch, plus ~250–320 µs from a
~15-op dispatch chain (a `scatter_add` over all M rows, etc.).

Fix: one `searchsorted` over an already globally-sorted composite key, zero H2D, all lane
ends computed as host constants and shipped in the existing single `cudaMemcpyAsync`.
Golden equivalence verified against the `scatter_add` path.

**What it means now — the transferable lesson.** At these timescales a **~15-op torch
dispatch chain on the host is a first-order cost**, and any pageable H2D inside the forward
is a hidden sync. When a change regresses, profile the *host* before redesigning the device
algorithm.

**Falsifier.** None; the fix was measured back to parity.

**Confidence: measured.** Capsules `20260804-042640` (E3), `043026` (E5).

---

## NR-10 — Pre-2026-07-29 performance numbers are void

Two lines, because it is already law in `CLAUDE.md`, `sweeps/SCHEMA.md`,
`.claude/skills/sweep/SKILL.md` and walkthrough §12. The harness enables `torch.use_deterministic_algorithms(True)` in **two**
places (`python/flux/testing/utils.py`, `python/flux/dist_utils.py`) — guarding one does
nothing — and deterministic `scatter_` takes a serial fallback ~500× slower. The compress
pack and relay index builds are `scatter_`-heavy; `hier` barely uses `scatter_`. So every
compress-vs-hier comparison before the guard landed was biased against compress (relay went
**41.9 → 11.9 ms** at 64 MiB once disabled).

**Good news for you: no committed capsule is contaminated** — every one records
`deterministic=0`. The only residue is prose numbers in
`docs/launch/comet_traffic_matrix_tests.md` predating 2026-07-29, now bannered.

**Confidence: measured.**

---

## NR-11 — `hier_compress_pack`'s `CONNECTIONS=2` is not actually validated

Flagged as folklore-in-progress rather than a settled result.

`sweeps/variants.py` called `CUDA_DEVICE_MAX_CONNECTIONS=2` "the validated best config" for
`hier_compress_pack`. Auditing the capsules: **all 15 `hier_compress_pack` cells in the entire
set are `conn=2`. There is no A/B.** All are from 2026-07-29 — before the a2av family was
pinned to `conn=8` and before `isolated` mode existed. The claim traces to a single
observation that `conn=2` took pack from 1.96 → 1.58 ms in the very first capsule, which is
a *cross-configuration* comparison, not a controlled one.

The wording has been softened in `variants.py`. **If you use `hier_compress_pack`, re-derive
its connection count.** Inheriting an unsupported claim as fact is the most expensive kind of
knowledge loss — it does not vanish, it misleads.

**Confidence: hunch (the original claim).** **Cost to re-test:** 2 cells.

---

## NR-12 — What the moonep/ultraep arms may and may not claim about upstream overlap

**Relitigation risk: high.** A fresh agent reading `moonep_overlap` will propose the same
flag for ultraep and call it "faithful"; a writeup agent reading the serialized `ultraep`
arm will report "UltraEP does not overlap anything" as if it were an upstream property.
Both are wrong in opposite directions. The port-fidelity taxonomy is canonicalized in
`sweeps/SCHEMA.md` (§ "Overlap and transport fidelity"); this entry exists so nobody
re-derives or mis-cites it.

**The four facts.**

1. **Overlap machinery is authentic in BOTH upstreams.** UltraEP launches `weight_sync` on
   a dedicated high-priority comm stream with `async_finish=True` and returns an event
   (`ultra_ep.cpp:253, 1095-1211`); MoonEP prefetches redundant-expert weights
   asynchronously during dispatch. Our *serialized* arms (`moonep`, `ultraep`) are
   deliberately-pessimistic ports, not upstream behavior.
2. **The join point differs between upstreams, and it is not symmetric.** MoonEP's prefetch
   is consumed at GEMM — `moonep_overlap`'s join-before-GEMM is authentic. UltraEP's
   reference integration joins weight_sync **before token dispatch** (overlap window =
   reroute only, ~40 µs under a 200-300 µs sync; the paper itself books weight_sync as
   exposed critical path, <300 µs). So for a future `ultraep_overlap`:
   `--ws_join dispatch` = authentic; `--ws_join gemm` = **counterfactual** (legal under the
   data dependency, but it prices Perlmutter's split fabrics — weight_sync on NVLink,
   dispatch on Slingshot — a tradeoff upstream never faced because both shared NVLink).
   Label it like `ultraep_domain16`, never quote it as UltraEP's behavior.
3. **The second NCCL communicator is port machinery, full stop.** Neither upstream has any
   communicator in its weight path: UltraEP is raw NVLink ld/st through `nvshmem_ptr` peer
   VAs from its own SM kernel (+ `__threadfence_system` epoch flags); MoonEP is TMA bulk
   copies / NVSwitch multicast into NVLink symmetric buffers. The separate communicator
   exists only because our ports move weights over NCCL P2P, and NCCL serializes ops per
   communicator (would otherwise serialize against the dispatch a2a).
4. **Transport authenticity ladder** (weight/prefetch movement, most→least authentic):
   CUDA-IPC peer copies on a comm stream (intra-node only; residual gap = CE-vs-SM
   contention, upstream copies burn SMs) > NVSHMEM `putmem_signal` (one-sided push + flag
   = the exact upstream signaling shape; cross-node on Perlmutter is proxy-mediated CXI,
   an extra caveat) > NCCL P2P send/recv (two-sided rendezvous, own protocol/chunking
   optimizations — least like upstream, but capsule-comparable with the moonep arms).
   `NVSHMEM_DISABLE_NCCL` in UltraEP's README governs ONLY the plan-metadata fcollect,
   never weight_sync — it is not evidence of an upstream NCCL weight path.

**Falsifier.** Upstream UltraEP moving its reference join after dispatch, or shipping a
cross-node weight_sync (today it hard-asserts `nvshmem_ptr != nullptr`, i.e. cannot).

**Confidence: mechanism** (read from both upstream sources; overlap deltas not yet measured
here — the first `ultraep_overlap` capsule turns the join-point cost into a number).

### NR-12 Amendment (2026-08-11) — four findings from the deep-dives that preceded the overlap/nvshmem arms

Facts 1-4 above stand, with fact 1 sharpened and fact 4 superseded as follows.

**(5) UltraEP direct mode has NO receiver-side signaling — the join IS the publication
mechanism.** `build_weight_sync_task_lists` keeps a task only if `master_rank == rank_idx`
(weight_sync.cu:259): the master's rank pushes into peer VAs with `wait_ready_slot = -1`,
`num_ready_signals = 0`; every flag/epoch/`__threadfence_system` path is relay-only. The
returned `EventHandle(comm_stream)` covers only the local rank's own outgoing pushes — nothing
anywhere waits for a rank's incoming replica weights. Cross-rank happens-before comes solely
from the reference integration's join-before-dispatch: each sender's pushes complete before it
enters the dispatch collective, and no receiver's dispatch completes before every sender
entered. **Corollary:** `ultraep_overlap_joingemm` is sound in our port ONLY because NCCL is
two-sided (irecv completion rides the ws-stream event); under upstream's unsignaled pushes the
same join point would be a correctness bug, which is presumably why upstream doesn't offer it.

**(6) MoonEP prefetch is a destination-side PULL, not a push.** `launch_prefetch`
(moonep/prefetch.py:317-385) remote-READS the home rank's mapped weight rows via TMA G2S and
stores locally into its own prefetch slots — zero cross-rank signaling (the source is immutable
parameter memory; tests/test_prefetch.py:16-18 states this explicitly). The NVSHMEM analog is
`getmem`, not put. Our NCCL isend/irecv prefetch is therefore DOUBLY a port artifact: wrong
initiator direction and a two-sided protocol. (MoonEP's pull is also a considered bandwidth
decision — grad_reduce.py:21-27 rejects remote writes because reads+writes split the NVLink
budget.)

**(7) MoonEP serializes dispatch+prefetch on ONE shared comm stream** (api.py:487-499: a
single `_comm_stream` per Buffer serves dispatch/prefetch/combine; with async_finish they
serialize with each other and overlap only main-stream compute; the upstream benchmark runs
`dispatch(); prefetch()` back-to-back on that stream). `moonep_overlap` (prefetch concurrent
with dispatch on a separate stream + separate communicator) is FINER-grained than upstream;
`moonep_overlap_shared` is the authentic-serialization counterpart (prefetch enqueued after
the a2av on the SAME communicator; overlap window = scatter only). Also for the record: MoonEP
has no inter-node path at all — its multi-node answer is NVLink-fabric domain extension
(nvl_shared_buffer.cuh fabric handles), never a network. Any inter-node transport in these
ports is extrapolation, judged by preservation of intra-domain semantics.

**(8) flux All2AllSingle's putmem_signal path is DEAD CODE, and the op is rejected for weight
movement.** The live path is `a2a_single_kernel_v2` (all2all_single_2d_impl.cu:191-211):
full-buffer memcpy into symmetric staging, one `nvshmemx_putmem_nbi_block` per destination
into fixed per-source slots, scalar copy-out, and TWO `nvshmemx_barrier_on_stream` per forward.
The `putmem_signal`+`signal_wait` kernel (:84-189) is never launched, and its signal slots are
never reset (single-shot if ever enabled). Put-then-barrier is actually CLOSER to MoonEP's
real dispatch (push + one system-scope exit barrier) than put+signal would be — the 2026-08-11
doc corrections fixed five sites that described the dead kernel. Weight-movement fit verdict:
REJECTED — dense per-(src,dst) staging (~1.5 GiB symmetric heap to move <=48 MiB of actual
replica weights at W=16), two extra full-payload local copies, and NVSHMEM_TEAM_WORLD barriers
that forbid a second instance on a side stream beside the token a2av (team-barrier collision +
NR-02 Class-B channel hazard). Weight movement stays NCCL `batch_isend_irecv` (declared port
artifact). The authentic upgrade, if weight-path transport fidelity ever becomes the question
under test: a custom kernel doing bare `putmem_nbi` pushes published by the subsequent
collective join (UltraEP direct) / `getmem` pulls (MoonEP prefetch) — future work only.

**Falsifier updates.** Fact 5: an upstream commit adding receiver-side flags to the direct
path. Fact 7: upstream giving prefetch its own stream. Fact 8: upstream flux switching the
live launch back to the signal kernel (with a reset).

**Confidence: mechanism** (all four read from source; the overlap deltas become measured
numbers in the ultraep M4 capsule).
