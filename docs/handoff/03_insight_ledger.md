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

### NR-12 Amendment (2026-08-11b) — the getmem pull is IMPLEMENTED; fact 8's "future work" clause is void

**(9) `flux.WeightPrefetchGetmem` (branch moonep-nvshmem-default) is the authentic weight
path fact 8 named.** SM device kernel issuing `nvshmemx_getmem_nbi_block` chunks per
(pair, chunk) work item, one host `nvshmemx_quiet_on_stream` as the join — destination
initiates, source ranks passive, zero signaling, no communicator, no barriers (host
`getmem_nbi_on_stream` fallback kept for A/B; measured identical). The [epn, ffn_shard, H]
weight home AND the [B, ...] prefetch slots live permanently on the symmetric heap —
residency moves, it does not grow (mirrors upstream's [E+B, H, H'] mapped tensor, and
enables a future B=3-4 read-through arm). Arms: `moonep_getmem`, `moonep_getmem_overlap`
(no second communicator — the fact-3 port machinery disappears on this path),
`moonep_nvshmem_getmem` (fully one-sided). Validated 2026-08-11 (jobs 56726073/56726303):
1n kernel/stream/nvshmem+getmem and 2n cross-node kernel/overlap/nvshmem+getmem all
bitwise-exact incl. the per-slot broadcast check; overlap `prefetch_wait` 0.003 ms.
CPU planner suite untouched (101 passed). The 4n16r trace capsule
(`sweeps/specs/moonep_getmem_pm4n_trace_iso.yaml`) is pending node-hours.

**(10) CXI proxy gets require a provider-registered LOCAL destination — ordinary
cudaMalloc memory segfaults.** Found by the a2av_comm_bench `prefetch` mode: 8/8 ranks
segfault on cross-node pulls into cudaMalloc dst (both impls; reproducible on demand via
`PREFETCH_DST_SYM=0`), while intra-node P2P gets don't care and the bench's older ag mode
(symmetric dst) always worked. Any future one-sided-read design on this fabric must put
the local buffer on the symmetric heap or register it.

**(11) Measured cross-node pull bandwidth (2n, per-GPU NIC): 16.5–17.5 GB/s/rank —
~70% of Slingshot line rate and ~40% faster than the NCCL isend/irecv prefetch it
replaces** (32 MiB worst-rank med 1.93 ms vs ~2.7 ms / ~12 GB/s in capsule
20260808-032217). Insensitive to chunking (1–64 chunks) and issue path (kernel vs
stream) — the proxy pipeline is the limiter; nothing to tune, 4 MiB default chunk
stands. This also answers NR-04's reopener for the get direction on CXI: no landing
ladder. Numbers + method: `a2av_comm_bench/docs/methodology.md` (prefetch section).

**Falsifier updates.** Fact 10: an NVSHMEM release note or test showing unregistered-dst
gets are supported on libfabric/CXI (would recast the segfault as a 3.2.5 bug). Fact 11:
the 4n capsule's prefetch_ms disagreeing materially with the bench-derived expectation
(~1.9-2.3 ms serialized).

**Confidence: measured** (facts 10-11 from the committed bench transcripts; fact 9's
end-to-end perf story becomes measured when the pending capsule lands).

## NR-13 — The mcast+gated regression: gateway legs must never sit in the issue window (and multicast is structurally starved under the MoonEP planner)

**Relitigation risk: high.** A fresh agent will read `moonep_fused_push_mcast_gated`'s
regression (capsules 20260813-024417: 7.50 vs 6.64 join; 20260814-082834 b64: 25.04 vs
22.23) and conclude "tile-gating doesn't work" or "multicast doesn't work". Both wrong.
Root-cause campaign 2026-08-14 (plan: curious-frolicking-clover; capsules
20260814-080654/-082834/-083715/-083903 + session-log fanoutskew evidence).

**The facts.**

1. **Multicast is structurally starved under MoonEP's planner.** Census over the real
   trace matrices AND synthetic families (b1..b64, 4n16r, G=128 k8): (expert, dest-node)
   prefetch groups are almost always singletons (0-2 multi-member groups, max fan 2, wire
   dedup 1.00-1.09x). Cause: the constructive planner's "each dest imports from at most
   one home group" + hottest-first shedding minimize distinct migrated experts. So the
   gateway machinery moves at most 1-2 legs — but even ONE saved 32 MiB inter-node leg is
   worth -1.4 ms when weights dominate (b1: mcast 4.76 vs direct 6.15, tight over 10
   iters, capsule 20260814-080654).
2. **The regression's causal trigger is the gateway leg, not the tile gate.** fanoutskew
   plans have ZERO gateway legs (all singleton -> all-direct even in mcast mode): there
   mcast_gated shows NO regression (7.56-7.73 vs join 7.64-8.01; two independent runs,
   session log only — partial capsules lost to an external scancel, rerun
   rc_controls for a capsule-grade copy). Every cell WITH a gateway leg regresses.
3. **Mechanism (phase-0 rank forensics + driver code): the gateway's
   CUStreamWaitValue64 sits in the weight-issue window, and the driver's pref_end
   event-join makes the gateway's forward launch — hence its token puts — hostage to its
   weight ingress.** In mcast_gated the sole gateway's prefetch_ms jumps 0.12 -> 2.49 ms
   and EVERY rank's fused window inflates ~+0.75 ms (one late token source stretches all
   windows). Under join the same wait is cheap because no rank's token wire starts before
   weights land (quiet NIC).
4. **Channel pressure is the dominant amplifier at b8: CUDA_DEVICE_MAX_CONNECTIONS=32
   eliminates the regression entirely** (capsule 20260814-083903: mcast 6.25, mcast_gated
   6.33) and makes both arms ~0.8 ms faster than their conn=8 twins — an NR-02 Class-B
   LATENCY coupling (wait + fan-out puts head-of-line on shared channels), not a hang.
5. **The weight-gate tile spin itself is real but secondary**: tile-trace sidecars
   (capsule 20260814-083715) show prefetch-slot problems' spin at 2.4x direct-gated and
   5x mcast-join — yet fact 2 shows gating without a gateway costs nothing, and
   direct-gated == direct-join at every budget incl. b64 (21.97 vs 22.23 — gated slightly
   better). "Not enough compute to hide" is REFUTED as the explanation (regression grows
   with budget instead of shrinking).
6. Open anomalies logged for follow-up: (a) the getmem pull costs 5.4-5.9 ms at EVERY
   budget including quiet-wire b1 — intrinsic, not contention; new lead on the NR-12
   fact-11 discrepancy; (b) push join gate_ms reads 0.4 ms at b1 vs 3.0 ms at b8 for
   identical weight legs — physically tight for 32 MiB cross-node transfers; possibly a
   value-benign early release (weights identical every iteration, so correctness cannot
   see it) — the nsys reps of 20260814-083715 are the place to check.

**Fix ranking (implementation is follow-up work):** (F-A) raise
CUDA_DEVICE_MAX_CONNECTIONS for the fused push arms — env-only, eliminates the
regression and is faster outright, needs a clean one-capsule A/B before canonicalizing;
(F-B) decouple gateway fan-out from the issue window (record pref_end before the gateway
wait section, or a dedicated fan-out stream / deferred-wire-style issue); (F-C)
plan-aware mcast (emit gateway legs only when the census finds multi-member groups; keep
them for weight-dominated budgets where the b1-style win lives); (F-D) schedule
prefetch-slot problems last in the static tile order.

**Falsifiers.** Fact 2: a fanoutskew-family capsule showing the regression with zero
gateway legs. Fact 4: a conn=32 capsule where the regression persists. Fact 1: a routing
family + planner config producing fan >= 3 groups at 4n16r (would revive multicast's
dedup case).

**Confidence: measured** for facts 1, 4, 5 (committed capsules/census); **mechanism**
for facts 2-3 (two session-log runs + rank forensics; capsule-grade fanoutskew copy
pending a calm queue).

### NR-13 Amendment (2026-08-14b) — F-B lands and resolves the regression at source; F-A demoted to "not needed"

**(7) The F-B choreography fix (WeightPushMulticast::forward_gateway split; driver
records pref_end after the wait-free home puts, gateway wait + NVLink fan-out run
concurrently with the fused forward, drained via a gw_end join before iteration end)
eliminates the mcast_gated regression within-capsule** (post-fix grid
20260814-123215, conn=8, one binary): b8 all four push arms 6.35-6.43 ms (pre-fix
mcast_gated was +0.76 over mcast join in 20260814-121208); b1 mcast_gated becomes the
BEST arm (4.82 vs mcast join 5.29 vs direct 6.2 — the multicast dedup win plus
gating). Ladder + all 30 grid cells correctness-green incl. the real 4n16r gateway
fan-out (V3a bitwise slots).

**(8) Post-F-B, conn=32 no longer changes the picture** (twin capsule
20260814-123721: b8 6.27-6.42, matching conn=8 within noise) — fact 4's channel
amplifier was only live while the F-B hazard existed. DECISION: keep the historical
CUDA_DEVICE_MAX_CONNECTIONS=8 pin on the fused arms (comparability), F-A closed as
"fixed by F-B; conn flip unnecessary".

**(9) F-C (--weight_push_mode auto + push_plan_stats census, arms
moonep_fused_push_auto[_gated])** resolves per-plan exactly as designed (mcast iff
n_multi_groups > 0; capsule info carries wpush_mode_resolved + census) and matches
the resolved arm's perf. Isolated b32 cells in both grids show +-3-5 ms single-cell
transients on random arms (the rerun-outliers gotcha, now observed on CXI too) — no
headline rests on an outlier cell.

**Falsifier updates.** Fact 7: a capsule on the post-F-B binary showing mcast_gated
regressing vs mcast join with a gateway leg present. Fact 8: any post-F-B capsule
where conn=32 vs conn=8 differs beyond cell noise.

**Confidence: measured** (both grids committed-pending; F-D implemented 2026-08-14 as
`FLUX_A2AV_SCHED_PREFETCH_LAST` — see NR-14; capsule id recorded there once the ladder
grid lands).

## NR-14 — Phase-ordered wire (E0–E2): resident tokens first, weights second, slot-last wavefront

**Design (user-settled, 2026-08-14).** Target wire order on the fused MoonEP path:
(1) resident-destined token rows fire first, (2) prefetch weight legs second, (3)
prefetch-only token rows last. Phase policy: a union row needed by BOTH a resident
expert and a prefetch slot on the same destination node travels in phase 1 — phases 1
and 3 are a DISJOINT PARTITION of the lb_union node union (no re-send tax). Ladder:
E0 offline census → E1 issue-order flip → E2 slot-last schedule (= NR-13's F-D) → E3
true two-round dispatch only if E0 shows payload and E1/E2 leave interference.

**E0 census (sweeps/predict_phase_split.py, offline, partition-asserted against
build_fused_metadata's U_mat).** Real Qwen3 layer-92 pernode trace, 4n16r k8 G=128
(matrices 737b5c/0971f3/2d7aee/f10c79):

| budget | phase1 (res+shared) | phase3 (pref-only) | pref frac | phase2 direct | w:t ratio | p3 slivers |
|---|---|---|---|---|---|---|
| b1  |   39.5 MiB |   3.9 MiB | 0.089 | 352 MiB | 8.12 | 19 |
| b8  |  310.6 MiB |  41.9 MiB | 0.119 | 320 MiB | 0.91 |  8 |
| b32 | 1253.5 MiB | 156.8 MiB | 0.111 | 320 MiB | 0.23 | 12 |
| b64 | 2498.1 MiB | 319.4 MiB | 0.113 | 384 MiB | 0.14 |  4 |

Synthetics: fanoutskew pref frac 0.19–0.25 (the heaviest), nodeskew 0.04–0.08,
remotefrac 0.04–0.05 (40+ sliver chunks at low budgets). Two structural reads:

1. **~89% of real-trace union rows are resident-needed and travel phase 1 under the
   policy regardless** — the deferrable phase-3 payload is thin (≈42 MiB at b8), and
   at b1 (the weight-dominated regime where ordering matters most) it is 3.9 MiB
   spread over mostly sub-256KiB relay slivers. E3's two-round machinery therefore
   buys little on this trace: E1+E2 carry the ordering value. Falsifier for this
   demotion: a routing family with pref frac ≳ 0.3 and non-sliver phase-3 chunks
   (fanoutskew b8/b64 is the closest, 0.25 with 8 MiB-scale chunks — the place to
   test E3 if it is ever built).
2. **Weight legs out-bytes the token wire 8:1 at b1, cross over near b8** — issue
   order is exactly the lever NR-13 fact 6(b) implicates, and the census also finds
   real multi-member fan-out groups on the trace (multi=1, max_fan 2–3), so mcast
   auto-resolution stays live.

Per-rank prefetch-slot GEMM-row fractions on the trace reach 0.53 (ranks 7/12/13),
so E2's tail effect (deferring slot problems also defers their output tiles) is a
real exposure — measured by the in-capsule A/B, predicted per-rank by the census
JSON (`pref_rows_per_rank`).

**E1 (driver-only, test_moe_moonep_fused_traffic.py `--weight_issue_order
tokens_first`; requires push + tiles gate).** The fused forward (token a2av + GEMM)
is host-enqueued BEFORE the weight push; the epoch is peeked
(`WeightPushMulticast::epoch()+1`, double-asserted after the fact). Pure
FIRE-ordering — no completion waits between the flows (the NR-13 lesson). Two coded
invariants: (a) DEADLOCK RULE — w_stream may wait only on the iteration-start event;
any later main-stream event would order the weight issue after the GEMM whose slot
tiles spin on those signals; (b) QUIET DEFERRAL — iteration i's weight nbi tail is
now quieted only by iteration i+1's a2av barrier_all. Benign in this benchmark
(immutable home rows, monotonic GEQ epochs, same-value rewrites, gw_end drain), NOT
free for a real multi-layer model with changing weights: an integration needs an
explicit wire-issued event or quiet. Under tokens_first, prefetch_ms is a concurrent
issue window and gate_ms ≡ 0 — only e2e_ms compares across issue orders. Residual
softness: host order ≈ NIC order is not a hardware guarantee; if nsys shows weight
puts overtaking, the follow-up is a `tokens_first_strict` wait on an op-exposed
wire-ISSUED event (still not a completion wait).

**E2 (= NR-13 F-D, `FLUX_A2AV_SCHED_PREFETCH_LAST=1`).** Bijective output-index
remap in `calc_sorted_problem_schedule_v2` (workspace_util.cu): all resident
problems precede all prefetch-slot problems, stage-major within each class; scoped
to the weight-gate branch (join mode and non-moonep arms are bit-identical no-ops).
Order-independence audit: problem_idx travels inside each schedule entry; gating
cumsum, weight gate, and output scatter key off problem_idx, not schedule position.

**Arms** (variants.py): `moonep_fused_push_auto_gated_{tokfirst,slotlast,
tokfirst_slotlast}`; spec `sweeps/specs/moonep_fused_ladder_pm4n_trace_iso.yaml`
(4 arms in-capsule vs auto_gated, trace b1/b8/b64 k8, isolated+e2e).

**Ladder capsule 20260814-145605 (one binary, 24/24 ok, correctness green,
deterministic=0).** Isolated max-rank e2e_ms means (and e2e-mode in parens):

| budget | auto_gated | tokfirst | slotlast | tokfirst_slotlast |
|---|---|---|---|---|
| b1  |  4.75 (4.98) | 5.46 (4.77) | 4.80 (4.76) | 5.35 (4.87) |
| b8  |  6.48 (6.82) | 7.00 (6.85) | 6.40 (6.99) | 7.05 (6.84) |
| b64 | 26.11 (26.55) | 22.19 (22.10) | 22.24 (22.13) | 22.24 (21.95) |

Three findings:

1. **The E1 falsifier FIRED**: tokens_first regresses isolated b1 by +15% and b8
   by +8% — precisely the weight-dominated regime the head-start argument targeted.
   Mechanism reading: in isolated (cold-wire, inference semantics) the legacy order
   issues the dominant weight legs while the token side is still in stage-1/2 host
   work, i.e. weights-first already IS the right priority when weights are the
   critical path; tokens_first delays weight landing behind the forward call's
   ~0.35 ms put-issuance window and the slot-tile spins pay for it. In pipelined
   e2e mode the regression vanishes (b1 4.77 vs 4.98 — mildly positive) because
   adjacent iterations already overlap the flows. tokfirst stays a non-default
   ablation arm.
2. **E2 slotlast is the strict-win knob**: isolated b1/b8 within noise (−1.3% at
   b8), −14.9% at b64 (26.11 → 22.24), and it needs no driver change. At b64 the
   token wire dominates (E0: w:t = 0.14) and the baseline's loss is slot tiles
   blocking the wavefront while their weights/rows are in flight — moving them
   last recovers ~3.9 ms.
3. **The two fixes are NON-ADDITIVE at b64** (all three treated arms land at
   ~22.2) — they recover the same stall from two sides (wire order vs consumption
   order), consistent with NR-13's "the pathology is ordering, not bytes".
   Combined also inherits tokfirst's small-budget regression → the recommended
   default is **slotlast alone** (`FLUX_A2AV_SCHED_PREFETCH_LAST=1` on the gated
   arms); canonicalization (flipping the knob default) left to a user decision.

**TODO-extension (user-flagged): the E3 lane cap.** A two-round dispatch needs
per-round gating lanes — `a2av_signal_buffer` [2W] and a [E, 2W] gating cumsum —
but the tile gate's ballot is a single warp: 2W ≤ 32 → **W ≤ 16**. Fits 4n16r with
zero headroom; before any E3 work at larger W the gate needs a second ballot warp
(or two-pass lane scan). Flagged 2026-08-14, revisit when node count grows.

**Falsifiers.** E1: a capsule where tokens_first regresses b1 — **FIRED in
20260814-145605 (isolated mode)**; the surviving claim is only the pipelined-e2e
neutrality and the b64 win. E2: a capsule where slotlast regresses a budget whose
per-rank pref-row fraction is ≤ 0.1 (tail effect should be invisible there) — not
observed. Census: any trace slice with pref frac ≳ 0.3 revives E3 (also demoted by
the capsule: tokfirst ≈ slotlast at b64 shows ordering, not a second round, was the
lever).

**Confidence:** census **measured** (offline, partition-asserted); E1/E2
**measured** (capsule 20260814-145605, one binary, correctness green). Open: nsys
confirmation that slotlast's b64 win is slot-tile spin relocation (tile-trace
sidecar rerun), and the b8 e2e +2.5% slotlast wiggle (single-capsule noise scale).
