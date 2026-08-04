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
