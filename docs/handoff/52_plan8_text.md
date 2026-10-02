# Plan 8 text (as approved 2026-10-01; preserved here when the plan file was replaced by plan 9)

Log of record for plan 8: `52_plan8_roundtrip_removal.md`. Plan 9: `53_plan9_graph_device_wire.md`.

# Plan 8: remove the per-layer host round trip (plan 7 stages B and C), deadlock-first, Nsight-verified

DRAFT 2026-10-01. Continues plan 7 (whose stage 0, A1, B4 and the device-metadata hang fix are done; C1, C2b, D1
written and gated, uncommitted). A1b folds into B3. Log of record: new handoff 52 (research tree); design rules R1-R5
and the hang analysis: handoff 51. Code: lopep `sglang-dev` (no numbers in lopep). Status: for user approval.

## Context

- 4n decode at 1 MiB (256 tokens per GPU): ours 2.76 ms per MoE layer vs stock 1.52 ms (round 9: device tables + one
  barrier; 0.59x). The round-8 Nsight capture shows the GPU idle 49 % of each layer-step, and the first size-dependent
  copy starts a median 719 us after the planning kernels finish (p10 674, p90 762; host tables, instrumented): the
  host must sync on the counts, build tables, then issue.
- Plan 8 makes every size the copies need GPU-computed (stage B), has a proxy thread issue the copies from
  GPU-written descriptors within tens of microseconds of planning (stage C3), and removes the per-layer planning sync
  (stage C4: capacity verdict deferred to once per forward). Paper semantics are unchanged: only WHERE sizes are
  computed and WHO issues the copies change.
- Every hang of plan 7 had one shape: a resident kernel spins on work whose launch has not happened, and the launch
  path waits on that kernel. Plan 8 makes GEMM tiles spin on proxy-issued work in every layer, so the deadlock model
  is a design input.
- Estimates (handoff 51 section 6): B -0.35..-0.45 ms/layer, C3+C4 -0.4..-0.6 ms/layer at 4n, bringing the layer to
  ~1.8-2.0 ms, still above the 1.37 ms needed for 1.1x; the Nsight analysis after each stage names the next floor
  (input to plan 9: D2 per-layer graph, swap decision A2, wire latency). Risk: without the sync the layer is bound by
  the slower of the GPU and the main thread's enqueue time (metric T3 decides).

## Loyalty contract
L1 outputs = torch fp32 reference within bf16 tolerance (bad_rows 0), token agreement vs stock; L2 routing and both
exchanges untouched; L3 calibration files unchanged; L4 swap decision bitwise = `swap.py`; L5 split values identical
host vs device; L6 put sizes / counts / union layout unchanged; L7 issue ORDER and gating unchanged; L8 no per-step
allocation; L9 every inter-node put a consumer gates on stays BLOCKING `putmem_signal_on_stream`, payload randomized
per step in every correctness cell; L10 capacity verdict precedes every write, identical on every rank, forced
growth passes.

## Deadlock model (classes; each with prevention and gate)
- H1 lazy code load beside a spinner (P6): explicit warm-up (kernel registry, NVSHMEM priming, one forward per bucket).
- H2 implicit device sync on the main thread while proxy work is pending: no host read of a GPU value inside a forward
  except the end-of-forward verdict; persistent buffers; growth quiesces the proxy first.
- H3 lock held across a blocking call (NVSHMEM host mutex): lock only around non-blocking enqueues.
- H4 proxy starved or asleep: no pinning, forward-active busy polling, fail fast.
- G1 GPU block-scheduler head-of-line: kernels released beside a spinner must fit beside one of its blocks (R5).
- G2 channel wait-order inversion (NR-02 class B): front-end waits at a shared hardware-queue head block same-rank
  writers enqueued later (R6 / R6' watermark rule, R7 / R7' device joins).
- E1 epoch / done-word reuse; E2 descriptor-ring overwrite; E3 buffer reuse without the per-layer sync.
- V1 deferred verdict (aborted step must make every spinner finish; verdict identical on every rank).
- W1 wire ordering (CLAUDE.md rule 5); F1 fabric stall (not a deadlock).
(10-01 outcome: R7' was wrong; probe P8 showed a dependent LAUNCH behind a spinner holds the queue head like a
front-end wait (class G3); drains made mandatory, p8-c34 e2ebb95.)

## Stages
- W: warm-up (kernel registry, NVSHMEM priming, activation variants), probe P7, registry + R5 tests.
- B: B0 persistent buffers; B1/B2 dispatch plan block on the device; B3 combine plan on the device (+A1b); B5 host
  builders deleted after word-for-word compares (NOT done: folded into plan 9).
- C: C1 batched copy-engine issue; C2b proxy hardening; C2 descriptors (simplified away: closures read the plan ring);
  C3 proxy issue from the plan blocks; C4 deferred verdict, no per-layer sync, SGLang redo hook.
- Nsight: captures N0 / NB / NC3 / NC4 with metrics T1-T8 (`52_timeline.py`).
- Order: parallel lanes L-D1, L-W, L-B1, L-B3, L-C; then merge B -> stage check + NB; C1 + C2b -> C3 -> stage check +
  NC3; C4 -> stage check + NC4 -> milestone M2 (4/8/16n, decode + prefill, 1/2/4 MiB; NOT run: folded into plan 9).

## Out of scope (plan 9 candidates)
A2 parallel swap decision; D2 per-layer CUDA graph; the two NCCL gathers as NVSHMEM on-stream gathers; the
one-routing-exchange redesign (closed 08-29); any change to the wire protocol or round schedule.
