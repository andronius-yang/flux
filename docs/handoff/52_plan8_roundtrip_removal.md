# Handoff 52: plan 8, removing the per-layer host round trip (stages W, B, C) - log of record

Plan: `~/.claude/plans/the-kv-cache-size-cozy-hamster.md` (plan 8, approved 10-01 09:40). Design rules R1-R5 and the hang
analysis: handoff 51; review amendments R6' / R7' and the C-stage protocol: plan 8, "Stage C". Code: lopep worktrees
(branch per lane, base `p8-base` 34c9f4c = the plan-7 C1 / C2b / D1 work), never pushed; `sglang-dev` stays at 8d3a8a0
until a lane lands with its gates. Numbers live here, never in lopep.

## Lanes (10-01, state at 12:20)

| lane | tree / branch | state |
|---|---|---|
| L-D1 one-barrier stress | `lopep_d1rt` (`bin/dev_ct128`) | DONE 10:08, green at 8n / 16n (below) |
| L-W warm-up, kernel registry | `lopep_w` / `p8-w` 861f234 | merged into p8-int, gated |
| L-B1 dispatch plan block + A1b | `lopep_b1` / `p8-b1` 3e1138b | merged into p8-int, gated |
| L-B3 combine plan block + A1b | `lopep_b3` / `p8-b3` 5b1d066 | merged into p8-int, gated |
| L-C proxy hardening | `lopep_c` / `p8-c` 65d6d4b | merged into p8-int, gated |
| integration (stage B) | `lopep_int` / `p8-int` 642b6b6, + NH-1 fix 4daa191 | gated; decode stage check round 10 (below) |
| C3 plan-driven wire (dispatch + combine) | `lopep_c3` / `p8-c3` 0619055, 50c0040 | 6 / 6 + 9 / 9 gates PASS |
| C4 deferred verdict | `lopep_c4` / `p8-c4` 8dca008 (agent) | merged with C3 as `p8-c34` 4b5219f; gate chain running |
| NH-1 serving hang (pre-plan-8) | fix in `p8-int` 4daa191 | ROOT-CAUSED + FIXED + VERIFIED (below) |

## L-D1: one barrier per layer at 8 and 16 nodes (10-01, jobs 59164174 8n debug, 59164173 16n regular, 40 GB)

Binary `bin/dev_ct128` (device tables on, proxy off), `LOPEP_LAYER_BARRIERS=1`; gate cells of `gate50.sh` (full batches,
60 steps, forced growth) plus varying-count cycle cells (`hang_probe2.sh`, reference on); then the same failing cells
with three barriers as control. Report `logs/p50/gate_d1_{8n,16n}_report.txt`.

| cell | 8n, 1 barrier | 16n, 1 barrier | control, 3 barriers |
|---|---|---|---|
| `s4096_swap_dev`, `flip_staged` | pass | pass | - |
| cycle, reference on, 12 steps / cycle, forced growth | pass / pass | pass / pass | - |
| `staged_ref_dev` (reference on) | 4 rows over tolerance in 32 ranks | 3 in 64 | 8n: 3 / 16n: 3 |
| `flip_ref` (reference on) | 11 rows in 32 ranks | 21 in 64 | 8n: 11 / 16n: 21 |

Every failing row is the bf16 tolerance edge of the torch reference (max error 0.0136-0.0150 vs atol 1e-2 + rtol
2e-2), with identical counts under three barriers. Verdict: one barrier is correct at 8 and 16 nodes; per plan 8
every later plan-8 arm uses one barrier, and the default flips with the lane's lopep commit.

## L-B1: the dispatch plan block on the device (10-01, job 59164556, 4n interactive, 40 GB)

`a2av_dispatch_plan_kernel` (one block, 256 threads; `src/dispatch/sort_util.cu`) writes the fixed-layout block of
`DispatchPlan` (`sort_util.h`): header (M, send rows, self copy), send-segment offsets, round-0 operands per local
peer, per target node (relay chunk bounds, own-only flag, own source, wire destination, <= 2L pull pieces with
absolute source rows), per source node (gateway window, staging offset, forward destinations). Launched inside
`derive_routed_meta`, its D2H rides the existing event. The wire and the wave pack read only the block; the host
builder (the old tables, now emitting the same block) remains as the reference (mode 0 / 2). A1b dispatch side:
stage 1 reads the expert of each copy from the routed ids on the device paths.

| gate | result |
|---|---|
| `tests/test_dispatch_plan_device.py` (W 4-64, i.e. 1-16 nodes; random, skewed, one / two experts; every rank; tight capacities -> error bits) | 90 cases identical to the numpy reference `routing.dispatch_plan_ref` |
| harness mode 2 (device block compared word for word with the host builder every layer-step), cycle counts, reference on, 8 steps x 3 layers, 1n and 4n | pass, 0 mismatches, 0 bad rows |
| harness mode 1 (wire uses the device block), 4n, cycle, reference on, swaps forced (`--skew-flip 1 --flip-every 2`) | pass, 0 bad rows, 55 swap moves |
| harness mode 1, 4n, forced growth (`--caps-scale 0.5`), swaps | pass, 3 growths |

Open in this lane: A1b stage-1 change not yet GPU-gated (committed 3e1138b after the gates above); B5 deletion of the
host builder after L-B3 merges; the plan kernel supports up to 16 nodes (static shared memory; 32 nodes needs a
dynamic-shared-memory variant).

## NH-1: serving hang of the round-9 binary in a swap step (10-01 10:35, job 59165935, stage-B check, arm r9)

Arm r9 = `lopep_d1rt` (`bin/dev_ct128`, device tables, proxy off, one barrier, NO warm-up), decode 1024 running per
rank, wave 1, first decode step after a prefill step (~4040 tokens per rank). Waves 256 / 512 and 1024 wave 0 passed;
round 9 ran the same configuration twice without a hang, so the hang is intermittent. The NCCL watchdog (600 s)
aborted the processes, which logs no `Traceback`, so `lib49.sh` would have waited out the 25 min client timeout (the
client was killed by hand at 10:48; the chain went on).
Evidence (`logs/p50/threads_j4nD10_kr9_{1024,all}.txt`, `locals_j4nD10_kr9.txt`, `cudagdb_bt_j4nD10_kr9.txt`):
- all 16 schedulers: main thread in `derive_routed_meta` `cudaEventSynchronize` of the NEXT layer-step (host ahead);
  every rank timed out on the same NCCL collective, SeqNum 91368, `_ALLGATHER_BASE` 128 -> 2048 (the next step's
  loads gather): no rank skew, every rank in the same layer-step l.
- node 0: DP0 combine GEMM (152 blocks, all resident) + pack + pre-reduce; DP1 DISPATCH GEMM (blocks 40-135 resident,
  136-199 pending); DP2, DP3 pre-reduce. Nodes 1-3: no kernel resident (streams parked on front-end waits, i.e. on
  node 0's combine wire).
- spin points (SASS): DP0 combine GEMM: `gate_of_expert[g] >= 0` then `ld.strong.sys` of a 64-bit word until
  `>= epoch` (W2 weight gate of an incoming slot); DP1 dispatch GEMM: group `>= weight_gate_group_start`, then the
  same 64-bit gate spin (W1 gate). Pack spins on a 0/1 flag (GEMM tile done), pre-reduce on pack flags.
- reading: DP1 waits a W1 gate of step l although every possible sender (its node peers DP0, DP2, DP3) is past its
  step-l dispatch GEMM, hence past its `lane_push(k=0)`; DP0 then waits DP1's W2 push (issued only before DP1's
  combine GEMM). Static review of `lane_device.cu` / `swap_decide.cu` / `swap.py` found no lost-update or ordering
  path (gate words monotonic, commits wait the push before the next arm, decision deterministic, epochs advance once
  per step on every rank). Open: the gate words' values at the hang (the process died before they were read).
- tooling added: `logs/p50/stall_watch.sh <job> <report>` (server log idle 90 s during a wave -> `gate_probe.sh`
  registers + 64-bit words around the spin addresses of every resident GEMM, `sgl_threads.sh` py-spy / gdb /
  cuda-gdb, then kills the client so the allocation is not wasted).
Relevance to plan 8: the r9 runtime has no warm-up (`lane_device_preload` is never called in serving) and its main
thread holds the GIL while blocked in the planning sync (the dump thread cannot run; NCCL logs "Could not acquire GIL").
The b8 arm (with warm-up) passed the same cells; one pass is no proof. C4 removes the planning sync, so `_nt_pin` /
`_sd_pin` single pinned buffers with non-blocking copies become E3 races there (listed for the C4 audit).

## L-INT: merged tree `lopep_int` / `p8-int` (642b6b6 = C + B1 + B3 + W), 4n gates (10-01, job 59165278)

Unit (1 GPU): dispatch plan 90 cases, combine plan 619 blocks + 2476 msplit / weight-gate tables, meta 104, swap decide
201, lane 149 swap steps, capacity 31, kernel registry (64 library kernels, all loaded), R5 resources: all pass.
Harness 4n (`--token-mode cycle --swap 1 --skew-flip 1 --flip-every 2 --ref 1`, 10 steps x 3 layers, warm-up 68
kernels / 36 layer-steps): `LOPEP_DEVICE_META=2` (word compare), `=1`, `=1` forced growth (`--caps-scale 0.5`, 3
growths), `=1` proxy on + enqueue audit, `=1` full batches: all PASS, 0 bad rows, 880 swap moves.

## C3 dispatch side: plan-driven dispatch wire (`lopep_c3` / `p8-c3` 0619055, job 59165278)

`LOPEP_WIRE_PLAN=1`: the dispatch wire program is ONE proxy step posted at today's first wire post point; it waits
(host poll) for its plan block in a 256-slot pinned ring that the planning chain fills (copy, then a stream write of the
slot epoch into mapped memory) and runs every op in place. Gates (same harness cells): proxy off, proxy on, + enqueue
audit, + 200 us proxy delay, `CUDA_DEVICE_MAX_CONNECTIONS=1`, forced growth (`--caps-scale 0.5`, 3 growths): 6 / 6
PASS, 0 bad rows, 880 swap moves (`logs/p50/c3_chain.log`, binary 3ef1a45be4e9). With the planning sync still in place
the step finds its block already published; the point is that posting it needs no host copy of the counts (C4).

### NH-1 recurrences (same allocation 59165935, 10-01)
- 10:54 stock arm (`baseA`, no lopep code): one scheduler (node nid003960) SEGFAULTED in SGLang `prepare_for_decode`
  (`schedule_batch.py:1671`, `seq_lens.clone()`), the other 15 then timed out in the next DP-attention gather (NCCL
  SeqNum 11083, 6 -> 96 ints). First segfault in 235 server logs since September; no ECC / remapped-row errors on the 16
  GPUs. Counted as an allocation anomaly, not as NH-1.
- 11:16 arm b8R (`lopep_int`, warm-up ON): hang at the same point as r9 (1024 running per rank, wave 1); the stall watcher
  probed it (`logs/p50/gates_j4nD10_d30_kb8R.txt`, `threads_j4nD10_d30_kb8R.txt`). Node 0: DP1 in the COMBINE GEMM (all 152
  blocks) + pack + pre-reduce; DP0, DP2, DP3 in the DISPATCH GEMM; NVSHMEM `barrier_on_stream` kernels resident on several
  ranks (one beside DP3's dispatch GEMM); nodes 1-3 otherwise idle. DP1's spin: W2 gate of expert slot 2
  (`gate_of_expert -> 2`), word = 14600 while every other raised gate word of DP1 (both matrices) = 14602 and the pad words
  = 0. So DP1's own `lane_arm` at epoch 14602 classified slot 2 as INCOMING (it raised every other slot), and no peer's
  W2 push of epoch 14602 ever landed; the word's last write was two epochs earlier. DP2's and DP3's gate arrays are all
  14602 (their dispatch GEMMs do not wait on a weight gate, so they wait on dispatch data). The b8 first pass, r9R and
  every round-9 arm passed the same cells: NH-1 is intermittent, present before plan 8, and independent of the warm-up.
- Missing evidence: which layer-step each rank's host and GPU are in (the dump thread cannot run: the main thread holds
  the GIL inside the planning sync) and the node's decision block (move lists) at epoch 14602. Next probe: release the
  GIL in the pybind of the planning sync, extend `SwapLane.debug_state` to the device lane (epoch, pulls, gate words,
  push counters, the node's move lists from the block), and have the stall watcher touch the dump triggers on every
  rank before it probes.

## C4 lane (agent, `lopep_c4` / `p8-c4` 8dca008) and the C3 + C4 merge (`p8-c34` 4b5219f, same worktree)

C4 (`LOPEP_DEFERRED_VERDICT=1`, default 0): a process-wide device verdict block (`src/core/verdict.h`) opened per
forward; every layer's demands kernel ORs its violation mask into it, the first violating layer latches its
demands, and from there the forward is degenerate (counts, splits and the dispatch plan block zeroed: zero GEMM tiles,
signal-only wire, zero-row receivers, barriers and epochs as usual). The plan kernels' consistency checks become
assert bits. One readback per forward (`SharedComm.forward_check`, all ranks confirm by all-reduce); on a violation
`recover()` quiesces, grows, re-primes with the warm-up and the caller redoes the forward (SGLang patch: tp_worker
`run_forward`, the redo through `_forward_raw`; `LOPEP_FORCE_ABORT=<layer>` for tests). Outside the wire nothing reads
a count on the host: capacity-bounded GEMM views, both GEMMs always launched, pack / pre-reduce sizes and the combine
wave-adapt decision on the device, swap moves counted on the device and read once per forward, GEMM-start marks as
stream writes.
Merge decisions: the dispatch ring copy is published AFTER the demands kernel (which zeroes a degenerate block); the
zeroing keeps the plan's sequence word (otherwise the ring waiter of a degenerate step would never match its slot:
a 60 s abort); the pack / pre-reduce read the device block through C4's fields in both modes. With both knobs
(`LOPEP_DEFERRED_VERDICT=1 LOPEP_WIRE_PLAN=1`) `derive_routed_meta` no longer synchronizes and the combine never waits
for its plan block on the host: the per-layer planning sync is gone. With the proxy on, `LOPEP_PROXY_DRAIN=0` is
required for the benefit (the dispatch-end drain would otherwise wait for the plan); the drain before the combine
receivers stays (it bounds the host lead to under one layer: it waits until the GPU has published that layer's
plans; R7' fused receivers would remove it). Gate chain: `logs/p50/c34_chain.sh` (unit + 13 harness cells, output
digests vs the knob-0 reference).

## Capture lesson (NB attempts 10-01 11:19 and 11:24, both lost) - CORRECTED
Root cause: two drivers on one allocation. The stage-B prefill driver (`jobS8.sh ... P`, tag P10) was chained after
the decode driver and started on 59165935 at 11:19:14; the capture chain waited only for the decode driver's pid and
started at the same second. Each driver's `clean` (lib49: SIGKILL of every step of the job) killed the other's
servers: both captures lost their report and the P10 prefill arms are CONTAMINATED (r9 run 1: 8 s / TTFT 0, stock 96 s,
b8R server failed at the release) - P10 is void and has to be rerun. Rule: before starting a driver on an allocation,
list every driver process that targets that job id (`ps -u $USER -o pid=,args= | awk` on the job id), not only the
one being waited for. The capture driver `logs/sglang/jobN11.sh` (node 0 in its own srun step under nsys with
`--capture-range-end=stop`, SIGTERM to the node-0 server after the wave, clean only after nsys exits) is kept: it also
makes the report independent of the other nodes' teardown. NB and the P10 rerun move to a later allocation.

### NH-1 reproduction (job 59167796, 10-01 11:35-11:44, `lopep_int` + GIL release + `LOPEP_DUMP=1`, 1024 per rank directly)
Two servers, two hangs within ~15 s of the first 1024-per-rank wave (reproducible). Dumps: every rank's host at the
same layer-step (1270 / 1355) with the same lane epoch (step + 1): no epoch skew. The device part of the dump blocked
(its synchronous read queued behind the hung legacy default stream, which SGLang computes on): fixed (reads on a
non-blocking stream into pinned buffers allocated at enable time, 5 s timeout), rerun `logs/p50/nh2_hunt.sh`.
Gate probes, all three hangs alike: the stuck kernel is a COMBINE GEMM spinning on the W2 gate word of one incoming
slot, holding exactly epoch - 2 while every other raised gate word holds the epoch (b8R: 14600 vs 14602, slot 2;
h1a: DP4 1268 vs 1270, slot 2; h1b: DP0 1353 vs 1355 slot 6, DP1 1353 vs 1355 slot 0). The node peers that could owe
the push sit in their DISPATCH GEMMs, before their W2 push point; in h1a DP5 / DP7 spin on a per-source arrival word
array whose first entry is one run behind (1310 vs 1311). Working reading: a wait cycle (receiver's combine GEMM ->
peer's W2 push -> peer's dispatch GEMM -> one source's dispatch data), not a lost push; the decision block dump
decides whether the peer owes the push.

## C3 combine side (`lopep_c3` / `p8-c3` 50c0040, job 59167398, binary 0514d2d5a19b): 9 / 9 PASS
`LOPEP_WIRE_PLAN=1`: the combine plan block's host copy goes into a 256-slot pinned ring published by its epoch
(`src/core/plan_ring.h`); the conv / wire / intra posts keep today's points (R6') but each step waits for its block
when it runs and reads its sizes from it; pack and pre-reduce read their row ranges from the device block.
`LOPEP_WIRE_PLAN=2` additionally gives the combine wire section no host copy at all (receivers wait every lane).
Cells (4n, cycle counts, swaps forced, reference on): level 1 proxy off / on; level 2 proxy off / on, + enqueue audit,
+ 200 us proxy delay, `CUDA_DEVICE_MAX_CONNECTIONS=1`, forced growth (3 growths), full batches: all PASS, 0 bad rows.

### NH-1 ROOT CAUSE (10-01 12:00, jobs 59168202 + 59168891): a swap-lane cross-rank wait cycle; fixed by pushing W2 early
Bisection (job 59168202, `lopep_int`, 1024 per rank, `LOPEP_DUMP=1` with the async device reads): swap ON hung in wave 1
(5 of 5 attempts since 10:35); swap OFF (`LOPEP_SWAP_OFF=1`, runtime debug override) ran all 4 waves clean (decode
step 223.1 ms). Dump of the swap-on hang (`logs/p50/dump_j4nDh2_d30_kh2a.txt`), stuck step epoch 7412: every rank
holds the IDENTICAL decision block (2 rounds, 10 moves; node 0: rank 0 pulls slot 4 from rank 1 (expert 71), rank 1
pulls slot 8 from rank 0 (expert 3)). Gate words: rank 1's W1 and W2 gates of slot 8 = 7412 (rank 0 pushed both:
rank 0 is past its W2 push point); rank 0's W1 gate of slot 4 = 7412 but its W2 gate = 7410 (rank 1 pushed W1, never
reached its W2 push). Kernels: rank 0 in the COMBINE GEMM (W2-gated, all blocks resident, spinning); rank 1 (and the
non-swapping ranks 2, 3) in the DISPATCH GEMM, every weight gate raised, waiting on inbound dispatch data; nodes 1-3
idle on front-end waits.
Mechanism: the staged lane pushed W2 right before the SENDER's combine GEMM, so a receiver's W2-gated combine GEMM
depended on the sender's entire dispatch phase (its dispatch GEMM, i.e. its inbound data). That data includes the
receiver's own share of the step (rank 0 is node 0's local-rank-0 gateway: the relay chunks from the other nodes
reach ranks 1-3 through rank 0's forwards, issued after rank 0's dispatch GEMM on rank 0's side streams). With rank 0's
gated combine GEMM resident on the whole device, the cycle closes: rank 0's GEMM <- rank 1's W2 push <- rank 1's
dispatch GEMM <- data forwarded through rank 0. (Which resource of rank 0 the forward lacks - SMs or a hardware-queue
position - is not identified; the fix does not depend on it.) Intermittent because it needs a swap step whose
receiver is a gateway that reaches its combine GEMM before its forwards drained: likelier at 1024 per rank.
Fix (`python/lopep/swap.py`, `W2_EARLY`, knob `LOPEP_W2_EARLY`, default 1): both matrices are pushed before the
sender's DISPATCH GEMM (device lane: `lane_push` for k = 0 and 1; host staged lane likewise); nothing is pushed before
the combine GEMM. No combine GEMM then waits on any peer's dispatch progress; the W1 gates already had this shape. The
push precedes the sender's own commits (stream order) and follows the receiver's previous-step commit (the step's
routing exchange orders them), so staging reuse is unchanged. Cost: the W2 copy moves onto the sender's pre-dispatch
critical path (swap steps with outgoing moves only). Verification: `logs/p50/nh3_verify.sh` (harness swap cell with
slot-content asserts, then two servers at 1024 per rank x 4 waves).

### C3 + C4 gates (job 59168080, binary dd7219dc4072): first cells
Unit: all pass (the deferred-verdict test's zeroing assertion updated for the kept sequence word). Harness 4n (24
steps x 3 layers, cycle counts, reference on): `ref` (knob 0), `v1` (deferred verdict), `v1_abort`
(`LOPEP_FORCE_ABORT=1`: forward aborted at MoE layer 1, redone: 1 redo), `v1_grow` (`--caps-scale 0.5`: 3 growths,
2 redos through the verdict): all PASS, 0 bad rows. Output digests (`--out-hash`) differ between EVERY pair of runs,
knob 0 vs 1 and within knob 1; so bitwise digests are not a usable cross-run criterion on this path. Hypothesis, not
confirmed: the completion-bucketed receiver folds each token's top-k contributions in lane-completion order
(timing-dependent fp32 summation). The redo gate therefore rests on the reference check (bad rows 0) plus "redos /
growths as expected"; a same-knob twin pair would test the hypothesis.

### Round-8 timeline (agent, `docs/handoff/52_timeline.py`, report `logs/sglang/nsys50/timeline_b1.txt`)
T1 reproduces 719 us (p10 674, p90 761) with planning end = the last plan-producer kernel before the dispatch GEMM
(`a2av_meta_pass2` in round 8) and first wire = first P2P / DtoD copy or put kernel; that first copy is an 8-byte
control word, the first payload copy comes at 913 us (863-961). T8 floor list (round 8, medians): GEMM1 end ->
GEMM2 start 1003 us (768 idle: host combine-derive tables after push0, launches); step start -> planning end 807
(289 idle); planning end -> first wire 720 (601 idle: plan_meta, host check, sync); GEMM2 end -> step end 573 (puts,
pre-reduce, barrier); first wire -> GEMM1 368; GEMM1 187; GEMM2 82. T6: no lookup / load over 100 us after the first
step, but first launches of the vectorized silu / mul took 139-194 us (lazy-load signature; the W warm-up must cover
the activation variants). T7: nothing over the R5 limit; `a2av_combine_pack<false>` sits exactly at 1024 x 32 regs.
NH-1 fix VERIFIED (job 59168891, `lopep_int` p8-int 4daa191, swaps on, `LOPEP_W2_EARLY=1` default): harness swap cell
PASS (880 swap moves, W1 / W2 slot contents asserted after every swap, 0 bad rows); two servers at 1024 per rank x 4
waves each: no hang (384 intervals each), decode step 231.2 / 230.9 ms (swap off on 59168202: 223.1 ms; stage-B arm
with the late push: 228.5 / 230.7 ms).

### C3 + C4 gate chain: two hangs, one cause (H1 lazy load in the warm-up, predicted class)
`ns_abort_grow` (no-sync config + forced abort + `--caps-scale 0.5`) and `ns_c1` (no-sync config +
`CUDA_DEVICE_MAX_CONNECTIONS=1`) hung in the WARM-UP, before the first timed step: main thread in `cudaLaunchKernel`
-> `cuLibraryGetModule` (lazy code load) of torch's `silu` inside `activation` (`layer.py:26`) while the dispatch
GEMM was resident spinning on proxy-issued wire (16 of 16 ranks in `ns_c1`). The no-sync config runs with
`LOPEP_PROXY_DRAIN=0`, so the proxy's posted dispatch work was not drained before the main thread's first launch of
an activation variant; P6 then holds all new GPU work behind the load: deadlock (class H1 of the plan's table, and
the round-8 T6 signature: silu / mul first launches of 139-194 us). The other no-sync cells passed by the luck of
which warm-up case first launched each variant. The single-row activation (m = 1) is the contiguous, vectorized
silu / mul variant - a real decode case, so it could also first-launch in steady state.
Fix (`p8-c34`): (1) the warm-up forces the per-layer drains on whatever `LOPEP_PROXY_DRAIN` says
(`C.wire_proxy_force_drain`, around `SharedComm.warmup`; also covers the re-prime in `recover`); (2) the warm-up launches
the activation's variants (m = 1, 2, 3) before any layer step. Re-gated in `logs/p50/stage34_go.sh` (both swap cells +
the abort / growth cell), and the NC4 capture's T6 tripwire checks for loads after the warm-up.
C3 + C4 chain final (job 59168080, binary dd7219dc4072): PASS ref, v1, v1_abort, v1_grow, v1_flip_abort, v1_proxy,
ns, ns_flip, ns_full (9); HUNG ns_abort_grow, ns_c1 (H1 warm-up load, above); NOT RUN ns_delay, ns_audit (the 60-min
allocation expired). `p8-c34` now: 0ec1643 (kept sequence word in the test; SGLang hook falls back to the plain
forward for an older lopep_sglang), c9c7e2b (NH-1 fix + dump tooling, cherry-pick of p8-int 4daa191, applied also
to C4's always-gated deferred branch), 7226206 (H1 warm-up fix). Re-gate of the failed / unrun cells + the stage check
(decode, prefill, NB / NC4 captures): `logs/p50/stage34_go.sh` -> `logs/sglang/stage34.sh`, one driver per allocation.

### Re-gate (job 59170930, `p8-c34` 7226206, binary a723f3c7f78d)
`fix_flip_abort` PASS (880 swap moves, 1 redo); `fix_ns_flip` PASS (no-sync config, 880 swap moves);
`fix_ns_abort_grow`: no hang (the H1 warm-up fix holds: warm-up with 4 skipped cases ran), but FAILED after growth #3
(step 18, "MoE layer 0 of the forward", recover + re-prime + redo): every rank's proxy aborted with "dispatch plan
ring: sequence 274 not published after 60 s (epoch 18)" - the ring's fail-fast guard. Seq 274 is the first to wrap
back onto slot 18; its publish (D2H + epoch stream write in `derive_routed_meta`) never executed within 60 s, i.e.
something ahead of it on the planning stream (the caller's stream) was blocked, plausibly a resident kernel waiting on
proxy work queued after the ring waiter (a cycle through the proxy's in-order queue) on the redo-after-growth path.
NOT yet root-caused: the abort came before the 300 s stack dump. Only the growth + redo path (serving runs at
calibrated capacities with 0 growths: stage-check arms unaffected). Next: repro with the stack dump at 40 s and the
ring waiter's slot / consumed / published counters in the abort message.
`fix_ns_c1` (no-sync config + `CUDA_DEVICE_MAX_CONNECTIONS=1`): HUNG again, new signature, 16 of 16 ranks in the warm-up's
combine forward, host in `GemmCombineOp::wait_plan_block` (`cudaEventSynchronize` on the combine plan block's D2H
event, derived on the op-owned meta stream). With one connection every stream shares one hardware queue in enqueue
order (G2), so a proxy-enqueued front-end wait ahead of the meta-stream derive blocks it; the same cell PASSED on
`p8-c3` (`c3c_l2_c1`, `c3_plan_p1_c1`), so the inversion came with the C4 merge or the two fixes. Bisection running
(`logs/p50/c1_bisect.sh`: C3-only knobs on the merged binary, proxy without the ring, inline, W2 late, stage-B tree).
Production runs at 24 connections (the stage check is unaffected as a measurement), but C4 does not ship before this
is closed. Ring-stall repro (`nh4_ring.sh`, job 59171358): PASSED (3 growths, 3 redos) - the ring stall is
intermittent; its timed dump found no processes (nothing captured).
Re-gate final: PASS `fix_flip_abort`, `fix_ns_flip`, `fix_ns_audit`; HUNG `fix_ns_c1` and `fix_ns_delay` (200 us proxy delay,
default connections: 16 of 16 ranks in the warm-up, host in the next case's input copy = stream sync behind a stuck
warm-up step); FAIL `fix_ns_abort_grow` (ring abort). Bisection (CDMC=1, same binary): deferred verdict OFF passes with
plan ring + proxy, proxy alone, inline -> the warm-up hangs need the deferred-verdict path (C4's always-gated swap
lane in the warm-up, where the NH-1 port moved the W2 push, or the forced warm-up drains).
Bisection b4 (deferred verdict ON, `LOPEP_W2_EARLY=0`): still HUNG in `wait_plan_block` (16 / 16) -> the NH-1 port is
cleared; the inversion lives in C4's deferred-verdict path itself and was masked before the H1 fix by the earlier
H1 hang at the same warm-up point. Working hypothesis (unconfirmed): with the deferred verdict on, the swap lane takes
C4's always-gated path, so the combine builds its device gate map and waits on the host for the combine plan block
before the combine GEMM; that block is derived on the meta stream, which in a shared hardware queue sits behind
proxy-issued front-end waits on remote signals. Repro with the enqueue-order audit and gdb / cuda-gdb dumps:
`logs/p50/nh5_inv.sh`.

### C4 deadlocks root-caused: dependent launches hold a shared hardware queue (class G3, probe P8)
Diagnostics build (uncommitted, `p8-c34`): tagged checkpoint events (`LOPEP_DBG_EVENTS=1`, 14 per layer-step,
printed by any timed-out waiter with the main thread's and the proxy's phase), ring / quiesce timeouts with
counters. Runs (`logs/p50/nh8_diag.sh`, `nh9_diag.sh`, jobs 59178246 / 59178459):
- CDMC=1 harness (`diag9_c1`), all 16 ranks identical: `cp replay start 1=ok`, `cp replay end 1=PEND`, dispatch GEMM
  pending, combine `derive entry=ok`, `tables=PEND`; the proxy has issued everything (posted 2, done 2); the GEMM spins
  on its OWN self-arrival signal (slot = own rank, value 0). So the replay's first ops (self copy, own signal) were
  enqueued but never ran, and the queue was blocked between the derive entry and the tables kernel.
- Serving (`d30_kc4f`, 24 connections): ring abort "sequence 111 not published after 150 s"; replay end ok on the
  reporting ranks; nodes 2 and 3 (and DP6) spin in the dispatch GEMM's weight gate (all gate words at 59, target
  above), the others in the combine pre-reduce waiting for those ranks' conv lanes. nh8 (same config): DP0 / DP3
  spun on their self-arrival signal, as in the CDMC=1 cell. Different secondary states, one shape.
Probe P8 (`50_copy_probes/p8_deporder.cu`, `run_p8.sh`, `run_p8b.sh`; logs `logs/p50/p8_deporder.log`,
`p8b_deporder.log`), one GPU: a kernel spins on `main` waiting for a flag; then optionally one op on `main`; then on
`cp` an event, a 4 MiB D2D copy and the flag's stream write.

| op on `main` behind the spinner | 1 connection | 8 / 24 / 32 |
|---|---|---|
| none | released | released |
| trivial kernel (stream-ordered behind the spinner) | BLOCKED | released |
| one-warp spin join (the `wait_geq_kernel` shape) | BLOCKED | released |
| event record | released | released |
| event on `cp` FIRST, then the trivial kernel, then the copy (`pre12_dep`) | event ok, copy BLOCKED | released |
| front-end wait on another stream (NR-02 class B, control) | BLOCKED | released |
| any of the above with `cp` at the highest stream priority | BLOCKED at 1 | released |

So (G3, new): a kernel launched behind a still-running kernel of its own stream holds the head of its hardware queue
until that kernel finishes, exactly like a front-end wait; stream priority does not give a separate queue; at 8+
connections, 80 pool streams created first still put `main` and `cp` on different queues (consistent with a
round-robin assignment at creation: in a serving process which pairs collide is per-rank luck). `pre12_dep` is the
`diag9_c1` signature exactly. Rule R7' was wrong: a spin-kernel join avoids the front-end wait on the word, but its
LAUNCH is stream-ordered behind the GEMM and holds the queue all the same.
Root cause of every C4 hang (CDMC=1 warm-up, proxy delay, serving, plausibly the abort+growth ring stall): the
no-sync configuration ran with `LOPEP_PROXY_DRAIN=0`, so after the dispatch GEMM launch the main thread enqueued the
join, the lane commit and the activation (all stream-ordered behind the spinning GEMM) BEFORE the proxy, still
waiting for the plan block to be published, enqueued the GEMM's producers (self copy, round-0 puts, relay puts,
gateway forwards). Any shared queue then deadlocks. The bisection never separated the knobs: every deferred-verdict-off
cell ran with the drains on (default) and every C4 cell with them off.
Fix (`p8-c34`): the drains are mandatory with the proxy on (`lopep_proxy_drain()` always true; `LOPEP_PROXY_DRAIN=0`
aborts with a message; `wire_proxy_force_drain` removed); checkpoints behind `LOPEP_DBG_EVENTS` (default off). Rule
R8 (replaces R7'): the main thread enqueues nothing stream-ordered behind a resident spinner, and no front-end wait,
before every producer of that spinner that the proxy issues has been enqueued (a drain). Cost: under C4 the main
thread now waits once per layer, at the end of the dispatch, until the proxy issued the layer's wire, i.e. until the
dispatch plan block was published by the GPU; the pack and the GEMM are already enqueued then, so T1 (planning done ->
first copy) keeps the proxy's latency, but the host can run at most to the dispatch end of the current layer ahead.
Gates: `logs/p50/g10_drain.sh` (CDMC 1 / 8, 200 us proxy delay, abort + growth incl. at CDMC=1, swap flips, one
serving arm).
Committed `p8-c34` e2ebb95 (drains mandatory + diagnostics). Next knob, uncommitted until gated: `LOPEP_DERIVE_EARLY=1`
(plan_overlap 2, proxy on): the dispatch op's end-of-step drain + join move into `finish()` (`set_split_finish`), and
`OverlapComm.issue_combine_meta_late` enqueues the combine derive (op-owned meta stream; its only front-end wait is on
the planning event, whose writer is enqueued before the GEMM, so no cycle) before calling it: the derive's host
work overlaps the drain's wait for the plan block instead of following it. A step that misses `finish()` fails at
the next `forward()` (FLUX_CHECK). Chain `logs/p50/s12_go.sh`: after g10, patch + build, gate (CDMC=1 and swap
flips with the knob), then round 12 (`logs/sglang/stage12.sh`: decode 256/512/1024 + prefill, arms b8f | c4 | c4e |
stock | b8fR | c4eR on one allocation, captures NB + NC4e via jobN11 with TAG jN12).
G10 gates (job 59179098, `p8-c34` e2ebb95, binary c59172a99651, drains mandatory, checkpoints on): ALL PASS, bad rows 0
everywhere: `d_c1` (CDMC=1, varying counts), `d_delay` (200 us proxy delay), `d_abort_grow` (3 growths, 3 redos, 880
swap moves), `d_flip` (880 swap moves), `d_c8`, `d_c1_grow` (CDMC=1 + abort + growth: 3 growths, 3 redos, 816 moves);
serving `d30_kc4d` (C4, swap lane on, two waves at 256 per rank) clean: 0 growths, 0 tracebacks, decode step median
129.84 ms (IQR 120.91-141.58; checkpoints on; another allocation than round 11, so indicative only). Every cell
that hung or aborted with `LOPEP_PROXY_DRAIN=0` passes: the G3 diagnosis stands. Round 12 (`s12_go.sh`) next.

### Round 12 decode (job 59179882, one allocation; p8-c34 2fae7aa binary cb4c23b10e35 for c4 / c4e)
Decode step medians, 4n, 256 / 512 / 1024 running per rank (ms): b8f 134.45 / 156.21 / 228.71, b8fR 134.42 / 157.03 /
228.56 (repeat within 0.5 %), c4 129.88 / 179.09 / 285.62, c4e 130.98 / 182.98 / 289.93, stock 84.55 / 143.19 /
273.23. SGLang per-layer MoE (median of the 480-step means): b8f 2.434 / 2.758 / 3.936, c4 2.241 / 3.196 / 5.126. So
C4 with drains wins at 1 MiB (-3.4 % step, -0.19 ms per layer) and LOSES with the batch (+0.44 / +1.19 ms per layer);
derive-early is neutral (inside the noise of c4).
Knob A/B in the harness (`logs/p50/g11_ab.sh`, job 59180491, same binary, layer-step medians over 16 ranks, full tokens,
swap on): 256 per rank: B 2.051, +C3 2.181, +C4 1.808, deferred verdict with the sync kept 1.953, C4 swap off 1.753,
B swap off 1.987; 1024: B 3.705 / 3.683, +C3 3.765, +C4 3.626 / 3.637, deferred-sync 3.707, C4 swap off 3.593. In the
harness C4 does not regress at 1024 (and C3's proxy issue alone costs ~0.13 ms at 256, which C4 recovers): the
regression is serving-specific. Process CPU affinity is 0-127 in both (not a contention artifact).
Root cause (code): on a deferred-verdict step `OverlapComm` bounds the step by `_m_cap = min(recv_cap, plan.vce.numel())`,
but `vce` is the GLOBAL routing [R*S, K]; the serving capacities are sized for the 4096-token prefill chunk, so m_cap
= R*S*K = 32k / 64k / 131k rows at 256 / 512 / 1024 per rank (vs ~2k / 4k / 8k computed rows), and every layer zeroes
`out_buf[:m_cap]` (x 1536 bf16) and runs the eager SwiGLU (silu temp + mul) over m_cap rows: ~1 GB of HBM traffic per
layer at 1024. The harness sizes its capacities for its own smax, hence no regression there. Fix (p8-c5, tree
`lopep_c3`): `C.zero_rows_bounded` / `C.silu_mul_bounded` (planner/lane_device.cu, 16-byte vectors, registered for the
warm-up preload) bounded by the dispatch plan block's row word (`dispatch_op.rows_dev()`, written by the planning
kernels in stream order before the prep and the activation); knob `LOPEP_BOUNDED_ACT` (default 1). Gate + measure:
`logs/p50/g13_bounded.sh` (round 13 decode: b8f | c4b | c4u = same binary unbounded | stock | repeats).
Capture fix: round 11's NB / NC4 captures failed because the second srun step (nodes 1..NN-1, `--exclude` node 0)
gets its own `SLURM_JOB_NODELIST`, so `server.sh` dialed the wrong dist-init head; `jobN11.sh` now exports the job's
node list into both steps.

### Round 13 decode (job 59180880, one allocation): the bounded fix (p8-c5 a2b248e, tree lopep_c3, binary d77ffc8e49d9)
Gates first (reference check, bad rows 0): CDMC=1, swap flips (880 moves), abort + growth (3 growths, 3 redos),
full batch at 1024 per rank. Decode step medians (ms), 256 / 512 / 1024 running per rank (`52_round13_arms.csv`):

| arm | 256 | 512 | 1024 |
|---|---|---|---|
| b8f (stage B) / repeat | 135.34 / 134.57 | 157.40 / 159.20 | 229.44 / 228.61 |
| **c4b (C3 + C4, bounded)** / repeat | **119.09 / 118.40** | **148.25 / 147.55** | **219.83 / 218.38** |
| c4u (same binary, `LOPEP_BOUNDED_ACT=0`) | 133.26 | 182.43 | 285.87 |
| stock | 86.11 | 144.65 | 273.41 |

c4b vs b8f: -12.0 / -5.8 / -4.2 %; vs stock 0.72x / 0.98x / 1.24x (b8f: 0.64x / 0.92x / 1.19x). c4u reproduces round
12's c4 on the same binary, so the regression was exactly the capacity-bounded zero + activation. SGLang per-layer
MoE (CSV layer_ms, includes attention-side bookkeeping of the hook): c4b 2.173 / 2.66 / 3.968 vs stock 1.534 / 2.616 /
5.017. Round-12 prefill (SMAX 256, input tok/s; c4 / c4e unbounded): stock 27132, b8f 22255 / 22619, c4 24823, c4e
25020 / 25017 (+11 % over b8f); round-13 prefill with c4b and the NC4b capture: `logs/p50/g14_prefill_cap.sh`.
Capture recipe fix #2: the two-step split cannot work (each Slingshot job step has its own network VNI; NCCL init
hung in `ncclCommInitRank` on every rank, round 12, job 59179882, NB lost); `jobN11.sh` back to the round-8 single
step (node 0 under nsys, `--capture-range-end=stop-shutdown --kill=sigterm`).
