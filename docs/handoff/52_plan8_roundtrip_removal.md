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
