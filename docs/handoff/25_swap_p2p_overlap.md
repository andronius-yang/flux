# Handoff 25 — intra-node expert swap: P2P transport, issue-point overlap, topic-shift verdict (2026-08-28)

Branch `pv2-swap2` (worktree `$PSCRATCH/workspace/andrewy/flux-pv2-swap2`,
forked from `pv2` @ 691f86b; binaries symlinked to the main checkout's —
same `.so` as every capsule in this handoff). Continues handoff 23 (PV2
placement) and the 2026-08-27 swap lane (`ours_swap.py`, commit 5733029).
Everything below is committed on `pv2-swap2` (last commit f7036ce +
this document). Session memory: `swap-p2p-overlap-campaign`.

## 0. TL;DR

1. **The swap's data movement is fully hidden.** The 8.27 arm's +2.5 ms
   (force vs decide-only) was *host* time — torch NCCL P2P enqueue +
   a per-pair table loop — not transfer. Replacing the transport with a
   symmetric-heap P2P copy (one `cudaMemcpy` over NVLink into the peer's
   staging + zero-SM stream-memops signalling) and vectorising the host
   path brings an every-iteration swap to **+0.36–0.50 ms (K2) /
   +0.21–0.40 ms (Qwen)** over the decide-only twin at 4n, **+0.4–0.8 ms
   at 8n**, with the fused l0 GEMM untouched. The NVLink copies (29.4 MB
   per matrix, 0.32–0.39 ms each) sit in the host planning gap under
   early issue, or 100 % under the GEMM / 84 % under the dispatch wire
   under late issue (§3, §4).
2. **Early issue is the right default.** Late issue (exchange enqueued
   after the l0 launch) makes the moved slot's tiles spin: GEMM 2.61 →
   2.87 ms and the post-GEMM barrier 0.45 → 0.80 ms at K2 b8 (+0.3–0.7 ms
   at b1/b8; within noise at b64). The kernel-side moved-last schedule on
   `ours` (fa841e8) is therefore *not needed* for the swap lane.
3. **Rebalance benefit is regime-bound.** Under the s1-canon
   oracle→batch drift there is nothing to recover (per-rank l0 spread ≤
   1.05 even for the static arm). Under a real topic-shift oracle
   (`opool=`, §5) LocCap routing absorbs the per-GPU skew at b1/b8; at
   b64 the stale placement costs uniformly more work per GPU (K2 4n l0
   mean 24.7 vs 19.7 ms re-solved). There the intra-node swap recovers
   **about half of the cross-node re-solve ceiling at 4n** (K2 total
   −7 % vs −12 %, l0 −12 % vs −20 %; Qwen −3.6 % vs −7.2 %) for 0.35 ms
   of place instead of 1.2–1.6 ms — and **~10 % (K2) / ~0 (Qwen) at 8n**
   (−0.9 % vs −12.6 %; −0.2 % vs −19.6 %). A topic shift changes *which*
   experts are hot, i.e. replica counts; a transposition cannot change
   replica counts. That is the structural ceiling.
4. `tau=1` ("swap on any predicted gain") is worse than the canonical
   `tau=512` (Qwen b1: 4.61 vs 4.23 ms): the decide model is
   equal-share-per-replica, the real router is capacity-aware.
5. All timed arms are **steady state**: swaps converge in warmup
   (7 → 6 → 5 → 0 at K2 4n), the cross-node re-solve fires once at
   iteration 0; timed totals include the per-iteration decide/solve cost
   and exclude the migration transient (§6).

## 1. Root cause of the 8.27 +2 ms (capsules 20260828-155459 / -161050 / -162008)

Force mode (tau = −1, one exchange per node-pair every iteration) vs the
decide-only twin (`swap0`, tau = ∞), 4n, medians of per-iteration
max-over-ranks:

| K2 4n (8.27 NCCL transport) | b1 | b8 | b64 |
|---|---|---|---|
| swap0 | 6.12 | 9.79 | 47.16 |
| force | 8.60 | 12.33 | 49.89 |
| force `place_ms` bracket | 2.78 | 2.88 | 2.88 |

The whole delta is the place bracket. nsys: the 24 `ncclDevKernel_SendRecv`
(~0.9 ms each) run in GPU-idle gaps with 0 % concurrency with GEMM/wire
— the movement finished inside the host planning window every time.
Driver split (gate log, K2 b8): `d2h 0.4 / decide 0.6 / apply+issue
2.3 ms` per fired swap. The 2.3 ms was: `apply_swaps` per-pair
torch-scalar loop (1.15 ms / 16 pairs on the login CPU),
`refresh_placement` = three blocking pageable H2D uploads (each a hidden
`cudaStreamSynchronize`), and `dist.batch_isend_irecv` (coalesced NCCL
group, four Work objects, four `wait()`s, two `torch.tensor(...,
device="cuda")` H2D copies, ~6 kernel launches).

## 2. Mechanism (python-only; `python/flux/testing/ours_swap.py`, driver knobs)

**Transport `--swap_xport p2p`.** Staging buffers `[ffn,H]`/`[H,ffn]`
and a 2-slot landed-signal live on the symmetric heap via
`flux.create_tensor_list`, which returns *node-local peer views*
(`nvshmem_ptr`). Per matrix, each member of a pair on the movement
stream: `peer_staging.copy_(my_slot)` (one contiguous cudaMemcpy over
NVLink — the "Memcpy PtoP" on the timeline) → store the epoch into the
peer's landed-signal (`cuStreamWriteValue64`, fallback `fill_`) →
`cuStreamWaitValue64(GEQ|FLUSH)` on **my** landed-signal (zero-SM; the
same primitive WPM's C++ `join` uses on this heap, via ctypes) →
`slot.copy_(staging)` → raise my slot's gate signal. Nobody writes a
peer's slot and nobody reads a peer's staging, so the only cross-rank
order is the landed-signal; the WPM remote-signal race class (handoff
20) cannot occur. Staging reuse across iterations is ordered by the
iteration's `l1_wait` → next all-gather. Intra-node only — the CXI
proxy ordering rule (SCHEMA rule 6) is not implicated.

**Issue point `--swap_issue early|late|split`.** `prepare()` runs in the
place bracket (epoch bump, raise the *unmoved* slots' signals on the
current stream, record the pre-l0 event); `issue_early()` right after;
`issue_late()` immediately after `l0_forward` is enqueued. The movement
stream only ever waits on the pre-l0 event — a `wait_stream` after the
l0 enqueue would deadlock against the gated GEMM. `split` = w1 early,
w2 late.

**Host trims (phase 1b).** `swap_plan` vectorised in numpy (all pairs of
all nodes at once; bitwise-identical to the loop incl. the top-8
prefilter set and the (gain, (e_h, e_l)) tie-break, 400-trial test),
`apply_swaps` numpy (slot-relabel map + row re-sort, 300-trial test),
`SwapTableSync` = pinned host mirrors + two non-blocking copies into the
planner's device tables (zero syncs). A first device-index variant (12
CUDA launches) was *slower* than the three blocking uploads (0.47 →
0.67 ms) and was dropped. Per fired swap now: decide 0.38–0.40, apply +
issue 0.42–0.47 (early) / 0.22–0.26 (late) ms.

## 3. Latency verdict (phase 1b v2)

4n, capsules 20260829-003817 (K2) / -005016 (Qwen), 18/18 each; `tot /
place / l0` medians:

| K2 4n | b1 | b8 | b64 |
|---|---|---|---|
| s1 reference | 5.89 / 0.01 / 1.66 | 9.51 / 0.01 / 3.66 | 46.28 / 0.00 / 19.3 |
| swap0 | 6.24 / 0.36 / 1.68 | 10.06 / 0.35 / 3.80 | 46.91 / 0.35 / 19.6 |
| force nccl (8.27 transport) | 7.28 / 1.10 / 1.73 | 10.94 / 1.01 / 3.70 | 47.27 / 1.03 / 19.7 |
| force **p2p early** | **6.60** / 0.69 / 1.67 | **10.46** / 0.70 / 3.66 | 47.41 / 0.72 / 19.8 |
| force p2p late | 7.28 / 0.68 / **2.39** | 10.75 / 0.55 / **4.23** | 46.95 / 0.55 / 19.4 |
| force p2p split | 6.72 / 0.65 / 1.88 | 10.65 / 0.65 / 3.72 | 47.43 / 0.66 / 19.7 |

| Qwen 4n | b1 | b8 | b64 |
|---|---|---|---|
| swap0 | 4.12 / 0.34 / 1.19 | 7.82 / 0.34 / 2.74 | 41.18 / 0.34 / 16.4 |
| force p2p early | 4.52 / 0.68 / 1.22 | 8.17 / 0.67 / 2.72 | 41.39 / 0.68 / 16.5 |
| force p2p late | 4.96 / 0.67 / 1.69 | 8.08 / 0.52 / 2.69 | 41.06 / 0.53 / 16.5 |

8n (capsules 20260829-010720 K2 / -011845 Qwen, 18/18): force p2p early
+0.4–0.8 ms over swap0, place 0.72–0.93 vs 0.35–0.40, l0 flat. The decide
is node-count-flat (vectorised), the exchange is per-node-local.

Note `swap0`'s own bracket rose 0.21 → 0.35 ms with the vectorised
decide (the numpy path does its full work even when the tau gate is
vacuous) — a cheap early-exit is a follow-up.

## 4. Timelines (measured, nsys, K2 b8 force, 4n, GPU 0 of node 0)

One timed iteration; `t = 0` = end of the previous iteration's l1 GEMM
(so the first ~1.1 ms is the previous combine tail); 1 column = 0.1 ms.

```
EARLY issue — capsule 20260829-010311, iteration = 10.97 ms
          0         1         2         3         4         5         6         7         8         9         10        11
HOST     |···············DDDDDDIIPPPPPPPPPPPPPPPPPPPPPP···································································|
main s7  |···········bbb·········pp····mm·······s····ii··GGGGGGGGGGGGGGGGGGGGGGGGGGbbbbbgwwLLLLLLLLLLLLLLLLLLLLLLLLLLLLL··|
nccl s18 |··············A··········AAAA···················································································|
MOVE s26 |·····················>>>c>>>c···················································································|
wire     |·~~~~~·······································~~~~~~~~~~~~~~~~~····································~~~~~~~~~~~···|
combine  |····rrrrrrr······································································CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC·|

LATE issue — capsule 20260829-000942, iteration = 11.77 ms
          0         1         2         3         4         5         6         7         8         9         10        11
HOST     |··············DDDDDDPPPPPPPPPPPPPPPPPPPPPPPPPP··II······························································|
main s7  |···········bb············pp···mm·······s····ii··GGGGGGGGGGGGGGGGGGGGGGGGGGGGGbbbbbbbbg·wwLLLLLLLLLLLLLLLLLLLLLLL|
nccl s18 |·············A·············A····················································································|
MOVE s26 |··················································>>>>c>>>·c····················································|
wire     |··~~~~········································~~~~~~~~~~~~~~············································~~~~~~~~|
combine  |····rrrrrrr··············································································CCCCCCCCCCCCCCCCCCCCCCC|
```

`A` all-gather of the load counts (iteration start, ~1.4 ms); `D` host
decide + table update (0.6 ms); `I` host enqueue of the exchange; `P`
host plan derive; `p m s i` plan/meta/scan/indexSelect kernels; `>` one
29.4 MB NVLink P2P copy into the peer's staging (0.32–0.39 ms; w1 then
w2); `c` local staging→slot copy after the peer's landed-signal; `~`
NVSHMEM inter-node puts (dispatch wire, then combine wire); `G` fused l0
GEMM (2.61 / 2.87 ms); `b` barriers; `g` gelu; `w` the l1 wait on the
movement event; `L`/`C` l1 GEMM splits and combine kernels.

Reading: under **early** the exchange (2.08–2.85 ms) completes 1.9 ms
before the GEMM launches (4.73); the gate finds its signal raised and
the `w` wait is a no-op. Under **late** the same two copies (4.99–6.02)
sit under the GEMM and the dispatch puts, but the moved slot's tiles
spin until 6.02: GEMM 2.61 → 2.87 ms, post-GEMM barrier 0.45 → 0.80 ms
(slowest rank). Same 0.7 ms of NVLink time; only its placement relative
to the gate differs. The swap decision depends only on the all-gathered
counts and the current slot table (pure function, no exchange) — not on
the PV2 global solve, which is why it can be issued this early; the
routing plan `P` is then derived on the swapped tables (same-iteration
benefit).

## 5. Topic-shift oracle (`opool=`) — does rebalancing pay?

**Why not the stale probe.** `s2_stale rot` rolls every expert's
instances by one rank: wrong *location* (cross-node lane always fires)
but each GPU inherits its neighbour's already-balanced load — no
per-GPU skew for a swap to act on. The regime with real skew is an
oracle solved on another topic: new family param
`opool=<bench>/<subject>` (`sweeps/gen_trace_routing.py`
`ensure_oracle_sidecars`; same layer + same decode window from the
other pool; folds into the matrix identity; homog sidecar bodies stay
byte-identical). K2: `mmlu/professional_law`; Qwen: `mmlu/philosophy`
(Qwen traces lack professional_law). No per-iteration reset: the static
s1 arm runs the whole window on the skewed placement, swap arms converge
in warmup, the quiet cross-node `s2_pv2` arm is the ceiling.

**Offline predictor** (equal-share-per-replica load model, b8; node
floor = max node total / mean, what intra-node swaps cannot go below;
ceiling = PV2 re-solved on the batch):

| model / basis | nodes | GPU imb | node floor | after swap | ceiling |
|---|---|---|---|---|---|
| K2 homog | 4n | 1.13 | 1.06 | 1.07 | 1.01 |
| K2 professional_law | 4n | 1.58 | 1.16 | 1.18 | 1.07 |
| K2 professional_law | 8n | 1.99 | 1.19 | 1.19 | 1.16 |
| K2 professional_law | 16n | 2.53 | 1.56 | 1.78 | 1.31 |
| Qwen philosophy | 4n | 1.70 | 1.17 | 1.17 | 1.15 |
| Qwen philosophy | 8n | 2.33 | 1.44 | 1.45 | 1.14 |
| Qwen philosophy | 16n | 2.89 | 2.01 | 2.49 | 1.21 |

The predictor overstates absolute GPU skew (hardware static arm: 1.02
vs predicted 1.58 — LocCap routing is capacity-aware, not equal-share);
its *structure* (floor rising with node count) is what the hardware
confirmed.

**Hardware** (gates 20260829-010710/-010939 4n, -012911/-014111 8n, all
green; A/B 4n -011029 K2 / -012034 Qwen, 8n -013139 K2 / -014206 Qwen,
15/15 each). Same-machinery rows (`--sizing capacity`); `s1` uses demand
sizing and is not a same-machinery comparator.

| K2 4n b64 | l0 mean | e2e | total | place |
|---|---|---|---|---|
| swap0 (skewed, no moves) | 24.67 | 53.1 | 55.9 | 0.37 |
| swap p2p tau=512 | 21.73 (−12 %) | 49.4 (−7 %) | 52.1 (−7 %) | 0.36 |
| swap p2p tau=1 | 21.90 | 49.3 | 52.3 | 0.36 |
| s2 pv2 re-solve (ceiling) | 19.68 (−20 %) | 44.5 (−16 %) | 49.0 (−12 %) | 1.60 |

| Qwen 4n b64 | l0 mean | e2e | total |
|---|---|---|---|
| swap0 | 18.8 | 43.1 | 46.7 |
| swap p2p tau=512 | 19.1 | 41.6 (−3.5 %) | 45.0 (−3.6 %) |
| s2 pv2 re-solve | 17.4 | 39.3 (−8.8 %) | 43.3 (−7.2 %) |

| 8n b64 | swap0 total | swap tau=512 | re-solve ceiling |
|---|---|---|---|
| K2 | 74.7 | 74.0 (−0.9 %) | 65.3 (−12.6 %) |
| Qwen | 66.9 | 66.7 (−0.2 %) | 53.8 (−19.6 %) |

At b1/b8 nothing moves at either scale: the per-rank l0 spread is
1.01–1.03 for *every* arm including the skewed static one — the router's
per-slot capacity (`f_cap`, proportional to budget) redirects overflow to
the other replica / fallback before any GEMM sees it. At b64 the spread
is still ~1.00 but the *mean* per-GPU work differs (K2 24.7 vs 19.7): the
stale placement costs uniformly more work, and only where capacity
binds. On Qwen the swap's gain is on the wire side (e2e), not the GEMM.

## 6. Caveats

- **Steady state only.** 5 warmup + 10 timed iterations on a fixed
  batch; movement happens in warmup (swap 7 → 6 → 5 → 0; re-solve fires
  at i0). Timed `place` = per-iteration decide (0.35) or PV2 solve + drift
  (1.2–1.8); the migration transient is excluded. Prior stale-regime
  numbers for a full WPM cross-node migration: 85–136 ms place (sharded),
  ~33 (no-shard), 4–5 (pull/getmem); the swap's is 3 rounds × ~1.3 ms with
  the copies hidden. Amortised: at 8n b64 the re-solve buys ~9 ms/iter,
  so even a 100 ms migration repays in ~11 iterations after a shift.
- **The 8n P2P cells are noisier** (b1 totals span 8.0–9.3 ms across
  arms); the early/late/split ordering is inside noise there.
- **force mode** is the movement probe, not a rebalancer: it oscillates
  at the fixed point by design.
- `opool` eval matrices are a *different sample* of the same eval window
  (the rng seed includes the family params) — compare within capsule.

## 7. WPM in one paragraph (for the next reader)

`flux.WeightPushMulticast` = the s2 lane's cross-node weight mover: one
symmetric tensor per matrix `[home rows | prefetch slots]` (the GEMMs read
the slots), NVSHMEM `putmem_signal` pushes home→slot with a multicast
gateway per node and optional NIC-sharding, and per-slot epoch signals
the fused GEMM's weight-gated tiles spin on. It works and has wins on
record (NIC-shard −34…−47 % at b1 in MoonEP; the −12…−20 % ceiling rows
above are WPM moves), but its per-move host issue cost is ms-to-tens-of-
ms, it has an open mid-iteration remote-signal race class (handoff 20;
sharded/late-w2 wedge family quarantined), and it is unaudited under the
CXI wire-ordering rule. The swap lane reuses WPM's *storage and gate
signals* and never its wire; the two are complementary (swap = cheap
first-order fix within a node; WPM = replica-count changes across
nodes).

## 8. Ledger

Arms (`sweeps/variants.py`): `ours_l01_s2_swap_force_{p2p,p2pl,p2ps}_r2`
(+ `gate_` twins), `ours_l01_s2_swap_{p2p,p2pl,p2ps}_r2`,
`ours_l01_s2_swap_p2p_t1_r2` (+ gate); 8.27 arms unchanged (explicit
`--swap_xport nccl` default). Driver knobs: `--swap_xport`,
`--swap_issue`, `--swap_tables upload|device`. Specs:
`pv2_swapp2p_{gate,ab}_{4n,8n}_{k2,qwen}`, `pv2_swapshift_{gate,ab}_{4n,8n}_{k2,qwen}`.

Capsules (all on the pre-merge binary): 20260828-233716/-234452 (p2p
gates), -234725/-235935 (phase-1 A/B), 20260829-000942 (nsys late),
-002839/-003546 (1b gates), -003817/-005016 (1b A/B), -010311 (nsys
early), -010720/-011845 (8n p2p A/B), -010710/-010939/-012911/-014111
(topic-shift gates), -011029/-012034/-013139/-014206 (topic-shift A/B).
Deleted (not data): 001428 (AttributeError, 0/3), 002003 (device-index
v1 gate — superseded code state).

## 9. Branch topology and merge plan

`ours` (+66, C++: WPM pull mode, `sched_expert_order`) and `pv2` (+19)
are siblings off `main` c27b576; `pv2-swap2` is +6 on `pv2`. A dry-run
merge of `ours` into `pv2` touches only `sweeps/variants.py` and the
driver (four additive hunks). `ours`' handoff file `23_movement_exposed_
latency.md` titles itself "Handoff 22" and collides with pv2's numbering
→ filed as `22b_` on merge. Order: `ours → main` (ff) → `pv2-swap2`
(contains pv2) merged on top → one rebuild (new binary generation) →
re-gate both lineages (swap gates; pull / moved-last gates) before any
cross-boundary quote. The `fast-split` worktree (FAST lane) is left
untouched.

## 10. Open items / next

- Early-exit in the vectorised `swap_plan` when no pair exceeds tau
  (restores swap0's 0.2 ms bracket).
- "Adapt every iteration" capsule with BOTH mechanisms under `opool`
  (swap force already exists; WPM `s2_stale` arms live on `ours` → needs
  the merged binary) for the per-iteration cost incl. movement.
- 16n topic-shift (predictor: node floor 1.56 K2 / 2.01 Qwen → expect
  the swap to recover little; the re-solve ceiling is the number that
  matters there).
- Merge of the moved-last mechanism is available after the rebuild but
  not required by the swap lane.
