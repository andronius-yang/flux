# Handoff 27 — Arms, canon state, and the trunk (2026-08-29 evening)

User-directed consolidation after the 8/29 comm canonicalization
(handoff 26) and the parallel swap-overlap session (handoff 25).

## 1. Trunk

`main` = `integrate` = `ours` = `pv2` = **f8163d8** (fast-forwarded 8/29
evening; every branch was a strict ancestor of pv2's head — 0 commits
lost; pre-ff heads tagged `pre-ff-8-29-main` / `pre-ff-8-29-integrate`).
f8163d8 carries BOTH sessions' work: the ours lineage (WPM pull mode,
moved-last `fa841e8`, keep-bonus), PV2 placement + canonicalization, the
swap lane with P2P transport + issue-point knob + both 16n-b64 fixes,
and the 8/29 comm canon (wave-adapt 48, combine-idx kernel, plan
graphs, late combine-meta overlap byte-gated <= 16 MiB).

**Binary policy (user ruling):** NOT frozen, `build/` NOT cleared — the
next session implements another ablation and will rebuild. The
installed lib (`libflux_cuda_ths_op.so` sha 3a5dc836eba2) matches
f8163d8's C++ bit-for-bit in source terms (zero src/include files newer
than it). Rule 4 still binds: one binary per headline table; record the
lib sha in every capsule manifest. Shared-checkout rule learned 8/29:
NO rebuilds by either session while any capsule is running (a rebuild
swapped the .so under the other session's campaign mid-flight).

## 2. The arms (current definitions + which optimizations each carries)

| arm (variant key) | placement | routing | comm transport | per-iter placement work | 8/29 canon it carries |
|---|---|---|---|---|---|
| Slipstream-only `l01_slipstream` | default, no replicas | plain top-k | Slipstream fused (LB_UNION dispatch; msplit/fused-pack/bucket combine) | none | wave-adapt 48, combine-idx kernel (binary defaults) |
| LLC / "PLL" `llc_l01_s1_pv2` | pv2, once at setup (oracle basis), r2 | LocCap sender-local (relaxed kernel) | **staged** a2av (epic-style runner, combine event-gated after full GEMM, ns1) | none | combine-idx kernel only (staged arms never take the wave path) |
| OURS s1 `ours_l01_s1_pv2_r2` | pv2 once at setup (oracle), r2 | LocCap | Slipstream fused | none (routing IS planned + timed every iteration = plan_ms) | all: wave-adapt, idx kernel, plan graphs, late overlap (`--plan_overlap 2`, OV2_MAX 16 MiB) |
| OURS s2-pv2 `ours_l01_s2_pv2_r2` | pv2 RE-SOLVED every iteration (quiet always-solve, gain threshold 0) | LocCap | Slipstream fused | solve + trigger + adoption + cross-node WPM weight migration (overlapped); moved-last optional (`FLUX_OURS_SCHED_MOVED_LAST`) | all of the above (s2 canon gates green 4n + 8n) |
| OURS s2-swap `ours_l01_s2_swap_r2` (handoff 25 record: force p2p EARLY) | pv2 at setup + intra-node expert SWAPS every iteration (greedy pair+swap, host integer) | LocCap | Slipstream fused | swap decision + NVLink P2P weight exchange (early issue); no cross-node moves; moved-last OFF is best | all of the above (inherited via `_SWAP_ARGS`; designed gate = §4) |

Decomposition reading for the paper: Slipstream-only isolates comm; LLC
isolates placement+routing (on the staged transport); OURS s1 = their
composition; s2-pv2 / s2-swap = two ways of adding live placement.

## 3. THE METHODOLOGY s2 ARM (planned, not yet implemented — user ruling)

The s2 presented in the METHODOLOGY section will be a NEW combined arm:
**quiet always-solve (placement check every iteration, as s2-pv2) AND
overlapped intra-node swaps every iteration (as s2-swap)** — one lane
that re-solves and also swaps, both overlapped. The two existing s2
arms (s2-pv2 = migration-only, s2-swap = swap-only) then move to the
ABLATION section. Open design points for whoever builds it: ordering of
the swap decision vs the re-solve within the place bracket (the swap
tables must feed the router the same iteration — handoff 25 §1), the
sizing envelope (the swap lane excludes the placement-independent
provable recv ceiling — 26daa3a — while the pv2 adoption lane still
carries it; at 16n b64 that ceiling is 4.9x the fold), and the
mode-2/early-issue adjacency in the driver (both issue right after the
l0 enqueue).

## 4. Designed gate pass on f8163d8 (this handoff's action)

**4n (job 57717798, 30-min interactive) — ALL GREEN, 9/9:**
- swap P2P early-issue arm under full canon
  (`ours_l01_s2_gate_swap_force_p2p_r2`, correctness on, per-iteration
  output checks): K2 b1/b16/b32 ok (114/169/159 s), Qwen b1/b16/b32 ok
  (46/48/51 s) — capsules 20260829-232125 (K2), 20260829-232849 (Qwen).
  b1+b16 exercise the mode-2-active range next to the swap lane's
  early issue; b32 exercises the OV2 byte-gate fallback (mode 0).
- moved-last gate (`mlgate_k2_4n_r2`: s2_gate_r2 / _ml / _mlw2 at K2
  b8): 3/3 ok (130/130/124 s) — capsule 20260829-233118. The WPM
  migration lane with moved-last and late-w2 is canon-consistent.
The previously incidental swap coverage is now designed-in at 4n.

**8n (job 57720475, debug) — ALL GREEN, 4/4:** swap P2P early-issue
arm under full canon, correctness on: K2 b1/b8 ok (120/154 s), Qwen
b1/b8 ok (47/51 s) — capsules 20260830-004148 (K2), 20260830-004623
(Qwen). With the regen lane's 8n s1/slipstream/s2-pv2 gates
(20260829-1428xx..143351) every arm is now gated at both scales on the
trunk binary.

Ops notes from getting the 8n window (GPU partition was FULL — 1635
allocated / 4 idle): (1) 8-node sallocs returned "Unable to allocate
resources: Connection timed out" client-side while the controller
sometimes GRANTED anyway — one such orphan grant (57720279) showed
RUNNING in squeue for ~7 min and then reported "job has expired" the
moment an srun used it (capsules 20260830-003926/003930 = those
0-second launch failures, not gate verdicts, left uncommitted);
(2) the fix that worked: a detached grant->gate->release chain script
(scratchpad gate8n_chain.sh pattern: salloc -I300 in a retry loop,
parse the jobid from ITS OWN output, run, scancel) — the grant is spent
the second it lands, so nothing idles or expires.

## 5. Open items carried forward

- mode-2 x >16 MiB RCA (handoff 26; byte-gated, not root-caused).
- 16n s2 sizing: placement-independent provable recv ceiling in the
  pv2 adoption lane (other session's cross-check #1).
- In-capsule COMET (and EPIC) head-to-head on ONE binary = the paper
  cell (all COMET comparisons to date are cross-build).
- Graded wave count for the 10-48 ratio band (cheap; K2 8n b8 sits on
  the boundary); v2 expert-major progressive flush behind its
  piece-ladder go/no-go (handoff 26 §3).
- Home quota ~96%: free space (NOT build/ — user ruling) before big
  capsule commits.
