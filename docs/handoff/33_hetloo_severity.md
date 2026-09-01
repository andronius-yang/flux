# Handoff 33 — Hetero-oracle severity: scenario math check + LOO GPU A/B (2026-08-31/09-01)

**Question (user, 8/31 eve):** handoff 32 showed s1 surviving the equal-blend
hetero oracle (swap premium only repaid at b64). Find an *extreme but
realistic* scenario — real traces only, no synthesized fanout skew — where
always-swap intra-node balancing (`ours_l01_s2_swap_force_p2p_r2`) pays off
severely in total_ms. Constraint: inter-node expert movement is OFF the
table (wire movement is the limiter; no overlap slack under the dispatch
wire).

## 1. Scenario math check (offline, same-code, CPU — 4n)

Driver: `33_hetloo_scenario_mathcheck.py` (this dir; results JSON alongside).
Same-code chain: `pv2_solve` -> `ours_swap.swap_orbit` (tau=1 fixpoint on the
eval batch = tk_dev semantics) -> LocCap `simulate_arm` (eps 0.0625, r2 slots
nlp=G/W+2). Eval = per-topic homog [64,96) batch, T from the b4 budget,
2 seeds, mean over topics. Arms: matched (own-topic [32,64) basis) / static
(scenario oracle basis) / +swap / full re-solve (bound; needs movement).

Mean static->swap imbalance (rmax delta % vs static; resolve % as bound):

| scenario | K2 static->swap (swap%, resolve%) | Qwen static->swap (swap%, resolve%) |
|---|---|---|
| anchor (equal blend, = handoff 32) | 1.183->1.086 (-8.1, -10.0) | 1.255->1.100 (-12.3, -15.3) |
| mix2tv (2 most-TV-distant pools) | 1.103->1.062 (-3.7, -2.8) | 1.201->1.082 (-9.9, -11.5) |
| advperm (hot<->cold permuted twin; synthetic ceiling probe) | 1.177->1.066 (-9.5, -8.0) | 1.319->1.098 (-16.7, -19.5) |
| **loo (oracle mix EXCLUDES eval topic)** | **1.414->1.138 (-19.5, -24.8)** | **1.435->1.138 (-20.7, -26.0)** |
| minority (eval topic 1/16 weight) | 1.279->1.089 (-14.9, -16.8) | 1.339->1.112 (-16.9, -20.6) |
| burst_b4 (block-32 request sampling) | 1.258->1.087 (-13.6) | 1.218->1.117 (-8.3) |
| burst_b1 / iid_b1 control | 1.224 vs 1.249 (~equal) | 1.250 vs 1.248 (~equal) |

Findings:
1. **Severity is monotone in how little the oracle saw of the eval topic**:
   equal blend 1.18-1.26 -> 1/16 share 1.28-1.34 -> never-seen (LOO)
   1.41-1.44. LOO is the realistic extreme; swap recovers 75-90% of the
   full-resolve rmax gap in every scenario, with ~zero wire recovery
   (structural — node assignment untouched).
2. **Pool-selection decorrelation is a dead end** (mix2tv < anchor on both
   models): a 2-pool blend is half-matched to each side.
3. **Intra-topic burstiness is a dead end**: block-by-request sampling ==
   iid at both b4 and b1 on both models (real per-request routing within a
   topic is close to the topic marginal).
4. Per-topic LOO picker (Qwen, b4): college_mathematics steepest (static
   imb 1.770, rmax 7248), philosophy 1.577, ZH college_math 1.725; TV range
   0.233-0.573 (max pair = EN world_history vs ZH college_math — language
   decorrelates more than subject).

## 2. GPU A/B (Qwen, LOO oracle excl. college_mathematics, homog college_mathematics eval)

Specs `sweeps/specs/hetloo_qwen_{4,8}n.yaml` (budgets 1,2,4,8,16,64 per user
ask — b32 deliberately absent). Arms in ONE capsule per topology:
`ours_l01_s1_pv2_r2` (static LOO-basis placement) vs
`ours_l01_s2_swap_force_p2p_r2` (per-iter solve + forced intra-node P2P
swap). Isolated mode, 10 iters, deterministic=0. Per-iter max-rank, median.

**4n** (gate a0c34746 GREEN: b4 nvshmem 10.496/11.746 vs anchor
10.904/12.072, -3.7/-2.7%; arms capsule **20260901-020046_8b32ee18**,
12/12 ok, alloc 57800982 ~0.52 nh):

| total_ms | b1 | b2 | b4 | b8 | b16 | b64 |
|---|---|---|---|---|---|---|
| s1 | 3.18 | 3.81 | 5.19 | 7.48 | 13.29 | 49.00 |
| s2 | 3.92 | 4.49 | 5.58 | 8.11 | **13.17** | **44.12** |
| s2 vs s1 | +23.3% | +17.8% | +7.5% | +8.4% | **-0.9%** | **-10.0%** |
| e2e delta | -1.7% | -1.4% | -1.3% | -2.6% | **-7.7%** | **-12.4%** |

**8n** (gate b24b7364 GREEN: 16.365/17.689 vs anchor 16.140/17.509,
+1.4/+1.0%; arms capsule **20260901-021323_59f6ad19**, 12/12 ok, alloc
57801006 7:54 = 1.05 nh):

| total_ms | b1 | b2 | b4 | b8 | b16 | b64 |
|---|---|---|---|---|---|---|
| s1 | 5.35 | 6.15 | 7.63 | 10.87 | 18.11 | 62.66 |
| s2 | 6.41 | 7.35 | 8.66 | 12.29 | 20.10 | 64.15 |
| s2 vs s1 | +19.8% | +19.5% | +13.5% | +13.1% | +11.0% | +2.4% |
| e2e delta | +3.4% | -3.1% | 0.0% | -0.3% | +2.5% | +1.1% |

## 3. Verdict

1. **The severe case exists and is 4n LOO**: s2 always-swap crosses at b16
   and wins b64 by **-4.9 ms total (-10.0%) / -5.75 ms e2e (-12.4%)** —
   vs -0.27 ms (-0.6%) at b64 under the equal blend (handoff 32). In the
   e2e bracket s2 wins at EVERY budget; the small-b total_ms losses are the
   flat ~0.7-0.9 ms host plan/apply premium. The CPU-predicted chain (LOO
   1.77 static imb -> swap recovery -> rows-bound ms) closed quantitatively.
2. **8n is the honest boundary: intra-node swap leverage collapses.** s2
   pays at every budget (total +2.4% even at b64), e2e a wash. Consistent
   with structure: at 8n the LOO residual increasingly sits BETWEEN nodes
   (swap cannot touch node assignment; the 4n math check already showed
   ~zero wire recovery), Qwen r2 headroom becomes 50% (2 spare slots on 4
   experts/GPU) so replication pre-absorbs hot experts, and the wire share
   of total grows, diluting the rows-bound component. The 8n/16n CPU
   extension (running at session close) will decompose shrink-of-residual
   vs failure-to-translate.
3. Paper framing: the always-swap contribution = **latency under topic
   drift at moderate scale (<=4n island) and the tail** (worst-topic
   compression, handoff 32 §4.3); at >=8n the story hands over to
   node-level placement (which is the movement/wire regime, currently
   off the table).

## 4. Incidents

- **Binary drift (rule 4).** The 8/31 campaign binary ths_op
  `ce939eb91073d55b` no longer exists: another session rebuilt the main
  tree's `python/flux/lib` (worktree symlink target) 8/31 ~17:55 PDT.
  All four hetloo capsules ran ths_op `505e4bedb6d1502f` / libflux_cuda
  `3d251dba29378431`. Within-capsule A/Bs valid; tonight's 4n-vs-8n
  same-build; NEVER-MIX against the 8/31 hetero campaign capsules. Gates
  green within +-5% of the ce939eb9-era anchors bound the drift (~1-4%).
- Both lane subagents stalled waiting on watchers that die on agent stop;
  orchestrator drove the runs directly (dup gate run 021256_1be08650 killed
  before its srun; no capsule dir left). One lane/orchestrator srun overlap
  risk resolved by pid-kill (never pkill -f).
- Self-oracle scope re-verified in the 98eb2c8 diff: only the epic-driver
  (llc) token list was broken 8/27-8/31; the ours driver always received
  --oracle_routing_file. llc lost everywhere even WITH self-oracle basis —
  a fortiori.

## 5. Ledger

Capsules (branch worktree-het-oracle, commit c0810d1): 4n gate
20260901-015953_a0c34746, 4n arms 20260901-020046_8b32ee18, 8n gate
20260901-021258_b24b7364, 8n arms 20260901-021323_59f6ad19. Specs commit
98d3b31. Math check: 33_hetloo_scenario_mathcheck.{py,json} (4n; 8n/16n
extension pending at write time). GPU cost ~1.6 nh m5350_g total.
