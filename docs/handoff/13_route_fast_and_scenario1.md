# 13 — LocCap routing fast path + scenario-1 canonical oracle (8.22.route)

Status 2026-08-22: landed on `place-fast` @ 6f38b80 (after 12's placement
work; one binary = main's .so + the standalone kernel extension). E2E
proof: jobs 57438952 (this doc) and 57423410 (doc 12). Related memory:
`placefast-campaign` (updated), report artifacts linked there.

## Router plan tail — what changed and the contract

`EpicIterPlanner._derive_from_phys_fast` (default ON for loccap routers,
`FLUX_PLL_FAST_TAIL=0` = legacy): sort-free, zero-D2H tail. Mechanics:
own-row sort (send wire order) + ONE [E] radix argsort keyed
(not-mine, src, phys, tok) whose first recv_cap slots are my arrivals in
canonical order (fixed-shape capacity padding — kills the legacy ragged
boolean-gather mid-phase sync) + O(E) index_adds (NEVER torch.bincount —
it hides device syncs and is capture-illegal) + vce built DIRECTLY from
phys_all in topk column order. The column-order freedom is PROVEN
contract-safe: each token has exactly K entries and the wire/scatter
order within a virtual slot depends only on (vslot, global token).
Guards updated accordingly: `python_meta_from_vce` (driver v2b guard
compares op vs python ON THE SAME vce), `check_against` vce compare =
per-token multiset under fast tail. `FLUX_PLL_TAIL_GRAPH=1` CUDA-graphs
the device half (pinned-staging blob D2H outside the graph).

Numbers (one binary, 4n, isolated): plan_ms qwen b8 2.87→1.71→**1.19**
(legacy→fast→graph), K3 b7 3.45→**1.40**; cell totals −22%/−18%;
emulated derive 6.1/6.5/21.8 → 1.1/1.4/5.5 ms (4n qwen/4n K3/32n, login).
K3 l0 correctness gate PASSED with `FLUX_PLL_RANDOM_PAYLOAD=1` on
fast+graph. Composition anchor, same ladder: **LLC l0 (pllf placement +
loccap_sl + fast tail) 6.03 ms vs EPIC d6 baseline 12.57 (−52%)**; comm
3.95 vs 4.89; plan 1.19 vs 6.69.

Boundary to dispatch (comm-comp overlap port readiness): device-resident
vce int32 + in/out splits + place_slots; ONE pinned blob D2H per
iteration whose only load-bearing host consumers are (a) the grouped-GEMM
segment sizes and (b) n_recv/pad asserts. Folding (a) needs device-side
group sizes in GemmGroupedV2 (or in-op derive) — the remaining boundary
item; everything else is already ingestion-clean.

## Kernel (tier-3 cover) — patched, dual-shipped

`pll_route3_kernel_t<KT>` (K∈{8,16}): coverage masks in registers +
KT-bit remaining set replaces gs[32] local-memory spills and the
O(NN·nrem) rescan. **R=128: 3.37→1.65 ms (2.0×)**; 4n parity. Shipped in
`src/cuda/moe_utils.cu` (source of truth — picked up at the next flux
rebuild) AND as the standalone JIT extension
(`flux/testing/_pll_sl_ext.cu` + `pll_sl_ext.py`, build dir
`$PSCRATCH/workspace/andrewy/pll_sl_ext_build`, dispatch
`FLUX_PLL_SL_EXT=1`) — e2e-proven multi-rank, same out_sha as the
in-binary kernel. Extract tooling note: the .cu is generated from
moe_utils.cu's router section; edit moe_utils.cu and re-extract.

## Remote-only-cap flavor (`FLUX_LOCCAP_REMOTE_CAP_ONLY=1`)

The special insight the routing exploration produced: tiers 1+2 are
intra-node by construction (zero wire bytes), so capping them buys no
wire and only rejects free locality — the eps budget should bind ONLY
the tier-3/forced cross-node residue. Implemented table-exactly in torch
(`remote_cap_only=` on loccap_route_sl / loccap_route_gpu) and kernel
(host-wrapper env + shares-kernel flag; per-rank-0 remote rows match
torch EXACTLY). Results: offline incidence −7.0% (qwen b8) / −6.1% (8n)
— reaching pure-locality (eps=inf) incidence at imbalance 1.30 instead
of 1.58, because remote hotspots stay capped; ~0 at saturated K3. E2E
(torch arm, l01, same placement): e2e −5.8%, l0 −7.0%; kernel-arm flavor
runs at kernel speed. NOT canonicalized — a capsule-grade eps×flavor
ladder is the next quotable step. Bounds/capacity mode absorb the flavor
(loccap_sl_bounds derives from realized tables).

## Scenario-1 canonical oracle (ratified design)

Recon (fork agent, full schema/coverage/validity tables in its report):
traces are per-request JSON keyed by layer; ti=0 prefill, ti>=1 decode
slots; qwen3 all 94 layers MoE. Construction "same decode slot across
requests" is STATISTICALLY UNUSABLE at real request counts (its own
sampling noise 128k ppm > 65-68k signal) and is the wrong production
analogy anyway. **Canonical: sem=homog, topic mmlu/philosophy, qwen3
layer 5; evaluated = decode slots 49..64 (w=16, all 311 requests);
oracle = slots 33..48 — literally the previous batch under continuous
batching.** S/N 3.2×, wrong-topic contrast 5.3×. K3-synth has NO layer 5
and no real temporal structure (i.i.d. from marginals) — formal analog:
layer 23, evaluated 97..160 / oracle 33..96, drift there is pure
sampling noise (record in spec notes; real Kimi-K2 traces exist on disk
if a real-temporal K3-class rung is ever needed).

Wiring landed: driver `--oracle_routing_file` (placement solves on the
oracle batch; dynamic lane still observes the evaluated batch) +
`epic_pll_oracle_{file,basis,drift_ppm}` facts. Prototype inputs at
`$PSCRATCH/workspace/andrewy/a2av_test_matrices/s1/` (eval+oracle
routing + eval matrix, qwen 4n b8). Measured: drift 80,543 ppm realized;
oracle-placement incidence 29,996 vs self-oracle 29,797 (+0.7%),
imbalance BETTER (1.114 vs 1.223), e2e +3% (repeat-confirmed; first
single-cell +17% was jitter); dynamic arm gain 18,695 ppm < 50k
threshold ⇒ trigger 0 (the stale-contig probe fires at 505k — the
apparatus discriminates correctly).

NOT yet durable: the sweeps-layer machinery — `dslots=<start>:<w>`
family param in gen_trace_routing (`_extract_rows` by ti; folds into
matrix identity), `ensure_oracle_routing` sidecar
(`<mid>.oracle_routing.txt`, PLACEMENT INPUT not routing), and the same
oracle basis for `ensure_eplb_load` (never mix information bases across
arms in a capsule). Prototype scripts in session scratch
(s1_extract/s1_analysis/scenario1_oracle_proto).

## Discovered constraint + queue

1. **l01 × loccap_sl is blocked by the frozen hcc combine inbuf** (loud
   assert "per-iteration routing variance is not supported by the frozen
   hcc inbuf yet"; partial-rank assert wedges the step — kill the STEP).
   Fix = combine-side capacity mode (reserve to the recv bound, slice
   per iteration) — the same treatment dispatch got; THE gating item for
   full-l01 kernel-arm totals.
2. Device-side grouped-GEMM segment sizes (kill the last host consumer
   of the blob).
3. Capsule-grade ladders: eps×flavor, budgets b1..b64 equivalence
   (K3 = 7-MiB multiples), pllf-static vs EPIC vs EPLB vs MoonEP under
   the shared scenario-1 basis (doc 12 protocol).
4. Scenario-1 durable machinery (above) + canonicalizing pllf as the
   default placement in future specs (user sign-off; pll_* exact arms
   stay for never-mix history).
5. d6 arms still run the legacy tail (their reroute-expansion needs the
   general path) — inflates baseline plan_ms; note when quoting plan
   premiums (comm/e2e comparisons unaffected).

## DATA CANON — SETTLED (user decisions 2026-08-22, end of session)

All future data-based sweeps use TWO canonical lanes; **K3 is retired**
(synthesized pools stay as historical never-mix capsules; the upscale
program is shelved).

- **Models**: Kimi-K2-Thinking (G=384, k=8, 61 layers) + Qwen3-235B
  (G=128, k=8, 94 layers). Same-prompt bonus: the dataset traced the
  same benchmark items through both (lcb 480/479 req, philosophy
  311/312).
- **Canonical topic**: `livecodebench/execution` (both models). Chosen
  on measurement: best volume + full slot coverage + M≈1.07 at the
  b8 rungs, strongest concentration on K2 (headroom 25.9/27.3% at
  8n/16n vs psychology 22.3/25.9). NOTE the model-dependent flip: on
  Qwen, philosophy carries MORE headroom (37–55%) than lcb (30–47%) —
  uniformity across models wins over per-lane maximization; philosophy
  is the second rung where the flip itself is the finding.
- **Control topic**: `mmlu/philosophy` (both models; K2's pulled
  2026-08-22, pool_sha 5d7333eb6a28; Qwen lcb pool_sha 17cdb0b644c7).
  K2 psychology (546 req) stays as the low-imbalance discriminator.
- **Layer knob** `s1layer ∈ {p5, m5}`: p5 = traced layer 5 (both);
  m5 = fifth-from-last (K2: 56, Qwen: 89). BOTH are canonical arms —
  measured orderings flip by model (K2: p5 more structured; Qwen: m5
  much more structured, e.g. philosophy m5 headroom 44.9/54.7% at
  4n/8n) — profile both, never collapse the knob.
- **Windows**: evaluated = decode slots [64, 96) (w=32, fully covered
  in lcb both models); oracle = previous window, gap ladder g ∈
  {0, 16, 32} → [32,64) / [16,48) / [0,32); wrong-topic control =
  the other topic's eval window. Measured drift (ppm, eval-vs-oracle):
  K2 lcb p5 104k/…/265k across the ladder, Qwen lcb p5 115k/…/264k;
  wrong-topic 394–626k (3.4–5.5× margins hold everywhere).
- **Eval batch composition**: REAL rows only, duplicated as needed
  (sampling-with-replacement of intact top-k sets), multiplicity M
  recorded as a cell fact. NO marginal sampling, NO cross-topic
  pooling, NO time-widening beyond the window. Worst corner: 32n×b64 →
  M≈39 (K2) — accepted; documented relief = w=48 at the cost of the
  g=32 rung.
- **Information basis**: every arm (EPLB/EPIC/ours) plans from the SAME
  oracle window — loads for the load-only baselines, rows for ours;
  self-oracle survives only as the labeled ceiling ablation.
- Remaining plumbing (unchanged from the queue above): dslots+layer
  extraction params in gen_trace_routing, oracle-routing sidecar,
  oracle-basis eplb_load, `shape: k2` preset.

### Canon TIGHTENED into SCHEMA protocol rule 10 (final user pass)

SCHEMA.md rule 10 is now the single normative source; deltas vs the
section above: NO control topic in default runs (philosophy = explicit
ablations only), `s1layer=p5` is the DEFAULT (m5 profiled only when
named), gap **g=0 is the DEFAULT** (ladder {0,16,32} for drift studies
only). Standing user instruction (also in memory): a prompt saying
"sweep" without explicitly naming s1layer/gap means DEFAULTS — never
expand data-source axes unprompted; ask when unsure.

## AUTHENTIC LLC E2E — LANDED (f080c90, job 57448022)

The full LLC stack (placement-fast from the previous-window oracle +
loccap_sl ext kernel + fast tail + graph + combine CAPACITY mode) runs
l01 end-to-end on BOTH canon lanes, correctness-gated with randomized
payloads. Structural items closed this pass:
1. **Combine capacity mode** (`enable_hc_combine(m_capacity=recv_cap)`,
   pack slices per iteration) — the combine metadata was ALREADY
   per-iteration in-window (derive_routed_meta + derive_combine_meta);
   only the buffers were exact-size.
2. **Grouped-GEMM splits lockstep — ROOT-CAUSE FIX with a disclosure**:
   enable_grouped_gemm held a REFERENCE to the setup splits while
   bind_iter_plan rebound `_group_splits_cpu` → the grouped GEMM
   segmented every iteration with SETUP sizes. Static-routing arms
   (d6/loccap_gpu) unaffected (stale==fresh). Relaxed kernel-arm cells:
   prior LATENCY numbers stand (same rows/FLOPs), but relaxed-iteration
   OUTPUTS were silently mis-segmented and were never numerically
   checked (only the final deterministic iteration is). Any future
   claim about relaxed-iteration outputs requires binaries/trees at or
   after this fix — never-mix note.
3. k2 shape preset; tail-graph outputs persist-copied post-replay
   (pool-across-streams insurance).

Numbers (canon data per SCHEMA rule 10, 4n b8, g=0, p5, one tree):
| lane | LLC l01 | EPIC l01 | delta | LLC l0 | plan lane |
| qwen | **12.28 ms** | 18.34 | −33% | 6.25 | 1.18 |
| K2   | **13.63 ms** | 17.50 | −22% | 7.50 | 1.14 |
(pllf+d6 controls: 19.33 / 18.80.) Realized oracle drift 135k / 206k
ppm — the placement tolerates it (scenario-1 authentic). Timing honesty:
placement untimed ONLY because oracle-based (pre-gating); ALL
post-gating computation (loads allgather, kernel route, phys exchange,
fast tail, combine metadata) is in the timed window; sizing constants
(bounds/f_cap) derive from the setup reference — capacity-only, no
in-window work hidden; the noted refinement is deriving them from the
oracle side + slack.

Next: capsule-grade sweep of this exact stack (the sweep-runner pllf/sl
arms + dslots durable plumbing), 8n `-q regular`, flavor ladder.

## BASELINE VERIFICATION PASS (job 57448901, canon data, l01, one tree)

All four expert-placement baselines, canon scenario-1 basis (EPLB/EPIC
consume ORACLE-WINDOW loads via s1c_*_oracle_load.json {version:1, G,
load}; MoonEP plans per-round natively — its authentic identity; LLC
consumes oracle-window rows). l0/l1 separable inside every l01 run
(per-phase CUDA events in all four drivers — no re-running needed).

| arm (authentic) | qwen l01 | K2 l01 | qwen l0/l1 | K2 l0/l1 | plan |
|---|---|---|---|---|---|
| MoonEP staged getmem | 23.77 | 28.82 | ~10.4/13.4 | ~10.8/17.7 | 2.3 timed |
| EPLB direct-a2av     | 23.16 | 23.30 | 8.45/8.26 | 8.78/9.24 | 6.0/4.7 timed |
| EPIC m=2 PEO hc      | 15.30 | 15.83 | 9.45/5.62 | 10.48/5.06 | ~0 UNTIMED (legacy m>1) |
| **LLC (ours)**       | **12.56** | **13.60** | 5.60/5.19 | 6.42/5.71 | 1.27/1.16 timed |

LLC vs best baseline (EPIC m=2): −18% qwen / −14% K2 — WITH LLC's
planning timed and EPIC's excluded (legacy accounting, m>1 has no
rule-5 planner; flag on every quote). vs EPLB: −46/−42%; vs MoonEP:
−47/−53% (MoonEP's un-deduped direct combine is its wall: 11.3/14.5 ms).

**Comm primitives from source (dispatch / combine / overlap):**
- MoonEP (test_moe_moonep_traffic.py, nvshmem+getmem, overlap OFF):
  dispatch = flux All2AllSingle one-sided NVSHMEM direct a2av (hidden +
  probs pair, single stage, no dedup); weights = per-round redundant-
  expert getmem pull (both projections, one join, serialized before
  GEMM); combine = route-weight scale → expert-side reverse-dedup →
  direct a2av transpose on the same pair → index_add at token home.
  Overlap: NONE (strictly serial; --overlap_prefetch is our ablation,
  off in the authentic arm).
- EPLB (test_moe_eplb_traffic.py, nvshmem): placement = DeepSeek EPLB
  global policy on the load vector; dispatch = pack → All2AllSingle
  direct a2av (no dedup — DeepEP-LL/decode transport class) → place;
  combine = combine_pack → reverse a2av same pair swapped splits →
  comb_dst home reduce. Overlap: NONE (single stream, sequential;
  driver :225-251). Per-iteration EplbIterPlanner (rule-5 timed).
- EPIC m=2 (test_moe_epic_traffic.py, hier_compress, groups 2):
  placement = §4.2 greedy replication on oracle loads; dispatch = Mode-2
  hier_compress a2av (PXN relay-identity: NVLink intra-node stage +
  inter-node node-dedup, GemmGroupedV2AGScatterOp.dispatch_only over
  virtual slots); combine = per-group TopkReduceScatterOp (hier +
  compress at nn>1). Overlap: **PEO phase pipelining IS present by
  design** (§5.2, perf_epic :342-375: all group dispatches on the comm
  stream, per-group compute gates on its own dispatch event → dispatch
  g1 ∥ GEMM g0, combine g0 ∥ gemm chain g1). NO tile-level fusion. The
  "expected none" holds for MoonEP/EPLB/LLC only.
- LLC m=1 (ours): same Mode-2 staged dispatch + capacity-mode
  TopkReduceScatterOp combine; single group ⇒ strictly staged, zero
  overlap (dispatch event fully precedes compute; combine gated on
  gemm1). Fused lb_union tile overlap deliberately NOT enabled here.

Fix required by this pass (committed): hcc combine pad-tail semantics —
bundle m_this includes the zero pad tail; the old exact-equality assert
made padded m>1 l01 un-runnable. LATENT before: zero-pad cells only.

## EPIC RULE-5 FAIRNESS PASS (job 57451349) — accounting now symmetric

EPIC m=1 AND m=2 run per-iteration TIMED planning on the general d6
fast tail (_fast_tail_g; D6 quota/expansion untouched — authentic
routing, only the shared tail class optimized). m=1 fast==legacy
bitwise; m=2 correctness-gated both models. Same-job l01 table (canon
data, oracle-window loads):

| arm | qwen | K2 | plan (timed) |
| EPIC m=1 d6 | 18.31 | 16.97 | 6.44 / 4.74 (≈5 ms = D6 expansion itself) |
| EPIC m=2 d6 (PEO) | 23.90 | 22.68 | 6.71 / 5.01 |
| LLC | 12.95 | 13.71 | 1.22 / 1.24 |

With planning timed, **PEO m=2 loses to m=1 at 4n** (l0 doubles: two
half-size dispatch+GEMM passes cost more than the pipeline hides —
re-confirms the 8.17 "PEO m-split loses at our scale" verdict under
symmetric accounting). LLC −29% vs the best EPIC (now m=1). NEVER-MIX:
pre-commit m=2 cells excluded planning. d6 tail graphing
(FLUX_PLL_TAIL_GRAPH) not enabled for d6 arms — available follow-up.
