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
