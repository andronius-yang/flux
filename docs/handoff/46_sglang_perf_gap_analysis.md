# Handoff 46: why the LoPEP layer wins in the harness and loses inside SGLang — record audit + profiling plan

Opened 2026-09-28 (session 3 of the SGLang lane). Authority for the integration itself stays
`45_sglang_integration.md`; this file is the performance-gap analysis the user asked for: what the
existing records already prove, what they cannot answer, and the measurement plan that isolates the
cost before any optimization is attempted. Nothing here was re-measured; every number is quoted from
`$PSCRATCH/workspace/andrewy/logs/sglang/` (chain/regime/prefill reports, `server_timing_*.log`,
`nsys/gate.out`) or `figs/main_perf_v5/figure_src.csv`.

## 1. The two claims are not measured in the same regime

| quantity | paper figure (replay, `--isolated`) | SGLang serving (DP attention, 16 ranks) |
|---|---|---|
| tokens per rank per layer-step | b1 = 128, b4 = 512, b16 = 2048 (H 4096) | decode: concurrency/16 = **1–40**; prefill chunk: up to 2048 |
| node count where the claim is made | 4n, 8n, 16n (win grows with nodes) | **4n only** (S5 = 8n/16n never run) |
| comparator | COMET / COMET+EPLB (flux all-gather + dense grouped GEMM, research build) | stock SGLang: NCCL all-gather + triton `fused_experts` + reduce-scatter, graphs off or on |
| what is timed | one layer, warm, fixed routing pool, host run-ahead unconstrained | 48–94 layers, scheduler, attention, DP padding, our adapter's extra collectives |

Paper Qwen rows (figure_src.csv, ms, `ours12` vs `comet_eplb` vs `comet`):

| nodes | b1 | b4 | b16 |
|---|---|---|---|
| 4n | 2.80 / 3.75 / 4.46 | 4.43 / 4.96 / 5.93 | 11.43 / 12.45 / 15.02 |
| 8n | 3.79 / 5.94 / 7.70 | 6.16 / 8.50 / 10.63 | 15.26 / 24.85 / 28.34 |
| 16n | 5.81 / 10.45 / 16.32 | 9.28 / 17.84 / 24.43 | 23.70 / 57.55 / 65.67 |

K2 4n b1 is a loss (3.80 vs 3.68). So the paper's own data says: at 4n the layer wins by 10–25 %
against a research all-gather baseline, and the large wins are at 8n/16n. Every SGLang measurement
was taken at 4n, i.e. at the paper's weakest point, and below its smallest budget in decode.

## 2. What the records already establish (decode, 30B, 4n, ours overlap arm, no swap)

Per-layer-step CUDA-event brackets on the forward stream (`server_timing_rebal{0,1}.log`, ShareGPT
c=16, mean over 500 steps, ms):

| phase | rebalance ON | rebalance OFF | what is inside |
|---|---|---|---|
| pad+loads | 0.19 | 0.40 | pad copies + `exchange_loads` NCCL all-gather |
| route+xchg | 0.16 | 0.16 | routing all-gather + tail graph |
| meta+check | 0.34 | 0.34 | `derive_routed_meta` D2H sync, numpy demands, `derive_combine` host loops |
| dispatch | 0.84 | 0.81 | dispatch op incl. wire, barrier, GEMM spinning on signals |
| act | 0.07 | 0.07 | SwiGLU |
| combine | 0.60 | 0.60 | combine op incl. wire, 2 barriers, prereduce/pack, GEMM |
| **bracket total** | **2.19** | **2.37** | |
| ITL (bench, c=16) | 156 (s1a) / 149 (n4c) | 130 (n4b) | |

Three facts follow directly:

1. **The adapter's rebalance is outside the bracket and is a net loss in decode.** The bracket
   dropped 0.18 ms but ITL rose ~19 ms per token = **+0.4 ms per layer** for the three
   `all_to_all_single` calls (hidden in, meta in, output out) plus the allocations around them.
   Rebalance is ON by default because it fixes prefill (235B TTFT median 2087 → 1181 ms); in
   decode the DP shards are already even and it only costs.
2. **The layer's own fixed cost is ~2.2 ms per layer-step and is not token-proportional**: the
   regime sweep (`regime_reg30b_report.txt`) shows our ITL flat at 180 ms from 2 to 40 tokens per
   rank and the baseline flat at ~80 ms (graphs off) / ~75 ms (graphs on). ITL percentiles
   (`bench_*.jsonl`) are tight for both arms (ours p95/median 135/128 at c=16), so it is a uniform
   per-step cost, not a tail.
3. **The nsys gate (`nsys/gate.out`, replay, 2n, 8 tok/rank, Qwen-235B shape) splits the 2.47 ms
   step into GPU busy 1.27 + host-induced gap 1.19, 132 launches.** Busy is not compute: CUTLASS
   GEMMs 0.41 (tiles spin on wire signals), 3 `barrier_on_stream` 0.35, 2 NCCL LL all-gathers 0.33,
   combine prereduce 0.25 + pack 0.16, proxy signal 0.13, memcpy 0.11. At 128 tok/rank the same
   step is 2.85 = 1.83 busy + 1.02 gap: **even the paper's b1 number carries ~1 ms of host gap.**
   In the serving-shaped run (`check_t8`) the loads all-gather alone is 1.29 ms per step: NCCL
   kernel time there is rank skew absorbed by the collective, not transfer.

Swap arm (staged lane, `s1_staged_timing.log`): movement is off the critical path (push+commit
0.17 ms) but the host decision costs 1.38 ms on swap steps and 0.49 ms (`loads.cpu()` sync) on every
other step, and the band trigger fires on 199 of 200 decode steps. This is a decode-regime routing
noise problem, not a mechanism problem; it is a pending user decision (floor/hysteresis).

## 3. What the records cannot answer (the actual gaps)

G1. **The stock path's per-layer cost has never been measured.** The "~0.5 ms" in handoff 45 is
    inferred. Re-deriving it from ITL: baseline graphs-off 72 ms / 48 layers = 1.5 ms per full
    layer; ours (rebalance ON) 149 / 48 = 3.1 ms per full layer with ~2.6 ms in the MoE (2.2 bracket
    + 0.4 rebalance). Same attention on both sides gives **stock MoE ≈ 1.0–1.1 ms graphs off**, and
    graphs-on ITL 34 ms implies ~0.4–0.5 ms captured. The gate's NO-GO compared a captured
    ours (~1.3 ms) against the *graphs-on* stock estimate; against the matched graphs-off baseline
    the same arithmetic is near parity. The verdict depends on a number nobody measured.
G2. **No prefill-step breakdown exists.** The collector dropped prefill steps (fixed 09-27 02:30),
    never rerun. The 235B prefill result (TTFT parity, ITL 455 vs 312) is unattributed.
G3. **No 8n/16n serving run.** Plan-2 stage 2G's 16n regime test was never submitted. The paper's
    advantage lives there.
G4. **No isolated head-to-head against the production kernels.** The paper's comparators are flux
    builds of all-gather+dense GEMM; SGLang's triton `fused_experts` + NCCL may be faster than COMET
    at equal M. Until a "stock" arm runs inside `bench/replay.py`, "the layer wins e2e" is a claim
    about research baselines, not about SGLang.
G5. **No profile of the serving process itself.** The nsys gate profiled `bench/replay.py` and
    `serving_check.py`; the real process has SGLang's streams, scheduler and the adapter's
    collectives (the channel-aliasing hangs already showed the two layouts differ).
G6. **The timing bracket starts after the adapter's work** (`t.mark("start")` is inside `step`,
    after `exact_counts` and the rebalance all-to-alls), and only two size classes are kept
    (`S_b < 64` decode, else prefill), so fixed vs proportional cost was never fitted.

## 4. Profiling strategy (ordered; each step has a decision it feeds)

P0. **Same-instrument stock measurement (closes G1).** Add the identical CUDA-event bracket around
    the baseline MoE block in the patched SGLang (`Qwen3MoeSparseMoeBlock.forward_normal` plus the
    communicator's `dp_gather_partial` / reduce-scatter that belong to it), env-gated, logged per
    step-size bin. Run baseline graphs-off and graphs-on at c=16/64/256 and the prefill-heavy load,
    30B 4n. Output: stock ms per layer-step per bin. Decision: the parity target for decode and
    prefill, in ms, at 4n.
P1. **Full-bracket, binned ledger for ours (closes G2, G6).** Move `start` before `exact_counts`
    and the rebalance; add marks `counts`, `rebal_in`, `rebal_out`; bin by tokens per rank
    (1–4, 5–16, 17–64, 65–256, 257–1024, 1025–2048) instead of two classes; fit t = a + b·M per
    phase. Run rebalance ON and OFF. Decision: which phases are fixed (a) and their sum vs P0.
P2. **nsys of the serving process (closes G5).** NVTX range per layer-step and per phase in
    `SharedComm.step`, `nsys profile -t cuda,nvtx,osrt --sample=cpu` on one rank of the 30B 4n
    server under the c=16 load and under the prefill-heavy load, both arms (`gap_report.py`
    extended with the phase ranges and the OS-runtime call names). Output per phase: GPU busy, gap,
    launch count, and what the host thread is doing in the gap (`cudaEventSynchronize`, NCCL host
    work, python). Decision: how much of the 1.2 ms gap is syncs (removable by S-A/S-B) vs launch
    rate (removable only by capture).
P3. **Isolated head-to-head at the paper's budgets against the production kernels (closes G4).**
    Add a `stock` arm to `bench/replay.py`: NCCL all-gather → sglang `fused_experts` (triton) on
    the local experts → NCCL reduce-scatter, eager and graph-captured. Run per rule 6 order
    b1, b4, b16 first, then sub-b1 points (8, 32 tokens per rank), at 4n, 8n, 16n
    (16n in `-q regular`). Decision: the crossover M and node count where the layer beats the stock
    kernels at all; this is the honest version of the paper's claim.
P4. **Round-trip ledger at tiny M.** Count serialized inter-node round trips per layer-step in ours
    (loads all-gather, routing all-gather, 3 barriers, dispatch intra/relay/gateway hops, combine
    conv/wire/intra ladders, proxy signals) against the stock path's two collectives; microbench
    one CXI `putmem_signal_on_stream` round trip, one 16-rank `barrier_all_on_stream`, one 16×1 KB
    LL all-gather. Decision: the latency floor of the current wire at decode sizes, i.e. whether
    any amount of host-side work removal can reach P0's target, or a small-M path is required.
P5. **16n serving regime test (closes G3).** 30B decode at the concurrency cap and prefill-heavy
    both models, ours vs baseline graphs off/on, calibration re-solved offline from the 4n dumps.
    Decision: whether the paper's 16n advantage survives inside SGLang at any per-rank token count.

Order: P0 and P1 need one 4n interactive allocation and a small patch each; P2 rides the same
allocation; P3 is harness work plus 4n/8n/16n; P4 is a 2n microbench; P5 is a regular-queue job.

## 5. Optimization candidates the records already justify (not yet done)

- **Conditional rebalance**: only when max/mean of the per-rank counts exceeds a threshold
  (prefill-mixed steps); saves ~0.4 ms per decode layer-step (~19 ms ITL at c=16) at zero risk.
- **Swap trigger floor / hysteresis in the decode regime** (pending user decision): removes the
  1.38 ms host decision on swap steps and lets the 0.49 ms loads sync be skipped when the floor
  is not reached.
- **Fold the loads exchange into the routing exchange** (one all-gather instead of two) and drop
  the per-step host sync via S-A device metadata (already built, `LOPEP_DEVICE_META=1`): P2 tells
  how much of the 0.34 + 0.19 ms this returns.
- **Small-M path** (design only after P4): below a per-rank token threshold, skip the hierarchical
  relay and the three barriers; one direct exchange each way. The gate's busy breakdown says the
  decode floor is the hop chain (barriers 0.35 + all-gathers 0.33 + GEMM-spin 0.41 + proxy 0.13),
  not compute, so this is the only lever that changes the floor.
- **Graph capture (S-B/S-C/S-D)** pays only the gap (~1.2 ms); do it after the floor is lowered.

## 6. Controls audit (user question 2026-09-28): what was and was not held constant

Held constant inside every chain (same allocation, same nodes, sequential arms): tp=dp=ep=16, DP
attention, flashinfer, `--chunked-prefill-size 32768` (2048 per rank), `--max-running-requests 256`,
`--disable-overlap-schedule`, `--mem-fraction-static` (0.85 30B / 0.88 235B), same prompt sets,
matched baseline = graphs off. Not constant:

| factor | baseline | ours | binds? |
|---|---|---|---|
| per-rank KV pool, 30B 80G nodes (n4b/s1a) | 664k tokens | 524k | no (usage ≤ 0.02) |
| per-rank KV pool, 30B 40G nodes (n4c/reg30b) | 295k | 155k | possibly at c=1024/2048 (ours timed out there) |
| per-rank KV pool, 235B 80G nodes (p235) | 160k | 38k (weights+heap 61.7 vs 40.3 GB) | no in the c=32 run (usage ≤ 0.15, 3 running/rank both) |
| node type across chains | n4b, s1a, p235 on 80 GB nodes; n4c, reg30b on 40 GB nodes | same | baseline ITL identical on both (72/73), so cross-chain reads of ours are the only ones affected; the rebalance +18 ms ITL was a same-node A/B (job 58942368) |
| CUDA_DEVICE_MAX_CONNECTIONS | SGLang default 4 | 24 (required) | part of the arm |
| timing collector | off | on in chain s1a and the `server_timing_*` runs | +overhead on ours only in those runs |
| redundant experts / expert location | none | +2 slots per rank, calibrated placement | by design |

Fix for future chains: pass `--max-total-tokens` equal on both arms (the smaller pool) so the
scheduler's admission is identical, and keep one node type per campaign.

Node counts: every serving run was 4 nodes (16 ranks). The 16n numbers in the record are the
paper's replay harness only. Allocation options: `debug` QOS = 30 min, max 8 nodes, 2 jobs per user
(`salloc -q debug -C gpu -N 8 --gpus-per-node=4 -t 30 -A m5350_g`); 16n needs `-q regular`.
