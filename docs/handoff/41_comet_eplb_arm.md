# Handoff 41 — COMET + EPLB arm (`l01_allgather_dense_eplb`), 2026-09-16

Reviewer request on the main-perf figure: add a separate **overlap +
placement** baseline — COMET as the communication/compute-overlap substrate,
EPLB as the placement + replication policy. User ruling (2026-09-16): build
it as a new arm and make it runnable; **results stay private** (no figure
row, no `figure_src` change) until the user has judged them. Branch `pv3`
(worktree `$PSCRATCH/workspace/andrewy/flux-pv3`, runs the MAIN tree's
binary through the `python/flux/lib` symlinks — this lane is python-only,
no rebuild).

## 1. Viability verdict

Viable and cheap: **no kernel change**. The fused COMET ops
(`GemmGroupedV2AGScatterOp` dense allgather + `GemmGroupedV2GatherRSOp`
dense) have exactly one destination rule — expert homing,
`dest = e // ep_nexperts` — and an EPLB plan already lives in that space:
`P = W * nlp` physical slots, slot `p` hosted by rank `p // nlp`, `p2l[p]`
the logical expert it replicates. Building the ops with `nexperts = P` and
feeding them routing in physical-slot space makes the unmodified kernels
execute the placement. This is the MoonEP virtual-expert-space mechanism
(`flux.testing.moonep_fused_map`, handoff/memory `moonep-fused-merge`)
applied to the EPLB plan; the OURS driver does the same for pv2 + LocCap
(`vce`).

What the arm therefore measures: COMET's dense allgather dispatch is
**placement-invariant** (every rank receives every token), so EPLB moves
only the grouped-GEMM rows (and the dense combine's send rows) between
ranks — "overlap + placement" = COMET's overlap with the residual
imbalance of the static pool-oracle placement instead of the home
placement. The EPLB arm `eplb_l01` (direct wire, no overlap) and plain
COMET `l01_allgather_dense` are the two neighbours; compare all three
inside one capsule (rule 4).

## 2. What was built

- `python/flux/testing/comet_eplb.py` — pure helpers: `load_pool_load`
  (the runner's `.oracle_load.json` sidecar), `build_comet_eplb_plan`
  (vendored `rebalance_experts` via `eplb_semantics.build_eplb_plan`,
  global policy, `nlp = G/W + 2`, asserts no empty slot),
  `derive_physical_routing_all` (setup reference = every rank's
  `EplbIterPlanner.derive_fused` shard), `fill_canonical_slot_weights`
  (replica slots get the canonical per-logical weights, same seeds as the
  eplb arm's one-time placement; returns the bytes a real placement would
  move), `comet_eplb_stats` (the eplb_* cell facts).
- `test/python/moe_combined/test_moe_l0l1_traffic.py` — `--placement eplb`
  (+ `--eplb_load_file/--eplb_policy/--eplb_redundant_per_rank/
  --eplb_replica_select/--eplb_no_interleave`): after the logical routing
  is loaded and matrix-checked, the plan is built (untimed, hash
  all-gathered across ranks), `choosed_experts`/`args.G` switch to the
  physical space, every op/buffer/reference is built for `P` experts;
  `plan_comm_fn` becomes `derive_fused()` (sender-local `local_spread`,
  no exchange) + the physical routing allgather — timed, inside the
  plan_comm bracket (`eplb_route_bracket=plan_comm` in the record);
  weights of both layers are overwritten in place with canonical
  per-logical values so the torch two-layer reference (same physical
  layout) is the correctness verdict with no special case. Records
  `comet_placement`, `G_logical`, `planner_impl=comet_eplb_physical_slots`,
  the eplb_* facts (imbalance before/after/pred, replicas, re-homed slots,
  load source + sha). `--impl flux|torch` only; not combinable with
  `--routing_sched_files` (v1).
- `sweeps/variants.py` — `l01_allgather_dense_eplb` (+ `_lstatic` twin),
  `eplb_load: True`.
- `sweeps/sweep.py` — the l01 driver branch passes `--eplb_load_file` for
  `eplb_load` variants; the oracle-load sidecar is generated for them; the
  dense combine send cap is widened to max(logical bound, 2 x mean
  rows/rank) (`eplb_dense_send_headroom`; global re-homing voids the
  column-sum bound; the op's check is collective so overflow aborts, never
  hangs). Other variants' env is byte-identical to before.
- `sweeps/SCHEMA.md` — arm paragraph next to the l01 driver notes.
- `test/python/moe_combined/test_comet_eplb_map.py` — CPU tier, 27 tests:
  homing/fidelity (`p2l[phys] == logical` entry-wise, k-order kept), rows
  == `plan.physical_rows_per_rank()`, setup reference == each rank's
  in-window shard, `local_spread` largest-remainder counts, `local_static`
  = src mod C, placement depends on the pool only, canonical replica
  weights identical + moved-bytes accounting, fuzz. **PASS 27/27** on the
  login node (env: see §4).
- Specs `sweeps/specs/comet_eplb_gate_4n_{k2,qwen}.yaml`: main-perf
  shape/pool (`dslots=64:32`, livecodebench layer 5), b1 + b16, arms
  `l01_allgather_dense`, `l01_allgather_dense_eplb`, `eplb_l01`, isolated,
  correctness ON, random payload per iteration.

## 3. Gate run (log §5)

Chain `$PSCRATCH/workspace/andrewy/logs/pv3/comet_eplb_gate_chain.sh`
(grant -> K2 capsule -> Qwen capsule -> release; output under
`logs/pv3/comet_eplb_gate_<ts>/`). Results appended in §5 when they land.

## 4. Ops notes

- **Site drift 2026-09-16:** `module.sh` now dies at `module load
  gcc/12.2.0` (module gone) after the `PrgEnv-gnu` swap error; nothing
  after it (cudatoolkit/nvshmem/nccl/conda) loads, `python` is absent.
  `pv3_env.sh` inherits the failure silently. Runtime-only recipe that
  works: `logs/pv3/comet_eplb_env.sh` (restore defaults; load
  cudatoolkit/12.4 nvshmem/3.2.5-1 nccl/2.24.3 conda/Miniforge3-25.11.0-1;
  activate; CUDA_HOME pinned to hpc_sdk 24.5/12.4 since the cudatoolkit
  module now points at 25.5/12.9; PYTHONPATH = the pv3 tree). Builds need
  the gcc 12 module story resolved first (gcc-native-mixed/12 exists).
- Dry run of the K2 gate: 6 cells; the eplb arm receives
  `--eplb_load_file <matrix>.oracle_load.json` (dslots previous-window
  basis, same file as `eplb_l01`), send cap 8192 (b1) / 24576 (b16, vs
  16384 for plain COMET), heap 13-14G.

- **NVSHMEM init SIGSEGV on EVERY arm (2026-09-16, ROOT-CAUSED + FIXED):**
  the 23:34 4n chain (job 58453128) failed all 12 cells in ~13 s each,
  plain COMET and `eplb_l01` included — rank stdout stops at "before
  flux_shm initialization". 1-PE probe (`logs/pv3/nvshmem_init_probe.py`)
  under gdb: `fi_control <- fi_dupinfo_ <- fi_getinfo_1_7 <-
  nvshmemi_libfabric_init_state (nvshmem_transport_libfabric.so.3)`, i.e.
  NVSHMEM 3.2.5-1-25.03 (built against libfabric 1.x, ABI 1.7) crashes in
  **libfabric 2.3.1's** 1.7-ABI compatibility shim. The Sep-14 compute-node
  image (driver 580.178.04 open kernel module, CPE 26.03 default) replaced
  `/opt/cray/libfabric/1.20.1` with 2.3.1. `nvshmem-info -a` still runs
  (no transport init) and torch CUDA works, which is why it looked
  NVSHMEM-specific. **Fix:** `module load libfabric/1.20.1` (NERSC's copy
  under `/global/common/software/nersc9/libfabric/1.20.1`) BEFORE the
  runtime modules — `logs/pv3/comet_eplb_env_fab120.sh`; probe passes on
  1 and 2 nodes (UID bootstrap + libfabric/CXI transport). This affects
  every Perlmutter lane, not this arm: any run after the 2026-09-14 image
  needs the libfabric 1.20.1 module until NERSC rebuilds NVSHMEM against
  2.x (or the tree is rebuilt against a new NVSHMEM).

## 5. Log

- 23:29 chain v1 died on `set -u` x broken module.sh; 23:34 chain v2
  granted job 58453128 immediately (4n interactive, 60 min) — all 12
  cells failed in the NVSHMEM init segfault above (capsules
  20260917-063449 K2 / the Qwen twin are launch failures, NOT verdicts;
  left uncommitted).
- 23:38 debug grant 58453253 (2n); 23:44 root cause + libfabric 1.20.1
  fix verified (§4); 23:46 2n K2 gate (`comet_eplb_gate_2n_k2`, 3 arms x
  b1/b16, correctness on) running on it: `l01_allgather_dense` b1 ok 16 s,
  `l01_allgather_dense_eplb` b1 **ok 23 s (correctness PASS, first
  hardware run of the arm)**.
- 23:49 **2n K2 gate COMPLETE 6/6 ok, correctness PASS on every cell**
  (capsule `20260917-064442_perlmutter_0057d181`, uncommitted — user
  commits). Median over iterations of the max over ranks, ms:

  | arm | b | plan_comm | plan | l0 | l1 | e2e | total |
  |---|---|---|---|---|---|---|---|
  | COMET `l01_allgather_dense` | 1 | 0.12 | 0.25 | 1.67 | 1.72 | 3.38 | 3.75 |
  | **COMET+EPLB** | 1 | 2.00 | 0.26 | 1.70 | 1.74 | 3.43 | 5.67 |
  | EPLB `eplb_l01` | 1 | 0.20 | 3.31 | 3.26 | 2.91 | 6.04 | 9.38 |
  | COMET | 16 | 0.16 | 0.30 | 4.83 | 5.42 | 10.28 | 10.75 |
  | **COMET+EPLB** | 16 | 2.09 | 0.30 | 4.61 | 5.08 | 9.72 | 12.11 |
  | EPLB | 16 | 0.16 | 3.09 | 14.57 | 16.04 | 30.36 | 33.56 |

  Reading (2n only, provisional): the placement does what it should on
  the pass itself — COMET+EPLB e2e is on par at b1 and 5% faster at b16
  (l0 −5%, l1 −6%; realized imbalance 1.30 → 1.13 on the b1 batch, pool
  prediction 1.00; 359/400 slots re-homed, 16 replicas). The arm's
  **total** is behind COMET because of the per-iteration logical→physical
  router (`EplbIterPlanner.derive_fused`, ~1.9 ms, charged to plan_comm):
  the eplb arm's own sender-local rule, which its staged arm pays inside
  plan_ms (3.1–3.3 ms there). That router is torch-op code with device
  syncs (bincount + reroute_expand); a fused kernel would bring it to the
  ~0.25 ms class of COMET's own derive. **Open question for the user:**
  keep the faithful eplb-arm router (honest, same code as `eplb_l01`) or
  optimize it before quoting (a reviewer may otherwise read the 2 ms as
  the harness, not the method). 4n K2 + Qwen gates running (job 58453556).
- 23:53 **4n K2 gate COMPLETE 6/6 ok, correctness PASS** (capsule
  `20260917-064929_perlmutter_7e567e69`, uncommitted). Same statistic:

  | arm | b | plan_comm | plan | l0 | l1 | e2e | total |
  |---|---|---|---|---|---|---|---|
  | COMET | 1 | 0.18 | 0.25 | 1.58 | 1.68 | 3.23 | 3.66 |
  | **COMET+EPLB** | 1 | 2.01 | 0.26 | 1.66 | 1.70 | 3.29 | 5.53 |
  | EPLB | 1 | 0.23 | 3.00 | 2.82 | 2.28 | 4.94 | 7.97 |
  | COMET | 16 | 0.26 | 0.33 | 7.28 | 7.90 | 15.17 | 15.76 |
  | **COMET+EPLB** | 16 | 2.23 | 0.34 | 6.65 | 7.02 | 13.75 | 16.45 |
  | EPLB | 16 | 0.21 | 3.08 | 15.81 | 16.63 | 32.26 | 35.28 |

  Realized imbalance 1.55/1.57 → 1.17/1.13 (b1/b16; pool prediction
  1.00; 32 replicas, 390/400 slots re-homed). e2e: on par at b1, **−9% at
  b16** (l0 −9%, l1 −11%); total: +51% at b1 / +4% at b16 — entirely the
  ~2.0–2.2 ms router bracket. For reference the main-perf 4n K2 COMET row
  (figure_src) is a different binary/day — never compare across.
- 23:55 **4n Qwen gate COMPLETE 6/6 ok, correctness PASS** (capsule
  `20260917-065339_perlmutter_c5acfefa`); chain DONE, job 58453556
  released.

  | arm | b | plan_comm | plan | l0 | l1 | e2e | total |
  |---|---|---|---|---|---|---|---|
  | COMET | 1 | 0.15 | 0.25 | 1.22 | 2.74 | 3.91 | 4.28 |
  | **COMET+EPLB** | 1 | 2.02 | 0.25 | 1.15 | 2.27 | 3.41 | 5.65 |
  | EPLB | 1 | 0.24 | 3.22 | 1.99 | 1.41 | 3.41 | 6.45 |
  | COMET | 16 | 0.28 | 0.38 | 6.32 | 7.06 | 13.50 | 14.13 |
  | **COMET+EPLB** | 16 | 2.23 | 0.40 | 5.60 | 5.96 | 11.57 | 14.13 |
  | EPLB | 16 | 0.24 | 3.07 | 15.33 | 16.16 | 31.15 | 34.28 |

  Realized imbalance 1.99/2.05 → 1.21/1.21 (147/400 slots re-homed,
  32 replicas). e2e −13% b1 / −14% b16 vs COMET; total +32% b1 / tie
  b16 — again the router bracket.

## 5b. Fused router kernel (2026-09-17, user directive) + main-perf lanes

User ruling: fuse the router, then gather main-perf data at 1/4/16 MiB
on 4n/8n/32n, queueing short walltimes.

- `python/flux/testing/_comet_eplb_ext.cu` + `comet_eplb_ext.py` (JIT
  torch extension, build dir `$PSCRATCH/workspace/andrewy/
  comet_eplb_ext_build`, gcc-native/12.3 + CUDA 12.4 pin — the env script
  now loads gcc-native/12.3): ONE launch `route_local(topk_i32 [S,K], l2p,
  lcnts, src, mode, interleave) -> dst_phys [S,K]`. Grid = one block per
  logical expert; pass 1 counts n, thread 0 derives the coprime interleave
  stride/offset exactly as `_interleave_params`; pass 2 scans the entries
  in order with a block-wide ballot prefix (deterministic ordinals), maps
  `qr = (ordinal*stride+offset) % n` to the instance by the closed-form
  largest-remainder prefix `(j+1)*base + min(j+1, rem)` (local_spread) or
  `src % C` (local_static), writes `l2p[l, replica]`. Bit-exact with
  `EplbIterPlanner.derive_fused` by construction (same ordinal order:
  dense routing = at most one entry per token per expert).
- Driver: `--eplb_router kernel|torch` (default kernel); at setup the
  kernel output is asserted equal to `derive_fused` on this rank's shard
  (drift guard, untimed), the per-iteration `plan_comm_fn` then runs the
  kernel + the physical-routing allgather. `eplb_router` is recorded.
- Tests: `test_comet_eplb_map.py` gained GPU parity tests (case battery x
  {local_spread, local_static} x {interleave on/off}, every source rank)
  and a large fuzz (G 384/896, K 8/16, S to 4096) with a kernel-vs-torch
  timing print. Run on the 4n lane before its specs.
- Specs `comet_eplb_mp_{4n,8n,32n}_{k2,qwen}.yaml`: main-perf pool,
  b1/4/16, `l01_allgather_dense` + `l01_allgather_dense_eplb`, isolated,
  10+5 iters, correctness off (gated 9/16). Lanes via
  `logs/pv3/chain_mp.sh` (status bus `logs/pv3/comet_eplb_status.txt`):
  mp4n interactive 35 min (+pytest), mp8n debug 30 min, mp32k / mp32q
  regular 20 min each (short walltimes for backfill).

### 5c. Main-perf results (fused router; median over iters of max over ranks, ms)

- 00:11 kernel parity on the 4n grant: 24/24 bit-exact; timing 0.073–0.077
  ms vs torch 1.78–1.86 ms (W 8/32, G 384/128).
- 4n K2 (capsule `20260917-071106_perlmutter_b2e95c85`, 6/6 ok):

  | arm | b | plan_comm | plan | l0 | l1 | e2e | total |
  |---|---|---|---|---|---|---|---|
  | COMET | 1 | 0.15 | 0.25 | 1.58 | 1.66 | 3.24 | 3.65 |
  | **COMET+EPLB** | 1 | 0.13 | 0.26 | 1.66 | 1.72 | 3.30 | 3.68 |
  | COMET | 4 | 0.16 | 0.27 | 2.57 | 3.00 | 5.49 | 5.94 |
  | **COMET+EPLB** | 4 | 0.18 | 0.27 | 2.49 | 2.64 | 5.10 | 5.54 |
  | COMET | 16 | 0.35 | 0.35 | 7.58 | 8.06 | 15.59 | 16.34 |
  | **COMET+EPLB** | 16 | 0.23 | 0.34 | 6.93 | 6.95 | 13.77 | 14.35 |

  total: +1% b1, −7% b4, −12% b16 (imbalance 1.57 → 1.13 at b16).
- 8n K2 (capsule `20260917-071215_perlmutter_49690677`, 6/6 ok):

  | arm | b | plan_comm | plan | l0 | l1 | e2e | total |
  |---|---|---|---|---|---|---|---|
  | COMET | 1 | 0.24 | 0.26 | 2.18 | 2.25 | 4.40 | 4.94 |
  | **COMET+EPLB** | 1 | 0.29 | 0.26 | 2.20 | 2.19 | 4.36 | 4.97 |
  | COMET | 4 | 0.33 | 0.29 | 4.05 | 4.94 | 8.84 | 9.56 |
  | **COMET+EPLB** | 4 | 0.30 | 0.29 | 3.95 | 4.37 | 8.32 | 9.00 |
  | COMET | 16 | 0.40 | 0.38 | 13.39 | 15.35 | 28.71 | 29.60 |
  | **COMET+EPLB** | 16 | 0.38 | 0.38 | 12.15 | 13.39 | 25.79 | 26.60 |

  total: +1% b1, −6% b4, −10% b16 (imbalance 2.00 → 1.12 at b16, 64
  replicas, 440/832 slots re-homed).
- 4n Qwen (capsule `20260917-071304_perlmutter_f72b0109`, 6/6 ok):

  | arm | b | plan_comm | plan | l0 | l1 | e2e | total |
  |---|---|---|---|---|---|---|---|
  | COMET | 1 | 0.15 | 0.25 | 1.17 | 2.75 | 3.91 | 4.27 |
  | **COMET+EPLB** | 1 | 0.16 | 0.25 | 1.12 | 2.26 | 3.37 | 3.75 |
  | COMET | 4 | 0.19 | 0.27 | 2.25 | 3.25 | 5.42 | 5.87 |
  | **COMET+EPLB** | 4 | 0.18 | 0.27 | 1.94 | 2.58 | 4.51 | 4.96 |
  | COMET | 16 | 0.28 | 0.41 | 6.30 | 7.44 | 13.65 | 14.31 |
  | **COMET+EPLB** | 16 | 0.31 | 0.40 | 5.78 | 6.00 | 11.79 | 12.45 |

  total: −12% b1, −16% b4, −13% b16 (imbalance 2.05 → 1.21 at b16).
- 8n Qwen (capsule `20260917-071417_perlmutter_eedaf8bb`, 6/6 ok):

  | arm | b | plan_comm | plan | l0 | l1 | e2e | total |
  |---|---|---|---|---|---|---|---|
  | COMET | 1 | 0.26 | 0.25 | 1.91 | 4.24 | 6.14 | 6.66 |
  | **COMET+EPLB** | 1 | 0.23 | 0.26 | 1.89 | 3.58 | 5.46 | 5.94 |
  | COMET | 4 | 0.31 | 0.30 | 3.70 | 4.93 | 8.55 | 9.24 |
  | **COMET+EPLB** | 4 | 0.30 | 0.31 | 3.58 | 4.32 | 7.89 | 8.50 |
  | COMET | 16 | 0.49 | 0.51 | 12.00 | 13.46 | 25.46 | 26.43 |
  | **COMET+EPLB** | 16 | 0.59 | 0.52 | 11.32 | 12.31 | 23.83 | 24.85 |

  total: −11% b1, −8% b4, −6% b16 (imbalance 2.43 → 1.19 at b16).
  4n + 8n lanes DONE 00:16 and released. **User clarification 00:23: main
  perf = 4n/8n/16n, no 32n** — the 32n jobs were cancelled before any
  grant (chains exited "no grant"), specs removed, 16n lane queued
  (`chain_mp.sh mp16n regular 16 25`, granted after 9 min as 58454941).
- 16n K2 (capsule `20260917-073316_perlmutter_07377401`, 6/6 ok):

  | arm | b | plan_comm | plan | l0 | l1 | e2e | total |
  |---|---|---|---|---|---|---|---|
  | COMET | 1 | 0.42 | 0.28 | 3.57 | 3.88 | 7.44 | 8.15 |
  | **COMET+EPLB** | 1 | 0.50 | 0.28 | 3.53 | 3.96 | 7.46 | 8.32 |
  | COMET | 4 | 0.61 | 0.32 | 8.90 | 9.10 | 17.90 | 18.83 |
  | **COMET+EPLB** | 4 | 0.75 | 0.33 | 8.14 | 8.81 | 16.69 | 17.61 |
  | COMET | 16 | 1.03 | 0.53 | 29.98 | 29.76 | 59.23 | 60.42 |
  | **COMET+EPLB** | 16 | 0.77 | 0.49 | 27.85 | 28.77 | 55.79 | 56.92 |

  total: +2% b1, −6% b4, −6% b16 (imbalance 2.41 → 1.24 at b16, 128
  replicas, 502/1664 slots re-homed). At 16n the dense allgather wire
  dominates both layers, so the GEMM-row gain is a smaller share.
- 16n Qwen (capsule `20260917-073515_perlmutter_ab54ff9f`, 6/6 ok); lane
  DONE + released 00:37.

  | arm | b | plan_comm | plan | l0 | l1 | e2e | total |
  |---|---|---|---|---|---|---|---|
  | COMET | 1 | 0.68 | 0.27 | 3.42 | 7.09 | 10.52 | 11.52 |
  | **COMET+EPLB** | 1 | 0.67 | 0.28 | 3.45 | 6.25 | 9.63 | 10.44 |
  | COMET | 4 | 0.96 | 0.36 | 7.58 | 9.18 | 16.76 | 18.03 |
  | **COMET+EPLB** | 4 | 0.83 | 0.36 | 7.78 | 8.84 | 16.76 | 17.84 |
  | COMET | 16 | 1.21 | 0.72 | 29.74 | 29.86 | 59.44 | 61.02 |
  | **COMET+EPLB** | 16 | 1.19 | 0.76 | 28.44 | 27.32 | 55.89 | 57.55 |

  total: −9% b1, −1% b4, −6% b16 (imbalance 3.36 → 1.33 at b16; pool
  prediction 1.07 — the Qwen 16n pool is the least predictable cell).

### 5d. Main-perf dataset summary (36/36 cells ok, 6 capsules, one binary)

COMET+EPLB total vs COMET, same capsule:

| | 1 MiB | 4 MiB | 16 MiB |
|---|---|---|---|
| 4n K2 | +1% | −7% | −12% |
| 4n Qwen | −12% | −16% | −13% |
| 8n K2 | +1% | −6% | −10% |
| 8n Qwen | −11% | −8% | −6% |
| 16n K2 | +2% | −6% | −6% |
| 16n Qwen | −9% | −1% | −6% |

Capsules: 071106 / 071304 (4n K2/Qwen), 071215 / 071417 (8n), 073316 /
073515 (16n). The private dataset for the eventual figure row is the
`l01_allgather_dense_eplb` cells of these six capsules; the in-capsule
`l01_allgather_dense` cells are the same-binary COMET anchor (the
committed figure_src COMET row is a different binary — never mix).

### 5e. COMET drift check (2026-09-17 07:52, capsule `20260917-145225_perlmutter_a063b0b1`)

Question: the official main-perf COMET row (Aug-24 libs `3649668a`) is 4-47%
slower than the same arm in the 09-17 capsules (Sep-10 libs `a0c60c75`,
the SAME libs as the plotted Sep-15 Ours rows). Binary or the Sep-14
compute image? Same-allocation 4n K2 re-run of COMET + the plotted Ours
arm (`ours_l01_s1_pv2_r2_pv3c_eps025`), ms:

| 4n K2 | COMET 08-24 (figure) | COMET 09-17 00:11 | COMET 09-17 07:52 | Ours 09-15 (figure) | Ours 09-17 07:52 |
|---|---|---|---|---|---|
| 1 MiB | 4.04 | 3.65 | 3.62 | 4.03 | 3.81 |
| 4 MiB | 6.26 | 5.94 | 5.93 | 6.11 | 5.92 |

- COMET is reproducible day to day on the Sep-10 libs (3.65 vs 3.62).
- Ours also moved on the new image, by ~5% (4.03 -> 3.81) — so the site
  image accounts for roughly that much; the rest of the COMET drift
  (10% at 4n b1; up to 47% at 16n b1, untested here) is the binary.
- **Fair same-day, same-binary ratio at 4n K2: Ours/COMET 0.95x (1 MiB),
  1.00x (4 MiB); Ours/COMET+EPLB (09-17 00:11 values) 0.97x / 0.94x.**
  The K2 small-budget groups where Ours is not the minimum in
  main_perf_v3 (4n 1+4 MiB, 8n 1 MiB) are therefore genuine, not a
  measurement artifact. Large budgets and Qwen are unaffected in sign.

### 5f. Plan-bracket share of the plotted Ours rows (2026-09-17, existing data)

The 18 plotted Ours cells are s1 arms (`ours_l01_s1_pv2_r2_pv3c_*`, dwire
at 16n K2 b1, slipstream at 8n Qwen b1): placement is solved ONCE at setup
(untimed); the per-iteration brackets are plan_comm (gating/probs
allgather, 0.15-0.8 ms) + plan (pv3c router + derive_routed/combine meta +
plan graphs, 0.6-2.8 ms); place_ms = 0. The s2 swap arm (re-solve + swap
plan every iteration, place_ms ~1 ms) is never the winner. Plan share of
Ours total: 7-27% (largest at 1 MiB, 16n). Ours e2e-only vs the baselines'
TOTAL: >= 1.03x vs COMET+EPLB in every group (worst 8n K2 b1 1.03x, 4n K2
b4 1.04x); e2e-vs-e2e (both sides stripped) still 0.95x at 4n K2 b4,
1.01x / 1.03x at 4n K2 b1 / 8n K2 b1. SCHEMA rule 5 caveat: planning is
timed by design (one-shot inference); stripping it from the report needs
the same treatment on every arm.

### 5g. Why Ours loses to COMET+EPLB at K2 1/4 MiB (2026-09-17, existing data)

Brackets (ms, iter-max median), Ours fused s1 pv3c vs COMET+EPLB:

| group | Ours plan_comm+plan / l0 / l1 | COMET+EPLB plan_comm+plan / l0 / l1 | CE gemm imb |
|---|---|---|---|
| 4n K2 b1 | 0.75 / 1.67 / 1.62 | 0.39 / 1.66 / 1.72 | 1.16 |
| 4n K2 b4 | 0.77 / 2.46 / 2.85 | 0.45 / 2.49 / 2.64 | 1.10 |
| 8n K2 b1 | 1.10 / 2.69 / 2.16 | 0.55 / 2.20 / 2.19 | 1.19 |
| 4n Qwen b1 (win) | 0.77 / 1.19 / 1.11 | 0.41 / 1.12 / 2.26 | 1.21 |

1. The whole 4n K2 b1 gap (0.35 ms) is the plan bracket: e2e ties
   (3.27 vs 3.30). Ours' per-iteration metadata (derive_routed_meta +
   combine meta + graph replay + probs packaging) costs 0.6-0.8 ms vs
   COMET's 0.26; the pv3c router itself is only 0.05-0.2 of it. Placement
   solve is 0 (setup). Fusing the meta derivation (as done for the EPLB
   router) is worth ~0.3-0.5 ms = 7-12% at 1 MiB.
2. 8n K2 b1: dispatch (2.69 vs 2.20) — the fused union wire has more
   serialized put/signal stages (intra-node dedup -> union -> gateway
   forward, ~120 us CXI floor each) than one dense allgather; at 72
   tok/GPU x 16-32 ranks the dense allgather is latency-optimal. A
   byte-gated dense-dispatch fallback at <= 1 MiB would take this back.
3. 4n K2 b4: combine (2.85 vs 2.64) — CE's balanced rows (imb 1.10) cut
   COMET's own combine from 3.00 to 2.64; Ours' msplit combine at 14 KB
   rows / 296 tok/GPU is wave-latency bound. Ours cells do NOT record
   gemm_rows_per_rank (the pv3c C=1/4 cap allows up to 1.25x) — add the
   fact to the ours driver before concluding on balance.
4. Shape, not scale: on Qwen (8 KB rows, 2x the tokens per MiB) COMET's
   dense combine is 2x slower per byte (2.26/3.58 ms) and Ours wins l1 by
   1.1-1.7 ms; on K2 the dense combine is cheap (1.72) so Ours' wire
   advantage does not appear until bandwidth-bound (>= 16 MiB).

### 5h. Inside the Ours plan bracket (NVTX probe 2026-09-17, capsule
`20260917-152059_perlmutter_2ce12c36`, spec `ours_planprobe_4n_k2_nsys`,
FLUX_OURS_NVTX=1, instrumented — breakdown only)

Host-side range medians per iteration, 4n K2, plotted arm
`ours_l01_s1_pv2_r2_pv3c_eps025` (node 0, 4 ranks):

| stage | b1 | b4 | in bracket |
|---|---|---|---|
| loads allgather d[R,G] | 0.16 (plan_comm) | 0.14 | plan_comm |
| plan.route_pv3 (kernel + kstats) | 0.08 | 0.08 | plan |
| plan.xchg_pack (phys+probs pack) | 0.07 | 0.07 | plan |
| plan.xchg_allgather (routing+probs) | 0.11 | 0.12 | plan |
| plan.vce_tail | 0.05 | 0.05 | plan |
| plan.derive_routed_meta (C++ fused, pinned-D2H sync) | 0.31 | 0.41 | plan |
| plan.m_this_host (D2H row count for sizing) | 0.05 | 0.05 | plan |
| plan.combine_meta_op + scale_build | 0.36 + 0.05 | 0.37 + 0.05 | side stream under l0 (plan_overlap 1) |

Sum of the in-bracket stages 0.67 / 0.78 ms == the event-timed plan_comm+
plan (0.75 / 0.77). Reading: the derive itself IS the fused C++ path and
costs what COMET's does (0.31 vs 0.25; +0.1 at b4 for the compress
unique counts). The extra ~0.4 ms is the global-demand router's chain —
loads allgather, route kernel, pack, vce, sizing D2H — five small
serialized steps each ending in a launch gap or host sync. Fusing
route+pack+vce into one kernel that emits the packed exchange buffer
saves ~0.2 ms; the loads allgather (0.16) is inherent to a demand-aware
router (EPLB's rule is sender-local). Best case ~0.3 ms: 4n K2 b1 4.03 ->
~3.7 = tie with COMET+EPLB (3.68), not a win; the 8n b1 dispatch gap and
the b4 combine gap (§5g) remain.

## 6. State at hand-back (2026-09-16 23:56)

- Arm complete and gated: 2n K2, 4n K2, 4n Qwen, 18/18 cells ok with the
  torch reference, random payload per iteration, one binary (the main
  tree's libs via the pv3 symlinks). Capsules 064442 / 064929 / 065339 are
  valid; **063449 and 063606 are the NVSHMEM-init launch failures** (all
  cells `failed`, no numbers) — safe to delete before committing.
- Everything is UNCOMMITTED on the pv3 worktree (user commits): 4 modified
  files (driver, sweep.py, variants.py, SCHEMA.md), new comet_eplb.py, the
  CPU test, three specs, this handoff, the three good capsules.
- Not done by design: no figure row / figure_src change (user: private).
  Suggested `row_id comet_eplb`, label "COMET+EPLB", when the user rules.
- Open decision: the ~2 ms per-iteration router (§5, 2n reading). Options
  (a) quote as is — it is the eplb arm's own sender-local rule, same
  code, same accounting class as `eplb_l01`'s plan_ms; (b) fuse
  derive_fused into one kernel (~0.25 ms class) before any figure use.
  Either way `e2e_ms` vs `total_ms` tell the two stories and both are in
  the capsules.
