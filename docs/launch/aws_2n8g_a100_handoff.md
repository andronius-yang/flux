> **SUPERSEDED (2026-08-04) — HISTORICAL ONLY.**
> This document was written *before* any of the work ran on hardware, and says
> so in several places ("never run on hardware"). Everything it anticipates has
> since happened: 124 sweep capsules, the sweep runner replacing its manual test
> ladder, and the branches it maps have all merged into `main`.
> **For the current handoff — including the AWS→Perlmutter migration — read
> `docs/handoff/00_START_HERE.md`.** Its §4 "Environment: AWS 2x8 vs Perlmutter"
> in particular is stale and should not be used for the migration.
> Kept for provenance: its §2 design points and §3 rejected-alternatives are
> still a useful record of what was under test and why.

# Handoff: Comet a2av work on an AWS 2-node 8x A100 cluster

Audience: bringing up this repository (branches `main`, `hier-compress`,
`hier-relay-balance`) on 2 nodes x 8 A100 (e.g. p4d.24xlarge) after developing
on NERSC Perlmutter (4x A100/node, Slurm, Slingshot). Covers what is on each
branch, the design decisions and their alternatives, what changes off
Perlmutter, and the exact validation order.

## 1. Branch map and validation status

| Branch | Adds | Hardware status |
|---|---|---|
| `main` | dense + a2av/a2av_ring/`a2av_hier` layer0, FAST baseline, layer1 multi-node port | validated on Perlmutter 1n + 4n (runbook `comet_traffic_matrix_tests.md`) |
| `hier-compress` | `a2av_hier_compress` — token-dedup wire semantics (design §11) | **never run on hardware**; index math validated by pure-python simulation |
| `hier-relay-balance` | balanced inter-node relay for compress (design §12) + `test_relay_balance_math.py` | **never run on hardware**; 156-case CPU simulation passes |

Design narrative: `comet_traffic_matrix_a2av.md` (§10 hier, §11 compress,
§12 balanced relay). Runbook with all knobs: `comet_traffic_matrix_tests.md`.

## 2. Key design points (what you are testing)

- **`a2av_hier` (§10)**: inter-node traffic travels as ONE aggregate per
  (rank, remote node) to the same-local-rank "gateway", which forwards each
  local destination's sub-chunk over NVLink. All puts are **host-issued NVSHMEM
  `putmem_signal` on streams** — deliberate, because the deploy targets
  (Slingshot, and equally EFA on AWS) are host-proxied transports with no
  IBGDA. GEMM tiles spin on per-source epoch signals
  (`signal_ptr[s] >= run_id`, system-scope acquire); signals are epoch-valued,
  init-zero, single-writer-per-slot, **never reset** — `clear_buffers()`
  deliberately does not touch them.
- **`a2av_hier_compress` (§11)**: the traffic matrix stays LOGICAL; the wire
  carries each token at most once per destination rank (intra-node) and once
  per destination node (union aggregate). Extra replicated metadata
  `a2av_unique_counts` (u[s][d], U[s][n]) drives all wire offsets on the host;
  receiver-side duplication is free (GEMM `gather_A` aliases recv rows).
  Gateways `index_select` exact subsets (needs SMs -> `sm_margin >= 1`
  FLUX_CHECKed multi-node) into scratch, forwarded with **non-nbi** puts
  (scratch is refilled per round; nbi has no local-completion guarantee).
- **Balanced relay (§12, default on `hier-relay-balance`)**: per round the
  node's L union segments form one canonical stream cut into L near-equal
  chunks (`chunk_bound()`, the single source of truth); relay rank k stages
  chunk k (boundary pieces pushed intra-node) and wire-puts it to the same-lr
  gateway. Round wire pace: `max_lr U[s][tn]` -> `ceil(total/L)`. Critical
  invariants:
  - *Deadlock rule*: every rank issues ALL rounds' piece puts before its first
    wire wait (`CUStreamWaitValue64` on relay-in slots). Never interleave.
  - *Token-level slice addressing*: a window cut inside a source's segment
    needs `cnt_in`/`cnt_before` from the device fwd-index build — one tiny
    pinned D2H, host-synced AFTER wire issue, right before the gateway loop.
  - *Signal aggregation*: destinations collapse L per-(round, gateway) slots
    into the per-source signals on a third stream (`CUStreamWriteValue64`,
    zero SMs) so the GEMM kernel is untouched.
  - GEMM launch gates on `relay_send_event_` (pieces issued), NOT
    `fetch_remote_event` (now contains cross-rank waits).

## 3. Alternatives (rejected / deferred — do not re-litigate blindly)

- **Device-addressed forwards** (`nvshmem_ptr` + ATen `index_copy_` into peer
  recv): rejected for now. Every in-repo peer-write-then-flag fences inside a
  kernel (`topk_gather_rs_v2.cu:669`); ATen kernels cannot, and NVSHMEM's
  signal-after-payload guarantee covers only its own puts. The sync-free
  variant also costs ~L x NVLink write amplification (fixed shapes, garbage
  rows). Deferred "option C" = same but exact-shaped via the D2H counts —
  worth an on-hardware visibility experiment on AWS (NVSwitch P2P), not a
  default.
- **Union broadcast** (`a2av_hier_bcast` candidate, §11 doc): gateway forwards
  the whole union to every peer — deletes the gather/scratch/sm_margin
  machinery, costs `(L-1)*U` intra-node bytes and `U`-sized recv regions.
  With L=8 the broadcast factor is worse than on Perlmutter's L=4; measure
  before considering.
- **Receiver pull** (peers `index_select` from gateway staging via
  `nvshmem_ptr`): exact bytes, one hop, parallel across receivers; needs
  arrival republication + per-receiver indices. Deferred.
- **Chunking granularity**: contiguous canonical chunking was chosen over
  excess-only workstealing because it is identity when U is uniform, minimal
  drift otherwise, and keeps every relay's wire put contiguous.

## 4. Environment: AWS 2x8 A100 vs Perlmutter 4x A100

Topology consequences first: `W=16, L=8, NN=2`. There is **exactly one
inter-node round per direction** (`dn = 1`), so the relay's entire effect is
balancing that single round's 8 sends; expect wins only on matrices whose
per-rank node-union bytes are skewed (the harness's
`a2av relay balance: identity X -> balanced Y` line predicts the bound).
Round-0 (intra-node, 7 peers over NVSwitch) is unchanged.

Setup differences:

- **`module.sh` is untracked and Perlmutter-specific — it will NOT be in your
  clone.** Recreate: CUDA >= 12.4 toolkit, gcc 12.x, a conda/venv with torch
  matching the CUDA, and NVSHMEM >= 3.x built with libfabric support
  (`NVSHMEM_LIBFABRIC_SUPPORT=1`) for EFA. Export `NVSHMEM_HOME` to that
  install — `setup.py`/`cpp_mod.py` give the env var strict precedence over
  pip-installed nvshmem. `export CUDA_HOME`, `TORCH_CUDA_ARCH_LIST=8.0`,
  `CC/CXX/CUDAHOSTCXX`. Submodules: `git submodule update --init --recursive`
  (NCCL + CUTLASS 4.0 under `3rdparty/`).
- **Build** (A100 everywhere, same as Perlmutter):
  `nproc=32 ./build.sh --arch 80 --sm-cores 108 --nvshmem --no_test --jobs 32`
  then `FLUX_BUILD_SKIP_CMAKE=1` for incremental rebuilds.
- **`launch.sh` assumes Slurm multi-node** (`SLURM_NNODES`, `scontrol show
  hostnames`) and Slingshot (`NVSHMEM_LIBFABRIC_PROVIDER=cxi`). On AWS:
  - Under ParallelCluster Slurm: works as-is, but export
    `NVSHMEM_LIBFABRIC_PROVIDER=efa` (and usually `FI_PROVIDER=efa`).
  - Without Slurm, run torchrun directly on each node with launch.sh's env:

    ```bash
    export NVSHMEM_BOOTSTRAP=UID NVSHMEM_DISABLE_CUDA_VMM=1 \
           CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_MODULE_LOADING=LAZY \
           NVSHMEM_REMOTE_TRANSPORT=libfabric NVSHMEM_LIBFABRIC_PROVIDER=efa \
           NVSHMEM_IB_ENABLE_IBGDA=0 NVSHMEM_SYMMETRIC_SIZE=4G
    torchrun --nnodes=2 --node_rank=<0|1> --nproc_per_node=8 \
      --rdzv_endpoint=<node0-ip>:23456 <test.py> <args...>
    ```
  - The whole design is host-issued push, so EFA's host-proxied NVSHMEM
    transport is exactly what it was built for. Intra-node `nvshmem_ptr` P2P
    (used by flux_shm barriers and the §12 deferred options) works over
    NVSwitch.
- **Traffic matrices are NOT in the repo** (Perlmutter kept them on
  `$PSCRATCH`). Format (see `python/flux/testing/traffic_matrix.py`): line 1 =
  W, then W x W byte counts, entries multiples of `chunk_bytes` (8192 = one
  4096-dim bf16 token), equal row sums, zero diagonal. Generator for uniform
  and hot-column matrices:

  ```python
  import sys, torch
  W, budget_mib, hot = int(sys.argv[1]), int(sys.argv[2]), len(sys.argv) > 3
  chunks_per_row = budget_mib * 1024 * 1024 // 8192
  m = torch.full((W, W), chunks_per_row // (W - 1), dtype=torch.long)
  m.fill_diagonal_(0)
  if hot:  # ~3x hot columns like the real a2av sets (keep row sums equal)
      cols = torch.randperm(W)[: max(W // 4, 1)]
      for r in range(W):
          take = [c for c in range(W) if c != r and c not in cols]
          for c in cols:
              if c == r: continue
              for t in take: m[r, c] += m[r, t] // 2; m[r, t] -= m[r, t] // 2
  rem = chunks_per_row - m.sum(1)  # top up row sums exactly
  for r in range(W):
      c = 0 if r != 0 else 1
      m[r, c] += rem[r]
  print(W); print("\n".join(" ".join(str(v * 8192) for v in row) for row in m.tolist()))
  ```

  Sanity-check any matrix without GPUs:
  `python3 python/flux/testing/traffic_matrix.py <matrix.txt> [G] [topk]`.
- Matrices must match world size: 16-rank files for 2x8, 8-rank files for the
  single-node runs. Token-count constraints (multi-node MoE): tokens divisible
  by `world_size * topk`; `max_m/topk` divisible by `world_size`.

## 5. Test procedure (in order; stop at the first failure)

Each numbered stage gates the next. All a2av tests self-verify against a torch
reference (allclose) every run.

1. **No GPU needed, immediately after clone** (branch `hier-relay-balance`):
   `python3 test/python/moe_ag_scatter/test_relay_balance_math.py` (156 cases)
   and `python3 test/python/moe_ag_scatter/test_fast_index_math.py`.
2. **Build sanity, 1 GPU**:
   `python3 test/python/gemm_only/test_gemm_only.py 4096 12288 6144 --dtype=float16`.
3. **Single node, 8 GPUs** (`./launch.sh`, no Slurm needed):
   - dense: `./launch.sh test/python/ag_gemm/test_ag_kernel.py 4096 49152 12288 --dtype=float16 --iters=10`
   - MoE layer0/layer1: `./launch.sh test/python/moe_ag_scatter/test_moe_ag.py`
     and `./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py -T 8 -E 1`.
   - traffic, all patterns, 8-rank matrix:
     `--comm_pattern` in {`a2av`, `a2av_ring`, `a2av_hier`, `a2av_hier_compress`}
     with `FLUX_A2AV_CHECK_COMPRESS=1 FLUX_A2AV_CHECK_IDENTITY=1`. NN=1
     degenerates compress to intra-node dedup puts and the relay is inactive —
     this run must behave identically on `hier-compress` and
     `hier-relay-balance`.
4. **Two nodes, 16 ranks, correctness** (16-rank matrices, uniform first, then
   hot-column):
   - `a2av_hier` on `main`/`hier-compress` — the known-good baseline.
   - compress identity: `FLUX_A2AV_RELAY_IDENTITY=1` on **every rank** +
     `FLUX_A2AV_CHECK_COMPRESS=1 --sm_margin 8` — first-ever hardware run of
     §11; debug any failure here before touching the relay.
   - compress relay (default env) + `FLUX_A2AV_CHECK_COMPRESS=1 --sm_margin 8`
     — first hardware run of §12. The CHECK asserts pack/gateway flag counts
     against u/U and the window split tiling; a wire-offset bug fails loudly
     here rather than as a silent allclose miss.
   - layer1 multi-node: `test_moe_gather_rs.py -M 40960 -T 16 -E 1`.
5. **Two nodes, timing** (`--iters 10 --warmup_iters 5`, CHECK envs OFF):
   - A/B identity vs relay on the same skewed matrices; the harness's
     `a2av relay balance` line is the upper bound of the wire-pace win —
     uniform matrices should show ~0 delta (relay takes the own-segment fast
     path), hot-column matrices should approach the printed ratio on the
     inter-node segment (`FLUX_A2AV_TIMING=1` splits the phases).
   - Sweep budgets (2/16/64 MiB rows) as in the runbook's Perlmutter tables
     for comparability.
6. **Knobs when things fail loudly**: `FLUX_A2AV_MAX_RECV_NTOKENS` /
   `FLUX_A2AV_MAX_STAGE_NTOKENS` / `FLUX_A2AV_MAX_RELAY_NTOKENS` (all checked
   collectively — a capacity failure aborts every rank, no hang);
   `NVSHMEM_SYMMETRIC_SIZE=4G` for big budgets; `--sm_margin >= 1` is
   FLUX_CHECKed for multi-node compress. `FLUX_A2AV_RELAY_IDENTITY` changes
   the wire layout — mismatched ranks corrupt silently, set it everywhere or
   nowhere.

## 6. Merge/PR state

`hier-relay-balance` (1 commit) sits on `hier-compress` (2 commits over
`main`). Merge order: `hier-compress` -> `main` after stage 4's identity run
passes; `hier-relay-balance` after the A/B. Nothing here touches the GEMM
kernels, pybind signatures, or `a2av_hier`, so a byte-identical fallback
(`a2av_hier`, or compress + `FLUX_A2AV_RELAY_IDENTITY=1`) exists at every
step.
