# Handoff 35 — H100 port on CSCS ALPS: the A2AV+GEMM weak-scaling figure, 2n → 32n (2026-09-06)

**Who this is for.** A fresh Claude Code session on the CSCS ALPS cluster (GH200 nodes:
NVIDIA Grace **aarch64** CPUs + **H100** GPUs, Slingshot-11 network) that has this
repository cloned, **no memory files**, and this document. Written on NERSC Perlmutter
(A100) by the session that owns the Perlmutter weak-scaling figure. Nothing in this
document was executed on an H100 — every H100-specific step is a **prediction that the
gate ladder in §6 must confirm**. Stop at the first red gate and root-cause it there.

**Read order before touching anything:** this file end to end → `CLAUDE.md` (the five
invariants) → `sweeps/SCHEMA.md` §"Protocol rules" → `figs/weak_scaling/SPEC.md`
§1/§4b/§4c → `.claude/skills/sweep/SKILL.md`. Then run §3 (discovery) and report back to
the operator before building (§2.3 lists what only they can tell you).

---

## 1. Goal, scope, deliverable — nothing more, nothing less

**Goal.** Prove that the OURS placement + routing + comm/compute-overlap stack scales the
same way on a newer architecture as it does on A100: reproduce the paper's
**A2AV+GEMM weak-scaling figure on H100**, node counts **2, 4, 8, 16, 32**, exactly.

**The figure** (`figs/weak_scaling/SPEC.md` REV 3.2, ver4): single column, two stacked
panels (**1 MiB** on top, **64 MiB** below, per-rank pre-topk send budget), x = nodes
(2..32, log2 spacing), lines = latency of **Ours** vs the **A2AV+GEMM ring**
(`l01_nvshmem`), bars = speedup of Ours over the ring. "Ours" at each node count is the
**minimum over two arms**: `ours_l01_s1_pv2_r2` (s1: pv2 placement + LocCap routing + r2
slack parity + Slipstream comm/comp overlap) and `ours_l01_s1_pv2_r2_dwire` (same
placement/routing over a direct staged a2av wire; it wins at 1 MiB from 16n on A100).
Rendered by `figs/weak_scaling/make_figure.py --baseline nvshmem --stacked` (+ the
single-budget `--baseline nvshmem` verA/verB renders from the same rows).

**Exact data to collect** (30 cells, all `isolated` mode, K2 shape, captured K2 routing):

| axis | values |
|---|---|
| nodes | 2, 4, 8, 16, 32 (4 GPUs/node → W = 8, 16, 32, 64, 128 ranks) |
| budget (per-rank pre-topk, MiB) | 1, 64 (`tokens_per_rank` 72 and 4,680) |
| arms (one capsule per node count, in this order) | `ours_l01_s1_pv2_r2_dwire`, `ours_l01_s1_pv2_r2`, `l01_nvshmem` |
| mode / iters | `isolated`, 10 timed + 5 warmup, `skip_correctness: true` (perf lane) |
| model shape (`shape: k2`) | G=384 experts, topk 8, H 7168, ffn 2048, bf16, chunk 14,336 B |
| routing family | `trace:model=Kimi-K2;pools=livecodebench/execution;layer=5;sem=homog;dslots=64:32` |

Plus two **gate** capsules whose latencies are never quoted (correctness ON): the 2n gate
(all three arms) and the 32n smoke (s1 only). Specs for everything are pre-written:
`sweeps/specs/h100_weak_gate_2n_k2.yaml`, `h100_weak_{2,4,8,16,32}n_k2.yaml`,
`h100_weak32_smoke_k2.yaml`.

**Deliverable** (all inside the clone, §7): the 7 capsules under `sweeps/results/runs/`,
`figs/weak_scaling/h100/figure_src.csv` (+ `figure_src.md`), the renders
`figs/weak_scaling/h100/weak_scaling_nvshmem_stacked.{pdf,png}` (+ verA/verB),
`figs/weak_scaling/h100/SESSION_LOG.md`, committed and pushed on the **`h100-weak-scaling`**
branch you were cloned onto (it is the branch that carries the routing data, §4.1).

**Out of scope — do not run unless the operator explicitly asks:** COMET
(`l01_allgather_dense`), the other budgets (2/4/8/16/32 MiB), Qwen/K3 shapes, e2e/phases/
nsys modes, any optimization or tuning of kernels, any new arm. If the user later wants
COMET, it is one extra variant line in the specs (H100-96GB may even fit COMET's 32n
64 MiB cell that OOM'd on A100-40GB, `figs/weak_scaling/figure_src.md`).

---

## 2. Precautions — read before the first command

### 2.1 The five things that are different on ALPS

1. **aarch64.** Nothing built on Perlmutter/AWS is reusable: no conda env, no wheels, no
   `.so`. Every dependency (torch with CUDA, NVSHMEM, the bundled NCCL, Flux) must be
   built or installed for `linux_aarch64`. Check `uname -m` first.
2. **sm90.** The whole OURS/a2av implementation lives in the **sm80/V2 ops**
   (`GemmGroupedV2AGScatterOp`, `GemmGroupedV2GatherRSOp`, `TopkReduceScatterOp`,
   `GemmGroupedV2`, `GemmOnly`). The kernel generator emits V2 kernels only for the Sm80/
   Sm89 registry keys and the Hopper V3 ops (which have none of our work) for Sm90. The
   port therefore does **not** use `--arch 90` alone. It compiles the **Sm80-tagged kernel
   space natively for `sm_90a`** and tells the runtime to look kernels up under the Sm80
   key — §5 has the exact build line; the hooks are already in the tree:
   - `CMakeLists.txt` `GEN_CUDAARCHS` (generator arch) decoupled from `CUDAARCHS` (nvcc
     gencode); `build.sh --gen-arch`.
   - `src/generator/gen_*.cc`: the Sm80 spaces now also carry the 132-SM key (`_H800{}`),
     so `--sm-cores 132` generates them.
   - `src/cuda/op_registry.cu`: `FLUX_ARCH_OVERRIDE=80` (and `FLUX_SM_CORE_OVERRIDE`,
     normally unneeded) — env-gated, stock behaviour when unset.
   - `python/flux/util.py get_arch()` honours the same variable, so the drivers' V2/V3
     selection and their `assert get_arch() < 90` rule-5 guards follow.
   `FLUX_ARCH_OVERRIDE=80` is exported by `env_alps.sh` AND recorded per cell by
   `sweeps/platforms/alps.yaml env:`. Never run a cell without it — the V3 ops will be
   picked and the arms will not even construct.
3. **Same network family as Perlmutter.** Slingshot-11 = libfabric **CXI** provider,
   i.e. the NVSHMEM transport path and *every wire-ordering lesson* (CLAUDE.md invariant 5,
   SCHEMA rule 6) carry over unchanged: the blocking wire is the default, correctness
   cells randomize the payload (runner does it), no nbi put_signal gating anywhere.
   `launch.sh` already selects `NVSHMEM_REMOTE_TRANSPORT=libfabric` +
   `NVSHMEM_LIBFABRIC_PROVIDER=cxi` when `SLURM_NNODES>1`.
4. **NVSHMEM must come with the libfabric transport.** Perlmutter's module was built by
   NERSC with it. On ALPS you must find/obtain an NVSHMEM 3.x whose `lib/` contains
   `nvshmem_transport_libfabric.so.3` (options in §4.2). Without it multi-node is
   impossible; single-node runs would still "work" and mislead you.
5. **torch.distributed's NCCL** needs the Slingshot plugin (aws-ofi-nccl) or it falls back
   to TCP sockets. The per-iteration topk all-gather is *timed* (`plan_comm_ms`, rule 5),
   so a sockets fallback inflates every arm's total. Verify once with `NCCL_DEBUG=INFO`
   on the 2n gate ("NET/OFI" / "Libfabric" in the log).

### 2.1b What the port does NOT do (say this in the caption; user check 2026-09-06)

The port adds **no optimization and no H100-specific code path**. Every kernel that runs
is the A100 design (CUTLASS-2.x `mma.sync`/`cp.async` grouped GEMM with the fused
scatter epilogue, plus the plain-CUDA a2av pack/wire/reduce kernels), recompiled for
`sm_90a`; the Hopper-native V3 machinery (wgmma/TMA, warp-specialized) is not used
because none of our work lives there. Consequences to state honestly:
- The GEMM leaves H100 throughput on the table relative to an H100-native kernel. It is
  the SAME GEMM in all three arms, so the arm-vs-arm comparison stays fair; absolute
  latencies are "A100-style kernels on H100", not "H100 peak".
- Kernel hyper-parameters are identical to the A100 runs: the only A100 tuning records
  in the tree are two int8 `GemmOnly` shapes (`src/comm_none/tuning_config/
  config_gemm_only_sm80_A100.cu`), so every bf16 GEMM of this figure already used the
  registry's first-registered fallback on A100 and does the same under the 132-SM key
  (same hparams space, same order). No runtime code branches on the SM-core key.
- The knobs are absolute counts left as they were (`sm_margin 8`, RS pack/reduce/
  pre-reduce blocks 10/8/6, 16 wire streams, `CUDA_DEVICE_MAX_CONNECTIONS`): on 132 SMs
  the GEMM simply gets the extra 24 SMs. Not retuned, not an optimization; do not tune.
- Whether anything is *slower* than it could be on H100 is a measurement, not a
  guarantee; the figure's claim is the scaling shape under identical code.

### 2.2 Rules inherited from the project (non-negotiable)

- CLAUDE.md invariants 1–5 and SCHEMA protocol rules 1–6. In particular: **`deterministic`
  must be 0** in every perf cell; **quote `isolated` mode only**; **one binary for the
  whole figure** — build once, `sha256sum python/flux/lib/libflux_cuda_ths_op.so`, write
  it in the session log, never rebuild between rungs (if you must, redo every rung).
- **Timing accounting** (SCHEMA rule 5): nothing to do — the three arms are already
  rule-5 drivers; just don't add caching.
- **Never `pkill -f` from a tool call** (it kills your own shell). Kill by pid from
  `pgrep`, or `scancel -s KILL <step>`.
- **Watch budgets:** a deadlock-prone cell gets ~60 s per expected iteration of patience,
  not 10 minutes; the runner's `timeout_s`/`idle_timeout_s` in the specs are sized for
  that — do not raise them to "wait longer".
- **Status cadence:** append to `figs/weak_scaling/h100/SESSION_LOG.md` at every event
  (build start/finish, each gate verdict, each capsule, each incident), not at the end.
- **Take job ids only from your own `salloc`/`sbatch` output**, release allocations the
  moment their capsules are written (`squeue --me`, `scancel <id>`), never let 32 GH200
  nodes idle.
- **No filesystem-wide searches** (`find /`, `du /capstor`, …). Bound every search to the
  clone, `$SCRATCH/<you>`, or a path the operator gave you. Locate software with
  `command -v`, `module spider`, `uenv image find`, `pip show`.
- Do not edit `sweeps/variants.py` arm definitions, `sweeps/sweep.py` sizing formulas, or
  any kernel: the point is to measure the *same* code on H100. Allowed edits: `env_alps.sh`,
  `sweeps/platforms/alps.yaml`, the h100 folder, and build-system fixes needed to compile
  on aarch64/sm90 (record each in the session log with the reason).

### 2.3 What only the operator can tell you (ask before building)

Ask these as ONE batch at the start (the paste-in prompt already asks them to prepare
the answers): (1) vCluster/login host and the Slurm **account**; (2) **partition/QOS**
for 2–32 GPU nodes, max walltime, whether `salloc` (interactive) is allowed at 32 nodes
or `sbatch` is expected, any reservation; (3) where the clone lives (must be a filesystem
the compute nodes see — on ALPS `$HOME` and `$SCRATCH` both are; note `$SCRATCH` purge
policy); (4) how they expect software to be provided: **uenv** image name (+view),
modules, or a container — and whether an **NVSHMEM with libfabric** already exists on
the system (ask them to check with the CSCS docs/support; otherwise §4.2 builds it);
(5) the path where they placed the shipped data tarball (§4.1); (6) whether they have
push rights to the fork (else you hand back a `git bundle`).

---

## 3. Discovery checklist (run first, paste results into `SESSION_LOG.md`)

Login node (bounded, cheap):

```bash
uname -m; cat /etc/os-release | head -3; nproc
sinfo -s; sacctmgr show assoc user=$USER format=account,partition,qos -p
scontrol show partition | grep -E 'PartitionName|MaxNodes|MaxTime|DefaultTime'
command -v uenv && uenv image find          # CSCS user environments (e.g. pytorch/…, prgenv-gnu/…)
module avail 2>&1 | grep -i -E 'cuda|nvshmem|gcc|cray-mpich|libfabric|nccl' | head -30
command -v nvcc && nvcc --version | tail -1
command -v fi_info && fi_info -l           # expect a 'cxi' provider
python3 --version; cmake --version | head -1; ninja --version 2>/dev/null
```

One 1-node allocation (5 minutes; use the operator's account/partition):

```bash
salloc -A <account> -p <partition> -N 1 --gpus-per-node=4 -t 00:10:00 --no-shell   # or the site's form
srun --jobid=<id> -N 1 --ntasks-per-node=1 --gpus-per-node=4 bash -lc '
  nvidia-smi -L; nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
  uname -m; nproc; fi_info -p cxi 2>&1 | head -5
  python3 -c "import torch;p=torch.cuda.get_device_properties(0);print(p.name,p.multi_processor_count,p.total_memory>>30,torch.__version__,torch.version.cuda)" 2>&1 | tail -1'
scancel <id>
```

Must-haves before continuing (write each verdict in the log):

| check | required | why |
|---|---|---|
| `uname -m` | `aarch64` | §2.1 (1) |
| GPUs per node | **4** (`ranks_per_node: 4` in `alps.yaml`) | W = 4 × nodes is the figure's x-axis; if it is not 4, STOP and ask — the rows are not comparable |
| GPU | H100, compute capability 9.0, SM count **132**, memory (96 or 80 GB) | `--sm-cores 132`; `sym_size_max_g` 32 fits either |
| fabric | `cxi` provider present | NVSHMEM transport |
| torch | CUDA build for aarch64, `torch.version.cuda` = the nvcc you will use (major.minor) | cmake configure fails on a minor mismatch (Perlmutter 12.4-vs-12.9 lesson) |
| gcc | ≤ the CUDA toolkit's max (CUDA 12.4: gcc ≤ 13; newer toolkits allow newer) | nvcc host compiler |
| cmake ≥ 3.17, ninja, python ≥ 3.8 with numpy, matplotlib | build + runner + figure | the runner uses `argparse(required=True)`, so not python 3.6 |
| NVSHMEM 3.x prefix with `include/nvshmem.h`, `lib/libnvshmem_host.so.3`, `lib/nvshmem_bootstrap_uid.so.3`, **`lib/nvshmem_transport_libfabric.so.3`** | all four | §2.1 (4); `nvshmem-info -a` (in `bin/`) prints the build's transport support |

Reference stack that produced the A100 figure (for "what version" questions; match the
*shape* of it, not the exact numbers): Python 3.11, torch 2.6.0+cu124, CUDA 12.4,
gcc 12.2, cmake 3.28, NVSHMEM 3.2.5 (site build, libfabric/CXI), bundled NCCL 2.19.3
(static, built by `build.sh`), CUTLASS 3.8 (submodule), `--arch 80 --sm-cores 108`.

---

## 4. Repository, data, environment

### 4.1 Clone the data branch and unpack the routing data

The routing data travels **inside git, on the `h100-weak-scaling` branch only** (main
does not carry it): `figs/weak_scaling/h100/data/` holds the Kimi-K2
`livecodebench/execution` pool as an xz-compressed tar split into sub-50 MiB parts
(GitHub's per-file limit is 100 MiB; the pool is 1.53 GB raw, ~17× smaller under xz),
the 10 pre-generated matrix sets, and `SHA256SUMS.txt`. Clone, switch, unpack:

```bash
git clone <fork url> flux && cd flux
git checkout h100-weak-scaling                                 # the data branch; do ALL work on it
git submodule update --init 3rdparty/nccl 3rdparty/cutlass     # NOT --recursive: 3rdparty/FAST is a
                                                              # private SSH submodule, unused by this lane
cd figs/weak_scaling/h100
mkdir -p raw matrices traces/moonshotai/Kimi-K2-Thinking/livecodebench logs
(cd data && sha256sum -c SHA256SUMS.txt)                       # every part + tarball line OK
cat data/traces_k2_lcb_execution.tar.xz.part-* | xz -dc | tar -x -C traces/moonshotai/Kimi-K2-Thinking/livecodebench
tar -xJf data/matrices_k2_weak.tar.xz -C .                     # -> matrices/ (50 files)
ls traces/moonshotai/Kimi-K2-Thinking/livecodebench/execution | wc -l    # 483: 481 json + pool.manifest.json + pool_cache/
ls matrices | wc -l                                             # 50
cd -
```

(`data/README.md` repeats these commands. Streaming through `xz -dc | tar -x` needs no
1.5 GB intermediate file; if you must materialize it, do so under `h100/logs/`, never
outside the clone.)

Why both: the runner derives every matrix id from the pool's content fingerprint
(`pool.manifest.json` → `poolsha` folded into the id), so `traces/` is required even
when matrices exist; the pre-generated `matrices/` (with their `.routing.txt`,
`.oracle_load.json`, `.oracle_routing.txt` sidecars) make the ids bit-identical to the
A100 campaign and skip regeneration (W=128 generation is slow). **Identity check**
after the first `--dry-run` with the data in place, the cell ids/matrix ids must be
exactly these (a different hex = wrong pool bytes; stop):

| nodes | 1 MiB matrix id | 64 MiB matrix id |
|---|---|---|
| 2 | `w8x4_trace-96c62e_b1_k8_id001` | `w8x4_trace-97cf97_b64_k8_id001` |
| 4 | `w16x4_trace-b935ab_b1_k8_id001` | `w16x4_trace-23798c_b64_k8_id001` |
| 8 | `w32x4_trace-9b8ce5_b1_k8_id001` | `w32x4_trace-4d7ed6_b64_k8_id001` |
| 16 | `w64x4_trace-b8bf46_b1_k8_id001` | `w64x4_trace-96c63f_b64_k8_id001` |
| 32 | `w128x4_trace-3d45b3_b1_k8_id001` | `w128x4_trace-c4f8a4_b64_k8_id001` |

(If the data branch is unusable, the pool can be re-fetched with
`sweeps/fetch_traces.py --model Kimi-K2 --pool livecodebench/execution` — gated HF
dataset, needs a token that accepted the terms — and the runner regenerates the
matrices; then the ids above are the check that the fetched bytes match.)

### 4.2 Software: three ways to get NVSHMEM + torch on aarch64 (pick by what §3 found)

Order of preference:

1. **Site-provided.** A uenv/module that ships NVSHMEM built against the system
   libfabric (ask CSCS; the Cray PE sometimes carries `nvshmem`). Set `NVSHMEM_HOME` to
   its prefix. Torch from the site's PyTorch uenv (it also carries the NCCL/aws-ofi-nccl
   plugin — check `NCCL_NET_PLUGIN`/`LD_LIBRARY_PATH` inside its view).
2. **pip wheel** `nvidia-nvshmem-cu12` (aarch64 wheels exist for 3.3.x): install into the
   venv, `NVSHMEM_HOME="$(python -c 'import nvidia.nvshmem;print(nvidia.nvshmem.__path__[0])')"`,
   then **check `lib/nvshmem_transport_libfabric.so.3` exists**; it dlopens the system
   `libfabric.so.1`, so that must be on `LD_LIBRARY_PATH` (Cray PE: `/opt/cray/libfabric/…/lib64`).
   Torch: `pip install torch --index-url https://download.pytorch.org/whl/cu<XYZ>` for the
   CUDA minor that matches the nvcc you will build with.
3. **Build NVSHMEM from source** (github.com/NVIDIA/nvshmem, 3.3+) with
   `NVSHMEM_LIBFABRIC_SUPPORT=1 LIBFABRIC_HOME=<system libfabric prefix>`
   `NVSHMEM_IBRC_SUPPORT=0 NVSHMEM_UCX_SUPPORT=0 NVSHMEM_MPI_SUPPORT=0
   NVSHMEM_SHMEM_SUPPORT=0 CUDA_HOME=… ` (cmake), install under
   `figs/weak_scaling/h100/nvshmem/` (already gitignored), `NVSHMEM_HOME` = that
   prefix. Build on a compute node.

Whatever the route, the **venv** (if any) goes to `figs/weak_scaling/h100/venv/`
(gitignored), and `env_alps.sh` is filled in accordingly. Do not install anything into
the operator's `$HOME`/`$SCRATCH` outside the clone.

### 4.3 `env_alps.sh`

A template with `TODO(alps)` markers is in the repo root. Fill it; keep its checks
(nvcc == torch CUDA, NVSHMEM headers + libfabric transport present, aarch64). It exports
`FLUX_ROOT` (auto), `FLUX_ARCH_OVERRIDE=80`, `TORCH_CUDA_ARCH_LIST=9.0`, and the CXI
transport defaults. If the site uses uenv, the file must be sourced *inside* `uenv start`
and the allocation/`srun` must carry the same image (`--uenv=`); if that is needed, add
the flag to `srun_extra` in `sweeps/platforms/alps.yaml` and note it in the log.

### 4.4 `sweeps/platforms/alps.yaml`

Pre-written. `data_root`/`matrices_root`/`traces_root` all resolve through `FLUX_ROOT`
into `figs/weak_scaling/h100/` (the user's rule: never pollute the collaborator's
workspace; all three are gitignored there). Confirm `ranks_per_node: 4`,
`sym_size_max_g: 32`, and that `srun --jobid=<id> --nodes=N --ntasks-per-node=1
--gpus-per-node=4` is the right srun form on ALPS (§3 test). The runner refuses a spec
with unresolved `${FLUX_ROOT}`, so always `source ./env_alps.sh` first.

---

## 5. Build (compute node, never the login node if it is small)

```bash
source ./env_alps.sh
# CPU-heavy: allocate a node (a GH200 node has 288 Grace cores); inside the allocation:
srun --jobid=<id> -N 1 --ntasks-per-node=1 bash -lc 'source ./env_alps.sh && \
  ./build.sh --arch 90 --gen-arch 80 --sm-cores 132 --nvshmem --no_test --jobs 64' \
  2>&1 | tee figs/weak_scaling/h100/logs/build_$(date -u +%Y%m%dT%H%M%SZ).log
```

What the flags do: `--arch 90` → nvcc gencode `sm_90a`/`compute_90a` (CMake rewrites
sm_90→sm_90a) for every `.cu`, and the bundled NCCL for sm_90; `--gen-arch 80` → the
generators emit the **Sm80** kernel space (V2 ops: our code) and no V3 kernels (nothing of
ours is there, and it halves compile time); `--sm-cores 132` → the 132-SM registry key
(the tree now lists it in the Sm80 spaces). Expected cmake output lines:
`CUDA_ARCHITECTURES=90` and `GENERATOR_ARCHITECTURES=80`. The pip step installs the
package editable into the active venv.

Incremental rebuilds: `FLUX_BUILD_SKIP_CMAKE=1` plus the same flags. `./build.sh
--clean-all` if cmake was ever configured with different arch flags (stale cache).

**Predicted failure modes and what to do:**

| symptom | cause / fix |
|---|---|
| cmake: CUDA version vs torch mismatch, or `find_package(CUDAToolkit)` picks another toolkit | one toolkit on PATH/`CUDA_HOME`, minor == `torch.version.cuda` (env checks) |
| generator step emits 0 registers for an op, or "Unsupported SM count" | `--sm-cores 132` missing / typo; `grep -c register build/src/moe_ag_scatter/registers/*.cu` must be > 0 |
| nvcc errors inside CUTLASS 2.x templates for `sm_90a` | first check it is not a gcc/nvcc version issue (gcc too new for the toolkit). If a genuine sm_90a incompatibility remains, use the **fallback**: `--arch 80 --gen-arch 80 --sm-cores 132` (compute_80 PTX is JIT-compiled for sm_90 by the driver); then export `CUDA_CACHE_PATH=$FLUX_ROOT/figs/weak_scaling/h100/logs/cuda_cache` and `CUDA_CACHE_MAXSIZE=4294967296` in `env_alps.sh`, and warm the cache with the single-node test before any timed run. Record which path was used. |
| link error `libnvshmem_host` / transport not found at runtime | `NVSHMEM_HOME/lib` on `LD_LIBRARY_PATH`; `python/flux/cpp_mod.py` preloads `libnvshmem_host.so.3` + `nvshmem_bootstrap_uid.so.3` from `NVSHMEM_HOME` |
| `import flux` → "unsupported arch: 90" or "Unsupported SM count" | `FLUX_ARCH_OVERRIDE=80` not exported; SM count not 132 (then set `FLUX_SM_CORE_OVERRIDE=132` only if the device really has 132 and detection is odd — otherwise STOP, wrong GPU) |
| setup.py wheel name says `linux_x86_64` | cosmetic (only `--package`); ignore |

After the build: `sha256sum python/flux/lib/libflux_cuda.so python/flux/lib/libflux_cuda_ths_op.so`
→ session log. Capability tags the arms require must be in the binary:
`grep -c FLUX_A2AV_SLIPSTREAM2_TAG python/flux/lib/libflux_cuda_ths_op.so` (> 0), the
runner reports `skipped_capability` otherwise (= stale/wrong build, never "run anyway").

---

## 6. Gate ladder (in order; one red = stop, root-cause, log)

G0 **import + kernel registry** (1 node, 1 GPU):
```bash
srun --jobid=<id> -N1 --ntasks-per-node=1 --gpus-per-node=4 bash -lc 'source ./env_alps.sh && \
  python -c "import flux; from flux.util import get_arch; print(get_arch()); \
  print(flux.GemmGroupedV2AGScatterOp, flux.GemmGroupedV2GatherRSOp, flux.GemmOnly)"'
```
expect `80` and the three classes. Then the gemm-only smoke
`python3 test/python/gemm_only/test_gemm_only.py 4096 12288 6144 --dtype=float16`.

G1 **single node, 4 GPUs** (launch.sh, 1 node): `./launch.sh test/python/moe_ag_scatter/test_moe_ag.py`
and `./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py -T 4 -E 1` inside a
1-node `srun … bash -lc 'source ./env_alps.sh && ./launch.sh …'`. Bitwise/allclose
passes required. (With the JIT fallback, this run also warms the CUDA cache.)

G2 **two nodes** (the CXI transport):
```bash
srun --jobid=<id> --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 bash -lc 'source ./env_alps.sh && ./launch.sh test/python/moe_ag_scatter/test_moe_ag.py'
srun --jobid=<id> --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 bash -lc 'source ./env_alps.sh && ./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py -M 40960 -T 8 -E 1'
```
Run the first one once more with `NCCL_DEBUG=INFO` and confirm the NCCL net is the
libfabric plugin (§2.1 (5)); log the line. If NVSHMEM init fails: `nvshmem-info -a`
(transport support), `fi_info -p cxi`, `NVSHMEM_DEBUG=INFO`, and the bootstrap knob
`NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=<hsn interface>` if the UID bootstrap hangs. Known
Perlmutter-era classes that are NOT bugs: `CUDA_MODULE_LOADING=LAZY` first-launch
deadlocks behind spin kernels are handled by ctor preloads; the NVSHMEM proxy needs
symmetric-heap sources (already the case in every arm).

G3 **the 2n gate sweep** (correctness ON, both budgets, three arms):
```bash
source ./env_alps.sh
python sweeps/sweep.py run --spec sweeps/specs/h100_weak_gate_2n_k2.yaml --dry-run   # ids = §4.1 table
python sweeps/sweep.py run --spec sweeps/specs/h100_weak_gate_2n_k2.yaml --jobid <id>
```
Bar: every cell `ok`, `correct_allclose=1` (bitwise where the arm reports it),
`deterministic=0`, `routing_mode=real`. Commit the capsule (the runner prints the
command). Never quote its latencies.

G4 **the 32n smoke** — at the top of the 32-node allocation, before the 32n perf spec:
`h100_weak32_smoke_k2.yaml`. Same bar.

---

## 7. The campaign

One allocation per rung, gate-first, runner-driven, capsule committed before the
allocation is released. The runner needs a **RUNNING** allocation and finds it by
`squeue -u $USER -t RUNNING` when exactly one exists (else pass `--jobid`).

| rung | spec | allocation (estimate; A100 runs were 10–25 min per capsule) |
|---|---|---|
| 2n gate | `h100_weak_gate_2n_k2.yaml` | 2 nodes, 45 min |
| 2n | `h100_weak_2n_k2.yaml` | 2 nodes, 30 min (can share the gate's allocation) |
| 4n | `h100_weak_4n_k2.yaml` | 4 nodes, 30 min |
| 8n | `h100_weak_8n_k2.yaml` | 8 nodes, 30 min |
| 16n | `h100_weak_16n_k2.yaml` | 16 nodes, 40 min |
| 32n | `h100_weak32_smoke_k2.yaml` **then** `h100_weak_32n_k2.yaml` | 32 nodes, 75 min |

≈ 60 node-hours total plus bring-up. Interactive form (`salloc … --no-shell`, then
`python sweeps/sweep.py run --spec … --jobid <id>` from the login node, then `scancel`)
is the project's preference; if ALPS queues 32 interactive nodes forever or the site
requires batch, use the sbatch template `figs/weak_scaling/h100/sbatch_rung.template.sh`
(it runs the same runner inside the job with `--jobid=$SLURM_JOB_ID`, stdout to
`figs/weak_scaling/h100/logs/`). Either way the capsule is identical.

Per capsule, after the run: statuses all `ok`; `deterministic` 0; the manifest's
`flux_libs` sha256 equals the build's; `git add sweeps/results/runs/<run_id> && git commit`
(the printed command). Log run_id + per-cell status + the isolated max-rank means the
runner prints.

**Incident playbook** (log every one with its class):
- `timeout`/`stuck` cell: read `figs/weak_scaling/h100/raw/<run_id>/cells/<cell>/torchrun/*/stderr`
  for the last rank message; a wedge at the very end with all iterations recorded is a
  teardown hang (the A100 32n dwire cell did this — the metrics are valid, the status is
  not; note it in `figure_src.md` and re-run the cell once with `sweep.py rerun`).
- `skipped_capacity`: the sizer's heap demand exceeded `sym_size_max_g`; on a 96 GB part
  you may raise the cap to 40 (record it); never edit the sizer.
- `skipped_capability`: wrong/stale binary — rebuild ONCE, then redo every rung already
  taken (one binary rule).
- OOM in torch at 64 MiB / 32n: unexpected on 80–96 GB; check no other process holds the
  GPU (`nvidia-smi` in the step) before anything else.
- One arm's e2e has a 100 ms+ outlier iteration: the A100 campaign saw an intermittent
  stall tail on Qwen 16n; with 10 iterations the mean is affected — re-run the capsule
  once (`sweep.py rerun`) and keep the clean one, noting both run_ids.

---

## 8. Figure

```bash
python3 figs/weak_scaling/h100/build_figure_src.py                      # -> figs/weak_scaling/h100/figure_src.csv
python3 figs/weak_scaling/make_figure.py --baseline nvshmem --stacked --src-dir figs/weak_scaling/h100
python3 figs/weak_scaling/make_figure.py --baseline nvshmem            --src-dir figs/weak_scaling/h100
python3 figs/weak_scaling/make_figure.py --baseline nvshmem --budget 1 --src-dir figs/weak_scaling/h100
```

`build_figure_src.py` reproduces the A100 dataset's statistic exactly (verified on the
Perlmutter capsules: mean over all rank×iteration samples of the recorder's rule-5
`total_ms`; the max-across-ranks mean is kept in a side column) and the column contract
of `figs/weak_scaling/figure_src.csv`; it picks the newest `ok` capsule per cell, writes
failed cells with an empty latency and their status (the generator then draws the
on-axis fail note), computes the ours speedups against min(s1, dwire), and warns loudly
if the capsules span more than one binary. Write `figs/weak_scaling/h100/figure_src.md`
in the style of `figs/weak_scaling/figure_src.md`: per row the capsule, any status
caveat, the binary sha, GPU/memory, NVSHMEM version and transport, and the build line
used. Render checks as in `SPEC.md` §5 (no label collisions, fonts embedded). The A100
values are in `figs/weak_scaling/figure_src.csv` for a plausibility glance: at 64 MiB the
ring is 2.4–3.1× slower than Ours at every node count, at 1 MiB the gap grows with node
count; H100 GEMMs are faster, the wire is the same Slingshot-11 class, so totals should
be lower and the *shape* similar. Do not tune anything to make it so.

**Hand back**: commit capsules + `figs/weak_scaling/h100/{figure_src.csv,figure_src.md,
SESSION_LOG.md,*.pdf,*.png}` + the filled `env_alps.sh`/`alps.yaml` + any build-system
fix on `h100-weak-scaling`, push it (or `git bundle create h100.bundle main..h100-weak-scaling`),
and finish with a summary that states: the binary sha, the GPU/memory, the NVSHMEM
source, which build path (native sm_90a or JIT fallback), every deviation from this
document, and the 10 (nodes, budget) speedups.

---

## 9. What was done on the Perlmutter side (2026-09-06) and what is untested

- Hooks: `CMakeLists.txt` (`GEN_CUDAARCHS`), 7× `src/*/CMakeLists.txt` (generators take
  `GEN_CUDAARCHS`), `build.sh --gen-arch`, `src/generator/gen_{moe_ag_scatter,moe_gather_rs,
  comm_none,ag_gemm,gemm_rs}.cc` (`_H800{}` on the Sm80 spaces), `src/cuda/op_registry.cu`
  (`FLUX_ARCH_OVERRIDE`, `FLUX_SM_CORE_OVERRIDE`), `python/flux/util.py` (`get_arch`
  override), `sweeps/sweep.py` (`--platform alps`), `figs/weak_scaling/make_figure.py`
  (`--src-dir/--out-dir`; Perlmutter renders verified byte-identical).
- New files: `env_alps.sh` (template), `sweeps/platforms/alps.yaml`, the seven
  `sweeps/specs/h100_*.yaml`, `figs/weak_scaling/h100/{README.md,.gitignore,
  SESSION_LOG.md,build_figure_src.py,sbatch_rung.template.sh}`; on the
  `h100-weak-scaling` branch only: `figs/weak_scaling/h100/data/` (the routing pool +
  matrices, §4.1) — never merge that directory into main.
- Verified here: the specs parse and expand to 6 cells each (dry run), the builder
  reproduces the published A100 numbers, the generator still renders the committed
  figures byte-for-byte, python files parse. **Not compiled anywhere**: the C++/CMake
  hooks (no H100 or spare A100 rebuild window). They are small and default-off; if the
  Perlmutter default build ever breaks on them, the diff is in `git log -1 -- src/cuda/op_registry.cu`.
- Untested predictions that the gates confirm: CUTLASS-2.x Sm80 kernels compile for
  `sm_90a` (SM89 uses the same trick today); the V2 registry lookup succeeds under the
  override with the 132-SM key; NVSHMEM/CXI on ALPS behaves like Perlmutter's.
