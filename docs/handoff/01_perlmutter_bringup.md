# Perlmutter bring-up — a gated ladder

Every step gates the next. **Stop at the first failure and fix it there.** This shape is
copied from `docs/launch/aws_efa_environment.md`'s validation ladder, which worked.

Each item is tagged:

- **[VERIFIED-AWS]** — measured/observed on the AWS cluster. Mechanism transfers; numbers may not.
- **[HYPOTHESIS-PM]** — never executed on Perlmutter. Treat as a claim to falsify, not a fact.

The single most important framing: **`sweeps/platforms/perlmutter.yaml` is entirely
`[HYPOTHESIS-PM]`.** Zero of the 124 capsules are Perlmutter. Nothing in that file has ever
been exercised by any code path.

---

## §0 — Do not "fix" these; they already work

Spend no time here. **[VERIFIED-AWS]** by construction (they are the fall-through path AWS
had to override):

- `launch.sh:26-27` sets `NVSHMEM_LIBFABRIC_PROVIDER=${NVSHMEM_LIBFABRIC_PROVIDER:-cxi}`.
  CXI is the **unconditional default**; AWS only won by exporting first via `env_aws.sh`.
  There is no platform branch anywhere. Source `module.sh`, not `env_aws.sh`, and the
  transport is correct.
- `launch.sh:17` auto-detects `nproc_per_node` from visible GPUs — yields 4 here, 8 there.
- `launch_fast.sh:37-39` is the only genuine `cxi` conditional (sets
  `SLURM_MPI_TYPE=cray_shasta`), and it is already right.
- The `3rdparty/FAST` submodule is pinned at `c46620d`, which **is** the tip of its
  `perlmutter` branch. No submodule change needed. (`aws-8gpu` is one commit ahead and
  strictly backward-compatible — accepts 4 *or* 8 GPUs — so the two branches should probably
  just be merged rather than maintained as a platform split.)
- Flux's own C++/Python is topology-parameterized. The 4-vs-8 hardcodes live only in
  `3rdparty/FAST` (`nvidia/alltoall_nvshmem.cpp:1172,1328`, `nvidia/flash_tester.py:407`),
  and the pinned branch wants 4.
- `CLAUDE.md:80`'s `-T 4 -E 1` guidance for `gather_rs` becomes **correct again** at 4
  GPUs/node.

---

## §1 — Repair `module.sh` before sourcing it

**[VERIFIED-AWS: the defect is real; it is a static read of the file]**

`module.sh:31` hardcodes:

```bash
export FLUX_ROOT="$HOME/workspace/changchen/andrewy/flux"
```

This does not match this checkout. `module.sh:32` then derives `CPATH` from it, so a wrong
`FLUX_ROOT` silently points the NCCL header path at another tree and the guard at
`module.sh:41-50` may pass or fail for the wrong reason. Fix it the way `env_aws.sh:12`
already does:

```bash
export FLUX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
```

Gate: `source ./module.sh` completes, and `echo $FLUX_ROOT` is this checkout.

---

## §2 — The software contract, and what to do if the conda env is gone

**[HYPOTHESIS-PM]** `module.sh:27` activates `$PSCRATCH/conda_envs/andrewy-comet`.
`$PSCRATCH` is purged periodically at NERSC. **Assume this may be gone.**

There is no Perlmutter rebuild recipe anywhere in the repo — this section is the only
record. The contract itself is written down in exactly one place, the header of
`env_aws.sh:2-4`, which describes itself as keeping *"the validated Perlmutter software
contract"*:

| Component | Pin | Why it is load-bearing |
|---|---|---|
| CUDA | **12.4** | The whole stack was validated here; 13.0 was explicitly rejected on AWS |
| torch | **2.6.0+cu124** | Must match the CUDA pin |
| host compiler | gcc/g++ **≤ 13** (Perlmutter used `gcc/12.2.0`) | CUDA 12.4 rejects newer |
| arch | `TORCH_CUDA_ARCH_LIST=8.0`, `--arch 80 --sm-cores 108` | A100, 108 SMs, both platforms |
| NVSHMEM | module `nvshmem/3.2.5-1` | **differs from AWS's 3.3.9 — see §6** |
| NCCL | module `nccl/2.24.3` + bundled headers on `CPATH` | |

Module stack loaded by `module.sh:9-27`: CPE 25.09 lmod defaults, `PrgEnv-gnu`,
`gcc/12.2.0`, `craype-x86-milan`, `cray-mpich`, `cudatoolkit/12.4`, `nvshmem/3.2.5-1`,
`nccl/2.24.3`, `conda/Miniforge3-25.11.0-1`.

If the conda env is gone: create a fresh env, install `torch==2.6.0` from the cu124 index,
and re-point `module.sh:27`. **Do not** substitute a newer torch/CUDA to make it build — the
sm80 fused-kernel stack is validated against this exact pin, and drift here invalidates
comparison against all 124 existing capsules.

Gate: `python -c "import torch; print(torch.__version__, torch.version.cuda)"` prints
`2.6.0+cu124 12.4`.

---

## §3 — Build (compute node)

**[VERIFIED-AWS: the invocation; the AWS head-node ban does not apply here]**

```bash
srun --jobid=<id> --nodes=1 --ntasks-per-node=1 bash -lc \
  'source ./module.sh && nproc=16 ./build.sh --arch 80 --sm-cores 108 --nvshmem --no_test --jobs 16'
```

- `--nvshmem` is mandatory for the MoE kernels.
- `FLUX_BUILD_SKIP_CMAKE=1` for incremental rebuilds once `build/CMakeCache.txt` exists.
- **[VERIFIED-AWS]** `build.sh` can exit 1 from a cosmetic `ninja -t compdb` failure in
  `merge_compile_commands` *even when the build and install succeeded*. Check for
  `Successfully installed byte_flux` before believing the exit code.

Gate: `python -c "import flux"` works, and
`sha256sum python/flux/lib/libflux_cuda_ths_op.so` records your new build identity — you
will need it (see `04`).

---

## §4 — Correctness before any performance work

**[VERIFIED-AWS mechanism]**

```bash
srun --nodes=2 --ntasks-per-node=1 ./launch.sh test/python/moe_ag_scatter/test_moe_ag.py
```

Then the a2av traffic test with `FLUX_A2AV_CHECK_COMPRESS=1` on a small budget. The bar is
`correct_allclose=1` on every rank. Bitwise-vs-torch may legitimately fail with determinism
off; allclose is the contract.

**[VERIFIED-AWS] Failure-mode fingerprints** — the recv-capacity check is *per-rank and
data-dependent, not collective*, so an overflow does not fail cleanly:

| Symptom | Cause |
|---|---|
| One or two ranks throw `FLUX_CHECK` recv-overflow; the rest sit at 100% GPU forever | Recv capacity too small on the skewed ranks. Raise `FLUX_A2AV_MAX_RECV_NTOKENS`. Looks like a hang; is not. |
| All ranks hang, no error, rank-dependent, intermittent | One of the two hazard classes. Go to `03` NR-02 — do **not** start tuning knobs. |
| Cell status `stuck` | Runner idle watchdog (`idle_timeout_s`, default 180 s) fired. |
| Whole run ~3× slow, one cell only | Fabric transient. On AWS this was an EFA behaviour; the Slingshot analogue is **unknown**. Rerun outliers before believing them. |

---

## §5 — Convert `perlmutter.yaml` from hypothesis to record

**[HYPOTHESIS-PM — all of it]** Field by field, from `sweeps/platforms/perlmutter.yaml`:

| Field | Current | Problem |
|---|---|---|
| `nsys_bin` | **absent** | Falls back to bare `nsys` (`sweep.py:511,882`). `module.sh` loads no nsight module, so this is likely the CUDA 12.4 bundle's nsys 2023.4.4 — the exact version AWS found drops most kernel records under multi-process torchrun. **Pin a working nsys or treat `nsys` mode as untrustworthy.** |
| `srun_extra` | `[]` | Placeholder. The runner emits only `srun --jobid=... --nodes=N --ntasks-per-node=1`; anything the interactive QOS needs beyond what `salloc` pinned must go here. |
| `env` | `{}` | Placeholder. AWS carried transport knobs via `env_aws.sh`; the Perlmutter counterparts (e.g. `NVSHMEM_DISABLE_GDRCOPY`, `SLURM_MPI_TYPE`) are set only by `launch_fast.sh`, i.e. **only for FAST-driver cells**. Flux-driver cells get whatever `launch.sh` sets and nothing more. |
| `sym_size_max_g` | `16` | Copied from the 8-rank AWS file. Same 40 GB A100s, but per-rank heap pressure differs at 4 ranks/node. Never measured on CXI. |
| `data_root`, `matrices_root` | `${PSCRATCH}/...` | `sweep.py:230-236` **hard-fails** if `$PSCRATCH` is unresolved. The `generated/` matrix corpus has never existed there. |
| `fabric: cxi` | — | Descriptive only. Nothing reads it to configure transport; it lands in `cells.csv` for provenance. It cannot go wrong, but it cannot save you either. |

As each field is exercised, change its comment from hypothesis to a recorded fact with the
run_id that proved it. That is how this file stops being a liability.

---

## §6 — NVSHMEM 3.2.5 vs 3.3.9: check symbols, don't hope

**[HYPOTHESIS-PM]** Everything measured on AWS ran against NVSHMEM **3.3.9**; Perlmutter's
module is **3.2.5-1**. The two features most likely to bite are recent additions:

- `nvshmemx_getmem_nbi_on_stream` — the **PULL relay** (phase 1) depends on it entirely.
- the fused `putmem_signal` family — the wire's put+signal fusion, which the whole Tier B
  window-gating design assumes is one operation.

Before trusting any number, grep `$NVSHMEM_HOME/include` for both and confirm the
stream-ordered variants exist with the same signatures. If the fused signal is missing or
differs, Tier B's correctness argument (the signal flips *the window's* slot as part of the
put) does not hold and must be re-established.

---

## §7 — Regenerate matrices, and the identity trap

**[VERIFIED-AWS: the mechanism; `sweeps/gen_matrix.py`]**

Matrices are generated on demand at `matrices_root` from a deterministic FNV-1a of
`family|params|WxL|budget|topk|chunk|id`.

**The trap: `matrix_id` includes L.** An AWS `w16x8_...` matrix (W=16, L=8) and a
Perlmutter `w16x4_...` matrix (W=16, L=4) are **different traffic** even at identical world
size, budget and topk. There is no such thing as "the same cell on the other platform."
Cross-platform comparison is comparison of *variants within a matrix*, never of absolute
milliseconds across matrices.

**[VERIFIED-AWS]** A second identity subtlety, discovered while writing this handoff: the
`remotefrac` family exists in more than one parameterisation. Alongside the canonical
`w16x8_remotefrac_*` there are `w16x8_remotefrac-494119_*` and `w16x8_remotefrac-228dc7_*`
— high-skew variants with **balance headroom 0.19 and 0.26** versus the canonical 0.90. See
`02`; they matter more than anything else you will regenerate.

---

## §8 — Knob scaling was anchored at L=8

**[HYPOTHESIS-PM — flagged as a risk, not a known failure]**

`sweep.py:252-268` (`scale_knobs`) derives `MAX_RECV/STAGE/RELAY_NTOKENS` and
`NVSHMEM_SYMMETRIC_SIZE` from budget, topk and `chunk_bytes` **only** — the formula is
independent of L and world size, and its anchor points came from topk16 sweeps at L=8.

`MAX_RELAY_NTOKENS` sizes the *balanced relay staging buffer*, and under the balanced relay
each of the L local ranks carries roughly `1/L` of the round. **Halving L to 4 roughly
doubles each relay's share while the formula returns the same cap.**

In practice the floor (`163840`) is generous — at b8/k8 it is ~20× the actual row count — so
small and medium budgets probably absorb the doubling. The risk concentrates at **b32/b64**,
where the computed cap rises above the floor and the margin shrinks.

Predicted symptom if it does bite: the per-rank recv/relay overflow fingerprint from §4 —
a couple of ranks throwing while the rest spin at 100% GPU. **Do not misdiagnose this as a
hang.** If it appears, raise the three `NTOKENS` knobs via `extra_env` and record the
override; do not silently edit the formula until you have a measurement justifying a new
anchor.

---

## §9 — First sweep: dry run, then one cell

```bash
python sweeps/sweep.py run --platform perlmutter --variants hier,hier_compress_union \
    --families remotefrac --budgets-mib 8 --topk 8 --G 128 --modes isolated --dry-run
```

`--dry-run` first, always. Check the cell list, the resolved matrix paths, and the derived
knobs before consuming allocation time.

Then drop `--dry-run` for a single cell. What to watch:

- `skipped_capability` on any cell means the built `.so` lacks that env knob — a stale
  build, not a config error. The runner probes the binary for knob strings.
- `deterministic` must be `0` in `cells.csv`.
- The runner prints the isolated aggregate at cell finish; that is the quotable number.

**Allocation economics.** The runner's per-cell timeout is 900 s and `idle_timeout_s` is
180 s. Perlmutter's interactive QOS has wall limits and queue waits that AWS's `--exclusive`
on-demand allocation did not, so plan sweeps to fit one allocation and use the `retries`
pass rather than re-running a whole spec.

Commit the capsule with the `git add && git commit` line the runner prints. **Record your
`libflux_cuda_ths_op.so` sha256 in the commit message** — `04` explains why that one habit
matters more than anything else in this document.

---

## §10 — Then, and only then

Go to `02_algorithm_state_and_next_moves.md`. The re-test agenda there is ordered for the
NN>2 / L=4 question and assumes you have a green capsule in hand.
