# 43 — Open-source (arXiv) repository plan — DRAFT for taste review (2026-09-25)

Status: draft written while the conn=24 16n re-read queues; nothing has moved yet.
User direction (9/24–25): **our system only** (no baselines, no comparisons), the three
optimizations as on/off knobs, two model shapes (Qwen3-235B, Kimi K2), the dataset fixed
(LiveCodeBench routing traces), one script/build reproduces the main-perf numbers, clean
enough to read like upstream Flux or FAST. The same layer interface is what the SGLang
integration (Perlmutter, Qwen only, possibly hbm80g nodes) will call for the TTFT/TPOT figure.

## 1. What the current arms actually are (ground truth for the mapping)

Effective driver flags (sweeps/variants.py, resolved 9/25):

| paper row / arm | driver | flags that define it | env deltas |
|---|---|---|---|
| **Ours (full)** `ours_l01_s1_pv2_r2_pv3c_eps{025,05}` | `ours` | `--place_solver pv2 --redundant_per_rank 2 --route_rule pv3c --eps C --sizing demand --plan_overlap 2` | `_OURS_ENV` |
| Ours + live swap `..._s2_swap_force_p2p_r2_pv3c_eps025` | `ours` | + `--scenario s2 --s2_swap 1 --swap_xport p2p --swap_issue early --swap_tau_rows -1 --place_gain_threshold_ppm 0 --sizing capacity` | `_OURS_ENV` |
| Ours, direct wire `..._dwire_pv3c_eps025` | `ours` | + `--wire direct` | conn=8 |
| Ours, no overlap `llc_l01_s1_pv2_pv3c_eps025` | **`epic`** (different driver) | `--transport hier_compress --placement pv2 --router pv3c --eps C --redundant_per_rank 2` | conn=8, PLL tail graph |
| twins (ablation only) `_legacy` / `_mm0` / `_rpr0_nop2p` | `ours` | same | `FLUX_A2AV_LB_MINMOVE=0`, `FLUX_A2AV_RELAY_PER_ROUND=0`, `FLUX_A2AV_RELAY_P2P_PULL=0` |

`_OURS_ENV` (the defaults to freeze into code): LB_UNION=1, FUSED_STAGE2=1, EARLY_LAUNCH=1,
RS_MSPLIT=1, RS_EAGER=0, RS_FUSED_PACK=1, WAVE_PACK=1, RS_BUCKET=1, RS_WIRE_STREAMS=16,
CUDA_DEVICE_MAX_CONNECTIONS=24 (9/25 ruling), RS_PACK_BLOCKS=10, RS_REDUCE_BLOCKS=8,
RS_PRERED_BLOCKS=6, FLUX_OURS_PLAN_GRAPH=1, FLUX_OURS_PLAN_SCALE_GRAPH=1; plus the code
defaults FLUX_A2AV_LB_MINMOVE=1, RELAY_PER_ROUND=1, RELAY_P2P_PULL=1, blocking wire=1.

Observation that shapes the plan: today "overlap off" is a *different driver* (epic) and
"direct wire" is a flag of the ours driver. In the new repo both must be knobs of ONE layer.

## 2. The three knobs (proposal — confirm)

| knob | values | maps to today |
|---|---|---|
| `swap` | `off` (default on LiveCodeBench: no-swap won most cells) / `on` | s1 vs s2 (`--scenario s2 --s2_swap 1 …`) |
| `wire` | `fused` (comm/comp overlap, default) / `direct` | default vs `--wire direct` |
| `router_c` | float, default 1/4 at ≤ 8 nodes, 1/2 at 16 (the figure's C) | `--route_rule pv3c --eps C` |

Open question for the user: is there a fourth, "no overlap at all" setting (the epic-driver
row), or is `wire=direct` the only non-overlap point the repo needs? Placement (pv2, 2 replicas
per rank) stays always-on; the minimal-move partition, per-round staging and P2P pulls are
paper semantics, frozen on (their `_legacy` twins do not ship).

## 3. Repository layout (proposal)

```
<name>/
  README.md                  what it is, build, run, expected-results table, citation
  LICENSE / NOTICE            Apache-2.0 with Flux (ByteDance) attribution for src/comm
  env/perlmutter.sh          module pins incl. libfabric/1.20.1, nvshmem/3.2.5-1, cudatoolkit/12.4, conda env
  env/aws.sh                 (only if we keep the EFA path; else drop)
  build.sh                   one command: cmake + pip editable; --arch 80 fixed, no options a user must know
  src/comm/                  the two vendored, modified Flux ops (moe_ag_scatter a2av dispatch, moe_gather_rs a2av combine),
                             coll/, generator/, cutlass_impls needed by them — nothing else from Flux (no ag_gemm, gemm_rs, sm90, triton, int8)
  src/planner/               pv3 router ext, PLL/pv2 placement ext (the two small .cu files)
  python/<pkg>/
    planner.py               PlannerConfig, SwapConfig, Planner.place(), Planner.plan()
    comm.py                  CommConfig, ExpertComm (dispatch_gemm / gemm_combine / set_weights / apply_placement)
    layer.py                 EPMoE(nn.Module): load_weights, prepare, forward
    traces.py                trace loading + budget scaling (the dslots/budget_mib semantics)
    _ext/                    pybind of src/comm + src/planner
  bench/
    replay.py                the only experiment driver (below)
    launch.sh                torchrun + Slurm wrapper (from launch.sh; env defaults frozen inside)
    configs/{qwen3,k2}.yaml  shape presets (G, topk, H, ffn) — the two models
  data/
    traces/                  LiveCodeBench routing traces for both models (or a fetch script + checksum if too large for git)
  results/
    expected.csv             produced BY THIS REPO's build (rule 4: numbers are build-specific) — the reproduction target
    plot.py                  renders the main-perf-style bars from a results CSV
  scripts/reproduce.sh       salloc/srun chain: gate → 2/4/8/16n grid → results/*.csv → plot
  docs/design.md             mechanism (planner, fused dispatch, combine, swap), knob semantics, platform notes (CXI wire rule, heap sizing, channel pin)
```

Not shipped: sweeps/ (runner, 517 specs, 865 capsules), docs/handoff, figs/, all worktrees and
branches, the baseline drivers (epic, eplb, moonep, fast, nvshmem, comet), the FAST submodule,
test/ except a small correctness test, the 35 `_TAG` build markers.

## 4. The layer interface (from the 9/25 discussion)

```python
@dataclass(frozen=True)
class PlannerConfig: num_experts, topk, ranks, ranks_per_node, slots_per_rank=2, swap: SwapConfig|None=None, router_c: float=0.25, relay="minmove"
@dataclass(frozen=True)
class SwapConfig: trigger="band", c_swap=0.0, transport="p2p", issue="early"
class Planner:
    def place(self, expert_load) -> Placement
    def plan(self, topk_ids, topk_weights, placement, stream=None) -> Plan     # device tensors + ready event; can run one step ahead

@dataclass
class CommConfig: wire="fused"|"direct", blocking_wire=True, per_round_staging=True, relay_p2p_pull=True,
                  wire_streams=16, msplit=1, wave_adapt=48, pack_blocks=10, reduce_blocks=8, prered_blocks=6,
                  symmetric_heap_bytes=None, debug: DebugConfig|None=None
class ExpertComm:
    def set_weights(self, w1_slots, w2_slots); def apply_placement(self, placement)
    def dispatch_gemm(self, x, plan) -> Tensor        # layer 0: a2av dispatch fused into grouped GEMM 1
    def gemm_combine(self, h, plan) -> Tensor         # layer 1: grouped GEMM 2 fused with the a2av combine

class EPMoE(torch.nn.Module):
    def load_weights(self, w1, w2); def prepare(self, topk_ids, topk_weights) -> Plan
    def forward(self, x, topk_ids, topk_weights, plan=None) -> Tensor
```

`bench/replay.py --model qwen3|k2 --nodes N --budget-mib B --swap off|on --wire fused|direct --router-c C`
is a thin driver: load traces → build EPMoE → per iteration: sync+barrier, plan, forward, record.

## 5. Environment-knob → argument conversion (the C++ side)

95 `getenv` names in the two ops (list in handoff 42 §10 notes / memory): ~35 `*_TAG` build
markers → delete; ~15 debug/trace (CHECK_IDENTITY, POISON, TIMING, TILE_TRACE, NVTX_PROXY,
TRACE_EPOCHS, SPIN_LIMIT, WAIT_FLUSH…) → `DebugConfig` or delete; ~45 tuning/semantics →
`CommConfig` fields with today's defaults, passed at op construction; the op reads nothing from
the environment. `CUDA_DEVICE_MAX_CONNECTIONS=24` and the NVSHMEM bootstrap/heap settings move
into `bench/launch.sh` (platform), documented in docs/design.md. Every conversion is gated by the
same-binary twin protocol (old env path vs new arg path, one capsule) before the old path is
deleted.

## 6. Sequencing

1. (now) conn=24 16n re-read → regenerate figs/main_perf figure_src from the 9/24–25 capsules
   → merge pv3 → main. Research tree = archive.
2. User taste pass on this document (names, layout, knob semantics, the "no overlap" question).
3. Extract: new repo, `src/comm` + `src/planner` + `python/<pkg>` + `bench` copied and trimmed;
   build; 1-node smoke; 4n gate (random payload); 4n Qwen replay = first reproduction check
   against the research-tree numbers (expect ≤ 3 % twin noise).
4. Env→arg conversion in the vendored ops with twin gates; delete the env path.
5. Full grid on the new build → `results/expected.csv` (≈ 30 node-hours) → README table.
6. SGLang: `EPMoE` as the EP-MoE backend; planner one step ahead; decode either graph-capturable
   or eager fallback; NVSHMEM beside SGLang's NCCL groups (CXI wedge class from the FAST work
   is the known risk); Qwen3-235B only; measure TTFT/TPOT.

## 7. Open questions for the user

- "router off" meaning (C = ∞ i.e. unconstrained? or a fixed hash route?) — or drop the knob and keep C as a float.
- Does the "no overlap" row (epic driver) need to exist in the repo, or is `wire=direct` enough?
- Keep K2 in the open-source example even though the SGLang run is Qwen-only? (traces are cheap; weights are not needed for the replay bench)
- Trace hosting: in-repo vs fetch script (size to be measured).
- Repo name / package name.

## 8. User rulings on §7 (2026-09-25)

1. **Router**: no "off" state; C is a configurable float and the reproduction script's defaults
   match the figure's tested values (1/4 at ≤ 8 nodes, 1/2 at 16).
2. **No-overlap row**: removed from the plan; `wire=direct` is the only non-overlap point.
3. **K2 stays** in the open-source example alongside Qwen (replay needs traces, not weights).
4. **Traces**: LiveCodeBench only (what main perf needs), kept in-repo if the size allows
   (measured below); the user may revisit.
5. **Repo / package name**: to be discussed with the postdoc; a placeholder is used and renamed
   later (package name appears only in imports, setup metadata and the README — a mechanical rename).
Also ruled earlier: 2n is not part of the reproduce script; swap-decision compression happens in
the extraction (band test first, decision off the critical path).
