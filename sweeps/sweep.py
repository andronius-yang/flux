#!/usr/bin/env python3
"""Sweep runner: reproducible perf sweeps of the layer0 dispatch variants.

    python sweeps/sweep.py run --platform aws --variants hier,hier_compress \
        --families remotefrac --budgets-mib 2,8 --topk 8 --G 128 --modes e2e,phases
    python sweeps/sweep.py rerun sweeps/results/runs/<run_id>
    python sweeps/sweep.py run ... --dry-run     # print cells + srun lines, exit

Contract: sweeps/SCHEMA.md. Every invocation produces one immutable run
capsule under sweeps/results/runs/<run_id>/ (manifest.json, spec.yaml,
metrics.csv, cells.csv — small, committed to git) plus a staging directory at
the platform data_root (rank JSONLs, torchrun logs, nsys/prof artifacts —
platform-local, referenced by path+hash from the manifest). The runner never
allocates (use `salloc ... --no-shell` first), never commits (it prints the
commit command), and always exports FLUX_TEST_DETERMINISTIC=0 for perf cells.
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_matrix  # noqa: E402
import gen_trace_routing  # noqa: E402
from variants import VARIANTS  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(REPO_ROOT, "sweeps", "results", "runs")
TEST = "test/python/moe_ag_scatter/test_moe_ag_traffic.py"
TEST_FAST = "test/python/moe_ag_scatter/test_moe_ag_fast_baseline.py"
TEST_MOONEP = "test/python/moe_ag_scatter/test_moe_moonep_traffic.py"
TEST_ULTRAEP = "test/python/moe_ag_scatter/test_moe_ultraep_traffic.py"
TEST_MOONEP_FUSED = "test/python/moe_ag_scatter/test_moe_moonep_fused_traffic.py"
TEST_EPLB = "test/python/moe_ag_scatter/test_moe_eplb_traffic.py"
TEST_EPIC = "test/python/moe_ag_scatter/test_moe_epic_traffic.py"
TEST_GATHER_RS = "test/python/moe_gather_rs/test_moe_gather_rs_traffic.py"
TEST_FAST_GATHER_RS = "test/python/moe_gather_rs/test_moe_gather_rs_fast_baseline.py"
TEST_MOONEP_L1 = "test/python/moe_gather_rs/test_moe_moonep_l1_traffic.py"
TEST_L01 = "test/python/moe_combined/test_moe_l0l1_traffic.py"

MODES = ("e2e", "isolated", "phases", "torchprof", "nsys")
MODE_ORDER = {m: i for i, m in enumerate(MODES)}  # clean numbers before perturbed

# an nsys capture below this is an empty/aborted report (a real single-node
# 8-rank rep measures ~1.3 MB); nsys cells missing a full-size rep per node
# get status nsys_empty instead of silently passing
NSYS_REP_MIN_BYTES = 100_000

# defaults for the fully-resolved spec; anything not overridden by --spec or
# flags is pinned here (constants like H/chunk_bytes change only via a spec)
SPEC_DEFAULTS = {
    "platform": None,
    "nodes": 2,
    "variants": ["hier", "hier_compress"],
    "families": ["remotefrac"],
    "budgets_mib": [2, 8],
    "topk": 8,
    "G": 128,
    "H": 4096,
    "chunk_bytes": 8192,
    "ffn_hidden": 4096,
    "dtype": "bfloat16",
    "iters": 10,
    "warmup_iters": 5,
    "profile_iters": 3,  # iters and warmup for torchprof/nsys cells
    "sm_margin": 8,
    "modes": ["e2e"],
    "matrix_instance": "001",
    # split-N pipeline depth for layer1 (gather_rs) cells; with N == H == 4096
    # the op constraint N/n_split % 1024 == 0 allows {1, 2, 4}; 4 is the
    # bench's own default so standalone-bench numbers stay comparable
    "n_split_l1": 4,
    "skip_correctness": False,
    "timeout_s": 900,
    "idle_timeout_s": 180,  # kill a cell whose logs stop growing for this long
    "retries": 1,  # re-run stuck/timeout/failed cells this many times at the end
    "extra_env": {},
    "notes": "",
}

RETRY_STATUSES = ("stuck", "timeout", "failed")

# stderr phase-mark formats from src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc
PHASE_PATTERNS = [
    (
        re.compile(
            r"\[a2av-timing\] rank (\d+) stage1 ([\d.]+) stage2 ([\d.]+)"
            r" gemmgate ([\d.]+) gemm ([\d.]+) barrier ([\d.]+) ms"
        ),
        ["stage1_ms", "stage2_ms", "gemmgate_ms", "a2av_gemm_ms", "barrier_ms"],
        1.0,
    ),
    (
        re.compile(
            r"\[a2av-stage2\] rank (\d+) mask ([\d.]+) keyA ([\d.]+) sortA ([\d.]+)"
            r" keyR ([\d.]+) sortR ([\d.]+) inv ([\d.]+) gather ([\d.]+)"
            r" scatter ([\d.]+) cnt ([\d.]+) cumsum ([\d.]+) ms"
        ),
        [
            "stage2_mask_ms",
            "stage2_keyA_ms",
            "stage2_sortA_ms",
            "stage2_keyR_ms",
            "stage2_sortR_ms",
            "stage2_inv_ms",
            "stage2_gather_ms",
            "stage2_scatter_ms",
            "stage2_cnt_ms",
            "stage2_cumsum_ms",
        ],
        1.0,
    ),
    (
        re.compile(
            r"\[a2av-host\] rank (\d+) enq_stage1 (\d+) us enq_stage2 (\d+) us"
            r" counts_wait (\d+) us"
        ),
        ["host_enq_stage1_ms", "host_enq_stage2_ms", "host_counts_wait_ms"],
        1e-3,
    ),
    (
        re.compile(
            r"\[a2av-relayfwd\] rank (\d+) dl ([\d.]+) flag ([\d.]+) canon ([\d.]+)"
            r" mask ([\d.]+) valid ([\d.]+) cumsum ([\d.]+) tgt ([\d.]+)"
            r" flatten ([\d.]+) scatter ([\d.]+) cnts ([\d.]+) d2h ([\d.]+) ms"
        ),
        [
            "relayfwd_dl_ms",
            "relayfwd_flag_ms",
            "relayfwd_canon_ms",
            "relayfwd_mask_ms",
            "relayfwd_valid_ms",
            "relayfwd_cumsum_ms",
            "relayfwd_tgt_ms",
            "relayfwd_flatten_ms",
            "relayfwd_scatter_ms",
            "relayfwd_cnts_ms",
            "relayfwd_d2h_ms",
        ],
        1.0,
    ),
]

CELLS_COLUMNS = [
    "run_id",
    "cell_id",
    "status",
    "platform",
    "variant",
    "comm_pattern",
    "mode",
    "matrix_id",
    "matrix_path",
    "matrix_sha256",
    "family",
    "family_params",
    "routing_mode",
    "routing_path",
    "routing_sha256",
    "budget_mib",
    "topk",
    "G",
    "H",
    "chunk_bytes",
    "dtype",
    "world_size",
    "nnodes",
    "ranks_per_node",
    "fabric",
    "ntokens",
    "tokens_per_rank",
    "iters",
    "warmup_iters",
    "sm_margin",
    "deterministic",
    "env_json",
    "git_sha",
    "git_dirty",
    "wire_ratio",
    "relay_ident_bytes",
    "relay_balanced_bytes",
    "correct_bitwise",
    "correct_allclose",
    "exit_code",
    "start_ts",
    "end_ts",
    "log_dir",
    "nsys_path",
    "prof_path",
    "notes",
    # appended 2026-08-16 (layer-axis campaign): old capsules simply lack the
    # columns — readers use csv.DictReader, never positional indexing
    "layer",
    "timing_mode",
]
METRICS_COLUMNS = [
    "run_id",
    "cell_id",
    "mode",
    "impl",
    "rank",
    "iter",
    "metric",
    "value_ms",
    "source",
]


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def load_yaml(path):
    with open(path) as f:
        text = f.read()
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        import miniyaml

        return miniyaml.loads(text)


def dump_yaml(obj, path):
    try:
        import yaml

        with open(path, "w") as f:
            yaml.safe_dump(obj, f, sort_keys=True, default_flow_style=False)
    except ImportError:
        import miniyaml

        with open(path, "w") as f:
            f.write(miniyaml.dumps(obj) + "\n")


def load_platform(name, dry=False):
    plat = load_yaml(os.path.join(REPO_ROOT, "sweeps", "platforms", f"{name}.yaml"))
    for key in ("data_root", "matrices_root", "traces_root"):
        if key not in plat:
            continue  # traces_root is optional (only trace-family cells need it)
        plat[key] = os.path.expandvars(plat[key])
        if "$" in plat[key]:
            if not dry:
                raise SystemExit(f"platform {name}: {key} has unresolved env vars: {plat[key]}")
            print(f"WARNING (dry-run): {key} unresolved on this host: {plat[key]}")
    return plat


def detect_jobid(explicit):
    if explicit:
        return str(explicit)
    r = sh(["squeue", "-u", os.environ.get("USER", ""), "-h", "-t", "RUNNING", "-o", "%i"])
    ids = [x for x in r.stdout.split() if x]
    if len(ids) != 1:
        raise SystemExit(
            f"need exactly one RUNNING allocation to autodetect --jobid, found {ids};"
            " run salloc --no-shell first or pass --jobid"
        )
    return ids[0]


def scale_knobs(budget_mib, topk, chunk_bytes):
    """LEGACY anchor formula (topk16 sweeps: post-topk 2..256 MiB -> 163840/6G,
    512 -> 262144/10G, 1024 -> 524288/16G). W-blind: the lb_union recv demand
    grows ~W*T while this cap is ~32*T, which broke b64/W32 and b32+b64/W64
    (2026-08-15). Kept only as the fallback when a cell's matrix artifacts are
    not on disk (e.g. some dry runs); real runs use exact_scale_knobs."""
    row_chunks = budget_mib * (1 << 20) * topk // chunk_bytes
    cap = max(163840, math.ceil(4 * row_chunks / 8192) * 8192)
    post_mib = budget_mib * topk
    if post_mib <= 256:
        sym_g = 6
    else:
        sym_g = min(16, math.ceil(6 + (post_mib - 256) * 10 / 768))
    return {
        "FLUX_A2AV_MAX_RECV_NTOKENS": str(cap),
        "FLUX_A2AV_MAX_STAGE_NTOKENS": str(cap),
        "FLUX_A2AV_MAX_RELAY_NTOKENS": str(cap),
        "NVSHMEM_SYMMETRIC_SIZE": f"{sym_g}G",
    }


def read_matrix_chunks(matrix_path, chunk_bytes):
    """Parse a matrix .txt (W, then W*W byte counts) into [W][W] row counts."""
    with open(matrix_path) as f:
        toks = f.read().split()
    w = int(toks[0])
    vals = [int(x) for x in toks[1 : 1 + w * w]]
    return [[vals[s * w + d] // chunk_bytes for d in range(w)] for s in range(w)]


_EXACT_KNOB_CACHE = {}
_MATRIX_STATS_CACHE = {}


def matrix_dedup_stats(matrix, spec, plat, routing_mode):
    """(chunks, u, U, T) for a matrix+routing pair — one parse + dedup
    derivation shared by the layer0 (exact_scale_knobs) and layer1
    (exact_rs_scale_knobs) sizers. u/U come from the routing file when
    routing_mode == real, else from the dealer closed form (the same split
    the layer0 sizer always used)."""
    key = (matrix["path"], routing_mode)
    if key not in _MATRIX_STATS_CACHE:
        cb = spec["chunk_bytes"]
        topk = spec["topk"]
        L = plat["ranks_per_node"]
        chunks = read_matrix_chunks(matrix["path"], cb)
        W = len(chunks)
        nn = W // L
        T = sum(chunks[0]) // topk  # budget invariant: row sums = T * topk
        if routing_mode == "real":
            routing = gen_trace_routing.read_routing(matrix["routing"])
            u, U = gen_trace_routing.real_dedup_stats(routing, W, L, T, spec["G"])
        else:
            u = gen_matrix.dealer_dedup_u(chunks, T)
            U = [
                [min(sum(chunks[s][m * L + j] for j in range(L)), T) for m in range(nn)]
                for s in range(W)
            ]
        _MATRIX_STATS_CACHE[key] = (chunks, u, U, T)
    return _MATRIX_STATS_CACHE[key]


def exact_scale_knobs(matrix, spec, plat, routing_mode):
    """Exact per-cell a2av knobs + heap from the SAME expressions the runtime
    FLUX_CHECKs (gen_matrix.a2av_knob_demands), computed from the on-disk
    matrix (+ routing file when routing_mode == real; dealer closed form
    otherwise). Per-knob sizing: RECV from the copies/union column max,
    STAGE/RELAY from their own bounds (the legacy uniform cap inflated
    stage/relay ~2.5-5x, which is what pushed b64 heaps to the 16G ceiling).
    Demands are rounded up to 8192 rows and floored at the legacy 163840 so
    every previously-passing small-budget cell keeps a byte-identical
    env_json. Returns (env dict, uncapped_sym_g) — the caller skips the cell
    as skipped_capacity when uncapped_sym_g exceeds the platform heap cap."""
    key = (matrix["path"], routing_mode)
    if key not in _EXACT_KNOB_CACHE:
        cb = spec["chunk_bytes"]
        topk = spec["topk"]
        L = plat["ranks_per_node"]
        chunks, u, U, T = matrix_dedup_stats(matrix, spec, plat, routing_mode)
        W = len(chunks)
        d = gen_matrix.a2av_knob_demands(chunks, u, U, L)

        def cap(rows):
            return max(163840, math.ceil(rows / 8192) * 8192)

        recv = cap(max(d["recv_copies"], d["recv_union"]))
        stage = cap(max(d["stage_hier"], d["stage_ident"], d["stage_lb"]))
        relay = cap(d["relay_lb"])
        # heap: whichever the ctor allocates — a2av buffers (2 send halves +
        # recv + stage + relay) or the dense gathered input (W*T rows), plus
        # 1G for signal buffers / NVSHMEM internals; floor at the legacy 6G
        send_rows = 2 * T * topk
        need_rows = max(send_rows + recv + stage + relay, W * T)
        sym_g = max(6, math.ceil(need_rows * cb / (1 << 30)) + 1)
        env = {
            "FLUX_A2AV_MAX_RECV_NTOKENS": str(recv),
            "FLUX_A2AV_MAX_STAGE_NTOKENS": str(stage),
            "FLUX_A2AV_MAX_RELAY_NTOKENS": str(relay),
            "NVSHMEM_SYMMETRIC_SIZE": f"{sym_g}G",
        }
        _EXACT_KNOB_CACHE[key] = (env, sym_g)
    env, sym_g = _EXACT_KNOB_CACHE[key]
    return dict(env), sym_g


_EXACT_RS_KNOB_CACHE = {}


def exact_rs_scale_knobs(matrix, spec, plat, routing_mode, comm_pattern):
    """Layer1 analog of exact_scale_knobs: exact FLUX_A2AV_RS_MAX_*_ROWS +
    heap for gather_rs-driver cells, from the same expressions as the op's
    collective FLUX_CHECKs (gen_matrix.a2av_rs_knob_demands; the layer1 wire
    is the TRANSPOSE of the dispatch matrix, handled inside that function —
    inputs stay in dispatch orientation). Demands round up to 8192 rows with
    NO legacy floor (new axis, no historical env_json to preserve). All four
    knobs are always set — the op only reads the panels its branch allocates.
    Heap sizing follows the ctor allocations (panels are [n_split, rows,
    N/n_split] so bytes = rows * chunk_bytes): send + recv(cpr, knob-free) +
    stage (non-compress) | conv + wire (compress), +1G slack, 6G floor.
    `dense` has no a2av panels; its multi-node ring/staging buffers are not
    audited here, so it gets the same conservative bound (verify at the 2n
    bring-up smoke — flagged in the campaign plan). Returns (env dict,
    uncapped_sym_g), same skipped_capacity contract as the layer0 sizer."""
    key = (matrix["path"], routing_mode, comm_pattern)
    if key not in _EXACT_RS_KNOB_CACHE:
        cb = spec["chunk_bytes"]
        topk = spec["topk"]
        L = plat["ranks_per_node"]
        chunks, u, U, T = matrix_dedup_stats(matrix, spec, plat, routing_mode)
        d = gen_matrix.a2av_rs_knob_demands(chunks, U, L)

        def cap(rows):
            return max(8192, math.ceil(rows / 8192) * 8192)

        send = cap(d["rs_send"])
        stage = cap(d["rs_stage"])
        conv = cap(d["rs_conv"])
        wire = cap(d["rs_wire"])
        cpr = T * topk  # recv panel rows: knob-free max_m/W exact (cc :233)
        if comm_pattern == "a2av_hier_compress":
            need_rows = send + cpr + conv + wire
        else:  # a2av_hier and (conservatively) dense
            need_rows = send + cpr + stage
        sym_g = max(6, math.ceil(need_rows * cb / (1 << 30)) + 1)
        env = {
            "FLUX_A2AV_RS_MAX_SEND_ROWS": str(send),
            "FLUX_A2AV_RS_MAX_STAGE_ROWS": str(stage),
            "FLUX_A2AV_RS_MAX_CONV_ROWS": str(conv),
            "FLUX_A2AV_RS_MAX_WIRE_ROWS": str(wire),
            "NVSHMEM_SYMMETRIC_SIZE": f"{sym_g}G",
        }
        _EXACT_RS_KNOB_CACHE[key] = (env, sym_g)
    env, sym_g = _EXACT_RS_KNOB_CACHE[key]
    return dict(env), sym_g


def parse_family(spec_str):
    """'hotcol:frac=0.7' -> ('hotcol', {'frac': 0.7}); 'uniform' -> ('uniform', {})."""
    name, _, rest = spec_str.partition(":")
    params = gen_matrix.parse_params(rest.split(";")) if rest else {}
    return name, params


def family_slug(name, params):
    defaults = gen_matrix.FAMILY_DEFAULT_PARAMS[name]
    merged = dict(defaults, **params)
    if merged == defaults:
        return name
    return f"{name}-{gen_matrix.fnv1a(json.dumps(merged, sort_keys=True)) & 0xFFFFFF:06x}"


def expand_cells(spec, plat):
    world = spec["nodes"] * plat["ranks_per_node"]
    cells = []
    for mode in sorted(spec["modes"], key=lambda m: MODE_ORDER[m]):
        for fam_str in spec["families"]:
            fam, fparams = parse_family(fam_str)
            # trace family: real per-token routing by default; dealer=1 keeps
            # the SAME matrix_id/bytes but feeds them through the synthetic
            # max-dedup dealer (paired counterfactual isolating token overlap)
            routing_mode = ""
            if fam == "trace":
                routing_mode = "dealer" if fparams.get("dealer") else "real"
            for budget in spec["budgets_mib"]:
                for vname in spec["variants"]:
                    if vname not in VARIANTS:
                        raise SystemExit(f"unknown variant {vname}; see sweeps/variants.py")
                    driver = VARIANTS[vname].get("driver", "flux")
                    if routing_mode == "real" and driver in ("fast", "fast_gather_rs"):
                        raise SystemExit(
                            f"variant {vname} (fast driver) cannot consume a routing"
                            f" file; use the dealer=1 arm for trace matrices"
                        )
                    if driver in ("fast", "fast_gather_rs"):
                        if spec["nodes"] < 2:
                            raise SystemExit(
                                f"variant {vname} requires nodes >= 2"
                                " (FAST asserts server_n > 1)"
                            )
                        if mode != "e2e":
                            print(
                                f"NOTE: {vname} x {mode} not generated — fast phase"
                                " metrics arrive free in e2e (host-blocking"
                                " alltoallv); --profile/nsys unsupported"
                            )
                            continue
                    if (driver in ("moonep", "ultraep", "eplb", "epic")
                            and mode == "phases"):
                        print(
                            f"NOTE: {vname} x phases not generated — its phase"
                            " metrics (plan_comm/pack/comm/scatter/prefetch/gemm)"
                            " arrive free in every mode via the recorder"
                        )
                        continue
                    if driver in ("gather_rs", "moonep_l1") and mode == "phases":
                        print(
                            f"NOTE: {vname} x phases not generated — the gather-rs"
                            " op has no FLUX_A2AV_TIMING stderr marks; a phases"
                            " cell would be an empty perturbed cell"
                        )
                        continue
                    if driver == "l01" and mode == "phases":
                        print(
                            f"NOTE: {vname} x phases not generated — the combined"
                            " bench reports l0/act/l1 sub-events via the recorder"
                            " in every mode; l0 phase marks would perturb the"
                            " window semantics"
                        )
                        continue
                    fslug = family_slug(fam, fparams)
                    # timing_mode is a cell axis ONLY for l1 flux (gather_rs)
                    # cells: isolated = in-forward index build, amortized =
                    # layer0-inherited indices (the combined-pass proxy)
                    timing_modes = [""]
                    if driver in ("gather_rs", "moonep_l1"):
                        timing_modes = ["isolated", "amortized"]
                    for tm in timing_modes:
                        tm_slug = {"isolated": "_tmiso", "amortized": "_tmamo"}.get(tm, "")
                        cells.append(
                            {
                                "cell_id": f"{vname}_{fslug}_b{budget}"
                                f"_k{spec['topk']}{tm_slug}_{mode}",
                                "variant": vname,
                                "mode": mode,
                                "family": fam,
                                "family_params": fparams,
                                "routing_mode": routing_mode,
                                "budget_mib": budget,
                                "world_size": world,
                                "layer": VARIANTS[vname].get("layer", "l0"),
                                "timing_mode": tm,
                            }
                        )
    return cells


def probe_capabilities(needed):
    """Search the built flux shared libraries for env-knob strings. Detects a
    stale build (source has the knob, binary doesn't) — cells requiring an
    absent knob are skipped instead of silently measuring the wrong thing."""
    # find_spec locates the installed flux package WITHOUT executing it
    # (importing flux needs a GPU; the runner lives on the login/head node)
    r = sh(
        [
            sys.executable,
            "-c",
            "import importlib.util, os;"
            " s = importlib.util.find_spec('flux');"
            " print(os.path.dirname(s.origin) if s else '')",
        ]
    )
    libdir = r.stdout.strip()
    if r.returncode != 0 or not libdir:
        libdir = os.path.join(REPO_ROOT, "python", "flux")  # editable-install layout
    if not os.path.isdir(libdir):
        raise SystemExit(f"cannot locate the flux package (tried find_spec and {libdir})")
    sos = sorted(
        glob.glob(os.path.join(libdir, "lib", "*.so*"))
        # Perlmutter's pip/setuptools lays the CUDA libs into lib64/ (AWS used
        # lib/); probe both so the capability check is layout-independent
        + glob.glob(os.path.join(libdir, "lib64", "*.so*"))
        + glob.glob(os.path.join(libdir, "*.so"))
    )
    if not sos:
        raise SystemExit(f"no shared libraries under {libdir}")
    found = {k: False for k in needed}
    for so in sos:
        with open(so, "rb") as f:
            blob = f.read()
        for k in needed:
            if not found[k] and k.encode() in blob:
                found[k] = True
    return found, sos


def git_info():
    sha = sh(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).stdout.strip()
    # capsules are data, not code: an uncommitted sibling capsule (e.g. from a
    # previous run in the same sequence) must not mark later runs dirty
    dirty = bool(
        sh(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)sweeps/results"],
            cwd=REPO_ROOT,
        ).stdout.strip()
    )
    return sha, dirty


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fast_sym_size(matrix_path, plat):
    """FAST heap sizing: the test's auto capacity is 4*max(max row sum, max col
    sum) and the heap holds ~3 capacity-sized buffers; 4x capacity gives
    headroom. Col sums are family-dependent, so parse the matrix itself."""
    with open(matrix_path) as f:
        toks = f.read().split()
    w = int(toks[0])
    vals = [int(x) for x in toks[1 : 1 + w * w]]
    rows = [sum(vals[i * w : (i + 1) * w]) for i in range(w)]
    cols = [sum(vals[i::w]) for i in range(w)]
    cap = 4 * max(max(rows), max(cols))
    sym_g = max(4, math.ceil(4 * cap / (1 << 30)))
    sym_max = plat.get("sym_size_max_g")
    if sym_max:
        sym_g = min(sym_g, int(sym_max))
    return f"{sym_g}G"


def moonep_sym_size(matrix_path, plat):
    """Symmetric-heap sizing for the moonep nvshmem arms. All2AllSingle
    allocates in+out staging of max_split*W rows each, twice (hidden rows +
    per-entry fp32 weights). max_split is the plan's largest per-(src,dst)
    representative-row count, bounded above by the largest matrix entry in
    chunks (dedup only shrinks it); weights staging is bounded by the same
    entry count at 4 bytes. 2x headroom, floor 2G.

    CAVEAT: the //1024 below hard-wires chunk_bytes = 8192 (H=4096 bf16):
    it is (entries = bytes/8192) * 4B written as bytes/1024. Kept as-is so
    existing moonep capsules' NVSHMEM_SYMMETRIC_SIZE stays byte-stable in
    env_json; ultraep_sym_size below does it chunk-aware."""
    with open(matrix_path) as f:
        toks = f.read().split()
    w = int(toks[0])
    vals = [int(x) for x in toks[1 : 1 + w * w]]
    max_pair_bytes = max(vals)  # = max_split_rows * chunk_bytes upper bound
    hidden = 2 * w * max_pair_bytes          # in + out staging, bf16 rows
    weights = 2 * w * (max_pair_bytes // 1024)  # entries * 4B << hidden
    sym_g = max(2, math.ceil(2 * (hidden + weights) / (1 << 30)))
    sym_max = plat.get("sym_size_max_g")
    if sym_max:
        sym_g = min(sym_g, int(sym_max))
    return f"{sym_g}G"


def moonep_getmem_sym_size(matrix_path, plat, spec, variant):
    """Symmetric-heap sizing for moonep arms with the getmem weight path.
    The [epn, ffn_shard, H] weight home AND the [B, ffn_shard, H] prefetch
    slots (B = epn default) both live PERMANENTLY on the heap — the home
    because remote gets read it, the slots because a proxy-mediated get's
    LOCAL destination must be provider-registered on CXI (ordinary
    cudaMalloc dst segfaults; found 2026-08-11). Residency moves, it does
    not grow: both replace ordinary-memory tensors, mirroring upstream's
    [E+B, H, H'] mapped weight tensor. Token staging is added only when the
    token transport is also nvshmem, using moonep_sym_size's All2AllSingle
    terms verbatim (incl. its frozen //1024 caveat) so mixed arms stay
    comparable. 2x headroom, floor 2G, plat cap."""
    with open(matrix_path) as f:
        toks = f.read().split()
    w = int(toks[0])
    chunk = int(spec["chunk_bytes"])  # = H * itemsize (bf16 row bytes)
    epn = int(spec["G"]) // w
    # epn home rows + B (= epn) prefetch slots, each ffn_shard rows of H*2B
    home = 2 * epn * int(spec["ffn_hidden"]) * chunk
    staging = 0
    if "nvshmem" in (variant.get("test_args") or []):
        vals = [int(x) for x in toks[1 : 1 + w * w]]
        max_pair_bytes = max(vals)
        staging = 2 * w * max_pair_bytes + 2 * w * (max_pair_bytes // 1024)
    sym_g = max(2, math.ceil(2 * (staging + home) / (1 << 30)))
    sym_max = plat.get("sym_size_max_g")
    if sym_max:
        sym_g = min(sym_g, int(sym_max))
    return f"{sym_g}G"


def ultraep_sym_size(matrix_path, plat, spec, variant):
    """Symmetric-heap sizing for the ultraep nvshmem arms. Two All2AllSingle
    ops (hidden rows n_dim=H; per-row fp32 probs n_dim=1 — NO dedup, so the
    probs splits equal the row splits), each allocating in+out staging of
    max_split*W rows at construction.

    Bound: the reroute can redirect a token only among instances of its
    logical target expert, and replicas live strictly inside that expert's
    NVL domain — so post-reroute rows src->dst are bounded by src's LOGICAL
    traffic to dst's whole domain: max over (src, domain) of the
    domain-summed matrix row slice. Chunk-aware (no H=4096 hard-wiring).
    2x headroom, floor 2G, capped by plat sym_size_max_g."""
    with open(matrix_path) as f:
        toks = f.read().split()
    w = int(toks[0])
    vals = [int(x) for x in toks[1 : 1 + w * w]]
    ta = variant.get("test_args") or []
    if "--nvl_domain_size" in ta:
        d = int(ta[ta.index("--nvl_domain_size") + 1])
    else:
        d = w // int(spec["nodes"])
    max_pair_bytes = max(
        sum(vals[src * w + dst] for dst in range(g * d, (g + 1) * d))
        for src in range(w)
        for g in range(w // d)
    )
    chunk = int(spec["chunk_bytes"])
    hidden = 2 * w * max_pair_bytes                   # in+out staging, row bytes
    probs = 2 * w * (max_pair_bytes // chunk) * 4     # one fp32 per row (no dedup)
    sym_g = max(2, math.ceil(2 * (hidden + probs) / (1 << 30)))
    sym_max = plat.get("sym_size_max_g")
    if sym_max:
        sym_g = min(sym_g, int(sym_max))
    return f"{sym_g}G"


def moonep_l1_sym_size(matrix, spec, plat, routing_mode, v):
    """Symmetric-heap sizing for the moonep VIRTUAL-SPACE layer1 cells. The
    runner cannot compute the MoonEP plan (torch-free login), so it bounds
    the virtual-space demands from the matrix: replication only ever REMOVES
    cross-node copies (a copy either stays dispatched or becomes home-local),
    so the matrix-based stage/conv/wire demands are UPPER bounds — but the
    send panel (max gemm rows on one owner) can EXCEED the matrix bound,
    because replication concentrates rows at token homes; bound it by
    2 * T * topk (the plan's balance objective with 2x slack). The driver
    sets the EXACT FLUX_A2AV_RS_MAX_* knobs from the plan itself
    (setdefault) — never pre-set them here (the moonep_fused contract)."""
    cb = int(spec["chunk_bytes"])
    topk = int(spec["topk"])
    L = plat["ranks_per_node"]
    chunks, u, U, T = matrix_dedup_stats(matrix, spec, plat, routing_mode)
    d = gen_matrix.a2av_rs_knob_demands(chunks, U, L)
    cpr = T * topk
    send_bound = 2 * cpr
    if v.get("l1_pattern") == "a2av_hier_compress":
        rows = send_bound + cpr + d["rs_conv"] + d["rs_wire"]
    else:
        rows = send_bound + cpr + d["rs_stage"]
    sym_g = max(6, math.ceil(rows * cb / (1 << 30)) + 1)
    sym_max = plat.get("sym_size_max_g")
    if sym_max:
        sym_g = min(sym_g, int(sym_max))
    return f"{sym_g}G"


def moonep_fused_sym_size(matrix_path, plat, spec):
    """Symmetric-heap sizing for the moonep_fused arm: the fused op's a2av
    buffers (send S*K rows; recv/stage/relay bounded by W*S, (NN-1)*S and
    (NN-1)*S rows respectively -- union regions never exceed S unique tokens
    per (source, dest) and the driver sets the EXACT FLUX_A2AV_MAX_* knobs
    from the plan, always <= these bounds) plus the permanent contiguous
    [epn+B, ffn_shard, H] weight tensor (B = epn default => 2*epn rows of
    ffn*chunk bytes). 2x headroom, floor 2G, plat cap."""
    with open(matrix_path) as f:
        toks = f.read().split()
    w = int(toks[0])
    vals = [int(x) for x in toks[1 : 1 + w * w]]
    chunk = int(spec["chunk_bytes"])
    topk = int(spec["topk"])
    n_entries = max(sum(vals[s * w : (s + 1) * w]) for s in range(w)) // chunk  # S*K
    s_tokens = n_entries // topk
    nn = int(spec["nodes"])
    a2av_rows = n_entries + w * s_tokens + 2 * max(nn - 1, 0) * s_tokens
    weights = 2 * (int(spec["G"]) // w) * int(spec["ffn_hidden"]) * chunk
    sym_g = max(2, math.ceil(2 * (a2av_rows * chunk + weights) / (1 << 30)))
    sym_max = plat.get("sym_size_max_g")
    if sym_max:
        sym_g = min(sym_g, int(sym_max))
    return f"{sym_g}G"


def eplb_sym_size(matrix_path, plat, spec):
    """Symmetric-heap sizing for the eplb nvshmem arm. Same two
    All2AllSingle ops as ultraep (no dedup: probs splits == row splits), but
    ultraep's domain-slice bound is UNSAFE here: EPLB's global policy
    re-homes experts to ANY rank, so a source's rows toward one dest are not
    bounded by its logical traffic to that dest's domain. Use the trivially
    safe row-sum bound instead: a source can send at most its whole
    post-topk emission (= T*topk rows = the matrix row sum) to one dest.
    2x headroom, floor 2G, capped by plat sym_size_max_g (same policy as the
    other sizers)."""
    with open(matrix_path) as f:
        toks = f.read().split()
    w = int(toks[0])
    vals = [int(x) for x in toks[1 : 1 + w * w]]
    max_pair_bytes = max(
        sum(vals[src * w : (src + 1) * w]) for src in range(w)
    )
    chunk = int(spec["chunk_bytes"])
    hidden = 2 * w * max_pair_bytes                   # in+out staging, row bytes
    probs = 2 * w * (max_pair_bytes // chunk) * 4     # one fp32 per row (no dedup)
    sym_g = max(2, math.ceil(2 * (hidden + probs) / (1 << 30)))
    sym_max = plat.get("sym_size_max_g")
    if sym_max:
        sym_g = min(sym_g, int(sym_max))
    return f"{sym_g}G"


def epic_sym_size(matrix_path, plat, spec, v):
    """Symmetric-heap sizing for the epic nvshmem arm: the eplb row-sum
    bound (global re-homing, no dedup — a source can send its whole
    post-topk emission to one dest) scaled by the runner's All2AllSingle
    split headroom (--a2a_split_headroom, default 2.0): the runner sizes
    max_split = headroom * initial max per-(group, pair) rows, and
    max-pair-rows <= row sum. Floor 2G, plat cap (same policy as eplb).

    hier_compress arms additionally allocate per-group fused-op dispatch
    buffers (send + recv/stage/relay, driver-sized exact via os.environ)
    and per-group TopkReduceScatterOp combine panels — conservatively 2x
    the row-sum basis on top of the All2AllSingle staging (the driver sets
    the exact FLUX_A2AV_MAX_* / FLUX_A2AV_RS_MAX_* knobs itself; the sweep
    must NOT pre-set them for this driver)."""
    headroom = 2.0
    ta = v.get("test_args") or []
    if "--a2a_split_headroom" in ta:
        headroom = float(ta[ta.index("--a2a_split_headroom") + 1])
    with open(matrix_path) as f:
        toks = f.read().split()
    w = int(toks[0])
    vals = [int(x) for x in toks[1 : 1 + w * w]]
    max_pair_bytes = max(
        sum(vals[src * w : (src + 1) * w]) for src in range(w)
    ) * headroom
    chunk = int(spec["chunk_bytes"])
    hidden = 2 * w * max_pair_bytes
    probs = 2 * w * (max_pair_bytes // chunk) * 4
    need = 2 * (hidden + probs)
    if "hier_compress" in ta:
        need += 4 * w * max_pair_bytes  # fused dispatch + combine panels
    sym_g = max(2, math.ceil(need / (1 << 30)))
    sym_max = plat.get("sym_size_max_g")
    if sym_max:
        sym_g = min(sym_g, int(sym_max))
    return f"{sym_g}G"


def build_cell_env(spec, plat, cell, staging, matrix):
    v = VARIANTS[cell["variant"]]
    matrix_path = matrix.get("path")
    env = {}
    env.update(plat.get("env") or {})
    if v.get("driver", "flux") in ("fast", "fast_gather_rs"):
        # FLUX_A2AV_* knobs are meaningless for FAST; its heap follows capacity
        # (the 4*max(row,col sum) bound is transpose-symmetric, so the same
        # sizing serves the layer1 combine direction)
        env["NVSHMEM_SYMMETRIC_SIZE"] = fast_sym_size(matrix_path, plat)
    elif v.get("driver", "flux") == "gather_rs":
        # layer1 flux cells: exact FLUX_A2AV_RS_MAX_*_ROWS + heap from the
        # collective FLUX_CHECK expressions (dispatch-orientation inputs;
        # the transpose lives inside a2av_rs_knob_demands)
        knobs, sym_g_required = exact_rs_scale_knobs(
            matrix, spec, plat, cell.get("routing_mode"), v["comm_pattern"]
        )
        env.update(knobs)
        # popped by run_cell for the skipped_capacity decision (never
        # reaches the child process environment)
        env["_A2AV_SYM_G_REQUIRED"] = str(sym_g_required)
        sym_max = plat.get("sym_size_max_g")
        if sym_max and int(env["NVSHMEM_SYMMETRIC_SIZE"][:-1]) > int(sym_max):
            env["NVSHMEM_SYMMETRIC_SIZE"] = f"{sym_max}G"
    elif v.get("driver", "flux") == "l01":
        # combined cells: BOTH ops allocate on one NVSHMEM heap — merge the
        # layer0 and layer1 exact knob sets and SUM their heap demands. The
        # l0 sizing runs even for the allgather/torch arms (their FLUX_A2AV_*
        # knobs are ignored by non-a2av paths; the heap term still covers the
        # dense gathered input bound).
        l0_knobs, l0_sym_g = exact_scale_knobs(
            matrix, spec, plat, cell.get("routing_mode")
        )
        l1_knobs, l1_sym_g = exact_rs_scale_knobs(
            matrix, spec, plat, cell.get("routing_mode"), v.get("l1_pattern", "dense")
        )
        env.update(l0_knobs)
        env.update(l1_knobs)
        # subtract the double-counted floors/overheads conservatively: keep
        # the plain sum minus one 1G overhead term, floor 6G
        sym_g = max(6, l0_sym_g + l1_sym_g - 1)
        env["NVSHMEM_SYMMETRIC_SIZE"] = f"{sym_g}G"
        env["_A2AV_SYM_G_REQUIRED"] = str(sym_g)
        sym_max = plat.get("sym_size_max_g")
        if sym_max and sym_g > int(sym_max):
            env["NVSHMEM_SYMMETRIC_SIZE"] = f"{sym_max}G"
    elif v.get("driver", "flux") == "moonep_l1":
        # virtual-space layer1 cells: the driver computes the EXACT
        # FLUX_A2AV_RS_MAX_* knobs from the plan (setdefault -- never
        # pre-set); heap from the matrix-derived upper bounds
        env["NVSHMEM_SYMMETRIC_SIZE"] = moonep_l1_sym_size(
            matrix, spec, plat, cell.get("routing_mode"), v
        )
    elif v.get("driver", "flux") == "moonep_fused":
        # the driver computes the EXACT FLUX_A2AV_MAX_* knobs from the plan
        # (setdefault -- never pre-set them here); heap must hold the fused
        # a2av buffers plus the permanent weight tensor
        env["NVSHMEM_SYMMETRIC_SIZE"] = moonep_fused_sym_size(matrix_path, plat, spec)
    elif v.get("driver", "flux") in ("moonep", "ultraep", "eplb", "epic"):
        # no FLUX_A2AV_* scale knobs ever; NVSHMEM heap only for the
        # one-sided-transport arms (All2AllSingle symmetric staging is
        # 2 ops x 2 bufs of max_split*W rows). moonep bounds max_split by
        # the largest per-pair matrix entry (dedup only shrinks it);
        # ultraep is no-dedup and reroute-redirected, so its bound is the
        # domain-summed slice (see ultraep_sym_size); eplb re-homes
        # globally, so only the full row-sum bound is safe (eplb_sym_size).
        ta = v.get("test_args") or []
        if "getmem" in ta and v.get("driver") == "moonep":
            # getmem weight path: heap must also hold the permanent weight
            # home (plus token staging when the token transport is nvshmem)
            env["NVSHMEM_SYMMETRIC_SIZE"] = moonep_getmem_sym_size(
                matrix_path, plat, spec, v
            )
        elif "nvshmem" in ta or "hier_compress" in ta:
            if v.get("driver") == "ultraep":
                env["NVSHMEM_SYMMETRIC_SIZE"] = ultraep_sym_size(
                    matrix_path, plat, spec, v
                )
            elif v.get("driver") == "eplb":
                env["NVSHMEM_SYMMETRIC_SIZE"] = eplb_sym_size(
                    matrix_path, plat, spec
                )
            elif v.get("driver") == "epic":
                env["NVSHMEM_SYMMETRIC_SIZE"] = epic_sym_size(
                    matrix_path, plat, spec, v
                )
            else:
                env["NVSHMEM_SYMMETRIC_SIZE"] = moonep_sym_size(matrix_path, plat)
    else:
        if matrix.get("path") and os.path.exists(matrix["path"]):
            knobs, sym_g_required = exact_scale_knobs(
                matrix, spec, plat, cell.get("routing_mode")
            )
            env.update(knobs)
            # popped by run_cell for the skipped_capacity decision (never
            # reaches the child process environment)
            env["_A2AV_SYM_G_REQUIRED"] = str(sym_g_required)
        else:
            env.update(scale_knobs(cell["budget_mib"], spec["topk"], spec["chunk_bytes"]))
        sym_max = plat.get("sym_size_max_g")
        if sym_max and int(env["NVSHMEM_SYMMETRIC_SIZE"][:-1]) > int(sym_max):
            env["NVSHMEM_SYMMETRIC_SIZE"] = f"{sym_max}G"
    env.update(v["env"])
    if cell["mode"] == "phases":
        env["FLUX_A2AV_TIMING"] = "1"
    if cell["mode"] == "isolated":
        # per-iteration device sync + rank barrier before each timed window
        # (SCHEMA.md: isolated per-layer latency; quote max-across-ranks)
        env["FLUX_SWEEP_ISOLATED_ITERS"] = "1"
    env["FLUX_TEST_DETERMINISTIC"] = "0"
    env["FLUX_SWEEP_RECORD_DIR"] = os.path.join(staging, "records")
    env["FLUX_EXTRA_TORCHRUN_ARGS"] = f"--redirects 3 --log-dir {os.path.join(staging, 'torchrun')}"
    env.update(spec.get("extra_env") or {})
    return env


def cell_launcher(cell, plat, staging):
    """Launcher argv for a flux-driver cell: ./launch.sh, wrapped in the nsys
    capture (with node-local pre-clean) for nsys cells. Shared by the layer0
    tail and the gather_rs early-return branch of build_cell_cmd."""
    if cell["mode"] != "nsys":
        return ["./launch.sh"]
    return [
        # node-local pre-clean: a killed nsys leaves multi-GB quadd session
        # data under /tmp/nvidia; once the root disk fills, every later
        # nsys dies with SIGBUS (mmap write on a full filesystem)
        "bash",
        "-c",
        'rm -rf /tmp/nvidia/nsight_systems /tmp/nsys-report-*.qdstrm; exec "$@"',
        "nsys-preclean",
        plat.get("nsys_bin") or "nsys",
        "profile",
        # job id in the name so a retried cell (same staging dir) never
        # silently overwrites the earlier attempt's capture
        "-o",
        os.path.join(staging, "nsys", "node%q{SLURM_NODEID}_%q{SLURM_JOB_ID}"),
        # NO osrt: the NVSHMEM/EFA proxy thread busy-polls fi_cq_read, and
        # osrt-tracing it is an event storm (~18 GB per minute of capture,
        # fills the node-local root disk and wedges the run)
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--trace-fork-before-exec=true",
        "--force-overwrite=true",
        "./launch.sh",
    ]


def build_cell_cmd(spec, plat, cell, jobid, matrix_path, staging, routing_path=None,
                   eplb_load_path=None):
    v = VARIANTS[cell["variant"]]
    profiling = cell["mode"] in ("torchprof", "nsys")
    iters = spec["profile_iters"] if profiling else spec["iters"]
    warmup = spec["profile_iters"] if profiling else spec["warmup_iters"]
    sm_margin = spec["sm_margin"]
    srun_prefix = [
        "srun",
        f"--jobid={jobid}",
        f"--nodes={spec['nodes']}",
        "--ntasks-per-node=1",
    ] + list(plat.get("srun_extra") or [])
    if v.get("driver", "flux") == "fast":
        test_args = [
            TEST_FAST,
            "--traffic_matrix",
            matrix_path,
            "--topk",
            str(spec["topk"]),
            "--G",
            str(spec["G"]),
            "--H",
            str(spec["H"]),
            "--chunk_bytes",
            str(spec["chunk_bytes"]),
            "--ffn_hidden_size",
            str(spec["ffn_hidden"]),
            "--dtype",
            spec["dtype"],
            "--iters",
            str(iters),
            "--warmup_iters",
            str(warmup),
            "--sm_margin",
            str(sm_margin),
        ]
        if spec["skip_correctness"]:
            test_args.append("--skip_correctness")
        return srun_prefix + ["./launch_fast.sh"] + test_args, sm_margin, iters, warmup
    if v.get("driver", "flux") == "fast_gather_rs":
        # layer1 FAST baseline: same launcher/constraints as the layer0 fast
        # arm, layer1 flag names (-N/-K/-G single-dash)
        test_args = [
            TEST_FAST_GATHER_RS,
            "--traffic_matrix",
            matrix_path,
            "--topk",
            str(spec["topk"]),
            "-G",
            str(spec["G"]),
            "-N",
            str(spec["H"]),
            "-K",
            str(spec["ffn_hidden"]),
            "--chunk_bytes",
            str(spec["chunk_bytes"]),
            "--dtype",
            spec["dtype"],
            "--iters",
            str(iters),
            "--warmup_iters",
            str(warmup),
            "--sm_margin",
            str(sm_margin),
        ]
        if spec["skip_correctness"]:
            test_args.append("--skip_correctness")
        return srun_prefix + ["./launch_fast.sh"] + test_args, sm_margin, iters, warmup
    if v.get("driver", "flux") == "l01":
        # combined layer0+1 continuous bench: full argv built here and
        # returned early (layer1-style single-dash dims; l0/l1 patterns and
        # --impl ride the variant's test_args)
        test_args = [
            TEST_L01,
            "--traffic_matrix",
            matrix_path,
            "--topk",
            str(spec["topk"]),
            "-G",
            str(spec["G"]),
            "-N",
            str(spec["H"]),
            "-K",
            str(spec["ffn_hidden"]),
            "--chunk_bytes",
            str(spec["chunk_bytes"]),
            "--dtype",
            spec["dtype"],
            "--iters",
            str(iters),
            "--warmup_iters",
            str(warmup),
            "--sm_margin",
            str(sm_margin),
            "--n_split",
            str(spec["n_split_l1"]),
        ]
        test_args += list(v.get("test_args") or [])
        if routing_path:
            test_args += ["--routing_file", routing_path]
        if spec["skip_correctness"]:
            test_args.append("--skip_correctness")
        if cell["mode"] == "torchprof":
            test_args.append("--profile")
        launcher = cell_launcher(cell, plat, staging)
        return srun_prefix + launcher + test_args, sm_margin, iters, warmup
    if v.get("driver", "flux") == "gather_rs":
        # layer1 flux cells: full argv built here and returned early — the
        # generic tail below appends --H/--ffn_hidden_size, which the l1
        # bench parser does not accept. sm_margin: no auto-bump (that rule is
        # layer0-compress-specific); the l1 a2av ladder needs
        # PACK+REDUCE(+PRERED) blocks = 3+3+2 defaults, covered by the spec
        # default 8.
        test_args = [
            TEST_GATHER_RS,
            "--traffic_matrix",
            matrix_path,
            "--comm_pattern",
            v["comm_pattern"],
            "--topk",
            str(spec["topk"]),
            "-G",
            str(spec["G"]),
            "-N",
            str(spec["H"]),
            "-K",
            str(spec["ffn_hidden"]),
            "--chunk_bytes",
            str(spec["chunk_bytes"]),
            "--dtype",
            spec["dtype"],
            "--iters",
            str(iters),
            "--warmup_iters",
            str(warmup),
            "--sm_margin",
            str(sm_margin),
            "--n_split",
            str(spec["n_split_l1"]),
            "--timing_mode",
            cell["timing_mode"],
        ]
        test_args += list(v.get("test_args") or [])
        if routing_path:
            test_args += ["--routing_file", routing_path]
        if spec["skip_correctness"]:
            test_args.append("--skip_correctness")
        if cell["mode"] == "torchprof":
            test_args.append("--profile")
        launcher = cell_launcher(cell, plat, staging)
        return srun_prefix + launcher + test_args, sm_margin, iters, warmup
    if v.get("driver", "flux") == "moonep_l1":
        # moonep virtual-space layer1: gather_rs-style argv (single-dash
        # dims, no --H/--ffn_hidden_size tail); the driver's --comm_pattern
        # is the variant's l1_pattern (comm_pattern stays the cells.csv label)
        test_args = [
            TEST_MOONEP_L1,
            "--traffic_matrix",
            matrix_path,
            "--comm_pattern",
            v["l1_pattern"],
            "--topk",
            str(spec["topk"]),
            "-G",
            str(spec["G"]),
            "-N",
            str(spec["H"]),
            "-K",
            str(spec["ffn_hidden"]),
            "--chunk_bytes",
            str(spec["chunk_bytes"]),
            "--dtype",
            spec["dtype"],
            "--iters",
            str(iters),
            "--warmup_iters",
            str(warmup),
            "--sm_margin",
            str(sm_margin),
            "--n_split",
            str(spec["n_split_l1"]),
            "--timing_mode",
            cell["timing_mode"],
        ]
        test_args += list(v.get("test_args") or [])
        if routing_path:
            test_args += ["--routing_file", routing_path]
        if spec["skip_correctness"]:
            test_args.append("--skip_correctness")
        if cell["mode"] == "torchprof":
            test_args.append("--profile")
        launcher = cell_launcher(cell, plat, staging)
        return srun_prefix + launcher + test_args, sm_margin, iters, warmup
    if (
        v["comm_pattern"] == "a2av_hier_compress"
        and spec["nodes"] > 1
        and v["env"].get("FLUX_A2AV_UNION_BCAST") != "1"
        and v["env"].get("FLUX_A2AV_LB_UNION") != "1"
    ):
        # gather-gateway paths need a free SM for the index_selects; the two
        # union-broadcast modes forward with pure CE puts and are exempt
        # (layer0 flux driver only — gather_rs returned above)
        sm_margin = max(1, sm_margin)
    if v.get("driver", "flux") in ("moonep", "ultraep", "moonep_fused",
                                   "eplb", "epic"):
        # same CLI as the flux driver minus --comm_pattern; variant-specific
        # flags (--transport nvshmem / --overlap_prefetch / --nvl_domain_size
        # / --weight_path / --groups / --migration) ride test_args
        test = {
            "moonep": TEST_MOONEP,
            "ultraep": TEST_ULTRAEP,
            "moonep_fused": TEST_MOONEP_FUSED,
            "eplb": TEST_EPLB,
            "epic": TEST_EPIC,
        }[v["driver"]]
        test_args = [test, "--traffic_matrix", matrix_path]
        test_args += list(v.get("test_args") or [])
        if v["driver"] == "eplb" and eplb_load_path:
            test_args += ["--eplb_load_file", eplb_load_path]
        if v["driver"] == "epic" and eplb_load_path:
            # same pool-oracle sidecar convention as the eplb arm (D7)
            test_args += ["--epic_load_file", eplb_load_path]
    else:
        test_args = [
            TEST,
            "--traffic_matrix",
            matrix_path,
            "--comm_pattern",
            v["comm_pattern"],
        ]
    test_args += [
        "--topk",
        str(spec["topk"]),
        "--G",
        str(spec["G"]),
        "--H",
        str(spec["H"]),
        "--chunk_bytes",
        str(spec["chunk_bytes"]),
        "--ffn_hidden_size",
        str(spec["ffn_hidden"]),
        "--dtype",
        spec["dtype"],
        "--iters",
        str(iters),
        "--warmup_iters",
        str(warmup),
        "--sm_margin",
        str(sm_margin),
    ]
    if routing_path:
        test_args += ["--routing_file", routing_path]
    if spec["skip_correctness"]:
        test_args.append("--skip_correctness")
    if cell["mode"] == "torchprof":
        test_args.append("--profile")
    launcher = cell_launcher(cell, plat, staging)
    return srun_prefix + launcher + test_args, sm_margin, iters, warmup


def run_cell(spec, plat, cell, jobid, matrix, run_dir_staging, dry):
    staging = os.path.join(run_dir_staging, "cells", cell["cell_id"])
    # fail loudly instead of silently downgrading the routing: a trace-family
    # cell without a valid routing_mode, or a "real" cell without its routing
    # artifact, would otherwise run dealer routing while being recorded as a
    # trace measurement (this happened via a retry-path key drop; see the
    # pristine-cell comment in cmd_run)
    if cell.get("family") == "trace" and cell.get("routing_mode") not in ("real", "dealer"):
        raise SystemExit(
            f"{cell['cell_id']}: trace cell has routing_mode="
            f"{cell.get('routing_mode')!r} (expected 'real' or 'dealer') — refusing"
            f" to run with silently degraded routing"
        )
    if cell.get("routing_mode") == "real" and not matrix.get("routing"):
        raise SystemExit(
            f"{cell['cell_id']}: routing_mode=real but the matrix has no routing"
            f" artifact ({matrix.get('path')}) — refusing to run dealer routing"
            f" under a 'real' label"
        )
    cmd, sm_margin, iters, warmup = build_cell_cmd(
        spec,
        plat,
        cell,
        jobid,
        matrix["path"],
        staging,
        routing_path=matrix.get("routing") if cell.get("routing_mode") == "real" else None,
        eplb_load_path=matrix.get("eplb_load"),
    )
    env_delta = build_cell_env(spec, plat, cell, staging, matrix)
    sym_g_required = env_delta.pop("_A2AV_SYM_G_REQUIRED", None)
    sym_max = plat.get("sym_size_max_g")
    if sym_g_required is not None and sym_max and int(sym_g_required) > int(sym_max):
        # exact demand cannot fit the platform's symmetric heap: record the
        # cell as capacity-skipped up front instead of launching a run that
        # is guaranteed to die on an overflow FLUX_CHECK / NVSHMEM init
        print(
            f"WARNING: {cell['cell_id']}: needs {sym_g_required}G symmetric heap"
            f" > platform cap {sym_max}G -> skipped_capacity"
        )
        return dict(
            cell,
            status="skipped_capacity",
            sm_margin=sm_margin,
            iters=iters,
            warmup=warmup,
            env_delta=env_delta,
            staging="",
            exit_code=None,
        )
    if dry:
        print(f"\n[{cell['cell_id']}]")
        print("  env: " + " ".join(f"{k}={v}" for k, v in sorted(env_delta.items())))
        print("  cmd: " + " ".join(cmd))
        return dict(
            cell,
            status="dry",
            sm_margin=sm_margin,
            iters=iters,
            warmup=warmup,
            env_delta=env_delta,
            staging=staging,
            exit_code=None,
        )
    for sub in ("records", "torchrun", "nsys"):
        os.makedirs(os.path.join(staging, sub), exist_ok=True)
    if cell["mode"] == "torchprof":
        # group_profile names its artifact from TORCHELASTIC_RUN_ID, which is
        # constant ("none") under launch.sh — a leftover from a crashed cell
        # would be collected by THIS cell below; clear it first
        for stale in glob.glob(os.path.join(REPO_ROOT, "prof", "moe_ag_scatter_traffic_*")):
            if os.path.isdir(stale):
                shutil.rmtree(stale)
            else:
                os.remove(stale)
    env = dict(os.environ)
    env.update(env_delta)
    start = time.time()
    print(f"[{cell['cell_id']}] start", flush=True)
    with open(os.path.join(staging, "srun.log"), "w") as logf:
        logf.write("+ " + " ".join(cmd) + "\n")
        logf.write("+ env " + json.dumps(env_delta, sort_keys=True) + "\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group so a kill reaps srun + children
        )
        status = _wait_with_watchdog(
            proc, staging, start, spec["timeout_s"], spec.get("idle_timeout_s")
        )
        exit_code = proc.returncode
        if status == "ok" and exit_code != 0:
            status = "failed"
        if status in ("timeout", "stuck"):
            exit_code = None
            with open(os.path.join(staging, "srun.log"), "a") as f:
                f.write(f"\n+ killed by runner: {status}\n")
    if cell["mode"] == "torchprof":
        # flux.group_profile writes chrome traces under cwd prof/; move can
        # cross filesystems (repo on /global home, staging on $PSCRATCH), so
        # shutil.move, not os.renames
        for d in glob.glob(os.path.join(REPO_ROOT, "prof", "moe_ag_scatter_traffic_*")):
            dest = os.path.join(staging, "prof", os.path.basename(d))
            if not os.path.exists(dest):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.move(d, dest)
    print(f"[{cell['cell_id']}] {status} ({time.time() - start:.0f}s)")
    return dict(
        cell,
        status=status,
        sm_margin=sm_margin,
        iters=iters,
        warmup=warmup,
        env_delta=env_delta,
        staging=staging,
        exit_code=exit_code,
        start_ts=start,
        end_ts=time.time(),
    )


def _latest_mtime(root, floor):
    latest = floor
    for dirpath, _, files in os.walk(root):
        for fn in files:
            try:
                latest = max(latest, os.stat(os.path.join(dirpath, fn)).st_mtime)
            except OSError:
                pass
    return latest


def _kill_group(proc):
    # SIGINT first with a generous grace: nsys traps it and finalizes the
    # .qdstrm -> .nsys-rep (a SIGKILL'd nsys loses the whole capture), and
    # torchrun forwards it cleanly to the ranks in every mode
    for sig, grace in ((signal.SIGINT, 30), (signal.SIGTERM, 15), (signal.SIGKILL, 10)):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def _wait_with_watchdog(proc, staging, start, timeout_s, idle_timeout_s):
    """Wait for the cell; kill it on absolute timeout or when its logs stop
    growing for idle_timeout_s (hung ranks — e.g. the per-rank recv-overflow
    FLUX_CHECK leaves the other ranks spinning at 100% GPU forever)."""
    while True:
        try:
            proc.wait(timeout=5)
            return "ok"
        except subprocess.TimeoutExpired:
            pass
        now = time.time()
        if timeout_s and now - start > timeout_s:
            _kill_group(proc)
            return "timeout"
        if idle_timeout_s and now - _latest_mtime(staging, start) > idle_timeout_s:
            _kill_group(proc)
            return "stuck"


def read_records(staging):
    """Parse the per-rank recorder JSONLs -> (per-rank meta, iters rows, info, correctness)."""
    metas, iters_rows, info, correctness = {}, [], {}, {}
    for path in sorted(glob.glob(os.path.join(staging, "records", "rank_*.jsonl"))):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                t = rec.get("type")
                if t == "meta":
                    metas[rec["rank"]] = rec
                elif t == "iters":
                    rank = int(os.path.basename(path)[5:8])
                    for i, val in enumerate(rec["values_ms"]):
                        iters_rows.append((rec["impl"], rank, i, rec["metric"], val))
                elif t == "cell_info":
                    info.update({k: v for k, v in rec.items() if k != "type"})
                elif t == "correctness":
                    rank = int(os.path.basename(path)[5:8])
                    correctness[rank] = (rec["bitwise"], rec["allclose"])
    return metas, iters_rows, info, correctness


def parse_phase_logs(staging, iters):
    """Scan torchrun per-rank stderr files for the [a2av-*] marks; keep the
    last `iters` occurrences per (rank, pattern-family) — earlier ones are
    warmup. Rank identity comes from the in-line `rank %d`, never file paths."""
    hits = {}  # (pattern_idx, rank) -> [ [values...], ... ]
    for path in glob.glob(os.path.join(staging, "torchrun", "**", "*"), recursive=True):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, errors="replace") as f:
                content = f.read()
        except OSError:
            continue
        for pi, (pat, _, _) in enumerate(PHASE_PATTERNS):
            for m in pat.finditer(content):
                rank = int(m.group(1))
                vals = [float(x) for x in m.groups()[1:]]
                hits.setdefault((pi, rank), []).append(vals)
    rows = []
    for (pi, rank), occurrences in hits.items():
        _, names, scale = PHASE_PATTERNS[pi]
        kept = occurrences[-iters:]
        for i, vals in enumerate(kept):
            for name, val in zip(names, vals):
                rows.append(("flux", rank, i, name, val * scale))
    return rows


def finalize(spec, plat, cells_done, matrices, run_id, run_dir_staging, probe, sos):
    capsule = os.path.join(RESULTS_ROOT, run_id)
    os.makedirs(capsule, exist_ok=True)
    git_sha, git_dirty = git_info()
    metrics_rows, cells_rows, artifacts = [], [], []

    for cell in cells_done:
        m = matrices[cell["cell_id"]]
        row = {c: "" for c in CELLS_COLUMNS}
        row.update(
            run_id=run_id,
            cell_id=cell["cell_id"],
            status=cell["status"],
            platform=plat["name"],
            variant=cell["variant"],
            comm_pattern=VARIANTS[cell["variant"]]["comm_pattern"],
            mode=cell["mode"],
            matrix_id=m["id"],
            matrix_path=m["path"],
            matrix_sha256=m["sha"],
            family=cell["family"],
            family_params=json.dumps(cell["family_params"], sort_keys=True),
            routing_mode=cell.get("routing_mode", ""),
            routing_path=m.get("routing", ""),
            routing_sha256=m.get("routing_sha", ""),
            budget_mib=cell["budget_mib"],
            topk=spec["topk"],
            G=spec["G"],
            H=spec["H"],
            chunk_bytes=spec["chunk_bytes"],
            dtype=spec["dtype"],
            world_size=cell["world_size"],
            nnodes=spec["nodes"],
            ranks_per_node=plat["ranks_per_node"],
            fabric=plat["fabric"],
            iters=cell["iters"],
            warmup_iters=cell["warmup"],
            sm_margin=cell["sm_margin"],
            env_json=json.dumps(cell["env_delta"], sort_keys=True),
            git_sha=git_sha,
            git_dirty=int(git_dirty),
            exit_code="" if cell.get("exit_code") is None else cell["exit_code"],
            start_ts=(
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(cell["start_ts"]))
                if cell.get("start_ts")
                else ""
            ),
            end_ts=(
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(cell["end_ts"]))
                if cell.get("end_ts")
                else ""
            ),
            log_dir=cell.get("staging", ""),
            notes="; ".join(x for x in (spec["notes"], cell.get("cell_note")) if x),
            layer=cell.get("layer", "l0"),
            timing_mode=cell.get("timing_mode", ""),
        )
        if cell["status"] in ("ok", "failed", "timeout"):
            metas, iters_rows, info, correctness = read_records(cell["staging"])
            if cell["status"] == "ok" and not metas:
                row["status"] = cell["status"] = "failed"  # exited 0 but recorded nothing
            for impl, rank, i, metric, val in iters_rows:
                metrics_rows.append(
                    (
                        run_id,
                        cell["cell_id"],
                        cell["mode"],
                        impl,
                        rank,
                        i,
                        metric,
                        round(val, 6),
                        "recorder",
                    )
                )
            if cell["mode"] == "isolated":
                # console-only quoted summary (SCHEMA: aggregation is the
                # summarizer's job; nothing persisted): per-iteration
                # max-across-ranks of e2e_ms, then stats over iterations
                by_iter = {}
                for impl, rank, i, metric, val in iters_rows:
                    if impl == "flux" and metric == "e2e_ms":
                        by_iter[i] = max(by_iter.get(i, 0.0), val)
                if by_iter:
                    mx = [by_iter[i] for i in sorted(by_iter)]
                    print(
                        f"  [{cell['cell_id']}] isolated max-rank e2e_ms: "
                        f"mean {sum(mx) / len(mx):.3f}  min {min(mx):.3f}  "
                        f"max {max(mx):.3f}  ({len(mx)} iters)"
                    )
            if cell["mode"] == "phases":
                for impl, rank, i, metric, val in parse_phase_logs(cell["staging"], cell["iters"]):
                    metrics_rows.append(
                        (
                            run_id,
                            cell["cell_id"],
                            cell["mode"],
                            impl,
                            rank,
                            i,
                            metric,
                            round(val, 6),
                            "stderr",
                        )
                    )
            if metas:
                row["deterministic"] = int(all(m["deterministic"] for m in metas.values()))
            if correctness:
                row["correct_bitwise"] = int(all(b for b, _ in correctness.values()))
                row["correct_allclose"] = int(all(a for _, a in correctness.values()))
            for k_src, k_dst in [
                ("ntokens", "ntokens"),
                ("tokens_per_rank", "tokens_per_rank"),
                ("wire_ratio", "wire_ratio"),
                ("relay_ident_bytes", "relay_ident_bytes"),
                ("relay_balanced_bytes", "relay_balanced_bytes"),
            ]:
                if k_src in info:
                    row[k_dst] = info[k_src]
            for p in sorted(glob.glob(os.path.join(cell["staging"], "records", "*.jsonl"))):
                artifacts.append({"path": p, "sha256": sha256_file(p), "bytes": os.path.getsize(p)})
            nsys_reps = sorted(glob.glob(os.path.join(cell["staging"], "nsys", "*.nsys-rep")))
            if nsys_reps:
                row["nsys_path"] = os.path.dirname(nsys_reps[0])
                for p in nsys_reps:
                    artifacts.append(
                        {"path": p, "sha256": sha256_file(p), "bytes": os.path.getsize(p)}
                    )
            if cell["mode"] == "nsys" and cell["status"] == "ok":
                # guard: one full-size rep per node, else the capture was lost
                # (killed nsys, wrong output path, empty trace) — never report
                # a profiling cell ok without its profile
                good = [p for p in nsys_reps if os.path.getsize(p) >= NSYS_REP_MIN_BYTES]
                if len(good) < spec["nodes"]:
                    row["status"] = cell["status"] = "nsys_empty"
            profs = sorted(glob.glob(os.path.join(cell["staging"], "prof", "*")))
            if profs:
                row["prof_path"] = os.path.join(cell["staging"], "prof")
                for p in profs:
                    if os.path.isfile(p):
                        artifacts.append(
                            {"path": p, "sha256": sha256_file(p), "bytes": os.path.getsize(p)}
                        )
        cells_rows.append(row)

    with open(os.path.join(capsule, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(METRICS_COLUMNS)
        w.writerows(metrics_rows)
    with open(os.path.join(capsule, "cells.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CELLS_COLUMNS)
        w.writeheader()
        w.writerows(cells_rows)
    dump_yaml(spec, os.path.join(capsule, "spec.yaml"))
    manifest = {
        "run_id": run_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": {k: plat[k] for k in ("name", "ranks_per_node", "fabric")},
        "host": os.uname().nodename,
        "git": {"sha": git_sha, "dirty": git_dirty},
        "flux_libs": [{"path": s, "sha256": sha256_file(s)} for s in sos],
        "capabilities": probe,
        "staging_root": run_dir_staging,
        "cells": [
            {"cell_id": c["cell_id"], "status": c["status"], "staging": c.get("staging", "")}
            for c in cells_done
        ],
        "artifacts": artifacts,
    }
    with open(os.path.join(capsule, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return capsule, metrics_rows, cells_rows


def cmd_run(spec, jobid_arg, dry):
    plat = load_platform(spec["platform"], dry=dry)
    if not dry and not os.environ.get("NVSHMEM_HOME"):
        raise SystemExit(
            "NVSHMEM_HOME unset — source the platform env first"
            " (env_aws.sh on AWS, module.sh on Perlmutter)"
        )
    for m in spec["modes"]:
        if m not in MODES:
            raise SystemExit(f"unknown mode {m}; choose from {MODES}")
    nsys_bin = plat.get("nsys_bin") or "nsys"
    if "nsys" in spec["modes"] and not dry and not shutil.which(nsys_bin):
        raise SystemExit(
            f"modes include nsys but {nsys_bin} not found — source the platform"
            " env (env_aws.sh / module.sh), set nsys_bin in the platform yaml,"
            " or drop the nsys mode"
        )
    cells = expand_cells(spec, plat)

    # matrices (generate-if-missing, sha-verified)
    matrices = {}
    if dry and "$" in plat["matrices_root"]:
        for cell in cells:
            # dealer is an arm marker, not a matrix param (same bytes both arms);
            # trace ids are approximate here (poolsha needs the trace files)
            mparams = {k: v for k, v in cell["family_params"].items() if k != "dealer"}
            mid = gen_matrix.matrix_id_of(
                cell["family"],
                dict(gen_matrix.FAMILY_DEFAULT_PARAMS[cell["family"]], **mparams),
                cell["world_size"],
                plat["ranks_per_node"],
                cell["budget_mib"],
                spec["topk"],
                spec["chunk_bytes"],
                spec["matrix_instance"],
            )
            matrices[cell["cell_id"]] = {
                "id": mid,
                "path": os.path.join(plat["matrices_root"], f"{mid}.txt"),
                "sha": "",
            }
    else:
        os.makedirs(plat["matrices_root"], exist_ok=True)
        for cell in cells:
            mparams = {k: v for k, v in cell["family_params"].items() if k != "dealer"}
            mid, path, sha = gen_matrix.ensure_matrix(
                cell["family"],
                mparams,
                cell["world_size"],
                plat["ranks_per_node"],
                cell["budget_mib"],
                spec["topk"],
                spec["chunk_bytes"],
                spec["matrix_instance"],
                plat["matrices_root"],
                nexperts=spec["G"],
                traces_root=plat.get("traces_root"),
            )
            matrices[cell["cell_id"]] = {"id": mid, "path": path, "sha": sha}
            if cell.get("routing_mode") == "real":
                rpath = path[: -len(".txt")] + ".routing.txt"
                with open(os.path.join(plat["matrices_root"], f"{mid}.meta.json")) as f:
                    rsha = json.load(f)["routing_sha256"]
                matrices[cell["cell_id"]].update({"routing": rpath, "routing_sha": rsha})
            if VARIANTS[cell["variant"]].get("driver") in ("eplb", "epic"):
                # predicted-load sidecar: the cell's exact full pools (the
                # oracle-ceiling prediction). Placement input only — matrix
                # identity is unchanged; the driver records the sha as a
                # cell fact.
                if cell["family"] == "trace":
                    lpath, lsha = gen_trace_routing.ensure_eplb_load(
                        dict(gen_matrix.FAMILY_DEFAULT_PARAMS["trace"], **mparams),
                        cell["world_size"],
                        plat["ranks_per_node"],
                        cell["budget_mib"],
                        spec["topk"],
                        spec["chunk_bytes"],
                        spec["matrix_instance"],
                        plat["matrices_root"],
                        traces_root=plat.get("traces_root"),
                        nexperts=spec["G"],
                    )
                    matrices[cell["cell_id"]].update(
                        {"eplb_load": lpath, "eplb_load_sha": lsha}
                    )
                else:
                    print(
                        f"NOTE: {cell['cell_id']}: eplb on a non-trace family"
                        " has no pool prediction — the driver falls back to"
                        " the batch's own load (self-oracle)"
                    )

    needed = sorted({k for v in spec["variants"] for k in VARIANTS[v]["requires"]})
    try:
        probe, sos = probe_capabilities(needed)
    except SystemExit as e:
        if not dry:
            raise
        print(f"WARNING (dry-run): capability probe unavailable ({e}); assuming all capable")
        probe, sos = {k: True for k in needed}, []
    # file-existence capabilities (e.g. FAST's libflash.so): probed by path,
    # recorded in the manifest, and the binary is hashed alongside the flux libs
    for rf in sorted({VARIANTS[v].get("requires_file") for v in spec["variants"]} - {None}):
        rf_abs = os.path.join(REPO_ROOT, rf)
        probe[rf] = os.path.isfile(rf_abs)
        if probe[rf]:
            sos.append(rf_abs)
    if "nsys" in spec["modes"]:
        # head-node view only (the empty-capture guard in finalize is the
        # authoritative check); recorded in the manifest capabilities
        found = shutil.which(plat.get("nsys_bin") or "nsys")
        probe["nsys"] = sh([found, "--version"]).stdout.strip() if found else False
    runnable, skipped = [], []
    for cell in cells:
        v = VARIANTS[cell["variant"]]
        missing = [k for k in v["requires"] if not probe.get(k)]
        rf = v.get("requires_file")
        if rf and not probe.get(rf):
            missing.append(rf)
        if missing:
            print(f"WARNING: {cell['cell_id']}: build lacks {missing} -> skipped_capability")
            skipped.append(
                dict(
                    cell,
                    status="skipped_capability",
                    sm_margin=spec["sm_margin"],
                    iters=spec["iters"],
                    warmup=spec["warmup_iters"],
                    env_delta={},
                    staging="",
                    exit_code=None,
                )
            )
        else:
            runnable.append(cell)

    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    digest = hashlib.sha256(
        (
            json.dumps(spec, sort_keys=True)
            + git_info()[0]
            + os.uname().nodename
            + str(time.time_ns())
        ).encode()
    ).hexdigest()[:8]
    run_id = f"{run_id}_{spec['platform']}_{digest}"
    run_dir_staging = os.path.join(plat["data_root"], run_id)

    jobid = None if dry else detect_jobid(jobid_arg)
    print(f"run_id: {run_id}")
    print(f"cells: {len(runnable)} runnable, {len(skipped)} skipped_capability")
    done = list(skipped)
    # pristine cell dicts for the retry pass: a retry must re-run the EXACT
    # cell as expanded, not a reconstruction (a key whitelist here once
    # dropped routing_mode, silently downgrading retried trace cells to
    # dealer routing while cells.csv siblings said "real")
    pristine = {c["cell_id"]: dict(c) for c in runnable}
    for cell in runnable:
        done.append(
            run_cell(spec, plat, cell, jobid, matrices[cell["cell_id"]], run_dir_staging, dry)
        )
    # one-shot (spec-configurable) retry of stuck/timeout/failed cells: the old
    # staging is preserved as <cell>.attemptN for forensics, the retry runs fresh
    if not dry:
        for attempt in range(1, int(spec.get("retries") or 0) + 1):
            bad_idx = [i for i, c in enumerate(done) if c["status"] in RETRY_STATUSES]
            if not bad_idx:
                break
            print(f"\nretry pass {attempt}: {len(bad_idx)} cells", flush=True)
            for i in bad_idx:
                old = done[i]
                if old.get("staging") and os.path.isdir(old["staging"]):
                    os.rename(old["staging"], f"{old['staging']}.attempt{attempt - 1}")
                fresh = dict(pristine[old["cell_id"]])
                redo = run_cell(
                    spec, plat, fresh, jobid, matrices[old["cell_id"]], run_dir_staging, dry
                )
                if redo["status"] == "ok":
                    redo["cell_note"] = f"recovered on retry {attempt} (was {old['status']})"
                else:
                    redo["cell_note"] = f"{old['status']}, retry {attempt}: {redo['status']}"
                done[i] = redo
    if dry:
        print("\n--dry-run: nothing executed, no capsule written")
        return
    capsule, metrics_rows, cells_rows = finalize(
        spec, plat, done, matrices, run_id, run_dir_staging, probe, sos
    )
    n_ok = sum(1 for r in cells_rows if r["status"] == "ok")
    print(f"\ncapsule: {capsule}")
    print(f"cells: {n_ok}/{len(cells_rows)} ok, metrics rows: {len(metrics_rows)}")
    bad = [r for r in cells_rows if r["status"] != "ok"]
    if bad:
        print("NON-OK CELLS:")
        for r in bad:
            print(f"  {r['cell_id']}: {r['status']}  {r['notes']}")
    print("\nto persist:")
    rel = os.path.relpath(capsule, REPO_ROOT)
    print(f"  git add {rel} && git commit -m 'sweep: {run_id} {spec['notes']}'".rstrip())


def parse_list(s):
    return [x for x in s.split(",") if x]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)
    rp = sub.add_parser("run", help="run a sweep")
    rp.add_argument("--spec", help="YAML spec file; explicit flags override its fields")
    rp.add_argument("--platform", choices=["aws", "perlmutter"])
    rp.add_argument("--jobid", help="Slurm allocation (autodetect if exactly one RUNNING)")
    rp.add_argument("--nodes", type=int)
    rp.add_argument("--variants", type=parse_list)
    rp.add_argument("--families", type=parse_list, help="e.g. remotefrac,hotcol:frac=0.7")
    rp.add_argument(
        "--budgets-mib", dest="budgets_mib", type=lambda s: [int(x) for x in parse_list(s)]
    )
    rp.add_argument("--topk", type=int)
    rp.add_argument("--G", type=int)
    rp.add_argument("--iters", type=int)
    rp.add_argument("--warmup-iters", dest="warmup_iters", type=int)
    rp.add_argument("--sm-margin", dest="sm_margin", type=int)
    rp.add_argument("--modes", type=parse_list)
    rp.add_argument(
        "--profile-iters",
        dest="profile_iters",
        type=int,
        help="iters AND warmup for torchprof/nsys cells (default 3)",
    )
    rp.add_argument("--matrix-instance", dest="matrix_instance")
    rp.add_argument(
        "--skip-correctness", dest="skip_correctness", action="store_true", default=None
    )
    rp.add_argument("--timeout-s", dest="timeout_s", type=int)
    rp.add_argument("--notes")
    rp.add_argument("--dry-run", action="store_true")
    rr = sub.add_parser("rerun", help="re-execute a capsule's spec")
    rr.add_argument("capsule", help="sweeps/results/runs/<run_id>")
    rr.add_argument("--jobid")
    rr.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.command == "rerun":
        spec = load_yaml(os.path.join(args.capsule, "spec.yaml"))
        merged = dict(SPEC_DEFAULTS, **spec)
        cmd_run(merged, args.jobid, args.dry_run)
        return

    spec = dict(SPEC_DEFAULTS)
    if args.spec:
        spec.update(load_yaml(args.spec))
    for key in SPEC_DEFAULTS:
        val = getattr(args, key, None)
        if val is not None:
            spec[key] = val
    if not spec["platform"]:
        raise SystemExit("--platform (or a spec with platform:) is required")
    cmd_run(spec, args.jobid, args.dry_run)


if __name__ == "__main__":
    main()
