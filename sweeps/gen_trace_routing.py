#!/usr/bin/env python3
"""The `trace` matrix family: batches sampled from real MoE routing traces.

Unlike the synthetic families in gen_matrix.py, `trace` produces TWO artifacts
per matrix_id under matrices_root:

    <matrix_id>.txt          — the derived [W][W] byte matrix (diagonal KEPT:
                               real tokens route to experts owned by their own
                               rank; wire bytes = row_sum - diag)
    <matrix_id>.routing.txt  — the per-token expert assignments that produced
                               it. line 1: "ntokens topk G"; then one token per
                               line (topk space-separated expert ids). Token t
                               is homed on rank t // (ntokens // W).

The bench must be run with --routing_file so the REAL token-overlap structure
is used; feeding only the byte matrix through the synthetic dealer
(traffic_matrix_to_choosed_experts) yields the MAXIMUM-dedup assignment and
overstates every dedup/union win (that pairing is available deliberately, as
the `dealer=1` arm — same bytes, max-dedup counterfactual).

Params (family string, e.g.
  "trace:pools=mmlu/college_mathematics+mmlu/philosophy;layer=61;sem=pernode"):
    pools — '+'-joined bench/subject pool specs fetched by fetch_traces.py.
            Order is SEMANTIC for sem=pernode (node i draws from pools[i]).
    layer — MoE layer index whose expert selections are replayed (int).
    sem   — homog  : exactly 1 pool, all ranks sample from it (topic-
                     homogeneous batch; receiver-side skew)
            pernode: len(pools) == nnodes, node i's ranks sample pools[i]
                     (per-node topic affinity; exporter-side asymmetry)
            mixed  : >= 2 pools, every rank samples the concatenation
    pool  — decode | prefill | all: which trace tokens form the pool.
    model — short model name from fetch_traces.MODEL_PREFIXES.
    poolsha — INJECTED (never spec-authored): first 12 hex of the combined
            content fingerprint of the referenced pools, so matrix identity is
            a pure function of the actual trace bytes.

Budget semantics are unchanged: tokens_per_rank T = budget_mib*2^20/chunk_bytes
pre-topk tokens per source rank, sampled with replacement from the pool (the
pool is a distribution, not a corpus; multiplicity is recorded in meta). Every
token contributes exactly topk distinct experts, so row sums are exactly
T*topk chunks and per-(src,expert) counts are <= T (always dealer-feasible).

Stdlib only — runs on login/head nodes without torch.
"""

import hashlib
import json
import os
import random

import gen_matrix
from fetch_traces import MODEL_PREFIXES

SEMS = ("homog", "pernode", "mixed")
POOLS = ("decode", "prefill", "all")

POOL_CACHE_VERSION = 1


def load_manifest(sdir):
    mpath = os.path.join(sdir, "pool.manifest.json")
    if not os.path.exists(mpath):
        raise SystemExit(
            f"missing {mpath} — fetch the pool first: python sweeps/fetch_traces.py"
            f" --pool <bench>/<subject>"
        )
    with open(mpath) as f:
        return json.load(f)


def _extract_rows(trace, layer, pool):
    """Yield expert-id rows for one query's trace (list of per-output-token
    dicts keyed by layer id; token 0 = prefill [n_input][topk], tokens 1+ =
    decode [1][topk])."""
    key = str(layer)
    for ti, tok in enumerate(trace):
        if key not in tok:
            raise ValueError(f"layer {layer} missing at output token {ti}")
        sel = tok[key]
        if sel is None:
            raise ValueError(f"layer {layer} is not a MoE layer (null selection)")
        is_prefill = ti == 0
        if (pool == "decode" and is_prefill) or (pool == "prefill" and not is_prefill):
            continue
        for row in sel:
            yield row


def _cache_path_and_header(sdir, manifest, layer, pool):
    cache_dir = os.path.join(sdir, "pool_cache")
    cache = os.path.join(cache_dir, f"layer{layer}_{pool}.txt")
    header = f"# v{POOL_CACHE_VERSION} pool_sha {manifest['pool_sha']} layer {layer} pool {pool}"
    return cache, header


def _write_cache(cache, header, rows):
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = cache + ".tmp"
    with open(tmp, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(" ".join(str(e) for e in r) + "\n")
    os.rename(tmp, cache)


def build_layer_caches(sdir, layers, pool="decode"):
    """Parse every trace JSON ONCE and write the per-layer caches for all
    requested layers (the per-layer path re-reads ~183 MiB of JSON per layer;
    a full 94-layer scan needs this bulk pass). layers=None = every MoE layer
    found in the first query."""
    assert pool in POOLS
    manifest = load_manifest(sdir)
    if layers is not None:
        todo = [
            ly
            for ly in layers
            if not _cache_valid(*_cache_path_and_header(sdir, manifest, ly, pool))
        ]
        if not todo:
            return
    by_layer = {}
    for finfo in manifest["files"]:
        fpath = os.path.join(sdir, finfo["name"])
        with open(fpath) as f:
            trace = json.load(f)
        if layers is None:
            layers = sorted(
                (int(k) for k, v in trace[0].items() if v is not None),
            )
            todo = [
                ly
                for ly in layers
                if not _cache_valid(*_cache_path_and_header(sdir, manifest, ly, pool))
            ]
            if not todo:
                return
        for ly in todo:
            try:
                by_layer.setdefault(ly, []).extend(
                    tuple(r) for r in _extract_rows(trace, ly, pool)
                )
            except ValueError as e:
                raise SystemExit(f"malformed trace {fpath}: {e}") from e
    for ly in todo:
        cache, header = _cache_path_and_header(sdir, manifest, ly, pool)
        _write_cache(cache, header, by_layer[ly])


def _cache_valid(cache, header):
    if not os.path.exists(cache):
        return False
    with open(cache) as f:
        return f.readline().rstrip("\n") == header


def load_layer_pool(sdir, layer, pool="decode"):
    """Return the [nrows][topk] expert-id pool for one (subject, layer), read
    through a small text cache (parsing 100 x ~2 MB JSONs takes ~10 s; the
    cache is ~500 KB and keyed by the manifest's pool_sha)."""
    assert pool in POOLS, f"pool must be one of {POOLS}, got {pool}"
    manifest = load_manifest(sdir)
    cache_dir = os.path.join(sdir, "pool_cache")
    cache = os.path.join(cache_dir, f"layer{layer}_{pool}.txt")
    header = f"# v{POOL_CACHE_VERSION} pool_sha {manifest['pool_sha']} layer {layer} pool {pool}"
    if os.path.exists(cache):
        with open(cache) as f:
            first = f.readline().rstrip("\n")
            if first == header:
                rows = [tuple(int(x) for x in line.split()) for line in f]
                if rows:
                    return rows
        # stale (pool_sha changed) or empty — fall through and rebuild

    rows = []
    for finfo in manifest["files"]:
        fpath = os.path.join(sdir, finfo["name"])
        with open(fpath) as f:
            trace = json.load(f)
        try:
            for row in _extract_rows(trace, layer, pool):
                topk = len(row)
                if len(set(row)) != topk:
                    raise ValueError(f"duplicate expert within a token's topk: {row}")
                if min(row) < 0:
                    raise ValueError(f"negative expert id: {row}")
                rows.append(tuple(row))
        except ValueError as e:
            raise SystemExit(f"malformed trace {fpath}: {e}") from e
    if not rows:
        raise SystemExit(f"empty pool: {sdir} layer {layer} pool {pool}")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise SystemExit(f"inconsistent topk widths {sorted(widths)} in {sdir} layer {layer}")

    os.makedirs(cache_dir, exist_ok=True)
    tmp = cache + ".tmp"
    with open(tmp, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(" ".join(str(e) for e in r) + "\n")
    os.rename(tmp, cache)
    return rows


def parse_pool_specs(pools_param):
    """'mmlu/college_mathematics+mmlu_ZH_CN/philosophy' -> list of (bench, subject)."""
    specs = []
    for spec in str(pools_param).split("+"):
        bench, _, subject = spec.partition("/")
        if not subject:
            raise SystemExit(f"pool spec must be bench/subject, got: {spec!r}")
        specs.append((bench, subject))
    return specs


def resolve_pools(traces_root, model, pools_param, layer, pool):
    """Return (specs, pools_rows, poolsha12). poolsha is order-independent
    (content identity); pool ORDER is carried by the pools param itself."""
    if not traces_root or "$" in str(traces_root):
        raise SystemExit(f"trace family needs a resolved traces_root, got: {traces_root!r}")
    if model not in MODEL_PREFIXES:
        raise SystemExit(f"unknown model {model!r}; known: {sorted(MODEL_PREFIXES)}")
    specs = parse_pool_specs(pools_param)
    pools_rows, shas = [], []
    for bench, subject in specs:
        sdir = os.path.join(traces_root, MODEL_PREFIXES[model], bench, subject)
        manifest = load_manifest(sdir)
        pools_rows.append(load_layer_pool(sdir, layer, pool))
        shas.append((f"{bench}/{subject}", manifest["pool_sha"]))
    combined = hashlib.sha256(
        "\n".join(f"{spec} {sha}" for spec, sha in sorted(shas)).encode()
    ).hexdigest()
    return specs, pools_rows, combined[:12]


def sample_routing(pools_rows, sem, W, L, T, rng):
    """Sample the [W*T][topk] routing: T tokens per source rank, drawn with
    replacement from the pool(s) in (rank-major, token) order from ONE rng
    stream — fully deterministic given the seed."""
    nn = W // L
    if sem == "homog":
        assert len(pools_rows) == 1, f"sem=homog needs exactly 1 pool, got {len(pools_rows)}"
        pick = lambda s: pools_rows[0]  # noqa: E731
    elif sem == "pernode":
        assert len(pools_rows) == nn, (
            f"sem=pernode needs exactly one pool per node ({nn}), got {len(pools_rows)}"
        )
        pick = lambda s: pools_rows[s // L]  # noqa: E731
    elif sem == "mixed":
        assert len(pools_rows) >= 2, "sem=mixed needs >= 2 pools"
        concat = [r for p in pools_rows for r in p]
        pick = lambda s: concat  # noqa: E731
    else:
        raise SystemExit(f"unknown sem {sem!r}; must be one of {SEMS}")
    routing = []
    for s in range(W):
        pool = pick(s)
        n = len(pool)
        for _ in range(T):
            routing.append(pool[rng.randrange(n)])
    return routing


def derive_matrix(routing, W, T, G, topk):
    """[W][W] chunk counts under contiguous ownership (expert e -> rank
    e // (G//W)); diagonal kept."""
    epr = G // W
    chunks = [[0] * W for _ in range(W)]
    for i, row in enumerate(routing):
        s = i // T
        for e in row:
            chunks[s][e // epr] += 1
    return chunks


def real_dedup_stats(routing, W, L, T, G):
    """The dedup counts the runtime derives from REAL routing (mirrors
    u_mat/U_mat in test_moe_ag_traffic.py): u[s][d] = tokens of source s
    needed by >= 1 expert on rank d; U[s][n] = tokens needed by node n's
    union. Returns (u, U)."""
    epr = G // W
    u = [[0] * W for _ in range(W)]
    U = [[0] * (W // L) for _ in range(W)]
    for i, row in enumerate(routing):
        s = i // T
        dst_ranks = {e // epr for e in row}
        for d in dst_ranks:
            u[s][d] += 1
        for n in {d // L for d in dst_ranks}:
            U[s][n] += 1
    return u, U


def read_routing(path):
    """Inverse of write_routing: returns the [ntokens][topk] expert-id rows."""
    with open(path) as f:
        ntokens, topk, _g = (int(x) for x in f.readline().split()[:3])
        routing = [[int(x) for x in line.split()] for line in f]
    if len(routing) != ntokens or any(len(r) != topk for r in routing):
        raise SystemExit(f"malformed routing file {path}")
    return routing


def write_routing(out_root, matrix_id, routing, topk, G):
    path = os.path.join(out_root, f"{matrix_id}.routing.txt")
    lines = [f"{len(routing)} {topk} {G}"]
    for row in routing:
        lines.append(" ".join(str(e) for e in row))
    body = "\n".join(lines) + "\n"
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(body)
    os.rename(tmp, path)
    return path, hashlib.sha256(body.encode()).hexdigest()


def routing_path_of(matrix_path):
    assert matrix_path.endswith(".txt")
    return matrix_path[: -len(".txt")] + ".routing.txt"


def _sha_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def prepare_trace(params, W, L, topk, matrix_instance, traces_root, nexperts, budget_mib,
                  chunk_bytes):
    """Validate params, resolve+fingerprint the pools, finalize params (poolsha
    injection, mixed-pool canonicalization) and compute the matrix_id. Returns
    (mid, params, specs, pools_rows, T)."""
    params = dict(params)
    for req in ("pools", "layer"):
        if req not in params:
            raise SystemExit(f"trace family requires param {req!r} (e.g. 'trace:pools=mmlu/"
                             f"college_mathematics;layer=61')")
    if nexperts is None:
        raise SystemExit("trace family requires --G (expert ids must map onto G)")
    assert nexperts % W == 0
    sem, pool, model = params["sem"], params["pool"], params["model"]
    layer = int(params["layer"])
    if sem not in SEMS:
        raise SystemExit(f"trace sem must be one of {SEMS}, got {sem!r}")

    specs, pools_rows, poolsha = resolve_pools(traces_root, model, params["pools"], layer, pool)
    if sem == "mixed":
        # order is not semantic for mixed: canonicalize so identity is stable
        params["pools"] = "+".join(f"{b}/{s}" for b, s in sorted(specs))
    params["poolsha"] = poolsha

    trace_topk = len(pools_rows[0][0])
    if trace_topk != topk:
        raise SystemExit(
            f"trace pools have topk={trace_topk} but the sweep asked topk={topk};"
            " subsampling k is not implemented — run with the native topk"
        )
    max_eid = max(max(r) for p in pools_rows for r in p)
    if max_eid >= nexperts:
        raise SystemExit(f"trace expert id {max_eid} >= G ({nexperts}); wrong --G for this model")

    T, _ = gen_matrix.budget_tokens(budget_mib, chunk_bytes, topk)  # pre-topk tokens/rank
    mid = gen_matrix.matrix_id_of(
        "trace", params, W, L, budget_mib, topk, chunk_bytes, matrix_instance
    )
    return mid, params, specs, pools_rows, T


def generate_trace(params, W, L, budget_mib, topk, chunk_bytes, matrix_instance, traces_root,
                   nexperts):
    """Sample routing + derived matrix (no writes). Returns
    (mid, params, specs, pools_rows, routing, chunks, T)."""
    mid, params, specs, pools_rows, T = prepare_trace(
        params, W, L, topk, matrix_instance, traces_root, nexperts, budget_mib, chunk_bytes
    )
    canon = gen_matrix.canonical_string(
        "trace", params, W, L, budget_mib, topk, chunk_bytes, matrix_instance
    )
    rng = random.Random(gen_matrix.fnv1a(canon))
    routing = sample_routing(pools_rows, params["sem"], W, L, T, rng)
    chunks = derive_matrix(routing, W, T, nexperts, topk)
    for s, row in enumerate(chunks):
        assert sum(row) == T * topk, f"internal: row {s} sums {sum(row)} != {T * topk}"
    # always dealer-feasible, but keep the invariant audited
    gen_matrix.check_feasible(chunks, W, topk, T, nexperts=nexperts)
    return mid, params, specs, pools_rows, routing, chunks, T


def ensure_trace_matrix(
    params,
    W,
    L,
    budget_mib,
    topk,
    chunk_bytes,
    matrix_instance,
    out_root,
    traces_root=None,
    nexperts=None,
):
    """Trace-family counterpart of gen_matrix.ensure_matrix. Returns
    (matrix_id, path, sha256, routing_path, routing_sha256)."""
    mid, params, specs, pools_rows, T = prepare_trace(
        params, W, L, topk, matrix_instance, traces_root, nexperts, budget_mib, chunk_bytes
    )
    path = os.path.join(out_root, f"{mid}.txt")
    rpath = os.path.join(out_root, f"{mid}.routing.txt")
    meta_path = os.path.join(out_root, f"{mid}.meta.json")
    if os.path.exists(path) and os.path.exists(rpath) and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        sha, rsha = _sha_file(path), _sha_file(rpath)
        if sha != meta["sha256"] or rsha != meta["routing_sha256"]:
            raise RuntimeError(
                f"trace matrix {mid} does not match its sidecar shas — refusing to"
                " overwrite; delete the .txt/.routing.txt/.meta.json trio to regenerate"
            )
        return mid, path, sha, rpath, rsha

    mid, params, specs, pools_rows, routing, chunks, T = generate_trace(
        params, W, L, budget_mib, topk, chunk_bytes, matrix_instance, traces_root, nexperts
    )

    os.makedirs(out_root, exist_ok=True)
    # routing first: a torn state must never have a matrix without its routing
    rpath, rsha = write_routing(out_root, mid, routing, topk, nexperts)
    diag_chunks = sum(chunks[s][s] for s in range(W))
    meta = {
        "matrix_id": mid,
        "family": "trace",
        "params": dict(params),
        "pool_specs": [f"{b}/{s}" for b, s in specs],
        "pool_rows": [len(p) for p in pools_rows],
        "W": W,
        "ranks_per_node": L,
        "budget_mib": budget_mib,
        "budget_semantics": "pre-topk send budget (nominal label); row_sum_bytes"
        " = effective_budget_bytes*topk, effective = round-to-chunk of"
        " budget_mib*2^20 (exact when chunk divides the budget)",
        "self_traffic_semantics": "diagonal kept (real self-routed tokens);"
        " wire bytes per row = row_sum - diag",
        "diag_bytes_total": diag_chunks * chunk_bytes,
        "sample_multiplicity": round(W * T / sum(len(p) for p in pools_rows), 2),
        "topk": topk,
        "G": nexperts,
        "chunk_bytes": chunk_bytes,
        "tokens_per_rank": T,
        "effective_budget_bytes": T * chunk_bytes,
        "row_sum_bytes": T * chunk_bytes * topk,
        "routing_file": os.path.basename(rpath),
        "routing_sha256": rsha,
        "seed": gen_matrix.fnv1a(
            gen_matrix.canonical_string(
                "trace", params, W, L, budget_mib, topk, chunk_bytes, matrix_instance
            )
        ),
        "matrix_instance": matrix_instance,
        "generator_version": gen_matrix.GENERATOR_VERSION,
    }
    path, sha = gen_matrix.write_matrix(out_root, mid, chunks, chunk_bytes, meta)
    return mid, path, sha, rpath, rsha


EPLB_LOAD_VERSION = 1


def pool_histogram(rows, G):
    """[G] topk-entry counts over one pool's rows."""
    freq = [0] * G
    for row in rows:
        for e in row:
            freq[e] += 1
    return freq


def combine_pool_loads(counts, nrows, sem):
    """Combine per-pool [G] histograms into one predicted load vector,
    mirroring the sampler's expected batch mix (see ensure_eplb_load).
    Returns (load [G] floats, weighting label)."""
    G = len(counts[0])
    if sem == "homog":
        assert len(counts) == 1
        return [float(c) for c in counts[0]], "homog"
    if sem == "pernode":
        load = [0.0] * G
        for hist, n in zip(counts, nrows):
            for e in range(G):
                load[e] += hist[e] / n / len(counts)
        return load, "pernode_equal_mix"
    assert sem == "mixed", sem
    return [float(sum(h[e] for h in counts)) for e in range(G)], "mixed_concat"


def ensure_eplb_load(
    params,
    W,
    L,
    budget_mib,
    topk,
    chunk_bytes,
    matrix_instance,
    out_root,
    traces_root=None,
    nexperts=None,
):
    """Predicted-load sidecar for the eplb arm: the FULL pool per-expert
    histogram of the cell's exact pools (the oracle-ceiling prediction; the
    sampled batch is drawn from these same pools). Writes
    <out_root>/<mid>.eplb_load.json and returns (path, sha256).

    Weighting mirrors the sampler's expected batch mix (sample_routing draws
    T tokens per rank):
      homog   — the single pool's raw counts;
      pernode — each pool's histogram normalized by its row count, then
                averaged (every node contributes L*T tokens regardless of
                pool size);
      mixed   — raw counts of the concatenation (every rank samples it).

    The load file is PLACEMENT INPUT, not routing: matrix identity is
    unchanged; the driver records the file sha as a cell fact instead.
    """
    mid, params, specs, pools_rows, T = prepare_trace(
        params, W, L, topk, matrix_instance, traces_root, nexperts, budget_mib, chunk_bytes
    )
    counts = [pool_histogram(rows, nexperts) for rows in pools_rows]
    load, weighting = combine_pool_loads(counts, [len(p) for p in pools_rows],
                                         params["sem"])

    blob = {
        "version": EPLB_LOAD_VERSION,
        "matrix_id": mid,
        "poolsha": params["poolsha"],
        "model": params["model"],
        "layer": int(params["layer"]),
        "pool": params["pool"],
        "sem": params["sem"],
        "pools": [f"{b}/{s}" for b, s in specs],
        "G": nexperts,
        "weighting": weighting,
        "pool_rows": [len(p) for p in pools_rows],
        "counts_per_pool": counts,
        "load": load,
    }
    body = json.dumps(blob, indent=1, sort_keys=True) + "\n"
    path = os.path.join(out_root, f"{mid}.eplb_load.json")
    if os.path.exists(path):
        with open(path) as f:
            existing = f.read()
        if existing != body:
            raise RuntimeError(
                f"eplb load sidecar {path} does not match a regeneration from"
                " the current pools — refusing to overwrite; delete it to"
                " regenerate"
            )
        return path, hashlib.sha256(existing.encode()).hexdigest()
    os.makedirs(out_root, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(body)
    os.rename(tmp, path)
    return path, hashlib.sha256(body.encode()).hexdigest()


def print_stats(routing, chunks, W, L, T, G, topk):
    """--print-only body for the trace family: real vs dealer-closed-form."""
    nn = W // L
    u, U = real_dedup_stats(routing, W, L, T, G)
    real = gen_matrix.dedup_stats_from_U(chunks, U, L, T)
    dealer = gen_matrix.dedup_round_stats(chunks, L, T)
    for s, row in enumerate(chunks):
        remote = sum(c for d, c in enumerate(row) if d // L != s // L)
        print(
            f"row {s:3d}: max {max(row):8d} diag_frac {row[s] / sum(row):.3f}"
            f" remote_frac {remote / sum(row):.3f}"
        )
    print("dedup per ordered node pair — REAL routing vs [dealer closed form]:")
    for (sn, dn_), (copies, uniq, ratio) in sorted(real["pair"].items()):
        _, du, dr = dealer["pair"][(sn, dn_)]
        print(
            f"  n{sn}->n{dn_}: copies {copies:8d}  unique {uniq:8d} [{du:8d}]"
            f"  dedup {ratio:.2f} [{dr:.2f}]"
        )
    print("per-dest-node round profile (dn ascending = gateway order), REAL U rows:")
    for m in range(nn):
        prof = "  ".join(f"dn{dn}(src n{sn}) {uu:7d}" for dn, sn, uu in real["round_profile"][m])
        print(f"  dest n{m}: {prof}")
    per_node = [real["col_rows"][n * L : (n + 1) * L] for n in range(nn)]
    print("per-dest-rank GEMM rows (column sums), by node:")
    for n, cols in enumerate(per_node):
        print(f"  node {n}: {' '.join(f'{c:7d}' for c in cols)}")
    print(
        f"headroom (relay_balanced/relay_ident): REAL {real['headroom']:.3f}"
        f"  [dealer closed form {dealer['headroom']:.3f} — INVALID for real routing]"
    )
