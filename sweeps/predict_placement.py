#!/usr/bin/env python3
"""PLACE-lambda: node-aware expert placement from pool co-occurrence, the
offline LocCap incidence simulator, and the <mid>.placement.json sidecar.

Offline, ZERO GPU — runs on a login node (torch CPU only, the
predict_phase_split.py precedent). This is the pre-registration instrument:
it computes, BEFORE any allocation, the predicted token-node incidence of
every (placement x router x eps) arm on the exact routing the GPU cell will
run, so the sweep confirms or falsifies numbers that are already committed.

Theory (session 8.19.theory): under the node-dedup transports
(hier_compress Mode-2, both directions) the inter-node wire rows of a token
equal its token-node incidence |S_t| = #distinct non-home serving nodes.
Placement's job is to make small incidence feasible (cluster co-occurring
experts per node, then per-node-first replication); the LocCap router
achieves it at runtime subject to per-rank caps (1+eps)*S*K.

Same-code guarantee: the simulator file-path-imports
python/flux/testing/loccap_semantics.py — the identical module the GPU
driver runs — so predicted incidence must equal realized incidence
bit-for-bit on the same (placement, routing) inputs. A mismatch in the
bring-up gate is a determinism bug, never noise.

Placement inputs are FULL POOL statistics (the eplb_load oracle-ceiling
discipline: the sampled batch is drawn from these same pools); the
simulation runs on the realized .routing.txt sidecar. The placement file is
PLACEMENT INPUT, not routing — matrix identity is unchanged; the driver
records the sidecar sha as a cell fact (the .eplb_load.json contract).

Subcommands:
  plan    — build one placement + sidecar for a (family, nodes, budget) cell
  table   — the pre-registration table across budgets/modes/eps (P1-P6 inputs)
  verify  — regenerate twice, assert byte-identical + structural invariants
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_matrix  # noqa: E402
import gen_trace_routing  # noqa: E402

PLACEMENT_VERSION = 1
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCCAP_PATH = os.path.join(_REPO, "python", "flux", "testing",
                            "loccap_semantics.py")
_spec = importlib.util.spec_from_file_location("loccap_semantics",
                                               _LOCCAP_PATH)
LC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LC)

EPS_LADDER_DEFAULT = (0.0625, 0.125, 0.25, 0.5, 1.0, math.inf)


def load_platform(name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "platforms", f"{name}.yaml")
    with open(path) as f:
        text = f.read()
    try:
        import yaml
        plat = yaml.safe_load(text)
    except ImportError:
        import miniyaml
        plat = miniyaml.loads(text)
    for key in ("matrices_root", "traces_root"):
        plat[key] = os.path.expandvars(plat[key])
        assert "$" not in plat[key], (key, plat[key])
    return plat


def resolve_family(family_str):
    name, params = None, None
    name, _, rest = family_str.partition(":")
    assert name == "trace", f"placement planning needs the trace family, got {name}"
    params = gen_matrix.parse_params(rest.split(";")) if rest else {}
    return dict(gen_matrix.FAMILY_DEFAULT_PARAMS["trace"], **params)


# --------------------------------------------------------------------------
# PLACE-lambda planner (numpy, deterministic)
# --------------------------------------------------------------------------

def _pools_for_nodes(pools_rows, sem, NN):
    """Per-NODE token pools (the home distribution each node samples)."""
    if sem == "pernode":
        assert len(pools_rows) == NN
        return [np.asarray(p, dtype=np.int64) for p in pools_rows]
    if sem == "homog":
        assert len(pools_rows) == 1
        p = np.asarray(pools_rows[0], dtype=np.int64)
        return [p] * NN
    # mixed: every rank samples the concatenation
    cat = np.concatenate([np.asarray(p, dtype=np.int64) for p in pools_rows])
    return [cat] * NN


def build_placement(pools_rows, sem, G, W, L, nlp, mode="nodeaware",
                    fm_passes=8, balance_tol=0.20, min_gain=0.0,
                    target_slots=None):
    """-> dict with hosts (per-expert sorted rank lists), primary_node,
    and planner stats. Modes:
      nodeaware — FM-refined rooted-lambda partition + per-node-first
                  coverage replication with bundle credit;
      rankconc  — same partition, replicas concentrated on home-node ranks
                  (the P4 equal-slot ablation; predicted incidence unchanged).
    """
    NN = W // L
    epn = G // W
    pools = _pools_for_nodes(pools_rows, sem, NN)
    w_u = [1.0 / len(p) for p in pools]

    # predicted per-expert load, node-equal mix (the eplb_load weighting)
    load_e = np.zeros(G)
    for u in range(NN):
        hist = np.bincount(pools[u].reshape(-1), minlength=G)
        load_e += w_u[u] * hist

    # inverted index per pool: expert -> token row ids
    e2t = []
    for u in range(NN):
        idx = [None] * G
        flat = pools[u]
        for g in range(G):
            idx[g] = np.flatnonzero((flat == g).any(axis=1))
        e2t.append(idx)

    # --- Stage A: rooted-lambda FM refinement from the contiguous seed ----
    primary = np.array([(g // epn) // L for g in range(G)], dtype=np.int64)
    cnt = []  # per pool u: [n_u, NN] serving-node expert counts
    for u in range(NN):
        c = np.zeros((len(pools[u]), NN), dtype=np.int32)
        np.add.at(c, (np.arange(len(pools[u]))[:, None], primary[pools[u]]),
                  1)
        cnt.append(c)

    def stage_a_cost():
        cost = 0.0
        for u in range(NN):
            visited = (cnt[u] > 0)
            visited[:, u] = False
            cost += w_u[u] * visited.sum()
        return cost

    node_loadA = np.zeros(NN)
    node_cntA = np.zeros(NN, dtype=np.int64)
    for g in range(G):
        node_loadA[primary[g]] += load_e[g]
        node_cntA[primary[g]] += 1
    load_cap = (1.0 + balance_tol) * load_e.sum() / NN
    cnt_cap = L * epn + (L * (nlp - epn)) // 2  # keep >= half the spare free

    cost0 = stage_a_cost()
    moves = 0
    order = sorted(range(G), key=lambda g: (-load_e[g], g))
    for _pass in range(fm_passes):
        moved_this_pass = 0
        for g in order:
            a = int(primary[g])
            best_b, best_delta = None, -1e-12
            for b in range(NN):
                if b == a:
                    continue
                if node_cntA[b] + 1 > cnt_cap:
                    continue
                if node_loadA[b] + load_e[g] > load_cap:
                    continue
                delta = 0.0
                for u in range(NN):
                    rows = e2t[u][g]
                    if rows.size == 0:
                        continue
                    gain = np.count_nonzero(cnt[u][rows, a] == 1) \
                        if a != u else 0
                    pay = np.count_nonzero(cnt[u][rows, b] == 0) \
                        if b != u else 0
                    delta += w_u[u] * (pay - gain)
                if delta < best_delta:
                    best_delta, best_b = delta, b
            if best_b is not None:
                b = best_b
                for u in range(NN):
                    rows = e2t[u][g]
                    if rows.size:
                        cnt[u][rows, a] -= 1
                        cnt[u][rows, b] += 1
                node_loadA[a] -= load_e[g]
                node_loadA[b] += load_e[g]
                node_cntA[a] -= 1
                node_cntA[b] += 1
                primary[g] = b
                moved_this_pass += 1
        moves += moved_this_pass
        if moved_this_pass == 0:
            break
    cost_fm = stage_a_cost()

    # --- Stage B: replication ---------------------------------------------
    inst_nodes = [{int(primary[g])} for g in range(G)]  # nodes holding g
    slots_left = [int(L * nlp - node_cntA[u]) for u in range(NN)]
    spent = 0
    inst_extra_rank = [[] for _ in range(G)]  # rankconc duplicates (rank ids)

    if mode == "nodeaware":
        while True:
            best = None  # (score, exact, g, u)
            for u in range(NN):
                if slots_left[u] <= 0:
                    continue
                for g in range(G):
                    if u in inst_nodes[g]:
                        continue
                    rows = e2t[u][g]
                    if rows.size == 0:
                        continue
                    p = int(primary[g])
                    c = cnt[u][rows, p]
                    exact = w_u[u] * float(np.count_nonzero(c == 1))
                    over = c[c > 1]
                    credit = w_u[u] * float((1.0 / over).sum()) if over.size \
                        else 0.0
                    score = exact + credit
                    if score <= min_gain:
                        continue
                    key = (score, exact, -g, -u)
                    if best is None or key > best[0]:
                        best = (key, g, u)
            if best is None:
                break
            _, g, u = best
            rows = e2t[u][g]
            p = int(primary[g])
            cnt[u][rows, p] -= 1
            cnt[u][rows, u] += 1
            inst_nodes[g].add(u)
            slots_left[u] -= 1
            spent += 1
    elif mode == "rankconc":
        assert target_slots is not None, \
            "rankconc needs target_slots (= the nodeaware run's spent slots)"
        # duplicate hottest experts on their home node's other ranks
        while spent < target_slots:
            cand = None
            for g in sorted(range(G),
                            key=lambda g: (-load_e[g] /
                                           (1 + len(inst_extra_rank[g])), g)):
                u = int(primary[g])
                if slots_left[u] <= 0:
                    continue
                if 1 + len(inst_extra_rank[g]) >= L:
                    continue  # one instance per rank max within the node
                cand = g
                break
            if cand is None:
                break
            g = cand
            u = int(primary[g])
            inst_extra_rank[g].append(None)  # rank chosen in assign stage
            slots_left[u] -= 1
            spent += 1
    else:
        raise SystemExit(f"unknown placement mode {mode!r}")

    # --- Stage C: LPT rank assignment within each node --------------------
    per_node_inst = [[] for _ in range(NN)]  # (g, load share)
    for g in range(G):
        n_inst = len(inst_nodes[g]) + len(inst_extra_rank[g])
        share = load_e[g] / n_inst
        for u in sorted(inst_nodes[g]):
            per_node_inst[u].append((g, share))
        for _ in inst_extra_rank[g]:
            per_node_inst[int(primary[g])].append((g, share))
    hosts = [[] for _ in range(G)]
    for u in range(NN):
        ranks = list(range(u * L, (u + 1) * L))
        rload = {r: 0.0 for r in ranks}
        rslots = {r: 0 for r in ranks}
        insts = sorted(per_node_inst[u], key=lambda t: (-t[1], t[0]))
        for g, share in insts:
            cand = [r for r in ranks
                    if rslots[r] < nlp and r not in hosts[g]]
            assert cand, (u, g, "no rank slot left — raise nlp")
            r = min(cand, key=lambda r: (rload[r], r))
            hosts[g].append(r)
            rload[r] += share
            rslots[r] += 1
    hosts = [sorted(h) for h in hosts]

    return {
        "hosts": hosts,
        "primary_node": primary.tolist(),
        "stats": {
            "mode": mode,
            "fm_passes": fm_passes,
            "fm_moves": moves,
            "balance_tol": balance_tol,
            "min_gain": min_gain,
            "rooted_lambda_cost_seed": round(cost0, 3),
            "rooted_lambda_cost_fm": round(cost_fm, 3),
            "replica_slots_spent": spent,
            "cnt_cap": cnt_cap,
        },
    }


# --------------------------------------------------------------------------
# Simulation on the realized routing
# --------------------------------------------------------------------------

def routing_tensor(rpath, W):
    rows = gen_trace_routing.read_routing(rpath)
    arr = torch.tensor(rows, dtype=torch.int64)
    assert arr.shape[0] % W == 0
    return arr.reshape(W, arr.shape[0] // W, arr.shape[1])


def simulate_arm(topk_all, hosts, nlp, L, router, eps=None):
    W = topk_all.shape[0]
    p2l, l2p, lcnts = LC.plan_tensors_from_hosts(hosts, W, nlp)
    if router == "d6":
        phys = LC.d6_route(topk_all, l2p, lcnts)
    else:
        phys = LC.loccap_route(topk_all, p2l, l2p, lcnts, nlp, L, eps)
    st = LC.incidence_stats(phys, nlp, L)
    R, S, K = phys.shape
    serve_rank = phys.long() // nlp
    serve_node = serve_rank // L
    home_rank = torch.arange(R).view(R, 1, 1)
    home_node = home_rank // L
    offdiag = int((serve_rank != home_rank).sum())
    nodedup = int((serve_node != home_node).sum())
    # per-home-node dedup egress rows (the hot-node NR-01 quantity)
    on_node = torch.zeros(R, S, W // L, dtype=torch.bool)
    on_node.scatter_(2, serve_node, True)
    remote_pt = on_node.sum(2) - on_node.gather(
        2, home_node.expand(R, S, 1)).squeeze(2).long()
    egress = remote_pt.sum(1).reshape(W // L, L).sum(1)
    return {
        "router": router if eps is None else f"loccap_eps{eps:g}",
        "incidence_remote": st["incidence_remote"],
        "mean_nodes_per_token": round(st["mean_nodes_per_token"], 4),
        "imbalance": round(st["imbalance_max_over_mean"], 4),
        "rows_per_rank_max": max(st["rows_per_rank"]),
        "offdiag_rows": offdiag,
        "internode_rows_nodedup": nodedup,
        "internode_rows_dedup": st["incidence_remote"],
        "dedup_ratio": round(1.0 - st["incidence_remote"] / nodedup, 4)
        if nodedup else 0.0,
        "hot_node_egress_rows": int(egress.max()),
        "route_hash": LC.route_hash(phys),
    }


def fixed_hosts(G, W):
    epn = G // W
    return [[g // epn] for g in range(G)]


# --------------------------------------------------------------------------
# Sidecar (the ensure_eplb_load discipline)
# --------------------------------------------------------------------------

def ensure_placement(params, W, L, budget_mib, topk, chunk_bytes,
                     matrix_instance, out_root, traces_root, nexperts,
                     redundant_per_rank=1, mode="nodeaware",
                     eps_ladder=EPS_LADDER_DEFAULT, fm_passes=8,
                     balance_tol=0.20, min_gain=0.0, target_slots=None):
    """Build (or validate) <mid>.placement[.mode].json. Returns (path, sha,
    blob). Never overwrites on mismatch (pre-registration immutability)."""
    mid, params, specs, pools_rows, T = gen_trace_routing.prepare_trace(
        params, W, L, topk, matrix_instance, traces_root, nexperts,
        budget_mib, chunk_bytes)
    epn = nexperts // W
    nlp = epn + redundant_per_rank

    # Fast-path revalidation: when the sidecar exists, rebuild only the
    # PLACEMENT (the code-drift-prone part, ~seconds) and verify it matches;
    # skip re-simulating the eps ladder (~minutes at b64) — the driver
    # hard-asserts each cell's realized route_hash/incidence against the
    # sidecar's predicted entries, so ladder integrity is enforced per run
    # anyway. Delete the sidecar to force a full regeneration.
    suffix_early = f"placement_{mode}_r{redundant_per_rank}"
    path_early = os.path.join(out_root, f"{mid}.{suffix_early}.json")
    if os.path.exists(path_early):
        with open(path_early) as f:
            existing_blob = json.loads(f.read())
        placed_chk = build_placement(
            pools_rows, params["sem"], nexperts, W, L, nlp, mode=mode,
            fm_passes=fm_passes, balance_tol=balance_tol, min_gain=min_gain,
            target_slots=target_slots)
        if (existing_blob.get("hosts") != placed_chk["hosts"]
                or existing_blob.get("planner") != placed_chk["stats"]):
            raise RuntimeError(
                f"placement sidecar {path_early} does not match a placement"
                " regeneration — refusing to overwrite; delete it to"
                " regenerate")
        with open(path_early) as f:
            existing = f.read()
        return (path_early,
                hashlib.sha256(existing.encode()).hexdigest(),
                existing_blob)

    placed = build_placement(pools_rows, params["sem"], nexperts, W, L, nlp,
                             mode=mode, fm_passes=fm_passes,
                             balance_tol=balance_tol, min_gain=min_gain,
                             target_slots=target_slots)

    # simulate on the realized routing (generate deterministically if the
    # sidecar files are absent; ensure_trace_matrix is idempotent)
    _, _, _, rpath, _ = gen_trace_routing.ensure_trace_matrix(
        dict(params), W, L, budget_mib, topk, chunk_bytes, matrix_instance,
        out_root, traces_root=traces_root, nexperts=nexperts)
    topk_all = routing_tensor(rpath, W)

    predicted = [simulate_arm(topk_all, placed["hosts"], nlp, L, "d6")]
    for eps in eps_ladder:
        predicted.append(simulate_arm(topk_all, placed["hosts"], nlp, L,
                                      "loccap", eps))

    blob = {
        "version": PLACEMENT_VERSION,
        "matrix_id": mid,
        "poolsha": params["poolsha"],
        "model": params["model"],
        "layer": int(params["layer"]),
        "pool": params["pool"],
        "sem": params["sem"],
        "pools": [f"{b}/{s}" for b, s in specs],
        "G": nexperts,
        "W": W,
        "ranks_per_node": L,
        "nlp": nlp,
        "redundant_per_rank": redundant_per_rank,
        "mode": mode,
        "planner": placed["stats"],
        "primary_node": placed["primary_node"],
        "hosts": placed["hosts"],
        "predicted": predicted,
    }
    body = json.dumps(blob, indent=1, sort_keys=True) + "\n"
    # mode + redundancy are part of the sidecar identity (a different
    # redundant_per_rank is a different placement; never a silent overwrite)
    suffix = f"placement_{mode}_r{redundant_per_rank}"
    path = os.path.join(out_root, f"{mid}.{suffix}.json")
    if os.path.exists(path):
        with open(path) as f:
            existing = f.read()
        if existing != body:
            raise RuntimeError(
                f"placement sidecar {path} does not match a regeneration —"
                " refusing to overwrite; delete it to regenerate")
        return path, hashlib.sha256(existing.encode()).hexdigest(), blob
    os.makedirs(out_root, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(body)
    os.rename(tmp, path)
    return path, hashlib.sha256(body.encode()).hexdigest(), blob


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _common(ap):
    ap.add_argument("--platform", default="perlmutter")
    ap.add_argument("--family", required=True)
    ap.add_argument("--nodes", type=int, required=True)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--G", type=int, default=128)
    ap.add_argument("--chunk-bytes", type=int, default=8192)
    ap.add_argument("--matrix-instance", default="001")
    ap.add_argument("--redundant-per-rank", type=int, default=1)
    ap.add_argument("--eps-ladder", default=None,
                    help="comma list; 'inf' allowed")


def _eps_ladder(arg):
    if arg is None:
        return EPS_LADDER_DEFAULT
    return tuple(math.inf if x.strip() == "inf" else float(x)
                 for x in arg.split(","))


def _setup(args):
    plat = load_platform(args.platform)
    W = args.nodes * plat["ranks_per_node"]
    L = plat["ranks_per_node"]
    params = resolve_family(args.family)
    return plat, W, L, params


def cmd_plan(args):
    plat, W, L, params = _setup(args)
    for mode in ("nodeaware", "rankconc"):
        target = None
        if mode == "rankconc":
            target = cmd_plan.last_spent
        path, sha, blob = ensure_placement(
            dict(params), W, L, args.budget_mib, args.topk, args.chunk_bytes,
            args.matrix_instance, plat["matrices_root"],
            plat["traces_root"], args.G,
            redundant_per_rank=args.redundant_per_rank, mode=mode,
            eps_ladder=_eps_ladder(args.eps_ladder), target_slots=target)
        cmd_plan.last_spent = blob["planner"]["replica_slots_spent"]
        print(f"{mode}: {path}\n  sha {sha}\n  planner {blob['planner']}")


cmd_plan.last_spent = None


def cmd_table(args):
    plat, W, L, params = _setup(args)
    eps_ladder = _eps_ladder(args.eps_ladder)
    epn = args.G // W
    nlp = epn + args.redundant_per_rank
    report = {"config": {"family": args.family, "nodes": args.nodes,
                         "W": W, "L": L, "G": args.G, "topk": args.topk,
                         "redundant_per_rank": args.redundant_per_rank},
              "budgets": {}}
    for budget in args.budgets_mib:
        _, _, _, rpath, _ = gen_trace_routing.ensure_trace_matrix(
            dict(params), W, L, budget, args.topk, args.chunk_bytes,
            args.matrix_instance, plat["matrices_root"],
            traces_root=plat["traces_root"], nexperts=args.G)
        topk_all = routing_tensor(rpath, W)
        rows = []
        rows.append(("fixed", simulate_arm(topk_all, fixed_hosts(args.G, W),
                                           nlp, L, "d6")))
        na_path, na_sha, na_blob = ensure_placement(
            dict(params), W, L, budget, args.topk, args.chunk_bytes,
            args.matrix_instance, plat["matrices_root"],
            plat["traces_root"], args.G,
            redundant_per_rank=args.redundant_per_rank, mode="nodeaware",
            eps_ladder=eps_ladder)
        for pred in na_blob["predicted"]:
            rows.append(("nodeaware", pred))
        rc_path, rc_sha, rc_blob = ensure_placement(
            dict(params), W, L, budget, args.topk, args.chunk_bytes,
            args.matrix_instance, plat["matrices_root"],
            plat["traces_root"], args.G,
            redundant_per_rank=args.redundant_per_rank, mode="rankconc",
            eps_ladder=eps_ladder,
            target_slots=na_blob["planner"]["replica_slots_spent"])
        for pred in rc_blob["predicted"]:
            rows.append(("rankconc", pred))

        base = rows[0][1]["internode_rows_dedup"]
        print(f"\n== b{budget} MiB, {args.nodes}n (W={W}) — internode dedup "
              f"rows, baseline fixed/d6 = {base} ==")
        print(f"{'placement':<10} {'router':<16} {'dedup rows':>10} "
              f"{'vs fixed':>9} {'imb':>6} {'hot egress':>10}")
        for name, p in rows:
            frac = (base - p["internode_rows_dedup"]) / base if base else 0.0
            print(f"{name:<10} {p['router']:<16} "
                  f"{p['internode_rows_dedup']:>10} {frac:>8.1%} "
                  f"{p['imbalance']:>6.3f} {p['hot_node_egress_rows']:>10}")
        report["budgets"][str(budget)] = {
            "baseline_fixed_d6": rows[0][1],
            "nodeaware": {"sidecar_sha": na_sha,
                          "planner": na_blob["planner"],
                          "predicted": na_blob["predicted"]},
            "rankconc": {"sidecar_sha": rc_sha,
                         "planner": rc_blob["planner"],
                         "predicted": rc_blob["predicted"]},
        }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1, sort_keys=True)
        print(f"\nreport written: {args.out}")


def cmd_verify(args):
    plat, W, L, params = _setup(args)
    epn = args.G // W
    nlp = epn + args.redundant_per_rank
    mid, params2, specs, pools_rows, T = gen_trace_routing.prepare_trace(
        dict(params), W, L, args.topk, args.matrix_instance,
        plat["traces_root"], args.G, args.budget_mib, args.chunk_bytes)
    a = build_placement(pools_rows, params2["sem"], args.G, W, L, nlp)
    b = build_placement(pools_rows, params2["sem"], args.G, W, L, nlp)
    assert a == b, "planner is not deterministic"
    hosts = a["hosts"]
    for g, h in enumerate(hosts):
        assert len(h) >= 1, (g, "expert unplaced")
        assert len(set(h)) == len(h), (g, "duplicate rank")
    per_rank = {}
    for g, h in enumerate(hosts):
        for r in h:
            per_rank[r] = per_rank.get(r, 0) + 1
    assert max(per_rank.values()) <= nlp, "rank slot capacity exceeded"
    # loccap must never lose to the D6 baseline it replaces
    _, _, _, rpath, _ = gen_trace_routing.ensure_trace_matrix(
        dict(params), W, L, args.budget_mib, args.topk, args.chunk_bytes,
        args.matrix_instance, plat["matrices_root"],
        traces_root=plat["traces_root"], nexperts=args.G)
    topk_all = routing_tensor(rpath, W)
    d6 = simulate_arm(topk_all, hosts, nlp, L, "d6")
    for eps in _eps_ladder(args.eps_ladder):
        lc = simulate_arm(topk_all, hosts, nlp, L, "loccap", eps)
        assert lc["internode_rows_dedup"] <= d6["internode_rows_dedup"], (
            eps, lc["internode_rows_dedup"], d6["internode_rows_dedup"])
        print(f"eps={eps:g}: dedup rows {lc['internode_rows_dedup']} "
              f"(d6 {d6['internode_rows_dedup']}) imb {lc['imbalance']:.3f}")
    print("OK: verify passed (deterministic, structural, loccap <= d6)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    _common(p)
    p.add_argument("--budget-mib", type=int, required=True)
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("table")
    _common(p)
    p.add_argument("--budgets-mib", type=lambda s: [int(x) for x in
                                                    s.split(",")],
                   required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_table)
    p = sub.add_parser("verify")
    _common(p)
    p.add_argument("--budget-mib", type=int, required=True)
    p.set_defaults(fn=cmd_verify)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
