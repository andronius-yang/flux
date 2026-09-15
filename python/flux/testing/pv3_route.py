"""PV3 — rotation water-fill replica routing (branch pv3, 2026-09-14).

THE router that implements the paper's routing constraints literally
(design-notation table + eq. routing-cost): for every expert e with global
demand D_e = sum_i d_i^e served by c_e replicas, the balanced reference
load of each replica is q^e = D_e / c_e (zero on non-hosting GPUs), and
the router must keep every replica's realized load inside

    [ floor((1 - C) q^e) , ceil((1 + C) q^e) ]        (constraint 2)

while preserving each source's expert selections (constraint 1) and
integrality (constraint 4). Constraint 3 (per-GPU aggregate within
(1 +- C) of sum_e q_j^e) is implied by constraint 2 with the same C in
real arithmetic (triangle inequality); with integer rounding it holds
within +- n_j rows, n_j = number of expert instances hosted on GPU j
(reported, never enforced separately — see pv3_check).

Algorithm ("rotation water-fill"): R rounds p = 0..R-1. In round p every
source rank i targets exactly one rank

    tgt(i, p) = ((u_i + p // L) mod NN) * L + ((l_i + p) mod L)

(u = node, l = rank-in-node): p = 0 is the source itself, p = 1..L-1 the
other ranks of its node, p >= L the remote nodes in rotation order — the
tier order IS locality order, and for fixed p the map i -> tgt is a
bijection, so every replica is visited by exactly one source per round
and every (source, replica) pair exactly once overall. When source i
visits replica j of expert e it takes

    take = min( left[i, e],                       # its remaining demand
                U_e - fill[e, j],                 # replica upper bound
                unassigned_e - deficit_others )   # lower-bound reserve

where deficit_others = sum over the OTHER replicas j' of max(0, Lb_e -
fill[e, j']). The reserve term keeps enough unassigned demand to fill
every replica up to its lower bound; the invariant unassigned_e >=
sum_j deficit_j holds throughout, and at the end every source is fully
placed with every replica inside [Lb_e, U_e] (proof in
docs/handoff/38_pv3_routing.md §3). There is NO forced regime: no
f_cap, no overflow counter, no repair pass.

Everything is a pure integer function of the allgathered demand histogram
d[R, G] and the placement, so every rank derives the identical tables
(the sender-local contract of the LocCap lane, unchanged); a rank then
materializes only its own S*K entries: the entries of (i, e) in canonical
token order consume e's replicas in the order source i visits them (p
ascending) with the table counts. Per-expert state is independent across
experts -> one thread per expert on the GPU (pv3 kernel), R * c_e
sequential steps each.

C is passed as a rational C_num / C_den so every bound is exact integer
arithmetic (paper C = 1/16 by the arms' --eps 0.0625 convention).

Module contract: imports torch ONLY (file-path importable by the offline
audit and the sweeps tooling, the loccap_semantics precedent).
"""

import hashlib
import math

import torch

UNASSIGNED = -1


def _ceil_div(a, b):
    return -((-a) // b)


def rotation_round(i, j, L, NN):
    """Round in which source rank i visits rank j (tensors or ints)."""
    u_i, l_i = i // L, i % L
    u_j, l_j = j // L, j % L
    return ((u_j - u_i) % NN) * L + ((l_j - l_i) % L)


def replica_table(ipr):
    """ipr [G, R] (phys slot or -1) -> (rep [G, Cmax] host rank ids
    ascending, -1 pad; c [G] replica counts)."""
    G, R = ipr.shape
    hosted = ipr >= 0
    c = hosted.long().sum(1)
    Cmax = max(int(c.max()), 1)
    ordn = hosted.long().cumsum(1) - 1
    rep = torch.full((G, Cmax), -1, dtype=torch.int64, device=ipr.device)
    g_idx, r_idx = hosted.nonzero(as_tuple=True)
    rep[g_idx, ordn[g_idx, r_idx]] = r_idx
    return rep, c


def pv3_bounds(D, c, C_num, C_den):
    """Per-expert integer bounds. D [G] global demand, c [G] replica
    counts (>= 1 wherever D > 0). U = ceil((1+C) D / c), Lb = floor((1-C)
    D / c); experts with c == 0 get U = Lb = 0 (no demand may exist)."""
    assert 0 <= C_num <= C_den and C_den > 0
    cc = c.clamp(min=1)
    U = _ceil_div((C_den + C_num) * D, C_den * cc)
    Lb = ((C_den - C_num) * D) // (C_den * cc)
    U = torch.where(c > 0, U, torch.zeros_like(U))
    Lb = torch.where(c > 0, Lb, torch.zeros_like(Lb))
    return U, Lb


def pv3_tables(d, ipr, L, C_num, C_den):
    """The deterministic table pass (every rank computes this identically).

    d [R, G] int64 demand histogram, ipr [G, R] int64 (phys slot or -1).
    Returns dict(M [G, R, Cmax] rows from source i to replica jj of e,
    rep [G, Cmax], c [G], D [G], U [G], Lb [G], fill [G, Cmax]).
    Vectorized over experts; sequential over (round, replica index) —
    R * Cmax small tensor steps."""
    R, G = d.shape
    NN = R // L
    dev = d.device
    d = d.long()
    rep, c = replica_table(ipr)
    Cmax = rep.shape[1]
    D = d.sum(0)
    assert bool((c[D > 0] >= 1).all()), "demand for an expert with no replica"
    U, Lb = pv3_bounds(D, c, C_num, C_den)
    valid = rep >= 0                                       # [G, Cmax]
    fill = torch.zeros(G, Cmax, dtype=torch.int64, device=dev)
    left = d.t().contiguous()                              # [G, R]
    unassigned = D.clone()
    M = torch.zeros(G, R, Cmax, dtype=torch.int64, device=dev)
    g_ar = torch.arange(G, device=dev, dtype=torch.int64)
    for p in range(R):
        du, dl = p // L, p % L
        for jj in range(Cmax):
            j = rep[:, jj]
            v = valid[:, jj]
            jj_safe = torch.where(v, j, torch.zeros_like(j))
            u_j, l_j = jj_safe // L, jj_safe % L
            # the unique source visiting j in round p
            i = ((u_j - du) % NN) * L + ((l_j - dl) % L)
            want = left[g_ar, i]
            deficit = (Lb.unsqueeze(1) - fill).clamp(min=0) * valid.long()
            def_tot = deficit.sum(1)
            def_j = deficit[:, jj]
            room = U - fill[:, jj]
            take = torch.minimum(torch.minimum(want, room),
                                 unassigned - (def_tot - def_j))
            take = torch.where(v, take.clamp(min=0), torch.zeros_like(take))
            M[g_ar, i, jj] = take
            fill[:, jj] += take
            left[g_ar, i] -= take
            unassigned -= take
    assert bool((unassigned == 0).all()), "pv3 stranded demand (impossible)"
    assert bool((left == 0).all())
    return dict(M=M, rep=rep, c=c, D=D, U=U, Lb=Lb, fill=fill, Cmax=Cmax)


def pv3_visit_order(rep, R, L):
    """For every source rank i and expert e: the replica indices jj in the
    order source i visits them (round ascending). Returns order [R, G,
    Cmax] (jj ids, pads last) and rounds [R, G, Cmax]."""
    G, Cmax = rep.shape
    NN = R // L
    dev = rep.device
    i = torch.arange(R, device=dev, dtype=torch.int64).view(R, 1, 1)
    j = rep.unsqueeze(0)                                   # [1, G, Cmax]
    valid = j >= 0
    j_safe = torch.where(valid, j, torch.zeros_like(j))
    pr = rotation_round(i, j_safe, L, NN)
    pr = torch.where(valid, pr, torch.full_like(pr, R + 1))  # pads last
    order = torch.argsort(pr, dim=2, stable=True)
    return order, torch.gather(pr, 2, order)


def pv3_route(topk_all, p2l, l2p, lcnts, nlp, ranks_per_node, C_num=1,
              C_den=16, return_tables=False):
    """Reference router: [R, S, K] logical expert ids -> ([R, S, K] int32
    physical slots, stats). Deterministic canonical materialization (the
    entries of (src, e) in token order consume src's visit sequence).
    Every rank must call with identical inputs."""
    R, S, K = topk_all.shape
    dev = topk_all.device
    G = int(lcnts.numel())
    L = ranks_per_node
    assert R % L == 0
    topk = topk_all.long()
    ipr = instance_phys_of_rank(l2p.to(dev), lcnts.to(dev), nlp, R)
    flat = topk.reshape(R, S * K)
    assert bool(((flat >= 0) & (flat < G)).all()), "expert ids out of range"
    d = torch.zeros(R * G, dtype=torch.int64, device=dev)
    d.index_add_(0, (torch.arange(R, device=dev, dtype=torch.int64)
                     .unsqueeze(1) * G + flat).reshape(-1),
                 torch.ones(R * S * K, dtype=torch.int64, device=dev))
    d = d.view(R, G)
    tab = pv3_tables(d, ipr, L, C_num, C_den)
    M, rep, Cmax = tab["M"], tab["rep"], tab["Cmax"]
    order, _ = pv3_visit_order(rep, R, L)                  # [R, G, Cmax]
    # per-(src, e) cumulative segment bounds in visit order
    M_vis = torch.gather(M.permute(1, 0, 2), 2, order)     # [R, G, Cmax]
    cum = torch.cumsum(M_vis, dim=2)
    rep_vis = torch.gather(rep.unsqueeze(0).expand(R, G, Cmax), 2, order)
    # canonical ordinal of every entry within its (src, e) segment
    ent_src = (torch.arange(R, device=dev, dtype=torch.int64)
               .unsqueeze(1).expand(R, S * K).reshape(-1))
    ent_g = flat.reshape(-1)
    key = ent_src * G + ent_g
    o = torch.argsort(key, stable=True)
    ks = key[o]
    idx = torch.arange(ks.numel(), device=dev, dtype=torch.int64)
    newgrp = torch.ones_like(ks, dtype=torch.bool)
    if ks.numel() > 1:
        newgrp[1:] = ks[1:] != ks[:-1]
    starts = torch.cummax(torch.where(newgrp, idx, torch.zeros_like(idx)),
                          0).values
    ordn = idx - starts                                    # [E] sorted
    src_o, g_o = ent_src[o], ent_g[o]
    cum_e = cum[src_o, g_o]                                # [E, Cmax]
    seg = (ordn.unsqueeze(1) >= cum_e).sum(1)              # [E]
    assert bool((seg < Cmax).all()), "ordinal beyond the segment table"
    j_o = rep_vis[src_o, g_o, seg]
    assert bool((j_o >= 0).all())
    phys_flat = torch.empty(R * S * K, dtype=torch.int64, device=dev)
    phys_flat[o] = ipr[g_o, j_o]
    phys = phys_flat.view(R, S, K)
    p2l_l = p2l.to(dev).long()
    assert bool(p2l_l[phys].eq(topk).all()), "conservation violated"
    stats = pv3_check(phys, topk, ipr, nlp, L, C_num, C_den, tab=tab)
    if return_tables:
        stats["tables"] = tab
        stats["d"] = d
        rows = torch.bincount(phys_flat // nlp, minlength=R)
        pair = torch.bincount(ent_src * R + phys_flat // nlp,
                              minlength=R * R).view(R, R)
        stats["rows_per_rank"] = rows
        stats["pair"] = pair
    return phys.to(torch.int32), stats


def instance_phys_of_rank(l2p, lcnts, nlp, R):
    """l2p [G, Cmax'] + lcnts [G] -> ipr [G, R] int64 (phys slot or -1)."""
    G, Cm = l2p.shape
    dev = l2p.device
    l2p_l = l2p.long()
    valid = (torch.arange(Cm, device=dev).unsqueeze(0)
             < lcnts.long().unsqueeze(1)) & (l2p_l >= 0)
    g_idx, j_idx = valid.nonzero(as_tuple=True)
    phys = l2p_l[g_idx, j_idx]
    ipr = torch.full((G, R), -1, dtype=torch.int64, device=dev)
    ipr[g_idx, phys // nlp] = phys
    return ipr


def pv3_check(phys, topk, ipr, nlp, L, C_num, C_den, tab=None):
    """Score ANY routing against the paper's constraints (integer form).

    Returns dict with: violations of constraint 2 (per-replica bounds),
    the realized per-replica ratio range, per-GPU aggregate ratio range vs
    Q_j = sum_e q_j^e, and constraint-3 violations counted (a) in the
    real-valued form |load - Q| <= C Q and (b) with the rounding
    allowance n_j."""
    R, S, K = phys.shape
    dev = phys.device
    G = ipr.shape[0]
    phys_l = phys.long().reshape(-1)
    g_flat = topk.long().reshape(-1)
    serve_r = phys_l // nlp
    hosted = ipr >= 0
    c = hosted.long().sum(1)
    D = torch.bincount(g_flat, minlength=G)
    U, Lb = pv3_bounds(D, c, C_num, C_den)
    # realized per-(e, rank) load
    load_gr = torch.zeros(G * R, dtype=torch.int64, device=dev)
    load_gr.index_add_(0, g_flat * R + serve_r,
                       torch.ones_like(g_flat))
    load_gr = load_gr.view(G, R)
    bad_host = (load_gr > 0) & ~hosted
    over = (load_gr - U.unsqueeze(1)).clamp(min=0) * hosted.long()
    under = (Lb.unsqueeze(1) - load_gr).clamp(min=0) * hosted.long()
    # ratios vs q (real): only for replicas with D > 0
    q = D.double() / c.clamp(min=1).double()
    ratio = torch.where(hosted & (D > 0).unsqueeze(1),
                        load_gr.double() / q.clamp(min=1e-9).unsqueeze(1),
                        torch.full_like(load_gr, 1.0, dtype=torch.float64))
    # per-GPU aggregate vs Q_j
    Q = (q.unsqueeze(1) * hosted.double()).sum(0)              # [R]
    load_r = load_gr.sum(0)
    n_j = hosted.long().sum(0)
    dev_r = (load_r.double() - Q).abs()
    C = C_num / C_den
    c3_real_viol = int((dev_r > C * Q + 1e-9).sum())
    c3_round_viol = int((dev_r > C * Q + n_j.double()).sum())
    return {
        "c2_over_rows": int(over.sum()),
        "c2_under_rows": int(under.sum()),
        "c2_replicas_violating": int(((over > 0) | (under > 0)).sum()),
        "nonhost_rows": int((load_gr * bad_host.long()).sum()),
        "replica_ratio_min": float(ratio.min()),
        "replica_ratio_max": float(ratio.max()),
        "gpu_ratio_min": float((load_r.double() / Q.clamp(min=1e-9)).min()),
        "gpu_ratio_max": float((load_r.double() / Q.clamp(min=1e-9)).max()),
        "c3_real_violations": c3_real_viol,
        "c3_rounded_violations": c3_round_viol,
        "rows_max": int(load_r.max()),
        "Q_max": float(Q.max()),
        "Q_over_SK_max": float(Q.max() / (S * K)),
        "imbalance_max_over_mean": float(load_r.max()) / (R * S * K / R),
    }


def route_hash(phys_all) -> int:
    h = hashlib.sha256()
    h.update(phys_all.contiguous().to(torch.int32).cpu().numpy().tobytes())
    return int.from_bytes(h.digest()[:8], "little", signed=True)


def incidence_remote(phys, nlp, L):
    """Sum over tokens of distinct non-home serving nodes (the node-dedup
    transport's wire objective) + remote rows."""
    R, S, K = phys.shape
    NN = R // L
    serve_node = (phys.long() // nlp) // L
    home = (torch.arange(R, device=phys.device) // L).view(R, 1, 1)
    on = torch.zeros(R, S, NN, dtype=torch.bool, device=phys.device)
    on.scatter_(2, serve_node, True)
    on.scatter_(2, home.expand(R, S, 1), False)
    remote_rows = int((serve_node != home).sum())
    return int(on.sum()), remote_rows


def pv3_materialize_cover(topk_all, tab, rep, ipr, nlp, L):
    """OPTIONAL cover-aware materialization (pv3-cover, 2026-09-14 offline
    prototype): same per-(source, replica) counts as pv3_route (the
    constraint proof is untouched), but tokens are assigned in canonical
    order with per-token node preference — an entry takes (1) a local
    segment (own rank, then own node), else (2) a segment on a remote node
    the token already touches, else (3) the first remote segment in visit
    order with rows left. Reduces token-node incidence toward LocCap's
    min-cover. Reference only (python per-token loop); the kernel twin
    would be the pll_route3-style per-token atomic-ticket pass."""
    R, S, K = topk_all.shape
    NN = R // L
    dev = topk_all.device
    G = rep.shape[0]
    M = tab["M"]                                           # [G, R, Cmax]
    Cmax = rep.shape[1]
    order, rounds = pv3_visit_order(rep, R, L)             # [R, G, Cmax]
    rep_vis = torch.gather(rep.unsqueeze(0).expand(R, G, Cmax), 2, order)
    M_vis = torch.gather(M.permute(1, 0, 2), 2, order).clone()
    phys = torch.empty(R, S, K, dtype=torch.int64, device=dev)
    tk = topk_all.long()
    for i in range(R):
        home = i // L
        left = M_vis[i].tolist()                           # [G][Cmax]
        reps = rep_vis[i].tolist()
        for s in range(S):
            touched = set()
            for k in range(K):
                g = int(tk[i, s, k])
                segs = left[g]
                rs = reps[g]
                best, bkey = -1, None
                for jj in range(Cmax):
                    j = rs[jj]
                    if j < 0 or segs[jj] <= 0:
                        continue
                    n = j // L
                    tier = 0 if n == home else (1 if n in touched else 2)
                    key = (tier, jj)
                    if bkey is None or key < bkey:
                        best, bkey = jj, key
                assert best >= 0
                j = rs[best]
                segs[best] -= 1
                touched.add(j // L)
                phys[i, s, k] = int(ipr[g, j])
    return phys


# ---------------------------------------------------------------------------
# PV3C — pv3 + a per-token VACATE pass on per-replica ticket budgets
# (2026-09-14, user-requested add-on: LocCap's node-cover idea under the
# paper's constraints). Tables = pv3's tables (both bounds already hold)
# plus two per-(source, expert, replica) budgets, both pure functions of d:
#   release[i,e,j] : rows source i may move OFF replica j (its slice of
#                    fill[e,j] - Lb_e, the lower-bound slack);
#   extra[i,e,j]   : rows source i may move ONTO replica j (its slice of
#                    U_e - fill[e,j], the upper-bound slack).
# Per token (own rows), a remote node is vacated when every entry of the
# token on it can move to a node the token already touches (or home) with
# release > 0 at the source replica and extra > 0 at the target; freed
# budgets return to the pool. Monotone in incidence; both bounds hold
# because every move is paid from the slack on both sides; remote rows
# never increase (moves go to touched nodes or home).
# ---------------------------------------------------------------------------

def _lr_split(total, weights):
    """Largest-remainder split of the integer `total` proportional to the
    integer weights (python ints, deterministic, ties lower index)."""
    W = sum(weights)
    if total <= 0 or W <= 0:
        return [0] * len(weights)
    base = [total * w // W for w in weights]
    rem = total - sum(base)
    frac = [(total * w) % W for w in weights]
    order = sorted(range(len(weights)), key=lambda k: (-frac[k], k))
    for k in order[:rem]:
        base[k] += 1
    return base


def pv3c_budgets(d, tab):
    """release / extra [G, R, Cmax] from pv3's tables (python loops over
    experts and replicas; the kernel does the same per expert thread)."""
    R, G = d.shape
    M, fill, U, Lb, c = tab["M"], tab["fill"], tab["U"], tab["Lb"], tab["c"]
    Cmax = tab["Cmax"]
    release = torch.zeros(G, R, Cmax, dtype=torch.int64)
    extra = torch.zeros(G, R, Cmax, dtype=torch.int64)
    d_l = d.t().tolist()
    M_l, fill_l = M.tolist(), fill.tolist()
    U_l, Lb_l, c_l = U.tolist(), Lb.tolist(), c.tolist()
    for g in range(G):
        for jj in range(c_l[g]):
            slack_lo = fill_l[g][jj] - Lb_l[g]
            if slack_lo > 0:
                w = [M_l[g][i][jj] for i in range(R)]
                rel = _lr_split(slack_lo, w)
                for i in range(R):
                    release[g, i, jj] = min(rel[i], w[i])
            slack_hi = U_l[g] - fill_l[g][jj]
            if slack_hi > 0:
                ex = _lr_split(slack_hi, d_l[g])
                for i in range(R):
                    extra[g, i, jj] = ex[i]
    return release, extra


def pv3c_route(topk_all, p2l, l2p, lcnts, nlp, ranks_per_node, C_num=1,
               C_den=16, return_tables=False, allow_home=True):
    """pv3c reference for ALL ranks: pv3's routing, then the per-token
    vacate pass on the release/extra budgets. Deterministic."""
    R, S, K = topk_all.shape
    L = ranks_per_node
    phys0, st0 = pv3_route(topk_all, p2l, l2p, lcnts, nlp, L, C_num, C_den,
                           return_tables=True)
    tab, d = st0["tables"], st0["d"]
    rep, Cmax = tab["rep"], tab["Cmax"]
    release, extra = pv3c_budgets(d.cpu(), tab)
    ipr = instance_phys_of_rank(l2p.cpu(), lcnts.cpu(), nlp, R)
    rep_l, ipr_l = rep.tolist(), ipr.tolist()
    topk = topk_all.long().cpu()
    phys = phys0.long().cpu().clone()
    moved = 0
    for i in range(R):
        home = i // L
        rel = release[:, i].tolist()                       # [G][Cmax]
        ext = extra[:, i].tolist()
        tk = topk[i].tolist()
        ph = phys[i].tolist()
        for s in range(S):
            node_of = [ph[s][k] // nlp // L for k in range(K)]
            while True:
                nodes = {}
                for k in range(K):
                    if node_of[k] != home:
                        nodes.setdefault(node_of[k], []).append(k)
                if len(nodes) < (1 if allow_home else 2):
                    break
                done_any = False
                for n in sorted(nodes, key=lambda x: (len(nodes[x]), x)):
                    targets = [m for m in nodes if m != n]
                    if allow_home:
                        targets = [home] + targets
                    plan = []
                    ok = True
                    for k in nodes[n]:
                        g = tk[s][k]
                        # source replica index of the current slot
                        jo = next(jj for jj in range(Cmax)
                                  if rep_l[g][jj] >= 0
                                  and ipr_l[g][rep_l[g][jj]] == ph[s][k])
                        if rel[g][jo] <= 0:
                            ok = False
                            break
                        tgt = None
                        for m in targets:
                            for jj in range(Cmax):
                                j = rep_l[g][jj]
                                if j < 0:
                                    break
                                if j // L == m and ext[g][jj] > 0:
                                    tgt = jj
                                    break
                            if tgt is not None:
                                break
                        if tgt is None:
                            ok = False
                            break
                        plan.append((k, g, jo, tgt))
                    if not ok:
                        continue
                    for k, g, jo, jt in plan:
                        rel[g][jo] -= 1
                        ext[g][jo] += 1        # freed room at the source
                        ext[g][jt] -= 1
                        rel[g][jt] += 1        # the moved row may move back
                        ph[s][k] = ipr_l[g][rep_l[g][jt]]
                        node_of[k] = rep_l[g][jt] // L
                        moved += 1
                    done_any = True
                    break
                if not done_any:
                    break
        phys[i] = torch.tensor(ph)
    p2l_l = p2l.cpu().long()
    assert bool(p2l_l[phys].eq(topk).all()), "conservation violated"
    stats = pv3_check(phys, topk, ipr, nlp, L, C_num, C_den)
    stats["moved"] = moved
    if return_tables:
        stats["tables"] = tab
        stats["release"], stats["extra"] = release, extra
        stats["d"] = d
        serve = phys.reshape(-1) // nlp
        src = torch.arange(R).repeat_interleave(S * K)
        stats["rows_per_rank"] = torch.bincount(serve, minlength=R)
        stats["pair"] = torch.bincount(src * R + serve,
                                       minlength=R * R).view(R, R)
    return phys.to(torch.int32), stats
