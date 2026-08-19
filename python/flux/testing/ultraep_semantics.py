"""UltraEP balancing-algorithm semantics ported to plain PyTorch.

UltraEP (Dots-Infra/UltraEP) balances MoE expert parallelism with real-time
expert replication: per layer/microbatch it takes the exact post-gating
per-expert load, solves a replication plan (which hot experts get replicas on
which ranks, and how many tokens each instance should serve), confines every
replica to the NVLink scale-up domain of its master, syncs master weights to
the replica slots over NVLink, and reroutes tokens logical -> physical.

UltraEP's kernels are SM90+/NVSHMEM (single solver CUDA kernel, one thread
block per NVLink domain), so this port re-implements the ALGORITHM, not the
kernels:

  * `solve_placement` below is a near-verbatim port of
    `quota_placement_solve_kernel` (UltraEP csrc/kernels/placement.cu,
    v1.0.0 @ 94cab09 / sm80-oracle branch): the fast-eps threshold probe
    (dynamic q-floor), the coarse capacity bisection, the 4-probe parallel
    bisection ladder (probe placement `lo + range*(w+1)/5` in int64, first
    feasible warp wins), the greedy export oracle (source ranks descending by
    load, experts descending by load, target = max remaining slack), and the
    locality-aware largest-remainder per-rank quota decomposition, with all
    tie-breaks preserved. The only floating-point ops in the kernel are two
    float32 ceils (threshold lower bound and fast-eps probe); they are
    reproduced with numpy float32.
  * `reroute_expand` ports `dense_quota_reroute_scatter_kernel`
    (csrc/kernels/reroute.cu): per-expert token ordinals in token order and
    the coprime-stride interleave permutation over the rank-quota prefix.
  * The fidelity contract is BIT-IDENTICAL outputs vs the real kernels via
    `ultra_ep._C.solve_placement_for_test` / `dense_reroute_for_test`
    (enforced by test_ultraep_planner.py's GPU tier on the SM80 oracle
    build). The planner is deterministic integer math: every rank computes
    the identical plan from the all-gathered per-rank loads, no broadcast.

Semantics notes for Perlmutter (4-GPU NVLink domains, EP16):
  * One independent solve per NVLink domain: cross-node imbalance is
    untouched by design. The reachable floor is
    `nvl_domain_lower_bound` = max over domains of (domain mean rank load)
    divided by the global mean rank load.
  * weight_sync is the `direct` plan by construction (UltraEP forces direct
    at NVL domain <= 8; adaptive relay additionally needs >= 4 replicas of
    one expert, impossible with 3 peer ranks).
"""

import math
from dataclasses import dataclass, field

import numpy as np
import torch

INT32_MAX = 2**31 - 1

# Compile-time constants of quota_placement_solve_kernel (placement.cu).
QUOTA_SOLVER_WARPS = 4              # 128 threads -> 4 probe lanes in the ladder
QUOTA_ORACLE_FAST_MAX_RETRY = 2     # 3 fast-eps attempts total
MAX_REPLICAS_PER_NVL = 512          # export-plan capacity per domain


def _ceilf(x: np.float32) -> int:
    """C `static_cast<int>(ceilf(x))` on a float32 value (x >= 0 here)."""
    return int(np.ceil(np.float32(x)))


@dataclass
class UltraEPConfig:
    """Shape math of ultra_ep.Manager for one EP group."""

    S: int              # tokens per rank
    K: int              # topk
    G: int              # global logical expert count (num_experts)
    R: int              # EP world size (num_ranks)
    H: int = 0          # hidden dim (unused by the planner, used by the runner)
    D: int = 4          # NVLink domain size (num_nvl_ranks)
    R_red: int = 2      # redundant expert slots per rank
    balance_threshold: float = 1.0          # ULTRA_EP_BALANCE_THRESHOLD
    min_tokens_per_replica: int = 1024      # ULTRA_EP_QUOTA_MIN_TOKENS_PER_REPLICA
    allow_zero_master_quota: bool = False   # ULTRA_EP_QUOTA_ALLOW_ZERO_MASTER_QUOTA
    locality_aware: bool = True             # ULTRA_EP_QUOTA_LOCALITY_AWARE
    oracle_eps: float = 0.01                # ULTRA_EP_QUOTA_ORACLE_EPS
    interleave: bool = True                 # ULTRA_EP_QUOTA_REROUTE_INTERLEAVE

    def __post_init__(self):
        assert self.G % self.R == 0, f"G ({self.G}) % R ({self.R}) != 0"
        assert self.R % self.D == 0, f"R ({self.R}) % D ({self.D}) != 0"
        assert self.R_red >= 0
        self.epn = self.G // self.R              # num_local_master_experts
        self.nlp = self.epn + self.R_red         # num_local_physical_experts
        self.P = self.R * self.nlp               # num_global_physical_experts
        self.num_domains = self.R // self.D
        self.E_dom = self.epn * self.D           # num_logical_per_nvl
        self.max_replicas_dim = self.R           # kernel hard-sets this
        self.N = self.S * self.K
        assert 0 < self.N < INT32_MAX
        assert self.R_red * self.D <= MAX_REPLICAS_PER_NVL


@dataclass
class DomainSolution:
    """Per-domain solver trace (threshold + which path produced the plan)."""

    threshold: int
    path: str                   # "none" | "fast" | "precheck" | "bisect"
    entries: list               # ordered [(expert_local, target_rank_local, quota)]
    export_sum: list            # [E_dom] tokens exported per domain-local expert


@dataclass
class UltraEPPlan:
    """Replicated global plan: every rank computes this identically.

    Tensor layouts match the ultra_ep._C solver outputs exactly (int32,
    max_replicas_dim = R columns, -1 padding for unused map slots, 0 padding
    for unused quota slots) so the GPU bit-equality tier is a plain
    torch.equal per tensor.
    """

    cfg: UltraEPConfig
    tpe: torch.Tensor                # [R, G] int32 tokens per expert per source
    p2l: torch.Tensor                # [P] int32 physical -> logical (-1 unused)
    l2p: torch.Tensor                # [G, R] int32 logical -> physical (col 0 master)
    lcnts: torch.Tensor              # [G] int32 instance count incl. master
    quota: torch.Tensor              # [G, R] int32 global per-instance quota
    quota_prefix: torch.Tensor       # [G, R] int32 running prefix over instances
    rank_quota_prefix: torch.Tensor  # [R, G, R] int32 per SOURCE rank prefix
    domain_solutions: list = field(default_factory=list)   # [num_domains]

    def plan_hash(self) -> int:
        """Order-stable digest for cross-rank determinism asserts."""
        import hashlib

        h = hashlib.sha256()
        for t in (self.tpe, self.p2l, self.l2p, self.lcnts, self.quota,
                  self.quota_prefix, self.rank_quota_prefix):
            h.update(t.contiguous().numpy().tobytes())
        ov = getattr(self, "phys_override", None)
        if ov is not None:
            # LocCap routing is part of the plan identity: the driver's
            # cross-rank all-gather then also asserts routing SPMD identity
            h.update(ov.contiguous().to(torch.int32).numpy().tobytes())
        h.update(",".join(
            f"{d.threshold}:{d.path}" for d in self.domain_solutions
        ).encode())
        return int.from_bytes(h.digest()[:8], "little", signed=True)

    # -- derived facts ------------------------------------------------------

    def physical_rows_per_rank(self) -> list:
        """Tokens each rank actually receives = sum of all source ranks'
        rank-quota allocations to instances hosted there (the real GEMM
        row counts, which can deviate from `quota` by per-source rounding)."""
        cfg = self.cfg
        ov = getattr(self, "phys_override", None)
        if ov is not None:  # LocCap: the override IS the physical routing
            return torch.bincount((ov.long() // cfg.nlp).reshape(-1),
                                  minlength=cfg.R).tolist()
        alloc = self.rank_quota_prefix.long().clone()          # [R, G, R]
        alloc[:, :, 1:] -= self.rank_quota_prefix.long()[:, :, :-1]
        rows = [0] * cfg.R
        l2p = self.l2p.long()
        for l in range(cfg.G):
            C = int(self.lcnts[l])
            for j in range(C):
                host = int(l2p[l, j]) // cfg.nlp
                rows[host] += int(alloc[:, l, j].sum())
        return rows

    def replica_summary(self) -> dict:
        counts = self.lcnts.long() - 1
        return {
            "total_replicas": int(counts.sum()),
            "max_replicas_per_expert": int(counts.max()),
            "slots_used": int(counts.sum()),
            "slots_total": self.cfg.R * self.cfg.R_red,
        }


def loads_from_topk(cfg: UltraEPConfig, topk_all: torch.Tensor) -> torch.Tensor:
    """[R, S, K] expert ids -> [R, G] int32 per-source-rank load histogram
    (the tensor UltraEP all-gathers before solving)."""
    R, G = cfg.R, cfg.G
    assert tuple(topk_all.shape) == (R, cfg.S, cfg.K)
    topk = topk_all.cpu().long().reshape(R, -1)
    assert bool(((topk >= 0) & (topk < G)).all()), "expert ids out of range"
    tpe = torch.zeros(R, G, dtype=torch.int32)
    for r in range(R):
        tpe[r] = torch.bincount(topk[r], minlength=G).to(torch.int32)
    return tpe


def nvl_domain_lower_bound(cfg: UltraEPConfig, tpe: torch.Tensor) -> float:
    """Reachable rank-level imbalance floor when replication is confined to
    NVLink domains (UltraEP tests/utils.py nvl_domain_physical_lower_bound):
    max over domains of (domain mean rank load) / (global mean rank load)."""
    loads = tpe.long().sum(dim=0)                        # [G] global expert loads
    rank_load = loads.reshape(cfg.R, cfg.epn).sum(dim=1).double()
    global_mean = float(rank_load.mean())
    if global_mean == 0:
        return 1.0
    domain_means = rank_load.reshape(cfg.num_domains, cfg.D).mean(dim=1)
    return float(domain_means.max()) / global_mean


# ---------------------------------------------------------------------------
# The solver: near-verbatim port of quota_placement_solve_kernel
# ---------------------------------------------------------------------------


def _capacity_feasible(rank_load, threshold, G):
    """warp_capacity_feasible: necessary condition sum(excess) <= sum(slack)."""
    excess = sum(max(rank_load[r] - threshold, 0) for r in range(G))
    slack = sum(max(threshold - rank_load[r], 0) for r in range(G))
    return excess <= slack


def _oracle_export_plan(loads, sorted_experts, rank_load, source_order,
                        threshold, G, epn, R_red, min_tokens_per_replica,
                        allow_zero_master_quota, use_dynamic_q_floor,
                        store_plan):
    """warp_build_export_plan<STORE_PLAN> with oracle_batch_k = 1.

    loads: [E_dom] domain expert loads; indices are domain-local.
    Returns (feasible, entries, export_sum). entries/export_sum only
    meaningful when store_plan (mirrors the kernel: probe runs discard them).
    """
    E = len(loads)
    export_sum = [0] * E
    occ = [1 << (l // epn) for l in range(E)]   # master rank pre-seeded
    excess = [max(rank_load[r] - threshold, 0) for r in range(G)]
    slack = [max(threshold - rank_load[r], 0) for r in range(G)]
    slots_used = [0] * G
    entries = []

    total_excess = sum(excess)
    total_slack = sum(slack)
    if total_excess == 0:
        return True, entries, export_sum
    if total_excess > total_slack:
        return False, entries, export_sum

    running_excess = total_excess
    running_slack = total_slack
    for source in source_order:
        need = excess[source]
        if need <= 0:
            continue
        if need > running_slack:
            return False, entries, export_sum

        for expert_local in sorted_experts[source]:
            keep_on_master = (
                1 if (not allow_zero_master_quota and loads[expert_local] > 0)
                else 0
            )
            available = max(
                loads[expert_local] - keep_on_master - export_sum[expert_local], 0
            )

            while need > 0 and available > 0:
                effective_min_tokens = min_tokens_per_replica
                if use_dynamic_q_floor:
                    feasible_slots = sum(
                        1 for r in range(G)
                        if slack[r] > 0 and slots_used[r] < R_red
                        and not (occ[expert_local] >> r) & 1
                        and min(need, slack[r], available) > 0
                    )
                    if feasible_slots <= 0:
                        break
                    effective_min_tokens = max(
                        1, (need + feasible_slots - 1) // feasible_slots
                    )

                # warp_find_best_target(_reg): max remaining slack among
                # ranks with slack, a free redundant slot, no copy of this
                # expert yet, and q >= effective floor; ties -> lowest rank.
                best_target = -1
                best_cap = -1
                for r in range(G):
                    if (slack[r] <= 0 or slots_used[r] >= R_red
                            or (occ[expert_local] >> r) & 1):
                        continue
                    q = min(need, slack[r], available)
                    if q < effective_min_tokens:
                        continue
                    if slack[r] > best_cap:
                        best_target = r
                        best_cap = slack[r]
                if best_target < 0:
                    break

                if store_plan and len(entries) >= MAX_REPLICAS_PER_NVL:
                    return False, entries, export_sum

                q = min(need, slack[best_target], available)
                if q < effective_min_tokens:
                    break

                if store_plan:
                    entries.append((expert_local, best_target, q))
                export_sum[expert_local] += q
                slack[best_target] -= q
                slots_used[best_target] += 1
                occ[expert_local] |= 1 << best_target
                running_excess -= q
                running_slack -= q
                need -= q
                available -= q

            if need == 0:
                break

        if need > 0:
            return False, entries, export_sum
        if running_excess > running_slack:
            return False, entries, export_sum

    return True, entries, export_sum


def _solve_domain(cfg: UltraEPConfig, loads, domain) -> DomainSolution:
    """One thread block of quota_placement_solve_kernel: solve one NVLink
    domain from the GLOBAL expert loads (list of ints, domain slice)."""
    G = cfg.D
    epn = cfg.epn
    E = cfg.E_dom

    rank_load = [
        sum(loads[r * epn + m] for m in range(epn)) for r in range(G)
    ]
    # warp_sort_source_ranks_by_load: descending load, ties -> lower rank.
    source_order = sorted(range(G), key=lambda r: (-rank_load[r], r))
    # per-rank expert insertion sort: descending load, ties -> lower id.
    sorted_experts = [
        sorted(range(r * epn, (r + 1) * epn), key=lambda e: (-loads[e], e))
        for r in range(G)
    ]

    do_binary_search = (cfg.R_red > 0 and G > 1)
    if not do_binary_search:
        return DomainSolution(threshold=0, path="none", entries=[],
                              export_sum=[0] * E)

    oracle = lambda threshold, dynamic, store: _oracle_export_plan(
        loads, sorted_experts, rank_load, source_order, threshold, G, epn,
        cfg.R_red, cfg.min_tokens_per_replica, cfg.allow_zero_master_quota,
        dynamic, store,
    )

    total_rank_load = sum(rank_load)
    max_rank_load = max(rank_load)
    bt = np.float32(max(np.float32(cfg.balance_threshold), np.float32(1.0)))
    lo = _ceilf(np.float32(total_rank_load) / np.float32(G) * bt)
    hi = max(max_rank_load, lo)

    # Fast-eps probe (dynamic q-floor; the ONLY path that uses it).
    eps = np.float32(max(np.float32(cfg.oracle_eps), np.float32(0.0)))
    fast_threshold = max(_ceilf(np.float32(lo) * (np.float32(1.0) + eps)), lo)
    fast_step = max(1, (hi + 99) // 100)
    for _ in range(QUOTA_ORACLE_FAST_MAX_RETRY + 1):
        feasible, entries, export_sum = oracle(fast_threshold, True, True)
        if feasible:
            return DomainSolution(threshold=fast_threshold, path="fast",
                                  entries=entries, export_sum=export_sum)
        fast_threshold = min(hi, fast_threshold + fast_step)

    # Coarse capacity bisection over [lo, hi) (necessary condition only).
    coarse_lo, coarse_hi = lo, hi
    while coarse_lo < coarse_hi:
        probe = coarse_lo + ((coarse_hi - coarse_lo) >> 1)
        if _capacity_feasible(rank_load, probe, G):
            coarse_hi = probe
        else:
            coarse_lo = probe + 1
    lo = coarse_lo

    # Full-oracle precheck at the coarse bound (static floor from here on).
    feasible, entries, export_sum = oracle(lo, False, True)
    if feasible:
        return DomainSolution(threshold=lo, path="precheck",
                              entries=entries, export_sum=export_sum)
    lo = min(lo + 1, hi)

    # 4-probe parallel bisection ladder (QUOTA_SOLVER_WARPS lanes).
    W = QUOTA_SOLVER_WARPS
    while lo < hi:
        rng = hi - lo
        if rng <= W:
            probes = [lo + w for w in range(W)]
            valid = [p < hi for p in probes]
        else:
            probes = [lo + (rng * (w + 1)) // (W + 1) for w in range(W)]
            valid = [True] * W
        feas = [
            valid[w] and oracle(probes[w], False, False)[0] for w in range(W)
        ]
        if rng <= W:
            best_probe = -1
            for w in range(W):
                if valid[w] and feas[w]:
                    best_probe = probes[w]
                    break
            if best_probe >= 0:
                lo = hi = best_probe
            else:
                lo = hi
        else:
            first = next((w for w in range(W) if feas[w]), -1)
            if first >= 0:
                hi = probes[first]
                if first > 0:
                    lo = probes[first - 1] + 1
            else:
                lo = probes[W - 1] + 1

    feasible, entries, export_sum = oracle(lo, False, True)
    assert feasible, f"domain {domain}: final threshold {lo} infeasible"
    return DomainSolution(threshold=lo, path="bisect",
                          entries=entries, export_sum=export_sum)


def _rank_quota_alloc_for_expert(cfg: UltraEPConfig, C, quota_row, host_rank,
                                 my_rank, my_load, domain_loads_col,
                                 domain_start_rank, locality_aware):
    """Per-(source rank, logical expert) quota decomposition: how many of MY
    tokens for this expert go to each of its C instances. Verbatim port of
    the kernel's fast/slow rank-quota paths (identical semantics)."""
    my_alloc = [0] * C
    remainders = [-1] * C

    if not locality_aware:
        rem_my = my_load
        total_quota = sum(quota_row[:C])
        if rem_my > 0 and total_quota > 0:
            assigned = 0
            for j in range(C):
                scaled = rem_my * quota_row[j]
                my_alloc[j] = scaled // total_quota
                remainders[j] = scaled % total_quota
                assigned += my_alloc[j]
            remaining = rem_my - assigned
            while remaining > 0:
                best_j = 0
                for j in range(1, C):
                    if remainders[j] > remainders[best_j]:
                        best_j = j
                my_alloc[best_j] += 1
                remainders[best_j] = -1
                remaining -= 1
        return my_alloc

    host_remaining = list(domain_loads_col)     # [D] tokens per domain rank
    remote_cap = [0] * C
    rem_my = my_load
    total_remote_cap = 0
    for j in range(C):
        q = quota_row[j]
        host_local = host_rank[j] - domain_start_rank
        assert 0 <= host_local < cfg.D
        local_fill = min(host_remaining[host_local], q)
        host_remaining[host_local] -= local_fill
        remote_cap[j] = q - local_fill
        total_remote_cap += remote_cap[j]
        if host_rank[j] == my_rank:
            my_alloc[j] += local_fill
            rem_my -= local_fill
    rem_my = max(rem_my, 0)

    if rem_my > 0 and total_remote_cap > 0:
        assigned = 0
        for j in range(C):
            if host_rank[j] == my_rank or remote_cap[j] <= 0:
                continue
            scaled = rem_my * remote_cap[j]
            share = scaled // total_remote_cap
            my_alloc[j] += share
            remainders[j] = scaled % total_remote_cap
            assigned += share
        remaining = rem_my - assigned
        while remaining > 0:
            best_j = -1
            for j in range(C):
                if host_rank[j] == my_rank or remote_cap[j] <= 0:
                    continue
                if (best_j < 0 or remainders[j] > remainders[best_j]
                        or (remainders[j] == remainders[best_j]
                            and (host_rank[j] < host_rank[best_j]
                                 or (host_rank[j] == host_rank[best_j]
                                     and j < best_j)))):
                    best_j = j
            assert best_j >= 0
            my_alloc[best_j] += 1
            remainders[best_j] = -1
            remaining -= 1
    return my_alloc


def build_rank_quota_prefix(cfg: UltraEPConfig, tpe, l2p, lcnts, quota,
                            my_rank, locality_aware=None) -> torch.Tensor:
    """[G, R] int32 rank-quota prefix for one source rank (the reroute
    table). Exposed separately so remote-traffic facts can be computed with
    locality on AND off from the same placement."""
    if locality_aware is None:
        locality_aware = cfg.locality_aware
    tpe_l = tpe.long()
    l2p_l = l2p.long()
    quota_l = quota.long()
    out = torch.zeros(cfg.G, cfg.max_replicas_dim, dtype=torch.int32)
    for l in range(cfg.G):
        C = int(lcnts[l])
        domain = (l // cfg.epn) // cfg.D
        domain_start_rank = domain * cfg.D
        host_rank = [int(l2p_l[l, j]) // cfg.nlp for j in range(C)]
        domain_loads_col = [
            int(tpe_l[domain_start_rank + r, l]) for r in range(cfg.D)
        ]
        my_alloc = _rank_quota_alloc_for_expert(
            cfg, C, [int(q) for q in quota_l[l, :C]], host_rank, my_rank,
            int(tpe_l[my_rank, l]), domain_loads_col, domain_start_rank,
            locality_aware,
        )
        prefix = 0
        for j in range(C):
            prefix += my_alloc[j]
            out[l, j] = prefix
    return out


def solve_placement(cfg: UltraEPConfig, tpe: torch.Tensor) -> UltraEPPlan:
    """Replicated placement solve from the all-gathered [R, G] loads.

    Every rank runs this on identical inputs and gets the identical plan
    (kernel property: deterministic integer math, no broadcast needed).
    """
    assert tuple(tpe.shape) == (cfg.R, cfg.G)
    loads_global = [int(x) for x in tpe.long().sum(dim=0)]

    p2l = torch.full((cfg.P,), -1, dtype=torch.int32)
    l2p = torch.full((cfg.G, cfg.max_replicas_dim), -1, dtype=torch.int32)
    lcnts = torch.zeros(cfg.G, dtype=torch.int32)
    quota = torch.zeros(cfg.G, cfg.max_replicas_dim, dtype=torch.int32)
    quota_prefix = torch.zeros(cfg.G, cfg.max_replicas_dim, dtype=torch.int32)

    domain_solutions = []
    for domain in range(cfg.num_domains):
        dom_start_rank = domain * cfg.D
        dom_start_log = dom_start_rank * cfg.epn
        loads = loads_global[dom_start_log:dom_start_log + cfg.E_dom]

        # Masters pinned at rank * nlp + local_idx.
        for i in range(cfg.E_dom):
            l_global = dom_start_log + i
            rank = dom_start_rank + i // cfg.epn
            phys = rank * cfg.nlp + i % cfg.epn
            p2l[phys] = l_global
            l2p[l_global, 0] = phys

        sol = _solve_domain(cfg, loads, domain)
        domain_solutions.append(sol)
        do_binary_search = (cfg.R_red > 0 and cfg.D > 1)

        expert_slot = [1] * cfg.E_dom
        next_slot = [0] * cfg.D
        for expert_local in range(cfg.E_dom):
            l_global = dom_start_log + expert_local
            master_quota = (
                loads[expert_local] - sol.export_sum[expert_local]
                if do_binary_search else loads[expert_local]
            )
            quota[l_global, 0] = master_quota
        for expert_local, target_local, q in sol.entries:
            l_global = dom_start_log + expert_local
            slot = expert_slot[expert_local]
            expert_slot[expert_local] += 1
            phys = ((dom_start_rank + target_local) * cfg.nlp + cfg.epn
                    + next_slot[target_local])
            next_slot[target_local] += 1
            p2l[phys] = l_global
            l2p[l_global, slot] = phys
            quota[l_global, slot] = q
        for expert_local in range(cfg.E_dom):
            l_global = dom_start_log + expert_local
            C = expert_slot[expert_local]
            lcnts[l_global] = C
            prefix = 0
            for j in range(C):
                prefix += int(quota[l_global, j])
                quota_prefix[l_global, j] = prefix

    rank_quota_prefix = torch.stack([
        build_rank_quota_prefix(cfg, tpe, l2p, lcnts, quota, src)
        for src in range(cfg.R)
    ])

    return UltraEPPlan(
        cfg=cfg, tpe=tpe.to(torch.int32), p2l=p2l, l2p=l2p, lcnts=lcnts,
        quota=quota, quota_prefix=quota_prefix,
        rank_quota_prefix=rank_quota_prefix,
        domain_solutions=domain_solutions,
    )


# ---------------------------------------------------------------------------
# Reroute: port of dense_quota_reroute_scatter_kernel
# ---------------------------------------------------------------------------


def _interleave_params(total: int, expert_id: int):
    """Coprime-stride permutation of [0, total): reroute.cu:189-209."""
    stride = total // 2 + 1
    if stride >= total:
        stride = total - 1
    if stride < 1:
        stride = 1
    while math.gcd(stride, total) != 1:
        stride += 1
        if stride >= total:
            stride = 1
    return stride, expert_id % total


def reroute_expand(cfg: UltraEPConfig, plan: UltraEPPlan, src: int,
                   topk_src: torch.Tensor):
    """Expand one source rank's routing logical -> physical.

    topk_src: [S, K] expert ids (each token's entries must be distinct, the
    dense routing-map contract). Returns (entry_token [n], entry_phys [n])
    int64 in the kernel's implicit order (expert-major, token order within
    an expert), n == S*K.
    """
    ov = getattr(plan, "phys_override", None)
    if ov is not None:
        return _expand_from_phys(cfg, plan, src, topk_src, ov[src])
    S, K = cfg.S, cfg.K
    assert tuple(topk_src.shape) == (S, K)
    topk = topk_src.cpu().long()
    rqp = plan.rank_quota_prefix[src].long()
    l2p = plan.l2p.long()
    lcnts = plan.lcnts.long()

    tokens_out, phys_out = [], []
    routing = torch.zeros(S, cfg.G, dtype=torch.bool)
    routing[torch.arange(S).unsqueeze(1), topk] = True
    assert int(routing.sum()) == S * K, "duplicate expert ids within a token"

    for l in range(cfg.G):
        tok = routing[:, l].nonzero(as_tuple=True)[0]
        n_tok = tok.numel()
        if n_tok == 0:
            continue
        C = int(lcnts[l])
        prefix = rqp[l, :C]
        total = int(prefix[C - 1]) if C > 0 else 0
        assert total == n_tok, (
            f"src {src} expert {l}: rank-quota total {total} != tokens {n_tok}"
        )
        ordinal = torch.arange(n_tok, dtype=torch.int64)
        if cfg.interleave and total > 1:
            stride, offset = _interleave_params(total, l)
            quota_rank = (ordinal * stride + offset) % total
        else:
            quota_rank = ordinal
        replica = torch.searchsorted(prefix, quota_rank, right=True)
        replica = torch.clamp(replica, max=max(C - 1, 0))
        tokens_out.append(tok)
        phys_out.append(l2p[l, replica])

    entry_token = torch.cat(tokens_out) if tokens_out else torch.zeros(0, dtype=torch.int64)
    entry_phys = torch.cat(phys_out) if phys_out else torch.zeros(0, dtype=torch.int64)
    return entry_token, entry_phys


def _expand_from_phys(cfg: UltraEPConfig, plan: UltraEPPlan, src: int,
                      topk_src: torch.Tensor, phys_src: torch.Tensor):
    """Per-token replica-selection seam (LocCap): expand one source rank's
    routing from an explicit [S, K] physical-slot choice
    (plan.phys_override) instead of the rank-quota prefix. Output order
    matches reroute_expand exactly (logical-expert-major, token asc within
    an expert); every downstream consumer re-sorts by (phys, token), the
    parity just keeps the layouts comparable across routers.

    Contract asserts (the seam self-defends): each entry's slot holds ITS
    expert (conservation through p2l), each token's experts are distinct,
    and each token's chosen slots are distinct (one instance per
    (token, logical))."""
    S, K = cfg.S, cfg.K
    assert tuple(topk_src.shape) == (S, K)
    topk = topk_src.cpu().long()
    phys = phys_src.cpu().long()
    assert tuple(phys.shape) == (S, K)
    p2l = plan.p2l.long()
    assert bool(p2l[phys].eq(topk).all()), (
        f"src {src}: phys_override conservation violated")
    tsrt = topk.sort(dim=1).values
    assert bool((tsrt[:, 1:] != tsrt[:, :-1]).all()), (
        "duplicate expert ids within a token")
    psrt = phys.sort(dim=1).values
    assert bool((psrt[:, 1:] != psrt[:, :-1]).all()), (
        "duplicate physical slot within a token")
    g_flat = topk.reshape(-1)
    s_flat = torch.arange(S, dtype=torch.int64).repeat_interleave(K)
    order = torch.argsort(g_flat * S + s_flat)
    return s_flat[order], phys.reshape(-1)[order]


def reroute_expand_dense(cfg: UltraEPConfig, plan: UltraEPPlan, src: int,
                         topk_src: torch.Tensor, probs: torch.Tensor = None):
    """[S, P] expanded routing map (+ probs) — the exact output layout of
    ultra_ep._C.dense_reroute_for_test, for the GPU bit-equality tier."""
    entry_token, entry_phys = reroute_expand(cfg, plan, src, topk_src)
    expanded = torch.zeros(cfg.S, cfg.P, dtype=torch.bool)
    expanded[entry_token, entry_phys] = True
    if probs is None:
        return expanded
    expanded_probs = torch.zeros(cfg.S, cfg.P, dtype=torch.float32)
    routing = torch.zeros(cfg.S, cfg.G, dtype=torch.bool)
    topk = topk_src.cpu().long()
    routing[torch.arange(cfg.S).unsqueeze(1), topk] = True
    logical = plan.p2l.long()[entry_phys]
    expanded_probs[entry_token, entry_phys] = probs.cpu()[entry_token, logical]
    return expanded, expanded_probs


def remote_token_fraction(cfg: UltraEPConfig, plan: UltraEPPlan,
                          locality_aware: bool) -> float:
    """Fraction of routed tokens whose instance lives off their source rank,
    under the given locality mode (recomputed from the same placement, so
    with/without-locality is a clean A/B)."""
    tpe = plan.tpe
    total = int(tpe.long().sum())
    if total == 0:
        return 0.0
    remote = 0
    for src in range(cfg.R):
        rqp = build_rank_quota_prefix(
            cfg, tpe, plan.l2p, plan.lcnts, plan.quota, src,
            locality_aware=locality_aware,
        ).long()
        alloc = rqp.clone()
        alloc[:, 1:] -= rqp[:, :-1]
        for l in range(cfg.G):
            for j in range(int(plan.lcnts[l])):
                host = int(plan.l2p[l, j]) // cfg.nlp
                if host != src:
                    remote += int(alloc[l, j])
    return remote / total


def wire_matrix(cfg: UltraEPConfig, plan: UltraEPPlan,
                topk_all: torch.Tensor) -> list:
    """[R][R] wire ROWS from src to dest under the plan (no dedup)."""
    rows = []
    for src in range(cfg.R):
        _, phys = reroute_expand(cfg, plan, src, topk_all[src])
        dest = phys // cfg.nlp
        rows.append(torch.bincount(dest, minlength=cfg.R).tolist())
    return rows


# ---------------------------------------------------------------------------
# Dispatch transport layout + per-rank runner (flux moonep-arm pattern)
# ---------------------------------------------------------------------------
#
# UltraEP's own data plane is NVLink load/store for weight sync and an
# external dispatcher (DeepEP-style a2av) for tokens. Without a shared
# address space across Slingshot nodes, this port stages: pack rerouted rows
# sorted by (physical expert, token) -> all_to_all_single -> destination-
# local placement into per-physical-expert segments -> grouped GEMM. The
# plan is replicated, so the receiver derives every placement index locally
# (zero extra handshakes). There is NO wire dedup: one row per (token,
# physical expert) entry, faithful to UltraEP/Megatron dispatch semantics
# (`ultraep_dup_rows` quantifies the delta vs MoonEP's dedup).


@dataclass
class RankCommLayout:
    """Index metadata for one rank, precomputed from the replicated plan
    (untimed-metadata contract: in the real system this is planner/reroute
    kernel output)."""

    rank: int
    # sender side
    send_row_index: torch.Tensor    # [n_send] int64 token row per packed row
    send_entry_logical: torch.Tensor  # [n_send] int64 logical expert per row
    send_counts: list               # [R] rows sent to each dest
    # receiver side
    recv_counts: list               # [R] rows received from each src
    place_slots: torch.Tensor       # [n_recv] int64 hidden_buf slot per row
    seg_start: list                 # [nlp] segment starts (local phys order)
    seg_rows: list                  # [nlp] real rows per local phys slot
    # GEMM segments: (local_phys_slot, seg_start, seg_end, logical_expert)
    gemm_segments: list
    # weight sync transfer list: (dest_rank, dest_slot_b, logical, home_rank)
    weight_sync_pairs: list


def build_comm_layout(plan: UltraEPPlan, rank: int,
                      topk_all: torch.Tensor,
                      pinned_masters: bool = True) -> RankCommLayout:
    cfg = plan.cfg
    R, nlp = cfg.R, cfg.nlp

    # Per-source expanded entries in canonical (phys, token) order: sorting
    # by phys is dest-major for free (dest = phys // nlp is monotone).
    ent_tok, ent_phys = [], []
    for src in range(R):
        t, p = reroute_expand(cfg, plan, src, topk_all[src])
        order = torch.argsort(p * (cfg.S + 1) + t, stable=True)
        ent_tok.append(t[order])
        ent_phys.append(p[order])

    my_tok, my_phys = ent_tok[rank], ent_phys[rank]
    my_dest = my_phys // nlp
    send_counts = torch.bincount(my_dest, minlength=R).tolist()
    send_entry_logical = plan.p2l.long()[my_phys]

    # Receiver: rows arrive src-major, (phys, token) order within src.
    phys_base = rank * nlp
    recv_counts, recv_phys_local = [], []
    for src in range(R):
        m = (ent_phys[src] // nlp) == rank
        recv_counts.append(int(m.sum()))
        recv_phys_local.append(ent_phys[src][m] - phys_base)
    all_local = (
        torch.cat(recv_phys_local) if recv_phys_local
        else torch.zeros(0, dtype=torch.int64)
    )
    seg_rows = torch.bincount(all_local, minlength=nlp)
    seg_start = torch.zeros(nlp, dtype=torch.int64)
    seg_start[1:] = torch.cumsum(seg_rows, dim=0)[:-1]
    # slot of row i = seg_start[phys] + occurrence index in (src, entry) order
    order = torch.argsort(all_local, stable=True)
    place_slots = torch.empty_like(all_local)
    positions = torch.arange(all_local.numel(), dtype=torch.int64)
    run_start = torch.where(
        torch.cat([torch.ones(1, dtype=torch.bool),
                   all_local[order][1:] != all_local[order][:-1]])
        if all_local.numel() else torch.zeros(0, dtype=torch.bool),
        positions, torch.full_like(positions, -1),
    )
    run_start = torch.cummax(run_start, dim=0).values if all_local.numel() else run_start
    occ_sorted = positions - run_start
    place_slots[order] = seg_start[all_local[order]] + occ_sorted

    # GEMM segments over local physical slots (masters then replicas).
    p2l = plan.p2l.long()
    segments = []
    for p in range(nlp):
        rows = int(seg_rows[p])
        if rows == 0:
            continue
        logical = int(p2l[phys_base + p])
        assert logical >= 0, f"rank {rank}: rows in unused slot {p}"
        start = int(seg_start[p])
        segments.append((p, start, start + rows, logical))

    # Weight sync (direct plan): every replica instance pulls its master's
    # weights home_rank -> host rank. All pairs intra-domain by construction.
    # Only meaningful under UltraEP's pinned-master invariant (masters at
    # rank*nlp+local, replicas in the redundant slots); plans with full
    # re-placement (eplb arm) pass pinned_masters=False and manage their own
    # one-time weight placement instead.
    l2p = plan.l2p.long()
    pairs = []
    if pinned_masters:
        for l in range(cfg.G):
            C = int(plan.lcnts[l])
            home = int(l2p[l, 0]) // nlp
            for j in range(1, C):
                phys = int(l2p[l, j])
                dest = phys // nlp
                b = phys % nlp - cfg.epn
                assert dest != home, "replica co-located with master"
                assert dest // cfg.D == home // cfg.D, "replica escaped NVL domain"
                pairs.append((dest, b, l, home))

    return RankCommLayout(
        rank=rank,
        send_row_index=my_tok,
        send_entry_logical=send_entry_logical,
        send_counts=send_counts,
        recv_counts=recv_counts,
        place_slots=place_slots,
        seg_start=seg_start.tolist(),
        seg_rows=seg_rows.tolist(),
        gemm_segments=segments,
        weight_sync_pairs=pairs,
    )


class UltraEPLayer0Runner:
    """One rank's UltraEP-semantics layer0 pipeline on NCCL + local scatters.

    Phase methods are single-purpose so the driver can bracket each with
    CUDA events: pack() / a2av() / place() / weight_sync() / gemm().
    """

    # Subclasses for plans without UltraEP's pinned-master invariant (eplb
    # arm) set this False so the layout skips weight_sync_pairs derivation.
    PINNED_MASTERS = True

    def __init__(self, plan: UltraEPPlan, rank: int, group, device,
                 topk_all: torch.Tensor, dtype=torch.bfloat16,
                 ffn_size_shard: int = 0, sync_fc2: bool = True):
        import torch.distributed as dist  # local import: keep CPU-importable

        self.dist = dist
        self.plan = plan
        self.cfg = plan.cfg
        self.rank = rank
        self.group = group
        self.device = device
        self.dtype = dtype
        self.ffn_size_shard = ffn_size_shard
        self.sync_fc2 = sync_fc2
        self.transport = "nccl"
        self._topk_all = topk_all
        cfg = self.cfg

        lay = build_comm_layout(plan, rank, topk_all,
                                pinned_masters=self.PINNED_MASTERS)
        self.lay = lay
        dev = device
        self.send_row_index = lay.send_row_index.to(dev)
        self.send_entry_logical = lay.send_entry_logical.to(dev)
        self.place_slots = lay.place_slots.to(dev)

        self.n_send = int(lay.send_row_index.numel())
        self.n_recv = sum(lay.recv_counts)

        H = cfg.H
        self.send_buf = torch.empty(self.n_send, H, dtype=dtype, device=dev)
        self.recv_buf = torch.empty(self.n_recv, H, dtype=dtype, device=dev)
        self.wsend_buf = torch.empty(self.n_send, dtype=torch.float32, device=dev)
        self.wrecv_buf = torch.empty(self.n_recv, dtype=torch.float32, device=dev)
        self.hidden_buf = torch.zeros(max(self.n_recv, 1), H, dtype=dtype, device=dev)
        self.weights_buf = torch.zeros(max(self.n_recv, 1), dtype=torch.float32,
                                       device=dev)
        if ffn_size_shard:
            self.out_buf = torch.zeros(
                max(self.n_recv, 1), ffn_size_shard, dtype=dtype, device=dev
            )
            self.replica_fc1 = torch.zeros(
                cfg.R_red, ffn_size_shard, H, dtype=dtype, device=dev
            )
            if sync_fc2:
                # Layer0 consumes only fc1; fc2 rides along (bitwise-verified)
                # so weight_sync bytes are faithful to UltraEP's full expert
                # sync (fc1 + fc2 per replica).
                self.replica_fc2 = torch.zeros(
                    cfg.R_red, H, ffn_size_shard, dtype=dtype, device=dev
                )

        # Wire accounting.
        self.send_rows_by_dest = lay.send_counts
        self.wire_row_bytes = H * self.send_buf.element_size()

    # -- deterministic synthetic fc2 (home-rank generated, sync-verified) ---

    def make_fc2_home(self) -> torch.Tensor:
        """[epn, H, ffn_shard] deterministic per-expert fc2 stand-in (the
        layer0 ctx has no fc2; values only need to be home-rank-derivable so
        the receiver can verify the sync bitwise)."""
        cfg = self.cfg
        gen = torch.Generator().manual_seed(9973 + self.rank)
        w = torch.rand(cfg.epn, cfg.H, self.ffn_size_shard,
                       dtype=torch.float32, generator=gen)
        return w.to(self.dtype).to(self.device)

    # -- transport selection ------------------------------------------------

    def enable_nvshmem(self, local_world_size: int, num_comm_sm: int = 8):
        """Swap NCCL for flux's one-sided NVSHMEM All2AllSingle.

        Live path (a2a_single_kernel_v2 in all2all_single_2d_impl.cu):
        cudaMemcpyAsync of the whole send buffer into symmetric staging,
        one nvshmemx_putmem_nbi_block per destination into the fixed slot
        max_split*n_dim*my_rank of the destination's staging, a scalar
        copy-out kernel, and TWO nvshmemx_barrier_on_stream team barriers
        per forward — put-then-barrier, the same publication shape as
        MoonEP's real dispatch (TMA push + one system-scope exit barrier)
        and the transport class of UltraEP's own external dispatchers
        (DeepEP/HybridEP are NVSHMEM one-sided). NOT putmem_signal: that
        kernel in the same file is dead code, never launched (NR-12).

        Requires flux.init_flux_shm(group) to have run. The ctor is
        collective and device-syncs — call at setup only. No dedup means
        per-pair rows == per-pair entries, so ONE max_split (the exact
        max over the replicated post-reroute wire matrix — safe: the op
        never checks per-pair splits against max_split, overflow would be
        silent) and ONE splits pair serve both the hidden and the probs
        instance."""
        import flux  # GPU-side only

        rows = wire_matrix(self.cfg, self.plan, self._topk_all)
        max_split = max(1, max(max(r) for r in rows))
        self._a2a_hidden = flux.All2AllSingle(
            self.group, max_split, self.cfg.H, local_world_size, self.dtype,
        )
        self._a2a_probs = flux.All2AllSingle(
            self.group, max_split, 1, local_world_size, torch.float32,
        )
        dev = self.device
        self._in_splits = torch.tensor(
            self.lay.send_counts, dtype=torch.int32, device=dev)
        self._out_splits = torch.tensor(
            self.lay.recv_counts, dtype=torch.int32, device=dev)
        self._num_comm_sm = num_comm_sm
        self.transport = "nvshmem"

    # -- timed phases -------------------------------------------------------

    def pack(self, inputs_shard: torch.Tensor, probs_shard: torch.Tensor):
        """Gather rerouted rows ((phys, token)-sorted) + per-entry probs."""
        torch.index_select(inputs_shard, 0, self.send_row_index, out=self.send_buf)
        w = probs_shard[self.send_row_index, self.send_entry_logical]
        self.wsend_buf.copy_(w)

    def a2av(self):
        """Wire: token rows + per-entry fp32 probs (identical splits: no
        dedup, one prob per wire row)."""
        if self.transport == "nvshmem":
            self._a2a_hidden.forward(
                self.send_buf, self.recv_buf,
                self._in_splits, self._out_splits, self._num_comm_sm,
            )
            self._a2a_probs.forward(
                self.wsend_buf.view(-1, 1), self.wrecv_buf.view(-1, 1),
                self._in_splits, self._out_splits, self._num_comm_sm,
            )
            return
        self.dist.all_to_all_single(
            self.recv_buf, self.send_buf,
            output_split_sizes=self.lay.recv_counts,
            input_split_sizes=self.lay.send_counts,
            group=self.group,
        )
        self.dist.all_to_all_single(
            self.wrecv_buf, self.wsend_buf,
            output_split_sizes=self.lay.recv_counts,
            input_split_sizes=self.lay.send_counts,
            group=self.group,
        )

    def place(self):
        """Placement scatter into per-physical-expert segments."""
        self.hidden_buf[:self.n_recv].index_copy_(0, self.place_slots, self.recv_buf)
        self.weights_buf[:self.n_recv].index_copy_(0, self.place_slots, self.wrecv_buf)

    def weight_sync(self, fc1_home: torch.Tensor, fc2_home: torch.Tensor = None,
                    group=None):
        """UltraEP weight_sync, `direct` plan: master weights -> replica
        slots, all pairs intra-NVLink-domain (asserted at layout build).
        fc1_home: this rank's [epn, ffn_shard, H] master fc1 shard."""
        group = group if group is not None else self.group
        cfg = self.cfg
        ops = []
        P2POp = self.dist.P2POp
        for dest, b, l, home in self.lay.weight_sync_pairs:
            e_local = l % cfg.epn
            if dest == self.rank:
                ops.append(P2POp(self.dist.irecv, self.replica_fc1[b], peer=home,
                                 group=group))
                if self.sync_fc2:
                    ops.append(P2POp(self.dist.irecv, self.replica_fc2[b],
                                     peer=home, group=group))
            elif home == self.rank:
                ops.append(P2POp(self.dist.isend, fc1_home[e_local].contiguous(),
                                 peer=dest, group=group))
                if self.sync_fc2:
                    assert fc2_home is not None
                    ops.append(P2POp(self.dist.isend, fc2_home[e_local].contiguous(),
                                     peer=dest, group=group))
        if ops:
            for req in self.dist.batch_isend_irecv(ops):
                req.wait()

    def gemm(self, gemm_only_op, fc1_home: torch.Tensor):
        """Per-segment GEMM over the received rows (no padding: the exact
        physical load imbalance IS the measurement)."""
        cfg = self.cfg
        for p, start, end, logical in self.lay.gemm_segments:
            w = (
                fc1_home[logical % cfg.epn] if p < cfg.epn
                else self.replica_fc1[p - cfg.epn]
            )
            gemm_only_op.forward(
                self.hidden_buf[start:end], w,
                output_buf=self.out_buf[start:end],
                fast_accum=False,
            )

    # -- accounting ----------------------------------------------------------

    def wire_matrix_row(self) -> list:
        """Bytes this rank puts on the wire per destination (hidden rows)."""
        return [c * self.wire_row_bytes for c in self.send_rows_by_dest]

    def weight_sync_recv_bytes(self) -> int:
        n = sum(1 for dest, _, _, _ in self.lay.weight_sync_pairs
                if dest == self.rank)
        per = self.ffn_size_shard * self.cfg.H * self.replica_fc1.element_size()
        if self.sync_fc2:
            per *= 2
        return n * per

    def dup_rows(self) -> int:
        """Wire rows MoonEP-style dedup would have saved: send entries whose
        (token, dest rank) pair repeats within one token's expansion."""
        tok = self.lay.send_row_index.cpu()
        if tok.numel() == 0:
            return 0
        # send entries are dest-major (counts per dest known), so rebuild the
        # per-entry dest vector from send_counts.
        dest = torch.cat([
            torch.full((c,), d, dtype=torch.int64)
            for d, c in enumerate(self.lay.send_counts)
        ])
        pair = tok * self.cfg.R + dest
        return int(pair.numel() - torch.unique(pair).numel())
