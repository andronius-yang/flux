################################################################################
#
# Copyright 2026 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
"""COMET + EPLB: an EPLB static placement (replication + re-homing) executed
by the UNMODIFIED COMET fused ops (dense allgather dispatch + dense combine).

Reviewer-requested baseline (2026-09-16): "overlap + placement", i.e. the
COMET substrate (tile-level comm/GEMM overlap) with DeepSeek's EPLB deciding
where each expert instance lives, instead of the default home placement.

Mechanism — the PHYSICAL-SLOT space (the MoonEP virtual-expert-space trick,
flux.testing.moonep_fused_map, applied to an EPLB plan): the fused ops have
no index-injection API; their only destination rule is expert homing
(dest = e // ep_nexperts). EPLB's plan already lives in exactly that space:
P = W * nlp physical slots, slot p hosted by rank p // nlp, p2l[p] = the
logical expert it replicates. So we

  1. build the plan once at setup (deployment scope, untimed, pool-oracle
     load sidecar — identical to the `eplb_l01` arm's placement),
  2. construct every op / buffer with nexperts = P (the ops then home
     slot p on rank p // nlp, i.e. execute the placement),
  3. per iteration (timed, SCHEMA rule 5) map this rank's logical routing
     [S, K] -> physical slots with EPLB's own sender-local replica rule
     (EplbIterPlanner.derive_fused: `local_spread` = per-source equal split,
     no exchange), all-gather the physical routing, and let the ops'
     derive_routed_meta / derive_combine_meta run unchanged,
  4. give every physical slot the canonical per-logical-expert weights
     (same seeds as the eplb arm's one-time placement), so replicas are
     bit-identical copies and the torch two-layer reference — which runs on
     the same physical layout — checks the fused pass with no special case.

Wire semantics: COMET's dense allgather is placement-invariant (every rank
receives every token), so the placement moves ONLY the grouped-GEMM rows
(and the dense combine's send rows) between ranks; the measurement is the
residual imbalance of the static placement on each batch, under COMET's
overlap. Nothing here touches the kernels.

Pure helpers (torch, CPU or CUDA); the driver hooks live in
test/python/moe_combined/test_moe_l0l1_traffic.py (--placement eplb).
"""

import hashlib
import json

import torch

from .eplb_semantics import (
    _FC1_SEED_BASE,
    _FC2_SEED_BASE,
    EplbIterPlanner,
    REPLICA_SELECT_MODES,
    build_eplb_plan,
    predicted_rows_per_rank,
    weight_placement_pairs,
)
from .ultraep_semantics import UltraEPConfig, UltraEPPlan, loads_from_topk

__all__ = [
    "load_pool_load",
    "build_comet_eplb_plan",
    "derive_physical_routing_all",
    "fill_canonical_slot_weights",
    "comet_eplb_stats",
]


def load_pool_load(path, G: int):
    """(pool_load list[G], sha16, source) from the runner's
    <mid>.eplb_load.json sidecar (version 1); path None -> (None, "", "batch")
    and the caller falls back to the batch's own load (self-oracle)."""
    if not path:
        return None, "", "batch"
    with open(path, "rb") as f:
        raw = f.read()
    blob = json.loads(raw)
    assert blob.get("version") == 1, f"unknown load-file version: {blob.get('version')}"
    assert int(blob["G"]) == G, f"load file G={blob['G']} != --G {G}"
    pool_load = blob["load"]
    assert len(pool_load) == G
    return pool_load, hashlib.sha256(raw).hexdigest()[:16], "pool"


def build_comet_eplb_plan(
    G: int,
    W: int,
    S: int,
    K: int,
    H: int,
    local_world_size: int,
    topk_all: torch.Tensor,
    pool_load,
    policy: str,
    rebalance_fn,
    redundant_per_rank: int = 2,
    interleave: bool = True,
    replica_select: str = "local_spread",
):
    """Setup-scope EPLB plan over the harness routing (CPU, replicated on
    every rank — the caller all-gathers plan.plan_hash() as the SPMD guard).

    topk_all: [W, S, K] int LOGICAL routing (CPU). pool_load: length-G
    predicted load or None (-> the batch's own aggregate load).
    Returns (cfg, plan, tpe). cfg.P is the physical slot count the COMET
    ops must be built with; cfg.nlp = G/W + redundant_per_rank slots/rank.
    """
    assert replica_select in ("local_spread", "local_static"), (
        f"COMET+EPLB is sender-local by construction; {replica_select!r} not supported"
    )
    assert replica_select in REPLICA_SELECT_MODES
    cfg = UltraEPConfig(
        S=S, K=K, G=G, R=W, H=H, D=local_world_size, R_red=redundant_per_rank,
        locality_aware=False, interleave=interleave,
    )
    tpe = loads_from_topk(cfg, topk_all)
    if pool_load is None:
        pool_load = tpe.long().sum(0).tolist()
    plan = build_eplb_plan(
        cfg, tpe, pool_load, policy, W // local_world_size, rebalance_fn,
        replica_select=replica_select,
    )
    # the global policy fills every slot (asserted inside build_eplb_plan:
    # lcnts.sum() == P); a hole would be an expert-less GEMM group
    assert int((plan.p2l >= 0).sum()) == cfg.P, "EPLB plan left a physical slot empty"
    return cfg, plan, tpe


def derive_physical_routing_all(planner: EplbIterPlanner, W: int) -> torch.Tensor:
    """[W*S, K] int32 physical routing for EVERY source rank, on the
    planner's device — the setup reference (gating args, buffer sizing,
    drift guard) for what the timed per-iteration path produces shard by
    shard. derive_fused is a pure function of (plan, topk_all[rank]), so
    looping the planner's rank field over all sources reproduces each
    rank's own in-window result bitwise."""
    saved = planner.rank
    try:
        shards = []
        for r in range(W):
            planner.rank = r
            shards.append(planner.derive_fused())
    finally:
        planner.rank = saved
    return torch.cat(shards, dim=0).contiguous()


def fill_canonical_slot_weights(
    plan: UltraEPPlan, rank: int, fc1: torch.Tensor, fc2: torch.Tensor
) -> int:
    """Overwrite this rank's per-slot weight tensors in place with the
    canonical per-LOGICAL-expert weights (seeded exactly like the eplb arm's
    EPLBLayer0Runner.make_canonical_fc1/fc2), so every replica of an expert
    is a bit-identical copy. fc1: [nlp, ffn, H]; fc2: [nlp, H, ffn] (the
    layer1 [E, N, K] layout). Returns the bytes a one-time placement would
    have moved onto this rank (slots whose logical expert is not one of the
    rank's default-home experts) — book-keeping only, nothing is sent."""
    cfg = plan.cfg
    nlp, epn = cfg.nlp, cfg.epn
    assert fc1.shape[0] == nlp and fc2.shape[0] == nlp, (fc1.shape, fc2.shape, nlp)
    ffn, H = fc1.shape[1], fc1.shape[2]
    assert tuple(fc2.shape[1:]) == (H, ffn), (fc2.shape, (H, ffn))
    moved_bytes = 0
    for j in range(nlp):
        logical = int(plan.p2l[rank * nlp + j])
        assert logical >= 0
        g1 = torch.Generator().manual_seed(_FC1_SEED_BASE + logical)
        w1 = torch.rand(ffn, H, dtype=torch.float32, generator=g1) * 0.01
        g2 = torch.Generator().manual_seed(_FC2_SEED_BASE + logical)
        w2 = torch.rand(H, ffn, dtype=torch.float32, generator=g2) * 0.01
        fc1[j].copy_(w1.to(fc1.dtype))
        fc2[j].copy_(w2.to(fc2.dtype))
        if not (rank * epn <= logical < (rank + 1) * epn):
            moved_bytes += (w1.numel() + w2.numel()) * fc1.element_size()
    return moved_bytes


def comet_eplb_stats(cfg: UltraEPConfig, plan: UltraEPPlan, tpe: torch.Tensor,
                     pool_load, phys_all: torch.Tensor) -> dict:
    """Cell-level facts for the recorder (rank 0), mirroring the eplb arm's
    columns so the two placements compare row-for-row. phys_all: [W*S, K]
    physical routing (any device)."""
    W = cfg.R
    rows_phys = torch.bincount(
        (phys_all.long().reshape(-1) // cfg.nlp).cpu(), minlength=W
    ).tolist()
    loads_g = tpe.long().sum(0)
    before = loads_g.reshape(W, cfg.epn).sum(1)
    mean = float(before.double().mean())
    imb_before = float(before.max()) / mean if mean else 1.0
    imb_after = max(rows_phys) / mean if mean else 1.0
    pred = predicted_rows_per_rank(plan, pool_load)
    pred_mean = sum(pred) / len(pred)
    imb_pred = max(pred) / pred_mean if pred_mean else 1.0
    rep = plan.replica_summary()
    return {
        "gemm_rows_per_rank_home": before.tolist(),
        "eplb_imbalance_before": imb_before,
        "eplb_imbalance_after": imb_after,
        "eplb_pred_imbalance": imb_pred,
        "eplb_replicas_total": rep["total_replicas"],
        "eplb_replicas_max_per_expert": rep["max_replicas_per_expert"],
        "eplb_slots_total": rep["slots_total"],
        "eplb_rehomed_slots": len(weight_placement_pairs(plan)),
        "eplb_physical_experts": cfg.P,
        "eplb_slots_per_rank": cfg.nlp,
    }
