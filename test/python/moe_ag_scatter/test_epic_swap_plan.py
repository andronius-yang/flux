################################################################################
#
# CPU invariants for the EPIC §4.3 IN-KERNEL swap path (--migration inkernel):
# the fused exchange kernel's contract is sized and sequenced entirely by
# host-side properties tested here — no GPU needed.
#
#   1. plan_migration_swaps yields AT MOST ONE swap per rank per round and
#      pairs are strictly intra-node (this sizes the op's ctor scratch to
#      exactly one expert's fc1+fc2 and legalizes the single flag/rank).
#   2. The per-rank descriptor derivation (the apply_migration_inkernel
#      logic) is pair-consistent: A names B as peer iff B names A, and the
#      slots cross-match the swap tuple.
#   3. The GLOBAL swap-round sequence (the exchange flag epoch) bumps iff a
#      round has swaps, so any pair shares the epoch value and each rank's
#      observed epochs are strictly increasing (the op's FLUX_CHECK_GT).
#
################################################################################
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flux.testing.epic_semantics import (  # noqa: E402
    apply_swaps,
    plan_migration_swaps,
)

from test_epic_planner import NAMED_CASES, build  # noqa: E402
from test_ultraep_planner import Case  # noqa: E402


def derive_descriptor(swaps, rank):
    """Pure replica of apply_migration_inkernel's per-rank derivation."""
    mine = [s for s in swaps if rank in (s[0], s[2])]
    assert len(mine) <= 1, mine
    if not mine:
        return None
    rh, a, rl, b, _gain = mine[0]
    return (rl, a) if rank == rh else (rh, b)


@pytest.mark.parametrize("case", NAMED_CASES, ids=lambda c: c.name)
def test_at_most_one_swap_per_rank_and_intra_node(case):
    cfg, topk_all, tpe, pool_load, plan, nn = build(case)
    D = cfg.R // nn
    for _round in range(12):
        swaps = plan_migration_swaps(plan, 0.0, D)
        seen = set()
        for rh, a, rl, b, gain in swaps:
            assert rh // D == rl // D, "EPIC swaps are strictly intra-node"
            assert rh != rl and gain > 0
            assert 0 <= a < cfg.nlp and 0 <= b < cfg.nlp
            for r in (rh, rl):
                assert r not in seen, f"rank {r} in two swaps in one round"
                seen.add(r)
        if not swaps:
            break
        apply_swaps(plan, swaps)


@pytest.mark.parametrize("case", NAMED_CASES, ids=lambda c: c.name)
def test_descriptor_pair_consistency(case):
    cfg, topk_all, tpe, pool_load, plan, nn = build(case)
    D = cfg.R // nn
    for _round in range(12):
        swaps = plan_migration_swaps(plan, 0.0, D)
        desc = {r: derive_descriptor(swaps, r) for r in range(cfg.R)}
        participants = {r for r, d in desc.items() if d is not None}
        assert len(participants) == 2 * len(swaps)
        for rh, a, rl, b, _gain in swaps:
            assert desc[rh] == (rl, a), "heavy rank swaps ITS slot a"
            assert desc[rl] == (rh, b), "light rank swaps ITS slot b"
        if not swaps:
            break
        apply_swaps(plan, swaps)


def test_global_epoch_shared_and_monotone_per_rank():
    case = Case("epoch", alpha=1.2, seed=3)
    cfg, topk_all, tpe, pool_load, plan, nn = build(case)
    D = cfg.R // nn
    swap_seq = 0
    observed = {r: [] for r in range(cfg.R)}
    for _round in range(20):
        swaps = plan_migration_swaps(plan, 0.0, D)
        if swaps:
            swap_seq += 1  # replicated bump iff the round has swaps
            for rh, _a, rl, _b, _g in swaps:
                # both pair members observe the SAME epoch value (the GEQ
                # handshake requirement)
                observed[rh].append(swap_seq)
                observed[rl].append(swap_seq)
            apply_swaps(plan, swaps)
        else:
            break
    for r, seq in observed.items():
        assert seq == sorted(set(seq)), (
            f"rank {r} epochs not strictly increasing: {seq} "
            "(the op asserts FLUX_CHECK_GT per launch)")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"] + sys.argv[1:]))
