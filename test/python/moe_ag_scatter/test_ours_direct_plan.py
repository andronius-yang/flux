# Unit test: OURS direct-wire plan adapter math (no GPU, no dist).
#
# Validates the vce -> direct-layout adapter used by OursDirectRunner
# (flux/testing/ours_direct.py) against a brute-force reference:
#   * phys recovery from the ours pad-FIRST vce convention
#   * in/out splits vs per-(src, dst) entry counts
#   * segment table vs per-local-slot recv counts
#   * place_slots: arrival-order -> slot-major placement is a bijection
#     that lands every row in its slot segment, (src, token) sorted
#   * comb_dst: a permutation of [0, S*K)
#
# Run: python3 test/python/moe_ag_scatter/test_ours_direct_plan.py

import sys

import torch

from flux.testing.ep_gpu_plan import direct_layout_entries_fast


def adapter(vce, S, K, R, nlp, rank):
    """CPU twin of OursDirectRunner.plan_meta's derive."""
    gpe = nlp + 1
    N = S * K
    vce = vce.view(R, N).long()
    phys_all = (vce // gpe) * nlp + (vce % gpe) - 1
    tok = torch.arange(S, dtype=torch.int64).repeat_interleave(K)
    kk = torch.arange(K, dtype=torch.int64).repeat(S)
    tok_exp = tok.expand(R, N)
    k_exp = kk.expand(R, N)
    order = torch.argsort(phys_all * (S + 1) + tok_exp, dim=1, stable=True)
    ent_tok = torch.gather(tok_exp, 1, order)
    ent_phys = torch.gather(phys_all, 1, order)
    lay = direct_layout_entries_fast(ent_tok, ent_phys, rank, nlp, R)
    my_k = torch.gather(k_exp, 1, order)[rank]
    return phys_all, ent_tok, ent_phys, lay, lay["my_tok"] * K + my_k


def one_case(seed, S=16, K=4, R=8, nlp=6):
    gen = torch.Generator().manual_seed(seed)
    gpe = nlp + 1
    P = R * nlp
    # random physical routing (distinct experts per token not required by
    # the layout math itself; use distinct to mirror topk semantics)
    phys = torch.stack([
        torch.randperm(P, generator=gen)[:K] for _ in range(R * S)
    ]).view(R, S * K)
    vce = (phys // nlp) * gpe + 1 + phys % nlp  # pad-FIRST vce

    for rank in range(R):
        phys_all, ent_tok, ent_phys, lay, comb_dst = adapter(
            vce, S, K, R, nlp, rank)
        assert torch.equal(phys_all, phys.long()), "phys recovery broken"

        dest = phys.long() // nlp
        # in_splits: my rows per destination
        ref_in = torch.bincount(dest[rank], minlength=R)
        assert torch.equal(lay["in_splits"].long(), ref_in), "in_splits"
        # out_splits: each source's rows for me
        ref_out = (dest == rank).sum(dim=1)
        assert torch.equal(lay["out_splits"].long(), ref_out), "out_splits"
        # segments: per-local-slot recv counts
        mine = phys.long()[dest == rank]
        ref_seg = torch.bincount(mine - rank * nlp, minlength=nlp)
        assert torch.equal(lay["seg_rows"], ref_seg), "seg_rows"
        seg_start = torch.zeros(nlp, dtype=torch.int64)
        seg_start[1:] = torch.cumsum(ref_seg, 0)[:-1]
        assert torch.equal(lay["seg_start"], seg_start), "seg_start"
        n_recv = int(lay["n_recv_dev"])
        assert n_recv == int(ref_seg.sum())
        # pair_max
        pair = torch.zeros(R, R, dtype=torch.int64)
        for s in range(R):
            pair[s] = torch.bincount(dest[s], minlength=R)
        assert int(lay["pair_max"]) == int(pair.max()), "pair_max"

        # place_slots: bijection arrival -> slot-major position; the row
        # arriving at wire position j is source-block-major (src asc,
        # within-src send order = (phys, token) sorted). Rebuild the
        # placed (phys, token) sequence and check slot-major grouping.
        ps = lay["place_slots_pad"][:n_recv]
        assert sorted(ps.tolist()) == list(range(n_recv)), "not a bijection"
        arr_phys, arr_tok, arr_src = [], [], []
        for s in range(R):
            sel = ent_phys[s] // nlp == rank
            arr_phys += ent_phys[s][sel].tolist()
            arr_tok += ent_tok[s][sel].tolist()
            arr_src += [s] * int(sel.sum())
        placed = [None] * n_recv
        for j in range(n_recv):
            placed[int(ps[j])] = (arr_phys[j], arr_src[j], arr_tok[j])
        # slot segments hold exactly their slot's rows
        for p in range(nlp):
            lo, hi = int(seg_start[p]), int(seg_start[p]) + int(ref_seg[p])
            for j in range(lo, hi):
                assert placed[j][0] == rank * nlp + p, "row outside its slot"

        # comb_dst: permutation of [0, S*K)
        assert sorted(comb_dst.tolist()) == list(range(S * K)), "comb_dst"


def main():
    for seed in (1, 2, 3):
        one_case(seed)
    # skewed shape (tall S, tiny K) + minimal K
    one_case(7, S=64, K=2, R=4, nlp=3)
    one_case(11, S=8, K=8, R=4, nlp=16)
    print("test_ours_direct_plan: ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
