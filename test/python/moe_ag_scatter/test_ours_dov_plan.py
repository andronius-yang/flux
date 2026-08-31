# Unit test: OURS direct-OVERLAP (dov) combine metadata math (no GPU,
# no dist). The dov runner replaces the l0 leg with the flat-mode fused
# op, whose OUT layout is the STABLE expert-major scatter
# (argsort(stable).argsort() over the global vce — bit-contracted by the
# harness setup audit against derive_routed_meta). This test validates
# the pure-index algebra OursDirectOverlapRunner.plan_meta builds on top
# of that layout, against brute-force simulations:
#   * splits/segments from sps column algebra
#   * place_slots: the arrival-order (src-major (s, p, token)) sequence
#     maps ONTO the stable OUT layout row holding the SAME copy — i.e.
#     the prefix-sum algebra AND the token-ascending interior assumption
#     agree with the stable scatter convention
#   * comb_dst == the copy ids of the reverse-wire return stream
#     (owners ascending, each owner's block = its cpack order restricted
#     to me), simulated independently — and it equals
#     argsort(my_vce, stable), the runner's cheap local derivation
#   * scale scatter: out row r receives exactly its copy's gate weight
#
# Run: python3 test/python/moe_ag_scatter/test_ours_dov_plan.py

import sys

import torch


def dov_meta(vce, S, K, R, nlp, rank):
    """CPU twin of OursDirectOverlapRunner.plan_meta's index algebra."""
    gpe = nlp + 1
    E_virt = R * gpe
    ep_start = rank * gpe
    vce_flat = vce.long().reshape(-1)
    # the op's stable scatter (harness-audited convention)
    sc = vce_flat.argsort(stable=True).argsort()
    splits = torch.bincount(vce_flat, minlength=E_virt)
    home = (torch.arange(R * S, dtype=torch.int64) // S).repeat_interleave(K)
    sps = torch.bincount(home * E_virt + vce_flat,
                         minlength=R * E_virt).view(R, E_virt)

    cnt = sps[:, ep_start:ep_start + gpe]           # [R, gpe]
    m_this = int(cnt.sum())
    in_splits = sps[rank].view(R, gpe).sum(1)
    out_splits = cnt.sum(1)
    seg_rows = cnt.sum(0)
    seg_start = torch.zeros(gpe, dtype=torch.int64)
    seg_start[1:] = torch.cumsum(seg_rows, 0)[:-1]

    flat = cnt.reshape(-1)
    arr_start = flat.cumsum(0) - flat
    col = cnt.sum(0)
    seg_start_dev = col.cumsum(0) - col
    out_start = (seg_start_dev.unsqueeze(0) + (cnt.cumsum(0) - cnt)
                 ).reshape(-1)
    place = (torch.arange(m_this, dtype=torch.int64)
             + torch.repeat_interleave(out_start - arr_start, flat))

    my_vce = vce.long()[rank * S:(rank + 1) * S].reshape(-1)
    comb_dst = torch.argsort(my_vce, stable=True)
    return dict(sc=sc, sps=sps, m_this=m_this, in_splits=in_splits,
                out_splits=out_splits, seg_rows=seg_rows,
                seg_start=seg_start, place=place, comb_dst=comb_dst,
                splits=splits)


def one_case(seed, S=16, K=4, R=8, nlp=6):
    gen = torch.Generator().manual_seed(seed)
    gpe = nlp + 1
    P = R * nlp
    phys = torch.stack([
        torch.randperm(P, generator=gen)[:K] for _ in range(R * S)
    ]).view(R * S, K)
    vce = (phys // nlp) * gpe + 1 + phys % nlp      # pad-FIRST vce
    vce_flat = vce.long().reshape(-1)
    E_virt = R * gpe
    owner = vce_flat // gpe
    home = (torch.arange(R * S, dtype=torch.int64) // S).repeat_interleave(K)

    for rank in range(R):
        m = dov_meta(vce, S, K, R, nlp, rank)
        ep_start = rank * gpe
        mine = owner == rank
        m_start = int((vce_flat < ep_start).sum())   # rows before my block

        # OUT layout: out_copy[row] = global copy id, my block only
        out_copy = torch.full((m['m_this'],), -1, dtype=torch.int64)
        rows_my = m['sc'][mine] - m_start
        assert rows_my.min() >= 0 and rows_my.max() < m['m_this'], \
            "stable scatter rows not contiguous in my block"
        out_copy[rows_my] = torch.arange(R * S * K,
                                         dtype=torch.int64)[mine]
        assert (out_copy >= 0).all()

        # arrival sequence: src-major, vslot within, copy-id (token) order
        arr_copy = []
        for s in range(R):
            for p in range(gpe):
                sel = (mine & (home == s)
                       & (vce_flat == ep_start + p))
                arr_copy += torch.arange(R * S * K,
                                         dtype=torch.int64)[sel].tolist()
        arr_copy = torch.tensor(arr_copy, dtype=torch.int64)
        assert arr_copy.numel() == m['m_this']

        # THE overlap-arm contract: place maps arrival j onto the OUT row
        # holding the same copy
        assert torch.equal(out_copy[m['place']], arr_copy), \
            "place_slots does not map arrival onto the stable OUT layout"

        # reverse-wire return stream: owners ascending, each owner's
        # cpack (= arrival) order restricted to source ME
        ret_copy = []
        for o in range(R):
            o_start = o * gpe
            for p in range(gpe):
                sel = ((home == rank)
                       & (vce_flat == o_start + p))
                ret_copy += torch.arange(R * S * K,
                                         dtype=torch.int64)[sel].tolist()
        ret_local = (torch.tensor(ret_copy, dtype=torch.int64)
                     - rank * S * K)
        assert ret_local.numel() == S * K
        assert torch.equal(m['comb_dst'], ret_local), \
            "comb_dst != the simulated reverse-wire return order"

        # scale scatter lands each copy's weight on its own out row
        probs = torch.rand(R * S * K, generator=gen)
        idx = m['sc'] - m_start
        valid = (idx >= 0) & (idx < m['m_this'])
        scale = torch.zeros(m['m_this'] + 1)
        scale.scatter_(0, torch.where(valid, idx,
                                      torch.full_like(idx, m['m_this'])),
                       probs)
        assert torch.equal(scale[:m['m_this']], probs[mine][
            torch.argsort(rows_my)]), "scale scatter mismatch"
    return True


def main():
    for seed in (1, 2, 3):
        one_case(seed)
    one_case(7, S=64, K=2, R=4, nlp=3)
    one_case(11, S=8, K=8, R=4, nlp=16)
    print("test_ours_dov_plan: ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
