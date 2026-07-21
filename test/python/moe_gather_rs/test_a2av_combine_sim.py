################################################################################
#
# Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
"""CPU-only simulation of the layer1 a2av_hier combine layout contract.

Validates, before any GPU run, that the mirror-layout ordering (send panel ==
layer0's recv layout, recv panel == layer0's send layout, copy-index tie-break)
round-trips correctly through:

  gemm rows (expert-major) --pack_index--> send panel (home-major)
    --direct per-(s,d) slices OR node aggregate + gateway sub-chunk forwarding-->
  recv panel (owner-major) --reduce_index--> per-token topk sum

for random routings, including zero (s,d) chunks, zero-row ranks, and both the
direct and hierarchical transport paths (which must land bit-identically).

Run: python3 test_a2av_combine_sim.py   (no GPU, no flux import)
"""

import torch


def stable_scatter_index(choosed_experts, nexperts):
    """calc_scatter_index_stable: copy (t, j) -> global expert-major row, within
    an expert ordered by global copy index (the tie-break the contract relies on)."""
    flat = choosed_experts.flatten().long()  # [ntokens * topk], copy-index order
    order = torch.argsort(flat, stable=True)  # rows in (expert, copy) order
    scatter = torch.empty_like(flat)
    scatter[order] = torch.arange(flat.numel(), dtype=torch.long)
    return scatter  # [n_copies]: copy -> A row


def build_indices(routing_idx, splits, rank, W):
    """The harness/op index math (sort-based reference form)."""
    m_full = routing_idx.numel()
    cpr = m_full // W
    E_loc = splits.numel() // W
    ep_m_start = int(splits[: rank * E_loc].sum())
    m_this = int(splits[rank * E_loc : (rank + 1) * E_loc].sum())
    iota_m = torch.arange(m_full, dtype=torch.long)
    copy_of_row = torch.empty(m_full, dtype=torch.long).scatter_(0, routing_idx, iota_m)
    copy_of_row = copy_of_row[ep_m_start : ep_m_start + m_this]
    home = copy_of_row // cpr
    pack_index = (home * max(m_this, 1) + torch.arange(m_this)).argsort()
    splits_cum = splits.cumsum(0)
    my = routing_idx[rank * cpr : (rank + 1) * cpr]
    e_of = torch.searchsorted(splits_cum, my, right=True)
    iota_c = torch.arange(cpr, dtype=torch.long)
    perm = (e_of * cpr + iota_c).argsort()
    reduce_index = torch.empty(cpr, dtype=torch.long).scatter_(0, perm, iota_c)
    return pack_index, reduce_index, ep_m_start, m_this


def simulate(W, L, G, topk, tokens_per_rank, seed, skew=False):
    torch.manual_seed(seed)
    NN = W // L
    E_loc = G // W
    ntokens = tokens_per_rank * W
    cpr = tokens_per_rank * topk
    # distinct experts per token (harness invariant); skew restricts choices to
    # the first half of the experts so later owner ranks have ZERO gemm rows
    pool = G // 2 if skew else G
    choosed = torch.stack([torch.randperm(pool)[:topk] for _ in range(ntokens)])
    splits = torch.bincount(choosed.flatten(), minlength=G).long()
    routing_idx = stable_scatter_index(choosed, G)  # copy -> A row
    cnt = (
        torch.bincount(
            (torch.arange(ntokens, dtype=torch.long) // tokens_per_rank).repeat_interleave(topk)
            * G
            + choosed.flatten().long(),
            minlength=W * G,
        )
        .view(W, G)
        .long()
    )
    # combine chunk matrix C[s][d] = sum over s's experts of cnt[d][e]
    C = torch.stack([cnt[:, s * E_loc : (s + 1) * E_loc].sum(1) for s in range(W)])  # [s][d]
    assert torch.equal(C.sum(1), splits.view(W, E_loc).sum(1))  # rows == owner gemm rows
    assert (C.sum(0) == cpr).all()  # every home receives exactly cpr copies

    # payload: each gemm row carries its global copy index (scalar "hidden")
    a_row_to_copy = torch.empty(ntokens * topk, dtype=torch.long).scatter_(
        0, routing_idx, torch.arange(ntokens * topk, dtype=torch.long)
    )

    recv_direct = [torch.full((cpr,), -1, dtype=torch.long) for _ in range(W)]
    recv_hier = [torch.full((cpr,), -1, dtype=torch.long) for _ in range(W)]
    send_panels = []
    packs = []
    for s in range(W):
        pack_index, _, ep_m_start, m_this = build_indices(routing_idx, splits, s, W)
        gemm_rows = a_row_to_copy[ep_m_start : ep_m_start + m_this]  # A-order payload
        send_panels.append(gemm_rows[pack_index])  # home-major panel
        packs.append(pack_index)

    def send_off(s, d):
        return int(C[s, :d].sum())

    def recv_off_of(s, d):
        return int(C[:s, d].sum())

    # direct transport: per-(s, d) contiguous slice
    for s in range(W):
        for d in range(W):
            rows = int(C[s, d])
            recv_direct[d][recv_off_of(s, d) : recv_off_of(s, d) + rows] = send_panels[s][
                send_off(s, d) : send_off(s, d) + rows
            ]

    # hierarchical transport: intra-node direct; inter-node ONE aggregate to the
    # same-local-rank gateway, then per-destination sub-chunk forwarding
    def node_chunk(s, n):
        return int(C[s, n * L : (n + 1) * L].sum())

    for s in range(W):
        sn, slr = s // L, s % L
        for d in range(sn * L, (sn + 1) * L):  # intra-node: direct slices
            rows = int(C[s, d])
            recv_hier[d][recv_off_of(s, d) : recv_off_of(s, d) + rows] = send_panels[s][
                send_off(s, d) : send_off(s, d) + rows
            ]
        for tn in range(NN):  # inter-node: aggregate -> gateway staging
            if tn == sn:
                continue
            g = tn * L + slr  # gateway rank
            agg = send_panels[s][send_off(s, tn * L) : send_off(s, tn * L) + node_chunk(s, tn)]
            # gateway forwards sub-chunks: segment interior is ascending global d
            within = 0
            for d in range(tn * L, (tn + 1) * L):
                rows = int(C[s, d])
                recv_hier[d][recv_off_of(s, d) : recv_off_of(s, d) + rows] = agg[
                    within : within + rows
                ]
                within += rows
            assert within == node_chunk(s, tn)
            assert g // L == tn  # relay is on the destination node: one hop max

    for d in range(W):
        assert (recv_direct[d] >= 0).all(), f"rank {d}: recv gap (direct)"
        assert torch.equal(recv_direct[d], recv_hier[d]), f"rank {d}: hier != direct"

    # reduce: local copy (t, j) -> recv row must recover exactly copy t*topk+j
    for d in range(W):
        _, reduce_index, _, _ = build_indices(routing_idx, splits, d, W)
        got = recv_direct[d][reduce_index]
        want = torch.arange(d * cpr, (d + 1) * cpr, dtype=torch.long)
        assert torch.equal(got, want), f"rank {d}: reduce index mismatch"


if __name__ == "__main__":
    cases = [
        dict(W=4, L=4, G=8, topk=2, tokens_per_rank=6),    # single node
        dict(W=8, L=4, G=16, topk=4, tokens_per_rank=8),   # 2 nodes
        dict(W=16, L=4, G=32, topk=4, tokens_per_rank=16), # 4 nodes
        dict(W=8, L=4, G=8, topk=1, tokens_per_rank=4),    # topk=1
        dict(W=8, L=2, G=16, topk=3, tokens_per_rank=5),   # odd shapes
        dict(W=8, L=4, G=16, topk=2, tokens_per_rank=6, skew=True),  # zero-row owner ranks
    ]
    for case in cases:
        for seed in range(5):
            simulate(seed=seed, **case)
        print(f"ok: {case}")
    print("✅ a2av_hier combine layout contract holds (pack -> transport -> reduce)")
