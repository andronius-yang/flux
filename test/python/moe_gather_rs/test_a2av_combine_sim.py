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

    # eager-reduce lane contract: the kernel maps each recv row to its source
    # rank by binary search over recv_cum (prefix sums of C[:, d]) and polls
    # that lane's per-split signal. The lane must equal the true owner rank of
    # the copy, and the per-source ranges must partition each token's rows —
    # this is what makes arrival-order accumulation sum each copy exactly once.
    for d in range(W):
        _, reduce_index, _, _ = build_indices(routing_idx, splits, d, W)
        recv_cum = C[:, d].cumsum(0)
        lanes = torch.searchsorted(recv_cum, reduce_index, right=True)
        local_copy = torch.arange(cpr, dtype=torch.long)
        t_global = d * tokens_per_rank + local_copy // topk
        owners = choosed[t_global, local_copy % topk].long() // E_loc
        assert torch.equal(lanes, owners), f"rank {d}: eager lane != copy owner"


def simulate_compress(W, L, G, topk, tokens_per_rank, seed, skew=False):
    """Compress (dedup) transport: one partial per (token, source node) crosses
    the wire. Source rank (n, lr) owns all wire rows destined to rank (tn, lr)
    (same-lr end-to-end); the source node's copies converge on it (conv panel),
    are merged per token (pre-reduce), and land in the destination's recv panel
    at offsets given by the compress chunk matrix C'. Payloads are SETS of
    global copy indices, so the final per-token union check fails on any
    double-counted or missing copy regardless of accumulation order.
    """
    torch.manual_seed(seed)
    NN = W // L
    E_loc = G // W
    ntokens = tokens_per_rank * W
    cpr = tokens_per_rank * topk
    pool = G // 2 if skew else G
    choosed = torch.stack([torch.randperm(pool)[:topk] for _ in range(ntokens)])
    splits = torch.bincount(choosed.flatten(), minlength=G).long()
    routing_idx = stable_scatter_index(choosed, G)
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
    C = torch.stack([cnt[:, s * E_loc : (s + 1) * E_loc].sum(1) for s in range(W)])

    # dedup counts, transposed U: Ucomb[d][n] = distinct tokens homed at rank d
    # with >= 1 copy owned on node n (the layer0 U-matrix recipe, consumed
    # transposed). This is the compress wire row count node n -> rank d.
    owner = choosed.long() // E_loc  # [ntokens, topk] owner rank per copy
    on_node = torch.zeros(ntokens, NN, dtype=torch.bool)
    on_node.scatter_(1, owner // L, True)
    Ucomb = (
        on_node.view(W, tokens_per_rank, NN).sum(1).long().t().contiguous()
    )  # [NN, W] -> transpose to index [d][n]
    Ucomb = Ucomb.t()  # [W(d), NN(n)]

    # compress chunk matrix C'[s][d]: own-node lanes keep per-rank chunks; the
    # remote lane materializes only at the same-lr source rank as Ucomb[d][n]
    Cp = torch.zeros(W, W, dtype=torch.long)
    for s in range(W):
        for d in range(W):
            if s // L == d // L:
                Cp[s, d] = C[s, d]
            elif s % L == d % L:
                Cp[s, d] = Ucomb[d, s // L]

    def send_off(s, d):
        return int(C[s, :d].sum())

    def recv_off_cp(s, d):
        return int(Cp[:s, d].sum())

    # payload: per gemm row, the singleton set of its global copy index
    a_row_to_copy = torch.empty(ntokens * topk, dtype=torch.long).scatter_(
        0, routing_idx, torch.arange(ntokens * topk, dtype=torch.long)
    )
    send_panels = []
    for s in range(W):
        pack_index, _, ep_m_start, m_this = build_indices(routing_idx, splits, s, W)
        send_panels.append(a_row_to_copy[ep_m_start : ep_m_start + m_this][pack_index])

    recv = [[None] * (int(Cp[:, d].sum())) for d in range(W)]

    def deliver(d, off, rows_payload):
        for i, p in enumerate(rows_payload):
            assert recv[d][off + i] is None, f"rank {d}: recv row {off+i} written twice"
            recv[d][off + i] = p

    for s in range(W):
        sn = s // L
        # own node: direct per-rank puts, no dedup (NVLink; zero wire savings)
        for d in range(sn * L, (sn + 1) * L):
            rows = [frozenset([int(c)]) for c in send_panels[s][send_off(s, d) : send_off(s, d) + int(C[s, d])]]
            deliver(d, recv_off_cp(s, d), rows)
    # remote nodes: convergence -> pre-reduce at (n, lr(d)) -> one put per (n, d)
    for n in range(NN):
        for tn in range(NN):
            if tn == n:
                continue
            for dl in range(L):
                d = tn * L + dl
                gw = n * L + dl  # source-side wire owner, same-lr as d
                # conv panel at gw for dest d: peer segments ls ascending, each
                # peer's rows in its send-panel (s -> d) order
                conv = []
                for ls in range(L):
                    s = n * L + ls
                    conv.extend(
                        int(c)
                        for c in send_panels[s][send_off(s, d) : send_off(s, d) + int(C[s, d])]
                    )
                # pre-reduce: merge copies per token, wire rows token-ascending
                # by home-local index (global copy index // topk orders tokens)
                by_token = {}
                for c in conv:
                    by_token.setdefault(c // topk, set()).add(c)
                tokens_sorted = sorted(by_token)
                assert len(tokens_sorted) == int(Ucomb[d, n]), (
                    f"wire rows {len(tokens_sorted)} != Ucomb[{d}][{n}] {int(Ucomb[d, n])}"
                )
                wire_rows = [frozenset(by_token[t]) for t in tokens_sorted]
                deliver(d, recv_off_cp(gw, d), wire_rows)

    # destination CSR (red_ptr/red_row), built INDEPENDENTLY via the transposed
    # one-cumsum identity: remote merged row position = exclusive count of
    # earlier home tokens with >= 1 copy on that node
    for d in range(W):
        dn, dl = d // L, d % L
        assert all(r is not None for r in recv[d]), f"rank {d}: recv gap (compress)"
        _, reduce_index, _, _ = build_indices(routing_idx, splits, d, W)
        tok_rows = [[] for _ in range(tokens_per_rank)]
        for i in range(cpr):  # own-node copies via today's reduce_index lanes
            row = int(reduce_index[i])
            # lane under C (uncompressed): recover, keep only own-node lanes
            lane = int(torch.searchsorted(C[:, d].cumsum(0), torch.tensor(row), right=True))
            if lane // L != dn:
                continue
            # own-node rows sit at the same intra-lane position under C'
            row_cp = recv_off_cp(lane, d) + (row - int(C[:lane, d].sum()))
            tok_rows[i // topk].append(row_cp)
        node_flags = on_node[d * tokens_per_rank : (d + 1) * tokens_per_rank]  # [tpr, NN]
        for n in range(NN):
            if n == dn:
                continue
            gw = n * L + dl
            cum = 0
            for tl in range(tokens_per_rank):
                if bool(node_flags[tl, n]):
                    tok_rows[tl].append(recv_off_cp(gw, d) + cum)
                    cum += 1
            assert cum == int(Ucomb[d, n])
        # reduce: union over the token's CSR rows == exactly its topk copies
        for tl in range(tokens_per_rank):
            got = set()
            total = 0
            for row in tok_rows[tl]:
                got |= recv[d][row]
                total += len(recv[d][row])
            t_global = d * tokens_per_rank + tl
            want = set(range(t_global * topk, (t_global + 1) * topk))
            assert got == want, f"rank {d} token {tl}: union {got} != {want}"
            assert total == topk, f"rank {d} token {tl}: copy double-counted"


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
    for case in cases:
        for seed in range(5):
            simulate_compress(seed=seed, **case)
        print(f"ok compress: {case}")
    print("✅ a2av_hier combine layout contract holds (pack -> transport -> reduce)")
    print("✅ compress contract holds (conv -> pre-reduce -> C' wire -> CSR reduce)")
