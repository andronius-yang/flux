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
"""Index math for the FAST alltoallv layer0 baseline (pure torch, no flux import).

Wire contract with FAST:
- send buffer (rank s) is destination-major: one contiguous block per destination
  rank d of M[s][d] bytes, blocks in global rank order 0..W-1;
- receive buffer (rank d) is source-major: one contiguous segment per source rank
  s of M[s][d] bytes, segments in global rank order, each segment in send order.

Because expert owner(e) = e // (G // W) is monotone in e, ONE stable argsort of
this rank's flattened choosed_experts simultaneously yields destination-major
blocks AND expert-ascending order within each block. The receive side then needs
a single gather (built host-side from cnt[s][e]) to become expert-contiguous in
the reference A-order (expert, source rank, local token, slot) — bitwise equal
to the block of scatter_inputs the reference grouped GEMM consumes.
"""

import torch


def build_pack_index(choosed_experts_local: torch.Tensor, topk: int) -> torch.Tensor:
    """Send-side row gather: send_rows = inputs_shard.index_select(0, pack_index).

    choosed_experts_local: [tokens_this_rank, topk] expert ids of this rank's tokens.
    Returns int64 [tokens_this_rank * topk] local token id per send row; rows are
    ordered by (expert asc, local token asc, slot asc), which is destination-major
    with expert-grouped blocks (owner is monotone in expert id).
    """
    ce = choosed_experts_local.reshape(-1).long().cpu()
    pack_order = torch.argsort(ce, stable=True)
    return pack_order // topk


def build_unpack_index(cnt: torch.Tensor, dst_rank: int, nexperts: int, world_size: int):
    """Recv-side row gather for rank dst_rank.

    cnt: [W, nexperts] int, cnt[s][e] = copies source rank s sends to expert e
    (column sums equal the global splits).
    Returns (unpack_index int64 [nrows_ep], split_cpu int32 [experts_per_rank]):
    gemm_input = recv_rows.index_select(0, unpack_index) is expert-contiguous, and
    within each expert ordered by (source rank asc, send order) — the reference
    A-order.
    """
    epr = nexperts // world_size
    cnt_local = cnt[:, dst_rank * epr : (dst_rank + 1) * epr].long().cpu()  # [W, epr]
    flow_rows = cnt_local.sum(dim=1)  # rows in the (s -> dst) wire flow
    flow_start = torch.cumsum(flow_rows, dim=0) - flow_rows  # exclusive cumsum
    off = torch.cumsum(cnt_local, dim=1) - cnt_local  # within-flow expert offsets
    pieces = []
    for j in range(epr):
        for s in range(world_size):
            n = int(cnt_local[s, j])
            if n == 0:
                continue
            base = int(flow_start[s] + off[s, j])
            pieces.append(torch.arange(base, base + n, dtype=torch.long))
    if pieces:
        unpack_index = torch.cat(pieces)
    else:
        unpack_index = torch.empty(0, dtype=torch.long)
    split_cpu = cnt_local.sum(dim=0).int()
    return unpack_index, split_cpu


def emulate_fast_wire(send_rows_per_rank, chunks):
    """CPU emulation of the FAST wire for the index-math unit test.

    send_rows_per_rank: list of W tensors [tokens*topk, H]; rank s's packed send
    rows (destination-major, from build_pack_index).
    chunks: [W, W] long, chunks[s][d] = rows s sends to d.
    Returns list of W tensors: rank d's source-major receive buffer.
    """
    world_size = len(send_rows_per_rank)
    send_start = torch.cumsum(chunks, dim=1) - chunks  # [W, W] exclusive row offsets
    recvs = []
    for d in range(world_size):
        segs = []
        for s in range(world_size):
            lo = int(send_start[s, d])
            segs.append(send_rows_per_rank[s][lo : lo + int(chunks[s, d])])
        recvs.append(torch.cat(segs, dim=0))
    return recvs
