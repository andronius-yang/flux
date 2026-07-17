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
"""Traffic-matrix-driven MoE routing generation.

A traffic matrix file describes inter-rank byte movement:
    line 1: W (number of ranks)
    then W rows x W cols of whitespace-separated uint64 byte counts,
    M[src][dst] = bytes sent from src rank to dst rank.

Every entry must be a multiple of chunk_bytes (= token hidden dim * dtype size,
e.g. 4096 * 2B bf16 = 8192B, one token), and every row must sum to the same
per-source-rank byte budget.

The routing built here makes tokens homed on rank s (global token t lives on
rank t // tokens_per_rank) choose experts owned by rank d for exactly
M[s][d] / chunk_bytes of their (token, topk-slot) copies. Expert e is owned by
rank e // (nexperts // W), matching the EP convention of the MoE tests.
"""

from typing import Tuple

import torch

__all__ = [
    "parse_traffic_matrix",
    "traffic_matrix_to_choosed_experts",
]


def parse_traffic_matrix(path: str) -> torch.Tensor:
    """Parse a traffic matrix file into a [W, W] int64 CPU tensor of bytes."""
    with open(path, "r") as f:
        tokens = f.read().split()
    assert len(tokens) >= 1, f"empty traffic matrix file: {path}"
    nranks = int(tokens[0])
    values = [int(x) for x in tokens[1:]]
    assert len(values) == nranks * nranks, (
        f"traffic matrix {path} declares {nranks} ranks but has {len(values)} values"
        f" (expect {nranks * nranks})"
    )
    matrix = torch.tensor(values, dtype=torch.int64).reshape(nranks, nranks)
    assert (matrix >= 0).all(), f"traffic matrix {path} has negative entries"
    return matrix


def traffic_matrix_to_choosed_experts(
    matrix_bytes: torch.Tensor,
    nexperts: int,
    topk: int,
    chunk_bytes: int,
) -> torch.Tensor:
    """Build a choosed_experts [ntokens, topk] tensor realizing a traffic matrix.

    Args:
        matrix_bytes: [W, W] int64 bytes, M[src][dst]
        nexperts: total number of experts G, must be a multiple of W
        topk: experts choosed per token, must divide each row's chunk count
        chunk_bytes: bytes of one routed token copy (hidden dim * dtype size)
    Returns:
        choosed_experts: [W * tokens_per_rank, topk] int32 (cuda if available),
            with distinct experts per token; token t is homed on rank
            t // tokens_per_rank.
    """
    assert matrix_bytes.dim() == 2 and matrix_bytes.shape[0] == matrix_bytes.shape[1]
    W = matrix_bytes.shape[0]
    assert nexperts % W == 0, f"nexperts ({nexperts}) must be a multiple of nranks ({W})"
    experts_per_rank = nexperts // W

    assert (
        matrix_bytes % chunk_bytes == 0
    ).all(), f"traffic matrix entries must be multiples of chunk_bytes ({chunk_bytes})"
    chunks = (matrix_bytes // chunk_bytes).cpu()

    row_chunks = chunks.sum(dim=1)
    assert (
        row_chunks == row_chunks[0]
    ).all(), f"traffic matrix rows must have equal chunk sums, got {row_chunks.tolist()}"
    row_chunks = int(row_chunks[0])
    assert row_chunks % topk == 0, (
        f"per-source chunk count ({row_chunks}) must be divisible by topk ({topk})"
        f" so each token routes exactly topk copies"
    )
    tokens_per_rank = row_chunks // topk

    choosed_experts = torch.empty(W * tokens_per_rank, topk, dtype=torch.int32)
    for s in range(W):
        # split each dst rank's chunk count round-robin over its experts
        base = chunks[s].repeat_interleave(experts_per_rank) // experts_per_rank
        rem = chunks[s] % experts_per_rank
        extra = (
            torch.arange(experts_per_rank).repeat(W) < rem.repeat_interleave(experts_per_rank)
        ).to(torch.int64)
        counts = base + extra  # [nexperts], per-expert copies from source rank s
        assert int(counts.sum()) == row_chunks

        max_count = int(counts.max())
        if max_count > tokens_per_rank:
            eid = int(counts.argmax())
            raise ValueError(
                f"source rank {s} routes {max_count} copies to expert {eid} but only has"
                f" {tokens_per_rank} tokens: distinct topk experts per token is infeasible."
                f" Increase --G (more experts per rank) or decrease --topk."
            )

        # sorted expert list dealt column-major: token t gets L[k * tokens_per_rank + t].
        # sortedness + max_count <= tokens_per_rank guarantee distinct experts per token.
        L = torch.repeat_interleave(torch.arange(nexperts, dtype=torch.int32), counts)
        choosed_experts[s * tokens_per_rank : (s + 1) * tokens_per_rank] = L.view(
            topk, tokens_per_rank
        ).t()

    # self-check: distinct topk experts per token
    sorted_experts, _ = choosed_experts.sort(dim=1)
    assert (
        (sorted_experts[:, 1:] - sorted_experts[:, :-1]) > 0
    ).all(), "internal error: duplicate expert within a token's topk"

    # self-check: reconstructed per-(src, dst-rank) chunk counts match the matrix exactly
    owner_rank = choosed_experts.to(torch.int64) // experts_per_rank
    for s in range(W):
        got = torch.bincount(
            owner_rank[s * tokens_per_rank : (s + 1) * tokens_per_rank].flatten(), minlength=W
        )
        assert (got == chunks[s]).all(), (
            f"internal error: source rank {s} realizes chunks {got.tolist()}"
            f" != matrix {chunks[s].tolist()}"
        )

    if torch.cuda.is_available():
        choosed_experts = choosed_experts.cuda()
    return choosed_experts


def traffic_matrix_stats(
    matrix_bytes: torch.Tensor, topk: int, chunk_bytes: int
) -> Tuple[int, int, int]:
    """Return (nranks, tokens_per_rank, ntokens) implied by a traffic matrix."""
    W = matrix_bytes.shape[0]
    row_chunks = int((matrix_bytes.sum(dim=1) // chunk_bytes)[0])
    tokens_per_rank = row_chunks // topk
    return W, tokens_per_rank, W * tokens_per_rank


if __name__ == "__main__":
    import sys

    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/global/u1/y/yufeid/workspace/changchen/andrewy/profiling/traffic/"
        "matrices/a2av/4n_16r/2mib/a2av_4n_16r_dist_001.txt"
    )
    nexperts = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    topk = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    chunk_bytes = 8192

    matrix = parse_traffic_matrix(path)
    W, tokens_per_rank, ntokens = traffic_matrix_stats(matrix, topk, chunk_bytes)
    print(f"matrix: {path}")
    print(f"nranks: {W}, tokens_per_rank: {tokens_per_rank}, ntokens: {ntokens}")

    choosed_experts = traffic_matrix_to_choosed_experts(matrix, nexperts, topk, chunk_bytes)
    splits = torch.bincount(choosed_experts.flatten().to(torch.int64).cpu(), minlength=nexperts)
    print(f"choosed_experts: {choosed_experts.shape} on {choosed_experts.device}")
    print(f"per-expert splits: {splits.tolist()}")
    experts_per_rank = nexperts // W
    rows_per_rank = splits.view(W, experts_per_rank).sum(dim=1)
    print(f"per-rank gemm rows: {rows_per_rank.tolist()}")
    print("all self-checks passed")
