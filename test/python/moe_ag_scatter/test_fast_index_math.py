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
"""CPU-only unit test of the FAST-baseline pack/unpack index math.

Emulates the FAST wire as pure permutations (no NVSHMEM, no GPU, no dist) and
asserts, for every destination rank, bitwise equality between the unpacked
receive buffer and the reference expert-major scatter order (the block of
scatter_inputs the grouped GEMM consumes). Runs on a login node:

    python3 test/python/moe_ag_scatter/test_fast_index_math.py
"""

import torch

from fast_baseline_utils import build_pack_index, build_unpack_index, emulate_fast_wire


def reference_scatter_order(choosed_experts: torch.Tensor) -> torch.Tensor:
    # calc_scatter_index_stable equivalent: stable argsort of the flattened
    # expert choices = A-order (expert, global token, slot)
    return torch.argsort(choosed_experts.reshape(-1).long(), stable=True)


def run_case(world_size: int, nexperts: int, topk: int, tokens_per_rank: int, h: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    ntokens = world_size * tokens_per_rank
    # distinct experts per token (same invariant traffic_matrix_to_choosed_experts enforces)
    choosed_experts = torch.stack(
        [torch.randperm(nexperts, generator=g)[:topk] for _ in range(ntokens)]
    ).int()

    epr = nexperts // world_size
    src_of_copy = (torch.arange(ntokens) // tokens_per_rank).repeat_interleave(topk)
    e_of_copy = choosed_experts.reshape(-1).long()
    cnt = (
        torch.bincount(src_of_copy * nexperts + e_of_copy, minlength=world_size * nexperts)
        .view(world_size, nexperts)
        .int()
    )
    owner_of_copy = e_of_copy // epr
    chunks = (
        torch.bincount(src_of_copy * world_size + owner_of_copy, minlength=world_size * world_size)
        .view(world_size, world_size)
        .long()
    )

    # every rank's token payload: unique values so any misplacement is caught
    inputs = torch.arange(ntokens * h, dtype=torch.float32).view(ntokens, h)

    send_rows_per_rank = []
    for s in range(world_size):
        ce_local = choosed_experts[s * tokens_per_rank : (s + 1) * tokens_per_rank]
        pack_index = build_pack_index(ce_local, topk) + s * tokens_per_rank
        send_rows_per_rank.append(inputs.index_select(0, pack_index))

    recvs = emulate_fast_wire(send_rows_per_rank, chunks)

    # reference: global stable scatter order, expert-major
    scatter_order = reference_scatter_order(choosed_experts)
    scatter_inputs = inputs.index_select(0, scatter_order // topk)
    splits = torch.bincount(e_of_copy, minlength=nexperts)

    for d in range(world_size):
        unpack_index, split_cpu = build_unpack_index(cnt, d, nexperts, world_size)
        gemm_input = recvs[d].index_select(0, unpack_index)
        input_offset = int(splits[: d * epr].sum())
        nrows_ep = int(splits[d * epr : (d + 1) * epr].sum())
        ref = scatter_inputs[input_offset : input_offset + nrows_ep]
        assert torch.equal(split_cpu.long(), splits[d * epr : (d + 1) * epr]), f"splits d={d}"
        assert gemm_input.shape == ref.shape, f"shape d={d}: {gemm_input.shape} vs {ref.shape}"
        assert torch.equal(gemm_input, ref), f"bitwise mismatch for dst rank {d}"


if __name__ == "__main__":
    cases = [
        dict(world_size=16, nexperts=32, topk=4, tokens_per_rank=64, h=8, seed=0),
        dict(world_size=16, nexperts=32, topk=4, tokens_per_rank=128, h=4, seed=1),
        dict(world_size=16, nexperts=32, topk=2, tokens_per_rank=64, h=8, seed=2),
        dict(world_size=16, nexperts=32, topk=1, tokens_per_rank=64, h=8, seed=3),
        dict(world_size=16, nexperts=16, topk=4, tokens_per_rank=32, h=8, seed=4),
        dict(world_size=4, nexperts=32, topk=4, tokens_per_rank=64, h=8, seed=5),
        dict(world_size=4, nexperts=4, topk=2, tokens_per_rank=16, h=8, seed=6),
    ]
    for case in cases:
        run_case(**case)
        print(f"OK {case}")
    print("all index-math cases passed")
