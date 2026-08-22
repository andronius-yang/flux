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
"""MoonEP LAYER1: the virtual-expert-space combine through the NORMAL fused
layer1 op (GemmGroupedV2GatherRSOp.forward_gather_rs — gemm2 + gather +
topk-reduce + reduce-scatter), a moonep twin of test_moe_gather_rs_traffic.py.

The MoonEP plan is re-encoded as routing over R*(epn+B) virtual experts (the
same mapping the fused layer0 arm uses): a token computed at home under a
prefetch-slot expert combines LOCALLY — no cross-node wire for replicated
rows — so the combine traffic is the dispatch matrix transposed MINUS the
replicated rows. The fused op needs zero changes: G := E_virt, per-rank
experts := gpe = epn + B, weights := the [gpe, N, K] virtual tensor whose
slot rows replicate their home expert's gemm2 matrix (exactly the layout a
WeightPushMulticast/getmem weight_full provides in the e2e driver; here the
slots are filled by a plain copy in UNTIMED setup — the standalone bench has
no overlap partner for weight movement).

Metadata (virtual gating args, splits_per_source, dedup counts, amortized
combine indices/CSRs) is recomputed per run but OUTSIDE the timed window,
reported as l1_index_build_ms — the l0l1 window-accounting convention.
Empty virtual slots (0-row experts) are guaranteed by the plan and exercise
the c9b82b6 empty-expert fix; the FLUX_A2AV_RS_MAX_* knobs are set exactly
from the plan via required_a2av_rs_knobs (parity-tested in
sweeps/test_knob_demands.py) BEFORE op construction.

Correctness gates:
  V1 flux-vs-torch: the fused op vs moe_gather_rs_forward_torch on the SAME
     virtual inputs (allclose, the standard bench gate).
  V2 replication transparency (small cells only, bounded by --v2_max_bytes):
     the torch VIRTUAL-space combine vs the torch ORIGINAL-space combine on
     matched per-(token, k) inputs, route weights, and home-replica weights —
     MoonEP's expert replication must be invisible in the combined output.
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed

import flux
import flux.testing
from flux.testing import (
    DTYPE_MAP,
    initialize_distributed,
    gen_moe_gating_args,
    moe_gather_rs_forward_torch,
    parse_traffic_matrix,
    traffic_matrix_to_choosed_experts,
)
from flux.testing.a2av_combine_indices import (
    build_a2av_combine_indices,
    build_a2av_compress_indices,
    build_a2av_unique_counts,
)
from flux.testing.moonep_fused_map import (
    build_fused_metadata,
    build_virtual_map,
    required_a2av_rs_knobs,
)
from flux.testing.moonep_semantics import MoonEPConfig, compute_moonep_plan
from flux.testing.recorder import RECORDER
from flux.testing.payload_probe import PayloadProbe, payload_probe_enabled
from flux.testing.traffic_matrix import (
    choosed_experts_to_matrix_chunks,
    load_routing_file,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traffic_matrix", type=str, required=True)
    parser.add_argument("--chunk_bytes", type=int, default=8192,
                        help="bytes of one routed copy (N * dtype size)")
    parser.add_argument("-N", type=int, default=4096, help="model hidden (gemm2 out)")
    parser.add_argument("-K", type=int, default=4096, help="ffn hidden (gemm2 in)")
    parser.add_argument("-G", type=int, default=32,
                        help="ORIGINAL global expert count (virtual space derived)")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--iters", default=10, type=int)
    parser.add_argument("--warmup_iters", default=5, type=int)
    parser.add_argument("--sm_margin", default=0, type=int)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--fastacc", default=False, action="store_true")
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--token_padding", type=int, default=128,
                        help="plan constant (identical to the layer0 moonep"
                        " arms so plan hashes match)")
    parser.add_argument("--comm_pattern", default="a2av_hier_compress",
                        choices=["dense", "a2av_hier", "a2av_hier_compress"],
                        help="combine transport over the VIRTUAL space;"
                        " compress degrades to hier on one node")
    parser.add_argument("--n_split", type=int, default=4)
    parser.add_argument("--timing_mode", default="isolated",
                        choices=["isolated", "amortized"],
                        help="isolated: the op builds combine indices"
                        " in-forward. amortized: the harness precomputes"
                        " everything a fused l0+l1 pipeline inherits (the"
                        " e2e driver's mode), untimed")
    parser.add_argument("--routing_file", type=str, default="")
    parser.add_argument("--skip_correctness", default=False, action="store_true")
    parser.add_argument("--v2_max_bytes", type=int, default=1 << 30,
                        help="run the V2 replication-transparency gate only"
                        " when the matched per-copy input tensor fits this"
                        " budget (it materializes [ntokens, topk, K])")
    return parser.parse_args()


ABSOLUTE_THRESHOLD_MAP = {torch.float16: 1e-2, torch.bfloat16: 2e-2}
RELATIVE_THRESHOLD_MAP = {torch.float16: 1e-2, torch.bfloat16: 2e-2}


def perf_gemm(iters, warmup_iters, name, fn):
    """Per-iteration CUDA-event timing (the layer1 bench harness verbatim)."""
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    torch.cuda.synchronize()
    torch.distributed.barrier()
    isolated = bool(int(os.getenv("FLUX_SWEEP_ISOLATED_ITERS", "0")))
    iso_sync_times = []
    output = None
    for i in range(total_iters):
        if isolated:
            t_iso = time.perf_counter()
            torch.cuda.synchronize()
            torch.distributed.barrier()
            iso_sync_times.append((time.perf_counter() - t_iso) * 1e3)
        nvtx_tag = f"iter{i}_warmup" if i < warmup_iters else f"iter{i}"
        start_events[i].record()
        with torch.cuda.nvtx.range(nvtx_tag):
            output = fn()
        end_events[i].record()
    times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            times.append(start_events[i].elapsed_time(end_events[i]))
    iter_times = {"e2e_ms": times}
    if isolated:
        iter_times["iso_sync_ms"] = iso_sync_times[warmup_iters:]
    return output, iter_times


def seg_rows_from_scatter(scatter_index, m_start, m_end):
    """(local_rows, token_idx, k_idx) of MY gemm segment, derived from the
    replicated scatter index — the inverse row map without materializing M."""
    ntokens, K = scatter_index.shape
    rows = scatter_index.flatten().long().cpu()
    mask = (rows >= m_start) & (rows < m_end)
    t_idx = (torch.arange(ntokens * K) // K)[mask]
    k_idx = (torch.arange(ntokens * K) % K)[mask]
    return (rows[mask] - m_start), t_idx, k_idx


if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    torch.use_deterministic_algorithms(False)
    RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()
    W = WORLD_SIZE
    L = W // NNODES

    args = parse_args()
    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = input_dtype
    assert args.N * input_dtype.itemsize == args.chunk_bytes
    assert args.G % W == 0
    # combine tile is 1024 or 512 (2026-08-21, K3 H=3584)
    assert args.N % args.n_split == 0 and (args.N // args.n_split) % 512 == 0

    matrix = parse_traffic_matrix(args.traffic_matrix)
    assert matrix.shape[0] == W
    if args.routing_file:
        choosed_experts = load_routing_file(args.routing_file, args.G, args.topk)
        got = choosed_experts_to_matrix_chunks(choosed_experts, W, args.G // W)
        assert torch.equal(got * args.chunk_bytes, matrix)
    else:
        if RANK == 0:
            print("WARNING: synthetic dealer routing (max-dedup); use a"
                  " --routing_file for real token-overlap semantics")
        choosed_experts = traffic_matrix_to_choosed_experts(
            matrix, args.G, args.topk, args.chunk_bytes
        )
    choosed_experts = choosed_experts.cpu().int()
    ntokens = choosed_experts.shape[0]
    assert ntokens % W == 0
    S = ntokens // W
    M = ntokens * args.topk

    # ---- MoonEP plan -> virtual space -> metadata (untimed, but reported) ----
    t0 = time.perf_counter()
    cfg = MoonEPConfig(S=S, K=args.topk, E=args.G, R=W, H=args.N,
                       token_padding=args.token_padding)
    topk_all = choosed_experts.reshape(W, S, args.topk)
    plan = compute_moonep_plan(cfg, topk_all)
    vmap = build_virtual_map(plan, topk_all)
    meta = build_fused_metadata(vmap, L)
    plan_host_ms = (time.perf_counter() - t0) * 1e3
    epn, B, gpe, E_virt = cfg.epn, cfg.B, vmap.gpe, vmap.E_virt

    # exact capacity knobs BEFORE op construction (ctor-read); explicit env wins
    for k, v in required_a2av_rs_knobs(meta, W, L).items():
        os.environ.setdefault(k, v)

    flux.init_flux_shm(TP_GROUP)

    use_compress = args.comm_pattern == "a2av_hier_compress"
    use_a2av = args.comm_pattern == "a2av_hier" or use_compress
    if use_compress and NNODES == 1 and RANK == 0:
        print("a2av_hier_compress on a single node degrades to plain a2av_hier")

    # ---- virtual gating + per-iteration combine metadata (outside window) ----
    t0 = time.perf_counter()
    gating_v = gen_moe_gating_args(
        E_virt, args.topk, ntokens,
        choosed_experts=vmap.virtual_choosed.long().cuda(),
    )
    split_cpu = gating_v.splits_gpu.to("cpu")
    assert torch.equal(split_cpu[:E_virt].int(), meta.splits), \
        "virtual gating splits != plan metadata splits"
    routing_idx = gating_v.scatter_index.flatten()
    splits_per_source_cpu = meta.splits_per_source  # [W, E_virt] int32 CPU
    unique_counts_cpu = None
    if use_compress and NNODES > 1:
        unique_counts_cpu = build_a2av_unique_counts(
            vmap.virtual_choosed.long(), W, NNODES, gpe
        )
    a2av_kwargs = {}
    if use_a2av:
        a2av_kwargs["splits_per_source"] = splits_per_source_cpu
        if unique_counts_cpu is not None:
            a2av_kwargs["a2av_unique_counts"] = unique_counts_cpu
        if args.timing_mode == "amortized":
            pack_index, reduce_index = build_a2av_combine_indices(
                routing_idx, split_cpu, RANK, W, args.topk
            )
            a2av_kwargs["a2av_pack_index"] = pack_index
            a2av_kwargs["a2av_reduce_index"] = reduce_index
            if unique_counts_cpu is not None:
                wire_ptr, wire_copy, red_ptr, red_row = build_a2av_compress_indices(
                    routing_idx, split_cpu, unique_counts_cpu, RANK, W, NNODES,
                    args.topk,
                )
                a2av_kwargs["a2av_wire_csr"] = [wire_ptr, wire_copy]
                a2av_kwargs["a2av_reduce_csr"] = [red_ptr, red_row]
    torch.cuda.synchronize()
    l1_index_build_ms = (time.perf_counter() - t0) * 1e3

    eid_start = RANK * gpe
    m_start = int(split_cpu[:eid_start].sum())
    M_cur = int(meta.m_per_rank[RANK])
    m_end = m_start + M_cur
    n_empty = int((meta.splits.view(W, gpe)[RANK] == 0).sum())

    # ---- weights: home rows + REPLICA slots (untimed setup; the e2e driver
    # fills the same layout via the weight-movement ops) ----
    g = torch.Generator().manual_seed(1234)
    full_w2 = (torch.rand((args.G, args.N, args.K), generator=g) * 0.02 - 0.01)
    full_w2 = full_w2.to(input_dtype)
    w_virtual = torch.zeros((gpe, args.N, args.K), dtype=input_dtype)
    w_virtual[:epn] = full_w2[RANK * epn:(RANK + 1) * epn]
    for b in range(B):
        e = int(plan.experts_to_copy[RANK, b])
        if e >= 0:
            w_virtual[epn + b] = full_w2[e]
    w_virtual = w_virtual.cuda()

    # ---- per-(token, k) route weights + matched inputs ----
    g = torch.Generator().manual_seed(777)
    w_tok = torch.rand((ntokens, args.topk), generator=g) + 0.5  # fp32 replicated
    local_rows_v, t_v, k_v = seg_rows_from_scatter(gating_v.scatter_index.cpu(), m_start, m_end)
    vec_scale_v = torch.zeros(max(M_cur, 1), dtype=torch.float32)
    vec_scale_v[local_rows_v] = w_tok[t_v, k_v]
    vec_scale_v = vec_scale_v[:M_cur].cuda() if M_cur > 0 else vec_scale_v.cuda()

    x_bytes = ntokens * args.topk * args.K * input_dtype.itemsize
    run_v2 = (not args.skip_correctness) and x_bytes <= args.v2_max_bytes
    if payload_probe_enabled():
        # wire-ordering audit: the combine input is re-randomized every
        # iteration, so the matched-input V2 identity (inputs == X rows) no
        # longer holds by construction -- V1 (fused vs torch on the FINAL
        # inputs, last-iteration output) is the audited gate
        run_v2 = False
    if run_v2:
        # matched per-copy activations: X[t, k] is the gemm2 input row of copy
        # (t, k) in EVERY space, so the two torch references are comparable
        g = torch.Generator().manual_seed(4242)
        X = (torch.rand((ntokens, args.topk, args.K), generator=g) * 0.2 - 0.1)
        X = X.to(input_dtype)
        inputs = torch.zeros((max(M_cur, 1), args.K), dtype=input_dtype)
        inputs[local_rows_v] = X[t_v, k_v]
        inputs = inputs[:M_cur].cuda()
    else:
        X = None
        inputs = ((torch.rand((max(M_cur, 1), args.K)) * 0.2 - 0.1)
                  .to(input_dtype)[:M_cur].cuda())
    input_scales = torch.ones((1,), dtype=torch.float32, device="cuda")
    weight_scales = torch.ones((gpe,), dtype=torch.float32, device="cuda")

    if RANK == 0:
        chunks_v = meta.splits_per_source.long().view(W, W, gpe).sum(2)
        wire = int(chunks_v.sum()) - int(chunks_v.diag().sum())
        orig = int(matrix.sum() - matrix.diag().sum()) // args.chunk_bytes
        print(f"moonep l1: E_virt {E_virt} (gpe {gpe} = epn {epn} + B {B}),"
              f" M {M}, empty slots on rank0: {n_empty}")
        print(f"combine copies crossing ranks: {wire} (dispatch matrix had"
              f" {orig}; the difference is MoonEP's replication win)")
        print(f"comm_pattern: {args.comm_pattern}, timing_mode:"
              f" {args.timing_mode}, l1_index_build {l1_index_build_ms:.1f} ms")

    # ---- the fused op over the virtual space ----
    op = flux.GemmGroupedV2GatherRSOp(
        TP_GROUP, E_virt, M, args.N, args.topk, output_dtype,
        1, W, 1,
        nnodes=NNODES, n_split=args.n_split,
        do_all_reduce=False, use_read_mode=False,
        a2av_hier=use_a2av and not use_compress,
        a2av_hier_compress=use_compress,
    )

    # wire-ordering audit (CLAUDE.md invariant 5): per-iteration payload
    # randomization of the combine input (in place; V1 recomputes the torch
    # reference from the FINAL inputs below)
    probe = PayloadProbe(inputs, RANK)
    _it = [0]

    def fn():
        probe.step(_it[0])
        _it[0] += 1
        return op.forward_gather_rs(
            inputs, w_virtual, split_cpu, routing_idx,
            input_scale=input_scales,
            weight_scale=weight_scales,
            output_vec_scale=vec_scale_v,
            fast_accum=args.fastacc,
            sm_margin=args.sm_margin,
            bias=None,
            **a2av_kwargs,
        )

    with flux.group_profile(
        name="moe_moonep_l1_" + os.environ["TORCHELASTIC_RUN_ID"],
        do_prof=args.profile, group=TP_GROUP,
    ):
        flux_output, iter_times = perf_gemm(
            args.iters, args.warmup_iters, f"moonep_l1 #{RANK}", fn
        )

    e2e = sum(iter_times["e2e_ms"]) / len(iter_times["e2e_ms"])
    print(f"rank {RANK}: moonep_l1 {e2e:.3f} ms")
    if RANK == 0:
        RECORDER.emit_info(
            ntokens=int(ntokens), topk=int(args.topk),
            comm_pattern=f"moonep_{args.comm_pattern}",
            timing_mode=args.timing_mode, n_split=int(args.n_split),
            moonep_gpe=int(gpe), moonep_E_virt=int(E_virt),
            moonep_empty_slots_rank0=int(n_empty),
            plan_host_ms=plan_host_ms,
            l1_index_build_ms=l1_index_build_ms,
        )
    RECORDER.emit_iters("flux", iter_times)
    TP_GROUP.barrier()

    if args.skip_correctness:
        RECORDER.flush()
        sys.exit(0)

    atol = ABSOLUTE_THRESHOLD_MAP[input_dtype]
    rtol = RELATIVE_THRESHOLD_MAP[input_dtype]

    # V1: fused op vs torch reference on the SAME virtual inputs
    torch_v = moe_gather_rs_forward_torch(
        TP_GROUP, M, eid_start, m_start, m_end,
        inputs, w_virtual, split_cpu,
        gating_v.gather_index, gating_v.topk_index, args.topk,
        input_scales, weight_scales, vec_scale_v,
        False, fast_acc=args.fastacc,
    )
    ok = True
    try:
        flux.torch_allclose(flux_output, torch_v, atol=atol, rtol=rtol)
    except Exception:  # noqa: BLE001
        ok = False
        RECORDER.emit_correctness(bitwise=False, allclose=False)
        RECORDER.flush()
        print(f"❌ rank {RANK}: V1 flux vs torch (virtual space) mismatch")
        if payload_probe_enabled():
            _bad = (~torch.isclose(flux_output.float(), torch_v.float(),
                                   atol=atol, rtol=rtol)).any(dim=1)
            print(f"  probe rank {RANK}: bad {int(_bad.sum())}/{flux_output.shape[0]} "
                  "rows (payload randomized per iteration)", flush=True)
    else:
        print(f"✅ rank {RANK}: V1 flux vs torch (virtual space) allclose")
    # collective verdict (wire-ordering audit hygiene): a per-rank assert
    # wedges the surviving ranks in the next collective and holds the srun
    # step; fail together instead.
    _flag = torch.tensor([int(ok)], device="cuda")
    torch.distributed.all_reduce(_flag, op=torch.distributed.ReduceOp.MIN,
                                 group=TP_GROUP)
    assert int(_flag) == 1, "correctness failed on at least one rank"

    # V2: replication transparency — virtual-space combine == original-space
    # combine on matched inputs (small cells only)
    if run_v2:
        epr = args.G // W
        gating_o = gen_moe_gating_args(
            args.G, args.topk, ntokens, choosed_experts=choosed_experts.long().cuda()
        )
        split_o = gating_o.splits_gpu.to("cpu")
        m_start_o = int(split_o[:RANK * epr].sum())
        m_cur_o = int(split_o[RANK * epr:(RANK + 1) * epr].sum())
        m_end_o = m_start_o + m_cur_o
        rows_o, t_o, k_o = seg_rows_from_scatter(
            gating_o.scatter_index.cpu(), m_start_o, m_end_o
        )
        inputs_o = torch.zeros((max(m_cur_o, 1), args.K), dtype=input_dtype)
        inputs_o[rows_o] = X[t_o, k_o]
        inputs_o = inputs_o[:m_cur_o].cuda()
        vec_o = torch.zeros(max(m_cur_o, 1), dtype=torch.float32)
        vec_o[rows_o] = w_tok[t_o, k_o]
        vec_o = vec_o[:m_cur_o].cuda()
        torch_o = moe_gather_rs_forward_torch(
            TP_GROUP, M, RANK * epr, m_start_o, m_end_o,
            inputs_o, full_w2[RANK * epr:(RANK + 1) * epr].cuda(), split_o,
            gating_o.gather_index, gating_o.topk_index, args.topk,
            input_scales, torch.ones((epr,), dtype=torch.float32, device="cuda"),
            vec_o, False, fast_acc=args.fastacc,
        )
        try:
            flux.torch_allclose(torch_v, torch_o, atol=atol, rtol=rtol)
        except Exception as e:  # noqa: BLE001
            ok = False
            RECORDER.emit_correctness(bitwise=False, allclose=False)
            RECORDER.flush()
            print(f"❌ rank {RANK}: V2 replication transparency VIOLATED")
            raise e
        print(f"✅ rank {RANK}: V2 virtual == original combine (replication"
              " transparent)")
    elif RANK == 0:
        print(f"V2 skipped (matched-input tensor {x_bytes} B >"
              f" --v2_max_bytes {args.v2_max_bytes})")

    RECORDER.emit_correctness(bitwise=False, allclose=ok)
    RECORDER.flush()
