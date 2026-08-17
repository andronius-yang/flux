# Parity + regression tests for the exact a2av knob computation
# (gen_matrix.a2av_knob_demands / sweep.exact_scale_knobs).
#
# Run:  python3 sweeps/test_knob_demands.py
#
# 1. Parity vs the torch reference (python/flux/testing/moonep_fused_map.py
#    required_a2av_knobs) on random routings — skipped when torch is absent
#    (login-node sweep runner never imports torch; this test does).
# 2. Dealer closed-form u/U vs a brute-force token-set count on the dealt
#    choosed_experts stream.
# 3. Env stability: small-budget cells must keep the legacy byte-identical
#    knob env (163840 floor / 6G) so existing capsules stay comparable.

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_matrix  # noqa: E402
import gen_trace_routing  # noqa: E402
import sweep  # noqa: E402


def random_routing(rng, W, T, G, topk):
    return [rng.sample(range(G), topk) for _ in range(W * T)]


def dealer_stream_u_U(chunks, W, L, T, G, topk):
    """Brute-force dedup counts of the sorted column-major dealer stream:
    source s deals its copy stream (tokens 0..T-1 cycling) into per-expert
    contiguous runs, experts column-major ascending — mirrors
    flux.testing.traffic_matrix_to_choosed_experts."""
    epr = G // W
    nn = W // L
    u = [[0] * W for _ in range(W)]
    U = [[0] * nn for _ in range(W)]
    for s in range(W):
        pos = 0
        rank_tokens = [set() for _ in range(W)]
        for d in range(W):
            for e in range(epr):
                copies = chunks[s][d] // epr + (1 if e < chunks[s][d] % epr else 0)
                # contiguous run of `copies` copies covers tokens pos..pos+copies-1 mod T
                for c in range(copies):
                    rank_tokens[d].add((pos + c) % T)
                pos += copies
        for d in range(W):
            u[s][d] = len(rank_tokens[d])
        for n in range(nn):
            un = set()
            for d in range(n * L, (n + 1) * L):
                un |= rank_tokens[d]
            U[s][n] = len(un)
    return u, U


def test_parity_vs_torch():
    try:
        import torch
        from flux.testing.moonep_fused_map import FusedMeta, required_a2av_knobs
    except Exception as e:  # noqa: BLE001
        print(f"SKIP parity_vs_torch (torch/flux unavailable: {e})")
        return
    rng = random.Random(7)
    W, L, T, G, topk = 16, 4, 64, 64, 4
    routing = random_routing(rng, W, T, G, topk)
    u, U = gen_trace_routing.real_dedup_stats(routing, W, L, T, G)
    chunks = gen_trace_routing.derive_matrix(routing, W, T, G, topk)
    d = gen_matrix.a2av_knob_demands(chunks, u, U, L)
    uc = torch.tensor([u[s] + U[s] for s in range(W)], dtype=torch.int32)
    m_per_rank = torch.tensor(
        [sum(chunks[s][dd] for s in range(W)) for dd in range(W)], dtype=torch.int64
    )
    meta = FusedMeta.__new__(FusedMeta)
    meta.a2av_unique_counts = uc
    meta.m_per_rank = m_per_rank
    ref = required_a2av_knobs(meta, W, L)
    assert int(ref["FLUX_A2AV_MAX_RECV_NTOKENS"]) == max(
        d["recv_copies"], d["recv_union"]
    ), (ref, d)
    assert int(ref["FLUX_A2AV_MAX_STAGE_NTOKENS"]) == max(d["stage_lb"], 1), (ref, d)
    assert int(ref["FLUX_A2AV_MAX_RELAY_NTOKENS"]) == max(d["relay_lb"], 1), (ref, d)
    print("OK parity_vs_torch")


def test_dealer_closed_form():
    rng = random.Random(11)
    W, L, T, G, topk = 8, 4, 32, 32, 4
    # random feasible matrix: row sums = T * topk
    chunks = []
    for _s in range(W):
        cuts = sorted(rng.sample(range(1, T * topk), W - 1))
        row = [b - a for a, b in zip([0] + cuts, cuts + [T * topk])]
        chunks.append(row)
    u_cf = gen_matrix.dealer_dedup_u(chunks, T)
    nn = W // L
    U_cf = [
        [min(sum(chunks[s][m * L + j] for j in range(L)), T) for m in range(nn)]
        for s in range(W)
    ]
    u_bf, U_bf = dealer_stream_u_U(chunks, W, L, T, G, topk)
    assert u_cf == u_bf, (u_cf, u_bf)
    assert U_cf == U_bf, (U_cf, U_bf)
    print("OK dealer_closed_form")


def test_small_budget_env_stable(tmp_dir="/tmp"):
    # a b2-like matrix (T=256, W=16): every demand < 163840 -> legacy env
    import tempfile

    rng = random.Random(3)
    W, L, T, topk = 16, 4, 256, 8
    chunks = []
    for _s in range(W):
        cuts = sorted(rng.sample(range(1, T * topk), W - 1))
        chunks.append([b - a for a, b in zip([0] + cuts, cuts + [T * topk])])
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"{W}\n")
        for row in chunks:
            f.write(" ".join(str(c * 8192) for c in row) + "\n")
        path = f.name
    spec = {"chunk_bytes": 8192, "topk": topk, "G": 128}
    plat = {"ranks_per_node": L}
    env, sym_g = sweep.exact_scale_knobs({"path": path}, spec, plat, "")
    legacy = sweep.scale_knobs(2, topk, 8192)
    assert env == legacy, (env, legacy)
    assert sym_g == 6, sym_g
    os.unlink(path)
    print("OK small_budget_env_stable")


def test_rs_demands_brute():
    """Layer1 (gather-rs) demand parity: gen_matrix.a2av_rs_knob_demands takes
    DISPATCH-orientation inputs; this brute force transcribes the C++ checks
    of gemm_grouped_v2_gather_rs.cc LITERALLY in WIRE orientation
    (chunk_at(s, d) == chunks[d][s]; :562-571 send, :600-615 stage,
    :687-710 conv/wire) so the two index paths are independent. U comes from
    the dealer stream brute force (dealer_stream_u_U), not the closed form,
    so a closed-form regression cannot mask an orientation bug here.

    RUNTIME COMPLEMENT (manual, needs GPUs; the analog of the layer0 8n
    validation): on a 2-node allocation run the compress bench with the
    computed knobs exactly at demand (expect green), then one knob at
    demand-1 (expect the matching collective FLUX_CHECK abort on ALL ranks,
    no hang), e.g.:
      srun -N2 --ntasks-per-node=1 --gpus-per-node=4 bash -lc 'source ./module.sh && \
        FLUX_A2AV_RS_MAX_WIRE_ROWS=<demand> ./launch.sh \
        test/python/moe_gather_rs/test_moe_gather_rs_traffic.py \
        --traffic_matrix <m.txt> --comm_pattern a2av_hier_compress -G 128 --topk 8'
      # then rerun with FLUX_A2AV_RS_MAX_WIRE_ROWS=<demand-1> -> expect
      # "a2av_hier_compress wire panel overflow" everywhere; repeat per knob.
      # Add FLUX_A2AV_RS_CHECK_IDENTITY=1 on one green run as a free
      # index-math audit."""
    rng = random.Random(23)
    W, L, T, G, topk = 16, 4, 96, 64, 4
    chunks = []
    for _s in range(W):
        cuts = sorted(rng.sample(range(1, T * topk), W - 1))
        chunks.append([b - a for a, b in zip([0] + cuts, cuts + [T * topk])])
    _u_bf, U_bf = dealer_stream_u_U(chunks, W, L, T, G, topk)
    d = gen_matrix.a2av_rs_knob_demands(chunks, U_bf, L)

    nn = W // L

    def chunk_at(s, dd):  # C++ wire orientation: owner s -> home dd
        return chunks[dd][s]

    send_bf = max(sum(chunk_at(s, dd) for dd in range(W)) for s in range(W))

    def node_chunk(s, n):
        return sum(chunk_at(s, dd) for dd in range(n * L, (n + 1) * L))

    stage_bf = conv_bf = wire_bf = 0
    for gn in range(nn):
        for gl in range(L):
            stage_bf = max(
                stage_bf,
                sum(node_chunk(ns * L + gl, gn) for ns in range(nn) if ns != gn),
            )
    for n2 in range(nn):
        for dl in range(L):
            conv_rows = wire_rows = 0
            for tn in range(nn):
                if tn == n2:
                    continue
                for ls in range(L):
                    conv_rows += chunk_at(n2 * L + ls, tn * L + dl)
                wire_rows += U_bf[tn * L + dl][n2]
            conv_bf = max(conv_bf, conv_rows)
            wire_bf = max(wire_bf, wire_rows)

    assert d["rs_send"] == send_bf, (d, send_bf)
    assert d["rs_stage"] == stage_bf, (d, stage_bf)
    assert d["rs_conv"] == conv_bf, (d, conv_bf)
    assert d["rs_wire"] == wire_bf, (d, wire_bf)
    # sanity: layer1 send bound == layer0 recv copies bound (same expression)
    u_cf = gen_matrix.dealer_dedup_u(chunks, T)
    d0 = gen_matrix.a2av_knob_demands(chunks, u_cf, U_bf, L)
    assert d["rs_send"] == d0["recv_copies"], (d, d0)
    print("OK rs_demands_brute")


def test_rs_scale_knobs_env():
    # exact_rs_scale_knobs: 8192-rounding, no legacy floor, per-pattern heap
    import tempfile

    rng = random.Random(5)
    W, L, T, topk = 16, 4, 256, 8
    chunks = []
    for _s in range(W):
        cuts = sorted(rng.sample(range(1, T * topk), W - 1))
        chunks.append([b - a for a, b in zip([0] + cuts, cuts + [T * topk])])
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"{W}\n")
        for row in chunks:
            f.write(" ".join(str(c * 8192) for c in row) + "\n")
        path = f.name
    spec = {"chunk_bytes": 8192, "topk": topk, "G": 128}
    plat = {"ranks_per_node": L}
    for pattern in ("a2av_hier", "a2av_hier_compress", "dense"):
        env, sym_g = sweep.exact_rs_scale_knobs({"path": path}, spec, plat, "", pattern)
        for k in (
            "FLUX_A2AV_RS_MAX_SEND_ROWS",
            "FLUX_A2AV_RS_MAX_STAGE_ROWS",
            "FLUX_A2AV_RS_MAX_CONV_ROWS",
            "FLUX_A2AV_RS_MAX_WIRE_ROWS",
        ):
            assert int(env[k]) % 8192 == 0 and int(env[k]) >= 8192, (k, env[k])
        assert env["NVSHMEM_SYMMETRIC_SIZE"] == f"{sym_g}G" and sym_g >= 6, env
    os.unlink(path)
    print("OK rs_scale_knobs_env")


def test_moonep_virtual_rs_demands():
    """Layer1 knob parity for the MOONEP VIRTUAL SPACE: the package-side
    transcription (flux.testing.moonep_fused_map.required_a2av_rs_knobs) must
    equal gen_matrix.a2av_rs_knob_demands on the virtual-space inputs, for
    real MoonEP plans — multi-node, single-node (all inter-node demands 0),
    and a hot-expert routing that leaves EMPTY virtual slots (the c9b82b6
    empty-expert territory the l1 integration must survive)."""
    try:
        import torch
        from flux.testing.moonep_fused_map import (
            build_fused_metadata,
            build_virtual_map,
            required_a2av_rs_knobs,
        )
        from flux.testing.moonep_semantics import MoonEPConfig, compute_moonep_plan
    except Exception as e:  # noqa: BLE001
        print(f"SKIP moonep_virtual_rs_demands (torch/flux unavailable: {e})")
        return
    rng = random.Random(19)

    def topk_all_random(R, S, G, K, hot=False):
        rows = []
        for _ in range(R * S):
            if hot and rng.random() < 0.8:
                # hot routing: 80% of tokens pick from the first K+1 experts
                # -> most virtual slots on most ranks end up with ZERO rows
                rows.append(rng.sample(range(K + 1), K))
            else:
                rows.append(rng.sample(range(G), K))
        return torch.tensor(rows, dtype=torch.int32).view(R, S, K)

    for name, (W, L, S, G, K, hot) in {
        "4n16r": (16, 4, 32, 64, 4, False),
        "2n8r": (8, 4, 24, 32, 4, False),
        "1n4r": (4, 4, 16, 16, 4, False),
        "2n8r_hot": (8, 4, 24, 32, 4, True),
    }.items():
        topk_all = topk_all_random(W, S, G, K, hot)
        cfg = MoonEPConfig(S=S, K=K, E=G, R=W, token_padding=128)
        plan = compute_moonep_plan(cfg, topk_all)
        vmap = build_virtual_map(plan, topk_all)
        meta = build_fused_metadata(vmap, L)
        if hot:
            assert bool((meta.splits == 0).any()), "hot case must have empty slots"
        got = required_a2av_rs_knobs(meta, W, L)
        gpe = meta.splits.numel() // W
        chunks_t = meta.splits_per_source.long().view(W, W, gpe).sum(2)
        chunks = [[int(chunks_t[s][o]) for o in range(W)] for s in range(W)]
        U = [
            [int(meta.a2av_unique_counts[s, W + m]) for m in range(W // L)]
            for s in range(W)
        ]
        ref = gen_matrix.a2av_rs_knob_demands(chunks, U, L)
        for knob, key in (
            ("FLUX_A2AV_RS_MAX_SEND_ROWS", "rs_send"),
            ("FLUX_A2AV_RS_MAX_STAGE_ROWS", "rs_stage"),
            ("FLUX_A2AV_RS_MAX_CONV_ROWS", "rs_conv"),
            ("FLUX_A2AV_RS_MAX_WIRE_ROWS", "rs_wire"),
        ):
            assert int(got[knob]) == max(ref[key], 1), (name, knob, got, ref)
        if W // L == 1:
            for key in ("rs_stage", "rs_conv", "rs_wire"):
                assert ref[key] == 0, (name, key, ref)
    print("OK moonep_virtual_rs_demands")


if __name__ == "__main__":
    test_dealer_closed_form()
    test_small_budget_env_stable()
    test_rs_demands_brute()
    test_rs_scale_knobs_env()
    test_parity_vs_torch()
    test_moonep_virtual_rs_demands()
    print("ALL OK")
