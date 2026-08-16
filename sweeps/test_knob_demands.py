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


if __name__ == "__main__":
    test_dealer_closed_form()
    test_small_budget_env_stable()
    test_parity_vs_torch()
    print("ALL OK")
