"""K3-at-scale per-rank memory budget assertions (allocation-free).

Promoted from the 2026-08-21 verification (session 8.21.place): asserts the
variable-hc-comm (loccap_sl kernel arm) per-rank memory model fits the A100
(40 GB) and the platform symmetric-heap cap (16 GB) at K3 canon for
8n/16n/32n across the K3 budget rungs. The model mirrors the code's
allocation paths: slot weights (eplb_semantics ctor), runner token buffers
(ultraep ctor + reserve_recv_capacity at the 1.25x cap+forced bound), hc
panels (structural (NN-1)*S union bounds — routing-independent), A2AS
per-pair panels (gamma=2 clamped), and the ctx WITHOUT the resident
input_full copy (alloc_input_full=False, 2026-08-21 harness fix).

Run: python3 sweeps/test_k3_scale_budget.py
"""

GB = 1 << 30
G, K, H, FFN, L = 896, 16, 3584, 3072, 4
chunk = H * 2
RED = 2
HBM_CAP = 40 * GB
SYM_CAP = 16 * GB
RECV_FACTOR = 1.25   # eps 0.0625 cap + forced slack (loccap_sl_bounds class)
A2AS_GAMMA = 2       # per-pair clamp multiple of the balanced mean


def per_rank_bytes(NN, S):
    R = NN * L
    nlp = G // R + RED
    E1 = S * K
    nrv = int(RECV_FACTOR * E1)
    weights = nlp * 2 * FFN * H * 2 + (G // R) * FFN * H * 2
    fixed = 2 * GB                       # CUDA/NCCL/NVSHMEM contexts (est.)
    ctx = S * H * 2 + E1 * FFN * 2       # inputs_shard + ctx outputs
    bufs = (E1 * H * 2                   # send_buf
            + 2 * nrv * chunk            # recv + hidden
            + nrv * FFN * 2)             # out_buf
    meta = 2 * R * S * (K + 2) * 4
    sym = (E1 * chunk                    # send panel
           + nrv * chunk                 # recv panel
           + 2 * (NN - 1) * S * chunk    # stage + relay (structural)
           + 2 * R * (A2AS_GAMMA * E1 // R) * chunk  # A2AS pair panels
           + 1 * GB)                     # slack
    return weights + fixed + ctx + bufs + meta + sym, sym


def main():
    print(f"{'cfg':16}{'total GB':>10}{'sym GB':>8}   (caps 40 / 16)")
    worst = (0, "")
    for NN in (8, 16, 32):
        for b in (7, 28, 56):
            S = (b // 7) * 1024
            tot, sym = per_rank_bytes(NN, S)
            tag = f"{NN}n b{b}"
            print(f"{tag:16}{tot / GB:>10.2f}{sym / GB:>8.2f}")
            assert tot <= HBM_CAP, (tag, "exceeds A100 HBM", tot / GB)
            assert sym <= SYM_CAP, (tag, "exceeds sym cap", sym / GB)
            if tot > worst[0]:
                worst = (tot, tag)
    print(f"OK: all K3 scale/budget cells fit "
          f"(worst {worst[1]} = {worst[0] / GB:.1f} GB of 40)")


if __name__ == "__main__":
    main()
