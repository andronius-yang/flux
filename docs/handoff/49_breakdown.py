"""49_breakdown.py <server log> <num layers> [--step-ms bin=ms ...]: where a step's time goes (handoff 49).

Reads the SGLang bracket of a server run with SGLANG_LAYER_TIMING_ATTN=1 (rank 0, CUDA events): per layer-step
"attn" (input norm + attention), "pre" (stock: DP gather before the MoE), "moe" (the MoE block), "post" (stock: DP
scatter), and per step "fwd" (the whole forward: every layer + embedding + final norm + LM head). Every report window
is weighted by its count. Per tokens-per-rank bin it prints the per-layer phases, the per-step totals over all layers,
the rest of the forward, and, when the scheduler's step time is given (decode), the time outside the forward
(sampling, scheduling, host gaps).

    python 49_breakdown.py server_jA_a30_ours.log 48 --step-ms 256=186.65 1024=264.43 4096=729.57
"""
import argparse
import re

LINE = re.compile(r"layer timing rank 0\] (FWD:)?(\w+) (\S+) n_pad<=(\d+): mean ms per layer-step over (\d+): "
                  r"total [0-9.]+ \| (.*)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("layers", type=int)
    ap.add_argument("--step-ms", nargs="*", default=[])
    a = ap.parse_args()
    step = {int(k): float(v) for k, v in (s.split("=") for s in a.step_ms)}
    lay, fwd = {}, {}
    for ln in open(a.log, errors="replace"):
        m = LINE.search(ln)
        if not m:
            continue
        is_fwd, mode, _pad, b, n = m.group(1), m.group(2), m.group(3), int(m.group(4)), int(m.group(5))
        parts = m.group(6).split()
        ph = {parts[i]: float(parts[i + 1]) for i in range(0, len(parts) - 1, 2)}
        d = (fwd if is_fwd else lay).setdefault((mode, b), {"n": 0})
        d["n"] += n
        for k, v in ph.items():
            d[k] = d.get(k, 0.0) + v * n
    for (mode, b) in sorted(lay):
        d = lay[(mode, b)]
        n = d["n"]
        if n < 2 * a.layers:
            continue
        ph = {k: d.get(k, 0.0) / n for k in ("attn", "pre", "moe", "post")}
        moe_block = ph["pre"] + ph["moe"] + ph["post"]
        per_layer = ph["attn"] + moe_block
        f = fwd.get((mode, b))
        fwd_ms = f["fwd"] / f["n"] if f and f["n"] else None
        L = a.layers
        line = (f"{mode:7s} <={b:5d} ({n // L} steps) per layer: attn {ph['attn']:.3f} | MoE block {moe_block:.3f} "
                f"(gather {ph['pre']:.3f} moe {ph['moe']:.3f} scatter {ph['post']:.3f}) ms || per step: attn "
                f"{L * ph['attn']:.1f}, MoE block {L * moe_block:.1f}")
        if fwd_ms is not None:
            line += f", rest of forward {fwd_ms - L * per_layer:.1f}, forward {fwd_ms:.1f} ms"
            if b in step:
                line += f", outside forward {step[b] - fwd_ms:.1f}, step {step[b]:.1f} ms"
                tot = step[b]
            else:
                tot = fwd_ms
            line += (f" | shares of {'step' if b in step else 'forward'}: attn {100 * L * ph['attn'] / tot:.0f} %, "
                     f"MoE block {100 * L * moe_block / tot:.0f} %")
        print(line)


if __name__ == "__main__":
    main()
