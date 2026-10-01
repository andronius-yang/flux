"""49_plan6_csv.py: plan 6 (handoff 49 round 9) results -> CSV, straight from the chain reports and server logs.

Plan 6 = Qwen3-30B-A3B in SGLang v0.5.3, 40 GB A100 nodes, lopep 068e5e4, 4n / 8n / 16n x prefill / decode x
1 / 2 / 4 MiB per rank (256 / 512 / 1024 tokens per rank). Reports: logs/sglang/j<N>nD6_report.txt (decode, jobD6.sh)
and j<N>nP6_report.txt (prefill, jobP6.sh). Per-layer times are the SGLang decoder bracket (MoE block incl. its
communication, rank 0, CUDA events), count-weighted over every report window of the class (decode: DECODE MAX
n_pad<=S; prefill: EXTEND SUM n_pad<=S); ours ran with LOPEP_TIMING=0, so its bracket is the same instrument as
stock's; the stock graphs-on arm runs with the bracket off (empty column).

Writes <out>_arms.csv (one row per arm and point) and <out>_cells.csv (one row per cell: ratios inside one allocation;
> 1 = ours faster). Usage:
    python 49_plan6_csv.py --logs $PSCRATCH/workspace/andrewy/logs/sglang --out 49_plan6
"""
import argparse
import csv
import os
import re
from collections import defaultdict

HDR = re.compile(r"^\S+ job(D\d|P\d) (\d+): (\d+) nodes")
WAVES = re.compile(r"^\S+ (d30_\w+): waves at")
DEC = re.compile(r"^running/rank (\d+): (\d+) intervals over (\d+) ranks \| decode step median ([0-9.]+) ms "
                 r"\(IQR ([0-9.]+)-([0-9.]+)\) \| output tok/s per GPU median (\d+) \| cuda graph \[([^\]]*)\]")
BENCH = re.compile(r"^\[(p30_\w+) lcbp\] (Successful requests|Benchmark duration \(s\)|Input token throughput \(tok/s\)|"
                   r"Mean TTFT \(ms\)|Median TTFT \(ms\)):\s+([0-9.]+)")
CTRL = re.compile(r"^\S+ ((?:d30|p30)_\w+): CONTROL per-rank \[max_total_num_tokens=(\d+)")
GROW = re.compile(r"^\S+ ((?:d30|p30)_\w+): growths (\d+) tracebacks (\d+)")
LAUNCH = re.compile(r"^\S+ ((?:d30|p30)_\w+): conn=(\S+) heap=(\S+)")
BRK = re.compile(r"layer timing rank 0\] (DECODE MAX|EXTEND SUM) n_pad<=(\d+): mean ms per layer-step over (\d+): "
                 r"total ([0-9.]+)")
ARM = {"ours": ("ours", 1), "oursR": ("ours", 2), "oursB": ("ours_plan6_binary", 1), "baseA": ("stock_graphs_off", 1), "baseG": ("stock_graphs_on", 1),
       # round 8 (plan 7 knob A/B, development tree): k<label>[R]
       "kb3": ("ours_bar3", 1), "kb3R": ("ours_bar3", 2), "kb1": ("ours_bar1", 1), "kb1R": ("ours_bar1", 2),
       "kp1": ("ours_proxy_bar3", 1), "kp1b1": ("ours_proxy_bar1", 1),
       "kb1dm0": ("ours_bar1_hosttables", 1), "kb1dm0R": ("ours_bar1_hosttables", 2),
       "kb1dm1": ("ours_bar1_devtables", 1), "kb1dm1R": ("ours_bar1_devtables", 2),
       # round 10 (plan 8 stage-B check): r9 = round-9 binary (lopep_d1rt), b8 = stage-B integration (lopep_int)
       "kr9": ("ours_round9", 1), "kr9R": ("ours_round9", 2), "kb8": ("ours_stageB", 1), "kb8R": ("ours_stageB", 2),
       # round 11 (plan 8 stage check after C3 + C4): b8f = stage B + NH-1 fix (lopep_int), c4 = C3 + C4 (lopep_c4,
       # deferred verdict + plan-driven wire + proxy, no per-layer planning sync)
       "kb8f": ("ours_stageB_fix", 1), "kb8fR": ("ours_stageB_fix", 2), "kc4": ("ours_C3C4", 1), "kc4R": ("ours_C3C4", 2)}
MIB = {256: 1, 512: 2, 1024: 4}


def bracket(log, cls, S):
    n = t = 0
    if not os.path.exists(log):
        return None, 0
    for ln in open(log, errors="replace"):
        if "fit" in ln:
            continue
        m = BRK.search(ln)
        if m and m.group(1) == cls and int(m.group(2)) == S:
            c = int(m.group(3))
            n += c
            t += c * float(m.group(4))
    return (round(t / n, 3) if n else None), n


def read(logs, N, kind):
    tag = f"j{N}n{kind}"
    rep = os.path.join(logs, f"{tag}_report.txt")
    rows, job, cur = [], None, None
    ctrl, grow, launch = {}, {}, {}
    bench = defaultdict(dict)
    dec = []
    if not os.path.exists(rep):
        return rows
    for ln in open(rep, errors="replace"):
        m = HDR.search(ln)
        if m:
            job = m.group(2)
            continue
        for rx, store in ((CTRL, ctrl), (GROW, grow), (LAUNCH, launch)):
            m = rx.search(ln)
            if m:
                store[m.group(1)] = m.groups()[1:]
        m = WAVES.search(ln)
        if m:
            cur = (m.group(1), job)
            continue
        m = DEC.search(ln)
        if m and cur:
            dec.append((cur[0], cur[1], int(m.group(1)), m))
            continue
        m = BENCH.search(ln)
        if m:
            bench[(m.group(1), job)][m.group(2).split(" (")[0]] = float(m.group(3))
    for name, jb, S, m in dec:
        arm, run = ARM[name.split("_", 1)[1]]
        lay, nl = bracket(os.path.join(logs, f"server_{tag}_{name}.log"), "DECODE MAX", S)
        rows.append(dict(nodes=N, phase="decode", budget_mib=MIB[S], tokens_per_rank=S, arm=arm, run=run, alloc_job=jb,
                         server=name, decode_step_ms=float(m.group(4)), step_iqr_lo=float(m.group(5)),
                         step_iqr_hi=float(m.group(6)), out_tok_s_per_gpu=int(m.group(7)), cuda_graph_used=m.group(8),
                         intervals=int(m.group(2)), layer_ms=lay, layer_steps=nl,
                         kv_pin_tokens=int(ctrl[name][0]) if name in ctrl else None,
                         heap=launch.get(name, ("", ""))[1], growths=grow.get(name, ("", ""))[0],
                         tracebacks=grow.get(name, ("", ""))[1]))
    for (name, jb), v in bench.items():
        S = int(re.search(r"_s(\d+)$", name).group(1))
        arm, run = ARM[name.split("_")[1]]
        lay, nl = bracket(os.path.join(logs, f"server_{tag}_{name}.log"), "EXTEND SUM", S)
        rows.append(dict(nodes=N, phase="prefill", budget_mib=MIB[S], tokens_per_rank=S, arm=arm, run=run, alloc_job=jb,
                         server=name, in_tok_s=v.get("Input token throughput"), mean_ttft_ms=v.get("Mean TTFT"),
                         median_ttft_ms=v.get("Median TTFT"), requests=int(v.get("Successful requests", 0)),
                         bench_s=v.get("Benchmark duration"), layer_ms=lay, layer_steps=nl,
                         kv_pin_tokens=int(ctrl[name][0]) if name in ctrl else None,
                         heap=launch.get(name, ("", ""))[1], growths=grow.get(name, ("", ""))[0],
                         tracebacks=grow.get(name, ("", ""))[1]))
    return rows


def cells(rows):
    out = []
    by = defaultdict(list)
    for r in rows:
        by[(r["nodes"], r["phase"], r["budget_mib"])].append(r)
    for (N, ph, b), rs in sorted(by.items()):
        ours = sorted([r for r in rs if r["arm"] == "ours"], key=lambda r: r["run"])
        for o in ours:
            same = [r for r in rs if r["arm"] != "ours" and r["alloc_job"] == o["alloc_job"]]
            for s in same:
                c = dict(nodes=N, phase=ph, budget_mib=b, tokens_per_rank=o["tokens_per_rank"], ours_run=o["run"],
                         stock_arm=s["arm"], alloc_job=o["alloc_job"])
                if ph == "decode":
                    c.update(ours_step_ms=o["decode_step_ms"], stock_step_ms=s["decode_step_ms"],
                             speedup_step=round(s["decode_step_ms"] / o["decode_step_ms"], 3))
                else:
                    c.update(ours_in_tok_s=o["in_tok_s"], stock_in_tok_s=s["in_tok_s"],
                             speedup_throughput=round(o["in_tok_s"] / s["in_tok_s"], 3),
                             ours_mean_ttft_ms=o["mean_ttft_ms"], stock_mean_ttft_ms=s["mean_ttft_ms"],
                             speedup_ttft=round(s["mean_ttft_ms"] / o["mean_ttft_ms"], 3))
                if o["layer_ms"] and s["layer_ms"]:
                    c.update(ours_layer_ms=o["layer_ms"], stock_layer_ms=s["layer_ms"],
                             speedup_layer=round(s["layer_ms"] / o["layer_ms"], 3))
                out.append(c)
    return out


def write(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", default="49_plan6")
    ap.add_argument("--lopep", default="068e5e4", help="lopep commit of the measured binary")
    ap.add_argument("--round", default="6", help="report suffix: 6 = plan 6 (jobD6/jobP6), 7 = plan-7 stage checks (jobD7/jobP7)")
    a = ap.parse_args()
    rows = [r for N in (4, 8, 16) for kind in (f"D{a.round}", f"P{a.round}") for r in read(a.logs, N, kind)]
    rows.sort(key=lambda r: (r["phase"], r["nodes"], r["budget_mib"], r["arm"], r["run"]))
    for r in rows:
        r.update(model="Qwen3-30B-A3B", gpu="A100-40GB", lopep=a.lopep)
    write(f"{a.out}_arms.csv", rows)
    cl = cells(rows)
    write(f"{a.out}_cells.csv", cl)
    print(f"{len(rows)} arm rows -> {a.out}_arms.csv, {len(cl)} cell rows -> {a.out}_cells.csv")


if __name__ == "__main__":
    main()
