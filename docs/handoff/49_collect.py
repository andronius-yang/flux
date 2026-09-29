"""49_collect.py <report> [...]: figure tables of the handoff-49 chains (numbers stay in this tree).

Prefill cells (bench lines "[<name> lcbp] ...", names like p235_<arm>_s<SMAX>): input throughput, mean / median
TTFT, duration; plus the per-layer MoE latency from the arm's server log: ours = the lopep ledger total of class
S<=<SMAX> (no-swap and +swap weighted by count), stock = the SGLang decoder bracket total of the EXTEND class.
Decode cells (the "running/rank N: ..." lines that follow a "<name>: waves" block): decode step median + IQR,
output tok/s per GPU; plus the per-layer bracket total of DECODE MAX n_pad<=N from the server log.

    python 49_collect.py $L/j4nB_report.txt --logs $L --tag j4nB
"""
import argparse
import os
import re
from collections import defaultdict

BENCH = re.compile(r"^\[(\S+) lcbp\] (Successful requests|Benchmark duration \(s\)|Input token throughput \(tok/s\)|"
                   r"Mean TTFT \(ms\)|Median TTFT \(ms\)):\s+([0-9.]+)")
DEC = re.compile(r"^running/rank (\d+): (\d+) intervals over (\d+) ranks \| decode step median ([0-9.]+) ms "
                 r"\(IQR ([0-9.]+)-([0-9.]+)\) \| output tok/s per GPU median (\d+) \| cuda graph \[([^\]]*)\]")
WAVES = re.compile(r"^\S+ (\S+): waves at")
LEDGER = re.compile(r"lopep timing rank 0\] S<=(\d+)(\+swap)?: mean ms per layer-step over (\d+): total ([0-9.]+)")
BRACKET = re.compile(r"layer timing rank 0\] (DECODE|EXTEND) (MAX|SUM) n_pad<=(\d+): mean ms per layer-step over (\d+): "
                     r"total ([0-9.]+)")


def bucket(n):
    b = 8
    while b < n:
        b *= 2
    return b


def ledger(log, S):
    """ours: count-weighted mean of the last report of classes S<=bucket(S) and S<=bucket(S)+swap; swap share."""
    last = {}
    for ln in open(log, errors="replace"):
        m = LEDGER.search(ln)
        if m and int(m.group(1)) == bucket(S):
            last[bool(m.group(2))] = (int(m.group(3)), float(m.group(4)))
    n = sum(v[0] for v in last.values())
    if not n:
        return None, None
    return sum(c * t for c, t in last.values()) / n, last.get(True, (0, 0))[0] / n


def bracket(log, kind, S):
    """SGLang decoder bracket: last report of <kind> n_pad<=bucket(S) (MAX mode preferred, else SUM)."""
    last = {}
    for ln in open(log, errors="replace"):
        m = BRACKET.search(ln)
        if m and m.group(1) == kind and int(m.group(3)) == bucket(S):
            last[m.group(2)] = float(m.group(5))
    return last.get("MAX", last.get("SUM"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--logs", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    pre = defaultdict(dict)
    dec = defaultdict(dict)
    for rep in a.reports:
        cur = None
        for ln in open(rep, errors="replace"):
            m = BENCH.search(ln)
            if m:
                pre[m.group(1)][m.group(2).split(" (")[0]] = float(m.group(3))
                continue
            m = WAVES.search(ln)
            if m:
                cur = m.group(1)
                continue
            m = DEC.search(ln)
            if m and cur:
                dec[cur][int(m.group(1))] = dict(step=float(m.group(4)), iqr=(float(m.group(5)), float(m.group(6))),
                                                 tps=int(m.group(7)), graph=m.group(8), n=int(m.group(2)))
    if pre:
        print("PREFILL  name                     SMAX  req  in_tok/s   meanTTFT  medTTFT  layer_ms  swap%")
        for name in sorted(pre, key=lambda x: (int(re.search(r"_s(\d+)", x).group(1)) if re.search(r"_s(\d+)", x) else 0, x)):
            v = pre[name]
            sm = re.search(r"_s(\d+)$", name)
            S = int(sm.group(1)) if sm else 0
            log = os.path.join(a.logs, f"server_{a.tag}_{name}.log")
            lay, sw = (ledger(log, S) if "ours" in name else (bracket(log, "EXTEND", S), None)) if os.path.exists(log) else (None, None)
            print(f"         {name:24s} {S:5d} {int(v.get('Successful requests', 0)):4d} {v.get('Input token throughput', 0):9.1f} "
                  f"{v.get('Mean TTFT', 0):9.1f} {v.get('Median TTFT', 0):8.1f}  {lay if lay is None else round(lay, 3)!s:8s} "
                  f"{'' if sw is None else f'{100 * sw:.1f}'}")
    if dec:
        print("DECODE   name        run/rank  step_ms (IQR)            tok/s/GPU  graph  layer_ms  swap%")
        for name in sorted(dec):
            log = os.path.join(a.logs, f"server_{a.tag}_{name}.log")
            for S in sorted(dec[name]):
                d = dec[name][S]
                lay = bracket(log, "DECODE", S) if os.path.exists(log) else None
                sw = ledger(log, S)[1] if ("ours" in name and os.path.exists(log)) else None
                print(f"         {name:12s} {S:8d}  {d['step']:8.2f} ({d['iqr'][0]:.1f}-{d['iqr'][1]:.1f})  {d['tps']:9d}  "
                      f"{d['graph']:5s}  {lay!s:8s}  {'' if sw is None else f'{100 * sw:.1f}'}")


if __name__ == "__main__":
    main()
