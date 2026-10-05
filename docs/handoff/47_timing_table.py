"""47_timing_table.py <tag> [--dir logs/sglang] [--min-n 4]

Ledger of a chain_prof run: parses every `server_<tag>_<arm>_s<smax>.log` for the periodic bracket reports
  [layer timing rank r] <MODE> <PAD> n_pad<=b: mean ms per layer-step over n: total t | pre a moe b post c
  [lopep|zepp timing rank r] S<=b[+swap]: mean ms per layer-step over n: total t | phase v ...
and the fit lines, and prints per arm x smax: the stock-span table per (mode, pad, bin) and the lopep
sub-phase table per bucket, weighting every report window by its step count. Warm-up windows land in the
tiny bins and do not pollute the regime bins."""
import argparse
import glob
import os
import re
from collections import defaultdict

RE_L = re.compile(r"\[layer timing rank \d+\] (\S+) (\S+) n_pad<=(\d+): mean ms per layer-step over (\d+): total ([\d.]+) \| (.*)")
RE_P = re.compile(r"\[(?:lopep|zepp) timing rank \d+\] (S<=\d+(?:\+swap)?): mean ms per layer-step over (\d+): total ([\d.]+) \| (.*)")
RE_FIT = re.compile(r"\[(?:layer|lopep|zepp) timing rank \d+\] fit (.*)")


def parse(path):
    stock, lopep, fits = defaultdict(lambda: [0, defaultdict(float)]), defaultdict(lambda: [0, defaultdict(float)]), []
    for line in open(path, errors="replace"):
        m = RE_L.search(line)
        if m:
            mode, pad, b, n, tot, rest = m.groups()
            n = int(n)
            acc = stock[(mode, pad, int(b))]
            acc[0] += n
            parts = rest.split()
            for k, v in zip(parts[0::2], parts[1::2]):
                acc[1][k] += float(v) * n
            acc[1]["total"] += float(tot) * n
            continue
        m = RE_P.search(line)
        if m:
            cls, n, tot, rest = m.groups()
            n = int(n)
            acc = lopep[cls]
            acc[0] += n
            parts = rest.split()
            for k, v in zip(parts[0::2], parts[1::2]):
                acc[1][k] += float(v) * n
            acc[1]["total"] += float(tot) * n
            continue
        m = RE_FIT.search(line)
        if m:
            fits.append(m.group(1))
    return stock, lopep, fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--dir", default=os.path.expandvars("$PSCRATCH/workspace/andrewy/logs/sglang"))
    ap.add_argument("--min-n", type=int, default=4)
    ap.add_argument("--fits", action="store_true")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, f"server_{a.tag}_*_s*.log")))
    for f in files:
        name = os.path.basename(f)[len(f"server_{a.tag}_"):-4]
        arm, smax = name.rsplit("_s", 1)
        stock, lopep, fits = parse(f)
        print(f"\n=== {arm} SMAX {smax} ({os.path.basename(f)})")
        rows = [(k, v) for k, v in stock.items() if v[0] >= a.min_n]
        if rows:
            print(f"  {'mode':<8}{'pad':<5}{'n_pad<=':>8}{'steps':>7}{'total':>8}{'pre':>7}{'moe':>8}{'post':>7}")
            for (mode, pad, b), (n, acc) in sorted(rows, key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
                print(f"  {mode:<8}{pad:<5}{b:>8}{n:>7}{acc['total'] / n:>8.3f}{acc.get('pre', 0) / n:>7.3f}"
                      f"{acc.get('moe', 0) / n:>8.3f}{acc.get('post', 0) / n:>7.3f}")
        rows = [(k, v) for k, v in lopep.items() if v[0] >= a.min_n]
        if rows:
            phases = []
            for _, (n, acc) in rows:
                for k in acc:
                    if k != "total" and k not in phases:
                        phases.append(k)
            print(f"  {'class':<12}{'steps':>7}{'total':>8}" + "".join(f"{p[:10]:>11}" for p in phases))
            for cls, (n, acc) in sorted(rows, key=lambda kv: (int(re.search(r'\d+', kv[0]).group()), kv[0])):
                print(f"  {cls:<12}{n:>7}{acc['total'] / n:>8.3f}" + "".join(f"{acc.get(p, 0) / n:>11.3f}" for p in phases))
        if a.fits and fits:
            seen = {}
            for s in fits:
                seen[s.split(":")[0]] = s
            for s in seen.values():
                print(f"  fit {s}")


if __name__ == "__main__":
    main()
