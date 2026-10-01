#!/usr/bin/env python3
"""Merge the probe logs (one per busy mode) into one CSV and print the SM-independence table.

    python summarize.py <log dir> [--rank 0]  -> writes results.csv next to this script, prints the table
"""
import argparse, csv, glob, os, sys
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("logdir")
ap.add_argument("--rank", type=int, default=0)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv"))
a = ap.parse_args()
rows = []
for f in sorted(glob.glob(os.path.join(a.logdir, "probe_*.log"))):
    for line in open(f):
        if line.startswith("ROW,"):
            p = line.rstrip("\n").split(",")
            rows.append(dict(rank=int(p[1]), probe=p[2], mech=p[3], buf=p[4], busy=p[5], bytes=int(p[6]), n=int(p[7]),
                             metric=p[8], median_us=float(p[9]), p90_us=float(p[10]), min_us=float(p[11])))
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows -> {a.out}")
# SM-independence table: median of the duration metric per (probe, mech, buf, bytes, n), idle vs spin vs stream
dur = {"P0": {"host_memcpy": "gpu_us", "sm_copy": "gpu_us", "conc3_memcpy": "gpu_us_all3"},
       "P2": {"if_chain": "copystart_to_end_us", "switch": "copystart_to_end_us"},
       "P3": {"proxy_loop": "flag_to_done_us", "proxy_batch": "flag_to_done_us"},
       "P4": {"cdp_memcpy": "launch_to_end_us"}}
tab = defaultdict(dict)
for r in rows:
    if r["rank"] != a.rank or r["buf"] != "ipc":
        continue
    m = dur.get(r["probe"], {}).get(r["mech"])
    if m and r["metric"] == m:
        tab[(r["probe"], r["mech"], r["bytes"], r["n"])][r["busy"]] = r["median_us"]
print(f"\n{'probe':5} {'mech':13} {'bytes':>9} {'n':>3} | {'idle':>8} {'spin':>8} {'stream':>8} | spin/idle stream/idle")
for k in sorted(tab):
    v = tab[k]
    i, s, t = v.get("none"), v.get("spin"), v.get("stream")
    f = lambda x: f"{x:8.1f}" if x is not None else f"{'-':>8}"
    r1 = f"{s / i:9.2f}" if i and s else f"{'-':>9}"
    r2 = f"{t / i:10.2f}" if i and t else f"{'-':>10}"
    print(f"{k[0]:5} {k[1]:13} {k[2]:9d} {k[3]:3d} | {f(i)} {f(s)} {f(t)} | {r1} {r2}")
