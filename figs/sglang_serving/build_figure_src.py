#!/usr/bin/env python3
"""Reports of one campaign job (data/<dataset>/jfig_<dataset>_report.txt) -> figure_src_<dataset>.csv, with the
validity checks of handoff 55 (exit 1 on any failure):
  decode   every point present for both systems (or recorded INFEASIBLE), KV usage max < 0.95
  prefill  median prefill chunk >= 0.9 x SMAX tokens per GPU (the budget was actually reached)
  servers  zero tracebacks and zero capacity growths
Budget (MiB per GPU) = tokens or running requests per GPU x hidden x 2 B.

    python3 build_figure_src.py 30b4n | 235b16n | 30b16n_40g
"""
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NODES = {"30b4n": 4, "235b16n": 16, "30b16n_40g": 16, "30b16n_80g": 16}
# Decode points taken from an earlier campaign instead of this dataset's reports (user ruling 10-03: the 1 MB decode
# bar is plan 10's 1.04x headline, R8 = job 59242113, 4n 40 GB nodes, ours = lopep p10-s7 d588f725 with the knobs that
# p10-f later made default; stock = the same lopep_d1rt launch scripts, SGLang graphs off). These replace every run of
# the point in the dataset's own reports. Format: per_gpu -> (report under data/<ds>/, stock arm regex, ours arm regex,
# note for the source column).
EXTERNAL_DECODE = {
    "30b4n": {256: ("plan10_r8/j4nDr8_report.txt", r"d30_baseA\d*$", r"d30_ks7\d*$",
                    "plan 10 R8 job 59242113 4n 40 GB nodes (other bars 80 GB) lopep p10-s7 d588f725")},
}


def external_decode(d, ds):
    out = {}
    for pr, (rel, stock_re, ours_re, note) in EXTERNAL_DECODE.get(ds, {}).items():
        cur = None
        for ln in open(os.path.join(d, rel)):
            m = re.search(r" (d30_\w+): waves at (\d+)", ln)
            if m:
                cur = m.group(1)
            m = re.match(r"running/rank (\d+): .* decode step median ([0-9.]+) ms", ln)
            if m and cur and int(m.group(1)) == pr:
                for sysname, rx in (("stock", stock_re), ("ours", ours_re)):
                    if re.match(rx, cur):
                        out.setdefault((pr, sysname), []).append((float(m.group(2)), f"{rel}:{cur}"))
        out[("note", pr)] = note
    return out


def main(ds):
    d = os.path.join(HERE, "data", ds)
    import glob
    # main run + supplements in time order; a later report that re-measures a prefill point replaces the earlier
    # values and errors of that point (e.g. a 16 MB re-run with a 4096-token calibration)
    # decode likewise: a later report that measures a decode point (running requests per GPU) replaces every earlier
    # run of that point (e.g. the 10-03 1 MB re-measurement on a 1 MB-sized server, jfig_30b4n_r2d)
    files = sorted(glob.glob(os.path.join(d, f"jfig_{ds}*_report.txt")), key=os.path.getmtime)
    later, later_dec = {}, {}
    for k, f in enumerate(files):
        for ln in open(f):
            m = re.search(r"\[p30_(?:kf|baseA)_s(\d+) lcbp\] Input token", ln)
            if m:
                later[int(m.group(1))] = k
            m = re.match(r"running/rank (\d+): ", ln)
            if m:
                later_dec[int(m.group(1))] = k
    rep = []
    for k, f in enumerate(files):
        for ln in open(f).read().splitlines():
            m = re.search(r"p30_(?:kf|baseA)_s(\d+)", ln)
            if m and later.get(int(m.group(1)), k) != k:
                continue                      # superseded by a later report of the same point
            m = re.match(r"running/rank (\d+): ", ln)
            if m and later_dec.get(int(m.group(1)), k) != k:
                continue
            rep.append((os.path.basename(f), ln))
    man = open(os.path.join(d, "manifest.txt")).read()
    model = re.search(r"^MODEL=(\S+)", man, re.M).group(1)
    hidden = json.load(open(os.path.join(model, "config.json")))["hidden_size"]
    mib = lambda per_gpu: per_gpu * hidden * 2 / 2**20
    errors, dec, pre, infeasible, fallback = [], {}, {}, set(), {}
    cur = None
    for fname, ln in rep:
        m = re.search(r" (d30_[a-z0-9]+): waves at (\d+)", ln)
        if m:
            cur = m.group(1)
        m = re.match(r"running/rank (\d+): .* decode step median ([0-9.]+) ms .* KV usage max ([0-9.]+)", ln)
        if m and cur:
            pr, step, kv = int(m.group(1)), float(m.group(2)), float(m.group(3))
            sysname = "ours" if cur.startswith("d30_o") else "stock"
            dec.setdefault((pr, sysname), []).append((step, f"{fname}:{cur}"))
            if kv >= 0.95:
                errors.append(f"{cur} {pr}: KV usage max {kv}")
        m = re.search(r" (d30_[a-z0-9]+): point (\d+) FALLBACK prompts (\S+):", ln)
        if m:
            fallback[int(m.group(2))] = m.group(3)
        m = re.search(r" (d30_[a-z0-9]+): point (\d+) INFEASIBLE", ln)
        if m:
            infeasible.add(int(m.group(2)))
        m = re.match(r"\[(p30_(kf|baseA)_s(\d+)) lcbp\] Input token throughput \(tok/s\):\s+([0-9.]+)", ln)
        if m:
            pre.setdefault((int(m.group(3)), "ours" if m.group(2) == "kf" else "stock"), []).append((float(m.group(4)), f"{fname}:{m.group(1)}"))
        m = re.search(r" (p30_\w+): prefill chunk tokens per GPU median (\d+) \(SMAX (\d+)", ln)
        if m and int(m.group(2)) < 0.9 * int(m.group(3)) and "FILL FAIL" not in ln:
            errors.append(f"{m.group(1)}: prefill chunk median {m.group(2)} < 0.9 x SMAX {m.group(3)}")
        m = re.search(r" (\S+): growths (\d+) tracebacks (\d+)", ln)
        if m and (int(m.group(2)) or int(m.group(3))):
            errors.append(f"{m.group(1)}: growths {m.group(2)} tracebacks {m.group(3)}")
        if "SERVER FAILED" in ln:
            errors.append(ln.strip())
    ext = external_decode(d, ds)
    ext_notes = {k[1]: v for k, v in ext.items() if k[0] == "note"}
    for k, v in ext.items():
        if k[0] != "note":
            dec[k] = v                        # replaces this dataset's own runs of the point
    rows = []
    for panel, data in (("decode", dec), ("prefill", pre)):
        points = sorted({k[0] for k in data} | (infeasible if panel == "decode" else set()))
        for p in points:
            for sysname in ("stock", "ours"):
                runs = data.get((p, sysname), [])
                if not runs:
                    status = "infeasible" if (panel == "decode" and p in infeasible) else "missing"
                    if status == "missing":
                        errors.append(f"{panel} {p}: no {sysname} value")
                    rows.append([panel, NODES[ds], round(mib(p)), p, sysname, "", "", status, ""])
                    continue
                vals = [v for v, _ in runs]
                rows.append([panel, NODES[ds], round(mib(p)), p, sysname, f"{st.mean(vals):.2f}",
                             ";".join(f"{v:.2f}" for v in vals), "measured",
                             f"data/{ds}/: " + " ".join(n for _, n in runs)
                             + (f" (shorter prompts {fallback[p]})" if panel == "decode" and p in fallback else "")
                             + (f" [EXTERNAL: {ext_notes[p]}]" if panel == "decode" and p in ext_notes else "")])
    out = os.path.join(HERE, f"figure_src_{ds}.csv")
    with open(out, "w") as f:
        f.write("panel,nodes,budget_mib,per_gpu,system,value,runs,status,source\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"wrote {out}: {len(rows)} rows")
    for r in rows:
        print("  ", r[:8])
    if errors:
        print("VALIDITY FAILURES:")
        for e in errors:
            print("  ", e)
        sys.exit(1)
    print("validity checks PASS")


if __name__ == "__main__":
    main(sys.argv[1])
