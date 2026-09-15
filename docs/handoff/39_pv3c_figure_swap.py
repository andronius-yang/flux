"""Swap the pv3c (paper-constraint router) values into the plotted figure
sources — main_perf, weak_scaling, ablation (LOO/matched) — from the backups
under figs/<fig>/pre_pv3c_20260915/ plus the aggregator's outputs. Rows that
are not plotted keep their pre-pv3c values with an explicit flag.
Run from the worktree root after 39_pv3c_recapture_aggregate.py."""
import csv, os, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
B = "pre_pv3c_20260915"
SHA = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"]).decode().strip()
chosen = list(csv.DictReader(open(os.path.join(HERE, "39_pv3c_chosen_dataset.csv"))))
src_pv3c = list(csv.DictReader(open(os.path.join(ROOT, "figs", "main_perf", "figure_src_pv3c.csv"))))
ws_pv3c = list(csv.DictReader(open(os.path.join(ROOT, "figs", "weak_scaling", "figure_src_pv3c.csv"))))

def prov(rows, nodes, C, kernel, match):
    """arm_variant / cell_id pattern of the pv3c rows behind a chosen cell."""
    r = [x for x in rows if x["nodes"] == nodes and x["C"] == C and x["kernel"] == kernel and match(x)]
    return (r[0]["arm_variant"], r[0]["cell_id"]) if r else ("", "")

# ---- main_perf ----
mp_old = list(csv.DictReader(open(os.path.join(ROOT, "figs", "main_perf", B, "figure_src.csv"))))
ch_mp = {(r["nodes"], r["key"]): r for r in chosen if r["figure"] == "main_perf"}
ROUTED = ("ours12", "ours12_dispatch", "ours2_nooverlap", "ours2_direct")
out, swapped = [], 0
for r in mp_old:
    r = dict(r)
    mk = "Qwen" if r["model"].startswith("Qwen") else "K2"
    key = (r["nodes"], f"{mk}/{r['row_id']}/{r['budget_mib']}")
    if r["row_id"] in ROUTED:
        if key in ch_mp and int(r["budget_mib"]) in (1, 4, 16) and r["nodes"] in ("4", "8", "16"):
            c = ch_mp[key]
            var, cid = prov(src_pv3c, c["nodes"], c["C"], c["kernel"],
                            lambda x: x["model"] == mk and x["row_id"] == r["row_id"] and x["budget_mib"] == r["budget_mib"])
            r.update(total_ms=c["pv3c_ms"], recorded_ms=c["pv3c_ms"], arm_variant=var, capsule=c["capsules"].replace(" ", ";"),
                     cell_id=cid, branch="pv3", git_sha=SHA,
                     flag=f"pv3c C={c['C']} kernel={c['kernel']} n_capsules={c['n_capsules']} d_vs_loccap_twin={c['d_vs_twin_pct']}%"
                          + (f" ({c['note']})" if c["note"] else ""))
            swapped += 1
        else:
            r["flag"] = (r["flag"] + "; " if r["flag"] else "") + "pre_pv3c LocCap router, budget/topology not plotted, NOT re-measured"
    out.append(r)
with open(os.path.join(ROOT, "figs", "main_perf", "figure_src.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(mp_old[0].keys())); w.writeheader(); w.writerows(out)
print(f"main_perf: {swapped} plotted Ours cells swapped (of {sum(1 for r in mp_old if r['row_id'] in ROUTED)} routed rows)")

# ---- weak_scaling ----
ws_old = list(csv.DictReader(open(os.path.join(ROOT, "figs", "weak_scaling", B, "figure_src.csv"))))
ch_ws = {(r["nodes"], r["key"]): r for r in chosen if r["figure"] == "weak_scaling"}
by = {}
for r in ws_old:
    r = dict(r)
    key = (r["nodes"], f"{r['system']}/{r['budget_mib']}")
    if r["system"] in ("ours", "dwire") and key in ch_ws:
        c = ch_ws[key]
        var, cid = prov(ws_pv3c, c["nodes"], c["C"], c["kernel"],
                        lambda x: x["system"] == r["system"] and x["budget_mib"] == r["budget_mib"])
        tot = float(c["pv3c_ms"])
        r.update(total_ms=f"{tot:.3f}", throughput_mtok_s=f"{int(r['total_tokens']) / tot / 1000:.4f}", status="ok",
                 capsule=c["capsules"].replace(" ", ";"), cell_id=cid, source_dataset="pv3c (handoff 39)",
                 note=f"pv3c C={c['C']} kernel={c['kernel']} n_capsules={c['n_capsules']} d_vs_loccap_twin={c['d_vs_twin_pct']}%" + (f" ({c['note']})" if c["note"] else ""))
    by.setdefault((r["nodes"], r["budget_mib"]), {})[r["system"]] = r
rows = []
for (n, b), d in by.items():
    best = min((float(d[s]["total_ms"]) for s in ("ours", "dwire") if s in d and d[s]["total_ms"]), default=None)
    for s, r in d.items():
        if s == "ours" and best is not None:
            for ref in ("comet", "nvshmem"):
                r[f"speedup_vs_{ref}"] = f"{float(d[ref]['total_ms']) / best:.4f}" if ref in d and d[ref]["total_ms"] else ""
        rows.append(r)
rows.sort(key=lambda r: (int(r["nodes"]), int(r["budget_mib"]), r["system"]))
with open(os.path.join(ROOT, "figs", "weak_scaling", "figure_src.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(ws_old[0].keys())); w.writeheader(); w.writerows(rows)
print(f"weak_scaling: {sum(1 for r in rows if r['source_dataset'].startswith('pv3c'))} Ours/dwire cells swapped, speedups recomputed on best-of(ours, dwire)")

# ---- ablation (LOO/matched): the three OURS arms take their _pv3c_14 twins ----
ab_old = list(csv.DictReader(open(os.path.join(ROOT, "figs", "ablation", B, "ablation_tables.csv"))))
ab_new = {(r["study"], r["arm"], r["metric"]): r for r in csv.DictReader(open(os.path.join(ROOT, "figs", "ablation", "ablation_tables_pv3c.csv")))}
out = []
for r in ab_old:
    r = dict(r)
    k = (r["study"], r["arm"] + "_pv3c_14", r["metric"])
    if k in ab_new:
        r.update(total_ms=ab_new[k]["total_ms"], sd=ab_new[k]["sd"], n_reps=ab_new[k]["n_reps"])
    out.append(r)
with open(os.path.join(ROOT, "figs", "ablation", "ablation_tables.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(ab_old[0].keys())); w.writeheader(); w.writerows(out)
it_old = list(csv.DictReader(open(os.path.join(ROOT, "figs", "ablation", B, "ablation_iter_tidy.csv"))))
it_new = list(csv.DictReader(open(os.path.join(ROOT, "figs", "ablation", "ablation_iter_tidy_pv3c.csv"))))
keep = [r for r in it_old if r["arm"] not in ("placement_swap_seq", "full_stack_seq", "full_stack_ovl")]
for r in it_new:
    if r["arm"].endswith("_pv3c_14"):
        r = dict(r); r["arm"] = r["arm"][:-len("_pv3c_14")]; keep.append(r)
with open(os.path.join(ROOT, "figs", "ablation", "ablation_iter_tidy.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(it_old[0].keys())); w.writeheader(); w.writerows(keep)
print("ablation: 3 OURS arms x 2 studies swapped in ablation_tables.csv / ablation_iter_tidy.csv (baseline rows untouched)")
