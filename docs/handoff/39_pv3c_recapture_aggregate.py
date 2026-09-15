"""pv3c recapture aggregator (2026-09-15, branch pv3, handoff 39).

Scans every capsule of the campaign (sweeps/results/runs/<prefix>*),
recomputes the campaign statistic from metrics.csv (per-iteration MAX
across ranks of total_ms, MEDIAN over timed iterations) and writes the
"pv3c fixed-constraint" datasets NEXT TO (never over) the plotted ones:

  figs/main_perf/figure_src_pv3c.csv      one row per (nodes, model, Ours row, budget, C)
  figs/weak_scaling/figure_src_pv3c.csv   one row per (nodes, system, budget, C)
  figs/ablation/ablation_iter_tidy_pv3c.csv + ablation_tables_pv3c.csv
  figs/ablation_cycling/results_tidy_pv3c.csv
  figs/case_study/pv3c_capsules.csv
  docs/handoff/39_pv3c_recapture_verdict.md   on-par tables + chosen C

Usage: python docs/handoff/39_pv3c_recapture_aggregate.py [--prefix 20260915]
"""
import csv
import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNS = os.path.join(ROOT, "sweeps", "results", "runs")
PREFIX = "20260915"
for i, a in enumerate(sys.argv):
    if a == "--prefix":
        PREFIX = sys.argv[i + 1]
# superseded capsules (handoff 38 §6.17: llc cells run with LocCap sizing/reference
# through the port slip) and pre-campaign junk — never aggregated
EXCLUDE = {"20260915-063112_perlmutter_d5152570", "20260915-063938_perlmutter_b50dd03a",
           # route-graph A/Bs (handoff 39 §11): their default pv3c arms ran with the
           # route graph ON (default at the time) — diagnostics, not dataset rows
           "20260915-151806_perlmutter_61a436d4", "20260915-161623_perlmutter_37cfc8f0"}
# single cells superseded by a same-campaign repeat (8n K2 llc b1 C=1/4 read +24.6%
# with equal lane brackets — stall class; repeat capsule 20260915-085958 = -0.2%)
# capsules from this run id on were measured on kernel v4 (handoff 39 §11):
# the chosen dataset prefers v4 rows for a cell when they exist; the C
# choice rule stays on the campaign rows (v4 only re-measured the chosen C).
V4_SINCE = "20260915-1611"
EXCLUDE_CELLS = {("20260915-084021_perlmutter_af0e914c",
                  "llc_l01_s1_pv2_pv3c_eps025_trace-610042_b1_k8_isolated"),
                 # 16n Qwen b16: the known intermittent ~350 ms stall (l1 340 ms
                 # with equal lane brackets); both arms repeated in the b16 ladder.
                 ("20260915-094347_perlmutter_28d64661",
                  "ours_l01_s1_pv2_r2_pv3c_eps05_trace-a2e8ab_b16_k8_isolated"),
                 ("20260915-094347_perlmutter_28d64661",
                  "ours_l01_s2_swap_force_p2p_r2_trace-a2e8ab_b16_k8_isolated")}

ROW_OF = {"ours_l01_s1_pv2_r2": "ours12",
          "ours_l01_s2_swap_force_p2p_r2": "ours12_dispatch",
          "llc_l01_s1_pv2": "ours2_nooverlap",
          "ours_l01_s1_pv2_r2_dwire": "ours2_direct"}
WEAK_OF = {"ours_l01_s1_pv2_r2": "ours", "ours_l01_s1_pv2_r2_dwire": "dwire"}
ABL_OF = {"ablation_l01_pr_swapall_pw_noov_p2p_r2": "placement_swap_seq",
          "ablation_l01_s2_swapall_pw_noov_p2p_r2": "full_stack_seq",
          "ablation_l01_s2_swapall_pw_p2p_r2": "full_stack_ovl"}
CYC_ARMS = {"ablation_l01_pr0_pv2_r2", "ours_l01_s1_pv2_r2", "ablation_l01_s2_swapall_p2p_r2",
            "ablation_l01_s2_swapall_noov_p2p_r2", "ablation_l01_s2_swapall_rp4_p2p_r2",
            "ablation_l01_s2_swapall_rp4_noov_p2p_r2", "ablation_l01_s2_swap_t1_rst_p2p_r2",
            "ablation_l01_s2_swapall_nr_p2p_r2", "ablation_l01_s2_swapall_nr_noov_p2p_r2",
            "ours_l01_s2_swap_p2p_t1_r2", "ablation_l01_s2_swap_t1_noov_p2p_r2",
            "ablation_l01_s2_swap_t1_rp4_p2p_r2", "ablation_l01_s2_swap_t1_rp4_noov_p2p_r2"}
CS_ARMS = {"ablation_l01_s2_swapall_rst_3d_early_str4_p2p_r2", "ablation_l01_s2_swapall_rst_3d_noov_str4_p2p_r2",
           "ablation_l01_s2_swapall_rst_3d_dual3_str4_p2p_r2"}


def split_variant(v):
    """'<base>_pv3c_eps025' -> (base, '1/4'); '<base>' -> (base, 'loccap').
    Route-graph A/B twins (`_rg` / `_nrg`, handoff 39 §11) are diagnostics,
    never dataset rows: they map to a base of None and are dropped."""
    if v.endswith("_rg") or v.endswith("_nrg"):
        return None, "diag"
    for suf, c in (("_pv3c_eps025", "1/4"), ("_pv3c_eps05", "1/2"), ("_pv3c_eps1", "1"), ("_pv3c", "1/16"),
                   ("_pv3_eps025", "pv3 1/4"), ("_pv3", "pv3 1/16")):
        if v.endswith(suf):
            return v[:-len(suf)], c
    return v, "loccap"


def load_capsule(d):
    cells = list(csv.DictReader(open(os.path.join(d, "cells.csv"))))
    per = defaultdict(lambda: defaultdict(dict))         # cell -> metric -> iter -> {rank: v}
    for r in csv.DictReader(open(os.path.join(d, "metrics.csv"))):
        per[r["cell_id"]][r["metric"]].setdefault(int(r["iter"]), {})[int(r["rank"])] = float(r["value_ms"])
    out = []
    for c in cells:
        cid = c["cell_id"]
        base, C = split_variant(c["variant"])
        fp = json.loads(c["family_params"]) if c.get("family_params") else {}
        series = {}
        stat = {}
        for m in ("total_ms", "plan_ms", "plan_comm_ms", "l0_ms", "l1_ms", "e2e_ms"):
            its = per[cid].get(m, {})
            if its:
                mx = [max(v.values()) for k, v in sorted(its.items())]
                series[m] = mx
                stat[m] = st.median(mx)
        kernel = "v4" if os.path.basename(d) >= V4_SINCE else "campaign"
        out.append(dict(run_id=os.path.basename(d), cell_id=cid, variant=c["variant"], base=base, C=C, kernel=kernel,
                        status=c["status"], nodes=int(c["nnodes"]), model=("Qwen" if "Qwen" in fp.get("model", "") else "K2"),
                        budget=int(c["budget_mib"]), mode=c["mode"], fp=fp, stat=stat, series=series,
                        deterministic=c.get("deterministic"), nsys_path=c.get("nsys_path", "")))
    return out


def main():
    caps = sorted(d for d in glob.glob(os.path.join(RUNS, PREFIX + "*"))
                  if os.path.basename(d) not in EXCLUDE)
    cells = []
    for d in caps:
        try:
            cells += load_capsule(d)
        except Exception as e:  # noqa: BLE001
            print("skip", d, e)
    cells = [c for c in cells if (c["run_id"], c["cell_id"]) not in EXCLUDE_CELLS and c["base"] is not None]
    ok = [c for c in cells if c["status"] == "ok" and c["mode"] == "isolated"]
    print(f"{len(caps)} capsules, {len(cells)} cells, {len(ok)} ok isolated")
    # existing plotted data
    fig_mp = {(r["nodes"], r["model"], r["row_id"], r["budget_mib"]): float(r["total_ms"])
              for r in csv.DictReader(open(os.path.join(ROOT, "figs", "main_perf", "figure_src.csv")))
              if r["row_id"].startswith("ours")}
    fig_ws = {(r["nodes"], r["system"], r["budget_mib"]): float(r["total_ms"])
              for r in csv.DictReader(open(os.path.join(ROOT, "figs", "weak_scaling", "figure_src.csv")))
              if r["system"] in ("ours", "dwire")}

    STALL_FACTOR = 2.0
    stalled = []

    def _fig_of(c):
        if c["base"] in ROW_OF:
            row = ROW_OF[c["base"]]
            return fig_mp.get((str(c["nodes"]), "Qwen3" if c["model"] == "Qwen" else "K2", row, str(c["budget"]))) \
                or fig_mp.get((str(c["nodes"]), c["model"], row, str(c["budget"])))
        return None

    for c in ok:
        if c["fp"].get("pools") == "livecodebench/execution" and "opool" not in c["fp"] and "sched" not in c["fp"]:
            f = _fig_of(c)
            if f and c["stat"]["total_ms"] > STALL_FACTOR * f:
                stalled.append(c)
    stalled_keys = {(c["run_id"], c["cell_id"]) for c in stalled}
    ok = [c for c in ok if (c["run_id"], c["cell_id"]) not in stalled_keys]

    def twin(c, group):
        """LocCap twin total for a pv3c cell: the in-capsule twin when it is clean, else the
        mean of the campaign's clean LocCap cells of the same (nodes, model, base, budget)
        (one binary across the campaign; the fallback is flagged in the rows)."""
        same = [o for o in group if o["base"] == c["base"] and o["C"] == "loccap" and o["nodes"] == c["nodes"]
                and o["budget"] == c["budget"] and o["model"] == c["model"] and o["fp"] == c["fp"]]
        incap = [o for o in same if o["run_id"] == c["run_id"]]
        if incap:
            return incap[0]["stat"]["total_ms"], ""
        if same:
            return st.mean(o["stat"]["total_ms"] for o in same), f"xcap{len(same)}"
        return None, ""

    # ---- main perf: plain lcb pool, no opool/sched, budgets 1/4/16 ----
    plain = [c for c in ok if c["fp"].get("pools") == "livecodebench/execution" and "opool" not in c["fp"]
             and "sched" not in c["fp"]]
    mp_rows = []
    for c in plain:
        if c["base"] in ROW_OF and c["budget"] in (1, 4, 16) and c["C"] in ("1/16", "1/4", "1/2", "1"):
            row = ROW_OF[c["base"]]
            fig = fig_mp.get((str(c["nodes"]), "Qwen3" if c["model"] == "Qwen" else "K2", row, str(c["budget"]))) \
                or fig_mp.get((str(c["nodes"]), c["model"], row, str(c["budget"])))
            tw, twsrc = twin(c, plain)
            mp_rows.append(dict(nodes=c["nodes"], model=c["model"], row_id=row, budget_mib=c["budget"], C=c["C"],
                                total_ms=round(c["stat"]["total_ms"], 3), plan_ms=round(c["stat"].get("plan_ms", 0), 3),
                                loccap_twin_ms=round(tw, 3) if tw else "", d_vs_twin_pct=round(100 * (c["stat"]["total_ms"] / tw - 1), 1) if tw else "",
                                twin_src=twsrc,
                                figure_ms=fig if fig else "", d_vs_figure_pct=round(100 * (c["stat"]["total_ms"] / fig - 1), 1) if fig else "",
                                arm_variant=c["variant"], capsule=c["run_id"], cell_id=c["cell_id"], stat="iter_max_median",
                                kernel=c["kernel"]))
    mp_rows.sort(key=lambda r: (r["nodes"], r["model"], r["row_id"], r["budget_mib"], r["C"], r["capsule"]))
    _write(os.path.join(ROOT, "figs", "main_perf", "figure_src_pv3c.csv"), mp_rows)

    # ---- weak scaling: K2 ours/dwire, budgets 1/64 ----
    ws_rows = []
    for c in plain:
        if c["base"] in WEAK_OF and c["model"] == "K2" and c["budget"] in (1, 64) and c["C"] in ("1/16", "1/4", "1/2", "1"):
            sysn = WEAK_OF[c["base"]]
            fig = fig_ws.get((str(c["nodes"]), sysn, str(c["budget"])))
            tw, twsrc = twin(c, plain)
            ws_rows.append(dict(nodes=c["nodes"], system=sysn, budget_mib=c["budget"], C=c["C"],
                                total_ms=round(c["stat"]["total_ms"], 3),
                                loccap_twin_ms=round(tw, 3) if tw else "", d_vs_twin_pct=round(100 * (c["stat"]["total_ms"] / tw - 1), 1) if tw else "",
                                twin_src=twsrc,
                                figure_ms=fig if fig else "", d_vs_figure_pct=round(100 * (c["stat"]["total_ms"] / fig - 1), 1) if fig else "",
                                arm_variant=c["variant"], capsule=c["run_id"], cell_id=c["cell_id"], stat="iter_max_median",
                                kernel=c["kernel"]))
    ws_rows.sort(key=lambda r: (r["nodes"], r["system"], r["budget_mib"], r["C"], r["capsule"]))
    _write(os.path.join(ROOT, "figs", "weak_scaling", "figure_src_pv3c.csv"), ws_rows)

    # ---- ablation figure: LOO (opool 7, proLaw eval) + matched (proLaw, no opool) ----
    abl_tidy, abl_tab = [], []
    rep_of = defaultdict(int)
    abl_cells = [c for c in ok if c["base"] in ABL_OF and c["fp"].get("pools") == "mmlu/professional_law"]
    for c in sorted(abl_cells, key=lambda c: c["run_id"]):
        study = "loo" if "opool" in c["fp"] else "matched"
        arm = ABL_OF[c["base"]] + ("" if c["C"] == "loccap" else f"_pv3c_{c['C'].replace('/', '')}")
        key = (study, arm, c["run_id"])
        rep_of[(study, arm)] += 0
        if key not in {(k[0], k[1], k[2]) for k in rep_of if len(k) == 3}:
            pass
        reps = sorted({o["run_id"] for o in abl_cells if o["base"] == c["base"] and o["C"] == c["C"]
                       and (("opool" in o["fp"]) == ("opool" in c["fp"]))})
        rep = reps.index(c["run_id"]) + 1
        for i, v in enumerate(c["series"]["total_ms"]):
            abl_tidy.append(dict(study=study, arm=arm, rep=rep, capsule=c["run_id"][-8:], iter=i, total_ms=round(v, 3)))
    by = defaultdict(list)
    for r in abl_tidy:
        by[(r["study"], r["arm"], r["rep"])].append(r["total_ms"])
    agg = defaultdict(lambda: defaultdict(list))
    for (study, arm, rep), vals in by.items():
        agg[(study, arm)]["it0"].append(vals[0])
        agg[(study, arm)]["rest_mean"].append(st.mean(vals[1:]) if len(vals) > 1 else vals[0])
    for (study, arm), d in sorted(agg.items()):
        for metric in ("it0", "rest_mean"):
            v = d[metric]
            abl_tab.append(dict(study=study, arm=arm, metric=metric, n_reps=len(v), total_ms=round(st.mean(v), 2),
                                sd=round(st.stdev(v), 2) if len(v) > 1 else ""))
    _write(os.path.join(ROOT, "figs", "ablation", "ablation_iter_tidy_pv3c.csv"), abl_tidy)
    _write(os.path.join(ROOT, "figs", "ablation", "ablation_tables_pv3c.csv"), abl_tab)

    # ---- cycling ablation: S-A (opool 8) per topic, S-C (sched, dwell 4) ----
    cyc = []
    for c in ok:
        if c["base"] in CYC_ARMS and ("opool" in c["fp"]) and c["budget"] == 64:
            op = c["fp"].get("opool", "")
            n = len([p for p in op.split("+") if p])
            scen = "S-A seen-8" if n == 8 and "sched" not in c["fp"] else ("S-C LOO-proLaw-8" if "sched" in c["fp"] else f"opool{n}")
            ser = c["series"]["total_ms"]
            cyc.append(dict(scenario=scen, variant=c["variant"], base=c["base"], C=c["C"], topic=c["fp"].get("pools", ""),
                            iters=len(ser), mean_ms=round(st.mean(ser), 3), median_ms=round(st.median(ser), 3),
                            it0_ms=round(ser[0], 3), capsule=c["run_id"], cell_id=c["cell_id"],
                            series=" ".join(f"{v:.2f}" for v in ser)))
    cyc.sort(key=lambda r: (r["scenario"], r["base"], r["C"], r["topic"], r["capsule"]))
    _write(os.path.join(ROOT, "figs", "ablation_cycling", "results_tidy_pv3c.csv"), cyc)

    # ---- case study: capsules + nsys paths ----
    cs = []
    for c in cells:
        if c["base"] in CS_ARMS:
            cs.append(dict(mode=c["mode"], variant=c["variant"], base=c["base"], C=c["C"], status=c["status"],
                           family=("S-C dwell4" if "sched" in c["fp"] else "plain lcb"),
                           total_ms=round(c["stat"]["total_ms"], 3) if "total_ms" in c["stat"] else "",
                           capsule=c["run_id"], cell_id=c["cell_id"], nsys_path=c["nsys_path"]))
    cs.sort(key=lambda r: (r["family"], r["base"], r["C"], r["mode"], r["capsule"]))
    _write(os.path.join(ROOT, "figs", "case_study", "pv3c_capsules.csv"), cs)

    # ---- verdict markdown ----
    lines = ["# pv3c recapture — on-par verdict (auto-generated by 39_pv3c_recapture_aggregate.py)", ""]
    lines.append("## main perf (Ours rows; Δ vs in-capsule LocCap twin / vs plotted figure value)")
    lines.append("| nodes | model | row | b | C | kernel | pv3c ms | twin ms | Δtwin | figure ms | Δfig | capsule |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in mp_rows:
        lines.append(f"| {r['nodes']} | {r['model']} | {r['row_id']} | {r['budget_mib']} | {r['C']} | {r['kernel']} | {r['total_ms']} | {r['loccap_twin_ms']}{(' ' + r['twin_src']) if r['twin_src'] else ''} | {r['d_vs_twin_pct']}% | {r['figure_ms']} | {r['d_vs_figure_pct']}% | {r['capsule'][:15]} |")
    lines.append("")
    lines.append("## weak scaling (K2)")
    lines.append("| nodes | system | b | C | kernel | pv3c ms | twin ms | Δtwin | figure ms | Δfig | capsule |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ws_rows:
        lines.append(f"| {r['nodes']} | {r['system']} | {r['budget_mib']} | {r['C']} | {r['kernel']} | {r['total_ms']} | {r['loccap_twin_ms']} | {r['d_vs_twin_pct']}% | {r['figure_ms']} | {r['d_vs_figure_pct']}% | {r['capsule'][:15]} |")
    # chosen C per topology (main perf + weak): minimize mean Δ vs twin over cells with both C present
    lines.append("")
    lines.append("## C choice per topology (candidates cover >= 90% of the plotted cells; paired mean Δ vs LocCap twin, ties within 1 pp broken by the smaller worst-case Δ)")
    bynode = defaultdict(lambda: defaultdict(list))
    for r in mp_rows + ws_rows:
        if r["d_vs_twin_pct"] != "":
            bynode[r["nodes"]][r["C"]].append(r["d_vs_twin_pct"])
    # Rule (pre-registered 03:30 as "mean Δ over cells with both C present", refined 03:40 for
    # coverage): candidate C's are those measured on >= 90% of the topology's plotted cells;
    # candidates are compared on PAIRED cells (measured at every candidate C): smallest mean Δ
    # vs twin, ties within 1 pp broken by the smaller worst-case Δ. Cells missing at the chosen
    # C are filled from the runner-up C in the chosen dataset and flagged.
    cellkey = lambda r: (r["nodes"], r.get("model", ""), r.get("row_id", r.get("system", "")), r["budget_mib"])
    per_c = defaultdict(lambda: defaultdict(dict))   # nodes -> C -> cellkey -> [Δ...]
    for r in mp_rows + ws_rows:
        if r["d_vs_twin_pct"] != "" and r["kernel"] == "campaign":
            per_c[r["nodes"]][r["C"]].setdefault(cellkey(r), []).append(r["d_vs_twin_pct"])
    order = {}

    def pick_c(n):
        d = per_c[n]
        allcells = set().union(*[set(v) for v in d.values()])
        cands = [c for c, v in d.items() if len(v) >= 0.9 * len(allcells)]
        paired = set.intersection(*[set(d[c]) for c in cands]) if cands else set()
        scored = []
        for c in cands:
            vals = [st.mean(d[c][k]) for k in paired]
            scored.append((c, st.mean(vals), max(vals), len(vals)))
        scored.sort(key=lambda t: t[1])
        best = scored[0]
        for t in scored[1:]:
            if t[1] - best[1] <= 1.0 and t[2] < best[2]:
                best = t
        order[n] = [best[0]] + [t[0] for t in scored if t[0] != best[0]] + \
                   [c for c in d if c not in cands]
        return best[0], scored, len(allcells)
    for n in sorted(per_c):
        best, scored, ncell = pick_c(n)
        cov = "; ".join(f"C={c}: {len(v)}/{ncell} cells" for c, v in sorted(per_c[n].items()))
        parts = [f"C={c}: mean {m:+.1f}% / worst {w:+.1f}% over {k} paired cells" for c, m, w, k in scored]
        lines.append(f"- {n}n: coverage {cov} | " + "; ".join(parts) + f" -> **C={best}**")
    lines.append("")
    lines.append(f"## stall-class cells excluded (iter-max median > {STALL_FACTOR}x the plotted value; the known Qwen-16n-b16 l1 stall)")
    for c in stalled:
        lines.append(f"- {c['run_id'][:15]} {c['cell_id']}: {c['stat']['total_ms']:.1f} ms (l1 {c['stat'].get('l1_ms', 0):.1f})")
    # chosen-C dataset: one value per plotted cell at the topology's chosen C (mean over capsules)
    chosen = {n: order[n][0] for n in order}
    ch_rows = []
    for name, rows, key in (("main_perf", mp_rows, lambda r: (r["nodes"], r["model"], r["row_id"], r["budget_mib"])),
                            ("weak_scaling", ws_rows, lambda r: (r["nodes"], r["system"], r["budget_mib"]))):
        grp = defaultdict(lambda: defaultdict(list))
        for r in rows:
            grp[key(r)][r["C"]].append(r)
        for k, byc in sorted(grp.items()):
            n = k[0]
            use = next((c for c in order.get(n, []) if c in byc), None)
            if use is None:
                continue
            rs = byc[use]
            v4 = [r for r in rs if r["kernel"] == "v4"]
            if v4:
                rs = v4
            fig = rs[0]["figure_ms"]
            v = st.mean(r["total_ms"] for r in rs)
            tw = [r["d_vs_twin_pct"] for r in rs if r["d_vs_twin_pct"] != ""]
            d = dict(figure=name, nodes=n, key="/".join(str(x) for x in k[1:]), C=use,
                     kernel=("v4" if v4 else "campaign"),
                     note=("" if use == chosen.get(n) else f"fill: chosen C={chosen.get(n)} missing"),
                     n_capsules=len(rs), pv3c_ms=round(v, 3), d_vs_twin_pct=round(st.mean(tw), 1) if tw else "",
                     figure_ms=fig, d_vs_figure_pct=round(100 * (v / fig - 1), 1) if fig else "",
                     capsules=" ".join(r["capsule"][:15] for r in rs))
            ch_rows.append(d)
    _write(os.path.join(HERE, "39_pv3c_chosen_dataset.csv"), ch_rows)
    lines.append("")
    lines.append("## chosen-C dataset (mean over capsules at the topology's chosen C; 39_pv3c_chosen_dataset.csv)")
    lines.append("| figure | nodes | cell | C | kernel | n | pv3c ms | Δtwin | figure ms | Δfig | note |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ch_rows:
        lines.append(f"| {r['figure']} | {r['nodes']} | {r['key']} | {r['C']} | {r['kernel']} | {r['n_capsules']} | {r['pv3c_ms']} | {r['d_vs_twin_pct']}% | {r['figure_ms']} | {r['d_vs_figure_pct']}% | {r['note']} |")
    lines.append("")
    lines.append("## ablation figure (K2 4n b64; per-rep it0 / rest_mean, mean over reps)")
    lines.append("| study | arm | metric | n | ms | sd |")
    lines.append("|---|---|---|---|---|---|")
    for r in abl_tab:
        lines.append(f"| {r['study']} | {r['arm']} | {r['metric']} | {r['n_reps']} | {r['total_ms']} | {r['sd']} |")
    lines.append("")
    lines.append(f"## cycling ablation: {len(cyc)} cells in results_tidy_pv3c.csv; case study: {len(cs)} cells in pv3c_capsules.csv")
    open(os.path.join(HERE, "39_pv3c_recapture_verdict.md"), "w").write("\n".join(lines) + "\n")
    print("wrote verdict with", len(mp_rows), "main-perf rows,", len(ws_rows), "weak rows,", len(abl_tab), "ablation table rows,", len(cyc), "cycling cells,", len(cs), "case-study cells")


def _write(path, rows):
    if not rows:
        print("no rows for", path)
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, len(rows))


if __name__ == "__main__":
    main()
