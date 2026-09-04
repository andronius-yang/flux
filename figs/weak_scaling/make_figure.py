#!/usr/bin/env python3
"""Weak-scaling figure generator (verA latency / verB throughput). See SPEC.md.

Usage:  python3 make_figure.py                    (COMET baseline; weak_scaling_ver{A,B}.{pdf,png})
        python3 make_figure.py --baseline nvshmem (NVSHMEM+GEMM ring baseline;
                                                   weak_scaling_nvshmem_ver{A,B}.{pdf,png})

Every aesthetic decision lives in CONFIG below; VERSIONS holds the per-version
differences. Values are true points/inches at final size (\\columnwidth).
"""
import csv
import math
import os
import sys

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # reproducible PDF bytes
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ============================== CONFIG =======================================
CONFIG = dict(
    NODES=[2, 4, 8, 16, 32],                  # x positions, equal spacing
    SYSTEMS=["comet", "ours"],                # draw order (ours on top)
    REF="comet",                              # speedup = ref / ours
    SERIES={                                  # identity: color + marker
        "comet": dict(label="COMET", color="#999933", marker="s"),
        "ours": dict(label="Ours", color="#4878b0", marker="o"),
        # ring baseline: the main figure's nvshmem_gemm sand (#ddaa33), triangle
        "nvshmem": dict(label="NVSHMEM+GEMM", color="#ddaa33", marker="^"),
    },
    LINE=dict(lw=1.1, ms=3.6, mew=0.0, mec=None),      # markers: no edge stroke
    BARS=dict(label="Speedup vs COMET", face="#d9d9d9", edge="#8f8f8f",   # label follows REF (see BASELINES)
              lw=0.4, width=0.55, fmt="{:.2f}×", label_pad_pt=1.5,
              label_pos="auto",   # auto: above the bar unless a line marker
                                  # sits in that band, then inside at the base
              label_h_pt=6.5),    # label glyph height used by the auto test
    BAR_HEADROOM=1.5,                         # right ylim = ceil_nice(max*this)
    HEADROOM=1.12,                            # left ylim = ceil_nice(max*this)
    NICE_STEPS=[(10, 1), (60, 2), (150, 5), (400, 10), (1e9, 50)],
    FAIL_NOTE=dict(text="OOM", glyph="✗", glyph_size=8, pad_pt=2.0,
                   text_weight="normal"),   # x on the axis baseline + note above
    INK=dict(primary="#0b0b0b", secondary="#52514e", muted="#898781",
             grid="#e1e0d9", axis="#c3c2b7", right="#6f6f6f"),
    # ---- layout ----
    FIG_W=3.33, FIG_H=1.45,                   # in; USENIX \columnwidth, <=1/5 page w/ caption
    MARGINS=dict(left=0.135, right=0.86, top=0.80, bottom=0.21),
    X_LABEL="Nodes",
    Y_LABELS=dict(A="Latency (ms)", B="Throughput (Mtok/s)"),
    RIGHT_LABEL="Speedup vs COMET",
    N_YTICKS=4, N_RTICKS=3,
    LEGEND=dict(ncol=3, y=1.02, colspacing=1.0, handlelength=1.6),
    # ---- typography ----
    FONT_FAMILY=["Helvetica", "Arial", "DejaVu Sans"],
    FONT_SIZES=dict(legend=6.5, label=6.5, tick=6, bar=5.8, note=5.5),
    # ---- baselines: what --baseline swaps (REF, draw order, labels, outputs) ----
    BASELINES={
        "comet": dict(REF="comet", SYSTEMS=["comet", "ours"],
                      RIGHT_LABEL="Speedup vs COMET", BAR_LABEL="Speedup vs COMET",
                      prefix="weak_scaling_ver"),
        "nvshmem": dict(REF="nvshmem", SYSTEMS=["nvshmem", "ours"],
                        RIGHT_LABEL="Speedup vs NVSHMEM", BAR_LABEL="Speedup vs NVSHMEM",
                        LEGEND=dict(colspacing=0.7),   # longer names: tighter row
                        prefix="weak_scaling_nvshmem_ver"),
    },
    # ---- versions ----
    VERSIONS={
        "A": dict(metric="total_ms", outputs=[("weak_scaling_verA.pdf", {}),
                                              ("weak_scaling_verA.png", {"dpi": 300})]),
        "B": dict(metric="throughput_mtok_s",
                  outputs=[("weak_scaling_verB.pdf", {}),
                           ("weak_scaling_verB.png", {"dpi": 300})]),
    },
)
# =============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))


def load(cfg):
    """-> data[(nodes, system)] = row dict; asserts the grid is complete."""
    data = {}
    with open(os.path.join(HERE, "figure_src.csv"), newline="") as f:
        for r in csv.DictReader(f):
            data[(int(r["nodes"]), r["system"])] = r
    for n in cfg["NODES"]:
        for s in cfg["SYSTEMS"]:
            assert (n, s) in data, f"missing row {n}n {s}"
    return data


def ceil_nice(v, cfg):
    step = next(s for bound, s in cfg["NICE_STEPS"] if v < bound)
    return step * math.ceil(v / step)


def plot(data, cfg, version):
    vcfg = cfg["VERSIONS"][version]
    metric = vcfg["metric"]
    fs = cfg["FONT_SIZES"]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": cfg["FONT_FAMILY"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.6, "axes.edgecolor": cfg["INK"]["axis"],
        "text.color": cfg["INK"]["primary"],
    })
    fig, ax = plt.subplots(figsize=(cfg["FIG_W"], cfg["FIG_H"]))
    fig.subplots_adjust(**cfg["MARGINS"])
    xs = list(range(len(cfg["NODES"])))
    ax2 = ax.twinx()

    # ---- speedup bars (right axis, behind everything) ----
    bars = cfg["BARS"]
    sp = []
    for i, n in enumerate(cfg["NODES"]):
        v = data[(n, "ours")]["speedup_vs_" + cfg["REF"]]
        if v:
            sp.append((i, float(v)))
    rmax = ceil_nice(max(v for _, v in sp) * cfg["BAR_HEADROOM"], cfg)
    ax2.set_ylim(0, rmax)
    ax2.bar([i for i, _ in sp], [v for _, v in sp], width=bars["width"],
            facecolor=bars["face"], edgecolor=bars["edge"], linewidth=bars["lw"],
            zorder=1)
    rpt = rmax / (cfg["FIG_H"] * ax2.get_position().height * 72)  # data/pt
    ax2.yaxis.set_major_locator(
        matplotlib.ticker.MaxNLocator(cfg["N_RTICKS"], integer=True))
    ax2.tick_params(axis="y", labelsize=fs["tick"], length=1.5, width=0.6,
                    pad=1.5, colors=cfg["INK"]["right"])
    ax2.set_ylabel(cfg["RIGHT_LABEL"], fontsize=fs["label"],
                   color=cfg["INK"]["right"], labelpad=2)
    ax2.spines["right"].set_color(cfg["INK"]["right"])
    for side in ("top", "left", "bottom"):
        ax2.spines[side].set_visible(False)

    # ---- lines (left axis) ----
    ln = cfg["LINE"]
    vals = []
    last = {}
    for s in cfg["SYSTEMS"]:
        st = cfg["SERIES"][s]
        pts = [(i, float(data[(n, s)][metric])) for i, n in enumerate(cfg["NODES"])
               if data[(n, s)][metric]]
        vals += [v for _, v in pts]
        last[s] = pts[-1]
        ax.plot([i for i, _ in pts], [v for _, v in pts], color=st["color"],
                marker=st["marker"], lw=ln["lw"], ms=ln["ms"], mew=ln["mew"],
                mec=ln["mec"], zorder=3, clip_on=False)
    ylim = ceil_nice(max(vals) * cfg["HEADROOM"], cfg)
    ax.set_ylim(0, ylim)

    # ---- speedup value labels (placed with knowledge of the markers) ----
    ax_h_pt = cfg["FIG_H"] * ax.get_position().height * 72
    band = (bars["label_pad_pt"] + bars["label_h_pt"]) / ax_h_pt   # axes frac
    mk = ln["ms"] * 0.6 / ax_h_pt                                    # marker radius, frac
    for i, v in sp:
        n = cfg["NODES"][i]
        marks = [float(data[(n, s)][metric]) / ylim for s in cfg["SYSTEMS"]
                 if data[(n, s)][metric]]
        top = v / rmax
        pos = bars["label_pos"]
        if pos == "auto":
            clash = any(top - mk <= m <= top + band + mk for m in marks)
            pos = "base" if clash else "top"
        if pos == "top":
            ax2.text(i, v + bars["label_pad_pt"] * rpt, bars["fmt"].format(v),
                     ha="center", va="bottom", fontsize=fs["bar"],
                     color=cfg["INK"]["primary"], zorder=4)
        else:
            ax2.text(i, bars["label_pad_pt"] * rpt, bars["fmt"].format(v),
                     ha="center", va="bottom", fontsize=fs["bar"],
                     color=cfg["INK"]["primary"], zorder=4)

    # ---- baseline fail marker: an x ON the x axis at each slot it lacks, note above ----
    fn = cfg["FAIL_NOTE"]
    ref = cfg["REF"]
    dpt = ylim / (cfg["FIG_H"] * ax.get_position().height * 72)   # data/pt
    for i, n in enumerate(cfg["NODES"]):
        if data[(n, ref)][metric]:
            continue
        ax.text(i, 0, fn["glyph"], ha="center", va="bottom", fontsize=fn["glyph_size"],
                color=cfg["SERIES"][ref]["color"], fontweight="bold", zorder=4,
                clip_on=False)   # seated on the x-axis baseline
        ax.text(i, (fn["pad_pt"] + fn["glyph_size"] * 1.0) * dpt, fn["text"],
                ha="center", va="bottom", fontsize=fs["note"],
                color=cfg["SERIES"][ref]["color"], fontweight=fn["text_weight"],
                zorder=4)

    # ---- chrome ----
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(n) for n in cfg["NODES"]], fontsize=fs["tick"])
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.tick_params(axis="y", labelsize=fs["tick"], length=1.5, width=0.6, pad=1.5)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(cfg["N_YTICKS"]))
    ax.set_xlabel(cfg["X_LABEL"], fontsize=fs["label"], labelpad=1.5)
    ax.set_ylabel(cfg["Y_LABELS"][version], fontsize=fs["label"], labelpad=2)
    # gridlines live on the LOWER axes (ax2) so they render beneath the bars
    # and their value labels; positions follow the left axis's ticks
    for y in ax.get_yticks():
        if 0 < y <= ylim:
            ax2.axhline(y / ylim * rmax, color=cfg["INK"]["grid"], lw=0.4, zorder=0)
    ax.set_zorder(ax2.get_zorder() + 1)   # lines above bars
    ax.patch.set_visible(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # ---- legend ----
    handles = [Line2D([], [], color=cfg["SERIES"][s]["color"],
                      marker=cfg["SERIES"][s]["marker"], lw=ln["lw"], ms=ln["ms"],
                      mew=ln["mew"], mec=ln["mec"], label=cfg["SERIES"][s]["label"])
               for s in cfg["SYSTEMS"]]
    handles.append(Patch(facecolor=bars["face"], edgecolor=bars["edge"],
                         linewidth=bars["lw"], label=bars["label"]))
    lg = cfg["LEGEND"]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, lg["y"]),
               ncol=lg["ncol"], frameon=False, fontsize=fs["legend"],
               columnspacing=lg["colspacing"], handlelength=lg["handlelength"],
               handletextpad=0.5, borderaxespad=0.0)
    return fig


def configure(baseline):
    """CONFIG specialized to one baseline (REF, SYSTEMS, labels, output names)."""
    b = CONFIG["BASELINES"][baseline]
    cfg = dict(CONFIG)
    cfg["REF"], cfg["SYSTEMS"], cfg["RIGHT_LABEL"] = b["REF"], b["SYSTEMS"], b["RIGHT_LABEL"]
    cfg["BARS"] = dict(CONFIG["BARS"], label=b["BAR_LABEL"])
    cfg["LEGEND"] = dict(CONFIG["LEGEND"], **b.get("LEGEND", {}))
    cfg["VERSIONS"] = {
        v: dict(vc, outputs=[(b["prefix"] + v + ext, kw) for ext, kw in
                             [(".pdf", {}), (".png", {"dpi": 300})]])
        for v, vc in CONFIG["VERSIONS"].items()}
    return cfg


def main():
    baseline = "comet"
    if "--baseline" in sys.argv:
        baseline = sys.argv[sys.argv.index("--baseline") + 1]
    cfg = configure(baseline)
    data = load(cfg)
    for version, vcfg in cfg["VERSIONS"].items():
        fig = plot(data, cfg, version)
        for name, kw in vcfg["outputs"]:
            fig.savefig(os.path.join(HERE, name), **kw)
            print("wrote", name)
        plt.close(fig)


if __name__ == "__main__":
    main()
