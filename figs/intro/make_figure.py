#!/usr/bin/env python3
"""Intro figure v1: (a) overlapped per-expert demand for two topics, fixed
expert-ID axis; (b) two 16x16 rank-to-rank traffic heatmaps (one per topic)
stacked on the right, shared color scale. Single-column NSDI, built at final
physical size. Every aesthetic value lives in CONFIG.

  python make_figure.py            # writes intro_v1.pdf + intro_v1.png next to this file
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = dict(
    SRC=os.path.join(HERE, "figure_src.csv"),
    OUT_STEM=os.path.join(HERE, "intro_v3"),
    LAYOUT="stack",                          # "stack" (v3: bars on top, heatmaps below) | "side" (v1/v2)
    BAR_FRAC=1 / 3,                          # v3: share of the plot height for the bar panel
    VGAP_IN=0.50,                            # v3.1: gap holds the x label + "(a)" sub-label + heatmap titles
    BAR_TOPICS=["livecodebench/execution"],  # v3.1: panel (a) shows LiveCodeBench only (prof. law's 33x outlier hides the shape)
    PANEL_LABELS=["(a) Expert activation frequency", "(b) GPU-to-GPU dispatch traffic"],
    SUBLABEL_A_IN=0.20, SUBLABEL_B_IN=0.40,  # distance below each panel's axes (inches)
    HM_XLABEL_IN=0.16,                       # v3.1: "Receiver rank" a little further below the maps
    HM_GAP_IN=0.14,                          # v3: gap between the two heatmaps (inches)
    # --- geometry (inches; USENIX column 3.33 in, text height 9.0 in) ---
    FIG_W=3.33, FIG_H=2.60,                  # v3.1: +0.2 in for the sub-labels under each panel
    LEFT=0.15, RIGHT=0.895, BOTTOM=0.185, TOP=0.96,   # LEFT holds the two-line y label + panel tags   # RIGHT leaves room for the colorbar label
    SPLIT=0.50,                             # fraction of the width given to the bar panel
    WSPACE_IN=0.42,                          # gap between bar panel and heatmaps (inches)
    HSPACE_IN=0.30,                          # gap between the two heatmaps (inches)
    CBAR_W_IN=0.07, CBAR_GAP_IN=0.05,        # shared colorbar right of the heatmaps
    # --- topics (display order = draw order; the second is drawn on top) ---
    TOPICS=["livecodebench/execution", "mmlu/professional_law"],
    TOPIC_NAMES={"livecodebench/execution": "LiveCodeBench", "mmlu/professional_law": "MMLU prof. law"},
    # --- bars: two overlappable colors (translucent; overlap reads as a third, darker tone) ---
    # v3.2: one blue family for both panels — bars = the dark end of the heatmap ramp
    BAR_COLORS=["#2171b5", "#6baed6"], BAR_ALPHA=0.95, BAR_LW=0.0, BAR_WIDTH=1.0,
    Y_LABEL="Normalized\ntoken count", X_LABEL="Expert ID",   # two lines: the v3 bar panel is only ~0.6 in tall
    # v2: ORIENT "h" = horizontal bars (x = normalized count, y = experts, most popular on top);
    #     SORT "each" = every topic sorted by its own count (y is then a rank, not an ID),
    #          "none" = fixed expert IDs (v1), or a topic name = both follow that topic's order
    ORIENT="v", SORT="each", SORTED_AXIS_LABEL="Expert rank",
    X_LOG=False, X_LOG_MIN=0.05,             # log count axis (companion render); bars start at X_LOG_MIN
    Y_MAX=None,                              # None = data max; a number clips (bars above are marked)
    UNIFORM_LINE=dict(color="#0b0b0b", lw=0.5, ls=(0, (2, 1.5))),
    # --- heatmaps ---
    CMAP="Blues",                            # v3.2: matplotlib sequential name (white -> dark blue), or a list of hex stops
    CMAP_NAME="custom",
    VMIN=0.0, VMAX=None,                     # shared scale; None = max over both matrices
    NODE_LINE=dict(color="#0b0b0b", lw=0.6),
    HM_EDGE_LW=0.5,                          # thin frame so the pale low cells do not dissolve into the page
    HM_XLABEL="Receiver GPU", HM_YLABEL="Sender GPU",   # v3.3: no "rank" in the intro
    CBAR_LABEL="Normalized traffic",
    HM_TICKS=[0, 4, 8, 12],
    # --- type ---
    FONT_FAMILY=["Helvetica", "Arial", "DejaVu Sans"],
    FS=dict(label=7, tick=6, legend=6.5, title=7, panel=6.5, cbar=6),
    INK="#0b0b0b", INK2="#52514e",
    PANEL_TAGS=["(a)", "(b)"],
    DPI=300,
)

def load(cfg):
    experts, cells, prov = {}, {}, []
    with open(cfg["SRC"]) as f:
        lines = [ln for ln in f if not ln.startswith("#") or prov.append(ln.strip())]
    for r in csv.DictReader(lines):
        if r["kind"] == "expert":
            experts.setdefault(r["topic"], {})[int(r["i"])] = float(r["value"])
        else:
            cells.setdefault(r["topic"], {})[(int(r["i"]), int(r["j"]))] = float(r["value"])
    G = max(max(d) for d in experts.values()) + 1
    W = max(max(s for s, _ in d) for d in cells.values()) + 1
    E = {t: np.array([experts[t][e] for e in range(G)]) for t in experts}
    M = {t: np.array([[cells[t][(s, d)] for d in range(W)] for s in range(W)]) for t in cells}
    L = int(next(p.split("L=")[1].split()[0] for p in prov if "L=" in p))
    return E, M, G, W, L, prov

def main():
    cfg = dict(CONFIG)
    # optional overrides: --ymax 12 (clip + annotate) --suffix _clip12
    args = sys.argv[1:]
    if "--ymax" in args:
        cfg["Y_MAX"] = float(args[args.index("--ymax") + 1])
    if "--logx" in args:
        cfg["X_LOG"] = True
    if "--suffix" in args:
        cfg["OUT_STEM"] += args[args.index("--suffix") + 1]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": cfg["FONT_FAMILY"],
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2, "ytick.major.size": 2})
    E, M, G, W, L, prov = load(cfg)
    fig = plt.figure(figsize=(cfg["FIG_W"], cfg["FIG_H"]))
    fw, fh = cfg["FIG_W"], cfg["FIG_H"]
    x0, x1, y0, y1 = cfg["LEFT"], cfg["RIGHT"], cfg["BOTTOM"], cfg["TOP"]
    cb_w, cb_g = cfg["CBAR_W_IN"] / fw, cfg["CBAR_GAP_IN"] / fw
    stack = cfg["LAYOUT"] == "stack"
    if stack:
        vg, hg = cfg["VGAP_IN"] / fh, cfg["HM_GAP_IN"] / fw
        H = y1 - y0 - vg
        bar_h = H * cfg["BAR_FRAC"]; hm_h = H - bar_h
        hm_w = (x1 - x0 - cb_w - cb_g - hg) / 2
        side = min(hm_w * fw, hm_h * fh)                    # square heatmaps
        hm_w, hm_h = side / fw, side / fh
        ax_bar = fig.add_axes([x0, y1 - bar_h, x1 - x0, bar_h])
        ax_hm = [fig.add_axes([x0, y0, hm_w, hm_h]), fig.add_axes([x0 + hm_w + hg, y0, hm_w, hm_h])]
        ax_cb = fig.add_axes([x0 + 2 * hm_w + hg + cb_g, y0, cb_w, hm_h])
        hm_x0 = x0
    else:
        ws, hs = cfg["WSPACE_IN"] / fw, cfg["HSPACE_IN"] / fh
        bar_w = (x1 - x0 - ws - cb_w - cb_g) * cfg["SPLIT"]
        hm_x0 = x0 + bar_w + ws
        hm_w = x1 - hm_x0 - cb_w - cb_g
        hm_h = (y1 - y0 - hs) / 2
        side = min(hm_w * fw, hm_h * fh)
        hm_w, hm_h = side / fw, side / fh
        ax_bar = fig.add_axes([x0, y0, bar_w, y1 - y0])
        ax_hm = [fig.add_axes([hm_x0, y1 - hm_h, hm_w, hm_h]),
                 fig.add_axes([hm_x0, y1 - 2 * hm_h - hs, hm_w, hm_h])]
        ax_cb = fig.add_axes([hm_x0 + hm_w + cb_g, y1 - 2 * hm_h - hs, cb_w, 2 * hm_h + hs])

    # ---- (a) overlapped bars ----
    ids = np.arange(G)
    ymax = cfg["Y_MAX"] or max(float(v.max()) for v in E.values()) * 1.04
    def ordered(t):
        v = E[t]
        if cfg["SORT"] == "each":
            return np.sort(v)[::-1]
        if cfg["SORT"] == "none":
            return v
        return v[np.argsort(E[cfg["SORT"]])[::-1]]
    horiz = cfg["ORIENT"] == "h"
    bar_topics = cfg.get("BAR_TOPICS") or cfg["TOPICS"]
    ymax = cfg["Y_MAX"] or float(np.ceil(max(float(E[t].max()) for t in bar_topics) / 5) * 5)   # next multiple of 5 so the top tick shows
    for t, c in zip(cfg["TOPICS"], cfg["BAR_COLORS"]):
        if t not in bar_topics:
            continue
        v = ordered(t)
        kw = dict(color=c, alpha=cfg["BAR_ALPHA"], linewidth=cfg["BAR_LW"], label=cfg["TOPIC_NAMES"][t], align="edge")
        if horiz:
            ax_bar.barh(ids, np.minimum(v, ymax), height=cfg["BAR_WIDTH"], **kw)
        else:
            ax_bar.bar(ids, np.minimum(v, ymax), width=cfg["BAR_WIDTH"], **kw)
        if cfg["Y_MAX"]:
            for e in np.where(v > ymax)[0]:
                xy = (ymax, e + .5) if horiz else (e + .5, ymax)
                ax_bar.annotate(f"{v[e]:.0f}×", xy, xytext=(-1, 0) if horiz else (0, -1), textcoords="offset points",
                                ha="right" if horiz else "center", va="center" if horiz else "top",
                                fontsize=cfg["FS"]["tick"], color=cfg["INK"])
    cat_label = cfg["X_LABEL"] if cfg["SORT"] == "none" else cfg["SORTED_AXIS_LABEL"]
    if horiz:
        ax_bar.axvline(1.0, **cfg["UNIFORM_LINE"], zorder=3)
        ax_bar.text(1.0, G - 1, " uniform", ha="left", va="bottom", fontsize=cfg["FS"]["tick"], color=cfg["INK2"])
        ax_bar.set_ylim(G, 0)                                     # most popular expert at the top
        if cfg["X_LOG"]:
            ax_bar.set_xscale("log"); ax_bar.set_xlim(cfg["X_LOG_MIN"], ymax)
            ax_bar.set_xticks([0.1, 1, 10]); ax_bar.set_xticklabels(["0.1", "1", "10"])
            ax_bar.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=(2, 5), numticks=20))
            ax_bar.tick_params(axis="x", which="minor", length=1.2)
        else:
            ax_bar.set_xlim(0, ymax)
        ax_bar.set_yticks([]); ax_bar.set_ylabel(cat_label, fontsize=cfg["FS"]["label"], labelpad=2)
        ax_bar.set_xlabel(cfg["Y_LABEL"], fontsize=cfg["FS"]["label"], labelpad=2)
        ax_bar.tick_params(axis="x", labelsize=cfg["FS"]["tick"], pad=1.5)
    else:
        ax_bar.axhline(1.0, **cfg["UNIFORM_LINE"], zorder=3)
        if not cfg["X_LOG"]:
            ax_bar.text(G * 0.99, 1.0, "uniform", ha="right", va="bottom", fontsize=cfg["FS"]["tick"], color=cfg["INK2"])
        ax_bar.set_xlim(0, G)
        ax_bar.set_xticks([]); ax_bar.set_xlabel(cat_label, fontsize=cfg["FS"]["label"], labelpad=2)
        ax_bar.set_ylabel(cfg["Y_LABEL"], fontsize=cfg["FS"]["label"], labelpad=2)
        if cfg["X_LOG"]:                                       # log count axis (vertical form)
            ax_bar.set_yscale("log"); ax_bar.set_ylim(cfg["X_LOG_MIN"], ymax)
            ax_bar.set_yticks([0.1, 1, 10]); ax_bar.set_yticklabels(["0.1", "1", "10"])
            ax_bar.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=(2, 5), numticks=20))
            ax_bar.tick_params(axis="y", which="minor", length=1.2)
            ax_bar.text(G * 0.99, 0.93, "uniform", ha="right", va="top", fontsize=cfg["FS"]["tick"], color=cfg["INK2"])
        else:
            ax_bar.set_ylim(0, ymax)
            ax_bar.set_yticks([t for t in ((0, 10, 20, 30) if ymax > 15 else (0, 5, 10)) if t <= ymax])
        ax_bar.tick_params(axis="y", labelsize=cfg["FS"]["tick"], pad=1.5)
    for sp in ("top", "right"):
        ax_bar.spines[sp].set_visible(False)
    if len(bar_topics) > 1:
        leg = ax_bar.legend(fontsize=cfg["FS"]["legend"], frameon=False, loc="upper right" if not horiz else "lower right",
                            handlelength=1.0, handletextpad=0.5, borderaxespad=0.2, labelspacing=0.3)
        for h in leg.legend_handles:
            h.set_alpha(cfg["BAR_ALPHA"])
    else:   # v3.3: still a legend entry with its color swatch
        leg = ax_bar.legend(fontsize=cfg["FS"]["legend"], frameon=False, loc="upper right" if not horiz else "lower right",
                            handlelength=1.0, handletextpad=0.5, borderaxespad=0.2)
        for h in leg.legend_handles:
            h.set_alpha(cfg["BAR_ALPHA"])

    # ---- (b) two heatmaps, shared scale ----
    cmap = (matplotlib.colormaps[cfg["CMAP"]] if isinstance(cfg["CMAP"], str)
            else LinearSegmentedColormap.from_list(cfg["CMAP_NAME"], cfg["CMAP"]))
    vmax = cfg["VMAX"] or max(float(m.max()) for m in M.values())
    im = None
    for ax, t in zip(ax_hm, cfg["TOPICS"]):
        im = ax.imshow(M[t], cmap=cmap, vmin=cfg["VMIN"], vmax=vmax, origin="upper",
                       interpolation="nearest", aspect="equal")
        for n in range(1, W // L):
            ax.axhline(n * L - .5, **cfg["NODE_LINE"]); ax.axvline(n * L - .5, **cfg["NODE_LINE"])
        for sp in ax.spines.values():
            sp.set_linewidth(cfg["HM_EDGE_LW"])
        ax.set_title(cfg["TOPIC_NAMES"][t], fontsize=cfg["FS"]["title"], pad=2, color=cfg["INK"])
        ax.set_xticks(cfg["HM_TICKS"]); ax.set_yticks(cfg["HM_TICKS"])
        ax.tick_params(labelsize=cfg["FS"]["tick"], pad=1.5, length=1.5)
    ax_hm[0].set_ylabel(cfg["HM_YLABEL"], fontsize=cfg["FS"]["label"], labelpad=2)
    if stack:                                              # labels once: y on the left map, x centered under both
        ax_hm[1].tick_params(labelleft=False)
        for ax in ax_hm:
            ax.tick_params(labelbottom=True)
        pos0, pos1 = ax_hm[0].get_position(), ax_hm[1].get_position()
        fig.text((pos0.x0 + pos1.x1) / 2, pos0.y0 - cfg["HM_XLABEL_IN"] / fh, cfg["HM_XLABEL"], ha="center", va="top",
                 fontsize=cfg["FS"]["label"])
    else:
        ax_hm[1].set_ylabel(cfg["HM_YLABEL"], fontsize=cfg["FS"]["label"], labelpad=2)
        ax_hm[0].tick_params(labelbottom=False)
        ax_hm[1].set_xlabel(cfg["HM_XLABEL"], fontsize=cfg["FS"]["label"], labelpad=2)
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label(cfg["CBAR_LABEL"], fontsize=cfg["FS"]["label"], labelpad=2)
    cb.ax.tick_params(labelsize=cfg["FS"]["cbar"], pad=1.5, length=1.5)
    cb.outline.set_linewidth(0.5)
    ticks = [t for t in (0, 0.5, 1, 1.5, 2) if t <= vmax]
    cb.set_ticks(ticks); cb.set_ticklabels([("1×" if t == 1 else f"{t:g}") for t in ticks])

    # panel tags
    if stack:   # v3.1: sub-labels centered UNDER each panel
        pb, p0, p1 = ax_bar.get_position(), ax_hm[0].get_position(), ax_hm[1].get_position()
        fig.text((pb.x0 + pb.x1) / 2, pb.y0 - cfg["SUBLABEL_A_IN"] / fh, cfg["PANEL_LABELS"][0],
                 fontsize=cfg["FS"]["panel"], va="top", ha="center")
        fig.text((p0.x0 + p1.x1) / 2, p0.y0 - cfg["SUBLABEL_B_IN"] / fh, cfg["PANEL_LABELS"][1],
                 fontsize=cfg["FS"]["panel"], va="top", ha="center")
    else:
        fig.text(x0 - 0.10, y1 + 0.005, cfg["PANEL_TAGS"][0], fontsize=cfg["FS"]["panel"], va="bottom", ha="left")
        fig.text(hm_x0 - 0.10, y1 + 0.005, cfg["PANEL_TAGS"][1], fontsize=cfg["FS"]["panel"], va="bottom", ha="left")

    for ext in ("pdf", "png"):
        fig.savefig(f"{cfg['OUT_STEM']}.{ext}", dpi=cfg["DPI"])
    print("wrote", cfg["OUT_STEM"], "pdf/png", f"{fw:.2f}x{fh:.2f} in", "vmax", round(vmax, 2), "ymax", round(ymax, 1))

if __name__ == "__main__":
    main()
