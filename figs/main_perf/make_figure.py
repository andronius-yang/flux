#!/usr/bin/env python3
"""Main performance figure generator. See SPEC.md; data from figure_src.csv.

Usage:  python3 make_figure.py        (writes main_perf.pdf + main_perf.png)

Every aesthetic decision lives in CONFIG below. Values are true points/inches
at final size: the PDF is placed at \\linewidth in a figure* environment.
"""
import csv
import math
import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # reproducible PDF bytes
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================== CONFIG =======================================
CONFIG = dict(
    # ---- data selection ----
    BUDGETS=[1, 4, 16],                       # MiB, group order left->right
    COLS=["4", "16"],                         # topology per subfigure column
    ROWS=["K2", "Qwen"],                      # model per subfigure row
    SYSTEMS=[                                 # fixed bar order (SPEC 2.1)
        "fast_gemm", "nvshmem_gemm", "moonep", "eplb", "epic", "comet", "OURS",
    ],
    OURS_CANDIDATES=[                         # best-of pool for the OURS bar
        "ours1_tokencomm", "ours2_nooverlap", "ours2_direct",
        "ours12", "ours12_dispatch",
    ],
    # ---- legend ----
    LEGEND_NAMES={                            # system -> legend text
        "fast_gemm": "FAST+GEMM", "nvshmem_gemm": "NVSHMEM A2AV+GEMM",
        "moonep": "MoonEP", "eplb": "EPLB", "epic": "EPIC",
        "comet": "COMET", "OURS": "Ours",
    },
    LEGEND_NCOL=7,                            # 7 = single row; 4 -> 4+3 rows
    LEGEND_COLSPACING=0.9,                    # matplotlib columnspacing
    # ---- colors (muted academic set, validated 2026-09-01 -- SPEC 5;
    #      revalidate on ANY change: display-adjacent CVD dE 16.1, normal 18.6) ----
    COLORS={
        "fast_gemm": "#c0653f", "nvshmem_gemm": "#ddaa33",
        "moonep": "#2f7a4d", "eplb": "#44bb99", "epic": "#9d4a4a",
        "comet": "#999933", "OURS": "#4878b0",
    },
    INK=dict(primary="#0b0b0b", secondary="#52514e", muted="#898781",
             grid="#e1e0d9", axis="#c3c2b7"),
    # ---- bars ----
    GROUP_WIDTH=0.84,                         # fraction of the unit group slot
    BAR_GAP_FRAC=0.18,                        # gap between bars, frac of bar w
    EDGE_LW=0.5,                              # bar edge stroke, pt
    OURS_EDGE_LW=0.8,                         # heavier edge on the Ours bar
    HATCHES={                                 # print/grayscale channel
        "fast_gemm": "////", "nvshmem_gemm": "--", "moonep": "\\\\\\\\",
        "eplb": "..", "epic": "xx", "comet": None, "OURS": None,
    },
    HATCH_LW=0.4,                             # pt
    # ---- truncation (SPEC 2.3) ----
    OUTLIER_FACTOR=1.5,   # bar is an outlier if > factor * next system's max
    HEADROOM=1.12,        # ylim = ceil_nice(headroom * tallest kept bar)
    YLIM_OVERRIDE={"4": None, "16": None},    # absolute per-column cap
    BREAK_MARK=dict(dy_frac=0.035,            # slash rise, frac of ylim
                    gap_frac=0.030,           # gap between the two slashes
                    y_frac=0.90,              # slash center height, frac ylim
                    halfw=0.75,               # slash half-width, frac of bar w
                    lw=1.4, color="white"),
    TRUNC_LABEL_FMT="{:.0f}",                 # true value over truncated bars
    ANNOTATE_SPEEDUP=False,                   # "x.yx" over the Ours bar
    # ---- layout ----
    FIG_W=7.0, FIG_H=3.6,                     # inches (NSDI \textwidth)
    MARGINS=dict(left=0.058, right=0.972, top=0.83, bottom=0.075),
    WSPACE=0.14, HSPACE=0.16,
    COL_TITLES={"4": "4 nodes", "16": "16 nodes"},
    COL_TITLE_PAD=11,                         # pt above axes; clears trunc labels
    COL_TITLE_WEIGHT="bold",
    ROW_TAGS={"K2": "K2", "Qwen": "Qwen"},    # model tag text
    ROW_TAG_STYLE="right",                    # "right" rotated edge label,
                                              # or "inside" top-left in-axes
    GROUP_LABELS={1: "1 MiB", 4: "4 MiB", 16: "16 MiB"},
    X_LABEL=None,          # axis-level x title; None = budgets defined in caption
    Y_LABEL="Latency (ms)",   # single shared label on the figure's left edge
    N_YTICKS=4,
    # ---- typography ----
    FONT_FAMILY=["Helvetica", "Arial", "DejaVu Sans"],
    FONT_SIZES=dict(legend=8, col_title=8.5, row_tag=8, ylabel=8,
                    tick=7, group_label=7.5, annot=6.5),
    # ---- output ----
    OUTPUTS=[("main_perf.pdf", {}), ("main_perf.png", {"dpi": 300})],
)
# =============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))


def load(cfg):
    """-> data[(nodes, model, budget)][system] = total_ms, OURS = best-of."""
    raw = {}
    with open(os.path.join(HERE, "figure_src.csv"), newline="") as f:
        for r in csv.DictReader(f):
            raw[(r["nodes"], r["model"], int(r["budget_mib"]), r["row_id"])] = \
                float(r["total_ms"])
    data = {}
    for nodes in cfg["COLS"]:
        for model in cfg["ROWS"]:
            for b in cfg["BUDGETS"]:
                cell = {}
                for sysname in cfg["SYSTEMS"]:
                    if sysname == "OURS":
                        pool = [raw[(nodes, model, b, c)]
                                for c in cfg["OURS_CANDIDATES"]
                                if (nodes, model, b, c) in raw]
                        assert pool, f"no OURS candidate for {nodes}n {model} b{b}"
                        cell[sysname] = min(pool)
                    else:
                        key = (nodes, model, b, sysname)
                        assert key in raw, f"missing cell {key}"
                        cell[sysname] = raw[key]
                data[(nodes, model, b)] = cell
    return data


def ceil_nice(v):
    step = 5 if v < 100 else 10
    return step * math.ceil(v / step)


def column_ylim(data, cfg, nodes):
    """Peel outlier systems (SPEC 2.3), cap from the tallest kept bar."""
    if cfg["YLIM_OVERRIDE"].get(nodes):
        return cfg["YLIM_OVERRIDE"][nodes]
    sys_max = {s: max(data[(nodes, m, b)][s] for m in cfg["ROWS"]
                      for b in cfg["BUDGETS"]) for s in cfg["SYSTEMS"]}
    kept = sorted(sys_max.values(), reverse=True)
    while len(kept) > 1 and kept[0] > cfg["OUTLIER_FACTOR"] * kept[1]:
        kept.pop(0)
    return ceil_nice(cfg["HEADROOM"] * kept[0])


def draw_break(ax, x, bar_w, ylim, cfg):
    bm = cfg["BREAK_MARK"]
    y0 = bm["y_frac"] * ylim
    dy, gap, hw = bm["dy_frac"] * ylim, bm["gap_frac"] * ylim, bm["halfw"] * bar_w
    for off in (-gap / 2, gap / 2):
        ax.plot([x - hw, x + hw], [y0 + off - dy / 2, y0 + off + dy / 2],
                color=bm["color"], lw=bm["lw"], solid_capstyle="butt",
                zorder=4, clip_on=False)


def plot(data, cfg):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": cfg["FONT_FAMILY"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "hatch.linewidth": cfg["HATCH_LW"],
        "axes.linewidth": 0.6, "axes.edgecolor": cfg["INK"]["axis"],
        "xtick.color": cfg["INK"]["primary"], "ytick.color": cfg["INK"]["primary"],
        "text.color": cfg["INK"]["primary"],
    })
    fig, axes = plt.subplots(len(cfg["ROWS"]), len(cfg["COLS"]),
                             figsize=(cfg["FIG_W"], cfg["FIG_H"]))
    fig.subplots_adjust(wspace=cfg["WSPACE"], hspace=cfg["HSPACE"],
                        **cfg["MARGINS"])

    nbars = len(cfg["SYSTEMS"])
    slot = cfg["GROUP_WIDTH"] / nbars
    bar_w = slot * (1 - cfg["BAR_GAP_FRAC"])
    ylims = {nodes: column_ylim(data, cfg, nodes) for nodes in cfg["COLS"]}

    for ri, model in enumerate(cfg["ROWS"]):
        for ci, nodes in enumerate(cfg["COLS"]):
            ax = axes[ri][ci]
            ylim = ylims[nodes]
            for gi, b in enumerate(cfg["BUDGETS"]):
                cell = data[(nodes, model, b)]
                for si, s in enumerate(cfg["SYSTEMS"]):
                    x = gi + (si - (nbars - 1) / 2) * slot
                    v = cell[s]
                    trunc = v > ylim
                    ax.bar(x, min(v, ylim), width=bar_w,
                           facecolor=cfg["COLORS"][s], hatch=cfg["HATCHES"][s],
                           edgecolor=cfg["INK"]["primary"],
                           linewidth=cfg["OURS_EDGE_LW"] if s == "OURS"
                           else cfg["EDGE_LW"], zorder=3)
                    if trunc:
                        draw_break(ax, x, bar_w, ylim, cfg)
                        ax.text(x, ylim * 1.015, cfg["TRUNC_LABEL_FMT"].format(v),
                                ha="center", va="bottom",
                                fontsize=cfg["FONT_SIZES"]["annot"],
                                color=cfg["INK"]["muted"])
                if cfg["ANNOTATE_SPEEDUP"]:
                    base = min(cell[s] for s in cfg["SYSTEMS"] if s != "OURS")
                    x = gi + ((nbars - 1) - (nbars - 1) / 2) * slot
                    ax.text(x, cell["OURS"] + 0.02 * ylim,
                            f"{base / cell['OURS']:.1f}x", ha="center",
                            va="bottom", fontsize=cfg["FONT_SIZES"]["annot"],
                            color=cfg["INK"]["muted"])

            ax.set_ylim(0, ylim)
            ax.set_xlim(-0.5, len(cfg["BUDGETS"]) - 0.5)
            ax.yaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(nbins=cfg["N_YTICKS"]))
            ax.tick_params(axis="y", labelsize=cfg["FONT_SIZES"]["tick"],
                           length=2, width=0.6)
            ax.grid(axis="y", color=cfg["INK"]["grid"], lw=0.4, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if cfg["ROW_TAG_STYLE"] == "inside":
                ax.text(0.015, 0.965, cfg["ROW_TAGS"][model],
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=cfg["FONT_SIZES"]["row_tag"],
                        color=cfg["INK"]["secondary"])
            elif ci == len(cfg["COLS"]) - 1:  # "right": rotated row label
                ax.text(1.022, 0.5, cfg["ROW_TAGS"][model],
                        transform=ax.transAxes, ha="left", va="center",
                        rotation=270, fontsize=cfg["FONT_SIZES"]["row_tag"],
                        color=cfg["INK"]["secondary"])
            if ri == 0:
                ax.set_title(cfg["COL_TITLES"][nodes],
                             pad=cfg["COL_TITLE_PAD"],
                             fontsize=cfg["FONT_SIZES"]["col_title"],
                             fontweight=cfg["COL_TITLE_WEIGHT"])
            if ri == len(cfg["ROWS"]) - 1:
                ax.set_xticks(range(len(cfg["BUDGETS"])))
                ax.set_xticklabels([cfg["GROUP_LABELS"][b]
                                    for b in cfg["BUDGETS"]],
                                   fontsize=cfg["FONT_SIZES"]["group_label"])
                ax.tick_params(axis="x", length=0)
            else:
                ax.set_xticks([])

    fig.supylabel(cfg["Y_LABEL"], x=0.012, fontsize=cfg["FONT_SIZES"]["ylabel"])
    if cfg["X_LABEL"]:
        fig.supxlabel(cfg["X_LABEL"], y=0.005,
                      fontsize=cfg["FONT_SIZES"]["ylabel"])

    handles = [Patch(facecolor=cfg["COLORS"][s], hatch=cfg["HATCHES"][s],
                     edgecolor=cfg["INK"]["primary"], linewidth=cfg["EDGE_LW"],
                     label=cfg["LEGEND_NAMES"][s]) for s in cfg["SYSTEMS"]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.005),
               ncol=cfg["LEGEND_NCOL"], frameon=False,
               fontsize=cfg["FONT_SIZES"]["legend"],
               columnspacing=cfg["LEGEND_COLSPACING"], handlelength=1.2,
               handleheight=0.9, handletextpad=0.5)
    return fig


def main():
    fig = plot(load(CONFIG), CONFIG)
    for name, kw in CONFIG["OUTPUTS"]:
        fig.savefig(os.path.join(HERE, name), **kw)
        print("wrote", name)


if __name__ == "__main__":
    main()
