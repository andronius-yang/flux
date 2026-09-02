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
    LEGEND_Y=1.01,                            # anchor (figure fraction)
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
    GROUP_WIDTH=0.90,                         # fraction of the unit group slot
    BAR_GAP_FRAC=0.10,                        # gap between bars, frac of bar w
    X_PAD=0.47,                               # half-slot padding at both ends
    EDGE_LW=0.45,                             # bar edge stroke, pt
    OURS_EDGE_LW=0.7,                         # heavier edge on the Ours bar
    HATCHES={                                 # print/grayscale channel
        "fast_gemm": "////", "nvshmem_gemm": "--", "moonep": "\\\\\\\\",
        "eplb": "..", "epic": "xx", "comet": None, "OURS": None,
    },
    HATCH_LW=0.4,                             # pt
    # ---- ceiling + truncation (SPEC 2.3) ----
    OUTLIER_FACTOR=1.5,   # bar is an outlier if > factor * next system's max
    HEADROOM=1.03,        # ylim = ceil_nice(headroom * tallest kept bar)
    NICE_STEPS=[(60, 2), (150, 5), (1e9, 10)],  # ceil_nice: step below bound
    YLIM_OVERRIDE={"4": None, "16": None},    # absolute per-column cap
    BREAK_MARK=dict(dy_frac=0.045,            # slash rise, frac of ylim
                    gap_frac=0.040,           # gap between the two slashes
                    y_frac=0.90,              # slash center height, frac ylim
                    halfw=0.52,               # slash half-width, frac of bar w
                                              # (<0.55 keeps it inside the slot)
                    lw=1.2, color="white"),
    TRUNC_LABEL_FMT="{:.0f}",                 # true value, inside truncated bar
    # ---- speedup annotations (SPEC 2.4) ----
    SPEEDUP=dict(
        on=True,
        ref="nvshmem_gemm",                   # ratio = ref_total / bar_total
        fmt="{:.2f}×",                   # two decimals + multiplication sign
        ref_label="1×",                  # mark on the reference bar itself
        weight="normal",                      # baselines: regular weight ...
        ours_weight="bold",                   # ... Ours: bold
        color="#0b0b0b",                      # all systems ...
        ours_color="#c1121f",                 # ... except Ours, highlighted red
        placement="auto",                     # auto | inside | above
        pad_pt=1.2,                           # gap between bar end and text
        char_w=0.62,                          # est. glyph width, em (bold sans)
        bbox_mode="bar",     # inside-label backing: "bar" = solid window in the
                             # bar's own color (text auto white/black), "white"
        bbox_pad=0.12,       # backing padding, em
        dark_text_L=0.40,    # OKLCH L below this -> white text on the window
                             # (black otherwise; none of the current fills
                             # are that dark, so all inside labels are black)
        skip_ref=False,                       # reference bar carries ref_label
        skip_truncated=True,                  # truncated bars show value instead
    ),
    # ---- layout ----
    FIG_W=7.0, FIG_H=1.85,     # in; USENIX text block 7x9 -> <=25% incl. caption
    MARGINS=dict(left=0.052, right=0.968, top=0.80, bottom=0.115),
    WSPACE=0.10, HSPACE=0.14,
    COL_TITLES={"4": "4 nodes", "16": "16 nodes"},
    COL_TITLE_PAD=2.5,                        # pt above axes
    COL_TITLE_WEIGHT="bold",
    ROW_TAGS={"K2": "K2", "Qwen": "Qwen"},    # model tag text
    ROW_TAG_STYLE="right",                    # "right" rotated edge label,
                                              # or "inside" top-left in-axes
    GROUP_LABELS={1: "1 MiB", 4: "4 MiB", 16: "16 MiB"},
    X_LABEL=None,          # axis-level x title; None = budgets defined in caption
    Y_LABEL="Latency (ms)",   # single shared label on the figure's left edge
    N_YTICKS=3,
    # ---- typography ----
    FONT_FAMILY=["Helvetica", "Arial", "DejaVu Sans"],
    FONT_SIZES=dict(legend=7, col_title=7.5, row_tag=7, ylabel=7,
                    tick=6.5, group_label=7, annot=5.8),
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


def ceil_nice(v, cfg):
    step = next(s for bound, s in cfg["NICE_STEPS"] if v < bound)
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
    return ceil_nice(cfg["HEADROOM"] * kept[0], cfg)


def draw_break(ax, x, bar_w, ylim, cfg):
    bm = cfg["BREAK_MARK"]
    y0 = bm["y_frac"] * ylim
    dy, gap, hw = bm["dy_frac"] * ylim, bm["gap_frac"] * ylim, bm["halfw"] * bar_w
    for off in (-gap / 2, gap / 2):
        ax.plot([x - hw, x + hw], [y0 + off - dy / 2, y0 + off + dy / 2],
                color=bm["color"], lw=bm["lw"], solid_capstyle="butt",
                zorder=4, clip_on=True)
    return y0 - gap / 2 - dy / 2       # lowest point of the glyph (data units)


def oklch_L(hexcolor):
    """OKLCH lightness of an sRGB hex (for the inside-label text color)."""
    r, g, b = (int(hexcolor.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
           for c in (r, g, b)]
    l = (0.4122214708 * lin[0] + 0.5363325363 * lin[1] + 0.0514459929 * lin[2]) ** (1 / 3)
    m = (0.2119034982 * lin[0] + 0.6806995451 * lin[1] + 0.1073969566 * lin[2]) ** (1 / 3)
    s_ = (0.0883024619 * lin[0] + 0.2817188376 * lin[1] + 0.6299787005 * lin[2]) ** (1 / 3)
    return 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s_


def vlabel(ax, x, y_anchor, text, cfg, *, color, weight, where, fontsize,
           bar_color=None):
    """Vertical (bottom-to-top) label: where='above' starts at y_anchor and
    runs upward; where='inside' ends at y_anchor and runs downward, on a
    solid backing window so hatches never cross the glyphs."""
    sp = cfg["SPEEDUP"]
    pad = sp["pad_pt"] * ax._data_per_pt
    if where == "above":
        y, va, bbox = y_anchor + pad, "bottom", None
    else:
        y, va = y_anchor - pad, "top"
        if sp["bbox_mode"] == "bar" and bar_color and color != sp["ours_color"]:
            # window in the bar's own color; text flips to white on dark bars
            bbox = dict(boxstyle=f"square,pad={sp['bbox_pad']}",
                        facecolor=bar_color, edgecolor="none")
            if oklch_L(bar_color) < sp["dark_text_L"]:
                color = "white"
        else:   # "white" mode, or a colored (Ours/muted) label: white backing
            bbox = dict(boxstyle=f"square,pad={sp['bbox_pad']}",
                        facecolor="white", edgecolor="none", alpha=0.9)
    ax.text(x, y, text, rotation=90, ha="center", va=va, fontsize=fontsize,
            fontweight=weight, color=color, zorder=5, bbox=bbox)


def label_len_data(ax, text, cfg, fontsize):
    """Estimated rotated-label length in data units (for placement)."""
    sp = cfg["SPEEDUP"]
    pts = len(text) * fontsize * sp["char_w"] + 2 * sp["pad_pt"]
    return pts * ax._data_per_pt


def place_speedup(ax, x, top, text, cfg, ylim, color, bar_color, weight):
    """auto: above the bar when it fits under the ceiling, else inside."""
    sp = cfg["SPEEDUP"]
    fs = cfg["FONT_SIZES"]["annot"]
    need = label_len_data(ax, text, cfg, fs)
    mode = sp["placement"]
    if mode == "auto":
        mode = "above" if top + need <= ylim else "inside"
    vlabel(ax, x, top, text, cfg, color=color, weight=weight,
           where=mode, fontsize=fs, bar_color=bar_color)


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
    sp = cfg["SPEEDUP"]

    for ri, model in enumerate(cfg["ROWS"]):
        for ci, nodes in enumerate(cfg["COLS"]):
            ax = axes[ri][ci]
            ylim = ylims[nodes]
            ax_h_pt = cfg["FIG_H"] * ax.get_position().height * 72
            ax._data_per_pt = ylim / ax_h_pt   # data units per point (y)
            for gi, b in enumerate(cfg["BUDGETS"]):
                cell = data[(nodes, model, b)]
                ref = cell[sp["ref"]]
                for si, s in enumerate(cfg["SYSTEMS"]):
                    x = gi + (si - (nbars - 1) / 2) * slot
                    v = cell[s]
                    trunc = v > ylim
                    top = min(v, ylim)
                    ax.bar(x, top, width=bar_w,
                           facecolor=cfg["COLORS"][s], hatch=cfg["HATCHES"][s],
                           edgecolor=cfg["INK"]["primary"],
                           linewidth=cfg["OURS_EDGE_LW"] if s == "OURS"
                           else cfg["EDGE_LW"], zorder=3)
                    if trunc:
                        glyph_lo = draw_break(ax, x, bar_w, ylim, cfg)
                        vlabel(ax, x, glyph_lo, cfg["TRUNC_LABEL_FMT"].format(v),
                               cfg, color=sp["color"], weight="normal",
                               where="inside", fontsize=cfg["FONT_SIZES"]["annot"],
                               bar_color=cfg["COLORS"][s])
                        if sp["skip_truncated"]:
                            continue
                    if not sp["on"] or (sp["skip_ref"] and s == sp["ref"]):
                        continue
                    text = (sp["ref_label"] if s == sp["ref"]
                            else sp["fmt"].format(ref / v))
                    place_speedup(ax, x, top, text, cfg, ylim,
                                  sp["ours_color"] if s == "OURS" else sp["color"],
                                  cfg["COLORS"][s],
                                  sp["ours_weight"] if s == "OURS" else sp["weight"])

            ax.set_ylim(0, ylim)
            ax.set_xlim(-cfg["X_PAD"], len(cfg["BUDGETS"]) - 1 + cfg["X_PAD"])
            ax.yaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(nbins=cfg["N_YTICKS"]))
            ax.tick_params(axis="y", labelsize=cfg["FONT_SIZES"]["tick"],
                           length=1.5, width=0.6, pad=1.5)
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
                ax.text(1.018, 0.5, cfg["ROW_TAGS"][model],
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
                ax.tick_params(axis="x", length=0, pad=1.5)
            else:
                ax.set_xticks([])

    fig.supylabel(cfg["Y_LABEL"], x=0.008, fontsize=cfg["FONT_SIZES"]["ylabel"])
    if cfg["X_LABEL"]:
        fig.supxlabel(cfg["X_LABEL"], y=0.005,
                      fontsize=cfg["FONT_SIZES"]["ylabel"])

    handles = [Patch(facecolor=cfg["COLORS"][s], hatch=cfg["HATCHES"][s],
                     edgecolor=cfg["INK"]["primary"], linewidth=cfg["EDGE_LW"],
                     label=cfg["LEGEND_NAMES"][s]) for s in cfg["SYSTEMS"]]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, cfg["LEGEND_Y"]),
               ncol=cfg["LEGEND_NCOL"], frameon=False,
               fontsize=cfg["FONT_SIZES"]["legend"],
               columnspacing=cfg["LEGEND_COLSPACING"], handlelength=1.2,
               handleheight=0.9, handletextpad=0.5, borderaxespad=0.0)
    return fig


def main():
    fig = plot(load(CONFIG), CONFIG)
    for name, kw in CONFIG["OUTPUTS"]:
        fig.savefig(os.path.join(HERE, name), **kw)
        print("wrote", name)


if __name__ == "__main__":
    main()
