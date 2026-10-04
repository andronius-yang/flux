#!/usr/bin/env python3
"""SGLang serving figure generator (one USENIX column): prefill input tokens/s and decode TPOT, stock SGLang vs Ours, per-rank
budget on the x axis. Data from a CSV in this directory (DATA below); every aesthetic decision lives in CONFIG.

Usage:  python3 make_figure.py [dataset]  (30b4n | 235b16n | 30b16n_40g -> figure_src_<dataset>.csv and
                                          sglang_serving_<dataset>.pdf/.png; no argument = the draft data)
Render with a Python that has matplotlib (e.g. the andrewy-comet conda env); fonts follow main_perf_v5.
Groups whose rows are 'pending' in the CSV are drawn as an empty slot marked "pending" (draft only).
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
    DATA="data_draft.csv",
    NODES="4",
    BUDGETS=[1, 4, 16],                    # per-rank budget, labeled in MB (paper convention)
    PANELS=[                               # (csv panel, in-axes title, y label, higher is better, y scale)
        ("prefill", "Prefill", "TPS (k)", True, 1e-3),
        ("decode", "Decode", "TPOT (ms)", False, 1.0),
    ],
    GROUP_FMT="{b} MB",                    # group label under each pair
    # second label row: tokens per GPU per layer-step (prefill: chunk tokens; decode: running requests, one token
    # each); 30B hidden 2048 x 2 B -> 1 MB = 256 tokens. Numbers only under each group, the unit once at the row's left.
    TOKENS={1: 256, 4: 1024, 16: 4096},
    TOK_FMT="{tok}", TOK_ROW_LABEL="tok/GPU",
    SYSTEMS=["stock", "ours"],             # bar order inside a group
    LEGEND_NAMES={"stock": "SGLang", "ours": "Ours"},
    COLORS={"stock": "#cfccc2", "ours": "#4878b0"},     # Ours = main_perf_v5's Ours blue
    HATCHES={"stock": "////", "ours": None},
    INK=dict(primary="#0b0b0b", secondary="#52514e", muted="#898781",
             grid="#e1e0d9", axis="#c3c2b7"),
    EDGE_LW=0.45, OURS_EDGE_LW=0.6, HATCH_LW=0.4,
    BAR_W=0.40, BAR_GAP=0.03,              # in group-slot units (a slot is 1.0): wide bars, little air
    XPAD=0.12,                             # extra x room outside the outer bars, slot units
    # speedup on top of Ours' bar, vertical (main_perf_v5 style); the y limit grows so every label fits
    SPEEDUP=dict(fmt="{:.2f}\u00d7", color="#c1121f", weight="bold", pad_pt=1.2, char_w=0.62, top_pad_pt=0.5),
    N_YTICKS=4,                            # at most, 0 included
    Y_HEAD=0.06,                           # y limit = tallest bar x (1 + Y_HEAD)
    # ---- layout, inches at final size (the PDF is placed at \columnwidth = 3.33 in; ~20 % of the 9 in column) ----
    FIG_W=3.33, FIG_H=1.75,
    TOP_IN=0.17,                           # legend strip
    BOTTOM_IN=0.22,                        # group labels + token row
    LEFT_IN=0.37, RIGHT_IN=0.02, GAP_IN=0.50,   # GAP = between the panels (holds the right panel's y labels)
    FONT_FAMILY=["Helvetica", "Arial", "DejaVu Sans"],
    FONT_SIZES=dict(legend=7, title=7, ylabel=6.8, tick=6.3, group=6.8, tok=5.8, speedup=5.8, pending=5.6),
    GROUP_DY_PT=2.0,                       # group label row: points below the x axis
    TOK_DY_PT=9.0,                         # token row: points below the x axis
    OUTPUTS=[("sglang_serving_draft.pdf", {}), ("sglang_serving_draft.png", {"dpi": 300})],
)
# =============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))


def load(cfg):
    data = {}
    with open(os.path.join(HERE, cfg["DATA"]), newline="") as f:
        for r in csv.DictReader(f):
            if r["nodes"] != cfg["NODES"]:
                continue
            v = float(r["value"]) if r["status"] == "measured" and r["value"] else None
            data[(r["panel"], int(r["budget_mib"]), r["system"])] = (v, int(r["per_gpu"]))
            if r["status"] == "infeasible":
                data[("infeasible", r["panel"], int(r["budget_mib"]))] = True
    return data


def nice_ticks(lim, max_ticks):
    """The smallest round step that puts at most max_ticks ticks (0 included) inside [0, lim]."""
    mag = 10 ** math.floor(math.log10(lim))
    for m in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10):
        if int(lim // (m * mag)) + 1 <= max_ticks:
            return m * mag, lim
    return mag * 10, lim


def main(cfg=CONFIG):
    ink, fs = cfg["INK"], cfg["FONT_SIZES"]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": cfg["FONT_FAMILY"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": cfg["HATCH_LW"],
        "axes.linewidth": 0.6, "axes.edgecolor": ink["axis"],
    })
    data = load(cfg)
    W, H = cfg["FIG_W"], cfg["FIG_H"]
    npan = len(cfg["PANELS"])
    pw = (W - cfg["LEFT_IN"] - cfg["RIGHT_IN"] - cfg["GAP_IN"] * (npan - 1)) / npan
    ph = H - cfg["TOP_IN"] - cfg["BOTTOM_IN"]
    fig = plt.figure(figsize=(W, H))
    bw, gap = cfg["BAR_W"], cfg["BAR_GAP"]
    offs = {"stock": -(bw + gap) / 2, "ours": (bw + gap) / 2}
    nb = len(cfg["BUDGETS"])
    for pi, (panel, title, ylabel, higher_better, yscale) in enumerate(cfg["PANELS"]):
        x0 = cfg["LEFT_IN"] + pi * (pw + cfg["GAP_IN"])
        ax = fig.add_axes([x0 / W, cfg["BOTTOM_IN"] / H, pw / W, ph / H])
        vals = [data.get((panel, b, s), (None, 0))[0] for b in cfg["BUDGETS"] for s in cfg["SYSTEMS"]]
        vals = [v * yscale for v in vals if v is not None]
        # y limit: every bar + Y_HEAD, and every speedup label on top of its Ours bar (label length L pt on an axis
        # ph*72 pt tall needs top >= y / (1 - L / (ph*72)))
        sp, lim = cfg["SPEEDUP"], max(vals) * (1 + cfg["Y_HEAD"])
        for b in cfg["BUDGETS"]:
            st_, ou_ = data.get((panel, b, "stock"), (None, 0))[0], data.get((panel, b, "ours"), (None, 0))[0]
            if st_ is not None and ou_ is not None:
                txt = sp["fmt"].format(ou_ / st_ if higher_better else st_ / ou_)
                lab = len(txt) * fs["speedup"] * sp["char_w"] + 2 * sp["pad_pt"] + sp["top_pad_pt"]
                lim = max(lim, ou_ * yscale / (1 - lab / (ph * 72)))
        step, top = nice_ticks(lim, cfg["N_YTICKS"])
        ax.set_ylim(0, top)
        ax.set_yticks([step * k for k in range(int(top // step) + 1)])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        edge = (bw + gap) / 2 + bw / 2 + cfg["XPAD"]
        ax.set_xlim(-edge, nb - 1 + edge)
        ax.yaxis.grid(True, color=ink["grid"], lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", labelsize=fs["tick"], length=1.5, width=0.6, pad=1.2, colors=ink["secondary"])
        ax.tick_params(axis="x", length=0, labelbottom=False)
        ax.set_ylabel(ylabel, fontsize=fs["ylabel"], labelpad=1.5, color=ink["primary"])
        ax.text(0.03, 0.97, title, transform=ax.transAxes, ha="left", va="top", fontsize=fs["title"],
                fontweight="bold", color=ink["primary"])
        below = lambda pt: matplotlib.transforms.offset_copy(ax.get_xaxis_transform(), fig=fig, y=-pt, units="points")
        for gi, b in enumerate(cfg["BUDGETS"]):
            stock = data.get((panel, b, "stock"), (None, 0))[0]
            ours = data.get((panel, b, "ours"), (None, 0))[0]
            if stock is None or ours is None:
                ax.text(gi, top * 0.04, "does not fit" if data.get(("infeasible", panel, b)) else "pending",
                        ha="center", va="bottom", fontsize=fs["pending"], style="italic", color=ink["muted"])
            else:
                for sname, v in (("stock", stock * yscale), ("ours", ours * yscale)):
                    ax.bar(gi + offs[sname], v, width=bw, facecolor=cfg["COLORS"][sname], hatch=cfg["HATCHES"][sname],
                           edgecolor=ink["primary"],
                           linewidth=cfg["OURS_EDGE_LW"] if sname == "ours" else cfg["EDGE_LW"], zorder=3)
                sp = cfg["SPEEDUP"]
                text = sp["fmt"].format(ours / stock if higher_better else stock / ours)
                dpp = top / (ph * 72)                       # data units per point (y)
                y = ours * yscale
                ty, va, bbox = y + sp["pad_pt"] * dpp, "bottom", None   # always on top (the y limit makes room)
                ax.text(gi + offs["ours"], ty, text, rotation=90, ha="center", va=va, fontsize=fs["speedup"],
                        fontweight=sp["weight"], color=sp["color"], zorder=5, bbox=bbox)
            ax.text(gi, 0, cfg["GROUP_FMT"].format(b=b), transform=below(cfg["GROUP_DY_PT"]), ha="center", va="top",
                    fontsize=fs["group"], color=ink["primary"])
            if cfg.get("TOKENS"):
                ax.text(gi, 0, cfg["TOK_FMT"].format(tok=cfg["TOKENS"][b]), transform=below(cfg["TOK_DY_PT"]),
                        ha="center", va="top", fontsize=fs["tok"], color=ink["secondary"])
        if cfg.get("TOKENS"):
            row = matplotlib.transforms.offset_copy(ax.transAxes, fig=fig, x=-1.5, y=-cfg["TOK_DY_PT"], units="points")
            ax.text(0, 0, cfg["TOK_ROW_LABEL"], transform=row, ha="right", va="top", fontsize=fs["tok"],
                    color=ink["secondary"])
    handles = [Patch(facecolor=cfg["COLORS"][s], hatch=cfg["HATCHES"][s], edgecolor=ink["primary"],
                     linewidth=cfg["OURS_EDGE_LW"] if s == "ours" else cfg["EDGE_LW"], label=cfg["LEGEND_NAMES"][s])
               for s in cfg["SYSTEMS"]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0 - 0.5 / 72 / H), ncol=len(handles),
               frameon=False, fontsize=fs["legend"], handlelength=1.3, handleheight=0.75, handletextpad=0.4,
               columnspacing=1.2, borderaxespad=0.0, borderpad=0.0)
    for name, kw in cfg["OUTPUTS"]:
        fig.savefig(os.path.join(HERE, name), **kw)


DATASETS = {"30b4n": "4", "235b16n": "16", "30b16n_40g": "16", "30b16n_80g": "16"}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ds = sys.argv[1]
        CONFIG.update(DATA=f"figure_src_{ds}.csv", NODES=DATASETS[ds],
                      OUTPUTS=[(f"sglang_serving_{ds}.pdf", {}), (f"sglang_serving_{ds}.png", {"dpi": 300})])
    main()
