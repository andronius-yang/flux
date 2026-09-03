#!/usr/bin/env python3
"""Ablation figure generator (SPEC.md REV 0, 2026-09-03).

One single-column bar chart, two groups (S-A professional_law cell, S-C
dwell-4 schedule), a 4-bar feature chain per group, two versions:
  verA: COMET > w/ token-dispatch overlap > + placement & routing > + expert-dispatch overlap
  verB: COMET > w/ placement & routing > + token-dispatch overlap > + expert-dispatch overlap
Reads figure_src.csv only.  Every adjustable parameter is in KNOBS.
    python figs/ablation_cycling/make_figure.py            # both versions
    python figs/ablation_cycling/make_figure.py --version A
"""
import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# KNOBS — every adjustable parameter of the figure (SPEC.md §1–§5)
# ----------------------------------------------------------------------------
KNOBS = dict(
    FIG_W=3.33, FIG_H=1.90,                       # inches (single column, 1/4 text height incl. caption)
    MARGINS=dict(left=0.135, right=0.995, bottom=0.17, top=0.78),
    VERSIONS={                                    # chain per version: (tidy arm key, legend text)
        "A": [("COMET", "COMET"),
              ("slip", "w/ token-dispatch comp. overlap"),
              ("static", "+ expert placement & routing"),
              ("swap", "+ expert-dispatch overlap")],
        "B": [("COMET", "COMET"),
              ("pr0", "w/ expert placement & routing"),
              ("static", "+ token-dispatch comp. overlap"),
              ("swap", "+ expert-dispatch overlap")],
    },
    SWAP_ARM="full",                              # "full" = composed full-orbit swap, "one" = one-round swap
    INCLUDE_SEQ=False,                            # insert the sequential-swap twin before the last bar
    SEQ_TEXT="+ expert dispatch, un-overlapped",
    GROUPS=[                                      # (group label, figure_src panel prefix, topic, statistic)
        ("S-A: seen basis\n(prof. law, 4 reps)", "A:", "proLaw", "mean over 16 timed iters"),
        ("S-C: drift schedule\n(1/8 unseen, 3 reps)", "B:", "ALL", "whole-schedule mean, mean over reps"),
    ],
    ARM_ROWS={                                    # tidy_arm names in figure_src.csv, per panel prefix
        "COMET": {"A:": "COMET", "B:": "COMET"},
        "slip": {"A:": "1 token-comm overlap", "B:": "1 token-comm overlap"},
        "pr0": {"A:": "2 placement only (pr0)", "B:": "2 placement only (pr0)"},
        "static": {"A:": "1+2 static", "B:": "1+2 static"},
        "swap_full": {"A:": "1+2 full-orbit swap OVL d4", "B:": "1+2 full-orbit swap OVL"},
        "swap_one": {"A:": "1+2 one-round swap OVL d4", "B:": "1+2 one-round swap OVL"},
        "seq_full": {"A:": "1+2 full-orbit swap SEQ d4", "B:": "1+2 full-orbit swap SEQ"},
        "seq_one": {"A:": "1+2 one-round swap SEQ d4", "B:": "1+2 one-round swap SEQ"},
    },
    MIN_REPS=3,
    # bars
    BAR_WIDTH=1.0, GROUP_GAP=0.7, EDGE_PAD=0.45,  # in slot units
    COMET_COLOR="#999933", OURS_COLOR="#4878b0",
    STACK_TINTS=[0.40, 0.70, 1.00],               # tint of OURS_COLOR for bars 2, 3, 4 (seq bar reuses bar 4's tint)
    STACK_HATCH=["///", "\\\\\\", "xx"],
    SEQ_HATCH="..",
    EDGE_INK="#0b0b0b", EDGE_LW=0.45, HATCH_LW=0.4,
    # labels
    SPEEDUP_FMT="{:.2f}×", COMET_LABEL="1.00×", LABEL_PAD=0.25, LABEL_ROT=90,
    FLAG_BELOW_ONE=True, INK_PRIMARY="#0b0b0b", INK_SECONDARY="#52514e", INK_MUTED="#898781",
    # axes
    Y_LABEL="Total latency (ms)", Y_ZERO=True, Y_BREAK_AT=40.0, HEADROOM=1.30,
    COMET_REF=True, REF_COLOR="#898781", REF_LW=0.4, REF_DASH=(2, 1.5),
    GRID_COLOR="#e1e0d9", GRID_LW=0.4, AXIS_COLOR="#c3c2b7",
    # legend / fonts
    LEGEND=dict(ncol=2, columnspacing=0.9, handlelength=1.2, handleheight=0.9, handletextpad=0.35),  # 2+2 rows: 4-in-a-row oversets 3.33 in at 6 pt
    LEGEND_X=0.5,                                 # legend center in figure fraction (page-centered, not axes-centered)
    FONT_FAMILY=["Helvetica", "Arial", "DejaVu Sans"],
    FONT_SIZES=dict(legend=5.8, ylabel=6.5, ticks=6.0, group=6.0, bar=5.8),
    OUTPUTS={"A": [("ablation_verA.pdf", {}), ("ablation_verA.png", {"dpi": 300})],
             "B": [("ablation_verB.pdf", {}), ("ablation_verB.png", {"dpi": 300})]},
)


def tint(hex_color, t):
    """t=1 -> the color, t=0 -> white."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(255 - (255 - c) * t) for c in (r, g, b))


def ceil_nice(v):
    step = 10.0 if v > 50 else 5.0
    return math.ceil(v / step) * step


def load_src(cfg):
    rows = list(csv.DictReader(open(os.path.join(HERE, "figure_src.csv"))))
    data = {}
    for glabel, prefix, topic, stat in cfg["GROUPS"]:
        vals = {}
        for key, per_panel in cfg["ARM_ROWS"].items():
            arm = per_panel[prefix]
            hit = [r for r in rows if r["panel"].startswith(prefix) and r["topic"] == topic
                   and r["statistic"] == stat and r["tidy_arm"] == arm]
            if len(hit) != 1:
                raise SystemExit(f"figure_src.csv: need exactly 1 row for {prefix} {topic} {stat!r} {arm!r}, got {len(hit)}")
            if int(hit[0]["n"]) < cfg["MIN_REPS"]:
                raise SystemExit(f"{arm} in {prefix}{topic}: n={hit[0]['n']} < MIN_REPS {cfg['MIN_REPS']}")
            vals[key] = float(hit[0]["value_ms"])
        data[glabel] = vals
    return data


def chain_for(cfg, version):
    chain = list(cfg["VERSIONS"][version])
    swap_key = "swap_full" if cfg["SWAP_ARM"] == "full" else "swap_one"
    seq_key = "seq_full" if cfg["SWAP_ARM"] == "full" else "seq_one"
    out = []
    for key, text in chain:
        if key == "swap":
            if cfg["INCLUDE_SEQ"]:
                out.append((seq_key, cfg["SEQ_TEXT"], "seq"))
            out.append((swap_key, text, "swap"))
        else:
            out.append((key, text, key))
    return out


def style_for(cfg, idx, role, n_ours):
    """fill, hatch for bar index idx (0 = COMET) — cumulative tint along the chain."""
    if role == "COMET":
        return cfg["COMET_COLOR"], None
    if role == "seq":
        return tint(cfg["OURS_COLOR"], cfg["STACK_TINTS"][-1]), cfg["SEQ_HATCH"]
    # ours bars: map position among the (non-seq) ours bars onto the tint list
    return tint(cfg["OURS_COLOR"], cfg["STACK_TINTS"][min(idx - 1, len(cfg["STACK_TINTS"]) - 1)]), \
        cfg["STACK_HATCH"][min(idx - 1, len(cfg["STACK_HATCH"]) - 1)]


def render(cfg, version):
    fs = cfg["FONT_SIZES"]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": cfg["FONT_FAMILY"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "hatch.linewidth": cfg["HATCH_LW"],
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    })
    data = load_src(cfg)
    chain = chain_for(cfg, version)
    n = len(chain)
    fig = plt.figure(figsize=(cfg["FIG_W"], cfg["FIG_H"]))
    ax = fig.add_axes([cfg["MARGINS"]["left"], cfg["MARGINS"]["bottom"],
                       cfg["MARGINS"]["right"] - cfg["MARGINS"]["left"],
                       cfg["MARGINS"]["top"] - cfg["MARGINS"]["bottom"]])
    bw, gap, pad = cfg["BAR_WIDTH"], cfg["GROUP_GAP"], cfg["EDGE_PAD"]
    x0 = pad
    ymax = 0.0
    group_centers = []
    handles = None
    for gi, (glabel, _p, _t, _s) in enumerate(cfg["GROUPS"]):
        vals = data[glabel]
        comet = vals["COMET"]
        xs = [x0 + i * bw + bw / 2 for i in range(n)]
        hs = []
        for i, (key, text, role) in enumerate(chain):
            v = vals[key]
            fill, hatch = style_for(cfg, i, role, n - 1)
            b = ax.bar(xs[i], v, width=bw, color=fill, hatch=hatch, edgecolor=cfg["EDGE_INK"],
                       linewidth=cfg["EDGE_LW"], zorder=3, label=text if gi == 0 else None)
            hs.append(b)
            sp = comet / v
            label = cfg["COMET_LABEL"] if role == "COMET" else cfg["SPEEDUP_FMT"].format(sp)
            ink = cfg["INK_SECONDARY"] if (cfg["FLAG_BELOW_ONE"] and sp < 1.0) else cfg["INK_PRIMARY"]
            ax.text(xs[i], v + cfg["LABEL_PAD"], label, rotation=cfg["LABEL_ROT"], ha="center", va="bottom",
                    fontsize=fs["bar"], color=ink, zorder=5)
            ymax = max(ymax, v)
        if cfg["COMET_REF"]:
            ax.plot([x0 - 0.15 * bw, x0 + n * bw + 0.15 * bw], [comet, comet], color=cfg["REF_COLOR"],
                    lw=cfg["REF_LW"], dashes=cfg["REF_DASH"], zorder=4)
        group_centers.append((x0 + n * bw / 2, glabel))
        if handles is None:
            handles = [Patch(facecolor=style_for(cfg, i, role, n - 1)[0], hatch=style_for(cfg, i, role, n - 1)[1],
                             edgecolor=cfg["EDGE_INK"], linewidth=cfg["EDGE_LW"], label=text)
                       for i, (key, text, role) in enumerate(chain)]
        x0 += n * bw + gap
    xmax = x0 - gap + pad
    ax.set_xlim(0, xmax)
    ax.set_xticks([c for c, _ in group_centers])
    ax.set_xticklabels([l for _, l in group_centers], fontsize=fs["group"], color=cfg["INK_PRIMARY"])
    ax.tick_params(axis="x", length=0, pad=2)
    top = ceil_nice(ymax * cfg["HEADROOM"])
    if cfg["Y_ZERO"]:
        ax.set_ylim(0, top)
    else:
        ax.set_ylim(cfg["Y_BREAK_AT"], top)
        ax.plot([0.0], [cfg["Y_BREAK_AT"]], marker=(2, 0, 30), color=cfg["INK_PRIMARY"], ms=5,
                clip_on=False, zorder=6)   # break glyph on the axis line
    ax.set_ylabel(cfg["Y_LABEL"], fontsize=fs["ylabel"], color=cfg["INK_PRIMARY"], labelpad=2)
    ax.tick_params(axis="y", labelsize=fs["ticks"], colors=cfg["INK_PRIMARY"], length=2, pad=1.5)
    ax.yaxis.grid(True, color=cfg["GRID_COLOR"], lw=cfg["GRID_LW"], zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(cfg["AXIS_COLOR"])
    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(cfg["LEGEND_X"], 1.0),
                     ncol=cfg["LEGEND"]["ncol"], frameon=False, fontsize=fs["legend"],
                     columnspacing=cfg["LEGEND"]["columnspacing"], handlelength=cfg["LEGEND"]["handlelength"],
                     handleheight=cfg["LEGEND"]["handleheight"], handletextpad=cfg["LEGEND"]["handletextpad"],
                     borderaxespad=0.0)
    for t in leg.get_texts():
        t.set_color(cfg["INK_PRIMARY"])
    for name, kw in cfg["OUTPUTS"][version]:
        fig.savefig(os.path.join(HERE, name), metadata={"CreationDate": None, "Producer": None, "Creator": None}
                    if name.endswith(".pdf") else None, **kw)
    plt.close(fig)
    # console summary
    print(f"ver{version}:")
    for glabel, _p, _t, _s in cfg["GROUPS"]:
        vals = data[glabel]
        print("  " + glabel.replace("\n", " ") + " | " + " | ".join(
            f"{text}: {vals[key]:.2f} ({vals['COMET'] / vals[key]:.2f}x)" for key, text, role in chain))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["A", "B", "both"], default="both")
    ap.add_argument("--seq", action="store_true", help="include the sequential-swap twin as a 5th bar")
    ap.add_argument("--swap", choices=["full", "one"], default=None)
    args = ap.parse_args()
    cfg = dict(KNOBS)
    if args.seq:
        cfg["INCLUDE_SEQ"] = True
    if args.swap:
        cfg["SWAP_ARM"] = args.swap
    for v in (["A", "B"] if args.version == "both" else [args.version]):
        render(cfg, v)


if __name__ == "__main__":
    main()
