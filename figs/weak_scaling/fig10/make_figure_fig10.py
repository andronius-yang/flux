#!/usr/bin/env python3
"""Paper Figure 10: weak scaling, (a) A100 over (b) H100, each 1 MB over 64 MB (2026-10-07).

Reproduces the layout of Figure 10 in the 2026-09-18 paper build exactly (figure size, axes rectangles, shared
legend / labels, measured from the PDF's vector geometry) with the panels drawn by ../make_figure.py's
draw_panel (same styling, ylims and labels as the stacked A100 render).

  python make_figure_fig10.py              -> weak_scaling_fig10.{pdf,png}        (A100 = ../figure_src.csv)
  python make_figure_fig10.py --original   -> weak_scaling_fig10_original.{pdf,png} (A100 = the 09-18 data, for the
                                              exactness check against the paper figure)
H100 = h100/figure_src.csv (recovered from the paper PDF; see README.md).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import make_figure as mf  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Figure 10 geometry in PDF points (page coordinates, y down), measured from the 09-18 build
FIG = (317.88000, 71.99945, 558.00208, 266.69299)          # x0, y0, x1, y1 of the figure
AX_X = (355.01599, 533.48511)                               # left spine, right end of the x axis
PANELS = [  # (hardware, budget, top, bottom)
    ("A100", 1, 88.58446, 121.21365), ("A100", 64, 128.06398, 160.69318),
    ("H100", 1, 172.95166, 205.58086), ("H100", 64, 212.43118, 245.06038)]
GROUP_X, LEFT_LABEL_X, RIGHT_LABEL_X = 334.575, 324.250, 549.590   # anchors of the rotated labels (matched to the PDF)
LEGEND_Y = 1 - 0.97 / (266.69299 - 71.99945)                      # legend anchor (matched to the PDF)
TAG_FMT = "{} MB (= {} tok/GPU)"


def frac_x(x):
    return (x - FIG[0]) / (FIG[2] - FIG[0])


def frac_y(y):
    return 1 - (y - FIG[1]) / (FIG[3] - FIG[1])


def main():
    original = "--original" in sys.argv
    src = {"A100": os.path.join(HERE, "a100_pre_gpuplan") if original else os.path.dirname(HERE),
           "H100": os.path.join(HERE, "h100")}
    cfg = mf.configure("nvshmem")
    st = cfg["STACKED"]
    cfg = dict(cfg, N_YTICKS=st["N_YTICKS"], LEGEND=dict(cfg["LEGEND"], y=LEGEND_Y),
               BARS=dict(cfg["BARS"], label_pos=st["label_pos"]))
    mf.set_rc(cfg)
    fig = plt.figure(figsize=((FIG[2] - FIG[0]) / 72, (FIG[3] - FIG[1]) / 72))
    for k, (hw, b, top, bot) in enumerate(PANELS):
        mf.SRC_DIR = src[hw]
        c = dict(cfg, BUDGET=b)
        data = mf.load(c)
        ax = fig.add_axes([frac_x(AX_X[0]), frac_y(bot), frac_x(AX_X[1]) - frac_x(AX_X[0]),
                           frac_y(top) - frac_y(bot)])
        mf.draw_panel(fig, ax, ax.twinx(), data, c, st["metric"], None, x_label=(k == len(PANELS) - 1),
                      legend=(k == 0), tag=TAG_FMT.format(b, st["tok_per_gpu"][b]), right_label=False)
        if k < len(PANELS) - 1:
            ax.set_xticklabels([])
        else:
            ax.xaxis.labelpad = 1.7   # "Nodes" baseline as in the PDF
    for a in fig.axes:
        a.patch.set_visible(False)   # no panel backgrounds (the PDF figure has none)
    fs = cfg["FONT_SIZES"]
    mid = frac_y((PANELS[0][2] + PANELS[-1][3]) / 2)
    fig.text(frac_x(LEFT_LABEL_X), mid, cfg["Y_LABELS"]["A"], rotation=90, ha="center", va="center",
             fontsize=fs["label"], color=cfg["INK"]["primary"])
    fig.text(frac_x(RIGHT_LABEL_X), mid, cfg["RIGHT_LABEL"], rotation=90, ha="center", va="center",
             fontsize=fs["label"], color=cfg["INK"]["right"])
    for tag, (p0, p1) in (("(a) A100", (PANELS[0], PANELS[1])), ("(b) H100", (PANELS[2], PANELS[3]))):
        fig.text(frac_x(GROUP_X), frac_y((p0[2] + p1[3]) / 2), tag, rotation=90, ha="center", va="center",
                 fontsize=7, fontweight="bold", color=cfg["INK"]["primary"])
    stem = "weak_scaling_fig10_original" if original else "weak_scaling_fig10"
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 300})):
        fig.savefig(os.path.join(HERE, stem + ext), **kw)
        print("wrote", stem + ext)


if __name__ == "__main__":
    main()
