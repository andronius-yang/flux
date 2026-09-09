#!/usr/bin/env python3
"""Intro figure v4: three stacked rows, single-column NSDI, final physical size.

  (a) expert activation frequency   — per-expert routed tokens / uniform, sorted (LiveCodeBench)
  (b) per-GPU compute load          — GEMM rows landing on each GPU / uniform, contiguous placement, both topics
  (c) NIC-to-NIC dispatch traffic   — two 16x16 maps (one per topic), same-node blocks zeroed, shared scale

Narrative: routing skew -> GPU compute imbalance -> NIC traffic imbalance. Colors follow the
later figures: compute = amber family, token communication = blue family; (a) is neutral ink.
Every aesthetic value lives in CONFIG.  `python make_figure.py [--logx] [--suffix _x]`
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
    OUT_STEM=os.path.join(HERE, "intro_v4"),
    # --- geometry, inches (USENIX column 3.33 in). Height is DERIVED from the stack below. ---
    FIG_W=3.33, LEFT_IN=0.50, RIGHT_IN=0.45,     # right margin holds the colorbar + its label
    TOP_IN=0.05, BOT_IN=0.06,
    ROW_A_IN=0.52, ROW_B_IN=0.38,                # bar panel heights
    GAP_AB_IN=0.36, GAP_BC_IN=0.56,   # (b) sub-label must clear the map titles              # x label + sub-label (+ map titles) between rows
    BELOW_C_IN=0.42,                             # tick labels + "Receiver NIC" + sub-label under the maps
    HM_GAP_IN=0.14, CBAR_W_IN=0.07, CBAR_GAP_IN=0.05,
    SUBLABEL_IN=dict(a=0.20, b=0.30, c=0.26),   # (b) has tick labels under it, so its sub-label sits lower    # sub-label distance below each row's axes
    HM_XLABEL_IN=0.14,                           # "Receiver NIC" below the maps (tick labels above it)
    # --- topics ---
    TOPICS=["livecodebench/execution", "mmlu/professional_law"],
    TOPIC_NAMES={"livecodebench/execution": "LiveCodeBench", "mmlu/professional_law": "MMLU prof. law"},
    A_TOPICS=["livecodebench/execution"],        # (a) shows one topic (prof. law's 33x outlier hides the shape)
    B_TOPICS=["livecodebench/execution", "mmlu/professional_law"],   # (b) both, grouped bars
    # --- (a) routing skew: neutral ink (no resource yet) ---
    A_COLOR="#4b5563", A_ALPHA=1.0, A_XLABEL="Expert ID", A_YLABEL="Normalized\ntoken count",
    X_LOG=False, X_LOG_MIN=0.05,
    # --- (b) compute: amber family = "Expert Comp." in the later figures (#eda100) ---
    B_COLORS=["#eda100", "#a86f00"], B_YLABEL="Normalized\ncompute", B_XLABEL="GPU",
    B_GROUP_W=0.78, B_YMAX=3.0,   # headroom for the legend above the 2.21x bar (both topics stay)                 # None = next 0.5 above the data max
    # --- (c) NIC traffic: blue family = "Token Comm." (#2a78d6) ---
    CMAP="Blues", VMIN=0.0, VMAX=None, NIC_ONLY=True,
    HM_XLABEL="Receiver NIC", HM_YLABEL="Sender NIC", CBAR_LABEL="Normalized traffic",
    HM_MAJOR=[0, 4, 8, 12],                      # labelled; minor ticks on every row/column
    HM_EDGE_LW=0.5, NODE_LINE=dict(color="#0b0b0b", lw=0.6),
    NODE_SEP=dict(color="#9ca3af", lw=0.5, ls=(0, (1.5, 1.5))),   # node separators in (b)
    UNIFORM_LINE=dict(color="#0b0b0b", lw=0.5, ls=(0, (2, 1.5))),
    PANEL_LABELS=["(a) Expert activation frequency", "(b) Per-GPU compute load",
                  "(c) NIC-to-NIC dispatch traffic"],
    FONT_FAMILY=["Helvetica", "Arial", "DejaVu Sans"],
    FS=dict(label=7, tick=6, legend=6.5, title=7, panel=6.5, cbar=6),
    INK="#0b0b0b", INK2="#52514e", DPI=300,
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
    L = int(next(p.split("L=")[1].split()[0] for p in prov if "L=" in p))
    E = {t: np.array([experts[t][e] for e in range(G)]) for t in experts}
    M = {t: np.array([[cells[t][(s, d)] for d in range(W)] for s in range(W)]) for t in cells}
    # (b): GEMM rows per GPU under contiguous placement (expert e -> GPU e // (G//W)), / uniform
    epr = G // W
    C = {t: np.array([E[t][r * epr:(r + 1) * epr].mean() for r in range(W)]) for t in E}
    return E, C, M, G, W, L, prov

def main():
    cfg = dict(CONFIG)
    args = sys.argv[1:]
    if "--logx" in args:
        cfg["X_LOG"] = True
    if "--suffix" in args:
        cfg["OUT_STEM"] += args[args.index("--suffix") + 1]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": cfg["FONT_FAMILY"],
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4,
                         "ytick.minor.width": 0.4, "xtick.major.size": 2, "ytick.major.size": 2,
                         "xtick.minor.size": 1.1, "ytick.minor.size": 1.1})
    E, C, M, G, W, L, prov = load(cfg)
    fs, ink, ink2 = cfg["FS"], cfg["INK"], cfg["INK2"]

    # ---- geometry in inches, top-down; figure height derived ----
    fw = cfg["FIG_W"]
    plot_w = fw - cfg["LEFT_IN"] - cfg["RIGHT_IN"]
    side = (plot_w - cfg["HM_GAP_IN"]) / 2                       # square maps fill the plot width
    fh = (cfg["TOP_IN"] + cfg["ROW_A_IN"] + cfg["GAP_AB_IN"] + cfg["ROW_B_IN"] + cfg["GAP_BC_IN"]
          + side + cfg["BELOW_C_IN"] + cfg["BOT_IN"])
    fig = plt.figure(figsize=(fw, fh))
    X0, PW = cfg["LEFT_IN"] / fw, plot_w / fw
    def ax_at(top_in, h_in, x0=X0, w=PW):
        return fig.add_axes([x0, 1 - (top_in + h_in) / fh, w, h_in / fh])
    y = cfg["TOP_IN"]
    ax_a = ax_at(y, cfg["ROW_A_IN"]); y += cfg["ROW_A_IN"] + cfg["GAP_AB_IN"]
    ax_b = ax_at(y, cfg["ROW_B_IN"]); y += cfg["ROW_B_IN"] + cfg["GAP_BC_IN"]
    ax_c = [ax_at(y, side, X0, side / fw), ax_at(y, side, X0 + (side + cfg["HM_GAP_IN"]) / fw, side / fw)]
    ax_cb = ax_at(y, side, X0 + (2 * side + cfg["HM_GAP_IN"] + cfg["CBAR_GAP_IN"]) / fw, cfg["CBAR_W_IN"] / fw)

    # ---- (a) expert activation frequency, sorted ----
    ids = np.arange(G)
    ymax = float(np.ceil(max(E[t].max() for t in cfg["A_TOPICS"]) / 5) * 5)
    for t in cfg["A_TOPICS"]:
        ax_a.bar(ids, np.sort(E[t])[::-1], width=1.0, color=cfg["A_COLOR"], alpha=cfg["A_ALPHA"],
                 linewidth=0, label=cfg["TOPIC_NAMES"][t], align="edge")
    ax_a.axhline(1.0, **cfg["UNIFORM_LINE"], zorder=3)
    ax_a.set_xlim(0, G); ax_a.set_xticks([])
    if cfg["X_LOG"]:
        ax_a.set_yscale("log"); ax_a.set_ylim(cfg["X_LOG_MIN"], ymax)
        ax_a.set_yticks([0.1, 1, 10]); ax_a.set_yticklabels(["0.1", "1", "10"])
        ax_a.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=(2, 5), numticks=20))
        ax_a.text(G * 0.99, 0.93, "uniform", ha="right", va="top", fontsize=fs["tick"], color=ink2)
    else:
        ax_a.set_ylim(0, ymax); ax_a.set_yticks([t for t in (0, 5, 10) if t <= ymax])
        ax_a.text(G * 0.99, 1.0, "uniform", ha="right", va="bottom", fontsize=fs["tick"], color=ink2)
    ax_a.set_xlabel(cfg["A_XLABEL"], fontsize=fs["label"], labelpad=2)
    ax_a.set_ylabel(cfg["A_YLABEL"], fontsize=fs["label"], labelpad=2)
    ax_a.tick_params(labelsize=fs["tick"], pad=1.5)
    ax_a.legend(fontsize=fs["legend"], frameon=False, loc="upper right", handlelength=1.0,
                handletextpad=0.5, borderaxespad=0.2)

    # ---- (b) per-GPU compute load, grouped bars, node separators ----
    nb = len(cfg["B_TOPICS"]); bw = cfg["B_GROUP_W"] / nb; xs = np.arange(W)
    bmax = cfg["B_YMAX"] or float(np.ceil(max(C[t].max() for t in cfg["B_TOPICS"]) / 0.5) * 0.5)
    for i, t in enumerate(cfg["B_TOPICS"]):
        ax_b.bar(xs - cfg["B_GROUP_W"] / 2 + (i + 0.5) * bw, C[t], width=bw, color=cfg["B_COLORS"][i],
                 linewidth=0, label=cfg["TOPIC_NAMES"][t])
    for n in range(1, W // L):
        ax_b.axvline(n * L - 0.5, **cfg["NODE_SEP"], zorder=0)
    ax_b.axhline(1.0, **cfg["UNIFORM_LINE"], zorder=3)
    ax_b.set_xlim(-0.6, W - 0.4); ax_b.set_ylim(0, bmax)
    ax_b.set_xticks(cfg["HM_MAJOR"]); ax_b.set_xticks(range(W), minor=True)
    ax_b.set_yticks([v for v in np.arange(0, bmax + 1e-9, 1.0)])
    ax_b.set_xlabel(cfg["B_XLABEL"], fontsize=fs["label"], labelpad=2)
    ax_b.set_ylabel(cfg["B_YLABEL"], fontsize=fs["label"], labelpad=2)
    ax_b.tick_params(labelsize=fs["tick"], pad=1.5)
    ax_b.legend(fontsize=fs["legend"], frameon=False, loc="upper right", ncol=2, handlelength=1.0,
                handletextpad=0.5, borderaxespad=0.2, columnspacing=0.8)
    for ax in (ax_a, ax_b):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # ---- (c) NIC-to-NIC maps ----
    if cfg["NIC_ONLY"]:
        nodes = np.arange(W) // L
        off = nodes[:, None] != nodes[None, :]
        M = {t: np.where(off, m / m[off].mean(), 0.0) for t, m in M.items()}
    cmap = (matplotlib.colormaps[cfg["CMAP"]] if isinstance(cfg["CMAP"], str)
            else LinearSegmentedColormap.from_list("custom", cfg["CMAP"]))
    vmax = cfg["VMAX"] or max(float(m.max()) for m in M.values())
    for ax, t in zip(ax_c, cfg["TOPICS"]):
        im = ax.imshow(M[t], cmap=cmap, vmin=cfg["VMIN"], vmax=vmax, origin="upper",
                       interpolation="nearest", aspect="equal")
        for n in range(1, W // L):
            ax.axhline(n * L - .5, **cfg["NODE_LINE"]); ax.axvline(n * L - .5, **cfg["NODE_LINE"])
        for sp in ax.spines.values():
            sp.set_linewidth(cfg["HM_EDGE_LW"])
        ax.set_title(cfg["TOPIC_NAMES"][t], fontsize=fs["title"], pad=2, color=ink)
        ax.set_xticks(cfg["HM_MAJOR"]); ax.set_yticks(cfg["HM_MAJOR"])
        ax.set_xticks(range(W), minor=True); ax.set_yticks(range(W), minor=True)
        ax.tick_params(labelsize=fs["tick"], pad=1.5, length=1.6)
        ax.tick_params(which="minor", length=1.0)
    ax_c[0].set_ylabel(cfg["HM_YLABEL"], fontsize=fs["label"], labelpad=2)
    ax_c[1].tick_params(labelleft=False)
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label(cfg["CBAR_LABEL"], fontsize=fs["label"], labelpad=2)
    cb.ax.tick_params(labelsize=fs["cbar"], pad=1.5, length=1.5)
    cb.outline.set_linewidth(0.5)
    ticks = [v for v in (0, 0.5, 1, 1.5, 2) if v <= vmax]
    cb.set_ticks(ticks); cb.set_ticklabels([("1×" if v == 1 else f"{v:g}") for v in ticks])
    p0, p1 = ax_c[0].get_position(), ax_c[1].get_position()
    fig.text((p0.x0 + p1.x1) / 2, p0.y0 - cfg["HM_XLABEL_IN"] / fh, cfg["HM_XLABEL"],
             ha="center", va="top", fontsize=fs["label"])

    # ---- sub-labels centered under each row ----
    for ax, key, lab in ((ax_a, "a", 0), (ax_b, "b", 1)):
        p = ax.get_position()
        fig.text((p.x0 + p.x1) / 2, p.y0 - cfg["SUBLABEL_IN"][key] / fh, cfg["PANEL_LABELS"][lab],
                 fontsize=fs["panel"], va="top", ha="center")
    fig.text((p0.x0 + p1.x1) / 2, p0.y0 - cfg["SUBLABEL_IN"]["c"] / fh, cfg["PANEL_LABELS"][2],
             fontsize=fs["panel"], va="top", ha="center")

    for ext in ("pdf", "png"):
        fig.savefig(f"{cfg['OUT_STEM']}.{ext}", dpi=cfg["DPI"])
    print("wrote", cfg["OUT_STEM"], f"{fw:.2f}x{fh:.2f} in", "vmax %.2f" % vmax,
          "compute max " + " ".join(f"{cfg['TOPIC_NAMES'][t]}={C[t].max():.2f}x" for t in cfg["B_TOPICS"]))

if __name__ == "__main__":
    main()
