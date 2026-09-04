#!/usr/bin/env python3
"""SPEC v1 three-lane motivation figure: NSDI single-column SVG (points) +
native draw.io twin (every bar / label / glyph an editable object) + rank
ledger CSV. Stdlib only. See SPEC_v1.md for the rulings.

  python figs/motivation/v1/build_v1_lanes.py \
      figs/motivation/phases_20260902-133340.json figs/motivation/phases_20260902-140327.json \
      --out figs/motivation/v1/lanes_v1 [--placement-frac 0.333] [--relative]
"""
import argparse, csv, html, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import build_figure_options as B  # noqa: E402  (rank_metrics / place_metrics / classify)

# ---- geometry (points) --------------------------------------------------------
COL_W = 241.0          # NSDI \columnwidth
L_GUT, R_GUT = 16.0, 50.0
LANE_H, LANE_GAP, RANK_GAP, VAR_GAP = 3.2, 0.5, 2.0, 10.0
TOP, HEAD_H, AXIS_H, LEGEND_H = 4.0, 9.0, 12.0, 11.0
FONT = "Helvetica, Arial, sans-serif"
INK, INK2, RULE = "#17191c", "#5a5e66", "#c9c6bd"
COL = {"inter": "#2a78d6", "intra": "#1baf7a", "gemm": "#eb6834", "place": "#eda100"}
ARMS = [("l01_nvshmem", "NVSHMEM a2av + GEMM"), ("eplb_l01_nvplace_bwire", "EPLB"), ("l01_allgather_dense", "COMET")]


def pick(arm, M, P, n=4):
    ranks = sorted(M, key=int); out, why = [], {}
    def add(r, w):
        if r not in out and len(out) < n: out.append(r); why[r] = w
    add(max(ranks, key=lambda r: M[r]["total"]), "longest layer-0 total")
    add(min(ranks, key=lambda r: M[r]["total"]), "shortest layer-0 total")
    if P: add(max(ranks, key=lambda r: P[r]["span"]), "longest placement")
    else: add(max(ranks, key=lambda r: M[r]["inter"]), "longest inter-node wire")
    add(max(ranks, key=lambda r: M[r]["gemm"]), "longest expert GEMM")
    add(min(ranks, key=lambda r: M[r]["inter"]), "shortest inter-node wire")
    add(min(ranks, key=lambda r: M[r]["gemm"]), "shortest expert GEMM")
    return sorted(out, key=int), why


def rank_data(cell, arm, it):
    M = {}
    for r in cell["ranks"]:
        m = B.rank_metrics(cell, r, it, arm)
        wire = m["lanes"]["inter"] + m["lanes"]["intra"]
        t0 = min(s["t0"] for s in wire)                         # origin = first dispatch wire event
        m["origin"] = t0; m["total"] = m["cut"] - t0
        for ln in ("inter", "intra", "gemm", "wait", "wait_post"):
            for s in m["lanes"][ln]: s["t0"] -= t0; s["t1"] -= t0
        # barrier release = end of the last pre-GEMM wait (ring/EPLB); none on COMET
        m["barrier"] = max([s["t1"] for s in m["lanes"]["wait"]], default=None) if arm != "l01_allgather_dense" else None
        M[r] = m
    P = {r: B.place_metrics(cell, r) for r in cell["ranks"]} if arm.startswith("eplb") else None
    return M, P


def annot(cell, arm, r):
    ri = int(r); W, rpn = cell["W"], cell["rpn"]
    if arm.startswith("eplb"):
        wb = cell["info"]["eplb_wire_bytes"]["0"]; rows = cell["info"]["gemm_rows_per_rank"]["0"][ri]
        ib = sum(wb[ri][d] for d in range(W) if d // rpn != ri // rpn)
    elif arm == "l01_allgather_dense":
        rows = cell["rows_per_rank"][ri]; ib = cell["tokens_per_rank"] * cell["H"] * 2 * (cell["nnodes"] - 1)
    else:
        rows = cell["rows_per_rank"][ri]; ib = cell["send_inter_bytes"][ri]
    return f"{rows / 1000:.1f}k rows · {ib / 2**20:.0f} MB"


class Doc:
    """collects primitives once; renders to SVG and to draw.io XML"""
    def __init__(self): self.items = []; self.h = 0
    def rect(self, x, y, w, h, color, layer, title=""): self.items.append(("rect", x, y, w, h, color, layer, title))
    def text(self, x, y, s, size, layer, anchor="start", color=INK, bold=False): self.items.append(("text", x, y, s, size, layer, anchor, color, bold))
    def line(self, x1, y1, x2, y2, color, layer, w=0.5): self.items.append(("line", x1, y1, x2, y2, color, layer, w))

    def svg(self):
        o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{COL_W}pt" height="{self.h:.1f}pt" viewBox="0 0 {COL_W} {self.h:.1f}" font-family="{FONT}">',
             f'<rect width="{COL_W}" height="{self.h:.1f}" fill="#ffffff"/>']
        for it in self.items:
            if it[0] == "rect":
                _, x, y, w, h, c, layer, title = it
                o.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 0.3):.2f}" height="{h:.2f}" fill="{c}">' + (f"<title>{html.escape(title)}</title>" if title else "") + "</rect>")
            elif it[0] == "text":
                _, x, y, s, size, layer, anchor, c, bold = it
                o.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" fill="{c}"' + (' font-weight="bold"' if bold else "") + f'>{html.escape(s)}</text>')
            else:
                _, x1, y1, x2, y2, c, layer, w = it
                o.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{c}" stroke-width="{w}"/>')
        o.append("</svg>"); return "\n".join(o)

    def drawio(self):
        S = 1.0 / 0.75   # draw.io units are px; 1 pt = 4/3 px so the page prints at NSDI size
        layers = ["bars", "glyphs", "axes", "labels"]
        cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
        for i, ln in enumerate(layers): cells.append(f'<mxCell id="L{i}" value="{ln}" style="locked=0" parent="0"/>')
        lid = {ln: f"L{i}" for i, ln in enumerate(layers)}; n = 10
        A = {"start": "left", "middle": "center", "end": "right"}
        for it in self.items:
            n += 1
            if it[0] == "rect":
                _, x, y, w, h, c, layer, title = it
                cells.append(f'<mxCell id="c{n}" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor={c};strokeColor=none;" vertex="1" parent="{lid[layer]}">'
                             f'<mxGeometry x="{x*S:.2f}" y="{y*S:.2f}" width="{max(w, 0.3)*S:.2f}" height="{h*S:.2f}" as="geometry"/></mxCell>')
            elif it[0] == "text":
                _, x, y, s, size, layer, anchor, c, bold = it
                w = max(8.0, 0.55 * size * len(s)) * S; hh = size * 1.4 * S
                xx = {"start": x * S, "middle": x * S - w / 2, "end": x * S - w}[anchor]
                st = f"text;html=1;fontSize={size * S:.1f};fontFamily=Helvetica;fontColor={c};align={A[anchor]};verticalAlign=middle;whiteSpace=nowrap;" + ("fontStyle=1;" if bold else "")
                cells.append(f'<mxCell id="c{n}" value="{html.escape(s, quote=True)}" style="{st}" vertex="1" parent="{lid[layer]}">'
                             f'<mxGeometry x="{xx:.2f}" y="{(y - size * 0.8) * S - hh / 2 + size * 0.5 * S:.2f}" width="{w:.2f}" height="{hh:.2f}" as="geometry"/></mxCell>')
            else:
                _, x1, y1, x2, y2, c, layer, w = it
                cells.append(f'<mxCell id="c{n}" value="" style="endArrow=none;html=1;strokeColor={c};strokeWidth={w * S:.2f};" edge="1" parent="{lid[layer]}">'
                             f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x1*S:.2f}" y="{y1*S:.2f}" as="sourcePoint"/><mxPoint x="{x2*S:.2f}" y="{y2*S:.2f}" as="targetPoint"/></mxGeometry></mxCell>')
        return ('<mxfile host="flux-motivation-v1"><diagram name="lanes_v1"><mxGraphModel dx="0" dy="0" grid="0" gridSize="1" guides="1" page="0" pageScale="1" '
                f'pageWidth="{COL_W*S:.0f}" pageHeight="{self.h*S:.0f}" background="#ffffff"><root>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")


def build(data, out, placement_frac, absolute, budget, n_ranks):
    cells = {c["variant"]: c for c in data["cells"].values() if c["budget_mib"] == budget and c["status"] == "ok"}
    panels = []
    for arm, title in ARMS:
        cell = cells[arm]; it = sorted(cell["ranks"]["0"]["iters"])[1]
        M, P = rank_data(cell, arm, it); chosen, why = pick(arm, M, P, n_ranks)
        panels.append((arm, title, cell, M, P, chosen, why, it))

    draw_w = COL_W - L_GUT - R_GUT
    pw = draw_w * placement_frac; gap = 4.0 if pw else 0.0
    tx0 = L_GUT + pw + gap; tw = draw_w - pw - gap
    tmax_all = max(M[r]["total"] for _, _, _, M, _, chosen, _, _ in panels for r in chosen) * 1.02
    rank_h = 3 * LANE_H + 2 * LANE_GAP
    D = Doc()
    y = TOP
    if pw: D.text(L_GUT, y + 6, "placement (rel.)", 6, "labels", color=INK2)
    D.text(tx0, y + 6, "token dispatch → expert GEMM" + (" (ms, shared scale)" if absolute else " (relative per baseline)"), 6, "labels", color=INK2)
    y += HEAD_H
    ledger = []
    for arm, title, cell, M, P, chosen, why, it in panels:
        D.text(L_GUT, y + 6, title, 7, "labels", bold=True)
        y += 8.5
        tmax = tmax_all if absolute else max(M[r]["total"] for r in chosen) * 1.02
        scale = tw / tmax
        pmax = max(P[r]["span"] for r in chosen) if P else None
        if not P and pw: D.text(L_GUT + pw / 2, y + rank_h * len(chosen) / 2 + 2, "no placement", 6, "labels", "middle", INK2)
        for r in chosen:
            m = M[r]; lanes_y = {"inter": y, "intra": y + LANE_H + LANE_GAP, "gemm": y + 2 * (LANE_H + LANE_GAP)}
            D.text(L_GUT - 2, y + rank_h / 2 + 2.2, f"r{r}", 6, "labels", "end")
            for ln in ("inter", "intra", "gemm"):
                D.line(tx0, lanes_y[ln] + LANE_H / 2, tx0 + tw, lanes_y[ln] + LANE_H / 2, RULE, "axes", 0.3)
                for s in m["lanes"][ln]:
                    D.rect(tx0 + s["t0"] * scale, lanes_y[ln], (s["t1"] - s["t0"]) * scale, LANE_H, COL[ln], "bars",
                           f"r{r} {ln} {s['t0']:.2f}–{s['t1']:.2f} ms")
            if m["barrier"] is not None:
                bx = tx0 + m["barrier"] * scale
                D.line(bx - 0.6, y - 0.3, bx - 0.6, y + rank_h + 0.3, INK, "glyphs", 0.5)
                D.line(bx + 0.6, y - 0.3, bx + 0.6, y + rank_h + 0.3, INK, "glyphs", 0.5)
            if P:
                ps = pw * (P[r]["span"] / pmax)
                for ln in ("inter", "intra"):
                    for s in P[r]["lanes"][ln]:
                        D.rect(L_GUT + s["t0"] / P[r]["span"] * ps, lanes_y[ln], (s["t1"] - s["t0"]) / P[r]["span"] * ps, LANE_H, COL["place"] if ln == "inter" else COL["intra"], "bars",
                               f"r{r} placement {ln} {s['t0']:.1f}–{s['t1']:.1f} ms")
                # label on the (empty) SM lane of the placement column, right-aligned to the column edge
                D.text(L_GUT + pw - 1, lanes_y["gemm"] + LANE_H - 0.2, f"{P[r]['span']:.0f} ms · {P[r]['inter_bytes'] / 2**30:.1f} GB", 5, "labels", "end", INK2)
            D.text(COL_W - 1, y + rank_h / 2 + 2, annot(cell, arm, r), 5.5, "labels", "end", INK2)
            ledger.append(dict(baseline=title, rank=r, node=int(r) // cell["rpn"], why=why[r], total_ms=round(m["total"], 2), inter_ms=round(m["inter"], 2),
                               intra_ms=round(m["intra"], 2), wait_ms=round(m["wait"], 2), gemm_ms=round(m["gemm"], 2), barrier_ms=None if m["barrier"] is None else round(m["barrier"], 2),
                               placement_ms=round(P[r]["span"], 1) if P else "", placement_inter_GB=round(P[r]["inter_bytes"] / 2**30, 2) if P else "", annotation=annot(cell, arm, r)))
            y += rank_h + RANK_GAP
        y += VAR_GAP - RANK_GAP
    # shared time axis
    if absolute:
        step = 5 if tmax_all > 12 else 2
        D.line(tx0, y, tx0 + tw, y, INK2, "axes", 0.5)
        t = 0
        while t <= tmax_all:
            D.line(tx0 + t * scale, y, tx0 + t * scale, y + 2, INK2, "axes", 0.5)
            D.text(tx0 + t * scale, y + 7.5, f"{t}", 6, "labels", "middle", INK2); t += step
        D.text(tx0 + tw, y + 7.5, "ms", 6, "labels", "end", INK2)
    y += AXIS_H
    # legend
    lx = L_GUT
    for key, lab in (("inter", "NIC RDMA"), ("intra", "NVLink"), ("gemm", "expert GEMM"), ("place", "placement put")):
        D.rect(lx, y + 1.5, 7, 4, COL[key], "bars"); D.text(lx + 9, y + 5.3, lab, 5.5, "labels", color=INK2); lx += 9 + 2.75 * len(lab) + 7
    D.line(lx, y + 0.5, lx, y + 6.5, INK, "glyphs", 0.5); D.line(lx + 1.4, y + 0.5, lx + 1.4, y + 6.5, INK, "glyphs", 0.5)
    D.text(lx + 4, y + 5.3, "barrier", 5.5, "labels", color=INK2)
    y += LEGEND_H
    D.h = y
    open(out + ".svg", "w").write(D.svg()); open(out + ".drawio", "w").write(D.drawio())
    with open(out + "_ranks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ledger[0].keys())); w.writeheader(); w.writerows(ledger)
    return panels, ledger, tmax_all


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--placement-frac", type=float, default=1 / 3); ap.add_argument("--relative", action="store_true")
    ap.add_argument("--budget", type=int, default=16); ap.add_argument("--ranks", type=int, default=4)
    a = ap.parse_args()
    data = {"cells": {}}
    for j in a.json: data["cells"].update(json.load(open(j))["cells"])
    panels, ledger, tmax = build(data, a.out, a.placement_frac, not a.relative, a.budget, a.ranks)
    print(f"shared scale 0–{tmax:.1f} ms; wrote {a.out}.svg / .drawio / _ranks.csv")
    for row in ledger: print(f"  {row['baseline']:<22} r{row['rank']:<3} node {row['node']}  {row['why']:<26} total {row['total_ms']:6.2f}  NIC {row['inter_ms']:5.2f}  wait {row['wait_ms']:5.2f}  GEMM {row['gemm_ms']:5.2f}  place {row['placement_ms']}")
