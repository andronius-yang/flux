#!/usr/bin/env python3
"""SPEC v2 cross-column three-lane motivation figure: SVG (points) + native
draw.io twin (layers background / bars / glyphs / axes / labels) + rank ledger.
Rulings in SPEC_v2.md. Stdlib only.

  python figs/motivation/v2/build_v2_lanes.py \\
      figs/motivation/phases_20260904-123815.json figs/motivation/phases_20260904-132050.json \\
      --out figs/motivation/v2/lanes_v2 [--budget 32] [--eplb-arm ringplace]
"""
import argparse, csv, html, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "v1"))
import build_v1_lanes as V  # noqa: E402  (rank_data / pick / Doc)

# ---- geometry (points) --------------------------------------------------------
TEXT_W = 504.0            # NSDI \textwidth
HEIGHT_FRAC = 0.25        # figure height / width
GAP = 14.0                # between subdiagrams
L_GUT = 24.0              # lane labels, first subdiagram only
LANE_H, LANE_GAP, RANK_GAP = 5.0, 1.0, 3.0
TITLE_H, AXIS_H, LEGEND_H, TOP = 17.0, 11.0, 12.0, 2.0
PLACE_FRAC = 0.4          # EPLB: expert-comm block share of the subdiagram width
FONT = "Helvetica, Arial, sans-serif"
INK, INK2, LINE = "#17191c", "#5a5e66", "#7d8289"
COL = {"token": "#2a78d6", "expert_comm": "#1baf7a", "comp": "#eda100"}
DASH = {"inter": "", "intra": "2,1.5", "gemm": "0.6,1.2"}     # NIC solid, NVLink dashed, GPU dotted
PANELS = [  # (arm, title) in left-to-right order (postdoc XML)
    ("l01_nvshmem", "Computation Imbalance + Communication Imbalance"),
    ("l01_allgather_dense", "Token Comm. Balanced + Expert Comp. Imbalance"),
    ("EPLB", "Expert Comp. Balanced → Comm. Imbalance"),
]
LANES = ("inter", "intra", "gemm"); LANE_LABEL = {"inter": "NIC RDMA", "intra": "NVLink", "gemm": "GPU"}


class Doc(V.Doc):
    """v1 Doc + dashed lines"""
    def dline(self, x1, y1, x2, y2, color, layer, w, dash):
        self.items.append(("dline", x1, y1, x2, y2, color, layer, w, dash))

    def svg(self):
        o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{TEXT_W}pt" height="{self.h:.1f}pt" viewBox="0 0 {TEXT_W} {self.h:.1f}" font-family="{FONT}">',
             f'<rect width="{TEXT_W}" height="{self.h:.1f}" fill="#ffffff"/>']
        for it in self.items:
            if it[0] == "rect":
                _, x, y, w, h, c, layer, title = it
                o.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 0.3):.2f}" height="{h:.2f}" fill="{c}">' + (f"<title>{html.escape(title)}</title>" if title else "") + "</rect>")
            elif it[0] == "text":
                _, x, y, s, size, layer, anchor, c, bold = it
                o.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" fill="{c}"' + (' font-weight="bold"' if bold else "") + f'>{html.escape(s)}</text>')
            elif it[0] == "line":
                _, x1, y1, x2, y2, c, layer, w = it
                o.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{c}" stroke-width="{w}"/>')
            else:
                _, x1, y1, x2, y2, c, layer, w, dash = it
                o.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{c}" stroke-width="{w}"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
        o.append("</svg>"); return "\n".join(o)

    def drawio(self):
        S = 1.0 / 0.75
        layers = ["background", "bars", "glyphs", "axes", "labels"]
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
                if it[0] == "line": _, x1, y1, x2, y2, c, layer, w = it; dash = ""
                else: _, x1, y1, x2, y2, c, layer, w, dash = it
                dst = ""
                if dash:
                    a, b = [float(v) for v in dash.split(",")]
                    dst = f"dashed=1;dashPattern={a * S:.1f} {b * S:.1f};"
                cells.append(f'<mxCell id="c{n}" value="" style="endArrow=none;html=1;strokeColor={c};strokeWidth={w * S:.2f};{dst}" edge="1" parent="{lid[layer]}">'
                             f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x1*S:.2f}" y="{y1*S:.2f}" as="sourcePoint"/><mxPoint x="{x2*S:.2f}" y="{y2*S:.2f}" as="targetPoint"/></mxGeometry></mxCell>')
        return ('<mxfile host="flux-motivation-v2"><diagram name="lanes_v2"><mxGraphModel dx="0" dy="0" grid="0" gridSize="1" guides="1" page="0" pageScale="1" '
                f'pageWidth="{TEXT_W*S:.0f}" pageHeight="{self.h*S:.0f}" background="#ffffff"><root>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")


def nice_step(t):
    for s in (1, 2, 5, 10, 20, 25, 50):
        if t / s <= 6: return s
    return 100


def build(data, out, budget, eplb_arm, n_ranks):
    cells = {c["variant"]: c for c in data["cells"].values() if c["budget_mib"] == budget and c["status"] == "ok"}
    panels = []
    for arm, title in PANELS:
        a = eplb_arm if arm == "EPLB" else arm
        c = cells[a]; it = sorted(c["ranks"]["0"]["iters"])[1]
        M, P = V.rank_data(c, a, it); chosen, why = V.pick(a, M, P, n_ranks)
        panels.append((a, title, c, M, P, chosen, why, it))

    H = TEXT_W * HEIGHT_FRAC
    sub_w = (TEXT_W - L_GUT - 2 * GAP) / 3
    rank_h = 3 * LANE_H + 2 * LANE_GAP
    body_h = n_ranks * rank_h + (n_ranks - 1) * RANK_GAP
    # vertical budget check: fit lanes into the 25 % height, shrink if needed
    avail = H - TOP - TITLE_H - AXIS_H - LEGEND_H
    scale_y = min(1.0, avail / body_h)
    lane_h, lane_gap, rank_gap = LANE_H * scale_y, LANE_GAP * scale_y, RANK_GAP * scale_y
    rank_h = 3 * lane_h + 2 * lane_gap; body_h = n_ranks * rank_h + (n_ranks - 1) * rank_gap
    D = Doc(); y0 = TOP + TITLE_H
    ledger = []
    for pi, (arm, title, c, M, P, chosen, why, it) in enumerate(panels):
        x0 = L_GUT + pi * (sub_w + GAP)
        # two-line title (split at the connector) so 6.5 pt bold fits a 155 pt panel
        for sep in (" + ", " → "):
            if sep in title:
                a, b = title.split(sep, 1); lines = [a + sep.strip(), b]; break
        else: lines = [title]
        for li, ln_ in enumerate(lines):
            D.text(x0 + sub_w / 2, TOP + 6.5 + li * 7.5, ln_, 6.5, "labels", "middle", INK, bold=True)
        place = P is not None
        pw = sub_w * PLACE_FRAC if place else 0.0; pgap = 6.0 if place else 0.0
        tx0 = x0 + pw + pgap; tw = sub_w - pw - pgap
        tmax = max(M[r]["total"] for r in chosen) * 1.02; sc = tw / tmax
        pmax = max(P[r]["span"] for r in chosen) if place else None
        y = y0
        for ri, r in enumerate(chosen):
            ly = {ln: y + i * (lane_h + lane_gap) for i, ln in enumerate(LANES)}
            # background resource lines (bottom layer): per subdiagram, not continuous
            for ln in LANES:
                yy = ly[ln] + lane_h / 2
                D.dline(tx0, yy, tx0 + tw, yy, LINE, "background", 0.5, DASH[ln])
                if place: D.dline(x0, yy, x0 + pw, yy, LINE, "background", 0.5, DASH[ln])
                if pi == 0 and ri == 0:
                    D.text(x0 - 2, ly[ln] + lane_h - 0.8, LANE_LABEL[ln], 4.2, "labels", "end", INK2)
            m = M[r]
            for ln in LANES:
                col = COL["comp"] if ln == "gemm" else COL["token"]
                for s in m["lanes"][ln]:
                    D.rect(tx0 + s["t0"] * sc, ly[ln], (s["t1"] - s["t0"]) * sc, lane_h, col, "bars", f"r{r} {ln} {s['t0']:.2f}–{s['t1']:.2f} ms")
            if m["barrier"] is not None:
                bx = tx0 + m["barrier"] * sc
                for dx in (-0.7, 0.7): D.line(bx + dx, y - 0.5, bx + dx, y + rank_h + 0.5, INK, "glyphs", 0.6)
            if place:
                ps = pw * P[r]["span"] / pmax
                for ln in ("inter", "intra"):
                    for s in P[r]["lanes"][ln]:
                        D.rect(x0 + s["t0"] / P[r]["span"] * ps, ly[ln], (s["t1"] - s["t0"]) / P[r]["span"] * ps, lane_h, COL["expert_comm"], "bars", f"r{r} placement {ln}")
                lab = f"{P[r]['span']:.0f} ms"; lw = 2.9 * len(lab) + 2
                D.rect(x0 + pw - 1 - lw, ly["gemm"] - 0.3, lw + 1, lane_h + 0.6, "#ffffff", "glyphs")   # halo over the dotted GPU line
                D.text(x0 + pw - 1, ly["gemm"] + lane_h - 0.6, lab, 5, "labels", "end", INK2)
            ledger.append(dict(panel=title, arm=arm, rank=r, node=int(r) // c["rpn"], why=why[r], total_ms=round(m["total"], 2), nic_ms=round(m["inter"], 2),
                               nvlink_ms=round(m["intra"], 2), wait_ms=round(m["wait"], 2), barrier_at_ms=None if m["barrier"] is None else round(m["barrier"], 2),
                               gemm_ms=round(m["gemm"], 2), placement_ms=round(P[r]["span"], 1) if place else "", capsule=c.get("capsule", "")))
            y += rank_h + rank_gap
        # per-subdiagram axis
        ay = y0 + body_h + 3
        D.line(tx0, ay, tx0 + tw, ay, INK2, "axes", 0.5)
        step = nice_step(tmax); t = 0
        while t <= tmax:
            D.line(tx0 + t * sc, ay, tx0 + t * sc, ay + 1.8, INK2, "axes", 0.5)
            if tx0 + t * sc < tx0 + tw - 9: D.text(tx0 + t * sc, ay + 7, f"{t}", 5, "labels", "middle", INK2)
            t += step
        D.text(tx0 + tw, ay + 7, "ms", 5, "labels", "end", INK2)
        if place: D.text(x0 + pw / 2, ay + 7, "expert comm. (relative)", 5, "labels", "middle", INK2)
    # legend: colors + barrier, then resource pattern ledger
    ly_ = y0 + body_h + AXIS_H + 2; lx = L_GUT
    for key, lab in (("token", "Token Comm."), ("expert_comm", "Expert Comm."), ("comp", "Expert Comp.")):
        D.rect(lx, ly_ + 1, 8, 4.5, COL[key], "bars"); D.text(lx + 10, ly_ + 5, lab, 5.5, "labels", color=INK2); lx += 10 + 2.9 * len(lab) + 9
    for dx in (0, 1.5): D.line(lx + dx, ly_, lx + dx, ly_ + 6.5, INK, "glyphs", 0.6)
    D.text(lx + 4, ly_ + 5, "Barrier", 5.5, "labels", color=INK2); lx += 4 + 2.9 * 7 + 16
    for ln in LANES:
        D.dline(lx, ly_ + 3.2, lx + 12, ly_ + 3.2, LINE, "background", 0.6, DASH[ln]); D.text(lx + 14, ly_ + 5, LANE_LABEL[ln], 5.5, "labels", color=INK2)
        lx += 14 + 2.9 * len(LANE_LABEL[ln]) + 9
    D.h = ly_ + LEGEND_H
    open(out + ".svg", "w").write(D.svg()); open(out + ".drawio", "w").write(D.drawio())
    with open(out + "_ranks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ledger[0].keys())); w.writeheader(); w.writerows(ledger)
    return panels, ledger, D.h


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=32); ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--eplb-arm", default="ringplace", choices=["ringplace", "bwire"]); ap.add_argument("--rule", default="v1.1")
    a = ap.parse_args()
    V.RULE = a.rule
    data = {"cells": {}}
    for j in a.json:
        cap = os.path.basename(j).replace("phases_", "").replace(".json", "")
        for k, c in json.load(open(j))["cells"].items():
            c["capsule"] = cap; data["cells"][k] = c
    arm = {"ringplace": "eplb_l01_nvplace_bwire_ringplace", "bwire": "eplb_l01_nvplace_bwire"}[a.eplb_arm]
    panels, ledger, h = build(data, a.out, a.budget, arm, a.ranks)
    print(f"wrote {a.out}.svg/.drawio/_ranks.csv  ({TEXT_W:.0f} x {h:.1f} pt)")
    for row in ledger: print(f"  {row['panel'][:22]:<22} r{row['rank']:<3} {row['why']:<26} total {row['total_ms']:6.2f} NIC {row['nic_ms']:6.2f} wait {row['wait_ms']:6.2f} GEMM {row['gemm_ms']:5.2f} place {row['placement_ms']}")
