#!/usr/bin/env python3
"""SPEC v3 cross-column three-lane motivation figure (2026-09-12): v2 layout
with panel 3 = the authentic MoonEP journey (per-batch expert movement, the
exposed-wire twin), problem/solution titles with red highlights and
cross-panel arrows. SVG (points) + native draw.io twin + rank ledger + png/pdf.
Rulings in SPEC_v3.md. Stdlib only (cairosvg CLI for png/pdf if present).

  python figs/motivation/v3/build_v3_lanes.py \\
      figs/motivation/phases_20260904-123815.json figs/motivation/phases_<moonep capsule>.json \\
      --out figs/motivation/v3/lanes_v3 [--budget 32] [--moonep-arm bwire|a2a]
"""
import argparse, csv, html, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("..", os.path.join("..", "v1"), os.path.join("..", "v2")):
    sys.path.insert(0, os.path.join(HERE, d))
import build_v1_lanes as V  # noqa: E402  (rank_data / pick)
import build_v2_lanes as V2  # noqa: E402  (Doc, nice_step)
import build_figure_options as B  # noqa: E402  (merge)

# ---- geometry (points) --------------------------------------------------------
TEXT_W = 504.0            # NSDI \textwidth
HEIGHT_FRAC = 0.27        # figure height / width (v2: 0.25; +arrow arc room)
GAP = 14.0                # between subdiagrams
L_GUT = 24.0              # lane labels, first subdiagram only
LANE_H, LANE_GAP, RANK_GAP = 5.0, 1.0, 3.0
ARC_H = 9.0               # room above the titles for the cross-panel arc
TITLE_H, AXIS_H, LEGEND_H, TOP = 17.0, 11.0, 12.0, 2.0
FONT = "Helvetica, Arial, sans-serif"
INK, INK2, LINE = "#17191c", "#5a5e66", "#7d8289"
RED = "#c0392b"           # unresolved problem in a title
ARROW = "#8a2e24"         # problem -> resolving subdiagram
COL = {"token": "#2a78d6", "expert_comm": "#1baf7a", "comp": "#eda100"}
DASH = {"inter": "", "intra": "2,1.5", "gemm": "0.6,1.2"}     # NIC solid, NVLink dashed, GPU dotted
LANES = ("inter", "intra", "gemm"); LANE_LABEL = {"inter": "NIC RDMA", "intra": "NVLink", "gemm": "GPU"}
# (arm key, title line 1, title line 2); each line = [(segment, red?), ...]
PANELS = [
    ("l01_nvshmem", [("Inter-node communication bottleneck", True), (" +", False)], [("Computation Imbalance", True)]),
    ("l01_allgather_dense", [("Communication Overlap +", False)], [("Computation Imbalance", True)]),
    ("MOONEP", [("Expert communication bottleneck", True), (" +", False)], [("Computation Balance", False)]),
]
# arrows: (from panel, from line, to panel, to line, kind) — a red problem to the subdiagram that resolves it
ARROWS = [(0, 0, 1, 0, "h"), (1, 1, 2, 1, "h"), (0, 1, 2, 1, "arc")]
MOONEP_ARMS = {"bwire": "moonep_l01_nvshmem_getmem_bwire", "a2a": "moonep_l01_nvshmem_getmem"}
MOONEP_L0 = ("plan_comm_ms", "plan_ms", "pack_ms", "comm_ms", "scatter_ms", "prefetch_ms", "gemm_ms")


class Doc(V2.Doc):
    """v2 Doc + rich (multi-color) text + arrows"""
    def rtext(self, x, y, segs, size, layer, anchor="middle", bold=True):
        self.items.append(("rtext", x, y, segs, size, layer, anchor, bold))

    def arrow(self, x1, y1, x2, y2, cx, cy, color, layer, w=0.6):
        self.items.append(("arrow", x1, y1, x2, y2, cx, cy, color, layer, w))

    @staticmethod
    def text_w(s, size, bold=False):
        return (0.58 if bold else 0.55) * size * len(s)

    def svg(self):
        o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{TEXT_W}pt" height="{self.h:.1f}pt" viewBox="0 0 {TEXT_W} {self.h:.1f}" font-family="{FONT}">',
             '<defs><marker id="ah" viewBox="0 0 6 6" refX="5.5" refY="3" markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
             f'<path d="M0,0.5 L6,3 L0,5.5 z" fill="{ARROW}"/></marker></defs>',
             f'<rect width="{TEXT_W}" height="{self.h:.1f}" fill="#ffffff"/>']
        for it in self.items:
            if it[0] == "rect":
                _, x, y, w, h, c, layer, title = it
                o.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 0.3):.2f}" height="{h:.2f}" fill="{c}">' + (f"<title>{html.escape(title)}</title>" if title else "") + "</rect>")
            elif it[0] == "text":
                _, x, y, s, size, layer, anchor, c, bold = it
                o.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" fill="{c}"' + (' font-weight="bold"' if bold else "") + f'>{html.escape(s)}</text>')
            elif it[0] == "rtext":
                _, x, y, segs, size, layer, anchor, bold = it
                spans = "".join(f'<tspan fill="{RED if red else INK}">{html.escape(s)}</tspan>' for s, red in segs)
                o.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" xml:space="preserve"' + (' font-weight="bold"' if bold else "") + f'>{spans}</text>')
            elif it[0] == "line":
                _, x1, y1, x2, y2, c, layer, w = it
                o.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{c}" stroke-width="{w}"/>')
            elif it[0] == "dline":
                _, x1, y1, x2, y2, c, layer, w, dash = it
                o.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{c}" stroke-width="{w}"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
            else:
                _, x1, y1, x2, y2, cx, cy, c, layer, w = it
                d = f"M{x1:.2f},{y1:.2f} L{x2:.2f},{y2:.2f}" if cx is None else f"M{x1:.2f},{y1:.2f} Q{cx:.2f},{cy:.2f} {x2:.2f},{y2:.2f}"
                o.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{w}" marker-end="url(#ah)"/>')
        o.append("</svg>"); return "\n".join(o)

    def drawio(self):
        S = 1.0 / 0.75
        layers = ["background", "bars", "glyphs", "axes", "labels"]
        cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
        for i, ln in enumerate(layers): cells.append(f'<mxCell id="L{i}" value="{ln}" style="locked=0" parent="0"/>')
        lid = {ln: f"L{i}" for i, ln in enumerate(layers)}; n = 10
        A = {"start": "left", "middle": "center", "end": "right"}
        def text_cell(x, y, value, size, anchor, layer, bold, width_chars):
            w = max(8.0, 0.55 * size * width_chars) * S; hh = size * 1.4 * S
            xx = {"start": x * S, "middle": x * S - w / 2, "end": x * S - w}[anchor]
            st = f"text;html=1;fontSize={size * S:.1f};fontFamily=Helvetica;align={A[anchor]};verticalAlign=middle;whiteSpace=nowrap;" + ("fontStyle=1;" if bold else "")
            return (f'<mxCell id="c{n}" value="{value}" style="{st}" vertex="1" parent="{lid[layer]}">'
                    f'<mxGeometry x="{xx:.2f}" y="{(y - size * 0.8) * S - hh / 2 + size * 0.5 * S:.2f}" width="{w:.2f}" height="{hh:.2f}" as="geometry"/></mxCell>')
        for it in self.items:
            n += 1
            if it[0] == "rect":
                _, x, y, w, h, c, layer, title = it
                cells.append(f'<mxCell id="c{n}" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor={c};strokeColor=none;" vertex="1" parent="{lid[layer]}">'
                             f'<mxGeometry x="{x*S:.2f}" y="{y*S:.2f}" width="{max(w, 0.3)*S:.2f}" height="{h*S:.2f}" as="geometry"/></mxCell>')
            elif it[0] == "text":
                _, x, y, s, size, layer, anchor, c, bold = it
                v = html.escape(f'<font color="{c}">{html.escape(s)}</font>', quote=True)
                cells.append(text_cell(x, y, v, size, anchor, layer, bold, len(s)))
            elif it[0] == "rtext":
                _, x, y, segs, size, layer, anchor, bold = it
                v = html.escape("".join(f'<font color="{RED if red else INK}">{html.escape(s)}</font>' for s, red in segs), quote=True)
                cells.append(text_cell(x, y, v, size, anchor, layer, bold, sum(len(s) for s, _ in segs)))
            elif it[0] in ("line", "dline"):
                if it[0] == "line": _, x1, y1, x2, y2, c, layer, w = it; dash = ""
                else: _, x1, y1, x2, y2, c, layer, w, dash = it
                dst = ""
                if dash:
                    a, b = [float(v) for v in dash.split(",")]
                    dst = f"dashed=1;dashPattern={a * S:.1f} {b * S:.1f};"
                cells.append(f'<mxCell id="c{n}" value="" style="endArrow=none;html=1;strokeColor={c};strokeWidth={w * S:.2f};{dst}" edge="1" parent="{lid[layer]}">'
                             f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x1*S:.2f}" y="{y1*S:.2f}" as="sourcePoint"/><mxPoint x="{x2*S:.2f}" y="{y2*S:.2f}" as="targetPoint"/></mxGeometry></mxCell>')
            else:
                _, x1, y1, x2, y2, cx, cy, c, layer, w = it
                pts = "" if cx is None else f'<Array as="points"><mxPoint x="{cx*S:.2f}" y="{cy*S:.2f}"/></Array>'
                cells.append(f'<mxCell id="c{n}" value="" style="endArrow=block;endFill=1;endSize=4;html=1;strokeColor={c};strokeWidth={w * S:.2f};{"curved=1;" if cx is not None else ""}" edge="1" parent="{lid[layer]}">'
                             f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x1*S:.2f}" y="{y1*S:.2f}" as="sourcePoint"/><mxPoint x="{x2*S:.2f}" y="{y2*S:.2f}" as="targetPoint"/>{pts}</mxGeometry></mxCell>')
        return ('<mxfile host="flux-motivation-v3"><diagram name="lanes_v3"><mxGraphModel dx="0" dy="0" grid="0" gridSize="1" guides="1" page="0" pageScale="1" '
                f'pageWidth="{TEXT_W*S:.0f}" pageHeight="{self.h*S:.0f}" background="#ffffff"><root>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")


# ---- MoonEP data adapter -------------------------------------------------------
UNKNOWN = set()


def moonep_classify(name):
    n = name
    if n.startswith("memcpy10"): return "intra"                       # token P2P put (NVLink)
    if n.startswith("memcpy"): return "prep"
    if "weight_prefetch_getmem" in n or "quiet" in n.lower(): return "expert"
    if "proxy_rma" in n: return "inter"                               # token put (NIC)
    if n.startswith("a2a_single"): return "inter"                     # staged a2a kernel (a2a arm only)
    if "barrier_on_stream" in n or "signal_wait_until" in n: return "wait"
    if n in ("Kernel", "Kernel2"): return "gemm"
    UNKNOWN.add(n); return "prep"


def moonep_rank_data(cell, it):
    """per-rank layer-0 window of the MoonEP journey: dispatch wire -> barrier
    -> per-batch expert pull (getmem kernel + quiet) -> per-segment GEMM.
    Expert-comm lane = NIC when any of this rank's pulls is inter-node, else
    NVLink (pull sets are homogeneous on the captured plans; mixed sets are
    split by byte share)."""
    rec = cell["recorded"]; rpn = cell["rpn"]
    iters = sorted(cell["ranks"][next(iter(cell["ranks"]))]["iters"]); k = iters.index(it)
    pairs_all = cell["info"].get("moonep_prefetch_pairs", {})
    M = {}
    for r in cell["ranks"]:
        cut = sum(rec[m][r][k] for m in MOONEP_L0)
        ev = cell["ranks"][r]["iters"][it]["events"]
        segs = [dict(cls=moonep_classify(e["name"]), t0=e["t0"], t1=e["t1"], bytes=e.get("bytes", 0), name=e["name"]) for e in ev if e["t0"] < cut]
        for s in segs: s["t1"] = min(s["t1"], cut)
        by = lambda c: [s for s in segs if s["cls"] == c]
        g, inter, intra, wait, ex = by("gemm"), by("inter"), by("intra"), by("wait"), by("expert")
        g_t0 = min([x["t0"] for x in g], default=cut)
        ex_t0 = min([x["t0"] for x in ex], default=g_t0)
        wire = inter + intra
        t0 = min([s["t0"] for s in wire], default=0.0)
        pairs = pairs_all.get(r, [])
        ri = int(r); n_inter = sum(1 for _, _, h in pairs if h // rpn != ri // rpn); n_intra = len(pairs) - n_inter
        ex_m = B.merge(ex, gap=0.2)
        for s in ex_m: s["task"] = "expert"
        if n_inter and n_intra:          # mixed: split each span by pull share (not seen on the captured plans)
            f = n_inter / len(pairs)
            ex_inter = [dict(s, t1=s["t0"] + (s["t1"] - s["t0"]) * f) for s in ex_m]
            ex_intra = [dict(s, t0=s["t0"] + (s["t1"] - s["t0"]) * f) for s in ex_m]
        elif n_inter: ex_inter, ex_intra = ex_m, []
        elif n_intra: ex_inter, ex_intra = [], ex_m
        else: ex_inter, ex_intra = [], []
        lanes = {"inter": B.merge(inter) + ex_inter, "intra": B.merge(intra) + ex_intra, "gemm": B.merge(g),
                 "wait": B.merge([w for w in wait if w["t0"] < ex_t0])}
        for ln in ("inter", "intra", "gemm", "wait"):
            for s in lanes[ln]:
                s.setdefault("task", "token"); s["t0"] -= t0; s["t1"] -= t0
        dur = lambda L: sum(x["t1"] - x["t0"] for x in L)
        M[r] = dict(cut=cut, origin=t0, total=cut - t0, gemm=dur(g), inter=dur(inter), intra=dur(intra), wait=dur(lanes["wait"]),
                    expert=dur(ex_m), expert_inter=dur(ex_inter), expert_intra=dur(ex_intra), n_pairs=len(pairs), n_inter=n_inter,
                    barrier=max([s["t1"] for s in lanes["wait"]], default=None), lanes=lanes)
    return M


def pick_moonep(M, n=4):
    ranks = sorted(M, key=int); out, why = [], {}
    def add(r, w):
        if r not in out and len(out) < n: out.append(r); why[r] = w
    add(max(ranks, key=lambda r: M[r]["expert"]), "longest expert pull")
    # shortest REAL pull, preferring an on-node (NVLink) pull so the NVLink
    # lane's expert-comm case is visible (user ruling 2026-09-12); a rank
    # with no expert assigned only when no such pull exists
    nvl = [r for r in ranks if M[r]["n_pairs"] and not M[r]["n_inter"]]
    if nvl: add(min(nvl, key=lambda r: M[r]["expert"]), "one-expert pull over NVLink")
    else: add(min(ranks, key=lambda r: M[r]["expert"]), "shortest expert pull")
    add(max(ranks, key=lambda r: M[r]["inter"]), "longest inter-node wire")
    add(min(ranks, key=lambda r: M[r]["inter"]), "shortest inter-node wire")
    # fallbacks when extremes coincide: a mid-length pull, then the total extremes
    mid = sorted(ranks, key=lambda r: M[r]["expert"])
    add(mid[len(mid) // 2], "median expert pull")
    add(max(ranks, key=lambda r: M[r]["total"]), "longest layer-0 total")
    add(min(ranks, key=lambda r: M[r]["total"]), "shortest layer-0 total")
    return sorted(out, key=int), why


def token_rank_data(cell, arm, it):
    M, _ = V.rank_data(cell, arm, it)
    for m in M.values():
        for ln in ("inter", "intra", "gemm", "wait"):
            for s in m["lanes"][ln]: s["task"] = "token"
        m["expert"] = 0.0
    return M


# ---- figure ---------------------------------------------------------------------
def build(data, out, budget, moonep_arm, n_ranks):
    cells = {c["variant"]: c for c in data["cells"].values() if c["budget_mib"] == budget and c["status"] == "ok"}
    panels = []
    for arm, l1, l2 in PANELS:
        a = moonep_arm if arm == "MOONEP" else arm
        c = cells.get(a)
        if c is None:
            panels.append((a, l1, l2, None, None, [], {}, None)); continue
        it = sorted(c["ranks"]["0"]["iters"])[1]
        if arm == "MOONEP":
            M = moonep_rank_data(c, it); chosen, why = pick_moonep(M, n_ranks)
        else:
            M = token_rank_data(c, a, it); chosen, why = V.pick(a, M, None, n_ranks)
        panels.append((a, l1, l2, c, M, chosen, why, it))

    H = TEXT_W * HEIGHT_FRAC
    sub_w = (TEXT_W - L_GUT - 2 * GAP) / 3
    rank_h = 3 * LANE_H + 2 * LANE_GAP
    body_h = n_ranks * rank_h + (n_ranks - 1) * RANK_GAP
    avail = H - TOP - ARC_H - TITLE_H - AXIS_H - LEGEND_H
    scale_y = min(1.0, avail / body_h)
    lane_h, lane_gap, rank_gap = LANE_H * scale_y, LANE_GAP * scale_y, RANK_GAP * scale_y
    rank_h = 3 * lane_h + 2 * lane_gap; body_h = n_ranks * rank_h + (n_ranks - 1) * rank_gap
    D = Doc(); y0 = TOP + ARC_H + TITLE_H
    ledger = []; TSIZE = 6.5; title_pos = {}
    for pi, (arm, l1, l2, c, M, chosen, why, it) in enumerate(panels):
        x0 = L_GUT + pi * (sub_w + GAP); xc = x0 + sub_w / 2
        for li, segs in enumerate((l1, l2)):
            y = TOP + ARC_H + 6.5 + li * 7.5
            D.rtext(xc, y, segs, TSIZE, "labels", "middle", bold=True)
            w = D.text_w("".join(s for s, _ in segs), TSIZE, bold=True)
            title_pos[(pi, li)] = (xc - w / 2, xc + w / 2, y)
        tx0, tw = x0, sub_w
        y = y0
        if M is None:
            for ri in range(n_ranks):
                for i, ln in enumerate(LANES):
                    yy = y + i * (lane_h + lane_gap) + lane_h / 2
                    D.dline(tx0, yy, tx0 + tw, yy, LINE, "background", 0.5, DASH[ln])
                y += rank_h + rank_gap
            D.text(xc, y0 + body_h / 2, "(capture pending)", 6, "labels", "middle", INK2)
            continue
        tmax = max(M[r]["total"] for r in chosen) * 1.02; sc = tw / tmax
        for ri, r in enumerate(chosen):
            ly = {ln: y + i * (lane_h + lane_gap) for i, ln in enumerate(LANES)}
            for ln in LANES:
                yy = ly[ln] + lane_h / 2
                D.dline(tx0, yy, tx0 + tw, yy, LINE, "background", 0.5, DASH[ln])
                if pi == 0 and ri == 0:
                    D.text(x0 - 2, ly[ln] + lane_h - 0.8, LANE_LABEL[ln], 4.2, "labels", "end", INK2)
            m = M[r]
            for ln in LANES:
                for s in m["lanes"][ln]:
                    col = COL["comp"] if ln == "gemm" else COL["expert_comm" if s.get("task") == "expert" else "token"]
                    D.rect(tx0 + s["t0"] * sc, ly[ln], (s["t1"] - s["t0"]) * sc, lane_h, col, "bars", f"r{r} {ln} {s.get('task', 'token')} {s['t0']:.2f}–{s['t1']:.2f} ms")
            if m["barrier"] is not None:
                bx = tx0 + m["barrier"] * sc
                for dx in (-0.7, 0.7): D.line(bx + dx, y - 0.5, bx + dx, y + rank_h + 0.5, INK, "glyphs", 0.6)
            ledger.append(dict(panel="".join(s for s, _ in l1) + " " + "".join(s for s, _ in l2), arm=arm, rank=r, node=int(r) // c["rpn"], why=why[r],
                               total_ms=round(m["total"], 2), nic_ms=round(m["inter"], 2), nvlink_ms=round(m["intra"], 2), wait_ms=round(m["wait"], 2),
                               barrier_at_ms=None if m["barrier"] is None else round(m["barrier"], 2), gemm_ms=round(m["gemm"], 2),
                               expert_comm_ms=round(m.get("expert", 0.0), 2), n_pulls=m.get("n_pairs", ""), n_pulls_inter=m.get("n_inter", ""),
                               iteration=it, capsule=c.get("capsule", "")))
            y += rank_h + rank_gap
        ay = y0 + body_h + 3
        D.line(tx0, ay, tx0 + tw, ay, INK2, "axes", 0.5)
        step = V2.nice_step(tmax); t = 0
        while t <= tmax:
            D.line(tx0 + t * sc, ay, tx0 + t * sc, ay + 1.8, INK2, "axes", 0.5)
            if tx0 + t * sc < tx0 + tw - 9: D.text(tx0 + t * sc, ay + 7, f"{t}", 5, "labels", "middle", INK2)
            t += step
        D.text(tx0 + tw, ay + 7, "ms", 5, "labels", "end", INK2)
    # arrows: from the red problem in a title to the subdiagram that resolves it
    for (fp, fl, tp, tl, kind) in ARROWS:
        sx0, sx1, sy = title_pos[(fp, fl)]; dx0, dx1, dy = title_pos[(tp, tl)]
        if kind == "h":
            D.arrow(sx1 + 2.5, sy - 2.2, dx0 - 2.5, dy - 2.2, None, None, ARROW, "glyphs")
        else:
            # arc over the intermediate title(s); lands on the target from above-left
            D.arrow(sx1 + 2.5, sy - 3.0, dx0 - 4.0, dy - 4.8, (sx1 + dx0) / 2, TOP - 12.0, ARROW, "glyphs")
    # legend
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
    if ledger:
        with open(out + "_ranks.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ledger[0].keys())); w.writeheader(); w.writerows(ledger)
    cairo = os.path.expanduser("~/.local/bin/cairosvg")
    if os.path.exists(cairo):
        for fmt, extra in (("png", ["-d", "400"]), ("pdf", [])):
            subprocess.run([cairo, out + ".svg", "-o", f"{out}.{fmt}"] + extra, check=False)
    return panels, ledger, D.h


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=32); ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--moonep-arm", default="bwire", choices=list(MOONEP_ARMS)); ap.add_argument("--rule", default="v1.1")
    a = ap.parse_args()
    V.RULE = a.rule
    data = {"cells": {}}
    for j in a.json:
        cap = os.path.basename(j).replace("phases_", "").replace(".json", "")
        for k, c in json.load(open(j))["cells"].items():
            c["capsule"] = cap; data["cells"][k] = c
    panels, ledger, h = build(data, a.out, a.budget, MOONEP_ARMS[a.moonep_arm], a.ranks)
    print(f"wrote {a.out}.svg/.drawio/_ranks.csv  ({TEXT_W:.0f} x {h:.1f} pt)")
    if UNKNOWN: print(f"note: {len(UNKNOWN)} kernel names outside the wire/expert/GEMM classes were treated as prep (plan/pack torch kernels)")
    for row in ledger: print(f"  {row['arm'][:26]:<26} r{row['rank']:<3} {row['why']:<26} total {row['total_ms']:6.2f} NIC {row['nic_ms']:6.2f} NVL {row['nvlink_ms']:5.2f} wait {row['wait_ms']:6.2f} expert {row['expert_comm_ms']:6.2f} GEMM {row['gemm_ms']:5.2f}")
