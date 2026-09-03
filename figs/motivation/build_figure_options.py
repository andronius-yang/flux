#!/usr/bin/env python3
"""Build the self-contained HTML with three motivation-diagram options from the
extraction JSON (extract_phases.py). Stdlib only; inline SVG; both themes.

Panels: NVSHMEM ring + GEMM (l01_nvshmem) / EPLB (eplb_l01_nvplace) / COMET
(l01_allgather_dense). Layer-0 window per rank = iteration start .. l0 end,
cut from the recorder's own events (plan_comm + plan + l0), so every drawn
span is device time attributed by launch correlation (never host timestamps).
"""
import argparse, html, json, math, statistics as st

CLASSES = {  # class -> (label, light, dark)
    "inter": ("inter-node wire (Slingshot)", "#2a78d6", "#3987e5"),
    "gemm": ("expert GEMM (SM)", "#eb6834", "#d95926"),
    "intra": ("intra-node wire (NVLink P2P)", "#1baf7a", "#199e70"),
    "place": ("expert-weight placement put", "#eda100", "#c98500"),
    "a2a": ("direct a2a dispatch (EPLB: puts + completion barrier; NVLink and Slingshot inseparable)", "#e87ba4", "#d55181"),
    "wait": ("barrier / signal wait", "hatch", "hatch"),
    "prep": ("index / pack / unpack / routing", "#c9c6bd", "#4a4944"),
}
ARMS = [
    ("l01_nvshmem", "A2AV + GEMM", "NVSHMEM blocking-put ring dispatch, then un-overlapped grouped GEMM"),
    ("eplb_l01_nvplace_bwire", "EPLB", "static pool-oracle placement (replicated hot experts), direct a2a dispatch with the wire exposed (one blocking put per destination), per-slot GEMM"),
    ("l01_allgather_dense", "COMET", "dense all-gather: inter-node fetch gate, intra-node P2P, tile-gated fused GEMM"),
]
EPLB_A2A = ("eplb_l01_nvplace", "EPLB, staged a2a kernel", "same placement / pack / place / GEMM; dispatch = All2AllSingle kernel (nbi puts complete inside the barrier)")
IS_EPLB = lambda arm: arm.startswith("eplb")


def classify(name, place=False):
    n = name
    if n.startswith("memcpy10"): return "place" if place else "intra"
    if n.startswith("memcpy"): return "prep"
    if "proxy_rma" in n: return "place" if place else "inter"
    if n.startswith("a2a_single"): return "inter"      # EPLB direct a2a kernel (intra+inter in one launch)
    if "barrier_on_stream" in n or "signal_wait_until" in n: return "wait"
    if n in ("Kernel", "Kernel2") or n.startswith("ep_topk_gather_rs"): return "gemm"
    return "prep"


def merge(segs, gap=0.03):
    """merge consecutive same-class segments separated by < gap ms"""
    out = []
    for s in sorted(segs, key=lambda x: x["t0"]):
        if out and out[-1]["cls"] == s["cls"] and s["t0"] - out[-1]["t1"] < gap:
            out[-1]["t1"] = max(out[-1]["t1"], s["t1"]); out[-1]["n"] += 1
            out[-1]["bytes"] += s.get("bytes", 0)
        else:
            out.append(dict(cls=s["cls"], t0=s["t0"], t1=s["t1"], n=1, bytes=s.get("bytes", 0), name=s["name"]))
    return out


def rank_metrics(cell, rk, it, arm=""):
    """per-rank layer-0 numbers for one timed iteration"""
    rec = cell["recorded"]; iters = sorted(cell["ranks"][next(iter(cell["ranks"]))]["iters"])
    k = iters.index(it)
    cut = sum(rec[m][rk][k] for m in ("plan_comm_ms", "plan_ms", "l0_ms"))
    ev = cell["ranks"][rk]["iters"][it]["events"]
    segs = [dict(cls=classify(e["name"]), t0=e["t0"], t1=e["t1"], bytes=e.get("bytes", 0), name=e["name"]) for e in ev if e["t0"] < cut]
    for s in segs: s["t1"] = min(s["t1"], cut)
    g = [s for s in segs if s["cls"] == "gemm"]
    g_t0 = min([x["t0"] for x in g], default=cut)
    if arm == "eplb_l01_nvplace":  # the staged a2a arm only (the exposed-wire twin keeps inter/intra/wait)
        # the direct-a2a kernel only ISSUES nbi puts; their completion (the
        # actual wire time) is forced inside the following barrier's quiet,
        # so wire + wait before the GEMM are one inseparable class
        for s in segs:
            if s["cls"] in ("inter", "wait") and s["t0"] < g_t0: s["cls"] = "a2a"
    a2a = [s for s in segs if s["cls"] == "a2a"]
    inter = [s for s in segs if s["cls"] == "inter"]
    intra = [s for s in segs if s["cls"] == "intra"]
    wait = [s for s in segs if s["cls"] == "wait"]
    dur = lambda L: sum(x["t1"] - x["t0"] for x in L)
    pre = lambda L: [x for x in L if x["t0"] < g_t0]
    m = dict(cut=cut, gemm=dur(g), gemm_t0=g_t0, gemm_t1=max([x["t1"] for x in g], default=0),
             inter=dur(inter) + dur(a2a), inter_t0=min([x["t0"] for x in inter + a2a], default=0), inter_t1=max([x["t1"] for x in inter + a2a], default=0),
             intra=dur(intra), intra_bytes=sum(x["bytes"] for x in intra), wait=dur(pre(wait)), wait_post=dur([x for x in wait if x["t0"] >= g_t0]),
             segs=merge(segs),
             lanes={"inter": merge(inter + a2a), "intra": merge(intra), "gemm": merge(g), "wait": merge(pre(wait)), "wait_post": merge([x for x in wait if x["t0"] >= g_t0])})
    d = pre(inter + a2a + intra + wait)
    m["dispatch_span"] = (max([x["t1"] for x in d], default=0) - min([x["t0"] for x in pre(inter + a2a + intra)], default=0))
    return m


def place_metrics(cell, rk):
    p = cell["ranks"][rk].get("place")
    if not p: return None
    ev = p["events"]
    wire = [e for e in ev if classify(e["name"], place=True) == "place"]
    inter = [e for e in wire if "proxy" in e["name"]]; intra = [e for e in wire if e["name"].startswith("memcpy10")]
    t0 = min(e["t0"] for e in wire); t1 = max(e["t1"] for e in wire)
    sends = cell["info"].get("eplb_weight_place_sends", {}).get(rk, [])
    W, rpn = cell["W"], cell["rpn"]; r = int(rk)
    inter_b = sum(b for h, _, b in sends if h // rpn != r // rpn); intra_b = sum(b for h, _, b in sends if h // rpn == r // rpn)
    return dict(span=t1 - t0, t0=t0, inter=sum(e["t1"] - e["t0"] for e in inter), intra=sum(e["t1"] - e["t0"] for e in intra),
                n_puts=len(sends), inter_bytes=inter_b, intra_bytes=intra_b, recv_bytes=cell["info"].get("eplb_weight_place_bytes", {}).get(rk, 0),
                segs=merge([dict(cls="place", t0=e["t0"] - t0, t1=e["t1"] - t0, bytes=e.get("bytes", 0), name=e["name"]) for e in wire], gap=0.05),
                lanes={"inter": merge([dict(cls="inter", t0=e["t0"] - t0, t1=e["t1"] - t0, bytes=0, name=e["name"]) for e in inter], 0.05),
                       "intra": merge([dict(cls="intra", t0=e["t0"] - t0, t1=e["t1"] - t0, bytes=e.get("bytes", 0), name=e["name"]) for e in intra], 0.05)})


def fmt_mb(b): return f"{b / 2**20:.0f} MB"
def fmt_gb(b): return f"{b / 2**30:.2f} GB"


def pick_ranks(cell, arm, M, P):
    """4-6 ranks that bracket the spread: extremes of compute, of inter-node wire, (placement)."""
    ranks = sorted(M, key=int)
    def arg(fn, key): return fn(ranks, key=key)
    chosen, why = [], []
    def add(r, reason):
        if r not in chosen and len(chosen) < 6:
            chosen.append(r); why.append((r, reason))
    add(arg(max, lambda r: M[r]["gemm"]), "longest expert GEMM (most routed rows)")
    add(arg(min, lambda r: M[r]["gemm"]), "shortest expert GEMM (fewest routed rows)")
    if IS_EPLB(arm):
        add(arg(max, lambda r: P[r]["span"]), "longest placement (pushes the most replicated experts)")
        add(arg(min, lambda r: P[r]["span"]), "shortest placement")
        if arm.endswith("_bwire"):
            add(arg(max, lambda r: M[r]["inter"]), "longest inter-node dispatch wire occupancy")
            add(arg(min, lambda r: M[r]["inter"]), "shortest inter-node dispatch wire occupancy")
        else:
            add(arg(max, lambda r: M[r]["dispatch_span"]), "longest dispatch (a2a kernel + barrier wait)")
            add(arg(min, lambda r: M[r]["dispatch_span"]), "shortest dispatch")
    else:
        add(arg(max, lambda r: M[r]["inter"]), "longest inter-node wire occupancy")
        add(arg(min, lambda r: M[r]["inter"]), "shortest inter-node wire occupancy")
        med = sorted(ranks, key=lambda r: M[r]["gemm"])[len(ranks) // 2]
        add(med, "median expert GEMM (the typical rank)")
    return sorted(chosen, key=int), dict(why)


# ------------------------------------------------------------------ SVG helpers
def esc(s): return html.escape(str(s), quote=True)

def rect(x, y, w, h, cls, title=None, extra=""):
    fill = f'fill="url(#hatch)"' if cls == "wait" else f'fill="var(--c-{cls})"'
    t = f"<title>{esc(title)}</title>" if title else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0.6):.1f}" height="{h:.1f}" {fill} rx="1" {extra}>{t}</rect>'

def text(x, y, s, cls="lbl", anchor="start", extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}" {extra}>{esc(s)}</text>'

def axis(x0, x1, y, tmax, scale, step):
    out = [f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}" class="ax"/>']
    t = 0
    while t <= tmax + 1e-9:
        x = x0 + t * scale
        out.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y + 4:.1f}" class="ax"/>')
        out.append(text(x, y + 14, f"{t:g}", "tick", "middle"))
        t += step
    return "".join(out)

def nice_step(tmax):
    for s in (0.5, 1, 2, 5, 10, 20, 25, 50, 100):
        if tmax / s <= 8: return s
    return 200


def svg_open(w, h): return f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" xmlns="http://www.w3.org/2000/svg">' + \
    '<defs><pattern id="hatch" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">' \
    '<rect width="6" height="6" fill="var(--c-wait-bg)"/><line x1="0" y1="0" x2="0" y2="6" stroke="var(--c-wait)" stroke-width="1.6"/></pattern></defs>'


def rank_label(cell, r, M, P=None):
    rows = cell["rows_per_rank"][int(r)]
    return f"rank {r} · node {int(r) // cell['rpn']}"


def stat_line(cell, arm, r, M, P):
    rows = cell["rows_per_rank"][int(r)]
    if arm == "l01_nvshmem":
        return f"{rows / 1000:.1f}k rows · {fmt_mb(cell['send_inter_bytes'][int(r)])} inter-node sent · {fmt_mb(cell['recv_inter_bytes'][int(r)])} inter-node recv"
    if arm == "l01_allgather_dense":
        fetch = cell["tokens_per_rank"] * cell["H"] * 2 * (cell["nnodes"] - 1)
        return f"{rows / 1000:.1f}k rows · {fmt_mb(fetch)} inter-node fetched (fixed) · {fmt_mb(M[r]['intra_bytes'])} NVLink"
    gr = cell["info"]["gemm_rows_per_rank"]["0"][int(r)]
    wb = cell["info"]["eplb_wire_bytes"]["0"]; W, rpn = cell["W"], cell["rpn"]; ri = int(r)
    inter_send = sum(wb[ri][d] for d in range(W) if d // rpn != ri // rpn)
    return f"{gr / 1000:.1f}k rows after placement · {fmt_mb(inter_send)} inter-node dispatch · placement {P[r]['n_puts']} puts, {fmt_gb(P[r]['inter_bytes'])} inter-node"


# ------------------------------------------------------------------ option A: phase strips
def option_a(cell, arm, title, M, P, chosen, iter_name):
    LW, RW, W = 150, 330, 1180
    rowh, gap = 22, 8
    place = IS_EPLB(arm)
    PW = 260 if place else 0                       # placement sub-axis width
    x0 = LW + (PW + 40 if place else 0)
    tmax = max(M[r]["cut"] for r in chosen) * 1.02
    scale = (W - RW - x0) / tmax
    h = 60 + len(chosen) * (rowh + gap) + 34
    out = [svg_open(W, h)]
    out.append(text(8, 22, f"{title}", "ttl"))
    out.append(text(8, 40, f"one layer-0 step, {iter_name}, K2 lcb 4n · device time per rank", "sub"))
    y = 60
    if place:
        pmax = max(P[r]["span"] for r in chosen) * 1.02; pscale = PW / pmax
        out.append(text(LW, 54, "one-shot placement (own scale)", "sub"))
        out.append(text(x0, 54, "per-step dispatch → GEMM", "sub"))
    for r in chosen:
        out.append(text(LW - 8, y + 15, rank_label(cell, r, M, P), "lbl", "end"))
        if place and P[r]["span"] * pscale < 1: pass
        if place:
            for s in P[r]["segs"]:
                out.append(rect(LW + s["t0"] * pscale, y, (s["t1"] - s["t0"]) * pscale, rowh, "place",
                                f"{s['n']} put(s), {s['t1'] - s['t0']:.2f} ms"))
            out.append(text(LW + P[r]["span"] * pscale + 4, y + 15, f"{P[r]['span']:.0f} ms", "val"))
            out.append(f'<line x1="{x0 - 20}" y1="{y + 2}" x2="{x0 - 12}" y2="{y + rowh - 2}" class="brk"/><line x1="{x0 - 16}" y1="{y + 2}" x2="{x0 - 8}" y2="{y + rowh - 2}" class="brk"/>')
        for s in M[r]["segs"]:
            if s["cls"] == "prep" and s["t1"] - s["t0"] < 0.15: continue
            out.append(rect(x0 + s["t0"] * scale, y, (s["t1"] - s["t0"]) * scale, rowh, s["cls"],
                            f"{CLASSES[s['cls']][0]}: {s['t0']:.2f}–{s['t1']:.2f} ms ({s['n']} launches)"))
        out.append(text(x0 + M[r]["cut"] * scale + 4, y + 15, f"{M[r]['cut']:.1f} ms", "val"))
        out.append(text(W - RW + 60, y + 15, stat_line(cell, arm, r, M, P), "stat"))
        y += rowh + gap
    out.append(axis(x0, x0 + tmax * scale, y + 2, tmax, scale, nice_step(tmax)))
    out.append(text(x0 + tmax * scale, y + 28, "ms since step start", "tick", "end"))
    if place:
        out.append(axis(LW, LW + pmax * pscale, y + 2, pmax, pscale, nice_step(pmax)))
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------ option B: three lanes per rank
def option_b(cell, arm, title, M, P, chosen, iter_name):
    LW, RW, W = 150, 330, 1180
    laneh, lanegap, rowgap = 9, 3, 12
    lanes = ["inter", "intra", "gemm"]
    place = IS_EPLB(arm)
    PW = 260 if place else 0
    x0 = LW + (PW + 40 if place else 0)
    tmax = max(M[r]["cut"] for r in chosen) * 1.02
    scale = (W - RW - x0) / tmax
    rowh = len(lanes) * (laneh + lanegap)
    h = 66 + len(chosen) * (rowh + rowgap) + 34
    out = [svg_open(W, h), text(8, 22, title, "ttl"), text(8, 40, f"{iter_name} · lanes: NIC (inter-node) / NVLink (intra-node) / SM (GEMM); hatched = barrier wait", "sub")]
    if place:
        pmax = max(P[r]["span"] for r in chosen) * 1.02; pscale = PW / pmax
        out.append(text(LW, 58, "one-shot placement (own scale)", "sub")); out.append(text(x0, 58, "per-step", "sub"))
    y = 66
    for r in chosen:
        out.append(text(LW - 8, y + rowh / 2 + 4, rank_label(cell, r, M, P), "lbl", "end"))
        for li, ln in enumerate(lanes):
            yy = y + li * (laneh + lanegap)
            out.append(f'<line x1="{x0}" y1="{yy + laneh / 2:.1f}" x2="{x0 + tmax * scale:.1f}" y2="{yy + laneh / 2:.1f}" class="lane"/>')
            out.append(text(x0 - 4 if not place else x0 - 4, yy + laneh - 1, {"inter": "NIC", "intra": "NVL", "gemm": "SM"}[ln], "lanelbl", "end"))
            for s in M[r]["lanes"][ln]:
                out.append(rect(x0 + s["t0"] * scale, yy, (s["t1"] - s["t0"]) * scale, laneh, s["cls"], f"{CLASSES[s['cls']][0]} {s['t0']:.2f}–{s['t1']:.2f} ms"))
            for s in M[r]["lanes"]["wait" if ln == "inter" else "wait_post"] if ln in ("inter", "gemm") else []:
                out.append(rect(x0 + s["t0"] * scale, yy, (s["t1"] - s["t0"]) * scale, laneh, "wait", f"wait {s['t0']:.2f}–{s['t1']:.2f} ms"))
            if place and ln in ("inter", "intra"):
                for s in P[r]["lanes"][ln]:
                    out.append(rect(LW + s["t0"] * pscale, yy, (s["t1"] - s["t0"]) * pscale, laneh, ln, f"placement put(s) {s['t0']:.1f}–{s['t1']:.1f} ms"))
        out.append(text(W - RW + 60, y + rowh / 2 + 4, stat_line(cell, arm, r, M, P), "stat"))
        y += rowh + rowgap
    out.append(axis(x0, x0 + tmax * scale, y + 2, tmax, scale, nice_step(tmax)))
    if place: out.append(axis(LW, LW + pmax * pscale, y + 2, pmax, pscale, nice_step(pmax)))
    out.append(text(x0 + tmax * scale, y + 28, "ms since step start", "tick", "end"))
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------ option C: imbalance ledger
def option_c(cells, iter_name):
    """three meters per arm: placement / dispatch wire / expert GEMM — all 16 ranks as dots, chosen ranks ringed"""
    W = 1180; LW = 210; MW = 250; gapx = 40; rowh = 30
    meters = [("place", "expert-weight placement"), ("dispatch", "dispatch (wire + wait)"), ("gemm", "expert GEMM")]
    h = 60 + len(cells) * (rowh + 22) + 40
    out = [svg_open(W, h), text(8, 22, "Where the imbalance sits — every rank, three meters, one binary", "ttl"),
           text(8, 40, f"{iter_name} · dots = 16 ranks (device ms), ring = ranks drawn in the timelines, label = max / mean", "sub")]
    maxes = {}
    for mi, (mk, _) in enumerate(meters):
        vals = []
        for arm, cell, M, P, chosen in cells:
            vals += [v for v in meter_vals(arm, M, P, mk).values()]
        maxes[mk] = max(vals) * 1.08 if vals else 1
    for mi, (mk, ml) in enumerate(meters):
        x = LW + mi * (MW + gapx)
        out.append(text(x, 58, ml, "sub"))
    y = 72
    for arm, cell, M, P, chosen in cells:
        title = [a[1] for a in ARMS if a[0] == arm][0]
        out.append(text(LW - 12, y + 18, title, "lbl", "end"))
        for mi, (mk, ml) in enumerate(meters):
            x = LW + mi * (MW + gapx); vals = meter_vals(arm, M, P, mk)
            if not vals:
                out.append(text(x, y + 18, "— none —", "tick")); continue
            sc = MW / maxes[mk]
            out.append(f'<line x1="{x}" y1="{y + 14}" x2="{x + MW}" y2="{y + 14}" class="lane"/>')
            cls = {"place": "place", "dispatch": "inter", "gemm": "gemm"}[mk]
            for r, v in vals.items():
                ring = 'stroke="var(--ink)" stroke-width="1.5"' if r in chosen else 'stroke="var(--surface)" stroke-width="2"'
                out.append(f'<circle cx="{x + v * sc:.1f}" cy="{y + 14}" r="4.5" fill="var(--c-{cls})" {ring}><title>rank {r}: {v:.2f} ms</title></circle>')
            mean = st.mean(vals.values()); mx = max(vals.values())
            out.append(text(x + MW + 6, y + 18, f"{mx / mean:.2f}×", "val"))
            out.append(text(x, y + 30, f"{min(vals.values()):.1f}–{mx:.1f} ms", "tick"))
        y += rowh + 22
    out.append("</svg>")
    return "".join(out)


def meter_vals(arm, M, P, mk):
    if mk == "place":
        return {r: P[r]["span"] for r in P} if IS_EPLB(arm) and P else {}
    if mk == "dispatch":
        return {r: M[r]["dispatch_span"] for r in M}
    return {r: M[r]["gemm"] for r in M}


# ------------------------------------------------------------------ tables
def table(cell, arm, M, P, chosen, why):
    rows = ["<table><thead><tr><th>rank</th><th>node</th><th>routed rows</th><th>inter-node bytes</th>" +
            ("<th>placement span</th><th>placement puts / inter-node send</th>" if P else "") +
            "<th>dispatch span</th><th>NIC wire (a2a for EPLB)</th><th>NVLink wire</th><th>wait before GEMM</th><th>wait after GEMM</th><th>GEMM0</th><th>layer-0 end</th><th>drawn</th></tr></thead><tbody>"]
    for r in sorted(M, key=int):
        m = M[r]; ri = int(r)
        if IS_EPLB(arm):
            wb = cell["info"]["eplb_wire_bytes"]["0"]; W, rpn = cell["W"], cell["rpn"]
            ib = sum(wb[ri][d] for d in range(W) if d // rpn != ri // rpn); rr = cell["info"]["gemm_rows_per_rank"]["0"][ri]
        elif arm == "l01_allgather_dense":
            ib = cell["tokens_per_rank"] * cell["H"] * 2 * (cell["nnodes"] - 1); rr = cell["rows_per_rank"][ri]
        else:
            ib = cell["send_inter_bytes"][ri]; rr = cell["rows_per_rank"][ri]
        pc = f"<td>{P[r]['span']:.0f} ms</td><td>{P[r]['n_puts']} / {fmt_gb(P[r]['inter_bytes'])}</td>" if P else ""
        mark = f"● {esc(why.get(r, ''))}" if r in chosen else ""
        rows.append(f"<tr class=\"{'sel' if r in chosen else ''}\"><td>{r}</td><td>{ri // cell['rpn']}</td><td>{rr:,}</td><td>{fmt_mb(ib)}</td>{pc}"
                    f"<td>{m['dispatch_span']:.2f}</td><td>{m['inter']:.2f}</td><td>{m['intra']:.2f}</td><td>{m['wait']:.2f}</td><td>{m['wait_post']:.2f}</td><td>{m['gemm']:.2f}</td><td>{m['cut']:.2f}</td><td class=\"why\">{mark}</td></tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def spread(vals):
    v = list(vals); return f"{min(v):.2f}–{max(v):.2f} ms, max/mean {max(v) / st.mean(v):.2f}×"


# ------------------------------------------------------------------ page
CSS = """
:root{color-scheme:light;--surface:#f7f6f2;--panel:#fdfcfa;--ink:#17191c;--ink2:#4f545c;--ink3:#7d8289;--rule:#dad7cf;--accent:#1c5cab;
--c-inter:#2a78d6;--c-gemm:#eb6834;--c-intra:#1baf7a;--c-place:#eda100;--c-a2a:#e87ba4;--c-wait:#8a8f96;--c-wait-bg:#ecebe6;--c-prep:#c9c6bd;--sel:#fff4d6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;--surface:#191a1c;--panel:#212326;--ink:#f1efe9;--ink2:#c3c2b7;--ink3:#8e918f;--rule:#3a3c40;--accent:#7fb2f0;
--c-inter:#3987e5;--c-gemm:#d95926;--c-intra:#199e70;--c-place:#c98500;--c-a2a:#d55181;--c-wait:#a2a6ab;--c-wait-bg:#2c2e32;--c-prep:#4a4944;--sel:#3a3320}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#191a1c;--panel:#212326;--ink:#f1efe9;--ink2:#c3c2b7;--ink3:#8e918f;--rule:#3a3c40;--accent:#7fb2f0;
--c-inter:#3987e5;--c-gemm:#d95926;--c-intra:#199e70;--c-place:#c98500;--c-a2a:#d55181;--c-wait:#a2a6ab;--c-wait-bg:#2c2e32;--c-prep:#4a4944;--sel:#3a3320}
body{background:var(--surface);color:var(--ink);font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;font-size:15px;line-height:1.55;margin:0}
main{max-width:1240px;margin:0 auto;padding:32px 28px 80px}
h1{font-family:"Newsreader","Iowan Old Style",Georgia,serif;font-weight:500;font-size:40px;line-height:1.1;margin:0 0 6px;text-wrap:balance}
h2{font-family:"Newsreader",Georgia,serif;font-weight:500;font-size:27px;margin:44px 0 8px;text-wrap:balance}
h3{font-size:16px;font-weight:600;margin:24px 0 6px}
p,li{max-width:72ch}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3);font-weight:600}
.lede{font-size:18px;color:var(--ink2);max-width:70ch}
.prov{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px 28px;padding:16px 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);margin:22px 0 8px;font-size:13.5px}
.prov b{display:block;color:var(--ink3);font-weight:600;font-size:11.5px;letter-spacing:.06em;text-transform:uppercase}
.mono,code,td,.tick,.val,.stat{font-family:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
code{font-size:13px;background:var(--panel);padding:1px 5px;border-radius:3px}
figure{margin:18px 0 6px;background:var(--panel);border:1px solid var(--rule);padding:14px 12px 8px;border-radius:4px;overflow-x:auto}
figcaption{font-size:13.5px;color:var(--ink2);padding:6px 4px 2px;max-width:none}
svg text{fill:var(--ink)}
svg .ttl{font:600 15px "IBM Plex Sans",system-ui,sans-serif}
svg .sub{font:12.5px "IBM Plex Sans",system-ui,sans-serif;fill:var(--ink2)}
svg .lbl{font:12.5px "IBM Plex Sans",system-ui,sans-serif}
svg .lanelbl{font:9.5px "IBM Plex Mono",monospace;fill:var(--ink3)}
svg .tick{font:11px "IBM Plex Mono",monospace;fill:var(--ink2)}
svg .val{font:11.5px "IBM Plex Mono",monospace;fill:var(--ink)}
svg .stat{font:11px "IBM Plex Mono",monospace;fill:var(--ink2)}
svg .ax{stroke:var(--rule);stroke-width:1}
svg .lane{stroke:var(--rule);stroke-width:1}
svg .brk{stroke:var(--ink3);stroke-width:1.2}
.legend{display:flex;flex-wrap:wrap;gap:8px 22px;font-size:13px;color:var(--ink2);margin:8px 0 0}
.legend span{display:inline-flex;align-items:center;gap:7px}
.sw{width:18px;height:11px;border-radius:2px;display:inline-block}
.sw.wait{background:repeating-linear-gradient(45deg,var(--c-wait) 0 1.5px,var(--c-wait-bg) 1.5px 5px)}
table{border-collapse:collapse;font-size:12.5px;margin:10px 0}
th{text-align:left;font-weight:600;color:var(--ink2);border-bottom:1px solid var(--rule);padding:6px 10px 6px 0;font-family:"IBM Plex Sans",system-ui,sans-serif;white-space:nowrap}
td{padding:4px 10px 4px 0;border-bottom:1px solid var(--rule);white-space:nowrap}
tr.sel td{background:var(--sel)}
td.why{font-family:"IBM Plex Sans",system-ui,sans-serif;white-space:normal;min-width:220px;color:var(--ink2)}
.wrap{overflow-x:auto}
.opt{border-left:3px solid var(--accent);padding-left:16px;margin-top:40px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px 36px}
@media (max-width:800px){.grid2{grid-template-columns:1fr}}
.verdict{background:var(--panel);border:1px solid var(--rule);padding:14px 18px;border-radius:4px;margin:14px 0}
ul{padding-left:20px}
details summary{cursor:pointer;color:var(--accent);font-weight:600}
a{color:var(--accent)}
"""

LEGEND = "".join(f'<span><i class="sw {k}" style="{"" if k == "wait" else f"background:var(--c-{k})"}"></i>{esc(v[0])}</span>' for k, v in CLASSES.items())


def make_panels(data, budget, arms=None):
    cells = {}
    for cid, c in data["cells"].items():
        if c["budget_mib"] == budget and c["status"] == "ok": cells[c["variant"]] = c
    panels = []
    for arm, title, desc in (arms or ARMS):
        if arm not in cells: continue
        cell = cells[arm]
        iters = sorted(cell["ranks"]["0"]["iters"])
        it = iters[len(iters) // 2]
        M = {r: rank_metrics(cell, r, it, arm) for r in cell["ranks"]}
        P = {r: place_metrics(cell, r) for r in cell["ranks"]} if IS_EPLB(arm) else None
        chosen, why = pick_ranks(cell, arm, M, P)
        panels.append((arm, title, desc, cell, M, P, chosen, why, it))
    return panels


def build(data, budget, out_path, capsule_meta, appendix_budget=None):
    panels = make_panels(data, budget)

    def panel_stats(arm, cell, M, P):
        s = [f"expert GEMM {spread(m['gemm'] for m in M.values())}", f"dispatch span {spread(m['dispatch_span'] for m in M.values())}"]
        if arm != "eplb_l01_nvplace": s.append(f"inter-node wire occupancy {spread(m['inter'] for m in M.values())}; barrier wait before GEMM {spread(m['wait'] for m in M.values())}")
        else: s.append(f"a2a kernel + completion barrier {spread(m['inter'] for m in M.values())} (inseparable; per-rank bytes carry the imbalance)")
        if any(m['wait_post'] > 0.05 for m in M.values()): s.append(f"barrier wait after GEMM (absorbs the compute spread) {spread(m['wait_post'] for m in M.values())}")
        if P: s.append(f"placement span {spread(p['span'] for p in P.values())}; inter-node placement send {min(p['inter_bytes'] for p in P.values()) / 2**30:.2f}–{max(p['inter_bytes'] for p in P.values()) / 2**30:.2f} GB per rank")
        return s

    H = []
    H.append(f"<title>Imbalance Shifts</title><style>{CSS}</style>")
    H.append('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">')
    H.append("<main>")
    H.append('<div class="eyebrow">Motivation figure · three options · measured timelines</div>')
    H.append("<h1>Fixing one imbalance moves it somewhere else</h1>")
    H.append('<p class="lede">Three existing MoE dispatch philosophies, one binary, one real routing batch. Every bar is device time from Nsight Systems on Perlmutter A100 nodes; every number beside a rank comes from the same capsule. No optimization of ours appears anywhere on this page.</p>')
    H.append('<div class="prov">' + "".join(f"<div><b>{esc(k)}</b>{esc(v)}</div>" for k, v in capsule_meta.items()) + "</div>")

    H.append("<h2>What the three panels claim, and what the traces say</h2>")
    H.append("<div class=\"grid2\">")
    for arm, title, desc, cell, M, P, chosen, why, it in panels:
        H.append(f"<div><h3>{esc(title)} <span class=\"eyebrow\">{esc(arm)}</span></h3><p>{esc(desc)}.</p><ul>" + "".join(f"<li>{esc(x)}</li>" for x in panel_stats(arm, cell, M, P)) + "</ul></div>")
    H.append("</div>")
    H.append('<div class="verdict"><b>Reading rule.</b> Sender-side dispatch bytes are equal on every rank by construction (each rank routes the same token budget), so dispatch imbalance shows up as <i>receive-side</i> incast and barrier wait, not as unequal send volume. COMET\'s inter-node fetch is byte-identical on every rank; what the trace shows is that it is fully <i>exposed</i> before the GEMM, while the GEMM itself spreads with the routed rows. EPLB flattens the GEMM and pays for it with a one-shot placement whose per-rank cost tracks how many replicas each home rank pushes.</div>')

    H.append("<h2>Legend, common to all options</h2><div class=\"legend\">" + LEGEND + "</div>")
    H.append("<p>Hover any bar for its exact span. The layer-0 window per rank ends at that rank's own recorded <code>l0_end</code> event; device work is attributed to the step by launch correlation, never by host timestamps. Ranks on different nodes are aligned at their step start, which follows a device sync and a world barrier (isolated mode).</p>")

    # ---- option A
    H.append('<section class="opt"><div class="eyebrow">Option A</div><h2>Phase strips — one row per rank, phases end to end</h2>')
    H.append("<p>The most literal reading of the three-panel brief: each rank is a strip of colored phases; the straggler is the longest strip and the imbalance is the ragged right edge. EPLB gets a second, own-scale axis for the one-shot placement so a 150 ms placement does not flatten a 17 ms step.</p>")
    for arm, title, desc, cell, M, P, chosen, why, it in panels:
        H.append(f"<figure>{option_a(cell, arm, title, M, P, chosen, it)}<figcaption><b>{esc(title)}.</b> Ranks {', '.join(chosen)} of 16: " + "; ".join(f"rank {r} = {esc(w)}" for r, w in why.items()) + ".</figcaption></figure>")
    H.append("<div class=\"grid2\"><div><h3>Why it works</h3><ul><li>Maps 1:1 onto the narrative: placement → dispatch → compute, and the reader sees which phase grows on each baseline.</li><li>Per-rank annotations (rows, bytes) sit on the row they explain.</li><li>Renders at quarter-page height per panel in the paper.</li></ul></div><div><h3>Risks</h3><ul><li>Overlap is invisible: COMET's NVLink copies run under its inter-node fetch, and a single strip has to pick one class per instant (the strip draws the launch order; hover shows the truth).</li><li>The own-scale placement axis needs a clear break glyph or reviewers will read it as per-step cost.</li></ul></div></div></section>")

    # ---- option B
    H.append('<section class="opt"><div class="eyebrow">Option B</div><h2>Three lanes per rank — NIC, NVLink, SM</h2>')
    H.append("<p>Same ranks, same window, but each rank is split into the three resources the story is about. Overlap becomes visible as vertical coincidence: COMET's NVLink lane fills while its NIC lane is busy and its SM lane is empty; the ring's SM lane waits for the last NIC put on every rank; EPLB's SM lane is flat across ranks while its placement lanes are not.</p>")
    for arm, title, desc, cell, M, P, chosen, why, it in panels:
        H.append(f"<figure>{option_b(cell, arm, title, M, P, chosen, it)}<figcaption><b>{esc(title)}.</b> Same rank selection as Option A. Hatched marks on the NIC lane are barrier / signal waits.</figcaption></figure>")
    H.append("<div class=\"grid2\"><div><h3>Why it works</h3><ul><li>Shows <i>where</i> the exposed time is: an empty SM lane under a busy NIC lane is the whole COMET argument in one glance.</li><li>Honest about overlap; nothing is hidden by class priority.</li><li>Sets up the paper's later figures, which use the same lane vocabulary.</li></ul></div><div><h3>Risks</h3><ul><li>Three lanes × five ranks × three panels is dense at column width; keep four ranks per panel in print.</li><li>EPLB's direct-a2a kernel mixes NVLink and NIC traffic inside one launch, so its NVLink lane is empty by construction — caption it.</li></ul></div></div></section>")

    # ---- option C
    H.append('<section class="opt"><div class="eyebrow">Option C</div><h2>The imbalance ledger — every rank, three meters</h2>')
    H.append("<p>Drops the timeline and keeps only the thesis: for each baseline, where does the per-rank spread live? All 16 ranks are dots on a shared millisecond scale per meter; the ringed dots are the ranks drawn above, so the two views cross-reference. The max/mean label is the number the text can quote.</p>")
    H.append(f"<figure>{option_c([(arm, cell, M, P, chosen) for arm, title, desc, cell, M, P, chosen, why, it in panels], panels[0][8])}<figcaption>Placement is a one-shot cost (EPLB only) on the same ms scale as the per-step meters — that is the point, not a mistake. Dispatch span = first wire event to last wait before the GEMM.</figcaption></figure>")
    H.append("<div class=\"grid2\"><div><h3>Why it works</h3><ul><li>Uses all 16 ranks, so no reviewer can ask whether the chosen ranks were cherry-picked.</li><li>The horizontal shift of the wide cluster from meter to meter across the three rows <i>is</i> the title of the section.</li><li>Compact: fits in a single column with room for the caption.</li></ul></div><div><h3>Risks</h3><ul><li>Loses the pipeline order; a reader who has not seen a timeline may not know that the dispatch meter precedes the GEMM meter.</li><li>Placement dots dwarf the others; the scale is per meter, which must be stated.</li></ul></div></div></section>")

    # ---- EPLB: exposed wire vs staged a2a kernel, same ranks
    a2a = make_panels(data, budget, [EPLB_A2A])
    bw = [p for p in panels if p[0] == "eplb_l01_nvplace_bwire"]
    if a2a and bw:
        arm, title, desc, cell, M, P, chosen, why, it = bw[0]
        a_arm, a_title, a_desc, a_cell, a_M, a_P, _, _, a_it = a2a[0]
        H.append('<section class="opt"><div class="eyebrow">Why the EPLB panel uses the exposed wire</div><h2>Same EPLB step, two dispatch wires</h2>')
        H.append("<p>The staged All2AllSingle kernel only issues non-blocking device puts; their completion happens inside the following barrier's quiet, so nsys shows one short kernel and one long barrier per rank and no per-destination structure. The side lane replaces only the wire: one blocking put per destination in ring order plus one world barrier, with pack, placement, place and GEMM byte-identical (the bitwise dispatch check passed). Same ranks in both strips.</p>")
        H.append(f"<figure>{option_a(cell, arm, 'EPLB, exposed wire (blocking put per destination)', M, P, chosen, it)}<figcaption>Exposed wire: each blue span is one inter-node put to one destination; NVLink puts are aqua; hatched = waiting at the world barrier for the other ranks.</figcaption></figure>")
        H.append(f"<figure>{option_a(a_cell, a_arm, 'EPLB, staged a2a kernel (what the paper baseline runs)', a_M, a_P, chosen, a_it)}<figcaption>Staged kernel: the same bytes move, but the wire time is inside the barrier's quiet (magenta), so the per-rank structure is invisible. The two captures are separate capsules on the same python-only binary; this comparison is about shape, not latency.</figcaption></figure>")
        H.append("</section>")

    # ---- rank-selection tables
    H.append("<h2>Rank selection and every labelled number</h2>")
    H.append("<p>Selection rule, applied identically per baseline: the two ranks at the extremes of the expert-GEMM duration (most and fewest routed rows), the two at the extremes of the inter-node wire (or, for EPLB, of placement and of dispatch), plus the median-GEMM rank as the typical case. Ranks are drawn from all four nodes; node membership is shown because inter-node incast depends on it. Highlighted rows are the drawn ranks; the last column gives the reason.</p>")
    for arm, title, desc, cell, M, P, chosen, why, it in panels:
        H.append(f"<h3>{esc(title)} — all 16 ranks, {esc(it)}, layer-0 window (ms unless noted)</h3><div class=\"wrap\">{table(cell, arm, M, P, chosen, why)}</div>")
        if P:
            H.append("<p>Placement puts are counted from the sender's ledger (<code>eplb_weight_place_sends</code>); each put is one expert's fc1 + fc2 (56 MB at K2). The placement span is the device time from the first put to the last on that rank, excluding the host-side weight synthesis, which is a harness artifact.</p>")

    if appendix_budget:
        ap_panels = make_panels(data, appendix_budget)
        H.append(f'<section class="opt"><div class="eyebrow">Appendix</div><h2>The same capsule at b{appendix_budget} — does the picture hold at 2× the tokens?</h2>')
        H.append("<p>Same binary, same routing pools, twice the pre-topk budget per rank. Option A strips and the ledger only; rank selection re-applied with the same rule, so the drawn ranks may differ.</p>")
        for arm, title, desc, cell, M, P, chosen, why, it in ap_panels:
            H.append(f"<figure>{option_a(cell, arm, title + f' (b{appendix_budget})', M, P, chosen, it)}<figcaption><b>{esc(title)}, b{appendix_budget}.</b> Ranks {', '.join(chosen)}: " + "; ".join(f"rank {r} = {esc(w)}" for r, w in why.items()) + ".</figcaption></figure>")
        H.append(f"<figure>{option_c([(arm, cell, M, P, chosen) for arm, title, desc, cell, M, P, chosen, why, it in ap_panels], ap_panels[0][8])}<figcaption>Ledger at b{appendix_budget}.</figcaption></figure>")
        for arm, title, desc, cell, M, P, chosen, why, it in ap_panels:
            H.append(f"<details><summary>{esc(title)} b{appendix_budget} — all 16 ranks</summary><div class=\"wrap\">{table(cell, arm, M, P, chosen, why)}</div></details>")
        H.append("</section>")

    H.append("<h2>Method notes</h2><ul>")
    H.append("<li>Capture: one sweep capsule, three arms, <code>modes: [nsys]</code> with <code>FLUX_SWEEP_ISOLATED_ITERS=1</code>; 3 warmup + 3 timed windows; the middle timed window is drawn. Instrumented mode: none of these spans are latency claims (SCHEMA rule 3).</li>")
    H.append("<li>Attribution: kernels and copies are joined to the host <code>iterN</code> NVTX range through their launch correlation id. The host range closes at enqueue (a few ms) while the device runs 15–35 ms, so timestamp filtering would drop most of the step.</li>")
    H.append("<li>Classes: inter-node wire = NVSHMEM proxy RMA kernels (one per remote put / fetch) and EPLB's direct-a2a kernel; NVLink = CUDA P2P copies; GEMM = the grouped-GEMM launches; wait = barrier and signal-wait kernels; everything else (index builds, packing, routing all-gather) is grey.</li>")
    H.append("<li>Bytes: routed rows and per-rank inter-node bytes come from the capsule's traffic matrix (column sums / off-node row sums at 14,336 B per routed token); COMET's inter-node fetch is the fixed remote-shard volume; EPLB's dispatch bytes are the recorded post-placement wire matrix.</li>")
    H.append("<li>EPLB placement wire: blocking NVSHMEM puts from each expert's original home (this capsule is the first with the port); the NCCL twin places identical bytes but shows one opaque SendRecv kernel.</li>")
    H.append("</ul>")
    H.append("</main>")
    open(out_path, "w").write("\n".join(H))
    return panels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+"); ap.add_argument("--budget", type=int, default=16); ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default="{}"); ap.add_argument("--appendix-budget", type=int, default=0)
    a = ap.parse_args()
    data = {"cells": {}}
    for j in a.json:
        data["cells"].update(json.load(open(j))["cells"])
    panels = build(data, a.budget, a.out, json.loads(a.meta), a.appendix_budget or None)
    for arm, title, desc, cell, M, P, chosen, why, it in panels:
        print(title, it, "chosen", chosen, {r: round(M[r]["gemm"], 2) for r in chosen})
