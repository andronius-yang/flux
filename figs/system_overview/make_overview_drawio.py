#!/usr/bin/env python3
"""System-overview figure generator (figs/system_overview, REV 0.3).

Emits `moe_overview.drawio` (uncompressed draw.io XML, hand-editable) and
`moe_overview_preview.png/.pdf` (matplotlib render of the SAME primitives,
review only). All geometry lives in KNOBS; routing = brief §12 (confirmed).

draw.io unit = 0.75 pt. Canvas ~654 x 215 units -> 6.8 x 2.2 in at scale 1.
Run: $PSCRATCH/conda_envs/andrewy-comet/bin/python make_overview_drawio.py
"""
import xml.sax.saxutils as su

# ----------------------------------------------------------------- KNOBS
K = dict(
    Y0=6, ROW_H=40, NODE_PAD=2, NODE_GAP=3, SLOT_Y0=3, SLOT_PITCH=12,
    EXP_H=10, EXP_W=54, EXP_X=272, TOK_W=14, TOK_H=10, HOME_X=48,
    TOK_Y=(8, 24),                       # two home tokens per GPU (row-relative)
    GW_X=205, GW_S=8,                    # gateway junction square
    GHOST_X=208,                         # ghost "would-land" slot (E4 row, GPU2)
    SIG_W=12, SIG_X={2: 396, 3: 414},    # pre-reduce boxes (per GPU)
    PART_X=560, OUT_X=606, OUT_OFF=9, BADGE=9, RING_R=8,
    NODE_X=12, NODE_W=632, GPU_LABEL_W=30,
    # dispatch gutter lanes (x); one in-flight glyph per NIC crossing
    LANES_DISPATCH=dict(A1=76, A1g=84, A2=104, B1=124, C2=152, D2=172),
    LANES_FWD=dict(A2=216, B1c=222, B1f=224),
    # combine gutter lanes
    LANES_COMBINE=dict(A1p=376, B1own=410, A2h=466, B1h=486, C2h=506, D2h=526),
    FONT_EXP=9, FONT_TOK=8, FONT_GPU=9, FONT_NODE=9, FONT_PHASE=10, FONT_LEG=8,
    STROKE_INTRA=1.0, STROKE_INTER=1.2, DASH_INTRA="0.7 1", DASH_INTER="4 2",
    COL=dict(A="#FFD966", B="#F19C99", C="#9999FF", D="#D4E1F5"),
    LINE=dict(A="#D9A400", B="#E4736E", C="#7070E8", D="#5B9BD5"),
    EXP_FILL="#E9F8E4", REPLICA_STROKE="#2E7D32", WEIGHT_COL="#7F7F7F",
    GHOST_COL="#9E9E9E", GW_FILL="#D9D9D9", SIG_FILL="#FFFFFF",
)

SLOTS = {
    0: [("Expert 0", "exp"), ("Expert 1", "exp"), ("", "free")],
    1: [("Expert 2", "exp"), ("Expert 3", "exp"), ("Expert 4'", "replica")],
    2: [("Expert 4", "exp"), ("Expert 5", "exp"), ("", "free")],
    3: [("Expert 6", "exp"), ("Expert 7", "exp"), ("", "free")],
}
HOME = {0: ["A1", "A2"], 1: ["B1", "B2"], 2: ["C1", "C2"], 3: ["D1", "D2"]}
# input stacks (gpu, slot) -> tokens nearest-first; the ONE wire-arriving token is LAST (leftmost)
IN = {(0, 0): ["A1"], (0, 1): ["C2"],
      (1, 0): ["B2"], (1, 1): ["D2"], (1, 2): ["B2", "A1"],
      (2, 0): ["C1", "B1"], (2, 1): ["C1", "C2", "A2"],
      (3, 0): ["D1", "B1"], (3, 1): ["D1", "D2", "A2"]}
OUT = IN  # the same rows leave each expert as partials
# home partial stacks: gpu -> token -> partial labels LEFTMOST-first (wire-arriving first)
PART = {0: {"A1": ["E4'", "E0"], "A2": ["Σ"]},
        1: {"B1": ["Σ"], "B2": ["E4'", "E2"]},
        2: {"C1": ["E4", "E5"], "C2": ["E1", "E5"]},
        3: {"D1": ["E6", "E7"], "D2": ["E3", "E7"]}}

# ------------------------------------------------------------ geometry
def node_y(n): return K["Y0"] + n * (2 * K["ROW_H"] + 2 * K["NODE_PAD"] + K["NODE_GAP"])
def gpu_y(r): return node_y(r // 2) + K["NODE_PAD"] + (r % 2) * K["ROW_H"]
def slot_y(r, s): return gpu_y(r) + K["SLOT_Y0"] + s * K["SLOT_PITCH"]
def home_tok(r, k): return (K["HOME_X"], gpu_y(r) + K["TOK_Y"][k])
def in_tok(r, s, j): return (K["EXP_X"] - 2 - K["TOK_W"] * (j + 1), slot_y(r, s))
def out_tok(r, s, j): return (K["EXP_X"] + K["EXP_W"] + K["OUT_OFF"] + K["TOK_W"] * j, slot_y(r, s))
def gw(r): return (K["GW_X"], gpu_y(r) + K["ROW_H"] // 2 - K["GW_S"] // 2)
def sig(r): return (K["SIG_X"][r], gpu_y(r) + K["ROW_H"] // 2 - K["EXP_H"] // 2)
def part_tok(r, k, j): return (K["PART_X"] + (K["TOK_W"] + 1) * j, gpu_y(r) + K["TOK_Y"][k])
def out_home(r, k): return (K["OUT_X"], gpu_y(r) + K["TOK_Y"][k])
BOUNDARY_Y = (node_y(0) + 2 * K["ROW_H"] + 2 * K["NODE_PAD"] + node_y(1)) / 2   # between the nodes

# ------------------------------------------------------------ primitives
P, E, _ids = [], [], {}
def nid(key):
    _ids[key] = _ids.get(key, f"c{len(_ids)+1}"); return _ids[key]
def rect(key, x, y, w, h, label="", kind="tok", color=None):
    P.append(dict(id=nid(key), x=x, y=y, w=w, h=h, label=label, kind=kind, color=color))
def text(key, x, y, w, h, label, size, rot=0, align="center", color="#000000", style=""):
    P.append(dict(id=nid(key), x=x, y=y, w=w, h=h, label=label, kind="text",
                  size=size, rot=rot, align=align, color=color, style=style))
def edge(key, src, tgt, points, kind, color="#000000", exit_="R", entry="L"):
    E.append(dict(id=nid(key), src=nid(src), tgt=nid(tgt), points=points,
                  kind=kind, color=color, exit=exit_, entry=entry))
def geom(key):
    return next(q for q in P if q["id"] == _ids[key])
def cy(key):
    p = geom(key); return p["y"] + p["h"] / 2
def lane_edge(key, src, tgt, lane, kind, color, entry="L"):
    pts = [(lane, cy(src)), (lane, cy(tgt))] if entry == "L" else \
          [(lane, cy(src)), (lane, geom(tgt)["y"] + (K["EXP_H"] if entry == "B" else 0))]
    edge(key, src, tgt, pts, kind, color, entry=entry)
def glyph(key, label, lane, color):
    """in-flight token glyph straddling the node boundary on lane x (ghost when color is None)"""
    rect(key, lane - K["TOK_W"] / 2, BOUNDARY_Y - K["TOK_H"] / 2, K["TOK_W"], K["TOK_H"],
         label, kind="tok" if color else "ghost", color=color)

W = K["NODE_W"]
for n in range(2):
    y = node_y(n); h = 2 * K["ROW_H"] + 2 * K["NODE_PAD"]
    rect(f"node{n}", K["NODE_X"], y, W, h, kind="node")
    text(f"nodelab{n}", 0, y + h / 2 - 20, 12, 40, f"Node{n}", K["FONT_NODE"], rot=-90, style="bi")
    for r in (2 * n, 2 * n + 1):
        rect(f"gpu{r}", K["NODE_X"] + 2, gpu_y(r), W - 4, K["ROW_H"], kind="gpu")
        text(f"gpulab{r}", K["NODE_X"] + 4, gpu_y(r) + 2, K["GPU_LABEL_W"], 12,
             f"GPU{r}", K["FONT_GPU"], align="left", style="bi")
for r, slots in SLOTS.items():
    for s, (lab, kind) in enumerate(slots):
        rect(f"exp{r}{s}", K["EXP_X"], slot_y(r, s), K["EXP_W"], K["EXP_H"], lab, kind=kind)
for r, toks in HOME.items():
    for k, t in enumerate(toks):
        x, y = home_tok(r, k); rect(f"home_{t}", x, y, K["TOK_W"], K["TOK_H"], t, color=K["COL"][t[0]])
for (r, s), toks in IN.items():
    for j, t in enumerate(toks):
        x, y = in_tok(r, s, j); rect(f"in_{t}_{r}{s}", x, y, K["TOK_W"], K["TOK_H"], t, color=K["COL"][t[0]])
for (r, s), toks in OUT.items():
    for j, t in enumerate(toks):
        x, y = out_tok(r, s, j); rect(f"out_{t}_{r}{s}", x, y, K["TOK_W"], K["TOK_H"], t + "'", color=K["COL"][t[0]])
for r in range(4):
    x, y = gw(r); rect(f"gw{r}", x, y, K["GW_S"], K["GW_S"], kind="gw")
for r in (2, 3):
    x, y = sig(r); rect(f"sig{r}", x, y, K["SIG_W"], K["EXP_H"], "Σ", kind="sig")
# ghost slot: where A1 would have landed without the replica (E4 row on GPU2)
rect("ghost_A1", K["GHOST_X"], slot_y(2, 0), K["TOK_W"], K["TOK_H"], "A1", kind="ghost")
for r, toks in PART.items():
    for k, (t, srcs) in enumerate(toks.items()):
        for j, _ in enumerate(srcs):
            x, y = part_tok(r, k, j); rect(f"part_{t}_{j}", x, y, K["TOK_W"], K["TOK_H"], t + "'", color=K["COL"][t[0]])
        x, y = out_home(r, k); rect(f"outh_{t}", x, y, K["TOK_W"], K["TOK_H"], t, color=K["COL"][t[0]])
        edge(f"red_{t}", f"part_{t}_{len(srcs)-1}", f"outh_{t}", [], "solid")

# ------------------------------------------------------------ edges
LD, LF, LC, LN = K["LANES_DISPATCH"], K["LANES_FWD"], K["LANES_COMBINE"], K["LINE"]
# (1) LocCap reroute: actual dotted hop to the replica + grey ghost path to E4 on GPU2
lane_edge("d_A1", "home_A1", "in_A1_12", LD["A1"], "intra", LN["A"])
lane_edge("d_A1g", "home_A1", "ghost_A1", LD["A1g"], "ghost", K["GHOST_COL"])
glyph("gl_A1g", "A1", LD["A1g"], None)
# (2) dispatch over the NIC: one row per (token, node) to the same-lr gateway; consume + forward
lane_edge("d_A2", "home_A2", "gw2", LD["A2"], "inter", LN["A"]);  glyph("gl_A2", "A2", LD["A2"], K["COL"]["A"])
edge("k_A2", "gw2", "in_A2_21", [], "intra", LN["A"])                                  # consumed at E5 (GPU2)
lane_edge("f_A2", "gw2", "in_A2_31", LF["A2"], "intra", LN["A"])                       # forwarded to E7 (GPU3)
lane_edge("d_B1", "home_B1", "gw3", LD["B1"], "inter", LN["B"]);  glyph("gl_B1", "B1", LD["B1"], K["COL"]["B"])
lane_edge("k_B1", "gw3", "in_B1_30", LF["B1c"], "intra", LN["B"])                      # consumed at E6 (GPU3)
lane_edge("f_B1", "gw3", "in_B1_20", LF["B1f"], "intra", LN["B"])                      # forwarded to E4 (GPU2)
lane_edge("d_C2", "home_C2", "gw0", LD["C2"], "inter", LN["C"]);  glyph("gl_C2", "C2", LD["C2"], K["COL"]["C"])
edge("k_C2", "gw0", "in_C2_01", [], "intra", LN["C"])
lane_edge("d_D2", "home_D2", "gw1", LD["D2"], "inter", LN["D"]);  glyph("gl_D2", "D2", LD["D2"], K["COL"]["D"])
edge("k_D2", "gw1", "in_D2_11", [], "intra", LN["D"])
# combine, intra-node partial straight home
lane_edge("c_A1", "out_A1_12", "part_A1_0", LC["A1p"], "intra", LN["A"])
# (3) pre-combine: own partial (left edge) + sibling partial (top/bottom) into Σ, one row over the NIC
edge("c_A2own", "out_A2_21", "sig2", [], "intra", LN["A"])
lane_edge("c_A2sib", "out_A2_31", "sig2", K["SIG_X"][2] + K["SIG_W"] / 2, "intra", LN["A"], entry="B")
lane_edge("h_A2", "sig2", "part_A2_0", LC["A2h"], "inter", LN["A"]);  glyph("gl_A2p", "A2'", LC["A2h"], K["COL"]["A"])
lane_edge("c_B1own", "out_B1_30", "sig3", LC["B1own"], "intra", LN["B"])
lane_edge("c_B1sib", "out_B1_20", "sig3", K["SIG_X"][3] + K["SIG_W"] / 2, "intra", LN["B"], entry="T")
lane_edge("h_B1", "sig3", "part_B1_0", LC["B1h"], "inter", LN["B"]);  glyph("gl_B1p", "B1'", LC["B1h"], K["COL"]["B"])
lane_edge("h_C2", "out_C2_01", "part_C2_0", LC["C2h"], "inter", LN["C"]);  glyph("gl_C2p", "C2'", LC["C2h"], K["COL"]["C"])
lane_edge("h_D2", "out_D2_11", "part_D2_0", LC["D2h"], "inter", LN["D"]);  glyph("gl_D2p", "D2'", LC["D2h"], K["COL"]["D"])
# expert weights: setup replica fill E4 -> E4'; per-iteration intra-node swap E5 <-> E6 (bracket)
edge("w_rep", "exp20", "exp12", [], "weight", K["WEIGHT_COL"], exit_="T", entry="B")
BX = K["EXP_X"] + K["EXP_W"] + 4
edge("w_swap", "exp21", "exp30", [(BX, slot_y(2, 1) + K["EXP_H"] / 2), (BX, slot_y(3, 0) + K["EXP_H"] / 2)],
     "swap", K["WEIGHT_COL"], exit_="R", entry="R")

# ------------------------------------------------------------ labels
H = node_y(1) + 2 * K["ROW_H"] + 2 * K["NODE_PAD"] + 2
text("ph_disp", 60, H, 160, 12, "Dispatch", K["FONT_PHASE"], style="i")
text("ph_comp", 200, H, 200, 12, "Compute", K["FONT_PHASE"], style="i")
text("ph_comb", 400, H, 240, 12, "Combine + Reduce", K["FONT_PHASE"], style="i")
B = K["BADGE"]
def tag(key, x, y, n):   # (x, y) = badge centre
    rect(key, x - B / 2, y - B / 2, B, B, str(n), kind="tag")
RX, RY = LD["A1"] + 5, cy("home_A1")
rect("ring1", RX - K["RING_R"], RY - K["RING_R"], 2 * K["RING_R"], 2 * K["RING_R"], kind="ring")
tag("t1", RX + K["RING_R"] + 3, RY - K["RING_R"] + 1, 1)                     # at the reroute fork
tag("t2a", K["GW_X"] + K["GW_S"] / 2, gw(2)[1] + K["GW_S"] + 5, 2)           # below gateway GPU2
tag("t2b", K["SIG_X"][2] + K["SIG_W"] + 4, sig(2)[1] + 13, 2)                # right-below Σ on GPU2
tag("t3", BX, (slot_y(2, 1) + slot_y(3, 0) + K["EXP_H"]) / 2, 3)             # mid swap bracket
LEG_Y = H + 13
leg = [("intra", "NVLink"), ("inter", "NIC"), ("ghost", "route w/o replica"),
       ("weight", "expert weights"), ("gw", "gateway (pack / forward)"), ("sig", "pre-reduce")]
x = 14
for kind, lab in leg:
    if kind in ("intra", "inter", "weight", "ghost"):
        rect(f"legl_{kind}", x, LEG_Y + 4, 16, 0, kind=f"legline_{kind}"); x += 19
    elif kind == "gw":
        rect("leg_gw", x + 3, LEG_Y + 1, K["GW_S"], K["GW_S"], kind="gw"); x += 14
    else:
        rect("leg_sig", x, LEG_Y, K["SIG_W"], K["EXP_H"], "Σ", kind="sig"); x += 15
    wlab = 5.2 * len(lab)
    text(f"legt_{kind}", x, LEG_Y - 1, wlab, 12, lab, K["FONT_LEG"], align="left"); x += wlab + 6
text("legt_tags", 14, LEG_Y + 11, 600, 12,
     "①  expert placement & routing      ②  token comm comp overlap      ③  expert-dispatch overlap",
     K["FONT_LEG"], align="left")
CANVAS_H = LEG_Y + 23

# ------------------------------------------------------------ draw.io XML
def style_rect(p):
    k = p["kind"]; fe = K["FONT_EXP"]
    return {
        "node": "rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#000000;strokeWidth=1;",
        "gpu": "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#000000;strokeWidth=0.7;dashed=1;dashPattern=2 1;",
        "exp": f"rounded=1;whiteSpace=wrap;html=1;fillColor={K['EXP_FILL']};strokeColor=#000000;strokeWidth=1;fontSize={fe};fontStyle=2;",
        "replica": f"rounded=1;whiteSpace=wrap;html=1;fillColor={K['EXP_FILL']};strokeColor={K['REPLICA_STROKE']};strokeWidth=1.5;fontSize={fe};fontStyle=3;",
        "free": "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#9E9E9E;strokeWidth=0.7;dashed=1;",
        "tok": f"rounded=0;whiteSpace=wrap;html=1;fillColor={p['color']};strokeColor=#000000;strokeWidth=0.5;fontSize={K['FONT_TOK']};spacing=0;spacingLeft=0;spacingRight=0;",
        "ghost": f"rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor={K['GHOST_COL']};strokeWidth=0.7;dashed=1;dashPattern=2 1;fontColor={K['GHOST_COL']};fontSize={K['FONT_TOK']};spacing=0;",
        "gw": f"rounded=0;whiteSpace=wrap;html=1;fillColor={K['GW_FILL']};strokeColor=#000000;strokeWidth=0.7;",
        "sig": f"rounded=0;whiteSpace=wrap;html=1;fillColor={K['SIG_FILL']};strokeColor=#000000;strokeWidth=1;fontSize={fe};",
        "tag": f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#000000;strokeColor=#FFFFFF;strokeWidth=0.6;fontColor=#FFFFFF;fontSize={K['FONT_TOK']};fontStyle=1;spacing=0;",
        "ring": "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=none;strokeColor=#000000;strokeWidth=0.8;",
    }[k]
def style_edge(e):
    k = e["kind"]
    ex = {"R": "exitX=1;exitY=0.5;", "T": "exitX=0.5;exitY=0;", "B": "exitX=0.5;exitY=1;"}[e["exit"]]
    en = {"L": "entryX=0;entryY=0.5;", "R": "entryX=1;entryY=0.5;", "T": "entryX=0.5;entryY=0;", "B": "entryX=0.5;entryY=1;"}[e["entry"]]
    s = f"edgeStyle=none;rounded=0;html=1;{ex}{en}strokeColor={e['color']};"
    return s + {
        "intra": f"endArrow=none;dashed=1;dashPattern={K['DASH_INTRA']};strokeWidth={K['STROKE_INTRA']};",
        "inter": f"endArrow=none;dashed=1;dashPattern={K['DASH_INTER']};strokeWidth={K['STROKE_INTER']};",
        "ghost": f"endArrow=none;dashed=1;dashPattern={K['DASH_INTER']};strokeWidth=0.8;",
        "weight": "endArrow=open;endFill=0;strokeWidth=1.2;",
        "swap": "endArrow=open;startArrow=open;endFill=0;startFill=0;strokeWidth=1.2;",
    }.get(k, "endArrow=none;strokeWidth=0.5;")

def write_drawio(path):
    out = ['<mxfile host="make_overview_drawio.py">', '  <diagram name="overview" id="sysov">',
           f'    <mxGraphModel dx="800" dy="400" grid="1" gridSize="2" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{int(K["NODE_X"]+W+10)}" pageHeight="{int(CANVAS_H)}" math="0" shadow="0">',
           '      <root>', '        <mxCell id="0" />', '        <mxCell id="1" parent="0" />']
    for p in P:
        if p["kind"] == "text":
            fs = {"": "", "i": "fontStyle=2;", "bi": "fontStyle=3;"}[p["style"]]
            st = (f"text;html=1;align={p['align']};verticalAlign=middle;strokeColor=none;fillColor=none;"
                  f"fontSize={p['size']};fontColor={p['color']};rotation={p['rot']};spacing=0;spacingLeft=1;{fs}")
        elif p["kind"].startswith("legline_"):
            kk = p["kind"].split("_")[1]
            col = K["GHOST_COL"] if kk == "ghost" else "#000000"
            st = style_edge(dict(kind=kk, color=col, exit="R", entry="L")).replace("exitX=1;exitY=0.5;entryX=0;entryY=0.5;", "")
            out.append(f'        <mxCell id="{p["id"]}" style="{st}" edge="1" parent="1">')
            out.append(f'          <mxGeometry width="50" height="50" relative="1" as="geometry"><mxPoint x="{p["x"]}" y="{p["y"]}" as="sourcePoint" /><mxPoint x="{p["x"]+p["w"]}" y="{p["y"]}" as="targetPoint" /></mxGeometry>')
            out.append('        </mxCell>'); continue
        else:
            st = style_rect(p)
        out.append(f'        <mxCell id="{p["id"]}" value="{su.escape(p["label"], {chr(34): "&quot;"})}" style="{st}" vertex="1" parent="1">')
        out.append(f'          <mxGeometry x="{p["x"]}" y="{p["y"]}" width="{p["w"]}" height="{p["h"]}" as="geometry" />')
        out.append('        </mxCell>')
    for e in E:
        out.append(f'        <mxCell id="{e["id"]}" style="{style_edge(e)}" edge="1" parent="1" source="{e["src"]}" target="{e["tgt"]}">')
        if e["points"]:
            pts = "".join(f'<mxPoint x="{x}" y="{y}" />' for x, y in e["points"])
            out.append(f'          <mxGeometry relative="1" as="geometry"><Array as="points">{pts}</Array></mxGeometry>')
        else:
            out.append('          <mxGeometry relative="1" as="geometry" />')
        out.append('        </mxCell>')
    out += ['      </root>', '    </mxGraphModel>', '  </diagram>', '</mxfile>']
    open(path, "w").write("\n".join(out) + "\n")

# ------------------------------------------------------------ preview
def preview(path_png, path_pdf):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Ellipse
    U = 0.75 / 72.0
    cw = K["NODE_X"] + W + 10
    fig = plt.figure(figsize=(cw * U, CANVAS_H * U))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, cw); ax.set_ylim(CANVAS_H, 0); ax.axis("off")
    byid = {p["id"]: p for p in P}
    def anchor(pid, side):
        p = byid[pid]
        return {"R": (p["x"] + p["w"], p["y"] + p["h"] / 2), "L": (p["x"], p["y"] + p["h"] / 2),
                "T": (p["x"] + p["w"] / 2, p["y"]), "B": (p["x"] + p["w"] / 2, p["y"] + p["h"])}[side]
    LS = {"intra": (0, (0.7, 1)), "inter": (0, (4, 2)), "ghost": (0, (4, 2))}
    # edges first (glyphs and boxes paint over them), then vertices
    for e in E:
        s = anchor(e["src"], e["exit"]); t = anchor(e["tgt"], e["entry"])
        xs = [s[0]] + [q[0] for q in e["points"]] + [t[0]]; ys = [s[1]] + [q[1] for q in e["points"]] + [t[1]]
        k = e["kind"]
        lw = {"intra": K["STROKE_INTRA"], "inter": K["STROKE_INTER"], "ghost": 0.8, "solid": 0.5}.get(k, 1.2)
        ax.plot(xs, ys, color=e["color"], lw=lw, linestyle=LS.get(k, "solid"), solid_capstyle="butt", zorder=1)
        if k in ("weight", "swap"):
            ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]), arrowprops=dict(arrowstyle="->", color=e["color"], lw=1.2, shrinkA=0, shrinkB=0))
            if k == "swap":
                ax.annotate("", xy=(xs[0], ys[0]), xytext=(xs[1], ys[1]), arrowprops=dict(arrowstyle="->", color=e["color"], lw=1.2, shrinkA=0, shrinkB=0))
    for p in P:
        k = p["kind"]
        if k == "text":
            ha = {"center": "center", "left": "left"}[p["align"]]
            ax.text(p["x"] + (p["w"] / 2 if ha == "center" else 1), p["y"] + p["h"] / 2, p["label"],
                    fontsize=p["size"] * 0.75, ha=ha, va="center", rotation=p["rot"], color=p["color"],
                    style="italic" if "i" in p["style"] else "normal", weight="bold" if "b" in p["style"] else "normal", zorder=3)
            continue
        if k.startswith("legline_"):
            kk = k.split("_")[1]
            ax.plot([p["x"], p["x"] + p["w"]], [p["y"], p["y"]], color=K["GHOST_COL"] if kk == "ghost" else "#000000",
                    lw=1.0, linestyle=LS.get(kk, "solid")); continue
        fc = {"node": "none", "gpu": "none", "exp": K["EXP_FILL"], "replica": K["EXP_FILL"], "free": "none", "tok": p["color"] or "white",
              "ghost": "white", "gw": K["GW_FILL"], "sig": K["SIG_FILL"], "tag": "#000000", "ring": "none"}[k]
        ec = {"node": "#000000", "gpu": "#000000", "exp": "#000000", "replica": K["REPLICA_STROKE"], "free": "#9E9E9E", "tok": "#000000",
              "ghost": K["GHOST_COL"], "gw": "#000000", "sig": "#000000", "tag": "white", "ring": "#000000"}[k]
        lw = {"node": 1, "gpu": 0.7, "exp": 1, "replica": 1.5, "free": 0.7, "tok": 0.5, "ghost": 0.7, "gw": 0.7, "sig": 1, "tag": 0.6, "ring": 0.8}[k]
        ls = {"gpu": (0, (2, 1)), "free": (0, (2, 1)), "ghost": (0, (2, 1))}.get(k, "solid")
        z = 2 if k not in ("node", "gpu") else 0
        if k in ("tag", "ring"):
            ax.add_patch(Ellipse((p["x"] + p["w"] / 2, p["y"] + p["h"] / 2), p["w"], p["h"], facecolor=fc, edgecolor=ec, lw=lw, zorder=z + 1))
            if p["label"]:
                ax.text(p["x"] + p["w"] / 2, p["y"] + p["h"] / 2, p["label"], fontsize=K["FONT_TOK"] * 0.75, ha="center", va="center", color="white", weight="bold", zorder=z + 2)
            continue
        ax.add_patch(Rectangle((p["x"], p["y"]), p["w"], p["h"], facecolor=fc, edgecolor=ec, lw=lw, linestyle=ls, zorder=z))
        if p["label"]:
            fs = {"exp": K["FONT_EXP"], "replica": K["FONT_EXP"], "tok": K["FONT_TOK"], "ghost": K["FONT_TOK"], "sig": K["FONT_EXP"]}.get(k, 8) * 0.75
            ax.text(p["x"] + p["w"] / 2, p["y"] + p["h"] / 2, p["label"], fontsize=fs, ha="center", va="center",
                    style="italic" if k in ("exp", "replica") else "normal", weight="bold" if k == "replica" else "normal",
                    color=K["GHOST_COL"] if k == "ghost" else "#000000", zorder=z + 1)
    fig.savefig(path_png, dpi=300); fig.savefig(path_pdf)
    print(f"canvas {cw} x {CANVAS_H} units = {cw*U:.2f} x {CANVAS_H*U:.2f} in")

if __name__ == "__main__":
    import os
    d = os.path.dirname(os.path.abspath(__file__))
    write_drawio(os.path.join(d, "moe_overview.drawio"))
    preview(os.path.join(d, "moe_overview_preview.png"), os.path.join(d, "moe_overview_preview.pdf"))
