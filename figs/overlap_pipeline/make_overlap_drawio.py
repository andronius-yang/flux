#!/usr/bin/env python
"""Draft generator for the token-comm/comp overlap figure (figs/overlap_pipeline).

Emits overlap_pipeline.drawio with two pages, `dispatch_v1` and `combine_v1`,
in the glyph vocabulary of sys.drawio/sp_mg_v3 (token squares, expert boxes,
node/GPU containers, per-resource timeline rows: NIC RDMA solid, NVLink
dashed, GPU dotted) plus the Sysv5/CSv1 timeline palette.  Every timeline
bar is placed from the MEASURED 4n K2 b64 medians recorded in 00_brief.md
§3 (efficient case, capsule 20260905-141104, rank 0 / node 0 for the
arrival order); the topology panels are schematic.

Pure XML, no draw.io CLI needed; review with render_drawio.py.
"""
import html
import xml.sax.saxutils as sx

# ---------------------------------------------------------------- [knobs]
PX_PER_MS = 15.0          # timeline scale
ROW_H = 13.0              # timeline row pitch
BAR_H = 6.67              # bar height (sp_mg_v3)
LABEL_W = 92.0            # row-label column
TL_X0 = 470.0             # timeline block x origin (topology panel to the left)
# palette (Sysv5 / CSv1 / sp_mg_v3)
C_TOKEN = '#2a78d6'       # token comm (NIC/NVLink bars)
C_COMP = '#eda100'        # expert compute
C_PLAN = '#2f8f9d'        # plan / meta
C_REDUCE = '#c8553d'      # top-k reduce / Σ pre-reduce (hatched)
C_WAIT = '#c9c8c0'
C_EXPERT_COMM = '#1baf7a'
TOK_BLUE, TOK_RED, TOK_YEL, TOK_LBLUE = '#9999FF', '#F19C99', '#FFD966', '#D4E1F5'
EXPERT_FILL = '#E9F8E4'
NODE_FILL, NODE_STROKE = '#f5f5f5', '#666666'
LINE = '#7d8289'

S_TEXT = 'text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align={a};verticalAlign=middle;rounded=0;fontSize={fs};fontStyle={fst};fontColor={fc};'
S_NODE = f'rounded=0;whiteSpace=wrap;html=1;fillColor={NODE_FILL};fontColor=#333333;strokeColor={NODE_STROKE};'
S_GPU = 'rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#000000;'
S_EXPERT = f'rounded=1;whiteSpace=wrap;html=1;fillColor={EXPERT_FILL};strokeWidth=1;fontSize=11;strokeColor=#000000;fontStyle=2;'
S_TOKEN = 'rounded=0;whiteSpace=wrap;html=1;fillColor={c};strokeColor=#000000;strokeWidth=0.75;'
S_TOKEN_REMOVED = 'rounded=0;whiteSpace=wrap;html=1;fillColor={c};strokeColor=#000000;strokeWidth=0.75;fillStyle=zigzag-line;'
S_BAR = 'rounded=0;whiteSpace=wrap;html=1;fillColor={c};strokeColor=none;fontSize=8;fontColor=#ffffff;'
S_BAR_SHIFT = 'rounded=0;whiteSpace=wrap;html=1;fillColor={c};strokeColor=#000000;strokeWidth=0.8;fillStyle=zigzag-line;fontSize=8;'
S_BAR_OUTLINE = 'rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor={c};strokeWidth=1;dashed=1;dashPattern=2 1;fontSize=8;fontColor={c};'
S_LINE_NIC = f'endArrow=none;html=1;strokeColor={LINE};strokeWidth=0.67;'
S_LINE_NVL = f'endArrow=none;html=1;strokeColor={LINE};strokeWidth=0.67;dashed=1;dashPattern=2.7 2.0;'
S_LINE_GPU = f'endArrow=none;html=1;strokeColor={LINE};strokeWidth=0.67;dashed=1;dashPattern=0.8 1.6;'
S_INTER = 'endArrow=openThin;endFill=0;endSize=4;html=1;rounded=0;strokeColor={c};strokeWidth=2;dashed=1;dashPattern=4 2;'
S_INTRA = 'endArrow=openThin;endFill=0;endSize=4;html=1;rounded=0;strokeColor={c};strokeWidth=1.3;dashed=1;dashPattern=0.7 1;'
S_GATE = 'endArrow=none;html=1;strokeColor=#000000;strokeWidth=1.2;'
S_CALLOUT = 'endArrow=openThin;endFill=0;endSize=3;html=1;rounded=0;strokeColor=#555555;strokeWidth=0.8;'


class Page:
    def __init__(self, name):
        self.name = name
        self.cells = []
        self.n = 1

    def _id(self):
        self.n += 1
        return f'{self.name}-{self.n}'

    def vertex(self, x, y, w, h, style, value=''):
        cid = self._id()
        self.cells.append(f'<mxCell id="{cid}" value="{sx.escape(value, {chr(34): "&quot;"})}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" as="geometry"/></mxCell>')
        return cid

    def edge(self, pts, style, value='', source=None, target=None):
        cid = self._id()
        st = ''
        if source:
            st += f' source="{source}"'
        if target:
            st += f' target="{target}"'
        geo = '<mxGeometry relative="1" as="geometry">'
        if not source:
            geo += f'<mxPoint x="{pts[0][0]:.2f}" y="{pts[0][1]:.2f}" as="sourcePoint"/>'
        if not target:
            geo += f'<mxPoint x="{pts[-1][0]:.2f}" y="{pts[-1][1]:.2f}" as="targetPoint"/>'
        mid = pts[1:-1]
        if mid:
            geo += '<Array as="points">' + ''.join(f'<mxPoint x="{x:.2f}" y="{y:.2f}"/>' for x, y in mid) + '</Array>'
        geo += '</mxGeometry>'
        self.cells.append(f'<mxCell id="{cid}" value="{sx.escape(value)}" style="{style}" edge="1" parent="1"{st}>{geo}</mxCell>')
        return cid

    def text(self, x, y, w, h, value, fs=10, align='center', bold=False, italic=False, color='#000000', rotation=None):
        fst = (1 if bold else 0) + (2 if italic else 0)
        st = S_TEXT.format(a=align, fs=fs, fst=fst, fc=color)
        if rotation is not None:
            st += f'rotation={rotation};'
        return self.vertex(x, y, w, h, st, value)

    def xml(self):
        body = ''.join(self.cells)
        return (f'<diagram name="{self.name}" id="{self.name}"><mxGraphModel dx="1400" dy="900" grid="0" gridSize="2" guides="1" tooltips="1" '
                f'connect="1" arrows="1" fold="1" page="0" pageScale="1" pageWidth="850" pageHeight="1100" math="0" shadow="0">'
                f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root></mxGraphModel></diagram>')


# ---------------------------------------------------------------- helpers
def tokens(p, x, y, n, color, removed=()):
    """Row of n 6x6 token squares; indices in `removed` are hatched (merged away)."""
    for i in range(n):
        st = (S_TOKEN_REMOVED if i in removed else S_TOKEN).format(c=color)
        p.vertex(x + 6 * i, y, 6, 6, st)


def legend(p, x, y):
    p.vertex(x, y, 560, 40, 'rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#666666;strokeWidth=1;')
    tokens(p, x + 8, y + 8, 4, TOK_BLUE)
    p.text(x + 34, y + 3, 40, 16, 'Tokens', fs=9, align='left', color='#5a5e66')
    p.vertex(x + 80, y + 9, 18, 6, S_BAR.format(c=C_TOKEN))
    p.text(x + 100, y + 3, 60, 16, 'Token comm.', fs=9, align='left', color='#5a5e66')
    p.vertex(x + 165, y + 9, 18, 6, S_BAR_SHIFT.format(c=C_TOKEN))
    p.text(x + 185, y + 3, 100, 16, 'Removed / shifted traffic', fs=9, align='left', color='#5a5e66')
    p.vertex(x + 295, y + 9, 18, 6, S_BAR.format(c=C_COMP))
    p.text(x + 315, y + 3, 50, 16, 'Compute', fs=9, align='left', color='#5a5e66')
    p.vertex(x + 365, y + 9, 18, 6, S_BAR_SHIFT.format(c=C_REDUCE))
    p.text(x + 385, y + 3, 60, 16, 'Σ pre-reduce', fs=9, align='left', color='#5a5e66')
    p.vertex(x + 450, y + 9, 18, 6, S_BAR.format(c=C_REDUCE))
    p.text(x + 470, y + 3, 60, 16, 'Top-k fold', fs=9, align='left', color='#5a5e66')
    p.edge([(x + 8, y + 30), (x + 40, y + 30)], S_LINE_NIC)
    p.text(x + 44, y + 22, 50, 16, 'NIC RDMA', fs=9, align='left', color='#5a5e66')
    p.edge([(x + 100, y + 30), (x + 132, y + 30)], S_LINE_NVL)
    p.text(x + 136, y + 22, 40, 16, 'NVLink', fs=9, align='left', color='#5a5e66')
    p.edge([(x + 180, y + 30), (x + 212, y + 30)], S_LINE_GPU)
    p.text(x + 216, y + 22, 30, 16, 'GPU', fs=9, align='left', color='#5a5e66')
    p.edge([(x + 260, y + 30), (x + 292, y + 30)], S_INTER.format(c='#000000'))
    p.text(x + 296, y + 22, 80, 16, 'Inter-node traffic', fs=9, align='left', color='#5a5e66')
    p.edge([(x + 380, y + 30), (x + 412, y + 30)], S_INTRA.format(c='#000000'))
    p.text(x + 416, y + 22, 80, 16, 'Intra-node traffic', fs=9, align='left', color='#5a5e66')
    p.edge([(x + 500, y + 26), (x + 500, y + 34)], S_GATE)
    p.text(x + 505, y + 22, 50, 16, 'gate opens', fs=9, align='left', color='#5a5e66')


class Timeline:
    """Rows of resource timelines; t in ms -> x."""

    def __init__(self, p, x0, y0, t0, t1):
        self.p, self.x0, self.y0, self.t0, self.t1 = p, x0, y0, t0, t1
        self.rows = {}
        self.ny = 0

    def x(self, t):
        return self.x0 + LABEL_W + (t - self.t0) * PX_PER_MS

    def row(self, key, label, kind, group=None):
        y = self.y0 + self.ny * ROW_H
        self.rows[key] = y
        self.ny += 1
        self.p.text(self.x0, y - 2, LABEL_W - 4, ROW_H, label, fs=8, align='right', color='#333333')
        st = {'nic': S_LINE_NIC, 'nvl': S_LINE_NVL, 'gpu': S_LINE_GPU}[kind]
        self.p.edge([(self.x(self.t0), y + BAR_H / 2), (self.x(self.t1), y + BAR_H / 2)], st)
        return y

    def gap(self, n=0.5):
        self.ny += n

    def bar(self, key, a, b, color, label='', shifted=False, outline=False):
        y = self.rows[key]
        st = (S_BAR_SHIFT if shifted else S_BAR_OUTLINE if outline else S_BAR).format(c=color)
        return self.p.vertex(self.x(a), y, max(1.0, (b - a) * PX_PER_MS), BAR_H, st, label)

    def gate(self, key, t, label='', dy=-4):
        y = self.rows[key]
        self.p.edge([(self.x(t), y - 2), (self.x(t), y + BAR_H + 2)], S_GATE)
        if label:
            self.p.text(self.x(t) - 30, y + dy - 9, 60, 9, label, fs=7, color='#333333')

    def group_label(self, key_first, key_last, label):
        y0 = self.rows[key_first]
        y1 = self.rows[key_last] + BAR_H
        self.p.text(self.x0 - 30, (y0 + y1) / 2 - 20, 26, 40, label, fs=8, bold=True, rotation=-90)

    def axis(self, ticks):
        y = self.y0 + self.ny * ROW_H + 2
        self.p.edge([(self.x(self.t0), y), (self.x(self.t1), y)], S_LINE_NIC)
        for t in ticks:
            self.p.edge([(self.x(t), y - 2), (self.x(t), y + 2)], S_LINE_NIC)
            self.p.text(self.x(t) - 10, y + 2, 20, 9, f'{t:g}', fs=7, color='#5a5e66')
        self.p.text(self.x(self.t1) + 2, y - 4, 20, 9, 'ms', fs=7, color='#5a5e66', align='left')


# ---------------------------------------------------------------- dispatch
def dispatch_page():
    p = Page('dispatch_v1')
    p.text(0, 0, 420, 18, '(a) Dispatch: split, then merge across 4 nodes; each window gates a row-slice of every local expert', fs=10, bold=True, align='left')
    legend(p, 470, 18)

    # --- topology (schematic, sender node 0 in detail; nodes 1-3 compact)
    X, Y = 0, 50
    p.vertex(X, Y, 250, 60, S_NODE)
    p.text(X + 2, Y + 1, 40, 12, 'Node 0', fs=9, bold=True, align='left')
    g0 = p.vertex(X + 8, Y + 16, 110, 38, S_GPU)
    p.text(X + 10, Y + 18, 30, 12, 'GPU0', fs=9, bold=True, align='left')
    tokens(p, X + 46, Y + 20, 6, TOK_BLUE, removed=(4, 5))
    p.text(X + 44, Y + 30, 76, 20, '① merge: 1 row per\n(token, dest node)', fs=6.5, align='left', color='#5a5e66')
    g1 = p.vertex(X + 130, Y + 16, 110, 38, S_GPU)
    p.text(X + 132, Y + 18, 30, 12, 'GPU1', fs=9, bold=True, align='left')
    tokens(p, X + 168, Y + 20, 8, TOK_RED, removed=(6, 7))
    p.text(X + 166, Y + 30, 76, 20, 'chunk 0 of the N0→N1\nstream pulled to GPU0', fs=6.5, align='left', color='#5a5e66')
    # split: GPU1 -> GPU0 NVLink (shifted)
    p.edge([(X + 168, Y + 23), (X + 84, Y + 23)], S_INTRA.format(c=TOK_RED))
    p.text(X + 60, Y + 3, 130, 10, '② split: equal ¼ chunks per NIC', fs=6.5, color='#333333')

    # destination nodes
    Y2 = 135
    p.vertex(X, Y2, 250, 60, S_NODE)
    p.text(X + 2, Y2 + 1, 40, 12, 'Node 1', fs=9, bold=True, align='left')
    g4 = p.vertex(X + 8, Y2 + 16, 110, 38, S_GPU)
    p.text(X + 10, Y2 + 18, 60, 12, 'GPU4 (gateway)', fs=9, bold=True, align='left')
    e1 = p.vertex(X + 94, Y2 + 18, 20, 20, S_EXPERT, 'E1')
    g5 = p.vertex(X + 130, Y2 + 16, 110, 38, S_GPU)
    p.text(X + 132, Y2 + 18, 30, 12, 'GPU5', fs=9, bold=True, align='left')
    e3 = p.vertex(X + 214, Y2 + 18, 20, 20, S_EXPERT, 'E3')
    tokens(p, X + 14, Y2 + 40, 4, TOK_BLUE)
    tokens(p, X + 38, Y2 + 40, 4, TOK_RED, removed=(2, 3))
    p.text(X + 12, Y2 + 47, 76, 8, 'window (N0→N1, chunk 0)', fs=6, align='left', color='#5a5e66')
    # wire GPU0 -> GPU4 same-lr gateway
    p.edge([(X + 30, Y + 54), (X + 30, Y2 + 16)], S_INTER.format(c='#000000'))
    p.text(X + 36, Y + 62, 160, 20, '③ wire: 1 blocking put per remote node,\nto the same-rank gateway of that node', fs=6.5, align='left', color='#333333')
    # fan-out GPU4 -> GPU5 NVLink
    p.edge([(X + 70, Y2 + 44), (X + 150, Y2 + 44)], S_INTRA.format(c=TOK_BLUE))
    p.text(X + 100, Y2 + 3, 90, 10, '④ gateway fan-out (NVLink)', fs=6.5, color='#333333')
    p.text(X + 134, Y2 + 36, 70, 16, '⑤ gate: E3 rows from\nthis window unblock', fs=6, align='left', color='#5a5e66')
    # compact nodes 2, 3
    for i, (nm, yy) in enumerate((('Node 2', 205), ('Node 3', 240))):
        p.vertex(X, yy, 250, 28, S_NODE)
        p.text(X + 2, yy + 1, 40, 12, nm, fs=9, bold=True, align='left')
        for k in range(4):
            p.vertex(X + 50 + 48 * k, yy + 6, 40, 16, S_GPU, f'GPU{4 * (i + 2) + k}')
    p.edge([(X + 22, Y + 54), (X - 6, Y + 54), (X - 6, 205 + 14), (X + 50, 205 + 14)], S_INTER.format(c='#000000'))
    p.edge([(X + 18, Y + 54), (X - 10, Y + 54), (X - 10, 240 + 14), (X + 50, 240 + 14)], S_INTER.format(c='#000000'))
    p.text(X, 272, 250, 30, 'Per rank per dispatch: 3 NIC puts (one per remote node), 3 NVLink pulls (split), '
           '12 NVLink fan-out puts as gateway. Every NIC carries ¼ of its node\'s stream to each destination.', fs=6.5, align='left', color='#5a5e66')

    # --- timeline (efficient case medians; rank-0 arrival order)
    T = Timeline(p, TL_X0, 80, 0, 25)
    T.row('g0nic', 'N0·GPU0  NIC', 'nic')
    T.row('g0nvl', 'N0·GPU0  NVLink', 'nvl')
    T.row('g0gpu', 'N0·GPU0  GPU', 'gpu')
    T.gap()
    T.row('g1nic', 'N0·GPU1  NIC', 'nic')
    T.row('g1nvl', 'N0·GPU1  NVLink', 'nvl')
    T.gap()
    T.row('g4nvl', 'N1·GPU4  NVLink', 'nvl')
    T.row('g5nvl', 'N1·GPU5  NVLink', 'nvl')
    T.row('g5gpu', 'N1·GPU5  GPU', 'gpu')
    T.axis([0, 5, 10, 15, 20, 25])

    # GPU0 (sender + gateway), measured: plan 0.7-2.6, pull 3.95-5.3, puts 5.14 +3.4 each, GEMM 5.34-19.95
    T.bar('g0gpu', 0.7, 2.6, C_PLAN, 'plan')
    T.bar('g0nvl', 3.0, 3.9, C_TOKEN, '', shifted=False)                    # intra dedup puts (own node)
    T.bar('g0nvl', 3.95, 5.3, C_TOKEN, 'pull', shifted=True)                # ② split: chunk pulled from node-mates
    for i, (lab, a) in enumerate((('→N3', 5.14), ('→N2', 8.54), ('→N1', 11.94))):
        T.bar('g0nic', a, a + 3.4, C_TOKEN, lab)
    # gateway forwards (3 rounds x 3 peers, 0.57 each) after each inbound window lands (8.5, 11.9, 15.4)
    for a in (8.6, 12.0, 15.5):
        for k in range(3):
            T.bar('g0nvl', a + 0.6 * k, a + 0.6 * k + 0.57, C_TOKEN, '')
    T.bar('g0gpu', 5.34, 19.95, C_COMP, 'grouped GEMM: own-node rows → window rows as they land')
    for t, lab in ((6.9, 'own'), (10.1, 'N1'), (11.9, 'N2'), (15.3, 'N3')):
        T.gate('g0gpu', t, lab)
    T.bar('g0gpu', 19.95, 23.3, C_WAIT, 'barrier')
    # GPU1: same shape, its NIC carries its own ¼ chunks (balanced), rotation differs
    T.bar('g1nvl', 3.0, 3.9, C_TOKEN, '')
    T.bar('g1nvl', 3.95, 5.3, C_TOKEN, 'pull', shifted=True)
    for lab, a in (('→N3', 5.3), ('→N2', 8.7), ('→N1', 12.1)):
        T.bar('g1nic', a, a + 3.4, C_TOKEN, lab)
    for a in (8.8, 12.2, 15.7):
        for k in range(3):
            T.bar('g1nvl', a + 0.6 * k, a + 0.6 * k + 0.57, C_TOKEN, '')
    # Node 1 side: GPU4 gateway receives (N0→N1, chunk 0) at ~15.4 (N0's third put) and fans out
    T.bar('g4nvl', 3.95, 5.3, C_TOKEN, 'pull', shifted=True)
    for a in (8.6, 12.0, 15.5):
        for k in range(3):
            T.bar('g4nvl', a + 0.6 * k, a + 0.6 * k + 0.57, C_TOKEN, '')
    p.text(T.x(15.5) - 40, T.rows['g4nvl'] - 10, 120, 9, 'window (N0→N1, c0) lands → fan-out', fs=6.5, color='#333333')
    # GPU5 receives fan-outs from the 4 gateways of its node (own lanes first), GEMM gated per window
    T.bar('g5nvl', 3.0, 3.9, C_TOKEN, '')
    for a in (9.2, 12.6, 16.1):
        T.bar('g5nvl', a, a + 0.6, C_TOKEN, '')
    T.bar('g5gpu', 5.34, 19.95, C_COMP, 'E3 tiles: own rows → N2 rows → N3 rows → N0 rows')
    for t, lab in ((6.9, 'own'), (9.8, 'N2'), (13.2, 'N3'), (16.7, 'N0')):
        T.gate('g5gpu', t, lab)
    T.bar('g5gpu', 19.95, 23.3, C_WAIT, '')
    p.text(T.x(0), T.rows['g5gpu'] + 26, 380, 22,
           'Rank 0 medians, K2 4n b64 efficient case (capsule 20260905-141104, iter4): first put 5.1 ms, three back-to-back 3.4 ms puts (NIC gap ≤0.01 ms), '
           'GEMM starts 0.2 ms after the first put on own-node rows; windows land 10.1 / 11.9 / 15.3; GEMM ends 1.4 ms after the last window.',
           fs=6.5, align='left', color='#5a5e66')
    return p


# ---------------------------------------------------------------- combine
def combine_page():
    p = Page('combine_v1')
    p.text(0, 0, 420, 18, '(b) Combine: destination-node waves, converge, Σ pre-reduce (new compute), one put per node', fs=10, bold=True, align='left')
    legend(p, 470, 18)

    X, Y = 0, 50
    # producer node 1 (hosts experts)
    p.vertex(X, Y, 250, 66, S_NODE)
    p.text(X + 2, Y + 1, 40, 12, 'Node 1', fs=9, bold=True, align='left')
    p.vertex(X + 8, Y + 16, 110, 44, S_GPU)
    p.text(X + 10, Y + 18, 60, 12, 'GPU4 (gateway)', fs=9, bold=True, align='left')
    p.vertex(X + 12, Y + 32, 20, 20, S_EXPERT, 'E1')
    sig = p.vertex(X + 60, Y + 34, 22, 18, 'ellipse;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#000000;fontSize=11;', 'Σ')
    p.text(X + 84, Y + 30, 34, 26, '③ pre-\nreduce', fs=6.5, align='left', color='#333333')
    p.vertex(X + 130, Y + 16, 110, 44, S_GPU)
    p.text(X + 132, Y + 18, 30, 12, 'GPU5', fs=9, bold=True, align='left')
    p.vertex(X + 214, Y + 20, 20, 20, S_EXPERT, 'E3')
    tokens(p, X + 136, Y + 40, 4, TOK_BLUE)
    p.text(X + 134, Y + 48, 76, 10, 'partials for N0 (wave 2)', fs=6, align='left', color='#5a5e66')
    p.edge([(X + 136, Y + 43), (X + 82, Y + 43)], S_INTRA.format(c=TOK_BLUE))
    p.text(X + 60, Y + 3, 130, 10, '② converge (NVLink, sender side)', fs=6.5, color='#333333')
    p.text(X + 34, Y + 30, 30, 20, '① GEMM\nby waves', fs=6, align='left', color='#5a5e66')

    # home node 0
    Y2 = 145
    p.vertex(X, Y2, 250, 56, S_NODE)
    p.text(X + 2, Y2 + 1, 40, 12, 'Node 0', fs=9, bold=True, align='left')
    p.vertex(X + 8, Y2 + 16, 110, 34, S_GPU)
    p.text(X + 10, Y2 + 18, 60, 12, 'GPU0 (home)', fs=9, bold=True, align='left')
    tokens(p, X + 14, Y2 + 36, 6, TOK_BLUE, removed=(4, 5))
    p.vertex(X + 60, Y2 + 30, 22, 18, 'ellipse;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#000000;fontSize=11;', 'Σ')
    p.text(X + 84, Y2 + 28, 34, 22, '⑤ top-k\nfold', fs=6.5, align='left', color='#333333')
    p.vertex(X + 130, Y2 + 16, 110, 34, S_GPU)
    p.text(X + 132, Y2 + 18, 30, 12, 'GPU1', fs=9, bold=True, align='left')
    p.edge([(X + 30, Y + 60), (X + 30, Y2 + 16)], S_INTER.format(c='#000000'))
    p.text(X + 36, Y + 68, 180, 30, '④ wire: 1 blocking put per remote node, straight\ninto the home rank (no receiver gateway);\n1 row per (token, source node)', fs=6.5, align='left', color='#333333')
    for i, (nm, yy) in enumerate((('Node 2', 210), ('Node 3', 245))):
        p.vertex(X, yy, 250, 28, S_NODE)
        p.text(X + 2, yy + 1, 40, 12, nm, fs=9, bold=True, align='left')
        for k in range(4):
            p.vertex(X + 50 + 48 * k, yy + 6, 40, 16, S_GPU, f'GPU{4 * (i + 2) + k}')
    p.edge([(X + 22, Y + 60), (X - 6, Y + 60), (X - 6, 210 + 14), (X + 50, 210 + 14)], S_INTER.format(c='#000000'))
    p.edge([(X + 18, Y + 60), (X - 10, Y + 60), (X - 10, 245 + 14), (X + 50, 245 + 14)], S_INTER.format(c='#000000'))
    p.text(X, 277, 250, 30, 'Mirror of dispatch: the gateway moves to the SENDER side and gains a compute stage (Σ): contributions to the same '
           '(token, dest) are numerically distinct partial sums, so bytes ∝ contributions unless summed before the NIC.', fs=6.5, align='left', color='#5a5e66')

    # --- timeline: l1 GEMM 24.5-35.4 (efficient), conv at +2.9/+5.6/+8.3/+10.9, puts at +7.3/+10.3/+12.0 all end 42.15, folds, end 47.8
    T = Timeline(p, TL_X0, 80, 24, 48)
    T.row('g4gpu', 'N1·GPU4  GPU', 'gpu')
    T.row('g4side', 'N1·GPU4  side stream', 'gpu')
    T.row('g4nvl', 'N1·GPU4  NVLink', 'nvl')
    T.row('g4nic', 'N1·GPU4  NIC', 'nic')
    T.gap()
    T.row('g5gpu', 'N1·GPU5  GPU', 'gpu')
    T.row('g5nvl', 'N1·GPU5  NVLink', 'nvl')
    T.gap()
    T.row('g0nvl', 'N0·GPU0  NVLink', 'nvl')
    T.row('g0side', 'N0·GPU0  side stream', 'gpu')
    T.axis([24, 28, 32, 36, 40, 44, 48])

    g1s, wave = 24.5, 2.72
    waves = (('w0 →N2', 0), ('w1 →N3', 1), ('w2 →N0', 2), ('w3 own', 3))
    for lab, i in waves:
        T.bar('g4gpu', g1s + i * wave, g1s + (i + 1) * wave, C_COMP, lab)
        T.gate('g4gpu', g1s + (i + 1) * wave, '', dy=0)
        T.bar('g5gpu', g1s + i * wave, g1s + (i + 1) * wave, C_COMP, lab)
    T.bar('g4gpu', 35.4, 36.0, C_WAIT, '')
    T.bar('g4side', g1s, g1s + 12.5, C_REDUCE, 'Σ pre-reduce: resident side-stream kernel, 6 CTAs', shifted=True)
    # converge bursts after each wave (3 peers x ~0.4 ms), own-node wave last
    for i in range(4):
        a = g1s + (i + 1) * wave + 0.2
        for k in range(3):
            T.bar('g4nvl', a + 0.45 * k, a + 0.45 * k + 0.4, C_TOKEN, '')
            T.bar('g5nvl', a + 0.45 * k, a + 0.45 * k + 0.4, C_TOKEN, '')
    p.text(T.x(g1s + wave) - 20, T.rows['g4nvl'] - 10, 90, 9, 'wave 0 closes → converge', fs=6.5, color='#333333')
    # wire: one put per remote node, staggered starts, concurrent streams, shared NIC → all end together
    for k, (lab, a) in enumerate((('→N2', g1s + 7.27), ('→N3', g1s + 10.3), ('→N0', g1s + 12.0))):
        y = T.rows['g4nic'] + k * (BAR_H / 3)
        p.vertex(T.x(a), y, (42.15 - a) * PX_PER_MS, BAR_H / 3 - 0.3, S_BAR.format(c=C_TOKEN) + 'fontSize=6;', lab)
    p.text(T.x(g1s + 7.27) - 60, T.rows['g4nic'] + BAR_H + 1, 120, 9, 'first put = wave 0 + conv + Σ (4.6 ms; 1.2 ms at 48 CTAs)', fs=6.5, color='#333333')
    # home GPU0: own-node lanes first (small folds), remote lanes as they land, final fold after last put
    T.bar('g0nvl', 35.6, 36.4, C_TOKEN, '')
    T.bar('g0nvl', 36.6, 37.3, C_TOKEN, '')
    T.bar('g0nvl', 37.6, 38.2, C_TOKEN, '')
    for a in (35.3, 36.5, 37.4, 39.4):
        T.bar('g0side', a, a + 0.3, C_REDUCE, '')
    T.gate('g0side', 42.15, 'last lane', dy=-4)
    T.bar('g0side', 42.3, 45.3, C_REDUCE, 'final fold')
    T.bar('g0side', 45.3, 47.8, C_WAIT, 'barrier')
    p.text(T.x(24), T.rows['g0side'] + 26, 380, 30,
           'Rank medians, K2 4n b64 efficient case (capsule 20260905-141104, iter4): l1 GEMM 24.5–35.4 in 4 destination-node waves (own node last); '
           'converge 2.9 ms after GEMM start; first put 7.3 ms after GEMM start (= wave 0 + 4.6 ms of Σ); 3 concurrent puts end together at 42.2; '
           'home fold 3.0 ms after the last lane; iteration end 47.8. Σ CTA ladder (capsule 20260910-132745): first-put offset 7.4/6.0/5.3/3.2 ms at 6/12/24/48 CTAs, l1 GEMM 10.0→13.3 ms.',
           fs=6.5, align='left', color='#5a5e66')
    return p


def main():
    pages = [dispatch_page(), combine_page()]
    doc = '<mxfile host="make_overlap_drawio.py" version="1">' + ''.join(pg.xml() for pg in pages) + '</mxfile>'
    open('overlap_pipeline.drawio', 'w').write(doc)
    print('wrote overlap_pipeline.drawio pages:', [pg.name for pg in pages])


if __name__ == '__main__':
    main()
