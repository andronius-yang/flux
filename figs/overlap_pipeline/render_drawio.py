#!/usr/bin/env python
"""Rough draw.io -> SVG/PNG renderer for review (no draw.io CLI on Perlmutter).

Usage: render_drawio.py <file.drawio> <page name> <out prefix> [scale]
Handles the subset used by the figure lanes: nested groups, rect/rounded/ellipse/
text vertices (fill, stroke, dashed, zigzag hatch approximated by a pattern),
edges with source/target or explicit points. Run with the andrewy-comet conda
python (needs cairosvg).
"""
import sys, re, zlib, base64, urllib.parse, html, xml.etree.ElementTree as ET
import cairosvg

path, name, out = sys.argv[1], sys.argv[2], sys.argv[3]
scale = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
root = ET.parse(path).getroot()
model = None
for d in root.findall('diagram'):
    if d.get('name') != name:
        continue
    if len(d) == 0:
        xml = urllib.parse.unquote(zlib.decompress(base64.b64decode(d.text.strip()), -15).decode())
        model = ET.fromstring(xml)
    else:
        model = d[0]
if model is None:
    sys.exit(f"page {name!r} not found")
cells = {c.get('id'): c for c in model.iter('mxCell')}


def style(c):
    s = {}
    for t in (c.get('style') or '').split(';'):
        if '=' in t:
            k, v = t.split('=', 1)
            s[k] = v
        elif t:
            s[t] = '1'
    return s


def parent_offset(c):
    x = y = 0.0
    p = cells.get(c.get('parent'))
    while p is not None:
        pg = p.find('mxGeometry')
        if pg is not None:
            x += float(pg.get('x') or 0)
            y += float(pg.get('y') or 0)
        p = cells.get(p.get('parent'))
    return x, y


def absgeo(c):
    g = c.find('mxGeometry')
    if g is None:
        return None
    ox, oy = parent_offset(c)
    return (float(g.get('x') or 0) + ox, float(g.get('y') or 0) + oy,
            float(g.get('width') or 0), float(g.get('height') or 0))


def col(v, default):
    if v is None or v == 'default':
        return default
    if v == 'none':
        return 'none'
    m = re.match(r'light-dark\(([^,]+),', v)
    return m.group(1) if m else v


els = []
minx = miny = 1e9
maxx = maxy = -1e9


def bb(x, y, w=0, h=0):
    global minx, miny, maxx, maxy
    minx, miny = min(minx, x), min(miny, y)
    maxx, maxy = max(maxx, x + w), max(maxy, y + h)


def clean(v):
    v = re.sub('<br[^>]*>', '\n', v or '')
    v = re.sub('<[^>]+>', '', v)
    return html.unescape(v).strip()


for c in cells.values():
    st = style(c)
    if c.get('vertex') == '1':
        geo = absgeo(c)
        if geo is None:
            continue
        x, y, w, h = geo
        bb(x, y, w, h)
        if 'group' in st:
            continue
        fill = col(st.get('fillColor'), '#ffffff')
        stroke = col(st.get('strokeColor'), '#000000')
        sw = float(st.get('strokeWidth', 1))
        if st.get('shape') == 'text' or 'text' in st:
            fill, stroke = 'none', 'none'
        if st.get('fillStyle') == 'zigzag-line' and fill != 'none':
            pid = f"hatch{len(els)}"
            els.append(f'<pattern id="{pid}" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
                       f'<rect width="4" height="4" fill="{fill}" fill-opacity="0.35"/><line x1="0" y1="0" x2="0" y2="4" stroke="{fill}" stroke-width="1.6"/></pattern>')
            fill = f'url(#{pid})'
        if 'ellipse' in st or st.get('shape') == 'ellipse':
            els.append(f'<ellipse cx="{x + w / 2}" cy="{y + h / 2}" rx="{w / 2}" ry="{h / 2}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        else:
            rx = 4 if st.get('rounded') == '1' else 0
            dash = ' stroke-dasharray="3,2"' if st.get('dashed') == '1' else ''
            els.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>')
        v = clean(c.get('value'))
        if v:
            fs = float(st.get('fontSize', 12))
            fc = col(st.get('fontColor'), '#000000')
            align = st.get('align', 'center')
            va = st.get('verticalAlign', 'middle')
            anchor = {'left': 'start', 'right': 'end'}.get(align, 'middle')
            tx = {'left': x + 2, 'right': x + w - 2}.get(align, x + w / 2)
            lines = v.split('\n')
            ty = {'top': y + fs, 'bottom': y + h - 2}.get(va, y + h / 2 + fs * 0.35 - (len(lines) - 1) * fs * 0.6)
            weight = ' font-weight="bold"' if st.get('fontStyle') in ('1', '3') else ''
            italic = ' font-style="italic"' if st.get('fontStyle') in ('2', '3') else ''
            rot = f' transform="rotate({st["rotation"]} {x + w / 2} {y + h / 2})"' if st.get('rotation') else ''
            for i, l in enumerate(lines):
                els.append(f'<text x="{tx}" y="{ty + i * fs * 1.2}" font-size="{fs}" fill="{fc}" text-anchor="{anchor}" font-family="Helvetica,Arial"{weight}{italic}{rot}>{html.escape(l)}</text>')
    elif c.get('edge') == '1':
        g = c.find('mxGeometry')
        ox, oy = parent_offset(c)
        pts, sp, tp = [], None, None

        def center(cid):
            cc = cells.get(cid)
            if cc is None:
                return None
            a = absgeo(cc)
            return (a[0] + a[2] / 2, a[1] + a[3] / 2)

        if g is not None:
            for p in g.findall('mxPoint'):
                if p.get('as') == 'sourcePoint':
                    sp = (float(p.get('x') or 0) + ox, float(p.get('y') or 0) + oy)
                if p.get('as') == 'targetPoint':
                    tp = (float(p.get('x') or 0) + ox, float(p.get('y') or 0) + oy)
            arr = g.find("Array[@as='points']")
            if arr is not None:
                pts = [(float(p.get('x') or 0) + ox, float(p.get('y') or 0) + oy) for p in arr.findall('mxPoint')]
        s = center(c.get('source')) or sp
        t = center(c.get('target')) or tp
        if s is None or t is None:
            continue
        allp = [s] + pts + [t]
        for p in allp:
            bb(p[0], p[1])
        stroke = col(st.get('strokeColor'), '#000000')
        sw = float(st.get('strokeWidth', 1))
        dash = f' stroke-dasharray="{st.get("dashPattern", "4 3").replace(" ", ",")}"' if st.get('dashed') == '1' else ''
        d = 'M ' + ' L '.join(f'{p[0]},{p[1]}' for p in allp)
        marker = '' if st.get('endArrow', 'classic') == 'none' else ' marker-end="url(#arr)"'
        els.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}"{dash}{marker}/>')
        v = clean(c.get('value'))
        if v:
            mx = (allp[0][0] + allp[-1][0]) / 2
            my = (allp[0][1] + allp[-1][1]) / 2
            els.append(f'<text x="{mx}" y="{my}" font-size="9" fill="#333" text-anchor="middle" font-family="Helvetica">{html.escape(v)}</text>')

pad = 16
W, H = maxx - minx + 2 * pad, maxy - miny + 2 * pad
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="{minx - pad} {miny - pad} {W} {H}">'
       f'<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="none" stroke="#333"/></marker></defs>'
       f'<rect x="{minx - pad}" y="{miny - pad}" width="{W}" height="{H}" fill="white"/>' + ''.join(els) + '</svg>')
open(out + '.svg', 'w').write(svg)
cairosvg.svg2png(bytestring=svg.encode(), write_to=out + '.png', scale=scale)
print(out + '.png', round(W), round(H))
