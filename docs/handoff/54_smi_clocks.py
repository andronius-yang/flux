"""R5: node-0 GPU clocks / power per arm and point. Windows: from 'dXX_<arm>: waves at N running' to the next report
event; samples with utilization > 50 % only (the decode waves). Throttle bits: 0x1 idle, 0x4 SW power cap, 0x8 HW
slowdown, 0x20 SW thermal, 0x40 HW thermal, 0x80 HW power brake."""
import re, sys, statistics as st, collections, datetime as dt
csv, rep = sys.argv[1], sys.argv[2]
day = None
S = []
for line in open(csv):
    p = [x.strip() for x in line.split(',')]
    if len(p) < 8: continue
    try:
        t = dt.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f")
    except ValueError:
        continue
    day = t.date()
    S.append((t, int(p[1]), float(p[2].split()[0]), float(p[3].split()[0]), float(p[4].split()[0]), int(p[5]), int(p[6], 16), float(p[7].split()[0])))
ev = []
for line in open(rep):
    m = re.match(r"(\d\d:\d\d:\d\d) (\S+): waves at (\d+) running", line)
    m2 = re.match(r"(\d\d:\d\d:\d\d) ", line)
    if m:
        ev.append((dt.datetime.combine(day, dt.datetime.strptime(m.group(1), "%H:%M:%S").time()), m.group(2), int(m.group(3))))
    elif m2:
        ev.append((dt.datetime.combine(day, dt.datetime.strptime(m2.group(1), "%H:%M:%S").time()), None, None))
ev.sort(key=lambda e: e[0])
rows = collections.defaultdict(list)
for (t0, arm, pt), (t1, _, _) in zip(ev, ev[1:]):
    if arm is None: continue
    for s in S:
        if t0 <= s[0] < t1 and s[7] > 50: rows[(arm, pt)].append(s)
print("arm point: samples | SM MHz median (p10) | mem MHz | power W median (p90) | temp C | throttle bits seen (fraction)")
for (arm, pt), v in rows.items():
    bits = collections.Counter()
    for s in v:
        for b in (0x4, 0x8, 0x20, 0x40, 0x80):
            if s[6] & b: bits[hex(b)] += 1
    q = lambda xs, f: sorted(xs)[int(f * (len(xs) - 1))]
    sm = [s[2] for s in v]; pw = [s[4] for s in v]
    print(f"{arm:12s} {pt:5d}: {len(v):5d} | {st.median(sm):6.0f} ({q(sm,.1):5.0f}) | {st.median([s[3] for s in v]):5.0f} | "
          f"{st.median(pw):6.1f} ({q(pw,.9):6.1f}) | {st.median([s[5] for s in v]):3.0f} | "
          + " ".join(f"{k}:{c/len(v):.2f}" for k, c in sorted(bits.items())))
