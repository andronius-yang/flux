#!/usr/bin/env python3
"""Analyze FLUX_A2AV_NVTX_PROXY layer-C tile-trace sidecars.

Inputs are the per-rank binaries the poller writes under FLUX_SWEEP_RECORD_DIR
(`a2av_tile_trace_r<rank>.bin`), self-contained per iteration. Outputs:

  --table          per-arrival cohort table (default on)
  --curves out.png regime plot: cumulative fired + in-flight vs time, arrival
                   verticals, capacity reference (two stacked subplots, one x)
  --gantt out.json Perfetto/Chrome trace JSON: one track per SM, a gray spin
                   slice (t_enter->t_fire) then a compute slice (t_fire->t_done)
                   named by the tile's attributed source
  --align nsys.sqlite  rebase the Gantt onto the nsys wall clock by pairing the
                   k-th AGScatter kernel launch with the k-th traced iteration
  --scan-ranks     per-rank starvation table over ALL ranks in the inputs:
                   longest contiguous window with in-flight < 50% of the rank's
                   own peak (longest_low_us) and the normalized in-flight
                   deficit integral (deficit_frac); marks the worst rank
  --compare LABEL=DIR[:RANK] ...  cross-run comparison figure (--compare-out):
                   one subplot per entry (fired cum. blue/left, in-flight
                   orange/right, arrival verticals), shared x-axis. RANK
                   defaults to that run's worst rank per --scan-ranks metric.

Epoch selection for --scan-ranks/--compare is hygiene-filtered: iterations
with span < 10 ms whose record count equals the rank's modal count (drops cold
warmup epochs and merged/partial blocks); the last such iteration is used.

Cohort attribution is dynamic: a tile belongs to the LAST-ARRIVING source of
its [seg_start, seg_end] span (device %globaltimer stamps); sources with no
valid stamp count as already-arrived. Static (seg_end) vs dynamic disagreement
is reported — it is the boundary-tile error of the live NVTX view.

All record times are the low 32 bits of absolute %globaltimer ns; they are
rebased against low32(t0_gt) with wrap-safe u32 arithmetic (exact < 4.3 s).
"""

import argparse
import glob
import os
import struct
import sys
from collections import defaultdict

MAGIC = 0xA2A71E5
HDR = struct.Struct("<IIQ4iQ2I")  # magic, version, epoch, rank, world, nnodes, nb, t0_gt, n, pad

C_FIRED = "#2a78d6"     # categorical slot 1 (blue)
C_INFLIGHT = "#eb6834"  # categorical slot 2 (orange)
C_INTRA = "#1baf7a"     # aqua — intra-node arrivals (matches NVTX green-ish)
C_INTER = "#e34948"     # red — inter-node arrivals (matches NVTX red)
C_REF = "#9a9994"       # neutral reference lines
C_TEXT = "#0b0b0b"
C_TEXT2 = "#52514e"
C_SURFACE = "#fcfcfb"
C_SPIN = "#c3c2b7"


def u32_rebase(t32, t0):
    return (int(t32) - (int(t0) & 0xFFFFFFFF)) & 0xFFFFFFFF


class Iteration:
    def __init__(self, epoch, rank, world, nnodes, nb, t0_gt):
        self.epoch, self.rank, self.world = epoch, rank, world
        self.nnodes, self.nb, self.t0_gt = nnodes, nb, t0_gt
        self.arrival_gt = []   # u64 abs, valid iff ready_seq >= epoch
        self.ready_seq = []
        self.expected = []
        self.rows = []         # v2: per-source ROW counts (ground truth; [] on v1)
        self.recs = []         # (problem, tile, smid, seg0, seg1, cta, t_enter, t_fire, t_done) ns rel

    def arrivals_rel(self):
        """{source: ns since t0} for sources with a valid device stamp."""
        out = {}
        for s in range(self.nb):
            if self.ready_seq[s] >= self.epoch and self.arrival_gt[s] > 0:
                out[s] = self.arrival_gt[s] - self.t0_gt
        return out

    def attribute(self):
        """Per record: (static seg_end, dynamic cohort source)."""
        arr = self.arrivals_rel()
        out = []
        for r in self.recs:
            seg0, seg1 = r[3], r[4]
            span = range(seg0, seg1 + 1)
            stamped = [(arr[s], s) for s in span if s in arr]
            dyn = max(stamped)[1] if stamped else seg1
            out.append((seg1, dyn))
        return out


def read_sidecar(path):
    iters = []
    with open(path, "rb") as f:
        while True:
            raw = f.read(HDR.size)
            if len(raw) < HDR.size:
                break
            magic, ver, epoch, rank, world, nnodes, nb, t0, n, _ = HDR.unpack(raw)
            if magic != MAGIC or ver not in (1, 2):
                raise SystemExit(f"{path}: bad header (magic {magic:#x} ver {ver})")
            it = Iteration(epoch, rank, world, nnodes, nb, t0)
            it.arrival_gt = list(struct.unpack(f"<{nb}Q", f.read(8 * nb)))
            it.ready_seq = list(struct.unpack(f"<{nb}Q", f.read(8 * nb)))
            it.expected = list(struct.unpack(f"<{nb}I", f.read(4 * nb)))
            if ver >= 2:
                it.rows = list(struct.unpack(f"<{nb}I", f.read(4 * nb)))
            for _ in range(n):
                tile, meta, te, tf, td, _pad = struct.unpack("<6I", f.read(24))
                it.recs.append((
                    tile >> 22, tile & 0x3FFFFF,          # problem, tile idx
                    meta >> 24, (meta >> 18) & 0x3F,      # smid, seg_start
                    (meta >> 12) & 0x3F, meta & 0xFFF,    # seg_end, cta
                    u32_rebase(te, t0), u32_rebase(tf, t0), u32_rebase(td, t0)))
            iters.append(it)
    return iters


def src_name(it, s):
    return "multi" if s == it.world else f"src{s}"


def is_inter(it, s):
    lw = it.world // it.nnodes if it.nnodes else it.world
    return s < it.world and (s // lw) != (it.rank // lw)


def cohort_table(it, out=sys.stdout):
    arr = it.arrivals_rel()
    attribution = it.attribute()
    coh = defaultdict(list)
    mismatch = 0
    for r, (stat, dyn) in zip(it.recs, attribution):
        coh[dyn].append(r)
        mismatch += stat != dyn
    print(f"\n== rank {it.rank} i{it.epoch}: {len(it.recs)} tiles, "
          f"{len(coh)} cohorts, static!=dynamic on {mismatch} tiles ==", file=out)
    print(f"{'cohort':>8} {'kind':>5} {'arrival_us':>10} {'n':>5} "
          f"{'fire_p50':>9} {'fire_p95':>9} {'drain_us':>9}", file=out)

    def pct(v, p):
        v = sorted(v)
        return v[min(len(v) - 1, int(p * len(v)))]

    if not arr:
        print("  (no device arrival stamps: no tile ever blocked in the "
              "backstop spin — SM-limited iteration; arrival_us falls back "
              "to '-', ramps are relative to each cohort's first fire)", file=out)
    order = sorted(coh, key=lambda s: arr.get(s, min(r[7] for r in coh[s])))
    for s in order:
        rs = coh[s]
        a = arr.get(s)
        fires = [r[7] for r in rs]
        done = max(r[8] for r in rs)
        base = a if a is not None else min(fires)
        a_txt = f"{a / 1e3:>10.1f}" if a is not None else f"{'-':>10}"
        print(f"{src_name(it, s):>8} {'inter' if is_inter(it, s) else 'intra':>5} "
              f"{a_txt} {len(rs):>5} "
              f"{(pct(fires, .5) - base) / 1e3:>9.1f} {(pct(fires, .95) - base) / 1e3:>9.1f} "
              f"{(done - base) / 1e3:>9.1f}", file=out)


def inflight_curve(it):
    """Step curve of tiles in flight: ([t_ns...], [count...]) starting at (0, 0)."""
    events = sorted([(r[7], +1) for r in it.recs] + [(r[8], -1) for r in it.recs])
    ts, vs, cur = [0], [0], 0
    for t, d in events:
        ts.append(t)
        vs.append(cur)
        cur += d
        ts.append(t)
        vs.append(cur)
    return ts, vs


def active_window(it):
    """(base, end) ns: the kernel-active window [first t_enter, last t_done].
    Robust to a stale epoch t0 (poller missed the epoch boundary): all
    starvation metrics use this window, never t0."""
    return min(r[6] for r in it.recs), max(r[8] for r in it.recs)


def starvation_metrics(it):
    """Low-in-flight statistics over the kernel-active window.

    cap is the rank's own peak in-flight (the SM-limited ceiling actually
    reached); longest_low is the longest contiguous window spent below
    cap/2 (the leading ramp counts — waiting for the wire IS starvation);
    deficit_frac integrates (cap - inflight) over the window, normalized so
    0 = pinned at capacity and 1 = fully idle.
    """
    base, end = active_window(it)
    span = end - base
    events = sorted([(r[7] - base, +1) for r in it.recs] +
                    [(r[8] - base, -1) for r in it.recs])
    cap = 0
    cur = 0
    for _, d in events:
        cur += d
        cap = max(cap, cur)
    thresh = cap / 2.0
    cur = 0
    prev_t = 0
    low_start = 0.0  # cur = 0 < thresh at window start
    longest_low = 0.0
    deficit = 0.0
    for t, d in events:
        if t > prev_t:
            deficit += (cap - cur) * (t - prev_t)
            prev_t = t
        was_low = cur < thresh
        cur += d
        if was_low and cur >= thresh:
            longest_low = max(longest_low, t - low_start)
        elif not was_low and cur < thresh:
            low_start = t
    if cur < thresh or events[-1][0] < span:
        longest_low = max(longest_low, span - low_start)
    return {
        "cap": cap,
        "span_us": span / 1e3,
        "first_fire_us": (min(r[7] for r in it.recs) - base) / 1e3,
        "longest_low_us": longest_low / 1e3,
        "deficit_frac": deficit / (cap * span) if cap and span else 0.0,
        "n_tiles": len(it.recs),
        "n_stamped": len(it.arrivals_rel()),
    }


def pick_clean(rank_iters):
    """(last hygiene-clean iteration, clean?) — clean: active window < 100 ms
    (rejects u32-wrap / merged multi-epoch blocks; slow algorithms keep their
    genuinely long iterations) and modal record count. Falls back to the last
    iteration (flagged dirty)."""
    from collections import Counter

    def w(it):
        b, e = active_window(it)
        return e - b

    cand = [it for it in rank_iters if it.recs and w(it) < 100e6]
    if not cand:
        return (rank_iters[-1], False) if rank_iters else (None, False)
    modal = Counter(len(it.recs) for it in cand).most_common(1)[0][0]
    best = [it for it in cand if len(it.recs) == modal]
    return ((best or cand)[-1], True)


def scan_ranks(iters, out=sys.stdout):
    """Starvation table over every rank; returns (worst_rank, {rank: (it, m)}).

    Cross-rank comparability: representative rows must be hygiene-clean AND
    from the globally modal epoch of the capture (same iteration everywhere —
    warmup/cold blocks never compete with steady-state ones). Ranks without
    such a block are shown (tagged !dirty) but excluded from worst-rank
    selection."""
    from collections import Counter
    by_rank = defaultdict(list)
    for it in iters:
        by_rank[it.rank].append(it)
    picked = {}
    for rank in sorted(by_rank):
        sel, clean = pick_clean(sorted(by_rank[rank], key=lambda i: i.epoch))
        if sel is not None:
            picked[rank] = (sel, clean)
    modal_epoch = Counter(
        sel.epoch for sel, clean in picked.values() if clean).most_common(1)
    modal_epoch = modal_epoch[0][0] if modal_epoch else None
    rows = {}
    for rank, (sel, clean) in picked.items():
        clean = clean and sel.epoch == modal_epoch
        rows[rank] = (sel, starvation_metrics(sel), clean)
    clean_rows = [r for r in rows if rows[r][2]]
    pool = clean_rows or list(rows)
    worst = max(pool, key=lambda r: (rows[r][1]["longest_low_us"],
                                     rows[r][1]["deficit_frac"]))
    print(f"{'rank':>4} {'iter':>5} {'tiles':>6} {'span_us':>8} {'first_fire':>10} "
          f"{'cap':>4} {'longest_low_us':>14} {'deficit':>8} {'stamped':>7}", file=out)
    for rank, (sel, m, clean) in rows.items():
        mark = "  <- worst" if rank == worst else ""
        dirty = "" if clean else "  !dirty(excluded)"
        print(f"{rank:>4} {sel.epoch:>5} {m['n_tiles']:>6} {m['span_us']:>8.1f} "
              f"{m['first_fire_us']:>10.1f} {m['cap']:>4} {m['longest_low_us']:>14.1f} "
              f"{m['deficit_frac']:>8.3f} {m['n_stamped']:>7}{dirty}{mark}", file=out)
    return worst, rows


def compare_png(entries, path):
    """entries: [(label, it, metrics)] -> stacked per-run subplots, shared x."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(entries)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(11, 2.3 * n + 1),
                             facecolor=C_SURFACE, squeeze=False)
    axes = axes[:, 0]
    for ax, (label, it, m) in zip(axes, entries):
        ax.set_facecolor(C_SURFACE)
        ax.grid(True, color="#e7e6e1", linewidth=0.8)
        for sp in ax.spines.values():
            sp.set_color("#d5d4cd")
        ax.tick_params(colors=C_TEXT2, labelsize=8)
        base, _end = active_window(it)
        fires = sorted(r[7] - base for r in it.recs)
        ax.step([f / 1e3 for f in fires], range(1, len(fires) + 1),
                where="post", color=C_FIRED, linewidth=1.8)
        ax.set_ylabel("fired", color=C_FIRED, fontsize=9)
        ax2 = ax.twinx()
        ax2.tick_params(colors=C_TEXT2, labelsize=8)
        ts, vs = inflight_curve(it)
        ax2.step([(t - base) / 1e3 if t else 0.0 for t in ts], vs, where="post",
                 color=C_INFLIGHT, linewidth=1.4, alpha=0.9)
        ax2.axhline(m["cap"], color=C_REF, linestyle="--", linewidth=1.0)
        ax2.set_ylabel("in flight", color=C_INFLIGHT, fontsize=9)
        arr = it.arrivals_rel()
        for s, a in sorted(arr.items(), key=lambda kv: kv[1]):
            x = (a - base) / 1e3  # same t0-relative frame as the tile records
            ax.axvline(x, color=C_INTER if is_inter(it, s) else C_INTRA,
                       linestyle=":", linewidth=1.1, alpha=0.85)
        ax.set_title(
            f"{label} — rank {it.rank} i{it.epoch}: longest_low "
            f"{m['longest_low_us']:.0f}µs, deficit {m['deficit_frac']:.3f}, "
            f"span {m['span_us']:.0f}µs",
            color=C_TEXT, fontsize=10, loc="left")
    axes[-1].set_xlabel("time since first tile enter (µs)", color=C_TEXT, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def regime_png(it, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fires = sorted(r[7] for r in it.recs)
    events = sorted([(r[7], +1) for r in it.recs] + [(r[8], -1) for r in it.recs])
    xs, ys, cur = [0.0], [0], 0
    for t, d in events:
        xs.append(t / 1e3)
        ys.append(cur)
        cur += d
        xs.append(t / 1e3)
        ys.append(cur)
    cap = max(ys)
    arr = it.arrivals_rel()  # sources with a true device stamp (blocked someone)
    # every other cohort gets a first-fire proxy vertical (upper bound on
    # arrival), drawn lighter and labeled with a ~ prefix
    first_fire = defaultdict(lambda: float("inf"))
    for r, (_stat, dyn) in zip(it.recs, it.attribute()):
        first_fire[dyn] = min(first_fire[dyn], r[7])
    proxies = {s: t for s, t in first_fire.items() if s not in arr}

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(11, 6.2), height_ratios=[1, 1],
        facecolor=C_SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(C_SURFACE)
        ax.grid(True, color="#e7e6e1", linewidth=0.8)
        for sp in ax.spines.values():
            sp.set_color("#d5d4cd")
        ax.tick_params(colors=C_TEXT2, labelsize=9)
    ax1.step([f / 1e3 for f in fires], range(1, len(fires) + 1),
             where="post", color=C_FIRED, linewidth=2)
    ax1.set_ylabel("tiles fired (cum.)", color=C_TEXT, fontsize=10)
    ax1.annotate("fired", (fires[-1] / 1e3, len(fires)), color=C_FIRED,
                 fontsize=10, ha="right", va="bottom")
    ax2.step(xs, ys, where="post", color=C_INFLIGHT, linewidth=2)
    ax2.axhline(cap, color=C_REF, linestyle="--", linewidth=1.2)
    ax2.annotate(f"capacity ≈ {cap}", (xs[-1], cap), color=C_TEXT2,
                 fontsize=9, ha="right", va="bottom")
    ax2.set_ylabel("tiles in flight", color=C_TEXT, fontsize=10)
    ax2.set_xlabel("time since kernel start (µs)", color=C_TEXT, fontsize=10)
    ax2.annotate("in-flight", (xs[len(xs) // 3], max(ys) * 0.55),
                 color=C_INFLIGHT, fontsize=10)
    for stamped, group in ((True, arr), (False, proxies)):
        for s, a in sorted(group.items(), key=lambda kv: kv[1]):
            col = C_INTER if is_inter(it, s) else C_INTRA
            alpha = 0.9 if stamped else 0.45
            for ax in (ax1, ax2):
                ax.axvline(a / 1e3, color=col, linestyle=":",
                           linewidth=1.4 if stamped else 1.0, alpha=alpha)
            label = src_name(it, s) if stamped else "~" + src_name(it, s)
            ax1.annotate(label, (a / 1e3, len(fires) * 1.01), color=col,
                         fontsize=8, ha="center", va="bottom", rotation=45,
                         alpha=1.0 if stamped else 0.6)
    ax1.set_title(
        f"a2av tile firing — rank {it.rank}, i{it.epoch} "
        "(verticals: device arrival, ~first-fire proxy; inter red, intra aqua)",
        color=C_TEXT, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def gantt_json(iters, path, align_db=None):
    import json
    offsets = {}
    if align_db:
        offsets = kernel_offsets(align_db, iters)
    ev = []
    for it in iters:
        base = offsets.get((it.rank, it.epoch), 0)
        pid = it.rank
        ev.append({"ph": "M", "pid": pid, "name": "process_name",
                   "args": {"name": f"rank {it.rank}"}})
        attribution = it.attribute()
        for r, (_stat, dyn) in zip(it.recs, attribution):
            smid = r[2]
            t_enter, t_fire, t_done = (r[6] + base) / 1e3, (r[7] + base) / 1e3, (r[8] + base) / 1e3
            if t_fire > t_enter:
                ev.append({"ph": "X", "pid": pid, "tid": smid, "ts": t_enter,
                           "dur": t_fire - t_enter, "name": "spin",
                           "cat": "spin", "args": {"src": src_name(it, dyn)}})
            ev.append({"ph": "X", "pid": pid, "tid": smid, "ts": t_fire,
                       "dur": max(t_done - t_fire, 0.001),
                       "name": f"i{it.epoch}.{src_name(it, dyn)}",
                       "cat": "inter" if is_inter(it, dyn) else "intra",
                       "args": {"problem": r[0], "tile": r[1], "cta": r[5]}})
    with open(path, "w") as f:
        json.dump({"traceEvents": ev, "displayTimeUnit": "ms"}, f)
    print(f"wrote {path} ({len(ev)} events) — open in ui.perfetto.dev")


def kernel_offsets(db_path, iters):
    """Pair the k-th AGScatter kernel start (per device) with the k-th traced
    iteration of the rank on that device. Best effort; falls back to 0."""
    import sqlite3
    db = sqlite3.connect(db_path)
    try:
        rows = db.execute(
            "SELECT k.start, k.deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.shortName = s.id "
            "WHERE s.value LIKE '%agscatter%' OR s.value LIKE '%AGScatter%' "
            "ORDER BY k.deviceId, k.start").fetchall()
    except sqlite3.OperationalError as e:
        print(f"--align: {e}; keeping relative times", file=sys.stderr)
        return {}
    per_dev = defaultdict(list)
    for start, dev in rows:
        per_dev[dev].append(start)
    out = {}
    for rank in {it.rank for it in iters}:
        its = sorted((it for it in iters if it.rank == rank), key=lambda i: i.epoch)
        starts = per_dev.get(rank % 8, [])
        if len(starts) != len(its):
            print(f"--align: rank {rank}: {len(starts)} kernels vs {len(its)} "
                  f"iterations; keeping relative times", file=sys.stderr)
            continue
        for it, st in zip(its, starts):
            out[(it.rank, it.epoch)] = st
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*",
                    help="sidecar file(s) or a records dir containing them")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--iter", default="last", help="epoch number or 'last'")
    ap.add_argument("--curves", metavar="PNG")
    ap.add_argument("--gantt", metavar="JSON")
    ap.add_argument("--align", metavar="SQLITE", help="nsys sqlite export for wall-clock rebase")
    ap.add_argument("--no-table", action="store_true")
    ap.add_argument("--scan-ranks", action="store_true",
                    help="per-rank starvation table over all ranks in the inputs")
    ap.add_argument("--compare", action="append", metavar="LABEL=DIR[:RANK]",
                    help="add a run to the comparison figure (repeatable)")
    ap.add_argument("--compare-out", metavar="PNG", default="a2av_compare.png")
    args = ap.parse_args()

    def load(paths):
        files = []
        for p in paths:
            files += sorted(glob.glob(os.path.join(p, "a2av_tile_trace_r*.bin"))) \
                if os.path.isdir(p) else [p]
        if not files:
            raise SystemExit(f"no sidecar files found under {paths}")
        return [it for f in files for it in read_sidecar(f)]

    if args.compare:
        entries = []
        for spec in args.compare:
            label, _, loc = spec.partition("=")
            if not loc:
                raise SystemExit(f"--compare wants LABEL=DIR[:RANK], got {spec!r}")
            loc, _, rank_s = loc.rpartition(":") if loc.rpartition(":")[2].isdigit() \
                else (loc, "", "")
            runs = load([loc])
            if rank_s:
                rank = int(rank_s)
            else:
                rank, _rows = scan_ranks(runs, out=sys.stderr)
                print(f"{label}: auto-picked worst rank {rank}", file=sys.stderr)
            sel, clean = pick_clean(sorted((i for i in runs if i.rank == rank),
                                           key=lambda i: i.epoch))
            if sel is None:
                raise SystemExit(f"{label}: no iterations for rank {rank}")
            if not clean:
                print(f"{label}: rank {rank} has no hygiene-clean epoch; "
                      f"using epoch {sel.epoch} (dirty)", file=sys.stderr)
            entries.append((label, sel, starvation_metrics(sel)))
        compare_png(entries, args.compare_out)
        return

    if not args.inputs:
        raise SystemExit("inputs required unless --compare is used")
    iters = load(args.inputs)
    if args.scan_ranks:
        scan_ranks(iters)
        if not (args.curves or args.gantt):
            return
    mine = sorted((it for it in iters if it.rank == args.rank), key=lambda i: i.epoch)
    if not mine:
        raise SystemExit(f"no iterations for rank {args.rank} "
                         f"(ranks present: {sorted({i.rank for i in iters})})")
    sel = mine[-1] if args.iter == "last" else \
        next((i for i in mine if i.epoch == int(args.iter)), None)
    if sel is None:
        raise SystemExit(f"iteration {args.iter} not found "
                         f"(have {[i.epoch for i in mine]})")

    if not args.no_table:
        cohort_table(sel)
    if args.curves:
        regime_png(sel, args.curves)
    if args.gantt:
        gantt_json([sel], args.gantt, align_db=args.align)


if __name__ == "__main__":
    main()
