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
  --gw-marks SQLITE|DIR  gateway "node-landing" markers (purple dashed): per
                   inter-node source, the device-side start of the same-local-
                   rank gateway's intra-node redistribution on this node — an
                   upper bound on when the payload finished its wire hop. Needs
                   the per-node nsys sqlite (all 8 local ranks share one
                   timebase; the destination and its gateways are always in the
                   same report). --wire hier|union|identity|balanced selects the
                   extraction rule (inferred from the path when possible);
                   balanced draws per-GATEWAY markers (the balanced relay cuts
                   the wire across gateways, so no per-source mapping exists).
                   --gw-marks-debug dumps the enqueue-ordered event tables.

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

C_FIRED = "#2a78d6"  # categorical slot 1 (blue)
C_INFLIGHT = "#eb6834"  # categorical slot 2 (orange)
C_INTRA = "#1baf7a"  # aqua — intra-node arrivals (matches NVTX green-ish)
C_INTER = "#e34948"  # red — inter-node arrivals (matches NVTX red)
C_GW = "#8054d1"  # purple — gateway node-landing markers (dashed)
C_REF = "#9a9994"  # neutral reference lines
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
        self.arrival_gt = []  # u64 abs, valid iff ready_seq >= epoch
        self.ready_seq = []
        self.expected = []
        self.rows = []  # v2: per-source ROW counts (ground truth; [] on v1)
        self.recs = []  # (problem, tile, smid, seg0, seg1, cta, t_enter, t_fire, t_done) ns rel

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
                it.recs.append(
                    (
                        tile >> 22,
                        tile & 0x3FFFFF,  # problem, tile idx
                        meta >> 24,
                        (meta >> 18) & 0x3F,  # smid, seg_start
                        (meta >> 12) & 0x3F,
                        meta & 0xFFF,  # seg_end, cta
                        u32_rebase(te, t0),
                        u32_rebase(tf, t0),
                        u32_rebase(td, t0),
                    )
                )
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
    print(
        f"\n== rank {it.rank} i{it.epoch}: {len(it.recs)} tiles, "
        f"{len(coh)} cohorts, static!=dynamic on {mismatch} tiles ==",
        file=out,
    )
    print(
        f"{'cohort':>8} {'kind':>5} {'arrival_us':>10} {'n':>5} "
        f"{'fire_p50':>9} {'fire_p95':>9} {'drain_us':>9}",
        file=out,
    )

    def pct(v, p):
        v = sorted(v)
        return v[min(len(v) - 1, int(p * len(v)))]

    if not arr:
        print(
            "  (no device arrival stamps: no tile ever blocked in the "
            "backstop spin — SM-limited iteration; arrival_us falls back "
            "to '-', ramps are relative to each cohort's first fire)",
            file=out,
        )
    order = sorted(coh, key=lambda s: arr.get(s, min(r[7] for r in coh[s])))
    for s in order:
        rs = coh[s]
        a = arr.get(s)
        fires = [r[7] for r in rs]
        done = max(r[8] for r in rs)
        base = a if a is not None else min(fires)
        a_txt = f"{a / 1e3:>10.1f}" if a is not None else f"{'-':>10}"
        print(
            f"{src_name(it, s):>8} {'inter' if is_inter(it, s) else 'intra':>5} "
            f"{a_txt} {len(rs):>5} "
            f"{(pct(fires, .5) - base) / 1e3:>9.1f} {(pct(fires, .95) - base) / 1e3:>9.1f} "
            f"{(done - base) / 1e3:>9.1f}",
            file=out,
        )


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
    events = sorted([(r[7] - base, +1) for r in it.recs] + [(r[8] - base, -1) for r in it.recs])
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
    modal_epoch = Counter(sel.epoch for sel, clean in picked.values() if clean).most_common(1)
    modal_epoch = modal_epoch[0][0] if modal_epoch else None
    rows = {}
    for rank, (sel, clean) in picked.items():
        clean = clean and sel.epoch == modal_epoch
        rows[rank] = (sel, starvation_metrics(sel), clean)
    clean_rows = [r for r in rows if rows[r][2]]
    pool = clean_rows or list(rows)
    worst = max(pool, key=lambda r: (rows[r][1]["longest_low_us"], rows[r][1]["deficit_frac"]))
    print(
        f"{'rank':>4} {'iter':>5} {'tiles':>6} {'span_us':>8} {'first_fire':>10} "
        f"{'cap':>4} {'longest_low_us':>14} {'deficit':>8} {'stamped':>7}",
        file=out,
    )
    for rank, (sel, m, clean) in rows.items():
        mark = "  <- worst" if rank == worst else ""
        dirty = "" if clean else "  !dirty(excluded)"
        print(
            f"{rank:>4} {sel.epoch:>5} {m['n_tiles']:>6} {m['span_us']:>8.1f} "
            f"{m['first_fire_us']:>10.1f} {m['cap']:>4} {m['longest_low_us']:>14.1f} "
            f"{m['deficit_frac']:>8.3f} {m['n_stamped']:>7}{dirty}{mark}",
            file=out,
        )
    return worst, rows


def compare_png(entries, path):
    """entries: [(label, it, metrics)] -> stacked per-run subplots, shared x."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(entries)
    fig, axes = plt.subplots(
        n, 1, sharex=True, figsize=(11, 2.3 * n + 1), facecolor=C_SURFACE, squeeze=False
    )
    axes = axes[:, 0]
    for ax, (label, it, m) in zip(axes, entries):
        ax.set_facecolor(C_SURFACE)
        ax.grid(True, color="#e7e6e1", linewidth=0.8)
        for sp in ax.spines.values():
            sp.set_color("#d5d4cd")
        ax.tick_params(colors=C_TEXT2, labelsize=8)
        base, _end = active_window(it)
        fires = sorted(r[7] - base for r in it.recs)
        ax.step(
            [f / 1e3 for f in fires],
            range(1, len(fires) + 1),
            where="post",
            color=C_FIRED,
            linewidth=1.8,
        )
        ax.set_ylabel("fired", color=C_FIRED, fontsize=9)
        ax2 = ax.twinx()
        ax2.tick_params(colors=C_TEXT2, labelsize=8)
        ts, vs = inflight_curve(it)
        ax2.step(
            [(t - base) / 1e3 if t else 0.0 for t in ts],
            vs,
            where="post",
            color=C_INFLIGHT,
            linewidth=1.4,
            alpha=0.9,
        )
        ax2.axhline(m["cap"], color=C_REF, linestyle="--", linewidth=1.0)
        ax2.set_ylabel("in flight", color=C_INFLIGHT, fontsize=9)
        arr = it.arrivals_rel()
        for s, a in sorted(arr.items(), key=lambda kv: kv[1]):
            x = (a - base) / 1e3  # same t0-relative frame as the tile records
            ax.axvline(
                x,
                color=C_INTER if is_inter(it, s) else C_INTRA,
                linestyle=":",
                linewidth=1.1,
                alpha=0.85,
            )
        ax.set_title(
            f"{label} — rank {it.rank} i{it.epoch}: longest_low "
            f"{m['longest_low_us']:.0f}µs, deficit {m['deficit_frac']:.3f}, "
            f"span {m['span_us']:.0f}µs",
            color=C_TEXT,
            fontsize=10,
            loc="left",
        )
    axes[-1].set_xlabel("time since first tile enter (µs)", color=C_TEXT, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def regime_png(it, path, gw_marks=None):
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
        2, 1, sharex=True, figsize=(11, 6.2), height_ratios=[1, 1], facecolor=C_SURFACE
    )
    for ax in (ax1, ax2):
        ax.set_facecolor(C_SURFACE)
        ax.grid(True, color="#e7e6e1", linewidth=0.8)
        for sp in ax.spines.values():
            sp.set_color("#d5d4cd")
        ax.tick_params(colors=C_TEXT2, labelsize=9)
    ax1.step(
        [f / 1e3 for f in fires], range(1, len(fires) + 1), where="post", color=C_FIRED, linewidth=2
    )
    ax1.set_ylabel("tiles fired (cum.)", color=C_TEXT, fontsize=10)
    ax1.annotate(
        "fired", (fires[-1] / 1e3, len(fires)), color=C_FIRED, fontsize=10, ha="right", va="bottom"
    )
    ax2.step(xs, ys, where="post", color=C_INFLIGHT, linewidth=2)
    ax2.axhline(cap, color=C_REF, linestyle="--", linewidth=1.2)
    ax2.annotate(
        f"capacity ≈ {cap}", (xs[-1], cap), color=C_TEXT2, fontsize=9, ha="right", va="bottom"
    )
    ax2.set_ylabel("tiles in flight", color=C_TEXT, fontsize=10)
    ax2.set_xlabel("time since kernel start (µs)", color=C_TEXT, fontsize=10)
    ax2.annotate("in-flight", (xs[len(xs) // 3], max(ys) * 0.55), color=C_INFLIGHT, fontsize=10)
    for stamped, group in ((True, arr), (False, proxies)):
        for s, a in sorted(group.items(), key=lambda kv: kv[1]):
            col = C_INTER if is_inter(it, s) else C_INTRA
            alpha = 0.9 if stamped else 0.45
            for ax in (ax1, ax2):
                ax.axvline(
                    a / 1e3,
                    color=col,
                    linestyle=":",
                    linewidth=1.4 if stamped else 1.0,
                    alpha=alpha,
                )
            label = src_name(it, s) if stamped else "~" + src_name(it, s)
            ax1.annotate(
                label,
                (a / 1e3, len(fires) * 1.01),
                color=col,
                fontsize=8,
                ha="center",
                va="bottom",
                rotation=45,
                alpha=1.0 if stamped else 0.6,
            )
    for x, label, flagged in gw_marks or []:
        alpha = 0.4 if flagged else 0.9
        for ax in (ax1, ax2):
            ax.axvline(x / 1e3, color=C_GW, linestyle="--", linewidth=1.4, alpha=alpha)
        # second annotation band, above the arrival labels
        ax1.annotate(
            ("!" if flagged else "") + label,
            (x / 1e3, len(fires) * 1.12),
            color=C_GW,
            fontsize=8,
            ha="center",
            va="bottom",
            rotation=45,
            alpha=alpha,
            annotation_clip=False,
        )
    gw_txt = "; purple dashed = node landing" if gw_marks else ""
    ax1.set_title(
        f"a2av tile firing — rank {it.rank}, i{it.epoch} "
        f"(arrivals: inter red, intra aqua, ~ = first-fire proxy{gw_txt})",
        color=C_TEXT,
        fontsize=11,
        loc="left",
        pad=30 if gw_marks else 6,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def gantt_json(iters, path, align_db=None, context=None):
    import json

    offsets = {}
    if align_db:
        offsets = kernel_offsets(align_db, iters, context=context)
    ev = []
    for it in iters:
        base = offsets.get((it.rank, it.epoch), 0)
        pid = it.rank
        ev.append(
            {"ph": "M", "pid": pid, "name": "process_name", "args": {"name": f"rank {it.rank}"}}
        )
        attribution = it.attribute()
        for r, (_stat, dyn) in zip(it.recs, attribution):
            smid = r[2]
            t_enter, t_fire, t_done = (r[6] + base) / 1e3, (r[7] + base) / 1e3, (r[8] + base) / 1e3
            if t_fire > t_enter:
                ev.append(
                    {
                        "ph": "X",
                        "pid": pid,
                        "tid": smid,
                        "ts": t_enter,
                        "dur": t_fire - t_enter,
                        "name": "spin",
                        "cat": "spin",
                        "args": {"src": src_name(it, dyn)},
                    }
                )
            ev.append(
                {
                    "ph": "X",
                    "pid": pid,
                    "tid": smid,
                    "ts": t_fire,
                    "dur": max(t_done - t_fire, 0.001),
                    "name": f"i{it.epoch}.{src_name(it, dyn)}",
                    "cat": "inter" if is_inter(it, dyn) else "intra",
                    "args": {"problem": r[0], "tile": r[1], "cta": r[5]},
                }
            )
    with open(path, "w") as f:
        json.dump({"traceEvents": ev, "displayTimeUnit": "ms"}, f)
    print(f"wrote {path} ({len(ev)} events) — open in ui.perfetto.dev")


# The a2av grouped GEMM demangles to cutlass::Kernel<..._agscatter_...>; its
# shortName is the useless literal 'Kernel', so match the demangled template.
# Block 0 of exactly this kernel stamps t0_gt at entry, so its device-side
# start IS the sidecar time origin on the nsys clock (sub-us skew).
GEMM_ANCHOR_SQL = (
    "SELECT k.start, r.start FROM CUPTI_ACTIVITY_KIND_KERNEL k "
    "JOIN StringIds s ON k.demangledName = s.id "
    "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId "
    "AND (r.globalTid >> 24) = (k.globalPid >> 24) "
    "WHERE k.deviceId = ? AND s.value LIKE '%cutlass::Kernel%' "
    "AND lower(s.value) LIKE '%agscatter%' ORDER BY k.start"
)


def gemm_anchors(db, device, n_expect=None):
    """Per-iteration GEMM device starts (~= t0_gt) and per-iteration HOST
    enqueue windows [rt of this GEMM launch, rt of the next). Under
    FLUX_A2AV_EARLY_LAUNCH the wire sequence is replayed directly behind its
    own GEMM launch, so the rt window is the exact FIFO extent of iteration
    k's stream work — device-time windows would mis-bucket bursts, which can
    execute a whole GEMM duration after their anchor. (None, None) on a count
    mismatch."""
    rows = db.execute(GEMM_ANCHOR_SQL, (device,)).fetchall()
    if not rows or (n_expect is not None and len(rows) != n_expect):
        return None, None
    starts = [r[0] for r in rows]
    rts = [r[1] for r in rows] + [rows[-1][1] + 3_600_000_000_000]
    return starts, list(zip(rts[:-1], rts[1:]))


def local_world(it):
    return it.world // it.nnodes if it.nnodes else it.world


def kernel_offsets(db_path, iters, context=None):
    """Map (rank, epoch) -> nsys GEMM anchor (~= t0_gt). Epoch values are
    nvshmem run-ids, one per executed iteration, so they pair with the
    device's k-th anchor by VALUE; `context` (all loaded iterations, default
    `iters`) establishes each rank's epoch base. Best effort; falls back to 0."""
    import sqlite3

    ctx = context or iters
    db = sqlite3.connect(db_path)
    out = {}
    try:
        for rank in {it.rank for it in iters}:
            eps = sorted({i.epoch for i in ctx if i.rank == rank})
            e_min, span = eps[0], eps[-1] - eps[0] + 1
            anchors, _ = gemm_anchors(
                db, rank % local_world(next(i for i in ctx if i.rank == rank)), n_expect=span
            )
            if anchors is None:
                print(
                    f"--align: rank {rank}: GEMM anchor count != {span} "
                    f"epochs; keeping relative times",
                    file=sys.stderr,
                )
                continue
            for it in (i for i in iters if i.rank == rank):
                if 0 <= it.epoch - e_min < len(anchors):
                    out[(it.rank, it.epoch)] = anchors[it.epoch - e_min]
    except sqlite3.OperationalError as e:
        print(f"--align: {e}; keeping relative times", file=sys.stderr)
        return {}
    finally:
        db.close()
    return out


# ---- gateway node-landing markers (--gw-marks) ------------------------------
# The wire hop of inter-node traffic lands on the same-local-rank gateway of
# the destination node, which then redistributes intra-node. The device-side
# start of that redistribution (CE copies for hier/union on cp_stream;
# index_select gathers + puts on the pack stream for identity/balanced) is a
# tight upper bound on the payload's node arrival: it executes directly behind
# the cuStreamWaitValue64 on the wire-arrival signal. Host/API timestamps are
# meaningless under FLUX_A2AV_EARLY_LAUNCH (the whole sequence is pre-enqueued)
# — but the host ENQUEUE ORDER (RUNTIME rows joined on correlationId + pid;
# correlationIds collide across the node's 8 processes) is the code's FIFO
# order, which device-start order does not preserve (copy engines overlap
# adjacent copies across streams). Boundaries count in enqueue order; marker
# times are device starts.

GW_COPIES_SQL = (
    "SELECT r.start, m.start, m.end, m.bytes, m.copyKind, m.streamId "
    "FROM CUPTI_ACTIVITY_KIND_MEMCPY m "
    "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = m.correlationId "
    "AND (r.globalTid >> 24) = (m.globalPid >> 24) "
    "WHERE m.deviceId = ? AND m.copyKind IN (8, 10) "  # 8 DtoD, 10 PtoP
    "AND r.start >= ? AND r.start < ? ORDER BY r.start"
)


def resolve_gw_db(path, node):
    if os.path.isfile(path):
        return path
    for d in (path, os.path.join(path, "nsys")):
        hits = sorted(glob.glob(os.path.join(d, f"node{node}_*.sqlite")))
        if hits:
            return hits[0]
        reps = sorted(glob.glob(os.path.join(d, f"node{node}_*.nsys-rep")))
        if reps:
            raise SystemExit(
                f"--gw-marks: no sqlite for node {node}; export first:\n"
                f"  nsys export --type sqlite -o {reps[0][:-9]}.sqlite {reps[0]}"
            )
    raise SystemExit(f"--gw-marks: no node{node}_*.sqlite under {path}")


def device_copies(db, device, lo, hi):
    """Enqueue-ordered D2D+P2P copies of one device inside one iteration's
    HOST enqueue window: [(rt_start, dev_start, dev_end, bytes, copyKind,
    streamId)]. The RUNTIME join drops nothing (checked once per device)."""
    if device not in device_copies._checked:
        device_copies._checked.add(device)
        n_all, n_join = [
            db.execute(s, (device,)).fetchone()[0]
            for s in (
                "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                "WHERE deviceId = ? AND copyKind IN (8, 10)",
                "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY m "
                "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r "
                "ON r.correlationId = m.correlationId "
                "AND (r.globalTid >> 24) = (m.globalPid >> 24) "
                "WHERE m.deviceId = ? AND m.copyKind IN (8, 10)",
            )
        ]
        if n_all != n_join:
            print(
                f"  gw dev {device}: {n_all - n_join} copies lack a RUNTIME "
                f"row; enqueue order incomplete",
                file=sys.stderr,
            )
    return db.execute(GW_COPIES_SQL, (device, lo, hi)).fetchall()


device_copies._checked = set()


def pack_stream_id(db, device):
    """Streams carrying payload-dtype index_select kernels minus the GEMM's
    stream — under EARLY_LAUNCH the surviving stream is the t_* tail (pack)
    stream that the identity/balanced gateway gathers ride. Returns a set."""
    cand = {
        r[0]
        for r in db.execute(
            "SELECT DISTINCT k.streamId FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds sn ON k.shortName = sn.id "
            "JOIN StringIds dn ON k.demangledName = dn.id "
            "WHERE k.deviceId = ? AND sn.value LIKE 'indexSelect%' "
            "AND dn.value NOT LIKE '%<long%'",
            (device,),
        )
    }
    gemm = {
        r[0]
        for r in db.execute(
            "SELECT DISTINCT k.streamId FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.demangledName = s.id "
            "WHERE k.deviceId = ? AND s.value LIKE '%cutlass::Kernel%' "
            "AND lower(s.value) LIKE '%agscatter%'",
            (device,),
        )
    }
    return cand - gemm


def pack_events(db, device, streams, lo, hi):
    """Device-start-ordered pack-stream events (index_select kernels + copies)
    of one iteration's HOST enqueue window: [(dev_start, dev_end, kind_str,
    bytes|None)]"""
    ph = ",".join("?" * len(streams))
    rt = (
        "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = "
        "{0}.correlationId AND (r.globalTid >> 24) = ({0}.globalPid >> 24)"
    )
    evs = [
        (s, e, "idxsel", None)
        for s, e in db.execute(
            f"SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            f"JOIN StringIds sn ON k.shortName = sn.id {rt.format('k')} "
            f"WHERE k.deviceId = ? AND k.streamId IN ({ph}) "
            f"AND sn.value LIKE 'indexSelect%' AND r.start >= ? AND r.start < ?",
            (device, *streams, lo, hi),
        )
    ]
    evs += [
        (s, e, "PtoP" if k == 10 else "DtoD", b)
        for s, e, b, k in db.execute(
            f"SELECT m.start, m.end, m.bytes, m.copyKind "
            f"FROM CUPTI_ACTIVITY_KIND_MEMCPY m {rt.format('m')} "
            f"WHERE m.deviceId = ? AND m.streamId IN ({ph}) "
            f"AND m.copyKind IN (8, 10) AND r.start >= ? AND r.start < ?",
            (device, *streams, lo, hi),
        )
    ]
    return sorted(evs)


def _stall_before(copies, boundary):
    """Device-time gap (ns) between the forward set and everything before it;
    None when the boundary is the window start."""
    if boundary == 0:
        return None
    return min(c[1] for c in copies[boundary:]) - max(c[2] for c in copies[:boundary])


def _signature_runs(copies, m, match):
    """Start indices i where copies[i:i+m] satisfies match(run) — the forward
    subsequence is located by its size signature, never by position (harness /
    companion copies may be enqueued after the forwards in the window)."""
    return [i for i in range(len(copies) - m + 1) if match(copies[i : i + m])]


def _run_result(copies, runs, m):
    i = runs[-1]
    note = f"ambiguous ({len(runs)} signature runs)" if len(runs) > 1 else ""
    return (
        dict(
            t=min(c[1] for c in copies[i : i + m]),
            stall=_stall_before(copies, i) if i else None,
            note=note,
        ),
        None,
    )


def extract_union(copies, L):
    """Forwards = the enqueue-contiguous run of exactly L equal-size copies
    with the self loopback (DtoD) enqueue-first: the WHOLE staged union goes
    to every local rank."""
    if len(copies) < L:
        return None, f"only {len(copies)} copies in window (< L={L})"

    def match(run):
        return len({c[3] for c in run}) == 1 and [i for i, c in enumerate(run) if c[4] == 8] == [0]

    runs = _signature_runs(copies, L, match)
    if not runs:
        return None, f"no run of {L} equal-size DtoD-first copies"
    return _run_result(copies, runs, L)


def extract_lb_union(copies, L):
    """lb_union: the forward is P >= 1 enqueue-consecutive runs of exactly L
    equal-size copies — one run per window piece, every piece broadcast to all
    L local ranks, DtoD loopback enqueue-first per run. P is not predictable
    from the sidecar (piece count needs U plus the chunk bounds), so chain
    consecutive matching runs and take the earliest device start of the chain
    (the first forward copy, gated by the node_sig wait)."""
    if len(copies) < L:
        return None, f"only {len(copies)} copies in window (< L={L})"

    def match(run):
        return len({c[3] for c in run}) == 1 and [i for i, c in enumerate(run) if c[4] == 8] == [0]

    runs = _signature_runs(copies, L, match)
    if not runs:
        return None, f"no run of {L} equal-size DtoD-first copies"
    chains = []
    for i in runs:
        if chains and i == chains[-1][-1] + L:
            chains[-1].append(i)
        else:
            chains.append([i])
    chain = chains[-1]
    i0, m = chain[0], L * len(chain)
    note = f"ambiguous ({len(chains)} chains)" if len(chains) > 1 else ""
    if len(chain) > 1:
        note = (note + "; " if note else "") + f"{len(chain)} pieces"
    return (
        dict(
            t=min(c[1] for c in copies[i0 : i0 + m]),
            stall=_stall_before(copies, i0) if i0 else None,
            note=note,
        ),
        None,
    )


def extract_hier(copies, L, g_lr, src, rows_by_local):
    """Forwards = the enqueue-contiguous run matching the exact per-destination
    size signature (mirror order, zero sub-chunks skipped) from sidecar
    rows[]. Sizes vary per destination, making the signature near-unique."""
    exp = []
    for dl in range(L):
        d_local = (g_lr - dl) % L
        rows = rows_by_local.get(d_local)
        if not rows:
            return None, f"no v2 sidecar rows[] for local rank {d_local}"
        if rows[src] > 0:
            exp.append((d_local, rows[src]))
    if not exp:
        return None, "source sends zero rows to every local rank"
    m = len(exp)
    if len(copies) < m:
        return None, f"{len(copies)} copies < {m} expected forwards"

    def match(run):
        if run[0][3] % exp[0][1]:
            return False
        rb = run[0][3] // exp[0][1]
        return (
            rb > 0
            and all(c[3] == e[1] * rb for c, e in zip(run, exp))
            and (exp[0][0] != g_lr or run[0][4] == 8)
        )  # self fwd is DtoD

    runs = _signature_runs(copies, m, match)
    if not runs:
        return None, (f"no run matching rows signature " f"{[e[1] for e in exp]} (x row_bytes)")
    return _run_result(copies, runs, m)


def extract_pack(evs, n_gather=None, n_put=None):
    """identity/balanced: the pack stream carries ONLY the gather tail
    (per-dest index_select + put), enqueued at dispatch and gated by the wire
    wait — so its first event by device start IS the node-landing marker.
    Expected-count soft checks (from sidecar rows[]) catch a polluted stream."""
    if not evs:
        return None, "no pack-stream events in window"
    note = ""
    if n_gather is not None:
        got_g = sum(1 for e in evs if e[2] == "idxsel")
        got_p = sum(1 for e in evs if e[2] == "PtoP")
        if (got_g, got_p) != (n_gather, n_put):
            note = (
                f"count mismatch: {got_g} gathers/{got_p} puts " f"vs expected {n_gather}/{n_put}"
            )
    return dict(t=evs[0][0], stall=None, note=note), None


def _dump_events(rows, t0, mark_t, out=sys.stderr):
    for r in rows:
        if len(r) == 6:  # enqueue-ordered copy tuple
            line = (
                f"    dev {(r[1] - t0) / 1e3:9.1f}us dur {(r[2] - r[1]) / 1e3:7.1f}us "
                f"{'DtoD' if r[4] == 8 else 'PtoP'} {r[3]:>10}B s{r[5]}"
            )
            t = r[1]
        else:  # pack event tuple
            line = (
                f"    dev {(r[0] - t0) / 1e3:9.1f}us dur {(r[1] - r[0]) / 1e3:7.1f}us "
                f"{r[2]:<6} {r[3] if r[3] is not None else '':>10}"
            )
            t = r[0]
        print(line + ("   <-- MARKER" if t == mark_t else ""), file=out)


def compute_gw_marks(db_path, sel, iters, wire, debug=False, stamped_only=False):
    """(marks, t0_dst_nsys, {local_rank: anchor_nsys at sel.epoch}) where
    marks = [(x_ns_rel_t0, label, flagged)] for the selected destination
    iteration, plus the printed audit table. t0/anchors let callers (--mega)
    place everything on the absolute node timeline. Every guard degrades to
    'print and skip'."""
    import sqlite3

    L, NN = local_world(sel), sel.nnodes
    if NN != 2:
        raise SystemExit(
            f"--gw-marks: nnodes={NN} unsupported (per-round "
            "window mapping is only implemented for 2 nodes)"
        )
    node = sel.rank // L
    db = sqlite3.connect(resolve_gw_db(db_path, node))
    by_rank = defaultdict(list)
    for it in iters:
        by_rank[it.rank].append(it)
    # epoch values are nvshmem run-ids: one per executed iteration, globally
    # synced. Map epoch -> anchor index by VALUE (epoch - min node epoch), not
    # by list position — robust to a rank's sidecar missing/merging a block.
    node_epochs = sorted({i.epoch for r, its in by_rank.items() for i in its if r // L == node})
    if sel.epoch not in node_epochs:
        print("--gw-marks: selected epoch missing from node sidecars", file=sys.stderr)
        return [], None, {}
    e_min, span = node_epochs[0], node_epochs[-1] - node_epochs[0] + 1
    k = sel.epoch - e_min
    anchors, windows = {}, {}
    for lr in range(L):
        a, w = gemm_anchors(db, lr, n_expect=span)
        if a is None:
            print(
                f"--gw-marks: dev {lr}: GEMM anchor count != {span} "
                f"(epochs {e_min}..{node_epochs[-1]}); gateway skipped",
                file=sys.stderr,
            )
            continue
        anchors[lr], windows[lr] = a, w
    dst_lr = sel.rank % L
    if dst_lr not in anchors:
        print("--gw-marks: destination device unanchored; markers skipped", file=sys.stderr)
        return [], None, {}
    t0_dst = anchors[dst_lr][k]
    # rows[] depends only on the (fixed) traffic matrix, so any epoch of the
    # rank serves; prefer the selected one, note when a fallback fills a gap
    rows_by_local = {}
    for lr in range(L):
        cand = sorted(by_rank.get(node * L + lr, []), key=lambda i: (i.epoch != sel.epoch, i.epoch))
        for it in cand:
            if it.rows:
                rows_by_local[lr] = it.rows
                if it.epoch != sel.epoch:
                    print(
                        f"  gw: rows[] for local rank {lr} from epoch "
                        f"{it.epoch} (none at i{sel.epoch}; matrix is "
                        f"iteration-invariant)",
                        file=sys.stderr,
                    )
                break
    arr = sel.arrivals_rel()
    marks = []
    hdr = (
        f"== gateway markers: wire={wire} node={node} dst rank {sel.rank} "
        f"i{sel.epoch} (x is us since dst GEMM anchor ~= t0) =="
    )
    lines = [
        hdr,
        f"{'gw':>4} {'src':>6} {'marker_us':>10} {'flag_us':>9} "
        f"{'delta_us':>9} {'stall_us':>9}  note",
    ]
    for lr in sorted(anchors):
        lo, hi = windows[lr][k]
        src = (1 - node) * L + lr if wire not in ("balanced", "lb_union") else None
        if wire in ("identity", "balanced"):
            # the gather tails are issued inline at dispatch, BEFORE their own
            # GEMM launch (never deferred — dispatch-starvation rule), so their
            # enqueue window is the preceding inter-launch segment
            lo = windows[lr][k - 1][0] if k else 0
            hi = windows[lr][k][0]
            streams = pack_stream_id(db, lr)
            if streams:
                evs = pack_events(db, lr, streams, lo, hi)
                n_g = n_p = None
                if wire == "identity" and all(rows_by_local.get(d) for d in range(L)):
                    n_g = sum(1 for d in range(L) if rows_by_local[d][src] > 0)
                    n_p = sum(1 for d in range(L) if d != lr and rows_by_local[d][src] > 0)
                res, err = extract_pack(evs, n_g, n_p)
            else:
                res, err = None, "no pack stream identified"
            if debug and streams:
                print(
                    f"  -- gw dev {lr} pack streams {sorted(streams)} "
                    f"({len(evs)} events in enqueue window, dev-time order)",
                    file=sys.stderr,
                )
                _dump_events(evs, t0_dst, res["t"] if res else None)
        else:
            if wire == "lb_union":
                # like the identity/balanced tails, the lb_union forward is
                # issued inline at dispatch (t_*, pure CE puts — no idxsel, so
                # pack_stream_id can't find it): its enqueue window is the
                # preceding inter-launch segment, but the signature match runs
                # over device copies like union
                lo = windows[lr][k - 1][0] if k else 0
                hi = windows[lr][k][0]
            copies = device_copies(db, lr, lo, hi)
            if wire == "union":
                res, err = extract_union(copies, L)
            elif wire == "lb_union":
                res, err = extract_lb_union(copies, L)
            else:
                res, err = extract_hier(copies, L, lr, src, rows_by_local)
            if debug:
                print(
                    f"  -- gw dev {lr} ({len(copies)} copies in enqueue " f"window, enqueue order)",
                    file=sys.stderr,
                )
                _dump_events(copies, t0_dst, res["t"] if res else None)
        s_lab = f"src{src}" if src is not None else "-"
        if res is None:
            lines.append(
                f"{lr:>4} {s_lab:>6} {'-':>10} {'-':>9} {'-':>9} " f"{'-':>9}  SKIP: {err}"
            )
            continue
        x = res["t"] - t0_dst
        flag = arr.get(src) if src is not None else None
        note = res.get("note", "")
        flagged = "no-stall" in note
        if flag is not None:
            delta = (flag - x) / 1e3
            if not 0 < delta < 5000:
                note += " WARN:marker/flag order"
                flagged = True
            f_txt, d_txt = f"{flag / 1e3:>9.1f}", f"{delta:>9.1f}"
        else:
            f_txt, d_txt = f"{'-':>9}", f"{'-':>9}"
        st = res["stall"]
        st_txt = f"{st / 1e3:>9.1f}" if st is not None else f"{'-':>9}"
        if stamped_only and src is not None and flag is None:
            note += " unstamped (not drawn)"
            lines.append(
                f"{lr:>4} {s_lab:>6} {x / 1e3:>10.1f} {f_txt} {d_txt} " f"{st_txt}  {note}"
            )
            continue
        lines.append(f"{lr:>4} {s_lab:>6} {x / 1e3:>10.1f} {f_txt} {d_txt} " f"{st_txt}  {note}")
        marks.append((x, f"gw:src{src}" if src is not None else f"gw{lr}", flagged))
    if wire in ("balanced", "lb_union") and marks:
        inter = [a for s, a in arr.items() if is_inter(sel, s)]
        if inter:
            first = min(inter) / 1e3
            lines.append(
                f"  min inter-node flag arrival on dst: {first:.1f}us "
                "(every gateway marker must precede it)"
            )
    print("\n".join(lines))
    db.close()
    return marks, t0_dst, {lr: a[k] for lr, a in anchors.items()}


def mega_png(iters, sel, gw_abs, anchors_k, path):
    """Node-wide absolute-timeline figure: one 'fired' strip per local rank
    (top block, blue) and one 'in-flight' strip per local rank (bottom block,
    orange), each placed at its rank's TRUE GEMM launch time. The node's
    gateway landings are single full-height verticals — fixed events on the
    node clock, identical across strips by construction. Per strip, only that
    rank's stamped inter-node flag flips are drawn (solid red)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = local_world(sel)
    node = sel.rank // L
    per = {
        it.rank % L: it
        for it in iters
        if it.rank // L == node and it.epoch == sel.epoch and it.recs
    }
    x0 = min(anchors_k.values())
    fired_max = max(len(it.recs) for it in per.values())
    curves, cap_max = {}, 0
    for lr, it in per.items():
        ts, vs = inflight_curve(it)
        curves[lr] = (sorted(r[7] for r in it.recs), ts, vs)
        cap_max = max(cap_max, max(vs))
    fig, axes = plt.subplots(
        2 * L, 1, sharex=True, figsize=(12.5, 0.78 * 2 * L + 1.6), facecolor=C_SURFACE
    )
    for i, ax in enumerate(axes):
        top = i < L
        lr = i if top else i - L
        ax.set_facecolor(C_SURFACE)
        ax.grid(True, axis="x", color="#e7e6e1", linewidth=0.7)
        for sp in ax.spines.values():
            sp.set_color("#d5d4cd")
        ax.tick_params(colors=C_TEXT2, labelsize=7)
        for x, _lab, flagged in gw_abs:
            ax.axvline(
                (x - x0) / 1e3,
                color=C_GW,
                linestyle="--",
                linewidth=1.1,
                alpha=0.35 if flagged else 0.7,
            )
        ax.set_ylabel(
            f"r{node * L + lr}",
            rotation=0,
            ha="right",
            va="center",
            color=C_FIRED if top else C_INFLIGHT,
            fontsize=9,
        )
        it = per.get(lr)
        a = anchors_k.get(lr)
        if it is None or a is None:
            ax.set_yticks([])
            continue
        if top:
            fires = curves[lr][0]
            ax.step(
                [(a + f - x0) / 1e3 for f in fires],
                range(1, len(fires) + 1),
                where="post",
                color=C_FIRED,
                linewidth=1.3,
            )
            ax.set_ylim(0, fired_max * 1.1)
        else:
            _, ts, vs = curves[lr]
            ax.step(
                [(a + t - x0) / 1e3 for t in ts], vs, where="post", color=C_INFLIGHT, linewidth=1.0
            )
            ax.axhline(cap_max, color=C_REF, linestyle="--", linewidth=0.6, alpha=0.5)
            ax.set_ylim(0, cap_max * 1.18)
        # magnitude anchor on the first strip of each block only
        ax.set_yticks([0, fired_max if top else cap_max] if lr == 0 else [])
        for s, t_arr in sorted(it.arrivals_rel().items()):
            if is_inter(it, s):
                ax.axvline(
                    (a + t_arr - x0) / 1e3, color=C_INTER, linestyle=":", linewidth=1.0, alpha=0.9
                )
    for x, lab, flagged in gw_abs:
        axes[0].annotate(
            lab,
            ((x - x0) / 1e3, fired_max * 1.18),
            color=C_GW,
            fontsize=8,
            ha="center",
            va="bottom",
            rotation=45,
            alpha=0.5 if flagged else 1.0,
            annotation_clip=False,
        )
    axes[0].set_title(
        f"a2av tile firing — node {node}, i{sel.epoch}, absolute node timeline"
        " (top: fired cum., bottom: in-flight; purple = node landings,"
        " red = stamped inter flag flips)",
        color=C_TEXT,
        fontsize=11,
        loc="left",
        pad=36,
    )
    axes[-1].set_xlabel("time since first GEMM launch on node (µs)", color=C_TEXT, fontsize=10)
    fig.tight_layout(h_pad=0.25)
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def infer_wire(path):
    p = path.lower()
    for key, mode in (
        ("hier_compress_lb_union", "lb_union"),
        ("hier_compress_union", "union"),
        ("hier_compress_identity", "identity"),
        ("hier_compress", "balanced"),
        ("hier", "hier"),
    ):
        if key in p:
            return mode
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("inputs", nargs="*", help="sidecar file(s) or a records dir containing them")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--iter", default="last", help="epoch number or 'last'")
    ap.add_argument("--curves", metavar="PNG")
    ap.add_argument("--gantt", metavar="JSON")
    ap.add_argument("--align", metavar="SQLITE", help="nsys sqlite export for wall-clock rebase")
    ap.add_argument("--no-table", action="store_true")
    ap.add_argument(
        "--scan-ranks",
        action="store_true",
        help="per-rank starvation table over all ranks in the inputs",
    )
    ap.add_argument(
        "--compare",
        action="append",
        metavar="LABEL=DIR[:RANK]",
        help="add a run to the comparison figure (repeatable)",
    )
    ap.add_argument("--compare-out", metavar="PNG", default="a2av_compare.png")
    ap.add_argument(
        "--gw-marks",
        metavar="SQLITE|DIR",
        help="per-node nsys sqlite (or dir holding node<k>_*.sqlite"
        " / an nsys/ subdir): add gateway node-landing markers",
    )
    ap.add_argument(
        "--wire",
        choices=["hier", "union", "identity", "balanced", "lb_union"],
        help="wire mode of the capture (default: inferred from the " "--gw-marks path)",
    )
    ap.add_argument(
        "--gw-marks-debug",
        action="store_true",
        help="dump the enqueue-ordered per-gateway event tables " "with the chosen boundary",
    )
    ap.add_argument(
        "--mega",
        metavar="PNG",
        help="node-wide absolute-timeline figure: per-rank fired "
        "strips (top) + in-flight strips (bottom) for the "
        "node of --rank at --iter, with the gateway landings "
        "as fixed full-height verticals; requires --gw-marks",
    )
    ap.add_argument(
        "--gw-stamped-only",
        action="store_true",
        help="draw only gateway markers whose source has a TRUE "
        "device arrival stamp (a tile actually spun on the "
        "flag flip); ~proxy-only sources are table-listed "
        "but not drawn. Per-source wires only (balanced "
        "markers have no source mapping and are kept)",
    )
    args = ap.parse_args()

    def load(paths):
        files = []
        for p in paths:
            files += (
                sorted(glob.glob(os.path.join(p, "a2av_tile_trace_r*.bin")))
                if os.path.isdir(p)
                else [p]
            )
        if not files:
            raise SystemExit(f"no sidecar files found under {paths}")
        return [it for f in files for it in read_sidecar(f)]

    if args.compare:
        if args.gw_marks:
            print("--gw-marks ignored with --compare", file=sys.stderr)
        entries = []
        for spec in args.compare:
            label, _, loc = spec.partition("=")
            if not loc:
                raise SystemExit(f"--compare wants LABEL=DIR[:RANK], got {spec!r}")
            loc, _, rank_s = (
                loc.rpartition(":") if loc.rpartition(":")[2].isdigit() else (loc, "", "")
            )
            runs = load([loc])
            if rank_s:
                rank = int(rank_s)
            else:
                rank, _rows = scan_ranks(runs, out=sys.stderr)
                print(f"{label}: auto-picked worst rank {rank}", file=sys.stderr)
            sel, clean = pick_clean(
                sorted((i for i in runs if i.rank == rank), key=lambda i: i.epoch)
            )
            if sel is None:
                raise SystemExit(f"{label}: no iterations for rank {rank}")
            if not clean:
                print(
                    f"{label}: rank {rank} has no hygiene-clean epoch; "
                    f"using epoch {sel.epoch} (dirty)",
                    file=sys.stderr,
                )
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
        raise SystemExit(
            f"no iterations for rank {args.rank} "
            f"(ranks present: {sorted({i.rank for i in iters})})"
        )
    sel = (
        mine[-1]
        if args.iter == "last"
        else next((i for i in mine if i.epoch == int(args.iter)), None)
    )
    if sel is None:
        raise SystemExit(f"iteration {args.iter} not found " f"(have {[i.epoch for i in mine]})")

    if not args.no_table:
        cohort_table(sel)
    gw = None
    if args.gw_marks:
        wire = args.wire or infer_wire(args.gw_marks)
        if wire is None:
            raise SystemExit(
                "--gw-marks: pass --wire (variant not inferable "
                "from the path; note flat a2av has no gateways)"
            )
        print(f"--wire {'given' if args.wire else 'inferred'}: {wire}")
        gw, gw_t0, gw_anchors = compute_gw_marks(
            args.gw_marks,
            sel,
            iters,
            wire,
            debug=args.gw_marks_debug,
            stamped_only=args.gw_stamped_only,
        )
        if args.mega:
            if gw_t0 is None:
                raise SystemExit("--mega: gateway markers unavailable")
            mega_png(iters, sel, [(x + gw_t0, lab, fl) for x, lab, fl in gw], gw_anchors, args.mega)
    elif args.mega:
        raise SystemExit("--mega requires --gw-marks")
    if args.curves:
        regime_png(sel, args.curves, gw_marks=gw)
    if args.gantt:
        gantt_json([sel], args.gantt, align_db=args.align, context=iters)


if __name__ == "__main__":
    main()
