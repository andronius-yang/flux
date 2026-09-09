#!/usr/bin/env python3
"""figs/case_study: cross-column NSDI timeline figure — one row per scenario
(efficient / skewed x overlapped swap / sequential swap / equal-split
routing), two story-extreme ranks per row, lanes NIC RDMA / NVLink / GPU
(main stream) / GPU (side streams), blocks coloured by TASK. SVG (points) +
native draw.io twin (layers background / bars / glyphs / axes / labels) +
rank ledger CSV. Stdlib only; primitives from the motivation v2 builder.

  python figs/case_study/build_case_study.py \\
      $PSCRATCH/workspace/andrewy/figs_data/case_study/timeline_20260905-121506.json \\
      --out figs/case_study/case_study

Draft 0.1 (2026-09-05) — rulings so far: separate rows for twins, 2 ranks per
row, colour = task, draw.io output like the motivation figure. Open: row
order/titles, shared vs per-row ms scale (knob SHARED_SCALE), the extra
GPU side-stream lane (knob GPU_SIDE_LANE), host lane (knob HOST_LANE).
"""
import argparse, csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "motivation", "v2")); sys.path.insert(0, os.path.join(HERE, "..", "motivation", "v1"))
import build_v2_lanes as V2  # noqa: E402  (Doc with dashed lines, SVG + draw.io renderers)

# ---- geometry (points) --------------------------------------------------------
TEXT_W = 504.0
L_GUT = 58.0                       # row titles + lane labels
R_PAD = 4.0
LANE_H, LANE_GAP, RANK_GAP, ROW_GAP = 4.2, 0.8, 2.5, 7.0
TOP, AXIS_H, LEGEND_H = 3.0, 11.0, 13.0
SHARED_SCALE = True                # one ms scale for every row (rows stay comparable)
GPU_SIDE_LANE = True               # split GPU into main stream / side streams (concurrent kernels would hide each other)
HOST_LANE = False                  # thin host-chain lane (plan.* / swap.* ranges); off per user preference
FONT = V2.FONT; INK, INK2, LINE = V2.INK, V2.INK2, V2.LINE
# colour = TASK (muted palette ruling: no pink/purple)
COL = {"token": "#2a78d6",        # Token Comm.  (NIC puts, NVLink token copies, dispatch staging copies)
       "expert_comm": "#1baf7a",  # Expert Comm. (expert-slot swap copies over NVLink)
       "comp": "#eda100",         # Expert Comp. (grouped GEMM l0 / l1)
       "reduce": "#c8553d",       # Top-k Reduce (pre-topk reduce, pack, bucket reduce)
       "plan": "#2f8f9d",         # Plan / Meta  (route, meta, sort, plan collectives)
       "wait": "#c9c8c0",         # Wait (barrier / signal spin on the stream)
       "host": "#8a8f98"}         # Host chain (only with HOST_LANE)
TASK_COL = {"nic.put": "token", "nvlink.token": "token", "copy.d2d": "token", "nvlink.swap": "expert_comm",
            "gemm.l0": "comp", "gemm.l1": "comp", "combine.prereduce": "reduce", "combine.pack": "reduce",
            "combine.reduce": "reduce", "combine.gather_rs": "reduce", "plan.compute": "plan", "plan.comm": "plan", "misc": "plan", "other": "plan",
            "barrier": "wait"}
SIDE_TASKS = {"combine.prereduce", "combine.pack", "combine.reduce", "combine.gather_rs"}   # side-stream kernels (concurrent with the l1 GEMM)
DASH = {"nic": "", "nvlink": "2,1.5", "gpu": "0.6,1.2", "gpu2": "0.6,1.2", "host": "1,1"}
LANE_LABEL = {"nic": "NIC RDMA", "nvlink": "NVLink", "gpu": "GPU", "gpu2": "GPU (side)", "host": "Host"}

# rows: (title line 1, title line 2, cell_id, iteration label)
SCHED = "trace-b549f7_b64_k8_nsys"; PLAIN = "trace-610042_b64_k8_nsys"
INCLUDE_GATED_COMET = False
INCLUDE_DUAL_V1 = False            # capture-3 "dual" (w2 gated on l0 completion: lands in the l0->l1 gap) next to dual2        # the stock (gated) COMET row per case, in addition to the overlapped baseline
ROWS = [
    ("Efficient", "COMET (gated)",     f"l01_allgather_dense_{PLAIN}", "iter4"),
    ("Efficient", "COMET, overlapped", f"l01_allgather_dense_nogate_c8_{PLAIN}", "iter4"),
    ("Efficient", "overlapped swap",   f"ours_l01_s2_swap_p2p_t1_r2_{PLAIN}", "iter4"),
    ("Efficient", "sequential swap",   f"ablation_l01_s2_swap_t1_noov_p2p_r2_{PLAIN}", "iter4"),
    ("Efficient", "equal-split routing", f"ablation_l01_s2_swap_t1_esplit_p2p_r2_{PLAIN}", "iter4"),
    ("Skewed", "COMET (gated)",        f"l01_allgather_dense_{SCHED}", "iter33"),
    ("Skewed", "COMET, overlapped",    f"l01_allgather_dense_nogate_c8_{SCHED}", "iter33"),
    ("Skewed", "overlapped swap",      f"ours_l01_s2_swap_p2p_t1_r2_{SCHED}", "iter33"),
    ("Skewed", "sequential swap",      f"ablation_l01_s2_swap_t1_noov_p2p_r2_{SCHED}", "iter33"),
    ("Skewed", "equal-split routing",  f"ablation_l01_s2_swap_t1_esplit_p2p_r2_{SCHED}", "iter33"),
]


def lanes_of(it):
    """iteration dict -> {lane: [(t0, t1, color_key, title)]}, tiny events dropped (< 0.05 ms)."""
    L = {k: [] for k in ("nic", "nvlink", "gpu", "gpu2", "host")}
    for x in it["events"]:
        if x["t1"] - x["t0"] < 0.05 and x["task"] not in ("nvlink.swap",): continue
        ck = TASK_COL.get(x["task"])
        if ck is None: continue                       # host-side copies etc.
        lane = x["lane"]
        if lane == "wait": lane = "gpu"
        if lane == "gpu" and GPU_SIDE_LANE and x["task"] in SIDE_TASKS: lane = "gpu2"
        if lane not in L: continue
        by = f" {x['bytes'] / 1e6:.0f} MB" if x.get("bytes") else ""
        ph = {"late": " [w1, dispatch side]", "l1": " [w2, combine side]", "early": " [host gap]"}.get(x.get("phase"), "")
        L[lane].append((x["t0"], x["t1"], ck, f"{x['task']}{ph}{by} {x['t0']:.2f}–{x['t1']:.2f} ms"))
    for h in it.get("host_ranges", []):
        L["host"].append((h["t0"], h["t1"], "host", h["name"]))
    return L


def story(it):
    s = it["summary"]
    return dict(gemm=s.get("gemm.l0", 0) + s.get("gemm.l1", 0), nic=s.get("_lane_nic", 0), nvl=s.get("_lane_nvlink", 0),
                wait=s.get("barrier", 0), reduce=sum(v for k, v in s.items() if k.startswith("combine.")), end=it["dev_end_ms"])


def pick2(cell, itname):
    ranks = sorted(cell["ranks"], key=int)
    S = {r: story(cell["ranks"][r]["iters"][itname]) for r in ranks}
    hi = max(ranks, key=lambda r: S[r]["gemm"]); lo = min(ranks, key=lambda r: S[r]["gemm"])
    return [(hi, "longest expert GEMM"), (lo, "shortest expert GEMM")], S


def build(data, out):
    lanes = ["nic", "nvlink", "gpu"] + (["gpu2"] if GPU_SIDE_LANE else []) + (["host"] if HOST_LANE else [])
    rank_h = len(lanes) * LANE_H + (len(lanes) - 1) * LANE_GAP
    row_h = 2 * rank_h + RANK_GAP
    tx0 = L_GUT; tw = TEXT_W - L_GUT - R_PAD
    rows = []
    for t1, t2, cid, itn in ROWS:
        if t2 == "COMET (gated)" and not INCLUDE_GATED_COMET: continue
        if t2 == "3D swap (l0-done gate)" and not INCLUDE_DUAL_V1: continue
        if cid not in data["cells"]: print("missing cell, row skipped:", cid, file=sys.stderr); continue
        c = data["cells"][cid]; chosen, S = pick2(c, itn)
        rows.append((t1, t2, c, itn, chosen, S))
    tmax_all = max(S[r]["end"] for *_, chosen, S in rows for r, _ in chosen) * 1.02
    D = V2.Doc(); y = TOP; ledger = []
    for t1, t2, c, itn, chosen, S in rows:
        tmax = tmax_all if SHARED_SCALE else max(S[r]["end"] for r, _ in chosen) * 1.02
        sc = tw / tmax
        # row title (left gutter, two lines) + lane labels for the first rank
        D.text(3, y + 7, t1, 6.5, "labels", "start", INK, bold=True)
        D.text(3, y + 14, t2, 5.2, "labels", "start", INK2)
        yy = y
        for ri, (r, why) in enumerate(chosen):
            it = c["ranks"][r]["iters"][itn]; L = lanes_of(it)
            ly = {ln: yy + i * (LANE_H + LANE_GAP) for i, ln in enumerate(lanes)}
            for ln in lanes:
                cy = ly[ln] + LANE_H / 2
                D.dline(tx0, cy, tx0 + tw, cy, LINE, "background", 0.5, DASH[ln])
                D.text(tx0 - 2, ly[ln] + LANE_H - 0.6, LANE_LABEL[ln], 3.6, "labels", "end", INK2)
                for (a, b, ck, title) in sorted(L[ln]):
                    D.rect(tx0 + a * sc, ly[ln], (b - a) * sc, LANE_H, COL[ck], "bars", f"r{r} {title}")
            # rank span tick (end of device work)
            ex = tx0 + S[r]["end"] * sc
            D.line(ex, yy - 0.4, ex, yy + rank_h + 0.4, INK, "glyphs", 0.5)
            ledger.append(dict(row=f"{t1} / {t2}", cell=c["variant"], family=c["family_params"], iteration=itn, rank=r, node=int(r) // c["rpn"],
                               why=why, span_ms=round(S[r]["end"], 2), gemm_ms=round(S[r]["gemm"], 2), nic_ms=round(S[r]["nic"], 2),
                               nvlink_ms=round(S[r]["nvl"], 2), reduce_ms=round(S[r]["reduce"], 2), wait_ms=round(S[r]["wait"], 2)))
            yy += rank_h + RANK_GAP
        y += row_h + ROW_GAP
    # shared axis
    ay = y - ROW_GAP + 2
    D.line(tx0, ay, tx0 + tw, ay, INK2, "axes", 0.5)
    step = V2.nice_step(tmax_all); t = 0; sc = tw / tmax_all
    while t <= tmax_all:
        D.line(tx0 + t * sc, ay, tx0 + t * sc, ay + 1.8, INK2, "axes", 0.5)
        if tx0 + t * sc < tx0 + tw - 9: D.text(tx0 + t * sc, ay + 7, f"{t}", 5, "labels", "middle", INK2)
        t += step
    D.text(tx0 + tw, ay + 7, "ms", 5, "labels", "end", INK2)
    # legend: task colours, span tick, resource patterns
    ly_ = ay + AXIS_H + 1; lx = L_GUT
    for key, lab in (("token", "Token Comm."), ("expert_comm", "Expert Comm."), ("comp", "Expert Comp."),
                     ("reduce", "Top-k Reduce"), ("plan", "Plan / Meta"), ("wait", "Wait")):
        D.rect(lx, ly_ + 1, 8, 4.5, COL[key], "bars"); D.text(lx + 10, ly_ + 5, lab, 5.2, "labels", color=INK2); lx += 10 + 2.75 * len(lab) + 7
    D.line(lx, ly_, lx, ly_ + 6.5, INK, "glyphs", 0.5); D.text(lx + 3, ly_ + 5, "iteration end", 5.2, "labels", color=INK2); lx += 3 + 2.75 * 13 + 9
    for ln in ("nic", "nvlink", "gpu"):
        D.dline(lx, ly_ + 3.2, lx + 12, ly_ + 3.2, LINE, "background", 0.6, DASH[ln]); D.text(lx + 14, ly_ + 5, LANE_LABEL[ln], 5.2, "labels", color=INK2)
        lx += 14 + 2.75 * len(LANE_LABEL[ln]) + 7
    D.h = ly_ + LEGEND_H
    open(out + ".svg", "w").write(D.svg()); open(out + ".drawio", "w").write(D.drawio())
    with open(out + "_ranks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ledger[0].keys())); w.writeheader(); w.writerows(ledger)
    return ledger, D.h, tmax_all


# --simple: only the OURS overlapped-swap rows (efficient + skewed) = 2 rows x 2 ranks
SIMPLE_ROWS = [r for r in ROWS if r[1] == "overlapped swap"]
# capture 3 (2026-09-09, 3D scheduling, capsule 20260909-074415, binary a4f418de): the
# 8-slot RESET-EVERY exchange, issue point = host gap (early) / sequential / 3D dual
# (w1 under l0, w2 under l1; dual3 = GEMM-start-mark gated, capsule 20260909-115108). Select with --rows cs3; iteration labels via --eff-iter/--skew-iter.
_RST = "ablation_l01_s2_swapall_rst_3d"
ROWS_CS3 = [
    ("Efficient", "COMET (gated)",       f"l01_allgather_dense_{PLAIN}", "EFF"),
    ("Efficient", "COMET, overlapped",   f"l01_allgather_dense_nogate_c8_{PLAIN}", "EFF"),
    ("Efficient", "swap in host gap",    f"{_RST}_early_str4_p2p_r2_{PLAIN}", "EFF"),
    ("Efficient", "sequential swap",     f"{_RST}_noov_str4_p2p_r2_{PLAIN}", "EFF"),
    ("Efficient", "3D swap (l0-done gate)", f"{_RST}_dual_str4_p2p_r2_{PLAIN}", "EFF"),
    ("Efficient", "3D-scheduled swap",   f"{_RST}_dual3_str4_p2p_r2_{PLAIN}", "EFF"),
    ("Efficient", "swap under l0 GEMM",  f"{_RST}_late3_str4_p2p_r2_{PLAIN}", "EFF"),
    ("Skewed", "COMET (gated)",          f"l01_allgather_dense_{SCHED}", "SKEW"),
    ("Skewed", "COMET, overlapped",      f"l01_allgather_dense_nogate_c8_{SCHED}", "SKEW"),
    ("Skewed", "swap in host gap",       f"{_RST}_early_str4_p2p_r2_{SCHED}", "SKEW"),
    ("Skewed", "sequential swap",        f"{_RST}_noov_str4_p2p_r2_{SCHED}", "SKEW"),
    ("Skewed", "3D swap (l0-done gate)", f"{_RST}_dual_str4_p2p_r2_{SCHED}", "SKEW"),
    ("Skewed", "3D-scheduled swap",      f"{_RST}_dual3_str4_p2p_r2_{SCHED}", "SKEW"),
    ("Skewed", "swap under l0 GEMM",     f"{_RST}_late3_str4_p2p_r2_{SCHED}", "SKEW"),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("json"); ap.add_argument("--out", required=True)
    ap.add_argument("--rows", choices=("cs2", "cs3"), default="cs2", help="row table: cs2 = 9/5 capture, cs3 = 9/9 3D-scheduling capture")
    ap.add_argument("--eff-iter", default="iter4"); ap.add_argument("--skew-iter", default="iter33")
    ap.add_argument("--simple", action="store_true", help="OURS overlapped swap only: Efficient + Skewed, 2 ranks each")
    a = ap.parse_args()
    if a.rows == "cs3":
        ROWS[:] = [(t1, t2, cid, {"EFF": a.eff_iter, "SKEW": a.skew_iter}[itn]) for t1, t2, cid, itn in ROWS_CS3]
    if a.simple: ROWS[:] = [r for r in ROWS if r[1] in ("overlapped swap", "3D-scheduled swap")]
    data = json.load(open(a.json))
    ledger, h, tmax = build(data, a.out)
    print(f"wrote {a.out}.svg/.drawio/_ranks.csv  ({TEXT_W:.0f} x {h:.1f} pt, shared scale 0..{tmax:.1f} ms)")
    for r in ledger:
        print(f"  {r['row']:<32} r{r['rank']:<3} {r['why']:<22} span {r['span_ms']:6.2f} GEMM {r['gemm_ms']:6.2f} NIC {r['nic_ms']:6.2f} NVL {r['nvlink_ms']:5.2f} reduce {r['reduce_ms']:5.2f} wait {r['wait_ms']:5.2f}")
