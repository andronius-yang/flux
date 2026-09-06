#!/usr/bin/env python3
"""figs/case_study: review page (figure at zoom + print size, rank ledger,
byte ledger, isolated-mode latency table, open rulings). CASE-STUDY ONLY.

  python figs/case_study/build_review.py <capsule_run_id> --fig figs/case_study/case_study \
      --ledger $PSCRATCH/workspace/andrewy/figs_data/case_study/byte_ledger_<run>.csv --out figs/case_study/case_study_review.html
"""
import argparse, collections, csv, html, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_case_study as B  # noqa: E402  (ROWS, INCLUDE_GATED_COMET)

ARM = {"l01_allgather_dense_nogate_c8": "COMET, overlapped (no gate, 8 conn.)", "l01_allgather_dense": "COMET (gated)",
       "ours_l01_s2_swap_p2p_t1_r2": "OURS overlapped swap", "ablation_l01_s2_swap_t1_noov_p2p_r2": "OURS sequential swap",
       "ablation_l01_s2_swap_t1_esplit_p2p_r2": "OURS equal-split routing"}


def latency_table(capsule):
    rows = [r for r in csv.DictReader(open(os.path.join("sweeps/results/runs", capsule, "metrics.csv"))) if r["mode"] == "isolated"]
    by = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for r in rows: by[r["cell_id"]][r["metric"]][int(r["iter"])].append(float(r["value_ms"]))
    def blk(cell, its):
        d = by[cell]
        return {m: st.median([max(d[m][i]) for i in its if i in d[m]]) for m in ("total_ms", "l0_ms", "l1_ms") if m in d}
    blocks = [("plain lcb (efficient)", "trace-610042_b64_k8_isolated", range(0, 10)),
              ("schedule, proLaw block (skewed)", "trace-b549f7_b64_k8_isolated", range(27, 31)),
              ("schedule, whole (32 iters)", "trace-b549f7_b64_k8_isolated", range(0, 32))]
    out = ["<table><tr><th>arm</th>" + "".join(f"<th>{b[0]}<br>total / l0 / l1</th>" for b in blocks) + "</tr>"]
    for v, lab in ARM.items():
        cells = [f"<td>{o.get('total_ms', 0):.1f} / {o.get('l0_ms', 0):.1f} / {o.get('l1_ms', 0):.1f}</td>"
                 for _, suf, its in blocks for o in [blk(f"{v}_{suf}", its)]]
        out.append(f"<tr><td>{lab}</td>{''.join(cells)}</tr>")
    return "".join(out) + "</table>"


def byte_table(ledger_csv):
    rows = list(csv.DictReader(open(ledger_csv)))
    out = ["<table><tr><th>row</th><th>rows/rank spread</th><th>NIC GB total</th><th>NIC spread</th><th>NVLink GB total</th><th>local fraction</th></tr>"]
    for t1, t2, cid, itn in B.ROWS:
        if t2 == "COMET (gated)" and not B.INCLUDE_GATED_COMET: continue
        k = int(itn.replace("iter", ""))
        rs = [r for r in rows if r["cell_id"] == cid and int(r["epoch_idx"]) == k]
        if not rs: out.append(f"<tr><td>{t1} / {t2}</td><td colspan=5>no sidecar rows (dense path records tiles only)</td></tr>"); continue
        tot = [int(r["rows_total"]) for r in rs]; inter = [int(r["bytes_inter"]) / 1e6 for r in rs]; intra = [int(r["bytes_intra"]) / 1e6 for r in rs]; self_ = [int(r["bytes_self"]) / 1e6 for r in rs]
        loc = (sum(intra) + sum(self_)) / max(1e-9, sum(inter) + sum(intra) + sum(self_))
        comet = cid.startswith("l01_allgather_dense")
        lab = f"{t1} / {t2}" + (" (rows = computed rows per source; wire = fixed allgather shards)" if comet else "")
        out.append(f"<tr><td>{lab}</td><td>{max(tot) / max(1, min(tot)):.2f}×</td><td>{sum(inter) / 1e3:.1f}</td><td>{max(inter) / max(1e-9, min(inter)):.2f}×</td><td>{sum(intra) / 1e3:.1f}</td><td>{loc:.2f}</td></tr>")
    return "".join(out) + "</table>"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("capsule"); ap.add_argument("--fig", required=True); ap.add_argument("--ledger", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rev", default="0.2")
    a = ap.parse_args()
    svg = open(a.fig + ".svg").read()
    led = list(csv.DictReader(open(a.fig + "_ranks.csv")))
    tbl = "".join(f"<tr><td>{html.escape(r['row'])}</td><td>r{r['rank']} (node {r['node']})</td><td>{html.escape(r['why'])}</td><td>{r['span_ms']}</td><td>{r['gemm_ms']}</td><td>{r['reduce_ms']}</td><td>{r['nic_ms']}</td><td>{r['nvlink_ms']}</td><td>{r['wait_ms']}</td></tr>" for r in led)
    h_pt = svg.split('height="')[1].split('pt')[0]
    page = f'''<title>Case Study Lanes</title>
<style>
:root{{--paper:#f7f6f2;--ink:#17191c;--ink2:#4f545c;--mute:#7d8289;--rule:#dad7cf;--card:#ffffff;--acc:#2a78d6}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--paper:#1b1d21;--ink:#ecebe6;--ink2:#c2c4c9;--mute:#8e939b;--rule:#3a3d44;--card:#23262b;--acc:#6ea6e8}}}}
:root[data-theme="dark"]{{--paper:#1b1d21;--ink:#ecebe6;--ink2:#c2c4c9;--mute:#8e939b;--rule:#3a3d44;--card:#23262b;--acc:#6ea6e8}}
body{{background:var(--paper);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.55;margin:0}}
main{{max-width:1180px;margin:0 auto;padding:32px 28px 80px}}
h1{{font-family:"Newsreader",Georgia,serif;font-weight:500;font-size:36px;margin:0 0 6px;text-wrap:balance}} h2{{font-family:"Newsreader",Georgia,serif;font-weight:500;font-size:25px;margin:34px 0 8px}}
.eyebrow{{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--mute);font-weight:600}}
p,li{{max-width:72ch}} figure{{margin:18px 0;background:#fff;border:1px solid var(--rule);padding:18px;border-radius:4px;overflow-x:auto}}
figcaption{{font-size:13.5px;color:var(--ink2);padding-top:8px}}
.zoom svg{{width:1120px;height:auto}} .actual svg{{width:504pt;height:auto}}
table{{border-collapse:collapse;font-size:12.5px;font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}} th,td{{text-align:left;padding:4px 12px 4px 0;border-bottom:1px solid var(--rule);white-space:nowrap;vertical-align:top}} th{{color:var(--mute);font-weight:600}}
.wrap{{overflow-x:auto}} code{{font-family:"IBM Plex Mono",monospace;font-size:13px}}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono&display=swap">
<main>
<div class="eyebrow">Case-study figure · draft {a.rev} · cross-column · 2026-09-05 · capsule {a.capsule}</div>
<h1>Case Study Lanes</h1>
<p>One row per scenario. Efficient = the plain livecodebench cell (fresh oracle placement); Skewed = the professional_law block of the S-C schedule (leave-one-out basis, placement carried over). Per case: the overlapped COMET baseline (dense allgather with its pre-GEMM gate dropped and 8 CUDA connections, so its GEMM runs alongside the inter-node fetch), then OURS with the overlapped one-round swap and its two twins (sequential swap, equal-split routing). Two ranks per row: the longest and the shortest expert GEMM of that iteration. Lanes: NIC RDMA (solid), NVLink (dashed), GPU main stream and GPU side streams (dotted). Colour = task; one shared ms scale; the black tick is the iteration's device end. Every row comes from one capsule and one binary. nsys mode = timelines and byte ledgers, never latency — the latency table below is from the same capsule's isolated cells.</p>
<h2>At 2.2×</h2><figure class="zoom">{svg}<figcaption>case_study.svg — 504 × {h_pt} pt. The draw.io twin (case_study.drawio) carries the same primitives on layers background / bars / glyphs / axes / labels.</figcaption></figure>
<h2>At print size</h2><figure class="actual">{svg}</figure>
<h2>Latency on the same binary (isolated mode, max rank per iteration, median over the block; ms)</h2>
<div class="wrap">{latency_table(a.capsule)}</div>
<h2>Rank ledger (ms of busy time inside the drawn iteration)</h2>
<div class="wrap"><table><tr><th>row</th><th>rank</th><th>why</th><th>span</th><th>GEMM l0+l1</th><th>top-k reduce</th><th>NIC busy</th><th>NVLink busy</th><th>wait</th></tr>{tbl}</table></div>
<h2>Byte ledger (receive side, all 16 ranks, from the tile-trace sidecar)</h2>
<div class="wrap">{byte_table(a.ledger)}</div>
<h2>Open rulings</h2>
<ul>
<li>Row set: overlapped COMET + three OURS rows per case (gated COMET available via <code>INCLUDE_GATED_COMET</code>).</li>
<li>Shared ms scale (current) vs per-row scale (<code>SHARED_SCALE</code>); GPU main/side split (<code>GPU_SIDE_LANE</code>); host lane off (<code>HOST_LANE</code>).</li>
<li>Six colours: Token Comm., Expert Comm., Expert Comp., Top-k Reduce, Plan/Meta, Wait. COMET's fused gather/top-k reduce-scatter kernel is drawn as Top-k Reduce on the GPU side lane.</li>
</ul>
</main>'''
    open(a.out, "w").write(page); print("wrote", a.out, len(page))


if __name__ == "__main__":
    main()
