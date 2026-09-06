#!/usr/bin/env python3
"""figs/case_study: receive-side byte ledger per rank per iteration from the
a2av tile-trace sidecars (CASE-STUDY ONLY, 2026-09-05).

The sidecar (sweeps/plot_a2av_trace.read_sidecar, v2) records, per l0
dispatch epoch, the ROW count each source rank delivered to this rank.
Rows x H x 2 bytes = bf16 payload. Sources on another node arrived over the
NIC, same-node sources over NVLink, self = no wire. Epochs are listed in
order; epoch k of a rank == iteration k of the run (warmups included), which
is verified against the number of NVTX-labelled iterations by the caller.

usage: python figs/case_study/byte_ledger.py <capsule_run_id> --csv <out.csv>
"""
import argparse, csv, glob, os, re, sys
sys.path.insert(0, "sweeps")
from plot_a2av_trace import read_sidecar, is_inter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capsule"); ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    cap_dir = os.path.join("sweeps/results/runs", a.capsule)
    out = []
    for row in csv.DictReader(open(os.path.join(cap_dir, "cells.csv"))):
        if row["mode"] != "nsys" or not row["nsys_path"]: continue
        H = int(row["H"]); cell_dir = os.path.dirname(row["nsys_path"])
        for f in sorted(glob.glob(os.path.join(cell_dir, "records", "a2av_tile_trace_r*.bin"))):
            rank = int(re.search(r"_r(\d+)\.bin$", f).group(1))
            its = read_sidecar(f)
            for k, it in enumerate(its):
                if not it.rows: continue
                self_rows = it.rows[rank] if rank < it.nb else 0
                inter = sum(r for s, r in enumerate(it.rows) if s < it.world and is_inter(it, s))
                intra = sum(r for s, r in enumerate(it.rows) if s < it.world and not is_inter(it, s) and s != rank)
                multi = sum(r for s, r in enumerate(it.rows) if s >= it.world)   # multi/relay bucket(s)
                n_inter_src = sum(1 for s, r in enumerate(it.rows) if s < it.world and is_inter(it, s) and r > 0)
                out.append(dict(cell_id=row["cell_id"], variant=row["variant"], family=row["family_params"],
                                rank=rank, epoch_idx=k, epoch=it.epoch, rows_total=sum(it.rows),
                                rows_self=self_rows, rows_intra=intra, rows_inter=inter, rows_multi=multi,
                                n_inter_sources=n_inter_src,
                                bytes_inter=inter * H * 2, bytes_intra=intra * H * 2, bytes_self=self_rows * H * 2,
                                tiles=len(it.recs)))
        print(row["cell_id"], "sidecars:", len(glob.glob(os.path.join(cell_dir, "records", "a2av_tile_trace_r*.bin"))),
              "epochs/rank:", len(its), file=sys.stderr)
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print("wrote", a.csv, len(out), "rows", file=sys.stderr)


if __name__ == "__main__":
    main()
