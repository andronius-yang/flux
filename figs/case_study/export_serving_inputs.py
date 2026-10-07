#!/usr/bin/env python3
"""figs/case_study: export the case-study workload for the serving-path capture (CS_v7, 2026-10-06).

The research runner's exact inputs, computed with the research code (CPU only): the oracle placement
(pv2_solve + build_nodeaware_plan, redundant 2 per rank) and, per topic, every rank's routing [W, S, K]
(the routing files of the sweep cells: LiveCodeBench = one file, the S-C schedule = 8 topic files with
dwell 4), plus the harness gate weights (rand + 0.5, seed 777, as test_moe_ours_traffic.py).

usage: python figs/case_study/export_serving_inputs.py --out <dir>   (writes lcb.pt and sc.pt)
"""
import argparse, os, sys, types
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for name, path in (("flux", f"{REPO}/python/flux"), ("flux.testing", f"{REPO}/python/flux/testing")):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m   # no libflux import (CPU only)
import torch
from flux.testing.traffic_matrix import load_routing_file
from flux.testing import placement_v2 as pv2mod, placelambda_fast as plfast
from flux.testing.ultraep_semantics import UltraEPConfig, loads_from_topk
from flux.testing.epic_semantics import build_nodeaware_plan

G, K, W, L, H = 384, 8, 16, 4, 7168
GEN = os.path.expandvars("$PSCRATCH/workspace/andrewy/a2av_test_matrices/generated")
FAMILIES = {
    "lcb": dict(oracle="w16x4_trace-23798c_b64_k8_id001.oracle_routing.txt",
                topics=["23798c"], dwell=1, cell_tag="trace-610042"),
    "sc": dict(oracle="w16x4_trace-8a3f81_b64_k8_id001.oracle_routing.txt",
               topics=["8a3f81", "f79be4", "92cbeb", "6f33ee", "24ac75", "d66733", "bfe4b5", "ee8fcb"], dwell=4,
               cell_tag="trace-b549f7"),
}


def export(tag, spec, out):
    tops = [load_routing_file(f"{GEN}/w16x4_trace-{h}_b64_k8_id001.routing.txt", G, K) for h in spec["topics"]]
    S = tops[0].shape[0] // W
    tops = [t.reshape(W, S, K).int() for t in tops]
    cfg = UltraEPConfig(S=S, K=K, G=G, R=W, H=H, D=L, R_red=2, locality_aware=False, interleave=True)
    tpe = torch.stack([loads_from_topk(cfg, t) for t in tops]).max(0).values
    oc = load_routing_file(f"{GEN}/{spec['oracle']}", G, K).view(W, -1, K).long()
    res = pv2mod.pv2_solve(plfast.demand_hist(oc, L, G).cpu(), L, cfg.nlp)
    pblob = {"version": 2, "G": G, "W": W, "nlp": cfg.nlp, "hosts": pv2mod.hosts_lists(res, G), "planner": dict(res["stats"])}
    plan = build_nodeaware_plan(cfg, tpe, pblob)
    probs = torch.rand((W * S, K), generator=torch.Generator().manual_seed(777)) + 0.5
    blob = dict(G=G, K=K, W=W, L=L, H=H, S=S, nlp=cfg.nlp, dwell=spec["dwell"], cell_tag=spec["cell_tag"],
                topics=spec["topics"], topk=torch.stack(tops), probs=probs.view(W, S, K),
                p2l=plan.p2l.int(), l2p=plan.l2p.int(), lcnts=plan.lcnts.int())
    torch.save(blob, os.path.join(out, f"{tag}.pt"))
    print(f"{tag}: S={S} nlp={cfg.nlp} topics={len(tops)} max lcnt={int(plan.lcnts.max())} -> {out}/{tag}.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for tag, spec in FAMILIES.items():
        export(tag, spec, a.out)
