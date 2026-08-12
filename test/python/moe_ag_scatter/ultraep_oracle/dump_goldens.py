"""Freeze golden outputs of UltraEP's REAL solver/reroute kernels for the
named planner cases, so test_ultraep_planner.py's golden tier can bit-check
the PyTorch port on machines without the ultra_ep SM80 oracle build.

Needs: one CUDA GPU + the ultra_ep SM80 oracle build (branch sm80-oracle).
Run:   python test/python/moe_ag_scatter/ultraep_oracle/dump_goldens.py

Storage layout (goldens/<case>.pt, CPU tensors):
  meta:   case params + ultra_ep git version + torch/cuda versions
  p2l, l2p, lcnts, quota, quota_prefix        (verified identical across all
                                               rank_quota_source_rank values)
  rank_quota_prefix                           [R, G, R] stacked per source
  reroute: per spot-checked src (0, 1), per interleave in (True, False):
    entry_token/entry_phys int32 sparse form of the expanded routing map
    (probs are recomputable: seed 1234+src, masked by the routing map)

Determinism gate: every kernel call runs twice and must be bitwise equal
before anything is written.
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import ultra_ep  # noqa: E402  (fails loudly without the oracle build)
import ultra_ep._C as _C  # noqa: E402

from test_ultraep_planner import NAMED_CASES, make_cfg, make_topk  # noqa: E402
from flux.testing.ultraep_semantics import loads_from_topk  # noqa: E402

SOLVER_OUTPUT_NAMES = ["p2l", "l2p", "lcnts", "quota", "quota_prefix",
                       "rank_quota_prefix"]


def solve(cfg, loads_gpu, tpe_gpu, src):
    return _C.solve_placement_for_test(
        loads_gpu, tpe_gpu,
        num_ranks=cfg.R,
        num_local_master_experts=cfg.epn,
        num_local_redundant_experts=cfg.R_red,
        num_nvl_ranks=cfg.D,
        legacy_placement=False,
        balance_threshold=cfg.balance_threshold,
        min_tokens_per_replica=cfg.min_tokens_per_replica,
        allow_zero_master_quota=cfg.allow_zero_master_quota,  # explicit!
        locality_aware=cfg.locality_aware,
        oracle_eps=cfg.oracle_eps,
        kernel_stage=1,
        rank_quota_source_rank=src,
    )


def main():
    torch.cuda.set_device(0)
    out_dir = os.path.join(_HERE, "goldens")
    os.makedirs(out_dir, exist_ok=True)

    for case in NAMED_CASES:
        cfg = make_cfg(case)
        topk_all = make_topk(case)
        tpe = loads_from_topk(cfg, topk_all)
        loads_gpu = tpe.long().sum(0).to(torch.int32).cuda()
        tpe_gpu = tpe.cuda()

        shared = None
        rqp_all = []
        for src in range(cfg.R):
            a = [t.cpu() for t in solve(cfg, loads_gpu, tpe_gpu, src)]
            b = [t.cpu() for t in solve(cfg, loads_gpu, tpe_gpu, src)]
            for name, x, y in zip(SOLVER_OUTPUT_NAMES, a, b):
                assert torch.equal(x, y), (
                    f"{case.name} src={src}: kernel nondeterminism in {name}"
                )
            if shared is None:
                shared = a[:5]
            else:
                for name, x, y in zip(SOLVER_OUTPUT_NAMES, shared, a[:5]):
                    assert torch.equal(x, y), (
                        f"{case.name}: {name} differs across source ranks"
                    )
            rqp_all.append(a[5])

        reroute = {}
        for src in (0, 1):
            if src >= cfg.R:
                continue
            routing = torch.zeros(cfg.S, cfg.G, dtype=torch.bool)
            routing[torch.arange(cfg.S).unsqueeze(1),
                    topk_all[src].long()] = True
            gen = torch.Generator().manual_seed(1234 + src)
            probs = torch.rand(cfg.S, cfg.G, generator=gen) * routing
            for interleave in (True, False):
                _, got_map = _C.dense_reroute_for_test(
                    routing.cuda(), probs.cuda(),
                    shared[1].cuda(), shared[2].cuda(), rqp_all[src].cuda(),
                    cfg.P, quota_reroute=True,
                    interleave_by_rank_quota=interleave,
                )
                tok, phys = got_map.cpu().nonzero(as_tuple=True)
                reroute[f"src{src}_il{int(interleave)}"] = {
                    "entry_token": tok.to(torch.int32),
                    "entry_phys": phys.to(torch.int32),
                }

        blob = {
            "meta": {
                "case": case.__dict__,
                "ultra_ep_version": getattr(ultra_ep, "__version__", "?"),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "rank_quota_prefix": torch.stack(rqp_all),
            "reroute": reroute,
        }
        blob.update(dict(zip(SOLVER_OUTPUT_NAMES[:5], shared)))
        path = os.path.join(out_dir, f"{case.name}.pt")
        torch.save(blob, path)
        print(f"wrote {path} ({os.path.getsize(path) // 1024} KiB)")

    print("goldens frozen.")


if __name__ == "__main__":
    main()
