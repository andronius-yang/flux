"""Loader for the COMET+EPLB fused sender-local router (_comet_eplb_ext.cu),
a standalone torch extension (pv3_ext precedent: no libflux_cuda rebuild).
Build once on a login node (gcc 12 + CUDA 12.4 pinned — see
logs/pv3/comet_eplb_env_fab120.sh); compute ranks load the cached .so from
the shared build dir on $PSCRATCH.

Usage:
    ext = load_ext()
    dst_phys = ext.route_local(topk_i32 [S,K] cuda, l2p_i32 [G,Cmax] cuda,
                               lcnts_i32 [G] cuda, src_rank, mode, interleave)
    # mode 0 = local_spread, 1 = local_static; bit-exact with
    # EplbIterPlanner.derive_fused (guarded at driver setup)
"""
import os

_ext = None
MODE = {"local_spread": 0, "local_static": 1}


def load_ext(verbose=False):
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_comet_eplb_ext.cu")
    build_dir = os.environ.get(
        "FLUX_COMET_EPLB_EXT_DIR",
        os.path.expandvars("$PSCRATCH/workspace/andrewy/comet_eplb_ext_build"))
    os.makedirs(build_dir, exist_ok=True)
    _ext = load(name="comet_eplb_ext", sources=[src],
                build_directory=build_dir,
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"],
                verbose=verbose)
    return _ext


def route_local(planner, ext=None):
    """One-launch twin of planner.derive_fused() for the local replica
    modes: [S, K] int32 physical slots on the planner's device."""
    ext = ext or load_ext()
    return ext.route_local(
        planner.topk_all[planner.rank].to(torch.int32).contiguous(),
        planner.l2p.to(torch.int32).contiguous(),
        planner.lcnts.to(torch.int32).contiguous(),
        int(planner.rank), MODE[planner.replica_select],
        bool(planner.cfg.interleave))


import torch  # noqa: E402  (after the loader so a missing toolchain fails late)
