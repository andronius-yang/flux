"""Loader for the PV3 rotation water-fill router kernel (_pv3_ext.cu),
built as a standalone torch extension so the routing A/B needs no
libflux_cuda rebuild (the pll_sl_ext precedent). Build once (login node,
CUDA 12.4 pinned — see perlmutter-cudatoolkit-drift); compute ranks load
the cached .so from the shared build dir on $PSCRATCH.

Usage:
    ext = load_ext()
    ws = torch.empty(ext.workspace_ints(G, R), dtype=torch.int32,
                     device="cuda")          # persistent per planner
    phys_own, stats = ext.route_pv3(topk_own_i32, d_i32, l2p_i32,
                                    lcnts_i32, my_rank, nlp, L,
                                    C_num, C_den, ws)
"""
import os

_ext = None


def load_ext(verbose=False):
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_pv3_ext.cu")
    build_dir = os.environ.get(
        "FLUX_PV3_EXT_DIR",
        os.path.expandvars("$PSCRATCH/workspace/andrewy/pv3_ext_build"))
    os.makedirs(build_dir, exist_ok=True)
    _ext = load(name="pv3_ext", sources=[src],
                build_directory=build_dir,
                extra_cuda_cflags=[
                    "-O3", "-gencode=arch=compute_80,code=sm_80"],
                verbose=verbose)
    return _ext


def c_rational(eps):
    """The arms' --eps float -> exact rational (C_num, C_den). eps must be
    a dyadic rational with denominator <= 2^16 (1/16, 1/8, 0.25, ...);
    inf is rejected (pv3 has no 'pure locality' mode — the paper's
    constraint always binds)."""
    import math
    assert math.isfinite(eps) and eps >= 0, f"pv3 needs finite eps: {eps}"
    den = 1 << 16
    num = int(round(eps * den))
    assert abs(num / den - eps) < 1e-12, f"eps {eps} is not dyadic"
    g = math.gcd(num, den) if num else den
    return num // g, den // g
