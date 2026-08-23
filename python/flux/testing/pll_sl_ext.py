"""Loader for the standalone patched sender-local router kernel
(_pll_sl_ext.cu — templated register-mask tier-3 + remote-cap flavor).
Build once on a login node (shared build dir on $PSCRATCH); compute
ranks load the cached .so. Dispatch: FLUX_PLL_SL_EXT=1 makes
EpicIterPlanner call this kernel instead of flux.placelambda_route_sl
(the old-binary kernel) — the kernel A/B without a flux rebuild."""
import os

_ext = None


def load_ext(verbose=False):
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_pll_sl_ext.cu")
    build_dir = os.environ.get(
        "FLUX_PLL_EXT_DIR",
        os.path.expandvars("$PSCRATCH/workspace/andrewy/pll_sl_ext_build"))
    os.makedirs(build_dir, exist_ok=True)
    _ext = load(name="pll_sl_ext", sources=[src],
                build_directory=build_dir,
                extra_cuda_cflags=[
                    "-O3", "-gencode=arch=compute_80,code=sm_80"],
                verbose=verbose)
    return _ext
