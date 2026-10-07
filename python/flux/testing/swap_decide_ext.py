"""Loader for the device intra-node swap decision (_swap_decide_ext.cu): the canonical
serving-path kernel (band-triggered orbit, per_pair = 1, in-place p2l / l2p rewrite, pull
lists in a result block) as a standalone torch extension, so the research runner can move
the swap decision onto the GPU without a libflux rebuild (the pv3_ext precedent). Build
once on the login node (CUDA 12.4 pinned); compute ranks load the cached .so from the
shared build dir on $PSCRATCH.

    ext = load_ext()
    blk = torch.empty(ext.swap_decide_out_ints(R, G, nlp, cap), dtype=torch.int64, device="cuda")
    ext.swap_decide(loads_i32[R, G], ntok_i64[R], S, K, p2l_i64, l2p_i32[G, R], lcnts_i32,
                    L, nlp, C, cap, max_rounds, blk)

C < 0 (e.g. -1) puts every node out of band, so the band never gates: the decision is then
the plain tau = 1 orbit (pair gap > 1 row, gain >= 1 row, cap changed slots per rank).
"""
import os

_ext = None


def load_ext(verbose=False):
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_swap_decide_ext.cu")
    build_dir = os.environ.get(
        "FLUX_SWAP_DECIDE_EXT_DIR",
        os.path.expandvars("$PSCRATCH/workspace/andrewy/swap_decide_ext_build"))
    os.makedirs(build_dir, exist_ok=True)
    _ext = load(name="swap_decide_ext", sources=[src], build_directory=build_dir,
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"],
                verbose=verbose)
    return _ext


def parse_block(blk, R, nlp, cap):
    """Host view of the result block (int64 CPU tensor) -> (rounds, err, all_moves, p2l_new),
    all_moves[r] = [(dst_slot, src_rank, src_slot, expert)] sorted by dst slot: the
    ours_swap.net_moves format."""
    b = blk.tolist() if hasattr(blk, "tolist") else list(blk)
    rounds, _total, err = b[0], b[1], b[2]
    off_mv = 3 + R
    moves = []
    for r in range(R):
        n = b[3 + r]
        base = off_mv + r * cap * 4
        moves.append([tuple(b[base + 4 * j: base + 4 * j + 4]) for j in range(n)])
    off_p2l = off_mv + R * cap * 4
    return rounds, err, moves, b[off_p2l: off_p2l + R * nlp]
