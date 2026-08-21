################################################################################
#
# FusedEpDispatch bring-up probes (campaign-2 S4 hard gate).
# probe: put->fence->signal(ADD) and putmem_signal(ADD) ordering on the
# LIVE transport (NVLink at 1n, Slingshot/CXI at 2n+) — any stale read
# hard-fails and FORBIDS building on the op. Run via launch.sh; multi-node
# via srun --nodes=N ./launch.sh ... (one launcher per node).
#
################################################################################
import argparse
from functools import partial

import torch
import torch.distributed

import flux

DIST_ENV = flux.get_dist_env()
TP_GROUP = DIST_ENV.get_world()
torch.cuda.set_device(DIST_ENV.LOCAL_RANK)
print = partial(print, flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    args = ap.parse_args()
    flux.init_flux_shm(TP_GROUP)
    torch.cuda.synchronize()
    W = DIST_ENV.WORLD_SIZE
    nnodes = max(W // DIST_ENV.LOCAL_WORLD_SIZE, 1)
    # tiny geometry — the probe only uses the probe buffers/signal
    op = flux.FusedEpDispatch(
        TP_GROUP, nnodes, 64, 256, 2, 2, 64, 256, torch.bfloat16, 1, 0)
    op.probe_signal_ordering(args.iters)
    if TP_GROUP.rank() == 0:
        print(f"✅ put->signal ordering clean over {args.iters} epochs x 2 "
              f"forms at W={W} nnodes={nnodes}")
    TP_GROUP.barrier()
