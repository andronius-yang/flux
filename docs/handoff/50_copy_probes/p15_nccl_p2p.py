"""p15_nccl_p2p.py -- handoff 53 probe P15: NCCL point-to-point bandwidth in the SGLang serving environment (torch's
NCCL with the Slingshot plugin, as SGLang uses it), on the plan-9 wire's pairs: rank r -> the same local rank on the
next node / from the previous node, every rank at once (batch_isend_irecv), eager and captured in a CUDA graph;
plus the all-gather of a whole bucket of hidden states (what stock SGLang's DP gather moves) for reference.
Run with bench/launch.sh (torchrun, one rank per GPU). Sizes 256 KiB .. 8 MiB; us per transfer over 20 in a row.
"""
import os
import statistics
import time

import torch
import torch.distributed as dist


def main():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank, W = dist.get_rank(), dist.get_world_size()
    L = torch.cuda.device_count()
    to, frm = (rank + L) % W, (rank - L) % W
    dev = torch.device("cuda", local)
    log = (lambda *s: print(*s, flush=True)) if rank == 0 else (lambda *s: None)
    log(f"P15 W={W} L={L} nccl {torch.cuda.nccl.version()} env NCCL_NET={os.environ.get('NCCL_NET')} "
        f"FI_PROVIDER={os.environ.get('FI_PROVIDER')}")
    reps = 20
    side = torch.cuda.Stream()
    for b in [256 << 10, 512 << 10, 1 << 20, 2 << 20, 4 << 20, 8 << 20]:
        sb = torch.empty(b, dtype=torch.uint8, device=dev)
        rb = torch.empty(b, dtype=torch.uint8, device=dev)

        def one():
            ops = [dist.P2POp(dist.isend, sb, to), dist.P2POp(dist.irecv, rb, frm)]
            for r in dist.batch_isend_irecv(ops):
                r.wait()
        for mode in ("eager", "graph"):
            g = None
            if mode == "graph":
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        one()
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, capture_error_mode="thread_local"):
                    for _ in range(reps):
                        one()
            ts = []
            for it in range(6):
                torch.cuda.synchronize()
                dist.barrier()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                if g is not None:
                    g.replay()
                else:
                    for _ in range(reps):
                        one()
                e1.record()
                e1.synchronize()
                if it >= 1:
                    ts.append(e0.elapsed_time(e1) * 1000 / reps)
            t = torch.tensor([statistics.median(ts)], device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            log(f"P15 op=nccl_sendrecv mode={mode} bytes={b} us_per_transfer={t.item():.1f} "
                f"GB/s={b / (t.item() * 1e3):.2f}")
    # reference: all-gather of a 256-token bucket of 2048-wide bf16 rows per rank (stock's DP gather payload)
    x = torch.empty(256, 2048, dtype=torch.bfloat16, device=dev)
    out = torch.empty(W * 256, 2048, dtype=torch.bfloat16, device=dev)
    ts = []
    for it in range(12):
        torch.cuda.synchronize()
        dist.barrier()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps):
            dist.all_gather_into_tensor(out, x)
        e1.record()
        e1.synchronize()
        if it >= 2:
            ts.append(e0.elapsed_time(e1) * 1000 / reps)
    t = torch.tensor([statistics.median(ts)], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    total = out.numel() * 2
    log(f"P15 op=allgather_256x2048_bf16 bytes_out={total} us={t.item():.1f} out_GB/s={total / (t.item() * 1e3):.2f} "
        f"remote_in_GB/s={(total * (W - L) / W) / (t.item() * 1e3):.2f}")
    os._exit(0)


if __name__ == "__main__":
    main()
