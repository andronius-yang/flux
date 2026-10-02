"""p11b_nccl_graph.py -- handoff 53 probes P11b + P11d (plan 9). Run with bench/launch.sh (torchrun, 1 rank per GPU).

P11b: the layer's two all-gathers (loads [W, G] int32; routing [W, 2 * S * K] int32) captured in a CUDA graph on a
DEDICATED process group, replayed while the default group runs eager collectives in between (SGLang's own NCCL use),
values checked every replay (each rank writes rank * 1000003 + iteration into its contribution before the replay).
P11d: graph scale: 48 layers x the power-of-two buckets, each graph = the two all-gathers + ~150 small kernels (the
plan-9 layer graph's node count), all in ONE shared mempool; reports device memory, capture + instantiate time,
host CPU per replay() call and GPU time per replay.
"""
import argparse
import os
import time

import torch
import torch.distributed as dist


def small_kernels(x, n):
    for _ in range(n):
        x.add_(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--buckets", default="8,16,32,64,128,256,512,1024,2048,4096")
    ap.add_argument("--nodes-per-graph", type=int, default=150)
    ap.add_argument("--skip-scale", action="store_true")
    a = ap.parse_args()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank, W = dist.get_rank(), dist.get_world_size()
    G, K = 128, 8
    lop = dist.new_group(list(range(W)))     # dedicated lopep communicator
    dev = torch.device("cuda", local)
    log = (lambda *s: print(*s, flush=True)) if rank == 0 else (lambda *s: None)
    log(f"INFO P11b W={W} torch {torch.__version__} nccl {torch.cuda.nccl.version()}")

    def make(S):
        loads_in = torch.zeros(G, dtype=torch.int32, device=dev)
        loads_out = torch.zeros(W * G, dtype=torch.int32, device=dev)
        r_in = torch.zeros(2 * S * K, dtype=torch.int32, device=dev)
        r_out = torch.zeros(W * 2 * S * K, dtype=torch.int32, device=dev)
        scratch = torch.zeros(1024, dtype=torch.float32, device=dev)
        return loads_in, loads_out, r_in, r_out, scratch

    def body(t, nk):
        loads_in, loads_out, r_in, r_out, scratch = t
        dist.all_gather_into_tensor(loads_out, loads_in, group=lop)
        small_kernels(scratch, nk // 2)
        dist.all_gather_into_tensor(r_out, r_in, group=lop)
        small_kernels(scratch, nk - nk // 2)

    # ---------------- P11b: correctness of captured gathers mixed with eager default-group collectives
    S = 256
    t = make(S)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            body(t, a.nodes_per_graph)       # warm the communicator and the kernels outside capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    pool = torch.cuda.graph_pool_handle()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool, capture_error_mode="thread_local"):
        body(t, a.nodes_per_graph)
    eager = torch.zeros(4, device=dev)
    bad = 0
    t0 = time.time()
    for it in range(a.iters):
        v = rank * 1000003 + it
        t[0].fill_(v)
        t[2].fill_(v)
        g.replay()
        eager.fill_(rank)
        dist.all_reduce(eager, group=None)                     # SGLang-like eager NCCL on the default group
        exp = torch.tensor([r * 1000003 + it for r in range(W)], dtype=torch.int32, device=dev)
        got_l = t[1].view(W, G)[:, 0]
        got_r = t[3].view(W, -1)[:, -1]
        bad += int((got_l != exp).sum()) + int((got_r != exp).sum())
        bad += int(eager[0].item() != sum(range(W)))
    torch.cuda.synchronize()
    tb = torch.tensor([bad], device=dev)
    dist.all_reduce(tb)
    log(f"P11b iters={a.iters} mismatches={int(tb.item())} wall_s={time.time() - t0:.2f} -> "
        f"{'ok' if int(tb.item()) == 0 else 'MISMATCH'}")

    if a.skip_scale:
        os._exit(0)
    # ---------------- P11d: scale: layers x buckets graphs in one shared pool
    buckets = [int(x) for x in a.buckets.split(",")]
    torch.cuda.synchronize()
    free0, total = torch.cuda.mem_get_info()
    res0 = torch.cuda.memory_reserved()
    graphs, tensors = [], []
    tc = time.time()
    for layer in range(a.layers):
        for S in buckets:
            tt = make(S)
            with torch.cuda.stream(side):
                body(tt, 4)                                   # warm the shapes (cheap)
            torch.cuda.current_stream().wait_stream(side)
            gg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gg, pool=pool, capture_error_mode="thread_local"):
                body(tt, a.nodes_per_graph)
            graphs.append(gg)
            tensors.append(tt)
    torch.cuda.synchronize()
    tcap = time.time() - tc
    free1, _ = torch.cuda.mem_get_info()
    res1 = torch.cuda.memory_reserved()
    # replay cost: layers in order for the 256 bucket (a decode forward)
    idx = [layer * len(buckets) + buckets.index(256) for layer in range(a.layers)]
    host = []
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for rep in range(20):
        ev0.record()
        for i in idx:
            h0 = time.perf_counter()
            graphs[i].replay()
            host.append(time.perf_counter() - h0)
        ev1.record()
        torch.cuda.synchronize()
    gpu_ms = ev0.elapsed_time(ev1)
    host.sort()
    # intrinsic launch cost: idle GPU before every replay, time only the replay() call
    idle = []
    for rep in range(3):
        for i in idx:
            torch.cuda.synchronize()
            h0 = time.perf_counter()
            graphs[i].replay()
            idle.append(time.perf_counter() - h0)
    torch.cuda.synchronize()
    idle.sort()
    st = torch.tensor([tcap, (free0 - free1) / 2**20, (res1 - res0) / 2**20, host[len(host) // 2] * 1e6,
                       host[len(host) * 9 // 10] * 1e6, gpu_ms / a.layers * 1e3, idle[len(idle) // 2] * 1e6,
                       idle[len(idle) * 9 // 10] * 1e6], device=dev, dtype=torch.float64)
    dist.all_reduce(st, op=dist.ReduceOp.MAX)
    log(f"P11d graphs={len(graphs)} capture+instantiate_s={st[0]:.1f} device_mem_MiB={st[1]:.0f} "
        f"pool_reserved_MiB={st[2]:.0f} replay_host_us(back-to-back) med={st[3]:.1f} p90={st[4]:.1f} "
        f"replay_host_us(idle GPU) med={st[6]:.1f} p90={st[7]:.1f} gpu_us_per_layer_graph={st[5]:.1f}")
    os._exit(0)                                       # skip the NCCL teardown (it hung in the S0 run)


if __name__ == "__main__":
    main()
