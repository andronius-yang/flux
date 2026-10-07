"""Strict correctness check of the Zepp layer (audit 2026-10-07, findings F2/F3/F7). Not a timing run.

replay.py --check compares with atol=1e-2 while |y| ~ 5e-4, so any output (zeros, stale) passes. Here:
  - relative error ||y - ref|| / ||ref|| (global, and per row) against the torch reference
  - a fresh payload every iteration AND rotating routing batches (batch 0 = the published batch the layer was
    sized and primed on; batches 1.. are other samples of the same eval pool), so a plan or graph that does not
    re-plan per iteration fails
  - negative controls per iteration: the previous iteration's output, the reference under batch-0 routing (when
    the batch changed), and zeros under the old atol metric
  - per-iteration redos (capacity re-runs) and swap moves
Run through the repository's launcher:  bench/launch.sh <this file> --model qwen3 --budget-mib 1 ...
"""
import argparse
import dataclasses
import os
import random
import sys

import torch
import torch.distributed as dist

ROOT = os.environ.get("ZEPP_ROOT_CHECK", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "bench"))
import traces  # noqa: E402
import replay as R  # noqa: E402
from zepp import SHAPES, EPMoE, EPMoEConfig  # noqa: E402
from zepp.config import tokens_per_rank  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--budget-mib", type=float, required=True)
ap.add_argument("--comm-strategy", default="overlap")
ap.add_argument("--swap", type=int, default=0)
ap.add_argument("--router-c", type=float, default=0.25)
ap.add_argument("--gpu-plan", type=int, default=1)
ap.add_argument("--act", default="gelu")
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--iters", type=int, default=8)
ap.add_argument("--batches", type=int, default=4)
ap.add_argument("--row-tol", type=float, default=5e-2)
ap.add_argument("--tol", type=float, default=2e-2)
args = ap.parse_args()

rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
world, L = int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_WORLD_SIZE"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", rank=rank, world_size=world)
group = dist.group.WORLD

shape = dataclasses.replace(SHAPES[args.model], act=args.act)
dtype = torch.bfloat16
T = tokens_per_rank(args.budget_mib, shape)
ts = traces.load_slice(args.model)
seed0 = traces.batch_seed(ts, world, L, args.budget_mib, shape.chunk_bytes)


def batch(k):
    if k == 0:
        rows = traces.sample_batch(ts, world, L, args.budget_mib, shape.chunk_bytes)
    else:
        rng = random.Random(seed0 + 7919 * k)
        pool = ts.eval_rows
        rows = [pool[rng.randrange(len(pool))] for _ in range(world * T)]
    return torch.tensor(rows, dtype=torch.int32).view(world, T, shape.topk)


B = [batch(k) for k in range(args.batches)]
pool = ts.pool_rows[: (len(ts.pool_rows) // world) * world]
pool_topk = torch.tensor(pool, dtype=torch.int32).view(world, -1, shape.topk)
gen_p = torch.Generator().manual_seed(777)
probs_all = torch.rand((world * T, shape.topk), generator=gen_p) + 0.5
probs_own = probs_all[rank * T:(rank + 1) * T]
probs_d = probs_own.cuda()
gen_x = torch.Generator(device="cuda").manual_seed(4242 + rank)

cfg = EPMoEConfig(shape=shape, ranks=world, ranks_per_node=L, max_tokens_per_rank=T,
                  comm_strategy=args.comm_strategy, swap=bool(args.swap), router_c=args.router_c,
                  gpu_plan=bool(args.gpu_plan))
layer = EPMoE(cfg, group, sizing_routing=B[0], pool_routing=pool_topk, probs_own=probs_own, dtype=dtype)
layer.load_weights(lambda e: R.expert_w1(e, shape.ffn1, shape.hidden, dtype),
                   lambda e: R.expert_w2(e, shape.ffn_hidden, shape.hidden, dtype))
layer.prime(B[0][rank].cuda(), probs_d)
if rank == 0:
    print(f"[strict] {args.model} nodes={world // L} b{args.budget_mib:g} {args.comm_strategy} swap={args.swap} "
          f"gpu_plan={int(getattr(layer, 'gpu_plan', 0))} T={T} batches={args.batches}", flush=True)


def moves():
    try:
        return int(layer.swap_moves)
    except Exception:
        return -1


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


ok_all, prev_y = True, None
for i in range(args.warmup + args.iters):
    k = 0 if i < args.warmup else (i - args.warmup) % args.batches
    ids = B[k][rank].cuda()
    gen_x.manual_seed(4242 + rank + 1000 * (i + 1))
    x = ((torch.rand((T, shape.hidden), device="cuda", generator=gen_x) * 0.02) - 0.01).to(dtype)
    layer.prep()
    torch.cuda.synchronize()
    dist.barrier()
    r0, m0 = getattr(layer, "redos", 0), moves()
    plan = layer.prepare(ids, probs_d)
    y = layer.forward(x, plan)
    torch.cuda.synchronize()
    yf = y.float().clone()
    if i >= args.warmup:
        ref = R.torch_reference(x, ids, probs_d, shape, dtype).float()
        g = rel(yf, ref)
        row = (yf - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-30)
        bad = int((row > args.row_tol).sum())
        stale = rel(prev_y, ref) if prev_y is not None else float("nan")
        wrong = rel(yf, R.torch_reference(x, B[0][rank].cuda(), probs_d, shape, dtype).float()) if k else float("nan")
        zeros_old = int(bool(torch.isclose(torch.zeros_like(ref), ref, atol=1e-2, rtol=1.5e-2).all()))
        v = torch.tensor([g, float(row.max()), stale if stale == stale else 0.0, wrong if wrong == wrong else 9.0],
                         device="cuda")
        dist.all_reduce(v, op=dist.ReduceOp.MAX)
        w = torch.tensor([float(bad), float(zeros_old), float(getattr(layer, "redos", 0) - r0)], device="cuda")
        dist.all_reduce(w)
        mn = torch.tensor([v[2].item() if stale == stale else 9.0, wrong if wrong == wrong else 9.0], device="cuda")
        dist.all_reduce(mn, op=dist.ReduceOp.MIN)
        it_ok = v[0].item() < args.tol and w[0].item() == 0
        ok_all &= it_ok
        if rank == 0:
            print(f"[strict] it {i} batch {k}: rel max {v[0].item():.2e} row-rel max {v[1].item():.2e} "
                  f"bad rows {int(w[0].item())}/{world * T} | controls: prev-output rel min {mn[0].item():.2f}, "
                  f"batch-0-routing rel min {mn[1].item():.2f}, zeros pass old atol on {int(w[1].item())}/{world} ranks "
                  f"| redos {int(w[2].item())} moves {moves() - m0 if m0 >= 0 else 'n/a'} -> {'PASS' if it_ok else 'FAIL'}",
                  flush=True)
    prev_y = yf
flag = torch.tensor([0 if ok_all else 1], device="cuda")
dist.all_reduce(flag)
if rank == 0:
    print(f"[strict] VERDICT {'PASS' if flag.item() == 0 else 'FAIL'}", flush=True)
sys.stdout.flush()
dist.barrier()
os._exit(0 if flag.item() == 0 else 1)
