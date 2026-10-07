#!/usr/bin/env python3
"""figs/case_study: the case-study workload through the SERVING path of the MoE layer (CS_v7, 2026-10-06).

One MoE layer on the serving runtime (SharedComm + LayerState of the Zepp / LoPEP library): device-issued
dispatch and combine wires, the staged swap lane, the device swap decision and the deferred capacity verdict
(no host read of a routing count inside the forward). Each iteration is one model forward of that single
layer (begin_forward / step / end_forward), eager (the 4680-token bucket is above the graph limit), isolated
by a device sync + barrier, and bracketed by an `iter<i>` NVTX range so figs/case_study/extract_timeline.py
attributes its device work.

Workload = the research runner's (export_serving_inputs.py): K2 shape with the one-matrix GELU expert of the
case-study arm, the oracle placement, the routing files (LiveCodeBench, or the 8-topic S-C schedule with
dwell 4: topic of iteration i = ((i - (total - 1)) // dwell) mod 8), gate weights rand + 0.5 (seed 777).
Swap policy = the case-study arm's (user ruling 2026-10-06, "force"): after the serving warm-up the decision's
band tolerance is -1 (every node out of band = the tau=1 orbit) and the placement (tables + slot weights) is
reset to the oracle before every timed iteration.

  bench/launch.sh <flux>/figs/case_study/serving_case_study.py --inputs sc.pt --pkg zepp --warmup 5 --iters 32
"""
import argparse
import importlib
import json
import os
import sys
import time

import torch
import torch.distributed as dist


def gen_w(kind, e, rows, cols, dtype):
    """the research runner's expert weights (test_moe_ours_traffic.gen_expert_w1/w2): rand * 0.02 - 0.01"""
    g = torch.Generator().manual_seed((10007 if kind == "w1" else 20011) + int(e))
    return ((torch.rand((rows, cols), generator=g) * 0.02) - 0.01).to(dtype)


def reference(x, ids, w, shape, dtype):
    """GELU MoE on this rank's tokens (bf16 GEMMs, fp32 accumulation), the research gate's reference"""
    S, K = ids.shape
    y = torch.zeros(S, shape.hidden, dtype=torch.float32, device=x.device)
    e_flat = ids.reshape(-1).long()
    t_flat = torch.arange(S, device=x.device).repeat_interleave(K)
    p_flat = w.reshape(-1).float()
    for e in torch.unique(e_flat).tolist():
        sel = (e_flat == e).nonzero(as_tuple=True)[0]
        rows = t_flat[sel]
        w1 = gen_w("w1", e, shape.ffn_hidden, shape.hidden, dtype).cuda()
        w2 = gen_w("w2", e, shape.hidden, shape.ffn_hidden, dtype).cuda()
        part = torch.nn.functional.gelu(x[rows] @ w1.t()) @ w2.t()
        y.index_add_(0, rows, part.float() * p_flat[sel].unsqueeze(1))
    return y.to(dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="export_serving_inputs.py output (lcb.pt / sc.pt)")
    ap.add_argument("--pkg", default="zepp", choices=("zepp", "lopep"), help="library providing the serving path")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--router-c", type=float, default=0.25)
    ap.add_argument("--force-swap", type=int, default=1, help="decision band -1 after the warm-up (tau=1 orbit)")
    ap.add_argument("--reset-every", type=int, default=1, help="oracle placement + slot weights before each timed iteration")
    ap.add_argument("--caps-scale", type=float, default=1.0)
    ap.add_argument("--check", type=int, default=0, help="compare every output with the PyTorch reference")
    ap.add_argument("--records", default="", help="directory for per-rank JSONL timings")
    a = ap.parse_args()

    pkg = importlib.import_module(a.pkg)
    serving = importlib.import_module(f"{a.pkg}.serving")
    placement_mod = importlib.import_module(f"{a.pkg}.placement")
    routing_mod = importlib.import_module(f"{a.pkg}.routing")
    swap_mod = importlib.import_module(f"{a.pkg}.swap")
    config_mod = importlib.import_module(f"{a.pkg}.config")

    dist.init_process_group("nccl")
    rank, W = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank % 4)))
    dev = torch.device("cuda")
    group = dist.group.WORLD
    inp = torch.load(a.inputs, weights_only=False)
    assert inp["W"] == W, f"inputs recorded for {inp['W']} ranks, running {W}"
    G, K, H, S, L, nlp = inp["G"], inp["K"], inp["H"], inp["S"], inp["L"], inp["nlp"]
    ffn = 2048
    shape = config_mod.ModelShape(G, K, H, ffn, act="gelu")
    cfg = config_mod.EPMoEConfig(shape=shape, ranks=W, ranks_per_node=L, max_tokens_per_rank=S,
                                 comm_strategy="overlap", swap=True, router_c=a.router_c)
    assert cfg.slots_per_rank == nlp, (cfg.slots_per_rank, nlp)
    dtype = torch.bfloat16
    t0 = time.time()
    p2l = inp["p2l"].int()
    pl0 = placement_mod.Placement(p2l, inp["l2p"].int(), inp["lcnts"].int(), {})

    # capacities: every topic x {oracle placement + its forced swap orbit}, elementwise max, x caps-scale
    C_dec = -1.0 if a.force_swap else a.router_c
    caps_t = []
    for k in range(inp["topk"].shape[0]):
        sizing = inp["topk"][k]
        load_g = torch.bincount(sizing.reshape(-1).long(), minlength=G)
        pls = [pl0] + swap_mod.swap_orbit(load_g, pl0, L, nlp, C_dec)
        caps_t.append(routing_mod.compute_capacities(sizing, pls, nlp, L, a.router_c, group))
    fields = ("recv_cap", "dispatch_recv", "dispatch_stage", "dispatch_relay", "combine_send", "combine_conv",
              "combine_wire", "pair_cap")
    caps0 = routing_mod.Capacities(**{f: max(int(a.caps_scale * max(getattr(c, f) for c in caps_t)), 1) for f in fields})
    if rank == 0:
        print(f"[cs] {a.pkg} W={W} S={S} G={G} H={H} ffn={ffn} gelu topics={inp['topk'].shape[0]} dwell={inp['dwell']} "
              f"caps={caps0} heap={os.environ.get('NVSHMEM_SYMMETRIC_SIZE')}", flush=True)

    try:
        shared = serving.SharedComm(cfg, group, caps0, dtype=dtype, layer_graphs=False)
    except TypeError:                            # lopep p10-f: graphs follow LOPEP_LAYER_GRAPH (eager above 1024 tokens)
        shared = serving.SharedComm(cfg, group, caps0, dtype=dtype)
    C = importlib.import_module(f"{a.pkg}._ext").C
    env = a.pkg.upper()
    want_wire = os.environ.get(f"{env}_WIRE_TRACE", "0") == "1" and hasattr(C, "wire_trace")
    want_pack = os.environ.get(f"{env}_PACK_TRACE", "0") == "1" and hasattr(C, "combine_pack_trace")
    wire_tr, pack_tr = [], []
    st = serving.LayerState(0, cfg, pl0, rank, dtype=dtype)
    for j in range(nlp):
        e = int(p2l[rank * nlp + j])
        if e >= 0:
            st.slots.w1[1 + j].copy_(gen_w("w1", e, ffn, H, dtype))
            st.slots.w2[1 + j].copy_(gen_w("w2", e, H, ffn, dtype))
    torch.cuda.synchronize()
    shared.prime(st)
    shared.warmup(st)
    torch.cuda.synchronize()
    if a.force_swap:
        band = shared._swap_band_c
        shared._swap_band_c = lambda kept=False: band(kept) if (shared._warmup_mode or kept) else -1.0
    snap = dict(p2l=st.p2l.clone(), l2p=st.l2p.clone(), placement=pl0.clone(),
                w1=st.slots.w1.cpu().pin_memory(), w2=st.slots.w2.cpu().pin_memory(), pad=st.pad_table.clone())
    if rank == 0:
        print(f"[cs] setup {time.time() - t0:.1f}s, buckets {shared.buckets}", flush=True)

    gx = torch.Generator(device="cuda").manual_seed(4242 + rank)
    x = ((torch.rand((S, H), device="cuda", generator=gx) * 0.02) - 0.01).to(dtype)
    w_rank = inp["probs"][rank].to(dev)
    topics = [t[rank].to(dev) for t in inp["topk"]]
    n_list = [S] * W
    total = a.warmup + a.iters
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(total)]
    moves, topic_of, bad_rows, max_err, aborts = [], [], 0, 0.0, 0
    for i in range(total):
        k = ((i - (total - 1)) // inp["dwell"]) % len(topics) if len(topics) > 1 else 0
        topic_of.append(k)
        if a.reset_every and i >= a.warmup:
            # untimed placement reset: device tables, host mirror, pad table, slot weights
            st.p2l.copy_(snap["p2l"])
            st.l2p.copy_(snap["l2p"])
            st.pad_table.copy_(snap["pad"])
            st.set_placement_host(snap["placement"].clone())
            st.refresh_pads_if_needed()          # host pad rebuild here, never inside the timed step
            st.slots.w1.copy_(snap["w1"])
            st.slots.w2.copy_(snap["w2"])
        torch.cuda.synchronize()
        dist.barrier()
        m0 = shared.swap_moves
        with torch.cuda.nvtx.range(f"iter{i}_warmup" if i < a.warmup else f"iter{i}"):
            evs[i][0].record()
            shared.begin_forward()
            y = shared.step(st, x, topics[k], w_rank, n_list)
            shared.end_forward()
            evs[i][1].record()
        abort = shared.forward_check()
        if abort is not None:
            aborts += 1
            if rank == 0:
                print(f"[cs] iter {i}: {abort}; recover (the iteration is void for the timeline)", flush=True)
            shared.recover(abort, st)
            shared.begin_forward()
            y = shared.step(st, x, topics[k], w_rank, n_list)
            shared.end_forward()
            assert shared.forward_check() is None
        moves.append(shared.swap_moves - m0)
        # device timestamp instruments (lopep debug builds), last launch wins: read back every iteration in the
        # untimed gap (the readback synchronizes the device)
        if want_wire:
            wire_tr.append(C.wire_trace().cpu().clone())
        if want_pack:
            pack_tr.append(C.combine_pack_trace().cpu()[:64].clone())
        if a.check:
            torch.cuda.synchronize()
            ref = reference(x, topics[k], w_rank, shape, dtype)
            bad = int((~torch.isclose(y.float(), ref.float(), atol=1e-2, rtol=1.5e-2)).any(dim=1).sum())
            err = float((y.float() - ref.float()).abs().max())
            bad_rows += bad
            max_err = max(max_err, err)
            print(f"[cs] r{rank} iter {i} topic {k} moves {moves[-1]} check {'OK' if bad == 0 else 'BAD'} "
                  f"({bad} rows) maxdiff {err:.3e} ref_absmax {float(ref.float().abs().max()):.3e}", flush=True)
    torch.cuda.synchronize()
    ms = [e0.elapsed_time(e1) for e0, e1 in evs]
    if a.records:
        os.makedirs(a.records, exist_ok=True)
        with open(os.path.join(a.records, f"rank_{rank:03d}.jsonl"), "w") as f:
            f.write(json.dumps(dict(type="meta", rank=rank, pkg=a.pkg, inputs=a.inputs)) + "\n")
            f.write(json.dumps(dict(type="iters", metric="step_ms", values_ms=ms[a.warmup:])) + "\n")
            f.write(json.dumps(dict(type="cell_info", ours_moves=moves, ours_topic=topic_of, aborts=aborts)) + "\n")
    if a.records and (wire_tr or pack_tr):
        import numpy as np
        np.savez(os.path.join(a.records, f"traces_r{rank}.npz"),
                 wire=torch.stack(wire_tr).numpy() if wire_tr else np.zeros(0),
                 pack=torch.stack(pack_tr).numpy() if pack_tr else np.zeros(0),
                 warmup=a.warmup, rank=rank)
    allms = [None] * W
    dist.all_gather_object(allms, ms)
    flag = torch.tensor([bad_rows], device=dev)
    dist.all_reduce(flag)
    if rank == 0:
        import statistics
        per_iter = [max(allms[r][i] for r in range(W)) for i in range(a.warmup, total)]
        print(f"[cs] step ms (host-event bracket, max over ranks): median {statistics.median(per_iter):.3f} "
              f"min {min(per_iter):.3f} max {max(per_iter):.3f}; moves/iter (rank 0) {moves}; aborts {aborts}"
              + (f"; check bad rows {int(flag)} -> {'PASS' if int(flag) == 0 else 'FAIL'}" if a.check else ""),
              flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
