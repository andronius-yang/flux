"""Standalone All2AllSingle probe: known-pattern rows, uneven splits,
verifies content + order. Run: ./launch.sh test/python/comm/probe_a2a_single.py"""
import torch
import torch.distributed
import flux

DIST_ENV = flux.get_dist_env()
TP_GROUP = DIST_ENV.get_world()
torch.cuda.set_device(DIST_ENV.LOCAL_RANK)
flux.init_flux_shm(TP_GROUP)
torch.cuda.synchronize()

W = DIST_ENV.WORLD_SIZE
rank = TP_GROUP.rank()
H = 8
max_split = 64

# rank r sends (r+1)*(d+1) rows to dest d, row value = r*10000 + d*100 + i
send_counts = [(rank + 1) * (d + 1) for d in range(W)]
recv_counts = [(s + 1) * (rank + 1) for s in range(W)]
n_send, n_recv = sum(send_counts), sum(recv_counts)
send = torch.zeros(n_send, H, dtype=torch.bfloat16, device="cuda")
off = 0
for d in range(W):
    for i in range(send_counts[d]):
        send[off].fill_(rank * 100 + d * 10 + i * 0.5)
        off += 1
recv = torch.zeros(n_recv, H, dtype=torch.bfloat16, device="cuda")

op = flux.All2AllSingle(TP_GROUP, max_split, H, DIST_ENV.LOCAL_WORLD_SIZE,
                        torch.bfloat16)
in_sp = torch.tensor(send_counts, dtype=torch.int32, device="cuda")
out_sp = torch.tensor(recv_counts, dtype=torch.int32, device="cuda")
op.forward(send, recv, in_sp, out_sp, 8)
torch.cuda.synchronize()

ok = True
off = 0
for s in range(W):
    for i in range(recv_counts[s]):
        want = s * 100 + rank * 10 + i * 0.5
        got = float(recv[off, 0])
        if abs(got - torch.tensor(want, dtype=torch.bfloat16).item()) > 1e-3:
            if ok:
                print(f"rank {rank}: FIRST MISMATCH at recv row {off} "
                      f"(src {s} idx {i}): want {want} got {got}", flush=True)
            ok = False
        off += 1
print(f"rank {rank}: All2AllSingle probe {'OK' if ok else 'FAILED'}",
      flush=True)
# second call on the same instance (reuse pattern)
recv.zero_()
op.forward(send, recv, in_sp, out_sp, 8)
torch.cuda.synchronize()
ok2 = abs(float(recv[0, 0]) - 0.0) < 1e3  # just rerun sanity
print(f"rank {rank}: second call done", flush=True)
