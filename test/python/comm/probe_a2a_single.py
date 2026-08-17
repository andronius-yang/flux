"""Standalone All2AllSingle probe: known-pattern rows, uneven splits,
verifies content + order. Run: ./launch.sh test/python/comm/probe_a2a_single.py

--mismatch: negative test for the 2026-08-17 cross-rank config check — each
rank passes a DIFFERENT max_split to the ctor; the expected outcome is a
loud FLUX_CHECK abort on EVERY rank at construction (never a hang, never
silent wire corruption)."""
import sys

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
# 128: safe margin over the max pair rows (rank+1)*(d+1) <= W*W; the old 64
# sat exactly at the W=8 boundary
max_split = 128

if "--mismatch" in sys.argv:
    try:
        flux.All2AllSingle(TP_GROUP, max_split + rank, H,
                           DIST_ENV.LOCAL_WORLD_SIZE, torch.bfloat16)
    except Exception as e:
        print(f"rank {rank}: mismatch ABORTED IN CTOR as required: "
              f"{str(e).splitlines()[0][:120]}", flush=True)
        sys.exit(0)
    print(f"rank {rank}: MISMATCH NOT DETECTED — contract check missing",
          flush=True)
    sys.exit(1)

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
