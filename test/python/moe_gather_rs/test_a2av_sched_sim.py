################################################################################
#
# Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
"""Enqueue-order deadlock check for the layer1 a2av_hier combine schedule.

Models the run_a2av_hier enqueue order under the PESSIMISTIC
CUDA_DEVICE_MAX_CONNECTIONS=1 semantics: per rank, every host-issued op
(front-end wait, put+signal, memset) lands in one FIFO in exact enqueue
order, and a blocked wait stalls every later-enqueued host op of that rank
regardless of stream. Persistent kernels (GEMM cascade, pack, eager reduce)
are RESIDENT: their launch never blocks, they progress on their own program
counters (SM reservation guarantees residency), and their internal spins
consume signals without occupying the front-end channel.

This is strictly harsher than the hardware (which sometimes overlaps streams
even at conn=1), so a schedule that terminates here is safe; a schedule that
deadlocks here is at minimum fragile. Precedent: the layer0 executor in
test_relay_balance_math.py (per-stream FIFOs); this one collapses each rank's
streams into one FIFO because that is the failure mode the layer0
walkthroughs documented for conn=1.

Run: python3 test_a2av_sched_sim.py   (no GPU, no flux import)
"""

import itertools


def build_programs(W, L, n_split, eager):
    """Return (host_progs, kernel_progs): rank -> list of ('wait'|'set', key) ops.

    Signal keys (all monotone-epoch, GEQ semantics):
      ('barrier', r, sid)      GEMM split-cascade flag on rank r (resident GEMM)
      ('gflag', r, g, sid)     pack kernel (g, sid) chunk flag on rank r
      ('arrival', r, ns, sid)  source node ns's aggregate landed at gateway r
      ('recv', r, s, sid)      source s's chunk for split sid delivered to r
    """
    NN = W // L
    host_progs, kernel_progs = {}, {}
    for r in range(W):
        n, lr = r // L, r % L

        # resident GEMM: releases split flags in ascending sid order
        gemm = [("set", ("barrier", r, sid)) for sid in range(n_split)]
        # resident pack kernel: per split, remote-node chunks first
        pack = []
        for sid in range(n_split):
            pack.append(("wait", ("barrier", r, sid)))
            for gi in range(NN):
                g = (n + 1 + gi) % NN
                pack.append(("set", ("gflag", r, g, sid)))
        kernels = [gemm, pack]

        # resident eager reduce kernel: consumes every per-source signal, split-major
        if eager:
            red = []
            for sid in range(n_split):
                for s in range(W):
                    red.append(("wait", ("recv", r, s, sid)))
            kernels.append(red)
        kernel_progs[r] = kernels

        # host FIFO: exact run_a2av_hier enqueue order (interleaved per split)
        ops = []
        for sid in range(n_split):
            if NN > 1:
                for gi in range(NN - 1):  # inter ladder
                    tn = (n + 1 + gi) % NN
                    g = tn * L + lr
                    ops.append(("wait", ("gflag", r, tn, sid)))
                    ops.append(("set", ("arrival", g, n, sid)))
            ops.append(("wait", ("gflag", r, n, sid)))  # intra ladder
            for dl in range(L):
                d = n * L + (lr - dl + L) % L
                ops.append(("set", ("recv", d, r, sid)))
            if NN > 1:
                for dn in range(1, NN):  # gateway forwards
                    ns = (n + dn) % NN
                    s = ns * L + lr
                    ops.append(("wait", ("arrival", r, ns, sid)))
                    for dl in range(L):
                        d = n * L + (lr - dl + L) % L
                        ops.append(("set", ("recv", d, s, sid)))
            if not eager:  # legacy reduce: wait-all-W then (non-blocking) launch
                for s in range(W):
                    ops.append(("wait", ("recv", r, s, sid)))
        host_progs[r] = ops
    return host_progs, kernel_progs


def run_schedule(W, L, n_split, eager, epochs=2):
    signals = {}

    def fired(key, run_id):
        return signals.get(key, 0) >= run_id

    for run_id in range(1, epochs + 1):
        host_progs, kernel_progs = build_programs(W, L, n_split, eager)
        host_pc = {r: 0 for r in host_progs}
        kern_pc = {(r, i): 0 for r in kernel_progs for i in range(len(kernel_progs[r]))}
        while True:
            progressed = False
            for r, ops in host_progs.items():
                while host_pc[r] < len(ops):
                    op, key = ops[host_pc[r]]
                    if op == "wait" and not fired(key, run_id):
                        break  # conn=1: stalls ALL later host ops of rank r
                    if op == "set":
                        signals[key] = run_id
                    host_pc[r] += 1
                    progressed = True
            for (r, i), _ in list(kern_pc.items()):
                prog = kernel_progs[r][i]
                while kern_pc[(r, i)] < len(prog):
                    op, key = prog[kern_pc[(r, i)]]
                    if op == "wait" and not fired(key, run_id):
                        break  # resident spin: blocks only this kernel
                    if op == "set":
                        signals[key] = run_id
                    kern_pc[(r, i)] += 1
                    progressed = True
            host_done = all(host_pc[r] == len(host_progs[r]) for r in host_progs)
            kern_done = all(kern_pc[k] == len(kernel_progs[k[0]][k[1]]) for k in kern_pc)
            if host_done and kern_done:
                break
            if not progressed:
                stuck_h = {r: host_progs[r][host_pc[r]] for r in host_progs
                           if host_pc[r] < len(host_progs[r])}
                raise AssertionError(
                    f"deadlock (W={W} L={L} n_split={n_split} eager={eager} "
                    f"epoch={run_id}): host stuck at {stuck_h}")
        # completeness: every (dest, source, split) delivery signal fired
        for d in range(W):
            for s in range(W):
                for sid in range(n_split):
                    assert fired(("recv", d, s, sid), run_id), (d, s, sid)


if __name__ == "__main__":
    grids = itertools.product(
        [(4, 4), (8, 4), (16, 4), (8, 2)],  # (W, L)
        [1, 4],                              # n_split
        [False, True],                       # eager
    )
    count = 0
    for (W, L), n_split, eager in grids:
        run_schedule(W, L, n_split, eager)
        count += 1
    print(f"PASS: {count} schedules terminate under pessimistic conn=1 semantics")
