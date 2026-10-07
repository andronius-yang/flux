"""Device dual3 swap lane for the serving runtime of the Zepp / LoPEP library (research tree only, case study,
2026-10-06).

The serving runtime moves expert weights through its staged lane: the sender pushes both matrices into the
receiver's staging before the dispatch GEMM, the receiver's GEMMs read the moved experts from the staging behind
per-slot gate words (moved-last schedule), and the staging is committed into the slot after each GEMM. This lane
keeps every part of that machinery (the device swap decision's result block, lane_arm, the gate words, the weight
pointer overrides, the commits) and changes only WHEN and BY WHOM the bytes move, to the 3D (dual3) schedule:

  phase 0 (W1)  the sender pushes under its dispatch GEMM: push_w1 (_dual3_ext.cu) on a side stream, spinning on
                the dispatch op's GEMM-start mark; the sender's W1 commit first waits until its pushes raised
                the receivers' gates (wait_pushed)
  phase 1 (W2)  the receiver pulls under its combine GEMM: pull_w2 on a side stream, spinning on the combine op's
                GEMM-start mark, reading the sender's slot over NVLink (the W2 slot storage is allocated on the
                symmetric heap so node peers can read it); the sender's W2 commit first waits for the receivers'
                acknowledgments (wait_acks)
  No stream waits or event joins on the side streams: a parked stream wait can block a hardware queue the
  forward stream shares (the first attempt hung that way).

Everything is driven by the device result block: no host read of the moves, no host-issued copy. Every step is
armed (deferred verdict); the kernels return at once when the block reports no swap. Epochs are the host
epochs (eager steps; `step_device` off), passed identically to the kernels, the marks and the GEMM gate kwargs.

    lane = install(shared, w2_views, C, group)   # after SharedComm.warmup(); w2_views = the node's symmetric W2 slots
"""
import os

import torch

_ext = None


def load_ext(verbose=False):
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dual3_ext.cu")
    build_dir = os.environ.get("FLUX_DUAL3_EXT_DIR",
                               os.path.expandvars("$PSCRATCH/workspace/andrewy/dual3_ext_build"))
    os.makedirs(build_dir, exist_ok=True)
    _ext = load(name="dual3_ext", sources=[src], build_directory=build_dir,
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"], verbose=verbose)
    return _ext


def symmetric_w2(C, group, gpe, H, ffn, dtype, local_rank):
    """W2 slot storage on the symmetric heap: (this rank's [gpe, H, ffn] view, every node peer's view)."""
    views = C.create_tensor_list([gpe * H, ffn], dtype, group)
    views[local_rank].zero_()                     # only this rank's storage (a peer may already be filling its own)
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)
    return views[local_rank].view(gpe, H, ffn), views


def install(shared, w2_views, C, group):
    """Turn the runtime's (warmed-up) device staged lane into the device dual3 lane, in place."""
    lane = shared.lane
    base = type(lane)
    assert lane._dev and lane.mode == "staged", "the dual3 device lane builds on the device staged lane"
    ext = load_ext()

    class Dual3DeviceLane(base):
        # no stream waits and no event joins on the side streams (see _dual3_ext.cu): the side streams fork from
        # forward-stream work already enqueued, the kernels spin on the GEMM-start marks, the commits join on words
        def staged_phase_before(self, k):
            assert self._dev_rounds is None, "dual3 device lane: deferred-verdict steps only"
            gpe = self.gpe
            cur = torch.cuda.current_stream()
            if k == 0:
                # W1: pushed by the sender under its dispatch GEMM
                self._dispatch_op.set_gemm_start_mark(self.epoch)
                side = self._d3_w1_stream
                self._d3_fork[0].record(cur)
                side.wait_event(self._d3_fork[0])
                with torch.cuda.stream(side):
                    ext.push_w1(self._blk, self._R, self.L, self.rank, self.nlp, self.cap, self.slots.slots(0),
                                self._slot_bytes[0], self._peer_stag[0], self._peer_gate,
                                self._mark_dispatch.data_ptr(), self.epoch, self._d3_ctr[0:1])
                gate = self._gate[0:gpe]
                return dict(weight_signal=gate[1:], weight_signal_epoch=self.epoch, weight_gate_group_start=1,
                            weight_ptr_override=self._ovr[0], sched_expert_order=self._sched_dev, sched_n_front=-1)
            # W2: pulled by the receiver under its combine GEMM
            self._combine_op.set_gemm_start_mark(self.epoch)
            side = self._d3_w2_stream
            self._d3_fork[1].record(cur)
            side.wait_event(self._d3_fork[1])
            with torch.cuda.stream(side):
                ext.pull_w2(self._blk, self._R, self.L, self.rank, self.nlp, self.cap, self._d3_peer_w2,
                            self._slot_bytes[1], self._d3_stag_w2, self._gate, self._d3_peer_ack,
                            self._mark_combine.data_ptr(), self.epoch, self._d3_ctr[1:2])
            gate = self._gate[gpe:2 * gpe]
            return dict(weight_signal=gate[1:], weight_signal_epoch=self.epoch, gate_of_expert=[],
                        weight_ptr_override=self._ovr[1], gate_map=self._gmap)

        def commit_after(self, k):
            # the commit overwrites the slots this rank gave away: their readers must be done
            if k == 0:
                ext.wait_pushed(self._blk, self._R, self.L, self.rank, self.cap, self._peer_gate, self.epoch)
            else:
                ext.wait_acks(self._blk, self._R, self.L, self.rank, self.cap, self._d3_ack, self.epoch)
            return base.commit_after(self, k)

    lane.__class__ = Dual3DeviceLane
    L, cap = lane.L, lane.cap
    lane.step_device = False                     # host epochs everywhere (eager steps)
    acks = C.create_tensor_list([L * cap], torch.int64, group, False, True)
    lane._d3_ack = acks[lane.local_rank]
    lane._d3_ack.zero_()
    lane._d3_peer_ack = torch.tensor([t.data_ptr() for t in acks], dtype=torch.int64, device="cuda")
    lane._d3_peer_w2 = torch.tensor([v.data_ptr() for v in w2_views], dtype=torch.int64, device="cuda")
    lane._d3_stag_w2 = lane._stag_w2_all[lane.local_rank].data_ptr()
    lane._d3_w1_stream, lane._d3_w2_stream = torch.cuda.Stream(), torch.cuda.Stream()
    lane._d3_fork = [torch.cuda.Event(), torch.cuda.Event()]
    lane._d3_ctr = torch.zeros(2, dtype=torch.int32, device="cuda")
    # first launches here (lazy module load), never beside a spinning GEMM: a block reporting no swap returns at once
    idle = torch.zeros(3 + lane._R + lane._R * cap * 4, dtype=torch.int64, device="cuda")
    ext.push_w1(idle, lane._R, L, lane.rank, lane.nlp, cap, lane._stag_w1_all[lane.local_rank], lane._slot_bytes[0], lane._peer_stag[0],
                lane._peer_gate, lane._mark_dispatch.data_ptr(), 0, lane._d3_ctr[0:1])
    ext.wait_pushed(idle, lane._R, L, lane.rank, cap, lane._peer_gate, 0)
    ext.pull_w2(idle, lane._R, L, lane.rank, lane.nlp, cap, lane._d3_peer_w2, lane._slot_bytes[1], lane._d3_stag_w2,
                lane._gate, lane._d3_peer_ack, lane._mark_combine.data_ptr(), 0, lane._d3_ctr[1:2])
    ext.wait_acks(idle, lane._R, L, lane.rank, cap, lane._d3_ack, 0)
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)
    return lane
