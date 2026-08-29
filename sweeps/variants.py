"""Canonical algorithm-variant table for the sweep runner (sweeps/SCHEMA.md).

A variant is a stable name for one measurable configuration of the layer0
dispatch: the --comm_pattern CLI flag plus any construction-time env knobs.
`requires` lists env-knob strings that must exist in the built libflux_cuda
binary — the runner probes the .so (see sweep.py) and marks cells whose
variant needs an absent knob as skipped_capability instead of running them
against a build that would silently ignore the env.
"""

VARIANTS = {
    # FAST load-balancing alltoallv + un-overlapped GemmGroupedV2 (the second
    # non-overlapped baseline). driver="fast" swaps launcher/test/arg-map in
    # sweep.py; e2e mode only (its phase decomposition is captured structurally
    # — host-blocking alltoallv — so there is no separate phases cell); needs
    # >= 2 nodes and a built libflash.so (scripts/build_fast.sh, per checkout).
    # Since 2026-08-21 it consumes --routing_file on real trace cells (FAST is
    # a comm-phase substitute: matrix AND gemm loads trace-derived) and is
    # rule-5 converted (in-window routing allgather + derive_fast_meta_gpu).
    "fast": dict(
        comm_pattern="fast_bvn_a2av",  # cells.csv label only, never a CLI flag
        driver="fast",
        env={},
        requires=[],
        requires_file="3rdparty/FAST/nvidia/libflash.so",
    ),
    # MoonEP-semantics redundant-expert dispatch (semantic port of
    # MoonshotAI/MoonEP; see python/flux/testing/moonep_semantics.py and the
    # vendored oracle under test/python/moe_ag_scatter/moonep_oracle/).
    # driver="moonep" swaps the test file in sweep.py. Pure NCCL + local
    # scatters + per-segment GemmOnly: no NVSHMEM heap, no FLUX_A2AV_* knobs.
    # NOTE 2026-08-11: the driver's own --transport default flipped to
    # nvshmem (fidelity-first — one-sided put-then-barrier is the authentic
    # port of MoonEP's one-sided writes); the nccl arms below pin
    # --transport nccl explicitly so their historical capsule meaning is
    # byte-identical to pre-flip cells.
    # NOTE 2026-08-12: --prefetch_transport default flipped to getmem the
    # same way (faithful baseline = nvshmem dispatch + getmem weight pull);
    # every NCCL-prefetch arm below pins --prefetch_transport nccl
    # explicitly for the same historical-meaning reason.
    # Phase metrics (plan_comm/pack/comm/scatter/prefetch/gemm) arrive free in
    # every mode via the recorder, so there is no separate phases cell.
    # CAN consume trace routing files (real token-overlap dedup semantics).
    # All moonep arms pin CUDA_DEVICE_MAX_CONNECTIONS=8 (2026-08-08 A/B,
    # capsules 20260808-015920 conn=1 vs 20260808-032217 conn=8): at conn=1
    # the single hardware queue serialized the overlap arms' cross-stream
    # work — prefetch_wait read ~0 while the cost hid inside pack_ms (6.4 ms)
    # and overlap showed no net win; at conn=8 the overlap genuinely runs
    # (pack 0.6 ms, overlap total 8.89 vs serialized 10.48 isolated). The
    # serialized arm also improved slightly (11.04 -> 10.48). Cells before
    # the pin ran conn=1 — env_json audits which is which; do not compare
    # across the boundary.
    "moonep": dict(
        comm_pattern="moonep_balanced_a2av",  # cells.csv label only, never a CLI flag
        driver="moonep",
        test_args=["--transport", "nccl", "--prefetch_transport", "nccl"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    # M4c: weight prefetch on a dedicated high-priority stream + separate
    # NCCL communicator, event-joined before GEMM — MoonEP's async_finish
    # comm-stream semantics. Emits prefetch_wait_ms (exposed stall) alongside
    # prefetch_ms (stream duration). Compare against `moonep` (serialized).
    "moonep_overlap": dict(
        comm_pattern="moonep_balanced_a2av",
        driver="moonep",
        test_args=["--transport", "nccl", "--prefetch_transport", "nccl",
                   "--overlap_prefetch"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    # AUTHENTIC-serialization counterpart of moonep_overlap (NR-12 fact 7):
    # upstream MoonEP shares ONE comm stream between dispatch and prefetch
    # (api.py:487-499) — they serialize with each other and overlap only
    # main-stream compute. This arm enqueues prefetch after the dispatch
    # a2av on the SAME communicator (overlap window = scatter only).
    # moonep_overlap (prefetch concurrent with dispatch, separate
    # communicator) is the finer-than-upstream counterpart.
    "moonep_overlap_shared": dict(
        comm_pattern="moonep_balanced_a2av",
        driver="moonep",
        test_args=["--transport", "nccl", "--prefetch_transport", "nccl",
                   "--overlap_prefetch", "--shared_comm_stream"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    # M4a: dispatch a2av over flux's one-sided NVSHMEM All2AllSingle.
    # Live path (a2a_single_kernel_v2): putmem_nbi per destination into
    # fixed per-source slots of symmetric staging + 2 team stream barriers
    # per call — put-then-barrier, which is actually CLOSER to MoonEP's
    # real dispatch (TMA push + single system-scope exit barrier) than
    # put+signal would be. (The putmem_signal kernel in
    # all2all_single_2d_impl.cu is dead code, never launched — NR-12.)
    # Same plan, pack order, placement, and correctness checks as `moonep`;
    # only the transport differs. Needs the NVSHMEM heap (runner sizes
    # symmetric bufs from the plan; sweep sets NVSHMEM_SYMMETRIC_SIZE).
    "moonep_nvshmem": dict(
        comm_pattern="moonep_balanced_a2av",
        driver="moonep",
        test_args=["--transport", "nvshmem", "--prefetch_transport", "nccl"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    # M4a + M4c combined: one-sided wire and overlapped prefetch.
    "moonep_nvshmem_overlap": dict(
        comm_pattern="moonep_balanced_a2av",
        driver="moonep",
        test_args=["--transport", "nvshmem", "--prefetch_transport", "nccl",
                   "--overlap_prefetch"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    # M4d: getmem weight prefetch — the authentic recreation of MoonEP's
    # destination-side pull (NR-12 fact 6: receiver remote-READS the home
    # rank's immutable weight rows; zero signaling; the NVSHMEM analog is
    # getmem, not put). Weights live permanently on the symmetric heap
    # (upstream memory model; enables the future B=3-4 read-through arm);
    # the SM kernel issues nvshmemx_getmem_nbi_block chunks + one
    # quiet_on_stream — no barriers, no communicator. Cross-node pulls are
    # proxy-mediated CXI (extrapolation per NR-12 fact 7). Token wire kept
    # NCCL here to isolate the weight-path change against the historical
    # `moonep` arm inside one capsule.
    "moonep_getmem": dict(
        comm_pattern="moonep_balanced_a2av",
        driver="moonep",
        test_args=["--transport", "nccl", "--prefetch_transport", "getmem"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    # M4d + M4c: getmem pull on the overlap side stream. Unlike the NCCL
    # overlap arm this needs NO second communicator (one-sided) and NO team
    # barrier coexists with the token a2av (the NR-02/fact-8 hazard that
    # rejected All2AllSingle for weights does not apply to bare gets).
    "moonep_getmem_overlap": dict(
        comm_pattern="moonep_balanced_a2av",
        driver="moonep",
        test_args=["--transport", "nccl", "--prefetch_transport", "getmem",
                   "--overlap_prefetch"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    # M4a + M4d: fully one-sided configuration — nvshmem token wire AND
    # getmem weight pull; the most transport-faithful moonep arm.
    "moonep_nvshmem_getmem": dict(
        comm_pattern="moonep_balanced_a2av",
        driver="moonep",
        test_args=["--transport", "nvshmem", "--prefetch_transport", "getmem"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    # AUTHENTIC L0+L1 (2026-08-17): the full staged journey — dispatch +
    # gemm1 + gelu + gemm2 + combine — zero overlap, faithful defaults.
    # The prefetch phase moves BOTH projections (w1+w2) back-to-back under
    # one join, mirroring upstream's one-pass/one-sync prefetch_weight
    # (MoonEP api.py:158-173; port models an ungated FFN => 2 of upstream's
    # 3 matrices, disclosed deviation). Combine = the dispatch mirror:
    # scale by route weights, reverse-dedup partial sums at the expert
    # side, DIRECT single-stage a2av transpose over the same All2AllSingle
    # (upstream has no inter-node path — NR-12 fact 7 extrapolation, same
    # declaration as the dispatch transport), index_add at token home.
    "moonep_l01_nvshmem_getmem": dict(
        comm_pattern="moonep_balanced_a2av",
        driver="moonep",
        layer="l01",
        test_args=["--transport", "nvshmem", "--prefetch_transport", "getmem",
                   "--layers", "l01"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    # MERGED ARM (our-optimization ablation, never quotable as MoonEP
    # behavior): the MoonEP plan drives the FUSED GemmGroupedV2AGScatterOp
    # through the virtual expert space (flux.testing.moonep_fused_map) —
    # flux lb_union compress wire + tile-level comm/GEMM overlap executing
    # MoonEP's exact placement (plan bit-identical to the staged arms).
    # Weights: getmem pull event-joined before forward (scenario 1). The
    # driver computes the EXACT FLUX_A2AV_MAX_* knobs from the plan
    # (sweep.py deliberately does not scale_knobs this driver; the ctor
    # defaults overflow under lb_union union-sized recv regions at low
    # topk). Compare against `moonep`/`moonep_nvshmem_getmem` (staged,
    # faithful) and `hier_compress_lb_union` (flux-native placement)
    # inside one capsule.
    "moonep_fused": dict(
        comm_pattern="moonep_fused_a2av",
        driver="moonep_fused",
        test_args=[],
        env={"FLUX_A2AV_LB_UNION": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_LB_UNION"],
    ),
    # moonep_fused with the weight PUSH (WeightPushMulticast, direct per-pair
    # CE putmem_signal from the home ranks; zero-SM join before forward =
    # the ungated A/B baseline for the M5 tile-gated arm). Same wire bytes
    # as getmem in the opposite direction; the initiator/dependency flip is
    # the measured variable (a straggler HOME now stalls the destination).
    "moonep_fused_push": dict(
        comm_pattern="moonep_fused_a2av",
        driver="moonep_fused",
        test_args=["--weight_path", "push"],
        env={"FLUX_A2AV_LB_UNION": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_LB_UNION"],
    ),
    # THE concurrency arm (scenario 2 proper): no destination-side join —
    # only prefetch-slot tiles spin on their slot's weight epoch signal
    # (weight-gated tiles), so token dispatch, weight push, and GEMM are all
    # in flight together and the two wire flows share the CXI proxy. A/B
    # against moonep_fused_push (same wire, gate moved to the stream
    # front-end): outputs should be torch.equal; the delta is pure
    # scheduling.
    "moonep_fused_push_gated": dict(
        comm_pattern="moonep_fused_a2av",
        driver="moonep_fused",
        test_args=["--weight_path", "push", "--weight_gate", "tiles"],
        env={"FLUX_A2AV_LB_UNION": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        # slot-last by binary default since NR-14 (see canonicalization note)
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_SCHED_PREFETCH_LAST"],
    ),
    # M4: weight multicast — ONE inter-node put per (expert, dest node) into
    # a plan-chosen gateway's own slot, NVLink CE fan-out to the other needy
    # ranks (paced by a zero-SM wait on the gateway's slot signal). Expert
    # replication is naturally a node-level multicast; the bench argues node
    # ingress is the binding resource (a2av_comm_bench methodology §prefetch),
    # so this is the wire-shape ablation vs _push. The capsule's
    # wpush_internode_bytes_{direct,mcast} record the dedup factor.
    "moonep_fused_push_mcast": dict(
        comm_pattern="moonep_fused_a2av",
        driver="moonep_fused",
        test_args=["--weight_path", "push", "--weight_push_mode", "mcast"],
        env={"FLUX_A2AV_LB_UNION": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_LB_UNION"],
    ),
    # M4 + M5: multicast weight wire AND weight-gated tiles — the full
    # scenario-2 configuration (dispatch, multicast weights, GEMM all
    # concurrent).
    "moonep_fused_push_mcast_gated": dict(
        comm_pattern="moonep_fused_a2av",
        driver="moonep_fused",
        test_args=["--weight_path", "push", "--weight_push_mode", "mcast",
                   "--weight_gate", "tiles"],
        env={"FLUX_A2AV_LB_UNION": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        # slot-last by binary default since NR-14 (see canonicalization note)
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_SCHED_PREFETCH_LAST"],
    ),
    # F-C (NR-13): plan-aware push — the driver engages the gateway
    # machinery only when the replicated census finds a real fan-out group
    # (n_multi_groups > 0), else runs pure direct. The resolved mode +
    # census land in the capsule info (wpush_mode_resolved, wpush_*).
    "moonep_fused_push_auto": dict(
        comm_pattern="moonep_fused_a2av",
        driver="moonep_fused",
        test_args=["--weight_path", "push", "--weight_push_mode", "auto"],
        env={"FLUX_A2AV_LB_UNION": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_LB_UNION"],
    ),
    # SCHEDULE-KNOB NOTE (2026-08-15, NR-14 amendment): slot-last
    # (FLUX_A2AV_SCHED_PREFETCH_LAST) stays DEFAULT OFF. Canonicalizing it was
    # attempted and reverted — under an order-controlled repeat it is a ~29%
    # REGRESSION at b64, not the -15% win capsule 20260814-145605 appeared to
    # show (that was a first-cell transient). The gated arms below therefore
    # run the historical interleaved order by default; `_slotlast` opts in and
    # `_slotinterleave` pins the default explicitly for A/B work. Every gated
    # arm lists the knob in `requires` so a build without it is skipped rather
    # than silently measured.
    # LESSON (SCHEMA.md protocol rule 4 in practice): at b64 a single cell can
    # sit 3-6 ms off its own twin. Never headline a schedule/ordering effect
    # from one capsule — repeat it, and reverse the arm order.
    "moonep_fused_push_auto_gated": dict(
        comm_pattern="moonep_fused_a2av",
        driver="moonep_fused",
        test_args=["--weight_path", "push", "--weight_push_mode", "auto",
                   "--weight_gate", "tiles"],
        env={"FLUX_A2AV_LB_UNION": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_SCHED_PREFETCH_LAST"],
    ),
    # EGRESS NIC-SHARDING (2026-08-17): cross-node weight legs byte-split
    # across the home node's same-local-rank wires with dest-side NVLink
    # reassembly (all L NICs on BOTH ends; SIGNAL_ADD arrival + dest-side
    # finalize SET keeps join()/tile gates unchanged). --weight_shard auto
    # shards legs >= the threshold; census + resolution RECORDER-audited
    # (wshard_*). The shard machinery waits ride a dedicated late-drained
    # stream (NR-13 F-B extended); shard_ms brackets its window.
    # OPTIMIZED L0+L1 (2026-08-17): fused l0 (lb_union + push auto + tile
    # gate) + the virtual-space fused gather-rs combine (compress, the W16
    # combined winner) with INHERITED metadata — built once at setup, the
    # no-recalc l01 contract. BOTH matrices push upfront in one issue window
    # (two WeightPushMulticast instances, same plan; op_w2.weight_full feeds
    # gather_rs with zero copies); gemm2 gates on an explicit op_w2.join()
    # (v1 — gemm2 tile gating is the named follow-up). prefetch_ms stays its
    # own bracket: the always-rent baseline any future persistent-experts
    # (keep-stale) arm is judged against.
    "moonep_fused_l01_push_auto_gated": dict(
        comm_pattern="moonep_fused_a2av",
        driver="moonep_fused",
        layer="l01",
        l1_pattern="a2av_hier_compress",
        test_args=["--weight_path", "push", "--weight_push_mode", "auto",
                   "--weight_gate", "tiles", "--layers", "l01"],
        env={"FLUX_A2AV_LB_UNION": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_SCHED_PREFETCH_LAST"],
    ),
    "moonep_fused_l01_push_auto_gated_shard": dict(
        comm_pattern="moonep_fused_a2av",
        driver="moonep_fused",
        layer="l01",
        l1_pattern="a2av_hier_compress",
        test_args=["--weight_path", "push", "--weight_push_mode", "auto",
                   "--weight_gate", "tiles", "--layers", "l01",
                   "--weight_shard", "auto"],
        env={"FLUX_A2AV_LB_UNION": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_SCHED_PREFETCH_LAST"],
    ),
    "moonep_fused_push_auto_shard": dict(
        comm_pattern="moonep_fused_a2av",
        driver="moonep_fused",
        test_args=["--weight_path", "push", "--weight_push_mode", "auto",
                   "--weight_shard", "auto"],
        env={"FLUX_A2AV_LB_UNION": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_LB_UNION"],
    ),
    "moonep_fused_push_auto_gated_shard": dict(
        comm_pattern="moonep_fused_a2av",
        driver="moonep_fused",
        test_args=["--weight_path", "push", "--weight_push_mode", "auto",
                   "--weight_gate", "tiles", "--weight_shard", "auto"],
        env={"FLUX_A2AV_LB_UNION": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_SCHED_PREFETCH_LAST"],
    ),
    # Explicit interleaved (stage-major) order — same as the default, pinned so
    # an A/B pair states its schedule on both sides instead of relying on the
    # binary default. This is the FAST arm at b64 (~22.2 ms vs slot-last ~28.7).
    "moonep_fused_push_auto_gated_slotinterleave": dict(
        comm_pattern="moonep_fused_a2av",
        driver="moonep_fused",
        test_args=["--weight_path", "push", "--weight_push_mode", "auto",
                   "--weight_gate", "tiles"],
        env={"FLUX_A2AV_LB_UNION": "1", "FLUX_A2AV_SCHED_PREFETCH_LAST": "0",
             "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_SCHED_PREFETCH_LAST"],
    ),
    # E1 (NR-14): tokens-first issue order — the fused forward (token a2av +
    # GEMM) is host-enqueued BEFORE the weight push, so the latency-critical
    # token windows own the NIC first and the bulk weight legs ride behind
    # them (pure fire-ordering, no completion waits). Driver-level flag, same
    # binary as auto_gated; A/B within one capsule.
    # E1 ALONE (issue-order flip without the slot-last schedule) — pins the
    # schedule knob off so this stays the single-variable ablation it was in
    # capsule 20260814-145605 after the default flip. NR-14: this arm REGRESSES
    # isolated b1/b8 (+15%/+8%); kept as an ablation, never a default.
    "moonep_fused_push_auto_gated_tokfirst": dict(
        comm_pattern="moonep_fused_a2av",
        driver="moonep_fused",
        test_args=["--weight_path", "push", "--weight_push_mode", "auto",
                   "--weight_gate", "tiles",
                   "--weight_issue_order", "tokens_first"],
        env={"FLUX_A2AV_LB_UNION": "1", "FLUX_A2AV_SCHED_PREFETCH_LAST": "0",
             "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_SCHED_PREFETCH_LAST"],
    ),
    # E2 = NR-13 F-D: FLUX_A2AV_SCHED_PREFETCH_LAST reorders the static tile
    # schedule so ALL resident problems precede ALL prefetch-slot problems
    # (stage-major within each class) — the weight spin moves to the tail of
    # the wavefront where it overlaps resident compute instead of idling the
    # persistent fleet (NR-13 fact 5: 2.4-5x slot-tile spin).
    # NOT a default: the order-controlled repeats (NR-14 amendment 2026-08-15)
    # make these the SLOW arms at b64 (~28.7 vs ~22.2 ms). Retained as the
    # opt-in ablation and so the ladder spec / capsule 20260814-145605 still
    # resolve to the same configuration.
    "moonep_fused_push_auto_gated_slotlast": dict(
        comm_pattern="moonep_fused_a2av",
        driver="moonep_fused",
        test_args=["--weight_path", "push", "--weight_push_mode", "auto",
                   "--weight_gate", "tiles"],
        env={"FLUX_A2AV_LB_UNION": "1", "FLUX_A2AV_SCHED_PREFETCH_LAST": "1",
             "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_SCHED_PREFETCH_LAST"],
    ),
    # E1 + E2 combined: the NR-14 headline arm (phase-ordered wire AND
    # slot-last wavefront).
    "moonep_fused_push_auto_gated_tokfirst_slotlast": dict(
        comm_pattern="moonep_fused_a2av",
        driver="moonep_fused",
        test_args=["--weight_path", "push", "--weight_push_mode", "auto",
                   "--weight_gate", "tiles",
                   "--weight_issue_order", "tokens_first"],
        env={"FLUX_A2AV_LB_UNION": "1", "FLUX_A2AV_SCHED_PREFETCH_LAST": "1",
             "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_SCHED_PREFETCH_LAST"],
    ),
    # UltraEP-semantics replicated-expert balancing (semantic port of
    # Dots-Infra/UltraEP; see python/flux/testing/ultraep_semantics.py, bit-
    # equality-tested vs the real kernels + vendored goldens under
    # test/python/moe_ag_scatter/ultraep_oracle/). driver="ultraep" swaps the
    # test file in sweep.py. Pure NCCL + local scatters + per-segment
    # GemmOnly (moonep pattern): no NVSHMEM heap, no FLUX_A2AV_* knobs.
    # Phase metrics (plan_comm/pack/comm/scatter/prefetch=weight_sync/gemm)
    # arrive free in every mode via the recorder. CAN consume trace routing
    # files. Key semantic contrasts vs moonep: replication confined to the
    # NVLink domain (per-node solves; gemm_rows_per_rank NOT constant —
    # residual imbalance floor ultraep_lb_floor is a cell fact), and NO wire
    # dedup (ultraep_dup_rows audits the delta). conn=8 pin inherited from
    # the moonep-family A/B (same NCCL + cross-stream shape).
    "ultraep": dict(
        comm_pattern="ultraep_quota_a2av",  # cells.csv label only, never a CLI flag
        driver="ultraep",
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    # Rack-scale-single-node counterfactual: the whole EP group treated as
    # one 16-GPU "NVLink domain" (what UltraEP's algorithm would do if the
    # scale-up fabric spanned all 4 nodes). Replicas and weight_sync then
    # cross nodes over Slingshot; LB floor -> 1.0. Prices the fabric
    # assumption; NOT a faithful Perlmutter deployment.
    "ultraep_domain16": dict(
        comm_pattern="ultraep_quota_a2av",
        driver="ultraep",
        test_args=["--nvl_domain_size", "16"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    # Overlapped weight_sync, AUTHENTIC join point (NR-12 fact 5): upstream
    # launches weight_sync async on a dedicated comm stream and joins BEFORE
    # token dispatch — the join is the publication mechanism for its
    # unsignaled peer-VA pushes (direct mode has no receiver-side
    # signaling), not a tuning choice. Overlap window = pack only;
    # weight_sync stays mostly exposed, matching the paper's own exposed-
    # critical-path accounting. Emits prefetch_wait_ms.
    "ultraep_overlap": dict(
        comm_pattern="ultraep_quota_a2av",
        driver="ultraep",
        test_args=["--overlap_ws"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    # COUNTERFACTUAL join point (label like ultraep_domain16, never quote as
    # UltraEP behavior): joins weight_sync before the GEMM instead — legal
    # under the data dependency and sound ONLY because this NCCL port is
    # two-sided (irecv completion rides the ws event; upstream's unsignaled
    # pushes would be unsound here). Prices Perlmutter's split fabrics
    # (weight_sync on NVLink, dispatch on Slingshot — a tradeoff upstream
    # never faced). Window = pack+comm+scatter.
    "ultraep_overlap_joingemm": dict(
        comm_pattern="ultraep_quota_a2av",
        driver="ultraep",
        test_args=["--overlap_ws", "--ws_join", "gemm"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    # One-sided NVSHMEM token transport (All2AllSingle live path: putmem_nbi
    # + 2 team barriers per call — the transport class of UltraEP's own
    # external dispatchers, DeepEP/HybridEP). Weight_sync stays NCCL P2P
    # (declared port artifact, NR-12 fact 8). Sweep sets
    # NVSHMEM_SYMMETRIC_SIZE via ultraep_sym_size (no-dedup, domain-bounded).
    "ultraep_nvshmem": dict(
        comm_pattern="ultraep_quota_a2av",
        driver="ultraep",
        test_args=["--transport", "nvshmem"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    # nvshmem transport x overlapped weight_sync, join=dispatch ONLY: the ws
    # NCCL work is absorbed by the main stream before the first team-barrier
    # kernel launches, so no NVSHMEM team collision. DO NOT create a
    # nvshmem + "--ws_join gemm" variant: that overlaps ws NCCL kernels with
    # team-barrier kernels on multiplexed hardware channels — NR-02 Class-B
    # surface, untested by design.
    "ultraep_nvshmem_overlap": dict(
        comm_pattern="ultraep_quota_a2av",
        driver="ultraep",
        test_args=["--transport", "nvshmem", "--overlap_ws"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    # EPLB static predicted-load placement (vendored deepseek-ai/EPLB @
    # d52c72d under test/python/moe_ag_scatter/eplb_oracle/; plan mapping in
    # python/flux/testing/eplb_semantics.py, driver test_moe_eplb_traffic.py).
    # ONE placement per cell — full re-placement, masters move too, global
    # policy (DeepSeek's decode deployment), same nlp = G/W + 2 slot budget
    # as ultraep — computed from the FULL trace pool histogram
    # (<mid>.eplb_load.json, generated by the runner from the cell's exact
    # pools; the oracle-ceiling prediction). Per iteration: NO solver, NO
    # weight movement — prefetch reads ~0 and the one-time placement is
    # book-kept as eplb_weight_place_bytes/_ms_oneshot. plan_comm (the [W,G]
    # loads all-gather) is KEPT per iteration: recv splits still need the
    # counts exchange; never claim it zero. Data plane identical to
    # ultraep_nvshmem (staged pack -> one-sided All2AllSingle -> place, no
    # dedup, unpadded per-physical-expert GEMM segments; the residual
    # imbalance of the static placement on each batch IS the measurement).
    # NCCL remains only for plan_comm and the one-time setup P2P. Sweep sets
    # NVSHMEM_SYMMETRIC_SIZE via eplb_sym_size (row-sum bound — the ultraep
    # domain bound is unsafe under global re-homing). conn=8 pin inherited
    # from the EP-arm family A/B.
    # ---- campaign-2 CANONICAL eplb arms (2026-08-20, planner v2a) ----
    # FusedEpDispatch wire: DeepEP-lineage fused dispatch — NO planning
    # step (the per-iteration plan is one [S,K] dst_phys tensor), the
    # counts exchange rides IN-LAUNCH (comm_ms, zero host collectives),
    # exact deterministic offsets, per-(slot,src) arrival signals, l01
    # combine via the recorded per-row header handle. Replica selection is
    # sender-local (NO pre-dispatch exchange; plan_comm records ~0 and
    # eplb_plan_comm_bytes=0): default local_spread = per-source equal
    # split == token round-robin COUNTS (SGLang dynamic-dispatch analog;
    # count-equivalent, not token-identity — the interleave permutes
    # within blocks); lstatic twin = src mod C (SGLang static-map/D6
    # class). The old staged arms below are RETIRED from specs (kept for
    # history; never quote against fused arms across the binary boundary
    # FLUX_FUSED_EP_DISPATCH_TAG — SCHEMA rule 4/5).
    "eplb_fused": dict(
        comm_pattern="eplb_fused_ep",
        driver="eplb",
        test_args=["--transport", "fused",
                   "--replica_select", "local_spread"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_FUSED_EP_DISPATCH_TAG"],
    ),
    "eplb_fused_l01": dict(
        comm_pattern="eplb_fused_ep",
        driver="eplb",
        layer="l01",
        test_args=["--transport", "fused",
                   "--replica_select", "local_spread",
                   "--layers", "l01"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_FUSED_EP_DISPATCH_TAG"],
    ),
    "eplb_fused_lstatic": dict(
        comm_pattern="eplb_fused_ep",
        driver="eplb",
        test_args=["--transport", "fused",
                   "--replica_select", "local_static"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_FUSED_EP_DISPATCH_TAG"],
    ),
    # EPLB full journey (2026-08-20): dispatch + GEMM0 + GELU + GEMM1 +
    # the staged combine mirror (reverse a2av on the same All2AllSingle
    # pair, swapped splits; deterministic comb_dst_slot home reduce). No
    # dedup anywhere (direct wire both directions — the authentic
    # DeepEP-LL/decode transport class). rule-5 accounting: per-iteration
    # GPU planner (EplbIterPlanner), plan_ms column.
    # RETIRED from specs 2026-08-20 (campaign 2: fused is canonical).
    "eplb_l01": dict(
        comm_pattern="eplb_static_a2av",
        driver="eplb",
        layer="l01",
        test_args=["--transport", "nvshmem", "--layers", "l01"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    "eplb": dict(
        comm_pattern="eplb_static_a2av",  # cells.csv label only, never a CLI flag
        driver="eplb",
        test_args=["--transport", "nvshmem"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    # EPIC baseline (flux/EPIC.pdf, SIGCOMM'26 Alibaba; semantics in
    # python/flux/testing/epic_semantics.py, driver test_moe_epic_traffic.py,
    # invariants test_epic_planner.py). Faithful launch-granularity port:
    # §5.2 PEO (m expert groups pipelined dispatch -> un-overlapped
    # GemmGroupedV2 on two streams; dispatch/compute staging = stream
    # in-order; NO flux GEMM-overlap machinery), §4.2 placement with
    # replication (redundancy greedy + GPU greedy + NIC-stage greedy, pool
    # oracle via the same .eplb_load.json sidecar as the eplb arm), §4.3
    # dynamic intra-host migration (_mig arms; per-step decision subsumed
    # by plan_comm, tau-gated swaps converge in warmup under the static
    # per-cell routing — steady-state decision cost is the measurement).
    # Transport = Mode-1/DeepEP-default analog: staged per-entry wire, NO
    # dedup (One row per (token, instance); dup counterfactuals emitted as
    # epic_dup_stats). Replica rule: src mod C (recorded assumption).
    # Heap via epic_sym_size (eplb row-sum bound x split headroom).
    # epic_m1 = no-overlap anchor (D9); epic_m1_place_none = fixed-placement
    # control. conn=8 pin inherited from the EP-arm family A/B.
    "epic_m1": dict(
        comm_pattern="epic_peo_a2av",  # cells.csv label only, never a CLI flag
        driver="epic",
        test_args=["--transport", "nvshmem", "--placement", "epic",
                   "--groups", "1"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    "epic_m2": dict(
        comm_pattern="epic_peo_a2av",
        driver="epic",
        test_args=["--transport", "nvshmem", "--placement", "epic",
                   "--groups", "2"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    "epic_m4": dict(
        comm_pattern="epic_peo_a2av",
        driver="epic",
        test_args=["--transport", "nvshmem", "--placement", "epic",
                   "--groups", "4"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    "epic_m2_mig": dict(
        comm_pattern="epic_peo_a2av",
        driver="epic",
        test_args=["--transport", "nvshmem", "--placement", "epic",
                   "--groups", "2", "--migration", "on"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    "epic_m4_mig": dict(
        comm_pattern="epic_peo_a2av",
        driver="epic",
        test_args=["--transport", "nvshmem", "--placement", "epic",
                   "--groups", "4", "--migration", "on"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    "epic_m1_place_none": dict(
        comm_pattern="epic_peo_a2av",
        driver="epic",
        test_args=["--transport", "nvshmem", "--placement", "none",
                   "--groups", "1"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    # EPIC v2 arms — full journey (l01: dispatch -> GEMM0 -> GELU -> GEMM1 ->
    # per-group combine -> terminal Sum) and the Mode-2 hier_compress
    # transport (dispatch via the additive dispatch_only binding, combine via
    # per-group TopkReduceScatterOp; requires a post-S2 binary — the
    # FLUX_A2AV_DISPATCH_ONLY_TAG probe string gates stale builds into
    # skipped_capability). hc arms default to the faithful PXN identity
    # relay; migration stays on the direct transport (risk containment).
    "epic_l01_m1": dict(
        comm_pattern="epic_peo_a2av", driver="epic", layer="l01",
        test_args=["--transport", "nvshmem", "--placement", "epic",
                   "--groups", "1", "--layers", "l01"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"}, requires=[],
    ),
    "epic_l01_m2": dict(
        comm_pattern="epic_peo_a2av", driver="epic", layer="l01",
        test_args=["--transport", "nvshmem", "--placement", "epic",
                   "--groups", "2", "--layers", "l01"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"}, requires=[],
    ),
    "epic_l01_m4": dict(
        comm_pattern="epic_peo_a2av", driver="epic", layer="l01",
        test_args=["--transport", "nvshmem", "--placement", "epic",
                   "--groups", "4", "--layers", "l01"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"}, requires=[],
    ),
    "epic_l01_m2_mig": dict(
        comm_pattern="epic_peo_a2av", driver="epic", layer="l01",
        test_args=["--transport", "nvshmem", "--placement", "epic",
                   "--groups", "2", "--layers", "l01", "--migration", "on"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"}, requires=[],
    ),
    "epic_hc_m1": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement", "epic",
                   "--groups", "1"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "epic_hc_m2": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement", "epic",
                   "--groups", "2"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "epic_hc_m4": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement", "epic",
                   "--groups", "4"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    # §4.3 migration on the Mode-2 wire: _mig = host-NCCL weight exchange
    # (the launch-granularity port; exchange cost in migration_ms), _mig_fused
    # = the paper-faithful in-kernel swap fused as phase 0 of the group-0
    # dispatch launch (exchange cost inside e2e/disp0, reported as
    # swap_fused_ms; compare the twins on total_ms). Fused arms need a
    # binary carrying the swap kernel (FLUX_A2AV_INKERNEL_SWAP_TAG).
    "epic_hc_m1_mig": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement", "epic",
                   "--groups", "1", "--migration", "on"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    # 2026-08-20 campaign-2 v2b BOUNDARY: since the FLUX_A2AV_INWINDOW_
    # META_TAG binary, these mig_fused arms run --hc_meta inwindow by
    # driver default — the op derives splits/stable-scatter/sps/uc on
    # device INSIDE the dispatch bracket (planner_impl=fused_dispatch);
    # pre-v2 capsules of the SAME arm names used python-side per-iteration
    # metadata (planner_impl=torch_gpu). Never compare across the tag
    # boundary (SCHEMA rules 4/5).
    "epic_hc_m1_mig_fused": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement", "epic",
                   "--groups", "1", "--migration", "inkernel"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_INKERNEL_SWAP_TAG",
                  "FLUX_A2AV_INWINDOW_META_TAG"],
    ),
    "epic_l01_hc_m1_mig": dict(
        comm_pattern="epic_hc_a2av", driver="epic", layer="l01",
        test_args=["--transport", "hier_compress", "--placement", "epic",
                   "--groups", "1", "--layers", "l01", "--migration", "on"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "epic_l01_hc_m1_mig_fused": dict(
        comm_pattern="epic_hc_a2av", driver="epic", layer="l01",
        test_args=["--transport", "hier_compress", "--placement", "epic",
                   "--groups", "1", "--layers", "l01", "--migration",
                   "inkernel"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_INKERNEL_SWAP_TAG",
                  "FLUX_A2AV_INWINDOW_META_TAG"],
    ),
    "epic_l01_hc_m1": dict(
        comm_pattern="epic_hc_a2av", driver="epic", layer="l01",
        # 2026-08-24 canonicalization (user decision): staged l1 (zero
        # comm/GEMM overlap) forfeits nothing at ns1 -> one round of puts
        test_args=["--transport", "hier_compress", "--placement", "epic",
                   "--groups", "1", "--layers", "l01", "--l1_n_split", "1"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "epic_l01_hc_m2": dict(
        comm_pattern="epic_hc_a2av", driver="epic", layer="l01",
        test_args=["--transport", "hier_compress", "--placement", "epic",
                   "--groups", "2", "--layers", "l01"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "epic_l01_hc_m4": dict(
        comm_pattern="epic_hc_a2av", driver="epic", layer="l01",
        test_args=["--transport", "hier_compress", "--placement", "epic",
                   "--groups", "4", "--layers", "l01"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    # ----- nodeaware placement + LocCap router (8.19.theory campaign) -----
    # PLACE-lambda co-occurrence placement from the <mid>.placement_*.json
    # sidecar (sweeps/predict_placement.py — the runner generates it per
    # cell) + per-token replica selection: --router d6 = the EPIC baseline
    # rule on the nodeaware placement; --router loccap eps = tiered
    # locality under per-rank caps (1+eps)*S*K, minimizing token-node
    # incidence (= the hc wire rows). _rankconc = equal-slot home-node-
    # concentrated replication, the P4 coverage-vs-concentration ablation.
    # _lbu = the Tier-B fused lb_union dispatch wire over the same virtual
    # slot space (replicas x lb_union integration; EARLY_LAUNCH +
    # FUSED_STAGE2 pinned inside enable_hier_compress). conn32 clones exist
    # for the fan-out-elimination ladder — never compare across conn.
    "epic_hc_m1_place_none": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement", "none",
                   "--groups", "1"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "epic_hc_m1_na": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement",
                   "nodeaware", "--groups", "1", "--router", "d6"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "epic_hc_m1_na_rankconc": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        placement_mode="rankconc",
        test_args=["--transport", "hier_compress", "--placement",
                   "nodeaware", "--groups", "1", "--router", "d6"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "epic_hc_m1_na_lc00": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement",
                   "nodeaware", "--groups", "1", "--router", "loccap",
                   "--eps", "0.0"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "epic_hc_m1_na_es": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement",
                   "nodeaware", "--groups", "1", "--router", "evensplit"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "epic_hc_m1_na_lc0625": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement",
                   "nodeaware", "--groups", "1", "--router", "loccap",
                   "--eps", "0.0625"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "epic_hc_m1_na_lc125": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement",
                   "nodeaware", "--groups", "1", "--router", "loccap",
                   "--eps", "0.125"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "epic_hc_m1_na_lc25": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement",
                   "nodeaware", "--groups", "1", "--router", "loccap",
                   "--eps", "0.25"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "epic_hc_m1_na_lc50": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement",
                   "nodeaware", "--groups", "1", "--router", "loccap",
                   "--eps", "0.5"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "epic_hc_m1_na_lcinf": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement",
                   "nodeaware", "--groups", "1", "--router", "loccap",
                   "--eps", "inf"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "epic_l01_hc_m1_na": dict(
        comm_pattern="epic_hc_a2av", driver="epic", layer="l01",
        test_args=["--transport", "hier_compress", "--placement",
                   "nodeaware", "--groups", "1", "--layers", "l01",
                   "--router", "d6"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "epic_l01_hc_m1_na_lc25": dict(
        comm_pattern="epic_hc_a2av", driver="epic", layer="l01",
        test_args=["--transport", "hier_compress", "--placement",
                   "nodeaware", "--groups", "1", "--layers", "l01",
                   "--router", "loccap", "--eps", "0.25"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "epic_l01_hc_m1_na_lcinf": dict(
        comm_pattern="epic_hc_a2av", driver="epic", layer="l01",
        test_args=["--transport", "hier_compress", "--placement",
                   "nodeaware", "--groups", "1", "--layers", "l01",
                   "--router", "loccap", "--eps", "inf"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    # ----- pll_* — PLACE-lambda/LocCap GPU port (8.21.place, OURS) -------
    # These are OUR optimization arms (PLACE-lambda placement + LocCap
    # routing, flux.testing.placelambda_gpu), distinct from the EPIC
    # baselines above — the epic driver is only the harness. All rule-5:
    # the loccap_gpu router re-derives per iteration ON DEVICE inside the
    # timed bracket (plan_ms; timing_accounting=per_iter_gpu). Placement =
    # the batch-observed device solver; the runner provisions the
    # placement_placelambda_gpu sidecar and the driver hard-asserts its
    # on-device solve equals it (cross-device oracle). loccap_gpu is a NEW
    # arm — a bounded-round approximation, never comparable to the exact
    # python loccap arms (epic_hc_m1_na_lc*). eps=0.0625 is the working
    # default from the confirmed flat basin [0, 0.125] — NOT canonicalized.
    # _dyn = the placement-ablation toggle ON: per-iteration timed solve +
    # move diff + trigger (place_ms); static arms solve once untimed
    # (place_solver_ms fact, ideal-stale semantics).
    "pll_hc_d6": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement",
                   "placelambda_gpu", "--groups", "1", "--router", "d6"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "pll_hc_lcg0625": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement",
                   "placelambda_gpu", "--groups", "1", "--router",
                   "loccap_gpu", "--eps", "0.0625"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "pll_hc_lcg0625_dyn": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement",
                   "placelambda_gpu", "--groups", "1", "--router",
                   "loccap_gpu", "--eps", "0.0625",
                   "--place_dynamic", "dynamic"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    # THE KERNEL ARM: relaxed sender-local fused router
    # (flux.placelambda_route_sl) — per-iteration kernel + phys-row
    # allgather timed in plan_ms; routing varies legitimately run-to-run;
    # verified by invariants + provable table bounds (auto f_cap) + a
    # final deterministic correctness iteration. Never compare its
    # routings bitwise against the loccap_gpu deterministic arm.
    "pll_hc_sl0625": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement",
                   "placelambda_gpu", "--groups", "1", "--router",
                   "loccap_sl", "--eps", "0.0625"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG",
                  "FLUX_PLACELAMBDA_ROUTE_SL_TAG"],
    ),
    # trigger probe on a STALE resident (fixed placement + dynamic
    # decision): measures the decision apparatus where re-placement
    # genuinely pays — the trigger must fire here and stay silent on the
    # _dyn arm above (resident == fresh)
    "pll_hc_dynprobe_stale": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement", "none",
                   "--groups", "1", "--router", "d6",
                   "--place_dynamic", "dynamic"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    # ----- pllf_* — PLACE-lambda FAST (8.22.placefast, OURS) ---------
    # Batched bounded-pass zero-D2H placement solver
    # (flux.testing.placelambda_fast): affinity seed + balance repair +
    # batched FM + batched replication; Stage C (rank assignment) off the
    # per-iteration path. NEW arm — placements never compared bitwise
    # against placelambda_gpu cells (same never-mix rule as loccap_gpu vs
    # exact loccap). _dyn = per-iteration warm-seeded solve + tensor
    # decision, CUDA-graph-captured (FLUX_PLACE_FAST_GRAPH=1), verdicts
    # ring-buffered (zero D2H in-window).
    "pllf_hc_lcg0625": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement",
                   "placelambda_fast", "--groups", "1", "--router",
                   "loccap_gpu", "--eps", "0.0625"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "pllf_hc_lcg0625_dyn": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement",
                   "placelambda_fast", "--groups", "1", "--router",
                   "loccap_gpu", "--eps", "0.0625",
                   "--place_dynamic", "dynamic"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "pllf_l01_hc_lcg0625": dict(
        comm_pattern="epic_hc_a2av", driver="epic", layer="l01",
        test_args=["--transport", "hier_compress", "--placement",
                   "placelambda_fast", "--groups", "1", "--layers", "l01",
                   "--router", "loccap_gpu", "--eps", "0.0625"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "pllf_l01_hc_lcg0625_dyn": dict(
        comm_pattern="epic_hc_a2av", driver="epic", layer="l01",
        test_args=["--transport", "hier_compress", "--placement",
                   "placelambda_fast", "--groups", "1", "--layers", "l01",
                   "--router", "loccap_gpu", "--eps", "0.0625",
                   "--place_dynamic", "dynamic"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    # l01 twins of the EXACT pll arms (same-capsule anchors for the fast
    # solver: static quality parity + the dynamic place_ms gap)
    "pll_l01_hc_lcg0625": dict(
        comm_pattern="epic_hc_a2av", driver="epic", layer="l01",
        test_args=["--transport", "hier_compress", "--placement",
                   "placelambda_gpu", "--groups", "1", "--layers", "l01",
                   "--router", "loccap_gpu", "--eps", "0.0625"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "pll_l01_hc_lcg0625_dyn": dict(
        comm_pattern="epic_hc_a2av", driver="epic", layer="l01",
        test_args=["--transport", "hier_compress", "--placement",
                   "placelambda_gpu", "--groups", "1", "--layers", "l01",
                   "--router", "loccap_gpu", "--eps", "0.0625",
                   "--place_dynamic", "dynamic"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    # CANONICAL LLC l01 arm (2026-08-23, user-directed; the quotable
    # "PLL/LLC (ours)" baseline for scenario-1 data campaigns — supersedes
    # pllf_l01_hc_lcg0625's loccap_gpu router, which stays for capsule
    # history): the settled-table stack of handoff 13 — PLACE-lambda FAST
    # placement solved STATICALLY on the previous-window oracle rows
    # (sweep.py passes --oracle_routing_file on dslots trace cells),
    # LocCap sender-local fused-kernel router (loccap_sl, eps 0.0625)
    # with the fast tail (FLUX_PLL_FAST_TAIL default 1) and the GRAPHED
    # tail (FLUX_PLL_TAIL_GRAPH=1 — the loccap tail is the only graphed
    # plan lane, as in the settled tables), Slipstream dispatch +
    # capacity-mode hcc combine (binary defaults post rule-11 flip,
    # probe-gated by the DEFAULT tags).
    "llc_l01_s1": dict(
        comm_pattern="epic_hc_a2av", driver="epic", layer="l01",
        test_args=["--transport", "hier_compress", "--placement",
                   "placelambda_fast", "--groups", "1", "--layers", "l01",
                   "--router", "loccap_sl", "--eps", "0.0625",
                   # 2026-08-24 canonicalization: staged l1 -> ns1
                   "--l1_n_split", "1"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8",
             "FLUX_PLL_TAIL_GRAPH": "1"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG",
                  "FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT16_TAG",
                  "FLUX_A2AV_RS_CTA_1086_TAG"],
    ),
    "epic_hc_m1_na_lbu": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement",
                   "nodeaware", "--groups", "1", "--router", "d6",
                   "--hc_wire", "lb_union"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG", "FLUX_A2AV_LB_UNION"],
    ),
    "epic_hc_m1_na_lbu_lc25": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement",
                   "nodeaware", "--groups", "1", "--router", "loccap",
                   "--eps", "0.25", "--hc_wire", "lb_union"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG", "FLUX_A2AV_LB_UNION"],
    ),
    "epic_hc_m1_na_lbu_lcinf": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement",
                   "nodeaware", "--groups", "1", "--router", "loccap",
                   "--eps", "inf", "--hc_wire", "lb_union"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG", "FLUX_A2AV_LB_UNION"],
    ),
    "epic_l01_hc_m1_na_lbu_lc25": dict(
        comm_pattern="epic_hc_a2av", driver="epic", layer="l01",
        test_args=["--transport", "hier_compress", "--placement",
                   "nodeaware", "--groups", "1", "--layers", "l01",
                   "--router", "loccap", "--eps", "0.25",
                   "--hc_wire", "lb_union"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG", "FLUX_A2AV_LB_UNION"],
    ),
    # conn-ladder clones (Capsule C): identical to their base arms except
    # the conn pin. Relay-identity arms never enable EARLY_LAUNCH (its
    # default is LB_UNION-conditioned); lbu arms pin it inside
    # enable_hier_compress — both stay single-axis across conn.
    "epic_hc_m1_conn32": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement", "epic",
                   "--groups", "1"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "32"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "epic_hc_m1_na_lc25_conn32": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement",
                   "nodeaware", "--groups", "1", "--router", "loccap",
                   "--eps", "0.25"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "32"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG"],
    ),
    "epic_hc_m1_na_lbu_lc25_conn32": dict(
        comm_pattern="epic_hc_a2av", driver="epic",
        test_args=["--transport", "hier_compress", "--placement",
                   "nodeaware", "--groups", "1", "--router", "loccap",
                   "--eps", "0.25", "--hc_wire", "lb_union"],
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "32"},
        requires=["FLUX_A2AV_DISPATCH_ONLY_TAG", "FLUX_A2AV_LB_UNION"],
    ),
    # dense baseline and raw a2av modes
    "allgather": dict(comm_pattern="allgather", env={}, requires=[]),
    # a2av family pins CUDA_DEVICE_MAX_CONNECTIONS=8 (2026-08-01 A/B: -2..-8%
    # e2e across all four variants at b8 remotefrac, correctness green; the
    # upstream conn=1 discipline protected spin-wait designs whose comm needed
    # SM kernels, which the a2av paths avoid post-launch). Cells before this
    # change ran at conn=1 — env_json in cells.csv audits which is which; do
    # not compare across the boundary.
    "a2av": dict(comm_pattern="a2av", env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"}, requires=[]),
    "a2av_ring": dict(
        comm_pattern="a2av_ring", env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"}, requires=[]
    ),
    # hierarchical dispatch family
    "hier": dict(comm_pattern="a2av_hier", env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"}, requires=[]),
    # token-dedup wire semantics; balanced inter-node relay is the default
    "hier_compress": dict(
        comm_pattern="a2av_hier_compress",
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_RELAY_IDENTITY"],  # any compress-capable build has this
    ),
    # compress with the fixed same-local-rank relay (design §11 identity wire)
    "hier_compress_identity": dict(
        comm_pattern="a2av_hier_compress",
        env={"FLUX_A2AV_RELAY_IDENTITY": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_RELAY_IDENTITY"],
    ),
    # gateway union broadcast (implies identity wire)
    "hier_compress_union": dict(
        comm_pattern="a2av_hier_compress",
        env={"FLUX_A2AV_UNION_BCAST": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_UNION_BCAST"],
    ),
    # balanced chunked wire + union-broadcast gateway: per-round wire bytes are
    # ceil(total/L) per rank (like hier_compress) but the gateway forwards its
    # whole staged window as pure-CE puts (like union) — no index build, no
    # gathers. Tier B (2026-08-04): forwards land per (gateway, round) WINDOW
    # with ring-rotated destination order; tiles unblock per landed window.
    # Must be set identically on all ranks (changes wire + recv layout).
    # CANONICAL BOUNDARY NOTE (2026-08-16, campaign canonicalization):
    # binaries built on/after this date default FUSED_STAGE2 and EARLY_LAUNCH
    # ON under LB_UNION=1 (E only when conn>1 — satisfied by the family
    # conn=8 pin), so this base arm now measures F+E ON. Absent env keys in
    # env_json therefore mean DIFFERENT configurations on either side of the
    # boundary — never byte-compare env_json across it; identify the binary by
    # the manifest flux_libs sha (git_sha is not a build identity). The
    # pre-flip meaning of this arm = today's
    # hier_compress_lb_union_nofused_noearly.
    "hier_compress_lb_union": dict(
        comm_pattern="a2av_hier_compress",
        env={"FLUX_A2AV_LB_UNION": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_LB_UNION"],
    ),
    # CLOSED — LOSER (verdict 2026-08-16, case closed by user directive):
    # N=FANOUT eager per-round gateway forwards LOSE +0.05..+0.6 ms on real
    # 4n trace routing under three-run sign agreement (capsules
    # 7ff7098d/1451c017/78f371c6 + reversed twins; handoff 07 §2). The knob
    # and this arm are retained as opt-in ABLATION ONLY — the knob default
    # stays OFF, and no future run may treat FANOUT as an open experiment or
    # expect a win. Mechanism (2026-08-07 design): each round's node_sig wait
    # + window puts on its own stream instead of the shipped ascending-round
    # single-stream order.
    "hier_compress_lb_union_eager": dict(
        comm_pattern="a2av_hier_compress",
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FANOUT": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8",
        },
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_FANOUT"],
    ),
    # Fused stage-2 consumer-build arms (2026-08-05): identical wire/gateway
    # semantics to their base variants, but the ATen key/argsort/index_select
    # chain + Tier B gating searchsorted are replaced by the fused sort_util
    # kernels. CANONICALIZED 2026-08-16: the binary now defaults FUSED_STAGE2
    # ON under LB_UNION (WINNER, -0.2..-0.7 ms, three-run sign agreement) —
    # hier_compress_lb_union_fused is an explicit pin of that default
    # (== base on post-flip binaries; kept for stated-both-sides A/B and
    # historical name continuity). The union_bcast variant keeps its opt-in
    # meaning (no default flip outside LB_UNION).
    "hier_compress_union_fused": dict(
        comm_pattern="a2av_hier_compress",
        env={
            "FLUX_A2AV_UNION_BCAST": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8",
        },
        requires=["FLUX_A2AV_UNION_BCAST", "FLUX_A2AV_FUSED_STAGE2"],
    ),
    "hier_compress_lb_union_fused": dict(
        comm_pattern="a2av_hier_compress",
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8",
        },
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_FUSED_STAGE2"],
    ),
    # 2^3 FACTORIAL COMPLETION on the lb_union base (2026-08-16 comm-only
    # campaign): F=FUSED_STAGE2, N=FANOUT (eager forwards), E=EARLY_LAUNCH.
    # Base/+F/+N are the three entries above; the five below complete the
    # cube so every pairwise/3-way interaction is attributable in-capsule.
    # Canonical suffix order: fused < eager < early (existing names keep
    # their historical spelling).
    # Static-guard survey (gemm_grouped_v2_ag_scatter.cc, 2026-08-16): the
    # only combination FLUX_CHECKs are E+PACK_OVERLAP (:687), N-requires-
    # LB_UNION (:677), and E-on-compress-needs-conn>1 (:689-697, met by the
    # family conn=8 pin) — nothing blocks F+N, F+E, N+E, or F+N+E.
    # DYNAMIC CAVEAT: E's deferred cp-stream wire replay has never executed
    # together with N's per-round fanout streams — a correctness-ON smoke
    # gates the *eager_early arms before any perf capsule.
    # E is a clean-mode configuration (unlike BLOCKING_WIRE); quotable
    # per-arm, per the SKILL.md "its own configuration" rule.
    # VERDICTS + CANONICALIZATION (2026-08-16, handoff 07 §2, three-run sign
    # agreement): F WIN, E WIN, N LOSS — the binary now defaults F+E ON under
    # LB_UNION (see the base arm's boundary note), so on post-flip binaries
    # the explicit "+F"/"+E" pins equal the base, and the cube's live
    # ablation axis is the explicit-OFF arms further below
    # (_nofused/_noearly/_nofused_noearly). All FANOUT (N) corners are
    # CLOSED-LOSER ablations per the eager arm's note — retained, never
    # re-run as open experiments.
    "hier_compress_lb_union_early": dict(
        comm_pattern="a2av_hier_compress",
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8",
        },
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_EARLY_LAUNCH"],
    ),
    "hier_compress_lb_union_fused_eager": dict(
        comm_pattern="a2av_hier_compress",
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_FANOUT": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8",
        },
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_FUSED_STAGE2", "FLUX_A2AV_FANOUT"],
    ),
    "hier_compress_lb_union_fused_early": dict(
        comm_pattern="a2av_hier_compress",
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8",
        },
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_FUSED_STAGE2", "FLUX_A2AV_EARLY_LAUNCH"],
    ),
    "hier_compress_lb_union_eager_early": dict(
        comm_pattern="a2av_hier_compress",
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FANOUT": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8",
        },
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_FANOUT", "FLUX_A2AV_EARLY_LAUNCH"],
    ),
    "hier_compress_lb_union_fused_eager_early": dict(
        comm_pattern="a2av_hier_compress",
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_FANOUT": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8",
        },
        requires=[
            "FLUX_A2AV_LB_UNION",
            "FLUX_A2AV_FUSED_STAGE2",
            "FLUX_A2AV_FANOUT",
            "FLUX_A2AV_EARLY_LAUNCH",
        ],
    ),
    # EXPLICIT-OFF ABLATION ARMS (post-canonicalization cube, 2026-08-16): on
    # binaries with the F+E default flip these pin the winners OFF to measure
    # their contribution against the canonical base inside one capsule.
    # _nofused_noearly reproduces the pre-flip base configuration exactly.
    "hier_compress_lb_union_nofused": dict(
        comm_pattern="a2av_hier_compress",
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "0",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8",
        },
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_FUSED_STAGE2"],
    ),
    "hier_compress_lb_union_noearly": dict(
        comm_pattern="a2av_hier_compress",
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "0",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8",
        },
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_EARLY_LAUNCH"],
    ),
    "hier_compress_lb_union_nofused_noearly": dict(
        comm_pattern="a2av_hier_compress",
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "0",
            "FLUX_A2AV_EARLY_LAUNCH": "0",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8",
        },
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_FUSED_STAGE2", "FLUX_A2AV_EARLY_LAUNCH"],
    ),
    # union broadcast + source-side pack overlap (dedicated stream,
    # double-buffered send); MAX_CONNECTIONS=2 lets the pack stream actually
    # run concurrently with compute.
    # CAVEAT (audited 2026-08-04): conn=2 was chosen 2026-07-29 and has NOT
    # been re-validated since. All 15 pack cells in the capsule set are conn=2
    # -- there is no committed A/B -- and all predate both the family conn=8
    # pin and isolated mode. Re-derive before quoting pack numbers.
    # See docs/handoff/03_insight_ledger.md NR-11.
    "hier_compress_pack": dict(
        comm_pattern="a2av_hier_compress",
        env={
            "FLUX_A2AV_UNION_BCAST": "1",
            "FLUX_A2AV_PACK_OVERLAP": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "2",
        },
        requires=["FLUX_A2AV_UNION_BCAST", "FLUX_A2AV_PACK_OVERLAP"],
    ),
    # ------------------------------------------------------------------
    # LAYER1 (gather-rs combine) variants, driver="gather_rs" (2026-08-16
    # layer-axis campaign). comm_pattern here selects the
    # GemmGroupedV2GatherRSOp ctor booleans via the l1 bench CLI; env knobs
    # are the FLUX_A2AV_RS_* family (sized per-cell by exact_rs_scale_knobs).
    # FLUX_A2AV_RS_MAX_SEND_ROWS in `requires` is the "build has the merged
    # layer1 a2av op" probe on every arm (a pre-merge .so would silently run
    # stock dense). Flux l1 cells carry a timing_mode axis (isolated =
    # in-forward index build; amortized = layer0-inherited indices — the
    # combined-pass proxy); never compare across timing_mode.
    # NOTE: a2av_hier_compress silently degrades to a2av_hier at nnodes == 1
    # (gather_rs.cc:409-413) — single-node l1_compress* cells measure hier.
    # EAGER (arrival-order persistent reduce) is orthogonal to hier/compress.
    # CLOSED — LOSER (verdict 2026-08-16, case closed by user directive):
    # FLUX_A2AV_RS_EAGER regresses standalone at BOTH scales (+15-35% W=8,
    # +2-11% W=16 vs legacy hier on trace) AND is the entire l01 composition
    # penalty (+18% identity violation at b8, ablation-attributed — handoff
    # 07 §4). Kernel + knob retained; the *_eager arms below are opt-in
    # ABLATIONS ONLY — treat eager as losing unless a future fwd+rev
    # sign-agreeing capsule overturns this note explicitly.
    "l1_dense": dict(
        comm_pattern="dense",
        driver="gather_rs",
        layer="l1",
        env={},
        requires=["FLUX_A2AV_RS_MAX_SEND_ROWS"],
    ),
    "l1_hier": dict(
        comm_pattern="a2av_hier",
        driver="gather_rs",
        layer="l1",
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_RS_MAX_SEND_ROWS"],
    ),
    "l1_hier_eager": dict(
        comm_pattern="a2av_hier",
        driver="gather_rs",
        layer="l1",
        env={"FLUX_A2AV_RS_EAGER": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_RS_MAX_SEND_ROWS", "FLUX_A2AV_RS_EAGER"],
    ),
    "l1_compress": dict(
        comm_pattern="a2av_hier_compress",
        driver="gather_rs",
        layer="l1",
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[
            "FLUX_A2AV_RS_MAX_SEND_ROWS",
            "FLUX_A2AV_RS_MAX_CONV_ROWS",
            "FLUX_A2AV_RS_MAX_WIRE_ROWS",
        ],
    ),
    "l1_compress_eager": dict(
        comm_pattern="a2av_hier_compress",
        driver="gather_rs",
        layer="l1",
        env={"FLUX_A2AV_RS_EAGER": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[
            "FLUX_A2AV_RS_MAX_SEND_ROWS",
            "FLUX_A2AV_RS_MAX_CONV_ROWS",
            "FLUX_A2AV_RS_MAX_WIRE_ROWS",
            "FLUX_A2AV_RS_EAGER",
        ],
    ),
    # MOONEP VIRTUAL-SPACE LAYER1 (2026-08-17): the MoonEP plan's combine
    # through the normal fused gather-rs op over R*(epn+B) virtual experts —
    # replicated (slot) rows combine LOCALLY, so cross-rank combine copies =
    # dispatch copies minus replication (the driver prints both). Weights
    # land in slots via untimed setup here; the e2e moonep drivers move them
    # with the weight ops. The DRIVER sets the exact FLUX_A2AV_RS_MAX_*
    # knobs from the plan (setdefault; parity test in test_knob_demands.py);
    # the runner only sizes the heap (moonep_l1_sym_size upper bounds) —
    # hence no RS-knob `requires`. l1_pattern doubles as the driver's
    # --comm_pattern and the heap-sizing branch selector.
    "moonep_l1_hier": dict(
        comm_pattern="moonep_a2av_hier",  # cells.csv label
        driver="moonep_l1",
        layer="l1",
        l1_pattern="a2av_hier",
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    "moonep_l1_compress": dict(
        comm_pattern="moonep_a2av_hier_compress",  # cells.csv label
        driver="moonep_l1",
        layer="l1",
        l1_pattern="a2av_hier_compress",
        env={"CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[],
    ),
    # FAST BvN alltoallv + un-overlapped GemmGroupedV2, layer1 direction
    # (compute -> communicate -> topk-sum; wire matrix = dispatch transpose).
    # Same constraints as the layer0 `fast` arm: e2e mode only (host-blocking
    # per iteration, so its e2e IS isolated semantics — see SCHEMA.md),
    # >= 2 nodes, libflash built per checkout, no real-routing consumption,
    # no timing_mode axis (its index metadata is untimed setup; the BvN
    # schedule recompute stays in-window per the one-shot rule).
    "l1_fast": dict(
        comm_pattern="fast_bvn_a2av_rs",  # cells.csv label only, never a CLI flag
        driver="fast_gather_rs",
        layer="l1",
        env={},
        requires=[],
        requires_file="3rdparty/FAST/nvidia/libflash.so",
    ),
    # ------------------------------------------------------------------
    # COMBINED layer0+1 continuous-pass arms, driver="l01"
    # (test/python/moe_combined/test_moe_l0l1_traffic.py): one timed window
    # per isolated iteration = layer0 forward (routing/schedule computed
    # once, in-window) -> GELU -> layer1 forward on inherited inverse
    # indices (python builders OUTSIDE the window, one-shot cost reported
    # as l1_index_build_ms — decided 2026-08-16). No timing_mode axis, no
    # phases cells. `l1_pattern` is consumed by the runner's heap sizing
    # (NVSHMEM_SYMMETRIC_SIZE = SUM of both layers' demands — two ops, one
    # heap). Validation identity per capsule: e2e(l01) ~= e2e(l0 isolated)
    # + act_ms + e2e(l1 tmamo) — sweeps/check_l01_identity.py.
    # l01_fast landed 2026-08-21 (rule-5 conversion): the credit-reset
    # question resolved via TWO flash_comm_t instances (one per wire
    # direction, resets outside the window; vendored refcount patch
    # scripts/fast_two_instance.patch). l01_lbunion_compress_eager
    # will NOT be added — the l1 eager verdict closed as a LOSS (2026-08-16;
    # see the layer1 CLOSED note).
    # FAST+FAST combined (2026-08-21): trace-driven dispatch alltoallv ->
    # grouped GEMM0 -> GELU -> grouped GEMM1 -> combine alltoallv (transposed
    # matrix) -> home topk-reduce; the authoritative unfused paper baseline on
    # REAL routing. driver="l01_fast": launch_fast.sh, e2e-only, >= 2 nodes,
    # heap = 2x fast_sym_size (two flash_comm_t instances), rule-5 in-window
    # allgather + derive_fast_l01_meta_gpu.
    "l01_fast": dict(
        comm_pattern="l01_fast_bvn_a2av",  # cells.csv label only
        driver="l01_fast",
        layer="l01",
        test_args=["--impl", "fast"],
        env={},
        requires=[],
        requires_file="3rdparty/FAST/nvidia/libflash.so",
        l1_pattern="fast",
    ),
    "l01_torch": dict(
        comm_pattern="l01_torch_unfused",  # cells.csv label only
        driver="l01",
        layer="l01",
        test_args=["--impl", "torch"],
        env={},
        requires=[],
        l1_pattern="dense",
    ),
    "l01_allgather_dense": dict(
        comm_pattern="l01_allgather_dense",  # cells.csv label only
        driver="l01",
        layer="l01",
        test_args=[
            "--impl", "flux",
            "--l0_comm_pattern", "allgather",
            "--l1_comm_pattern", "dense",
            # 2026-08-24 canonicalization (user decision): dense l1 ns2 —
            # NEEDS FLUX_RS_NSPLIT_512_TAG (K2 dense ns2 demoted to 7 before)
            "--n_split", "2",
        ],
        # FLUX_RS_BLOCKS canonicalized 3 -> 20 (2026-08-21): the dense combine
        # was CTA-starved at topk-16 (K3 b32 e2e 104 -> 60 ms over the {3..24}
        # knob sweep, knee ~20). Env flip = rule-4-style boundary; never
        # byte-compare env_json across it.
        env={"FLUX_RS_BLOCKS": "20"},
        requires=["FLUX_A2AV_RS_MAX_SEND_ROWS"],
        l1_pattern="dense",
    ),
    # CORRECTED best pairing (2026-08-16 ablation, 2n b8): l1 EAGER OFF —
    # the eager persistent reduce is both a standalone l1 loss (15-35% vs
    # legacy hier on trace) AND the entire l01 composition penalty (+18%
    # identity violation at b8; eager-off closes it to -1% and lands
    # 9.95 vs eager-pairing 14.4 ms). E(early-launch) exonerated (B-ablation
    # no-op). SUPERSEDED as the reference by l01_lbunion_compress (W=16
    # best-pairing A/B, 2026-08-16 — compress wins every budget); retained as
    # the standing A/B pairing arm.
    "l01_lbunion_hier": dict(
        comm_pattern="l01_lbunion_hier",  # cells.csv label only
        driver="l01",
        layer="l01",
        test_args=[
            "--impl", "flux",
            "--l0_comm_pattern", "a2av_hier_compress",
            "--l1_comm_pattern", "a2av_hier",
        ],
        # a2av combine CTA canonicalization 3/3/2 -> 6/6/4 (2026-08-22): same
        # starvation class as dense's FLUX_RS_BLOCKS — K3 b32 e2e 61.9 -> 51.0
        # over the knob ladder, knee at 6/6/4 (12/12/8 regresses: margin
        # starves the GEMMs). Env flip = never-byte-compare boundary.
        # a2av combine CTA canonicalization 3/3/2 -> 6/6/4 (2026-08-22): same
        # starvation class as dense's FLUX_RS_BLOCKS — K3 b32 e2e 61.9 -> 51.0
        # over the knob ladder, knee at 6/6/4 (12/12/8 regresses: margin
        # starves the GEMMs). Env flip = never-byte-compare boundary.
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "6", "FLUX_A2AV_RS_REDUCE_BLOCKS": "6", "FLUX_A2AV_RS_PRERED_BLOCKS": "4"},
        requires=[
            "FLUX_A2AV_LB_UNION",
            "FLUX_A2AV_FUSED_STAGE2",
            "FLUX_A2AV_EARLY_LAUNCH",
            "FLUX_A2AV_RS_MAX_SEND_ROWS",
        ],
        l1_pattern="a2av_hier",
    ),
    # REFERENCE COMBINED CONFIG (promoted 2026-08-16, item-2
    # canonicalization): the W=16 best-pairing A/B (capsules 9378fed5/397ac0fa,
    # 14/14 budgets, fwd+rev) has the compress pairing winning EVERY budget —
    # 10.6 ms at b8, -52% vs stock, -45..-52% across b2-b64. The l01 window
    # inherits compress's CSRs from layer0 (amortized semantics; SCHEMA.md),
    # so compress's isolated-mode in-forward CSR-build penalty does not apply
    # here. Keep the stories straight: STANDALONE l1 verdicts differ (hier
    # wins iso at small budgets — handoff 07 §3.1/§4.1); never quote this
    # combined win as a standalone-l1 recommendation.
    # Slipstream canonicalization (2026-08-23, M4): the RS split wire
    # (RS_WIRE_STREAMS=2) and CTA 10/8/6 are BINARY DEFAULTS now; the env
    # pins below are explicit pins of those defaults (spec self-
    # documentation, same pattern as the F+E pins). The DEFAULT-tag
    # requires make pre-flip binaries skipped_capability instead of
    # silently measuring 6/6/4 single-stream. NEVER byte-compare env_json
    # across the flip; pre-flip capsules stay valid via their recorded env.
    "l01_lbunion_compress": dict(
        comm_pattern="l01_lbunion_compress",  # cells.csv label only
        driver="l01",
        layer="l01",
        test_args=[
            "--impl", "flux",
            "--l0_comm_pattern", "a2av_hier_compress",
            "--l1_comm_pattern", "a2av_hier_compress",
            # 2026-08-24 canonicalization (user decision): l1 combine ns2
            # (bridges the split ladder interior; overrides the spec's
            # n_split_l1 by argparse last-wins) + 16 wire lanes
            "--n_split", "2",
        ],
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "FLUX_A2AV_RS_WIRE_STREAMS": "16",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "10", "FLUX_A2AV_RS_REDUCE_BLOCKS": "8", "FLUX_A2AV_RS_PRERED_BLOCKS": "6"},
        requires=[
            "FLUX_A2AV_LB_UNION",
            "FLUX_A2AV_FUSED_STAGE2",
            "FLUX_A2AV_EARLY_LAUNCH",
            "FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT16_TAG",
            "FLUX_A2AV_NSPLIT_HONOR_TAG",
            "FLUX_A2AV_RS_CTA_1086_TAG",
            "FLUX_A2AV_RS_MAX_SEND_ROWS",
            "FLUX_A2AV_RS_MAX_CONV_ROWS",
            "FLUX_A2AV_RS_MAX_WIRE_ROWS",
        ],
        l1_pattern="a2av_hier_compress",
    ),
    # Slipstream v2 (2026-08-24): M-split destination-wave combine. The l1
    # GEMM is decomposed into (ring wave of dest nodes, expert) ROW
    # sub-problems (n_split MUST be 1 — waves replace column splits) whose
    # cascade flags release the per-node pack->conv->prered->wire ladder
    # DURING the GEMM: put count stays at the ns1 minimum (NN-1 blocking
    # puts/rank) WITH M-axis pipelining — the structural resolution of
    # handoff 16's "n_split multiplies the proxy-bound put count" tension.
    # Eager arrival-order receiver reduce defaults ON under msplit. Layer0
    # dispatch identical to l01_slipstream. FLUX_A2AV_RS_WAVE_NODES (default
    # 1 = per-node waves) is the tile-quantization dial. Own never-mix
    # boundary: FLUX_A2AV_RS_MSPLIT_TAG (default-off knob — one binary
    # serves v1 and v2 arms).
    "l01_slipstream_v2": dict(
        comm_pattern="l01_slipstream_v2",  # cells.csv label only
        driver="l01",
        layer="l01",
        test_args=[
            "--impl", "flux",
            "--l0_comm_pattern", "a2av_hier_compress",
            "--l1_comm_pattern", "a2av_hier_compress",
            "--n_split", "1",  # msplit requires ns1 (argparse last-wins over spec)
        ],
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "FLUX_A2AV_RS_MSPLIT": "1",
            "FLUX_A2AV_RS_WIRE_STREAMS": "16",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "10", "FLUX_A2AV_RS_REDUCE_BLOCKS": "8", "FLUX_A2AV_RS_EAGER": "1", "FLUX_A2AV_RS_FUSED_PACK": "0", "FLUX_A2AV_WAVE_PACK": "0", "FLUX_A2AV_RS_BUCKET": "0", "FLUX_A2AV_RS_PRERED_BLOCKS": "6"},
        requires=[
            "FLUX_A2AV_LB_UNION",
            "FLUX_A2AV_FUSED_STAGE2",
            "FLUX_A2AV_EARLY_LAUNCH",
            "FLUX_A2AV_RS_MSPLIT_TAG",
            "FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT16_TAG",
            "FLUX_A2AV_NSPLIT_HONOR_TAG",
            "FLUX_A2AV_RS_CTA_1086_TAG",
            "FLUX_A2AV_RS_MAX_SEND_ROWS",
            "FLUX_A2AV_RS_MAX_CONV_ROWS",
            "FLUX_A2AV_RS_MAX_WIRE_ROWS",
        ],
        l1_pattern="a2av_hier_compress",
    ),
    # v2 twin with the eager arrival-order receiver reduce DISABLED (legacy
    # wait-all + CSR reduce). 4n b8 discriminators (capsules 20260824-100751 /
    # -100931): the eager kernel costs ~1.0 ms of l1 at W=16 AND inflates l0
    # by ~0.4-0.5 ms (mechanism open); eager0 makes v2 TIE v1 on qwen at 4n.
    # Scale twins decide the eager policy (receive-side overlap should matter
    # more at 8n/16n where arrivals spread over a longer wire).
    "l01_slipstream_v2_noeager": dict(
        comm_pattern="l01_slipstream_v2_noeager",  # cells.csv label only
        driver="l01",
        layer="l01",
        test_args=[
            "--impl", "flux",
            "--l0_comm_pattern", "a2av_hier_compress",
            "--l1_comm_pattern", "a2av_hier_compress",
            "--n_split", "1",
        ],
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "FLUX_A2AV_RS_MSPLIT": "1",
            "FLUX_A2AV_RS_EAGER": "0",
            "FLUX_A2AV_RS_WIRE_STREAMS": "16",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "10", "FLUX_A2AV_RS_REDUCE_BLOCKS": "8", "FLUX_A2AV_RS_FUSED_PACK": "0", "FLUX_A2AV_WAVE_PACK": "0", "FLUX_A2AV_RS_BUCKET": "0", "FLUX_A2AV_RS_PRERED_BLOCKS": "6"},
        requires=[
            "FLUX_A2AV_LB_UNION",
            "FLUX_A2AV_FUSED_STAGE2",
            "FLUX_A2AV_EARLY_LAUNCH",
            "FLUX_A2AV_RS_MSPLIT_TAG",
            "FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT16_TAG",
            "FLUX_A2AV_NSPLIT_HONOR_TAG",
            "FLUX_A2AV_RS_CTA_1086_TAG",
            "FLUX_A2AV_RS_MAX_SEND_ROWS",
            "FLUX_A2AV_RS_MAX_CONV_ROWS",
            "FLUX_A2AV_RS_MAX_WIRE_ROWS",
        ],
        l1_pattern="a2av_hier_compress",
    ),
    # v2 + LANE-CHAIN receiver (Slipstream v2b, 2026-08-24): per-lane
    # front-end waits in EXPECTED arrival order (descending ring, own node
    # last) release per-lane scatter-adds into an fp32 accumulator + one
    # finalize cast — O(W) waits replace the eager kernel's per-element
    # system-scope polling AND recover the receive-side overlap wait-all
    # forfeits. Requires the gen-7+ binary (FLUX_A2AV_RS_LANE_CHAIN_TAG).
    "l01_slipstream_v2_lanechain": dict(
        comm_pattern="l01_slipstream_v2_lanechain",  # cells.csv label only
        driver="l01",
        layer="l01",
        test_args=[
            "--impl", "flux",
            "--l0_comm_pattern", "a2av_hier_compress",
            "--l1_comm_pattern", "a2av_hier_compress",
            "--n_split", "1",
        ],
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "FLUX_A2AV_RS_MSPLIT": "1",
            "FLUX_A2AV_RS_LANE_CHAIN": "1",
            "FLUX_A2AV_RS_WIRE_STREAMS": "16",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "10", "FLUX_A2AV_RS_REDUCE_BLOCKS": "8", "FLUX_A2AV_RS_EAGER": "0", "FLUX_A2AV_RS_FUSED_PACK": "0", "FLUX_A2AV_WAVE_PACK": "0", "FLUX_A2AV_RS_BUCKET": "0", "FLUX_A2AV_RS_PRERED_BLOCKS": "6"},
        requires=[
            "FLUX_A2AV_LB_UNION",
            "FLUX_A2AV_FUSED_STAGE2",
            "FLUX_A2AV_EARLY_LAUNCH",
            "FLUX_A2AV_RS_MSPLIT_TAG",
            "FLUX_A2AV_RS_LANE_CHAIN_TAG",
            "FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT16_TAG",
            "FLUX_A2AV_NSPLIT_HONOR_TAG",
            "FLUX_A2AV_RS_CTA_1086_TAG",
            "FLUX_A2AV_RS_MAX_SEND_ROWS",
            "FLUX_A2AV_RS_MAX_CONV_ROWS",
            "FLUX_A2AV_RS_MAX_WIRE_ROWS",
        ],
        l1_pattern="a2av_hier_compress",
    ),
    # gen-8a (user-approved 2026-08-24): size-ordered waves on the noeager
    # base — remote waves sorted by descending segment size (globally
    # derivable), own node last. Pre-registered falsifier: incast bunching at
    # hot destinations. Requires the gen-8 binary.
    "l01_slipstream_v2_sizeord": dict(
        comm_pattern="l01_slipstream_v2_sizeord",
        driver="l01",
        layer="l01",
        test_args=[
            "--impl", "flux",
            "--l0_comm_pattern", "a2av_hier_compress",
            "--l1_comm_pattern", "a2av_hier_compress",
            "--n_split", "1",
        ],
        env={
            "FLUX_A2AV_LB_UNION": "1", "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1", "FLUX_A2AV_RS_MSPLIT": "1",
            "FLUX_A2AV_RS_EAGER": "0", "FLUX_A2AV_RS_WAVE_ORDER": "1",
            "FLUX_A2AV_RS_WIRE_STREAMS": "16",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "10", "FLUX_A2AV_RS_REDUCE_BLOCKS": "8", "FLUX_A2AV_RS_FUSED_PACK": "0", "FLUX_A2AV_WAVE_PACK": "0", "FLUX_A2AV_RS_BUCKET": "0", "FLUX_A2AV_RS_PRERED_BLOCKS": "6"},
        requires=[
            "FLUX_A2AV_LB_UNION", "FLUX_A2AV_FUSED_STAGE2", "FLUX_A2AV_EARLY_LAUNCH",
            "FLUX_A2AV_RS_MSPLIT_TAG", "FLUX_A2AV_RS_WAVE_ORDER_TAG",
            "FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT16_TAG", "FLUX_A2AV_NSPLIT_HONOR_TAG",
            "FLUX_A2AV_RS_CTA_1086_TAG", "FLUX_A2AV_RS_MAX_SEND_ROWS",
            "FLUX_A2AV_RS_MAX_CONV_ROWS", "FLUX_A2AV_RS_MAX_WIRE_ROWS",
        ],
        l1_pattern="a2av_hier_compress",
    ),
    # gen-8c (user-approved 2026-08-24): epilogue-fused pack on the noeager
    # base — the l1 GEMM scatters the dest-major send panel directly
    # (ScatterD + K-side gate-coefficient pre-fold); the pack kernel runs as
    # a flag relay. NOTE the gen-8 binary's ScatterD epilogue (identity iota)
    # touches EVERY gather_rs arm's D-write microscopically — full rule-4
    # boundary; all comparisons re-baselined within gen-8 capsules.
    "l01_slipstream_v2_fusedpack": dict(
        comm_pattern="l01_slipstream_v2_fusedpack",
        driver="l01",
        layer="l01",
        test_args=[
            "--impl", "flux",
            "--l0_comm_pattern", "a2av_hier_compress",
            "--l1_comm_pattern", "a2av_hier_compress",
            "--n_split", "1",
        ],
        env={
            "FLUX_A2AV_LB_UNION": "1", "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1", "FLUX_A2AV_RS_MSPLIT": "1",
            "FLUX_A2AV_RS_EAGER": "0", "FLUX_A2AV_RS_FUSED_PACK": "1",
            "FLUX_A2AV_RS_WIRE_STREAMS": "16",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "10", "FLUX_A2AV_RS_REDUCE_BLOCKS": "8", "FLUX_A2AV_WAVE_PACK": "0", "FLUX_A2AV_RS_BUCKET": "0", "FLUX_A2AV_RS_PRERED_BLOCKS": "6"},
        requires=[
            "FLUX_A2AV_LB_UNION", "FLUX_A2AV_FUSED_STAGE2", "FLUX_A2AV_EARLY_LAUNCH",
            "FLUX_A2AV_RS_MSPLIT_TAG", "FLUX_A2AV_RS_FUSED_PACK_TAG",
            "FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT16_TAG", "FLUX_A2AV_NSPLIT_HONOR_TAG",
            "FLUX_A2AV_RS_CTA_1086_TAG", "FLUX_A2AV_RS_MAX_SEND_ROWS",
            "FLUX_A2AV_RS_MAX_CONV_ROWS", "FLUX_A2AV_RS_MAX_WIRE_ROWS",
        ],
        l1_pattern="a2av_hier_compress",
    ),
    # gen-8b (user-approved 2026-08-24): dispatch wave-pack on the noeager
    # base — layer0's producer pack splits per send segment (own first, then
    # remote in mirror wire order) so each wire round's put gates on ITS
    # segment instead of the whole pack. Requires the gen-8 binary.
    "l01_slipstream_v2_wavepack": dict(
        comm_pattern="l01_slipstream_v2_wavepack",
        driver="l01",
        layer="l01",
        test_args=[
            "--impl", "flux",
            "--l0_comm_pattern", "a2av_hier_compress",
            "--l1_comm_pattern", "a2av_hier_compress",
            "--n_split", "1",
        ],
        env={
            "FLUX_A2AV_LB_UNION": "1", "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1", "FLUX_A2AV_RS_MSPLIT": "1",
            "FLUX_A2AV_RS_EAGER": "0", "FLUX_A2AV_WAVE_PACK": "1",
            "FLUX_A2AV_RS_WIRE_STREAMS": "16",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "10", "FLUX_A2AV_RS_REDUCE_BLOCKS": "8", "FLUX_A2AV_RS_FUSED_PACK": "0", "FLUX_A2AV_RS_BUCKET": "0", "FLUX_A2AV_RS_PRERED_BLOCKS": "6"},
        requires=[
            "FLUX_A2AV_LB_UNION", "FLUX_A2AV_FUSED_STAGE2", "FLUX_A2AV_EARLY_LAUNCH",
            "FLUX_A2AV_RS_MSPLIT_TAG", "FLUX_A2AV_WAVE_PACK_TAG",
            "FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT16_TAG", "FLUX_A2AV_NSPLIT_HONOR_TAG",
            "FLUX_A2AV_RS_CTA_1086_TAG", "FLUX_A2AV_RS_MAX_SEND_ROWS",
            "FLUX_A2AV_RS_MAX_CONV_ROWS", "FLUX_A2AV_RS_MAX_WIRE_ROWS",
        ],
        l1_pattern="a2av_hier_compress",
    ),
    # gen-8 COMPOSED arm: fused pack (l1 GEMM scatters the send panel) +
    # dispatch wave-pack (l0 per-segment pack) on the noeager base — the two
    # gate-winning gen-8 mechanisms together; sizeord excluded (consistent
    # loser at 4n gates + 16n gate).
    "l01_slipstream_v2_fpwp": dict(
        comm_pattern="l01_slipstream_v2_fpwp",
        driver="l01",
        layer="l01",
        test_args=[
            "--impl", "flux",
            "--l0_comm_pattern", "a2av_hier_compress",
            "--l1_comm_pattern", "a2av_hier_compress",
            "--n_split", "1",
        ],
        env={
            "FLUX_A2AV_LB_UNION": "1", "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1", "FLUX_A2AV_RS_MSPLIT": "1",
            "FLUX_A2AV_RS_EAGER": "0", "FLUX_A2AV_RS_FUSED_PACK": "1",
            "FLUX_A2AV_WAVE_PACK": "1",
            "FLUX_A2AV_RS_WIRE_STREAMS": "16",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "10", "FLUX_A2AV_RS_REDUCE_BLOCKS": "8", "FLUX_A2AV_RS_BUCKET": "0", "FLUX_A2AV_RS_PRERED_BLOCKS": "6"},
        requires=[
            "FLUX_A2AV_LB_UNION", "FLUX_A2AV_FUSED_STAGE2", "FLUX_A2AV_EARLY_LAUNCH",
            "FLUX_A2AV_RS_MSPLIT_TAG", "FLUX_A2AV_RS_FUSED_PACK_TAG",
            "FLUX_A2AV_WAVE_PACK_TAG",
            "FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT16_TAG", "FLUX_A2AV_NSPLIT_HONOR_TAG",
            "FLUX_A2AV_RS_CTA_1086_TAG", "FLUX_A2AV_RS_MAX_SEND_ROWS",
            "FLUX_A2AV_RS_MAX_CONV_ROWS", "FLUX_A2AV_RS_MAX_WIRE_ROWS",
        ],
        l1_pattern="a2av_hier_compress",
    ),
    # lanechain receiver diagnostics (2026-08-24, user-directed "why does
    # reduce-throughout lose"): CTA ladder discriminating bandwidth-bound (H1,
    # more CTAs don't help) vs fold-starvation (H2, they do). Clones of
    # l01_slipstream_v2_lanechain with REDUCE_BLOCKS 16 / 24.
    "l01_slipstream_v2_lanechain_rb16": dict(
        comm_pattern="l01_slipstream_v2_lanechain_rb16",
        driver="l01", layer="l01",
        test_args=["--impl", "flux", "--l0_comm_pattern", "a2av_hier_compress",
                   "--l1_comm_pattern", "a2av_hier_compress", "--n_split", "1"],
        env={"FLUX_A2AV_LB_UNION": "1", "FLUX_A2AV_FUSED_STAGE2": "1",
             "FLUX_A2AV_EARLY_LAUNCH": "1", "FLUX_A2AV_RS_MSPLIT": "1",
             "FLUX_A2AV_RS_LANE_CHAIN": "1", "FLUX_A2AV_RS_WIRE_STREAMS": "16",
             "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "10",
             "FLUX_A2AV_RS_REDUCE_BLOCKS": "16", "FLUX_A2AV_RS_EAGER": "0", "FLUX_A2AV_RS_FUSED_PACK": "0", "FLUX_A2AV_WAVE_PACK": "0", "FLUX_A2AV_RS_BUCKET": "0", "FLUX_A2AV_RS_PRERED_BLOCKS": "6"},
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_FUSED_STAGE2", "FLUX_A2AV_EARLY_LAUNCH",
                  "FLUX_A2AV_RS_MSPLIT_TAG", "FLUX_A2AV_RS_LANE_CHAIN_TAG",
                  "FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT16_TAG", "FLUX_A2AV_NSPLIT_HONOR_TAG",
                  "FLUX_A2AV_RS_CTA_1086_TAG", "FLUX_A2AV_RS_MAX_SEND_ROWS",
                  "FLUX_A2AV_RS_MAX_CONV_ROWS", "FLUX_A2AV_RS_MAX_WIRE_ROWS"],
        l1_pattern="a2av_hier_compress",
    ),
    "l01_slipstream_v2_lanechain_rb24": dict(
        comm_pattern="l01_slipstream_v2_lanechain_rb24",
        driver="l01", layer="l01",
        test_args=["--impl", "flux", "--l0_comm_pattern", "a2av_hier_compress",
                   "--l1_comm_pattern", "a2av_hier_compress", "--n_split", "1"],
        env={"FLUX_A2AV_LB_UNION": "1", "FLUX_A2AV_FUSED_STAGE2": "1",
             "FLUX_A2AV_EARLY_LAUNCH": "1", "FLUX_A2AV_RS_MSPLIT": "1",
             "FLUX_A2AV_RS_LANE_CHAIN": "1", "FLUX_A2AV_RS_WIRE_STREAMS": "16",
             "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "10",
             "FLUX_A2AV_RS_REDUCE_BLOCKS": "24", "FLUX_A2AV_RS_EAGER": "0", "FLUX_A2AV_RS_FUSED_PACK": "0", "FLUX_A2AV_WAVE_PACK": "0", "FLUX_A2AV_RS_BUCKET": "0", "FLUX_A2AV_RS_PRERED_BLOCKS": "6"},
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_FUSED_STAGE2", "FLUX_A2AV_EARLY_LAUNCH",
                  "FLUX_A2AV_RS_MSPLIT_TAG", "FLUX_A2AV_RS_LANE_CHAIN_TAG",
                  "FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT16_TAG", "FLUX_A2AV_NSPLIT_HONOR_TAG",
                  "FLUX_A2AV_RS_CTA_1086_TAG", "FLUX_A2AV_RS_MAX_SEND_ROWS",
                  "FLUX_A2AV_RS_MAX_CONV_ROWS", "FLUX_A2AV_RS_MAX_WIRE_ROWS"],
        l1_pattern="a2av_hier_compress",
    ),
    # completion-bucketed register receiver (Slipstream gen-10, 2026-08-24,
    # user-directed): arrival-order folding at wait-all's 1x bytes -- tokens
    # bucket by their last contribution's chain position; each lane wait folds
    # exactly the tokens it completes. The receiver idea projected to win where
    # lane-chain lost (no 4-5x scratch RMW amplification).
    "l01_slipstream_v2_bucket": dict(
        comm_pattern="l01_slipstream_v2_bucket",
        driver="l01", layer="l01",
        test_args=["--impl", "flux", "--l0_comm_pattern", "a2av_hier_compress",
                   "--l1_comm_pattern", "a2av_hier_compress", "--n_split", "1"],
        env={"FLUX_A2AV_LB_UNION": "1", "FLUX_A2AV_FUSED_STAGE2": "1",
             "FLUX_A2AV_EARLY_LAUNCH": "1", "FLUX_A2AV_RS_MSPLIT": "1",
             "FLUX_A2AV_RS_BUCKET": "1", "FLUX_A2AV_RS_WIRE_STREAMS": "16",
             "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "10",
             "FLUX_A2AV_RS_REDUCE_BLOCKS": "8", "FLUX_A2AV_RS_EAGER": "0", "FLUX_A2AV_RS_FUSED_PACK": "0", "FLUX_A2AV_WAVE_PACK": "0", "FLUX_A2AV_RS_PRERED_BLOCKS": "6"},
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_FUSED_STAGE2", "FLUX_A2AV_EARLY_LAUNCH",
                  "FLUX_A2AV_RS_MSPLIT_TAG", "FLUX_A2AV_RS_BUCKET_TAG",
                  "FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT16_TAG", "FLUX_A2AV_NSPLIT_HONOR_TAG",
                  "FLUX_A2AV_RS_CTA_1086_TAG", "FLUX_A2AV_RS_MAX_SEND_ROWS",
                  "FLUX_A2AV_RS_MAX_CONV_ROWS", "FLUX_A2AV_RS_MAX_WIRE_ROWS"],
        l1_pattern="a2av_hier_compress",
    ),
    "l01_slipstream_v2_fpwp_bucket": dict(
        comm_pattern="l01_slipstream_v2_fpwp_bucket",
        driver="l01", layer="l01",
        test_args=["--impl", "flux", "--l0_comm_pattern", "a2av_hier_compress",
                   "--l1_comm_pattern", "a2av_hier_compress", "--n_split", "1"],
        env={"FLUX_A2AV_LB_UNION": "1", "FLUX_A2AV_FUSED_STAGE2": "1",
             "FLUX_A2AV_EARLY_LAUNCH": "1", "FLUX_A2AV_RS_MSPLIT": "1",
             "FLUX_A2AV_RS_EAGER": "0", "FLUX_A2AV_RS_FUSED_PACK": "1",
             "FLUX_A2AV_WAVE_PACK": "1", "FLUX_A2AV_RS_BUCKET": "1",
             "FLUX_A2AV_RS_WIRE_STREAMS": "16",
             "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "10",
             "FLUX_A2AV_RS_REDUCE_BLOCKS": "8", "FLUX_A2AV_RS_PRERED_BLOCKS": "6"},
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_FUSED_STAGE2", "FLUX_A2AV_EARLY_LAUNCH",
                  "FLUX_A2AV_RS_MSPLIT_TAG", "FLUX_A2AV_RS_FUSED_PACK_TAG",
                  "FLUX_A2AV_WAVE_PACK_TAG", "FLUX_A2AV_RS_BUCKET_TAG",
                  "FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT16_TAG", "FLUX_A2AV_NSPLIT_HONOR_TAG",
                  "FLUX_A2AV_RS_CTA_1086_TAG", "FLUX_A2AV_RS_MAX_SEND_ROWS",
                  "FLUX_A2AV_RS_MAX_CONV_ROWS", "FLUX_A2AV_RS_MAX_WIRE_ROWS"],
        l1_pattern="a2av_hier_compress",
    ),
    # fpwp + lanechain-rb24 receiver composition (2026-08-24, CTA-ladder
    # follow-up): the ladder showed lanechain's loss = fold starvation at
    # REDUCE_BLOCKS=8 (rb24 beat noeager at b64). This arm asks whether the
    # canon fpwp config wants the lanechain receiver at rb24.
    "l01_slipstream_v2_fpwp_lcrb24": dict(
        comm_pattern="l01_slipstream_v2_fpwp_lcrb24",
        driver="l01", layer="l01",
        test_args=["--impl", "flux", "--l0_comm_pattern", "a2av_hier_compress",
                   "--l1_comm_pattern", "a2av_hier_compress", "--n_split", "1"],
        env={"FLUX_A2AV_LB_UNION": "1", "FLUX_A2AV_FUSED_STAGE2": "1",
             "FLUX_A2AV_EARLY_LAUNCH": "1", "FLUX_A2AV_RS_MSPLIT": "1",
             "FLUX_A2AV_RS_EAGER": "0", "FLUX_A2AV_RS_FUSED_PACK": "1",
             "FLUX_A2AV_WAVE_PACK": "1", "FLUX_A2AV_RS_LANE_CHAIN": "1",
             "FLUX_A2AV_RS_WIRE_STREAMS": "16",
             "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "10",
             "FLUX_A2AV_RS_REDUCE_BLOCKS": "24", "FLUX_A2AV_RS_BUCKET": "0", "FLUX_A2AV_RS_PRERED_BLOCKS": "6"},
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_FUSED_STAGE2", "FLUX_A2AV_EARLY_LAUNCH",
                  "FLUX_A2AV_RS_MSPLIT_TAG", "FLUX_A2AV_RS_FUSED_PACK_TAG",
                  "FLUX_A2AV_WAVE_PACK_TAG", "FLUX_A2AV_RS_LANE_CHAIN_TAG",
                  "FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT16_TAG", "FLUX_A2AV_NSPLIT_HONOR_TAG",
                  "FLUX_A2AV_RS_CTA_1086_TAG", "FLUX_A2AV_RS_MAX_SEND_ROWS",
                  "FLUX_A2AV_RS_MAX_CONV_ROWS", "FLUX_A2AV_RS_MAX_WIRE_ROWS"],
        l1_pattern="a2av_hier_compress",
    ),
    # bare-defaults twin: NO RS env at all — must equal the pinned arm
    # within noise (the default-flip validation cell; keep for future
    # binary-identity checks)
    "l01_wincast_bare": dict(
        comm_pattern="l01_wincast_bare",
        driver="l01",
        layer="l01",
        test_args=[
            "--impl", "flux",
            "--l0_comm_pattern", "a2av_hier_compress",
            "--l1_comm_pattern", "a2av_hier_compress",
        ],
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=[
            "FLUX_A2AV_LB_UNION",
            "FLUX_A2AV_FUSED_STAGE2",
            "FLUX_A2AV_EARLY_LAUNCH",
            "FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT16_TAG",
            "FLUX_A2AV_NSPLIT_HONOR_TAG",
            "FLUX_A2AV_RS_CTA_1086_TAG",
            "FLUX_A2AV_RS_MAX_SEND_ROWS",
            "FLUX_A2AV_RS_MAX_CONV_ROWS",
            "FLUX_A2AV_RS_MAX_WIRE_ROWS",
        ],
        l1_pattern="a2av_hier_compress",
    ),
    # l0 = lb_union + the P3 factorial WINNERS (2026-08-16, three-run sign
    # agreement over capsules 7ff7098d/1451c017/78f371c6 + bhigh twins):
    # F=FUSED_STAGE2 win (-0.2..-0.7 ms, b1-b8+b32), E=EARLY_LAUNCH win
    # (-0.3..-1.8 ms, b2/b16-b64), N=FANOUT loss (+0.05..+0.6, b2-b16) —
    # so F+E on, N off. l1 = hier + eager reduce (binary-B pairing).
    # CLOSED — LOSER (2026-08-16): the eager pairing lost the composition
    # A/B (14.4 vs 9.95 ms at b8) — retained as the l01 eager ablation only,
    # per the layer1 CLOSED note above.
    # ---- Mission-4 dispatch-audit candidates (2026-08-23) ------------------
    # A/B ablations of l01_lbunion_compress, one knob each, paired in-capsule
    # against the control on ONE binary (rule 4). Mechanisms:
    #   pull2s : FLUX_A2AV_RELAY_PULL_STREAM — relay phase-1 pulls on their
    #            own stream; round-dn wire put waits ONLY round dn's pull
    #            event (the shipped single-stream order serialized round 1's
    #            blocking put behind round NN-1's staging, zero data dep).
    #   rswire2: FLUX_A2AV_RS_WIRE_STREAMS=2 — l1 compress wire ladder's
    #            independent (sid, tn) blocking puts parity-split over two
    #            internode streams (conn>1 only).
    #   fanout : FLUX_A2AV_FANOUT re-test under TODAY'S regime (blocking
    #            wire + real traces; the NR-06-era loss predates both, and
    #            blocking relay sends make cross-round arrival inversion at
    #            gateways MORE likely). Ablation-only per the 8.17 closure —
    #            a canonicalization flip is a user decision.
    "l01_lbunion_compress_pull2s": dict(
        comm_pattern="l01_lbunion_compress_pull2s",
        driver="l01",
        layer="l01",
        test_args=[
            "--impl", "flux",
            "--l0_comm_pattern", "a2av_hier_compress",
            "--l1_comm_pattern", "a2av_hier_compress",
        ],
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "FLUX_A2AV_RELAY_PULL_STREAM": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "6", "FLUX_A2AV_RS_REDUCE_BLOCKS": "6", "FLUX_A2AV_RS_PRERED_BLOCKS": "4"},
        requires=[
            "FLUX_A2AV_LB_UNION",
            "FLUX_A2AV_FUSED_STAGE2",
            "FLUX_A2AV_EARLY_LAUNCH",
            "FLUX_A2AV_RELAY_PULL_STREAM",
            "FLUX_A2AV_RS_MAX_SEND_ROWS",
            "FLUX_A2AV_RS_MAX_CONV_ROWS",
            "FLUX_A2AV_RS_MAX_WIRE_ROWS",
        ],
        l1_pattern="a2av_hier_compress",
    ),
    "l01_lbunion_compress_rswire2": dict(
        comm_pattern="l01_lbunion_compress_rswire2",
        driver="l01",
        layer="l01",
        test_args=[
            "--impl", "flux",
            "--l0_comm_pattern", "a2av_hier_compress",
            "--l1_comm_pattern", "a2av_hier_compress",
            # 2026-08-24 canonicalization (user decision): l1 combine ns2
            # (bridges the split ladder interior; overrides the spec's
            # n_split_l1 by argparse last-wins) + 16 wire lanes
            "--n_split", "2",
        ],
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "FLUX_A2AV_RS_WIRE_STREAMS": "16",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "6", "FLUX_A2AV_RS_REDUCE_BLOCKS": "6", "FLUX_A2AV_RS_PRERED_BLOCKS": "4"},
        requires=[
            "FLUX_A2AV_LB_UNION",
            "FLUX_A2AV_FUSED_STAGE2",
            "FLUX_A2AV_EARLY_LAUNCH",
            "FLUX_A2AV_RS_WIRE_STREAMS",
            "FLUX_A2AV_RS_MAX_SEND_ROWS",
            "FLUX_A2AV_RS_MAX_CONV_ROWS",
            "FLUX_A2AV_RS_MAX_WIRE_ROWS",
        ],
        l1_pattern="a2av_hier_compress",
    ),
    "l01_lbunion_compress_fanout": dict(
        comm_pattern="l01_lbunion_compress_fanout",
        driver="l01",
        layer="l01",
        test_args=[
            "--impl", "flux",
            "--l0_comm_pattern", "a2av_hier_compress",
            "--l1_comm_pattern", "a2av_hier_compress",
        ],
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "FLUX_A2AV_FANOUT": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "6", "FLUX_A2AV_RS_REDUCE_BLOCKS": "6", "FLUX_A2AV_RS_PRERED_BLOCKS": "4"},
        requires=[
            "FLUX_A2AV_LB_UNION",
            "FLUX_A2AV_FUSED_STAGE2",
            "FLUX_A2AV_EARLY_LAUNCH",
            "FLUX_A2AV_FANOUT",
            "FLUX_A2AV_RS_MAX_SEND_ROWS",
            "FLUX_A2AV_RS_MAX_CONV_ROWS",
            "FLUX_A2AV_RS_MAX_WIRE_ROWS",
        ],
        l1_pattern="a2av_hier_compress",
    ),
    "l01_lbunion_compress_m4stack": dict(
        comm_pattern="l01_lbunion_compress_m4stack",
        driver="l01",
        layer="l01",
        test_args=[
            "--impl", "flux",
            "--l0_comm_pattern", "a2av_hier_compress",
            "--l1_comm_pattern", "a2av_hier_compress",
        ],
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "FLUX_A2AV_RELAY_PULL_STREAM": "1",
            "FLUX_A2AV_RS_WIRE_STREAMS": "2",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "6", "FLUX_A2AV_RS_REDUCE_BLOCKS": "6", "FLUX_A2AV_RS_PRERED_BLOCKS": "4"},
        requires=[
            "FLUX_A2AV_LB_UNION",
            "FLUX_A2AV_FUSED_STAGE2",
            "FLUX_A2AV_EARLY_LAUNCH",
            "FLUX_A2AV_RELAY_PULL_STREAM",
            "FLUX_A2AV_RS_WIRE_STREAMS",
            "FLUX_A2AV_RS_MAX_SEND_ROWS",
            "FLUX_A2AV_RS_MAX_CONV_ROWS",
            "FLUX_A2AV_RS_MAX_WIRE_ROWS",
        ],
        l1_pattern="a2av_hier_compress",
    ),
    # M4 l1 CTA-budget probe (2026-08-23): the pack kernel's fixed 6 CTAs
    # move ALL M*N bytes and the pre-reduce's 4 CTAs merge the whole conv
    # panel — payload-proportional work behind fixed budgets (audit 4e).
    # Probe wider budgets on top of the rswire2 winner at high b.
    "l01_lbunion_compress_rsw2_cta10": dict(
        comm_pattern="l01_lbunion_compress_rsw2_cta10",
        driver="l01",
        layer="l01",
        test_args=[
            "--impl", "flux",
            "--l0_comm_pattern", "a2av_hier_compress",
            "--l1_comm_pattern", "a2av_hier_compress",
            # 2026-08-24 canonicalization (user decision): l1 combine ns2
            # (bridges the split ladder interior; overrides the spec's
            # n_split_l1 by argparse last-wins) + 16 wire lanes
            "--n_split", "2",
        ],
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "FLUX_A2AV_RS_WIRE_STREAMS": "16",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "10", "FLUX_A2AV_RS_REDUCE_BLOCKS": "8", "FLUX_A2AV_RS_PRERED_BLOCKS": "6"},
        requires=[
            "FLUX_A2AV_LB_UNION",
            "FLUX_A2AV_FUSED_STAGE2",
            "FLUX_A2AV_EARLY_LAUNCH",
            "FLUX_A2AV_RS_WIRE_STREAMS",
            "FLUX_A2AV_RS_MAX_SEND_ROWS",
            "FLUX_A2AV_RS_MAX_CONV_ROWS",
            "FLUX_A2AV_RS_MAX_WIRE_ROWS",
        ],
        l1_pattern="a2av_hier_compress",
    ),
    "l01_lbunion_compress_rsw2_cta4": dict(
        comm_pattern="l01_lbunion_compress_rsw2_cta4",
        driver="l01",
        layer="l01",
        test_args=[
            "--impl", "flux",
            "--l0_comm_pattern", "a2av_hier_compress",
            "--l1_comm_pattern", "a2av_hier_compress",
            # 2026-08-24 canonicalization (user decision): l1 combine ns2
            # (bridges the split ladder interior; overrides the spec's
            # n_split_l1 by argparse last-wins) + 16 wire lanes
            "--n_split", "2",
        ],
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "FLUX_A2AV_RS_WIRE_STREAMS": "16",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "4", "FLUX_A2AV_RS_REDUCE_BLOCKS": "4", "FLUX_A2AV_RS_PRERED_BLOCKS": "3"},
        requires=[
            "FLUX_A2AV_LB_UNION",
            "FLUX_A2AV_FUSED_STAGE2",
            "FLUX_A2AV_EARLY_LAUNCH",
            "FLUX_A2AV_RS_WIRE_STREAMS",
            "FLUX_A2AV_RS_MAX_SEND_ROWS",
            "FLUX_A2AV_RS_MAX_CONV_ROWS",
            "FLUX_A2AV_RS_MAX_WIRE_ROWS",
        ],
        l1_pattern="a2av_hier_compress",
    ),
    "l01_lbunion_hier_eager": dict(
        comm_pattern="l01_lbunion_hier_eager",  # cells.csv label only
        driver="l01",
        layer="l01",
        test_args=[
            "--impl", "flux",
            "--l0_comm_pattern", "a2av_hier_compress",
            "--l1_comm_pattern", "a2av_hier",
        ],
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "FLUX_A2AV_RS_EAGER": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8",
        },
        requires=[
            "FLUX_A2AV_LB_UNION",
            "FLUX_A2AV_FUSED_STAGE2",
            "FLUX_A2AV_EARLY_LAUNCH",
            "FLUX_A2AV_RS_MAX_SEND_ROWS",
            "FLUX_A2AV_RS_EAGER",
        ],
        l1_pattern="a2av_hier",
    ),
}

# Slipstream (2026-08-23, user-named; briefly "WinCast" in the first
# canonicalization commit): the concise canonical name for the dispatch
# leg of the LLC trio — PLACE-lambda (placement), LocCap (routing),
# Slipstream (dispatch) — i.e. hier_compress + LB_UNION Tier-B windowed
# union broadcast + FUSED_STAGE2 + EARLY_LAUNCH + split RS wire +
# rebalanced combine CTAs, all binary defaults. Alias KEYS only:
# cells.csv labels stay the historical comm_pattern strings so capsules
# remain comparable (incl. the one l01_wincast_bare flip-validation
# capsule, 20260823-1144*).
# 2026-08-25 (user decision): EPIC-parity demand-sized PLL twin — identical
# routing/kernels/audits to llc_l01_s1; only the frozen-buffer sizing contract
# differs (realized reference + drift cushions instead of provable caps).
# Unblocks 16n b64 (heap fits like EPIC's) and shrinks the loccap_sl tail pad
# (plan_ms never-mix vs capacity-mode capsules).
VARIANTS["llc_l01_s1_demand"] = dict(
    VARIANTS["llc_l01_s1"],
    test_args=(VARIANTS["llc_l01_s1"]["test_args"]
               + ["--llc_sizing", "demand"]),
)
# 2026-08-27 (branch pv2, user canon ruling): PV2 placement swapped into
# the llc stack — same LocCap sender-local kernel routing, same staged
# transport, same sizing contract; only the placement solver changes
# (placement_v2: stateless node-aware greedy from the demand histogram,
# ~1-2 ms host — the cheaper, equal alternative to PLACE-lambda-FAST per
# the 4n/8n A/B, handoff 23). THE canonical "PLL/LLC" datapoint lane
# going forward (datapoint skill updated); never bitwise-compare its
# placements/out_sha against placelambda cells.
VARIANTS["llc_l01_s1_pv2"] = dict(
    VARIANTS["llc_l01_s1"],
    test_args=[a if a != "placelambda_fast" else "pv2"
               for a in VARIANTS["llc_l01_s1"]["test_args"]]
              # slack-parity pin (datapoint skill rule: never rely on the
              # driver default even though it IS 2)
              + ["--redundant_per_rank", "2"],
)

VARIANTS["slipstream"] = VARIANTS["hier_compress_lb_union"]
# SCHEMA rule 13 (2026-08-24, user decision): l01_slipstream now names the
# OFFICIAL Slipstream — the destination-driven token-centric combine (msplit
# destination waves + epilogue-fused pack + dispatch wave-pack + the
# completion-bucketed register receiver), all binary defaults gated by
# FLUX_A2AV_SLIPSTREAM2_TAG. The pre-supersession column-split canon lives on
# as l01_slipstream_v1 (alias of the historical l01_lbunion_compress, whose
# cells.csv label it keeps). Pre-rule-13 capsules labeled l01_slipstream
# measured the v1 config — never compare by variant name across the flip.
def _v1_pinned(base):
    # rule 13: the v1 arms must run the AUTHENTIC pre-supersession config on
    # gen-11+ binaries, where the mechanism knobs default ON — pin all five
    # to their old-canon values (all off; eager's old default was 0 too).
    d = dict(base)
    d["env"] = dict(base["env"])
    for k in ("FLUX_A2AV_RS_MSPLIT", "FLUX_A2AV_RS_EAGER", "FLUX_A2AV_RS_FUSED_PACK",
              "FLUX_A2AV_WAVE_PACK", "FLUX_A2AV_RS_BUCKET"):
        d["env"].setdefault(k, "0")
    return d
VARIANTS["l01_slipstream_v1"] = _v1_pinned(VARIANTS["l01_lbunion_compress"])
VARIANTS["l01_slipstream_v1_bare"] = _v1_pinned(VARIANTS["l01_wincast_bare"])
VARIANTS["l01_slipstream"] = dict(
    comm_pattern="l01_slipstream",
    driver="l01", layer="l01",
    test_args=["--impl", "flux", "--l0_comm_pattern", "a2av_hier_compress",
               "--l1_comm_pattern", "a2av_hier_compress", "--n_split", "1"],
    env={"FLUX_A2AV_LB_UNION": "1", "FLUX_A2AV_FUSED_STAGE2": "1",
         "FLUX_A2AV_EARLY_LAUNCH": "1", "FLUX_A2AV_RS_MSPLIT": "1",
         "FLUX_A2AV_RS_EAGER": "0", "FLUX_A2AV_RS_FUSED_PACK": "1",
         "FLUX_A2AV_WAVE_PACK": "1", "FLUX_A2AV_RS_BUCKET": "1",
         "FLUX_A2AV_RS_WIRE_STREAMS": "16",
         "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "10",
         "FLUX_A2AV_RS_REDUCE_BLOCKS": "8", "FLUX_A2AV_RS_PRERED_BLOCKS": "6"},
    requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_FUSED_STAGE2", "FLUX_A2AV_EARLY_LAUNCH",
              "FLUX_A2AV_SLIPSTREAM2_TAG", "FLUX_A2AV_RS_MSPLIT_TAG",
              "FLUX_A2AV_RS_FUSED_PACK_TAG", "FLUX_A2AV_WAVE_PACK_TAG",
              "FLUX_A2AV_RS_BUCKET_TAG",
              "FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT16_TAG", "FLUX_A2AV_NSPLIT_HONOR_TAG",
              "FLUX_A2AV_RS_CTA_1086_TAG", "FLUX_A2AV_RS_MAX_SEND_ROWS",
              "FLUX_A2AV_RS_MAX_CONV_ROWS", "FLUX_A2AV_RS_MAX_WIRE_ROWS"],
    l1_pattern="a2av_hier_compress",
)
# bare-defaults twin of the NEW canon: no RS/mechanism env at all — the
# binary defaults must reproduce the pinned arm within noise (flip gate).
VARIANTS["l01_slipstream_bare"] = dict(
    comm_pattern="l01_slipstream_bare",
    driver="l01", layer="l01",
    test_args=["--impl", "flux", "--l0_comm_pattern", "a2av_hier_compress",
               "--l1_comm_pattern", "a2av_hier_compress", "--n_split", "1"],
    env={"FLUX_A2AV_LB_UNION": "1", "FLUX_A2AV_FUSED_STAGE2": "1",
         "FLUX_A2AV_EARLY_LAUNCH": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
    requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_FUSED_STAGE2", "FLUX_A2AV_EARLY_LAUNCH",
              "FLUX_A2AV_SLIPSTREAM2_TAG",
              "FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT16_TAG", "FLUX_A2AV_NSPLIT_HONOR_TAG",
              "FLUX_A2AV_RS_CTA_1086_TAG", "FLUX_A2AV_RS_MAX_SEND_ROWS",
              "FLUX_A2AV_RS_MAX_CONV_ROWS", "FLUX_A2AV_RS_MAX_WIRE_ROWS"],
    l1_pattern="a2av_hier_compress",
)


# ---------------------------------------------------------------------------
# OURS (2026-08-24 fusion campaign, worktree `ours`): the integrated
# PLACE-lambda + LocCap + Slipstream-v2 arm — LLC's placement/routing feeding
# the fused overlapped dispatch(LB_UNION Tier-B + wave-pack)+GEMM0 and the
# Slipstream v2 combine (msplit + fused-pack + bucket, ns1) through the
# virtual-slot space, one op pair, per-iteration rule-5 plan. Scenario 1:
# oracle placement (static, untimed, reported), relaxed loccap_sl kernel
# routing per iteration. Driver: test_moe_ours_traffic.py (fresh driver —
# the llc/slipstream baselines stay byte-untouched for in-capsule A/Bs).
# Mechanism-vector pins mirror l01_slipstream so binary-default drift can
# never change arm identity.
_OURS_ENV = {
    "FLUX_A2AV_LB_UNION": "1", "FLUX_A2AV_FUSED_STAGE2": "1",
    "FLUX_A2AV_EARLY_LAUNCH": "1", "FLUX_A2AV_RS_MSPLIT": "1",
    "FLUX_A2AV_RS_EAGER": "0", "FLUX_A2AV_RS_FUSED_PACK": "1",
    "FLUX_A2AV_WAVE_PACK": "1", "FLUX_A2AV_RS_BUCKET": "1",
    "FLUX_A2AV_RS_WIRE_STREAMS": "16",
    # conn=32 is the OURS family's single canonical pin (2026-08-25):
    # resolves the s2 channel-aliasing hang (qwen stale-b32) AND the K2
    # torn-row race (shard-chain exposure); the 16n l0 probes showed -5%
    # l0 at b64. The family runs ~21 streams; the historical conn=8 pin
    # was sized for the 4n-era a2av mix. Pre-flip conn=8 capsules are
    # env_json-documented, never byte-compared across the flip.
    "CUDA_DEVICE_MAX_CONNECTIONS": "32", "FLUX_A2AV_RS_PACK_BLOCKS": "10",
    "FLUX_A2AV_RS_REDUCE_BLOCKS": "8", "FLUX_A2AV_RS_PRERED_BLOCKS": "6",
    # 2026-08-29 CANON (handoff 26 §4, user ruling): lossless plan graphs
    # ON + (via the --plan_overlap 2 base args) late combine-meta overlap;
    # the binary same-day flips FLUX_A2AV_RS_WAVE_ADAPT default to 48.
    # Never byte-compare env_json/totals across the 8/29 boundary; the
    # _legacy twin below pins the pre-8/29 behavior for regression A/Bs.
    "FLUX_OURS_PLAN_GRAPH": "1", "FLUX_OURS_PLAN_SCALE_GRAPH": "1",
}
_OURS_REQUIRES = [
    "FLUX_A2AV_LB_UNION", "FLUX_A2AV_FUSED_STAGE2", "FLUX_A2AV_EARLY_LAUNCH",
    "FLUX_A2AV_SLIPSTREAM2_TAG", "FLUX_A2AV_RS_MSPLIT_TAG",
    "FLUX_A2AV_RS_FUSED_PACK_TAG", "FLUX_A2AV_WAVE_PACK_TAG",
    "FLUX_A2AV_RS_BUCKET_TAG", "FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT16_TAG",
    "FLUX_A2AV_NSPLIT_HONOR_TAG", "FLUX_A2AV_RS_CTA_1086_TAG",
    "FLUX_A2AV_DISPATCH_ONLY_TAG", "FLUX_PLACELAMBDA_ROUTE_SL_TAG",
    "FLUX_A2AV_RS_MAX_SEND_ROWS", "FLUX_A2AV_RS_MAX_CONV_ROWS",
    "FLUX_A2AV_RS_MAX_WIRE_ROWS",
]
VARIANTS["ours_l01_s1"] = dict(
    comm_pattern="ours_l01", driver="ours", layer="l01",
    test_args=["--eps", "0.0625", "--sizing", "demand",
               "--plan_overlap", "2"],
    env=dict(_OURS_ENV),
    requires=list(_OURS_REQUIRES),
)
# plan-overlap A/B twin (combine-meta derive + scale build under the fused
# l0 on a side stream) — identical otherwise
VARIANTS["ours_l01_s1_ov"] = dict(
    VARIANTS["ours_l01_s1"],
    test_args=["--eps", "0.0625", "--sizing", "demand",
               "--plan_overlap", "2"],
)
# correctness gate twin: per-iteration output validation under relaxed
# routing + random payload (perturbs timing — gate cells only)
VARIANTS["ours_l01_s1_gate"] = dict(
    VARIANTS["ours_l01_s1"],
    test_args=["--eps", "0.0625", "--sizing", "demand",
               "--plan_overlap", "2", "--check_iters", "1"],
)
VARIANTS["ours_l01_s1_gate_ov"] = dict(
    VARIANTS["ours_l01_s1"],
    test_args=["--eps", "0.0625", "--sizing", "demand",
               "--plan_overlap", "2", "--check_iters", "1"],
)
# ---- plan-lane cost-knob probe arms (2026-08-25 16n plan-gap attack;
# knobs documented in flux/testing/ours.py module header, all default OFF
# in the canonical arm). Compare nw/pg arms against _pre (same tail
# buffers) to isolate the wire / graph deltas.
VARIANTS["ours_l01_s1_pre"] = dict(
    VARIANTS["ours_l01_s1"],
    env=dict(_OURS_ENV, FLUX_OURS_PLAN_PREALLOC="1"),
)
VARIANTS["ours_l01_s1_pg"] = dict(
    VARIANTS["ours_l01_s1"],
    env=dict(_OURS_ENV, FLUX_OURS_PLAN_GRAPH="1",
             FLUX_OURS_PLAN_SCALE_GRAPH="1"),
)
# lossless wire cut: phys int16 + probs fp32 bit-split (6 B/entry)
VARIANTS["ours_l01_s1_nw1"] = dict(
    VARIANTS["ours_l01_s1"],
    env=dict(_OURS_ENV, FLUX_OURS_PLAN_XCHG_NARROW="1"),
)
# llc wire parity: + probs bf16 (4 B/entry). LOSSY probs rounding —
# out_sha never-mix vs narrow<2 arms (allclose gates still bind).
VARIANTS["ours_l01_s1_nw2"] = dict(
    VARIANTS["ours_l01_s1"],
    env=dict(_OURS_ENV, FLUX_OURS_PLAN_XCHG_NARROW="2"),
)
# combined candidate: narrow-2 wire + both plan graphs
VARIANTS["ours_l01_s1_planfast"] = dict(
    VARIANTS["ours_l01_s1"],
    env=dict(_OURS_ENV, FLUX_OURS_PLAN_XCHG_NARROW="2",
             FLUX_OURS_PLAN_GRAPH="1", FLUX_OURS_PLAN_SCALE_GRAPH="1"),
)
# 4n correctness gates for the new knob paths (run BEFORE scale probes)
VARIANTS["ours_l01_s1_gate_nw1"] = dict(
    VARIANTS["ours_l01_s1_gate"],
    env=dict(_OURS_ENV, FLUX_OURS_PLAN_XCHG_NARROW="1"),
)
VARIANTS["ours_l01_s1_gate_planfast"] = dict(
    VARIANTS["ours_l01_s1_gate"],
    env=dict(_OURS_ENV, FLUX_OURS_PLAN_XCHG_NARROW="2",
             FLUX_OURS_PLAN_GRAPH="1", FLUX_OURS_PLAN_SCALE_GRAPH="1"),
)
# (allin stack defined after the combo knob arms below)
# scenario-2 arms (live re-placement + OVERLAPPED weight movement; WPM
# multicast + NIC-shard + per-slot weight-gated tiles). s2 sizes at the
# provable caps (--sizing capacity implied in-driver for s2).
VARIANTS["ours_l01_s2"] = dict(
    VARIANTS["ours_l01_s1"],
    test_args=["--eps", "0.0625", "--sizing", "capacity",
               "--plan_overlap", "2", "--scenario", "s2",
               "--place_gain_threshold_ppm", "0"],
)
# gate: per-iteration output checks + stale-resident (movement EVERY
# iteration) + the rule-6c weight payload probe (WPM wire audit)
VARIANTS["ours_l01_s2_gate"] = dict(
    VARIANTS["ours_l01_s1"],
    test_args=["--eps", "0.0625", "--sizing", "capacity",
               "--plan_overlap", "2", "--scenario", "s2",
               "--place_gain_threshold_ppm", "0",
               "--check_iters", "1", "--s2_stale", "rot",
               "--s2_force_trigger", "1", "--s2_wprobe", "1"],
)
# perf worst case: movement fires every timed iteration
VARIANTS["ours_l01_s2_stale"] = dict(
    VARIANTS["ours_l01_s1"],
    test_args=["--eps", "0.0625", "--sizing", "capacity",
               "--plan_overlap", "2", "--scenario", "s2",
               "--place_gain_threshold_ppm", "0",
               "--s2_stale", "rot", "--s2_force_trigger", "1"],
)
# ablation: one landing join before GEMM0 instead of weight-gated tiles
VARIANTS["ours_l01_s2_stale_join"] = dict(
    VARIANTS["ours_l01_s1"],
    test_args=["--eps", "0.0625", "--sizing", "capacity",
               "--plan_overlap", "2", "--scenario", "s2",
               "--place_gain_threshold_ppm", "0",
               "--s2_stale", "rot", "--s2_force_trigger", "1",
               "--s2_join", "join"],
)
# hang triage probes (K2 4n s2, 2026-08-25)
VARIANTS["ours_l01_s2_gate_noshard"] = dict(
    VARIANTS["ours_l01_s1"],
    test_args=["--eps", "0.0625", "--sizing", "capacity",
               "--plan_overlap", "2", "--scenario", "s2",
               "--place_gain_threshold_ppm", "0",
               "--check_iters", "1", "--s2_stale", "rot",
               "--s2_force_trigger", "1", "--s2_wprobe", "1",
               "--weight_shard", "off"],
)
VARIANTS["ours_l01_s2_gate_direct"] = dict(
    VARIANTS["ours_l01_s2_gate_noshard"],
    env=dict(VARIANTS["ours_l01_s1"]["env"], FLUX_OURS_S2_MCAST="0"),
)
VARIANTS["ours_l01_s2_gate_join"] = dict(
    VARIANTS["ours_l01_s1"],
    test_args=["--eps", "0.0625", "--sizing", "capacity",
               "--plan_overlap", "2", "--scenario", "s2",
               "--place_gain_threshold_ppm", "0",
               "--check_iters", "1", "--s2_stale", "rot",
               "--s2_force_trigger", "1", "--s2_wprobe", "1",
               "--s2_join", "join"],
)
# relay-identity wire twin (2026-08-25, 16n b32+ crossover attack): the
# fused pipeline with per-rank-exact delivery (LB_UNION=0 -> balanced
# relay + identity forward) instead of the Tier-B union broadcast. On
# LocCap-placed traffic the union's node-dedup premise is weak (copies
# already rank-consolidated); llc's staged wire is relay-identity and won
# 16n b32/b64 — this arm isolates wire-mode from staged-vs-fused.
# EARLY_LAUNCH default is LB_UNION-conditioned; pinned ON explicitly
# (conn=8 satisfies its guard) so overlap identity is preserved.
VARIANTS["ours_l01_s1_ri"] = dict(
    VARIANTS["ours_l01_s1"],
    env=dict(VARIANTS["ours_l01_s1"]["env"],
             FLUX_A2AV_LB_UNION="0", FLUX_A2AV_EARLY_LAUNCH="1",
             FLUX_A2AV_FUSED_STAGE2="1"),
    requires=[r for r in VARIANTS["ours_l01_s1"]["requires"]
              if r != "FLUX_A2AV_LB_UNION"],
)
VARIANTS["ours_l01_s1_ri_gate"] = dict(
    VARIANTS["ours_l01_s1_ri"],
    test_args=VARIANTS["ours_l01_s1"]["test_args"][:-2]
              + ["--plan_overlap", "2", "--check_iters", "1"],
)
VARIANTS["ours_l01_s2_stale_c32"] = dict(
    VARIANTS["ours_l01_s2_stale"],
    env=dict(VARIANTS["ours_l01_s1"]["env"],
             CUDA_DEVICE_MAX_CONNECTIONS="32"),
)
VARIANTS["ours_l01_s2_gate_c32"] = dict(
    VARIANTS["ours_l01_s2_gate"],
    env=dict(VARIANTS["ours_l01_s1"]["env"],
             CUDA_DEVICE_MAX_CONNECTIONS="32"),
)

# ---- moved-last GEMM reschedule + late-w2 flow scheduling (2026-08-26
# exposed-movement-latency session; eager adoption kept). Knobs:
#   ml  = FLUX_OURS_SCHED_MOVED_LAST=1 — defer THIS iteration's moved
#         slots' problems behind every resident problem in the static
#         schedule (per-iteration moved set; generalizes the retracted
#         NR-14 class reorder, which deferred ALWAYS-resident weights).
#   w2l = FLUX_OURS_S2_W2_LATE=1 — issue the l1 weight pushes after the
#         fused l0 forward is enqueued (dispatch owns the proxy-queue
#         head; w2 runway = l0 + gelu; join_w2 unchanged).
VARIANTS["ours_l01_s2_stale_ml"] = dict(
    VARIANTS["ours_l01_s2_stale"],
    env=dict(VARIANTS["ours_l01_s2_stale"]["env"],
             FLUX_OURS_SCHED_MOVED_LAST="1"),
)
VARIANTS["ours_l01_s2_stale_w2l"] = dict(
    VARIANTS["ours_l01_s2_stale"],
    env=dict(VARIANTS["ours_l01_s2_stale"]["env"],
             FLUX_OURS_S2_W2_LATE="1"),
)
VARIANTS["ours_l01_s2_stale_mlw2"] = dict(
    VARIANTS["ours_l01_s2_stale"],
    env=dict(VARIANTS["ours_l01_s2_stale"]["env"],
             FLUX_OURS_SCHED_MOVED_LAST="1", FLUX_OURS_S2_W2_LATE="1"),
)
# quiet-regime null twin (movement rare: expect no change vs ours_l01_s2)
VARIANTS["ours_l01_s2_mlw2"] = dict(
    VARIANTS["ours_l01_s2"],
    env=dict(VARIANTS["ours_l01_s2"]["env"],
             FLUX_OURS_SCHED_MOVED_LAST="1", FLUX_OURS_S2_W2_LATE="1"),
)
# strict gates (check_iters + stale rot + weight payload probe)
VARIANTS["ours_l01_s2_gate_ml"] = dict(
    VARIANTS["ours_l01_s2_gate"],
    env=dict(VARIANTS["ours_l01_s2_gate"]["env"],
             FLUX_OURS_SCHED_MOVED_LAST="1"),
)
VARIANTS["ours_l01_s2_gate_mlw2"] = dict(
    VARIANTS["ours_l01_s2_gate"],
    env=dict(VARIANTS["ours_l01_s2_gate"]["env"],
             FLUX_OURS_SCHED_MOVED_LAST="1", FLUX_OURS_S2_W2_LATE="1"),
)
# v2-combine mechanism ablations (2026-08-25, 16n b64 l1 gap vs llc's
# plain staged wait-all): one knob off each, canon otherwise. Decide the
# single shipping combine config; ablations never ship as knobs.
for _k, _env in (("noms", {"FLUX_A2AV_RS_MSPLIT": "0"}),
                 ("nofp", {"FLUX_A2AV_RS_FUSED_PACK": "0"}),
                 ("nobkt", {"FLUX_A2AV_RS_RECV_BUCKET": "0"})):
    VARIANTS[f"ours_l01_s1_{_k}"] = dict(
        VARIANTS["ours_l01_s1"],
        env=dict(VARIANTS["ours_l01_s1"]["env"], **_env))
# PRODUCTION-SEMANTICS unified arm (2026-08-25, user direction: s2 as the
# eventual sole OURS branch): same s2 machinery resident, but the drift
# prefilter (10k ppm) and gain threshold (50k ppm) actually GATE — on
# stable 7-baseline traffic the solve is skipped (place ~= drift check),
# under drift it triggers. The threshold-0 ours_l01_s2 stays the
# always-solve upper-bound probe.
VARIANTS["ours_l01_s2_prod"] = dict(
    VARIANTS["ours_l01_s1"],
    test_args=["--eps", "0.0625", "--sizing", "capacity",
               "--plan_overlap", "2", "--scenario", "s2"],
)
# 16n-loss RCA probe twins (2026-08-25 pm campaign): one mechanism knob each,
# canon otherwise. pull = dispatch relay pull/put stream decoupling (H3: the
# round-1 blocking put queues FIFO behind all NN-1 rounds' NVLink relay pulls
# on cp_stream_inter_node; knob shipped in the binary, default OFF). cta =
# combine CTA partition 10/8/6 -> 14/12/10 (H2: fixed grids do budget-
# proportional prered/reduce/pack work; the 3/3/2->10/8/6 flip's dose-response
# was still rising at b64). combo = pull + cta + fused-pack OFF (H5) stacked.
VARIANTS["ours_l01_s1_pull"] = dict(
    VARIANTS["ours_l01_s1"],
    env=dict(VARIANTS["ours_l01_s1"]["env"],
             FLUX_A2AV_RELAY_PULL_STREAM="1"),
)
VARIANTS["ours_l01_s1_cta"] = dict(
    VARIANTS["ours_l01_s1"],
    env=dict(VARIANTS["ours_l01_s1"]["env"],
             FLUX_A2AV_RS_PACK_BLOCKS="14",
             FLUX_A2AV_RS_REDUCE_BLOCKS="12",
             FLUX_A2AV_RS_PRERED_BLOCKS="10"),
)
VARIANTS["ours_l01_s1_combo"] = dict(
    VARIANTS["ours_l01_s1"],
    env=dict(VARIANTS["ours_l01_s1"]["env"],
             FLUX_A2AV_RELAY_PULL_STREAM="1",
             FLUX_A2AV_RS_PACK_BLOCKS="14",
             FLUX_A2AV_RS_REDUCE_BLOCKS="12",
             FLUX_A2AV_RS_PRERED_BLOCKS="10",
             FLUX_A2AV_RS_FUSED_PACK="0"),
)
# correctness gates for the probe knobs (rule 6: pull re-streams the relay
# pulls — ordering must be re-proven under random payload before adoption)
VARIANTS["ours_l01_s1_pull_gate"] = dict(
    VARIANTS["ours_l01_s1_pull"],
    test_args=VARIANTS["ours_l01_s1"]["test_args"] + ["--check_iters", "1"],
)
VARIANTS["ours_l01_s1_combo_gate"] = dict(
    VARIANTS["ours_l01_s1_combo"],
    test_args=VARIANTS["ours_l01_s1"]["test_args"] + ["--check_iters", "1"],
)
# full stack candidate: combo (pull + cta 14/12/10 + fp-off) + planfast
_ALLIN_ENV = dict(VARIANTS["ours_l01_s1_combo"]["env"],
                  FLUX_OURS_PLAN_XCHG_NARROW="2",
                  FLUX_OURS_PLAN_GRAPH="1", FLUX_OURS_PLAN_SCALE_GRAPH="1")
VARIANTS["ours_l01_s1_allin"] = dict(
    VARIANTS["ours_l01_s1"], env=dict(_ALLIN_ENV),
)
VARIANTS["ours_l01_s1_gate_allin"] = dict(
    VARIANTS["ours_l01_s1_gate"], env=dict(_ALLIN_ENV),
)
# H4 arrival-dynamic combine receiver (post-rebuild only: the tag in
# `requires` makes these arms un-runnable on binaries without the kernel)
_DYN_REQUIRES = _OURS_REQUIRES + ["FLUX_A2AV_RS_RECV_DYN_V2_TAG"]
VARIANTS["ours_l01_s1_dyn"] = dict(
    VARIANTS["ours_l01_s1"],
    env=dict(VARIANTS["ours_l01_s1"]["env"], FLUX_A2AV_RS_RECV_DYN="1"),
    requires=list(_DYN_REQUIRES),
)
# full candidate stack + dyn receiver
VARIANTS["ours_l01_s1_next"] = dict(
    VARIANTS["ours_l01_s1"],
    env=dict(_ALLIN_ENV, FLUX_A2AV_RS_RECV_DYN="1"),
    requires=list(_DYN_REQUIRES),
)
VARIANTS["ours_l01_s1_gate_dyn"] = dict(
    VARIANTS["ours_l01_s1_gate"],
    env=dict(VARIANTS["ours_l01_s1"]["env"], FLUX_A2AV_RS_RECV_DYN="1"),
    requires=list(_DYN_REQUIRES),
)
VARIANTS["ours_l01_s1_gate_next"] = dict(
    VARIANTS["ours_l01_s1_gate"],
    env=dict(_ALLIN_ENV, FLUX_A2AV_RS_RECV_DYN="1"),
    requires=list(_DYN_REQUIRES),
)
# dyn + own-first production (own-lane prered emitted FIRST so token
# completion gates on remote arrivals — the regime where arrival-order
# folding pays; own-last is the muting case)
VARIANTS["ours_l01_s1_dynof"] = dict(
    VARIANTS["ours_l01_s1"],
    env=dict(VARIANTS["ours_l01_s1"]["env"], FLUX_A2AV_RS_RECV_DYN="1",
             FLUX_A2AV_RS_OWN_FIRST="1"),
    requires=list(_DYN_REQUIRES),
)
VARIANTS["ours_l01_s1_nextof"] = dict(
    VARIANTS["ours_l01_s1"],
    env=dict(_ALLIN_ENV, FLUX_A2AV_RS_RECV_DYN="1",
             FLUX_A2AV_RS_OWN_FIRST="1"),
    requires=list(_DYN_REQUIRES),
)
VARIANTS["ours_l01_s1_gate_nextof"] = dict(
    VARIANTS["ours_l01_s1_gate"],
    env=dict(_ALLIN_ENV, FLUX_A2AV_RS_RECV_DYN="1",
             FLUX_A2AV_RS_OWN_FIRST="1"),
    requires=list(_DYN_REQUIRES),
)
# combine-wire-balanced placement finalize (2026-08-25 nsys: per-rank
# NVSHMEM proxy serialization + within-node outbound skew 272/121/53/15 MB
# — placelambda_fast._finalize_hosts_wirebal; node membership unchanged,
# within-node rank identity re-assigned by remote-served-rows LPT)
VARIANTS["ours_l01_s1_wb"] = dict(
    VARIANTS["ours_l01_s1"],
    env=dict(VARIANTS["ours_l01_s1"]["env"],
             FLUX_OURS_PLACE_WIREBAL="1"),
)
VARIANTS["ours_l01_s1_nextwb"] = dict(
    VARIANTS["ours_l01_s1"],
    env=dict(_ALLIN_ENV, FLUX_A2AV_RS_RECV_DYN="1",
             FLUX_OURS_PLACE_WIREBAL="1"),
    requires=list(_DYN_REQUIRES),
)
VARIANTS["ours_l01_s1_gate_wb"] = dict(
    VARIANTS["ours_l01_s1_gate"],
    env=dict(VARIANTS["ours_l01_s1"]["env"],
             FLUX_OURS_PLACE_WIREBAL="1"),
)
VARIANTS["ours_l01_s1_gate_nextwb"] = dict(
    VARIANTS["ours_l01_s1_gate"],
    env=dict(_ALLIN_ENV, FLUX_A2AV_RS_RECV_DYN="1",
             FLUX_OURS_PLACE_WIREBAL="1"),
    requires=list(_DYN_REQUIRES),
)
# shipping-candidate stack WITHOUT the dyn kernel (dyn hangs at 16n b32+,
# open defect 2026-08-25 pm): pull + cta + fp-off + planfast + wirebal
VARIANTS["ours_l01_s1_allinwb"] = dict(
    VARIANTS["ours_l01_s1"],
    env=dict(_ALLIN_ENV, FLUX_OURS_PLACE_WIREBAL="1"),
)
VARIANTS["ours_l01_s1_gate_allinwb"] = dict(
    VARIANTS["ours_l01_s1_gate"],
    env=dict(_ALLIN_ENV, FLUX_OURS_PLACE_WIREBAL="1"),
)
# replica-headroom PARITY probe (2026-08-25 finding: EPIC/EPLB/llc all run
# --redundant_per_rank 2 while OURS hardcoded R_red=0 — the only arm with
# zero replica slots; at qwen 16n that means 128 slots for 128 experts and
# no way to split the ~1GB hot-expert proxy load that llc's solver CAN
# split. r2 = same stack at slot parity; canon stays R_red=0 until user
# ruling (one-decision rule).
VARIANTS["ours_l01_s1_r2"] = dict(
    VARIANTS["ours_l01_s1"],
    test_args=VARIANTS["ours_l01_s1"]["test_args"]
              + ["--redundant_per_rank", "2"],
)
VARIANTS["ours_l01_s1_allinwb_r2"] = dict(
    VARIANTS["ours_l01_s1_allinwb"],
    test_args=VARIANTS["ours_l01_s1"]["test_args"]
              + ["--redundant_per_rank", "2"],
)
VARIANTS["ours_l01_s1_gate_r2"] = dict(
    VARIANTS["ours_l01_s1_gate"],
    test_args=VARIANTS["ours_l01_s1_gate"]["test_args"]
              + ["--redundant_per_rank", "2"],
)
VARIANTS["ours_l01_s1_gate_allinwb_r2"] = dict(
    VARIANTS["ours_l01_s1_gate_allinwb"],
    test_args=VARIANTS["ours_l01_s1_gate_allinwb"]["test_args"]
              + ["--redundant_per_rank", "2"],
)
# wirebal ONLY + replica parity (2026-08-25 user direction: expected big win
# = wirebal/dyn, not the knob stack; this is the minimal-mechanism record
# candidate — placement-driven, canonical combine knobs untouched)
VARIANTS["ours_l01_s1_wb_r2"] = dict(
    VARIANTS["ours_l01_s1_wb"],
    test_args=VARIANTS["ours_l01_s1"]["test_args"]
              + ["--redundant_per_rank", "2"],
)
VARIANTS["ours_l01_s1_gate_wb_r2"] = dict(
    VARIANTS["ours_l01_s1_gate_wb"],
    test_args=VARIANTS["ours_l01_s1_gate_wb"]["test_args"]
              + ["--redundant_per_rank", "2"],
)
# dyn-v2 receiver stacked on the parity winner (dyn effect measurement)
VARIANTS["ours_l01_s1_wb_r2_dyn"] = dict(
    VARIANTS["ours_l01_s1_wb_r2"],
    env=dict(VARIANTS["ours_l01_s1_wb_r2"]["env"],
             FLUX_A2AV_RS_RECV_DYN="1"),
    requires=list(_DYN_REQUIRES),
)
VARIANTS["ours_l01_s1_gate_wb_r2_dyn"] = dict(
    VARIANTS["ours_l01_s1_gate_wb_r2"],
    env=dict(VARIANTS["ours_l01_s1_gate_wb_r2"]["env"],
             FLUX_A2AV_RS_RECV_DYN="1"),
    requires=list(_DYN_REQUIRES),
)
# scenario-2 at replica parity (2026-08-26): the r2 counterpart of
# ours_l01_s2 (always-solve threshold-0 quiet — place lane timed every
# iteration) for the slack-parity record campaign. Same never-mix
# boundary as the s1 r2 arms: slack-2 cells never compare to R_red=0.
VARIANTS["ours_l01_s2_r2"] = dict(
    VARIANTS["ours_l01_s2"],
    test_args=VARIANTS["ours_l01_s2"]["test_args"]
              + ["--redundant_per_rank", "2"],
)
VARIANTS["ours_l01_s2_gate_r2"] = dict(
    VARIANTS["ours_l01_s2_gate"],
    test_args=VARIANTS["ours_l01_s2_gate"]["test_args"]
              + ["--redundant_per_rank", "2"],
)

# moved-last / late-w2 x r2 (2026-08-26 session): movement scheduling
# validated + measured at replica parity (the production-shaped regime).
VARIANTS["ours_l01_s2_gate_r2_ml"] = dict(
    VARIANTS["ours_l01_s2_gate_r2"],
    env=dict(VARIANTS["ours_l01_s2_gate_r2"]["env"],
             FLUX_OURS_SCHED_MOVED_LAST="1"),
)
VARIANTS["ours_l01_s2_gate_r2_mlw2"] = dict(
    VARIANTS["ours_l01_s2_gate_r2"],
    env=dict(VARIANTS["ours_l01_s2_gate_r2"]["env"],
             FLUX_OURS_SCHED_MOVED_LAST="1", FLUX_OURS_S2_W2_LATE="1"),
)
# stale (movement-every-iteration) at replica parity — the pll worst-case
# comparator for the pv2 A/B (2026-08-27, branch pv2)
VARIANTS["ours_l01_s2_stale_r2"] = dict(
    VARIANTS["ours_l01_s2_stale"],
    test_args=VARIANTS["ours_l01_s2_stale"]["test_args"]
              + ["--redundant_per_rank", "2"],
)
for _k, _env in (("ml", {"FLUX_OURS_SCHED_MOVED_LAST": "1"}),
                 ("w2l", {"FLUX_OURS_S2_W2_LATE": "1"}),
                 ("mlw2", {"FLUX_OURS_SCHED_MOVED_LAST": "1",
                           "FLUX_OURS_S2_W2_LATE": "1"})):
    VARIANTS[f"ours_l01_s2_stale_r2_{_k}"] = dict(
        VARIANTS["ours_l01_s2_stale_r2"],
        env=dict(VARIANTS["ours_l01_s2_stale_r2"]["env"], **_env))
# movement-ISSUE cost arms (2026-08-27, K2 4n finding: ~110ms host
# enqueue of shard-chunk stream ops in stale place_ms):
#   noshard  = --weight_shard off (at high move counts inter-expert
#              parallelism already covers the NICs; sharding is redundant
#              16x enqueue volume)
#   bigchunk = FLUX_OURS_S2_SHARD_CHUNK=8MB (4x fewer chunk ops, keeps
#              the /L NIC split)
VARIANTS["ours_l01_s2_stale_noshard"] = dict(
    VARIANTS["ours_l01_s2_stale"],
    test_args=VARIANTS["ours_l01_s2_stale"]["test_args"]
              + ["--weight_shard", "off"],
)
VARIANTS["ours_l01_s2_stale_bigchunk"] = dict(
    VARIANTS["ours_l01_s2_stale"],
    env=dict(VARIANTS["ours_l01_s2_stale"]["env"],
             FLUX_OURS_S2_SHARD_CHUNK=str(8 << 20)),
)
# round-4 PULL movement (2026-08-27): FLUX_OURS_S2_PULL=1 — destination
# getmem + local epoch SET, tokens-first issue (gets enqueue after the
# fused l0 forward), no gateway/shard chain, no remote signals. Compare
# within-regime against the push champion (noshard).
VARIANTS["ours_l01_s2_gate_pull"] = dict(
    VARIANTS["ours_l01_s2_gate"],
    env=dict(VARIANTS["ours_l01_s2_gate"]["env"], FLUX_OURS_S2_PULL="1"),
)
VARIANTS["ours_l01_s2_stale_pull"] = dict(
    VARIANTS["ours_l01_s2_stale"],
    env=dict(VARIANTS["ours_l01_s2_stale"]["env"], FLUX_OURS_S2_PULL="1"),
)
VARIANTS["ours_l01_s2_stale_r2_pull"] = dict(
    VARIANTS["ours_l01_s2_stale_r2"],
    env=dict(VARIANTS["ours_l01_s2_stale_r2"]["env"],
             FLUX_OURS_S2_PULL="1"),
)
# oracle-drift probe (2026-08-27 user experiment): resident reset to the
# PRE-BATCH ORACLE solve every iteration -> each timed window re-moves
# exactly the oracle->batch drift set (expected: few experts). Question:
# is SMALL movement completely overlapped (premium vs quiet ~ 0)?
VARIANTS["ours_l01_s2_oracle"] = dict(
    VARIANTS["ours_l01_s2"],
    test_args=VARIANTS["ours_l01_s2"]["test_args"]
              + ["--s2_stale", "oracle", "--s2_force_trigger", "1"],
)
VARIANTS["ours_l01_s2_oracle_noshard"] = dict(
    VARIANTS["ours_l01_s2_oracle"],
    test_args=VARIANTS["ours_l01_s2_oracle"]["test_args"]
              + ["--weight_shard", "off"],
)
# minimal-movement oracle probe v2 (2026-08-27): the kb=0 always-regime
# re-derives ~full placement from the oracle seed (362 moves/iter K2 —
# premise-buster, capsules 5bc126dd/7f670907). Threshold=1 leaves the
# ALWAYS regime -> warm solve runs with keep_bonus=90090 stickiness and
# the cover decision; force_trigger adopts whenever adds exist. Expected:
# only drift-DEMANDED moves (the user's 1-2 expert class).
VARIANTS["ours_l01_s2_oracle_kb"] = dict(
    VARIANTS["ours_l01_s2"],
    test_args=["--eps", "0.0625", "--sizing", "capacity",
               "--plan_overlap", "2", "--scenario", "s2",
               "--place_gain_threshold_ppm", "1",
               "--s2_stale", "oracle", "--s2_force_trigger", "1"],
)
VARIANTS["ours_l01_s2_oracle_kb_noshard"] = dict(
    VARIANTS["ours_l01_s2_oracle_kb"],
    test_args=VARIANTS["ours_l01_s2_oracle_kb"]["test_args"]
              + ["--weight_shard", "off"],
)
# kb ladder (v3): kb=0 gave 362 moves/iter, kb=90090 gave 0 — probe the
# drift-demanded band in between (noshard base = canon candidate)
for _kb in ("2000", "10000", "30000"):
    VARIANTS[f"ours_l01_s2_oracle_kb{_kb}_ns"] = dict(
        VARIANTS["ours_l01_s2_oracle_kb_noshard"],
        test_args=VARIANTS["ours_l01_s2_oracle_kb_noshard"]["test_args"]
                  + ["--place_keep_bonus", _kb],
    )
VARIANTS["ours_l01_s2_oracle_pull"] = dict(
    VARIANTS["ours_l01_s2_oracle"],
    env=dict(VARIANTS["ours_l01_s2_oracle"]["env"], FLUX_OURS_S2_PULL="1"),
)
VARIANTS["ours_l01_s2_stale_r2_noshard"] = dict(
    VARIANTS["ours_l01_s2_stale_r2"],
    test_args=VARIANTS["ours_l01_s2_stale_r2"]["test_args"]
              + ["--weight_shard", "off"],
)
VARIANTS["ours_l01_s2_stale_noshard_nocache"] = dict(
    VARIANTS["ours_l01_s2_stale_noshard"],
    env=dict(VARIANTS["ours_l01_s2_stale_noshard"]["env"],
             PYTORCH_NO_CUDA_MEMORY_CACHING="1"),
)
VARIANTS["ours_l01_s2_stale_noshard_direct"] = dict(
    VARIANTS["ours_l01_s2_stale_noshard"],
    env=dict(VARIANTS["ours_l01_s2_stale_noshard"]["env"],
             FLUX_OURS_S2_MCAST="0"),
)
VARIANTS["ours_l01_s2_stale_noshard_dbg"] = dict(
    VARIANTS["ours_l01_s2_stale_noshard"],
    env=dict(VARIANTS["ours_l01_s2_stale_noshard"]["env"],
             FLUX_OURS_S2_DEBUG_SYNC="1"),
)
VARIANTS["ours_l01_s2_stale_r2_bigchunk"] = dict(
    VARIANTS["ours_l01_s2_stale_r2"],
    env=dict(VARIANTS["ours_l01_s2_stale_r2"]["env"],
             FLUX_OURS_S2_SHARD_CHUNK=str(8 << 20)),
)
VARIANTS["ours_l01_s2_stale_bigchunk16"] = dict(
    VARIANTS["ours_l01_s2_stale"],
    env=dict(VARIANTS["ours_l01_s2_stale"]["env"],
             FLUX_OURS_S2_SHARD_CHUNK=str(16 << 20)),
)
VARIANTS["ours_l01_s2_stale_bigchunk_ml"] = dict(
    VARIANTS["ours_l01_s2_stale"],
    env=dict(VARIANTS["ours_l01_s2_stale"]["env"],
             FLUX_OURS_S2_SHARD_CHUNK=str(8 << 20),
             FLUX_OURS_SCHED_MOVED_LAST="1"),
)
VARIANTS["ours_l01_s2_stale_bigchunk_w2l"] = dict(
    VARIANTS["ours_l01_s2_stale"],
    env=dict(VARIANTS["ours_l01_s2_stale"]["env"],
             FLUX_OURS_S2_SHARD_CHUNK=str(8 << 20),
             FLUX_OURS_S2_W2_LATE="1"),
)
VARIANTS["ours_l01_s2_r2_quiet"] = dict(
    VARIANTS["ours_l01_s2"],
    test_args=VARIANTS["ours_l01_s2"]["test_args"]
              + ["--redundant_per_rank", "2"],
)

# ---- PV2 arms (branch pv2, 2026-08-27): stateless node-aware greedy
# placement (flux/testing/placement_v2.py) swapped into the OURS stack
# via --place_solver pv2. Same fused transport, same LocCap routing, same
# WPM movement machinery — the A/B isolates the placement lane (solve +
# decision + adoption tail). NEW ARM FAMILY: never compare pv2 cells
# against pll-placement cells' out_sha (different placements route
# differently); allclose gates bind as always. r2 twins carry the slack
# boundary (never-mix vs R_red=0).
for _base in ("ours_l01_s2", "ours_l01_s2_stale", "ours_l01_s2_gate",
              "ours_l01_s1"):
    _pv2name = _base + "_pv2"
    VARIANTS[_pv2name] = dict(
        VARIANTS[_base],
        test_args=VARIANTS[_base]["test_args"]
                  + ["--place_solver", "pv2"],
    )
    VARIANTS[_pv2name + "_r2"] = dict(
        VARIANTS[_base],
        test_args=VARIANTS[_base]["test_args"]
                  + ["--place_solver", "pv2", "--redundant_per_rank", "2"],
    )
# ---- step-0 combine-floor ablations (2026-08-29 low-budget diagnosis):
# the l1 msplit dest-node row-split re-reads EVERY expert's w2 panel from
# HBM once per wave (make_workspace_kernel builds n_waves x E full-N
# sub-problems) — ~4 weight passes at 4n vs COMET's 1 (column split keeps
# weight traffic invariant). K2 w2 = 764 MB/rank/pass => the measured
# ~1.9 ms l1 GEMM floor; Qwen (126 MB) doesn't feel it. These arms dial
# the pass count on the SAME binary: wn3 = ring waves of 3 nodes (2-3
# passes, rank-dependent), msp0 = msplit+fused-pack off (1 pass, legacy
# single gate; bucket receiver kept), msp0_nb = + bucket off (wait-all —
# the pre-v2 receiver control). DIAGNOSTIC arms — never headline cells.
VARIANTS["ours_l01_s1_pv2_r2_wn3"] = dict(
    VARIANTS["ours_l01_s1_pv2_r2"],
    env=dict(VARIANTS["ours_l01_s1_pv2_r2"]["env"],
             FLUX_A2AV_RS_WAVE_NODES="3"),
)
VARIANTS["ours_l01_s1_pv2_r2_msp0"] = dict(
    VARIANTS["ours_l01_s1_pv2_r2"],
    env=dict(VARIANTS["ours_l01_s1_pv2_r2"]["env"],
             FLUX_A2AV_RS_MSPLIT="0", FLUX_A2AV_RS_FUSED_PACK="0"),
)
VARIANTS["ours_l01_s1_pv2_r2_msp0_nb"] = dict(
    VARIANTS["ours_l01_s1_pv2_r2"],
    env=dict(VARIANTS["ours_l01_s1_pv2_r2"]["env"],
             FLUX_A2AV_RS_MSPLIT="0", FLUX_A2AV_RS_FUSED_PACK="0",
             FLUX_A2AV_RS_BUCKET="0"),
)
# pre-8/29 LEGACY twin (regression A/B comparator): waves-always, no plan
# graphs, inline combine meta — pins every 8/29 canon flip back off.
def _ours_legacy_args(args):
    out = list(args)
    out[out.index("--plan_overlap") + 1] = "0"
    return out


_OURS_LEGACY_ENV = {"FLUX_A2AV_RS_WAVE_ADAPT": "0",
                    "FLUX_OURS_PLAN_GRAPH": "0",
                    "FLUX_OURS_PLAN_SCALE_GRAPH": "0",
                    "FLUX_A2AV_RS_COMBINE_IDX_KERNEL": "0"}
VARIANTS["ours_l01_s1_pv2_r2_legacy"] = dict(
    VARIANTS["ours_l01_s1_pv2_r2"],
    test_args=_ours_legacy_args(VARIANTS["ours_l01_s1_pv2_r2"]["test_args"]),
    env=dict(VARIANTS["ours_l01_s1_pv2_r2"]["env"], **_OURS_LEGACY_ENV),
)
# slipstream legacy twin: only the binary-level flips apply there
VARIANTS["l01_slipstream_legacy"] = dict(
    VARIANTS["l01_slipstream"],
    env=dict(VARIANTS["l01_slipstream"]["env"], **{
        "FLUX_A2AV_RS_WAVE_ADAPT": "0",
        "FLUX_A2AV_RS_COMBINE_IDX_KERNEL": "0"}),
)
# s1 gate twin for the pv2 static path
VARIANTS["ours_l01_s1_gate_pv2_r2"] = dict(
    VARIANTS["ours_l01_s1_gate"],
    test_args=VARIANTS["ours_l01_s1_gate"]["test_args"]
              + ["--place_solver", "pv2", "--redundant_per_rank", "2"],
)
# Byte-adaptive wave collapse (2026-08-29, binary tag
# FLUX_A2AV_RS_WAVE_ADAPT_TAG): per-iteration host rule — run the legacy
# single-gate GEMM (one weight pass) when (n_waves-1) weight re-read bytes
# > 48x the remote combine-wire bytes, keep dest-node waves otherwise.
# Record CANDIDATE at slack parity; canon flip = user decision. Requires
# the 8/29 rebuild (also carries COMBINE_IDX_KERNEL default-on — never
# byte-compare env_json across that boundary; rule 4 binaries).
VARIANTS["ours_l01_s1_pv2_r2_wa"] = dict(
    VARIANTS["ours_l01_s1_pv2_r2"],
    env=dict(VARIANTS["ours_l01_s1_pv2_r2"]["env"],
             FLUX_A2AV_RS_WAVE_ADAPT="48"),
    requires=VARIANTS["ours_l01_s1_pv2_r2"]["requires"]
             + ["FLUX_A2AV_RS_WAVE_ADAPT_TAG",
                "FLUX_A2AV_RS_COMBINE_IDX_KERNEL_TAG"],
)
VARIANTS["ours_l01_s1_gate_pv2_r2_wa"] = dict(
    VARIANTS["ours_l01_s1_gate_pv2_r2"],
    env=dict(VARIANTS["ours_l01_s1_gate_pv2_r2"]["env"],
             FLUX_A2AV_RS_WAVE_ADAPT="48",
             FLUX_A2AV_RS_CHECK_IDENTITY="1"),
    requires=VARIANTS["ours_l01_s1_pv2_r2"]["requires"]
             + ["FLUX_A2AV_RS_WAVE_ADAPT_TAG",
                "FLUX_A2AV_RS_COMBINE_IDX_KERNEL_TAG"],
)


def _ours_ov_args(args):
    """test_args with --plan_overlap flipped 0 -> 1 (s1-only re-exploration
    2026-08-29, user-directed: the 8/25 removal was the s2 x movement race;
    the s1 ov path was gate-green + perf-positive — handoff 20)."""
    out = list(args)
    i = out.index("--plan_overlap")
    out[i + 1] = "1"
    return out


# plan-lane re-exploration arms (2026-08-29, current binary, python-only):
# _ov  = combine-meta derive + scale build on a side stream under the fused
#        l0 (--plan_overlap 1; residue lands in l1_ms honestly — module
#        timing contract). s1 ONLY: s2 x ov stays BANNED until the movement
#        race is root-caused (8/25 user ruling stands for s2).
# _pf  = lossless plan graphs (FLUX_OURS_PLAN_GRAPH + SCALE_GRAPH; the 8/25
#        planfast stack MINUS the lossy narrow-2 wire).
# _ovpf = both. Factorial vs _wa isolates each effect.
VARIANTS["ours_l01_s1_pv2_r2_wa_ov"] = dict(
    VARIANTS["ours_l01_s1_pv2_r2_wa"],
    test_args=_ours_ov_args(VARIANTS["ours_l01_s1_pv2_r2_wa"]["test_args"]),
)
VARIANTS["ours_l01_s1_pv2_r2_wa_pf"] = dict(
    VARIANTS["ours_l01_s1_pv2_r2_wa"],
    env=dict(VARIANTS["ours_l01_s1_pv2_r2_wa"]["env"],
             FLUX_OURS_PLAN_GRAPH="1", FLUX_OURS_PLAN_SCALE_GRAPH="1"),
)
VARIANTS["ours_l01_s1_pv2_r2_wa_ovpf"] = dict(
    VARIANTS["ours_l01_s1_pv2_r2_wa_pf"],
    test_args=_ours_ov_args(VARIANTS["ours_l01_s1_pv2_r2_wa"]["test_args"]),
)
# _ov2 = LATE plan overlap (2026-08-29 pm, user-directed): combine meta
# issued AFTER the l0 enqueue — host meta work runs under the executing
# GEMM, kernels on the sm_margin headroom via a derive-done event (NOT
# wait_stream). Fixes mode 1's relabeling (host issue before the l0
# launches). Stacked on the graphs (_pf). s1 ONLY (s2 x ov ban stands).
def _ours_ov2_args(args):
    out = list(args)
    out[out.index("--plan_overlap") + 1] = "2"
    return out


VARIANTS["ours_l01_s1_pv2_r2_wa_pfov2"] = dict(
    VARIANTS["ours_l01_s1_pv2_r2_wa_pf"],
    test_args=_ours_ov2_args(
        VARIANTS["ours_l01_s1_pv2_r2_wa_pf"]["test_args"]),
)
# bisect twins for the 8/29 mode-2 GATE slowdown (~4 min/iter vs ~10 s;
# perf cells normal — gate-machinery interaction, not the production path):
# _ov2 = mode 2, graphs OFF, identity ON; _pfov2_noid = mode 2, graphs ON,
# identity OFF. Against _pfov2 (graphs ON, identity ON, the slow one) this
# pins graphs x mode2 vs identity x mode2 vs the check loop itself.
VARIANTS["ours_l01_s1_gate_pv2_r2_wa_ov2"] = dict(
    VARIANTS["ours_l01_s1_gate_pv2_r2_wa"],
    test_args=_ours_ov2_args(
        VARIANTS["ours_l01_s1_gate_pv2_r2_wa"]["test_args"]),
)
VARIANTS["ours_l01_s1_gate_pv2_r2_wa_pfov2_noid"] = dict(
    VARIANTS["ours_l01_s1_gate_pv2_r2_wa"],
    test_args=_ours_ov2_args(
        VARIANTS["ours_l01_s1_gate_pv2_r2_wa"]["test_args"]),
    env=dict(VARIANTS["ours_l01_s1_gate_pv2_r2_wa"]["env"],
             FLUX_OURS_PLAN_GRAPH="1", FLUX_OURS_PLAN_SCALE_GRAPH="1",
             FLUX_A2AV_RS_CHECK_IDENTITY="0"),
)
# route-global restructure (2026-08-29, handoff 26 §4; user-directed
# "merge two allgathers into one"): ONE topk+probs collective + the
# deterministic quota route (route_global_quota) computed by every rank
# for every rank — replaces the d-allgather + relaxed-kernel +
# decisions-allgather chain. s1 only; NEVER-MIX out_sha vs kernel-routed
# arms (different pairing); allclose gates bind. Stacked on the record
# candidate (wa + graphs + late overlap). Gate = mode-2 protocol
# (identity OFF) + per-iteration output checks.
VARIANTS["ours_l01_s1_pv2_r2_wa_pfov2_rg"] = dict(
    VARIANTS["ours_l01_s1_pv2_r2_wa_pfov2"],
    test_args=VARIANTS["ours_l01_s1_pv2_r2_wa_pfov2"]["test_args"]
              + ["--route_global", "1"],
)
VARIANTS["ours_l01_s1_gate_pv2_r2_wa_pfov2_rg"] = dict(
    VARIANTS["ours_l01_s1_gate_pv2_r2_wa_pfov2_noid"],
    test_args=VARIANTS["ours_l01_s1_gate_pv2_r2_wa_pfov2_noid"]["test_args"]
              + ["--route_global", "1"],
    env=dict(VARIANTS["ours_l01_s1_gate_pv2_r2_wa_pfov2_noid"]["env"],
             FLUX_OURS_RG_CHECK="1"),
    requires=VARIANTS["ours_l01_s1_pv2_r2"]["requires"]
             + ["FLUX_PLACELAMBDA_ROUTE_GLOBAL_TAG"],
)
VARIANTS["ours_l01_s1_gate_pv2_r2_wa_pfov2"] = dict(
    VARIANTS["ours_l01_s1_gate_pv2_r2_wa"],
    test_args=_ours_ov2_args(
        VARIANTS["ours_l01_s1_gate_pv2_r2_wa"]["test_args"]),
    env=dict(VARIANTS["ours_l01_s1_gate_pv2_r2_wa"]["env"],
             FLUX_OURS_PLAN_GRAPH="1", FLUX_OURS_PLAN_SCALE_GRAPH="1"),
)
VARIANTS["ours_l01_s1_gate_pv2_r2_wa_ovpf"] = dict(
    VARIANTS["ours_l01_s1_gate_pv2_r2_wa"],
    test_args=_ours_ov_args(
        VARIANTS["ours_l01_s1_gate_pv2_r2_wa"]["test_args"]),
    env=dict(VARIANTS["ours_l01_s1_gate_pv2_r2_wa"]["env"],
             FLUX_OURS_PLAN_GRAPH="1", FLUX_OURS_PLAN_SCALE_GRAPH="1"),
)
# ---- intra-node expert SWAP arms (branch pv2, 2026-08-27; ours_swap.py —
# EPIC §4.3 analog on the OURS stack): per-iteration greedy pair+swap
# INSIDE each node, exchanged over NVLink on the movement stream, NO
# cross-node migration (pv2 adoption lane disabled). The swap decision is
# timed in the place bracket (sub-ms host integer). swap0 = the
# decide-but-never-swap twin (tau=inf): identical machinery + decision
# cost, zero movement — the A/B comparator.
_SWAP_ARGS = ["--eps", "0.0625", "--sizing", "capacity",
              "--plan_overlap", "2", "--scenario", "s2",
              "--place_gain_threshold_ppm", "0",
              "--place_solver", "pv2", "--s2_swap", "1",
              "--redundant_per_rank", "2"]
VARIANTS["ours_l01_s2_swap_r2"] = dict(
    VARIANTS["ours_l01_s1"], test_args=list(_SWAP_ARGS),
)
VARIANTS["ours_l01_s2_swap0_r2"] = dict(
    VARIANTS["ours_l01_s1"],
    test_args=_SWAP_ARGS + ["--swap_tau_rows", "1000000000"],
)
VARIANTS["ours_l01_s2_gate_swap_r2"] = dict(
    VARIANTS["ours_l01_s1"],
    test_args=_SWAP_ARGS + ["--check_iters", "1"],
)
# tau=1 gate twin: forces the NVLink exchange to actually FIRE (the
# canonical tau=512 gate can be vacuous when the oracle->batch drift
# opens no >=tau pair gap — observed K2 4n b8: 0 swaps/iter)
VARIANTS["ours_l01_s2_gate_swap_t1_r2"] = dict(
    VARIANTS["ours_l01_s1"],
    test_args=_SWAP_ARGS + ["--check_iters", "1",
                            "--swap_tau_rows", "1"],
)
# FORCE mode (tau=-1, user direction 2026-08-27): best exchange per pair
# regardless of gain — oscillates at the fixed point, so NVLink movement
# fires EVERY iteration. The always-overlapped probe: compare its
# e2e/total against swap0 (decide-only) in the same capsule; full overlap
# = parity. Gate twin runs it under per-iteration correctness.
VARIANTS["ours_l01_s2_swap_force_r2"] = dict(
    VARIANTS["ours_l01_s1"],
    test_args=_SWAP_ARGS + ["--swap_tau_rows", "-1"],
)
VARIANTS["ours_l01_s2_gate_swap_force_r2"] = dict(
    VARIANTS["ours_l01_s1"],
    test_args=_SWAP_ARGS + ["--check_iters", "1",
                            "--swap_tau_rows", "-1"],
)

# ---- P2P transport + issue-point arms (branch pv2-swap2, 2026-08-28) ----
# The 8.28 4n capsules attributed the whole force-vs-swap0 delta
# (+2.0..2.9 ms) to the HOST apply+issue path (per-pair table loop +
# torch NCCL P2P enqueue). p2p = symmetric-heap staging with node-local
# peer views: one cudaMemcpy over NVLink + zero-SM landed-signal wait;
# apply_swaps vectorized. Issue point: early (place bracket, leads the
# plan derive), late (after the fused l0 enqueue — the reorder probe:
# moved slot's tiles spin until landing, exchange rides under
# dispatch/GEMM), split (w1 early, w2 late). nccl arms above are the
# same-capsule comparators (explicit default, unchanged semantics).
_SWAP_P2P = _SWAP_ARGS + ["--swap_xport", "p2p"]
for _iss, _tag in (("early", "p2p"), ("late", "p2pl"), ("split", "p2ps")):
    VARIANTS[f"ours_l01_s2_swap_force_{_tag}_r2"] = dict(
        VARIANTS["ours_l01_s1"],
        test_args=_SWAP_P2P + ["--swap_issue", _iss,
                               "--swap_tau_rows", "-1"],
    )
    VARIANTS[f"ours_l01_s2_gate_swap_force_{_tag}_r2"] = dict(
        VARIANTS["ours_l01_s1"],
        test_args=_SWAP_P2P + ["--swap_issue", _iss, "--check_iters", "1",
                               "--swap_tau_rows", "-1"],
    )
    VARIANTS[f"ours_l01_s2_swap_{_tag}_r2"] = dict(
        VARIANTS["ours_l01_s1"],
        test_args=_SWAP_P2P + ["--swap_issue", _iss],
    )

# tau=1 swap arms (2026-08-28, topic-shift oracle test): under a REAL
# per-GPU skew (opool= oracle basis) the canonical tau=512 rows is a
# budget-relative threshold (b1: ~585 rows/rank => never fires); tau=1
# accepts any positive-gain exchange — the "does intra-node rebalance pay"
# probe. Comparators in-capsule: s1_pv2_r2 (static on the skewed oracle
# placement), swap0 (decide-only), s2_pv2_r2 (cross-node migration ceiling).
VARIANTS["ours_l01_s2_swap_p2p_t1_r2"] = dict(
    VARIANTS["ours_l01_s1"],
    test_args=_SWAP_P2P + ["--swap_issue", "early", "--swap_tau_rows", "1"],
)
VARIANTS["ours_l01_s2_gate_swap_p2p_t1_r2"] = dict(
    VARIANTS["ours_l01_s1"],
    test_args=_SWAP_P2P + ["--swap_issue", "early", "--swap_tau_rows", "1",
                           "--check_iters", "1"],
)
