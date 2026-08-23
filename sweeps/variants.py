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
        test_args=["--transport", "hier_compress", "--placement", "epic",
                   "--groups", "1", "--layers", "l01"],
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
    # WinCast canonicalization (2026-08-23, M4): the RS split wire
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
        ],
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "FLUX_A2AV_RS_WIRE_STREAMS": "2",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_A2AV_RS_PACK_BLOCKS": "10", "FLUX_A2AV_RS_REDUCE_BLOCKS": "8", "FLUX_A2AV_RS_PRERED_BLOCKS": "6"},
        requires=[
            "FLUX_A2AV_LB_UNION",
            "FLUX_A2AV_FUSED_STAGE2",
            "FLUX_A2AV_EARLY_LAUNCH",
            "FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT2_TAG",
            "FLUX_A2AV_RS_CTA_1086_TAG",
            "FLUX_A2AV_RS_MAX_SEND_ROWS",
            "FLUX_A2AV_RS_MAX_CONV_ROWS",
            "FLUX_A2AV_RS_MAX_WIRE_ROWS",
        ],
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
            "FLUX_A2AV_RS_WIRE_STREAMS_DEFAULT2_TAG",
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
        ],
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "FLUX_A2AV_RS_WIRE_STREAMS": "2",
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
        ],
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "FLUX_A2AV_RS_WIRE_STREAMS": "2",
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
        ],
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FUSED_STAGE2": "1",
            "FLUX_A2AV_EARLY_LAUNCH": "1",
            "FLUX_A2AV_RS_WIRE_STREAMS": "2",
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

# WinCast (2026-08-23): the concise canonical name for the dispatch leg of
# the LLC trio — PLACE-lambda (placement), LocCap (routing), WinCast
# (dispatch) — i.e. hier_compress + LB_UNION Tier-B windowed union
# broadcast + FUSED_STAGE2 + EARLY_LAUNCH + split RS wire + rebalanced
# combine CTAs, all binary defaults. Alias KEYS only: cells.csv labels stay
# the historical comm_pattern strings so capsules remain comparable.
VARIANTS["wincast"] = VARIANTS["hier_compress_lb_union"]
VARIANTS["l01_wincast"] = VARIANTS["l01_lbunion_compress"]
