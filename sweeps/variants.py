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
    "hier_compress_lb_union": dict(
        comm_pattern="a2av_hier_compress",
        env={"FLUX_A2AV_LB_UNION": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
        requires=["FLUX_A2AV_LB_UNION"],
    ),
    # TEMPORARY A/B arm (2026-08-07, NR-06 re-check on L=4/CXI/realistic
    # traces): lb_union with EAGER per-round gateway forwards — each round's
    # node_sig wait + window puts on its own stream instead of the shipped
    # ascending-round single-stream order, so a late round never head-of-line
    # blocks a later one. Canonicalize (make default or delete knob + this
    # entry) once the eager-vs-ring capsule verdict is in the ledger.
    "hier_compress_lb_union_eager": dict(
        comm_pattern="a2av_hier_compress",
        env={
            "FLUX_A2AV_LB_UNION": "1",
            "FLUX_A2AV_FANOUT": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8",
        },
        requires=["FLUX_A2AV_LB_UNION", "FLUX_A2AV_FANOUT"],
    ),
    # TEMPORARY A/B arms for the fused stage-2 consumer build (2026-08-05):
    # identical wire/gateway semantics to their base variants, but the ATen
    # key/argsort/index_select chain + Tier B gating searchsorted are replaced
    # by the fused sort_util kernels. Canonicalize (flip the knob default,
    # drop these) after the phases + isolated verdict.
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
}
