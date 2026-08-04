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
