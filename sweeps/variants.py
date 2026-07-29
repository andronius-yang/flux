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
    "a2av": dict(comm_pattern="a2av", env={}, requires=[]),
    "a2av_ring": dict(comm_pattern="a2av_ring", env={}, requires=[]),
    # hierarchical dispatch family
    "hier": dict(comm_pattern="a2av_hier", env={}, requires=[]),
    # token-dedup wire semantics; balanced inter-node relay is the default
    "hier_compress": dict(
        comm_pattern="a2av_hier_compress",
        env={},
        requires=["FLUX_A2AV_RELAY_IDENTITY"],  # any compress-capable build has this
    ),
    # compress with the fixed same-local-rank relay (design §11 identity wire)
    "hier_compress_identity": dict(
        comm_pattern="a2av_hier_compress",
        env={"FLUX_A2AV_RELAY_IDENTITY": "1"},
        requires=["FLUX_A2AV_RELAY_IDENTITY"],
    ),
    # gateway union broadcast (implies identity wire)
    "hier_compress_union": dict(
        comm_pattern="a2av_hier_compress",
        env={"FLUX_A2AV_UNION_BCAST": "1"},
        requires=["FLUX_A2AV_UNION_BCAST"],
    ),
    # union broadcast + source-side pack overlap (dedicated stream,
    # double-buffered send); MAX_CONNECTIONS=2 lets the pack stream actually
    # run concurrently with compute (the validated best config)
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
