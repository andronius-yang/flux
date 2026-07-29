"""Canonical algorithm-variant table for the sweep runner (sweeps/SCHEMA.md).

A variant is a stable name for one measurable configuration of the layer0
dispatch: the --comm_pattern CLI flag plus any construction-time env knobs.
`requires` lists env-knob strings that must exist in the built libflux_cuda
binary — the runner probes the .so (see sweep.py) and marks cells whose
variant needs an absent knob as skipped_capability instead of running them
against a build that would silently ignore the env.
"""

VARIANTS = {
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
    # double-buffered send)
    "hier_compress_pack": dict(
        comm_pattern="a2av_hier_compress",
        env={"FLUX_A2AV_UNION_BCAST": "1", "FLUX_A2AV_PACK_OVERLAP": "1"},
        requires=["FLUX_A2AV_UNION_BCAST", "FLUX_A2AV_PACK_OVERLAP"],
    ),
}
