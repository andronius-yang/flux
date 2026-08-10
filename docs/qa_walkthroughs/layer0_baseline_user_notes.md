# Layer0 baseline data movement — original notes (user, 2026-07-30)

These are the original notes, verbatim, that the annotated pass in
`layer0_baseline_annotated.md` walks through line by line. Context: after many
rounds of iteration on the layer0 a2av variants, this is an attempt to nail
down the true timeline of data movement in the **baseline** path (Flux's
built-in hierarchical all-gather in `GemmGroupedV2AGScatterOp`), before moving
on to the optimizations.

## Terminology proposed

- `L` = number of local ranks
- `N` = number of nodes
- `R` = total number of ranks

## The notes

> firstly, the input, and output.
>
> in my understanding, at least for layer 0, the producer starts off ready, it
> is a tensor of token embeddings. in our profiling and sweep, this is the
> budget that each rank has. each one of these tokens have passed through the
> gating algorithm and have decided which topk it wants to enter/reside on.
> fundamentally, in our tests, this is passed in as pre-known metadata, but
> actually acquiring it requires a pre-requisite allgather that is not
> separately timed (and relatively cheap, since it doesn't actually move token
> bytes, just metadata of token bytes).
>
> now onto the consumer, the consumer is also a piece of memory in every rank,
> but here, we launch a CUTLASS tiled matrix multiplication. all i know is
> that, when enough rows come and fill up this tensor such that a tile can
> fire, some signaling mechanism sets off the tiling multiplication to proceed
> on these rows. the fact that some of these rows arrive early, and some of
> these rows arrive late, is precisely the overlap that communication brings
> with computation. i know that we do not need to order the tiles in the way
> they are consumed, since the GEMM API allows an index input that will fire
> based on arbitrary ordering in the buffer (e.g. if i pass in [1,3,2,0] into
> the GEMM API, it will treat the logical row 0 as physical row 1, etc.), but
> how we can identify that these rows have arrived is a question of signaling
> and synchronization.
>
> now, onto my explanation of the hierarchical allgather, which is flux's
> default in-built communication pattern. in my understanding, the ordering in
> which different ranks talk to each other is all pre-determined and calculated
> at the start. in fact, this calculation can be done, in parallel, as the gemm
> tiles first process and sift through what is locally available on the local
> rank's producer. then, it schedules L-1 rounds of local transfer, precomputed
> in ring such that there is no incast in the scale-up NVLINK bandwidth. at the
> same time, an inter-node transfer is being conducted by ranks with their
> corresponding local rank id remote nodes. upon receiving, they can start to
> redistribute these tokens locally (not filtering or anything), and so in the
> ordering in which tokens arrive (all tokens arrive), its the local rank
> first, then local node ranks next, then remote ranks after.
>
> what i'm confused here is the actual signaling/synchronization mechanism.
> i.e., if the remote puts are nbi, is there anything stopping the inter-node
> rounds from occurring? after a node issues a put_nbi does it immediately
> issue the next round of nbi? i know that the intra-node redistribution
> depends on the arrival of the corresponding inter-node segment, but how is
> this signaling/synchronization dependency held? lastly, perhaps theres a
> nvshmem barrier that restricts all ranks from progressing. where is this
> installed?
>
> i want to get a better understanding of the baseline first, before i
> continue on to the optimizations.
