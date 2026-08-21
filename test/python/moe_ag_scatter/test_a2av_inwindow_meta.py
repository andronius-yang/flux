################################################################################
#
# v2b in-window metadata derivation vs the python reference, bitwise
# (campaign-2 S6 gate). Exercises GemmGroupedV2AGScatterOp.derive_routed_meta
# on adversarial replicated routings — uniform, heavy skew, empty experts,
# pad-heavy — and run-twice determinism. The stable scatter index is the
# load-bearing property: replicated data means a non-deterministic sort
# (calc_scatter_index) would desynchronize ranks. Run via launch.sh (4 GPUs).
#
################################################################################
import os
from functools import partial

import torch
import torch.distributed

import flux
from flux.testing.moonep_fused_map import (
    FusedMeta,
    _stable_scatter_index,
    required_a2av_knobs,
)

DIST_ENV = flux.get_dist_env()
TP_GROUP = DIST_ENV.get_world()
torch.cuda.set_device(DIST_ENV.LOCAL_RANK)
print = partial(print, flush=True)


def reference_meta(vce: torch.Tensor, S: int, gpe: int, W: int, L: int):
    """The build_epic_hc_bundles metadata recipe (epic_semantics), verbatim."""
    E_virt = W * gpe
    ntokens, K = vce.shape
    home_of_token = torch.arange(ntokens, dtype=torch.int64) // S
    vce_l = vce.long()
    scatter_index = _stable_scatter_index(vce)
    splits = torch.bincount(vce_l.flatten(), minlength=E_virt).int()
    src_of_copy = home_of_token.repeat_interleave(K)
    splits_per_source = (
        torch.bincount(src_of_copy * E_virt + vce_l.flatten(),
                       minlength=W * E_virt)
        .view(W, E_virt).int().contiguous())
    owner = vce_l // gpe
    flags = torch.zeros(ntokens, W, dtype=torch.bool)
    flags.scatter_(1, owner, True)
    u_mat = flags.view(W, S, W).sum(1)
    nn = W // L
    U_mat = flags.view(ntokens, nn, L).any(dim=2).view(W, S, nn).sum(1)
    a2av_unique_counts = torch.cat([u_mat, U_mat], dim=1).int().contiguous()
    m_per_rank = splits.long().view(W, gpe).sum(1)
    return FusedMeta(
        scatter_index=scatter_index,
        splits=splits,
        splits_per_source=splits_per_source,
        a2av_unique_counts=a2av_unique_counts,
        m_per_rank=m_per_rank,
    )


def make_cases(S: int, K: int, W: int, gpe: int):
    """Replicated adversarial routings [W*S, K] (identical on every rank:
    fixed seeds). Pad slot of home h is h*gpe + (gpe-1), the epic pad
    convention; 'live' virtual experts are the first gpe-1 per rank."""
    ntokens = W * S
    E_virt = W * gpe
    nlp = gpe - 1
    home = torch.arange(ntokens, dtype=torch.int64) // S
    pad_vslot = (home * gpe + nlp).unsqueeze(1)
    live = torch.tensor(
        [r * gpe + s for r in range(W) for s in range(nlp)])
    cases = {}
    g = torch.Generator().manual_seed(20260820)
    cases["uniform"] = live[
        torch.randint(len(live), (ntokens, K), generator=g)]
    skew = live[torch.randint(len(live), (ntokens, K), generator=g)]
    hot = live[0]
    mask = torch.rand(ntokens, K, generator=g) < 0.9
    cases["skew90"] = torch.where(mask, hot.expand_as(skew), skew)
    # empty experts: only every 4th live slot is ever routed to
    sparse = live[::4]
    cases["empty_experts"] = sparse[
        torch.randint(len(sparse), (ntokens, K), generator=g)]
    # pad-heavy: ~80% of entries are the home pad slot (the shape the epic
    # migration produces when groups drain)
    padh = live[torch.randint(len(live), (ntokens, K), generator=g)]
    mask = torch.rand(ntokens, K, generator=g) < 0.8
    cases["pad_heavy"] = torch.where(mask, pad_vslot.expand_as(padh), padh)
    return {k: v.int() for k, v in cases.items()}


if __name__ == "__main__":
    flux.init_flux_shm(TP_GROUP)
    torch.cuda.synchronize()
    W = DIST_ENV.WORLD_SIZE
    L = DIST_ENV.LOCAL_WORLD_SIZE
    nnodes = max(W // L, 1)
    S, K, gpe, H = 512, 4, 5, 256
    E_virt = W * gpe
    ntokens = W * S

    cases = make_cases(S, K, W, gpe)
    # capacity knobs: max demand across cases, exact (no headroom needed —
    # contents vary per call, allocation is one-shot)
    knobs = {}
    for vce in cases.values():
        m = reference_meta(vce, S, gpe, W, L)
        for k, v in required_a2av_knobs(m, W, L).items():
            knobs[k] = max(knobs.get(k, 0), int(v))
    for k, v in knobs.items():
        os.environ[k] = str(v)

    # EP == world (tp 1), the epic hc configuration
    ep_group = DIST_ENV.new_group(list(range(W)))
    tp_env = flux.DistEnvTPWithEP(tp_group=TP_GROUP, nnodes=nnodes,
                                  ep_group=ep_group)
    moe_args = flux.MoeArguments(
        max_ntokens=ntokens, hidden=H, ffn_hidden=H,
        nexperts=E_virt, topk=K,
        input_dtype=torch.bfloat16, output_dtype=torch.bfloat16)
    op = flux.GemmGroupedV2AGScatterOp(
        tp_env=tp_env, moe_args=moe_args,
        a2av_dispatch=True, a2av_hier_compress=True)

    rank = TP_GROUP.rank()
    for name, vce in cases.items():
        ref = reference_meta(vce, S, gpe, W, L)
        vce_dev = vce.cuda()
        sd, scd, sps, uc = op.derive_routed_meta(vce_dev)
        sd2, scd2, sps2, uc2 = op.derive_routed_meta(vce_dev)
        assert torch.equal(scd, scd2) and torch.equal(sd, sd2) and \
            torch.equal(sps, sps2) and torch.equal(uc, uc2), (
                f"{name}: derive_routed_meta is non-deterministic")
        assert torch.equal(sd.cpu(), ref.splits), f"{name}: splits"
        assert torch.equal(scd.cpu(), ref.scatter_index), (
            f"{name}: stable scatter index != python argsort(stable) ref")
        assert torch.equal(sps, ref.splits_per_source), (
            f"{name}: splits_per_source")
        assert torch.equal(uc, ref.a2av_unique_counts), (
            f"{name}: a2av_unique_counts")
        if rank == 0:
            print(f"  case {name}: bitwise ok (+determinism)")
    TP_GROUP.barrier()
    if rank == 0:
        print(f"✅ in-window meta derivation bitwise vs python reference, "
              f"{len(cases)} cases x run-twice, W={W} nnodes={nnodes}")
