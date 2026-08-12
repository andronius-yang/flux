"""Dedup-plan set-comparison and planning invariants for the MoonEP port.

_dedup_group_map / dedup_plan_semantic_errors are vendored verbatim from
MoonshotAI/MoonEP @ 7745ffa (tests/kernel_test_utils.py): MoonEP's dedup
structure ordering is documented-unstable, so equality is defined on the
{primary_loff: sorted duplicate loffs} group map, never on element order.

planning_invariant_errors_lite is the same file's planning_invariant_errors
minus the MoonEP meta_buf layout-offset checks (TOPK0_OFF/ORDER_OFF/...),
which describe the NVLink scratch buffer the port does not have.
"""

import torch

DEDUP_PLAN_FIELDS = (
    "dup_groups",
    "dup_loffs",
    "dup_counts",
)


def _dedup_group_map(plan, *, max_print=5):
    """Build {primary_loff: sorted duplicate loff tuple} plus internal
    consistency errors. The builder allocates the compact prefixes with
    per-warp atomicAdds, so ``dup_groups`` / ``dup_loffs`` ordering is not
    stable — only the group set is compared."""
    groups = plan.dup_groups.cpu()
    dup_loffs = plan.dup_loffs.cpu()
    counts = plan.dup_counts.cpu()
    NvS = dup_loffs.numel()
    errors = []
    mapping = {}

    group_count = int(counts[0].item())
    dup_count_total = int(counts[1].item())
    if group_count < 0 or group_count > NvS:
        errors.append(f"dup group count {group_count} out of range [0, {NvS}]")
        group_count = max(0, min(group_count, NvS))
    if dup_count_total < 0 or dup_count_total > NvS:
        errors.append(f"dup loff count {dup_count_total} out of range [0, {NvS}]")
        dup_count_total = max(0, min(dup_count_total, NvS))

    seen_dups = []
    for group_idx in range(group_count):
        primary, dup_start, dup_count = (
            int(v) for v in groups[group_idx].tolist()
        )
        if not (0 <= primary < NvS):
            errors.append(f"dup group {group_idx} primary {primary} out of range")
            continue
        if not (0 <= dup_start <= dup_count_total):
            errors.append(f"dup group {group_idx} dup_start {dup_start} out of range")
            continue
        if dup_count <= 0 or dup_start + dup_count > dup_count_total:
            errors.append(
                f"dup group {group_idx} invalid dup range "
                f"start={dup_start} count={dup_count} total={dup_count_total}"
            )
            continue
        dups = []
        for offset_t in dup_loffs[dup_start:dup_start + dup_count].tolist():
            offset = int(offset_t)
            if not (0 <= offset < NvS):
                errors.append(
                    f"dup group {group_idx} contains out-of-range dup loff {offset}"
                )
                continue
            dups.append(offset)
            seen_dups.append(offset)
        if primary in mapping:
            errors.append(f"primary loff {primary} appears in multiple dup groups")
        if primary in dups:
            errors.append(f"dup group {group_idx} lists its own primary {primary}")
        mapping[primary] = tuple(sorted(dups))

    seen_dup_set = set(seen_dups)
    if len(seen_dups) != len(seen_dup_set):
        errors.append("dup_loffs contains repeated duplicate rows")
    if len(seen_dups) != dup_count_total:
        errors.append(
            f"dup loff count mismatch: header={dup_count_total}, used={len(seen_dups)}"
        )
    overlap = seen_dup_set & set(mapping.keys())
    if overlap:
        errors.append(
            f"rows appear both as primary and duplicate: {sorted(overlap)[:max_print]}"
        )

    return mapping, errors


def dedup_plan_semantic_errors(name, actual, expected, max_print=5):
    errors = []
    for field in DEDUP_PLAN_FIELDS:
        actual_t = getattr(actual, field)
        expected_t = getattr(expected, field)
        if actual_t.dtype != expected_t.dtype:
            errors.append(
                f"{field} dtype actual={actual_t.dtype} expected={expected_t.dtype}"
            )
        if tuple(actual_t.shape) != tuple(expected_t.shape):
            errors.append(
                f"{field} shape actual={tuple(actual_t.shape)} "
                f"expected={tuple(expected_t.shape)}"
            )
        if errors:
            return errors

    actual_counts = actual.dup_counts.cpu()
    expected_counts = expected.dup_counts.cpu()
    for idx, label in ((0, "dup group count"), (1, "dup loff count")):
        a = int(actual_counts[idx].item())
        e = int(expected_counts[idx].item())
        if a != e:
            errors.append(f"{name} {label} actual={a} expected={e}")

    actual_map, group_errors = _dedup_group_map(actual, max_print=max_print)
    expected_map, expected_group_errors = _dedup_group_map(
        expected, max_print=max_print
    )
    errors.extend(f"{name} actual {err}" for err in group_errors[:max_print])
    errors.extend(f"{name} expected {err}" for err in expected_group_errors[:max_print])
    if actual_map != expected_map:
        actual_keys = set(actual_map)
        expected_keys = set(expected_map)
        missing = sorted(expected_keys - actual_keys)[:max_print]
        extra = sorted(actual_keys - expected_keys)[:max_print]
        mismatched = [
            key for key in sorted(actual_keys & expected_keys)
            if actual_map[key] != expected_map[key]
        ][:max_print]
        errors.append(
            f"{name} duplicate group map differs: "
            f"missing={missing} extra={extra} mismatched={mismatched}"
        )

    return errors


def planning_invariant_errors_lite(cfg, dst, cu_seqlens, experts_to_copy):
    """MoonEP planning invariants on the port's tensors (no meta_buf layout).

    cfg needs: R, E, B, NvS, N (= S*K), token_padding.
    dst is one rank's [N] encoded dst; cu_seqlens one rank's [E+B];
    experts_to_copy the full [R, B].
    """
    errors = []
    R, E, B, NvS, N = cfg.R, cfg.E, cfg.B, cfg.NvS, cfg.N
    token_padding = cfg.token_padding

    if dst.dtype != torch.int32 or tuple(dst.shape) != (N,):
        errors.append(f"dst must be int32 [{N}], got {dst.dtype} {tuple(dst.shape)}")
    else:
        dst_cpu = dst.cpu()
        raw_dst = torch.where(dst_cpu < 0, -dst_cpu - 1, dst_cpu)
        dest_rank = torch.div(raw_dst, NvS, rounding_mode="floor")
        local_off = raw_dst % NvS
        if not torch.all((dest_rank >= 0) & (dest_rank < R)):
            errors.append("dst contains an out-of-range destination rank")
        if not torch.all((local_off >= 0) & (local_off < NvS)):
            errors.append("dst contains an out-of-range local offset")

    cu_cpu = cu_seqlens.cpu()
    if cu_seqlens.dtype != torch.int32 or tuple(cu_seqlens.shape) != (E + B,):
        errors.append(
            f"cu_seqlens must be int32 [{E + B}], "
            f"got {cu_seqlens.dtype} {tuple(cu_seqlens.shape)}"
        )
    else:
        prev = 0
        for gid, cur_t in enumerate(cu_cpu.tolist()):
            cur = int(cur_t)
            seg_len = cur - prev
            if seg_len < 0:
                errors.append(f"cu_seqlens decreases at group {gid}")
                break
            if seg_len and seg_len % token_padding != 0:
                errors.append(
                    f"group {gid} segment length {seg_len} is not divisible "
                    f"by token_padding={token_padding}"
                )
                break
            prev = cur
        if int(cu_cpu[-1].item()) > NvS:
            errors.append(f"cu_seqlens total {int(cu_cpu[-1].item())} exceeds NvS={NvS}")

    copy_cpu = experts_to_copy.cpu()
    if experts_to_copy.dtype != torch.int32 or tuple(experts_to_copy.shape) != (R, B):
        errors.append(
            f"experts_to_copy must be int32 [{R}, {B}], "
            f"got {experts_to_copy.dtype} {tuple(experts_to_copy.shape)}"
        )
    elif not torch.all((copy_cpu == -1) | ((copy_cpu >= 0) & (copy_cpu < E))):
        errors.append("experts_to_copy contains invalid expert ids")

    return errors
