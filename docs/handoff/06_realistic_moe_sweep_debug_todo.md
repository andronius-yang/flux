# Realistic-MoE-trace sweep — debug/fix TODO (2026-08-15)

Handoff from the 4n/8n/16n canonical-baselines sweep on real Qwen3-235B
routing (`mmlu/high_school_world_history`, layer 92, homog, topk=8, G=128,
non-blocking wire). Branch `worktree-realistic-moe-sweep-campaign` (based on
`realistic-moe-input` @ `864af17`), commits `3a040c0`..`19ce9f9`. Capsules:
`20260815-011216_perlmutter_1a0e0037` + `20260815-012546_perlmutter_f390057c`
+ `20260815-012649_perlmutter_1dee50c2` (4n), `20260815-074430_perlmutter_3ccbcb47`
+ `20260815-075822_perlmutter_67595af3` + `20260815-081416_perlmutter_bb881f76`
(8n), `20260815-105733_perlmutter_244825e3` (16n, partial — see #1/#3 below).

This doc is **just the debug queue**. It is not a re-derivation of the sweep
itself — read the capsules' `spec.yaml`/`cells.csv` for exact repro commands.

---

## 1. `moonep_fused_push_auto_gated` crashes at every budget at 16n (W64)

Fails **even with `--skip-correctness`** — not a checker artifact, a real
crash. Zero latency data exists for this arm at 16n. It is the "latest
optimized MoonEP" arm (virtual-expert dispatch + concurrent weight
push/mcast + weight-gated tiles) — currently unusable at the largest scale
tested.

Not yet inspected: the raw stderr from any 16n cell for this variant. Start
there — unlike the `hier_compress_lb_union` hang (below), this fails in
~11s, so it should have a clean traceback, not a spin.

## 2. `allgather` + `moonep_fused_push_auto_gated` fail `torch_allclose` at 16n, deterministically

Confirmed across every budget tested (b1, b2, b4, b8, b16) at 16n/W64 with
correctness ON. **Do not treat "passes under `--skip-correctness`" as a fix**
— that only skips the check, it proves nothing about the underlying compute.
The real signal: `hier_compress_lb_union`, `eplb`, and `moonep_nvshmem_getmem`
all pass the *same* check cleanly on the *same* routing/matrix data at the
same matrix instance. If this were shared data corruption or a cross-session
race, all five arms would fail together. Only two did, consistently — that
points at a genuine arm-specific correctness bug at W=64.

Debug dumps now land in `/tmp` (fixed, commit `19ce9f9`, env override
`FLUX_DEBUG_DUMP_DIR`) instead of crashing on repo-tree disk quota — rerun
one budget with correctness ON and actually inspect the mismatched `x`/`y`/
`moe_ctx` tensors instead of routing around the check.

## 3. `hier_compress_lb_union` hangs at b64/W32 (8n); fails differently at b32+b64/W64 (16n)

At 8n: reproducibly **hangs** (not a crash) — confirmed via two independent
attempts, each killed by the 250s idle watchdog after spinning at 100% GPU.
Capsule `20260815-081416_perlmutter_bb881f76` documents this as a confirmed
negative result (0/1 ok, by design).

At 16n: **fails fast** at both b32 and b64 instead of hanging — a different
symptom. Unclear if same root cause manifesting differently at larger W, or
a distinct bug. This is the flagship "best flux optimization" arm; it breaks
down at exactly the scale/budget combination that matters most.

Before assuming a logic bug: `sweeps/SCHEMA.md` §knobs and
`docs/handoff/01_perlmutter_bringup.md` §8 flag that `FLUX_A2AV_MAX_RECV_NTOKENS`
knob-scaling (`scale_knobs()` in `sweeps/sweep.py`) was anchored at L=8 and
may under-provision at higher node/world counts. Check that formula against
W=32/64 before chasing a logic bug — this exact under-provisioning risk was
predicted, not discovered fresh.

## 4. `fast` fails at b64 at both 4n and 8n (not just the previously-known 16n failure)

Lower priority — already a known-fragile arm (SIGSEGV at 16n/64 ranks was
documented before this campaign). This widens the known instability
envelope down to 4n/8n as well. Not urgent, just don't be surprised by it.

---

## Priority order

1 and 2 first — #1 because a headline arm has zero data at scale, #2 because
it means the 16n `allgather`/`moonep_fused_push_auto_gated` latency numbers
already reported from this campaign are unverified and shouldn't be quoted
as ground truth until someone looks at the actual mismatch. #3 is next most
important (flagship arm, scale-dependent breakdown) but has a concrete
formula to check before assuming a bug. #4 is a documentation-only item.
