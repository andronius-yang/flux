# Realistic-MoE-trace sweep — debug/fix TODO (2026-08-15)

> **STATUS UPDATE (2026-08-15, debug session, branch `worktree-debug`): all four
> items root-caused; #1/#2/#3 fixed in code.** Summary — details in the
> per-item annotations below:
>
> - **#1 and #2 are ONE kernel bug**: the `process_tile` source-segment gate
>   (`src/moe_ag_scatter/cutlass_impls/ag_scatter_gemm_grouped_with_absmax.h`)
>   built its wait range from a single-warp 32-lane ballot. At W=64 any tile
>   whose rows extend past `split_accum[31]` got an empty `m_end` ballot
>   (`segment_end == -1`) and **waited for nothing** — including its sources
>   < 32 — so the allgather arm read zero-cleared unarrived rows (the exact
>   band/source pattern was reproduced offline from the routing files:
>   b8 zero rows = sources 5,6 = the last-fetched node in ring order; b1 =
>   the fetch frontier, sources 31-34). moonep_fused's `assert W <= 32`
>   guarded the same cliff — its "crash" was that designed precondition, and
>   it never reached `torch_allclose` (item #2's moonep half was a
>   mischaracterization; the deterministic allclose failure was
>   allgather-only). FIXED: 64-bit segment masks from two ballots + strided
>   spin loop; host `FLUX_CHECK(world_size <= 64)`; test assert relaxed to 64.
> - **#3 is the predicted knob under-provisioning**, confirmed exactly:
>   `scale_knobs()` had no W term (~32·T cap) while the lb_union recv demand
>   grows ~W·T. Demands from the failure logs (271685 @W32/b64; 259218 @W64/b32;
>   518504 @W64/b64) are reproduced digit-exact by the new
>   `gen_matrix.a2av_knob_demands`. The 8n "hang" vs 16n "fast fail" split is
>   two check sites for one cause: per-rank `:1147` (one rank throws, 31 spin
>   → watchdog) vs collective `:1262`. FIXED: exact per-cell knob computation
>   (`sweep.exact_scale_knobs`, per-knob sizing, `skipped_capacity` status;
>   SCHEMA.md §knobs rewritten) + the C++ checks made collective and the
>   compress-path `:1147` unit mismatch corrected. GPU-validated at 8n:
>   the b64/W32 cell runs green with RECV=294912 (dealer and real routing).
> - **NEW (found during audit): the sweep retry path silently dropped
>   `routing_mode`** (`sweep.py` retry whitelist), so every retried trace cell
>   — exactly the 8n/b64 and 16n/b32+b64 failures — ran DEALER routing while
>   recorded as part of a real-trace family (`routing_mode=''` in cells.csv is
>   the fingerprint; the demand numbers above are dealer-inflated). FIXED:
>   retries reuse the pristine cell dict; loud guards refuse to run a trace
>   cell with degraded routing. **Audit any capsule with `family=trace AND
>   routing_mode=''` before quoting it as real-trace data.**
> - **#4 revised**: the 4n b64 `fast` failure is an NVSHMEM team-creation
>   error at the 16G symmetric-heap ceiling (`team_internal.cpp:531`); a
>   manual rerun with NCCL_DEBUG=INFO *hung* instead (300 s timeout) — same
>   heap-ceiling territory, still doc-only. The "8n b64" claim is
>   **unsupported**: no such cell exists in any capsule or data-root staging
>   dir (8n `fast` failures on record are b2/b8 from 2026-08-05).
> - **Also explained**: 4n/8n b64 allgather+lb_union correctness-ON failures
>   are torch-REFERENCE-path CUDA OOM (the reference materializes the full
>   unsharded tensor; 8-16 GiB) — that is why `--skip_correctness` "fixed"
>   them; the flux arm was fine. And lb_union at 16n is allclose-clean but
>   never bitwise-exact (`correct_bitwise=0`) — presumed reduction order,
>   unverified, keep in mind when using it as a control arm.

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
