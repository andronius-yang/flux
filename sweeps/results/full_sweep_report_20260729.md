# Full layer0 sweep — 2026-07-29, 2n × 8× A100 (EFA), remotefrac topk=8 G=128

Capsules (committed on `consolidate-sweeps`):
- `20260729-152424_aws_b4b75d99` — b2–32, e2e+phases, torch reference ON (unfused decomposition + correctness)
- `20260729-155138_aws_27585c83` — b64, e2e+phases, `--skip-correctness` (512 MiB-row reference OOMs a 40GB A100)
- `20260729-155656_aws_160ab505` — b8, torchprof timelines, all variants

Hygiene: **104/104 cells ok, 0 stuck / 0 timeout / 0 failed** (idle-watchdog armed at 180 s, never fired; retry pass unused). All b2–32 + torchprof cells 16/16 allclose. `deterministic=0` audited on every rank. Single git sha `18ebf888`, byte-identical flux libs across all three runs (manifest-verified). Budgets are pre-topk send budgets.

## Fused e2e, max-rank ms (e2e mode only)

| variant | b2 | b4 | b8 | b16 | b32 | b64 |
|---|---|---|---|---|---|---|
| allgather | 1.75 | 3.00 | 5.06 | 9.09 | 16.24 | 31.18 |
| a2av | 2.68 | 5.86 | 8.21 | 20.91 | 32.73 | 65.77 |
| a2av_ring | 2.67 | 5.94 | 8.23 | 21.17 | 32.42 | 66.03 |
| hier | 2.92 | 5.83 | 7.96 | 21.47 | 32.82 | 64.14 |
| hier_compress | 2.30 | 3.08 | 4.57 | 8.06 | 15.60 | 28.71 |
| hier_compress_identity | 1.86 | 2.62 | 4.28 | 7.70 | 14.86 | 27.53 |
| hier_compress_union | 1.57 | 2.35 | 3.74 | 6.80 | 12.65 | 23.99 |
| **hier_compress_pack** | **1.45** | **2.10** | **3.38** | **5.95** | **11.09** | **21.41** |

pack/hier ratio: 0.50 / 0.36 / 0.42 / 0.28 / 0.34 / 0.33.

## Unfused reference (torch comm → scatter → gemm), max-rank ms

| budget | comm | scatter | gemm | total | best fused (pack) |
|---|---|---|---|---|---|
| 2 | 1.38 | 2.24 | 0.70 | 4.32 | 1.45 |
| 4 | 2.83 | 4.54 | 1.25 | 8.62 | 2.10 |
| 8 | 3.96 | 8.44 | 2.09 | 14.49 | 3.38 |
| 16 | 6.11 | 16.08 | 2.95 | 25.15 | 5.95 |
| 32 | 12.61 | 42.71 | 5.98 | 61.31 | 11.09 |

(b64 unfused not measured — reference OOMs; see capsule 2 note.)

## Fused phase breakdown (phases mode — perturbed, breakdown only)

b64, max-rank ms: a2av/hier spend **44–47 ms in the barrier phase** (of ~64 ms e2e) — recv-side hot-rank skew; the GEMM waits on the hottest receiver. The compress family's dedup'd recv (wire ratio 0.737, flat across budgets) collapses that to ~2.8–4.5 ms. That, not send-byte savings, is the dominant effect on this matrix family. Balanced-relay's extra fwd-build (stage2 4.6 / gemmgate 7.5 at b64) is why it trails identity/union; relay balance is only 0.90x here (unions nearly saturate at topk 8).

## Key takeaways

1. **hier_compress_pack is the fastest variant at every budget 2–64 MiB** — 2–3.5x over hier, and it also beats dense allgather everywhere.
2. **Dense allgather beats raw a2av/a2av_ring/hier on remotefrac** (0.42–0.63x hier): the non-compress a2av family pays the recv-skew barrier; allgather's fixed dense wire sidesteps it.
3. hier ≈ a2av ≈ a2av_ring within ±3% at all budgets — hierarchical aggregation alone buys nothing on this skew; dedup + bcast + pack overlap is where the win is.
4. Balanced relay costs more than its 0.90x send-balance gain at topk 8 — its case remains low-topk union skew.
5. Fused best vs unfused reference: 3.0–5.5x.

## FAST baseline column (added later on 2026-07-29, capsules `20260729-234034_aws_663d735d` b2–32 + `20260729-234205_aws_935d3c70` b64)

`impl=fast` = FAST load-balancing alltoallv + un-overlapped GemmGroupedV2, now a first-class runner variant (`driver="fast"`; libflash sha pinned in the manifests). 6/6 cells ok, 16/16 bitwise+allclose at b2–32; b64 skip-correctness by convention. Max-rank ms:

| budget | e2e | pack | schedule | fill | wire | unpack | gemm | reset (outside) |
|---|---|---|---|---|---|---|---|---|
| 2 | 7.92 | 0.14 | 5.15 | 0.10 | 2.74 | 0.20 | 0.58 | 1.04 |
| 4 | 9.04 | 0.25 | 4.51 | 0.15 | 3.42 | 0.34 | 0.92 | 0.88 |
| 8 | 13.35 | 0.48 | 5.16 | 0.26 | 6.57 | 0.62 | 1.64 | 1.18 |
| 16 | 21.30 | 0.97 | 6.10 | 0.44 | 12.37 | 1.09 | 3.24 | 2.39 |
| 32 | 33.01 | 1.80 | 4.46 | 0.82 | 19.75 | 1.97 | 6.66 | 1.93 |
| 64 | 63.87 | 3.32 | 6.24 | 1.56 | 39.76 | 3.64 | 15.50 | 4.06 |

Takeaways vs the earlier columns:
- **FAST ≈ hier at b16–64** (21.3/33.0/63.9 vs 21.5/32.8/64.1) — the un-overlapped balanced alltoallv matches the fused-but-recv-skew-bound hier family at scale, but stays 3x behind `hier_compress_pack` everywhere (21.4 at b64).
- **FAST beats the unfused torch reference at b16–32** (21.3 vs 25.2, 33.0 vs 61.3) — load balancing + tight staging wins over dense allgather + big scatter as budgets grow — but loses at b2 (7.9 vs 4.3) where the flat ~4.5–6 ms BvN `schedule_ms` recompute floors it (schedule is INSIDE e2e; see SCHEMA).
- Per-iteration rows caught a 2.02x single-iteration EFA wire transient at b8 (worst rank 13) — visible, not silently averaged.

## Stuck/failed cells

None. (Watchdog + one-shot retry are now permanent runner features: a cell whose logs stall >3 min is killed as `stuck`, retried once, and highlighted in cells.csv and the run summary.)

Timelines: chrome traces per variant at `/home/ubuntu/sweep_data/20260729-155656_aws_160ab505/cells/*_torchprof/prof/` (load in chrome://tracing / Perfetto).
