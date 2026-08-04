# Build ledger — what makes 124 AWS capsules interpretable

Capsules are immutable and self-describing about **configuration**. They are nearly silent
about **code identity**. This document closes that gap. It is frozen: the AWS cluster is gone
and nothing here can be re-derived by measurement.

**If you read one thing:** 308 of 405 cells were produced from a **dirty working tree**, so
`git checkout <git_sha>` on a decision capsule gives you the **wrong code**. The reproduction
record is `cells.csv` `env_json` **plus** the `flux_libs` sha256 in `manifest.json`.

---

## 1. Where build identity actually lives

Each `manifest.json` has a `flux_libs` array with the sha256 of `libflux_cuda.so` and
`libflux_cuda_ths_op.so` as built at run time. This is the only record of *which binary
produced a number*, and nothing else in the repo references it.

Across the 124 capsules: **14 distinct `libflux_cuda.so`** and **28 distinct
`libflux_cuda_ths_op.so`** builds, against only 5 distinct `git_sha` values. That ratio is
the whole problem — `git_sha` does not identify a build.

Get it for any capsule:

```bash
python3 -c "import json,sys;d=json.load(open(sys.argv[1]));[print(l['sha256'][:8], l['path'].split('/')[-1]) for l in d['flux_libs']]" \
  sweeps/results/runs/<run_id>/manifest.json
```

---

## 2. The `git_sha` clusters, and what each was really running

| `git_sha` | cells | What the dirty tree actually contained |
|---|---:|---|
| `5346897` | 93 | The Tier B development arc. **Committed afterwards as `7d4b3b9` (isolated mode) + `232f371` (Tier B window gating + PULL relay).** Recorded in commit `8549311`. |
| `6da24e1` | 90 | The nsys/NVTX-proxy + `EARLY_LAUNCH` + `conn=8` era (Jul 31 – Aug 2). Landed across `b3f56ca`, `6d32ec7`. |
| `dc25f71` | 90 | The lossless tile-trace sidecar and `gw-marks` work. Landed as `cf4d7bb`. |
| `18ebf88` | 24 | Sweep-runner bring-up (watchdog, retries). |
| `afc1930` | 11 | The `hier-relay-balance` → `consolidate-sweeps` merge point. |

97 cells are clean (`git_dirty=0`), mostly the 2026-07-29 full sweep.

---

## 3. The build ledger

`ths_op` sha prefix, capsule count, date range, and what it was where identifiable. Ordered
chronologically. Builds marked **†** produced results quoted elsewhere in this handoff.

| ths_op | n | Range (2026) | What it was |
|---|---:|---|---|
| `46010fd0` | 7 | 07-29 13:31 → 23:42 | The 104-cell full sweep + FAST baseline column. Clean tree. |
| `ddffaabb` | 9 | 07-31 03:31 → 06:56 | nsys bring-up. |
| `4728c8c1` | 8 | 07-31 06:59 → 08-01 02:30 | Pre-`conn=8` arm of the Aug-1 A/B — see §4. |
| `239e8d29` | 7 | 07-31 15:16 → 15:35 | |
| `f2c63c07`, `4fbc0ddc`, `5cd94dfb` | 1 each | 07-31 15:45 → 08-01 01:42 | One-off smokes. |
| `de8dd913` | 3 | 08-01 02:30 → 02:31 | Post-change arm of the Aug-1 A/B — see §4. |
| `3acf2b0f` | 8 | 08-01 03:58 → 04:01 | |
| `57849a4f`, `1ba0b774` | 1 each | 08-01 04:08 / 05:10 | |
| `a3667ce9` | 10 | 08-01 04:36 → 04:55 | |
| `cb3e5b77` | 22 | 08-01 05:34 → 12:43 | Largest single-build block. `EARLY_LAUNCH` / `BLOCKING_WIRE` era. |
| `4394bf5f`, `35e98784`, `dde1673a`, `89e18814` | 1–2 each | 08-01 13:40 → 08-02 05:52 | Sidecar + gw-marks development. |
| `6201fca6` | 3 | 08-03 01:58 → 02:53 | Lossless sidecar validation. |
| `ec45c245` | 2 | 08-03 03:15 → 03:33 | The `FI_EFA_USE_DEVICE_RDMA` A/B (NR-05). |
| `6d86e529` † | 6 | 08-03 03:41 → 13:26 | **The high-skew `remotefrac` matrices (headroom 0.19 / 0.26) were run on this build** — unpaired. See §5. |
| `4982b37e` † | 2 | 08-03 15:08 → 15:11 | **PULL-vs-push relay decision capsule `20260803-150832`.** |
| `1be8dc11` † | 3 | 08-03 15:18 → 08-04 01:32 | Post-pull baseline (b8/k8 −1.56%). |
| `afa9b674` † | 3 | 08-04 01:39 → 01:46 | **Regressed build** (+19.25% at b8/k8) — NR-09. |
| `91b97767` † | 3 | 08-04 03:41 → 03:46 | Still regressed (+18.51%). |
| `834cbae8` † | 1 | 08-04 03:49 | Still regressed (+14.48%). |
| `251a890a` | 1 | 08-04 04:19 | |
| `b13f8916` † | 6 | 08-04 04:25 → 04:40 | **The "b8 parity, 3.598 ms" build.** Regression fixed. Widely quoted — but *not* the final build. |
| `5e6f3588` † | 10 | 08-04 04:46 → 07:14 | **The final shipped build.** Fan-out A/B/C, budget scaling, and the topk=4 win all came from here. **Matches the `.so` currently on disk** (`libflux_cuda.so` = `b2b6d52e…`, `ths_op` = `5e6f3588…`). |

**The last build is still on disk** at `python/flux/lib/` (Aug 4 03:43 / 04:45). If you ever
need to check a claim against the exact binary that made it, that is the one — until the
next `./build.sh` overwrites it.

---

## 4. The Aug-1 A/B that has no written independent variable

Capsules `20260801-022951` and `-023032` ran `ths_op 4728c8c1`; `-023011`, `-023052`,
`-023112` ran `de8dd913` — at **byte-identical `env_json`, identical `git_sha`, identical
variant, budget and mode**. The only difference is the binary. The independent variable
appears in no commit, no env delta, and no capsule note.

From the timing and surrounding work this is almost certainly the `CUDA_DEVICE_MAX_CONNECTIONS`
family-pin boundary, but **that is inference, not record**. Treat those five capsules as a
comparison whose variable is unknown.

**Related correction.** `docs/qa_walkthroughs/layer0_a2av_walkthrough.md` §12 states the
conn=1→8 boundary is "auditable ONLY via `env_json`". That is wrong and has been fixed:
pre-change capsules have **no `CUDA_DEVICE_MAX_CONNECTIONS` key in `env_json` at all** —
the value came from `launch.sh`'s default and was never part of the runner's env delta. The
conn=1 arm is identifiable only by **key absence plus date**.

---

## 5. Orphan variants: capsule specs that can no longer be re-run

Four variant names appear in committed capsule specs but were **never committed to
`sweeps/variants.py`**, and their knobs no longer exist in the source:

| Variant in capsules | Knob used | Outcome |
|---|---|---|
| `hier_compress_lb_union_push` | `FLUX_A2AV_PHASE1=1` | Lost. |
| `hier_compress_lb_union_pull` | `FLUX_A2AV_PHASE1=2` | **Won (+3.7%); shipped unconditionally** in `232f371`. |
| `hier_compress_lb_union_fanpar` | `FLUX_A2AV_FANOUT=1` | Lost (NR-06). |
| `hier_compress_lb_union_fanring` | `FLUX_A2AV_FANOUT=2` | **Won; shipped unconditionally** as ring rotation. |

`sweep.py rerun` on those capsules **fails with unknown-variant**. Both winners are now
unconditional behaviour, so there is nothing to recover — but the losing arms cannot be
reproduced without reconstructing the knobs. The full env for every one is preserved in
`cells.csv` `env_json`.

Also note the high-skew matrices from §3: `w16x8_remotefrac-494119_*` (headroom 0.19) and
`w16x8_remotefrac-228dc7_*` (0.26). These were generated to create balance headroom and were
run on build `6d86e529` — but **never as a paired `lb_union` vs `union` isolated comparison**.
That missing experiment is item **M3** in `02` §6 and is the highest-value thing you can run.

---

## 6. The comparison rule these numbers force

Measured at handoff time from the capsule set:

| Comparison | Observed spread, same configuration |
|---|---|
| Paired arms **inside one capsule** | the signal — all headline claims live here |
| **Same build**, different capsule | ~0.3–1.7% at b8/b32/b64; up to ~4.8% at b2 |
| **Different build**, same config | **6–33%** |

Every headline claim in this project is ≤7%. That is **below the cross-build spread**, so a
cross-build comparison can manufacture a result of either sign. This is not hypothetical: the
b8/k8 series reads as anything from −1.6% to +19% depending only on which build you pick.

**Rule: compare arms within one capsule and one `flux_libs` hash.** If you must compare
across capsules, require the same build hash *and* treat differences below the per-budget
spread above as noise. `sweeps/SCHEMA.md` protocol rule 4 has been amended to say this; it
previously said `git_sha` may differ "that's often the point" and said nothing about the binary.

---

## 7. Calibration anchors — normal vs alarming

Rough shape checks for a fresh platform. These are **AWS at L=8/NN=2** and will not transfer
numerically, but the *ratios* and *orders of magnitude* are a sanity net.

- `isolated` max-rank e2e at b8/k8, compress family: **≈ 3.6 ms**. Under ~2 ms or over ~6 ms
  at that config means something is wrong with the shape, not the variant.
- b32 ≈ 12 ms, b64 ≈ 24 ms — roughly linear in budget above b8.
- `wire_ratio` (dedup effectiveness) on canonical `remotefrac`: **0.737 at topk=8**, 0.825 at
  topk=4, 0.925 at topk=2. Higher topk ⇒ more duplicate tokens ⇒ better dedup.
- Balance headroom on canonical `remotefrac`: 0.90 (k8) / 0.75 (k4) / 0.575 (k2).
- Run-to-run noise grows as budget shrinks: ~0.3% at b32, ~1.5% at b8, ~5% at b2. **Small
  budgets need repeats; large ones mostly don't.**

---

## 8. One capsule is untracked on purpose

`sweeps/results/runs/20260803-031503_aws_ccbd9492/` is **not committed**, deliberately, per
commit `68409d4`: it is the contention-voided arm of the `FI_EFA` A/B with `status=stuck`.
It lived only on the now-decommissioned cluster.

Consequence for NR-05: that experiment has **one surviving arm**, which is why its verdict is
labelled *weakly measured* rather than measured. If you find the directory in a backup it is
worth keeping; otherwise accept the loss and do not restate NR-05 more confidently than the
evidence allows.
