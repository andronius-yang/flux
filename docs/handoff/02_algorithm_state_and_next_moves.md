# Algorithm state and next moves (as of 2026-08-04)

The daily driver. Read `01` first and have one green capsule before acting on this.

---

## 1. The variant landscape

All layer0, all `comm_pattern=a2av_hier_compress` unless noted. Canonical table lives in
`sweeps/variants.py`; this is the *why*.

| Variant | Wire | Idea |
|---|---|---|
| `hier` | full hierarchical a2av | Baseline. No dedup. |
| `hier_compress` | dedup + **balanced** relay | Token-dedup wire; inter-node relay load-balanced across local ranks. |
| `hier_compress_identity` | dedup + fixed relay | Design §11 identity wire: same-local-rank relay, no balancing. |
| `hier_compress_union` | dedup + union broadcast | Gateway broadcasts its whole staged union to every local peer as pure-CE puts. No index build, no SM gathers. **The variant to beat.** |
| `hier_compress_pack` | union + source pack overlap | Pack for iteration n+1 on a separate stream. Pins `CONNECTIONS=2`. |
| `hier_compress_lb_union` | **balanced chunked** wire + union gateway | The Tier B variant. Balanced per-round wire bytes *and* the union gateway. |

`allgather`, `a2av`, `a2av_ring`, and `fast` (the un-overlapped FAST BvN baseline) are
controls.

**Tier B has no environment knob of its own.** It is folded unconditionally into
`FLUX_A2AV_LB_UNION=1`. Do not hunt for a `FLUX_A2AV_TIER_B` — it does not exist, and the
knobs that briefly did (`FLUX_A2AV_FANOUT`, `FLUX_A2AV_DEBUG_GATING`) were deleted at
canonicalization. See `04` for the orphan-variant names that survive in old capsule specs.

---

## 2. The predictive ratio — compute this before you burn an allocation

The single most useful artifact of the AWS week. Every compress cell emits
`relay_balanced_bytes` and `relay_ident_bytes` in `cells.csv`. Their ratio is the
**balance headroom**: how much the balanced relay could possibly shrink the per-round
critical-path wire.

```
headroom = relay_balanced_bytes / relay_ident_bytes
```

`hier_compress_lb_union` pays a *fixed* extra intra-node pass (the phase-1 pull) to buy that
reduction. If headroom is 0.90, the entire prize is 10% of the wire — and the wire is not
the whole latency. **A variant cannot win a race it has no room to win.**

Measured on the canonical `remotefrac` matrix at 2 nodes / 16 ranks:

| topk | headroom | interpretation |
|---|---|---|
| 8 | **0.900** | almost no room — union already balanced the wire |
| 4 | 0.750 | some room |
| 2 | 0.575 | most room |

*Why headroom degrades as topk rises:* with 8 copies spread over 2 nodes, nearly every token
has a copy on each node, so every rank's remote union saturates to roughly its whole token
count regardless of routing skew. Dedup has already done the balancing. **This is the key
L/NN-dependent quantity — see §5.**

**Do this first on any new shape:** run one cheap cell, read the ratio, and predict. If
headroom is ~0.9, do not expect `lb_union` to win and do not spend a sweep proving it.

---

## 3. What actually won — the build-controlled grid

Reconstructed from capsules at handoff time: `isolated` mode, mean over iterations of the
per-iteration max across ranks, `lb_union` vs `hier_compress_union`, **paired within each
capsule**. `build` is the `libflux_cuda_ths_op.so` sha256 prefix. Negative delta = lb_union wins.

| topk | budget | build | lb_union | union | Δ% | capsule |
|---:|---:|---|---:|---:|---:|---|
| 4 | 8 | `5e6f3588` | 3.124 | 3.280 | **−4.75** | 20260804-0713 |
| 4 | 8 | `5e6f3588` | 3.095 | 3.336 | **−7.21** | 20260804-0714 |
| 4 | 2 | `5e6f3588` | 1.727 | 1.615 | +6.94 | 20260804-0714 |
| 4 | 32 | `5e6f3588` | 7.935 | 7.823 | +1.44 | 20260804-0714 |
| 2 | 8 | `5e6f3588` | 2.747 | 2.630 | +4.44 | 20260804-0712 |
| 8 | 8 | `5e6f3588` | 3.743 | 3.606 | +3.82 | 20260804-0446 |
| 8 | 8 | `5e6f3588` | 3.694 | 3.661 | +0.90 | 20260804-0450 |
| 8 | 8 | `b13f8916` | 3.598 | 3.630 | −0.86 | 20260804-0426 |
| 8 | 8 | `1be8dc11` | 3.578 | 3.635 | −1.56 | 20260803-1518 |
| 8 | 8 | `4982b37e` | 3.774 | 3.817 | −1.12 | 20260803-1508 |
| 8 | 8 | `afa9b674` | 4.332 | 3.632 | +19.25 | 20260804-0140 |
| 8 | 8 | `91b97767` | 4.410 | 3.721 | +18.51 | 20260804-0346 |
| 8 | 8 | `834cbae8` | 4.252 | 3.714 | +14.48 | 20260804-0349 |
| 8 | 2 | `5e6f3588` | 1.903–1.995 | 1.820–1.862 | +4.6…+7.2 | 0446, 0450 |
| 8 | 32 | `5e6f3588` | 12.005–12.158 | 11.826–11.858 | +1.2…+2.8 | 0446, 0450, 0649 |
| 8 | 64 | `5e6f3588` | 24.607–24.747 | 24.296–24.405 | +1.3…+1.4 | 0653, 0654 |

**How to read this, honestly:**

- **There is exactly one winning configuration: topk=4, budget 8 MiB.** It is the only cell
  that wins by more than noise, and it reproduced twice on the final build (−4.8%, −7.2%).
- **On the final shipped build, topk=8/b8 is a slight loss** (+0.90%, +3.82%), not parity.
  The frequently-quoted "b8 parity, 3.598 ms" came from build `b13f8916`, an *earlier*
  binary. Build identity was the hidden variable in the b8/k8 series; the +14…+19% rows are
  the interim regression (`afa9b674`, `91b97767`, `834cbae8`), not a property of the design.
- Everything else loses: b2 by 4–7%, b32 by 1–5%, b64 by ~1.3%.

**The remembered rule is a conjecture, not a law.** It was stated as "headroom ≤ ~0.8 AND
payload in an amortization band." It has **one confirming cell** (k4/b8) and a standing
counterexample: **k2 has the best headroom of all (0.575) and still loses** (+4.44% at b8).
Keep the table. Treat the rule as a hypothesis worth testing on the new shape — §6 says how.

---

## 4. Tier B in one page (mechanism sketch, not a walkthrough)

For the full mechanism read `docs/qa_walkthroughs/layer0_a2av_walkthrough.md`; this is what
you need to *reason* about it.

**The idea.** Before Tier B, a destination's tiles waited on per-*source* signal slots, so
nothing unblocked until a whole node's contribution had arrived. Tier B re-keys signal-slot
identity from **source** to **delivering window**.

- The gateway sends **one contiguous put per destination per round**, whose fused signal
  flips *the window's* slot. This is sound because a window is a `chunk_bound` cut of the
  canonical stream, hence contiguous in every destination's recv image.
- **The count invariant that makes it general:** there are exactly `(NN−1)·L` windows, which
  equals the number of remote sources at every NN. So the existing signal-slot space is
  reused with no resizing, and it stays inside `kMaxBuckets=65` and the 64-bit masks up to
  NN=8. **This is why Tier B should survive NN>2 unchanged** — worth confirming, not assuming.
- Destination iteration is **ring-rotated** (`(my_lr + 1 + dn + dl) % L`), staggered per
  gateway and round. A/B/C verdict: capsule `20260804-043026` — ring significant at b2, ties
  elsewhere, never worse; parallel lanes measurably worse.
- The per-tile gate is re-keyed by feeding a device-computed per-(expert, window) cumsum to
  `args.accum_per_rank_ptr`. **One `searchsorted`, zero H2D**, riding the existing meta
  arena. Zero kernel changes.
- Schedule is **dense static**; the claimer stays off for lb_union (at G=128 the claimer
  itself costs ~0.8 ms).

**Relay phase 1 is a PULL** (separate change, same commit `232f371`): each relay announces
its own pack via `a2av_pack_ready_sig_`, then `getmem_nbi`s its chunk from each peer. This
retired `a2av_relay_sig_` and put 8 relays on 8 copy engines instead of chaining on each
source's put FIFO. Decision capsule `20260803-150832` (+3.7%).

**The payoff is occupancy, not wire time.** Same-protocol nsys pair (`20260804-043510`
lb_union vs `044057` union-bc): on the node with **spread** landings, mean tile-occupancy
deficit 0.225 vs 0.311 (**−28%**). On the node with **clustered** landings, 0.41–0.52 for
both — parity, because nothing can unblock before data lands. **Tier B's win scales with
landing spread.** That is the load-bearing sentence of this whole document; §6 turns it into
a prediction.

---

## 5. The L=8 → L=4 transfer model

Which measured quantities move when the shape changes, and in which direction. This is
judgment, not measurement — it exists so you re-test the right things instead of all things.

| Quantity | Depends on | Direction at L=4 / NN>2 | Confidence |
|---|---|---|---|
| **Balance headroom** | topk, NN, L, traffic skew | **Improves (falls) as NN rises** — with more nodes the topk copies spread thinner, so remote unions saturate less. At NN=4 the k8 headroom should drop below 0.90. | inferred-from-mechanism |
| Union broadcast amplification | L | Gateway broadcasts to L−1 peers: **cheaper at L=4** (3 peers, not 7). Favours `union`. | mechanism, and the repo already said so (`aws_2n8g_a100_handoff.md:69`) |
| Relay parallelism | L | "8 relays on 8 copy engines" becomes 4 on 4. Each relay carries ~2× the rows. **Weakens the PULL relay's advantage.** | mechanism |
| Landing spread | fabric, NIC binding | Slingshot binds NICs per GPU where p4d shared 4 NICs across 8 GPUs. **Spread should change character — likely widen across nodes.** Favours Tier B (§4). | hunch — the highest-value unknown |
| Per-relay capacity need | L | Doubles at L=4 while `scale_knobs` returns the same cap. Risk concentrated at b32/b64. See `01` §8. | mechanism |
| The b8 inversion (§7) | budget, GEMM length | Budget-driven, **not** L-driven. Should reproduce. | measured, transfer inferred |
| Absolute milliseconds | everything | **Do not transfer.** Different matrix (`matrix_id` includes L), different fabric, different NVSHMEM. | certain |
| FAST BvN schedule floor | host CPU | ~4.4 ms/iter on p4d CPUs vs ~0.9 ms on Perlmutter. **The FAST baseline will look substantially better here** — do not read that as a regression in flux. | measured on both |

---

## 6. Next moves, ordered for the NN>2 / L=4 question

Each is a falsifiable prediction with the experiment that would kill it. Run them in this
order; each is cheap and the early ones re-aim the later ones.

**M1 — Measure headroom at NN=4 before measuring anything else.**
One cheap cell per topk. Prediction: headroom at topk=8 **falls below 0.90** at NN=4, because
copies spread thinner. Falsified if it stays ≈0.90 — in which case `lb_union` has no room at
NN=4 either and the whole line of work is closed at high topk.
```bash
python sweeps/sweep.py run --platform perlmutter --variants hier_compress_lb_union \
    --families remotefrac --budgets-mib 8 --topk 2,4,8 --G 128 --modes isolated --nodes 4 --dry-run
```

**M2 — Re-test the one win at L=4.** topk=4, b8, paired arms, one build.
Prediction: the −5…−7% win survives, possibly larger (thinner copies ⇒ more headroom).
Falsified if it vanishes ⇒ the win was an L=8 artifact.

**M3 — Test the headroom conjecture where it makes its strongest claim.**
**This is the experiment that was never run on AWS.** High-skew `remotefrac` matrices exist
with headroom **0.19 and 0.26** (`w16x8_remotefrac-494119_*`, `w16x8_remotefrac-228dc7_*`) —
but no capsule ever paired `lb_union` against `union` on them in `isolated` mode. The
conjecture predicts a *large* win there. If `lb_union` loses at headroom 0.19, the headroom
framing is dead and §2 should be rewritten. Regenerate the equivalent at L=4 and run paired
arms. **Highest information per allocation-hour of anything in this list.**

**M4 — The landing-spread prediction.** nsys pair, `lb_union` vs `union`, same protocol,
same build (as `20260804-043510` / `044057` did). Prediction: Slingshot's per-GPU NIC binding
plus NN=4 widens landing spread, so the occupancy deficit gap exceeds the −28% seen on AWS's
spread-landing node, and topk=8/b8 moves from slight loss toward a win. Falsified if
landings cluster, in which case Tier B is fabric-limited here too.

**M5 — Confirm the b8 inversion is budget-driven, not L-driven.** b2/b8/b32 at NN=4.

Throughout: **paired arms inside one capsule on one build.** See `04` and `SCHEMA.md`.

---

## 7. The budget inversion, stated honestly

Tier B's occupancy advantage **inverts above ~8 MiB**. Mean tile-occupancy deficit,
lb_union vs union: b8 `0.347 vs 0.382` (−9%), b32 `0.230 vs 0.211` (+9%), b64
`0.268 vs 0.219` (+23%).

*Mechanism:* tiles/rank grow 2048 → 8192 → 16896 from b8 → b32 → b64, so the GEMM becomes
long enough to hide the collective on its own (absolute deficits fall from ~0.35–0.38 to
~0.21–0.27). Incremental unblocking has little left to buy, while lb_union's fixed extra
intra-node pass scales linearly with bytes and is charged in full.

**The correction that matters:** commit `8549311` frames this as "the advantage inverts
above b8", which implies lb_union *won* below b8. It did not — it was at best parity at
b8/k8 and **lost at b2** (+4.6…+21%). The honest statement is that lb_union has one win
(k4/b8) and the budget trend explains why the loss grows at large budgets.

**b64 is not validated.** Cells failed 2-of-4 on first attempt. That is a capacity limit,
not a variant bug — both arms OOM the 40 GB A100 at G=128/k8 (10 GiB symmetric heap + an
8 GiB harness allocation + fragmentation). Fix that works and is fair to both arms:
`extra_env: PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"`.

---

## 8. Code loose ends carried forward

Known, harmless, worth cleaning when you next touch the file:

- `src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc` ~line 3143: two contradictory
  comment blocks above an unchanged line. The first describes a claimer-based candidate that
  was **abandoned**; the code takes the dense static schedule. Delete the first.
- `src/moe_ag_scatter/workspace_util.cu` `bucket_of`: the seg_end-lane bucketing **shipped**
  and is correct, but it is **inert for lb_union** — that variant takes the dense static
  schedule, where `get_a2av_ws()` returns `{}` and no buckets are built. It affects only the
  plain a2av dispatch mode. (This resolves an apparent contradiction between the commit
  message, which says it shipped, and older notes saying "tried, no help" — both are true,
  in different scopes.)
- `a2av_gw_round_sig_` is still allocated and counted in the symmetric-MB report for
  lb_union, though only the gather arm reads it now.
- `a2av_gating_cumsum_` is allocated once and reused; the `accum_per_rank_ptr` selector keys
  on `.defined()` rather than freshness, so a change in `E` or `W` across forwards would
  throw on `copy_`. Fine today, fragile if expert count becomes dynamic.

---

## 9. Out of scope

`origin/a2av-hier-layer1` has 4 unmerged commits (split-pipelined hierarchical alltoallv
combine with per-split topk reduce, for **layer1**). Safe on GitHub, deliberately not
covered by this handoff. If layer1 becomes live again, decide merge order against the Tier B
changes now on `main` before starting.
