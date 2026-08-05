# The starvation campaign (2026-08-05): lb_union's static schedule starves the GEMM on a constructible, realistic distribution

Goal and constraint: with **zero changes to `src/` or `python/flux/`**, find a realistic
traffic distribution that demonstrably starves the compute tile kernel under
`hier_compress_lb_union`, as motivation for a runtime (compute-aware) scheduling
algorithm. Everything here is sweeps-side tooling plus four capsules.

**Headline: confirmed.** On a per-node exporter-skew matrix (`fanoutskew`, two hot
exporter nodes), the hot destination nodes' GPUs go **fully idle for 1.2–2.5 ms per
iteration** (tile-occupancy deficit 0.29–0.455) under lb_union, while the *same matrix on
the same build* leaves the union arm's deficit at 0.073–0.107. The blocked window's data
dependencies are fully attributed: the thin rounds' windows had signaled ~3 ms before the
wavefront could use them — the compute was ordered behind a fat round the schedule
insists on consuming first. Load balance balanced every lane perfectly (headroom = 1.000
by construction) and the GPU starved anyway: **the pathology is ordering and
compute-per-byte, two things the balancer cannot see.**

---

## 1. Mechanism recap (why starvation is structurally possible)

- The lb_union tile schedule is a **static wavefront**: stage-major in ring order over
  window slots (`shift_rank_to_order`), round-robin dealt to persistent CTAs, no
  skip-ahead, grid exactly fills the SMs. A CTA spinning on an unarrived window burns its
  SM slot; all CTAs reach a late stage together (`workspace_util.cu:36-159`,
  `ag_scatter_grouped_problem_visitor.hpp:97-122`).
- A window's signal fires only after the gateway's **serialized blocking re-broadcast**,
  rounds ascending with a head-of-line `t_wait` (`gemm_grouped_v2_ag_scatter.cc:2519-41`).
- Relay wire puts are **round-serialized per NIC**, targets descending the ring
  (`:2453-2455`) — volume in an early round delays every later round's put.
- The dealer's closed form `U[s][n] = T·min(1, f·topk)` makes wire volume and dedup ONE
  knob; compute rows ≥ wire rows always. Per-node wire stagger exists only below the
  saturation threshold `f < (NN−1)/topk` — which is why the demonstration needs NN ≥ 3
  and why no NN=2 matrix could produce it.

Hypotheses H1 (rotation anti-alignment), H2 (gateway round head-of-line), H2b (relay
round serialization), H4 (byte-equal ≠ compute-equal) — see the analysis session notes
and `sweeps/predict_starvation.py`'s module docstring for the full statements.

## 2. New tooling (all in `sweeps/`, all committed)

| Piece | What it does | Validation |
|---|---|---|
| `gen_matrix.py` `fanoutskew` family | per-NODE exporter skew: `nodefracs[i]` = node i's uniform remote fraction; no shuffle; NN≥3 | `--print-only` reproduces hand arithmetic; existing family matrix_ids unchanged |
| `gen_matrix.py dedup_round_stats()` | closed-form U_mat / node-pair dedup / per-dest round profile / column sums / headroom | reproduces the documented AWS headrooms 0.900 / 0.193 / 0.265 exactly |
| `predict_starvation.py` | stdlib-only offline predictor: exact dealer replica + window cut + gating geometry + static schedule, with a modeled arrival timeline → per-stage stall table, predicted deficit | dealer bit-equal to `traffic_matrix.py` (`--selftest-dealer`); sidecar `rows[]` reproduced **exactly** incl. ±1-row chunk-remainder fingerprints (ranks 0/6 of capsule 20260805-044841) |
| `predict_starvation.py --compare-sidecar` | per-stage predicted signal vs measured `arrival_gt` vs first tile fire | used for all attributions below |
| Specs `starve_pm4n_{iso,trace}_{k8,k4}_v1.yaml` | the campaign cells; EARLY_LAUNCH=1 campaign-wide (user-approved), captures under ISOLATED discipline, **no BLOCKING_WIRE** (relay serialization is the mechanism under test) | dry-run verified |

Timing-model calibration (from the 20260805-044841 capture): GEMM ≈ 4.8 rows/µs,
gateway CE put ≈ 58 GB/s effective, t0_wire ≈ 500 µs. Note that capture is a
**different build** than the campaign's, so calibration is advisory; event ORDER and all
row/tile accounting in the predictor are exact regardless.

## 3. The three arms and what they measured

All NN=4 / L=4 / W=16 / b32 / G=128, `FLUX_A2AV_EARLY_LAUNCH=1` everywhere, paired
lb_union vs union inside each capsule, one build (`libflux_cuda.so` sha `38e7350f…` — the
fused-stage2 worktree build with the FUSED_STAGE2 knob off, i.e. baseline behavior).

### Capsules

| capsule | mode | arms |
|---|---|---|
| `20260805-120335_perlmutter_52eb1fe2` | isolated (clean) | uniform k8 + fanoutskew(0.9,0.1,0.9,0.1) k8 |
| `20260805-120537_perlmutter_9f659cb6` | isolated (clean) | fanoutskew(0.9,0.25,0.25,0.25) k4 |
| `20260805-120645_perlmutter_0a4f1e8a` | nsys + tile-trace | uniform k8 + fanoutskew two-hot k8 |
| `20260805-120857_perlmutter_0b3180a3` | nsys + tile-trace | fanoutskew one-hot k4 |

### Arm 1 — `fanoutskew (0.9,0.1,0.9,0.1)` k8: **the demonstration**

Two hot exporter nodes (0, 2) ⇒ hot destinations combine a short local runway (0.8T
rows) with fat inbound (round profile thin/FAT/thin: 4368 / 16384 / 4368 U rows).

- **lb_union hot ranks: deficit 0.29–0.455, longest_low 1.2–2.5 ms** (trace capsule
  0a4f1e8a; per-rank: r0-r3 = 0.378/0.407/0.449/0.290, r8-r11 = 0.378/0.412/0.455/0.325
  — node 0 ≈ node 2, the designed twin replication). Thin ranks: 0.09–0.11.
- **union, same matrix, same build: hot ranks 0.073–0.107.** The starvation is a
  property of lb_union's windowed rounds + static schedule, not of the traffic.
- **Attribution (rank 10, worst):** tiles entered the gate at ~647 µs; the fat round's
  first-expected window `win(n0,l2)` arrived at **3410 µs → 2.76 ms whole-wavefront
  spin**; the remaining fat windows then fired in a serialized-broadcast ladder
  (3704/3978/4361 µs). The thin round n3's windows had fired at **326 µs** — data
  present ~3 ms before the wavefront was allowed to touch it (H2: arrived-but-ordered-
  behind; the schedule, not the wire, is the constraint). Gateway 2's fat-round
  redistribution shows a **2240 µs stall** waiting on wire (H2b: the relay put the fat
  window to its ring predecessor first). Megadiagram: in-flight collapses to zero
  node-wide in a per-rank-staggered band — `$PSCRATCH/workspace/andrewy/
  megadiagrams_starve/mega_{lb_union,union}_fanoutskew_k8_node{0,2}.png` (+ gwmarks
  tables; NN>2 caveat: the `w<j>` markers of non-stalled gateways look misattributed —
  trust the tile-trace stamps, which are exact).
- Clean isolated latency (capsule 52eb1fe2): lb 14.72 vs union 14.16 ms (+4.0%) — the
  critical path sits on the *thin* nodes, so Arm 1's starvation is latency-invisible by
  design; the claim is the per-rank geometry: **the node with 3.3× less compute is the
  one idling 40%.**

### Arm 0 — `uniform` k8: the realism anchor

The most vanilla distribution there is: every round saturates (all 12 round-streams =
16384 U rows), headroom exactly 1.000 — lb_union's balancing has literally nothing to
improve. Measured: lb deficits 0.086–0.132 with arrival stamps present (real, but mild —
the predictor's 0.34 was pessimistic; at k8-uniform every stage carries enough rows that
the wavefront mostly covers the ordering effects). Clean isolated: **lb 11.73 vs union
10.81 ms (+8.5%)** — on plain uniform traffic lb_union pays its fixed pull and ordering
costs for zero achievable benefit. That, plus Arm 1, is the two-sentence motivation:
*when balance can't win it loses outright, and when traffic is node-heterogeneous its
own schedule starves the GPU.*

### Arm 2 — `fanoutskew (0.9,0.25,0.25,0.25)` k4: the decoupling surprise

Clean isolated: **lb 7.02 vs union 8.32 ms (−15.6%, lb wins big)** — yet in the traces
lb's GEMM-side deficits are *higher* than union's (hot 0.30–0.38 vs 0.23–0.29, thin
0.14–0.16 vs 0.08–0.19). Latency and tile-starvation decouple: the e2e difference lives
outside the GEMM span (union's per-source aggregation/broadcast path on a one-hot-
exporter matrix at k4). The predicted between-destination contrast (fat-at-dn1 vs
fat-at-dn3) did **not** materialize at k4 (thin-node deficits ≈ equal) — the predictor's
wire constants are too pessimistic for this fabric at these sizes. Do not quote a
mechanism for the k4 e2e win until the union arm's wire path is attributed.

## 4. Hypothesis verdicts

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H2 (gateway round HOL → arrived-but-unsignaled / ordered-behind) | **confirmed** | rank-10 attribution: thin windows fired 326 µs, wavefront blocked until 3410 µs on the fat round |
| H2b (relay round serialization delays later rounds) | **confirmed** | gateway 2's 2240 µs redistribution stall; fat window put second on the relay's NIC |
| H4 (byte-balanced ≠ compute-balanced windows) | **confirmed by construction + measurement** | headroom 1.000 matrices starve; hot-node columns 3.73T starve while 12.27T columns don't |
| H1 (rotation anti-alignment, own-lane window near-last) | **visible, small** | within-round fire ladders show schedule-first windows arriving mid-rotation; magnitude ≪ H2/H2b at these shapes |
| H3 (pack coupling → clustered landings, NN=2) | **untested** | optional phase skipped; existing high-skew capture shows the 0.99-rank's 1.25 ms *local* pack block (rank 7), consistent but not dispositive |

## 5. What this buys the runtime-algorithm pitch

The blocked resource is never bytes — it is *order*. All three levers a runtime
algorithm could pull are now measured, separable, and attributable with shipped tooling:

1. **Consume-order freedom** (the claimer exists in-tree, disabled for its 0.8 ms cost —
  rank 10 lost 2.76 ms to schedule rigidity in one iteration; the price comparison is
  now concrete).
2. **Broadcast-order freedom** (gateway could broadcast thin arrived rounds during the
  fat round's wire wait — the H2 gap is exactly that opportunity, ~3 ms on hot ranks).
3. **Round-order freedom** (relays could reorder per-destination puts by
  compute-per-byte, which `a2av_gating_cumsum_` already quantifies host-side).

## 6. Reproduction & provenance

- Tooling commits (branch `worktree-starvation-campaign`): fanoutskew family + stats,
  predictor (+ comparator), four specs; capsule commits follow each run.
- Matrices regenerate deterministically; verify via `matrix_sha256`
  (`w16x4_uniform_b32_k8_id001`, `w16x4_fanoutskew_b32_k8_id001`,
  `w16x4_fanoutskew-acc8ac_b32_k4_id001`).
- Raw staging + tile-trace bins archived (they are NOT manifest-hashed):
  `$PSCRATCH/workspace/andrewy/sweep_data/archives/` (SHA256SUMS inside; both the four
  20260805-morning captures and the two campaign captures). $PSCRATCH purge policy
  applies — re-archive elsewhere if this matters in 8 weeks.
- Predict / attribute / re-render:
  ```bash
  python sweeps/predict_starvation.py --family fanoutskew --W 16 --ranks-per-node 4 \
      --budget-mib 32 --topk 8 --G 128 --ranks 10 \
      --compare-sidecar <records>/a2av_tile_trace_r10.bin
  python sweeps/plot_a2av_trace.py <records> --rank 10 --wire lb_union \
      --gw-marks <node2.sqlite> --mega mega.png
  ```

## 7. Loose ends carried forward

- Predictor wire/broadcast constants need per-fabric calibration on a same-build capture
  (its k4 stall magnitudes overshot; structure was right).
- The k4 e2e inversion (lb −15.6%) is unexplained at the wire level — attribute before
  using it in any claim.
- NN>2 gw-marker round attribution in `plot_a2av_trace.py` remains unverified for
  non-stalled gateways; the sidecar `rows[]`-matching path in the predictor is the
  trustworthy alternative.
- H3 (NN=2 lb-vs-identity landing spread) remains open; cheap if wanted.
- The campaign ran on the fused-stage2 worktree build (knob off). If canonicalization
  lands a new build, re-pair one capsule before extending the ledger.
