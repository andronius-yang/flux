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

**Headroom is a property of the traffic matrix, not of topk.** The canonical `remotefrac`
uses `fracs = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90)`. Two deliberately high-skew
parameterisations also exist in the capsule record and reach **headroom 0.19 and 0.26**:

| `fracs` | headroom | matrix id fragment |
|---|---|---|
| `0.01 ×7, 0.99` | **0.19** | `remotefrac-494119` (b8), `-da835f` (b2), `-773d88` (b32) |
| `0.02 ×7, 0.90` | **0.26** | `remotefrac-228dc7` (b8), `-f70811` (b2), `-9b7dea` (b32) |

**And on those matrices `lb_union` does win — at large budgets.** See §3.

**Do this first on any new shape:** run one cheap cell, read the ratio, and predict. Headroom
~0.9 means there is almost nothing to win; headroom ~0.2 means there is, and §3 says where.

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

### The high-skew matrices — where the headroom hypothesis actually gets tested

Everything above is the **canonical** matrix (headroom 0.90 at k8), i.e. the case with almost
nothing to win. The high-skew matrices from §2 tell a different and more coherent story.

These cells are **`e2e` mode, not `isolated`** — so under the never-mix rule they are *not*
quotable as latency. But they are **paired arms inside one capsule on one build**, which
makes the *ordering* meaningful. Read them as a direction, not a number.

| headroom | budget | lb_union | union | Δ% | capsule |
|---:|---:|---:|---:|---:|---|
| 0.19 | 2 | 1.456 | 1.361 | +7.01 | 20260803-0341 |
| 0.19 | 8 | 3.452 | 3.445 | +0.21 | 20260803-0341 |
| 0.19 | 32 | 11.947 | 12.308 | **−2.94** | 20260803-0341 |
| 0.26 | 2 | 1.499 | 1.413 | +6.09 | 20260803-0341 |
| 0.26 | 8 | 3.440 | 3.447 | −0.21 | 20260803-0341 |
| 0.26 | 8 | 3.405 | 3.399 | +0.18 | 20260803-0410 |
| 0.26 | 32 | 11.855 | 12.098 | **−2.01** | 20260803-0341 |
| 0.26 | 32 | 11.879 | 12.292 | **−3.36** | 20260803-0410 |

**This inverts the canonical-matrix conclusion at b32, 3 runs out of 3.** On the canonical
matrix `lb_union` *loses* at b32 (+1.2…+5.0%); on the high-skew matrices it *wins*
(−2.0…−3.4%).

**The synthesis that fits all the data.** Two independent conditions, and the interaction
between them is what the single-condition rule missed:

1. **Headroom** decides whether balancing the wire can pay at all.
2. **Budget** decides whether the wire is worth balancing — at b2 the fixed intra-node pass
   (phase-1 pull) dominates and lb_union always loses; at b32 the wire dominates, so with
   real headroom the balancing wins, and without it the fixed cost is charged for nothing.

So the win moves to **large** budgets when headroom is real, and collapses to a single narrow
cell (k4/b8) when it is not. The earlier framing — "the advantage inverts above ~8 MiB" — is
true *only on the high-headroom canonical matrix*, and §7 is scoped accordingly.

**Status: promising, not established.** The high-skew evidence is `e2e`-mode only, from a
single day and two capsules on one build (`6d86e529`). The isolated-mode confirmation is
**M3 in §6** and is the highest-value experiment available.

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

**M3 — Confirm the high-skew win in `isolated` mode. Highest value in this list.**
The high-skew matrices were paired on AWS but **only in `e2e`/`phases` mode** (§3), which
cannot be quoted as latency. The e2e ordering says `lb_union` wins at b32 by 2–3.4% (3/3).
Repeat it in `isolated` mode. If it holds, the headroom framing is established and the
operating rule becomes "high skew + large budget"; if it evaporates, the framing is dead and
§2/§3 should be rewritten around the single k4/b8 cell.

**Generating them — they are fully reproducible, but the `fracs` list must be redesigned
for L=4.** The matrices are not stored anywhere; they are regenerated deterministically and
verified by `matrix_sha256`. The AWS parameters were:

```bash
# headroom 0.19 (AWS, L=8)
python sweeps/gen_matrix.py --family remotefrac --W 16 --ranks-per-node 8 \
    --budget-mib 8 --topk 8 --param fracs=0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.99
# headroom 0.26 (AWS, L=8):  --param fracs=0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.9
```

**Trap:** `gen_matrix.py` uses `fracs[:L]` — only the **first L** entries — then shuffles them
per node. At L=4 the lists above truncate to `[0.01,0.01,0.01,0.01]` and `[0.02]*4`, i.e.
**uniform, no skew, no headroom at all**, silently. The skew lives in the *last* element.
Rebuild the list for L=4 so the spread survives truncation, e.g.
`--param fracs=0.01,0.05,0.30,0.99`, and **verify by reading the emitted
`relay_balanced_bytes/relay_ident_bytes` before trusting the cell.**

**M4 — The landing-spread prediction.** nsys pair, `lb_union` vs `union`, same protocol,
same build (as `20260804-043510` / `044057` did). Prediction: Slingshot's per-GPU NIC binding
plus NN=4 widens landing spread, so the occupancy deficit gap exceeds the −28% seen on AWS's
spread-landing node, and topk=8/b8 moves from slight loss toward a win. Falsified if
landings cluster, in which case Tier B is fabric-limited here too.

**M5 — Confirm the b8 inversion is budget-driven, not L-driven.** b2/b8/b32 at NN=4.

Throughout: **paired arms inside one capsule on one build.** See `04` and `SCHEMA.md`.

---

## 7. The budget inversion — real, but scoped to the high-headroom matrix

**Scope first:** everything in this section is the **canonical** `remotefrac` matrix
(headroom 0.90), where there is almost no wire imbalance to remove. On the high-skew
matrices the sign flips at b32 (§3), so do not state this as a general property of Tier B.

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

## 9. Out of scope (RESOLVED 2026-08-16)

Historical: `origin/a2av-hier-layer1` (split-pipelined hierarchical alltoallv
combine with per-split topk reduce, for **layer1**) was unmerged when this
handoff was written. It has since merged to main (2026-08-11) together with
the eager arrival-order reduce and the compress combine wire, and the sweep
runner grew a layer axis (`l1_*` variants, `timing_mode` isolated/amortized;
`sweeps/SCHEMA.md` is the authority). Layer1 compress has never run on a GPU
(CPU-sim only) — bring-up ladder first.

---

## 10. MoonEP-semantics arm (added 2026-08-07)

A new measurable variant `moonep`: a semantic port of MoonshotAI/MoonEP's
redundant-expert dispatch (planning rebalances hot experts onto under-loaded
ranks so every rank GEMMs exactly S*K rows; dedup'd representative rows only
on the wire; static expert-grouped [NvS, H] layout; per-iteration weight
prefetch for the B = E/R redundant expert slots). MoonEP's own kernels are
Hopper/NVLink-domain-only (TMA + NVSwitch multicast, no RDMA path), so the
port re-implements the algorithm on NCCL + local scatters + per-segment
GemmOnly; A100/Slingshot-safe, no NVSHMEM heap.

- Semantics anchor: the planner is bit-identical to MoonEP's own executable
  oracle (vendored at `test/python/moe_ag_scatter/moonep_oracle/`), enforced
  by `test/python/moe_ag_scatter/test_moonep_planner.py` over MoonEP's 18
  planning cases (imported from the sibling MoonEP checkout when present),
  R=16 Perlmutter-shaped cases, and lognormal fuzz. Runs on a login node,
  no GPU.
- Implementation: `python/flux/testing/moonep_semantics.py` (replicated
  vectorized planner + `MoonEPLayer0Runner`), driver
  `test/python/moe_ag_scatter/test_moe_moonep_traffic.py` (phase-evented:
  plan_comm/pack/comm/scatter/prefetch/gemm; dispatch content checked
  bitwise vs the plan, prefetch vs an independent broadcast, GEMM vs
  torch.matmul). Sweep plumbing mirrors the `fast` driver precedent
  (`driver="moonep"` in `sweeps/variants.py` / `sweep.py`); metric and
  cell-fact meanings in `sweeps/SCHEMA.md`.
- Known deviations (disclosed, unavoidable off-NVLink): two-sided staged
  a2av + local placement instead of one-sided direct-into-slot writes (the
  two port-added local copies are their own metrics, `pack_ms` +
  inside `scatter_ms`, so `comm_ms` stays pure wire); replicated planning
  instead of rank-0 + hardware multicast (wire cost `plan_comm_ms`);
  prefetch serialized in the timed window (MoonEP overlaps it on a comm
  stream — bias is AGAINST the port); layer0 prefetches 1 weight matrix vs
  MoonEP training's 3.
- First capsule spec: `sweeps/specs/moonep_pm4n_trace_iso.yaml` (moonep vs
  allgather vs lb_union, real Qwen3 decode trace b8/k8 G=128, isolated +
  e2e, correctness on). The balance fingerprint to look for:
  `gemm_rows_per_rank` constant for moonep, skewed for the baselines.

**M4 addendum (2026-08-08).** Two mechanism-fidelity arms landed after user
review (plan-reuse was dropped: inference re-plans per activation):

- `moonep_nvshmem[_overlap]` — dispatch a2av over flux's one-sided
  `All2AllSingle` (`nvshmemx_putmem_nbi_block` per destination + 2 team
  stream barriers — the live `a2a_single_kernel_v2` path; correction
  2026-08-11: this entry previously described the file's dead
  `putmem_signal` kernel, see NR-12 fact 8; still sender-driven,
  receiver-passive), replacing NCCL grouped send/recv, which was a
  dishonest stand-in for MoonEP's one-sided writes. Same plan, pack order,
  placement indices, correctness gates — bitwise-exact on first bring-up,
  proving semantic invariance across transports.
- `moonep_overlap` / `moonep_nvshmem_overlap` — prefetch on a dedicated
  high-priority stream + separate NCCL communicator, event-joined before
  GEMM: MoonEP's `async_finish` comm-stream overlap, which the serialized
  port deliberately pessimized. New metric `prefetch_wait_ms` = exposed
  stall at the join; 1-node bring-up showed prefetch fully hidden
  (~0.002 ms exposed) with contention honestly visible in pack/comm.
- Spec: `sweeps/specs/moonep_m4_pm4n_trace_iso.yaml` (2x2 transport/overlap
  grid, real trace b8/k8, isolated + e2e, correctness on).

**M4 conn=8 correction (2026-08-08, capsule 20260808-032217).** The first M4
capsule (20260808-015920) ran the moonep arms at launch.sh's default
CUDA_DEVICE_MAX_CONNECTIONS=1, and its "overlap buys nothing on NCCL"
reading was a QUEUE ARTIFACT: with one hardware connection the prefetch
stream's work serialized into the main stream's windows (prefetch_wait ~0
was an attribution illusion — the cost sat inside pack_ms at 6.4 ms). The
conn=8 rerun (same arms, same trace, only the connections knob) shows the
honest picture, isolated max-rank: moonep_overlap 8.89 ms vs moonep
serialized 10.48 ms — MoonEP's async_finish-style overlap genuinely hides
most of the ~4.7 ms prefetch (1.46 ms stays exposed at the join and ~1.3 ms
reappears as comm contention). Best lawful configuration is now
moonep_overlap (NCCL + overlapped prefetch). nvshmem arms: serialized
13.96 ms unchanged (one-sided putmem comm ~7.1 ms is conn-insensitive),
nvshmem_overlap improves to 10.02 ms but still trails NCCL. All four moonep
variants now pin CUDA_DEVICE_MAX_CONNECTIONS=8 in variants.py; pre-pin
cells ran conn=1 — audit env_json, never compare across the boundary.
