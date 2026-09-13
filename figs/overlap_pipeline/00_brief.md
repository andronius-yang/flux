# Token comm/comp overlap figure — brief (opened 2026-09-10)

Lane for the methodology diagram that applies **split, then merge** to the
whole a2av dispatch/combine workflow at **4 nodes**: how the network flow is
broken into windows, what computation each window's arrival unblocks, and how
the combine side mirrors the dispatch side with one new compute stage.
`sys.drawio` (user upload, 26 pages) is the master drawing file; page
`sp_mg_v3` already explains split/merge as a *flow* mechanism on 2 nodes
(Split / Merge / Imbalanced NIC Egress / Split-then-Merge, per-GPU NIC and
NVLink timelines, "removed/shifted traffic" hatched bars). This figure reuses
that vocabulary and does NOT repeat it. Same conventions as
`figs/methodology/README.md` (every aesthetic value is a `[knob]`, rulings
dated, mechanism claims cite code/handoff/capsule). The methodology README
reserves `figs/methodology/token_comm_overlap/` for exactly this figure —
this lane is that sub-lane under the user's chosen name; cross-link or move is
an open ruling (§6).

Directive 1 check (methodology README): everything drawn here is the canon
Slipstream path run by the main-figure OURS arm (hier_compress + lb_union
+ fused stage-2 + blocking wire on layer 0; msplit + fused pack + bucket
receiver + Σ pre-reduce, wave-adapt 48, on layer 1). The Σ CTA ladder (§3.3)
is an ablation and is quoted as evidence for the dial only, never drawn as
the canon.

## 1. The three questions the figure must answer (user, 9/10)

1. **Split, then merge at 4 nodes, from one node's NIC point of view.** Does
   the pipeline remove the egress bubbles/bottlenecks of a single node? Where
   do "shifted" and "removed" traffic act?
2. **Gating.** After the flow is broken into windows, what computation does
   the arrival of one window (a group of tokens) unblock?
3. **Combine as the mirror.** How is overlap guaranteed there; how was the
   layer-1 GEMM order chosen to match the comm-driven schedule; and what
   *new* compute resource had to be introduced on the combine side?

Scenario ruling (user): 4 nodes; NIC view focused on one rank on one node,
that node drawn with two ranks; the computation it unblocks drawn elsewhere.

## 2. Mechanism ground truth (code-traced)

All layer-0 refs `src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc`,
layer-1 refs `src/moe_gather_rs/ths_op/gemm_grouped_v2_gather_rs.cc` unless
noted; `NN=4` nodes, `L=4` GPUs/node, `W=16`.

### 2.1 Dispatch (layer 0) — one sending rank `(node n, local rank lr)`

| step | what | fabric | code |
|---|---|---|---|
| ① merge | dedup to one row per (token, destination **rank**) for own-node peers and one **union** row per (token, destination **node**) for remote nodes (`u_mat` / `U_mat`, PXN identity) | — | `:1489-1524`, `:1549-1563` |
| ② split | the node's whole outbound stream to remote node `tn` (its 4 union segments) is cut into **L equal chunks** by `chunk_bound`; relay rank `k` pulls chunk `k` from its node-mates over NVLink into a staging buffer (all 3 rounds staged before any wire wait) | NVLink | `:1344-1360`, `:2893-2899`, `:2930-3069` |
| ③ wire | **one blocking `putmem_signal` per remote node** (`NN-1 = 3` per rank), mirror ring order `tn=(n-dn)%NN`, into the **same-local-rank gateway** `(tn, lr)` | NIC | `:3091-3168` |
| ④ gateway fan-out | as gateway for inbound rounds: wait the single-writer `node_sig[ns]`, then **one contiguous NVLink put per local destination** of the whole window, destinations visited in a per-(gateway, round) ring rotation so every destination sees a different first window | NVLink | `:3181-3248` (rotation `:3222-3227`, A/B capsule 20260804-043026) |
| ⑤ gate | the fused signal flips slot `ns*L+gl` (window identity); every GEMM tile spins (system-scope acquire) on the non-empty lanes it spans | GPU | `cutlass_impls/ag_scatter_gemm_grouped_with_absmax.h:569-702` |

Per rank per dispatch: 3 NIC puts, 3 NVLink pulls, 3 intra-node dedup puts,
12 gateway fan-out puts. The GEMM launch is gated on the relay staging event
(`relay_send_event_`, `:3070-3075`), i.e. it starts before any remote byte
lands and consumes own-node rows first.

**Window** = chunk `gl` of the `(source node ns → my node)` stream; 12 remote
windows + 4 local lanes = 16 lanes = `W`, which is why the signal array is
reused with no kernel change (`:3186-3191`; handoff 14 §E).

**What one window unblocks.** The fused stage-2 A order is lane-monotone
within each expert (`sort_util.h:163-181`, SCHEMA rule 9), so lane `w` owns a
contiguous M-slice `[cum[e][w-1], cum[e][w])` of **every** local expert `e`.
One window's arrival therefore releases, across all local experts at once,
the tiles covering that window's rows — a **row-slice release, not an expert
release**. The static Dim-M schedule is built stage-major: own local rank
first, then own node (the intra-node put order), then remote nodes at
ascending ring offset (= put-round order), and **inside each remote node's
block the four windows in the order the gateway fan-out rotation delivers
them to this rank** (`FLUX_A2AV_SCHED_ROT_ALIGN`, §13; `sort_util.h`
`shift_lane_to_order_rot`, `workspace_util.cu:36-130`). A tile is assigned
the latest stage it spans, so nothing is scheduled ahead of its dependency,
and the fleet consumes windows in delivery order at both levels.
(`moved-last` / `sched_expert_order` gates on *weight* arrival and belongs to
the expert-dispatch figure, not this one.)

**Shifted vs removed traffic, from one node's egress.** Removed: one union
row per (token, remote node) — 27-42 % of inter-node bytes on the trace cells
(insight ledger NR-01, `docs/handoff/03_insight_ledger.md:22-49`); NR-01's
caveat stands: the byte saving alone was worth ~0.3 ms, the win is the
structural one below. Shifted: the ¼-chunk pull moves bytes from a heavy
rank's NIC to its node-mates' NICs over NVLink (sender side), and the gateway
fan-out moves the per-destination copies off the NIC (receiver side). The
structural effect on a NIC: `NN-1` equal-sized messages per rank instead of up
to `W-L = 12` unequal ones, issued back-to-back on one stream — the per-NIC
egress is one contiguous block whose length is `ceil(node_total/L)` per round
regardless of which GPU's tokens they are (handoff 14 §C: equal cut is
near-optimal at `BW_nvlink >> BW_wire`; boundary snapping refuted).

### 2.2 Combine (layer 1) — the mirror

| step | what | fabric | code |
|---|---|---|---|
| ① split (M) | the grouped GEMM `x·w2` is decomposed into `n_waves × E` row sub-problems, **one wave per destination node**, ring order with **own node LAST**; the top-k coefficients are pre-folded on the K side and the epilogue writes straight into the destination-major send panel (fused pack, no extra pass) | GPU | `:177-184`, `:4001-4085`, `:4240-4247` |
| gate | tile→problem→**wave** completion counters (`barrier[wave]`); the pack relay flips `group_flags[tn]` when all E experts' rows for that destination node are done | GPU | `cutlass_impls/gather_rs_gemm_grouped_with_absmax.h:630-660`, `a2av_combine.cu:147-179` |
| ② converge | each rank pushes its slice for destination rank `(tn, dl)` over NVLink to the **same-local-rank peer** `gw=(n, dl)` — the gateway is on the **sender** side | NVLink | `:2360-2455` |
| ③ Σ pre-reduce | `a2av_combine_prereduce_kernel`: a **resident side-stream kernel** (`a2av_prered_stream_`, default **6 CTAs**, `get_a2av_prered_blocks`) spins on the 4 converge signals, then merges each row's CSR contributions in fp32 into one wire row per (token, source node) — **the new compute** | GPU (side) | `a2av_combine.cu:291-390`, `:1627`, `:2155-2210`, `:251-256` |
| ④ wire | **one blocking `putmem_signal` per remote node**, straight into the home rank's recv panel (no receiver gateway), on separate wire streams so successive waves pipeline | NIC | `:2460-2545`, SCHEMA rule 12 |
| ⑤ home fold | bucketed receiver: per-lane `cuStreamWaitValue64` in expected arrival order (own-node lanes first) releases `a2av_combine_bucket_reduce` over exactly the tokens that lane completes; exposed tail = one bucket | GPU (side) | `:2766-2890` |

Why the mirror needs a compute stage the dispatch side does not: dispatch
merges by *identity* (a pack — copies of the same token are the same bytes);
combine contributions to the same (token, destination) are **numerically
distinct partial sums**, so wire bytes are ∝ contributions unless they are
summed before the NIC (`figs/system_overview/00_brief.md:268-288`; handoff 21).
The price is the wave re-read of `w2` (4 passes at 4n; the true marginal is
0.43 ms, the split/padding 1.12 ms, handoff 28 §1) and the **wave-adapt 48**
dial (collapse to the single-gate GEMM when the re-read bytes exceed 48× the
remote wire bytes, `:246-268`, handoff 26 §2.1); at b64 the wave path is
always taken.

How the GEMM order matches the comm schedule: waves ARE the comm schedule
(destination-node ring, own node last so all remote waves go first and the
first wire put can fire after ~¼ of the GEMM); `RS_WAVE_ORDER` (largest
segment first) was an ablation that lost. Under an in-flight expert swap the
gated experts are ordered last *inside* every wave (handoff 36 §3) — the wave
axis is untouched.

Overlap guarantee: the cascade lets the wire of wave `k` overlap the GEMM of
waves `>k`, and the bucketed fold overlaps the lanes still in flight. It is a
pipeline, not a proof of full hiding — see §3.2 for how much is exposed at 4n.

## 3. Measured 4n structure (nsys captures, medians over 16 ranks)

Source: `$PSCRATCH/workspace/andrewy/figs_data/case_study/timeline_20260905-141104.json`
(capsule `sweeps/results/runs/20260905-141104_perlmutter_fdd1c4af`, arm
`ours_l01_s2_swap_p2p_t1_r2`, K2 4n b64 top-8, nsys mode — breakdown only,
never latency, SCHEMA rule 3). EFFICIENT = plain lcb `iter4`, SKEWED = S-C
proLaw `iter33` (the case-study pair, `figs/case_study/00_data_note.md`).
Extraction: the per-lane event dump in this session (`extract_timeline.py`
classes: `nic.put` = `nvshmemi_proxy_rma_*_blocking`, `nvlink.token` =
`memcpy_p2p`, `combine.prereduce` = `a2av_combine_prereduce_kernel`).

### 3.1 Dispatch

| landmark (ms from iteration start) | efficient | skewed |
|---|---|---|
| plan/meta bracket (host chain + NCCL allgather + meta kernels) | 0.7 → 2.6 | 0.9 → 3.2 |
| NVLink relay pull (② split), 3 copies | 3.95 → 5.30 | 4.5 → 5.3 |
| first inter-node put issued | 5.14 [4.2..5.8] | 6.09 [5.0..6.5] |
| puts per rank / each / inter-put gap | 3 / 3.4 ms / ≤0.01 ms | 3 / 3.7 ms / ≤0.1 ms |
| NIC busy per rank (contiguous) | 10.4 [8.7..11.4] | 11.6 [10.1..14.2] |
| l0 GEMM start (= first put + 0.2) / duration | 5.34 / 14.6 | 6.19 / 18.5 |
| own-node lanes landed | 6.65 | 7.09 |
| first remote window landed | 9.6 [8.4..10.4] | 11.0 [8.7..15.4] |
| last remote window landed | 18.6 [17.6..21.8] | 22.7 [20.8..25.0] |
| GEMM end − last window (exposed tail) | 1.45 [1.1..1.7] | 1.49 [0.5..5.4] |
| gateway fan-outs per rank | 9 (3 rounds × 3 peers), 0.57 ms each | 9, 0.61 ms |
| post-l0 barrier (inter-rank skew) | 3.3 [0.4..4.2] | 2.9 [0.4..5.5] |

Rank 0 (node 0) arrival order in both cases: own lanes, then node 1, node 2,
node 3 (mirror ring: node `k` sends to `k-1` first). Rank 5 sees 2, 3, 0 —
the rotation spreads first arrivals (agent-checked against `proxy_ranges`).

Egress contiguity, OURS vs the overlapped COMET baseline
(`l01_allgather_dense_nogate_c8`, same capsule, skewed iter33): OURS 16/16
ranks have ≤0.1 ms of idle inside their l0 NIC span; COMET's spans contain
0.1–3.2 ms of idle (different primitive — dense allgather — so quote as "the
OURS egress is gapless", not as a like-for-like bubble count).

### 3.2 Combine

| landmark | efficient | skewed |
|---|---|---|
| l1 GEMM start / duration (4 waves, own node last) | 24.5 / 10.9 | 29.3 / 10.2 |
| Σ pre-reduce resident kernel (side stream) | 24.5 → 37.0 (12.5) | 29.3 → 41.2 (11.9) |
| first converge burst (wave 0 closes) | GEMM + 2.9 | GEMM + 2.9 |
| converge copies per rank | 12 = 4 waves × 3 peers, ~0.3–0.8 ms each | 12 |
| **first wire put** | GEMM + 7.3 = wave 0 + **4.6** | GEMM + 7.4 = wave 0 + 4.5 |
| puts per rank | 3 (one per remote node) on 3 streams, staggered starts, all end together | 3 |
| last put end | 42.2 | 49.6 |
| home fold: small own-node folds / final fold | 35.3–39.7 / 42.3 → 45.3 (3.0) | … / 50.5 → 53.3 (2.8) |
| iteration end | 47.8 | 57.7 |

Reading: at 4n b64 the combine wire (≈10.5 ms of NIC) starts 7.3 ms into a
10.9 ms GEMM, so ~7 ms of wire plus the 3 ms final fold are **exposed** after
the GEMM. The dial is the Σ latency between a wave closing and its put.

### 3.3 Σ CTA ladder (capsule 20260910-132745, `timeline_20260910-132745.json`, skewed iter33, early arm)

| Σ CTAs | first put − l1 GEMM start | l1 GEMM duration | iteration end |
|---|---|---|---|
| 6 (canon) | 7.4 | 10.0 | 66.2 |
| 12 | 6.0 | 10.2 | 64.7 |
| 24 | 5.3 | 11.4 | 59.0 |
| 48 | 3.2 | 13.3 | 57.7 |

More resident Σ blocks pull the first put earlier (wave 0 + 4.6 → 1.2 ms) and
slow the GEMM they share SMs with (+3 ms at 48). This is the "new compute
resource" trade-off in one line. Ablation-only; the canon draws 6 CTAs.
(Iteration-end deltas are nsys-mode, indicative only.)

## 4. Diagram spec — `overlap_pipeline.drawio` (generated by `make_overlap_drawio.py`)

Two pages, one per sub-figure, same legend (sp_mg_v3 glyphs + Sysv5/CSv1
palette: token comm `#2a78d6`, compute `#eda100`, plan `#2f8f9d`, Σ/fold
`#c8553d`, wait `#c9c8c0`, hatched = removed/shifted). Left: schematic
topology (Node 0 with GPU0+GPU1 in detail, Node 1 with the gateway GPU4 and
GPU5, Nodes 2–3 compact). Right: resource timelines at `[knob] 15 px/ms`,
bars placed from §3 efficient medians (rank-0 arrival order).

**(a) dispatch_v1** — rows `N0·GPU0 {NIC, NVLink, GPU}`, `N0·GPU1 {NIC,
NVLink}`, `N1·GPU4 NVLink`, `N1·GPU5 {NVLink, GPU}`.
- NIC rows: three back-to-back puts labelled →N3 →N2 →N1 (mirror ring), equal
  length on GPU0 and GPU1 → answers Q1 (balanced, gapless egress).
- NVLink rows: hatched "pull" before the first put (② split = shifted
  traffic), then three fan-out bursts as inbound windows land (④).
- GPU rows: GEMM starts with the first put on own-node rows; gate ticks
  labelled own/N1/N2/N3 mark each window's landing → answers Q2 (a tick
  releases the row-slice of every local expert; GPU5's bar says "E3 tiles:
  own rows → N2 → N3 → N0").
- `N1·GPU4 NVLink` carries the callout "window (N0→N1, c0) lands → fan-out",
  linking the NIC of GPU0 to the compute it unblocks elsewhere.

**(b) combine_v1** — rows `N1·GPU4 {GPU, side stream, NVLink, NIC}`,
`N1·GPU5 {GPU, NVLink}`, `N0·GPU0 {NVLink, side stream}`.
- GPU row: GEMM segmented into waves w0→N2, w1→N3, w2→N0, w3 own (gate ticks
  at wave ends) → Q3 "how the order matches comm".
- Side-stream row: hatched Σ bar spanning the GEMM ("resident side-stream
  kernel, 6 CTAs") → Q3 "the new compute resource".
- NVLink row: converge bursts after each wave; NIC row: three thin stacked
  bars (concurrent streams sharing one NIC) starting at wave 0 + 4.6 ms, all
  ending together.
- Home rows: small own-node folds, "last lane" tick, final fold, barrier.

Review renders: `dispatch_v1.png/svg`, `combine_v1.png/svg`
(`render_drawio.py`, approximate — no ①–⑤ glyphs, no text wrap).

## 5. Structural insights to carry into the drawing (ordered)

1. **The window axis is the destination node**, on both layers: 3 puts per
   rank per layer at 4n, not per-expert traffic. Windows per rank: 12 remote
   + 4 local lanes on dispatch; 4 waves on combine.
2. **Dispatch is compute-paced at b64**: the wire (10–12 ms) finishes 3–5 ms
   before the GEMM (15–18 ms); the last window lands ~1.5 ms before GEMM end.
   Draw the GEMM as the longer bar; the bubbles are the plan bracket before
   the first put and the post-GEMM barrier, not NIC idle.
3. **Per-NIC egress is one contiguous block** of three equal-length puts on
   every rank of a node (split), while merge decides its length. That is the
   single-node answer to Q1; sp_mg_v3's "Imbalanced NIC Egress" panel is the
   before-picture.
4. **A window releases rows, not experts**: label the gate ticks with the
   source node, not with an expert name.
5. **Combine is wire-paced and the wire starts late**: first put at wave 0 +
   4.6 ms; ~7 ms of wire and the 3 ms final fold are exposed. The honest
   figure shows partial overlap on the combine side.
6. **The Σ stage is a resident kernel beside the GEMM**, sized by CTAs; its
   only role in the picture is to sit between a wave's converge and its put.
7. **Gateway on the receiver side for dispatch, on the sender side for
   combine** — the mirror is not a reflection of the same box.
8. The first inter-node put through the CXI proxy costs ~120 µs fixed per
   put (handoff 26 §3b), which is why windows are per node and not finer.

## 6. Open rulings (user)

1. Lane location: keep `figs/overlap_pipeline/` or move under
   `figs/methodology/token_comm_overlap/` (the README's reserved sub-lane).
2. Which case to draw: efficient medians (current draft) or the skewed
   iteration (longer windows, same structure, 1.5x rows spread).
3. Whether the combine sub-figure shows the exposed wire tail honestly
   (current draft: yes) or is schematic with the wire fully under the GEMM.
4. Whether the Σ CTA trade-off (§3.3) appears in the figure (a two-line
   inset) or only in the text.
5. `sys.drawio` (1.5 MB, user's master file) — keep untracked here or commit
   a copy.

## 7. Files

- `sys.drawio` — user upload (all diagrams so far; `sp_mg_v3` = flow split/merge).
- `make_overlap_drawio.py` → `overlap_pipeline.drawio` (pages `dispatch_v1`, `combine_v1`).
- `render_drawio.py` — review renderer (conda `andrewy-comet` python; cairosvg).
- `dispatch_v1.{png,svg}`, `combine_v1.{png,svg}` — draft renders.
- Data: `$PSCRATCH/workspace/andrewy/figs_data/case_study/timeline_20260905-141104.json`,
  `timeline_20260910-132745.json`, `byte_ledger_20260905-141104.csv`.

## 8. Σ pre-reduce kernel: microbenchmark (2026-09-10, job 58173823, 1 A100, standalone)

Question (user): is the Σ kernel slow because it is CTA-starved or because it is
written inefficiently, and would a dense zero-padded slot layout (the union-window
analog of the dispatch fan-out) make it cheap at 6–12 CTAs?
`bench_prereduce.cu` (+ `bench_prereduce_run_20260910.log`): bf16, n=7168, one
"wave" = wire rows × ~1.7–2.7 contributions in the shipped conv layout (peer
segments, token-ascending), 10-rep cudaEvent medians, all variants bit-checked
against the shipped structure (0 diffs).

| kernel (512 thr/CTA) | 6 CTAs | 12 | 24 | 48 | notes |
|---|---|---|---|---|---|
| k0 shipped structure (thread per 16 B pack, serial CSR walk) | 106 GB/s | 186 | 334 | 616 | linear in CTAs = per-SM latency-bound, ~17 GB/s per CTA |
| k1 CSR + ILP (2 packs/thread, indices prefetched, all loads issued first) | 159–188 | 298–351 | 559–655 | 994–1132 | 1.5–1.9× k0 at equal CTAs; 4/8 packs per thread regress (register pressure) |
| **k4 CSR + cp.async 6-stage streaming (row ring in smem, 86 KB)** | **222–240** | **427–468** | 796–886 | 1173–1257 | **2.2–2.4× k0 at 6 CTAs; 12 CTAs ≈ HBM/3** |
| k3 dense slots + presence mask (no zero traffic) | 90–114 | 178–224 | 347–436 | 667–807 | ≈ k1 at equal packs/thread: the CSR indirection costs ~5 % |
| k2 dense slots, zero-filled, unconditional S-way sum | 110 (1.8× bytes) | 219 | 432 | 822 | 1.8× slower than k0 at 6 CTAs on wall time: the zeros cost bandwidth |

Wave-0 sizing (12 000 wire rows, 470 MB moved): k0 4.41 ms / k1 2.94 / k4 1.96 at
6 CTAs; k4 1.00 ms at 12 CTAs. The production wave-0 Σ is 4.4 ms at 6 CTAs
(§3.2), so the shipped kernel runs at about its standalone rate.

Verdicts:
1. **The kernel is the culprit, not the CTA budget.** It keeps one dependent
   16 B load per thread in flight (index → data → accumulate → next), so 6
   CTAs hold ~50 KB in flight and saturate at ~17 GB/s per SM. A streaming
   rewrite (k4) reaches the 48-CTA ladder point (wave 0 + ~1.2 ms) with 12
   CTAs, i.e. the l1 GEMM grid shrinks 76→70 instead of 76→34.
2. **Padding does not pay.** The CSR walk is warp-uniform and L1/L2-resident
   (k3 ≈ k1); a zero-filled dense layout adds ~1.8× bytes and loses at every
   CTA count. The only padded variant that is not slower is the masked one,
   and it buys ~5 %. Not worth the extra NVLink bytes or the window-sized
   per-peer panels.
3. Caveats: standalone GPU (no GEMM competing for HBM; the l1 GEMM streams
   w2 at ~300 GB/s during Σ, so expect k4 at 6 CTAs closer to ~1.5× than
   2.3× in situ); synthetic contribution distribution (mean 1.7–2.7, topk-8
   bound); k4 needs 86 KB dynamic smem per CTA (fits a margin SM; the GEMM
   is not on those SMs). The wave-0 → put path also contains the conv wait
   and the wire-flag handshake (~0.3 ms), which no kernel change removes.

Next step if pursued: port k4's structure into `a2av_combine_prereduce_kernel`
(keep the per-tn signal spin and the block-count handshake; replace the
grid-stride pack loop with a per-CTA contiguous wire-row range + cp.async
ring), gate it behind `FLUX_A2AV_RS_PRERED_STREAM=1`, and re-run the 4n b64
nsys pair at 6 and 12 CTAs. Expected: first combine put at wave 0 + 1.5–2 ms
at 6 CTAs, ~1 ms at 12, with the l1 GEMM within +0.3 ms.

## 9. Streaming Σ kernel in situ — round 1 (capsule 20260910-232321, flux-prered, 4n b64, early arm)

Port: `patch_prered_stream.py` → `a2av_combine_prereduce_stream_kernel` behind
`FLUX_A2AV_RS_PRERED_STREAM=1` (tag `FLUX_A2AV_RS_PRERED_STREAM_TAG`, library
`libflux_cuda.so` fee1a3d5, ths_op unchanged d3bb40c7). Gate (capsule 224649,
reset-every proLaw, `--check_iters 1`): early arm @6 and @12 CTAs green; the
dual3 (default OURS) arm @6 went stuck — the same wedge class as dual3 × raised
grid (memory `combine-tail-decomposition-4n-b64`).

Isolated (median of rank-max), ms:

| arm | eff total / l1 | skew total / l1 |
|---|---|---|
| canon (shipped Σ, 6 CTAs) | 45.66 / 26.58 | 47.40 / 26.74 |
| streaming v1 @6 | 44.69 / 25.66 | 47.07 / 26.89 |
| streaming v1 @12 | 44.88 / 25.19 | 46.51 / 26.00 |
| shipped @48 (reference) | 41.38 / 22.01 | 43.03 / 22.59 |

nsys landmarks (16-rank medians, ms rel. to l1 GEMM start; conv0 = first converge
copy, put0 = first wire put, Σstart/Σspan = pre-reduce kernel):

| arm | fam | l1 GEMM | Σstart | Σspan | conv0 | put0 | putEnd | end |
|---|---|---|---|---|---|---|---|---|
| canon | eff | 10.6 | 0.0 | 12.2 | 2.8 | 7.4 | 17.8 | 26.6 |
| streaming v1 @6 | eff | 10.5 | **6.1** | 4.5 | **6.4** | 8.3 | 19.1 | 24.7 |
| streaming v1 @12 | eff | 11.0 | 6.4 | 3.8 | 6.6 | 8.1 | 18.7 | 22.8 |
| shipped @48 | eff | 14.1 | 0.0 | 10.0 | 2.3 | 3.5 | 13.7 | 21.1 |

**Diagnosis.** The kernel did its work 3× faster (Σspan 4.5 vs 12.2 ms, l1 GEMM
untouched, unlike @48 which costs the GEMM +3.5 ms) but it could not START until
~6 ms into the GEMM: `cuobjdump` shows 108 regs/thread (6 stages; 86 at 2/4) × 512
threads = 44–55K registers per CTA, while an SM already holding one l1 GEMM CTA
(128 thr × 254 regs) has only ~32K registers left, and SMs holding two have none.
The streaming CTAs therefore waited for GEMM CTAs to retire, and — since ~20 a2av
streams alias onto 8 hardware queues — the converge chain queued behind them
(conv0 2.8 → 6.4 ms). The first put moved only ~0.5 ms; l1 −0.9..−1.4 ms.
Residency budget for a resident side kernel next to the GEMM: **≤32K regs
(64/thread at 512 thr) and ≤~36 KB smem** (one 64 KB GEMM CTA + carveout).
This is also the mechanism behind the dual3 × raised-grid wedge: a resident CTA
that cannot be placed head-of-line-blocks its hardware queue, and dual3's gated
tiles wait on a pull copy behind it.

Round 2 (v2 patch): `__launch_bounds__(512, 2)` (64-reg cap), 2 packs/thread
(n_per ≤ 8192), `FLUX_A2AV_RS_PRERED_STAGES` (default 2 = 28 KB smem; 4 = 56 KB),
arms `prs` (2 stages), `prs4` (4 stages), `pb12_prs`; gate + pair rerun.

## 10. Streaming Σ kernel v2 in situ — early arm (capsule 20260911-000727, 20/20, binary a0c60c75)

v2 = `__launch_bounds__(512,2)` (50–52 regs), 2 packs/thread, ring depth
`FLUX_A2AV_RS_PRERED_STAGES` (2 = 28 KB default, 4 = 56 KB). Gate (capsule
235745): early @6 (2 and 4 stages), early @12, **and dual3 @6 all green** — the
register fix removed the dual3 wedge that v1 hit. dual3 @12 still wedges (gate
capsule of the dual3 chain): the resident budget on dual3 tops out at 6 Σ CTAs
next to 10 pack + 8 reduce CTAs and the movement streams' pull copies.

Isolated (median of rank-max), ms:

| arm | eff total / l0 / l1 | skew total / l0 / l1 |
|---|---|---|
| canon (shipped Σ, 6 CTAs) | 43.82 / 19.25 / 24.68 | 46.73 / 20.77 / 25.88 |
| streaming 2-stage @6 | 41.67 / 19.39 / 22.08 | 44.59 / 20.64 / 23.89 |
| streaming 4-stage @6 | 40.87 / 19.18 / 21.53 | 44.83 / 20.72 / 24.12 |
| **streaming 2-stage @12** | **40.16 / 19.19 / 20.80** | **43.83 / 20.90 / 22.78** |
| shipped @48 (reference) | 41.76 / 19.15 / 22.48 | 43.18 / 20.80 / 22.83 |

nsys landmarks (16-rank medians, ms rel. to l1 GEMM start):

| arm | fam | l1 GEMM | Σstart | conv0 | put0 | putEnd | foldEnd | end |
|---|---|---|---|---|---|---|---|---|
| canon | eff / skew | 10.8 / 10.1 | 0.0 | 2.8 / 2.9 | 7.5 / 7.6 | 18.9 / 21.4 | 22.0 / 25.1 | 24.2 / 31.1 |
| streaming 2-st @6 | eff / skew | 11.1 / 10.3 | 0.0 | 2.8 / 2.7 | 6.2 / 5.8 | 16.6 / 18.1 | 19.6 / 23.0 | 20.5 / 27.5 |
| streaming 4-st @6 | eff / skew | 11.4 / 10.1 | 0.0 | 2.9 / 2.7 | 5.9 / 5.5 | 16.7 / 18.4 | 19.7 / 23.1 | 20.4 / 27.0 |
| streaming 2-st @12 | eff / skew | 11.7 / 10.8 | 0.0 | 3.1 / 2.9 | 5.2 / 4.7 | 15.6 / 17.1 | 18.6 / 22.1 | 19.7 / 26.0 |
| shipped @48 | eff / skew | 14.6 / 13.4 | 0.0 | 2.3 / 2.2 | 3.5 / 3.3 | 14.6 / 16.2 | 19.5 / 22.2 | 21.6 / 25.8 |

Reading:
- **Wave-overlap potential.** The Σ of wave 0 now takes 3.3 ms (2-st @6) /
  3.0 (4-st) / 2.1 (2-st @12) after the converge instead of 4.7, so the wire
  starts 1.3 / 1.6 / 2.3 ms earlier and — since the wire drains at line rate —
  ends that much earlier; the fold end and the iteration end follow
  one-for-one (end −3.7 / −3.8 / −4.5 ms efficient, −3.6 / −4.1 / −5.1 skewed
  in nsys mode). The shipped kernel at 48 CTAs still starts the wire earliest
  (3.5) but pays +3.8 ms of l1 GEMM for it, which is why the streaming @12
  arm wins on total.
- **total_ms (isolated).** Streaming @6 buys −2.2 / −2.1 ms of total
  (−2.6 / −2.0 of l1); @12 buys −3.7 / −2.9 (−3.9 / −3.1 of l1), ahead of the
  48-CTA ladder point on the efficient case and within 0.6 ms on the skewed
  block, with l0 unchanged. The in-situ Σ speed-up is ~1.5× per CTA (HBM is
  shared with the w2 stream of the GEMM), not the 2.3× of the standalone
  bench; ring depth 4 vs 2 is worth ~0.3 ms.
- **Residency is the design constraint** for every resident side kernel:
  ≤32K registers and ≤~36 KB smem per CTA (one 64 KB GEMM CTA per SM plus
  carveout). v1 violated it and lost the whole gain (§9).

## 11. Streaming Σ kernel v2 on the dual3 arm (capsule 20260911-004921, 8/8, binary a0c60c75)

dual3 = the default OURS arm (user ruling 9/10: w1 under the l0 GEMM, w2 under the
l1 GEMM, GEMM-start-mark gated). Gate: dual3 + streaming @6 green (capsule 235745);
dual3 + streaming @12 wedged 2/2 (capsule 003853) → the dual3-safe dose is 6 CTAs.

Isolated (median of rank-max), ms:

| arm | eff total / l0 / l1 / place | skew total / l0 / l1 / place |
|---|---|---|
| dual3 canon (shipped Σ, 6 CTAs) | 45.12 / 19.39 / 25.67 / 1.21 | 47.27 / 20.47 / 27.14 / 1.46 |
| **dual3 + streaming 2-stage @6** | **40.82 / 19.16 / 21.43 / 1.21** | **44.16 / 20.45 / 23.76 / 1.42** |
| Δ | −4.3 / −0.2 / −4.2 / 0 | −3.1 / 0 / −3.4 / 0 |

nsys landmarks (16-rank medians, ms rel. to l1 GEMM start; swap = NVLink expert
movement blocks, w1 rel. to the l0 GEMM start, w2 rel. to the l1 GEMM start):

| arm | fam | l1 GEMM | Σstart | conv0 | put0 | putEnd | foldEnd | end | w1 | w2 (busy, under Σ) |
|---|---|---|---|---|---|---|---|---|---|---|
| dual3 canon | eff | 10.8 | 0.0 | 2.6 | 8.3 | 18.8 | 22.1 | 25.6 | — (no swap fires) | 0.32 ms, 0.32 |
| dual3 canon | skew | 10.2 | 0.0 | 2.5 | 8.3 | 21.3 | 28.8 | 33.4 | −0.04..0.63 | 0.00..0.54 (0.76, 0.76) |
| dual3 + streaming @6 | eff | 11.3 | 0.0 | 2.6 | **5.5** | 16.1 | 19.2 | **20.4** | — | 0.32, 0.32 |
| dual3 + streaming @6 | skew | 10.3 | 0.0 | 2.5 | **5.1** | 18.3 | 26.0 | **28.8** | −0.05..0.62 | 0.00..0.54 (0.75, 0.74) |

Reading (dual3):
- **Expert-movement overlap is untouched.** Both NVLink blocks sit exactly where
  handoff 36 §13.1 put them: w1 starts with the l0 GEMM, w2 starts with the l1
  GEMM, all busy time under a GEMM. The w2 pulls (0.0–0.54 ms) run while the
  streaming Σ kernel is resident but still spinning on the wave-0 converge
  signals (first converge at 2.5 ms), so Σ and the w2 copies never compete for
  HBM, and Σ's early residency (start 0.0) is what keeps the movement stream's
  pull copy from stalling — the v1 kernel, which could not become resident,
  is exactly what wedged this arm.
- **Wave-overlap.** wave 0 → first put drops from 5.7 ms to 2.6–3.0 ms; the
  wire ends 2.7–3.0 ms earlier, the final fold 2.8–2.9 ms earlier, and the
  iteration end 4.6–5.2 ms earlier in nsys mode (−4.3 / −3.1 isolated). The
  remaining exposed combine tail is the line-rate wire (≈10–13 ms from the
  first put) plus the ~3 ms final fold; the next lever is the wire start
  floor (wave 0 close + converge ≈ 2.6 ms) and the fold, not Σ.
- **Dose.** 12 Σ CTAs wedge dual3 (2/2) even with the residency-safe kernel:
  the resident population (10 pack + 8 reduce + 12 Σ + movement pulls) no
  longer fits the single-GEMM SMs. 6 is the dual3-safe dose; the early/no-swap
  arms tolerate 12 (§10) for a further −1.5 / −0.8 ms.

## 12. Recommendation (2026-09-10 eve)

1. **Canonicalize** `FLUX_A2AV_RS_PRERED_STREAM=1`, `FLUX_A2AV_RS_PRERED_STAGES=2`,
   `FLUX_A2AV_RS_PRERED_BLOCKS=6` as binary defaults (tag
   `FLUX_A2AV_RS_PRERED_STREAM_TAG`, never-mix boundary vs pre-flip capsules,
   SCHEMA rule 4). Do not raise the grid: 12 is a no-swap-only dose.
2. **Which variants it touches.** The Σ stage exists only in the
   `a2av_hier_compress` combine, so the flip applies automatically to every
   arm of that lineage (OURS s1/s2/dual3, LLC, EPIC-compress twins) and to
   nothing else: COMET/dense, FAST, NVSHMEM/NCCL ring baselines have no
   pre-reduce. It is a mechanism improvement of "our" combine, drawn in the
   methodology figure as the same Σ box — only its cost changes.
3. **Partial re-sweep, not a full one.** Baseline numbers cannot change (they
   never run the kernel); what changes is the OURS-family bars and, per rule 4,
   the requirement that every quoted comparison come from one binary. So:
   - re-run the plotted OURS arm (`ours_l01_s1_pv2_r2`, and dual3 where it is
     quoted) at **b16 / b32 / b64 × 4n / 8n / 16n × K2 / Qwen** — the cells
     where the wave path is engaged (wave-adapt 48 collapses b1–b8 to the
     single-gate GEMM, where Σ runs after the GEMM and the tail is put-count
     bound) — in fresh capsules that each also carry the strongest baseline of
     that group (COMET overlapped at 4n/8n, NVSHMEM ring at 16n) for the
     same-binary anchor: 18 OURS cells + 18 anchors ≈ 36 cells, ≈ 2 nh at 4n,
     ≈ 6 nh at 8n, ≈ 14 nh at 16n (regular QOS for 8n/16n);
   - one b1–b8 spot check per topology (K2, OURS only) to prove no regression
     where the collapse rule fires; expect ties;
   - the case-study rows (`casestudy3d` dual3) and the ablation-cycling
     schedules are already same-binary pairs here (capsules 000727/004921) and
     need no re-run for the figure text; the case-study figure itself should
     be recut from 004921 if it is to show the new combine.
   A full 7-baseline × 3-topology × 7-budget re-sweep (~40 nh) would only
   re-measure baselines that cannot move; the partial plan bounds the binary
   drift with the per-capsule anchors instead.
4. **Housekeeping.** Everything lives UNCOMMITTED in the flux-prered worktree
   (kernel patch in `src/moe_gather_rs/a2av_combine.cu`, `sweeps/variants.py`
   arms, five specs, capsules 224649 / 232321 / 235745 / 003853 / 000727 /
   004921); the reusable patch is `figs/overlap_pipeline/patch_prered_stream.py`.
   The main checkout and flux-3dsched were restored to their prior binaries.

## 13. Window consume order = delivery order (addon, 2026-09-11)

**What was wrong.** The lb_union static schedule ordered each remote node's
L window lanes own-lane-first (`shift_rank_to_order`, ring shift by my local
rank) while the gateway fan-out visits destinations in a rotation
(`dlg = (g + 1 + dn + dl) % L`, ag_scatter `.cc` ~:3228, NR-06 A/B kept it),
so e.g. GPU4 received node 0's windows as W1, W0, W3, W2 but consumed them
W0, W1, W2, W3. Node-to-node order (put rounds) was already aligned. This is
handoff 05's H1 ("rotation anti-alignment", verdict "visible, small",
never fixed).

**Fix (option 1, user-queued 9/11).** Env knob `FLUX_A2AV_SCHED_ROT_ALIGN=1`
→ `args.a2av_rot_align` (only the window-keyed lb_union schedule:
`union_bcast && !relay_identity && gating cumsum`). New maps
`shift_lane_to_order_rot` / `revert_order_to_lane_rot` (`sort_util.h`):
own node keeps the ring shift (matches the intra-node dedup put order
`dlg = (my_lr - dl) % L`); remote lane (ns, gl) at ring offset dn gets
in-block position `(my_lr - gl - 1 - dn) mod L`. CPU-verified against a
simulation of the gateway loop (L=4/8, NN=2..16, all ranks, 0 mismatches).
Host-side table only; kernel unchanged; rotation (NVLink balance) kept.

**Verdict, 4n (K2 lcb, isolated, 16 iters, one binary, worktree
`flux-rotalign`).** Gate capsule `20260911-075057_perlmutter_3be47df1`
(knob off/on × b8/b64, `--check_iters 1`, random payload): 4/4 ok, every
iteration "gate OK, 0 bad rows". A/B capsule
`20260911-080007_perlmutter_30297449`, median over iters of
max-across-ranks, ms:

| budget | total off → on | l0 off → on | l0 Δ |
|---|---|---|---|
| b1 | 4.01 → 4.00 | 1.66 → 1.65 | −1.0 % |
| b4 | 5.88 → 5.96 | 2.39 → 2.35 | −1.3 % |
| b16 | 14.23 → 14.14 | 6.11 → 5.88 | −3.9 % |
| b64 | 46.35 → 45.24 | 19.19 → 17.96 | −6.4 % |

l1 flat everywhere (the b4 total +1.3 % is l1 noise). User ruling: on par
or better ⇒ H1 CLOSED; the figure draws the GEMM consuming windows in
NVLink delivery order.

**Verdict, 8n** (same binary, debug QOS, 12 iters, `sm_margin 8`), capsule
`20260911-085902_perlmutter_4b3a99df`, 8/8 ok, deterministic 0:

| budget | total off → on | l0 off → on | l0 Δ |
|---|---|---|---|
| b1 | 5.94 → 5.79 | 2.76 → 2.75 | −0.4 % |
| b4 | 8.51 → 8.60 | 3.67 → 3.73 | +1.7 % |
| b16 | 19.15 → 19.13 | 7.92 → 8.02 | +1.2 % |
| b64 | 64.82 → 63.99 | 26.55 → 26.47 | −0.3 % |

On par: every delta sits inside the per-cell interquartile range. The 4n
layer-0 gain does not reproduce at 8n, consistent with each remote node's
windows being ~half the bytes at 8n (7 remote nodes share the stream), so
the within-node order costs less. Canon flip (knob into the
`ours_l01_s1_pv2_r2` env or binary default) = user ruling.
