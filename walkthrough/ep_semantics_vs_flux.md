# MoonEP/UltraEP vs flux/Comet — Two Answers to the Same Straggler, Measured

**Scope**: the measured comparison between the EP-balancing arms (`moonep*`, `ultraep*`)
and the flux/Comet layer0 arms (`allgather`, `hier_compress_lb_union`) on Perlmutter
EP16 (4 nodes × 4 A100), real Qwen3-235B decode routing (layer 92 trace), grounded
exclusively in the four committed capsules
`sweeps/results/runs/20260808-{011748,015920,032217,090536}_perlmutter_*`. All four were
produced by **one binary** (`libflux_cuda.so` sha256 `23035ab8…`, manifests agree), which
is what makes the cross-capsule reads below lawful under `sweeps/SCHEMA.md` protocol
rule 4.

Companion docs:
[`ep_semantics_moonep_ultraep.md`](ep_semantics_moonep_ultraep.md) (the algorithms this
document assumes — read it first; its §1.3 toy example is reused below),
[`comet_layer0_communication_patterns.md`](comet_layer0_communication_patterns.md) (what
the flux arms actually do).

---

## 1. Two answers to the same straggler

Recall the toy of the companion doc: rank 1 homes the hot expert and receives 16 of 32
rows — twice its fair share. The two families answer differently:

- **flux/Comet: keep the placement, hide the wait.** Tokens go where the router said;
  the fused kernel decomposes the shared tensor into per-source segments and lets the
  grouped GEMM consume each segment the moment it arrives, so communication latency
  disappears behind tile computation. The a2av variants (`allgather`,
  `hier_compress_lb_union`, …) then shrink and re-shape what crosses the wire. But the
  hot rank still *computes* 16 rows: **compute skew survives overlap by construction.**
- **MoonEP/UltraEP: fix the placement, pay for the fix.** The expert kernel is relocated
  (token migration / expert replication) so no rank computes much more than its share —
  and the traffic flattens as a side effect. The price is new critical-path work that
  flux simply does not have: a planning wire (`plan_comm`), a pack copy, and above all
  **weight movement** (prefetch / weight_sync).

So the measured question is never "which is faster" in the abstract. It is: **does the
GEMM time bought back by balance exceed the weight-movement and planning time spent
buying it — at this shape, on this fabric?** On Perlmutter, at the swept shape, the
answer today is no (§3.1) — and the phase breakdowns say precisely why, which is the
point of measuring semantics rather than marketing.

---

## 2. What the sweep measures for each arm

Three reading rules from `sweeps/SCHEMA.md:105-176` (violating them produces nonsense):

1. **Latency = `isolated` mode**, max-rank per iteration, mean over iterations. Flux
   arms report it as `e2e_ms` (their comm/scatter/gemm phases *overlap*, so summing
   phases double-counts); EP arms report `total_ms` plus six *serialized* phases that do
   sum — except on `*_overlap` arms, where phases run concurrently and only `total_ms`
   is meaningful (`sweeps/SCHEMA.md:134-138`).
2. **The balance fingerprint is `gemm_rows_per_rank`.** Same routing, three shapes:

   ```
   flux arms   [7374, 5728, 14497, 5076, 5962, 9108, 10259, 3837, 11011, 6751, 11151, 9861, 4845, 5019, 5421, 15172]
   moonep      [8832, 8832,  8960, 8832, 8576, 8704,  8832, 8576,  8704, 8704,  8704, 8704, 8832, 8960,  8704, 8832]
   ultraep     [7922, 8251,  8251, 8251, 7364, 7365,  7365, 7072,  9904, 9108,  9901, 9861, 7694, 7693,  7381, 7689]
   ```

   Flux: 3,837–15,172 (the router's skew, verbatim — max/mean 1.85). MoonEP: constant
   8192 real rows + padding (perfect balance, §4.3 of the companion). UltraEP: flattened
   *within* each 4-rank node but node 2 (ranks 8–11) still high — per-domain solving
   leaves cross-node skew untouched, exactly as designed (§5.4).
3. **Wire matrices are outputs, not inputs.** `moonep_wire_bytes` differs from the input
   traffic matrix *by design* (the plan rebalanced it), and `ultraep_wire_bytes` counts
   un-dedup'd rows; neither may be compared to a flux arm's wire bytes as if they moved
   the same logical payload.

---

## 3. Measured results

### 3.1 The headline capsule: moonep vs the flux arms (20260808-011748)

Isolated latency (max-rank mean), plus the EP arm's serialized phase breakdown:

| arm | isolated latency | phase breakdown (ms) |
|---|---|---|
| `hier_compress_lb_union` | **4.10 ms** | (fused; phases overlap, not summable) |
| `allgather` | 5.62 ms | (fused) |
| `moonep` | 10.42 ms | plan_comm 0.36 · pack 0.35 · comm 3.44 · scatter 0.32 · **prefetch 4.68** · gemm 1.47 |

Disclosure: in this capsule the `moonep` cell ran at the launcher default
`CUDA_DEVICE_MAX_CONNECTIONS=1` (env_json audited; the conn=8 pin in
`sweeps/variants.py:32-40` postdates it), and prefetch is serialized — both known
pessimizations. Granting MoonEP its best lawful configuration from the conn=8 grid
(§3.3): **8.89 ms**. Same build, so the comparison stands: the balanced arm is ~2.2×
slower than `hier_compress_lb_union` at this shape.

The phase column says exactly why, and it is the honest core of this document:

- **The GEMM savings are real but small.** moonep's balanced gemm = 1.47 ms. The flux
  arms' skewed GEMM work rides inside a 4.10–5.62 ms fused window — even attributing all
  of `lb_union`'s window to GEMM would bound the possible saving at ~2.6 ms. At
  H=4096/ffn-shard=4096, 8k–15k rows of bf16 GEMM is simply not where this layer's time
  goes on A100.
- **The weight movement dwarfs the savings.** prefetch = 4.68 ms (rank 0 alone receives
  32 MiB of expert weights per activation over Slingshot,
  `moonep_prefetch_recv_bytes` — one full 4096×4096 bf16 fc1 shard) — larger by itself
  than `lb_union`'s entire layer. MoonEP's home turf makes this term small in
  ways Perlmutter cannot: NVSwitch-domain weight multicast, and inference `B=3–4` with
  symmetric-memory overflow instead of full per-activation prefetch
  (`MoonEP/README.md:56-59`).
- **The comparison is also a dedup-regime comparison.** moonep's comm (3.44 ms) carries
  dedup'd representative rows — semantically what the flux `lb_union` line already
  exploits with its union/compress machinery, but staged two-sided instead of fused.

> **Conceptual anchor.** Balance is a *lever on the GEMM term only*. Whether the lever
> pays depends on the GEMM's share of the layer, and on what the fabric charges for
> moving weights. At S·K=8192, H=4096 on A100/Slingshot the GEMM share is ~15–35% and
> weights are expensive — the lever loses. Scale the expert (larger H'), grow the batch,
> or move weights over NVSwitch, and the same arithmetic flips sign; that regime
> dependence, not a winner, is the transferable finding.

### 3.2 The conn=1 → conn=8 correction (20260808-015920 vs -032217)

The first M4 transport/overlap grid ran at conn=1 and read "overlap buys nothing on
NCCL" — a **queue artifact**: with one hardware connection the prefetch stream's work
serialized into the main stream's windows, `prefetch_wait` read ~0 while the cost hid
inside `pack_ms` (6.4 ms). The conn=8 rerun flipped it
(`docs/handoff/02_algorithm_state_and_next_moves.md:377-392`); isolated max-rank:

| arm | conn=1 (015920) | conn=8 (032217) |
|---|---|---|
| `moonep` (serialized) | 11.04 | 10.48 |
| `moonep_overlap` | 11.08 | **8.89** (prefetch_wait 1.46 exposed) |
| `moonep_nvshmem` | 13.95 | 13.96 |
| `moonep_nvshmem_overlap` | 12.14 | 10.02 |

Three durable lessons: (i) MoonEP's `async_finish`-style prefetch overlap genuinely
hides most of the ~4.7 ms weight movement once the hardware can express concurrency —
`moonep_overlap` is the honest best configuration; (ii) the one-sided NVSHMEM transport
(the *more* faithful port of MoonEP's wire) is conn-insensitive and slower here
(putmem comm ~7.1 ms) — transport fidelity and performance fidelity pulled in opposite
directions on this fabric; (iii) never compare across the conn boundary — the pin is now
in `variants.py`, and `env_json` audits every historical cell.

### 3.3 MoonEP vs UltraEP vs the rack-scale counterfactual (20260808-090536)

All three arms at conn=8, one capsule, isolated max-rank:

| | `moonep` | `ultraep` (D=4, faithful) | `ultraep_domain16` (counterfactual) |
|---|---|---|---|
| total | 10.49 | **10.23** | 13.11 |
| plan_comm | 0.23 (topk allgather) | 0.14 (8 KiB loads) | 0.13 |
| comm | 3.59 | 3.91 (no dedup; `dup_rows` 1347) | 3.95 |
| prefetch / weight_sync | 4.76 | 4.29 (64 MiB fc1+fc2, intra-node) | **7.10** (128 MiB, crosses Slingshot) |
| gemm | 1.47 (exact balance) | 1.63 (residual imbalance 1.209) | 1.50 (imbalance 1.029) |
| imbalance before → after | 1.852 → exact | 1.852 → 1.209 (floor 1.183) | 1.852 → 1.029 (floor 1.0) |
| solver | greedy, global | T=[8251,7365,9903,7692], all `fast` | T=[8426], `fast` |
| replicas | (token migration) | 11 of 32 slots, ≤2/expert | 14 of 32 |

This one table contains the whole design space:

- **UltraEP-D4 edges out MoonEP (10.23 vs 10.49)** despite worse balance: it moved fewer
  weight bytes to fewer places (11 replicas, all intra-node) and paid a far smaller
  planning wire (an 8 KiB loads table vs the full `[S,K]` routing all-gather). Its penalty shows exactly where theory says: gemm 1.63 vs 1.47 — the
  residual cross-node imbalance (floor 1.183) priced in silicon.
- **The rack-scale thesis, priced.** domain16 buys near-perfect balance (1.029,
  gemm back to 1.50) but its weight_sync doubles in bytes and crosses nodes: +2.8 ms,
  swamping the ~0.13 ms of GEMM it recovered. On a true rack-scale NVLink domain that
  sync is the 0.2–0.3 ms of `UltraEP/README.md:171-180` — the fabric assumption *is* the
  algorithm's viability, which is precisely the claim in the UltraEP paper's title.
- **Locality and dedup are small at this shape**: remote-token fraction 0.914 vs 0.932
  (locality on/off), 1347 dup rows (~1%) — real, audited, not decisive here.

### 3.4 Fairness ledger

Every known bias runs **against** the EP ports
(`docs/handoff/02_algorithm_state_and_next_moves.md:346-353`): two-sided staged a2av
plus two port-added local copies (measured separately in `pack`/`scatter` so `comm`
stays pure wire); replicated planning pays a visible `plan_comm` where MoonEP multicasts
a rank-0 plan in hardware; the base arm serializes prefetch (restored by
`moonep_overlap`); host planners (72–142 ms) excluded under the untimed-metadata
contract because the real kernels do it in ~0.1 ms. Protocol rules in force throughout:
compare inside one capsule/build; `deterministic = 0` on every quoted cell (verified);
`isolated` for latency, never phases-mode; never sum phases on overlap arms.

---

## 4. Open questions / next moves

- **The combine side is unported.** Both EP arms measure layer0 only; MoonEP's `combine`
  (gather + weighted reduce off the saved plan) is the mirror image against flux layer1
  (`moe_gather_rs`) and would double-count the balance benefit if it holds there too.
- **Shape sweep to find the crossover.** §3.1's arithmetic predicts the EP arms win when
  the GEMM term grows (larger ffn shard / batch) or the weight term shrinks
  (`B=3–4` inference prefetch semantics instead of full per-activation prefetch — the
  MoonEP README's own inference recommendation is *not yet an arm*).
- **UltraEP layer-60 control** (`sweeps/specs/ultraep_pm4n_trace60_iso.yaml`): the
  median-skew control for §3.3's layer-92 skew case — same three arms, one capsule.
- **Overlap for UltraEP**: weight_sync is un-overlapped in the port; upstream hides
  reroute under weight distribution and overlaps sync with `async_finish`
  (`UltraEP/README.md:138-142`) — the `moonep_overlap` precedent applies directly.
- **Hybrid question the toy already poses**: flux's overlap and EP's balance attack
  different terms of the same sum; nothing forbids a balanced *and* fused arm — MoonEP
  semantics feeding a Comet-style fused GEMM instead of staged `GemmOnly` is the obvious
  end state and the hardest port.
