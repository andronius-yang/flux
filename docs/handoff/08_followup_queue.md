# 08 — Follow-up queue after the comm-only layer-axis campaign

Handoff for a fresh session starting the post-campaign work. Read
`docs/handoff/07_comm_only_layer_axis_campaign.md` FIRST — it is the campaign
authority (verdicts, capsule ledger, both bug narratives); `sweeps/SCHEMA.md`
governs all sweep interpretation; `docs/handoff/07_dashboard.html` is the
self-contained visual summary (open locally in a browser). The auto-memory
files (`comm-only-l0-l1-sweep-plan`, `l1-a2av-lazy-load-hang`,
`l1-cascade-empty-expert-hang`) carry the same facts in ledger form.

State of the tree: everything is merged to local `main` (NOT pushed anywhere
except the campaign branch `worktree-comm-sweep-layer-axis` on the fork).
The installed binary in `python/flux/lib/` is **binary C** (both hang fixes;
ths-op sha256 prefix `bd25c058…` was binary B — verify against the newest
capsule manifests, e.g. `20260816-397ac0fa`, before trusting). Worktrees
`comm-sweep-layer-axis`, `l1-hang-debug`, `l1-nn4-debug` are all merged —
purge when convenient (`git worktree remove`), remembering the
lib64-shadows-lib and stale-.so gotchas in the memory ledger.

Environment: `source ./module.sh`; account **m5350_g** (m4243_g exhausted);
salloc+srun only, never sbatch; scancel the moment jobs finish; take jobids
only from your own salloc output (two sessions collided on this once).

---

## 1. Canonicalization patches (highest value, smallest risk)

**What**: make the campaign's winner configs the defaults.
- Flip `FLUX_A2AV_FUSED_STAGE2` and `FLUX_A2AV_EARLY_LAUNCH` to default-ON
  when `FLUX_A2AV_LB_UNION=1` (env reads in
  `src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc`; EARLY_LAUNCH
  keeps its existing guards — forbidden with PACK_OVERLAP, needs conn>1 on
  compress paths, which the variant env already satisfies).
- Delete the `hier_compress_lb_union_eager` (FANOUT) A/B arm from
  `sweeps/variants.py` — its comment explicitly awaited this verdict —
  and collapse the factorial corner arms into base + explicit ablations.

**Why**: three-run sign-agreement verdicts (handoff 07 §2): F wins
−0.2…−0.7 ms, E wins −0.3…−1.8 ms, N (FANOUT) loses +0.05…+0.6 ms on real
4n trace routing.

**How/validate**: after the default flip, old capsules' env_json means
something different — never byte-compare env across the boundary (add the
same style of dated note the conn=8 pin has in variants.py). Rebuild, then
one 2n correctness-ON smoke of lb_union base (which now silently runs F+E)
+ `--skip_correctness` off; update SCHEMA/SKILL notes. C++ change ⇒ rebuild
required (login builds ≤8 jobs; tests OFF — the test binaries hit a GLIBCXX
link error vs libnvshmem, which is why build.sh must run with `--no_test`
and a fresh cmake cache).

## 2. Combined default pairing = compress

**What**: make `lb_union(F+E) + a2av_hier_compress` the reference combined
configuration: in `sweeps/variants.py`, promote the winning pairing (see
`l01_lbunion_compress` added by the 4n session; align naming), and document
in SCHEMA that combined cells inherit compress's CSRs (amortized semantics)
so compress's isolated-mode build penalty does not apply there.

**Why**: the W=16 best-pairing A/B (capsules `9378fed5`/`397ac0fa`, 14/14
×2) — compress pairing wins EVERY budget: 10.6 ms b8, −52% vs stock,
−45…−52% across b2–b64. Standalone-l1 verdicts differ (hier wins iso at
small budgets) — keep both stories straight per handoff 07 §3.1/§4.1.

## 3. Eager arrival-order reduce: disposition

**What**: decide the fate of `FLUX_A2AV_RS_EAGER`. Evidence: regression at
BOTH scales standalone (+15–35% W=8, +2–11% W=16) AND the entire combined
composition penalty (+18% identity violation at b8, ablation-attributed —
handoff 07 §4). Options: (a) keep as opt-in ablation with a verdict comment
(cheapest, preserves history), (b) delete kernel + knob. Recommend (a) now,
(b) only with user sign-off. Either way, annotate `l1_hier_eager` /
`l1_compress_eager` in variants.py with the verdict so nobody re-runs them
expecting a win.

## 4. gemm_rs `set_barrier_ptr` audit (correctness risk, unaudited)

**What**: the empty-expert split-cascade bug (fixed in moe_gather_rs by
`c9b82b6`) divides the full problem list by the non-empty problem count.
The dense-MLP sibling `src/gemm_rs/` has its own `set_barrier_ptr`
machinery that was NEVER audited for the same pattern.

**How**: read the gemm_rs split-cascade / barrier-pointer code for the same
full-list-index vs non-empty-count mismatch; the trigger is any zero-size
segment. If affected: minimal repro (a split with an empty segment), apply
the analogous one-liner, 2n validation. If clean: record the audit result
in handoff 07's open-items so it stops being listed.

## 5. `l01_fast` (fast+fast combined arm)

**What**: `test/python/moe_combined/test_moe_l0l1_traffic.py --impl fast`
is a deliberate `NotImplementedError` stub. Open question: one
`flash_comm_t` doing TWO alltoallv calls per timed window (dispatch matrix
M then combine Mᵀ) — do FAST's credits/signals need an in-window
`alltoallv_reset` between the calls? Fallback design: two comm objects.

**How**: resolve at a 2n bring-up (capacity formula `4·max(row,col sum)` is
transpose-symmetric, so sizing is fine either way); validate against the
torch two-layer reference; then add the `l01_fast` variant (e2e-only,
≥2 nodes, `requires_file` libflash) and a small capsule. Reference
implementations: `test/python/moe_gather_rs/test_moe_gather_rs_fast_baseline.py`
(the l1 direction, validated) and the layer0 fast bench.

## 6. 16n W64 closure debt (pre-campaign, still open)

**What**: the 2026-08-15 W64 fixes (predicate segment gate 55e0273, exact
knobs, retry-routing) were never re-validated at 16n: allgather
correctness-ON b1–b16, `hier_compress_lb_union` b32/b64, and (if EP arms
matter again) moonep_fused at W64. Now ALSO worth folding in: layer1 at
16n has never run (the empty-expert fix makes it possible for the first
time).

**How**: needs `-q regular` (interactive QOS caps at 4 nodes), G question
(G=192 was the historical 16n deviation — the trace generator will state
the minimum; document whichever is used), NVSHMEM_SYMMETRIC_SIZE handled by
the exact sizers. Budget several hours of queue+run; capsule protocol as
always (fwd+rev if any verdict is claimed).

## 7. Small hygiene items

- The campaign's variants.py still carries the five factorial corner arms
  and the binary-A-era comments; prune once item 1 lands.
- `docs/handoff/00_START_HERE.md` should gain a one-line pointer to
  handoff 07 as the layer-axis campaign authority.
- The two debug fixes (`1550b67` lazy-load preload, `c9b82b6` empty-expert)
  are candidates for an upstream PR to bytedance/flux — user's call.
- Capsule count is growing (150+ under sweeps/results/runs) — no action
  needed, but analysis scripts should filter by the campaign notes strings.

---

Protocol reminders that bit us and will bite again: compare arms only
within one capsule/one BUILD (manifest `flux_libs` sha, not git_sha);
ordering-sensitive claims need fwd+rev sign agreement; trace cells with
`routing_mode=''` are dealer-poisoned; W=8 and W=16 never mix; when trace
cells hang, bincount the routing for empty experts before anything else.
