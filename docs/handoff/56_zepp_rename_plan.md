# 56 — Rename campaign LoPEP → Zepp (2026-10-04, EXECUTING)

Lives on PSCRATCH (`logs/zepp_rename/`) because home is over quota (40.33 / 40 GiB on 10-04). It belongs in the
research tree's `docs/handoff/` once home has space.

Scope: rename the system in the open-source repository's two branches (`master` = published tree,
`sglang-dev`) and in the release snapshot, without breaking the frozen experiment record and without
any silent behavior change. Everything below was measured on 2026-10-04.

## Directives (user, 2026-10-04): these supersede §0's recommendations where they differ

1. Names as in §3.
2. Zepp has no expansion. Paper: *Zepp: Accelerating Distributed MoE Serving under Relaxed Balance Constraints*.
3. Release author email (also the user's email): `androniusyang@gmail.com`, so the author is `Zepp authors <androniusyang@gmail.com>`
   (author and committer).
4. A clean release: a new `zepp` repo with branch `main` = the system (master 6658951, renamed) and branch
   `sglang-dev` = the consolidated, final canonical serving tree = the last tested one, p10-f 3d83066 (= sglang-dev
   tip; used for the SGLang figure and the p10_fv validation).
5. Claude must not appear anywhere in the zepp commit history. The lopep history has 0 hits in messages, authors and
   committers, but file contents in 27ed592 (and their removal in c6217a0 / 975b22d) contain "CLAUDE.md". That
   history also contains the measurement CSVs removed on 9/28 and every old name. So **the zepp history starts
   fresh**: `main` = one commit and `sglang-dev` = one consolidated commit on top. The full lopep history stays in
   `lopep/` as the record.
6. D6 (old snapshot): the user did not know what it was. `lopep_release` is left untouched (no move, no delete).
- Everything stays on PSCRATCH (`$PSCRATCH/workspace/andrewy/zepp`). No remote, no push.
- Consolidation fixes in sglang-dev (release rule 1): the only 3 provenance hits are the default conda env and the
  cache path in `env/perlmutter_sglang.sh` and the `conda create -p` line in `integrations/sglang/README.md`.
  0 timing numbers, 0 data files.
- Home is full, so research-tree changes (oss_audit.sh, CLAUDE.md, the four log parsers) are staged as a patch under
  `logs/zepp_rename/research_tree/` and applied once home has space.

## 0. Decisions for the user before P0 (answered above)

| # | Decision | Recommendation |
|---|----------|----------------|
| D1 | Spellings (§3 table): `Zepp` / `zepp` / `ZEPP_*`, adapter `zepp_sglang`, SGLang backend value `zepp`, library `libzepp_cuda.so`, class `ZeppFusedMoE` | as in §3 |
| D2 | What replaces "LoPEP (**Lo**cality-**P**referred **E**xpert **P**arallelism)" in README line 3 and the paper title in README line 12 (`*LoPEP: Locality-Preferred Expert Parallelism*`). Zepp is not an acronym of that phrase. The paper's own title has to change at the same time. | User and postdoc decide; "locality-preferred routing" stays as the name of the mechanism |
| D3 | Release author email. `Zepp authors <zepp@users.noreply.github.com>` (the pattern used today) **resolves to the existing GitHub user `zepp`** (a private person's account, created 2011). GitHub would show that person as the author of the release commit. | Use an address the authors control (the publishing org's noreply address, or a project mailbox). Decide the GitHub org first. The org cannot be named `zepp` because that user exists. |
| D4 | Repo layout | **New local clone `zepp/`** that carries only `master` + `sglang-dev`. `lopep/` and its 17 worktrees stay frozen as the record. The alternative is `mv lopep zepp` + `git worktree repair`. It breaks the 17 ops-script references to `andrewy/lopep`, the conda editable install, and the paths cited in the handoffs, and it mixes old-named experiment branches into the new repo. |
| D5 | History | **Forward rename commits** on each branch, the same mechanism the 9/28 rename used (cf5b146 on master, c38fa02 on sglang-dev). Rewriting history would orphan every cited hash (p10-f 3d83066, 068e5e4, …). |
| D6 | Old snapshot `lopep_release` (199eec0) | Move it to `archive/`. Do not delete it. It has never been pushed. |

Name-risk notes for D1, not blockers: "Zepp" is the consumer brand of Zepp Health Corp. (ticker ZEPP, Zepp OS, the
Zepp app), so web search results for the name will be crowded. The PyPI name `zepp` is free today (404), and no
installed package in andrewy-sglang or andrewy-comet uses it.

## 1. Inventory (measured 10-04)

| Surface | Measured |
|---|---|
| `master` 6658951 | 27 files; lines per spelling: `lopep` 54, `LOPEP` 11, `LoPEP` 4; 13 paths under `python/lopep/`; 3 `LOPEP_*` vars |
| `sglang-dev` 3d83066 (= p10-f) | 83 files; `LOPEP` 520, `lopep` 342, `LoPEP` 9, `Lopep` 9; 21 paths (`python/lopep/`, `integrations/sglang/lopep_sglang/`); **110 distinct `LOPEP_*` vars** read at 44 `getenv` sites (C++) and 95 `os.environ` sites (python) |
| `lopep_release` 199eec0 | 1 commit, tree ac6b34c == master's tree |
| Remotes | **none** on `lopep` or `lopep_release`. Nothing is published, so the whole campaign is local and reversible. |
| Collisions | `zepp` (any case) occurs nowhere in either branch, in the cutlass submodule, or in the conda envs, so the map is collision-free and invertible. No binary file contains the name. |
| 9/28 precedent | 0 `moe_ep` leftovers in content on either branch. It did leave the untracked `build_stale_moe_ep/` in `lopep/` (see H2). |
| Outside the branches | SGLang clone `sglang/` (working-tree patch, 23 refs incl. the backend value); conda editable `lopep_sglang` → `lopep/integrations/sglang/lopep_sglang`; 45 calib dirs holding `lopep_config.json` (no name in the content); ops scripts `logs/sglang` 64 + `logs/p50` 119 + `logs/p55` 1 (paths into 18 `lopep*` trees + `LOPEP_*` exports); log parsers keyed on `[lopep timing rank` / NVTX `lopep.step`; `scripts/oss_audit.sh`, CLAUDE.md release section, memory |

## 2. What stays LoPEP on purpose (records)

Handoffs 44–55, `figs/*/data` manifests and figure_src, `49_plan6_csv.py`'s `lopep=` column, logs, branches
p8-*/p9-*/p10-* and the 17 worktrees (they hold the binaries behind every serving number), `sglang-dev` history
before the rename commit, and the `lopep/` repo itself. The rename runs forward from the two branch tips and does
not rewrite records.

## 3. Name map (one file, `rename_map.tsv`, read by the forward script, the reverse check and the audit)

| Old (case-sensitive) | New | Examples |
|---|---|---|
| `LOPEP` | `ZEPP` | 110 env vars, `project(LOPEP …)`, `LOPEP_KERNEL_ENTRY`, `MoeA2ABackend.LOPEP` |
| `LoPEP` | `Zepp` | prose, NOTICE, `LoPEPFusedMoE` |
| `Lopep` | `Zepp` | `LopepDbgSlot`, `kLopepDbgEvents` |
| `lopep` | `zepp` | package, `lopep._C`, `liblopep_cuda.so`, CMake targets `lopep_dispatch/_combine/_direct_wire/_direct_gemm`, `lopep_add_cu_obj_lib`, `lopep_sglang`, `is_lopep`, `forward_lopep`, `--moe-a2a-backend lopep`, `lopep_config.json`, `[lopep]` log prefix, NVTX `lopep.step`, `lopep_dbg_*` |
| path `python/lopep/` | `python/zepp/` | `git mv` |
| path `integrations/sglang/lopep_sglang/` | `integrations/sglang/zepp_sglang/` | `git mv` |

The script replaces each spelling case-sensitively. After it runs, `git grep -i lopep` and `git ls-files | grep -i lopep`
must both return 0. That catches any casing the table missed.

Hand edits, not done by the script: README D2 lines, the README layout block (each `python/zepp/` line is one
character shorter, so the column alignment breaks), the NOTICE header, and the release author (D3).

## 4. Hazards and the control that closes each

- **H1 Silent knob drop (the most serious hazard).** Knobs are read by name at 139 sites. If `LOPEP_SWAP_OFF=1` is
  exported to a zepp binary, it is ignored and the run uses the default. The reverse also holds: `ZEPP_*` reaching a
  frozen lopep binary is ignored. *Control:* a prefix tripwire in the private launch layer (lib49.sh / server wrappers,
  right before torchrun/server start, inside the srun'd shell). In a zepp tree, any `^LOPEP_` aborts the launch; in a
  `lopep*` tree, any `^ZEPP_` aborts it. Keep the tripwire out of the repo, because the public tree must not contain
  the old name. Old ops scripts are **not** mass-converted: their env prefix belongs to the binary they ran. Positive
  control: `LOPEP_SWAP_OFF=1` plus a zepp launch must abort.
- **H2 Stale shadow package.** Checkout keeps ignored files (`python/lopep/lib/liblopep_cuda.so`, `_C*.so`,
  `__pycache__`), so `import lopep` keeps loading the OLD binary through PYTHONPATH. *Control:* the fresh clone (D4)
  has no leftovers. Gate: `python -c "import lopep"` must fail in the zepp environment.
- **H3 Stale CMake cache.** It holds `project(LOPEP)` and the install prefix `python/lopep`. *Control:* a fresh
  `build/`. Never reuse a cache across the rename.
- **H4 Lane provenance.** Two SGLang trees and two adapters exist side by side. *Control:* every server prints one
  provenance line (`sglang.__file__`, adapter `__file__`, `zepp.__file__`, the realpath of the loaded `.so`), and the
  driver asserts each path is under the intended tree. During P4, both lanes pin PYTHONPATH explicitly. The editable
  installs are flipped only in P5.
- **H5 SGLang patch.** The backend choices live in the patch, so the current clone only accepts `lopep`. *Control:*
  `git -C sglang worktree add ../sglang_zepp a4a3d82` gives a pristine v0.5.3. Apply the renamed shipped patch there
  and pass `git apply --check` first. Then apply the translated `47_layer_bracket_sglang.patch` (1 ref) and
  `layer_step_timing.py`. Gate: `git -C sglang_zepp diff` == translate(`git -C sglang diff`). The old clone stays
  untouched for frozen lanes.
- **H6 Calibrations.** The new `server.sh` default is `$ZEPP_CALIB_DIR/zepp_config.json`, and the 45 existing dirs
  hold `lopep_config.json`. The content has no name in it. *Control:* add a `zepp_config.json` symlink in each dir
  (non-destructive). runtime.py raises when `ZEPP_CONFIG` is unset, which is loud and acceptable.
- **H7 Log parsers return zero rows silently.** These match `[lopep timing rank` / `lopep.step`:
  `47_timing_table.py`, `47_gap_report2.py`, `49_collect.py`, `49_hostcalls.py`, and `logs/sglang/{lib49, chain_prof,
  probe49, rc8_serve, rc8_harness, timing_run}.sh`, `logs/p50/{diag_ledger, p10_r4}.sh`. *Control:* accept
  `(?:lopep|zepp)`. Test each parser on one old log (row count unchanged) and one new log (rows > 0).
- **H8 Audit blind spots.** `oss_audit.sh` drops NOTICE/LICENSE from the provenance grep, so a leftover "LoPEP" in
  NOTICE would pass. `lopep` is not banned, and the author string and default path are hardcoded. *Control:* add a
  separate old-name check (`lopep|moe_ep|moe-ep|libra ?x`) with **no** file exclusions, over contents, `git ls-files`
  paths and commit messages. Update the author (D3) and the default path. Negative control: the new audit run on the
  OLD `lopep_release` must FAIL.
- **H9 Author identity.** See D3.
- **H10 Old names in sglang-dev history** (`MOE_EP_*`, `LOPEP_*` in messages). This is harmless while publication is
  snapshot-only (release rule 4). Rule: `sglang-dev` is never pushed with its history. If it is ever published, it
  gets its own snapshot.
- **H11 Hash citations.** Forward commits only (D5). Frozen branches are never moved. `sglang-dev` advances past
  3d83066 and `p10-f` stays on it.
- **H12 Symbol names change** (kernel and device-function names, `LopepDbgSlot`). *Control:*
  `tests/test_kernel_registry.py` (registry == the `.so`'s kernel list) must pass on the new library.
- **H13 Proof of no behavior change.** (a) Text: apply the inverse map (`ZEPP→LOPEP`, `Zepp→LoPEP`, `zepp→lopep`,
  plus paths) to the mechanical commit, and the result must equal the parent tree **byte for byte**, with modes,
  after normalizing the parent's `Lopep→LoPEP`. That proves the commit is the map and nothing else. (b) Binary: per
  kernel, after demangling and the inverse map, `cuobjdump -res-usage` (regs/stack/smem) must be identical, and the
  SASS opcode sequences must be identical (operands are ignored because printf-string constant offsets move).
  (c) Runtime gates in P4.
- **H14 Future merges.** Both branches get commits from the same script, so the 27 shared files carry identical
  hunks.
- **H15 Formatting.** Every hit makes its line one character shorter, so no limit is exceeded. No formatter runs on
  the mechanical commit (it would break H13a). Re-align columns by hand in the prose commit.
- **H16 Quiet point.** No allocation may run on any lopep tree, and the SGLang figure work (handoff 55) must be
  committed first. Today `squeue --me` shows two pending jobs (`zd_hh_1004_verl_onestep_kl0`, `muxopd`). They are not
  this lane's jobs and must not be touched.
- **H17 Rules and memory drift.** CLAUDE.md's release section, `oss_audit.sh` and memory say LoPEP. Update them in P7
  in the same change set as the snapshot, so a later session does not audit for the wrong name.

## 5. Phases and gates

**P0 Freeze (login, ~10 min).** Record D1–D6. Confirm the tips (master 6658951, sglang-dev 3d83066, release 199eec0)
and stop on any drift. Back up to `logs/zepp_rename/`: `git -C lopep bundle create lopep_pre_rename.bundle --all`, a
tar of `lopep_release`, `git -C sglang diff` plus the untracked file, and `pip freeze` of andrewy-sglang.

**P1 Tooling (login).** Write `rename_map.tsv` and `zepp_rename.py` in `logs/zepp_rename/` (private: they contain the
old name). The script has three modes: `--dry-run` (every changed line plus per-spelling counts), `--apply`, and
`--verify <parent> <commit>` (H13a). Gate: dry-run counts == §1 for both branches.

**P2 New repo + commits (login).** `git clone -b sglang-dev lopep zepp`, `git branch master origin/master`,
`git remote remove origin`, init cutlass with `--reference ../lopep/3rdparty/cutlass`. On each branch: commit A (the
mechanical change, from the script) and commit B (prose: D2, layout block, NOTICE; on sglang-dev also the integration
README). Messages follow cf5b146 / c38fa02, e.g. "Rename the system to Zepp: package zepp, adapter zepp_sglang,
library libzepp_cuda, ZEPP_* environment, SGLang backend 'zepp'". No trailer (release rule). Gates: `--verify` exact
on both A commits; case-insensitive grep over contents and paths = 0 on both tips; B's diff reviewed by hand.

**P3 Build + static proof (compute node, `--jobs` ≤ 8 if on a login node).** Fresh builds of both zepp tips. Compare
the sglang-dev build against the existing `lopep_p10f` build (3d83066) and the master build against a build of
6658951. Rebuild the old side in the same session only if res-usage differs (module drift). Gates: H13b identical;
kernel registry test PASS; unit tests (`tests/*.py`) PASS.

**P4 Runtime gates (one 4n interactive allocation, 40 GB nodes, ~45 min ≈ 3 nh; one driver per job; scancel at the
end).** master: 1n `--check` 6/6 + demo 3/3 (the 9/26 gates), and `reproduce.sh` preflight messages name
`ZEPP_CONDA_ENV`. sglang-dev: `serving_check` 1n; 4n serving smoke + token agreement at the p10-f validation
thresholds; tripwire and parser positive controls (H1, H7); provenance lines asserted (H4). Optional perf
insurance: 1 MiB decode ABBA, zepp vs lopep_p10f on the same nodes. Expected parity within the R7/R8 pair spread.
Report it as measured and do not rerun.

**P5 Environment flip (login).** `pip uninstall lopep_sglang`, `pip install -e --no-deps` for
`zepp/integrations/sglang/zepp_sglang` and `sglang_zepp/python`. Frozen lanes keep working through the explicit
PYTHONPATH pins in a `frozen_lopep_env.sh` wrapper. Add calib symlinks (H6), the tripwire in lib49 (H1), the parser
updates (H7), and new ops templates that point at `zepp`. Gate: the H4 provenance check on both lanes from a fresh
shell.

**P6 Release snapshot (login).** Update `oss_audit.sh` (H8, D3, default path `zepp_release`). Build `zepp_release`
from `git archive master` + `git init` + one commit by the D3 author. Gates: `AUDIT PASS` on `zepp_release`; the new
audit FAILS on the old `lopep_release` (negative control); the `zepp_release` tree hash == zepp master's tree hash.
Then move `lopep_release` to `archive/` (D6).

**P7 Records.** Rename the CLAUDE.md release section, its paths, and the ban list (add `lopep`). Write this handoff's
execution log with the pre→post hash table (master 6658951 → A/B, sglang-dev 3d83066 → A'/B') and the gate results.
Update the memory entries (open-source-repo-plan, lopep-release-rules, no-claude-commit-trailer, sglang-integration-lane,
MEMORY.md lines). Commit the research tree. The zepp repo is not pushed; publishing stays the user's step.

## 6. Rollback

Everything is local. `lopep/`, `lopep_release` (in `archive/`), the `sglang/` clone and the bundle are never modified.
Rollback = delete `zepp/`, `zepp_release/`, `sglang_zepp/`; reinstall the old editable installs from the P0 `pip freeze`;
remove the tripwire and the calib symlinks.

## 7. Execution log (2026-10-04)

All artifacts are in `$PSCRATCH/workspace/andrewy/logs/zepp_rename/`.

- **P0 16:55** tips recorded (`P0_tips.txt`: master 6658951 tree ac6b34c, sglang-dev = p10-f 3d83066 tree 11956d2,
  lopep_release 199eec0 tree == master), `lopep_pre_rename.bundle` (verify okay, 65 heads), `P0_sglang_clone.diff` +
  `P0_layer_step_timing.py`, `P0_pip_freeze_sglang.txt`.
- **P1** `zepp_rename.py` (map, dry-run, apply, verify, leftovers). Dry-run: master 27 files / 81 occurrences
  (LOPEP 13, LoPEP 4, lopep 64) / 13 paths; 3d83066 83 files / 940 (LOPEP 553, LoPEP 9, Lopep 10, lopep 368) / 21 paths;
  0 collisions.
- **P2** private work clone `work/` (full history, never published). Mechanical commits zm 22063e5 (master) and zs
  3241b6a (3d83066): `verify` EXACT on both (every blob == F(old), modes and gitlink equal, inverse holds), leftovers 0/0.
  Prose commits: zm 3811f22 (README: "**Zepp**, an expert-parallel ...", the new paper title, layout column) and zs
  6d65d3a (same README edits; `env/perlmutter_sglang.sh` requires `ZEPP_CONDA_ENV`, caches under `ZEPP_CACHE_DIR`
  (default `$SCRATCH/zepp_cache`); integration README install line uses `$ZEPP_CONDA_ENV`). Release content checks on
  both: provenance 0 (old names included, NOTICE not exempt), numbers 0, data files 0.
- **zepp repo** `$PSCRATCH/workspace/andrewy/zepp`: `git init -b main`, commits made with `commit-tree` from the work
  trees, tree hashes equal (main d00ba5c == zm, sglang-dev 63fbd9e == zs). main 6021e44 "Initial release",
  sglang-dev 5c3fe70 "Serving path and SGLang v0.5.3 integration" (parent main). Author = committer = `Zepp authors
  <androniusyang@gmail.com>`. Local config uses the same identity; a commit-msg hook refuses claude/anthropic/
  co-authored-by. No remote. CUTLASS submodule df8a550 (depth 1 from GitHub, because the lopep copy is shallow).
  Directive 5 check: 2 commits, 0 hits in messages, identities, file contents of every commit (binary included) and
  paths; 0 old names; 0 unreachable objects; no other refs.
- **P6 audit** `research_tree/scripts/oss_audit.sh` (new): zepp **AUDIT PASS**. Negative controls: lopep_release FAIL
  (old names, history shape, author); the full-history work repo FAIL (incl. "claude/anthropic in a file of some
  commit" = the 27ed592 blobs).
- **P3 builds** (login, shared build slots, 8 jobs): zepp_rt (sglang-dev, CUDA 12.9) and zepp_rt_main (main, CUDA
  12.4), rc 0, ~3.5 min each. **Static proof H13b** (`sass_compare.py`, demangled names through the map): sglang-dev
  vs lopep_p10f (the 10-02 build that p10_fv validated, md5 59bb9af580b7) = 103/103 functions identical in
  REG/STACK/SHARED/LOCAL and SASS opcode sequence (150256 instructions); main vs a fresh build of 6658951 = 55/55
  identical (54216 instructions). Negative control: main vs sglang-dev library = DIFFERENT (comparator can fail).
- **SGLang lane**: `sglang_zepp` = worktree of the sglang clone at a4a3d82 + the translated clone diff (== the clone's
  diff under the map, every line except the blob-hash `index` lines) + layer_step_timing.py. The release's shipped
  patch applies cleanly (`git apply --check`) to pristine v0.5.3. A PYTHONPATH pin beats the editable install, so
  **the shared conda env is not touched** (replaces P5's editable flip; the frozen lanes keep working unchanged).
- **P5 ops** `ops/`: lib_zepp.sh (lane follows the tree; tripwire; provenance assertion), probe_zepp.sh, tok_zepp.sh,
  fv_zepp.sh. Login dry test: tripwire 5/5 correct (passed / inherited / reverse / two clean cases), provenance
  points into the lane's trees, `zepp.heap` == `lopep.heap` (6G).
- **Research tree (staged, home full)**: `research_tree/research_tree.patch` (CLAUDE.md section, oss_audit.sh, four
  parsers, this handoff), `git apply --check` clean. Parser regression: 47_timing_table.py edited output == original
  output on an old log, == the same rows on the renamed log; the ORIGINAL parser on the renamed log drops the ledger
  table (H7 demonstrated). 49_collect.py LEDGER regex matches both names with identical groups.
- **P4 GPU validation** (job 59341542, 4n hbm40g interactive, 17:19:52-17:56:36, cancelled at the end; the first salloc
  hit a Slurm "Connection timed out", the retry was granted at the estimated start). `fv.log`, `zT1_report.txt`,
  `probe/`, `srv/`:
  - tripwire 3/3 refused (LOPEP_ knob passed to zepp, inherited by zepp, ZEPP_ knob to the frozen lopep tree).
  - unit tests 10/10 PASS on zepp_rt (r5, kernel registry vs libzepp_cuda.so, route_fused 437 cases, deferred_verdict
    with the same kernel counts as 10-02, meta 104 bitwise, dispatch plan 90, plan_device 64 routings / 619 blocks /
    2476 tables bitwise, lane_device 149 swap steps, swap_decide 201 decisions, capacity_host 31 routings). The
    route_fused pair counts differ slightly from 10-02: relaxed tickets, "equal in distribution" per the test.
  - gates 5/5 PASS, 0 bad rows: flip / c1 / full / eager identical to 10-02 field for field (moves 880, recv_cap
    4627 / 23560). grow: 3 growths at the same steps (0, 9, 15) for the same fields; later demands differ by a
    few rows (relaxed tickets), so ceil(1.5 x demand) gives recv_cap 3741 vs 3743.
  - main 9/9 PASS: bench --check qwen3/k2 x overlap/direct/swap 0 bad rows; demo max|err| 0.0002415 x3 (= the 9/26
    release check).
  - serving token agreement (Qwen3-30B 4n decode config, 48 prompts x 64 greedy tokens; provenance asserted on every
    node; backend 'zepp' served): p10-f vs p10-f (same binary) 22/48 identical (46 %); p10-f vs zepp 26/48 (54 %)
    and 23/48 (48 %); stock vs p10-f 46 %, stock vs zepp 50 %. 0 tracebacks; zepp.heap == lopep.heap (6G); zepp
    free GPU memory 5.58 GB == p10-f.
  - harness parity, 256 tok/rank (layer-step ms, no-move n=6 / move n=24): p10-f 1.334/1.404, zepp 1.347/1.439, p10-f
    1.342/1.412, zepp 1.407/1.374. Move medians: zepp mean 1.407 vs p10-f 1.408 (zepp's own spread 4.7 %); n=6
    medians noisy. Instrumented harness, not a latency quote.
  - FOUND: the first zepp parity run failed loudly (FileNotFoundError `<calib>/zepp_config.json`), because
    serving_check `--calib <dir>` reads the renamed file. This is H6 for the harness path (the server path was covered
    by ZEPP_CONFIG). Fixed with the plan's H6 control: a `zepp_config.json -> lopep_config.json` symlink in all 28 calib
    dirs under logs/sglang (non-destructive). `ops/finish_parity.sh` stopped the driver before its scancel, ran the
    missing zepp sample, then cancelled, keeping the parity order balanced.

## 8. Open items (2026-10-04)

1. Home is over quota (40.33 / 40 GiB), so `research_tree/research_tree.patch` (CLAUDE.md release section, oss_audit.sh,
   four parsers, this handoff) and the memory updates wait for space. Apply with `git apply` from the flux root.
2. README cites "Figure 9 and Section 6" of the paper. Confirm the numbering in the renamed paper.
3. Not done by design: no push (directive: stays on PSCRATCH); lopep_release untouched (D6); old ops scripts not
   converted (bound to the lopep binaries; the zepp lane uses `ops/`).

**10-04 ~22:00 user ruling:** the author and committer are `Andrew Yang <androniusyang@gmail.com>`. The zepp commits were
regenerated with the same trees (main 6021e44 -> 45ba36a, sglang-dev 5c3fe70 -> 84c137e), the old objects pruned, and
the audit's expected author and the CLAUDE.md rule updated; audit PASS. sglang-dev is being pruned (handoff 57).

**10-04 23:26:** after the sglang-dev prune (handoff 57) zepp was regenerated: main 6da0bcf (constants comment without
a measured number), sglang-dev 83a7a59 (pruned tree); audit PASS.
