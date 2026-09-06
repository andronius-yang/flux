# figs/system_overview

Cross-column system overview figure for the implementation / system design
section. `00_brief.md` is the discussion record (purpose, code-traced
ground truth of the main-perf arm in §10, draft decisions in §11).

- `moe.drawio` / `moe.pdf` — user's single-column base figure (2 nodes x 2 GPUs).
- `make_overview_drawio.py` — generator; emits `moe_overview.drawio` and a
  matplotlib preview (`moe_overview_preview.png/.pdf`). All geometry and
  routing are knobs at the top of the script.
- `moe_overview.drawio` — the extended figure*, hand-editable in draw.io.
