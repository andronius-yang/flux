# sweeps/ — persistent, reproducible perf sweeps

Every sweep produces an immutable **run capsule** under `sweeps/results/runs/`
(committed to git: manifest, resolved spec, per-iteration metrics.csv,
cells.csv) plus platform-local raw logs at the data root. The data contract —
budget semantics, column dictionaries, mode rules — lives in
[SCHEMA.md](SCHEMA.md); read that before interpreting any numbers.

## Quickstart (AWS)

```bash
source ./env_aws.sh
salloc --partition=a100 --nodes=2 --exclusive --no-shell
python sweeps/sweep.py run --platform aws \
    --variants hier,hier_compress,hier_compress_union \
    --families remotefrac --budgets-mib 2,8,64 \
    --topk 8 --G 128 --modes e2e,phases --dry-run   # inspect cells first
# drop --dry-run to execute; then commit the printed capsule path
```

On Perlmutter: `source ./module.sh`, `salloc --qos interactive -C gpu
--account m4243_g -N 2 --no-shell`, `--platform perlmutter`. Everything else
is identical — topology (4 vs 8 ranks/node), fabric, and data roots come from
`platforms/*.yaml`, and matrix identity embeds `WxL` so results can't be
cross-contaminated.

## Pieces

| file | role |
|---|---|
| `sweep.py` | runner: cells → srun → merge → capsule; `run`, `rerun`, `--dry-run` |
| `variants.py` | canonical variant names → `--comm_pattern` + env knobs + capability requirements (incl. the `fast` baseline, which swaps in `launch_fast.sh` + its own test via `driver="fast"`) |
| `gen_matrix.py` | deterministic traffic-matrix generator (FNV-1a-seeded families) |
| `platforms/*.yaml` | per-platform topology, fabric, data/matrices roots |
| `specs/example.yaml` | annotated spec (flags override; resolved spec always saved in the capsule) |
| `results/runs/` | committed capsules |

A one-off ad-hoc ask is just `run` with flags — the capsule still records the
equivalent spec, so nothing is one-off in hindsight. Aggregation/plotting is
deliberately out of scope here (metrics.csv is raw per-iteration data; pivot
in pandas/polars/Excel downstream).
