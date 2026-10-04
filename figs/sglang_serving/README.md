# SGLang serving figure (DRAFT)

One USENIX column (3.33 in): stock SGLang vs Ours, per-rank data on the x axis (labeled in MB like the other paper
figures; 1 / 2 / 4 = 256 / 512 / 1024 tokens or running requests per GPU, 2048 hidden x 2 B), two panels:

- **Prefill**: TTFT (ms) from `sglang.bench_serving` (960 LiveCodeBench prompts, output 4 tokens, 256 concurrent, no
  rate limit). The metric for this panel is still open (TTFT vs input throughput; see below).
- **Decode**: TPOT (ms) = the decode step median over 192 scheduler intervals at a fixed running batch
  (`docs/handoff/49_decode_wave.py`), i.e. the per-token latency of every request in the lockstep batch.

Speedup (red, under the Ours bar) = stock / ours.

Files: `data_draft.csv` (values, per-run values, provenance), `make_figure.py` (all style in CONFIG; fonts and the Ours
blue follow `figs/main_perf_v5`), `sglang_serving_draft.pdf/.png`. Render: `andrewy-comet` conda python (has matplotlib).

Draft data (4 nodes, Qwen3-30B-A3B, A100 40 GB; round R8, job 59242113, lopep p10-s7 = p10-f code): decode at 1 / 2 /
4 MB measured; prefill measured at 1 MB only (2 / 4 MB pending, the campaign fills them). All data stays in this
directory; nothing is written to the SGLang or lopep repositories.

Open: TTFT vs throughput for prefill. In this saturated closed-loop benchmark the mean TTFT moved 1115 -> 1478 ms
between two identical stock runs on one allocation (input throughput moved 2 %), so TTFT needs a different experiment
(fixed request rate below saturation, or many more requests) to be quotable.
