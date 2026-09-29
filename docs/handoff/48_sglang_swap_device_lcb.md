# Handoff 48: swap decision on the GPU, the 8n swap-step combine growth, LiveCodeBench serving traffic

Opened 2026-09-28 (session 3, after handoff 47). User rulings (same day):
1. Serving traffic = LiveCodeBench execution-v2 only (479 problems = the problem set of the paper's Qwen3-235B routing
   traces, "Patterns behind Chaos", arXiv 2510.05497), official code-execution prompt + chat template (the traces' own
   prompt wording is unpublished: assumption). DISJOINT split by contest date: history = 239 older problems
   (2023-05-07 .. 2023-08-13), the only traffic the placement is calibrated on (stock server, same model, same benchmark
   shape, per node count); eval = 240 newer problems (2023-08-13 .. 2023-11-25), benchmarked, repeated with
   --disable-radix-cache on every arm. Random-token traffic dropped completely (removed from logs/sglang/bench1.sh).
   Prompt tokens (Qwen3 tokenizer, chat template): median ~330, history total 78k, eval total 81k.
2. Swap trigger unchanged (paper text): node band test on reference loads with C, heaviest<->lightest pairing, only
   max-reducing exchanges, cap 8 per rank, up to 32 rounds.
3. Swap decision moved to the GPU with identical semantics, no host wait, serving path only.
4. Root-cause the 8n b16 combine growth on swap steps (47: 15.3 vs 6.3 ms) before any swap-on claim at 8n.

## Branches
- B1 root cause (8n debug x2): (A) harness serving_check, Qwen3-30B shape, 2048 tokens/rank on every rank, popularity
  shifting every step (SYNTHETIC routing, used only to force swaps for the lane mechanism A/B, never a performance
  row): swap off / staged / overlap (benchmark lane) / inline, then nsys of the staged lane; (B) serving: 8n LiveCodeBench
  history calibration, then ours under nsys on eval traffic, per-phase attribution split by swap-class label.
  Hypothesis: the staged lane's W2 push is issued on the sender's forward stream right before the sender's combine GEMM,
  so receivers' gated combine tiles wait on the sender's dispatch, spread by the combine barriers. Candidate fix if
  confirmed: issue W2 with W1 before the dispatch GEMM (paper §4.3 "initiates weight transfers early").
- B2 device decision: single-block CUDA kernel `swap_decide` (src/planner/swap_decide.cu) = decide_swaps + pad-row
  correction + l2p rebuild + net_moves, tables rewritten in place, result block copied to pinned memory and read after the
  step's existing planning sync; adapter passes its once-per-pass exact counts (removes the per-layer count gather +
  host read, serving.py ~394, where both 47 hangs were caught). Knob LOPEP_SWAP_DECIDE=host|device|compare.
- B3 workload: 48_lcb_workload.py (prompt files under caches/hf/lcb), bench1.sh works lcbp/lcbd (eval) and lcbhp/lcbhd
  (history), chain_calib_lcb.sh (record on stock + solve, factor 2.5).

## Log (newest last)
- 09-28 20:10 prep: harness options (qwen3-30b/-235b shapes, --token-mode full, --flip-every, --ref), NVTX swap-class
  labels (lopep Timing), per-label summary in 47_gap_report2.py; B1 jobs requested (debug 59056922 harness, 59056921 serving).
- 09-28 20:25 B2 BUILT + VERIFIED ON 1n (lopep sglang-dev working tree, uncommitted; lopep_t28 rebuilt 19:19,
  build_t28_swapdecide2.log): src/planner/swap_decide.cu (single block; top-8 selection without per-thread arrays,
  the router's 450 MB local-memory lesson), binding C.swap_decide / swap_decide_out_ints, serving.py modes
  host|device|compare (device: tables rewritten in place before routing, result block read after the planning
  sync, lane armed after meta+check), adapter passes the pass's exact counts (no per-layer count gather).
  tests/test_swap_decide.py (1 GPU): PASS, 201 swap decisions (4145 moves, up to 4 rounds) + 219 in-band steps
  bitwise identical to the host (pad-corrected loads, p2l, l2p, pull lists) over R = 4..64, G = 32/128/384.
  serving_check 1n (job 59057071): compare mode small shape (28 moves, 1 growth), starved rank (8 moves), Qwen3-30B
  shape full 512-token batches (12 moves): all 0 bad rows, no compare assertion; device mode small: PASS, same moves.
  Timing pair 1n, 30B shape, 512 tok/rank (only 1-2 swap steps of 120 on one node): per-rank layer-step medians
  host 2.31-2.39 vs device 2.32-2.36 ms (no difference); rank-0 ledger: decision phases host 0.17 vs device 0.07 +
  arm 0.05 ms, but pad+loads 0.21 -> 0.42 (attribution shift of rank skew into the loads gather once the host no
  longer blocks mid-step; the step medians are unchanged). Inconclusive by design: the gain needs swap-heavy steps
  (4n/8n, LiveCodeBench serving).
- 09-28 20:10 LiveCodeBench calibrations written (history half, stock server, factor 2.5, swap-on config):
  calib_30b8n_lcbp_overlap_s1 (8n, prefill shape), calib_30b4n_lcbp_overlap_s1 (4n, prefill shape),
  calib_30b4n_lcbd_overlap_s1 (4n, decode shape, OSL 64).
- 09-28 20:10 B1 serving run at 8n (job 59056921, 30B, SMAX 2048, host decision, LiveCodeBench eval x16, c=512): the
  nsys report was LOST (job cancelled 60 s after the capture closed, before nsys finished writing; fix = poll for the
  .nsys-rep before scancel). Timing ledger (rank 0) survived. FINDINGS: (1) with history-matched placement the band
  trigger STILL fires on ~95 % of regime steps at 8n (S<=2048: 47-48 of 48 steps per window; S<=1024: 44-47 of 48),
  i.e. at 8n (6 slots per GPU) the high swap rate is NOT mainly the traffic mismatch; (2) the combine growth on swap
  steps is GONE with LiveCodeBench traffic: swap-step combine 4.1-4.8 ms vs 4.1 on the rare no-swap steps (47 random-token
  run: 15.3 vs 6.3), so the +9 ms was tied to the random-token / gsm8k-placement mismatch at 8n (heavy imbalance, many
  moves), not to every swap; (3) swap-step layer-step at 2048: 12.0-13.7 ms, of which the host decision is 1.55-1.91 ms
  (dispatch 3.9-4.5, combine 4.1-4.8) -> the device decision addresses ~13 % of the step directly.
- 09-28 20:10 4n session (job 59057726): the compare arm served no traffic (lcb_exec_eval_x4.json had not been built;
  built now); driver replaced by lcb4n_b.sh: compare rerun after the prefill A/B chain, then the decode chain.
- 09-28 20:08 NEW CORRECTNESS ITEM: serving_check at 8n (job 59056922), Qwen3-30B shape, all 32 ranks at 2048 tokens,
  synthetic skew (70 % of demand on 25 % of experts, popularity flipping every step), SWAP OFF, host path: CUDA illegal
  memory access on rank 28, surfaced asynchronously after step 0 (next H2D). Not the 09-26 workspace race (d5ed3cc is
  on sglang-dev and in lopep_t28). The same ops ran clean at 8n / 2048 in serving (LiveCodeBench and random traffic),
  so the trigger is the skewed full-batch pattern. To localize: smaller reproduction (2n/4n, same skew) with
  CUDA_LAUNCH_BLOCKING=1 / compute-sanitizer, then cuda-gdb attach (the 09-26 method).
- 09-28 20:40 4n LIVECODEBENCH PREFILL A/B (job 59057726, 30B, hbm80g, calib_30b4n_lcbp_overlap_s1 = history half,
  eval half x8 (1920 prompts, 648k prompt tokens), 16 requests per rank, KV 400k, radix cache off, same nodes, one
  binary; host = LOPEP_SWAP_DECIDE=host, device = device):
  | tokens/rank | input tok/s host / device / stock | per layer-step (steady windows) host / device / stock |
  | 128  | 9323 / 9841 / 16730 | 3.67 / 3.51 / 2.02 |
  | 512  | 24023 / 25222 / 34151 | ~3.75 (3.67-3.97) / ~3.59 (3.54-3.65) / ~3.2 (3.01-3.45) |
  | 2048 | 26791 / 36717 / 38094 | 6.36-6.90 / 6.19-6.53 / 10.3-12.5 |
  Swap-step share with history-matched placement at 4n: 3-4 % at 128, ~6 % at 2048 (random-token/gsm8k: 37-63 %):
  at 4n the high swap rate WAS the traffic mismatch. Device decision: decision phase 0.21 -> 0.04-0.07 ms on no-swap
  steps; on swap steps the host now spends 0.72-0.76 ms arming the lane AFTER the planning sync (GPU idle; before, that
  work overlapped the planning) -> next item. Throughput rows at 2048 are ramp-limited (17-24 s runs, ~20 forward
  passes): quote the per-layer column; longer runs needed for throughput.
- 09-28 20:35 8n ILLEGAL ACCESS narrowed (job 59056922): static popularity at 2048 full batches -> PASS on all 32 ranks
  (no growth). Both faulting runs had a capacity growth first under shifting demand: swap0 grew at step 3 (recv_cap
  25786 -> 45467, dispatch_recv 39257 -> 71382) and faulted right after; staged grew at step 3 and faulted after
  step 20 with no second growth. CUDA_LAUNCH_BLOCKING=1 is useless here (the fused ops deadlock when launches are
  serialized: GEMM tiles spin on signals from other streams). HYPOTHESIS: the in-place resize (resize_capacities, verified
  only at 1n with small shapes) leaves a capacity-sized buffer at its old size; a later high-demand step overruns it.
  Consistent with today's serving: r5 235B 4n grew then hung; r3ours 8n grew once and survived; factor-2.5 calibrations
  never grew and never faulted. REPRODUCTION to run: serving_check --caps-scale 0.5 (forced growth), Qwen3-30B shape,
  full 2048, swap off, static popularity, 2n and 4n; then CUDA_DEVICE_WAITS_ON_EXCEPTION=1 + cuda-gdb attach (09-26
  method) and an audit of every allocation that depends on Capacities vs what resize_capacities reallocates.
  Envelope (normal launches, 1 layer, 4 steps, swap off): static 2048 PASS (no growth); flipping 1024 PASS (1 growth);
  flipping 512 PASS (1 growth). The fault has only appeared at 2048 with flipping demand over 3 layers x 60 steps. Next:
  forced-growth reproduction at 2048 (--caps-scale 0.5), 2n/4n, 3 layers, 60 steps, swap off.
- 09-28 20:55 4n COMPARE MODE IN SERVING (lcb4n_cmp2, SMAX 512, eval x4): host and device decisions asserted identical on
  every layer-step: 1410 swap layer-steps + ~1950 no-swap layer-steps, no mismatch (the 4 tracebacks in the log are the
  chain's teardown killing the schedulers after the bench). B2 correctness closed in real serving at 4n.
- 09-28 21:00 4n DECODE (lcb4n_d, calib_30b4n_lcbd = history half decode shape, eval x16 OSL 64, c=2048): achieved ~1180
  concurrent but decode batches stayed <= 16 tokens/rank (requests queue behind prefill: mean TTFT 22-60 s), so this row
  is small-batch decode, NOT the 128/rank target. output tok/s / mean ITL: host 1009 / 237, device 1096 / 218, stock
  graphs off 2407 / 83, stock graphs on 2901 / 67. Per layer-step DECODE MAX n_pad<=16: host 4.51, device 4.13, stock
  0.95; swap share at <=16 tok/rank 69-70 % (routing noise at tiny batches, as on 09-27). Prefill steps inside the same
  runs (EXTEND 2048): ours 6.97-7.08 vs stock 11.46. Next decode design: longer outputs (OSL 256+) and concurrency sized
  so running requests reach 128 per rank.
- Lane-arming exposure (device decision): on swap steps the host arms the lane after the planning sync for 0.72-0.76 ms
  (swap_arm) while the GPU idles (lane.prepare builds the move lists and a pageable torch.tensor(keep) H2D, then index_fill);
  next B2 item: arm from the device result (keep mask + gate fills in the kernel or one small kernel), no host loop.
